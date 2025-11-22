"""
HouseGym RL Environment - Enhanced Version
==========================================

Core Design Principles:
1. Pure random candidate selection (no artificial bias)
2. Batch arrival system (simulates real assessment timeline)
3. Capacity ramp system (simulates contractor mobilization)
4. Focus on robustness and generalization, not peak performance

Architecture:
- BaseEnv: Base class with all common constraints
- RLEnv: For RL training (continuous action space)
- BaselineEnv: For baseline policies (LJF/SJF/Random)
- OracleEnv: No candidate limit (quantifies information cost)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
from typing import Optional, Dict, List, Any
import warnings
from pathlib import Path
import pickle

# Import configuration
import config
from config import (
    MAX_STEPS,
    COMBINED_ARRIVAL_CAPACITY_CONFIG,
    WORK_PARAMS,
    CMAX_BY_LEVEL,
    DATA_DIR
)

# ============================================================================
# Auxiliary Classes
# ============================================================================

class TrueBatchArrival:
    """
    Simulates realistic damage assessment timeline.
    Houses are revealed in batches, not all at once.

    Based on Lombok reality:
    - Batch 1 (Day 0): Emergency assessment identifies major damage
    - Batch 2 (Day 21): Detailed assessment identifies moderate damage
    - Batch 3 (Day 36): Complete assessment identifies minor damage
    """

    def __init__(self, tasks_df: pd.DataFrame, config: dict, seed: int):
        """
        Initialize batch arrival system.

        Args:
            tasks_df: DataFrame with all houses
            config: Batch arrival configuration
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.config = config
        self.revealed_ids = set()
        self.current_day = -1

        # Shuffle all houses and split them purely by ratios (no severity bias)
        days = config["days"]
        ratios = config["ratios"]
        if len(days) != len(ratios):
            raise ValueError("Batch arrival config must have matching days and ratios")

        all_ids = tasks_df.index.tolist()
        self.rng.shuffle(all_ids)

        total = len(all_ids)
        ratios_arr = np.asarray(ratios, dtype=float)
        float_counts = ratios_arr * total
        batch_sizes = np.floor(float_counts).astype(int)
        remainder = total - int(batch_sizes.sum())
        if remainder > 0:
            fractional = float_counts - batch_sizes
            order = np.argsort(-fractional)
            for idx in order[:remainder]:
                batch_sizes[idx] += 1

        schedule = {}
        cursor = 0
        for day, size in zip(days, batch_sizes.tolist()):
            batch = all_ids[cursor:cursor + size]
            cursor += size
            schedule[day] = batch

        # Safety: assign any leftover houses to the last batch
        if cursor < total:
            schedule[days[-1]].extend(all_ids[cursor:])

        self.schedule = schedule

    def step_to_day(self, day: int) -> List[int]:
        """
        Advance to given day and return newly revealed houses.
        
        Debug: Track revealed_ids updates
        """
        new_arrivals = []

        # Check all scheduled days up to current day
        for scheduled_day, houses in self.schedule.items():
            if scheduled_day <= day and scheduled_day > self.current_day:
                # Debug: print what's happening
                # if len(houses) > 0:
                #     print(f"    [DEBUG] Day {day}: Batch at day {scheduled_day} releasing {len(houses)} houses")
                #     print(f"            Before: revealed_ids has {len(self.revealed_ids)} houses")

                new_arrivals.extend(houses)
                self.revealed_ids.update(houses)

                # if len(houses) > 0:
                #     print(f"            After:  revealed_ids has {len(self.revealed_ids)} houses")

        self.current_day = day

        # if len(new_arrivals) > 0:
        #     print(f"    [DEBUG] Day {day}: Total new arrivals = {len(new_arrivals)}, revealed_ids total = {len(self.revealed_ids)}")
        
        return new_arrivals

    def is_complete(self) -> bool:
        """Check if all houses have been revealed."""
        return self.current_day >= max(self.schedule.keys())

    def get_revealed_count(self) -> int:
        """Get number of houses revealed so far."""
        return len(self.revealed_ids)


class StaticArrival:
    """
    Static arrival: All houses known from day 0.
    Used as control/baseline for experiments.
    """

    def __init__(self, tasks_df: pd.DataFrame):
        self.revealed_ids = set(tasks_df.index.tolist())
        self.current_day = -1  # Must be -1 so first step_to_day(0) triggers

    def step_to_day(self, day: int) -> List[int]:
        if day == 0 and self.current_day < 0:
            self.current_day = day
            return list(self.revealed_ids)
        self.current_day = day
        return []

    def is_complete(self) -> bool:
        return True

    def get_revealed_count(self) -> int:
        return len(self.revealed_ids)


class CapacityRamp:
    """
    Models gradual contractor mobilization.
    Capacity grows from 0 to max over time.

    Based on Lombok reality:
    - Days 0-36: Planning phase (K=0)
    - Days 36-216: Linear ramp-up
    - Day 216+: Full capacity
    """

    def __init__(self, max_capacity: int, config: dict):
        """
        Initialize capacity ramp.

        Args:
            max_capacity: Maximum number of contractors
            config: Capacity ramp configuration
        """
        self.max_K = max_capacity
        self.warmup = config["warmup_days"]
        self.rise = config["rise_days"]
        self.full_capacity_day = self.warmup + self.rise

    def get_capacity(self, day: int) -> int:
        """
        Get effective capacity for given day.

        Args:
            day: Current simulation day

        Returns:
            Number of available contractors
        """
        if day < self.warmup:
            # Planning phase, no construction
            return 0
        elif day < self.full_capacity_day:
            # Linear ramp-up
            progress = (day - self.warmup) / self.rise
            return int(self.max_K * progress)
        else:
            # Full capacity reached
            return self.max_K


class FixedCapacity:
    """
    Fixed capacity: Full capacity from day 0.
    Used as control/baseline for experiments.
    """

    def __init__(self, max_capacity: int):
        self.max_K = max_capacity

    def get_capacity(self, day: int) -> int:
        return self.max_K


class WaitingQueue:
    """
    Manages the queue of houses waiting for reconstruction.
    Houses enter when assessed, leave when completed.
    """

    def __init__(self):
        self.waiting_ids = []

    def add(self, house_ids: List[int]):
        """Add newly assessed houses to queue."""
        self.waiting_ids.extend(house_ids)

    def remove(self, house_ids: List[int]):
        """Remove completed houses from queue."""
        self.waiting_ids = [h for h in self.waiting_ids if h not in house_ids]

    def get_all(self) -> List[int]:
        """Get all houses currently in queue."""
        return self.waiting_ids.copy()

    def size(self) -> int:
        """Get current queue size."""
        return len(self.waiting_ids)


# ============================================================================
# Base Environment Class
# ============================================================================

class BaseEnv(gym.Env):
    """
    Base environment implementing all common constraints.

    Key features:
    - Batch arrival (information gradually revealed)
    - Capacity ramp (resources gradually available, currently disabled)
    - Pure random candidate selection (no artificial bias)
    - Fair constraints for all strategies (RL/baseline)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        region_key: str,
        num_contractors: Optional[int] = None,
        M_ratio: float = 0.10,
        M_min: int = 512,
        M_max: int = 2048,
        use_batch_arrival: bool = True,
        use_capacity_ramp: bool = False,
        stochastic_duration: bool = True,
        observation_noise: float = 0.02,
        capacity_noise: float = 0.02,
        seed: Optional[int] = None,
        max_steps: int = MAX_STEPS,
    ):
        """
        Initialize base environment.

        Args:
            region_key: Region name from REGION_CONFIG
            num_contractors: Number of contractors (if None, use region default)
            M_ratio: Proportion of queue to sample as candidates (default: 0.10 = 10%)
            M_min: Minimum number of candidates (default: 512)
            M_max: Maximum number of candidates (default: 2048)
            use_batch_arrival: Whether to use batch arrival system
            use_capacity_ramp: Whether to use capacity ramp system
            stochastic_duration: Whether work progress is stochastic (default: True)
            observation_noise: Std dev of observation noise as fraction of true value (default: 0.02)
            capacity_noise: Range of capacity reduction (default: 0.02 = 98%-100% of base)
            seed: Random seed
            max_steps: Maximum simulation days
        """
        super().__init__()

        # Set random generator
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        # Load region configuration
        if region_key not in config.REGION_CONFIG:
            raise ValueError(f"Unknown region: {region_key}")

        self.region_key = region_key
        self.region_config = config.REGION_CONFIG[region_key]

        # Set number of contractors
        if num_contractors is None:
            num_contractors = self.region_config["num_contractors"]
        self.num_contractors = num_contractors

        # Set adaptive M parameters
        self.M_ratio = M_ratio
        self.M_min = M_min
        self.M_max = M_max
        self.max_steps = max_steps

        # Set stochasticity parameters
        self.use_batch_arrival = use_batch_arrival
        self.stochastic_duration = stochastic_duration
        self.observation_noise = observation_noise
        self.obs_noise = observation_noise
        self.capacity_noise = capacity_noise

        # Load scenario data
        self.tasks_df = self._load_scenario(region_key)

        # Initialize work arrays
        self._arr_total = self.tasks_df['man_days_total'].values.astype(np.float32)
        self._arr_rem = self.tasks_df['man_days_remaining'].values.copy().astype(np.float32)
        self._arr_dmg = self.tasks_df['damage_level'].values.astype(np.int32)
        self._arr_cmax = self.tasks_df['cmax_per_day'].values.astype(np.float32)

        # Fix Critical Issue #3: Compute fixed total work for consistent reward scaling
        self._total_work_all = float(sum(self._arr_total))

        # Initialize systems
        if use_batch_arrival:
            self.arrival_system = TrueBatchArrival(
                self.tasks_df,
                COMBINED_ARRIVAL_CAPACITY_CONFIG["batch_arrival"],
                seed if seed is not None else 42
            )
        else:
            self.arrival_system = StaticArrival(self.tasks_df)

        self.use_capacity_ramp = bool(use_capacity_ramp and config.CAPACITY_RAMP_ENABLED)
        if use_capacity_ramp and not self.use_capacity_ramp:
            warnings.warn(
                "Capacity ramp requested but globally disabled; using fixed capacity.",
                UserWarning,
            )

        if self.use_capacity_ramp:
            self.capacity_system = CapacityRamp(
                num_contractors,
                COMBINED_ARRIVAL_CAPACITY_CONFIG["capacity_ramp"]
            )
        else:
            self.capacity_system = FixedCapacity(num_contractors)

        # Initialize queue
        self.waiting_queue = WaitingQueue()

        # State variables
        self.day = 0
        self.last_completion = 0.0
        self.last_completion_count = 0  # For tracking progress reward

        # Long-term reward tracking variables
        self.waiting_time = np.zeros(len(self.tasks_df), dtype=np.int32)
        self.last_queue_size = 0
        self.last_workers_used = 0
        self.last_workers_available = 0

        # Daily decision state
        self.pending_candidates: List[int] = []
        self.pending_allocation: Dict[int, int] = {}
        self.current_day_capacity: int = 0
        self.current_day_initial_candidates: int = 0

        # Define spaces
        self._define_spaces()

    def _load_scenario(self, region_key: str) -> pd.DataFrame:
        """
        Load or generate scenario data for a region.

        Args:
            region_key: Region name

        Returns:
            DataFrame with house reconstruction tasks
        """
        # Try to load from pickle if exists
        pkl_path = DATA_DIR / f"{region_key.lower().replace(' ', '_')}_tasks.pkl"
        if pkl_path.exists():
            return pd.read_pickle(pkl_path)

        # Otherwise generate synthetic data
        config = self.region_config
        damage_dist = config["damage_dist"]

        tasks = []
        house_id = 0

        for damage_level, count in enumerate(damage_dist):
            for _ in range(count):
                # Sample work duration
                params = WORK_PARAMS[damage_level]
                median = params["median"]
                sigma = params["sigma"]

                # Log-normal distribution
                mean_log = np.log(median)
                work_days = self.rng.lognormal(mean_log, sigma)
                work_days = max(1, int(work_days))

                # Get crew cap
                cmax = CMAX_BY_LEVEL[damage_level]

                tasks.append({
                    'house_id': house_id,
                    'damage_level': damage_level,
                    'man_days_total': work_days,
                    'man_days_remaining': work_days,
                    'cmax_per_day': cmax,
                })
                house_id += 1

        return pd.DataFrame(tasks).set_index('house_id')

    def _get_M(self, queue_size: int) -> int:
        """
        Calculate adaptive M based on current queue size.

        M = ratio × queue_size, clamped to [M_min, M_max].

        Args:
            queue_size: Current number of houses in queue

        Returns:
            Number of candidates to sample
        """
        M = int(queue_size * self.M_ratio)
        return max(self.M_min, min(self.M_max, M))

    def _define_spaces(self):
        """
        Define observation and action spaces.

        Uses M_max to ensure fixed space dimensions across episodes.
        """
        # Observation: 6 global features + M_max*4 candidate features
        obs_dim = 6 + self.M_max * 4
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action: M_max candidate scores + 1 no-op signal
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.M_max + 1,),
            dtype=np.float32
        )

    def _get_capacity_for_day(self, day: int) -> int:
        """Sample capacity for a specific day (with noise if enabled)."""
        base_capacity = self.capacity_system.get_capacity(day)
        if self.capacity_noise > 0 and base_capacity > 0:
            noise_factor = self.rng.uniform(1.0 - self.capacity_noise, 1.0)
            base_capacity = int(base_capacity * noise_factor)
        return max(0, base_capacity)

    def _effective_capacity(self) -> int:
        """
        Return the capacity sampled for the current day (for logging).
        """
        return self.current_day_capacity

    def _begin_day_if_needed(self) -> None:
        """Initialize per-day state if we are starting a new day."""
        # Only initialize if we don't already have candidates for today
        if self.pending_candidates:
            return

        # Reveal new houses for this day
        new_arrivals = self.arrival_system.step_to_day(self.day)
        if new_arrivals:
            self.waiting_queue.add(new_arrivals)

        # Sample capacity for the day
        capacity = self._get_capacity_for_day(self.day)
        self.current_day_capacity = capacity
        self.pending_allocation = {}
        self.pending_candidates = []
        self.current_day_initial_candidates = 0

        # No tasks or no capacity → nothing to do until advance
        if capacity <= 0 or self.waiting_queue.size() == 0:
            return

        # Sample today's candidate set once
        queue_ids = self.waiting_queue.get_all()
        candidates = self._select_candidates(queue_ids)
        self.pending_candidates = candidates.copy()
        self.current_day_initial_candidates = len(self.pending_candidates)

    def _advance_day(self):
        """Apply accumulated allocation and move to the next day."""
        allocation = dict(self.pending_allocation)
        obs, reward, terminated, truncated, info = self._execute_allocation(allocation)

        # Reset day state (for next step)
        self.pending_candidates = []
        self.pending_allocation = {}
        self.current_day_capacity = 0
        self.current_day_initial_candidates = 0

        if not terminated and not truncated:
            # Prepare next day's candidates and observation
            self._begin_day_if_needed()
            obs = self._get_obs()

        return obs, reward, terminated, truncated, info

    @property
    def decision_day(self) -> int:
        """
        Get the day value used for current/last decisions.

        During step N, decisions are made using day N-1.
        This property returns the day that was actually used for decision-making.
        """
        # During a step, self.day hasn't been incremented yet, so it IS the decision day
        # After a step completes, self.day has been incremented, so decision_day = day - 1
        # This is mainly useful after step() completes
        return max(0, self.day - 1) if self.day > 0 else 0

    @property
    def completed_days(self) -> int:
        """
        Get the number of fully completed days.

        After step N, this returns N (days 0 through N-1 have been completed).
        """
        return self.day

    def _apply_work(self, allocation: Dict[int, int]):
        """
        Apply work allocation with stochastic progress.

        If stochastic_duration is enabled, work progress has random variation
        around the expected value (σ = 20% of expected progress).

        Args:
            allocation: Dictionary mapping house_id to number of workers
        """
        for house_id, workers in allocation.items():
            if self.stochastic_duration:
                # Expected progress = number of workers allocated
                expected_progress = workers

                # Actual progress with noise (σ = 20% of expected)
                # This models variability in worker efficiency, weather, materials, etc.
                actual_progress = self.rng.normal(
                    expected_progress,
                    expected_progress * 0.20
                )
                actual_progress = max(0, actual_progress)
            else:
                # Deterministic progress (legacy mode)
                actual_progress = workers

            # Update remaining work
            self._arr_rem[house_id] -= actual_progress
            self._arr_rem[house_id] = max(0, self._arr_rem[house_id])

    def _queue_penalty_sigmoid(
        self,
        waiting_time: int,
        midpoint: int = 50,
        steepness: float = 0.1,
        max_penalty: float = 0.1
    ) -> float:
        """
        Sigmoid penalty function for queue waiting time.

        Penalty curve:
        - waiting_time < midpoint: small penalty (gradual increase)
        - waiting_time = midpoint: penalty = max_penalty / 2
        - waiting_time >> midpoint: penalty → max_penalty (saturates)

        Args:
            waiting_time: Days the house has been waiting
            midpoint: Inflection point (default: 50 days)
            steepness: Controls how fast penalty grows (default: 0.1)
            max_penalty: Maximum penalty value (default: 0.1)

        Returns:
            Negative penalty value (always ≤ 0)
        """
        penalty = max_penalty / (1 + np.exp(-steepness * (waiting_time - midpoint)))
        return -penalty

    def _calculate_reward(self) -> Dict[str, float]:
        """
        Calculate reward with action-dependent progress (Reward V3).

        Components:
        1. Progress: Work amount completed this step (action-dependent, continuous)
        2. Completion Bonus: Houses finished this step (sparse reward)
        3. Queue Penalty: Average waiting time (linear, avoids saturation)

        Key improvements over V2.1:
        - Progress is now based on work amount (∂r/∂a ≠ 0), not house count
        - Queue penalty uses average waiting time (no longer saturates at -5)
        - Adjusted weights to emphasize action-dependent signals

        Returns:
            Dictionary with individual components and total scaled reward
        """
        revealed_ids = list(self.arrival_system.revealed_ids)
        num_revealed = len(revealed_ids)

        # Component 1: Action-Dependent Progress (工作量进度)
        # Calculate actual work completed this step (continuous, action-dependent)
        work_completed_this_step = 0.0
        for house_id in revealed_ids:
            # Work done = reduction in remaining work
            work_diff = float(self.last_arr_rem[house_id]) - float(self._arr_rem[house_id])
            work_done = max(0.0, work_diff)
            work_completed_this_step += work_done

        # Fix Critical Issue #3: Use fixed denominator to maintain temporal consistency
        # Previously used sum(revealed_ids) which changed with batch arrival
        progress_reward = float(work_completed_this_step / self._total_work_all if self._total_work_all > 0 else 0.0)

        # Component 2: Completion Bonus (完成房屋奖励)
        # Sparse bonus for finishing houses (secondary signal)
        current_completed = sum(1 for h in revealed_ids if self._arr_rem[h] <= 0)
        new_completed = max(0, current_completed - self.last_completion_count)
        completion_bonus = float(new_completed / num_revealed if num_revealed > 0 else 0.0)

        # Component 3: Queue Penalty (队列惩罚 - 修复饱和问题)
        # Use average waiting time to avoid saturation
        queue_size = self.waiting_queue.size()
        if queue_size > 0:
            waiting_times = [self.waiting_time[h] for h in self.waiting_queue.get_all()]
            avg_waiting = float(np.mean(waiting_times))
            # Linear penalty: only penalize when average wait exceeds threshold
            queue_penalty = float(-max(0.0, (avg_waiting - 40.0) / 100.0))
        else:
            queue_penalty = 0.0

        # Capacity usage (for monitoring only, not used in reward)
        capacity_usage_monitor = 0.0
        if self.last_workers_available > 0:
            capacity_usage_monitor = self.last_workers_used / self.last_workers_available

        # Weighted combination (before scaling)
        raw_reward = (
            progress_reward * 10.0 +      # Main signal: action-dependent work progress
            completion_bonus * 5.0 +      # Secondary: house completion bonus
            queue_penalty * 1.0           # Soft constraint: queue management
        )

        # Scale up by 100x for better SAC learning dynamics
        scaled_reward = raw_reward * 100.0

        # Update state for next step
        self.last_completion_count = current_completed
        self.last_arr_rem = self._arr_rem.copy()

        # Return all components for TensorBoard logging
        return {
            "total": scaled_reward,
            "progress": progress_reward,
            "completion": completion_bonus,
            "queue_penalty": queue_penalty,
            "capacity_usage_monitor": capacity_usage_monitor,
            "raw_total": raw_reward,
        }

    def _select_candidates(self, queue_ids: List[int]) -> List[int]:
        """
        Adaptive random candidate selection.

        Samples M candidates where M = ratio × queue_size, clamped to [M_min, M_max].
        Uses instance RNG for true stochasticity across different runs.

        Args:
            queue_ids: All houses in queue

        Returns:
            Randomly selected M candidates
        """
        # Calculate adaptive M
        M = self._get_M(len(queue_ids))

        if len(queue_ids) <= M:
            return queue_ids

        # Use instance RNG for stochastic selection
        # This allows different policies to see different candidate sequences,
        # creating a truly stochastic environment for RL to learn from
        selected = self.rng.choice(queue_ids, size=M, replace=False)
        return selected.tolist()

    def step(self, action):
        """
        Execute one daily batch decision.

        Each step processes an entire day:
        1. Begin new day (reveal new houses, sample candidates)
        2. Get allocation from policy (RL or baseline)
        3. Execute allocation and advance to next day
        4. Return observation and reward

        Args:
            action: Action from policy (or None for baseline)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Initialize a new day if needed
        self._begin_day_if_needed()

        # Get allocation for today
        if self.waiting_queue.size() == 0 or len(self.pending_candidates) == 0 or self.current_day_capacity <= 0:
            # No houses to work on or no capacity → empty allocation
            allocation = {}
        elif action is None:
            # Baseline policy
            allocation = self._baseline_allocate()
        else:
            # RL policy
            candidates_view = self.pending_candidates[:self.M_max]
            allocation = self._allocate_from_candidates(
                candidates_view,
                action,
                self.current_day_capacity,
            )

        # Store allocation and advance day
        self.pending_allocation = allocation
        obs, reward, terminated, truncated, info = self._advance_day()

        return obs, reward, terminated, truncated, info

    def _allocate_from_candidates(
        self,
        candidates: List[int],
        action: Any,
        K: int
    ) -> Dict[int, int]:
        """
        Allocate workers to candidates.
        To be implemented by subclasses.

        Args:
            candidates: Selected candidate houses
            action: Action from agent/policy
            K: Available capacity

        Returns:
            Dictionary mapping house_id to number of workers
        """
        raise NotImplementedError("Subclass must implement _allocate_from_candidates")

    def _baseline_allocate(self) -> Dict[int, int]:
        """
        Baseline policies override this to allocate workers for the day.

        Returns:
            Dictionary mapping house_id to number of workers
        """
        raise NotImplementedError("Baseline environments must implement _baseline_allocate")

    def _execute_allocation(self, allocation: Dict[int, int]):
        """
        Execute allocation and update state.

        Args:
            allocation: Dictionary mapping house_id to workers

        Returns:
            Standard gym step outputs
        """
        # ===== 新增：Debug allocation =====
        # if self.day % 10 == 0 or self.day in [37, 38, 39, 40]:
        #     print(f"    [ALLOCATION DEBUG] Day {self.day}:")
        #     print(f"      Allocation dict size: {len(allocation)}")
        #     if len(allocation) > 0:
        #         total_workers = sum(allocation.values())
        #         print(f"      Total workers allocated: {total_workers}")
        #         print(f"      Sample allocations: {list(allocation.items())[:3]}")
        #     else:
        #         print(f"      ⚠️  Allocation is EMPTY!")
        # ==================================

        # Update tracking metrics BEFORE applying work
        self.last_queue_size = self.waiting_queue.size()
        self.last_workers_used = sum(allocation.values()) if allocation else 0
        self.last_workers_available = self.current_day_capacity

        # Apply work with stochastic progress
        self._apply_work(allocation)

        # Remove completed houses
        completed = [
            h for h in self.waiting_queue.get_all()
            if self._arr_rem[h] <= 0
        ]
        self.waiting_queue.remove(completed)

        # Update waiting times
        for house_id in completed:
            self.waiting_time[house_id] = 0  # Reset for completed houses

        for house_id in self.waiting_queue.get_all():
            self.waiting_time[house_id] += 1  # Increment for waiting houses

        # Advance day
        self.day += 1

        # Calculate completion (only for revealed houses)
        revealed_ids = list(self.arrival_system.revealed_ids)

        # Debug: track completion calculation
        # if self.day % 50 == 0 or self.day < 40:
        #     print(f"    [DEBUG] Day {self.day}: Calculating completion...")
        #     print(f"            revealed_ids length: {len(revealed_ids)}")
        #     if len(revealed_ids) > 0:
        #         completed_count = sum(1 for h in revealed_ids if self._arr_rem[h] <= 0)
        #         print(f"            completed houses: {completed_count}")
        #         print(f"            completion: {completed_count / len(revealed_ids):.3f}")

        if len(revealed_ids) > 0:
            # Count completed houses
            revealed_completed = sum(1 for h in revealed_ids if self._arr_rem[h] <= 0)
            completion = revealed_completed / len(revealed_ids)
        else:
            completion = 0.0
            # if self.day % 50 == 0:
            #     print(f"            WARNING: revealed_ids is empty at day {self.day}!")

        # Calculate reward using new four-component reward function
        reward_dict = self._calculate_reward()
        reward = reward_dict["total"]
        self.last_completion = completion

        # Check termination
        terminated = (
            self.waiting_queue.size() == 0 and
            self.arrival_system.is_complete()
        )
        truncated = self.day >= self.max_steps

        # Get observation
        obs = self._get_obs()

        # Calculate average queue time for revealed houses
        revealed_ids = list(self.arrival_system.revealed_ids)
        if len(revealed_ids) > 0:
            avg_queue_time = float(np.mean([self.waiting_time[h] for h in revealed_ids]))
        else:
            avg_queue_time = 0.0

        info = {
            "day": self.day,  # Keep for backward compatibility
            "completed_days": self.day,  # More explicit: days 0 through day-1 are complete
            "decision_day": self.day - 1 if self.day > 0 else 0,  # Day used for this step's decisions
            "completion": completion,
            "completion_rate": completion,  # Alias for evaluate_ppo.py compatibility
            "avg_queue_time": avg_queue_time,  # Average waiting time for revealed houses
            "queue_size": self.waiting_queue.size(),
            "K": self.current_day_capacity,
            "revealed_count": self.arrival_system.get_revealed_count(),
            "M": self.current_day_initial_candidates,
            "day_advanced": True,  # Signal that a real day has passed
            # Reward components for TensorBoard logging
            "reward_progress": reward_dict["progress"],
            "reward_completion": reward_dict["completion"],
            "reward_queue_penalty": reward_dict["queue_penalty"],
            "reward_raw_total": reward_dict["raw_total"],
            # Monitoring metrics (not used in reward)
            "capacity_usage_monitor": reward_dict["capacity_usage_monitor"],
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        Build observation vector.

        Structure:
        - 6 global features
        - M_max*4 candidate features (padded with zeros if M < M_max)
        """
        # Global features
        revealed_ids = list(self.arrival_system.revealed_ids)

        if len(revealed_ids) > 0:
            revealed_total = self._arr_total[revealed_ids].sum()
            revealed_remain = self._arr_rem[revealed_ids].sum()
            revealed_major = (self._arr_dmg[revealed_ids] == 2).sum()

            remain_ratio = revealed_remain / max(1, revealed_total)
            major_ratio = revealed_major / max(1, len(revealed_ids))
        else:
            remain_ratio = 0.0
            major_ratio = 0.0

        global_features = np.array([
            self.day / max(1.0, self.max_steps),
            float(self.current_day_capacity),
            float(self.waiting_queue.size()),
            float(len(revealed_ids)),
            remain_ratio,
            major_ratio,
        ], dtype=np.float32)

        # Candidate features (use M_max for fixed size, pad with zeros)
        candidates = self.pending_candidates if self.pending_candidates else []

        candidate_features = np.zeros((self.M_max, 4), dtype=np.float32)
        for i, house_id in enumerate(candidates):
            if i >= self.M_max:
                break

            # Get true remaining work
            true_remain = self._arr_rem[house_id]

            # Add observation noise if enabled
            if self.obs_noise > 0:
                # Noisy observation: true_value + N(0, σ * true_value)
                # This models uncertainty in damage assessment
                noise = self.rng.normal(0, true_remain * self.obs_noise)
                observed_remain = max(0, true_remain + noise)
            else:
                # Perfect information (legacy mode)
                observed_remain = true_remain

            candidate_features[i] = [
                observed_remain,
                self._arr_dmg[house_id],
                self._arr_cmax[house_id],
                1.0  # mask: valid
            ]

        # Flatten and concatenate
        obs = np.concatenate([
            global_features,
            candidate_features.reshape(-1)
        ])

        return obs

    def reset(self, *, seed: Optional[int] = None, options=None):
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed = seed

        # Reset work arrays
        self._arr_rem = self._arr_total.copy()
        self.last_arr_rem = self._arr_rem.copy()  # Track previous step's remaining work

        # Reset systems
        if isinstance(self.arrival_system, TrueBatchArrival):
            self.arrival_system = TrueBatchArrival(
                self.tasks_df,
                COMBINED_ARRIVAL_CAPACITY_CONFIG["batch_arrival"],
                self.seed if self.seed is not None else 42
            )
        elif isinstance(self.arrival_system, StaticArrival):
            # Fix Critical Issue #1: Reset StaticArrival to avoid empty queue on second episode
            self.arrival_system = StaticArrival(self.tasks_df)

        self.waiting_queue = WaitingQueue()
        self.day = 0
        self.last_completion = 0.0
        self.last_completion_count = 0

        # Reset long-term reward tracking
        self.waiting_time = np.zeros(len(self.tasks_df), dtype=np.int32)
        self.last_queue_size = 0
        self.last_workers_used = 0
        self.last_workers_available = 0
        self.pending_candidates = []
        self.pending_allocation = {}
        self.current_day_capacity = 0
        self.current_day_initial_candidates = 0

        # Initial batch arrival
        initial_arrivals = self.arrival_system.step_to_day(0)

        # ===== 新增：Debug =====
        # print(f"[RESET DEBUG] Initial arrivals from arrival_system: {len(initial_arrivals)} houses")
        # print(f"[RESET DEBUG] Queue size before add: {self.waiting_queue.size()}")
        # =======================

        if initial_arrivals:
            self.waiting_queue.add(initial_arrivals)

        # ===== 新增：Debug =====
        # print(f"[RESET DEBUG] Queue size after add: {self.waiting_queue.size()}")

        # Prepare the first day so observations reflect valid candidates
        self._begin_day_if_needed()

        obs = self._get_obs()
        info = {
            "day": 0,
            "completed_days": 0,
            "decision_day": 0,
            "completion": 0.0,
            "queue_size": self.waiting_queue.size(),
            "revealed_count": self.arrival_system.get_revealed_count(),
            "M": len(self.pending_candidates),
        }

        return obs, info


# ============================================================================
# Environment Subclasses
# ============================================================================

class RLEnv(BaseEnv):
    """
    RL Environment: Learn allocation policy.
    Action is continuous scores for candidates.
    """

    def _allocate_from_candidates(
        self,
        candidates: List[int],
        action: np.ndarray,
        K: int
    ) -> Dict[int, int]:
        """
        Hybrid weighted allocation: Softmax-based but cmax-aware.

        Algorithm:
        1. Use softmax to get "ideal allocation" based on priorities
        2. Clip by cmax constraints
        3. Redistribute capacity limited by cmax
        4. Round to integers preserving total capacity

        Args:
            candidates: List of house IDs available for allocation
            action: Action vector with priority scores
            K: Total available capacity (contractors)

        Returns:
            Dictionary mapping house_id → num_contractors
        """
        M = len(candidates)

        # Edge case: no candidates or no capacity
        if M == 0 or K <= 0:
            return {}

        # Extract priority scores for actual candidates
        priorities = action[:M].astype(np.float64)

        # Get cmax for each candidate
        cmax_array = np.array([
            self._arr_cmax[candidates[i]]
            for i in range(M)
        ], dtype=np.float64)

        # Fix Critical Issue #2: Get remaining work for each candidate
        remaining_work_array = np.array([
            self._arr_rem[candidates[i]]
            for i in range(M)
        ], dtype=np.float64)

        # Step 1: Softmax allocation (ideal distribution)
        # Subtract max for numerical stability
        exp_p = np.exp(priorities - np.max(priorities))
        probs = exp_p / (exp_p.sum() + 1e-10)  # Add epsilon to prevent division by zero
        ideal_allocation = probs * K

        # Step 2: Clip by cmax AND remaining_work constraints
        # This prevents over-allocation to nearly-complete houses
        allocation = np.minimum(ideal_allocation, np.minimum(cmax_array, remaining_work_array))

        # Step 3: Redistribute leftover capacity
        used = allocation.sum()
        leftover = K - used

        if leftover > 1e-6:  # If there's significant leftover capacity
            # Find houses that can accept more contractors
            can_take_more = cmax_array - allocation
            can_take_more = np.maximum(can_take_more, 0)

            if can_take_more.sum() > 1e-6:
                # Redistribute proportionally based on original priorities
                # but only to houses that can accept more
                eligible_mask = can_take_more > 1e-6
                eligible_probs = probs * eligible_mask

                if eligible_probs.sum() > 1e-6:
                    eligible_probs = eligible_probs / eligible_probs.sum()
                    extra_allocation = eligible_probs * leftover
                    extra_allocation = np.minimum(extra_allocation, can_take_more)
                    allocation += extra_allocation

        # Step 4: Convert to integers while preserving total capacity
        int_allocation = np.floor(allocation).astype(int)
        remaining = int(K - int_allocation.sum())

        if remaining > 0:
            # Distribute remaining contractors based on fractional parts
            fractional = allocation - int_allocation

            # Only give to houses that haven't hit their cmax
            can_accept = (int_allocation < cmax_array)
            fractional[~can_accept] = -1  # Mark ineligible

            # Give to houses with largest fractional parts
            if np.any(can_accept):
                top_indices = np.argsort(-fractional)[:remaining]
                # Double-check they can actually accept
                for idx in top_indices:
                    if int_allocation[idx] < cmax_array[idx]:
                        int_allocation[idx] += 1

        # Step 5: Build result dictionary (only include houses with allocation > 0)
        result = {}
        for i in range(M):
            if int_allocation[i] > 0:
                result[candidates[i]] = int(int_allocation[i])

        # Validation assertions (can be removed in production)
        total_allocated = sum(result.values())
        assert total_allocated <= K, f"Over-allocated: {total_allocated} > {K}"
        for house_id, num in result.items():
            assert num <= self._arr_cmax[house_id], \
                f"House {house_id} allocated {num} > cmax {self._arr_cmax[house_id]}"

        return result


class BaselineEnv(BaseEnv):
    """
    Baseline Environment: Fixed priority rules.
    Supports LJF, SJF, and Random policies.
    """

    def __init__(
        self,
        region_key: str,
        policy: str,
        num_contractors: Optional[int] = None,
        M_ratio: float = 0.10,
        M_min: int = 512,
        M_max: int = 2048,
        use_batch_arrival: bool = True,
        use_capacity_ramp: bool = False,
        stochastic_duration: bool = True,
        observation_noise: float = 0.15,
        capacity_noise: float = 0.10,
        seed: Optional[int] = None,
        max_steps: int = MAX_STEPS,
    ):
        """
        Initialize baseline environment.

        Args:
            policy: "LJF", "SJF", or "Random"
            Other args same as BaseEnv
        """
        super().__init__(
            region_key=region_key,
            num_contractors=num_contractors,
            M_ratio=M_ratio,
            M_min=M_min,
            M_max=M_max,
            use_batch_arrival=use_batch_arrival,
            use_capacity_ramp=use_capacity_ramp,
            stochastic_duration=stochastic_duration,
            observation_noise=observation_noise,
            capacity_noise=capacity_noise,
            seed=seed,
            max_steps=max_steps,
        )

        assert policy in ["LJF", "SJF", "Random"], \
            f"Policy must be LJF, SJF, or Random, got {policy}"
        self.policy = policy

    def step(self, action=None):
        """Baseline doesn't use action input."""
        return super().step(action=None)

    def _baseline_allocate(self) -> Dict[int, int]:
        """
        Greedy allocation based on policy priority.

        Allocates contractors greedily:
        1. Sort candidates by policy (LJF/SJF/Random)
        2. Greedily allocate respecting cmax constraints
        3. Continue until capacity exhausted

        Returns:
            Dictionary mapping house_id to number of contractors
        """
        if len(self.pending_candidates) == 0 or self.current_day_capacity <= 0:
            return {}

        candidates = self.pending_candidates.copy()

        # Sort by policy
        if self.policy == "LJF":
            # Longest Job First: prioritize houses with most total work
            candidates.sort(key=lambda h: self._arr_total[h], reverse=True)
        elif self.policy == "SJF":
            # Shortest Job First: prioritize houses with least total work
            candidates.sort(key=lambda h: self._arr_total[h])
        elif self.policy == "Random":
            # Random: shuffle candidates
            self.rng.shuffle(candidates)

        # Greedy allocation
        allocation = {}
        remaining_capacity = self.current_day_capacity

        for house_id in candidates:
            if remaining_capacity <= 0:
                break

            # How many contractors can we give to this house?
            cmax = int(np.ceil(self._arr_cmax[house_id]))
            remaining_work = int(np.ceil(self._arr_rem[house_id]))
            give = min(cmax, remaining_work, remaining_capacity)

            if give > 0:
                allocation[house_id] = give
                remaining_capacity -= give

        return allocation


class OracleEnv(BaseEnv):
    """
    Oracle Environment: No candidate limit.
    Sees entire queue, used to quantify information cost.
    """

    def __init__(
        self,
        region_key: str,
        policy: str,
        num_contractors: Optional[int] = None,
        M_ratio: float = 0.10,
        M_min: int = 512,
        M_max: int = 2048,
        use_batch_arrival: bool = True,
        use_capacity_ramp: bool = False,
        stochastic_duration: bool = True,
        observation_noise: float = 0.15,
        capacity_noise: float = 0.10,
        seed: Optional[int] = None,
        max_steps: int = MAX_STEPS,
    ):
        """
        Initialize oracle environment.

        Args:
            policy: "LJF" or "SJF"
            Other args same as BaseEnv
        """
        super().__init__(
            region_key=region_key,
            num_contractors=num_contractors,
            M_ratio=M_ratio,
            M_min=M_min,
            M_max=M_max,
            use_batch_arrival=use_batch_arrival,
            use_capacity_ramp=use_capacity_ramp,
            stochastic_duration=stochastic_duration,
            observation_noise=observation_noise,
            capacity_noise=capacity_noise,
            seed=seed,
            max_steps=max_steps,
        )

        assert policy in ["LJF", "SJF"], \
            f"Oracle policy must be LJF or SJF, got {policy}"
        self.policy = policy

    def _select_candidates(self, queue_ids: List[int]) -> List[int]:
        """Oracle sees entire queue (no M limit)."""
        return queue_ids

    def _baseline_allocate(self) -> Dict[int, int]:
        """
        Oracle greedy allocation with full queue visibility.

        Same as BaselineEnv but sees entire queue (no M limit).

        Returns:
            Dictionary mapping house_id to number of contractors
        """
        if len(self.pending_candidates) == 0 or self.current_day_capacity <= 0:
            return {}

        candidates = self.pending_candidates.copy()

        # Sort by policy
        if self.policy == "LJF":
            candidates.sort(key=lambda h: self._arr_total[h], reverse=True)
        else:  # SJF
            candidates.sort(key=lambda h: self._arr_total[h])

        # Greedy allocation
        allocation = {}
        remaining_capacity = self.current_day_capacity

        for house_id in candidates:
            if remaining_capacity <= 0:
                break

            cmax = int(np.ceil(self._arr_cmax[house_id]))
            remaining_work = int(np.ceil(self._arr_rem[house_id]))
            give = min(cmax, remaining_work, remaining_capacity)

            if give > 0:
                allocation[house_id] = give
                remaining_capacity -= give

        return allocation


# ============================================================================
# Legacy Support
# ============================================================================

class HousegymRLENV(RLEnv):
    """
    Legacy class for backward compatibility.
    Deprecated - use RLEnv instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "HousegymRLENV is deprecated. Use RLEnv instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Try to adapt old arguments
        if 'tasks_df' in kwargs:
            # Old interface with tasks_df
            tasks_df = kwargs.pop('tasks_df')
            resources = kwargs.pop('resources', {})
            region_key = resources.get('region_name', 'Mataram')
            num_contractors = resources.get('workers', None)

            super().__init__(
                region_key=region_key,
                num_contractors=num_contractors,
                **kwargs
            )
        else:
            super().__init__(*args, **kwargs)

    def last_candidate_view(self):
        """Legacy method for getting candidate info."""
        candidates = self.pending_candidates
        if not candidates:
            queue_ids = self.waiting_queue.get_all()
            candidates = self._select_candidates(queue_ids) if queue_ids else []

        remain = np.zeros(self.M_max, dtype=np.float32)
        mask = np.zeros(self.M_max, dtype=np.float32)

        for i, h in enumerate(candidates[:self.M_max]):
            remain[i] = self._arr_rem[h]
            mask[i] = 1.0

        return {"remain": remain, "mask": mask}
