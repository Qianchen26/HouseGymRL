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
from typing import Optional, Dict, List, Tuple, Any
import warnings
from pathlib import Path
import pickle

# Import configuration
import config
from config import (
    M_CANDIDATES,
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
        observation_noise: float = 0.15,
        capacity_noise: float = 0.10,
        use_longterm_reward: bool = True,
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
            observation_noise: Std dev of observation noise as fraction of true value (default: 0.15)
            capacity_noise: Range of capacity reduction (default: 0.10 = 90%-100% of base)
            use_longterm_reward: Whether to use long-term reward function (default: True)
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
        self.stochastic_duration = stochastic_duration
        self.obs_noise = observation_noise
        self.capacity_noise = capacity_noise
        self.use_longterm_reward = use_longterm_reward

        # Load scenario data
        self.tasks_df = self._load_scenario(region_key)

        # Initialize work arrays
        self._arr_total = self.tasks_df['man_days_total'].values.astype(np.float32)
        self._arr_rem = self.tasks_df['man_days_remaining'].values.copy().astype(np.float32)
        self._arr_dmg = self.tasks_df['damage_level'].values.astype(np.int32)
        self._arr_cmax = self.tasks_df['cmax_per_day'].values.astype(np.float32)

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

        # Long-term reward tracking variables
        self.waiting_time = np.zeros(len(self.tasks_df), dtype=np.int32)
        self.last_queue_size = 0
        self.last_workers_used = 0
        self.last_workers_available = 0

        # Intra-day tracking state
        self.pending_candidates: List[int] = []
        self.pending_allocation: Dict[int, int] = {}
        self.capacity_remaining: int = 0
        self.current_day_capacity: int = 0
        self.step_in_day: int = 0
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

    def _build_action_mask(self) -> np.ndarray:
        """Build an action mask for the current pending candidates."""
        mask = np.zeros(self.M_max + 1, dtype=np.float32)
        num_candidates = min(len(self.pending_candidates), self.M_max)
        if num_candidates > 0:
            mask[:num_candidates] = 1.0
        mask[self.M_max] = 1.0  # no-op always allowed
        return mask

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
        if not (self.capacity_remaining <= 0 and self.step_in_day == 0 and not self.pending_allocation):
            return

        # Reveal new houses for this day
        new_arrivals = self.arrival_system.step_to_day(self.day)
        if new_arrivals:
            self.waiting_queue.add(new_arrivals)

        # Sample capacity for the day
        capacity = self._get_capacity_for_day(self.day)
        self.capacity_remaining = capacity
        self.current_day_capacity = capacity
        self.current_day_initial_candidates = 0
        self.pending_allocation = {}
        self.pending_candidates = []
        self.step_in_day = 0

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

        # Reset intra-day state
        self.pending_candidates = []
        self.pending_allocation = {}
        self.capacity_remaining = 0
        self.current_day_capacity = 0
        self.current_day_initial_candidates = 0
        self.step_in_day = 0

        info["day_advanced"] = True
        info["step_in_day"] = 0

        if not terminated and not truncated:
            self._begin_day_if_needed()
            obs = self._get_obs()

        info["action_mask"] = self._build_action_mask()
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

    def get_candidate_seed_for_step(self, step_num: int) -> int:
        """
        Get the seed that was/will be used for candidate selection in step N.

        Args:
            step_num: The step number (1-indexed)

        Returns:
            The seed value used for candidate selection

        Example:
            After 50 steps, to reproduce the candidates from step 50:
            seed = env.get_candidate_seed_for_step(50)  # Returns seed for day 49
        """
        # Step N uses day N-1 for decisions
        decision_day = max(0, step_num - 1)
        return (self.seed if self.seed is not None else 42) + decision_day * 1000

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

    def _calculate_reward(self, completed: List[int]) -> float:
        """
        Calculate long-term reward with multiple components.

        Components:
        1. Completion reward: Houses completed (no damage weighting)
        2. Queue reduction bonus: Encourages clearing the queue
        3. Urgency penalty: Punishes long waiting times
        4. Worker efficiency bonus: Encourages efficient resource use

        Args:
            completed: List of house IDs completed this step

        Returns:
            Combined reward value
        """
        if not self.use_longterm_reward:
            # Simple immediate reward (legacy mode)
            revealed_ids = list(self.arrival_system.revealed_ids)
            if len(revealed_ids) > 0:
                return len(completed) / len(revealed_ids)
            return 0.0

        # Component 1: Immediate completion (NO damage weighting per user request)
        completion_reward = 0.0
        revealed_ids = list(self.arrival_system.revealed_ids)
        if len(revealed_ids) > 0:
            completion_reward = len(completed) / len(revealed_ids)

        # Component 2: Queue reduction bonus
        current_queue = self.waiting_queue.size()
        if self.last_queue_size > 0:
            queue_reduction = (self.last_queue_size - current_queue) / self.last_queue_size
            queue_bonus = max(0, queue_reduction) * 0.1
        else:
            queue_bonus = 0.0

        # Component 3: Urgency penalty (punish long waits)
        urgency_penalty = 0.0
        for house_id in self.waiting_queue.get_all():
            if self.waiting_time[house_id] > 50:  # Houses waiting > 50 days
                urgency_penalty -= (self.waiting_time[house_id] - 50) / 10000

        # Component 4: Worker efficiency bonus
        if self.last_workers_available > 0:
            efficiency = self.last_workers_used / self.last_workers_available
            efficiency_bonus = efficiency * 0.03
        else:
            efficiency_bonus = 0.0

        # Combine components with weights
        total_reward = (
            completion_reward * 1.0 +      # Main signal: complete houses
            queue_bonus * 0.2 +             # Secondary: reduce queue
            urgency_penalty * 0.1 +         # Penalty: avoid long waits
            efficiency_bonus * 0.05         # Bonus: use workers efficiently
        )

        return total_reward

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
        """Execute one intra-day decision (choose next house or end the day)."""
        # Initialize a new day if needed
        self._begin_day_if_needed()

        # Nothing to do → advance day immediately
        if self.capacity_remaining <= 0 or self.waiting_queue.size() == 0 or len(self.pending_candidates) == 0:
            return self._advance_day()

        # Determine selection source (RL vs. baseline)
        if action is None:
            chosen_house_id, is_noop = self._baseline_pick_next()
        else:
            candidates_view = self.pending_candidates[:self.M_max]
            chosen_house_id, is_noop = self._allocate_from_candidates(
                candidates_view,
                action,
                self.capacity_remaining,
            )

        # If policy decides to end the day (or nothing to choose)
        if is_noop or chosen_house_id is None:
            return self._advance_day()

        # Allocate as many workers as possible to the selected house
        cmax = int(np.ceil(self._arr_cmax[chosen_house_id]))
        remain = int(np.ceil(self._arr_rem[chosen_house_id]))
        give = min(cmax, remain, self.capacity_remaining)

        if give > 0:
            self.pending_allocation[chosen_house_id] = (
                self.pending_allocation.get(chosen_house_id, 0) + give
            )
            self.capacity_remaining -= give

        # Remove the house from today's candidate set
        if chosen_house_id in self.pending_candidates:
            self.pending_candidates.remove(chosen_house_id)

        # End of day conditions
        if self.capacity_remaining <= 0 or len(self.pending_candidates) == 0:
            return self._advance_day()

        # Otherwise remain in the same day (reward=0)
        self.step_in_day += 1
        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "day": self.day,
            "day_advanced": False,
            "step_in_day": self.step_in_day,
            "K_remaining": self.capacity_remaining,
            "queue_size": self.waiting_queue.size(),
            "completion": self.last_completion,
            "M": len(self.pending_candidates),
            "decision_day": self.day,
            "revealed_count": self.arrival_system.get_revealed_count(),
            "action_mask": self._build_action_mask(),
        }
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

    def _baseline_pick_next(self) -> Tuple[Optional[int], bool]:
        """Baseline policies override this to pick a house or end the day."""
        raise NotImplementedError("Baseline environments must implement _baseline_pick_next")

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

        # Calculate reward using new long-term reward function
        reward = self._calculate_reward(completed)
        self.last_completion = completion

        # Check termination
        terminated = (
            self.waiting_queue.size() == 0 and
            self.arrival_system.is_complete()
        )
        truncated = self.day >= self.max_steps

        # Get observation
        obs = self._get_obs()

        info = {
            "day": self.day,  # Keep for backward compatibility
            "completed_days": self.day,  # More explicit: days 0 through day-1 are complete
            "decision_day": self.day - 1 if self.day > 0 else 0,  # Day used for this step's decisions
            "completion": completion,
            "queue_size": self.waiting_queue.size(),
            "K": self.current_day_capacity,
            "revealed_count": self.arrival_system.get_revealed_count(),
            "M": self.current_day_initial_candidates,
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
            float(self.capacity_remaining),
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

        # Reset systems
        if isinstance(self.arrival_system, TrueBatchArrival):
            self.arrival_system = TrueBatchArrival(
                self.tasks_df,
                COMBINED_ARRIVAL_CAPACITY_CONFIG["batch_arrival"],
                self.seed if self.seed is not None else 42
            )

        self.waiting_queue = WaitingQueue()
        self.day = 0
        self.last_completion = 0.0

        # Reset long-term reward tracking
        self.waiting_time = np.zeros(len(self.tasks_df), dtype=np.int32)
        self.last_queue_size = 0
        self.last_workers_used = 0
        self.last_workers_available = 0
        self.pending_candidates = []
        self.pending_allocation = {}
        self.capacity_remaining = 0
        self.current_day_capacity = 0
        self.step_in_day = 0
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
            "day_advanced": False,
            "step_in_day": 0,
            "action_mask": self._build_action_mask(),
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
    ) -> Tuple[Optional[int], bool]:
        """
        Pick a single candidate (or no-op) based on action scores.

        Returns:
            (house_id, is_noop)
        """
        num_candidates = len(candidates)
        if num_candidates == 0 or K <= 0:
            return None, True

        candidate_scores = action[:num_candidates]
        noop_index = self.M_max if self.M_max < len(action) else len(action) - 1
        noop_score = float(action[noop_index])

        # if self.day % 10 == 0 or self.day in [37, 38, 39, 40]:
        #     print(f"    [RL ALLOC DEBUG] Day {self.day}:")
        #     print(f"      K available: {K}")
        #     print(f"      Candidates: {num_candidates}")
        #     if num_candidates > 0:
        #         print(f"      Candidate scores: min={candidate_scores.min():.3f}, max={candidate_scores.max():.3f}, mean={candidate_scores.mean():.3f}")
        #     print(f"      No-op score: {noop_score:.3f}")

        if num_candidates == 0 or (num_candidates > 0 and noop_score >= float(candidate_scores.max())):
            return None, True

        idx = int(np.argmax(candidate_scores))
        return candidates[idx], False


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
        use_longterm_reward: bool = True,
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
            use_longterm_reward=use_longterm_reward,
            seed=seed,
            max_steps=max_steps,
        )

        assert policy in ["LJF", "SJF", "Random"], \
            f"Policy must be LJF, SJF, or Random, got {policy}"
        self.policy = policy

    def step(self, action=None):
        """Baseline doesn't use action input."""
        return super().step(action=None)

    def _baseline_pick_next(self) -> Tuple[Optional[int], bool]:
        if len(self.pending_candidates) == 0 or self.capacity_remaining <= 0:
            return None, True

        candidates = self.pending_candidates

        if self.policy == "LJF":
            selected = max(candidates, key=lambda h: self._arr_total[h])
        elif self.policy == "SJF":
            selected = min(candidates, key=lambda h: self._arr_total[h])
        elif self.policy == "Random":
            selected = int(self.rng.choice(candidates))
        else:
            selected = int(self.rng.choice(candidates))

        return selected, False


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
        use_longterm_reward: bool = True,
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
            use_longterm_reward=use_longterm_reward,
            seed=seed,
            max_steps=max_steps,
        )

        assert policy in ["LJF", "SJF"], \
            f"Oracle policy must be LJF or SJF, got {policy}"
        self.policy = policy

    def _select_candidates(self, queue_ids: List[int]) -> List[int]:
        """Oracle sees entire queue (no M limit)."""
        return queue_ids

    def _baseline_pick_next(self) -> Tuple[Optional[int], bool]:
        if len(self.pending_candidates) == 0 or self.capacity_remaining <= 0:
            return None, True

        if self.policy == "LJF":
            selected = max(self.pending_candidates, key=lambda h: self._arr_total[h])
        else:
            selected = min(self.pending_candidates, key=lambda h: self._arr_total[h])

        return selected, False


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
