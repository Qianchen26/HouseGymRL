"""
HouseGym RL Environment

Simulates post-disaster housing reconstruction scheduling.

Classes:
    - BaseEnv: Base class with all common constraints
    - RLEnv: For RL training (continuous action space)
    - BaselineEnv: For baseline policies (LJF/SJF/Random)
    - OracleEnv: No candidate limit (quantifies information cost)

Design Principles:
    - Top-M selection by waiting time (fairness-first visibility)
    - Batch arrival system (simulates real assessment timeline)
    - Stochastic work progress and observation noise (robustness)

Architecture:
    Houses arrive in batches, enter a waiting queue, and are surfaced as
    candidates based on waiting time. The agent allocates contractors to
    candidates, respecting per-house capacity limits (cmax). Work progresses
    stochastically until completion.
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
    DATA_DIR,
    ACTION_DIM,
    ACTION_LOW,
    ACTION_HIGH,
    MAX_QUEUE_SIZE,
    SOFTMAX_TEMPERATURE,
)

class BatchArrival:
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
        """Advance to given day and return newly revealed houses."""
        new_arrivals = []

        for scheduled_day, houses in self.schedule.items():
            if scheduled_day <= day and scheduled_day > self.current_day:
                new_arrivals.extend(houses)
                self.revealed_ids.update(houses)

        self.current_day = day
        return new_arrivals

    def is_complete(self) -> bool:
        """Check if all houses have been revealed."""
        return self.current_day >= max(self.schedule.keys())

    def get_revealed_count(self) -> int:
        """Get number of houses revealed so far."""
        return len(self.revealed_ids)


class CapacityRamp:
    """
    Models gradual contractor mobilization (disabled by default).

    Capacity grows from 0 to max over time. Based on Lombok reality:
    - Days 0-36: Planning phase (K=0)
    - Days 36-216: Linear ramp-up
    - Day 216+: Full capacity

    Note: Controlled by CAPACITY_RAMP_ENABLED in config.py. Currently disabled
    in favor of FixedCapacity for simpler training dynamics.
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


class BaseEnv(gym.Env):
    """
    Base environment implementing all common constraints.

    Key features:
    - Batch arrival (information gradually revealed)
    - Capacity ramp (resources gradually available, currently disabled)
    - Top-M candidate selection by waiting time (fairness-first visibility)
    - Fair constraints for all strategies (RL/baseline)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        region_key: str,
        num_contractors: Optional[int] = None,
        M_ratio: float = 0.10,
        M_min: int = 1024,
        M_max: int = 1024,
        use_capacity_ramp: bool = False,
        stochastic_duration: bool = True,
        observation_noise: float = 0.15,
        capacity_noise: float = 0.10,
        seed: Optional[int] = None,
        max_steps: int = MAX_STEPS,
        capacity_ceiling: Optional[int] = None,
        use_legacy_capacity_ceiling: bool = False,
    ):
        """
        Initialize base environment.

        Args:
            region_key: Region name from REGION_CONFIG
            num_contractors: Number of contractors (if None, use region default)
            M_ratio: Proportion of queue to sample as candidates (default: 0.10 = 10%)
            M_min: Minimum number of candidates (default: 512)
            M_max: Maximum number of candidates (default: 2048)
            use_capacity_ramp: Whether to use capacity ramp system
            stochastic_duration: Whether work progress is stochastic (default: True)
            observation_noise: Std dev of observation noise as fraction of true value (default: 0.15)
            capacity_noise: Range of capacity reduction (default: 0.10 = 90%-100% of base)
            seed: Random seed
            max_steps: Maximum simulation days
            capacity_ceiling: Maximum daily capacity (if None and use_legacy_capacity_ceiling=False, no ceiling)
            use_legacy_capacity_ceiling: If True, use legacy formula M_max * max(cmax_per_day)
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
        self.observation_noise = observation_noise
        self.obs_noise = observation_noise
        self.capacity_noise = capacity_noise

        # Load scenario data
        self.tasks_df = self._load_scenario(region_key)

        # Verify index is 0-based continuous (required for array indexing)
        expected_ids = set(range(len(self.tasks_df)))
        actual_ids = set(self.tasks_df.index)
        assert expected_ids == actual_ids, \
            f"house_id not 0-based continuous: expected 0..{len(self.tasks_df)-1}, got {sorted(actual_ids)[:5]}..."

        # Initialize work arrays
        self._arr_total = self.tasks_df['man_days_total'].values.astype(np.float32)
        self._arr_rem = self.tasks_df['man_days_remaining'].values.copy().astype(np.float32)
        self._arr_dmg = self.tasks_df['damage_level'].values.astype(np.int32)
        self._arr_cmax = self.tasks_df['cmax_per_day'].values.astype(np.float32)

        # Set capacity ceiling based on parameters
        if use_legacy_capacity_ceiling:
            # Legacy behavior: M_max * max(cmax_per_day) - truncates large contractor pools
            self.capacity_ceiling = int(self.M_max * float(self._arr_cmax.max()))
        elif capacity_ceiling is not None:
            # Explicit ceiling provided
            self.capacity_ceiling = capacity_ceiling
        else:
            # No ceiling - allows full contractor capacity
            self.capacity_ceiling = None

        # Total work across all houses (fixed denominator for reward scaling)
        self._total_work_all = float(sum(self._arr_total))

        # Initialize systems
        self.arrival_system = BatchArrival(
            self.tasks_df,
            COMBINED_ARRIVAL_CAPACITY_CONFIG["batch_arrival"],
            seed if seed is not None else 42
        )

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
            df = pd.read_pickle(pkl_path)
            # Ensure 0-based continuous index for array indexing
            if 'original_house_id' not in df.columns:
                df['original_house_id'] = df.index
            df = df.reset_index(drop=True)
            df.index.name = 'house_id'
            return df

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

        Step 1: Dict observation with padding and mask
        Step 2: Learned heuristic (6-dimensional weight vector)
        """
        # Observation: Dict with global features, candidate features, and mask
        from config import MAX_QUEUE_SIZE, OBS_G, OBS_HOUSE_FEATURES, HEURISTIC_ACTION_DIM

        self.observation_space = spaces.Dict({
            'global': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(OBS_G,),  # 6 global features
                dtype=np.float32
            ),
            'candidates': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(MAX_QUEUE_SIZE, OBS_HOUSE_FEATURES),  # (1024, 6)
                dtype=np.float32
            ),
            'mask': spaces.Box(
                low=0,
                high=1,
                shape=(MAX_QUEUE_SIZE,),  # (1024,)
                dtype=np.float32
            )
        })

        # Action: per-candidate scores (direct allocation)
        # Each candidate gets a score from the policy, used for softmax allocation
        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(ACTION_DIM,),  # 1024 scores (one per candidate slot)
            dtype=np.float32
        )

    def _get_capacity_for_day(self, day: int) -> int:
        """Sample capacity for a specific day (with noise if enabled)."""
        base_capacity = self.capacity_system.get_capacity(day)
        if self.capacity_noise > 0 and base_capacity > 0:
            noise_factor = self.rng.uniform(1.0 - self.capacity_noise, 1.0)
            base_capacity = int(base_capacity * noise_factor)
        if self.capacity_ceiling is not None:
            base_capacity = min(base_capacity, self.capacity_ceiling)
        return max(0, base_capacity)

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
                # Do not exceed remaining work
                actual_progress = min(actual_progress, self._arr_rem[house_id])
            else:
                # Deterministic progress
                actual_progress = min(workers, self._arr_rem[house_id])

            # Update remaining work
            self._arr_rem[house_id] -= actual_progress
            self._arr_rem[house_id] = max(0, self._arr_rem[house_id])

    def _calculate_reward(self) -> Dict[str, float]:
        """
        Calculate multi-component reward signal.

        Components:
        1. Progress: Work amount completed this step (action-dependent)
        2. Completion Bonus: Houses finished this step (sparse)
        3. Queue Penalty: Average waiting time (capped to avoid saturation)

        Returns:
            Dictionary with individual components and total scaled reward
        """
        revealed_ids = list(self.arrival_system.revealed_ids)
        num_revealed = len(revealed_ids)

        # Component 1: Work Progress (action-dependent)
        work_completed_this_step = 0.0
        for house_id in revealed_ids:
            work_diff = float(self.last_arr_rem[house_id]) - float(self._arr_rem[house_id])
            work_done = max(0.0, work_diff)
            work_completed_this_step += work_done

        # Use fixed denominator (total work) for temporal consistency
        progress_reward = float(work_completed_this_step / self._total_work_all if self._total_work_all > 0 else 0.0)

        # Component 2: Completion Bonus (sparse signal for finishing houses)
        current_completed = sum(1 for h in revealed_ids if self._arr_rem[h] <= 0)
        new_completed = max(0, current_completed - self.last_completion_count)
        completion_bonus = float(new_completed / num_revealed if num_revealed > 0 else 0.0)

        # Component 3: Queue Penalty (average waiting time, capped to avoid saturation)
        queue_size = self.waiting_queue.size()
        if queue_size > 0:
            waiting_times = [self.waiting_time[h] for h in self.waiting_queue.get_all()]
            avg_waiting = float(np.mean(waiting_times))
            # Linear penalty with relaxed threshold/slope and cap
            # threshold=80, slope=/300, cap=-0.5 to prevent dominating reward
            raw_penalty = -max(0.0, (avg_waiting - 80.0) / 300.0)
            queue_penalty = float(max(-0.5, raw_penalty))
        else:
            queue_penalty = 0.0

        # Capacity usage (for monitoring only, not used in reward)
        capacity_usage_monitor = 0.0
        if self.last_workers_available > 0:
            capacity_usage_monitor = self.last_workers_used / self.last_workers_available

        # Weighted combination (before scaling)
        raw_reward = (
            progress_reward * 15.0 +     # Boost progress signal
            completion_bonus * 7.0 +     # Boost completion signal
            queue_penalty * 0.05         # Reduce queue penalty weight
        )

        # Scale up by 100x for better learning dynamics
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
        Top-M candidate selection by waiting time.

        Samples M candidates where M = ratio × queue_size, clamped to [M_min, M_max],
        then takes the longest-waiting houses first.
        Uses instance RNG for true stochasticity across different runs.

        Args:
            queue_ids: All houses in queue

        Returns:
            Top-M candidates by waiting time (longest waiting first)
        """
        queue_size = len(queue_ids)
        if queue_size == 0:
            return []

        M = self._get_M(queue_size)

        # Sort by waiting time (descending). Tie-breaker by house_id for determinism.
        sorted_ids = sorted(queue_ids, key=lambda h: (self.waiting_time[h], h), reverse=True)

        if queue_size <= M:
            return sorted_ids

        return sorted_ids[:M]

    def step(self, action):
        """
        Execute one daily batch decision.

        Each step processes an entire day:
        1. Begin new day (reveal new houses, sample candidates)
        2. Build candidate snapshot (for consistent obs/allocation/reward)
        3. Get allocation from policy (RL or baseline)
        4. Execute allocation and advance to next day
        5. Return observation and reward

        Args:
            action: Action from policy (or None for baseline)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Initialize a new day if needed
        self._begin_day_if_needed()

        # Build candidate snapshot for this step (CRITICAL: must be first!)
        # This ensures allocation, reward, and observation use the same candidate set
        self._build_candidates_obs()

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

        if len(revealed_ids) > 0:
            revealed_completed = sum(1 for h in revealed_ids if self._arr_rem[h] <= 0)
            completion = revealed_completed / len(revealed_ids)
        else:
            completion = 0.0

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

    def _build_candidates_obs(self) -> None:
        """
        Construct normalized feature matrix and mask for candidate houses, then cache.

        This function MUST be called at the beginning of each step() to ensure that
        allocation, reward computation, and observation all use the same candidate snapshot.

        Args:
            None (uses self.pending_candidates from internal state)

        Returns:
            None (caches results in self._last_candidates_obs, self._last_mask, self._last_candidate_ids)

        Requires:
            - self.pending_candidates is populated (list of house IDs)
            - len(self.pending_candidates) <= MAX_QUEUE_SIZE

        Ensures:
            - self._last_candidates_obs.shape == (MAX_QUEUE_SIZE, OBS_HOUSE_FEATURES)
            - self._last_mask.shape == (MAX_QUEUE_SIZE,)
            - self._last_mask[i] == 1.0 for i < len(candidate_ids), else 0.0
            - All features are normalized using FEATURE_SCALES

        Side effects:
            - Sets self._last_candidates_obs
            - Sets self._last_mask
            - Sets self._last_candidate_ids
        """
        from config import MAX_QUEUE_SIZE, OBS_HOUSE_FEATURES, FEATURE_SCALES

        # Get candidate house IDs (Step 1: still M=1024)
        candidate_ids = self.pending_candidates if self.pending_candidates else []

        N = len(candidate_ids)
        assert N <= MAX_QUEUE_SIZE, f"Candidate pool exceeds MAX_QUEUE_SIZE: {N} > {MAX_QUEUE_SIZE}"

        # Initialize padding arrays
        candidates = np.zeros((MAX_QUEUE_SIZE, OBS_HOUSE_FEATURES), dtype=np.float32)
        mask = np.zeros(MAX_QUEUE_SIZE, dtype=np.float32)

        # Fill features for actual candidates (normalized for consistency with scoring)
        for i, house_id in enumerate(candidate_ids):
            if i >= MAX_QUEUE_SIZE:
                break

            # Extract raw features
            # Get true remaining work with optional observation noise
            true_remain = self._arr_rem[house_id]
            if self.obs_noise > 0:
                # Noisy observation: true_value + N(0, σ * true_value)
                noise = self.rng.normal(0, true_remain * self.obs_noise)
                observed_remain = max(0, true_remain + noise)
            else:
                observed_remain = true_remain

            waiting_time = float(self.waiting_time[house_id])
            total_work = float(self._arr_total[house_id])
            damage_level = float(self._arr_dmg[house_id])
            cmax = float(self._arr_cmax[house_id])

            # Normalize using FEATURE_SCALES from config
            candidates[i] = [
                observed_remain / FEATURE_SCALES[0],  # Remaining work (scale: 100)
                waiting_time / FEATURE_SCALES[1],     # Days waiting (scale: 100)
                total_work / FEATURE_SCALES[2],       # Total workload (scale: 100)
                damage_level / FEATURE_SCALES[3],     # Severity 0-2 (scale: 2)
                cmax / FEATURE_SCALES[4],             # Max daily capacity (scale: 10)
                1.0                                    # Bias term (always 1.0)
            ]
            mask[i] = 1.0  # Mark as valid candidate

        # Cache for use in scoring and observation
        self._last_candidates_obs = candidates
        self._last_mask = mask
        self._last_candidate_ids = candidate_ids

        # Postcondition checks
        assert self._last_candidates_obs.shape == (MAX_QUEUE_SIZE, OBS_HOUSE_FEATURES)
        assert self._last_mask.shape == (MAX_QUEUE_SIZE,)
        assert np.sum(self._last_mask) == min(N, MAX_QUEUE_SIZE), "Mask count mismatch"

    def _score_candidates(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute priority scores for all candidates using learned weights.

        This implements the Learned Heuristic approach (Step 2):
        score_i = w^T @ features_i for each candidate i

        Args:
            weights: (HEURISTIC_ACTION_DIM,) weight vector from policy

        Returns:
            scores: (MAX_QUEUE_SIZE,) priority scores for all candidates
                   - Valid candidates have scores = weights @ features
                   - Padded positions have scores = 0.0 (masked by _last_mask)

        Requires:
            - _build_candidates_obs() has been called (sets _last_candidates_obs and _last_mask)
            - weights.shape == (HEURISTIC_ACTION_DIM,)

        Ensures:
            - scores.shape == (MAX_QUEUE_SIZE,)
            - scores[i] = 0.0 for all i where _last_mask[i] == 0.0
            - All scores are non-negative (weights and features are normalized to [0, 1])

        Example:
            weights = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])  # From policy
            scores = self._score_candidates(weights)  # (1024,) scores
            # scores[i] = 0.5*remain + 0.3*waiting + 0.1*total + ... for valid candidates
        """
        from config import MAX_QUEUE_SIZE, HEURISTIC_ACTION_DIM

        # Precondition checks
        assert hasattr(self, '_last_candidates_obs'), "Must call _build_candidates_obs() before scoring"
        assert hasattr(self, '_last_mask'), "Must call _build_candidates_obs() before scoring"
        assert weights.shape == (HEURISTIC_ACTION_DIM,), \
            f"Weights shape mismatch: {weights.shape} != ({HEURISTIC_ACTION_DIM},)"

        # Compute scores: matrix multiply (1024, 6) @ (6,) = (1024,)
        scores = self._last_candidates_obs @ weights  # Element-wise weighted sum

        # Apply mask: set padded positions to 0.0
        scores = scores * self._last_mask

        # Postcondition checks
        assert scores.shape == (MAX_QUEUE_SIZE,), \
            f"Scores shape mismatch: {scores.shape} != ({MAX_QUEUE_SIZE},)"
        assert np.all(scores >= 0), "Scores should be non-negative"
        assert np.all(scores[self._last_mask == 0] == 0), "Masked positions should have score 0"

        return scores.astype(np.float32)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Construct and return the current observation as a dictionary.

        Args:
            None (uses internal state)

        Returns:
            dict with keys:
                - 'global': np.ndarray of shape (OBS_G,) - global environment features
                - 'candidates': np.ndarray of shape (MAX_QUEUE_SIZE, OBS_HOUSE_FEATURES)
                - 'mask': np.ndarray of shape (MAX_QUEUE_SIZE,) - 1.0 for valid, 0.0 for padding

        Requires:
            - _build_candidates_obs() has been called (sets cached attributes)

        Ensures:
            - Returned dict conforms to self.observation_space
            - global features are normalized to [0, 1] or similar reasonable range
            - candidates and mask are directly from cache (no duplication)

        Raises:
            RuntimeError: If _build_candidates_obs() was not called before this
        """
        from config import MAX_QUEUE_SIZE, OBS_G, OBS_HOUSE_FEATURES

        # Build global features (existing logic, keep unchanged)
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

        # Use cached candidates and mask (constructed by _build_candidates_obs)
        if not hasattr(self, '_last_candidates_obs'):
            raise RuntimeError("Must call _build_candidates_obs() before _get_obs()")

        obs_dict = {
            'global': global_features,
            'candidates': self._last_candidates_obs,
            'mask': self._last_mask
        }

        # Postcondition check
        assert obs_dict['global'].shape == (OBS_G,), \
            f"Global shape mismatch: {obs_dict['global'].shape} != ({OBS_G},)"
        assert obs_dict['candidates'].shape == (MAX_QUEUE_SIZE, OBS_HOUSE_FEATURES), \
            f"Candidates shape mismatch: {obs_dict['candidates'].shape} != ({MAX_QUEUE_SIZE}, {OBS_HOUSE_FEATURES})"
        assert obs_dict['mask'].shape == (MAX_QUEUE_SIZE,), \
            f"Mask shape mismatch: {obs_dict['mask'].shape} != ({MAX_QUEUE_SIZE},)"

        return obs_dict

    def reset(self, *, seed: Optional[int] = None, options=None):
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed = seed

        # Reset work arrays
        self._arr_rem = self._arr_total.copy()
        self.last_arr_rem = self._arr_rem.copy()  # Track previous step's remaining work

        # Reset systems
        if isinstance(self.arrival_system, BatchArrival):
            self.arrival_system = BatchArrival(
                self.tasks_df,
                COMBINED_ARRIVAL_CAPACITY_CONFIG["batch_arrival"],
                self.seed if self.seed is not None else 42
            )

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
        if initial_arrivals:
            self.waiting_queue.add(initial_arrivals)

        # Prepare the first day so observations reflect valid candidates
        self._begin_day_if_needed()

        # Build initial candidate snapshot for observation
        self._build_candidates_obs()

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
        Direct score-based allocation: Policy outputs per-candidate scores.

        Algorithm:
        1. Use action directly as per-candidate scores (no learned weights)
        2. Mask invalid candidates with -inf
        3. Use softmax (with temperature) to get allocation distribution
        4. Clip by cmax and remaining work constraints
        5. Redistribute leftover capacity
        6. Round to integers preserving total capacity

        Args:
            candidates: List of house IDs available for allocation
            action: Per-candidate scores from policy. Shape: (ACTION_DIM,)
            K: Total available capacity (contractors)

        Returns:
            Dictionary mapping house_id → num_contractors
        """
        M = len(candidates)

        # Edge case: no candidates or no capacity
        if M == 0 or K <= 0:
            return {}

        # Safety check: verify candidates match cached IDs
        assert M <= len(self._last_candidate_ids), \
            f"Candidates ({M}) exceed cached candidates ({len(self._last_candidate_ids)})"
        assert candidates == self._last_candidate_ids[:M], \
            "Candidates list doesn't match cached candidate IDs"

        # Direct scores from policy action (first M elements correspond to candidates)
        scores = action[:M].astype(np.float64).copy()

        # Mask invalid positions (shouldn't happen if M <= actual candidates, but safety)
        # The policy already masked invalid positions with -1e9, but we ensure here

        # Apply temperature scaling for softmax
        priorities = scores / SOFTMAX_TEMPERATURE

        # Get cmax for each candidate
        cmax_array = np.array([
            self._arr_cmax[candidates[i]]
            for i in range(M)
        ], dtype=np.float64)

        # Remaining work per candidate (prevents over-allocation to nearly-done houses)
        remaining_work_array = np.array([
            self._arr_rem[candidates[i]]
            for i in range(M)
        ], dtype=np.float64)

        # Softmax allocation (ideal distribution)
        # Subtract max for numerical stability
        exp_p = np.exp(priorities - np.max(priorities))
        probs = exp_p / (exp_p.sum() + 1e-10)  # Add epsilon to prevent division by zero
        ideal_allocation = probs * K

        # Clip by cmax AND remaining_work constraints
        allocation = np.minimum(ideal_allocation, np.minimum(cmax_array, remaining_work_array))

        # Redistribute leftover capacity
        used = allocation.sum()
        leftover = K - used

        if leftover > 1e-6:  # If there's significant leftover capacity
            # Find houses that can accept more contractors
            can_take_more = np.minimum(cmax_array, remaining_work_array) - allocation
            can_take_more = np.maximum(can_take_more, 0)

            if can_take_more.sum() > 1e-6:
                # Redistribute proportionally based on original priorities
                eligible_mask = can_take_more > 1e-6
                eligible_probs = probs * eligible_mask

                if eligible_probs.sum() > 1e-6:
                    eligible_probs = eligible_probs / eligible_probs.sum()
                    extra_allocation = eligible_probs * leftover
                    extra_allocation = np.minimum(extra_allocation, can_take_more)
                    allocation += extra_allocation

        # Convert to integers while preserving total capacity
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
                for idx in top_indices:
                    if int_allocation[idx] < cmax_array[idx]:
                        int_allocation[idx] += 1

        # Build result dictionary (only include houses with allocation > 0)
        result = {}
        for i in range(M):
            if int_allocation[i] > 0:
                result[candidates[i]] = int(int_allocation[i])

        # Validation assertions
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
        M_min: int = 1024,
        M_max: int = 1024,
        use_capacity_ramp: bool = False,
        stochastic_duration: bool = True,
        observation_noise: float = 0.15,
        capacity_noise: float = 0.10,
        seed: Optional[int] = None,
        max_steps: int = MAX_STEPS,
        capacity_ceiling: Optional[int] = None,
        use_legacy_capacity_ceiling: bool = False,
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
            use_capacity_ramp=use_capacity_ramp,
            stochastic_duration=stochastic_duration,
            observation_noise=observation_noise,
            capacity_noise=capacity_noise,
            seed=seed,
            max_steps=max_steps,
            capacity_ceiling=capacity_ceiling,
            use_legacy_capacity_ceiling=use_legacy_capacity_ceiling,
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
        M_min: int = 1024,
        M_max: int = 1024,
        use_capacity_ramp: bool = False,
        stochastic_duration: bool = True,
        observation_noise: float = 0.15,
        capacity_noise: float = 0.10,
        seed: Optional[int] = None,
        max_steps: int = MAX_STEPS,
        capacity_ceiling: Optional[int] = None,
        use_legacy_capacity_ceiling: bool = False,
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
            use_capacity_ramp=use_capacity_ramp,
            stochastic_duration=stochastic_duration,
            observation_noise=observation_noise,
            capacity_noise=capacity_noise,
            seed=seed,
            max_steps=max_steps,
            capacity_ceiling=capacity_ceiling,
            use_legacy_capacity_ceiling=use_legacy_capacity_ceiling,
        )

        assert policy in ["LJF", "SJF"], \
            f"Oracle policy must be LJF or SJF, got {policy}"
        self.policy = policy

    def _define_spaces(self):
        """
        Override to use total houses instead of fixed MAX_QUEUE_SIZE.

        Oracle sees the entire queue, so observation space must accommodate
        all houses, not just the M_max candidate limit.
        """
        from config import OBS_G, OBS_HOUSE_FEATURES, HEURISTIC_ACTION_DIM

        total_houses = len(self.tasks_df)

        self.observation_space = spaces.Dict({
            'global': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(OBS_G,),  # 6 global features
                dtype=np.float32
            ),
            'candidates': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_houses, OBS_HOUSE_FEATURES),  # (N, 6) where N = total houses
                dtype=np.float32
            ),
            'mask': spaces.Box(
                low=0,
                high=1,
                shape=(total_houses,),  # (N,)
                dtype=np.float32
            )
        })

        # Action space: same as BaseEnv (learned heuristic weights)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(HEURISTIC_ACTION_DIM,),  # 6 weights
            dtype=np.float32
        )

    def _build_candidates_obs(self) -> None:
        """
        Override to build observation for all houses in queue (no M limit).

        Uses total houses instead of MAX_QUEUE_SIZE for observation arrays.
        Caches results in existing field names: _last_candidates_obs, _last_mask, _last_candidate_ids.
        """
        from config import OBS_HOUSE_FEATURES, FEATURE_SCALES

        total_houses = len(self.tasks_df)

        # Get candidate house IDs (Oracle sees entire queue)
        candidate_ids = self.pending_candidates if self.pending_candidates else []

        # Initialize arrays sized to total houses (not MAX_QUEUE_SIZE)
        candidates = np.zeros((total_houses, OBS_HOUSE_FEATURES), dtype=np.float32)
        mask = np.zeros(total_houses, dtype=np.float32)

        # Fill features for actual candidates
        for i, house_id in enumerate(candidate_ids):
            if i >= total_houses:
                break

            # Extract raw features (same logic as BaseEnv)
            true_remain = self._arr_rem[house_id]
            if self.obs_noise > 0:
                noise = self.rng.normal(0, true_remain * self.obs_noise)
                observed_remain = max(0, true_remain + noise)
            else:
                observed_remain = true_remain

            waiting_time = float(self.waiting_time[house_id])
            total_work = float(self._arr_total[house_id])
            damage_level = float(self._arr_dmg[house_id])
            cmax = float(self._arr_cmax[house_id])

            # Normalize using FEATURE_SCALES from config
            candidates[i] = [
                observed_remain / FEATURE_SCALES[0],
                waiting_time / FEATURE_SCALES[1],
                total_work / FEATURE_SCALES[2],
                damage_level / FEATURE_SCALES[3],
                cmax / FEATURE_SCALES[4],
                1.0  # Bias term
            ]
            mask[i] = 1.0

        # Cache using existing field names
        self._last_candidates_obs = candidates
        self._last_mask = mask
        self._last_candidate_ids = candidate_ids

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Override to return observation with dynamic shape (total_houses instead of MAX_QUEUE_SIZE).
        """
        from config import OBS_G, OBS_HOUSE_FEATURES

        total_houses = len(self.tasks_df)

        # Build global features (same logic as BaseEnv)
        revealed_ids = list(self.arrival_system.revealed_ids)

        if len(revealed_ids) > 0:
            revealed_total = self._arr_total[revealed_ids].sum()
            revealed_remaining = self._arr_rem[revealed_ids].sum()
            completion_rate = 1.0 - (revealed_remaining / revealed_total) if revealed_total > 0 else 0.0
        else:
            revealed_total = 0.0
            revealed_remaining = 0.0
            completion_rate = 0.0

        global_features = np.array([
            float(self.day) / self.max_steps,
            float(self.current_day_capacity) / max(self.num_contractors or 1, 1),
            float(len(self.pending_candidates)) / total_houses,
            float(self.waiting_queue.size()) / total_houses,
            completion_rate,
            float(revealed_remaining) / max(float(self._total_work_all), 1.0),
        ], dtype=np.float32)

        # Verify cache exists
        if not hasattr(self, '_last_candidates_obs') or self._last_candidates_obs is None:
            raise RuntimeError("Must call _build_candidates_obs() before _get_obs()")

        obs_dict = {
            'global': global_features,
            'candidates': self._last_candidates_obs,
            'mask': self._last_mask
        }

        # Postcondition check with dynamic shape
        assert obs_dict['global'].shape == (OBS_G,), \
            f"Global shape mismatch: {obs_dict['global'].shape} != ({OBS_G},)"
        assert obs_dict['candidates'].shape == (total_houses, OBS_HOUSE_FEATURES), \
            f"Candidates shape mismatch: {obs_dict['candidates'].shape} != ({total_houses}, {OBS_HOUSE_FEATURES})"
        assert obs_dict['mask'].shape == (total_houses,), \
            f"Mask shape mismatch: {obs_dict['mask'].shape} != ({total_houses},)"

        return obs_dict

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


class HousegymRLENV(RLEnv):
    """Deprecated compatibility wrapper. Use RLEnv instead."""

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
