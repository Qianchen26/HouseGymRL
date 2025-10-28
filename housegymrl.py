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
from config import (
    M_CANDIDATES,
    MAX_STEPS,
    COMBINED_ARRIVAL_CAPACITY_CONFIG,
    REGION_CONFIG,
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

        # Group houses by damage level
        major = tasks_df[tasks_df['damage_level'] == 2].index.tolist()
        moderate = tasks_df[tasks_df['damage_level'] == 1].index.tolist()
        minor = tasks_df[tasks_df['damage_level'] == 0].index.tolist()

        # Shuffle within each group (assessment order is random within priority)
        self.rng.shuffle(major)
        self.rng.shuffle(moderate)
        self.rng.shuffle(minor)

        # Assign to batches based on damage priority
        days = config["days"]
        ratios = config["ratios"]

        # Batch 1: Mostly major damage (emergency response)
        n_major_1 = int(len(major) * 0.7)
        n_moderate_1 = int(len(moderate) * 0.1)
        batch1 = major[:n_major_1] + moderate[:n_moderate_1]

        # Batch 2: Remaining major + most moderate
        n_major_2 = len(major) - n_major_1
        n_moderate_2 = int(len(moderate) * 0.6)
        batch2 = major[n_major_1:] + moderate[n_moderate_1:n_moderate_1+n_moderate_2]

        # Batch 3: Remaining moderate + all minor
        batch3 = moderate[n_moderate_1+n_moderate_2:] + minor

        # Ensure all houses are assigned
        all_houses = set(batch1 + batch2 + batch3)
        missing = set(tasks_df.index.tolist()) - all_houses
        if missing:
            batch3.extend(list(missing))

        self.schedule = {
            days[0]: batch1,
            days[1]: batch2,
            days[2]: batch3,
        }

    def step_to_day(self, day: int) -> List[int]:
        """
        Advance to given day and return newly revealed houses.

        Args:
            day: Current simulation day

        Returns:
            List of house IDs that became visible today
        """
        new_arrivals = []

        # Check all scheduled days up to current day
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


class StaticArrival:
    """
    Static arrival: All houses known from day 0.
    Used as control/baseline for experiments.
    """

    def __init__(self, tasks_df: pd.DataFrame):
        self.revealed_ids = set(tasks_df.index.tolist())
        self.current_day = 0

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
    - Capacity ramp (resources gradually available)
    - Pure random candidate selection (no artificial bias)
    - Fair constraints for all strategies (RL/baseline)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        region_key: str,
        num_contractors: Optional[int] = None,
        use_batch_arrival: bool = True,
        use_capacity_ramp: bool = True,
        seed: Optional[int] = None,
        max_steps: int = MAX_STEPS,
    ):
        """
        Initialize base environment.

        Args:
            region_key: Region name from REGION_CONFIG
            num_contractors: Number of contractors (if None, use region default)
            use_batch_arrival: Whether to use batch arrival system
            use_capacity_ramp: Whether to use capacity ramp system
            seed: Random seed
            max_steps: Maximum simulation days
        """
        super().__init__()

        # Set random generator
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        # Load region configuration
        if region_key not in REGION_CONFIG:
            raise ValueError(f"Unknown region: {region_key}")

        self.region_key = region_key
        self.region_config = REGION_CONFIG[region_key]

        # Set number of contractors
        if num_contractors is None:
            num_contractors = self.region_config["num_contractors"]
        self.num_contractors = num_contractors

        # Set M and max_steps
        self.M = M_CANDIDATES
        self.max_steps = max_steps

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

        if use_capacity_ramp:
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

    def _define_spaces(self):
        """Define observation and action spaces."""
        # Observation: 6 global features + M*4 candidate features
        obs_dim = 6 + self.M * 4
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action: M continuous scores [0,1]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.M,),
            dtype=np.float32
        )

    def _effective_capacity(self) -> int:
        """Get current effective capacity based on ramp."""
        return self.capacity_system.get_capacity(self.day)

    def _select_candidates(self, queue_ids: List[int]) -> List[int]:
        """
        PURE RANDOM candidate selection with DETERMINISTIC seeding.

        This is the key innovation: No artificial bias.
        RL must learn from natural random samples.

        CRITICAL: Uses day-based seed to ensure all methods see
        the same candidates on the same day (Information Parity).

        Args:
            queue_ids: All houses in queue

        Returns:
            Randomly selected M candidates (deterministic for given day)
        """
        if len(queue_ids) <= self.M:
            return queue_ids

        # CRITICAL: Use day-based seed for candidate selection
        # This ensures all methods see the same candidates on the same day
        # regardless of their internal RNG state
        selection_seed = (self.seed if self.seed is not None else 42) + self.day * 1000
        selection_rng = np.random.default_rng(selection_seed)

        # Pure random selection - no bias, no pre-filtering
        selected = selection_rng.choice(queue_ids, size=self.M, replace=False)
        return selected.tolist()

    def step(self, action):
        """
        Execute one day of reconstruction.

        Steps:
        1. Check for new arrivals (batch system)
        2. Get current capacity (ramp system)
        3. Select candidates (pure random)
        4. Allocate workers (subclass-specific)
        5. Update state and calculate reward
        """
        # Step 1: Check new arrivals
        new_arrivals = self.arrival_system.step_to_day(self.day)
        if new_arrivals:
            self.waiting_queue.add(new_arrivals)

        # Step 2: Get current capacity
        K_available = self._effective_capacity()

        # Step 3: Handle zero capacity (planning phase)
        if K_available == 0:
            self.day += 1
            obs = self._get_obs()
            reward = 0.0
            terminated = False
            truncated = self.day >= self.max_steps
            info = {
                "day": self.day,
                "K": 0,
                "queue_size": self.waiting_queue.size(),
                "completion": self.last_completion,
            }
            return obs, reward, terminated, truncated, info

        # Step 4: Allocation logic
        queue_ids = self.waiting_queue.get_all()

        if len(queue_ids) == 0:
            # Empty queue
            allocation = {}
        else:
            # Select candidates and allocate
            candidates = self._select_candidates(queue_ids)
            allocation = self._allocate_from_candidates(candidates, action, K_available)

        # Step 5: Execute allocation
        return self._execute_allocation(allocation)

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

    def _execute_allocation(self, allocation: Dict[int, int]):
        """
        Execute allocation and update state.

        Args:
            allocation: Dictionary mapping house_id to workers

        Returns:
            Standard gym step outputs
        """
        # Apply work
        for house_id, workers in allocation.items():
            self._arr_rem[house_id] -= workers
            self._arr_rem[house_id] = max(0, self._arr_rem[house_id])

        # Remove completed houses
        completed = [
            h for h in self.waiting_queue.get_all()
            if self._arr_rem[h] <= 0
        ]
        self.waiting_queue.remove(completed)

        # Advance day
        self.day += 1

        # Calculate completion (only for revealed houses)
        revealed_ids = list(self.arrival_system.revealed_ids)
        if len(revealed_ids) > 0:
            # Count completed houses
            revealed_completed = sum(1 for h in revealed_ids if self._arr_rem[h] <= 0)
            completion = revealed_completed / len(revealed_ids)
        else:
            completion = 0.0

        # Reward = completion delta
        reward = completion - self.last_completion
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
            "day": self.day,
            "completion": completion,
            "queue_size": self.waiting_queue.size(),
            "K": self._effective_capacity(),
            "revealed_count": self.arrival_system.get_revealed_count(),
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        Build observation vector.

        Structure:
        - 6 global features
        - M*4 candidate features
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
            float(self._effective_capacity()),
            float(self.waiting_queue.size()),
            float(len(revealed_ids)),
            remain_ratio,
            major_ratio,
        ], dtype=np.float32)

        # Candidate features
        queue_ids = self.waiting_queue.get_all()
        candidates = self._select_candidates(queue_ids) if queue_ids else []

        candidate_features = np.zeros((self.M, 4), dtype=np.float32)
        for i, house_id in enumerate(candidates[:self.M]):
            candidate_features[i] = [
                self._arr_rem[house_id],
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

        # Initial batch arrival
        initial_arrivals = self.arrival_system.step_to_day(0)
        if initial_arrivals:
            self.waiting_queue.add(initial_arrivals)

        obs = self._get_obs()
        info = {"day": 0}

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
        Allocate based on RL-predicted scores.

        Args:
            candidates: Selected M candidates
            action: Continuous scores [0,1]^M
            K: Available capacity

        Returns:
            Allocation dictionary
        """
        # Extract valid scores
        valid_scores = action[:len(candidates)]

        # Sort by score (descending)
        sorted_indices = np.argsort(-valid_scores)
        sorted_candidates = [candidates[i] for i in sorted_indices]

        # Greedy allocation
        allocation = {}
        remaining_K = K

        for house_id in sorted_candidates:
            if remaining_K <= 0:
                break

            cmax = int(self._arr_cmax[house_id])
            remain = int(self._arr_rem[house_id])
            give = min(cmax, remain, remaining_K)

            if give > 0:
                allocation[house_id] = give
                remaining_K -= give

        return allocation


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
        use_batch_arrival: bool = True,
        use_capacity_ramp: bool = True,
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
            use_batch_arrival=use_batch_arrival,
            use_capacity_ramp=use_capacity_ramp,
            seed=seed,
            max_steps=max_steps,
        )

        assert policy in ["LJF", "SJF", "Random"], \
            f"Policy must be LJF, SJF, or Random, got {policy}"
        self.policy = policy

    def step(self, action=None):
        """Baseline doesn't use action input."""
        return super().step(action=None)

    def _allocate_from_candidates(
        self,
        candidates: List[int],
        action: Any,
        K: int
    ) -> Dict[int, int]:
        """
        Allocate using fixed priority rule.

        Args:
            candidates: Selected M candidates
            action: Not used
            K: Available capacity

        Returns:
            Allocation dictionary
        """
        if self.policy == "LJF":
            # Longest Job First: Sort by total work (descending)
            sorted_candidates = sorted(
                candidates,
                key=lambda h: -self._arr_total[h]
            )

        elif self.policy == "SJF":
            # Shortest Job First: Sort by total work (ascending)
            sorted_candidates = sorted(
                candidates,
                key=lambda h: self._arr_total[h]
            )

        elif self.policy == "Random":
            # Random order
            sorted_candidates = candidates.copy()
            self.rng.shuffle(sorted_candidates)

        # Greedy allocation
        allocation = {}
        remaining_K = K

        for house_id in sorted_candidates:
            if remaining_K <= 0:
                break

            cmax = int(self._arr_cmax[house_id])
            remain = int(self._arr_rem[house_id])
            give = min(cmax, remain, remaining_K)

            if give > 0:
                allocation[house_id] = give
                remaining_K -= give

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
        use_batch_arrival: bool = True,
        use_capacity_ramp: bool = True,
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
            use_batch_arrival=use_batch_arrival,
            use_capacity_ramp=use_capacity_ramp,
            seed=seed,
            max_steps=max_steps,
        )

        assert policy in ["LJF", "SJF"], \
            f"Oracle policy must be LJF or SJF, got {policy}"
        self.policy = policy

    def _select_candidates(self, queue_ids: List[int]) -> List[int]:
        """Oracle sees entire queue (no M limit)."""
        return queue_ids

    def _allocate_from_candidates(
        self,
        candidates: List[int],
        action: Any,
        K: int
    ) -> Dict[int, int]:
        """
        Allocate using policy on entire queue.

        Args:
            candidates: All houses in queue
            action: Not used
            K: Available capacity

        Returns:
            Allocation dictionary
        """
        if self.policy == "LJF":
            sorted_all = sorted(candidates, key=lambda h: -self._arr_total[h])
        elif self.policy == "SJF":
            sorted_all = sorted(candidates, key=lambda h: self._arr_total[h])
        else:
            sorted_all = candidates

        # Greedy allocation
        allocation = {}
        remaining_K = K

        for h in sorted_all:
            if remaining_K <= 0:
                break

            give = min(
                self._arr_cmax[h],
                self._arr_rem[h],
                remaining_K
            )

            if give > 0:
                allocation[h] = give
                remaining_K -= give

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
        queue_ids = self.waiting_queue.get_all()
        candidates = self._select_candidates(queue_ids) if queue_ids else []

        remain = np.zeros(self.M, dtype=np.float32)
        mask = np.zeros(self.M, dtype=np.float32)

        for i, h in enumerate(candidates[:self.M]):
            remain[i] = self._arr_rem[h]
            mask[i] = 1.0

        return {"remain": remain, "mask": mask}