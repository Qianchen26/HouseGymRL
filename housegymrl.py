from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config import (
    BATCH_ARRIVAL_CONFIG,
    M_CANDIDATES,
    MAX_STEPS,
    OBS_F,
    OBS_G,
)


def allocate_workers(
    scores: np.ndarray,
    K: int,
    cmax: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Vectorized Stage-1 allocator turning fractional scores into integer crews."""
    scores = np.nan_to_num(scores, nan=0.0)
    scores = np.clip(scores, 0.0, 1.0).astype(np.float32, copy=False)
    mask_f = mask.astype(np.float32, copy=False)

    weights = scores * mask_f
    if weights.sum() <= 1e-8:
        weights = mask_f
    total = weights.sum()
    if total > 0:
        weights = weights / total

    expected = weights * float(K)
    base = np.floor(expected)
    alloc = np.minimum(base, cmax).astype(np.int32, copy=False)

    rem = int(K - int(alloc.sum()))
    if rem > 0:
        fractional = (expected - base).astype(np.float32, copy=False)
        blocked = (mask_f <= 0.0) | (alloc >= cmax)
        fractional = np.where(blocked, -np.inf, fractional)
        valid_cnt = int(np.sum(~blocked))
        if valid_cnt > 0:
            take = min(rem, valid_cnt)
            top_idx = np.argpartition(-fractional, take - 1)[:take]
            alloc[top_idx] += 1
            rem -= take

    idle = max(rem, 0)
    assert alloc.sum() <= K
    return alloc, idle


class BatchArrivalScheduler:
    """Deterministic three-batch arrival scheduler."""

    def __init__(self, total_houses: int, seed: Optional[int]) -> None:
        self.total = int(total_houses)
        rng = np.random.default_rng(seed)
        ids = np.arange(self.total, dtype=np.int32)
        rng.shuffle(ids)

        days = BATCH_ARRIVAL_CONFIG["days"]
        ratios = BATCH_ARRIVAL_CONFIG["ratios"]
        if len(days) != 3 or len(ratios) != 3:
            raise ValueError("Expected exactly three batch days and ratios.")

        r0, r1, r2 = ratios
        n0 = int(r0 * self.total)
        n1 = int(r1 * self.total)
        n2 = self.total - n0 - n1
        n2 = max(0, n2)
        if n0 < 0 or n1 < 0 or n2 < 0:
            raise ValueError("Invalid batch sizes derived from ratios.")

        self.schedule: Dict[int, np.ndarray] = {
            int(days[0]): ids[:n0].copy(),
            int(days[1]): ids[n0:n0 + n1].copy(),
            int(days[2]): ids[n0 + n1:].copy(),
        }

    def get_arrivals(self, day: int) -> np.ndarray:
        return self.schedule.get(int(day), np.array([], dtype=np.int32))

    def verify_total(self) -> bool:
        return sum(len(v) for v in self.schedule.values()) == self.total


class BaseEnv(gym.Env):
    """Shared environment mechanics for both baseline and RL policies."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        tasks_df: Optional[pd.DataFrame] = None,
        resources: Optional[Dict] = None,
        scenario_sampler: Optional[Callable[[], Tuple[pd.DataFrame, Dict, Dict]]] = None,
        *,
        M: int = M_CANDIDATES,
        max_steps: int = MAX_STEPS,
        seed: Optional[int] = None,
        batch_arrival: bool = True,
        fill_non_candidates: bool = True,
        k_ramp: Optional[Callable[[int], float]] = None,
        candidate_policy: str = "random",
    ) -> None:
        super().__init__()
        self.M = int(M)
        self.max_steps = int(max_steps)
        self.scenario_sampler = scenario_sampler
        self.seed = int(seed) if seed is not None else 0
        self.rng = np.random.default_rng(seed)

        self.batch_arrival = bool(batch_arrival)
        self.fill_non_candidates = bool(fill_non_candidates)
        self.k_ramp = k_ramp
        policy = str(candidate_policy).lower()
        if policy == "uniform":
            policy = "random"
        allowed_policies = {"random", "backlog", "queue_ljf", "queue_sjf"}
        if policy not in allowed_policies:
            raise ValueError(
                f"Unsupported candidate_policy='{candidate_policy}'. Expected one of {sorted(allowed_policies)}."
            )
        self.candidate_policy = policy

        self.writer = None
        self.tb_every = 0
        self._global_step = 0

        self._last_candidates: Optional[Dict[str, np.ndarray]] = None
        self._last_info: Optional[Dict[str, float]] = None

        self.tasks_df = None
        self._arr_total = None
        self._arr_rem = None
        self._arr_cmax = None
        self._arr_dmg = None
        self._arr_delay = None
        self._initial_delay = None
        self._arrived_mask = None
        self._eligible_mask = None
        self._initial_rem = None
        self._total_work_scalar = 0.0

        self.scheduler: Optional[BatchArrivalScheduler] = None
        self.K = 0
        self.region_name = "UNKNOWN"
        self.day = 0
        self.done_flag = False
        self.idle_history: list[int] = []

        if tasks_df is None or resources is None:
            if scenario_sampler is None:
                raise ValueError("Provide (tasks_df, resources) or a scenario_sampler().")
            tasks_df, resources, _ = scenario_sampler()

        self._load_scenario(tasks_df, resources)

        obs_dim = OBS_G + self.M * OBS_F
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.M,), dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    # Core mechanics
    # ------------------------------------------------------------------ #
    def _load_scenario(self, tasks_df: pd.DataFrame, resources: Dict) -> None:
        df = tasks_df.copy().reset_index(drop=True)
        required = [
            "man_days_total",
            "man_days_remaining",
            "delay_days_remaining",
            "cmax_per_day",
            "damage_level",
        ]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        if "house_id" not in df.columns:
            df["house_id"] = np.arange(len(df), dtype=np.int32)

        df["delay_days_remaining"] = 0

        self.tasks_df = df
        self.K = int(resources["workers"])
        self.region_name = str(resources.get("region_name", "UNKNOWN"))

        self._arr_total = df["man_days_total"].to_numpy(dtype=np.int32, copy=True)
        self._arr_rem = df["man_days_remaining"].to_numpy(dtype=np.int32, copy=True)
        self._initial_rem = self._arr_rem.copy()
        self._arr_cmax = df["cmax_per_day"].to_numpy(dtype=np.int32, copy=True)
        self._arr_dmg = df["damage_level"].to_numpy(dtype=np.int32, copy=True)
        self._arr_delay = df["delay_days_remaining"].to_numpy(dtype=np.int32, copy=True)
        self._initial_delay = self._arr_delay.copy()

        self._arrived_mask = np.zeros(len(df), dtype=bool)
        self._eligible_mask = np.zeros(len(df), dtype=bool)
        self._total_work_scalar = float(self._arr_total.sum())

        if self.batch_arrival:
            self.scheduler = BatchArrivalScheduler(len(df), self.seed)
        else:
            self.scheduler = None

        if self.scheduler is not None and not self.scheduler.verify_total():
            raise ValueError("BatchArrivalScheduler mismatch with total households.")

        self.day = 0
        self.done_flag = False
        self.idle_history.clear()

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed = int(seed)

        if self.scenario_sampler is not None:
            tasks_df, resources, _ = self.scenario_sampler()
            self._load_scenario(tasks_df, resources)
        else:
            self._arr_rem = self._initial_rem.copy()
            self._arr_delay = self._initial_delay.copy()
            self._arrived_mask.fill(False)
            self._eligible_mask.fill(False)
            self.day = 0
            self.done_flag = False
            self.idle_history.clear()

        if not self.batch_arrival or self.scheduler is None:
            self._arrived_mask.fill(True)
        else:
            self._arrived_mask.fill(False)
            self._apply_arrivals(0)

        self._update_eligibility()
        self._refresh_candidates()
        self._last_info = None

        self._global_step = 0
        return self._get_obs(), {}

    def _apply_arrivals(self, day: int) -> None:
        if not self.batch_arrival or self.scheduler is None:
            self._arrived_mask.fill(True)
            return
        arrivals = self.scheduler.get_arrivals(day)
        if arrivals.size > 0:
            self._arrived_mask[arrivals] = True

    def _update_eligibility(self) -> None:
        np.logical_and(self._arrived_mask, self._arr_rem > 0, out=self._eligible_mask)

    def _effective_capacity(self, day_override: Optional[int] = None) -> int:
        base = int(self.K)
        day = self.day if day_override is None else int(day_override)
        if self.k_ramp is not None:
            try:
                ratio = float(self.k_ramp(day))
            except Exception:
                ratio = 1.0
            ratio = max(0.0, min(1.0, ratio))
            base = int(round(ratio * self.K))
        return max(0, min(int(self.K), base))

    def _sanitize_candidate_allocation(
        self, allocation: np.ndarray, cand_snapshot: Dict[str, np.ndarray], K_eff: int
    ) -> np.ndarray:
        alloc = np.zeros(self.M, dtype=np.int32)
        flat = np.asarray(allocation, dtype=np.int32).reshape(-1)
        take = min(flat.size, self.M)
        alloc[:take] = flat[:take]

        idx = cand_snapshot["idx"]
        mask = cand_snapshot["mask"] > 0.5
        valid = idx >= 0
        active = mask & valid

        alloc = np.where(active, alloc, 0)

        cmax = cand_snapshot["cmax"].astype(np.int32, copy=False)
        alloc = np.minimum(alloc, cmax)

        remain_cap = np.zeros(self.M, dtype=np.int32)
        valid_idx = idx[active]
        if valid_idx.size > 0:
            remain_cap[active] = self._arr_rem[valid_idx].astype(np.int32, copy=False)
        alloc = np.minimum(alloc, remain_cap)

        total = int(alloc.sum())
        if total > K_eff:
            over = total - K_eff
            order = np.argsort(-alloc)
            for pos in order:
                if over <= 0:
                    break
                if alloc[pos] <= 0:
                    continue
                drop = min(over, alloc[pos])
                alloc[pos] -= drop
                over -= drop

        return alloc

    def _apply_candidate_allocation(
        self, allocation: np.ndarray, cand_snapshot: Dict[str, np.ndarray]
    ) -> int:
        used = 0
        idx = cand_snapshot["idx"]
        for pos, row in enumerate(idx):
            give = int(allocation[pos])
            if give <= 0 or row < 0:
                continue
            actual = min(give, int(self._arr_rem[row]))
            if actual <= 0:
                continue
            self._arr_rem[row] -= actual
            used += actual
        return used

    def _allocate_stage_two(
        self, remaining_capacity: int, cand_snapshot: Dict[str, np.ndarray]
    ) -> int:
        if not self.fill_non_candidates or remaining_capacity <= 0:
            return 0

        extra_mask = self._eligible_mask.copy()
        in_cand = cand_snapshot["idx"]
        valid_in_cand = in_cand[in_cand >= 0]
        if valid_in_cand.size > 0:
            extra_mask[valid_in_cand] = False

        extra_idx = np.nonzero(extra_mask)[0]
        if extra_idx.size == 0:
            return 0

        rem = self._arr_rem[extra_idx]
        order = np.argsort(rem, kind="stable")

        used = 0
        for idx in extra_idx[order]:
            if remaining_capacity <= 0:
                break
            give = min(
                remaining_capacity,
                int(self._arr_cmax[idx]),
                int(self._arr_rem[idx]),
            )
            if give <= 0:
                continue
            self._arr_rem[idx] -= give
            remaining_capacity -= give
            used += give
        return used

    def _compute_completion(self) -> Tuple[float, float]:
        total_work = self._total_work_scalar
        remain_work = float(self._arr_rem.sum())
        completion_md = 1.0 - (remain_work / (total_work + 1e-8))

        completed = int(np.sum(self._arr_rem <= 0))
        total = max(1, self._arr_rem.size)
        completion_hh = completed / total
        return completion_md, completion_hh

    def _build_info(
        self,
        *,
        K_eff: int,
        allocated: int,
        idle: int,
        completion_hh: float,
        num_candidates: int,
        num_eligible: int,
        day: int,
        unfinished_houses: int,
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, float]:
        completion_hh = float(np.clip(completion_hh, 0.0, 1.0))
        info: Dict[str, float] = {
            "completion": float(completion_hh),
            "completion_hh": float(completion_hh),
            "allocated_workers": int(allocated),
            "idle_workers": int(idle),
            "num_candidates": int(num_candidates),
            "num_eligible": int(num_eligible),
            "day": int(day),
            "K_effective": int(K_eff),
            "unfinished_houses": int(unfinished_houses),
            "terminated": bool(terminated),
            "done": bool(terminated),
            "truncated": bool(truncated),
        }
        assert info["allocated_workers"] + info["idle_workers"] == info["K_effective"]
        completion_val = info["completion"]
        assert 0.0 - 1e-6 <= completion_val <= 1.0 + 1e-6, "Completion out of [0,1] bounds."
        return info

    def _maybe_log(self, info: Dict[str, float], completion_md: float) -> None:
        if self.writer is None or self.tb_every <= 0:
            return
        self._global_step += 1
        if self._global_step % self.tb_every != 0:
            return
        self.writer.add_scalar("env/day", info["day"], self._global_step)
        self.writer.add_scalar("env/completion_household", info["completion_hh"], self._global_step)
        self.writer.add_scalar("env/completion_man_days", completion_md, self._global_step)
        self.writer.add_scalar("env/idle_workers", info["idle_workers"], self._global_step)

    def _refresh_candidates(self) -> None:
        M = self.M
        eligible_mask = self._arrived_mask & (self._arr_rem > 0)
        eligible = np.nonzero(eligible_mask)[0]

        if eligible.size == 0:
            idx = np.full(M, -1, dtype=np.int32)
            self._emit_candidates(idx)
            return

        if eligible.size <= M:
            idx = np.full(M, -1, dtype=np.int32)
            idx[: eligible.size] = eligible
            self._emit_candidates(idx)
            return

        policy = self.candidate_policy
        if policy == "queue_ljf":
            major = eligible[self._arr_dmg[eligible] == 2]
            if major.size > 0:
                major = major[np.argsort(-self._arr_rem[major], kind="mergesort")]
            others = eligible[self._arr_dmg[eligible] != 2]
            if others.size > 0:
                others = self.rng.permutation(others)
            order = np.concatenate([major, others])
        elif policy == "queue_sjf":
            minor = eligible[self._arr_dmg[eligible] == 0]
            if minor.size > 0:
                minor = minor[np.argsort(self._arr_rem[minor], kind="mergesort")]
            others = eligible[self._arr_dmg[eligible] != 0]
            if others.size > 0:
                others = self.rng.permutation(others)
            order = np.concatenate([minor, others])
        elif policy == "backlog":
            order = eligible[np.argsort(-self._arr_rem[eligible], kind="mergesort")]
        else:  # "random"
            order = self.rng.permutation(eligible)

        order = order[:M]
        if order.size < M:
            idx = np.full(M, -1, dtype=np.int32)
            idx[: order.size] = order
        else:
            idx = order.astype(np.int32, copy=False)

        self._emit_candidates(idx)

    def _emit_candidates(self, idx: np.ndarray) -> None:
        M = self.M
        if idx.shape[0] != M:
            raise ValueError(f"Candidate idx length {idx.shape[0]} does not match M={M}.")

        remain = np.zeros(M, dtype=np.float32)
        delay = np.zeros(M, dtype=np.float32)
        dmg = np.zeros(M, dtype=np.float32)
        cmax = np.zeros(M, dtype=np.float32)

        valid_rows = idx >= 0
        if valid_rows.any():
            sel = idx[valid_rows]
            remain[valid_rows] = self._arr_rem[sel].astype(np.float32, copy=False)
            delay[valid_rows] = self._arr_delay[sel].astype(np.float32, copy=False)
            dmg[valid_rows] = self._arr_dmg[sel].astype(np.float32, copy=False)
            cmax[valid_rows] = self._arr_cmax[sel].astype(np.float32, copy=False)

        mask = ((remain > 0) & valid_rows).astype(np.float32, copy=False)

        self._last_candidates = {
            "idx": idx.astype(np.int32, copy=False),
            "remain": remain,
            "delay": delay,
            "dmg": dmg,
            "cmax": cmax,
            "mask": mask,
        }

    def _get_obs(self) -> np.ndarray:
        total_work = self._total_work_scalar
        remain_sum = float(self._arr_rem.sum())
        backlog_cnt = int(np.sum(self._arr_rem > 0))
        lvl1 = float(self._arr_rem[self._arr_dmg == 1].sum())
        lvl2 = float(self._arr_rem[self._arr_dmg == 2].sum())

        g = np.array(
            [
                self.day / max(1.0, float(self.max_steps)),
                float(self.K),
                float(backlog_cnt),
                remain_sum / (total_work + 1e-8),
                lvl1 / (total_work + 1e-8),
                lvl2 / (total_work + 1e-8),
            ],
            dtype=np.float32,
        )

        cand = self._last_candidates
        assert cand is not None, "Candidates not initialized. Call reset() first."
        stacked = np.stack(
            [cand["remain"], cand["delay"], cand["dmg"], cand["cmax"]], axis=1
        ).astype(np.float32, copy=False)
        return np.concatenate([g, stacked.reshape(-1)], axis=0).astype(np.float32, copy=False)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def step(self, action):
        if action is None:
            raise ValueError("Action must not be None.")

        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            arr = np.zeros(self.M, dtype=np.float32)
        if arr.size < self.M:
            padded = np.zeros(self.M, dtype=np.float32)
            padded[: arr.size] = arr
            arr = padded
        elif arr.size > self.M:
            arr = arr[: self.M]

        if not np.all(np.isfinite(arr)):
            raise ValueError("Environment received non-finite action values.")

        cand = self._last_candidates
        assert cand is not None, "Call reset() before stepping the environment."

        is_int_alloc = np.all(arr >= 0.0) and np.allclose(arr, np.round(arr))
        if is_int_alloc:
            allocation = np.round(arr).astype(np.int32, copy=False)
            return self._step_with_allocation(allocation)

        scores = np.clip(arr.astype(np.float32, copy=False), 0.0, 1.0)
        K_eff = self._effective_capacity()
        allocation, _ = allocate_workers(scores, K_eff, cand["cmax"], cand["mask"])
        return self._step_with_allocation(allocation, K_override=K_eff)

    def _step_with_allocation(
        self,
        allocation: np.ndarray,
        *,
        K_override: Optional[int] = None,
    ):
        if self.done_flag:
            obs = self._get_obs()
            assert self._last_info is not None, "Done flag set but last info missing."
            info = self._last_info
            terminated_flag = bool(
                info.get("terminated", info.get("done", True))
            )
            truncated_flag = bool(info.get("truncated", False))
            return obs, 0.0, terminated_flag, truncated_flag, info

        cand_snapshot = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self._last_candidates.items()}  # type: ignore[arg-type]
        K_eff = int(K_override) if K_override is not None else self._effective_capacity()
        num_candidates = int(np.sum(cand_snapshot["mask"] > 0))

        self._update_eligibility()
        num_eligible_before = int(self._eligible_mask.sum())

        completion_md_before, completion_hh_before = self._compute_completion()

        sanitized = self._sanitize_candidate_allocation(allocation, cand_snapshot, K_eff)
        used_stage1 = self._apply_candidate_allocation(sanitized, cand_snapshot)
        self._update_eligibility()

        rem_capacity = max(0, K_eff - used_stage1)
        used_stage2 = self._allocate_stage_two(rem_capacity, cand_snapshot)
        total_alloc = used_stage1 + used_stage2
        idle = max(0, K_eff - total_alloc)

        completion_md_after, completion_hh_after = self._compute_completion()
        completion_md_after = float(np.clip(completion_md_after, 0.0, 1.0))
        completion_hh_after = float(np.clip(completion_hh_after, 0.0, 1.0))
        reward = completion_hh_after - completion_hh_before

        current_day = self.day
        unfinished_houses = int(np.sum(self._arr_rem > 0))
        terminated = unfinished_houses == 0
        next_day = self.day + 1
        truncated = (next_day >= self.max_steps) and not terminated

        self.day = next_day
        self.done_flag = terminated or truncated

        if not self.done_flag:
            self._apply_arrivals(self.day)
            self._update_eligibility()
            self._refresh_candidates()
        else:
            self._update_eligibility()

        info = self._build_info(
            K_eff=K_eff,
            allocated=total_alloc,
            idle=idle,
            completion_hh=completion_hh_after,
            num_candidates=num_candidates,
            num_eligible=num_eligible_before,
            day=current_day,
            unfinished_houses=unfinished_houses,
            terminated=terminated,
            truncated=truncated,
        )

        self.idle_history.append(idle)
        self._last_info = info
        self._maybe_log(info, completion_md_after)

        obs = self._get_obs()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def last_candidate_view(self) -> Dict[str, np.ndarray]:
        assert self._last_candidates is not None, "Call reset() before accessing candidates."
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self._last_candidates.items()
        }

    def set_summary_writer(self, writer, *, tb_every: int = 200) -> None:
        self.writer = writer
        self.tb_every = int(tb_every)
        self._global_step = 0

    def effective_capacity(self) -> int:
        return self._effective_capacity()


class BaselineEnv(BaseEnv):
    """Environment expecting integer allocations (baseline dispatcher)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        high = np.full(self.M, self.K, dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=high, dtype=np.float32)

    def step(self, action: np.ndarray):
        arr = np.asarray(action)
        if not np.all(np.isfinite(arr)):
            raise ValueError("BaselineEnv received non-finite allocations.")
        if not np.allclose(arr, np.round(arr)):
            raise ValueError("BaselineEnv expects integer allocations per candidate.")
        return super().step(arr.astype(np.int32, copy=False))


class RLEnv(BaseEnv):
    """Environment converting fractional scores into allocations."""

    pass
