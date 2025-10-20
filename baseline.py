from __future__ import annotations

from typing import Literal

import numpy as np

from housegymrl import BaselineEnv

BaselinePolicy = Literal["LJF", "SJF", "RANDOM"]


def make_baseline_allocation(env: BaselineEnv, policy: BaselinePolicy = "LJF") -> np.ndarray:
    """Return integer crew allocations following queue-based heuristics."""
    cand = env.last_candidate_view()
    idx = cand["idx"]
    remain = cand["remain"]
    dmg = cand["dmg"].astype(np.int32, copy=False)
    cmax = cand["cmax"].astype(np.int32, copy=False)
    mask = cand["mask"] > 0.5

    M = idx.shape[0]
    allocation = np.zeros(M, dtype=np.int32)

    valid = mask & (idx >= 0) & (remain > 0)
    if not valid.any():
        return allocation

    vidx = np.where(valid)[0]
    rng = np.random.default_rng(env.seed + env.day)
    policy_upper = policy.upper()

    major = vidx[dmg[vidx] == 2]
    moderate = vidx[dmg[vidx] == 1]
    minor = vidx[dmg[vidx] == 0]

    if major.size > 0:
        rng.shuffle(major)
    if moderate.size > 0:
        rng.shuffle(moderate)
    if minor.size > 0:
        rng.shuffle(minor)

    if policy_upper == "LJF":
        order = np.concatenate([major, moderate, minor])
    elif policy_upper == "SJF":
        order = np.concatenate([minor, moderate, major])
    elif policy_upper == "RANDOM":
        order = rng.permutation(vidx)
    else:
        raise ValueError(f"Unknown baseline policy: {policy}")

    K_eff = env.effective_capacity()
    remaining_capacity = int(K_eff)

    for pos in order:
        if remaining_capacity <= 0:
            break
        give = min(
            remaining_capacity,
            int(cmax[pos]),
            int(np.ceil(remain[pos])),
        )
        if give <= 0:
            continue
        allocation[pos] = give
        remaining_capacity -= give

    assert allocation.sum() <= K_eff
    assert np.all(allocation <= cmax)
    assert np.all(allocation >= 0)
    return allocation
