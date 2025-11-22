# Daily Batch Redesign - Baseline Documentation

**Date**: 2025-11-18
**Purpose**: Document current state before redesigning from intra-day to daily batch decisions

## Current Implementation (Intra-day Decisions)

### Episode Length Calculation

For average synthetic scenario:
- Total houses (H): 34,252
- Total contractors (K): 4,797
- Average steps per episode: **~1,800,000 steps**

Breakdown:
```
With capacity ramp disabled (current training setup):
- Days needed: ~500 days
- Average allocations per day: ~4,797 (full capacity)
- But each allocation is ONE step
- Steps per day: ~3,600 (varies by queue size and cmax constraints)
- Total steps: ~500 days × 3,600 steps/day = ~1.8M steps
```

### Current Step Mechanism

Each `step()` call:
1. Samples M candidates from queue (once per day)
2. Agent selects ONE house from candidates
3. Allocates min(cmax, remaining_work, capacity_remaining) contractors
4. Removes house from today's candidates
5. Either:
   - Continues in same day (if capacity > 0 and candidates remain) → reward = 0
   - Advances to next day (if capacity exhausted or no candidates) → reward calculated

### Training Infeasibility

With 5M timesteps budget:
- Episodes completed: 5,000,000 / 1,800,000 = **2.78 episodes**
- TensorBoard shows: Only `train/*` metrics, no `rollout/*` episode metrics
- Learning challenge: Credit assignment with γ=0.99 over 1.8M steps is effectively impossible

### Current Observation Space
- Shape: (6 + M_max*4,) = (6 + 2048*4,) = (8198,)
- Global features: 6
- Candidate features: M_max × 4 (padded with zeros)

### Current Action Space
- Shape: (M_max + 1,) = (2049,)
- Candidate scores: M_max = 2048
- No-op score: 1

---

## Target Implementation (Daily Batch Decisions)

### Target Episode Length
- Steps per episode: **~500 steps** (one step per day)
- With 5M timesteps: 5,000,000 / 500 = **10,000 episodes**
- Training time estimate: 2-3 hours (vs. never completing)

### Target Step Mechanism

Each `step()` call:
1. Begin new day: sample M candidates from queue
2. Agent outputs priority scores for all M candidates
3. **Hybrid weighted allocation**:
   - Softmax scores → ideal allocation
   - Clip by cmax constraints
   - Redistribute leftover capacity
   - Convert to integer allocation
4. Execute full day's allocation
5. Advance to next day → reward calculated
6. Return next day's observation

### Observation/Action Spaces
- **No changes needed** - current spaces already support batch decisions
- Action vector already represents priority scores for all candidates

---

## Key Design Principles Preserved

1. **RL Learning Value**: Agent still learns context-dependent priority scoring
2. **Cmax Constraints**: Hybrid allocation respects per-house daily limits
3. **Partial Observability**: M-sampling still creates information scarcity
4. **Batch Arrival**: Houses still revealed over time
5. **Stochasticity**: Duration noise and capacity noise preserved

---

## Validation Criteria

Implementation succeeds if:
- [ ] Episode completes in ~500 steps (not 1.8M)
- [ ] TensorBoard shows `rollout/ep_rew_mean` after first episode
- [ ] Hybrid allocation respects cmax constraints (verified by unit test)
- [ ] Sum of allocations ≤ K (verified by assertion)
- [ ] Training shows learning signal (rewards improve over episodes)
- [ ] Baselines (LJF/SJF/Random) complete in ~500 steps
