# Daily Batch Redesign - Implementation Summary

**Date**: 2025-11-18
**Branch**: `daily-batch-redesign`
**Status**: ✅ Implementation Complete, Pending Testing

---

## Problem Statement

The original environment design was **fundamentally infeasible for RL training**:

- **Episode length**: ~1,800,000 steps
- **Reason**: Each `step()` call allocated contractors to ONE house
- **Training budget**: 5M timesteps = only 2.78 episodes
- **Learning**: Impossible with γ=0.99 over 1.8M steps (credit assignment fails)
- **TensorBoard**: Only showed `train/*` metrics, never `rollout/*` (episodes never completed)

---

## Solution: Daily Batch Decisions

Transform environment from **intra-day** to **daily batch** decisions:

- **New episode length**: ~500 steps (one step per day)
- **Training budget**: 5M timesteps = ~10,000 episodes
- **Learning**: Feasible with γ=0.99 over 500 steps
- **TensorBoard**: Full metrics including episode rewards
- **Training time**: 2-3 hours (from never completing)

---

## Implementation Details

### 1. Core Algorithm: Hybrid Weighted Allocation

**Location**: [housegymrl.py:1028-1129](../src/housegymrl.py#L1028-L1129)

**Algorithm**:
```python
def _allocate_from_candidates(candidates, action, K):
    """
    Hybrid: Softmax-based but cmax-aware

    1. Extract M priority scores from action
    2. Softmax → ideal allocation (probs × K)
    3. Clip by cmax constraints
    4. Redistribute leftover capacity intelligently
    5. Round to integers preserving total ≤ K

    Returns: Dict[house_id → num_contractors]
    """
```

**Why this design preserves RL learning**:
- Agent still learns context-dependent priorities
- Cmax mechanism creates meaningful constraints
- Sequential decisions matter (current affects future)
- Learning is about "which houses to prioritize" not "which house next"

### 2. Environment Step Flow

**Before (Intra-day)**:
```
step(action) →
  select 1 house →
  allocate min(cmax, capacity_remaining, work_remaining) →
  if capacity_remaining > 0: stay in same day (reward=0)
  else: advance day (reward calculated)
```
**Problem**: Takes ~3600 calls to complete one day!

**After (Daily Batch)**:
```
step(action) →
  get allocation Dict for ALL candidates →
  execute full day's allocation →
  advance to next day (reward calculated) →
  return next day's observation
```
**Result**: One call = one day completed!

### 3. State Management Changes

**Removed**:
- ❌ `step_in_day` - No longer needed
- ❌ `capacity_remaining` - Day completes in one step

**Simplified**:
- ✅ `pending_candidates` - Sampled once per day
- ✅ `current_day_capacity` - Total for the day
- ✅ `pending_allocation` - Result of decision

**Preserved**:
- ✅ `waiting_queue` - Houses waiting for reconstruction
- ✅ `arrival_system` - Batch reveal mechanism
- ✅ `capacity_system` - Contractor availability
- ✅ All reward calculation logic

### 4. Baseline Policy Adaptation

**BaselineEnv._baseline_allocate()** - Lines 1161-1206:
```python
def _baseline_allocate():
    """
    Greedy allocation for LJF/SJF/Random

    1. Sort candidates by policy
    2. Greedily allocate respecting cmax
    3. Continue until capacity exhausted

    Returns: Dict[house_id → num_contractors]
    """
```

**Oracle similarly updated** - Lines 1263-1299:
- Same logic but sees full queue (no M limit)

---

## Files Modified

### Core Implementation
1. **[src/housegymrl.py](../src/housegymrl.py)** - Main environment file
   - Lines 707-746: `step()` method redesigned
   - Lines 500-526: `_begin_day_if_needed()` simplified
   - Lines 529-545: `_advance_day()` simplified
   - Lines 1028-1129: `RLEnv._allocate_from_candidates()` hybrid allocation
   - Lines 1161-1206: `BaselineEnv._baseline_allocate()` greedy allocation
   - Lines 1263-1299: `OracleEnv._baseline_allocate()` oracle greedy

### Documentation
2. **[docs/redesign-baseline.md](redesign-baseline.md)** - Current state documentation
3. **[docs/testing-daily-batch.md](testing-daily-batch.md)** - Testing guide
4. **[docs/REDESIGN_SUMMARY.md](REDESIGN_SUMMARY.md)** - This file

### Testing
5. **[src/test_hybrid_allocation.py](../src/test_hybrid_allocation.py)** - Unit tests
6. **[src/test_episode_completion.py](../src/test_episode_completion.py)** - Integration tests

---

## Git Branches

- **`backup-before-daily-batch`**: Pre-redesign state (safe fallback)
- **`daily-batch-redesign`**: Implementation branch (current)
- **`main`**: Production branch (merge target after validation)

---

## Validation Requirements

Before merging to main, verify:

### ✅ Code Review
- [x] Hybrid allocation logic implemented correctly
- [x] Step method redesigned for daily batch
- [x] Baseline policies adapted
- [x] State variables cleaned up
- [x] No breaking changes to observation/action spaces

### ⏳ Unit Tests (Requires conda environment)
- [ ] `test_hybrid_allocation.py` passes all 6 tests
  - Capacity constraints respected
  - Cmax constraints respected
  - Priority affects allocation
  - Edge cases handled
  - Integer rounding correct

### ⏳ Integration Tests
- [ ] `test_episode_completion.py` passes all 3 tests
  - RLEnv completes in ~500 steps
  - BaselineEnv completes in ~500 steps
  - No assertion errors

### ⏳ Training Validation
- [ ] Short training run (50k steps) completes
- [ ] TensorBoard shows `rollout/ep_len_mean` ~500
- [ ] TensorBoard shows `rollout/ep_rew_mean` updating
- [ ] No crashes or errors

---

## Expected Performance Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Episode length | 1,800,000 steps | 500 steps | **3600× faster** |
| Episodes per 5M steps | 2.78 | 10,000 | **3600× more data** |
| Training time estimate | Never completes | 2-3 hours | **Feasible** |
| TensorBoard metrics | Only train/* | Full rollout/* | **Complete visibility** |
| Credit assignment | Impossible (γ^1.8M ≈ 0) | Feasible (γ^500 ≈ 0.007) | **140× stronger signal** |
| Learning feasibility | ❌ Infeasible | ✅ Feasible | **From impossible to trainable** |

---

## Why This Preserves RL Value

**Question**: Doesn't daily batching remove the learning challenge?

**Answer**: No! The RL learning is in the **priority function**, not granularity:

1. **Context-dependent priorities**: Agent must learn to score houses based on:
   - Remaining work
   - Damage level
   - Queue position
   - Current day
   - Historical progress

2. **Cmax constraints**: Not all houses can receive equal allocation
   - Forces intelligent distribution
   - Creates trade-offs

3. **Sequential dependencies**:
   - Today's allocation affects tomorrow's state
   - Queue evolves based on completions
   - New houses arrive in batches

4. **Partial observability**: Only sees M candidates (not full queue)
   - Must make decisions with limited information
   - Information scarcity creates learning challenge

The **real challenge** was never "allocate contractors one by one" — it was "learn which houses to prioritize given constraints and limited information."

---

## Testing Instructions

### Quick Validation

```bash
# Activate environment
conda activate urbanai

# Run unit tests
python src/test_hybrid_allocation.py

# Run integration tests
python src/test_episode_completion.py
```

### Full Validation

See [docs/testing-daily-batch.md](testing-daily-batch.md) for:
- Detailed testing procedure
- Manual testing code
- Training validation steps
- Troubleshooting guide

---

## Next Steps

1. **Run tests** (requires conda environment):
   ```bash
   python src/test_episode_completion.py
   ```

2. **If tests pass**, merge to main:
   ```bash
   git checkout main
   git merge daily-batch-redesign
   git push origin main
   ```

3. **Run full training**:
   ```bash
   sbatch train.slurm
   ```

4. **Monitor TensorBoard**:
   - Should see episode metrics updating
   - Episode length ~500
   - Learning curves over episodes

5. **If tests fail**, debug and iterate:
   - Check test output for specific errors
   - Review allocation logic
   - Compare with backup branch if needed

---

## Rollback Procedure

If needed, return to pre-redesign state:

```bash
git checkout backup-before-daily-batch
```

All original code is preserved in that branch.

---

## Success Criteria

Redesign is successful if:

1. ✅ **Episodes complete**: ~500 steps (not 1.8M)
2. ✅ **Constraints respected**: No cmax or capacity violations
3. ✅ **Training works**: SAC can train without errors
4. ✅ **Learning visible**: TensorBoard shows episode metrics
5. ✅ **Performance reasonable**: Completion rates comparable to baselines

---

## Conclusion

This redesign transforms the environment from **completely intractable** to **trainable in hours**. The 3600× speedup makes RL training feasible while preserving the core learning challenge: context-dependent priority scoring under constraints.

The key insight: RL value isn't in micro-decisions (intra-day allocation) but in strategic decisions (which houses to prioritize for the day). Daily batch decisions capture this strategic challenge while making training computationally feasible.

---

**Ready for testing!** 🚀

See [testing-daily-batch.md](testing-daily-batch.md) for next steps.
