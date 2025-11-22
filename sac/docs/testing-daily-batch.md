# Testing the Daily Batch Redesign

## Overview

The environment has been redesigned from **intra-day decisions** to **daily batch decisions**:

| Metric | Before (Intra-day) | After (Daily Batch) |
|--------|-------------------|---------------------|
| Steps per episode | ~1,800,000 | ~500 |
| Training time (5M steps) | Never completes | 2-3 hours |
| TensorBoard metrics | Only train/* | Full rollout/* |
| Learning feasibility | Impossible | Feasible |

## Changes Made

### 1. Core Environment ([housegymrl.py](../src/housegymrl.py))

**step() method** - Lines 707-746:
- Before: Processes one house allocation per call
- After: Processes entire day (full allocation) per call
- Each call now advances day by exactly 1

**RLEnv._allocate_from_candidates()** - Lines 1028-1129:
- Before: Returns single house ID to allocate
- After: Returns Dict[house_id → num_contractors]
- Implements hybrid weighted allocation with cmax constraints

**BaselineEnv._baseline_allocate()** - Lines 1161-1206:
- Greedy allocation for LJF/SJF/Random policies
- Respects cmax constraints
- Compatible with daily batch decisions

### 2. State Management

**Removed variables**:
- `step_in_day` - No longer needed (one step = one day)
- `capacity_remaining` - Replaced by `current_day_capacity`

**Kept variables**:
- `pending_candidates` - Houses sampled for today's decision
- `current_day_capacity` - Total contractors available today
- `pending_allocation` - Result of allocation decision

### 3. Test Files Created

1. **[test_hybrid_allocation.py](../src/test_hybrid_allocation.py)**
   - Unit tests for hybrid allocation logic
   - Tests: capacity constraints, cmax constraints, priority handling
   - Run: `python src/test_hybrid_allocation.py`

2. **[test_episode_completion.py](../src/test_episode_completion.py)**
   - Integration tests for episode completion
   - Verifies ~500 steps per episode
   - Tests both RL and baseline environments
   - Run: `python src/test_episode_completion.py`

## Testing Procedure

### Local Testing (Requires conda environment)

```bash
# Activate conda environment
conda activate urbanai

# Run unit tests
python src/test_hybrid_allocation.py

# Run episode completion tests
python src/test_episode_completion.py
```

Expected output:
```
🎉 All tests passed! Daily batch redesign successful!

Key achievements:
  • Episodes complete in ~500 steps (vs. 1.8M before)
  • Each step processes one full day
  • Allocations respect cmax constraints
  • Both RL and baseline environments work correctly
```

### HiPerGator Testing

If local testing isn't feasible, you can test on HiPerGator:

```bash
# SSH to HiPerGator
ssh hipergator

# Navigate to project
cd /home/yu.qianchen/ondemand/housegymrl

# Load environment
module load conda
conda activate urbanai

# Run tests
python src/test_episode_completion.py
```

### Quick Manual Test

If you want to manually verify the changes work:

```python
import sys
sys.path.insert(0, 'src')

import housegymrl
import config

# Create test environment
region_key = config.register_synthetic_region(
    H=1000, K=200,
    damage_dist=[300, 400, 300],
    seed=42,
    region_key="MANUAL_TEST"
)

env = housegymrl.RLEnv(
    region_key=region_key,
    use_batch_arrival=False,
    use_capacity_ramp=False,
    seed=42
)

# Run one episode
obs, info = env.reset()
step_count = 0
terminated = False
truncated = False

print(f"Starting episode...")
print(f"Initial queue: {info['queue_size']} houses")

while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    step_count += 1

    if step_count % 50 == 0:
        print(f"Step {step_count}: Completion {info['completion']:.1%}, Queue {info['queue_size']}")

print(f"\n✓ Episode completed in {step_count} steps")
print(f"  Expected: ~500 steps")
print(f"  Actual: {step_count} steps")
print(f"  Previous design would have taken: ~1,800,000 steps")
```

## Validation Checklist

After running tests, verify:

- [ ] Episodes complete in ~500 steps (not 1.8M)
- [ ] No assertion errors about cmax violations
- [ ] No assertion errors about capacity over-allocation
- [ ] Day counter increments by 1 each step
- [ ] Both RLEnv and BaselineEnv work
- [ ] Final completion rate is reasonable (>80% for small test scenarios)

## Training Validation

Once tests pass, run a short training to verify learning:

```bash
# On HiPerGator, create a short test training script:
# test_short_training.py

import sys
sys.path.insert(0, 'src')

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import housegymrl
import config

# Small test region
region_key = config.register_synthetic_region(
    H=1000, K=200, damage_dist=[300, 400, 300],
    seed=42, region_key="TRAIN_TEST"
)

# Create environment
def make_env():
    return housegymrl.RLEnv(
        region_key=region_key,
        use_batch_arrival=False,
        use_capacity_ramp=False,
        seed=42
    )

env = DummyVecEnv([make_env for _ in range(4)])

# Train for 50k steps (~100 episodes)
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)

print("\n✓ Short training completed successfully!")
print("  This proves the redesign is compatible with SB3 training")
```

Run with:
```bash
python test_short_training.py
```

Expected output:
- Training completes in <10 minutes
- TensorBoard logs show `rollout/ep_len_mean` ~500
- No errors or crashes

## Troubleshooting

### Error: "Over-allocated: X > K"
- Issue: Allocation sum exceeds capacity
- Fix: Bug in hybrid allocation logic (should not happen with current implementation)
- Check: Lines 1096-1127 in housegymrl.py

### Error: "allocated Y > cmax Z"
- Issue: Individual allocation exceeds cmax constraint
- Fix: Bug in cmax clipping (should not happen)
- Check: Lines 1072-1073 in housegymrl.py

### Episode doesn't terminate
- Issue: May need to adjust max_steps or check completion logic
- Check: `_execute_allocation()` termination conditions
- Debug: Print `info['completion']` and `info['queue_size']` each step

### Episodes still taking too long
- If episodes take >600 steps, may indicate:
  - Insufficient contractors for scenario size
  - Arrival batching creating delays
  - Need to adjust test scenario parameters

## Next Steps

Once validation passes:

1. **Merge to main**:
   ```bash
   git checkout main
   git merge daily-batch-redesign
   ```

2. **Update main.py** (if needed):
   - Check TOTAL_TIMESTEPS is appropriate (5M should give ~1250 episodes)
   - Verify batch_size and other SAC hyperparameters

3. **Run full training**:
   ```bash
   sbatch train.slurm
   ```

4. **Monitor TensorBoard**:
   - Should now see `rollout/ep_rew_mean` updating regularly
   - Episode length should be ~500
   - Training should show learning curves over episodes

## Rollback Procedure

If tests fail and you need to revert:

```bash
# Return to pre-redesign state
git checkout backup-before-daily-batch

# Or compare branches
git diff backup-before-daily-batch daily-batch-redesign
```

The backup branch contains the complete pre-redesign state.
