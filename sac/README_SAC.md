# HouseGym RL

Reinforcement learning for post-disaster housing recovery scheduling using Soft Actor-Critic (SAC).

## What is this project?

This project applies deep reinforcement learning to optimize resource allocation in post-disaster housing reconstruction. Given limited construction workers and uncertain damage assessments, how should we schedule repairs across hundreds of damaged houses to maximize recovery speed and fairness?

**Problem characteristics:**
- **Long-horizon**: 500-day episodes with sequential decision-making
- **Batch arrival**: Houses revealed in stages (day 0, 30, 60...) as damage assessments complete
- **Resource constraints**: Limited workers with noisy capacity estimates
- **Competing objectives**: Fast completion vs fairness (avoid abandoning difficult houses)

**Dataset**: Lombok earthquake (Indonesia) reconstruction data
- 300+ houses across 7 administrative regions
- ~372,000 total man-days of construction work
- Historical completion data for baseline validation

## Why Reinforcement Learning?

Traditional heuristics (Longest Job First, Shortest Job First, Random) fail to capture:

1. **Sequential dependencies**: Early decisions affect future state and available actions
2. **Uncertainty**: Noisy observations (±2%) and stochastic work progress (±20%)
3. **Multi-objective optimization**: Progress, completion rate, and queue fairness must be balanced
4. **Long-term consequences**: Greedy allocation can strand difficult houses

RL learns policies that adapt to queue state, resource availability, and time pressure without hand-crafted rules.

## How does it work?

### Environment Design

**Daily Batch Decision Framework** (500 steps/episode):
- Each step = one day of allocation decisions
- Agent observes top-M houses (weighted by softmax priorities)
- Agent outputs continuous weights for allocation distribution
- Allocation algorithm applies weights with capacity constraints
- Reduced from 1.8M steps (per-house decisions) for training feasibility

**Observation Space** (dimension: M × 4):
```python
# Top-M houses selected by softmax(damage_remaining / (queue_time + 1))
[
    [damage_level, damage_remaining_fraction, total_mandays, queue_days],
    ...  # M houses
]
# M dynamically adjusts to queue size: M = min(max(0.8 × queue_size, 256), 512)
```

**Action Space** (dimension: M):
- Continuous weights in [0, 1] for each house
- Transformed to allocation via hybrid weighted algorithm
- Invalid actions (exceeding capacity) handled by allocation constraints

**Reward Function V2.1** (3-component architecture):
```python
# Component 1: Dense Progress (houses completed this step / total revealed)
progress = (current_completed - last_completed) / num_revealed

# Component 2: Cumulative Completion (completion fraction)
completion = current_completed / num_revealed

# Component 3: Sigmoid Queue Penalty (capped at -5.0)
queue_penalty = sum(sigmoid_penalty(wait_time)) for waiting houses
queue_penalty = max(queue_penalty, -5.0)  # Prevent signal dominance

# Final reward (scaled ×100 for SAC learning dynamics)
reward = (progress × 1.0 + completion × 1.0 + queue_penalty × 0.1) × 100
```

Key improvements from V1:
- Removed `capacity_usage` (always ~1.0 by design, uninformative)
- Capped queue penalty to prevent dominance over positive signals
- Dense progress reward for better gradient signals

**Allocation Algorithm** (Hybrid Weighted):
1. Apply softmax to action weights: `priorities = softmax(action)`
2. Compute tentative allocation: `K × priorities` (K = available workers)
3. Apply per-house capacity constraints: `min(allocation[i], cmax[i])`
4. Redistribute leftover capacity proportionally to houses with remaining headroom
5. Add ±20% noise to simulate construction variability

### Training Configuration

**Algorithm**: Soft Actor-Critic (SAC)
- Entropy-regularized off-policy actor-critic
- Handles continuous action spaces with stochastic policies
- Hyperparameters:
  - Buffer size: 1.5M (3000 episodes × 500 steps)
  - Batch size: 256
  - Learning starts: 25k steps (5% buffer fill)
  - Learning rate: 3e-4
  - Noise: observation ±2%, capacity ±2%, work progress ±20%

**Training duration**: 500k steps (~1000 episodes, ~3-4 hours on HiPerGator GPU)

**Expected performance**:
- Episode reward: +50 to +200 (scaled)
- Queue penalty: -5.0 to 0 (capped)
- Completion rate: Target 60-80% by day 500

## Code Structure

```
housegym_rl/
├── src/
│   ├── housegymrl.py          # Core RL environment (RLEnv + BaseEnv)
│   ├── main.py                # SAC training pipeline with TensorBoard logging
│   ├── evaluate.py            # Evaluation pipeline (RL vs baselines)
│   ├── baseline.py            # Baseline strategies (LJF/SJF/Random)
│   ├── config.py              # Region configs and hyperparameters
│   ├── synthetic_scenarios.py # Generate training scenarios
│   ├── lombok_data.pkl        # Real Lombok reconstruction data
│   └── test_*.py              # Unit tests for allocation/completion
├── train.slurm                # HiPerGator SLURM training script
├── evaluate.slurm             # HiPerGator SLURM evaluation script
├── artifacts/                 # Generated outputs (models, logs, results)
│   ├── models/                # Trained SAC policies
│   ├── runs/                  # TensorBoard logs
│   ├── logs/                  # SLURM stdout/stderr
│   └── results/               # Evaluation CSVs and plots
├── data/                      # Additional datasets
├── docs/                      # Documentation and guides
└── PROJECT_OVERVIEW.md        # Detailed project context (may be outdated)
```

### Key Modules

**housegymrl.py** (~1200 lines)
- `BaseEnv`: Core environment mechanics (batch arrival, allocation, reward)
- `RLEnv(BaseEnv)`: RL-specific observation/action handling
- `_calculate_reward()`: Reward V2.1 implementation ([lines 642-710](src/housegymrl.py#L642-L710))
- `_allocate_from_candidates()`: Hybrid weighted allocation ([lines 1082-1110](src/housegymrl.py#L1082-L1110))
- `_queue_penalty_sigmoid()`: Smooth queue penalty function ([lines 615-640](src/housegymrl.py#L615-L640))

**main.py** (~1000 lines)
- SAC model initialization with Stable-Baselines3
- `RewardComponentLogger`: TensorBoard callback for reward decomposition ([lines 867-906](src/main.py#L867-L906))
- Training loop with checkpointing

**evaluate.py** (~500 lines)
- Rollout trained policies and baselines
- Generate comparison metrics (completion rate, makespan, fairness)
- Save results to CSV

**config.py** (~200 lines)
- `REGION_CONFIG`: 7 Lombok regions with historical data
- `MAX_STEPS = 500`: Episode length
- Batch arrival configuration: `days=[0, 30, 60]`, `ratios=[0.4, 0.35, 0.25]`

## Quick Start

### Local Testing (5k steps)

```bash
cd src
python main.py  # Modify TOTAL_TIMESTEPS=5000 for quick test
```

Monitor training:
```bash
tensorboard --logdir ../runs/sac_diverse/
```

### HiPerGator Training

```bash
# On local machine: push code to GitHub
git push origin main

# On HiPerGator login node
cd ~/ondemand/housegymrl
git pull origin main

# Submit training job (72 hours, 100GB RAM, 1 GPU)
sbatch train.slurm

# Monitor logs
tail -f artifacts/logs/train_*.out
```

### Evaluation

```bash
# After training completes
sbatch evaluate.slurm

# Download results to local
scp -r yu.qianchen@hpg.rc.ufl.edu:~/ondemand/housegymrl/artifacts/results/ ./artifacts/
```

## Monitoring and Debugging

### TensorBoard Metrics

**Rollout metrics**:
- `rollout/ep_rew_mean`: Episode reward (target: positive values)
- `rollout/ep_len_mean`: Episode length (should be 500)

**Reward components** (for debugging reward balance):
- `reward/progress`: Dense progress signal (0 to ~0.01 per step)
- `reward/completion`: Cumulative completion (0 to 1)
- `reward/queue_penalty`: Queue penalty (-5.0 to 0)
- `reward/raw_total`: Unscaled total reward (-0.5 to +2 expected)

**Training metrics**:
- `train/actor_loss`: Actor network loss
- `train/critic_loss`: Critic network loss (target: decrease from 15k to 1-5k)
- `train/ent_coef`: Entropy coefficient (automatic tuning)

**Monitoring metrics** (not in reward):
- `monitor/capacity_usage`: Worker utilization (always ~1.0 by design)

### Common Issues

**Reward stagnation or negative episode rewards**:
- Check `reward/queue_penalty` - should stay in [-5.0, 0], not -370
- Check `reward/raw_total` - should trend positive, not -37
- Verify capacity_usage is ~1.0 (confirms allocation algorithm works)

**Critic loss oscillation (5k-15k)**:
- Reduce noise levels (observation_noise, capacity_noise)
- Increase learning_starts or batch_size
- Check reward scaling (should be ×100)

**Episode length != 500**:
- Environment truncation error
- Check MAX_STEPS in config.py

## Key Innovations

1. **Daily Batch Redesign**: Reduced episode length from 1.8M steps (allocate to one house per step) to 500 steps (allocate to all houses per day). Makes RL training feasible while preserving decision complexity.

2. **Hybrid Weighted Allocation**: Combines RL-learned priorities with softmax normalization and capacity constraints. Avoids infeasible allocations while allowing learned preferences.

3. **Reward V2.1 Architecture**:
   - Dense progress signal (every step counts)
   - Sigmoid queue penalty (prevents penalty explosion)
   - Removed uninformative components (capacity_usage always 1.0)
   - Component-level logging for debugging

4. **Noise Calibration**: Reduced noise levels (15% → 2%) to stabilize critic learning while retaining realistic uncertainty.

## Baseline Comparison

**Longest Job First (LJF)**: Prioritize houses with most remaining work
- Rationale: Complete difficult houses early to avoid long tails
- Weakness: May neglect quick wins

**Shortest Job First (SJF)**: Prioritize houses with least remaining work
- Rationale: Maximize completion count quickly
- Weakness: May strand difficult houses indefinitely

**Random**: Uniform random allocation
- Rationale: Unbiased baseline
- Weakness: No strategic planning

**RL (SAC)**: Learned adaptive strategy
- Learns to balance progress, completion, and fairness
- Adapts to queue state and time pressure
- Expected to outperform heuristics on multi-objective metrics

## Development Notes

**Running tests**:
```bash
cd src
python test_hybrid_allocation.py  # Verify allocation algorithm
python test_episode_completion.py  # Verify episode mechanics
```

**Modifying reward function**:
1. Edit `_calculate_reward()` in [housegymrl.py:642-710](src/housegymrl.py#L642-L710)
2. Update `RewardComponentLogger` in [main.py:867-906](src/main.py#L867-L906)
3. Update info dict in `step()` method
4. Test locally before HiPerGator deployment

**Adding new baselines**:
1. Add strategy to [baseline.py](src/baseline.py)
2. Update `evaluate.py` to include new strategy
3. Rerun evaluation

## Citation

Based on Lombok earthquake reconstruction data. This is research code for disaster recovery optimization.

## Contact

For questions about implementation details or code review feedback, please open an issue on GitHub.
