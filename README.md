# HouseGym RL

Deep reinforcement learning for post-disaster housing recovery scheduling.

## Project Description

This project applies reinforcement learning to optimize resource allocation in disaster recovery scenarios. Given limited construction workers and uncertain damage assessments, how should we schedule repairs across thousands of damaged houses to maximize recovery speed while maintaining fairness?

**Problem characteristics:**
- **Long-horizon decision-making**: 500-day episodes with sequential allocation
- **Batch arrival**: Houses revealed in stages as damage assessments complete
- **Resource constraints**: Limited workers with noisy capacity estimates
- **Multi-objective**: Balance completion speed, fairness, and queue management

**Dataset**: Lombok earthquake (Indonesia, 2018) reconstruction data
- 7 administrative regions with varying damage distributions
- 3 damage severity levels: Minor, Moderate, Major
- Historical completion data for validation

**Method**: Proximal Policy Optimization (PPO) with:
- Top-M candidate selection by waiting time
- Softmax-based allocation with capacity constraints
- Multi-scenario training for robustness

## Project Structure

```
housegym_rl/
├── README.md
├── data/
│   └── lombok_data.pkl       # Lombok earthquake reconstruction data
├── demo/
│   └── demo_notebook.ipynb   # Interactive demonstration
└── src/
    ├── housegymrl.py         # Environment: BaseEnv, RLEnv, BaselineEnv, OracleEnv
    ├── main_ppo.py           # Training script
    ├── evaluate_ppo.py       # Evaluation script
    ├── baseline.py           # Baseline policies (LJF, SJF, Random)
    ├── config.py             # Configuration constants
    ├── ppo_configs.py        # PPO hyperparameters
    └── synthetic_scenarios.py # Synthetic data generator
```

## Guide

### Installation

```bash
# Clone repository
git clone <repository-url>
cd housegym_rl

# Install dependencies
pip install numpy pandas gymnasium stable-baselines3 tensorboard
```

### Training

**Local training:**
```bash
cd src
python main_ppo.py --experiment-name my_experiment --timesteps 500000
```

**Multi-scenario training (recommended for robustness):**
```bash
python main_ppo.py --experiment-name robust_agent --use-synthetic --timesteps 2000000
```

**Key training arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--experiment-name` | required | Name for saving outputs |
| `--timesteps` | 1,000,000 | Total training steps |
| `--n-envs` | 16 | Parallel environments |
| `--use-synthetic` | False | Train on 180 synthetic scenarios |
| `--obs-noise` | 0.15 | Observation noise level |
| `--capacity-noise` | 0.10 | Capacity noise level |

### Evaluation

```bash
python evaluate_ppo.py \
    --checkpoint-dir runs/my_experiment \
    --test-regions Mataram Sumbawa "Central Lombok" \
    --crew-levels 0.3 0.5 0.7 1.0 \
    --compare-baselines
```

### Monitoring

```bash
# TensorBoard
tensorboard --logdir runs/

# Key metrics to watch:
# - rollout/ep_rew_mean: Episode reward (should increase)
# - train/clip_fraction: Policy change magnitude (target: 0.1-0.2)
# - train/entropy_loss: Exploration level (should stay > -3)
```

### HiPerGator (SLURM)

Training and evaluation scripts are provided in `scripts/`:

```bash
# Submit training job
sbatch scripts/train.slurm

# Submit evaluation job
sbatch scripts/evaluate.slurm

# Monitor job
squeue -u $USER
tail -f slurm-*.out
```

## Environment Overview

The `RLEnv` environment simulates daily allocation decisions:

1. **Observation** (6150 dimensions):
   - 6 global features: day, capacity, queue_size, revealed_count, remain_ratio, major_ratio
   - 1024 × 6 candidate features: remaining_work, waiting_time, total_work, damage_level, cmax, mask

2. **Action** (1024 dimensions):
   - Priority scores for each candidate (continuous, 0-1)
   - Converted to allocations via softmax + capacity constraints

3. **Reward** (3 components):
   - Progress: Work completed this step
   - Completion bonus: Houses finished
   - Queue penalty: Average waiting time (capped)

## Baseline Comparison

| Policy | Strategy | Strengths |
|--------|----------|-----------|
| LJF | Longest Job First | Reduces long-tail completion |
| SJF | Shortest Job First | Maximizes early completions |
| Random | Random priorities | Unbiased baseline |
| Oracle-LJF/SJF | Full queue visibility | Upper bound (no M limit) |

## Citation

Research code for disaster recovery optimization using deep reinforcement learning, based on Lombok earthquake reconstruction data.
