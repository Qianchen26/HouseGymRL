# HouseGym RL

Reinforcement learning for post-disaster housing reconstruction scheduling.

## Project Description

This project uses reinforcement learning to optimize how construction workers are assigned to damaged houses after a disaster. The core challenge: given limited workers and incomplete information about damage, how should we prioritize repairs to finish reconstruction as quickly and fairly as possible?

**Key challenges:**
- **Sequential decisions over time**: Each day, decide which houses get workers (500-day episodes)
- **Incomplete information**: Houses are discovered gradually as damage assessments complete
- **Limited resources**: Fixed number of workers, noisy capacity estimates
- **Multiple goals**: Fast completion, short wait times, fair treatment

**Method**: Proximal Policy Optimization (PPO) trained on synthetic scenarios for robustness.

## Dataset

Based on the 2018 Lombok earthquake (Indonesia) reconstruction data.

| Split | Source | Description |
|-------|--------|-------------|
| **Training** | 180 synthetic scenarios | Generated from 3 damage distribution families (major-dominant, balanced, minor-dominant) with randomized house counts and worker ratios |
| **Testing** | 7 real Lombok regions | Mataram, West/North/Central/East Lombok, West Sumbawa, Sumbawa |

Each region has:
- 3 damage levels: Minor, Moderate, Major
- Different damage distributions and contractor counts
- Used for evaluating generalization to unseen scenarios

## Project Structure

```
housegym_rl/
├── README.md
├── data/
│   └── lombok_data.pkl       # Lombok earthquake reconstruction data
├── demo/
│   └── demo_notebook.ipynb   # Interactive demonstration
└── src/
    ├── housegymrl.py         # Environment classes
    ├── main_ppo.py           # Training script
    ├── evaluate_ppo.py       # Evaluation script
    ├── baseline.py           # Baseline policies (LJF, SJF, Random)
    ├── config.py             # Configuration constants
    ├── ppo_configs.py        # PPO hyperparameters
    └── synthetic_scenarios.py # Training data generator
```

## Quick Start

### Demo Notebook

The easiest way to understand the project is to run the demo notebook:

```bash
cd demo
jupyter notebook demo_notebook.ipynb
```

The notebook walks through:
1. **Environment setup**: Create an environment and visualize the observation space
2. **Single episode rollout**: Run one episode and plot the completion curve over time
3. **Baseline comparison**: Compare LJF, SJF, and Random policies on the same scenario
4. **Trained agent evaluation**: Load a trained PPO model and compare against baselines

Running the full notebook takes about 5 minutes and produces plots showing how different policies perform.

### Installation

```bash
git clone <repository-url>
cd housegym_rl
pip install numpy pandas gymnasium stable-baselines3 tensorboard
```

### Training

```bash
cd src

# Single-region training (quick test)
python main_ppo.py --experiment-name test_run --timesteps 100000

# Multi-scenario training (recommended)
python main_ppo.py --experiment-name robust_agent --use-synthetic --timesteps 2000000
```

**Training arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiment-name` | required | Name for saving outputs |
| `--timesteps` | 1,000,000 | Total training steps |
| `--n-envs` | 16 | Parallel environments |
| `--use-synthetic` | False | Train on 180 synthetic scenarios |

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
tensorboard --logdir runs/
```

Key metrics:
- `rollout/ep_rew_mean`: Episode reward (should increase)
- `train/clip_fraction`: Policy change per update (target: 0.1-0.2)
- `train/entropy_loss`: Exploration level (should stay above -3)

## Environment Overview

The `RLEnv` simulates daily worker allocation:

1. **Observation** (6150 dimensions):
   - 6 global features: day, capacity, queue size, etc.
   - 1024 × 6 candidate features: remaining work, waiting time, damage level, etc.

2. **Action** (1024 dimensions):
   - Priority score for each candidate house
   - Converted to worker assignments via softmax

3. **Reward**:
   - Work progress (main signal)
   - Completion bonus (houses finished)
   - Queue penalty (waiting time)

## Baselines

| Policy | Strategy |
|--------|----------|
| LJF | Longest Job First - prioritize hardest jobs |
| SJF | Shortest Job First - maximize completions |
| Random | Random assignment |
| Oracle | Full visibility (upper bound) |
