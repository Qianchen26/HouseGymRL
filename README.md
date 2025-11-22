# HouseGym RL

Deep reinforcement learning for post-disaster housing recovery scheduling.

## Overview

This project applies reinforcement learning to optimize resource allocation in disaster recovery scenarios. Given limited construction workers and uncertain damage assessments, how should we schedule repairs across hundreds of damaged houses to maximize recovery speed while maintaining fairness?

**Problem characteristics:**
- **Long-horizon decision-making**: 500-day episodes with sequential allocation
- **Batch arrival**: Houses revealed in stages as damage assessments complete
- **Resource constraints**: Limited workers with noisy capacity estimates
- **Multi-objective**: Balance completion speed, fairness, and queue management

**Dataset**: Real-world Lombok earthquake (Indonesia) reconstruction data
- 300+ houses across 7 administrative regions
- ~372,000 total man-days of construction work
- Historical completion data for validation

## Project Structure

```
housegym_rl/
├── README.md              # This file
├── data/                  # Shared datasets
│   └── lombok_data.pkl   # Lombok earthquake reconstruction data
│
├── sac/                   # Soft Actor-Critic implementation
│   ├── src/              # SAC training and evaluation code
│   ├── scripts/          # SLURM scripts for HiPerGator
│   ├── docs/             # SAC-specific documentation
│   ├── models/           # Trained SAC models
│   ├── runs/             # TensorBoard logs
│   ├── results/          # Evaluation results
│   └── README_SAC.md     # Detailed SAC documentation
│
└── ppo/                   # Proximal Policy Optimization implementation (current)
    ├── src/              # PPO training and evaluation code
    ├── scripts/          # SLURM scripts for HiPerGator
    ├── configs/          # Training configurations
    ├── notebooks/        # Jupyter notebooks for demos
    ├── docs/             # PPO-specific documentation
    └── README.md         # Detailed PPO documentation
```

## Quick Start

### SAC (Legacy Implementation)

For the original Soft Actor-Critic implementation:
- See [sac/README_SAC.md](sac/README_SAC.md) for complete documentation
- Trained models available in [sac/models/](sac/models/)
- Results and analysis in [sac/results/](sac/results/)

### PPO (Current Implementation)

For the latest Proximal Policy Optimization implementation:
- See [ppo/README.md](ppo/README.md) for complete documentation
- Training scripts in [ppo/scripts/](ppo/scripts/)
- Demo notebooks in [ppo/notebooks/](ppo/notebooks/)

**Recommended for new work**: Use the PPO implementation, as it has better stability and is actively maintained.

## Why Two Implementations?

This project initially used **SAC (Soft Actor-Critic)**, an off-policy algorithm well-suited for continuous action spaces. However, we later migrated to **PPO (Proximal Policy Optimization)** for several reasons:

1. **Stability**: PPO's clipped objective provides more stable training
2. **Sample efficiency**: On-policy training better fits our episodic problem structure
3. **CPU-friendly**: PPO performs well on CPU clusters, better suited for HiPerGator
4. **Hyperparameter robustness**: PPO is less sensitive to hyperparameter choices

Both implementations are preserved for:
- **Comparison**: Benchmark different RL algorithms on the same problem
- **Research**: Study algorithm-specific behaviors on disaster recovery scheduling
- **Reproducibility**: All published results remain accessible

## Key Features

### Environment Design

- **Daily batch decision framework**: Allocate workers across all houses each day
- **Top-M observation pruning**: Focus on most critical houses using softmax priorities
- **Hybrid weighted allocation**: Convert RL actions to feasible allocations with capacity constraints
- **Realistic uncertainty**: Observation noise, capacity fluctuations, stochastic work progress

### Reward Function V2.1

```python
# Three-component reward architecture
progress_reward = (houses_completed_today / total_revealed) × 1.0
completion_reward = (total_completed / total_revealed) × 1.0
queue_penalty = sigmoid_penalty(waiting_times) × 0.1  # capped at -5.0

reward = (progress + completion + queue_penalty) × 100
```

### Baseline Comparison

- **Longest Job First (LJF)**: Prioritize most difficult houses
- **Shortest Job First (SJF)**: Maximize completion count
- **Random**: Unbiased baseline
- **RL (SAC/PPO)**: Learned adaptive strategy

## Development

### Running Tests
```bash
# SAC tests
cd sac/src
python test_hybrid_allocation.py

# PPO tests
cd ppo/src
python -m pytest tests/
```

### HiPerGator Training

```bash
# SAC training
cd sac
sbatch scripts/train.slurm

# PPO training
cd ppo
sbatch scripts/train.slurm
```

### Monitoring

```bash
# TensorBoard (SAC)
tensorboard --logdir sac/runs/

# TensorBoard (PPO)
tensorboard --logdir ppo/runs/
```

## Citation

Based on Lombok earthquake reconstruction data. Research code for disaster recovery optimization using deep reinforcement learning.

## Contact

For questions or collaboration, please open an issue on GitHub.
