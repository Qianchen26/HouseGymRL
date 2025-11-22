# HouseGym RL: PPO Implementation for Disaster Recovery Scheduling

Proximal Policy Optimization (PPO) implementation for post-disaster housing reconstruction scheduling in Indonesia. This project trains reinforcement learning agents to allocate limited contractor crews across damaged houses while managing uncertainty in arrival times, work duration, and resource availability.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Environment Description](#environment-description)
- [PPO Implementation](#ppo-implementation)
- [Ablation Framework](#ablation-framework)
- [Evaluation](#evaluation)
- [File Structure](#file-structure)
- [Usage Guide](#usage-guide)
- [Configuration Reference](#configuration-reference)
- [HiPerGator Deployment](#hipergator-deployment)
- [Troubleshooting](#troubleshooting)
- [Development Notes](#development-notes)
- [Migration from SAC](#migration-from-sac)

## Overview

### Problem Statement

After a disaster, reconstruction agencies must schedule contractor crews to repair damaged houses. The scheduling problem has the following characteristics:

- **Variable queue size**: Houses arrive in batches over time (days 0, 30, 60), totaling 14,517 houses across regions
- **Limited resources**: 60-200 contractor crews available, each processing 1-5 houses per day
- **Multiple objectives**: Maximize completion rate, minimize average waiting time
- **Uncertainty sources**:
  - Batch arrival timing (3 batches with ratios 0.4/0.35/0.25)
  - Stochastic work duration (±20% noise on expected completion)
  - Observation noise (15% σ on reported remaining work)
  - Capacity fluctuation (crews available in range [90%, 100%] of base capacity)

### Solution Approach

This implementation uses PPO, an on-policy actor-critic algorithm, to learn scheduling policies. The agent:

1. Observes current queue state (2054-dimensional vector: 6 global features + 512 candidates × 4 features each)
2. Outputs priority scores for top-512 candidates (continuous values in [0, 1])
3. Receives reward based on completion progress, finished houses, and queue penalties
4. Updates policy using clipped objective and generalized advantage estimation (GAE)

### Key Features

- **Fixed observation dimension**: Uses M=512 candidate selection to avoid variable input size (14,517 → 512)
- **Bug fixes**: Resolves 4 critical issues from SAC version:
  1. StaticArrival reset bug causing empty queues on episode reset
  2. Allocation ignoring remaining_work constraints
  3. Progress reward dynamic denominator breaking temporal consistency
  4. VecNormalize save/load path inconsistency
- **Systematic ablation**: Framework for testing uncertainty mechanism combinations (10 configurations across 4 stages)
- **Comprehensive evaluation**: Cross-region, cross-crew-level testing with baseline comparisons (LJF, SJF, Random)
- **HiPerGator integration**: SLURM scripts for batch job submission on UF HPC cluster

## Installation

### Prerequisites

- Python 3.9 or higher
- Conda (recommended) or venv
- 16GB+ RAM for training
- CUDA-compatible GPU (optional, CPU training supported)

### Local Setup

```bash
# Clone repository
cd /path/to/housegym_rl/ppo

# Create conda environment
conda create -n housegym_rl python=3.9
conda activate housegym_rl

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import gymnasium; import stable_baselines3; print('Installation successful')"
```

### HiPerGator Setup

```bash
# SSH to HiPerGator
ssh your_username@hpg.rc.ufl.edu

# Navigate to project directory
cd /home/yu.qianchen/ondemand/housegymrl_ppo/ppo

# Load modules
module load conda
conda activate housegym_rl

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run setup script
cd scripts
bash setup_env.sh
```

## Quick Start

### Training a Single Model

```bash
cd src

# Train PPO on Mataram region for 500k steps
python main_ppo.py \
    --experiment-name my_first_experiment \
    --region Mataram \
    --timesteps 500000 \
    --n-envs 16
```

**Output locations**:
- Model: `runs/my_first_experiment/model.zip`
- VecNormalize: `runs/my_first_experiment/vecnormalize.pkl`
- TensorBoard logs: `runs/my_first_experiment/tb_logs/`
- Checkpoints: `runs/my_first_experiment/checkpoints/`

### Evaluating a Trained Model

```bash
cd src

# Evaluate on test scenarios
python evaluate_ppo.py \
    --checkpoint-dir runs/my_first_experiment \
    --output-dir evaluation_results/my_first_experiment \
    --test-regions Mataram Sumbawa "Central Lombok" \
    --crew-levels 0.3 0.5 0.7 1.0 \
    --n-seeds 5 \
    --compare-baselines
```

**Output files**:
- `evaluation_results/my_first_experiment/evaluation_results.csv`: PPO results
- `evaluation_results/my_first_experiment/all_methods_comparison.csv`: PPO vs baselines

### Running Ablation Experiments

```bash
cd src

# Run all 10 ablation experiments sequentially
python ablation_framework.py \
    --run-all \
    --config-dir ../configs/ablation

# Or run in parallel (requires joblib)
python ablation_framework.py \
    --run-all \
    --config-dir ../configs/ablation \
    --parallel 4

# Compare results only (if experiments already run)
python ablation_framework.py \
    --compare-only \
    --config-dir ../configs/ablation
```

**Output locations**:
- Individual results: `experiments/ablation/[experiment_name]/results/`
- Comparison summary: `experiments/ablation/comparison_summary.csv`
- Markdown report: `experiments/ablation/ablation_report.md`

## Environment Description

### Observation Space

Dimension: 2054 (fixed)

**Global features (6 dimensions)**:
1. Current day (integer, range [0, 500])
2. Houses in queue (integer, range [0, 14517])
3. Houses completed (integer, range [0, 14517])
4. Available capacity (float, number of crews ready)
5. Cumulative allocated (float, total crew-days allocated so far)
6. Completion rate (float, fraction in [0, 1])

**Per-candidate features (512 candidates × 4 features = 2048 dimensions)**:

For each of 512 candidates (top houses by queue time):
1. Remaining work (float, days remaining with 15% observation noise)
2. Queue time (float, days waited since arrival)
3. Original work estimate (float, expected days at arrival)
4. Deadline urgency (float, days until deadline)

**Candidate selection**: Fixed M=512 candidates selected by longest queue time. If fewer than 512 houses in queue, padding with zeros.

### Action Space

Dimension: 512 (continuous, bounded [0, 1])

**Action interpretation**:
1. Agent outputs raw priority scores p_i ∈ [0, 1] for i = 1, ..., 512
2. Softmax normalization: w_i = exp(p_i) / Σ exp(p_j)
3. Ideal allocation: ideal_i = w_i × total_capacity
4. Constraint clipping: allocation_i = min(ideal_i, capacity_max_i, remaining_work_i)
5. Integer rounding: crews_i = round(allocation_i)

**Key constraint**: Respects remaining_work to avoid over-allocation to nearly-complete houses (fixes Critical Issue #2 from SAC version).

### Reward Function (V3)

Three components:

**1. Progress reward** (action-dependent):
```
r_progress = work_completed_this_step / total_work_all_houses
```
where denominator is fixed at episode start (fixes Critical Issue #3).

Range: [0, ~0.01] per step (depends on allocation efficiency)

**2. Completion bonus**:
```
r_completion = num_finished_this_step × 2.0
```
Sparse signal: +2.0 per house finished.

**3. Queue penalty**:
```
r_queue = -min(current_queue_size / 5000, 2.0)
```
Bounded penalty: range [-2.0, 0.0]

**Total reward**:
```
r_total = r_progress + r_completion + r_queue
```

Typical range per step: [-2.0, +0.5]

Episode reward (500 steps): typically [-500, +200] depending on performance

### Uncertainty Mechanisms

Four independent mechanisms (can be toggled via configuration):

**1. Batch Arrival** (`use_batch_arrival`):
- **Enabled**: Houses arrive in 3 batches at days 0, 30, 60 with ratios 0.4, 0.35, 0.25
- **Disabled**: All houses revealed at day 0 (perfect information)

**2. Stochastic Duration** (`stochastic_duration`):
- **Enabled**: Actual work = expected_work × (1 + ε), where ε ~ N(0, 0.2²)
- **Disabled**: Actual work = expected_work (deterministic)

**3. Observation Noise** (`observation_noise`):
- **Enabled**: Observed remaining_work = true_remaining × (1 + η), where η ~ N(0, σ²), σ=0.15
- **Disabled**: Observed remaining_work = true_remaining (perfect observations)
- **Note**: Observations clipped to [0, 2×true_remaining] to prevent negative values

**4. Capacity Noise** (`capacity_noise`):
- **Enabled**: Daily capacity = base_capacity × U(1-δ, 1), where δ=0.10
- **Disabled**: Daily capacity = base_capacity (fixed resources)

### Episode Termination

Episode ends when:
1. Current day ≥ max_steps (default: 500 days), OR
2. All houses completed (early termination)

Final info dictionary includes:
- `completion_rate`: Fraction of houses finished
- `avg_queue_time`: Mean waiting time in days
- `houses_completed`: Total count
- `episode_length`: Steps taken

### Regions and Scenarios

Seven regions from Indonesia 2018 earthquake dataset:

| Region | Houses | Base Contractors | Crew Levels Tested |
|--------|--------|------------------|-------------------|
| Mataram | 5,982 | 200 | 60, 100, 140, 200 |
| West Lombok | 3,129 | 100 | 30, 50, 70, 100 |
| North Lombok | 2,639 | 80 | 24, 40, 56, 80 |
| Central Lombok | 1,523 | 60 | 18, 30, 42, 60 |
| East Lombok | 894 | 40 | 12, 20, 28, 40 |
| West Sumbawa | 229 | 20 | 6, 10, 14, 20 |
| Sumbawa | 121 | 10 | 3, 5, 7, 10 |

**Crew levels**: Tested at 30%, 50%, 70%, 100% of base contractors to simulate resource scarcity.

## PPO Implementation

### Algorithm Overview

PPO (Proximal Policy Optimization) is an on-policy actor-critic algorithm that:

1. **Collects experience**: Runs current policy for N steps across parallel environments
2. **Computes advantages**: Uses GAE to estimate advantage function A(s, a)
3. **Updates policy**: Maximizes clipped objective to prevent large policy changes
4. **Updates value function**: Minimizes squared error between predicted and actual returns

**Why PPO for this problem**:
- **Sparse rewards**: GAE provides better credit assignment than TD methods
- **Stability**: Clipped objective prevents destructive policy updates
- **On-policy**: Handles variable observation dimensions better than off-policy methods
- **Sample efficiency**: Reuses data for K epochs per collection phase

### Hyperparameters

Default configuration (`ppo_configs.py: PPOConfig`):

```python
learning_rate = 3e-4        # Adam optimizer learning rate
n_steps = 2048              # Steps per environment before update
batch_size = 256            # Mini-batch size for SGD
n_epochs = 10               # Optimization epochs per update
gamma = 0.99                # Discount factor
gae_lambda = 0.95           # GAE parameter
clip_range = 0.2            # PPO clipping range
ent_coef = 0.01             # Entropy regularization coefficient
vf_coef = 0.5               # Value function loss coefficient
max_grad_norm = 0.5         # Gradient clipping threshold
```

**Update frequency**: With 16 environments and n_steps=2048, policy updates every 16×2048 = 32,768 timesteps.

**Total updates**: For 500k timesteps, ~15 policy updates occur.

### Network Architecture

**Policy network** (actor):
- Input: 2054-dimensional observation vector
- Hidden layers: [64, 64] (MLP, ReLU activation)
- Output: 512-dimensional action (mean), 512-dimensional log-std
- Activation: Tanh squashing to [0, 1]

**Value network** (critic):
- Input: 2054-dimensional observation vector
- Hidden layers: [64, 64] (MLP, ReLU activation)
- Output: Single scalar value estimate

**Parameter count**: ~270k trainable parameters

### Normalization

**Observation normalization** (`VecNormalize`):
- Maintains running mean μ and std σ for each observation dimension
- Normalized obs = (raw_obs - μ) / (σ + ε)
- Clipped to [-10, 10] to prevent outliers
- **Critical**: Training statistics saved to `vecnormalize.pkl` and reused during evaluation

**Reward normalization**:
- Maintains running return estimate with discount γ=0.99
- Normalized reward = raw_reward / (std_return + ε)
- Clipped to [-10, 10]
- **Note**: Normalization disabled during evaluation (training=False)

### Training Loop

Implemented in `main_ppo.py`:

```python
# Pseudocode
vec_env = setup_vec_env(n_envs=16, env_config=ENV_DEFAULT)
model = create_ppo_model(vec_env, ppo_config=PPO_DEFAULT)

for timestep in range(0, total_timesteps, n_steps * n_envs):
    # 1. Collect experience
    rollout = collect_rollouts(vec_env, n_steps)  # 16 envs × 2048 steps

    # 2. Compute advantages using GAE
    advantages = compute_gae(rollout, gamma=0.99, gae_lambda=0.95)

    # 3. Update policy (10 epochs)
    for epoch in range(10):
        for batch in minibatch_sampler(rollout, batch_size=256):
            # Compute ratio r = π_new / π_old
            ratio = exp(log_prob_new - log_prob_old)

            # Clipped objective
            loss_clip = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

            # Value function loss
            loss_vf = 0.5 * (V_pred - V_target)²

            # Entropy bonus
            loss_ent = -entropy(π_new)

            # Total loss
            loss = loss_clip + 0.5 * loss_vf + 0.01 * loss_ent

            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(parameters, max_norm=0.5)
            optimizer.step()

    # 4. Log metrics
    log_to_tensorboard(episode_rewards, policy_loss, value_loss)

    # 5. Save checkpoint (every 50k steps)
    if timestep % 50000 == 0:
        save_checkpoint(model, vec_env)
```

### Callbacks

Two callbacks used during training:

**1. CheckpointCallback**:
- Frequency: Every 50,000 timesteps
- Saves: `{experiment}_model_{timesteps}_steps.zip`
- Location: `runs/{experiment}/checkpoints/`
- Includes: Model parameters + VecNormalize statistics

**2. EvalCallback** (optional):
- Frequency: Every 100,000 timesteps
- Runs: 10 evaluation episodes on separate environment
- Saves: Best model based on mean reward
- Location: `runs/{experiment}/eval/`

### TensorBoard Logging

Metrics logged every 10 episodes:

- `rollout/ep_rew_mean`: Mean episode reward
- `rollout/ep_len_mean`: Mean episode length
- `train/policy_loss`: PPO clipped objective loss
- `train/value_loss`: Value function MSE
- `train/entropy_loss`: Policy entropy
- `train/explained_variance`: Fraction of return variance explained by value function
- `train/clip_fraction`: Fraction of updates triggering PPO clipping

**View logs**:
```bash
tensorboard --logdir runs/{experiment}/tb_logs/
```

## Ablation Framework

### Motivation

The ablation study systematically tests which uncertainty mechanisms are necessary for training robust policies. Starting from a deterministic baseline, mechanisms are added individually and in combination to identify:

1. **Critical mechanisms**: Which uncertainties significantly impact policy performance?
2. **Synergies**: Do certain combinations interact (positively or negatively)?
3. **Minimal configuration**: What is the simplest uncertainty setup that still trains effective policies?

### Experimental Design

**4 Stages, 10 Configurations**:

**Stage 0: Deterministic Baseline**
- `stage0_deterministic`: All uncertainty disabled
- Purpose: Verify basic learnability with perfect information

**Stage 1: Single Mechanisms**
- `stage1a_batch_only`: Only batch arrival
- `stage1b_stochastic_only`: Only stochastic duration (±20%)
- `stage1c_obs_noise_only`: Only observation noise (σ=15%)
- `stage1d_capacity_noise_only`: Only capacity noise (10% reduction)
- Purpose: Identify primary obstacles to learning

**Stage 2: Dual Combinations**
- `stage2a_batch_stochastic`: Batch arrival + stochastic duration
- `stage2b_obs_capacity`: Observation noise + capacity noise
- `stage2c_batch_obs`: Batch arrival + observation noise
- Purpose: Test mechanism interactions

**Stage 3: Full and Recommended**
- `stage3a_all_except_capacity`: Batch + stochastic + obs noise (recommended)
- `stage3b_current_all`: All 4 mechanisms (current setup)
- Purpose: Compare recommended minimal uncertainty vs full complexity

### Configuration Files

YAML files in `configs/ablation/`:

```yaml
# Example: stage1a_batch_only.yaml
experiment_name: stage1a_batch_only
description: 单一不确定性 - 仅批次到达（day 0/30/60 with ratios 0.4/0.35/0.25）
stage: 1

environment:
  use_batch_arrival: true
  stochastic_duration: false
  observation_noise: 0.0
  capacity_noise: 0.0

training:
  total_timesteps: 500000
  n_envs: 16

evaluation:
  test_regions:
    - Mataram
    - Sumbawa
    - Central Lombok
  crew_levels:
    - 0.3
    - 0.5
    - 0.7
    - 1.0
  n_seeds: 5
```

### Running Ablation Suite

**Sequential execution** (one experiment at a time):
```bash
cd src
python ablation_framework.py --run-all --parallel 1
```

**Parallel execution** (4 experiments simultaneously):
```bash
cd src
python ablation_framework.py --run-all --parallel 4
```

**HiPerGator batch submission**:
```bash
cd scripts
./submit_ablation_suite.sh
# Submits SLURM job array with 10 tasks
```

### Output Structure

```
experiments/ablation/
├── stage0_deterministic/
│   ├── logs/
│   │   ├── train.out
│   │   └── eval.out
│   └── results/
│       └── evaluation_results.csv
├── stage1a_batch_only/
│   └── ...
├── ...
├── comparison_summary.csv
└── ablation_report.md
```

**comparison_summary.csv** columns:
- `experiment`: Configuration name
- `stage`: 0-3
- `completion_rate_mean`: Mean completion rate across all test scenarios
- `completion_rate_std`: Standard deviation
- `avg_queue_time_mean`: Mean queue time in days
- `avg_queue_time_std`: Standard deviation
- `episode_reward_mean`: Mean episode reward
- `episode_reward_std`: Standard deviation

### Analysis Functions

**AblationSuite.compare_results()**:
- Aggregates results from all experiments
- Computes summary statistics (mean, std) grouped by experiment
- Saves to `comparison_summary.csv`
- Prints ranked comparison table

**AblationSuite.generate_report()**:
- Creates markdown report with:
  - Summary table
  - Stage descriptions
  - Key findings (best configuration)
- Saves to `ablation_report.md`

## Evaluation

### Evaluation Protocol

**Cross-scenario testing**:
- **Regions**: 3 test regions (Mataram, Sumbawa, Central Lombok)
- **Crew levels**: 4 levels (30%, 50%, 70%, 100% of base contractors)
- **Seeds**: 5 independent runs per scenario
- **Total scenarios**: 3 × 4 × 5 = 60 evaluation episodes

**Baseline comparisons**:
- **LJF** (Longest Job First): Prioritize houses with most remaining work
- **SJF** (Shortest Job First): Prioritize houses with least remaining work
- **Random**: Random allocation (not included by default)

### Evaluation Metrics

**Primary metrics**:
1. **Completion rate**: Fraction of houses finished within episode (range [0, 1])
2. **Average queue time**: Mean waiting time in days for all houses
3. **Episode reward**: Total cumulative reward (range typically [-500, +200])

**Secondary metrics**:
4. **Episode length**: Number of steps taken (≤500)
5. **Standard deviation**: Across 5 seeds, measures policy variance

### Running Evaluation

**Single experiment**:
```bash
cd src
python evaluate_ppo.py \
    --checkpoint-dir runs/my_experiment \
    --output-dir evaluation_results/my_experiment \
    --test-regions Mataram Sumbawa "Central Lombok" \
    --crew-levels 0.3 0.5 0.7 1.0 \
    --n-seeds 5 \
    --compare-baselines
```

**Ablation experiments** (after training):
```bash
cd src
python ablation_framework.py \
    --compare-only \
    --config-dir ../configs/ablation
```

**HiPerGator**:
```bash
cd scripts
sbatch evaluate.slurm runs/my_experiment evaluation_results/my_experiment
```

### Output Files

**evaluation_results.csv** (PPO only):

| region | crew_level | seed | completion_rate | avg_queue_time | episode_reward | episode_length |
|--------|-----------|------|----------------|---------------|---------------|---------------|
| Mataram | 0.5 | 0 | 0.892 | 45.3 | 124.5 | 500 |
| Mataram | 0.5 | 1 | 0.885 | 46.1 | 121.3 | 500 |
| ... | ... | ... | ... | ... | ... | ... |

**all_methods_comparison.csv** (with baselines):

| region | crew_level | seed | method | completion_rate | avg_queue_time | episode_reward | episode_length |
|--------|-----------|------|--------|----------------|---------------|---------------|---------------|
| Mataram | 0.5 | 0 | PPO | 0.892 | 45.3 | 124.5 | 500 |
| Mataram | 0.5 | 0 | LJF | 0.834 | 52.7 | 98.2 | 500 |
| Mataram | 0.5 | 0 | SJF | 0.841 | 51.4 | 102.1 | 500 |
| ... | ... | ... | ... | ... | ... | ... | ... |

### Interpreting Results

**Completion rate**:
- **>0.90**: Policy handles resource allocation effectively
- **0.70-0.90**: Moderate performance, room for improvement
- **<0.70**: Policy struggles, may need hyperparameter tuning or more training

**Average queue time**:
- **<40 days**: Queue management effective
- **40-60 days**: Acceptable under resource constraints
- **>60 days**: Queue buildup indicates allocation inefficiency

**Episode reward**:
- **>100**: Strong performance (high progress + completion, low queue penalties)
- **0-100**: Moderate performance
- **<0**: Poor performance (queue penalties dominate)

**Comparison to baselines**:
- PPO should outperform LJF/SJF on completion rate by 5-15%
- Queue time may be comparable or slightly better
- Episode reward should be significantly higher due to action-dependent progress reward

## File Structure

```
ppo/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── src/                               # Source code
│   ├── housegymrl.py                  # Core environment (2049 lines, 4 critical bugs fixed)
│   ├── config.py                      # Environment constants (M_FIXED=512, regions)
│   ├── baseline.py                    # Baseline policies (LJF, SJF, Oracle)
│   ├── synthetic_scenarios.py         # Scenario generation utilities
│   ├── ppo_configs.py                 # PPO hyperparameter configurations (155 lines)
│   ├── main_ppo.py                    # PPO training script (428 lines)
│   ├── evaluate_ppo.py                # Evaluation script (409 lines)
│   └── ablation_framework.py          # Ablation experiment framework (518 lines)
│
├── configs/                           # Configuration files
│   └── ablation/                      # Ablation YAML configs (10 files)
│       ├── stage0_deterministic.yaml
│       ├── stage1a_batch_only.yaml
│       ├── stage1b_stochastic_only.yaml
│       ├── stage1c_obs_noise_only.yaml
│       ├── stage1d_capacity_noise_only.yaml
│       ├── stage2a_batch_stochastic.yaml
│       ├── stage2b_obs_capacity.yaml
│       ├── stage2c_batch_obs.yaml
│       ├── stage3a_all_except_capacity.yaml
│       └── stage3b_current_all.yaml
│
├── scripts/                           # HiPerGator SLURM scripts
│   ├── README.md                      # Script usage documentation
│   ├── train.slurm                    # Single training job
│   ├── train_ablation.slurm           # Ablation job array
│   ├── evaluate.slurm                 # Evaluation job
│   ├── submit_ablation_suite.sh       # Batch submission helper
│   └── setup_env.sh                   # Environment setup
│
├── notebooks/                         # Jupyter notebooks
│   └── demo_notebook.ipynb            # Quick start demo (9 cells)
│
├── runs/                              # Training outputs (gitignored)
│   └── {experiment_name}/
│       ├── model.zip                  # Trained model
│       ├── vecnormalize.pkl           # Normalization statistics
│       ├── tb_logs/                   # TensorBoard logs
│       └── checkpoints/               # Periodic checkpoints
│
├── evaluation_results/                # Evaluation outputs (gitignored)
│   └── {experiment_name}/
│       ├── evaluation_results.csv
│       └── all_methods_comparison.csv
│
└── experiments/                       # Ablation outputs (gitignored)
    └── ablation/
        ├── {experiment_name}/
        │   ├── logs/
        │   └── results/
        ├── comparison_summary.csv
        └── ablation_report.md
```

## Usage Guide

### Training Custom Experiments

**Basic training**:
```bash
python main_ppo.py \
    --experiment-name custom_exp \
    --region Mataram \
    --timesteps 500000
```

**Disable batch arrival**:
```bash
python main_ppo.py \
    --experiment-name no_batch \
    --no-batch-arrival
```

**Disable stochastic duration**:
```bash
python main_ppo.py \
    --experiment-name deterministic_work \
    --no-stochastic
```

**Adjust observation noise**:
```bash
python main_ppo.py \
    --experiment-name low_noise \
    --obs-noise 0.05
```

**Adjust capacity noise**:
```bash
python main_ppo.py \
    --experiment-name fixed_capacity \
    --capacity-noise 0.0
```

**Combine flags**:
```bash
python main_ppo.py \
    --experiment-name minimal_uncertainty \
    --no-batch-arrival \
    --obs-noise 0.0 \
    --capacity-noise 0.0
```

**Resume from checkpoint**:
```bash
python main_ppo.py \
    --experiment-name resumed_exp \
    --resume-from runs/custom_exp/checkpoints/custom_exp_model_250000_steps.zip \
    --timesteps 1000000
```

### Evaluation Options

**Minimal evaluation** (single region, 1 seed):
```bash
python evaluate_ppo.py \
    --checkpoint-dir runs/my_exp \
    --test-regions Mataram \
    --crew-levels 1.0 \
    --n-seeds 1
```

**Full evaluation** (3 regions, 4 crew levels, 5 seeds, with baselines):
```bash
python evaluate_ppo.py \
    --checkpoint-dir runs/my_exp \
    --output-dir evaluation_results/my_exp \
    --test-regions Mataram Sumbawa "Central Lombok" \
    --crew-levels 0.3 0.5 0.7 1.0 \
    --n-seeds 5 \
    --compare-baselines
```

**Custom regions**:
```bash
python evaluate_ppo.py \
    --checkpoint-dir runs/my_exp \
    --test-regions "West Lombok" "East Lombok" \
    --crew-levels 0.5 1.0 \
    --n-seeds 3
```

### Ablation Experiments

**Run specific stages**:
```bash
# Manually specify configs
python ablation_framework.py \
    --config-dir ../configs/ablation
# Then edit the script to load only desired configs
```

**Run subset sequentially**:
```python
# In Python script
from ablation_framework import AblationSuite

suite = AblationSuite()
suite.add_experiment("../configs/ablation/stage0_deterministic.yaml")
suite.add_experiment("../configs/ablation/stage1a_batch_only.yaml")
suite.run_all_sequential()
suite.compare_results()
```

**Run in parallel** (4 jobs):
```bash
python ablation_framework.py \
    --run-all \
    --parallel 4
```

### TensorBoard Visualization

**Start TensorBoard**:
```bash
tensorboard --logdir runs/
```

**View specific experiment**:
```bash
tensorboard --logdir runs/my_experiment/tb_logs/
```

**Compare experiments**:
```bash
tensorboard --logdir runs/ --port 6006
```

**Key plots**:
- `rollout/ep_rew_mean`: Episode reward over time (should increase during training)
- `rollout/ep_len_mean`: Episode length (should stabilize at 500 or decrease if early completion)
- `train/policy_loss`: PPO loss (should decrease initially then stabilize)
- `train/explained_variance`: Value function accuracy (should approach 1.0)

## Configuration Reference

### EnvironmentConfig

```python
@dataclass
class EnvironmentConfig:
    M_min: int = 512                    # Min candidates (fixed for PPO)
    M_max: int = 512                    # Max candidates (fixed for PPO)
    use_batch_arrival: bool = True      # Enable batch arrival (day 0/30/60)
    stochastic_duration: bool = True    # Enable random work progress (±20%)
    observation_noise: float = 0.15     # Observation noise σ (fraction of true value)
    capacity_noise: float = 0.10        # Capacity reduction range (fraction)
    use_capacity_ramp: bool = False     # Enable capacity ramp (disabled)
    max_steps: int = 500                # Episode length limit
```

### PPOConfig

```python
@dataclass
class PPOConfig:
    learning_rate: float = 3e-4         # Adam learning rate
    n_steps: int = 2048                 # Steps per environment before update
    batch_size: int = 256               # Mini-batch size for SGD
    n_epochs: int = 10                  # Optimization epochs per update
    gamma: float = 0.99                 # Discount factor
    gae_lambda: float = 0.95            # GAE lambda (bias-variance tradeoff)
    clip_range: float = 0.2             # PPO clipping range ε
    ent_coef: float = 0.01              # Entropy coefficient
    vf_coef: float = 0.5                # Value function loss coefficient
    max_grad_norm: float = 0.5          # Gradient clipping threshold
    device: str = 'auto'                # 'auto', 'cpu', or 'cuda'
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    total_timesteps: int = 500_000      # Total training steps
    n_envs: int = 16                    # Number of parallel environments
    save_freq: int = 50_000             # Checkpoint save frequency (steps)
    eval_freq: int = 100_000            # Evaluation frequency (steps)
    log_interval: int = 10              # TensorBoard logging interval (episodes)
```

### Predefined Configurations

**PPO variants**:
- `PPO_DEFAULT`: Standard hyperparameters
- `PPO_FAST`: Faster updates (lr=5e-4, n_steps=1024, batch_size=128, n_epochs=5)
- `PPO_STABLE`: More stable (lr=1e-4, n_steps=4096, batch_size=512, n_epochs=15, clip=0.1)

**Training variants**:
- `TRAINING_DEFAULT`: 500k steps, 16 envs
- `TRAINING_QUICK`: 100k steps, 8 envs (for testing)
- `TRAINING_LONG`: 2M steps, 32 envs (for production)

**Environment variants**:
- `ENV_DEFAULT`: All uncertainty enabled (batch, stochastic, obs=0.15, capacity=0.10)
- `ENV_DETERMINISTIC`: All uncertainty disabled
- `ENV_MINIMAL_UNCERTAINTY`: Only batch + stochastic (obs=0.0, capacity=0.0)

## HiPerGator Deployment

### Prerequisites

- HiPerGator account with compute allocation
- Conda environment installed
- Project uploaded to `/home/yu.qianchen/ondemand/housegymrl_ppo`

### Setup Steps

```bash
# 1. SSH to HiPerGator
ssh your_username@hpg.rc.ufl.edu

# 2. Navigate to project
cd /home/yu.qianchen/ondemand/housegymrl_ppo/ppo

# 3. Load conda module
module load conda

# 4. Create environment (first time only)
conda create -n housegym_rl python=3.9
conda activate housegym_rl
pip install -r requirements.txt

# 5. Run setup script
cd scripts
bash setup_env.sh

# 6. Edit SLURM scripts
nano train.slurm
# Replace your_account and your_qos with your allocation
```

### Submitting Jobs

**Single training**:
```bash
cd scripts
sbatch train.slurm my_experiment 500000 Mataram
```

**Ablation suite**:
```bash
cd scripts
./submit_ablation_suite.sh
```

**Evaluation**:
```bash
cd scripts
sbatch evaluate.slurm runs/my_experiment evaluation_results/my_experiment
```

### Monitoring Jobs

**Check queue**:
```bash
squeue -u $USER
```

**View job details**:
```bash
squeue -j 123456
```

**Check logs** (while running):
```bash
tail -f logs/train_123456.out
```

**Cancel job**:
```bash
scancel 123456
```

### Resource Requests

**Training** (train.slurm):
- CPUs: 16 (matches n_envs for parallel rollouts)
- Memory: 32GB
- Time: 24 hours (500k steps ~10-16 hours)
- Partition: hpg-default

**Evaluation** (evaluate.slurm):
- CPUs: 4 (sequential evaluation)
- Memory: 16GB
- Time: 8 hours (60 scenarios ~3-5 hours)
- Partition: hpg-default

**Ablation array** (train_ablation.slurm):
- Array size: 10 (one per config)
- Per-task resources: Same as training
- Total resource usage: 10× training

## Troubleshooting

### Installation Issues

**Problem**: `ImportError: No module named gymnasium`
```bash
# Solution
pip install gymnasium
```

**Problem**: `ImportError: No module named stable_baselines3`
```bash
# Solution
pip install stable-baselines3
```

**Problem**: `ModuleNotFoundError: No module named 'housegymrl'`
```bash
# Solution: Ensure you're in the src/ directory
cd ppo/src
python main_ppo.py ...
```

### Training Issues

**Problem**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`
- **Cause**: Observation dimension mismatch (not 2054)
- **Solution**: Verify M_FIXED=512 in config.py, ensure VecNormalize uses correct stats

**Problem**: Training is very slow
- **Solution**: Use more parallel environments (`--n-envs 32`), or reduce n_steps
- **Check**: CPU usage (`htop`), ensure SubprocVecEnv is used (not DummyVecEnv)

**Problem**: Reward is not improving
- **Check**: TensorBoard logs (`rollout/ep_rew_mean`)
- **Solution**: Increase training time, adjust learning_rate (try 1e-4 or 5e-4)
- **Debug**: Run deterministic environment first (stage0_deterministic.yaml) to verify learnability

**Problem**: `FileNotFoundError: model.zip not found`
- **Cause**: Training did not complete successfully
- **Solution**: Check training logs for errors, ensure sufficient disk space

### Evaluation Issues

**Problem**: `FileNotFoundError: vecnormalize.pkl not found`
- **Cause**: VecNormalize stats not saved during training
- **Solution**: Re-train with `save_vecnormalize=True` in CheckpointCallback (default)

**Problem**: Evaluation results are worse than training
- **Cause**: Observation normalization stats mismatch
- **Solution**: Load VecNormalize stats from training (`vec_norm.obs_rms = training_stats`)

**Problem**: Baseline comparison fails
- **Cause**: BaselineEnv import error or incorrect policy name
- **Solution**: Verify `from housegymrl import BaselineEnv`, use policy="LJF" or "SJF"

### HiPerGator Issues

**Problem**: Job pending (PD) indefinitely
- **Check**: `squeue -j <jobid>` for reason
- **Common causes**: Invalid QOS, account over allocation, resource request too large
- **Solution**: Reduce resource request, check allocation with `sbank balance`

**Problem**: Job fails immediately
- **Check**: `logs/train_<jobid>.out` and `logs/train_<jobid>.err`
- **Common causes**: Conda environment not activated, wrong working directory
- **Solution**: Verify paths in SLURM script, ensure `module load conda; conda activate housegym_rl`

**Problem**: Out of memory error
- **Solution**: Increase `#SBATCH --mem=64gb`, reduce `--n-envs`

**Problem**: Timeout
- **Solution**: Increase `#SBATCH --time=48:00:00`, reduce `--timesteps`

## Development Notes

### Code Quality Standards

**Docstring format**:
- Function purpose in first line
- Args section with explicit types and ranges
- Returns section with shape/structure
- Example usage (optional but encouraged)

**Example**:
```python
def evaluate_on_scenario(
    model: PPO,
    vec_norm: VecNormalize,
    region: str,
    crew_level: float,
    n_seeds: int = 5
) -> pd.DataFrame:
    """
    Evaluate model on a single scenario across multiple seeds.

    Args:
        model: Trained PPO model instance.
        vec_norm: VecNormalize instance with training statistics.
        region: Region name (string, e.g., "Mataram").
        crew_level: Crew availability fraction (float in [0.1, 1.0]).
        n_seeds: Number of evaluation seeds (integer).

    Returns:
        DataFrame with n_seeds rows and columns:
        - seed: int
        - completion_rate: float (fraction in [0, 1])
        - avg_queue_time: float (days)
        - episode_reward: float
        - episode_length: int
    """
```

**Naming conventions**:
- Functions: lowercase_with_underscores
- Classes: CapitalizedWords
- Constants: UPPERCASE_WITH_UNDERSCORES
- Private methods: _leading_underscore

**Type hints**: Use wherever possible for clarity

**Comments**: Explain why, not what (code should be self-documenting)

### Testing

**Unit tests** (not yet implemented, recommended for future):
```bash
pytest src/test_housegymrl.py
pytest src/test_ablation_framework.py
```

**Integration test** (quick training run):
```bash
python main_ppo.py \
    --experiment-name test_run \
    --timesteps 10000 \
    --n-envs 4
```

**Smoke test** (verify imports):
```bash
python -c "from housegymrl import RLEnv, BaselineEnv; from ppo_configs import PPOConfig"
```

### Linting

**Check code style**:
```bash
flake8 src/ --max-line-length=120 --ignore=E501,W503
```

**Common issues**:
- Line length >120: Split into multiple lines
- Unused imports: Remove or comment out
- Missing whitespace: Add around operators

### Git Workflow

**Recommended** (not yet initialized):
```bash
git init
git add .
git commit -m "Initial PPO implementation with ablation framework"
git remote add origin <your_repo_url>
git push -u origin main
```

**.gitignore recommendations**:
```
runs/
evaluation_results/
experiments/
logs/
__pycache__/
*.pyc
*.pyo
*.pkl
*.zip
.ipynb_checkpoints/
```

## Migration from SAC

This PPO implementation fixes 4 critical bugs from the SAC version:

### Critical Issue #1: StaticArrival Reset Bug

**Problem** (SAC version):
```python
def reset(self):
    # StaticArrival object never reset
    # Second episode starts with empty queue
```

**Fix** (PPO version, housegymrl.py:1006-1008):
```python
elif isinstance(self.arrival_system, StaticArrival):
    # Recreate StaticArrival to ensure fresh queue on each episode
    self.arrival_system = StaticArrival(self.tasks_df)
```

### Critical Issue #2: Allocation Ignoring Remaining Work

**Problem** (SAC version):
```python
allocation = np.minimum(ideal_allocation, cmax_array)
# Over-allocates to nearly-complete houses
# House with 0.1 days remaining gets 5 crews allocated
```

**Fix** (PPO version, housegymrl.py:1104-1118):
```python
remaining_work_array = np.array([
    self._arr_rem[candidates[i]]
    for i in range(M)
], dtype=np.float64)

allocation = np.minimum(ideal_allocation, np.minimum(cmax_array, remaining_work_array))
# Respects remaining work constraint
```

### Critical Issue #3: Dynamic Progress Reward Denominator

**Problem** (SAC version):
```python
progress_reward = work_completed / sum(self._arr_rem)
# Denominator changes every step as houses complete
# Breaks temporal consistency: same work yields different rewards at t=0 vs t=100
```

**Fix** (PPO version, housegymrl.py:343, 677-679):
```python
# In __init__:
self._total_work_all = float(sum(self._arr_total))

# In _calculate_reward:
progress_reward = work_completed / self._total_work_all
# Fixed denominator ensures temporal consistency
```

### Critical Issue #4: VecNormalize Save/Load Path Inconsistency

**Problem** (SAC version):
```python
# Training saves to: runs/{experiment}/vecnormalize.pkl
# Evaluation loads from: {checkpoint_dir}/vecnormalize.pkl
# Paths don't match if checkpoint_dir != runs/{experiment}
```

**Fix** (PPO version, main_ppo.py:259, 316-317):
```python
save_dir = Path(f"runs/{experiment_name}")
model.save(str(save_dir / "model"))
vec_env.save(str(save_dir / "vecnormalize.pkl"))
# Consistent save/load paths
```

### Additional Improvements

**Deleted unused code**:
- `_build_action_mask()`: Action masking not used in continuous action space
- `get_candidate_seed_for_step()`: Seed generation unused
- `use_longterm_reward` parameter: Long-term reward not implemented
- `completed` parameter in `_calculate_reward()`: Unused

**Fixed M handling**:
- SAC version: M adaptive (reduces observation dimension over time)
- PPO version: M fixed at 512 (constant observation dimension)

**Improved documentation**:
- Explicit type hints and dimension specifications
- Clear examples in docstrings
- No vague terms ("available crews" → "integer, number of contractor crews ready")

---

**Total lines**: 1053

**Last updated**: 2025

**Maintainer**: HouseGym RL Team

**License**: Not specified

**Contact**: See repository issues for bug reports and feature requests
