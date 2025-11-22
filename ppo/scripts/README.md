# SLURM Scripts for HiPerGator

This directory contains SLURM job scripts for running PPO training and evaluation on UF HiPerGator.

## Quick Start

1. **First-time setup**:
   ```bash
   cd /home/yu.qianchen/ondemand/housegymrl_ppo/ppo/scripts
   bash setup_env.sh
   ```

2. **Update SLURM parameters**: Edit `*.slurm` files and replace:
   - `your_account` with your HiPerGator account
   - `your_qos` with your QOS allocation

3. **Submit jobs**:
   ```bash
   # Single training experiment
   sbatch train.slurm my_experiment 500000 Mataram

   # Full ablation suite (10 experiments in parallel)
   ./submit_ablation_suite.sh

   # Evaluate trained model
   sbatch evaluate.slurm runs/my_experiment evaluation_results/my_experiment
   ```

## Script Descriptions

### train.slurm
Single PPO training experiment.

**Usage**:
```bash
sbatch train.slurm [experiment_name] [timesteps] [region]
```

**Arguments**:
- `experiment_name`: Unique identifier (default: "ppo_default")
- `timesteps`: Total training steps (default: 500000)
- `region`: Training region (default: "Mataram")

**Resources**:
- 16 CPUs, 32GB RAM
- 24-hour time limit
- Output: `logs/train_[jobid].out`

**Example**:
```bash
sbatch train.slurm baseline_experiment 1000000 Sumbawa
```

### train_ablation.slurm
Batch ablation experiments using job arrays.

**Usage**:
```bash
sbatch train_ablation.slurm
```

**Behavior**:
- Launches 10 parallel jobs (array indices 0-9)
- Each job trains one ablation configuration
- Configurations: stage0 through stage3b

**Resources**:
- 16 CPUs, 32GB RAM per job
- 24-hour time limit per job
- Output: `logs/ablation_[arrayid]_[taskid].out`

**Configurations**:
- `stage0_deterministic`: All uncertainty disabled
- `stage1a_batch_only`: Only batch arrival
- `stage1b_stochastic_only`: Only stochastic duration
- `stage1c_obs_noise_only`: Only observation noise (15%)
- `stage1d_capacity_noise_only`: Only capacity noise (10%)
- `stage2a_batch_stochastic`: Batch + stochastic
- `stage2b_obs_capacity`: Observation + capacity noise
- `stage2c_batch_obs`: Batch + observation noise
- `stage3a_all_except_capacity`: Recommended config
- `stage3b_current_all`: Full uncertainty

### evaluate.slurm
Model evaluation on test scenarios.

**Usage**:
```bash
sbatch evaluate.slurm [checkpoint_dir] [output_dir]
```

**Arguments**:
- `checkpoint_dir`: Directory containing model.zip (default: "runs/ppo_default")
- `output_dir`: Output directory for results (default: "evaluation_results")

**Resources**:
- 4 CPUs, 16GB RAM
- 8-hour time limit
- Output: `logs/eval_[jobid].out`

**Behavior**:
- Evaluates on 3 regions: Mataram, Sumbawa, Central Lombok
- Tests 4 crew levels: 30%, 50%, 70%, 100%
- Runs 5 seeds per scenario
- Compares with baselines (LJF, SJF)

**Example**:
```bash
sbatch evaluate.slurm runs/stage0_deterministic evaluation_results/stage0
```

### submit_ablation_suite.sh
Convenience script to submit all ablation experiments at once.

**Usage**:
```bash
./submit_ablation_suite.sh
```

**Behavior**:
- Submits `train_ablation.slurm` job array
- Prints evaluation commands for after training completes
- Creates logs directory if needed

### setup_env.sh
Environment setup and verification script.

**Usage**:
```bash
bash setup_env.sh
```

**Behavior**:
- Creates directory structure (logs, runs, evaluation_results, experiments)
- Verifies conda environment exists
- Checks Python dependencies
- Prints next steps

## Directory Structure

After setup and job execution:

```
ppo/
├── scripts/
│   ├── train.slurm
│   ├── train_ablation.slurm
│   ├── evaluate.slurm
│   ├── submit_ablation_suite.sh
│   ├── setup_env.sh
│   └── README.md (this file)
├── logs/
│   ├── train_123456.out
│   ├── ablation_123457_0.out
│   └── eval_123458.out
├── runs/
│   ├── stage0_deterministic/
│   │   ├── model.zip
│   │   ├── vecnormalize.pkl
│   │   ├── tb_logs/
│   │   └── checkpoints/
│   └── [other experiments]/
└── evaluation_results/
    └── [experiment]/
        ├── evaluation_results.csv
        └── all_methods_comparison.csv
```

## Monitoring Jobs

**Check job status**:
```bash
squeue -u $USER
```

**Check specific job**:
```bash
squeue -j 123456
```

**View job output** (while running):
```bash
tail -f logs/train_123456.out
```

**Cancel job**:
```bash
scancel 123456
```

**Cancel all your jobs**:
```bash
scancel -u $USER
```

**Job array commands**:
```bash
# Cancel specific task in array
scancel 123456_3

# Cancel entire array
scancel 123456
```

## Resource Usage Guidelines

### Training Jobs (train.slurm, train_ablation.slurm)
- **CPUs**: 16 (matches n_envs for parallel environment collection)
- **Memory**: 32GB (sufficient for vectorized environments + model)
- **Time**: 24 hours (500k timesteps ≈ 10-16 hours)

### Evaluation Jobs (evaluate.slurm)
- **CPUs**: 4 (evaluation is sequential, less CPU needed)
- **Memory**: 16GB
- **Time**: 8 hours (5 seeds × 12 scenarios ≈ 3-5 hours)

## Troubleshooting

**Problem**: Job fails immediately
- Check logs in `logs/` directory
- Verify conda environment: `conda env list`
- Verify working directory exists: `/home/yu.qianchen/ondemand/housegymrl_ppo`

**Problem**: Import errors
- Activate environment: `conda activate housegym_rl`
- Check dependencies: `pip list | grep stable-baselines3`
- Reinstall: `pip install -r ../requirements.txt`

**Problem**: Out of memory
- Reduce n_envs (edit YAML configs or use `--n-envs` flag)
- Request more memory in SLURM script: `#SBATCH --mem=64gb`

**Problem**: Job timeout
- Increase time limit: `#SBATCH --time=48:00:00`
- Reduce timesteps: Edit YAML configs or use `--timesteps` flag

**Problem**: Cannot find model.zip
- Check runs/ directory: `ls runs/[experiment_name]/`
- Verify training completed successfully: check logs
- Ensure correct checkpoint path in evaluate.slurm

## Advanced Usage

**Resume training from checkpoint**:
```bash
# Edit train.slurm to add --resume-from flag
python main_ppo.py \
    --experiment-name my_experiment \
    --resume-from runs/my_experiment/checkpoints/my_experiment_model_100000_steps.zip \
    --timesteps 1000000
```

**Custom ablation config**:
```bash
# Create custom YAML in configs/ablation/
# Update CONFIG_FILES array in train_ablation.slurm
# Adjust --array=0-N to match number of configs
```

**Parallel evaluation**:
```bash
# Submit multiple evaluation jobs for different experiments
for exp in stage0_deterministic stage1a_batch_only stage1b_stochastic_only; do
    sbatch evaluate.slurm runs/$exp evaluation_results/$exp
done
```

## Reference

- HiPerGator Documentation: https://help.rc.ufl.edu/doc/SLURM_Commands
- SLURM Job Arrays: https://slurm.schedmd.com/job_array.html
- Stable-Baselines3 Documentation: https://stable-baselines3.readthedocs.io/
