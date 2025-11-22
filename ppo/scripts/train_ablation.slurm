#!/bin/bash
#SBATCH --job-name=ppo_ablation
#SBATCH --output=/home/yu.qianchen/ondemand/housegymrl/ppo/logs/ablation_%A_%a.out
#SBATCH --error=/home/yu.qianchen/ondemand/housegymrl/ppo/logs/ablation_%A_%a.err
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --partition=hpg-default
#SBATCH --account=vivian.wong
#SBATCH --qos=vivian.wong-b

# Batch ablation experiments using SLURM job arrays
# Usage: sbatch train_ablation.slurm
# This will launch 10 parallel jobs, one for each ablation config

module purge
module load conda
conda activate urbanai

# Set working directory to ppo/ (not ppo/src/)
cd /home/yu.qianchen/ondemand/housegymrl/ppo

# Map array task ID to config file
CONFIG_FILES=(
    "stage0_deterministic"
    "stage1a_batch_only"
    "stage1b_stochastic_only"
    "stage1c_obs_noise_only"
    "stage1d_capacity_noise_only"
    "stage2a_batch_stochastic"
    "stage2b_obs_capacity"
    "stage2c_batch_obs"
    "stage3a_all_except_capacity"
    "stage3b_current_all"
)

CONFIG_NAME=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
CONFIG_PATH="configs/ablation/${CONFIG_NAME}.yaml"

echo "=================================================="
echo "PPO Ablation Experiment"
echo "=================================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Config: $CONFIG_NAME"
echo "Config file: $CONFIG_PATH"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Working directory: $(pwd)"
echo "Started: $(date)"
echo "=================================================="

# Verify config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Run training for this ablation experiment (from ppo/ directory)
python src/ablation_framework.py \
    --config-dir configs/ablation \
    --experiment "$CONFIG_PATH"

EXIT_CODE=$?

echo "=================================================="
echo "Experiment $CONFIG_NAME completed with exit code: $EXIT_CODE"
echo "Finished: $(date)"
echo "=================================================="

exit $EXIT_CODE
