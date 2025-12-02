#!/bin/bash
#SBATCH --job-name=ppo_train
#SBATCH --output=/home/yu.qianchen/ondemand/housegymrl/ppo/logs/train_%j.out
#SBATCH --error=/home/yu.qianchen/ondemand/housegymrl/ppo/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120gb
#SBATCH --time=24:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --account=vivian.wong
#SBATCH --qos=vivian.wong

# Single PPO training experiment on HiPerGator
# Usage: sbatch train.slurm [experiment_name] [timesteps] [region]

module purge
module load conda
module load cuda/12.4.1
conda activate urbanai

# Enable Python unbuffered output (important for real-time logging)
export PYTHONUNBUFFERED=1

# Verify GPU availability
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Set working directory to ppo/ (not ppo/src/)
cd /home/yu.qianchen/ondemand/housegymrl/ppo

# Parse arguments
EXPERIMENT_NAME=${1:-"ppo_default"}
TIMESTEPS=${2:-1000000}
USE_SYNTHETIC=${3:-"true"}  # Default to synthetic training for robustness

echo "=================================================="
echo "PPO Training Job (GPU)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: $EXPERIMENT_NAME"
echo "Timesteps: $TIMESTEPS"
echo "Use Synthetic: $USE_SYNTHETIC"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Working directory: $(pwd)"
echo "Started: $(date)"
echo "=================================================="

# Calculate n_envs (leave 4 CPUs for main process)
N_ENVS=$((SLURM_CPUS_PER_TASK - 4))
echo "N_ENVS: $N_ENVS"

# Run training (from ppo/ directory)
if [ "$USE_SYNTHETIC" = "true" ]; then
    echo "Training mode: Multi-scenario (180 synthetic scenarios)"
    python src/main_ppo.py \
        --experiment-name "$EXPERIMENT_NAME" \
        --timesteps "$TIMESTEPS" \
        --n-envs "$N_ENVS" \
        --use-synthetic
else
    echo "Training mode: Single region (Mataram)"
    python src/main_ppo.py \
        --experiment-name "$EXPERIMENT_NAME" \
        --region "Mataram" \
        --timesteps "$TIMESTEPS" \
        --n-envs "$N_ENVS"
fi

EXIT_CODE=$?

echo "=================================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "Finished: $(date)"
echo "=================================================="

exit $EXIT_CODE
