#!/bin/bash
#SBATCH --job-name=ppo_train
#SBATCH --output=/home/yu.qianchen/ondemand/housegymrl/ppo/logs/train_%j.out
#SBATCH --error=/home/yu.qianchen/ondemand/housegymrl/ppo/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --partition=hpg-default
#SBATCH --account=vivian.wong
#SBATCH --qos=vivian.wong-b

# Single PPO training experiment on HiPerGator
# Usage: sbatch train.slurm [experiment_name] [timesteps] [region]

module purge
module load conda
conda activate urbanai

# Enable Python unbuffered output (important for real-time logging)
export PYTHONUNBUFFERED=1

# Set working directory to ppo/ (not ppo/src/)
cd /home/yu.qianchen/ondemand/housegymrl/ppo

# Parse arguments
EXPERIMENT_NAME=${1:-"ppo_default"}
TIMESTEPS=${2:-500000}
REGION=${3:-"Mataram"}

echo "=================================================="
echo "PPO Training Job"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: $EXPERIMENT_NAME"
echo "Timesteps: $TIMESTEPS"
echo "Region: $REGION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Working directory: $(pwd)"
echo "Started: $(date)"
echo "=================================================="

# Run training (from ppo/ directory)
python src/main_ppo.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --region "$REGION" \
    --timesteps "$TIMESTEPS" \
    --n-envs "$SLURM_CPUS_PER_TASK"

EXIT_CODE=$?

echo "=================================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "Finished: $(date)"
echo "=================================================="

exit $EXIT_CODE
