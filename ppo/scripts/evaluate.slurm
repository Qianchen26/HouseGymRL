#!/bin/bash
#SBATCH --job-name=ppo_eval
#SBATCH --output=/home/yu.qianchen/ondemand/housegymrl/ppo/logs/eval_%j.out
#SBATCH --error=/home/yu.qianchen/ondemand/housegymrl/ppo/logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=08:00:00
#SBATCH --partition=hpg-default
#SBATCH --account=vivian.wong
#SBATCH --qos=vivian.wong-b

# PPO model evaluation on test scenarios
# Usage: sbatch evaluate.slurm [checkpoint_dir] [output_dir]

module purge
module load conda
conda activate urbanai

# Set working directory to ppo/ (not ppo/src/)
cd /home/yu.qianchen/ondemand/housegymrl/ppo

# Parse arguments
CHECKPOINT_DIR=${1:-"runs/ppo_default"}
OUTPUT_DIR=${2:-"evaluation_results"}

echo "=================================================="
echo "PPO Evaluation Job"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Working directory: $(pwd)"
echo "Started: $(date)"
echo "=================================================="

# Verify checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

if [ ! -f "$CHECKPOINT_DIR/model.zip" ]; then
    echo "ERROR: Model file not found: $CHECKPOINT_DIR/model.zip"
    exit 1
fi

# Run evaluation (from ppo/ directory)
python src/evaluate_ppo.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --test-regions Mataram Sumbawa "Central Lombok" \
    --crew-levels 0.3 0.5 0.7 1.0 \
    --n-seeds 5 \
    --compare-baselines

EXIT_CODE=$?

echo "=================================================="
echo "Evaluation completed with exit code: $EXIT_CODE"
echo "Results saved to: $OUTPUT_DIR"
echo "Finished: $(date)"
echo "=================================================="

exit $EXIT_CODE
