#!/bin/bash
# Submit all ablation experiments as a batch
# This script submits 10 parallel training jobs followed by evaluation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "Submitting Ablation Suite"
echo "=================================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Submit training job array
echo "Submitting training job array (10 experiments)..."
TRAIN_JOB_ID=$(sbatch --parsable train_ablation.slurm)
echo "Training job array submitted: $TRAIN_JOB_ID"

# Wait for training to complete, then submit evaluation
echo ""
echo "To evaluate all experiments after training completes, run:"
echo "  cd ../src"
echo "  python ablation_framework.py --compare-only"
echo ""
echo "Or submit evaluation jobs individually:"
for config in stage0_deterministic stage1a_batch_only stage1b_stochastic_only \
              stage1c_obs_noise_only stage1d_capacity_noise_only \
              stage2a_batch_stochastic stage2b_obs_capacity stage2c_batch_obs \
              stage3a_all_except_capacity stage3b_current_all; do
    echo "  sbatch evaluate.slurm runs/$config evaluation_results/$config"
done

echo ""
echo "=================================================="
echo "Job submission complete"
echo "Monitor jobs with: squeue -u \$USER"
echo "=================================================="
