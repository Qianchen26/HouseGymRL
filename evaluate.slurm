#!/bin/bash
#SBATCH --job-name=housegym_eval
#SBATCH --output=/home/yu.qianchen/ondemand/housegymrl/artifacts/logs/eval_%j.out
#SBATCH --error=/home/yu.qianchen/ondemand/housegymrl/artifacts/logs/eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yu.qianchen@ufl.edu
#SBATCH --account=vivian.wong
#SBATCH --qos=vivian.wong
#SBATCH --partition=hpg-turin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#SBATCH --gpus=1

set -eo pipefail  # Removed -u to allow unbound variables in conda activation

echo "======================================================================"
echo "Evaluation job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "======================================================================"

module purge
module load conda

# Try to load CUDA (suppress errors if not available)
module load cuda/12.4 2>/dev/null || module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || echo "CUDA not loaded, will use CPU if available"

conda activate urbanai

BASE_DIR="/home/yu.qianchen/ondemand/housegymrl"
LOG_DIR="${BASE_DIR}/artifacts/logs"
mkdir -p "${LOG_DIR}" "${BASE_DIR}/results" "${BASE_DIR}/runs" "${BASE_DIR}/artifacts/results"

# Ensure PYTHONPATH is clean (avoid numpy import conflicts)
unset PYTHONPATH

# Change to src directory to run scripts (avoids PYTHONPATH issues)
cd "${BASE_DIR}/src"

# Diagnostic: Test numpy import
echo ""
echo "Environment Diagnostics:"
echo "  Python: $(which python)"
echo "  Working directory: $(pwd)"
if ! python -c "import numpy; print(f'  ✅ NumPy {numpy.__version__} loaded')" 2>&1; then
    echo ""
    echo "❌ ERROR: NumPy import failed on compute node!"
    echo "Please fix the conda environment on the login node before resubmitting."
    echo ""
    exit 1
fi
echo ""

DATASET_PATH="../results/synthetic_training_dataset.csv"
if [ ! -f "$DATASET_PATH" ]; then
    echo "Dataset not found; generating synthetic scenarios for evaluation context..."
    python synthetic_scenarios.py --seed 42 --output "$DATASET_PATH"
fi

echo "Running evaluation pipeline (evaluate.py)"
python evaluate.py

EXIT_CODE=$?

echo "======================================================================"
echo "Evaluation job finished on $(date) with exit code ${EXIT_CODE}"
echo "======================================================================"

if command -v sacct &>/dev/null; then
    sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,MaxVMSize,CPUTime
fi

exit $EXIT_CODE
