#!/bin/bash
# Environment setup script for HiPerGator
# Run this once before submitting jobs

set -e

echo "=================================================="
echo "HouseGym RL PPO Environment Setup"
echo "=================================================="

# Create necessary directories
echo "Creating directory structure..."
mkdir -p logs
mkdir -p runs
mkdir -p evaluation_results
mkdir -p experiments/ablation

echo "Directory structure created:"
echo "  logs/                    - SLURM job logs"
echo "  runs/                    - Training outputs"
echo "  evaluation_results/      - Evaluation outputs"
echo "  experiments/ablation/    - Ablation experiment results"
echo ""

# Check conda environment
echo "Checking conda environment..."
if conda env list | grep -q "housegym_rl"; then
    echo "✓ Conda environment 'housegym_rl' found"
else
    echo "✗ Conda environment 'housegym_rl' not found"
    echo ""
    echo "To create the environment, run:"
    echo "  conda create -n housegym_rl python=3.9"
    echo "  conda activate housegym_rl"
    echo "  pip install -r ../requirements.txt"
    exit 1
fi

# Check Python dependencies
echo ""
echo "Checking key dependencies..."
conda activate housegym_rl

python -c "import gymnasium" 2>/dev/null && echo "✓ gymnasium" || echo "✗ gymnasium (install: pip install gymnasium)"
python -c "import stable_baselines3" 2>/dev/null && echo "✓ stable-baselines3" || echo "✗ stable-baselines3 (install: pip install stable-baselines3)"
python -c "import pandas" 2>/dev/null && echo "✓ pandas" || echo "✗ pandas"
python -c "import numpy" 2>/dev/null && echo "✓ numpy" || echo "✗ numpy"

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Update SLURM account/qos in *.slurm files"
echo "  2. Submit single training: sbatch train.slurm experiment_name"
echo "  3. Submit ablation suite: ./submit_ablation_suite.sh"
echo "  4. Monitor jobs: squeue -u \$USER"
echo ""
