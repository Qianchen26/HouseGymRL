#!/bin/bash
# Fix conda environment on HiPerGator login node
# Run this BEFORE submitting SLURM jobs

echo "========================================="
echo "Fixing Conda Environment (urbanai)"
echo "========================================="

# Ensure we're on login node (not compute node)
if [[ "$SLURM_JOB_ID" != "" ]]; then
    echo "❌ ERROR: This script must be run on the LOGIN NODE, not inside a SLURM job!"
    echo "Please exit the SLURM job and run this on login8."
    exit 1
fi

echo ""
echo "Step 1: Activate conda environment"
echo "-----------------------------------"
conda activate urbanai || {
    echo "❌ Failed to activate urbanai environment"
    exit 1
}

echo ""
echo "Step 2: Check current Python and packages"
echo "-----------------------------------"
which python
python --version
echo ""
echo "Current package versions:"
pip list | grep -E "numpy|scipy|pandas|torch|stable-baselines3" || echo "  (packages not found)"

echo ""
echo "Step 3: Uninstall potentially broken packages"
echo "-----------------------------------"
pip uninstall -y numpy scipy

echo ""
echo "Step 4: Reinstall numpy and scipy with clean cache"
echo "-----------------------------------"
pip install --no-cache-dir numpy==2.0.2
pip install --no-cache-dir scipy

echo ""
echo "Step 5: Verify installation"
echo "-----------------------------------"
python -c "import numpy; print(f'✅ NumPy {numpy.__version__}')" || {
    echo "❌ NumPy still broken"
    exit 1
}
python -c "import scipy; print(f'✅ SciPy {scipy.__version__}')" || {
    echo "❌ SciPy still broken"
    exit 1
}
python -c "import pandas; print(f'✅ Pandas {pandas.__version__}')" || {
    echo "❌ Pandas still broken"
    exit 1
}

echo ""
echo "Step 6: Test full import chain (like main.py)"
echo "-----------------------------------"
python -c "
import numpy as np
import pandas as pd
from numpy.linalg import matrix_power
import torch
from stable_baselines3 import SAC
print('✅ All imports successful!')
print(f'   NumPy: {np.__version__}')
print(f'   Pandas: {pd.__version__}')
print(f'   PyTorch: {torch.__version__}')
" || {
    echo "❌ Full import test failed"
    exit 1
}

echo ""
echo "========================================="
echo "✅ Environment fixed successfully!"
echo "========================================="
echo ""
echo "You can now submit your SLURM jobs:"
echo "  sbatch train.slurm"
echo "  sbatch evaluate.slurm"
echo ""
