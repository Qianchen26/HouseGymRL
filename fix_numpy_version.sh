#!/bin/bash
# Fix numpy version incompatibility on HiPerGator
# Run this on HiPerGator: bash fix_numpy_version.sh

echo "========================================="
echo "Fixing Numpy Version Compatibility"
echo "========================================="

conda activate urbanai

echo ""
echo "Current versions:"
pip list | grep -E "numpy|scipy|pandas"

echo ""
echo "Installing compatible numpy version (2.0.2)..."
pip install "numpy>=2.0,<2.3"

echo ""
echo "Verifying installation:"
python -c "import numpy; import scipy; import pandas; print(f'✅ numpy {numpy.__version__}'); print(f'✅ scipy {scipy.__version__}'); print(f'✅ pandas {pandas.__version__}')"

echo ""
echo "Testing import from src directory:"
cd /home/yu.qianchen/ondemand/housegymrl/src
python -c "import numpy; print(f'✅ SUCCESS: numpy {numpy.__version__} works from src/')"

echo ""
echo "========================================="
echo "Fix complete! Now run: sbatch train.slurm"
echo "========================================="
