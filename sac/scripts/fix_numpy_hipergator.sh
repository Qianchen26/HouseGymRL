#!/bin/bash
# 在HiPerGator上运行此脚本来修复numpy问题
# Usage: bash fix_numpy_hipergator.sh

echo "========================================="
echo "Numpy Import Issue - Comprehensive Fix"
echo "========================================="

cd /home/yu.qianchen/ondemand/housegymrl

echo ""
echo "Step 1: Check for conflicting files"
echo "-------------------------------------"
find . -maxdepth 2 -name "numpy*" -o -name "linalg*" 2>/dev/null | grep -v ".git" | grep -v "__pycache__"

echo ""
echo "Step 2: Check src directory"
echo "-------------------------------------"
ls -la src/ | grep -E "\.py$"

echo ""
echo "Step 3: Current conda environment"
echo "-------------------------------------"
conda activate urbanai
which python
python --version

echo ""
echo "Step 4: Check numpy installation"
echo "-------------------------------------"
python -c "import sys; print('Python path:'); print('\n'.join(sys.path[:3]))"

echo ""
echo "Step 5: Try importing numpy from BASE_DIR"
echo "-------------------------------------"
cd /home/yu.qianchen/ondemand/housegymrl
python -c "import numpy; print(f'SUCCESS: numpy {numpy.__version__}')" 2>&1

echo ""
echo "Step 6: Try importing numpy from src/"
echo "-------------------------------------"
cd /home/yu.qianchen/ondemand/housegymrl/src
python -c "import numpy; print(f'SUCCESS: numpy {numpy.__version__}')" 2>&1

echo ""
echo "Step 7: Reinstall numpy if needed"
echo "-------------------------------------"
read -p "Reinstall numpy? (y/n): " answer
if [ "$answer" = "y" ]; then
    pip install --force-reinstall --no-cache-dir numpy
    echo "Numpy reinstalled"
fi

echo ""
echo "========================================="
echo "Diagnostic complete"
echo "========================================="
