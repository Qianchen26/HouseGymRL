#!/bin/bash
# Numpy environment diagnostic script
# Run this on HiPerGator to diagnose the numpy import issue

echo "========================================="
echo "Numpy Environment Diagnostic"
echo "========================================="

echo ""
echo "1. Current directory:"
pwd

echo ""
echo "2. Python location:"
which python

echo ""
echo "3. Python version:"
python --version

echo ""
echo "4. Check for numpy files in current directory:"
find . -maxdepth 2 -name "numpy*" -type f -o -name "numpy" -type d | grep -v "__pycache__" | grep -v ".git"

echo ""
echo "5. PYTHONPATH:"
echo "$PYTHONPATH"

echo ""
echo "6. Try importing numpy:"
python -c "import sys; print('sys.path:'); [print(f'  {p}') for p in sys.path[:5]]"

echo ""
echo "7. Check numpy installation:"
python -c "import sys; import os; print('Attempting to import numpy...'); import numpy; print(f'SUCCESS: numpy {numpy.__version__} from {numpy.__file__}')" 2>&1

echo ""
echo "8. Check for conflicting files in src/:"
ls -la src/ | grep -i numpy

echo ""
echo "========================================="
echo "Diagnostic complete"
echo "========================================="
