#!/bin/bash
#SBATCH --job-name=test_cpus
#SBATCH --output=/home/yu.qianchen/ondemand/housegymrl/ppo/logs/test_cpus_%j.out
#SBATCH --error=/home/yu.qianchen/ondemand/housegymrl/ppo/logs/test_cpus_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8gb
#SBATCH --time=00:30:00
#SBATCH --partition=hpg-default
#SBATCH --account=vivian.wong
#SBATCH --qos=vivian.wong

# Test CPU availability and performance
# Usage: sbatch test_cpus.slurm [num_cpus]

module purge
module load conda
conda activate urbanai

# Parse number of CPUs (default: SLURM allocation)
TEST_CPUS=${1:-$SLURM_CPUS_PER_TASK}

echo "========================================================================"
echo "CPU Test Job"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Testing with: $TEST_CPUS CPUs"
echo "Started: $(date)"
echo "========================================================================"
echo ""

cd /home/yu.qianchen/ondemand/housegymrl/ppo

# Run CPU test
python scripts/test_cpus.py

echo ""
echo "========================================================================"
echo "Test completed: $(date)"
echo "========================================================================"
