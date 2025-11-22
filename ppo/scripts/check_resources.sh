#!/bin/bash
# HiPerGator Resource Check Script
# Usage: bash check_resources.sh

echo "========================================================================"
echo "HiPerGator Resource Check for $USER"
echo "========================================================================"
echo ""

echo "=========================================="
echo "1. Account Information"
echo "=========================================="
sacctmgr show associations user=$USER format=account,qos%50,maxcpus,maxjobs,maxnodes -p | column -t -s '|'
echo ""

echo "=========================================="
echo "2. QOS Limits (Quality of Service)"
echo "=========================================="
sacctmgr show qos format=name,priority,maxwall,maxcpus,maxjobs,maxnodes where qos=vivian.wong,vivian.wong-b -p | column -t -s '|'
echo ""

echo "=========================================="
echo "3. Current Resource Usage"
echo "=========================================="
squeue -u $USER --format="%.10i %.9P %.30j %.8u %.2t %.10M %.6D %C %m"
echo ""
if [ $(squeue -u $USER | wc -l) -eq 1 ]; then
    echo "No jobs currently running"
fi
echo ""

echo "=========================================="
echo "4. Partition Information"
echo "=========================================="
echo "Showing CPU-based partitions:"
sinfo -p hpg-default,hpg-milan,hpg-rome --format="%.12P %.5a %.10l %.6D %.6t %.8c %.8m %.10C" | head -20
echo ""

echo "=========================================="
echo "5. Account Resource Allocation Summary"
echo "=========================================="
# Check total allocation
sshare -A vivian.wong --format=Account,User,RawShares,NormShares,RawUsage,EffectvUsage
echo ""

echo "=========================================="
echo "6. Recommended Configurations"
echo "=========================================="
echo "Based on your account: vivian.wong"
echo ""
echo "Standard QOS (vivian.wong):"
echo "  - Recommended CPUs: 8-16"
echo "  - Max walltime: Check above"
echo "  - Use for: Regular training jobs"
echo ""
echo "Burst QOS (vivian.wong-b):"
echo "  - Recommended CPUs: 16-32"
echo "  - Max walltime: Check above"
echo "  - Use for: High-priority/urgent jobs"
echo ""

echo "=========================================="
echo "7. Sample Job Submission Commands"
echo "=========================================="
echo "# Small test job (8 CPUs, 4 hours):"
echo "sbatch --account=vivian.wong --qos=vivian.wong --cpus-per-task=8 --mem=16gb --time=04:00:00 ppo/scripts/train.slurm"
echo ""
echo "# Standard training (16 CPUs, 24 hours) - CURRENT CONFIG:"
echo "sbatch --account=vivian.wong --qos=vivian.wong-b --cpus-per-task=16 --mem=32gb --time=24:00:00 ppo/scripts/train.slurm"
echo ""
echo "# Large training (32 CPUs, 48 hours):"
echo "sbatch --account=vivian.wong --qos=vivian.wong-b --cpus-per-task=32 --mem=64gb --time=48:00:00 ppo/scripts/train.slurm"
echo ""

echo "========================================================================"
echo "Resource Check Complete"
echo "========================================================================"
