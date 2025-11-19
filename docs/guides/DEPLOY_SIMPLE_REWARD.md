# 🚀 Deploy Simplified Reward & Evaluate 7 Regions

## ✅ Changes Made

**Reward Function Simplified:**
```python
# OLD (Multi-component, caused learning issues)
total_reward = completion + queue_bonus + urgency_penalty + efficiency_bonus

# NEW (Simple, direct optimization)
reward = completion_reward  # = houses_completed / total_houses
```

**Location:** [housegymrl.py:559-643](housegymrl.py#L559-L643)

---

## ⚡ Quick Deploy Commands

### 1. Upload Updated Code

```bash
scp housegymrl.py yu.qianchen@hpg.rc.ufl.edu:/home/yu.qianchen/ondemand/housegymrl/code/
```

### 2. Verify Upload

```bash
ssh yu.qianchen@hpg.rc.ufl.edu 'grep -A 5 "SIMPLIFIED REWARD" /home/yu.qianchen/ondemand/housegymrl/code/housegymrl.py'
```

应该看到：
```
# SIMPLIFIED REWARD: Return only completion reward
# (Multi-component reward commented out below for future reference)
return completion_reward
```

### 3. Submit Training Job

```bash
ssh yu.qianchen@hpg.rc.ufl.edu 'cd /home/yu.qianchen/ondemand/housegymrl && sbatch train_sac.slurm && squeue -u yu.qianchen'
```

---

## 📊 Monitor Training (2-3 hours)

### TensorBoard (Real-time)

Already running at: **http://localhost:37000**

**Watch for:**
- **ep_rew_mean**: Should be small positive (0.001 - 0.01)
  - ❌ Before: 0.77 (dominated by efficiency bonus)
  - ✅ Now: ~0.005 (pure completion signal)
- **ep_len_mean**: Should decrease over time
  - Target: < current (322 days)
  - Hope: approach 250-280 days

### Check Job Status

```bash
ssh yu.qianchen@hpg.rc.ufl.edu 'squeue -u yu.qianchen'
```

### View Training Log

```bash
ssh yu.qianchen@hpg.rc.ufl.edu 'tail -f /home/yu.qianchen/ondemand/housegymrl/logs/train_*.out'
```

---

## 📋 After Training Completes

### Option A: Evaluate All Regions (Recommended)

**7 Regions to test:**
1. Mataram
2. West Lombok
3. Central Lombok
4. East Lombok
5. North Lombok
6. Sumbawa
7. West Sumbawa

**Evaluation script** (check if `evaluate.py` supports your regions):

```bash
ssh yu.qianchen@hpg.rc.ufl.edu << 'EOF'
cd /home/yu.qianchen/ondemand/housegymrl
module load conda
conda activate urbanai
export PYTHONPATH="${PWD}/code:${PYTHONPATH}"

# Evaluate on all regions
for region in "Mataram" "West Lombok" "Central Lombok" "East Lombok" "North Lombok" "Sumbawa" "West Sumbawa"; do
    echo "Evaluating: $region"
    python code/evaluate.py \
        --region "$region" \
        --model-path models/sac_final.zip \
        --n-episodes 10 \
        --output results/
done

echo "Done! Results in results/"
EOF
```

### Option B: Quick Test on Mataram Only

```bash
ssh yu.qianchen@hpg.rc.ufl.edu << 'EOF'
cd /home/yu.qianchen/ondemand/housegymrl
module load conda
conda activate urbanai
export PYTHONPATH="${PWD}/code:${PYTHONPATH}"

python code/evaluate.py \
    --region "Mataram" \
    --model-path models/sac_final.zip \
    --n-episodes 10

EOF
```

---

## 🎯 Success Criteria

### Primary Goal (Hypothesis 1)
✅ **RL is a usable method**
- RL should NOT be worst in ALL regions
- Expect: RL ranks 2-3 out of 4 in some regions

### Example Success Pattern:
| Region | LJF | SJF | Random | RL | RL Rank |
|--------|-----|-----|--------|----|----|
| Mataram | 206 | 222 | 214 | **240** | 3/4 ✓ |
| West Lombok | 233 | 246 | 238 | **230** | 🥇 ✓✓ |
| North Lombok | 413 | 426 | 413 | **420** | 3/4 ✓ |
| ... | | | | | |

→ **Conclusion**: RL is usable (not always worst)

### Secondary Goal (Hypothesis 2 - Exploratory)
✅✅ **RL has specific advantages**
- RL performs best in certain region types
- Example patterns to look for:
  - Large regions (H > 50k): RL > others
  - High uncertainty: RL more robust
  - Specific damage distributions: RL adapts better

---

## 📥 Download Results

```bash
scp yu.qianchen@hpg.rc.ufl.edu:/home/yu.qianchen/ondemand/housegymrl/results/*.csv results/
```

---

## 🔍 Analyze Results

Look for:

1. **Where is RL NOT worst?**
   - Count regions where RL rank < 4

2. **Any pattern in RL advantages?**
   - Region size correlation?
   - Damage distribution correlation?
   - Uncertainty level correlation?

3. **Long-tail issues?**
   - Check t95 - t90 gap
   - If > 20 days consistently → consider adding long-tail prevention later

---

## ⏱️ Timeline

| Time | Checkpoint |
|------|-----------|
| **Now** | Upload code ✓ |
| **+5 min** | Training starts |
| **+30 min** | Check TensorBoard (ep_rew_mean should be small positive) |
| **+2-3 hours** | Training complete |
| **+3-4 hours** | All 7 regions evaluated |
| **+4 hours** | Results analysis → Decision on long-tail prevention |

---

## 🚨 Troubleshooting

### If ep_rew_mean is still 0.77
- Code didn't update → re-upload
- Check: `grep "return completion_reward" housegymrl.py`

### If training OOMs again
- Should not happen (already 100GB)
- If it does, reduce N_ENVS to 4

### If ep_len_mean doesn't improve
- Wait for full 500k steps
- If still > 300 at end → need deeper investigation

---

**Ready to deploy? Run the commands above! 🚀**
