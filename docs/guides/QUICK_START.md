# 🚀 Quick Start Guide - Deploy & Train with Fixed Reward

**Status:** Ready to deploy
**Estimated time:** 5 minutes setup + 2-3 hours training

---

## ✅ What's been fixed

**Problem:** Urgency penalty accumulated across all houses → reward -5 to -23
**Fix:** Normalized penalty (avg + max) → reward -0.5 to +0.1

**Changes made:**
- [housegymrl.py:596-612](housegymrl.py#L596-L612) - Fixed reward function
- Added normalized average urgency penalty
- Added max wait penalty for long-tail prevention

---

## 🎯 One-Command Deployment

```bash
./deploy_and_train.sh
```

This will:
1. ✅ Test reward fix locally
2. ✅ Upload to HiPerGator
3. ✅ Submit training job
4. ✅ Start TensorBoard
5. ✅ Create SSH tunnel to localhost:37000

---

## 📊 TensorBoard Monitoring

### Quick Launch (if you already ran deployment)

```bash
# In a new terminal
./start_tensorboard_tunnel.sh
```

Then open: **http://localhost:37000**

### Key Metrics to Watch

| Metric | Before (Buggy) | After (Fixed) | What to look for |
|--------|----------------|---------------|------------------|
| **ep_rew_mean** | -4563 ❌ | -0.5 to +0.1 ✓ | Should be POSITIVE or near-zero |
| **Trend** | Flat ❌ | Increasing ✓ | Should go UP over time |
| **ep_len_mean** | 322 (flat) ❌ | Decreasing ✓ | Should go DOWN (faster completion) |

### Success Indicators (after ~50k steps)

- ✅ `ep_rew_mean` > -1.0 (ideally > 0)
- ✅ `ep_rew_mean` shows upward trend
- ✅ `ep_len_mean` < 300 days (and decreasing)
- ✅ No more flat curves

---

## 📋 Manual Steps (if needed)

### Step 1: Upload Code

```bash
scp housegymrl.py yu.qianchen@hpg.rc.ufl.edu:/home/yu.qianchen/ondemand/housegymrl/code/
```

### Step 2: Submit Job

```bash
ssh yu.qianchen@hpg.rc.ufl.edu
cd /home/yu.qianchen/ondemand/housegymrl
sbatch train_sac.slurm
```

### Step 3: Start TensorBoard on HiPerGator

```bash
# On HiPerGator
cd /home/yu.qianchen/ondemand/housegymrl
pkill -f tensorboard  # Kill old instances
nohup tensorboard --logdir=runs/sac_diverse --port=6006 --bind_all > logs/tensorboard.log 2>&1 &
```

### Step 4: SSH Tunnel to Local

```bash
# On your Mac (port 37000 as you mentioned)
ssh -N -L 37000:localhost:6006 yu.qianchen@hpg.rc.ufl.edu
```

### Step 5: Open TensorBoard

```
http://localhost:37000
```

---

## 🔍 Monitoring Training

### Check Job Status

```bash
ssh yu.qianchen@hpg.rc.ufl.edu 'squeue -u yu.qianchen'
```

### View Training Log

```bash
ssh yu.qianchen@hpg.rc.ufl.edu 'tail -f /home/yu.qianchen/ondemand/housegymrl/logs/train_*.out'
```

### Check Reward (quick check)

```bash
ssh yu.qianchen@hpg.rc.ufl.edu 'grep "ep_rew_mean" /home/yu.qianchen/ondemand/housegymrl/logs/train_*.out | tail -5'
```

---

## ⏱️ Timeline

| Time | What to expect |
|------|----------------|
| **0 min** | Job submitted |
| **5 min** | Job starts, initial setup |
| **30 min** | First metrics in TensorBoard (~25k steps) |
| **1 hour** | Clear trend visible (~100k steps) |
| **2-3 hours** | Training complete (500k steps) |

---

## 🎯 Expected Results

### Training Metrics (TensorBoard)

| Stage | ep_rew_mean | ep_len_mean |
|-------|-------------|-------------|
| Start (0-50k) | -0.5 to 0.0 | ~300 days |
| Middle (100k-300k) | 0.0 to +0.05 | ~280 days |
| End (500k) | +0.05 to +0.1 | ~260 days |

### Evaluation Results (after training)

| Region | Baseline (Random) | RL (Before) | RL (Expected) |
|--------|-------------------|-------------|---------------|
| **Mataram** | 214 days | 214 days ❌ | **~195 days** ✓ |
| **West Lombok** | 238 days | 238 days ❌ | **~220 days** ✓ |
| **North Lombok** | 413 days | 413 days ❌ | **~380 days** ✓ |

**Success criteria:**
- ✅ RL makespan < Random makespan by 5-10%
- ✅ RL approaches LJF/SJF performance in large regions

---

## 🚨 Troubleshooting

### Issue: Reward still negative (-5 or worse)

**Check:**
```bash
ssh yu.qianchen@hpg.rc.ufl.edu 'grep "avg_urgency_penalty" /home/yu.qianchen/ondemand/housegymrl/code/housegymrl.py'
```

**Fix:** Re-upload housegymrl.py
```bash
scp housegymrl.py yu.qianchen@hpg.rc.ufl.edu:/home/yu.qianchen/ondemand/housegymrl/code/
```

### Issue: TensorBoard not accessible

**Kill and restart:**
```bash
ssh yu.qianchen@hpg.rc.ufl.edu 'pkill -f tensorboard'
ssh yu.qianchen@hpg.rc.ufl.edu 'cd /home/yu.qianchen/ondemand/housegymrl && nohup tensorboard --logdir=runs/sac_diverse --port=6006 --bind_all &'
```

**Check tunnel:**
```bash
lsof -i:37000  # Should show SSH process
```

### Issue: Training not improving

**After 100k steps, if reward still flat:**
1. Check reward range is reasonable (-1 to +1, not -5 to -20)
2. Check entropy is decreasing (exploration → exploitation)
3. May need more training time (try 1M steps)

---

## 📞 Quick Commands Cheat Sheet

```bash
# Deploy everything
./deploy_and_train.sh

# Just start TensorBoard tunnel
./start_tensorboard_tunnel.sh

# Check job status
ssh yu.qianchen@hpg.rc.ufl.edu 'squeue -u yu.qianchen'

# View log (replace JOB_ID)
ssh yu.qianchen@hpg.rc.ufl.edu 'tail -f /home/yu.qianchen/ondemand/housegymrl/logs/train_JOB_ID.out'

# Kill training job (if needed)
ssh yu.qianchen@hpg.rc.ufl.edu 'scancel JOB_ID'

# Download results after training
./download_results.sh
```

---

## 🎉 Success Checklist

After deployment:
- [ ] Job is running (check with `squeue`)
- [ ] TensorBoard accessible at http://localhost:37000
- [ ] Can see "Scalars" tab with ep_rew_mean
- [ ] ep_rew_mean is NOT -4563 (should be > -2)

After 30 minutes:
- [ ] ep_rew_mean shows upward trend
- [ ] ep_len_mean shows downward trend
- [ ] No error messages in training log

After 2-3 hours:
- [ ] Training completed (500k steps)
- [ ] Final ep_rew_mean > -0.5 (ideally > 0)
- [ ] Ready for evaluation

---

**Good luck! 🚀**

Check TensorBoard in 30 minutes to see if reward is positive and increasing.
