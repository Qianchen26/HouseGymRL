# 🚀 HiPerGator 训练快速开始

**完整设置只需要5步！**

---

## 📋 概览

| 步骤 | 在哪里执行 | 时间 | 命令 |
|------|----------|------|------|
| **0-2** | HiPerGator | 2分钟 | 清理和创建目录 |
| **3** | 本地Mac | 5分钟 | `bash upload.sh` |
| **4** | HiPerGator | 15分钟 | `bash setup_hpg.sh` |
| **5** | HiPerGator | 1分钟 | `sbatch train_sac.slurm` |

---

## 🎯 执行步骤

### Step 0-2: 清理环境（HiPerGator）

```bash
# SSH登录
ssh yu.qianchen@hpg.rc.ufl.edu

# 加载conda模块
module load conda

# 移除旧环境（如果存在）
conda env remove -n housegym -y

# 清理旧文件
cd /home/yu.qianchen/ondemand
rm -rf housegymrl

# 创建新目录结构
mkdir -p housegymrl/{code,data,logs,models/checkpoints,results,runs/sac_diverse}
cd housegymrl
ls -la  # 验证目录创建
```

**✅ 完成后断开SSH，回到本地Mac**

---

### Step 3: 上传文件（本地Mac）

```bash
cd /Users/qianchenyu/Documents/housegym_rl
bash upload.sh
```

**等待显示 "✅ UPLOAD COMPLETE!"**

---

### Step 4: 配置环境（HiPerGator）

```bash
# SSH回到HiPerGator
ssh yu.qianchen@hpg.rc.ufl.edu
cd /home/yu.qianchen/ondemand/housegymrl

# 运行环境配置（10-15分钟）
bash setup_hpg.sh
```

**等待显示 "✅ URBANAI ENVIRONMENT SETUP COMPLETE!"**

验证环境：
```bash
module load conda
conda activate urbanai
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"  # 应该是True
```

---

### Step 5: 提交训练（HiPerGator）

```bash
cd /home/yu.qianchen/ondemand/housegymrl
sbatch train_sac.slurm
```

记下Job ID：
```bash
squeue -u yu.qianchen
# 记下JOBID那一列的数字，例如: 12345678
```

**✅ 训练开始！预计4-6小时完成**

---

## 📊 监控训练进度

### 查看任务状态（HiPerGator）

```bash
ssh yu.qianchen@hpg.rc.ufl.edu

# 查看任务队列
squeue -u yu.qianchen

# 查看任务详情
scontrol show job <JOB_ID>
```

### 查看训练日志（HiPerGator）

```bash
ssh yu.qianchen@hpg.rc.ufl.edu
cd /home/yu.qianchen/ondemand/housegymrl

# 实时日志
tail -f logs/train_*.out

# 按Ctrl+C停止查看

# 查看最后100行
tail -100 logs/train_*.out

# 搜索错误
grep -i error logs/train_*.err
```

### 查看TensorBoard（可选）

**在HiPerGator上启动TensorBoard：**
```bash
ssh yu.qianchen@hpg.rc.ufl.edu
cd /home/yu.qianchen/ondemand/housegymrl
module load conda
conda activate urbanai

# 启动TensorBoard（后台）
tensorboard --logdir=runs --port=6006 --bind_all &
```

**在本地Mac建立SSH隧道：**
```bash
# 新终端窗口
ssh -L 6006:localhost:6006 yu.qianchen@hpg.rc.ufl.edu
```

**打开浏览器访问：** `http://localhost:6006`

---

## 📥 下载结果

### 训练完成后（4-6小时后）

在本地Mac运行：
```bash
cd /Users/qianchenyu/Documents/housegym_rl
bash download_results.sh
```

这会下载：
- ✅ 训练好的模型 → `models/`
- ✅ VecNormalize统计 → `models/`
- ✅ 训练数据集 → `results/`
- ✅ 训练日志 → `logs/`
- ✅ TensorBoard日志 → `runs/`（可选）

---

## 🎓 评估结果

下载完成后：

```bash
cd /Users/qianchenyu/Documents/housegym_rl

# 1. 检查模型文件
ls -lh models/

# 2. 查看训练日志末尾（确认完成）
tail logs/train_*.out

# 3. 启动Jupyter评估
jupyter notebook main.ipynb

# 在notebook中运行Cell 4.1+ 来对比SAC vs Baselines
```

---

## ⏱️ 时间表

| 阶段 | 时间 |
|------|------|
| **设置和上传** | ~20分钟 |
| **训练 (1M timesteps)** | 4-6小时 |
| **下载结果** | 5-10分钟 |
| **总计** | ~5-7小时 |

**建议：** 在下午提交训练，第二天早上下载结果并评估。

---

## 📁 核心文件

| 文件 | 用途 |
|------|------|
| `upload.sh` | 上传所有文件到HiPerGator |
| `setup_hpg.sh` | 在HiPerGator上配置urbanai环境 |
| `train_sac.slurm` | SLURM训练脚本 |
| `download_results.sh` | 下载训练结果 |
| `SETUP_INSTRUCTIONS.md` | 详细步骤说明 |

---

## 🐛 常见问题

### Q: 上传失败 "Connection refused"？
```bash
# 测试SSH连接
ssh yu.qianchen@hpg.rc.ufl.edu "echo 'OK'"
```

### Q: 目录创建失败？
```bash
# 确保不在要删除的目录内
cd /home/yu.qianchen
rm -rf ondemand/housegymrl
mkdir -p ondemand/housegymrl
```

### Q: conda activate失败？
```bash
# 重新初始化conda
module load conda
conda init bash
source ~/.bashrc
conda activate urbanai
```

### Q: CUDA不可用？
```bash
# 检查PyTorch（在GPU节点上）
python -c "import torch; print(torch.cuda.is_available())"

# 应该是True，如果是False：
module list  # 确认conda已加载
```

### Q: 如何知道训练完成？
查看日志末尾：
```bash
tail logs/train_*.out
```

应该看到：
```
====================================================================
Job finished on ...
Exit code: 0
====================================================================
✅ TRAINING COMPLETED!
```

### Q: 训练失败怎么办？
```bash
# 查看错误日志
cat logs/train_*.err

# 查看任务状态
sacct -j <JOB_ID>

# 常见原因：
# 1. CUDA不可用 → 检查SLURM脚本--gpus设置
# 2. 内存不足 → 减小batch_size或n_envs
# 3. 文件缺失 → 重新运行upload.sh
```

---

## ✅ 完整执行清单

- [ ] **HiPerGator**: 加载conda `module load conda`
- [ ] **HiPerGator**: 移除旧环境 `conda env remove -n housegym -y`
- [ ] **HiPerGator**: 清理旧文件 `rm -rf /home/yu.qianchen/ondemand/housegymrl`
- [ ] **HiPerGator**: 创建新目录结构
- [ ] **本地Mac**: 上传文件 `bash upload.sh`
- [ ] **HiPerGator**: 配置环境 `bash setup_hpg.sh`
- [ ] **HiPerGator**: 提交训练 `sbatch train_sac.slurm`
- [ ] **HiPerGator**: 记下Job ID
- [ ] **等待**: 4-6小时训练完成
- [ ] **本地Mac**: 下载结果 `bash download_results.sh`
- [ ] **本地Mac**: 评估 `jupyter notebook main.ipynb`

---

## 📞 获取帮助

- **详细文档**: [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md)
- **Debug文档**: [CLAUDE.md](CLAUDE.md)
- **HiPerGator支持**: support@rc.ufl.edu
- **SLURM文档**: https://help.rc.ufl.edu/doc/SLURM_Commands

---

## 🎯 现在就开始！

```bash
# 第一步 - 在HiPerGator上清理
ssh yu.qianchen@hpg.rc.ufl.edu

# 按照Step 0-2执行...
```

**完整步骤详见上方！** 🚀
