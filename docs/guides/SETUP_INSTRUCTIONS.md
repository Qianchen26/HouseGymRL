# HiPerGator 完整设置指令

从头开始的完整步骤。

---

## Step 0: 清理旧环境（HiPerGator）

SSH登录HiPerGator：
```bash
ssh yu.qianchen@hpg.rc.ufl.edu
```

加载conda并移除旧环境：
```bash
# 加载conda模块
module load conda

# 查看现有环境
conda env list

# 移除housegym环境（如果存在）
conda env remove -n housegym -y

# 确认已删除
conda env list
```

---

## Step 1: 清理旧文件（HiPerGator）

在HiPerGator上继续：
```bash
# 进入ondemand目录
cd /home/yu.qianchen/ondemand

# 删除旧的housegymrl目录（如果存在）
rm -rf housegymrl

# 确认已删除
ls -la
```

---

## Step 2: 创建新目录结构（HiPerGator）

```bash
# 创建主目录
mkdir -p /home/yu.qianchen/ondemand/housegymrl

# 创建子目录
cd /home/yu.qianchen/ondemand/housegymrl
mkdir -p code data logs models/checkpoints results runs/sac_diverse

# 验证结构
tree -L 2 .
# 或者
ls -la
ls -la models/
ls -la runs/
```

应该看到：
```
/home/yu.qianchen/ondemand/housegymrl/
├── code/
├── data/
├── logs/
├── models/
│   └── checkpoints/
├── results/
└── runs/
    └── sac_diverse/
```

---

## Step 3: 上传文件（本地Mac）

在本地Mac打开终端：
```bash
cd /Users/qianchenyu/Documents/housegym_rl
```

运行上传脚本（我会创建一个完整的上传脚本）：
```bash
bash upload.sh
```

这会上传：
- ✅ Python代码 → `code/`
- ✅ 数据文件 → `data/`
- ✅ 环境配置 → `setup_hpg.sh`
- ✅ 训练脚本 → `train_sac.slurm`
- ✅ 下载脚本 → `download_results.sh`

---

## Step 4: 配置urbanai环境（HiPerGator）

SSH回到HiPerGator：
```bash
ssh yu.qianchen@hpg.rc.ufl.edu
cd /home/yu.qianchen/ondemand/housegymrl
```

运行环境配置脚本：
```bash
bash setup_hpg.sh
```

等待10-15分钟，直到看到：
```
✅ URBANAI ENVIRONMENT SETUP COMPLETE!
```

验证环境：
```bash
module load conda
conda activate urbanai

# 检查Python和包
python --version  # 应该是 Python 3.11.x
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"  # 应该是 True
python -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')"
```

---

## Step 5: 提交训练任务（HiPerGator）

```bash
cd /home/yu.qianchen/ondemand/housegymrl
sbatch train_sac.slurm
```

获取任务ID：
```bash
squeue -u yu.qianchen
```

你会看到类似：
```
JOBID    PARTITION  NAME           USER         ST  TIME  NODES
12345678 gpu        housegym_sac   yu.qianchen  R   0:30  1
```

---

## Step 6: 监控训练（HiPerGator）

### 查看实时日志
```bash
# 替换<JOB_ID>为你的任务ID
tail -f logs/train_<JOB_ID>.out
```

按 `Ctrl+C` 停止查看。

### 查看错误日志
```bash
tail -f logs/train_<JOB_ID>.err
```

### 检查任务状态
```bash
squeue -u yu.qianchen
```

### 查看GPU使用情况
```bash
# 如果任务正在运行
srun --jobid=<JOB_ID> --pty nvidia-smi
```

---

## Step 7: 访问TensorBoard（两种方式）

### 方式A: HiPerGator上启动TensorBoard（推荐）

在HiPerGator上：
```bash
module load conda
conda activate urbanai
cd /home/yu.qianchen/ondemand/housegymrl

# 启动TensorBoard（后台运行）
tensorboard --logdir=runs --port=6006 --bind_all &

# 记下TensorBoard进程ID
echo $!
```

然后在本地Mac建立SSH隧道：
```bash
ssh -L 6006:localhost:6006 yu.qianchen@hpg.rc.ufl.edu
```

打开浏览器访问：`http://localhost:6006`

### 方式B: 下载后本地查看

等训练完成后，下载runs目录到本地，然后：
```bash
cd /Users/qianchenyu/Documents/housegym_rl
tensorboard --logdir=runs
```

---

## Step 8: 下载结果（本地Mac）

### 手动下载

在本地Mac：
```bash
cd /Users/qianchenyu/Documents/housegym_rl
bash download_results.sh
```

这会下载：
- ✅ 训练好的模型 → `models/`
- ✅ VecNormalize统计 → `models/`
- ✅ 合成数据集 → `results/`
- ✅ TensorBoard日志 → `runs/`
- ✅ 训练日志 → `logs/`

### 自动下载（可选）

我会创建一个脚本，在训练完成后自动下载结果。

---

## Step 9: 本地评估（本地Mac）

下载完成后：
```bash
cd /Users/qianchenyu/Documents/housegym_rl
jupyter notebook main.ipynb
```

运行评估cells（Cell 4.1+）来对比SAC vs Baselines。

---

## 📋 完整命令清单

### HiPerGator命令（依次执行）
```bash
# 登录
ssh yu.qianchen@hpg.rc.ufl.edu

# 加载conda模块
module load conda

# 清理旧环境
conda env remove -n housegym -y

# 清理旧文件
cd /home/yu.qianchen/ondemand
rm -rf housegymrl

# 创建新结构
mkdir -p housegymrl/code housegymrl/data housegymrl/logs housegymrl/models/checkpoints housegymrl/results housegymrl/runs/sac_diverse
cd housegymrl

# 等待文件上传完成...

# 配置环境
bash setup_hpg.sh

# 提交训练
sbatch train_sac.slurm

# 监控
squeue -u yu.qianchen
tail -f logs/train_*.out
```

### 本地Mac命令（依次执行）
```bash
# 上传文件
cd /Users/qianchenyu/Documents/housegym_rl
bash upload.sh

# 等待训练完成...

# 下载结果
bash download_results.sh

# 评估
jupyter notebook main.ipynb
```

---

## ⏱️ 预计时间

| 步骤 | 时间 |
|------|------|
| Step 0-2: 清理和创建目录 | 2分钟 |
| Step 3: 上传文件 | 2-5分钟 |
| Step 4: 配置环境 | 10-15分钟 |
| Step 5: 提交任务 | 1分钟 |
| Step 6-7: 训练（1M timesteps） | 4-6小时 |
| Step 8: 下载结果 | 5-10分钟 |
| Step 9: 评估 | 按需 |
| **总计** | **~5-7小时** |

---

## 🆘 问题排查

### Q: conda env remove报错？
```bash
# 先deactivate
conda deactivate
# 再删除
conda env remove -n housegym -y
```

### Q: 目录删除失败（权限问题）？
```bash
# 检查当前位置
pwd
# 应该不在housegymrl目录内
cd /home/yu.qianchen/ondemand
rm -rf housegymrl
```

### Q: 上传文件失败？
```bash
# 检查SSH连接
ssh yu.qianchen@hpg.rc.ufl.edu "echo 'Connection OK'"

# 检查目标目录存在
ssh yu.qianchen@hpg.rc.ufl.edu "ls -la /home/yu.qianchen/ondemand/housegymrl"
```

### Q: TensorBoard无法访问？
```bash
# 确保SSH隧道正在运行
ssh -L 6006:localhost:6006 yu.qianchen@hpg.rc.ufl.edu

# 在另一个终端查看TensorBoard是否运行
ssh yu.qianchen@hpg.rc.ufl.edu "ps aux | grep tensorboard"
```

---

## 🎯 现在开始

**执行顺序**：

1. 在HiPerGator上执行 Step 0-2（清理和创建目录）
2. 在本地Mac上执行 Step 3（上传文件）
3. 在HiPerGator上执行 Step 4-5（配置和训练）
4. 等待训练完成（4-6小时）
5. 在本地Mac上执行 Step 8（下载结果）
6. 在本地Mac上执行 Step 9（评估）

准备好了吗？从Step 0开始！🚀
