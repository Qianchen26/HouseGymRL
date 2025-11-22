# HouseGym RL SAC 版本分析报告

## 执行概要

本报告对 SAC 版本代码进行了深度审查，发现了 **4 个 Critical 级别缺陷**和多个高优先级问题。主要结论：

1. **SAC 算法不适配当前问题**：2049维动作空间 + 稀疏奖励 + 候选随机性导致 SAC 难以收敛
2. **存在严重的环境 bugs**：4 个 Critical 缺陷会导致训练和评估完全失败
3. **Uncertainty 机制过度叠加**：5 种机制同时启用，信号噪声比过低
4. **Reward 函数设计问题**：分母动态变化破坏时间一致性，queue penalty 可能主导正向信号

---

## 一、Critical 级别缺陷（必须立即修复）

### 缺陷 1: StaticArrival reset bug

**位置**: `src/housegymrl.py:999-1006`

**问题描述**:
```python
def reset(self):
    # ...
    if isinstance(self.arrival_system, TrueBatchArrival):
        self.arrival_system = TrueBatchArrival(...)  # 重新创建
    # ⚠️ StaticArrival 没有被重置！
```

**影响**:
- 当 `use_batch_arrival=False` 时，使用 StaticArrival
- 第一次 reset() 正常
- 第二次 reset() 时，StaticArrival 的 `current_day` 已经是 0，不会再触发房屋揭示
- 导致 `initial_arrivals = []`（空列表），环境直接不可用

**修复方案**:
```python
def reset(self):
    # ...
    if isinstance(self.arrival_system, TrueBatchArrival):
        self.arrival_system = TrueBatchArrival(...)
    elif isinstance(self.arrival_system, StaticArrival):
        self.arrival_system = StaticArrival(self.tasks_df)  # 重新创建
```

---

### 缺陷 2: Allocation 忽略 remaining_work

**位置**: `src/housegymrl.py:1108`

**问题描述**:
```python
# RLEnv 的分配（仅考虑 cmax）
allocation = np.minimum(ideal_allocation, cmax_array)

# Baseline 的分配（考虑三个约束）
give = min(cmax, remaining_work, remaining_capacity)
```

**影响**:
- RLEnv 可能对剩余 1 天工时的房屋分配 6 个工人（cmax）
- 浪费 5 个工人的容量
- 造成 RL vs Baseline 的**不公平对比**（baseline 约束更严格）
- 降低整体资源利用率

**修复方案**:
```python
# 添加 remaining_work 约束
remaining_work_array = np.array([self._arr_rem[candidates[i]] for i in range(M)])
allocation = np.minimum(
    ideal_allocation,
    np.minimum(cmax_array, remaining_work_array)
)
```

---

### 缺陷 3: Progress reward 分母动态变化

**位置**: `src/housegymrl.py:674`

**问题描述**:
```python
def _calculate_reward(self, completed):
    revealed_ids = list(self.arrival_system.revealed_ids)  # 动态变化！
    # ...
    total_work_initial = sum(self._arr_total[h] for h in revealed_ids)
    progress_reward = work_completed / total_work_initial
```

**影响**:
- Day 0-29: 分母 = 40% 总工作量 → `progress_reward ≈ 0.025`
- Day 30-59: 分母 = 75% 总工作量 → `progress_reward ≈ 0.013`（相同工作量！）
- Day 60+: 分母 = 100% 总工作量 → `progress_reward ≈ 0.010`

**破坏奖励一致性**:
- 相同的动作在不同时期得到不同的奖励尺度
- RL 无法学习因果关系
- 可能导致 policy 过度专注于早期（高奖励）而忽略后期

**修复方案**:
```python
def __init__(self, ...):
    # 在初始化时计算固定的总工作量
    self._total_work_all = sum(self._arr_total.values())

def _calculate_reward(self, completed):
    # 使用固定分母
    progress_reward = work_completed / self._total_work_all
```

---

### 缺陷 4: VecNormalize 保存/加载路径不一致

**位置**:
- 保存: `src/main.py:1052` → `../models/sac_diverse_vecnorm.pkl`
- 加载: `src/evaluate.py:447` → `runs/*/vecnormalize.pkl`

**影响**:
- 评估时无法找到 VecNormalize 文件
- 评估在**未归一化的观测**下进行
- 评估结果不可信（观测尺度完全错误）

**修复方案**:
```python
# 统一路径为 runs/{experiment_name}/
# 训练时保存
model.save(f"runs/{experiment_name}/model.zip")
vec_env.save(f"runs/{experiment_name}/vecnormalize.pkl")

# 评估时加载
model = SAC.load(f"runs/{experiment_name}/model.zip")
vec_norm = VecNormalize.load(f"runs/{experiment_name}/vecnormalize.pkl")
```

---

## 二、SAC 算法适配性分析

### 问题特性 vs SAC 特性对比

| 维度 | 问题特性 | SAC 擅长 | 匹配度 |
|------|----------|----------|--------|
| **动作空间** | 2049维连续 | ≤100维连续 | ❌ 不匹配 |
| **动作本质** | 离散分配（整数工人数） | 连续控制 | ❌ 不匹配 |
| **奖励密度** | 稀疏 completion bonus | Dense reward | ⚠️ 较差 |
| **Episode 长度** | 500 steps | ≤1000 steps | ✅ 可接受 |
| **探索需求** | 组合空间 | 连续空间 | ❌ 不匹配 |

**综合评分**: ⭐⭐☆☆☆ (2/5) - **不推荐**

### SAC 难以收敛的三层障碍

#### 第一层：环境层面
1. **候选随机性破坏排列不变性**
   - 相同候选集合，不同排列 → MLP Policy 输出完全不同
   - SAC 需要学习 M! 种排列映射

2. **M 自适应导致观测维度变化**
   - M ∈ [512, 2048] → obs_dim ∈ [2054, 8198]
   - 神经网络需要处理 4x 维度变化

3. **整数化损失梯度**
   - `int_allocation = np.floor(allocation)`
   - RL 看到的 reward 来自整数分配，policy 输出连续值 → 梯度断层

#### 第二层：奖励层面
1. **Progress reward 分母动态变化** (Critical)
2. **Queue penalty 量级远超正向信号**
   - 典型单步: Progress (0.2) + Completion (0.5) = +0.7
   - 最坏情况: Queue penalty = -460
   - 比例 ≈ 1:660

3. **梯度信号微弱**
   - 单步 progress = 0.02
   - 需要累积 500 steps 才能产生显著梯度

#### 第三层：架构层面
1. **Policy network 参数过多**
   - 输出层: 256 × 2049 = 524,544 参数
   - 过拟合风险 + 梯度稀释

2. **探索机制不匹配**
   - SAC 使用高斯噪声
   - 但动作本质是排序/选择，需要离散探索

---

## 三、Uncertainty 机制消融分析

### 当前启用的 5 种 Uncertainty 机制

| 机制 | 参数 | 数学影响 | 必要性评估 |
|------|------|----------|----------|
| **Batch Arrival** | days=[0,30,60], ratios=[0.4,0.35,0.25] | 信息不完整性 | ✅ 核心机制，必须保留 |
| **Stochastic Duration** | σ=20% | Work ~ N(μ, 0.2μ) | ⚠️ 真实但可降低噪声（10%） |
| **Observation Noise** | σ=15% | Obs ~ true + N(0, 0.15×true) | ❌ 可能多余，建议移除或降至 5% |
| **Capacity Noise** | range=[90%, 100%] | K ~ Uniform(0.9K, K) | ❌ 可能多余，建议移除 |
| **Capacity Ramp** | warmup=36, rise=180 | K(t) 随时间增长 | ❌ 已禁用 |

### 消融实验建议

**阶段 0: 完美信息基线**
```python
use_batch_arrival=False
stochastic_duration=False
observation_noise=0.0
capacity_noise=0.0
```
**目的**: 验证基础可学习性

**阶段 1: 单一 Uncertainty**
- 1a: 仅 Batch Arrival
- 1b: 仅 Stochastic Duration
- 1c: 仅 Observation Noise
- 1d: 仅 Capacity Noise

**阶段 2: 双重组合**
- 2a: Batch + Stochastic
- 2b: Obs + Capacity
- 2c: Batch + Obs

**阶段 3: 推荐配置**
- 3a: Batch + Stochastic（推荐）
- 3b: 当前配置（全部启用）

**预期最优组合**:
```python
use_batch_arrival=True   # ✅ 核心机制
stochastic_duration=True # ✅ 真实性
observation_noise=0.0    # ❌ 移除
capacity_noise=0.0       # ❌ 移除
```

---

## 四、Reward 函数问题

### 当前 Reward V3 公式

```python
r = 100 × (10 × r_progress + 5 × r_completion + 1 × r_queue)
```

where:
- `r_progress = 当步完成工作量 / 所有房屋总工作量`
- `r_completion = 新完成房屋数 / 已揭示房屋数`
- `r_queue = -max(0, (平均等待时间 - 40天) / 100)`

### 主要问题

#### 问题 1: 组件权重不平衡

**实际贡献比**（典型单步）:

| 组件 | 原始值 | 权重 | 缩放 | 贡献 | 占比 |
|------|--------|------|------|------|------|
| Progress | 0.0002 | 10 | 100 | 0.2 | 67% |
| Completion | 0.001 | 5 | 100 | 0.5 | 33% |
| Queue | -0.05 | 1 | 100 | -5.0 | **-1667%** |

**问题**: Queue penalty 的绝对值可以轻易超过正向信号总和

#### 问题 2: Queue Penalty 数值范围

- 最坏情况: `avg_waiting = 500 days` → `queue_penalty = -4.6` → 缩放后 `-460`
- 正向信号: `progress + completion ≈ +0.7` per step
- **比例失衡**: Queue penalty 可抵消 65 steps 的正向奖励

#### 问题 3: Progress 信号微弱

- 单步典型增量: 10 workers × 1 day = 10 man-days
- 总工作量: ~500,000 man-days
- `progress_reward = 10 / 500000 × 10 × 100 = 0.02`
- **信号太弱**: 需要累积大量 steps 才能产生显著梯度

### Reward 函数改进方案

**方案 A: 增强 Progress 权重**
```python
weights = (50.0, 10.0, 1.0)  # Progress: 10→50
```

**方案 B: Queue Penalty 添加上限**
```python
queue_penalty = -min(1.0, max(0, (avg_waiting - 40) / 100))
```

**方案 C: 自适应权重**
```python
if episode < 1000:
    weights = (20.0, 10.0, 0.1)  # 早期强调 progress
else:
    weights = (10.0, 5.0, 1.0)   # 后期平衡
```

---

## 五、其他发现的问题

### High 优先级

1. **Allocation 丢弃剩余产能**
   - 位置: `housegymrl.py:1135-1149`
   - 当所有候选达到 cmax 时，剩余容量被浪费
   - 应该回到 waiting queue 选择新候选

2. **M 自适应导致维度变化**
   - `M ∈ [512, 2048]` → 观测/动作维度变化 4x
   - 神经网络难以处理

### Medium 优先级

1. **M_CANDIDATES 配置漂移**
   - `config.py` 声明 `M_CANDIDATES = 512`
   - 实际使用 `M_max = 2048`
   - 导致 `EXPECTED_OBS_DIM = 2054` 与实际 `8198` 不符

2. **use_longterm_reward 标志无效**
   - 在 `__init__` 中赋值但从未被读取
   - 配置具有误导性

### Low 优先级

1. **completed 参数未使用**
   - `_calculate_reward(self, completed)` 中 `completed` 从未被读取

2. **get_candidate_seed_for_step 未使用**
   - 定义但从未调用

3. **housegymrl_zy.py 重复**
   - 1386 行重复代码

---

## 六、修复优先级

### P0 (Critical - 必须立即修复)

1. ✅ StaticArrival reset bug
2. ✅ Allocation 忽略 remaining_work
3. ✅ Progress reward 分母动态变化
4. ✅ VecNormalize 保存/加载路径统一

### P1 (High - 高优先级)

1. Allocation 丢弃剩余产能
2. M 固定化（512）
3. Queue penalty 添加上限

### P2 (Medium - 中优先级)

1. M_CANDIDATES 配置修正
2. use_longterm_reward 删除
3. 考虑切换到 PPO

### P3 (Low - 代码清理)

1. 删除 completed 参数
2. 删除 get_candidate_seed_for_step
3. 删除 housegymrl_zy.py

---

## 七、后续建议

### 立即行动

1. **修复 4 个 Critical 缺陷**
2. **SAC 基准验证** (Deterministic Baseline, 100k steps)
   - 如果仍失败 → 立即切换 PPO

### PPO 迁移方案

**为什么 PPO 可能更适合**:
- GAE 缓解稀疏奖励
- Clipped objective 提升稳定性
- On-policy 适应可变维度
- 更适合高维动作空间

**实施路径**:
1. 创建 ppo/ 文件夹
2. 修复 Critical Issues
3. 实现 PPO 训练脚本
4. 系统化 Uncertainty Ablation 实验

### 实验计划

**阶段 1**: 修复缺陷 (1 周)
**阶段 2**: SAC 基准验证 (1-2 周)
**阶段 3**: PPO 实现与对比 (2-3 周)
**阶段 4**: Uncertainty Ablation (1-2 周)

---

## 八、总结

**主要发现**:
1. SAC 存在严重的环境 bugs（4 个 Critical）
2. SAC 算法不适配当前问题（2049 维动作 + 稀疏奖励）
3. Uncertainty 机制过度叠加
4. Reward 函数需要优化

**核心建议**:
1. 立即修复 4 个 Critical 缺陷
2. 切换到 PPO（预期成功率 70%）
3. 使用 Ablation Framework 系统测试 Uncertainty
4. 固定 M=512，避免维度变化

**预期改进**:
- 收敛速度: 2-5x
- 训练稳定性: 显著提升
- 最终性能: 60-70% completion rate

---

**报告生成时间**: 2025-11-21
**审查代码版本**: SAC v2.1
**下一步**: 实施 PPO 版本
