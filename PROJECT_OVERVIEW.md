# HouseGym RL 项目说明（提交给 ChatGPT Deep Research）

## 1. 背景

- **研究场景**：模拟灾后房屋恢复调度（以印度尼西亚 Lombok 行政区数据为原型），探索在资源波动、评估滞后的复杂情境下应如何分配施工队伍。
- **目标**：构建一套可重复的仿真与评估流水线，用强化学习（SAC）与固定策略（LJF/SJF/Random）进行对比，分析不同机制配置下的恢复进度与公平性。我们并不预设 “RL 一定更优”，重点是理解各机制与策略之间的行为差异，并为后续研究提供可以扩展的框架。
- **当前成果**：
  - 已完成多次 SAC 训练（`scripts/cluster/train_sac_hpg.py`，timesteps=1M）。
  - 评估脚本 `src/housegymrl/evaluation.py` 会生成 `artifacts/results/complete_results_*.csv`，输出 RL 与 baseline 在 7 个 region × 若干 availability 的表现。
  - 近期改动：batch arrival 采用混合分批、capacity ramp 默认关闭，噪声与 reward 设置保持训练/评估一致。

---

## 2. 代码结构与机制设定（含函数/变量关系与 rationale）

### 2.1 总体结构

```
housegym_rl/
├── src/housegymrl/
│   ├── housegymrl.py          # BaseEnv/RLEnv/BaselineEnv + 环境机制
│   ├── baseline.py            # 基线策略 CLI + rollouts
│   ├── config.py              # Region 配置、batch arrival/候选池参数
│   ├── evaluate.py            # 单区域评估与可视化
│   └── evaluation.py           # 全区域批量评估 (CLI)
├── scripts/cluster/
│   ├── train_sac_hpg.py       # SAC 训练主程序
│   ├── train_sac.slurm        # HiPerGator 训练作业
│   └── evaluate_sac.slurm     # HiPerGator 评估作业
├── tests/                     # 单元/集成测试
├── artifacts/                 # 模型、VecNormalize、TensorBoard、结果 CSV
└── docs/guides/               # 操作说明 (START_HERE, TensorBoard 等)
```

### 2.2 核心模块与函数

| 模块/函数 | 关键输入 | 关键输出 | 说明与 rationale |
|-----------|----------|----------|-------------------|
| `BaseEnv.__init__(region_key,..., use_capacity_ramp=False, observation_noise, capacity_noise, max_steps)` (`housegymrl.py`) | region 配置、机制开关、噪声参数 | 初始化后的环境实例；内部持有 `arrival_system`, `capacity_system`, 任务数组等 | 统一管理 batch arrival / capacity / noise；默认关闭 ramp 以贴近现实“快速恢复产能”，也避免在 500 天截断下无法完成任务。 |
| `TrueBatchArrival(tasks_df, config, seed)` | 原始任务 DataFrame；配置 `days=[0,20,40,60]`, `ratios=[0.30,0.30,0.25,0.15]` | `self.schedule: day -> list[house_id]` | 先按 damage level 打乱，再按比例切分并在每批混合；保证总量与原数据一致，同时模拟分批揭示。 |
| `_execute_allocation(allocation)` | `{house_id: workers}` 分配 | `(obs, reward, terminated, truncated, info)` | 核心 step：应用工作量（含随机噪声）→ 更新完成率 → `reward = _calculate_reward(done)` → 返回 observation。`info["completion"]` 用于 TensorBoard / 评估。 |
| `_calculate_reward(done)` | `done` flag | 标量 reward | 包含 progress reward、time penalty、longtail penalty、terminal makespan/longtail 加成、未完成惩罚。progress 使用 `self.completion_ratio - self._last_completion_ratio`，longtail 利用 `completion_times` 的分位数。 |
| `RLEnv` / `BaselineEnv` | 继承自 `BaseEnv` | RL 连续动作/基线规则输出 | RL 版本在 `_allocate_from_candidates` 中根据 action 排序分配；Baselines 通过 `policy` (LJF/SJF/Random)。训练与评估都传入相同参数（batch arrival=true, capacity ramp=false, observation_noise=0.05, capacity_noise=0.05）。 |
| `train_sac_hpg.make_training_env(rank, seed, region_keys, training_df)` | 训练 DAG 参数 | `SubprocVecEnv` 的 env 函数 | 负责注册 synthetic region、创建 `RLEnv`、设置 VecNormalize；rationale：SAC 需要多进程、obs/reward 归一化以稳定训练。 |
| `evaluation.evaluate_single_config(...)` | region 配置、policy、VecNormalize 路径 | episode 数据 + CSV 行 | RL 路径：创建 DummyVecEnv、加载训练期 VecNormalize stats、`vec_env.training=False`。Baseline 路径：调用 `create_baseline_env`。 |

### 2.3 机制参数一览

| 机制 | 配置位置 | 当前数值 | rationale |
|------|----------|---------|-----------|
| **Batch arrival** | `config.py` + `TrueBatchArrival` | `days=[0,20,40,60]`，`ratios=[0.30,0.30,0.25,0.15]`，每批混合 damage | 模拟分批评估，但避免“批次唯一 damage”导致极端拥堵。 |
| **Capacity ramp** | `BaseEnv` | 默认 `use_capacity_ramp=False`（训练/评估均同） | 现实中承包商会快速恢复；关闭 ramp 可提升完成率并更容易对比真实数据。 |
| **候选池** | `config.M_MIN=256`, `M_MAX=512`, `M_RATIO=0.8` | 自适应 M=0.8×queue，clamp 到 [256,512] | 控制观测维度，保持 RL/ Baseline 观测一致；在大队列下仍保留随机性。 |
| **噪声** | `observation_noise=0.05`, `capacity_noise=0.05`； `_apply_work` 中 ±20% 进度噪声 | 与训练/评估保持一致 | 模拟现实误差与施工效率波动，同时考验策略鲁棒性。 |
| **Reward** | `_calculate_reward` | `progress_delta*scale + α·Δcompleted + makespan_ema + fairness_ema + terminal_bonus/-penalty` | 以进度为主信号，组合 per-house completion、makespan/fairness 平滑辅助，终止仅给轻量 bonus/罚值，并通过 `info["reward_breakdown"]` 暴露各组件调参。 |
| **Action Loop** | `BaseEnv.step()` | 单次动作 = 选择 1 个候选任务或 no-op，直到资源耗尽或主动结束 | 每天多步决策：`pending_candidates` 记录当日可调度任务，`action_mask` 屏蔽资源不可达，no-op 或资源耗尽触发 `_advance_day()`，并在 `info["day_advanced"]` / `info["action_mask"]` 中暴露状态。 |

> Reward 参数集中在 `config.REWARD_CONFIG`（`progress_scale`、`completion_alpha/norm`、`success_bonus_range`、`failure_penalty`、EMA 系数、权重等）并提供 `reward_config_repr()` 方便在 HiPerGator 日志中记录实验配置。
| **SAC 超参** | `train_sac_hpg.py` | `timesteps=1e6`, `n_envs=8`, `lr=3e-4`, `batch=512`, `buffer=500k` | 参考 SB3 官方推荐，适配高维连续动作和较长 episode。 |
| **VecNormalize** | 训练：`norm_obs=True`, `norm_reward=True`；评估：加载 stats，`norm_reward=False` | 保持 obs/reward 尺度稳定，评估时返回真实回报。 |

---

## 3. 当前遇到的困难

1. **机制设定的取舍**  
   - Batch arrival 虽保证混合，但仍导致 0/20/40/60 之前无任务、500 天硬截断时 completion 低。需要判断是否还原为 “一次性揭示” 才能对比真实数据，还是保持 stress-test 模式。
   - 噪声/候选池参数目前根据经验设置，尚未系统评估其对完成率/学习稳定性的影响。

2. **调参与模型表现**  
   - SAC 曲线稳定但 `rollout/ep_rew_mean` 未显著上升；availability=1.0 的 completion 仍 ~40%。难以判断是 reward 设计问题、max_steps 太紧、还是 agent 无法探索到有效策略。
   - `MAX_DAYS=500` 是否应该提升到 700+？如果提升将更偏离真实 449 天；如何在“贴近真实”与“让策略学习到东西”之间取舍？

3. **研究框架/实验设计**  
   - 目前只有一个 CSV 输出所有 region × availability，缺少分场景解读（比如 availability=1.0 vs 0.3 的独立分析）。
   - 未定义“真实模式 vs stress-test 模式”的对照流程；若要与真实数据对比，需要先构建无 batch arrival/低噪声版本并评估 baseline。
   - 缺少系统的 smoke check / CI 流程来确保上传后的代码、依赖、训练/评估脚本均一致。

---

## 4. 希望 ChatGPT Deep Research 协助的方向

1. **机制调整建议**  
   - 如何设计 batch arrival / MAX_DAYS / 噪声参数，使得模型既能与真实 80% completion 对齐，又能用于 stress-test？是否需要引入更细致的现实数据（如按区划的揭示顺序）？
2. **算法与 reward 优化**  
   - 当前 reward 是否需要重新加权或增添新项（例如早期完成奖励、capacity 利用率）？SAC 超参是否要扩展（多 seed、更长训练、额外 regularization）？
3. **实验与评价框架**  
   - 如何划分“真实模式 vs stress 模式”并系统比较 RL 与基线，避免单个全局 CSV 难以解读？
   - 是否需要新的指标（如 longtail completion、fairness variance）帮助解释低完成率与策略行为？
4. **工具与流程**  
   - 设计一套 smoke check 脚本/流程（Git hash、import test、unit test、mini 训练+评估）以自动验证上传后的环境是否一致。

---
