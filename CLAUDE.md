# 灾后房屋修复调度系统代码重构任务书（修订版）

## 项目背景与设计哲学的重大转变

欢迎来到这个项目的重构工作。在开始之前，我需要帮助您理解一个重要的认知转变，这个转变将指导我们所有的修改工作。  
这个项目模拟灾后房屋修复的资源调度问题。想象地震后有数千栋受损房屋等待修复，我们有有限的建筑工人，需要决定每天派谁去修哪些房屋。项目的目标是对比两类调度方法的性能：传统的启发式策略（如优先修最严重的、优先修最快能完的）和强化学习算出的策略。  
但是在实现这个对比时，原代码犯了一个根本性的概念错误，它试图让这两类本质不同的策略使用相同的决策框架。让我解释为什么这是错误的，以及我们应该如何修正。

### 强化学习策略的约束本质

强化学习策略本质上是一个神经网络，这个网络的输入层和输出层都必须有固定的维度。这不是设计选择，而是深度学习的技术约束。如果今天网络看到一百个房屋的特征向量，明天看到五千个，网络的权重矩阵根本无法定义。  
因此RL策略必须通过一个固定大小的"观察窗口"来感知世界。在我们的代码中，这个窗口就是候选池，大小M等于九十六。每天RL策略只能看到这九十六个候选房屋的信息，必须学会如何给它们打分排序。候选池的选择策略、候选池的每日刷新、神经网络如何处理固定维度的输入，这些都是RL必须面对的真实约束。  
但关键的洞察是，这个约束只对RL存在。

### 启发式策略的全局视野

LJF策略不是一个需要学习的函数，它是一个明确的规则："看所有等待中的房屋，按损毁等级优先、剩余工作量大的优先排序，然后从队列头开始分配工人直到容量用完"。这个规则的描述中没有任何关于固定窗口、候选池、M个位置的概念。  
LJF是一个算法过程，不是一个参数化的函数。它可以处理十个房屋，也可以处理一万个房屋，执行方式完全相同。它不需要固定输入维度，因为它不需要训练，不需要权重矩阵，不需要反向传播。  
当原代码强行让baseline策略也通过候选池机制运作时，它给这些策略戴上了不必要的镣铐。想象一个有全景视野的调度员被要求蒙上眼睛，每次只能看桌上的九十六份工单，然后我们批评他的决策不够全局优化。这是不公平的，因为限制本身就是人为强加的。

### 新的设计原则

基于这个理解，我们确立了新的设计原则。RL策略和baseline策略应该有各自独立的评估路径，反映它们本质上的不同。  
RL策略继续使用RLEnv环境，通过候选池观察世界，学习在有限视野下的最优决策。候选池对RL不是bug而是feature，是它必须克服的真实约束。  
baseline策略应该有自己简化的评估循环，直接操作环境的完整任务状态，对所有符合条件的房屋排序并贪心分配。没有候选池的中间层，没有M的限制，让策略规则以最自然的方式执行。  
这样的设计让性能对比更有意义。我们不是比较两个都受候选池限制的策略，而是比较受限视野的学习策略和全局视野的手工策略。如果RL能在这种更严格的对比下表现良好，才真正证明了学习的价值。

现在让我们进入具体的修改任务。

---

## 核心修改任务清单

我将修改工作组织成五个主要任务。这些任务有明确的依赖关系，您应该按照我给出的顺序依次完成。每个任务都是独立的代码模块，完成后可以立即测试验证，然后再进入下一个任务。

---

## 任务一：创建baseline策略的独立评估函数

这是整个重构的核心任务。您将创建一套新的函数来评估baseline策略，这套函数完全独立于候选池机制，直接操作环境的任务状态。

### 修改什么

创建新的Python文件或在现有文件中添加新的函数模块，实现baseline策略的简化评估循环。这个循环每天对所有符合条件的房屋排序，然后贪心分配资源，不经过候选池的中间层。

### 修改哪里

我建议在evaluate.py文件中添加这些新函数，因为它们在概念上属于评估逻辑的一部分。但您也可以选择创建一个新文件如baseline_eval.py来保持模块化。

### 怎么改

首先实现排序规则函数。这些函数接收所有符合条件的房屋索引和它们的属性，返回排序后的索引数组。在evaluate.py文件的开头，在现有的导入语句之后，添加这些函数定义：

```python
# ---------------------------------------------------------------------
# Baseline策略的纯粹排序规则实现
# 这些函数直接操作所有eligible房屋，不受候选池限制
# ---------------------------------------------------------------------

def sort_houses_by_ljf(eligible_indices: np.ndarray, 
                       arr_dmg: np.ndarray, 
                       arr_rem: np.ndarray) -> np.ndarray:
    """
    按照LJF规则对房屋排序：重度损毁优先，重度内按剩余工作量降序。
    
    这个函数实现了LJF的纯粹定义，不涉及候选池的概念。
    它可以处理任意数量的房屋，从十个到一万个都是同样的逻辑。
    
    参数:
        eligible_indices: 所有当前符合条件的房屋的索引数组
        arr_dmg: 整个环境的损毁等级数组
        arr_rem: 整个环境的剩余工作量数组
    
    返回:
        排序后的房屋索引数组，队列头是最高优先级
    """
    if len(eligible_indices) == 0:
        return eligible_indices
    
    # 提取这些符合条件房屋的属性
    dmg = arr_dmg[eligible_indices]
    rem = arr_rem[eligible_indices]
    
    # 使用numpy的lexsort进行两级排序
    # lexsort的工作方式是从最后一个键到第一个键依次排序
    # 我们希望首先按damage降序排列（重度=2最优先）
    # 在相同damage内按remaining降序排列（工作量大的优先）
    # 使用负号将降序转换为升序，因为lexsort只支持升序
    sort_keys = (-dmg, -rem)
    sorted_positions = np.lexsort(sort_keys)
    
    return eligible_indices[sorted_positions]


def sort_houses_by_sjf(eligible_indices: np.ndarray,
                       arr_dmg: np.ndarray,
                       arr_rem: np.ndarray) -> np.ndarray:
    """
    按照SJF规则对房屋排序：轻度损毁优先，轻度内按剩余工作量升序。
    
    SJF的哲学是快速完成容易的任务，提高早期的完工率。
    
    参数和返回值与sort_houses_by_ljf相同。
    """
    if len(eligible_indices) == 0:
        return eligible_indices
    
    dmg = arr_dmg[eligible_indices]
    rem = arr_rem[eligible_indices]
    
    # SJF使用升序排列，轻度损毁（0）最优先，剩余工作少的优先
    sort_keys = (dmg, rem)
    sorted_positions = np.lexsort(sort_keys)
    
    return eligible_indices[sorted_positions]


def sort_houses_randomly(eligible_indices: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """
    随机排序房屋，作为性能基准。
    
    参数:
        eligible_indices: 符合条件的房屋索引
        rng: numpy的随机数生成器，确保可重复性
    
    返回:
        随机打乱后的索引数组
    """
    return rng.permutation(eligible_indices)
```

接下来实现核心的评估循环函数。这个函数将替代原来通过环境step方法的评估方式，它直接管理每日的排序和分配过程：

```python
def evaluate_baseline_simple(
    region_key: str,
    policy_name: str,
    *,
    k_ramp: Optional[Callable[[int], float]],
    seed: int = 42,
    max_steps: int = MAX_STEPS,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    使用简化循环评估baseline策略，不通过候选池机制。
    
    这个函数实现了baseline策略应有的评估方式：
    每天看所有符合条件的房屋，按策略规则全局排序，然后贪心分配资源。
    
    这个函数不创建BaselineEnv对象，而是直接管理任务状态的演化。
    它借用了BaseEnv中的一些核心逻辑（如批次到达、容量爬坡），
    但去掉了候选池相关的所有复杂性。
    
    参数:
        region_key: 要评估的地区名称
        policy_name: 策略名称，'LJF'、'SJF'或'RANDOM'
        k_ramp: 容量爬坡函数，如果为None则使用满容量
        seed: 随机种子，确保可重复性
        max_steps: 最大模拟天数
    
    返回:
        (completion_curve, info_history)
        completion_curve: 每天的完工率数组
        info_history: 每天的详细信息字典列表
    """
    # 创建任务数据和资源配置
    cfg = config.REGION_CONFIG[region_key]
    task_seed = int(cfg.get("seed", seed))
    rng = np.random.default_rng(task_seed)
    tasks_df = create_tasks_from_real_config(cfg, rng)
    K_base = int(cfg["num_contractors"])
    
    # 初始化任务状态数组
    # 这些数组记录每个房屋的当前状态
    arr_total = tasks_df["man_days_total"].to_numpy(dtype=np.int32, copy=True)
    arr_rem = tasks_df["man_days_remaining"].to_numpy(dtype=np.int32, copy=True)
    arr_dmg = tasks_df["damage_level"].to_numpy(dtype=np.int32, copy=True)
    arr_cmax = tasks_df["cmax_per_day"].to_numpy(dtype=np.int32, copy=True)
    
    # 批次到达调度器
    # 房屋不是同时到达的，而是分三批在第0、7、14天到达
    scheduler = BatchArrivalScheduler(len(tasks_df), task_seed)
    arrived_mask = np.zeros(len(tasks_df), dtype=bool)
    
    # 用于随机策略的随机数生成器
    random_rng = np.random.default_rng(seed + 1000)
    
    # 记录历史数据
    completion_curve = []
    info_history = []
    
    # 主循环：模拟每一天的调度过程
    for day in range(max_steps):
        # 步骤1：处理今天到达的新批次房屋
        arrivals = scheduler.get_arrivals(day)
        if arrivals.size > 0:
            arrived_mask[arrivals] = True
        
        # 步骤2：确定哪些房屋当前符合分配条件
        # 符合条件意味着：已经到达并且还有剩余工作量
        eligible_mask = arrived_mask & (arr_rem > 0)
        eligible_indices = np.where(eligible_mask)[0]
        
        # 步骤3：计算今天的有效工人容量
        # 考虑容量爬坡的影响
        if k_ramp is not None:
            try:
                ramp_ratio = float(k_ramp(day))
            except Exception:
                ramp_ratio = 1.0
            ramp_ratio = max(0.0, min(1.0, ramp_ratio))
            K_eff = int(round(ramp_ratio * K_base))
        else:
            K_eff = K_base
        K_eff = max(0, K_eff)
        
        # 步骤4：检查是否所有任务都已完成
        if len(eligible_indices) == 0:
            # 所有房屋都完工了，记录完成状态并结束
            completion_curve.append(1.0)
            info_history.append({
                'completion': 1.0,
                'completion_hh': 1.0,
                'allocated_workers': 0,
                'idle_workers': K_eff,
                'K_effective': K_eff,
                'num_eligible': 0,
                'day': day,
                'done': True,
                'unfinished_houses': 0,
            })
            break
        
        # 步骤5：根据策略对符合条件的房屋排序
        # 这是策略的核心差异所在
        if policy_name.upper() == 'LJF':
            queue = sort_houses_by_ljf(eligible_indices, arr_dmg, arr_rem)
        elif policy_name.upper() == 'SJF':
            queue = sort_houses_by_sjf(eligible_indices, arr_dmg, arr_rem)
        elif policy_name.upper() == 'RANDOM':
            queue = sort_houses_randomly(eligible_indices, random_rng)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")
        
        # 步骤6：贪心分配资源
        # 从队列头开始，依次给每个房屋分配工人，直到容量用完
        allocated_workers = 0
        remaining_capacity = K_eff
        
        for house_idx in queue:
            if remaining_capacity <= 0:
                break
            
            # 计算这个房屋今天可以分配多少工人
            # 受三个因素限制：剩余容量、房屋的日容量上限、房屋的剩余工作量
            give = min(
                remaining_capacity,
                int(arr_cmax[house_idx]),
                int(arr_rem[house_idx])
            )
            
            if give > 0:
                # 应用分配：减少房屋的剩余工作量
                arr_rem[house_idx] -= give
                allocated_workers += give
                remaining_capacity -= give
        
        # 步骤7：计算今天结束后的完工度
        total_work = float(arr_total.sum())
        remaining_work = float(arr_rem.sum())
        completion_fraction = 1.0 - (remaining_work / (total_work + 1e-8))
        completion_fraction = float(np.clip(completion_fraction, 0.0, 1.0))
        
        completed_houses = int(np.sum(arr_rem <= 0))
        total_houses = len(arr_rem)
        completion_hh = completed_houses / max(1, total_houses)
        
        idle_workers = K_eff - allocated_workers
        unfinished_houses = int(np.sum(arr_rem > 0))
        
        # 步骤8：记录这一天的状态
        completion_curve.append(completion_hh)
        info_history.append({
            'completion': completion_hh,
            'completion_hh': completion_hh,
            'allocated_workers': allocated_workers,
            'idle_workers': idle_workers,
            'K_effective': K_eff,
            'num_eligible': len(eligible_indices),
            'day': day,
            'done': (unfinished_houses == 0),
            'unfinished_houses': unfinished_houses,
        })
        
        # 如果所有房屋都完工了，提前结束循环
        if unfinished_houses == 0:
            break
    
    return np.array(completion_curve, dtype=np.float32), info_history
```

最后，添加一个包装函数，使新的评估方法能够方便地集成到现有的评估流程中：

```python
def evaluate_baseline_strategies(
    region_key: str,
    *,
    k_ramp: Optional[Callable[[int], float]],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    评估所有baseline策略（LJF、SJF、RANDOM）并返回汇总指标。
    
    这个函数是evaluate_region函数的简化版本，专门用于baseline评估。
    它使用新的简化评估循环，不依赖候选池机制。
    
    返回:
        每个策略的指标字典列表，包含makespan、utilization、AUC等
    """
    results = []
    
    for policy_name in ['LJF', 'SJF', 'RANDOM']:
        # 运行完整评估
        curve, infos = evaluate_baseline_simple(
            region_key,
            policy_name,
            k_ramp=k_ramp,
            seed=seed,
        )
        
        # 计算各项指标
        makespan = compute_makespan_all_done(curve, infos)
        utilization = compute_utilization(infos)
        auc_metrics = {
            f"auc@{tp}": compute_auc(curve, tp) 
            for tp in config.AUC_TIME_POINTS
        }
        t_metrics = {
            "t80": compute_t_percentile(curve, 0.80),
            "t90": compute_t_percentile(curve, 0.90),
            "t95": compute_t_percentile(curve, 0.95),
        }
        
        # 汇总结果
        cfg = config.REGION_CONFIG[region_key]
        total_houses = int(np.sum(cfg.get("damage_dist", [0, 0, 0])))
        
        record = {
            "region": region_key,
            "strategy": policy_name,
            "makespan": makespan,
            "utilization": utilization,
            "final_completion": float(curve[-1]) if len(curve) > 0 else 0.0,
            "H_total": total_houses,
            "K": int(cfg["num_contractors"]),
        }
        record.update(auc_metrics)
        record.update(t_metrics)
        
        results.append(record)
    
    return results
```

### 为什么改

这个新的评估路径实现了我们讨论的设计哲学：让baseline策略以最自然的方式运作，不受候选池的人为约束。  
原来的代码试图让baseline通过BaselineEnv环境来运行，这意味着baseline必须适应环境的候选池机制。但baseline策略的本质是全局排序加贪心分配，它不需要候选池这个中间概念。强行使用候选池不仅增加了复杂性，还人为降低了baseline的性能上界。  
新的评估函数直接实现了策略的数学定义。每天它看到所有等待中的房屋，用策略规则对它们排序，然后从头到尾分配资源。这个过程简单、直接、高效，没有任何不必要的抽象层。  
更重要的是，这个实现让baseline策略成为了真正的性能上界。当我们报告LJF的makespan是三百天时，这个数字代表的是如果完美执行LJF规则理论上能达到的最好性能。它为RL策略设定了一个清晰的目标：如果RL能接近或超越这个数字，说明学习算法确实发现了有价值的模式。

---

## 任务二：修复main.ipynb中的字段名不匹配错误

虽然我们创建了新的baseline评估路径，原代码中仍有一些严重的bug需要修复。这些bug不仅影响旧代码的正确性，理解它们也帮助我们认识为什么需要重构。

### 修改什么

main.ipynb中的_baseline_order函数试图从候选池视图中读取字段，但使用了错误的字段名。这导致函数无法获取房屋的损毁等级和剩余工作量，策略行为完全错误。

### 修改哪里

打开main.ipynb文件，找到定义\_baseline_order和\_baseline_alloc这两个函数的代码单元格。

### 怎么改

找到\_baseline_order函数中访问view字典的代码。将错误的字段名改为正确的字段名：

将这一行：

```python
damage = view.get("damage_level", None)
```

改为：

```python
damage = view.get("dmg", None)
```

将这个尝试多个字段名的段落：

```python
rem = view.get("arr_rem", None)
if rem is None:
    rem = view.get("rem_days", None)
if rem is None:
    rem = view.get("remaining_days", None)
```

简化为：

```python
rem = view.get("remain", None)
```

删除当rem为None时使用cmax作为fallback的逻辑。当数据缺失时，函数应该明确报错而不是静默使用错误的数据继续运行：

删除这段代码：

```python
if rem is None:
    rem = cmax if cmax is not None else np.ones_like(order)
```

添加一个检查，确保关键数据都能正确获取：

```python
if damage is None or rem is None:
    raise ValueError(
        f"无法从候选池视图获取必要字段。"
        f"damage={'已获取' if damage is not None else '缺失'}, "
        f"remain={'已获取' if rem is not None else '缺失'}"
    )
```

### 为什么改

虽然我们推荐使用新的评估路径，但修复这个bug仍然重要，因为它帮助我们理解原代码为什么产生了混乱的结果。  
字段名不匹配是一个静默失败的例子。代码没有崩溃，没有抛出异常，而是静静地使用None值或fallback值继续运行。这导致LJF策略无法识别重度损毁房屋，SJF策略无法按剩余工作量排序，所有策略都退化为某种基于日容量上限的排序。  
这种退化解释了为什么您之前看到的结果中每个策略都偶尔获胜。当所有策略都因为bug而表现得差不多时，谁赢谁输主要取决于随机性。修复这个bug后，即使在旧的候选池框架下，策略之间的差异也会变得更明显。  
更重要的是，这个bug揭示了复杂抽象层的危险。候选池视图是一个中间数据结构，它需要与环境的内部状态、策略函数的期望、不同代码版本的接口保持同步。任何一处不同步都会导致难以调试的问题。新的设计通过减少抽象层次，降低了这类错误发生的可能性。

---

## 任务三：在evaluate.py中集成新的baseline评估路径

现在您已经创建了新的baseline评估函数，需要将它们集成到主评估流程中，替代原来的候选池评估方式。

### 修改什么

修改evaluate.py中的evaluate_region函数和main函数，让它们调用新的evaluate_baseline_strategies函数来评估baseline策略，同时保持RL策略的评估不变。

### 修改哪里

在evaluate.py文件中，找到evaluate_region函数的定义，以及main函数中调用evaluate_region的代码。

### 怎么改

首先修改evaluate_region函数的策略列表构建逻辑。找到当前构建strategies列表的代码：

```python
strategies: List[Tuple[str, Callable[[np.ndarray, BaselineEnv], np.ndarray], type]] = []

strategies.extend(
    [
        (
            "LJF",
            lambda _obs, env: make_baseline_allocation(env, "LJF"),
            BaselineEnv,
        ),
        (
            "SJF",
            lambda _obs, env: make_baseline_allocation(env, "SJF"),
            BaselineEnv,
        ),
        (
            "RANDOM",
            lambda _obs, env: make_baseline_allocation(env, "RANDOM"),
            BaselineEnv,
        ),
    ]
)
```

将整个evaluate_region函数改为分两个部分处理：先评估baseline策略，再评估RL策略。找到函数的主体，替换为这个新的结构：

```python
def evaluate_region(
    region: str,
    obs_series: pd.Series,
    *,
    ramp_fn: Callable[[int], float],
    model: Optional[SAC],
    writer: SummaryWriter,
) -> List[Dict[str, object]]:
    """
    评估一个地区的所有策略（baseline和RL）并返回指标。
    
    这个函数现在使用两种不同的评估路径：
    1. baseline策略使用简化的全局排序评估
    2. RL策略使用标准的环境step循环评估
    """
    all_metrics: List[Dict[str, object]] = []
    plot_curves: List[Tuple[str, np.ndarray]] = []
    
    horizon_obs = len(obs_series)
    obs_curve = obs_series.to_numpy(dtype=np.float32)
    obs_days = int(horizon_obs)
    obs_final = float(obs_curve[-1]) if obs_curve.size > 0 else 0.0
    
    # ====================================================================
    # 第一部分：评估baseline策略（LJF、SJF、RANDOM）
    # 使用新的简化评估路径，不通过候选池
    # ====================================================================
    
    print(f"  [Baseline] 使用简化评估路径...")
    baseline_results = evaluate_baseline_strategies(
        region,
        k_ramp=ramp_fn,
        seed=config.EVAL_SEED,
    )
    
    # 为每个baseline策略计算完整的指标集
    for result in baseline_results:
        strategy_name = result['strategy']
        
        # 重新运行一次以获取完整曲线用于绘图
        # 这次运行的种子必须与上面一致，确保结果可重复
        curve, infos = evaluate_baseline_simple(
            region,
            strategy_name,
            k_ramp=ramp_fn,
            seed=config.EVAL_SEED,
        )
        
        plot_curves.append((strategy_name, curve))
        
        # 如果有观测数据，计算RMSE
        # aligned模式：不使用容量爬坡，运行到观测数据的长度
        if obs_curve.size > 0:
            aligned_curve, _ = evaluate_baseline_simple(
                region,
                strategy_name,
                k_ramp=None,  # 满容量
                seed=config.EVAL_SEED,
                max_steps=horizon_obs,
            )
            rmse_aligned = compute_rmse(aligned_curve, obs_curve)
            
            # overlap模式：使用容量爬坡，但只比较重叠部分
            overlap_curve = align_curve(curve, horizon_obs)
            rmse_overlap = compute_rmse(overlap_curve, obs_curve)
            
            result['rmse_aligned'] = rmse_aligned
            result['rmse_overlap'] = rmse_overlap
        else:
            result['rmse_aligned'] = float('nan')
            result['rmse_overlap'] = float('nan')
        
        result['obs_days'] = obs_days
        result['obs_final'] = obs_final
        
        # 保存曲线到CSV文件
        curves_dir = config.TAB_DIR / "curves"
        curves_dir.mkdir(parents=True, exist_ok=True)
        
        curve_df = pd.DataFrame({
            'day': np.arange(len(curve)),
            'completion': curve
        })
        curve_df.to_csv(
            curves_dir / f"{region}_{strategy_name}_ramp.csv",
            index=False
        )
        
        # 记录到TensorBoard
        tag_base = f"{region}/{strategy_name}"
        writer.add_scalar(f"{tag_base}/makespan", result['makespan'], 0)
        writer.add_scalar(f"{tag_base}/utilization", result['utilization'], 0)
        writer.add_scalar(f"{tag_base}/final_completion", result['final_completion'], 0)
        if 'rmse_aligned' in result and np.isfinite(result['rmse_aligned']):
            writer.add_scalar(f"{tag_base}/rmse_aligned", result['rmse_aligned'], 0)
        
        all_metrics.append(result)
    
    # ====================================================================
    # 第二部分：评估RL策略（如果模型可用）
    # 继续使用环境step评估，因为RL需要候选池机制
    # ====================================================================
    
    if model is not None:
        print(f"  [RL] 使用环境step评估...")
        
        # 为RL创建标准的RLEnv环境
        env_ramp = make_region_env(
            region,
            RLEnv,
            k_ramp=ramp_fn,
            batch_arrival=True,
            fill_non_candidates=False,  # 关闭Stage 2保持一致性
            candidate_policy='random',  # RL使用随机候选池
        )
        
        # 运行RL策略
        action_fn = lambda obs, _env: model.predict(obs, deterministic=True)[0]
        ramp_curve, ramp_infos = rollout_to_completion(env_ramp, action_fn)
        plot_curves.append(("RL", ramp_curve))
        
        # 计算指标
        H_total = int(len(env_ramp.tasks_df)) if hasattr(env_ramp, 'tasks_df') else 0
        K_total = int(getattr(env_ramp, 'K', 0))
        
        makespan = compute_makespan_all_done(ramp_curve, ramp_infos)
        utilization = compute_utilization(ramp_infos)
        auc_metrics = {
            f"auc@{tp}": compute_auc(ramp_curve, tp)
            for tp in config.AUC_TIME_POINTS
        }
        t_metrics = {
            "t80": compute_t_percentile(ramp_curve, 0.80),
            "t90": compute_t_percentile(ramp_curve, 0.90),
            "t95": compute_t_percentile(ramp_curve, 0.95),
        }
        
        # RMSE计算（如果有观测数据）
        if obs_curve.size > 0:
            env_aligned = make_region_env(
                region, RLEnv,
                k_ramp=None,
                batch_arrival=True,
                fill_non_candidates=False,
                candidate_policy='random',
            )
            aligned_curve, _ = rollout_aligned(env_aligned, action_fn, horizon_obs)
            rmse_aligned = compute_rmse(aligned_curve, obs_curve)
            
            overlap_curve = align_curve(ramp_curve, horizon_obs)
            rmse_overlap = compute_rmse(overlap_curve, obs_curve)
        else:
            rmse_aligned = float('nan')
            rmse_overlap = float('nan')
        
        rl_record = {
            "region": region,
            "strategy": "RL",
            "makespan": makespan,
            "utilization": utilization,
            "final_completion": float(ramp_curve[-1]) if len(ramp_curve) else 0.0,
            "H_total": H_total,
            "K": K_total,
            "rmse_aligned": rmse_aligned,
            "rmse_overlap": rmse_overlap,
            "obs_days": obs_days,
            "obs_final": obs_final,
        }
        rl_record.update(auc_metrics)
        rl_record.update(t_metrics)
        
        all_metrics.append(rl_record)
        
        # 保存RL曲线
        curves_dir = config.TAB_DIR / "curves"
        curve_df = pd.DataFrame({
            'day': np.arange(len(ramp_curve)),
            'completion': ramp_curve
        })
        curve_df.to_csv(curves_dir / f"{region}_RL_ramp.csv", index=False)
        
        # 记录到TensorBoard
        tag_base = f"{region}/RL"
        writer.add_scalar(f"{tag_base}/makespan", makespan, 0)
        writer.add_scalar(f"{tag_base}/utilization", utilization, 0)
        writer.add_scalar(f"{tag_base}/final_completion", rl_record['final_completion'], 0)
    
    # ====================================================================
    # 第三部分：生成对比图表
    # ====================================================================
    
    plot_region_curves(region, obs_curve, plot_curves, writer=writer)
    
    return all_metrics
```

### 为什么改

这个修改实现了我们的核心设计原则：让baseline和RL各自使用最适合它们的评估方式。  
baseline策略现在通过简化的循环评估，直接对全局任务排序并分配。这反映了手工启发式策略的本质特征：它们是明确的规则，能看到完整的问题状态，不需要学习过程。  
RL策略继续通过环境的step方法评估，经过候选池观察世界。这反映了学习策略的约束：它必须在有限视野下做决策，这个限制是神经网络架构的必然结果。  
这种分离让性能对比更有意义。我们不是在比较两个都受候选池限制的策略，而是在比较全局最优的启发式和受限学习的策略。如果RL能在这种更严格的对比中表现良好，这才真正证明了学习的价值，证明它能在约束条件下发现人类专家设计的规则难以捕捉的模式。

---

## 任务四：更新main.ipynb中的评估流程

main.ipynb是完整的训练和评估流程的入口。您需要更新它以使用新的baseline评估路径。

### 修改什么

修改main.ipynb中训练后评估部分的代码，让它调用evaluate.py中更新后的函数。同时您可以选择性地删除或注释掉旧的\_baseline_order和\_baseline_alloc函数，因为新的评估路径不再需要它们。

### 修改哪里

找到main.ipynb中训练循环结束后、开始评估的代码段。这通常在文件的后半部分，在model.learn完成之后。

### 怎么改

找到当前调用evaluate_region的循环。原代码可能是这样的：

```python
all_rows = []
for region_key in TRAIN_REGION_KEYS:
    print(f"[Eval] Region = {region_key}")
    rows = evaluate_region(region_key, model=model, vecnorm_src=vec_env, max_days=config.MAX_STEPS, seed=SEED+2024)
    all_rows.extend(rows)
```

这个循环使用的是旧的evaluate_region签名。由于我们已经在evaluate.py中重写了这个函数，main.ipynb中的调用可能需要调整以匹配新的接口。但如果evaluate.py中的evaluate_region保持了向后兼容的接口，这部分可能不需要大的改动。  
更重要的是，您应该删除或注释掉不再需要的辅助函数。找到\_baseline_order、\_baseline_alloc这些函数的定义，在它们前面添加注释说明它们已被废弃：

```python
# =====================================================================
# 以下函数已废弃，不再使用
# 新的baseline评估使用evaluate.py中的evaluate_baseline_simple函数
# 这些旧函数保留仅用于参考或向后兼容性测试
# =====================================================================

# def _baseline_order(view, policy: str, rng: random.Random):
#     ...

# def _baseline_alloc(env, policy: str, rng: random.Random):
#     ...
```

如果您确认这些函数完全不再被使用，可以直接删除它们以保持代码清洁。  
另外，删除或注释掉rollout_region函数中对baseline策略的处理逻辑。这个函数原本试图同时处理RL和baseline，但现在baseline有独立的评估路径，不应该再通过rollout_region来评估。

### 为什么改

清理废弃代码是重构的重要部分。保留不再使用的函数会让代码库变得混乱，未来的维护者可能不确定哪些代码是活跃的，哪些是历史遗留。  
通过明确标记或删除废弃代码，您让代码库传达了一个清晰的信息：这是我们当前使用的正确方式。这不仅帮助他人，也帮助您自己，当六个月后重新查看这个项目时，您不会疑惑为什么有两套评估逻辑。  
同时，保持对旧代码的注释说明也很有价值。它记录了设计演化的历史，帮助理解为什么做出这些改变。如果未来需要回顾旧的实现方式（比如为了向后兼容性测试或理解某个旧实验的结果），注释提供了路标。

---

## 任务五：验证和对比测试

完成代码修改后，您需要系统地验证新实现的正确性。这个任务不是修改代码，而是编写和运行测试来确保一切按预期工作。

### 修改什么

创建一套验证测试，确认新的baseline评估路径产生正确的结果，并且与理论预期一致。

### 修改哪里

您可以创建一个新的Jupyter notebook叫做validation_tests.ipynb，或者在main.ipynb中添加一个专门的测试单元格。

### 怎么改

创建以下几个测试来验证实现的正确性。

#### 测试一：排序规则的正确性

这个测试验证LJF和SJF的排序逻辑是否符合定义：

```python
# 创建一个小的测试场景
test_indices = np.array([0, 1, 2, 3, 4, 5])
test_dmg = np.array([2, 2, 1, 1, 0, 0])  # 两个major，两个moderate，两个minor
test_rem = np.array([50, 30, 40, 20, 35, 15])  # 不同的剩余工作量

# 测试LJF排序
ljf_queue = sort_houses_by_ljf(test_indices, test_dmg, test_rem)

print("LJF队列顺序（房屋索引）:", ljf_queue)
print("对应的损毁等级:", test_dmg[ljf_queue])
print("对应的剩余工作:", test_rem[ljf_queue])

# 验证：前两个应该是major（dmg=2），且50应该排在30前面
assert test_dmg[ljf_queue[0]] == 2, "第一个应该是重度损毁"
assert test_dmg[ljf_queue[1]] == 2, "第二个应该是重度损毁"
assert test_rem[ljf_queue[0]] == 50, "重度损毁中工作量大的应该优先"
assert test_rem[ljf_queue[1]] == 30, "重度损毁中工作量小的应该其次"

print("✓ LJF排序测试通过")

# 测试SJF排序
sjf_queue = sort_houses_by_sjf(test_indices, test_dmg, test_rem)

print("\nSJF队列顺序（房屋索引）:", sjf_queue)
print("对应的损毁等级:", test_dmg[sjf_queue])
print("对应的剩余工作:", test_rem[sjf_queue])

# 验证：前两个应该是minor（dmg=0），且15应该排在35前面
assert test_dmg[sjf_queue[0]] == 0, "第一个应该是轻度损毁"
assert test_dmg[sjf_queue[1]] == 0, "第二个应该是轻度损毁"
assert test_rem[sjf_queue[0]] == 15, "轻度损毁中工作量小的应该优先"
assert test_rem[sjf_queue[1]] == 35, "轻度损毁中工作量大的应该其次"

print("✓ SJF排序测试通过")
```

#### 测试二：完整评估的可重复性

验证多次运行相同种子的评估会产生完全相同的结果：

```python
# 选择一个小的region进行快速测试
test_region = 'Mataram'
test_ramp = create_unified_ramp()
test_seed = 42

# 运行两次评估
curve1, infos1 = evaluate_baseline_simple(
    test_region, 'LJF',
    k_ramp=test_ramp,
    seed=test_seed,
)

curve2, infos2 = evaluate_baseline_simple(
    test_region, 'LJF',
    k_ramp=test_ramp,
    seed=test_seed,
)

# 验证结果完全相同
assert np.allclose(curve1, curve2), "相同种子应该产生相同的完工曲线"
assert len(infos1) == len(infos2), "应该有相同天数的记录"

for i in range(len(infos1)):
    assert infos1[i]['allocated_workers'] == infos2[i]['allocated_workers'], \
        f"第{i}天的分配应该相同"
    assert np.isclose(infos1[i]['completion'], infos2[i]['completion']), \
        f"第{i}天的完工度应该相同"

print("✓ 可重复性测试通过")
print(f"  Makespan: {compute_makespan_all_done(curve1, infos1)}")
print(f"  Final completion: {curve1[-1]:.4f}")
```

#### 测试三：与理论预期的一致性

验证评估结果符合基本的合理性检查：

```python
# 对所有策略运行评估
for policy in ['LJF', 'SJF', 'RANDOM']:
    curve, infos = evaluate_baseline_simple(
        'Mataram', policy,
        k_ramp=test_ramp,
        seed=test_seed,
    )
    
    # 基本合理性检查
    assert len(curve) > 0, f"{policy}: 曲线不应为空"
    assert curve[0] >= 0.0, f"{policy}: 初始完工度应该非负"
    assert curve[-1] <= 1.0, f"{policy}: 最终完工度不应超过1.0"
    assert np.all(np.diff(curve) >= -1e-6), f"{policy}: 完工度应该单调非递减"
    
    # 验证资源分配的一致性
    for info in infos:
        allocated = info['allocated_workers']
        idle = info['idle_workers']
        K_eff = info['K_effective']
        assert allocated + idle == K_eff, \
            f"{policy}: 分配+闲置应该等于有效容量"
        assert allocated >= 0, f"{policy}: 分配不应为负"
        assert idle >= 0, f"{policy}: 闲置不应为负"
    
    makespan = compute_makespan_all_done(curve, infos)
    print(f"✓ {policy}: makespan={makespan}, final={curve[-1]:.4f}")
```

#### 测试四：策略对比的合理性

虽然我们不能预先知道哪个策略一定更好，但我们可以验证策略之间确实有显著差异，而不是都产生几乎相同的结果：

```python
# 收集所有策略的makespan
makespans = {}
for policy in ['LJF', 'SJF', 'RANDOM']:
    curve, infos = evaluate_baseline_simple(
        'Mataram', policy,
        k_ramp=test_ramp,
        seed=test_seed,
    )
    makespans[policy] = compute_makespan_all_done(curve, infos)

print("Makespan对比:")
for policy, ms in makespans.items():
    print(f"  {policy}: {ms}")

# 验证策略之间有差异（不是所有策略都产生完全相同的结果）
unique_makespans = len(set(makespans.values()))
assert unique_makespans > 1, \
    "不同策略应该产生不同的结果，否则可能实现有误"

print(f"✓ 策略差异测试通过（{unique_makespans}个不同的makespan值）")
```

### 为什么改

验证测试是确保代码正确性的最后一道防线。重构后的代码虽然概念更清晰，但实现中仍可能有细微的bug。系统的测试帮助我们尽早发现这些问题。  
排序规则测试确认了策略的核心逻辑正确实现了理论定义。可重复性测试确认了随机种子的使用是正确的，这对科学实验至关重要。合理性测试捕捉那些可能通过编译但逻辑错误的bug，比如完工度变成负数或超过百分之百。策略对比测试确认了不同策略确实产生了有意义的差异，而不是都退化为相同的行为。  
这些测试不仅验证当前的实现，也为未来的修改提供了回归测试基准。当您或其他人在六个月后修改代码时，运行这些测试可以快速确认改动没有破坏现有的正确性。

---

## 实施策略与风险管理

完成这些任务需要细心和耐心。让我给您一些实施建议，帮助您降低风险并确保顺利完成。

### 渐进式实施的重要性

不要试图一次性完成所有修改。每完成一个任务，立即运行相关的测试验证正确性，然后再进入下一个任务。如果在任务三时发现问题，回到任务一检查基础函数的实现。渐进式的工作方式让您能够快速定位问题，而不是在完成所有修改后面对一堆交织的错误不知从何下手。

### 保留旧代码作为对照

在删除旧的实现之前，先确保新的实现完全工作。您可以在一段时间内让两套代码共存，对比它们的结果。如果新旧实现产生了差异，这可能暗示新实现有bug，也可能揭示旧实现的问题。通过对比，您可以更有信心地判断哪个是正确的。

### 版本控制是您的朋友

在开始修改之前，为当前的代码创建一个git分支或至少做一个完整备份。在每个任务完成并通过测试后，提交一次代码。这样如果后续修改出了问题，您可以回退到最近一个工作的版本，而不是从头开始。

### 文档化您的发现

在修改过程中，您可能会发现其他小的问题或改进机会。记录这些发现，但不要立即去修它们。保持专注于当前的任务清单，将额外的发现记录在一个TODO列表中，等主要重构完成后再处理。这避免了范围蔓延，防止重构变成一个永远完不成的项目。

### 性能对比的解释准备

当您完成修改并生成新的评估报告时，结果可能与之前有显著差异。这是预期的，因为我们修复了bug并改变了评估框架。准备好解释这些差异：旧结果中baseline的性能可能被候选池限制人为降低了，新结果展示的是它们的真实潜力。RL与baseline的对比可能变得更具挑战性，但这让RL的优势（如果存在）更有说服力。

### 与研究目标的对齐

始终记住这些修改的最终目的：让实验结果能够支撑清晰的研究结论。每个修改都应该增强结果的可解释性、可重复性和可信度。如果某个修改让代码更复杂而没有带来这些好处，重新考虑它是否必要。

### 寻求反馈的时机

当您完成任务一和任务五（创建新函数并验证其正确性）后，这是一个好的时机向同事或导师展示进度并寻求反馈。在投入大量时间集成到主评估流程之前，确认基础设计得到认可。早期反馈可以避免后期大规模返工。

---

## 预期结果与成功标准

当您完成所有任务后，代码库应该呈现以下特征，这些是验证重构成功的标准。

### 代码清晰性

任何阅读代码的人都应该能够清楚理解baseline策略和RL策略的评估方式为什么不同。注释和函数文档应该解释这种差异的原因：不是因为历史遗留或实现便利，而是因为它们本质上需要不同的框架。

### 结果可解释性

新生成的性能报告应该展现清晰的模式。LJF、SJF和RANDOM之间应该有明显且可解释的性能差异。您应该能够指着某个地区说"LJF在这里表现好是因为这个地区有大量重度损毁房屋且工作量分布呈长尾"，或者"SJF在这里表现好是因为轻度损毁房屋占多数"。

### 性能数值的合理性

所有指标都应该在合理范围内。Makespan应该在几十到几百天之间，不会是负数或上万天。Utilization应该在零到一之间，完工率应该最终达到接近百分之百。如果看到不合理的数值，这是实现有bug的信号。

### RL与baseline的对比更严格但更公平

RL可能不再在所有地区都优于baseline，因为baseline现在展现的是全局最优的性能。但如果RL在某些地区确实表现更好，这个优势是真实且有意义的，它证明了学习算法在有限视野约束下仍能发现有价值的模式。这种结果更有研究价值，因为它经受了更严格的测试。

### 代码可维护性

未来如果需要添加新的baseline策略（比如EDF最早截止期优先），实现过程应该很简单：在evaluate.py中添加一个新的排序函数，然后在策略列表中添加这个新策略的名称。不需要修改环境类，不需要调整候选池机制，不需要担心多处实现不一致。

---

## 总结与展望

这份任务书指导您完成了一次重要的设计重构。重构的核心洞察是认识到RL策略和baseline策略的本质差异，并让代码结构反映这种差异。  
RL策略是需要学习的参数化函数，受固定输入维度的约束，必须通过候选池观察世界。这不是缺陷而是特征，是深度学习架构的必然结果。候选池机制是RL评估的合理组成部分。  
baseline策略是明确的规则，不需要学习，不需要固定维度，应该以最自然的方式运作：全局排序加贪心分配。强加候选池约束是概念错误，会人为降低性能并混淆结果解释。  
通过创建独立的评估路径，我们让每种策略以最适合它的方式运作。这不仅让代码更清晰，更让性能对比更有意义。RL与全局最优baseline的对比，比RL与同样受限baseline的对比，提供了更严格的测试和更深刻的洞察。  
完成这些修改后，您的代码库将成为一个清晰、正确、可维护的实验平台。它产生的结果将具有坚实的技术基础，能够支撑可靠的研究结论。这是高质量科学软件的标志：让正确的事情容易做，让错误的事情难以发生。
