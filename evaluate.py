from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from torch.utils.tensorboard import SummaryWriter

import config
from baseline import make_baseline_allocation
from housegymrl import BaselineEnv, RLEnv, BatchArrivalScheduler


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
    sort_keys = (-rem, -dmg)
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
    sort_keys = (rem, dmg)
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


def evaluate_baseline_simple(
    region_key: str,
    policy_name: str,
    *,
    k_ramp: Optional[Callable[[int], float]],
    seed: int = 42,
    max_steps: int = config.MAX_STEPS,
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


def evaluate_baseline_strategies(
    region_key: str,
    *,
    k_ramp: Optional[Callable[[int], float]],
    seed: int = 42,
) -> List[Dict[str, object]]:
    """
    评估所有baseline策略（LJF、SJF、RANDOM）并返回汇总指标。

    这个函数是evaluate_region函数的简化版本��专门用于baseline评估。
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


# --------------------------------------------------------------------------- #
# Sampling and environment construction
# --------------------------------------------------------------------------- #
def _sample_lognormal(median: float, sigma: float, size: int, rng: np.random.Generator) -> np.ndarray:
    mu = np.log(median)
    samples = rng.lognormal(mu, sigma, size=size)
    return np.maximum(1, np.round(samples)).astype(np.int32)


def create_tasks_from_real_config(region_cfg: Dict, rng: np.random.Generator) -> pd.DataFrame:
    rows: List[Tuple[int, int, int, int, int, int]] = []
    house_id = 0
    for level, count in enumerate(region_cfg["damage_dist"]):
        if count <= 0:
            continue
        params = config.WORK_PARAMS[level]
        durations = _sample_lognormal(params["median"], params["sigma"], count, rng)
        cmax = config.CMAX_BY_LEVEL[level]
        for dur in durations:
            rows.append((level, int(dur), int(dur), 0, cmax, house_id))
            house_id += 1
    return pd.DataFrame(
        rows,
        columns=[
            "damage_level",
            "man_days_total",
            "man_days_remaining",
            "delay_days_remaining",
            "cmax_per_day",
            "house_id",
        ],
    )


def create_unified_ramp() -> Callable[[int], float]:
    """
    创建统一的容量爬坡函数。
    
    Warmup阶段：前warmup_days天完全没有资源（capacity=0）
    Rise阶段：从第(warmup_days+1)天开始线性增长，在rise_days天内达到满容量
    
    修复了off-by-one问题：现在warmup_days=60意味着day 0到59没有资源，
    day 60（第61天）开始有资源。
    """
    params = config.UNIFIED_RAMP_PARAMS

    def ramp(day: int) -> float:
        warmup = params["warmup_days"]
        rise = params["rise_days"]
        
        # Warmup阶段：day < warmup时没有资源
        if day < warmup:
            return 0.0
        
        # Rise阶段：从day=warmup开始线性增长
        # day=warmup时，capacity = 1/rise_days（一个小的正数）
        # day=warmup+rise_days-1时，capacity = 1.0（满容量）
        days_since_warmup_started = day - warmup + 1
        progress = days_since_warmup_started / rise
        
        return float(min(params["capacity_ratio"], max(0.0, progress)))

    return ramp

def make_region_env(
    region_key: str,
    env_cls,
    *,
    k_ramp: Optional[Callable[[int], float]],
    batch_arrival: bool = True,
) -> BaselineEnv:
    cfg = config.REGION_CONFIG[region_key]
    seed = int(cfg.get("seed", config.EVAL_SEED))
    rng = np.random.default_rng(seed)
    tasks_df = create_tasks_from_real_config(cfg, rng)
    resources = {"workers": int(cfg["num_contractors"]), "region_name": region_key}
    return env_cls(
        tasks_df=tasks_df,
        resources=resources,
        max_steps=config.MAX_STEPS,
        seed=seed,
        k_ramp=k_ramp,
        batch_arrival=batch_arrival,
    )


# --------------------------------------------------------------------------- #
# Observed data utilities
# --------------------------------------------------------------------------- #
REGION_ALIASES = {
    "Mataram": "mataram",
    "West Lombok": "lombokbarat",
    "Central Lombok": "lomboktengah",
    "North Lombok": "lombokutara",
    "East Lombok": "lomboktimur",
    "West Sumbawa": "sumbawabarat",
    "Sumbawa": "sumbawa",
}


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" not in df.columns:
            raise ValueError("Observed DataFrame must include a DatetimeIndex or 'date' column.")
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
    else:
        df = df.copy()

    idx = pd.to_datetime(df.index)
    try:
        idx = idx.tz_localize(None)
    except Exception:
        pass
    df.index = idx.normalize()

    df = df.groupby(df.index).max().sort_index()
    return df


def _pick_total_comp_series(df: pd.DataFrame, alias: str) -> Optional[pd.Series]:
    cols_lower = {c.lower(): c for c in df.columns}
    key = f"{alias}_total_comp"
    if key in cols_lower:
        return pd.to_numeric(df[cols_lower[key]], errors="coerce")
    parts = [f"{alias}_comp_rb", f"{alias}_comp_rs", f"{alias}_comp_rr"]
    got = [cols_lower[p] for p in parts if p in cols_lower]
    if got:
        return sum(pd.to_numeric(df[g], errors="coerce") for g in got)
    return None


def _daily_align_monotone_fraction(series: pd.Series, denominator: float) -> pd.Series:
    s = series.copy()
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)
    if pd.isna(s.iloc[0]):
        s.iloc[0] = 0.0
    s = s.ffill().fillna(0.0)
    s = s.cummax()

    denom = max(1.0, float(denominator))
    s = (s.astype(float) / denom).clip(0.0, 1.0)
    s.index = pd.RangeIndex(start=0, stop=len(s), step=1)
    s.name = "obs_completion"
    return s


def _pick_target_total(df: pd.DataFrame, alias: str) -> Optional[float]:
    cols_lower = {c.lower(): c for c in df.columns}
    parts = [f"{alias}_target_rb", f"{alias}_target_rs", f"{alias}_target_rr"]
    got = [cols_lower[p] for p in parts if p in cols_lower]
    if not got:
        return None
    tgt = sum(pd.to_numeric(df[g], errors="coerce") for g in got)
    last = tgt.dropna().iloc[-1] if tgt.dropna().size > 0 else None
    if last is None or not np.isfinite(last):
        return None
    return float(last)


def load_observed(path: Path) -> Dict[str, pd.Series]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Observed data not found at {p}")

    if p.suffix != ".pkl":
        raise ValueError("Observed data must be a pickled pandas DataFrame.")

    df = pd.read_pickle(p)
    if isinstance(df, dict):
        df = next(iter(df.values()))
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected DataFrame in observed pickle, got {type(df)}")
    df = _ensure_datetime_index(df)

    observed: Dict[str, pd.Series] = {}
    for region, cfg in config.REGION_CONFIG.items():
        alias = REGION_ALIASES.get(region)
        if alias is None:
            continue
        series = _pick_total_comp_series(df, alias)
        if series is None:
            continue
        series.index = df.index
        target_total = _pick_target_total(df, alias)
        denom = target_total if target_total and target_total > 0 else series.dropna().iloc[-1]
        observed[region] = _daily_align_monotone_fraction(series, denom)
    return observed


# --------------------------------------------------------------------------- #
# Rollout helpers
# --------------------------------------------------------------------------- #
PolicyFn = Callable[[np.ndarray, BaselineEnv], np.ndarray]


def rollout(
    env: BaselineEnv,
    policy_fn: PolicyFn,
    *,
    horizon: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    obs = env.reset()
    completions: List[float] = []
    infos: List[Dict[str, float]] = []
    steps = 0

    while True:
        if horizon is not None and steps >= horizon:
            break
        action = policy_fn(obs, env)
        obs, _, terminated, truncated, info = env.step(action)
        completions.append(info["completion"])
        infos.append(info)
        steps += 1

        if terminated or truncated:
            if horizon is None:
                break
            if steps >= horizon:
                break
            last_completion = completions[-1]
            last_info = infos[-1]
            while steps < horizon:
                completions.append(last_completion)
                infos.append(last_info)
                steps += 1
            break

    if horizon is not None and steps < horizon:
        last_completion = completions[-1] if completions else 0.0
        last_info = infos[-1] if infos else {
            "allocated_workers": 0,
            "idle_workers": env.K,
            "K_effective": env.K,
            "completion": last_completion,
            "completion_hh": last_completion,
            "num_candidates": 0,
            "num_eligible": 0,
            "day": env.day,
        }
        while steps < horizon:
            completions.append(last_completion)
            infos.append(last_info)
            steps += 1

    return np.asarray(completions, dtype=np.float32), infos


def rollout_to_completion(env: BaselineEnv, policy_fn: PolicyFn) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    return rollout(env, policy_fn, horizon=None)


def rollout_aligned(env: BaselineEnv, policy_fn: PolicyFn, horizon: int) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    return rollout(env, policy_fn, horizon=horizon)


def rollout_to_target(
    env: BaselineEnv,
    policy_fn: PolicyFn,
    *,
    target_completion: float,
    max_days: int,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    _ = target_completion  # maintained for compatibility; termination relies on env signals
    obs = env.reset()
    completions: List[float] = []
    infos: List[Dict[str, float]] = []

    for _ in range(max_days):
        action = policy_fn(obs, env)
        obs, _, terminated, truncated, info = env.step(action)
        completions.append(info["completion"])
        infos.append(info)
        if terminated or truncated:
            break

    return np.asarray(completions, dtype=np.float32), infos


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_makespan_all_done(
    curve_like: Iterable[float],
    info_like: Iterable[Dict[str, float]],
) -> float:
    del curve_like  # curve retained for signature compatibility
    days = [
        float(info.get("day", 0))
        for info in info_like
        if isinstance(info, dict)
        and info.get("done") is True
        and int(info.get("unfinished_houses", -1)) == 0
    ]
    return float(np.min(days)) if days else float("inf")


def compute_auc(curve: np.ndarray, time_point: int) -> float:
    D = min(len(curve), time_point)
    if D <= 1:
        return 0.0
    x = np.arange(D)
    y = curve[:D]
    return float(np.trapz(y, x) / max(1.0, time_point))


def compute_t_percentile(curve: np.ndarray, percentile: float) -> float:
    idx = np.where(curve >= percentile)[0]
    return float(idx[0]) if idx.size > 0 else float("nan")


def compute_utilization(infos: Iterable[Dict[str, float]]) -> float:
    ratios: List[float] = []
    for info in infos:
        K_eff = max(1, int(info["K_effective"]))
        ratios.append(info["allocated_workers"] / K_eff)
    return float(np.mean(ratios)) if ratios else 0.0


def compute_rmse(sim_curve: np.ndarray, obs_curve: np.ndarray) -> float:
    D = min(len(sim_curve), len(obs_curve))
    if D == 0:
        return float("nan")
    return float(np.sqrt(np.mean((sim_curve[:D] - obs_curve[:D]) ** 2)))


def align_curve(curve: np.ndarray, target_len: int) -> np.ndarray:
    if len(curve) >= target_len:
        return curve[:target_len]
    if len(curve) == 0:
        return np.zeros(target_len, dtype=np.float32)
    pad_value = curve[-1]
    pad = np.full(target_len - len(curve), pad_value, dtype=np.float32)
    return np.concatenate([curve, pad])


# --------------------------------------------------------------------------- #
# Model utilities
# --------------------------------------------------------------------------- #
def load_latest_model() -> Optional[SAC]:
    try:
        runs = sorted(Path("runs").glob("*/sac_model.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            return None
        model = SAC.load(str(runs[0]))
        check_model_dimensions(model)
        print(f"[Eval] Loaded model: {runs[0]}")
        return model
    except FileNotFoundError:
        return None
    except Exception as exc:
        print(f"[Eval] Failed to load model ({exc}).")
        raise


def check_model_dimensions(model: SAC) -> None:
    action_dim = int(np.prod(model.action_space.shape))
    if action_dim != config.MAX_HOUSES:
        raise ValueError(
            f"Loaded model expects action dimension {action_dim}, but config.MAX_HOUSES={config.MAX_HOUSES}. "
            "Please retrain the model or adjust config.MAX_HOUSES accordingly before evaluation."
        )

    obs_dim = int(np.prod(model.observation_space.shape))
    expected = config.OBS_G + config.MAX_HOUSES * config.OBS_F
    if obs_dim != expected:
        raise ValueError(
            f"Model observation dimension {obs_dim} does not match expected {expected}. "
            "Ensure training and evaluation configurations are aligned."
        )


# --------------------------------------------------------------------------- #
# Evaluation orchestration
# --------------------------------------------------------------------------- #
def evaluate_region(
    region: str,
    obs_series: pd.Series,
    *,
    ramp_fn: Callable[[int], float],
    model: Optional[SAC],
    writer: SummaryWriter,
    vecnorm_src: Optional = None,
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
        if 'rmse_overlap' in result and np.isfinite(result['rmse_overlap']):
            writer.add_scalar(f"{tag_base}/rmse_overlap", result['rmse_overlap'], 0)

        all_metrics.append(result)



    # ====================================================================
    # 第二部分：评估RL策略（如果模型可用）
    # 使用VecNormalize包装环境，确保观测归一化与训练时一致
    # ====================================================================

    if model is not None:
        print(f"  [RL] 使用环境step评估...")
        
        # 检查是否提供了训练时的VecNormalize统计量
        if vecnorm_src is None:
            print(f"  [警告] 未提供vecnorm_src参数，RL评估将跳过")
            print(f"  [提示] 请在调用evaluate_region时传入训练时的vec_env")
        else:
            # 导入VecEnv相关工具
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            
            # 为RL创建标准的RLEnv环境
            single_env_ramp = make_region_env(
                region,
                RLEnv,
                k_ramp=ramp_fn,
                batch_arrival=True,
            )
            
            # 包装为VecEnv（必须先包装为VecEnv才能使用VecNormalize）
            vec_env_ramp = DummyVecEnv([lambda: single_env_ramp])
            
            # 用VecNormalize包装，设置为评估模式
            vec_env_ramp = VecNormalize(
                vec_env_ramp,
                norm_obs=True,
                norm_reward=False,
                training=False,
                clip_obs=10.0
            )
            
            # 关键步骤：从训练环境同步归一化统计量
            vec_env_ramp.obs_rms = vecnorm_src.obs_rms
            vec_env_ramp.ret_rms = vecnorm_src.ret_rms
            vec_env_ramp.clip_obs = vecnorm_src.clip_obs
            vec_env_ramp.training = False
            vec_env_ramp.norm_reward = False
            
            print(f"    已同步训练时的观测归一化统计")

            # 运行RL策略
            action_fn = lambda obs, _env: model.predict(obs, deterministic=True)[0]
            ramp_curve, ramp_infos = rollout_to_completion(vec_env_ramp, action_fn)
            plot_curves.append(("RL", ramp_curve))

            # 获取原始环境的属性（需要unwrap）
            original_env = vec_env_ramp.venv.envs[0]
            H_total = int(len(original_env.tasks_df)) if hasattr(original_env, 'tasks_df') else 0
            K_total = int(getattr(original_env, 'K', 0))

            # 计算指标
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
                # 为aligned评估创建另一个环境
                single_env_aligned = make_region_env(
                    region, RLEnv,
                    k_ramp=None,
                    batch_arrival=True,
                )
                vec_env_aligned = DummyVecEnv([lambda: single_env_aligned])
                vec_env_aligned = VecNormalize(
                    vec_env_aligned,
                    norm_obs=True,
                    norm_reward=False,
                    training=False,
                    clip_obs=10.0
                )
                # 同样需要同步统计量
                vec_env_aligned.obs_rms = vecnorm_src.obs_rms
                vec_env_aligned.ret_rms = vecnorm_src.ret_rms
                vec_env_aligned.clip_obs = vecnorm_src.clip_obs
                vec_env_aligned.training = False
                vec_env_aligned.norm_reward = False
                
                aligned_curve, _ = rollout_aligned(vec_env_aligned, action_fn, horizon_obs)
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
            if np.isfinite(rmse_aligned):
                writer.add_scalar(f"{tag_base}/rmse_aligned", rmse_aligned, 0)
            if np.isfinite(rmse_overlap):
                writer.add_scalar(f"{tag_base}/rmse_overlap", rmse_overlap, 0)
            for tp, value in auc_metrics.items():
                writer.add_scalar(f"{tag_base}/{tp}", value, 0)
            for key, value in t_metrics.items():
                writer.add_scalar(f"{tag_base}/{key}", value, 0)

    # ====================================================================
    # 第三部分：生成对比图表
    # ====================================================================

    plot_region_curves(region, obs_curve, plot_curves, writer=writer)

    return all_metrics


def plot_region_curves(
    region: str,
    obs_curve: np.ndarray,
    strategy_curves: List[Tuple[str, np.ndarray]],
    *,
    writer: Optional[SummaryWriter] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(obs_curve)), obs_curve, label="Observed", linewidth=2.0, color="black")
    max_len = max(len(curve) for _, curve in strategy_curves) if strategy_curves else len(obs_curve)

    for name, curve in strategy_curves:
        ax.plot(np.arange(len(curve)), curve, label=name, linewidth=1.5)

    ax.set_xlabel("Day")
    ax.set_ylabel("Completion")
    ax.set_title(f"{region} — Observed vs Simulated")
    ax.set_xlim(0, max_len)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = config.FIG_DIR / f"{region}_comparison.png"
    fig.savefig(fig_path)
    if writer is not None:
        writer.add_figure(f"{region}/comparison", fig, global_step=0)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def main() -> None:
    ramp_fn = create_unified_ramp()
    tb_dir = config.OUTPUT_DIR / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    observed = load_observed(config.OBSERVED_DATA_PATH)
    if not observed:
        raise RuntimeError("No observed regions found. Verify observed data columns and config.")

    model = None
    try:
        model = load_latest_model()
    except Exception as exc:
        print(f"[Eval] Model loading skipped ({exc}). Proceeding with baselines only.")

    all_metrics: List[Dict[str, object]] = []

    for region, series in observed.items():
        print(f"[Eval] Region: {region} ({len(series)} observed days)")
        region_metrics = evaluate_region(region, series, ramp_fn=ramp_fn, model=model, writer=writer)
        all_metrics.extend(region_metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.sort_values(["region", "strategy"], inplace=True)
    metrics_path = config.TAB_DIR / "metrics_eval.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[Eval] Metrics written to {metrics_path}")

    summary_lines = [
        "Evaluation Summary",
        "==================",
    ]
    for region in metrics_df["region"].unique():
        summary_lines.append(f"\nRegion: {region}")
        sub = metrics_df[metrics_df["region"] == region]
        for _, row in sub.iterrows():
            summary_lines.append(
                f"  {row['strategy']:>8s} | makespan={row['makespan']:.1f} | "
                f"rmse_aligned={row['rmse_aligned']:.4f} | rmse_overlap={row['rmse_overlap']:.4f} | "
                f"util={row['utilization']:.3f} | final={row['final_completion']:.3f}"
            )
    summary_path = config.TAB_DIR / "metrics_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"[Eval] Summary written to {summary_path}")

    writer.close()


if __name__ == "__main__":
    main()
