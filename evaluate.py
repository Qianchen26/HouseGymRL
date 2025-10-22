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


# =====================================================================
# P0级修复：信息对等的 Baseline 评估
# =====================================================================

def _baseline_allocate_from_candidates(
    candidates: Dict[str, np.ndarray],
    policy: str,
    K_effective: int,
    seed: int
) -> np.ndarray:
    """
    在候选池内应用 baseline 策略（LJF / SJF / RANDOM）

    关键信息：
    ---------
    - 只从候选池（M=96 个）中选择，而非所有 eligible 房屋
    - 确保与 RL 在相同信息集下对比（信息对等）

    算法步骤：
    ---------
    1. 提取候选池信息
    2. 过滤有效候选（mask=True, idx>=0, remain>0）
    3. 按策略排序（LJF / SJF / RANDOM）
    4. 贪心分配直到容量用尽

    排序规则（参考文献）：
    ---------------------
    - LJF（Longest Job First，Graham 1969）：
      按 (damage_level 降序, remaining_work 降序) 排序，
      理由：优先处理重度损坏和大工作量任务。

    - SJF（Shortest Job First，Pinedo 2016）：
      按 (damage_level 升序, remaining_work 升序) 排序，
      理由：优先完成容易任务，提高早期完工率。

    - RANDOM：
      均匀随机排列，用作零假设基线。

    参数：
    -----
    candidates : Dict[str, np.ndarray]
        候选池字典，包含：
        - idx: 房屋全局索引 (M,)
        - remain: 剩余工作量 (M,)
        - dmg: 损坏等级 0/1/2 (M,)
        - cmax: 日容量上限 (M,)
        - mask: 有效性掩码 (M,)
    policy : str
        策略名称：'LJF'、'SJF' 或 'RANDOM'
    K_effective : int
        当天有效工人容量
    seed : int
        随机种子（用于 RANDOM 策略的可重现性）

    返回：
    -----
    allocation : np.ndarray
        整数分配数组，形状 (M,)，每个位置表示分配给该候选的工人数

    学术依据：
    ---------
    - Vinyals et al. (2019, Nature): “对称约束确保公平对比”
    - Bengio et al. (2021, EJOR): “匹配信息条件是对比的前提”
    """
    # 步骤 1：提取候选池信息
    idx = candidates['idx']
    remain = candidates['remain']
    dmg = candidates['dmg']
    cmax = candidates['cmax']
    mask = candidates['mask'] > 0.5

    M = len(idx)
    allocation = np.zeros(M, dtype=np.int32)

    # 步骤 2：过滤有效候选（mask=True AND idx>=0 AND remain>0）
    valid = mask & (idx >= 0) & (remain > 0)
    valid_positions = np.where(valid)[0]

    if len(valid_positions) == 0:
        return allocation  # 无有效候选，返回全零分配

    # 步骤 3：按策略排序
    if policy.upper() == 'LJF':
        # Longest Job First：重度损坏优先，同级内工作量大的优先
        dmg_subset = dmg[valid_positions]
        rem_subset = remain[valid_positions]
        # 字典序排序：主键 = -damage（降序），次键 = -remaining（降序）
        # lexsort 从右到左读取键，因此顺序是 (-rem, -dmg)
        sort_keys = np.lexsort((-rem_subset, -dmg_subset))
        sorted_positions = valid_positions[sort_keys]

    elif policy.upper() == 'SJF':
        # Shortest Job First：轻度损坏优先，同级内工作量小的优先
        dmg_subset = dmg[valid_positions]
        rem_subset = remain[valid_positions]
        # 字典序排序：主键 = damage（升序），次键 = remaining（升序）
        sort_keys = np.lexsort((rem_subset, dmg_subset))
        sorted_positions = valid_positions[sort_keys]

    elif policy.upper() == 'RANDOM':
        # 随机排列：均匀随机打乱所有有效候选
        rng = np.random.default_rng(seed)
        sorted_positions = rng.permutation(valid_positions)

    else:
        raise ValueError(f"Unknown policy: {policy}. Expected 'LJF', 'SJF', or 'RANDOM'.")

    # 步骤 4：贪心分配
    remaining_capacity = K_effective

    for pos in sorted_positions:
        if remaining_capacity <= 0:
            break  # 容量用尽，停止分配

        # 分配量受三类约束：
        # 1. 剩余容量
        # 2. 房屋的日容量上限（cmax）
        # 3. 房屋的剩余工作量（remain）
        give = min(
            remaining_capacity,
            int(cmax[pos]),
            int(remain[pos])
        )

        if give > 0:
            allocation[pos] = give
            remaining_capacity -= give

    return allocation


def evaluate_baseline_with_candidate_pool(
    region_key: str,
    policy_name: str,
    *,
    k_ramp: Optional[Callable[[int], float]],
    seed: int = 42,
    max_steps: int = config.MAX_STEPS,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    使用候选池约束评估 baseline 策略，确保与 RL 信息对等。

    核心改变：
    ---------
    旧方法（evaluate_baseline_simple）：
    - 直接操作数组，能看到所有 eligible 房屋
    - 全局排序后贪心分配
    - 信息优势：可观察 100% 的任务

    新方法（本函数）：
    - 通过 BaselineEnv，仅能看到 M=96 个候选
    - 在候选池内排序与分配
    - 信息受限：只观察到总任务的 0.13% ~ 0.66%（与 RL 相同）

    学术重要性：
    -----------
    违背信息对等 → 无效对比 → 学术不严谨

    参考：
    - Vinyals et al. (2019, Nature): AlphaStar 因信息不对称被批评，重训后强制加入对称约束
    - Bengio et al. (2021, EJOR): “对比要求匹配相关问题分布和信息条件”
    - Agarwal et al. (2021, NeurIPS): “没有对照对比，无法归因性能提升”

    参数：
    -----
    region_key : str
        区域名称，如 'Mataram'、'West Lombok' 等
    policy_name : str
        策略：'LJF'、'SJF' 或 'RANDOM'
    k_ramp : Optional[Callable[[int], float]]
        容量爬坡函数：输入天数返回容量比例 [0, 1]；None 表示满容量
    seed : int
        随机种子
    max_steps : int
        最大模拟天数

    返回：
    -----
    completion_curve : np.ndarray
        每天的完成率（按房屋数计算），形状 (T,)，范围 [0, 1]
    info_history : List[Dict[str, float]]
        每天的详细信息，包含 completion、allocated_workers 等

    示例如下：
    -----
    >>> curve, infos = evaluate_baseline_with_candidate_pool(
    ...     'Mataram', 'LJF', k_ramp=None, seed=42
    ... )
    >>> print(f"Makespan: {len(curve)} days")
    >>> print(f"Final completion: {curve[-1]:.3f}")
    """
    # 步骤 1：创建 BaselineEnv（包含候选池机制）
    # 关键：使用 BaselineEnv 而非直接操作数组
    env = make_region_env(
        region_key,
        BaselineEnv,  # 使用环境类，确保候选池约束
        k_ramp=k_ramp,
        batch_arrival=True,
    )

    # 初始化记录
    completion_curve = []
    info_history = []
    obs = env.reset()

    # 步骤 2：主模拟循环
    for day in range(max_steps):
        # 步骤 3：获取候选池（关键：只有 M=96 个可见）
        candidates = env.last_candidate_view()

        # 步骤 4：在候选池内应用 baseline 策略
        allocation = _baseline_allocate_from_candidates(
            candidates=candidates,
            policy=policy_name,
            K_effective=env.effective_capacity(),
            seed=seed + day  # 每天不同种子，用于 RANDOM 策略
        )

        # 步骤 5：执行动作
        obs, reward, terminated, truncated, info = env.step(allocation)

        # 步骤 6：记录状态
        completion_curve.append(info['completion'])
        info_history.append(info)

        # 步骤 7：检查是否结束
        if terminated or truncated:
            break

    return np.array(completion_curve, dtype=np.float32), info_history




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
    params = config.UNIFIED_RAMP_PARAMS

    def ramp(day: int) -> float:
        if day < params["warmup_days"]:
            return 0.0
        progress = (day - params["warmup_days"]) / params["rise_days"]
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
        M=config.M_CANDIDATES,
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
    """
    通用 rollout 函数，支持 Gymnasium 单环境和 VecEnv 接口
    """
    # 检测是否为 VecEnv
    is_vecenv = hasattr(env, 'num_envs') or hasattr(env, 'venv')
    
    obs = env.reset()
    completions: List[float] = []
    infos: List[Dict[str, float]] = []
    steps = 0

    while True:
        if horizon is not None and steps >= horizon:
            break
        
        action = policy_fn(obs, env)
        
        # 根据环境类型处理 step 返回值
        if is_vecenv:
            # VecEnv: (obs, rewards, dones, infos) - 4 个返回值
            obs, rewards, dones, infos_vec = env.step(action)
            
            # 提取第一个环境的信息（假设只有一个环境）
            info = infos_vec[0] if isinstance(infos_vec, (list, tuple)) and len(infos_vec) > 0 else {}
            done = bool(dones[0]) if isinstance(dones, np.ndarray) else bool(dones)
            
            completions.append(info.get("completion", 0.0))
            infos.append(info)
            
            if done:
                break
        else:
            # Gymnasium: (obs, reward, terminated, truncated, info) - 5 个返回值
            obs, _, terminated, truncated, info = env.step(action)
            completions.append(info["completion"])
            infos.append(info)
            
            if terminated or truncated:
                break
        
        steps += 1
        
        # 如果达到 horizon，填充剩余步数
        if horizon is not None and steps >= horizon:
            break

    # 处理 horizon 填充（如果需要）
    if horizon is not None and steps < horizon:
        last_completion = completions[-1] if completions else 0.0
        last_info = infos[-1] if infos else {
            "allocated_workers": 0,
            "idle_workers": 0,
            "K_effective": 0,
            "completion": last_completion,
            "completion_hh": last_completion,
            "num_candidates": 0,
            "num_eligible": 0,
            "day": steps,
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
    _ = target_completion  # 保持签名兼容；终止依赖 env 的信号
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
    del curve_like  # 仅为保持接口一致
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
    if action_dim != config.M_CANDIDATES:
        raise ValueError(
            f"Loaded model expects action dimension {action_dim}, but config.M_CANDIDATES={config.M_CANDIDATES}. "
            "Please retrain the model or adjust config.M_CANDIDATES accordingly before evaluation."
        )

    obs_dim = int(np.prod(model.observation_space.shape))
    expected = config.OBS_G + config.M_CANDIDATES * config.OBS_F
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
    评估一个地区的所有策略（baseline 与 RL），并汇总指标。

    当前函数同时使用两种不同的评估路径：
    1) baseline 策略：使用“候选池约束”的简化评估
    2) RL 策略：使用环境 step 循环评估
    """
    all_metrics: List[Dict[str, object]] = []
    plot_curves: List[Tuple[str, np.ndarray]] = []

    horizon_obs = len(obs_series)
    obs_curve = obs_series.to_numpy(dtype=np.float32)
    obs_days = int(horizon_obs)
    obs_final = float(obs_curve[-1]) if obs_curve.size > 0 else 0.0

    # ====================================================================
    # 第一部分：评估 baseline 策略（LJF / SJF / RANDOM）
    # ✅ P0 修复：使用候选池约束，确保与 RL 信息对等
    # ====================================================================

    print(f"  [Baseline] 使用候选池约束评估（M={config.M_CANDIDATES}）...")

    for policy_name in ['LJF', 'SJF', 'RANDOM']:
        # 使用新的“候选池约束评估”函数
        curve, infos = evaluate_baseline_with_candidate_pool(
            region,
            policy_name,
            k_ramp=ramp_fn,
            seed=config.EVAL_SEED,
        )

        # 计算指标
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

        # RMSE（如果有观测数据）
        if obs_curve.size > 0:
            # Aligned 模式：满容量运行到观测数据长度
            aligned_curve, _ = evaluate_baseline_with_candidate_pool(
                region,
                policy_name,
                k_ramp=None,  # 满容量
                seed=config.EVAL_SEED,
                max_steps=horizon_obs,
            )
            rmse_aligned = compute_rmse(aligned_curve, obs_curve)

            # Overlap 模式：截断到观测长度
            overlap_curve = align_curve(curve, horizon_obs)
            rmse_overlap = compute_rmse(overlap_curve, obs_curve)
        else:
            rmse_aligned = float('nan')
            rmse_overlap = float('nan')

        # 汇总
        cfg = config.REGION_CONFIG[region]
        total_houses = int(np.sum(cfg.get("damage_dist", [0, 0, 0])))

        result = {
            "region": region,
            "strategy": policy_name,
            "makespan": makespan,
            "utilization": utilization,
            "final_completion": float(curve[-1]) if len(curve) > 0 else 0.0,
            "H_total": total_houses,
            "K": int(cfg["num_contractors"]),
            "rmse_aligned": rmse_aligned,
            "rmse_overlap": rmse_overlap,
            "obs_days": obs_days,
            "obs_final": obs_final,
        }
        result.update(auc_metrics)
        result.update(t_metrics)

        all_metrics.append(result)
        plot_curves.append((policy_name, curve))

        # 保存曲线数据
        curves_dir = config.TAB_DIR / "curves"
        curves_dir.mkdir(parents=True, exist_ok=True)

        curve_df = pd.DataFrame({
            'day': np.arange(len(curve)),
            'completion': curve
        })
        curve_df.to_csv(
            curves_dir / f"{region}_{policy_name}_constrained.csv",
            index=False
        )

        # TensorBoard 记录
        tag_base = f"{region}/{policy_name}"
        writer.add_scalar(f"{tag_base}/makespan", makespan, 0)
        writer.add_scalar(f"{tag_base}/utilization", utilization, 0)
        writer.add_scalar(f"{tag_base}/final_completion", result['final_completion'], 0)
        if np.isfinite(rmse_aligned):
            writer.add_scalar(f"{tag_base}/rmse_aligned", rmse_aligned, 0)
        if np.isfinite(rmse_overlap):
            writer.add_scalar(f"{tag_base}/rmse_overlap", rmse_overlap, 0)

    # ====================================================================
    # 第二部分：评估 RL 策略（若模型可用）
    # 使用 VecNormalize 包装环境，确保观测统计与训练一致
    # ====================================================================

    if model is not None:
        print(f"  [RL] 使用环境 step 评估...")

        # 检查是否提供训练时的 VecNormalize 统计
        if vecnorm_src is None:
            print(f"  [警告] 未提供 vecnorm_src 参数，RL 评估将跳过")
            print(f"  [提示] 调用 evaluate_region 时请传入训练时的 vec_env")
        else:
            # 导入 VecEnv 相关工具
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

            # 为 RL 构建带爬坡的 RLEnv
            single_env_ramp = make_region_env(
                region,
                RLEnv,
                k_ramp=ramp_fn,
                batch_arrival=True,
            )

            # 包装为 VecEnv（必须先包装为 VecEnv 才能用 VecNormalize）
            vec_env_ramp = DummyVecEnv([lambda: single_env_ramp])

            # 用 VecNormalize 包装，设置为评估模式
            vec_env_ramp = VecNormalize(
                vec_env_ramp,
                norm_obs=True,
                norm_reward=False,
                training=False,
                clip_obs=10.0
            )

            # 关键步骤：从训练环境同步观测统计
            vec_env_ramp.obs_rms = vecnorm_src.obs_rms
            vec_env_ramp.ret_rms = vecnorm_src.ret_rms
            vec_env_ramp.clip_obs = vecnorm_src.clip_obs
            vec_env_ramp.training = False
            vec_env_ramp.norm_reward = False

            print(f"    已同步训练时的观测统计")

            # 运行 RL 策略
            action_fn = lambda obs, _env: model.predict(obs, deterministic=True)[0]
            ramp_curve, ramp_infos = rollout_to_completion(vec_env_ramp, action_fn)
            plot_curves.append(("RL", ramp_curve))

            # 取得原始环境的属性（可能需要 unwrap）
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

            # RMSE（如果有观测数据）
            if obs_curve.size > 0:
                # 为 aligned 评估再建一个“满容量”环境
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
                # 同步统计
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

            # 保存 RL 曲线
            curves_dir = config.TAB_DIR / "curves"
            curve_df = pd.DataFrame({
                'day': np.arange(len(ramp_curve)),
                'completion': ramp_curve
            })
            curve_df.to_csv(curves_dir / f"{region}_RL_ramp.csv", index=False)

            # 记录到 TensorBoard
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
