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
from housegymrl import BaselineEnv, RLEnv


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
    obs, _ = env.reset()
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
    obs, _ = env.reset()
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
) -> List[Dict[str, object]]:
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

    if model is not None:
        strategies.insert(
            0,
            (
                "RL",
                lambda obs, _env: model.predict(obs, deterministic=True)[0],
                RLEnv,
            ),
        )

    horizon_obs = len(obs_series)
    obs_curve = obs_series.to_numpy(dtype=np.float32)
    obs_days = int(horizon_obs)
    obs_final = float(obs_curve[-1]) if obs_curve.size > 0 else 0.0

    metrics: List[Dict[str, object]] = []
    plot_curves: List[Tuple[str, np.ndarray]] = []

    for strategy_name, action_fn, env_cls in strategies:
        env_ramp = make_region_env(region, env_cls, k_ramp=ramp_fn, batch_arrival=True)
        ramp_curve, ramp_infos = rollout_to_completion(env_ramp, action_fn)
        plot_curves.append((strategy_name, ramp_curve))

        H_total = int(len(env_ramp.tasks_df)) if getattr(env_ramp, "tasks_df", None) is not None else 0
        K_total = int(getattr(env_ramp, "K", 0))

        env_aligned = make_region_env(region, env_cls, k_ramp=None, batch_arrival=True)
        aligned_curve, _ = rollout_aligned(env_aligned, action_fn, horizon_obs)

        overlap_curve = align_curve(ramp_curve, horizon_obs)
        rmse_aligned = compute_rmse(aligned_curve, obs_curve)
        rmse_overlap = compute_rmse(overlap_curve, obs_curve)

        makespan = compute_makespan_all_done(ramp_curve, ramp_infos)
        auc_metrics = {
            f"auc@{tp}": compute_auc(ramp_curve, tp) for tp in config.AUC_TIME_POINTS
        }
        utilization = compute_utilization(ramp_infos)
        t_metrics = {
            "t80": compute_t_percentile(ramp_curve, 0.80),
            "t90": compute_t_percentile(ramp_curve, 0.90),
            "t95": compute_t_percentile(ramp_curve, 0.95),
        }

        curves_dir = config.TAB_DIR / "curves"
        curves_dir.mkdir(parents=True, exist_ok=True)
        ramp_df = pd.DataFrame({"day": np.arange(len(ramp_curve)), "completion": ramp_curve})
        ramp_df.to_csv(curves_dir / f"{region}_{strategy_name}_ramp.csv", index=False)
        aligned_df = pd.DataFrame({"day": np.arange(len(aligned_curve)), "completion": aligned_curve})
        aligned_df.to_csv(curves_dir / f"{region}_{strategy_name}_aligned.csv", index=False)

        record = {
            "region": region,
            "strategy": strategy_name,
            "rmse_aligned": rmse_aligned,
            "rmse_overlap": rmse_overlap,
            "makespan": makespan,
            "utilization": utilization,
            "final_completion": float(ramp_curve[-1]) if len(ramp_curve) else 0.0,
            "H_total": H_total,
            "K": K_total,
            "obs_days": obs_days,
            "obs_final": obs_final,
        }
        record.update(auc_metrics)
        record.update(t_metrics)
        metrics.append(record)

        tag_base = f"{region}/{strategy_name}"
        writer.add_scalar(f"{tag_base}/rmse_aligned", rmse_aligned, 0)
        writer.add_scalar(f"{tag_base}/rmse_overlap", rmse_overlap, 0)
        writer.add_scalar(f"{tag_base}/makespan", makespan, 0)
        writer.add_scalar(f"{tag_base}/utilization", utilization, 0)
        writer.add_scalar(f"{tag_base}/final_completion", record["final_completion"], 0)
        for tp, value in auc_metrics.items():
            writer.add_scalar(f"{tag_base}/{tp}", value, 0)
        for key, value in t_metrics.items():
            writer.add_scalar(f"{tag_base}/{key}", value, 0)

    plot_region_curves(region, obs_curve, plot_curves, writer=writer)
    return metrics


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
