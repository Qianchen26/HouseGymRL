from __future__ import annotations
from pathlib import Path
import pickle, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List
import warnings

from stable_baselines3 import SAC

# Import new environment classes
from housegymrl import RLEnv, BaselineEnv, OracleEnv, HousegymRLENV

from baseline import create_baseline_env, create_oracle_env
import config
from config import (
    M_CANDIDATES, MAX_STEPS,
    register_synthetic_region, COMBINED_ARRIVAL_CAPACITY_CONFIG,
    DATA_DIR, OBSERVED_DATA_PATH
)



# ------------------- Config -------------------
# Change this to your local path if needed
DATA_PATH = OBSERVED_DATA_PATH  # Use config path

RESULTS_DIR = Path("results")
(RESULTS_DIR / "curves").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "figs").mkdir(parents=True, exist_ok=True)

# Mapping names
REGION_ALIASES = {
    "Mataram":        "mataram",
    "West Lombok":    "lombokbarat",
    "Central Lombok": "lomboktengah",
    "North Lombok":   "lombokutara",
    "East Lombok":    "lomboktimur",
    "West Sumbawa":   "sumbawabarat",
    "Sumbawa":        "sumbawa",
}

def linear_ramp(warmup_days: int = 150, rise_days: int = 180, cap: float = 1.0):
    """Simulate a delay in capacity ramp-up."""
    warmup_days = int(warmup_days)
    rise_days   = max(1, int(rise_days))
    cap = float(max(0.0, min(1.0, cap)))
    def _r(d: int) -> float:
        if d < warmup_days: return 0.0
        return min(cap, (d - warmup_days) / float(rise_days))
    return _r

def infer_region_warmup_days(
    observed_map: dict[str, "pd.Series"],
    *,
    frac_threshold: float = 0.01,   # "non-zero" threshold (e.g., 1% completion)
    sustain_days: int = 7,          # how many days we require it to stay above the threshold
    clip_max: int | None = None     # optionally clip warmup to avoid extreme values
) -> dict[str, int]:

    warmups: dict[str, int] = {}
    for reg, s in observed_map.items():
        y = np.asarray(s.to_numpy(), dtype=float)
        D = len(y)
        w = max(1, min(int(sustain_days), D))

        idx = None
        # sustained threshold crossing
        for i in range(0, D - w + 1):
            if (y[i] >= frac_threshold) and np.all(y[i:i+w] >= frac_threshold):
                idx = i
                break

        # fallback: first strictly >0
        if idx is None:
            nz = np.where(y > 0.0)[0]
            idx = int(nz[0]) if nz.size > 0 else 0

        if clip_max is not None:
            idx = int(min(idx, int(clip_max)))

        warmups[reg] = int(max(0, idx))
    return warmups

def build_region_ramps(
    observed_map: dict[str, "pd.Series"],
    *,
    rise_days: int = 120,   # how fast to climb to full capacity after the warmup
    cap: float = 1.0,       # no reserve requested (keep 1.0)
    frac_threshold: float = 0.01,
    sustain_days: int = 7,
    clip_max: int | None = None
) -> dict[str, callable]:    # based on each region's first non-zero day

    warmup_by_region = infer_region_warmup_days(
        observed_map,
        frac_threshold=frac_threshold,
        sustain_days=sustain_days,
        clip_max=clip_max,
    )
    region_ramp = {
        reg: linear_ramp(warmup_days=wd, rise_days=rise_days, cap=cap)
        for reg, wd in warmup_by_region.items()
    }
    return region_ramp, warmup_by_region


# ------------------- Synthetic generator -------------------
_rng = np.random.default_rng()

def _sample_pert(a, m, b, n):
    lam = 4.0
    alpha = 1 + lam * (m - a) / (b - a + 1e-8)
    beta  = 1 + lam * (b - m) / (b - a + 1e-8)
    x = _rng.beta(alpha, beta, size=n)
    return a + x * (b - a)

def generate_synthetic_scenario(region_name: str, H: int, K: int, seed: int | None):
    """Generate synthetic damage distribution for training."""
    rng = np.random.default_rng(seed)
    props = rng.dirichlet(np.array([1.2,1.0,0.9], dtype=float))
    counts = np.maximum(1, np.round(props * H).astype(int))
    counts[np.argmax(counts)] += (H - counts.sum())

    # Return damage distribution
    return counts.tolist()


def make_synth_env(
    H_min: int = 10_000,
    H_max: int = 100_000,
    worker_ratio: float | tuple[float, float] = (0.1, 0.25),
    seed: int | None = None,
    verbose: bool = False,
    use_batch_arrival: bool = True,  # Use new features for training
    use_capacity_ramp: bool = False,
) -> RLEnv:
    """
    Create a synthetic environment for training.

    Args:
        H_min: Minimum number of houses
        H_max: Maximum number of houses
        worker_ratio: Contractor ratio or range
        seed: Random seed
        verbose: Print statistics
        use_batch_arrival: Use batch arrival system
        use_capacity_ramp: Use capacity ramp system

    Returns:
        RLEnv configured with synthetic data
    """
    rng = np.random.default_rng(seed)
    H = int(rng.integers(int(H_min), int(H_max) + 1))

    if isinstance(worker_ratio, (tuple, list)) and len(worker_ratio) == 2:
        lo, hi = float(worker_ratio[0]), float(worker_ratio[1])
        rho = float(rng.uniform(min(lo, hi), max(lo, hi)))
    else:
        rho = float(worker_ratio)

    K = max(1, int(round(rho * H)))

    # Generate synthetic damage distribution
    damage_dist = generate_synthetic_scenario("SYNTH", H, K, seed)

    # Register synthetic region dynamically
    region_key = register_synthetic_region(H, K, damage_dist, seed)

    if verbose:
        minor, moderate, major = damage_dist
        total = sum(damage_dist)
        print(f"Synthetic scenario: {region_key}")
        print(f"  Total: {total}, Minor: {minor}, Moderate: {moderate}, Major: {major}")
        print(f"  Contractors: {K} (ratio: {K/H:.2%})")

    # Create RLEnv with synthetic region
    return RLEnv(
        region_key=region_key,
        num_contractors=K,
        use_batch_arrival=use_batch_arrival,
        use_capacity_ramp=use_capacity_ramp,
        seed=seed,
        max_steps=MAX_STEPS,
    )


def make_region_env(
    region_key: str,
    use_batch_arrival: bool = True,
    use_capacity_ramp: bool = False,
    seed: Optional[int] = None
) -> RLEnv:
    """
    Create environment for a real region (for testing).

    Args:
        region_key: Region name from REGION_CONFIG
        use_batch_arrival: Use batch arrival system
        use_capacity_ramp: Use capacity ramp system
        seed: Random seed

    Returns:
        RLEnv configured for the region
    """
    if region_key not in config.REGION_CONFIG:
        raise ValueError(f"Unknown region: {region_key}")

    return RLEnv(
        region_key=region_key,
        use_batch_arrival=use_batch_arrival,
        use_capacity_ramp=use_capacity_ramp,
        seed=seed,
        max_steps=MAX_STEPS,
    )


# ------------------- Compatibility Functions -------------------

def make_region_env_with_M(region_key: str, M_override: int, k_ramp=None) -> HousegymRLENV:
    """Legacy function for backward compatibility."""
    warnings.warn(
        "make_region_env_with_M is deprecated. Use make_region_env instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Create new environment (M is now fixed at 512)
    env = make_region_env(region_key)
    return env


def infer_M_from_model(model_obj) -> int:
    """Infer M from model observation space."""
    from config import OBS_G, OBS_F
    obs_dim = int(model_obj.observation_space.shape[0])
    M = (obs_dim - OBS_G) // OBS_F
    assert OBS_G + M * OBS_F == obs_dim, \
        f"obs_dim={obs_dim} not compatible with OBS_G={OBS_G}, OBS_F={OBS_F}"
    return int(M)


# ------------------- Observed data loader -------------------
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has proper datetime index."""
    # 1) DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date")
        else:
            raise ValueError("Observed DataFrame must have a DatetimeIndex or a 'date' column.")
    else:
        df = df.copy()

    idx = pd.to_datetime(df.index)
    try:
        idx = idx.tz_localize(None)
    except Exception:
        pass
    df.index = idx.normalize()

    # 2) duplicates
    df = df.groupby(df.index).max()
    df = df.sort_index()
    return df


def _pick_total_comp_series(df: pd.DataFrame, alias: str) -> pd.Series | None:
    cols_lower = {c.lower(): c for c in df.columns}
    key = f"{alias}_total_comp"
    if key in cols_lower:
        return pd.to_numeric(df[cols_lower[key]], errors="coerce")
    # Fallback: sum of comp_rb/rs/rr
    parts = [f"{alias}_comp_rb", f"{alias}_comp_rs", f"{alias}_comp_rr"]
    got = [cols_lower[p] for p in parts if p in cols_lower]
    if got:
        return sum(pd.to_numeric(df[g], errors="coerce") for g in got)
    return None


def _daily_align_monotone_fraction(s: pd.Series, denominator: float) -> pd.Series:
    s = s.copy()
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


def load_observed(DATA_PATH: str | Path) -> dict[str, pd.Series]:
    p = Path(DATA_PATH)
    if not p.exists():
        print(f"[WARN] Observed data file not found: {p}")
        return {}

    if p.suffix == ".pkl":
        obj = pd.read_pickle(p)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, pd.DataFrame) and any(c.lower().endswith("_total_comp") for c in v.columns):
                    obj = v
                    break
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"Expected a pickled DataFrame; got {type(obj)}")
        df = _ensure_datetime_index(obj)
    else:
        raise ValueError("DATA_PATH must be a .pkl of a DataFrame with DatetimeIndex.")

    out = {}
    for region_name in config.REGION_CONFIG.keys():
        if region_name.startswith("SYNTH"):  # Skip synthetic regions
            continue
        alias = REGION_ALIASES.get(region_name)
        if not alias:
            continue
        s = _pick_total_comp_series(df, alias)
        if s is None:
            continue
        s.index = df.index

        target_total = _pick_target_total(df, alias)
        observed_final = float(s.dropna().iloc[-1]) if s.dropna().size > 0 else 0.0
        denominator = target_total if (target_total is not None and target_total > 0) else observed_final

        out[region_name] = _daily_align_monotone_fraction(s, denominator)
    return out


def _pick_target_total(df: pd.DataFrame, alias: str) -> float | None:
    cols_lower = {c.lower(): c for c in df.columns}
    parts = [f"{alias}_target_rb", f"{alias}_target_rs", f"{alias}_target_rr"]
    got = [cols_lower[p] for p in parts if p in cols_lower]
    if not got:
        return None
    total = sum(pd.to_numeric(df[g], errors="coerce").dropna().iloc[0] if not df[g].dropna().empty else 0
                for g in got)
    return float(total) if total > 0 else None


# ------------------- Metrics -------------------
def rmse(sim: np.ndarray, obs: np.ndarray) -> float:
    min_len = min(len(sim), len(obs))
    return float(np.sqrt(np.mean((sim[:min_len] - obs[:min_len])**2)))


def auc_at_T(curve: np.ndarray, T: int) -> float:
    if len(curve) == 0: return 0.0
    y = curve[:min(T, len(curve))]
    auc = np.trapz(y) / max(1, len(y))
    return float(auc)


def t_reach(curve: np.ndarray, thr: float) -> float | float("nan"):
    idx = np.where(curve >= thr)[0]
    return float(idx[0]) if idx.size > 0 else float("nan")


# ------------------- Rollout function -------------------
def rollout(
    env,
    model: Optional[SAC] = None,
    max_days: int = 1000
) -> np.ndarray:
    """
    Run a complete rollout of the environment. The caller is responsible for
    providing an environment configured with the desired policy.

    Args:
        env: Environment instance (RLEnv, BaselineEnv, OracleEnv, ...)
        model: SAC model used when rolling out an `RLEnv`
        max_days: Maximum simulation days

    Returns:
        Array of completion values per day
    """
    obs, info = env.reset()
    traj = []
    simulated_days = 0

    while simulated_days < max_days:
        if isinstance(env, RLEnv) and model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = None

        obs, r, done, trunc, info = env.step(action)

        if info.get("day_advanced", False):
            traj.append(info.get("completion", 0.0))
            simulated_days += 1

        if done or trunc:
            break

    return np.array(traj, dtype=float)


# ------------------- Main evaluation -------------------
def main():
    """Main evaluation function."""
    print("=" * 60)
    print("EVALUATION WITH NEW ARCHITECTURE")
    print("=" * 60)

    # 1) Load observed data
    observed = load_observed(DATA_PATH)
    if not observed:
        print("[WARN] No observed data found. Only synthetic testing available.")
    else:
        print(f"Loaded observed data for {len(observed)} regions")
        for reg, s in observed.items():
            print(f"  {reg}: {len(s)} days, final={s.iloc[-1]:.3f}")

    # 2) Try load latest SAC model
    model = None
    try:
        runs = sorted(Path("runs").glob("*/sac_model.zip"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            model = SAC.load(str(runs[0]))
            print(f"\nLoaded SAC model: {runs[0]}")
            M_inferred = infer_M_from_model(model)
            print(f"  Inferred M={M_inferred} (config M={M_CANDIDATES})")
    except Exception as e:
        print(f"\n[WARN] No SAC model loaded: {e}")
        print("  Running baselines only")

    # 3) Test on synthetic environment first
    print("\n" + "=" * 60)
    print("TESTING ON SYNTHETIC ENVIRONMENT")
    print("=" * 60)

    test_env = make_synth_env(
        H_min=5000, H_max=10000,
        worker_ratio=(0.15, 0.20),
        seed=999,
        verbose=True
    )

    policies = ["LJF", "SJF", "Random"]
    if model is not None:
        policies.insert(0, "SAC")

    for policy in policies:
        print(f"\nTesting {policy}...")
        if policy == "SAC":
            traj = rollout(test_env, model=model, max_days=300)
        else:
            # Create baseline environment
            baseline_env = create_baseline_env(
                region_key=test_env.region_key,
                policy=policy,
                num_contractors=test_env.num_contractors,
                seed=999
            )
            traj = rollout(baseline_env, max_days=300)

        if len(traj) > 0:
            final = traj[-1]
            t80 = t_reach(traj, 0.80)
            t90 = t_reach(traj, 0.90)
            print(f"  Final: {final:.3f}, T80: {t80:.0f}, T90: {t90:.0f}")
        else:
            print(f"  No trajectory generated")

    # 4) Evaluate on real regions if data available
    if observed:
        print("\n" + "=" * 60)
        print("EVALUATION ON REAL REGIONS")
        print("=" * 60)

        rows = []
        for reg, obs_series in observed.items():
            print(f"\nRegion: {reg}")
            obs_curve = obs_series.to_numpy()
            D_obs = len(obs_curve)

            # Create figure
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(D_obs), obs_curve,
                    label="Observed", linewidth=2, color='black', linestyle='--')

            for policy in policies:
                print(f"  Running {policy}...")

                if policy == "SAC" and model is not None:
                    env = make_region_env(reg, seed=42)
                    sim = rollout(env, model=model, max_days=D_obs)
                else:
                    env = create_baseline_env(
                        region_key=reg,
                        policy=policy,
                        seed=42
                    )
                    sim = rollout(env, max_days=D_obs)

                # Save curve
                pd.DataFrame({
                    "day": np.arange(len(sim)),
                    "completion": sim
                }).to_csv(RESULTS_DIR / "curves" / f"{reg}_{policy}.csv", index=False)

                # Calculate metrics
                row = {
                    "region": reg,
                    "strategy": policy,
                    "days_obs": D_obs,
                    "rmse": rmse(sim, obs_curve),
                    "auc@200": auc_at_T(sim, 200),
                    "auc@300": auc_at_T(sim, 300),
                    "t80": t_reach(sim, 0.80),
                    "t90": t_reach(sim, 0.90),
                    "t95": t_reach(sim, 0.95),
                    "final_completion": float(sim[-1]) if len(sim) > 0 else 0.0,
                    "makespan": t_reach(sim, 0.99),
                }
                rows.append(row)

                # Plot
                plt.plot(sim, label=policy, alpha=0.8)

                print(f"    RMSE: {row['rmse']:.4f}, Final: {row['final_completion']:.3f}")

            plt.axvline(x=D_obs, linestyle="--", linewidth=1, color='gray', alpha=0.5)
            plt.title(f"{reg} - Observed vs Simulated")
            plt.xlabel("Day")
            plt.ylabel("Cumulative Completion (0..1)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / "figs" / f"{reg}_compare.png", dpi=150)
            plt.close()

        # Save metrics
        if rows:
            metrics_df = pd.DataFrame(rows)
            metrics_path = RESULTS_DIR / "metrics_eval.csv"
            metrics_df.to_csv(metrics_path, index=False)
            print(f"\nMetrics saved to: {metrics_path}")

            # Print summary
            print("\n" + "=" * 60)
            print("SUMMARY BY STRATEGY")
            print("=" * 60)
            summary = metrics_df.groupby("strategy")[["rmse", "final_completion"]].mean()
            print(summary)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
