"""
PPO Model Evaluation Script

Evaluates trained PPO models across multiple test regions, crew levels, and seeds.
Compares PPO performance against baselines (LJF, SJF, Random).

Usage:
    python evaluate_ppo.py --checkpoint-dir runs/my_experiment --output-dir results/
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent))

from housegymrl import RLEnv, BaselineEnv
from config import REGION_CONFIG, MAX_STEPS


def load_model_and_vecnorm(checkpoint_dir: str) -> Tuple[PPO, VecNormalize]:
    """
    Load trained PPO model and VecNormalize statistics.

    Args:
        checkpoint_dir: Directory containing model.zip and vecnormalize.pkl (string).
            Expected structure:
            {checkpoint_dir}/
            ├── model.zip
            └── vecnormalize.pkl

    Returns:
        Tuple of (PPO model, VecNormalize instance).

    Raises:
        FileNotFoundError: If model or vecnormalize files are missing.

    Example:
        >>> model, vec_norm = load_model_and_vecnorm("runs/experiment1")
    """
    checkpoint_path = Path(checkpoint_dir)

    model_file = checkpoint_path / "model.zip"
    vecnorm_file = checkpoint_path / "vecnormalize.pkl"

    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")
    if not vecnorm_file.exists():
        raise FileNotFoundError(f"VecNormalize not found: {vecnorm_file}")

    print(f"Loading model from: {model_file}")
    print(f"Loading VecNormalize from: {vecnorm_file}")

    # Load model (without environment first)
    model = PPO.load(str(model_file))

    # Load VecNormalize stats (will be applied to eval env later)
    vec_norm_dummy = VecNormalize.load(
        str(vecnorm_file),
        DummyVecEnv([lambda: RLEnv("Mataram", M_min=1024, M_max=1024)])
    )

    return model, vec_norm_dummy


def evaluate_on_scenario(
    model: PPO,
    vec_norm: VecNormalize,
    region: str,
    crew_level: float,
    n_seeds: int = 5
) -> pd.DataFrame:
    """
    Evaluate model on a single scenario across multiple seeds.

    Args:
        model: Trained PPO model instance.
        vec_norm: VecNormalize instance with training statistics.
        region: Region name (string, e.g., "Mataram").
        crew_level: Crew availability fraction (float in [0.1, 1.0]).
            Example: 0.5 means 50% of region's base contractors.
        n_seeds: Number of evaluation seeds (integer).

    Returns:
        DataFrame with n_seeds rows and columns:
        - seed: int
        - completion_rate: float (fraction of houses completed)
        - avg_queue_time: float (average waiting days)
        - episode_reward: float (total episode reward)
        - total_completion_days: int (days to complete all houses)
        - num_contractors: int (actual contractor count used)
        - day1, day2, ...: float (completion rate at each day)

    Example:
        >>> results = evaluate_on_scenario(model, vec_norm, "Mataram", 0.5, n_seeds=5)
        >>> print(f"Mean completion: {results['completion_rate'].mean():.1%}")
    """
    base_contractors = REGION_CONFIG[region]["num_contractors"]
    num_contractors = int(base_contractors * crew_level)

    results = []

    for seed in range(n_seeds):
        # Create eval environment factory (fix lambda closure bug)
        def make_env(seed_val=seed, num_contr=num_contractors, reg=region):
            return RLEnv(
                region_key=reg,
                num_contractors=num_contr,
                M_min=1024,
                M_max=1024,
                use_batch_arrival=True,
                stochastic_duration=True,
                observation_noise=0.15,
                capacity_noise=0.10,
                seed=1000 + seed_val,
            )

        # Wrap with VecNormalize (use training stats, disable training mode)
        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(vec_env, training=False, norm_obs=True, norm_reward=False)
        vec_env.obs_rms = vec_norm.obs_rms  # Use training normalization stats

        # Run episode and track daily completion
        obs = vec_env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        completion_history = []  # Track completion rate at each day

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            episode_reward += reward[0]
            step_count += 1
            # Record completion rate for this day
            daily_completion = info[0].get('completion_rate', 0.0)
            completion_history.append(daily_completion)

        # Extract metrics from final info
        final_info = info[0]
        completion_rate = final_info.get('completion_rate', 0.0)
        avg_queue_time = final_info.get('avg_queue_time', 0.0)

        # Pad completion_history to MAX_STEPS with final completion rate
        final_rate = completion_history[-1] if completion_history else 0.0
        while len(completion_history) < MAX_STEPS:
            completion_history.append(final_rate)

        # Build result row with daily completion columns
        row = {
            'seed': seed,
            'completion_rate': completion_rate,
            'avg_queue_time': avg_queue_time,
            'episode_reward': episode_reward,
            'total_completion_days': step_count,
            'num_contractors': num_contractors,
        }
        # Add daily completion columns (day1, day2, ..., day500)
        for day_idx, comp in enumerate(completion_history, start=1):
            row[f'day{day_idx}'] = comp

        results.append(row)

        vec_env.close()

    return pd.DataFrame(results)


def evaluate_cross_scenarios(
    model: PPO,
    vec_norm: VecNormalize,
    test_regions: List[str],
    crew_levels: List[float],
    n_seeds: int = 5
) -> pd.DataFrame:
    """
    Evaluate model across multiple regions and crew levels.

    Args:
        model: Trained PPO model.
        vec_norm: VecNormalize with training stats.
        test_regions: List of region names (list of strings).
        crew_levels: List of crew availability fractions (list of floats).
        n_seeds: Number of seeds per scenario (integer).

    Returns:
        DataFrame with columns:
        - region: string
        - crew_level: float
        - num_contractors: int (actual contractor count)
        - seed: int
        - completion_rate: float
        - avg_queue_time: float
        - episode_reward: float
        - total_completion_days: int
        - day1, day2, ...: float (completion rate at each day)

    Example:
        >>> results = evaluate_cross_scenarios(
        ...     model, vec_norm,
        ...     ["Mataram", "Sumbawa"],
        ...     [0.5, 1.0],
        ...     n_seeds=3
        ... )
        >>> # Shape: (2 regions × 2 levels × 3 seeds, 7) = (12, 7)
    """
    all_results = []

    total_scenarios = len(test_regions) * len(crew_levels)
    current = 0

    for region in test_regions:
        for crew_level in crew_levels:
            current += 1
            print(f"\n[{current}/{total_scenarios}] Evaluating {region} @ {crew_level:.0%} crew...")
            sys.stdout.flush()

            scenario_results = evaluate_on_scenario(
                model, vec_norm, region, crew_level, n_seeds
            )

            scenario_results['region'] = region
            scenario_results['crew_level'] = crew_level

            all_results.append(scenario_results)

    combined = pd.concat(all_results, ignore_index=True)

    # Reorder columns: fixed columns first, then day columns in order
    fixed_cols = ['region', 'crew_level', 'num_contractors', 'seed', 'completion_rate',
                  'avg_queue_time', 'episode_reward', 'total_completion_days']
    day_cols = sorted([c for c in combined.columns if c.startswith('day')],
                      key=lambda x: int(x[3:]))  # Sort by day number
    combined = combined[fixed_cols + day_cols]

    return combined


def run_baseline(
    region: str,
    crew_level: float,
    policy: str,
    n_seeds: int = 5
) -> pd.DataFrame:
    """
    Run baseline policy for comparison.

    Args:
        region: Region name (string).
        crew_level: Crew availability fraction (float).
        policy: Baseline policy name (string: "LJF", "SJF", or "Random").
        n_seeds: Number of seeds (integer).

    Returns:
        DataFrame with same structure as evaluate_on_scenario output.
    """
    base_contractors = REGION_CONFIG[region]["num_contractors"]
    num_contractors = int(base_contractors * crew_level)

    results = []

    for seed in range(n_seeds):
        env = BaselineEnv(
            region_key=region,
            policy=policy,
            num_contractors=num_contractors,
            M_min=1024,
            M_max=1024,
            seed=1000 + seed,
        )

        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        completion_history = []  # Track completion rate at each day

        while not done:
            action = env.action_space.sample()  # Baseline doesn't use action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            # Record completion rate for this day
            daily_completion = info.get('completion_rate', 0.0)
            completion_history.append(daily_completion)

        # Pad completion_history to MAX_STEPS with final completion rate
        final_rate = completion_history[-1] if completion_history else 0.0
        while len(completion_history) < MAX_STEPS:
            completion_history.append(final_rate)

        # Build result row with daily completion columns
        row = {
            'seed': seed,
            'completion_rate': info.get('completion_rate', 0.0),
            'avg_queue_time': info.get('avg_queue_time', 0.0),
            'episode_reward': episode_reward,
            'total_completion_days': step_count,
            'num_contractors': num_contractors,
        }
        # Add daily completion columns (day1, day2, ..., day500)
        for day_idx, comp in enumerate(completion_history, start=1):
            row[f'day{day_idx}'] = comp

        results.append(row)

    return pd.DataFrame(results)


def main():
    """Command-line interface for PPO evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO model on test scenarios",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing model.zip and vecnormalize.pkl",
    )

    parser.add_argument(
        "--test-regions",
        type=str,
        nargs="+",
        default=[
            "Mataram", "West Lombok", "North Lombok", "Central Lombok",
            "East Lombok", "West Sumbawa", "Sumbawa"
        ],
        help="Test region names (all 7 Lombok regions by default)",
    )

    parser.add_argument(
        "--crew-levels",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7, 1.0],
        help="Crew availability levels (fractions)",
    )

    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of evaluation seeds per scenario",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )

    parser.add_argument(
        "--ppo-only",
        action="store_true",
        help="Only evaluate PPO (skip baselines LJF, SJF)",
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("PPO Model Evaluation")
    print(f"{'='*80}\n")

    # Load model
    try:
        model, vec_norm = load_model_and_vecnorm(args.checkpoint_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Evaluate PPO
    print(f"\nEvaluating PPO across {len(args.test_regions)} regions × {len(args.crew_levels)} crew levels × {args.n_seeds} seeds...")

    ppo_results = evaluate_cross_scenarios(
        model, vec_norm,
        args.test_regions,
        args.crew_levels,
        args.n_seeds,
    )

    ppo_results['method'] = 'PPO'

    # Collect all results (PPO + baselines)
    all_results = [ppo_results]

    # Run baselines unless --ppo-only is specified
    if not args.ppo_only:
        print("\nRunning baseline comparisons (LJF, SJF)...")

        for policy in ["LJF", "SJF"]:
            for region in args.test_regions:
                for crew_level in args.crew_levels:
                    print(f"  {policy} @ {region} ({crew_level:.0%})...")

                    baseline_results = run_baseline(region, crew_level, policy, args.n_seeds)
                    baseline_results['region'] = region
                    baseline_results['crew_level'] = crew_level
                    baseline_results['method'] = policy

                    all_results.append(baseline_results)

    # Combine all methods into single DataFrame
    all_methods = pd.concat(all_results, ignore_index=True)

    # Reorder columns: fixed columns first, then day columns in order
    fixed_cols = ['region', 'crew_level', 'num_contractors', 'seed', 'method',
                  'completion_rate', 'avg_queue_time', 'episode_reward', 'total_completion_days']
    day_cols = sorted([c for c in all_methods.columns if c.startswith('day')],
                      key=lambda x: int(x[3:]))
    all_methods = all_methods[fixed_cols + day_cols]

    # Save to single comprehensive CSV
    output_file = output_dir / "all_methods_comparison.csv"
    all_methods.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Print summary by method
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    summary = all_methods.groupby(['method', 'region', 'crew_level', 'num_contractors']).agg({
        'completion_rate': ['mean', 'std'],
        'total_completion_days': ['mean', 'std'],
        'avg_queue_time': ['mean', 'std'],
    }).round(3)
    print(summary)

    # Print completion days comparison if baselines were run
    if not args.ppo_only:
        comparison = all_methods.groupby(['method', 'region', 'crew_level', 'num_contractors'])['total_completion_days'].mean().unstack(level=0)
        print("\n" + "="*80)
        print("COMPLETION DAYS COMPARISON (PPO vs Baselines)")
        print("="*80)
        print(comparison.round(1))

    print("="*80 + "\n")
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
