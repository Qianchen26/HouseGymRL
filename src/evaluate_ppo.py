"""
PPO model evaluation script for HouseGym RL.

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
from config import REGION_CONFIG


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
        - episode_length: int (number of steps taken)

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

        # Run episode
        obs = vec_env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            episode_reward += reward[0]
            step_count += 1

        # Extract metrics from final info
        final_info = info[0]
        completion_rate = final_info.get('completion_rate', 0.0)
        avg_queue_time = final_info.get('avg_queue_time', 0.0)

        results.append({
            'seed': seed,
            'completion_rate': completion_rate,
            'avg_queue_time': avg_queue_time,
            'episode_reward': episode_reward,
            'episode_length': step_count,
        })

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
        DataFrame with shape (n_regions × n_crew_levels × n_seeds, 7) and columns:
        - region: string
        - crew_level: float
        - seed: int
        - completion_rate: float
        - avg_queue_time: float
        - episode_reward: float
        - episode_length: int

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

    # Reorder columns
    cols = ['region', 'crew_level', 'seed', 'completion_rate',
            'avg_queue_time', 'episode_reward', 'episode_length']
    combined = combined[cols]

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

        while not done:
            action = env.action_space.sample()  # Baseline doesn't use action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1

        results.append({
            'seed': seed,
            'completion_rate': info.get('completion_rate', 0.0),
            'avg_queue_time': info.get('avg_queue_time', 0.0),
            'episode_reward': episode_reward,
            'episode_length': step_count,
        })

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
        default=["Mataram", "Sumbawa", "Central Lombok"],
        help="Test region names",
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
        "--compare-baselines",
        action="store_true",
        help="Also evaluate baselines (LJF, SJF) for comparison",
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

    # Save PPO results
    ppo_file = output_dir / "evaluation_results.csv"
    ppo_results.to_csv(ppo_file, index=False)
    print(f"\nPPO results saved to: {ppo_file}")

    # Print summary
    summary = ppo_results.groupby(['region', 'crew_level']).agg({
        'completion_rate': ['mean', 'std'],
        'avg_queue_time': ['mean', 'std'],
    }).round(3)

    print("\n" + "="*80)
    print("PPO EVALUATION SUMMARY")
    print("="*80)
    print(summary)
    print("="*80 + "\n")

    # Compare with baselines if requested
    if args.compare_baselines:
        print("\nRunning baseline comparisons...")

        all_baseline_results = []

        for policy in ["LJF", "SJF"]:
            for region in args.test_regions:
                for crew_level in args.crew_levels:
                    print(f"  {policy} @ {region} ({crew_level:.0%})...")

                    baseline_results = run_baseline(region, crew_level, policy, args.n_seeds)
                    baseline_results['region'] = region
                    baseline_results['crew_level'] = crew_level
                    baseline_results['method'] = policy

                    all_baseline_results.append(baseline_results)

        baseline_combined = pd.concat(all_baseline_results, ignore_index=True)

        # Combine PPO and baselines
        all_methods = pd.concat([ppo_results, baseline_combined], ignore_index=True)
        all_methods_file = output_dir / "all_methods_comparison.csv"
        all_methods.to_csv(all_methods_file, index=False)

        print(f"\nAll methods comparison saved to: {all_methods_file}")

        # Print comparison summary
        comparison = all_methods.groupby(['method', 'region', 'crew_level'])['completion_rate'].mean().unstack(level=0)
        print("\n" + "="*80)
        print("COMPLETION RATE COMPARISON (PPO vs Baselines)")
        print("="*80)
        print(comparison.round(3))
        print("="*80 + "\n")

    print("Evaluation completed!")


if __name__ == "__main__":
    main()
