"""
Oracle Baseline Experiment

Runs Oracle-LJF and Oracle-SJF (no M limit, full queue visibility) to establish
performance ceiling and quantify information value.

Usage:
    python src/run_oracle_experiment.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))

from housegymrl import OracleEnv
from config import REGION_CONFIG

def run_oracle_evaluation(
    regions: List[str],
    policies: List[str] = ["LJF", "SJF"],
    n_seeds: int = 5,
    seed_offset: int = 1000
) -> pd.DataFrame:
    """
    Run Oracle baselines (no M limit) across regions and seeds.

    Args:
        regions: List of region names
        policies: List of policies ("LJF", "SJF")
        n_seeds: Number of evaluation seeds
        seed_offset: Starting seed (use 1000 to match existing evaluation)

    Returns:
        DataFrame with columns: region, policy, seed, completion_rate,
                               total_days, avg_queue_time, num_contractors
    """
    results = []
    total_scenarios = len(regions) * len(policies) * n_seeds
    current = 0

    print("=" * 80)
    print("ORACLE BASELINE EVALUATION")
    print("=" * 80)
    print(f"Regions: {len(regions)}")
    print(f"Policies: {policies}")
    print(f"Seeds: {n_seeds}")
    print(f"Total episodes: {total_scenarios}")
    print("=" * 80 + "\n")

    for region in regions:
        num_contractors = REGION_CONFIG[region]["num_contractors"]

        for policy in policies:
            for seed_idx in range(n_seeds):
                current += 1
                seed = seed_offset + seed_idx

                print(f"[{current}/{total_scenarios}] {policy} @ {region} (seed {seed_idx})...", end=" ")
                sys.stdout.flush()

                # Create Oracle environment (no M limit)
                env = OracleEnv(
                    region_key=region,
                    policy=policy,
                    num_contractors=num_contractors,
                    use_capacity_ramp=False,  # Match existing evaluation
                    seed=seed
                )

                # Run episode
                obs, info = env.reset()
                done = False
                episode_reward = 0.0
                step_count = 0
                queue_times = []

                while not done:
                    # Oracle env uses internal policy, action is ignored
                    obs, reward, terminated, truncated, info = env.step(None)
                    done = terminated or truncated
                    episode_reward += reward
                    step_count += 1

                    # Track queue time
                    if 'avg_queue_time' in info:
                        queue_times.append(info['avg_queue_time'])

                # Extract final metrics
                completion_rate = info.get('completion_rate', 0.0)
                avg_queue_time = np.mean(queue_times) if queue_times else info.get('avg_queue_time', 0.0)

                results.append({
                    'region': region,
                    'policy': f'Oracle-{policy}',
                    'seed': seed_idx,
                    'completion_rate': completion_rate,
                    'total_days': step_count,
                    'avg_queue_time': avg_queue_time,
                    'episode_reward': episode_reward,
                    'num_contractors': num_contractors
                })

                print(f"{step_count} days, {completion_rate:.1%} complete")

    return pd.DataFrame(results)


def main():
    """Main execution function."""
    # Define evaluation scope (100% crew only for quick validation)
    regions = [
        "Mataram", "West Lombok", "North Lombok", "Central Lombok",
        "East Lombok", "West Sumbawa", "Sumbawa"
    ]

    # Run Oracle evaluation
    oracle_results = run_oracle_evaluation(regions, policies=["LJF", "SJF"], n_seeds=5)

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "oracle_evaluation.csv"

    oracle_results.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print("="*80)

    # Print summary
    print("\n" + "="*80)
    print("ORACLE EVALUATION SUMMARY")
    print("="*80)

    summary = oracle_results.groupby(['region', 'policy']).agg({
        'completion_rate': 'mean',
        'total_days': 'mean',
        'avg_queue_time': 'mean',
        'num_contractors': 'first'
    }).round(2)

    print(summary.to_string())
    print("="*80 + "\n")

    # Load existing baseline results for comparison
    existing_file = output_dir / "all_methods_comparison.csv"
    if existing_file.exists():
        print("\n" + "="*80)
        print("COMPARISON: Oracle vs Regular Baselines")
        print("="*80)

        df_existing = pd.read_csv(existing_file)
        df_100 = df_existing[df_existing['crew_level'] == 1.0]

        # Get mean days for regular LJF/SJF
        regular_perf = df_100.groupby(['region', 'method'])['total_completion_days'].mean().reset_index()
        regular_perf = regular_perf.pivot(index='region', columns='method', values='total_completion_days')

        # Get Oracle performance
        oracle_perf = oracle_results.groupby(['region', 'policy'])['total_days'].mean().reset_index()
        oracle_perf = oracle_perf.pivot(index='region', columns='policy', values='total_days')

        # Merge
        comparison = regular_perf.copy()
        if 'Oracle-LJF' in oracle_perf.columns:
            comparison['Oracle-LJF'] = oracle_perf['Oracle-LJF']
            if 'LJF' in comparison.columns:
                comparison['LJF_improvement%'] = (
                    (comparison['LJF'] - comparison['Oracle-LJF']) / comparison['LJF'] * 100
                )

        if 'Oracle-SJF' in oracle_perf.columns:
            comparison['Oracle-SJF'] = oracle_perf['Oracle-SJF']
            if 'SJF' in comparison.columns:
                comparison['SJF_improvement%'] = (
                    (comparison['SJF'] - comparison['Oracle-SJF']) / comparison['SJF'] * 100
                )

        print(comparison.round(1).to_string())

        # Summary statistics
        print("\n" + "="*80)
        print("IMPROVEMENT SUMMARY")
        print("="*80)
        if 'LJF_improvement%' in comparison.columns:
            ljf_improvements = comparison['LJF_improvement%'].dropna()
            print(f"Oracle-LJF vs LJF: Mean improvement = {ljf_improvements.mean():.1f}% "
                  f"(range: {ljf_improvements.min():.1f}% to {ljf_improvements.max():.1f}%)")

        if 'SJF_improvement%' in comparison.columns:
            sjf_improvements = comparison['SJF_improvement%'].dropna()
            print(f"Oracle-SJF vs SJF: Mean improvement = {sjf_improvements.mean():.1f}% "
                  f"(range: {sjf_improvements.min():.1f}% to {sjf_improvements.max():.1f}%)")

        print("="*80 + "\n")

        # Decision guidance
        print("="*80)
        print("INTERPRETATION")
        print("="*80)
        improvements = []
        if 'LJF_improvement%' in comparison.columns:
            improvements.extend(comparison['LJF_improvement%'].dropna().tolist())
        if 'SJF_improvement%' in comparison.columns:
            improvements.extend(comparison['SJF_improvement%'].dropna().tolist())

        if improvements:
            mean_improvement = np.mean(improvements)
            if mean_improvement < 5:
                print("✗ Oracle improvement < 5%: Information/M-limit NOT important")
                print("  → Focus on environment contribution narrative")
                print("  → Limited value in further RL optimization")
            elif mean_improvement < 20:
                print("○ Oracle improvement 5-20%: Moderate potential")
                print("  → Consider targeted improvements (larger M, better candidate selection)")
            else:
                print("✓ Oracle improvement > 20%: Significant potential!")
                print("  → Pursue RL improvements: larger M, attention, hierarchical RL")

        print("="*80 + "\n")


if __name__ == "__main__":
    main()
