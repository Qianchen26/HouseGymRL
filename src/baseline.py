"""
Baseline Policies for House Reconstruction Scheduling

Provides baseline policies for comparison with RL:
- LJF (Longest Job First): Prioritizes houses with longest total work
- SJF (Shortest Job First): Prioritizes houses with shortest total work
- Random: Random prioritization
- Oracle-LJF/SJF: Same policies but with full queue visibility (no M limit)

These fixed policies serve to identify failure modes. The goal is not to beat
them everywhere, but to be robust across different scenarios.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

# Import our environment classes
from housegymrl import BaselineEnv, OracleEnv, HousegymRLENV
import config
from config import CROSS_AVAILABILITY_SCENARIOS


def create_baseline_env(
    region_key: str,
    policy: str,
    num_contractors: Optional[int] = None,
    use_batch_arrival: bool = True,
    use_capacity_ramp: bool = False,
    seed: Optional[int] = None,
) -> BaselineEnv:
    """
    Create a baseline environment with specified policy.

    Args:
        region_key: Region name from REGION_CONFIG
        policy: "LJF", "SJF", or "Random"
        num_contractors: Number of contractors (if None, use region default)
        use_batch_arrival: Whether to use batch arrival system
        use_capacity_ramp: Whether to use capacity ramp system
        seed: Random seed

    Returns:
        BaselineEnv instance
    """
    return BaselineEnv(
        region_key=region_key,
        policy=policy,
        num_contractors=num_contractors,
        use_batch_arrival=use_batch_arrival,
        use_capacity_ramp=use_capacity_ramp,
        seed=seed,
    )


def create_oracle_env(
    region_key: str,
    policy: str,
    num_contractors: Optional[int] = None,
    use_batch_arrival: bool = True,
    use_capacity_ramp: bool = False,
    seed: Optional[int] = None,
) -> OracleEnv:
    """
    Create an oracle environment (no candidate limit).

    Args:
        region_key: Region name from REGION_CONFIG
        policy: "LJF" or "SJF"
        num_contractors: Number of contractors (if None, use region default)
        use_batch_arrival: Whether to use batch arrival system
        use_capacity_ramp: Whether to use capacity ramp system
        seed: Random seed

    Returns:
        OracleEnv instance
    """
    return OracleEnv(
        region_key=region_key,
        policy=policy,
        num_contractors=num_contractors,
        use_batch_arrival=use_batch_arrival,
        use_capacity_ramp=use_capacity_ramp,
        seed=seed,
    )


def run_baseline_rollout(
    env: BaselineEnv,
    max_days: int = 1000,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a complete rollout with a baseline policy.

    Args:
        env: BaselineEnv or OracleEnv instance
        max_days: Maximum simulation days
        verbose: Whether to print progress

    Returns:
        Dictionary with rollout results:
        - completion_curve: Daily completion rates
        - final_completion: Final completion rate
        - makespan: Days to reach 100% (or max_days if not reached)
        - total_days: Total simulation days
    """
    env.reset()

    completion_curve = []
    simulated_days = 0

    while simulated_days < max_days:
        _, _, terminated, truncated, info = env.step(None)

        if info.get('day_advanced', False):
            completion = info.get('completion', 0.0)
            completion_curve.append(completion)

            if verbose and simulated_days % 100 == 0:
                print(f"Day {simulated_days}: {completion:.1%} complete, Queue: {info.get('queue_size', 0)}")

            simulated_days += 1

        if terminated or truncated:
            break

    completion_curve = np.array(completion_curve)

    if len(completion_curve) == 0:
        return {
            'completion_curve': completion_curve,
            'final_completion': 0.0,
            'makespan': 0,
            'total_days': 0,
        }

    # Calculate makespan (days to reach 100%)
    if completion_curve[-1] >= 0.99:
        makespan = np.argmax(completion_curve >= 0.99) + 1
    else:
        makespan = len(completion_curve)

    return {
        'completion_curve': completion_curve,
        'final_completion': completion_curve[-1] if len(completion_curve) > 0 else 0.0,
        'makespan': makespan,
        'total_days': len(completion_curve),
    }


def compare_baselines(
    region_key: str,
    seed: int = 42,
    max_days: int = 1000,
    use_batch_arrival: bool = True,
    use_capacity_ramp: bool = True,
) -> pd.DataFrame:
    """
    Compare all baseline policies on a given region.

    Args:
        region_key: Region name from REGION_CONFIG
        seed: Random seed for reproducibility
        max_days: Maximum simulation days
        use_batch_arrival: Whether to use batch arrival
        use_capacity_ramp: Whether to use capacity ramp

    Returns:
        DataFrame with comparison results
    """
    results = []

    # Test regular baselines
    for policy in ["LJF", "SJF", "Random"]:
        env = create_baseline_env(
            region_key=region_key,
            policy=policy,
            use_batch_arrival=use_batch_arrival,
            use_capacity_ramp=use_capacity_ramp,
            seed=seed,
        )

        rollout = run_baseline_rollout(env, max_days=max_days)

        results.append({
            'policy': policy,
            'type': 'baseline',
            'final_completion': rollout['final_completion'],
            'makespan': rollout['makespan'],
            'total_days': rollout['total_days'],
        })

    # Test oracle baselines
    for policy in ["LJF", "SJF"]:
        env = create_oracle_env(
            region_key=region_key,
            policy=policy,
            use_batch_arrival=use_batch_arrival,
            use_capacity_ramp=use_capacity_ramp,
            seed=seed,
        )

        rollout = run_baseline_rollout(env, max_days=max_days)

        results.append({
            'policy': f"Oracle-{policy}",
            'type': 'oracle',
            'final_completion': rollout['final_completion'],
            'makespan': rollout['makespan'],
            'total_days': rollout['total_days'],
        })

    df = pd.DataFrame(results)
    df['region'] = region_key

    return df


def test_baseline_robustness(
    policy: str,
    region_key: str,
    availability_levels: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    seed: int = 42,
    max_days: int = 1000,
) -> pd.DataFrame:
    """
    Test baseline robustness across different resource availability levels.

    Args:
        policy: Baseline policy to test
        region_key: Region name
        availability_levels: List of availability levels to test
        seed: Random seed
        max_days: Maximum simulation days

    Returns:
        DataFrame with robustness test results
    """
    results = []
    max_contractors = config.REGION_CONFIG[region_key]["num_contractors"]

    for availability in availability_levels:
        num_contractors = int(max_contractors * availability)

        env = create_baseline_env(
            region_key=region_key,
            policy=policy,
            num_contractors=num_contractors,
            use_batch_arrival=True,
            use_capacity_ramp=True,
            seed=seed,
        )

        rollout = run_baseline_rollout(env, max_days=max_days)

        results.append({
            'availability': availability,
            'num_contractors': num_contractors,
            'max_contractors': max_contractors,
            'final_completion': rollout['final_completion'],
            'makespan': rollout['makespan'],
        })

    df = pd.DataFrame(results)
    df['policy'] = policy
    df['region'] = region_key

    # Calculate robustness metrics
    df['completion_variance'] = df['final_completion'].var()
    df['makespan_variance'] = df['makespan'].var()

    return df


def make_baseline_scores(env: HousegymRLENV, policy: str = "SJF") -> np.ndarray:
    """
    Legacy function for backward compatibility.
    Generate scores for current candidate buffer.

    Args:
        env: HousegymRLENV instance (legacy)
        policy: "SJF", "LJF", or "Random"

    Returns:
        Array of scores for candidates
    """
    import warnings
    warnings.warn(
        "make_baseline_scores is deprecated. Use BaselineEnv instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Get candidate info using legacy method
    c = env.last_candidate_view()
    remain, mask = c["remain"], c["mask"]
    M = remain.shape[0]
    scores = np.zeros(M, dtype=np.float32)

    valid = mask > 0.0
    if not valid.any():
        return scores

    if policy == "SJF":
        # Shorter jobs get higher scores
        r = remain.copy()
        r[~valid] = r[valid].max() + 1.0
        scores = (r.max() + 1e-6) - r
    elif policy == "LJF":
        # Longer jobs get higher scores
        r = remain.copy()
        r[~valid] = -1.0
        scores = r - r.min()
    else:  # RANDOM
        rng = np.random.default_rng()
        scores = rng.random(M, dtype=np.float32)
        scores[~valid] = 0.0

    return np.clip(np.nan_to_num(scores, nan=0.0), 0.0, None)


if __name__ == "__main__":
    """Demo: Compare baselines on Mataram region."""

    print("="*60)
    print(" BASELINE POLICY COMPARISON")
    print("="*60)

    # Compare all baselines
    print("\n1. Comparing baselines on Mataram...")
    df = compare_baselines(
        region_key="Mataram",
        seed=42,
        use_batch_arrival=True,
        use_capacity_ramp=True,
    )

    print("\nResults:")
    print(df.to_string(index=False))

    # Information cost analysis
    print("\n2. Information Cost Analysis:")
    oracle_ljf = df[df['policy'] == 'Oracle-LJF']['makespan'].values[0]
    regular_ljf = df[df['policy'] == 'LJF']['makespan'].values[0]
    info_cost = (regular_ljf - oracle_ljf) / oracle_ljf * 100
    print(f"   LJF makespan: {regular_ljf} days")
    print(f"   Oracle-LJF makespan: {oracle_ljf} days")
    print(f"   Information cost: {info_cost:.1f}% longer due to M=512 limit")

    # Robustness test
    print("\n3. Testing LJF robustness across availability levels...")
    robustness_df = test_baseline_robustness(
        policy="LJF",
        region_key="Mataram",
        availability_levels=[0.1, 0.5, 1.0],
        seed=42,
    )

    print("\nRobustness results:")
    for _, row in robustness_df.iterrows():
        print(f"   Availability {row['availability']:.1f}: "
              f"Completion={row['final_completion']:.1%}, "
              f"Makespan={row['makespan']} days")

    print("\n" + "="*60)
    print(" KEY INSIGHTS")
    print("="*60)
    print("1. Different policies excel in different conditions")
    print("2. Oracle shows the cost of information limitation")
    print("3. Performance varies significantly with resource availability")
    print("4. No single baseline is universally optimal")
    print("   → This motivates learning adaptive policies with RL!")
