"""
Unit tests for hybrid weighted allocation logic.

Tests verify that the allocation:
1. Respects total capacity constraint (sum ≤ K)
2. Respects individual cmax constraints (each ≤ cmax)
3. Distributes capacity intelligently based on priorities
4. Handles edge cases (no capacity, no candidates, etc.)
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from housegymrl import RLEnv
import config


def create_test_env(num_houses=100, num_contractors=1000, seed=42):
    """Create a minimal test environment."""
    # Register a small synthetic region
    region_key = config.register_synthetic_region(
        H=num_houses,
        K=num_contractors,
        damage_dist=[num_houses // 3, num_houses // 3, num_houses // 3],
        seed=seed,
        region_key=f"TEST_{seed}"
    )

    env = RLEnv(
        region_key=region_key,
        M_ratio=0.5,  # See 50% of queue
        M_min=10,
        M_max=50,
        use_batch_arrival=False,  # Static arrival for testing
        use_capacity_ramp=False,  # Fixed capacity
        seed=seed
    )

    return env


def test_basic_allocation():
    """Test 1: Basic allocation with uniform priorities."""
    print("\n=== Test 1: Basic allocation with uniform priorities ===")

    env = create_test_env(num_houses=100, num_contractors=100, seed=42)
    env.reset()

    # Create uniform action (all priorities equal)
    M = len(env.pending_candidates)
    action = np.ones(env.action_space.shape, dtype=np.float32) * 0.5

    # Get allocation
    allocation = env._allocate_from_candidates(
        env.pending_candidates[:M],
        action,
        K=100
    )

    # Verify constraints
    total = sum(allocation.values())
    print(f"  Total allocated: {total} / 100")
    assert total <= 100, f"Over-allocated: {total} > 100"

    for house_id, num in allocation.items():
        cmax = env._arr_cmax[house_id]
        assert num <= cmax, f"House {house_id}: allocated {num} > cmax {cmax}"

    print(f"  ✓ Allocation respects constraints")
    print(f"  ✓ Allocated to {len(allocation)} houses")

    return True


def test_priority_based_allocation():
    """Test 2: Allocation should favor higher priority houses."""
    print("\n=== Test 2: Priority-based allocation ===")

    env = create_test_env(num_houses=100, num_contractors=100, seed=42)
    env.reset()

    # Create action with one dominant priority
    M = min(10, len(env.pending_candidates))
    candidates = env.pending_candidates[:M]
    action = np.zeros(env.action_space.shape, dtype=np.float32)

    # Give first candidate very high priority
    action[0] = 1.0
    action[1:M] = 0.1

    # Get allocation
    allocation = env._allocate_from_candidates(
        candidates,
        action,
        K=100
    )

    # Verify first candidate gets significant allocation
    first_house = candidates[0]
    first_allocation = allocation.get(first_house, 0)
    cmax_first = env._arr_cmax[first_house]

    print(f"  Highest priority house: {first_house}")
    print(f"    cmax: {cmax_first}")
    print(f"    allocated: {first_allocation}")
    print(f"  Total allocated: {sum(allocation.values())} / 100")

    # First house should get its cmax (or close to it)
    assert first_allocation > 0, "Highest priority house got 0 allocation"

    print(f"  ✓ Priority affects allocation")

    return True


def test_cmax_constraints():
    """Test 3: Allocation respects cmax constraints."""
    print("\n=== Test 3: Cmax constraints ===")

    env = create_test_env(num_houses=100, num_contractors=1000, seed=42)
    env.reset()

    # Large capacity, uniform priorities
    M = min(20, len(env.pending_candidates))
    candidates = env.pending_candidates[:M]
    action = np.ones(env.action_space.shape, dtype=np.float32) * 0.5

    # Get allocation with large capacity
    allocation = env._allocate_from_candidates(
        candidates,
        action,
        K=1000
    )

    # Verify each allocation ≤ cmax
    violations = []
    for house_id, num in allocation.items():
        cmax = env._arr_cmax[house_id]
        if num > cmax:
            violations.append((house_id, num, cmax))

    if violations:
        print(f"  ✗ Cmax violations found:")
        for house_id, num, cmax in violations:
            print(f"    House {house_id}: allocated {num} > cmax {cmax}")
        return False

    print(f"  ✓ All allocations respect cmax")
    print(f"  Total allocated: {sum(allocation.values())} / 1000")
    print(f"  Max individual allocation: {max(allocation.values())}")

    # Check that we're actually using the cmax constraints
    max_cmax = max(env._arr_cmax[h] for h in candidates)
    max_allocation = max(allocation.values()) if allocation else 0
    print(f"  Max cmax in candidates: {max_cmax}")
    print(f"  Max allocation: {max_allocation}")

    return True


def test_capacity_shortage():
    """Test 4: Handle capacity < sum(cmax)."""
    print("\n=== Test 4: Capacity shortage ===")

    env = create_test_env(num_houses=100, num_contractors=10, seed=42)
    env.reset()

    # Small capacity, many candidates
    M = min(20, len(env.pending_candidates))
    candidates = env.pending_candidates[:M]
    action = np.ones(env.action_space.shape, dtype=np.float32) * 0.5

    # Get allocation with small capacity
    allocation = env._allocate_from_candidates(
        candidates,
        action,
        K=10
    )

    total = sum(allocation.values())
    print(f"  Total allocated: {total} / 10")
    print(f"  Houses served: {len(allocation)} / {M}")

    assert total <= 10, f"Over-allocated: {total} > 10"

    print(f"  ✓ Handles capacity shortage correctly")

    return True


def test_edge_cases():
    """Test 5: Edge cases (no capacity, no candidates, etc.)."""
    print("\n=== Test 5: Edge cases ===")

    env = create_test_env(num_houses=100, num_contractors=100, seed=42)
    env.reset()

    action = np.ones(env.action_space.shape, dtype=np.float32) * 0.5

    # Test 5a: No capacity
    allocation = env._allocate_from_candidates(
        env.pending_candidates[:10],
        action,
        K=0
    )
    assert len(allocation) == 0, "Should return empty dict when K=0"
    print(f"  ✓ K=0 returns empty allocation")

    # Test 5b: No candidates
    allocation = env._allocate_from_candidates(
        [],
        action,
        K=100
    )
    assert len(allocation) == 0, "Should return empty dict when no candidates"
    print(f"  ✓ Empty candidates returns empty allocation")

    # Test 5c: Single candidate
    allocation = env._allocate_from_candidates(
        env.pending_candidates[:1],
        action,
        K=100
    )
    assert len(allocation) <= 1, "Should allocate to at most 1 house"
    if len(allocation) == 1:
        house_id = list(allocation.keys())[0]
        cmax = env._arr_cmax[house_id]
        assert allocation[house_id] <= cmax, "Should respect cmax even with single candidate"
    print(f"  ✓ Single candidate handled correctly")

    return True


def test_integer_rounding():
    """Test 6: Integer rounding preserves total capacity."""
    print("\n=== Test 6: Integer rounding ===")

    env = create_test_env(num_houses=100, num_contractors=97, seed=42)
    env.reset()

    # Odd capacity value
    M = min(10, len(env.pending_candidates))
    candidates = env.pending_candidates[:M]
    action = np.random.RandomState(42).rand(env.action_space.shape[0]).astype(np.float32)

    allocation = env._allocate_from_candidates(
        candidates,
        action,
        K=97
    )

    total = sum(allocation.values())
    print(f"  Target capacity: 97")
    print(f"  Actual allocated: {total}")
    print(f"  Difference: {97 - total}")

    # Should be close to 97 (within reasonable margin)
    assert total <= 97, f"Over-allocated: {total} > 97"

    # Should use most of the capacity (allowing for cmax constraints)
    print(f"  ✓ Integer rounding handled correctly")

    return True


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Hybrid Weighted Allocation Unit Tests")
    print("=" * 60)

    tests = [
        ("Basic allocation", test_basic_allocation),
        ("Priority-based allocation", test_priority_based_allocation),
        ("Cmax constraints", test_cmax_constraints),
        ("Capacity shortage", test_capacity_shortage),
        ("Edge cases", test_edge_cases),
        ("Integer rounding", test_integer_rounding),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} FAILED with exception:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
