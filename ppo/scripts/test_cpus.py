#!/usr/bin/env python3
"""
Test CPU availability and multiprocessing performance.
"""

import os
import sys
import time
import multiprocessing as mp
import numpy as np


def cpu_intensive_task(n):
    """CPU-intensive task for testing."""
    result = 0
    for i in range(1000000):
        result += np.sin(i) * np.cos(i)
    return result


def test_cpu_info():
    """Display CPU information."""
    print("=" * 80)
    print("CPU Information")
    print("=" * 80)
    print(f"Total CPU cores (physical + logical): {mp.cpu_count()}")
    print(f"Usable CPUs (os.cpu_count()): {os.cpu_count()}")

    # Check SLURM environment
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    slurm_job_cpus = os.environ.get('SLURM_JOB_CPUS_PER_NODE')

    if slurm_cpus:
        print(f"SLURM allocated CPUs (SLURM_CPUS_PER_TASK): {slurm_cpus}")
    if slurm_job_cpus:
        print(f"SLURM job CPUs (SLURM_JOB_CPUS_PER_NODE): {slurm_job_cpus}")

    # CPU affinity
    if hasattr(os, 'sched_getaffinity'):
        affinity = os.sched_getaffinity(0)
        print(f"CPU affinity (allowed CPUs): {len(affinity)} cores")
        print(f"  Cores: {sorted(affinity)}")

    print()


def test_multiprocessing(n_workers):
    """Test multiprocessing performance."""
    print(f"Testing with {n_workers} workers...")

    start_time = time.time()

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(cpu_intensive_task, range(n_workers))

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.2f} seconds")
    print(f"  Throughput: {n_workers / elapsed:.2f} tasks/second")

    return elapsed


def main():
    """Main test function."""
    print("\n" + "=" * 80)
    print("HiPerGator CPU Test")
    print("=" * 80 + "\n")

    # Display CPU info
    test_cpu_info()

    # Performance test
    print("=" * 80)
    print("Multiprocessing Performance Test")
    print("=" * 80)

    max_cpus = os.cpu_count() or mp.cpu_count()
    test_configs = [1, 4, 8, 16]

    # Only test configs that are <= available CPUs
    test_configs = [n for n in test_configs if n <= max_cpus]

    if max_cpus > 16:
        test_configs.append(min(32, max_cpus))

    print(f"Testing configurations: {test_configs}")
    print()

    results = {}
    for n_workers in test_configs:
        elapsed = test_multiprocessing(n_workers)
        results[n_workers] = elapsed

    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print(f"{'Workers':<10} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 80)

    baseline = results[1]
    for n_workers in sorted(results.keys()):
        elapsed = results[n_workers]
        speedup = baseline / elapsed
        print(f"{n_workers:<10} {elapsed:<12.2f} {speedup:<10.2f}x")

    print("\n" + "=" * 80)
    print("Recommendation for PPO Training")
    print("=" * 80)

    # Find optimal number of workers
    max_speedup = max(baseline / results[n] for n in results)
    optimal_workers = [n for n in results if baseline / results[n] >= max_speedup * 0.9][0]

    print(f"Available CPUs: {max_cpus}")
    print(f"Optimal workers for parallel tasks: {optimal_workers}")
    print(f"Expected speedup: {max_speedup:.2f}x over single-core")
    print()
    print("For PPO training:")
    print(f"  Recommended n_envs: {optimal_workers}")
    print(f"  This will run {optimal_workers} environments in parallel")
    print()


if __name__ == "__main__":
    main()
