from __future__ import annotations

from datetime import datetime
from pathlib import Path

# Work duration parameters (per damage level)
WORK_PARAMS = {
    0: {"median": 30, "sigma": 0.4},
    1: {"median": 60, "sigma": 0.4},
    2: {"median": 120, "sigma": 0.4},
}

# Per-household daily crew cap by damage level
CMAX_BY_LEVEL = {0: 2, 1: 4, 2: 6}

# ============================================================================
# Batch Arrival + Capacity Ramp Configuration
# ============================================================================
COMBINED_ARRIVAL_CAPACITY_CONFIG = {
    # Batch arrival: Houses revealed over time
    "batch_arrival": {
        "days": [0, 30, 60],  # Days when new batches arrive
        "ratios": [0.40, 0.35, 0.25],  # Proportion of houses in each batch
    },

    # Capacity ramp: Reconstruction crew availability over time
    "capacity_ramp": {
        "warmup_days": 36,  # Days 0-36: K=0 (planning phase)
        "rise_days": 180,   # Days 36-216: K grows linearly from 0 to max
        "full_capacity_day": 216,  # Day 216+: K = max
    }
}

# Temporary switch: disable capacity ramp (forces fixed capacity even if configs exist)
CAPACITY_RAMP_ENABLED = False

# ============================================================================
# Candidate Selection Strategy
# ============================================================================
# IMPORTANT: Candidate selection is now PURE RANDOM
# - No artificial bias (no LJF/SJF pre-filtering)
# - RL must learn to identify important tasks from natural samples
# - Tests true robustness when candidate quality is not guaranteed
#
# This design choice is intentional:
# 1. More honest: We don't assume we know how to pre-filter
# 2. More natural: Real-world information acquisition is random
# 3. Stronger test: RL must work with whatever candidates appear
# 4. True robustness: Performance when we can't control input quality

CANDIDATE_SELECTION = "pure_random"  # The only strategy - no configuration needed

# Environment sizing constants
# PPO version: M is FIXED at 512 to avoid variable observation dimensions
M_FIXED = 512  # Fixed candidate pool size (no longer adaptive)
MAX_STEPS = 500
OBS_G = 6  # Global features (day, capacity, queue_size, etc.)
OBS_F = 4  # Per-candidate features (remaining_work, damage_level, etc.)
EXPECTED_OBS_DIM = OBS_G + M_FIXED * OBS_F  # 6 + 512*4 = 2054 dimensions

# ============================================================================
# Cross-availability configuration
# ============================================================================
CREW_AVAILABILITY_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

# Region configuration (counts follow v5 reference; seeds default to 42)
REGION_CONFIG = {
    "Mataram": {"damage_dist": [9500, 3672, 1345], "num_contractors": 9917, "seed": 42},
    "West Lombok": {"damage_dist": [45218, 13556, 14069], "num_contractors": 45208, "seed": 43},
    "North Lombok": {"damage_dist": [8889, 4772, 42049], "num_contractors": 22996, "seed": 44},
    "Central Lombok": {"damage_dist": [16639, 3096, 4483], "num_contractors": 15048, "seed": 45},
    "East Lombok": {"damage_dist": [12209, 4657, 10104], "num_contractors": 15404, "seed": 46},
    "West Sumbawa": {"damage_dist": [13078, 3803, 1283], "num_contractors": 10200, "seed": 47},
    "Sumbawa": {"damage_dist": [9652, 2756, 1374], "num_contractors": 10360, "seed": 48},
}

# Total scenarios: 7 regions × 6 availability levels = 42 scenarios
CROSS_AVAILABILITY_SCENARIOS = []
for region_name, config in REGION_CONFIG.items():
    max_crew = config["num_contractors"]
    for availability in CREW_AVAILABILITY_LEVELS:
        CROSS_AVAILABILITY_SCENARIOS.append({
            "region": region_name,
            "availability": availability,
            "actual_crew": int(max_crew * availability),
            "max_crew": max_crew,
            "scenario_id": f"{region_name}_av{availability:.1f}"
        })

# Output directory (基础目录，不包含时间戳)
OUTPUT_BASE_DIR = Path("output")
OUTPUT_DIR = OUTPUT_BASE_DIR / "default"  # 默认目录
FIG_DIR = OUTPUT_DIR / "fig"
TAB_DIR = OUTPUT_DIR / "tab"

# 确保基础目录存在
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Evaluation Configuration
# ============================================================================
EVAL_SEED = 42
AUC_TIME_POINTS = [200, 300]
MAKESPAN_THRESHOLD = 1.0

# Robustness Evaluation Metrics
ROBUSTNESS_METRICS = {
    "performance_stability": {
        "description": "Inverse of performance variance across scenarios",
        "lower_is_better": False,
    },
    "worst_case_ratio": {
        "description": "Worst performance / average baseline performance",
        "lower_is_better": False,
    },
    "adaptation_speed": {
        "description": "Performance recovery after condition change",
        "lower_is_better": False,
    },
    "failure_rate": {
        "description": "Fraction of scenarios with catastrophic failure",
        "lower_is_better": True,
        "failure_threshold": 0.7,  # Performance < 70% of baseline = failure
    },
}

# Cross-validation settings for generalization testing
GENERALIZATION_CONFIG = {
    "train_regions": ["Mataram"],  # Train on one region
    "test_regions": ["West Lombok", "North Lombok", "Central Lombok",
                     "East Lombok", "West Sumbawa", "Sumbawa"],  # Test on others
    "metrics": ["completion_rate", "makespan", "rmse"],
}

# Data location
# Point to project root data directory (from ppo/src/ -> housegymrl/data/)
DATA_DIR = Path(__file__).parent.parent.parent / "data"
OBSERVED_DATA_PATH = DATA_DIR / "lombok_data.pkl"

# ============================================================================
# Support for Synthetic Scenarios
# ============================================================================
def register_synthetic_region(
    H: int,
    K: int,
    damage_dist: list,
    seed: int | None = None,
    region_key: str | None = None,
):
    """
    Register a synthetic region dynamically for training.

    Args:
        H: Total number of houses
        K: Number of contractors
        damage_dist: Distribution [minor, moderate, major]
        seed: Random seed

    Returns:
        str: Region key for the synthetic scenario
    """
    if region_key is None:
        region_key = f"SYNTH_{H}_{K}_{seed if seed else 'random'}"
    REGION_CONFIG[region_key] = {
        "damage_dist": damage_dist,
        "num_contractors": K,
        "is_synthetic": True,
        "seed": seed
    }
    return region_key
