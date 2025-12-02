"""
Configuration for HouseGym RL Environment

Contains all constants and parameters for:
- Work duration and crew limits per damage level
- Batch arrival and capacity ramp schedules
- Region data (Lombok earthquake)
- Evaluation settings
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

# Work duration parameters (per damage level)
WORK_PARAMS = {
    0: {"median": 30, "sigma": 0.4},
    1: {"median": 60, "sigma": 0.4},
    2: {"median": 120, "sigma": 0.4},
}

# Per-household daily crew cap by damage level
CMAX_BY_LEVEL = {0: 2, 1: 4, 2: 6}

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

# Disable capacity ramp (forces fixed capacity for simpler training dynamics)
CAPACITY_RAMP_ENABLED = False

# Candidate selection strategy: longest-waiting Top-M
CANDIDATE_SELECTION = "longest_wait"

# Environment sizing constants
MAX_QUEUE_SIZE = 1024  # Maximum candidate pool size (Step 1: keep 1024, can expand in Step 2/3)
MAX_STEPS = 500  # Maximum episode length
OBS_G = 6  # Global features (day, capacity, queue_size, etc.)
OBS_HOUSE_FEATURES = 6  # Features per candidate house: remain, waiting, total_work, damage, cmax, bias

# Backward compatibility aliases
M_FIXED = MAX_QUEUE_SIZE  # Legacy name for MAX_QUEUE_SIZE
OBS_F = OBS_HOUSE_FEATURES  # Legacy name for OBS_HOUSE_FEATURES
EXPECTED_OBS_DIM = OBS_G + M_FIXED * OBS_F  # 6 + 1024*6 = 6150 dimensions (for flat obs)

# Feature normalization scales (consistent between observation and scoring)
FEATURE_SCALES = np.array([
    100.0,  # observed_remain (typical range: 0-100+)
    100.0,  # waiting_time (typical range: 0-100+ days)
    100.0,  # total_work (typical range: 0-120)
    2.0,    # damage_level (0, 1, 2 → normalized to 0, 0.5, 1.0)
    10.0,   # cmax (max daily capacity: 2, 4, 6 → normalized to 0.2, 0.4, 0.6)
    1.0     # bias term (always 1.0, no scaling)
], dtype=np.float64)

# Learned heuristic parameters (Step 2 - legacy)
HEURISTIC_ACTION_DIM = 6  # Legacy: Number of weight parameters for learned heuristic
SCORING_FEATURES = [
    'observed_remain',
    'waiting_time',
    'total_work',
    'damage_level',
    'cmax',
    'bias'
]

# Action space parameters (per-candidate scores)
ACTION_DIM = MAX_QUEUE_SIZE  # Each candidate gets a score from policy
ACTION_LOW = -5.0  # Action lower bound (scores for softmax allocation)
ACTION_HIGH = 5.0  # Action upper bound (scores for softmax allocation)

# Allocation parameters (Step 2)
SOFTMAX_TEMPERATURE = 0.3  # Lower temperature to further concentrate allocation

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
def get_cross_availability_scenarios():
    """
    Generate cross-availability scenarios at call time.

    This function generates scenarios dynamically to avoid import-time
    issues when REGION_CONFIG is modified.

    Returns:
        List of scenario dictionaries with keys:
        - region: Region name
        - availability: Crew availability fraction
        - actual_crew: Number of contractors at this availability
        - max_crew: Maximum contractors for the region
        - scenario_id: Unique identifier string
    """
    scenarios = []
    for region_name, config in REGION_CONFIG.items():
        max_crew = config["num_contractors"]
        for availability in CREW_AVAILABILITY_LEVELS:
            scenarios.append({
                "region": region_name,
                "availability": availability,
                "actual_crew": int(max_crew * availability),
                "max_crew": max_crew,
                "scenario_id": f"{region_name}_av{availability:.1f}"
            })
    return scenarios


# Output directories (without timestamp)
OUTPUT_BASE_DIR = Path("output")
OUTPUT_DIR = OUTPUT_BASE_DIR / "default"
FIG_DIR = OUTPUT_DIR / "fig"
TAB_DIR = OUTPUT_DIR / "tab"

OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

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
DATA_DIR = Path(__file__).parent.parent / "data"
OBSERVED_DATA_PATH = DATA_DIR / "lombok_data.pkl"


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
