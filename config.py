from __future__ import annotations

from datetime import datetime
from pathlib import Path

# Work duration parameters (per damage level)
WORK_PARAMS = {
    0: {"median": 30, "sigma": 0.4},
    1: {"median": 40, "sigma": 0.4},
    2: {"median": 50, "sigma": 0.4},
}

# Per-household daily crew cap by damage level
CMAX_BY_LEVEL = {0: 2, 1: 4, 2: 6}

# Three-batch arrival schedule
BATCH_ARRIVAL_CONFIG = {"days": [0, 7, 14], "ratios": [0.30, 0.40, 0.30]}

# Unified ramp configuration (used for train and eval)
UNIFIED_RAMP_PARAMS = {"warmup_days": 60, "rise_days": 120, "capacity_ratio": 1.0}

# Environment sizing constants
M_CANDIDATES = 96
MAX_STEPS = 1500
OBS_G = 6
OBS_F = 4
EXPECTED_OBS_DIM = OBS_G + M_CANDIDATES * OBS_F

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

# Output directory (Ã¥Å¸ÂºÃ§Â¡â‚¬Ã§â€ºÂ®Ã¥Â½â€¢Ã¯Â¼Å’Ã¤Â¸ÂÃ¥Å’â€¦Ã¥ÂÂ«Ã¦â€”Â¶Ã©â€”Â´Ã¦Ë†Â³)
OUTPUT_BASE_DIR = Path("output")
OUTPUT_DIR = OUTPUT_BASE_DIR / "default"  # Ã©Â»ËœÃ¨Â®Â¤Ã§â€ºÂ®Ã¥Â½â€¢
FIG_DIR = OUTPUT_DIR / "fig"
TAB_DIR = OUTPUT_DIR / "tab"

# Ã§Â¡Â®Ã¤Â¿ÂÃ¥Å¸ÂºÃ§Â¡â‚¬Ã§â€ºÂ®Ã¥Â½â€¢Ã¥Â­ËœÃ¥Å“Â¨
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Evaluation constants
EVAL_SEED = 42
AUC_TIME_POINTS = [200, 300]
MAKESPAN_THRESHOLD = 1.0

# Data location
DATA_DIR = Path("data")
OBSERVED_DATA_PATH = DATA_DIR / "lombok_data.pkl"