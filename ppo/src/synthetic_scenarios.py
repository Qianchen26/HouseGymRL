"""Utility for generating structured synthetic damage scenarios.

Three scenario families are supported:
1. Major-dominant
2. Balanced
3. Minor-dominant

Each family produces 50 standard samples (light jitter) plus 10 extreme
variants (strong bias + different crew ratios). Results can be exported to a
CSV file and/or registered with the global REGION_CONFIG via
``config.register_synthetic_region``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import register_synthetic_region


@dataclass(frozen=True)
class ScenarioTemplate:
    name: str
    base_mix: Tuple[float, float, float]
    dominant_index: int | None
    house_range: Tuple[int, int]
    contractor_ratio: Tuple[float, float]
    extreme_ratio: Tuple[float, float]


TEMPLATES: List[ScenarioTemplate] = [
    ScenarioTemplate(
        name="major_dominant",
        base_mix=(0.2, 0.25, 0.55),
        dominant_index=2,
        house_range=(10_000, 60_000),
        contractor_ratio=(0.10, 0.20),
        extreme_ratio=(0.05, 0.12),
    ),
    ScenarioTemplate(
        name="balanced",
        base_mix=(0.34, 0.33, 0.33),
        dominant_index=None,
        house_range=(8_000, 50_000),
        contractor_ratio=(0.12, 0.22),
        extreme_ratio=(0.07, 0.28),
    ),
    ScenarioTemplate(
        name="minor_dominant",
        base_mix=(0.55, 0.3, 0.15),
        dominant_index=0,
        house_range=(12_000, 70_000),
        contractor_ratio=(0.08, 0.18),
        extreme_ratio=(0.03, 0.10),
    ),
]


def _jitter_mix(
    base_mix: Tuple[float, float, float],
    rng: np.random.Generator,
    std: float,
) -> np.ndarray:
    """Apply Gaussian noise to a base mix and renormalize."""
    noise = rng.normal(0.0, std, size=3)
    raw = np.array(base_mix, dtype=float) + noise
    raw = np.clip(raw, 1e-3, None)
    return raw / raw.sum()


def _apply_extreme_bias(
    mix: np.ndarray,
    template: ScenarioTemplate,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """For extreme samples, bias toward/away from specific damage levels."""
    biased = mix.copy()

    if template.dominant_index is not None:
        idx = template.dominant_index
        biased[idx] = min(0.92, biased[idx] + intensity)
    else:
        # Balanced scenario extremes randomly favor minor or major
        idx = rng.choice([0, 2])
        biased[idx] = min(0.85, biased[idx] + intensity)

    # renormalize while keeping other components positive
    remainder = 1.0 - biased[idx]
    others = [i for i in range(3) if i != idx]
    current_other_sum = biased[others].sum()
    if current_other_sum <= 0:
        biased[others] = remainder / len(others)
    else:
        biased[others] *= remainder / current_other_sum

    return biased


def _counts_from_mix(mix: np.ndarray, total: int) -> Tuple[int, int, int]:
    raw = mix * total
    counts = np.floor(raw).astype(int)
    remainder = total - counts.sum()
    if remainder > 0:
        residual = raw - counts
        order = np.argsort(-residual)
        for idx in order[:remainder]:
            counts[idx] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def generate_scenarios(
    random_seed: int,
    normal_per_template: int = 50,
    extreme_per_template: int = 10,
    normal_std: float = 0.04,
    extreme_std: float = 0.12,
) -> pd.DataFrame:
    """Produce a DataFrame of synthetic regions for all templates."""
    rng = np.random.default_rng(random_seed)
    rows = []

    for template in TEMPLATES:
        total_samples = normal_per_template + extreme_per_template
        for idx in range(total_samples):
            is_extreme = idx >= normal_per_template
            std = extreme_std if is_extreme else normal_std

            mix = _jitter_mix(template.base_mix, rng, std)
            if is_extreme:
                mix = _apply_extreme_bias(mix, template, intensity=0.18, rng=rng)

            house_low, house_high = template.house_range
            total_houses = int(rng.integers(house_low, house_high + 1))
            ratio_low, ratio_high = (
                template.extreme_ratio if is_extreme else template.contractor_ratio
            )
            contractor_ratio = float(rng.uniform(ratio_low, ratio_high))
            num_contractors = max(1, int(round(total_houses * contractor_ratio)))

            minor, moderate, major = _counts_from_mix(mix, total_houses)
            # Fix rounding to ensure total matches
            adjust = total_houses - (minor + moderate + major)
            if adjust != 0:
                major += adjust

            scenario_id = (
                f"SCN_{template.name}_{'ext' if is_extreme else 'std'}"
                f"_{idx:02d}_{random_seed}"
            )
            rows.append(
                {
                    "region_key": scenario_id,
                    "scenario": template.name,
                    "cluster": template.name,
                    "is_extreme": bool(is_extreme),
                    "total_houses": total_houses,
                    "num_contractors": num_contractors,
                    "contractor_ratio": contractor_ratio,
                    "minor_count": minor,
                    "moderate_count": moderate,
                    "major_count": major,
                    "seed": int(rng.integers(0, 2**31 - 1)),
                }
            )

    return pd.DataFrame(rows)


def register_dataframe(df: pd.DataFrame) -> None:
    """Register scenarios in REGION_CONFIG."""
    for row in df.itertuples(index=False):
        damage_dist = [int(row.minor_count), int(row.moderate_count), int(row.major_count)]
        register_synthetic_region(
            H=int(row.total_houses),
            K=int(row.num_contractors),
            damage_dist=damage_dist,
            seed=int(row.seed),
            region_key=str(row.region_key),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate structured synthetic scenarios")
    parser.add_argument("--output", type=Path, default=Path("../results/synthetic_training_dataset.csv"),
                        help="CSV path to write the generated scenarios")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--normal", type=int, default=50, help="Normal samples per template")
    parser.add_argument("--extreme", type=int, default=10, help="Extreme samples per template")
    parser.add_argument("--register", action="store_true", help="Register scenarios in config after generation")
    args = parser.parse_args()

    df = generate_scenarios(
        random_seed=args.seed,
        normal_per_template=args.normal,
        extreme_per_template=args.extreme,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} scenarios to {args.output}")

    if args.register:
        register_dataframe(df)
        print("Scenarios registered in REGION_CONFIG")


if __name__ == "__main__":
    main()
