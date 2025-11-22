"""
Ablation experiment framework for systematic uncertainty mechanism testing.

This module provides tools to run multiple ablation experiments in parallel,
testing different combinations of uncertainty mechanisms to identify the
optimal configuration for PPO training.

Usage:
    python ablation_framework.py --run-all --parallel 4
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import pandas as pd
import subprocess
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent))

from ppo_configs import PPOConfig, TrainingConfig, EnvironmentConfig


@dataclass
class AblationConfig:
    """
    Configuration for a single ablation experiment.

    Attributes:
        experiment_name: Unique experiment identifier (string).
        description: Human-readable description (string, Chinese or English).
        stage: Experiment stage number (integer, 0-3).
        environment: Environment configuration dict with uncertainty parameters.
        training: Training configuration dict.
        evaluation: Evaluation configuration dict.
    """

    experiment_name: str
    description: str
    stage: int
    environment: Dict
    training: Dict
    evaluation: Dict


class AblationExperiment:
    """
    Manages a single ablation experiment: training and evaluation.

    Attributes:
        config: AblationConfig dataclass.
        log_dir: Path to experiment log directory.
    """

    def __init__(self, config_path: str):
        """
        Initialize ablation experiment from YAML configuration file.

        Args:
            config_path: Path to YAML configuration file (string).

        Example:
            >>> exp = AblationExperiment("configs/ablation/stage0_deterministic.yaml")
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        self.config = AblationConfig(**config_dict)
        self.log_dir = Path(f"experiments/ablation/{self.config.experiment_name}")
        self.results_file = self.log_dir / "results.csv"

    def setup_directories(self) -> None:
        """
        Create experiment directory structure.

        Creates:
            experiments/ablation/{experiment_name}/
            ├── logs/
            ├── checkpoints/
            └── results/

        Returns:
            None
        """
        (self.log_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.log_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.log_dir / "results").mkdir(parents=True, exist_ok=True)

    def run_training(self) -> int:
        """
        Execute PPO training for this experiment.

        Calls main_ppo.py via subprocess with experiment-specific parameters.

        Returns:
            Return code (integer): 0 if successful, non-zero otherwise.

        Side effects:
            - Writes training logs to {log_dir}/logs/train.out
            - Saves model to runs/{experiment_name}/

        Example:
            >>> exp = AblationExperiment("config.yaml")
            >>> return_code = exp.run_training()
        """
        self.setup_directories()

        cmd = [
            "python",
            "main_ppo.py",
            "--experiment-name", self.config.experiment_name,
            "--timesteps", str(self.config.training["total_timesteps"]),
            "--n-envs", str(self.config.training["n_envs"]),
        ]

        # Add uncertainty flags
        if not self.config.environment["use_batch_arrival"]:
            cmd.append("--no-batch-arrival")
        if not self.config.environment["stochastic_duration"]:
            cmd.append("--no-stochastic")
        if self.config.environment["observation_noise"] != 0.15:
            cmd.extend(["--obs-noise", str(self.config.environment["observation_noise"])])
        if self.config.environment["capacity_noise"] != 0.10:
            cmd.extend(["--capacity-noise", str(self.config.environment["capacity_noise"])])

        print(f"\n[{self.config.experiment_name}] Starting training...")
        print(f"Command: {' '.join(cmd)}")

        log_file = self.log_dir / "logs" / "train.out"
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

        if result.returncode == 0:
            print(f"[{self.config.experiment_name}] Training completed successfully")
        else:
            print(f"[{self.config.experiment_name}] Training failed with code {result.returncode}")

        return result.returncode

    def run_evaluation(self) -> pd.DataFrame:
        """
        Run evaluation on test regions with multiple seeds.

        Args:
            None

        Returns:
            DataFrame with columns: region, crew_level, seed, completion_rate,
            avg_queue_time, episode_reward, episode_length.
            Shape: (n_regions × n_crew_levels × n_seeds, 7)

        Side effects:
            - Saves results to {log_dir}/results/evaluation_results.csv

        Example:
            >>> results_df = exp.run_evaluation()
            >>> print(results_df.groupby('region')['completion_rate'].mean())
        """
        print(f"\n[{self.config.experiment_name}] Starting evaluation...")

        cmd = [
            "python",
            "evaluate_ppo.py",
            "--checkpoint-dir", f"runs/{self.config.experiment_name}",
            "--output-dir", str(self.log_dir / "results"),
        ]

        # Add test regions
        cmd.append("--test-regions")
        cmd.extend(self.config.evaluation["test_regions"])

        # Add crew levels
        cmd.append("--crew-levels")
        cmd.extend([str(x) for x in self.config.evaluation["crew_levels"]])

        # Add n_seeds
        cmd.extend(["--n-seeds", str(self.config.evaluation["n_seeds"])])

        log_file = self.log_dir / "logs" / "eval.out"
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

        if result.returncode == 0:
            print(f"[{self.config.experiment_name}] Evaluation completed")
            results_file = self.log_dir / "results" / "evaluation_results.csv"
            if results_file.exists():
                return pd.read_csv(results_file)
            else:
                print(f"Warning: Results file not found: {results_file}")
                return pd.DataFrame()
        else:
            print(f"[{self.config.experiment_name}] Evaluation failed")
            return pd.DataFrame()


class AblationSuite:
    """
    Manages a collection of ablation experiments for batch execution.

    Attributes:
        experiments: List of AblationExperiment instances.
        results_dir: Path to consolidated results directory.
    """

    def __init__(self, results_dir: str = "experiments/ablation"):
        """
        Initialize ablation suite.

        Args:
            results_dir: Base directory for all ablation results (string).
        """
        self.experiments: List[AblationExperiment] = []
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def add_experiment(self, config_path: str) -> None:
        """
        Add an experiment to the suite.

        Args:
            config_path: Path to experiment YAML config (string).

        Returns:
            None

        Example:
            >>> suite = AblationSuite()
            >>> suite.add_experiment("configs/ablation/stage0_deterministic.yaml")
        """
        exp = AblationExperiment(config_path)
        self.experiments.append(exp)
        print(f"Added experiment: {exp.config.experiment_name}")

    def add_all_from_directory(self, config_dir: str = "../configs/ablation") -> None:
        """
        Add all YAML configs from a directory.

        Args:
            config_dir: Directory containing YAML files (string).

        Returns:
            None

        Side effects:
            Sorts experiments by filename to ensure consistent order.
        """
        config_path = Path(config_dir)
        yaml_files = sorted(config_path.glob("*.yaml"))

        for yaml_file in yaml_files:
            self.add_experiment(str(yaml_file))

        print(f"\nTotal experiments added: {len(self.experiments)}")

    def run_all_sequential(self) -> None:
        """
        Run all experiments sequentially (one after another).

        Returns:
            None

        Example:
            >>> suite = AblationSuite()
            >>> suite.add_all_from_directory()
            >>> suite.run_all_sequential()
        """
        print(f"\n{'='*80}")
        print(f"Running {len(self.experiments)} experiments SEQUENTIALLY")
        print(f"{'='*80}\n")

        for i, exp in enumerate(self.experiments, 1):
            print(f"\n[{i}/{len(self.experiments)}] {exp.config.experiment_name}")
            print(f"Description: {exp.config.description}")

            # Training
            return_code = exp.run_training()
            if return_code != 0:
                print(f"Skipping evaluation due to training failure")
                continue

            # Evaluation
            exp.run_evaluation()

        print(f"\n{'='*80}")
        print("All experiments completed")
        print(f"{'='*80}\n")

    def run_all_parallel(self, n_jobs: int = 4) -> None:
        """
        Run all experiments in parallel using joblib.

        Args:
            n_jobs: Number of parallel jobs (integer).

        Returns:
            None

        Note:
            Requires joblib package: pip install joblib

        Example:
            >>> suite.run_all_parallel(n_jobs=4)
        """
        try:
            from joblib import Parallel, delayed
        except ImportError:
            print("Error: joblib not installed. Install with: pip install joblib")
            print("Falling back to sequential execution...")
            self.run_all_sequential()
            return

        print(f"\n{'='*80}")
        print(f"Running {len(self.experiments)} experiments in PARALLEL ({n_jobs} jobs)")
        print(f"{'='*80}\n")

        def run_exp(exp):
            print(f"Starting: {exp.config.experiment_name}")
            exp.run_training()
            exp.run_evaluation()
            return exp.config.experiment_name

        Parallel(n_jobs=n_jobs)(delayed(run_exp)(exp) for exp in self.experiments)

        print(f"\n{'='*80}")
        print("All parallel experiments completed")
        print(f"{'='*80}\n")

    def compare_results(self) -> pd.DataFrame:
        """
        Aggregate and compare results from all experiments.

        Returns:
            DataFrame with columns: experiment, stage, completion_rate_mean,
            completion_rate_std, avg_queue_time_mean, avg_queue_time_std,
            episode_reward_mean, episode_reward_std.
            Shape: (n_experiments, 8)

        Side effects:
            - Saves summary to: {results_dir}/comparison_summary.csv
            - Prints summary table to console

        Example:
            >>> summary = suite.compare_results()
            >>> print(summary.sort_values('completion_rate_mean', ascending=False))
        """
        print("\nAggregating results from all experiments...")

        all_results = []

        for exp in self.experiments:
            results_file = exp.log_dir / "results" / "evaluation_results.csv"

            if not results_file.exists():
                print(f"Warning: No results found for {exp.config.experiment_name}")
                continue

            df = pd.read_csv(results_file)
            df['experiment'] = exp.config.experiment_name
            df['stage'] = exp.config.stage
            all_results.append(df)

        if not all_results:
            print("No results to compare!")
            return pd.DataFrame()

        # Combine all results
        combined = pd.concat(all_results, ignore_index=True)

        # Compute summary statistics
        summary = combined.groupby(['experiment', 'stage']).agg({
            'completion_rate': ['mean', 'std'],
            'avg_queue_time': ['mean', 'std'],
            'episode_reward': ['mean', 'std'],
        }).round(3)

        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()

        # Sort by completion rate (descending)
        summary = summary.sort_values('completion_rate_mean', ascending=False)

        # Save summary
        summary_file = self.results_dir / "comparison_summary.csv"
        summary.to_csv(summary_file, index=False)

        print(f"\nSummary saved to: {summary_file}")
        print("\n" + "="*80)
        print("ABLATION RESULTS SUMMARY")
        print("="*80)
        print(summary.to_string(index=False))
        print("="*80 + "\n")

        return summary

    def generate_report(self, output_path: str = None) -> None:
        """
        Generate markdown report summarizing all experiments.

        Args:
            output_path: Path to save report (string or None).
                If None, saves to {results_dir}/ablation_report.md

        Returns:
            None

        Side effects:
            Creates markdown file with tables and analysis.
        """
        if output_path is None:
            output_path = self.results_dir / "ablation_report.md"

        summary = self.compare_results()

        if summary.empty:
            print("No results to generate report")
            return

        # Generate markdown content
        md_lines = [
            "# Uncertainty Ablation Experiment Report",
            "",
            "## Overview",
            f"- Total experiments: {len(self.experiments)}",
            f"- Stages: {sorted(summary['stage'].unique())}",
            "",
            "## Results Summary",
            "",
            summary.to_markdown(index=False),
            "",
            "## Stage Descriptions",
            "",
            "### Stage 0: Deterministic Baseline",
            "- All uncertainty mechanisms disabled",
            "- Perfect information for verifying basic learnability",
            "",
            "### Stage 1: Single Uncertainty",
            "- Test each mechanism individually",
            "- Identify primary obstacles",
            "",
            "### Stage 2: Dual Combinations",
            "- Test synergies and conflicts between mechanisms",
            "",
            "### Stage 3: Full and Recommended Configurations",
            "- Compare current setup vs. recommended minimal uncertainty",
            "",
            "## Key Findings",
            "",
            f"Best configuration: **{summary.iloc[0]['experiment']}**",
            f"- Completion rate: {summary.iloc[0]['completion_rate_mean']:.1%} ± {summary.iloc[0]['completion_rate_std']:.1%}",
            f"- Average queue time: {summary.iloc[0]['avg_queue_time_mean']:.1f} ± {summary.iloc[0]['avg_queue_time_std']:.1f} days",
            "",
        ]

        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))

        print(f"Report generated: {output_path}")


def main():
    """Command-line interface for ablation framework."""
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for uncertainty mechanisms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        default="../configs/ablation",
        help="Directory containing YAML experiment configs",
    )

    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all experiments in config directory",
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel jobs (1 = sequential)",
    )

    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare existing results without running experiments",
    )

    args = parser.parse_args()

    # Create suite
    suite = AblationSuite()

    if args.run_all:
        suite.add_all_from_directory(args.config_dir)

        if not args.compare_only:
            if args.parallel > 1:
                suite.run_all_parallel(n_jobs=args.parallel)
            else:
                suite.run_all_sequential()

    # Compare results
    suite.compare_results()
    suite.generate_report()


if __name__ == "__main__":
    main()
