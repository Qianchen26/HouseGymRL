"""
PPO Training Script for HouseGym RL

Trains a PPO agent on the disaster recovery scheduling problem using
Stable-Baselines3. Supports multi-scenario training via synthetic data,
parallel environments, and VecNormalize for observation/reward scaling.

Usage:
    python main_ppo.py --experiment-name my_experiment --timesteps 500000
    python main_ppo.py --use-synthetic --timesteps 2000000
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from housegymrl import RLEnv
from ppo_configs import PPOConfig, TrainingConfig, EnvironmentConfig
from synthetic_scenarios import generate_scenarios, register_dataframe


class ProgressCallback(BaseCallback):
    """Custom callback to print training progress in SLURM logs and ensure TensorBoard flush."""

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.rollout_count = 0

    def _on_training_start(self):
        self.start_time = time.time()
        print(f"[Callback] Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        sys.stdout.flush()

    def _on_rollout_end(self):
        """Called after each rollout collection - print progress and flush TensorBoard."""
        if self.start_time is None:
            return True

        self.rollout_count += 1
        elapsed = time.time() - self.start_time
        progress = self.num_timesteps / self.total_timesteps * 100
        remaining = elapsed / self.num_timesteps * (self.total_timesteps - self.num_timesteps) if self.num_timesteps > 0 else 0

        # Print progress after EVERY rollout
        print(f"[Rollout {self.rollout_count}] Steps: {self.num_timesteps:,}/{self.total_timesteps:,} "
              f"({progress:.1f}%) | "
              f"Elapsed: {elapsed/60:.1f}min | "
              f"ETA: {remaining/60:.1f}min")

        # Print training metrics from logger if available
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            # Try to get episode reward mean from logger
            if hasattr(self.model.logger, 'name_to_value'):
                metrics = self.model.logger.name_to_value
                if 'rollout/ep_rew_mean' in metrics:
                    print(f"  └─ Episode reward (mean): {metrics['rollout/ep_rew_mean']:.2f}")
                if 'rollout/ep_len_mean' in metrics:
                    print(f"  └─ Episode length (mean): {metrics['rollout/ep_len_mean']:.1f}")

        sys.stdout.flush()

        # Force TensorBoard flush to disk every rollout
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            # Explicitly dump the logger to ensure TensorBoard writes
            self.model.logger.dump(step=self.num_timesteps)

        return True

    def _on_step(self):
        return True


class RewardComponentCallback(BaseCallback):
    """
    Log reward components to TensorBoard by aggregating info dicts.

    Expects env infos to include:
    - reward_progress
    - reward_completion
    - reward_queue_penalty
    - reward_raw_total
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.progress = []
        self.completion = []
        self.queue_penalty = []
        self.raw_total = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            if "reward_progress" in info:
                self.progress.append(info["reward_progress"])
            if "reward_completion" in info:
                self.completion.append(info["reward_completion"])
            if "reward_queue_penalty" in info:
                self.queue_penalty.append(info["reward_queue_penalty"])
            if "reward_raw_total" in info:
                self.raw_total.append(info["reward_raw_total"])
        return True

    def _on_rollout_end(self) -> bool:
        def _log_mean(name: str, data: list):
            if data:
                self.logger.record(name, float(np.mean(data)))
        _log_mean("reward_components/progress", self.progress)
        _log_mean("reward_components/completion", self.completion)
        _log_mean("reward_components/queue_penalty", self.queue_penalty)
        _log_mean("reward_components/raw_total", self.raw_total)
        self.progress.clear()
        self.completion.clear()
        self.queue_penalty.clear()
        self.raw_total.clear()
        return True


def create_training_env(
    rank: int,
    env_config: EnvironmentConfig,
    region_key: str = "Mataram",
    synthetic_region_keys: Optional[List[str]] = None,
    synthetic_seed: int = 42,
) -> Callable:
    """
    Create a single training environment.

    Args:
        rank: Environment rank for seed generation (integer, 0 to n_envs-1).
        env_config: Environment configuration dataclass containing uncertainty parameters.
        region_key: Region name from REGION_CONFIG (string, e.g., "Mataram").
            Used when synthetic_region_keys is None.
        synthetic_region_keys: List of synthetic region keys for multi-scenario training.
            If provided, each env randomly selects from these keys.
        synthetic_seed: Seed for generating synthetic scenarios (must match main process).

    Returns:
        Callable that creates and returns a Monitor-wrapped RLEnv instance.

    Example:
        >>> env_fn = create_training_env(0, ENV_DEFAULT, "Mataram")
        >>> env = env_fn()
    """
    def _init():
        # Select region: use synthetic rotation if available, else fixed region
        if synthetic_region_keys is not None and len(synthetic_region_keys) > 0:
            # IMPORTANT: Register synthetic scenarios in subprocess
            # SubprocVecEnv runs in separate processes that don't inherit main process state
            synthetic_df = generate_scenarios(random_seed=synthetic_seed)
            register_dataframe(synthetic_df)

            rng = np.random.default_rng(42 + rank)
            selected_region = rng.choice(synthetic_region_keys)
        else:
            selected_region = region_key

        env = RLEnv(
            region_key=selected_region,
            M_min=env_config.M_min,
            M_max=env_config.M_max,
            use_batch_arrival=env_config.use_batch_arrival,
            stochastic_duration=env_config.stochastic_duration,
            observation_noise=env_config.observation_noise,
            capacity_noise=env_config.capacity_noise,
            use_capacity_ramp=env_config.use_capacity_ramp,
            max_steps=env_config.max_steps,
            seed=42 + rank,
        )
        env = Monitor(env)
        return env
    return _init


def setup_vec_env(
    n_envs: int,
    env_config: EnvironmentConfig,
    region_key: str = "Mataram",
    use_subprocess: bool = True,
    synthetic_region_keys: Optional[List[str]] = None,
    synthetic_seed: int = 42,
) -> VecNormalize:
    """
    Set up vectorized environment with normalization.

    Args:
        n_envs: Number of parallel environments (integer, e.g., 16).
        env_config: Environment configuration dataclass.
        region_key: Region name (string). Used when synthetic_region_keys is None.
        use_subprocess: Use SubprocVecEnv for parallelization (boolean).
            If True, environments run in separate processes (faster).
            If False, use DummyVecEnv (sequential, easier debugging).
        synthetic_region_keys: List of synthetic region keys for multi-scenario training.
            If provided, each env randomly selects from these keys.

    Returns:
        VecNormalize instance wrapping vectorized environments.
        - Observations normalized to mean=0, std=1
        - Rewards normalized with discount factor gamma=0.99
        - Clipping applied: obs in [-10, 10], reward in [-10, 10]

    Example:
        >>> vec_env = setup_vec_env(8, ENV_DEFAULT)
        >>> obs = vec_env.reset()  # shape: (8, 2054)
    """
    env_fns = [
        create_training_env(i, env_config, region_key, synthetic_region_keys, synthetic_seed)
        for i in range(n_envs)
    ]

    if use_subprocess and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    return vec_env


def create_ppo_model(
    vec_env: VecNormalize,
    ppo_config: PPOConfig,
    tensorboard_log: str,
    verbose: int = 1
) -> PPO:
    """
    Create PPO model with specified hyperparameters.

    Args:
        vec_env: Vectorized environment with normalization.
        ppo_config: PPO hyperparameter configuration dataclass.
        tensorboard_log: Path to TensorBoard log directory (string).
        verbose: Verbosity level (0: silent, 1: info, 2: debug).

    Returns:
        PPO model instance ready for training.

    Example:
        >>> model = create_ppo_model(vec_env, PPO_DEFAULT, "runs/experiment1/tb_logs")
    """
    # Use MultiInputPolicy for Dict observation space
    # Network architecture: [256, 256] handles Dict obs from environment
    policy_kwargs = dict(net_arch=[256, 256])

    model = PPO(
        "MultiInputPolicy",  # Changed from MlpPolicy for Dict obs support
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=ppo_config.learning_rate,
        n_steps=ppo_config.n_steps,
        batch_size=ppo_config.batch_size,
        n_epochs=ppo_config.n_epochs,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
        clip_range=ppo_config.clip_range,
        ent_coef=ppo_config.ent_coef,
        vf_coef=ppo_config.vf_coef,
        max_grad_norm=ppo_config.max_grad_norm,
        device=ppo_config.device,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
    )

    return model


def setup_callbacks(
    experiment_name: str,
    save_dir: str,
    training_config: TrainingConfig,
    eval_env: Optional[VecNormalize] = None
) -> CallbackList:
    """
    Configure training callbacks for checkpointing and evaluation.

    Args:
        experiment_name: Experiment name for file naming (string).
        save_dir: Base directory for saving checkpoints (string, e.g., "runs/exp1").
        training_config: Training configuration dataclass.
        eval_env: Optional evaluation environment for periodic eval (VecNormalize or None).

    Returns:
        CallbackList containing CheckpointCallback and optionally EvalCallback.

    Side effects:
        Creates directories: {save_dir}/checkpoints/, {save_dir}/eval/

    Example:
        >>> callbacks = setup_callbacks("test_run", "runs/test", TRAINING_DEFAULT)
    """
    checkpoint_dir = Path(save_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Progress callback for SLURM logging
    progress_callback = ProgressCallback(total_timesteps=training_config.total_timesteps)

    checkpoint_callback = CheckpointCallback(
        save_freq=training_config.save_freq,
        save_path=str(checkpoint_dir),
        name_prefix=f"{experiment_name}_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    reward_callback = RewardComponentCallback()

    callbacks = [progress_callback, reward_callback, checkpoint_callback]

    if eval_env is not None:
        eval_dir = Path(save_dir) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(eval_dir),
            log_path=str(eval_dir),
            eval_freq=training_config.eval_freq,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

    return CallbackList(callbacks)


def train_ppo(
    experiment_name: str,
    region_key: str = "Mataram",
    ppo_config: PPOConfig = PPOConfig(),
    training_config: TrainingConfig = TrainingConfig(),
    env_config: EnvironmentConfig = EnvironmentConfig(),
    resume_from: Optional[str] = None,
    use_synthetic: bool = False,
    synthetic_seed: int = 42,
) -> None:
    """
    Main PPO training function.

    Args:
        experiment_name: Experiment name (string, used for saving and logging).
        region_key: Training region name (string, e.g., "Mataram").
            Used when use_synthetic is False.
        ppo_config: PPO hyperparameters (PPOConfig dataclass).
        training_config: Training loop settings (TrainingConfig dataclass).
        env_config: Environment settings (EnvironmentConfig dataclass).
        resume_from: Path to checkpoint to resume from (string or None).
        use_synthetic: If True, train on 180 synthetic scenarios for robustness.
        synthetic_seed: Seed for generating synthetic scenarios.

    Side effects:
        - Creates directory structure: runs/{experiment_name}/
        - Saves final model to: runs/{experiment_name}/model.zip
        - Saves VecNormalize to: runs/{experiment_name}/vecnormalize.pkl
        - Writes TensorBoard logs to: runs/{experiment_name}/tb_logs/

    Returns:
        None

    Example:
        >>> train_ppo("test_run", timesteps=100000)
    """
    print(f"\n{'='*80}")
    print(f"Starting PPO Training: {experiment_name}")
    print(f"{'='*80}\n")

    # Handle synthetic scenario generation
    synthetic_region_keys = None
    if use_synthetic:
        print("Generating 180 synthetic training scenarios...")
        synthetic_df = generate_scenarios(random_seed=synthetic_seed)
        register_dataframe(synthetic_df)
        synthetic_region_keys = synthetic_df['region_key'].tolist()
        print(f"Registered {len(synthetic_region_keys)} synthetic scenarios")
        print(f"Training mode: Multi-scenario rotation (robustness training)")
    else:
        print(f"Region: {region_key}")
        print(f"Training mode: Single region")

    print(f"Total timesteps: {training_config.total_timesteps:,}")
    print(f"Parallel envs: {training_config.n_envs}")
    print(f"Uncertainty: batch_arrival={env_config.use_batch_arrival}, "
          f"stochastic={env_config.stochastic_duration}, "
          f"obs_noise={env_config.observation_noise}, "
          f"cap_noise={env_config.capacity_noise}\n")

    # Setup directories
    save_dir = Path(f"runs/{experiment_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    tb_log_dir = save_dir / "tb_logs"
    tb_log_dir.mkdir(exist_ok=True)

    # Create vectorized environment
    print("Creating vectorized environment...")
    vec_env = setup_vec_env(
        n_envs=training_config.n_envs,
        env_config=env_config,
        region_key=region_key,
        use_subprocess=True,
        synthetic_region_keys=synthetic_region_keys,
        synthetic_seed=synthetic_seed,
    )

    # Create or load model
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        model = PPO.load(resume_from, env=vec_env)

        # Try to load VecNormalize stats
        vecnorm_path = Path(resume_from).parent / "vecnormalize.pkl"
        if vecnorm_path.exists():
            vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
            print(f"Loaded VecNormalize from {vecnorm_path}")
    else:
        print("Creating new PPO model...")
        model = create_ppo_model(
            vec_env=vec_env,
            ppo_config=ppo_config,
            tensorboard_log=str(tb_log_dir),
            verbose=1,
        )

    # Setup callbacks
    print("Setting up callbacks...")
    callbacks = setup_callbacks(
        experiment_name=experiment_name,
        save_dir=str(save_dir),
        training_config=training_config,
        eval_env=None,  # Can add eval env later if needed
    )

    # Train
    print(f"\n{'='*80}")
    print("Training started...")
    print(f"Total timesteps: {training_config.total_timesteps:,}")
    print(f"Steps per rollout: {ppo_config.n_steps * training_config.n_envs:,}")
    print(f"Expected rollouts: {training_config.total_timesteps // (ppo_config.n_steps * training_config.n_envs)}")
    print(f"{'='*80}\n")
    print("Starting first rollout (this may take a few minutes)...")
    import sys
    sys.stdout.flush()  # Force output to appear immediately

    model.learn(
        total_timesteps=training_config.total_timesteps,
        callback=callbacks,
        log_interval=training_config.log_interval,
        progress_bar=False,  # Disable progress bar for SLURM
    )

    print("\n[Training loop completed]")
    sys.stdout.flush()

    # Save final model and VecNormalize
    print("\nSaving final model...")
    model.save(str(save_dir / "model"))
    vec_env.save(str(save_dir / "vecnormalize.pkl"))

    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Model saved to: {save_dir / 'model.zip'}")
    print(f"VecNormalize saved to: {save_dir / 'vecnormalize.pkl'}")
    print(f"TensorBoard logs: {tb_log_dir}")
    print(f"{'='*80}\n")

    vec_env.close()


def main():
    """Command-line interface for PPO training."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent on HouseGym disaster recovery problem",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Experiment name for saving and logging",
    )

    parser.add_argument(
        "--region",
        type=str,
        default="Mataram",
        choices=["Mataram", "West Lombok", "North Lombok", "Central Lombok",
                 "East Lombok", "West Sumbawa", "Sumbawa"],
        help="Training region name",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps",
    )

    parser.add_argument(
        "--n-envs",
        type=int,
        default=16,
        help="Number of parallel environments",
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g., runs/exp1/checkpoints/model_100000_steps.zip)",
    )

    # Uncertainty toggles
    parser.add_argument(
        "--no-batch-arrival",
        action="store_true",
        help="Disable batch arrival (all houses revealed at day 0)",
    )

    parser.add_argument(
        "--no-stochastic",
        action="store_true",
        help="Disable stochastic work duration",
    )

    parser.add_argument(
        "--obs-noise",
        type=float,
        default=0.15,
        help="Observation noise level (0.0 = perfect info)",
    )

    parser.add_argument(
        "--capacity-noise",
        type=float,
        default=0.10,
        help="Capacity noise level (0.0 = fixed capacity)",
    )

    # Synthetic training
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Train on 180 synthetic scenarios for robustness (ignores --region)",
    )

    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=42,
        help="Seed for synthetic scenario generation",
    )

    args = parser.parse_args()

    # Build configurations
    training_config = TrainingConfig(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
    )

    env_config = EnvironmentConfig(
        use_batch_arrival=not args.no_batch_arrival,
        stochastic_duration=not args.no_stochastic,
        observation_noise=args.obs_noise,
        capacity_noise=args.capacity_noise,
    )

    # Run training
    train_ppo(
        experiment_name=args.experiment_name,
        region_key=args.region,
        ppo_config=PPOConfig(),
        training_config=training_config,
        env_config=env_config,
        resume_from=args.resume_from,
        use_synthetic=args.use_synthetic,
        synthetic_seed=args.synthetic_seed,
    )


if __name__ == "__main__":
    main()
