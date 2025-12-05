"""
PPO Hyperparameter Configurations

Defines PPO configurations for different experimental scenarios.
All configurations use fixed M=1024 to avoid variable observation dimensions.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PPOConfig:
    """
    PPO algorithm hyperparameters.

    Attributes:
        learning_rate: Learning rate for optimizer, controls parameter update step size.
            Range: [1e-5, 1e-3]. Lower values → more stable but slower learning.
        n_steps: Number of environment steps collected before each policy update.
            Range: [1024, 4096]. Higher values → more diverse experience but slower updates.
        batch_size: Mini-batch size for gradient descent.
            Range: [64, 512]. Must divide n_steps evenly.
        n_epochs: Number of optimization epochs per update.
            Range: [4, 20]. Higher values → more thorough optimization but risk overfitting.
        gamma: Discount factor for future rewards.
            Range: [0.95, 0.999]. Higher values → prioritize long-term rewards.
        gae_lambda: GAE lambda parameter, controls bias-variance tradeoff in advantage estimation.
            Range: [0.9, 0.99]. Higher values → lower bias but higher variance.
        clip_range: PPO clipping range for probability ratio.
            Range: [0.1, 0.3]. Controls maximum policy change per update.
        ent_coef: Entropy coefficient, encourages exploration.
            Range: [0.0, 0.1]. Higher values → more random actions.
        vf_coef: Value function loss coefficient.
            Range: [0.1, 1.0]. Controls importance of value function training.
        max_grad_norm: Maximum gradient norm for clipping.
            Range: [0.3, 1.0]. Prevents gradient explosion.
        target_kl: Target KL divergence for early stopping.
            Range: [0.01, 0.05]. If KL exceeds this, stop current epoch.
            Set to None to disable.
        device: Device for computation ('auto', 'cpu', 'cuda').
    """

    learning_rate: float = 1e-4
    n_steps: int = 2048
    batch_size: int = 512
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.10
    ent_coef: float = 0.01  # Entropy normalized in policy (per-candidate mean), safe to use
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None  # KL early stopping threshold
    device: str = 'auto'


@dataclass
class TrainingConfig:
    """
    Training loop configuration.

    Attributes:
        total_timesteps: Total environment steps to train.
            Example: 500,000 steps ≈ 1,000 episodes (500 steps each).
        n_envs: Number of parallel environments.
            Range: [4, 32]. More envs → faster data collection but more CPU/memory.
        save_freq: Checkpoint save frequency in environment steps.
            Example: 50,000 steps → save every 100 episodes.
        eval_freq: Evaluation frequency in environment steps.
            Example: 100,000 steps → evaluate every 200 episodes.
        log_interval: TensorBoard logging interval in episodes.
            Example: 10 → log average reward every 10 episodes.
    """

    total_timesteps: int = 1_000_000
    n_envs: int = 16
    save_freq: int = 50_000
    eval_freq: int = 100_000
    log_interval: int = 1  # Log every episode for faster feedback


@dataclass
class EnvironmentConfig:
    """
    Environment configuration for PPO training.

    Attributes:
        M_min: Minimum number of candidates (fixed at 512 for PPO).
        M_max: Maximum number of candidates (fixed at 512 for PPO).
        stochastic_duration: Enable random work progress (±20% noise).
        observation_noise: Observation noise standard deviation as fraction of true value.
            Example: 0.15 → σ = 15% of true remaining work.
        capacity_noise: Capacity reduction range as fraction.
            Example: 0.10 → capacity sampled from [90%, 100%] of base.
        use_capacity_ramp: Enable capacity ramp (typically disabled).
        max_steps: Maximum episode length in days.
        capacity_ceiling: Maximum daily capacity. If None and use_legacy_capacity_ceiling=False,
            no ceiling is applied. Default: None.
        use_legacy_capacity_ceiling: If True, use legacy formula M_max * max(cmax_per_day).
            This can truncate large contractor pools. Default: False.
    """

    M_min: int = 1024
    M_max: int = 1024
    stochastic_duration: bool = True
    observation_noise: float = 0.15
    capacity_noise: float = 0.10
    use_capacity_ramp: bool = False
    max_steps: int = 500
    capacity_ceiling: Optional[int] = None
    use_legacy_capacity_ceiling: bool = False


# Predefined configuration
PPO_DEFAULT = PPOConfig(
    learning_rate=5e-5,    # Will use linear decay in main_ppo.py
    n_steps=2048,          # Rollout length before each update
    batch_size=512,        # Mini-batch size for SGD
    n_epochs=4,            # Optimization passes per rollout
    clip_range=0.08,       # V1 config: best final reward (630)
    ent_coef=0.02,         # Entropy bonus for exploration
)

TRAINING_DEFAULT = TrainingConfig(
    total_timesteps=1_000_000,
    n_envs=16,
    save_freq=50_000,
    eval_freq=100_000,
    log_interval=1,
)

ENV_DEFAULT = EnvironmentConfig(
    M_min=1024,
    M_max=1024,
    stochastic_duration=True,
    observation_noise=0.15,
    capacity_noise=0.10,
    use_capacity_ramp=False,
    max_steps=500,
)
