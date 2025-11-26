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
        device: Device for computation ('auto', 'cpu', 'cuda').
    """

    learning_rate: float = 2e-5
    n_steps: int = 2048
    batch_size: int = 512
    n_epochs: int = 3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.06
    ent_coef: float = 0.03
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
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
        use_batch_arrival: Enable batch arrival mechanism (days 0/30/60).
        stochastic_duration: Enable random work progress (±20% noise).
        observation_noise: Observation noise standard deviation as fraction of true value.
            Example: 0.15 → σ = 15% of true remaining work.
        capacity_noise: Capacity reduction range as fraction.
            Example: 0.10 → capacity sampled from [90%, 100%] of base.
        use_capacity_ramp: Enable capacity ramp (typically disabled).
        max_steps: Maximum episode length in days.
    """

    M_min: int = 1024
    M_max: int = 1024
    use_batch_arrival: bool = True
    stochastic_duration: bool = True
    observation_noise: float = 0.15
    capacity_noise: float = 0.10
    use_capacity_ramp: bool = False
    max_steps: int = 500


# Predefined configurations

PPO_DEFAULT = PPOConfig()

PPO_FAST = PPOConfig(
    learning_rate=5e-4,
    n_steps=1024,
    batch_size=128,
    n_epochs=5,
)

PPO_STABLE = PPOConfig(
    learning_rate=1e-4,
    n_steps=4096,
    batch_size=512,
    n_epochs=15,
    clip_range=0.1,
)

TRAINING_DEFAULT = TrainingConfig()

TRAINING_QUICK = TrainingConfig(
    total_timesteps=100_000,
    n_envs=8,
    save_freq=25_000,
)

TRAINING_LONG = TrainingConfig(
    total_timesteps=2_000_000,
    n_envs=32,
    save_freq=100_000,
)

ENV_DEFAULT = EnvironmentConfig()

ENV_DETERMINISTIC = EnvironmentConfig(
    use_batch_arrival=False,
    stochastic_duration=False,
    observation_noise=0.0,
    capacity_noise=0.0,
)

ENV_MINIMAL_UNCERTAINTY = EnvironmentConfig(
    use_batch_arrival=True,
    stochastic_duration=True,
    observation_noise=0.0,
    capacity_noise=0.0,
)
