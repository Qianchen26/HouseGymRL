"""
Attention-based Policy for HouseGymRL.

This module implements a single-head attention mechanism with per-candidate
scoring for allocation decisions. Each candidate receives a score from the
policy, enabling direct control over resource allocation.

Architecture:
    Input:
        - global_features: (B, 6) system state
        - candidates: (B, MAX_Q, 6) house features
        - mask: (B, MAX_Q) validity mask (1.0=valid, 0.0=padding)

    Processing:
        1. Encode candidates: Linear(6 -> 64) -> house_embeddings (B, MAX_Q, 64)
        2. Encode global as query: Linear(6 -> 64) -> query
        3. Single-head attention: query attends to house_embeddings -> context
        4. Score head: Each candidate embedding -> MLP -> scalar score
        5. Value head: Pooled embeddings + global -> MLP -> value

    Output:
        - action: (B, MAX_Q) per-candidate scores for softmax allocation
        - value: (B, 1) state value estimate
"""

from __future__ import annotations

from typing import Dict, Tuple, Type, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Constants (should match config.py)
ATTENTION_EMBED_DIM = 64
ATTENTION_NUM_HEADS = 1
GLOBAL_FEATURE_DIM = 6
HOUSE_FEATURE_DIM = 6


class AttentionEncoder(nn.Module):
    """
    Single-head attention encoder for aggregating candidate house features.

    Query: derived from global system state (what is the current situation?)
    Key/Value: derived from candidate house features (which houses matter?)

    The attention mechanism produces a context vector that summarizes the
    most relevant aspects of the candidate queue given the current state.

    Attributes:
        house_encoder: Linear layer to encode house features to embeddings.
        query_encoder: Linear layer to encode global features to query.
        attention: Multi-head attention layer (configured as single-head).
    """

    def __init__(
        self,
        house_feature_dim: int = HOUSE_FEATURE_DIM,
        global_feature_dim: int = GLOBAL_FEATURE_DIM,
        embed_dim: int = ATTENTION_EMBED_DIM,
        num_heads: int = ATTENTION_NUM_HEADS,
    ) -> None:
        """
        Initialize the attention encoder.

        Args:
            house_feature_dim: Dimension of each house's feature vector.
            global_feature_dim: Dimension of global state features.
            embed_dim: Dimension of embeddings for attention computation.
            num_heads: Number of attention heads (default 1 for interpretability).

        Requires:
            - house_feature_dim > 0
            - global_feature_dim > 0
            - embed_dim > 0
            - num_heads >= 1
            - embed_dim must be divisible by num_heads

        Ensures:
            - All encoder layers are initialized
            - Attention layer is configured with correct dimensions
        """
        super().__init__()

        assert house_feature_dim > 0, f"house_feature_dim must be positive, got {house_feature_dim}"
        assert global_feature_dim > 0, f"global_feature_dim must be positive, got {global_feature_dim}"
        assert embed_dim > 0, f"embed_dim must be positive, got {embed_dim}"
        assert num_heads >= 1, f"num_heads must be at least 1, got {num_heads}"
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Encode house features to embedding space
        self.house_encoder = nn.Linear(house_feature_dim, embed_dim)

        # Encode global features to query
        self.query_encoder = nn.Linear(global_feature_dim, embed_dim)

        # Single-head attention (num_heads=1 for interpretability)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        global_features: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted context and per-candidate embeddings.

        Args:
            global_features: Global system state features.
                Shape: (batch_size, global_feature_dim)
            candidates: Candidate house features (padded to MAX_Q).
                Shape: (batch_size, max_queue_size, house_feature_dim)
            mask: Validity mask for candidates.
                Shape: (batch_size, max_queue_size)
                Values: 1.0 for valid candidates, 0.0 for padding.

        Returns:
            Tuple of:
                - house_embeddings: Per-candidate embeddings for scoring.
                    Shape: (batch_size, max_queue_size, embed_dim)
                - attention_context: Aggregated representation of relevant candidates.
                    Shape: (batch_size, embed_dim)
                - attention_weights: Attention weights over candidates (for visualization).
                    Shape: (batch_size, 1, max_queue_size)

        Requires:
            - global_features.shape == (B, global_feature_dim)
            - candidates.shape == (B, MAX_Q, house_feature_dim)
            - mask.shape == (B, MAX_Q)
            - mask values are 0.0 or 1.0

        Ensures:
            - house_embeddings.shape == (B, MAX_Q, embed_dim)
            - attention_context.shape == (B, embed_dim)
            - attention_weights.shape == (B, 1, MAX_Q)
        """
        batch_size = global_features.shape[0]
        max_queue_size = candidates.shape[1]

        # Precondition checks
        assert global_features.dim() == 2, f"Expected 2D global_features, got shape {global_features.shape}"
        assert candidates.dim() == 3, f"Expected 3D candidates, got shape {candidates.shape}"
        assert mask.dim() == 2, f"Expected 2D mask, got shape {mask.shape}"
        assert global_features.shape[0] == candidates.shape[0] == mask.shape[0], "Batch size mismatch"
        assert candidates.shape[1] == mask.shape[1], "Sequence length mismatch between candidates and mask"

        # Encode house features to Key/Value embeddings
        house_embeddings = self.house_encoder(candidates)  # (B, MAX_Q, embed_dim)

        # Encode global features to Query
        query = self.query_encoder(global_features)  # (B, embed_dim)
        query = query.unsqueeze(1)  # (B, 1, embed_dim)

        # Create attention mask: True means IGNORE this position
        # mask=1.0 means valid, so we invert: key_padding_mask = (mask == 0.0)
        key_padding_mask = (mask == 0.0)  # (B, MAX_Q), True for padding

        # Apply single-head attention
        attention_output, attention_weights = self.attention(
            query=query,
            key=house_embeddings,
            value=house_embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,  # Average over heads (only 1 head, so no effect)
        )
        # attention_output: (B, 1, embed_dim)
        # attention_weights: (B, 1, MAX_Q)

        # Squeeze query dimension
        attention_context = attention_output.squeeze(1)  # (B, embed_dim)

        # Postcondition checks
        assert house_embeddings.shape == (batch_size, max_queue_size, self.embed_dim), \
            f"Expected embeddings shape ({batch_size}, {max_queue_size}, {self.embed_dim}), got {house_embeddings.shape}"
        assert attention_context.shape == (batch_size, self.embed_dim), \
            f"Expected context shape ({batch_size}, {self.embed_dim}), got {attention_context.shape}"
        assert attention_weights.shape == (batch_size, 1, max_queue_size), \
            f"Expected weights shape ({batch_size}, 1, {max_queue_size}), got {attention_weights.shape}"

        return house_embeddings, attention_context, attention_weights


class AttentionFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Dict observation space with attention mechanism.

    This extractor takes the Dict observation from HouseGymEnv and produces:
    1. Per-candidate embeddings for scoring (stored for policy access)
    2. Aggregated context for value estimation
    3. Global features for both policy and value networks

    Attributes:
        attention_encoder: AttentionEncoder for processing candidates.
        _last_house_embeddings: Cached embeddings from last forward pass.
        _last_mask: Cached mask from last forward pass.
        _last_attention_weights: Cached attention weights for visualization.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        embed_dim: int = ATTENTION_EMBED_DIM,
    ) -> None:
        """
        Initialize the attention-based feature extractor.

        Args:
            observation_space: Dict observation space with keys:
                - 'global': Box(global_feature_dim,)
                - 'candidates': Box(max_queue_size, house_feature_dim)
                - 'mask': Box(max_queue_size,)
            embed_dim: Dimension of attention embeddings.

        Ensures:
            - features_dim == global_feature_dim + embed_dim (e.g., 6 + 64 = 70)
        """
        # Validate observation space structure
        assert isinstance(observation_space, gym.spaces.Dict), \
            f"Expected Dict observation space, got {type(observation_space)}"
        assert 'global' in observation_space.spaces, "Missing 'global' in observation space"
        assert 'candidates' in observation_space.spaces, "Missing 'candidates' in observation space"
        assert 'mask' in observation_space.spaces, "Missing 'mask' in observation space"

        global_dim = observation_space['global'].shape[0]
        house_dim = observation_space['candidates'].shape[1]
        self.max_queue_size = observation_space['candidates'].shape[0]

        # Output features = global + attention context
        features_dim = global_dim + embed_dim

        super().__init__(observation_space, features_dim=features_dim)

        self.global_dim = global_dim
        self.embed_dim = embed_dim

        # Attention encoder for policy
        self.attention_encoder = AttentionEncoder(
            house_feature_dim=house_dim,
            global_feature_dim=global_dim,
            embed_dim=embed_dim,
        )

        # Cached tensors from last forward pass (for policy/value heads)
        self._last_house_embeddings: Optional[torch.Tensor] = None
        self._last_mask: Optional[torch.Tensor] = None
        self._last_attention_weights: Optional[torch.Tensor] = None
        self._last_global_features: Optional[torch.Tensor] = None
        self._last_attention_context: Optional[torch.Tensor] = None

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from Dict observation using attention.

        Args:
            observations: Dict with keys:
                - 'global': torch.Tensor of shape (B, global_dim)
                - 'candidates': torch.Tensor of shape (B, MAX_Q, house_dim)
                - 'mask': torch.Tensor of shape (B, MAX_Q)

        Returns:
            features: Concatenated [global, attention_context].
                Shape: (B, features_dim) where features_dim = global_dim + embed_dim

        Side effects:
            - Caches house_embeddings, mask, attention_weights for policy/value heads
        """
        global_features = observations['global']
        candidates = observations['candidates']
        mask = observations['mask']

        # Precondition checks
        assert global_features.dim() == 2, f"Expected 2D global, got {global_features.shape}"
        assert candidates.dim() == 3, f"Expected 3D candidates, got {candidates.shape}"
        assert mask.dim() == 2, f"Expected 2D mask, got {mask.shape}"

        batch_size = global_features.shape[0]

        # Get embeddings and attention context
        house_embeddings, attention_context, attention_weights = self.attention_encoder(
            global_features, candidates, mask
        )

        # Cache for policy/value heads access
        self._last_house_embeddings = house_embeddings
        self._last_mask = mask
        self._last_attention_weights = attention_weights.detach()
        self._last_global_features = global_features
        self._last_attention_context = attention_context

        # Concatenate global features and attention context
        features = torch.cat([global_features, attention_context], dim=-1)

        # Postcondition check
        assert features.shape == (batch_size, self._features_dim), \
            f"Expected features shape ({batch_size}, {self._features_dim}), got {features.shape}"

        return features

    def get_last_house_embeddings(self) -> Optional[torch.Tensor]:
        """Get cached house embeddings from last forward pass. Shape: (B, MAX_Q, embed_dim)"""
        return self._last_house_embeddings

    def get_last_mask(self) -> Optional[torch.Tensor]:
        """Get cached mask from last forward pass. Shape: (B, MAX_Q)"""
        return self._last_mask

    def get_last_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights from last forward pass. Shape: (B, 1, MAX_Q)"""
        return self._last_attention_weights

    def get_last_global_features(self) -> Optional[torch.Tensor]:
        """Get cached global features from last forward pass. Shape: (B, global_dim)"""
        return self._last_global_features

    def get_last_attention_context(self) -> Optional[torch.Tensor]:
        """Get cached attention context from last forward pass. Shape: (B, embed_dim)"""
        return self._last_attention_context


class ScoreHead(nn.Module):
    """
    MLP head that outputs a scalar score for each candidate embedding.

    Applies the same MLP to each candidate independently to produce per-candidate
    scores for allocation. Uses default PyTorch initialization (Kaiming) for
    meaningful gradient signal.
    """

    def __init__(self, input_dim: int = ATTENTION_EMBED_DIM * 2, hidden_dim: int = 64) -> None:
        """
        Initialize the score head.

        Args:
            input_dim: Dimension of input (embed_dim * 2 for concatenated context).
            hidden_dim: Hidden layer dimension (default 64).
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Use default PyTorch initialization (Kaiming) for meaningful gradients

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute per-candidate scores.

        Args:
            embeddings: Candidate embeddings. Shape: (B, MAX_Q, embed_dim)
            mask: Validity mask. Shape: (B, MAX_Q), 1.0=valid, 0.0=padding

        Returns:
            scores: Per-candidate scores. Shape: (B, MAX_Q)
                    Invalid positions are set to -1e9 for softmax masking.
        """
        # Apply MLP to each candidate: (B, MAX_Q, embed_dim) -> (B, MAX_Q, 1)
        scores = self.mlp(embeddings).squeeze(-1)  # (B, MAX_Q)

        # Mask invalid positions with large negative value
        scores = scores.masked_fill(mask == 0.0, -1e9)

        return scores


class AttentionActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-Critic policy with per-candidate scoring for allocation.

    This policy outputs a score for each candidate, enabling direct control
    over resource allocation via softmax distribution.

    Architecture:
        - Feature extractor: AttentionFeaturesExtractor (produces embeddings + context)
        - Score head: Per-candidate embedding -> scalar score (B, MAX_Q)
        - Value head: Pooled embeddings + global features -> value (B, 1)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Box,
        lr_schedule,
        **kwargs,
    ) -> None:
        """
        Initialize the attention-based actor-critic policy.

        Args:
            observation_space: Dict observation space from environment.
            action_space: Box action space of shape (MAX_QUEUE_SIZE,).
            lr_schedule: Learning rate schedule function.
            **kwargs: Additional arguments passed to parent class.
        """
        # Set custom features extractor
        kwargs['features_extractor_class'] = AttentionFeaturesExtractor
        kwargs['features_extractor_kwargs'] = {'embed_dim': ATTENTION_EMBED_DIM}

        # Use empty net_arch - we'll build custom heads
        kwargs['net_arch'] = []

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **kwargs,
        )

        # Get dimensions from feature extractor
        self.max_queue_size = observation_space['candidates'].shape[0]
        self.embed_dim = ATTENTION_EMBED_DIM
        self.global_dim = observation_space['global'].shape[0]

        # Build custom score head for policy (per-candidate scores)
        # Input: house_embeddings (embed_dim) + attention_context (embed_dim) = embed_dim * 2
        self.score_head = ScoreHead(input_dim=self.embed_dim * 2)

        # Learnable log_std for action distribution (one per action dimension)
        # Initialize to 0.0 (std=1.0) for sufficient initial exploration
        self.log_std = nn.Parameter(torch.zeros(self.max_queue_size))

        # Build custom value head: pooled embeddings + global -> value
        # Input: global_dim + embed_dim (from mean pooling)
        value_input_dim = self.global_dim + self.embed_dim
        self.custom_value_net = nn.Sequential(
            nn.Linear(value_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # CRITICAL FIX: Rebuild optimizer to include custom modules
        # The parent class creates the optimizer in __init__(), but our custom
        # modules (score_head, log_std, custom_value_net) are defined after
        # super().__init__(), so they're not in the optimizer's param_groups.
        # We must rebuild the optimizer to include all parameters.
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute actions, values, and log probabilities.

        Uses ScoreHead output as action mean, with learnable log_std for exploration.
        Actions are squashed to [-1, 1] via tanh.

        Args:
            obs: Dict observation with 'global', 'candidates', 'mask' keys.
            deterministic: If True, use mean action instead of sampling.

        Returns:
            Tuple of (actions, values, log_probs).
        """
        # Extract features (this caches embeddings and mask)
        _ = self.extract_features(obs)

        # Get cached embeddings and mask from feature extractor
        house_embeddings = self.features_extractor.get_last_house_embeddings()
        mask = self.features_extractor.get_last_mask()
        global_features = self.features_extractor.get_last_global_features()
        attention_context = self.features_extractor.get_last_attention_context()

        # Concatenate attention_context with each candidate embedding
        # attention_context: (B, embed_dim) -> (B, MAX_Q, embed_dim)
        ctx_expanded = attention_context.unsqueeze(1).expand(-1, self.max_queue_size, -1)
        combined = torch.cat([house_embeddings, ctx_expanded], dim=-1)  # (B, MAX_Q, embed_dim*2)

        # Compute per-candidate scores as action mean (B, MAX_Q)
        # ScoreHead applies mask internally (invalid positions get -1e9)
        action_mean = self.score_head(combined, mask)

        # Compute value using pooled embeddings
        values = self._compute_value(house_embeddings, mask, global_features)

        # Build Gaussian distribution with ScoreHead output as mean
        std = torch.exp(self.log_std).expand_as(action_mean)
        distribution = torch.distributions.Normal(action_mean, std)

        if deterministic:
            # Use mean directly
            actions = action_mean
        else:
            # Sample from distribution
            actions = distribution.rsample()

        # Clamp to action space bounds (no tanh to avoid saturation)
        actions = torch.clamp(actions, -5.0, 5.0)

        # Compute log_prob only for VALID positions (mask out invalid)
        # This prevents masked positions from dominating the gradient
        log_prob = distribution.log_prob(actions)
        log_prob = (log_prob * mask).sum(dim=-1)

        return actions, values, log_prob

    def _compute_value(
        self,
        house_embeddings: torch.Tensor,
        mask: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute state value using pooled candidate embeddings.

        Args:
            house_embeddings: Per-candidate embeddings. Shape: (B, MAX_Q, embed_dim)
            mask: Validity mask. Shape: (B, MAX_Q)
            global_features: Global state features. Shape: (B, global_dim)

        Returns:
            values: State values. Shape: (B, 1)
        """
        # Masked mean pooling over valid candidates
        mask_expanded = mask.unsqueeze(-1)  # (B, MAX_Q, 1)
        masked_embeddings = house_embeddings * mask_expanded  # Zero out invalid

        # Sum and divide by count of valid candidates
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        pooled = masked_embeddings.sum(dim=1) / valid_counts  # (B, embed_dim)

        # Concatenate with global features
        value_input = torch.cat([global_features, pooled], dim=-1)  # (B, global_dim + embed_dim)

        # Compute value
        values = self.custom_value_net(value_input)  # (B, 1)

        return values

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions for PPO update.

        Args:
            obs: Dict observation.
            actions: Actions to evaluate (already squashed). Shape: (B, MAX_Q)

        Returns:
            Tuple of (values, log_probs, entropy).
        """
        # Extract features (this caches embeddings and mask)
        _ = self.extract_features(obs)

        # Get cached data
        house_embeddings = self.features_extractor.get_last_house_embeddings()
        mask = self.features_extractor.get_last_mask()
        global_features = self.features_extractor.get_last_global_features()
        attention_context = self.features_extractor.get_last_attention_context()

        # Compute value
        values = self._compute_value(house_embeddings, mask, global_features)

        # Concatenate attention_context with each candidate embedding
        ctx_expanded = attention_context.unsqueeze(1).expand(-1, self.max_queue_size, -1)
        combined = torch.cat([house_embeddings, ctx_expanded], dim=-1)  # (B, MAX_Q, embed_dim*2)

        # Compute action mean from ScoreHead
        action_mean = self.score_head(combined, mask)

        # Build Gaussian distribution
        std = torch.exp(self.log_std).expand_as(action_mean)
        distribution = torch.distributions.Normal(action_mean, std)

        # Compute log_prob only for VALID positions (no tanh correction needed)
        log_prob = distribution.log_prob(actions)
        log_prob = (log_prob * mask).sum(dim=-1)

        # Compute entropy as MEAN over valid positions only
        # This normalizes entropy by number of valid candidates, avoiding 1024-dim dominance
        valid_counts = mask.sum(dim=-1).clamp(min=1.0)
        entropy = (distribution.entropy() * mask).sum(dim=-1) / valid_counts

        return values, log_prob, entropy

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict state values for given observations.

        Args:
            obs: Dict observation.

        Returns:
            values: State values. Shape: (B, 1)
        """
        # Extract features (this caches embeddings and mask)
        _ = self.extract_features(obs)

        # Get cached data
        house_embeddings = self.features_extractor.get_last_house_embeddings()
        mask = self.features_extractor.get_last_mask()
        global_features = self.features_extractor.get_last_global_features()

        return self._compute_value(house_embeddings, mask, global_features)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights from the feature extractor."""
        if hasattr(self.features_extractor, 'get_last_attention_weights'):
            return self.features_extractor.get_last_attention_weights()
        return None
