"""MLP policy with a value head (Phase 5 BC + Phase 7 PPO).

The model is a shared trunk + an action head (logits over the action space) +
a value head (scalar).  It implements no mahjong rules (MODEL_SPEC 2); legality
is enforced purely by the legal-action mask.

Phase-5 BC checkpoints (which stored ``mlp.*`` keys without a value head) can be
loaded via :func:`remap_bc_state_dict` / ``load_state_dict(strict=False)``; the
value head is then randomly initialised.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from ..features.action_space import ACTION_SPACE_SIZE, legal_mask
from ..features.encoder import ObservationEncoder


@dataclass
class PolicyOutput:
    """Legal-masked logits and a scalar value estimate."""

    logits: torch.Tensor
    value: torch.Tensor | None = None


def _build_trunk(feature_dim: int, hidden_sizes) -> nn.Sequential:
    layers = []
    prev = feature_dim
    for hidden in hidden_sizes:
        layers.append(nn.Linear(prev, hidden))
        layers.append(nn.ReLU())
        prev = hidden
    return nn.Sequential(*layers)


def remap_bc_state_dict(state_dict: dict) -> dict:
    """Map Phase-5 BC keys (``mlp.{i}.*``) to the trunk/action-head layout."""
    mlp_keys = [k for k in state_dict if k.startswith("mlp.")]
    if not mlp_keys:
        return dict(state_dict)
    max_idx = max(int(k.split(".")[1]) for k in mlp_keys)
    new = {}
    for k, v in state_dict.items():
        if k.startswith("mlp."):
            idx = int(k.split(".")[1])
            rest = ".".join(k.split(".")[2:])
            new[f"action_head.{rest}" if idx == max_idx else f"trunk.{idx}.{rest}"] = v
        else:
            new[k] = v
    return new


class MLPPolicy(nn.Module):
    """Batched core: features ``[B, F]`` + mask ``[B, A]`` -> (logits, value)."""

    def __init__(self, feature_dim: int, action_size: int, hidden_sizes=(256, 256), value_separate_trunk: bool = False,
                 policy_temperature: float = 1.0) -> None:
        super().__init__()
        if policy_temperature <= 0:
            raise ValueError('policy_temperature must be positive')
        self.feature_dim = feature_dim
        self.action_size = action_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.value_separate_trunk = value_separate_trunk
        self.policy_temperature = float(policy_temperature)
        self.trunk = _build_trunk(feature_dim, hidden_sizes)
        last = hidden_sizes[-1] if hidden_sizes else feature_dim
        self.action_head = nn.Linear(last, action_size)
        self.value_head = nn.Linear(last, 1)
        # H1：value 使用独立 trunk（默认 False=共享，行为逐位不变）
        if value_separate_trunk:
            self.value_trunk = _build_trunk(feature_dim, hidden_sizes)

    def _heads(self, features: torch.Tensor):
        h = self.trunk(features)
        logits = self.action_head(h)
        if getattr(self, "value_separate_trunk", False):
            value = self.value_head(self.value_trunk(features)).squeeze(-1)
        else:
            value = self.value_head(h).squeeze(-1)
        return logits, value

    def forward(self, features: torch.Tensor, legal_mask: torch.Tensor) -> PolicyOutput:
        logits, value = self._heads(features)
        if self.policy_temperature != 1.0:
            logits = logits / self.policy_temperature
        masked = logits.masked_fill(~legal_mask, float("-inf"))
        return PolicyOutput(logits=masked, value=value)

    def action_probs(self, features: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(features, legal_mask).logits, dim=-1)

    def log_probs(self, features: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.forward(features, legal_mask).logits, dim=-1)

    def sample(self, features: torch.Tensor, legal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one legal action id per row; return (action_ids, log_probs)."""
        probs = self.action_probs(features, legal_mask)
        action_ids = torch.multinomial(probs, 1).squeeze(-1)
        logp = torch.log(probs.gather(-1, action_ids.unsqueeze(-1)).squeeze(-1) + 1e-12)
        return action_ids, logp

    def evaluate_actions(
        self, features: torch.Tensor, legal_mask: torch.Tensor, action_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (log_probs, values, entropy) for the given action ids."""
        out = self.forward(features, legal_mask)
        logp_all = torch.log_softmax(out.logits, dim=-1)
        log_probs = logp_all.gather(-1, action_ids.unsqueeze(-1)).squeeze(-1)
        values = out.value
        probs = torch.softmax(out.logits, dim=-1)
        # entropy over the legal distribution only; masked (-inf) log-probs give
        # 0 * -inf = NaN, so zero them out (their probability is already 0).
        safe_logp = torch.where(torch.isfinite(logp_all), logp_all, torch.zeros_like(logp_all))
        entropy = -(probs * safe_logp).sum(-1)
        return log_probs, values, entropy


class RiverSequencePolicy(MLPPolicy):
    """MLP policy with a shared tile-embedding/GRU encoder for three rivers.

    Input remains one flat tensor for rollout compatibility: a 397-value dense
    prefix followed by three padded 128-token rivers. Token IDs are embedded
    and each relative opponent river is independently encoded by the same GRU.
    The three summaries are concatenated in relative-seat order before the MLP.
    """

    def __init__(
        self,
        feature_dim: int,
        action_size: int,
        hidden_sizes=(256, 256),
        *,
        tile_embedding_dim: int = 16,
        river_hidden_size: int = 64,
        river_num_layers: int = 1,
        value_separate_trunk: bool = False,
        value_separate_encoder: bool = False,
        value_public_coefficients: list[float] | None = None,
        policy_temperature: float = 1.0,
    ) -> None:
        from ..features.encoder_v3 import (
            DENSE_FEATURE_DIM_V3,
            FEATURE_DIM_V3,
            MAX_OPPONENT_RIVER,
            RIVER_COUNT_V3,
            RIVER_PAD_ID_V3,
            RIVER_VOCAB_SIZE_V3,
        )

        if feature_dim != FEATURE_DIM_V3:
            raise ValueError(f"feature-v3 model needs {FEATURE_DIM_V3} inputs, got {feature_dim}")
        if tile_embedding_dim <= 0 or river_hidden_size <= 0 or river_num_layers <= 0:
            raise ValueError("river encoder dimensions must be positive")
        if value_separate_encoder and not value_separate_trunk:
            raise ValueError("value_separate_encoder requires value_separate_trunk")
        super().__init__(
            DENSE_FEATURE_DIM_V3 + RIVER_COUNT_V3 * river_hidden_size,
            action_size,
            hidden_sizes,
            value_separate_trunk=value_separate_trunk,
            policy_temperature=policy_temperature,
        )
        # Public feature_dim describes the serialized encoder output; the MLP
        # trunk consumes the dense prefix plus the three learned summaries.
        self.dense_feature_dim = DENSE_FEATURE_DIM_V3
        self.river_count = RIVER_COUNT_V3
        self.river_length = MAX_OPPONENT_RIVER
        self.river_pad_id = RIVER_PAD_ID_V3
        self.tile_embedding_dim = tile_embedding_dim
        self.river_hidden_size = river_hidden_size
        self.river_num_layers = river_num_layers
        self.feature_dim = FEATURE_DIM_V3
        self.tile_embedding = nn.Embedding(
            RIVER_VOCAB_SIZE_V3, tile_embedding_dim, padding_idx=RIVER_PAD_ID_V3
        )
        self.river_encoder = nn.GRU(
            tile_embedding_dim, river_hidden_size, num_layers=river_num_layers,
            batch_first=True,
        )
        self.value_separate_encoder = value_separate_encoder
        self.value_public_coefficients = value_public_coefficients
        if value_public_coefficients is not None:
            if len(value_public_coefficients) != 17 or value_separate_trunk or value_separate_encoder:
                raise ValueError('public value requires 17 coefficients and no separate neural critic')
            self.value_public = nn.Linear(16, 1)
            with torch.no_grad():
                self.value_public.bias.fill_(float(value_public_coefficients[0]))
                self.value_public.weight.copy_(torch.tensor(value_public_coefficients[1:], dtype=torch.float32).reshape(1, 16))
        if value_separate_encoder:
            self.value_tile_embedding = deepcopy(self.tile_embedding)
            self.value_river_encoder = deepcopy(self.river_encoder)

    def _encode_rivers(self, features, embedding, encoder):
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"feature-v3 expected [B,{self.feature_dim}], got {tuple(features.shape)}"
            )
        dense = features[:, :self.dense_feature_dim]
        tokens = features[:, self.dense_feature_dim:].reshape(
            -1, self.river_count, self.river_length
        ).long()
        flat_tokens = tokens.reshape(-1, self.river_length)
        lengths = flat_tokens.ne(self.river_pad_id).sum(dim=1)
        packed_lengths = lengths.clamp_min(1).to(device="cpu")
        embedded = embedding(flat_tokens)
        packed = pack_padded_sequence(
            embedded, packed_lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = encoder(packed)
        summaries = hidden[-1]
        summaries = summaries * lengths.gt(0).to(summaries.dtype).unsqueeze(-1)
        summaries = summaries.reshape(features.shape[0], self.river_count * self.river_hidden_size)
        return torch.cat((dense, summaries), dim=-1)

    def _heads(self, features: torch.Tensor):
        actor_features = self._encode_rivers(features, self.tile_embedding, self.river_encoder)
        if self.value_public_coefficients is not None:
            seat = features[:, 385:389]
            scores = features[:, 389:393]
            own = (seat * scores).sum(-1, keepdim=True)
            others = (scores.sum(-1, keepdim=True) - own) / 3
            public = torch.cat(((own-25)/10, (own-others)/10,
                                features[:, 367:375], features[:, 375:377]/10, seat), dim=-1)
            return self.action_head(self.trunk(actor_features)), self.value_public(public).squeeze(-1)
        if not self.value_separate_encoder:
            return super()._heads(actor_features)
        critic_features = self._encode_rivers(
            features, self.value_tile_embedding, self.value_river_encoder
        )
        logits = self.action_head(self.trunk(actor_features))
        value = self.value_head(self.value_trunk(critic_features)).squeeze(-1)
        return logits, value


def build_policy_model(
    feature_version: str,
    feature_dim: int,
    action_size: int,
    hidden_sizes=(256, 256),
    *,
    model_config: dict | None = None,
    value_separate_trunk: bool = False,
) -> MLPPolicy:
    """Construct the policy architecture named by a versioned model config."""
    config = model_config or {}
    if feature_version == "feature-v3":
        return RiverSequencePolicy(
            feature_dim,
            action_size,
            hidden_sizes,
            tile_embedding_dim=int(config.get("tile_embedding_dim", 16)),
            river_hidden_size=int(config.get("river_hidden_size", 64)),
            river_num_layers=int(config.get("river_num_layers", 1)),
            value_separate_trunk=value_separate_trunk,
            value_separate_encoder=bool(config.get("value_separate_encoder", False)),
            value_public_coefficients=config.get("value_public_coefficients"),
            policy_temperature=config.get("policy_temperature", 1.0),
        )
    return MLPPolicy(
        feature_dim, action_size, hidden_sizes,
        value_separate_trunk=value_separate_trunk,
    )


class Policy(nn.Module):
    """Observation-level interface: encode + mask + (logits, value)."""

    def __init__(
        self,
        encoder: ObservationEncoder,
        feature_dim: int,
        action_size: int = ACTION_SPACE_SIZE,
        hidden_sizes=(256, 256),
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.mlp = MLPPolicy(feature_dim, action_size, hidden_sizes)

    def forward(self, observation, legal_actions) -> PolicyOutput:
        """Encode a single observation + mask -> PolicyOutput(logits, value)."""
        features = self.encoder.encode(observation)
        mask = legal_mask(legal_actions).unsqueeze(0)
        return self.mlp(features.unsqueeze(0), mask)

    def sample_action(self, observation, legal_actions) -> tuple[int, float]:
        """Sample a legal action id and return its log probability."""
        features = self.encoder.encode(observation)
        mask = legal_mask(legal_actions).unsqueeze(0)
        action_id, logp = self.mlp.sample(features.unsqueeze(0), mask)
        return int(action_id.item()), float(logp.item())

    def evaluate_actions(self, observation, legal_actions, action_ids: torch.Tensor):
        """Return scalar (log_prob, value, entropy) for the given action id (single obs)."""
        features = self.encoder.encode(observation)
        mask = legal_mask(legal_actions).unsqueeze(0)
        log_probs, values, entropy = self.mlp.evaluate_actions(features.unsqueeze(0), mask, action_ids)
        return log_probs.squeeze(0), values.squeeze(0), entropy.squeeze(0)
