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

import torch
import torch.nn as nn

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

    def __init__(self, feature_dim: int, action_size: int, hidden_sizes=(256, 256)) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.action_size = action_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.trunk = _build_trunk(feature_dim, hidden_sizes)
        last = hidden_sizes[-1] if hidden_sizes else feature_dim
        self.action_head = nn.Linear(last, action_size)
        self.value_head = nn.Linear(last, 1)

    def _heads(self, features: torch.Tensor):
        h = self.trunk(features)
        logits = self.action_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def forward(self, features: torch.Tensor, legal_mask: torch.Tensor) -> PolicyOutput:
        logits, value = self._heads(features)
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
