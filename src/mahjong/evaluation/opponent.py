"""Opponent abstraction for self-play / evaluation (Phase 8).

Every opponent exposes the same ``decide(observation, legal_actions, rng)``
interface.  ``legal_actions`` is the environment's full legal action list for
the seat (``Action`` objects plus, rarely, the special ``kyuushu_kyuuhai``
string); an opponent must return one of those elements.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..decision.action import Action
from ..features import ACTION_SPACE_SIZE, FEATURE_DIM
from ..features.action_space import action_to_id, legal_mask
from ..features.encoder import ObservationEncoder
from ..model.policy import MLPPolicy, remap_bc_state_dict
from ..training.checkpoint import load_checkpoint


@dataclass
class Opponent:
    """Abstract opponent with version metadata (TRAINING_SPEC 10)."""

    opponent_id: str
    type: str  # "random" | "bc" | "policy"
    model_version: str
    checkpoint: str | None = None
    config: dict = field(default_factory=dict)

    def decide(self, observation, legal_actions: list, rng: random.Random):
        raise NotImplementedError


def load_policy(checkpoint: str | Path, device: str | torch.device = "cpu") -> tuple[ObservationEncoder, MLPPolicy]:
    """Load an encoder + policy (with value head) from a Phase 5/7 checkpoint."""
    path = Path(checkpoint)
    checkpoint_dir = path.parent if path.is_file() else path
    ckpt = load_checkpoint(checkpoint_dir)
    hidden = tuple(ckpt.get("config", {}).get("model", {}).get("hidden_sizes", [128, 128]))
    encoder = ObservationEncoder()
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, hidden)
    model.load_state_dict(remap_bc_state_dict(ckpt["state_dict"]), strict=False)
    model.to(device)
    model.eval()
    return encoder, model


class RandomOpponent(Opponent):
    def __init__(self, opponent_id: str = "random", model_version: str = "random-v1"):
        super().__init__(opponent_id=opponent_id, type="random", model_version=model_version)

    def decide(self, observation, legal_actions: list, rng: random.Random):
        return rng.choice(legal_actions)


class PolicyOpponent(Opponent):
    """Samples from a policy (with legal mask); never samples an illegal action."""

    def __init__(
        self,
        *,
        opponent_id: str,
        type: str,
        model_version: str,
        checkpoint: str | Path,
        config: dict | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__(
            opponent_id=opponent_id,
            type=type,
            model_version=model_version,
            checkpoint=str(checkpoint),
            config=config or {},
        )
        self.device = torch.device(device)
        self.encoder, self.model = load_policy(checkpoint, self.device)

    def decide(self, observation, legal_actions: list, rng: random.Random):
        actions = [a for a in legal_actions if isinstance(a, Action)]
        if not actions:
            return legal_actions[0]  # e.g. the special kyuushu_kyuuhai string
        features = self.encoder.encode(observation).to(self.device).unsqueeze(0)
        mask = legal_mask(actions).to(self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model(features, mask)
            probs = torch.softmax(out.logits, dim=-1)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(rng.randrange(1 << 31))
            action_id = int(torch.multinomial(probs, 1, generator=generator).squeeze(-1).item())
        return next(a for a in actions if action_to_id(a) == action_id)


__all__ = ["Opponent", "PolicyOpponent", "RandomOpponent", "load_policy"]
