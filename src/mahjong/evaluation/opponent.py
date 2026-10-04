"""Opponent abstraction for self-play / evaluation (Phase 8).

Every opponent exposes the same ``decide(observation, legal_actions, rng)``
interface.  ``legal_actions`` is the environment's full legal action list for
the seat (``Action`` objects plus, rarely, the special ``kyuushu_kyuuhai``
string); an opponent must return one of those elements.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..decision.action import Action, ActionType
from ..features import ACTION_SPACE_SIZE, FEATURE_DIM
from ..features.action_space import action_to_id, legal_mask
from ..features.encoder import ObservationEncoder
from ..model.policy import MLPPolicy, remap_bc_state_dict
from ..rules.tiles import HONORS, normalize
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


def _is_ppo_checkpoint_file(path: Path) -> bool:
    """True if ``path`` is a PPO training checkpoint (``checkpoint_type == "ppo"``)."""
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return False
    return isinstance(payload, dict) and (
        payload.get("checkpoint_type") == "ppo" or "model_state_dict" in payload
    )


def load_policy(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    checkpoint_type: str | None = None,
) -> tuple[ObservationEncoder, MLPPolicy]:
    """Load an encoder + policy from a BC or PPO checkpoint.

    - ``checkpoint_type="ppo"``（或文件为 PPO checkpoint）→ ``model_state_dict`` 直接构造
      ``MLPPolicy``（含 value head；``strict=False`` 兼容缺 value head 的旧权重）。
    - 否则走既有 BC 路径（``load_checkpoint`` + ``remap_bc_state_dict``，保持不变）。
    """
    path = Path(checkpoint)
    encoder = ObservationEncoder()

    if checkpoint_type == "ppo" or (checkpoint_type is None and _is_ppo_checkpoint_file(path)):
        from ..training.ppo_checkpoint import load_ppo_checkpoint

        ckpt = load_ppo_checkpoint(path, map_location="cpu")
        state = ckpt.get("model_state_dict") or {}
        hidden = tuple((ckpt.get("config") or {}).get("model", {}).get("hidden_sizes", [128, 128]))
        model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, hidden)
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        return encoder, model

    # BC 路径（保持不变）
    checkpoint_dir = path.parent if path.is_file() else path
    ckpt = load_checkpoint(checkpoint_dir)
    hidden = tuple(ckpt.get("config", {}).get("model", {}).get("hidden_sizes", [128, 128]))
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
    """Policy opponent with configurable decision mode (§17).

    ``mode="sampling"`` (default): ``torch.multinomial`` (stochastic, rng-seeded).
    ``mode="greedy"``: ``argmax`` over legal logits (deterministic, ignores rng).
    """

    def __init__(
        self,
        *,
        opponent_id: str,
        type: str,
        model_version: str,
        checkpoint: str | Path,
        config: dict | None = None,
        device: str | torch.device = "cpu",
        mode: str = "sampling",
        checkpoint_type: str | None = None,
    ):
        super().__init__(
            opponent_id=opponent_id,
            type=type,
            model_version=model_version,
            checkpoint=str(checkpoint),
            config=config or {},
        )
        if mode not in ("greedy", "sampling"):
            raise ValueError(f"mode must be greedy|sampling, got {mode!r}")
        self.device = torch.device(device)
        self.mode = mode
        self.encoder, self.model = load_policy(checkpoint, self.device, checkpoint_type=checkpoint_type)

    def decide(self, observation, legal_actions: list, rng: random.Random):
        actions = [a for a in legal_actions if isinstance(a, Action)]
        if not actions:
            return legal_actions[0]  # e.g. the special kyuushu_kyuuhai string
        features = self.encoder.encode(observation).to(self.device).unsqueeze(0)
        mask = legal_mask(actions).to(self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model(features, mask)
            if self.mode == "greedy":
                action_id = int(out.logits.argmax(dim=-1).item())
            else:
                probs = torch.softmax(out.logits, dim=-1)
                generator = torch.Generator(device="cpu")
                generator.manual_seed(rng.randrange(1 << 31))
                action_id = int(torch.multinomial(probs, 1, generator=generator).squeeze(-1).item())
        return next(a for a in actions if action_to_id(a) == action_id)


class RuleOpponent(Opponent):
    """Simple explainable heuristic baseline (§18).

    Decision rules (in priority order):
      1. TSUMO / RON → take the win.
      2. RIICHI → declare (riichi is only legal when tenpai).
      3. PASS → pass (never call pon/chi/daiminkan).
      4. KAN ankan → take (closed kan); other KAN → pass.
      5. DISCARD → discard the least useful tile (isolated terminal/honour first).
    """

    def __init__(self, opponent_id: str = "rule", model_version: str = "rule-v1"):
        super().__init__(opponent_id=opponent_id, type="rule", model_version=model_version)

    @staticmethod
    def _keep_value(tile: str, hand_counts: Counter) -> float:
        t = normalize(tile)
        v = 0.0
        if hand_counts[t] >= 2:
            v += 2.0  # keep pairs/triplets
        if t in HONORS:
            v += 0.2  # honours mostly useless unless yakuhai
        elif t[0] in "19":
            v += 0.5  # terminals
        else:
            v += 1.0  # middle tiles form sequences
        return v

    def decide(self, observation, legal_actions: list, rng: random.Random):
        actions = [a for a in legal_actions if isinstance(a, Action)]
        if not actions:
            return legal_actions[0]

        for a in actions:
            if a.type in (ActionType.TSUMO, ActionType.RON):
                return a
        for a in actions:
            if a.type is ActionType.RIICHI:
                return a
        for a in actions:
            if a.type is ActionType.PASS:
                return a
        for a in actions:
            if a.type is ActionType.KAN and a.kan_kind == "ankan":
                return a

        discards = [a for a in actions if a.type is ActionType.DISCARD]
        if discards:
            hand_counts = Counter(normalize(t) for t in observation.hand)
            return min(discards, key=lambda a: self._keep_value(a.tile, hand_counts))
        return actions[0]


__all__ = ["Opponent", "PolicyOpponent", "RandomOpponent", "RuleOpponent", "load_policy"]
