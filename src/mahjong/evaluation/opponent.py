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
from ..features.encoder_v2 import FEATURE_DIM_V2, ObservationEncoderV2
from ..features.encoder_v3 import FEATURE_DIM_V3, ObservationEncoderV3
from ..features.action_space_v2 import ACTION_SPACE_SIZE_V2, action_to_id_v2, legal_mask_v2
from ..model.policy import MLPPolicy, build_policy_model, remap_bc_state_dict
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
    if checkpoint_type == "ppo" or (checkpoint_type is None and _is_ppo_checkpoint_file(path)):
        from ..training.ppo_checkpoint import load_ppo_checkpoint

        ckpt = load_ppo_checkpoint(path, map_location="cpu")
        state = ckpt.get("model_state_dict") or {}
        hidden = tuple((ckpt.get("config") or {}).get("model", {}).get("hidden_sizes", [128, 128]))
        config = ckpt.get("config") or {}
        encoder, feat_dim, action_size = _schema_components(config)
        model = build_policy_model(
            config.get("feature_version", "feature-v1"), feat_dim, action_size, hidden,
            model_config=config.get("model") or {},
            value_separate_trunk=bool(config.get("model", {}).get("value_separate_trunk", False)),
        )
        model.load_state_dict(state, strict=config.get("feature_version") == "feature-v3")
        model.to(device)
        model.eval()
        model.action_schema_version = config.get("action_schema_version", "action-v1")
        model.allow_kyuushu = bool(config.get("allow_kyuushu", False))
        return encoder, model

    # BC 路径：a file path is authoritative; a directory resolves best.pt
    # first, then the legacy model.pt format.  Previously a supplied best.pt
    # path was ignored and its parent/model.pt was loaded instead.
    if path.is_file():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    else:
        best_file = path / "best.pt"
        model_file = path / "model.pt"
        if best_file.is_file():
            ckpt = torch.load(best_file, map_location="cpu", weights_only=False)
        elif model_file.is_file():
            ckpt = load_checkpoint(path)
        else:
            raise FileNotFoundError(f"no BC checkpoint found at {path} (expected best.pt or model.pt)")
    config = ckpt.get("config") or {}
    hidden = tuple(config.get("model", {}).get("hidden_sizes", [128, 128]))
    encoder, feat_dim, action_size = _schema_components(config)
    model = build_policy_model(
        config.get("feature_version", "feature-v1"), feat_dim, action_size, hidden,
        model_config=config.get("model") or {},
    )
    state = ckpt.get("state_dict")
    if state is None:
        raise ValueError(f"BC checkpoint has no state_dict: {path}")
    model.load_state_dict(
        remap_bc_state_dict(state), strict=config.get("feature_version") == "feature-v3"
    )
    model.to(device)
    model.eval()
    model.action_schema_version = config.get("action_schema_version", "action-v1")
    model.allow_kyuushu = bool(config.get("allow_kyuushu", False))
    return encoder, model


def _schema_components(config: dict):
    feature_version = config.get("feature_version", "feature-v1")
    action_version = config.get("action_schema_version", "action-v1")
    if feature_version == "feature-v2" and action_version == "action-v2":
        return ObservationEncoderV2(), FEATURE_DIM_V2, ACTION_SPACE_SIZE_V2
    if feature_version == "feature-v3" and action_version == "action-v2":
        return ObservationEncoderV3(), FEATURE_DIM_V3, ACTION_SPACE_SIZE_V2
    if feature_version == "feature-v1" and action_version == "action-v1":
        return ObservationEncoder(), FEATURE_DIM, ACTION_SPACE_SIZE
    raise ValueError(f"unsupported checkpoint schema: {feature_version}/{action_version}")


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
        temperature: float = 1.0,
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
        self.temperature = temperature
        self.encoder, self.model = load_policy(checkpoint, self.device, checkpoint_type=checkpoint_type)

    def decide(self, observation, legal_actions: list, rng: random.Random):
        actions = [
            a for a in legal_actions
            if isinstance(a, Action)
            and (a.type is not ActionType.KYUUSHU_KYUHAI or self.model.allow_kyuushu)
        ]
        if not actions:
            return legal_actions[0]
        features = self.encoder.encode(observation).to(self.device).unsqueeze(0)
        if self.model.action_schema_version == "action-v2":
            mask = legal_mask_v2(actions, allow_kyuushu=self.model.allow_kyuushu).to(self.device).unsqueeze(0)
            to_id = action_to_id_v2
        else:
            mask = legal_mask(actions).to(self.device).unsqueeze(0)
            to_id = action_to_id
        with torch.no_grad():
            out = self.model(features, mask)
            if self.mode == "greedy":
                action_id = int(out.logits.argmax(dim=-1).item())
            else:
                # 温度采样：logits/T 后再 masked softmax；T=1.0 与现有行为逐位一致
                probs = torch.softmax(out.logits / self.temperature, dim=-1)
                generator = torch.Generator(device=probs.device)
                generator.manual_seed(rng.randrange(1 << 31))
                action_id = int(torch.multinomial(probs, 1, generator=generator).squeeze(-1).item())
        return next(a for a in actions if to_id(a) == action_id)


class RuleOpponent(Opponent):
    """Simple explainable heuristic baseline (§18).

    Decision rules (in priority order):
      1. TSUMO / RON → take the win.
      2. RIICHI → declare (riichi is only legal when tenpai).
      3. PASS → pass (never call pon/chi/daiminkan).
      4. KAN ankan → take (closed kan); other KAN → pass.
      5. DISCARD → discard the least useful tile (isolated terminal/honour first).
    """

    def __init__(self, opponent_id: str = "rule", model_version: str = "rule-v2-win-first"):
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

        if self.model_version != "rule-v1":
            for a in actions:
                if a.type in (ActionType.TSUMO, ActionType.RON):
                    return a
        # Optional abort only after accepting any legal win.
        for a in actions:
            if a.type is ActionType.KYUUSHU_KYUHAI:
                return a

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
