"""Training opponent pool with config-driven proportional sampling (Phase 10.3 §12/§13).

Opponent probabilities come **from config** (never hard-coded); sampling is
seeded (same seed ⇒ same sequence).  Historical checkpoint registration is a
thin helper that scans a directory for milestone checkpoints.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch


class TrainingOpponentPool:
    """Weighted opponent sampler keyed by opponent id.

    ``opponents``: ``{id: checkpoint_or_factory}`` (checkpoint path or any label).
    ``probabilities``: ``{id: float}`` (must come from config).
    """

    def __init__(self, opponents: dict[str, object], probabilities: dict[str, float], *, seed: int):
        ids = list(probabilities)
        if not ids:
            raise ValueError("opponent pool probabilities must not be empty")
        total = sum(probabilities.values())
        if total <= 0:
            raise ValueError("opponent pool probabilities must sum to > 0")
        # 归一化 + 只保留 probabilities 中声明的 id
        self._ids = [i for i in ids if probabilities.get(i, 0.0) > 0]
        self._weights = [probabilities[i] for i in self._ids]
        self._opponents = opponents
        self._rng = random.Random(seed)
        self.seed = seed

    def sample(self, n: int = 1) -> list[str]:
        """Sample ``n`` opponent ids with replacement (seeded, reproducible)."""
        return self._rng.choices(self._ids, weights=self._weights, k=n)

    def sample_one(self) -> str:
        return self.sample(1)[0]

    def ids(self) -> list[str]:
        return list(self._ids)

    def opponent(self, opponent_id: str):
        return self._opponents.get(opponent_id)


def register_historical_checkpoints(
    pool_dir: str | Path,
    *,
    prefix: str = "historical",
    pattern: str = "*.pt",
) -> dict[str, str]:
    """Register milestone checkpoints under ``pool_dir`` as historical opponents.

    Returns ``{opponent_id: checkpoint_path}`` for every matching checkpoint
    (sorted by name); do not overwrite an existing ``latest`` — the caller keeps
    ``current-ppo`` separately.
    """
    d = Path(pool_dir)
    if not d.is_dir():
        return {}
    out: dict[str, str] = {}
    for p in sorted(d.glob(pattern)):
        if p.name in ("latest.pt", "best.pt"):
            continue  # 里程碑只注册 epoch_*.pt / 显式 checkpoint
        out[f"{prefix}:{p.stem}"] = str(p)
    return out


def sample_seats(pool: "TrainingOpponentPool", game_index: int, *, learner_type: str = "self", n_opponents: int = 3) -> tuple[int, list[str]]:
    """Assign the learner to ``game_index % 4`` and sample ``n_opponents`` for the other seats.

    Returns ``(learner_seat, seat_opponent_types)`` where ``seat_opponent_types`` has length 4.
    """
    learner_seat = game_index % 4
    sampled = pool.sample(n_opponents)
    types: list[str | None] = [None] * 4
    types[learner_seat] = learner_type
    it = iter(sampled)
    for s in range(4):
        if s != learner_seat:
            types[s] = next(it)
    return learner_seat, [t or "unknown" for t in types]


class FrozenPolicyOpponent:
    """冻结的当前策略副本（current-ppo）：周期性快照，只读、不更新。"""

    opponent_id = "current-ppo"
    model_version = "current-ppo"
    type = "ppo"

    def __init__(self, model, encoder, device="cpu", mode="sampling", temperature=1.0):
        self.model = model
        self.encoder = encoder
        self.device = torch.device(device)
        self.mode = mode
        self.temperature = temperature

    def decide(self, observation, legal_actions, rng):
        from ..decision.action import Action
        from ..features.action_space import action_to_id, legal_mask

        actions = [a for a in legal_actions if isinstance(a, Action)]
        if not actions:
            return legal_actions[0]
        feats = self.encoder.encode(observation).to(self.device).unsqueeze(0)
        mask = legal_mask(actions).to(self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model(feats, mask)
            if self.mode == "greedy":
                aid = int(out.logits.argmax(dim=-1).item())
            else:
                probs = torch.softmax(out.logits / self.temperature, dim=-1)
                gen = torch.Generator(device=probs.device).manual_seed(rng.randrange(1 << 31))
                aid = int(torch.multinomial(probs, 1, generator=gen).squeeze(-1).item())
        return next(a for a in actions if action_to_id(a) == aid)


def build_opponent(opponent_type: str, *, bc_checkpoint, checkpoint_map=None, current_ppo=None, device="cpu", mode="sampling", temperature=1.0):
    """Build one opponent object for training（reuse evaluation opponents）。"""
    from ..evaluation import PolicyOpponent, RandomOpponent, RuleOpponent

    if opponent_type == "random-v1":
        return RandomOpponent(opponent_id="random", model_version="random-v1")
    if opponent_type == "rule-v1":
        return RuleOpponent(opponent_id="rule", model_version="rule-v1")
    if opponent_type == "bc-v2.1":
        return PolicyOpponent(opponent_id="bc_v2_1", type="bc", model_version="bc-v2.1", checkpoint=bc_checkpoint, mode=mode, temperature=temperature, device=device)
    if opponent_type == "ppo-v1":
        path = (checkpoint_map or {}).get("ppo-v1")
        if not path:
            raise ValueError("ppo-v1 opponent requires checkpoint_map['ppo-v1']")
        return PolicyOpponent(opponent_id="ppo_v1", type="ppo", model_version="ppo-v1", checkpoint=path, mode=mode, temperature=temperature, checkpoint_type="ppo", device=device)
    if opponent_type == "historical" or opponent_type.startswith("historical:"):
        # 通用 "historical"（pool 配置键）→ 取第一个已注册的 historical checkpoint
        cm = checkpoint_map or {}
        path = cm.get(opponent_type) if opponent_type.startswith("historical:") else None
        if not path:
            keys = [k for k in cm if k.startswith("historical:")]
            if not keys:
                raise ValueError("historical opponent requires checkpoint_map with historical:* entries (register_historical_checkpoints)")
            path = cm[keys[0]]
            opponent_type = keys[0]
        return PolicyOpponent(opponent_id=opponent_type, type="ppo", model_version=opponent_type, checkpoint=path, mode=mode, temperature=temperature, checkpoint_type="ppo", device=device)
    if opponent_type == "current-ppo":
        if current_ppo is None:
            raise ValueError("current-ppo opponent requires a frozen (model, encoder) snapshot")
        model, encoder = current_ppo
        return FrozenPolicyOpponent(model, encoder, device=device, mode=mode, temperature=temperature)
    raise ValueError(f"unknown opponent_type {opponent_type!r}")


def collect_trajectory_with_opponents(env, model, encoder, device, *, learner_seat: int, opponents: dict, max_steps: int, greedy: bool = False, seed: int = 0):
    """Run one game where ``learner_seat`` uses ``model`` and other seats use ``opponents[seat].decide``.

    只收集 learner 的轨迹用于更新（对手轨迹仅用于 reward 归属，不参与梯度）。
    Returns ``(traj, seat_opponent_types, steps, illegal)``；``traj`` 为 4-seat buffer dict（learner 有完整字段）。
    """
    from ..decision.action import Action
    from ..features.action_space import action_to_id, legal_mask
    from .ppo import attribute_step_rewards, flush_terminal_rewards
    from .ppo_buffer import TrajectoryBuffer, TrajectoryStep
    import random as _random

    env.reset()
    traj = {s: TrajectoryBuffer() for s in range(4)}
    pending = {s: 0.0 for s in range(4)}
    rng = _random.Random(seed)
    illegal = 0
    steps = 0
    while not env.done() and steps < max_steps:
        seat = env.state.turn
        legal = [a for a in env.legal_actions(seat) if isinstance(a, Action)]
        if seat == learner_seat:
            obs = env.observation(seat)
            feats = encoder.encode(obs).to(device)
            mask = legal_mask(legal).to(device)
            with torch.no_grad():
                out = model(feats.unsqueeze(0), mask.unsqueeze(0))
                logp = torch.log_softmax(out.logits, dim=-1)
                if greedy:
                    aid = int(logp.argmax(dim=-1).item())
                else:
                    aid = int(torch.multinomial(torch.exp(logp), 1).squeeze(-1).item())
            action = next(a for a in legal if action_to_id(a) == aid)
            rewards, done = env.step(action)
            reward = attribute_step_rewards(rewards, seat, traj, pending)
            traj[seat].append(
                TrajectoryStep(features=feats.cpu(), action_id=aid, legal_mask=mask.cpu(),
                               old_log_prob=float(logp[0, aid].item()), value=float(out.value[0].item()),
                               reward=reward, done=bool(done))
            )
        else:
            obs = env.observation(seat)
            action = opponents[seat].decide(obs, legal, rng)
            if action not in legal:
                illegal += 1
            rewards, done = env.step(action)
            reward = attribute_step_rewards(rewards, seat, traj, pending)
            # 对手 seat 也记录一步（仅 reward/done 有效），供 reward 归属与 per-opponent 归因
            traj[seat].append(
                TrajectoryStep(features=None, action_id=0, legal_mask=None, old_log_prob=0.0, value=0.0, reward=reward, done=bool(done))
            )
        steps += 1
    if env.done():
        flush_terminal_rewards(env, traj, pending)
    return traj, steps, illegal


__all__ = ["TrainingOpponentPool", "build_opponent", "collect_trajectory_with_opponents", "register_historical_checkpoints", "sample_seats"]
