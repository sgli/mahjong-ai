"""Training opponent pool with config-driven proportional sampling (Phase 10.3 §12/§13).

Opponent probabilities come **from config** (never hard-coded); sampling is
seeded (same seed ⇒ same sequence).  Historical checkpoint registration is a
thin helper that scans a directory for milestone checkpoints.
"""

from __future__ import annotations

import random
import math
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
        if any(not math.isfinite(float(weight)) or float(weight) < 0 for weight in probabilities.values()):
            raise ValueError("opponent pool probabilities must be finite and non-negative")
        total = sum(probabilities.values())
        if total <= 0:
            raise ValueError("opponent pool probabilities must sum to > 0")
        # 归一化 + 只保留 probabilities 中声明的 id
        self._ids = [i for i in ids if probabilities.get(i, 0.0) > 0]
        if not self._ids:
            raise ValueError("opponent pool must contain at least one positive probability")
        self._weights = [probabilities[i] for i in self._ids]
        self._opponents = opponents
        self._rng = random.Random(seed)
        self.seed = seed

    def sample(self, n: int = 1, *, seed: int | None = None) -> list[str]:
        """Sample ``n`` opponent ids; an optional per-game seed is stateless."""
        rng = random.Random(seed) if seed is not None else self._rng
        return rng.choices(self._ids, weights=self._weights, k=n)

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
    # A milestone may be a directory containing only latest.pt.  In that case
    # register that checkpoint under the directory name; otherwise register
    # explicit immutable snapshots and ignore mutable aliases.
    latest = d / "latest.pt"
    candidates = sorted(d.glob(pattern))
    snapshots = [p for p in candidates if p.name not in ("latest.pt", "best.pt")]
    if latest.is_file() and not snapshots:
        out[f"{prefix}:{d.name}"] = str(latest)
        return out
    for p in snapshots:
        out[f"{prefix}:{p.stem}"] = str(p)
    return out


def sample_seats(pool: "TrainingOpponentPool", game_index: int, *, learner_type: str = "self", n_opponents: int = 3) -> tuple[int, list[str]]:
    """Assign the learner to ``game_index % 4`` and sample ``n_opponents`` for the other seats.

    Returns ``(learner_seat, seat_opponent_types)`` where ``seat_opponent_types`` has length 4.
    """
    learner_seat = game_index % 4
    # Stateless per-game sampling makes the opponent assignment reproducible
    # across resume and avoids replaying the same assignment each epoch.
    sampled = pool.sample(n_opponents, seed=pool.seed + game_index)
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
        from ..decision.action import Action, ActionType
        from ..features.policy_schema import policy_action_ops

        to_id, make_mask, allow_abort = policy_action_ops(self.encoder, self.model)
        actions = [a for a in legal_actions if isinstance(a, Action) and (a.type is not ActionType.KYUUSHU_KYUHAI or allow_abort)]
        if not actions:
            return legal_actions[0]
        feats = self.encoder.encode(observation).to(self.device).unsqueeze(0)
        mask = make_mask(actions).to(self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model(feats, mask)
            if self.mode == "greedy":
                aid = int(out.logits.argmax(dim=-1).item())
            else:
                probs = torch.softmax(out.logits / self.temperature, dim=-1)
                gen = torch.Generator(device=probs.device).manual_seed(rng.randrange(1 << 31))
                aid = int(torch.multinomial(probs, 1, generator=gen).squeeze(-1).item())
        return next(a for a in actions if to_id(a) == aid)


def build_opponent(opponent_type: str, *, bc_checkpoint, checkpoint_map=None, current_ppo=None, device="cpu", mode="sampling", temperature=1.0):
    """Build one opponent object for training（reuse evaluation opponents）。"""
    from ..evaluation import PolicyOpponent, RandomOpponent, RuleOpponent

    if opponent_type == "random-v1":
        return RandomOpponent(opponent_id="random", model_version="random-v1")
    if opponent_type == "rule-v1":
        return RuleOpponent(opponent_id="rule", model_version="rule-v1")
    if opponent_type == "bc-v2.1":
        return PolicyOpponent(opponent_id="bc_v2_1", type="bc", model_version="bc-v2.1", checkpoint=bc_checkpoint, mode=mode, temperature=temperature, device=device)
    if opponent_type == "bc-v3":
        checkpoint = (checkpoint_map or {}).get("bc-v3", bc_checkpoint)
        return PolicyOpponent(opponent_id="bc_v3", type="bc", model_version="bc-v3", checkpoint=checkpoint, mode=mode, temperature=temperature, device=device)
    if opponent_type == "bc-v4":
        checkpoint = (checkpoint_map or {}).get("bc-v4", bc_checkpoint)
        return PolicyOpponent(opponent_id="bc_v4", type="bc", model_version="bc-v4", checkpoint=checkpoint, mode=mode, temperature=temperature, device=device)
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


def collect_trajectory_with_opponents(
    env, model, encoder, device, *, learner_seat: int, opponents: dict, max_steps: int,
    greedy: bool = False, seed: int = 0, max_learner_steps: int | None = None,
    learner_temperature: float = 1.0,
):
    """Run one game where ``learner_seat`` uses ``model`` and other seats use ``opponents[seat].decide``.

    只收集 learner 的轨迹用于更新（对手轨迹仅用于 reward 归属，不参与梯度）。
    Returns ``(traj, steps, illegal)`` normally. With ``max_learner_steps``, also returns
    ``(bootstrap_value, budget_truncated)`` after reaching the next learner decision boundary.
    """
    from ..decision.action import Action, ActionType
    from ..features.policy_schema import policy_action_ops
    from .ppo import attribute_step_rewards, flush_terminal_rewards
    from .ppo_buffer import TrajectoryBuffer, TrajectoryStep
    import random as _random

    if max_learner_steps is not None and max_learner_steps <= 0:
        raise ValueError("max_learner_steps must be positive")
    if learner_temperature <= 0:
        raise ValueError('learner_temperature must be positive')

    to_id, make_mask, allow_abort = policy_action_ops(encoder, model)
    v2_opponent_menu = getattr(encoder, "feature_version", "feature-v1") in ("feature-v2", "feature-v3")

    env.reset()
    traj = {s: TrajectoryBuffer() for s in range(4)}
    pending = {s: 0.0 for s in range(4)}
    rng = _random.Random(seed)
    illegal = 0
    steps = 0
    learner_steps = 0
    while not env.done():
        if max_learner_steps is not None and learner_steps >= max_learner_steps and env.state.turn == learner_seat:
            break
        if steps >= max_steps and max_learner_steps is None:
            break
        if steps >= max_steps and learner_steps < max_learner_steps:
            raise RuntimeError(
                f"game exceeded max_steps={max_steps} before learner budget "
                f"({learner_steps}/{max_learner_steps}) was reached"
            )
        # Once the final learner action is sampled, permit only the short
        # opponent-action tail needed to reach the next learner observation.
        if max_learner_steps is not None and learner_steps >= max_learner_steps and steps >= max_steps + 64:
            raise RuntimeError("could not reach learner bootstrap observation within 64 environment steps")
        seat = env.state.turn
        # The learner's unsupervised abort action is masked. Opponents receive
        # their own complete legal menu and apply their own policy semantics.
        legal = [a for a in env.legal_actions(seat) if isinstance(a, Action)
                 and ((seat != learner_seat and v2_opponent_menu)
                      or a.type is not ActionType.KYUUSHU_KYUHAI or allow_abort)]
        if seat == learner_seat:
            obs = env.observation(seat)
            feats = encoder.encode(obs).to(device)
            mask = make_mask(legal).to(device)
            with torch.no_grad():
                out = model(feats.unsqueeze(0), mask.unsqueeze(0))
                logp = torch.log_softmax(out.logits / learner_temperature, dim=-1)
                if greedy:
                    aid = int(logp.argmax(dim=-1).item())
                else:
                    aid = int(torch.multinomial(torch.exp(logp), 1).squeeze(-1).item())
            action = next(a for a in legal if to_id(a) == aid)
            rewards, done = env.step(action)
            reward = attribute_step_rewards(rewards, seat, traj, pending)
            traj[seat].append(
                TrajectoryStep(features=feats.cpu(), action_id=aid, legal_mask=mask.cpu(),
                               old_log_prob=float(logp[0, aid].item()), value=float(out.value[0].item()),
                               reward=reward, done=bool(done))
            )
            learner_steps += 1
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
        if traj[learner_seat].steps:
            traj[learner_seat].steps[-1].done = True
    if max_learner_steps is None:
        return traj, steps, illegal
    if learner_steps > max_learner_steps:
        raise RuntimeError(f"collected {learner_steps} learner transitions, target was {max_learner_steps}")
    if learner_steps < max_learner_steps and not env.done():
        raise RuntimeError(f"rollout ended with {learner_steps}/{max_learner_steps} learner transitions")
    budget_truncated = learner_steps == max_learner_steps and not env.done()
    bootstrap_value = 0.0
    if budget_truncated:
        if env.state.turn != learner_seat:
            raise RuntimeError("learner budget cut did not reach the learner's next decision state")
        obs = env.observation(learner_seat)
        feats = encoder.encode(obs).to(device)
        legal = [a for a in env.legal_actions(learner_seat) if isinstance(a, Action) and (a.type is not ActionType.KYUUSHU_KYUHAI or allow_abort)]
        mask = make_mask(legal).to(device)
        with torch.no_grad():
            bootstrap_value = float(model(feats.unsqueeze(0), mask.unsqueeze(0)).value[0].item())
    return traj, steps, illegal, bootstrap_value, budget_truncated


__all__ = ["TrainingOpponentPool", "build_opponent", "collect_trajectory_with_opponents", "register_historical_checkpoints", "sample_seats"]
