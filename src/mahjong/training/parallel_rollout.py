"""Parallel / batched rollout (Phase 10.3 §9/§10/§11).

Runs ``N`` MahjongEnv workers **in-process with batch policy inference**:
collect observations across envs → batch encode → batch legal mask → one policy
forward → N actions.  This is the profiling-justified approach (t87: env is
only ~8% of rollout time; per-step encode + per-step forward are ~78%).

Semantics must match the sequential reference (``ppo.collect_trajectory``):
same seed + same action sequence ⇒ same state transitions / rewards / result.
"""

from __future__ import annotations

import numpy as np
import torch

from ..decision.action import Action
from ..features.action_space import ACTION_SPACE_SIZE, action_to_id, legal_mask
from ..features.encoder import FEATURE_DIM, _WINDS, ObservationEncoder
from ..model.policy import MLPPolicy
from ..rules.tiles import RED_FIVE, TILE_TO_INDEX
from .ppo import attribute_step_rewards, flush_terminal_rewards
from .ppo_buffer import TrajectoryBuffer, TrajectoryStep

_TILE_IDX = dict(TILE_TO_INDEX)
for _r, _b in RED_FIVE.items():
    _TILE_IDX[_r] = TILE_TO_INDEX[_b]


def encode_observations_batch(observations) -> torch.Tensor:
    """Batch-encode ``N`` PlayerObservations → ``[N, FEATURE_DIM]`` (§10).

    Bitwise identical to ``ObservationEncoder.encode`` (verified by tests).
    """
    n = len(observations)
    feats = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    for i, obs in enumerate(observations):
        seat = obs.seat

        def add(tiles, off):
            for t in tiles:
                feats[i, off + _TILE_IDX[t]] += 1.0

        add(obs.hand, 0)
        add((t for m in obs.melds for t in m.tiles), 34)
        add((t for slot in obs.opponents_melds for m in slot for t in m.tiles), 68)
        add(obs.discards[seat], 102)
        add((t for s in range(4) if s != seat for t in obs.discards[s]), 136)
        add(obs.dora_markers, 170)

        feats[i, 204 + _WINDS[obs.bakaze]] = 1.0
        feats[i, 208 + (obs.kyoku - 1)] = 1.0
        feats[i, 212] = float(obs.honba)
        feats[i, 213] = float(obs.kyotaku)
        feats[i, 214 + obs.oya] = 1.0
        feats[i, 218 + obs.turn] = 1.0
        feats[i, 222 + seat] = 1.0
        for s_idx, s in enumerate(obs.scores):
            feats[i, 226 + s_idx] = float(s) / 1000.0
        for r_idx, r in enumerate(obs.riichi):
            feats[i, 230 + r_idx] = 1.0 if r else 0.0
    return torch.from_numpy(feats)


def legal_mask_batch(legal_lists) -> torch.Tensor:
    """Batch legal mask → ``[N, ACTION_SPACE_SIZE]`` bool（每 env 独立，不串状态）。"""
    n = len(legal_lists)
    masks = np.zeros((n, ACTION_SPACE_SIZE), dtype=np.bool_)
    for i, legal in enumerate(legal_lists):
        for a in legal:
            masks[i, action_to_id(a)] = True
    return torch.from_numpy(masks)


def collect_batch_trajectory(
    envs,
    model: MLPPolicy,
    encoder: ObservationEncoder,
    device,
    *,
    max_steps: int,
    greedy: bool = False,
) -> tuple[list[dict[int, TrajectoryBuffer]], list[bool], int, int]:
    """Run one game on each env simultaneously with batched inference.

    Returns ``(trajs, dones, steps, illegal)`` where ``trajs[i]`` is a per-seat
    ``TrajectoryBuffer`` dict (same schema as sequential ``collect_trajectory``).
    """
    for env in envs:
        env.reset()
    n = len(envs)
    trajs: list[dict[int, TrajectoryBuffer]] = [{s: TrajectoryBuffer() for s in range(4)} for _ in range(n)]
    pendings: list[dict] = [{s: 0.0 for s in range(4)} for _ in range(n)]
    dones = [False] * n
    illegal = 0
    steps = 0

    while not all(dones) and steps < max_steps:
        active = [i for i in range(n) if not dones[i]]
        if not active:
            break

        legal_list, seat_list, obs_list = [], [], []
        for i in active:
            env = envs[i]
            seat = env.state.turn
            legal = [a for a in env.legal_actions(seat) if isinstance(a, Action)]
            legal_list.append(legal)
            seat_list.append(seat)
            obs_list.append(env.observation(seat))

        # §10 批量编码：observations → [N,234] + [N,270]（与逐步 encode 严格一致）
        feat_batch = encode_observations_batch(obs_list).to(device)
        mask_batch = legal_mask_batch(legal_list).to(device)
        with torch.no_grad():
            out = model(feat_batch, mask_batch)
            log_probs = torch.log_softmax(out.logits, dim=-1)
            if greedy:
                action_ids = log_probs.argmax(dim=-1)
            else:
                probs = torch.exp(log_probs)
                action_ids = torch.multinomial(probs, 1).squeeze(-1)
        values = out.value

        for j, i in enumerate(active):
            env = envs[i]
            seat = seat_list[j]
            legal = legal_list[j]
            aid = int(action_ids[j].item())
            if not mask_batch[j][aid].item():
                illegal += 1  # should never happen (masked argmax/sampling)
            action = next(a for a in legal if action_to_id(a) == aid)
            rewards, d = env.step(action)
            reward = attribute_step_rewards(rewards, seat, trajs[i], pendings[i])
            trajs[i][seat].append(
                TrajectoryStep(
                    features=feat_batch[j].cpu(),
                    action_id=aid,
                    legal_mask=mask_batch[j].cpu(),
                    old_log_prob=float(log_probs[j, aid].item()),
                    value=float(values[j].item()),
                    reward=reward,
                    done=bool(d),
                )
            )
            dones[i] = d
        steps += 1

    for i in range(n):
        if envs[i].done():
            flush_terminal_rewards(envs[i], trajs[i], pendings[i])
    return trajs, dones, steps, illegal


__all__ = ["collect_batch_trajectory"]
