"""Reward-v1 regression (Phase 10.2 §10/§10.1)."""

from __future__ import annotations

import random

import pytest

from mahjong.decision.action import ActionType
from mahjong.environment import MahjongEnv, RewardConfig
from mahjong.training import attribute_step_rewards, flush_terminal_rewards


def test_reward_config_defaults():
    cfg = RewardConfig()
    assert cfg.score_delta_scale == 0.001
    assert cfg.placement_bonus == (2.0, 1.0, -1.0, -2.0)


def _empty_traj():
    from mahjong.training import TrajectoryBuffer

    return {s: TrajectoryBuffer() for s in range(4)}


def _add_step(buf, reward=0.0):
    from mahjong.training import TrajectoryStep

    buf.append(TrajectoryStep(features=None, action_id=0, legal_mask=None, old_log_prob=0.0, value=0.0, reward=reward, done=False))


def test_attribute_step_rewards_pending_flush():
    """A seat with no transition yet holds its delta in pending until it acts."""
    traj = _empty_traj()
    for s in (0, 1):
        _add_step(traj[s])
    pending = {s: 0.0 for s in range(4)}
    # seat 2 has no transition: its -2.0 delta is pending
    r = attribute_step_rewards({0: 2.0, 2: -2.0, 3: 0.0}, 0, traj, pending)
    assert r == 2.0
    assert pending[2] == -2.0
    # seat 2 later acts: pending flushes into its reward
    _add_step(traj[2])
    r2 = attribute_step_rewards({2: 0.5}, 2, traj, pending)
    assert r2 == 0.5 + -2.0  # its own delta + flushed pending


def test_flush_terminal_rewards_placement_bonus_only_terminal():
    """Placement bonus is added only by flush_terminal_rewards at game end."""
    env = MahjongEnv(seed=3)
    env.reset()
    env.state.final_ranks = [0, 1, 2, 3]
    traj = _empty_traj()
    for s in range(4):
        _add_step(traj[s], reward=1.0)
    pending = {s: 0.0 for s in range(4)}
    flush_terminal_rewards(env, traj, pending)
    # rank 0 gets +2.0, rank 1 +1.0, rank 2 -1.0, rank 3 -2.0
    assert traj[0].steps[-1].reward == 1.0 + 2.0
    assert traj[1].steps[-1].reward == 1.0 + 1.0
    assert traj[2].steps[-1].reward == 1.0 - 1.0
    assert traj[3].steps[-1].reward == 1.0 - 2.0


def test_reward_conservation_full_game():
    """Sum of all four seats' accumulated rewards == 0 (deltas + placement bonus)."""
    env = MahjongEnv(seed=11)
    env.reset()
    rng = random.Random(11)
    steps = 0
    while not env.done():
        seat = env.state.turn
        legal = env.legal_actions(seat)
        action = next((a for a in legal if getattr(a, "type", None) is ActionType.PASS), None)
        if action is None:
            action = rng.choice(legal)
        env.step(action)
        steps += 1
        assert steps < 100_000
    total = sum(env.reward(s) for s in range(4))
    assert abs(total) < 1e-6, f"reward not conserved: {total}"


def test_winner_gets_positive_placement_bonus():
    """Rank-0 seat's accumulated reward includes +2.0 placement bonus."""
    env = MahjongEnv(seed=11)
    env.reset()
    rng = random.Random(11)
    while not env.done():
        seat = env.state.turn
        legal = env.legal_actions(seat)
        action = next((a for a in legal if getattr(a, "type", None) is ActionType.PASS), None)
        if action is None:
            action = rng.choice(legal)
        env.step(action)
    rank0 = env.state.final_ranks.index(0)
    # reward(rank0) = score_delta*scale + 2.0 placement bonus
    assert env.reward(rank0) >= 2.0 - 1e-6
