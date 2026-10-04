"""reward winsorize（E-B）单元测试。"""

from __future__ import annotations

from mahjong.training import compute_gae
from mahjong.training.ppo import _ingest_trajectory
from mahjong.training.ppo_buffer import TrajectoryBuffer, TrajectoryStep


def _step(reward, value=0.0, done=False):
    return TrajectoryStep(features=None, action_id=0, legal_mask=None, old_log_prob=0.0, value=value, reward=reward, done=done)


def _traj(seat0_rewards):
    t = {s: TrajectoryBuffer() for s in range(4)}
    for r in seat0_rewards:
        t[0].append(_step(r))
    return t


def test_reward_clip_zero_no_change():
    t = _traj([5.0, -7.0, 2.0])
    buffer = {"features": [], "action_ids": [], "log_probs": [], "values": [], "rewards": [], "masks": [], "dones": []}
    _ingest_trajectory(t, buffer, [], [], gamma=0.99, lam=0.95, reward_clip=0.0)
    assert buffer["rewards"] == [5.0, -7.0, 2.0]


def test_reward_clip_winsorizes():
    t = _traj([5.0, -7.0, 2.0])
    buffer = {"features": [], "action_ids": [], "log_probs": [], "values": [], "rewards": [], "masks": [], "dones": []}
    _ingest_trajectory(t, buffer, [], [], gamma=0.99, lam=0.95, reward_clip=2.0)
    assert buffer["rewards"] == [2.0, -2.0, 2.0]
    assert all(-2.0 <= r <= 2.0 for r in buffer["rewards"])


def test_gae_uses_clipped_rewards():
    """GAE 确实用截断后的 reward（手算核对）。"""
    rewards = [10.0, -10.0]  # 会被 clip 到 [2, -2]
    clipped = [max(-2.0, min(2.0, r)) for r in rewards]
    values = [0.0, 0.0]
    dones = [False, True]
    adv, ret = compute_gae(clipped, values, dones, gamma=1.0, lam=1.0)
    # t1: delta=-2, gae=-2; t0: delta=2 + (-2)=0, gae=0
    assert adv == [0.0, -2.0]
