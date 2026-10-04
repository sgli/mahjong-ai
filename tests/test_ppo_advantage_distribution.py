"""GAE / return / reward distribution audit (Phase 10.3 诊断 §4)."""

from __future__ import annotations

import math

import torch

from mahjong.training import compute_gae


def test_compute_gae_known_sequence():
    # gamma=1, lam=1：advantages = 未来 return 的逆转
    rewards = [1.0, 1.0, 1.0]
    values = [0.0, 0.0, 0.0]
    dones = [False, False, True]
    adv, ret = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)
    # t2: delta=1, gae=1; t1: delta=1, gae=1+1*1*1*1=2; t0: delta=1, gae=1+1*1*1*2=3
    assert adv == [3.0, 2.0, 1.0]
    assert ret == [3.0, 2.0, 1.0]  # value=0 → return == advantage


def test_compute_gae_terminal_resets():
    # done 处 nonterminal=0 → 不把下一条轨迹的 value 传回来
    rewards = [1.0, 1.0, 1.0]
    values = [0.5, 0.5, 0.5]
    dones = [False, True, False]
    adv, ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    assert len(adv) == 3 and len(ret) == 3
    assert all(math.isfinite(a) for a in adv)
    # t1 done：其 delta 只看自身（nonterminal=0，next_value 不计）
    assert abs(ret[1] - adv[1] - values[1]) < 1e-9


def test_distribution_stats_finite():
    rewards = [0.5, -1.0, 2.0, 0.0, 1.5]
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    dones = [False, False, False, False, True]
    adv, ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    for seq in (rewards, values, adv, ret):
        t = torch.tensor(seq)
        mean = t.mean().item()
        std = t.std().item()
        mn = t.min().item()
        mx = t.max().item()
        assert math.isfinite(mean) and math.isfinite(std) and math.isfinite(mn) and math.isfinite(mx)
