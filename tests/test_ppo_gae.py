"""GAE correctness regression (Phase 10.2 §12)."""

from __future__ import annotations

import torch

from mahjong.training import compute_gae


def test_single_step_terminal():
    # terminal step: delta = r - v, advantage = delta (no bootstrap)
    adv, ret = compute_gae([1.0], [0.5], [True], gamma=0.99, lam=0.95)
    assert torch.allclose(torch.tensor(adv), torch.tensor([0.5]), atol=1e-6)
    assert torch.allclose(torch.tensor(ret), torch.tensor([1.0]), atol=1e-6)


def test_terminal_bootstrap_zero():
    # after a terminal step, next_value must be 0 (no leakage)
    adv, ret = compute_gae([1.0, 1.0], [0.5, 0.5], [False, True], gamma=1.0, lam=1.0)
    # t=1 (terminal): delta = 1 - 0.5 = 0.5; t=0: delta = 1 + 1*0.5 - 0.5 = 1.0
    assert torch.allclose(torch.tensor(adv), torch.tensor([1.5, 0.5]), atol=1e-6)


def test_episode_boundary_no_leak():
    # each contiguous episode must be computed independently (no cross-episode leakage)
    rewards = [1.0, 0.0]  # first episode ends at index 0
    values = [0.5, 0.5]
    dones = [True, True]  # both terminal -> two single-step episodes
    adv, ret = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)
    assert torch.allclose(torch.tensor(adv), torch.tensor([0.5, -0.5]), atol=1e-6)


def test_zero_positive_negative_reward():
    adv, ret = compute_gae([0.0, 1.0, -1.0], [0.0, 0.0, 0.0], [False, False, True], gamma=1.0, lam=1.0)
    # terminal t=2: delta=-1, adv=-1; t=1: delta=1 + (-1) = 0; t=0: delta=0 + 0 = 0
    assert torch.allclose(torch.tensor(adv), torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)
    assert torch.allclose(torch.tensor(ret), torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)


def test_known_hand_calculated():
    # same as existing test_ppo.py hand example (cross-check)
    adv, ret = compute_gae([1.0, 0.0, 1.0], [0.5, 0.5, 0.5], [False, False, True], gamma=0.9, lam=1.0)
    assert torch.allclose(torch.tensor(adv), torch.tensor([1.31, 0.4, 0.5]), atol=1e-6)
    assert torch.allclose(torch.tensor(ret), torch.tensor([1.81, 0.9, 1.0]), atol=1e-6)
