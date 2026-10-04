"""PPO ratio / clipping regression (Phase 10.2 §13)."""

from __future__ import annotations

import torch

from mahjong.training import clipped_surrogate


def test_ratio_less_than_one_minus_clip_eps():
    # ratio < 1-eps with positive advantage -> clipped up to 1-eps
    s = clipped_surrogate(torch.tensor([-1.0]), torch.tensor([0.0]), torch.tensor([1.0]), 0.2)
    # ratio = exp(-1) ≈ 0.3679; clip(ratio, 0.8, 1.2) = 0.8; min(0.3679*1, 0.8*1) = 0.3679
    assert torch.allclose(s, torch.tensor([torch.exp(torch.tensor(-1.0))]), atol=1e-6)


def test_sign_direction():
    """min(ratio*A, clip(ratio)*A) must never flip the sign of A."""
    A = torch.tensor([1.0, -1.0, 1.0, -1.0])
    logp_new = torch.tensor([0.0, 2.0, -2.0, 0.5])
    logp_old = torch.tensor([0.0, 0.0, 0.0, 0.0])
    s = clipped_surrogate(logp_new, logp_old, A, 0.2)
    assert (s * A >= 0).all()  # same sign as advantage
