"""PPO ratio / clipping diagnostics (Phase 10.3 诊断 §3)."""

from __future__ import annotations

import math

import torch

from mahjong.training import clip_fraction, clipped_surrogate


def test_ratio_positive_advantage_clipped():
    old = torch.zeros(1)
    new = torch.tensor([math.log(1.5)])  # ratio=1.5 > 1.2
    A = torch.tensor([1.0])
    # min(1.5*1, 1.2*1) = 1.2（被 clip）
    out = clipped_surrogate(new, old, A, 0.2)
    assert abs(out.item() - 1.2) < 1e-6


def test_ratio_negative_advantage_not_clipped_high():
    old = torch.zeros(1)
    new = torch.tensor([math.log(1.5)])  # ratio=1.5
    A = torch.tensor([-1.0])
    # min(-1.5, -1.2) = -1.5（未被 clip，因为更负）
    out = clipped_surrogate(new, old, A, 0.2)
    assert abs(out.item() - (-1.5)) < 1e-6


def test_ratio_low_negative_advantage_clipped():
    old = torch.zeros(1)
    new = torch.tensor([math.log(0.5)])  # ratio=0.5 < 0.8
    A = torch.tensor([-1.0])
    # min(-0.5, -0.8) = -0.8（被 clip 到 0.8*A）
    out = clipped_surrogate(new, old, A, 0.2)
    assert abs(out.item() - (-0.8)) < 1e-6


def test_ratio_in_band_no_clip():
    old = torch.zeros(1)
    new = torch.tensor([math.log(1.1)])  # ratio=1.1 ∈ [0.8, 1.2]
    A = torch.tensor([2.0])
    out = clipped_surrogate(new, old, A, 0.2)
    assert abs(out.item() - 2.2) < 1e-6


def test_clip_fraction_value():
    old = torch.zeros(2)
    new = torch.tensor([math.log(1.5), math.log(1.0)])  # |ratio-1|: 0.5, 0.0
    frac = clip_fraction(old, new, 0.2)
    assert abs(frac.item() - 0.5) < 1e-6
