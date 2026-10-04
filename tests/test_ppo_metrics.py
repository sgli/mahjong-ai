"""PPO diagnostic metrics: approx_kl / clip_fraction fixed definitions (§14)."""

from __future__ import annotations

import torch

from mahjong.training import approx_kl, clip_fraction, ppo_loss
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM
from mahjong.model import MLPPolicy


def test_approx_kl_fixed_definition():
    old = torch.tensor([0.5, 1.0, -0.2])
    new = torch.tensor([0.0, 0.5, -0.2])
    # approx_kl = mean(old - new)
    assert torch.allclose(approx_kl(old, new), torch.tensor((0.5 + 0.5 + 0.0) / 3), atol=1e-6)


def test_clip_fraction_fixed_definition():
    old = torch.tensor([0.0, 0.0, 0.0])
    new = torch.tensor([0.0, torch.log(torch.tensor(1.5)), torch.log(torch.tensor(0.5))])
    # ratio = [1, 1.5, 0.5]; |ratio-1| = [0, 0.5, 0.5] > 0.2 -> 2/3
    assert torch.allclose(clip_fraction(old, new, 0.2), torch.tensor(2 / 3), atol=1e-6)


def test_clip_fraction_all_inside_is_zero():
    old = torch.zeros(5)
    new = torch.tensor([0.0, 0.1, -0.1, 0.15, -0.15])  # all |ratio-1| < 0.2
    assert torch.allclose(clip_fraction(old, new, 0.2), torch.tensor(0.0), atol=1e-6)


def test_ppo_loss_reports_kl_and_clip_fraction():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    features = torch.randn(8, FEATURE_DIM)
    mask = torch.zeros(8, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[:, :5] = True
    action_ids = torch.randint(0, 5, (8,))
    old_logp = torch.zeros(8)
    advantages = torch.randn(8)
    returns = torch.randn(8)
    loss, metrics = ppo_loss(
        model, features, mask, action_ids, old_logp, advantages, returns,
        clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
    )
    for k in ("approx_kl", "clip_fraction", "mean_ratio", "policy_loss", "value_loss", "entropy"):
        assert k in metrics and torch.isfinite(torch.tensor(metrics[k]))
