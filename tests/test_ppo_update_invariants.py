"""PPO zero-advantage invariant (Phase 10.3 诊断 §3)."""

from __future__ import annotations

import torch

from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM
from mahjong.model import MLPPolicy
from mahjong.training import ppo_loss


def test_zero_advantage_policy_logits_unchanged():
    """advantages=0 + ent_coef=0 → 一次 update 后 policy logits 不变（浮点误差内）。"""
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    feats = torch.randn(8, FEATURE_DIM)
    masks = torch.zeros(8, ACTION_SPACE_SIZE, dtype=torch.bool)
    masks[:, :100] = True
    aids = torch.randint(0, 100, (8,))

    with torch.no_grad():
        out = model(feats, masks)
        old_logits = out.logits.clone()
        old_logp = torch.log_softmax(out.logits, dim=-1).gather(-1, aids.unsqueeze(-1)).squeeze(-1)
        returns = out.value.squeeze(-1).clone()  # return = value（adv=0）

    advantages = torch.zeros(8)
    loss, _ = ppo_loss(
        model, feats, masks, aids, old_logp, advantages, returns,
        clip_eps=0.2, vf_coef=0.5, ent_coef=0.0,
    )
    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad():
        new_logits = model(feats, masks).logits
    assert torch.allclose(old_logits, new_logits, atol=1e-6)
