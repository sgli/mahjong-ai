"""BC anchor loss（PPO + β·KL(π_θ‖π_BC)）测试（Phase 10.3 §10 实验 D）。"""

from __future__ import annotations

import torch

from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM
from mahjong.model import MLPPolicy
from mahjong.training import ppo_loss


def _batch(model):
    feats = torch.randn(4, FEATURE_DIM)
    masks = torch.zeros(4, ACTION_SPACE_SIZE, dtype=torch.bool)
    masks[:, :50] = True
    aids = torch.randint(0, 50, (4,))
    with torch.no_grad():
        out = model(feats, masks)
        old_logp = torch.log_softmax(out.logits, dim=-1).gather(-1, aids.unsqueeze(-1)).squeeze(-1)
        ret = out.value.squeeze(-1) + torch.randn(4)
    adv = torch.randn(4)
    return feats, masks, aids, old_logp, adv, ret


def test_beta_zero_equivalent_with_or_without_bc_model():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    bc_model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    bc_model.eval()
    feats, masks, aids, old_logp, adv, ret = _batch(model)
    l1, m1 = ppo_loss(model, feats, masks, aids, old_logp, adv, ret, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01)
    l2, m2 = ppo_loss(model, feats, masks, aids, old_logp, adv, ret, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, bc_model=bc_model, bc_anchor_beta=0.0)
    assert l1.item() == l2.item()
    for k in m1:
        assert m1[k] == m2[k], k


def test_beta_positive_kl_value_matches_manual():
    torch.manual_seed(1)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    bc_model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    bc_model.eval()
    feats, masks, aids, old_logp, adv, ret = _batch(model)
    beta = 0.01
    loss, m = ppo_loss(model, feats, masks, aids, old_logp, adv, ret, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, bc_model=bc_model, bc_anchor_beta=beta)

    theta_logp = model.log_probs(feats, masks)
    bc_logp = bc_model.log_probs(feats, masks)
    theta_probs = torch.exp(theta_logp)
    safe_theta = torch.where(torch.isfinite(theta_logp), theta_logp, torch.zeros_like(theta_logp))
    safe_bc = torch.where(torch.isfinite(bc_logp), bc_logp, torch.zeros_like(bc_logp))
    manual_kl = (theta_probs * (safe_theta - safe_bc)).sum(-1).mean()
    assert abs(m["kl_to_bc"] - manual_kl.item()) < 1e-6


def test_beta_positive_grad_flows_and_bc_frozen():
    torch.manual_seed(2)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    bc_model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    bc_model.eval()
    bc_before = [p.detach().clone() for p in bc_model.parameters()]
    feats, masks, aids, old_logp, adv, ret = _batch(model)
    loss, _ = ppo_loss(model, feats, masks, aids, old_logp, adv, ret, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, bc_model=bc_model, bc_anchor_beta=0.01)
    loss.backward()
    # 梯度流到 policy 参数
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    # BC 参数冻结不变
    for p, b in zip(bc_model.parameters(), bc_before):
        assert torch.equal(p.detach(), b)


def _grad_norm(model):
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += p.grad.abs().sum().item()
    return s


def test_anchor_gradient_scales_with_beta():
    """anchor 梯度确实到达 policy 参数（β 越大 gradnorm 变化越大，回归捕获“anchor 被 detach”）。"""
    torch.manual_seed(3)
    bc_model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32)).eval()
    for p in bc_model.parameters():
        p.requires_grad_(False)

    def run(beta):
        model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
        feats, masks, aids, old_logp, adv, ret = _batch(model)
        loss, _ = ppo_loss(model, feats, masks, aids, old_logp, adv, ret, clip_eps=0.2, vf_coef=0.5, ent_coef=0.0, bc_model=bc_model, bc_anchor_beta=beta)
        model.zero_grad()
        loss.backward()
        return _grad_norm(model)

    g0 = run(0.0)
    g1 = run(1.0)
    g10 = run(10.0)
    assert g1 != g0  # 梯度确实因 anchor 改变
    assert abs(g10 - g0) > abs(g1 - g0)  # 梯度随 β 放大（单调，允许方向变化）


def test_anchor_pulls_theta_toward_bc_toy():
    """玩具：只保留 anchor 项（adv=0、vf=0、ent=0），β=1.0 多步后 KL(θ‖BC) 显著下降。"""
    torch.manual_seed(4)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (8, 8))
    bc_model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (8, 8)).eval()
    for p in bc_model.parameters():
        p.requires_grad_(False)
    feats = torch.randn(2, FEATURE_DIM)
    masks = torch.zeros(2, ACTION_SPACE_SIZE, dtype=torch.bool)
    masks[:, :4] = True
    aids = torch.tensor([0, 1])
    with torch.no_grad():
        out = model(feats, masks)
        old_logp = torch.log_softmax(out.logits, -1).gather(-1, aids.unsqueeze(-1)).squeeze(-1)
        ret = out.value.squeeze(-1) + torch.randn(2)
    adv = torch.zeros(2)

    def kl():
        tl = model.log_probs(feats, masks)
        bl = bc_model.log_probs(feats, masks)
        tp = torch.exp(tl)
        st = torch.where(torch.isfinite(tl), tl, torch.zeros_like(tl))
        sb = torch.where(torch.isfinite(bl), bl, torch.zeros_like(bl))
        return (tp * (st - sb)).sum(-1).mean().item()

    before = kl()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(30):
        loss, _ = ppo_loss(model, feats, masks, aids, old_logp, adv, ret, clip_eps=0.2, vf_coef=0.0, ent_coef=0.0, bc_model=bc_model, bc_anchor_beta=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert kl() < before * 0.05  # 显著拉向 BC
