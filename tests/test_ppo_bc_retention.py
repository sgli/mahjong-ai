"""BC retention metric correctness (Phase 10.3 诊断 §2)."""

from __future__ import annotations

import torch

from mahjong.evaluation import retention_metrics


def _random_logits(batch, dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, dim, generator=g)


def _masks(batch, dim, n_legal=50):
    m = torch.zeros(batch, dim, dtype=torch.bool)
    for i in range(batch):
        m[i, i % (dim - n_legal): i % (dim - n_legal) + n_legal] = True
    return m


def test_kl_zero_for_identical_and_nonnegative():
    logits = _random_logits(8, 270)
    masks = _masks(8, 270)
    m = retention_metrics(logits, logits, masks)
    assert m["kl_bc_ppo"].item() < 1e-6
    assert m["kl_ppo_bc"].item() < 1e-6
    assert m["agreement"].item() == 1.0
    assert m["bc_entropy"].item() >= 0.0
    assert m["ppo_entropy"].item() >= 0.0


def test_kl_nonnegative_for_different_logits():
    a = _random_logits(16, 270, seed=1)
    b = _random_logits(16, 270, seed=2)
    masks = _masks(16, 270)
    m = retention_metrics(a, b, masks)
    assert m["kl_bc_ppo"].item() >= -1e-6
    assert m["kl_ppo_bc"].item() >= -1e-6
    assert 0.0 <= m["agreement"].item() <= 1.0
    assert 0.0 <= m["p_bc_action_given_ppo"].item() <= 1.0


def test_deterministic():
    a = _random_logits(8, 270, seed=3)
    b = _random_logits(8, 270, seed=4)
    masks = _masks(8, 270)
    r1 = retention_metrics(a, b, masks)
    r2 = retention_metrics(a, b, masks)
    for k in r1:
        assert torch.equal(r1[k], r2[k]), k


def test_kl_finite_with_masked_inf_logits():
    """非法位 logits 被 mask 成 -inf 时 KL 必须有限（回归：0*(-inf) 的 NaN）。"""
    a = torch.full((4, 270), float("-inf"))
    b = torch.full((4, 270), float("-inf"))
    masks = _masks(4, 270)
    # 给合法位填有限值
    for i in range(4):
        a[i, masks[i]] = torch.randn(int(masks[i].sum()))
        b[i, masks[i]] = torch.randn(int(masks[i].sum()))
    m = retention_metrics(a, b, masks)
    assert torch.isfinite(m["kl_bc_ppo"]).all()
    assert torch.isfinite(m["kl_ppo_bc"]).all()
