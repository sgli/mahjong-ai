"""H1：value 独立 trunk（Phase 10.3）。"""

from __future__ import annotations

import torch

from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM
from mahjong.model import MLPPolicy
from mahjong.training.ppo import init_ppo_from_bc

_BC = "experiments/bc_v2.1/checkpoints/best.pt"


def _bc_state_dict():
    import torch as _t

    return _t.load(_BC, map_location="cpu")["state_dict"]


def _fixed_batch(model):
    feats = torch.randn(4, FEATURE_DIM)
    masks = torch.zeros(4, ACTION_SPACE_SIZE, dtype=torch.bool)
    masks[:, :50] = True
    return feats, masks


def test_default_false_unchanged():
    torch.manual_seed(0)
    a = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (16, 16))
    torch.manual_seed(0)
    b = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (16, 16), value_separate_trunk=False)
    fa, ma = _fixed_batch(a)
    assert torch.equal(a(fa, ma).logits, b(fa, ma).logits)
    assert not hasattr(b, "value_trunk")


def test_separate_trunk_policy_equals_bc():
    if not __import__("pathlib").Path(_BC).exists():
        import pytest

        pytest.skip("no bc checkpoint")
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (256, 256), value_separate_trunk=True)
    init_ppo_from_bc(model, _bc_state_dict(), reinit_value_head=True)
    bc = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (256, 256))
    from mahjong.model import remap_bc_state_dict

    bc.load_state_dict({k: v for k, v in remap_bc_state_dict(_bc_state_dict()).items() if not k.startswith("value_head.")}, strict=False)
    feats, masks = _fixed_batch(model)
    assert torch.equal(model(feats, masks).logits, bc(feats, masks).logits)  # policy 侧逐位等于 BC


def test_value_loss_grad_isolated_from_policy():
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (16, 16), value_separate_trunk=True)
    feats, masks = _fixed_batch(model)
    feats = feats.clone().requires_grad_(True)
    out = model(feats, masks)
    loss = (out.value ** 2).sum()
    loss.backward()
    for n, p in model.named_parameters():
        if n.startswith("value_trunk.") or n.startswith("value_head."):
            assert p.grad is not None and p.grad.abs().sum() > 0
        elif n.startswith("trunk.") or n.startswith("action_head."):
            assert p.grad is None or p.grad.abs().sum() == 0  # policy 侧无 value 梯度
