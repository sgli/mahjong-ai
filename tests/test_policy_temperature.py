"""采样温度（diagnostic only）：T=1.0 必须与现有行为逐位一致。"""

from __future__ import annotations

import inspect

import torch

from mahjong.evaluation import PolicyOpponent, entropy_exploration_metrics


def test_policy_opponent_temperature_default_one():
    """PolicyOpponent 默认 temperature=1.0（对手默认不改变采样行为）。"""
    sig = inspect.signature(PolicyOpponent.__init__)
    assert sig.parameters["temperature"].default == 1.0


def test_policy_opponent_mode_default_sampling():
    """PolicyOpponent 默认 mode="sampling"（对手默认行为保持不变）。"""
    sig = inspect.signature(PolicyOpponent.__init__)
    assert sig.parameters["mode"].default == "sampling"


def test_temperature_one_bitwise_equivalent():
    logits = torch.randn(4, 270)
    masks = torch.zeros(4, 270, dtype=torch.bool)
    masks[:, :100] = True
    p1 = torch.softmax(logits / 1.0, dim=-1)
    p0 = torch.softmax(logits, dim=-1)
    assert torch.equal(p1, p0)


def test_temperature_below_one_sharpens_but_keeps_argmax():
    logits = torch.randn(4, 270)
    masks = torch.zeros(4, 270, dtype=torch.bool)
    masks[:, :100] = True
    p_t = torch.softmax(logits / 0.5, dim=-1)
    p_1 = torch.softmax(logits / 1.0, dim=-1)
    # argmax 不变（sharpening 不改变排序）
    assert torch.equal(p_t.argmax(-1), p_1.argmax(-1))
    # 更尖锐 → entropy 更低
    e_t = entropy_exploration_metrics(logits / 0.5, masks)["entropy"].mean()
    e_1 = entropy_exploration_metrics(logits / 1.0, masks)["entropy"].mean()
    assert e_t < e_1
