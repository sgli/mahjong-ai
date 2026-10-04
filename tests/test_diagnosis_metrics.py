"""Entropy/exploration + value-quality metric correctness (Phase 10.3 诊断 §5/§6)."""

from __future__ import annotations

import math

import torch

from mahjong.evaluation import entropy_exploration_metrics, value_quality_metrics


def test_uniform_distribution_normalized_entropy_one():
    # 合法 8 个动作，均匀分布 → entropy=log(8)，归一化熵=1，argmax_prob=1/8
    logits = torch.full((2, 270), float("-inf"))
    masks = torch.zeros(2, 270, dtype=torch.bool)
    for i in range(2):
        masks[i, :8] = True
        logits[i, :8] = 0.0
    m = entropy_exploration_metrics(logits, masks)
    assert torch.allclose(m["legal_count"], torch.tensor([8.0, 8.0]))
    assert torch.allclose(m["normalized_entropy"], torch.ones(2), atol=1e-5)
    assert torch.allclose(m["argmax_prob"], torch.full((2,), 1 / 8), atol=1e-5)


def test_single_legal_action_normalized_entropy_zero_not_nan():
    # 只有 1 个合法动作 → entropy=0、log(1)=0 → 归一化熵应为 0（非 NaN）
    logits = torch.full((1, 270), float("-inf"))
    masks = torch.zeros(1, 270, dtype=torch.bool)
    masks[0, 5] = True
    logits[0, 5] = 0.0
    m = entropy_exploration_metrics(logits, masks)
    assert torch.isfinite(m["normalized_entropy"]).all()
    assert m["normalized_entropy"].item() == 0.0


def test_sharp_distribution_low_entropy():
    # 单峰分布 → 熵低、argmax_prob 高
    logits = torch.full((1, 270), float("-inf"))
    masks = torch.zeros(1, 270, dtype=torch.bool)
    masks[0, :4] = True
    logits[0, 0] = 10.0
    logits[0, 1] = 0.0
    logits[0, 2] = 0.0
    logits[0, 3] = 0.0
    m = entropy_exploration_metrics(logits, masks)
    assert m["entropy"].item() < 0.2
    assert m["argmax_prob"].item() > 0.9


def test_value_quality_perfect_prediction():
    v = torch.tensor([1.0, 2.0, 3.0, 4.0])
    r = v.clone()
    m = value_quality_metrics(v, r)
    assert m["mae"] < 1e-6
    assert m["rmse"] < 1e-6
    assert abs(m["correlation"] - 1.0) < 1e-5
    assert m["explained_variance"] > 0.999


def test_value_quality_imperfect_finite():
    v = torch.tensor([0.0, 1.0, 2.0, 3.0])
    r = torch.tensor([0.5, 0.5, 3.0, 2.0])
    m = value_quality_metrics(v, r)
    assert math.isfinite(m["mae"]) and math.isfinite(m["rmse"])
    assert -1.0 <= m["correlation"] <= 1.0
