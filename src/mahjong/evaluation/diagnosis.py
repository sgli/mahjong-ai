"""Entropy/exploration + value-function diagnosis metrics (Phase 10.3 诊断 §5/§6)."""

from __future__ import annotations

import torch


def entropy_exploration_metrics(logits: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-sample entropy/exploration metrics over the legal distribution."""
    logp = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(logp)
    safe = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))
    entropy = -(probs * safe).sum(-1)

    legal_count = masks.sum(-1).float().clamp(min=1.0)
    # 归一化熵 = entropy / log(legal_count)（均匀分布时为 1）；
    # legal_count==1 时 entropy=0，log(1)=0 → 用 1 代替分母避免 0/0=NaN（归一化=0，即完全确定）
    safe_log_count = torch.where(legal_count > 1, torch.log(legal_count), torch.ones_like(legal_count))
    normalized = entropy / safe_log_count

    argmax_prob = probs.max(-1).values  # greedy 动作概率（分布尖锐度）

    return {
        "legal_count": legal_count,
        "entropy": entropy,
        "normalized_entropy": normalized,
        "argmax_prob": argmax_prob,
    }


def value_quality_metrics(values: torch.Tensor, returns: torch.Tensor) -> dict[str, float]:
    """Value function quality vs actual return (§6)."""
    v = values.float()
    r = returns.float()
    diff = v - r
    mae = diff.abs().mean().item()
    rmse = (diff ** 2).mean().sqrt().item()
    corr = torch.corrcoef(torch.stack([v, r]))[0, 1].item()
    var_r = r.var(unbiased=False).item()
    ev = 1.0 - (diff.var(unbiased=False).item() / (var_r + 1e-12))
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "correlation": float(corr),
        "explained_variance": float(ev),
    }


__all__ = ["entropy_exploration_metrics", "value_quality_metrics"]
