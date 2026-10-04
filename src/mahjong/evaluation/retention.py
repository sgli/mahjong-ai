"""BC retention metrics (Phase 10.3 诊断 §2).

All metrics are computed over the legal action distribution only (illegal logits
are ``-inf`` after masking).  Numerically stable: masked ``-inf`` log-probs are
zeroed for entropy, and KL uses ``probs * (logp_a - logp_b)`` so a zero-prob
term never produces NaN.
"""

from __future__ import annotations

import torch


def _masked_dist(logits: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logp = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(logp)
    safe = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))
    entropy = -(probs * safe).sum(-1)
    return probs, logp, entropy


def retention_metrics(bc_logits: torch.Tensor, ppo_logits: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compare BC vs PPO on the same (masked) logits; returns per-batch means."""
    bc_probs, bc_logp, bc_ent = _masked_dist(bc_logits, masks)
    ppo_probs, ppo_logp, ppo_ent = _masked_dist(ppo_logits, masks)

    # 非法位 logp = -inf：差值用 safe（零化）版本避免 -inf - (-inf) = NaN；
    # 该位 probs=0，故对 KL 贡献为 0（数值稳定）。
    bc_safe = torch.where(torch.isfinite(bc_logp), bc_logp, torch.zeros_like(bc_logp))
    ppo_safe = torch.where(torch.isfinite(ppo_logp), ppo_logp, torch.zeros_like(ppo_logp))
    kl_bc_ppo = (bc_probs * (bc_safe - ppo_safe)).sum(-1)
    kl_ppo_bc = (ppo_probs * (ppo_safe - bc_safe)).sum(-1)

    bc_arg = bc_logp.argmax(-1)
    ppo_arg = ppo_logp.argmax(-1)
    agreement = (bc_arg == ppo_arg).float()
    p_bc_given_ppo = ppo_probs.gather(-1, bc_arg.unsqueeze(-1)).squeeze(-1)

    return {
        "agreement": agreement.mean(),
        "kl_bc_ppo": kl_bc_ppo.mean(),
        "kl_ppo_bc": kl_ppo_bc.mean(),
        "bc_entropy": bc_ent.mean(),
        "ppo_entropy": ppo_ent.mean(),
        "p_bc_action_given_ppo": p_bc_given_ppo.mean(),
    }


__all__ = ["retention_metrics"]
