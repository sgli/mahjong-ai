"""Training (Phase 5 BC + Phase 7 PPO)."""

from .bc import compute_metrics, evaluate, iter_batches, train
from .checkpoint import load_checkpoint, save_checkpoint
from .ppo import (
    attribute_step_rewards,
    clipped_surrogate,
    collect_trajectory,
    compute_gae,
    flush_terminal_rewards,
    ppo_loss,
    train_ppo,
)

__all__ = [
    "attribute_step_rewards",
    "clipped_surrogate",
    "collect_trajectory",
    "compute_gae",
    "compute_metrics",
    "evaluate",
    "flush_terminal_rewards",
    "iter_batches",
    "load_checkpoint",
    "ppo_loss",
    "save_checkpoint",
    "train",
    "train_ppo",
]
