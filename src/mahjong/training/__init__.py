"""Training (Phase 5 BC + Phase 7 PPO)."""

from .bc import (
    ParquetBatchDataset,
    ParquetIterableDataset,
    collate_record_batches,
    compute_metrics,
    evaluate,
    iter_batches,
    make_collate_fn,
    train,
    train_dataloader,
)
from .checkpoint import (
    capture_rng_states,
    load_checkpoint,
    load_training_checkpoint,
    restore_rng_states,
    save_checkpoint,
    save_training_checkpoint,
)
from .fast_encode import encode_record_batch_fast, encode_row_fast, encode_rows_fast
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
    "ParquetBatchDataset",
    "ParquetIterableDataset",
    "attribute_step_rewards",
    "capture_rng_states",
    "clipped_surrogate",
    "collate_record_batches",
    "collect_trajectory",
    "compute_gae",
    "compute_metrics",
    "encode_record_batch_fast",
    "encode_row_fast",
    "encode_rows_fast",
    "evaluate",
    "flush_terminal_rewards",
    "iter_batches",
    "load_checkpoint",
    "load_training_checkpoint",
    "make_collate_fn",
    "ppo_loss",
    "restore_rng_states",
    "save_checkpoint",
    "save_training_checkpoint",
    "train",
    "train_dataloader",
    "train_ppo",
]
