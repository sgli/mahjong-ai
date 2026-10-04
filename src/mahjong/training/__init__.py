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
    approx_kl,
    attribute_step_rewards,
    clip_fraction,
    clipped_surrogate,
    collect_trajectory,
    compute_gae,
    flush_terminal_rewards,
    init_ppo_from_bc,
    ppo_loss,
    train_ppo,
)
from .ppo_buffer import EpisodeMetadata, TrajectoryBuffer, TrajectoryStep
from .ppo_checkpoint import (
    capture_ppo_rng_states,
    load_ppo_checkpoint,
    restore_ppo_rng_states,
    save_ppo_checkpoint,
)

__all__ = [
    "ParquetBatchDataset",
    "ParquetIterableDataset",
    "attribute_step_rewards",
    "EpisodeMetadata",
    "TrajectoryBuffer",
    "TrajectoryStep",
    "approx_kl",
    "capture_ppo_rng_states",
    "capture_rng_states",
    "clip_fraction",
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
    "init_ppo_from_bc",
    "iter_batches",
    "load_checkpoint",
    "load_ppo_checkpoint",
    "load_training_checkpoint",
    "make_collate_fn",
    "ppo_loss",
    "restore_ppo_rng_states",
    "restore_rng_states",
    "save_checkpoint",
    "save_ppo_checkpoint",
    "save_training_checkpoint",
    "train",
    "train_dataloader",
    "train_ppo",
]
