"""Behavior Cloning training (MODEL_SPEC 8).

Reads the ``decision-v1`` Parquet dataset, encodes observations, and trains an
MLP policy with masked cross-entropy loss.  Data is streamed one Parquet file at
a time so the whole dataset never has to fit in memory.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from ..dataset.schema import row_to_sample
from ..features.action_space import action_to_id, legal_mask
from ..features.encoder import FEATURE_DIM, ObservationEncoder
from .fast_encode import encode_record_batch_fast, encode_rows_fast

log = logging.getLogger("training.bc")


def row_to_tensors(row: dict, encoder: ObservationEncoder):
    sample = row_to_sample(row)
    features = encoder.encode(sample.observation)
    action_id = action_to_id(sample.action)
    mask = legal_mask(sample.legal_actions)
    return features, action_id, mask


def _stack(feat_list, act_list, mask_list, device, pin_memory=False):
    features = torch.stack(feat_list)
    action_ids = torch.tensor(act_list, dtype=torch.long)
    masks = torch.stack(mask_list)
    if pin_memory and device.type == "cuda":
        features = features.pin_memory()
        action_ids = action_ids.pin_memory()
        masks = masks.pin_memory()
    return (
        features.to(device, non_blocking=pin_memory),
        action_ids.to(device, non_blocking=pin_memory),
        masks.to(device, non_blocking=pin_memory),
    )


def iter_batches(
    parquet_files: Iterable[Path],
    encoder: ObservationEncoder,
    batch_size: int,
    device: torch.device,
    pin_memory: bool = False,
):
    """Stream ``(features, action_ids, legal_masks)`` batches from Parquet files.

    Record batches are read lazily via :meth:`pyarrow.parquet.ParquetFile.iter_batches`,
    so neither the whole dataset nor a whole file is materialised in memory.
    """
    feat_list, act_list, mask_list = [], [], []
    for path in parquet_files:
        parquet = pq.ParquetFile(path)
        for record_batch in parquet.iter_batches(batch_size=batch_size):
            for row in record_batch.to_pylist():
                f, a, m = row_to_tensors(row, encoder)
                feat_list.append(f)
                act_list.append(a)
                mask_list.append(m)
                if len(feat_list) >= batch_size:
                    yield _stack(feat_list, act_list, mask_list, device, pin_memory)
                    feat_list, act_list, mask_list = [], [], []
    if feat_list:
        yield _stack(feat_list, act_list, mask_list, device, pin_memory)


def compute_metrics(logits: torch.Tensor, action_ids: torch.Tensor, masks: torch.Tensor) -> dict:
    """Metrics for one batch (masked CE loss, accuracy, top-k, illegal mass/rate)."""
    masked = logits.masked_fill(~masks, float("-inf"))
    loss = F.cross_entropy(masked, action_ids)
    masked_probs = torch.softmax(masked, -1)
    pred = masked.argmax(-1)
    accuracy = (pred == action_ids).float().mean()

    k3 = min(3, masked.size(-1))
    k5 = min(5, masked.size(-1))
    topk3 = masked.topk(k3, dim=-1).indices
    topk5 = masked.topk(k5, dim=-1).indices
    top3 = (topk3 == action_ids.unsqueeze(-1)).any(-1).float().mean()
    top5 = (topk5 == action_ids.unsqueeze(-1)).any(-1).float().mean()

    # Illegal-action diagnostics on the *masked* policy (must be ~0: this is the
    # whole point of the legal mask).
    illegal_prob = (masked_probs * (~masks).float()).sum(-1).mean()
    pred_legal = masks.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
    illegal_rate = (~pred_legal).float().mean()

    return {
        "loss": float(loss.item()),
        "accuracy": float(accuracy.item()),
        "top3": float(top3.item()),
        "top5": float(top5.item()),
        "illegal_prob": float(illegal_prob.item()),
        "illegal_rate": float(illegal_rate.item()),
    }


def _average(metrics_list) -> dict:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}


@torch.no_grad()
def evaluate(
    model,
    val_files: Iterable[Path],
    encoder: ObservationEncoder,
    batch_size: int,
    device: torch.device,
    use_amp: bool = False,
    pin_memory: bool = False,
) -> dict:
    model.eval()
    metrics = []
    for f, a, m in iter_batches(val_files, encoder, batch_size, device, pin_memory):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(f, m)
        metrics.append(compute_metrics(out.logits, a, m))
    return _average(metrics)


def train(
    model,
    optimizer,
    train_files: Iterable[Path],
    val_files: Iterable[Path],
    encoder: ObservationEncoder,
    *,
    epochs: int,
    batch_size: int,
    device: torch.device,
    log_interval_steps: int = 50,
    use_amp: bool = False,
    pin_memory: bool = False,
) -> list[dict]:
    """Run BC training and return per-epoch train/val metrics."""
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_metrics = []
        for step, (features, action_ids, masks) in enumerate(
            iter_batches(train_files, encoder, batch_size, device, pin_memory)
        ):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(features, masks)
                loss = F.cross_entropy(out.logits, action_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mets = compute_metrics(out.logits.detach(), action_ids, masks)
            train_metrics.append(mets)
            if step % log_interval_steps == 0:
                log.info(
                    "epoch %d step %d loss=%.4f acc=%.4f illegal_rate=%.4f",
                    epoch,
                    step,
                    mets["loss"],
                    mets["accuracy"],
                    mets["illegal_rate"],
                )
        train_avg = _average(train_metrics)
        val_avg = evaluate(model, val_files, encoder, batch_size, device, use_amp=use_amp, pin_memory=pin_memory)
        log.info("epoch %d train=%s val=%s", epoch, train_avg, val_avg)
        history.append({"epoch": epoch, "train": train_avg, "val": val_avg})
    return history


# -- DataLoader pipeline (parallel parquet decode + feature encoding) ---------
class ParquetIterableDataset(torch.utils.data.IterableDataset):
    """Stream parquet shards; each DataLoader worker gets a disjoint shard subset.

    Shard partitioning is ``shards[worker_id::num_workers]``, so with a fixed
    shard list + seed the assignment is deterministic and no shard is duplicated
    across workers.
    """

    def __init__(self, shards: list[Path]):
        super().__init__()
        self.shards = list(shards)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            my_shards = self.shards
        else:
            my_shards = self.shards[worker.id :: worker.num_workers]
        for shard in my_shards:
            parquet = pq.ParquetFile(shard)
            # 行组级流式读取：一次只 materialize 一个 record batch
            for record_batch in parquet.iter_batches(batch_size=1024):
                for row in record_batch.to_pylist():
                    yield row


def _collate_rows(rows: list[dict], encoder: ObservationEncoder):
    """Top-level collate (picklable for Windows spawn DataLoader workers)."""
    feat, act, mask = [], [], []
    for row in rows:
        sample = row_to_sample(row)
        feat.append(encoder.encode(sample.observation))
        act.append(action_to_id(sample.action))
        mask.append(legal_mask(sample.legal_actions))
    features = torch.stack(feat)
    action_ids = torch.tensor(act, dtype=torch.long)
    masks = torch.stack(mask)
    return features, action_ids, masks


def make_collate_fn(encoder: ObservationEncoder):
    """collate: 把 raw parquet rows 编码成 (features, action_ids, masks)，返回 CPU 张量。

    特征编码在 DataLoader worker 进程内完成（并行）；pin_memory 由 DataLoader 处理。
    """
    import functools

    return functools.partial(_collate_rows, encoder=encoder)


class ParquetBatchDataset(torch.utils.data.IterableDataset):
    """Stream parquet *record batches* (columnar) instead of individual rows.

    Each DataLoader worker gets a disjoint shard subset (same deterministic rule
    as ``ParquetIterableDataset``); a yielded item is one Arrow RecordBatch of
    ``batch_size`` rows, encoded by :func:`encode_record_batch_fast` in collate.
    """

    def __init__(self, shards: list[Path], batch_size: int = 512):
        super().__init__()
        self.shards = list(shards)
        self.batch_size = batch_size

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        my_shards = self.shards[worker.id :: worker.num_workers] if worker is not None else self.shards
        for shard in my_shards:
            parquet = pq.ParquetFile(shard)
            for record_batch in parquet.iter_batches(batch_size=self.batch_size):
                yield record_batch


def collate_record_batches(batches) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate one or more Arrow RecordBatches -> (features, action_ids, masks)."""
    feats, aids, masks = [], [], []
    for rb in batches:
        f, a, m = encode_record_batch_fast(rb)
        feats.append(f)
        aids.append(a)
        masks.append(m)
    return torch.cat(feats), torch.cat(aids), torch.cat(masks)


def _shuffle_shards(shards: list[Path], seed: int) -> list[Path]:
    """Deterministic shard shuffle (``shard_order = f(seed)``)."""
    import random as _random

    rng = _random.Random(seed)
    out = list(shards)
    rng.shuffle(out)
    return out


def train_dataloader(
    model,
    optimizer,
    train_shards: list[Path],
    val_shards: list[Path],
    encoder: ObservationEncoder,
    *,
    epochs: int,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    prefetch_factor: int = 4,
    log_interval_steps: int = 50,
    use_amp: bool = False,
    start_epoch: int = 0,
    on_epoch_end=None,
    pin_memory: bool = False,
    shuffle_seed: int | None = None,
    val_max_samples: int | None = None,
) -> list[dict]:
    """DataLoader 版 BC 训练（与 ``train`` 返回相同 history 结构）。

    ``pin_memory`` 现在由配置显式控制（§11.1 修正）：False 时 DataLoader 不 pin。
    ``shuffle_seed`` 非空时，每 epoch 的 shard 顺序由 ``shuffle_seed + epoch`` 决定
    （同 seed + 同 epoch ⇒ 完全相同顺序，§12）。
    ``val_max_samples`` 非空时，每 epoch 只采样该样本数做快速验证（§13）。
    """
    history = []
    for epoch in range(start_epoch + 1, epochs + 1):
        train_shards_epoch = (
            _shuffle_shards(train_shards, shuffle_seed + epoch) if shuffle_seed is not None else train_shards
        )
        train_dl = torch.utils.data.DataLoader(
            ParquetBatchDataset(train_shards_epoch, batch_size),
            batch_size=1,
            collate_fn=collate_record_batches,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=pin_memory and device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
        val_dl = torch.utils.data.DataLoader(
            ParquetBatchDataset(val_shards, batch_size),
            batch_size=1,
            collate_fn=collate_record_batches,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=pin_memory and device.type == "cuda",
            persistent_workers=num_workers > 0,
        )

        model.train()
        train_metrics = []
        prof = {"data_time": 0.0, "forward_time": 0.0, "backward_time": 0.0, "optimizer_time": 0.0, "batch_time": 0.0, "samples": 0}
        batch_start = time.perf_counter()
        for step, (features, action_ids, masks) in enumerate(train_dl):
            data_end = time.perf_counter()
            prof["data_time"] += data_end - batch_start

            features = features.to(device, non_blocking=True)
            action_ids = action_ids.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(features, masks)
                loss = F.cross_entropy(out.logits, action_ids)
            fwd_end = time.perf_counter()
            prof["forward_time"] += fwd_end - data_end

            optimizer.zero_grad()
            loss.backward()
            bwd_end = time.perf_counter()
            prof["backward_time"] += bwd_end - fwd_end

            optimizer.step()
            opt_end = time.perf_counter()
            prof["optimizer_time"] += opt_end - bwd_end
            prof["samples"] += features.shape[0]

            mets = compute_metrics(out.logits.detach(), action_ids, masks)
            train_metrics.append(mets)
            if step % log_interval_steps == 0:
                log.info(
                    "epoch %d step %d loss=%.4f acc=%.4f illegal_rate=%.4f",
                    epoch, step, mets["loss"], mets["accuracy"], mets["illegal_rate"],
                )
            batch_start = time.perf_counter()
        prof["batch_time"] = prof["data_time"] + prof["forward_time"] + prof["backward_time"] + prof["optimizer_time"]
        prof["samples_per_sec"] = prof["samples"] / prof["batch_time"] if prof["batch_time"] else 0.0
        if device.type == "cuda":
            torch.cuda.synchronize()
            prof["gpu_util"] = None  # nvidia-smi outside python; memory below
            prof["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
            prof["gpu_memory_max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
        train_avg = _average(train_metrics)

        model.eval()
        val_metrics = []
        val_samples = 0
        with torch.no_grad():
            for features, action_ids, masks in val_dl:
                features = features.to(device, non_blocking=True)
                action_ids = action_ids.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = model(features, masks)
                val_metrics.append(compute_metrics(out.logits, action_ids, masks))
                val_samples += features.shape[0]
                if val_max_samples is not None and val_samples >= val_max_samples:
                    break
        val_avg = _average(val_metrics)
        val_avg["sampled"] = val_samples
        log.info("epoch %d train=%s val=%s profile=%s", epoch, train_avg, val_avg, prof)
        entry = {"epoch": epoch, "train": train_avg, "val": val_avg, "profile": prof}
        history.append(entry)
        if on_epoch_end is not None:
            on_epoch_end(epoch, train_avg, val_avg, prof)
    return history


__all__ = [
    "FEATURE_DIM",
    "ParquetIterableDataset",
    "compute_metrics",
    "evaluate",
    "iter_batches",
    "make_collate_fn",
    "row_to_tensors",
    "train",
    "train_dataloader",
]
