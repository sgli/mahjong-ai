"""Behavior Cloning training (MODEL_SPEC 8).

Reads the ``decision-v1`` Parquet dataset, encodes observations, and trains an
MLP policy with masked cross-entropy loss.  Data is streamed one Parquet file at
a time so the whole dataset never has to fit in memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from ..dataset.schema import row_to_sample
from ..features.action_space import action_to_id, legal_mask
from ..features.encoder import FEATURE_DIM, ObservationEncoder

log = logging.getLogger("training.bc")


def row_to_tensors(row: dict, encoder: ObservationEncoder):
    sample = row_to_sample(row)
    features = encoder.encode(sample.observation)
    action_id = action_to_id(sample.action)
    mask = legal_mask(sample.legal_actions)
    return features, action_id, mask


def _stack(feat_list, act_list, mask_list, device):
    features = torch.stack(feat_list).to(device)
    action_ids = torch.tensor(act_list, dtype=torch.long, device=device)
    masks = torch.stack(mask_list).to(device)
    return features, action_ids, masks


def iter_batches(
    parquet_files: Iterable[Path],
    encoder: ObservationEncoder,
    batch_size: int,
    device: torch.device,
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
                    yield _stack(feat_list, act_list, mask_list, device)
                    feat_list, act_list, mask_list = [], [], []
    if feat_list:
        yield _stack(feat_list, act_list, mask_list, device)


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
) -> dict:
    model.eval()
    metrics = []
    for f, a, m in iter_batches(val_files, encoder, batch_size, device):
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
) -> list[dict]:
    """Run BC training and return per-epoch train/val metrics."""
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_metrics = []
        for step, (features, action_ids, masks) in enumerate(
            iter_batches(train_files, encoder, batch_size, device)
        ):
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
        val_avg = evaluate(model, val_files, encoder, batch_size, device)
        log.info("epoch %d train=%s val=%s", epoch, train_avg, val_avg)
        history.append({"epoch": epoch, "train": train_avg, "val": val_avg})
    return history


__all__ = [
    "FEATURE_DIM",
    "compute_metrics",
    "evaluate",
    "iter_batches",
    "row_to_tensors",
    "train",
]
