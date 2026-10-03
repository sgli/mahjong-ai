#!/usr/bin/env python
"""Evaluate a BC checkpoint on validation / test splits (Phase 10.1.1-C §13).

Usage::

    python scripts/evaluate_bc.py --checkpoint experiments/bc_v2/checkpoints/best.pt \
        --dataset-dir data/processed/decision-v2 --mode sampled --samples 100000 \
        --out experiments/bc_v2/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM  # noqa: E402
from mahjong.model import MLPPolicy, remap_bc_state_dict  # noqa: E402
from mahjong.training import (  # noqa: E402
    ParquetBatchDataset,
    collate_record_batches,
    compute_metrics,
    load_training_checkpoint,
)


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _resolve_checkpoint(checkpoint: str) -> Path:
    p = Path(checkpoint)
    if p.is_dir():
        for name in ("best.pt", "latest.pt", "model.pt"):
            cand = p / name
            if cand.exists():
                return cand
        raise SystemExit(f"no best.pt/latest.pt/model.pt under {p}")
    if not p.exists():
        raise SystemExit(f"checkpoint not found: {p}")
    return p


def _load_model(ckpt: dict, device: torch.device) -> MLPPolicy:
    config = ckpt.get("config") or {}
    model_cfg = config.get("model") or {}
    hidden = tuple(model_cfg.get("hidden_sizes", [256, 256]))
    feat_dim = int(config.get("feature_dim", FEATURE_DIM))
    action_size = int(config.get("action_space_size", ACTION_SPACE_SIZE))
    model = MLPPolicy(feat_dim, action_size, hidden).to(device)
    model.load_state_dict(remap_bc_state_dict(ckpt.get("state_dict") or {}), strict=False)
    model.eval()
    return model


def evaluate_split(
    model: MLPPolicy,
    split_dir: Path,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    max_samples: int | None,
) -> dict:
    shards = sorted(split_dir.glob("*.parquet"))
    if not shards:
        return {"error": f"no parquet under {split_dir}"}
    dl = torch.utils.data.DataLoader(
        ParquetBatchDataset(shards, batch_size),
        batch_size=1,
        collate_fn=collate_record_batches,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers else None,
        pin_memory=device.type == "cuda",
    )
    metrics_list = []
    samples = 0
    with torch.no_grad():
        for features, action_ids, masks in dl:
            features = features.to(device, non_blocking=True)
            action_ids = action_ids.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            out = model(features, masks)
            metrics_list.append(compute_metrics(out.logits, action_ids, masks))
            samples += features.shape[0]
            if max_samples is not None and samples >= max_samples:
                break
    n = len(metrics_list)
    avg = {k: sum(m[k] for m in metrics_list) / n for k in metrics_list[0]}
    avg["sampled"] = samples
    return avg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a BC checkpoint on val/test.")
    ap.add_argument("--checkpoint", required=True, help="checkpoint file or directory (best.pt/latest.pt/epoch_N.pt/model.pt)")
    ap.add_argument("--dataset-dir", default="data/processed/decision-v2")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--mode", default="sampled", choices=["full", "sampled"])
    ap.add_argument("--samples", type=int, default=100000, help="samples per split (mode=sampled)")
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--out", default=None, help="output json path (default: <checkpoint_dir>/eval.json)")
    args = ap.parse_args(argv)

    device = _resolve_device(args.device)
    ckpt_path = _resolve_checkpoint(args.checkpoint)
    ckpt = load_training_checkpoint(ckpt_path, map_location="cpu")
    model = _load_model(ckpt, device)

    dataset_dir = Path(args.dataset_dir)
    max_samples = args.samples if args.mode == "sampled" else None
    result = {
        "checkpoint": str(ckpt_path),
        "dataset_dir": str(dataset_dir),
        "mode": args.mode,
        "samples": args.samples if args.mode == "sampled" else None,
        "dataset_version": ckpt.get("config", {}).get("dataset_version"),
        "feature_version": ckpt.get("config", {}).get("feature_version"),
        "model_version": ckpt.get("config", {}).get("model_version"),
        "device": str(device),
        "splits": {},
    }

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        result["splits"][split] = evaluate_split(
            model,
            dataset_dir / split,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
            max_samples=max_samples,
        )
        print(f"{split}: {json.dumps(result['splits'][split], ensure_ascii=False)}")

    out = Path(args.out) if args.out else ckpt_path.parent / "eval.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"eval result -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
