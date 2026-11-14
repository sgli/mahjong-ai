#!/usr/bin/env python
"""Evaluate a BC checkpoint on validation / test splits (Phase 10.1.1-C §13).

Usage::

    python scripts/evaluate_bc.py --checkpoint experiments/bc_v2/checkpoints/best.pt \
        --dataset-dir data/processed/decision-v2 --mode sampled --samples 100000 \
        --out experiments/bc_v2/eval.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM  # noqa: E402
from mahjong.features import ObservationEncoder  # noqa: E402
from mahjong.features.action_space_v2 import ACTION_SPACE_SIZE_V2  # noqa: E402
from mahjong.features.encoder_v2 import FEATURE_DIM_V2, ObservationEncoderV2  # noqa: E402
from mahjong.features.encoder_v3 import FEATURE_DIM_V3, ObservationEncoderV3  # noqa: E402
from mahjong.training.fast_encode import encode_record_batch_fast, encode_record_batch_fast_v2  # noqa: E402
from mahjong.training.fast_encode_v3 import encode_record_batch_fast_v3  # noqa: E402
from mahjong.training.bc import MetricAccumulator
from mahjong.model import build_policy_model, remap_bc_state_dict  # noqa: E402
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


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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
    if config.get("feature_version") == "feature-v2" and config.get("action_schema_version") == "action-v2":
        default_feat_dim, default_action_size = FEATURE_DIM_V2, ACTION_SPACE_SIZE_V2
    elif config.get("feature_version") == "feature-v3" and config.get("action_schema_version") == "action-v2":
        default_feat_dim, default_action_size = FEATURE_DIM_V3, ACTION_SPACE_SIZE_V2
    else:
        default_feat_dim, default_action_size = FEATURE_DIM, ACTION_SPACE_SIZE
    feat_dim = int(config.get("feature_dim", default_feat_dim))
    action_size = int(config.get("action_space_size", default_action_size))
    model = build_policy_model(
        config.get("feature_version", "feature-v1"), feat_dim, action_size, hidden,
        model_config=model_cfg,
    ).to(device)
    model.load_state_dict(
        remap_bc_state_dict(ckpt.get("state_dict") or {}),
        strict=config.get("feature_version") == "feature-v3",
    )
    model.eval()
    return model


def _collate_eval_batches(
    batches, *, encoder, schema_v2: bool, schema_v3: bool, include_groups: bool = False
):
    if schema_v3:
        encoded = [encode_record_batch_fast_v3(rb) for rb in batches]
    elif schema_v2:
        encoded = [encode_record_batch_fast_v2(rb) for rb in batches]
    else:
        encoded = [encode_record_batch_fast(rb) for rb in batches]
    result = tuple(encoded[0]) if len(encoded) == 1 else tuple(
        torch.cat([x[i] for x in encoded]) for i in range(3)
    )
    if not include_groups:
        return result
    meta = []
    for rb in batches:
        columns = [rb.column(name).to_pylist() for name in ("decision_kind", "action_type", "action_tile", "action_consumed")]
        for kind, action_type, action_tile, consumed in zip(*columns):
            red_label = bool((action_tile or "").endswith("r") or any(t.endswith("r") for t in (consumed or [])))
            meta.append((kind, action_type, red_label))
    return result + (meta,)


def evaluate_split(
    model: MLPPolicy,
    split_dir: Path,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    max_samples: int | None,
    encoder=None,
    schema_v2: bool = False,
    schema_v3: bool = False,
    include_groups: bool = False,
) -> dict:
    shards = sorted(split_dir.glob("*.parquet"))
    if not shards:
        return {"error": f"no parquet under {split_dir}"}
    dl = torch.utils.data.DataLoader(
        ParquetBatchDataset(shards, batch_size),
        batch_size=1,
        collate_fn=__import__("functools").partial(
            _collate_eval_batches,
            encoder=encoder,
            schema_v2=schema_v2,
            schema_v3=schema_v3,
            include_groups=include_groups,
        ),
        num_workers=num_workers,
        prefetch_factor=4 if num_workers else None,
        pin_memory=device.type == "cuda",
    )
    metrics = MetricAccumulator()
    group_counts = defaultdict(lambda: [0, 0]) if include_groups else None
    samples = 0
    with torch.no_grad():
        for batch in dl:
            features, action_ids, masks = batch[:3]
            meta = batch[3] if include_groups else None
            if max_samples is not None:
                remaining = max_samples - samples
                if remaining <= 0:
                    break
                features, action_ids, masks = features[:remaining], action_ids[:remaining], masks[:remaining]
                if include_groups:
                    meta = meta[:remaining]
            features = features.to(device, non_blocking=True)
            action_ids = action_ids.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            out = model(features, masks)
            # Keep metric totals on-device; synchronizing six scalar metrics on
            # every batch serializes the evaluator and starves the input workers.
            metrics.update(
                compute_metrics(out.logits, action_ids, masks, tensor_values=True),
                len(action_ids),
            )
            if include_groups:
                predictions = out.logits.argmax(dim=-1).cpu().tolist()
                labels = action_ids.cpu().tolist()
                for pred, label, (kind, action_type, red_label) in zip(predictions, labels, meta):
                    for group in (f"decision:{kind}", f"action:{action_type}"):
                        group_counts[group][0] += int(pred == label)
                        group_counts[group][1] += 1
                    if red_label:
                        group_counts["red_label"][0] += int(pred == label)
                        group_counts["red_label"][1] += 1
            samples += features.shape[0]
            if max_samples is not None and samples >= max_samples:
                break
    avg = metrics.compute()
    avg["sampled"] = samples
    if include_groups:
        avg["group_top1_accuracy"] = {
            key: {"accuracy": good / total, "samples": total}
            for key, (good, total) in sorted(group_counts.items()) if total
        }
    return avg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a BC checkpoint on val/test.")
    ap.add_argument("--checkpoint", required=True, help="checkpoint file or directory (best.pt/latest.pt/epoch_N.pt/model.pt)")
    ap.add_argument("--dataset-dir", default=None,
                    help="dataset directory (default: checkpoint config dataset_dir)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--mode", default="sampled", choices=["full", "sampled"])
    ap.add_argument("--samples", type=int, default=100000, help="samples per split (mode=sampled)")
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--out", default=None, help="output json path (default: <checkpoint_dir>/eval.json)")
    ap.add_argument("--include-groups", action="store_true",
                    help="also compute per-decision/action/red-label accuracy (slower)")
    args = ap.parse_args(argv)

    device = _resolve_device(args.device)
    ckpt_path = _resolve_checkpoint(args.checkpoint)
    ckpt = load_training_checkpoint(ckpt_path, map_location="cpu")
    model = _load_model(ckpt, device)
    config = ckpt.get("config") or {}
    if config.get("feature_version", "feature-v1") == "feature-v2":
        encoder = ObservationEncoderV2()
    elif config.get("feature_version") == "feature-v3":
        encoder = ObservationEncoderV3()
    else:
        encoder = ObservationEncoder()

    dataset_dir = Path(args.dataset_dir or config.get("dataset_dir") or "data/processed/decision-v2")
    dataset_manifest_path = dataset_dir / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    max_samples = args.samples if args.mode == "sampled" else None
    result = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": _sha256(ckpt_path),
        "dataset_dir": str(dataset_dir),
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "mode": args.mode,
        "samples": args.samples if args.mode == "sampled" else None,
        "dataset_version": dataset_manifest.get("dataset_version"),
        "checkpoint_dataset_version": config.get("dataset_version"),
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
            encoder=encoder,
            schema_v2=config.get("feature_version") == "feature-v2",
            schema_v3=config.get("feature_version") == "feature-v3",
            include_groups=args.include_groups,
        )
        print(f"{split}: {json.dumps(result['splits'][split], ensure_ascii=False)}")

    out = Path(args.out) if args.out else ckpt_path.parent / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"eval result -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
