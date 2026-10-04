#!/usr/bin/env python
"""BC retention evaluation (Phase 10.3 诊断 §2).

Fixed game-level subset → compare BC vs PPO checkpoints on the SAME records:
agreement / KL(BC||PPO) / KL(PPO||BC) / BC entropy / PPO entropy / P(BC action|PPO).

Usage::

    python scripts/eval_bc_retention.py --num-shards 5 --out experiments/bc_retention/retention.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from mahjong.evaluation import load_policy, retention_metrics  # noqa: E402
from mahjong.training import ParquetBatchDataset, collate_record_batches  # noqa: E402


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_HERE.parent, check=True)
        return out.stdout.strip()[:12]
    except Exception:
        return None


def _resolve_device(spec: str) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if spec == "auto" else torch.device(spec)


def _load_model(checkpoint: str, device: torch.device):
    _, model = load_policy(checkpoint, device)
    model.eval()
    return model


def _aggregate(acc: dict, m: dict, n: int):
    for k, v in m.items():
        acc[k] = acc.get(k, 0.0) + float(v.item()) * n
    acc.setdefault("samples", 0)
    acc["samples"] += n


def evaluate(bc_model, ppo_model, shards, *, batch_size, device, num_workers, max_samples):
    dl = torch.utils.data.DataLoader(
        ParquetBatchDataset(shards, batch_size),
        batch_size=1,
        collate_fn=collate_record_batches,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers else None,
        pin_memory=device.type == "cuda",
    )
    acc: dict[str, float] = {"samples": 0}
    with torch.no_grad():
        for features, action_ids, masks in dl:
            features = features.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            bc_out = bc_model(features, masks)
            ppo_out = ppo_model(features, masks)
            m = retention_metrics(bc_out.logits, ppo_out.logits, masks)
            _aggregate(acc, m, features.shape[0])
            if acc["samples"] >= max_samples:
                break
    n = acc["samples"]
    return {k: (v / n if k != "samples" else v) for k, v in acc.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BC retention evaluation.")
    ap.add_argument("--dataset-dir", default="data/processed/decision-v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-shards", type=int, default=5, help="fixed first-N shards (deterministic subset)")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=1_000_000)
    ap.add_argument("--bc-checkpoint", default="experiments/bc_v2.1/checkpoints/best.pt")
    ap.add_argument(
        "--ppo-checkpoints",
        default="experiments/ppo_v2/milestones/ppo-20k/latest.pt,"
        "experiments/ppo_v2/milestones/ppo-50k/latest.pt,"
        "experiments/ppo_v2/milestones/ppo-100k/latest.pt",
    )
    ap.add_argument("--out", default="experiments/bc_retention/retention.json")
    args = ap.parse_args(argv)

    device = _resolve_device(args.device)
    split_dir = Path(args.dataset_dir) / args.split
    shards = sorted(split_dir.glob("*.parquet"))[: args.num_shards]
    if not shards:
        raise SystemExit(f"no parquet under {split_dir}")

    bc_model = _load_model(args.bc_checkpoint, device)
    ppo_paths = [p.strip() for p in args.ppo_checkpoints.split(",") if p.strip()]

    manifest = {
        "selection_rule": f"first {args.num_shards} shards of split '{args.split}' (sorted lexicographically), deterministic",
        "shards": [s.name for s in shards],
        "num_shards": len(shards),
        "max_samples": args.max_samples,
        "git_commit": _git_commit(),
    }

    rows = []
    for cp in ppo_paths:
        ppo_model = _load_model(cp, device)
        r = evaluate(bc_model, ppo_model, shards, batch_size=args.batch_size, device=device, num_workers=args.num_workers, max_samples=args.max_samples)
        rows.append({"checkpoint": cp, **{k: round(v, 6) for k, v in r.items()}})
        print(json.dumps(rows[-1], ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "subset.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    out.write_text(json.dumps({"manifest": manifest, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
