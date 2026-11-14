#!/usr/bin/env python
"""Validate v2 feature/label reconstruction from existing decision Parquet."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyarrow.parquet as pq  # noqa: E402
import torch
import yaml
from itertools import islice

from mahjong.training.fast_encode import encode_record_batch_fast_v2  # noqa: E402
from mahjong.dataset.legality import audit_legal_row
from mahjong.training.data_gate import source_hashes, dataset_identity


def _audit_files(files: list[Path], limit: int | None = None) -> dict:
    torch.set_num_threads(1)
    counts: Counter[str] = Counter()
    files_scanned = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512):
            if limit is not None:
                batch = batch.slice(0, limit - counts["rows"])
            features, labels, masks = encode_record_batch_fast_v2(batch)
            if not bool(features.isfinite().all()):
                raise ValueError(f"non-finite feature tensor in {path.name}")
            if not bool(masks.gather(1, labels.unsqueeze(1)).all()):
                raise ValueError(f"gold label absent from v2 legal mask: {path.name}")
            rows = batch.to_pylist()
            for row in rows:
                counts.update(audit_legal_row(row))
                counts["rows"] += 1
                counts[f"decision:{row['decision_kind']}"] += 1
                counts[f"action:{row['action_type']}"] += 1
                counts["red_label"] += int(
                    (row["action_tile"] or "").endswith("r")
                    or any(t.endswith("r") for t in (row["action_consumed"] or []))
                )
            if limit is not None and counts["rows"] >= limit:
                files_scanned += 1
                return {"files_scanned": files_scanned, **counts}
        files_scanned += 1
    return {"files_scanned": files_scanned, **counts}


def audit_split(split_dir: Path, limit: int | None, workers: int = 1) -> dict:
    files = sorted(split_dir.glob("*.parquet"))
    if limit is not None or workers == 1 or not files:
        return {"files_available": len(files), **_audit_files(files, limit)}
    counts = Counter()
    groups = [files[i:i + 8] for i in range(0, len(files), 8)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_audit_files, group) for group in groups]
        for future in as_completed(futures):
            counts.update(future.result())
    return {"files_available": len(files), **counts}


def main(argv=None) -> int:
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/processed/decision-v2")
    ap.add_argument("--max-rows-per-split", type=int, default=10_000,
                    help="0 scans all rows in every split")
    ap.add_argument("--out", default="experiments/phase10_3_7/data_audit.json")
    ap.add_argument("--raw-data-dir", type=Path, help="raw source override; defaults to configs/data.yaml")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel shard groups for complete audits; sampled scans remain ordered")
    args = ap.parse_args(argv)
    if args.workers < 1:
        ap.error("--workers must be positive")
    limit = args.max_rows_per_split or None
    result = {
        "source_sha256": source_hashes(),
        "dataset_identity": dataset_identity(args.dataset_dir),
        "dataset_dir": str(Path(args.dataset_dir)),
        "dataset_manifest": json.loads((Path(args.dataset_dir) / "manifest.json").read_text(encoding="utf-8")),
        "limit_per_split": limit,
        "feature_version": "feature-v2",
        "action_schema_version": "action-v2",
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        result["splits"][split] = audit_split(Path(args.dataset_dir) / split, limit, args.workers)
        print(split, json.dumps(result["splits"][split], ensure_ascii=False))
    raw_config = yaml.safe_load((ROOT / "configs/data.yaml").read_text(encoding="utf-8"))
    raw_dir = args.raw_data_dir or Path(raw_config["data"]["raw_dir"])
    examples = list(islice((p for p in raw_dir.iterdir() if p.name.endswith(".mjai.json")), 3)) if raw_dir.is_dir() else []
    result["raw_source_dir"] = str(raw_dir)
    result["raw_source_available"] = bool(examples)
    result["raw_examples"] = [p.name for p in examples]
    result["raw_mjai_files_available"] = None  # inventory not fully counted
    result["passed"] = limit is None and all(
        r.get("rows", 0) > 0 and r.get("context_verified_rows", 0) == r["rows"]
        and r["files_scanned"] == r["files_available"] for r in result["splits"].values()
    )
    result["blocking_reason"] = None if result["passed"] else "sampled coverage or legacy rows without reconstructable replay legality context"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"audit -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
