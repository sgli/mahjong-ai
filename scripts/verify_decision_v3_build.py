"""Check a completed decision-v3 build before the expensive all-row audit."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyarrow.parquet as pq

from mahjong.training.data_gate import source_hashes


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(dataset_dir: Path, frozen_dir: Path, quarantine_path: Path) -> dict:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((dataset_dir / "build_plan.json").read_text(encoding="utf-8"))
    freeze = json.loads((frozen_dir / "split_freeze.json").read_text(encoding="utf-8"))
    if manifest["dataset_version"] != "decision-v3" or manifest["config"]["diagnostic_only"]:
        raise ValueError("not a completed formal decision-v3 dataset")
    if (manifest["config"]["split_map_sha256"] != freeze["mapping_sha256"]
            or plan["mapping_sha256"] != freeze["mapping_sha256"]
            or digest(frozen_dir / "game_split_map.parquet") != freeze["mapping_parquet_sha256"]):
        raise ValueError("frozen split map identity drift")
    if digest(quarantine_path) != manifest["config"]["quarantine_sha256"]:
        raise ValueError("quarantine identity drift")
    source = source_hashes()
    source["scripts/build_decision_v3.py"] = digest(ROOT / "scripts/build_decision_v3.py")
    if source != plan["source_sha256"]:
        raise ValueError("builder/converter source drift")
    source_digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    if source_digest != manifest["config"]["source_sha256"]:
        raise ValueError("manifest source digest drift")

    task_count = (plan["requested_unique_games"] + plan["games_per_task"] - 1) // plan["games_per_task"]
    marker_dir = dataset_dir / "_completed"
    markers = sorted(marker_dir.glob("task-*.json"))
    if len(markers) != task_count:
        raise ValueError(f"completed task count {len(markers)} != {task_count}")
    totals = Counter()
    quarantined = []
    raw_digests = []
    named_shards = set()
    for task_id, path in enumerate(markers):
        if path.name != f"task-{task_id:06d}.json":
            raise ValueError(f"missing/duplicate task marker at index {task_id}")
        marker = json.loads(path.read_text(encoding="utf-8"))
        if marker["task_id"] != task_id or marker["signature"] != plan["signature"]:
            raise ValueError(f"task marker identity drift: {task_id}")
        totals.update(marker["counts"])
        totals["assigned_games"] += marker["games"]
        quarantined.extend(marker.get("quarantined_ids", []))
        raw_digests.append(marker["raw_content_sha256"])
        for relative in marker["shards"]:
            if relative in named_shards:
                raise ValueError(f"duplicate shard reference: {relative}")
            named_shards.add(relative)
    if sorted(quarantined) != manifest["config"]["quarantined_game_ids"]:
        raise ValueError("quarantine marker IDs differ from manifest")
    if totals["assigned_games"] != plan["requested_unique_games"]:
        raise ValueError("assigned game count drift")
    if hashlib.sha256("".join(raw_digests).encode("ascii")).hexdigest() != manifest["config"]["raw_content_sha256_by_task_digest"]:
        raise ValueError("raw content task digest drift")

    observed_shards = set()
    split_summary = {}
    for split in ("train", "validation", "test"):
        files = sorted((dataset_dir / split).glob("*.parquet"))
        row_count = 0
        for path in files:
            observed_shards.add(path.relative_to(dataset_dir).as_posix())
            row_count += pq.ParquetFile(path).metadata.num_rows
        expected = manifest["counts"]["splits"][split]
        if (len(files) != expected["files"] or row_count != expected["samples"]
                or totals[f"rows:{split}"] != row_count
                or totals[f"games:{split}"] != expected["games"]):
            raise ValueError(f"{split} manifest/marker/Parquet count disagreement")
        split_summary[split] = {"shards": len(files), "rows": row_count,
                                "games": expected["games"]}
    if observed_shards != named_shards:
        raise ValueError(f"orphan/missing shards: observed={len(observed_shards)} referenced={len(named_shards)}")
    if any(dataset_dir.rglob("*.tmp")) or (dataset_dir / "build_error.json").exists():
        raise ValueError("incomplete temporary shard or failure marker remains")
    if sum(x["rows"] for x in split_summary.values()) != manifest["counts"]["samples"]:
        raise ValueError("total row count drift")
    return {"passed": True, "tasks": task_count,
            "assigned_games": totals["assigned_games"],
            "quarantined_games": len(quarantined), "splits": split_summary,
            "manifest_sha256": digest(dataset_dir / "manifest.json"),
            "split_map_sha256": freeze["mapping_sha256"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=ROOT / "data/processed/decision-v3")
    ap.add_argument("--frozen-dir", type=Path, default=ROOT / "experiments/phase10_3_9")
    ap.add_argument("--quarantine", type=Path, default=ROOT / "experiments/phase10_3_9/quarantine.json")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/phase10_3_9/full_build_preflight.json")
    args = ap.parse_args()
    result = verify(args.dataset_dir, args.frozen_dir, args.quarantine)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
