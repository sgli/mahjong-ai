"""Freeze the legacy game split from the exact raw-file universe.

The old manifest gives the seed/ratios/counts, while the original split.py
assigns a sorted set of game IDs.  This script additionally checks the old
processed-game ledger and selected real Parquet shards before writing a map.
"""
from __future__ import annotations

import argparse
from collections import Counter
import filecmp
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyarrow as pa
import pyarrow.parquet as pq

from mahjong.dataset.split import SPLIT_NAMES, assign_splits


def raw_universe(raw_root: Path):
    """Return canonical IDs/paths and known duplicate paths."""
    paths = {}
    duplicates = []
    counts = Counter()
    for folder in sorted(raw_root.iterdir()):
        if not folder.is_dir():
            continue
        with os.scandir(folder) as stream:
            for item in stream:
                if item.is_file() and item.name.endswith(".mjai.json"):
                    game_id = item.name[:-5]  # Path.stem preserves the .mjai suffix
                    relpath = f"{folder.name}/{item.name}"
                    if game_id in paths:
                        pair = (paths[game_id], relpath)
                        if set(p.split("/")[0] for p in pair) != {
                            "majsoul-throne-2025", "tenhou-houou-2024"
                        }:
                            raise ValueError(f"unrecognized duplicate ID source: {pair}")
                        duplicates.append(pair)
                        if folder.name == "tenhou-houou-2024":
                            paths[game_id] = relpath
                    else:
                        paths[game_id] = relpath
                    counts[folder.name] += 1
    return sorted(paths.items()), duplicates, dict(sorted(counts.items()))


def freeze(raw_root: Path, legacy_dir: Path, out_dir: Path, sample_shards: int = 9):
    manifest_path = legacy_dir / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = old["config"]
    ratios = tuple(config["split_ratios"][s] for s in SPLIT_NAMES)
    entries, duplicates, source_counts = raw_universe(raw_root)
    if len(entries) + len(duplicates) != old["counts"]["files_requested"]:
        raise ValueError("raw file universe count differs from old manifest")
    if len(duplicates) != 184424:
        raise ValueError("duplicate source count changed; review before freezing")
    duplicate_bytes = 0
    for idx, (first, second) in enumerate(duplicates, 1):
        first_path, second_path = raw_root / first, raw_root / second
        if not filecmp.cmp(first_path, second_path, shallow=False):
            raise ValueError(f"same game_id has different raw bodies: {first}, {second}")
        duplicate_bytes += first_path.stat().st_size
        if idx % 25000 == 0:
            print(f"verified {idx}/{len(duplicates)} duplicate raw pairs", flush=True)
    assignment = assign_splits((gid for gid, _ in entries), ratios, config["seed"])

    processed_lines = [gid for path in sorted(legacy_dir.glob("_processed_[0-9]*.txt"))
                       for gid in path.read_text(encoding="utf-8").splitlines() if gid]
    processed = set(processed_lines)
    if not processed.issubset(assignment):
        raise ValueError("old processed ledger contains game IDs absent from raw source")
    if len(processed_lines) != old["counts"]["files_processed"]:
        raise ValueError("old processed ledger count differs from manifest")
    if processed != set((legacy_dir / "_processed_games.txt").read_text(encoding="utf-8").splitlines()):
        raise ValueError("combined old ledger differs from worker ledgers")
    processed_counts = Counter(assignment[gid] for gid in processed_lines)
    for split in SPLIT_NAMES:
        if processed_counts[split] != old["counts"]["splits"][split]["games"]:
            raise ValueError(f"old processed {split} count differs from manifest")

    checked_rows = checked_shards = 0
    for split in SPLIT_NAMES:
        shards = sorted((legacy_dir / split).glob("*.parquet"))
        if not shards:
            raise ValueError(f"missing old Parquet split: {split}")
        indexes = {round(i * (len(shards) - 1) / max(sample_shards - 1, 1))
                   for i in range(sample_shards)}
        for idx in sorted(indexes):
            path = shards[idx]
            for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=["game_id"]):
                for gid in batch.column(0).to_pylist():
                    if assignment.get(gid) != split:
                        raise ValueError(f"old Parquet split mismatch: {path}, {gid}")
                    checked_rows += 1
            checked_shards += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    map_path = out_dir / "game_split_map.parquet"
    if map_path.exists():
        raise FileExistsError(f"refusing to replace frozen split map: {map_path}")
    schema = pa.schema([("game_id", pa.string()), ("raw_relpath", pa.string()), ("split", pa.string())])
    writer = pq.ParquetWriter(map_path.with_suffix(".parquet.tmp"), schema, compression="zstd")
    digest = hashlib.sha256()
    split_counts = Counter()
    try:
        for start in range(0, len(entries), 50000):
            part = entries[start:start + 50000]
            for gid, relpath in part:
                digest.update(f"{gid}\t{relpath}\t{assignment[gid]}\n".encode())
                split_counts[assignment[gid]] += 1
            writer.write_table(pa.Table.from_pydict({
                "game_id": [gid for gid, _ in part],
                "raw_relpath": [rel for _, rel in part],
                "split": [assignment[gid] for gid, _ in part],
            }, schema=schema))
    finally:
        writer.close()
    map_path.with_suffix(".parquet.tmp").replace(map_path)
    frozen = {
        "raw_root": str(raw_root.resolve()),
        "legacy_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "legacy_git_commit": old["git_commit"],
        "split_seed": config["seed"], "split_ratios": dict(zip(SPLIT_NAMES, ratios)),
        "split_algorithm": "src/mahjong/dataset/split.py:assign_splits sorted IDs + Python random.Random(seed).shuffle",
        "mapping_sha256": digest.hexdigest(),
        "mapping_parquet_sha256": hashlib.file_digest(map_path.open("rb"), "sha256").hexdigest(),
        "total_raw_files": len(entries) + len(duplicates), "unique_game_ids": len(entries),
        "duplicate_file_pairs_verified_byte_equal": len(duplicates),
        "duplicate_source_pair": ["majsoul-throne-2025", "tenhou-houou-2024"],
        "duplicate_source_bytes": duplicate_bytes,
        "source_counts": source_counts,
        "assigned_games": dict(split_counts),
        "old_processed_files": len(processed_lines),
        "old_processed_unique_games": len(processed),
        "old_failed_files": len(entries) + len(duplicates) - len(processed_lines),
        "old_processed_by_split": dict(processed_counts),
        "old_parquet_shards_checked": checked_shards,
        "old_parquet_rows_checked": checked_rows,
    }
    (out_dir / "split_freeze.json").write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")
    return frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=Path("F:/Mahjong AI/mahjong DB"))
    ap.add_argument("--legacy-dir", type=Path, default=ROOT / "data/processed/decision-v2")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/phase10_3_9")
    ap.add_argument("--sample-shards-per-split", type=int, default=9)
    args = ap.parse_args()
    print(json.dumps(freeze(args.raw_root, args.legacy_dir, args.out, args.sample_shards_per_split),
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
