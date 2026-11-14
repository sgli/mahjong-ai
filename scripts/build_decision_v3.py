"""Resumable, deduplicated contextual decision-v3 rebuild from frozen splits.

Pilot is the default. ``--full`` requires the same frozen map and writes to a
separate output directory. Neither path mutates the old decision-v2 dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyarrow as pa
import pyarrow.parquet as pq

from mahjong.dataset.manifest import build_manifest, git_commit, write_manifest
from mahjong.dataset.schema import parquet_schema, sample_to_row
from mahjong.dataset.split import SPLIT_NAMES
from mahjong.decision.extractor import DecisionExtractor
from mahjong.features.action_space_v2 import ACTION_SCHEMA_VERSION as ACTION_SCHEMA_VERSION_V2
from mahjong.features.encoder_v2 import FEATURE_VERSION_V2
from mahjong.parser import MjaiParser
from mahjong.replay import ReplayEngine
from mahjong.training.data_gate import source_hashes


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def select_games(mapping_path: Path, processed: set[str], pilot_per_group: int):
    """Choose unique old-successful IDs; pilot spans source directory × split."""
    all_rows = []
    group_counts = Counter()
    seen = set()
    for batch in pq.ParquetFile(mapping_path).iter_batches(batch_size=20000):
        for gid, relpath, split in zip(*(batch.column(i).to_pylist() for i in range(3))):
            if gid not in processed:
                continue
            if gid in seen:
                raise ValueError(f"duplicate ID in frozen mapping: {gid}")
            seen.add(gid)
            if split not in SPLIT_NAMES:
                raise ValueError(f"invalid split in frozen map: {split}")
            group = (relpath.split("/")[0], split)
            if pilot_per_group and group_counts[group] >= pilot_per_group:
                continue
            group_counts[group] += 1
            all_rows.append((gid, relpath, split))
    if not pilot_per_group and seen != processed:
        raise ValueError("frozen map does not cover all old successful game IDs")
    if pilot_per_group and len({s for _, _, s in all_rows}) != 3:
        raise ValueError("pilot misses a data split")
    return all_rows


def _build_task(task):
    task_id, games, raw_root, output, max_rows, signature, quarantine = task
    raw_root, output = Path(raw_root), Path(output)
    marker = output / "_completed" / f"task-{task_id:06d}.json"
    if marker.exists():
        done = json.loads(marker.read_text(encoding="utf-8"))
        expected_quarantine = [gid for gid, _, _ in games if gid in quarantine]
        if (done["signature"] != signature or done["games"] != len(games)
                or done.get("quarantined_ids", []) != expected_quarantine):
            raise ValueError(f"completed task identity drift: {task_id}")
        if any(not (output / name).is_file() for name in done["shards"]):
            raise ValueError(f"completed task missing shard: {task_id}")
        return done

    parser = MjaiParser()
    buffers = {split: [] for split in SPLIT_NAMES}
    seq = Counter()
    counts = Counter()
    shards = []
    raw_digest = hashlib.sha256()
    quarantined_ids = []

    def flush(split):
        rows = buffers[split]
        if not rows:
            return
        relative = f"{split}/part-{task_id:06d}-{seq[split]:03d}.parquet"
        target = output / relative
        if target.exists() or target.with_suffix(".parquet.tmp").exists():
            raise FileExistsError(f"incomplete prior task output requires inspection: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(rows, schema=parquet_schema()), temporary,
                       compression="zstd")
        temporary.replace(target)
        shards.append(relative)
        seq[split] += 1
        counts[f"shards:{split}"] += 1
        buffers[split] = []

    for gid, relpath, split in games:
        if gid in quarantine:
            quarantined_ids.append(gid)
            counts["quarantined_games"] += 1
            continue
        path = raw_root / relpath
        if path.stem != gid:
            raise ValueError(f"source ID mismatch: {path}")
        events = []
        game_digest = hashlib.sha256()
        with path.open(encoding="utf-8", newline="") as handle:
            for line_no, line in enumerate(handle, 1):
                game_digest.update(line.encode("utf-8"))
                events.append(parser.parse_line(line, line_no))
        raw_digest.update(gid.encode("utf-8") + b"\t" + game_digest.hexdigest().encode("ascii") + b"\n")
        states = list(ReplayEngine(game_id=gid).replay(events))
        extractor = DecisionExtractor(game_id=gid)
        samples = list(extractor.extract(states, events))
        if extractor.inconsistencies:
            raise ValueError(f"{path}: {extractor.inconsistencies[:3]}")
        if not samples:
            raise ValueError(f"old successful game now has zero samples: {path}")
        for sample in samples:
            if sample.action not in sample.legal_actions:
                raise ValueError(f"gold action absent from contextual menu: {path}")
            buffers[split].append(sample_to_row(sample))
            counts[f"rows:{split}"] += 1
            counts[f"action:{sample.action.type.value}"] += 1
            if len(buffers[split]) >= max_rows:
                flush(split)
        counts[f"games:{split}"] += 1
    for split in SPLIT_NAMES:
        flush(split)
    done = {"task_id": task_id, "signature": signature, "games": len(games),
            "raw_content_sha256": raw_digest.hexdigest(),
            "quarantined_ids": quarantined_ids,
            "counts": dict(counts), "shards": shards}
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(done, sort_keys=True), encoding="utf-8")
    temporary.replace(marker)
    return done


def build(*, frozen_dir, legacy_dir, output, full=False, pilot_per_group=3,
          workers=4, games_per_task=100, rows_per_shard=50000,
          quarantine_file=None, reset_incomplete_tasks=False):
    frozen_dir, legacy_dir, output = map(Path, (frozen_dir, legacy_dir, output))
    freeze = json.loads((frozen_dir / "split_freeze.json").read_text(encoding="utf-8"))
    mapping_path = frozen_dir / "game_split_map.parquet"
    if _sha256(mapping_path) != freeze["mapping_parquet_sha256"]:
        raise ValueError("frozen split mapping changed")
    if _sha256(legacy_dir / "manifest.json") != freeze["legacy_manifest_sha256"]:
        raise ValueError("legacy manifest changed")
    source = source_hashes()
    source["scripts/build_decision_v3.py"] = _sha256(Path(__file__))
    source_digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    processed_path = legacy_dir / "_processed_games.txt"
    processed_sha256 = _sha256(processed_path)
    processed = set(processed_path.read_text(encoding="utf-8").splitlines())
    if len(processed) != freeze["old_processed_unique_games"]:
        raise ValueError("legacy processed-game ledger changed")
    quarantine_file = Path(quarantine_file) if quarantine_file else None
    quarantine_records = json.loads(quarantine_file.read_text(encoding="utf-8")) if quarantine_file else []
    if not isinstance(quarantine_records, list):
        raise ValueError("quarantine must be a JSON list")
    quarantine = {}
    for record in quarantine_records:
        gid = record["game_id"]
        if (gid in quarantine or gid not in processed or not record.get("reason_codes")
                or not record.get("raw_sha256") or not record.get("event_steps")):
            raise ValueError(f"invalid or duplicate quarantine entry: {gid}")
        quarantine[gid] = record
    games = select_games(mapping_path, processed, 0 if full else pilot_per_group)
    selected = {gid: rel for gid, rel, _ in games}
    for gid, record in quarantine.items():
        if gid in selected:
            if selected[gid] != record["raw_relpath"]:
                raise ValueError(f"quarantine path differs from frozen map: {gid}")
            if _sha256(Path(freeze["raw_root"]) / selected[gid]) != record["raw_sha256"]:
                raise ValueError(f"quarantined source content changed: {gid}")
    signature = hashlib.sha256(json.dumps({
        "mapping": freeze["mapping_sha256"], "source": source_digest,
        "processed": processed_sha256,
        "mode": "full" if full else "pilot", "games": len(games),
        "games_per_task": games_per_task, "rows_per_shard": rows_per_shard,
    }, sort_keys=True).encode()).hexdigest()
    plan = {"signature": signature, "mapping_sha256": freeze["mapping_sha256"],
            "old_processed_ledger_sha256": processed_sha256,
            "quarantine_file": str(quarantine_file.resolve()) if quarantine_file else None,
            "source_sha256": source, "mode": "full" if full else "pilot",
            "requested_unique_games": len(games), "workers": workers,
            "games_per_task": games_per_task, "rows_per_shard": rows_per_shard,
            "excluded_duplicate_files": freeze["duplicate_file_pairs_verified_byte_equal"],
            "excluded_old_failed_files": freeze["old_failed_files"]}
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "build_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("output directory contains a different build plan")
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    incomplete = []
    for split in SPLIT_NAMES:
        for shard in (output / split).glob("part-*.parquet*"):
            match = re.fullmatch(r"part-(\d{6})-\d{3}\.parquet(?:\.tmp)?", shard.name)
            if match and not (output / "_completed" / f"task-{match.group(1)}.json").exists():
                incomplete.append(shard)
    if incomplete:
        if not reset_incomplete_tasks:
            raise ValueError(f"{len(incomplete)} incomplete task shards found; inspect and rerun with --reset-incomplete-tasks")
        if not output.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("refusing incomplete-task cleanup outside repository")
        for shard in incomplete:
            shard.unlink()
        (output / "incomplete_cleanup.json").write_text(json.dumps(
            {"removed_incomplete_shards": [str(p.relative_to(output)) for p in incomplete]}, indent=2),
            encoding="utf-8")
    tasks = [(i // games_per_task, games[i:i + games_per_task], freeze["raw_root"],
              str(output), rows_per_shard, signature, quarantine)
             for i in range(0, len(games), games_per_task)]
    totals = Counter()
    raw_content_by_task = {}
    completed = 0
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build_task, task): task[0] for task in tasks}
        for future in as_completed(futures):
            try:
                done = future.result()
            except Exception as exc:
                failure = {"task_id": futures[future], "error": repr(exc),
                           "completed_tasks": completed, "signature": signature}
                (output / "build_error.json").write_text(
                    json.dumps(failure, indent=2), encoding="utf-8")
                for pending in futures:
                    pending.cancel()
                raise
            totals.update(done["counts"])
            raw_content_by_task[done["task_id"]] = done["raw_content_sha256"]
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                print(f"tasks {completed}/{len(tasks)} rows {sum(totals[f'rows:{s}'] for s in SPLIT_NAMES)} "
                      f"elapsed {time.perf_counter() - started:.1f}s", flush=True)
    quarantined_ids = sorted({gid for marker in (output / "_completed").glob("task-*.json")
                              for gid in json.loads(marker.read_text(encoding="utf-8")).get("quarantined_ids", [])})
    if quarantined_ids != sorted(set(quarantine) & selected.keys()):
        raise ValueError("completed-task quarantine set differs from registered quarantine")
    counts = {"files_requested": len(games), "files_processed": len(games) - len(quarantined_ids),
              "files_error": len(quarantined_ids), "games": len(games) - len(quarantined_ids),
              "quarantined_games": len(quarantined_ids),
              "samples": sum(totals[f"rows:{s}"] for s in SPLIT_NAMES),
              "samples_skipped": 0,
              "splits": {s: {"games": totals[f"games:{s}"], "samples": totals[f"rows:{s}"],
                             "files": totals[f"shards:{s}"]} for s in SPLIT_NAMES}}
    raw_content_digest = hashlib.sha256("".join(
        raw_content_by_task[i] for i in range(len(tasks))).encode("ascii")).hexdigest()
    manifest = build_manifest(
        dataset_version="decision-v3" if full else "decision-v3-pilot",
        action_schema_version=ACTION_SCHEMA_VERSION_V2,
        feature_version=FEATURE_VERSION_V2,
        config={"split_map_sha256": freeze["mapping_sha256"],
                "legacy_manifest_sha256": freeze["legacy_manifest_sha256"],
                "old_processed_ledger_sha256": processed_sha256,
                "raw_content_sha256_by_task_digest": raw_content_digest,
                "quarantine_sha256": _sha256(quarantine_file) if quarantine_file else None,
                "quarantined_game_ids": quarantined_ids,
                "deduplicated_by_game_id": True, "old_successful_games_only": True,
                "diagnostic_only": not full, "source_sha256": source_digest},
        counts=counts, git_commit_hash=git_commit(ROOT))
    write_manifest(output / "manifest.json", manifest)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-dir", type=Path, default=ROOT / "experiments/phase10_3_9")
    ap.add_argument("--legacy-dir", type=Path, default=ROOT / "data/processed/decision-v2")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--pilot-per-source-split", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--games-per-task", type=int, default=100)
    ap.add_argument("--rows-per-shard", type=int, default=50000)
    ap.add_argument("--quarantine-file", type=Path,
                    default=ROOT / "experiments/phase10_3_9/quarantine.json")
    ap.add_argument("--reset-incomplete-tasks", action="store_true",
                    help="remove only unmarked partial shards from failed tasks after inspection")
    args = ap.parse_args()
    if min(args.workers, args.games_per_task, args.rows_per_shard) < 1:
        ap.error("workers/task/shard sizes must be positive")
    if args.full and args.pilot_per_source_split != 3:
        ap.error("--pilot-per-source-split does not apply to --full")
    out = args.output or ROOT / "data/processed" / ("decision-v3" if args.full else "decision-v3-pilot")
    print(json.dumps(build(frozen_dir=args.frozen_dir, legacy_dir=args.legacy_dir,
                           output=out, full=args.full,
                           pilot_per_group=args.pilot_per_source_split,
                           workers=args.workers, games_per_task=args.games_per_task,
                           rows_per_shard=args.rows_per_shard,
                           quarantine_file=args.quarantine_file,
                           reset_incomplete_tasks=args.reset_incomplete_tasks), indent=2), flush=True)


if __name__ == "__main__":
    main()
