#!/usr/bin/env python
"""Build a versioned, chunked, game-split Parquet decision dataset.

Usage::

    python scripts/build_dataset.py --data-dir "F:/Mahjong AI/mahjong DB/tenhou-houou-2026" \
        --output data/processed/decision-v1 --limit 20
    python scripts/build_dataset.py --limit 200 --chunk-size 50000 --split 0.8,0.1,0.1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml  # noqa: E402

from mahjong.dataset import (  # noqa: E402
    FEATURE_VERSION,
    DatasetBuilder,
    assign_splits,
    build_manifest,
    git_commit,
    write_manifest,
)
from mahjong.decision.action import ACTION_SCHEMA_VERSION  # noqa: E402

log = logging.getLogger("build_dataset")

_DEFAULT_CONFIG = _HERE.parent / "configs" / "data.yaml"


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir:
        return Path(args.data_dir)
    env = os.environ.get("MAHJONG_DATA_DIR")
    if env:
        return Path(env)
    config_path = Path(args.config) if args.config else _DEFAULT_CONFIG
    cfg = _load_config(config_path)
    raw = (cfg.get("data") or {}).get("raw_dir")
    if raw:
        return Path(raw)
    raise SystemExit(
        "no data directory specified: use --data-dir, set MAHJONG_DATA_DIR, "
        "or edit configs/data.yaml"
    )


def iter_mjai_files(data_dir: Path) -> list[Path]:
    if data_dir.is_file():
        return [data_dir]
    # 支持 data_dir 下的多级子目录（如 tenhou-houou-* / majsoul-throne-*）
    return sorted(data_dir.rglob("*.mjai.json"))


def _process_chunk(task: tuple) -> dict:
    """Worker: process one chunk of ``(path_str, split)`` and write per-worker shards.

    Runs in a child process (ProcessPoolExecutor spawn), so all mahjong imports
    are done locally.  Returns per-worker stats for the master to merge.
    """
    files, worker_id, output_dir, chunk_size = task

    from mahjong.dataset.schema import parquet_schema, sample_to_row
    from mahjong.decision.extractor import DecisionExtractor
    from mahjong.parser import MjaiParser
    from mahjong.replay import ReplayEngine
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(output_dir)
    buffers: dict[str, list] = {"train": [], "validation": [], "test": []}
    seq = 0
    processed: list[str] = []
    errors: list[str] = []
    stats = {
        "files": 0,
        "games": 0,
        "rounds": 0,
        "samples": 0,
        "samples_skipped": 0,
        "files_error": 0,
        "splits": {s: {"games": 0, "samples": 0, "files": 0} for s in ("train", "validation", "test")},
    }

    def flush(split: str):
        nonlocal seq
        rows = buffers[split]
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=parquet_schema())
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, d / f"part-{worker_id:02d}-{seq:05d}.parquet")
        stats["splits"][split]["files"] += 1
        seq += 1
        buffers[split].clear()

    parser = MjaiParser()
    for path_str, split in files:
        path = Path(path_str)
        game_id = path.stem
        try:
            events = []
            for line_no, line in enumerate(path.open("r", encoding="utf-8"), 1):
                events.append(parser.parse_line(line, line_no))
        except Exception as exc:  # parse error
            stats["files_error"] += 1
            errors.append(f"{path.name}: parse: {exc}")
            continue

        try:
            states = list(ReplayEngine(game_id=game_id).replay(events))
        except Exception as exc:
            stats["files_error"] += 1
            errors.append(f"{path.name}: replay: {exc}")
            continue

        try:
            samples = list(DecisionExtractor(game_id=game_id).extract(states, events))
        except Exception as exc:
            stats["files_error"] += 1
            errors.append(f"{path.name}: extract: {type(exc).__name__}: {exc}")
            continue

        added = 0
        round_ids = set()
        for s in samples:
            if s.action not in s.legal_actions:
                stats["samples_skipped"] += 1
                continue
            buffers[split].append(sample_to_row(s))
            added += 1
            round_ids.add(s.round_id)
            # 周期刷盘：单 split buffer 达到 chunk_size 立即 flush，避免 OOM
            if len(buffers[split]) >= chunk_size:
                flush(split)

        if added > 0:
            stats["games"] += 1
            stats["samples"] += added
            stats["rounds"] += len(round_ids)
            stats["splits"][split]["games"] += 1
            stats["splits"][split]["samples"] += added
        stats["files"] += 1
        processed.append(game_id)

    for split in ("train", "validation", "test"):
        flush(split)

    # 续传状态：每个 worker 写自己的已完成 game 列表
    (out / f"_processed_{worker_id}.txt").write_text(
        "\n".join(processed) + ("\n" if processed else ""), encoding="utf-8"
    )
    stats["errors"] = errors
    return stats


def parse_ratios(text: str) -> tuple[float, float, float]:
    parts = [float(x) for x in text.split(",")]
    if len(parts) != 3:
        raise SystemExit("--split must be three comma-separated ratios, e.g. 0.8,0.1,0.1")
    return (parts[0], parts[1], parts[2])


def _parallel_build(
    *,
    files: list[Path],
    output_dir: Path,
    ratios: tuple[float, float, float],
    seed: int,
    dataset_version: str,
    num_workers: int,
    chunk_size: int,
    resume: bool,
    processed_games: set[str],
) -> tuple[dict, float, list[str]]:
    """Multi-process build with deterministic split + resume."""
    import concurrent.futures

    output_dir.mkdir(parents=True, exist_ok=True)
    game_ids = [p.stem for p in files]
    assignment = assign_splits(game_ids, ratios, seed)  # 主进程对全部 game_id 先切分

    # 续传：剔除已处理 game
    todo = [(str(p), assignment[p.stem]) for p in files if p.stem not in processed_games]

    # 分块：round-robin 到各 worker
    chunks: list[list[tuple[str, str]]] = [[] for _ in range(num_workers)]
    for i, item in enumerate(todo):
        chunks[i % num_workers].append(item)

    started = time.perf_counter()
    all_stats: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(_process_chunk, (chunk, wid, str(output_dir), chunk_size))
            for wid, chunk in enumerate(chunks)
        ]
        for fut in concurrent.futures.as_completed(futures):
            all_stats.append(fut.result())
    elapsed = time.perf_counter() - started

    # 合并统计
    merged = {
        "files_requested": len(files),
        "files_processed": 0,
        "files_error": 0,
        "games": 0,
        "rounds": 0,
        "samples": 0,
        "samples_skipped": 0,
        "splits": {s: {"games": 0, "samples": 0, "files": 0} for s in ("train", "validation", "test")},
    }
    errors: list[str] = []
    for st in all_stats:
        merged["files_processed"] += st["files"]
        merged["files_error"] += st["files_error"]
        merged["games"] += st["games"]
        merged["rounds"] += st["rounds"]
        merged["samples"] += st["samples"]
        merged["samples_skipped"] += st["samples_skipped"]
        for s in ("train", "validation", "test"):
            merged["splits"][s]["games"] += st["splits"][s]["games"]
            merged["splits"][s]["samples"] += st["splits"][s]["samples"]
            merged["splits"][s]["files"] += st["splits"][s]["files"]
        errors.extend(st.get("errors", []))

    # 续传状态：合并各 worker 的已完成 game 写入统一状态文件
    all_processed = sorted(processed_games)
    for f in sorted(output_dir.glob("_processed_*.txt")):
        all_processed.extend(ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
    state = output_dir / "_processed_games.txt"
    state.write_text("\n".join(sorted(set(all_processed))) + "\n", encoding="utf-8")

    manifest = build_manifest(
        dataset_version=dataset_version,
        action_schema_version=ACTION_SCHEMA_VERSION,
        feature_version=FEATURE_VERSION,
        config={
            "chunk_size": 0,
            "split_ratios": {s: ratios[i] for i, s in enumerate(("train", "validation", "test"))},
            "seed": seed,
            "source_dir": None,
            "limit": 0,
            "num_workers": num_workers,
        },
        counts=merged,
        git_commit_hash=git_commit(_HERE.parent),
    )
    write_manifest(output_dir / "manifest.json", manifest)
    return manifest, elapsed, errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a Parquet decision dataset.")
    ap.add_argument("--data-dir", help="directory (or single file) of .mjai.json files")
    ap.add_argument("--config", help="path to a YAML config (default: configs/data.yaml)")
    ap.add_argument("--output", help="output dataset directory (default: data/processed/decision-v1)")
    ap.add_argument("--chunk-size", type=int, default=100_000, help="rows per Parquet chunk")
    ap.add_argument("--split", default="0.8,0.1,0.1", help="train,validation,test ratios")
    ap.add_argument("--seed", type=int, default=0, help="split seed")
    ap.add_argument("--dataset-version", default=None, help="dataset version (default: decision-v1)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N files (0 = all)")
    ap.add_argument("--resume", action="store_true", help="skip game ids already listed in the output state file")
    ap.add_argument("--num-workers", type=int, default=1, help="parallel worker processes (default: 1)")
    ap.add_argument("--progress-interval", type=int, default=1000, help="log progress every N files (0 = disabled)")
    ap.add_argument("--quiet", action="store_true", help="suppress progress logs")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    data_dir = resolve_data_dir(args)
    files = iter_mjai_files(data_dir)
    if args.limit:
        files = files[: args.limit]
    if not files:
        log.error("no .mjai.json files found under %s", data_dir)
        return 1

    output_dir = Path(args.output) if args.output else Path("data/processed/decision-v1")
    ratios = parse_ratios(args.split)
    dataset_version = args.dataset_version or "decision-v1"

    # 断点续传：记录已完成 game_id（每 run 结束后追加写回）
    state_file = output_dir / "_processed_games.txt"
    processed_games: set[str] = set()
    if args.resume and state_file.exists():
        processed_games = {ln.strip() for ln in state_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
        log.info("resume: %d game ids already processed", len(processed_games))

    if args.num_workers > 1:
        manifest, elapsed, errors = _parallel_build(
            files=files,
            output_dir=output_dir,
            ratios=ratios,
            seed=args.seed,
            dataset_version=dataset_version,
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
            resume=args.resume,
            processed_games=processed_games,
        )
        # 打印与串行一致的汇总
        counts = manifest["counts"]
        print("== dataset build summary (parallel) ==")
        print(f"output:                 {output_dir}")
        print(f"dataset version:        {manifest['dataset_version']}")
        print(f"feature version:        {manifest['feature_version']}")
        print(f"git commit:             {manifest['git_commit']}")
        print(f"files requested:        {counts['files_requested']}")
        print(f"files processed:        {counts['files_processed']}")
        print(f"files error:            {counts['files_error']}")
        print(f"games:                  {counts['games']}")
        print(f"rounds:                 {counts['rounds']}")
        print(f"samples:                {counts['samples']}")
        print(f"samples skipped:        {counts['samples_skipped']}")
        print(f"elapsed:                {elapsed:.1f}s")
        print("per split:")
        for name in ("train", "validation", "test"):
            s = counts["splits"][name]
            print(f"  {name:10s} games={s['games']:7d} samples={s['samples']:9d} files={s['files']}")
        if errors:
            print(f"\nerrors (first {min(len(errors), 20)}):")
            for message in errors[:20]:
                print(f"  {message}")
        clean = counts["files_error"] == 0 and counts["samples_skipped"] == 0
        return 0 if clean else 1

    builder = DatasetBuilder(
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        ratios=ratios,
        seed=args.seed,
        dataset_version=dataset_version,
        source_dir=data_dir,
        limit=args.limit,
        git_commit_hash=git_commit(_HERE.parent),
        progress_interval=args.progress_interval,
        processed_games=processed_games,
    )

    started = time.perf_counter()
    manifest = builder.build(files)
    elapsed = time.perf_counter() - started

    # 写回已完成 game 列表（用于下一次 --resume）
    all_processed = sorted(processed_games | builder.new_processed_games)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text("\n".join(all_processed) + ("\n" if all_processed else ""), encoding="utf-8")

    counts = manifest["counts"]
    print("== dataset build summary ==")
    print(f"output:                 {output_dir}")
    print(f"dataset version:        {manifest['dataset_version']}")
    print(f"action schema version:  {manifest['action_schema_version']}")
    print(f"feature version:        {manifest['feature_version']}")
    print(f"git commit:             {manifest['git_commit']}")
    print(f"files requested:        {counts['files_requested']}")
    print(f"files processed:        {counts['files_processed']}")
    print(f"files error:            {counts['files_error']}")
    print(f"games:                  {counts['games']}")
    print(f"samples:                {counts['samples']}")
    print(f"samples skipped:        {counts['samples_skipped']}")
    print(f"elapsed:                {elapsed:.1f}s")
    print()
    print("per split:")
    for name in ("train", "validation", "test"):
        s = counts["splits"][name]
        print(f"  {name:10s} games={s['games']:7d} samples={s['samples']:9d} files={s['files']}")

    if builder.errors:
        print(f"\nerrors (first {min(len(builder.errors), 20)}):")
        for message in builder.errors[:20]:
            print(f"  {message}")

    clean = counts["files_error"] == 0 and counts["samples_skipped"] == 0
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
