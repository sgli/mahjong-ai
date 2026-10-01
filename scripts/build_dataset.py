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

from mahjong.dataset import DatasetBuilder, git_commit  # noqa: E402

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
    return sorted(data_dir.glob("*.mjai.json"))


def parse_ratios(text: str) -> tuple[float, float, float]:
    parts = [float(x) for x in text.split(",")]
    if len(parts) != 3:
        raise SystemExit("--split must be three comma-separated ratios, e.g. 0.8,0.1,0.1")
    return (parts[0], parts[1], parts[2])


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

    builder = DatasetBuilder(
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        ratios=ratios,
        seed=args.seed,
        dataset_version=args.dataset_version or "decision-v1",
        source_dir=data_dir,
        limit=args.limit,
        git_commit_hash=git_commit(_HERE.parent),
    )

    started = time.perf_counter()
    manifest = builder.build(files)
    elapsed = time.perf_counter() - started

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
