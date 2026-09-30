#!/usr/bin/env python
"""Inspect ``.mjai.json`` data: parse, validate and report statistics.

Streams files one at a time so the dataset never has to fit in memory, then
prints the counts required by ``docs/DATA_SPEC.md`` section 12.

Usage::

    python scripts/inspect_mjai.py --data-dir "F:/Mahjong AI/mahjong DB/tenhou-houou-2026"
    python scripts/inspect_mjai.py --limit 100   # quick smoke run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Allow running as a plain script without installing the package.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml  # noqa: E402

from mahjong.parser import (  # noqa: E402
    GameValidator,
    MjaiParseError,
    MjaiParser,
    StatsCollector,
    validate_event,
)

log = logging.getLogger("inspect_mjai")

_DEFAULT_CONFIG = _HERE.parent / "configs" / "data.yaml"


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def resolve_data_dir(args: argparse.Namespace) -> Path:
    """Resolve the data directory: CLI > env var > config file."""
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


def _append_example(examples: list[str], text: str, limit: int) -> None:
    if len(examples) < limit:
        examples.append(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect .mjai.json data.")
    ap.add_argument("--data-dir", help="directory (or single file) of .mjai.json files")
    ap.add_argument("--config", help="path to a YAML config (default: configs/data.yaml)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N files (0 = all)")
    ap.add_argument("--max-issues", type=int, default=20, help="max issue examples to print")
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

    parser = MjaiParser()
    stats = StatsCollector()

    parse_error_count = 0
    parse_error_examples: list[str] = []
    event_issue_count = 0
    game_issue_count = 0
    issue_examples: list[str] = []

    started = time.perf_counter()
    for idx, path in enumerate(files, 1):
        validator = GameValidator()
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                try:
                    event = parser.parse_line(line, line_no)
                except MjaiParseError as exc:
                    parse_error_count += 1
                    _append_example(
                        parse_error_examples,
                        f"{path.name}:{exc.line_no}: {exc}",
                        args.max_issues,
                    )
                    continue

                validator.add(event, line_no)
                stats.add_event(event)
                for issue in validate_event(event):
                    event_issue_count += 1
                    _append_example(
                        issue_examples,
                        f"{path.name}:{issue.line_no}: [{issue.category}] {issue.message}",
                        args.max_issues,
                    )

        validator.finish()
        for issue in validator.issues:
            game_issue_count += 1
            _append_example(
                issue_examples,
                f"{path.name}:{issue.line_no}: [{issue.category}] {issue.message}",
                args.max_issues,
            )

        if not args.quiet and (idx % 1000 == 0 or idx == len(files)):
            elapsed = time.perf_counter() - started
            log.info(
                "processed %d/%d files (%.1f files/s, %.0f events/s)",
                idx,
                len(files),
                idx / elapsed if elapsed else 0,
                stats.events_total / elapsed if elapsed else 0,
            )

    stats.files = len(files)
    elapsed = time.perf_counter() - started

    print(stats.render())
    print()
    print("== validation ==")
    print(f"parse errors:  {parse_error_count}")
    print(f"event issues:  {event_issue_count}")
    print(f"game issues:   {game_issue_count}")
    print(f"total events:  {stats.events_total}")
    print(f"elapsed:       {elapsed:.1f}s")
    print(f"throughput:    {stats.events_total / elapsed:.0f} events/s" if elapsed else "n/a")

    if parse_error_examples:
        print(f"\nparse error examples (first {len(parse_error_examples)}):")
        for example in parse_error_examples:
            print(f"  {example}")
    if issue_examples:
        print(f"\nvalidation issue examples (first {len(issue_examples)}):")
        for example in issue_examples:
            print(f"  {example}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
