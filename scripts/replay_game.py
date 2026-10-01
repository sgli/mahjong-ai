#!/usr/bin/env python
"""Replay real ``.mjai.json`` files and report state consistency.

Streams one game file at a time, rebuilds the full ``ReplayState`` with the
Replay Engine and reports:

- how many files replayed cleanly;
- hard replay errors (bad sequences, e.g. a discarded tile not in hand);
- non-fatal inconsistencies (settled scores / kyotaku that do not match the
  next ``start_kyoku``).

Usage::

    python scripts/replay_game.py --data-dir "F:/Mahjong AI/mahjong DB/tenhou-houou-2026"
    python scripts/replay_game.py --limit 20   # quick smoke run
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

from mahjong.parser import MjaiParseError, MjaiParser  # noqa: E402
from mahjong.replay import ReplayEngine, ReplayInconsistency  # noqa: E402

log = logging.getLogger("replay_game")

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
    ap = argparse.ArgumentParser(description="Replay .mjai.json files and check consistency.")
    ap.add_argument("--data-dir", help="directory (or single file) of .mjai.json files")
    ap.add_argument("--config", help="path to a YAML config (default: configs/data.yaml)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N files (0 = all)")
    ap.add_argument("--max-issues", type=int, default=20, help="max issue examples to print")
    ap.add_argument("--show", action="store_true", help="print the final state of the first file")
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

    files_processed = 0
    files_clean = 0
    files_hard_error = 0
    files_inconsistent = 0
    kyoku_total = 0
    state_total = 0

    hard_error_examples: list[str] = []
    inconsistency_examples: list[str] = []
    shown_sample = False

    started = time.perf_counter()
    for idx, path in enumerate(files, 1):
        game_id = path.stem
        events = []
        parse_error = None
        try:
            for line_no, line in enumerate(Path(path).open("r", encoding="utf-8"), 1):
                try:
                    events.append(parser.parse_line(line, line_no))
                except MjaiParseError as exc:
                    parse_error = f"{exc.line_no}: {exc}"
                    break
        except OSError as exc:
            parse_error = str(exc)

        if parse_error is not None:
            files_hard_error += 1
            _append_example(hard_error_examples, f"{path.name}: {parse_error}", args.max_issues)
            continue

        engine = ReplayEngine(game_id=game_id)
        try:
            last_state = None
            for state in engine.replay(events):
                state_total += 1
                last_state = state
            kyoku_total += sum(1 for e in events if e.type.value == "start_kyoku")
            if args.show and not shown_sample:
                print(f"== sample final state ({path.name}) ==")
                print(last_state)
                shown_sample = True
        except ReplayInconsistency as exc:
            files_hard_error += 1
            _append_example(
                hard_error_examples,
                f"{path.name}: [{exc.event_type} @step {exc.step}] {exc}",
                args.max_issues,
            )
            continue

        if engine.inconsistencies:
            files_inconsistent += 1
            for msg in engine.inconsistencies:
                _append_example(inconsistency_examples, f"{path.name}: {msg}", args.max_issues)
        else:
            files_clean += 1

        files_processed += 1
        if not args.quiet and (idx % 1000 == 0 or idx == len(files)):
            elapsed = time.perf_counter() - started
            log.info(
                "processed %d/%d files (%.1f files/s, %.0f states/s)",
                idx,
                len(files),
                idx / elapsed if elapsed else 0,
                state_total / elapsed if elapsed else 0,
            )

    elapsed = time.perf_counter() - started

    print("== replay summary ==")
    print(f"files requested:      {len(files)}")
    print(f"files processed:      {files_processed}")
    print(f"files clean:          {files_clean}")
    print(f"files hard errors:    {files_hard_error}")
    print(f"files inconsistent:   {files_inconsistent}")
    print(f"kyokus replayed:      {kyoku_total}")
    print(f"states produced:      {state_total}")
    print(f"elapsed:              {elapsed:.1f}s")

    if hard_error_examples:
        print(f"\nhard error examples (first {len(hard_error_examples)}):")
        for example in hard_error_examples:
            print(f"  {example}")
    if inconsistency_examples:
        print(f"\ninconsistency examples (first {len(inconsistency_examples)}):")
        for example in inconsistency_examples:
            print(f"  {example}")

    # Non-zero exit when anything was not replayed cleanly, so CI can gate on it.
    return 0 if (files_hard_error == 0 and files_inconsistent == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
