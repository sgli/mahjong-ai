#!/usr/bin/env python
"""Extract decision samples from real ``.mjai.json`` data and verify coverage.

For every file: parse -> replay -> DecisionExtractor, then report:

- decision point / sample counts (by window kind and by action type);
- the number of human actions NOT in the generated legal actions (with
  examples) — the core correctness check for the rule engine.

Usage::

    python scripts/extract_decisions.py --limit 20      # smoke
    python scripts/extract_decisions.py                 # full dataset
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml  # noqa: E402

from mahjong.decision.extractor import DecisionExtractor  # noqa: E402
from mahjong.parser import MjaiParseError, MjaiParser  # noqa: E402
from mahjong.replay import ReplayEngine, ReplayInconsistency  # noqa: E402

log = logging.getLogger("extract_decisions")

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


def _append_example(examples: list[str], text: str, limit: int) -> None:
    if len(examples) < limit:
        examples.append(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract decision samples from .mjai.json files.")
    ap.add_argument("--data-dir", help="directory (or single file) of .mjai.json files")
    ap.add_argument("--config", help="path to a YAML config (default: configs/data.yaml)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N files (0 = all)")
    ap.add_argument("--max-issues", type=int, default=30, help="max issue examples to print")
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
    files_parse_error = 0
    files_replay_error = 0
    files_extract_error = 0
    files_with_mismatch = 0
    sample_total = 0
    kind_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    coverage_fail = 0
    unresolved_total = 0
    mismatch_examples: list[str] = []
    unresolved_examples: list[str] = []
    replay_error_examples: list[str] = []
    extract_error_examples: list[str] = []

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
            files_parse_error += 1
            continue

        try:
            states = list(ReplayEngine(game_id=game_id).replay(events))
        except ReplayInconsistency as exc:
            files_replay_error += 1
            _append_example(
                replay_error_examples,
                f"{path.name}: [{exc.event_type} @step {exc.step}] {exc}",
                args.max_issues,
            )
            continue

        extractor = DecisionExtractor(game_id=game_id)

        try:
            file_fail = 0
            for sample in extractor.extract(states, events):
                sample_total += 1
                kind_counts[sample.decision_point.kind] += 1
                action_counts[sample.action.type.value] += 1
                if sample.action not in sample.legal_actions:
                    file_fail += 1
                    coverage_fail += 1
                    _append_example(
                        mismatch_examples,
                        f"{path.name}:{sample.decision_point.step} seat={sample.player_id} "
                        f"action={sample.action!r} legal={[repr(a) for a in sample.legal_actions]}",
                        args.max_issues,
                    )
        except Exception as exc:  # rule/extractor bug must not abort the whole run
            files_extract_error += 1
            _append_example(
                extract_error_examples,
                f"{path.name}: {type(exc).__name__}: {exc}",
                args.max_issues,
            )
            continue

        for message in extractor.inconsistencies:
            # "not in legal actions" is already counted above; count the rest.
            if "not in legal actions" not in message:
                unresolved_total += 1
                _append_example(unresolved_examples, f"{path.name}: {message}", args.max_issues)

        files_processed += 1
        files_clean += 1 if (file_fail == 0 and len(extractor.inconsistencies) == 0) else 0
        files_with_mismatch += 1 if file_fail > 0 else 0

        if not args.quiet and (idx % 1000 == 0 or idx == len(files)):
            elapsed = time.perf_counter() - started
            log.info(
                "processed %d/%d files (%.1f files/s, %.0f samples/s)",
                idx,
                len(files),
                idx / elapsed if elapsed else 0,
                sample_total / elapsed if elapsed else 0,
            )

    elapsed = time.perf_counter() - started

    print("== decision extraction summary ==")
    print(f"files requested:        {len(files)}")
    print(f"files processed:        {files_processed}")
    print(f"files parse errors:     {files_parse_error}")
    print(f"files replay errors:    {files_replay_error}")
    print(f"files extract errors:   {files_extract_error}")
    print(f"files clean:            {files_clean}")
    print(f"files with mismatch:    {files_with_mismatch}")
    print(f"decision samples:       {sample_total}")
    print(f"  discard decisions:    {kind_counts.get('discard', 0)}")
    print(f"  response decisions:   {kind_counts.get('response', 0)}")
    print(f"coverage failures:      {coverage_fail}")
    print(f"unresolved actions:     {unresolved_total}")
    print(f"elapsed:                {elapsed:.1f}s")
    print()
    print("action type distribution:")
    for name, count in sorted(action_counts.items()):
        print(f"  {name:10s} {count}")

    if mismatch_examples:
        print(f"\ncoverage failure examples (first {len(mismatch_examples)}):")
        for example in mismatch_examples:
            print(f"  {example}")
    if unresolved_examples:
        print(f"\nunresolved action examples (first {len(unresolved_examples)}):")
        for example in unresolved_examples:
            print(f"  {example}")
    if replay_error_examples:
        print(f"\nreplay error examples (first {len(replay_error_examples)}):")
        for example in replay_error_examples:
            print(f"  {example}")
    if extract_error_examples:
        print(f"\nextract error examples (first {len(extract_error_examples)}):")
        for example in extract_error_examples:
            print(f"  {example}")

    clean = (
        coverage_fail == 0
        and unresolved_total == 0
        and files_parse_error == 0
        and files_replay_error == 0
        and files_extract_error == 0
    )
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
