#!/usr/bin/env python
"""Classify dataset build errors (dataset_errors.jsonl) into dirty-replay vs engine-edge.

Usage::

    python scripts/classify_errors.py --errors data/processed/decision-v2/dataset_errors.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def classify(errors: list[dict]) -> dict:
    by_type = Counter(e["error_type"] for e in errors)
    by_event = Counter((e.get("event_type") or "none") for e in errors if e["error_type"] == "replay")
    # 按 error_type 抽样前若干条详情
    samples: dict[str, list[dict]] = {}
    for e in errors:
        samples.setdefault(e["error_type"], [])
        if len(samples[e["error_type"]]) < 5:
            samples[e["error_type"]].append(e)
    return {
        "total": len(errors),
        "by_error_type": dict(by_type),
        "replay_by_event_type": dict(by_event),
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Classify dataset build errors.")
    ap.add_argument("--errors", required=True, help="path to dataset_errors.jsonl")
    args = ap.parse_args(argv)

    path = Path(args.errors)
    if not path.exists():
        raise SystemExit(f"error manifest not found: {path}")

    errors = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = classify(errors)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
