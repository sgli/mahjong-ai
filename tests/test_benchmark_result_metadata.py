"""Benchmark result.json metadata (Phase 10.3 诊断 B) roundtrip + defaults."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import benchmark_parallel as bp  # noqa: E402


def test_defaults_opponent_mode_temperature():
    args = bp._build_parser().parse_args([])
    assert args.opponent_mode == "greedy"
    assert args.opponent_temperature == 1.0
    assert args.candidate_mode == "greedy"
    assert args.candidate_temperature == 1.0
    assert args.opponents == "bc_v3"


def test_bc_v21_directory_resolves_to_best_checkpoint():
    directory = Path("experiments/bc_v2.1/checkpoints")
    if not (directory / "best.pt").is_file():
        return
    assert Path(bp._effective_checkpoint(str(directory))) == (directory / "best.pt").resolve()
    assert len(bp._sha256(str(directory))) == 64


def test_metadata_roundtrip():
    report = {
        "candidate": "experiments/ppo_fix/c/latest.pt",
        "candidate_mode": "greedy",
        "candidate_temperature": 0.5,
        "opponent_mode": "sampling",
        "opponent_temperature": 1.0,
        "runner_version": "abc123",
        "created_at": "2025-01-01T00:00:00+00:00",
        "matchups": {"rule": {"opponent_mode": "sampling", "opponent_temperature": 1.0, "candidate_metrics": {}, "opponents_metrics": {}}},
    }
    s = json.dumps(report)
    back = json.loads(s)
    assert back["candidate_mode"] == "greedy"
    assert back["candidate_temperature"] == 0.5
    assert back["opponent_mode"] == "sampling"
    assert back["opponent_temperature"] == 1.0
    assert back["runner_version"] == "abc123"
    assert back["created_at"] == "2025-01-01T00:00:00+00:00"


def test_old_format_readable_with_none_defaults():
    old = {"candidate": "x", "seed": 42, "matchups": {}}
    back = json.loads(json.dumps(old))
    assert back.get("candidate_mode") is None
    assert back.get("candidate_temperature") is None
    assert back.get("opponent_mode") is None
    assert back.get("opponent_temperature") is None
