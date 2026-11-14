#!/usr/bin/env python
"""Paired analysis for the preregistered BC-v3 matchups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mahjong.evaluation.validation import validate_game_rows


def _read(path: Path, matchup: str) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["matchups"][matchup]["raw_games"]
    validate_game_rows(raw, expected_seeds=range(320000, 321000))
    for key in ("candidate_mode", "opponent_mode"):
        if payload.get(key) != "greedy":
            raise ValueError(f"registration requires {key}=greedy")
    for key in ("candidate_temperature", "opponent_temperature"):
        if payload.get(key) != 1.0:
            raise ValueError(f"registration requires {key}=1.0")
    if not payload.get("environment_source_sha256"):
        raise ValueError("missing environment source identity")
    rows = {}
    for row in raw:
        if row["seat"] != (row["seed"] - 320000) % 4:
            raise ValueError("candidate seat rotation mismatch")
        ranks = row["final_ranks"]
        seat = int(row["seat"])
        rows[int(row["seed"])] = {
            "seat": seat,
            "candidate_rank": int(ranks[seat]),
            "opponent_mean_rank": sum(int(ranks[s]) for s in range(4) if s != seat) / 3.0,
            "illegal_actions": int(row["illegal_actions"]),
        }
    return rows


def _xorshift32(state: int) -> int:
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFFFFFF
    return state & 0xFFFFFFFF


def _quantile(values: list[float], q: float) -> float:
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def paired_analysis(control_path: Path, treatment_path: Path, matchup: str, *, seed: int, anchor_sha256: str | None = None) -> dict:
    cp = json.loads(control_path.read_text(encoding="utf-8"))
    tp = json.loads(treatment_path.read_text(encoding="utf-8"))
    if anchor_sha256 is None:
        manifest = Path(__file__).resolve().parents[1] / "experiments/phase10_3_7/source_manifest.json"
        anchor_sha256 = json.loads(manifest.read_text(encoding="utf-8"))["bc_v2_1_anchor_sha256"]
    if not anchor_sha256 or cp.get("candidate_checkpoint_sha256") != anchor_sha256:
        raise ValueError("control is not the registered BC-v2.1 anchor")
    if not tp.get("candidate_checkpoint_sha256"):
        raise ValueError("treatment checkpoint identity missing")
    if cp.get("environment_source_sha256") != tp.get("environment_source_sha256"):
        raise ValueError("paired evaluations use different environment sources")
    if not cp.get("benchmark_runner_sha256") or cp["benchmark_runner_sha256"] != tp.get("benchmark_runner_sha256"):
        raise ValueError("paired evaluations use different or unidentified runners")
    if matchup == "bc_v2_1":
        if any(p["matchups"][matchup].get("opponent_checkpoint_sha256") != anchor_sha256 for p in (cp, tp)):
            raise ValueError("BC opponent checkpoint differs from registered anchor")
    elif matchup == "rule":
        if any(p["matchups"][matchup].get("opponent") != "rule@rule-v2-win-first" for p in (cp, tp)):
            raise ValueError("Rule behavior version mismatch")
    control, treatment = _read(control_path, matchup), _read(treatment_path, matchup)
    if set(control) != set(treatment):
        raise ValueError("paired results must contain identical game seeds")
    seeds = sorted(control)
    if any(control[s]["seat"] != treatment[s]["seat"] for s in seeds):
        raise ValueError("paired results must use identical candidate seats")
    if matchup == "bc_v2_1":
        effects = [
            (treatment[s]["candidate_rank"] - treatment[s]["opponent_mean_rank"])
            for s in seeds
        ]
        estimand = "BC-v3 candidate-minus-same-game-BC-opponent-mean-rank"
    elif matchup == "rule":
        effects = [treatment[s]["candidate_rank"] - control[s]["candidate_rank"] for s in seeds]
        estimand = "candidate mean rank: BC-v3 minus BC-v2.1 against Rule"
    else:
        raise ValueError(f"unsupported registered matchup: {matchup}")
    point = sum(effects) / len(effects)
    state = seed & 0xFFFFFFFF
    if state == 0:
        raise ValueError("bootstrap seed must be nonzero")
    draws = []
    for _ in range(10_000):
        total = 0.0
        for _ in seeds:
            state = _xorshift32(state)
            total += effects[state % len(effects)]
        draws.append(total / len(effects))
    draws.sort()
    ci = [_quantile(draws, 0.025), _quantile(draws, 0.975)]
    passed = (point <= -0.02 and ci[1] < 0) if matchup == "bc_v2_1" else (point <= 0.02 and ci[1] < 0.05)
    return {
        "passed": passed,
        "matchup": matchup,
        "estimand": estimand,
        "games": len(seeds),
        "effect_rank": point,
        "ci95": ci,
        "control_candidate_mean_rank": sum(control[s]["candidate_rank"] for s in seeds) / len(seeds),
        "treatment_candidate_mean_rank": sum(treatment[s]["candidate_rank"] for s in seeds) / len(seeds),
        "control_illegal_actions": sum(control[s]["illegal_actions"] for s in seeds),
        "treatment_illegal_actions": sum(treatment[s]["illegal_actions"] for s in seeds),
        "bootstrap": "10000 paired game-seed resamples; xorshift32; modulo indexing; linear percentile interpolation",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--treatment", type=Path, required=True)
    ap.add_argument("--matchup", choices=("bc_v2_1", "rule"), required=True)
    ap.add_argument("--seed", type=int, default=20261108)
    args = ap.parse_args(argv)
    result = paired_analysis(args.control, args.treatment, args.matchup, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
