#!/usr/bin/env python
"""Paired PPO-vs-BC-v4 benchmark with a BC-v4 mirror control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from statistics import NormalDist
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from benchmark_parallel import _candidate_metrics, run_matchup  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _arm(candidate_checkpoint: Path, candidate_type: str, candidate_version: str,
         *, games: int, seed: int, workers: int, opponent_checkpoint: Path) -> dict:
    rows, elapsed = run_matchup(
        opponent_type="bc_v4",
        games=games,
        base_seed=seed,
        num_workers=workers,
        candidate_checkpoint=str(candidate_checkpoint),
        candidate_mode="greedy",
        candidate_type=candidate_type,
        candidate_version=candidate_version,
        candidate_temperature=1.0,
        opponent_temperature=1.0,
        opponent_mode="greedy",
        opponent_checkpoint=str(opponent_checkpoint),
    )
    from mahjong.evaluation.validation import validate_game_rows

    validate_game_rows(rows, expected_seeds=range(seed, seed + games))
    expected_seat = {seed + i: i % 4 for i in range(games)}
    if len(rows) != games or any(row["seat"] != expected_seat[row["seed"]] for row in rows):
        raise ValueError(f"{candidate_version}: incomplete or mismatched seat rotation")
    return {
        "candidate_version": candidate_version,
        "candidate_type": candidate_type,
        "candidate_checkpoint_sha256": _sha256(candidate_checkpoint),
        "elapsed_seconds": round(elapsed, 3),
        "metrics": _candidate_metrics(rows),
        "raw_games": rows,
    }


def mirror_tolerance(n: int, mode: str) -> float:
    if n <= 0:
        raise ValueError("mirror sample count must be positive")
    if mode == "fixed-v1":
        return 0.02
    if mode == "normal99-v2":
        return NormalDist().inv_cdf(0.995) * (1.25 / n) ** 0.5
    raise ValueError(f"unknown mirror sanity mode: {mode}")


def _paired_analysis(candidate_rows, mirror_rows, *, bootstrap_seed: int, replicates: int,
                     mirror_sanity_mode: str = "fixed-v1") -> dict:
    candidate = {r["seed"]: r for r in candidate_rows}
    mirror = {r["seed"]: r for r in mirror_rows}
    if len(candidate) != len(candidate_rows) or len(mirror) != len(mirror_rows):
        raise ValueError("duplicate seed in an arm")
    if candidate.keys() != mirror.keys():
        raise ValueError("candidate and mirror seed sets differ")

    seeds = sorted(candidate)
    differences = np.empty(len(seeds), dtype=np.float64)
    candidate_ranks = np.empty(len(seeds), dtype=np.int8)
    mirror_ranks = np.empty(len(seeds), dtype=np.int8)
    pairs = []
    candidate_illegal = mirror_illegal = 0
    for i, seed in enumerate(seeds):
        c, m = candidate[seed], mirror[seed]
        if c["seat"] != m["seat"] or c["game_index"] != m["game_index"]:
            raise ValueError(f"paired seat/index mismatch at seed {seed}")
        seat = c["seat"]
        cr, mr = int(c["final_ranks"][seat]), int(m["final_ranks"][seat])
        candidate_ranks[i], mirror_ranks[i] = cr, mr
        differences[i] = cr - mr
        candidate_illegal += int(c["illegal_actions"])
        mirror_illegal += int(m["illegal_actions"])
        pairs.append({"seed": seed, "seat": seat, "candidate_rank": cr,
                      "mirror_rank": mr, "rank_difference": cr - mr})

    n = len(seeds)
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    boot = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 100):
        end = min(start + 100, replicates)
        indices = rng.integers(0, n, size=(end - start, n), dtype=np.int32)
        boot[start:end] = differences[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975], method="linear")
    seats = {str(s): sum(candidate[seed]["seat"] == s for seed in seeds) for s in range(4)}
    mirror_mean = float(mirror_ranks.mean())
    tolerance = mirror_tolerance(n, mirror_sanity_mode)
    mirror_sanity = (abs(mirror_mean - 1.5) <= tolerance and mirror_illegal == 0
                     and seats == {str(s): n // 4 for s in range(4)})
    candidate_valid = all(r["completed"] and r["error"] is None for r in candidate_rows)
    mirror_valid = all(r["completed"] and r["error"] is None for r in mirror_rows)
    valid = candidate_valid and mirror_valid and candidate_illegal == 0 and mirror_illegal == 0 and mirror_sanity
    effect = float(differences.mean())
    return {
        "games_per_arm": n,
        "paired_seed_range": [min(seeds), max(seeds)],
        "candidate_mean_rank": float(candidate_ranks.mean()),
        "mirror_mean_rank": mirror_mean,
        "paired_rank_difference_mean": effect,
        "paired_rank_difference_sample_sd": float(differences.std(ddof=1)) if n > 1 else 0.0,
        "paired_rank_difference_bootstrap_ci95": [float(ci_low), float(ci_high)],
        "bootstrap": {"replicates": replicates, "seed": bootstrap_seed,
                      "generator": "NumPy PCG64", "interval": "percentile; linear quantile interpolation"},
        "seat_counts": seats,
        "candidate_illegal_actions": candidate_illegal,
        "mirror_illegal_actions": mirror_illegal,
        "candidate_all_games_valid": candidate_valid,
        "mirror_all_games_valid": mirror_valid,
        "mirror_sanity_mode": mirror_sanity_mode,
        "mirror_sanity_tolerance": tolerance,
        "mirror_sanity_passed": mirror_sanity,
        "valid_run": valid,
        "go_thresholds": {"rank_difference_at_most": -0.020, "ci95_upper_below": 0.0},
        "decision": ("INVALID" if not valid and mirror_sanity_mode != "fixed-v1" else
                     "GO" if valid and effect <= -0.020 and ci_high < 0 else "NO_GO"),
        "paired_rows": pairs,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-checkpoint", type=Path, required=True)
    ap.add_argument("--mirror-checkpoint", type=Path, required=True)
    ap.add_argument("--opponent-checkpoint", type=Path, required=True)
    ap.add_argument("--games-per-arm", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--bootstrap-seed", type=int, required=True)
    ap.add_argument("--bootstrap-replicates", type=int, default=10000)
    ap.add_argument("--mirror-sanity-mode", choices=("fixed-v1", "normal99-v2"), default="fixed-v1")
    ap.add_argument("--stage-id", default="phase10.2.1-ppo-strength")
    ap.add_argument("--candidate-version", default="ppo-v4-seed42")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pilot-only", action="store_true")
    args = ap.parse_args(argv)

    candidate_path = args.candidate_checkpoint.resolve()
    mirror_path = args.mirror_checkpoint.resolve()
    opponent_path = args.opponent_checkpoint.resolve()
    for p in (candidate_path, mirror_path, opponent_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_paths = (
        Path(__file__), _HERE / "benchmark_parallel.py",
        _ROOT / "src/mahjong/evaluation/opponent.py",
        _ROOT / "src/mahjong/evaluation/selfplay.py",
        _ROOT / "src/mahjong/evaluation/validation.py",
        _ROOT / "src/mahjong/model/policy.py",
    )
    metadata = {
        "stage": f"{args.stage_id}-pilot" if args.pilot_only else args.stage_id,
        "formal_preregistered_run": not args.pilot_only,
        "started_local": datetime.now().astimezone().isoformat(),
        "candidate": str(candidate_path), "candidate_sha256": _sha256(candidate_path),
        "candidate_type": "ppo", "mirror": str(mirror_path), "mirror_sha256": _sha256(mirror_path),
        "mirror_type": "bc", "opponent": str(opponent_path), "opponent_sha256": _sha256(opponent_path),
        "feature_version": "feature-v3", "action_schema_version": "action-v2",
        "candidate_mode": "greedy", "mirror_mode": "greedy", "opponent_mode": "greedy",
        "temperature": 1.0, "games_per_arm": args.games_per_arm,
        "seed_start": args.seed, "seed_end_inclusive": args.seed + args.games_per_arm - 1,
        "seat_rotation": "game_index % 4; candidate/mirror paired on same seed and seat",
        "num_workers": args.num_workers, "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "mirror_sanity_mode": args.mirror_sanity_mode,
        "python": sys.version, "platform": platform.platform(),
        "source_sha256": {str(p.relative_to(_ROOT)): _sha256(p) for p in source_paths},
        "time_cap": None, "live_monitoring": False,
    }
    _write_json(out / "run_metadata.json", metadata)
    start = time.perf_counter()
    print(f"Starting PPO candidate arm: {args.games_per_arm} games", flush=True)
    cand = _arm(candidate_path, "ppo", args.candidate_version, games=args.games_per_arm,
                seed=args.seed, workers=args.num_workers, opponent_checkpoint=opponent_path)
    _write_json(out / "candidate_arm.json", cand)
    print(f"Candidate arm complete in {cand['elapsed_seconds']:.1f}s", flush=True)
    print(f"Starting BC-v4 mirror arm: {args.games_per_arm} games", flush=True)
    mirror = _arm(mirror_path, "bc", "bc-v4-mirror", games=args.games_per_arm,
                  seed=args.seed, workers=args.num_workers, opponent_checkpoint=opponent_path)
    _write_json(out / "mirror_arm.json", mirror)
    result = {
        **metadata,
        "candidate_arm_elapsed_seconds": cand["elapsed_seconds"],
        "mirror_arm_elapsed_seconds": mirror["elapsed_seconds"],
        "total_elapsed_seconds": round(time.perf_counter() - start, 3),
        "candidate_metrics": cand["metrics"], "mirror_metrics": mirror["metrics"],
        "analysis": _paired_analysis(cand["raw_games"], mirror["raw_games"],
                                     bootstrap_seed=args.bootstrap_seed,
                                     replicates=args.bootstrap_replicates,
                                     mirror_sanity_mode=args.mirror_sanity_mode),
    }
    paired_rows = result["analysis"].pop("paired_rows")
    if args.pilot_only:
        result["analysis"]["decision"] = "PILOT_ONLY"
    _write_json(out / "result.json", result)
    with (out / "paired_rank_differences.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("seed", "seat", "candidate_rank", "mirror_rank", "rank_difference"))
        writer.writeheader()
        writer.writerows(paired_rows)
    print(json.dumps(result["analysis"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
