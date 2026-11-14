#!/usr/bin/env python
"""Run a paired BC candidate-vs-mirror benchmark on shared seeds/seats."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
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


def _arm(candidate_checkpoint: str, candidate_version: str, *, games: int, seed: int, workers: int, opponent_checkpoint: str):
    rows, elapsed = run_matchup(
        opponent_type="bc_v3",
        games=games,
        base_seed=seed,
        num_workers=workers,
        candidate_checkpoint=candidate_checkpoint,
        candidate_mode="greedy",
        candidate_type="bc",
        candidate_version=candidate_version,
        candidate_temperature=1.0,
        opponent_temperature=1.0,
        opponent_mode="greedy",
        opponent_checkpoint=opponent_checkpoint,
    )
    from mahjong.evaluation.validation import validate_game_rows

    validate_game_rows(rows, expected_seeds=range(seed, seed + games))
    expected_seats = {seed + i: i % 4 for i in range(games)}
    if len(rows) != games or any(row["seat"] != expected_seats[row["seed"]] for row in rows):
        raise ValueError(f"{candidate_version}: incomplete or mismatched candidate-seat rotation")
    return {
        "candidate_version": candidate_version,
        "candidate_checkpoint_sha256": _sha256(Path(candidate_checkpoint)),
        "elapsed_seconds": round(elapsed, 3),
        "metrics": _candidate_metrics(rows),
        "raw_games": rows,
    }


def _paired_analysis(candidate_rows, mirror_rows, *, bootstrap_seed: int, replicates: int) -> dict:
    candidate_by_seed = {row["seed"]: row for row in candidate_rows}
    mirror_by_seed = {row["seed"]: row for row in mirror_rows}
    if len(candidate_by_seed) != len(candidate_rows) or len(mirror_by_seed) != len(mirror_rows):
        raise ValueError("duplicate seed in an arm")
    if candidate_by_seed.keys() != mirror_by_seed.keys():
        raise ValueError("candidate and mirror seed sets differ")

    pairs = []
    differences = np.empty(len(candidate_rows), dtype=np.float64)
    mirror_ranks = np.empty(len(candidate_rows), dtype=np.int8)
    candidate_ranks = np.empty(len(candidate_rows), dtype=np.int8)
    candidate_illegal = 0
    mirror_illegal = 0
    for i, seed in enumerate(sorted(candidate_by_seed)):
        c = candidate_by_seed[seed]
        m = mirror_by_seed[seed]
        if c["seat"] != m["seat"] or c["game_index"] != m["game_index"]:
            raise ValueError(f"paired seat/index mismatch at seed {seed}")
        seat = c["seat"]
        c_rank = int(c["final_ranks"][seat])
        m_rank = int(m["final_ranks"][seat])
        candidate_ranks[i] = c_rank
        mirror_ranks[i] = m_rank
        differences[i] = c_rank - m_rank
        candidate_illegal += int(c["illegal_actions"])
        mirror_illegal += int(m["illegal_actions"])
        pairs.append({"seed": seed, "seat": seat, "candidate_rank": c_rank, "mirror_rank": m_rank, "rank_difference": c_rank - m_rank})

    n = len(differences)
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    bootstrap_means = np.empty(replicates, dtype=np.float64)
    chunk = 100
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = rng.integers(0, n, size=(stop - start, n), dtype=np.int32)
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975], method="linear")

    seats = [candidate_by_seed[s]["seat"] for s in sorted(candidate_by_seed)]
    seat_counts = {str(s): seats.count(s) for s in range(4)}
    mirror_mean_rank = float(mirror_ranks.mean())
    mirror_sanity = (
        abs(mirror_mean_rank - 1.5) <= 0.02
        and mirror_illegal == 0
        and set(mirror_ranks.tolist()).issubset({0, 1, 2, 3})
        and seat_counts == {"0": n // 4, "1": n // 4, "2": n // 4, "3": n // 4}
    )
    candidate_mean_rank = float(candidate_ranks.mean())
    effect = float(differences.mean())
    candidate_valid = all(row["completed"] and row["error"] is None for row in candidate_rows)
    mirror_valid = all(row["completed"] and row["error"] is None for row in mirror_rows)
    valid = candidate_valid and mirror_valid and candidate_illegal == 0 and mirror_illegal == 0 and mirror_sanity
    go = valid and effect <= -0.020 and ci_high < 0

    return {
        "games_per_arm": n,
        "paired_seed_range": [min(candidate_by_seed), max(candidate_by_seed)],
        "candidate_mean_rank": candidate_mean_rank,
        "mirror_mean_rank": mirror_mean_rank,
        "paired_rank_difference_mean": effect,
        "paired_rank_difference_sample_sd": float(differences.std(ddof=1)),
        "paired_rank_difference_bootstrap_ci95": [float(ci_low), float(ci_high)],
        "bootstrap": {"replicates": replicates, "seed": bootstrap_seed, "generator": "NumPy PCG64", "interval": "percentile; linear quantile interpolation"},
        "seat_counts": seat_counts,
        "candidate_illegal_actions": candidate_illegal,
        "mirror_illegal_actions": mirror_illegal,
        "candidate_all_games_valid": candidate_valid,
        "mirror_all_games_valid": mirror_valid,
        "mirror_sanity_tolerance": 0.02,
        "mirror_sanity_passed": mirror_sanity,
        "valid_run": valid,
        "go_thresholds": {"rank_difference_at_most": -0.020, "ci95_upper_below": 0.0},
        "decision": "GO" if go else "NO_GO",
        "paired_rows": pairs,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-checkpoint", required=True)
    ap.add_argument("--mirror-checkpoint", required=True)
    ap.add_argument("--opponent-checkpoint", required=True)
    ap.add_argument("--games-per-arm", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num-workers", type=int, required=True)
    ap.add_argument("--bootstrap-seed", type=int, required=True)
    ap.add_argument("--bootstrap-replicates", type=int, default=10000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidate_path = Path(args.candidate_checkpoint).resolve()
    mirror_path = Path(args.mirror_checkpoint).resolve()
    opponent_path = Path(args.opponent_checkpoint).resolve()
    started = datetime.now().astimezone().isoformat()
    source_paths = (
        _HERE / "benchmark_paired_bc.py",
        _HERE / "benchmark_parallel.py",
        _ROOT / "src/mahjong/evaluation/opponent.py",
        _ROOT / "src/mahjong/evaluation/selfplay.py",
        _ROOT / "src/mahjong/evaluation/validation.py",
        _ROOT / "src/mahjong/model/policy.py",
        _ROOT / "src/mahjong/features/encoder_v2.py",
        _ROOT / "src/mahjong/features/encoder_v3.py",
    )
    metadata = {
        "stage": "10.1.1-strength",
        "started_local": started,
        "candidate": str(candidate_path),
        "candidate_sha256": _sha256(candidate_path),
        "mirror": str(mirror_path),
        "mirror_sha256": _sha256(mirror_path),
        "opponent": str(opponent_path),
        "opponent_sha256": _sha256(opponent_path),
        "feature_versions": {"candidate": "feature-v3", "mirror": "feature-v2", "action": "action-v2"},
        "candidate_mode": "greedy",
        "mirror_mode": "greedy",
        "opponent_mode": "greedy",
        "temperatures": {"candidate": 1.0, "mirror": 1.0, "opponent": 1.0},
        "games_per_arm": args.games_per_arm,
        "seed_start": args.seed,
        "seed_end_inclusive": args.seed + args.games_per_arm - 1,
        "seat_rotation": "game_index % 4; same seat for paired candidate/mirror seed",
        "num_workers": args.num_workers,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "runner_sha256": _sha256(Path(__file__)),
        "parallel_runner_sha256": _sha256(_HERE / "benchmark_parallel.py"),
        "source_sha256": {str(p.relative_to(_ROOT)): _sha256(p) for p in source_paths},
        "python": sys.version,
        "platform": platform.platform(),
        "estimated_minutes": "90-130 based on prior 20k-pair, 16-worker run: 6069 seconds",
        "time_cap": None,
        "live_monitoring": False,
    }
    _write_json(out / "run_metadata.json", metadata)

    start = time.perf_counter()
    print("Starting candidate arm (BC-v4 vs 3x BC-v3)", flush=True)
    cand = _arm(str(candidate_path), "bc-v4", games=args.games_per_arm, seed=args.seed, workers=args.num_workers, opponent_checkpoint=str(opponent_path))
    _write_json(out / "candidate_arm.json", cand)
    print(f"Candidate arm complete in {cand['elapsed_seconds']:.1f}s", flush=True)

    print("Starting mirror arm (BC-v3 vs 3x BC-v3)", flush=True)
    mirror = _arm(str(mirror_path), "bc-v3-mirror", games=args.games_per_arm, seed=args.seed, workers=args.num_workers, opponent_checkpoint=str(opponent_path))
    _write_json(out / "mirror_arm.json", mirror)
    print(f"Mirror arm complete in {mirror['elapsed_seconds']:.1f}s", flush=True)

    if cand["candidate_checkpoint_sha256"] != metadata["candidate_sha256"] or mirror["candidate_checkpoint_sha256"] != metadata["mirror_sha256"]:
        raise ValueError("checkpoint SHA changed during run")
    if cand["metrics"]["games"] != args.games_per_arm or mirror["metrics"]["games"] != args.games_per_arm:
        raise ValueError("arm game count mismatch")

    result = {
        **metadata,
        "candidate_arm_elapsed_seconds": cand["elapsed_seconds"],
        "mirror_arm_elapsed_seconds": mirror["elapsed_seconds"],
        "total_elapsed_seconds": round(time.perf_counter() - start, 3),
        "candidate_metrics": cand["metrics"],
        "mirror_metrics": mirror["metrics"],
        "analysis": _paired_analysis(
            cand["raw_games"], mirror["raw_games"],
            bootstrap_seed=args.bootstrap_seed,
            replicates=args.bootstrap_replicates,
        ),
    }
    paired_rows = result["analysis"].pop("paired_rows")
    _write_json(out / "result.json", result)
    with (out / "paired_rank_differences.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("seed", "seat", "candidate_rank", "mirror_rank", "rank_difference"))
        writer.writeheader()
        writer.writerows(paired_rows)
    print(json.dumps({key: value for key, value in result["analysis"].items()}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
