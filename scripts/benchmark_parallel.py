#!/usr/bin/env python
"""Parallel benchmark runner (Phase 10.1 Task4).

Each worker process runs an independent subset of games; the seed of every game
is ``base_seed + game_index`` and the candidate seat is ``game_index % 4``, so
the assignment is independent of the worker count and reproducible.

Usage::

    python scripts/benchmark_parallel.py --candidate experiments/bc_v2/checkpoints \
        --bc-v1 experiments/exp_0001/checkpoint --games 10000 --num-workers 16 \
        --out experiments/benchmark_v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _bench_worker(task: tuple) -> list[dict]:
    """Worker: run ``game_indices`` games of bc-v2 vs (random|bc_v1)."""
    opponent_type, game_indices, base_seed, bc2_path, bc1_path = task

    from mahjong.evaluation import PolicyOpponent, RandomOpponent
    from mahjong.evaluation.benchmark import _seat_opponents
    from mahjong.evaluation.selfplay import run_game

    bc2 = PolicyOpponent(opponent_id="bc_v2", type="bc", model_version="bc-v2", checkpoint=bc2_path)
    if opponent_type == "random":
        opps = [RandomOpponent(opponent_id="random", model_version="random-v1") for _ in range(3)]
    elif opponent_type == "bc_v1":
        opps = [
            PolicyOpponent(opponent_id="bc_v1", type="bc", model_version="bc-v1", checkpoint=bc1_path)
            for _ in range(3)
        ]
    else:
        raise ValueError(f"unknown opponent_type {opponent_type!r}")

    rows = []
    for gi in game_indices:
        seat = gi % 4
        seats = _seat_opponents(bc2, opps, seat)
        gr = run_game(seats, seed=base_seed + gi)
        rows.append(
            {
                "game_index": gi,
                "seat": seat,
                "final_ranks": list(gr.final_ranks),
                "final_scores": list(gr.final_scores),
                "stats": [dict(s) for s in gr.stats],
                "illegal_actions": gr.illegal_actions,
                "steps": gr.steps,
                "error": gr.error,
            }
        )
    return rows


def _metrics(rows: list[dict]) -> dict:
    from mahjong.evaluation.benchmark import _aggregate

    ranks = [r["final_ranks"][r["seat"]] for r in rows]
    stats = [r["stats"][r["seat"]] for r in rows]
    scores = [r["final_scores"][r["seat"]] for r in rows]
    m = _aggregate(ranks, stats, scores)
    total_illegal = sum(r["illegal_actions"] for r in rows)
    total_steps = sum(r["steps"] for r in rows)
    m["illegal_rate"] = total_illegal / max(total_steps, 1)
    m["games"] = len(rows)
    return m


def run_parallel(
    *,
    opponent_type: str,
    games: int,
    base_seed: int,
    num_workers: int,
    bc2_path: str,
    bc1_path: str,
) -> tuple[list[dict], float]:
    import concurrent.futures

    indices = list(range(games))
    chunks = [indices[i::num_workers] for i in range(num_workers)]
    chunks = [c for c in chunks if c]

    t0 = time.perf_counter()
    all_rows: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(_bench_worker, (opponent_type, chunk, base_seed, bc2_path, bc1_path))
            for chunk in chunks
        ]
        for fut in concurrent.futures.as_completed(futures):
            all_rows.extend(fut.result())
    elapsed = time.perf_counter() - t0

    all_rows.sort(key=lambda r: r["game_index"])
    return all_rows, elapsed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Parallel BC benchmark.")
    ap.add_argument("--candidate", default="experiments/bc_v2/checkpoints", help="bc-v2 checkpoint dir")
    ap.add_argument("--bc-v1", default="experiments/exp_0001/checkpoint", help="bc-v1 checkpoint dir")
    ap.add_argument("--games", type=int, default=10000, help="total games (split across opponent types)")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--games-random", type=int, default=0, help="games vs random (0 = half of --games)")
    ap.add_argument("--games-bc1", type=int, default=0, help="games vs bc-v1 (0 = half of --games)")
    ap.add_argument("--out", default="experiments/benchmark_v2")
    args = ap.parse_args(argv)

    bc2_path = str(Path(args.candidate).resolve())
    bc1_path = str(Path(args.bc_v1).resolve())

    g_random = args.games_random or (args.games // 2)
    g_bc1 = args.games_bc1 or (args.games - g_random)

    report = {"seed": args.seed, "num_workers": args.num_workers, "candidate": "bc_v2@bc-v2"}
    total_elapsed = 0.0

    for label, opponent_type, g in (
        ("vs_random", "random", g_random),
        ("vs_bc1", "bc_v1", g_bc1),
    ):
        print(f"== {label}: {g} games, {args.num_workers} workers ==", flush=True)
        rows, elapsed = run_parallel(
            opponent_type=opponent_type,
            games=g,
            base_seed=args.seed,
            num_workers=args.num_workers,
            bc2_path=bc2_path,
            bc1_path=bc1_path,
        )
        total_elapsed += elapsed
        m = _metrics(rows)
        report[label] = {"metrics": m, "games": g, "elapsed_s": round(elapsed, 1)}
        print(json.dumps(m, ensure_ascii=False), flush=True)

    report["total_elapsed_s"] = round(total_elapsed, 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "_parallel_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
