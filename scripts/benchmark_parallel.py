#!/usr/bin/env python
"""Parallel benchmark runner (Phase 10.1.1-D).

Each worker process runs an independent subset of games; the seed of every game
is ``base_seed + game_index`` and the candidate seat is ``game_index % 4``, so
the assignment is independent of the worker count and reproducible.

Usage::

    python scripts/benchmark_parallel.py \
        --candidate experiments/bc_v2.1/checkpoints --candidate-mode greedy \
        --opponents random,rule,bc_v1 --games-per-opponent 5000 \
        --num-workers 16 --out experiments/benchmark_v21
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


def _build_opponent(opponent_type: str, checkpoint, *, mode: str, temperature: float):
    """Construct one opponent（CLI 与 worker 共用同一路径，供测试直接调用）。"""
    from mahjong.evaluation import PolicyOpponent, RandomOpponent, RuleOpponent

    if opponent_type == "random":
        return RandomOpponent(opponent_id="random", model_version="random-v1")
    if opponent_type == "rule":
        return RuleOpponent(opponent_id="rule", model_version="rule-v1")
    if opponent_type in ("bc_v1", "bc_v2", "bc_v2_1"):
        version = {"bc_v1": "bc-v1", "bc_v2": "bc-v2", "bc_v2_1": "bc-v2.1"}[opponent_type]
        return PolicyOpponent(
            opponent_id=opponent_type,
            type="bc",
            model_version=version,
            checkpoint=checkpoint,
            mode=mode,
            temperature=temperature,
        )
    raise ValueError(f"unknown opponent_type {opponent_type!r}")


def _bench_worker(task: tuple) -> list[dict]:
    """Worker: run ``game_indices`` games of candidate vs 3x opponent."""
    (
        game_indices,
        base_seed,
        candidate_checkpoint,
        candidate_mode,
        candidate_type,
        candidate_version,
        candidate_temperature,
        opponent_temperature,
        opponent_mode,
        opponent_type,
        opponent_checkpoint,
    ) = task

    from mahjong.evaluation import PolicyOpponent
    from mahjong.evaluation.benchmark import _seat_opponents
    from mahjong.evaluation.selfplay import run_game

    candidate = PolicyOpponent(
        opponent_id="candidate",
        type=candidate_type,
        model_version=candidate_version,
        checkpoint=candidate_checkpoint,
        mode=candidate_mode,
        checkpoint_type="ppo" if candidate_type == "ppo" else None,
        temperature=candidate_temperature,
    )
    opps = [_build_opponent(opponent_type, opponent_checkpoint, mode=opponent_mode, temperature=opponent_temperature) for _ in range(3)]

    rows = []
    for gi in game_indices:
        seat = gi % 4
        seats = _seat_opponents(candidate, opps, seat)
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


def _candidate_metrics(rows: list[dict]) -> dict:
    from mahjong.evaluation.benchmark import _aggregate as _agg

    ranks = [r["final_ranks"][r["seat"]] for r in rows]
    stats = [r["stats"][r["seat"]] for r in rows]
    scores = [r["final_scores"][r["seat"]] for r in rows]
    m = _agg(ranks, stats, scores)
    m["games"] = len(rows)
    m["illegal_rate"] = sum(r["illegal_actions"] for r in rows) / max(sum(r["steps"] for r in rows), 1)
    return m


def _opponents_metrics(rows: list[dict]) -> dict:
    from mahjong.evaluation.benchmark import _aggregate as _agg

    ranks, stats, scores = [], [], []
    for r in rows:
        for seat in range(4):
            if seat != r["seat"]:
                ranks.append(r["final_ranks"][seat])
                stats.append(r["stats"][seat])
                scores.append(r["final_scores"][seat])
    m = _agg(ranks, stats, scores)
    m["games"] = len(rows) * 3
    return m


def run_matchup(
    *,
    opponent_type: str,
    games: int,
    base_seed: int,
    num_workers: int,
    candidate_checkpoint: str,
    candidate_mode: str,
    candidate_type: str,
    candidate_version: str,
    candidate_temperature: float,
    opponent_temperature: float,
    opponent_mode: str,
    opponent_checkpoint: str,
) -> tuple[list[dict], float]:
    import concurrent.futures

    indices = list(range(games))
    chunks = [indices[i::num_workers] for i in range(num_workers)]
    chunks = [c for c in chunks if c]

    t0 = time.perf_counter()
    all_rows: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(
                _bench_worker,
                (chunk, base_seed, candidate_checkpoint, candidate_mode, candidate_type, candidate_version, candidate_temperature, opponent_temperature, opponent_mode, opponent_type, opponent_checkpoint),
            )
            for chunk in chunks
        ]
        for fut in concurrent.futures.as_completed(futures):
            all_rows.extend(fut.result())
    elapsed = time.perf_counter() - t0
    all_rows.sort(key=lambda r: r["game_index"])
    return all_rows, elapsed


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Parallel BC benchmark (Phase 10.1.1-D).")
    ap.add_argument("--candidate", default="experiments/bc_v2.1/checkpoints", help="candidate checkpoint dir/file")
    ap.add_argument("--candidate-mode", default="sampling", choices=["greedy", "sampling"])
    ap.add_argument("--candidate-type", default="bc", choices=["bc", "ppo"], help="candidate checkpoint type (bc/ppo)")
    ap.add_argument("--candidate-version", default="candidate", help="candidate model_version label")
    ap.add_argument("--candidate-temperature", type=float, default=1.0, help="sampling temperature (T=1.0 = 默认，行为不变)")
    ap.add_argument("--opponent-temperature", type=float, default=1.0, help="模型对手的采样温度（默认 1.0；真镜像需与 candidate 同值）")
    ap.add_argument("--opponent-mode", default="sampling", choices=["greedy", "sampling"], help="模型对手的决策模式（默认 sampling；真镜像需与 candidate 同值）")
    ap.add_argument("--opponents", default="random", help="comma list: random,bc_v1,rule,bc_v2,bc_v2_1")
    ap.add_argument("--bc-v1-checkpoint", default="experiments/exp_0001/checkpoint")
    ap.add_argument("--bc-v2-opponent", default=None, help="bc-v2 checkpoint path (enables 'bc_v2' opponent)")
    ap.add_argument("--bc-v2.1-checkpoint", default="experiments/bc_v2.1/checkpoints", help="bc-v2.1 checkpoint path (enables 'bc_v2_1' opponent)")
    ap.add_argument("--games-per-opponent", type=int, default=5000, help="games per opponent type (each matchup)")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="experiments/benchmark_v21")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)

    candidate_path = str(Path(args.candidate).resolve())
    bc_v1_path = str(Path(args.bc_v1_checkpoint).resolve())
    bc_v2_path = str(Path(args.bc_v2_opponent).resolve()) if args.bc_v2_opponent else None
    bc_v21_path = str(Path(getattr(args, "bc_v2.1_checkpoint")).resolve())

    opponent_types = [s.strip() for s in args.opponents.split(",") if s.strip()]
    checkpoint_for = {
        "random": None,
        "rule": None,
        "bc_v1": bc_v1_path,
        "bc_v2": bc_v2_path,
        "bc_v2_1": bc_v21_path,
    }
    for ot in opponent_types:
        if ot not in checkpoint_for:
            raise SystemExit(f"unknown opponent type {ot!r} (choose from random,bc_v1,rule,bc_v2,bc_v2_1)")
        if ot == "bc_v2" and not bc_v2_path:
            raise SystemExit("--bc-v2-opponent is required when opponent 'bc_v2' is requested")

    import datetime
    import subprocess as _sp

    def _git_commit() -> str | None:
        try:
            out = _sp.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_HERE.parent, check=True)
            return out.stdout.strip()[:12]
        except Exception:
            return None

    report = {
        "candidate": args.candidate,
        "candidate_type": args.candidate_type,
        "candidate_version": args.candidate_version,
        "candidate_mode": args.candidate_mode,
        "candidate_temperature": args.candidate_temperature,
        "opponent_mode": args.opponent_mode,
        "opponent_temperature": args.opponent_temperature,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "games_per_opponent": args.games_per_opponent,
        "runner_version": _git_commit(),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "matchups": {},
    }

    total_elapsed = 0.0
    for ot in opponent_types:
        print(f"== candidate vs {ot}: {args.games_per_opponent} games, {args.num_workers} workers ==", flush=True)
        rows, elapsed = run_matchup(
            opponent_type=ot,
            games=args.games_per_opponent,
            base_seed=args.seed,
            num_workers=args.num_workers,
            candidate_checkpoint=candidate_path,
            candidate_mode=args.candidate_mode,
            candidate_type=args.candidate_type,
            candidate_version=args.candidate_version,
            candidate_temperature=args.candidate_temperature,
            opponent_temperature=args.opponent_temperature,
            opponent_mode=args.opponent_mode,
            opponent_checkpoint=checkpoint_for[ot],
        )
        total_elapsed += elapsed
        report["matchups"][ot] = {
            "candidate_metrics": _candidate_metrics(rows),
            "opponents_metrics": _opponents_metrics(rows),
            "opponent_mode": args.opponent_mode,
            "opponent_temperature": args.opponent_temperature,
            "opponent": (
                "random@random-v1"
                if ot == "random"
                else "rule@rule-v1"
                if ot == "rule"
                else "bc_v1@bc-v1"
                if ot == "bc_v1"
                else "bc_v2@bc-v2"
            ),
            "elapsed_s": round(elapsed, 1),
        }
        print(json.dumps(report["matchups"][ot], ensure_ascii=False), flush=True)

    report["total_elapsed_s"] = round(total_elapsed, 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
