"""Benchmark: fixed opponents, per-seat metrics, promotion flow (Phase 9)."""

from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ..dataset.manifest import git_commit
from ..environment import MahjongEnv
from .opponent import Opponent, RandomOpponent
from .selfplay import GameResult, run_round

log = logging.getLogger("evaluation.benchmark")

RULES_VERSION = "tenhou-v1"


def compute_metrics(results: list[GameResult], seat: int) -> dict:
    """Compute the 8 benchmark metrics for one seat across games."""
    n = len(results)
    if n == 0:
        return {}
    ranks = [r.final_ranks[seat] for r in results]
    stats = [r.stats[seat] for r in results]
    scores = [r.final_scores[seat] for r in results]

    win_count = sum(s["win_count"] for s in stats)
    dealt_in = sum(s["dealt_in_count"] for s in stats)
    riichi = sum(s["riichi_count"] for s in stats)
    call = sum(s["call_count"] for s in stats)
    win_points = sum(s["win_points"] for s in stats)

    return {
        "mean_rank": sum(ranks) / n,
        "rank_distribution": [ranks.count(k) for k in range(4)],
        "win_rate": win_count / n,
        "deal_in_rate": dealt_in / n,
        "riichi_rate": riichi / n,
        "call_rate": call / n,
        "avg_win_points": win_points / win_count if win_count else 0.0,
        "mean_score_change": (sum(scores) / n) - 25000,
    }


def _seat_opponents(candidate: Opponent, opponents: list[Opponent], candidate_seat: int) -> list[Opponent]:
    assert len(opponents) == 3
    seats: list[Opponent | None] = [None, None, None, None]
    seats[candidate_seat] = candidate
    it = iter(opponents)
    for s in range(4):
        if s != candidate_seat:
            seats[s] = next(it)
    return [o for o in seats if o is not None]


def smoke_test(candidate: Opponent, seed: int = 0) -> bool:
    """Candidate can decide on a real observation without crashing."""
    env = MahjongEnv(seed=seed)
    env.reset()
    import random as _random

    rng = _random.Random(seed)
    for _ in range(20):
        seat = env.state.turn
        legal = env.legal_actions(seat)
        candidate.decide(env.observation(seat), legal, rng)
        # advance deterministically with a random opponent
        action = rng.choice(legal)
        env.step(action)
        if env.done():
            env.reset()
    return True


def rules_test(candidate: Opponent, opponents: list[Opponent], *, games: int, seed: int, candidate_seat: int) -> tuple[bool, list[GameResult]]:
    """Fixed-games sanity: no crash and illegal action rate == 0."""
    seats = _seat_opponents(candidate, opponents, candidate_seat)
    results = run_round(seats, games=games, seed=seed)
    ok = all(r.error is None and r.illegal_actions == 0 for r in results)
    return ok, results


@dataclass
class PromotionResult:
    candidate_id: str
    candidate_version: str
    candidate_seat: int
    opponents: dict[int, str]
    games: int
    seed: int
    rules: str
    metrics: dict
    per_game: list[dict]
    illegal_rate: float
    smoke_passed: bool
    rules_passed: bool
    promoted: bool | None
    baseline_metrics: dict | None
    timestamp: str
    git_commit: str


def run_promotion(
    candidate: Opponent,
    opponents: list[Opponent],
    *,
    baseline: Opponent | None,
    games: int,
    seed: int,
    candidate_seat: int = 0,
    rules_games: int = 2,
    output_dir: str | Path = "experiments/benchmark",
) -> PromotionResult:
    """smoke -> rules -> benchmark -> compare (vs baseline) -> save (no overwrite)."""
    smoke_ok = smoke_test(candidate, seed=seed)
    rules_ok, rules_results = rules_test(candidate, opponents, games=rules_games, seed=seed, candidate_seat=candidate_seat)

    seats = _seat_opponents(candidate, opponents, candidate_seat)
    results = run_round(seats, games=games, seed=seed)
    metrics = compute_metrics(results, candidate_seat)

    baseline_metrics = None
    promoted = None
    if baseline is not None:
        base_seats = _seat_opponents(baseline, opponents, candidate_seat)
        base_results = run_round(base_seats, games=games, seed=seed)
        baseline_metrics = compute_metrics(base_results, candidate_seat)
        promoted = metrics["mean_rank"] < baseline_metrics["mean_rank"]  # lower rank is better

    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    result = PromotionResult(
        candidate_id=candidate.opponent_id,
        candidate_version=candidate.model_version,
        candidate_seat=candidate_seat,
        opponents={s: f"{o.opponent_id}@{o.model_version}" for s, o in enumerate(seats)},
        games=games,
        seed=seed,
        rules=RULES_VERSION,
        metrics=metrics,
        per_game=[
            {
                "seed": r.seed,
                "final_scores": r.final_scores,
                "final_ranks": r.final_ranks,
                "stats": r.stats,
                "error": r.error,
            }
            for r in results
        ],
        illegal_rate=sum(r.illegal_actions for r in results) / max(sum(r.steps for r in results), 1),
        smoke_passed=smoke_ok,
        rules_passed=rules_ok,
        promoted=promoted,
        baseline_metrics=baseline_metrics,
        timestamp=timestamp,
        git_commit=git_commit(Path.cwd()),
    )

    out_dir = Path(output_dir) / f"{candidate.opponent_id}-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)  # never overwrite
    (out_dir / "result.json").write_text(
        json.dumps(_dataclass_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("benchmark result saved to %s", out_dir)
    return result


def _dataclass_to_dict(obj) -> dict:
    d = obj.__dict__.copy()
    return d


__all__ = [
    "PromotionResult",
    "RULES_VERSION",
    "compute_metrics",
    "run_promotion",
    "rules_test",
    "smoke_test",
]
