"""Self-play runner (Phase 8): arbitrary four-seat opponent combinations."""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass, field

from ..environment import MahjongEnv
from .opponent import Opponent

log = logging.getLogger("evaluation.selfplay")


@dataclass
class GameResult:
    seed: int
    opponents: dict[int, str]  # seat -> opponent_id@version
    final_scores: list[int]
    final_ranks: list[int]
    steps: int
    illegal_actions: int
    stats: list[dict] = field(default_factory=list)  # per-seat SeatStats as dicts
    error: str | None = None


def run_game(
    opponents: list[Opponent],
    *,
    seed: int,
    env: MahjongEnv | None = None,
    max_steps: int = 100_000,
) -> GameResult:
    """Run one hanchan with the given 4 opponents (by seat order)."""
    assert len(opponents) == 4
    env = env or MahjongEnv(seed=seed)
    rng = random.Random(seed)
    env.reset(seed=None)
    steps = 0
    illegal = 0
    error = None
    try:
        while not env.done():
            seat = env.state.turn
            legal = env.legal_actions(seat)
            action = opponents[seat].decide(env.observation(seat), legal, rng)
            if action not in legal:
                illegal += 1
                log.warning("illegal action %r for seat %d (seed %d)", action, seat, seed)
            env.step(action)
            steps += 1
            if steps > max_steps:
                error = "max steps exceeded"
                break
    except Exception as exc:  # noqa: BLE001 - one bad game must not stop a round
        error = repr(exc)
    return GameResult(
        seed=seed,
        opponents={s: f"{opponents[s].opponent_id}@{opponents[s].model_version}" for s in range(4)},
        final_scores=list(env.state.scores),
        final_ranks=list(env.state.final_ranks or [0, 1, 2, 3]),
        steps=steps,
        illegal_actions=illegal,
        stats=[asdict(st) for st in env.state.stats],
        error=error,
    )


def run_round(
    opponents: list[Opponent],
    *,
    games: int,
    seed: int,
    continue_on_error: bool = True,
) -> list[GameResult]:
    """Run ``games`` hanchan; a single game exception is recorded, not fatal."""
    results = []
    for g in range(games):
        gseed = seed + g
        result = run_game(opponents, seed=gseed)
        results.append(result)
        if result.error and not continue_on_error:
            break
    return results


def summarize(results: list[GameResult]) -> dict:
    """Aggregate final ranks / scores / illegal rate across games."""
    n = len(results)
    total_illegal = sum(r.illegal_actions for r in results)
    total_steps = sum(r.steps for r in results)
    rank_sums = [0, 0, 0, 0]
    for r in results:
        for seat, rank in enumerate(r.final_ranks):
            rank_sums[seat] += rank
    return {
        "games": n,
        "errors": sum(1 for r in results if r.error),
        "illegal_rate": total_illegal / max(total_steps, 1),
        "mean_rank": [s / max(n, 1) for s in rank_sums],
        "mean_score": [
            sum(r.final_scores[seat] for r in results) / max(n, 1) for seat in range(4)
        ],
    }


__all__ = ["GameResult", "run_game", "run_round", "summarize"]
