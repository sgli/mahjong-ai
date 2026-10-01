"""Versioned opponent pool (Phase 8, TRAINING_SPEC 9)."""

from __future__ import annotations

from .opponent import Opponent, PolicyOpponent, RandomOpponent


class OpponentPool:
    """Registry of opponents keyed by ``opponent_id`` (latest registration wins).

    Supports querying by id and listing; each :class:`Opponent` carries its
    ``type`` / ``model_version`` / ``checkpoint`` metadata.
    """

    def __init__(self) -> None:
        self._opponents: dict[str, Opponent] = {}

    def add(self, opponent: Opponent) -> None:
        self._opponents[opponent.opponent_id] = opponent

    def get(self, opponent_id: str) -> Opponent:
        if opponent_id not in self._opponents:
            raise KeyError(f"unknown opponent id: {opponent_id!r}")
        return self._opponents[opponent_id]

    def list(self) -> list[Opponent]:
        return list(self._opponents.values())

    def ids(self) -> list[str]:
        return list(self._opponents.keys())

    def by_type(self, type: str) -> list[Opponent]:
        return [o for o in self._opponents.values() if o.type == type]

    def __len__(self) -> int:
        return len(self._opponents)

    def __contains__(self, opponent_id: str) -> bool:
        return opponent_id in self._opponents


def default_pool(device: str | None = None) -> OpponentPool:
    """A baseline pool: random + (if checkpoints exist) BC and current PPO."""
    pool = OpponentPool()
    pool.add(RandomOpponent(opponent_id="random", model_version="random-v1"))
    # These are optional conveniences; callers add their own checkpoints.
    return pool


__all__ = ["OpponentPool", "default_pool"]
