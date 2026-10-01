"""Deterministic game-level train/validation/test splitting.

Splitting is per *complete game* (DATA_SPEC section 8, TRAINING_SPEC section 3):
every sample of a game goes to the same split, so no game leaks across splits.

The assignment is a seeded shuffle of the sorted game ids followed by exact
proportional boundaries; the same game ids + seed always produce the same
assignment.
"""

from __future__ import annotations

import random

SPLIT_NAMES = ("train", "validation", "test")


def assign_splits(
    game_ids,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> dict[str, str]:
    """Return ``{game_id: split_name}`` for every distinct game id.

    ``ratios`` is ``(train, validation, test)`` and must sum to 1.0.
    """
    if len(ratios) != 3:
        raise ValueError("ratios must have exactly 3 entries (train, validation, test)")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")

    unique = sorted(set(game_ids))
    rng = random.Random(seed)
    shuffled = list(unique)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = min(n, round(n * ratios[0]))
    n_val = min(n - n_train, round(n * ratios[1]))
    # test gets the remainder.

    assignment: dict[str, str] = {}
    for gid in shuffled[:n_train]:
        assignment[gid] = "train"
    for gid in shuffled[n_train : n_train + n_val]:
        assignment[gid] = "validation"
    for gid in shuffled[n_train + n_val :]:
        assignment[gid] = "test"
    return assignment
