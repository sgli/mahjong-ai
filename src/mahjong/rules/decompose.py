"""Enumerate standard winning-hand decompositions (4 sets + 1 pair).

A winning concealed hand decomposes into ``4 - len(melds)`` concealed sets
(runs / triplets) plus one pair; combined with the (already formed) melds this
gives every possible interpretation of the hand, which scoring uses to pick the
highest-scoring interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .tiles import ALL_TILES, normalize, to_counts


@dataclass(frozen=True)
class Group:
    kind: str  # "run" | "triplet" | "quad"
    tiles: tuple[str, ...]  # base (normalised) tiles, sorted
    open: bool  # True for called melds, False for concealed groups (ankan is closed)


@dataclass(frozen=True)
class Structure:
    sets: tuple[Group, ...]  # exactly 4 groups
    pair: str  # base tile of the pair


def _first_nonzero(counts) -> int:
    for i, c in enumerate(counts):
        if c:
            return i
    return -1


def _enumerate_sets(counts, sets_left: int):
    """Yield lists of concealed Group objects for the remaining sets."""
    if sets_left == 0:
        if all(c == 0 for c in counts):
            yield []
        return

    i = _first_nonzero(counts)
    if i < 0:
        return

    # triplet
    if counts[i] >= 3:
        c = list(counts)
        c[i] -= 3
        for rest in _enumerate_sets(tuple(c), sets_left - 1):
            yield [Group("triplet", (ALL_TILES[i],) * 3, False)] + rest

    # run
    if i < 27 and i % 9 <= 6 and counts[i] and counts[i + 1] and counts[i + 2]:
        c = list(counts)
        c[i] -= 1
        c[i + 1] -= 1
        c[i + 2] -= 1
        for rest in _enumerate_sets(tuple(c), sets_left - 1):
            yield [Group("run", (ALL_TILES[i], ALL_TILES[i + 1], ALL_TILES[i + 2]), False)] + rest


def _meld_to_group(meld) -> Group:
    tiles = tuple(sorted(normalize(t) for t in meld.tiles))
    kind = meld.kind
    if kind == "chi":
        return Group("run", tiles, True)
    if kind == "pon":
        return Group("triplet", tiles, True)
    if kind == "ankan":
        return Group("quad", tiles, False)  # closed kan
    if kind == "daiminkan":
        return Group("quad", tiles, True)
    if kind == "kakan":
        return Group("quad", tiles, True)
    raise ValueError(f"unknown meld kind {kind!r}")


def iter_structures(hand_tiles, melds) -> Iterator[Structure]:
    """Yield every standard interpretation of a winning hand."""
    counts = to_counts(hand_tiles)
    sets_needed = 4 - len(melds)

    meld_groups = [_meld_to_group(m) for m in melds]

    for pair_idx in range(34):
        if counts[pair_idx] >= 2:
            c2 = list(counts)
            c2[pair_idx] -= 2
            for concealed_sets in _enumerate_sets(tuple(c2), sets_needed):
                yield Structure(tuple(concealed_sets + meld_groups), ALL_TILES[pair_idx])


__all__ = ["Group", "Structure", "iter_structures"]
