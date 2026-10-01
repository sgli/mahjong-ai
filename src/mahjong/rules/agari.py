"""Winning-hand (agari), shanten, tenpai and furiten logic.

Standard riichi rules.  Hands are passed as iterables of tile strings (red
fives are normalised internally) plus a list of open :class:`Meld` objects
(``kind``/``tiles``).  A kan meld occupies four tiles but still counts as one
"set" for the 4-sets + 1-pair decomposition; the concealed-hand size invariant
(``len(hand) + 3 * len(melds)`` is 13 at rest or 14 on a winning draw) keeps
that consistent.

Furiten implemented here is **discard furiten** (a winning tile already in the
player's own river).  Temporary furiten (passing a winning ron while tenpai, or
while riichi) is a stateful rule handled later by the Environment; for decision
extraction we only need the common discard-furiten check (see notes).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from .tiles import (
    ALL_TILES,
    HONORS,
    ORPHANS,
    index_of,
    normalize,
    to_counts,
)


def _is_valid_meld_count(melds) -> bool:
    return len(melds) <= 4


@lru_cache(maxsize=65_536)
def _can_decompose(counts: tuple[int, ...], sets_needed: int, pair_needed: int) -> bool:
    """Can ``counts`` (34-tuple) form ``sets_needed`` sets + ``pair_needed`` pairs?

    Sets are triplets (three of a kind) or runs (three consecutive suited
    tiles).  Only used for standard-form agari checking.
    """
    i = 0
    while i < 34 and counts[i] == 0:
        i += 1
    if i == 34:
        return sets_needed == 0 and pair_needed == 0
    if sets_needed < 0 or pair_needed < 0:
        return False

    # Use a pair.
    if pair_needed > 0 and counts[i] >= 2:
        c = list(counts)
        c[i] -= 2
        if _can_decompose(tuple(c), sets_needed, pair_needed - 1):
            return True

    # Use a triplet.
    if counts[i] >= 3:
        c = list(counts)
        c[i] -= 3
        if _can_decompose(tuple(c), sets_needed - 1, pair_needed):
            return True

    # Use a run (only suited tiles, ranks 1..7 can start a run).
    if i < 27 and i % 9 <= 6 and counts[i] >= 1 and counts[i + 1] >= 1 and counts[i + 2] >= 1:
        c = list(counts)
        c[i] -= 1
        c[i + 1] -= 1
        c[i + 2] -= 1
        if _can_decompose(tuple(c), sets_needed - 1, pair_needed):
            return True

    return False


def _standard_from_counts(counts: list[int], num_melds: int) -> bool:
    if num_melds > 4:
        return False
    if max(counts) > 4:
        return False
    return _can_decompose(tuple(counts), 4 - num_melds, 1)


def is_standard_agari(hand_tiles: Iterable[str], melds) -> bool:
    """Standard form: ``4 - len(melds)`` sets + 1 pair from the concealed hand."""
    if not _is_valid_meld_count(melds):
        return False
    return _standard_from_counts(to_counts(hand_tiles), len(melds))


def _chiitoi_from_counts(counts: list[int]) -> bool:
    return sum(1 for c in counts if c == 2) == 7 and all(c in (0, 2) for c in counts)


def is_chiitoi(hand_tiles: Iterable[str]) -> bool:
    """Seven pairs (requires 14 tiles and 7 distinct pairs, no melds)."""
    tiles = [normalize(t) for t in hand_tiles]
    if len(tiles) != 14:
        return False
    return _chiitoi_from_counts(to_counts(tiles))


def _kokushi_from_tiles(tiles) -> bool:
    if len(tiles) != 14:
        return False
    if any(t not in ORPHANS for t in tiles):
        return False
    return len(set(tiles)) == 13


def is_kokushi(hand_tiles: Iterable[str]) -> bool:
    """Thirteen orphans (14 tiles: one of each orphan + one duplicate)."""
    return _kokushi_from_tiles([normalize(t) for t in hand_tiles])


def is_agari(hand_tiles: Iterable[str], melds=()) -> bool:
    """True if ``hand_tiles`` (including any winning tile) + ``melds`` is a win.

    ``hand_tiles`` is the concealed hand and must contain the tile that
    completes the hand (a drawn tile for tsumo, the claimed tile for ron).
    """
    tiles = [normalize(t) for t in hand_tiles]
    if len(tiles) + 3 * len(melds) != 14:
        return False
    counts = to_counts(tiles)
    if len(melds) == 0:
        if _chiitoi_from_counts(counts):
            return True
        if _kokushi_from_tiles(tiles):
            return True
    return _standard_from_counts(counts, len(melds))


# -- shanten -------------------------------------------------------------------
@lru_cache(maxsize=65_536)
def _standard_best(counts: tuple[int, ...], sets_needed: int, sets: int, partial: int, pair: int) -> int:
    """Maximise ``2*complete + partial + pair`` for a standard hand.

    ``sets_needed`` is the number of sets still required from the concealed
    hand (``4 - len(melds)``).  A partial is a two-tile group one draw away from
    a set (a pair-as-triplet or a two-tile run).  ``pair`` is the final pair
    flag (0 or 1).
    """
    i = 0
    while i < 34 and counts[i] == 0:
        i += 1
    if i == 34:
        return 2 * sets + partial + pair

    best = -1

    # Skip this tile type entirely (all copies treated as isolated).
    c = list(counts)
    c[i] = 0
    best = max(best, _standard_best(tuple(c), sets_needed, sets, partial, pair))

    # Final pair.
    if pair == 0 and counts[i] >= 2:
        c = list(counts)
        c[i] -= 2
        best = max(best, _standard_best(tuple(c), sets_needed, sets, partial, 1))

    # Complete triplet.
    if sets + partial < sets_needed and counts[i] >= 3:
        c = list(counts)
        c[i] -= 3
        best = max(best, _standard_best(tuple(c), sets_needed, sets + 1, partial, pair))

    # Complete run.
    if (
        sets + partial < sets_needed
        and i < 27
        and i % 9 <= 6
        and counts[i] >= 1
        and counts[i + 1] >= 1
        and counts[i + 2] >= 1
    ):
        c = list(counts)
        c[i] -= 1
        c[i + 1] -= 1
        c[i + 2] -= 1
        best = max(best, _standard_best(tuple(c), sets_needed, sets + 1, partial, pair))

    # Partial triplet (pair that could become a triplet).
    if sets + partial < sets_needed and counts[i] >= 2:
        c = list(counts)
        c[i] -= 2
        best = max(best, _standard_best(tuple(c), sets_needed, sets, partial + 1, pair))

    # Partial run (two adjacent tiles, e.g. 45).
    if (
        sets + partial < sets_needed
        and i < 27
        and i % 9 <= 7
        and counts[i] >= 1
        and counts[i + 1] >= 1
    ):
        c = list(counts)
        c[i] -= 1
        c[i + 1] -= 1
        best = max(best, _standard_best(tuple(c), sets_needed, sets, partial + 1, pair))

    # Partial run with a gap (e.g. 46).
    if (
        sets + partial < sets_needed
        and i < 27
        and i % 9 <= 6
        and counts[i] >= 1
        and counts[i + 2] >= 1
    ):
        c = list(counts)
        c[i] -= 1
        c[i + 2] -= 1
        best = max(best, _standard_best(tuple(c), sets_needed, sets, partial + 1, pair))

    return best


def _chiitoi_shanten(tiles) -> int:
    counts = to_counts(tiles)
    pairs = sum(1 for c in counts if c >= 2)
    kinds = sum(1 for c in counts if c >= 1)
    return 6 - pairs + max(0, 7 - kinds)


def _kokushi_shanten(tiles) -> int:
    orphans = [t for t in tiles if t in ORPHANS]
    distinct = len(set(orphans))
    has_pair = False
    seen = set()
    for t in orphans:
        if t in seen:
            has_pair = True
            break
        seen.add(t)
    return 13 - distinct - (1 if has_pair else 0)


def shanten(hand_tiles: Iterable[str], melds=()) -> int:
    """Minimum number of tiles to replace to reach tenpai.

    Returns ``-1`` for a complete winning hand.
    """
    tiles = [normalize(t) for t in hand_tiles]
    if is_agari(tiles, melds):
        return -1
    sets_needed = 4 - len(melds)
    best = _standard_best(tuple(to_counts(tiles)), sets_needed, 0, 0, 0)
    result = 2 * sets_needed - best
    # Chiitoi / kokushi only apply to closed hands; both 13-tile (at rest) and
    # 14-tile (after a draw) hands must consider them, e.g. 6 pairs + 2 singles
    # is chiitoi tenpai.
    if len(melds) == 0 and len(tiles) in (13, 14):
        result = min(result, _chiitoi_shanten(tiles), _kokushi_shanten(tiles))
    return result


def tenpai_tiles(hand_tiles: Iterable[str], melds=()) -> list[str]:
    """The winning tiles (waits) for a hand at rest (13 - 3*len(melds) tiles)."""
    tiles = [normalize(t) for t in hand_tiles]
    waits = []
    for t in ALL_TILES:
        if is_agari(tiles + [t], melds):
            waits.append(t)
    return waits


def is_tenpai(hand_tiles: Iterable[str], melds=()) -> bool:
    return shanten(hand_tiles, melds) == 0


def is_furiten(hand_tiles: Iterable[str], melds=(), own_discards: Iterable[str] = ()) -> bool:
    """Discard furiten: any wait is already in the player's own river."""
    waits = set(tenpai_tiles(hand_tiles, melds))
    if not waits:
        return False
    discard_values = {normalize(t) for t in own_discards}
    return bool(waits & discard_values)


__all__ = [
    "is_agari",
    "is_standard_agari",
    "is_chiitoi",
    "is_kokushi",
    "is_tenpai",
    "is_furiten",
    "shanten",
    "tenpai_tiles",
]
