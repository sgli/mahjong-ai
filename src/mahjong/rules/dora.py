"""Dora (宝牌) counting — pure functions.

Tenhou dora mapping:

- suited indicator ``n`` -> dora ``n+1`` (``9`` wraps to ``1``);
- wind indicator ``E -> S -> W -> N -> E``;
- dragon indicator ``P(白) -> F(發) -> C(中) -> P(白)``.

Red fives (``5mr/5pr/5sr``) each count as one additional dora (aka).
"""

from __future__ import annotations

from .tiles import normalize, suit_of, rank_of

_WIND_NEXT = {"E": "S", "S": "W", "W": "N", "N": "E"}
_DRAGON_NEXT = {"P": "F", "F": "C", "C": "P"}
_AKA = ("5mr", "5pr", "5sr")


def dora_tile(indicator: str) -> str:
    """Return the tile a dora indicator points to."""
    ind = normalize(indicator)
    suit = suit_of(ind)
    if suit is not None:
        rank = rank_of(ind)
        nxt = rank + 1 if rank < 9 else 1
        return f"{nxt}{suit}"
    if ind in _WIND_NEXT:
        return _WIND_NEXT[ind]
    if ind in _DRAGON_NEXT:
        return _DRAGON_NEXT[ind]
    raise ValueError(f"invalid dora indicator {indicator!r}")


def _all_tiles(hand, melds) -> list[str]:
    tiles = [normalize(t) for t in hand]
    for meld in melds:
        tiles.extend(normalize(t) for t in meld.tiles)
    return tiles


def count_dora(
    hand,
    melds,
    dora_indicators=(),
    ura_indicators=(),
    *,
    include_aka: bool = True,
) -> int:
    """Return the number of dora han (table + ura + aka) in the winning hand."""
    tiles = _all_tiles(hand, melds)
    total = 0
    for indicator in tuple(dora_indicators) + tuple(ura_indicators):
        target = dora_tile(indicator)
        total += tiles.count(target)
    if include_aka:
        for aka in _AKA:
            total += sum(1 for t in hand if t == aka)
            for meld in melds:
                total += sum(1 for t in meld.tiles if t == aka)
    return total


def aka_dora_count(hand, melds) -> int:
    """Number of red fives in hand + melds (each = 1 dora)."""
    total = 0
    for aka in _AKA:
        total += sum(1 for t in hand if t == aka)
        for meld in melds:
            total += sum(1 for t in meld.tiles if t == aka)
    return total


__all__ = ["aka_dora_count", "count_dora", "dora_tile"]
