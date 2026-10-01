"""Tile primitives for the unified rule core.

Tiles are the same strings used by the Mjai parser: ``1m``..``9m``,
``1p``..``9p``, ``1s``..``9s`` plus honours ``E S W N P F C``, and the three
red fives ``5mr 5pr 5sr``.  For hand-shape logic (agari / shanten / waits /
furiten) the red fives are normalised to their base ``5m/5p/5s`` because being
red does not change the tile's suit or rank; it only matters for scoring (a
later phase).
"""

from __future__ import annotations

#: Base tile types, in a canonical order (suits first, then honours).
ALL_TILES: tuple[str, ...] = tuple(
    [f"{n}{s}" for s in "mps" for n in "123456789"] + list("ESWNPFC")
)

#: Index of each base tile in ``ALL_TILES`` (0..33).
TILE_TO_INDEX: dict[str, int] = {t: i for i, t in enumerate(ALL_TILES)}

#: Red five -> base five.
RED_FIVE: dict[str, str] = {"5mr": "5m", "5pr": "5p", "5sr": "5s"}

#: Honours (no runs can be formed from them).
HONORS: frozenset[str] = frozenset("ESWNPFC")

#: The 13 terminal/honour tiles used by kokushi musou.
ORPHANS: frozenset[str] = frozenset(
    {"1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C"}
)

SUIT_RANKS = "123456789"


def normalize(tile: str) -> str:
    """Return the base tile for a (possibly red) tile string."""
    return RED_FIVE.get(tile, tile)


def suit_of(tile: str) -> str | None:
    """Return ``'m'``/``'p'``/``'s'`` for a suited tile, else ``None``."""
    t = normalize(tile)
    if len(t) >= 2 and t[1] in "mps":
        return t[1]
    return None


def rank_of(tile: str) -> int | None:
    """Return 1..9 for a suited tile, else ``None``."""
    t = normalize(tile)
    if len(t) >= 2 and t[0] in SUIT_RANKS:
        return int(t[0])
    return None


def index_of(tile: str) -> int:
    """Index (0..33) of a tile in ``ALL_TILES`` (red fives map to base 5)."""
    return TILE_TO_INDEX[normalize(tile)]


def to_counts(tiles) -> list[int]:
    """Count base-tile occurrences over the 34 tile types."""
    counts = [0] * 34
    for tile in tiles:
        counts[index_of(tile)] += 1
    return counts


def sorted_tiles(tiles) -> tuple[str, ...]:
    """Return tiles sorted by the canonical tile order (red fives preserved)."""
    return tuple(sorted(tiles, key=lambda t: (index_of(t), t)))
