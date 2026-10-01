"""Tile wall: shuffle, deal, draw, dead wall (dora / ura / kan replacements).

Deterministic: the same seed produces the same wall.  ``aka_flag`` swaps one
``5m/5p/5s`` for the red ``5mr/5pr/5sr`` (136 tiles total).
"""

from __future__ import annotations

import random

from ..rules.tiles import ALL_TILES

_DEAD_SIZE = 14
_N_DORA = 5
_N_URA = 5


class Wall:
    def __init__(self, seed: int, aka_flag: bool = True) -> None:
        rng = random.Random(seed)
        tiles: list[str] = [t for t in ALL_TILES for _ in range(4)]
        if aka_flag:
            for normal, red in (("5m", "5mr"), ("5p", "5pr"), ("5s", "5sr")):
                tiles.remove(normal)
                tiles.append(red)
        rng.shuffle(tiles)

        self.dead = tiles[-_DEAD_SIZE:]  # last 14 tiles
        self.live = tiles[:-_DEAD_SIZE]

        # dead wall layout: [0..4] dora indicators, [5..9] ura indicators,
        # [10..13] kan replacement draws.
        self.dora_indicators: list[str] = [self.dead[0]]
        self.ura_indicators: tuple[str, ...] = tuple(self.dead[_N_DORA : _N_DORA + _N_URA])
        self._kan_draws = self.dead[_N_DORA + _N_URA :]
        self._kan_count = 0

    @property
    def remaining(self) -> int:
        return len(self.live)

    def draw(self) -> str:
        if not self.live:
            raise IndexError("wall is exhausted")
        return self.live.pop()

    def kan_draw(self) -> str:
        """Draw the replacement tile after a kan and flip the next dora."""
        if self._kan_count >= len(self._kan_draws):
            raise IndexError("no kan replacement tiles left")
        tile = self._kan_draws[self._kan_count]
        self._kan_count += 1
        if len(self.dora_indicators) < _N_DORA:
            self.dora_indicators.append(self.dead[len(self.dora_indicators)])
        return tile


__all__ = ["Wall"]
