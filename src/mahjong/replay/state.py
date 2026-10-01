"""Replay state data classes (Phase 2: Replay Engine).

These are immutable snapshots produced by :class:`ReplayEngine` after each
event.  The fields follow ``docs/DATA_SPEC.md`` section 4, with a few
documented extensions that later phases (Decision Extraction) will reuse:

- ``Meld.from_`` / ``Meld.called`` record where a called tile came from;
- ``PlayerState.discard_tsumogiri`` keeps the ``tsumogiri`` flag per discard;
- ``ReplayState.names`` / ``aka_flag`` / ``kyoku_first`` mirror ``start_game``;
- ``ReplayState.step`` / ``event_type`` record which event produced the state;
- ``ReplayState.ura_markers`` keeps the most recent ``hora`` ura markers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Meld:
    """A called set/kan.  ``tiles`` is the full meld (3 tiles for chi/pon,
    4 tiles for kan).  ``from_`` is the seat the called tile came from and
    ``called`` is that tile itself (both ``None`` for a closed kan)."""

    kind: str  # "chi" | "pon" | "daiminkan" | "ankan" | "kakan"
    tiles: tuple[str, ...]
    from_: int | None = None
    called: str | None = None

    @property
    def is_kan(self) -> bool:
        return self.kind in {"daiminkan", "ankan", "kakan"}


@dataclass(frozen=True)
class PlayerState:
    """One player's view at a point in the replay."""

    seat: int
    hand: tuple[str, ...] = ()
    melds: tuple[Meld, ...] = ()
    discards: tuple[str, ...] = ()
    discard_tsumogiri: tuple[bool, ...] = ()
    riichi: bool = False
    ippatsu: bool = False
    score: int = 0


@dataclass(frozen=True)
class ReplayState:
    """A full game state snapshot after applying one Mjai event."""

    game_id: str | None
    round_id: str  # f"{bakaze}{kyoku}", e.g. "E1"; empty before start_kyoku
    bakaze: str
    kyoku: int
    honba: int
    kyotaku: int
    oya: int
    scores: tuple[int, int, int, int]
    players: tuple[PlayerState, PlayerState, PlayerState, PlayerState]
    dora_markers: tuple[str, ...]
    ura_markers: tuple[str, ...]
    turn: int
    names: tuple[str, str, str, str] = ("", "", "", "")
    aka_flag: bool = False
    kyoku_first: int = 0
    step: int = 0
    event_type: str = ""
