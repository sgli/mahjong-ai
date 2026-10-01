"""Player-perspective observation (visible state only).

This is the single projection point that guarantees no information leakage:
``PlayerObservation`` is built *exclusively* from public fields of a
``ReplayState`` — own hand, own melds, the four rivers, dora indicators, the
round/score/riichi info and every player's *revealed* melds.  It deliberately
never copies opponents' concealed hands, the wall, ``ura_markers``, or anything
from later events.

Concealed-kan (ankan) visibility note: under Tenhou rules an ankan is placed
with the two middle tiles face up and the two end tiles face down, so *which
tile* the ankan is is public information for all players (the "concealed" in
ankan only means it came from the hand and keeps the hand closed).  Therefore
opponents' ankan ``tiles`` are intentionally included here, not masked.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..replay.state import Meld, ReplayState


@dataclass(frozen=True)
class PlayerObservation:
    """What one player can see when making a decision."""

    seat: int
    bakaze: str
    kyoku: int
    honba: int
    kyotaku: int
    oya: int
    scores: tuple[int, int, int, int]
    riichi: tuple[bool, bool, bool, bool]
    hand: tuple[str, ...]
    melds: tuple[Meld, ...]
    #: opponents' melds by seat; the player's own slot is ``()``.  Ankan tiles
    #: are included (visible to all players under Tenhou rules).
    opponents_melds: tuple[tuple[Meld, ...], tuple[Meld, ...], tuple[Meld, ...], tuple[Meld, ...]]
    discards: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]
    dora_markers: tuple[str, ...]
    turn: int

    @classmethod
    def from_state(cls, state: ReplayState, seat: int) -> "PlayerObservation":
        """Project a full (hidden-info) ReplayState onto one player's view.

        The projection is explicit so a ``ReplayState`` (which contains
        opponents' hands) is never passed straight into an observation.
        """
        players = state.players
        return cls(
            seat=seat,
            bakaze=state.bakaze,
            kyoku=state.kyoku,
            honba=state.honba,
            kyotaku=state.kyotaku,
            oya=state.oya,
            scores=state.scores,
            riichi=tuple(p.riichi for p in players),
            hand=tuple(sorted(players[seat].hand)),
            melds=players[seat].melds,
            opponents_melds=tuple(players[s].melds if s != seat else () for s in range(4)),
            discards=tuple(p.discards for p in players),
            dora_markers=state.dora_markers,
            turn=state.turn,
        )
