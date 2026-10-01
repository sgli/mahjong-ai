"""Environment state (full, hidden-information internal state).

The environment keeps opponents' concealed hands, the wall and ura indicators
internally; :meth:`MahjongEnv.observation` projects only the visible subset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..replay.state import Meld
from .wall import Wall

#: Phases of the turn structure.
PHASE_DISCARD = "discard"  # current player must discard / riichi / tsumo / kan
PHASE_RESPONSE = "response"  # responders to a discard choose ron/pon/chi/kan/pass
PHASE_CHANKAN = "chankan"  # responders to a kakan choose ron/pass
PHASE_ENDED = "ended"


@dataclass
class Player:
    seat: int
    hand: list[str] = field(default_factory=list)
    melds: list[Meld] = field(default_factory=list)
    discards: list[str] = field(default_factory=list)
    riichi: bool = False
    ippatsu: bool = False
    temporary_furiten: bool = False  # 临时振听：立直见逃后直到下次摸牌前不可荣和
    discard_called: bool = False  # 该玩家舍牌是否被他人 吃/碰/大明杠（用于流局满贯）
    score: int = 25000
    last_drawn: str | None = None  # the tile just drawn (for tsumogiri / riichi discard)
    drawn_this_turn: bool = False  # True after a draw (tsumo/rinchan/deal), False after a call


@dataclass
class SeatStats:
    """Read-only per-seat game statistics (Phase 9 benchmark)."""

    win_count: int = 0  # 和牌次数
    tsumo_count: int = 0  # 自摸次数
    ron_count: int = 0  # 荣和次数
    dealt_in_count: int = 0  # 放铳次数
    riichi_count: int = 0  # 立直次数
    call_count: int = 0  # 副露次数（吃/碰/杠）
    win_points: int = 0  # 和牌所得点数合计


@dataclass
class EnvState:
    round_wind: str = "E"
    kyoku: int = 1
    honba: int = 0
    kyotaku: int = 0
    oya: int = 0
    scores: list[int] = field(default_factory=lambda: [25000, 25000, 25000, 25000])
    players: list[Player] = field(default_factory=lambda: [Player(i) for i in range(4)])
    wall: Wall | None = None
    dora_indicators: list[str] = field(default_factory=list)
    ura_indicators: tuple[str, ...] = ()
    turn: int = 0
    phase: str = PHASE_DISCARD
    # response-phase context
    discarder: int | None = None
    discard_tile: str | None = None
    pending_responders: list[int] = field(default_factory=list)
    response_choices: dict[int, object] = field(default_factory=dict)
    # game end
    game_over: bool = False
    final_ranks: list[int] | None = None
    # first draw flag for each player (kyuushu kyuuhai / double riichi detection)
    first_draw_done: list[bool] = field(default_factory=lambda: [False] * 4)
    # tenpai flags for ryukyoku settlement
    tenpai_flags: list[bool] = field(default_factory=lambda: [False] * 4)
    # per-seat benchmark statistics (read-only, accumulated over the whole game)
    stats: list[SeatStats] = field(default_factory=lambda: [SeatStats() for _ in range(4)])

    def menzen(self, seat: int) -> bool:
        return not any(m.kind in ("chi", "pon", "daiminkan", "kakan") for m in self.players[seat].melds)

    def hand_size_after_draw(self, seat: int) -> int:
        return 14 - 3 * len(self.players[seat].melds)

    def hand_size_at_rest(self, seat: int) -> int:
        return 13 - 3 * len(self.players[seat].melds)


__all__ = [
    "PHASE_CHANKAN",
    "PHASE_DISCARD",
    "PHASE_ENDED",
    "PHASE_RESPONSE",
    "EnvState",
    "Player",
    "SeatStats",
]
