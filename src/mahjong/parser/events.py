"""Typed Mjai events.

The schema below is derived from the actual ``.mjai.json`` data (JSON Lines,
one event per line) under ``tenhou-houou-2026`` and follows the event types
listed in ``docs/DATA_SPEC.md``.

Each event is an immutable dataclass.  The ``type`` attribute is a class
variable so it is available on every instance without being part of the
constructor arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, TypeAlias


class EventType(str, Enum):
    """Every event type present in the raw data."""

    START_GAME = "start_game"
    START_KYOKU = "start_kyoku"
    TSUMO = "tsumo"
    DAHAI = "dahai"
    CHI = "chi"
    PON = "pon"
    DAIMINKAN = "daiminkan"
    ANKAN = "ankan"
    KAKAN = "kakan"
    DORA = "dora"
    REACH = "reach"
    REACH_ACCEPTED = "reach_accepted"
    HORA = "hora"
    RYUKYOKU = "ryukyoku"
    END_KYOKU = "end_kyoku"
    END_GAME = "end_game"


# The three kan variants.  ``docs/DATA_SPEC.md`` refers to a single ``kan``
# event; the actual data spells it out as ``daiminkan`` / ``ankan`` / ``kakan``.
KAN_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.DAIMINKAN,
        EventType.ANKAN,
        EventType.KAKAN,
    }
)


@dataclass(frozen=True)
class StartGame:
    type: ClassVar[EventType] = EventType.START_GAME
    names: tuple[str, str, str, str]
    kyoku_first: int
    aka_flag: bool


@dataclass(frozen=True)
class StartKyoku:
    type: ClassVar[EventType] = EventType.START_KYOKU
    bakaze: str
    dora_marker: str
    kyoku: int
    honba: int
    kyotaku: int
    oya: int
    scores: tuple[int, int, int, int]
    tehais: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]


@dataclass(frozen=True)
class Tsumo:
    type: ClassVar[EventType] = EventType.TSUMO
    actor: int
    pai: str


@dataclass(frozen=True)
class Dahai:
    type: ClassVar[EventType] = EventType.DAHAI
    actor: int
    pai: str
    tsumogiri: bool


@dataclass(frozen=True)
class Chi:
    type: ClassVar[EventType] = EventType.CHI
    actor: int
    target: int
    pai: str
    consumed: tuple[str, str]


@dataclass(frozen=True)
class Pon:
    type: ClassVar[EventType] = EventType.PON
    actor: int
    target: int
    pai: str
    consumed: tuple[str, str]


@dataclass(frozen=True)
class Daiminkan:
    type: ClassVar[EventType] = EventType.DAIMINKAN
    actor: int
    target: int
    pai: str
    consumed: tuple[str, str, str]


@dataclass(frozen=True)
class Ankan:
    type: ClassVar[EventType] = EventType.ANKAN
    actor: int
    consumed: tuple[str, str, str, str]


@dataclass(frozen=True)
class Kakan:
    type: ClassVar[EventType] = EventType.KAKAN
    actor: int
    pai: str
    consumed: tuple[str, str, str]


@dataclass(frozen=True)
class Dora:
    type: ClassVar[EventType] = EventType.DORA
    dora_marker: str


@dataclass(frozen=True)
class Reach:
    type: ClassVar[EventType] = EventType.REACH
    actor: int


@dataclass(frozen=True)
class ReachAccepted:
    type: ClassVar[EventType] = EventType.REACH_ACCEPTED
    actor: int


@dataclass(frozen=True)
class Hora:
    type: ClassVar[EventType] = EventType.HORA
    actor: int
    target: int
    deltas: tuple[int, int, int, int]
    ura_markers: tuple[str, ...]


@dataclass(frozen=True)
class Ryukyoku:
    type: ClassVar[EventType] = EventType.RYUKYOKU
    deltas: tuple[int, int, int, int]


@dataclass(frozen=True)
class EndKyoku:
    type: ClassVar[EventType] = EventType.END_KYOKU


@dataclass(frozen=True)
class EndGame:
    type: ClassVar[EventType] = EventType.END_GAME


#: Union of all concrete event dataclasses, used for static type checking.
Event: TypeAlias = (
    StartGame
    | StartKyoku
    | Tsumo
    | Dahai
    | Chi
    | Pon
    | Daiminkan
    | Ankan
    | Kakan
    | Dora
    | Reach
    | ReachAccepted
    | Hora
    | Ryukyoku
    | EndKyoku
    | EndGame
)
