"""Versioned action schema (DATA_SPEC section 6).

Eight legacy action types plus typed kyuushu are supported. ``Action`` is an immutable dataclass; the
``consumed`` tuple is normalised to sorted order in ``__post_init__`` so two
semantically identical actions compare equal regardless of the order the tiles
were listed in the source event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

#: Legacy action schema version. V2 datasets/checkpoints record their schema
#: version explicitly while sharing this typed action object.
ACTION_SCHEMA_VERSION = "action-v1"
ACTION_SCHEMA_VERSION_V2 = "action-v2"


class ActionType(str, Enum):
    DISCARD = "discard"
    CHI = "chi"
    PON = "pon"
    KAN = "kan"
    RIICHI = "riichi"
    RON = "ron"
    TSUMO = "tsumo"
    PASS = "pass"
    KYUUSHU_KYUHAI = "kyuushu_kyuuhai"


#: Kan sub-kinds carried by ``Action.kan_kind`` for ``ActionType.KAN``.
KAN_KINDS = frozenset({"daiminkan", "ankan", "kakan"})


@dataclass(frozen=True)
class Action:
    """A mahjong action.

    Field usage per type:

    - DISCARD/RIICHI: ``tile`` = discarded tile.
    - CHI/PON: ``tile`` = claimed tile, ``consumed`` = 2 tiles from hand,
      ``target`` = discarder.
    - KAN: ``kan_kind`` in {daiminkan, ankan, kakan}; ``tile`` = claimed tile
      (daiminkan) or added tile (kakan), ``consumed`` = 3 (daiminkan/kakan) or
      4 (ankan) tiles.
    - RON: ``target`` = discarder.
    - TSUMO/PASS: no fields.
    """

    type: ActionType
    tile: str | None = None
    consumed: tuple[str, ...] = ()
    target: int | None = None
    kan_kind: str | None = None

    def __post_init__(self) -> None:
        if self.consumed and tuple(sorted(self.consumed)) != self.consumed:
            object.__setattr__(self, "consumed", tuple(sorted(self.consumed)))
        if self.kan_kind is not None and self.kan_kind not in KAN_KINDS:
            raise ValueError(f"invalid kan_kind {self.kan_kind!r}")

    def to_dict(self, schema_version: str = ACTION_SCHEMA_VERSION) -> dict:
        return {
            "schema_version": schema_version,
            "type": self.type.value,
            "tile": self.tile,
            "consumed": list(self.consumed),
            "target": self.target,
            "kan_kind": self.kan_kind,
        }

    def __repr__(self) -> str:  # compact, deterministic
        parts = [self.type.value]
        if self.tile is not None:
            parts.append(f"tile={self.tile}")
        if self.consumed:
            parts.append(f"consumed={list(self.consumed)}")
        if self.target is not None:
            parts.append(f"target={self.target}")
        if self.kan_kind is not None:
            parts.append(f"kan={self.kan_kind}")
        return "Action(" + ", ".join(parts) + ")"
