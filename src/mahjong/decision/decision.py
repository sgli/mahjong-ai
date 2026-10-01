"""Decision point and decision sample data classes (DATA_SPEC section 7)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .action import Action
from .observation import PlayerObservation


@dataclass(frozen=True)
class DecisionPoint:
    """A decision window for one seat at one event index."""

    round_id: str
    seat: int
    kind: str  # "discard" (after draw / after chi/pon) or "response" (after discard)
    step: int  # event index (0-based) that opened the window


@dataclass(frozen=True)
class DecisionSample:
    """One (observation, legal_actions, human action) triple."""

    game_id: str | None
    round_id: str
    player_id: int
    observation: PlayerObservation
    legal_actions: tuple[Action, ...]
    action: Action
    metadata: dict = field(default_factory=dict)
    decision_point: DecisionPoint | None = None
