"""Structural validation for parsed Mjai events.

Phase 1 performs *basic structural* checks only:

- per-event field ranges and tile well-formedness;
- stream-level game / kyoku boundaries.

Full consistency checks (hand tile counts, score settlement, turn ordering)
belong to the Replay Engine and are intentionally out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .events import (
    Ankan,
    Chi,
    Dahai,
    Daiminkan,
    Dora,
    EndGame,
    EndKyoku,
    Event,
    EventType,
    Hora,
    Kakan,
    Pon,
    Reach,
    ReachAccepted,
    Ryukyoku,
    StartGame,
    StartKyoku,
    Tsumo,
)

#: All well-formed tile strings, including the three red fives.
VALID_TILES: frozenset[str] = frozenset(
    [f"{n}{s}" for s in "mps" for n in "123456789"]
    + ["5mr", "5pr", "5sr"]
    + ["E", "S", "W", "N", "P", "F", "C"]
)

WINDS: frozenset[str] = frozenset({"E", "S", "W", "N"})


@dataclass(frozen=True)
class ValidationIssue:
    category: str
    message: str
    line_no: int | None = None


def _seat_issue(value: int, name: str) -> str | None:
    if not (0 <= value <= 3):
        return f"{name}={value} is out of range 0..3"
    return None


def _tile_issue(tile: str, name: str) -> str | None:
    if tile not in VALID_TILES:
        return f"{name}={tile!r} is not a valid tile"
    return None


def _tiles_issue(tiles: tuple[str, ...], name: str) -> list[str]:
    issues = []
    for tile in tiles:
        msg = _tile_issue(tile, name)
        if msg:
            issues.append(msg)
    return issues


def _collect_actor_target(event: object, actor: int, target: int | None) -> list[str]:
    issues = []
    msg = _seat_issue(actor, "actor")
    if msg:
        issues.append(msg)
    if target is not None:
        msg = _seat_issue(target, "target")
        if msg:
            issues.append(msg)
    return issues


def validate_event(event: Event) -> list[ValidationIssue]:
    """Return structural issues for a single event (empty list means valid)."""
    issues: list[ValidationIssue] = []

    if isinstance(event, StartGame):
        if event.kyoku_first < 0:
            issues.append(ValidationIssue("event", f"kyoku_first={event.kyoku_first} is negative"))
        for i, name in enumerate(event.names):
            if not name:
                issues.append(ValidationIssue("event", f"names[{i}] is empty"))

    elif isinstance(event, StartKyoku):
        if event.bakaze not in WINDS:
            issues.append(ValidationIssue("event", f"bakaze={event.bakaze!r} is not a wind"))
        if not (1 <= event.kyoku <= 4):
            issues.append(ValidationIssue("event", f"kyoku={event.kyoku} is out of range 1..4"))
        if event.honba < 0:
            issues.append(ValidationIssue("event", f"honba={event.honba} is negative"))
        if event.kyotaku < 0:
            issues.append(ValidationIssue("event", f"kyotaku={event.kyotaku} is negative"))
        msg = _seat_issue(event.oya, "oya")
        if msg:
            issues.append(ValidationIssue("event", msg))
        msg = _tile_issue(event.dora_marker, "dora_marker")
        if msg:
            issues.append(ValidationIssue("event", msg))
        for seat, hand in enumerate(event.tehais):
            if len(hand) != 13:
                issues.append(
                    ValidationIssue("event", f"tehais[{seat}] has {len(hand)} tiles, expected 13")
                )
            issues.extend(
                ValidationIssue("event", m) for m in _tiles_issue(hand, f"tehais[{seat}]")
            )

    elif isinstance(event, Tsumo):
        issues.extend(
            ValidationIssue("event", m) for m in _collect_actor_target(event, event.actor, None)
        )
        msg = _tile_issue(event.pai, "pai")
        if msg:
            issues.append(ValidationIssue("event", msg))

    elif isinstance(event, Dahai):
        issues.extend(
            ValidationIssue("event", m) for m in _collect_actor_target(event, event.actor, None)
        )
        msg = _tile_issue(event.pai, "pai")
        if msg:
            issues.append(ValidationIssue("event", msg))

    elif isinstance(event, (Chi, Pon, Daiminkan)):
        issues.extend(
            ValidationIssue("event", m)
            for m in _collect_actor_target(event, event.actor, event.target)
        )
        msg = _tile_issue(event.pai, "pai")
        if msg:
            issues.append(ValidationIssue("event", msg))
        issues.extend(
            ValidationIssue("event", m) for m in _tiles_issue(event.consumed, "consumed")
        )

    elif isinstance(event, Ankan):
        issues.extend(
            ValidationIssue("event", m) for m in _collect_actor_target(event, event.actor, None)
        )
        issues.extend(
            ValidationIssue("event", m) for m in _tiles_issue(event.consumed, "consumed")
        )

    elif isinstance(event, Kakan):
        issues.extend(
            ValidationIssue("event", m) for m in _collect_actor_target(event, event.actor, None)
        )
        msg = _tile_issue(event.pai, "pai")
        if msg:
            issues.append(ValidationIssue("event", msg))
        issues.extend(
            ValidationIssue("event", m) for m in _tiles_issue(event.consumed, "consumed")
        )

    elif isinstance(event, Dora):
        msg = _tile_issue(event.dora_marker, "dora_marker")
        if msg:
            issues.append(ValidationIssue("event", msg))

    elif isinstance(event, (Reach, ReachAccepted)):
        issues.extend(
            ValidationIssue("event", m) for m in _collect_actor_target(event, event.actor, None)
        )

    elif isinstance(event, Hora):
        issues.extend(
            ValidationIssue("event", m)
            for m in _collect_actor_target(event, event.actor, event.target)
        )
        issues.extend(
            ValidationIssue("event", m) for m in _tiles_issue(event.ura_markers, "ura_markers")
        )

    elif isinstance(event, (Ryukyoku, EndKyoku, EndGame)):
        pass

    return issues


@dataclass
class GameValidator:
    """Stateful validator for the event stream of a single game file."""

    issues: list[ValidationIssue] = field(default_factory=list)
    _start_game_count: int = field(default=0, init=False)
    _end_game_count: int = field(default=0, init=False)
    _start_kyoku_count: int = field(default=0, init=False)
    _end_kyoku_count: int = field(default=0, init=False)
    _seen_start_game: bool = field(default=False, init=False)
    _seen_end_game: bool = field(default=False, init=False)
    _kyoku_open: bool = field(default=False, init=False)

    def _issue(self, message: str, line_no: int | None) -> None:
        self.issues.append(ValidationIssue("game", message, line_no))

    def add(self, event: Event, line_no: int | None = None) -> None:
        t = event.type

        if t is EventType.START_GAME:
            self._start_game_count += 1
            if self._seen_start_game:
                self._issue("multiple start_game events", line_no)
            if self._kyoku_open:
                self._issue("start_game while a kyoku is still open", line_no)
            self._seen_start_game = True

        elif t is EventType.START_KYOKU:
            self._start_kyoku_count += 1
            if not self._seen_start_game:
                self._issue("start_kyoku before start_game", line_no)
            if self._kyoku_open:
                self._issue("start_kyoku while previous kyoku is not ended", line_no)
            self._kyoku_open = True

        elif t is EventType.END_KYOKU:
            self._end_kyoku_count += 1
            if not self._kyoku_open:
                self._issue("end_kyoku without a matching start_kyoku", line_no)
            self._kyoku_open = False

        elif t is EventType.END_GAME:
            self._end_game_count += 1
            if self._kyoku_open:
                self._issue("end_game while a kyoku is still open", line_no)
            self._seen_end_game = True

        else:
            if not self._kyoku_open:
                self._issue(f"{t.value} appears outside a kyoku", line_no)

        if self._seen_end_game and t is not EventType.END_GAME:
            self._issue(f"{t.value} appears after end_game", line_no)

    def finish(self) -> None:
        if not self._seen_start_game:
            self._issue("no start_game event", None)
        if self._seen_start_game and not self._seen_end_game:
            self._issue("no end_game event", None)
        if self._kyoku_open:
            self._issue("final kyoku is not ended", None)
        if self._start_kyoku_count != self._end_kyoku_count:
            self._issue(
                f"start_kyoku count ({self._start_kyoku_count}) != "
                f"end_kyoku count ({self._end_kyoku_count})",
                None,
            )
