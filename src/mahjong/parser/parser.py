"""Mjai JSONL parser.

The raw ``.mjai.json`` data is JSON Lines: one JSON object per line, where the
whole file is a single game (``start_game`` ... ``end_game``).  This module
converts each line into a typed :class:`Event` and performs basic structural
checks (JSON validity, required fields, field types).

It does **not** implement mahjong state transitions — that belongs to the
Replay Engine (a later phase).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Callable

from .events import (
    Ankan,
    Chi,
    Dahai,
    Daiminkan,
    Dora,
    EndGame,
    EndKyoku,
    Event,
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


class MjaiParseError(Exception):
    """Raised when a line cannot be parsed into a typed event."""

    def __init__(self, message: str, *, line_no: int | None = None) -> None:
        self.line_no = line_no
        super().__init__(message)

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.line_no is None:
            return super().__str__()
        return f"line {self.line_no}: {super().__str__()}"


def _require(obj: Mapping[str, Any], key: str) -> Any:
    if key not in obj:
        raise MjaiParseError(f"missing required field {key!r}")
    return obj[key]


def _as_str(obj: Mapping[str, Any], key: str) -> str:
    value = _require(obj, key)
    if not isinstance(value, str):
        raise MjaiParseError(f"field {key!r} must be a string, got {type(value).__name__}")
    return value


def _as_int(obj: Mapping[str, Any], key: str) -> int:
    value = _require(obj, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MjaiParseError(f"field {key!r} must be an int, got {type(value).__name__}")
    return value


def _as_bool(obj: Mapping[str, Any], key: str) -> bool:
    value = _require(obj, key)
    if not isinstance(value, bool):
        raise MjaiParseError(f"field {key!r} must be a bool, got {type(value).__name__}")
    return value


def _as_str_tuple(obj: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = _require(obj, key)
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise MjaiParseError(f"field {key!r} must be a list of strings")
    return tuple(value)


def _as_int_tuple(obj: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = _require(obj, key)
    if not isinstance(value, list) or not all(
        isinstance(x, int) and not isinstance(x, bool) for x in value
    ):
        raise MjaiParseError(f"field {key!r} must be a list of ints")
    return tuple(value)


def _as_tile_rows(obj: Mapping[str, Any], key: str) -> tuple[tuple[str, ...], ...]:
    value = _require(obj, key)
    if not isinstance(value, list) or not all(
        isinstance(row, list) and all(isinstance(x, str) for x in row) for row in value
    ):
        raise MjaiParseError(f"field {key!r} must be a list of tile lists")
    return tuple(tuple(row) for row in value)


def _parse_start_game(obj: Mapping[str, Any]) -> StartGame:
    names = _as_str_tuple(obj, "names")
    if len(names) != 4:
        raise MjaiParseError("field 'names' must have exactly 4 entries")
    return StartGame(
        names=(names[0], names[1], names[2], names[3]),
        kyoku_first=_as_int(obj, "kyoku_first"),
        aka_flag=_as_bool(obj, "aka_flag"),
    )


def _parse_start_kyoku(obj: Mapping[str, Any]) -> StartKyoku:
    scores = _as_int_tuple(obj, "scores")
    if len(scores) != 4:
        raise MjaiParseError("field 'scores' must have exactly 4 entries")
    tehais = _as_tile_rows(obj, "tehais")
    if len(tehais) != 4:
        raise MjaiParseError("field 'tehais' must have exactly 4 rows")
    return StartKyoku(
        bakaze=_as_str(obj, "bakaze"),
        dora_marker=_as_str(obj, "dora_marker"),
        kyoku=_as_int(obj, "kyoku"),
        honba=_as_int(obj, "honba"),
        kyotaku=_as_int(obj, "kyotaku"),
        oya=_as_int(obj, "oya"),
        scores=(scores[0], scores[1], scores[2], scores[3]),
        tehais=(tehais[0], tehais[1], tehais[2], tehais[3]),
    )


def _parse_tsumo(obj: Mapping[str, Any]) -> Tsumo:
    return Tsumo(actor=_as_int(obj, "actor"), pai=_as_str(obj, "pai"))


def _parse_dahai(obj: Mapping[str, Any]) -> Dahai:
    return Dahai(
        actor=_as_int(obj, "actor"),
        pai=_as_str(obj, "pai"),
        tsumogiri=_as_bool(obj, "tsumogiri"),
    )


def _parse_chi(obj: Mapping[str, Any]) -> Chi:
    consumed = _as_str_tuple(obj, "consumed")
    if len(consumed) != 2:
        raise MjaiParseError("field 'consumed' for chi must have exactly 2 entries")
    return Chi(
        actor=_as_int(obj, "actor"),
        target=_as_int(obj, "target"),
        pai=_as_str(obj, "pai"),
        consumed=(consumed[0], consumed[1]),
    )


def _parse_pon(obj: Mapping[str, Any]) -> Pon:
    consumed = _as_str_tuple(obj, "consumed")
    if len(consumed) != 2:
        raise MjaiParseError("field 'consumed' for pon must have exactly 2 entries")
    return Pon(
        actor=_as_int(obj, "actor"),
        target=_as_int(obj, "target"),
        pai=_as_str(obj, "pai"),
        consumed=(consumed[0], consumed[1]),
    )


def _parse_daiminkan(obj: Mapping[str, Any]) -> Daiminkan:
    consumed = _as_str_tuple(obj, "consumed")
    if len(consumed) != 3:
        raise MjaiParseError("field 'consumed' for daiminkan must have exactly 3 entries")
    return Daiminkan(
        actor=_as_int(obj, "actor"),
        target=_as_int(obj, "target"),
        pai=_as_str(obj, "pai"),
        consumed=(consumed[0], consumed[1], consumed[2]),
    )


def _parse_ankan(obj: Mapping[str, Any]) -> Ankan:
    consumed = _as_str_tuple(obj, "consumed")
    if len(consumed) != 4:
        raise MjaiParseError("field 'consumed' for ankan must have exactly 4 entries")
    return Ankan(
        actor=_as_int(obj, "actor"),
        consumed=(consumed[0], consumed[1], consumed[2], consumed[3]),
    )


def _parse_kakan(obj: Mapping[str, Any]) -> Kakan:
    consumed = _as_str_tuple(obj, "consumed")
    if len(consumed) != 3:
        raise MjaiParseError("field 'consumed' for kakan must have exactly 3 entries")
    return Kakan(
        actor=_as_int(obj, "actor"),
        pai=_as_str(obj, "pai"),
        consumed=(consumed[0], consumed[1], consumed[2]),
    )


def _parse_dora(obj: Mapping[str, Any]) -> Dora:
    return Dora(dora_marker=_as_str(obj, "dora_marker"))


def _parse_reach(obj: Mapping[str, Any]) -> Reach:
    return Reach(actor=_as_int(obj, "actor"))


def _parse_reach_accepted(obj: Mapping[str, Any]) -> ReachAccepted:
    return ReachAccepted(actor=_as_int(obj, "actor"))


def _parse_hora(obj: Mapping[str, Any]) -> Hora:
    deltas = _as_int_tuple(obj, "deltas")
    if len(deltas) != 4:
        raise MjaiParseError("field 'deltas' for hora must have exactly 4 entries")
    ura_markers = _as_str_tuple(obj, "ura_markers")
    return Hora(
        actor=_as_int(obj, "actor"),
        target=_as_int(obj, "target"),
        deltas=(deltas[0], deltas[1], deltas[2], deltas[3]),
        ura_markers=ura_markers,
    )


def _parse_ryukyoku(obj: Mapping[str, Any]) -> Ryukyoku:
    deltas = _as_int_tuple(obj, "deltas")
    if len(deltas) != 4:
        raise MjaiParseError("field 'deltas' for ryukyoku must have exactly 4 entries")
    return Ryukyoku(deltas=(deltas[0], deltas[1], deltas[2], deltas[3]))


def _parse_end_kyoku(obj: Mapping[str, Any]) -> EndKyoku:
    return EndKyoku()


def _parse_end_game(obj: Mapping[str, Any]) -> EndGame:
    return EndGame()


_EVENT_PARSERS: dict[str, Callable[[Mapping[str, Any]], Event]] = {
    "start_game": _parse_start_game,
    "start_kyoku": _parse_start_kyoku,
    "tsumo": _parse_tsumo,
    "dahai": _parse_dahai,
    "chi": _parse_chi,
    "pon": _parse_pon,
    "daiminkan": _parse_daiminkan,
    "ankan": _parse_ankan,
    "kakan": _parse_kakan,
    "dora": _parse_dora,
    "reach": _parse_reach,
    "reach_accepted": _parse_reach_accepted,
    "hora": _parse_hora,
    "ryukyoku": _parse_ryukyoku,
    "end_kyoku": _parse_end_kyoku,
    "end_game": _parse_end_game,
}


def parse_event(obj: Mapping[str, Any]) -> Event:
    """Convert a raw event object (dict) into a typed event."""
    if not isinstance(obj, Mapping):
        raise MjaiParseError("event must be a JSON object")
    event_type = obj.get("type")
    if not isinstance(event_type, str):
        raise MjaiParseError("event is missing a string 'type' field")
    parser = _EVENT_PARSERS.get(event_type)
    if parser is None:
        raise MjaiParseError(f"unknown event type {event_type!r}")
    return parser(obj)


class MjaiParser:
    """Streaming parser for ``.mjai.json`` (JSON Lines) files."""

    def parse_line(self, line: str, line_no: int | None = None) -> Event:
        """Parse a single JSONL line into a typed event."""
        stripped = line.strip()
        if not stripped:
            raise MjaiParseError("empty line", line_no=line_no)
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise MjaiParseError(f"invalid JSON: {exc.msg}", line_no=line_no) from exc
        try:
            return parse_event(obj)
        except MjaiParseError as exc:
            if exc.line_no is None:
                exc.line_no = line_no
            raise

    def parse_file(
        self,
        path: str | Path,
        *,
        on_error: Callable[[MjaiParseError], None] | None = None,
    ) -> Iterator[Event]:
        """Stream events from a file.

        By default the first malformed line raises :class:`MjaiParseError`.
        When ``on_error`` is provided it is called for each bad line and the
        line is skipped, so a single bad line does not stop the whole run.
        """
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                try:
                    yield self.parse_line(line, line_no)
                except MjaiParseError as exc:
                    if on_error is None:
                        raise
                    on_error(exc)
