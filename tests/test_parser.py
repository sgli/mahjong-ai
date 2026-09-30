"""Tests for the Mjai JSONL parser."""

import json

import pytest

from mahjong.parser import MjaiParseError, MjaiParser, parse_event
from mahjong.parser.events import (
    Ankan,
    Chi,
    Dahai,
    Daiminkan,
    Dora,
    EndGame,
    EndKyoku,
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


def test_parse_start_game():
    event = parse_event(
        {"type": "start_game", "names": ["a", "b", "c", "d"], "kyoku_first": 0, "aka_flag": True}
    )
    assert isinstance(event, StartGame)
    assert event.names == ("a", "b", "c", "d")
    assert event.kyoku_first == 0
    assert event.aka_flag is True
    assert event.type is EventType.START_GAME


def test_parse_start_kyoku():
    raw = {
        "type": "start_kyoku",
        "bakaze": "E",
        "dora_marker": "W",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [["1m"] * 13, ["2m"] * 13, ["3m"] * 13, ["4m"] * 13],
    }
    event = parse_event(raw)
    assert isinstance(event, StartKyoku)
    assert event.bakaze == "E"
    assert event.dora_marker == "W"
    assert event.kyoku == 1
    assert event.scores == (25000, 25000, 25000, 25000)
    assert len(event.tehais) == 4
    assert all(len(hand) == 13 for hand in event.tehais)


@pytest.mark.parametrize(
    "raw, cls",
    [
        ({"type": "tsumo", "actor": 0, "pai": "6m"}, Tsumo),
        ({"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": False}, Dahai),
        ({"type": "chi", "actor": 2, "target": 1, "pai": "8p", "consumed": ["6p", "7p"]}, Chi),
        ({"type": "pon", "actor": 1, "target": 3, "pai": "F", "consumed": ["F", "F"]}, Pon),
        (
            {"type": "daiminkan", "actor": 0, "target": 3, "pai": "1s", "consumed": ["1s", "1s", "1s"]},
            Daiminkan,
        ),
        ({"type": "ankan", "actor": 1, "consumed": ["E", "E", "E", "E"]}, Ankan),
        ({"type": "kakan", "actor": 3, "pai": "P", "consumed": ["P", "P", "P"]}, Kakan),
        ({"type": "dora", "dora_marker": "S"}, Dora),
        ({"type": "reach", "actor": 1}, Reach),
        ({"type": "reach_accepted", "actor": 1}, ReachAccepted),
        (
            {
                "type": "hora",
                "actor": 2,
                "target": 3,
                "deltas": [0, 0, 1000, -1000],
                "ura_markers": [],
            },
            Hora,
        ),
        ({"type": "ryukyoku", "deltas": [1000, 1000, 1000, -3000]}, Ryukyoku),
        ({"type": "end_kyoku"}, EndKyoku),
        ({"type": "end_game"}, EndGame),
    ],
)
def test_parse_event_types(raw, cls):
    event = parse_event(raw)
    assert isinstance(event, cls)
    assert event.type.value == raw["type"]


def test_parse_chi_consumed_are_tuple():
    event = parse_event({"type": "chi", "actor": 2, "target": 1, "pai": "8p", "consumed": ["6p", "7p"]})
    assert event.consumed == ("6p", "7p")


def test_parse_hora_ura_markers():
    event = parse_event(
        {
            "type": "hora",
            "actor": 1,
            "target": 1,
            "deltas": [-6100, 19300, -6100, -6100],
            "ura_markers": ["9m"],
        }
    )
    assert event.ura_markers == ("9m",)


def test_unknown_event_type_raises():
    with pytest.raises(MjaiParseError, match="unknown event type"):
        parse_event({"type": "not_an_event"})


def test_missing_type_raises():
    with pytest.raises(MjaiParseError, match="missing a string 'type'"):
        parse_event({"actor": 0})


def test_missing_field_raises():
    with pytest.raises(MjaiParseError, match="missing required field 'pai'"):
        parse_event({"type": "tsumo", "actor": 0})


def test_wrong_field_type_raises():
    with pytest.raises(MjaiParseError, match="must be an int"):
        parse_event({"type": "tsumo", "actor": "0", "pai": "6m"})


def test_wrong_scores_length_raises():
    with pytest.raises(MjaiParseError, match="exactly 4"):
        parse_event(
            {
                "type": "start_kyoku",
                "bakaze": "E",
                "dora_marker": "W",
                "kyoku": 1,
                "honba": 0,
                "kyotaku": 0,
                "oya": 0,
                "scores": [1, 2, 3],
                "tehais": [["1m"] * 13] * 4,
            }
        )


def test_parse_line_with_line_number():
    parser = MjaiParser()
    line = '{"type":"tsumo","actor":0,"pai":"6m"}'
    event = parser.parse_line(line, line_no=3)
    assert isinstance(event, Tsumo)
    assert event.pai == "6m"


def test_parse_line_invalid_json():
    parser = MjaiParser()
    with pytest.raises(MjaiParseError, match="invalid JSON"):
        parser.parse_line("{not json", line_no=1)


def test_parse_line_reports_line_number():
    parser = MjaiParser()
    with pytest.raises(MjaiParseError) as exc_info:
        parser.parse_line('{"type":"tsumo"}', line_no=42)
    assert exc_info.value.line_no == 42


def test_parse_file_streaming(tmp_path):
    path = tmp_path / "game.mjai.json"
    path.write_text(
        "\n".join(
            [
                '{"type":"start_game","names":["a","b","c","d"],"kyoku_first":0,"aka_flag":true}',
                '{"type":"start_kyoku","bakaze":"E","dora_marker":"W","kyoku":1,"honba":0,'
                '"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],'
                '"tehais":' + json.dumps([["1m"] * 13] * 4) + "}",
                '{"type":"tsumo","actor":0,"pai":"6m"}',
                '{"type":"end_kyoku"}',
                '{"type":"end_game"}',
            ]
        ),
        encoding="utf-8",
    )
    parser = MjaiParser()
    events = list(parser.parse_file(path))
    assert [e.type for e in events] == [
        EventType.START_GAME,
        EventType.START_KYOKU,
        EventType.TSUMO,
        EventType.END_KYOKU,
        EventType.END_GAME,
    ]


def test_parse_file_on_error_skips_bad_line(tmp_path):
    path = tmp_path / "bad.mjai.json"
    path.write_text(
        "\n".join(
            [
                '{"type":"start_game","names":["a","b","c","d"],"kyoku_first":0,"aka_flag":true}',
                '{"type":"tsumo"}',  # missing pai
                '{"type":"end_game"}',
            ]
        ),
        encoding="utf-8",
    )
    parser = MjaiParser()
    errors = []
    events = list(parser.parse_file(path, on_error=errors.append))
    assert len(errors) == 1
    assert errors[0].line_no == 2
    assert [e.type for e in events] == [EventType.START_GAME, EventType.END_GAME]


def test_parse_file_raises_by_default(tmp_path):
    path = tmp_path / "bad.mjai.json"
    path.write_text('{"type":"tsumo"}\n', encoding="utf-8")
    parser = MjaiParser()
    with pytest.raises(MjaiParseError):
        list(parser.parse_file(path))
