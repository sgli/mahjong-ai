"""Tests for structural validation."""

import pytest

from mahjong.parser import GameValidator, parse_event, validate_event
from mahjong.parser.events import (
    Ankan,
    Dahai,
    EndGame,
    EndKyoku,
    StartGame,
    StartKyoku,
    Tsumo,
)


def _issues(event):
    return [i.message for i in validate_event(event)]


def test_valid_tsumo_has_no_issues():
    assert _issues(Tsumo(actor=0, pai="6m")) == []


def test_actor_out_of_range():
    assert "out of range" in _issues(Tsumo(actor=4, pai="6m"))[0]


def test_invalid_tile():
    assert "not a valid tile" in _issues(Tsumo(actor=0, pai="0m"))[0]


def test_red_five_is_valid_tile():
    assert _issues(Tsumo(actor=0, pai="5mr")) == []


def test_start_kyoku_hand_length():
    event = StartKyoku(
        bakaze="E",
        dora_marker="W",
        kyoku=1,
        honba=0,
        kyotaku=0,
        oya=0,
        scores=(25000, 25000, 25000, 25000),
        tehais=(("1m",) * 12, ("2m",) * 13, ("3m",) * 13, ("4m",) * 13),
    )
    messages = _issues(event)
    assert any("expected 13" in m for m in messages)


def test_start_kyoku_bad_bakaze():
    event = StartKyoku(
        bakaze="X",
        dora_marker="W",
        kyoku=1,
        honba=0,
        kyotaku=0,
        oya=0,
        scores=(25000, 25000, 25000, 25000),
        tehais=(("1m",) * 13, ("2m",) * 13, ("3m",) * 13, ("4m",) * 13),
    )
    assert any("not a wind" in m for m in _issues(event))


def test_ankan_consumed_tile_validity():
    assert _issues(Ankan(actor=0, consumed=("E", "E", "E", "E"))) == []
    assert any("not a valid tile" in m for m in _issues(Ankan(actor=0, consumed=("E", "E", "E", "Z"))))


def _valid_kyoku_hand():
    return (("1m",) * 13, ("2m",) * 13, ("3m",) * 13, ("4m",) * 13)


def test_game_validator_accepts_valid_stream():
    v = GameValidator()
    v.add(StartGame(names=("a", "b", "c", "d"), kyoku_first=0, aka_flag=True), 1)
    v.add(StartKyoku("E", "W", 1, 0, 0, 0, (25000, 25000, 25000, 25000), _valid_kyoku_hand()), 2)
    v.add(Tsumo(actor=0, pai="6m"), 3)
    v.add(Dahai(actor=0, pai="6m", tsumogiri=True), 4)
    v.add(EndKyoku(), 5)
    v.add(EndGame(), 6)
    v.finish()
    assert v.issues == []


def test_game_validator_missing_start_game():
    v = GameValidator()
    v.add(Tsumo(actor=0, pai="6m"), 1)
    v.finish()
    messages = [i.message for i in v.issues]
    assert any("before start_game" in m or "no start_game" in m for m in messages)


def test_game_validator_start_kyoku_before_start_game():
    v = GameValidator()
    v.add(StartKyoku("E", "W", 1, 0, 0, 0, (25000, 25000, 25000, 25000), _valid_kyoku_hand()), 1)
    v.finish()
    assert any("start_kyoku before start_game" in i.message for i in v.issues)


def test_game_validator_dahai_outside_kyoku():
    v = GameValidator()
    v.add(StartGame(names=("a", "b", "c", "d"), kyoku_first=0, aka_flag=True), 1)
    v.add(Dahai(actor=0, pai="6m", tsumogiri=True), 2)
    v.finish()
    assert any("outside a kyoku" in i.message for i in v.issues)


def test_game_validator_kyoku_balance():
    v = GameValidator()
    v.add(StartGame(names=("a", "b", "c", "d"), kyoku_first=0, aka_flag=True), 1)
    v.add(StartKyoku("E", "W", 1, 0, 0, 0, (25000, 25000, 25000, 25000), _valid_kyoku_hand()), 2)
    v.add(EndGame(), 3)
    v.finish()
    assert any("final kyoku is not ended" in i.message for i in v.issues)


def test_game_validator_event_after_end_game():
    v = GameValidator()
    v.add(StartGame(names=("a", "b", "c", "d"), kyoku_first=0, aka_flag=True), 1)
    v.add(StartKyoku("E", "W", 1, 0, 0, 0, (25000, 25000, 25000, 25000), _valid_kyoku_hand()), 2)
    v.add(EndKyoku(), 3)
    v.add(EndGame(), 4)
    v.add(Tsumo(actor=0, pai="6m"), 5)
    v.finish()
    assert any("after end_game" in i.message for i in v.issues)
