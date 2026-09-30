"""Tests for typed event definitions."""

from mahjong.parser.events import (
    EventType,
    KAN_EVENT_TYPES,
    Dahai,
    EndGame,
    Tsumo,
)


def test_event_type_values():
    assert EventType.START_GAME.value == "start_game"
    assert EventType.START_KYOKU.value == "start_kyoku"
    assert EventType.TSUMO.value == "tsumo"
    assert EventType.DAHAI.value == "dahai"
    assert EventType.CHI.value == "chi"
    assert EventType.PON.value == "pon"
    assert EventType.DAIMINKAN.value == "daiminkan"
    assert EventType.ANKAN.value == "ankan"
    assert EventType.KAKAN.value == "kakan"
    assert EventType.DORA.value == "dora"
    assert EventType.REACH.value == "reach"
    assert EventType.REACH_ACCEPTED.value == "reach_accepted"
    assert EventType.HORA.value == "hora"
    assert EventType.RYUKYOKU.value == "ryukyoku"
    assert EventType.END_KYOKU.value == "end_kyoku"
    assert EventType.END_GAME.value == "end_game"


def test_type_is_classvar():
    assert Tsumo(actor=0, pai="6m").type is EventType.TSUMO
    assert Dahai(actor=0, pai="6m", tsumogiri=True).type is EventType.DAHAI
    assert EndGame().type is EventType.END_GAME


def test_events_are_frozen():
    event = Tsumo(actor=0, pai="6m")
    try:
        event.actor = 1  # type: ignore[misc]
    except Exception as exc:  # dataclasses.FrozenInstanceError
        assert "FrozenInstanceError" in type(exc).__name__
    else:
        raise AssertionError("frozen dataclass should reject attribute assignment")


def test_kan_event_types():
    assert KAN_EVENT_TYPES == {
        EventType.DAIMINKAN,
        EventType.ANKAN,
        EventType.KAKAN,
    }
