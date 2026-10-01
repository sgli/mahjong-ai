"""Tests for decision extraction (Action / Observation / DecisionSample / Extractor)."""

import dataclasses

from mahjong.decision import (
    ACTION_SCHEMA_VERSION,
    Action,
    ActionType,
    DecisionSample,
    PlayerObservation,
)
from mahjong.decision.extractor import DecisionExtractor
from mahjong.parser.events import (
    Dahai,
    Daiminkan,
    Dora,
    EndGame,
    EndKyoku,
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
from mahjong.replay import ReplayEngine
from mahjong.replay.state import Meld, PlayerState, ReplayState


def _sg(names=("a", "b", "c", "d")):
    return StartGame(names=names, kyoku_first=0, aka_flag=True)


def _sk(tehais, oya=0, kyoku=1):
    return StartKyoku(
        bakaze="E",
        dora_marker="2m",
        kyoku=kyoku,
        honba=0,
        kyotaku=0,
        oya=oya,
        scores=(25000, 25000, 25000, 25000),
        tehais=tehais,
    )


def _fill(tiles, n=13, filler="1m"):
    out = list(tiles)
    while len(out) < n:
        out.append(filler)
    return tuple(out)


def _extract(events, game_id="g"):
    states = list(ReplayEngine(game_id=game_id).replay(events))
    extractor = DecisionExtractor(game_id=game_id)
    samples = list(extractor.extract(states, events))
    return extractor, samples


# -- action schema -------------------------------------------------------------
def test_action_schema_version():
    assert ACTION_SCHEMA_VERSION == "action-v1"


def test_eight_action_types_construct():
    Action(ActionType.DISCARD, tile="5m")
    Action(ActionType.CHI, tile="8p", consumed=("7p", "6p"), target=0)
    Action(ActionType.PON, tile="F", consumed=("F", "F"), target=3)
    Action(ActionType.KAN, kan_kind="ankan", consumed=("E", "E", "E", "E"))
    Action(ActionType.RIICHI, tile="5m")
    Action(ActionType.RON, target=2)
    Action(ActionType.TSUMO)
    Action(ActionType.PASS)


def test_action_consumed_normalized_to_sorted():
    a = Action(ActionType.CHI, tile="8p", consumed=("7p", "6p"), target=0)
    assert a.consumed == ("6p", "7p")


def test_action_rejects_bad_kan_kind():
    import pytest

    with pytest.raises(ValueError):
        Action(ActionType.KAN, kan_kind="nope", consumed=("E", "E", "E", "E"))


# -- observation ---------------------------------------------------------------
def _make_state():
    players = tuple(
        PlayerState(
            seat=s,
            hand=tuple([f"{s + 1}m"] * 13),
            melds=(Meld(kind="pon", tiles=("E", "E", "E"), from_=0, called="E"),) if s == 1 else (),
            discards=(f"{s + 1}p",),
            riichi=(s == 2),
            score=25000,
        )
        for s in range(4)
    )
    return ReplayState(
        game_id="g",
        round_id="E1",
        bakaze="E",
        kyoku=1,
        honba=0,
        kyotaku=0,
        oya=0,
        scores=(25000, 25000, 25000, 25000),
        players=players,
        dora_markers=("2m",),
        ura_markers=("3m",),
        turn=0,
    )


def test_observation_projects_visible_subset_only():
    state = _make_state()
    obs = PlayerObservation.from_state(state, seat=0)

    assert obs.hand == tuple(["1m"] * 13)
    assert obs.melds == ()
    assert obs.opponents_melds[0] == ()
    assert obs.opponents_melds[1] == state.players[1].melds
    assert obs.riichi == (False, False, True, False)
    assert obs.dora_markers == ("2m",)

    # No hidden-information fields may exist on the observation.
    field_names = {f.name for f in dataclasses.fields(obs)}
    assert "ura_markers" not in field_names
    assert not any("hand" in n for n in field_names if "opponent" in n)
    assert not any("wall" in n for n in field_names)
    assert "hand" in field_names  # own hand is visible


def test_observation_keeps_opponent_ankan_tiles_visible():
    # Under Tenhou rules an ankan shows its two middle tiles face up, so which
    # tile the ankan is is public for all players.  Seat 1 must therefore still
    # see seat 0's ankan tiles (this test locks in the correct behaviour and
    # prevents it being "fixed" into a leak-removal regression).
    players = tuple(
        PlayerState(
            seat=s,
            hand=tuple([f"{s + 1}m"] * 13),
            melds=(Meld(kind="ankan", tiles=("6p", "6p", "6p", "6p")),) if s == 0 else (),
            score=25000,
        )
        for s in range(4)
    )
    state = ReplayState(
        game_id="g",
        round_id="E1",
        bakaze="E",
        kyoku=1,
        honba=0,
        kyotaku=0,
        oya=0,
        scores=(25000, 25000, 25000, 25000),
        players=players,
        dora_markers=("2m",),
        ura_markers=(),
        turn=0,
    )

    obs_opponent = PlayerObservation.from_state(state, seat=1)
    opponent_melds = obs_opponent.opponents_melds[0]
    assert len(opponent_melds) == 1
    assert opponent_melds[0].kind == "ankan"
    assert opponent_melds[0].tiles == ("6p", "6p", "6p", "6p")  # visible to opponents

    obs_own = PlayerObservation.from_state(state, seat=0)
    assert obs_own.melds[0].kind == "ankan"
    assert obs_own.melds[0].tiles == ("6p", "6p", "6p", "6p")  # own ankan visible


# -- extractor: human action reconstruction -------------------------------------
def test_extractor_discard_decision():
    events = [
        _sg(),
        _sk((_fill(()), _fill(()), _fill(()), _fill(()))),
        Tsumo(actor=0, pai="5m"),
        Dahai(actor=0, pai="5m", tsumogiri=True),
        EndKyoku(),
        EndGame(),
    ]
    extractor, samples = _extract(events)
    assert extractor.inconsistencies == []
    discard_samples = [s for s in samples if s.decision_point.kind == "discard"]
    assert len(discard_samples) >= 1
    s = discard_samples[0]
    assert s.player_id == 0
    assert s.action == Action(ActionType.DISCARD, tile="5m")
    assert Action(ActionType.DISCARD, tile="5m") in s.legal_actions


def test_extractor_riichi_reconstruction():
    tenpai = ("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p")
    events = [
        _sg(),
        _sk((tenpai, _fill(()), _fill(()), _fill(()))),
        Tsumo(actor=0, pai="5m"),
        Reach(actor=0),
        Dahai(actor=0, pai="5m", tsumogiri=False),
        ReachAccepted(actor=0),
        EndKyoku(),
        EndGame(),
    ]
    extractor, samples = _extract(events)
    discard = [s for s in samples if s.decision_point.kind == "discard"][0]
    assert discard.action == Action(ActionType.RIICHI, tile="5m")
    assert Action(ActionType.RIICHI, tile="5m") in discard.legal_actions
    assert extractor.inconsistencies == []


def test_extractor_pon_response():
    events = [
        _sg(),
        _sk((_fill(("C", "C")), _fill(("C",), filler="2m"), _fill((), filler="3m"), _fill((), filler="4m"))),
        Tsumo(actor=0, pai="5m"),
        Dahai(actor=0, pai="1m", tsumogiri=False),
        Tsumo(actor=1, pai="9m"),
        Dahai(actor=1, pai="C", tsumogiri=False),
        Pon(actor=0, target=1, pai="C", consumed=("C", "C")),
        Dahai(actor=0, pai="1m", tsumogiri=False),
        EndKyoku(),
        EndGame(),
    ]
    extractor, samples = _extract(events)
    assert extractor.inconsistencies == []
    response = [s for s in samples if s.decision_point.kind == "response"][0]
    assert response.player_id == 0
    assert response.action == Action(ActionType.PON, tile="C", consumed=("C", "C"), target=1)
    assert response.action in response.legal_actions


def test_extractor_reports_human_not_in_legal():
    # Seat 0 riichis with a hand that is not tenpai after discard -> the RIICHI
    # action is reconstructed but must not be in legal actions.
    closed_non_tenpai = ("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "4p", "5p")
    events = [
        _sg(),
        _sk((closed_non_tenpai, _fill(()), _fill(()), _fill(()))),
        Tsumo(actor=0, pai="5m"),
        Reach(actor=0),
        Dahai(actor=0, pai="5m", tsumogiri=False),
        ReachAccepted(actor=0),
        EndKyoku(),
        EndGame(),
    ]
    extractor, samples = _extract(events)
    discard = [s for s in samples if s.decision_point.kind == "discard"][0]
    assert discard.action == Action(ActionType.RIICHI, tile="5m")
    assert any("not in legal actions" in m for m in extractor.inconsistencies)


def test_extractor_pon_on_riichi_discard():
    # reach -> dahai -> reach_accepted -> pon : reach_accepted must be skipped
    # so the pon is captured as the responder's action (not PASS).
    events = [
        _sg(),
        _sk((_fill(("C",)), _fill(("C", "C"), filler="2m"), _fill((), filler="3m"), _fill((), filler="4m"))),
        Tsumo(actor=0, pai="5m"),
        Reach(actor=0),
        Dahai(actor=0, pai="C", tsumogiri=False),
        ReachAccepted(actor=0),
        Pon(actor=1, target=0, pai="C", consumed=("C", "C")),
        Dahai(actor=1, pai="2m", tsumogiri=False),
        EndKyoku(),
        EndGame(),
    ]
    extractor, samples = _extract(events)
    assert extractor.inconsistencies == []
    responses = [s for s in samples if s.decision_point.kind == "response" and s.player_id == 1]
    assert responses and responses[0].action == Action(ActionType.PON, tile="C", consumed=("C", "C"), target=0)


def test_extractor_skips_dora_between_rinchan_tsumo_and_discard():
    # daiminkan -> tsumo (rinchan) -> dora -> dahai : the dora event sits
    # between the draw and the discard and must be skipped.
    events = [
        _sg(),
        _sk((_fill(("1s", "1s", "1s")), _fill(()), _fill(()), _fill(("1s",), filler="4m"))),
        Tsumo(actor=3, pai="9m"),
        Dahai(actor=3, pai="1s", tsumogiri=False),
        Daiminkan(actor=0, target=3, pai="1s", consumed=("1s", "1s", "1s")),
        Tsumo(actor=0, pai="4p"),
        Dora(dora_marker="8p"),
        Dahai(actor=0, pai="4p", tsumogiri=True),
        EndKyoku(),
        EndGame(),
    ]
    extractor, samples = _extract(events)
    assert extractor.inconsistencies == []
    discards = [s for s in samples if s.decision_point.kind == "discard" and s.player_id == 0]
    assert any(s.action == Action(ActionType.DISCARD, tile="4p") for s in discards)


def test_extractor_chankan_ron_after_kakan():
    # seat 0 pons 9p then kakan's the 4th 9p; seat 2 robs the kan (chankan).
    events = [
        _sg(),
        _sk((
            _fill(("9p", "9p", "9p")),
            _fill(("9p",), filler="2m"),
            ("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "7p", "8p", "5p", "5p"),
            _fill((), filler="4m"),
        )),
        Tsumo(actor=0, pai="3m"),
        Dahai(actor=0, pai="1m", tsumogiri=False),
        Tsumo(actor=1, pai="6m"),
        Dahai(actor=1, pai="9p", tsumogiri=False),
        Pon(actor=0, target=1, pai="9p", consumed=("9p", "9p")),
        Dahai(actor=0, pai="1m", tsumogiri=False),
        Tsumo(actor=0, pai="3m"),
        Kakan(actor=0, pai="9p", consumed=("9p", "9p", "9p")),
        Hora(actor=2, target=0, deltas=(-8000, 0, 8000, 0), ura_markers=()),
        EndKyoku(),
        EndGame(),
    ]
    extractor, samples = _extract(events)
    assert extractor.inconsistencies == []
    chankan = [s for s in samples if s.decision_point.kind == "response" and s.action.type is ActionType.RON]
    assert chankan and chankan[0].action == Action(ActionType.RON, target=0)
    assert chankan[0].player_id == 2


def test_extractor_skips_last_draw_before_ryukyoku():
    events = [
        _sg(),
        _sk((_fill(()), _fill(()), _fill(()), _fill(()))),
        Tsumo(actor=0, pai="5m"),
        Ryukyoku(deltas=(0, 0, 0, 0)),
        EndKyoku(),
        EndGame(),
    ]
    extractor, samples = _extract(events)
    assert extractor.inconsistencies == []
    assert samples == []


# -- decision sample assembly ---------------------------------------------------
def test_decision_sample_assembly():
    state = _make_state()
    obs = PlayerObservation.from_state(state, seat=0)
    sample = DecisionSample(
        game_id="g",
        round_id="E1",
        player_id=0,
        observation=obs,
        legal_actions=(Action(ActionType.DISCARD, tile="1m"),),
        action=Action(ActionType.DISCARD, tile="1m"),
        metadata={"kind": "discard"},
    )
    assert sample.game_id == "g"
    assert sample.round_id == "E1"
    assert sample.player_id == 0
    assert sample.observation is obs
    assert sample.action in sample.legal_actions
