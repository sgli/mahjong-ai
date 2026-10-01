"""Tests for the Phase 2 Replay Engine."""

import pytest

from mahjong.parser.events import (
    Ankan,
    Chi,
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
from mahjong.replay import ReplayEngine, ReplayInconsistency, ReplayState


def _sg(names=("a", "b", "c", "d")):
    return StartGame(names=names, kyoku_first=0, aka_flag=True)


def _sk(tehais, scores=(25000, 25000, 25000, 25000), oya=0, kyotaku=0, kyoku=1, dora="2m"):
    return StartKyoku(
        bakaze="E",
        dora_marker=dora,
        kyoku=kyoku,
        honba=0,
        kyotaku=kyotaku,
        oya=oya,
        scores=scores,
        tehais=tehais,
    )


def _fill(tiles, n=13, filler="1m"):
    out = list(tiles)
    while len(out) < n:
        out.append(filler)
    return tuple(out)


def _run(events, game_id=None):
    engine = ReplayEngine(game_id=game_id)
    return engine, list(engine.replay(events))


def _hands(p0=(), p1=(), p2=(), p3=()):
    return (_fill(p0), _fill(p1), _fill(p2), _fill(p3))


# -- initialization ------------------------------------------------------------
def test_start_game_and_kyoku_initialization():
    engine, states = _run(
        [
            _sg(),
            _sk(_hands(p0=("1m",) * 0)),
        ],
        game_id="g1",
    )
    s0, s1 = states

    assert s0.event_type == "start_game"
    assert s0.names == ("a", "b", "c", "d")
    assert s0.aka_flag is True
    assert s0.kyoku_first == 0
    assert s0.round_id == ""

    assert s1.event_type == "start_kyoku"
    assert s1.game_id == "g1"
    assert s1.round_id == "E1"
    assert s1.bakaze == "E"
    assert s1.kyoku == 1
    assert s1.honba == 0
    assert s1.kyotaku == 0
    assert s1.oya == 0
    assert s1.turn == 0
    assert s1.scores == (25000, 25000, 25000, 25000)
    assert s1.dora_markers == ("2m",)
    assert [len(p.hand) for p in s1.players] == [13, 13, 13, 13]
    assert all(p.score == 25000 for p in s1.players)
    assert s1.players[0].seat == 0


# -- tsumo / dahai -------------------------------------------------------------
def test_tsumo_dahai():
    _, states = _run(
        [
            _sg(),
            _sk(_hands(p0=("1m",) * 0)),
            Tsumo(actor=0, pai="5m"),
            Dahai(actor=0, pai="5m", tsumogiri=True),
        ]
    )
    assert states[2].players[0].hand == tuple(sorted(("5m",) + ("1m",) * 13))
    final = states[3]
    p0 = final.players[0]
    assert "5m" not in p0.hand
    assert p0.discards == ("5m",)
    assert p0.discard_tsumogiri == (True,)
    assert len(p0.hand) == 13
    assert final.turn == 1


# -- chi / pon -----------------------------------------------------------------
def test_chi():
    hands = _hands(p0=("6p", "7p"), p1=("8p",))
    _, states = _run(
        [
            _sg(),
            _sk(hands),
            Tsumo(actor=1, pai="9m"),
            Dahai(actor=1, pai="8p", tsumogiri=False),
            Chi(actor=0, target=1, pai="8p", consumed=("6p", "7p")),
            Dahai(actor=0, pai="1m", tsumogiri=False),
        ]
    )
    final = states[-1]
    p0 = final.players[0]
    assert [m.kind for m in p0.melds] == ["chi"]
    assert sorted(p0.melds[0].tiles) == ["6p", "7p", "8p"]
    assert p0.melds[0].called == "8p"
    assert p0.melds[0].from_ == 1
    assert len(p0.hand) == 10  # 13 - 2 (chi) - 1 (dahai)
    assert "8p" not in final.players[1].discards
    assert final.turn == 1  # next player after actor 0's discard


def test_pon():
    hands = _hands(p0=("F", "F"), p1=("F",))
    _, states = _run(
        [
            _sg(),
            _sk(hands),
            Tsumo(actor=1, pai="9m"),
            Dahai(actor=1, pai="F", tsumogiri=False),
            Pon(actor=0, target=1, pai="F", consumed=("F", "F")),
            Dahai(actor=0, pai="1m", tsumogiri=False),
        ]
    )
    final = states[-1]
    p0 = final.players[0]
    assert [m.kind for m in p0.melds] == ["pon"]
    assert sorted(p0.melds[0].tiles) == ["F", "F", "F"]
    assert p0.melds[0].called == "F"
    assert len(p0.hand) == 10
    assert "F" not in final.players[1].discards


# -- the three kan variants ----------------------------------------------------
def test_daiminkan():
    hands = _hands(p0=("1s", "1s", "1s"), p3=("1s",))
    _, states = _run(
        [
            _sg(),
            _sk(hands),
            Tsumo(actor=3, pai="9m"),
            Dahai(actor=3, pai="1s", tsumogiri=False),
            Daiminkan(actor=0, target=3, pai="1s", consumed=("1s", "1s", "1s")),
            Dora(dora_marker="5m"),
            Tsumo(actor=0, pai="7p"),
            Dahai(actor=0, pai="7p", tsumogiri=True),
        ]
    )
    final = states[-1]
    p0 = final.players[0]
    assert [m.kind for m in p0.melds] == ["daiminkan"]
    assert sorted(p0.melds[0].tiles) == ["1s", "1s", "1s", "1s"]
    assert p0.melds[0].from_ == 3
    assert p0.melds[0].called == "1s"
    assert len(p0.hand) == 10
    assert "1s" not in final.players[3].discards
    assert final.dora_markers == ("2m", "5m")


def test_ankan():
    hands = _hands(p0=("E", "E", "E", "E"))
    _, states = _run(
        [
            _sg(),
            _sk(hands),
            Tsumo(actor=0, pai="1m"),
            Ankan(actor=0, consumed=("E", "E", "E", "E")),
            Dora(dora_marker="6p"),
            Tsumo(actor=0, pai="5m"),
            Dahai(actor=0, pai="5m", tsumogiri=True),
        ]
    )
    final = states[-1]
    p0 = final.players[0]
    assert [m.kind for m in p0.melds] == ["ankan"]
    assert sorted(p0.melds[0].tiles) == ["E", "E", "E", "E"]
    assert p0.melds[0].from_ is None
    assert p0.melds[0].called is None
    assert len(p0.hand) == 10
    assert final.dora_markers == ("2m", "6p")


def test_kakan():
    # actor 0 pons "P", then later adds the 4th "P" from hand via kakan.
    hands = _hands(p0=("P", "P", "P"), p1=("P",))
    _, states = _run(
        [
            _sg(),
            _sk(hands),
            Tsumo(actor=1, pai="9m"),
            Dahai(actor=1, pai="P", tsumogiri=False),
            Pon(actor=0, target=1, pai="P", consumed=("P", "P")),
            Dahai(actor=0, pai="1m", tsumogiri=False),
            Tsumo(actor=0, pai="2m"),
            Kakan(actor=0, pai="P", consumed=("P", "P", "P")),
            Dora(dora_marker="3s"),
            Tsumo(actor=0, pai="5m"),
            Dahai(actor=0, pai="5m", tsumogiri=True),
        ]
    )
    final = states[-1]
    p0 = final.players[0]
    assert [m.kind for m in p0.melds] == ["kakan"]
    assert sorted(p0.melds[0].tiles) == ["P", "P", "P", "P"]
    assert p0.melds[0].from_ == 1
    assert p0.melds[0].called == "P"
    assert len(p0.hand) == 10


# -- reach / reach_accepted ----------------------------------------------------
def test_reach_and_reach_accepted():
    _, states = _run(
        [
            _sg(),
            _sk(_hands(p0=("6m",))),
            Tsumo(actor=0, pai="5m"),
            Reach(actor=0),
            Dahai(actor=0, pai="5m", tsumogiri=False),
            ReachAccepted(actor=0),
        ]
    )
    assert states[3].players[0].riichi is True
    assert states[3].players[0].score == 25000  # reach itself costs nothing
    final = states[5]
    assert final.players[0].riichi is True
    assert final.players[0].ippatsu is True
    assert final.players[0].score == 24000
    assert final.kyotaku == 1


def test_ippatsu_closes_on_next_dahai():
    _, states = _run(
        [
            _sg(),
            _sk(_hands(p0=("6m",))),
            Tsumo(actor=0, pai="5m"),
            Reach(actor=0),
            Dahai(actor=0, pai="5m", tsumogiri=False),
            ReachAccepted(actor=0),
            Tsumo(actor=1, pai="9m"),
            Dahai(actor=1, pai="9m", tsumogiri=True),
            Tsumo(actor=0, pai="7m"),
            Dahai(actor=0, pai="7m", tsumogiri=True),
        ]
    )
    assert states[5].players[0].ippatsu is True
    assert states[-1].players[0].ippatsu is False


# -- hora / ryukyoku settlement + cross-kyoku consistency -----------------------
def test_hora_score_settlement_and_consistency():
    events = [
        _sg(),
        _sk(_hands(p0=("1m",) * 0)),
        Tsumo(actor=0, pai="5m"),
        Reach(actor=0),
        Dahai(actor=0, pai="5m", tsumogiri=False),
        ReachAccepted(actor=0),
        Tsumo(actor=1, pai="9m"),
        Dahai(actor=1, pai="9m", tsumogiri=True),
        Tsumo(actor=0, pai="7m"),
        Hora(actor=0, target=0, deltas=(2000, -1000, -500, -500), ura_markers=()),
        EndKyoku(),
        _sk(_hands(p0=("1m",) * 0), scores=(26000, 24000, 24500, 24500), kyotaku=0),
        EndGame(),
    ]
    engine, states = _run(events)
    after_hora = [s for s in states if s.event_type == "hora"][0]
    assert after_hora.scores == (26000, 24000, 24500, 24500)
    assert after_hora.kyotaku == 0
    assert engine.inconsistencies == []


def test_reach_without_accept_then_ryukyoku_still_pays_stick():
    # Real-data quirk: ``reach -> dahai -> ryukyoku`` omits reach_accepted but
    # the stick is still paid.
    events = [
        _sg(),
        _sk(_hands(p0=("1m",) * 0)),
        Tsumo(actor=0, pai="5m"),
        Reach(actor=0),
        Dahai(actor=0, pai="5m", tsumogiri=False),
        Ryukyoku(deltas=(0, 0, 0, 0)),
        EndKyoku(),
        _sk(_hands(p0=("1m",) * 0), scores=(24000, 25000, 25000, 25000), kyotaku=1),
        EndGame(),
    ]
    engine, states = _run(events)
    after_ryukyoku = [s for s in states if s.event_type == "ryukyoku"][0]
    assert after_ryukyoku.scores == (24000, 25000, 25000, 25000)
    assert after_ryukyoku.kyotaku == 1
    assert engine.inconsistencies == []


def test_reach_without_accept_then_hora_does_not_pay_stick():
    # Ron on the riichi discard: no reach_accepted and no stick is deducted.
    events = [
        _sg(),
        _sk(_hands(p0=("1m",) * 0)),
        Tsumo(actor=0, pai="5m"),
        Reach(actor=0),
        Dahai(actor=0, pai="5m", tsumogiri=False),
        Hora(actor=1, target=0, deltas=(-1800, 1800, 0, 0), ura_markers=()),
        EndKyoku(),
        _sk(_hands(p0=("1m",) * 0), scores=(23200, 26800, 25000, 25000), kyotaku=0),
        EndGame(),
    ]
    engine, _ = _run(events)
    assert engine.inconsistencies == []


def test_ryukyoku_kyotaku_carries_over():
    events = [
        _sg(),
        _sk(_hands(p0=("1m",) * 0)),
        Tsumo(actor=0, pai="5m"),
        Reach(actor=0),
        Dahai(actor=0, pai="5m", tsumogiri=False),
        ReachAccepted(actor=0),
        Ryukyoku(deltas=(3000, -1000, -1000, -1000)),
        EndKyoku(),
        _sk(_hands(p0=("1m",) * 0), scores=(27000, 24000, 24000, 24000), kyotaku=1),
        EndGame(),
    ]
    engine, states = _run(events)
    after_ryukyoku = [s for s in states if s.event_type == "ryukyoku"][0]
    assert after_ryukyoku.scores == (27000, 24000, 24000, 24000)
    assert after_ryukyoku.kyotaku == 1
    assert engine.inconsistencies == []


def test_score_mismatch_is_recorded_not_silent():
    events = [
        _sg(),
        _sk(_hands(p0=("1m",) * 0)),
        Tsumo(actor=0, pai="5m"),
        Dahai(actor=0, pai="5m", tsumogiri=False),
        Hora(actor=0, target=0, deltas=(1000, -1000, 0, 0), ura_markers=()),
        EndKyoku(),
        # deliberately wrong next scores -> should be recorded, not raised
        _sk(_hands(p0=("1m",) * 0), scores=(99999, 25000, 25000, 25000)),
        EndGame(),
    ]
    engine, _ = _run(events)
    assert len(engine.inconsistencies) == 1
    assert "score mismatch" in engine.inconsistencies[0]


# -- boundaries ----------------------------------------------------------------
def test_end_kyoku_end_game_boundaries():
    _, states = _run(
        [
            _sg(),
            _sk(_hands(p0=("1m",) * 0)),
            EndKyoku(),
            EndGame(),
        ]
    )
    assert [s.event_type for s in states] == ["start_game", "start_kyoku", "end_kyoku", "end_game"]


# -- illegal sequences ----------------------------------------------------------
def test_dahai_tile_not_in_hand_raises():
    with pytest.raises(ReplayInconsistency, match="not in hand"):
        list(
            ReplayEngine().replay(
                [
                    _sg(),
                    _sk(_hands(p0=("1m",) * 0)),
                    Tsumo(actor=0, pai="5m"),
                    Dahai(actor=0, pai="9s", tsumogiri=False),
                ]
            )
        )


def test_chi_claimed_tile_not_latest_discard_raises():
    with pytest.raises(ReplayInconsistency, match="latest discard"):
        list(
            ReplayEngine().replay(
                [
                    _sg(),
                    _sk(_hands(p0=("6p", "7p"), p1=("8p",))),
                    Tsumo(actor=1, pai="9m"),
                    Dahai(actor=1, pai="8p", tsumogiri=False),
                    # "9p" is not in target's river at all
                    Chi(actor=0, target=1, pai="9p", consumed=("6p", "7p")),
                ]
            )
        )


def test_kakan_without_matching_pon_raises():
    with pytest.raises(ReplayInconsistency, match="no pon meld"):
        list(
            ReplayEngine().replay(
                [
                    _sg(),
                    _sk(_hands(p0=("P",))),
                    Tsumo(actor=0, pai="1m"),
                    Kakan(actor=0, pai="P", consumed=("P", "P", "P")),
                ]
            )
        )


# -- determinism ----------------------------------------------------------------
def test_replay_is_deterministic():
    events = [
        _sg(),
        _sk(_hands(p0=("1m",) * 0)),
        Tsumo(actor=0, pai="5m"),
        Dahai(actor=0, pai="5m", tsumogiri=True),
        EndKyoku(),
        EndGame(),
    ]
    s1 = list(ReplayEngine(game_id="g").replay(events))
    s2 = list(ReplayEngine(game_id="g").replay(events))
    assert s1 == s2
    assert isinstance(s1[1], ReplayState)
