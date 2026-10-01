"""Regression tests for rule-core cleanups (temporary furiten, houtei,
riichi kan restriction, ankan consumed validation)."""

import pytest

from mahjong.decision.action import Action, ActionType
from mahjong.environment import MahjongEnv
from mahjong.environment.state import PHASE_RESPONSE
from mahjong.rules import Structure, Group, WinContext, compute_yaku, validate_ankan_consumed


def _tenpai_5p_hand():
    # 13 tiles waiting on 5p (ittsuu), so RON is legal even without houtei
    return sorted(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"])


def _no_yaku_except_houtei_hand():
    # 234m 567m 234p 678p + 9m single -> waiting 9m tanki; no other yaku
    return sorted(["2m", "3m", "4m", "5m", "6m", "7m", "2p", "3p", "4p", "6p", "7p", "8p", "9m"])


# -- item 1: temporary furiten --------------------------------------------------
def test_temporary_furiten_set_on_riichi_pass():
    env = MahjongEnv(seed=2)
    env.reset()
    p1 = env.state.players[1]
    p1.hand = _tenpai_5p_hand()
    p1.riichi = True
    env.state.discarder = 0
    env.state.discard_tile = "5p"
    env.state.pending_responders = [1, 2, 3]
    env.state.turn = 1
    env.state.phase = PHASE_RESPONSE
    legal = env.legal_actions(1)
    assert any(a.type is ActionType.RON for a in legal)
    env.step(Action(ActionType.PASS))
    assert p1.temporary_furiten is True


def test_temporary_furiten_blocks_ron():
    env = MahjongEnv(seed=2)
    env.reset()
    p1 = env.state.players[1]
    p1.hand = _tenpai_5p_hand()
    p1.temporary_furiten = True
    assert env._can_ron(1, "5p", 0) is False


def test_temporary_furiten_cleared_on_draw():
    env = MahjongEnv(seed=2)
    env.reset()
    p1 = env.state.players[1]
    p1.temporary_furiten = True
    env._draw_for(1)
    assert p1.temporary_furiten is False


# -- item 2: houtei ------------------------------------------------------------
def test_houtei_yaku_detected():
    structure = Structure(
        sets=(
            Group("run", ("2m", "3m", "4m"), False),
            Group("run", ("5m", "6m", "7m"), False),
            Group("run", ("2p", "3p", "4p"), False),
            Group("run", ("6p", "7p", "8p"), False),
        ),
        pair="9m",
    )
    ctx = WinContext(round_wind="E", seat_wind="W", win_tile="9m", is_tsumo=False, is_houtei=True)
    result = compute_yaku(structure, ctx)
    assert ("houtei", 1) in result.yaku


def test_houtei_enables_ron_on_last_discard():
    env = MahjongEnv(seed=2)
    env.reset()
    p1 = env.state.players[1]
    p1.hand = _no_yaku_except_houtei_hand()  # waits 9m; no yaku except houtei
    # wall empty -> last discard (houtei)
    env.state.wall.live = []
    assert env._can_ron(1, "9m", 0) is True
    # wall still has tiles -> no houtei, no yaku
    env.state.wall.live = ["5p"]
    assert env._can_ron(1, "9m", 0) is False


# -- item 3: riichi kan restriction --------------------------------------------
def test_riichi_ankan_allowed_when_wait_unchanged():
    env = MahjongEnv(seed=2)
    env.reset()
    p0 = env.state.players[0]
    # 1111m + 234m + 567m + 789m + 5p; last_drawn=1m; wait 5p both before/after ankan
    p0.hand = sorted(["1m", "1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "7m", "8m", "9m", "5p"])
    p0.riichi = True
    p0.drawn_this_turn = True
    p0.last_drawn = "1m"
    env.state.turn = 0
    env.state.phase = "discard"
    legal = env.legal_actions(0)
    ankans = [a for a in legal if a.type is ActionType.KAN and getattr(a, "kan_kind", None) == "ankan"]
    assert len(ankans) == 1
    assert tuple(ankans[0].consumed) == ("1m", "1m", "1m", "1m")


def test_riichi_ankan_forbidden_when_wait_changes():
    env = MahjongEnv(seed=2)
    env.reset()
    p0 = env.state.players[0]
    # 1111m + 234m + 567m + 123p + 9m; last_drawn=9m; ankan 1111m breaks the wait
    p0.hand = sorted(["1m", "1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "1p", "2p", "3p", "9m"])
    p0.riichi = True
    p0.drawn_this_turn = True
    p0.last_drawn = "9m"
    env.state.turn = 0
    env.state.phase = "discard"
    legal = env.legal_actions(0)
    ankans = [a for a in legal if a.type is ActionType.KAN and getattr(a, "kan_kind", None) == "ankan"]
    assert len(ankans) == 0


# -- item 4: ankan consumed validation -----------------------------------------
def test_validate_ankan_consumed():
    assert validate_ankan_consumed(("1m", "1m", "1m", "1m")) is True
    assert validate_ankan_consumed(("5m", "5m", "5m", "5mr")) is True  # red five same value
    assert validate_ankan_consumed(("1m", "1m", "1m")) is False  # wrong count
    assert validate_ankan_consumed(("1m", "2m", "3m", "4m")) is False  # not same value


def test_env_rejects_invalid_ankan():
    env = MahjongEnv(seed=2)
    env.reset()
    with pytest.raises(ValueError):
        env._apply_kan(0, Action(ActionType.KAN, consumed=("1m", "2m", "3m", "4m"), kan_kind="ankan"))
