"""Tests for nagashi mangan (流局满贯) + user-confirmed Majsoul rules."""

import inspect
from dataclasses import replace

from mahjong.environment import MahjongEnv
from mahjong.rules import MAJSOUL_RULES, TENHOU_RULES
from mahjong.rules import config as config_module

_TERMINALS = ["1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C"]
_NOTEN = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "4p", "5p"]


def _setup_nagashi(env, seat):
    env.reset()
    for s in range(4):
        env.state.players[s].hand = sorted(_NOTEN)
        env.state.players[s].discards = ["5m"]  # middle discard -> not nagashi
        env.state.players[s].discard_called = False
    env.state.players[seat].discards = list(_TERMINALS)
    env.state.players[seat].discard_called = False


def test_nagashi_non_dealer_8000():
    env = MahjongEnv(seed=1, rules=MAJSOUL_RULES)  # 雀魂有流局满贯
    _setup_nagashi(env, seat=1)
    before = env.state.scores[:]
    env._resolve_ryukyoku(abort=False)
    # seat 1 (non-dealer): dealer pays 4000, other two pay 2000 each
    assert env.state.scores[1] == before[1] + 8000
    assert env.state.scores[0] == before[0] - 4000
    assert env.state.scores[2] == before[2] - 2000
    assert env.state.scores[3] == before[3] - 2000


def test_nagashi_dealer_12000():
    env = MahjongEnv(seed=1, rules=MAJSOUL_RULES)
    _setup_nagashi(env, seat=0)
    before = env.state.scores[:]
    env._resolve_ryukyoku(abort=False)
    # seat 0 (dealer): each of 3 pays 4000
    assert env.state.scores[0] == before[0] + 12000
    assert env.state.scores[1] == before[1] - 4000
    assert env.state.scores[2] == before[2] - 4000
    assert env.state.scores[3] == before[3] - 4000


def test_nagashi_rejected_middle_discard():
    env = MahjongEnv(seed=1)
    env.reset()
    env.state.players[1].discards = ["1m", "9m", "E", "5m"]  # middle 5m
    env.state.players[1].discard_called = False
    assert env._is_nagashi(1) is False


def test_nagashi_rejected_called_discard():
    env = MahjongEnv(seed=1)
    env.reset()
    env.state.players[1].discards = ["1m", "9m", "E", "S"]
    env.state.players[1].discard_called = True
    assert env._is_nagashi(1) is False


def test_nagashi_disabled_uses_normal_settlement():
    from mahjong.rules import RulesConfig
    env = MahjongEnv(seed=1, rules=replace(TENHOU_RULES, nagashi_mangan=False))
    _setup_nagashi(env, seat=1)
    before = env.state.scores[:]
    env._resolve_ryukyoku(abort=False)
    # all noten -> no tenpai transfer; no mangan either
    assert env.state.scores == before


def test_majsoul_user_confirmed_fields():
    assert MAJSOUL_RULES.four_wind_abort is True
    assert MAJSOUL_RULES.four_kan_abort is True
    assert MAJSOUL_RULES.four_riichi_abort is True
    assert MAJSOUL_RULES.nine_terminals_abort is True
    assert MAJSOUL_RULES.kiriage_mangan is False
    assert MAJSOUL_RULES.red_fives == 3
    assert MAJSOUL_RULES.nagashi_mangan is True
    assert TENHOU_RULES.nagashi_mangan is True  # 用户确认：天凤有流局满贯
    src = inspect.getsource(config_module)
    assert "用户确认" in src
