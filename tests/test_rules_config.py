"""Tests for RulesConfig + atamahane/multi-ron switch + kiriage mangan + abort switches."""

import inspect
from dataclasses import replace

from mahjong.decision.action import Action, ActionType
from mahjong.environment import MahjongEnv
from mahjong.environment.state import PHASE_RESPONSE
from mahjong.replay.state import Meld
from mahjong.rules import MAJSOUL_RULES, TENHOU_RULES, RulesConfig, base_points_from, ron_payment
from mahjong.rules import config as config_module


def _double_ron_env(rules):
    env = MahjongEnv(seed=1, rules=rules)
    env.reset()
    hand = sorted(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"])
    for s in (1, 2):
        env.state.players[s].hand = hand
        env.state.players[s].discards = []
    env.state.discarder = 0
    env.state.discard_tile = "5p"
    env.state.response_choices = {
        1: Action(ActionType.RON, target=0),
        2: Action(ActionType.RON, target=0),
        3: Action(ActionType.PASS),
    }
    env.state.pending_responders = []
    env.state.phase = PHASE_RESPONSE
    return env


def test_atamahane_true_single_winner():
    env = _double_ron_env(TENHOU_RULES)
    before = env.state.scores[:]
    env._resolve_responses()
    # only the closest seat (1) wins; seat 2 unchanged
    assert env.state.scores[1] > before[1]
    assert env.state.scores[0] < before[0]
    assert env.state.scores[2] == before[2]


def test_multi_ron_false_both_winners_conserve_points():
    env = _double_ron_env(MAJSOUL_RULES)
    before_total = sum(env.state.scores)
    env._resolve_responses()
    # both winners gain, discarder loses, total conserved
    assert env.state.scores[1] > 25000
    assert env.state.scores[2] > 25000
    assert env.state.scores[0] < 25000
    assert sum(env.state.scores) == before_total


def test_same_scenario_differs_between_presets():
    tenhou = _double_ron_env(TENHOU_RULES)
    majsoul = _double_ron_env(MAJSOUL_RULES)
    tenhou._resolve_responses()
    majsoul._resolve_responses()
    # seat 2 wins only under majsoul (multi-ron)
    assert tenhou.state.scores[2] == 25000
    assert majsoul.state.scores[2] > 25000


def test_preset_field_values_and_sources():
    assert TENHOU_RULES.atamahane is True
    assert MAJSOUL_RULES.atamahane is False
    assert MAJSOUL_RULES.agari_yame is True
    assert MAJSOUL_RULES.west_round is True
    src = inspect.getsource(config_module)
    assert "无头跳" in src  # majsoul multi-ron source note
    assert "东南战西入" in src  # majsoul west-round source note


def test_rules_config_roundtrip():
    d = MAJSOUL_RULES.to_dict()
    assert RulesConfig.from_dict(d) == MAJSOUL_RULES
    assert d["atamahane"] is False


# -- kiriage mangan ------------------------------------------------------------
def test_kiriage_mangan_base_points():
    # 4 han 30 fu
    assert base_points_from(4, 30, kiriage_mangan=False) == 1920
    assert base_points_from(4, 30, kiriage_mangan=True) == 2000
    # 3 han 60 fu
    assert base_points_from(3, 60, kiriage_mangan=False) == 1920
    assert base_points_from(3, 60, kiriage_mangan=True) == 2000
    # 4 han 40 fu already mangan in both cases
    assert base_points_from(4, 40, kiriage_mangan=False) == 2000
    assert base_points_from(4, 40, kiriage_mangan=True) == 2000


def test_kiriage_mangan_payments():
    # base 1920 -> non-dealer 7700, dealer 11600
    assert ron_payment(1920, is_dealer=False) == 7700
    assert ron_payment(1920, is_dealer=True) == 11600
    # base 2000 (mangan) -> non-dealer 8000, dealer 12000
    assert ron_payment(2000, is_dealer=False) == 8000
    assert ron_payment(2000, is_dealer=True) == 12000


# -- abort switches ------------------------------------------------------------
def test_four_wind_abort_switch():
    env_on = MahjongEnv(seed=1, rules=TENHOU_RULES)
    env_on.reset()
    for s in range(4):
        env_on.state.players[s].discards = ["E"]
    assert env_on._check_four_wind_abort() is True

    env_off = MahjongEnv(seed=1, rules=replace(TENHOU_RULES, four_wind_abort=False))
    env_off.reset()
    for s in range(4):
        env_off.state.players[s].discards = ["E"]
    assert env_off._check_four_wind_abort() is False


def test_four_kan_abort_switch():
    def _setup(env):
        env.reset()
        env.state.players[0].melds = [
            Meld(kind="ankan", tiles=("1m", "1m", "1m", "1m"), from_=None, called=None),
            Meld(kind="daiminkan", tiles=("2m", "2m", "2m", "2m"), from_=1, called="2m"),
        ]
        env.state.players[1].melds = [
            Meld(kind="ankan", tiles=("3m", "3m", "3m", "3m"), from_=None, called=None),
            Meld(kind="kakan", tiles=("4m", "4m", "4m", "4m"), from_=0, called="4m"),
        ]

    env_on = MahjongEnv(seed=1, rules=TENHOU_RULES)
    _setup(env_on)
    assert env_on._check_four_kan_abort() is True

    env_off = MahjongEnv(seed=1, rules=replace(TENHOU_RULES, four_kan_abort=False))
    _setup(env_off)
    assert env_off._check_four_kan_abort() is False


def test_four_riichi_abort_switch():
    env_on = MahjongEnv(seed=1, rules=TENHOU_RULES)
    env_on.reset()
    for s in range(4):
        env_on.state.players[s].riichi = True
    assert env_on._check_four_riichi_abort() is True

    env_off = MahjongEnv(seed=1, rules=replace(TENHOU_RULES, four_riichi_abort=False))
    env_off.reset()
    for s in range(4):
        env_off.state.players[s].riichi = True
    assert env_off._check_four_riichi_abort() is False


def test_nine_terminals_abort_switch():
    hand = ["1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C"]
    env_on = MahjongEnv(seed=1, rules=TENHOU_RULES)
    env_on.reset()
    env_on.state.players[1].hand = sorted(hand)
    env_on.state.first_draw_done[1] = False
    assert env_on._can_kyuushu(1) is True

    env_off = MahjongEnv(seed=1, rules=replace(TENHOU_RULES, nine_terminals_abort=False))
    env_off.reset()
    env_off.state.players[1].hand = sorted(hand)
    env_off.state.first_draw_done[1] = False
    assert env_off._can_kyuushu(1) is False
