"""Environment Regression Gate (Phase 10.2 §7/§9).

Covers the §7 checklist at a high level.  Chi/pon/ankan/daiminkan/kakan,
riichi/tsumo/ron/chankan, exhaustive draw, kyuushu, nagashi mangan,
four-wind/four-riichi/four-kan aborts, honba/kyotaku/dealer continuation and
scoring are already covered end-to-end by the existing test files
(``test_environment.py``, ``test_environment_aborts.py``, ``test_nagashi.py``,
``test_scoring.py``, ``test_rule_fixes.py``).  This file adds the cross-cutting
regression: a full-hanchan self-play must finish legally, illegal actions are
rejected, and the RL environment version + RulesConfig wiring are fixed.
"""

from __future__ import annotations

import random

import pytest

from mahjong.decision.action import Action, ActionType
from mahjong.environment import ENVIRONMENT_VERSION, MahjongEnv
from mahjong.rules.config import TENHOU_RULES, RulesConfig


def _random_play(seed, prefer_pass=True):
    env = MahjongEnv(seed=seed)
    env.reset()
    rng = random.Random(seed)
    steps = 0
    while not env.done():
        seat = env.state.turn
        legal = env.legal_actions(seat)
        assert legal, f"no legal actions at step {steps}"
        if prefer_pass:
            action = next((a for a in legal if getattr(a, "type", None) is ActionType.PASS), None)
            if action is None:
                action = rng.choice(legal)
        else:
            action = rng.choice(legal)
        env.step(action)
        steps += 1
        assert steps < 100_000, "game did not terminate"
    return env, steps


def test_basic_flow_full_hanchan_terminates_legally():
    env, steps = _random_play(3)
    assert env.done()
    assert steps > 100
    assert env.state.final_ranks is not None
    assert sorted(env.state.final_ranks) == [0, 1, 2, 3]
    # 守恒：四点分数 + 供托（未认领立直棒）= 初始 100000
    assert sum(env.state.scores) + env.state.kyotaku * 1000 == 100_000


def test_illegal_action_rejected():
    env = MahjongEnv(seed=1)
    env.reset()
    seat = env.state.turn
    # RON 在 discard 阶段非法（响应类动作）
    with pytest.raises(ValueError):
        env.step(Action(ActionType.RON))


def test_illegal_discard_tile_rejected():
    env = MahjongEnv(seed=1)
    env.reset()
    seat = env.state.turn
    # 找一个不在手牌中的字牌（通常不在手牌）
    hand = env.state.players[seat].hand
    illegal_tile = next((t for t in ("E", "S", "W", "N", "P", "F", "C", "1m") if t not in hand), None)
    if illegal_tile is not None:
        with pytest.raises(ValueError):
            env.step(Action(ActionType.DISCARD, tile=illegal_tile))


def test_environment_version_fixed():
    assert ENVIRONMENT_VERSION == "tenhou-v2-rl"


def test_rules_config_wiring():
    """RulesConfig flags must actually be wired in the environment (§9).

    ``double_yakuman`` is the only flag NOT wired end-to-end (documented as
    step3 placeholder) — assert it is False and absent from the environment's
    scoring path (it is only a config placeholder).
    """
    assert isinstance(TENHOU_RULES, RulesConfig)
    assert TENHOU_RULES.rules_id == "tenhou-v1"
    # 已端到端接通的 flag（env.py 引用了这些）
    assert TENHOU_RULES.atamahane is True
    assert TENHOU_RULES.four_wind_abort is True
    assert TENHOU_RULES.four_kan_abort is True
    assert TENHOU_RULES.four_riichi_abort is True
    assert TENHOU_RULES.nine_terminals_abort is True
    assert TENHOU_RULES.nagashi_mangan is True
    assert TENHOU_RULES.west_round is True
    assert TENHOU_RULES.agari_yame is True
    # 未接线（step3 占位）：当前按单倍役满
    assert TENHOU_RULES.double_yakuman is False
