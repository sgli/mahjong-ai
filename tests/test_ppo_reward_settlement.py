"""Reward regression: ron/tsumo settlement, dealer continuation, honba/kyotaku, trajectory↔delta (§10.1)."""

from __future__ import annotations

import random

import torch

from mahjong.decision.action import Action, ActionType
from mahjong.environment import MahjongEnv, PHASE_DISCARD
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder
from mahjong.model import MLPPolicy
from mahjong.training import collect_trajectory

# 14-tile closed tanyao winning hand (222m 345m 678m 234p 55p)
_WIN_14 = ["2m", "2m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "2p", "3p", "4p", "5p", "5p"]


def _setup(env, seat, hand, *, dora=("1m",), honba=0, kyotaku=0, oya=0, drawn=True):
    p = env.state.players[seat]
    p.hand = list(hand)
    p.melds = []
    p.riichi = False
    p.drawn_this_turn = drawn
    p.last_drawn = hand[-1] if drawn else None
    env.state.dora_indicators = list(dora)
    env.state.ura_indicators = ()
    env.state.honba = honba
    env.state.kyotaku = kyotaku
    env.state.oya = oya
    env.state.scores = [25000, 25000, 25000, 25000]
    env._accumulated = {s: 0.0 for s in range(4)}


def test_ron_payment_attribution():
    env = MahjongEnv(seed=0)
    env.reset()
    # seat 1 rons seat 0's discard; seat 1 waits on 5p (13 tiles)
    _setup(env, 1, _WIN_14[:-1], drawn=False, kyotaku=0)
    env.state.discard_tile = "5p"
    rewards = env._resolve_ron(seat=1, discarder=0)
    assert rewards[1] > 0  # winner gains
    assert rewards[0] < 0  # discarder pays
    assert rewards[2] == 0.0  # other two unaffected
    assert rewards[3] == 0.0
    # conservation (no kyotaku): winner + discarder == 0
    assert abs(rewards[0] + rewards[1]) < 1e-9


def test_tsumo_dealer_settlement():
    env = MahjongEnv(seed=0)
    env.reset()
    _setup(env, 0, _WIN_14, oya=0, kyotaku=0)
    rewards = env._resolve_tsumo(seat=0)
    assert rewards[0] > 0
    # dealer tsumo: three non-dealers each pay the same amount
    assert abs(rewards[1] - rewards[2]) < 1e-9
    assert abs(rewards[2] - rewards[3]) < 1e-9
    # conservation: winner + 3 losers == 0 (no kyotaku)
    assert abs(sum(rewards.values())) < 1e-9


def test_tsumo_nondealer_settlement():
    env = MahjongEnv(seed=0)
    env.reset()
    _setup(env, 1, _WIN_14, oya=0, kyotaku=0)
    rewards = env._resolve_tsumo(seat=1)
    assert rewards[1] > 0
    # non-dealer tsumo: dealer (seat 0) pays twice as much as the two others
    assert abs(rewards[0] - 2 * rewards[2]) < 1e-6
    assert abs(rewards[2] - rewards[3]) < 1e-9
    assert abs(sum(rewards.values())) < 1e-9


def test_dealer_continuation_no_terminal_reward():
    env = MahjongEnv(seed=0)
    env.reset()
    _setup(env, 0, _WIN_14, oya=0, kyotaku=0)
    rewards = env._resolve_tsumo(seat=0)  # dealer wins → renchan
    # renchan 推进本场，但不产生 placement bonus / terminal reward
    assert env.state.honba == 1
    assert not env.done()
    # 返回的 rewards 只有 score delta（无 placement bonus），且守恒（kyotaku=0）
    assert abs(sum(rewards.values())) < 1e-9
    # 非胜者座位只有负 delta（无任何 placement bonus 混入）
    assert all(rewards[s] <= 0 for s in (1, 2, 3))


def test_honba_kyotaku_not_double_counted():
    env = MahjongEnv(seed=0)
    env.reset()
    _setup(env, 0, _WIN_14, oya=0, honba=1, kyotaku=1)
    rewards = env._resolve_tsumo(seat=0)
    # kyotaku (供托) 只加到 winner 一次；honba 只计入各家的付款（守恒）
    # winner 的 delta = 3*each + kyotaku*1000；sum(rewards) = kyotaku*1000*scale
    expected_sum = env.reward_config.score_delta_scale * 1000  # kyotaku * 1000 * scale
    assert abs(sum(rewards.values()) - expected_sum) < 1e-9
    # kyotaku 清零，只计一次
    assert env.state.kyotaku == 0


def _play_no_riichi(env, seed):
    """Play a full game always discarding the first tile (never riichi/tsumo/ron/call)."""
    rng = random.Random(seed)
    steps = 0
    while not env.done():
        seat = env.state.turn
        legal = env.legal_actions(seat)
        action = next((a for a in legal if getattr(a, "type", None) is ActionType.DISCARD), None)
        if action is None:
            action = rng.choice(legal)
        env.step(action)
        steps += 1
        assert steps < 100_000
    return steps


def test_trajectory_score_delta_decomposition_no_riichi():
    """§10.1 第 8 项：无立直场景下 sum(rewards) == scale*(final_score-25000) + placement_bonus[rank]。

    注：有立直时存在已知 bug（立直棒 -1000 只改 p.score、不入 state.scores 也不返回 reward delta），
    会在有立直的对局中破坏此关系，故本测试用无立直对局做真实断言。
    """
    env = MahjongEnv(seed=19)
    env.reset()
    _play_no_riichi(env, 19)
    assert env.done()
    scale = env.reward_config.score_delta_scale
    for s in range(4):
        rank = env.state.final_ranks[s]
        score_delta = env.state.scores[s] - 25000
        expected = scale * score_delta + env.reward_config.placement_bonus[rank]
        assert abs(env.reward(s) - expected) < 1e-6, f"seat {s}: {env.reward(s)} != {expected}"


def test_riichi_immediate_settlement():
    """立直宣言立即扣 1000 入 state.scores + kyotaku+1 + reward 含 -scale*1000（§10.1 用户裁定）。"""
    env = MahjongEnv(seed=0)
    env.reset()
    seat = 0
    p = env.state.players[seat]
    p.hand = list(_WIN_14)  # 14 张，打 5p 后听 5p
    p.melds = []
    p.riichi = False
    p.score = 25000
    p.drawn_this_turn = True
    p.last_drawn = "5p"
    env.state.oya = 0
    env.state.phase = PHASE_DISCARD
    env.state.turn = seat
    env._accumulated = {s: 0.0 for s in range(4)}

    before = env.state.scores[seat]
    rewards, _ = env.step(Action(ActionType.RIICHI, tile="5p"))
    assert p.riichi is True
    assert env.state.scores[seat] == before - 1000
    assert env.state.kyotaku == 1
    assert rewards[seat] == -env.reward_config.score_delta_scale * 1000


def test_trajectory_matches_score_delta_full_game():
    """全对局（任意是否有立直）：sum(traj rewards) == scale*(final_score-25000) + placement_bonus[rank]。"""
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    encoder = ObservationEncoder()
    env = MahjongEnv(seed=13)
    traj, steps, illegal = collect_trajectory(env, model, encoder, torch.device("cpu"), max_steps=20000)
    assert illegal == 0
    assert env.done()
    scale = env.reward_config.score_delta_scale
    for s in range(4):
        total = sum(st.reward for st in traj[s].steps)
        rank = env.state.final_ranks[s]
        score_delta = env.state.scores[s] - 25000
        expected = scale * score_delta + env.reward_config.placement_bonus[rank]
        assert abs(total - expected) < 1e-6, f"seat {s}: {total} != {expected}"


def test_conservation_scores_plus_kyotaku():
    """全对局守恒：sum(state.scores) + kyotaku*1000 == 100000。"""
    env = MahjongEnv(seed=3)
    env.reset()
    rng = random.Random(3)
    steps = 0
    while not env.done():
        seat = env.state.turn
        legal = env.legal_actions(seat)
        action = next((a for a in legal if getattr(a, "type", None) is ActionType.PASS), None)
        if action is None:
            action = rng.choice(legal)
        env.step(action)
        steps += 1
        assert steps < 100_000
    assert sum(env.state.scores) + env.state.kyotaku * 1000 == 100_000
