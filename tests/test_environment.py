"""Tests for the Mahjong Environment (Phase 6b)."""

import os
import random
from pathlib import Path

import pytest

from mahjong.decision.action import Action, ActionType
from mahjong.environment import KYUUSHU_ACTION, MahjongEnv, RewardConfig, legal_actions_from_replay
from mahjong.environment.state import (
    PHASE_CHANKAN,
    PHASE_DISCARD,
    PHASE_RESPONSE,
    EnvState,
)
from mahjong.replay.state import Meld


def _random_play(seed, *, prefer_pass=True):
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
        assert steps < 100_000, "too many steps"
    return env, steps


def _make_winning_hand():
    return ["2m", "2m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "2p", "3p", "4p", "5p", "5p"]


def _set_discard_state(env, seat, hand):
    env.state.players[seat].hand = sorted(hand)
    env.state.players[seat].drawn_this_turn = True
    env.state.players[seat].last_drawn = None
    env.state.phase = PHASE_DISCARD
    env.state.turn = seat


# -- deal / determinism --------------------------------------------------------
def test_deal_sizes():
    env = MahjongEnv(seed=1)
    env.reset()
    assert [len(p.hand) for p in env.state.players] == [14, 13, 13, 13]
    assert env.state.turn == env.state.oya == 0
    assert len(env.state.dora_indicators) == 1


def test_deal_deterministic():
    a = MahjongEnv(seed=7)
    a.reset()
    b = MahjongEnv(seed=7)
    b.reset()
    assert [p.hand for p in a.state.players] == [p.hand for p in b.state.players]
    assert a.state.dora_indicators == b.state.dora_indicators


def test_full_hanchan_selfplay():
    env, steps = _random_play(3)
    assert env.done()
    assert env.state.final_ranks is not None
    assert sorted(env.state.final_ranks) == [0, 1, 2, 3]
    assert steps > 100  # a full game takes many steps


def test_full_hanchan_deterministic():
    _, s1 = _random_play(11)
    env2, s2 = _random_play(11)
    assert s1 == s2
    assert env2.state.scores == env2.state.scores  # same env built twice
    # rebuild fresh envs and compare scores
    e1 = MahjongEnv(seed=11)
    e1.reset()
    rng1 = random.Random(11)
    while not e1.done():
        legal = e1.legal_actions()
        act = next((a for a in legal if getattr(a, "type", None) is ActionType.PASS), None) or rng1.choice(legal)
        e1.step(act)
    e2 = MahjongEnv(seed=11)
    e2.reset()
    rng2 = random.Random(11)
    while not e2.done():
        legal = e2.legal_actions()
        act = next((a for a in legal if getattr(a, "type", None) is ActionType.PASS), None) or rng2.choice(legal)
        e2.step(act)
    assert e1.state.scores == e2.state.scores
    assert e1.state.final_ranks == e2.state.final_ranks


# -- basic actions -------------------------------------------------------------
def test_discard_moves_tile_to_river():
    env = MahjongEnv(seed=2)
    env.reset()
    seat = env.state.turn
    legal = env.legal_actions(seat)
    d = next(a for a in legal if a.type is ActionType.DISCARD)
    before = len(env.state.players[seat].hand)
    env.step(d)
    assert len(env.state.players[seat].hand) == before - 1
    assert env.state.players[seat].discards[-1] == d.tile
    assert env.state.phase == PHASE_RESPONSE


def test_illegal_action_rejected():
    env = MahjongEnv(seed=2)
    env.reset()
    seat = env.state.turn
    legal = env.legal_actions(seat)
    tiles_in_hand = set(env.state.players[seat].hand)
    bad = Action(ActionType.DISCARD, tile=("9s" if "9s" not in tiles_in_hand else "9p"))
    import pytest

    with pytest.raises(ValueError):
        env.step(bad)


def test_riichi_deducts_score_and_adds_kyotaku():
    env = MahjongEnv(seed=2)
    env.reset()
    # craft a closed tenpai hand so riichi is legal
    p = env.state.players[0]
    p.hand = sorted(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p", "5p"])
    p.drawn_this_turn = True
    p.score = 25000
    env.state.turn = 0
    env.state.phase = PHASE_DISCARD
    legal = env.legal_actions(0)
    r = next(a for a in legal if a.type is ActionType.RIICHI)
    env.step(r)
    assert env.state.players[0].riichi is True
    assert env.state.players[0].score == 24000
    assert env.state.kyotaku == 1


# -- tsumo / ron settlement ----------------------------------------------------
def test_tsumo_win_settlement():
    env = MahjongEnv(seed=2)
    env.reset()
    env.state.players[0].hand = _make_winning_hand()
    env.state.players[0].last_drawn = "5p"
    env.state.players[0].drawn_this_turn = True
    env.state.phase = PHASE_DISCARD
    env.state.turn = 0
    legal = env.legal_actions(0)
    assert any(a.type is ActionType.TSUMO for a in legal)
    rewards, done = env.step(Action(ActionType.TSUMO))
    assert done is False  # kyoku ends, game continues
    assert env.state.kyoku == 1 and env.state.honba >= 1  # dealer tsumo -> renchan


def test_ron_win_settlement_with_kyotaku():
    env = MahjongEnv(seed=2)
    env.reset()
    # seat 1 tenpai waiting 5p; seat 0 discards 5p
    env.state.players[1].hand = sorted(
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"]
    )
    env.state.kyotaku = 1
    env.state.discarder = 0
    env.state.discard_tile = "5p"
    env.state.pending_responders = [1, 2, 3]
    env.state.phase = PHASE_RESPONSE
    env.state.turn = 1
    legal = env.legal_actions(1)
    assert any(a.type is ActionType.RON for a in legal)
    before = env.state.scores[:]
    env.step(Action(ActionType.RON, target=0))
    env.step(Action(ActionType.PASS))  # seat 2
    env.step(Action(ActionType.PASS))  # seat 3 -> resolve ron
    assert env.state.scores[1] > before[1]
    assert env.state.scores[0] < before[0]
    assert env.state.kyotaku == 0


def test_furiten_blocks_ron():
    env = MahjongEnv(seed=2)
    env.reset()
    env.state.players[1].hand = sorted(
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"]
    )
    env.state.players[1].discards = ["5p"]  # discard furiten on 5p
    env.state.discarder = 0
    env.state.discard_tile = "5p"
    env.state.pending_responders = [1, 2, 3]
    env.state.phase = PHASE_RESPONSE
    env.state.turn = 1
    legal = env.legal_actions(1)
    assert not any(a.type is ActionType.RON for a in legal)


# -- observation no hidden info -------------------------------------------------
def test_observation_has_no_hidden_info():
    env = MahjongEnv(seed=2)
    env.reset()
    obs = env.observation(1)
    assert obs.hand == tuple(env.state.players[1].hand)
    assert not any("opponent" in n and "hand" in n for n in obs.__dataclass_fields__)
    assert "ura_markers" not in obs.__dataclass_fields__
    # opponents' concealed hands are not exposed (only their melds / rivers)
    assert obs.opponents_melds[0] == ()


# -- reward --------------------------------------------------------------------
def test_reward_accumulates_and_final_bonus():
    env = MahjongEnv(seed=3, reward_config=RewardConfig(score_delta_scale=0.001, placement_bonus=(2.0, 1.0, -1.0, -2.0)))
    env, _ = _random_play(3)
    assert env.done()
    # placement bonuses applied; check the rank-1 seat got its bonus
    rank0 = env.state.final_ranks.index(0)
    assert env.reward(rank0) >= 2.0 - 1e-6


# -- chankan (robbing a kakan) --------------------------------------------------
def test_chankan_ron_window():
    env = MahjongEnv(seed=2)
    env.reset()
    # seat 0 has a pon of 9p + a 9p in hand (for kakan); seat 1 waits on 9p
    env.state.players[0].hand = sorted(["9p", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p"])
    env.state.players[0].melds = [Meld(kind="pon", tiles=("9p", "9p", "9p"), from_=1, called="9p")]
    env.state.players[0].drawn_this_turn = True
    env.state.phase = PHASE_DISCARD
    env.state.turn = 0
    legal = env.legal_actions(0)
    kakan = next((a for a in legal if a.type is ActionType.KAN and a.kan_kind == "kakan"), None)
    assert kakan is not None
    env.step(kakan)
    assert env.state.phase == PHASE_CHANKAN


# -- exhaustive draw -----------------------------------------------------------
def test_cross_validation_with_replay():
    """Environment legal actions must contain the human action from real replays."""
    data_dir = Path(os.environ.get("MAHJONG_TEST_DATA_DIR", "F:/Mahjong AI/mahjong DB/tenhou-houou-2026"))
    if not data_dir.is_dir():
        pytest.skip("real data not available")

    from mahjong.decision.extractor import DecisionExtractor
    from mahjong.parser import MjaiParser
    from mahjong.replay import ReplayEngine

    files = sorted(data_dir.glob("*.mjai.json"))[:20]
    parser = MjaiParser()
    checked = 0
    for path in files:
        events = list(parser.parse_file(path))
        states = list(ReplayEngine(game_id=path.stem).replay(events))
        extractor = DecisionExtractor(game_id=path.stem)
        for sample in extractor.extract(states, events):
            i = sample.decision_point.step
            replay_state = states[i]
            event = events[i]
            kind = sample.decision_point.kind
            if kind == "discard":
                drawn_tile = event.pai if event.type.value == "tsumo" else None
                legal = legal_actions_from_replay(
                    replay_state, seat=sample.player_id, phase=PHASE_DISCARD, drawn_tile=drawn_tile
                )
            else:  # response (dahai / chankan)
                phase = PHASE_CHANKAN if event.type.value == "kakan" else PHASE_RESPONSE
                legal = legal_actions_from_replay(
                    replay_state,
                    seat=sample.player_id,
                    phase=phase,
                    discarder=event.actor,
                    discard_tile=event.pai,
                )
            assert sample.action in legal, (
                f"{path.name}: step {i} seat {sample.player_id} action {sample.action!r} "
                f"not in {[repr(a) for a in legal]}"
            )
            checked += 1
    assert checked > 1000


def test_exhaustive_draw_tenpai_settlement():
    env = MahjongEnv(seed=2)
    env.reset()
    for s in range(4):
        env.state.players[s].hand = sorted([f"{n}m" for n in "123456789"] + ["1p", "2p", "3p", "5p"])
        env.state.players[s].drawn_this_turn = False
    # make seats 0,1 tenpai (13-tile) and 2,3 noten
    env.state.players[0].hand = sorted(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"])
    env.state.players[1].hand = sorted(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"])
    env.state.players[2].hand = sorted(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "4p", "5p"])
    env.state.players[3].hand = sorted(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "4p", "5p"])
    before = env.state.scores[:]
    env._resolve_ryukyoku(abort=False)
    assert sum(env.state.scores) == sum(before)
    assert env.state.scores[0] > before[0] or env.state.scores[1] > before[1]  # tenpai players gain
