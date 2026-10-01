"""Regression tests for abortive draws + agari-yame (Phase cleanup)."""

from mahjong.decision.action import Action, ActionType
from mahjong.environment import MahjongEnv
from mahjong.environment.state import PHASE_DISCARD, PHASE_RESPONSE
from mahjong.replay.state import Meld


def _winning_hand():
    return ["2m", "2m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "2p", "3p", "4p", "5p", "5p"]


def _respond_all_pass(env, discarder=0, tile="E"):
    env.state.discarder = discarder
    env.state.discard_tile = tile
    env.state.phase = PHASE_RESPONSE
    env.state.response_choices = {s: Action(ActionType.PASS) for s in ((discarder + 1) % 4, (discarder + 2) % 4, (discarder + 3) % 4)}
    env.state.pending_responders = []


# -- item 1: four wind ---------------------------------------------------------
def test_four_wind_abort_detection():
    env = MahjongEnv(seed=1)
    env.reset()
    for s in range(4):
        env.state.players[s].discards = ["E"]
        env.state.players[s].melds = []
    assert env._check_four_wind_abort() is True
    env.state.players[3].discards = ["S"]
    assert env._check_four_wind_abort() is False


def test_four_wind_abort_ends_kyoku():
    env = MahjongEnv(seed=1)
    env.reset()
    for s in range(4):
        env.state.players[s].discards = ["E"]
    _respond_all_pass(env, discarder=0, tile="E")
    honba = env.state.honba
    env._resolve_responses()
    assert env.state.honba == honba + 1  # abort -> renchan
    assert env.state.phase == PHASE_DISCARD


# -- item 2: four kan ----------------------------------------------------------
def test_four_kan_abort_two_players():
    env = MahjongEnv(seed=1)
    env.reset()
    env.state.players[0].melds = [
        Meld(kind="ankan", tiles=("1m", "1m", "1m", "1m"), from_=None, called=None),
        Meld(kind="daiminkan", tiles=("2m", "2m", "2m", "2m"), from_=1, called="2m"),
    ]
    env.state.players[1].melds = [
        Meld(kind="ankan", tiles=("3m", "3m", "3m", "3m"), from_=None, called=None),
        Meld(kind="kakan", tiles=("4m", "4m", "4m", "4m"), from_=0, called="4m"),
    ]
    assert env._check_four_kan_abort() is True


def test_four_kan_no_abort_one_player():
    env = MahjongEnv(seed=1)
    env.reset()
    env.state.players[0].melds = [
        Meld(kind="ankan", tiles=(f"{r}m",) * 4, from_=None, called=None) for r in ("1", "2", "3", "4")
    ]
    assert env._check_four_kan_abort() is False


# -- item 3: four riichi -------------------------------------------------------
def test_four_riichi_abort():
    env = MahjongEnv(seed=1)
    env.reset()
    for s in range(4):
        env.state.players[s].riichi = True
    _respond_all_pass(env, discarder=0, tile="5p")
    honba = env.state.honba
    env._resolve_responses()
    assert env.state.honba == honba + 1


# -- item 4: agari-yame --------------------------------------------------------
def _setup_tsumo(env, round_wind, kyoku, scores, seat=0):
    env.state.round_wind = round_wind
    env.state.kyoku = kyoku
    env.state.scores = scores
    env.state.oya = seat
    env.state.players[seat].hand = sorted(_winning_hand())
    env.state.players[seat].last_drawn = "5p"
    env.state.players[seat].drawn_this_turn = True
    env.state.turn = seat
    env.state.phase = PHASE_DISCARD


def test_agari_yame_all_last_top_ends_game():
    env = MahjongEnv(seed=1)
    env.reset()
    _setup_tsumo(env, "S", 4, [30000, 25000, 20000, 25000])
    legal = env.legal_actions(0)
    assert any(a.type is ActionType.TSUMO for a in legal)
    rewards, done = env.step(Action(ActionType.TSUMO))
    assert done is True
    assert env.state.game_over is True


def test_agari_yame_not_top_continues():
    env = MahjongEnv(seed=1)
    env.reset()
    _setup_tsumo(env, "S", 4, [25000, 30000, 20000, 25000])  # seat 1 is top
    rewards, done = env.step(Action(ActionType.TSUMO))
    assert done is False
    assert env.state.game_over is False


def test_agari_yame_non_all_last_not_end():
    env = MahjongEnv(seed=1)
    env.reset()
    _setup_tsumo(env, "E", 1, [30000, 25000, 20000, 25000])
    rewards, done = env.step(Action(ActionType.TSUMO))
    assert done is False
    assert env.state.game_over is False
