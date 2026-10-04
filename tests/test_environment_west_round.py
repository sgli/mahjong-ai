"""West round (西入) end-to-end regression (Phase 10.2 §7/§9)."""

from __future__ import annotations

from mahjong.environment import MahjongEnv


def test_west_round_triggers_when_no_player_reaches_30000():
    env = MahjongEnv(seed=0)
    env.reset()
    env.state.round_wind = "S"
    env.state.kyoku = 4
    env.state.oya = 0
    env.state.scores = [25000, 25000, 25000, 25000]  # 无人 >= 30000
    env._end_kyoku(renchan=False)
    assert env.state.round_wind == "W"  # 西入
    assert env.state.kyoku == 1
    assert not env.done()
    assert sum(env.state.scores) + env.state.kyotaku * 1000 == 100_000


def test_no_west_round_when_player_reaches_30000():
    env = MahjongEnv(seed=0)
    env.reset()
    env.state.round_wind = "S"
    env.state.kyoku = 4
    env.state.oya = 0
    env.state.scores = [32000, 25000, 25000, 25000]  # 有人 >= 30000
    env._end_kyoku(renchan=False)
    assert env.done()  # 终局，不西入
    assert env.state.final_ranks is not None
    assert sorted(env.state.final_ranks) == [0, 1, 2, 3]
    # 1 位是最高分（32000）
    assert env.state.final_ranks[0] == 0


def test_west_round_final_settlement_ranks():
    env = MahjongEnv(seed=0)
    env.reset()
    env.state.round_wind = "S"
    env.state.kyoku = 4
    env.state.oya = 0
    env.state.scores = [26000, 27000, 25000, 24000]
    env._end_kyoku(renchan=False)  # 进入西场
    assert env.state.round_wind == "W"
    # 西场最后一局结束（无连庄）→ 终局，顺位按分数排序
    env.state.round_wind = "W"
    env.state.kyoku = 4
    env._end_kyoku(renchan=False)
    assert env.done()
    assert sorted(env.state.final_ranks) == [0, 1, 2, 3]
    # 1 位是最高分（27000）
    assert env.state.final_ranks[1] == 0
