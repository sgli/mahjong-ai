"""Regression tests for A1/A2 + V3 (kiriage mangan + ron-triplet fu)."""

from mahjong.replay.state import Meld
from mahjong.rules import TENHOU_RULES, WinContext, score_hand


def test_fu_floor_30_open_ryanmen_ron():
    # open all-runs + non-yakuhai pair + ryanmen ron -> 30 fu (not 20)
    hand = ["3m", "4m", "5m", "6m", "7m", "8m", "1s", "2s", "3s", "5p", "5p"]
    melds = [Meld(kind="chi", tiles=("2p", "3p", "4p"), from_=1, called="2p")]
    ctx = WinContext(round_wind="E", seat_wind="W", win_tile="3m", is_tsumo=False, is_menzen=False)
    r = score_hand(hand, melds, "3m", ctx)
    assert r.fu == 30


def test_pinfu_tsumo_20_ron_30_unchanged():
    # pinfu branch unchanged
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "2p", "3p", "4p", "5s", "5s"]
    ctx_tsumo = WinContext(round_wind="E", seat_wind="W", win_tile="1m", is_tsumo=True, is_menzen=True)
    assert score_hand(hand, [], "1m", ctx_tsumo).fu == 20
    ctx_ron = WinContext(round_wind="E", seat_wind="W", win_tile="1m", is_tsumo=False, is_menzen=True)
    assert score_hand(hand, [], "1m", ctx_ron).fu == 30


def test_sanankou_ron_shanpon_excluded():
    hand = ["1m", "1m", "1m", "2p", "2p", "2p", "6s", "7s", "8s", "5s", "5s", "5s", "7s", "7s"]
    ctx_ron = WinContext(round_wind="E", seat_wind="W", win_tile="5s", is_tsumo=False, is_menzen=True)
    r_ron = score_hand(hand, [], "5s", ctx_ron)
    assert ("sanankou", 2) not in r_ron.yaku


def test_sanankou_tsumo_shanpon_counted():
    hand = ["1m", "1m", "1m", "2p", "2p", "2p", "6s", "7s", "8s", "5s", "5s", "5s", "7s", "7s"]
    ctx_tsumo = WinContext(round_wind="E", seat_wind="W", win_tile="5s", is_tsumo=True, is_menzen=True)
    r_tsumo = score_hand(hand, [], "5s", ctx_tsumo)
    assert ("sanankou", 2) in r_tsumo.yaku


def test_tenhou_kiriage_mangan_disabled():
    # 数据实证：天凤 4番30符 = 11600（非满贯），不用切上满贯
    assert TENHOU_RULES.kiriage_mangan is False


def test_ron_completed_triplet_is_open_fu():
    # 123456789m + 999s(ron 完成双碰) + 55p -> 999s 按明刻(4符)：30+4=34 -> 40符
    hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "9s", "9s", "9s", "5p", "5p"]
    ctx = WinContext(round_wind="E", seat_wind="W", win_tile="9s", is_tsumo=False, is_menzen=True)
    r = score_hand(hand, [], "9s", ctx)
    assert r.fu == 40


def test_tsumo_completed_triplet_stays_closed_fu():
    # 同手牌自摸：999s 按暗刻(8符)：22+8=30 -> 30符
    hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "9s", "9s", "9s", "5p", "5p"]
    ctx = WinContext(round_wind="E", seat_wind="W", win_tile="9s", is_tsumo=True, is_menzen=True)
    r = score_hand(hand, [], "9s", ctx)
    assert r.fu == 30


def test_double_wind_pair_4_fu():
    from mahjong.replay.state import Meld

    hand = ["1m", "2m", "3m", "8s", "8s", "8s", "C", "C", "C", "E", "E"]
    melds = [Meld(kind="pon", tiles=("P", "P", "P"), from_=0, called="P")]
    ctx = WinContext(round_wind="E", seat_wind="E", win_tile="8s", is_tsumo=True, is_menzen=False)
    r = score_hand(hand, melds, "8s", ctx)
    # 连风雀头 +4：22 + 4(888s) + 8(CCC) + 4(PPP) + 4(EE连风) = 42 -> 50
    assert r.fu == 50


def test_high_point_chooses_pinfu_over_kanchan():
    # win=6m 可作 456m(两面) 或 567m(嵌张)；含平和时应选两面(平和 20符)
    hand = ["4m", "5m", "6m", "5m", "6m", "7m", "4p", "5p", "6p", "7s", "8s", "9s", "5p", "5p"]
    ctx = WinContext(round_wind="E", seat_wind="W", win_tile="6m", is_tsumo=True, is_menzen=True)
    r = score_hand(hand, [], "6m", ctx)
    assert ("pinfu", 1) in r.yaku
    assert r.fu == 20


def test_high_point_chooses_kanchan_without_pinfu():
    # win=6m 可作 456m(两面,0符) 或 567m(嵌张,+2符)；无平和时选嵌张(40符)
    hand = ["4m", "4p", "5m", "5m", "5p", "6m", "6m", "6p", "6s", "6s", "7m", "N", "N", "N"]
    ctx = WinContext(round_wind="E", seat_wind="W", win_tile="6m", is_tsumo=True, is_menzen=True)
    r = score_hand(hand, [], "6m", ctx)
    assert r.fu == 40


def test_aka_dora_counts_in_hand_and_meld():
    from mahjong.replay.state import Meld
    from mahjong.rules.dora import count_dora

    # 赤5 在副露中计 +1
    meld = Meld(kind="chi", tiles=("5mr", "6m", "7m"), from_=0, called="5mr")
    assert count_dora([], [meld], [], [], include_aka=True) == 1
    # 赤5 同时吃「5 的表宝牌」+「红宝牌」（4m 指示 → 5m）
    assert count_dora(["5mr"], [], ["4m"], [], include_aka=True) == 2


def test_ura_dora_only_for_riichi():
    hand = ["2s", "3s", "4m", "4s", "5m", "5p", "6m", "6p", "7p", "7p", "8p", "9p", "9s", "9s"]
    # 非立直：ura(3m→4m) 不应计入
    ctx_ron = WinContext(round_wind="E", seat_wind="W", win_tile="4m", is_tsumo=False, is_menzen=True, is_riichi=False)
    r = score_hand(hand, [], "4m", ctx_ron, dora_indicators=["1p"], ura_indicators=["3m"])
    assert r.dora == 0
    # 立直：ura 计入
    ctx_riichi = WinContext(round_wind="E", seat_wind="W", win_tile="4m", is_tsumo=False, is_menzen=True, is_riichi=True)
    r2 = score_hand(hand, [], "4m", ctx_riichi, dora_indicators=["1p"], ura_indicators=["3m"])
    assert r2.dora == 1


def test_ryukyoku_tenpai_transfer_no_honba():
    from mahjong.environment import MahjongEnv

    env = MahjongEnv(seed=1)
    env.reset()
    env.state.honba = 1
    tenpai_hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"]
    noten_hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "4p", "5p"]
    env.state.players[0].hand = sorted(tenpai_hand)
    for s in (1, 2, 3):
        env.state.players[s].hand = sorted(noten_hand)
    before = env.state.scores[:]
    env._resolve_ryukyoku(abort=False)
    # 1 家听牌：+3000，其余各 -1000（不叠加本场）
    assert env.state.scores[0] == before[0] + 3000
    assert env.state.scores[1] == before[1] - 1000
    assert env.state.scores[2] == before[2] - 1000
    assert env.state.scores[3] == before[3] - 1000
