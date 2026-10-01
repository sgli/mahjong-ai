"""Tests for fu / han / points scoring and dora."""

from mahjong.replay.state import Meld
from mahjong.rules import (
    WinContext,
    base_points_from,
    dora_tile,
    ron_payment,
    score_hand,
    tsumo_payments,
)
from mahjong.rules.decompose import Group, Structure
from mahjong.rules.yaku import _wait_type


def ts(s):
    return s.split()


def sc(hand_s, win, *, tsumo=False, menzen=True, rw="E", sw="W", melds=(), dora=(), ura=(), **kw):
    ctx = WinContext(round_wind=rw, seat_wind=sw, win_tile=win, is_tsumo=tsumo, is_menzen=menzen, **kw)
    return score_hand(ts(hand_s), melds, win, ctx, dora_indicators=dora, ura_indicators=ura)


def names(r):
    return [n for n, _ in r.yaku]


# -- dora mapping --------------------------------------------------------------
def test_dora_tile_mapping():
    assert dora_tile("1m") == "2m"
    assert dora_tile("9m") == "1m"
    assert dora_tile("5p") == "6p"
    assert dora_tile("E") == "S"
    assert dora_tile("N") == "E"
    assert dora_tile("P") == "F"
    assert dora_tile("C") == "P"


# -- known fu / points ---------------------------------------------------------
def test_pinfu_ron():
    r = sc("2m 3m 4m 5m 6m 7m 2p 3p 4p 6p 7p 8p 1s 1s", "6p")
    assert ("pinfu", 1) in r.yaku
    assert r.fu == 30
    assert r.han == 1
    assert r.base_points == 240
    assert ron_payment(r.base_points, False) == 1000


def test_pinfu_tsumo():
    r = sc("2m 3m 4m 5m 6m 7m 2p 3p 4p 6p 7p 8p 1s 1s", "6p", tsumo=True)
    assert ("pinfu", 1) in r.yaku and ("menzen_tsumo", 1) in r.yaku
    assert r.fu == 20
    assert r.han == 2
    assert tsumo_payments(r.base_points, False) == (700, 400, 400)


def test_tanyao_40_fu_ron():
    r = sc("2m 2m 2m 3m 4m 5m 6m 7m 8m 2p 3p 4p 5p 5p", "5p")
    assert ("tanyao", 1) in r.yaku
    assert r.fu == 40
    assert ron_payment(r.base_points, False) == 1300


def test_dragon_40_fu_ron():
    r = sc("1m 2m 3m 4p 5p 6p 7s 8s 9s P P P 5m 5m", "P")
    assert ("dragon", 1) in r.yaku
    assert r.fu == 40
    assert ron_payment(r.base_points, False) == 1300


def test_chiitoi_25_fu():
    r = sc("1m 1m 2m 2m 3p 3p 4p 4p 5s 5s 6s 6s 7s 7s", "7s")
    assert ("chiitoi", 2) in r.yaku
    assert r.fu == 25
    assert r.han == 2
    assert ron_payment(r.base_points, False) == 1600


def test_dealer_pinfu_ron():
    r = sc("2m 3m 4m 5m 6m 7m 2p 3p 4p 6p 7p 8p 1s 1s", "6p")
    assert ron_payment(r.base_points, True) == 1500


def test_open_tanyao():
    melds = (Meld(kind="pon", tiles=("2m", "2m", "2m"), from_=0, called="2m"),)
    r = sc("3m 4m 5m 6m 7m 8m 2p 3p 4p 5p 5p", "5p", melds=melds, menzen=False)
    assert ("tanyao", 1) in r.yaku
    assert r.fu == 30
    assert ron_payment(r.base_points, False) == 1000


# -- caps (via dora to control han) --------------------------------------------
def _tanyao_hand():
    return "2m 2m 2m 3m 4m 5m 6m 7m 8m 2p 3p 4p 5p 5p"


def test_mangan_5_han():
    r = sc(_tanyao_hand(), "5p", dora=("2m", "2m", "2m", "2m"))  # 4 dora on 3m (one copy) + tanyao = 5 han
    assert r.han == 5
    assert r.base_points == 2000
    assert ron_payment(r.base_points, False) == 8000


def test_haneman_6_han():
    r = sc(_tanyao_hand(), "5p", dora=("2m",) * 5)
    assert r.han == 6
    assert r.base_points == 3000
    assert ron_payment(r.base_points, False) == 12000


def test_baiman_8_han():
    r = sc(_tanyao_hand(), "5p", dora=("2m",) * 7)
    assert r.han == 8
    assert r.base_points == 4000
    assert ron_payment(r.base_points, False) == 16000


def test_sanbaiman_11_han():
    r = sc(_tanyao_hand(), "5p", dora=("2m",) * 10)
    assert r.han == 11
    assert r.base_points == 6000
    assert ron_payment(r.base_points, False) == 24000


def test_kazoe_yakuman_13_han():
    r = sc(_tanyao_hand(), "5p", dora=("2m",) * 12)
    assert r.han == 13
    assert r.base_points == 8000


def test_yakuman_ron():
    r = sc("1m 9m 1p 9p 1s 9s E S W N P F C 1m", "1m")
    assert r.is_yakuman
    assert r.base_points == 8000
    assert ron_payment(r.base_points, False) == 32000
    assert ron_payment(r.base_points, True) == 48000


# -- dora (table / ura / aka) --------------------------------------------------
def test_table_dora():
    r = sc(_tanyao_hand(), "5p", dora=("2m",))  # 2m indicator -> 3m dora (one 3m present)
    assert r.dora == 1
    assert r.han == 2


def test_ura_dora_with_riichi():
    r = sc(_tanyao_hand(), "5p", ura=("2m",), is_riichi=True)
    assert r.dora == 1
    assert ("riichi", 1) in r.yaku


def test_aka_dora_red_five():
    hand = "2m 3m 4m 4m 5m 6m 6m 7m 8m 2p 3p 4p 5pr 5p"
    r = sc(hand, "5p")
    assert r.dora == 1  # one red five (5pr) = 1 aka dora


# -- base point cap table (direct) ---------------------------------------------
def test_base_points_cap_table():
    assert base_points_from(1, 30) == 240
    assert base_points_from(2, 30) == 480
    assert base_points_from(3, 30) == 960
    assert base_points_from(3, 40) == 1280
    assert base_points_from(4, 30) == 1920
    assert base_points_from(4, 40) == 2000  # mangan (4 han 40 fu)
    assert base_points_from(3, 70) == 2000  # mangan (3 han 70 fu)
    assert base_points_from(5, 20) == 2000
    assert base_points_from(6, 30) == 3000
    assert base_points_from(8, 25) == 4000
    assert base_points_from(11, 30) == 6000
    assert base_points_from(13, 30) == 8000
    assert base_points_from(2, 25, yakuman_count=1) == 8000


# -- situational yaku in scoring ------------------------------------------------
def test_situational_han():
    hand = "1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p 5p"
    assert sc(hand, "5p", is_rinshan=True).han >= 1
    assert sc(hand, "5p", is_chankan=True).han >= 1
    assert sc(hand, "5p", tsumo=True, is_haitei=True).han >= 2
    assert sc(hand, "5p", is_houtei=True).han >= 1


# -- wait shape boundaries (F1 regression) --------------------------------------
def _wait(run_tiles, win, pair="5s"):
    structure = Structure(sets=(Group("run", tuple(sorted(run_tiles)), False),), pair=pair)
    return _wait_type(structure, win)


def test_wait_type_penchan_ryanmen_boundaries():
    assert _wait(("1m", "2m", "3m"), "1m") == "ryanmen"
    assert _wait(("1m", "2m", "3m"), "3m") == "penchan"
    assert _wait(("7m", "8m", "9m"), "7m") == "penchan"
    assert _wait(("7m", "8m", "9m"), "9m") == "ryanmen"
    assert _wait(("4m", "5m", "6m"), "4m") == "ryanmen"
    assert _wait(("4m", "5m", "6m"), "6m") == "ryanmen"
    assert _wait(("4m", "5m", "6m"), "5m") == "kanchan"


def test_wait_type_all_suits_boundaries():
    for suit in "mps":
        assert _wait((f"1{suit}", f"2{suit}", f"3{suit}"), f"1{suit}") == "ryanmen"
        assert _wait((f"1{suit}", f"2{suit}", f"3{suit}"), f"3{suit}") == "penchan"
        assert _wait((f"7{suit}", f"8{suit}", f"9{suit}"), f"7{suit}") == "penchan"
        assert _wait((f"7{suit}", f"8{suit}", f"9{suit}"), f"9{suit}") == "ryanmen"


def test_pinfu_waiting_on_1_is_ryanmen():
    r = sc("1m 2m 3m 4p 5p 6p 7s 8s 9s 2p 3p 4p 5s 5s", "1m")
    assert ("pinfu", 1) in r.yaku
    assert r.has_yaku
    assert r.fu == 30


def test_123_wait_3_is_penchan_not_pinfu():
    r = sc("1m 2m 3m 4p 5p 6p 7s 8s 9s 2p 3p 4p 5s 5s", "3m")
    assert ("pinfu", 1) not in r.yaku
    assert r.fu == 40  # 30 + 2 penchan = 32 -> round 40


def test_789_wait_7_is_penchan():
    r = sc("7m 8m 9m 4p 5p 6p 7s 8s 9s 2p 3p 4p 5s 5s", "7m")
    assert ("pinfu", 1) not in r.yaku
    assert r.fu == 40


def test_789_wait_9_is_ryanmen_pinfu():
    r = sc("7m 8m 9m 4p 5p 6p 7s 8s 9s 2p 3p 4p 5s 5s", "9m")
    assert ("pinfu", 1) in r.yaku
    assert r.fu == 30
