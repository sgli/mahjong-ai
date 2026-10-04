"""Double yakuman (方案 A) regression (Phase 10.3 §7)."""

from __future__ import annotations

from mahjong.replay.state import Meld
from mahjong.rules import WinContext, score_hand


def _ctx(win_tile):
    return WinContext(round_wind="E", seat_wind="S", win_tile=win_tile, is_tsumo=True, is_menzen=False)


def test_daisuushii_double():
    # 大四喜：3 副露风刻 + 手内 1 暗风刻 + 雀头（非四暗刻）
    melds = [
        Meld(kind="pon", tiles=("E", "E", "E"), from_=1, called="E"),
        Meld(kind="pon", tiles=("S", "S", "S"), from_=2, called="S"),
        Meld(kind="pon", tiles=("W", "W", "W"), from_=3, called="W"),
    ]
    hand = ["N", "N", "N", "5p", "5p"]
    single = score_hand(hand, melds, "5p", _ctx("5p"), double_yakuman=False)
    double = score_hand(hand, melds, "5p", _ctx("5p"), double_yakuman=True)
    assert single.base_points == 8000
    assert double.base_points == 16000
    assert double.yakuman.count("daisuushii") == 2


def test_kokushi_13sided_double():
    hand = ["1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C", "1m"]
    single = score_hand(hand, [], "1m", _ctx("1m"), double_yakuman=False)
    double = score_hand(hand, [], "1m", _ctx("1m"), double_yakuman=True)
    assert single.base_points == 8000
    assert double.base_points == 16000
    assert double.yakuman.count("kokushi_musou") == 2


def test_non_double_yakuman_unaffected():
    # 大三元：非双倍役满，double_yakuman 开关不应影响
    hand = ["P", "P", "P", "F", "F", "F", "C", "C", "C", "1m", "2m", "3m", "5p", "5p"]
    single = score_hand(hand, [], "5p", _ctx("5p"), double_yakuman=False)
    double = score_hand(hand, [], "5p", _ctx("5p"), double_yakuman=True)
    assert single.base_points == 8000
    assert double.base_points == 8000
    assert double.yakuman == ["daisangen"]
