"""Tests for yaku detection (truth + han)."""

from mahjong.replay.state import Meld
from mahjong.rules import WinContext, score_hand


def ts(s):
    return s.split()


def sc(hand_s, win, *, tsumo=False, menzen=True, rw="E", sw="W", melds=(), **kw):
    ctx = WinContext(round_wind=rw, seat_wind=sw, win_tile=win, is_tsumo=tsumo, is_menzen=menzen, **kw)
    return score_hand(ts(hand_s), melds, win, ctx)


def names(r):
    return [n for n, _ in r.yaku]


def han_of(r, name):
    for n, h in r.yaku:
        if n == name:
            return h
    return None


# -- one han -------------------------------------------------------------------
def test_riichi_and_double_riichi():
    r = sc("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p 5p", "5p", is_riichi=True)
    assert han_of(r, "riichi") == 1

    r2 = sc("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p 5p", "5p", is_double_riichi=True)
    assert han_of(r2, "double_riichi") == 2
    assert "riichi" not in names(r2)


def test_ippatsu_and_menzen_tsumo():
    r = sc("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p 5p", "5p", is_riichi=True, is_ippatsu=True)
    assert han_of(r, "ippatsu") == 1

    r2 = sc("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p 5p", "5p", tsumo=True)
    assert han_of(r2, "menzen_tsumo") == 1


def test_tanyao():
    r = sc("2m 3m 4m 4m 5m 6m 6m 7m 8m 2p 3p 4p 5p 5p", "5p")
    assert han_of(r, "tanyao") == 1


def test_pinfu():
    r = sc("2m 3m 4m 5m 6m 7m 2p 3p 4p 6p 7p 8p 1s 1s", "6p")
    assert han_of(r, "pinfu") == 1


def test_yakuhai_winds_and_dragons():
    r = sc("1m 2m 3m 4m 5m 6m 7m 8m 9m W W W 5p 5p", "W")  # seat wind W
    assert han_of(r, "seat_wind") == 1

    r2 = sc("1m 2m 3m 4m 5m 6m 7m 8m 9m E E E 5p 5p", "E")  # round wind E
    assert han_of(r2, "round_wind") == 1

    r3 = sc("1m 2m 3m 4m 5m 6m 7m 8m 9m P P P 5p 5p", "P")
    assert han_of(r3, "dragon") == 1


def test_iipeiko():
    r = sc("1m 2m 3m 1m 2m 3m 4m 5m 6m 7s 8s 9s 5m 5m", "5m")
    assert han_of(r, "iipeiko") == 1


def test_rinshan_chankan_haitei_houtei():
    hand = "1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p 5p"
    assert han_of(sc(hand, "5p", is_rinshan=True), "rinshan") == 1
    assert han_of(sc(hand, "5p", is_chankan=True), "chankan") == 1
    assert han_of(sc(hand, "5p", is_haitei=True, tsumo=True), "haitei") == 1
    assert han_of(sc(hand, "5p", is_houtei=True), "houtei") == 1


# -- two han -------------------------------------------------------------------
def test_sanshoku_doukou():
    r = sc("2m 2m 2m 2p 2p 2p 2s 2s 2s 7m 8m 9m 5m 5m", "5m")
    assert han_of(r, "sanshoku_doukou") == 2


def test_sanankou():
    r = sc("2m 2m 2m 3m 3m 3m 4m 4m 4m 5p 6p 7p 5s 5s", "5s")
    assert han_of(r, "sanankou") == 2


def test_sankantsu():
    melds = (
        Meld(kind="ankan", tiles=("2m", "2m", "2m", "2m")),
        Meld(kind="daiminkan", tiles=("3m", "3m", "3m", "3m"), from_=0, called="3m"),
        Meld(kind="kakan", tiles=("4m", "4m", "4m", "4m"), from_=0, called="4m"),
    )
    r = sc("5m 5m 5m 5s 5s", "5s", melds=melds)
    assert han_of(r, "sankantsu") == 2


def test_toitoi():
    melds = (Meld(kind="pon", tiles=("E", "E", "E"), from_=0, called="E"),)
    r = sc("2m 2m 2m 3m 3m 3m 4m 4m 4m 5s 5s", "5s", melds=melds)
    assert han_of(r, "toitoi") == 2


def test_chiitoi():
    r = sc("1m 1m 2m 2m 3p 3p 4p 4p 5s 5s 6s 6s 7s 7s", "7s")
    assert han_of(r, "chiitoi") == 2


def test_chanta():
    r = sc("1m 2m 3m 7p 8p 9p 7s 8s 9s S S S P P", "P")
    assert han_of(r, "chanta") == 2


def test_ittsuu():
    r = sc("1m 2m 3m 4m 5m 6m 7m 8m 9m 2p 3p 4p 5s 5s", "5s")
    assert han_of(r, "ittsuu") == 2


def test_shousangen():
    r = sc("P P P F F F 1m 2m 3m 4m 5m 6m C C", "C")
    assert han_of(r, "shousangen") == 2


def test_honroutou():
    melds = (Meld(kind="pon", tiles=("E", "E", "E"), from_=0, called="E"),)
    r = sc("1m 1m 1m 9m 9m 9m 9p 9p 9p 1s 1s", "1s", melds=melds)
    assert han_of(r, "honroutou") == 2


def test_sanshoku_doujun():
    r = sc("1m 2m 3m 1p 2p 3p 1s 2s 3s 7m 8m 9m 5s 5s", "5s")
    assert han_of(r, "sanshoku_doujun") == 2


# -- three / six han -----------------------------------------------------------
def test_honitsu():
    r = sc("1m 2m 3m 4m 5m 6m 7m 8m 9m 2m 3m 4m P P", "P")
    assert han_of(r, "honitsu") == 3


def test_junchan():
    r = sc("1m 2m 3m 7p 8p 9p 1s 2s 3s 7s 8s 9s 1m 1m", "1m")
    assert han_of(r, "junchan") == 3


def test_ryanpeiko():
    r = sc("1m 2m 3m 1m 2m 3m 4p 5p 6p 4p 5p 6p 5s 5s", "5s")
    assert han_of(r, "ryanpeiko") == 3


def test_chinitsu():
    r = sc("1m 2m 3m 4m 5m 6m 7m 8m 9m 2m 3m 4m 5m 5m", "5m")
    assert han_of(r, "chinitsu") == 6


# -- yakuman -------------------------------------------------------------------
def test_kokushi():
    r = sc("1m 9m 1p 9p 1s 9s E S W N P F C 1m", "1m")
    assert "kokushi_musou" in r.yakuman


def test_suuankou():
    r = sc("1m 1m 1m 2m 2m 2m 3m 3m 3m 4m 4m 4m 5s 5s", "5s")
    assert "suuankou" in r.yakuman


def test_daisangen():
    r = sc("P P P F F F C C C 1m 2m 3m 5s 5s", "5s")
    assert "daisangen" in r.yakuman


def test_tsuuiisou():
    r = sc("E E E S S S P P P F F F C C", "C")
    assert "tsuuiisou" in r.yakuman


def test_shousuushii_and_daisuushii():
    r = sc("E E E S S S W W W 1m 2m 3m N N", "N")
    assert "shousuushii" in r.yakuman

    r2 = sc("E E E S S S W W W N N N 5s 5s", "5s")
    assert "daisuushii" in r2.yakuman


def test_chinroutou():
    r = sc("1m 1m 1m 9m 9m 9m 1p 1p 1p 9p 9p 9p 1s 1s", "1s")
    assert "chinroutou" in r.yakuman


def test_ryuuiisou():
    r = sc("2s 2s 2s 3s 3s 3s 4s 4s 4s 6s 6s 6s 8s 8s", "8s")
    assert "ryuuiisou" in r.yakuman


def test_chuuren_poutou():
    r = sc("1m 1m 1m 2m 3m 4m 5m 6m 7m 8m 9m 9m 9m 1m", "1m")
    assert "chuuren_poutou" in r.yakuman


def test_tenhou_chiihou():
    hand = "1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p 5p"
    assert "tenhou" in sc(hand, "5p", is_tenhou=True).yakuman
    assert "chiihou" in sc(hand, "5p", is_chiihou=True).yakuman


def test_no_yaku_hand_not_win():
    # valid shape (agari) but no yaku -> must not be treated as a winning hand
    from mahjong.rules import is_agari

    hand = "2m 3m 4m 5m 6m 7m 2p 3p 4p 6p 7p 8p 9m 9m"
    assert is_agari(hand.split())
    r = sc(hand, "9m")
    assert not r.has_yaku
    assert r.yaku == [] and r.yakuman == []
