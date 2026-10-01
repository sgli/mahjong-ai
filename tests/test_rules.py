"""Tests for the unified rule core (agari / shanten / tenpai / furiten / calls)."""

from mahjong.replay.state import Meld
from mahjong.rules import (
    ankan_actions,
    can_riichi,
    chi_actions,
    daiminkan_actions,
    discard_legal_actions,
    is_agari,
    is_furiten,
    is_tenpai,
    kakan_actions,
    pon_actions,
    response_legal_actions,
    shanten,
    tenpai_tiles,
)
from mahjong.decision.action import Action, ActionType


def ts(s):
    return s.split()


# -- agari ---------------------------------------------------------------------
def test_standard_agari():
    assert is_agari(ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p 5p"))


def test_agari_with_open_pon():
    melds = (Meld(kind="pon", tiles=("E", "E", "E"), from_=0, called="E"),)
    assert is_agari(ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 5m 5m"), melds)


def test_chiitoi_agari():
    assert is_agari(ts("1m 1m 2m 2m 3m 3m 4m 4m 5m 5m 6m 6m 7m 7m"))


def test_kokushi_agari():
    assert is_agari(ts("1m 9m 1p 9p 1s 9s E S W N P F C 1m"))


def test_not_agari_13_tiles():
    assert not is_agari(ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p"))


def test_not_agari_14_tiles_no_pair():
    assert not is_agari(ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p 7p"))


def test_agari_red_five_normalized():
    # red five is a normal five for the hand shape
    assert is_agari(ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5pr 5p"))


# -- shanten -------------------------------------------------------------------
def test_shanten_agari_is_minus_one():
    assert shanten(ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p 5p")) == -1


def test_shanten_tenpai_pair_wait():
    assert shanten(ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p")) == 0


def test_shanten_tenpai_set_wait():
    assert shanten(ts("1m 1m 1m 2m 2m 2m 3m 3m 3m 4m 4m 5m 6m")) == 0


def test_shanten_one():
    assert shanten(ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 4p 5p")) == 1


def test_shanten_two():
    assert shanten(ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 5p 9p 1s")) == 2


def test_shanten_chiitoi_14_tile_six_pairs_two_singles():
    # 6 pairs + 2 singles (after a draw) -> chiitoi shanten 0
    assert shanten(ts("2m 2m 5m 5s 5s 6m 6m 6p 6p 7p 7p 8m E E")) == 0


def test_shanten_chiitoi_tenpai():
    assert shanten(ts("1m 1m 2m 2m 3m 3m 4m 4m 5m 5m 6m 6m 7m")) == 0


def test_shanten_kokushi_tenpai():
    assert shanten(ts("1m 9m 1p 9p 1s 9s E S W N P F C")) == 0


# -- tenpai / waits ------------------------------------------------------------
def test_tenpai_tiles_pair_wait():
    assert tenpai_tiles(ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p")) == ["5p"]


def test_tenpai_tiles_multi_wait():
    # 3 runs + 555p triplet + 6p single -> waits 4p (456 run + 55 pair),
    # 6p (555 + 66 pair), 7p (567 run + 55 pair)
    waits = set(tenpai_tiles(ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 5p 5p 5p 6p")))
    assert waits == {"4p", "6p", "7p"}


def test_is_tenpai():
    assert is_tenpai(ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p"))


# -- furiten -------------------------------------------------------------------
def test_furiten_discard():
    hand = ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p")
    assert is_furiten(hand, own_discards=["5p"])
    assert not is_furiten(hand, own_discards=["1m"])


def test_furiten_not_tenpai():
    assert not is_furiten(ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p"), own_discards=["5p"])


def test_furiten_red_five_matches_base_five():
    hand = ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p")
    assert is_furiten(hand, own_discards=["5pr"])  # red five counts as the same tile


# -- chi / pon / kan feasibility ------------------------------------------------
def test_chi_red_five():
    actions = chi_actions(["5mr", "6m", "1m", "2m"], "7m", target=0)
    assert any(a.consumed == ("5mr", "6m") for a in actions)


def test_chi_middle_and_low_patterns():
    actions = chi_actions(["1p", "2p", "3p", "4p"], "3p", target=2)
    consumed = {a.consumed for a in actions}
    assert ("1p", "2p") in consumed  # 123
    assert ("2p", "4p") in consumed  # 234
    assert ("4p", "5p") not in consumed  # 345 needs a 5p in hand


def test_chi_honor_is_impossible():
    assert chi_actions(["E", "S", "W"], "N", target=0) == []


def test_pon_two_copies():
    actions = pon_actions(["C", "C", "1m"], "C", target=1)
    assert actions == [Action(ActionType.PON, tile="C", consumed=("C", "C"), target=1)]


def test_pon_with_red_and_normal_five():
    actions = pon_actions(["5m", "5mr", "1m"], "5m", target=1)
    assert any(a.consumed == ("5m", "5mr") for a in actions)


def test_daiminkan_needs_three():
    assert daiminkan_actions(["1s", "1s", "1s"], "1s", target=3) == [
        Action(ActionType.KAN, tile="1s", consumed=("1s", "1s", "1s"), target=3, kan_kind="daiminkan")
    ]
    assert daiminkan_actions(["1s", "1s"], "1s", target=3) == []


def test_ankan_needs_four():
    assert ankan_actions(["E", "E", "E", "E"]) == [
        Action(ActionType.KAN, consumed=("E", "E", "E", "E"), kan_kind="ankan")
    ]
    assert ankan_actions(["E", "E", "E"]) == []


def test_kakan_upgrades_pon():
    melds = (Meld(kind="pon", tiles=("P", "P", "P"), from_=0, called="P"),)
    actions = kakan_actions(["P", "1m"], melds)
    assert any(a.kan_kind == "kakan" and a.tile == "P" for a in actions)


def test_riichi_legal_when_hand_is_agari_before_discard():
    # The drawn hand is a complete win (123m 789m 456s EEE 55m), but discarding
    # 9m leaves a tenpai hand, so RIICHI(9m) is a legal (decline-the-win) move.
    hand = ts("1m 2m 3m 5m 5m 7m 8m 9m 4s 5s 6s E E E")
    legal = discard_legal_actions(hand=hand, melds=(), riichi=False, score=25000, drawn_tile="3m")
    assert Action(ActionType.RIICHI, tile="9m") in legal
    assert Action(ActionType.TSUMO) in legal


def test_riichi_allowed_with_closed_ankan():
    ankan = (Meld(kind="ankan", tiles=("E", "E", "E", "E")),)
    pon = (Meld(kind="pon", tiles=("E", "E", "E"), from_=0, called="E"),)
    assert can_riichi(hand=[], melds=ankan, score=25000)
    assert not can_riichi(hand=[], melds=pon, score=25000)


# -- assembled legal actions ----------------------------------------------------
def test_discard_legal_actions_include_riichi_and_tsumo():
    # closed hand, tenpai after discarding one tile -> riichi is legal
    hand = ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p 5p")  # 14 tiles, already agari
    legal = discard_legal_actions(hand=hand, melds=(), riichi=False, score=25000, drawn_tile="5p")
    types = {a.type for a in legal}
    assert ActionType.TSUMO in types  # hand is winning on the draw


def test_discard_legal_actions_riichi_requires_closed():
    hand = ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p")
    melds = (Meld(kind="pon", tiles=("E", "E", "E"), from_=0, called="E"),)
    legal = discard_legal_actions(hand=hand, melds=melds, riichi=False, score=25000, drawn_tile="5p")
    assert all(a.type is not ActionType.RIICHI for a in legal)


def test_response_legal_actions_ron_pon_pass():
    hand = ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p")  # tenpai wait 5p
    legal = response_legal_actions(
        hand=hand, melds=(), own_discards=["1s"], riichi=False,
        claimed_tile="5p", discarder=0, is_next=True,
    )
    types = {a.type for a in legal}
    assert ActionType.RON in types
    assert ActionType.PASS in types


def test_response_legal_actions_furiten_blocks_ron():
    hand = ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p")  # wait 5p
    legal = response_legal_actions(
        hand=hand, melds=(), own_discards=["5p"], riichi=False,
        claimed_tile="5p", discarder=0, is_next=True,
    )
    assert all(a.type is not ActionType.RON for a in legal)


def test_response_legal_actions_riichi_only_ron_pass():
    hand = ts("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p")
    legal = response_legal_actions(
        hand=hand, melds=(), own_discards=["1s"], riichi=True,
        claimed_tile="5p", discarder=0, is_next=True,
    )
    types = {a.type for a in legal}
    assert types <= {ActionType.RON, ActionType.PASS}
