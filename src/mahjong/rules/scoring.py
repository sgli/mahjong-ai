"""Fu (符) + han (番) -> points (点数) — pure functions, Tenhou rules.

Scoring flow:

1. Dora (table + ura + aka) is counted (see :mod:`dora`) and added to han;
   it does **not** affect fu and is not itself a yaku.
2. Fu is computed from the hand interpretation (base 20 / menzen ron +10 /
   tsumo +2 except pinfu / set fu / pair fu / wait fu), rounded up to the next
   10 (except pinfu 20/30 and chiitoi 25).
3. Base points = ``fu * 2 ** (2 + han)`` with mangan caps:
   5+ han or (4 han, 40+ fu) or (3 han, 70+ fu) -> 2000 (mangan);
   6-7 han -> 3000 (haneman); 8-10 -> 4000 (baiman); 11-12 -> 6000 (sanbaiman);
   13+ -> 8000 (kazoe yakuman).  Each yakuman = 8000 base.

Assumptions (Tenhou):

- Kiriage mangan (切上満貫) is **not** used.
- Double-yakuman variants are treated as single yakuman (see yaku.py notes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .agari import is_chiitoi, is_kokushi, is_standard_agari
from .decompose import iter_structures
from .dora import count_dora
from .tiles import normalize, rank_of, suit_of
from .yaku import (
    WinContext,
    YakuResult,
    _GREEN,
    _is_honor,
    _is_middle,
    _is_pinfu,
    _is_terminal,
    _is_terminal_or_honor,
    _is_yakuhai,
    _wait_type,
    _wait_types,
    compute_yaku,
)


@dataclass
class ScoreResult:
    yaku: list[tuple[str, int]] = field(default_factory=list)
    yakuman: list[str] = field(default_factory=list)
    han: int = 0  # total han (yaku han + dora)
    dora: int = 0
    fu: int = 0
    base_points: int = 0
    is_yakuman: bool = False

    @property
    def yakuman_count(self) -> int:
        return len(self.yakuman)

    @property
    def has_yaku(self) -> bool:
        return bool(self.yaku) or bool(self.yakuman)


# -- fu ------------------------------------------------------------------------
def _compute_fu(structure, ctx: WinContext, wait_type: str | None = None) -> int:
    is_pinfu = _is_pinfu(structure, ctx, wait_type)
    fu = 20
    if ctx.is_menzen and not ctx.is_tsumo:
        fu += 10  # menzen ron
    if ctx.is_tsumo and not is_pinfu:
        fu += 2  # tsumo (pinfu tsumo is fixed 20)

    for g in structure.sets:
        if g.kind == "run":
            continue
        terminal = _is_terminal_or_honor(g.tiles[0])
        # 荣和完成双碰的刻子按明刻计符（和牌张来自他家）
        ron_completed = not ctx.is_tsumo and normalize(ctx.win_tile) in g.tiles
        if g.kind == "triplet":
            if g.open or ron_completed:
                fu += 4 if terminal else 2
            else:
                fu += 8 if terminal else 4
        elif g.kind == "quad":
            if g.open:
                fu += 16 if terminal else 8
            else:
                fu += 32 if terminal else 16

    if _is_yakuhai(structure.pair, ctx):
        fu += 2
        if structure.pair == ctx.seat_wind == ctx.round_wind:
            fu += 2  # 连风雀头（座风==场风）计 +4（用户纠正：天凤连风 +4）

    if not is_pinfu:
        wait = wait_type if wait_type is not None else _wait_type(structure, ctx.win_tile)
        if wait in ("kanchan", "penchan", "tanki"):
            fu += 2

    if is_pinfu:
        return 20 if ctx.is_tsumo else 30
    return max(((fu + 9) // 10) * 10, 30)  # round up to next 10; non-pinfu floor 30


# -- chiitoi (seven pairs) -----------------------------------------------------
def _chiitoi_yaku(hand, ctx: WinContext):
    tiles = [normalize(t) for t in hand]
    yaku = [("chiitoi", 2)]
    yakuman: list[str] = []

    if ctx.is_double_riichi:
        yaku.append(("double_riichi", 2))
    elif ctx.is_riichi:
        yaku.append(("riichi", 1))
    if ctx.is_ippatsu:
        yaku.append(("ippatsu", 1))
    if ctx.is_tsumo:
        yaku.append(("menzen_tsumo", 1))
    if ctx.is_rinshan:
        yaku.append(("rinshan", 1))
    if ctx.is_chankan:
        yaku.append(("chankan", 1))
    if ctx.is_haitei:
        yaku.append(("haitei", 1))
    if ctx.is_houtei:
        yaku.append(("houtei", 1))
    if all(_is_middle(t) for t in tiles):
        yaku.append(("tanyao", 1))

    suits = {suit_of(t) for t in tiles if suit_of(t) is not None}
    has_honors = any(_is_honor(t) for t in tiles)
    if len(suits) == 1 and has_honors:
        yaku.append(("honitsu", 3))
    if len(suits) == 1 and not has_honors:
        yaku.append(("chinitsu", 6))
    if all(_is_terminal_or_honor(t) for t in tiles):
        yaku.append(("honroutou", 2))
    if all(_is_honor(t) for t in tiles):
        yakuman.append("tsuuiisou")
    if all(_is_terminal(t) for t in tiles):
        yakuman.append("chinroutou")
    if all(t in _GREEN for t in tiles):
        yakuman.append("ryuuiisou")

    return yaku, yakuman


# -- base points ---------------------------------------------------------------
def base_points_from(han: int, fu: int, yakuman_count: int = 0, *, kiriage_mangan: bool = False) -> int:
    if yakuman_count:
        return 8000 * yakuman_count
    if han >= 13:
        return 8000  # kazoe yakuman
    if han >= 11:
        return 6000  # sanbaiman
    if han >= 8:
        return 4000  # baiman
    if han >= 6:
        return 3000  # haneman
    if han >= 5:
        return 2000  # mangan
    if han == 4 and fu >= 40:
        return 2000
    if han == 3 and fu >= 70:
        return 2000
    if kiriage_mangan and ((han == 4 and fu == 30) or (han == 3 and fu == 60)):
        return 2000  # 切上满贯：4番30符 / 3番60符 按满贯计
    return fu * (2 ** (2 + han))


def _round100(points: int) -> int:
    return ((points + 99) // 100) * 100


def _kokushi_13sided(hand, win_tile: str) -> bool:
    """国士無双十三面待ち：去掉和牌张后仍为 13 种不同的幺九牌。"""
    from ..rules.tiles import normalize

    pre = [normalize(t) for t in hand]
    w = normalize(win_tile)
    for i, t in enumerate(pre):
        if t == w:
            pre.pop(i)
            break
    return len(set(pre)) == 13


def ron_payment(base_points: int, is_dealer: bool) -> int:
    """Amount the discarder pays on a ron (rounded up to 100)."""
    return _round100(base_points * (6 if is_dealer else 4))


def tsumo_payments(base_points: int, is_dealer: bool) -> tuple[int, int, int]:
    """Payments on a tsumo win, as (dealer_pays, non_dealer_pays, non_dealer_pays)."""
    if is_dealer:
        each = _round100(base_points * 2)
        return (each, each, each)
    dealer = _round100(base_points * 2)
    other = _round100(base_points)
    return (dealer, other, other)


# -- main ----------------------------------------------------------------------
def score_hand(
    hand,
    melds,
    win_tile: str,
    context: WinContext,
    dora_indicators=(),
    ura_indicators=(),
    kiriage_mangan: bool = False,
    double_yakuman: bool = False,
) -> ScoreResult:
    """Score a winning hand.  ``hand`` is the concealed hand including the win tile.

    A hand that is valid under several interpretations (e.g. seven pairs and
    ryanpeiko) is scored under the highest one.
    """
    # 里宝牌只在立直（含双立直）时有效
    effective_ura = ura_indicators if (context.is_riichi or context.is_double_riichi) else ()
    dora = count_dora(hand, melds, dora_indicators, effective_ura)
    candidates: list[ScoreResult] = []

    # kokushi (thirteen orphans) — yakuman
    if len(melds) == 0 and is_kokushi(hand):
        yakuman = ["kokushi_musou"]
        if double_yakuman and _kokushi_13sided(hand, win_tile):
            yakuman = ["kokushi_musou", "kokushi_musou"]  # 国士無双十三面：双倍
        if context.is_tenhou:
            yakuman.append("tenhou")
        if context.is_chiihou:
            yakuman.append("chiihou")
        candidates.append(
            ScoreResult(
                yaku=[], yakuman=yakuman, han=13 + dora, dora=dora, fu=0,
                base_points=8000 * len(yakuman), is_yakuman=True,
            )
        )

    # chiitoi (seven pairs) — 25 fu, own yaku path
    if len(melds) == 0 and is_chiitoi(hand):
        yaku, yakuman = _chiitoi_yaku(hand, context)
        if yakuman:
            candidates.append(
                ScoreResult(
                    yaku=yaku, yakuman=yakuman, han=13 + dora, dora=dora, fu=25,
                    base_points=8000 * len(yakuman), is_yakuman=True,
                )
            )
        else:
            han = sum(h for _, h in yaku) + dora
            candidates.append(
                ScoreResult(
                    yaku=yaku, yakuman=[], han=han, dora=dora, fu=25,
                    base_points=base_points_from(han, 25, kiriage_mangan=kiriage_mangan), is_yakuman=False,
                )
            )

    # standard form
    if is_standard_agari(hand, melds):
        for structure in iter_structures(hand, melds):
            # 高点法：番数与符数来自同一分解、同一和牌张归属；枚举每种听牌形解释
            for wt in _wait_types(structure, win_tile):
                yr = compute_yaku(structure, context, wait_type=wt, double_yakuman=double_yakuman)
                fu = _compute_fu(structure, context, wait_type=wt)
                if yr.yakuman:
                    candidates.append(
                        ScoreResult(
                            yaku=yr.yaku, yakuman=yr.yakuman, han=13 + dora, dora=dora, fu=fu,
                            base_points=8000 * len(yr.yakuman), is_yakuman=True,
                        )
                    )
                else:
                    han = yr.han + dora
                    candidates.append(
                        ScoreResult(
                            yaku=yr.yaku, yakuman=[], han=han, dora=dora, fu=fu,
                            base_points=base_points_from(han, fu, kiriage_mangan=kiriage_mangan), is_yakuman=False,
                        )
                    )

    assert candidates, "hand is not a winning hand"
    # 高点法：先比基本点，同点取高符（如 3番40符 与 4番20符 同点时取 3番40符）
    return max(
        candidates,
        key=lambda r: (int(r.is_yakuman), r.yakuman_count, r.base_points, r.fu),
    )


__all__ = [
    "ScoreResult",
    "base_points_from",
    "ron_payment",
    "score_hand",
    "tsumo_payments",
]
