"""Yaku (役) detection — pure functions.

Input is a winning hand interpretation (:class:`Structure`) plus a
:class:`WinContext` carrying situational facts (riichi / ippatsu / tsumo /
menzen / win tile / rinshan / chankan / haitei / houtei / winds / tenhou /
chiihou).

Regular yaku are returned as ``(name, han)``; yakuman as a list of names (each
counts as one yakuman).  Dora is counted separately (see :mod:`dora`).

Notes / assumptions (Tenhou):

- Double riichi (2 han) replaces riichi (1 han).
- ``honroutou`` (混老头) is 2 han flat; it does not stack with ``chanta``.
- ``junchan`` (纯全) does not stack with ``chanta`` or ``chinroutou``.
- Pair fu: yakuhai pair = 2 fu; double-wind 連風 (seat wind == round wind) = 4 fu.
- Double-yakuman variants (大四喜 / 国士13面 / 四暗刻単騎 / 純正九蓮) are treated
  as a single yakuman here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tiles import normalize, rank_of, suit_of

_DRAGONS = ("P", "F", "C")
_WINDS = ("E", "S", "W", "N")
_GREEN = {"2s", "3s", "4s", "6s", "8s", "F"}


@dataclass(frozen=True)
class WinContext:
    round_wind: str  # "E"/"S"/"W"
    seat_wind: str  # "E"/"S"/"W"/"N"
    win_tile: str  # the tile completing the hand (may be red five)
    is_tsumo: bool = False
    is_menzen: bool = True
    is_riichi: bool = False
    is_double_riichi: bool = False
    is_ippatsu: bool = False
    is_rinshan: bool = False
    is_chankan: bool = False
    is_haitei: bool = False
    is_houtei: bool = False
    is_tenhou: bool = False
    is_chiihou: bool = False


@dataclass
class YakuResult:
    yaku: list[tuple[str, int]] = field(default_factory=list)
    yakuman: list[str] = field(default_factory=list)

    @property
    def han(self) -> int:
        return sum(h for _, h in self.yaku)

    @property
    def has_yaku(self) -> bool:
        return bool(self.yaku) or bool(self.yakuman)


# -- helpers -------------------------------------------------------------------
def _all_tiles(structure) -> list[str]:
    tiles = []
    for g in structure.sets:
        tiles.extend(g.tiles)
    tiles.append(structure.pair)
    tiles.append(structure.pair)
    return tiles


def _is_terminal(tile: str) -> bool:
    return rank_of(tile) in (1, 9)


def _is_honor(tile: str) -> bool:
    return suit_of(tile) is None


def _is_terminal_or_honor(tile: str) -> bool:
    return _is_terminal(tile) or _is_honor(tile)


def _is_middle(tile: str) -> bool:
    r = rank_of(tile)
    return r is not None and 2 <= r <= 8


def _is_yakuhai(tile: str, ctx: WinContext) -> bool:
    return tile in _DRAGONS or tile == ctx.seat_wind or tile == ctx.round_wind


def _wait_type(structure, win_tile: str) -> str:
    """tanki / shanpon / kanchan / penchan / ryanmen for a structure + win tile."""
    w = normalize(win_tile)
    if structure.pair == w:
        return "tanki"
    for g in structure.sets:
        if g.open:
            continue  # 副露组不参与听牌形判定（和牌张只落在手内组）
        if g.kind == "run" and w in g.tiles:
            a, b, c = sorted(g.tiles)
            if w == b:
                return "kanchan"
            if w == a:
                # waiting on the lowest tile: penchan only for 7-8-9 wait 7
                return "penchan" if a in ("7m", "7p", "7s") else "ryanmen"
            # waiting on the highest tile: penchan only for 1-2-3 wait 3
            return "penchan" if c in ("3m", "3p", "3s") else "ryanmen"
        if g.kind in ("triplet", "quad") and w in g.tiles:
            return "shanpon"
    return "ryanmen"


def _wait_types(structure, win_tile: str) -> list[str]:
    """All wait interpretations for a structure + win tile (高点法候选).

    When the winning tile fits several sets (e.g. 456m+567m with win 6m), each
    assignment is a distinct wait type that affects fu and pinfu.
    """
    w = normalize(win_tile)
    types: list[str] = []
    if structure.pair == w:
        types.append("tanki")
    for g in structure.sets:
        if g.open:
            continue  # 副露组不参与听牌形判定（和牌张只落在手内组）
        if g.kind == "run" and w in g.tiles:
            a, b, c = sorted(g.tiles)
            if w == b:
                types.append("kanchan")
            elif w == a:
                types.append("penchan" if a in ("7m", "7p", "7s") else "ryanmen")
            else:
                types.append("penchan" if c in ("3m", "3p", "3s") else "ryanmen")
        elif g.kind in ("triplet", "quad") and w in g.tiles:
            types.append("shanpon")
    # dedupe preserving order
    seen: list[str] = []
    for t in types:
        if t not in seen:
            seen.append(t)
    return seen or ["ryanmen"]


def _is_pinfu(structure, ctx: WinContext, wait_type: str | None = None) -> bool:
    if not ctx.is_menzen:
        return False
    if any(g.kind != "run" for g in structure.sets):
        return False
    if _is_yakuhai(structure.pair, ctx):
        return False
    wt = wait_type if wait_type is not None else _wait_type(structure, ctx.win_tile)
    return wt == "ryanmen"


def _chuuren_ok(tiles) -> bool:
    suits = {suit_of(t) for t in tiles if suit_of(t) is not None}
    if len(suits) != 1 or any(_is_honor(t) for t in tiles):
        return False
    suit = next(iter(suits))
    ranks = [0] * 10
    for t in tiles:
        if suit_of(t) == suit:
            ranks[rank_of(t)] += 1
    return ranks[1] >= 3 and ranks[9] >= 3 and all(ranks[r] >= 1 for r in range(2, 9))


# -- yaku detection ------------------------------------------------------------
def compute_yaku(structure, ctx: WinContext, wait_type: str | None = None) -> YakuResult:
    """Detect yaku for one interpretation of a winning hand."""
    tiles = _all_tiles(structure)
    result = YakuResult()
    has_honors = any(_is_honor(t) for t in tiles)
    all_terminal_or_honor = all(_is_terminal_or_honor(t) for t in tiles)
    all_terminal = all(_is_terminal(t) for t in tiles)
    all_honor = all(_is_honor(t) for t in tiles)

    triplets = [g.tiles[0] for g in structure.sets if g.kind in ("triplet", "quad")]
    # 暗刻 = 三张均出自自己手牌；荣和完成双碰时含和牌张的那组刻子不算暗刻
    closed_triplets = [
        g.tiles[0]
        for g in structure.sets
        if g.kind in ("triplet", "quad")
        and not g.open
        and (ctx.is_tsumo or normalize(ctx.win_tile) not in g.tiles)
    ]
    quads = [g for g in structure.sets if g.kind == "quad"]
    run_sigs = sorted((suit_of(g.tiles[0]), rank_of(g.tiles[0])) for g in structure.sets if g.kind == "run")
    concealed_runs = sorted(
        (suit_of(g.tiles[0]), rank_of(g.tiles[0])) for g in structure.sets if g.kind == "run" and not g.open
    )

    # ---- situational ---------------------------------------------------------
    if ctx.is_double_riichi:
        result.yaku.append(("double_riichi", 2))
    elif ctx.is_riichi:
        result.yaku.append(("riichi", 1))
    if ctx.is_ippatsu:
        result.yaku.append(("ippatsu", 1))
    if ctx.is_tsumo and ctx.is_menzen:
        result.yaku.append(("menzen_tsumo", 1))
    if ctx.is_rinshan:
        result.yaku.append(("rinshan", 1))
    if ctx.is_chankan:
        result.yaku.append(("chankan", 1))
    if ctx.is_haitei:
        result.yaku.append(("haitei", 1))
    if ctx.is_houtei:
        result.yaku.append(("houtei", 1))
    if ctx.is_tenhou:
        result.yakuman.append("tenhou")
    if ctx.is_chiihou:
        result.yakuman.append("chiihou")

    # ---- one han -------------------------------------------------------------
    if all(_is_middle(t) for t in tiles):
        result.yaku.append(("tanyao", 1))
    if _is_pinfu(structure, ctx, wait_type):
        result.yaku.append(("pinfu", 1))
    for t in triplets:
        if t == ctx.seat_wind:
            result.yaku.append(("seat_wind", 1))
        if t == ctx.round_wind:
            result.yaku.append(("round_wind", 1))
        if t in _DRAGONS:
            result.yaku.append(("dragon", 1))
    if ctx.is_menzen:
        dup = {s for s in concealed_runs if concealed_runs.count(s) >= 2}
        if len(dup) >= 2:
            result.yaku.append(("ryanpeiko", 3))
        elif dup:
            result.yaku.append(("iipeiko", 1))

    # ---- two han -------------------------------------------------------------
    for r in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        if all((s, r) in run_sigs for s in ("m", "p", "s")):
            result.yaku.append(("sanshoku_doujun", 2 if ctx.is_menzen else 1))
            break
    tri_by_suit = {}
    for t in triplets:
        tri_by_suit.setdefault(suit_of(t), []).append(rank_of(t))
    for r in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        if all(r in tri_by_suit.get(s, ()) for s in ("m", "p", "s")):
            result.yaku.append(("sanshoku_doukou", 2))
            break
    if len(closed_triplets) >= 4:
        result.yakuman.append("suuankou")
    elif len(closed_triplets) >= 3:
        result.yaku.append(("sanankou", 2))
    if len(quads) >= 4:
        result.yakuman.append("suukantsu")
    elif len(quads) >= 3:
        result.yaku.append(("sankantsu", 2))
    if all(g.kind in ("triplet", "quad") for g in structure.sets):
        result.yaku.append(("toitoi", 2))
    for s in ("m", "p", "s"):
        if all((s, r) in run_sigs for r in (1, 4, 7)):
            result.yaku.append(("ittsuu", 2 if ctx.is_menzen else 1))
            break
    dragon_triplets = [t for t in triplets if t in _DRAGONS]
    if len(dragon_triplets) >= 3:
        result.yakuman.append("daisangen")
    elif len(dragon_triplets) == 2 and structure.pair in _DRAGONS:
        result.yaku.append(("shousangen", 2))
    if all_terminal_or_honor:
        result.yaku.append(("honroutou", 2))

    # chanta / junchan (mutually exclusive; also exclude honroutou/chinroutou)
    junchan_ok = (
        not has_honors
        and not all_terminal
        and all(any(_is_terminal(t) for t in g.tiles) for g in structure.sets)
        and _is_terminal(structure.pair)
    )
    chanta_ok = (
        not all_terminal_or_honor
        and all(any(_is_terminal_or_honor(t) for t in g.tiles) for g in structure.sets)
        and _is_terminal_or_honor(structure.pair)
    )
    if junchan_ok:
        result.yaku.append(("junchan", 3 if ctx.is_menzen else 2))
    elif chanta_ok:
        result.yaku.append(("chanta", 2 if ctx.is_menzen else 1))

    # ---- three han -----------------------------------------------------------
    if len({suit_of(t) for t in tiles if suit_of(t) is not None}) == 1 and has_honors:
        result.yaku.append(("honitsu", 3 if ctx.is_menzen else 2))

    # ---- six han -------------------------------------------------------------
    if len({suit_of(t) for t in tiles if suit_of(t) is not None}) == 1 and not has_honors:
        result.yaku.append(("chinitsu", 6 if ctx.is_menzen else 5))

    # ---- shape yakuman -------------------------------------------------------
    if all_honor:
        result.yakuman.append("tsuuiisou")
    if all(t in _GREEN for t in tiles):
        result.yakuman.append("ryuuiisou")
    if all_terminal:
        result.yakuman.append("chinroutou")
    wind_triplets = [t for t in triplets if t in _WINDS]
    if len(wind_triplets) >= 4:
        result.yakuman.append("daisuushii")
    elif len(wind_triplets) == 3 and structure.pair in _WINDS:
        result.yakuman.append("shousuushii")
    if ctx.is_menzen and _chuuren_ok(tiles):
        result.yakuman.append("chuuren_poutou")

    return result


__all__ = ["WinContext", "YakuResult", "compute_yaku"]
