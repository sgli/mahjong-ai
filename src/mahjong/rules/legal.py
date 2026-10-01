"""Legal action generation for decision extraction.

All functions take only the decision player's visible information (own hand,
own melds, own river, riichi flag, score, and the claimed tile / discarder
seat).  They never inspect opponents' concealed hands or the wall.

Red fives are treated as the same value as their base five for call/kan
matching, but the *exact* tile string is preserved in the generated
``consumed``/``tile`` so the generated action matches the source event
(``5mr`` vs ``5m``).
"""

from __future__ import annotations

from itertools import combinations

from ..decision.action import Action, ActionType
from .agari import is_agari, is_furiten, is_tenpai, shanten
from .tiles import normalize, rank_of, suit_of


def _group_by_value(tiles) -> dict[str, list[str]]:
    """Group exact tile strings by their base value."""
    groups: dict[str, list[str]] = {}
    for tile in tiles:
        groups.setdefault(normalize(tile), []).append(tile)
    return groups


def _multisets(items: list[str], k: int) -> list[tuple[str, ...]]:
    """k-element subsets of the actual hand tiles (no reuse of a single tile)."""
    seen = set()
    out = []
    for combo in combinations(items, k):
        key = tuple(sorted(combo))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _remove_one(hand, tile: str) -> list[str]:
    out = list(hand)
    out.remove(tile)
    return out


# -- calls / kans --------------------------------------------------------------
def chi_actions(hand, claimed_tile: str, target: int) -> list[Action]:
    """CHI actions: enumerate the run(s) that can use the claimed tile."""
    rank = rank_of(claimed_tile)
    suit = suit_of(claimed_tile)
    if rank is None or suit is None:
        return []
    groups = _group_by_value(hand)
    patterns = []
    if rank >= 3:
        patterns.append((rank - 2, rank - 1))
    if 2 <= rank <= 8:
        patterns.append((rank - 1, rank + 1))
    if rank <= 7:
        patterns.append((rank + 1, rank + 2))

    actions = []
    for v1, v2 in patterns:
        opts1 = groups.get(f"{v1}{suit}", [])
        opts2 = groups.get(f"{v2}{suit}", [])
        for a in opts1:
            for b in opts2:
                actions.append(
                    Action(ActionType.CHI, tile=claimed_tile, consumed=tuple(sorted((a, b))), target=target)
                )
    return actions


def pon_actions(hand, claimed_tile: str, target: int) -> list[Action]:
    """PON actions (need 2 tiles of the claimed value in hand)."""
    base = normalize(claimed_tile)
    group = _group_by_value(hand).get(base, [])
    if len(group) < 2:
        return []
    return [
        Action(ActionType.PON, tile=claimed_tile, consumed=consumed, target=target)
        for consumed in _multisets(group, 2)
    ]


def daiminkan_actions(hand, claimed_tile: str, target: int) -> list[Action]:
    """DAIMINKAN actions (need 3 tiles of the claimed value in hand)."""
    base = normalize(claimed_tile)
    group = _group_by_value(hand).get(base, [])
    if len(group) < 3:
        return []
    return [
        Action(ActionType.KAN, tile=claimed_tile, consumed=consumed, target=target, kan_kind="daiminkan")
        for consumed in _multisets(group, 3)
    ]


def ankan_actions(hand) -> list[Action]:
    """ANKAN actions (any value held four times in hand)."""
    actions = []
    for base, group in _group_by_value(hand).items():
        if len(group) == 4:
            actions.append(Action(ActionType.KAN, consumed=tuple(sorted(group)), kan_kind="ankan"))
    return actions


def validate_ankan_consumed(consumed) -> bool:
    """Defensive check: ankan ``consumed`` must be 4 tiles of the same value."""
    if len(consumed) != 4:
        return False
    bases = {normalize(t) for t in consumed}
    return len(bases) == 1


def kakan_actions(hand, melds) -> list[Action]:
    """KAKAN actions: add a hand tile to an existing pon meld."""
    actions = []
    groups = _group_by_value(hand)
    for meld in melds:
        if meld.kind != "pon":
            continue
        base = normalize(meld.tiles[0])
        for extra in groups.get(base, []):
            actions.append(
                Action(
                    ActionType.KAN,
                    tile=extra,
                    consumed=tuple(sorted(meld.tiles)),
                    kan_kind="kakan",
                )
            )
    return actions


# -- riichi --------------------------------------------------------------------
def can_riichi(hand, melds, score: int) -> bool:
    """A riichi declaration is possible for a closed hand with enough points.

    An ankan (concealed kan) keeps the hand closed, so only open melds
    (chi / pon / daiminkan / kakan) block riichi.
    """
    open_melds = [m for m in melds if m.kind in ("chi", "pon", "daiminkan", "kakan")]
    return len(open_melds) == 0 and score >= 1000


# -- assembled legal action sets ----------------------------------------------
def discard_legal_actions(*, hand, melds, riichi: bool, score: int, drawn_tile: str | None = None) -> list[Action]:
    """Legal actions for a discard decision.

    ``drawn_tile`` distinguishes the two discard windows:

    - a *draw* decision (after ``tsumo``): ``drawn_tile`` is the tile just
      drawn, and tsumo-win / ankan / kakan / riichi are possible;
    - a *call* decision (after ``chi``/``pon``): ``drawn_tile`` is ``None`` and
      only discarding is possible (no draw, hand is open).

    When the player is already in riichi only the drawn tile may be discarded
    (tsumogiri).
    """
    is_draw = drawn_tile is not None
    actions: list[Action] = []
    distinct = sorted(set(hand))
    if riichi:
        if is_draw:
            actions.append(Action(ActionType.DISCARD, tile=drawn_tile))
    else:
        for tile in distinct:
            actions.append(Action(ActionType.DISCARD, tile=tile))
        if is_draw and can_riichi(hand, melds, score):
            # Riichi requires the hand *after* the discard to be tenpai.  Only
            # compute the per-tile tenpai checks when the post-draw hand is
            # already tenpai-or-better (shanten <= 0); a hand with shanten >= 1
            # cannot reach tenpai with a single discard.
            if shanten(hand, melds) <= 0:
                for tile in distinct:
                    if is_tenpai(_remove_one(hand, tile), melds):
                        actions.append(Action(ActionType.RIICHI, tile=tile))

    if is_draw:
        if is_agari(hand, melds):
            actions.append(Action(ActionType.TSUMO))
        actions.extend(ankan_actions(hand))
        actions.extend(kakan_actions(hand, melds))
    return actions


def chankan_legal_actions(
    *,
    hand,
    melds,
    own_discards,
    added_tile: str,
    kan_declarer: int,
) -> list[Action]:
    """Legal actions when another player adds a tile via kakan (chankan).

    Only a ron on the added tile (or pass) is possible — the kan tile cannot be
    called for chi/pon/kan.
    """
    actions: list[Action] = []
    if is_agari(list(hand) + [added_tile], melds) and not is_furiten(hand, melds, own_discards):
        actions.append(Action(ActionType.RON, target=kan_declarer))
    actions.append(Action(ActionType.PASS))
    return actions


def response_legal_actions(
    *,
    hand,
    melds,
    own_discards,
    riichi: bool,
    claimed_tile: str,
    discarder: int,
    is_next: bool,
) -> list[Action]:
    """Legal actions when another player discarded ``claimed_tile``."""
    actions: list[Action] = []

    if is_agari(list(hand) + [claimed_tile], melds) and not is_furiten(hand, melds, own_discards):
        actions.append(Action(ActionType.RON, target=discarder))

    if not riichi and len(melds) < 4:
        if is_next:
            actions.extend(chi_actions(hand, claimed_tile, discarder))
        actions.extend(pon_actions(hand, claimed_tile, discarder))
        actions.extend(daiminkan_actions(hand, claimed_tile, discarder))

    actions.append(Action(ActionType.PASS))
    return actions


__all__ = [
    "can_riichi",
    "chi_actions",
    "pon_actions",
    "daiminkan_actions",
    "ankan_actions",
    "kakan_actions",
    "validate_ankan_consumed",
    "chankan_legal_actions",
    "discard_legal_actions",
    "response_legal_actions",
]
