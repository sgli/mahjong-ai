"""Versioned action mapping that preserves red-five identity (action-v2).

The action list starts with the stable feature-v1 actions. It adds only
physically possible one-red placements, then a final kyuushu kyuuhai ID.
Legal masks exclude contextually impossible variants.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from ..decision.action import Action, ActionType
from ..rules.tiles import RED_FIVE
from .action_space import ACTION_SPACE_SIZE as V1_ACTION_SPACE_SIZE
from .action_space import action_from_id as action_from_id_v1
from .action_space import action_to_id as action_to_id_v1
from .action_space import action_dict_to_id as action_dict_to_id_v1

ACTION_SCHEMA_VERSION = "action-v2"
_RED_TILES = {red: base for red, base in RED_FIVE.items()}
_FIVES = frozenset(RED_FIVE.values())


def _key(action: Action) -> tuple[int, int]:
    """Stable key: v1 base id plus red-role bit mask."""
    base_id = action_to_id_v1(action)
    flags = int(action.tile in _RED_TILES)
    for i, tile in enumerate(action.consumed):
        if i >= 4:
            raise ValueError(f"action has too many consumed tiles: {action!r}")
        if tile in _RED_TILES:
            flags |= 1 << (i + 1)
    return base_id, flags


def _red_variants(action: Action):
    """Enumerate the base action plus each one-red placement that can exist."""
    yield action
    if action.tile in _FIVES:
        yield Action(action.type, tile=f"{action.tile}r", consumed=action.consumed,
                     target=action.target, kan_kind=action.kan_kind)
    for i, tile in enumerate(action.consumed):
        if tile not in _FIVES:
            continue
        consumed = list(action.consumed)
        consumed[i] = f"{tile}r"
        yield Action(action.type, tile=action.tile, consumed=tuple(consumed),
                     target=action.target, kan_kind=action.kan_kind)


# Keep all normal v1 actions in their original order, then append their red
# variants immediately after each base id. The kyuushu action occupies one final
# id. Duplicate variants from identical consumed tiles are deduplicated.
_ACTIONS: list[Action] = []
_KEY_TO_ID: dict[tuple[int, int], int] = {}
for _base_id in range(V1_ACTION_SPACE_SIZE):
    _base = action_from_id_v1(_base_id)
    for _variant in sorted(((_key(a), a) for a in _red_variants(_base)), key=lambda item: item[0]):
        _variant_key, _variant_action = _variant
        if _variant_key not in _KEY_TO_ID:
            _KEY_TO_ID[_variant_key] = len(_ACTIONS)
            _ACTIONS.append(_variant_action)
_KYUUSHU_ID = len(_ACTIONS)
ACTION_SPACE_SIZE_V2 = _KYUUSHU_ID + 1


def action_to_id_v2(action: Action) -> int:
    """Map an action injectively, preserving red tile roles."""
    if action.type is ActionType.KYUUSHU_KYUHAI:
        return _KYUUSHU_ID
    key = _key(action)
    try:
        return _KEY_TO_ID[key]
    except KeyError as exc:
        raise ValueError(f"action variant is not representable in action-v2: {action!r}") from exc


@lru_cache(maxsize=8192)
def _action_components_to_id_v2(
    action_type: str,
    tile: str | None,
    consumed: tuple[str, ...],
    kan_kind: str | None,
) -> int:
    """Cached pure mapping keyed only by serialized action semantics."""
    # Canonicalize only on a cache miss. Legal menus repeat action components
    # many times per batch, so sorting before entering this cache wastes work.
    consumed = tuple(sorted(consumed))
    if action_type == ActionType.KYUUSHU_KYUHAI.value:
        return _KYUUSHU_ID
    data = {"type": action_type, "tile": tile, "consumed": consumed, "kan_kind": kan_kind}
    base_id = action_dict_to_id_v1(data)
    flags = int(tile in _RED_TILES)
    for i, consumed_tile in enumerate(consumed):
        if i >= 4:
            raise ValueError("action-v2 supports at most four consumed tiles")
        if consumed_tile in _RED_TILES:
            flags |= 1 << (i + 1)
    try:
        return _KEY_TO_ID[(base_id, flags)]
    except KeyError as exc:
        raise ValueError(f"serialized action variant is not representable: {data!r}") from exc


def action_dict_to_id_v2(data: dict) -> int:
    """Map a row's serialized action dict without constructing an Action.

    Legal-action menus repeat the same small set of action variants across many
    rows. Cache their pure schema mapping to avoid rebuilding the v1 id and red
    role flags for each occurrence.
    """
    consumed = tuple(data.get("consumed") or ())
    return _action_components_to_id_v2(
        data["type"], data.get("tile"), consumed, data.get("kan_kind")
    )


def action_from_id_v2(action_id: int) -> Action:
    """Decode an action-v2 id to its canonical action (target is contextual)."""
    if not 0 <= action_id < ACTION_SPACE_SIZE_V2:
        raise IndexError(f"action_id {action_id} out of range [0, {ACTION_SPACE_SIZE_V2})")
    if action_id == _KYUUSHU_ID:
        return Action(ActionType.KYUUSHU_KYUHAI)
    try:
        return _ACTIONS[action_id]
    except IndexError as exc:
        raise IndexError(f"action_id {action_id} out of range [0, {ACTION_SPACE_SIZE_V2})") from exc


def legal_mask_v2(legal_actions, *, allow_kyuushu: bool = False) -> torch.Tensor:
    """Build an action-v2 mask. Kyuushu stays disabled until supervised labels exist."""
    mask = torch.zeros(ACTION_SPACE_SIZE_V2, dtype=torch.bool)
    for action in legal_actions:
        if action.type is ActionType.KYUUSHU_KYUHAI and not allow_kyuushu:
            continue
        mask[action_to_id_v2(action)] = True
    return mask

