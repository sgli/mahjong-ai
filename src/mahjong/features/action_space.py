"""Action space and action <-> id mapping (``feature-v1``).

The action space is a fixed, finite enumeration of every action the model can
output.  It covers the 8 action types from ``decision/action.py`` with their
parameters, and maps each to a unique integer id so the policy can output
logits over ``ACTION_SPACE_SIZE`` values.

Design notes
------------

- ``target`` is *not* part of the action identity: ron / pon / chi / daiminkan
  always target the player who just discarded (or the kakan declarer for
  chankan), which is determined by the decision context, not a free choice.
  Ids therefore encode only ``(type, tile/consumed/kan_kind)``.
- Red fives are normalised to their base five so ``5mr``/``5m`` share one id
  (they are the same tile for hand-shape and action purposes).
- Chi is enumerated as ``(claimed tile, consumed pair)`` over the 63 possible
  patterns (21 per suit).

Space layout (``ACTION_SPACE_SIZE = 270``)::

    DISCARD       34   ids 0..33       (one per tile)
    RIICHI        34   ids 34..67      (one per tile)
    TSUMO          1   id 68
    PASS           1   id 69
    RON            1   id 70
    PON           34   ids 71..104     (one per claimed tile)
    CHI           63   ids 105..167    (claimed tile + pattern)
    KAN ankan     34   ids 168..201
    KAN daiminkan 34   ids 202..235
    KAN kakan     34   ids 236..269
"""

from __future__ import annotations

from ..decision.action import Action, ActionType
from ..rules.tiles import ALL_TILES, TILE_TO_INDEX, normalize

#: Version of the feature/action-space encoding.
FEATURE_VERSION = "feature-v1"

_NUM_TILES = len(ALL_TILES)  # 34

_OFFSET_DISCARD = 0
_OFFSET_RIICHI = _OFFSET_DISCARD + _NUM_TILES  # 34
_OFFSET_TSUMO = _OFFSET_RIICHI + _NUM_TILES  # 68
_OFFSET_PASS = _OFFSET_TSUMO + 1  # 69
_OFFSET_RON = _OFFSET_PASS + 1  # 70
_OFFSET_PON = _OFFSET_RON + 1  # 71
_OFFSET_CHI = _OFFSET_PON + _NUM_TILES  # 105

# -- chi pattern enumeration ---------------------------------------------------
_CHI_PATTERNS: list[tuple[str, tuple[str, str]]] = []
for _suit in "mps":
    for _rank in range(1, 10):
        _tile = f"{_rank}{_suit}"
        _patterns = []
        if _rank >= 3:
            _patterns.append((f"{_rank - 2}{_suit}", f"{_rank - 1}{_suit}"))
        if 2 <= _rank <= 8:
            _patterns.append((f"{_rank - 1}{_suit}", f"{_rank + 1}{_suit}"))
        if _rank <= 7:
            _patterns.append((f"{_rank + 1}{_suit}", f"{_rank + 2}{_suit}"))
        for _consumed in _patterns:
            _CHI_PATTERNS.append((_tile, tuple(sorted(_consumed))))

_CHI_COUNT = len(_CHI_PATTERNS)  # 63
_CHI_INDEX = {(t, c): i for i, (t, c) in enumerate(_CHI_PATTERNS)}

_OFFSET_ANKAN = _OFFSET_CHI + _CHI_COUNT  # 168
_OFFSET_DAIMINKAN = _OFFSET_ANKAN + _NUM_TILES  # 202
_OFFSET_KAKAN = _OFFSET_DAIMINKAN + _NUM_TILES  # 236

ACTION_SPACE_SIZE = _OFFSET_KAKAN + _NUM_TILES  # 270


def action_to_id(action: Action) -> int:
    """Map an Action to its fixed integer id (normalising red fives, dropping
    the context-dependent ``target``)."""
    t = action.type
    if t is ActionType.DISCARD:
        return _OFFSET_DISCARD + TILE_TO_INDEX[normalize(action.tile)]
    if t is ActionType.RIICHI:
        return _OFFSET_RIICHI + TILE_TO_INDEX[normalize(action.tile)]
    if t is ActionType.TSUMO:
        return _OFFSET_TSUMO
    if t is ActionType.PASS:
        return _OFFSET_PASS
    if t is ActionType.RON:
        return _OFFSET_RON
    if t is ActionType.PON:
        return _OFFSET_PON + TILE_TO_INDEX[normalize(action.tile)]
    if t is ActionType.CHI:
        tile = normalize(action.tile)
        consumed = tuple(sorted(normalize(x) for x in action.consumed))
        return _OFFSET_CHI + _CHI_INDEX[(tile, consumed)]
    if t is ActionType.KAN:
        kind = action.kan_kind
        if kind == "ankan":
            base = normalize(action.consumed[0])
            return _OFFSET_ANKAN + TILE_TO_INDEX[base]
        if kind == "daiminkan":
            return _OFFSET_DAIMINKAN + TILE_TO_INDEX[normalize(action.tile)]
        if kind == "kakan":
            return _OFFSET_KAKAN + TILE_TO_INDEX[normalize(action.tile)]
        raise ValueError(f"invalid kan_kind {kind!r}")
    raise ValueError(f"unsupported action type {t!r}")


def action_dict_to_id(d: dict) -> int:
    """Map a plain action dict (as stored in ``legal_actions`` JSON) to its id.

    Mirrors :func:`action_to_id` exactly (no ``Action`` object construction).
    """
    t = d["type"]
    tile = d.get("tile")
    consumed = tuple(d.get("consumed") or [])
    kan_kind = d.get("kan_kind")
    if t == "discard":
        return _OFFSET_DISCARD + TILE_TO_INDEX[normalize(tile)]
    if t == "riichi":
        return _OFFSET_RIICHI + TILE_TO_INDEX[normalize(tile)]
    if t == "tsumo":
        return _OFFSET_TSUMO
    if t == "pass":
        return _OFFSET_PASS
    if t == "ron":
        return _OFFSET_RON
    if t == "pon":
        return _OFFSET_PON + TILE_TO_INDEX[normalize(tile)]
    if t == "chi":
        tile_n = normalize(tile)
        consumed_n = tuple(sorted(normalize(x) for x in consumed))
        return _OFFSET_CHI + _CHI_INDEX[(tile_n, consumed_n)]
    if t == "kan":
        if kan_kind == "ankan":
            return _OFFSET_ANKAN + TILE_TO_INDEX[normalize(consumed[0])]
        if kan_kind == "daiminkan":
            return _OFFSET_DAIMINKAN + TILE_TO_INDEX[normalize(tile)]
        if kan_kind == "kakan":
            return _OFFSET_KAKAN + TILE_TO_INDEX[normalize(tile)]
        raise ValueError(f"invalid kan_kind {kan_kind!r}")
    raise ValueError(f"unsupported action type {t!r}")


def action_from_id(action_id: int) -> Action:
    """Inverse of :func:`action_to_id` (returns the canonical Action, target=None)."""
    if not 0 <= action_id < ACTION_SPACE_SIZE:
        raise IndexError(f"action_id {action_id} out of range [0, {ACTION_SPACE_SIZE})")
    if action_id < _OFFSET_RIICHI:
        return Action(ActionType.DISCARD, tile=ALL_TILES[action_id - _OFFSET_DISCARD])
    if action_id < _OFFSET_TSUMO:
        return Action(ActionType.RIICHI, tile=ALL_TILES[action_id - _OFFSET_RIICHI])
    if action_id == _OFFSET_TSUMO:
        return Action(ActionType.TSUMO)
    if action_id == _OFFSET_PASS:
        return Action(ActionType.PASS)
    if action_id == _OFFSET_RON:
        return Action(ActionType.RON)
    if action_id < _OFFSET_CHI:
        tile = ALL_TILES[action_id - _OFFSET_PON]
        return Action(ActionType.PON, tile=tile, consumed=(tile, tile))
    if action_id < _OFFSET_ANKAN:
        tile, consumed = _CHI_PATTERNS[action_id - _OFFSET_CHI]
        return Action(ActionType.CHI, tile=tile, consumed=consumed)
    if action_id < _OFFSET_DAIMINKAN:
        tile = ALL_TILES[action_id - _OFFSET_ANKAN]
        return Action(ActionType.KAN, kan_kind="ankan", consumed=(tile, tile, tile, tile))
    if action_id < _OFFSET_KAKAN:
        tile = ALL_TILES[action_id - _OFFSET_DAIMINKAN]
        return Action(ActionType.KAN, tile=tile, consumed=(tile, tile, tile), kan_kind="daiminkan")
    tile = ALL_TILES[action_id - _OFFSET_KAKAN]
    return Action(ActionType.KAN, tile=tile, consumed=(tile, tile, tile), kan_kind="kakan")


def canonical_action(action: Action) -> Action:
    """Return the canonical (normalised, target-less) form of ``action``."""
    return action_from_id(action_to_id(action))


def legal_mask(legal_actions) -> "torch.Tensor":
    """Boolean mask of shape ``[ACTION_SPACE_SIZE]`` with True for legal ids."""
    import torch  # lazy import: keeps dataset/parser path torch-free

    mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)
    for action in legal_actions:
        mask[action_to_id(action)] = True
    return mask
