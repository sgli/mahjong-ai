"""Fast row/batch encoder (Phase 10.1.1-B §5/§6/§7/§8).

Reads Parquet/Arrow row columns directly and produces the ``feature-v1``
``[234]`` feature, the target action id, and the ``[270]`` legal mask WITHOUT
rebuilding the ``DecisionSample / PlayerObservation / Meld / Action`` object
tree.  It must be bitwise identical to ``ObservationEncoder.encode`` +
``action_to_id`` + ``legal_mask`` (verified by tests).
"""

from __future__ import annotations

import json

import numpy as np
import torch

from ..features.action_space import ACTION_SPACE_SIZE, action_dict_to_id
from ..features.encoder import FEATURE_DIM, _WINDS
from ..rules.tiles import RED_FIVE, TILE_TO_INDEX

_TILE_IDX: dict[str, int] = dict(TILE_TO_INDEX)
for _r, _b in RED_FIVE.items():
    _TILE_IDX[_r] = TILE_TO_INDEX[_b]

_BLOCKS = 6  # hand / melds / opponents melds / own discards / opp discards / dora


def _add_tile_counts(features: np.ndarray, row_idx: int, tiles, offset: int) -> None:
    """Accumulate ``tiles`` counts into ``features[..., offset:offset+34]``."""
    if not tiles:
        return
    if features.ndim == 1:
        for t in tiles:
            features[offset + _TILE_IDX[t]] += 1.0
    else:
        for t in tiles:
            features[row_idx, offset + _TILE_IDX[t]] += 1.0


def _meld_tiles(melds: list) -> list:
    return [t for m in melds for t in m["tiles"]]


def encode_row_fast(row: dict) -> torch.Tensor:
    """Fast single-row encode (bitwise identical to ``ObservationEncoder.encode``)."""
    seat = int(row["player_id"])
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)

    _add_tile_counts(feats, 0, row["hand"], 0)
    _add_tile_counts(feats, 0, _meld_tiles(json.loads(row["melds"])), 34)
    opp = json.loads(row["opponents_melds"])
    _add_tile_counts(feats, 0, [t for slot in opp for m in slot for t in m["tiles"]], 68)
    discards = json.loads(row["discards"])
    _add_tile_counts(feats, 0, discards[seat], 102)
    _add_tile_counts(feats, 0, [t for s in range(4) if s != seat for t in discards[s]], 136)
    _add_tile_counts(feats, 0, row["dora_markers"], 170)

    feats[204 + _WINDS[row["bakaze"]]] = 1.0
    feats[208 + (int(row["kyoku"]) - 1)] = 1.0
    feats[212] = float(row["honba"])
    feats[213] = float(row["kyotaku"])
    feats[214 + int(row["oya"])] = 1.0
    feats[218 + int(row["turn"])] = 1.0
    feats[222 + seat] = 1.0
    for i, s in enumerate(row["scores"]):
        feats[226 + i] = float(s) / 1000.0
    for i, r in enumerate(row["riichi"]):
        feats[230 + i] = 1.0 if r else 0.0

    return torch.from_numpy(feats)


def _row_action_id(row: dict) -> int:
    return action_dict_to_id(
        {
            "type": row["action_type"],
            "tile": row["action_tile"],
            "consumed": row["action_consumed"] or [],
            "kan_kind": row["action_kan_kind"],
        }
    )


def encode_rows_fast(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch encode a list of plain-dict rows -> (features [B,234], action_ids [B], masks [B,270])."""
    B = len(rows)
    feats = np.zeros((B, FEATURE_DIM), dtype=np.float32)
    masks = np.zeros((B, ACTION_SPACE_SIZE), dtype=np.bool_)
    action_ids = np.empty(B, dtype=np.int64)

    for i, row in enumerate(rows):
        seat = int(row["player_id"])
        _add_tile_counts(feats, i, row["hand"], 0)
        _add_tile_counts(feats, i, _meld_tiles(json.loads(row["melds"])), 34)
        opp = json.loads(row["opponents_melds"])
        _add_tile_counts(feats, i, [t for slot in opp for m in slot for t in m["tiles"]], 68)
        discards = json.loads(row["discards"])
        _add_tile_counts(feats, i, discards[seat], 102)
        _add_tile_counts(feats, i, [t for s in range(4) if s != seat for t in discards[s]], 136)
        _add_tile_counts(feats, i, row["dora_markers"], 170)

        feats[i, 204 + _WINDS[row["bakaze"]]] = 1.0
        feats[i, 208 + (int(row["kyoku"]) - 1)] = 1.0
        feats[i, 212] = float(row["honba"])
        feats[i, 213] = float(row["kyotaku"])
        feats[i, 214 + int(row["oya"])] = 1.0
        feats[i, 218 + int(row["turn"])] = 1.0
        feats[i, 222 + seat] = 1.0
        for s_idx, s in enumerate(row["scores"]):
            feats[i, 226 + s_idx] = float(s) / 1000.0
        for r_idx, r in enumerate(row["riichi"]):
            feats[i, 230 + r_idx] = 1.0 if r else 0.0

        action_ids[i] = _row_action_id(row)
        for a in json.loads(row["legal_actions"]):
            masks[i, action_dict_to_id(a)] = True

    return torch.from_numpy(feats), torch.from_numpy(action_ids), torch.from_numpy(masks)


def encode_record_batch_fast(rb) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Columnar extraction from an Arrow RecordBatch (avoids full ``to_pylist`` per row).

    Still needs per-row JSON parsing for ``melds/opponents_melds/discards/
    legal_actions``, but skips object-tree reconstruction and per-sample tensor
    allocation.
    """
    columns = {
        "hand": rb.column("hand").to_pylist(),
        "player_id": rb.column("player_id").to_pylist(),
        "melds": rb.column("melds").to_pylist(),
        "opponents_melds": rb.column("opponents_melds").to_pylist(),
        "discards": rb.column("discards").to_pylist(),
        "dora_markers": rb.column("dora_markers").to_pylist(),
        "scores": rb.column("scores").to_pylist(),
        "riichi": rb.column("riichi").to_pylist(),
        "bakaze": rb.column("bakaze").to_pylist(),
        "kyoku": rb.column("kyoku").to_pylist(),
        "honba": rb.column("honba").to_pylist(),
        "kyotaku": rb.column("kyotaku").to_pylist(),
        "oya": rb.column("oya").to_pylist(),
        "turn": rb.column("turn").to_pylist(),
        "action_type": rb.column("action_type").to_pylist(),
        "action_tile": rb.column("action_tile").to_pylist(),
        "action_consumed": rb.column("action_consumed").to_pylist(),
        "action_kan_kind": rb.column("action_kan_kind").to_pylist(),
        "legal_actions": rb.column("legal_actions").to_pylist(),
    }
    B = len(columns["hand"])
    feats = np.zeros((B, FEATURE_DIM), dtype=np.float32)
    masks = np.zeros((B, ACTION_SPACE_SIZE), dtype=np.bool_)
    action_ids = np.empty(B, dtype=np.int64)

    for i in range(B):
        seat = int(columns["player_id"][i])
        _add_tile_counts(feats, i, columns["hand"][i], 0)
        _add_tile_counts(feats, i, _meld_tiles(json.loads(columns["melds"][i])), 34)
        opp = json.loads(columns["opponents_melds"][i])
        _add_tile_counts(feats, i, [t for slot in opp for m in slot for t in m["tiles"]], 68)
        discards = json.loads(columns["discards"][i])
        _add_tile_counts(feats, i, discards[seat], 102)
        _add_tile_counts(feats, i, [t for s in range(4) if s != seat for t in discards[s]], 136)
        _add_tile_counts(feats, i, columns["dora_markers"][i], 170)

        feats[i, 204 + _WINDS[columns["bakaze"][i]]] = 1.0
        feats[i, 208 + (int(columns["kyoku"][i]) - 1)] = 1.0
        feats[i, 212] = float(columns["honba"][i])
        feats[i, 213] = float(columns["kyotaku"][i])
        feats[i, 214 + int(columns["oya"][i])] = 1.0
        feats[i, 218 + int(columns["turn"][i])] = 1.0
        feats[i, 222 + seat] = 1.0
        for s_idx, s in enumerate(columns["scores"][i]):
            feats[i, 226 + s_idx] = float(s) / 1000.0
        for r_idx, r in enumerate(columns["riichi"][i]):
            feats[i, 230 + r_idx] = 1.0 if r else 0.0

        action_ids[i] = action_dict_to_id(
            {
                "type": columns["action_type"][i],
                "tile": columns["action_tile"][i],
                "consumed": columns["action_consumed"][i] or [],
                "kan_kind": columns["action_kan_kind"][i],
            }
        )
        for a in json.loads(columns["legal_actions"][i]):
            masks[i, action_dict_to_id(a)] = True

    return torch.from_numpy(feats), torch.from_numpy(action_ids), torch.from_numpy(masks)


__all__ = [
    "encode_record_batch_fast",
    "encode_row_fast",
    "encode_rows_fast",
]
