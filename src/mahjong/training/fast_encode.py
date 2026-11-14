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
from ..features.action_space_v2 import ACTION_SPACE_SIZE_V2, action_dict_to_id_v2
from ..features.encoder_v2 import FEATURE_DIM_V2, MAX_OPPONENT_RIVER, GLOBAL_DIM_V2
from ..rules.tiles import RED_FIVE, TILE_TO_INDEX

_TILE_IDX: dict[str, int] = dict(TILE_TO_INDEX)
for _r, _b in RED_FIVE.items():
    _TILE_IDX[_r] = TILE_TO_INDEX[_b]
_RED_TILE_IDX = {tile: index for index, tile in enumerate(("5mr", "5pr", "5sr"))}
_TILE_IDX_V2 = dict(_TILE_IDX)
_TILE_IDX_V2.update({tile: 34 + index for tile, index in _RED_TILE_IDX.items()})


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


def _tile_idx_v2(tile: str) -> int:
    return _TILE_IDX_V2[tile]


def _add_v2_counts(dst: np.ndarray, tiles, offset: int, *, base_counts: bool = False) -> None:
    if not tiles:
        return
    for tile in tiles:
        idx = _TILE_IDX[tile] if base_counts else _tile_idx_v2(tile)
        dst[offset + idx] += 1.0
        if base_counts and tile in _RED_TILE_IDX:
            # Base count plus a separate aka count within this 37-wide block.
            red_idx = _RED_TILE_IDX[tile]
            dst[offset + 34 + red_idx] += 1.0


def _meld_tiles_json(melds: list) -> list[str]:
    return [tile for meld in melds for tile in meld["tiles"]]


def encode_record_batch_fast_v2(rb, *, compact_transport: bool = False):
    """Fast columnar conversion to feature-v2/action-v2 training tensors.

    ``compact_transport`` stores the integral feature prefix as uint8 and emits
    the 30 floating global features separately. Training reconstructs the exact
    float32 feature tensor on the GPU; this cuts DataLoader IPC/pinned-memory
    traffic by roughly 4x without quantizing model inputs.
    """
    cols = {name: rb.column(name).to_pylist() for name in (
        "hand", "player_id", "melds", "opponents_melds", "discards", "dora_markers",
        "scores", "riichi", "bakaze", "kyoku", "honba", "kyotaku", "oya", "turn",
        "action_type", "action_tile", "action_consumed", "action_kan_kind", "legal_actions",
    )}
    batch_size = len(cols["hand"])
    feats = np.zeros(
        (batch_size, FEATURE_DIM_V2),
        dtype=np.uint8 if compact_transport else np.float32,
    )
    global_features = np.zeros((batch_size, GLOBAL_DIM_V2), dtype=np.float32) if compact_transport else None
    masks = np.zeros((batch_size, ACTION_SPACE_SIZE_V2), dtype=np.bool_)
    labels = np.empty(batch_size, dtype=np.int64)
    # Blocks: hand(37), own meld(37), three opponent melds(111), own river(37),
    # three opponent histograms(111), ordered rivers(3*128*37), dora(34), globals(30).
    off_hand = 0
    off_own_meld = 37
    off_opp_meld = 74
    off_own_river = 185
    off_opp_river = 222
    off_sequence = 333
    off_dora = off_sequence + 3 * MAX_OPPONENT_RIVER * 37
    off_global = off_dora + 34
    for i in range(batch_size):
        seat = int(cols["player_id"][i])
        _add_v2_counts(feats[i], cols["hand"][i], off_hand, base_counts=True)
        own_melds = json.loads(cols["melds"][i])
        _add_v2_counts(feats[i], _meld_tiles_json(own_melds), off_own_meld, base_counts=True)
        all_melds = json.loads(cols["opponents_melds"][i])
        relative_seats = ((seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4)
        for rel, abs_seat in enumerate(relative_seats):
            tiles = _meld_tiles_json(all_melds[abs_seat])
            _add_v2_counts(feats[i], tiles, off_opp_meld + rel * 37, base_counts=True)

        rivers = json.loads(cols["discards"][i])
        _add_v2_counts(feats[i], rivers[seat], off_own_river)
        for rel, abs_seat in enumerate(relative_seats):
            river = rivers[abs_seat]
            if len(river) > MAX_OPPONENT_RIVER:
                raise ValueError(f"river overflow in parquet batch: {len(river)} > {MAX_OPPONENT_RIVER}")
            _add_v2_counts(feats[i], river, off_opp_river + rel * 37)
            seq_base = off_sequence + rel * MAX_OPPONENT_RIVER * 37
            for pos, tile in enumerate(river):
                feats[i, seq_base + pos * 37 + _tile_idx_v2(tile)] = 1.0

        for tile in cols["dora_markers"][i]:
            feats[i, off_dora + _TILE_IDX[tile]] += 1.0
        global_row = global_features[i] if compact_transport else feats[i, off_global:off_global + GLOBAL_DIM_V2]
        global_row[_WINDS[cols["bakaze"][i]]] = 1.0
        global_row[4 + int(cols["kyoku"][i]) - 1] = 1.0
        global_row[8] = float(cols["honba"][i])
        global_row[9] = float(cols["kyotaku"][i])
        global_row[10 + int(cols["oya"][i])] = 1.0
        global_row[14 + int(cols["turn"][i])] = 1.0
        global_row[18 + seat] = 1.0
        for j, score in enumerate(cols["scores"][i]):
            global_row[22 + j] = float(score) / 1000.0
        for j, riichi in enumerate(cols["riichi"][i]):
            global_row[26 + j] = float(bool(riichi))

        label_action = {
            "type": cols["action_type"][i], "tile": cols["action_tile"][i],
            "consumed": cols["action_consumed"][i] or [], "kan_kind": cols["action_kan_kind"][i],
        }
        labels[i] = action_dict_to_id_v2(label_action)
        for action in json.loads(cols["legal_actions"][i]):
            if action["type"] == "kyuushu_kyuuhai":
                continue  # no supervised kyuushu examples in this source dataset
            masks[i, action_dict_to_id_v2(action)] = True

    globals_view = global_features if compact_transport else feats[:, off_global:]
    if not np.isfinite(globals_view).all():
        raise ValueError("non-finite feature-v2 global features")
    if not masks[np.arange(batch_size), labels].all():
        raise ValueError("gold label absent from feature-v2 legal mask")
    result = (torch.from_numpy(feats), torch.from_numpy(labels), torch.from_numpy(masks))
    if compact_transport:
        return (*result, torch.from_numpy(global_features))
    return result


__all__ = [
    "encode_record_batch_fast",
    "encode_record_batch_fast_v2",
    "encode_row_fast",
    "encode_rows_fast",
]
