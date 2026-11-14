"""Columnar feature-v3 encoder for decision-v3 parquet record batches."""

from __future__ import annotations

import json

import numpy as np
import torch

from ..features.action_space_v2 import ACTION_SPACE_SIZE_V2, action_dict_to_id_v2
from ..features.encoder_v2 import MAX_OPPONENT_RIVER
from ..features.encoder_v3 import (
    DENSE_FEATURE_DIM_V3,
    FEATURE_DIM_V3,
    RIVER_PAD_ID_V3,
    RIVER_TOKEN_COUNT_V3,
)
from ..rules.tiles import index_of
from .fast_encode import _TILE_IDX_V2, _add_v2_counts, _meld_tiles_json


def encode_record_batch_fast_v3(rb):
    """Return ``(features, labels, masks)`` with a compact token suffix.

    The first 397 columns retain feature-v2's non-sequence feature order. The
    final 384 entries are 0..36 tile IDs or 37 (PAD), represented exactly as
    floats for compatibility with the existing trajectory and batch APIs.
    """
    names = (
        "hand", "player_id", "melds", "opponents_melds", "discards", "dora_markers",
        "scores", "riichi", "bakaze", "kyoku", "honba", "kyotaku", "oya", "turn",
        "action_type", "action_tile", "action_consumed", "action_kan_kind", "legal_actions",
    )
    cols = {name: rb.column(name).to_pylist() for name in names}
    batch_size = len(cols["hand"])
    dense = np.zeros((batch_size, DENSE_FEATURE_DIM_V3), dtype=np.float32)
    river_tokens = np.full(
        (batch_size, RIVER_TOKEN_COUNT_V3), RIVER_PAD_ID_V3, dtype=np.float32
    )
    masks = np.zeros((batch_size, ACTION_SPACE_SIZE_V2), dtype=np.bool_)
    labels = np.empty(batch_size, dtype=np.int64)
    off_hand, off_own_meld, off_opp_meld = 0, 37, 74
    off_own_river, off_opp_river = 185, 222
    off_dora, off_global = 333, 367

    for i in range(batch_size):
        seat = int(cols["player_id"][i])
        relative_seats = ((seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4)
        _add_v2_counts(dense[i], cols["hand"][i], off_hand, base_counts=True)
        own_melds = json.loads(cols["melds"][i])
        _add_v2_counts(dense[i], _meld_tiles_json(own_melds), off_own_meld, base_counts=True)

        opponent_melds = json.loads(cols["opponents_melds"][i])
        for rel, abs_seat in enumerate(relative_seats):
            _add_v2_counts(
                dense[i], _meld_tiles_json(opponent_melds[abs_seat]),
                off_opp_meld + rel * 37, base_counts=True,
            )

        rivers = json.loads(cols["discards"][i])
        _add_v2_counts(dense[i], rivers[seat], off_own_river)
        for rel, abs_seat in enumerate(relative_seats):
            river = rivers[abs_seat]
            if len(river) > MAX_OPPONENT_RIVER:
                raise ValueError(
                    f"opponent seat {abs_seat} river has {len(river)} tiles; "
                    f"feature-v3 limit is {MAX_OPPONENT_RIVER}"
                )
            _add_v2_counts(dense[i], river, off_opp_river + rel * 37)
            token_offset = rel * MAX_OPPONENT_RIVER
            for pos, tile in enumerate(river):
                river_tokens[i, token_offset + pos] = _TILE_IDX_V2[tile]

        for tile in cols["dora_markers"][i]:
            dense[i, off_dora + index_of(tile)] += 1.0
        dense[i, off_global + {"E": 0, "S": 1, "W": 2, "N": 3}[cols["bakaze"][i]]] = 1.0
        dense[i, off_global + 4 + int(cols["kyoku"][i]) - 1] = 1.0
        dense[i, off_global + 8] = float(cols["honba"][i])
        dense[i, off_global + 9] = float(cols["kyotaku"][i])
        dense[i, off_global + 10 + int(cols["oya"][i])] = 1.0
        dense[i, off_global + 14 + int(cols["turn"][i])] = 1.0
        dense[i, off_global + 18 + seat] = 1.0
        for j, score in enumerate(cols["scores"][i]):
            dense[i, off_global + 22 + j] = float(score) / 1000.0
        for j, riichi in enumerate(cols["riichi"][i]):
            dense[i, off_global + 26 + j] = float(bool(riichi))

        labels[i] = action_dict_to_id_v2({
            "type": cols["action_type"][i], "tile": cols["action_tile"][i],
            "consumed": cols["action_consumed"][i] or [], "kan_kind": cols["action_kan_kind"][i],
        })
        for action in json.loads(cols["legal_actions"][i]):
            if action["type"] == "kyuushu_kyuuhai":
                continue
            masks[i, action_dict_to_id_v2(action)] = True

    if not np.isfinite(dense).all():
        raise ValueError("non-finite feature-v3 dense features")
    if not masks[np.arange(batch_size), labels].all():
        raise ValueError("gold label absent from feature-v3 legal mask")
    features = np.concatenate((dense, river_tokens), axis=1)
    if features.shape[1] != FEATURE_DIM_V3:
        raise AssertionError(f"feature-v3 dim mismatch: {features.shape[1]} != {FEATURE_DIM_V3}")
    return torch.from_numpy(features), torch.from_numpy(labels), torch.from_numpy(masks)
