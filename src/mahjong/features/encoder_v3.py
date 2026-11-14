"""Feature-v3: dense public state plus tokenized relative opponent rivers.

The raw decision-v3 rows already retain the ordered discard arrays. This
encoder replaces feature-v2's 3 x 128 x 37 river one-hot block with integer
tile tokens; :class:`RiverSequencePolicy` embeds and sequence-encodes them.
"""

from __future__ import annotations

import torch

from ..decision.observation import PlayerObservation
from .encoder_v2 import (
    MAX_OPPONENT_RIVER,
    _base_vector,
    _melds_with_red_counts,
    _onehot,
    _tile_id,
    _tile_vector,
    _with_red_counts,
)

FEATURE_VERSION_V3 = "feature-v3"
RIVER_COUNT_V3 = 3
RIVER_VOCAB_SIZE_V3 = 38  # 37 normal/aka identities + PAD
RIVER_PAD_ID_V3 = 37
RIVER_TOKEN_COUNT_V3 = RIVER_COUNT_V3 * MAX_OPPONENT_RIVER
DENSE_FEATURE_DIM_V3 = 397
FEATURE_DIM_V3 = DENSE_FEATURE_DIM_V3 + RIVER_TOKEN_COUNT_V3


def encode_observation_v3(observation: PlayerObservation) -> torch.Tensor:
    """Encode one observation into dense features and padded river token IDs.

    The flattened token suffix is integer-valued float32 so the existing
    batch/trajectory interfaces can carry it without a second tensor channel.
    The model converts that suffix to integer IDs before embedding.
    """
    seat = observation.seat
    opponents = ((seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4)
    rivers = torch.full((RIVER_COUNT_V3, MAX_OPPONENT_RIVER), RIVER_PAD_ID_V3, dtype=torch.long)
    for rel, abs_seat in enumerate(opponents):
        river = observation.discards[abs_seat]
        if len(river) > MAX_OPPONENT_RIVER:
            raise ValueError(
                f"opponent seat {abs_seat} river has {len(river)} tiles; "
                f"feature-v3 limit is {MAX_OPPONENT_RIVER}"
            )
        for pos, tile in enumerate(river):
            rivers[rel, pos] = _tile_id(tile)

    dense_parts = [
        _with_red_counts(observation.hand),
        _melds_with_red_counts(observation.melds),
        *[_melds_with_red_counts(observation.opponents_melds[s]) for s in opponents],
        _tile_vector(observation.discards[seat]),
        *[_tile_vector(observation.discards[s]) for s in opponents],
        _base_vector(observation.dora_markers),
        _onehot({"E": 0, "S": 1, "W": 2, "N": 3}[observation.bakaze], 4),
        _onehot(observation.kyoku - 1, 4),
        torch.tensor([float(observation.honba), float(observation.kyotaku)]),
        _onehot(observation.oya, 4),
        _onehot(observation.turn, 4),
        _onehot(seat, 4),
        torch.tensor([s / 1000.0 for s in observation.scores]),
        torch.tensor([float(r) for r in observation.riichi]),
    ]
    dense = torch.cat(dense_parts)
    if dense.numel() != DENSE_FEATURE_DIM_V3:
        raise AssertionError(f"feature-v3 dense dim mismatch: {dense.numel()} != {DENSE_FEATURE_DIM_V3}")
    result = torch.cat((dense, rivers.flatten().to(torch.float32)))
    if result.numel() != FEATURE_DIM_V3:
        raise AssertionError(f"feature-v3 dim mismatch: {result.numel()} != {FEATURE_DIM_V3}")
    return result


class ObservationEncoderV3:
    feature_version = FEATURE_VERSION_V3
    feature_dim = FEATURE_DIM_V3
    action_schema_version = "action-v2"

    def encode(self, observation: PlayerObservation) -> torch.Tensor:
        return encode_observation_v3(observation)

    def __call__(self, observation: PlayerObservation) -> torch.Tensor:
        return self.encode(observation)
