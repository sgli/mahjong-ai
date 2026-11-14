"""Feature-v2 encoder: explicit aka counts and per-opponent ordered rivers."""

from __future__ import annotations

import torch

from ..decision.observation import PlayerObservation
from ..rules.tiles import ALL_TILES, RED_FIVE, normalize

FEATURE_VERSION_V2 = "feature-v2"
_BASE_SIZE = len(ALL_TILES)
_TILE_SIZE = _BASE_SIZE + len(RED_FIVE)
_BASE_INDEX = {tile: i for i, tile in enumerate(ALL_TILES)}
_RED_INDEX = {red: _BASE_SIZE + i for i, red in enumerate(RED_FIVE)}
MAX_OPPONENT_RIVER = 128
GLOBAL_DIM_V2 = 30
_GLOBAL_DIM = GLOBAL_DIM_V2
FEATURE_DIM_V2 = (
    37  # hand counts plus three aka counts
    + 37  # own meld counts plus aka counts
    + 3 * 37  # each opponent melds by relative seat, plus aka
    + 37  # own river, including red tile identity
    + 3 * 37  # each opponent river histogram
    + 3 * MAX_OPPONENT_RIVER * _TILE_SIZE  # ordered opponent rivers
    + 34  # dora indicators
    + _GLOBAL_DIM
)


def _tile_id(tile: str) -> int:
    return _RED_INDEX[tile] if tile in _RED_INDEX else _BASE_INDEX[normalize(tile)]


def _tile_vector(tiles) -> torch.Tensor:
    values = torch.zeros(_TILE_SIZE, dtype=torch.float32)
    for tile in tiles:
        values[_tile_id(tile)] += 1
    return values


def _meld_vector(melds) -> torch.Tensor:
    return _tile_vector(tile for meld in melds for tile in meld.tiles)


def _base_vector(tiles) -> torch.Tensor:
    counts = torch.zeros(_BASE_SIZE, dtype=torch.float32)
    for tile in tiles:
        counts[_BASE_INDEX[normalize(tile)]] += 1
    return counts


def _with_red_counts(tiles) -> torch.Tensor:
    tiles = tuple(tiles)
    base = _base_vector(tiles)
    aka = torch.tensor([tiles.count(t) for t in RED_FIVE], dtype=torch.float32)
    return torch.cat((base, aka))


def _melds_with_red_counts(melds) -> torch.Tensor:
    return _with_red_counts(tile for meld in melds for tile in meld.tiles)


def _onehot(index: int, size: int) -> torch.Tensor:
    result = torch.zeros(size, dtype=torch.float32)
    result[index] = 1.0
    return result


def encode_observation_v2(observation: PlayerObservation) -> torch.Tensor:
    seat = observation.seat
    opponents = [(seat + delta) % 4 for delta in (1, 2, 3)]
    sequence = torch.zeros((3, MAX_OPPONENT_RIVER, _TILE_SIZE), dtype=torch.float32)
    for rel, abs_seat in enumerate(opponents):
        river = observation.discards[abs_seat]
        if len(river) > MAX_OPPONENT_RIVER:
            raise ValueError(
                f"opponent seat {abs_seat} river has {len(river)} tiles; "
                f"feature-v2 limit is {MAX_OPPONENT_RIVER}"
            )
        for pos, tile in enumerate(river):
            sequence[rel, pos, _tile_id(tile)] = 1.0

    opponent_melds = [
        _melds_with_red_counts(observation.opponents_melds[s]) for s in opponents
    ]
    opponent_rivers = [_tile_vector(observation.discards[s]) for s in opponents]
    parts = [
        _with_red_counts(observation.hand),
        _melds_with_red_counts(observation.melds),
        *opponent_melds,
        _tile_vector(observation.discards[seat]),
        *opponent_rivers,
        sequence.flatten(),
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
    result = torch.cat(parts)
    if result.numel() != FEATURE_DIM_V2:
        raise AssertionError(f"feature-v2 dim mismatch: {result.numel()} != {FEATURE_DIM_V2}")
    return result


class ObservationEncoderV2:
    feature_version = FEATURE_VERSION_V2
    feature_dim = FEATURE_DIM_V2
    action_schema_version = "action-v2"

    def encode(self, observation: PlayerObservation) -> torch.Tensor:
        return encode_observation_v2(observation)

    def __call__(self, observation: PlayerObservation) -> torch.Tensor:
        return self.encode(observation)

