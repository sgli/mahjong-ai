"""Feature encoder (``feature-v1``).

Deterministically maps a :class:`PlayerObservation` (the visible projection) to
a fixed-size float tensor of shape ``[FEATURE_DIM]``.

It consumes *only* the observation's public fields and never touches a
``ReplayState``, raw event, opponent concealed hand, the wall, ura markers or
any future information (PROJECT_PLAN 3.3, DATA_SPEC 13).

Layout (``FEATURE_DIM = 234``):

=====================  ====  ==================================================
block                  dim   content
=====================  ====  ==================================================
own hand               34    tile-count vector
own melds              34    tile-count vector (own revealed melds)
opponents melds        34    tile-count vector (other players' revealed melds)
own discards           34    tile-count vector
opponents discards     34    tile-count vector
dora indicators        34    tile-count vector
bakaze                 4     one-hot (E/S/W/N)
kyoku                  4     one-hot (1..4)
honba                  1     scalar (raw)
kyotaku                1     scalar (raw)
oya                    4     one-hot
turn                   4     one-hot
seat (self)            4     one-hot
scores                 4     scalars (points / 1000)
riichi                 4     scalars (0/1 per player)
=====================  ====  ==================================================
"""

from __future__ import annotations

import torch

from ..decision.observation import PlayerObservation
from ..rules.tiles import ALL_TILES, TILE_TO_INDEX, normalize
from .action_space import FEATURE_VERSION

_WINDS = {"E": 0, "S": 1, "W": 2, "N": 3}
_NUM_TILES = len(ALL_TILES)

FEATURE_DIM = _NUM_TILES * 6 + 4 * 5 + 2 + 4 + 4  # 234


def _count_vector(tiles) -> torch.Tensor:
    counts = torch.zeros(_NUM_TILES, dtype=torch.float32)
    for tile in tiles:
        counts[TILE_TO_INDEX[normalize(tile)]] += 1.0
    return counts


def _meld_counts(melds) -> torch.Tensor:
    counts = torch.zeros(_NUM_TILES, dtype=torch.float32)
    for meld in melds:
        for tile in meld.tiles:
            counts[TILE_TO_INDEX[normalize(tile)]] += 1.0
    return counts


def _onehot(index: int, size: int) -> torch.Tensor:
    vec = torch.zeros(size, dtype=torch.float32)
    vec[index] = 1.0
    return vec


def encode_observation(observation: PlayerObservation) -> torch.Tensor:
    """Encode a single observation to ``[FEATURE_DIM]`` float32 tensor."""
    seat = observation.seat
    parts = [
        _count_vector(observation.hand),
        _meld_counts(observation.melds),
        _meld_counts(m for slot in observation.opponents_melds for m in slot),
        _count_vector(observation.discards[seat]),
        _count_vector(
            t for s in range(4) if s != seat for t in observation.discards[s]
        ),
        _count_vector(observation.dora_markers),
        _onehot(_WINDS[observation.bakaze], 4),
        _onehot(observation.kyoku - 1, 4),
        torch.tensor([float(observation.honba)], dtype=torch.float32),
        torch.tensor([float(observation.kyotaku)], dtype=torch.float32),
        _onehot(observation.oya, 4),
        _onehot(observation.turn, 4),
        _onehot(seat, 4),
        torch.tensor([float(s) / 1000.0 for s in observation.scores], dtype=torch.float32),
        torch.tensor([1.0 if r else 0.0 for r in observation.riichi], dtype=torch.float32),
    ]
    return torch.cat(parts)


class ObservationEncoder:
    """Callable encoder (keeps ``feature_version`` and ``feature_dim`` metadata)."""

    feature_version = FEATURE_VERSION
    feature_dim = FEATURE_DIM

    def encode(self, observation: PlayerObservation) -> torch.Tensor:
        return encode_observation(observation)

    def __call__(self, observation: PlayerObservation) -> torch.Tensor:
        return self.encode(observation)
