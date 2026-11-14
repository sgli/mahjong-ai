"""Fast row/batch encoder must be bitwise identical to the reference encoder."""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from mahjong.dataset.schema import row_to_sample
from mahjong.features import ObservationEncoder
from mahjong.features.action_space import action_to_id, legal_mask
from mahjong.features.action_space_v2 import action_to_id_v2, legal_mask_v2
from mahjong.features.encoder_v2 import ObservationEncoderV2
from mahjong.training.fast_encode import (
    encode_record_batch_fast,
    encode_record_batch_fast_v2,
    encode_row_fast,
    encode_rows_fast,
)


def _first_shard() -> Path | None:
    root = Path("data/processed/decision-v1")
    for split in ("train", "validation", "test"):
        d = root / split
        if d.is_dir():
            shards = sorted(d.glob("*.parquet"))
            if shards:
                return shards[0]
    return None


def _first_v2_source_shard() -> Path | None:
    root = Path("data/processed/decision-v2/train")
    shards = sorted(root.glob("*.parquet")) if root.is_dir() else []
    return shards[0] if shards else None


@pytest.mark.skipif(_first_shard() is None, reason="no decision-v1 parquet available")
def test_encode_row_fast_bitwise_identical():
    shard = _first_shard()
    table = pq.read_table(shard)
    # cap to a few thousand rows for test speed
    rows = table.slice(0, min(3000, table.num_rows)).to_pylist()
    encoder = ObservationEncoder()
    for row in rows:
        reference = encoder.encode(row_to_sample(row).observation)
        fast = encode_row_fast(row)
        assert torch.equal(reference, fast), f"mismatch for row {row['game_id']}"


@pytest.mark.skipif(_first_shard() is None, reason="no decision-v1 parquet available")
def test_encode_rows_fast_batch_identity():
    shard = _first_shard()
    table = pq.read_table(shard)
    rows = table.slice(0, min(500, table.num_rows)).to_pylist()
    encoder = ObservationEncoder()
    features, action_ids, masks = encode_rows_fast(rows)
    assert features.shape == (len(rows), 234)
    assert action_ids.shape == (len(rows),)
    assert masks.shape == (len(rows), 270)
    for i, row in enumerate(rows):
        sample = row_to_sample(row)
        assert torch.equal(features[i], encoder.encode(sample.observation))
        assert action_ids[i].item() == action_to_id(sample.action)
        assert torch.equal(masks[i], legal_mask(sample.legal_actions))


@pytest.mark.skipif(_first_shard() is None, reason="no decision-v1 parquet available")
def test_encode_record_batch_fast_matches_rows():
    shard = _first_shard()
    table = pq.read_table(shard)
    rb = table.slice(0, min(2000, table.num_rows)).to_batches(max_chunksize=1000)[0]
    rows = rb.to_pylist()
    f1, a1, m1 = encode_rows_fast(rows)
    f2, a2, m2 = encode_record_batch_fast(rb)
    assert torch.equal(f1, f2)
    assert torch.equal(a1, a2)
    assert torch.equal(m1, m2)


@pytest.mark.skipif(_first_v2_source_shard() is None, reason="no decision-v2 parquet available")
def test_encode_record_batch_fast_v2_matches_reference():
    shard = _first_v2_source_shard()
    rb = next(pq.ParquetFile(shard).iter_batches(batch_size=32))
    rows = rb.to_pylist()
    features, labels, masks = encode_record_batch_fast_v2(rb)
    encoder = ObservationEncoderV2()
    assert features.shape[1] == encoder.feature_dim
    for i, row in enumerate(rows):
        sample = row_to_sample(row)
        assert torch.equal(features[i], encoder.encode(sample.observation))
        assert labels[i].item() == action_to_id_v2(sample.action)
        assert torch.equal(masks[i], legal_mask_v2(sample.legal_actions))
