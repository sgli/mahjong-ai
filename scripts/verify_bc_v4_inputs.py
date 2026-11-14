#!/usr/bin/env python
"""Bounded BC-v4 feature-v3 parity and policy-forward gate."""
from __future__ import annotations
import json
from dataclasses import replace
from pathlib import Path
import sys
import torch
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mahjong.features.encoder_v2 import MAX_OPPONENT_RIVER, _tile_id
from mahjong.features.encoder_v3 import FEATURE_DIM_V3, RIVER_PAD_ID_V3, ObservationEncoderV3
from mahjong.features.policy_schema import resolve_policy_schema
from mahjong.model import build_policy_model
from mahjong.training.bc import row_to_tensors
from mahjong.dataset.schema import row_to_sample
from mahjong.training.data_gate import require_full_audit
from mahjong.training.fast_encode_v3 import encode_record_batch_fast_v3


def main() -> int:
    data = ROOT / "data/processed/decision-v3"
    audit = ROOT / "experiments/bc_v3/reference/decision_v3_full_audit.json"
    require_full_audit(audit, data)
    encoder = ObservationEncoderV3()
    all_features, all_labels, all_masks = [], [], []
    per_split = {}
    max_seen = 0
    for split in ("train", "validation", "test"):
        shards = sorted((data / split).glob("*.parquet"))
        if not shards:
            raise RuntimeError(f"no parquet shards for {split}")
        indices = sorted(set((0, len(shards) // 2, len(shards) - 1)))
        checked = 0
        for index in indices:
            pf = pq.ParquetFile(shards[index])
            rb = next(pf.iter_batches(batch_size=24))
            rows = rb.to_pylist()
            fast_f, fast_y, fast_m = encode_record_batch_fast_v3(rb)
            slow = [row_to_tensors(row, encoder) for row in rows]
            slow_f = torch.stack([x[0] for x in slow])
            slow_y = torch.tensor([x[1] for x in slow], dtype=torch.long)
            slow_m = torch.stack([x[2] for x in slow])
            if not (torch.equal(fast_f, slow_f) and torch.equal(fast_y, slow_y) and torch.equal(fast_m, slow_m)):
                raise AssertionError(f"fast/reference mismatch in {split} shard {index}")
            tokens = fast_f[:, 397:].reshape(-1, 3, MAX_OPPONENT_RIVER).to(torch.long)
            if torch.any((tokens < 0) | (tokens > RIVER_PAD_ID_V3)):
                raise AssertionError(f"invalid river token in {split} shard {index}")
            for river in tokens.reshape(-1, MAX_OPPONENT_RIVER).tolist():
                pad_positions = [i for i, tile in enumerate(river) if tile == RIVER_PAD_ID_V3]
                if pad_positions and any(tile != RIVER_PAD_ID_V3 for tile in river[pad_positions[0]:]):
                    raise AssertionError("non-PAD token occurs after PAD")
                max_seen = max(max_seen, next((i for i, t in enumerate(river) if t == RIVER_PAD_ID_V3), MAX_OPPONENT_RIVER))
            if not fast_m.gather(1, fast_y[:, None]).all():
                raise AssertionError(f"gold action absent from mask in {split} shard {index}")
            all_features.append(fast_f)
            all_labels.append(fast_y)
            all_masks.append(fast_m)
            checked += len(rows)
        per_split[split] = {"shards_sampled": len(indices), "rows_checked": checked}
    features = torch.cat(all_features)
    labels = torch.cat(all_labels)
    masks = torch.cat(all_masks)
    encoder_obj, feature_dim, action_size = resolve_policy_schema("feature-v3", "action-v2")
    if feature_dim != FEATURE_DIM_V3 or features.shape[1] != feature_dim or action_size != 325:
        raise AssertionError("feature/action schema dimensions differ")
    model = build_policy_model(
        "feature-v3", feature_dim, action_size, (256, 256),
        model_config={"tile_embedding_dim": 16, "river_hidden_size": 64, "river_num_layers": 1},
    ).eval()
    base_observation = row_to_sample(rows[0]).observation
    relative_seat = (base_observation.seat + 1) % 4
    river_128 = tuple(("1m", "2m", "3m") * 42 + ("4m", "5m"))
    discards = list(base_observation.discards)
    discards[relative_seat] = river_128
    full_observation = replace(base_observation, discards=tuple(discards))
    full_features = encoder.encode(full_observation)
    expected_river = torch.tensor(
        [_tile_id(tile) for tile in river_128] + [RIVER_PAD_ID_V3] * (MAX_OPPONENT_RIVER - len(river_128)),
        dtype=torch.long,
    )
    if not torch.equal(full_features[397:].reshape(3, MAX_OPPONENT_RIVER)[0].long(), expected_river):
        raise AssertionError("128-tile river order or padding mismatch")
    discards[relative_seat] = river_128 + ("6m",)
    try:
        encoder.encode(replace(base_observation, discards=tuple(discards)))
    except ValueError:
        pass
    else:
        raise AssertionError("129-tile river was not rejected")
    with torch.inference_mode():
        out = model(features, masks)
        full_out = model(full_features.unsqueeze(0), masks[:1])
    legal_logits = out.logits[masks]
    if (not torch.isfinite(legal_logits).all() or not torch.isfinite(out.value).all()
            or not torch.isfinite(full_out.logits[masks[:1]]).all()
            or not torch.isfinite(full_out.value).all()):
        raise AssertionError("non-finite legal logits or values")
    if not torch.isneginf(out.logits[~masks]).all():
        raise AssertionError("illegal actions were not masked")
    print(json.dumps({
        "data_gate": "PASS", "parity": "PASS", "model_forward": "PASS",
        "feature_dim": feature_dim, "dense_dim": 397, "action_size": action_size,
        "sampled": per_split, "total_rows_checked": len(labels),
        "max_river_length_seen": max_seen,
        "boundary_128": "PASS", "overflow_129_rejected": "PASS",
    }, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
