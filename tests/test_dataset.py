"""Tests for the Phase 4 Decision Dataset (schema / split / builder / manifest)."""

import pyarrow as pa
import pyarrow.parquet as pq

from mahjong.dataset import (
    DATASET_VERSION,
    DatasetBuilder,
    assign_splits,
    parquet_schema,
    row_to_sample,
    sample_to_row,
)
from mahjong.decision import Action, ActionType, DecisionSample, PlayerObservation
from mahjong.decision.decision import DecisionPoint


def _sample(game_id, player_id, hand=("1m", "2m", "3m"), action_tile="1m"):
    obs = PlayerObservation(
        seat=player_id,
        bakaze="E",
        kyoku=1,
        honba=0,
        kyotaku=0,
        oya=0,
        scores=(25000, 25000, 25000, 25000),
        riichi=(False, False, False, False),
        hand=hand,
        melds=(),
        opponents_melds=((), (), (), ()),
        discards=((), (), (), ()),
        dora_markers=("2m",),
        turn=player_id,
    )
    action = Action(ActionType.DISCARD, tile=action_tile)
    legal = (action, Action(ActionType.PASS))
    point = DecisionPoint(round_id="E1", seat=player_id, kind="discard", step=0)
    return DecisionSample(
        game_id=game_id,
        round_id="E1",
        player_id=player_id,
        observation=obs,
        legal_actions=legal,
        action=action,
        metadata={"kind": "discard"},
        decision_point=point,
    )


# -- schema / round-trip --------------------------------------------------------
def test_sample_row_roundtrip(tmp_path):
    sample = _sample("g1", 0, hand=("1m", "2m", "3m", "5mr"))
    row = sample_to_row(sample)
    table = pa.Table.from_pylist([row], schema=parquet_schema())
    path = tmp_path / "roundtrip.parquet"
    pq.write_table(table, path)
    read = pq.read_table(path).to_pylist()[0]
    assert row_to_sample(read) == sample


def test_schema_has_no_hidden_observation_fields():
    names = parquet_schema().names
    assert "hand" in names  # own hand is visible
    assert not any("opponent" in n and "hand" in n for n in names)
    assert "ura_markers" not in names
    assert "wall" not in names


# -- split ----------------------------------------------------------------------
def test_assign_splits_deterministic_and_disjoint():
    ids = [f"g{i}" for i in range(20)]
    a1 = assign_splits(ids, seed=7)
    a2 = assign_splits(ids, seed=7)
    assert a1 == a2
    assert set(a1.keys()) == set(ids)
    assert set(a1.values()) <= {"train", "validation", "test"}


def test_assign_splits_rejects_bad_ratios():
    import pytest

    with pytest.raises(ValueError):
        assign_splits(["g1"], ratios=(0.5, 0.5, 0.5), seed=0)


# -- builder --------------------------------------------------------------------
def test_split_does_not_cross_game(tmp_path):
    builder = DatasetBuilder(output_dir=tmp_path, chunk_size=1000, ratios=(0.5, 0.3, 0.2), seed=42)
    game_ids = [f"g{i}" for i in range(10)]
    builder.assign_splits(game_ids)
    for gid in game_ids:
        builder.add_game_samples(gid, [_sample(gid, s) for s in range(4)])
    manifest = builder.finish()

    seen = {}
    for split in ("train", "validation", "test"):
        for f in (tmp_path / split).glob("*.parquet"):
            for row in pq.read_table(f).to_pylist():
                gid = row["game_id"]
                if gid in seen:
                    assert seen[gid] == split, f"{gid} leaked across splits"
                else:
                    seen[gid] = split

    assert len(seen) == 10
    total = sum(manifest["counts"]["splits"][s]["samples"] for s in ("train", "validation", "test"))
    assert total == 40


def test_chunking_produces_multiple_files(tmp_path):
    builder = DatasetBuilder(output_dir=tmp_path, chunk_size=3, ratios=(1.0, 0.0, 0.0), seed=0)
    builder.assign_splits(["g1"])
    builder.add_game_samples("g1", [_sample("g1", 0) for _ in range(10)])
    builder.finish()

    train_files = list((tmp_path / "train").glob("*.parquet"))
    assert len(train_files) >= 4
    total_rows = sum(pq.read_table(f).num_rows for f in train_files)
    assert total_rows == 10


def test_manifest_content(tmp_path):
    builder = DatasetBuilder(output_dir=tmp_path, chunk_size=100, seed=0)
    builder.assign_splits(["g1", "g2", "g3"])
    for gid in ("g1", "g2", "g3"):
        builder.add_game_samples(gid, [_sample(gid, 0)])
    manifest = builder.finish()

    assert manifest["dataset_version"] == DATASET_VERSION
    assert manifest["action_schema_version"] == "action-v1"
    assert manifest["feature_version"] == "none"
    assert manifest["counts"]["games"] == 3
    assert manifest["counts"]["samples"] == 3
    assert manifest["counts"]["samples_skipped"] == 0
    assert (tmp_path / "manifest.json").exists()


def test_builder_skips_invalid_sample(tmp_path):
    sample = _sample("g1", 0)
    # force an illegal human action (not in legal_actions)
    bad = DecisionSample(
        game_id=sample.game_id,
        round_id=sample.round_id,
        player_id=sample.player_id,
        observation=sample.observation,
        legal_actions=sample.legal_actions,
        action=Action(ActionType.PON, tile="9p", consumed=("9p", "9p"), target=0),
        metadata=sample.metadata,
        decision_point=sample.decision_point,
    )
    builder = DatasetBuilder(output_dir=tmp_path, chunk_size=100, seed=0)
    builder.assign_splits(["g1"])
    builder.add_game_samples("g1", [bad])
    manifest = builder.finish()
    assert manifest["counts"]["samples_skipped"] == 1
    assert manifest["counts"]["samples"] == 0
