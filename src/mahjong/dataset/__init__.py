"""Decision Dataset (Phase 4): versioned, chunked, game-split Parquet dataset."""

from .builder import DatasetBuilder
from .manifest import build_manifest, git_commit, write_manifest
from .schema import (
    COLUMN_DOCS,
    DATASET_VERSION,
    FEATURE_VERSION,
    action_from_dict,
    action_to_dict,
    meld_from_dict,
    meld_to_dict,
    observation_to_columns,
    parquet_schema,
    row_to_sample,
    sample_to_row,
)
from .split import SPLIT_NAMES, assign_splits

__all__ = [
    "COLUMN_DOCS",
    "DATASET_VERSION",
    "FEATURE_VERSION",
    "DatasetBuilder",
    "SPLIT_NAMES",
    "action_from_dict",
    "action_to_dict",
    "assign_splits",
    "build_manifest",
    "git_commit",
    "meld_from_dict",
    "meld_to_dict",
    "observation_to_columns",
    "parquet_schema",
    "row_to_sample",
    "sample_to_row",
    "write_manifest",
]
