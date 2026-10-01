"""Checkpoint save/load (MODEL_SPEC 11, TRAINING_SPEC 5)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml


def save_checkpoint(
    checkpoint_dir: str | Path,
    model_state_dict: dict,
    *,
    config: dict,
    metrics: dict,
    dataset_version: str,
    feature_version: str,
    action_schema_version: str,
    environment_version: str = "none",
    git_commit_hash: str | None = None,
    training_step: int = 0,
    rules: str | None = None,
) -> Path:
    """Write ``model.pt`` plus the version/config/metrics sidecar files."""
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)

    model_dict = {
        "state_dict": model_state_dict,
        "config": config,
        "metrics": metrics,
        "training_step": training_step,
    }
    if rules is not None:
        model_dict["rules"] = rules
    torch.save(model_dict, directory / "model.pt")
    with (directory / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    with (directory / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    (directory / "dataset_version.txt").write_text(dataset_version, encoding="utf-8")
    (directory / "feature_version.txt").write_text(feature_version, encoding="utf-8")
    (directory / "action_schema_version.txt").write_text(action_schema_version, encoding="utf-8")
    (directory / "environment_version.txt").write_text(environment_version, encoding="utf-8")
    if rules is not None:
        (directory / "rules.txt").write_text(rules, encoding="utf-8")
    if git_commit_hash:
        (directory / "git_commit.txt").write_text(git_commit_hash, encoding="utf-8")
    return directory


def load_checkpoint(checkpoint_dir: str | Path, map_location: str = "cpu") -> dict:
    """Load the ``model.pt`` dict (state_dict / config / metrics / step)."""
    return torch.load(Path(checkpoint_dir) / "model.pt", map_location=map_location)
