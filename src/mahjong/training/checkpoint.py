"""Checkpoint save/load (MODEL_SPEC 11, TRAINING_SPEC 5)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import yaml

_TRAINING_STATE_KEYS = (
    "state_dict",
    "optimizer_state_dict",
    "epoch",
    "global_step",
    "best_val_loss",
    "best_val_accuracy",
    "config",
    "metrics",
    "rng_state",
    "rng_state_torch_cpu",
    "rng_state_torch_cuda",
    "scaler_state",
)


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


def capture_rng_states() -> dict:
    """Capture Python + Torch CPU/CUDA RNG states for deterministic resume."""
    states = {
        "rng_state": random.getstate(),
        "rng_state_torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["rng_state_torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: dict) -> None:
    """Restore RNG states captured by :func:`capture_rng_states` (best effort)."""
    if "rng_state" in states and states["rng_state"] is not None:
        random.setstate(states["rng_state"])
    if "rng_state_torch_cpu" in states and states["rng_state_torch_cpu"] is not None:
        cpu_state = states["rng_state_torch_cpu"]
        if hasattr(cpu_state, "cpu"):
            cpu_state = cpu_state.cpu()
        torch.set_rng_state(cpu_state)
    if (
        torch.cuda.is_available()
        and "rng_state_torch_cuda" in states
        and states["rng_state_torch_cuda"] is not None
    ):
        cuda_states = states["rng_state_torch_cuda"]
        try:
            torch.cuda.set_rng_state_all(cuda_states)
        except (TypeError, RuntimeError):
            # 从 CPU 加载时张量在 CPU，搬回 CUDA 再设置
            torch.cuda.set_rng_state_all([s.cuda() if hasattr(s, "cuda") else s for s in cuda_states])


def save_training_checkpoint(
    path: str | Path,
    *,
    model_state_dict: dict,
    optimizer_state_dict: dict | None,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    best_val_accuracy: float,
    config: dict,
    metrics: dict,
    scaler_state: dict | None = None,
    rng_states: dict | None = None,
) -> Path:
    """Save the full training state (resume-capable)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "config": config,
        "metrics": metrics,
        "rng_state": (rng_states or {}).get("rng_state"),
        "rng_state_torch_cpu": (rng_states or {}).get("rng_state_torch_cpu"),
        "rng_state_torch_cuda": (rng_states or {}).get("rng_state_torch_cuda"),
        "scaler_state": scaler_state,
    }
    torch.save(payload, p)
    return p


def load_training_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    """Load a training checkpoint; missing (legacy) fields default to ``None``."""
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError(f"training checkpoint must be a dict, got {type(payload)}")
    # backward-compat: legacy checkpoints only had state_dict/config/metrics/training_step
    if "epoch" not in payload and "training_step" in payload:
        payload.setdefault("epoch", 0)
        payload.setdefault("global_step", payload["training_step"])
    for key in _TRAINING_STATE_KEYS:
        payload.setdefault(key, None)
    return payload
