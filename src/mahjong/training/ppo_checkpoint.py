"""PPO checkpoint (Phase 10.2 §5/§25): full training state, distinct from BC.

Field list::

    checkpoint_type / checkpoint_version
    model_state_dict / optimizer_state_dict / scheduler_state_dict?
    epoch / global_step / best_metric / best_metric_name
    config / metrics_history
    seed / env_seed
    python_rng_state / torch_cpu_rng_state / torch_cuda_rng_state
    feature_version / action_schema_version / environment_version
    reward_version / model_version
    git_commit
    scaler_state? (AMP)
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

CHECKPOINT_TYPE = "ppo"
CHECKPOINT_VERSION = "ppo-v1"


def capture_ppo_rng_states() -> dict:
    states = {
        "python_rng_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return states


def restore_ppo_rng_states(states: dict) -> None:
    if states.get("python_rng_state") is not None:
        random.setstate(states["python_rng_state"])
    cpu = states.get("torch_cpu_rng_state")
    if cpu is not None:
        torch.set_rng_state(cpu.cpu() if hasattr(cpu, "cpu") else cpu)
    cuda = states.get("torch_cuda_rng_state")
    if torch.cuda.is_available() and cuda is not None:
        try:
            torch.cuda.set_rng_state_all(cuda)
        except (TypeError, RuntimeError):
            torch.cuda.set_rng_state_all([s.cuda() if hasattr(s, "cuda") else s for s in cuda])


def save_ppo_checkpoint(
    path: str | Path,
    *,
    model_state_dict: dict,
    optimizer_state_dict: dict | None,
    epoch: int,
    global_step: int,
    best_metric: float,
    best_metric_name: str,
    config: dict,
    metrics_history: list[dict],
    seed: int,
    env_seed: int,
    feature_version: str,
    action_schema_version: str,
    environment_version: str,
    reward_version: str,
    model_version: str,
    git_commit_hash: str | None = None,
    scaler_state: dict | None = None,
    scheduler_state_dict: dict | None = None,
    rng_states: dict | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rng = rng_states or capture_ppo_rng_states()
    payload = {
        "checkpoint_type": CHECKPOINT_TYPE,
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
        "config": config,
        "metrics_history": metrics_history,
        "seed": seed,
        "env_seed": env_seed,
        "python_rng_state": rng.get("python_rng_state"),
        "torch_cpu_rng_state": rng.get("torch_cpu_rng_state"),
        "torch_cuda_rng_state": rng.get("torch_cuda_rng_state"),
        "feature_version": feature_version,
        "action_schema_version": action_schema_version,
        "environment_version": environment_version,
        "reward_version": reward_version,
        "model_version": model_version,
        "git_commit": git_commit_hash,
        "scaler_state": scaler_state,
    }
    torch.save(payload, p)
    return p


def load_ppo_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError(f"PPO checkpoint must be a dict, got {type(payload)}")
    if payload.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError(f"not a PPO checkpoint: checkpoint_type={payload.get('checkpoint_type')!r}")
    return payload


__all__ = [
    "CHECKPOINT_TYPE",
    "CHECKPOINT_VERSION",
    "capture_ppo_rng_states",
    "load_ppo_checkpoint",
    "restore_ppo_rng_states",
    "save_ppo_checkpoint",
]
