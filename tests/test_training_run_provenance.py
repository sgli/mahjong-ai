"""Resolved PPO configuration and resume compatibility guards."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import train_ppo  # noqa: E402


def _config():
    return {
        "model": {"hidden_sizes": [256, 256]},
        "training": {"epochs": 1, "seed": 42, "lr": 2e-5},
        "feature_version": "feature-v1",
        "action_schema_version": "action-v1",
        "environment_version": "tenhou-v2-rl",
        "reward_version": "reward-v1",
        "opponent_pool_config": {"bc-v2.1": 1.0},
        "opponent_checkpoints_resolved": {},
        "provenance": {"bc_checkpoint_sha256": "bc-hash", "bc_anchor_checkpoint_sha256": "anchor-hash", "opponent_checkpoint_sha256": {}, "git": {"commit": "abc", "dirty": False}},
    }


def test_resume_signature_allows_only_target_epoch_extension():
    original = _config()
    extended = _config()
    extended["training"]["epochs"] = 2
    assert train_ppo._resume_signature(original) == train_ppo._resume_signature(extended)
    extended["training"]["lr"] = 1e-5
    assert train_ppo._resume_signature(original) != train_ppo._resume_signature(extended)


def test_resume_signature_tracks_checkpoint_and_code_provenance():
    original = _config()
    changed = _config()
    changed["provenance"]["bc_checkpoint_sha256"] = "different-bc-hash"
    assert train_ppo._resume_signature(original) != train_ppo._resume_signature(changed)
