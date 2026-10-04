"""PPO checkpoint save/load round-trip (§5/§25)."""

from __future__ import annotations

import random
import sys

import pytest
import torch

from mahjong.training.ppo_checkpoint import (
    CHECKPOINT_TYPE,
    CHECKPOINT_VERSION,
    capture_ppo_rng_states,
    load_ppo_checkpoint,
    restore_ppo_rng_states,
    save_ppo_checkpoint,
)


def _make_ckpt(tmp_path, optimizer_state=True):
    model = torch.nn.Linear(4, 3)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    # advance optimizer a bit
    model(torch.randn(2, 4)).sum().backward()
    opt.step()
    return save_ppo_checkpoint(
        tmp_path / "epoch_1.pt",
        model_state_dict=model.state_dict(),
        optimizer_state_dict=opt.state_dict() if optimizer_state else None,
        epoch=1,
        global_step=1234,
        best_metric=0.5,
        best_metric_name="mean_reward",
        config={"model": {"hidden_sizes": [256, 256]}},
        metrics_history=[{"epoch": 1, "mean_reward": 0.5}],
        seed=42,
        env_seed=7,
        feature_version="feature-v1",
        action_schema_version="action-v1",
        environment_version="tenhou-v2-rl",
        reward_version="reward-v1",
        model_version="ppo-v1",
        git_commit_hash="abc123",
    ), model, opt


def test_ppo_checkpoint_roundtrip(tmp_path):
    path, model, opt = _make_ckpt(tmp_path)
    ckpt = load_ppo_checkpoint(path, map_location="cpu")

    assert ckpt["checkpoint_type"] == CHECKPOINT_TYPE == "ppo"
    assert ckpt["checkpoint_version"] == CHECKPOINT_VERSION == "ppo-v1"
    assert ckpt["epoch"] == 1
    assert ckpt["global_step"] == 1234
    assert ckpt["best_metric"] == 0.5
    assert ckpt["best_metric_name"] == "mean_reward"
    assert ckpt["seed"] == 42
    assert ckpt["env_seed"] == 7
    assert ckpt["feature_version"] == "feature-v1"
    assert ckpt["action_schema_version"] == "action-v1"
    assert ckpt["environment_version"] == "tenhou-v2-rl"
    assert ckpt["reward_version"] == "reward-v1"
    assert ckpt["model_version"] == "ppo-v1"
    assert ckpt["git_commit"] == "abc123"

    # model state round-trip
    m2 = torch.nn.Linear(4, 3)
    m2.load_state_dict(ckpt["model_state_dict"])
    assert torch.equal(m2.weight, model.weight)
    assert torch.equal(m2.bias, model.bias)

    # optimizer state round-trip
    opt2 = torch.optim.Adam(m2.parameters(), lr=0.01)
    opt2.load_state_dict(ckpt["optimizer_state_dict"])
    assert opt2.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"]

    # metrics history
    assert ckpt["metrics_history"][0]["epoch"] == 1


def test_ppo_checkpoint_rejects_non_ppo(tmp_path):
    torch.save({"state_dict": {}}, tmp_path / "x.pt")
    with pytest.raises(ValueError):
        load_ppo_checkpoint(tmp_path / "x.pt")


def test_ppo_rng_restore_matches_sequences():
    """L2：python / torch CPU / torch CUDA 分别恢复后随机序列一致。"""
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    states = capture_ppo_rng_states()

    ref_py = [random.random() for _ in range(5)]
    ref_cpu = torch.rand(5)
    ref_cuda = torch.rand(5, device="cuda") if torch.cuda.is_available() else None

    restore_ppo_rng_states(states)

    assert [random.random() for _ in range(5)] == ref_py
    assert torch.equal(torch.rand(5), ref_cpu)
    if torch.cuda.is_available():
        assert torch.equal(torch.rand(5, device="cuda"), ref_cuda)
    else:
        assert states.get("torch_cuda_rng_state") is None  # 无 CUDA 时未捕获
