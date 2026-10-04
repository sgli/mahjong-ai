"""PPO resume: Run A (continuous) ≈ Run B (save → resume) (§6)."""

from __future__ import annotations

import random
import sys

import pytest
import torch

from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder
from mahjong.model import MLPPolicy
from mahjong.training import train_ppo
from mahjong.training.ppo_checkpoint import (
    capture_ppo_rng_states,
    load_ppo_checkpoint,
    restore_ppo_rng_states,
    save_ppo_checkpoint,
)


def _train_cfg(**overrides):
    cfg = dict(
        num_envs=1,
        env_seed=0,
        epochs=2,
        steps_per_epoch=20,
        batch_size=16,
        update_epochs=2,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        device=torch.device("cpu"),
        max_episode_steps=100,
    )
    cfg.update(overrides)
    return cfg


def _new_model():
    return MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (64, 64))


@pytest.mark.parametrize("bitwise", [True])
def test_ppo_resume_matches_continuous(tmp_path, bitwise):
    encoder = ObservationEncoder()

    # ---- Run A: continuous 2 epochs ----
    torch.manual_seed(42)
    random.seed(42)
    model_a = _new_model()
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-4)
    hist_a = train_ppo(model_a, encoder, opt_a, **_train_cfg(epochs=2))

    # ---- Run B: 1 epoch → save → resume → 1 more epoch ----
    torch.manual_seed(42)
    random.seed(42)
    model_b = _new_model()
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-4)
    hist_b1 = train_ppo(model_b, encoder, opt_b, **_train_cfg(epochs=1))
    ckpt_path = save_ppo_checkpoint(
        tmp_path / "latest.pt",
        model_state_dict=model_b.state_dict(),
        optimizer_state_dict=opt_b.state_dict(),
        epoch=hist_b1[-1]["epoch"],
        global_step=hist_b1[-1]["global_step"],
        best_metric=float("inf"),
        best_metric_name="mean_reward",
        config={},
        metrics_history=hist_b1,
        seed=42,
        env_seed=0,
        feature_version="feature-v1",
        action_schema_version="action-v1",
        environment_version="tenhou-v2-rl",
        reward_version="reward-v1",
        model_version="ppo-v1",
        rng_states=capture_ppo_rng_states(),
    )

    # new "process": new model + optimizer, restore everything
    ckpt = load_ppo_checkpoint(ckpt_path, map_location="cpu")
    model_b2 = _new_model()
    model_b2.load_state_dict(ckpt["model_state_dict"])
    opt_b2 = torch.optim.Adam(model_b2.parameters(), lr=1e-4)
    opt_b2.load_state_dict(ckpt["optimizer_state_dict"])
    restore_ppo_rng_states(ckpt)
    hist_b2 = train_ppo(
        model_b2,
        encoder,
        opt_b2,
        **_train_cfg(epochs=2, start_epoch=ckpt["epoch"], global_step=ckpt["global_step"]),
    )

    # ---- compare ----
    hist_b_full = hist_b1 + hist_b2
    assert hist_a[-1]["epoch"] == hist_b_full[-1]["epoch"] == 2
    assert hist_a[-1]["global_step"] == hist_b_full[-1]["global_step"]
    assert len(hist_a) == len(hist_b_full) == 2

    # L1：直接比对 metrics（逐 epoch 的 train/val 数值指标）与 Run A 对应 epoch 一致
    for ea, eb in zip(hist_a, hist_b_full):
        assert ea["epoch"] == eb["epoch"]
        for key in ("policy_loss", "value_loss", "entropy", "mean_reward", "mean_return", "illegal_rate"):
            assert abs(ea[key] - eb[key]) < 1e-6, f"epoch {ea['epoch']} metric {key}: {ea[key]} != {eb[key]}"

    pa = list(model_a.parameters())
    pb = list(model_b2.parameters())
    max_diff = max((x - y).abs().max().item() for x, y in zip(pa, pb))
    if bitwise:
        # CPU deterministic path: env is seeded per kyoku, policy uses torch RNG (restored)
        assert all(torch.equal(x, y) for x, y in zip(pa, pb)), f"params differ (max diff {max_diff})"
    else:
        assert max_diff < 1e-5, f"params differ (max diff {max_diff})"
