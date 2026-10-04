"""PPO checkpoint v2 extended fields (§22) round-trip + backward compat."""

from __future__ import annotations

import torch

from mahjong.training.ppo_checkpoint import load_ppo_checkpoint, save_ppo_checkpoint


def _base_kwargs():
    m = torch.nn.Linear(4, 3)
    return dict(
        model_state_dict=m.state_dict(),
        optimizer_state_dict=None,
        epoch=1,
        global_step=1000,
        best_metric=0.5,
        best_metric_name="mean_reward",
        config={"model": {"hidden_sizes": [256, 256]}},
        metrics_history=[{"epoch": 1}],
        seed=42,
        env_seed=7,
        feature_version="feature-v1",
        action_schema_version="action-v1",
        environment_version="tenhou-v2-rl",
        reward_version="reward-v1",
        model_version="ppo-v2",
    )


def test_ppo_checkpoint_v2_new_fields_roundtrip(tmp_path):
    path = save_ppo_checkpoint(
        tmp_path / "ckpt.pt",
        **_base_kwargs(),
        opponent_pool_version="pool-v1",
        opponent_pool_config={"random-v1": 0.1, "rule-v1": 0.2},
        rollout_version="rollout-v1",
        training_stage="pilot",
    )
    ckpt = load_ppo_checkpoint(path)
    assert ckpt["opponent_pool_version"] == "pool-v1"
    assert ckpt["opponent_pool_config"] == {"random-v1": 0.1, "rule-v1": 0.2}
    assert ckpt["rollout_version"] == "rollout-v1"
    assert ckpt["training_stage"] == "pilot"


def test_ppo_checkpoint_v2_backward_compat(tmp_path):
    # 旧 checkpoint 缺新字段 → load 后补 None，不崩
    path = save_ppo_checkpoint(tmp_path / "old.pt", **_base_kwargs())
    ckpt = load_ppo_checkpoint(path)
    assert ckpt["opponent_pool_version"] is None
    assert ckpt["opponent_pool_config"] is None
    assert ckpt["rollout_version"] is None
    assert ckpt["training_stage"] is None
