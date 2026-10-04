"""Benchmark 加载 PPO checkpoint（Phase 10.2 §19）：类型识别 + logits 一致性 + BC 回归。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mahjong.decision.action import Action
from mahjong.environment import MahjongEnv
from mahjong.evaluation import load_policy
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder
from mahjong.features.action_space import legal_mask
from mahjong.model import MLPPolicy
from mahjong.training import load_ppo_checkpoint

_PPO_CKPT = Path("experiments/ppo_v1/latest.pt")


@pytest.mark.skipif(not _PPO_CKPT.exists(), reason="no ppo checkpoint")
def test_load_policy_ppo_logits_bitwise():
    encoder, model = load_policy(_PPO_CKPT, "cpu")  # 自动识别 PPO
    ckpt = load_ppo_checkpoint(_PPO_CKPT, map_location="cpu")
    hidden = tuple(ckpt["config"]["model"]["hidden_sizes"])
    ref_model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, hidden)
    ref_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    ref_model.eval()

    env = MahjongEnv(seed=0)
    env.reset()
    obs = env.observation(env.state.turn)
    legal = [a for a in env.legal_actions(env.state.turn) if isinstance(a, Action)]
    feats = encoder.encode(obs)
    mask = legal_mask(legal)
    with torch.no_grad():
        a = model(feats.unsqueeze(0), mask.unsqueeze(0)).logits
        b = ref_model(feats.unsqueeze(0), mask.unsqueeze(0)).logits
    assert torch.equal(a, b)


@pytest.mark.skipif(not Path("experiments/bc_v2/checkpoints/model.pt").exists(), reason="no bc checkpoint")
def test_load_policy_bc_regression():
    encoder, model = load_policy("experiments/bc_v2/checkpoints", "cpu")  # BC 路径不变
    assert model is not None
    env = MahjongEnv(seed=0)
    env.reset()
    obs = env.observation(env.state.turn)
    feats = encoder.encode(obs)
    legal = [a for a in env.legal_actions(env.state.turn) if isinstance(a, Action)]
    mask = legal_mask(legal)
    with torch.no_grad():
        out = model(feats.unsqueeze(0), mask.unsqueeze(0))
    assert out.logits.shape == (1, ACTION_SPACE_SIZE)
