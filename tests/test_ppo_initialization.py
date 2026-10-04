"""BC → PPO initialization must keep policy logits bitwise identical (§4)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from mahjong.decision.action import Action
from mahjong.environment import MahjongEnv
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder
from mahjong.features.action_space import action_to_id, legal_mask
from mahjong.model import MLPPolicy, remap_bc_state_dict
from mahjong.training import init_ppo_from_bc, load_training_checkpoint

_BC_BEST = Path("experiments/bc_v2.1/checkpoints/best.pt")


@pytest.mark.skipif(not _BC_BEST.exists(), reason="bc_v2.1 best.pt not available")
def test_bc_ppo_logits_bitwise_equal():
    ckpt = load_training_checkpoint(_BC_BEST, map_location="cpu")
    state = ckpt.get("state_dict") or {}
    cfg = ckpt.get("config") or {}
    hidden = tuple((cfg.get("model") or {}).get("hidden_sizes", [256, 256]))

    # BC model（完整加载 trunk + action head + value head）
    bc_model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, hidden)
    bc_model.load_state_dict(remap_bc_state_dict(state), strict=False)
    bc_model.eval()

    # PPO model（BC trunk + action head 保留，value head 重初始化）
    ppo_model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, hidden)
    init_ppo_from_bc(ppo_model, state)
    ppo_model.eval()

    # 用真实环境构造多个 observation + legal mask 做抽样比对
    encoder = ObservationEncoder()
    env = MahjongEnv(seed=0)
    env.reset()
    checked = 0
    for _ in range(20):
        if env.done():
            env.reset()
        seat = env.state.turn
        legal = [a for a in env.legal_actions(seat) if isinstance(a, Action)]
        obs = env.observation(seat)
        features = encoder.encode(obs)
        mask = legal_mask(legal)
        with torch.no_grad():
            bc_logits = bc_model(features.unsqueeze(0), mask.unsqueeze(0)).logits
            ppo_logits = ppo_model(features.unsqueeze(0), mask.unsqueeze(0)).logits
        assert torch.equal(bc_logits, ppo_logits), f"logits differ at sample {checked}"
        checked += 1
        # advance a step deterministically (take the first legal action)
        action_id = int(torch.nonzero(mask)[0].item())
        action = next(a for a in legal if action_to_id(a) == action_id)
        env.step(action)
    assert checked >= 10
