"""reward score/placement scale 覆盖（F 实验）。"""

from __future__ import annotations

from mahjong.environment import MahjongEnv


def test_default_reward_scale_unchanged():
    env = MahjongEnv(seed=0)
    assert env.reward_config.score_delta_scale == 0.001
    assert env.reward_config.placement_bonus == (2.0, 1.0, -1.0, -2.0)


def test_override_score_scale():
    env = MahjongEnv(seed=0)
    env.reward_config.score_delta_scale = 0.0  # F1 纯顺位
    assert env.reward_config.score_delta_scale == 0.0
    env.reward_config.score_delta_scale = 0.0002  # F2
    assert env.reward_config.score_delta_scale == 0.0002
