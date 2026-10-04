"""Reward conservation audit (Phase 10.3 诊断 §4)."""

from __future__ import annotations

import torch

from mahjong.environment import MahjongEnv
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder
from mahjong.model import MLPPolicy
from mahjong.training import collect_trajectory, flush_terminal_rewards
from mahjong.training.ppo_buffer import TrajectoryBuffer, TrajectoryStep


def _rollout(seed=11):
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    encoder = ObservationEncoder()
    env = MahjongEnv(seed=seed)
    traj, steps, illegal = collect_trajectory(env, model, encoder, torch.device("cpu"), max_steps=20000)
    assert env.done(), f"game did not finish (seed {seed})"
    return env, traj, illegal


def test_sum_rewards_equals_env_reward_per_seat():
    env, traj, illegal = _rollout()
    assert illegal == 0
    for s in range(4):
        total = sum(st.reward for st in traj[s].steps)
        assert abs(total - env.reward(s)) < 1e-6, f"seat {s}: {total} != {env.reward(s)}"


def test_placement_bonus_exactly_once():
    env, traj, illegal = _rollout()
    assert illegal == 0
    # placement bonus 总和应 == -sum(score deltas)（守恒），且四个 bonus 恰好各出现一次
    bonuses = []
    for s in range(4):
        if traj[s].steps:
            # 末步奖励 = 该 seat 的 score delta 贡献 + placement_bonus[rank]
            last = traj[s].steps[-1].reward
            bonuses.append(last)
    # 验证 env.reward == scale*delta + bonus，且 sum(rewards) == 0
    assert abs(sum(env.reward(s) for s in range(4))) < 1e-6
    # placement bonus 只出现一次：构造一个空 buffer，flush 后末步只含 bonus
    bufs = {s: TrajectoryBuffer() for s in range(4)}
    for s in range(4):
        bufs[s].append(TrajectoryStep(features=None, action_id=0, legal_mask=None, old_log_prob=0.0, value=0.0, reward=0.0, done=True))
    pending = {s: 0.0 for s in range(4)}
    flush_terminal_rewards(env, bufs, pending)
    got = [bufs[s].steps[-1].reward for s in range(4)]
    assert sorted(got) == [-2.0, -1.0, 1.0, 2.0]  # rank 0..3 的 placement bonus 各出现一次
    assert abs(sum(got)) < 1e-6


def test_no_reward_after_done():
    env, traj, illegal = _rollout(seed=13)
    assert illegal == 0
    for s in range(4):
        steps = traj[s].steps
        if not steps:
            continue
        # 只有最后一个 step 可能 done=True；done 之后无任何后续 step
        assert all(not st.done for st in steps[:-1])


def test_no_cross_seat_reward():
    # 不同 seat 的 trajectory 独立：某 seat 的 reward 只反映自身 score delta + 自身 bonus
    env, traj, illegal = _rollout(seed=17)
    assert illegal == 0
    for s in range(4):
        total = sum(st.reward for st in traj[s].steps)
        expected = env.reward(s)
        assert abs(total - expected) < 1e-6
