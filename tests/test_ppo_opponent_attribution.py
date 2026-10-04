"""Per-opponent / per-seat attribution（Phase 10.3 §4/§11）单元测试。"""

from __future__ import annotations

from mahjong.training.attribution import attribute_game
from mahjong.training.ppo_buffer import TrajectoryBuffer, TrajectoryStep


def _step(reward, value, done=False):
    return TrajectoryStep(features=None, action_id=0, legal_mask=None, old_log_prob=0.0, value=value, reward=reward, done=done)


def _game():
    traj = {s: TrajectoryBuffer() for s in range(4)}
    traj[0].append(_step(0.5, 0.1, False))
    traj[0].append(_step(-1.0, 0.2, True))
    traj[1].append(_step(2.0, 0.3, True))
    # seat 2,3 无轨迹
    return traj


def test_per_opponent_grouping_and_hand_checked_values():
    traj = _game()
    seat_types = ["rule-v1", "bc-v2.1", "rule-v1", "rule-v1"]
    ranks = [0, 1, 2, 3]
    per_opp, per_seat = attribute_game(traj, seat_types, ranks, gamma=0.99, lam=0.95)

    assert set(per_opp.keys()) == {"rule-v1", "bc-v2.1"}
    # episodes 之和 == 有轨迹的 seat 数
    assert sum(o["episodes"] for o in per_opp.values()) == 2

    rule = per_opp["rule-v1"]
    bc = per_opp["bc-v2.1"]
    # seat 0 的 episode reward = 0.5 + (-1.0) = -0.5
    assert abs(rule["episode_reward"] - (-0.5)) < 1e-9
    assert rule["episodes"] == 1
    assert rule["win"] == 1  # rank 0
    assert rule["rank_sum"] == 0
    # seat 1 reward = 2.0
    assert abs(bc["episode_reward"] - 2.0) < 1e-9
    assert bc["win"] == 0
    assert bc["rank_sum"] == 1


def test_per_seat_sample_count_sums():
    traj = _game()
    seat_types = ["rule-v1", "bc-v2.1", "rule-v1", "rule-v1"]
    ranks = [0, 1, 2, 3]
    _, per_seat = attribute_game(traj, seat_types, ranks, gamma=0.99, lam=0.95)
    total = sum(per_seat[s]["reward"]["n"] for s in per_seat)
    assert total == 3  # seat0 2 步 + seat1 1 步
    assert per_seat[0]["reward"]["n"] == 2
    assert per_seat[1]["reward"]["n"] == 1
    # return = advantage + value，至少 finite
    assert per_seat[0]["return"]["n"] == 2


def test_empty_game_returns_empty():
    traj = {s: TrajectoryBuffer() for s in range(4)}
    per_opp, per_seat = attribute_game(traj, ["self"] * 4, None, gamma=0.99, lam=0.95)
    assert per_opp == {}
    assert per_seat == {}
