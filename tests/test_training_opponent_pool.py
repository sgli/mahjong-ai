"""Opponent pool 接入训练（Phase 10.3 §12/§13）单元测试。"""

from __future__ import annotations

import torch

from mahjong.environment import MahjongEnv
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder
from mahjong.model import MLPPolicy
from mahjong.training import TrainingOpponentPool, build_opponent, collect_trajectory_with_opponents, register_historical_checkpoints, sample_seats, train_ppo
from mahjong.training.attribution import attribute_game
from mahjong.training.ppo import opponent_pool_game_index


def _pool(seed=42):
    probs = {"random-v1": 0.5, "rule-v1": 0.5}
    return TrainingOpponentPool({}, probs, seed=seed)


def test_sample_seats_rotation_and_reproducible():
    pool = _pool()
    seats = []
    for gi in range(8):
        ls, types = sample_seats(pool, gi, learner_type="self")
        seats.append(ls)
        assert types[ls] == "self"
        assert len(types) == 4
    # 轮换覆盖 4 个座位
    assert set(seats) == {0, 1, 2, 3}
    # 同 seed 可复现
    pool2 = _pool()
    assert sample_seats(pool2, 3) == sample_seats(_pool(), 3)


def test_build_opponent_random_rule():
    r = build_opponent("random-v1", bc_checkpoint="")
    ru = build_opponent("rule-v1", bc_checkpoint="")
    assert r.opponent_id == "random"
    assert ru.opponent_id == "rule"


def test_collect_trajectory_with_opponents_runs():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (16, 16))
    model.eval()
    encoder = ObservationEncoder()
    env = MahjongEnv(seed=1)
    learner_seat = 1
    opponents = {0: build_opponent("random-v1", bc_checkpoint=""), 2: build_opponent("rule-v1", bc_checkpoint=""), 3: build_opponent("rule-v1", bc_checkpoint="")}
    traj, steps, illegal = collect_trajectory_with_opponents(env, model, encoder, torch.device("cpu"), learner_seat=learner_seat, opponents=opponents, max_steps=300, greedy=True, seed=7)
    assert illegal == 0
    assert steps > 0
    # learner 有轨迹
    assert traj[learner_seat].steps


def test_collect_trajectory_with_opponents_caps_learner_transitions_and_bootstraps():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (16, 16))
    model.eval()
    encoder = ObservationEncoder()
    env = MahjongEnv(seed=1)
    learner_seat = 1
    opponents = {
        0: build_opponent("random-v1", bc_checkpoint=""),
        2: build_opponent("rule-v1", bc_checkpoint=""),
        3: build_opponent("rule-v1", bc_checkpoint=""),
    }
    traj, env_steps, illegal, bootstrap, truncated = collect_trajectory_with_opponents(
        env, model, encoder, torch.device("cpu"), learner_seat=learner_seat,
        opponents=opponents, max_steps=500, max_learner_steps=4, greedy=True, seed=7,
    )
    assert len(traj[learner_seat].steps) == 4
    assert env_steps >= 4
    assert illegal == 0
    assert truncated
    assert traj[learner_seat].steps[-1].done is False
    assert torch.isfinite(torch.tensor(bootstrap))


def test_attribution_multi_opponent_types():
    """opponent pool 启用后 per_opponent 出现多种对手类型。"""
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (16, 16))
    model.eval()
    encoder = ObservationEncoder()
    env = MahjongEnv(seed=2)
    learner_seat = 0
    opponents = {1: build_opponent("random-v1", bc_checkpoint=""), 2: build_opponent("rule-v1", bc_checkpoint=""), 3: build_opponent("rule-v1", bc_checkpoint="")}
    traj, steps, illegal = collect_trajectory_with_opponents(env, model, encoder, torch.device("cpu"), learner_seat=learner_seat, opponents=opponents, max_steps=500, greedy=True, seed=3)
    assert illegal == 0
    seat_types = ["self", "random-v1", "rule-v1", "rule-v1"]
    ranks = env.state.final_ranks if env.done() else [0, 1, 2, 3]
    per_opp, _ = attribute_game(traj, seat_types, ranks, gamma=0.99, lam=0.95)
    assert set(per_opp.keys()) == {"self", "random-v1", "rule-v1"}
    assert sum(o["episodes"] for o in per_opp.values()) == 4  # 四个 seat 各一条轨迹


def test_train_ppo_with_pool_runs():
    """train_ppo 启用 opponent_pool 后能正常跑（finite、illegal 0）。"""
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (16, 16))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    encoder = ObservationEncoder()
    pool = TrainingOpponentPool({}, {"random-v1": 0.5, "rule-v1": 0.5}, seed=0)
    hist = train_ppo(
        model, encoder, opt, num_envs=1, env_seed=0, epochs=1, steps_per_epoch=100, batch_size=32,
        update_epochs=1, gamma=0.99, lam=0.95, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
        device=torch.device("cpu"), max_episode_steps=200,
        opponent_pool=pool, opponent_bc_checkpoint="", opponent_pool_seed=0,
    )
    m = hist[0]
    assert m["illegal_rate"] == 0.0
    assert m["steps"] > 0
    assert all(m[k] == m[k] for k in m if isinstance(m[k], float))


def test_train_ppo_learner_transition_budget_is_exact():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (16, 16))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    encoder = ObservationEncoder()
    pool = TrainingOpponentPool({}, {"random-v1": 0.5, "rule-v1": 0.5}, seed=0)
    hist = train_ppo(
        model, encoder, opt, num_envs=1, env_seed=0, epochs=1, steps_per_epoch=7, batch_size=8,
        update_epochs=1, gamma=0.99, lam=0.95, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
        device=torch.device("cpu"), max_episode_steps=500,
        opponent_pool=pool, opponent_bc_checkpoint="", opponent_pool_seed=0,
        step_budget_unit="learner_transitions",
    )
    metrics = hist[0]
    assert metrics["step_budget_unit"] == "learner_transitions"
    assert metrics["target_steps"] == 7
    assert metrics["learner_transitions"] == 7
    assert metrics["rollout_buffer_samples"] == 7


def test_build_ppo_v1_opponent():
    from pathlib import Path

    if not Path("experiments/ppo_v1/latest.pt").exists():
        import pytest

        pytest.skip("no ppo_v1 checkpoint")
    opp = build_opponent("ppo-v1", bc_checkpoint="", checkpoint_map={"ppo-v1": "experiments/ppo_v1/latest.pt"})
    assert opp.opponent_id == "ppo_v1"
    assert opp.model_version == "ppo-v1"


def test_build_current_ppo_opponent():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (16, 16))
    model.eval()
    encoder = ObservationEncoder()
    opp = build_opponent("current-ppo", bc_checkpoint="", current_ppo=(model, encoder))
    assert opp.opponent_id == "current-ppo"


def test_unavailable_opponent_raises():
    import pytest

    with pytest.raises(ValueError):
        build_opponent("ppo-v1", bc_checkpoint="", checkpoint_map={})
    with pytest.raises(ValueError):
        build_opponent("historical:epoch_20k", bc_checkpoint="", checkpoint_map={})
    with pytest.raises(ValueError):
        build_opponent("current-ppo", bc_checkpoint="", current_ppo=None)


def test_historical_registration_to_build_chain():
    """register_historical_checkpoints → checkpoint_map → build_opponent 链路。"""
    from pathlib import Path

    import pytest as _pytest

    reg = register_historical_checkpoints(Path("experiments/ppo_v2/milestones/ppo-20k"))
    if not reg:
        _pytest.skip("no historical checkpoints")
    key = next(iter(reg))
    opp = build_opponent(key, bc_checkpoint="", checkpoint_map=reg)
    assert opp.opponent_id == key


def test_historical_milestone_directory_registers_its_latest_file(tmp_path):
    milestone = tmp_path / "ppo-20k"
    milestone.mkdir()
    checkpoint = milestone / "latest.pt"
    checkpoint.write_bytes(b"checkpoint")
    registered = register_historical_checkpoints(milestone)
    assert registered == {"historical:ppo-20k": str(checkpoint)}


def test_pool_assignment_is_stateless_for_a_game_index():
    pool = TrainingOpponentPool({}, {"random-v1": 1.0, "rule-v1": 1.0}, seed=123)
    first = sample_seats(pool, (1 << 32) | 17)
    second = sample_seats(pool, (1 << 32) | 17)
    assert first == second


def test_opponent_pool_game_seed_namespace_does_not_repeat_across_epochs():
    indices = [opponent_pool_game_index(epoch, game) for epoch in range(4) for game in range(10)]
    assert len(indices) == len(set(indices))
    assert opponent_pool_game_index(2, 7) == opponent_pool_game_index(2, 7)


