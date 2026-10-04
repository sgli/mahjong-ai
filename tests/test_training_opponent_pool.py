"""Opponent pool 接入训练（Phase 10.3 §12/§13）单元测试。"""

from __future__ import annotations

import torch

from mahjong.environment import MahjongEnv
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder
from mahjong.model import MLPPolicy
from mahjong.training import TrainingOpponentPool, build_opponent, collect_trajectory_with_opponents, register_historical_checkpoints, sample_seats, train_ppo
from mahjong.training.attribution import attribute_game


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


def test_ppo_fix_configs_no_opponent_pool():
    """§10 的 9 个实验配置应为纯自对弈（无 opponent_pool_config），否则在新接线会报错。"""
    import yaml
    from pathlib import Path

    d = Path("configs/experiments/ppo_fix")
    for name in ("b", "c", "d1", "d2", "d3", "d4", "d5", "d6", "e1"):
        cfg = yaml.safe_load((d / f"{name}.yaml").read_text(encoding="utf-8"))
        assert "opponent_pool_config" not in cfg, f"{name}.yaml 不应含 opponent_pool_config"


def test_v2_config_pool_assembles_no_missing_checkpoint():
    """configs/train_ppo_v2.yaml 的五类对手 checkpoint_map 可完整组装（缺失则显式失败）。"""
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(Path("configs/train_ppo_v2.yaml").read_text(encoding="utf-8"))
    pool_config = cfg["opponent_pool_config"]
    opp_ckpts = cfg.get("opponent_checkpoints") or {}
    assert "ppo-v1" in opp_ckpts
    assert Path(opp_ckpts["ppo-v1"]).exists()
    checkpoint_map = {"ppo-v1": opp_ckpts["ppo-v1"]}
    for entry in opp_ckpts.get("historical") or []:
        p = Path(entry)
        assert p.is_dir(), f"historical dir missing: {entry}"
        checkpoint_map.update(register_historical_checkpoints(p))
    for ot in pool_config:
        if ot == "ppo-v1":
            assert "ppo-v1" in checkpoint_map
        elif ot == "historical":
            assert any(k.startswith("historical:") for k in checkpoint_map)
