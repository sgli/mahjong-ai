"""Tests for opponent pool + self-play (Phase 8)."""

import random

import torch

from mahjong.decision.action import Action
from mahjong.environment import MahjongEnv
from mahjong.evaluation import (
    OpponentPool,
    PolicyOpponent,
    RandomOpponent,
    run_game,
    run_round,
)
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM
from mahjong.model import MLPPolicy
from mahjong.training import save_checkpoint


def _save_small_policy(tmp_path):
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    save_checkpoint(
        tmp_path, model.state_dict(),
        config={"model": {"hidden_sizes": [32, 32]}}, metrics={},
        dataset_version="decision-v1", feature_version="feature-v1",
        action_schema_version="action-v1", training_step=0,
    )
    return tmp_path


def _legal_from_env(env):
    seat = env.state.turn
    return seat, env.legal_actions(seat), env.observation(seat)


def test_random_opponent_only_legal():
    env = MahjongEnv(seed=1)
    env.reset()
    rng = random.Random(0)
    opponent = RandomOpponent()
    for _ in range(200):
        seat, legal, obs = _legal_from_env(env)
        action = opponent.decide(obs, legal, rng)
        assert action in legal
        env.step(action)
        if env.done():
            env.reset()


def test_policy_opponent_mask_never_illegal(tmp_path):
    ckpt = _save_small_policy(tmp_path)
    opponent = PolicyOpponent(opponent_id="p", type="policy", model_version="v1", checkpoint=ckpt)
    env = MahjongEnv(seed=2)
    env.reset()
    rng = random.Random(0)
    illegal = 0
    for _ in range(200):
        seat, legal, obs = _legal_from_env(env)
        action = opponent.decide(obs, legal, rng)
        if action not in legal:
            illegal += 1
        env.step(action)
        if env.done():
            env.reset()
    assert illegal == 0


def test_opponent_pool_register_dedup_version():
    pool = OpponentPool()
    r1 = RandomOpponent(opponent_id="random", model_version="random-v1")
    r2 = RandomOpponent(opponent_id="random", model_version="random-v2")
    pool.add(r1)
    pool.add(r2)  # dedup: same id, latest wins
    assert len(pool) == 1
    assert pool.get("random").model_version == "random-v2"
    assert "random" in pool
    assert pool.list()[0].type == "random"


def test_selfplay_multi_opponent_round():
    env = MahjongEnv(seed=3)
    opponents = [RandomOpponent(opponent_id=f"r{s}", model_version="random-v1") for s in range(4)]
    results = run_round(opponents, games=3, seed=10)
    assert len(results) == 3
    for r in results:
        assert r.error is None
        assert sorted(r.final_ranks) == [0, 1, 2, 3]
        assert r.illegal_actions == 0
        assert set(r.opponents) == {0, 1, 2, 3}


def test_same_seed_reproducible():
    a = run_game([RandomOpponent() for _ in range(4)], seed=42)
    b = run_game([RandomOpponent() for _ in range(4)], seed=42)
    assert a.final_scores == b.final_scores
    assert a.final_ranks == b.final_ranks


def test_single_game_error_does_not_stop_round():
    class Broken(RandomOpponent):
        def decide(self, observation, legal_actions, rng):
            raise RuntimeError("boom")

    opponents = [Broken(), RandomOpponent(), RandomOpponent(), RandomOpponent()]
    results = run_round(opponents, games=2, seed=20)
    assert len(results) == 2
    assert any(r.error for r in results)  # errors recorded, round continued
