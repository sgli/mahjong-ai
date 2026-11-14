"""Versioned PPO entry checks at actual environment decision boundaries."""

from __future__ import annotations

import random
import sys
from types import SimpleNamespace
from pathlib import Path

import torch
import pytest

from mahjong.decision.action import Action, ActionType
from mahjong.environment import MahjongEnv
from mahjong.evaluation.opponent import RandomOpponent
from mahjong.features.policy_schema import policy_action_ops, resolve_policy_schema
from mahjong.model import MLPPolicy
from mahjong.training.opponent_pool import FrozenPolicyOpponent, collect_trajectory_with_opponents
from mahjong.training.parallel_rollout import collect_batch_trajectory
from mahjong.training.ppo import collect_trajectory, init_ppo_from_bc

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import train_ppo  # noqa: E402


def _v2_policy(seed=19):
    torch.manual_seed(seed)
    encoder, feature_dim, action_size = resolve_policy_schema("feature-v2", "action-v2")
    model = MLPPolicy(feature_dim, action_size, (16, 16))
    model.action_schema_version = "action-v2"
    model.allow_kyuushu = False
    model.eval()
    return model, encoder


def _steps(traj):
    return {seat: buf.steps for seat, buf in traj.items()}


def test_v2_reference_and_batch_sampling_match_actions_rewards_and_masks():
    torch.set_num_threads(1)
    model, encoder = _v2_policy()
    device = torch.device("cpu")
    seeds = (530020, 530021)
    batch_envs = [MahjongEnv(seed=s) for s in seeds]
    batch_trajs, _, batch_steps, batch_illegal = collect_batch_trajectory(
        batch_envs, model, encoder, device, max_steps=24, base_seed=10320,
    )
    assert batch_illegal == 0 and batch_steps == 24
    for i, seed in enumerate(seeds):
        ref_env = MahjongEnv(seed=seed)
        ref_traj, ref_steps, ref_illegal = collect_trajectory(
            ref_env, model, encoder, device, max_steps=24,
            rng_seed=10320, env_index=i, rng_version="rng-v2",
        )
        assert ref_illegal == 0 and ref_steps == 24
        assert (ref_env.state.scores, ref_env.state.turn, ref_env.state.kyoku) == (
            batch_envs[i].state.scores, batch_envs[i].state.turn, batch_envs[i].state.kyoku,
        )
        for seat in range(4):
            ref, batched = _steps(ref_traj)[seat], _steps(batch_trajs[i])[seat]
            assert len(ref) == len(batched)
            for left, right in zip(ref, batched):
                assert left.action_id == right.action_id
                assert left.reward == right.reward and left.done == right.done
                assert torch.equal(left.features, right.features)
                assert torch.equal(left.legal_mask, right.legal_mask)
                assert left.legal_mask[left.action_id]
                assert abs(left.old_log_prob - right.old_log_prob) < 1e-5
                assert abs(left.value - right.value) < 1e-5


def test_v2_opponent_pool_and_frozen_policy_use_exact_action_schema():
    torch.set_num_threads(1)
    model, encoder = _v2_policy()
    env = MahjongEnv(seed=530022)
    env.reset()
    learner_seat = env.state.turn
    legal = env.legal_actions(learner_seat)
    frozen = FrozenPolicyOpponent(model, encoder, mode="greedy")
    chosen = frozen.decide(env.observation(learner_seat), legal, random.Random(7))
    to_id, make_mask, _ = policy_action_ops(encoder, model)
    assert chosen in legal and make_mask(legal)[to_id(chosen)]

    opponents = {seat: RandomOpponent() for seat in range(4) if seat != learner_seat}
    traj, _, illegal, bootstrap, truncated = collect_trajectory_with_opponents(
        MahjongEnv(seed=530022), model, encoder, torch.device("cpu"),
        learner_seat=learner_seat, opponents=opponents, max_steps=300,
        max_learner_steps=2, greedy=True, seed=530022,
    )
    assert illegal == 0 and len(traj[learner_seat].steps) == 2
    assert isinstance(bootstrap, float) and isinstance(truncated, bool)
    for step in traj[learner_seat].steps:
        assert step.features.shape == (model.feature_dim,)
        assert step.legal_mask.shape == (model.action_size,)
        assert step.legal_mask[step.action_id]


def test_same_schema_bc_initialization_preserves_actor_logits():
    torch.set_num_threads(1)
    source, encoder = _v2_policy()
    target, _ = _v2_policy(seed=20)
    init_ppo_from_bc(target, source.state_dict())
    env = MahjongEnv(seed=530023)
    env.reset()
    seat = env.state.turn
    features = encoder.encode(env.observation(seat)).unsqueeze(0)
    _, make_mask, _ = policy_action_ops(encoder, source)
    mask = make_mask(env.legal_actions(seat)).unsqueeze(0)
    with torch.no_grad():
        assert torch.equal(source(features, mask).logits, target(features, mask).logits)


def test_v2_opponent_receives_its_own_abort_option():
    class MenuSeen(Exception):
        pass

    class SpyOpponent:
        def decide(self, observation, legal_actions, rng):
            assert any(action.type is ActionType.KYUUSHU_KYUHAI for action in legal_actions)
            raise MenuSeen

    class OneDecisionEnv:
        state = SimpleNamespace(turn=0)

        def reset(self):
            pass

        def done(self):
            return False

        def legal_actions(self, seat):
            return [Action(ActionType.KYUUSHU_KYUHAI), Action(ActionType.DISCARD, tile="1m")]

        def observation(self, seat):
            return None

    model, encoder = _v2_policy()
    with pytest.raises(MenuSeen):
        collect_trajectory_with_opponents(
            OneDecisionEnv(), model, encoder, torch.device("cpu"),
            learner_seat=1, opponents={0: SpyOpponent()}, max_steps=1,
        )


def test_v2_fresh_training_refuses_to_overwrite_existing_run(tmp_path):
    (tmp_path / "epoch_1.pt").write_bytes(b"previous run")
    with pytest.raises(FileExistsError, match="already contains a run"):
        train_ppo._guard_fresh_checkpoint_dir(tmp_path, feature_version="feature-v2", resume=None)
    train_ppo._guard_fresh_checkpoint_dir(tmp_path, feature_version="feature-v2", resume="latest.pt")
    train_ppo._guard_fresh_checkpoint_dir(tmp_path, feature_version="feature-v1", resume=None)
