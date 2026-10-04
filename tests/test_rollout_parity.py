"""Parallel/batch rollout parity (Phase 10.3 §30)."""

from __future__ import annotations

import torch

from mahjong.decision.action import Action
from mahjong.environment import MahjongEnv
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder
from mahjong.features.action_space import action_to_id, legal_mask
from mahjong.model import MLPPolicy
from mahjong.training.parallel_rollout import collect_batch_trajectory, encode_observations_batch


def _run_greedy_sequential(env, model, encoder, device, max_steps=300):
    env.reset()
    illegal = 0
    steps = 0
    while not env.done() and steps < max_steps:
        seat = env.state.turn
        legal = [a for a in env.legal_actions(seat) if isinstance(a, Action)]
        obs = env.observation(seat)
        feats = encoder.encode(obs).to(device)
        mask = legal_mask(legal).to(device)
        with torch.no_grad():
            out = model(feats.unsqueeze(0), mask.unsqueeze(0))
            aid = int(out.logits.argmax(dim=-1).item())
        if not mask[aid].item():
            illegal += 1
        action = next(a for a in legal if action_to_id(a) == aid)
        env.step(action)
        steps += 1
    return env, illegal, steps


def _model_encoder():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    model.eval()
    return model, ObservationEncoder()


def test_batch_greedy_matches_sequential():
    model, encoder = _model_encoder()
    device = torch.device("cpu")
    seeds = [1, 2, 3]

    seq_envs = [MahjongEnv(seed=s) for s in seeds]
    batch_envs = [MahjongEnv(seed=s) for s in seeds]

    seq_results = [_run_greedy_sequential(e, model, encoder, device, max_steps=400) for e in seq_envs]
    trajs, dones, steps, illegal = collect_batch_trajectory(batch_envs, model, encoder, device, max_steps=400, greedy=True)

    assert illegal == 0
    for i, s in enumerate(seeds):
        seq_env, seq_illegal, _ = seq_results[i]
        batch_env = batch_envs[i]
        assert seq_illegal == 0
        assert seq_env.state.final_ranks == batch_env.state.final_ranks, f"seed {s} ranks differ"
        assert seq_env.state.scores == batch_env.state.scores, f"seed {s} scores differ"


def test_batch_reward_conservation():
    model, encoder = _model_encoder()
    device = torch.device("cpu")
    envs = [MahjongEnv(seed=s) for s in (11, 12, 13)]
    trajs, dones, steps, illegal = collect_batch_trajectory(envs, model, encoder, device, max_steps=2000, greedy=True)
    assert illegal == 0
    for i, env in enumerate(envs):
        for s in range(4):
            total = sum(st.reward for st in trajs[i][s].steps)
            assert abs(total - env.reward(s)) < 1e-6, f"env {i} seat {s}: {total} != {env.reward(s)}"


def test_batch_same_seed_reproducible():
    model, encoder = _model_encoder()
    device = torch.device("cpu")
    a = [MahjongEnv(seed=5) for _ in range(2)]
    b = [MahjongEnv(seed=5) for _ in range(2)]
    _, _, _, il_a = collect_batch_trajectory(a, model, encoder, device, max_steps=400, greedy=True)
    _, _, _, il_b = collect_batch_trajectory(b, model, encoder, device, max_steps=400, greedy=True)
    assert il_a == il_b == 0
    assert [e.state.final_ranks for e in a] == [e.state.final_ranks for e in b]
    assert [e.state.scores for e in a] == [e.state.scores for e in b]


def test_batch_encode_matches_sequential_bitwise():
    """§10 批量编码与逐步 encoder.encode 严格一致（bitwise）。"""
    encoder = ObservationEncoder()
    env = MahjongEnv(seed=0)
    env.reset()
    obs_list = [env.observation(s) for s in range(4)]
    batch = encode_observations_batch(obs_list)
    assert batch.shape == (4, FEATURE_DIM)
    for i, obs in enumerate(obs_list):
        assert torch.equal(batch[i], encoder.encode(obs)), f"obs {i} differs"
