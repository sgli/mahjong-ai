"""Phase 10.3 诊断 #2：优化 rollout vs reference 的 encode/mask/logits/log-prob/value/ratio parity。"""

from __future__ import annotations

import math

import torch

from mahjong.decision.action import Action
from mahjong.environment import MahjongEnv
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder
from mahjong.features.action_space import legal_mask
from mahjong.model import MLPPolicy
from mahjong.training import collect_trajectory
from mahjong.training.parallel_rollout import (
    collect_batch_trajectory,
    encode_observations_batch,
    legal_mask_batch,
)


def _model_encoder(device="cpu", hidden=(32, 32)):
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, hidden).to(device)
    model.eval()
    return model, ObservationEncoder()


def _real_obs_list(seed=0):
    env = MahjongEnv(seed=seed)
    env.reset()
    return [env.observation(s) for s in range(4)]


def test_encode_parity_bitwise():
    encoder = ObservationEncoder()
    obs = _real_obs_list()
    batch = encode_observations_batch(obs)
    for i, o in enumerate(obs):
        assert torch.equal(batch[i], encoder.encode(o))


def test_legal_mask_parity_bitwise():
    env = MahjongEnv(seed=0)
    env.reset()
    for s in range(4):
        legal = [a for a in env.legal_actions(s) if isinstance(a, Action)]
        b = legal_mask_batch([legal])
        ref = legal_mask(legal)
        assert torch.equal(b[0], ref)


def _four_env_obs_legal(seed=0):
    envs = [MahjongEnv(seed=seed + i) for i in range(4)]
    obs, legal = [], []
    for env in envs:
        env.reset()
        seat = env.state.turn
        obs.append(env.observation(seat))
        legal.append([a for a in env.legal_actions(seat) if isinstance(a, Action)])
    return obs, legal


def test_logits_parity_quantified():
    model, encoder = _model_encoder()
    obs, legal = _four_env_obs_legal()
    feats = encode_observations_batch(obs)
    mask = legal_mask_batch(legal)
    with torch.no_grad():
        batched = model(feats, mask).logits
    for i, o in enumerate(obs):
        with torch.no_grad():
            single = model(encoder.encode(o).unsqueeze(0), legal_mask(legal[i]).unsqueeze(0)).logits[0]
        # 仅比较合法位（非法位被 mask 成 -inf，-inf - -inf = NaN）
        finite = torch.isfinite(batched[i]) & torch.isfinite(single)
        assert finite.any()
        diff = (batched[i][finite] - single[finite]).abs().max().item()
        assert diff < 1e-5, f"logits diff {diff}"  # 仅浮点（batch GEMM 归约顺序）


def test_value_parity_quantified():
    model, encoder = _model_encoder()
    obs, legal = _four_env_obs_legal()
    feats = encode_observations_batch(obs)
    mask = legal_mask_batch(legal)
    with torch.no_grad():
        batched_v = model(feats, mask).value
    for i, o in enumerate(obs):
        with torch.no_grad():
            single_v = model(encoder.encode(o).unsqueeze(0), legal_mask(legal[i]).unsqueeze(0)).value[0]
        diff = (batched_v[i] - single_v).abs().max().item()
        assert diff < 1e-5, f"value diff {diff}"


def test_batch_greedy_old_log_prob_is_log_softmax():
    """batch rollout 的 old_log_prob 必须是所选 action 的 log_softmax（非 0、非未 mask）。"""
    model, encoder = _model_encoder()
    envs = [MahjongEnv(seed=s) for s in (1, 2)]
    trajs, dones, steps, illegal = collect_batch_trajectory(envs, model, encoder, torch.device("cpu"), max_steps=50, greedy=True)
    assert illegal == 0
    seen = 0
    for ti in trajs:
        for buf in ti.values():
            for st in buf.steps:
                with torch.no_grad():
                    out = model(st.features.unsqueeze(0), st.legal_mask.unsqueeze(0))
                    new_logp = torch.log_softmax(out.logits, dim=-1)[0, st.action_id].item()
                # log_softmax 与更新路径同源 → ratio 应 ≈ 1（batch vs single forward 仅浮点差）
                assert abs(math.exp(new_logp - st.old_log_prob) - 1.0) < 1e-6
                assert math.isfinite(st.old_log_prob)
                seen += 1
    assert seen > 0


def test_sequential_ratio_one_self_check():
    """reference（逐步）rollout 的 old_log_prob 与更新路径 log_softmax 的 ratio≈1（量化）。"""
    model, encoder = _model_encoder()
    env = MahjongEnv(seed=11)
    traj, steps, illegal = collect_trajectory(env, model, encoder, torch.device("cpu"), max_steps=300)
    assert illegal == 0
    ratios = []
    for buf in traj.values():
        for st in buf.steps:
            with torch.no_grad():
                out = model(st.features.unsqueeze(0), st.legal_mask.unsqueeze(0))
                new_logp = torch.log_softmax(out.logits, dim=-1)[0, st.action_id].item()
            ratios.append(math.exp(new_logp - st.old_log_prob))
    assert ratios
    mean = sum(ratios) / len(ratios)
    mx = max(abs(r - 1.0) for r in ratios)
    # 逐步 rollout 用 log(softmax+1e-12)，更新路径用 log_softmax → 仅浮点差，ratio≈1
    assert mx < 1e-5, f"max |ratio-1|={mx}（应仅浮点差）"
    assert abs(mean - 1.0) < 1e-6
