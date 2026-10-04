"""§8：reference vs optimized rollout 全字段 fixed-seed parity（Phase 10.3）。"""

from __future__ import annotations

import torch

from mahjong.decision.action import Action
from mahjong.environment import MahjongEnv
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder
from mahjong.features.action_space import action_to_id, legal_mask
from mahjong.model import MLPPolicy
from mahjong.training import attribute_step_rewards, collect_trajectory, compute_gae, flush_terminal_rewards
from mahjong.training.parallel_rollout import collect_batch_trajectory
from mahjong.training.ppo_buffer import TrajectoryBuffer, TrajectoryStep


def _model_encoder(seed=0):
    torch.manual_seed(seed)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    model.eval()
    return model, ObservationEncoder()


def _sequential_greedy_full(env, model, encoder, device, max_steps=300):
    """逐步 greedy 收集（与 collect_batch_trajectory 相同的 reward 归属 + log_softmax 口径）。"""
    env.reset()
    bufs = {s: TrajectoryBuffer() for s in range(4)}
    pending = {s: 0.0 for s in range(4)}
    steps = 0
    while not env.done() and steps < max_steps:
        seat = env.state.turn
        legal = [a for a in env.legal_actions(seat) if isinstance(a, Action)]
        obs = env.observation(seat)
        feats = encoder.encode(obs).to(device)
        mask = legal_mask(legal).to(device)
        with torch.no_grad():
            out = model(feats.unsqueeze(0), mask.unsqueeze(0))
            logp = torch.log_softmax(out.logits, dim=-1)
            aid = int(logp.argmax(dim=-1).item())
        action = next(a for a in legal if action_to_id(a) == aid)
        rewards, done = env.step(action)
        reward = attribute_step_rewards(rewards, seat, bufs, pending)
        bufs[seat].append(
            TrajectoryStep(
                features=feats.cpu(), action_id=aid, legal_mask=mask.cpu(),
                old_log_prob=float(logp[0, aid].item()), value=float(out.value[0].item()),
                reward=reward, done=bool(done),
            )
        )
        steps += 1
    if env.done():
        flush_terminal_rewards(env, bufs, pending)
    return bufs, steps


def _flatten(bufs):
    """返回 (action_ids, rewards, dones, old_log_probs, values) 的逐 seat 拼接列表。"""
    aids, rews, dns, logps, vals = [], [], [], [], []
    for s in range(4):
        for st in bufs[s].steps:
            aids.append(st.action_id)
            rews.append(st.reward)
            dns.append(st.done)
            logps.append(st.old_log_prob)
            vals.append(st.value)
    return aids, rews, dns, logps, vals


def test_full_field_parity_single_env():
    """batch=1 时 reference（逐步 greedy）与 optimized（batch greedy）全字段一致。"""
    model, encoder = _model_encoder()
    device = torch.device("cpu")
    seq_env = MahjongEnv(seed=11)
    opt_env = MahjongEnv(seed=11)

    seq_bufs, seq_steps = _sequential_greedy_full(seq_env, model, encoder, device, max_steps=400)
    opt_trajs, opt_dones, opt_steps, illegal = collect_batch_trajectory([opt_env], model, encoder, device, max_steps=400, greedy=True)

    assert illegal == 0
    opt_bufs = opt_trajs[0]

    sa, sr, sd, sl, sv = _flatten(seq_bufs)
    oa, orw, od, ol, ov = _flatten(opt_bufs)

    # 离散量必须精确
    assert sa == oa, "action_id mismatch"
    assert sr == orw, "reward mismatch"
    assert sd == od, "done mismatch"

    # 连续量：batch=1 → 同一次 forward，应 bitwise 一致
    assert len(sl) == len(ol) == len(sv) == len(ov)
    for i in range(len(sl)):
        assert abs(sl[i] - ol[i]) < 1e-7, f"old_log_prob[{i}] {sl[i]} vs {ol[i]}"
        assert abs(sv[i] - ov[i]) < 1e-7, f"value[{i}] {sv[i]} vs {ov[i]}"


def test_advantage_return_parity():
    """同 rewards/values/dones 下 GAE 的 advantage/return 一致。"""
    rewards = [0.0, 1.0, -1.0, 0.5]
    values = [0.1, 0.2, 0.3, 0.4]
    dones = [False, False, False, True]
    a1, r1 = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    a2, r2 = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    assert a1 == a2
    assert r1 == r2


def test_sampling_batch_reproducible():
    """rollout-v2 sampling 在固定 seed 下可复现（同一 batch multinomial 消耗）。"""
    probs = torch.full((4, 270), 1e-9)
    probs[:, :8] = 1 / 8
    torch.manual_seed(0)
    a = torch.multinomial(probs, 1).squeeze(-1)
    torch.manual_seed(0)
    b = torch.multinomial(probs, 1).squeeze(-1)
    assert torch.equal(a, b)


def test_sampling_rng_order_diverges_batch_vs_sequential():
    """方案 2 依据：batch multinomial 与逐行 multinomial 的 RNG 消耗顺序不同（状态不同）。"""
    probs = torch.full((4, 270), 1e-9)
    probs[:, :8] = 1 / 8
    torch.manual_seed(0)
    torch.multinomial(probs, 1)
    state_batch = torch.get_rng_state().clone()

    torch.manual_seed(0)
    for i in range(4):
        torch.multinomial(probs[i : i + 1], 1)
    state_seq = torch.get_rng_state()
    assert not torch.equal(state_batch, state_seq)  # RNG 消耗顺序不同 → 状态不同


def test_sampling_per_env_rng_reproducible():
    """每 env 独立 RNG：batch rollout 的 sampling 动作序列固定 seed 可复现。"""
    model, encoder = _model_encoder()
    envs = [MahjongEnv(seed=s) for s in (1, 2)]
    t1, _, _, il1 = collect_batch_trajectory(envs, model, encoder, torch.device("cpu"), max_steps=50, greedy=False, base_seed=0)
    envs2 = [MahjongEnv(seed=s) for s in (1, 2)]
    t2, _, _, il2 = collect_batch_trajectory(envs2, model, encoder, torch.device("cpu"), max_steps=50, greedy=False, base_seed=0)
    assert il1 == il2 == 0
    a1 = [st.action_id for buf in t1[0].values() for st in buf.steps]
    a2 = [st.action_id for buf in t2[0].values() for st in buf.steps]
    assert a1 == a2


def test_sampling_rng_v2_reference_batch_action_id_exact():
    """rng-v2：reference（per-env generator）与 optimized（per-env generator）action_id 逐位一致。"""
    model, encoder = _model_encoder()
    device = torch.device("cpu")
    env_ref = MahjongEnv(seed=1)
    ref_traj, ref_steps, _ = collect_trajectory(env_ref, model, encoder, device, max_steps=100, rng_seed=0, env_index=0, rng_version="rng-v2")
    env_batch = MahjongEnv(seed=1)
    batch_trajs, _, _, illegal = collect_batch_trajectory([env_batch], model, encoder, device, max_steps=100, greedy=False, base_seed=0)
    assert illegal == 0
    ra = [st.action_id for buf in ref_traj.values() for st in buf.steps]
    ba = [st.action_id for buf in batch_trajs[0].values() for st in buf.steps]
    assert ra == ba
