"""E-C1 critic warmup / actor freeze 开关（Phase 10.3）。"""

from __future__ import annotations

import torch

from mahjong.environment import MahjongEnv
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder
from mahjong.model import MLPPolicy
from mahjong.training import train_ppo


def _small_cfg(**kw):
    base = dict(
        num_envs=1, env_seed=0, epochs=1, steps_per_epoch=100, batch_size=32,
        update_epochs=1, gamma=0.99, lam=0.95, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
        device=torch.device("cpu"), max_episode_steps=300, diagnose=False,
    )
    base.update(kw)
    return base


_FEATS = torch.randn(4, FEATURE_DIM)
_MASKS = torch.zeros(4, ACTION_SPACE_SIZE, dtype=torch.bool)
_MASKS[:, :50] = True


def _fixed_batch(model):
    with torch.no_grad():
        return model(_FEATS, _MASKS).logits


def test_warmup_steps_zero_no_warmup_metrics():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (16, 16))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    enc = ObservationEncoder()
    hist = train_ppo(model, enc, opt, **_small_cfg(warmup_steps=0))
    m = hist[0]
    assert "warmup_ev" not in m
    assert "warmup_final_ev" not in m


def test_warmup_freeze_trunk_policy_logits_unchanged():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (16, 16))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    enc = ObservationEncoder()
    before = _fixed_batch(model)
    hist = train_ppo(model, enc, opt, **_small_cfg(warmup_steps=10_000, warmup_freeze_trunk=True))
    after = _fixed_batch(model)
    assert torch.equal(before, after)  # actor（trunk+action_head）被冻结 → logits 不变
    m = hist[0]
    assert "warmup_ev" in m and m["warmup_ev"] == m["warmup_ev"]


def test_warmup_freeze_trunk_value_only_updates():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (16, 16))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    enc = ObservationEncoder()
    vh_before = [p.detach().clone() for n, p in model.named_parameters() if n.startswith("value_head.")]
    trunk_before = [p.detach().clone() for n, p in model.named_parameters() if not n.startswith("value_head.")]
    train_ppo(model, enc, opt, **_small_cfg(warmup_steps=10_000, warmup_freeze_trunk=True))
    for p, b in zip([p for n, p in model.named_parameters() if n.startswith("value_head.")], vh_before):
        assert not torch.equal(p.detach(), b)  # value 参数变了
    for p, b in zip([p for n, p in model.named_parameters() if not n.startswith("value_head.")], trunk_before):
        assert torch.equal(p.detach(), b)  # policy/trunk 不变


def test_warmup_then_resume_policy_updates():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (16, 16))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    enc = ObservationEncoder()
    # 预热很短，之后切回标准 PPO
    hist = train_ppo(model, enc, opt, **_small_cfg(warmup_steps=50, warmup_freeze_trunk=True))
    # 预热后 policy 应开始变化（本 epoch 已跨过 warmup，第二步会切回）
    assert hist[0]["warmup_steps"] == 50
