"""PPO update direction + multi-epoch drift (Phase 10.3 诊断 §3/§7)."""

from __future__ import annotations

import torch

from mahjong.environment import MahjongEnv
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder
from mahjong.model import MLPPolicy
from mahjong.training import approx_kl, ppo_loss, train_ppo


def _batch(model):
    feats = torch.randn(16, FEATURE_DIM)
    masks = torch.zeros(16, ACTION_SPACE_SIZE, dtype=torch.bool)
    masks[:, :100] = True
    aids = torch.randint(0, 100, (16,))
    with torch.no_grad():
        out = model(feats, masks)
        old_logp = torch.log_softmax(out.logits, dim=-1).gather(-1, aids.unsqueeze(-1)).squeeze(-1)
        old_entropy = _entropy(out.logits)
    return feats, masks, aids, old_logp, out, old_entropy


def _entropy(logits):
    logp = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(logp)
    safe = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))
    return -(probs * safe).sum(-1).mean()


def test_positive_advantage_raises_log_prob():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    feats, masks, aids, old_logp, out, old_entropy = _batch(model)
    advantages = torch.ones(16) * 2.0
    returns = advantages + out.value.squeeze(-1)

    loss, metrics = ppo_loss(
        model, feats, masks, aids, old_logp, advantages, returns,
        clip_eps=0.2, vf_coef=0.5, ent_coef=0.0,
    )
    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad():
        new_out = model(feats, masks)
        new_logp = torch.log_softmax(new_out.logits, dim=-1).gather(-1, aids.unsqueeze(-1)).squeeze(-1)
    assert (new_logp - old_logp).mean() > 0  # 正 advantage ⇒ log_prob 上升


def test_update_epoch_records_ratio_and_kl():
    """模拟多 epoch：每轮 update 后 ratio 均值/幅度 与 approx_kl 可记录（finite）。"""
    torch.manual_seed(1)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    feats, masks, aids, old_logp, out, old_entropy = _batch(model)
    advantages = torch.randn(16)
    returns = advantages + out.value.squeeze(-1)

    epochs = []
    for _ in range(4):
        loss, metrics = ppo_loss(
            model, feats, masks, aids, old_logp, advantages, returns,
            clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            new_out = model(feats, masks)
            new_logp = torch.log_softmax(new_out.logits, dim=-1).gather(-1, aids.unsqueeze(-1)).squeeze(-1)
            ratio = torch.exp(new_logp - old_logp)
            kl = approx_kl(old_logp, new_logp).item()
        epochs.append({
            "kl": kl,
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
            "ratio_min": ratio.min().item(),
            "ratio_max": ratio.max().item(),
        })
        old_logp = new_logp  # 下轮以新为 old

    assert len(epochs) == 4
    for e in epochs:
        assert all(map(lambda v: v == v, e.values()))  # finite（非 NaN）


def test_train_ppo_diagnose_records_update_epoch_and_distribution():
    """§7：diagnose=True 时 metrics 必须含 update_epoch_* / ratio_* / clip_* / 分布统计。"""
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (16, 16))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    enc = ObservationEncoder()
    hist = train_ppo(
        model, enc, opt, num_envs=1, env_seed=0, epochs=1, steps_per_epoch=100, batch_size=32,
        update_epochs=2, gamma=0.99, lam=0.95, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
        device=torch.device("cpu"), max_episode_steps=300, diagnose=True,
    )
    m = hist[0]
    assert "update_epoch_1_approx_kl" in m
    assert "update_epoch_1_mean_ratio" in m
    assert "update_epoch_1_clip_fraction" in m
    assert "update_epoch_1_clip_positive_fraction" in m
    assert "update_epoch_1_clip_negative_fraction" in m
    assert "rewards_mean" in m and "rewards_std" in m
    assert "advantages_mean" in m and "returns_mean" in m


def test_train_ppo_diagnose_false_no_update_epoch_keys():
    """diagnose=False 时行为不变：不出现 update_epoch_* / 分布统计键。"""
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (16, 16))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    enc = ObservationEncoder()
    hist = train_ppo(
        model, enc, opt, num_envs=1, env_seed=0, epochs=1, steps_per_epoch=100, batch_size=32,
        update_epochs=2, gamma=0.99, lam=0.95, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
        device=torch.device("cpu"), max_episode_steps=300, diagnose=False,
    )
    m = hist[0]
    assert not any(k.startswith("update_epoch_") for k in m)
    assert "rewards_mean" not in m
