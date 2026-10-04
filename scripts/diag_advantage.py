#!/usr/bin/env python
"""Advantage noise diagnostic (Phase 10.3 E-D).

For each checkpoint: small greedy rollout → value quality + advantage
between/within-episode variance + advantage↔final-result correlation/sign-accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from mahjong.environment import MahjongEnv  # noqa: E402
from mahjong.evaluation import load_policy, value_quality_metrics  # noqa: E402
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder  # noqa: E402
from mahjong.model import MLPPolicy  # noqa: E402
from mahjong.training import collect_trajectory, compute_gae  # noqa: E402


def _device(spec):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if spec == "auto" else torch.device(spec)


def audit(checkpoint, device, *, games=4, max_steps=20000, gamma=0.99, lam=0.95):
    _, model = load_policy(checkpoint, device)
    model.eval()
    encoder = ObservationEncoder()

    all_values = []
    all_returns = []
    # per-seat-episode: mean advantage, final rank, final score
    ep_adv_mean = []
    ep_rank = []
    ep_score = []

    for g in range(games):
        env = MahjongEnv(seed=g)
        traj, steps, illegal = collect_trajectory(env, model, encoder, device, max_steps=max_steps, rng_seed=0, env_index=0, rng_version="rng-v1")
        if illegal:
            continue
        ranks = env.state.final_ranks if env.done() else None
        scores = env.state.scores
        for s in range(4):
            buf = traj[s]
            if not buf.steps:
                continue
            rewards = [st.reward for st in buf.steps]
            values = [st.value for st in buf.steps]
            dones = [st.done for st in buf.steps]
            adv, ret = compute_gae(rewards, values, dones, gamma, lam)
            all_values.extend(values)
            all_returns.extend(ret)
            ep_adv_mean.append(sum(adv) / len(adv))
            if ranks is not None:
                ep_rank.append(ranks[s])
            ep_score.append(scores[s])

    vq = value_quality_metrics(torch.tensor(all_values), torch.tensor(all_returns))

    # advantage between/within variance
    n_eps = len(ep_adv_mean)
    grand_mean = sum(ep_adv_mean) / max(n_eps, 1)
    var_between = sum((m - grand_mean) ** 2 for m in ep_adv_mean) / max(n_eps, 1)
    # within：用 all advantage 的总方差 − between（近似）
    all_adv = []
    # 重新收集 all advantage（简便：用 return-value 近似 = advantage）
    for g in range(games):
        env = MahjongEnv(seed=g)
        traj, steps, illegal = collect_trajectory(env, model, encoder, device, max_steps=max_steps, rng_seed=0, env_index=0, rng_version="rng-v1")
        for s in range(4):
            buf = traj[s]
            if not buf.steps:
                continue
            rewards = [st.reward for st in buf.steps]
            values = [st.value for st in buf.steps]
            dones = [st.done for st in buf.steps]
            adv, ret = compute_gae(rewards, values, dones, gamma, lam)
            all_adv.extend(adv)

    all_adv_t = torch.tensor(all_adv)
    var_total = float(all_adv_t.var(unbiased=False).item())
    var_within = max(0.0, var_total - var_between)

    # advantage vs final result correlation + sign accuracy
    corr_adv_rank = None
    sign_acc = None
    if ep_rank and len(ep_rank) == n_eps:
        adv_t = torch.tensor(ep_adv_mean)
        rank_t = torch.tensor(ep_rank, dtype=torch.float32)
        corr_adv_rank = float(torch.corrcoef(torch.stack([adv_t, rank_t]))[0, 1].item())
        median = rank_t.median().item()
        pos = adv_t > 0
        if pos.any():
            sign_acc = float((rank_t[pos] < median).float().mean().item())  # advantage>0 且 rank 更优(<median) 的比例

    # constant baseline: 用全局均值 return 替代 value，重算 advantage 的相关性
    const_val = float(torch.tensor(all_returns).mean().item())
    const_adv = [r - const_val for r in all_returns]
    # 简化：常数基线的 advantage 相关性（用 return 相对全局均值）
    corr_const = None
    if ep_rank:
        # 每 seat-episode 的 mean(const_adv) 与 rank
        ep_const = []
        idx = 0
        for g in range(games):
            env = MahjongEnv(seed=g)
            traj, steps, illegal = collect_trajectory(env, model, encoder, device, max_steps=max_steps, rng_seed=0, env_index=0, rng_version="rng-v1")
            for s in range(4):
                buf = traj[s]
                if not buf.steps:
                    continue
                rewards = [st.reward for st in buf.steps]
                values = [st.value for st in buf.steps]
                dones = [st.done for st in buf.steps]
                adv, ret = compute_gae(rewards, values, dones, gamma, lam)
                # 常数基线 advantage = return - const_val
                cadv = [r - const_val for r in ret]
                ep_const.append(sum(cadv) / len(cadv))
        if ep_const:
            cc_t = torch.tensor(ep_const)
            corr_const = float(torch.corrcoef(torch.stack([cc_t, torch.tensor(ep_rank, dtype=torch.float32)]))[0, 1].item())

    return {
        "checkpoint": checkpoint,
        "games": games,
        "samples": len(all_values),
        "value_quality": {k: round(v, 4) for k, v in vq.items()},
        "advantage": {
            "var_total": round(var_total, 4),
            "var_between_episodes": round(var_between, 4),
            "var_within": round(var_within, 4),
            "between_share": round(var_between / max(var_total, 1e-12), 4),
            "corr_adv_vs_rank": round(corr_adv_rank, 4) if corr_adv_rank is not None else None,
            "sign_accuracy_adv_pos": round(sign_acc, 4) if sign_acc is not None else None,
            "corr_const_baseline_vs_rank": round(corr_const, 4) if corr_const is not None else None,
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="experiments/bc_v2.1/checkpoints/best.pt,experiments/ppo_fix/d6/latest.pt,experiments/ppo_fix/g999/epoch_1.pt,experiments/ppo_fix/g999/epoch_2.pt,experiments/ppo_fix/g999/epoch_5.pt")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="experiments/diag_advantage/results.json")
    args = ap.parse_args(argv)

    device = _device(args.device)
    rows = []
    for cp in [p.strip() for p in args.checkpoints.split(",") if p.strip()]:
        rows.append(audit(cp, device, games=args.games))
        print(json.dumps(rows[-1], ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
