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


def _correlation(a, b):
    if len(a) < 2 or len(a) != len(b):
        return None
    x, y = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    if not bool(x.var() > 0) or not bool(y.var() > 0):
        return None
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1])


def audit(checkpoint, device, *, games=4, max_steps=20000, gamma=0.99, lam=0.95):
    encoder, model = load_policy(checkpoint, device)
    model.eval()
    episodes = []
    for g in range(games):
        env = MahjongEnv(seed=g)
        traj, steps, illegal = collect_trajectory(env, model, encoder, device, max_steps=max_steps, rng_seed=g, env_index=0, rng_version="rng-v2")
        if illegal or not env.done():
            raise ValueError(f"diagnostic game {g} incomplete/illegal; no unpaired statistics emitted")
        for seat in range(4):
            buf = traj[seat]
            if not buf.steps:
                continue
            rewards = [st.reward for st in buf.steps]
            values = [st.value for st in buf.steps]
            dones = [st.done for st in buf.steps]
            adv, ret = compute_gae(rewards, values, dones, gamma, lam)
            episodes.append({"seed": g, "seat": seat, "rewards": rewards, "values": values,
                             "dones": dones, "adv": adv, "returns": ret, "rank": env.state.final_ranks[seat]})
    if not episodes:
        raise ValueError("diagnostic has no trajectories")
    all_values = [v for ep in episodes for v in ep["values"]]
    all_returns = [v for ep in episodes for v in ep["returns"]]
    all_adv = [v for ep in episodes for v in ep["adv"]]
    count = len(all_adv)
    grand_mean = sum(all_adv) / count
    ep_means = [sum(ep["adv"]) / len(ep["adv"]) for ep in episodes]
    var_total = sum((v - grand_mean) ** 2 for v in all_adv) / count
    var_between = sum(len(ep["adv"]) * (mean - grand_mean) ** 2 for ep, mean in zip(episodes, ep_means)) / count
    var_within = sum(sum((v - mean) ** 2 for v in ep["adv"]) for ep, mean in zip(episodes, ep_means)) / count
    const_val = sum(all_returns) / len(all_returns)
    const_means = []
    for ep in episodes:
        cadv, _ = compute_gae(ep["rewards"], [const_val] * len(ep["rewards"]), ep["dones"], gamma, lam)
        const_means.append(sum(cadv) / len(cadv))
    ranks = [ep["rank"] for ep in episodes]
    median = torch.tensor(ranks, dtype=torch.float32).median().item()
    positive = [rank < median for mean, rank in zip(ep_means, ranks) if mean > 0]
    return {
        "checkpoint": str(checkpoint), "games": games, "samples": count,
        "diagnostic_version": "paired-trajectories-v2",
        "episode_keys": [[ep["seed"], ep["seat"]] for ep in episodes],
        "value_quality": value_quality_metrics(torch.tensor(all_values), torch.tensor(all_returns)),
        "advantage": {
            "var_total": var_total, "var_between_episodes": var_between, "var_within": var_within,
            "between_share": var_between / var_total if var_total else 0.0,
            "corr_adv_vs_rank": _correlation(ep_means, ranks),
            "sign_accuracy_adv_pos": sum(positive) / len(positive) if positive else None,
            "corr_const_baseline_vs_rank": _correlation(const_means, ranks),
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
