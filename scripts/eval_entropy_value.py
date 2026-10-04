#!/usr/bin/env python
"""Entropy/exploration + value-function diagnosis (Phase 10.3 诊断 §5/§6).

Part 1 (offline, fixed subset): legal action count / normalized entropy /
BC & PPO entropy / BC & PPO action probability / agreement (reuse t93 subset).
Part 2 (small env rollout): value prediction vs GAE return → MAE/RMSE/corr/EV.

Usage::

    python scripts/eval_entropy_value.py --num-shards 3 --max-samples 300000 \
        --rollout-steps 20000 --out experiments/bc_retention/entropy_value.json
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
from mahjong.evaluation import (  # noqa: E402
    entropy_exploration_metrics,
    load_policy,
    retention_metrics,
    value_quality_metrics,
)
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder  # noqa: E402
from mahjong.model import MLPPolicy  # noqa: E402
from mahjong.training import ParquetBatchDataset, collate_record_batches, collect_trajectory, compute_gae  # noqa: E402


def _device(spec):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if spec == "auto" else torch.device(spec)


def _load(checkpoint, device):
    _, model = load_policy(checkpoint, device)
    model.eval()
    return model


def entropy_part(bc_model, ppo_model, shards, *, batch_size, device, num_workers, max_samples):
    dl = torch.utils.data.DataLoader(
        ParquetBatchDataset(shards, batch_size), batch_size=1, collate_fn=collate_record_batches,
        num_workers=num_workers, prefetch_factor=4 if num_workers else None, pin_memory=device.type == "cuda",
    )
    acc = {}
    n = 0
    with torch.no_grad():
        for features, action_ids, masks in dl:
            features = features.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            bc_out = bc_model(features, masks)
            ppo_out = ppo_model(features, masks)
            ret = retention_metrics(bc_out.logits, ppo_out.logits, masks)
            ppo_exp = entropy_exploration_metrics(ppo_out.logits, masks)
            bc_exp = entropy_exploration_metrics(bc_out.logits, masks)
            # PPO action prob given BC = BC prob at PPO argmax
            ppo_logp = torch.log_softmax(ppo_out.logits, dim=-1)
            ppo_arg = ppo_logp.argmax(-1)
            bc_probs = torch.exp(torch.log_softmax(bc_out.logits, dim=-1))
            ppo_action_prob_given_bc = bc_probs.gather(-1, ppo_arg.unsqueeze(-1)).squeeze(-1)

            batch_metrics = {
                **{k: float(v.item()) for k, v in ret.items()},
                "legal_count": float(ppo_exp["legal_count"].mean().item()),
                "legal_count_min": float(ppo_exp["legal_count"].min().item()),
                "legal_count_max": float(ppo_exp["legal_count"].max().item()),
                "ppo_normalized_entropy": float(ppo_exp["normalized_entropy"].mean().item()),
                "bc_normalized_entropy": float(bc_exp["normalized_entropy"].mean().item()),
                "ppo_argmax_prob": float(ppo_exp["argmax_prob"].mean().item()),
                "bc_argmax_prob": float(bc_exp["argmax_prob"].mean().item()),
                "bc_action_prob_given_ppo": float(ret["p_bc_action_given_ppo"].item()),
                "ppo_action_prob_given_bc": float(ppo_action_prob_given_bc.mean().item()),
            }
            b = features.shape[0]
            for k, v in batch_metrics.items():
                acc[k] = acc.get(k, 0.0) + v * b
            n += b
            if n >= max_samples:
                break
    return {k: round(v / n, 6) for k, v in acc.items()}, n


def value_part(model, encoder, device, *, seed=11, max_steps=20000, gamma=0.99, lam=0.95):
    env = MahjongEnv(seed=seed)
    traj, steps, illegal = collect_trajectory(env, model, encoder, device, max_steps=max_steps)
    if not env.done():
        return {"error": f"game not done (seed {seed})"}, illegal
    values, returns = [], []
    for s in range(4):
        buf = traj[s]
        if not buf.steps:
            continue
        rewards = [st.reward for st in buf.steps]
        vals = [st.value for st in buf.steps]
        dones = [st.done for st in buf.steps]
        adv, ret = compute_gae(rewards, vals, dones, gamma, lam)
        values.extend(vals)
        returns.extend(ret)
    m = value_quality_metrics(torch.tensor(values), torch.tensor(returns))
    m["illegal"] = illegal
    m["samples"] = len(values)
    return {k: round(v, 6) for k, v in m.items()}, illegal


def main(argv=None):
    ap = argparse.ArgumentParser(description="Entropy + value diagnosis.")
    ap.add_argument("--dataset-dir", default="data/processed/decision-v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-shards", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=300_000)
    ap.add_argument("--rollout-steps", type=int, default=20000)
    ap.add_argument("--bc-checkpoint", default="experiments/bc_v2.1/checkpoints/best.pt")
    ap.add_argument("--ppo-checkpoints", default="experiments/ppo_v2/milestones/ppo-20k/latest.pt,experiments/ppo_v2/milestones/ppo-50k/latest.pt,experiments/ppo_v2/milestones/ppo-100k/latest.pt")
    ap.add_argument("--out", default="experiments/bc_retention/entropy_value.json")
    args = ap.parse_args(argv)

    device = _device(args.device)
    shards = sorted((Path(args.dataset_dir) / args.split).glob("*.parquet"))[: args.num_shards]
    bc_model = _load(args.bc_checkpoint, device)
    encoder = ObservationEncoder()
    ppo_paths = [p.strip() for p in args.ppo_checkpoints.split(",") if p.strip()]

    # BC 的 value quality（BC 无 value head，value 无意义 → 记 entropy 部分即可；value 部分只用 PPO）
    rows = []
    for cp in ppo_paths:
        ppo_model = _load(cp, device)
        ent, n = entropy_part(bc_model, ppo_model, shards, batch_size=args.batch_size, device=device, num_workers=args.num_workers, max_samples=args.max_samples)
        val, illegal = value_part(ppo_model, encoder, device, max_steps=args.rollout_steps)
        rows.append({"checkpoint": cp, "samples": n, "entropy_exploration": ent, "value_quality": val, "illegal": illegal})
        print(json.dumps({"checkpoint": cp, "entropy": ent, "value": val}, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
