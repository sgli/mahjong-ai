#!/usr/bin/env python
"""Sampling gap offline diagnosis (Phase 10.3 诊断 H1/H2/H3).

Fixed subset (t93) → per-model distribution stats vs BC reference:
top-1 prob / entropy / effective support / bad-tail mass / action-type split.

Usage::

    python scripts/eval_sampling_gap.py --num-shards 3 --max-samples 300000 \
        --models experiments/bc_v2.1/checkpoints/best.pt,experiments/ppo_fix/c/latest.pt,experiments/ppo_fix/d6/latest.pt
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

from mahjong.evaluation import load_policy  # noqa: E402
from mahjong.training import ParquetBatchDataset, collate_record_batches  # noqa: E402

_TYPE_OF = [None] * 270
for i in range(270):
    if i < 34:
        _TYPE_OF[i] = "discard"
    elif i < 68:
        _TYPE_OF[i] = "riichi"
    elif i == 68:
        _TYPE_OF[i] = "tsumo"
    elif i == 69:
        _TYPE_OF[i] = "pass"
    elif i == 70:
        _TYPE_OF[i] = "ron"
    elif i < 105:
        _TYPE_OF[i] = "pon"
    elif i < 168:
        _TYPE_OF[i] = "chi"
    else:
        _TYPE_OF[i] = "kan"


def _device(spec):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if spec == "auto" else torch.device(spec)


def main(argv=None):
    ap = argparse.ArgumentParser(description="sampling gap offline diagnosis")
    ap.add_argument("--dataset-dir", default="data/processed/decision-v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-shards", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=300_000)
    ap.add_argument("--models", default="experiments/bc_v2.1/checkpoints/best.pt,experiments/ppo_fix/c/latest.pt,experiments/ppo_fix/d6/latest.pt")
    ap.add_argument("--out", default="experiments/bc_retention/sampling_gap.json")
    args = ap.parse_args(argv)

    device = _device(args.device)
    shards = sorted((Path(args.dataset_dir) / args.split).glob("*.parquet"))[: args.num_shards]
    model_paths = [p.strip() for p in args.models.split(",") if p.strip()]
    models = {}
    for mp in model_paths:
        _, m = load_policy(mp, device)
        m.eval()
        models[mp] = m

    dl = torch.utils.data.DataLoader(
        ParquetBatchDataset(shards, args.batch_size), batch_size=1, collate_fn=collate_record_batches,
        num_workers=args.num_workers, prefetch_factor=4 if args.num_workers else None, pin_memory=device.type == "cuda",
    )

    # 累加器：每个模型 top1 总和/平方和、entropy、bad-tail 质量、按类型计数
    stats = {mp: {"top1_sum": 0.0, "top1_sq": 0.0, "ent_sum": 0.0, "ent_sq": 0.0, "supp_sum": 0.0, "badtail_mass_sum": 0.0, "lowp_frac_sum": 0.0, "type_argmax": {}, "n": 0} for mp in model_paths}

    bc_probs_cache = None
    n = 0
    with torch.no_grad():
        for features, action_ids, masks in dl:
            features = features.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            action_ids = action_ids.to(device)
            # BC 分布作为 reference（bad-tail 依据）
            bc_out = models[model_paths[0]](features, masks)
            bc_logp = torch.log_softmax(bc_out.logits, dim=-1)
            bc_probs = torch.exp(bc_logp)
            bc_bad = bc_probs < 0.01  # BC 认为极不可能的动作位

            for mp in model_paths:
                out = models[mp](features, masks)
                logp = torch.log_softmax(out.logits, dim=-1)
                probs = torch.exp(logp)
                top1 = probs.max(-1).values
                safe = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))
                ent = -(probs * safe).sum(-1)
                b = features.shape[0]
                st = stats[mp]
                st["top1_sum"] += top1.sum().item()
                st["top1_sq"] += (top1 ** 2).sum().item()
                st["ent_sum"] += ent.sum().item()
                st["ent_sq"] += (ent ** 2).sum().item()
                st["supp_sum"] += torch.exp(ent).sum().item()
                # bad-tail：PPO 在 BC 低概率位的质量
                st["badtail_mass_sum"] += probs.masked_select(bc_bad).sum().item()
                # 采样到低概率动作比例（p<0.05 的合法动作质量）
                legal = probs.masked_fill(~masks, 0.0)
                st["lowp_frac_sum"] += (legal > 0).float().sum().item() and (legal < 0.05).float().sum().item() / (legal > 0).float().sum().item() if (legal > 0).any() else 0.0
                # argmax 动作类型分布
                arg = logp.argmax(-1)
                for a in arg.cpu().tolist():
                    t = _TYPE_OF[a]
                    st["type_argmax"][t] = st["type_argmax"].get(t, 0) + 1
                st["n"] += b
            n += b
            if n >= args.max_samples:
                break

    rows = []
    for mp in model_paths:
        st = stats[mp]
        nn = st["n"]
        mean = st["top1_sum"] / nn
        rows.append({
            "checkpoint": mp,
            "samples": nn,
            "top1_mean": round(mean, 6),
            "top1_std": round((st["top1_sq"] / nn - mean ** 2) ** 0.5, 6),
            "entropy_mean": round(st["ent_sum"] / nn, 6),
            "entropy_std": round((st["ent_sq"] / nn - (st["ent_sum"] / nn) ** 2) ** 0.5, 6),
            "effective_support_mean": round(st["supp_sum"] / nn, 6),
            "badtail_mass_under_bc_001": round(st["badtail_mass_sum"] / nn, 6),
            "argmax_type_dist": st["type_argmax"],
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in rows:
        print(json.dumps({k: v for k, v in r.items() if k != "argmax_type_dist"}, ensure_ascii=False))
        print("  argmax_type_dist:", r["argmax_type_dist"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
