#!/usr/bin/env python
"""RL rollout performance profiler (Phase 10.3 §8). 先 profiling 后优化。

Reuses the exact rollout stages of ``collect_trajectory`` with granular timers
(no training-behaviour change).  Prints a per-stage time breakdown, steps/sec,
games/hour and GPU/CPU info.

Usage::

    python scripts/profile_rollout.py --episodes 4 --max-steps 800 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from mahjong.decision.action import Action  # noqa: E402
from mahjong.environment import MahjongEnv  # noqa: E402
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder  # noqa: E402
from mahjong.features.action_space import action_to_id, legal_mask  # noqa: E402
from mahjong.model import MLPPolicy  # noqa: E402


def _resolve_device(spec: str) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if spec == "auto" else torch.device(spec)


def profile_rollout(model, encoder, env, *, max_steps: int, device: torch.device) -> dict:
    env.reset()
    t = Counter()
    total_steps = 0
    while not env.done() and total_steps < max_steps:
        seat = env.state.turn
        t0 = time.perf_counter()
        legal = [a for a in env.legal_actions(seat) if isinstance(a, Action)]
        t1 = time.perf_counter()
        obs = env.observation(seat)
        features = encoder.encode(obs).to(device)
        t2 = time.perf_counter()
        mask = legal_mask(legal).to(device)
        t3 = time.perf_counter()
        with torch.no_grad():
            out = model(features.unsqueeze(0), mask.unsqueeze(0))
            probs = torch.softmax(out.logits, dim=-1)
            action_id = int(torch.multinomial(probs, 1).squeeze(-1).item())
        t4 = time.perf_counter()
        action = next(a for a in legal if action_to_id(a) == action_id)
        rewards, done = env.step(action)
        t5 = time.perf_counter()
        # trajectory write cost (dict/step append, mirrors TrajectoryStep)
        step = (features.cpu(), action_id, mask.cpu(), float(0.0), float(out.value[0].item()), float(0.0), bool(done))
        t6 = time.perf_counter()

        t["legal_actions"] += t1 - t0
        t["feature_encode"] += t2 - t1
        t["legal_mask"] += t3 - t2
        t["forward_sample"] += t4 - t3
        t["env_step"] += t5 - t4
        t["trajectory_write"] += t6 - t5
        total_steps += 1

    return dict(t), total_steps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RL rollout profiler.")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    device = _resolve_device(args.device)
    torch.manual_seed(0)
    encoder = ObservationEncoder()
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (256, 256)).to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    total_steps = 0
    agg = Counter()
    t_start = time.perf_counter()
    for e in range(args.episodes):
        env = MahjongEnv(seed=e)
        times, steps = profile_rollout(model, encoder, env, max_steps=args.max_steps, device=device)
        agg.update(times)
        total_steps += steps
    wall = time.perf_counter() - t_start

    total = sum(agg.values())
    report = {
        "device": str(device),
        "episodes": args.episodes,
        "total_env_steps": total_steps,
        "wall_seconds": round(wall, 2),
        "steps_per_sec": round(total_steps / wall, 1),
        "games_per_hour": round(args.episodes / wall * 3600, 1),
        "stage_seconds": {k: round(v, 4) for k, v in agg.items()},
        "stage_share": {k: round(v / total, 4) for k, v in agg.items()},
    }
    if device.type == "cuda":
        torch.cuda.synchronize()
        report["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
        report["gpu_memory_max_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
