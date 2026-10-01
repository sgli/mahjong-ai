#!/usr/bin/env python
"""Random self-play: verify the environment can deterministically finish a hanchan.

Usage::

    python scripts/play_selfplay.py --games 20 --seed 0
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mahjong.decision.action import ActionType  # noqa: E402
from mahjong.environment import MahjongEnv  # noqa: E402


def play_one(seed: int) -> tuple[int, list[int], list[int]]:
    env = MahjongEnv(seed=seed)
    env.reset()
    rng = random.Random(seed)
    steps = 0
    while not env.done():
        legal = env.legal_actions()
        # prefer PASS in response windows so the game keeps flowing; otherwise random
        action = next((a for a in legal if getattr(a, "type", None) is ActionType.PASS), None)
        if action is None:
            action = rng.choice(legal)
        env.step(action)
        steps += 1
        if steps > 100_000:
            raise RuntimeError("too many steps")
    return steps, env.state.scores, env.state.final_ranks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Random self-play smoke test.")
    ap.add_argument("--games", type=int, default=20, help="number of games to play")
    ap.add_argument("--seed", type=int, default=0, help="starting seed")
    args = ap.parse_args(argv)

    total_steps = 0
    for g in range(args.games):
        seed = args.seed + g
        steps, scores, ranks = play_one(seed)
        total_steps += steps
        print(f"game {g} (seed {seed}): steps={steps} scores={scores} ranks={ranks}")
    print(f"played {args.games} games, avg steps={total_steps / args.games:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
