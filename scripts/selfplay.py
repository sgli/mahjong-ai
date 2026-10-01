#!/usr/bin/env python
"""Run self-play matches between pool opponents.

Usage::

    python scripts/selfplay.py --config configs/selfplay.yaml
    python scripts/selfplay.py --games 3 --seed 1 --matches all_random ppo_vs_random
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402
import yaml  # noqa: E402

from mahjong.evaluation import OpponentPool, PolicyOpponent, RandomOpponent, run_round, summarize  # noqa: E402

log = logging.getLogger("selfplay")

_DEFAULT_CONFIG = _HERE.parent / "configs" / "selfplay.yaml"


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def _build_pool(pool_cfg: dict, device: str) -> OpponentPool:
    pool = OpponentPool()
    for oid, spec in (pool_cfg or {}).items():
        if spec.get("type") == "random":
            pool.add(RandomOpponent(opponent_id=oid, model_version=spec.get("model_version", "random-v1")))
            continue
        checkpoint = spec.get("checkpoint")
        if not checkpoint or not Path(checkpoint).exists():
            log.warning("skipping opponent %r: checkpoint %r not found", oid, checkpoint)
            continue
        pool.add(
            PolicyOpponent(
                opponent_id=oid,
                type=spec.get("type", "policy"),
                model_version=spec.get("model_version", f"{oid}-v1"),
                checkpoint=checkpoint,
                device=device,
            )
        )
    return pool


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Self-play matches between pool opponents.")
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--games", type=int, help="override games per match")
    ap.add_argument("--seed", type=int, help="override seed")
    ap.add_argument("--matches", nargs="*", help="only run these match names")
    args = ap.parse_args(argv)

    cfg = _load_config(Path(args.config))
    games = args.games if args.games is not None else cfg.get("games", 5)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    device = cfg.get("device", "cpu")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    random.seed(seed)
    torch.manual_seed(seed)

    pool = _build_pool(cfg.get("pool") or {}, device)
    matches = cfg.get("matches") or []
    if args.matches:
        wanted = set(args.matches)
        matches = [m for m in matches if m.get("name") in wanted]

    print(f"== self-play: {len(matches)} match(es), {games} game(s) each, seed {seed} ==")
    overall = {"matches": []}
    for match in matches:
        name = match.get("name", "?")
        seats = match.get("seats")
        if len(seats) != 4:
            log.error("match %r must have exactly 4 seats", name)
            continue
        try:
            opponents = [pool.get(oid) for oid in seats]
        except KeyError as exc:
            log.error("match %r references unknown opponent: %s", name, exc)
            continue

        results = run_round(opponents, games=games, seed=seed + len(overall["matches"]) * 1000)
        summary = summarize(results)
        print(f"match {name}: {json.dumps(summary)}")
        overall["matches"].append(
            {
                "name": name,
                "seats": seats,
                "opponents": {s: f"{o.opponent_id}@{o.model_version}" for s, o in enumerate(opponents)},
                "summary": summary,
                "games": [
                    {"seed": r.seed, "opponents": r.opponents, "scores": r.final_scores, "ranks": r.final_ranks, "error": r.error}
                    for r in results
                ],
            }
        )
        if any(r.error for r in results):
            log.warning("match %r had %d game error(s)", name, sum(1 for r in results if r.error))

    out = Path(cfg.get("output", "experiments/selfplay_results.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
