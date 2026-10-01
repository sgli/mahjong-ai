#!/usr/bin/env python
"""Run a benchmark + promotion flow for a candidate checkpoint.

Usage::

    python scripts/benchmark.py --config configs/benchmark.yaml
    python scripts/benchmark.py --games 10 --seed 1
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

from mahjong.evaluation import OpponentPool, PolicyOpponent, RandomOpponent, run_promotion  # noqa: E402

log = logging.getLogger("benchmark")

_DEFAULT_CONFIG = _HERE.parent / "configs" / "benchmark.yaml"


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def _build_pool(pool_cfg: dict) -> OpponentPool:
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
            )
        )
    return pool


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark + promotion flow.")
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--games", type=int, help="override benchmark games")
    ap.add_argument("--seed", type=int, help="override seed")
    args = ap.parse_args(argv)

    cfg = _load_config(Path(args.config))
    games = args.games if args.games is not None else cfg.get("games", 10)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    random.seed(seed)
    torch.manual_seed(seed)

    pool = _build_pool(cfg.get("pool") or {})
    candidate = pool.get(cfg["candidate_id"])
    opponents = [pool.get(oid) for oid in cfg["opponent_ids"]]
    baseline = pool.get(cfg["baseline_id"]) if cfg.get("baseline_id") else None

    result = run_promotion(
        candidate,
        opponents,
        baseline=baseline,
        games=games,
        seed=seed,
        candidate_seat=cfg.get("candidate_seat", 0),
        rules_games=cfg.get("rules_games", 2),
        output_dir=cfg.get("output_dir", "experiments/benchmark"),
    )

    print("== benchmark result ==")
    print(f"candidate:        {result.candidate_id}@{result.candidate_version}")
    print(f"opponents:        {result.opponents}")
    print(f"games:            {result.games}  seed: {result.seed}")
    print(f"smoke/rules:      {result.smoke_passed}/{result.rules_passed}")
    print(f"illegal_rate:     {result.illegal_rate}")
    print(f"metrics:          {json.dumps(result.metrics)}")
    if result.baseline_metrics is not None:
        print(f"baseline metrics: {json.dumps(result.baseline_metrics)}")
        print(f"promoted:         {result.promoted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
