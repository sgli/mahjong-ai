#!/usr/bin/env python
"""Train a PPO policy (self-play) on the Mahjong Environment (CPU small scale).

Usage::

    python scripts/train_ppo.py --config configs/train_ppo.yaml
    python scripts/train_ppo.py --epochs 2 --steps-per-epoch 200
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

from mahjong.dataset.manifest import git_commit  # noqa: E402
from mahjong.decision.action import Action  # noqa: E402
from mahjong.environment import MahjongEnv  # noqa: E402
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder  # noqa: E402
from mahjong.model import MLPPolicy, remap_bc_state_dict  # noqa: E402
from mahjong.training import load_checkpoint, save_checkpoint, train_ppo  # noqa: E402

log = logging.getLogger("train_ppo")

_DEFAULT_CONFIG = _HERE.parent / "configs" / "train_ppo.yaml"


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, encoder, device, games: int, seed: int) -> dict:
    """Play N self-play games and report mean reward / mean rank."""
    rewards = []
    ranks = []
    for g in range(games):
        env = MahjongEnv(seed=seed + 10_000 + g)
        env.reset()
        while not env.done():
            seat = env.state.turn
            legal = [a for a in env.legal_actions(seat) if isinstance(a, Action)]
            obs = env.observation(seat)
            features = encoder.encode(obs).to(device)
            from mahjong.features.action_space import action_to_id, legal_mask

            mask = legal_mask(legal).to(device)
            with torch.no_grad():
                out = model(features.unsqueeze(0), mask.unsqueeze(0))
                probs = torch.softmax(out.logits, -1)
                action_id = int(torch.multinomial(probs, 1).squeeze(-1).item())
            action = next(a for a in legal if action_to_id(a) == action_id)
            env.step(action)
        for s in range(4):
            rewards.append(env.reward(s))
            ranks.append(env.state.final_ranks[s])
    return {
        "mean_reward": sum(rewards) / len(rewards),
        "mean_rank": sum(ranks) / len(ranks),
        "games": games,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train a PPO policy (self-play).")
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG), help="path to train_ppo.yaml")
    ap.add_argument("--epochs", type=int, help="override epochs")
    ap.add_argument("--steps-per-epoch", type=int, help="override steps_per_epoch")
    ap.add_argument("--seed", type=int, help="override seed")
    args = ap.parse_args(argv)

    cfg = _load_config(Path(args.config))
    model_cfg = cfg.get("model") or {}
    train_cfg = cfg.get("training") or {}

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    hidden_sizes = tuple(model_cfg.get("hidden_sizes", [128, 128]))
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 3)
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch is not None else train_cfg.get("steps_per_epoch", 400)
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 0)
    device = torch.device(train_cfg.get("device", "cpu"))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    _seed_everything(seed)

    encoder = ObservationEncoder()
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, hidden_sizes).to(device)

    bc_path = cfg.get("bc_checkpoint")
    if bc_path and Path(bc_path).exists():
        bc = load_checkpoint(bc_path)
        model.load_state_dict(remap_bc_state_dict(bc["state_dict"]), strict=False)
        log.info("loaded BC checkpoint %s (value head randomly initialised)", bc_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 0.0005))

    history = train_ppo(
        model,
        encoder,
        optimizer,
        num_envs=train_cfg.get("num_envs", 2),
        env_seed=train_cfg.get("env_seed", 0),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=train_cfg.get("batch_size", 64),
        update_epochs=train_cfg.get("update_epochs", 4),
        gamma=train_cfg.get("gamma", 0.99),
        lam=train_cfg.get("lam", 0.95),
        clip_eps=train_cfg.get("clip_eps", 0.2),
        vf_coef=train_cfg.get("vf_coef", 0.5),
        ent_coef=train_cfg.get("ent_coef", 0.01),
        device=device,
        max_episode_steps=train_cfg.get("max_episode_steps", 5000),
    )

    eval_metrics = evaluate(model, encoder, device, games=cfg.get("eval_games", 5), seed=seed)

    config = {
        "model": {"hidden_sizes": list(hidden_sizes)},
        "feature_dim": FEATURE_DIM,
        "action_space_size": ACTION_SPACE_SIZE,
        "training": {k: v for k, v in train_cfg.items() if k not in ("device",)},
    }
    metrics = {"history": history, "evaluation": eval_metrics}
    git_hash = git_commit(_HERE.parent)
    save_checkpoint(
        checkpoint_dir,
        model.state_dict(),
        config=config,
        metrics=metrics,
        dataset_version=cfg.get("dataset_version", "decision-v1"),
        feature_version=cfg.get("feature_version", "feature-v1"),
        action_schema_version=cfg.get("action_schema_version", "action-v1"),
        environment_version=cfg.get("environment_version", "none"),
        git_commit_hash=git_hash,
        training_step=epochs,
    )

    print("== PPO training summary ==")
    print(f"epochs:              {epochs}")
    print(f"seed:                {seed}")
    print(f"final epoch metrics: {json.dumps(history[-1]) if history else '{}'}")
    print(f"evaluation:          {json.dumps(eval_metrics)}")
    print(f"checkpoint:          {checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
