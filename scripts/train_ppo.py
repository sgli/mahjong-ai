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
from mahjong.environment import ENVIRONMENT_VERSION, MahjongEnv  # noqa: E402
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder  # noqa: E402
from mahjong.model import MLPPolicy  # noqa: E402
from mahjong.training import train_ppo  # noqa: E402
from mahjong.training.ppo import init_ppo_from_bc  # noqa: E402
from mahjong.training.ppo_checkpoint import (  # noqa: E402
    capture_ppo_rng_states,
    load_ppo_checkpoint,
    restore_ppo_rng_states,
    save_ppo_checkpoint,
)

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
    ap.add_argument("--resume", help="resume from a PPO checkpoint (latest.pt / epoch_N.pt)")
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
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 0.0005))

    config = {
        "model": {"hidden_sizes": list(hidden_sizes)},
        "feature_dim": FEATURE_DIM,
        "action_space_size": ACTION_SPACE_SIZE,
        "training": {k: v for k, v in train_cfg.items() if k != "device"},
    }
    env_seed = train_cfg.get("env_seed", 0)

    start_epoch = 0
    global_step = 0
    best_metric = float("inf")
    best_metric_name = "mean_reward"  # 默认按 mean_reward 取最佳（更大更好则需反向；这里用 value_loss 语义）
    prev_history: list[dict] = []

    if args.resume:
        ckpt = load_ppo_checkpoint(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch") or 0
        global_step = ckpt.get("global_step") or 0
        best_metric = ckpt.get("best_metric") if ckpt.get("best_metric") is not None else float("inf")
        best_metric_name = ckpt.get("best_metric_name") or "mean_reward"
        prev_history = ckpt.get("metrics_history") or []
        restore_ppo_rng_states(ckpt)
        # resume 时沿用 checkpoint 的 config（若存在）覆盖当前 config
        if ckpt.get("config"):
            config = ckpt["config"]
        log.info("resume from %s: epoch=%d global_step=%d", args.resume, start_epoch, global_step)
    else:
        bc_path = cfg.get("bc_checkpoint")
        if bc_path and Path(bc_path).exists():
            from mahjong.training import load_training_checkpoint

            bc_ckpt = load_training_checkpoint(bc_path, map_location="cpu")
            init_ppo_from_bc(model, bc_ckpt.get("state_dict") or {})
            log.info("initialised from BC checkpoint %s (value head re-initialised)", bc_path)

    best_epoch = start_epoch
    snapshot_paths: dict[int, Path] = {}

    def on_epoch_end(epoch: int, epoch_metrics: dict) -> None:
        nonlocal best_epoch, best_metric
        snap = checkpoint_dir / f"epoch_{epoch}.pt"
        save_ppo_checkpoint(
            snap,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch,
            global_step=epoch_metrics.get("global_step", global_step),
            best_metric=best_metric,
            best_metric_name=best_metric_name,
            config=config,
            metrics_history=prev_history + [epoch_metrics],
            seed=seed,
            env_seed=env_seed,
            feature_version=cfg.get("feature_version", "feature-v1"),
            action_schema_version=cfg.get("action_schema_version", "action-v1"),
            environment_version=cfg.get("environment_version", ENVIRONMENT_VERSION),
            reward_version=cfg.get("reward_version", "reward-v1"),
            model_version=cfg.get("model_version", "ppo-v1"),
            git_commit_hash=git_commit(_HERE.parent),
            rng_states=capture_ppo_rng_states(),
        )
        snapshot_paths[epoch] = snap
        # mean_reward 越大越好；其它指标按小更好（这里固定 mean_reward 越大越好）
        metric_val = epoch_metrics.get("mean_reward", -float("inf"))
        if metric_val > best_metric:
            best_metric = metric_val
            best_epoch = epoch

    history = train_ppo(
        model,
        encoder,
        optimizer,
        num_envs=train_cfg.get("num_envs", 2),
        env_seed=env_seed,
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
        start_epoch=start_epoch,
        global_step=global_step,
        on_epoch_end=on_epoch_end,
    )

    history = prev_history + history
    eval_metrics = evaluate(model, encoder, device, games=cfg.get("eval_games", 5), seed=seed)

    # latest.pt（最终状态）+ best.pt（最佳 epoch 快照）
    final_epoch = history[-1]["epoch"] if history else 0
    final_step = history[-1]["global_step"] if history else global_step
    save_ppo_checkpoint(
        checkpoint_dir / "latest.pt",
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        epoch=final_epoch,
        global_step=final_step,
        best_metric=best_metric,
        best_metric_name=best_metric_name,
        config=config,
        metrics_history=history,
        seed=seed,
        env_seed=env_seed,
        feature_version=cfg.get("feature_version", "feature-v1"),
        action_schema_version=cfg.get("action_schema_version", "action-v1"),
        environment_version=cfg.get("environment_version", ENVIRONMENT_VERSION),
        reward_version=cfg.get("reward_version", "reward-v1"),
        model_version=cfg.get("model_version", "ppo-v1"),
        git_commit_hash=git_commit(_HERE.parent),
        rng_states=capture_ppo_rng_states(),
    )
    import shutil

    if best_epoch in snapshot_paths:
        shutil.copyfile(snapshot_paths[best_epoch], checkpoint_dir / "best.pt")
    log.info("saved latest.pt / best.pt (best_epoch=%d metric=%s)", best_epoch, best_metric_name)

    print("== PPO training summary ==")
    print(f"epochs:              {epochs}")
    print(f"seed:                {seed}")
    print(f"final epoch metrics: {json.dumps(history[-1]) if history else '{}'}")
    print(f"evaluation:          {json.dumps(eval_metrics)}")
    print(f"checkpoint:          {checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
