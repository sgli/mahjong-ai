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
    ap.add_argument("--bc-anchor-beta", type=float, help="BC anchor β（默认从 config 读，0.0=关闭）")
    ap.add_argument("--bc-anchor-checkpoint", help="BC anchor checkpoint（冻结的 π_BC）")
    ap.add_argument("--rollout-version", help="rollout-v1（reference，默认）| rollout-v2（optimized batch）")
    ap.add_argument("--attribution", action="store_true", help="启用 per-opponent/per-seat 归因埋点（§4/§11）")
    ap.add_argument("--diagnose", action="store_true", default=None, help="启用 update-epoch KL/ratio/clip 分项 + 分布统计（§7；默认从 config training.diagnose）")
    ap.add_argument("--no-diagnose", action="store_false", dest="diagnose", help="显式关闭 diagnose")
    ap.add_argument("--rng-version", help="rng-v1（全局流，默认）| rng-v2（per-env，sampling 逐位对齐）")
    ap.add_argument("--warmup-steps", type=int, help="critic 预热步数（默认 0=关；预热期只训 value）")
    ap.add_argument("--warmup-freeze-trunk", action="store_true", default=None, help="预热期冻结 trunk+action_head（严格 actor 冻结）")
    ap.add_argument("--reward-clip", type=float, help="per-step reward winsorize 阈值（默认 0=关；仅尺度处理）")
    ap.add_argument("--reward-score-scale", type=float, help="覆盖 reward score 项系数（默认 null=用 reward-v1 的 0.001）")
    ap.add_argument("--value-separate-trunk", action="store_true", default=False, help="value 使用独立 trunk（默认 false=共享）")
    args = ap.parse_args(argv)

    cfg = _load_config(Path(args.config))
    model_cfg = cfg.get("model") or {}
    train_cfg = cfg.get("training") or {}

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    hidden_sizes = tuple(model_cfg.get("hidden_sizes", [128, 128]))
    value_separate_trunk = bool(model_cfg.get("value_separate_trunk", False)) if not args.value_separate_trunk else True
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 3)
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch is not None else train_cfg.get("steps_per_epoch", 400)
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 0)
    device = torch.device(train_cfg.get("device", "cpu"))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    _seed_everything(seed)

    encoder = ObservationEncoder()
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, hidden_sizes, value_separate_trunk=value_separate_trunk).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 0.0005))

    config = {
        "model": {"hidden_sizes": list(hidden_sizes), "value_separate_trunk": value_separate_trunk},
        "feature_dim": FEATURE_DIM,
        "action_space_size": ACTION_SPACE_SIZE,
        "training": {k: v for k, v in train_cfg.items() if k != "device"},
    }
    env_seed = train_cfg.get("env_seed", 0)

    # BC anchor（实验 D）：β 默认 0（等价纯 PPO）；β>0 时加载冻结的 π_BC
    bc_anchor_beta = args.bc_anchor_beta if args.bc_anchor_beta is not None else train_cfg.get("bc_anchor_beta", 0.0)
    bc_anchor_ckpt = args.bc_anchor_checkpoint or train_cfg.get("bc_anchor_checkpoint") or cfg.get("bc_checkpoint")
    bc_model = None
    if bc_anchor_beta > 0.0:
        if not bc_anchor_ckpt or not Path(bc_anchor_ckpt).exists():
            raise SystemExit(f"bc anchor checkpoint not found: {bc_anchor_ckpt}")
        from mahjong.evaluation import load_policy

        _, bc_model = load_policy(bc_anchor_ckpt, device)
        bc_model.eval()
        for p in bc_model.parameters():
            p.requires_grad_(False)
        log.info("BC anchor enabled: beta=%s checkpoint=%s", bc_anchor_beta, bc_anchor_ckpt)
    config["training"]["bc_anchor_beta"] = bc_anchor_beta
    config["training"]["bc_anchor_checkpoint"] = bc_anchor_ckpt

    # Opponent pool（§12/§13）：配置了 opponent_pool_config 才启用（默认 None = self-play）
    opponent_pool = None
    opponent_checkpoint_map = None
    current_ppo = None
    opponent_pool_config = cfg.get("opponent_pool_config")
    if opponent_pool_config:
        from mahjong.training import TrainingOpponentPool, register_historical_checkpoints

        pool_seed = train_cfg.get("opponent_pool_seed", seed)
        opponent_pool = TrainingOpponentPool({}, dict(opponent_pool_config), seed=pool_seed)
        log.info("Opponent pool enabled: %s", opponent_pool_config)

        # checkpoint_map：ppo-v1 + historical:*（只读）
        opp_ckpts = cfg.get("opponent_checkpoints") or {}
        opponent_checkpoint_map = {}
        if "ppo-v1" in opp_ckpts:
            opponent_checkpoint_map["ppo-v1"] = opp_ckpts["ppo-v1"]
        for entry in opp_ckpts.get("historical") or []:
            p = Path(entry)
            if p.is_dir():
                opponent_checkpoint_map.update(register_historical_checkpoints(p))
            else:
                opponent_checkpoint_map[f"historical:{p.stem}"] = str(p)

        # current-ppo：初始快照（冻结当前策略；未实现周期性刷新，文档标注）
        if "current-ppo" in opponent_pool_config:
            import copy

            current_ppo = (copy.deepcopy(model), encoder)
            log.info("current-ppo snapshot created (initial, no periodic refresh)")

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
        bc_model=bc_model,
        bc_anchor_beta=bc_anchor_beta,
        rollout_version=args.rollout_version or train_cfg.get("rollout_version", "rollout-v1"),
        attribution=args.attribution or train_cfg.get("attribution", False),
        diagnose=args.diagnose if args.diagnose is not None else train_cfg.get("diagnose", True),
        opponent_pool=opponent_pool,
        opponent_bc_checkpoint=cfg.get("bc_checkpoint"),
        opponent_checkpoint_map=opponent_checkpoint_map,
        current_ppo=current_ppo,
        opponent_mode=train_cfg.get("opponent_mode", "sampling"),
        opponent_temperature=train_cfg.get("opponent_temperature", 1.0),
        opponent_pool_seed=train_cfg.get("opponent_pool_seed", seed),
        rng_version=args.rng_version or train_cfg.get("rng_version", "rng-v1"),
        warmup_steps=args.warmup_steps if args.warmup_steps is not None else train_cfg.get("warmup_steps", 0),
        warmup_freeze_trunk=args.warmup_freeze_trunk if args.warmup_freeze_trunk is not None else train_cfg.get("warmup_freeze_trunk", True),
        reward_clip=args.reward_clip if args.reward_clip is not None else train_cfg.get("reward_clip", 0.0),
        reward_score_scale=args.reward_score_scale if args.reward_score_scale is not None else train_cfg.get("reward_score_scale"),
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
