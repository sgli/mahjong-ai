#!/usr/bin/env python
"""Train a PPO policy (self-play) on the Mahjong Environment (CPU small scale).

Usage::

    python scripts/train_ppo.py --config <ppo-config.yaml>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import random
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402
import yaml  # noqa: E402

from mahjong.decision.action import Action, ActionType  # noqa: E402
from mahjong.environment import ENVIRONMENT_VERSION, MahjongEnv  # noqa: E402
from mahjong.features.policy_schema import policy_action_ops, resolve_policy_schema  # noqa: E402
from mahjong.model import build_policy_model  # noqa: E402
from mahjong.training import train_ppo  # noqa: E402
from mahjong.training.ppo import init_ppo_from_bc  # noqa: E402
from mahjong.training.ppo_checkpoint import (  # noqa: E402
    capture_ppo_rng_states,
    load_ppo_checkpoint,
    restore_ppo_rng_states,
    save_ppo_checkpoint,
)

log = logging.getLogger("train_ppo")

def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def _guard_fresh_checkpoint_dir(path: Path, *, feature_version: str, resume: str | None) -> None:
    """Never overwrite an existing versioned PPO run with a fresh invocation."""
    if (feature_version in {"feature-v2", "feature-v3"} and not resume and path.is_dir()
            and ((path / "resolved_config.json").exists() or any(path.glob("*.pt")))):
        raise FileExistsError(f"{feature_version} PPO checkpoint directory already contains a run: {path}; use a new path or --resume")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _sha256(path: str | Path) -> str | None:
    p = Path(path)
    if p.is_dir():
        p = p / "best.pt" if (p / "best.pt").is_file() else p / "latest.pt"
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _git_identity(repo: Path) -> dict:
    safe = f"safe.directory={repo.resolve()}"
    def run(args):
        try:
            return subprocess.run(["git", "-c", safe, *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": run(["rev-parse", "HEAD"]), "dirty": bool(run(["status", "--porcelain"]))}


def _resume_signature(config: dict) -> dict:
    training = dict(config.get("training") or {})
    # Increasing the target epoch count is a normal continuation; all other
    # training-affecting fields must remain identical.
    training.pop("epochs", None)
    return {
        "model": config.get("model"),
        "training": training,
        "feature_version": config.get("feature_version"),
        "action_schema_version": config.get("action_schema_version"),
        "environment_version": config.get("environment_version"),
        "reward_version": config.get("reward_version"),
        "checkpoint_selection": config.get("checkpoint_selection"),
        "bc_checkpoint_sha256": (config.get("provenance") or {}).get("bc_checkpoint_sha256"),
        "bc_anchor_checkpoint_sha256": (config.get("provenance") or {}).get("bc_anchor_checkpoint_sha256"),
        "opponent_pool_config": config.get("opponent_pool_config"),
        "opponent_checkpoints_resolved": config.get("opponent_checkpoints_resolved"),
        "opponent_checkpoint_sha256": (config.get("provenance") or {}).get("opponent_checkpoint_sha256"),
        "git": (config.get("provenance") or {}).get("git"),
    }


def evaluate(model, encoder, device, games: int, seed: int) -> dict:
    """Play N self-play games and report mean reward / mean rank."""
    to_id, make_mask, allow_abort = policy_action_ops(encoder, model)
    rewards = []
    ranks = []
    for g in range(games):
        env = MahjongEnv(seed=seed + 10_000 + g)
        env.reset()
        while not env.done():
            seat = env.state.turn
            legal = [a for a in env.legal_actions(seat) if isinstance(a, Action) and (a.type is not ActionType.KYUUSHU_KYUHAI or allow_abort)]
            obs = env.observation(seat)
            features = encoder.encode(obs).to(device)
            mask = make_mask(legal).to(device)
            with torch.no_grad():
                out = model(features.unsqueeze(0), mask.unsqueeze(0))
                probs = torch.softmax(out.logits, -1)
                action_id = int(torch.multinomial(probs, 1).squeeze(-1).item())
            action = next(a for a in legal if to_id(a) == action_id)
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
    ap.add_argument("--config", type=Path, required=True, help="PPO config YAML")
    ap.add_argument("--epochs", type=int, help="override epochs")
    ap.add_argument("--steps-per-epoch", type=int, help="override steps_per_epoch")
    ap.add_argument("--seed", type=int, help="override seed")
    ap.add_argument("--resume", help="resume from a PPO checkpoint (latest.pt / epoch_N.pt)")
    ap.add_argument("--allow-config-override", action="store_true", help="explicitly allow a resume with changed training configuration")
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

    feature_version = cfg.get("feature_version", "feature-v1")
    action_version = cfg.get("action_schema_version", "action-v1")
    encoder, feature_dim, action_size = resolve_policy_schema(feature_version, action_version)
    checkpoint_selection = cfg.get("checkpoint_selection", "latest_epoch" if feature_version in {"feature-v2", "feature-v3"} else "legacy_mean_reward")
    if feature_version in {"feature-v2", "feature-v3"} and checkpoint_selection != "latest_epoch":
        raise ValueError("versioned PPO checkpoints must use latest_epoch until independent game selection")
    if feature_version in {"feature-v2", "feature-v3"} and cfg.get("environment_version") != ENVIRONMENT_VERSION:
        raise ValueError("versioned PPO config environment_version must match the current environment")
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    _guard_fresh_checkpoint_dir(checkpoint_dir, feature_version=feature_version, resume=args.resume)
    hidden_sizes = tuple(model_cfg.get("hidden_sizes", [128, 128]))
    value_separate_trunk = bool(model_cfg.get("value_separate_trunk", False)) if not args.value_separate_trunk else True
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 3)
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch is not None else train_cfg.get("steps_per_epoch", 400)
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 0)
    device = torch.device(train_cfg.get("device", "cpu"))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    _seed_everything(seed)

    model = build_policy_model(
        feature_version, feature_dim, action_size, hidden_sizes,
        model_config=model_cfg,
        value_separate_trunk=value_separate_trunk,
    ).to(device)
    model.action_schema_version = action_version
    model.allow_kyuushu = bool(cfg.get("allow_kyuushu", False)) if action_version == "action-v2" else False
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 0.0005))

    config = copy.deepcopy(cfg)
    config["checkpoint_selection"] = checkpoint_selection
    config["environment_version"] = ENVIRONMENT_VERSION
    config["model"] = {**model_cfg, "hidden_sizes": list(hidden_sizes), "value_separate_trunk": value_separate_trunk}
    config["feature_dim"] = feature_dim
    config["action_space_size"] = action_size
    config["training"] = {**train_cfg, "epochs": epochs, "steps_per_epoch": steps_per_epoch, "seed": seed}
    config["training"]["step_budget_unit"] = train_cfg.get("step_budget_unit", "environment_steps")
    env_seed = train_cfg.get("env_seed", 0)

    # BC anchor（实验 D）：β 默认 0（等价纯 PPO）；β>0 时加载冻结的 π_BC
    bc_anchor_beta = args.bc_anchor_beta if args.bc_anchor_beta is not None else train_cfg.get("bc_anchor_beta", 0.0)
    call_anchor_beta = float(train_cfg.get("call_anchor_beta", 0.0))
    if call_anchor_beta < 0:
        raise ValueError('call_anchor_beta must be nonnegative')
    bc_anchor_ckpt = args.bc_anchor_checkpoint or train_cfg.get("bc_anchor_checkpoint") or cfg.get("bc_checkpoint")
    bc_model = None
    if bc_anchor_beta > 0.0 or call_anchor_beta > 0.0:
        if not bc_anchor_ckpt or not Path(bc_anchor_ckpt).exists():
            raise SystemExit(f"bc anchor checkpoint not found: {bc_anchor_ckpt}")
        from mahjong.evaluation import load_policy

        _, bc_model = load_policy(bc_anchor_ckpt, device)
        if (bc_model.feature_dim != feature_dim or bc_model.action_size != action_size
                or bc_model.action_schema_version != action_version
                or bool(bc_model.allow_kyuushu) != bool(model.allow_kyuushu)):
            raise ValueError("BC anchor checkpoint schema differs from PPO policy schema")
        bc_model.eval()
        # Anchor the same temperature-scaled policy that generates learner data.
        bc_model.policy_temperature = model.policy_temperature
        for p in bc_model.parameters():
            p.requires_grad_(False)
        log.info("BC anchor enabled: beta=%s checkpoint=%s", bc_anchor_beta, bc_anchor_ckpt)
    config["training"]["bc_anchor_beta"] = bc_anchor_beta
    config["training"]["call_anchor_beta"] = call_anchor_beta
    config["training"]["bc_anchor_checkpoint"] = bc_anchor_ckpt

    # Opponent pool（§12/§13）：resolve every configured id before starting training.
    opponent_pool = None
    opponent_checkpoint_map = None
    current_ppo = None
    opponent_pool_config = dict(cfg.get("opponent_pool_config") or {})
    if opponent_pool_config:
        from mahjong.training import TrainingOpponentPool, register_historical_checkpoints

        pool_seed = train_cfg.get("opponent_pool_seed", seed)
        # checkpoint_map：ppo-v1 + stable historical ids (read-only)
        opp_ckpts = cfg.get("opponent_checkpoints") or {}
        opponent_checkpoint_map = {}
        if "ppo-v1" in opp_ckpts:
            p = Path(opp_ckpts["ppo-v1"])
            if not p.is_file():
                raise SystemExit(f"ppo-v1 opponent checkpoint not found: {p}")
            opponent_checkpoint_map["ppo-v1"] = str(p)
        for entry in opp_ckpts.get("historical") or []:
            p = Path(entry)
            if p.is_dir():
                opponent_checkpoint_map.update(register_historical_checkpoints(p))
            else:
                if not p.is_file():
                    raise SystemExit(f"historical opponent checkpoint not found: {p}")
                opponent_checkpoint_map[f"historical:{p.parent.name if p.name in ('latest.pt', 'best.pt') else p.stem}"] = str(p)

        # Allow the fixed BC opponents to be distinct from the BC initializer.
        # Phase 10.2 uses both BC-v3 (historical anchor) and BC-v4 (initializer).
        for opponent_id in ("bc-v3", "bc-v4"):
            if opponent_id not in opp_ckpts:
                continue
            p = Path(opp_ckpts[opponent_id])
            if not p.is_file():
                raise SystemExit(f"{opponent_id} opponent checkpoint not found: {p}")
            opponent_checkpoint_map[opponent_id] = str(p.resolve())

        # Expand a legacy historical bucket into explicit stable checkpoint IDs.
        if "historical" in opponent_pool_config:
            historical_ids = sorted(k for k in opponent_checkpoint_map if k.startswith("historical:"))
            if not historical_ids:
                raise SystemExit("opponent pool has nonzero historical weight but no historical checkpoints resolved")
            weight = float(opponent_pool_config.pop("historical"))
            for opponent_id in historical_ids:
                opponent_pool_config[opponent_id] = opponent_pool_config.get(opponent_id, 0.0) + weight / len(historical_ids)

        available_ids = {"random-v1", "rule-v1", "bc-v3" if feature_version in {"feature-v2", "feature-v3"} else "bc-v2.1"}
        if feature_version == "feature-v3":
            available_ids.add("bc-v4")
        if feature_version in {"feature-v2", "feature-v3"} and any(k.startswith("historical:") or k == "ppo-v1" for k in opponent_pool_config if opponent_pool_config[k] > 0):
            raise ValueError("versioned PPO pool cannot use legacy PPO-v1 or unvalidated historical opponents")
        if feature_version == "feature-v1" and "ppo-v1" in opponent_checkpoint_map:
            available_ids.add("ppo-v1")
        if feature_version == "feature-v1":
            available_ids.update(k for k in opponent_checkpoint_map if k.startswith("historical:"))
        if "current-ppo" in opponent_pool_config:
            available_ids.add("current-ppo")
        unresolved = [k for k, weight in opponent_pool_config.items() if weight > 0 and k not in available_ids]
        if unresolved:
            raise SystemExit(f"opponent pool has unresolved nonzero ids: {unresolved}; available={sorted(available_ids)}")
        opponent_pool = TrainingOpponentPool({}, opponent_pool_config, seed=pool_seed)
        log.info("Opponent pool enabled (resolved): %s", opponent_pool_config)

        # current-ppo：初始快照（冻结当前策略；未实现周期性刷新，文档标注）
        if "current-ppo" in opponent_pool_config:
            current_ppo = (copy.deepcopy(model), encoder)
            log.info("current-ppo snapshot created (initial, no periodic refresh)")

    # Store the effective runtime settings and immutable input identities, not
    # only the subset of the YAML previously copied into checkpoints.
    rollout_version = args.rollout_version or train_cfg.get("rollout_version", "rollout-v1")
    rng_version = args.rng_version or train_cfg.get("rng_version", "rng-v1")
    diagnose = args.diagnose if args.diagnose is not None else train_cfg.get("diagnose", True)
    attribution = args.attribution or train_cfg.get("attribution", False)
    reward_clip = args.reward_clip if args.reward_clip is not None else train_cfg.get("reward_clip", 0.0)
    reward_score_scale = args.reward_score_scale if args.reward_score_scale is not None else train_cfg.get("reward_score_scale")
    config["training"].update({
        "device": str(device), "bc_anchor_beta": bc_anchor_beta, "call_anchor_beta": call_anchor_beta,
        "bc_anchor_checkpoint": str(Path(bc_anchor_ckpt).resolve()) if bc_anchor_ckpt else None,
        "rollout_version": rollout_version, "rng_version": rng_version,
        "diagnose": diagnose, "attribution": attribution,
        "reward_clip": reward_clip, "reward_score_scale": reward_score_scale,
        "warmup_steps": args.warmup_steps if args.warmup_steps is not None else train_cfg.get("warmup_steps", 0),
        "warmup_freeze_trunk": args.warmup_freeze_trunk if args.warmup_freeze_trunk is not None else train_cfg.get("warmup_freeze_trunk", True),
    })
    config["opponent_pool_config"] = opponent_pool_config or None
    config["opponent_checkpoints_resolved"] = opponent_checkpoint_map or {}
    dataset_manifest = _HERE.parent / "data" / "processed" / str(cfg.get("dataset_version", "decision-v2")) / "manifest.json"
    config["provenance"] = {
        "config_file": str(Path(args.config).resolve()),
        "config_sha256": _sha256(args.config),
        "bc_init_checkpoint": str(Path(cfg["bc_checkpoint"]).resolve()) if cfg.get("bc_checkpoint") else None,
        "bc_checkpoint_sha256": _sha256(cfg.get("bc_checkpoint")) if cfg.get("bc_checkpoint") else None,
        "bc_anchor_checkpoint": str(Path(bc_anchor_ckpt).resolve()) if bc_anchor_ckpt else None,
        "bc_anchor_checkpoint_sha256": _sha256(bc_anchor_ckpt) if bc_anchor_ckpt else None,
        "dataset_manifest": str(dataset_manifest.resolve()) if dataset_manifest.is_file() else None,
        "dataset_manifest_sha256": _sha256(dataset_manifest) if dataset_manifest.is_file() else None,
        "opponent_checkpoint_sha256": {k: _sha256(v) for k, v in (opponent_checkpoint_map or {}).items()},
        "git": _git_identity(_HERE.parent),
        "argv": list(sys.argv if argv is None else ["train_ppo.py", *argv]),
    }
    config["resume_signature"] = _resume_signature(config)
    config["resolved_config_sha256"] = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()

    start_epoch = 0
    global_step = 0
    best_metric = -float("inf")
    best_metric_name = "epoch" if checkpoint_selection == "latest_epoch" else "mean_reward"
    prev_history: list[dict] = []

    if args.resume:
        ckpt = load_ppo_checkpoint(args.resume, map_location="cpu")
        saved_config = ckpt.get("config") or {}
        saved_signature = saved_config.get("resume_signature")
        if saved_signature != config["resume_signature"] and not args.allow_config_override:
            raise SystemExit("resume config differs from checkpoint; use --allow-config-override to record the change explicitly")
        if (ckpt.get("feature_version"), ckpt.get("action_schema_version")) != (feature_version, action_version):
            raise ValueError("resume checkpoint schema differs from requested policy schema")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch") or 0
        global_step = ckpt.get("global_step") or 0
        best_metric = ckpt.get("best_metric") if ckpt.get("best_metric") is not None else -float("inf")
        if not math.isfinite(best_metric):
            best_metric = -float("inf")
        best_metric_name = ckpt.get("best_metric_name") or "mean_reward"
        prev_history = ckpt.get("metrics_history") or []
        restore_ppo_rng_states(ckpt)
        if saved_signature != config["resume_signature"]:
            config["resume_override"] = {"previous_signature": saved_signature, "new_signature": config["resume_signature"]}
        if epochs <= start_epoch:
            raise SystemExit(f"target epochs ({epochs}) must exceed checkpoint epoch ({start_epoch})")
        log.info("resume from %s: epoch=%d global_step=%d", args.resume, start_epoch, global_step)
    else:
        bc_path = cfg.get("bc_checkpoint")
        if feature_version in {"feature-v2", "feature-v3"} and (not bc_path or not Path(bc_path).is_file()):
            raise ValueError("versioned PPO requires an existing same-schema BC initialization checkpoint")
        if bc_path and Path(bc_path).exists():
            from mahjong.training import load_training_checkpoint

            bc_ckpt = load_training_checkpoint(bc_path, map_location="cpu")
            bc_cfg = bc_ckpt.get("config") or {}
            if (bc_cfg.get("feature_version", "feature-v1"), bc_cfg.get("action_schema_version", "action-v1")) != (feature_version, action_version):
                raise ValueError("BC initialization checkpoint schema differs from PPO policy schema")
            init_ppo_from_bc(model, bc_ckpt.get("state_dict") or {})
            log.info("initialised from BC checkpoint %s (value head re-initialised)", bc_path)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.pop("resolved_config_sha256", None)
    config["resolved_config_sha256"] = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()
    (checkpoint_dir / "resolved_config.json").write_text(_canonical_json(config) + "\n", encoding="utf-8")

    best_epoch = start_epoch
    snapshot_paths: dict[int, Path] = {}

    def on_epoch_end(epoch: int, epoch_metrics: dict) -> None:
        nonlocal best_epoch, best_metric
        snap = checkpoint_dir / f"epoch_{epoch}.pt"
        metric_val = float(epoch) if checkpoint_selection == "latest_epoch" else epoch_metrics.get("mean_reward", -float("inf"))
        if checkpoint_selection == "latest_epoch" or metric_val > best_metric:
            best_metric = metric_val
            best_epoch = epoch
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
            environment_version=ENVIRONMENT_VERSION,
            reward_version=cfg.get("reward_version", "reward-v1"),
            model_version=cfg.get("model_version", "ppo-v1"),
            git_commit_hash=(config.get("provenance") or {}).get("git", {}).get("commit"),
            rng_states=capture_ppo_rng_states(),
            opponent_pool_version=cfg.get("opponent_pool_version"),
            opponent_pool_config=opponent_pool_config or None,
            rollout_version=rollout_version,
            training_stage=cfg.get("training_stage"),
        )
        snapshot_paths[epoch] = snap

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
        warmup_epochs=train_cfg.get("warmup_epochs", 0),
        decision_only=train_cfg.get("decision_only", False),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.0),
        target_kl=train_cfg.get("target_kl", 0.0),
        device=device,
        max_episode_steps=train_cfg.get("max_episode_steps", 5000),
        start_epoch=start_epoch,
        global_step=global_step,
        on_epoch_end=on_epoch_end,
        bc_model=bc_model,
        bc_anchor_beta=bc_anchor_beta,
        call_anchor_beta=call_anchor_beta,
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
        step_budget_unit=train_cfg.get("step_budget_unit", "environment_steps"),
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
        environment_version=ENVIRONMENT_VERSION,
        reward_version=cfg.get("reward_version", "reward-v1"),
        model_version=cfg.get("model_version", "ppo-v1"),
        git_commit_hash=(config.get("provenance") or {}).get("git", {}).get("commit"),
        rng_states=capture_ppo_rng_states(),
        opponent_pool_version=cfg.get("opponent_pool_version"),
        opponent_pool_config=opponent_pool_config or None,
        rollout_version=rollout_version,
        training_stage=cfg.get("training_stage"),
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
