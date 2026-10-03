#!/usr/bin/env python
"""Train a Behavior Cloning baseline on the decision-v1 Parquet dataset.

Usage::

    python scripts/build_dataset.py --limit 200 --output data/processed/decision-v1
    python scripts/train_bc.py --config configs/train_bc.yaml
    python scripts/train_bc.py --epochs 2 --lr 0.001 --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402
import yaml  # noqa: E402

from mahjong.dataset.manifest import git_commit  # noqa: E402
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder  # noqa: E402
from mahjong.model import MLPPolicy  # noqa: E402
from mahjong.training import (  # noqa: E402
    capture_rng_states,
    restore_rng_states,
    save_checkpoint,
    save_training_checkpoint,
    train,
    train_dataloader,
)

log = logging.getLogger("train_bc")

_DEFAULT_CONFIG = _HERE.parent / "configs" / "train_bc.yaml"


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _parquet_files(dataset_dir: Path, split: str) -> list[Path]:
    d = dataset_dir / split
    if not d.is_dir():
        raise SystemExit(f"split directory not found: {d} (build the dataset first)")
    return sorted(d.glob("*.parquet"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train a Behavior Cloning baseline.")
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG), help="path to train_bc.yaml")
    ap.add_argument("--dataset-dir", help="override dataset_dir")
    ap.add_argument("--checkpoint-dir", help="override checkpoint_dir")
    ap.add_argument("--epochs", type=int, help="override epochs")
    ap.add_argument("--lr", type=float, help="override learning rate")
    ap.add_argument("--seed", type=int, help="override seed")
    ap.add_argument("--batch-size", type=int, help="override batch size")
    ap.add_argument("--device", help="override device (cpu/cuda)")
    ap.add_argument("--num-workers", type=int, help="override DataLoader num_workers")
    ap.add_argument("--prefetch-factor", type=int, help="override DataLoader prefetch_factor")
    ap.add_argument("--resume", help="resume from a training checkpoint (latest.pt / epoch_N.pt path)")
    ap.add_argument("--save-best-metric", default="val_loss", help="metric to select best.pt: val_loss or val_accuracy")
    args = ap.parse_args(argv)

    cfg = _load_config(Path(args.config))
    model_cfg = cfg.get("model") or {}
    train_cfg = cfg.get("training") or {}

    dataset_dir = Path(args.dataset_dir or cfg["dataset_dir"])
    checkpoint_dir = Path(args.checkpoint_dir or cfg["checkpoint_dir"])
    hidden_sizes = tuple(model_cfg.get("hidden_sizes", [256, 256]))
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 3)
    lr = args.lr if args.lr is not None else train_cfg.get("lr", 0.001)
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 0)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 512)
    device_str = args.device or train_cfg.get("device", "auto")
    device = _resolve_device(device_str)
    use_amp = train_cfg.get("use_amp", False) and device.type == "cuda"
    pin_memory = train_cfg.get("pin_memory", False)
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0)
    prefetch_factor = args.prefetch_factor if args.prefetch_factor is not None else train_cfg.get("prefetch_factor", 4)
    log_interval = train_cfg.get("log_interval_steps", 50)
    shuffle_seed = train_cfg.get("shuffle_seed", seed)  # §12：shard 顺序 = f(seed, epoch)
    val_cfg = cfg.get("validation") or {}
    val_max_samples = val_cfg.get("samples") if val_cfg.get("mode") == "sampled" else None  # §13
    train_limit_shards = train_cfg.get("limit_train_shards", 0)  # §15：部分训练数据

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    train_files = _parquet_files(dataset_dir, "train")
    val_files = _parquet_files(dataset_dir, "validation")
    if not train_files:
        raise SystemExit(f"no train parquet files under {dataset_dir}/train")
    if train_limit_shards:
        train_files = train_files[: train_limit_shards]

    _seed_everything(seed)

    encoder = ObservationEncoder()
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if device.type == "cuda":
        log.info("CUDA device: %s", torch.cuda.get_device_name(0))

    start_epoch = 0
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    prev_history: list[dict] = []
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    if args.resume:
        from mahjong.training import load_training_checkpoint

        ckpt = load_training_checkpoint(args.resume, map_location="cpu")
        # 权重 + 优化器 + epoch/global_step + best + RNG
        model.load_state_dict(ckpt.get("state_dict") or model.state_dict(), strict=False)
        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch") or 0
        best_val_loss = ckpt.get("best_val_loss") if ckpt.get("best_val_loss") is not None else float("inf")
        best_val_accuracy = ckpt.get("best_val_accuracy") or 0.0
        prev_history = (ckpt.get("metrics") or {}).get("history", [])
        if ckpt.get("scaler_state") is not None and scaler is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        restore_rng_states(ckpt)
        log.info("resume from %s: epoch=%d global_step=%s best_val_loss=%s",
                 args.resume, start_epoch, ckpt.get("global_step"), best_val_loss)

    config = {
        "model": {"hidden_sizes": list(hidden_sizes)},
        "feature_dim": FEATURE_DIM,
        "action_space_size": ACTION_SPACE_SIZE,
        "training": {"epochs": epochs, "lr": lr, "batch_size": batch_size, "seed": seed},
        "dataset_dir": str(dataset_dir),
        "dataset_version": cfg.get("dataset_version", "decision-v1"),
        "feature_version": cfg.get("feature_version", "feature-v1"),
        "model_version": cfg.get("model_version", "bc-v1"),
        "device": str(device),
        "use_amp": use_amp,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "pin_memory": pin_memory,
        "save_best_metric": args.save_best_metric,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }

    # per-epoch 快照 + best/latest 跟踪
    best_epoch = start_epoch
    snapshot_paths: dict[int, Path] = {}

    def on_epoch_end(epoch: int, train_avg: dict, val_avg: dict, prof: dict) -> None:
        nonlocal best_epoch, best_val_loss, best_val_accuracy
        # 保存 epoch_N.pt（完整训练状态）
        snap = checkpoint_dir / f"epoch_{epoch}.pt"
        save_training_checkpoint(
            snap,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch,
            global_step=int(sum(h.get("profile", {}).get("samples", 0) for h in prev_history) + prof.get("samples", 0)),
            best_val_loss=best_val_loss,
            best_val_accuracy=best_val_accuracy,
            config=config,
            metrics={"train": train_avg, "validation": val_avg, "history": prev_history + [{"epoch": epoch, "train": train_avg, "val": val_avg, "profile": prof}]},
            scaler_state=scaler.state_dict() if scaler is not None else None,
            rng_states=capture_rng_states(),
        )
        snapshot_paths[epoch] = snap
        # best 判定
        if args.save_best_metric == "val_accuracy":
            metric_val = val_avg.get("accuracy", 0.0)
            improved = metric_val > best_val_accuracy
            if improved:
                best_val_accuracy = metric_val
        else:
            metric_val = val_avg.get("loss", float("inf"))
            improved = metric_val < best_val_loss
            if improved:
                best_val_loss = metric_val
        if improved:
            best_epoch = epoch

    started = time.perf_counter()
    history = train_dataloader(
        model,
        optimizer,
        train_files,
        val_files,
        encoder,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        log_interval_steps=log_interval,
        use_amp=use_amp,
        start_epoch=start_epoch,
        on_epoch_end=on_epoch_end,
        pin_memory=pin_memory,
        shuffle_seed=shuffle_seed,
        val_max_samples=val_max_samples,
    )
    elapsed = time.perf_counter() - started

    history = prev_history + history

    final_train = history[-1]["train"] if history else {}
    final_val = history[-1]["val"] if history else {}

    metrics = {
        "train": final_train,
        "validation": final_val,
        "history": history,
        "elapsed_seconds": round(elapsed, 2),
    }
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

    # latest.pt（最终状态）+ best.pt（最佳 epoch 快照）
    global_step = int(sum(h.get("profile", {}).get("samples", 0) for h in history))
    save_training_checkpoint(
        checkpoint_dir / "latest.pt",
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        epoch=history[-1]["epoch"] if history else 0,
        global_step=global_step,
        best_val_loss=best_val_loss,
        best_val_accuracy=best_val_accuracy,
        config=config,
        metrics=metrics,
        scaler_state=scaler.state_dict() if scaler is not None else None,
        rng_states=capture_rng_states(),
    )
    import shutil

    if best_epoch in snapshot_paths:
        shutil.copyfile(snapshot_paths[best_epoch], checkpoint_dir / "best.pt")
    log.info("saved latest.pt / best.pt (best_epoch=%d, metric=%s)", best_epoch, args.save_best_metric)

    print("== BC training summary ==")
    print(f"dataset:             {dataset_dir}")
    print(f"feature dim:         {FEATURE_DIM}")
    print(f"action space size:   {ACTION_SPACE_SIZE}")
    print(f"hidden sizes:        {list(hidden_sizes)}")
    print(f"epochs:              {epochs}")
    print(f"seed:                {seed}")
    print(f"elapsed:             {elapsed:.1f}s")
    print(f"final train:         {json.dumps(final_train)}")
    print(f"final validation:    {json.dumps(final_val)}")
    print(f"checkpoint:          {checkpoint_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
