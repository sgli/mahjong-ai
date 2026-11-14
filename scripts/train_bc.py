#!/usr/bin/env python
"""Train a Behavior Cloning baseline on the decision-v1 Parquet dataset.

Usage::

    python scripts/build_dataset.py --limit 200 --output data/processed/decision-v1
    python scripts/train_bc.py --config configs/train_bc_v4.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

# DataLoader workers import NumPy/PyTorch under Windows spawn. Cap nested BLAS
# pools so process-level parallelism does not create dozens of BLAS threads per
# worker (which caused OpenBLAS allocation failures at high worker counts).
for _thread_env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402
import yaml  # noqa: E402

from mahjong.dataset.manifest import git_commit  # noqa: E402
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder  # noqa: E402
from mahjong.features.action_space_v2 import ACTION_SPACE_SIZE_V2  # noqa: E402
from mahjong.features.encoder_v2 import FEATURE_DIM_V2, ObservationEncoderV2  # noqa: E402
from mahjong.features.encoder_v3 import FEATURE_DIM_V3, ObservationEncoderV3  # noqa: E402
from mahjong.model import build_policy_model  # noqa: E402
from mahjong.training import (  # noqa: E402
    capture_rng_states,
    restore_rng_states,
    save_checkpoint,
    save_training_checkpoint,
    train,
    train_dataloader,
)

log = logging.getLogger("train_bc")

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


def _validate_resume(ckpt, cfg, hidden_sizes, batch_size, seed, lr, manifest_hash, save_best_metric="val_loss",
                     *, shuffle_seed=None, validation_samples=None, limit_train_shards=0, use_amp=False):
    old = ckpt.get("config") or {}
    for key, default in (("feature_version", "feature-v1"), ("action_schema_version", "action-v1"), ("dataset_version", "decision-v1")):
        if old.get(key, default) != cfg.get(key, default):
            raise ValueError(f"resume {key} mismatch")
    for key in ("environment_version", "dataset_transform"):
        if old.get(key) != cfg.get(key):
            raise ValueError(f"resume {key} mismatch")
    if old.get("save_best_metric", "val_loss") != save_best_metric:
        raise ValueError("resume best metric mismatch")
    if tuple((old.get("model") or {}).get("hidden_sizes", [256, 256])) != hidden_sizes:
        raise ValueError("resume model architecture mismatch")
    if old.get("model") != cfg.get("model"):
        raise ValueError("resume model architecture parameters differ")
    if old.get("dataset_manifest_sha256") != manifest_hash:
        raise ValueError("resume dataset manifest mismatch")
    for key, value in (("batch_size", batch_size), ("seed", seed), ("lr", lr)):
        if (old.get("training") or {}).get(key) != value:
            raise ValueError(f"resume training.{key} mismatch")
    for key, value in (("shuffle_seed", shuffle_seed), ("limit_train_shards", limit_train_shards)):
        if (old.get("training") or {}).get(key) != value:
            raise ValueError(f"resume training.{key} mismatch")
    if old.get("validation_samples") != validation_samples or old.get("use_amp") != use_amp:
        raise ValueError("resume validation/precision settings mismatch")
    from mahjong.training.data_gate import source_hashes
    if old.get("source_sha256") != source_hashes():
        raise ValueError("resume source identity mismatch")
    if not ckpt.get("state_dict") or ckpt.get("optimizer_state_dict") is None:
        raise ValueError("resume requires complete model and optimizer states")


def _copy_snapshot(source, destination):
    import shutil
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train a Behavior Cloning baseline.")
    ap.add_argument("--config", type=Path, required=True, help="path to the selected BC config")
    ap.add_argument("--dataset-dir", help="override dataset_dir")
    ap.add_argument("--checkpoint-dir", help="override checkpoint_dir")
    ap.add_argument("--epochs", type=int, help="override epochs")
    ap.add_argument("--lr", type=float, help="override learning rate")
    ap.add_argument("--seed", type=int, help="override seed")
    ap.add_argument("--batch-size", type=int, help="override batch size")
    ap.add_argument("--device", help="override device (cpu/cuda)")
    ap.add_argument("--num-workers", type=int, help="override DataLoader num_workers")
    ap.add_argument("--prefetch-factor", type=int, help="override DataLoader prefetch_factor")
    ap.add_argument("--limit-train-shards", type=int, help="optional shard limit for pipeline smoke checks")
    ap.add_argument("--validation-samples", type=int, help="override sampled validation budget")
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
    metric_interval = train_cfg.get("metric_interval_steps", 1)
    shuffle_seed = train_cfg.get("shuffle_seed", seed)  # §12：shard 顺序 = f(seed, epoch)
    val_cfg = cfg.get("validation") or {}
    val_max_samples = (
        args.validation_samples if args.validation_samples is not None
        else val_cfg.get("samples") if val_cfg.get("mode") == "sampled" else None
    )  # §13
    train_limit_shards = (
        args.limit_train_shards if args.limit_train_shards is not None
        else train_cfg.get("limit_train_shards", 0)
    )  # §15：部分训练数据

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    train_files = _parquet_files(dataset_dir, "train")
    val_files = _parquet_files(dataset_dir, "validation")
    if not train_files:
        raise SystemExit(f"no train parquet files under {dataset_dir}/train")
    if train_limit_shards:
        train_files = train_files[: train_limit_shards]

    feature_version = cfg.get("feature_version", "feature-v1")
    if feature_version in {"feature-v2", "feature-v3"} and not train_limit_shards:
        from mahjong.training.data_gate import require_full_audit
        require_full_audit(cfg.get("data_audit_path", "experiments/phase10_3_7/data_audit.json"), dataset_dir)
    elif feature_version in {"feature-v2", "feature-v3"}:
        log.warning("bounded pipeline diagnostic only: full data legality gate not passed")

    _seed_everything(seed)

    action_schema_version = cfg.get("action_schema_version", "action-v1")
    if feature_version == "feature-v2" and action_schema_version == "action-v2":
        encoder = ObservationEncoderV2()
        feature_dim, action_size = FEATURE_DIM_V2, ACTION_SPACE_SIZE_V2
    elif feature_version == "feature-v3" and action_schema_version == "action-v2":
        encoder = ObservationEncoderV3()
        feature_dim, action_size = FEATURE_DIM_V3, ACTION_SPACE_SIZE_V2
    elif feature_version == "feature-v1" and action_schema_version == "action-v1":
        encoder = ObservationEncoder()
        feature_dim, action_size = FEATURE_DIM, ACTION_SPACE_SIZE
    else:
        raise SystemExit(f"unsupported feature/action schema pair: {feature_version}/{action_schema_version}")
    model_config = dict(model_cfg)
    model_config["hidden_sizes"] = list(hidden_sizes)
    model = build_policy_model(
        feature_version, feature_dim, action_size, hidden_sizes,
        model_config=model_config,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if device.type == "cuda":
        log.info("CUDA device: %s", torch.cuda.get_device_name(0))

    manifest_path = dataset_dir / "manifest.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else None
    resume_samples = 0
    restored_best_epoch = 0
    start_epoch = 0
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    prev_history: list[dict] = []
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    if args.resume:
        from mahjong.training import load_training_checkpoint

        ckpt = load_training_checkpoint(args.resume, map_location="cpu")
        # 权重 + 优化器 + epoch/global_step + best + RNG
        _validate_resume(ckpt, cfg, hidden_sizes, batch_size, seed, lr, manifest_hash, args.save_best_metric,
                         shuffle_seed=shuffle_seed, validation_samples=val_max_samples,
                         limit_train_shards=train_limit_shards, use_amp=use_amp)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch") or 0
        best_val_loss = ckpt.get("best_val_loss") if ckpt.get("best_val_loss") is not None else float("inf")
        best_val_accuracy = ckpt.get("best_val_accuracy") or 0.0
        prev_history = (ckpt.get("metrics") or {}).get("history", [])
        resume_samples = int(ckpt.get("global_step") or 0)
        restored_best_epoch = ckpt.get("best_epoch")
        if restored_best_epoch is None:
            raise ValueError("resume checkpoint lacks reliable best_epoch; use a repaired full-state checkpoint")
        if ckpt.get("scaler_state") is not None and scaler is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        restore_rng_states(ckpt)
        log.info("resume from %s: epoch=%d global_step=%s best_val_loss=%s",
                 args.resume, start_epoch, ckpt.get("global_step"), best_val_loss)

    if epochs <= start_epoch:
        raise ValueError("requested epochs must exceed resume epoch")

    transform_paths = [
        _HERE.parent / "src/mahjong/features/action_space_v2.py",
        _HERE.parent / "src/mahjong/features/encoder_v2.py",
        _HERE.parent / "src/mahjong/training/fast_encode.py",
    ]
    if feature_version == "feature-v3":
        transform_paths.extend((
            _HERE.parent / "src/mahjong/features/encoder_v3.py",
            _HERE.parent / "src/mahjong/training/fast_encode_v3.py",
        ))
    transform_hashes = {
        str(path.relative_to(_HERE.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in transform_paths if path.is_file()
    } if feature_version in {"feature-v2", "feature-v3"} else None

    config = {
        "model": model_config,
        "feature_dim": feature_dim,
        "action_space_size": action_size,
        "training": {
            "epochs": epochs, "lr": lr, "batch_size": batch_size, "seed": seed,
            "metric_interval_steps": metric_interval,
            "limit_train_shards": train_limit_shards,
            "shuffle_seed": shuffle_seed,
        },
        "validation_samples": val_max_samples,
        "data_worker_threads": 1,
        "dataset_dir": str(dataset_dir),
        "dataset_version": cfg.get("dataset_version", "decision-v1"),
        "dataset_transform": cfg.get("dataset_transform"),
        "dataset_manifest_sha256": hashlib.sha256((dataset_dir / "manifest.json").read_bytes()).hexdigest()
        if (dataset_dir / "manifest.json").is_file() else None,
        "feature_version": feature_version,
        "action_schema_version": action_schema_version,
        "allow_kyuushu": bool(cfg.get("allow_kyuushu", False)),
        "model_version": cfg.get("model_version", "bc-v1"),
        "environment_version": cfg.get("environment_version"),
        "pipeline_diagnostic_only": bool(train_limit_shards),
        "device": str(device),
        "use_amp": use_amp,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "pin_memory": pin_memory,
        "save_best_metric": args.save_best_metric,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "transform_code_sha256": transform_hashes,
    }
    from mahjong.training.data_gate import source_hashes
    config["source_sha256"] = source_hashes()

    # per-epoch 快照 + best/latest 跟踪
    best_epoch = restored_best_epoch
    cumulative_history = list(prev_history)
    cumulative_samples = resume_samples
    snapshot_paths: dict[int, Path] = {}
    if args.resume and best_epoch:
        from mahjong.training import load_training_checkpoint
        source_dir = Path(args.resume).parent
        best_source = next((p for p in (source_dir / f"epoch_{best_epoch}.pt", source_dir / "best.pt", Path(args.resume))
                            if p.is_file() and load_training_checkpoint(p).get("epoch") == best_epoch), None)
        if best_source is None:
            raise ValueError("resume requires the historical best checkpoint artifact")
        _copy_snapshot(best_source, checkpoint_dir / "best.pt")

    def on_epoch_end(epoch: int, train_avg: dict, val_avg: dict, prof: dict) -> None:
        nonlocal best_epoch, best_val_loss, best_val_accuracy, cumulative_samples
        cumulative_samples += int(prof.get("samples", 0))
        cumulative_history.append({"epoch": epoch, "train": train_avg, "val": val_avg, "profile": dict(prof)})
        loss_value = val_avg.get("loss", float("inf"))
        accuracy_value = val_avg.get("accuracy", 0.0)
        improved = accuracy_value > best_val_accuracy if args.save_best_metric == "val_accuracy" else loss_value < best_val_loss
        best_val_loss = min(best_val_loss, loss_value)
        best_val_accuracy = max(best_val_accuracy, accuracy_value)
        if improved:
            best_epoch = epoch
        snap = checkpoint_dir / f"epoch_{epoch}.pt"
        save_training_checkpoint(
            snap, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch, global_step=cumulative_samples, best_epoch=best_epoch,
            best_val_loss=best_val_loss, best_val_accuracy=best_val_accuracy, config=config,
            metrics={"train": train_avg, "validation": val_avg, "history": list(cumulative_history)},
            scaler_state=scaler.state_dict() if scaler is not None else None,
            rng_states=capture_rng_states(),
        )
        snapshot_paths[epoch] = snap
        import shutil
        _copy_snapshot(snap, checkpoint_dir / "latest.pt")
        if improved:
            _copy_snapshot(snap, checkpoint_dir / "best.pt")

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
        metric_interval_steps=metric_interval,
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
    global_step = cumulative_samples
    save_training_checkpoint(
        checkpoint_dir / "latest.pt",
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        epoch=history[-1]["epoch"] if history else 0,
        global_step=global_step,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_accuracy=best_val_accuracy,
        config=config,
        metrics=metrics,
        scaler_state=scaler.state_dict() if scaler is not None else None,
        rng_states=capture_rng_states(),
    )
    import shutil

    if best_epoch in snapshot_paths:
        _copy_snapshot(snapshot_paths[best_epoch], checkpoint_dir / "best.pt")
    log.info("saved latest.pt / best.pt (best_epoch=%d, metric=%s)", best_epoch, args.save_best_metric)

    print("== BC training summary ==")
    print(f"dataset:             {dataset_dir}")
    print(f"feature dim:         {feature_dim}")
    print(f"action space size:   {action_size}")
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
