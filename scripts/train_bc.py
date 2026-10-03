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
from mahjong.training import save_checkpoint, train, train_dataloader  # noqa: E402

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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    train_files = _parquet_files(dataset_dir, "train")
    val_files = _parquet_files(dataset_dir, "validation")
    if not train_files:
        raise SystemExit(f"no train parquet files under {dataset_dir}/train")

    _seed_everything(seed)

    encoder = ObservationEncoder()
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if device.type == "cuda":
        log.info("CUDA device: %s", torch.cuda.get_device_name(0))

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
    )
    elapsed = time.perf_counter() - started

    final_train = history[-1]["train"] if history else {}
    final_val = history[-1]["val"] if history else {}

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
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
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
