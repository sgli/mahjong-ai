#!/usr/bin/env python
"""Batch-size throughput sweep tool (Phase 10.1.1-A §10).

对每个 batch size 跑少量 batch 的 forward+backward，输出 samples/sec、GPU memory，
供 before/after 对比与 batch-size 选择。

与 ``scripts/train_bc.py`` 使用**同一条训练数据管线**
（``ParquetBatchDataset`` + ``collate_record_batches``，即 fast batch encoder），
并跳过 ``--warmup`` 个 batch 之后才开始计时，避免把 DataLoader / worker 启动开销
计入吞吐（早期版本用了旧的 sample-level collate 且只跑 20 步，数值严重失真）。

用法（**运行前需 captain 批准**）::

    python scripts/bench_batch_sweep.py \
        --dataset-dir data/processed/decision-v2 \
        --batch-sizes 512,1024,2048,4096,8192,16384 \
        --steps 50 --warmup 5 --num-workers 16 --amp
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM  # noqa: E402
from mahjong.model import MLPPolicy  # noqa: E402
from mahjong.training import ParquetBatchDataset, collate_record_batches  # noqa: E402


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def measure(
    dataset_dir: Path,
    batch_size: int,
    steps: int,
    device: torch.device,
    num_workers: int,
    use_amp: bool,
    warmup: int = 5,
) -> dict:
    """Measure steady-state throughput at ``batch_size`` (skip ``warmup`` batches)."""
    train = sorted((dataset_dir / "train").glob("*.parquet"))
    if not train:
        raise SystemExit(f"no train parquet shards under {dataset_dir / 'train'}")

    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (256, 256)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        ParquetBatchDataset(train, batch_size),
        batch_size=1,  # 每个 item 已经是一个 batch_size 行的 RecordBatch
        collate_fn=collate_record_batches,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers else None,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    samples = 0
    t0: float | None = None
    for step, (features, action_ids, masks) in enumerate(loader):
        if step == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        features = features.to(device, non_blocking=True)
        action_ids = action_ids.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(features, masks)
            loss = torch.nn.functional.cross_entropy(out.logits, action_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t0 is not None:
            samples += features.shape[0]
        if step + 1 >= warmup + steps:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) if t0 is not None else float("nan")
    return {
        "batch_size": batch_size,
        "steps": steps,
        "warmup": warmup,
        "samples_per_sec": round(samples / dt, 1) if t0 is not None and dt > 0 else None,
        "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1) if device.type == "cuda" else None,
        "gpu_memory_max_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1) if device.type == "cuda" else None,
        "elapsed_s": round(dt, 2) if t0 is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Batch-size throughput sweep (current training pipeline).")
    ap.add_argument("--dataset-dir", default="data/processed/decision-v2")
    ap.add_argument("--batch-sizes", default="512,1024,2048,4096,8192,16384")
    ap.add_argument("--steps", type=int, default=500, help="计时的 batch 数（warmup 之外）")
    ap.add_argument("--warmup", type=int, default=20, help="计时前跳过的 warmup batch 数")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args(argv)

    device = _resolve_device(args.device)
    sizes = [int(x) for x in args.batch_sizes.split(",")]
    results = []
    for bs in sizes:
        r = measure(Path(args.dataset_dir), bs, args.steps, device, args.num_workers, args.amp, args.warmup)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
