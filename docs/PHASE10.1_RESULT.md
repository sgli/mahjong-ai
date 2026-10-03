# Phase 10.1 结果与交付清单（PHASE10.1_RESULT）

> 对应规范：[`docs/PHASE10.1.md`](PHASE10.1.md)（Large Scale Behavior Cloning Training）
> 完成日期：2026-10-03 ｜ 数据版本：`decision-v2` ｜ 模型版本：`bc-v2`

## 0. 结论

Phase 10.1 的 4 个 Task 全部完成：从**完整 mjai 数据集**构建 `decision-v2` → 训练前验证放行 → 训练大规模 BC 模型 `bc-v2` → 对 random / bc-v1 完成 **10000 局** Benchmark。

- bc-v2 vs random：**压倒性优势**（1 位率 99.2%，mean_rank 0.012）；
- bc-v2 vs bc-v1：**明显优势**（1 位率 69.7%，mean_rank 0.455），体现 decision-v2 大规模训练带来的提升；
- 全程 `illegal_rate = 0`。

**两个重要遗留问题**（详见 §5）：**P1 训练 GPU 利用率低（~6%）导致单 epoch 耗时 13.2h**；**P2 仅训练 1 epoch**（配置为 5）。

---

## 1. Task 1：数据集扩展（decision-v2）✅

- **命令**：`python scripts/build_dataset.py --data-dir "F:/Mahjong AI/mahjong DB" --output data/processed/decision-v2 --dataset-version decision-v2 --seed 42 --num-workers 16 --chunk-size 100000`
- **规模**：

| 项 | 值 |
|---|---|
| 处理文件 | 2,257,268 / 2,257,337（99.997%） |
| 局数（rounds） | 16,688,207 |
| **样本数（samples）** | **1,465,275,164（~14.65 亿）** |
| parquet 分片 | 14,674 |
| 磁盘占用 | ~113 GB |

- **切分**（按 game_id 完整切分、seed 外置确定性）：train 1,172,197,630 ｜ validation 146,516,502 ｜ test 146,561,032 样本
- **schema 不变**：`feature-v1` / `action-v1`；
- **manifest**：`data/processed/decision-v2/manifest.json`（dataset_version / feature_version / game_count / round_count / sample_count / split_ratio / created_at / git_commit）；
- **错误**：69 文件 replay 不一致（`tile X is not in hand`，脏牌谱）＋ 5 样本 `action ∉ legal_actions` → **跳过不入库**；
- **并行构建**：`--num-workers`（单进程 6.2 文件/s → 16 worker ~43 文件/s，全量 ~11.4h）；`--resume` 支持断点续传。

## 2. Task 2：数据集验证 ✅（放行训练）

- **脚本**：`scripts/validate_dataset.py` → 报告 `docs/DATASET_VALIDATION_REPORT.md`
- game_id split：**无跨 game**（抽样 overlap=∅）
- action ∈ legal_actions：**illegal = 0**
- observation 泄漏：无对手暗手 / 无牌山 / 无 ura / 无未来信息
- feature version：manifest 与 schema 一致（`feature-v1`）
- 抽样方法：全量统计用 manifest + parquet metadata（不读数据体）；内容校验抽样 18 shard / 180 万行（≈0.12%）/ 2763 game，行数与 manifest 完全一致。

## 3. Task 3：大规模 BC 训练（bc-v2）✅（1 epoch）

- **配置**：`configs/train_bc_v2.yaml`（`dataset_dir=decision-v2`、`device=auto→cuda`、`use_amp=bf16`、`pin_memory`、`batch_size=512`、`lr=1e-3`、`seed=42`、`num_workers=16`、`prefetch_factor=4`）
- **数据管线**：`ParquetIterableDataset` + `DataLoader(num_workers=16, prefetch_factor=4, pin_memory, persistent_workers)`（每 worker 分互不重叠 shard，shard 列表确定性）
- **模型**：MLP `[256, 256]`，feature dim **234** → action space **270**（架构不变）
- **训练**：**1 epoch**、**47,562 秒（13.2 小时）**、GPU = RTX 4070 Ti SUPER、bf16 混合精度

| 指标 | train | validation |
|---|---|---|
| loss | 0.8959 | 0.8965 |
| accuracy | 0.6806 | **0.6792** |
| top3 | 0.9154 | **0.9170** |
| top5 | 0.9678 | 0.9693 |
| illegal_rate | 0.0 | 0.0 |

- **checkpoint**：`experiments/bc_v2/checkpoints/`（`model.pt` + 版本 sidecar + `metrics.json` + `config.yaml`）
- **内存安全**：DataLoader 并行加载下 pagefile 0/0、worker 内存受控（无 OOM）

## 4. Task 4：Benchmark（≥10000 局）✅

- **脚本**：`scripts/benchmark_parallel.py`（`ProcessPoolExecutor`；`seed = 42 + game_index`（确定性、与并行度无关）；候选座位 `game_index % 4` 轮换）
- **报告**：`docs/BENCHMARK_BC_V2_REPORT.md`

### bc-v2 vs random（5000 局）

| 指标 | bc-v2 | random 基线 |
|---|---|---|
| mean_rank（越低越好） | **0.012** | 1.489 |
| rank 分布（1/2/3/4 位） | [4959, 28, 7, 6] | [1257, 1270, 1245, 1228] |
| win_per_game（次/局） | 2.623 | 0.020 |
| deal_in_per_game（次/局） | 0.011 | 0.017 |
| riichi / call（次/局） | 1.638 / 8.770 | 0.007 / 21.160 |

### bc-v2 vs bc-v1（5000 局）

| 指标 | bc-v2 |
|---|---|
| mean_rank | **0.455** |
| rank 分布 | [3487, 960, 343, 210] |
| win_per_game（次/局） | 2.373 |
| deal_in_per_game（次/局） | 0.297 |
| riichi / call（次/局） | 1.809 / 8.142 |

- **并行正确性**：200 局串行（1 worker）与并行（4 worker）指标完全一致；5000 局并行结果与串行（t68）一致；
- **内存（硬性检查，16 worker）**：pagefile Current/Peak **0/0 MB**、空闲物理内存 ~37 GB 稳定、20 进程 WS 合计 ~7.6 GB；
- **耗时**：合计 **1452 秒（24.2 分钟）**（vs random 378s + vs bc-v1 1074s），对比串行 ~100–140 分钟约 **5–6× 加速**。

---

## 5. 问题与遗留（Problems & Caveats）

### P1【重要】训练 GPU 利用率低，导致耗时较长
- **现象**：bc-v2 训练时 **GPU 利用率仅 ~6%**（显存 582 MiB / 16376 MiB），瓶颈在 **CPU 特征编码**——MLP 太小、数据喂不满 GPU；
- **影响**：单 epoch（11.7 亿训练样本）耗时 **13.2 小时**；完整 5 epoch 约 66 小时（~3 天）；
- **已做优化**：数据管线由「单线程流式 `iter_batches`」改为 `ParquetIterableDataset + DataLoader(num_workers=16)`，吞吐由 ~2,914 样本/s 提升至 ~25,000 样本/s（约 8×），但**仍受 CPU 特征编码限制**（GPU 依旧空闲）；
- **未做（后续可选）**：**预编码特征缓存**——把 observation → 234 维特征一次性编码落盘，训练时直接读特征，可望把 GPU 喂满、显著缩短每 epoch 耗时。

### P2【重要】仅训练 1 epoch
- `configs/train_bc_v2.yaml` 中 `epochs: 5`，但**实际只跑了 1 epoch**（13.2h）；
- **原因**：5 epoch 约 66 小时（~3 天），单会话内不可行；
- **影响**：bc-v2 只见过训练数据 1 遍，val acc 0.6792（较 bc-v1 的 ~0.55 明显提升，但**可能尚未完全收敛**）；
- **后续**：可在有空时续训更多 epoch——注意当前 `train_bc.py` **未实现「从 checkpoint 续训」**，需先补该能力，否则每次都要从头训。

### P3【中】脏牌谱：69 文件 replay 不一致
- 2,257,337 文件中 **69 个（0.003%）** 在 replay 时 `tile 'X' is not in hand`，判定为**牌谱自身脏数据/事件乱序**，已整体跳过，不影响其余 99.997%。

### P4【中】Benchmark random 基线口径异常
- vs random 的 `baseline_metrics`：`win_per_game` 0.020（极低）、`call_per_game` 21.160（极高）——random 策略几乎不和牌却频繁副露；
- 该基线指标可信度存疑（**不影响 bc-v2 自身指标**），后续可复核 random 对手的实现或指标口径。

### P5【低】decision-v2 数据集未入库
- `data/processed/decision-v2`（~113GB）被 `.gitignore` 排除，未提交 git；需保留本地副本，或按 `manifest.json` 重建。

### P6【低】bc-v1 对手较慢
- bc-v2 vs bc-v1 的对局需每步 4 个 MLP 推理，比对 random 慢约 2.8×（1074s vs 378s）。

---

## 6. 产物清单

| 产物 | 路径 | 是否入库 |
|---|---|---|
| decision-v2 数据集 | `data/processed/decision-v2/` | 否（~113GB，gitignore） |
| 数据集 manifest | `data/processed/decision-v2/manifest.json` | 否（同上） |
| 数据集验证报告 | `docs/DATASET_VALIDATION_REPORT.md` | 是 |
| 训练配置 | `configs/train_bc_v2.yaml` | 是 |
| bc-v2 checkpoint | `experiments/bc_v2/checkpoints/`（`model.pt` 786KB） | 是（force-add） |
| 训练指标 | `experiments/bc_v2/checkpoints/metrics.json` | 是 |
| 并行 Benchmark 脚本 | `scripts/benchmark_parallel.py` | 是 |
| Benchmark 结果 | `experiments/benchmark_v2/_parallel_result.json` | 是 |
| Benchmark 报告 | `docs/BENCHMARK_BC_V2_REPORT.md` | 是 |
| 项目状态 | `docs/STATUS.md` | 是 |

## 7. 复现命令

```powershell
# Task1 构建 decision-v2（并行；中断后续传加 --resume）
.venv\Scripts\python.exe scripts\build_dataset.py --data-dir "F:/Mahjong AI/mahjong DB" ^
    --output data/processed/decision-v2 --dataset-version decision-v2 --seed 42 ^
    --num-workers 16 --chunk-size 100000

# Task2 数据集验证
.venv\Scripts\python.exe scripts\validate_dataset.py --dataset-dir data/processed/decision-v2

# Task3 BC 训练（num_workers/prefetch 等在 config 内）
.venv\Scripts\python.exe scripts\train_bc.py --config configs/train_bc_v2.yaml --epochs 1

# Task4 Benchmark（并行 16 worker）
.venv\Scripts\python.exe scripts\benchmark_parallel.py --num-workers 16
```

## 8. 下一步建议

1. **预编码特征缓存**（解决 P1：把 GPU 喂满，缩短每 epoch 耗时）；
2. **实现 checkpoint 续训 + 续训更多 epoch**（解决 P2：让 bc-v2 充分收敛）；
3. 补充 **Rule baseline** 对手，丰富 Benchmark 对比；
4. 复核 random 基线指标口径（P4）；
5. 之后进入 **PPO / RL** 阶段（当前暂缓）。
