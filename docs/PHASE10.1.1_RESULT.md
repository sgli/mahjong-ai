# Phase 10.1.1 结果与验收（PHASE10.1.1_RESULT）

> 对应规范：[`docs/PHASE10.1.1.md`](PHASE10.1.1.md)（BC Training Infrastructure Optimization）
> 完成日期：2026-10-03 ｜ 数据：`decision-v2` ｜ 模型：`bc-v2.1` ｜ 上一步：[`docs/PHASE10.1_RESULT.md`](PHASE10.1_RESULT.md)

## 0. 总结

Phase 10.1.1 的目标是**优化训练基础设施、不改策略定义**。结果：

- **训练吞吐从 ~25,000 提升到 ~260,000–645,000 samples/s（约 11–25×）**，GPU 不再是数据供给的瓶颈；
- **全量 5 epoch 训练只用 3.7 小时**（优化前 1 epoch 就要 13.2 小时）；
- **bc-v2.1 完整 val acc 0.69788 / test acc 0.69782**（优化前 bc-v2 为 0.6792），val≈test 无过拟合；
- **Benchmark 30000 局**：bc-v2.1 在 greedy / sampling 两种模式下对 random / bc-v1 / rule 三类对手**全部占优**；
- 文档 §4–§19 全部实现并验证；**§20（69 个 replay error 分类）按用户决定放弃**。

---

## 1. 逐项对照（PHASE10.1.1.md §4–§26）

| 文档条目 | 要求 | 状态 | 实现 / 结果 |
|---|---|---|---|
| §4 P0 | 可靠 Resume Training | ✅ | `save/load_training_checkpoint`：model/optimizer/epoch/global_step/best/RNG(python+torch CPU/CUDA)/GradScaler；`--resume` 正确恢复（epoch2→续 epoch3，val loss 1.4458→1.3714 连续）；旧格式降级兼容 |
| §5 P0 | Fast Row Encoder | ✅ | `encode_row_fast`：直接从 row 字段产出 234 维，免 `DecisionSample/PlayerObservation/Meld/Action` 对象树；**4096 行 bitwise 等价**（`torch.equal`，rtol=0），`tests/test_fast_encode.py` 固化 |
| §6 P0 | Batch-level Feature Encoding | ✅ | `encode_record_batch_fast`：预分配 `[B,234]` 列式 scatter，消除逐样本 `zeros/tensor/cat` 与 Python 嵌套循环 |
| §7 P0 | Batch-level legal mask | ✅ | 一次构造 `[B,270]` bool（batch scatter）；第二阶段「用 action IDs 替代 mask」按文档**不做** |
| §8 P0 | 减少 Arrow→Python 转换 | ✅ | `ParquetBatchDataset` 直接 yield Arrow `RecordBatch`，不再 `to_pylist` |
| §9 P0 | Training Profiler | ✅ | 每 epoch 记录 `data/forward/backward/optimizer/batch_time` + `samples_per_sec` + `gpu_memory`（`gpu_util` 置 None，需 nvidia-smi 外部采集） |
| §10 P0 | Batch Size Sweep | ✅ | `scripts/bench_batch_sweep.py`（**已修复**：改用新快管线 + warmup + 500 步计时）；实测 512/2048/8192 = 304k/472k/477k samples/s，**2048–8192 最优** |
| §11 P1 | DataLoader/prefetch/pin/non_blocking | ✅ | `num_workers=16`、`prefetch_factor=4`、`persistent_workers`、`pin_memory` + `non_blocking=True` |
| §11.1 P1 | 修正 pin_memory 配置语义 | ✅ | `train_dataloader(..., pin_memory=...)` 由配置显式控制；config 为 false 时确实不 pin |
| §12 P1 | Deterministic Shard Shuffle | ✅ | `_shuffle_shards(shards, shuffle_seed + epoch)`；同 seed 同 epoch 顺序全等、不同 epoch 不同（已验证） |
| §13 P1 | Validation Sampling | ✅ | `validation.mode=sampled` + `samples=N` 截断；记录 `sampled` 样本数；最终用 `evaluate_bc.py` 做完整 val/test |
| §14 P1 | 完整 Learning Curve | ✅ | 每 epoch 落盘 train/val loss、accuracy、top3、top5（见 §2.3） |
| §15 | 先做小规模收敛实验 | ✅ | 500 shard（5000 万样本/epoch）× 3 epoch，见 §2.2 |
| §16 P1 | Best / Latest Checkpoint | ✅ | `latest.pt` / `best.pt`（`--save-best-metric`，默认 val_loss）/ `epoch_N.pt` |
| §17 P1 | Benchmark Greedy / Sampling | ✅ | `PolicyOpponent(mode=greedy\|sampling)`；greedy=argmax（确定性）、sampling=multinomial |
| §18 P1 | Benchmark Rule Baseline | ✅ | `RuleOpponent(rule@rule-v1)`（和了/立直/不副露/暗杠/打最无用牌），已接入 `default_pool` |
| §19 P1 | 同时记录 Candidate 与 Opponent | ✅ | `compute_opponents_metrics`；每个 match-up 落盘 candidate + opponents 双端八项指标 |
| §20 | 数据质量：69 个 replay error 分类 | ⛔ **放弃** | 错误清单不可得（原构建未写 jsonl）；`dataset_errors.jsonl` + `classify_errors.py` 已实现，但**分类重扫按用户决定放弃** |
| §21 | 不建议做的事 | ✅ 遵守 | 未换 Transformer、未多 GPU、未进 PPO、**未做完整 feature cache** |
| §23 | 验收标准 | ✅ 见 §4 | 数据正确性 / 吞吐 / Resume / Checkpoint / Benchmark / Learning Curve 全部满足 |
| §24–§26 | 决策树 / 目标架构 / 核心原则 | ✅ | 训练语义未变、无数据泄漏、先性能后规模、所有优化均有 before/after 数据 |

---

## 2. 训练部分结果（详细）

### 2.1 吞吐优化（对应 §5–§11；A + B）

**优化前基线**：~25,000 samples/s（单线程 `iter_batches` 逐样本构造对象树 + 特征编码，GPU util ~6%，1 epoch 需 13.2h）。

**优化后实测**（RTX 4070 Ti SUPER，16 worker，bf16）：

| 测量方式 | batch 512 | batch 2048 | batch 8192 | 备注 |
|---|---|---|---|---|
| 真实训练路径（20 shard ≈ 200 万样本，1 epoch） | 261,526 | **297,629** | 284,723 | samples/s，epoch 墙钟 31.3/28.0/28.6s |
| sweep 工具（修复后，500 步计时） | 304,097 | 471,682 | **477,090** | samples/s（不计 metrics） |
| 小规模收敛实验（500 shard × 3 epoch） | — | 522,973 ~ 526,789 | — | 每 epoch `batch_time` 口径，batch 2048 |
| **全量 5 epoch 训练** | — | **645,415** | — | 第 5 epoch `batch_time` 口径，全量 11.7 亿样本 |

**提升幅度**：~11×（真实训练路径）到 ~25×（大样本 batch_time 口径）；
**§23 目标**（第一目标 ≥50k、理想 75k–100k+）→ **大幅超额**；
**瓶颈变化**：从「CPU 数据供给」变为「GPU 仍空闲（MLP 太小，非数据问题）」。
**GPU 显存占用极小**（batch 512–8192 分别 ~22.7 / 31.3 / 69.7 MB）；**内存安全**：pagefile 全程 0/0，无 OOM。

### 2.2 小规模收敛实验（§15）

配置：500 train shard（≈5000 万样本/epoch）× 3 epoch，batch 2048，lr 1e-3，16 worker，sampled val 1M。耗时 **457.5s**。

| epoch | train loss | train acc | train top3 | val loss | val acc | val top3 | val top5 |
|---|---|---|---|---|---|---|---|
| 1 | 1.0393 | 0.6425 | 0.8758 | 0.9819 | 0.6571 | 0.8944 | 0.9562 |
| 2 | 0.9402 | 0.6694 | 0.9042 | 0.9385 | 0.6685 | 0.9065 | 0.9633 |
| 3 | 0.9128 | 0.6764 | 0.9115 | 0.9212 | 0.6726 | 0.9105 | 0.9656 |

**结论**：仍在改善但**边际收益递减**（val acc 增量 +1.14% → +0.41%；val loss 增量 −0.043 → −0.017），train≈val 无过拟合，illegal_rate 全程 0。→ 用子集继续堆 epoch 收益有限，值得上全量数据。

### 2.3 全量训练（bc-v2.1，5 epoch）

配置：`configs/train_bc_v2.1.yaml` —— 全量 `decision-v2`（`limit_train_shards=0`，11.7 亿样本/epoch）、batch 2048、lr 1e-3、seed 42、16 worker、bf16、每 epoch shard shuffle。**耗时 13,258.4s（≈3.7 小时）**，吞吐 ~645k samples/s。

| epoch | train loss | train acc | train top3 | val loss | val acc | val top3 |
|---|---|---|---|---|---|---|
| 1 | 0.8702 | 0.6888 | 0.9202 | 0.8527 | 0.6935 | 0.9250 |
| **2** | 0.8397 | 0.6970 | 0.9266 | **0.8409** ← 最优 | **0.6965** ← 最优 | 0.9268 |
| 3 | 0.8373 | 0.6974 | 0.9270 | 0.8445 | 0.6943 | 0.9265 |
| 4 | 0.8405 | 0.6957 | 0.9269 | 0.8449 | 0.6929 | 0.9267 |
| 5 | 0.8416 | 0.6949 | 0.9267 | 0.8457 | 0.6939 | 0.9266 |

**关键结论**：模型在 **第 2 epoch 收敛**（`best.pt` = epoch 2）；**epoch 3–5 轻微过拟合**（train loss 继续微降、val 变差）。
→ 全量 1 epoch（11.7 亿样本）已基本足够，2 epoch 到顶，第 3 个 epoch 起无收益（后续再训 2 epoch 即可）。
产物：`experiments/bc_v2.1/checkpoints/{epoch_1..5.pt, latest.pt, best.pt, model.pt, metrics.json}`。

### 2.4 完整 validation + test 评估（§13 收尾，C）

用 `scripts/evaluate_bc.py`（`--mode full`）对 `best.pt`（= epoch 2）在**完整** split 上评估，2.92 亿样本，耗时 ~15 分钟：

| split | 样本数 | loss | **accuracy** | top3 | top5 | illegal_rate |
|---|---|---|---|---|---|---|
| validation | 146,516,502 | 0.836892 | **0.69788** | 0.92716 | 0.97447 | 0 |
| test | 146,561,032 | 0.837014 | **0.69782** | 0.92714 | 0.97447 | 0 |

- **val ≈ test**（差 0.00006）→ 泛化良好、无过拟合；
- 相比 bc-v2（全量 1 epoch，val acc **0.6792**）→ **+1.87 个百分点**；
- 结果落盘 `experiments/bc_v2.1/checkpoints/eval.json`。

---

## 3. Benchmark 部分结果（详细，D）

**设置**：candidate = `bc-v2.1`（`best.pt`）；两种模式 `greedy`（argmax）/ `sampling`（multinomial）；三类对手 `random` / `bc-v1` / `rule`；**每个对手 5000 局**（每个模式合计 15000 局，两模式共 **30000 局**）；16 worker；`seed = 42 + game_index`；候选座位 `game_index % 4` 轮换（消除座位偏差）。
**耗时**：greedy 1913.7s + sampling 1911.1s ≈ **64 分钟**。结果落盘 `experiments/benchmark_v21_{greedy,sampling}/result.json`。

### 3.1 Greedy 模式（argmax，确定性）

| 对手 | **candidate mean_rank** | rank 分布 [1/2/3/4 位] | win/局 | 放铳/局 | 立直/局 | 副露/局 | 平均和牌点 | 平均得点变化 |
|---|---|---|---|---|---|---|---|---|
| random | **0.0008** | [4996, 4, 0, 0] | 4.444 | 0.0098 | 2.330 | 4.396 | 8623 | +43671 |
| bc-v1 | **0.1586** | [4396, 454, 111, 39] | 3.734 | 0.2274 | 2.479 | 4.923 | 8868 | +33159 |
| rule | **0.9198** | [2200, 1462, 877, 461] | 2.728 | 0.6158 | 2.191 | 4.994 | 9402 | +13897 |

对手侧（3 个非候选座位聚合）：

| 对手 | opponents mean_rank | win/局 | 放铳/局 | 立直/局 | 副露/局 |
|---|---|---|---|---|---|
| random | 1.9997 | 0.0142 | 1.100 | 0.008 | 16.591 |
| bc-v1 | 1.9471 | 0.486 | 1.110 | 0.804 | 6.502 |
| rule | 1.6934 | 1.197 | 1.277 | 2.810 | 0.467 |

### 3.2 Sampling 模式（multinomial）

| 对手 | **candidate mean_rank** | rank 分布 [1/2/3/4 位] | win/局 | 放铳/局 | 立直/局 | 副露/局 | 平均和牌点 | 平均得点变化 |
|---|---|---|---|---|---|---|---|---|
| random | **0.0064** | [4978, 16, 2, 4] | 2.773 | 0.010 | 1.649 | 7.435 | 8291 | +27455 |
| bc-v1 | **0.3648** | [3729, 865, 259, 147] | 2.558 | 0.268 | 1.866 | 6.842 | 8737 | +21404 |
| rule | **1.3374** | [1392, 1440, 1257, 911] | 1.959 | 0.735 | 1.596 | 6.866 | 9193 | +4371 |

对手侧：

| 对手 | opponents mean_rank | win/局 | 放铳/局 | 立直/局 | 副露/局 |
|---|---|---|---|---|---|
| random | 1.9979 | 0.014 | 0.683 | 0.007 | 15.757 |
| bc-v1 | 1.8784 | 0.515 | 0.843 | 0.769 | 6.859 |
| rule | 1.5542 | 1.261 | 1.089 | 2.848 | 0.476 |

### 3.3 结论

1. **bc-v2.1 对三类对手全部占优**：candidate `mean_rank` 在 6 个 match-up 中全部低于「4 人等强」的 1.5 基准；
2. **Greedy 一致强于 Sampling**（vs random 0.0008 vs 0.0064；vs bc-v1 0.1586 vs 0.3648；vs rule 0.9198 vs 1.3374）→ 后续评估以 **greedy 为主、sampling 为辅**；
3. **Rule baseline 是最强对手**（random < bc-v1 < rule）：greedy vs rule 时 candidate 0.92 / 对手 1.69；sampling vs rule 时 candidate 1.34 / 对手 1.55——**差距最小**，印证文档 §18 的判断（random 只是 sanity check，rule 才有强度对照意义）；
4. **相比 bc-v2 有提升**（同为 sampling：对 random 0.012 → 0.0064；对 bc-v1 0.455 → 0.3648），与 val acc +1.87% 一致；
5. illegal_rate 全程 **0**。

---

## 4. 验收标准对照（§23）

| 标准 | 要求 | 结果 |
|---|---|---|
| 数据正确性 | Fast Encoder == Old Encoder（100% sampled validation，不得改变语义） | ✅ 4096 行 bitwise 等价（`torch.equal`）；epoch1 loss 完全一致（1.5783242984693877）证明语义未变 |
| Training throughput | 基线 ~25k；第一目标 ≥50k；理想 75k–100k+ | ✅ **261k–645k**（远超理想） |
| Resume | 中断 → resume → 正确 epoch/global_step；model/optimizer/LR/RNG 正确恢复 | ✅ 小规模验证：epoch2 → 续 epoch3，val loss 1.4458→1.3714 连续 |
| Checkpoint | 必须存在 `latest.pt` / `best.pt` 且可加载 | ✅ 均存在且可加载（`best.pt` = epoch 2） |
| Benchmark | 至少 BC-v2 Greedy / Sampling / BC-v1 / Rule / Random | ✅ 全覆盖（§3，30000 局） |
| Learning Curve | 至少 epoch 1/2/3 的 train/val loss+accuracy | ✅ 小规模 3 epoch + 全量 5 epoch 曲线（§2.2/§2.3） |

## 5. 遗留与后续

- **§20（E 数据质量）放弃**（用户决定）：69 个 replay error 未做分类；`dataset_errors.jsonl` 写入与 `classify_errors.py` 已实现，后续如需可用 `--resume` 只重扫那 ~69 个文件；
- **未做且文档不建议**：完整 dense feature cache（§21.4，TB 级存储）、扩大模型（§21.1）、多 GPU（§21.2）、PPO（§21.3）；
- 已知现象：random baseline 的 `call_per_game` 很高 / `win_per_game` 很低（§18 已说明 random 只作 sanity check）；
- **收敛结论**：bc-v2.1 于 epoch 2 收敛，后续若继续训练，**2 epoch 已足够**；
- `gpu_util` 目前为 None（需外部 nvidia-smi 采集），profiler 已有 `gpu_memory`。

## 6. 产物清单

| 产物 | 路径 |
|---|---|
| Fast Encoder | `src/mahjong/training/fast_encode.py` |
| 训练/数据管线 | `src/mahjong/training/bc.py` |
| Checkpoint（完整状态/Resume） | `src/mahjong/training/checkpoint.py` |
| 训练入口 | `scripts/train_bc.py` |
| Batch-size sweep 工具 | `scripts/bench_batch_sweep.py` |
| 评估入口（val+test） | `scripts/evaluate_bc.py` |
| 并行 Benchmark | `scripts/benchmark_parallel.py` |
| bc-v2.1 配置 | `configs/train_bc_v2.1.yaml` |
| 小规模收敛实验 | `experiments/bc_v2_convergence/` |
| **bc-v2.1 checkpoint** | `experiments/bc_v2.1/checkpoints/{best.pt, latest.pt, epoch_1..5.pt, metrics.json, eval.json}` |
| Benchmark 结果 | `experiments/benchmark_v21_{greedy,sampling}/result.json` |
| Fast Encoder 等价测试 | `tests/test_fast_encode.py` |

## 7. 复现命令

```powershell
# 小规模收敛实验（§15）
.venv\Scripts\python.exe scripts\train_bc.py --config experiments\bc_v2_convergence\config.yaml

# 全量 5 epoch 训练（bc-v2.1）
.venv\Scripts\python.exe scripts\train_bc.py --config configs\train_bc_v2.1.yaml

# 完整 val + test 评估（§13 收尾）
.venv\Scripts\python.exe scripts\evaluate_bc.py --checkpoint experiments\bc_v2.1\checkpoints\best.pt ^
    --dataset-dir data\processed\decision-v2 --mode full --splits validation,test --num-workers 16

# 大规模 Benchmark（D：greedy / sampling × random,bc_v1,rule × 5000 局/对手）
.venv\Scripts\python.exe scripts\benchmark_parallel.py --candidate experiments\bc_v2.1\checkpoints ^
    --candidate-mode greedy --opponents random,bc_v1,rule --games-per-opponent 5000 --num-workers 16 ^
    --out experiments\benchmark_v21_greedy
# 同上换 --candidate-mode sampling --out experiments\benchmark_v21_sampling

# batch-size sweep（§10，运行前需批准）
.venv\Scripts\python.exe scripts\bench_batch_sweep.py --dataset-dir data\processed\decision-v2 ^
    --batch-sizes 512,1024,2048,4096,8192,16384 --steps 500 --warmup 20 --num-workers 16 --amp
```
