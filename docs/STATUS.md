# 项目状态（STATUS）

> 维护方式：每次阶段完成后更新此文件，避免文档与代码状态漂移。

## 当前完成阶段

- **Phase 1**：Mjai Parser（读取、事件、数据统计）
- **Phase 2**：Replay Engine（状态重建）
- **Phase 3**：Decision Extraction（决策点抽取）
- **Phase 4**：Dataset（decision-v1 schema + manifest）
- **Phase 5**：Feature Encoder（feature-v1）
- **Phase 6**：Rules（立直麻将规则）
- **Phase 7**：Environment（环境/合法动作/结算）
- **Phase 8**：PPO + Self-play（训练框架骨架）
- **Phase 9**：Benchmark（对局/评估骨架）
- **Phase 10**：BC Baseline 训练 + Benchmark（`experiments/exp_0001`，bc-v1）
- **Phase 10.1**：大规模 BC 训练（decision-v2 → bc-v2 → 10000 局 Benchmark）
  - 数据集 `data/processed/decision-v2`：2,257,268 局 / 1,465,275,164 样本 / ~113GB / 14,674 parquet
  - 训练：bc-v2，全量 1 epoch、13.2 小时、val acc **0.6792**
  - Benchmark：10000 局（见 `docs/BENCHMARK_BC_V2_REPORT.md`）
- **Phase 10.1.1**：训练基础设施优化 → **bc-v2.1**（见 `docs/PHASE10.1.1_RESULT.md`）
  - 吞吐优化：~25k → **261k–645k samples/s（约 11–25×）**；优化前 1 epoch 13.2h，优化后全量 5 epoch 仅 3.7h
  - 基础设施：完整 Resume（optimizer/epoch/global_step/RNG/GradScaler）+ best/latest/epoch_N checkpoint + Training Profiler + batch-size sweep + **Fast Row Encoder（4096 行 bitwise 等价）**
  - 训练：bc-v2.1 全量 5 epoch，**best = epoch 2**（已收敛，epoch 3–5 轻微过拟合）
  - 评估：**val acc 0.69788 / test acc 0.69782**（val≈test，无过拟合）
  - Benchmark：**30000 局**（greedy/sampling × random/bc-v1/rule × 5000 局/对手），全部占优
  - **未做**：E 数据质量（69 例 replay error 分类，按用户决定放弃）

## 最新 commit

- `828c715`（Phase 10.1.1：训练基础设施优化 + bc-v2.1 + 30000 局 Benchmark）
- `80fa706`（Phase 10.1：decision-v2 + bc-v2 + 10000 局 Benchmark）

## 版本

| 项 | 值 |
|---|---|
| 数据版本（dataset_version） | `decision-v2`（大规模）；`decision-v1`（早期 baseline） |
| feature/action schema 版本 | `feature-v1`（action `action-v1`） |
| 模型版本 | **`bc-v2.1`**（`experiments/bc_v2.1/checkpoints`，best = epoch 2）；`bc-v2`；`bc-v1`（`experiments/exp_0001/checkpoint`） |
| 数据源 | 天凤 / 雀魂 `.mjai.json`（全量 ~226 万文件） |

## 关键指标

| 模型 | 训练 | val acc | test acc |
|---|---|---|---|
| bc-v1 | decision-v1 小规模 | ~0.55 | — |
| bc-v2 | decision-v2 全量 1 epoch | 0.6792 | — |
| **bc-v2.1** | decision-v2 全量 5 epoch（best = epoch 2） | **0.69788** | **0.69782** |

Benchmark（bc-v2.1，每对手 5000 局，candidate mean_rank 越低越好）：

| 模式 | vs random | vs bc-v1 | vs rule |
|---|---|---|---|
| greedy | 0.0008 | 0.1586 | 0.9198 |
| sampling | 0.0064 | 0.3648 | 1.3374 |

## 下一阶段目标

- **可选训练**：按已验证的收敛结论，**2 epoch 全量重训已足够**；如需进一步提速可考虑预编码特征缓存；更大模型/更大 batch 待评估；
- **RL（PPO）阶段**：当前暂缓，需用户确认后再进入；
- 遗留：E 数据质量（69 例 replay error 分类）已放弃。

## 检查状态

**Phase 10.1**
- [x] feature version 统一（`feature-v1`）
- [x] dataset manifest + validation（`docs/DATASET_VALIDATION_REPORT.md`）
- [x] BC 全量训练（bc-v2）+ checkpoint
- [x] Benchmark ≥10000 局（`docs/BENCHMARK_BC_V2_REPORT.md`）

**Phase 10.1.1**
- [x] Fast Encoder bitwise 等价（old == fast，4096 行 `torch.equal`）
- [x] 吞吐达标（§23 目标 ≥50k / 理想 75k–100k+ → 实测 261k–645k）
- [x] Resume 正确恢复（epoch/global_step/optimizer/RNG）
- [x] `latest.pt` / `best.pt` 存在且可加载
- [x] Learning curve（小规模 3 epoch + 全量 5 epoch）
- [x] 完整 val + test 评估（`scripts/evaluate_bc.py`）
- [x] Benchmark Greedy/Sampling + Rule baseline + 双端指标（30000 局）
- [ ] E 数据质量（69 例错误分类）—— **放弃**
