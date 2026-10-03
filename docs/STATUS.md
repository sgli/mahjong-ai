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
  - 数据集：`data/processed/decision-v2`（2,257,268 局 / 1,465,275,164 样本 / ~113GB / 14,674 parquet）
  - 训练：bc-v2，1 epoch、13.2 小时、val acc **0.6792** / top3 **0.9170** / illegal_rate 0
  - Benchmark：10000 局（vs random 5000 + vs bc-v1 5000），并行 16 worker / 24.2 分钟，见 `docs/BENCHMARK_BC_V2_REPORT.md`

## 最新 commit

- `80fa706`（Phase 10.1：decision-v2 + bc-v2 + 10000 局 Benchmark）
- `2b2092b`（Phase 10 核查修复）

## 版本

| 项 | 值 |
|---|---|
| 数据版本（dataset_version） | `decision-v2`（大规模）；`decision-v1`（早期 baseline） |
| feature/action schema 版本 | `feature-v1`（action `action-v1`） |
| 模型版本 | `bc-v2`（`experiments/bc_v2/checkpoints`）；`bc-v1`（`experiments/exp_0001/checkpoint`） |
| 数据源 | 天凤 / 雀魂 `.mjai.json`（全量 ~226 万文件） |

## 下一阶段目标

- 可选优化：预编码特征缓存（训练提速）、更大规模/更多 epoch 训练、补充规则基线对手（Rule baseline）；
- 之后再进入 RL（PPO）阶段（当前暂缓）。

## 训练前检查状态（Phase 10.1）

- [x] feature version 统一（`feature-v1`）
- [x] dataset manifest 存在（`data/processed/decision-v2/manifest.json`）
- [x] dataset validation（见 `docs/DATASET_VALIDATION_REPORT.md`）
- [x] BC smoke + 全量训练（bc-v2，1 epoch）
- [x] checkpoint 保存/加载（`experiments/bc_v2/checkpoints`）
- [x] Benchmark ≥10000 局（见 `docs/BENCHMARK_BC_V2_REPORT.md`）
