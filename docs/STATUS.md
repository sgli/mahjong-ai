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
- **Phase 10**：BC Baseline 训练 + Benchmark（`experiments/exp_0001`，BC vs Random 已跑通）

## 最新 commit

- `42cb626`

## 版本

| 项 | 值 |
|---|---|
| 数据版本（dataset_version） | `decision-v1` |
| feature/action schema 版本 | `feature-v1`（action `action-v1`） |
| 模型版本 | `bc-v1`（`experiments/exp_0001/checkpoint`） |
| 数据源 | 天凤/雀魂 `.mjai.json` |

## 下一阶段目标

- 后续优化：在稳定 BC baseline 基础上，可考虑 Rule baseline 对比、更大数据训练；再进入 RL（PPO）阶段（暂缓大规模训练）。

## 训练前检查状态

- [x] feature version 统一（`feature-v1`）
- [x] dataset manifest 存在（`data/processed/decision-v1/manifest.json`）
- [x] dataset validation（见 `docs/DATASET_VALIDATION_REPORT.md`）
- [x] BC smoke test
- [x] checkpoint 保存/加载
- [x] Benchmark 流程（见 `docs/BENCHMARK_REPORT.md`）
