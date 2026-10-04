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
- **Phase 10.2**：PPO / RL 闭环验证与 Pilot（见 `docs/PHASE10.2_RESULT.md`）
  - **门禁**：§4–§16 全部 P0/P1 实现并配 regression test；**独立门禁审计 PASS**（reviewer 自行复跑确认无弱断言、无 §3 违规）
  - 修复 **2 个真实 bug**：荒牌流局 deltas 丢失、立直棒结算缺失（均配回归测试；`validate_rules` 匹配率无回归）
  - **Pilot-A/B/C**（target 2k/5k/10k，实际 **2,371 / 6,964 / 10,891** steps）：**全部健康**（全指标 finite、无 NaN/Inf、`illegal_rate=0`、env 正常结束、checkpoint/resume 正常）
  - **Benchmark 12000 局**：PPO-v1 对 random / rule 占优；对 BC-v2.1 greedy 略优、**sampling 落败**（如实记录，符合 §32 定位）
  - **判定：PASS**（§28 DoD 逐条通过）→ 允许进入 Phase 10.3（**需用户决定**，§31.11 不得自动进入）

## 最新 commit

- `118ce32`（Phase 10.1.1 STATUS 更新）
- `828c715`（Phase 10.1.1：训练基础设施优化 + bc-v2.1 + 30000 局 Benchmark）
- Phase 10.2 全套见 git log 最新提交

## 版本

| 项 | 值 |
|---|---|
| 数据版本（dataset_version） | `decision-v2`（大规模）；`decision-v1`（早期 baseline） |
| feature/action schema 版本 | `feature-v1` / `action-v1`（270 actions） |
| Environment 版本 | **`tenhou-v2-rl`**（RL 用，Phase 10.2 §9 固定） |
| Reward 版本 | **`reward-v1`**（0.001×score_delta + placement [2,1,−1,−2]） |
| 模型版本 | **`ppo-v1`**（`experiments/ppo_v1`，Pilot-C 10,891 steps）；**`bc-v2.1`**（`experiments/bc_v2.1/checkpoints`，best = epoch 2）；`bc-v2`；`bc-v1` |
| 数据源 | 天凤 / 雀魂 `.mjai.json`（全量 ~226 万文件） |

## 关键指标

### BC

| 模型 | 训练 | val acc | test acc |
|---|---|---|---|
| bc-v1 | decision-v1 小规模 | ~0.55 | — |
| bc-v2 | decision-v2 全量 1 epoch | 0.6792 | — |
| **bc-v2.1** | decision-v2 全量 5 epoch（best = epoch 2） | **0.69788** | **0.69782** |

### BC-v2.1 Benchmark（每对手 5000 局，candidate mean_rank 越低越好）

| 模式 | vs random | vs bc-v1 | vs rule |
|---|---|---|---|
| greedy | 0.0008 | 0.1586 | 0.9198 |
| sampling | 0.0064 | 0.3648 | 1.3374 |

### PPO-v1 Pilot（全指标 finite，illegal_rate 0）

| Pilot | target / 实际 steps | policy_loss | value_loss | entropy | approx_kl | clip_fraction |
|---|---|---|---|---|---|---|
| A | 2,000 / 2,371 | −0.00034 | 4.09 | 0.2775 | 0.00209 | 0.0092 |
| B | 5,000 / 6,964 | −0.00120 | 7.43 | 0.2766 | 0.00458 | 0.0265 |
| C | 10,000 / 10,891 | −0.00058 | 7.09 | 0.2812 | 0.00645 | 0.0456 |

### PPO-v1 Benchmark（每对手 2000 局，candidate mean_rank）

| 模式 | vs random | vs rule | vs BC-v2.1 |
|---|---|---|---|
| greedy | 0.0020 | 0.6185 | 1.0750 |
| sampling | 0.0150 | 1.1360 | **1.7315** ⚠️（正面对抗落败） |

## 下一阶段目标

- **Phase 10.3（大规模 PPO / Self-play）**：需**用户确认**后启动（§31.11 不得自动进入）；
- 进入前建议优先处理：**R1** PPO 对 BC 的增益验证（当前 sampling 正面对抗 BC-v2.1 落败）、**R3** 多 seed、**R4** rollout 吞吐（并行 env / vectorized）、**R2** `double_yakuman` 未接线；
- 遗留（已知、非阻塞）：E 数据质量（69 例 replay error 分类）放弃；profiler `gpu_util` 为 None（需外部 nvidia-smi）。

## 检查状态

**Phase 10.1 / 10.1.1** —— 已完成（见 `docs/PHASE10.1_RESULT.md`、`docs/PHASE10.1.1_RESULT.md`）

**Phase 10.2（§28 DoD）** —— 全部通过
- [x] bc-v2.1 best.pt 加载；PPO trunk/action head 与 BC 严格一致（bitwise）；value head 重初始化
- [x] Environment regression PASS；Replay↔Environment 交叉验证 PASS（3581 点 0 mismatch）；environment_version 明确
- [x] reward-v1 明确；reward attribution / terminal reward regression PASS（§10.1 八条均有真实数值断言）
- [x] GAE / PPO ratio / PPO clipping regression PASS
- [x] trajectory schema 明确（`ppo_buffer.py` 并接入 rollout）；PPO metrics 完整（14 项）；approx_kl / clip_fraction 已实现；advantage normalization 语义固定（rollout-wide）
- [x] PPO checkpoint 完整（含 optimizer/RNG/metadata）；PPO resume PASS（Run A == Run B，bitwise）；fixed seed reproducibility PASS
- [x] Pilot-A / B / C PASS；无 NaN/Inf；`illegal_rate = 0`；Environment 无异常终止；checkpoint 可加载；resume 可继续
- [x] PPO Benchmark 完成（12000 局）；BC-v2.1 / PPO / Rule / Random 对比完成；Greedy + Sampling 均完成
- [x] `docs/PHASE10.2.md` 已保存；`docs/PHASE10.2_RESULT.md` 已生成；`STATUS.md` 已更新
