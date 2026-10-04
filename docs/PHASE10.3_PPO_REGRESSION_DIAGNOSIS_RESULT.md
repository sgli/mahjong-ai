# Phase 10.3 PPO 回归诊断与修复：最终结果

> 对应规范：`docs/PHASE10.3_PPO_REGRESSION_DIAGNOSIS.md`（§1–§18）
> 阶段规范：`docs/PHASE10.3.md` ｜ 完成日期：2026-10-04
> 本文取代此前的 `PHASE10.3_BENCHMARK_REPORT.md` 与早期版本的诊断结论文档（两者已因下述**基准方法学缺陷**而作废/删除）

---

## 0. 最终结论（一句话版）

```text
RL Infrastructure            : PASS
Strategy Improvement         : 有限（marginal）
PPO 相对 BC-v2.1             : greedy 打平（3-seed 1.5043 vs 1.5020）；vs rule 小幅领先（0.5640 vs 0.5770）
sampling 相对 BC             : 打平或略差（T=0.3 打平；T=1.0 略差）
退化根因                      : D. 学习率过大导致的 BC 遗忘  +  G. 纯自对弈漂移（主因）
是否恢复 50k/1M 扩容          : 否（未达 §16 GO）
```

**三句话**：
1. **单调退化的根因已定位并部分修复**：主因是 **纯自对弈漂移（对手池可消除）** 与 **学习率 1e-4 过大（导致 BC 灾难性遗忘）**；`lr=2e-5` + `BC anchor β=1.0` + **对手池** 三者叠加后，**训练不再退化**。
2. **sampling 明显劣于 BC 的现象被证伪为「评估口径伪影」**：真实原因是**推理温度未校准**（策略分布过平），且旧版 harness **未把候选的 mode/温度传播给对手**，导致一批历史结论不可用（见 §7 撤回清单）。
3. **但增益有限**：最优配置 **F1** 在 3-seed 公平口径下与 BC **打平**（vs BC 1.5043 vs 1.5020），仅在 **vs rule** 上小幅领先（0.5640 vs 0.5770，≈2.1×std）。按 §16/§38 判据，属 **Infrastructure PASS / Strategy Improvement 有限**，**不构成扩大算力的证据**。

---

## 1. 规范 §1–§18 执行清单

| 章节 | 要求 | 状态 | 证据位置 |
|---|---|---|---|
| §2 | 定量化 BC forgetting（固定子集 6 指标 + 判定） | ✅ | `evaluation/retention.py`、`scripts/eval_bc_retention.py`、`tests/test_ppo_bc_retention.py`、`experiments/bc_retention/` |
| §3 | PPO update 机械审计（zero-advantage / ratio 三区间 / 正负 clip / 前后 logits·logprob·entropy / 方向） | ✅ | `tests/test_ppo_update_invariants.py`、`test_ppo_ratio_diagnostics.py`、`test_ppo_update_epochs.py` |
| §4 | GAE/return/reward 分布（按 epoch，**按 seat / 对手聚合**）+ reward conservation | ✅ | `test_ppo_advantage_distribution.py`、`test_ppo_reward_conservation.py`、`training/attribution.py` |
| §5 | entropy / exploration 诊断（含「不先改 ent_coef」） | ✅ | `evaluation/diagnosis.py`、`scripts/eval_entropy_value.py`、`test_diagnosis_metrics.py` |
| §6 | value 质量审计（MAE/RMSE/corr/EV） | ✅ | 同上 + `scripts/diag_advantage.py` |
| §7 | **必须**分别记录 update-epoch KL / ratio 统计 / clip 正负比例 | ✅ | `diagnose` 已暴露且**默认开启**（`training.diagnose: true`） |
| §8 | 优化 rollout 接入正式训练 + 全字段 parity（**性能优化不得改变语义**） | ✅ | 默认 `rollout-v2`+`rng-v2`；`tests/test_ppo_rollout_parity.py`（greedy 精确、**sampling 逐位**） |
| §9 | 复核历史 benchmark 矛盾（1000/5000 局 + 记录元数据） | ✅ | `experiments/benchmark_rule_recheck/`（0.5690 vs 0.5678 → runner 版本差异） |
| §10 | 单变量修复实验 A/B/C/D + seed42 筛选 + 3-seed | ✅ | `configs/experiments/ppo_fix/*.yaml`(9) + README |
| §11 | 按对手类型记录 episode reward/return/win/rank；curriculum 暂缓 | ✅ | 对手池全链路 + **pool_a 产出真实 6 类对手归因数据** |
| §12 | reward 暂禁 → 前置满足后才做 reward 实验 | ✅ | 前置（§11）已满足 → F 实验已完成 |
| §13 | 推荐执行顺序 1–18 | ✅ | 全部完成（含 17 接入优化 rollout、18 未扩容） |
| §14 | 禁止事项 | 🟡 2 项需说明 | 见 §8 边界说明 |
| §15 | 指定测试文件 + 覆盖清单 | ✅ | 7 个指定文件名齐全（含 `test_ppo_rollout_parity.py`）+ 10 项覆盖 |
| §16 | GO / NO-GO | ✅ 已判定 | **未达 GO**；NO-GO 后续独立实验（Reward/Representation/Opponent）已执行 |
| §17 | 修复完成标准 16 项 | ✅ 全部可测 | 见 §6 |
| §18 | 根因归类 A–H | ✅ | 见 §3 |

**规范已无未完成项。**

---

## 2. 关键实验数据

### 2.1 BC Retention（固定子集：`decision-v2/test` 字典序前 3 shard = 30 万决策，确定性）

| 模型 | agreement ↑ | KL(BC‖PPO) ↓ | KL(PPO‖BC) ↓ | PPO entropy ↓ |
|---|---|---|---|---|
| BC-v2.1（参照） | 1.0000 | 0 | 0 | 0.7881 |
| PPO-20k | 0.8297 | 0.1059 | 0.1239 | 0.9125 |
| PPO-50k | 0.7855 | 0.1698 | 0.2179 | 0.9917 |
| PPO-100k | **0.7561** | **0.2210** | **0.2997** | **1.0472** |
| B（lr=5e-5） | 0.8449 | 0.0878 | 0.0983 | 0.8750 |
| C（lr=2e-5） | 0.8540 | 0.0785 | 0.0845 | 0.8430 |
| D1–D3（β=0.001/0.003/0.01） | 0.8300 | ~0.1055 | — | ~0.9125 |
| D4/D5（β=1.0/3.0） | 0.8508 / **0.8604** | 0.0860 / 0.0775 | — | 0.8990 / 0.8829 |
| **D6（lr=2e-5+β=1.0）** | **0.8599** | **0.0735** | 0.0789 | 0.8420 |
| E1（ent_coef=0） | 0.8297 | 0.1058 | — | 0.9107 |
| **F1（最终配置）** | **0.8579** | 0.0754 | 0.0802 | 0.8348 |

**结论**：`agreement↓ + KL↑ + entropy↑` 随训练单调成立 → **BC 遗忘确认**；**学习率是驱动因素**（B/C 修复）；**ent_coef 不是**（E1 无效）；**规格 §10 建议的 β=0.001–0.01 量级不足**，需 β≥1.0。

### 2.2 公平口径强度（candidate vs `bc_v2.1` 对手，**两侧同 mode/温度**，seed 42，1000 局/对手，greedy）

> 镜像基线（BC 候选 vs BC 对手）= **1.5020**（证明 harness 公平：1.50 ≈ 1.5）

| 配置 | vs BC | vs rule |
|---|---|---|
| BC-v2.1（基准） | 1.5020 | 0.5690 |
| g999（lr2e-5+β1.0+γ0.999，纯自对弈）20k/42k/105k | 1.4760 / 1.5270 / **1.6270** | — |
| ec1a（critic 预热 100k 后 PPO）21k/43k | 1.4670 / **1.5860** | 0.5690 / 0.5840 |
| E-B clip2（reward 截断 ±2）20k/44k | 1.4890 / 1.5340 | 0.5460 / 0.5770 |
| E-B clip5（±5）20k/44k | 1.4870 / 1.5610 | 0.5390 / 0.5700 |
| **pool_a（对手池）** 20k/44k/66k/88k/110k | 1.5160 / **1.4880** / 1.4940 / 1.5020 / 1.5460 | 0.5500 / 0.5590 / **0.5520** / 0.5620 / 0.5700 |
| **F1（pool+纯顺位）** 20k/44k | 1.4930 / **1.4850** | 0.5590 / **0.5560** |
| F2（pool+score×1/5）20k/44k | 1.5070 / 1.4820 | 0.5580 / 0.5570 |
| H1（独立 value trunk）20k/44k | 1.4930 / 1.5130 | 0.5520 / 0.5620 |

**关键读法**：
- **纯自对弈下，所有配置在 20k 之后单调退化**（g999/ec1a/E-B 全部如此）；
- **启用对手池后，20k→88k 不再退化**（pool_a：1.516→1.488→1.494→1.502）→ **自对弈漂移是退化的主因**；
- **F1 是当前最佳**：两个时间点、两个对手都优于 BC，且不退化；
- 幅度**小**（vs rule 领先约 0.013，vs BC 约打平）。

### 2.3 F1 的三项验证（§17）

**① 3-seed 公平 greedy**

| seed | vs BC | vs rule |
|---|---|---|
| 42 / 43 / 44 | 1.4850 / 1.5530 / 1.4750 | 0.5560 / 0.5650 / 0.5710 |
| **均值** | **1.5043 ± 0.0347** | **0.5640 ± 0.0062** |
| BC-v2.1 对照 | 1.5020（镜像） | **0.5770 ± 0.0062** |

→ **vs rule：小幅但一致领先（+0.013，≈2.1×std）**；**vs BC：打平**。

**② sampling 公平对抗（seed42，两侧同配置）**

| T | F1 | BC 镜像 | 判定 |
|---|---|---|---|
| 1.0 | 1.5370 | 1.4730 | 略差 |
| 0.3 | 1.4930 | 1.4740 | 打平 |

**③ retention**：F1 agreement **0.8579**、KL(BC‖PPO) 0.0754（≈最优修复候选，远好于退化后的 0.756）✅

### 2.4 温度扫描（解释 sampling 差距）

| T | BC-v2.1 | D6 | 谁优 |
|---|---|---|---|
| 1.0 | **0.9580** | 1.0210 | BC |
| 0.5 | 0.6860 | **0.6260** | D6 |
| 0.3 | 0.5940 | **0.5460** | D6 |
| 0.1 | **0.5350** | 0.5460 | BC（≈噪声） |

**结论**：sampling 差距是**未校准温度（T=1.0）下的伪影**；在各自最优温度下两者打平（BC 0.5350@T=0.1 vs D6 0.5460@T=0.3）。

---

## 3. 各轴结论（§18 根因归类）

| 轴 | 判定 | 证据 |
|---|---|---|
| **A. PPO update 实现** | ❌ **排除** | zero-advantage / ratio / clipping / update 方向 / GAE / reward conservation 全项通过 |
| **B. advantage / return 实现** | ❌ **排除** | GAE 已知序列精确、advantage 归一化生效 |
| **C. value 估计** | 🟡 **次要** | EV 长期为负（BC −0.069、g999 20k −0.032→105k −0.185）；但 **critic 预热到 EV>0 后仍退化**（ec1a 21k 1.4670→43k 1.5860） |
| **D. BC 灾难性遗忘** | ✅ **主因之一** | lr=1e-4 下 agreement 0.830→0.756；`lr=2e-5` 修复到 0.854 |
| **E. entropy / 探索** | ❌ **排除** | ent_coef=0（E1）几乎无效果 |
| **F. reward 对齐** | ✅ **有效（小幅）** | score 项占 reward 91% 而评估为 mean_rank；纯顺位奖励（F1）在两个指标上都略优于 BC |
| **G. 对手课程 / 自对弈漂移** | ✅ **主因（最大收益）** | 纯自对弈 20k 后必退化；**启用对手池后 20k→88k 不退化** |
| **H. 表示层** | ❌ **排除** | 独立 value trunk（H1，梯度隔离）44k 反退化到 1.5130，无增益 |
| （附）γ 折扣 | ❌ 单独无效 | γ=0.999 单独不改变退化；但它是**顺位奖励可见**的前提（γ^614: 0.0021→0.54） |

**最合理机制链**：
```text
纯自对弈（所有 4 席同一策略）
  + lr=1e-4 过大
  → value 梯度经共享 trunk 影响策略 / BC 能力被遗忘
  → 策略漂移（小幅 KL 0.03 却掉强度 0.12）
  → 越训越差
修复：对手池（消除漂移） + lr=2e-5 + BC anchor β=1.0 + γ=0.999
```

---

## 4. 最终最佳配置（F1）

```yaml
# experiments/ppo_fix/configs/f1.yaml
对手池: random-v1 0.10 / rule-v1 0.20 / bc-v2.1 0.30 / ppo-v1 0.20 / historical 0.20
reward_score_scale: 0.0          # 纯顺位奖励（placement [2,1,-1,-2]）
lr: 2.0e-5
bc_anchor_beta: 1.0
gamma: 0.999
value_separate_trunk: false      # 共享 trunk（H1 已证独立 trunk 无益）
rollout_version: rollout-v1      # 与既有实验可比；rollout-v2/rng-v2 亦可
seed: 42
epochs: 2                        # ≈44k steps 为最佳点（再久会退化）
```

**复现**：
```powershell
.venv\Scripts\python.exe scripts\train_ppo.py --config configs\experiments\ppo_fix\f1.yaml --epochs 2
.venv\Scripts\python.exe scripts\benchmark_parallel.py --candidate experiments\ppo_fix\f1\epoch_2.pt ^
  --candidate-type ppo --candidate-mode greedy --candidate-temperature 1.0 ^
  --opponent-mode greedy --opponent-temperature 1.0 ^
  --opponents bc_v2_1,rule --games-per-opponent 1000 --seed 42 --num-workers 16 --out <out>
```

---

## 5. §16 Go / No-Go

| GO 判据 | 结果 |
|---|---|
| BC retention 明显高于原 PPO | ✅ 0.8579（F1）vs 0.8297 |
| entropy 不再异常上升 | ✅ 0.8348 vs 0.9125 |
| **sampling vs BC 明显改善** | ❌ **未满足**（F1 sampling：T=1.0 略差、T=0.3 打平） |
| greedy vs Rule 不恶化 | ✅ 0.5640 vs BC 0.5770（小幅领先） |
| correctness 无 regression | ✅ illegal_rate 0，pytest 全绿 |

**NO-GO 分支（规范要求）**：`停止扩大 PPO → 转入 Representation / Reward / Curriculum 独立实验` —— **已执行**（H1 / F / 对手池），结论：**Reward 与 Opponent 有效，Representation 无效**。

**因此：不恢复 50k → 100k → 500k → 1M。** 扩容需先出现更强的增益证据（§38）。

---

## 6. §17 修复完成标准（16 项）

| # | 项目 | 状态 |
|---|---|---|
| 1 | BC retention 已量化 | ✅ |
| 2 | PPO update mechanical audit PASS | ✅ |
| 3 | zero-advantage invariant PASS | ✅ |
| 4 | ratio/clipping regression PASS | ✅ |
| 5 | reward conservation PASS | ✅ |
| 6 | GAE/return distribution 已审计 | ✅ |
| 7 | update-epoch KL 已记录 | ✅（默认开启） |
| 8 | entropy behavior 已解释 | ✅（决策随机化，非合理探索） |
| 9 | value quality 已评估 | ✅（EV 全负，随训练恶化） |
| 10 | benchmark historical discrepancy 已解决 | ✅（runner 版本差异） |
| 11 | optimized rollout/reference parity PASS | ✅（greedy 精确 + sampling 逐位） |
| 12 | optimized rollout 已接入训练 | ✅（默认 `rollout-v2`+`rng-v2`） |
| 13 | 至少一个修复方案在 20k 有效 | ✅（D6 / F1） |
| 14 | 3 seed 验证有效 | ✅ 已测：**「不劣于 BC」成立**；「超过 BC」仅 vs-rule 小幅成立 |
| 15 | sampling vs BC 不再系统性恶化 | ✅ 已测：greedy 打平、sampling 打平/略差（早期大幅恶化已消失） |
| 16 | greedy vs Rule 不持续恶化 | ✅ |

---

## 7. 重要：基准方法学缺陷与**撤回清单**

`scripts/benchmark_parallel.py` 修复前存在缺陷：**`--candidate-mode` 与 `--candidate-temperature` 只作用于候选，未传播给模型对手**（对手硬编码 `sampling` / `T=1.0`）。证据：用 `bc_v2.1` 同时作候选与对手（理论镜像）得到 **1.1640（T=0.5）/ 1.0480（T=0.3）/ 0.9630（T=0.1）≠ 1.5**。

**已修复**：新增 `--opponent-mode` / `--opponent-temperature`（默认 `sampling` / `1.0`，保持旧行为），并写入 `result.json`（含 `runner_version`、`created_at`）。

### 撤回清单（不得再引用）

| 结论 | 出处 | 原因 |
|---|---|---|
| PPO-v1 greedy vs BC-v2.1 = 1.0750 | Phase 10.2 | mode 未传播 |
| PPO-20k/50k/100k greedy vs BC = 1.1053/1.2590/1.3803 | Phase 10.3 Stage C | mode 未传播 |
| BC-v2.1 greedy vs bc-v1 = 0.1586 等 `vs bc_*` greedy 数字 | Phase 10.1.1 | mode 未传播 |
| 「D6 greedy 0.9983 首次超过 BC」 | 本阶段早期 | mode 未传播 |
| T≠1.0 下的 `vs bc_v2.1` 正面对抗（如 1.1590） | 本阶段早期 | 温度未传播 |

**仍有效**：所有 `vs rule` / `vs random` 数字；`sampling @ T=1.0` 的 `vs bc_v2.1`（两侧对称）；§2.2 表中**两侧同配置**的全部新数据。

---

## 8. §14 禁止事项的边界说明（如实登记）

| 项 | 说明 |
|---|---|
| 「不得大规模 hyperparameter sweep」 | 本阶段共执行约 **15 次训练**（B/C/D1-6/E1/g999/ec1a/clip2/clip5/pool_a/f1/f2/h1）。**性质为逐轴单变量诊断**（每次由上一轮证据驱动，非盲扫），但数量如实登记 |
| 「不得因 benchmark 变差修改 benchmark」 | 修改了 harness（`--opponent-mode/--opponent-temperature` + result 元数据），**性质为修复公平性缺陷**（并导致上表结论作废），非迁就结果 |
| 「不得同时改 reward+lr+entropy」 | F1 的 reward 改动与基线（pool_a）**保持同一 lr**，为单变量；lr 的改动来自更早的独立实验 D |
| 其余 7 条 | 均未违反（未扩容至 500k/1M、未改 feature/action/BC-v2.1/环境规则、未删里程碑、未用 training reward 宣称变强） |

---

## 9. 本阶段新增产物（代码 / 测试 / 配置）

**源码**：`evaluation/retention.py`、`evaluation/diagnosis.py`、`training/attribution.py`、`training/opponent_pool.py`、`training/parallel_rollout.py`、`training/ppo.py`（rollout_version/rng_version/diagnose/warmup/reward_scale/value_separate_trunk/opponent_pool 等）、`training/ppo_checkpoint.py`（v2 字段）、`model/policy.py`、`scripts/train_ppo.py`、`scripts/benchmark_parallel.py`、`scripts/eval_bc_retention.py`、`scripts/eval_entropy_value.py`、`scripts/eval_sampling_gap.py`、`scripts/diag_advantage.py`、`scripts/profile_rollout.py`、`scripts/cross_validate_replay.py`

**配置**：`configs/train_ppo_v2.yaml`（默认 `rollout-v2`+`rng-v2`+`diagnose: true`）、`configs/experiments/ppo_fix/{b,c,d1..d6,e1}.yaml` + README

**测试**：`test_ppo_update_invariants`、`test_ppo_ratio_diagnostics`、`test_ppo_update_epochs`、`test_ppo_advantage_distribution`、`test_ppo_reward_conservation`、`test_ppo_bc_retention`、`test_ppo_bc_anchor`、`test_ppo_rollout_parity`、`test_rollout_parity`、`test_rollout_logprob_parity`、`test_diagnosis_metrics`、`test_policy_temperature`、`test_benchmark_opponent_propagation`、`test_benchmark_result_metadata`、`test_training_opponent_pool`、`test_ppo_opponent_attribution`、`test_ppo_warmup`、`test_reward_scale_override`、`test_value_separate_trunk`、`test_opponent_device`、`test_double_yakuman` 等

**pytest：437 passed**

---

## 10. 剩余风险与建议

| 风险 | 说明 |
|---|---|
| R1 增益有限 | 最优配置仅与 BC 打平（vs BC）、小幅领先（vs rule +0.013）；**不足以支撑大规模扩容** |
| R2 长训练仍会退化 | pool_a 在 110k 又退化（1.5460）→ 对手池延缓但未彻底消除漂移 |
| R3 评估分辨率 | 单次 1000 局的 mean_rank 跨 seed 波动可达 ±0.035；结论需以 3-seed 为准 |
| R4 F1 的 sampling 未占优 | T=1.0 略差、T=0.3 打平 |
| R5 价值函数长期为负 EV | 属 C（残余），critic 预热/独立 trunk 均未根治 |
| R6 `rng-v2` 可比性 | 默认已切 `rollout-v2`+`rng-v2`；与既有 `rng-v1` 实验**不可逐位比较**（复现旧实验需 `--rollout-version rollout-v1 --rng-version rng-v1`） |

**建议方向（若要继续）**：按 §38「算力扩张须有增益证据」，当前证据仅支持**小步前进**：优先做 ① 更强/更长的对手课程与 curriculum 调度；② 奖励的进一步对齐（局内顺位/放铳惩罚等）；③ 评估分辨率提升（更多 seed/局数）。**不建议**直接上 500k/1M。
