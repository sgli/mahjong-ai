# Phase 10.3 Benchmark 详细报告

> 范围：Phase 10.1 / 10.1.1 / 10.2 / 10.3 的全部 Benchmark（统一协议、多 seed、双端指标）
> 生成日期：2026-10-04 ｜ Runner：`scripts/benchmark_parallel.py` ｜ 数据：`decision-v2` ｜ Feature：`feature-v1`(234) ｜ Action：`action-v1`(270) ｜ Environment：`tenhou-v2-rl` ｜ Reward：`reward-v1`

---

## 0. 摘要

**一句话结论：RL 基础设施（rollout→reward→GAE→PPO→checkpoint→resume→benchmark）已验证正确；但在当前配置下，PPO 训练不仅没有超过 BC-v2.1，反而随训练单调退化——Strategy improvement = FAIL/INCONCLUSIVE（Phase 10.3 §38 预判情形）。**

关键数据（多 seed 42/43/44，1000 局/对手，candidate `mean_rank`，1.5 = 与对手打成平手）：

| candidate | greedy vs rule | sampling vs rule | greedy vs BC-v2.1 | sampling vs BC-v2.1 |
|---|---|---|---|---|
| **BC-v2.1** | **0.5770** ±0.006 | **0.9527** ±0.005 | — | — |
| PPO-20k | 0.6430 ±0.016 | 1.2127 ±0.002 | 1.1053 ±0.029 | 1.7323 ±0.022 |
| PPO-50k | 0.7857 ±0.022 | 1.4350 ±0.020 | 1.2590 ±0.042 | 1.9533 ±0.019 |
| **PPO-100k** | 0.8533 ±0.024 | **1.5353** ±0.036 | 1.3803 ±0.007 | **2.0637** ±0.022 |

**四条曲线全部单调恶化**（越训越差），跨 seed 稳定（std ≤ 0.045）→ 系统性现象，非噪声。

---

## 1. Benchmark 协议（§5 固定项）

| 项 | 固定值 |
|---|---|
| rules | `tenhou-v1`（RulesConfig）｜Environment `tenhou-v2-rl` |
| feature / action | `feature-v1`（234 维）/ `action-v1`（270 actions） |
| seed 方案 | `game_seed = base_seed + game_index`（与并行度无关，确定性） |
| 候选座位 | `seat = game_index % 4`（**轮换**，消除座位偏差） |
| 对手构成 | 每个 match-up 固定单一对手类型 × 3 个座位 |
| 局数 | 1000 或 5000 局/对手（见各节） |
| 模式 | `greedy`（argmax，确定性）/ `sampling`（multinomial，rng-seeded） |
| 并行 | ProcessPoolExecutor，16 worker |
| 复现性 | 同 seed + 同局数 ⇒ 结果完全可复现；跨 seed 报 mean/std/min/max |

**指标定义**（双端：candidate + opponents 3 座聚合）：
`mean_rank`（越低越好，0=全 1 位，1.5=与对手均势）、`rank_distribution[1/2/3/4位]`、`win_per_game`、`deal_in_per_game`、`riichi_per_game`、`call_per_game`、`avg_win_points`、`mean_score_change`、`illegal_rate`。
（注：`*_per_game` = 次/局，**不是**百分比——Phase 10 核查项 M1 已改名修正。）

---

## 2. 历史 Benchmark（对照基线）

### 2.1 Phase 10.1 — bc-v2（5000 局/对手，seed 42）

| 对手 | mean_rank | rank 分布 | win/局 | 放铳/局 | 立直/局 | 副露/局 |
|---|---|---|---|---|---|---|
| random | 0.0120 | [4959, 28, 7, 6] | 2.623 | 0.011 | 1.638 | 8.770 |
| bc-v1 | 0.4552 | [3487, 960, 343, 210] | 2.373 | 0.297 | 1.809 | 8.142 |

### 2.2 Phase 10.1.1 — bc-v2.1（5000 局/对手，seed 42，30000 局）

| 模式 | 对手 | mean_rank | rank 分布 | win/局 | 放铳/局 | 立直/局 | 副露/局 | scoreΔ |
|---|---|---|---|---|---|---|---|---|
| greedy | random | 0.0008 | [4996, 4, 0, 0] | 4.444 | 0.0098 | 2.330 | 4.396 | +43671 |
| greedy | bc-v1 | 0.1586 | [4396, 454, 111, 39] | 3.734 | 0.2274 | 2.479 | 4.923 | +33159 |
| greedy | rule | 0.9198* | [2200, 1462, 877, 461] | 2.728 | 0.6158 | 2.191 | 4.994 | +13897 |
| sampling | random | 0.0064 | [4978, 16, 2, 4] | 2.773 | 0.010 | 1.649 | 7.435 | +27455 |
| sampling | bc-v1 | 0.3648 | [3729, 865, 259, 147] | 2.558 | 0.268 | 1.866 | 6.842 | +21404 |
| sampling | rule | 1.3374* | [1392, 1440, 1257, 911] | 1.959 | 0.735 | 1.596 | 6.866 | +4371 |

> ⚠️ **\* 与 §3 新基线不一致**：同 seed 42 下，此处 bc-v2.1 vs rule = greedy **0.9198** / sampling **1.3374**，而 §3 多 seed 基线 = **0.5770 / 0.9527**（差距很大）。
> 两次都使用 `scripts/benchmark_parallel.py`，但**版本不同**（2.2 是 t76 重写后、t86 支持 PPO 前的早期版本；§3 是当前版本），且局数不同（5000 vs 1000）。
> **以 §3 的当前多 seed 结果为准**；2.2 的 rule 数值应视为**已被取代**。**建议后续单独复测一次 5000 局/对手，确认是「版本差异」还是「局数效应」**（列为遗留待查项）。

### 2.3 Phase 10.2 — ppo-v1（10,891 steps，2000 局/对手，seed 42）

| 模式 | 对手 | mean_rank | rank 分布 | win/局 | 放铳/局 | 立直/局 | 副露/局 | scoreΔ |
|---|---|---|---|---|---|---|---|---|
| greedy | random | 0.0020 | [1996, 4, 0, 0] | 4.152 | 0.006 | 2.094 | 4.446 | +37685 |
| greedy | rule | 0.6185 | [1151, 541, 228, 80] | 3.494 | 0.426 | 2.132 | 5.346 | +18604 |
| greedy | bc_v2.1 | 1.0750 | [784, 557, 384, 275] | 2.635 | 1.040 | 2.174 | 4.446 | +7189 |
| sampling | random | 0.0150 | [1979, 15, 3, 3] | 2.332 | 0.009 | 1.274 | 9.319 | +21570 |
| sampling | rule | 1.1360 | [676, 639, 422, 263] | 2.217 | 0.625 | 1.526 | 8.778 | +5789 |
| sampling | bc_v2.1 | 1.7315 | [355, 457, 558, 630] | 1.666 | 1.321 | 1.453 | 7.494 | −3531 |

---

## 3. Phase 10.3 §5/§6 多 seed Baseline（**当前权威基线**）

**设置**：candidates = BC-v2.1 / PPO-v1(10.9k)；对局 = BC→{random, rule}、PPO→{rule, BC-v2.1}；**seeds 42/43/44**；**1000 局/对手**；模式 greedy + sampling；共 24000 局。

### 3.1 跨 seed 汇总（mean ± std，min–max）

| candidate | mode | 对手 | mean_rank | std | min–max |
|---|---|---|---|---|---|
| BC-v2.1 | greedy | random | **0.0007** | 0.0005 | 0.0000–0.0010 |
| BC-v2.1 | greedy | rule | **0.5770** | 0.0062 | 0.5690–0.5840 |
| BC-v2.1 | sampling | random | **0.0057** | 0.0029 | 0.0020–0.0090 |
| BC-v2.1 | sampling | rule | **0.9527** | 0.0050 | 0.9460–0.9580 |
| PPO-v1 | greedy | rule | **0.6243** | 0.0104 | 0.6170–0.6390 |
| PPO-v1 | greedy | BC-v2.1 | **1.0813** | 0.0137 | 1.0620–1.0920 |
| PPO-v1 | sampling | rule | **1.1547** | 0.0451 | 1.1160–1.2180 |
| PPO-v1 | sampling | BC-v2.1 | **1.7003** | 0.0279 | 1.6740–1.7390 |

### 3.2 逐 seed 明细（mean_rank / win per game / deal-in per game / scoreΔ）

| candidate | mode | 对手 | seed 42 | seed 43 | seed 44 |
|---|---|---|---|---|---|
| BC-v2.1 | greedy | random | 0.0000 / 4.570 / 0.009 / +42399 | 0.0010 / 4.593 / 0.011 / +42236 | 0.0010 / 4.659 / 0.009 / +42773 |
| BC-v2.1 | greedy | rule | 0.5690 / 3.786 / 0.464 / +21770 | 0.5840 / 3.643 / 0.478 / +20576 | 0.5780 / 3.684 / 0.491 / +20952 |
| BC-v2.1 | sampling | random | 0.0020 / 2.796 / 0.013 / +26304 | 0.0090 / 2.837 / 0.008 / +26726 | 0.0060 / 2.834 / 0.004 / +27710 |
| BC-v2.1 | sampling | rule | 0.9580 / 2.669 / 0.622 / +10408 | 0.9540 / 2.541 / 0.581 / +9808 | 0.9460 / 2.552 / 0.593 / +9240 |
| PPO-v1 | greedy | rule | 0.6390 / 3.514 / 0.432 / +18767 | 0.6170 / 3.458 / 0.410 / +18833 | 0.6170 / 3.426 / 0.435 / +17927 |
| PPO-v1 | greedy | BC-v2.1 | 1.0900 / 2.607 / 1.037 / +6776 | 1.0620 / 2.626 / 1.094 / +7204 | 1.0920 / 2.690 / 1.128 / +7036 |
| PPO-v1 | sampling | rule | 1.1160 / 2.253 / 0.615 / +6190 | 1.1300 / 2.170 / 0.599 / +5457 | 1.2180 / 2.198 / 0.686 / +4503 |
| PPO-v1 | sampling | BC-v2.1 | 1.7390 / 1.630 / 1.323 / −3813 | 1.6740 / 1.715 / 1.322 / −2908 | 1.6880 / 1.714 / 1.336 / −3446 |

**基线结论**：
1. **BC-v2.1 是强基线**：对 rule 达 greedy 0.577 / sampling 0.953（均明显优于均势 1.5），对 random 近乎全胜；
2. **PPO-v1 未超过 BC-v2.1**：对 rule 反而更差（0.624 vs 0.577；1.155 vs 0.953）；对 BC-v2.1 仅 greedy 略优（1.081 < 1.5），**sampling 明显落败**（1.700 且 scoreΔ 为负）。

---

## 4. Phase 10.3 §17 Stage C 综合 Benchmark（PPO 课程曲线）

**设置**：candidates = BC-v2.1 / PPO-20k / PPO-50k / PPO-100k（同一训练谱系，resume 累积：22,106 → 52,777 → 103,355 steps）；seeds 42/43/44；1000 局/对手；共 48000 局。

### 4.1 跨 seed 汇总（mean ± std）

| candidate | greedy vs rule | sampling vs rule | greedy vs BC-v2.1 | sampling vs BC-v2.1 |
|---|---|---|---|---|
| BC-v2.1 | **0.5770** ±0.006 | **0.9527** ±0.005 | — | — |
| PPO-20k | 0.6430 ±0.016 | 1.2127 ±0.002 | 1.1053 ±0.029 | 1.7323 ±0.022 |
| PPO-50k | 0.7857 ±0.022 | 1.4350 ±0.020 | 1.2590 ±0.042 | 1.9533 ±0.019 |
| PPO-100k | 0.8533 ±0.024 | **1.5353** ±0.036 | 1.3803 ±0.007 | **2.0637** ±0.022 |

### 4.2 训练曲线（同族 checkpoint，单调性一目了然）

| 指标 | BC-v2.1 | PPO-20k | PPO-50k | PPO-100k | 趋势 |
|---|---|---|---|---|---|
| greedy vs rule | 0.577 | 0.643 | 0.786 | 0.853 | **单调变差** |
| sampling vs rule | 0.953 | 1.213 | 1.435 | 1.535 | **单调变差（100k 已 >1.5）** |
| greedy vs BC | — | 1.105 | 1.259 | 1.380 | **单调变差（逼近均势）** |
| sampling vs BC | — | 1.732 | 1.953 | 2.064 | **单调变差（明显劣于 BC）** |

### 4.3 逐 seed 明细（mean_rank / win per game / scoreΔ）

| candidate | mode | 对手 | seed 42 | seed 43 | seed 44 |
|---|---|---|---|---|---|
| PPO-20k | greedy | rule | 0.6210 / 3.387 / +18125 | 0.6560 / 3.413 / +17966 | 0.6520 / 3.381 / +17276 |
| PPO-20k | greedy | BC-v2.1 | 1.1370 / 2.628 / +6791 | 1.1120 / 2.685 / +6982 | 1.0670 / 2.609 / +7044 |
| PPO-20k | sampling | rule | 1.2150 / 2.129 / +4309 | 1.2120 / 2.011 / +4196 | 1.2110 / 2.094 / +3973 |
| PPO-20k | sampling | BC-v2.1 | 1.7010 / 1.622 / −3998 | 1.7440 / 1.636 / −4354 | 1.7520 / 1.622 / −3797 |
| PPO-50k | greedy | rule | 0.7550 / 3.012 / +14453 | 0.8010 / 2.972 / +13524 | 0.8010 / 2.942 / +13198 |
| PPO-50k | greedy | BC-v2.1 | 1.2990 / 2.273 / +3358 | 1.2010 / 2.356 / +4921 | 1.2770 / 2.355 / +3709 |
| PPO-50k | sampling | rule | 1.4320 / 1.713 / −124 | 1.4610 / 1.614 / −170 | 1.4120 / 1.724 / +373 |
| PPO-50k | sampling | BC-v2.1 | 1.9270 / 1.317 / −7383 | 1.9680 / 1.276 / −7522 | 1.9650 / 1.287 / −7724 |
| PPO-100k | greedy | rule | 0.8310 / 2.776 / +12481 | 0.8860 / 2.725 / +11608 | 0.8430 / 2.748 / +11859 |
| PPO-100k | greedy | BC-v2.1 | 1.3760 / 2.132 / +1957 | 1.3900 / 2.093 / +2000 | 1.3750 / 2.190 / +2256 |
| PPO-100k | sampling | rule | 1.5800 / 1.483 / −2340 | 1.5340 / 1.548 / −1740 | 1.4920 / 1.570 / −1031 |
| PPO-100k | sampling | BC-v2.1 | 2.0940 / 1.131 / −9417 | 2.0500 / 1.190 / −9498 | 2.0470 / 1.182 / −8542 |

### 4.4 训练健康指标（同一谱系）

| 指标 | A (20k) | B (50k) | C (100k) | 观察 |
|---|---|---|---|---|
| policy_loss | 0.00097 | 0.00160 | 0.00017 | finite |
| value_loss | 6.820 | 4.568 | **3.231** | 持续下降（value 在学习） |
| **entropy** | 0.2780 | 0.3122 | **0.3418** | **单调上升**（与性能下降同时发生） |
| mean_ratio | 0.9988 | 0.99982 | 0.99970 | 健康（≈1） |
| approx_kl | 0.00622 | 0.00518 | 0.00465 | 健康（小） |
| clip_fraction | 0.0477 | 0.0439 | 0.0444 | 健康（小） |
| mean_reward | −0.0556 | −0.1250 | −0.0375 | finite（对手池混合，非纯自对弈） |
| illegal_rate | 0 | 0 | 0 | ✅ |
| NaN / Inf | 无 | 无 | 无 | ✅ |

**注意**：`mean_ratio≈1`、`approx_kl≈0.005`、`clip_fraction≈0.045` 都在健康区间（不像 PPO 数值爆炸），**但 entropy 单调上升 + 策略单调变差** → 典型「稳定地学坏」，而非训练不收敛。

---

## 5. 分析

### 5.1 §20 Strength Gate 判定（PPO-100k）

| Gate | 要求 | 结果 |
|---|---|---|
| A：vs Rule | PPO 应明显优于 Rule（且不能输） | ❌ greedy 0.853（优于 rule 但在退化）；**sampling 1.535 > 1.5（已输给 rule）** |
| B：vs BC | greedy 下 PPO mean rank < BC，且多 seed 方向一致 | ❌ PPO vs rule 0.853 **>** BC vs rule 0.577 → **PPO 不如 BC**，且 3 seed 方向一致（都是变差） |
| C：Sampling | 若长期显著劣于 BC → NOT READY | ❌ **是**（100k 2.064 vs BC 0.953；scoreΔ 为负） |
| D：illegal | `illegal_rate = 0` | ✅ |

### 5.2 诊断进展（已做 / 待做）

| 诊断 | 结论 |
|---|---|
| **#2 rollout log-prob/value/mask parity** | ✅ **已排除**：encode/mask bitwise 一致；logits/value 仅 <1e-5 浮点差；`ratio==1` 自检 ≤1.2e-7。**rollout parity 不是退化原因。**<br>过程中发现并修复一个真 bug（batch 路径 `old_log_prob` 曾写死 0.0），但**该路径未接入训练**，故不影响已跑结果。 |
| **#3 遗忘量化**（待做） | 量化 PPO 对 BC 的动作一致率随训练变化 → 验证是否**灾难性遗忘**（LR 1e-4 + 无 KL/BC 锚点） |
| **#1 更新正确性**（待做） | 审计 advantage/return/value 目标/clip/更新次序；加「zero-advantage → 策略不变」等机械性测试 |
| 其他疑点 | entropy 上升 / 探索 / 对手池过拟合 / reward 与强度是否对齐（§17） |

### 5.3 附带发现

1. **优化 rollout 未接入训练**：t88/t90 实现的 batch encode+forward（3,588 steps/s）**没有被 `train_ppo` 使用**（仍走逐步 reference，端到端 ~510 steps/s）。Phase 10.3 的「高吞吐」目前尚未兑现，Stage D/E（500k/1M）会明显偏慢。
2. **2.2 与 §3 的 bc-v2.1 vs rule 数值矛盾**（0.9198 vs 0.5770，同 seed 42）——需确认是「benchmark 脚本版本差异」还是「局数效应」，否则会影响所有基于 2.2 的对比叙述。
3. **PPO 现有最佳仍是 PPO-v1 / PPO-20k 量级**，继续训练（≥50k）在四项指标上**全部变差**。

---

## 6. 结论

```text
RL infrastructure            : PASS   （闭环正确/可复现/可 resume；parity 已排除）
Rollout 性能工程              : PASS   （已实现 3.5× 优化，但尚未接入训练 → 待接线）
Opponent Pool / Multi-seed    : PASS   （可复现、可配置、historical 注册可用）
Strength improvement          : FAIL / INCONCLUSIVE
```

- 按 **§38**：这是文档明确预判的情形 → **不应继续无止境增加算力**，而应进入**策略诊断**（Representation / Reward / Opponent Curriculum 的独立实验）；
- 按 **§18/§32**：**correctness gate 失败 → 停止扩容 → 定位问题**（本次已停止 500k/1M）；
- **PPO 目前尚未证明能稳定超过 BC-v2.1**；BC-v2.1（0.577 / 0.953 vs rule）仍是当前最强策略。

### 建议的下一步（按优先级）

1. **#3 遗忘量化**（最可能解释退化：PPO 把 BC 的能力训丢了）；
2. **#1 更新正确性机械性测试**（zero-advantage 不变、GAE/return 手算、value 目标）；
3. **把优化 rollout 接入训练**（兑现 3.5× 吞吐，为后续大规模实验降本）；
4. **复测 §2.2 的矛盾数值**（确定权威对照）；
5. 若 #1/#3 均排除 → 进入 §38 的独立实验阶段（reward shaping / opponent curriculum / representation），**且不得立刻改 reward**（§27）。

---

## 7. 附录：数据位置

| 数据 | 路径 |
|---|---|
| Phase 10.3 多 seed baseline（24000 局） | `experiments/benchmark_baseline/{bc_v2.1,ppo_v1}_{mode}_seed{N}/result.json` |
| Phase 10.3 Stage C benchmark（48000 局） | `experiments/benchmark_ppo_v2/{bc_v2.1,ppo_20k,ppo_50k,ppo_100k}_{mode}_seed{N}/result.json` |
| PPO 课程 checkpoint | `experiments/ppo_v2/milestones/ppo-{20k,50k,100k}/` |
| Phase 10.2 benchmark（12000 局） | `experiments/benchmark_ppo_v1_{greedy,sampling}/result.json` |
| Phase 10.1.1 benchmark（30000 局） | `experiments/benchmark_v21_{greedy,sampling}/result.json` |
| Runner | `scripts/benchmark_parallel.py` |
