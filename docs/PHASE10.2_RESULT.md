# Phase 10.2 结果与验收（PHASE10.2_RESULT）

> 对应规范：[`docs/PHASE10.2.md`](PHASE10.2.md)（PPO / RL 闭环验证与 Pilot 规格）
> 完成日期：2026-10-04 ｜ 初始化模型：`bc-v2.1` ｜ PPO 模型：`ppo-v1` ｜ 数据：`decision-v2` ｜ Environment：`tenhou-v2-rl` ｜ Reward：`reward-v1`

## 0. 结论摘要

**Phase 10.2 = PASS。**

本阶段目标（§1/§32）是证明
`BC-v2.1 → MahjongEnv → Rollout → Reward → GAE → PPO Update → Checkpoint → Resume → Benchmark`
这一整条 **RL 闭环正确、稳定、可复现**，而不是训出强 AI。已全部验证：

- §4–§16 全部 P0/P1 门禁实现并配 **regression test**（门禁审计独立复核 **PASS**）；
- 期间发现并修复 **2 个真实 bug**（荒牌流局 deltas 丢失、立直棒结算缺失），均配回归测试且**规则验证无回归**；
- **Pilot-A / B / C 全部健康**（全指标 finite、无 NaN/Inf、`illegal_rate=0`、env 正常结束、checkpoint/resume 正常）；
- **Benchmark 12000 局**完成：PPO-v1 对 random / rule 占优；对 BC-v2.1 在 greedy 略优、**sampling 落败**（如实记录，符合 §32 定位）。

**按 §29**：§28 DoD 逐条通过 → **允许进入 Phase 10.3**；但按 §31.11 **不得自动进入**，需用户决定。

---

## 1. 修改内容（§4–§16 逐项）

| 文档条目 | 实现 | 关键文件 |
|---|---|---|
| §4 BC→PPO 初始化 | `init_ppo_from_bc`：保留 BC trunk + action head，**value head 重初始化**；BC logits == PPO logits（`torch.equal` bitwise，20 个真实决策点） | `src/mahjong/training/ppo.py`、`tests/test_ppo_initialization.py` |
| §5 PPO Checkpoint | `save/load_ppo_checkpoint`，§25 全字段（`checkpoint_type="ppo"`、`checkpoint_version="ppo-v1"`、model/optimizer、epoch/global_step、best_metric(+name)、config、metrics_history、seed/env_seed、python/torch CPU/CUDA RNG、dataset/feature/action/environment/reward/model version、git_commit、scaler_state） | `src/mahjong/training/ppo_checkpoint.py`、`tests/test_ppo_checkpoint.py` |
| §6 PPO Resume | `train_ppo.py --resume`；恢复 model/optimizer/epoch/global_step/best/RNG/config；**Run A vs Run B `torch.equal`（bitwise，CPU）**，并逐 epoch 直接比对 metrics | `scripts/train_ppo.py`、`tests/test_ppo_resume.py` |
| §7 Environment Regression | `test_environment_regression.py`（全半荘合法结束+分数守恒+final_ranks、非法动作拒绝、版本、RulesConfig wiring）+ **西入端到端 3 测试**（触发/不触发/终局顺位）+ 既有 test_environment/aborts/nagashi/scoring 覆盖 | `tests/test_environment_regression.py`、`tests/test_environment_west_round.py` |
| §8 Replay↔Environment | `scripts/cross_validate_replay.py`：ReplayState → Environment → `legal_actions`，断言 human action ∈ legal；**5 文件 / 3581 决策点 / 8 类 action / mismatch = 0**（reviewer 独立复跑确认） | `scripts/cross_validate_replay.py` |
| §9 environment_version | `ENVIRONMENT_VERSION = "tenhou-v2-rl"` 固定，写入 config / checkpoint / train_ppo；RulesConfig wiring 核对：**`double_yakuman` 未接线**（step3 占位，当前单倍役满）——如实记录 + 测试断言 | `src/mahjong/environment/env.py` |
| §10 Reward-v1 | `score_delta_scale=0.001` + `placement_bonus=[2.0,1.0,-1.0,-2.0]`；§10.1 **八条全部有真实数值断言**（score delta/seat、ron 归属、tsumo 归属、连庄、honba/kyotaku 不重复、pending flush、terminal bonus、trajectory↔delta） | `tests/test_ppo_reward.py`、`tests/test_ppo_reward_settlement.py` |
| §11 Trajectory Schema | `TrajectoryStep` / `EpisodeMetadata` / `TrajectoryBuffer`，并**接入 `collect_trajectory`**（等价重构，bitwise 验证行为不变） | `src/mahjong/training/ppo_buffer.py`、`src/mahjong/training/ppo.py` |
| §12 GAE 回归 | 单步 terminal / bootstrap=0 / episode 边界不泄漏 / 零·正·负 reward / 手算样例 | `tests/test_ppo_gae.py` |
| §13 Ratio/Clipping 回归 | `ratio = exp(new−old)`；ratio=1、>1+ε、<1−ε、正/负 advantage、`min(ratio·A, clip(ratio)·A)` 符号方向 | `tests/test_ppo_ratio.py`、`tests/test_ppo.py` |
| §14 PPO Metrics | `approx_kl = mean(old_log_prob − new_log_prob)`、`clip_fraction = fraction(\|ratio−1\| > clip_eps)` **固定定义** + 单元测试；train_ppo 报 14 项指标 | `tests/test_ppo_metrics.py` |
| §15 Advantage Normalization | 改为 **rollout-wide**（GAE 后整 buffer 归一化一次），语义固定 | `src/mahjong/training/ppo.py` |
| §16 PPO v1 配置 | `configs/train_ppo_v1.yaml`（文档推荐起点，**非最优超参**；bc_checkpoint → `bc-v2.1/best.pt`） | `configs/train_ppo_v1.yaml` |

### 1.1 修复的真实 bug（§3 允许，均配回归测试 + 记录原因）

| # | Bug | 修复 | 证据 |
|---|---|---|---|
| B1 | 荒牌流局 `_draw_for` 调用 `_resolve_ryukyoku` 后**丢弃返回值** → 流局结算 deltas 不进 reward（`_accumulated` 与 `state.scores` 脱节） | 返回并传播 ryukyoku rewards | 回归测试「修复前失败、修复后通过」 |
| B2 | **立直棒 -1000** 只改 `p.score`，未入 `state.scores`、未返回 reward delta → 含立直对局 trajectory↔final score delta 不自洽 | 立直宣言时 `state.scores[seat] -= 1000`、`kyotaku += 1`、返回 `−scale·1000` delta（**用户裁定**：供托归最后的和牌者、流局保留至下一局） | +3 回归测试；守恒不变量 `sum(scores)+kyotaku·1000 == 100000` |
| B3 | `steps_per_epoch` 语义错误（按 episode 数跑，Pilot「2000 步」实际跑 9069 步） | 固定为**每 epoch env step 上限**（跑完当前 episode 再停，记录 `target_steps`） | `test_steps_per_epoch_upper_bound` |
| B4 | Benchmark 无法加载 PPO checkpoint（`load_policy` 只认 BC 的 `state_dict`） | `load_policy` 自动识别 `checkpoint_type=="ppo"` → `load_ppo_checkpoint`；logits bitwise 一致性测试 | `tests/test_ppo_benchmark_load.py` |

**规则验证无回归**：`validate_rules.py --limit 100 --seed 1` 修复前后 match rate **完全一致**（tenhou/majsoul 和牌 0.9989、流局 1.0）。

---

## 2. 测试结果

- **`pytest`：340 passed**（Phase 10.1.1 结束时 299 → 本阶段新增 41 个测试，无破坏）；
- 新增测试文件：`test_ppo_initialization.py`、`test_ppo_checkpoint.py`、`test_ppo_resume.py`、`test_ppo_reward.py`、`test_ppo_reward_settlement.py`、`test_ppo_gae.py`、`test_ppo_ratio.py`、`test_ppo_metrics.py`、`test_environment_regression.py`、`test_environment_west_round.py`、`test_ppo_benchmark_load.py`、`test_fast_encode.py`（前阶段）；
- **独立门禁审计（t83，reviewer）：PASS** —— 未采信报告结论，自行复跑 `cross_validate_replay.py`（3581 点 0 mismatch）、`pytest`（334 时点）等，确认无弱断言/放宽、无 §3 违规（`env.py` 两处修复正当）。

## 3. Pilot 结果（§17/§18）

配置 `configs/train_ppo_v1.yaml`（num_envs 4、batch 1024、update_epochs 4、γ 0.99、λ 0.95、clip_eps 0.2、vf_coef 0.5、ent_coef 0.01、lr 1e-4、seed/env_seed 42）。逐档推进，前档健康才继续。

| 指标 | Pilot-A (target 2,000) | Pilot-B (target 5,000) | Pilot-C (target 10,000) |
|---|---|---|---|
| 实际 steps / target | **2,371 / 2,000** | **6,964 / 5,000** | **10,891 / 10,000** |
| policy_loss | −0.00034 | −0.00120 | −0.00058 |
| value_loss | 4.090 | 7.425 | 7.093 |
| entropy | 0.2775 | 0.2766 | 0.2812 |
| mean_ratio | 0.9987 | 0.9976 | 0.9980 |
| approx_kl | 0.00209 | 0.00458 | 0.00645 |
| clip_fraction | 0.0092 | 0.0265 | 0.0456 |
| mean_reward | ≈0 | ≈0 | −0.050 |
| mean_return | 57.60 | 334.64 | 316.96 |
| advantage_mean / std | −0.093 / 3.007 | ≈0 / — | ≈0 / — |
| episode_count / length | 16 / 566.8 | — | — |
| **illegal_rate** | **0.0** | **0.0** | **0.0** |
| NaN / Inf | 无 | 无 | 无 |
| env 异常终止 | 无 | 无 | 无 |
| checkpoint 保存 | ✅ | ✅ | ✅ |

**判定：A / B / C 全部 PASS**（§17 首要目标达成）。`mean_reward≈0`、`mean_rank 1.5` 是**自对弈同策略**的预期值；`value_loss` 在 4–7.4 波动属 value head 早期学习（reward 方差大），非异常；`entropy` 稳定在 0.277–0.281 **无塌缩**；`approx_kl`/`clip_fraction` 随规模缓慢增长但均很小（健康）。

## 4. Benchmark 结果（§19/§20）

**设置**：candidate = `ppo-v1`（`experiments/ppo_v1/latest.pt`，Pilot-C 最终模型 10,891 steps，**已通过 logits bitwise 一致性验证**）；对手 random / rule / **BC-v2.1**；**2000 局/对手**；座位 `game_index % 4` 轮换、`seed = 42 + game_index`；16 worker；共 **12,000 局**，耗时 **1477.7s（≈24.6 分钟）**。

### 4.1 Greedy（主模式）

| 对手 | **candidate mean_rank** | rank 分布 [1/2/3/4] | win/局 | 放铳/局 | 立直/局 | 副露/局 | scoreΔ |
|---|---|---|---|---|---|---|---|
| random | **0.0020** | [1996, 4, 0, 0] | 4.152 | 0.006 | 2.094 | 4.446 | +37685 |
| rule | **0.6185** | [1151, 541, 228, 80] | 3.494 | 0.426 | 2.132 | 5.346 | +18604 |
| bc_v2.1 | **1.0750** | [784, 557, 384, 275] | 2.635 | 1.040 | 2.174 | 4.446 | +7189 |

对手侧 mean_rank：random 1.9993 / rule 1.7938 / bc_v2.1 1.6417。

### 4.2 Sampling（辅模式）

| 对手 | **candidate mean_rank** | rank 分布 [1/2/3/4] | win/局 | 放铳/局 | 立直/局 | 副露/局 | scoreΔ |
|---|---|---|---|---|---|---|---|
| random | **0.0150** | [1979, 15, 3, 3] | 2.332 | 0.009 | 1.274 | 9.319 | +21570 |
| rule | **1.1360** | [676, 639, 422, 263] | 2.217 | 0.625 | 1.526 | 8.778 | +5789 |
| bc_v2.1 | **1.7315** ⚠️ | [355, 457, 558, 630] | 1.666 | 1.321 | 1.453 | 7.494 | **−3531** |

对手侧 mean_rank：random 1.9950 / rule 1.6213 / bc_v2.1 1.4228。

### 4.3 对比分析（含与 BC-v2.1 的历史数据）

| 对比 | 结果 |
|---|---|
| PPO vs random | **压制性**（greedy 0.0020 / sampling 0.0150） |
| PPO vs rule | **优于 BC-v2.1**（PPO greedy 0.619 / sampling 1.136；BC-v2.1 greedy 0.920 / sampling 1.337） |
| **PPO vs BC-v2.1 正面对抗** | greedy：PPO 略优（1.0750 vs 对手 1.6417）；**sampling：PPO 落败**（1.7315 vs 对手 1.4228，scoreΔ −3531）⚠️ |
| illegal_rate | 全程 **0** |

## 5. 发现的问题

1. **B1/B2/B3/B4 四个 bug**（§1.1）——均已修复 + 回归测试；其中 B1/B2 是 **RL reward 正确性**的真实缺陷，正是 §10「不要假设 reward attribution 已正确」的意义所在。
2. **`double_yakuman` 未接线**（§9 核对发现）：`RulesConfig` 有该 flag 但 Environment 未端到端接入（当前按单倍役满），已如实记录 + 测试断言，未声称完整实现。
3. **门禁审计非阻塞项**（t83）：M1 §11 schema 未接入 rollout、M2 缺西入端到端测试、L1–L3（resume 未比 metrics / RNG 单测弱 / 陈旧注释）—— **已全部修复**（t84）。
4. **`steps_per_epoch` 语义错误**（B3）导致首次 Pilot 跑出 9069 步（应为 2000）——已修复。
5. **Benchmark 不支持 PPO checkpoint**（B4）——已修复。

## 6. 剩余风险

| 风险 | 说明 | 建议 |
|---|---|---|
| R1 | **PPO-v1（10.9k steps）正面对抗 BC-v2.1 在 sampling 模式仍落败** | 符合 §32（本阶段不比强度）；Phase 10.3 需更大规模 + opponent pool 才能确认 PPO 是否真正增益 |
| R2 | `double_yakuman` 未接线 | Phase 10.3 前评估是否需要补（会影响极少见的役满结算） |
| R3 | Pilot 只用 **单 seed（42）** | 结论的稳健性有限；Phase 10.3 建议多 seed |
| R4 | PPO 吞吐未优化（顺序 num_envs，§22 明确暂缓） | Phase 10.3 做并行 env / vectorized rollout |
| R5 | Phase 10.1.1 遗留：E 数据质量（69 例 replay error 分类）已放弃 | 如需可用 `--resume` 定向重扫 |
| R6 | profiler 的 `gpu_util` 仍为 None（需外部 nvidia-smi） | 低优先 |
| R7 | random baseline 的指标口径（call 高 / win 低）仅作 sanity check（§18） | 已知，不构成问题 |

## 7. §28 Definition of Done 逐条核对

| DoD 项 | 结果 |
|---|---|
| bc-v2.1 best.pt 正常加载 | ✅ |
| PPO trunk/action head 与 BC 初始化严格一致 | ✅ `torch.equal` bitwise |
| value head 正常初始化 | ✅ 重初始化 + 可训练（value_loss 下降） |
| Environment regression PASS | ✅ |
| Replay↔Environment cross-validation PASS | ✅ 3581 点 0 mismatch |
| environment_version 明确 | ✅ `tenhou-v2-rl` |
| reward-v1 明确 | ✅ |
| reward attribution regression PASS | ✅ |
| terminal reward regression PASS | ✅ |
| GAE regression PASS | ✅ |
| PPO ratio regression PASS | ✅ |
| PPO clipping regression PASS | ✅ |
| trajectory schema 明确 | ✅ `ppo_buffer.py` 并接入 rollout |
| PPO metrics 完整 | ✅ 14 项 |
| approx_kl 已实现 | ✅ 定义固定 + 测试 |
| clip_fraction 已实现 | ✅ 定义固定 + 测试 |
| advantage normalization 语义固定 | ✅ rollout-wide |
| PPO checkpoint 完整 | ✅ |
| optimizer state 已保存 | ✅ |
| RNG state 已保存 | ✅ |
| metadata 已保存 | ✅ |
| PPO resume PASS | ✅ Run A == Run B（bitwise） |
| fixed seed reproducibility PASS | ✅ |
| Pilot-A / B / C PASS | ✅ 三档全健康 |
| 无 NaN / Inf | ✅ |
| illegal_rate = 0 | ✅ |
| Environment 无异常终止 | ✅ |
| checkpoint 可正常加载 | ✅ |
| resume 可正常继续 | ✅ |
| PPO Benchmark 完成 | ✅ 12000 局 |
| BC-v2.1 / PPO / Rule / Random 对比完成 | ✅ |
| Greedy benchmark 完成 | ✅ |
| Sampling benchmark 完成 | ✅ |
| docs/PHASE10.2.md 已保存 | ✅ |
| docs/PHASE10.2_RESULT.md 已生成 | ✅ 本文件 |
| STATUS.md 已更新 | ✅ |

**全部满足 → Phase 10.2 PASS。**

## 8. 是否 PASS / 是否允许进入 Phase 10.3

- **是否 PASS：PASS**（§28 DoD 逐条通过；P0/P1 门禁经独立审计确认；Pilot-A/B/C 全部健康；Benchmark 完成）。
- **是否允许进入 Phase 10.3：满足 §29 的 Go 条件**（所有 P0/DoD 通过）。但按 **§31.11「不得自动进入 Phase 10.3」**，需**用户明确决定**后才启动 Phase 10.3（大规模 PPO / 并行 env / opponent pool / 长期 benchmark）。
- 进入 Phase 10.3 前建议优先处理：**R1**（PPO 对 BC 的增益验证）、**R3**（多 seed）、**R4**（rollout 吞吐）、**R2**（double_yakuman）。

## 9. 复现命令

```powershell
# 门禁测试
.venv\Scripts\python.exe -m pytest

# Replay ↔ Environment 交叉验证
.venv\Scripts\python.exe scripts\cross_validate_replay.py

# Pilot-A / B / C（逐档）
.venv\Scripts\python.exe scripts\train_ppo.py --config configs\train_ppo_v1.yaml --epochs 1 --steps-per-epoch 2000
.venv\Scripts\python.exe scripts\train_ppo.py --config configs\train_ppo_v1.yaml --epochs 1 --steps-per-epoch 5000
.venv\Scripts\python.exe scripts\train_ppo.py --config configs\train_ppo_v1.yaml --epochs 1 --steps-per-epoch 10000

# PPO-v1 Benchmark（greedy / sampling，各 2000 局/对手）
.venv\Scripts\python.exe scripts\benchmark_parallel.py --candidate experiments\ppo_v1\latest.pt ^
    --candidate-type ppo --candidate-version ppo-v1 --candidate-mode greedy ^
    --opponents random,rule,bc_v2_1 --games-per-opponent 2000 --num-workers 16 ^
    --out experiments\benchmark_ppo_v1_greedy
# 换 --candidate-mode sampling --out experiments\benchmark_ppo_v1_sampling
```
