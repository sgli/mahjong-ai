# Phase 10.3 PPO 回归诊断与修复指导规格

> 本文用于 AI Coding Agent 直接执行。当前禁止继续无条件扩大 PPO 到 500k/1M+，先定位 Phase 10.3 的系统性策略退化。

## 1. 当前结论

Phase 10.3 的 RL 基础设施通过了 correctness / reproducibility / resume / benchmark 基本验证，但 **Strategy Improvement = FAIL / INCONCLUSIVE**。

当前权威多 seed 结果：

| Candidate | Greedy vs Rule | Sampling vs Rule | Greedy vs BC | Sampling vs BC |
|---|---:|---:|---:|---:|
| BC-v2.1 | **0.5770** | **0.9527** | — | — |
| PPO-20k | 0.6430 | 1.2127 | 1.1053 | 1.7323 |
| PPO-50k | 0.7857 | 1.4350 | 1.2590 | 1.9533 |
| PPO-100k | 0.8533 | 1.5353 | 1.3803 | 2.0637 |

四条曲线均随训练恶化，且 3 seeds 的 std 较小，因此不是简单随机噪声。

训练健康指标却没有出现典型 PPO 爆炸：

- `illegal_rate = 0`
- 无 NaN / Inf
- `value_loss`：6.820 → 4.568 → 3.231
- entropy：0.2780 → 0.3122 → 0.3418
- approx_kl：0.00622 → 0.00518 → 0.00465
- clip_fraction：约 0.045
- mean_ratio ≈ 1

因此当前更像：

```text
BC 强策略
  ↓
PPO 更新
  ↓
policy drift / BC forgetting
  ↓
entropy 上升
  ↓
决策质量下降
  ↓
继续训练进一步退化
```

**禁止直接继续 500k / 1M+。**

---

## 2. 第一优先级：量化 BC Forgetting

新增固定的 `BC Retention Evaluation`。

使用 `decision-v2` 的固定 game-level evaluation subset，建议 10万～100万条决策，不能每次随机变化。

比较：

```text
BC-v2.1
PPO-20k
PPO-50k
PPO-100k
```

至少计算：

```text
argmax action agreement
P(BC action | PPO)
KL(BC || PPO)
KL(PPO || BC)
BC entropy
PPO entropy
```

并输出：

```text
checkpoint
agreement
KL(BC||PPO)
KL(PPO||BC)
BC entropy
PPO entropy
```

如果：

```text
agreement ↓
KL ↑
entropy ↑
```

则确认存在明显 policy drift / forgetting。

---

## 3. 第二优先级：PPO Update 机械审计

新增：

```text
tests/test_ppo_update_invariants.py
tests/test_ppo_ratio_diagnostics.py
tests/test_ppo_update_epochs.py
```

必须验证：

### Zero-advantage invariant

固定 batch：

```text
advantages = 0
```

一次 update 后 policy logits 应在浮点误差内保持不变。

### Ratio

验证：

```text
ratio = exp(new_log_prob - old_log_prob)
```

覆盖：

```text
ratio < 1-eps
ratio ∈ [1-eps,1+eps]
ratio > 1+eps
```

同时测试 positive / negative advantage 的 clipping。

### Update 前后

固定 batch，记录：

```text
old logits
old log_prob
old entropy
```

update 后记录：

```text
new logits
new log_prob
new entropy
ratio
KL
```

确认参数更新方向和 objective 一致。

---

## 4. 第三优先级：GAE / Return / Reward 审计

`compute_gae()` 当前形式：

```python
delta = reward + gamma * next_value * nonterminal - value
gae = delta + gamma * lam * nonterminal * gae
return = advantage + value
```

必须确认输入确实是每个 seat 的连续 trajectory。

新增统计：

```text
reward mean/std/min/max
value mean/std/min/max
advantage mean/std/min/max
return mean/std/min/max
```

至少按 epoch 记录，最好能按 seat / opponent type 聚合。

### Reward conservation

必须测试：

```text
sum(trajectory rewards)
=
environment score deltas
+
placement bonuses
```

并验证：

- placement bonus 只出现一次；
- pending reward 正确 flush；
- terminal transition 正确；
- done 后没有后续 reward；
- 不同 seat 不串 reward。

---

## 5. 第四优先级：Entropy / Exploration 诊断

当前 entropy：

```text
20k  0.2780
50k  0.3122
100k 0.3418
```

与强度同步恶化。

必须增加：

```text
legal action count
entropy / legal_action_count
BC entropy
PPO entropy
BC action probability
PPO action probability
```

目标是判断：

> PPO 是合理探索，还是把 BC 原本明确的决策逐渐随机化。

**不要先把 `ent_coef=0.01` 改成 0。** 先完成诊断，避免同时改变多个变量。

---

## 6. Value Function 审计

`value_loss` 下降不代表 policy 变强。

必须增加：

```text
MAE
RMSE
correlation
explained_variance
```

比较：

```text
value prediction
vs
actual return
```

如果 critic 越来越好而 actor 越来越差，则重点调查 actor objective / reward alignment。

---

## 7. PPO 多 epoch 漂移诊断

当前：

```yaml
update_epochs: 4
clip_eps: 0.2
```

不能只看 epoch 汇总的 KL。

必须分别记录：

```text
update_epoch_1_kl
update_epoch_2_kl
update_epoch_3_kl
update_epoch_4_kl

ratio_mean/std/min/max
clip_fraction
clip_positive_fraction
clip_negative_fraction
```

判断一次 rollout 的多轮更新是否持续推动 policy。

---

## 8. 优化 Rollout 必须接入正式训练

报告明确指出已经实现的 batch encode + forward 优化路径约 3588 steps/s，但当前 `train_ppo()` 仍使用逐 step reference 路径，端到端约 510 steps/s。

当前路径类似：

```python
features = encoder.encode(obs)
mask = legal_mask(legal)
model(features.unsqueeze(0), mask.unsqueeze(0))
```

要求：

```text
reference rollout
    ↓
fixed-seed parity
    ↓
optimized rollout
    ↓
接入 train_ppo
```

必须比较：

```text
action_id
old_log_prob
value
reward
done
advantage
return
```

在明确浮点误差内一致。

性能优化不能改变语义。

---

## 9. 复核历史 Benchmark 矛盾

报告指出旧结果：

```text
BC-v2.1 vs Rule
greedy 0.9198
sampling 1.3374
```

而当前统一 runner 的多 seed baseline：

```text
greedy 0.5770
sampling 0.9527
```

必须用当前 runner 重新跑：

```text
BC-v2.1 vs Rule
1000 games
5000 games
```

记录：

```text
git commit
runner version
config
seed
game count
```

最终确定权威基线。

---

## 10. 第一轮修复实验

在完成 P0 诊断后，只做单变量实验：

### A
原 PPO，20k。

### B
`lr = 5e-5`，20k。

### C
`lr = 2e-5`，20k。

### D
PPO + BC anchor，20k。

BC anchor 作为独立实验，例如：

```text
L = PPO_loss + beta * KL(pi || pi_BC)
```

初始只测试：

```text
beta = 0.001 / 0.003 / 0.01
```

不得大规模 sweep。

每个实验先用 seed 42 筛选，再扩展到 42/43/44。

---

## 11. Opponent Curriculum 暂时不是第一修复项

当前 pool：

```yaml
random-v1: 0.10
rule-v1: 0.20
bc-v2.1: 0.30
ppo-v1: 0.20
historical: 0.20
```

先记录每种 opponent 对应的：

```text
episode reward
return
win/rank
```

只有在 update / forgetting 问题排除后，才进行 curriculum 实验。

---

## 12. Reward 暂时禁止修改

不要因为 PPO 输给 BC 就立即修改 Reward-v1。

先：

```text
update audit
→ forgetting
→ entropy
→ value
→ opponent attribution
→ 再做 reward experiment
```

否则无法归因。

---

## 13. 推荐执行顺序

```text
1. 阅读 PHASE10.3_BENCHMARK_REPORT.md
2. 阅读 ppo.py / policy.py / benchmark.py
3. BC retention evaluator
4. BC / PPO-20k / 50k / 100k retention
5. PPO update mechanical audit
6. zero-advantage test
7. ratio / clipping tests
8. GAE / return / reward distribution
9. reward conservation regression
10. update-epoch KL
11. entropy / exploration diagnostics
12. value quality diagnostics
13. 确认第一根因
14. 单变量 20k 修复实验
15. 3-seed 验证
16. 再决定是否恢复 50k / 100k
17. 最后接入 optimized rollout
18. 所有 gate 通过后才能恢复 500k / 1M+
```

---

## 14. 禁止事项

```text
❌ 继续直接 500k / 1M
❌ 大规模 hyperparameter sweep
❌ 同时修改 reward + lr + entropy
❌ 修改 Mahjong Environment 规则
❌ 修改 feature-v1
❌ 修改 action-v1
❌ 修改 BC-v2.1
❌ 因 benchmark 变差修改 benchmark
❌ 删除 PPO-20k / 50k / 100k
❌ 用 training reward 上升宣布 PPO 变强
```

---

## 15. 必须新增测试

```text
tests/test_ppo_update_invariants.py
tests/test_ppo_bc_retention.py
tests/test_ppo_advantage_distribution.py
tests/test_ppo_reward_conservation.py
tests/test_ppo_update_epochs.py
tests/test_ppo_ratio_diagnostics.py
tests/test_ppo_rollout_parity.py
```

覆盖：

```text
[ ] zero-advantage invariant
[ ] ratio exactness
[ ] positive/negative clipping
[ ] update-epoch KL
[ ] ratio statistics
[ ] reward conservation
[ ] terminal reward placement
[ ] BC action agreement
[ ] BC/PPO KL
[ ] reference/optimized rollout parity
```

---

## 16. Go / No-Go

### GO

只有某方案满足：

```text
BC retention 明显高于原 PPO
+
entropy 不再异常持续上升
+
sampling vs BC 明显改善
+
greedy vs Rule 不恶化
+
correctness 无 regression
```

才允许进入：

```text
50k → 100k
```

### NO-GO

如果：

```text
BC agreement 持续下降
+
KL 持续上升
+
entropy 持续上升
+
strength 持续下降
```

则：

```text
停止扩大 PPO
```

转入：

```text
Representation / Reward / Curriculum
```

独立实验。

---

## 17. Phase 10.3 修复完成标准

```text
[ ] BC retention 已量化
[ ] PPO update mechanical audit PASS
[ ] zero-advantage invariant PASS
[ ] ratio/clipping regression PASS
[ ] reward conservation PASS
[ ] GAE/return distribution 已审计
[ ] update-epoch KL 已记录
[ ] entropy behavior 已解释
[ ] value quality 已评估
[ ] benchmark historical discrepancy 已解决
[ ] optimized rollout/reference parity PASS
[ ] optimized rollout 已接入训练
[ ] 至少一个修复方案在 20k 有效
[ ] 3 seed 验证有效
[ ] sampling vs BC 不再系统性恶化
[ ] greedy vs Rule 不持续恶化
```

然后才能恢复：

```text
20k
↓
50k
↓
100k
↓
500k
↓
1M+
```

---

## 18. 最终目标

本阶段不是“让 PPO 尽快跑到 1M steps”。

真正目标是回答：

> **为什么 PPO 从 BC-v2.1 出发后，会在数值稳定的情况下持续失去策略强度？**

最终必须把根因归类为至少一种：

```text
A. PPO update implementation
B. Advantage / return
C. Value estimation
D. BC catastrophic forgetting
E. Entropy / exploration
F. Reward alignment
G. Opponent curriculum
H. Representation limitation
```

当前明确结论：

```text
Phase 10.3 Infrastructure : PASS
Phase 10.3 Strength       : FAIL / INCONCLUSIVE
继续扩大训练              : NO
进入策略诊断              : YES
```
