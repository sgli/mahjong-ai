# Benchmark Report — BC Baseline vs Random

- 对手版本：
  - BC Policy：`bc@bc-v1`（checkpoint `experiments/exp_0001/checkpoint/model.pt`）
  - Random：`random@random-v1`
- game count：各 30 局（candidate seat 0，对手 seat 1/2/3 均为 random）
- seed：42
- 规则版本：`tenhou-v1`
- git commit：`ed9b75b`
- illegal_rate：全程 **0.0**（smoke/rules 均通过）

## 结果表

| 指标 | BC Policy | Random（baseline） |
|---|---|---|
| mean_rank（越低越好） | **0.167** | 0.700 |
| rank_distribution（1/2/3/4 位） | [27, 1, 2, 0] | [19, 4, 4, 3] |
| win_rate（和牌率） | **76.7%** | 0.0% |
| deal_in_rate（放铳率） | **0.0%** | 6.7% |
| riichi_rate（立直率） | 90.0% | 3.3% |
| call_rate（副露率） | 9.47 | 20.8 |
| avg_win_points（平均和牌点数） | 11517 | 0（无和牌） |
| mean_score_change（平均得点变化） | +10363 | -357 |

## 结论

- BC Policy 显著优于 Random：mean_rank 0.167 vs 0.700，和牌率 76.7% vs 0%，放铳率 0%。
- illegal_rate = 0（合法动作 mask 全程生效，无非法动作）。
- 候选通过 smoke 与 rules 测试，promoted = True（BC rank 低于 random baseline）。

## 说明

- 本 benchmark 的 BC 对手固定 seat 0，其余三家为 Random；这是「人类策略 vs 随机」的第一版基线对比，用于验证训练与推理链路可运行、策略确实学到了优于随机的决策。
