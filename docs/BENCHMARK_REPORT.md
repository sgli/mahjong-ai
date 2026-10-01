# Benchmark Report — BC Baseline vs Random

- 对手版本：
  - BC Policy：`bc@bc-v1`（checkpoint `experiments/exp_0001/checkpoint`）
  - Random：`random@random-v1`
- game count：各 32 局（candidate 在 4 个座位间轮换，对手为 random）
- seed：42
- 规则版本：`tenhou-v1`
- git commit：`42cb626`
- illegal_rate：全程 **0.0**（smoke/rules 均通过）

> 说明：`win/deal_in/riichi/call` 指标单位是「次/局」，不是百分比。

## 结果表

| 指标 | BC Policy | Random（baseline） |
|---|---|---|
| mean_rank（越低越好） | **0.406** | 1.500 |
| rank_distribution（1/2/3/4 位） | [26, 1, 3, 2] | [9, 6, 9, 8] |
| win_per_game（和牌 次/局） | **0.781** | 0.031 |
| deal_in_per_game（放铳 次/局） | **0.0** | 0.0 |
| riichi_per_game（立直 次/局） | 0.750 | 0.0 |
| call_per_game（副露 次/局） | 9.531 | 20.063 |
| avg_win_points（平均和牌点数） | 9884 | 7700 |
| mean_score_change（平均得点变化） | +9097 | +84 |

## 结论

- BC Policy 显著优于 Random：mean_rank 0.406 vs 1.500，和牌 0.781 次/局 vs 0.031 次/局，放铳 0 次/局。
- illegal_rate = 0（合法动作 mask 全程生效，无非法动作）。
- 候选通过 smoke 与 rules 测试，promoted = True。
- 候选座位每局轮换（seat 0→1→2→3→0…），消除固定座位偏差。
