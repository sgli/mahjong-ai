# Benchmark Report — bc-v2 vs random / bc-v1（Phase 10.1 Task4）

- candidate：bc-v2（experiments/bc_v2/checkpoints，decision-v2，1 epoch，MLP [256,256]）
- 对手：random（random-v1）、bc-v1（experiments/exp_0001/checkpoint）
- 局数：合计 **10000 局**（vs random 5000 + vs bc-v1 5000）
- seed：42（每局 seed = 42 + game_index，与并行度无关）
- 座位：候选每局轮换（seat = game_index % 4）
- 并行：16 workers（ProcessPoolExecutor），每 worker 独立跑对局
- 指标口径：`*_per_game` = 次/局（不是百分比）

## checkpoint 校验
- bc-v2 可加载；dataset_version=decision-v2、model_version=bc-v2、feature_dim=234、action=270。
- metrics.json 最终 validation：loss 0.8965 / acc 0.6792 / top3 0.9170 / illegal_rate 0.0。

## 结果表

### bc-v2 vs random（5000 局）
| 指标 | bc-v2 | random（baseline） |
|---|---|---|
| mean_rank（越低越好） | **0.012** | 1.489 |
| rank_distribution（1/2/3/4 位） | [4959, 28, 7, 6] | [1257, 1270, 1245, 1228] |
| win_per_game（次/局） | **2.623** | 0.020 |
| deal_in_per_game（次/局） | 0.011 | 0.017 |
| riichi_per_game（次/局） | 1.638 | 0.007 |
| call_per_game（次/局） | 8.770 | 21.160 |
| avg_win_points | 8535 | 4664 |
| mean_score_change | +26606 | -5 |

### bc-v2 vs bc-v1（5000 局）
| 指标 | bc-v2 |
|---|---|
| mean_rank（越低越好） | **0.455** |
| rank_distribution（1/2/3/4 位） | [3487, 960, 343, 210] |
| win_per_game（次/局） | **2.373** |
| deal_in_per_game（次/局） | 0.297 |
| riichi_per_game（次/局） | 1.809 |
| call_per_game（次/局） | 8.142 |
| avg_win_points | 8754 |
| mean_score_change | +19363 |

- illegal_rate：全程 **0.0**（smoke/rules/benchmark 均通过）。

## 正确性（并行 == 串行）
- 200 局 vs random：串行（num_workers=1）与并行（num_workers=4）指标完全一致（mean_rank 0.01、rank_dist [198,2,0,0]、win 2.7、illegal 0）。
- 5000 局 vs random 并行结果与 t68 串行结果一致（[4959,28,7,6]、win 2.6228），证明 seed 按 game_index 派生、与并行度无关。

## 内存（硬性检查，16 workers）
- pagefile CurrentUsage **0 MB**、PeakUsage **0 MB**（无增长）。
- 空闲物理内存 ~37 GB，全程稳定。
- 20 个 python 进程 WorkingSet 合计 ~7.6 GB，稳定不增长。
- 每 worker 内存上界 ≈ 4 个 MLP 模型（bc-v2 + 3×bc-v1，每个 ~2MB）+ MahjongEnv + torch/pyarrow import ≈ ~300-500MB；16 worker ≈ ~7-8GB（实测 7.6GB，无 OOM）。

## 耗时与加速比
| 规模 | 方式 | 耗时 | 吞吐 |
|---|---|---|---|
| 200 局 | 串行 | ~100s | ~2 局/s |
| 5000 局 vs random | 并行 16 worker | 378s | 13.2 局/s |
| 5000 局 vs bc-v1 | 并行 16 worker | 1074s | 4.7 局/s |
| 10000 局合计 | 并行 16 worker | **1452s（24.2 min）** | ~6.9 局/s |

- 对比串行 ~100-140min，16 worker 并行 ~24min，约 **5-6× 加速**（bc-v1 对手因 4 个 MLP 每步推理，比 random 慢 ~2.8×）。

## 结论
- bc-v2 对 random：压倒性优势（1 位率 99.2%，mean_rank 0.012）。
- bc-v2 对 bc-v1：明显优势（1 位率 69.7%，mean_rank 0.455），说明 decision-v2 大规模训练带来的提升。
- illegal_rate = 0，无非法动作。
