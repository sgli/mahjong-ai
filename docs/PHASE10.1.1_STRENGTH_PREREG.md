# Phase 10.1.1 BC-v4 对局强度预注册

日期：2026-11-12。此预注册先于本轮正式强度对局运行；不得根据中间结果改种子、样本数、对局模式或门槛。

## 问题与估计量

评估 BC-v4 相对于冻结 BC-v3 的实际四人麻将对局表现。每个随机种子分别跑 candidate 与镜像控制局，主估计量为同种子、同候选席位下的顺位差：

`d_i = BC-v4候选局的候选顺位 - BC-v3镜像局的候选顺位`

顺位用 0 表示第一名、3 表示第四名；负值表示 BC-v4 较好。报告 20,000 个配对差的均值及 paired percentile bootstrap 95% CI。

## 冻结配置

| 项目 | 固定值 |
| --- | --- |
| BC-v4 候选 checkpoint | `experiments/bc_v4/checkpoints_20w/best.pt`；SHA-256 `814c6eba4e163d126cff7bc27194f11612263c745328abd1b1d9ae94b6716937` |
| BC-v3 镜像/对手 checkpoint | `experiments/bc_v3/checkpoints/best.pt`；SHA-256 `f67f03483930491bded216e06411c72ab6ede9bf81c18506a1a749fe2d1a6971` |
| 候选 arm | BC-v4 对 3 个 BC-v3 |
| 镜像 arm | BC-v3 对 3 个同 checkpoint 的 BC-v3 |
| 配对种子 | `600000` 至 `619999`（20,000 个）；每一 seed 两个 arm 完全相同 |
| 候选席位 | `game_index % 4`；每个 seat 5,000 局，两个 arm 相同 |
| 动作选择 | Greedy，candidate/mirror/opponents 均 temperature 1.0 |
| 并行度 | 16 workers，与历史 20,000 对跑批一致 |
| Bootstrap | 10,000 次 paired resample，NumPy PCG64 seed 10319，百分位区间，线性分位插值 |
| runner | `scripts/benchmark_paired_bc.py`，两 arm 逐一运行并核验 seed/seat/checkpoint；每局原始数据保留 |

新种子范围避开已留存的 BC-v3 20,000 对实验（500000–519999）。历史对局使用 Greedy/T=1、16 workers，报告总耗时 6,069 秒（约 101 分钟）；本次预计 **90–130 分钟**，以 40,000 局总量和 BC-v4 模型成本差异给出范围。按项目约束，不设时间上限、不实时监控；运行自然完成后由用户通知再读取。

## 有效性与通过标准

- 两个 arm 都必须恰好 20,000 局，seed 集合、候选席位和 game index 一一匹配；无未完成局、异常或非法动作。任一不满足则结果无效，不能用额外种子补样本。
- 镜像 sanity 通过条件：镜像候选平均顺位距离 1.5 不超过 0.02；非法动作 0；四席位各恰好 5,000 局。
- GO 需同时满足：配对均值 `<= -0.020`、配对 95% CI 上界 `< 0`、有效性全部通过、镜像 sanity 通过。
- 任一 GO 条件未满足即 NO-GO；不追加对局、不更改门槛、不启动 PPO。

## 产物与分析

输出目录：`experiments/bc_v4/strength_20k_seed600000_v1/`。保留 `run_metadata.json`、两 arm 原始结果、配对逐 seed rank 差 CSV 和最终 `result.json`。执行命令、runner/checkpoint SHA、实际每 arm 与总耗时写入产物；分析脚本运行在两 arm 完成之后。
