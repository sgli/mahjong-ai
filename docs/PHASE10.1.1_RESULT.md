# Phase 10.1.1 结果记录：BC-v4 基线

状态：Phase 10.1.1 完成；正式训练、全量离线 test 与预注册强度门均完成，强度门判定 GO。更新日期：2026-11-13。

## 正式训练结果

- 配置：`configs/train_bc_v4.yaml`；decision-v3 全量训练，feature-v3/action-v2，随机初始化，5 epochs，seed 42，batch 2048，20 workers，prefetch 4，metric interval 1000。
- 实际时长约 6.67 小时。五轮每轮约 79.2–79.5 分钟，训练 profile 吞吐约 225.6k rows/s。
- best checkpoint 依据 validation loss 选择为 epoch 2：`experiments/bc_v4/checkpoints_20w/best.pt`，SHA-256 `814c6eba4e163d126cff7bc27194f11612263c745328abd1b1d9ae94b6716937`。
- 数据 manifest SHA-256：`f3569b7ba7450e5cc4509eaaee975c609f5b3a2572a063e43f0e1beb8d4fac3b`；配置和逐 epoch 记录见 `experiments/bc_v4/checkpoints_20w/config.yaml` 与 `metrics.json`。

| Epoch | Train loss（全量累计） | Train top-1（每1000步抽样） | Validation loss（抽样5M） | Validation top-1 | Top-3 | Top-5 | Illegal rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.848429 | 69.3404% | 0.827283 | 70.2045% | 93.2370% | 97.6767% | 0 |
| 2 | 0.816726 | 70.4324% | **0.817612** | **70.4237%** | 93.3795% | 97.7485% | 0 |
| 3 | 0.814562 | 70.5008% | 0.817857 | 70.2917% | 93.4491% | **97.7794%** | 0 |
| 4 | 0.815607 | 70.4278% | 0.820588 | 70.2471% | 93.3789% | 97.7508% | 0 |
| 5 | 0.815651 | 70.4102% | 0.819746 | 70.3141% | 93.4354% | 97.7686% | 0 |

训练分类指标实际分别来自约 1.05–1.09M 个样本（约全训练集的 0.10%），是按间隔采样的诊断值；train loss 是全量逐 batch 累计，validation 是每轮固定抽取 5M 行。validation 指标用于 checkpoint 选择，不是强度结论。

## 评估器 CPU 诊断与修复

- 首次全量 test 评估启动后，用户观察 CPU 利用率偏低并要求停止；两分钟测量系统 CPU 均值 35.78%，Python 进程组均值 31.73%。该进程树已停止，未产出完整结果，不能使用。
- 检查发现评估循环每个 batch 对 6 个 CUDA 指标调用 `.item()`，另有逐行 decision/action/red-label 统计与预测回拷。本阶段正式门槛不需要这些分组指标。
- `scripts/evaluate_bc.py` 现默认不计算分组统计，保留 `--include-groups` 可选项；六项指标在 GPU 累积并在评估结束时统一转到 CPU。`py_compile` 通过。
- 优化后 5M test 性能 pilot 用时约 44 秒，非法率 0；实际活跃计算段最后两次采样系统 CPU 均值约 85.3%，Python 进程组约 79.6%。短窗 pilot 按数据顺序只用于性能诊断，不参与 BC-v3 强度或全量准确率结论。遥测见 `experiments/bc_v4/eval_cpu_pilot_5m_monitor.telemetry.json`。

## 全量 test 与阶段门

- 优化后的正式全量评估于 19:39:12 启动、19:45:42 写出结果，约 6.5 分钟；原估时 18–22 分钟偏保守。完整结果 `experiments/bc_v4/eval_test_full_20w_optimized.json`，启动信息 `experiments/bc_v4/eval_test_full_20w_optimized.started.json`。
- 同 split 指标对照：

| 指标 | BC-v3 | BC-v4 | 差值（BC-v4 − BC-v3） |
| --- | ---: | ---: | ---: |
| Test 行数 | 134,574,358 | 134,574,358 | 相同 |
| Loss | 0.842940 | 0.809803 | -0.033137 |
| Top-1 | 69.6998% | 70.7013% | +1.0016 pp |
| Top-3 | 92.8226% | 93.4707% | +0.6481 pp |
| Top-5 | 97.4715% | 97.7909% | +0.3194 pp |
| Illegal rate | 0 | 0 | 相同 |

- 身份核验通过：checkpoint SHA-256 `814c6eba4e163d126cff7bc27194f11612263c745328abd1b1d9ae94b6716937`；test 使用的 decision-v3 manifest SHA-256 `f3569b7ba7450e5cc4509eaaee975c609f5b3a2572a063e43f0e1beb8d4fac3b`；feature-v3/action-v2。
- **预注册强度门通过（GO）。**按冻结协议完成20,000个配对种子（40,000局），seed 600000–619999；candidate BC-v4 对 3×BC-v3，mirror BC-v3 对 3×BC-v3；Greedy/T=1、同 seat、16 workers。候选均位1.4449，镜像均位1.50025，配对顺位差（candidate − mirror）为 **−0.05535**，bootstrap 95% CI **[−0.07495125, −0.03645]**。差值满足 `<= −0.020` 且区间上界 `< 0`；镜像均位落入预注册 1.5±0.02；40,000局全部有效，两侧非法动作均为0。`valid_run=true`、`mirror_sanity_passed=true`、`decision=GO`。结果文件：`experiments/bc_v4/strength_20k_seed600000_v1/result.json`；配对逐种子数据：`paired_rank_differences.csv`。
- **耗时核验：**candidate arm 10,912.281秒，mirror arm 13,158.837秒，总计 24,073.529秒（约6小时41分13秒）。启动时根据历史相近跑批估算90–130分钟，显著低估；本次实际耗时已追加到 `docs/memory/50-lessons.md`，后续同 runner 估时应采用此实测校准并说明不同硬件/代码版本不确定性。

## 阶段结论与后续

Phase 10.1.1 的输入正确性、BC-v4 五轮训练、全量离线 test 和预注册实际对局强度门均已完成。BC-v4 通过预注册强度门，可作为 Phase 10.2 PPO 初始化策略。该结论适用于本次固定的 BC-v3 对手池、Greedy/T=1 和评估协议；PPO 尚未启动，必须另行预注册训练预算、评估种子、对手池与强度门槛。
