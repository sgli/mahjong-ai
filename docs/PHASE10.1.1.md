# Phase 10.1.1：BC-v4 基线训练与门槛评估

状态：Phase 10.1.1 完成，预注册强度门 GO。更新：2026-11-13。

## 当前项目状态

- **阶段与运行**：项目已重新开始 Phase 10；BC-v4 正式五轮训练于 2026-11-12 14:21 +08:00 完成，实际约 6.67 小时，20 workers、prefetch 4，checkpoint 在 `experiments/bc_v4/checkpoints_20w`。第一轮完整 test 于 19:22 启动，因用户要求先停并优化评估代码而在未产出结果时终止，产物不得使用。优化后的全量 test 于 19:39–19:45 完成，PID 18052 已退出。
- **当前基线**：BC-v3 best，epoch 3，checkpoint `experiments/bc_v3/checkpoints/best.pt`，SHA-256 `f67f03483930491bded216e06411c72ab6ede9bf81c18506a1a749fe2d1a6971`。它是项目指定的强度锚点；历史比较未证明 BC-v3 强于 BC-v2.1。基线完整证据见 [BC_V3_DATA.md](BC_V3_DATA.md)。
- **基线离线结果**：`decision-v3` 完整 test 共 134,574,358 行；BC-v3 top-1 69.6998%、illegal rate 0。
- **训练数据**：保留 `data/processed/decision-v3`，共 2,072,841 局、1,345,632,173 行；manifest SHA-256 `f3569b7ba7450e5cc4509eaaee975c609f5b3a2572a063e43f0e1beb8d4fac3b`。BC-v4 直接复用该数据，不重建 Parquet。
- **BC-v4 配置**：`configs/train_bc_v4.yaml`，输入 `feature-v3/action-v2`，模型随机初始化；有界输入正确性门已通过。此次运行通过 CLI 覆盖 `num_workers=20`，prefetch 仍为4。训练 top-k/非法率每1000步抽样，loss 逐 batch 精确累计，validation 逻辑未变。
- **清理状态**：旧 PHASE10.x 文档、非基线模型配置/跑批结果、旧 processed 数据和阶段专用诊断材料已按项目重启要求清理。`README.md` 内容过时，不作为当前状态来源。
- **性能诊断**：同 16 workers/prefetch 4 下，BC-v3 14,605 维输入训练 profile 吞吐约 141.8k rows/s、`data_time` 约 56%；BC-v4 781 维配置 150 秒诊断到 step 14,000，抽样指标优化前约到 step 13,000。采样运行的 CPU 仍约 32.5%、GPU 约 62.8%；CPU 未饱和但处理步数增加。静态 batch-size sweep 中 2048 快于 4096/8192，故保留 2048。
- **评估性能诊断**：原评估每批同步 6 个 CUDA 标量，并逐行构造 decision/action/red-label 统计。已改为默认关闭非门槛分组统计、GPU 累积指标后最后同步；短窗 5M test pilot 非法率为 0，活跃计算段 CPU 达到约 85%（较旧评估 2 分钟均值约 35.8%）。这只是性能 pilot，不作全量模型结论。代码在 [evaluate_bc.py](../scripts/evaluate_bc.py)，采样见 `experiments/bc_v4/eval_cpu_pilot_5m_monitor.telemetry.json`。
- **BC-v4 全量 test**：134,574,358 行；loss 0.809803、top-1 70.7013%、top-3 93.4707%、top-5 97.7909%、illegal rate 0。相对 BC-v3 同一 split：top-1 +1.0016 pp、top-3 +0.6481 pp、top-5 +0.3194 pp、loss -0.033137，非法率同为 0。checkpoint SHA 与 decision-v3 manifest SHA 均与冻结身份匹配。完整报告见 [PHASE10.1.1_RESULT.md](PHASE10.1.1_RESULT.md) 和 `experiments/bc_v4/eval_test_full_20w_optimized.json`。
- **强度跑批结果**：按冻结预注册完成候选和镜像各20,000局（40,000局），seed 600000–619999、同 seat、Greedy/T=1、16 workers。候选均位1.4449，镜像均位1.50025；配对差 −0.05535，bootstrap 95% CI [−0.07495125, −0.03645]；镜像 sanity、全局有效局和零非法动作均通过，GO。详细数据见 [PHASE10.1.1_RESULT.md](PHASE10.1.1_RESULT.md) 与 `experiments/bc_v4/strength_20k_seed600000_v1/result.json`。
- **运行状态**：跑批已结束，PID 10056 不再运行。总耗时 24,073.529 秒（约 6 小时 41 分）；candidate arm 10,912.281 秒，mirror arm 13,158.837 秒。预估 90–130 分钟明显偏低，记录在项目记忆 lessons 中用于后续估时校准。

## 目标

从干净的模型实验目录重新开始 Phase 10。保留 BC-v3 作为冻结强度锚点，使用 `feature-v3/action-v2` 在 `decision-v3` 上从零训练 BC-v4。旧模型的训练和跑批产物不参与新实验。BC-v3 历史数据见 [BC_V3_DATA.md](BC_V3_DATA.md)。

## 固定设计

- 模型：`bc-v4`，shared tile embedding + GRU 编码三家对手的有序牌河；397 维 dense 输入，三段各最多 128 个 tile token；动作仍为 `action-v2` 的 325 类。
- 数据：复用 `data/processed/decision-v3` 的既有冻结 split 与标签，不重建 Parquet；先运行 feature-v3 运行时验证。
- 初始化：随机初始化，不从 feature-v2 的 BC-v3 checkpoint 加载权重。
- 训练：5 epochs，按 `configs/train_bc_v4.yaml` 固定 seed 与优化器/批处理设置；不在跑后调参以追 test。
- 训练诊断：loss 精确累计；训练 top-k/非法率按 `metric_interval_steps=1000` 抽样并随 checkpoint 标注样本量；validation 指标仍按原配置计算。
- 强度锚点：`experiments/bc_v3/checkpoints/best.pt`。离线准确率仅用于训练诊断，模型强度以冻结的配对对局门为准。
- 无效样本、越界牌河、gold 不在 legal mask、非法动作率非零均阻止进入正式对局阶段。

## 阶段门

1. **输入正确性门**：在固定 train/validation/test 分片上检查 feature-v3 特征/序列形状、牌河顺序与长度、padding、动作标签和 legal mask；比较串行/快速编码一致性。失败则修复后重跑此有界检查，不启动全量训练。
2. **BC-v4 训练**：通过输入门后再启动完整 5 epochs。训练前根据本机该编码版本的实测 pilot 估时。估计超过 20 分钟时，不设时长上限、不实时监控；待用户告知跑完后继续。
3. **离线评估门**：与 BC-v3 在同一冻结 test split 上报告 top-1/top-3/loss/illegal rate，并检查模型、配置、数据 manifest 和特征版本身份。此门不单独宣称强度提升。
4. **强度确认门**：使用冻结的随机种子、席位轮换、Greedy/T=1 和同种子 BC-v3 镜像控制，至少 20,000 对有效局；预注册主效应为 BC-v4 平均顺位减 BC-v3 镜像平均顺位。GO 条件为点估计 `<= -0.020` 且配对 95% CI 上界 `< 0`，同时镜像 sanity 通过、非法动作 0。否则停止，不追加样本或改门槛。
5. 只有强度门通过，才另行决定是否设计 PPO；Phase 10 本身不启动 PPO。

## 执行状态

- [x] 保留 BC-v3 checkpoint、配置、训练/评估和强度结果证据。
- [x] 确认 decision-v3 原始有序牌河可派生 feature-v3，无需重建数据。
- [x] 完成有界 feature-v3/模型正确性验证并记录结果。
- [x] 完成 pilot/全配置短窗吞吐估算与 CPU/GPU 监控；确认 2048 batch 在静态 sweep 中最快。
- [x] 对训练期指标同步/频率做 A/B，并将 BC-v4 诊断指标调整为间隔抽样；完成 64-shard epoch 汇总 pilot。
- [x] 完成 BC-v4 正式 5 epochs（20 workers / prefetch 4；训练记录见 `experiments/bc_v4/checkpoints_20w/metrics.json`）。
- [x] 完成同 split 全量 test 离线评估（134,574,358 行；结果见 `experiments/bc_v4/eval_test_full_20w_optimized.json`）。
- [x] 完成预注册的 20,000 对 BC-v4 vs BC-v3 镜像强度评估；20k/arm、seed 600000–619999，GO（结果见 `PHASE10.1.1_RESULT.md`）。

## 下一步

Phase 10.1.1 强度门通过，BC-v4 可作为 Phase 10.2 PPO 的初始化策略。PPO 尚未启动；开始前需单独确定并预注册训练预算、评估种子、对手池和强度门槛。

## 项目约束

- 新特征版本不复用旧 checkpoint 权重；不得用 test 结果反复调参。
