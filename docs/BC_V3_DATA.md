# BC-v3 基线数据与证据

更新日期：2026-11-12。本文是 BC-v3 配置、训练、评估与强度比较的集中记录。它仅记录可核验事实；旧 PHASE10.x 文件清理后，以本文件作为 BC-v3 历史来源。

## 基线定义

- 用途：作为重新开始的 Phase 10 中 BC-v4 的强度比较锚点；这是项目决策，不代表已证明 BC-v3 强于 BC-v2.1。
- Checkpoint：`experiments/bc_v3/checkpoints/best.pt`，SHA-256 `f67f03483930491bded216e06411c72ab6ede9bf81c18506a1a749fe2d1a6971`，对应 epoch 3。
- 模型版本：`bc-v3`；数据 `decision-v3`；输入 `feature-v2`（14,605 维）；动作 `action-v2`（325 个动作）；环境 `tenhou-v3-ron-context`。
- 训练配置：`configs/train_bc_v3.yaml`；5 epochs、seed 42、batch 2048、lr 0.001、16 workers、prefetch 4、AMP、hidden sizes `[256,256]`。
- 数据身份：manifest SHA-256 `f3569b7ba7450e5cc4509eaaee975c609f5b3a2572a063e43f0e1beb8d4fac3b`；全量数据 2,072,841 局、1,345,632,173 行；test 134,574,358 行。全量审计与构建预检的归档副本在 `experiments/bc_v3/reference/`。

## 训练和离线评估

- 5 轮训练总耗时 44,137.49 秒；每轮训练行数 1,076,481,812；epoch 3 的抽样验证 loss 最低（0.854001），故保留 epoch 3 为 best。
- 同一完整修正 test 上 masked top-1 69.6998%、top-3 92.8226%、loss 0.842940、illegal rate 0，覆盖 134,574,358 行。
- 相对当时 BC-v2.1 的同 test top-1 69.7666%，差 −0.0668 个百分点；BC-v3 仍高于当时预注册离线下限 69.2666%。离线门通过不等于强度门通过。

## 强度比较结果

| 比较 | 样本 | BC-v3 减 BC-v2.1 配对平均顺位 | 95% CI | 结论 |
|---|---:|---:|---:|---|
| 初始 Greedy/T=1 比较 | 1,000 对 | −0.00400 | [−0.09867, +0.09067] | 未达当时门槛 |
| 后续模型选型 Greedy/T=1 | 20,000 对 | −0.01345 | [−0.03300, +0.00520] | 未达点估计和区间门槛 |

两组结果均未证明 BC-v3 对局强于 BC-v2.1。后续仍由用户指定 BC-v3 为强度锚点，因此 BC-v4 评估应直接比较 BC-v3；不得把上述 NO-GO 改写成 BC-v3 已胜出。

## 保留的机器可读产物

- 训练配置和原始 checkpoint：`configs/train_bc_v3.yaml`、`experiments/bc_v3/checkpoints/`。
- 训练 metrics、完整 test、离线门、决策-v3 全行审计、构建预检、训练输入冻结：`experiments/bc_v3/reference/`。
- 1,000 对逐局结果和分析：`experiments/bc_v3/reference/benchmark_1000/`。
- 20,000 对逐局结果和分析：`experiments/bc_v3/reference/model_selection_20000/`。
- 20,000 对数据 SHA-256：`363b8ec22f650ead35fed325c9afe655be380d944329f2b991406180096be3bb`。

## BC-v4 使用边界

BC-v4 使用新的 `feature-v3`（397 个 dense 特征与最多 3×128 个有序牌河 token，经共享 tile embedding 和 GRU 编码）及既有 `action-v2`。训练从随机初始化开始，不加载 BC-v3 权重。复用 `decision-v3` Parquet 是可行的：有序牌河仍在原始 `discards` 列中，训练时可生成新输入；无需重建 Parquet。正式训练前仍须做 feature-v3 编码、标签/合法 mask 与模型前向的有界正确性检查，并确认完整数据门禁接受新配置。
