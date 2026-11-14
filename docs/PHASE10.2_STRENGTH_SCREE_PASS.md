# Phase 10.2 PPO 强度筛选路径汇总

更新日期：2026-11-15。汇总从启动 PPO 到 Phase 10.2.19 首次通过正式强度门的调整路径；对应阶段原文归档于 [docs/PHASE10.2/](PHASE10.2/)。平均顺位越低越好；配对差为 PPO−BC-v4 镜像。正式强度 GO 门为配对差 ≤ −0.020、95% CI 上界 < 0，且镜像 sanity、对局完整性和非法动作门均通过。筛选、安全和机制诊断不等同于正式强度 GO。表中“—”表示该阶段没有 PPO 对 BC-v4 镜像的配对顺位强度数据。

| 阶段 | 评估类型与局数 | 配置调整（相对上一轮） | 模型权重/训练方式（相对上一轮） | PPO平均顺位 | BC-v4镜像平均顺位 | 配对差及95%CI | 结果 |
|---|---|---|---|---:|---:|---|---|
| [10.2.1](PHASE10.2/PHASE10.2.1_RESULT.md) | 正式，20,000 对/臂 | 初始 PPO 配方；首轮基准 | 从 BC-v4 初始化，训练 PPO，100k learner transitions | 2.03945 | 1.49495 | +0.54450 [+0.52560, +0.56410] | 有效 NO-GO |
| [10.2.2](PHASE10.2/PHASE10.2.2_RESULT.md) | 正式，10,000 对/臂 | 基本沿用 10.2.1，只新增 MLP critic trunk 隔离；embedding/GRU 仍共享 | 新 run，从 BC-v4 初始化；不接续 10.2.1 PPO checkpoint | 1.5838 | 1.4994 | +0.0844 [+0.0568, +0.1122] | 有效 NO-GO |
| [10.2.3](PHASE10.2/PHASE10.2.3_RESULT.md) | 正式，10,000 对/臂 | gamma 从 0.99 调至 1.0；其余沿用部分 critic 隔离方案 | 新 run，从 BC-v4 初始化 | 1.5744 | 1.4730 | +0.1014 [+0.0745, +0.1282] | INVALID：镜像未通过预注册 sanity；不计有效 NO-GO |
| [10.2.4](PHASE10.2/PHASE10.2.4_RESULT.md) | 正式，10,000 对/臂 | critic 隔离扩展到独立 embedding/GRU；修正 Rule 菜单并改用 normal99-v2 镜像门 | actor 从 BC-v4 初始化；critic 独立复制初始化并重置 value head；新 run | 1.5657 | 1.5058 | +0.0599 [+0.0328, +0.08670] | 有效 NO-GO |
| [10.2.5](PHASE10.2/PHASE10.2.5_RESULT.md) | 正式，10,000 对/臂 | 组合调整：critic-only warmup、decision-only 统计、梯度控制、BC anchor 与 KL 控制 | 从 BC-v4 新训；11 epochs/110k transitions，包含 10k critic warmup；非续训 | 1.4987 | 1.5106 | −0.0119 [−0.0358, +0.01160] | 有效 NO-GO；CI 跨 0 |
| [10.2.6](PHASE10.2/PHASE10.2.6_RESULT.md) | 机制诊断，768 局；sampling/greedy | 暂不改 PPO 配置；检验冻结策略下 critic 对真实完整回报的预测 | 不训练 actor 或 PPO；固定 actor 采集并评估 critic | — | — | — | critic 主门 INCONCLUSIVE；不是强度 NO-GO |
| [10.2.7](PHASE10.2/PHASE10.2.7_RESULT.md) | critic probe，448 局；256/64/128 训练/开发/测试 | 用完整回报监督检查 critic 可学习性 | BC-v4 actor 冻结，只训练独立 critic；不训练 PPO actor | — | — | — | INCONCLUSIVE；未证明 critic 相对常数有样本外收益 |
| [10.2.8](PHASE10.2/PHASE10.2.8_RESULT.md) | 公开 value 确认，128 局 | 核验公开状态线性 value 基线 | 只用既有训练划分拟合系数；BC-v4 sampling 行为确认，不训练 PPO | — | — | — | value 预测门 SUPPORTED；不是强度结论 |
| [10.2.9](PHASE10.2/PHASE10.2.9_RESULT.md) | critic residual probe，448 局；256/64/128 划分 | 在公开线性 value 上增加残差网络 | BC-v4 actor 冻结，只训练 critic residual；不训练 PPO actor | — | — | — | 残差增益 INCONCLUSIVE；不接入 PPO |
| [10.2.10 安全门](PHASE10.2/PHASE10.2.10_RESULT.md) | 安全小批，128 对；非强度晋级门 | 启用版本化公开状态线性 value；其余正式训练控制冻结 | BC-v4 actor 新初始化；公开 value 系数来自 10.2.9 训练划分，执行新 PPO 训练 | 1.5000 | 1.4141 | +0.08594 [−0.08594, +0.25781] | 预设安全门通过；不代表强度 GO |
| [10.2.10 正式](PHASE10.2/PHASE10.2.10_RESULT.md) | 正式，10,000 对/臂 | 延续公开 value 配置 | 同一新训 PPO checkpoint；非续接旧 PPO | 1.5055 | 1.4871 | +0.0184 [−0.0053, +0.0422] | 有效 NO-GO |
| [10.2.11 小批](PHASE10.2/PHASE10.2.11_RESULT.md) | 资源筛选，1,000 对/臂 | 新增吃碰机会 Bernoulli KL beta=1.0；保留全动作 BC anchor=0.1 及其他设置 | 从 BC-v4 新初始化并训练；不续接 10.2.10 checkpoint | 1.528 | 1.581 | −0.053 [−0.124, +0.018] | 通过资源筛选，可进入正式评估；不是强度 GO |
| [10.2.11 正式](PHASE10.2/PHASE10.2.11_RESULT.md) | 正式，10,000 对/臂 | 配置不变 | 使用上述新训 checkpoint | 1.4867 | 1.4958 | −0.0091 [−0.032, +0.014] | 有效 NO-GO |
| [10.2.12 诊断](PHASE10.2/PHASE10.2.12_RESULT.md) | 策略模式诊断，384 局；另比较 T=0.5 | 固定策略比较 sampling、温度采样和 greedy 的分布差异 | 冻结 10.2.11 PPO 权重；不训练 actor，不作强度评估 | — | — | — | T=0.5 冻结策略行为诊断通过工程门；仅据此设计温度版训练 |
| [10.2.12 小批](PHASE10.2/PHASE10.2.12_RESULT.md) | 资源筛选，2,000 对/臂 | policy temperature 从 1.0 改为 0.5，并统一采样、old logprob、PPO loss 与 BC anchor 的概率分布 | 从 BC-v4 新初始化并训练 100k；保留吃碰约束；不续接旧 PPO | 1.4595 | 1.4545 | +0.005 [−0.041, +0.051] | 有效小批 NO-GO；未进正式 10k |
| [10.2.13 小批](PHASE10.2/PHASE10.2.13_RESULT.md) | 资源筛选，2,000 对/臂 | 温度恢复 1.0；score scale 从 0.001 降至 0.00005；critic 使用匹配目标的公开 value 初始化 | 从 BC-v4 新初始化；新训 PPO，保留 lambda=0.95、吃碰约束及其他冻结项 | 1.4765 | 1.5120 | −0.0355 [−0.087, +0.0175] | 有效小批 NO-GO；未进正式 10k |
| [10.2.14 纯顺位臂](PHASE10.2/PHASE10.2.14_RESULT.md) | 三臂筛选，2,000 对/臂 | 纯顺位 reward scale=0；与混合目标作同 seed 对照 | seed43；从同一 BC-v4 起点新训纯顺位 PPO | 1.4845 | 1.5215 | −0.0370 [−0.0875, +0.0120] | 小批未通过 GO 门 |
| [10.2.14 混合臂](PHASE10.2/PHASE10.2.14_RESULT.md) | 三臂筛选，2,000 对/臂 | rank + 0.05 score；与纯顺位臂匹配训练设置 | seed43；从同一 BC-v4 起点新训混合目标 PPO | 1.5175 | 1.5215 | −0.0040 [−0.05651, +0.0495] | 小批未通过 GO 门；纯顺位对混合的差为 −0.0330 [−0.0865, +0.0210]，胜负不确定 |
| [10.2.15](PHASE10.2/PHASE10.2.15_RESULT.md) | 对既有纯顺位 checkpoint 正式评估，10,000 对/臂 | 不改模型配置；用全新评估 seed 测 10.2.14 纯顺位 checkpoint | 不重训；沿用 seed43 checkpoint | 1.4974 | 1.5141 | −0.0167 [−0.0398, +0.0058] | 有效 NO-GO |
| [10.2.16](PHASE10.2/PHASE10.2.16_RESULT.md) | 资源筛选，2,000 对/臂 | 纯顺位 GAE lambda 从 0.95 调至 1.0 | 从 BC-v4 新初始化；seed44 新训 100k | 1.512 | 1.4935 | +0.0185 [−0.0345125, +0.0705] | 有效小批 NO-GO；未进正式 10k |
| [10.2.17](PHASE10.2/PHASE10.2.17_RESULT.md) | 训练中断；未进行强度评估 | 纯顺位 GAE lambda 从 1.0 调至 0.99 | seed45 从 BC-v4 新训；因集中诊断约束在 epoch 5/10 后中止，仅完成 50k/100k | — | — | — | 不完整训练；不能作为模型结果，未运行 screen |
| [10.2.18](PHASE10.2/PHASE10.2.18_RESULT.md) | 多因素机制诊断；两个 checkpoint 各 384 局 | 集中审查 GAE、critic、actor/行为锚及 KL/clip；只留下 lambda=0.99 的可证伪候选 | 不训练新 actor；不恢复 10.2.17 部分 checkpoint | — | — | — | 诊断支持重新验证 lambda=0.99；其他可疑因素证据不足 |
| [10.2.19 小批](PHASE10.2/PHASE10.2.19_RESULT.md) | 资源筛选，2,000 对/臂 | 纯顺位 lambda 从 0.95 调至 0.99；其余按预注册配置冻结 | 从 BC-v4 新初始化；新训练 seed46、100k；不是续训 10.2.17 | 1.4595 | 1.5250 | −0.0655 [−0.1185, −0.0130] | 预注册小批 GO；允许进入正式强度评估 |
| [10.2.19 正式](PHASE10.2/PHASE10.2.19_RESULT.md) | 正式，10,000 对/臂 | 配置不变 | 使用 10.2.19 同一 seed46 checkpoint；未续训或重选权重 | 1.4847 | 1.5136 | −0.0289 [−0.0519, −0.0055] | 有效 GO；首个通过正式强度门的 PPO 候选，仍需独立训练 seed 确认 |

截至 10.2.19，首个正式 GO 来自纯顺位、GAE lambda=0.99、训练 seed46 的新 PPO；该单 seed 结果尚不足以晋级并替代 BC-v4 强度基线。下一步是按预注册方案用独立训练 seed 和新评估 seed 做确认，不依据本轮结果继续调参。
