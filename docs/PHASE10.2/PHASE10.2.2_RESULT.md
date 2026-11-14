# Phase 10.2.2 结果：critic trunk 隔离 PPO

状态：100,000 learner-transition 训练、validation drift 诊断和10,000-pair strength benchmark 均完成；预注册判定 NO-GO。日期：2026-11-13。

2026-11-13实现审查更正：本轮仅分离MLP trunk，tile embedding/GRU仍共享，故下文“critic隔离”应按部分隔理解。此前“critic梯度只进入value路径”的表述不成立；完整隔离修复与梯度证据见`PHASE10.2.4_RESULT.md`。对局结果与NO-GO不变。

## 固定方案与训练

方案与10k-pair评估范围见 [PHASE10.2.2_PREREG.md](PHASE10.2.2_PREREG.md)。本轮唯一训练变量为 `model.value_separate_trunk=true`，其余预算、初始化 checkpoint、对手池与超参保持 Phase 10.2.1 设置。

- seed 42；100,000 learner transitions、400,069 environment steps、10 epochs；feature-v3/action-v2，reward-v1。
- BC-v3 / BC-v4 / Rule-v1 对手权重 0.50 / 0.25 / 0.25；每轮非法率 0。
- 实际训练时长约 **569秒（9分29秒）**，与 Phase 10.2.1 572.8秒接近。
- 配置：`configs/train_ppo_v4_critic_isolated.yaml`；输出：`experiments/ppo_v4_critic_isolated/seed42_100k/`。
- epoch10 `best.pt` SHA-256 `8ffa51e5d897605e50cf71ec6785af30377c3c7922c1fb86ef9d8b1bc17b94a`；配置 SHA-256 `0ebaf366e62433a4db75de9f3dd6e29012c82ce269c4669a94df13b5ae53228d`。

## 事后 validation 策略漂移诊断

固定使用 decision-v3 validation 前100,000行（按排序 shard 前缀）；诊断不用于 checkpoint 选择或调参。对比原共享 trunk PPO 的 epoch10 与本轮隔离 trunk epoch10：

| 指标 | 共享 trunk epoch10 | 隔离 trunk epoch10 |
|---|---:|---:|
| 与 BC-v4 greedy action 一致率 | 67.127% | 85.606% |
| 对 gold action top-1 | 58.414% | 67.244% |
| KL(BC-v4 ‖ PPO) | 0.4010 nat | 0.07434 nat |
| PPO entropy | 1.0752 nat | 0.8176 nat |

epoch1→10 drift 也更小：隔离 trunk 的 greedy agreement 从96.152%降至85.606%，KL从0.00601升至0.07434；原共享 trunk则88.916%降至67.127%、KL从0.05221升至0.40101。结果支持“共享 critic loss 是 actor 漂移的重要来源”这一机制，但仍不能证明它是原强度失败的唯一原因。离线 top-1 仍比 BC-v4 低3.10个百分点；强度结论只由对局门决定。

原始诊断：`experiments/ppo_v4/diagnostics/policy_drift_by_epoch_validation_100k.json` 与 `experiments/ppo_v4_critic_isolated/diagnostics/policy_drift_by_epoch_validation_100k.json`。

## 预注册强度评估

按用户于评估启动前的指令，由20,000对调整为 **10,000对**；效应阈值与镜像 sanity 不变。Candidate 为隔离 trunk PPO epoch10，对手3×BC-v4；mirror 为 BC-v4 对3×BC-v4；seed 640000–649999，Greedy/T=1，16 workers，bootstrap 10,000次 seed 10321。输出：`experiments/ppo_v4_critic_isolated/strength_10k_seed640000_v1/`。实际耗时 **5,613.693秒（约93分34秒）**，candidate arm 2,857.825秒，mirror arm 2,754.031秒。

| 指标 | PPO critic-isolated | BC-v4 mirror |
|---|---:|---:|
| 局数 | 10,000 | 10,000 |
| 平均顺位 | 1.5838 | 1.4994 |
| 顺位分布（0/1/2/3） | 2,283 / 2,428 / 2,457 / 2,832 | 2,495 / 2,497 / 2,527 / 2,481 |
| 和牌/局 | 2.3747 | 2.2463 |
| 放铳/局 | 1.6025 | 1.3615 |
| 立直/局 | 2.2167 | 2.5410 |
| 副露/局 | 10.0315 | 4.6929 |
| 非法动作 | 0 | 0 |

配对顺位差（PPO−mirror）为 **+0.0844**，bootstrap 95% CI **[+0.0568, +0.1122]**；mirror mean rank 1.4994，在 `1.5±0.02` 内。CSV 独立复算与 result.json 完全一致，种子唯一且覆盖640000–649999，席位各2,500局，两臂均完整有效。

**预注册判定：NO-GO。** critic trunk 隔离确实明显减少 validation 上的 BC 策略漂移，且本轮强度比共享 trunk PPO 好得多；但仍显著弱于 BC-v4。对局行为表现为副露约为镜像的2.14倍，放铳更多、立直更少。该证据说明仅隔离 critic 梯度不足以恢复基线强度；当前 PPO 不能晋升，也不应继续沿同一100k配方续训。BC-v4 继续作为强度基线。

## 后续

进一步审查 reward/GAE 后，优先检验回报时间尺度：reward-v1 对立直立即记−1，终局顺位奖仅±1/±2；本轮平均每条 learner 轨迹613.5步，`gamma=.99` 使远期顺位奖励大幅折扣。BC KL anchor 暂列次级候选；先按 [PHASE10.2.3_PREREG.md](PHASE10.2.3_PREREG.md) 只改 `gamma` 为1.0。此机制仍属待验证假设。
