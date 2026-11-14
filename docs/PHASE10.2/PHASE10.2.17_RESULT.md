# Phase 10.2.17 结果：纯顺位 λ=0.99

## 预注册

见`PHASE10.2.17.md`。单变量为GAE λ=.99；其余按纯顺位基线保持。

## Pilot / 100k / screen

Pilot已通过：4096 transitions、lambda .99、非法0、loss有限。主训练因用户指出未遵守`10-project.md`集中诊断约束，于epoch5后被终止；日志显示epoch1–5各完成10,000 transitions、非法率0。该run只有50k/100k，不作为完整训练结果，未进行training verification和seed1120000–1121999 screen。部分checkpoint/log保留在`experiments/ppo_v4_rank_lambda099/seed45_100k/`用于审计，不续训此run。

中断原因：Phase10.2.16已单独尝试λ1.0，随后又把λ调为.99另跑一轮；这违反`docs/memory/10-project.md`要求出现失败/卡门时集中分析多种原因并一次组合调整。λ=.99批已停止，不再单变量续跑。下一步须先完成集中诊断、冻结组合方案后再决定是否开新跑批。
