# Phase 10.2.18：纯顺位失败的多因素诊断

日期：2026-11-14。目标是先全面检查最近纯顺位 PPO 的候选归因，决定哪些机制有证据支持调整；不把 λ=.99 中断训练当作模型结果，不复用旧强度测试 seed。

## 诊断范围

1. 终局顺位奖的 GAE 信用传播与方差：在固定策略轨迹上比较 λ=.95/.99/1.0 的 GAE 与 MC−V 符号及方差。
2. Value critic：用新 seed 完整对局，在 sampling 与 greedy 两种部署分布上比较纯顺位预测与校准常数基线。
3. Actor 与行为锚：比较 λ=.95、1.0 checkpoint 与 BC-v4 在相同训练/开发状态上的动作一致率、弃牌/吃碰分歧及立直概率。
4. KL/clip 控制：审查训练更新的 target-KL early-stop、approx-KL、clip fraction、BC-anchor KL。
5. 实验有效性：核对 checkpoint/config/source SHA、seed/seat、完整局、奖励账本和合法动作。

## 预注册式判断规则

- 只有多项独立证据指向同一可调机制时才组合调整；孤立线索不作为根因。
- 多因素审查若只留下一个有证据支持、可证伪的处理，可以单独验证该子假设；这不等于按超参数网格逐项试跑。
- 不根据 critic skill、动作一致率或训练 EV 推断实战强度；晋级仍由新 seed 配对顺位、CI、镜像与合法性门决定。
- 任何下一轮训练都用新训练 seed；被中断的 Phase10.2.17 checkpoint 不恢复、不用于强度结论。

## 结果

具体数据、独立核验和结论见 [`PHASE10.2.18_RESULT.md`](PHASE10.2.18_RESULT.md)。
