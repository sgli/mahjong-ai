# Phase10.2.6：冻结策略的 critic / 完整回报联合诊断

2026-11-14。用户已要求整合PHASE10.2.5_DIAGNOSTIC_REPORT并按本方案执行。阶段范围为集中诊断、条件分流和下一轮组合设计；本入口不训练actor，也不自动启动10k强度评估。10.2.5有效NO_GO不改判。

## 1. 本轮调整的判断

保留10.2.5完整critic隔离、decision-only、BC anchor、分组梯度裁剪和更新控制。先核验critic/回报信号，不默认扩到1M PPO、不因低EV重训BC、不再为每个局部变量单独跑长强度评估。

报告原建议以greedy为critic主验证分布，本方案修正为sampling主分析、greedy辅助：learner训练时sampling，greedy用于强度评估；两种模式的期望回报不必相等。诊断critic目标是reward-v1剩余累计回报，最终顺位只作分层和奖励分解，不将value直接当rank预测。

## 2. 冻结身份与采集协议

- 候选：`experiments/ppo_v4_combined/seed42_110k/best.pt`，SHA c62ab8289004c5d59599f3c1b374e717d51a057232515836ee5e80bfe1b31937；actor、critic均冻结，无优化器。
- 对手：三个greedy/T1 BC-v4，SHA 814c6eba4e163d126cff7bc27194f11612263c745328abd1b1d9ae94b6716937。
- 当前feature-v3/action-v2、环境、reward-v1不变，gamma1；候选分别sampling/T1与greedy/T1。
- 校准种子921000–921127（每模式128游戏）；保留种子921128–921383（每模式256游戏），总768游戏。seat=(seed−921000)%4，每个split四席平衡。
- 同seed两模式配对，但不合并样本量；跨模式比较须按seed成对分析。每游戏只提取候选learner轨迹，三家不同策略的轨迹不作为该critic的on-policy样本。
- 只接受自然完整终局，max_game_steps=20000为逻辑异常防护，不是时间上限。若不完整、illegal、奖励账本不一致、模型/源文件身份变化则整批failed；不得静默筛掉失败游戏或补新seed。
- 8个CPU worker，各torch线程1，每worker缓存模型；以独立seed确定采样，不共享全局随机流。pilot通过后按其吞吐估时。
- 正式配置`configs/diagnose_phase10_2_6.json`，SHA 5dbfd3221e5f3d7b318c69e7db43004ffe5692f80b4199c71a5bb619722b94c2；入口`scripts/diagnose_phase10_2_6.py --config configs/diagnose_phase10_2_6.json`。

## 3. 一次采集，复用多项诊断

每个完整游戏落盘压缩npz：模型输入、合法mask、动作/旧logprob、V、reward、done、阶段、终局分数/顺位、MC returns、点数与顺位bonus贡献。逐game保存SHA和manifest。既有评估JSON没有逐状态轨迹，旧8场审计仅存汇总，因此需要新采集；不重建decision-v3 Parquet。

主目标G_t=sum从当前transition开始到终局的reward，终局bonus已在最后transition内，只计一次。严格检查逐状态长度、最后done、合法动作和四席点数账本。阶段按当时可见bakaze/kyoku分层；终局按实际rank分层，当前环境无显式完整end-reason字段，不编造终局原因。

### Critic样本外质量

- 主样本：保留集sampling中的多合法动作状态；每游戏先平均误差，再每游戏等权，避免长游戏/强制动作支配结果。
- 对照：校准集拟合的常数均值，及辅助仿射校准a+bV。所有参数只用校准集，不使用保留集均值拟合基线。
- 主指标：skill=1−mean_game(MSE_critic)/mean_game(MSE_constant)。游戏级bootstrap10000次，PCG64 seed10326，百分位95%区间。
- 下界>0为诊断PASS；上界<0为FAIL；跨0或退化目标为INCONCLUSIVE。该门用于工程排序，不是强度GO，也不是证明critic有用的必要条件；隐藏信息和回报噪声可降低可预测性。
- 辅助输出：MAE、逐游戏EV/相关性均值及游戏级bootstrap区间、校准分箱、全状态口径、阶段/终局rank分层、greedy模式；辅助指标不可替换主指标来挽救判定。区间固定校准参数，反映保留游戏抽样不确定性，不覆盖全部训练种子或基线拟合不确定性。

### 优势与动作信号

同一缓存轨迹离线计算lambda0.95/0.97/1.0，比较GAE与MC−V的偏差、符号和方差，按吃/碰/过/立直等统计可选机会与所选动作优势。无需为每个lambda另跑环境。

lambda1等于MC−V是代数恒等式，不能以其相关性最高认定最优。单次实际回报不是反事实Q；按所选动作汇总不能估计吃/碰相对过的因果价值。若需决定lambda或actor更新强度，应在后续训练/开发数据上检验学习效果，不用本轮保留集反复选参。

## 4. 完成后的分流

| 结果 | 下一轮调整方向 |
|---|---|
| sampling FAIL，仿射校准明显改善 | 优先核查尺度/偏置、目标和critic训练覆盖；做冻结actor的critic学习诊断 |
| sampling FAIL，仿射校准也弱 | 区分覆盖/拟合/不可预测噪声；用新训练与开发游戏检验MC监督是否能泛化，不自动扩大网络 |
| sampling INCONCLUSIVE | 保留不确定结论；不自动补样本到显著，按区间与分层制定有界下一步 |
| sampling PASS，greedy弱 | 优先研究策略模式差异，不能用greedy不佳否定sampling critic |
| critic有预测价值但优势质量可疑 | 比较critic目标与GAE时间尺度的组合，小训练验证；不按MC恒等式选lambda |
| 信号可用但策略增益弱 | 将有效选择/完整游戏预算与更新约束联合设计，随后只做一次正式强度门 |

新候选若从BC-v4初始化，其critic预训练需使用匹配冻结BC-v4采样策略的数据，不能把当前PPO的回报直接当成BC策略on-policy标签。本轮保留集诊断后封存，新的微调/选参另用训练/开发数据及最终独立验证。

报告输出后先完成归因与组合方案冻结，再启动actor训练。新强度候选唯一选择、训练预算和种子在启动前冻结；继续10k、配对差<=−0.020且CI上界<0、normal99-v2及illegal0。GO之后才独立训练seed确认。

## 5. 验证、时间和产物

- 单进程CUDA pilot：16场，66.487秒，9255 learner transitions；逐步回报与点数/bonus分解重建通过，最大账本误差7.11e-15；只作正确性/吞吐证据，pilot诊断结果不调正式门。
- 11项单测通过：包括当前奖励只计一次、GAE lambda1恒等式、三态诊断门、校准/保留集隔离，以及既有GAE边界回归。
- 8-worker CPU pilot已完成：32场15.958秒；正式768场按此吞吐估计6–12分钟。小批固定开销与长游戏差异保留不确定性，不将CUDA单进程时间直接除8。
- 预计>20分钟的批次不设时间上限、不实时监控，等用户通知完成；训练/强度不由本脚本串联。
- 状态与输出：`experiments/ppo_v4_critic_diagnostic/phase10_2_6_v1/{state.json,manifest.json,result.json}`，每模式分目录存npz。
- 汇总入口：`docs/PHASE10.2.6_RESULT.md`。不提交任何改动。

## 6. 执行反馈（2026-11-14）

768场正式诊断已完成，285.856秒、450842 learner transitions；sampling主skill −3.490%、95%CI[−8.009%,+0.664%]，INCONCLUSIVE。执行证据不足分支，不自动扩样或训练actor。独立核验通过；具体结果与下一轮critic可学习性方向见[PHASE10.2.6_RESULT.md](PHASE10.2.6_RESULT.md)。
