# Phase10.2.7：BC-v4 actor 下的 critic 可学习性对照

2026-11-14。承接Phase10.2.6主门INCONCLUSIVE。固定BC-v4 sampling actor和三家greedy BC-v4，采集新的完整游戏，训练独立critic来预测当前决策至终局reward-v1累计回报。该试验只回答可学习性，不构成PPO强度证据，也不直接晋升critic checkpoint。

## 冻结协议

- 模型身份：BC-v4 best SHA814c6eba4e163d126cff7bc27194f11612263c745328abd1b1d9ae94b6716937。feature-v3/action-v2；只使用learnerset轨迹。
- 448场新seed930000–930447：训练256、开发64、最终保留128，三组整局分离，seat=index%4。与10.2.6诊断seed无交集，保留集只在开发选完epoch后使用一次。
- 8 CPU workers并行采集。pilot16场7.609秒；正式采集估4–8分钟。每场必须完整、illegal0、账本一致，源数据逐game保存，不能从旧PPO策略轨迹借用回报。
- 模型从BC-v4复制actor和独立value encoder/trunk，value head重置；actor所有参数冻结。critic用完整游戏MC returns、全状态MSE训练，Adam lr1e-4、每轮遍历256游戏、3轮、按整局更新、value组梯度裁剪0.5。
- 每轮开发集按多合法动作状态逐游戏等权MSE选择最低epoch，早轮同分优先；不看最终保留集选epoch。训练集常数预测按每游戏选择状态回报均值拟合。最终保留集使用相同口径，PCG64 seed10327、10000次整局bootstrap；skill=1−critic MSE/常数MSE，下界>0为PASS，上界<0为FAIL，否则INCONCLUSIVE。基线只由训练集确定。
- 这只是工程路由门，不能直接证明更强策略。即使PASS，当前probe的critic训练目标与标准PPO的GAE/GAE bootstrap不同，正式使用前须验证初始化与继续PPO的数值和策略健康。
- 配置SHA：采集 f69a18fee82bbb4ab0e4db19213aa140b737aa409e48a43a91523c0b70152cf5，probe be8df80ba06ddd8f3fde589bd2c7051be563bacf04ac250895e1cf2475f68b1c。入口`scripts/collect_phase10_2_7.py`和`scripts/probe_phase10_2_7.py`。两阶段均无时间上限，预计总5–11分钟；若运行估时/吞吐超过20分钟，停止实时监控等自然完成。

## 分流

训练和开发均无法改善：检查奖励对齐、优化和表示；训练改善但开发/保留不足：优先查覆盖和过拟合。保留集PASS：再冻结下一版critic预训练+稳定化PPO组合方案及唯一10k强度门，不能把该probe模型直接当成PPO候选。FAIL/INCONCLUSIVE均不自动追加本批次或扩大actor预算，报告不确定性。

## pilot

16场BC-v4采集7.609秒；8训练/4开发/4保留的两轮probe1.422秒，actor逐参数冻结、输出有限。样本极小，pilot skill不作质量结论。

## 正式反馈（2026-11-14）

448场采集156.328秒；3轮probe15.203秒。训练skill+2.506%、开发+0.059%、测试−1.327%，测试区间[−3.265%,+0.782%]，INCONCLUSIVE。actor逐参数与BC相同。详见[PHASE10.2.7_RESULT.md](PHASE10.2.7_RESULT.md)。
