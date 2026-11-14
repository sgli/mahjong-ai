# Phase10.2.10：公开线性value的PPO候选

2026-11-14。10.2.8/10.2.9证明公开状态线性预测器在新保留集有回报预测信息，残差网络未增加样本外收益。版本化`model.value_public_coefficients`承载公开value，actor仍是BC-v4 feature-v3 GRU，value只依赖决策时公开状态并在PPO中可训练。系数来自10.2.9训练256游戏，不使用开发/测试/确认集拟合。该证据只支持改善基线的工程实验，不证明PPO顺位会改善。

## 正确性与pilot

模型value输出逐位对照独立NumPy公开特征计算；value-only梯度不触碰actor，BC初始化actor logits逐位相同，state_dict严格保存加载；与GAE和warmup相关测试合计14项通过。2×2048 learner-transition CUDA pilot通过，非法0、loss有限、optimizer均执行，value/actor都有更新；checkpoint可加载，2轮数值不作强度证据。epoch2 rollout更新前对自举GAE target的EV0.648，只作短跑健康指标，不是独立MC泛化证据。

小批配对安全检查：候选pilot epoch2 vs3×greedy BC-v4，镜像BC-v4 vs3×BC-v4，128对新seed980000–980127、16 workers、greedy/T1。仅筛明显崩溃：完整合法、checkpoint身份一致、配对顺位差<=+0.20；镜像sanity采用normal99-v2作有效性参考。样本太小不作GO/晋升和超参选择。失败则停止，不进入正式训练。

## 正式冻结方案

配置`configs/train_ppo_v4_public_value.yaml`，SHA 1718b504b5ced5f02b7dcb1309a64f44bd45880f1d82894e53d0e77071ae72a2。从冻结BC-v4最佳模型初始化，seed42、10×10000 learner transitions、gamma1/lambda0.95、decision-only、BC anchor0.1、actor/critic梯度裁剪0.5、target KL0.01、lr5e-5、无额外entropy奖励，对手3×greedy BC-v4。latest epoch10是唯一候选；不以离线gold/top1或短跑强度挑checkpoint。新value以公开系数初始化但训练中更新，模型版本写入checkpoint并严格加载。

训练健康门：100k精确预算、全部loss有限且illegal0，选中best与latest权重一致，config SHA未变，value模式与系数正确；失败即停止，不进入强度评估。

正式强度评估：每臂10000局，candidate PPO vs3×BC-v4，mirror BC-v4 vs3×BC-v4，seed720000–729999、seat=index%4，16 workers、bootstrap PCG64 seed10331/10000次、normal99-v2。GO要求配对平均顺位差<=−0.020、95%CI上界<0、镜像有效且非法0；其余有效结果NO_GO，有效性失败INVALID。不补样本/不事后调门槛。

估时：上次完整critic100k actor+10k warmup训练35分31秒，新模式省去value GRU但没有同规模实测，pilot4k约32秒只作参考；正式100k暂估20–35分钟。10k配对参考上次100分19秒，预计90–120分钟。训练+评估总约2–3小时；不设上限、不实时监控，流水线自动在健康门后进入强度评估。
