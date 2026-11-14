# Phase10.2.12 执行结果

2026-11-14。诊断范围和决策边界见`PHASE10.2.12.md`。本轮小批筛选有效但未过门，停止该候选；BC-v4仍是强度基线。

## 既有BC状态的剩余动作分歧

`scripts/analyze_phase10_2_12_actions.py`只读10.2.9训练前32局/开发前32局，不读最终test；原始结果`experiments/ppo_v4_call_anchor/phase10_2_12_actions_v1/result.json`。训练/开发可选状态5649/5779，同状态贪心一致率0.9411/0.9397；非吃碰分歧323/347次，其中双方均选择打牌但牌张不同309/340次，吃碰相关分歧仅8/6次。开发组立直合法机会154次，PPO−BC立直概率局均+0.01053。故剩余分歧主要在打牌选择；该BC采样状态无反事实动作结果，不能判断PPO或BC的弃牌更好，单凭计数不预注册弃牌约束。

## 新完整轨迹：训练与部署策略模式差异

配置`configs/diagnose_phase10_2_12.json`、checkpoint SHA`450a38c6…65ad`；新seed990000–990191，sampling/greedy各64校准+128保留，共384完整游戏，8 CPU workers，实际126.749秒。状态complete，轨迹账本、SHA、seed/split、mask与终局检查通过；原始结果`experiments/ppo_v4_call_anchor/phase10_2_12_diagnostic_v1/result.json`。同seed同初始状态、同greedy BC-v4对手，只有learner使用sampling/T1或greedy。

sampling保留128局的可选状态value相对校准常数MSE skill+2.019%、游戏级95%CI[−2.874%,+7.163%]，INCONCLUSIVE；greedy为−20.075%、CI[−32.539%,−7.787%]，FAIL。greedy每局内value/MC相关均值0.022，区间跨0；sampling为0.344。初始状态value在配对模式相同，均值−10.634，但完整回报sampling均值−9.514、greedy−0.730。配对128局greedy−sampling完整回报+8.784、游戏级bootstrap95%CI[+4.522,+13.114]；顺位差−0.492、CI[−0.734,−0.258]。sampling每局点数贡献均值−8.819、顺位bonus−0.695，greedy分别−0.762、+0.031。模式差主要来自得分回报，而非账本错误。每次决策的critic原本在sampling策略下训练，不能要求其无偏预测greedy回报；这也暴露了训练采样分布与贪心部署分布的明显差距，但尚未证明它是10.2.11未过强度门的唯一原因。

sampling保留局λ0.95的GAE与MC−V符号一致率局均0.666、局内方差5.663；λ0.97为0.710/8.685；λ1为1.0/59.728。λ1一致率是代数恒等，且方差约10.5倍，不据此提高λ。greedy的对应value失配也不意味着应把贪心动作直接塞进现有PPO重要性比率：贪心行为概率为1，与模型softmax行为概率不一致，会破坏on-policy目标。

下一步仅用**预先固定的T=0.5**做冻结10.2.11策略的中间温度小量采集，与同seed的T1/greedy按局配对；T0.5未训练新actor，不作为强度。若其回报/顺位明显向greedy靠近，再设计行为概率与PPO loss严格一致的版本化温度方案，并先经小批量强度筛选。若无此迹象则停止，不直接跑下一轮100k/10k。

## 冻结策略T=0.5的前置门

复用同seed990000–990191及相同校准/保留划分，`configs/diagnose_phase10_2_12_temperature_05.json`仅将采样温度设为0.5，新384完整游戏126秒。独立脚本`.venv/Scripts/python.exe scripts/verify_phase10_2_12_temperature.py`核对两批原始轨迹SHA、同seed/seat、greedy轨迹动作与MC逐项一致；16个保留游戏抽样决策的保存logprob对T0.5模型重算最大绝对差2.62e−5（CPU单状态与batch数值误差，容差1e−4），PASS。

保留128局T1/T0.5/greedy完整回报均值−9.514/−2.759/−0.730，均位1.977/1.570/1.484。T0.5−T1配对回报+6.755，游戏级95%CI[+2.890,+10.597]；均位差−0.406，CI[−0.633,−0.188]。相对T1→greedy差距，点估计回报走完76.9%、顺位走完82.5%；满足事前区间和至少一半差距的全部工程门。独立产物`experiments/ppo_v4_call_anchor/phase10_2_12_temperature_05_v1/independent_verification.json`中`engineering_gate=PASS`。这是冻结旧策略的行为采样诊断，不证明新策略训练后相对BC-v4的强度；也不意味着直接用greedy采样训练PPO是合法的。

因此进入**一次**版本化T0.5训练设计：模型前向策略logits缩放，采样器、保存old_logprob、PPO ratio/entropy、BC anchor均读取同一已缩放分布；评估仍greedy，argmax不变。保留10.2.11的其他配置（包括吃碰约束），从BC-v4重新初始化。先定向测试与短跑，再冻结100k训练和新seed小批强度门；是否跑10k正式强度评估取决于小批门，不能凭本轮冻结策略诊断直接启动。

## T0.5策略接口与短跑健康门

`MLPPolicy/RiverSequencePolicy.policy_temperature`默认1.0，T1旧路径不做额外缩放；新配置在模型forward中输出`logits/T`，因此采样与PPO loss使用同一概率。训练加载冻结BC anchor后设同温度，初始政策约束不会因温度不同而产生虚假KL。新配置SHA=`c2096e93dfbb8c0367d857b6328cdf02b721015eb4545a16fa3aa24171111d2c`。`pytest tests/test_policy_temperature_v1.py tests/test_call_anchor_v1.py tests/test_public_value_v1.py tests/test_ppo_combined_controls.py -q`共13项通过：概率缩放、greedy不变、同温度行为logprob/ratio与BC KL、旧约束回归。

2×2048 CUDA pilot完成约31秒，精确4096 learner transitions、两轮非法0、loss有限、16次更新/轮；mean ratio1.00052/1.00027，checkpoint严格加载恢复T0.5、model_version=`ppo-v4-tempered-v1`。额外在真实三家greedy BC-v4对手下采集一场完整游戏，3049环境步/763 learner步、非法0；保存old_logprob与同checkpoint批量重算T0.5 logprob最大绝对差1.70e−5（CPU batch/单状态容差1e−4），`experiments/ppo_v4_tempered/pilot_2x2048/collector_verification.json`为PASS。短跑只证实目标接口正确和能更新，不是强度证据。允许进入冻结100k正式训练。

## T0.5正式训练与行为门

seed42正式10轮100k learner transitions完成，400075环境步、166完整游戏、非法各轮0、policy/value loss有限；日志15:19:07–15:31:32约12分25秒，在事前12–20分钟估时内。best/latest同为epoch10且逐参数一致，checkpoint严格恢复`policy_temperature=0.5`、`model_version=ppo-v4-tempered-v1`；best SHA=`4c4da26d3af9d0f76a93830c9dd6a9787a6c62737c964b6d47bf5583838dbdf1`，配置SHA=`c2096e93dfbb8c0367d857b6328cdf02b721015eb4545a16fa3aa24171111d2c`。10轮实际645次optimizer updates（10.2.11为800）；固定target KL0.01在第1/2/3/5/6/7/10轮提前停止更新，说明温度使相同KL控制更频繁触发。此差异是温度方案的实际后果，不能暗中放宽KL或把训练预算解释成800次更新。

固定BC-v4 sampling状态按同温度BC比较：训练/开发各32局的PPO−BC吃碰概率差−0.00310/−0.00225，开发低于事前+0.015上限；greedy动作一致率0.96135/0.95941，开发高于0.90。训练组同向。行为门通过；来源`experiments/ppo_v4_tempered/phase10_2_12_behavior_v1/result.json`，同状态对照不是对局强度证据。因此进入全新seed2000对小批资源筛选，结果如下。

## 新seed 2000对小批筛选与判定

`scripts/benchmark_paired_ppo.py`按冻结协议运行seed800000–801999、候选/BC-v4镜像各2000完整游戏，16 workers、PCG64 bootstrap seed10344/10000次、greedy与3×BC-v4对手。实际两臂耗时614.262/614.863秒，总计1230.473秒（20分30秒），处于事前18–23分钟估计内。候选/镜像checkpoint SHA分别为`4c4da26d…8dbdf1`/`814c6eba…b6716937`。两臂各2000局、每座位各500局、非法动作0，runner `valid_run=true`。原始逐局、CSV和结果在`experiments/ppo_v4_tempered/screen_2000pair_seed800000_v1/`；独立命令`.venv/Scripts/python.exe scripts/verify_phase10_2_12_screen.py`核对seed/seat、逐局合法性、SHA、CSV及bootstrap，`independent_verification.json`为`status=PASS`。

候选均位1.4595、镜像1.4545，配对均差（候选−镜像）**+0.005**，95%CI **[−0.041,+0.051]**；镜像距1.5为0.0455，满足事前≤0.05的额外sanity门。强度点估计未达到≤−0.020，区间上界也未低于0，因此事前小批总门`screen_pass=false`。这是**有效的小批NO_GO**，并非正式10k强度结论；按冻结规则不补seed、不改温度或阈值、不启动seed820000–829999的10k/臂正式评估。

## 原因边界与后续

本轮实现正确性、训练健康、BC同状态行为门和对局完整性均通过，尚无证据支持把失败归于非法动作、错配logprob或无效benchmark。冻结10.2.11策略的T0.5诊断说明**采样**对局回报向greedy靠近；对固定权重，温度缩放不改变greedy的argmax。新模型要在greedy强度上获益，仍需训练更新改变动作排序且改变方向有利。本轮开发BC状态greedy一致率95.94%，小批配对差+0.005且区间跨0，没有观察到满足门槛的净收益。固定KL阈值下仅645次更新（旧方案800次）可能限制策略变化，但这只是待检验解释，不能由本轮跨seed比较或行为一致率直接断定因果，更不能事后放宽KL重跑同批seed。

停止10.2.12温度版PPO候选，保留T0.5一致性接口和全部原始证据。下一步先在已有训练/开发轨迹中离线审查**打牌动作排序改变与优势/终局回报是否同向**，并区分采样分布修正与贪心部署强度；不复用封存测试资料或强度seed提出机制。若无法提出可证伪机制与新seed小批门，则不再启动PPO训练。BC-v4继续为强度基线。

## 小批未过门后的集中离线排查

2026-11-14按`docs/memory/10-project.md`的集中排查约束复查训练更新、奖励/顺位口径、采样与贪心部署。复现命令`.venv/Scripts/python.exe scripts/analyze_phase10_2_12_followup.py`；只读10.2.9已有BC-v4 sampling训练256局、开发64局与10.2.11/10.2.12训练checkpoint的`metrics_history`，不读封存测试或新强度seed。逐文件SHA、奖励总账与终局顺位奖励断言通过；产物`experiments/ppo_v4_tempered/phase10_2_12_followup_v1/result.json`。

1. **训练探索/更新约束：**同样100k learner transitions，旧T1/新T0.5分别有800/645次优化更新，后者7/10轮触发固定`target_kl=0.01`早停。全状态平均策略熵0.25086/0.12233，choice fraction均约0.30；熵包含强制动作，仅用于同口径对比。完整训练局168/166，规模相近。T0.5明显压低采样熵并使早停频繁，但旧T1的800次更新也未过正式强度门，故“更新少”不能单独解释PPO长期未过门；两次训练不同轨迹，不能把跨run差异视为净因果效应。
2. **奖励与目标：**`reward-v1`点数差按0.001计，终局顺位奖为`(2,1,-1,-2)`，后者仅在最后一步入账；训练`gamma=1, lambda=.95`。已有BC训练/开发局的点数回报标准差17.063/18.598，顺位奖标准差1.435/1.643；点数与顺位仍有较强相关（rank与score回报相关−0.825/−0.868），不能把点数塑形简单判为错误。仅计算**直接奖励项**经`(gamma*lambda)^距离`传播到choice状态的trace，开发集顺位/点数平均绝对值0.063/1.178，88.65%的choice状态距终局>60个learner步骤（0.95^60≈0.046）；顺位直接贡献在早期动作上较弱。该计算不含value bootstrap，**不是完整GAE**，不证明rank信息完全不能通过critic传递；但结合旧10.2.11模型在10.2.12保留局greedy下critic MC skill−20.075%，长期顺位信用分配值得优先审查。
3. **部署证据边界：**固定10.2.11权重的T0.5采样改善不改变greedy argmax；新候选在固定BC开发状态与BC贪心动作一致率95.94%，实际新seed2000对差+0.005、CI跨0。现有证据只支持“没有显示净顺位收益”，不能证明个别弃牌排序改动有害，也不能据不同seed的10.2.11正式10k与本轮2k差值判定新旧候选孰优。已用过的强度seed只用于这项结果的审计，不作为新机制选择依据。

综合判断：未发现新的非法动作、logprob错配或benchmark无效证据；更可信的**待验证机制组**是低熵探索和KL早停限制有效更新、点数主导的短程训练信号与终局顺位目标只部分对齐、以及sampling critic到greedy部署的口径差异。三者都不足以单独构成“已定位根因”。现有10.2.9 BC轨迹`npz`只有执行动作及其实际MC回报，没有替代弃牌后的状态/回报；10.2.12训练仅保存checkpoint和汇总历史，没有可供重算的逐动作优势。因此不能凭现有文件核对PPO翻转弃牌的反事实价值。下一步先设计有界的动作分歧分层与on-policy critic诊断，若要判断替代弃牌收益，须在新训练/开发seed保存可复现的环境分叉并配对rollout；未验证机制前不直接统一改参训练或运行长强度评估。当前无新训练或强度跑批。

执行约束纠正：本轮2k筛选事前估计18–23分钟、实际20分30秒；上轮曾持续等待并查看进度，违反`10-project.md`关于预计>20分钟启动后立即停止实时监控的约定。该规则也适用于小规模训练/诊断/筛选；后续此类批次启动后不再等待或轮询，待用户通知完成再核验。

进一步用已存在的前32训练/前32开发BC采样轨迹比较新候选的贪心动作，命令`.venv/Scripts/python.exe scripts/analyze_phase10_2_12_actions.py --ppo experiments/ppo_v4_tempered/seed42_100k/best.pt --out experiments/ppo_v4_tempered/phase10_2_12_actions_matched_v2 --match-bc-temperature`，结果`.../result.json`。训练/开发choice状态5649/5779，动作一致率96.13%/95.94%；同为打牌但牌张不同208/224次，吃碰相关翻转5/7次。开发立直概率PPO−BC为+0.0150（每局均值），但贪心立直类别翻转仅3次。这些是BC所经历状态上的描述性差异，没有替代动作的反事实回报，不能推断哪张弃牌更好。首次未把BC对照调到T0.5的`phase10_2_12_actions_v1`仅动作argmax计数可用，概率差混入了温度口径，**不得使用其立直概率数值**；已以`matched_v2`重算。
