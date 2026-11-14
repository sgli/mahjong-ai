# Phase10.2.13 执行结果

2026-11-14。诊断与门控见`PHASE10.2.13.md`。当前无新PPO训练或强度结论；BC-v4仍为强度基线。

## 既有训练/开发数据的顺位可预测性

命令`.venv/Scripts/python.exe scripts/analyze_phase10_2_13_rank.py`，约3.7秒；只读10.2.9中BC-v4采样训练256局/开发64局及其SHA，按可选状态和每局等权拟合17维公开状态ridge1线性基线，仅训练集拟合系数与常数，开发集按游戏bootstrap 10000次。结果`experiments/ppo_v4_tempered/phase10_2_13_rank_baseline_v1/result.json`。

顺位奖`(2,1,-1,-2)`的开发MSE：常数2.81984、线性2.18783，skill+22.413%，游戏级95%CI[+5.451%,+36.206%]。相同特征对score-only的skill−3.821%、CI[−9.392%,+2.195%]，对当前score+placement总回报为−4.320%、CI[−9.433%,+1.128%]。这支持“可观察公开状态含顺位预测信号”，并指出当前混合回报的简单公开value在这份开发集不稳定；不证明纯rank奖励的actor训练会更强。开发集已用于机制设计，正式训练前若选新目标，需另设独立确认数据与对应critic初始化，不能再把本集当最终验证。

将旧score分量乘0.1的探索性rank-dominant目标在同开发集skill+2.813%、95%CI[−7.245%,+12.830%]，仍不确定；产物`phase10_2_13_rank_baseline_v2/result.json`。据训练集方差事前固定更偏顺位的0.05 score分量供全新128局确认，而不继续用这份开发集调系数。

## 10.2.12候选的新完整对局critic诊断

配置`configs/diagnose_phase10_2_13.json`、新seed1030000–1030191，sampling(T0.5)/greedy各64校准+128保留，共384完整游戏、8 CPU workers、126.724秒，处于2–4分钟事前估时；模型/对手SHA匹配。原始轨迹`experiments/ppo_v4_tempered/phase10_2_13_critic_diagnostic_v1/`；独立命令`.venv/Scripts/python.exe scripts/verify_phase10_2_13_critic.py`复核逐局seed/seat、384文件SHA、终局/合法mask、MC回报和点数账本（最大误差6.11e−6）、游戏级bootstrap，PASS。

保留128局的可选状态：sampling的critic/校准常数MSE为242.759/202.846，skill−19.677%、95%CI[−30.178%,−9.304%]，**FAIL**；greedy为291.196/227.128，skill−28.208%、CI[−39.423%,−16.624%]，**FAIL**。两模式局内value/MC相关均值仅0.0757/−0.0187；仿射校准后skill−0.602%/−0.097%，区间均跨0。采样critic本身已表现不佳，不只是部署greedy切换问题；仿射结果也未显示可简单修复的排序信号。GAE λ0.95→0.97采样符号一致率0.665→0.714、局内方差5.82→8.73，不能据相关性单独选择λ。训练日志约0.58的更新前EV来自自举目标，不能替代独立完整回报校验。

该结果给下一轮的优先级：先验证与新目标匹配的公开critic初始化和独立预测性，再决定是否启动actor；本诊断不证明critic是唯一强度失败根因，也不允许重用本轮保留局调参。

## 新BC-v4完整局对rank-dominant目标的独立确认

配置`configs/collect_phase10_2_13_confirm.json`，seed1040000–1040127、8 CPU workers、BC-v4 sampling对3×greedy BC-v4，128场完整游戏，实际43.875秒（事前估1–2分钟）；`state.json`为complete，全部文件由确认脚本逐个SHA核验。仅用旧10.2.9训练256局拟合ridge1公开状态线性模型与常数，事前固定score权重0.05。命令`.venv/Scripts/python.exe scripts/confirm_phase10_2_13_rank.py`，结果`experiments/ppo_v4_tempered/phase10_2_13_bc_confirm_v1/rank_confirmation.json`。

新确认128局：纯placement线性/常数MSE1.5695/2.4238，skill+35.244%、游戏级95%CI[+27.291%,+41.977%]；`placement+0.05×score`线性/常数MSE3.3217/4.1866，skill+20.659%、CI[+13.564%,+26.696%]。两项事前工程门均PASS；确认集不用于调score缩放或选其他候选。由此允许设计**一次组合调整**：回到T1采样以恢复探索、把score_delta_scale由0.001改为0.00005以令顺位成为主目标、公开value以训练256局拟合的新混合目标系数初始化；其余BC锚、吃碰锚、KL和100k预算保持。这个组合不可分解归因，也不能预先宣称实战提升。

## 组合方案与pilot

正式配置`configs/train_ppo_v4_rank_shaped.yaml`、SHA`07e523f0d376b85e4b7d6a3afd4b6ea373a8c8510617c0d16722b8f8e98cf2a5`，pilot配置SHA`620ee7330da3c9fc378f8f61fffd95506461fb9583d8422b7ebcfa09e0f5f231`。17个value_public系数与独立确认产物中的训练集拟合值逐项一致。`pytest tests/test_public_value_v1.py tests/test_ppo_combined_controls.py tests/test_policy_temperature_v1.py -q`共10项通过。

2×2048 CUDA pilot约47秒完成4096 learner transitions、两轮16次更新、非法0、loss有限；`best.pt`严格加载，`latest.pt`含完整两轮训练史（单轮快照只含当轮历史，不据此误判pilot缺轮）。额外完整真实对局seed1060000共1971环境步/494 learner步、非法0，保存old_logprob与同权重重算最大误差1.20e−5，reward账本相对`0.00005×(最终分−25000)+placement bonus`误差4.44e−16；BC初始化actor逐参数一致。命令`.venv/Scripts/python.exe scripts/verify_phase10_2_13_pilot.py`，产物`experiments/ppo_v4_rank_shaped/pilot_2x2048/collector_verification.json`为PASS。Pilot只证明接口与短程数值健康；允许按冻结配置进入100k正式训练。

为避免后续critic诊断默认使用旧0.001 reward，复制并版本化`diagnose_phase10_2_13_shaped.py`，显式设置环境`reward_score_scale=0.00005`，保留旧诊断脚本及其身份哈希。新诊断器8场pilot在8.235秒完成，`decision=PILOT_ONLY`；8份完整轨迹的MC与`0.00005×(最终分−当前分)+placement`最大误差4.89e−8。此pilot只验证新奖励账本，不能给出critic泛化结论。

正式100k训练于2026-11-14 16:34:10本地启动，PID12548，日志`experiments/ppo_v4_rank_shaped/seed42_100k.stderr.log`；同类10.2.11/12实测约12分18秒/12分25秒，预计12–18分钟。结果待训练自然完成后核验；若超过20分钟立即停止实时监控，不设批次时间上限。

训练约16:46:26完成，实际约12分16秒，处于预计12–18分钟内。`.venv/Scripts/python.exe scripts/verify_phase10_2_13_training.py`独立核验10轮/100k learner transitions、400140环境步、160完整游戏、非法各轮0、loss有限、800 optimizer updates、KL提前停更0轮；best/latest/epoch10逐参数一致。best SHA`51b66d84e261a89b294c5d1dec521f84e50ec016b32ad8b87e007c45f7b34e17`，配置SHA与事前冻结值一致，reward版本、score_scale、T1及公开value初值记录均一致。产物`experiments/ppo_v4_rank_shaped/seed42_100k/training_verification.json`为PASS。训练loss不是强度证据。

固定10.2.9 BC采样前32训练/前32开发局同状态行为：T1下训练/开发greedy一致率93.84%/93.96%，开发吃碰概率PPO−BC每局均值+0.00022，均满足≥90%和≤+0.015行为门；见`experiments/ppo_v4_rank_shaped/phase10_2_13_behavior_v1/result.json`。不以BC轨迹value MSE代表新PPO on-policy critic，进入新seed1050000–1050191完整局诊断。

## 新奖励口径的on-policy critic门

`configs/diagnose_phase10_2_13_shaped.json`、新seed1050000–1050191、sampling(T1)/greedy各64校准+128保留，8 CPU workers、384完整游戏，实际127.601秒，在2–4分钟估时内。`scripts/verify_phase10_2_13_shaped.py`独立复核checkpoint/对手/源码SHA、逐局seed/座位/终局/合法mask/文件SHA、MC与缩放点数账本（最大误差1.53e−7）、游戏级bootstrap，`independent_verification.json`为PASS。

保留128局可选状态：sampling的critic/校准常数MSE为3.08055/3.63098，skill**+15.159%**、95%CI[+5.799%,+24.233%]，PASS；greedy为4.54753/5.15173，skill+11.728%、CI[−2.645%,+24.989%]，INCONCLUSIVE。主采样门通过，贪心部署critic尚无显著改善证据。sampling/greedy局内value/MC相关均值−0.632/−0.758；这里同一局的终局顺位bonus为常数，而已获得分数会从“未来点数差”目标中扣除，故该相关性不能单独解释为动作价值排序错误；需要反事实动作证据。λ0.95的采样GAE符号一致率0.648、局内方差0.0852；λ0.97为0.684/0.1300，不能只据符号一致率提高λ。

按照事前门控，训练身份/数值、固定BC行为、采样critic均通过，允许一次新seed2000对配对强度筛选。此处不宣称强度提升，且不得依据greedy critic区间或局内相关性事后改门。预计18–23分钟（前轮同类1230.473秒）；按项目约束启动后立即停止实时监控，待用户通知自然完成后核验原始逐局、镜像及配对CI。

2000对小批于2026-11-14 16:50:49本地启动，PID19216；候选SHA`51b66d84…7b34e17`、BC-v4镜像与三家对手、seed840000–841999、16 workers、bootstrap10350/10000次、normal99-v2、`--pilot-only`，结果位于`experiments/ppo_v4_rank_shaped/screen_2000pair_seed840000_v1/`。估时18–23分钟，实际总耗时1208.224秒（20分8秒；候选609.334秒、镜像597.546秒）；遵守约束未作实时监控。独立复核命令`.venv/Scripts/python.exe scripts/verify_phase10_2_13_screen.py`：逐局/CSV/种子/座位/checkpoint SHA/bootstrap复算PASS；两臂各2000局完整、非法0，镜像均位1.512，距1.5为0.012，sanity通过。候选均位1.4765，配对差−0.0355，但95% CI[−0.087,+0.0175]上界高于0；故严格小批门`screen_pass=false`。这是有效小批NO_GO，不启动预留seed860000–869999的正式10k评估、不补seed、不调门槛。BC-v4继续为强度基线。结果`experiments/ppo_v4_rank_shaped/screen_2000pair_seed840000_v1/independent_verification.json`。

## 纯顺位与混合奖励的取舍

不能直接比较“placement skill +35.2%”与“placement+0.05×score skill +20.7%”并据数值选纯顺位：skill各自相对于不同尺度、不同噪声的常数预测MSE基线归一化；纯顺位目标更低方差、更容易预测，较高skill不代表训练出的PPO对局更强，也不是两个目标间的成对检验。0.05混合权重的设计理由，是让顺位成为主目标，同时保留稠密点数变化作辅助信用信号，避免纯终局离散回报过稀；但这只是有根据的工程折衷，独立预测门没有证明它优于纯顺位。此次混合候选均位点估计−0.0355、CI跨0，不能区分真实收益、无效或损失，也不能据此认定纯顺位会更好。

因此本轮结论是**不进入正式10k**。若继续比较纯顺位，应另行预注册一个全新候选和未使用seed，沿用同样训练/critic/小批门，与混合方案做同口径对照；不能把已看过的840000–841999用于纯顺位候选或反复改权重。需比较最终rank metric及paired CI，而非跨目标的critic skill。
