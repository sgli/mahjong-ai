# Phase10.2.4 完整critic隔离：正式运行预注册

2026-11-13，启动前冻结。配置`configs/train_ppo_v4_full_critic.yaml`，输出`experiments/ppo_v4_full_critic/seed42_100k`，初始化BC-v4 best SHA `814c6eba4e163d126cff7bc27194f11612263c745328abd1b1d9ae94b6716937`。

## 审查与研究边界

确认并修复：旧value_separate_trunk遗漏tile embedding/GRU隔离；feature-v3 collector误删对手九种九牌菜单；固定镜像±0.02门未按10k规模校准。13项定向测试通过，完整编码隔离此前512-transition CUDA pilot成功。

其他可能因素：100k transitions仅约百余整场、reward-v1偏重分数而评估看顺位、GAE lambda0.95的直接长程传递衰减、sampling训练对手池与greedy BC-v4评估对手不同、无BC anchor、无梯度裁剪或target-KL停止。这些因素尚未被证明是失败主因；本次不同时调参。GAE直接权重衰减不能解释为critic完全学不到远期信息，critic bootstrap仍可传递信息。

本轮训练研究变量为完整critic编码器隔离（新增value_separate_encoder=true）；gamma回到10.2.2的0.99，lambda0.95。另有Rule菜单正确性修复，因此与历史10.2.2的比较有混杂，不能声称纯粹识别隔离效果。正式问题是修正后候选能否超过冻结BC-v4。

## 固定训练

seed42，从BC-v4重新初始化，独立embedding/GRU/MLP/value head；10×10000 learner transitions；Adam1e-4、batch512、4 update epochs、clip0.2、vf0.5、entropy0.01；不启用BC anchor/warmup/reward clip。BC-v3/BC-v4/Rule池0.50/0.25/0.25，sampling/T=1，env_seed620000、pool_seed42。固定epoch10，不能按离线或中途对局挑选。

训练完成核验10轮、100k learner transitions、illegal0、有限指标和实际配置，再自动启动评估；异常则停止流水线并写失败记录。

## 固定评估

10,000配对种子680000–689999；candidate对3×BC-v4、mirror BC-v4对3×BC-v4；greedy/T=1，16 workers，座位按game_index%4。PCG64 bootstrap10000次、seed10323。

镜像使用显式normal99-v2：窗口`1.5 ± NormalDist.inv_cdf(.995)*sqrt(1.25/n)`，10k约±0.0288。这是同策略四席选中顺位近似独立均匀时的99%正态近似窗口，不保证任意种子关联下的误拒率。完整有效局、身份、座位、配对和illegal0要求继续保留。镜像/完整性失败记INVALID，停止并诊断，不补样本或放宽窗口；不回溯改判旧实验。

有效评估下GO仍要求paired rank差<=−0.020且95%CI上界<0，否则NO_GO。单seed GO后仍须独立训练seed确认，不直接晋升。

## 时长与执行

旧100k训练约9.5分钟，本轮多一条GRU路径，预计12–20分钟；旧10k/臂评估约92分钟，本轮候选推理也多一条GRU，预计100–160分钟；完整流水线预计约2–3小时，存在硬件负载与实际序列长度不确定性。流水线不设时间上限、不实时监控，用户通知完成后再分析结果。输出状态`experiments/ppo_v4_full_critic/phase10_2_4_run_v1/state.json`，训练/评估stdout和stderr同目录。
