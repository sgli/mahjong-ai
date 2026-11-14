# Phase10.2.5：集中归因与组合调整

2026-11-14：按用户要求改为多因素集中检查、一次组合调整后统一训练与一次10k-pair评估。不再为每项局部假设各跑一次强度门；组合通过后再按需消融。

## 已检查的机制及证据

- 奖励记账：8场真实完整游戏、四席累计奖励与点数差×0.001+终局顺位奖一致，最大误差3.55e-15。未在该样本发现漏记或重复；不代表穷尽所有规则分支。保留reward-v1记账，不加入人为副露惩罚。
- 轨迹边界：当前opponent-pool路径在完整终局将learner最后一步标done；预算截断推进到下一learner决策再bootstrap；GAE及终止/截断回归测试通过。未发现本轮路径新的边界错误。
- 时间尺度：平均数百次learner决策导致gamma0.99折扣远期点数/顺位奖励。组合改gamma1.0保留远期目标；lambda仍0.95，critic仍承担长期估计。旧gamma1.0实验没有证明提升，本项是设计选择。
- 同状态行为：诊断共4755条决策，碰机会93次，BC/PPO概率33.325%/51.228%；吃机会195次，概率13.107%/23.281%。说明这批状态下候选更倾向副露，不只是机会数量变化。立直仅26个机会，PPO概率反而更高，不能从整场立直次数下降直接推断其压制立直。
- 强制动作：69.632%状态只有一个合法动作。旧actor目标和advantage归一化包含这些状态，影响有效决策信号的尺度。新版本保留全部状态训练critic，仅以多合法动作状态归一化advantage并平均policy/entropy/anchor项；这是优化口径调整，不是旧PPO公式必然错误的证明。
- 局部梯度：BC采集的前512个可选状态（103次吃/碰机会）上，policy梯度范数2.20249、entropy系数0.01时0.02066；对副露概率的负梯度方向分别−0.51734/+0.00906。entropy在本样本局部推动副露，但较小且policy相反，不能认定entropy为主要根因；这不是Adam完整更新模拟。
- critic质量：旧PPO四条诊断轨迹的GAE目标解释方差约0.032–0.240，长程MC目标解释方差多数接近0或为负。样本很小，GAE目标本身依赖critic；仅支持增加预热和观测，不证明某个固定预热预算足够。
- 训练分布：旧混合采样对手与greedy BC-v4评估不同。组合先对齐到三个greedy BC-v4，learner仍sampling；会缩小泛化覆盖，结果只支持该固定对手协议。

证据：`experiments/ppo_v4_combined/diagnostics/mechanisms_8games.json`（33.24秒）与`mechanisms_8games_gradients.json`（35.72秒）；脚本`scripts/audit_ppo_mechanisms.py`。混合状态样本是描述性诊断，动作分组raw advantage不能作为反事实动作收益比较。

## 实现修复与控制

1. 修正warmup=True仍将BC anchor加入loss的问题；预热只训练value。
2. 修正完整分离critic预热时只训练value_head的问题；现在所有value_模块可训练，整个actor冻结。这两项上一轮未启用，不是10.2.4失败的直接原因。
3. 新增显式warmup_epochs，避免把environment global_step误当learner预算；旧warmup_steps仍为环境步口径。
4. decision_only默认false，新组合true；critic保留全部样本。每次update pass重洗牌。
5. max_grad_norm=0.5，actor/critic参数组分别裁剪；非有限loss/梯度报错，不让critic大梯度压缩actor步长。
6. target_kl=0.01：每minibatch更新前对有选择样本计算非负k3估计mean(exp(logratio)-1-logratio)，超限停止本epoch剩余更新。它是随机估计，不是严格全分布KL上限；不回滚已发生更新，也不替代BC anchor。
7. 开启BC anchor beta0.1，关闭额外entropy奖励，lr由1e-4减为5e-5；这些数值是冻结的设计选择，尚无强度提升证据。

## 验证结果

定向测试两组共63 passed/3 skipped，补充KL停止集成测试通过（独立新测试1项）；涉及warmup、anchor、GAE、奖励守恒/结算、更新、resume、checkpoint和菜单。组合pilot为2×2048 learner transitions，环境步16391，illegal0，约40秒含1场self-play smoke。

pilot checkpoint逐参数核对：epoch1 actor与BC-v4完全一致；epoch2 actor发生更新；两轮各16次optimizer更新。epoch2 rollout value解释方差−0.06493，不能声称critic已经成熟。证据`experiments/ppo_v4_combined/pilot_2x2048/control_verification.json`。pilot只验证链路与控制，不做强度结论。

## 统一方案与正式评估预注册

- 配置：`configs/train_ppo_v4_combined.yaml`；SHA256：a99dcedb7e8d8bb41eb74b868a04e4acc5f7a7ac766654a92fd00c13a40cfdb4。
- 保留feature-v3/action-v2、完整critic隔离、冻结BC-v4初始化；不重建Parquet、不重训BC、不续训失败PPO。
- seed42，11×10000 learner transitions：首轮10000只预热critic，后10轮共100000 actor筛选预算。epoch11为唯一候选，不按test/gold top-1/强度挑选checkpoint。
- gamma1.0/lambda0.95、batch512、4 update passes、clip0.2、vf0.5；其余变更见上文与配置。对手BC-v4权重1、greedy/T1；learner采样。
- 训练健康门：精确预算、finite loss、illegal0、预热actor逐位不变、两个隔离选项启用、配置身份未改变、best与latest epoch11权重一致。失败即停，不自动评估。
- 强度：candidate PPO vs3×BC-v4，mirror BC-v4 vs3×BC-v4，均greedy/T1；每臂10000游戏，配对种子700000–709999，seat=index%4；16 workers；bootstrap10000、PCG64 seed10325。
- GO：配对平均顺位差<=−0.020且95%CI上界<0；mirror normal99-v2容差z(.995)*sqrt(1.25/10000)=0.02879865，完整有效且illegal0。有效未过为NO_GO，完整性/镜像失败为INVALID；不补样本、不事后调门槛。
- 新训练估计12–25分钟：以完整隔离100k实测10分53秒、加一轮预热/anchor与额外KL前向估计；小pilot约40秒仅交叉检查。10k配对评估参考上轮100分9秒，预计90–120分钟；整条流水线预计约1小时45分–2小时30分。不设时间上限、不实时监控。
- 启动入口：`.venv/Scripts/python.exe scripts/run_phase10_2_5.py`，自动训练、健康门、一次正式评估。路径`experiments/ppo_v4_combined/phase10_2_5_run_v1/state.json`。

组合通过时只能认定整套方案有效；不将提升归给某一项。保持BC-v4为强度基线直至正式门通过。
