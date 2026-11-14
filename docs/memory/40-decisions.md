# 决策记录

- 2026-11-12：用户指定 BC-v3 为后续强度基线，并要求项目从 Phase 10 重新开始。故 BC-v2.1 等其他模型跑批、结果和配置清理；仅保留 BC-v3 与新 BC-v4 所需的 decision-v3。来源：本会话用户明确指令；实施状态见 `docs/PHASE10.md`。
- 2026-11-12：BC-v4 输入使用 feature-v3/action-v2；三家有序牌河由共享 tile embedding + GRU 编码；随机初始化。根据现存 Parquet `discards` 原列派生，不重建数据。依据：`docs/PHASE10.md` 的设计与现有 schema `src/mahjong/dataset/schema.py`。

- 2026-11-12：清理 `configs/experiments/`、旧 benchmark YAML、self-play 输出和仅依赖旧 phase/checkpoint 的脚本；保留 `tests/test_ppo_v3_entry.py`，移除直接要求已删除 PPO 配置的两条测试。通用训练/评估/self-play/PPO CLI 改为必须显式传 `--config`。依据：文件引用检查；未运行测试。
- 2026-11-12：用户明确 STATUS.md 用于阶段步骤完成后的总结；新 Phase 10 分为 10.1 BC-v4 baseline 与 10.2 PPO，10.2 仅在 10.1 强度门通过后启动。当前 10.1 尚未开始正确性检查或训练。依据：`docs/STATUS.md`、用户本轮指令。
- 2026-11-12：为 BC-v4 DataLoader 训练期指标保留 GPU tensor 累计、仅在日志/epoch 汇总时转标量，避免每 batch 多次 `.item()` 同步；静态 batch A/B 快 13.7%，完整配置 150 秒只快约 2.6%，故作为低风险优化保留但仍需完整 epoch 汇总验证。依据：`src/mahjong/training/bc.py:86-116,449-457`；`experiments/bc_v4/metrics_sync_ab.py` 与 `async_metrics_150s/telemetry.json`。
- 2026-11-12：将 BC-v4 `metric_interval_steps` 设为1000：loss仍逐 batch 精确累计，top-k/非法率只在采样 batch 计算；full-config 150秒由 step13000推进到step14000。batch size维持2048（静态 sweep吞吐最高）；每步 finite-loss 检查维持原样（更稀疏检查未改善整链路）。依据：`configs/train_bc_v4.yaml`、`experiments/bc_v4/sampled_metrics_150s/`、`batch_size_sweep.py`、`sampled_checks_150s/`。
- 2026-11-12：预注册 BC-v4 vs BC-v3 实际强度门：20,000 个配对种子600000–619999；候选 BC-v4 vs 3×BC-v3，镜像 BC-v3 vs 3×BC-v3；同seed/seat、Greedy/T=1、16 workers。主效应为逐seed候选顺位差均值，PCG64 bootstrap 10k、seed10319；GO仍需均值<=−0.020、95%CI上界<0、镜像均位1.5±0.02、全局合法且无失败局。依据：`docs/PHASE10.1.1_STRENGTH_PREREG.md`；历史 20k 配对耗时6069s。
- 2026-11-13：Phase 10.1.1 强度门判定 GO；BC-v4 可作为 Phase 10.2 PPO 初始化策略。PPO 不因该门自动启动，仍须先单独冻结训练预算、评估种子、对手池和强度门。依据：`experiments/bc_v4/strength_20k_seed600000_v1/result.json`、`docs/PHASE10.1.1_RESULT.md`。本决定仅支持本次预注册对手与评估条件下的阶段推进。
- 2026-11-13：冻结 Phase 10.2.1 单变量 PPO 筛选：BC-v4 init，100,000 learner transitions，seed42；对手池 BC-v3/BC-v4/Rule-v1 = 0.50/0.25/0.25；BC anchor 与 warmup 均关闭。评估为与镜像 BC-v4 做20,000配对种子620000–639999对局；GO后仍需独立 seed 确认。依据：`docs/PHASE10.2/PHASE10.2.1_PREREG.md` 与 `configs/train_ppo_v4.yaml`。
- 2026-11-13：修正 PPO opponent-pool rollout 的已知低效和配置缺口：同一 run 内缓存冻结 opponent 实例，支持单独提供 BC-v3 与 BC-v4 checkpoint，避免把初始化策略误当 BC-v3 opponent。依据：`src/mahjong/training/ppo.py:369-422`、`scripts/train_ppo.py`、`src/mahjong/training/opponent_pool.py`；尚待 bounded pilot 实际验证。
- 2026-11-13：Phase10.2.1 PPO seed42/100k 的20k-pair强度门 NO-GO；pair delta +0.54450、CI全正。按冻结方案停止此配置，不追加样本、不改阈值、不跑第二 seed；BC-v4保留为当前强度基线。依据：`experiments/ppo_v4/strength_20k_seed620000_v1/result.json` 及 `docs/PHASE10.2/PHASE10.2.1_RESULT.md`。
- 2026-11-13：用户将新 Phase10.2.2 PPO paired strength evaluation 从20k改为10k配对；种子固定640000–649999，原效应阈值与 mirror sanity 不变，并在开跑前更新 prereg。预估90–120分钟，样本量减半会降低区间精度。依据：用户本轮指令、`docs/PHASE10.2/PHASE10.2.2_PREREG.md`。
- 2026-11-13：Phase10.2.2 critic trunk隔离在validation上减轻策略漂移，但10k强度门NO-GO（差+0.0844，CI全正）；保持BC-v4基线，停止该机制、不追加样本。下一假设须单独预注册，候选为在隔离critic基础上添加BC KL anchor。依据：paired result、`docs/PHASE10.2/PHASE10.2.2_RESULT.md`。
- 2026-11-13：进一步审查后将BC KL anchor候选后移，先做Phase10.2.3单变量gamma0.99→1.0；保持独立critic及其他训练配置、10k paired强度门。依据：`src/mahjong/training/ppo.py:56-70,173-184`、`src/mahjong/environment/env.py:355-361,66-67`、10.2.2 checkpoint的100k learner steps/163条轨迹；方案见`docs/PHASE10.2/PHASE10.2.3_PREREG.md`。
- 2026-11-13：Phase10.2.3镜像sanity失败（1.4730对预注册1.5±0.02），故本轮正式门无效、不得晋升gamma1.0模型；不事后修改阈值或补样本。候选均位1.5744与paired差+0.1014提示偏弱，只作探索性证据。下一项实验前先重新预注册镜像有效性口径。依据：`experiments/ppo_v4_gamma1/strength_10k_seed660000_v1/result.json`、CSV独立复算。
- 2026-11-13：修正旧critic隔离归因：value_separate_trunk仅分离MLP，GRU/embedding仍共享。新增显式value_separate_encoder完整隔离，默认false维持旧checkpoint语义；下一实验以10.2.2的gamma0.99为对照仅新增该选项，待镜像口径预注册后训练。证据src/mahjong/model/policy.py、src/mahjong/training/ppo.py、tests/test_river_critic_isolation.py。
- 2026-11-13：新评估显式使用normal99-v2镜像窗口z(.995)*sqrt(1.25/n)，10k约±0.0288；无效单列INVALID。该校准依赖近似独立均匀顺位假设，旧fixed-v1结果不改判。修复feature-v3 Rule对手菜单后，历史对比存在混杂，不能独立归因全部提升于critic隔离。方案docs/PHASE10.2/PHASE10.2.4_PREREG.md。

- 2026-11-14：Phase10.2.4有效NO-GO，保留完整critic隔离与菜单正确性修复，BC-v4继续作为强度基线；不扩展本配方、不晋升候选。下一步先做有界动作级归因，再冻结新实验。旧部分隔离与本轮种子/菜单不同，不将数值差缩小解释为已证实改善。证据docs/PHASE10.2/PHASE10.2.4_RESULT.md及对应result.json。

- 2026-11-14：按用户要求以组合方案替代逐项单变量长评估。保留完整critic隔离，修正warmup anchor/冻结；启用decision_only、1轮critic预热、BC anchor0.1、entropy0、lr5e-5、分组梯度裁剪0.5、target KL0.01、gamma1.0；训练对齐greedy BC-v4。这些是组合设计选择，非已证明根因；不重建数据/重训BC。方案与代价见docs/PHASE10.2/PHASE10.2.5.md；回滚只需使用旧配置（新控制默认关闭），历史产物保持不变。每update pass重洗牌属于通用更新行为调整。

- 2026-11-14：10.2.5未过预注册门，不晋升、不补样本、不改阈值；保留组合修复和稳定性控制，BC-v4仍为强度基线。下一步集中检查critic独立泛化和回报目标，再决定critic预训练/actor预算组合，不再逐项长评估。CI跨0不能宣称超过BC或等效；本轮未启动新训练。证据docs/PHASE10.2/PHASE10.2.5_RESULT.md。

- 2026-11-14：按用户要求修改10.2.5诊断报告并执行10.2.6：critic主验证采用与训练匹配的sampling，greedy单列；目标为reward-v1剩余回报，非直接rank。128校准/256保留seed×2模式，共768游戏；游戏级主MSE skill三态门，离线复用GAE比较；不据lambda1恒等式选参，不恢复逐项长评估。方案docs/PHASE10.2/PHASE10.2.6.md。

- 2026-11-14：10.2.6执行INCONCLUSIVE分支，不追加样本直到显著、不扩大actor、不直接删anchor或改lambda1。主critic尚未证实超过常数，仿射校准也未显示充分收益；下一步优先设计与BC初始化匹配的冻结BC-v4 actor/完整回报critic学习对照，另用训练开发验证划分，不用本轮保留集调参。依据docs/PHASE10.2/PHASE10.2.6_RESULT.md。

- 2026-11-14：10.2.7测试门INCONCLUSIVE，BC-v4继续为强度基线；不以训练MSE下降直接扩critic预热/actor预算，也不重复使用最终测试集调参。下一步先在训练/开发游戏分析目标可预测性及口径差异，再制定一次组合调整。证据docs/PHASE10.2/PHASE10.2.7_RESULT.md。

- 2026-11-14：公开状态线性基线在新128场确认集上优于常数和旧critic，故下一步优先研究critic显式基线+残差及训练/评估状态口径；不将其直接接入PPO或宣称强度改善。10.2.8确认集已用于结果核验并封存，改模型需新的划分。依据docs/PHASE10.2/PHASE10.2.8_RESULT.md。

- 2026-11-14：10.2.9残差网络未显示额外样本外收益，不直接接入PPO或增加actor训练；下一步考察版本化公开线性value的GAE接口与跨局稳健性。开发集线性模型不佳，不能仅凭新测试正向结果宣称稳定泛化；新模型需新测试。依据docs/PHASE10.2/PHASE10.2.9_RESULT.md。
- 2026-11-14：10.2.10采用可训练的16维公开局面线性value，以10.2.9仅训练划分拟合的系数初始化；BC-v4 actor表示保持原样。新128对安全门通过后，冻结seed42/100k训练与一次新seed10k对正式强度门；仅用epoch10，不按pilot或离线top1挑checkpoint。理由为10.2.8/10.2.9公开基线有样本外预测信号而神经残差无稳定增益；不据此断言实战提升。回滚可使用旧配置/BC-v4 checkpoint，不改历史结果。依据docs/PHASE10.2/PHASE10.2.10.md、docs/PHASE10.2/PHASE10.2.10_RESULT.md。
- 2026-11-14：10.2.10有效NO_GO后停止本公开value PPO配方，不补seed或事后调整GO门；BC-v4继续是强度锚点。公开value离线预测信息未转化为顺位收益，现有汇总显示副露偏高但不证明因果。下一步仅在现有轨迹中有界分析动作分歧/优势方向，再决定能否预注册新机制；新候选必须使用新评估seed。依据docs/PHASE10.2/PHASE10.2.10_RESULT.md及strength_10k_seed720000_v1/independent_verification.json。
- 2026-11-14：10.2.11对探索性吃碰概率偏移设单变量beta1.0定向BC Bernoulli KL，原全动作anchor和其余10.2.10训练设置保持；不是已证实失败根因。先pilot/100k健康/同状态行为门，再1000对新seed资源筛选；全部通过才启动另一个新seed10k正式强度，GO门不变。小批均差−0.053但CI跨0且镜像接近边界，故只支持正式评估而非晋升。回滚关闭call_anchor_beta或使用10.2.10配置，旧产物保留。依据docs/PHASE10.2/PHASE10.2.11.md、docs/PHASE10.2/PHASE10.2.11_RESULT.md。
- 2026-11-14：10.2.11正式10k强度门有效NO_GO，不晋升PPO、不续训本配方、不补本轮seed或改变GO门；BC-v4保持强度基线。定向吃碰约束将对局副露差降至+0.0584/局，但未证明顺位提升；跨10.2.10/10.2.11不同seed差异不能作干预净因果效应。下一步只在训练/开发轨迹上先核非吃碰分歧及优势/终局回报对齐，再定义新方案。依据docs/PHASE10.2/PHASE10.2.11_RESULT.md和strength_10k_seed780000_v1/independent_verification.json。
- 2026-11-14：10.2.12在同seed完整轨迹证实T1 sampling与greedy部署回报差大，固定策略T0.5收窄分布差且通过预设工程门后，冻结单变量`model.policy_temperature=0.5`候选；其余10.2.11设置不动。模型前向统一缩放策略logits，rollout old_logprob、PPO ratio、BC anchor使用同温度；不能直接用贪心行为配原softmax比率。先13项测试、pilot、100k健康、行为门，再新seed2000对严格小批（均差≤−.020、CI上界<0、镜像距1.5≤.05），只通过才启动独立seed10k正式门。回滚用旧配置T1，旧checkpoint保持。证据docs/PHASE10.2/PHASE10.2.12.md及docs/PHASE10.2/PHASE10.2.12_RESULT.md。
- 2026-11-14：10.2.12有效2000对小批筛选NO_GO，不启动预留seed820000–829999的10k/臂正式强度评估，不补seed或改T/KL/门槛。T0.5冻结旧策略采样回报改善未成为新模型greedy强度证据；现有候选均差+0.005、CI跨0。保留BC-v4强度基线；下一步限于已有训练/开发轨迹的弃牌排序与回报方向离线审查，不复用封存测试或强度seed，若无可证伪机制则不训练。依据`docs/PHASE10.2/PHASE10.2.12.md`、`docs/PHASE10.2/PHASE10.2.12_RESULT.md`及小批独立核验；回滚方案为继续使用BC-v4 checkpoint，温度接口默认T1。
- 2026-11-14：遵循`docs/memory/10-project.md`的集中排查约束，10.2.12小批NO_GO后不立即提出或训练下一单变量候选；先把低熵/KL早停、reward-v1点数与顺位信号、sampling critic/greedy部署、弃牌排序翻转一起核证，才决定是否有可证伪的统一调整。现有离线trace只支持“顺位直接信号较弱”，不证明奖励bug或失败根因；不复用封存test或强度seed选机制。证据`docs/PHASE10.2/PHASE10.2.12_RESULT.md`及`phase10_2_12_followup_v1/result.json`。
- 2026-11-14：10.2.13集中归因后选择一次rank-dominant组合候选：10.2.12新critic独立sampling/greedy均FAIL；新BC确认中0.05score+rank公开状态基线skill+20.659%且CI全正。组合恢复T1探索、score_scale从0.001改0.00005、用只在旧训练256局拟合的公开value系数初始化，保留lambda0.95、KL0.01、吃碰/BC锚及100k预算。先pilot/训练健康/新seedcritic门，再新seed2000对小批，只有全过才正式10k；不能拆分归因或复用旧强度seed。回滚用BC-v4 checkpoint和10.2.12配置；依据`docs/PHASE10.2/PHASE10.2.13.md`及`docs/PHASE10.2/PHASE10.2.13_RESULT.md`。

- 2026-11-14：10.2.13 rank-dominant 2000对小批数据有效但强度筛选NO_GO；不进入正式10k、不续训/补seed或调整门槛，BC-v4继续作为强度基线。若比较纯顺位与0.05混合目标，必须另行预注册、使用新训练与评估seed并比较最终配对顺位效果；critic skill跨目标不可直接选优。依据`docs/PHASE10.2/PHASE10.2.13_RESULT.md`与screen独立核验。

- 2026-11-14：因10.2.13混合目标小批有效NO_GO，按用户同意启动Phase10.2.14纯顺位/混合目标同seed对照；两个目标从BC-v4重新初始化并匹配critic，禁止直接对比不同目标的critic skill。胜负以新seed配对顺位、CI及各自BC镜像门判定；未分胜负则不挑点估计赢家、不扩样/改门。预注册`docs/PHASE10.2/PHASE10.2.14.md`。

- 2026-11-14：10.2.14两目标比较结果有效但不分胜负：纯顺位点估计优于混合0.033 rank，95%CI[−0.0865,+0.021]跨0；两臂相对BC-v4镜像的小批CI均跨0。两候选都NO_GO，不运行10k、不改阈值/扩本评估seed，BC-v4继续基线。是否做独立训练seed确认留待单独决定；同时后续PPO并行度须先有界校准。依据`docs/PHASE10.2/PHASE10.2.14_RESULT.md`。

- 2026-11-14：用户要求针对纯顺位checkpoint的CI继续推进，冻结新Phase10.2.15独立10k评估seed890000–899999，维持原GO门，不将其与已查看的2k数据合并作显著性判断。此操作只降低模型强度估计的抽样误差，不改变模型。结果需通过配对CI、镜像、完整性和非法门。依据`docs/PHASE10.2/PHASE10.2.15.md`。

- 2026-11-14：Phase10.2.15纯顺位seed43独立10k有效NO_GO，不对本批补seed/改门、不晋升；BC-v4保持基线。较小批点估计−0.037在新10k为−0.0167且CI跨0，不能以旧批合并后的表面精度推翻独立门。依据`docs/PHASE10.2/PHASE10.2.15_RESULT.md`。

- 2026-11-14：纯顺位seed43独立10k仍未过门后，选择只调整GAE λ=.95→1.0以减少长回合终局名次奖励的GAE衰减；每座位轨迹140–168步时纯终局残差权重约0.00076–0.00018。训练目标/BC锚/KL/网络保持，先pilot、100k和新seed2k screen；若screen不过则NO_GO。依据`docs/PHASE10.2/PHASE10.2.16.md`。

- 2026-11-14：纯顺位λ=1.0 seed44通过训练健康但新seed2k筛选NO_GO（delta+0.0185、CI跨0），不进10k。训练优势std高、EV低的比较受训练seed混杂，只作线索；下一次预注册λ=.99折衷，所有其他超参数相同。依据`docs/PHASE10.2/PHASE10.2.16_RESULT.md`。

- 2026-11-14：λ=1纯顺位候选NO_GO后，按约140–168本座位动作的终局残差衰减与MC方差线索，冻结λ=.99为下一可证伪筛选；其他参数不变，必须pilot/100k健康/新seed2k GO才讨论正式评估。Phase10.2.17已启动PID17760。

- 2026-11-14：用户重申10-project约束后，终止单变量λ=.99流水线（epoch5/10），不把部分run用于强度结论。后续先综合诊断纯顺位相关的奖励传播、critic/优势方差、actor/BC锚约束、训练/评估随机性，再冻结一次组合方案；未完成前不启动新实验。

- 2026-11-14：按Phase10.2.18多因素审查结果，不将BC/call anchor、critic架构或target-KL一并修改，因为现有数据没有证明它们是限制因素；独立critic相对常数基线通过、anchor行为分歧小且无反事实效应、KL early-stop很少。仅GAE .99得到固定轨迹符号一致率与方差折衷证据，故以全新seed46重做单一、可证伪筛选；不是禁止组合实验，也不是逐点λ扫参。配置/门槛见`docs/PHASE10.2/PHASE10.2.19.md`，依据`docs/PHASE10.2/PHASE10.2.18_RESULT.md`。
- 2026-11-14：Phase10.2.19 seed46通过正式10k配对顺位门（delta−0.0289，95% CI[−0.0519,−0.0055]，镜像/非法/完整性PASS）。该单训练seed结果不自动晋级；下一步用新训练seed和新评估seed做独立确认，确认前BC-v4仍是强度基线，且不以本轮结果继续调参。依据`docs/PHASE10.2/PHASE10.2.19_RESULT.md`及`docs/memory/10-project.md`晋级约束。
