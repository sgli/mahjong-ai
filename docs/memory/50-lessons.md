# 踩坑与规则

- 2026-11-12：feature-v2 的 BC-v3 checkpoint 与 feature-v3 输入形状/语义不同，不能直接作为 BC-v4 权重初始化。BC-v4 从零开始；先做 encoder/model 形状与 mask 正确性验证，再启动全量训练。证据：旧 checkpoint metadata `experiments/bc_v3/checkpoints/config.yaml`；新配置 `configs/train_bc_v4.yaml`。
- 2026-11-12：数据集版本与特征版本分离；decision-v3 Parquet 保留 ordered discard arrays，新特征可运行时派生。不要为了 feature-v3 重建完整 decision-v3。

- 2026-11-12：删除旧配置时同步检查 CLI 默认路径和测试夹具；旧 `benchmark.yaml`、`train_bc.yaml`、`train_ppo_v3.yaml`、`selfplay.yaml` 被删除后，相关入口改为显式要求 `--config`，引用已清理配置的两条 PPO 配置测试移除。证据：`scripts/benchmark.py`、`scripts/train_bc.py`、`scripts/train_ppo.py`、`scripts/selfplay.py`；测试未运行。
- 2026-11-12：BC-v4 有界 CPU pilot 的 16 workers/prefetch 4 运行中，整机 CPU 采样峰值约 52.6%，但不持续；整段均值 16.2% 被验证阶段和 DataLoader 重启开销稀释。训练 profile 显示 data_time 2.297/11.633s，吞吐 161,330 rows/s。不要用整段均值推断训练区间占用；Windows worker 进程在训练/验证间退出重建，简单累加 `Get-Process` CPU 秒会因进程集合变化失真。证据：`experiments/bc_v4/cpu_monitor_pilot_2/cpu_summary.json`、`stderr.log`。
- 2026-11-12：完整 BC-v4 配置持续训练 150 秒时，系统 CPU 仍只在 30.5% 中位数、36.2% 峰值，未到 95%；因此低占用不是小 shard 样本或 validation 阶段单独造成。当前证据确认资源没有吃满，但不能区分供数速度、GPU/批处理节奏或 worker 分配；下一步如需定位，需同步采集阶段性 data_time/GPU 利用率与每个 worker CPU，而不是只加 workers。证据：`experiments/bc_v4/cpu_monitor_150s/cpu_summary.json`、`stderr.log`。
- 2026-11-12：正式配置监控表明低 CPU 不是 worker 数设置错误：16 worker 稳态约 40% 单核占用，GPU 约 62% 利用；feature-v3 从 14,605 维 one-hot 缩到 781 维（其中对手河 14,208 one-hot 位变 384 token），降低 CPU/IPC 负担，且 BC-v4 试跑步速高于 BC-v3 的 141.8k rows/s。CPU 不满不代表训练更慢；不要以 95% 占用作为目标指标。证据：BC-v3 `metrics.json`，BC-v4 `cpu_worker_gpu_150s/telemetry.json` 与训练日志；速度对比仅为部分 run 推断。
- 2026-11-12：`compute_metrics` 原先每个 batch 将 6 个 CUDA 标量分别 `.item()`，训练循环另用 `bool(torch.isfinite(loss))` 同步；静态 batch A/B 指标路径有 13.7% 步时开销。将指标改为 detached tensor 累加后，全链路 CPU +4.5 个百分点、step 0→13000 仅快约 2.6%，说明逐批同步是次要贡献，不是低 CPU 的主因。仍需完整 epoch 验证数值汇总；未运行测试套件。证据：`src/mahjong/training/bc.py:86-116, 450-457`、`experiments/bc_v4/metrics_sync_ab.py`、两轮 150s telemetry。
- 2026-11-12：补充用户资源偏好：目标仍是缩短训练总时间，用户要求尽量把 CPU/内存推近 95%；本轮结果只说明当前配置没有饱和，不能把“95%无必要”记录成用户改了偏好。之后应以吞吐/总时长为主结果，并解释任何未达到目标的瓶颈。
- 2026-11-12：feature-v3 pack lengths GPU→CPU 虽是同步点，但静态 A/B 的 CPU预计算长度路径慢5.5%，因此不要因代码表面存在 `.to('cpu')` 就直接假定它是瓶颈；要用同批 A/B保留真实模型结果。
- 2026-11-12：训练分类诊断指标每1000步抽样后，150秒全配置多跑到step14000，loss仍精确，验证仍按原样；小 pilot 只有917步，只采一批，样本准确率有明显启动偏差。必须把 `metrics_sampled_samples/fraction` 和 exact loss 分开报告。
- 2026-11-12：[评估器实测] 每 batch 将六个CUDA指标 `.item()` 同步并逐行生成非必要分组统计，会拖慢BC全量评估；关闭分组统计并将指标留在GPU累计后，5M test pilot活跃阶段CPU由此前两分钟约35.8%观测提升到约85.3%整机/79.6% Python组。该pilot只证明CPU供给明显改善，完整准确率仍以全量test报告为准。证据：`scripts/evaluate_bc.py:92-185`、`experiments/bc_v4/eval_cpu_monitor_2m.json`、`experiments/bc_v4/eval_cpu_pilot_5m_monitor.telemetry.json`。
- 2026-11-12：[时长校准] 优化后5M test pilot约44秒，按行数线性外推全量曾估18–22分钟；实际134.57M full test约6.5分钟。短窗pilot的worker启动成本和阶段速率会使全量线性估时严重偏差；以后优先用完整同类评估耗时校准。
- 2026-11-12：静态增大 batch 到4096/8192反而降低 rows/s并增显存；另将 finite check间隔拉长虽静态微快，但全配置没有多推进step，故维持batch2048及每步有限值检查。不能把局部 kernel/step提升直接外推到完整 DataLoader训练。
- 2026-11-13：[时长校准] 20,000配对强度跑批原估90–130分钟（依据相近历史跑批6,069秒），实际总耗时24,073.529秒（约6小时41分），其中两arm分别10,912.281秒和13,158.837秒。该历史值不足以预测本次耗时；未来同 runner/硬件估时优先采用本次同规模结果，若环境或代码不同则明确估时区间的不确定性。证据：`experiments/bc_v4/strength_20k_seed600000_v1/result.json` 的估时、分臂耗时和总耗时字段。
- 2026-11-13：[PPO代码审查] opponent-pool 模式每局都构造 PolicyOpponent，导致同一 BC checkpoint/model 按游戏数重复加载；原配置也只有一个 `bc_checkpoint`，不能同时区分 BC-v3 历史对手和 BC-v4 初始化策略。当前改为每 run 缓存冻结 opponent 实例，并支持 checkpoint map 的 `bc-v3`/`bc-v4` 分离。证据：`src/mahjong/training/ppo.py:369-422`、`scripts/train_ppo.py`、`src/mahjong/training/opponent_pool.py`；bounded pilot 待验证。
- 2026-11-13：[PPO时长校准] 512 learner-transition pilot约10秒，线性外推100k时估25–45分钟；正式10×10k PPO run 实际572.8秒（9分33秒），约175 learner transitions/s。小 pilot 固定启动/单游戏成本占比过大，不适合直接按 transition 数比例外推；后续以正式同类 run 估时，pilot只确认流程和下界吞吐。证据：`experiments/ppo_v4/pilot_512_seed42/`、`experiments/ppo_v4/seed42_100k/result.json`。
- 2026-11-13：[PPO强度评估耗时] 20k/arm paired PPO-vs-BC-v4评估原估6–8h，实际11,161.422s（约3h06m）；candidate/mirror分别5,685.246s/5,473.755s。此前BC-v4-vs-BC-v3 20k/arm为24,073.529s，未能直接外推到新候选/对手同 feature-v3 的评估。今后用最近同 schema、同评估 runner 的实测校准。证据：`experiments/ppo_v4/strength_20k_seed620000_v1/result.json`、旧BC run `experiments/bc_v4/strength_20k_seed600000_v1/result.json`。
- 2026-11-13：[PPO漂移归因] 第一轮验证集诊断显示共享 actor/critic trunk 下10轮后 BC→PPO KL0.401、greedy agreement67.1%；启用独立 value trunk 的同预算 run降至KL0.0743、agreement85.6%。这支持critic梯度影响 actor 表征的归因，但 actor policy/entropy 更新仍造成可见漂移；离线指标不能证明强度。证据：两份 `policy_drift_by_epoch_validation_100k.json`、`docs/PHASE10.2.2_RESULT.md`。
- 2026-11-13：[PPO强度归因] critic trunk隔离明显降低validation漂移，但10k对仍显著输给BC-v4（paired差+0.0844，CI全正）；候选副露更多、放铳增加、立直减少。critic共享是策略漂移的重要来源，但不是强度失败的完整解释。10k/臂耗时5613.693s；按同runner实测，不使用简单总局比例忽略启动和两臂差异。证据：`experiments/ppo_v4_critic_isolated/strength_10k_seed640000_v1/result.json`。
- 2026-11-13：[回报时间尺度] reward-v1立直立即−1、终局顺位奖±1/±2；10.2.2 checkpoint的100k learner transitions/163条learner轨迹≈613.5步/轨迹。gamma0.99对613步终局奖励折扣约0.0021，GAE直接传播(.99×.95)^613近零。说明顺位目标长期信用分配存在设计风险，尚非实证因果结论。证据：`src/mahjong/environment/env.py:66-67,355-361`、`src/mahjong/training/ppo.py:56-70,173-184`、`experiments/ppo_v4_critic_isolated/seed42_100k/latest.pt`的metrics_history。
- 2026-11-13：[10k镜像门误拒风险] Phase10.2.3 10k镜像均位1.473、四席均位1.464/1.4876/1.4724/1.468，checkpoint/配对/合法性均正常；固定1.5±0.02窗口失败。若被选rank近似独立均匀，10k下标准误约0.0112，±0.02仅约1.79个标准误，双侧误拒约7%；实际偏移可能只是抽样波动，也不能排除种子关联。未来样本量变动时先校准sanity门，不能事后修改本轮阈值。证据：`experiments/ppo_v4_gamma1/strength_10k_seed660000_v1/result.json`与原始镜像局。
- 2026-11-13：[完整梯度隔离] 旧MLP隔离测试不能覆盖feature-v3前置序列编码：value-only反传实测actor embedding/GRU非零梯度。完整隔离需要单独embedding/GRU副本、BC初始化复制及value-only优化后actor logits不变的回归测试；已修复并通过。此前10.2.2/10.2.3只能解释为部分隔离。证据tests/test_river_critic_isolation.py、docs/PHASE10.2.4_RESULT.md。
- 2026-11-13：collector仅识别feature-v2时开放对手九种九牌，遗漏feature-v3，导致Rule训练菜单与自身语义不一致；已修复且fake-env回归覆盖。证据src/mahjong/training/opponent_pool.py、tests/test_phase10_2_4_gates.py。此缺陷影响频率尚未量化，不能宣称为主要失败原因。

- 2026-11-14：完整critic隔离实际启用后，validation KL0.077772仍接近部分隔离0.074336，10k强度有效NO-GO（差+0.0599）。纠正梯度路径漏洞不等于解决全部策略退化；副露9.3506/游戏对镜像4.7157仅为行为线索，须按机会与动作优势核验，不能直接归因。证据docs/PHASE10.2.4_RESULT.md、experiments/ppo_v4_full_critic/diagnostics/policy_drift_by_epoch_validation_100k.json。

- 2026-11-14：warmup=True旧实现仍加BC anchor且默认只训练value_head，完整分离critic的编码器/trunk被冻住；修复为预热排除anchor、所有value_模块可训。旧10.2.4未启用这两项，不能据此解释其失败。测试tests/test_ppo_combined_controls.py，真实pilot epoch1 actor逐位不变；强制动作无policy梯度却影响旧平均/归一化尺度，新增decision_only可选口径。证据docs/PHASE10.2.5.md。

- 2026-11-14：组合控制减轻漂移（KL0.07777→0.01733），但10k配对CI仍跨0。末轮critic rollout EV仅0.03616，且为自举目标的更新前指标，不能当独立泛化证据；actor只含30617次有效选择/163完整游戏。KL早停仅损失6/800次更新，不支持其阻断大部分训练的说法。训练实际35m31s超出12–25m估时，后续同配置按该实测校准；性能根因未profile。证据docs/PHASE10.2.5_RESULT.md。

- 2026-11-14：sampling训练critic不能仅凭greedy回报误差判失败；reward-v1 value不是rank预测。10.2.6采样主skill区间跨0，不能误记FAIL或无用；lambda1与MC−V一致是恒等式，其局内优势方差60.876对lambda0.95的5.690，不能因相关性高选它。8-worker CPU诊断768游戏仅285.856s，单进程16游戏66.487s不宜直接外推多进程。证据docs/PHASE10.2.6_RESULT.md。
- 2026-11-14：10.2.10副露多+0.8724/局，但同seed副露差与顺位差Pearson仅−0.0252；汇总行为差不能推出多副露导致强度下降。BC状态同场概率对照证明旧PPO吃碰倾向+0.025左右，10.2.11定向KL可压到开发+0.00109；行为修正也不能当强度证据。复现`.venv/Scripts/python.exe scripts/analyze_phase10_2_11_behavior.py`（默认输出若已存在需指定新目录）；证据docs/PHASE10.2.11.md及RESULT.md。
- 2026-11-14：10.2.11小批1000对差−0.053、CI跨0，镜像均位1.581接近±0.091容差边界；正式独立10k差−0.0091、CI跨0、镜像1.4958，未过GO。小批资源筛选通过不等于模型强度已确认，镜像靠边时尤其不能把点估计当真实效应；修正吃碰行为也不等于顺位受益。复现`.venv/Scripts/python.exe scripts/verify_phase10_2_11_strength.py`；证据docs/PHASE10.2.11_RESULT.md。
- 2026-11-14：10.2.12新128对同seed完整游戏中，T1 sampling learner均位1.977、greedy1.484；同初始value−10.634分别面对完整回报−9.514/−0.730。greedy下critic MC skill−20.075%不等于sampling训练critic本身坏，说明策略模式分布差；冻结策略T0.5收窄至greedy差距约77%回报/83%顺位。直接用greedy轨迹训练PPO会令行为概率1与softmax比率不符，必须保持采样logprob与loss同一分布。复现`scripts/verify_phase10_2_12_temperature.py`；证据docs/PHASE10.2.12_RESULT.md。
- 2026-11-14：固定策略T0.5采样相对T1顺位改善−0.406（128配对局），但训练后greedy强度小批候选−BC镜像差+0.005、CI[−0.041,+0.051]。温度缩放对固定权重不改变argmax；采样分布改善不能代替新权重的贪心动作排序净收益。复现`.venv/Scripts/python.exe scripts/verify_phase10_2_12_temperature.py`及`.venv/Scripts/python.exe scripts/verify_phase10_2_12_screen.py`；证据`docs/PHASE10.2.12_RESULT.md`。
- 2026-11-14：预计18–23分钟的10.2.12小批曾被持续等待和查看进度，违背`10-project.md`的预计>20分钟规则；即使是“小规模”训练/筛选，也应启动后立即停止实时监控，等用户通知自然完成后核验。实际批次耗时1230.473秒。复盘见`docs/PHASE10.2.12_RESULT.md`；用户本轮明确纠正。
- 2026-11-14：`reward-v1`点数回报与顺位较相关（已有BC开发64局相关−0.868），但其波动尺度远高于终局顺位奖；`lambda=.95`下开发choice状态顺位/点数直接trace约0.063/1.178。直接trace排除了value bootstrap，不能冒充完整GAE或认定奖励bug；训练日志EV是自举目标口径，也不能替代独立greedy预测检查。复现`.venv/Scripts/python.exe scripts/analyze_phase10_2_12_followup.py`，证据`docs/PHASE10.2.12_RESULT.md`。
- 2026-11-14：比较不同温度checkpoint的同状态动作**概率**时，BC对照必须设相同温度；greedy argmax不受正温度缩放影响，但概率差会受影响。10.2.12首次`phase10_2_12_actions_v1`使用BC/T1对PPO/T0.5，立直概率差不可用；`--match-bc-temperature`重算`phase10_2_12_actions_matched_v2`后开发差+0.0150。复现`scripts/analyze_phase10_2_12_actions.py`，证据`docs/PHASE10.2.12_RESULT.md`。
- 2026-11-14：10.2.12训练日志更新前自举EV约0.58，但新完整局的sampling critic对真实MC相对常数skill−19.7%且CI全负，greedy−28.2%也全负；训练EV不能充当独立critic质量门。旧BC train/dev与新128场BC确认显示纯rank和rank-dominant公开状态目标都有可预测信息，需保持训练reward与critic初始化同口径。复现`scripts/verify_phase10_2_13_critic.py`和`scripts/confirm_phase10_2_13_rank.py`；证据`docs/PHASE10.2.13_RESULT.md`。
- 2026-11-14：10.2.13新reward scale0.00005必须在独立critic采集时同步设置；旧诊断器默认0.001会把训练critic与不同目标比较，故版本化新诊断器并用8场pilot及384场独立账本验证。新采样critic相对常数skill+15.159% CI全正；greedy区间跨0。局内value/MC相关为负，但终局rank bonus在同局恒定、未来点数差随当前分数变化，相关性不能单独当作动作排序错误。复现`scripts/verify_phase10_2_13_shaped.py`；证据`docs/PHASE10.2.13_RESULT.md`。

- 2026-11-14：小批candidate相对镜像点估计−0.0355仍不足以晋级；95% CI[−0.087,+0.0175]跨0，预注册门要求CI上界<0。critic skill+35.244%（纯rank）与+20.659%（混合）按各自常数基线归一化，不能跨目标比较为实战收益；奖励取舍需看新seed配对强度结果。screen实际1208.224秒，18–23分钟估时下未实时监控。证据`docs/PHASE10.2.13_RESULT.md`。

- 2026-11-14：[推测] Phase10.2.14训练CPU低利用率与配置吻合：两臂均`num_envs=1`、CUDA，PPO采样器只创建一个环境并串行循环；评估的16 workers只用于benchmark。配置沿用时应显式检查训练采样并行度和吞吐；在匹配对照跑批中途不能只改一臂。证据`configs/train_ppo_v4_pure_rank.yaml:54-55`、`src/mahjong/training/ppo.py:412,511-529`、`scripts/benchmark_parallel.py:177-185`；当前利用率数值来自用户观测，未做实时遥测。

- 2026-11-14：10.2.14纯顺位−混合目标点估计−0.033但游戏级95%CI[−0.0865,+0.021]跨0；2000局区间只表示单个训练seed下的评估误差，不能证明跨seed普适或两目标等效。执行时主bootstrap seed曾误写10362，原始逐局按预注册10360复算后仍INCONCLUSIVE；保存实际seed和独立复核避免隐藏偏差。证据`docs/PHASE10.2.14_RESULT.md`、`verify_phase10_2_14_eval.py`。

- 2026-11-14：[假设]纯顺位回报仅终局发放，λ=.95时按同座位约140–168次决策反向传播，terminal TD残差权重约0.00076–0.00018；公开critic训练EV较高但不保证含私有手牌条件的动作优势排序。λ=1可无衰减传播但可能增大方差，须用有界新seed screen检验。此前λ=1方差诊断基于混合点数奖励，不能直接外推纯顺位。见`docs/PHASE10.2.16.md`及`src/mahjong/training/ppo.py:60-70,289-314`。

- 2026-11-14：纯顺位λ=1.0候选优势std约0.88–1.34、更新前EV0.14–0.38，screen delta+0.0185 CI[−0.0345,+0.0705]有效NO_GO；λ=.95 seed43的训练EV更高但不是同训练seed直接比较，不能断言lambda导致EV差。完整rank reward需要信用传播与低方差折衷，λ=.99作为后续可证伪假设。见`docs/PHASE10.2.16_RESULT.md`。

- 2026-11-14：[假设]纯顺位λ=.95终局残差对早期同座位动作衰减至约0.0002–0.0008；λ=1 seed44训练EV低、advantages方差较高但与λ=.95 seed43存在seed混杂。λ=.99理论残差权重约0.185–0.245，可作为传播/方差折衷待screen验证，不能先验宣称提升。见`docs/PHASE10.2.17.md`。

- 2026-11-14：λ=.95→1.0→.99连续单变量跑批违背用户`10-project.md`“问题或卡门时先集中排查多种原因并一次组合调整”约束；第二轮λ=.99于epoch5停止。往后纯顺位NO_GO须先综合审查credit传播、critic方差/表示、BC/KL限制与seed不确定性，不能逐个lambda值试跑。依据`docs/PHASE10.2.17_RESULT.md`。

- 2026-11-14（更正）：用户明确单变量跑批没有被禁止。问题在于失败后不能默认每次改一个参数、连续低效试跑；应先全面排查多种因素，只有某个子假设清晰且值得隔离时才单独验证；若证据支持因素共同作用，再做组合调整。Phase10.2.18据此审查多因素后仅保留λ=.99为有证据候选，没有足够证据拼凑anchor/KL/critic改动。依据用户本轮说明及`docs/PHASE10.2.18_RESULT.md`。

- 2026-11-15：文档归档路径更新：40份 Phase 10.2 文件现位于 docs/PHASE10.2/，汇总见 docs/PHASE10.2_STRENGTH_SCREE_PASS.md。本文历史条目中的旧 docs/PHASE10.2.X.md 路径按新目录映射读取；历史记录正文未改。
