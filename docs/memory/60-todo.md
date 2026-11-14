# 当前待办

## P0
- [x] 对 feature-v3 编码、序列 padding/顺序、action-v2 gold/legal mask、policy 前向做有界正确性检查并记录结果（`scripts/verify_bc_v4_inputs.py`）。
- [x] 使用 64-shard pilot 测 feature-v3 吞吐并做小型 CPU 监控；正式训练估时与 CPU 采样见 `30-experiments.md`。
- [x] 对照 BC-v3/BC-v4 的数据维度、完整配置 CPU/GPU/worker 占用和逐批指标同步开销；诊断 run 只到 step 13000，不是模型结果。
- [x] 将 BC-v4 训练诊断指标改为每1000步抽样、精确累计 loss；完成全配置150秒对照和64-shard epoch汇总 pilot。
- [x] 完成 BC-v4 正式 5 epoch 训练（20 workers/prefetch4）；best epoch2及抽样指标口径见 `30-experiments.md`。
- [x] 定位并优化 BC 全量评估器逐批同步与非必要逐行分组统计；短窗 CPU pilot 已见改善。
- [x] 完成优化后的 BC-v4 全量 test（134,574,358 行），核验 checkpoint/data 身份及指标；离线结果高于 BC-v3，非法率均为0。

## P1
- [x] 2026-11-14 完成10.2.10公开value版本化、14项定向测试、2×2048 pilot与128对小批安全门；具体数值见docs/PHASE10.2/PHASE10.2.10_RESULT.md。
- [x] 10.2.10正式100k训练健康门通过，并自动启动10k/臂强度评估；见docs/PHASE10.2/PHASE10.2.10_RESULT.md。
- [x] 10.2.10强度评估自然完成后核验CSV、身份、镜像和预注册GO门；有效NO_GO，见docs/PHASE10.2/PHASE10.2.10_RESULT.md。
- [x] 基于现有BC轨迹完成吃碰机会同状态分歧分析，提出10.2.11可证伪的定向约束机制；未复用10.2.10强度seed调参。
- [x] 10.2.11通过正确性、pilot、100k健康、同状态行为门和新seed1000对资源筛选；10k/臂正式评估已启动。
- [x] 10.2.11强度评估自然完成并独立核验模型SHA、逐局/CSV/bootstrap、镜像与预注册GO门；有效NO_GO，见docs/PHASE10.2/PHASE10.2.11_RESULT.md。
- [x] 10.2.12分析非吃碰动作分歧、新完整游戏GAE/MC与sampling/greedy分布；冻结T0.5单变量候选并完成100k健康与行为门。
- [x] 10.2.12新seed2000对筛选完成并独立核验；事前门失败，不启动新seed10k/臂正式评估。
- [x] 10.2.12后续只读训练/开发轨迹集中核对旧/新PPO更新预算、奖励传播和rank相关性；见docs/PHASE10.2/PHASE10.2.12_RESULT.md。
- [x] 已有前32训练/前32开发BC轨迹复核新PPO相对BC-v4的动作翻转类别，并同温度比较立直概率；主要分歧为弃牌牌张，尚无反事实收益证据。
- [ ] 设计有界新训练/开发seed诊断：分层抽取弃牌分歧，保存环境状态并配对分叉rollout，以估计替代弃牌的反事实价值；旧BC轨迹仅有执行动作/MC，不能回答此问题。
- [ ] 审查新PPO on-policy critic对早期状态远期回报的预测质量及greedy部署口径；未有机制证据前不调KL/温度/奖励并重训。
- [x] 10.2.13冻结10.2.12模型新384场完整局，独立核验sampling/greedy critic均显著低于校准常数；旧critic训练EV不可替代独立回报验证。
- [x] 新128场BC-v4确认rank-dominant公开value目标可预测，拟合系数只用旧训练256局；冻结组合配置与2×2048 pilot、真实奖励账本检查通过。
- [x] 10.2.13 100k训练核验模型身份/非法/更新；新seed完整局sampling critic与固定BC行为门通过，允许新2000对小批。
- [x] 新seed840000–841999的2000对小批自然完成并独立复核逐局/镜像/bootstrap；有效NO_GO，未启动独立seed10k正式强度评估。
- [x] 同 decision-v3 完整 test 完成 BC-v4 与 BC-v3 离线对照；5M pilot只作性能与冒烟结果。
- [x] 完成预注册强度 run 并核验 GO 门（20k配对种子；paired effect −0.05535，95% CI [−0.07495125, −0.03645]；详见 `30-experiments.md`）。
- [x] 冻结 Phase 10.2.1 PPO 筛选预算、对手池与评估 GO 门（`docs/PHASE10.2/PHASE10.2.1_PREREG.md`）。
- [x] 完成 512 learner-transition PPO pilot；训练/评估路径与 checkpoint 写出正常、illegal rate 0，记录于 `30-experiments.md`。
- [x] 完成 seed42 / 100k learner-transition PPO 主筛选并核对指标、身份和非法率（`docs/PHASE10.2/PHASE10.2.1_RESULT.md`）。
- [x] 完成正式20k/臂配对 PPO 强度评估并核验 NO-GO（paired delta +0.54450，95% CI全正；镜像 sanity 通过），详见 `docs/PHASE10.2/PHASE10.2.1_RESULT.md`。
- [x] 完成 Phase10.2.2 10k-pair强度评估并独立复算CSV、paired CI、镜像sanity和局完整性；NO-GO，详见 `docs/PHASE10.2/PHASE10.2.2_RESULT.md`。
- [x] 基于终局顺位奖折扣诊断，冻结Phase10.2.3单变量gamma1.0训练与10k-pair门，见`docs/PHASE10.2/PHASE10.2.3_PREREG.md`。
- [x] Phase10.2.3 100k训练完成，核对checkpoint/schema/gamma与illegal0；10k-pair评估已启动。
- [x] Phase10.2.3 10k-pair完成，独立复算result/CSV与身份；镜像sanity失败，正式门无效，记录于`docs/PHASE10.2/PHASE10.2.3_RESULT.md`。
- [x] 下一次评估使用预注册normal99-v2镜像口径（10k约±0.0288），旧评估维持fixed-v1。
- [x] 修复feature-v3 critic隔离遗漏embedding/GRU；10项定向测试与512-transition CUDA pilot通过（docs/PHASE10.2/PHASE10.2.4_RESULT.md）。
- [x] 冻结PHASE10.2.4_PREREG并启动完整critic隔离100k训练→10k-pair评估流水线，PID2712。
- [x] 2026-11-14核验Phase10.2.4完整流水线、checkpoint及10k强度门；独立复算一致，有效NO-GO，详见docs/PHASE10.2/PHASE10.2.4_RESULT.md。
- [x] 2026-11-14集中检查奖励/轨迹/动作机会/局部梯度/训练分布，修复预热问题并冻结组合方案，见docs/PHASE10.2/PHASE10.2.5.md。
- [x] 组合pilot两轮共4096条通过，actor预热逐位不变；已启动统一110k训练→10k配对评估，PID16460。
- [x] 2026-11-14核验10.2.5结果与11轮漂移；有效NO_GO，详见docs/PHASE10.2/PHASE10.2.5_RESULT.md。
- [x] 2026-11-14完成10.2.6固定actor sampling/greedy完整回报诊断及独立核验；768游戏，主门INCONCLUSIVE，详见docs/PHASE10.2/PHASE10.2.6_RESULT.md。
- [x] 完成BC-v4 sampling actor的critic可学习性对照：10.2.7使用448局并按256/64/128训练、开发、测试拆分；测试为INCONCLUSIVE，不据此宣称critic有效。
- [ ] 保持 BC-v4 为强度锚点；不续训 Phase10.2.1 原 PPO 配方。

## 已完成的整理
- [x] 清理旧 Phase 10.3 配置、输出和阶段专用脚本；保留 BC-v3 结果归档、decision-v3 和通用代码。
- [x] 清理依赖已删除配置的测试夹具；保留不依赖旧 checkpoint 的 PPO 入口回归测试。
- [x] 由用户检查清理结果后，继续 feature-v3 正确性检查与 BC-v4 训练。
- [x] 将 STATUS.md 当前状态整理并收录到 `docs/PHASE10.md`；未运行测试或训练。
- [x] 按用户说明重整 STATUS.md：移除过时模型阶段详情，保留 Phase 1–9 基础完成摘要，并定义 10.1 BC-v4、门控后的 10.2 PPO。

- [x] 冻结Phase10.2.14纯顺位 vs 0.05混合目标对照方案、同seed训练配置与新评估seed。
- [x] Phase10.2.14两臂100k训练及seed880000–881999三臂评估完成；训练、逐局和镜像验证PASS，结果不确定且两臂均NO_GO，不启动正式10k。
- [ ] Phase10.2.14完成后，单独做有界PPO采样吞吐对照，量化`num_envs=1`与可用多环境并行的learner transitions/s、CPU/GPU利用率和训练总时长；在公平性和吞吐证据充分前，不改变当前已启动的配对两臂。

- [x] 按用户要求冻结纯顺位checkpoint的独立10k评估，使用新seed 890000–899999及原GO门。
- [x] Phase10.2.15自然完成并核验逐局/CI/镜像门；有效NO_GO，仅针对seed43 checkpoint。

- [x] 分析纯顺位seed43终局奖励与GAE衰减，冻结λ=1.0单变量方案、新训练seed和2k screen门。
- [x] Phase10.2.16自然完成并核验pilot/100k训练/seed1100000–1101999 screen；有效NO_GO，不进入10k。

- [x] Phase10.2.16 λ=1纯顺位screen有效NO_GO；记录高优势方差/低EV为机制线索，不作跨seed因果结论。
- [x] 预注册Phase10.2.17纯顺位λ=.99折衷筛选，冻结seed45训练与新seed1120000–1121999小批门。
- [x] 终止违反集中诊断约束的Phase10.2.17单变量λ=.99 run；仅pilot和epoch1–5训练完成，未做screen。
- [x] 完成 Phase10.2.18 纯顺位多因素审查：检查信用传播/方差、独立critic、actor/锚、KL/clip及有效性；证据只支持完整新seed筛 λ=.99，不支持随意叠加其他超参。
- [x] 预注册并启动 Phase10.2.19 seed46 pipeline（pilot→100k→2k screen；仅screen GO时进入10k）；PID13688，启动后停止监控。
- [x] Phase10.2.19 100k训练与新seed 2k screen完成并独立复核；screen通过预注册GO门，正式10k已自动启动。
- [x] Phase10.2.19正式10k评估完成并独立复核GO：delta−0.0289、95% CI[−0.0519,−0.0055]、镜像sanity与非法动作门通过。
- [ ] 预注册全新训练seed及评估seed，独立确认Phase10.2.19 seed46正式强度结果；确认前不晋级、不替换BC-v4基线。
- [x] 2026-11-15 汇总全部Phase10.2阶段的PPO强度调整路径至 `docs/PHASE10.2_STRENGTH_SCREE_PASS.md`，并将40份源Markdown归档至 `docs/PHASE10.2/`；更新引用及记忆索引，未改STATUS.md、代码或跑批数据。
