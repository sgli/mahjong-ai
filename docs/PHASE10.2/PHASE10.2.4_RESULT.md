# Phase10.2.4：完整 critic 编码路径隔离修复

2026-11-14：100k训练与10k-pair正式评估均已完成，独立核验一致，`valid_run=true`、`decision=NO_GO`。完整critic隔离实际启用，但候选仍弱于BC-v4；保留正确性修复，BC-v4继续作为强度基线。

## 确认的问题

`RiverSequencePolicy._heads`先通过共享tile_embedding/river_encoder生成特征，再交给可分离的MLP trunk。因此旧`value_separate_trunk=true`只隔离MLP，critic loss仍可更新actor的embedding和GRU。只反传value loss的实际复现测到embedding梯度绝对值和0.000534，GRU输入权重梯度0.015534（一次随机合成输入；数值不是固定门槛）。这是实现与“critic完全隔离”设计表述不符的缺陷。

Phase10.2.2和10.2.3的实际机制均是“MLP分离、序列编码共享”。原对局结果保留，不能将它们解释成已经检验了完整critic隔离。共享trunk是否导致全部强度失败仍未确认。

## 修复与验证

- 新增显式`model.value_separate_encoder=true`，要求`value_separate_trunk=true`；critic拥有独立embedding、GRU、MLP和value head。
- BC初始化将actor的embedding/GRU权重复制到critic副本；actor初始logits保持一致。
- 默认false保留旧架构及state_dict布局；真实BC-v4、旧gamma1 PPO和新pilot checkpoint均通过现有`load_policy`加载，分别识别为false/false/true。
- `tests/test_river_critic_isolation.py`覆盖value-only优化后actor参数梯度为空、logits逐位不变、BC复制、strict保存恢复和非法配置；与`tests/test_ppo_gae.py`合计10项通过。`git diff --check`通过。
- CUDA pilot：`configs/train_ppo_v4_full_critic_pilot.yaml`，512 learner transitions、2047环境步、1次预算截断、illegal0，约8秒完成训练/保存和1局self-play smoke。输出`experiments/ppo_v4_full_critic/pilot_512_seed42/`；pilot不作强度证据。

## 启动记录（2026-11-13）

已准备`configs/train_ppo_v4_full_critic.yaml`：从BC-v4重新初始化、gamma0.99/lambda0.95，沿用Phase10.2.2其余设置，只新增完整critic编码器隔离；100k预算。新架构多一次GRU计算，旧100k耗时约9.5分钟只能作参考，512 pilot不能可靠外推全量时长。

进一步修复feature-v3训练collector误删对手九种九牌菜单的问题；Rule对手可自行选择其完整合法菜单。新增normal99-v2镜像窗口，10k约±0.0288，无效评估明确记INVALID，旧默认fixed-v1保留。13项定向测试通过（test_phase10_2_4_gates、test_river_critic_isolation、test_ppo_gae）。

正式方案见`PHASE10.2.4_PREREG.md`：100k训练、10k配对种子680000–689999、bootstrap10323、16 workers，预计完整流水线2–3小时。流水线状态`experiments/ppo_v4_full_critic/phase10_2_4_run_v1/state.json`，训练与评估日志同目录。不实时监控，等待用户通知自然完成后分析。

新增菜单正确性修复会影响Rule对手，因此与历史10.2.2存在混杂；本轮检验修正后候选对BC-v4的强度，不能声称全部差异均由完整critic隔离单独造成。

## 正式结果（2026-11-14）

流水线state为complete。训练10 epochs、100,000 learner transitions、400,054环境步、152局完整游戏；各epoch非法率0。训练652.937秒，评估6,009.315秒，流水线共6,663.312秒（约1小时51分）。checkpoint确认两个隔离选项均为true。

| 指标 | PPO候选 | BC-v4镜像 |
|---|---:|---:|
| 完整有效游戏数 | 10,000 | 10,000 |
| 平均顺位（0起算，越低越好） | 1.5657 | 1.5058 |
| 一至四位次数 | 2430/2269/2515/2786 | 2442/2533/2550/2475 |
| 和牌次数/游戏 | 2.3968 | 2.2347 |
| 放铳次数/游戏 | 1.5920 | 1.3575 |
| 副露次数/游戏 | 9.3506 | 4.7157 |
| 立直次数/游戏 | 2.2930 | 2.5315 |
| 平均和牌点数 | 8369.28 | 9190.80 |

配对差为 **+0.0599**，bootstrap95%CI **[+0.0328,+0.0867025]**；未达到差<=−0.020且CI上界<0的门槛。镜像通过normal99-v2窗口（±0.02879865），两臂非法动作0，故是有效NO-GO。行为次数以整场游戏为分母，并非每手牌或每次机会的概率；只能定位异常方向，不能据此证明因果。

独立核验了种子680000–689999唯一且完整、各座位2500局、原始顺位排列、配对CSV、checkpoint身份，并以PCG64 seed10323重算10,000次bootstrap，结果一致。配置SHA256：`77c49e6ca711532b97e13d5bb5c297d3a367d6e33730a4bc9f32879d91ea03dc`；候选SHA256：`cddc16fd1cbbfaf8dcbfabd3361ef466c11b144fa526d60e742b31912f67cea0`。

证据：`experiments/ppo_v4_full_critic/phase10_2_4_run_v1/state.json`；`experiments/ppo_v4_full_critic/strength_10k_seed680000_v1/`下的`result.json`、`paired_rank_differences.csv`和`independent_verification.json`。

## 漂移诊断与结论

固定validation前100k行事后诊断耗时5.21秒，epoch10与BC一致率85.544%、gold top-1 67.270%（BC 70.341%）、KL(BC‖PPO)0.077772；旧部分隔离分别85.606%、67.244%、0.074336。完整隔离没有在这些描述指标上表现出进一步减轻漂移。命令：`.venv/Scripts/python.exe experiments/ppo_v4/diagnostics/diag_policy_drift_epochs.py --run-dir experiments/ppo_v4_full_critic/seed42_100k --output experiments/ppo_v4_full_critic/diagnostics/policy_drift_by_epoch_validation_100k.json`。该抽样诊断不用于重选checkpoint或替代强度门。

本轮差+0.0599小于旧部分隔离的+0.0844，但评估种子不同、区间重叠且菜单修复有混杂，不能认定完整隔离带来了实战提升。完整梯度隔离已修正设计漏洞，但仍不足以让本配方超过BC-v4。高副露、较多放铳及较低和牌点数仍存在，剩余原因尚未确证。

## 后续

停止本配方扩展，不追加强度样本、不晋升候选、不自动延长训练。先做有界的决策级归因：按副露/过/立直等动作统计合法机会、优势分布、奖励构成，并区分policy与entropy更新的影响；据证据再预注册一个单变量实验。上述是下一步诊断计划，本轮未执行或启动新训练。
