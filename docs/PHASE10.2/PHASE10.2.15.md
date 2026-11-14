# Phase 10.2.15：纯顺位 checkpoint 独立CI确认

日期：2026-11-14。Phase10.2.14中纯顺位seed43 checkpoint在2,000局上的BC-v4配对差点估计为−0.037，但95% CI上界为+0.012，未过小批门。用户明确要求在纯顺位基础上针对CI继续推进。本阶段通过一批全新对局降低该checkpoint强度估计的抽样误差，不训练或修改模型；CI变窄是评估精度变化，不是模型性能优化。

## 冻结对象和评估设计

- 候选checkpoint：`experiments/ppo_v4_rank_compare/pure_rank_seed43_100k/best.pt`，预期SHA `e1a571f89c0929e03ed3dad5958dd1b4ed64c069d968b7cd3f91fe9e3fdbd33d`。
- 对手/镜像：BC-v4 greedy checkpoint `experiments/bc_v4/checkpoints_20w/best.pt`，SHA `814c6eba4e163d126cff7bc27194f11612263c745328abd1b1d9ae94b6716937`。候选对3家greedy BC-v4；另跑同seed BC-v4镜像。
- 10,000局/臂，seed 890000–899999，16 workers，候选席位按`game_index mod 4`轮换；bootstrap PCG64 seed 10370，10,000次percentile 95% CI；镜像用normal99-v2。此seed段与Phase10.2.14的880000–881999不重叠，也未见其在既有项目记录出现。
- GO：所有局完整、非法0、镜像sanity通过；候选减镜像的paired mean rank ≤−0.020且95% CI上界<0。否则本确认批为NO_GO，不追加局数、不改门。该独立10k只评估既有单训练seed checkpoint，不能证明跨训练seed稳定。
- 配置记录：`configs/benchmark_phase10_2_15_pure_rank.json`；输出：`experiments/ppo_v4_rank_compare/pure_rank_confirm_seed890000_10k_v1/`。

## 估时与运行约束

Phase10.2.14评估实测1693.419秒完成纯/混合/镜像各2000局，即约564秒/2000局臂。当前两臂共20,000局预计约5,640秒/94分钟；结合波动，估90–110分钟。超过20分钟，故不设上限、不实时监控，启动后等用户通知自然完成再核验。

结果记入`PHASE10.2.15_RESULT.md`。不提交，不更新STATUS.md。
