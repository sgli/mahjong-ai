# Phase 10.2.14：纯顺位与顺位主导混合目标比较预注册

日期：2026-11-14。目标是检验：在其余 PPO 设置相同的情况下，纯终局顺位奖励与顺位主导混合奖励（`0.00005 × 点数差 + placement bonus`）哪一个能得到更好的 greedy 实战顺位。BC-v4仍是强度基线。本阶段不是已证明有效的配方，也不继承旧候选的强度结论。

## 依据与限制

Phase 10.2.13 中，纯顺位公开状态预测 skill +35.244%（95% CI [+27.291%, +41.977%]），混合目标 skill +20.659%（[+13.564%, +26.696%]）。两项 skill 使用各自目标的常数预测MSE作分母，不能相互比较成强度结论。旧混合PPO在seed840000–841999有效小批的配对差为−0.0355，但CI [−0.087,+0.0175]跨0，因此NO_GO。新阶段须从同一个BC-v4 actor初始化重新训练两种目标，不能把旧混合checkpoint与新纯顺位checkpoint直接当作公平对照。

## 预注册训练方案

- 共同起点：`experiments/bc_v4/checkpoints_20w/best.pt`（SHA `814c6eba4e163d126cff7bc27194f11612263c745328abd1b1d9ae94b6716937`）。
- 两臂均为100,000 learner transitions（10×10,000），seed=43、env_seed=630000、opponent_pool_seed=43；相同网络、T=1、greedy BC-v4训练对手、学习率、batch、更新轮数、gamma/lambda、KL、BC/call anchor与checkpoint选择。
- 唯一实验处理是回报目标及其目标匹配的公开critic初值：纯顺位`reward_score_scale=0`，使用旧训练256局拟合的placement系数；混合`reward_score_scale=0.00005`，使用相同训练256局拟合的rank-dominant系数。两个系数均来自已经冻结的10.2.13确认产物`experiments/ppo_v4_tempered/phase10_2_13_bc_confirm_v1/rank_confirmation.json`，不使用确认局拟合。
- 配置：`configs/train_ppo_v4_pure_rank.yaml` 与 `configs/train_ppo_v4_rank_mix_replication.yaml`。产物目录分别为`experiments/ppo_v4_rank_compare/pure_rank_seed43_100k/`与`.../rank_mix_seed43_100k/`。先后顺序纯顺位再混合，避免GPU并发争用。逐臂核验100k、epoch、有限loss、非法动作、奖励版本、scale、critic初值、actor起点与checkpoint SHA。

## 预注册评估与判定

- 两个checkpoint都用greedy策略，对阵3家greedy BC-v4。使用同一新seed范围880000–881999、每臂2000局、同一候选席位轮换（game_index mod 4）、16 workers；另跑同seed同口径BC-v4镜像2000局。旧840000–841999及860000–869999均不复用。
- 主比较是逐seed纯顺位−混合顺位（负值偏向纯顺位），使用PCG64、预注册bootstrap seed 10360、10,000次游戏级配对bootstrap percentile 95% CI。纯顺位胜出门：均差≤−0.020且CI上界<0；混合胜出门：均差≥+0.020且CI下界>0；其余为比较不确定，不挑点估计赢家。
- 实用门：每个候选分别对同seed BC镜像计算paired rank差。只有均差≤−0.020、95% CI上界<0、镜像均位距1.5≤normal99-v2容差、逐局完整、非法0，才记为该100k筛选种子上的候选GO。候选GO只授权讨论独立训练seed确认/正式强度评估，不代表晋升。若两候选都未GO，则不继续正式评估；若比较区间跨0，则不宣称目标胜负。
- 若所有门都有效但未分出赢家，停止本阶段，不补seed、不改变阈值、不在本评估集调参；根据不确定度另行决定是否需要独立训练种子复现。

## 估时与执行约束

每个100k训练臂参考10.2.13实际约12分16秒，估计12–18分钟。完整两臂连续训练预计24–36分钟，故整个训练链超过20分钟：启动后立即停止实时监控，不设时间上限，等用户通知自然完成后再核验并进入评估。评估历史2000对两臂（候选+镜像）实际1208.224秒；本次三个2000局臂（纯、混合、共同镜像）按吞吐约估30–35分钟，同样不设上限、不实时监控，等用户通知完成。

所有计划、训练和评估证据写入`PHASE10.2.14_RESULT.md`。不提交代码，不恢复/更新STATUS.md。
