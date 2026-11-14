# Phase 10.2.1：BC-v4 初始化的 PPO 单变量筛选

状态：Phase 10.2.1 完成，预注册强度筛选 NO-GO。更新日期：2026-11-13。

## 冻结方案

预注册文件：[PHASE10.2.1_PREREG.md](PHASE10.2.1_PREREG.md)；正式配置：[train_ppo_v4.yaml](../configs/train_ppo_v4.yaml)。本次只检验从 BC-v4 初始化后做 100,000 learner transitions PPO fine-tuning 的强度变化。

- schema/environment：feature-v3/action-v2，tenhou-v3-ron-context，reward-v1。
- seed 42；10 epochs × 10,000 learner transitions；Adam lr 1e-4，batch 512，4 update epochs；BC anchor 与 warmup 关闭。
- 每个非 learner seat 独立抽样：BC-v3 50%、BC-v4 25%、Rule-v1 25%，sampling/T=1。
- 输出 `experiments/ppo_v4/seed42_100k/`；冻结的 BC-v3 与 BC-v4 权重 SHA 见 `resolved_config.json`。
- 后续强度门：PPO greedy vs 3×BC-v4，并以 BC-v4 vs 3×BC-v4 作镜像；各 20,000 局、配对 seeds 620000–639999。GO 需 rank 差 <= −0.020、bootstrap 95% CI 上界 < 0、镜像均位1.5±0.02、全部有效且非法动作0。单 seed 通过只准进入独立 seed 确认。

## 启动前代码修复

审查发现 opponent-pool rollout 原先每局都重新构造并加载固定策略对手，且同一 `bc_checkpoint` 被同时用作初始化和 BC 对手，无法将 BC-v3/BC-v4 分开。现已缓存每个 run 的 opponent 实例，并增加 `opponent_checkpoints` 显式映射，分别加载两份冻结 checkpoint。对应实现见 `src/mahjong/training/ppo.py`、`scripts/train_ppo.py`、`src/mahjong/training/opponent_pool.py`。

## Pilot 与主训练

- Bounded pilot 配置：`configs/train_ppo_v4_pilot.yaml`；512 learner transitions，实收 512 transitions/2,047 环境步，1 个预算截断游戏、0 个完整游戏，illegal rate 0。loss/梯度更新、单局 smoke evaluation 与 PPO checkpoint 写出成功；初始化 checkpoint SHA 与 BC-v4 冻结 SHA 一致，对手 SHA 分别匹配 BC-v3/BC-v4。
- Pilot 由启动至完成约 10 秒量级；完整主 run 按此吞吐粗估约25–45分钟，估时超过20分钟，故不设时间上限、不实时监控。
- 正式 run PID 15864 已退出；10 epochs 全部完成，共 100,000 learner transitions / 400,142 environment steps，训练约 572.8 秒（9分33秒）。非法率每 epoch 均为 0。结果见 `experiments/ppo_v4/seed42_100k/result.json`，日志 `stdout.log` / `stderr.log`。
- run 时长比 pilot 线性估算的 25–45 分钟短；本次正式吞吐约 175 learner transitions/s。正式同类 run 的实测比小 pilot 更可靠，后续不再用 512-transition pilot 的启动期耗时作线性外推。
- 训练日志最后一轮：policy loss −0.00435、value loss 2.2003、entropy 0.3462、approx KL 0.00339、clip fraction 0.0365、illegal rate 0。训练 mean reward 波动且不是强度门。
- `best.pt` 为 epoch 10（checkpoint_selection=`latest_epoch`），SHA-256 `5e1600412b7ce64e0c67333f865a5c3095c4a8dc138fd353ba0f2132ebc707fd`；`latest.pt` SHA-256 `a61f691449c2b096b14c6fecfa6954e224698a8be7965388a4896ce0a1b23e73`。5 局 self-play smoke mean rank 1.5，仅用于程序通路检查，不能据此判断强度。
- 配对 runner 的实现 pilot 用正式 seed 范围外的 900000–900015，每臂16局；两臂局均有效、illegal 0、seat轮换和配对CSV正常。mirror mean rank 为0.875，但 n=16 不足以评估1.5±0.02门槛，此值不作强度结论。pilot 输出 `experiments/ppo_v4/paired_runner_pilot_900000/`。
- **预注册强度评估完成：NO-GO。**两 arm 各20,000局、seed620000–639999全部有效，镜像均位1.49495且 sanity 通过，两侧非法动作0；PPO 均位2.03945，BC-v4镜像均位1.49495，配对差 +0.54450，95% CI [0.52560, 0.56410125]。CSV独立复算与报告相同。此固定方案不追加样本、不更改门槛、不启动独立 seed。完整结果见 [PHASE10.2.1_RESULT.md](PHASE10.2.1_RESULT.md) 和 `experiments/ppo_v4/strength_20k_seed620000_v1/result.json`。
- 正式强度跑批实际用时11,161.422秒（约3小时6分）；candidate 5,685.246秒，mirror 5,473.755秒。原估6–8小时偏高，已记录校准。
