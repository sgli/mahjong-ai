# Phase 10.2.19：纯顺位 λ=.99 新 seed 筛选

日期：2026-11-14。该方案依据 Phase10.2.18 的多因素审查。除 GAE λ=.99 外，不改 reward、网络、critic 初始化、BC/call anchor、KL/clip、训练预算、对手池或部署策略。Phase10.2.17 的中断训练不续跑，使用新训练 seed46。

## 候选及假设

- 假设：终局纯顺位奖励下，λ=.95 会较强衰减远期信用；λ=1 的 MC 方差较高；λ=.99 能在保留更多远期信用的同时控制方差。新轨迹诊断显示符号一致率约 .83、局内方差约 .14；这些是机制依据，不是强度证据。
- BC-v4 actor 初始化与 critic 初值沿用已通过验证的 Phase10.2.14 纯顺位配置；纯顺位 reward scale=0。
- 训练配置：100,000 learner transitions，10×10,000；seed46/env_seed660000/opponent_pool_seed46；`num_envs=1`、CUDA、16 更新 minibatch/epoch 设定等沿用基线；GAE lambda=.99。
- 先做 4,096-transition pilot；数值有限、非法动作 0、checkpoint 与 lambda/reward 元数据正确才开始正式筛选。
- 新 screen：seed1,140,000–1,141,999，候选 greedy 对 3×greedy BC-v4，BC-v4 同 seed/座位镜像；2,000 对、16 workers、bootstrap PCG64 seed10400/10,000 replicates、normal99-v2。
- screen GO 门：数据/模型身份有效、两臂完整且非法动作 0、镜像 sanity 通过、均差≤−0.020 且 95% CI 上界<0。任一失败即 NO_GO，不补 seed、不改门。
- 只有 screen GO 才用同一 checkpoint 开始独立 10,000 对正式强度评估，seed1,150,000–1,159,999，bootstrap seed10401；保持同一配对、镜像、非法动作和 GO 门。screen 不通过则不启动。

## 吞吐估时与监控

- 4,096-transition pilot 历史同类实测约 51 秒。
- 100k PPO 训练以 Phase10.2.16 同类实测 12分42秒为基准，预计 12–18 分钟。
- 2,000 对 screen 以 Phase10.2.16 的 1,541.948 秒为基准，预计 24–30 分钟。
- pilot、训练、screen 合计预计约 37–49 分钟；若通过再加正式 10k 评估，按 Phase10.2.15 同类实测 5,579.487 秒（约93分钟）估算。
- 流水线无时间上限、无实时监控。超过20分钟即停止监控，等待自然完成后再核验；不提交、不更新 STATUS.md。

配置与核验：`configs/train_ppo_v4_pure_rank_lambda099_audit_seed46.yaml`、`configs/train_ppo_v4_pure_rank_lambda099_audit_seed46_pilot.yaml`、`configs/benchmark_phase10_2_19_lambda099_screen.json`、`configs/benchmark_phase10_2_19_lambda099_strength.json`；对应 pipeline 与独立核验脚本在 `scripts/run_phase10_2_19.py` 及 `scripts/verify_phase10_2_19_*.py`。
