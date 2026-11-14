# Phase 10.2.1 结果：BC-v4 初始化 PPO 单变量筛选

状态：训练与预注册 20,000 paired seeds 强度评估均完成；阶段结论 NO-GO。日期：2026-11-13。

## 固定实验

方案见 [PHASE10.2.1_PREREG.md](PHASE10.2.1_PREREG.md)，resolved config 和机器可读结果见 `experiments/ppo_v4/seed42_100k/resolved_config.json` 与 `result.json`。

- BC-v4 初始化，feature-v3/action-v2，environment `tenhou-v3-ron-context`，reward-v1；seed 42。
- 10 epochs，每 epoch 10,000 learner transitions；总计 100,000 learner transitions 和 400,142 environment steps。
- 对手池：BC-v3 50%、BC-v4 25%、Rule-v1 25%；Adam lr 1e-4、batch 512、4 update epochs；BC anchor/warmup/reward clip 均关闭。
- checkpoint selection 为 `latest_epoch`，因此 `best.pt` 指向 epoch 10；它不是按对局强度筛选的 best policy。

## 训练结果

| Epoch | Policy loss | Value loss | Entropy | Approx KL | Clip fraction | Mean reward | Illegal rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00062 | 4.8134 | 0.2788 | 0.00901 | 0.06449 | 4.6188 | 0 |
| 2 | -0.00419 | 6.6069 | 0.3027 | 0.00653 | 0.05277 | -4.2750 | 0 |
| 3 | -0.00045 | 5.2372 | 0.3181 | 0.00553 | 0.05239 | -11.1294 | 0 |
| 4 | -0.00241 | 2.4464 | 0.3515 | 0.00246 | 0.03503 | -4.2938 | 0 |
| 5 | -0.00111 | 3.4359 | 0.3456 | 0.00325 | 0.03689 | -2.0471 | 0 |
| 6 | -0.00153 | 3.1744 | 0.3437 | 0.00358 | 0.04395 | -7.5429 | 0 |
| 7 | -0.00617 | 2.2885 | 0.3542 | 0.00319 | 0.03456 | -8.2000 | 0 |
| 8 | -0.00410 | 2.5294 | 0.3570 | 0.00345 | 0.03705 | -6.8867 | 0 |
| 9 | -0.00564 | 2.6154 | 0.3423 | 0.00332 | 0.03001 | -5.4176 | 0 |
| 10 | -0.00435 | 2.2003 | 0.3462 | 0.00339 | 0.03652 | -6.1867 | 0 |

对手池自博弈训练的 mean reward 波动，不作为强度证据。训练后脚本运行 5 局 self-play smoke，mean rank 1.5；样本太小且不是固定基线对局，同样不作强度结论。

## 耗时、身份与产物

- 开始时间 2026-11-13 07:18:50；最终 checkpoint 写出时间 07:28:22；总耗时约 **572.8 秒（9分33秒）**。正式吞吐约 **175 learner transitions/s**。
- pilot 512 transitions 的 10 秒量级耗时线性外推为 25–45 分钟，实际正式 run 远短于估算。正式全量同类吞吐优先于小 pilot；旧估算不沿用。
- BC-v4 初始化 SHA-256 `814c6eba4e163d126cff7bc27194f11612263c745328abd1b1d9ae94b6716937`；BC-v3 对手 SHA-256 `f67f03483930491bded216e06411c72ab6ede9bf81c18506a1a749fe2d1a6971`。
- `best.pt` epoch 10 SHA-256 `5e1600412b7ce64e0c67333f865a5c3095c4a8dc138fd353ba0f2132ebc707fd`；`latest.pt` SHA-256 `a61f691449c2b096b14c6fecfa6954e224698a8be7965388a4896ce0a1b23e73`。
- 输出目录：`experiments/ppo_v4/seed42_100k/`；摘要：`result.json`；逐 epoch checkpoint、resolved config、stdout/stderr logs 同目录。

## 预注册强度评估

固定方法见 [PHASE10.2.1_PREREG.md](PHASE10.2.1_PREREG.md)。候选 PPO greedy/T=1 对 3×BC-v4，镜像为 BC-v4 对 3×同一 BC-v4；seed 620000–639999、每臂20,000局、席位按 game index 轮换。运行目录 `experiments/ppo_v4/strength_20k_seed620000_v1/`，执行耗时11,161.422秒（约3小时6分），候选臂5,685.246秒、镜像臂5,473.755秒。

| 指标 | PPO candidate | BC-v4 mirror |
|---|---:|---:|
| 局数 | 20,000 | 20,000 |
| 平均顺位 | 2.03945 | 1.49495 |
| 顺位分布（0/1/2/3） | 1,979 / 3,703 / 5,868 / 8,450 | 4,993 / 5,084 / 4,954 / 4,969 |
| 非法动作 | 0 | 0 |
| 完整有效局 | 20,000 | 20,000 |

逐 seed 配对差（PPO − BC-v4 mirror）为 **+0.54450**，bootstrap 95% CI **[+0.52560, +0.56410125]**，sample SD 1.39718。镜像均位1.49495，落在 1.5±0.02，sanity 通过；种子覆盖、席位数（每席5,000局）和两侧 checkpoint SHA 均核验通过。对 paired CSV 进行独立复算，均值与 bootstrap CI 和 `result.json` 一致。

**预注册判定：NO-GO。**GO 要求配对差 `<= −0.020` 且 CI 上界 `< 0`；实际差为正且区间完全大于0，PPO 明显未达到基线。该结论不支持这一个固定 seed42/100k PPO 配方；不能从中单独归因是学习率、奖励、对手池或其他因素导致。按预注册停止，不追加样本、不改阈值、不启动独立 seed；BC-v4 保持当前通过强度门的策略，不晋升 PPO checkpoint。

机器可读原始结果 `experiments/ppo_v4/strength_20k_seed620000_v1/result.json`，逐 seed 数据 `paired_rank_differences.csv`，双臂原始局 `candidate_arm.json` / `mirror_arm.json`。训练产物与身份见上文 `experiments/ppo_v4/seed42_100k/result.json`。
