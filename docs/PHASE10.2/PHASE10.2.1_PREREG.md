# Phase 10.2.1 PPO 单变量筛选预注册

预注册日期：2026-11-13（Asia/Shanghai）  
状态：冻结；先完成 512 learner-transition pilot，再启动下述固定主 run。  
父阶段：Phase 10.1.1 GO；BC-v4 checkpoint SHA-256 `814c6eba4e163d126cff7bc27194f11612263c745328abd1b1d9ae94b6716937`。

## 研究问题与假设

唯一研究变量是：从已通过强度门的 BC-v4 初始化，在固定环境与对手分布下，用 PPO 更新策略 100,000 个 learner transitions 后，是否比冻结 BC-v4 更强。无 BC anchor、无 value warmup、无 reward shaping 调整；不根据训练中间指标改变方案。

方向性假设：PPO 候选在预注册镜像评估中降低平均顺位。单 seed 仅作筛选；通过后仍需独立训练 seed 复现，才能考虑升级为强度基线。

## 固定主 run

- 输入/动作/环境：`feature-v3/action-v2`、`tenhou-v3-ron-context`、`reward-v1`；初始化完全使用冻结 BC-v4 actor，critic value head 重置。
- 配置：`configs/train_ppo_v4.yaml`；seed 42；10 epochs × 精确 10,000 learner transitions = 100,000 learner transitions；每轮 batch 512、4 update epochs、Adam lr `1e-4`、gamma `0.99`、lambda `0.95`、clip `0.2`。
- 对手池按每个非 learner seat 独立抽样：冻结 BC-v3 `0.50`、冻结 BC-v4 `0.25`、Rule-v1 `0.25`；sampling、T=1。两 BC checkpoint 分别通过 `opponent_checkpoints` 显式绑定，和 PPO 初始化 checkpoint 区分；采样和环境 seed 固定在 config 中。
- PPO policy 不施加 BC KL anchor（beta 0）；不做 value-only warmup；reward clip 0；不启用 attribution。
- 采用 rollout-v1：当前对手池路径逐局收集，rollout-v2 的批量自博弈实现不会用于该路径。
- 输出独立存放于 `experiments/ppo_v4/seed42_100k`，从不覆盖 BC-v4；不续跑已有 PPO checkpoint。

## 启动前 pilot

以同一 schema、模型、对手池和优化器配置，临时执行 1 epoch × 512 learner transitions，输出到独立 pilot 目录。仅核验 BC actor 初始化一致、loss/梯度/采样合法、trajectory buffer 与 checkpoint 可写，并测该路径吞吐。pilot 不属于正式 100k 样本，不与主 run 合并；异常即停止并修复，不开始主 run。

## 预注册强度评估

- 方案：冻结 PPO checkpoint greedy/T=1 对 3×冻结 BC-v4；镜像 arm 为冻结 BC-v4 对 3×同一 BC-v4。candidate 和 mirror 使用相同 seed/seat，席位按 `game_index % 4` 轮换。
- 两 arm 各 20,000 局，seed 620000–639999；主效应为逐 seed candidate seat rank 减 mirror seat rank；95% CI 使用 PCG64 bootstrap 10,000 次，bootstrap seed 10320。
- GO：配对顺位差均值 `<= -0.020`、95% CI 上界 `< 0`、镜像均位在 `1.5±0.02`、两 arm 各 20,000 局全部有效且非法动作 0。任一项不满足则筛选未通过，不追加样本、不调门槛。
- 通过只允许启动另一个独立 seed 的确认 run；单 seed 结果不直接替换 BC-v4 强度基线。
- 评估预计约 7–9 小时：最近同硬件/16-worker/20k-per-arm 基准总耗时 24,073.529 秒（约6小时41分），PPO policy 额外 value head 的影响未知。按项目约束，此类 >20 分钟跑批不设时间上限且不实时监控，完成后等待用户通知。

## 时长规则与偏差处理

项目没有相同 PPO 对手池路径的可用吞吐记录。pilot 预计 2–8 分钟（首轮粗估；可能因 episode 长度与 CUDA 同步而偏离），记录实际 learner transitions、环境步、游戏数和耗时，再据实估计 100k 主 run。若主 run 预计超过 20 分钟，不限时、不实时监控。训练中不以 CPU/GPU 占用率单独判成功；记录 transitions/s 与总耗时。
