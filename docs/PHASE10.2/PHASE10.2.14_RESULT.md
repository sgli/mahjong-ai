# Phase 10.2.14 结果：纯顺位 vs 顺位主导混合目标

## 预注册

见`PHASE10.2.14.md`。比较两种回报目标对greedy PPO最终顺位的影响；两臂从相同BC-v4 actor、相同训练seed/env seed、相同100k预算与超参数开始，仅回报目标与匹配的公开critic初值不同。评估使用新seed 880000–881999，每个候选各2000局及同seed BC-v4镜像。

## 执行与结果

两臂训练身份核验均为PASS，命令`.venv/Scripts/python.exe scripts/verify_phase10_2_14_training.py`。纯顺位：100,000 learner transitions、400,209环境步、165完整局、795 optimizer updates、1轮KL early-stop、非法0，实际12分15.095秒，best SHA `e1a571f89c0929e03ed3dad5958dd1b4ed64c069d968b7cd3f91fe9e3fdbd33d`。混合：100,000 learner transitions、400,164环境步、161完整局、800 optimizer updates、无KL early-stop、非法0，实际12分14.482秒，best SHA `5cc934b668e5bb911edbd846efc80a21de8312dfd1dbbd0fe89bd141bc361b7e`。两者使用同一BC-v4初始化SHA、seed43/env630000和预注册设置。

独立评估核验命令`.venv/Scripts/python.exe scripts/verify_phase10_2_14_eval.py`，产物`experiments/ppo_v4_rank_compare/eval_seed880000_2000_v1/independent_verification.json`状态PASS：纯、混合和BC-v4镜像各2000局，seed880000–881999逐局完整、席位匹配、非法0；镜像均位1.5215，离1.5为0.0215，低于normal99-v2容差0.06440。评估总耗时1693.419秒（28分13.419秒）；全流程自17:28:56至18:21:42，约52分46秒。

| 指标 | 纯顺位 | 顺位主导混合 | BC-v4镜像 |
|---|---:|---:|---:|
| 平均顺位（越低越好） | 1.4845 | 1.5175 | 1.5215 |
| 对镜像配对差 | −0.0370 | −0.0040 | — |
| 对镜像配对95% CI | [−0.0875, +0.0120] | [−0.05651, +0.0495] | — |
| 预注册小批GO | 否 | 否 | sanity通过 |

纯顺位减混合目标均差为−0.0330，表面点估计偏向纯顺位，但按预注册bootstrap seed 10360、PCG64、10,000次游戏级配对bootstrap重算的95% CI为[−0.0865,+0.0210]，跨0且未达到赢家门；因此**目标间胜负不确定**。纯顺位和混合目标各自对镜像的CI上界也都高于0，两者都未通过≤−0.020且CI上界<0的小批GO门。**不启动正式10,000对强度评估，BC-v4继续作为强度基线。**

执行偏差：原评估脚本误将目标间主比较bootstrap seed设为10362，与预注册10360不符。未重跑昂贵对局；审计脚本从冻结的原始逐局数据按预注册seed 10360独立重算，并更新`result.json`，纠正后结论仍为INCONCLUSIVE。原脚本已修正，后续执行使用seed10360。此偏差不改变对局数据/排序，只改变bootstrap随机重采样序列。

本比较只有一个训练seed/目标；2000局CI反映该训练checkpoint下的评估不确定性，不能代替跨训练seed的复现。当前结果足以否决两臂直接进入正式强度评估，但不能证明两奖励目标等效，也不能认定纯顺位优越。

## 训练并行度说明

两份训练配置的`training.num_envs`均为1、`device`为CUDA；PPO采样器据此只创建一个`MahjongEnv`并在单个Python训练进程中逐环境采样（`src/mahjong/training/ppo.py:412,511-529`）。评估脚本的16进程池仅用于稍后的对局评估，不作用于训练（`scripts/compare_phase10_2_14.py:23-24`、`scripts/benchmark_parallel.py:177-185`）。因此用户观察到的训练CPU低利用率符合当前配置。该设置是我沿用既有候选配置时的遗漏；现在不改正在执行的任一臂，以免造成两方案设置不一致。

## 当前状态

2026-11-14，Phase10.2.14顺序pipeline完整结束；监督日志`experiments/ppo_v4_rank_compare/pipeline.log`。两训练臂健康身份核验PASS，三臂2000局独立核验PASS；但目标间与对BC-v4镜像的区间均未过门，阶段结论INCONCLUSIVE/NO_GO，不进入正式10k。
