# Phase 10.2.2：隔离 critic trunk 的 PPO 单变量筛选

预注册日期：2026-11-13（Asia/Shanghai）  
状态：方案冻结；仅改变 critic 是否与 actor 共用 trunk。  
基线：Phase 10.2.1 seed42 PPO；强度锚点：冻结 BC-v4。

## 失败归因与可检验假设

Phase 10.2.1 的 20,000 配对强度评估明确 NO-GO。错误动作、schema 混用、reward 终局项和 learner-transition 截断的代码审查未发现明显实现错位。对固定的 decision-v3 validation 前 100,000 行做事后策略漂移分析（只作诊断，不参与选择或门槛）：epoch 1→10，BC/PPO greedy 一致率从 88.916% 降至 67.127%，PPO gold top-1 从 68.847% 降至 58.414%，BC→PPO KL 从 0.0522 升至 0.4010 nat；PPO 熵总体高于 BC-v4。

当前可检验机制是 **critic/value loss 经共享 trunk 改变 actor 表征**：初始 value head 随机、actor 和 critic 共用 trunk，`vf_coef=0.5`，价值回归梯度可直接更新共享 trunk。证据支持该机制为风险来源，但尚不能证明其为唯一根因；policy-gradient 与 entropy 项也会改变 actor。

## 唯一实验变量

启用 `model.value_separate_trunk: true`。模型初始化时仍从相同 BC-v4 trunk 复制初始 value trunk，value head 重新初始化；之后 critic 梯度仅更新独立 value trunk/head，actor trunk 只接收 policy 与 entropy 梯度。训练器、奖励、对手池和其余超参保持不变。原 Phase 10.2.1 配置与产物保持冻结。

配置：`configs/train_ppo_v4_critic_isolated.yaml`。初始化 BC-v4 SHA-256 `814c6eba4e163d126cff7bc27194f11612263c745328abd1b1d9ae94b6716937`。

## 固定训练与评估

- seed 42；10×10,000 learner transitions；总预算 100,000；same environment/opponent seeds as Phase 10.2.1。
- feature-v3/action-v2、tenhou-v3-ron-context/reward-v1；BC-v3 50%、BC-v4 25%、Rule-v1 25%，sampling/T=1。
- Adam `1e-4`、batch 512、4 update epochs、gamma `.99`、lambda `.95`、clip `.2`、value coef `.5`、entropy coef `.01`；无 BC anchor、无 warmup、无 reward clip。
- 预先固定使用 epoch 10/latest checkpoint，不从 validation/test 挑 checkpoint。
- 若训练完成，使用相同 paired runner 做 candidate PPO vs 3×BC-v4、mirror BC-v4 vs 3×BC-v4；**10,000 对** seeds `640000–649999`，Greedy/T=1、席位轮换，10,000次 PCG64 bootstrap、seed `10321`。GO 门仍为 paired rank difference `<= -0.020`、bootstrap 95% CI 上界 `< 0`、mirror mean rank `1.5±0.02`、两臂全局有效且非法动作 0。否则 NO-GO 并停止。

## 解释边界

若新模型在验证集上的 actor drift 更小但强度仍未过门，只能认为 trunk 隔离减弱漂移，不能把它称作策略提升；该机制单独判 NO-GO，不追加样本、不临时调超参。离线 100k validation 仅用于事后刻画漂移，不用于门槛、checkpoint 选择或调参。将配对数由20,000降至10,000是用户在强度评估启动前明确调整；效应门槛不变，置信区间会较宽，结论按10k样本解释，不与20k样本精度等同。

## 时长估算

正式 Phase 10.2.1 完成同预算 100k run 用时 572.8 秒（9分33秒），本 run 使用相同 rollout/transition 预算但 value trunk 增加约一份 trunk 计算，预计 **10–16 分钟**，不确定性主要来自 GPU/序列编码吞吐波动；完成后以本次实测校准。20k/臂 paired PPO 评估最近实测 11,161 秒（3小时06分）；按吞吐近似减半，10k/臂预计 **90–120分钟**。按约束，若强度评估正式启动，将不设时间上限、不实时监控，完成后等待用户通知。
