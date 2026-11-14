# Phase 10.2.3：延长 PPO 回报时间尺度的单变量筛选

预注册日期：2026-11-13（Asia/Shanghai）。状态：训练启动前冻结。强度锚点仍为 BC-v4；直接对照是 Phase 10.2.2 独立 value trunk PPO。

## 机制与证据

Phase 10.2.2 相对 BC-v4 的 10k 配对顺位差为 +0.0844，95% CI [+0.0568,+0.1122]，NO-GO。副露10.0315/局对镜像4.6929、放铳1.6025对1.3615、立直2.2167对2.5410。validation 策略漂移虽比共享 trunk 小，强度仍下降。

当前 reward-v1 将分数变化乘0.001（千点记1），立直动作立即奖励−1，终局顺位奖仅为 `(2,1,-1,-2)`。`gamma=.99`、`lambda=.95` 作用于 learner 每次决策。Phase 10.2.2 的checkpoint记录100,000 learner transitions、163条 learner 轨迹，平均每条613.5次决策；从约613步前看，终局顺位奖的折扣因子 `.99^613≈.0021`，直接 GAE 传递因子 `(.99×.95)^613` 接近零。这是训练目标与顺位评估不一致的可检验风险；它与行为变化相符，但不是已确证的唯一因果解释。

## 唯一变量

只改 `training.gamma: 0.99 → 1.0`，让 critic 目标对应未折扣的未来 reward；`lambda=.95`、reward-v1、value_separate_trunk=true、BC anchor=0、训练和对手池等其余设置均与 Phase 10.2.2 一致。GAE 的直接传递仍按 `.95^d` 衰减，因此本实验检验的是 gamma 的效应，不能单独检验完整长程信用分配。配置：`configs/train_ppo_v4_gamma1.yaml`，独立输出目录 `experiments/ppo_v4_gamma1/seed42_100k/`。

## 冻结训练与评估

- 从同一 BC-v4 best checkpoint 初始化 actor，critic 独立 trunk从 BC trunk复制、value head重置；seed42；10×10,000 learner transitions，总100,000；同环境 seed、对手池 seed。
- feature-v3/action-v2、tenhou-v3-ron-context/reward-v1；对手 BC-v3 50%、BC-v4 25%、Rule-v1 25%，sampling/T=1。
- Adam lr1e-4、batch512、4 update epochs、clip0.2、vf coef0.5、ent coef0.01、lambda0.95；无BC anchor、warmup、reward clip。
- 固定使用 epoch10/latest checkpoint；不根据validation top-1或中途对局选择。
- 强度评估：candidate PPO对3×BC-v4；mirror BC-v4对3×BC-v4；seed `660000–669999`，10,000对、每臂10,000局、席位轮换，Greedy/T=1、16 workers；PCG64 bootstrap 10,000次，seed10322。
- GO：paired rank difference（PPO−mirror）`<= -0.020`、95% CI上界`<0`、mirror mean rank `1.5±0.02`、两臂全部有效且非法动作0。否则NO-GO；不追加样本、不临时调门槛。单 seed GO 仍需独立训练 seed确认才可晋升。

## 时长与解释

上轮同预算训练约569秒，预计本轮9–12分钟；10k/臂同runner正式评估耗时5613.693秒，预计90–120分钟。训练完成先核对总步数、非法率和checkpoint身份，再启动正式评估；评估超过20分钟，不设时间上限、不实时监控，等待自然完成及用户通知。

若此实验仍 NO-GO，不能仅据本次推断 lambda、reward scale 或 BC anchor 无效；它们均未改变。训练 reward 和 validation gold top-1只作诊断，强度结论由预注册配对门给出。
