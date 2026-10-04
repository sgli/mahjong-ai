# Phase 10.2：PPO / RL 闭环验证与 Pilot 指导规格

> **文档性质：AI Coding Agent 直接执行规格**
>
> 本阶段必须严格按照本文执行。除非本文明确允许，不得自行扩大范围、改变项目架构、修改既有策略定义或直接进入 Phase 10.3 大规模 PPO。

---

# 1. 阶段结论

Phase 10.1.1 已完成验收，可以进入下一阶段。

当前基线：

- Dataset：`decision-v2`
- BC Model：`bc-v2.1`
- BC Best Checkpoint：`experiments/bc_v2.1/checkpoints/best.pt`
- Feature：`feature-v1`
- Feature dimension：`234`
- Action Space：`270`
- Environment：`MahjongEnv`
- Rules：以当前项目实际实现及测试为准
- PPO：已有基础实现，但目前仍属于小规模训练骨架

Phase 10.1.1 已验证：

- Fast Encoder 与旧 Encoder bitwise identical
- Batch Feature Encoding 已完成
- Batch Legal Mask 已完成
- Arrow → Python 转换已显著减少
- DataLoader / prefetch / pin_memory / non_blocking 已完成
- Resume / latest / best checkpoint 已完成 BC 训练链路
- Deterministic shard shuffle 已完成
- Validation sampling 已完成
- BC learning curve 已完成
- Greedy / Sampling benchmark 已完成
- Rule baseline 已完成
- Candidate / Opponent 双侧指标已完成
- 全量 BC-v2.1 已训练 5 epoch
- 最优 checkpoint 为 epoch 2
- full validation / test 已完成
- BC-v2.1 已成为下一阶段 RL 的初始化模型

因此：

```text
Phase 10.1.1 PASS
        ↓
Phase 10.2 PPO / RL 闭环验证与 Pilot
        ↓
如果全部通过
        ↓
Phase 10.3 Large-scale PPO / Self-play
```

**重要：Phase 10.2 的目标不是训练出最终强 AI，而是证明 RL 闭环是正确、稳定、可恢复、可评估的。**

---

# 2. 为什么不能直接进入大规模 PPO

当前 PPO 实现已经具备：

- GAE
- clipped PPO objective
- value head
- self-play trajectory collection
- legal action mask
- reward attribution
- 基础 PPO update

但当前实现仍存在明显的工程缺口：

1. `train_ppo.py` 没有完整 PPO `--resume` 机制。
2. 当前 PPO checkpoint 主要保存模型权重和训练结果，不能作为完整 PPO training state 恢复。
3. 没有 PPO 专用的完整 checkpoint schema。
4. 没有验证 BC trunk/action head 初始化后与 BC 模型输出严格一致。
5. PPO metrics 缺少 `approx_kl`、`clip_fraction` 等关键诊断指标。
6. 当前 advantage 是按 mini-batch 单独 normalization，需要明确并固定语义。
7. 当前 rollout / PPO buffer 使用大量 Python list + `torch.stack`，适合 Pilot，不适合 Phase 10.3 大规模训练。
8. `num_envs` 当前仍然是顺序运行环境，不是真正的并行环境。
9. 当前 PPO evaluation 不是正式 Benchmark，只是小规模 self-play evaluation。
10. Reward attribution / reward conservation 尚需要专门 regression tests。
11. 尚没有完整的 PPO interruption → resume → deterministic equivalence 验收。

因此 Phase 10.2 必须先解决这些问题。

---

# 3. 本阶段禁止事项

Phase 10.2 禁止：

- 修改 `feature-v1` 的语义
- 修改 `action-v1` / 270 action schema
- 修改 BC-v2.1 checkpoint 本身
- 为了 Benchmark 提升而修改 Mahjong Environment
- 同时大幅修改模型结构和 PPO 算法
- 直接训练 100M / 1B 级 PPO steps
- 直接做大规模 multi-GPU
- 直接引入 Ray / JAX / distributed training
- 直接建立 TB 级 feature cache
- 用 Benchmark 结果反向修改规则
- 删除已有测试或历史实验结果
- 用“训练跑通”代替正确性验收
- 未通过 Phase 10.2 DoD 就自动进入 Phase 10.3

如果发现 Environment / Rules 有真实 bug：

> 修复 bug，但必须增加 regression test，并明确记录原因；不得为了让 PPO reward / benchmark 更好而改变规则。

---

# 4. P0：BC-v2.1 → PPO 初始化正确性

## 4.1 默认 BC checkpoint

Phase 10.2 默认使用：

```text
experiments/bc_v2.1/checkpoints/best.pt
```

不要默认使用旧 `bc-v1` 或 `bc-v2`。

## 4.2 初始化原则

PPO model：

```text
BC trunk
   ↓
BC action head
   ↓
保留 BC policy

Value head
   ↓
重新初始化
```

BC action policy 在初始化瞬间必须保持不变。

## 4.3 必须增加测试

同一个 observation、同一个 legal mask：

```text
BC model logits
        ==
PPO model logits after BC initialization
```

要求：

- bitwise equal 优先
- 至少 `torch.equal()`
- 如果由于 dtype / device 原因无法 bitwise equal，必须说明原因并使用严格 `allclose`

Value head 不要求与 BC 相同，因为 BC 没有训练好的 value head。

建议新增：

```text
 tests/test_ppo_initialization.py
```

---

# 5. P0：建立完整 PPO Checkpoint

PPO checkpoint 必须与普通 BC checkpoint 区分。

建议：

```text
experiments/ppo_v1/checkpoints/
    latest.pt
    best.pt
    epoch_001.pt
    epoch_002.pt
    ...
```

每个 PPO checkpoint 至少保存：

```text
model_state_dict
optimizer_state_dict

epoch
global_step

best_metric
best_metric_name

config
metrics_history

seed
env_seed

python_rng_state
torch_cpu_rng_state
torch_cuda_rng_state

feature_version
action_schema_version
environment_version
reward_version
model_version

git_commit
```

如果使用 AMP：

```text
GradScaler state
```

如果未来增加 scheduler：

```text
scheduler_state_dict
```

也必须纳入 checkpoint。

---

# 6. P0：PPO Resume

CLI 必须支持：

```bash
python scripts/train_ppo.py \
    --config configs/train_ppo_v1.yaml \
    --resume experiments/ppo_v1/checkpoints/latest.pt
```

Resume 后必须恢复：

```text
model
optimizer
epoch
global_step
best metric
RNG
training configuration
```

不能只加载 model weights。

## 6.1 Resume 验收

至少构造：

```text
Run A:
    连续训练 N 个 epoch

Run B:
    训练到 K epoch
    保存 checkpoint
    重新启动进程
    resume
    继续到 N epoch
```

在相同 seed、相同环境 seed、相同配置下，比较：

- model parameters
- optimizer state
- epoch
- global step
- training metrics
- RNG continuation

目标：

```text
Run A ≈ Run B
```

对于确定性路径，应尽量做到严格一致。

如果 GPU / CUDA 非完全确定导致无法 bitwise identical，必须明确记录允许误差以及原因。

建议新增：

```text
 tests/test_ppo_checkpoint.py
 tests/test_ppo_resume.py
```

---

# 7. P0：Environment Regression Gate

在 PPO Pilot 前，必须确认 Environment 可以作为 RL 环境使用。

至少覆盖：

### 基本流程

- start game
- start kyoku
- tsumo
- dahai
- end kyoku
- end game

### 副露 / 杠

- chi
- pon
- ankan
- daiminkan
- kakan

### 和牌 / 立直

- riichi
- tsumo
- ron
- chankan

### 流局 / 特殊结束

- exhaustive draw
- kyuushu
- nagashi mangan
- four-wind abort
- four-riichi abort
- four-kan abort

### 点数 / 场况

- honba
- kyotaku
- dealer continuation
- south round
- west round
- final settlement
- final ranks

### 非法动作

所有非法 action 必须被拒绝。

不要因为 PPO 需要某个行为就放宽 legal action validation。

---

# 8. P0：Replay ↔ Environment Cross Validation

使用真实 `.mjai.json` 牌谱，在真实决策点验证：

```text
ReplayState
    ↓
Environment state
    ↓
legal_actions
```

真实牌谱中的实际 action 必须出现在 Environment legal action 中。

至少覆盖：

- discard
- riichi
- chi
- pon
- ankan
- daiminkan
- kakan
- ron
- tsumo
- pass

如果发现 mismatch：

1. 保存 game_id
2. 保存 kyoku
3. 保存 event index
4. 保存 player seat
5. 保存 replay state
6. 保存 legal actions
7. 保存 recorded action
8. 分类为 data issue / replay issue / environment issue

然后增加 regression test。

---

# 9. P0：Environment Version

必须明确一个 RL Environment version，例如：

```text
environment_version: tenhou-v2-rl
```

具体版本字符串可以由项目当前实际情况决定，但必须：

- 固定
- 写入 config
- 写入 checkpoint
- 写入实验结果
- 写入 benchmark

同时检查 `RulesConfig` 与 Environment 实际 wiring 是否一致。

**不要仅根据注释判断某条规则已经实现。必须以代码 + 测试为准。**

如果某个 RulesConfig flag 尚未真正接入 end-to-end：

- 明确记录
- 暂不声称完整实现
- 增加 TODO / regression test

---

# 10. P0：Reward-v1 固化

Phase 10.2 第一版 Pilot 固定 reward，不要边训练边改 reward。

默认 reward：

```yaml
reward_version: reward-v1

score_delta_scale: 0.001
placement_bonus:
  - 2.0
  - 1.0
  - -1.0
  - -2.0
```

语义：

```text
score delta reward
        +
terminal placement bonus
```

## 10.1 Reward regression

必须增加测试：

- 每个 seat 的 score delta 正确
- ron payment 正确归属
- tsumo settlement 正确归属
- dealer continuation 不产生错误 reward
- honba / kyotaku 不重复计入
- pending reward 能正确 flush
- terminal placement bonus 只在终局产生
- 每个 seat 的 reward trajectory 与最终 score delta 能对应

特别关注当前实现的：

```text
attribute_step_rewards()
flush_terminal_rewards()
```

不要假设当前 reward attribution 已经完全正确。

建议新增：

```text
 tests/test_ppo_reward.py
```

---

# 11. P0：Trajectory Schema

不要继续让 trajectory 永久依赖松散 Python dict。

Phase 10.2 至少明确 schema，建议结构：

```text
TrajectoryStep
    features
    action_id
    legal_mask
    old_log_prob
    value
    reward
    done
```

Trajectory / Episode metadata 建议包含：

```text
episode_id
seat
step
seed
policy_version
feature_version
action_schema_version
environment_version
reward_version
```

Phase 10.2 不要求立即做极致性能优化，但必须把数据语义固定下来。

---

# 12. P0：GAE 正确性

保持当前：

```text
GAE(γ, λ)
```

但必须补齐 regression tests：

- 单步 terminal
- 多步 non-terminal
- terminal bootstrap = 0
- episode boundary
- known hand-calculated example
- zero reward
- positive reward
- negative reward

必须确保：

```text
advantage
return
```

计算不跨 episode 泄漏。

---

# 13. P0：PPO Ratio / Clipping 正确性

核心：

```text
ratio = exp(new_log_prob - old_log_prob)
```

必须测试：

- ratio = 1
- ratio > 1 + clip_eps
- ratio < 1 - clip_eps
- positive advantage
- negative advantage

并验证：

```text
min(ratio * A,
    clip(ratio) * A)
```

符号方向不能写反。

当前 `ppo_loss()` 已有基础实现，Phase 10.2 要做的是把它固化成 regression gate，而不是重新设计 PPO objective。

---

# 14. P1：PPO Metrics

当前 metrics 不足以判断 PPO 是否健康。

必须增加：

```text
policy_loss
value_loss
entropy
mean_ratio
approx_kl
clip_fraction
mean_reward
mean_return
advantage_mean
advantage_std
episode_count
episode_length
steps
illegal_rate
```

其中：

### approx_kl

至少采用一种明确、稳定的定义，例如：

```text
approx_kl = mean(old_log_prob - new_log_prob)
```

或者项目明确选择的等价近似，但必须固定并写入文档。

### clip_fraction

定义为：

```text
fraction(|ratio - 1| > clip_eps)
```

必须增加单元测试。

---

# 15. P1：Advantage Normalization

当前代码是：

```text
每个 mini-batch 单独 normalization
```

Phase 10.2 必须明确语义并固定。

推荐：

```text
每个 rollout buffer 完成 GAE 后
↓
全 buffer normalization
↓
再进入 PPO mini-batch update
```

即：

```text
advantages = (advantages - mean) / (std + 1e-8)
```

但 mean/std 应基于整个 rollout buffer，而不是每个 mini-batch。

如果最终决定保持当前 mini-batch normalization，也必须写入 `TRAINING_SPEC` / Phase 10.2 文档并通过测试；不得隐式变化。

推荐采用 rollout-wide normalization。

---

# 16. P1：PPO v1 配置

新建：

```text
configs/train_ppo_v1.yaml
```

推荐初始配置：

```yaml
checkpoint_dir: "experiments/ppo_v1"
bc_checkpoint: "experiments/bc_v2.1/checkpoints/best.pt"
dataset_version: "decision-v2"
feature_version: "feature-v1"
action_schema_version: "action-v1"
environment_version: "tenhou-v2-rl"
reward_version: "reward-v1"
model_version: "ppo-v1"

model:
  hidden_sizes: [256, 256]

training:
  seed: 42
  env_seed: 42
  device: "cuda"

  num_envs: 4
  epochs: 5
  steps_per_epoch: 5000

  batch_size: 1024
  update_epochs: 4

  gamma: 0.99
  lam: 0.95
  clip_eps: 0.2

  vf_coef: 0.5
  ent_coef: 0.01
  lr: 0.0001

  max_episode_steps: 5000

evaluation:
  games: 1000
```

这些参数只是 Pilot 起点，不是最终最优超参数。

AI Agent 不得因为看到这些值就自动进行大规模超参搜索。

---

# 17. P1：Pilot 训练规模

Phase 10.2 不做大规模 PPO。

推荐分三档：

```text
Pilot-A: 2,000 steps
Pilot-B: 5,000 steps
Pilot-C: 10,000 steps
```

每一档只有在前一档健康时才继续。

第一目标不是赢棋，而是：

```text
loss finite
value loss finite
entropy finite
approx_kl finite
clip_fraction finite
no NaN
no Inf
illegal_rate = 0
Environment 无 crash
checkpoint 正常保存
resume 正常
```

---

# 18. Pilot 健康指标

重点观察：

```text
policy_loss
value_loss
entropy
approx_kl
clip_fraction
mean_reward
mean_return
advantage_mean
advantage_std
illegal_rate
```

警戒信号包括：

- NaN / Inf
- entropy 快速塌缩到接近 0
- ratio 长期严重偏离 1
- approx_kl 异常增大
- clip_fraction 长期接近 1
- value loss 爆炸
- reward 极端漂移
- illegal_rate > 0
- Environment episode 无法正常结束

发现异常时：

> 停止扩大训练规模，先定位问题。

---

# 19. P1：PPO Evaluation

当前 `train_ppo.py` 的 evaluation 只是小规模 self-play，不应作为正式强度 Benchmark。

Phase 10.2 需要把 PPO Pilot 接入已有 Benchmark 框架。

至少比较：

```text
PPO-v1 Pilot
BC-v2.1
Rule
Random
```

首要模式：

```text
Greedy
```

辅助模式：

```text
Sampling
```

Benchmark 必须固定：

- rules
- seed scheme
- candidate seat rotation
- game count
- opponent composition
- checkpoint
- environment version
- reward version（如需要记录）

不要因为 PPO 结果不好而修改 Benchmark 规则。

---

# 20. Benchmark 指标

至少记录：

```text
mean rank
rank distribution
win rate
deal-in rate
riichi rate
call rate
average win points
score change
illegal rate
```

Primary：

```text
mean rank
```

Secondary：

```text
rank distribution
win rate
deal-in rate
riichi rate
call rate
```

训练 reward 不能作为唯一强度指标。

---

# 21. P1：Self-play 的范围

Phase 10.2 可以继续使用：

```text
current PPO vs current PPO
```

作为 RL loop correctness test。

但不要把它当成最终 Self-play 架构。

Phase 10.3 再建立正式 opponent pool：

```text
bc-v2.1
rule-v1
random-v1
历史 PPO checkpoint
current PPO
```

并支持不同策略组合。

---

# 22. P2：性能优化暂缓

Phase 10.2 不要求解决：

- 真正并行 MahjongEnv
- multiprocessing rollout
- vectorized environment
- batch policy inference
- GPU rollout
- Ray
- JAX
- multi-GPU

原因：

```text
先证明 RL 正确
        ↓
再优化 RL throughput
```

Phase 10.3 再进行大规模性能工程。

---

# 23. 推荐代码结构

建议逐步整理为：

```text
src/mahjong/training/
├── ppo.py
├── ppo_buffer.py
├── rollout.py
├── ppo_checkpoint.py
└── checkpoint.py
```

职责：

### rollout.py

```text
Environment
    ↓
Policy
    ↓
Trajectory
```

### ppo_buffer.py

```text
Trajectory
    ↓
GAE
    ↓
Returns
    ↓
Batch
```

### ppo.py

```text
PPO loss
PPO update
training loop
```

### ppo_checkpoint.py

```text
save
load
resume
RNG
optimizer
metadata
```

不要求一次性重构全部代码。

优先保证测试与语义，再做结构清理。

---

# 24. 必须新增 / 完善测试

建议至少：

```text
tests/test_ppo_initialization.py
tests/test_ppo_checkpoint.py
tests/test_ppo_resume.py
tests/test_ppo_rollout.py
tests/test_ppo_reward.py
tests/test_ppo_metrics.py
tests/test_ppo_gae.py
```

重点覆盖：

```text
BC initialization
checkpoint save/load
resume
determinism
trajectory legality
reward conservation
GAE
PPO ratio
PPO clipping
approx KL
clip fraction
advantage normalization
```

---

# 25. PPO Checkpoint 建议格式

不要复用 BC checkpoint 的语义。

建议：

```python
{
    "checkpoint_type": "ppo",
    "checkpoint_version": "ppo-v1",

    "model_state_dict": ...,
    "optimizer_state_dict": ...,

    "epoch": ...,
    "global_step": ...,

    "best_metric": ...,
    "best_metric_name": ...,

    "config": ...,
    "metrics_history": ...,

    "seed": ...,
    "env_seed": ...,

    "python_rng_state": ...,
    "torch_cpu_rng_state": ...,
    "torch_cuda_rng_state": ...,

    "dataset_version": "decision-v2",
    "feature_version": "feature-v1",
    "action_schema_version": "action-v1",
    "environment_version": "tenhou-v2-rl",
    "reward_version": "reward-v1",
    "model_version": "ppo-v1",

    "git_commit": ...,
}
```

字段可按实际代码调整，但语义必须保留。

---

# 26. Experiment Metadata

每次 PPO Pilot 必须记录：

```text
git commit
config
seed
env seed
model version
feature version
action schema version
environment version
reward version
BC checkpoint
steps
batch size
learning rate
gamma
lambda
clip epsilon
vf coef
entropy coef
```

训练结果必须能够回答：

> 这个 PPO 模型究竟是从哪个 BC checkpoint、哪个代码版本、哪个 Environment、哪个 Reward、哪个 seed 训练出来的？

---

# 27. Phase 10.2 实施顺序

严格按以下顺序：

```text
1. 阅读全部现有 docs
        ↓
2. 阅读现有 PPO / Environment / Reward / Checkpoint 代码
        ↓
3. P0 BC initialization test
        ↓
4. P0 PPO checkpoint
        ↓
5. P0 PPO resume
        ↓
6. P0 Environment regression
        ↓
7. P0 Replay ↔ Environment cross validation
        ↓
8. P0 Reward regression
        ↓
9. P0 GAE / PPO ratio / clipping regression
        ↓
10. P1 PPO metrics
        ↓
11. P1 advantage normalization 固化
        ↓
12. train_ppo_v1.yaml
        ↓
13. Pilot-A 2k steps
        ↓
14. Pilot-B 5k steps
        ↓
15. Pilot-C 10k steps
        ↓
16. Benchmark
        ↓
17. PHASE10.2_RESULT.md
        ↓
18. STATUS.md
```

任何 P0 失败，都不得进入下一步的大规模训练。

---

# 28. Definition of Done

Phase 10.2 只有满足以下条件才能标记 PASS：

```text
[ ] bc-v2.1 best.pt 正常加载
[ ] PPO trunk/action head 与 BC 初始化严格一致
[ ] value head 正常初始化

[ ] Environment regression PASS
[ ] Replay ↔ Environment cross-validation PASS
[ ] environment_version 明确

[ ] reward-v1 明确
[ ] reward attribution regression PASS
[ ] terminal reward regression PASS

[ ] GAE regression PASS
[ ] PPO ratio regression PASS
[ ] PPO clipping regression PASS

[ ] trajectory schema 明确
[ ] PPO metrics 完整
[ ] approx_kl 已实现
[ ] clip_fraction 已实现
[ ] advantage normalization 语义固定

[ ] PPO checkpoint 完整
[ ] optimizer state 已保存
[ ] RNG state 已保存
[ ] metadata 已保存

[ ] PPO resume PASS
[ ] fixed seed reproducibility PASS

[ ] Pilot-A PASS
[ ] Pilot-B PASS
[ ] Pilot-C PASS

[ ] 无 NaN / Inf
[ ] illegal_rate = 0
[ ] Environment 无异常终止
[ ] checkpoint 可正常加载
[ ] resume 可正常继续

[ ] PPO Benchmark 完成
[ ] BC-v2.1 / PPO / Rule / Random 对比完成
[ ] Greedy benchmark 完成
[ ] Sampling benchmark 完成

[ ] docs/PHASE10.2.md 已更新/保存
[ ] docs/PHASE10.2_RESULT.md 已生成
[ ] STATUS.md 已更新
```

---

# 29. Go / No-Go

最终决策必须是：

```text
                    BC-v2.1
                       │
                       ▼
          Phase 10.2 PPO Pilot
                       │
              ┌────────┴────────┐
              │                 │
             FAIL              PASS
              │                 │
              ▼                 ▼
        修复基础设施       Phase 10.3
        / Environment      Large-scale PPO
        / Reward                │
              │                ▼
              └──────→   Self-play
                               │
                               ▼
                         Opponent Pool
                               │
                               ▼
                           Benchmark
```

**只有 Phase 10.2 所有 P0 / DoD 通过，才能宣布进入 Phase 10.3。**

---

# 30. Phase 10.3 预览

Phase 10.3 才处理：

- large-scale PPO
- 并行 Environment
- vectorized rollout
- batch inference
- multiprocessing
- opponent pool
- historical checkpoints
- large-scale self-play
- multi-GPU（如 profiling 证明必要）
- 更大模型
- 更长训练
- PPO hyperparameter sweep
- 正式长期 benchmark

Phase 10.2 不提前实现这些内容。

---

# 31. AI Coding Agent 执行规则

AI Coding Agent 必须遵守：

1. 开始前先阅读：
   - `PROJECT_PLAN.md`
   - `ARCHITECTURE.md`
   - `DATA_SPEC.md`
   - `MODEL_SPEC.md`
   - `ENVIRONMENT_SPEC.md`
   - `TRAINING_SPEC.md`
   - `CODEX_RULES.md`
   - `PHASE10.1.1.md`
   - `PHASE10.1.1_RESULT.md`
   - 当前 PPO / Environment / Checkpoint / Benchmark 实现

2. 不得推翻现有架构。

3. 不得修改 `feature-v1` 语义。

4. 不得修改 `action-v1` / 270 action schema。

5. 不得修改 `bc-v2.1` checkpoint。

6. 不得为了 benchmark 改规则。

7. 不得猜测麻将规则；规则问题必须以代码、测试、项目规则文档和真实牌谱交叉验证。

8. 每个核心修改必须增加 regression test。

9. 优先修复 P0，再进行 Pilot。

10. Pilot 失败时停止扩大规模，先定位原因。

11. 不得自动进入 Phase 10.3。

12. 不得删除历史实验结果。

13. 最终必须生成：

```text
 docs/PHASE10.2_RESULT.md
```

并报告：

```text
修改内容
测试结果
Pilot 结果
Benchmark 结果
发现的问题
剩余风险
是否 PASS
是否允许进入 Phase 10.3
```

---

# 32. 最终目标

Phase 10.2 不是为了证明：

> “PPO 已经比 BC 强。”

而是为了证明：

> **BC-v2.1 → Mahjong Environment → Rollout → Reward → GAE → PPO Update → Checkpoint → Resume → Benchmark 这一整条 RL 闭环已经正确、稳定、可复现。**

只有这个闭环被证明可靠，才值得投入大规模 PPO / Self-play 的计算资源。
