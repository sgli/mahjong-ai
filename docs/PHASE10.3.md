# Phase 10.3：Large-scale PPO / Self-play / Opponent Pool / Strength Validation

> **文档性质：AI Coding Agent 直接执行规格**
>
> 本阶段建立在 `Phase 10.2 = PASS` 的基础上。
> 本文不是建议清单，而是下一阶段的执行边界、工程目标、实验规范与验收标准。
> 除非本文明确允许，不得自行扩大范围、修改既有策略定义或为了 benchmark 结果修改麻将规则。

---

# 1. 阶段结论

Phase 10.2 已通过全部 DoD：

```text
BC-v2.1
   ↓
MahjongEnv
   ↓
Rollout
   ↓
Reward-v1
   ↓
GAE
   ↓
PPO Update
   ↓
Checkpoint
   ↓
Resume
   ↓
Benchmark
```

整条 RL 闭环已经完成正确性、可恢复性和基础稳定性验证。

因此：

```text
Phase 10.2 PASS
        ↓
Phase 10.3
Large-scale PPO / Self-play / Opponent Pool
        ↓
Strength Validation
        ↓
Phase 10.4 / 下一阶段
```

Phase 10.3 的核心目标已经从：

> “RL pipeline 是否正确？”

升级为：

> **“在正确的 RL pipeline 上，PPO 是否能够通过更大规模训练、自对弈和多样化对手真正获得稳定的策略增益？”**

---

# 2. Phase 10.2 已知基线

必须把以下结果作为 Phase 10.3 的固定历史基线，不得覆盖。

## 2.1 BC 基线

```text
model: bc-v2.1
feature: feature-v1
feature_dim: 234
action_space: 270
dataset: decision-v2
```

BC-v2.1 是 Phase 10.3 的主要初始化策略。

默认初始化：

```text
experiments/bc_v2.1/checkpoints/best.pt
```

## 2.2 PPO-v1 基线

Phase 10.2 最终 PPO：

```text
model: ppo-v1
steps: 10,891
seed: 42
environment: tenhou-v2-rl
reward: reward-v1
```

Benchmark：

```text
2000 games / opponent
seat = game_index % 4
seed = 42 + game_index
16 workers
```

## 2.3 已知 Benchmark 结果

### Greedy

```text
vs random : mean rank 0.0020
vs rule   : mean rank 0.6185
vs BC     : mean rank 1.0750
```

### Sampling

```text
vs random : mean rank 0.0150
vs rule   : mean rank 1.1360
vs BC     : mean rank 1.7315   ← 当前明显弱于 BC
```

因此 Phase 10.3 必须特别关注：

```text
PPO vs BC-v2.1
```

不能只因为：

```text
PPO > random
PPO > rule
```

就认为 PPO 已经成功。

---

# 3. Phase 10.3 核心目标

本阶段包含五个目标：

```text
P0  建立多 seed / 固定 Benchmark 基线
        ↓
P0  完成 RL rollout 性能工程
        ↓
P1  建立 Opponent Pool
        ↓
P1  进行中规模 → 大规模 PPO
        ↓
P1  验证 PPO 是否真正超过 BC-v2.1
```

最终不是单纯追求：

```text
training reward ↑
```

而是追求：

```text
独立 Benchmark
        ↓
PPO 在多个 seed / 多种对手下
表现出稳定的相对强度提升
```

---

# 4. 本阶段禁止事项

Phase 10.3 仍然禁止：

1. 修改 `feature-v1` 语义。
2. 修改 `action-v1` / 270 action schema，除非发现真实设计错误并单独建立版本迁移方案。
3. 修改 `bc-v2.1` checkpoint。
4. 为了提升 benchmark 修改 Mahjong Environment 规则。
5. 根据 PPO 输棋结果临时修改 reward。
6. 把 Benchmark 对手固定成弱对手后宣布 PPO 成功。
7. 只用单一 seed 宣布模型提升。
8. 删除 Phase 10.2 实验结果。
9. 覆盖旧 checkpoint。
10. 未完成 baseline 就开始大规模训练。
11. 在没有 profiling 证据时盲目引入 Ray/JAX/multi-GPU。
12. 同时大幅修改模型结构、reward、environment、benchmark，导致实验无法归因。

如果发现真实规则 bug：

```text
发现
 ↓
复现
 ↓
修复
 ↓
regression test
 ↓
重新跑受影响 benchmark
```

不得静默修改。

---

# 5. P0：Phase 10.3 Benchmark Baseline 固化

在任何大规模 PPO 前，先建立新的严格 baseline。

至少固定：

```text
BC-v2.1
PPO-v1 10.9k checkpoint
Rule
Random
```

并使用统一 Benchmark runner。

必须固定：

```text
rules
environment_version
action_schema_version
feature_version
seed scheme
candidate seat rotation
opponent composition
game count
mode
worker count
```

建议正式 Benchmark seed：

```text
seed = 42 + game_index
```

但 Phase 10.3 必须额外增加多 seed Benchmark。

推荐：

```text
benchmark seeds:
42
43
44
```

如果成本允许，再增加：

```text
45
46
```

## 5.1 两种模式

必须保留：

```text
Greedy   = primary
Sampling = secondary
```

不能因为 Sampling 对 PPO 不利就删除 Sampling。

---

# 6. P0：多 Seed 强度基线

Phase 10.2 只有 seed=42，因此 Phase 10.3 首先验证历史结论是否稳定。

至少运行：

```text
BC-v2.1 vs Rule
BC-v2.1 vs Random
PPO-v1 vs Rule
PPO-v1 vs BC-v2.1
```

至少：

```text
3 seeds
```

建议每个 seed：

```text
1000~2000 games / opponent
```

如果结果差异巨大，不得直接进入超大规模训练。

先分析 variance。

## 6.1 强度指标

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
average win points
score delta
illegal rate
```

同时记录置信区间或至少：

```text
mean
std
min
max
```

多 seed 结果。

---

# 7. P0：double_yakuman 决策门

Phase 10.2 已明确发现：

```text
double_yakuman = flag exists
但 Environment 尚未 end-to-end 接线
```

Phase 10.3 开始大规模训练前必须做一次明确决策：

```text
方案 A：补齐 double_yakuman
方案 B：确认 Tenhou target rule 下本阶段暂不支持，并固定记录
```

要求：

- 不得默认为“已经实现”。
- 如果补齐，必须增加 scoring/environment regression tests。
- 如果暂不补齐，必须在 `environment_version` / Phase 10.3 文档中明确记录。

**不能在大规模训练过程中途改变。**

---

# 8. P0：RL Rollout 性能 Profiling

Phase 10.2 已证明正确，但当前 PPO 仍然：

```text
num_envs = sequential
encoder = per-step
policy inference = per-step
trajectory = Python objects/list
```

Phase 10.3 必须开始性能工程。

第一原则：

> **先 profiling，再优化。**

至少记录：

```text
env_step_time
feature_encode_time
legal_mask_time
policy_forward_time
action_sample_time
trajectory_write_time
GAE_time
PPO_update_time
total_rollout_time
total_update_time
steps/sec
games/hour
GPU utilization
GPU memory
CPU utilization
```

如果 GPU profiling 无法从 PyTorch 得到可靠利用率，使用外部 `nvidia-smi` 或等价工具记录。

---

# 9. P1：并行 Environment

Phase 10.3 的第一项性能改造：

```text
sequential env
     ↓
parallel env
```

优先选择：

```text
multiprocessing / process pool
```

或项目实际 profiling 后证明更合适的方案。

暂不强制 Ray/JAX。

## 9.1 语义要求

并行版本必须与单环境参考实现一致：

```text
same seed
same initial state
same policy action sequence
        ↓
state transition identical
reward identical
final result identical
```

必须新增 regression test。

## 9.2 不允许

不能为了吞吐：

- 删除合法动作检查
- 删除 reward settlement
- 删除 environment reset
- 修改随机种子语义
- 修改 action mask

---

# 10. P1：Batch Policy Inference

当前：

```text
one env step
 ↓
one observation
 ↓
one model forward
```

目标：

```text
N envs
 ↓
collect observations
 ↓
batch encoder
 ↓
batch legal masks
 ↓
one policy forward
 ↓
N actions
```

必须保持：

```text
legal action mask = 100% correct
```

不能因为 batch inference 而让不同环境共享错误状态。

---

# 11. P1：Trajectory / Rollout Buffer 性能优化

Phase 10.2 的：

```python
Python list
torch.stack()
```

可以作为 correctness reference，但 Phase 10.3 不应继续作为主要大规模路径。

目标：

```text
preallocated tensor buffer
```

至少预分配：

```text
features
legal_masks
action_ids
old_log_probs
values
rewards
dones
advantages
returns
```

要求：

```text
reference implementation
        ==
optimized implementation
```

在固定随机轨迹上：

```text
action
reward
old_log_prob
value
advantage
return
```

均应一致或在明确允许的浮点误差内一致。

---

# 12. P1：Opponent Pool

Phase 10.3 正式建立 opponent pool。

最小池：

```text
random-v1
rule-v1
bc-v2.1
ppo-v1
current-ppo
```

后续加入：

```text
historical PPO checkpoints
```

例如：

```text
ppo-20k
ppo-50k
ppo-100k
```

不要只训练：

```text
current PPO vs current PPO
```

否则容易产生策略坍缩或过拟合当前策略。

---

# 13. Opponent Pool Sampling

每局开始时选择 opponent policy。

至少支持：

```text
uniform sampling
```

例如：

```text
random  10%
rule    20%
BC      30%
PPO     20%
history 20%
```

以上只是初始建议，不是最终固定比例。

比例必须写入 config，不能硬编码。

## 13.1 Candidate Seat

继续使用：

```text
candidate_seat = game_index % 4
```

保证公平轮换。

---

# 14. P1：PPO Training Curriculum

不要直接从 10k steps 跳到数亿 steps。

建议：

```text
Stage A
20k steps

Stage B
50k steps

Stage C
100k steps

Stage D
500k steps

Stage E
1M+ steps
```

每个阶段只有满足健康门禁才继续。

---

# 15. Stage A：20k Pilot-Scale PPO

目的：

```text
验证性能优化后的 PPO 与 Phase 10.2 reference implementation 一致
```

要求：

- loss finite
- no NaN/Inf
- entropy 正常
- approx_kl 正常
- clip_fraction 正常
- illegal_rate = 0
- environment 无异常终止
- checkpoint 正常
- resume 正常
- optimized rollout 与 reference 对齐

这一步不是为了证明强度。

---

# 16. Stage B：50k PPO

开始观察：

```text
PPO vs BC
PPO vs Rule
PPO vs historical PPO
```

每完成一个 checkpoint：

```text
保存
 ↓
固定 Benchmark
 ↓
记录 mean rank
```

不要只保存最终模型。

---

# 17. Stage C：100k PPO

100k 是第一个真正的策略质量观察点。

必须至少比较：

```text
BC-v2.1
PPO-20k
PPO-50k
PPO-100k
```

在相同 Benchmark seeds 下测试。

观察：

```text
mean rank
win rate
deal-in rate
riichi rate
call rate
score delta
```

特别关注：

```text
Sampling vs BC
```

如果 Sampling 继续明显恶化，需要分析：

```text
policy entropy
exploration
reward shaping
opponent overfitting
value estimation
```

不得第一时间修改 reward。

---

# 18. Stage D/E：500k → 1M+

只有在 100k 阶段表现健康时才进入。

目标：

```text
large-scale PPO
```

但每个 checkpoint 都必须可复现。

推荐 checkpoint：

```text
20k
50k
100k
200k
500k
1M
```

实际保存频率可按吞吐调整，但必须保留关键 milestone。

---

# 19. P1：多 Seed PPO Training

不能只训练：

```text
seed=42
```

正式强度结论至少使用：

```text
seed 42
seed 43
seed 44
```

每个 seed：

```text
独立 checkpoint
独立 metrics
独立 benchmark
```

最终报告：

```text
mean
std
best
worst
```

如果只有一个 seed 有提升，不能宣布稳定提升。

---

# 20. P1：正式 Strength Gate

Phase 10.3 最重要的验收不是：

```text
PPO training completed
```

而是：

> PPO 是否稳定超过 BC-v2.1？

## Gate A：Against Rule

PPO 应明显优于 Rule baseline。

## Gate B：Against BC

至少 Greedy 下：

```text
PPO mean rank < BC mean rank
```

并且多 seed 方向一致。

## Gate C：Sampling

Sampling 是重要的鲁棒性指标。

不要求第一版必须立刻全面超过 BC，但如果 Sampling 长期显著劣于 BC：

```text
Phase 10.3 = NOT READY FOR FINAL
```

需要进入诊断阶段，而不是继续无条件扩大训练。

## Gate D：Illegal

必须：

```text
illegal_rate = 0
```

---

# 21. 不使用训练 Reward 作为唯一成功标准

以下不能单独证明 PPO 变强：

```text
mean_reward ↑
value_loss ↓
policy_loss ↓
```

正式判断必须依赖：

```text
independent benchmark
```

而且 Benchmark 对手必须固定。

---

# 22. PPO Checkpoint 规范

继续沿用 Phase 10.2：

```text
checkpoint_type
checkpoint_version
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
RNG states
feature_version
action_schema_version
environment_version
reward_version
model_version
git_commit
```

额外增加：

```text
opponent_pool_version
opponent_pool_config
rollout_version
training_stage
```

这样才能回答：

> 这个 PPO checkpoint 到底是在什么 opponent pool、什么 rollout 实现、什么 seed 下训练出来的？

---

# 23. Experiment Tracking

每次实验至少记录：

```text
git commit
config
seed
env seed
BC checkpoint
initial checkpoint
training stage
steps
batch size
num envs
rollout implementation
opponent pool
opponent probabilities
learning rate
gamma
lambda
clip epsilon
vf coef
entropy coef
reward version
environment version
feature version
action schema version
```

必须保存：

```text
training metrics
benchmark metrics
checkpoint path
```

---

# 24. 推荐目录结构

逐步整理为：

```text
src/mahjong/training/
├── ppo.py
├── ppo_buffer.py
├── rollout.py
├── parallel_rollout.py
├── ppo_checkpoint.py
├── opponent_pool.py
└── checkpoint.py
```

Benchmark：

```text
scripts/benchmark.py
scripts/benchmark_parallel.py
```

训练：

```text
scripts/train_ppo.py
```

不要求一次性完成所有重构。

优先顺序：

```text
correctness
 ↓
reference parity
 ↓
profiling
 ↓
optimization
```

---

# 25. 新配置文件

建议：

```text
configs/train_ppo_v2.yaml
```

至少包含：

```yaml
checkpoint_dir: "experiments/ppo_v2"
bc_checkpoint: "experiments/bc_v2.1/checkpoints/best.pt"
feature_version: "feature-v1"
action_schema_version: "action-v1"
environment_version: "tenhou-v2-rl"
reward_version: "reward-v1"
model_version: "ppo-v2"
rollout_version: "rollout-v2"
opponent_pool_version: "pool-v1"

model:
  hidden_sizes: [256, 256]

training:
  seed: 42
  env_seed: 42
  device: "cuda"

  num_envs: 16
  steps_per_epoch: 20000
  batch_size: 2048
  update_epochs: 4

  gamma: 0.99
  lam: 0.95
  clip_eps: 0.2
  vf_coef: 0.5
  ent_coef: 0.01
  lr: 0.0001

  max_episode_steps: 5000

evaluation:
  games: 2000
  seeds: [42, 43, 44]

opponent_pool:
  random-v1: 0.10
  rule-v1: 0.20
  bc-v2.1: 0.30
  ppo-v1: 0.20
  historical: 0.20
```

这些参数是起点，不是最终超参数。

AI Agent 不得看到配置后直接进行大规模 sweep。

---

# 26. P1：Model Architecture 暂不激进升级

Phase 10.3 第一阶段继续：

```text
MLP 256x256
```

原因：

当前最大未知量不是模型容量，而是：

```text
PPO 是否真的能从 BC 基础上产生稳定策略改进
```

因此不要同时：

```text
换 Transformer
+
改 feature
+
改 reward
+
改 PPO
```

如果 100k~1M PPO 已经证明训练机制有效，再单独开启模型升级实验。

---

# 27. P1：Reward 暂时保持 reward-v1

继续固定：

```yaml
score_delta_scale: 0.001
placement_bonus:
  - 2.0
  - 1.0
  - -1.0
  - -2.0
```

Phase 10.3 前半段不得修改。

原因：

如果 reward、model、rollout、opponent pool 同时变化，将无法判断性能变化来源。

如果出现 reward 问题：

```text
真实 bug
 ↓
修复
 ↓
regression
 ↓
version bump
```

而不是直接改数值。

---

# 28. Self-play 策略

Phase 10.3 允许：

```text
current PPO vs current PPO
```

但正式训练不能长期只使用这一模式。

推荐：

```text
Opponent Pool
       ↓
 ┌─────┼─────┐
Random Rule BC Historical PPO
       ↓
Current PPO
```

核心目标：

```text
避免策略只适应自身
```

---

# 29. Historical Checkpoint Pool

当 PPO 达到：

```text
20k
50k
100k
500k
```

应自动注册为 historical opponent。

例如：

```text
ppo-20k
ppo-50k
ppo-100k
```

不要只保留 latest。

Historical checkpoint 是后续避免策略遗忘的重要基础。

---

# 30. P1：Regression Test 要求

必须新增/完善：

```text
tests/test_parallel_rollout.py
tests/test_rollout_parity.py
tests/test_opponent_pool.py
tests/test_ppo_multiseed.py
tests/test_ppo_checkpoint_v2.py
```

重点：

```text
single env == parallel env
reference rollout == optimized rollout
same seed reproducibility
opponent sampling reproducibility
checkpoint resume
legal action mask
reward conservation
```

---

# 31. 性能验收

进入 500k / 1M PPO 前，必须知道：

```text
steps/sec
games/hour
CPU utilization
GPU utilization
GPU memory
rollout/update ratio
```

优化目标不是某个固定数字，而是：

```text
Phase 10.2 reference
        ↓
profiling
        ↓
优化
        ↓
吞吐明显提升
```

同时：

```text
behavior identical
```

---

# 32. Phase 10.3 实施顺序

严格按以下顺序：

```text
1. 阅读全部现有 docs
        ↓
2. 阅读 PHASE10.2.md / PHASE10.2_RESULT.md
        ↓
3. 固化 BC/PPO-v1 多 seed baseline
        ↓
4. double_yakuman 决策门
        ↓
5. rollout profiling
        ↓
6. parallel environment
        ↓
7. batch policy inference
        ↓
8. optimized trajectory buffer
        ↓
9. reference parity tests
        ↓
10. 20k PPO
        ↓
11. 50k PPO
        ↓
12. 100k PPO
        ↓
13. opponent pool
        ↓
14. historical checkpoint pool
        ↓
15. multi-seed PPO
        ↓
16. 500k / 1M+ PPO
        ↓
17. formal benchmark
        ↓
18. strength gate
        ↓
19. PHASE10.3_RESULT.md
        ↓
20. STATUS.md
```

任何 correctness gate 失败：

```text
停止扩大训练
 ↓
修复
 ↓
regression
 ↓
重新验证
```

---

# 33. Definition of Done

Phase 10.3 至少满足：

```text
[ ] Phase 10.2 baseline 完整保留
[ ] BC-v2.1 baseline 固化
[ ] PPO-v1 baseline 固化
[ ] 多 seed benchmark 完成
[ ] double_yakuman 状态明确

[ ] rollout profiling 完成
[ ] env step profiling 完成
[ ] policy inference profiling 完成
[ ] steps/sec 已记录
[ ] games/hour 已记录

[ ] parallel environment 实现
[ ] reference / parallel parity PASS
[ ] batch inference 实现
[ ] optimized buffer 实现
[ ] optimized/reference parity PASS

[ ] opponent pool 实现
[ ] opponent probabilities 配置化
[ ] historical checkpoints 支持
[ ] opponent sampling 可复现

[ ] PPO 20k PASS
[ ] PPO 50k PASS
[ ] PPO 100k PASS
[ ] PPO 500k/1M（若前序 gate 通过）

[ ] multi-seed PPO 完成
[ ] checkpoint/resume PASS
[ ] illegal_rate = 0
[ ] 无 NaN/Inf
[ ] Environment 无异常终止

[ ] Greedy benchmark 完成
[ ] Sampling benchmark 完成
[ ] PPO vs Random
[ ] PPO vs Rule
[ ] PPO vs BC-v2.1
[ ] PPO vs historical PPO

[ ] mean rank 已统计
[ ] rank distribution 已统计
[ ] win rate 已统计
[ ] deal-in rate 已统计
[ ] riichi rate 已统计
[ ] call rate 已统计
[ ] average win points 已统计
[ ] score delta 已统计
[ ] illegal rate 已统计

[ ] multi-seed mean/std 已统计
[ ] strength gate 已执行
[ ] PHASE10.3_RESULT.md 已生成
[ ] STATUS.md 已更新
```

---

# 34. Strength Gate：最终 Go / No-Go

Phase 10.3 最终必须明确：

```text
                    PPO-v2
                       │
                       ▼
               Multi-seed Benchmark
                       │
             ┌─────────┴─────────┐
             │                   │
            FAIL                PASS
             │                   │
             ▼                   ▼
       进入诊断阶段         下一阶段
       不继续盲目扩容       模型升级 / 更大规模
```

## PASS 至少要求

### 1. Correctness

```text
illegal_rate = 0
```

### 2. Stability

```text
no NaN
no Inf
checkpoint/resume PASS
multi-seed reproducible
```

### 3. Benchmark

PPO 不得只依赖 Random baseline。

至少必须在：

```text
Rule
BC-v2.1
```

上显示出有意义的提升趋势。

### 4. Multi-seed

提升不能只出现在单个 seed。

### 5. Sampling

如果 Sampling 长期明显落后 BC：

```text
不要宣布最终策略已经成熟
```

进入诊断：

```text
exploration
reward
entropy
value
opponent distribution
policy collapse
```

---

# 35. Phase 10.3 不要求解决的问题

以下可以继续延后：

```text
Transformer
CNN
更高维 feature
LLM policy
multi-GPU
Ray
JAX
TB 级 feature cache
超大规模 hyperparameter sweep
```

除非 profiling / benchmark 明确证明必要，否则不要提前进入。

---

# 36. Phase 10.4 预览

如果 Phase 10.3 成功，后续可以独立开展：

```text
Phase 10.4
Model / Representation Upgrade
```

候选方向：

```text
structured tile encoder
attention
Transformer
action-conditioned policy
value network upgrade
更丰富的时序状态表示
```

以及：

```text
large-scale self-play
league training
exploitability / robustness evaluation
```

但 Phase 10.3 不提前实现这些。

---

# 37. AI Coding Agent 执行规则

AI Agent 必须：

1. 开始前阅读：
   - `PROJECT_PLAN.md`
   - `ARCHITECTURE.md`
   - `DATA_SPEC.md`
   - `MODEL_SPEC.md`
   - `ENVIRONMENT_SPEC.md`
   - `TRAINING_SPEC.md`
   - `CODEX_RULES.md`
   - `PHASE10.1.1.md`
   - `PHASE10.1.1_RESULT.md`
   - `PHASE10.2.md`
   - `PHASE10.2_RESULT.md`
2. 保留所有历史 checkpoint 和 benchmark。
3. 不修改 `bc-v2.1`。
4. 不改变 feature-v1/action-v1 语义。
5. 不为了 benchmark 修改麻将规则。
6. 不同时修改 reward + model + environment + benchmark。
7. 先 profiling，再性能优化。
8. 每项优化必须有 reference parity test。
9. 先 20k，再 50k，再 100k。
10. 前一级不健康不得进入下一级。
11. 必须多 seed。
12. 必须建立 opponent pool。
13. 必须保留 historical checkpoints。
14. Sampling benchmark 不得删除。
15. 训练 reward 不得作为唯一成功标准。
16. PPO 输给 BC 时先分析，不得立即改 reward。
17. 不得自动跳入 Phase 10.4。
18. 最终生成：

```text
docs/PHASE10.3_RESULT.md
```

并明确报告：

```text
修改内容
测试结果
性能结果
20k/50k/100k/500k/1M 结果
Opponent Pool
Multi-seed
Benchmark
PPO vs BC
PPO vs Rule
PPO vs Random
PPO vs Historical PPO
发现的问题
剩余风险
是否 PASS
是否允许进入下一阶段
```

---

# 38. 最终目标

Phase 10.3 的最终目标不是：

> “训练一个 PPO 模型并且跑完 1M steps。”

而是证明：

```text
BC-v2.1
   ↓
PPO
   ↓
高吞吐 Rollout
   ↓
Opponent Pool
   ↓
Historical Checkpoints
   ↓
Multi-seed Training
   ↓
Independent Benchmark
   ↓
稳定超过 BC-v2.1
```

如果最终结果是：

```text
PPO 训练稳定
但
PPO 长期无法稳定超过 BC
```

那么 Phase 10.3 应判定为：

```text
RL infrastructure PASS
Strategy improvement FAIL / INCONCLUSIVE
```

此时不应该继续无止境增加计算量，而应该进入策略诊断 / Representation / Reward / Opponent Curriculum 的独立实验阶段。

**计算规模增加必须建立在策略增益证据之上。**
