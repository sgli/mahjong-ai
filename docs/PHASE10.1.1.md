# Phase 10.1.1 训练基础设施优化指导建议

> 基于最新的代码，以及：
>
> - `docs/PHASE10.1.md`
> - `docs/PHASE10.1_RESULT.md`
> - `docs/BENCHMARK_BC_V2_REPORT.md`
> - 当前 Dataset / Feature / Training / Checkpoint / Benchmark 实现
>
> 本文用于指导后续 Codex / AI Coding Agent 实施代码修改。

---

## 1. 文档目的

Phase 10.1 已经完成了大规模 Behavior Cloning（BC）训练的第一轮验证，但当前训练吞吐较低：

- 数据集约 `1.465B` decision samples
- BC-v2 已完成首轮大规模训练
- 1 epoch 约 `13.2h`
- 当前 GPU 利用率较低，显存占用也很低
- BC-v2 Benchmark 已显示相对于 BC-v1 有明显改善
- 但当前只完成了 1 epoch，尚不能判断 BC 是否已经收敛

因此下一阶段不应立即进入 PPO，而应先完成：

> **Phase 10.1.1：BC Training Infrastructure Optimization**

目标不是改变 Mahjong 策略定义，而是：

1. 提升训练吞吐
2. 降低 CPU / Python preprocessing 开销
3. 实现可靠 Resume
4. 实现 Best / Latest checkpoint
5. 建立完整 learning curve
6. 改进 Benchmark 评估方式
7. 在确认 BC 收敛后，再决定是否进入 PPO

---

# 2. 当前状态与核心判断

当前训练链路大致为：

```text
decision-v2
    ↓
Parquet
    ↓
Arrow RecordBatch
    ↓
to_pylist()
    ↓
row_to_sample()
    ↓
ObservationEncoder.encode()
    ↓
legal_mask()
    ↓
MLPPolicy
    ↓
GPU
```

当前模型：

```text
234 input features
        ↓
256
        ↓
256
        ↓
270 action logits
```

模型本身非常小，因此目前主要问题不是 GPU 算力不足，而是：

> **CPU / Python 数据预处理无法持续向 GPU 提供足够的 batch。**

当前约：

```text
25,000 samples/s
```

因此 1 epoch 约：

```text
13.2h
```

如果未来完整训练 5 epoch，则成本会非常高。

---

# 3. 第一原则：先优化 Pipeline，不要先扩大模型

当前不建议因为 GPU utilization 低而直接：

- 换 Transformer
- 大幅扩大 MLP
- 增加隐藏层
- 直接上多 GPU

原因：

```text
当前 GPU utilization 很低
        ↓
CPU preprocessing 是主要瓶颈
        ↓
扩大模型无法解决数据供给问题
```

正确顺序：

```text
优化数据读取
    ↓
优化 Feature Encoder
    ↓
Batch-level preprocessing
    ↓
DataLoader / prefetch / pinned memory
    ↓
提高 samples/sec
    ↓
确认 GPU 真正成为瓶颈
    ↓
再考虑扩大模型
```

---

# 4. P0：实现可靠 Resume Training

这是最高优先级之一。

当前 checkpoint 已经保存模型参数和部分训练信息，但不能作为完整的训练状态恢复机制。

对于：

```text
13.2h / epoch
```

的训练，必须支持：

```text
epoch 1
    ↓
保存 checkpoint
    ↓
程序中断
    ↓
第二天 resume
    ↓
epoch 2
```

而不能重新从 epoch 1 开始。

## 4.1 Checkpoint 至少保存

建议保存：

```text
model_state_dict
optimizer_state_dict

epoch
global_step

best_val_loss
best_val_accuracy

dataset_version
feature_version
action_schema_version
environment_version
model_version

training_config
metrics

Python RNG state
Torch CPU RNG state
Torch CUDA RNG state
```

如果使用 AMP，还应保存：

```text
GradScaler state
```

---

## 4.2 建议 checkpoint 目录

```text
experiments/bc_v2/
└── checkpoints/
    ├── latest.pt
    ├── best.pt
    ├── epoch_001.pt
    ├── epoch_002.pt
    └── ...
```

语义：

- `latest.pt`：最近一次保存
- `best.pt`：验证集指标最佳
- `epoch_N.pt`：可选的历史 checkpoint

---

## 4.3 CLI

建议支持：

```bash
python scripts/train_bc.py \
    --config configs/train_bc_v2.yaml \
    --resume experiments/bc_v2/checkpoints/latest.pt
```

Resume 后应继续：

```text
epoch
global_step
optimizer state
learning-rate state
random state
```

而不是只加载模型权重。

---

# 5. P0：实现 Fast Row Encoder

这是目前最重要的性能优化之一。

当前路径类似：

```text
Arrow row
    ↓
row_to_sample()
    ↓
DecisionSample
    ↓
PlayerObservation
    ↓
ObservationEncoder
```

对于十几亿样本，这种对象创建开销非常大。

## 5.1 建议新增

例如：

```python
encode_row_fast(row)
```

直接从 Parquet / Arrow row 中读取训练所需字段：

```text
hand
melds
opponents_melds
discards
dora_markers
scores
riichi
bakaze
kyoku
honba
kyotaku
oya
turn
seat
```

直接产生：

```text
234-dim feature
```

避免为训练重新构造完整：

```text
DecisionSample
PlayerObservation
Meld
Action
ActionType
```

对象树。

---

## 5.2 重要约束

Fast Encoder 必须保证：

> **Feature Semantics 完全不变。**

不能因为优化性能而修改：

- tile encoding
- meld encoding
- discard encoding
- score encoding
- riichi encoding
- turn encoding
- seat encoding
- feature ordering

必须增加：

```text
old encoder
vs
fast encoder
```

的自动验证。

建议随机抽取大量 decision samples：

```text
old_features == fast_features
```

对于 float 特征可以使用：

```python
torch.equal()
```

或：

```python
torch.allclose(
    old,
    new,
    rtol=0,
    atol=0
)
```

如果实现完全确定性，优先要求 bitwise identical。

---

# 6. P0：Batch-level Feature Encoding

当前大量操作是 sample-level：

```text
sample
 ↓
torch.zeros()
 ↓
Python loop
 ↓
torch.tensor()
 ↓
torch.cat()
```

这类操作乘以十几亿次，会产生非常大的 Python / allocator 开销。

目标是：

```text
RecordBatch
    ↓
Batch Encoder
    ↓
[B, 234]
```

而不是：

```text
row
 ↓
[234]
```

逐条生成。

---

## 6.1 优先优化的操作

当前 Encoder 中大量存在：

```python
torch.zeros()
torch.tensor()
torch.cat()
```

以及：

```python
for tile in ...
```

应尽量改成：

- 预分配 batch tensor
- batch-level scatter
- NumPy / Arrow columnar operation
- 避免大量小 Tensor allocation
- 避免 Python nested loops

---

# 7. P0：优化 legal action mask

当前 Action Space：

```text
270 actions
```

当前每个 sample 创建：

```text
[270] bool Tensor
```

再逐 action 设置。

对于十几亿样本，这是大量小 Tensor allocation。

## 第一阶段建议

至少改成 batch-level mask：

```text
[B, 270]
```

一次性构造。

例如：

```text
batch action IDs
        ↓
batch scatter
        ↓
[B, 270] bool
```

而不是：

```text
sample 1 → zeros(270)
sample 2 → zeros(270)
sample 3 → zeros(270)
...
```

---

## 第二阶段

如果后续 profiling 证明 mask 仍然是瓶颈，再考虑：

```text
legal action IDs
```

代替完整：

```text
270 bool mask
```

但这属于后续优化，不应作为第一版修改。

---

# 8. P0：减少 Arrow → Python 转换

当前存在：

```python
record_batch.to_pylist()
```

这会把 columnar Arrow 数据转换成大量 Python dict / list / scalar。

对于：

```text
1.465B samples
```

这种转换成本非常高。

目标：

```text
Arrow RecordBatch
        ↓
batch-level extraction
        ↓
batch encoder
```

尽可能减少：

```text
Arrow
 ↓
Python dict
 ↓
Python objects
```

---

# 9. P0：增加 Training Profiler

后续所有优化必须以数据为依据。

训练日志建议记录：

```text
data_time
forward_time
backward_time
optimizer_time
batch_time

samples_per_second

GPU utilization
GPU memory
CPU utilization
```

例如：

```text
epoch=1
step=10000

data_time=...
forward_time=...
backward_time=...
optimizer_time=...
batch_time=...

samples/sec=...
```

---

## 9.1 评价优化的核心指标

第一指标：

```text
samples/sec
```

第二指标：

```text
epoch wall-clock time
```

辅助指标：

```text
GPU utilization
GPU memory
CPU utilization
```

不要单纯追求：

```text
GPU utilization = 100%
```

如果：

```text
GPU 20%
50k samples/s
```

比：

```text
GPU 80%
45k samples/s
```

更快，则应选择前者。

---

# 10. P0：Batch Size Sweep

当前 batch size：

```text
512
```

对于：

```text
234 input
MLP [256, 256]
270 output
```

很可能偏小。

建议测试：

```text
512
1024
2048
4096
8192
16384
```

如果显存允许，可以进一步测试：

```text
32768
```

每个 batch size 测：

```text
samples/sec
GPU utilization
GPU memory
CPU utilization
epoch time
```

选择吞吐最高的方案。

---

# 11. P1：DataLoader / Prefetch / Pinned Memory

目标：

```text
CPU preprocessing
        ↓
prefetch
        ↓
pinned memory
        ↓
non_blocking CPU → GPU
        ↓
GPU
```

训练循环应尽可能类似：

```python
for features, action_ids, masks in loader:
    features = features.to(
        device,
        non_blocking=True,
    )

    action_ids = action_ids.to(
        device,
        non_blocking=True,
    )

    masks = masks.to(
        device,
        non_blocking=True,
    )

    out = model(features, masks)
```

---

## 11.1 修正 pin_memory 配置语义

当前代码中存在一个配置层面的细节：

训练配置读取了：

```text
pin_memory
```

但 DataLoader 内部又根据：

```text
device.type == "cuda"
```

决定 pin memory。

应统一配置语义，避免：

```yaml
pin_memory: false
```

但实际仍启用 pin memory。

---

# 12. P1：Deterministic Shard Shuffle

当前 dataset 已经按照 shard / worker 进行读取。

建议加入：

```text
seed + epoch
```

决定 shard 顺序。

例如：

```text
epoch 1:
A B C D E

epoch 2:
D A E C B

epoch 3:
C E A B D
```

要求：

```text
相同 seed
+
相同 epoch
=
完全相同的 shard 顺序
```

这样既能改善训练样本顺序，也不会破坏实验可复现性。

---

# 13. P1：Validation Sampling

当前 validation 数据规模非常大。

如果每一个 epoch 都完整跑：

```text
146M validation samples
```

会进一步增加训练时间。

建议增加：

```yaml
validation:
  mode: sampled
  samples: 5000000
```

例如：

```text
每个 epoch
    ↓
固定 5M validation samples
    ↓
快速判断趋势
```

最终训练完成后：

```text
best checkpoint
    ↓
full validation
    ↓
test
```

---

# 14. P1：建立完整 Learning Curve

当前 BC-v2 只有：

```text
epoch 1
```

因此：

```text
67.92% validation accuracy
```

不能说明 BC 已经收敛。

建议至少观察：

```text
epoch 1
epoch 2
epoch 3
```

记录：

```text
train loss
train accuracy

validation loss
validation accuracy

top-3 accuracy
top-5 accuracy
```

然后再决定是否继续：

```text
epoch 4
epoch 5
```

---

# 15. 不建议直接跑完整 5 epoch

在完成性能优化后，建议先做小规模收敛实验。

例如：

```text
部分训练数据
    ↓
epoch 1
epoch 2
epoch 3
```

观察：

```text
train loss
val loss
train accuracy
val accuracy
```

如果曲线仍明显改善，再进行完整数据训练。

这样可以避免：

```text
13h × 5
```

的高成本试错。

---

# 16. P1：Best / Latest Checkpoint

建议：

```text
latest.pt
```

每次保存最新训练状态。

同时：

```text
best.pt
```

根据 validation 指标保存最佳模型。

例如：

```text
best validation loss
```

或：

```text
best validation accuracy
```

具体指标应在配置中明确。

---

# 17. Benchmark：增加 Greedy / Sampling 两种模式

当前 `PolicyOpponent` 使用 stochastic sampling：

```text
torch.multinomial()
```

因此 Benchmark 测量的是：

> stochastic policy behavior

建议增加：

## Greedy

```text
argmax(policy probability)
```

用于测试：

> deterministic policy

## Sampling

```text
multinomial(policy)
```

用于测试：

> stochastic policy

以后 checkpoint 可以分别报告：

```text
BC-v2
├── Greedy benchmark
└── Sampling benchmark
```

---

# 18. Benchmark：增加 Rule Baseline

当前 Random baseline 是：

```text
从所有合法 action 中随机选择
```

因此可能随机：

- discard
- chi
- pon
- kan
- riichi
- ron
- pass

所以：

```text
call_per_game 很高
win_per_game 很低
```

这种 Random baseline 主要用于：

> sanity check

而不是强度基准。

建议增加：

```text
Rule Baseline
```

作为真正有意义的传统策略对照。

初版不需要很强，可以基于：

```text
shanten
ukeire
役种
打点
危险度
简单防守
```

形成一个可解释的 heuristic policy。

---

# 19. Benchmark：同时记录 Candidate 与 Opponent

Benchmark 不应只记录 candidate。

建议：

```text
candidate metrics
+
opponent metrics
```

同时记录：

```text
mean rank
rank distribution
win rate
deal-in rate
riichi rate
call rate
average win points
score change
```

这样未来：

```text
BC-v3 vs BC-v2
```

可以判断差异到底来自哪里。

---

# 20. 数据质量：69 个 replay error

当前数据中约：

```text
69 个 replay errors
```

相对于十几亿 decision samples，比例极低。

因此：

> 不应因为这 69 个文件阻塞大规模训练。

但建议建立：

```text
dataset_errors.jsonl
```

记录：

```json
{
  "game_id": "...",
  "file": "...",
  "error_type": "...",
  "event_type": "...",
  "step": 123,
  "message": "..."
}
```

以后统计：

```text
error_type
出现次数
具体牌谱
具体事件
```

以判断到底是：

```text
脏牌谱
```

还是：

```text
Replay Engine edge case
```

---

# 21. 不建议现在做的事情

## 21.1 暂不直接换 Transformer

当前 GPU utilization 很低。

先解决：

```text
CPU → GPU pipeline
```

再讨论模型规模。

---

## 21.2 暂不直接多 GPU

单 GPU 还没有充分利用。

先把单 GPU pipeline 做到稳定高吞吐。

---

## 21.3 暂不直接进入 PPO

当前只完成 BC-v2 的 1 epoch。

应该先确认：

```text
BC convergence
```

否则 PPO 会同时引入：

```text
BC 问题
Environment 问题
Reward 问题
RL training 问题
```

难以定位。

---

## 21.4 暂不直接创建完整 Feature Cache

如果每条样本保存：

```text
234 float32
+
270 legal mask
```

对于约：

```text
1.465B samples
```

会产生 TB 级额外存储。

因此第一阶段不要直接采用：

```text
decision-v2
    ↓
完整 dense feature dataset
```

先优化：

```text
Fast Encoder
Batch Encoder
Arrow batch processing
```

只有 profiling 证明仍然不够快时，再考虑压缩后的 feature cache。

---

# 22. 推荐实施顺序

## Phase 10.1.1-A：训练基础设施

```text
1. Resume checkpoint
2. Best/latest checkpoint
3. Training profiler
4. Batch size benchmark
```

---

## Phase 10.1.1-B：数据管线优化

```text
5. Fast row encoder
6. Fast encoder correctness tests
7. Batch-level feature encoding
8. Batch-level legal mask
9. 减少 to_pylist / Python object creation
10. DataLoader prefetch / pin / non_blocking
```

---

## Phase 10.1.1-C：训练策略

```text
11. Deterministic shard shuffle
12. Validation sampling
13. Learning curve
14. Resume full training
```

---

## Phase 10.1.1-D：Benchmark

```text
15. Greedy benchmark
16. Sampling benchmark
17. Rule baseline
18. Candidate / opponent metrics
```

---

## Phase 10.1.1-E：数据质量

```text
19. Error manifest
20. 69 replay errors 分类
```

---

# 23. 建议的验收标准

Phase 10.1.1 不应以“代码完成”作为结束条件，而应以以下指标验收。

## 数据正确性

```text
Fast Encoder == Old Encoder
```

要求：

```text
100% sampled validation
```

不能出现 feature semantics 改变。

---

## Training throughput

当前基线：

```text
≈25k samples/s
```

目标：

```text
第一目标：≥50k samples/s
```

理想：

```text
75k~100k+ samples/s
```

最终以实际 benchmark 为准，不硬性要求某一个数字。

---

## Resume

必须验证：

```text
训练中断
    ↓
resume
    ↓
继续正确 epoch / global step
```

并验证：

```text
model
optimizer
LR
RNG
```

都正确恢复。

---

## Checkpoint

必须存在：

```text
latest.pt
best.pt
```

并能够加载。

---

## Benchmark

至少：

```text
BC-v2 Greedy
BC-v2 Sampling
BC-v1
Rule Baseline
Random Baseline
```

---

## Learning Curve

至少得到：

```text
epoch 1
epoch 2
epoch 3
```

的：

```text
train loss
val loss
train accuracy
val accuracy
```

---

# 24. Phase 10.1.1 完成后的决策树

```text
                BC-v2
                  │
                  ▼
        Phase 10.1.1 optimization
                  │
                  ▼
          50k+ samples/sec
                  │
                  ▼
        epoch 1 / 2 / 3 curve
                  │
          ┌───────┴────────┐
          │                │
       继续下降          已收敛
          │                │
          ▼                ▼
      继续 BC          Best BC
                           │
                           ▼
                    Benchmark
                           │
                 ┌─────────┴─────────┐
                 │                   │
              表现稳定             需要改进
                 │                   │
                 ▼                   ▼
              PPO/RL            继续 BC / Model
                 │
                 ▼
             Self-play
                 │
                 ▼
          Opponent Pool
                 │
                 ▼
             Benchmark
```

---

# 25. 推荐最终训练架构

```text
                    decision-v2
                         │
                         ▼
                ┌─────────────────┐
                │ Parquet / Arrow │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Batch Reader    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Fast Batch      │
                │ Feature Encoder │
                └────────┬────────┘
                         │
                         ▼
                CPU Tensor Batch
                         │
                  pin_memory
                         │
                  non_blocking
                         ▼
                       CUDA
                         │
                         ▼
                 MLP Policy
              234 → 256 → 256
                         │
                         ▼
                    270 actions
                         │
                         ▼
                  Legal Mask
                         │
                         ▼
                   Prediction
```

---

# 26. 核心原则

整个 Phase 10.1.1 必须遵守以下原则：

### 原则 1：不改变 Mahjong 语义

性能优化不能改变：

- Replay
- Observation
- Feature
- Action Space
- Legal Action
- Reward
- Environment

---

### 原则 2：不引入数据泄漏

继续保证：

```text
train / validation / test
```

按照完整：

```text
game_id
```

进行隔离。

---

### 原则 3：先性能，再规模

先：

```text
25k
→
50k
→
75k+
```

再考虑：

```text
更大模型
更多 epoch
多 GPU
```

---

### 原则 4：所有性能优化必须有 benchmark

不能只说：

> “理论上应该更快。”

必须提供：

```text
before
vs
after
```

的：

```text
samples/sec
epoch time
CPU
GPU
memory
```

对比。

---

### 原则 5：不要为了 GPU utilization 而 GPU utilization

真正目标：

> **最大化有效训练吞吐，同时保持训练结果完全一致。**

---

# 27. 最终结论

Phase 10.1 已经证明：

```text
大规模 decision-v2
        ↓
BC-v2
        ↓
可以正常训练
        ↓
Benchmark 相比 BC-v1 有明显提升
```

因此当前项目不需要推倒重来。

下一阶段最合理的工作是：

```text
Phase 10.1
大规模 BC 第一轮
        ↓
Phase 10.1.1
训练基础设施优化
        ↓
Fast Encoder
Batch Pipeline
Resume
Checkpoint
Profiler
Benchmark
        ↓
BC 多 epoch 收敛验证
        ↓
确认 Best BC
        ↓
再决定 PPO
```

**不要现在为了低 GPU utilization 去扩大模型，也不要直接进入 PPO。**

当前最有价值的工程工作，是把已经验证正确的 BC pipeline 做到：

```text
高吞吐
可 Resume
可复现
可验证
可 Benchmark
```

然后再进入 RL 阶段。
