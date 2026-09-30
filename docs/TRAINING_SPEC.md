# 日本立直麻将 AI 训练规范

## 1. 总体训练路线

```text
Decision Dataset
      ↓
Behavior Cloning
      ↓
BC Policy
      ↓
Mahjong Environment
      ↓
PPO / RL
      ↓
Self-play
      ↓
Opponent Pool
      ↓
Benchmark
```

---

## 2. Behavior Cloning

输入：

```text
Observation
Legal Actions
```

目标：

```text
Human Action
```

输出：

```text
P(Action | Observation)
```

第一版使用监督学习。

---

## 3. BC Dataset

训练、验证、测试必须按 Game 切分。

禁止：

```text
随机 DecisionSample 切分
```

---

## 4. BC Training

至少记录：

```text
dataset version
feature version
model version
batch size
learning rate
epochs
seed
train loss
validation loss
accuracy
top-k accuracy
```

---

## 5. BC Checkpoint

训练完成后保存：

```text
model.pt
config.yaml
metrics.json
dataset_version.txt
feature_version.txt
git_commit.txt
```

---

## 6. RL

初始算法：

```text
PPO
```

Policy 至少需要：

```text
action logits
action sampling
log probability
```

训练时需要：

```text
trajectory
reward
return
advantage
policy update
value update
```

---

## 7. Reward

Reward 应与麻将最终目标相关。

候选：

```text
score delta
final score
placement
placement + score
```

具体方案应通过实验确定。

---

## 8. PPO 训练阶段

概念流程：

```text
Policy
 ↓
Environment Workers
 ↓
Trajectories
 ↓
Advantage
 ↓
PPO Update
 ↓
Checkpoint
 ↓
Evaluation
```

---

## 9. Self-play

Self-play 不应只让：

```text
Current Model vs Current Model
```

长期运行。

建立：

```text
Opponent Pool
```

包括：

```text
BC Policy
Historical Checkpoints
Current Policy
Older RL Policies
Fixed Baseline
```

---

## 10. Opponent Pool

每个 Opponent 必须带版本：

```text
opponent_id
model_version
checkpoint
config
```

训练时记录对手组合。

---

## 11. Benchmark

每个候选 Checkpoint 必须进行独立 Benchmark。

记录：

```text
model
opponents
game count
rules
seed
results
```

---

## 12. Benchmark 指标

可记录：

```text
平均顺位
顺位分布
和牌率
放铳率
立直率
副露率
平均和牌点数
平均得点变化
```

不要仅用训练 loss 判断麻将策略强度。

---

## 13. Checkpoint 晋级

推荐流程：

```text
训练新模型
   ↓
Smoke Test
   ↓
规则测试
   ↓
固定 Benchmark
   ↓
与基准模型比较
   ↓
保存结果
```

不要自动覆盖历史模型。

---

## 14. 实验目录

```text
experiments/
└── exp_0001/
    ├── config.yaml
    ├── logs/
    ├── checkpoints/
    ├── metrics.json
    └── README.md
```

---

## 15. 大规模训练

只有基础训练链路稳定后，才考虑：

```text
多 GPU
多进程 Environment
分布式 Self-play
Ray
JAX
批量推理
```

优化前必须保留 Reference 实现用于正确性验证。

---

## 16. 实验可复现

每次正式实验必须记录：

```text
代码版本
数据版本
模型版本
环境版本
配置
seed
GPU
训练步数
checkpoint
Benchmark
```

---

## 17. 训练失败处理

出现：

- loss NaN
- 非法 Action
- 环境崩溃
- Replay 不一致
- reward 异常
- Benchmark 异常

时，应优先定位基础设施问题，不应通过修改 reward 或训练参数掩盖问题。
