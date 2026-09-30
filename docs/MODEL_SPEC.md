# 日本立直麻将 AI 模型规范

## 1. 模型目标

模型负责：

```text
Observation + Legal Actions
            ↓
       Action Policy
```

训练阶段分为：

```text
Behavior Cloning
        ↓
Initial Policy
        ↓
Reinforcement Learning
        ↓
Self-play
```

---

## 2. 模型与规则解耦

模型不得实现：

- 吃是否合法
- 碰是否合法
- 立直是否合法
- 和牌是否合法
- 牌局状态转移

这些由规则系统负责。

模型只负责：

> 在给定 Observation 和 Legal Actions 后评估行动。

---

## 3. Action Mask

模型输出完整 Action Space 的 logits 后，应使用 Legal Action Mask。

概念：

```text
raw logits
    ↓
legal mask
    ↓
masked logits
    ↓
softmax
    ↓
policy
```

非法动作不得被采样。

---

## 4. 第一版模型

推荐：

```text
Observation Encoder
        ↓
MLP
        ↓
Action Head
```

第一版目标：

- 跑通训练
- 验证 Dataset
- 验证 Action Mask
- 建立 Baseline

不是追求最终强度。

---

## 5. 后续模型

可以逐步升级：

```text
MLP
 ↓
Structured Encoder
 ↓
CNN / Attention
 ↓
Transformer
 ↓
Policy + Value Network
```

模型升级不得要求重写 Replay Engine。

---

## 6. Observation Encoder

建议将输入按结构拆分：

```text
Own Hand
Melds
Own Discards
Opponent Discards
Dora
Round State
Scores
Player Status
Visible Table State
```

编码方式应由实验决定。

---

## 7. Policy Output

推荐统一：

```python
PolicyOutput(
    logits,
    value=None,
)
```

RL 阶段可以扩展 Value Head。

---

## 8. Behavior Cloning

目标：

```text
P(a | s)
```

Loss：

```text
Cross Entropy
```

并结合 Legal Action Mask。

需要记录：

```text
train loss
validation loss
action accuracy
top-k accuracy
illegal probability / illegal action rate
```

---

## 9. BC 的定位

BC 是：

```text
人类数据
 ↓
初始 Policy
```

不是最终麻将最优策略。

原因：

- 人类行为存在差异
- 数据包含不同水平玩家
- 人类行为不一定最优
- BC 优化的是行为拟合，而不是最终得分

---

## 10. RL 阶段

Policy 应支持：

```text
action sampling
action log probability
value estimation
```

用于 PPO。

---

## 11. Checkpoint

至少保存：

```text
model weights
model config
feature version
action schema version
dataset version
environment version
git commit
training step
metrics
```

---

## 12. 模型接口

```python
class Policy:

    def forward(self, observation, legal_mask):
        ...

    def sample_action(self, observation, legal_actions):
        ...

    def evaluate_actions(self, observation, actions):
        ...
```

---

## 13. 模型测试

必须验证：

```text
输入 shape 正确
输出 shape 正确
Mask 正确
非法动作概率被屏蔽
checkpoint 可以保存
checkpoint 可以加载
相同 seed 可以复现
```

---

## 14. 模型升级原则

不要因为 Transformer 更先进就直接替换。

必须通过 Benchmark 判断新模型是否改善实际任务指标。

模型结构、训练参数和数据版本都必须记录。
