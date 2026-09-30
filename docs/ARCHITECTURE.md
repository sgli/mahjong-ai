# 日本立直麻将 AI 系统架构规范

## 1. 架构总览

```text
                    ┌───────────────────┐
                    │   Raw Mjai Data   │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │   Data Pipeline   │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │    Mjai Parser    │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │   Replay Engine   │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │ Decision Extractor│
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │ Feature Encoder   │
                    └─────────┬─────────┘
                              ↓
              ┌───────────────┴──────────────┐
              ↓                              ↓
      Behavior Cloning                 Offline Evaluation
              ↓
       Initial Policy
              ↓
      Mahjong Environment
              ↓
             PPO
              ↓
          Self-play
              ↓
       Opponent Pool
              ↓
          Benchmark
```

---

## 2. 模块职责

### Parser

职责：

- 读取 JSON
- 转换为 Typed Event
- 做基础结构校验

不负责：

- 麻将状态转移
- 策略判断
- 模型训练

### Replay Engine

职责：

- 按事件顺序更新游戏状态
- 保持状态一致性
- 输出每一步 ReplayState

不负责：

- 判断好坏
- 计算策略
- 训练模型

### Decision Extractor

职责：

- 找到决策窗口
- 生成 Legal Actions
- 提取实际 Human Action
- 构建 DecisionSample

### Feature Encoder

职责：

- ReplayState → Tensor
- Action → Model Action ID
- 保证 Observation 无未来信息

### Model

职责：

- 根据 Observation 产生 Action logits / value
- 使用 Legal Action Mask

不得实现麻将规则。

### Environment

职责：

- 状态
- 合法动作
- 状态转移
- 终局
- Reward

Environment 应尽可能复用统一规则核心。

### Training

职责：

- DataLoader
- BC
- PPO
- Self-play
- Checkpoint

### Evaluation

职责：

- 离线指标
- 实际对局 Benchmark
- 版本比较

---

## 3. 推荐项目结构

```text
mahjong-ai/
├── data/
│   ├── raw/
│   ├── validated/
│   └── processed/
├── src/
│   ├── replay/
│   ├── dataset/
│   ├── features/
│   ├── model/
│   ├── environment/
│   ├── training/
│   └── evaluation/
├── scripts/
├── tests/
├── configs/
├── docs/
└── README.md
```

---

## 4. 技术栈

初始版本：

```text
Python 3.11+
PyTorch
NumPy
PyArrow
Parquet
Pytest
PyYAML
TensorBoard
```

后续可选：

```text
CUDA
JAX
Ray
Weights & Biases
```

不应在基础链路尚未验证前引入复杂分布式组件。

---

## 5. 核心接口

```python
class ReplayEngine:
    def replay(self, events):
        ...


class DecisionExtractor:
    def extract(self, state):
        ...


class FeatureEncoder:
    def encode(self, state, decision):
        ...


class Policy:
    def predict(self, observation, legal_actions):
        ...


class MahjongEnv:
    def reset(self):
        ...

    def step(self, action):
        ...

    def legal_actions(self):
        ...
```

---

## 6. 数据流边界

推荐：

```text
Event
  ↓
ReplayState
  ↓
DecisionPoint
  ↓
Observation
  ↓
Tensor
  ↓
Model
```

禁止：

```text
Model → 修改 ReplayState
FeatureEncoder → 修改 ReplayState
Training → 自己实现麻将规则
```

---

## 7. 可扩展原则

第一版可以使用简单 MLP。

未来允许：

```text
MLP
 ↓
CNN / Structured Encoder
 ↓
Transformer
 ↓
Policy + Value
```

但是接口应保持稳定，使模型架构替换不影响 Replay 和 Dataset。
