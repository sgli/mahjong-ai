# 日本立直麻将 AI 项目总规划

## 1. 文档定位

本文档是日本立直麻将 AI 项目的**总体工程规格书**，主要供 Codex、Coding Agent 和项目开发人员读取。

本文档不承担 AI 教程职责，不规定个人学习路线。

Codex 在开发过程中应以本文档及其拆分出的专项规格文档为项目约束。若实现任务与文档冲突，应优先发现并报告冲突，而不是自行改变总体架构。

专项文档：

- `ARCHITECTURE.md`：系统架构与模块边界
- `DATA_SPEC.md`：牌谱、Replay、Decision Dataset 数据规范
- `MODEL_SPEC.md`：模型与策略网络规范
- `ENVIRONMENT_SPEC.md`：麻将环境、动作、状态转移与奖励规范
- `TRAINING_SPEC.md`：BC、RL、Self-play 与实验规范
- `CODEX_RULES.md`：Codex 开发行为规范

---

## 2. 项目目标

使用约几百GB 的 `.mjai.json` 日本立直麻将牌谱，构建一个通用麻将 AI。

目标不是模仿某一名职业选手，而是：

1. 从大量人类牌谱中学习初始策略；
2. 使用正确的麻将状态重建和决策数据作为训练基础；
3. 使用 Behavior Cloning 获得初始 Policy；
4. 在可靠的麻将 Environment 中进行 Reinforcement Learning；
5. 通过 Self-play 和 Opponent Pool 持续优化；
6. 建立独立 Benchmark 判断不同版本策略的实际对局表现。

总体路线：

```text
Mjai 牌谱
    ↓
数据验证
    ↓
Mjai Parser
    ↓
Replay Engine
    ↓
Decision Extraction
    ↓
Decision Dataset
    ↓
Feature Encoder
    ↓
Behavior Cloning
    ↓
Initial Policy
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
    ↓
新版本 Policy
```

---

## 3. 核心原则

### 3.1 正确性优先

优先级：

```text
麻将规则正确
>
Replay 正确
>
Decision Dataset 正确
>
训练链路正确
>
模型效果
>
性能优化
```

禁止为了尽快训练而跳过 Replay 和 Environment 验证。

### 3.2 严格模块解耦

```text
Parser
  ↓
Replay
  ↓
Decision
  ↓
Feature
  ↓
Model
  ↓
Environment
  ↓
Training
  ↓
Evaluation
```

各层不得无理由互相侵入。

### 3.3 严禁未来信息泄漏

Policy Observation 只能包含决策时玩家合法可获得的信息。

不得输入：

- 对手隐藏手牌
- 未来摸牌
- 未来河牌
- 未来鸣牌
- 未来立直
- 最终和牌结果
- 最终顺位
- 未来分数变化

未来信息可以作为离线分析数据存在，但不得进入模型 Observation。

### 3.4 可复现

所有正式训练和 Benchmark 必须记录：

- Git commit
- dataset version
- feature version
- action schema version
- environment version
- model config
- training config
- random seed
- checkpoint
- evaluation result

---

## 4. 技术路线

### Phase 0：项目基础设施

建立 Python 项目、依赖、测试、配置、日志和目录。

### Phase 1：牌谱数据层

实现 Mjai Parser、数据检查、数据统计和验证。

### Phase 2：Replay Engine

从事件流还原完整麻将状态。

### Phase 3：Decision Extraction

从 ReplayState 中识别所有决策点、合法动作和牌谱实际动作。

### Phase 4：Decision Dataset

把原始牌普数据转换为可训练的分块 Dataset。

### Phase 5：Behavior Cloning

训练第一版人类行为初始化 Policy。

### Phase 6：Mahjong Environment

实现可靠、可测试的麻将环境。

### Phase 7：Reinforcement Learning

以 BC Policy 为基础进行 PPO 等强化学习。

### Phase 8：Self-play

建立自动对局和 Opponent Pool。

### Phase 9：Benchmark

建立独立评估和模型晋级机制。

### Phase 10：规模化

在系统正确后再考虑多 GPU、分布式 Self-play、JAX/Ray 等。

---

## 5. 当前第一目标

第一阶段不是训练 AI。

第一目标：

```text
Mjai Parser
    ↓
Replay Engine
    ↓
Decision Extraction
```

完成并验证这一条链路后，才进入模型训练。

---

## 6. Definition of Done

长期系统至少应具备：

```text
[ ] 原始牌谱验证
[ ] 确定性 Replay
[ ] 正确 Decision Extraction
[ ] 无未来信息泄漏
[ ] 版本化 Dataset
[ ] Feature Encoder
[ ] Behavior Cloning
[ ] 合法动作 Mask
[ ] 正确 Mahjong Environment
[ ] PPO / RL
[ ] Self-play
[ ] Opponent Pool
[ ] 独立 Benchmark
[ ] Checkpoint 管理
[ ] 实验记录
[ ] 大规模数据处理
[ ] 自动化回归测试
```
