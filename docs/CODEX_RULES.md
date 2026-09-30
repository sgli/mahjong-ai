# Codex 开发规范

## 1. 文档优先

Codex 必须先读取：

```text
docs/PROJECT_PLAN.md
docs/ARCHITECTURE.md
docs/DATA_SPEC.md
docs/MODEL_SPEC.md
docs/ENVIRONMENT_SPEC.md
docs/TRAINING_SPEC.md
docs/CODEX_RULES.md
```

再执行项目级任务。

---

## 2. 不允许擅自改变总体架构

如果发现当前架构存在问题：

1. 说明问题；
2. 指出受影响模块；
3. 提出修改方案；
4. 不要偷偷完成架构迁移。

局部实现可以自主决定；架构级变更必须显式说明。

---

## 3. 按阶段开发

当前阶段未完成前，不得主动实现后续阶段。

例如：

```text
Parser 阶段
```

不要主动加入：

```text
Transformer
PPO
Self-play
```

---

## 4. 不猜测麻将规则

如果输入牌谱、项目规则或已有实现不足以确定某个规则：

```text
标记假设
```

并要求确认。

禁止把猜测当成确定规则写入核心规则系统。

---

## 5. 先测试后扩展

新增核心逻辑时必须同时增加测试。

尤其是：

```text
Parser
Replay
Action
Legal Action
Environment
Reward
```

---

## 6. 保持模块边界

禁止：

```text
Parser 实现策略
Model 实现规则
Feature Encoder 修改 State
Training 自己重建 Mahjong Rules
```

---

## 7. 禁止未来信息泄漏

任何 Observation 生成代码都必须检查：

```text
是否读取了未来 Event？
是否读取了隐藏牌？
是否读取了最终结果？
```

---

## 8. 大数据处理要求

几百GB数据禁止整体加载到内存。

必须使用：

```text
streaming
chunk
Parquet
DataLoader
```

等方式。

---

## 9. 修改前检查现有代码

Codex 修改代码前应：

1. 阅读相关模块；
2. 阅读相关测试；
3. 确认现有接口；
4. 再修改。

不要因为任务简单就直接重写整个模块。

---

## 10. 保持最小修改

优先：

```text
最小必要修改
```

避免：

```text
无关重构
文件大规模重命名
无关依赖升级
```

---

## 11. 第三方依赖

增加依赖前说明：

```text
为什么需要
解决什么问题
是否可以使用现有依赖完成
```

不要为了一个小功能引入大型框架。

---

## 12. 配置外置

以下内容不要硬编码：

```text
dataset path
batch size
learning rate
seed
model size
training steps
environment rules
reward configuration
```

应通过 config 管理。

---

## 13. 运行测试

每次完成任务后至少运行与修改范围相关的测试。

最终报告：

```text
Tests:
pytest ...

Result:
PASS / FAIL

Known Issues:
...
```

---

## 14. 代码质量

要求：

```text
Python type hints
清晰命名
合理模块边界
必要注释
避免重复代码
```

不要求为了形式而过度抽象。

---

## 15. 日志

重要处理过程必须有可读日志。

例如：

```text
records processed
games processed
rounds processed
invalid records
decision samples
throughput
```

---

## 16. 错误处理

数据处理不能因为单个坏文件导致整个 几百GB Pipeline 无提示终止。

应：

```text
记录错误
保存错误上下文
继续处理（若当前阶段允许）
```

最终输出错误统计。

对于核心 Replay 不一致，应根据任务要求决定是：

```text
skip + report
```

还是：

```text
fail fast
```

不得静默忽略。

---

## 17. 完成任务后的报告

每次 Codex 完成一个任务后，必须报告：

```text
## 修改文件

...

## 实现内容

...

## 测试

...

## 测试结果

...

## 已知限制

...

## 下一步

...
```

---

## 18. Codex Task 模板

```text
# Task

## Goal

## Context

## Scope

## Requirements

## Constraints

## Tests

## Acceptance Criteria

## Deliverables
```

---

## 19. 第一阶段 Task

```text
# Task: 初始化项目并实现 Mjai Parser

## Goal

创建 Python 项目，并实现第一版 Mjai JSON Parser 和牌谱统计工具。

## Requirements

1. 创建项目结构。
2. 创建 typed Event。
3. 实现 Mjai JSON Parser。
4. 支持当前数据中实际出现的事件类型。
5. 创建 inspect_mjai.py。
6. 统计游戏、局和事件。
7. 创建 pytest。
8. 创建 README。

## Constraints

禁止实现：

- AI 模型
- Behavior Cloning
- PPO
- RL
- Self-play
- 策略逻辑

禁止一次性加载 几百GB 数据。

## Acceptance Criteria

- 能读取真实 Mjai 文件。
- Parser 测试通过。
- inspect_mjai.py 可以运行。
- 能输出基本牌谱统计。
- 没有未来阶段代码。

## Completion Report

按照 CODEX_RULES.md 第 17 节报告。
```

---

## 20. 第二阶段 Task

```text
# Task: 实现 Replay Engine

## Goal

将 Mjai Event 序列还原为完整 ReplayState。

## Requirements

实现：

- ReplayState
- ReplayEngine
- start_game
- start_kyoku
- tsumo
- dahai
- chi
- pon
- kan
- reach
- reach_accepted
- hora
- end_kyoku
- end_game

## Constraints

禁止实现 AI 策略。

## Acceptance Criteria

真实牌谱可以完整 Replay；
状态转换确定；
测试通过。
```

---

## 21. 第三阶段 Task

```text
# Task: 实现 Decision Extraction

## Goal

从 ReplayState 提取监督学习决策点。

## Requirements

实现：

- DecisionPoint
- Action
- Legal Actions
- Human Action
- DecisionSample

支持：

- discard
- chi
- pon
- kan
- riichi
- ron
- tsumo
- pass

## Critical Constraint

Observation 不得包含任何未来信息。

## Acceptance Criteria

能够从真实 Replay 中提取完整决策样本。
```

---

## 22. 不要为了“完整”而提前开发

项目开发中最重要的行为之一：

> 如果当前阶段已经完成验收，就停止扩展，等待下一阶段任务。

不要自行增加：

```text
Transformer
PPO
Self-play
分布式训练
```

除非任务明确要求。

---

## 23. 最终目标

Codex 的角色是：

```text
工程实现者
测试执行者
代码维护者
```

不是：

```text
项目架构的自主决策者
```

项目总体技术方向由项目文档确定。

当 Codex 发现更好的实现方式时，应提出建议，而不是未经确认修改总体架构。
