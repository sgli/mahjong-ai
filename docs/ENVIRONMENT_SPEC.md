# 日本立直麻将 Environment 规范

## 1. 目标

建立一个可靠的日本立直麻将环境，为 RL 和 Self-play 提供统一的状态转移系统。

核心：

```text
Observation
    ↓
Legal Actions
    ↓
Action
    ↓
Rule Transition
    ↓
New State
    ↓
Reward
```

---

## 2. 环境接口

概念：

```python
class MahjongEnv:

    def reset(self):
        ...

    def step(self, action):
        ...

    def legal_actions(self):
        ...

    def observation(self):
        ...

    def done(self):
        ...

    def reward(self):
        ...
```

可以根据实际实现扩展 API。

---

## 3. State

Environment State 至少需要表达：

```text
游戏状态
局状态
四家点数
庄家
当前玩家
手牌
副露
河牌
立直状态
一发状态
宝牌
牌山状态
本场
供托
```

隐藏信息只能存在于 Environment 内部，不能自动暴露给 Policy。

---

## 4. Legal Action

每个状态必须能够生成完整合法动作集合。

例如：

```text
DISCARD
CHI
PON
KAN
RIICHI
RON
TSUMO
PASS
```

具体可用动作由状态决定。

---

## 5. Action Validation

`step(action)` 必须验证 Action。

非法 Action：

```text
必须拒绝
```

不能：

```text
自动修正
```

因为自动修正会隐藏 Policy 和规则系统中的 Bug。

---

## 6. 状态转移

每个 Action 必须产生确定的规则转移。

例如：

```text
TSUMO
 ↓
玩家手牌变化
 ↓
产生打牌决策
```

```text
Dahai
 ↓
河牌变化
 ↓
其他玩家获得响应机会
```

```text
Chi / Pon / Kan
 ↓
副露更新
 ↓
进入对应后续决策
```

---

## 7. 统一规则核心

Replay Engine 和 Environment 尽量使用同一套规则核心。

目标：

```text
Replay
Environment
Evaluation
```

三者对于相同状态和动作应得到一致的规则结果。

如果必须使用两套实现，应建立交叉验证测试。

---

## 8. 终局

环境必须正确处理：

```text
和牌
流局
局结束
半庄结束
```

以及：

```text
本场
供托
庄家
点数变化
最终顺位
```

---

## 9. Reward

Reward 必须可配置。

候选：

```text
最终点数
点数变化
最终顺位
组合 reward
```

不要在第一版将 reward 永久写死。

---

## 10. RL 接口要求

环境必须支持：

```text
reset
step
legal_actions
observation
reward
done
```

并支持大量连续对局。

---

## 11. Determinism

给定：

```text
相同规则
相同 seed
相同初始状态
相同 Action
```

应得到一致结果。

---

## 12. Environment 测试

必须覆盖：

```text
发牌
摸牌
打牌
吃
碰
杠
立直
荣和
自摸
流局
终局
点数结算
```

同时测试非法动作。

---

## 13. 性能

初期优先正确。

后期才优化：

```text
批量 Environment
多进程
GPU Environment
JAX
Ray
```

优化后必须与 Reference Environment 做一致性测试。
