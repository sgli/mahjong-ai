# 日本立直麻将 AI 数据规范

## 1. 目标

定义从原始 `.mjai.json` 到训练 Decision Dataset 的完整数据链路。

```text
Raw Mjai
  ↓
Parse
  ↓
Validate
  ↓
Replay
  ↓
Decision Extraction
  ↓
Feature Encoding
  ↓
Training Dataset
```

---

## 2. 原始数据

原始 `.mjai.json` 是只读数据。

目录：

```text
data/raw/mjai/
```

禁止原地修改。

---

## 3. Mjai Event

至少支持项目数据中出现的：

```text
start_game
start_kyoku
tsumo
dahai
chi
pon
kan
reach
reach_accepted
hora
end_kyoku
end_game
```

Parser 必须以实际数据 schema 为准，不允许凭空扩展字段含义。

---

## 4. ReplayState

建议包含：

```text
game_id
round_id
bakaze
kyoku
honba
kyotaku
oya
scores

players:
  seat
  hand
  melds
  discards
  riichi
  ippatsu
  score

dora_markers
turn
```

根据规则和牌谱实际信息，可以扩展字段。

---

## 5. 决策窗口

决策不能只定义为：

```text
tsumo → dahai
```

还必须处理：

```text
discard
chi
pon
kan
riichi
ron
tsumo
pass
```

实际可用动作由当前 State 决定。

---

## 6. Action Schema

概念：

```python
Action(
    type,
    tile=None,
    consumed=None,
    target=None,
)
```

例如：

```text
DISCARD(tile)
CHI(consumed, target)
PON(consumed, target)
KAN(...)
RIICHI(tile)
RON(target)
TSUMO
PASS
```

Action schema 必须版本化。

---

## 7. DecisionSample

```python
DecisionSample(
    game_id,
    round_id,
    player_id,
    observation,
    legal_actions,
    action,
    metadata,
)
```

其中：

### observation

只包含决策时可见信息。

### legal_actions

由规则引擎生成。

### action

牌谱中的实际动作。

### metadata

可以保存：

- 数据源
- 规则版本
- 时间
- 玩家信息
- 未来结果等离线分析字段

但 metadata 中任何未来信息不得进入 Observation。

---

## 8. 数据切分

必须按完整 Game 切分。

```text
Games
 ├── train
 ├── validation
 └── test
```

禁止按 DecisionSample 随机切分。

---

## 9. 几百GB 数据处理

禁止：

```python
json.load(几百GB_file)
```

推荐：

```text
streaming
→ parse
→ replay
→ decision
→ chunk
→ parquet
```

建议按合理 chunk size 输出多个文件。

---

## 10. Dataset Version

例如：

```text
raw-v1
validated-v1
decision-v1
decision-v2
```

数据处理逻辑变化时创建新版本，不覆盖旧版本。

---

## 11. 数据质量检查

至少检查：

```text
JSON 合法性
事件顺序
游戏边界
局边界
玩家数量
手牌合法性
牌数量一致性
分数一致性
Replay 是否成功
Decision 是否可提取
Observation 是否包含非法信息
```

---

## 12. 数据统计

正式处理前应统计：

```text
文件数
游戏数
局数
玩家数（若数据支持）
事件数
tsumo 数
dahai 数
chi 数
pon 数
kan 数
reach 数
hora 数
规则配置分布
来源分布
```

如果数据包含平台或段位信息，可以额外统计，但不得假设字段一定存在。

---

## 13. 数据泄漏原则

严禁将以下内容放入训练 Observation：

```text
未来牌
未来动作
对手隐藏手牌
最终顺位
最终得点
未来和牌
未来立直
未来宝牌信息
```

---

## 14. 数据存储

推荐：

```text
Parquet
```

原因：

- 分块
- 列式
- 高效读取
- 支持大规模 Dataset

具体 schema 应在第一版数据统计后确定，而不是提前过度设计。
