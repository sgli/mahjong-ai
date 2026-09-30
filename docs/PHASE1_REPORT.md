# 第一阶段完成报告：项目初始化 + Mjai Parser + 数据统计 + 验证

> 项目：`mahjong-ai`（日本立直麻将 AI）
> 阶段：Phase 1（对应 `docs/CODEX_RULES.md` §19 第一阶段 Task）
> 状态：**已完成并通过验收**
> 数据源：`F:\Mahjong AI\mahjong DB\tenhou-houou-2026`

---

## 1. 目标与范围

本阶段唯一目标：正确解析 `.mjai.json` 牌谱、做基础结构校验并输出统计。

```text
项目初始化
  +
Mjai Parser
  +
数据统计
  +
验证
```

**明确不包含**（按文档约束，禁止提前开发）：

- AI 模型 / Behavior Cloning / PPO / RL / Self-play
- 策略逻辑
- Replay Engine、Decision Extraction 等后续阶段代码
- 一次性加载数百 GB 数据（必须流式处理）

---

## 2. 交付文件

在 `F:\Mahjong AI\mahjong-ai` 下新建：

```text
pyproject.toml                     # 项目元数据 + pytest 配置（src layout）
.gitignore
configs/data.yaml                  # 数据路径外置配置
README.md                          # 项目 README
src/mahjong/__init__.py
src/mahjong/parser/__init__.py
src/mahjong/parser/events.py       # 16 种类型化事件（不可变 dataclass）
src/mahjong/parser/parser.py       # JSONL 流式解析器
src/mahjong/parser/validate.py     # 基础结构校验（事件字段 + 局/游戏边界）
src/mahjong/parser/stats.py        # 流式统计聚合
scripts/inspect_mjai.py            # 检查与统计命令行入口
tests/test_events.py
tests/test_parser.py
tests/test_validate.py
tests/test_stats.py
tests/test_integration_real_data.py
data/{raw,validated,processed}/.gitkeep
```

另建立虚拟环境 `.venv`（Python 3.12.3），安装依赖：

```text
pytest 9.1.1
PyYAML 6.0.3
```

---

## 3. 实现内容

### 3.1 Typed Event（类型化事件）

以**实际数据 schema** 为准，支持全部 16 种事件（不可变 `frozen dataclass`）：

```text
start_game     start_kyoku    tsumo          dahai
chi            pon            daiminkan      ankan
kakan          dora           reach          reach_accepted
hora           ryukyoku       end_kyoku      end_game
```

### 3.2 Mjai Parser（解析器）

- 流式逐行解析 JSONL（一个 JSON 对象一行），绝不整文件载入内存。
- 严格校验字段类型与必填项（`MjaiParseError` 携带行号上下文）。
- 支持坏行跳过并记录（`parse_file(..., on_error=...)`），单坏行不致中断整条流水线。

### 3.3 基础结构校验

- `validate_event`：座位 0–3、牌合法性、手牌 13 张、副露张数、宝牌指示牌合法性等。
- `GameValidator`：游戏/局边界、`start_kyoku`/`end_kyoku` 配对、事件越界检查。

> 完整一致性校验（手牌数守恒、点棒结算、回合顺序）属于 Phase 2 Replay Engine，不在本阶段范围。

### 3.4 数据统计

`StatsCollector` 聚合：游戏数、局数、各事件数、规则/配置分布（`aka_flag`、`bakaze`、`kyoku_first`、`tsumogiri`）。

### 3.5 inspect_mjai.py（命令行入口）

数据路径解析优先级：CLI `--data-dir` > 环境变量 `MAHJONG_DATA_DIR` > `configs/data.yaml`（未硬编码路径）。每 1000 文件输出进度日志。

```powershell
# 快速冒烟（前 100 个文件）
.\.venv\Scripts\python.exe scripts\inspect_mjai.py --limit 100

# 全量统计
.\.venv\Scripts\python.exe scripts\inspect_mjai.py
```

---

## 4. 测试

```text
pytest -v  →  51 passed in 0.15s
```

覆盖范围：

- 事件类型定义与不可变性
- 16 种事件解析
- 非法输入（未知类型 / 缺字段 / 错类型 / 错长度）
- 流式解析与坏行跳过
- 字段与局/游戏边界校验
- 统计聚合
- 读取真实数据的集成测试（无数据时自动 skip）

---

## 5. 全量数据统计结果

运行 `inspect_mjai.py` 全量处理结果（耗时 40.8s，吞吐约 32 万事件/秒）：

| 指标 | 值 |
|---|---|
| 文件数 | 12,188 |
| 游戏数 | 12,188 |
| 局数 | 129,179 |
| 事件总数 | 13,084,580 |
| 不同玩家数 | 1,017 |
| 解析错误 | **0** |
| 事件级问题 | **0** |
| 游戏级问题 | **0** |

### 事件类型分布

| 事件 | 数量 |
|---|---|
| `dahai` | 6,205,192 |
| `tsumo` | 6,013,345 |
| `pon` | 145,275 |
| `start_kyoku` | 129,179 |
| `end_kyoku` | 129,179 |
| `hora` | 109,592 |
| `chi` | 102,364 |
| `reach` | 93,838 |
| `reach_accepted` | 90,571 |
| `ryukyoku` | 20,313 |
| `start_game` | 12,188 |
| `end_game` | 12,188 |
| `dora` | 10,585 |
| `ankan` | 6,048 |
| `kakan` | 4,062 |
| `daiminkan` | 661 |
| **kan（合计）** | **10,771** |

### 规则 / 配置分布

| 配置项 | 分布 |
|---|---|
| `aka_flag` | `True`: 12,188 |
| `bakaze` | `E`: 70,031 · `S`: 58,536 · `W`: 612 |
| `kyoku_first` | `0`: 12,188 |
| `tsumogiri` | `False`: 3,846,056 · `True`: 2,359,136 |

---

## 6. 已知限制 / 说明

1. **Python 版本**：系统默认 `python` 为 3.9.13，文档要求 3.11+；本机另有 `py -3.12`（3.12.3），已用其建立 `.venv` 并满足要求。
2. **`kan` → 实际三变体**：`DATA_SPEC.md` 将杠统一记为 `kan`，但真实数据拆分为 `daiminkan` / `ankan` / `kakan`；按「以实际数据 schema 为准」实现，并已在 README 注明。
3. **目录布局微调**：代码统一放在 `src/mahjong/parser/`（命名空间包）而非文档示例的 `src/parser/`，以避免与 Python 标准库 `parser` 重名；总体架构（Parser → Replay → …）未变。
4. **数据特征**：全部 `aka_flag=true`、`kyoku_first=0`；`bakaze` 除 E/S 外出现 612 个 `W`（西入局），已按合法风向处理。
5. **范围控制**：完整一致性校验（手牌数守恒、点棒结算、回合顺序）属于 Phase 2；原始 660MB 数据未复制进仓库，仅通过配置引用。

---

## 7. 下一步

等待进入第二阶段 Task（Replay Engine）：

```text
ReplayState
ReplayEngine
start_game / start_kyoku / tsumo / dahai / chi / pon / kan / reach /
reach_accepted / hora / end_kyoku / end_game 状态转移
```

在进入下一阶段前，不主动扩展后续阶段代码。
