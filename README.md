# mahjong-ai

日本立直麻将 AI 项目。当前处于 **第一阶段（Phase 1）**：

```text
项目初始化
  +
Mjai Parser
  +
数据统计
  +
验证
```

本阶段的唯一目标是：正确解析 `.mjai.json` 牌谱、做基础结构校验并输出统计，
**不涉及** 任何 AI 模型、Replay、策略或训练逻辑。

## 文档

项目的总体架构与各阶段规格见 [`docs/`](docs/)，开发前必读：

- `docs/PROJECT_PLAN.md` — 总体工程规格
- `docs/ARCHITECTURE.md` — 模块边界与推荐目录
- `docs/DATA_SPEC.md` — 数据链路规范
- `docs/CODEX_RULES.md` — 开发行为规范（含各阶段 Task 定义）

## 数据

原始 `.mjai.json` 数据为 **只读**，位于仓库之外（本机为
`F:\Mahjong AI\mahjong DB\tenhou-houou-2026`）。数据路径通过
[`configs/data.yaml`](configs/data.yaml) 外置配置，可用 CLI 参数
`--data-dir` 或环境变量 `MAHJONG_DATA_DIR` 覆盖。

## 目录结构

```text
mahjong-ai/
├── configs/            # 配置（数据路径等，禁止硬编码进代码）
├── data/               # raw / validated / processed（占位，原始数据在仓库外）
├── src/mahjong/
│   └── parser/         # Phase 1：typed events + JSONL parser + 校验 + 统计
├── scripts/
│   └── inspect_mjai.py # 数据检查与统计入口
├── tests/              # pytest 测试
├── docs/               # 项目文档
└── pyproject.toml
```

## 环境搭建

需要 Python 3.11+（本机使用 Python 3.12）：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## 运行

```powershell
# 快速冒烟（前 100 个文件）
.\.venv\Scripts\python.exe scripts\inspect_mjai.py --limit 100

# 全量统计
.\.venv\Scripts\python.exe scripts\inspect_mjai.py
```

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## 事件类型

解析器以实际数据 schema 为准，支持以下 16 种事件：

`start_game`, `start_kyoku`, `tsumo`, `dahai`, `chi`, `pon`,
`daiminkan`, `ankan`, `kakan`, `dora`, `reach`, `reach_accepted`,
`hora`, `ryukyoku`, `end_kyoku`, `end_game`

> 说明：`docs/DATA_SPEC.md` 中将杠统一记作 `kan`；实际数据区分为
> `daiminkan`（大明杠）/ `ankan`（暗杠）/ `kakan`（加杠），解析器按实际 schema 实现。

## 模块边界

- `parser/events.py` — 类型化事件（不可变 dataclass）
- `parser/parser.py` — JSONL 流式解析（不实现规则状态转移）
- `parser/validate.py` — 基础结构校验（事件字段 + 局/游戏边界）
- `parser/stats.py` — 流式统计聚合
- `scripts/inspect_mjai.py` — 命令行入口

完整的状态重建、决策提取等属于后续阶段，不在本阶段实现。
