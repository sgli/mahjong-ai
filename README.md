# mahjong-ai

日本立直麻将 AI 项目。当前进度：**第一阶段（Mjai Parser + 数据统计 + 验证）**、
**第二阶段（Replay Engine）**、**第三阶段（Decision Extraction）**、
**第四阶段（Decision Dataset）**、**第五阶段（Behavior Cloning）**、
**第六阶段 6a（役判定 + 符 + 点数 + 宝牌）**、**第六阶段 6b（Mahjong Environment）**、
**第七阶段（PPO / RL）**、**第八阶段（Self-play + Opponent Pool）** 与
**第九阶段（Benchmark 固定基准 + 指标 + 晋级）** 已完成。

```text
项目初始化
  +
Mjai Parser（16 种类型化事件 + 结构校验 + 统计）
  +
Replay Engine（事件序列 → ReplayState，确定性 + 一致性校验）
  +
Decision Extraction（统一规则核心 + 决策点 + 合法动作 + 决策样本）
  +
Decision Dataset（decision-v1 Parquet，按 Game 切分 train/val/test）
  +
Behavior Cloning（feature-v1 编码 + MLP Policy + BC 训练 + checkpoint）
  +
规则核心补全 6a（役判定 + 符 + 点数 + 宝牌，天凤规则）
  +
Mahjong Environment 6b（确定性完整半庄 + 与 Replay 交叉验证）
  +
PPO / RL 7（Value Head + GAE + clipped PPO + 自对弈轨迹 + checkpoint）
  +
Self-play + Opponent Pool 8（随机/BC/PPO 多样对手池 + 任意四座组合对局）
  +
Benchmark 9（固定对手基准 + 8 项指标 + smoke/rules/benchmark/compare 晋级）
```

已实现的部分**不涉及** 分布式/多 GPU（第十阶段）。

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
│   ├── parser/         # Phase 1：typed events + JSONL parser + 校验 + 统计
│   ├── replay/         # Phase 2：ReplayState + ReplayEngine（确定性状态重建）
│   ├── rules/          # Phase 3：统一规则核心（agari/shanten/furiten/合法动作）
│   ├── decision/       # Phase 3：Action/Observation/DecisionSample/Extractor
│   ├── dataset/        # Phase 4：Parquet schema + 流式 builder + game 级 split + manifest
│   ├── features/       # Phase 5：Action Space + Feature Encoder（feature-v1）
│   ├── model/          # Phase 5：MLP Policy（legal mask → softmax）
│   ├── training/       # Phase 5：BC 训练 + checkpoint；Phase 7：PPO
│   ├── environment/    # Phase 6b：MahjongEnv（确定性完整半庄 + 交叉验证）
│   └── evaluation/     # Phase 8：OpponentPool + Self-play 对局运行器
├── scripts/
│   ├── inspect_mjai.py        # 数据检查与统计入口
│   ├── replay_game.py         # 真实牌谱 Replay + 一致性检查入口
│   ├── extract_decisions.py   # 决策样本提取 + 合法动作覆盖校验入口
│   ├── build_dataset.py       # decision-v1 Parquet 数据集构建入口
│   ├── train_bc.py            # BC 训练入口
│   └── play_selfplay.py       # 随机自对弈验证入口
├── configs/
│   ├── data.yaml
│   └── train_bc.yaml          # BC 训练配置（lr/batch/epochs/seed/模型尺寸）
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

# 真实牌谱 Replay + 一致性检查（前 20 个文件）
.\.venv\Scripts\python.exe scripts\replay_game.py --limit 20

# 决策样本提取 + 合法动作覆盖校验（前 20 个文件）
.\.venv\Scripts\python.exe scripts\extract_decisions.py --limit 20

# 构建 decision-v1 Parquet 数据集（前 20 个文件冒烟）
.\.venv\Scripts\python.exe scripts\build_dataset.py --limit 20 --output data/processed/decision-v1

# 训练 BC baseline（先构建数据集，再训练）
.\.venv\Scripts\python.exe scripts\train_bc.py --epochs 2 --seed 0
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
- `replay/state.py` — ReplayState / PlayerState / Meld（不可变状态快照）
- `replay/engine.py` — ReplayEngine（事件顺序 → 确定性状态 + 一致性校验）
- `rules/tiles.py` — 牌常量 / 归一化 / 计数
- `rules/agari.py` — 和牌判定 / 向听 / 听牌 / 振听
- `rules/legal.py` — 吃碰杠立直可行性 + 合法动作生成
- `rules/decompose.py` — 和牌型分解（4 面子 + 雀头枚举）
- `rules/dora.py` — 表宝牌 / 里宝牌 / 赤宝牌计数
- `rules/yaku.py` — 役判定（1/2/3/6 番与役满 + 情境役）
- `rules/scoring.py` — 符 + 番 → 点数（亲家子家/自摸荣和/封顶）
- `decision/action.py` — Action（8 类）+ ACTION_SCHEMA_VERSION
- `decision/observation.py` — PlayerObservation（玩家可见状态投影，无泄漏）
- `decision/decision.py` — DecisionPoint / DecisionSample
- `decision/extractor.py` — DecisionExtractor（事件序列 → 决策样本 + 覆盖校验）
- `dataset/schema.py` — Parquet schema + 行序列化/反序列化（decision-v1）
- `dataset/split.py` — 按完整 Game 的确定性 train/validation/test 切分
- `dataset/builder.py` — 流式分块 Parquet 构建器
- `dataset/manifest.py` — 数据集清单（版本/计数/配置/git commit）
- `features/action_space.py` — 动作空间（270 维）+ action↔id 映射 + legal mask
- `features/encoder.py` — ObservationEncoder（feature-v1，234 维，只读可见投影）
- `model/policy.py` — MLPPolicy / Policy（legal mask → softmax）
- `training/bc.py` — BC 流式训练 + metrics（loss/acc/top-k/illegal rate）
- `training/checkpoint.py` — checkpoint 保存/加载（权重/配置/指标/版本/git commit）
- `scripts/inspect_mjai.py` — 数据检查与统计入口
- `scripts/replay_game.py` — 真实牌谱 Replay + 一致性检查入口
- `scripts/extract_decisions.py` — 决策样本提取 + 覆盖校验入口
- `scripts/build_dataset.py` — decision-v1 Parquet 数据集构建入口
- `scripts/train_bc.py` — BC 训练入口

PPO/RL、Self-play、Opponent Pool、Benchmark、Environment 属于后续阶段，不在当前实现范围。
