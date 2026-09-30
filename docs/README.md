# 日本立直麻将 AI 项目文档

这是一套面向 Codex / Coding Agent 的项目规格文档。

## 文档关系

```text
PROJECT_PLAN.md
      │
      ├── ARCHITECTURE.md
      ├── DATA_SPEC.md
      ├── MODEL_SPEC.md
      ├── ENVIRONMENT_SPEC.md
      ├── TRAINING_SPEC.md
      └── CODEX_RULES.md
```

## 使用方式

将这些文件放入项目：

```text
mahjong-ai/
└── docs/
    ├── PROJECT_PLAN.md
    ├── ARCHITECTURE.md
    ├── DATA_SPEC.md
    ├── MODEL_SPEC.md
    ├── ENVIRONMENT_SPEC.md
    ├── TRAINING_SPEC.md
    ├── CODEX_RULES.md
    └── README.md
```

然后要求 Codex 在执行项目任务前读取 `docs/PROJECT_PLAN.md` 以及与当前任务相关的专项规范。

## 开发主线

```text
Mjai
 ↓
Parser
 ↓
Replay
 ↓
Decision Dataset
 ↓
Behavior Cloning
 ↓
Environment
 ↓
PPO
 ↓
Self-play
 ↓
Opponent Pool
 ↓
Benchmark
```

## 当前阶段

第一阶段只做：

```text
项目初始化
+
Mjai Parser
+
数据统计
+
验证
```

完成后再进入 Replay。
