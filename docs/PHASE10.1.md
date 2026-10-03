# Mahjong AI Phase 10.1 Large Scale Behavior Cloning Training


当前状态：

Phase 1-10 baseline 已完成。

已有：

- Mjai Parser
- Replay Engine
- Decision Dataset
- Feature Encoder
- BC Policy
- Benchmark Pipeline


目标：

训练大规模 Human Policy Model。


---

## Task 1 Dataset Expansion


使用完整 mjai 数据集。


要求：

生成：

dataset_version:

decision-v2


feature:

feature-v1


action:

action-v1


必须保持：

schema 不变。


生成：

manifest.json


记录：

- game_count
- round_count
- sample_count
- split
- git_commit


---

## Task 2 Dataset Validation


训练前必须执行：


检查：

1. game_id split

2. action in legal_actions

3. observation leakage

4. feature version


生成：

DATASET_VALIDATION_REPORT.md


如果失败：

停止训练。


---

## Task 3 Large Scale BC Training


使用：

PyTorch


保持：

当前 Policy Architecture


不要修改：

- Replay
- Dataset Schema
- Feature Encoder


训练记录：

必须包含：

- dataset_version
- feature_version
- model_version
- git_commit
- seed
- batch_size
- learning_rate
- epoch


保存：

experiments/


结构：


experiments/

bc_v2/

    config.yaml

    checkpoints/

    metrics.json

    README.md


---

## Task 4 Evaluation


训练完成后：

执行 Benchmark。


至少：

10000 games


对手：

- random
- bc-v1
- baseline


记录：

- mean rank
- rank distribution
- win rate
- deal in rate
- riichi rate
- call rate


生成：

BENCHMARK_BC_V2_REPORT.md


---

## 禁止

不要：

- 修改 Feature
- 修改 Action Space
- 修改 Environment
- 引入 PPO
- 引入 Self-play


目标：

获得稳定的大规模 BC Policy。