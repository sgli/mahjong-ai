# Dataset Validation Report

- 数据集版本：`decision-v1`
- feature 版本：`feature-v1`（action `action-v1`）
- git commit：`ed9b75b`
- 数据源：`F:/Mahjong AI/mahjong DB/tenhou-houou-2026`
- 构建命令：`scripts/build_dataset.py --data-dir .../tenhou-houou-2026 --output data/processed/decision-v1 --limit 160 --seed 42 --split 0.8,0.1,0.1`

## 1. Split 完整性（按 game_id 切分，无跨 game）

| split | games | samples |
|---|---|---|
| train | 128 | 83409 |
| validation | 16 | 11019 |
| test | 16 | 10516 |
| 合计 | 160 | 104944 |

- train/validation/test 之间 **game_id 无交集**（cross-split overlap = ∅）。
- 切分在 `DatasetBuilder.assign_splits` 按 game_id 一次性分配，单个 game 的所有样本只落入一个 split。

## 2. DecisionSample 正确性（action ∈ legal_actions）

- `samples_skipped = 0`（构建器在 `add_game_samples` 中跳过 action 不在 legal_actions 的样本）。
- 独立复验：遍历 104944 行 `row_to_sample`，`action not in legal_actions` 的样本数 = **0**。
- 非法 label 目标 0，已达成。

## 3. Observation 无泄漏（无未来信息/无对手暗手/无牌山/无 ura）

- Parquet 观察列仅含公开信息：`bakaze, kyoku, honba, kyotaku, oya, scores, riichi, hand(自己), melds(自己), opponents_melds(已亮副露), discards(四家河), dora_markers(表宝牌), turn`。
- **对手暗手、牌山、里宝牌（ura）不在 schema 中**，结构性无泄漏。
- 未来信息只能出现在 `metadata` 离线列，不会进入 observation。
- 抽查样例：`hand=(1m,2m,2s,3p,4p,5m,6p,6s,7p,7p,9m,C,E,P)`, `melds=[]`, `opponents_melds=[[],[],[],[]]`, `legal_actions=13`, `action=discard 9m` —— 合法且无暗信息。

## 4. 发现的问题（已处理，如实记录）

- **stale parquet 残留**：本次构建前 `data/processed/decision-v1/train/` 存在上一轮遗留的 `train-00003.parquet`（旧时间戳），导致独立验证读到的 game/sample 数与 manifest 不一致。
  - 处理：删除该残留文件后复验，与 manifest 完全一致（128/16/16 games，83409/11019/10516 samples）。
  - 建议：`build_dataset.py` 构建前应清理目标目录（或每次写到带版本/时间戳的新目录），避免旧 chunk 混入。此点不影响本次 manifest 的正确性。

## 5. 结论

- Split 完整、无跨 game；illegal label = 0；observation 无泄漏。
- 数据集可用于 Phase 10 BC baseline 训练。
