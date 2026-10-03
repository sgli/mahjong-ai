# Dataset Validation Report — decision-v2

- dataset: `data/processed/decision-v2`
- dataset_version: `decision-v2`
- feature_version: `feature-v1`（manifest 与 schema 一致）
- action_schema_version: `action-v1`
- git_commit: `2b2092b`

## 全量统计（manifest + parquet metadata）

| split | files | games | samples |
|---|---|---|---|
| train | 11729 | 1,805,781 | 1,172,197,630 |
| validation | 1473 | 225,737 | 146,516,502 |
| test | 1472 | 225,750 | 146,561,032 |
| 合计 | 14674 | 2,257,268 | 1,465,275,164 |

- files_requested=2,257,337；files_processed=2,257,268；files_error=69（replay 错误已跳过）。
- samples_skipped=5（构建时 5 个非法 label 样本已跳过，未写入数据集）。
- parquet metadata 统计的每 split 行数与 manifest samples **完全一致**。

## Checks

1. **game_id split（无跨 game）**：抽样 train/validation/test 各 6 个 shard（共 1,800,000 行），三个 split 的 game_id **overlap = ∅**；切分由主进程 `assign_splits` 对全量 game_id 一次性完成，结构性保证无跨 game。
2. **action ∈ legal_actions**：抽样 1,800,000 行，illegal label = **0**（构建时已跳过 5，数据集内为 0）。
3. **observation 泄漏**：观察列仅公开信息 `bakaze/kyoku/honba/kyotaku/oya/scores/riichi/hand(自己)/melds(自己)/opponents_melds(已亮副露)/discards(四家河)/dora_markers(表宝牌)/turn`；**无对手暗手、无牌山、无 ura、无未来信息**（未来信息只在 metadata 离线列）。
4. **feature version**：manifest `feature-v1` == schema `feature-v1`。
5. **抽查样例**：train `2021020423gm-00a9-0000-8e385c50.mjai` seat2，hand=14 张、dora_markers=('1p',)、discards=[7,7,6,6]，字段齐全。

## 抽样方法说明

- 全量统计用 manifest + parquet metadata（`ParquetFile.metadata.num_rows`），不读数据体。
- 内容校验（game_id/illegal/schema）采用抽样：每 split 随机抽 6 个 shard（共 18 shard / 1,800,000 行 ≈ 全量样本的 0.12%），覆盖 2,763 个 game。
- 泄漏与 feature version 为结构性检查（schema 层保证），非抽样。

## 结论

✅ **通过，放行训练。** 若任一检查失败则停止训练。
