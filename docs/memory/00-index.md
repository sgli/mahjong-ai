# 项目记忆索引

updated: 2026-11-15
branch: master（工作区有未提交改动）
phase: Phase10.2.19 λ=.99 seed46正式10k强度GO；独立训练seed确认待进行
current_goal: 预注册新训练seed确认，验证单seed强度结果能否复现
best_result: BC-v4 仍为当前基线；PPO seed46通过正式10k门（delta−0.0289，95% CI[−0.0519,−0.0055]），独立训练seed待确认
running: 无；Phase10.2.19正式10k于23:53完成，pipeline退出
blocking: 单个训练seed虽过正式门，仍需独立训练seed确认；BC-v4暂为强度基线
next_step: 预注册独立训练seed及全新评估seed，不按本轮结果调参
do_not: 不提交；不更新STATUS.md；不续跑Phase10.2.17 partial；预计>20m批次启动后停止监控

## 导航

| 文件 | 内容 |
|---|---|
| [10-project.md](10-project.md) | 目标、版本、约束 |
| [20-env.md](20-env.md) | 已验证环境信息 |
| [30-experiments.md](30-experiments.md) | BC-v3 保留证据与新阶段状态 |
| [40-decisions.md](40-decisions.md) | Phase 10 重启决策 |
| [50-lessons.md](50-lessons.md) | 数据复用和版本边界 |
| [60-todo.md](60-todo.md) | 当前待办 |
| [PHASE10.2_STRENGTH_SCREE_PASS.md](../PHASE10.2_STRENGTH_SCREE_PASS.md) | 从首轮 PPO 到 seed46 正式强度 GO 的汇总路径 |
| [PHASE10.2/](../PHASE10.2/) | Phase 10.2 原始方案、预注册和结果归档；旧记忆路径统一按此目录解析 |

## 最近变更

- 2026-11-15：40份 `PHASE10.2.*.md` 归档到 `docs/PHASE10.2/`；汇总表位于 `docs/PHASE10.2_STRENGTH_SCREE_PASS.md`。仅整理文档，未改模型或跑批。
- 2026-11-14：Phase10.2.19正式10k seed46 GO：delta−0.0289、95% CI[−0.0519,−0.0055]，镜像/逐局/非法门PASS，实耗5668.839s；须独立训练seed确认后才晋级。
- 2026-11-14：Phase10.2.19 screen独立核验GO：2k对delta−0.0655、95% CI[−0.1185,−0.0130]，mirror/数据/非法门通过；18分51秒。
- 2026-11-14：Phase10.2.19 seed46 100k训练PASS：400113 env steps/165完整局/800更新/KL-stop0/illegal0；checkpoint SHA 96028583…92c2e。
- 2026-11-14：用户澄清允许单变量实验，但失败后须优先多因素诊断；组合改动须有相互作用/多因素证据支持。
