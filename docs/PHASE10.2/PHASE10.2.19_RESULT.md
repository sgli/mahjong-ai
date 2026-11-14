# Phase 10.2.19 结果：纯顺位 λ=.99 新 seed 筛选

状态：完成。Phase10.2.19 纯顺位 λ=.99 seed46 的正式 10,000 对评估与独立核验均通过预注册强度门；这是单个训练 seed 的 GO，晋级前仍需独立训练 seed 确认。

## 流水线记录

- 启动：`.venv/Scripts/python.exe scripts/run_phase10_2_19.py`，工作目录 `F:\Mahjong AI\mahjong-ai`；日志 `experiments/ppo_v4_rank_lambda099_audit/pipeline.log`。
- Pilot 4096 transitions 验证通过后继续主训练；100k 训练独立核验 PASS：400,113 env steps、165 完整局、800 updates、KL early-stop 0、非法动作0；config SHA `6b996763…1fece2`，checkpoint SHA `96028583…92c2e`。
- 2,000 对 screen：seed 1,140,000–1,141,999；16 workers；实耗 1,131.006 秒（18分51秒）。独立核验 PASS：候选均位1.4595、BC-v4镜像1.5250、配对差−0.0655，95% CI[−0.1185,−0.0130]；镜像容差0.06440且sanity通过、每臂2,000局/各座位500局、非法动作0。预注册筛选门通过。
- 正式强度评估：seed 1,150,000–1,159,999，10,000对、16 workers、bootstrap PCG64 seed10401。22:18:34启动，23:53:04结束，实耗5,668.839秒（94分29秒）；候选臂2,864.289秒，镜像臂2,800.322秒。
- 正式结果：candidate checkpoint SHA `96028583826ccc25bd7f8234313cbba1827d5f4fafe7c770160769c3b4f92c2e`；候选均位1.4847，BC-v4镜像1.5136；配对差−0.0289，95% CI[−0.0519,−0.0055]。每臂10,000局、各座位2,500局；两臂完整、非法动作0。镜像偏差|1.5136−1.5|=0.0136，小于normal99-v2容差0.02880，sanity通过。
- `scripts/verify_phase10_2_19_eval.py strength` 独立复算逐局、seed/座位、SHA、bootstrap CI与镜像门，PASS；决策 GO，满足均差≤−0.020及CI上界<0。
- 独立验证：`scripts/verify_phase10_2_19_pilot.py`、`verify_phase10_2_19_training.py`、`verify_phase10_2_19_eval.py`。
- 当前不把 Phase10.2.17 的部分训练 checkpoint 当作本阶段数据；10.2.19 使用 seed46 新训练。

## 结论与后续

screen与正式10k结果均达到预注册GO门，正式结果已独立复核。seed46的纯顺位λ=.99候选通过本次强度门；这支持继续做独立训练seed确认，但不能仅凭一个训练seed宣称跨seed稳健或自动替换BC-v4基线。后续确认应使用全新训练seed和未使用过的评估seed，不根据本轮强度结果再调参。
