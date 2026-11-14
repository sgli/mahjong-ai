# Phase 10.2.3 结果：gamma=1.0 单变量筛选

状态：训练与10k-pair评估均完成；镜像 sanity 失败使正式强度门无效，不能晋升。日期：2026-11-13。预注册见 [PHASE10.2.3_PREREG.md](PHASE10.2.3_PREREG.md)。

2026-11-13实现审查更正：本轮value_separate_trunk只分离MLP，embedding/GRU仍共享；本轮没有检验完整critic隔离。缺陷、修复与有界验证见`PHASE10.2.4_RESULT.md`。

## 训练完成

- 只将 gamma 从0.99改为1.0；其余训练设置与 Phase10.2.2 独立 value trunk方案相同。配置 `configs/train_ppo_v4_gamma1.yaml`，SHA-256 `479de60059ceaba240f1ed62e0743c4517ed52d788314f9374ce8610d8ff76bb`。
- seed42，10 epochs×10,000 learner transitions；总100,000 learner transitions、399,974 environment steps；各轮 illegal rate 0。checkpoint `experiments/ppo_v4_gamma1/seed42_100k/best.pt` 为固定 epoch10，SHA-256 `a7e14c9ba83b64d92da61c21a2d3fe8d51569d290410b0c73b5bc558d92bdf55`。
- 训练从21:39:18到21:48:41，约563秒（9分23秒）；5局self-play smoke均位1.5，不作强度证据。

## 事后诊断

固定 decision-v3 validation 前100,000行，epoch10 与 BC-v4 greedy 一致率85.912%、gold top-1 67.573%、KL(BC‖PPO)0.07470 nat。Phase10.2.2 gamma0.99分别85.606%、67.244%、0.07434 nat。差异较小，不能据此推断对局强度或改选checkpoint。原始输出 `experiments/ppo_v4_gamma1/diagnostics/policy_drift_by_epoch_validation_100k.json`。

## 强度评估

按预注册，candidate PPO对3×BC-v4、mirror BC-v4对3×BC-v4；seed660000–669999，每臂10,000局，16 workers，Greedy/T=1，bootstrap seed10322。输出 `experiments/ppo_v4_gamma1/strength_10k_seed660000_v1/`。实际耗时5,519.305秒（约92分），candidate 2,806.233秒、mirror 2,708.994秒。

| 指标 | gamma1 PPO | BC-v4 mirror |
|---|---:|---:|
| 局数 | 10,000 | 10,000 |
| 平均顺位 | 1.5744 | 1.4730 |
| 顺位分布（0/1/2/3） | 2,350 / 2,367 / 2,472 / 2,811 | 2,534 / 2,551 / 2,566 / 2,349 |
| 和牌/局 | 2.3587 | 2.2651 |
| 放铳/局 | 1.6290 | 1.3320 |
| 立直/局 | 2.3105 | 2.5726 |
| 副露/局 | 9.8691 | 4.6971 |
| 非法动作 | 0 | 0 |

CSV独立复算确认10,000个唯一种子、四席各2,500局、两臂全部完成且非法动作0；checkpoint SHA与冻结值匹配。逐seed配对差（PPO−mirror）+0.1014，bootstrap95%CI [+0.0745,+0.1282]，与result.json一致。

**正式评估有效性未通过。** mirror均位1.4730，落在预注册的`1.5±0.02`以外；`valid_run=false`，runner记`NO_GO`。不能事后放宽阈值或将本轮当成有效的正式强度门。候选均位1.5744，按10k局的样本标准误约0.0113，描述性95%区间约[1.5523,1.5965]，且配对差区间全正；这些是候选偏弱的探索性证据，不代替有效门槛。

各席镜像均位为1.4640/1.4876/1.4724/1.4680，均无单席异常；checkpoint、局数、轮转和配对重算均正常。镜像整体偏移约2.4个标准误。若同策略四席的被选顺位近似独立均匀，10k样本的固定`±0.02`窗口约有7%的随机误拒率；这为当前失败提供合理解释，但不能排除种子与座位相关的系统偏差。该阈值必须在下一轮预注册前重新校准，本轮不补样本。

## 与 Phase 10.2.2 的比较及后续

gamma0.99候选在另一组10k种子上的均位1.5838，本轮gamma1.0候选1.5744，差−0.0094；两轮镜像均位分别1.4994与1.4730。种子范围不同且本轮镜像 sanity 失败，不能把配对差+0.0844→+0.1014解释为gamma使强度变差。行为方面仍明显多副露、更多放铳、较少立直；仅改gamma没有消除该模式。

BC-v4仍是强度基线，不晋升gamma1.0 checkpoint。下一步先修订未来评估的镜像有效性门，随后调查reward规模与GAE长程信用分配；新训练须独立预注册。
