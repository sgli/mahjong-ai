# Phase10.2.10 执行结果

2026-11-14：版本化 value、14 项定向测试和 2×2048 pilot 已通过。pilot 的 epoch2 checkpoint 可由正式 PPO loader 严格恢复为 `RiverSequencePolicy`，公开 value 模式仍在。

## 小批量安全门（已完成）

在冻结的 seed980000–980127、128 对、16 workers、bootstrap seed10330 上，候选均位1.5000，BC-v4镜像均位1.4141，配对差+0.08594，95% bootstrap CI[−0.08594,+0.25781]。candidate/mirror均为完整有效局，非法动作0，座位各32，normal99-v2镜像sanity通过；`valid_run=true`。配对差满足预设的≤+0.20安全门，故允许100k正式训练。该CI跨0且仅128对，不推论强度提升；runner打印的`NO_GO`属于正式强度GO门，对本轮安全筛选不适用。证据：`experiments/ppo_v4_public_value/safety_128pair_v1/result.json`、同目录逐局及paired CSV。

独立读取paired CSV复算确认：128行、128个唯一seed、seat=index%4、均差0.0859375与JSON一致；候选文件SHA也与JSON身份字段一致。该核验不复用runner的配对均值计算。

## 正式训练与强度评估

已检查自动门控脚本`run_phase10_2_10.py`：训练结束后逐项核验10轮/100k learner transitions、非法率与loss、冻结配置与安全结果SHA、best/latest同权重、公开value checkpoint加载；只有通过才将状态推进到`evaluation`并启动预注册的10k/臂强度评估。

正式训练实测733.687秒（12分14秒），比事前20–35分钟估计快；10轮共100,000 learner transitions、400,107 environment steps、163完整游戏、800次optimizer更新，各轮非法率均0，loss均有限。epoch10的KL-to-BC=0.005346；更新前自举target的value EV=0.613635，仅作训练内诊断，不当独立MC泛化证明。best/latest同为epoch10且权重一致，best SHA-256=`43174f67c8cf1eb625828b3810a5eb80f1f13d581d84090a6a3ec4c962d85c7f`，checkpoint model_version=`ppo-v4-public-value-v1`，严格加载保持公开value模式。冻结配置SHA=`1718b504b5ced5f02b7dcb1309a64f44bd45880f1d82894e53d0e77071ae72a2`。健康门通过。

正式评估使用新seed720000–729999、每臂10,000局、16 workers、bootstrap seed10331；candidate为上述epoch10 checkpoint，镜像/对手均为冻结BC-v4。无时间上限，事前预计评估90–120分钟（参考上轮同规模100分19秒）；实际候选臂2760.935秒、镜像臂2742.208秒、评估总5504.937秒（91分45秒），全流水线6239.484秒（约104分钟）。状态文件`experiments/ppo_v4_public_value/phase10_2_10_run_v1/state.json`标记`complete`。

## 10,000对正式强度门：有效 NO_GO

PPO候选均位1.5055，BC-v4镜像均位1.4871；配对平均顺位差（PPO−镜像）**+0.0184**，逐对bootstrap 95%CI **[−0.0053,+0.0422]**。预注册GO要求差≤−0.020且CI上界<0；两项均未达到。镜像偏离理论1.5为−0.0129，落在normal99-v2的±0.02880内；两臂各10,000完整有效局、非法动作0、四座各2500，故这是有效NO_GO而非无效跑批。区间跨0，不能断言候选在目标对手池上显著更差，也不能把未显著差异解释成等效。

独立核验`python scripts/verify_phase10_2_10.py`通过：直接读取两臂原始逐局文件，核对720000–729999每个seed恰好一次、game_index与seat=index%4、四名次排列、完成状态/错误/非法动作；由原始顺位重建10,000条CSV逐行，重新使用PCG64 seed10331做10,000次逐对bootstrap，均差和CI与runner一致。candidate SHA=`43174f67c8cf1eb625828b3810a5eb80f1f13d581d84090a6a3ec4c962d85c7f`，镜像/对手BC-v4 SHA=`814c6eba4e163d126cff7bc27194f11612263c745328abd1b1d9ae94b6716937`，均与实际文件匹配。核验产物：`experiments/ppo_v4_public_value/strength_10k_seed720000_v1/independent_verification.json`；原始结果和CSV同目录。

行为侧候选每局副露5.5935、放铳1.3831、立直2.5176，镜像分别4.7211、1.3443、2.5628。副露差+0.8724/局与过去高副露现象方向相同，是后续动作机会级归因的线索；汇总计数不能证明副露是强度未达标的因果原因。训练内value EV和公开线性基线的离线回报预测收益，也不能替代对局顺位证据。停止本配方扩展，BC-v4保持强度基线；不补本轮样本、不调整门槛、不据此晋升PPO。下一步先在现有对局与训练轨迹上定位哪些动作类别和局面导致PPO/BC选择分歧、优势估计方向是否与终局回报对齐，再决定是否有值得预注册的新机制；任何后续新候选使用新的评估seed。
