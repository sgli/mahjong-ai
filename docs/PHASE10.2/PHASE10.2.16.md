# Phase 10.2.16：纯顺位 PPO 的终局信用分配

日期：2026-11-14。纯顺位seed43 checkpoint在独立10k对BC-v4的均差−0.0167、95%CI[−0.0398,+0.0058]，有效NO_GO。结合轨迹长度，检验单变量`GAE λ: 0.95 → 1.0`是否改善终局名次奖励向早期动作传播。

## 依据与不确定性

纯顺位回报只在终局支付；训练统计约每10k learner transitions含15–18完整局，即每局约560–670个所有座位的决策transition，约140–168个每座位决策。GAE递推残差按`γλ`逐步衰减；λ=.95时，纯终局残差经过140–168个本座位动作后权重约`0.95^140–0.95^168 = 0.00076–0.00018`。因此早期动作的优势主要依赖value bootstrap近似。当前critic初始化来自公开状态特征，无法编码私有手牌；该结构可能在早期动作排序上有偏差。训练日志更新前EV约0.90–0.97，但不是独立终局排序质量证据。

将λ设为1使终局回报无衰减地进入每个同局动作的Monte Carlo回报，可能降低GAE偏差；代价是优势方差增大。旧λ诊断的高方差结论来自点数混合目标，不能直接否决有界纯顺位目标下的λ=1。此假设可被短程数值健康与新seed强度小批证伪。

## 单变量设计

- 起点：BC-v4同一checkpoint与Phase10.2.14纯顺位critic系数；纯顺位reward scale=0，placement bonus不变。
- 主候选100k learner transitions，seed44/env_seed640000/opponent_pool_seed44。网络、lr、gamma1、PPO clip、BC anchor0.1、call anchor1.0、target KL0.01、num_envs1等照纯顺位配置，仅λ从0.95改成1.0；配置`configs/train_ppo_v4_pure_rank_lambda1.yaml`。
- 先跑2×2048 pilot（同λ和奖励），核对4096 transitions、finite loss、illegal0、checkpoint读写；配置`configs/train_ppo_v4_pure_rank_lambda1_pilot.yaml`。
- pilot通过后跑100k训练并核验所有epoch、checkpoint SHA、λ/reward/初值与非法率。
- 新seed1100000–1101999做2,000局screen，候选greedy对3×BC-v4，BC-v4同seed镜像，16 workers；bootstrap seed10380、10000次、normal99-v2。screen GO需配对均差≤−0.020、95%CI上界<0、mirror sanity/完整性/非法门通过；否则本配方NO_GO，不补seed/调门/转正式10k。

## 执行时长

参考纯顺位100k实测12分15秒，预计12–18分钟；同类2000对候选+镜像实测Phase10.2.13约1208秒，预计18–23分钟。含pilot/训练/screen全流程估约31–41分钟，超过20分钟；启动后停止实时监控，等用户通知跑完后再核验。

结果写入`PHASE10.2.16_RESULT.md`。不提交，不更新STATUS.md。
