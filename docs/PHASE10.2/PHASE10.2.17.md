# Phase 10.2.17：纯顺位 GAE λ=0.99 折衷筛选

日期：2026-11-14。Phase10.2.15纯顺位λ=.95 seed43 checkpoint的独立10k NO_GO；Phase10.2.16纯顺位λ=1.0 seed44在新2k screen也NO_GO（差+0.0185，95%CI跨0）。λ=1训练记录显示advantages std约0.88–1.34、更新前EV约0.14–0.38；λ=.95 seed43记录分别约0.16–0.22、EV0.90–0.97。不同训练seed意味着这些差异不能因果归于λ，但支持一个可证伪折衷：λ=.99保留更多延迟终局排序信息，同时避免MC λ=1的全部方差。

λ=.99时每座位约140–168步后的终局TD残差系数为`.99^140–.99^168 ≈ 0.245–0.185`，高于λ=.95的约0.00076–0.00018，低于λ=1的1.0。实际结果而非该近似决定是否保留。

## 单变量与门槛

使用纯顺位reward（scale0）、Phase10.2.14同一训练256局拟合的placement critic初值、BC-v4 actor起点；其余网络、PPO设置、gamma1、BC/call anchors、KL、预算、num_envs1与λ=1候选相同，只改λ为.99。新训练seed45/env650000/opponent seed45；先2×2048 pilot，再100k transitions。配置`configs/train_ppo_v4_pure_rank_lambda099.yaml`及pilot配置。

Pilot须4096 transitions、finite loss、非法0、checkpoint读写正常。主训练须100k、各轮finite、illegal0、checkpoint身份/λ正确。通过后用新seed1120000–1121999做候选greedy对3×greedy BC-v4及同seed BC镜像，各2000局、16 workers、bootstrap PCG64 seed10390/10k replicate、normal99-v2。screen GO要求均差≤−.020、CI上界<0、完整/非法/镜像全通过；否则NO_GO，不扩seed/改门/跑正式10k。

## 估时

100k训练历史约12–13分钟，预计12–18分钟；同配置上轮Phase10.2.16 2k候选+镜像screen实际1541.948秒（25分42秒），预计24–30分钟。本流水线估约37–49分钟；超过20分钟，无上限、不实时监控，等用户通知自然完成。结果记入`PHASE10.2.17_RESULT.md`。不提交，不更新STATUS.md。
