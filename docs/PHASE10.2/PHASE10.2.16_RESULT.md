# Phase 10.2.16 结果：纯顺位 λ=1 credit propagation

## 预注册

见`PHASE10.2.16.md`。唯一处理为GAE λ由0.95改为1.0，保留纯顺位目标及其他PPO训练设置。

## Pilot、100k训练与screen

Pipeline于2026-11-14 20:25:40启动（PID11320），21:04:58结束，全程约39分19秒。

Pilot验证4096 transitions、lambda1.0、非法0、数值有限，PASS，51.2秒。100k训练验证PASS：100,000 transitions、400,186环境步、170完整局、800 optimizer updates、KL early stop 0、非法0；实际12分41.9秒，checkpoint SHA`b673039721ebfb514df002360ebca06c4bca9ce58e69017d93c8ccc343d376e9`。

新seed1100000–1101999 screen实际1541.948秒（25分42秒）。独立原始逐局核验两臂各2000局、席位各500、完整、非法0、checkpoint身份匹配，bootstrap复算PASS；BC-v4镜像均位1.4935，距1.5为0.0065，sanity通过。候选均位1.512，配对差+0.0185，95%CI[−0.0345125,+0.0705]。有效NO_GO，且候选点估计方向为弱于镜像。λ=1.0未达到均差≤−0.020及CI上界<0，不进入正式10k。

训练轨迹中lambda1.0的每epoch advantages std约0.88–1.34，更新前EV约0.14–0.38；Phase10.2.14 λ=.95 seed43对应约0.16–0.22及EV0.90–0.97。由于训练seed/目标路径不同，这些差异仅是机制线索，不能归因于λ。它们与rank-only长期credit需求和λ1方差增大风险一致，故下轮只筛λ=.99折衷，其他参数不变。
