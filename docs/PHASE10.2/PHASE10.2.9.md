# Phase10.2.9：公开状态基线加独立critic残差

2026-11-14。依据10.2.8新确认集的线性公开状态模型优于旧critic，开展有界可学习性probe。保持BC-v4 sampling actor和greedy BC-v4对手；不接入PPO、不使用10.2.8确认集。先从新训练集拟合公开状态ridge基线，再只训练独立value encoder/trunk/head去预测残差；actor严格冻结。

新种子950000–950447，256训练/64开发/128最终验证，整局划分、四席平衡。公开基线仅用训练集可选状态拟合，ridge=1；残差训练三轮Adam lr1e-4、每游戏一步、梯度裁剪0.5。主loss为可选状态MSE加0.25倍强制状态MSE，以及0.01倍残差幅度平方。公开状态输入、系数与critic权重分别保存；该probe没有把额外基线接入通用PPO model接口。

开发集按可选状态、每游戏等权残差模型MSE选轮；最终验证一次比较残差模型和同批训练集公开基线，skill=1−MSE_residual/MSE_public，整局bootstrap10000次、PCG64 seed10329。CI下界>0为PASS，上界<0为FAIL，其余INCONCLUSIVE。常数基线辅助报告。即便PASS，尚需验证value接口、GAE和actor完整训练健康，再谈强度。

采集配置SHA 007b39a4977c05bddd087f15f267efe30c697b0e62bdeab19ece31ac04c956fa；probe配置SHA 0a0a2bab90782cafceb0795b927381f895871b78810a74ff1c9fc83c4ed32dd7。入口`scripts/collect_phase10_2_7.py configs/collect_phase10_2_9.json`及`scripts/probe_phase10_2_9_residual.py configs/probe_phase10_2_9.json`。pilot使用此前独立16局，仅验证链路；正式采集预计3–5分钟，probe约1分钟。旧10.2.7/10.2.8测试与确认游戏继续封存。

## 执行结论（2026-11-14）

448场正式采集146.860秒，残差probe13.906秒。开发集与最终验证均未见相对公开线性基线收益：最终skill−0.259%，CI[−1.238%,+0.691%]，INCONCLUSIVE。公开基线相对常数在新验证集skill+6.294%，CI[+2.862%,+9.647%]；开发集表现波动。证据与后续见[PHASE10.2.9_RESULT.md](PHASE10.2.9_RESULT.md)。
