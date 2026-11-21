# 400M SWA / gRet 训练进度与结果

北京时间 2026-11-21 21:11 的快照。SWA baseline 已完成，gRet baseline 仍在训练。后续队列为 SWA A → gRet A → SWA B → gRet B。[六组配置与运行入口](../../scripts/README.md#400m-六组训练)

| 实验 | 成功更新 | 最近一次验证 | 验证 loss | 用时 | AMP 梯度重试 | checkpoint 回退 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SWA baseline | 40,000 / 40,000 | 40,000 步 | **2.94313** | 15:03:42 | 5 | 0 |
| gRet baseline | 18,494 / 40,000 | 18,250 步 | **3.06624** | 已跑 7:27:41 | 0 | 0 |

在 **18,250 步**的同 token 对照中，SWA 和 gRet 的验证 loss 分别为 **3.11442** 和 **3.06624**，gRet 低 **0.04817**。

两组均从第 0 步启动一次，八张卡的日志未记录非有限前向。SWA 的 5 次 AMP 梯度溢出均在降低 loss scale 后按原步重试，最终 40,000 步 checkpoint 已通过队列校验。gRet 的数值稳定性记录覆盖本次快照。

两组沿用 Appendix D 的 400M 架构档位，宽度 1024、24 层、FFN 3072、独立输入输出权重。训练共用固定版本 FineWeb-Edu、FP16 混合精度、每步 262,144 tokens 和 40,000 步预算。冻结版本为 `ea779aa`。[冻结配置与代码](https://github.com/chestercc1997/yoco/tree/ea779aa879215fe13010b21caacf3f6040eec4a1)

用时从队列启动计起。完整组计至最终 checkpoint 写盘，训练中组计至快照的最后一次成功更新。验证每 250 步记录一次。

[汇总 CSV](../../evaluation/400m_raw_kv_20261120/summary.csv) · [逐步训练 CSV（gzip）](../../evaluation/400m_raw_kv_20261120/training_curves.csv.gz) · [验证曲线 CSV](../../evaluation/400m_raw_kv_20261120/validation_curves.csv) · [AMP 溢出记录 CSV](../../evaluation/400m_raw_kv_20261120/amp_overflows.csv)
