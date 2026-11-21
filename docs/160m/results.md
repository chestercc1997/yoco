# 160M SWA / gRet 训练结果

八组均完成 40,000 步，最终 checkpoint 已通过训练队列校验。最后一组 gRet C 于北京时间 2026-11-20 16:39 完成。[冻结训练配置与数据](README.md)

| 架构 | 方法 | 最终验证 loss | 相对 baseline Δ | 最后 200 步平均训练 loss | VertiKV 效率 | 训练用时 | 达到 baseline 最终 loss 的时间差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SWA | Baseline | 3.16642 | — | 3.16392 | 1.000× | 6:36:46 | — |
| SWA | A | 3.15680 | −0.00962 | 3.15415 | 1.056× | 6:58:16 | −17 分 40 秒 |
| SWA | B | **3.15495** | **−0.01147** | **3.15204** | **1.068×** | 7:07:49 | −14 分 19 秒 |
| SWA | C | 3.16143 | −0.00499 | 3.15861 | 1.030× | 7:12:08 | +11 分 04 秒 |
| gRet | Baseline | 3.15155 | — | 3.14796 | 1.000× | 4:45:44 | — |
| gRet | A | 3.14984 | −0.00171 | 3.14651 | 1.008× | 4:56:20 | +3 分 14 秒 |
| gRet | B | **3.13905** | **−0.01250** | **3.13576** | **1.071×** | 4:58:46 | −22 分 21 秒 |
| gRet | C | 3.15116 | −0.00039 | 3.14756 | 1.002× | 4:59:13 | +11 分 37 秒 |

B 在两种架构上都取得最低的最终验证 loss。SWA A 最早达到本架构 baseline 的最终 loss。SWA C 最终验证 loss 低于 SWA baseline，训练总用时增加 **8.9%**。gRet C 的验证 loss 比 baseline 低 **0.00039**，训练总用时增加 **4.7%**。结果来自单个 seed。

## 测量口径与数据

- 每组用时从队列启动到最终 checkpoint 写盘。达标时间取验证记录中首次达到同架构 baseline 最终验证 loss 的时间，从各组启动计时，验证间隔为 250 步。
- VertiKV 效率按第 39,801–40,000 步的 token 数逐点对齐，计算 `(方法训练 loss / baseline 训练 loss)^(-1 / 0.057)` 后取平均。该分数估计同 token 下的损失收益。实际运行成本对应训练用时和吞吐。[计算口径](../analysis/VERTIKV_TRAINING_EFFICIENCY.md)
- “160M”指论文架构档位。含独立输入输出矩阵的实际总参数量见 CSV。训练和验证均使用本轮固定版 FineWeb-Edu，历史 softmax 结果采用各自当时的数据与训练方案。[本轮数据与设置](README.md) · [历史结果](../history/README.md)
- 八组都从第 0 步连续训练至 40,000 步，训练与验证 loss 有限。51 次 AMP 梯度溢出均经 loss scale 回退和原步重试后继续训练。八卡日志未记录非有限前向或 checkpoint 恢复。

[汇总 CSV](../../evaluation/160m_raw_kv_20261118/summary.csv) · [逐步训练 CSV（gzip）](../../evaluation/160m_raw_kv_20261118/training_curves.csv.gz) · [验证曲线 CSV](../../evaluation/160m_raw_kv_20261118/validation_curves.csv) · [AMP 溢出记录 CSV](../../evaluation/160m_raw_kv_20261118/amp_overflows.csv)
