# 400M SWA / gRet 训练结果

SWA 和 gRet 的 Baseline、raw A、raw B 共六组均完成 40,000 步。最后一组 gRet B 于北京时间 2026-11-25 01:37 完成。每组保留每 5,000 步的完整 checkpoint，最终 checkpoint 已通过队列的 SHA-256 校验。下游评测待完成。 [六组配置与运行入口](../../scripts/README.md#400m-六组训练)

| 架构 | 方法 | 最后一步训练 loss | 末 200 步平均训练 loss | 最终验证 loss ↓ | 总训练用时 | 效率提升 | 达到 Baseline 最终验证 loss 的实测用时差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SWA | Baseline | 2.931291 | 2.930691 | 2.943127 | 15:03:42 | — | — |
| SWA | A | 2.915910 | 2.914464 | 2.927350 | 16:09:11 | +10.24% | **−49 分 34 秒** |
| SWA | B | 2.916735 | 2.914117 | **2.926895** | 16:33:58 | **+10.47%** | −34 分 13 秒 |
| gRet | Baseline | 2.900697 | 2.900860 | 2.916549 | 16:10:48 | — | — |
| gRet | A | 2.888028 | 2.889346 | 2.905943 | 17:16:35 | +7.23% | −31 分 17 秒 |
| gRet | B | 2.878636 | 2.880580 | **2.897827** | 17:42:25 | **+13.11%** | **−1 小时 1 分 12 秒** |

方法 B 在两种架构上都取得了最低的最终验证 loss。SWA A 和 gRet B 分别最快达到各自 Baseline 的最终验证 loss。结果来自单个 seed。

## 测量口径与数据

- 总训练用时从队列启动计到最终 checkpoint 写盘。达标时间取各方法首次达到同架构 Baseline 最终验证 loss 的验证点，负值表示省时。验证间隔为 250 步。
- 效率提升按第 39,801–40,000 步的累计 token 数逐点对齐，将 `(Baseline 训练 loss / 方法训练 loss)^(1 / 0.057)` 取平均，再减 1 转为百分比。 [效率计算口径](../analysis/VERTIKV_TRAINING_EFFICIENCY.md)
- 六组采用宽度 1024、24 层、FFN 3072、独立输入输出权重、固定版本 FineWeb-Edu 和 FP16 混合精度，每步 262,144 tokens。 [冻结配置与代码](https://github.com/chestercc1997/yoco/tree/ea779aa879215fe13010b21caacf3f6040eec4a1)
- 六组均从第 0 步连续训练。36 次 AMP 梯度溢出均通过降低 loss scale 后按原步重试，八卡日志未记录非有限前向或 checkpoint 恢复。

[汇总 CSV](../../evaluation/400m_raw_kv_20261120/summary.csv) · [逐步训练 CSV（gzip）](../../evaluation/400m_raw_kv_20261120/training_curves.csv.gz) · [验证曲线 CSV](../../evaluation/400m_raw_kv_20261120/validation_curves.csv) · [AMP 溢出记录 CSV](../../evaluation/400m_raw_kv_20261120/amp_overflows.csv)
