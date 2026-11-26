# 400M SWA / gRet 训练结果

SWA 和 gRet 的 Baseline、raw A、raw B 共六组均完成 40,000 步及八项全量下游评测。最后一组 raw gRet B 于北京时间 2026-11-25 01:37 完成训练。每组保留每 5,000 步的完整 checkpoint，最终 checkpoint 已通过队列的 SHA-256 校验。 [六组配置与运行入口](../../scripts/README.md#400m-六组训练)

| 架构 | 方法 | 最后一步训练 loss | 末 200 步平均训练 loss | 最终验证 loss ↓ | 总训练用时 | 效率提升 | 达到 Baseline 最终验证 loss 的实测用时差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SWA | Baseline | 2.931291 | 2.930691 | 2.943127 | 15:03:42 | — | — |
| SWA | A | 2.915910 | 2.914464 | 2.927350 | 16:09:11 | +10.24% | **−49 分 34 秒** |
| SWA | B | 2.916735 | 2.914117 | **2.926895** | 16:33:58 | **+10.47%** | −34 分 13 秒 |
| gRet | Baseline | 2.900697 | 2.900860 | 2.916549 | 16:10:48 | — | — |
| gRet | A | 2.888028 | 2.889346 | 2.905943 | 17:16:35 | +7.23% | −31 分 17 秒 |
| gRet | B | 2.878636 | 2.880580 | **2.897827** | 17:42:25 | **+13.11%** | **−1 小时 1 分 12 秒** |

方法 B 在两种架构上都取得了最低的最终验证 loss。SWA A 和 gRet B 分别最快达到各自 Baseline 的最终验证 loss。结果来自单个 seed。

## raw 模型的下游评测

准确率为八项全量零样本任务的等权平均。百分点差和相对增幅均以同架构 Baseline 为准。 [逐任务准确率](../../evaluation/400m_raw_kv_20261120/downstream.csv)

| 架构 | 方法 | 八项平均准确率 ↑ | 相对 Baseline 的百分点差 | 相对增幅 |
| --- | --- | ---: | ---: | ---: |
| SWA | Baseline | 43.40% | — | — |
| SWA | A | 42.42% | −0.98 | −2.25% |
| SWA | B | **43.82%** | **+0.42** | **+0.97%** |
| gRet | Baseline | **44.72%** | — | — |
| gRet | A | 41.70% | −3.02 | −6.75% |
| gRet | B | 42.36% | −2.36 | −5.27% |

## 同配方 gRet softmax 对照

softmax A/B 共用上表的 gRet Baseline、seed 42、训练代码和数据清单，对前 12 层 K/V 融合系数使用温度 1 的 softmax。两组均训练到 40,000 步并完成同口径的八项下游评测。B 曾在 10,000 步完整保存后暂停，待 raw 下游评测结束，恢复同一 checkpoint 继续训练。 [同配方 Baseline 复现说明](../YOCO_BASELINE_REPRODUCTION.md)

| 400M gRet | 最终验证 loss ↓ | 末 200 步效率系数 | 相对效率 | 八项平均准确率 ↑ | 相对 Baseline 的百分点差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | **2.916549** | 1.0000× | — | **44.72%** | — |
| softmax A | 2.918049 | 0.9951× | −0.49% | 43.91% | −0.80 |
| softmax B | 2.917263 | 0.9986× | −0.14% | 44.15% | −0.57 |

同一配方下，raw A/B 的末段 loss 效率较高，softmax A/B 在八项下游评测中的平均分高于各自对应的 raw 方法。验证 loss 和下游准确率分别报告。 [softmax 汇总与 30,000 步对照](../../evaluation/gret_400m_softmax_20261125/summary.csv) · [逐步训练 CSV（gzip）](../../evaluation/gret_400m_softmax_20261125/training_curves.csv.gz) · [验证曲线 CSV](../../evaluation/gret_400m_softmax_20261125/validation_curves.csv) · [AMP 记录](../../evaluation/gret_400m_softmax_20261125/amp_overflows.csv) · [逐任务准确率](../../evaluation/gret_400m_softmax_20261125/downstream.csv)

## 测量口径与数据

- 总训练用时从队列启动计到最终 checkpoint 写盘。达标时间取各方法首次达到同架构 Baseline 最终验证 loss 的验证点，负值表示省时。验证间隔为 250 步。
- 效率提升按第 39,801–40,000 步的累计 token 数逐点对齐，将 `(Baseline 训练 loss / 方法训练 loss)^(1 / 0.057)` 取平均，再减 1 转为百分比。 [效率计算口径](../analysis/VERTIKV_TRAINING_EFFICIENCY.md)
- 六组采用宽度 1024、24 层、FFN 3072、独立输入输出权重、固定版本 FineWeb-Edu 和 FP16 混合精度，每步 262,144 tokens。 [冻结配置与代码](https://github.com/chestercc1997/yoco/tree/ea779aa879215fe13010b21caacf3f6040eec4a1)
- raw 对照的六组均从第 0 步连续训练。36 次 AMP 梯度溢出均通过降低 loss scale 后按原步重试，八卡日志未记录非有限前向或 checkpoint 恢复。

[汇总 CSV](../../evaluation/400m_raw_kv_20261120/summary.csv) · [逐步训练 CSV（gzip）](../../evaluation/400m_raw_kv_20261120/training_curves.csv.gz) · [验证曲线 CSV](../../evaluation/400m_raw_kv_20261120/validation_curves.csv) · [AMP 溢出记录 CSV](../../evaluation/400m_raw_kv_20261120/amp_overflows.csv)
