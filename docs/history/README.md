# 历史实验与结果

| 实验 | 结果与原始指标 | 设计及运行记录 |
| --- | --- | --- |
| gRet 160M baseline/A/B | [三组报告](../../evaluation/ab_baseline_20261104/README.md) | [历史 A/B 公式](plan.md) · [四卡方案](../PLAN_AB_4GPU.md) · [一页设计图](../plan_feishu_concise.pdf) |
| gRet 400M baseline/A/B | [三组报告](../../evaluation/gret_400m_3way_20261108T194918Z/README.md) | [baseline](GRET_400M_BASELINE_QUEUE.md) · [A 数值修复](PLAN_A_400M_8GPU.md) · [B](PLAN_B_400M_8GPU.md) |
| SWA 160M、400M baseline/A/B | [160M](../../evaluation/swa_160m_ab_20261107/README.md) · [400M](../../evaluation/swa_400m_ab_20261111T054430Z/README.md) | [160M 冻结设置](../SWA_160M_AB.md) · [400M 冻结设置](../SWA_400M_AB.md) |
| 830M baseline | SWA 完成 40k 和八项评测，gRet 暂停 | [队列与恢复记录](BASELINE_830M_QUEUE.md) · [SWA 设置](SWA_830M_BASELINE.md) |
| 早期两卡实验 | [完整训练验收](../evidence/full-v2-convergence-review.md) · [短程对照](../evidence/paired-v2-convergence-review.md) | [三来源融合设计](DYNAMIC_YOCO_THEORY_AND_EXPERIMENTS.md) · [早期 baseline 选型依据](BASELINE_EVIDENCE.md) |
| 32 卡 YOCO / Transformer | [训练与推理报告](../../evaluation/YOCO_BASELINE_REPORT.md) | [2K 推理测量](../../evaluation/INFERENCE_2K_32GPU.md) |

九组已完成模型的统一计算量、loss 和下游对照见[九组汇总](../../evaluation/model_comparison_20261111/README.md)。
