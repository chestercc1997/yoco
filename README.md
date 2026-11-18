# YOCO baseline 与 K/V 聚合实验

本仓库包含 160M、400M、830M 三个论文规模档位的 gated retention（gRet）和滑动窗口注意力（SWA）模型，以及已完成的 baseline、方法 A 和方法 B 实验。**从头训练 baseline 请使用 `configs/appendix_d/` 中的六份配置。** 它们使用论文附录 D 的 FFN 宽度、独立的输入输出权重和 YOCO 的样本边界，继续采用本地固定版本 FineWeb-Edu 和 FP16 混合精度配方。[配置与 FP16 故障分析](docs/YOCO_BASELINE_AUDIT.md)

| 规模档位 | 层数 | 模型宽度 | FFN 宽度 | 从头训练配置 |
| --- | ---: | ---: | ---: | --- |
| 160M | 12 | 768 | 2304 | [gRet](configs/appendix_d/160m_gret.json) · [SWA](configs/appendix_d/160m_swa.json) |
| 400M | 24 | 1024 | 3072 | [gRet](configs/appendix_d/400m_gret.json) · [SWA](configs/appendix_d/400m_swa.json) |
| 830M | 24 | 1536 | 4608 | [gRet](configs/appendix_d/830m_gret.json) · [SWA](configs/appendix_d/830m_swa.json) |

## 从原始数据到训练

在仓库根目录使用项目的 Python 环境。A100 环境定义见 [environment.a100.yml](environment.a100.yml)。以下示例假定已下载锁定版本的原始 parquet。六份配置共用一份样本清单，只需准备一次。准备脚本支持通过省略 `--offline` 下载原始分片。

```bash
python tools/prepare_data.py --offline --config configs/appendix_d/400m_gret.json
python tools/verify_data.py --config configs/appendix_d/*.json
```

从头训练使用新的运行目录。以下示例需要单机八张 GPU。恢复历史 checkpoint 时，使用该 run 冻结的 release、配置和 manifest。历史 400M 的 FFN 宽度为 2816，旧权重与本表的模型形状不同。

```bash
torchrun --standalone --nproc-per-node=8 -m runtime.train \
  --config configs/appendix_d/400m_gret.json --run-dir runs/appendix-d-400m-gret
```

## 历史结果

下列结果来自各实验当时冻结的配置。新的六份配置尚无此处可引用的完整训练结果。

| 实验 | 状态与记录 |
| --- | --- |
| gRet 160M 和 400M，SWA 160M | 九组均完成 40,000 步与八项下游评测。[九组汇总](evaluation/model_comparison_20261111/README.md) · [gRet 160M](evaluation/ab_baseline_20261104/README.md) · [gRet 400M](evaluation/gret_400m_3way_20261108T194918Z/README.md) · [SWA 160M](evaluation/swa_160m_ab_20261107/README.md) |
| SWA 400M baseline / A / B | 三组均完成 40,000 步与八项下游评测。[曲线与原始指标](evaluation/swa_400m_ab_20261111T054430Z/README.md) |
| 830M baseline | SWA 已完成。gRet 在第 9793 步出现非有限前向后暂停，状态写入修复已通过原失败批次回放，可恢复的完整 checkpoint 在第 5004 步。[数值审计](docs/YOCO_BASELINE_AUDIT.md) · [队列记录](docs/BASELINE_830M_QUEUE.md) |
| 早期 YOCO / Transformer 对比 | 32 张 V100 上的训练、下游与推理测量。[实验报告](evaluation/YOCO_BASELINE_REPORT.md) · [2K 推理](evaluation/INFERENCE_2K_32GPU.md) · [长上下文推理](evaluation/LONG_CONTEXT_INFERENCE.md) |

## 仓库导航

| 目录 | 内容 |
| --- | --- |
| [runtime](runtime) | 模型、训练器、分词器和 checkpoint 恢复 |
| [configs](configs) | 新 baseline 以及历史实验配置 |
| [tools](tools/README.md) · [scripts](scripts/README.md) | 数据准备、验证、故障回放和各实验入口 |
| [evaluation](evaluation) · [benchmarks](benchmarks) | 下游评测、推理基准、原始指标和绘图 |
| [tests](tests/README.md) | 模型、数据、数值稳定性和恢复检查 |
| [docs](docs/README.md) · [provenance](provenance) | 实验报告索引、上游版本和数据来源 |
