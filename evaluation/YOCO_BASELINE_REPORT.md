# YOCO 与 Transformer Baseline 对比报告

## 测试目的

比较约 160M 参数的 YOCO 与 Transformer baseline 在相同训练数据、训练步数和硬件条件下的模型质量与推理效率。

## 测试配置

- 训练上下文长度：2K
- 推理精度：FP16
- 硬件：4 个节点，共 32 张 V100 32GB
- 模型：YOCO 166.7M，Transformer 162.0M
- Checkpoint：均训练到 40,000 step
- 下游评测：`lm-evaluation-harness`，8 个完整 zero-shot 官方任务
- 推理测试：每张卡运行一个完整模型副本，batch/GPU=1

## 模型质量

| 指标 | Transformer | YOCO |
| --- | ---: | ---: |
| 验证 loss | 3.1613 | 3.1637 |
| 8 项下游宏平均 | 46.63% | 46.71% |

两者的验证 loss 和下游准确率基本相同。YOCO 没有因为架构变化出现明显的质量下降。

## 固定 2K 推理

输入为 2048 tokens，随后生成 64 tokens。

| 32 卡结果 | Transformer | YOCO | YOCO 相对表现 |
| --- | ---: | ---: | ---: |
| Prefill 时间 | 0.123 秒 | 0.035 秒 | **3.53× 更快** |
| Decode 时间 | 1.051 秒 | 1.171 秒 | 0.90× |
| 端到端时间 | 1.174 秒 | 1.213 秒 | 0.97× |
| 聚合吞吐 | 57.55K tok/s | 55.70K tok/s | 0.97× |
| KV/状态缓存 | 74.25 MiB/请求 | 10.69 MiB/请求 | **减少 85.6%** |
| 峰值显存 | 430.6 MiB/卡 | 395.7 MiB/卡 | 减少 8.1% |

YOCO 的 prefill 和 KV cache 优势明显。但是在固定 2K 输入后连续生成 64 tokens 时，当前 V100 实现的 YOCO decode 稍慢，抵消了 prefill 节省的时间，所以端到端速度与 Transformer 接近。

## 动态 Shape 推理

动态测试覆盖 8 种非单调组合，prompt 长度为 128–2048，生成长度为 16–64，并同时记录首次运行与预热后的结果。

| 32 卡动态测试 | Transformer | YOCO | YOCO 相对表现 |
| --- | ---: | ---: | ---: |
| Trace 总时间 | 6.568 秒 | 5.585 秒 | **1.18× 更快** |
| 聚合吞吐 | 40.73K tok/s | 47.90K tok/s | **1.18×** |
| Prefill 吞吐 | 233.9K tok/s | 1,243.0K tok/s | **5.32×** |
| Decode 吞吐 | 1,779.8 tok/s | 1,808.9 tok/s | 1.02× |

在动态 workload 中，8 种 shape 的 YOCO 端到端加速均大于 1，加速范围为 **1.08×–1.42×**。这说明其优势不限于单一固定输入形状。

## 结论

YOCO 与 Transformer 的模型质量基本一致。YOCO 最明确的收益是更快的 prefill 和更小的 KV cache；在动态请求、短生成和更长上下文场景中，这些优势更容易转化为端到端吞吐提升。

对于本次固定 2K+64 测试，YOCO 的端到端速度没有提高，主要原因是当前 V100 eager 实现的逐 token decode 稍慢。YOCO 的训练吞吐也没有优于 baseline，因此它更适合被描述为一种具有推理内存和长上下文 prefill 优势的架构，而不是在所有训练和推理场景下都更快。

## 图表与数据

- 总览图：`evaluation/figures/04_yoco_transformer_2k_32gpu_inference.png`
- 数据表：`evaluation/figures/inference_2k_32gpu_comparison.csv`
- 原始汇总：`evaluation/figures/inference_2k_32gpu_summary.json`
