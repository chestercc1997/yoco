# YOCO 与 Transformer 长上下文推理效率对比

本实验沿用官方 YOCO 生成流程：prefill 处理除最后一个 prompt token 外的上下文；YOCO 在 prefill 阶段跳过 cross-decoder；随后从最后一个 prompt token 开始逐 token 解码。实验使用已经完成配对训练的 YOCO 166.7M 与 Transformer 162.0M checkpoint。

## 实验设置

- 硬件：单张 Tesla V100-PCIE-32GB
- 精度：FP16
- batch size：1
- 上下文长度：2K、4K、8K、16K、32K
- 生成长度：64 tokens
- 统计：每个 shape 完整预热后运行 5 次，报告中位数
- Transformer prefill 运行全部 12 层；YOCO prefill 按官方生成路径只运行 self-decoder，并准备共享 KV cache
- GPU memory 使用 PyTorch `max_memory_allocated` 实测；incremental cache 由运行时 state 中 tensor storage 实测

模型只在 2K context 上训练。4K–32K 结果仅用于架构效率 profiling，不代表长上下文语言建模或检索质量。

## 实测结果

| Context | YOCO prefill 加速 | YOCO decode 加速 | YOCO 端到端加速 | Cache 缩小 | 峰值显存降低倍数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 12.15× | 1.02× | 1.25× | 8.80× | 1.09× |
| 4K | 9.06× | 1.07× | 1.33× | 10.13× | 1.18× |
| 8K | 7.00× | 1.29× | 1.60× | 10.98× | 1.32× |
| 16K | 6.78× | 1.98× | 2.35× | 11.46× | 1.47× |
| 32K | 9.74× | 2.08× | 2.71× | 11.72× | 1.60× |

32K 时，Transformer / YOCO 的 prefill latency 分别为 1.566 / 0.161 秒，decode throughput 为 17.12 / 35.69 tokens/s。端到端 throughput（prompt 加 64 个生成 token）为 6,197 / 16,810 tokens/s。增量 cache 为 1,154 / 98.5 MiB，总峰值 allocated memory 为 2.01 / 1.26 GiB。

小模型下，权重和中间 activation 占据固定显存，因此 11.72× 的 cache 差异在总峰值上被稀释为 1.60×。随着模型层数、batch 和 context 增大，KV cache 所占比例会上升，YOCO 的总显存优势才会更接近论文中的长上下文结果。

## 与论文实验的关系

论文的推理 profiling 使用 3B 模型、H100-80GB、32K–1M context、最后生成 1,024 tokens，并使用 Triton gated-retention kernel；Transformer baseline 使用 GQA、Flash-Decoding 和 kernel fusion。本实验复现的是相同的比较维度和官方计算流程，但缩放到现有 160M checkpoint、V100 backend、2K–32K context 和 64-token generation。因此本实验可以验证趋势，不能作为论文绝对数值的严格复现。

原始测量保存在 `figures/long_context_yoco_v100.json` 与 `figures/long_context_transformer_v100.json`，逐长度对比保存在 `figures/long_context_inference_comparison.csv`。运行脚本为 `tools/benchmark_long_context_inference.py`，绘图脚本为 `tools/plot_long_context_inference.py`。
