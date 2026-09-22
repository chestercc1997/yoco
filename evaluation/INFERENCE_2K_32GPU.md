# YOCO vs Transformer：2K 下游质量与 1–32 卡推理实测

## 结论

在这对约 160M、训练长度 2K 的 checkpoint 上，YOCO 与 Transformer 的下游质量持平：8 个任务的主指标宏平均分别为 **46.71%** 和 **46.63%**。YOCO 的确定优势是 prefill 和缓存；2K+64 固定形状、32 卡时，prefill 吞吐为 Transformer 的 **3.53×**，prefill 后缓存小 **8.80×**，生成 64 token 后的总状态缓存小 **6.95×**。

YOCO 并非在所有 workload 上都更快。固定 2K prompt + 64-token generation 中，这个未做 kernel fusion 的 V100 实现 decode 吞吐只有 Transformer 的 **0.90×**，因此端到端总吞吐为 **0.97×**。在更接近线上请求的动态 shape trace 中，YOCO 通过减少不同长度请求的 prefill 成本，在 32 卡端到端达到 **1.18×**，单卡达到 **1.40×**。

## 测试口径

- 模型：YOCO 166.7M 与 Transformer 162.0M，均为 step 40,000，FP16 推理。
- 硬件：4 节点 × 8 张 V100 32GB。32 卡采用完整模型副本的数据并行 serving；每卡处理独立请求。因此 32 卡数据表示总吞吐/服务容量，不表示单条请求 latency 缩短 32 倍。
- 固定形状：prompt=2048，greedy generation=64，batch/GPU=1；精确 shape 预热后重复 5 次，取中位数。跨 rank latency 取最慢 rank。
- 动态形状：固定 seed `20260922`，顺序为 `128+16, 1024+32, 256+64, 1792+16, 512+32, 2048+64, 768+16, 1536+64`。保留首次 shape pass，并对后续 3 passes 逐 shape 取中位数。顺序故意非单调，以覆盖 allocator、短/长 prompt 与短/长 generation 的组合。
- prefill 遵循官方 YOCO generation flow：YOCO 在 prompt 阶段跳过 cross-decoder，随后逐 token recurrent decode。Transformer 使用相同 checkpoint、dtype、batch 和 token 数。

## 官方下游测试

完整运行 `lm-evaluation-harness 0.4.5` 的 8 个官方任务，zero-shot、无 limit、seed=42。数据 revision、task YAML、dataset cache、adapter 和 checkpoint 均保存 SHA256 provenance。质量评测使用自然 prompt shape，模型最大上下文为 2K，没有为了计时而将样本人工 padding 到 2K。

主指标选取：ARC、HellaSwag、PIQA、OpenBookQA、SciQ 使用 `acc_norm`；BoolQ 和 WinoGrande 使用 `acc`。

| 任务 | Transformer | YOCO | YOCO 差值 |
| --- | ---: | ---: | ---: |
| ARC Easy | 47.60% | 46.09% | -1.52 pp |
| ARC Challenge | 25.43% | 25.51% | +0.09 pp |
| HellaSwag | 31.00% | 31.33% | +0.33 pp |
| PIQA | 62.51% | 62.46% | -0.05 pp |
| WinoGrande | 49.88% | 51.78% | +1.89 pp |
| BoolQ | 60.12% | 61.53% | +1.41 pp |
| OpenBookQA | 29.40% | 30.80% | +1.40 pp |
| SciQ | 67.10% | 64.20% | -2.90 pp |
| **宏平均** | **46.63%** | **46.71%** | **+0.08 pp** |

## 固定 2K+64 runtime

| GPU | 模型 | runtime | 聚合端到端吞吐 | prefill 吞吐 | decode 吞吐 | KV/状态缓存/请求 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Transformer | 0.915 s | 2,307 tok/s | 22,351 tok/s | 77.8 tok/s | 74.25 MiB |
| 1 | YOCO | 0.954 s | 2,213 tok/s | 88,556 tok/s | 68.8 tok/s | 10.69 MiB |
| 8 | Transformer | 1.082 s | 15,609 tok/s | 189,991 tok/s | 520.3 tok/s | 74.25 MiB |
| 8 | YOCO | 1.138 s | 14,836 tok/s | 508,449 tok/s | 461.5 tok/s | 10.69 MiB |
| 32 | Transformer | 1.174 s | 57,548 tok/s | 532,463 tok/s | 1,948.7 tok/s | 74.25 MiB |
| 32 | YOCO | 1.213 s | 55,698 tok/s | 1,879,068 tok/s | 1,748.6 tok/s | 10.69 MiB |

从 1 卡到 32 卡，端到端吞吐扩展效率为 Transformer **77.9%**、YOCO **78.7%**。由于不同节点同时完成一个 serving batch，32 卡 runtime 使用最慢 rank；它比单卡略高是正常的 straggler 效应。

固定 2K 的模型加状态峰值显存为 Transformer **430.6 MiB/GPU**、YOCO **395.7 MiB/GPU**，YOCO 低 **8.1%**。这个小模型的权重仍占主要部分，所以总显存差距远小于 KV cache 差距；模型、batch、上下文增大后，缓存优势会更重要。

## 动态 shape 鲁棒性

| GPU | 模型 | trace runtime | 聚合端到端吞吐 | prefill 吞吐 | decode 吞吐 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Transformer | 5.835 s | 1,433 tok/s | 9,291 tok/s | 61.2 tok/s |
| 1 | YOCO | 4.173 s | 2,003 tok/s | 59,486 tok/s | 75.3 tok/s |
| 32 | Transformer | 6.568 s | 40,730 tok/s | 233,861 tok/s | 1,779.8 tok/s |
| 32 | YOCO | 5.585 s | 47,898 tok/s | 1,242,996 tok/s | 1,808.9 tok/s |

32 卡动态 trace 中，YOCO 的 prefill、decode、端到端分别为 Transformer 的 **5.32×、1.02×、1.18×**。8 个 shape 的端到端加速全部大于 1，范围 **1.08×–1.42×**。首次 shape pass 的端到端吞吐也为 Transformer 的 **1.25×**，说明结果不是只靠重复同一 shape 的预热获得。

## 解释与限制

固定 2K+64 中 prefill 只占总 runtime 的一小部分，而 64 次逐 token decode 占主导。因此 YOCO 即使将 prefill 大幅缩短，也会被当前实现略慢的 decode 抵消。动态 trace 含短 generation 和不同 prompt 长度，prefill 权重增大，所以 YOCO 的端到端优势显现。

这些数字适用于当前 160M checkpoint、PyTorch eager backend 和 V100。YOCO 论文的主要系统结果使用更大模型、H100、32K–1M 上下文以及优化 kernel，不能把论文中的绝对加速倍数直接套到这里。这里的比较重点是同硬件、同 checkpoint 训练预算、同输入 trace 下的实测差异。

## 产物

- 图：`evaluation/figures/04_yoco_transformer_2k_32gpu_inference.png` 和 `.pdf`
- 表：`evaluation/figures/inference_2k_32gpu_comparison.csv`
- 汇总：`evaluation/figures/inference_2k_32gpu_summary.json`
- 固定 shape 原始结果：`evaluation/raw/2k_32gpu/{yoco,transformer}_w{1,8,32}_b1.json`
- 动态 shape 原始结果：`evaluation/raw/2k_32gpu/{yoco,transformer}_dynamic_w{1,32}_b1.json`
- 可复现脚本：`tools/benchmark_2k_distributed_inference.py`、`tools/benchmark_dynamic_shape_inference.py`、`tools/launch_32gpu_inference.sh`
