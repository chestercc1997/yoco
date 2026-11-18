# baseline、raw A/B 和深度注意力 C

六个来源是同一 token 在 self-decoder 第 1–6 层的输出。每个来源分别经过共享的 KV 归一化和投影，得到候选 Kᵢ、Vᵢ。各组的查询流都从第六层输出开始，融合后的 K/V 构造一次，供后六层复用。

| 组 | 怎样得到权重 | 初始行为 |
| --- | --- | --- |
| baseline | 直接取 K₆、V₆ | 第六层是唯一来源 |
| A | 六个全局可学习参数 aᵢ，直接计算 ΣaᵢKᵢ、ΣaᵢVᵢ。所有 token 共用系数 | 每项 1/6 |
| B | 每个 token、每层计算 aᵢ = bᵢ + ⟨q, RMSNorm(hᵢ)⟩ / √768，用同一组系数加权 K 和 V | q 为零，每个偏置 bᵢ 为 1/6 |
| C | 两个可学习查询 qᴷ、qⱽ 分别与六层的 RMSNorm(hᵢ) 点积、除以 √768，各自沿深度做 softmax，得到独立的 K 和 V 权重 | 两个查询均为零，两组权重均匀 |

**A/B 直接使用 raw 值**，允许负权重，权重和可以自由变化。C 借鉴 Attention Residuals 的可学习查询，在共享 K/V 的入口分别选择键和值的来源。历史 A/B 实验对系数做 softmax，对应结果按旧设计解读。[实现](../../runtime/memory_fusion.py) · [Attention Residuals](https://arxiv.org/abs/2603.15031)

启动前通过 61 项自动化测试。八个全尺寸模型在 A100 上用同一批 4×2048 个位置做 eager FP16 前反向。各组参数梯度有限，新增打分参数均得到非零梯度，A/B/C 的初始 logits 逐元素一致。[路由与窗口测试](../../tests/test_raw_kv.py)
