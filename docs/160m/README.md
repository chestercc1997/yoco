# 160M 新配方与八组训练

比较 SWA 和 gated retention（gRet）两种 self-decoder。每种架构从头训练 baseline、A、B、C，使用相同的数据顺序和公共主干初始化。A/B 直接学习可正可负的跨层系数，C 为 K 和 V 分别学习深度注意力。[最新结果、效率和训练曲线](results.md) · [四组设计与公式](fusion.md) · [SWA 实现与实测](swa.md)

## 配置与代码版本

本轮队列从提交 `0ab578b` 冻结训练代码。下表链接到仓库中的八份配置。两种架构共用训练与数据设置，同一架构四组只改实验名和 K/V 融合设置。[本轮冻结版本](https://github.com/chestercc1997/yoco/tree/0ab578bda029c27ea2173af52414c3e2fc88c07b) · [数据准备与启动命令](../../scripts/README.md#160m-八组训练)

| self-decoder | baseline | A · raw 全局系数 | B · raw 逐 token 系数 | C · K/V 独立深度注意力 |
| --- | --- | --- | --- | --- |
| SWA · 12 头，窗口 1024 | [配置](../../configs/appendix_d/160m_swa.json) | [配置](../../configs/160m_raw_kv/swa_A.json) | [配置](../../configs/160m_raw_kv/swa_B.json) | [配置](../../configs/160m_raw_kv/swa_C.json) |
| gRet · 3 个 retention 头 | [配置](../../configs/appendix_d/160m_gret.json) | [配置](../../configs/160m_raw_kv/gret_A.json) | [配置](../../configs/160m_raw_kv/gret_B.json) | [配置](../../configs/160m_raw_kv/gret_C.json) |

## 本轮训练配方

| 项目 | 设置 |
| --- | --- |
| 模型 | 12 层，宽 768，FFN 2304。前六层 self-decoder，后六层共享全局 K/V。输入 embedding 与输出投影独立 |
| 数据 | FineWeb-Edu 的 `sample-10BT` 子集，revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`，按 YOCO 边界生成 2049-token 记录，错位得到长度 2048 的输入和标签。校验通过 14 个分片，共 4,582,341 条训练样本、4,557 条验证样本 |
| 预算 | seed 42，8×A100，每卡 4 条，累积 4 次，每步 262,144 tokens，40,000 步共 10,485,760,000 tokens |
| 优化器与学习率 | AdamW，β₁ = 0.9、β₂ = 0.98、ε = 1e-8，weight decay 0.05，梯度范数裁剪阈值 1.0。前 375 次更新线性升至 1.5e-4，随后线性衰减，第 40,000 步降至 0 |
| 数值与执行 | FP16 混合精度、FP32 参数、动态 loss scale（初始 128）、fused AdamW。八组均关闭 activation checkpointing。沿用已修复的 gRet 累积与状态写入。SWA 原生执行，gRet 沿用 160M 配置的模型与交叉熵编译 |
| 验证与恢复 | 每 250 步验证一次，每卡 64 个 microbatch，八卡共 2048 条样本，每 5000 步保存完整 checkpoint。异常退出停队、保留现场，恢复时核对完整状态和数据步序 |

此前实验使用绑定权重与连续 token 流。本轮所有组使用独立权重和新的切块格式。共用层的初始化在同架构四组间逐字节一致。[baseline 配置](../../configs/appendix_d) · [方法配置](../../configs/160m_raw_kv) · [配方与 FP16 修复依据](../analysis/YOCO_BASELINE_AUDIT.md)

## 训练顺序与日志

单机八卡串行，先 SWA baseline，再 gRet baseline，随后按 A、B、C 各跑 SWA → gRet。每组完成 40,000 步并核验最终 checkpoint 后进入下一组。[队列脚本](../../scripts/run_160m_raw_kv.sh)

本机运行目录为 `runs/appendix-d-160m-raw-v1/`。队列顺序见 `queue.log`，各组的进度、验证 loss 和异常事件见 `<组名>/rank0.jsonl`。公开汇总给出实测用时、验证曲线、末段训练效率和 AMP 事件。[运行结果](results.md)
