# 160M 新配方与八组训练

比较 SWA 和 gated retention（gRet）两种 self-decoder。每种架构从头训练 baseline、A、B、C，使用相同的数据顺序和公共主干初始化。A/B 直接学习可正可负的跨层系数，C 为 K 和 V 分别学习深度注意力。[四组设计与公式](fusion.md) · [SWA 实现与实测](swa.md)

## 本轮训练配方

| 项目 | 设置 |
| --- | --- |
| 模型 | 12 层，宽 768，FFN 2304。前六层 self-decoder，后六层共享全局 K/V。输入 embedding 与输出投影独立 |
| 数据 | 固定版本 FineWeb-Edu，按 YOCO 边界生成 2049-token 记录，错位得到长度 2048 的输入和标签。校验通过 14 个分片，共 4,582,341 条训练样本、4,557 条验证样本 |
| 预算 | seed 42，8×A100，每卡 4 条，累积 4 次，每步 262,144 tokens，40,000 步共 10,485,760,000 tokens |
| 数值与执行 | FP16 混合精度、FP32 参数、动态 loss scale、fused AdamW。沿用已修复的 gRet 累积与状态写入。SWA 原生执行，gRet 沿用 160M 配置的模型与交叉熵编译 |
| 验证与恢复 | 每 250 步验证，每 5000 步保存完整 checkpoint。异常退出停队、保留现场，恢复时核对完整状态和数据步序 |

此前实验使用绑定权重与连续 token 流。本轮所有组使用独立权重和新的切块格式。共用层的初始化在同架构四组间逐字节一致。[baseline 配置](../../configs/appendix_d) · [方法配置](../../configs/160m_raw_kv) · [配方与 FP16 修复依据](../analysis/YOCO_BASELINE_AUDIT.md)

## 顺序与单组耗时

单机八卡串行，先 SWA baseline，再 gRet baseline，随后按 A、B、C 各跑 SWA → gRet。每组完成 40,000 步并核验最终 checkpoint 后进入下一组。[队列脚本](../../scripts/run_160m_raw_kv.sh)

**SWA baseline 预计约 6 小时 36 分钟。** 2026-11-18 北京时间 17:03 启动，19:37 已完成 15,656 步。最近 5000 步按日志时间戳计为 0.594 秒/步，含验证和 checkpoint 写盘。按此速度，当时剩余约 4 小时，预计 23:40 左右完成。估算范围是单组训练；全量数据准备单独计时。其他七组的本轮耗时要以各自实测为准。

本机运行目录为 `runs/appendix-d-160m-raw-v1/`。队列顺序见 `queue.log`，各组的真实进度、验证 loss 和异常事件见 `<组名>/rank0.jsonl`。完成后按同 token 比较验证曲线、末段 train loss、梯度与 AMP 事件，再从 checkpoint 检查各层系数及其符号。
