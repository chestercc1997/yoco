# YOCO 实验文档

| 分类 | 文档 |
| --- | --- |
| **本轮 160M 训练** | [训练结果、曲线和下游评测](160m/results.md) · [配方与顺序](160m/README.md) · [baseline、A、B、C 如何构造 K/V](160m/fusion.md) · [SWA 窗口验证与消融](160m/swa.md) |
| **本轮 400M 训练** | [训练结果、raw 与 softmax 对照、下游评测](400m/results.md) · [raw 配方与启动命令](../scripts/README.md#400m-六组训练) |
| 配置与数值稳定性 | [Baseline 配置与复现步骤](YOCO_BASELINE_REPRODUCTION.md) · [论文、微软代码与本地配置对照](analysis/YOCO_OFFICIAL_CONFIGURATION.md) · [历史配方差异与 FP16 溢出](analysis/YOCO_BASELINE_AUDIT.md) |
| 效率分析 | [YOCO 与 VertiKV 的相对训练效率](analysis/VERTIKV_TRAINING_EFFICIENCY.md) · [历史 A100 实测加速](BASELINE_A100_ACCELERATION.md) |
| 历史实验 | [各规模的结果、曲线和恢复记录](history/README.md) |
| 架构教程 | [YOCO 原理](YOCO_EXPLAINED.md) · [从 Transformer 到 YOCO 与 CED](tutorials/CED_YOCO_TUTORIAL.md) · [DeepSeek V4.1](tutorials/DEEPSEEK_V41_TUTORIAL.md) |
| 原始材料 | [实验核验记录](evidence) · [早期 baseline 来源](baseline_sources) |
