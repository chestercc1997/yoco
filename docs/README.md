# YOCO 架构与实验文档

## A/B 结果与历史执行计划

- [SWA 160M baseline / A / B](SWA_160M_AB.md)：三组40000步、24项完整下游均已完成；含设计公式、冻结设置、七张对比图、原始曲线和验收证据。
- [400M 方法 A 八卡训练](PLAN_A_400M_8GPU.md)：已完成 40000 步及八项下游，验证 loss 2.918184，宏平均 48.5270%；保留故障修复与恢复证据。
- [A100 八卡训练加速与复用](BASELINE_A100_ACCELERATION.md)：执行设置、启动与续训命令、速度测量和原始 loss 核查。

- [A/B 训练与测试说明](PLAN_AB_4GPU.md)：A 是全局可学习六来源 K/V 权重；B 是按 hidden states 生成条件权重。
- [公式与范围](plan.md)：两种方法均融合前六层候选 K/V，供后六层共享。
- [一页 A/B 设计说明](plan_feishu_concise.pdf)：共同流程、全局权重与逐 token 权重的核心公式；[生成脚本](../tools/render_plan_feishu_concise.py)。
- **2026-11-05：baseline 已更新为本次加速 FP16 运行，三组数据、七张对比图和八项下游已同步。** [三组完整结果、loss/6ND 曲线与脱敏数据](../evaluation/ab_baseline_20261104/README.md)。四卡计划为另一执行记录，不混作本次完成证据。

## 历史两卡实验与背景

1. [YOCO 架构讲解](YOCO_EXPLAINED.md)：baseline 的计算路径、共享 K/V、prefill 和 decode。
2. [动态 YOCO 的理论动机 公式与实验](DYNAMIC_YOCO_THEORY_AND_EXPERIMENTS.md)：ABC 的研究目标、动态融合与梯度公式、代码实现和当前结果。
3. [两卡完整训练记录](FULL_2GPU_V2_TRAINING.md)：配对训练设置、checkpoint 恢复与实验追溯。

旧两卡 baseline / token 动态 v2 两组已完成 40,000 步；这不是本轮 A/B。单 seed 的最终验证 loss 为 baseline 3.288449、动态 v2 3.284284；尚未建立训练计算效率优势。

## 实验依据

- [完整训练验收](evidence/full-v2-convergence-review.md) · [原始汇总 JSON](evidence/full-v2-convergence-review.json)
- [两卡短程验收](evidence/paired-v2-convergence-review.md)
- [ABC 聊天理解摘录](evidence/abc-research-context.md)
- [早期 baseline 材料核查](BASELINE_EVIDENCE.md)：历史候选与来源，配套副本在 `baseline_sources/`。

## 图文教程

- [从 Transformer 到 YOCO 与 CED](tutorials/CED_YOCO_TUTORIAL.md) · [离线 HTML](tutorials/CED_YOCO_TUTORIAL.html)
- [DeepSeek V4.1 架构讲解](tutorials/DEEPSEEK_V41_TUTORIAL.md) · [离线 HTML](tutorials/DEEPSEEK_V41_TUTORIAL.html)

`assets/` 保留当前文档使用的图片与生成脚本，`evidence/` 保留实验原始证据。旧计划备份保留在本地 archives/，不作为当前执行入口。
