# 160M A/B 四卡完整训练与测试

> 历史四卡执行计划。2026-11-04 已完成并发布的是另一次 8×A800 A/B 运行，见[三组结果与 compute 报告](../evaluation/ab_baseline_20261104/README.md)。下文排队状态为当时快照，不代表当前调度器状态。

2026-11-03 用户要求取消重复 baseline 训练，仅运行以下两种新方法。原作业 793493 于北京时间 10:35 取消，停在 baseline 第 11004 步；原快照、日志、checkpoint 保留。

| 组别 | 记忆构造 | 新增参数 |
| --- | --- | ---: |
| A / learned_kv | 六个全局可学习 logits 经 softmax，直接聚合 H1–H6 的候选 K/V | 6 |
| B / conditional_kv | 对各来源 hidden states 做无参数 RMSNorm，用共享打分向量和六个层偏置生成每 token 配比 | 774 |

两组候选均为共享 KV RMSNorm 和共享 K/V 投影作用于每个 self block 输出的结果，再以同一组权重分别聚合 K/V。K 的位置旋转在融合后执行。在第 7 层前构建一次，第 7–12 层分别生成 Q 并复用融合 K/V。查询主路径从 H6 开始。

A/B 共用主体初始参数，seed=42，初始配比均为六层等权。A 不使用旧 static 路径：旧版先混合 hidden states 再做 KV norm，与本次候选 K/V 聚合通常不等价。

单个 allocation：1 节点、4×H100、16 CPU、192G、4 天，不指定节点。两组均先执行四卡 synthetic smoke 和 checkpoint 恢复检查；全部通过后，A、B 依次各从头训练 40000 步。序列长 2048，microbatch=4、累积=8，每步 262144 tokens，验证 2048 序列。保持原 40000 步 LR schedule。

训练后，两组分别在一张卡执行 ARC Easy、ARC Challenge、BoolQ、HellaSwag、OpenBookQA、PIQA、WinoGrande、SciQ 的完整评测。数据锁使用快照内 evaluation/plan_ab_data_locks，离线缓存复用 eval-20261102。评测路径按 A/B 命名，不写入旧 baseline/dynamic 结果。

独立 HPC3 快照：`experiments/plan-ab-full-4gpu-20261103-v2`。

训练目录：`runs/yoco-160m-plan-ab-4gpu-20261103-v2-{A,B}`。

最终标记：快照内 `runs/eval-plan-ab/PIPELINE_SUCCESS`；只有两组训练和 16 项完整下游评测成功并通过汇总检查后写入。训练完成不等于全流程完成。

当前作业 **794750**，2026-11-03 10:47 提交；实时检查为排队（Priority）。HPC3 容器五项 A/B 测试与八项数据离线加载均通过，完整训练和下游评测未完成。

当前作业记录：本地 `runs/setup/plan-ab-active-job.json`，远端快照 `submitted.json`；如只有 submission-intent，先核对 Slurm 后恢复，禁止盲目重提。

旧两卡 baseline：788651，40000 步，验证 loss 3.288449，原权重不覆盖。本次主要配对比较为 A/B；baseline 为历史参考，跨卡数 wall time 不直接作公平速度比较。

验证：本地五项 A/B 合约测试通过，涵盖显式 K/V 公式和参考梯度、主体/初始输出一致、退化至 H6 baseline、因果/共享缓存、激活重计算和优化器恢复。HPC3 容器复验、离线数据检查及 Slurm 实时状态以提交记录和日志为准。

首次部署 v1 在提交前发现 PIQA 数据锁与既有离线评测不一致，未提交 GPU 作业。v2 独立保存与 eval-20261102 一致的八项数据锁，其中 PIQA revision 为 b40a463b4ac9c77c6e74e23c5d873f5c2139d8fa；原 evaluation/data_locks 的历史锁保留不改。


2026-11-03 用户最新明确要求代码和文档立即推送 GitHub main，不等待排队、训练或评测。原等待全流程后发布的后台任务已停止，旧条件已被本次指令取代。作业 794750 保持原调度；README 与文档明确实验结果尚未产生。tools/watch_plan_ab.py 仅保留为可选工具，目前未运行，勿自动重启发布任务。
