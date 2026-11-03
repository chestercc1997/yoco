# 160M conditional K/V pilot

**2026-11-03 更新：**作业 793493 已按用户要求取消，不再重训 baseline。新任务仅训练 A（全局可学习六层 K/V 聚合）和 B（conditional 六层 K/V 聚合），见 [PLAN_AB_4GPU.md](PLAN_AB_4GPU.md)。以下 793493 内容为历史记录。

**2026-11-02 更新：**用户要求完整训练和下游测试，原 1000 步作业 793353 已取消，替换为 HPC3 作业 **793493**。单节点四卡，先完整训练 baseline 至 40000 步，再完整训练 conditional K/V 至 40000 步；每组均先做四卡 smoke 和 checkpoint 恢复检查。随后并行进行两组的 ARC Easy、ARC Challenge、BoolQ、HellaSwag、OpenBookQA、PIQA、WinoGrande、SciQ 完整下游评测并汇总。作业时限 4 天，当前排队；训练与下游评测结果尚未产生。独立快照为 `experiments/plan-conditional-kv-full-4gpu-v1`，运行目录为 `runs/yoco-160m-plan-full-4gpu-v1-{baseline,conditional}`。八项离线评测数据已在 HPC3 容器加载验证；两组全局 batch 均为 262144 tokens，验证样本均为 2048。

以下为已取消的短程作业 793353 的历史记录：baseline 与 conditional K/V 各 2 张 GPU，原计划作为单个 4 卡 HPC3 作业并行运行。

动态方法逐 token 对第 1–6 个 self-decoder 输出分别计算无仿射 RMS 归一化后的共享打分向量与层偏置，温度为 1；同一组 softmax 权重分别聚合每层经 baseline 共享 K/V norm 和投影得到的候选 K、V。之后旋转 K，供六层 cross-decoder 共用。baseline 和动态版同 seed、相同语料及训练参数，新增 774 个可训练参数；其他共有参数初始化相同。

两组各从头试跑 1000 步，仍使用 40000 步学习率计划；每组先运行两卡合成 smoke 和 checkpoint 恢复检查。独立快照位于 HPC3 `/data/share/jzhou945/cchen099/yoco/experiments/plan-conditional-kv-160m-v1`，结果位于 `runs/yoco-160m-plan-conditional-kv-v1-{baseline,dynamic}`。训练完成与结果优劣尚待作业结束后验证，不能由提交成功推断。

验证：本地 CPU 的新方法公式、梯度、原参数初始化、因果和增量缓存、激活重计算、优化器恢复测试通过；原有融合测试通过。HPC3 容器运行新方法测试通过，数据 28 个文件 SHA256 通过，配置及样本索引与原训练配方一致。作业内仍会执行 GPU smoke 和恢复检查。
