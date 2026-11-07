---
name: yoco-reproduction
description: Maintain YOCO baseline and A/B reproduction experiments, including numerical debugging, acceleration, checkpoint recovery, and reporting, while preserving the user-approved training recipe exactly.
---

# YOCO 复现：训练设置不可变

用户的目的为复现 baseline，再只引入方法 A/B 的已声明结构差异。适用于 `/root/yoco` 及其训练副本。先读仓库 `AGENTS.md` 的最新授权与对应 run 的冻结配置；已有作业先查 ACTIVE、状态、GPU，禁止重复启动。

**修 bug、加速、自动恢复均不能改动任何训练 setting 参数。** 不得通过改变学习率/衰减/warmup、优化器/betas/eps/weight decay、grad clip、精度/AMP 策略、microbatch/累积/全局 batch、数据/tokenizer/revision/顺序、seed、序列长度、训练步数/token 预算、验证频率/规模、checkpoint 策略，来消除 NaN、降低 loss 或宣称加速。不得跳过、替换失败 batch，清空优化器/RNG/scaler，或从仅含模型的诊断快照续训。当前完整 recipe 以对应 run 的 `release/configs/*.json` 和数据 manifest 为准；每 5000 步保存并全部保留。

baseline/A/B 必须沿用同一变体的共同 backbone 初始化和训练配方。仅 A/B 各自声明的融合模块不同。切换 gRet/SWA、扩展模型规模或另定实验配置，须以用户明确提出的新实验为依据，并另建记录，不能作为故障修复。区分本仓库已批准的固定 FineWeb 复现配方与论文原始语料；发现与官方不一致时报告具体差别，禁止悄悄“对齐”正在运行的配置或宣称严格官方复刻。

恢复时校验 checkpoint SHA、fingerprint、模型/optimizer 有限性及八卡 RNG、scaler、步数；保留失败现场与原始日志，回到最近健康完整 checkpoint，按同一步数和相同数据顺序重跑。正常 AMP 溢出按原策略重试；单个 noisy batch 不等于持续发散。自动恢复须有有界重试与重复故障诊断，不能无限回滚。使用仓库 `tools/watch_swa_training.py` 时核验 `guard-contract.json`，它不授予修改配置的权限。

允许修正实现错误，使实际计算遵守原模型公式和既定精度接口；这不允许全局或局部擅自改精度来掩盖问题。涉及中间算术精度、内核或计算路径的修改，必须明确记录、用捕获现场和参考前后向证明必要性，并检查正常范围的 loss/梯度、checkpoint 恢复和配置完全一致。若无法满足原 setting，应停止该修复路线，在持续 Goal 内继续定位可兼容实现；不能自动放宽 setting 或把它冒称“完全不影响精度”。

只采用通过同批 loss/梯度与恢复核验的执行加速。编译、融合 optimizer 或重计算选项须单独记录且三组一致；禁止把改变算术语义、精度或数据预算叫作执行加速。旧 release/checkpoint 保持不可变，修复建立新快照并记录来源与 hash。完成训练、全量下游、收敛审查及图表汇总后才能报告完成；单 seed 差异作描述，不保证最终指标逐位一致。
