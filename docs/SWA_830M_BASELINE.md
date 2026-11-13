# YOCO-SWA 830M baseline

用户在SWA400M baseline/A/B完成后授权下一档baseline。官方模型档位为160M、400M、830M、1.4B等；本轮沿用SWA结构，仅启动830M baseline。

## 配置与边界

模型依据[YOCO论文Appendix D/Table7](https://arxiv.org/html/2405.05254v2#A4)：24层、dim1536、12个attention/KV heads（head dim128）、FFN4608（3d）；SWA窗口1024。CPU实际计数为 **838315008** 参数。保留新seed42未训练初始化，后续A/B若获授权可加载共同随机基座；本轮不启动A/B。

除新规模需要的model字段及实验标识，配置继承[SWA400M baseline](../configs/yoco_400m_swa_baseline_8gpu.json)。training/data/seed/tied_embeddings/execution保持相同：

| 项目 | 值 |
| --- | --- |
| GPU | 8×A100 80GB |
| 正式训练 | 40000步，10485760000 tokens |
| Batch | 每卡microbatch4，累积4，sequence2048；262144 tokens/步 |
| 优化器 | AdamW，betas(0.9,0.98)，eps1e-8，weight decay0.05 |
| LR | 1.5e-4，warmup375，线性衰减至0 |
| 精度 | FP16 mixed / FP32参数，原AMP scale128、增长2、退避0.5、周期2000 |
| 执行 | native，compile关闭，fused AdamW开启，TF32开启，activation checkpointing关闭 |
| 验证 | 每250步，64 batches/rank，共2048条序列 |
| 保存 | 每5000步完整模型/optimizer/scaler/八卡RNG，全部保留 |
| 数据 | 原FineWeb-Edu sample-10BT revision、tokenizer、manifest及顺序，seed42 |

这是现有已批准数据配方上的官方830M架构档位实验；并非声称复用了论文原始语料。已有400M配置与旧release均不改写。

## 执行与恢复

入口为[controller](../tools/run_swa_830m_baseline.py)、[guard](../tools/watch_swa_830m_baseline.py)、[CPU预检](../tools/preflight_swa_830m_baseline.py)。先核对 `runs/setup/swa-830m-baseline-active.json` 或准备记录，禁止重复启动。

CPU预检核验重复seed的初始权重按位一致、完整模型有限前向/反向及因果性、窗口边界和增量cache。正式训练前运行真实数据八卡0→2→4、连续0→4、独立重复及前向前完整状态按位恢复；使用既有模型max_abs2e-5、relative_L2 2e-6、loss5e-5阈值，绝不放宽。通过的前4步计入正式40k预算。

guard监控八卡非有限前向、持续验证退化、停滞和进程退出。失败现场归档；最近健康完整checkpoint校验SHA/fingerprint和有限性后，恢复模型、优化器、scaler、八卡RNG及原数据步序。每组最多两次自动回退，重复故障停止并保留诊断。正常AMP拒绝尝试按原策略重试同一批次，不能改学习率、精度、batch等来掩盖异常。冻结配置与训练源码变更会被拒绝。

完成训练后自动运行原八项完整零样本下游，导出原始train/valid曲线、compute proxy、梯度、吞吐与下游图表。完整结果需要另行核验渲染并发布；当前文档不表示训练已经完成。
