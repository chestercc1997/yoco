# SWA 400M：baseline → 方法A → 方法B连续八卡训练

用户于2026-11-11授权运行三组完整SWA实验，先baseline，后A、B。八张A100 80GB PCIe串行使用，各组完成训练和八项完整下游后自动进入下一组，不需要手动启动下一组。

运行目录：`runs/yoco-swa-400m-ab-8gpu-20261111T054430Z`。实时入口：`runs/setup/swa-400m-active.json`。tmux：`yoco-swa-400m-8gpu`，监督器`tools/watch_swa_400m.py`管理控制器`tools/run_swa_400m.py`。先看状态，禁止重复启动。历史gRet三组和SWA160M三组均保留，不重跑。

当前已通过baseline真实数据八卡恢复/重复与eval smoke，正式训练首100步已核验；[启动快照](evidence/swa-400m-launch-20261111.json)为有时间戳的历史记录，实时进度以ACTIVE和run状态为准。A/B八卡检查和训练将在前组完整结束后依次运行，当前未启动。

## 模型与共同初始化

这是沿用现有400M档宽深配置的新SWA实验：24层、dim1024、FFN2816、self/attention/KV heads均16（每头64维），窗口1024，词表100288、tied embeddings、rope10000、seed42。SWA移除gRet门控结构，实际参数量如下，不能称其与gRet精确同参数量。

| 组别 | 实际参数量 | 相对baseline新增 | 共享K/V来源 |
| --- | ---: | ---: | --- |
| baseline | 387958784 | 0 | self-decoder最后一层 |
| A / learned_kv | 387958796 | 12 | 前12层候选K/V，全局可学习softmax权重 |
| B / conditional_kv | 387959820 | 1036 | 前12层候选K/V，逐token条件权重 |

A/B的temperature固定1，控制参数零初始化、初始12来源等权。三组来自同一份新生成的SWA随机基座，**不加载训练后的gRet或其他模型权重**。初始化保存在`runs/setup/swa-400m-initializations/{baseline,A,B}.pt`，各自SHA随release冻结。

[CPU预检](evidence/swa-400m-preflight-20261111.json)已确认198个公共state tensors逐位相同，A/B初始输出逐位相同，末来源退化与baseline输出差异为0，因果性前缀差异为0，融合控制梯度非零且全部梯度有限；1024窗口边界和逐token/分块缓存等价检查通过。

## 固定训练配方

三个配置：[baseline](../configs/yoco_400m_swa_baseline_8gpu.json)、[A](../configs/yoco_400m_swa_A_8gpu.json)、[B](../configs/yoco_400m_swa_B_8gpu.json)。三组model/training/data/seed/execution相同，只有已声明的融合模块不同。training/data与既有SWA160M、gRet400M逐项相等，execution沿用已验证的SWA native设置。

| 项目 | 设置 |
| --- | --- |
| 预算 | 每组40000步，10485760000 tokens，三组共120000步 |
| batch | 8卡，microbatch4，累积4，序列2048，全局262144 tokens/步 |
| 优化器 | AdamW，LR0.00015，warmup375，线性衰减到0；betas0.9/0.98、eps1e-8、weight decay0.05、clip1 |
| 精度 | FP16 mixed / FP32参数，原AMP scale128、增长2、退避0.5、增长周期2000 |
| 数据 | 原固定FineWeb-Edu sample-10BT revision、manifest、tokenizer与顺序，seed42 |
| 执行 | fused AdamW开启，activation recomputation关闭，compile关闭，TF32沿用原设置 |
| 验证 | 每250步，64 batches/rank × microbatch4 × 8卡 = 2048序列 |
| 保存 | 每5000步完整模型/optimizer/scaler/八卡RNG，全部保留，共每组8份 |

每组在自己的正式训练前完成真实数据八卡0→2→4恢复、连续0→4与独立重复对照、前向前模型/optimizer/scaler/八卡RNG按位加载核验，以及评测接口smoke。原容差不放宽：模型最大绝对差2e-5、相对L2为2e-6、loss最大差5e-5；恢复前核心状态按位一致。受检前4步计入正式预算。baseline先检查并训练，A/B各自在前一组完整结束后检查并启动。baseline的[实际八卡恢复证据](evidence/swa-400m-baseline-resume-check-20261111.json)已通过：最大loss差1.7166e-5，恢复/连续模型最大绝对差1.6313e-6；八卡各791张量在前向前按位核对。[冻结契约](evidence/swa-400m-guard-contract-20261111.json)锁定三组配置与训练代码。

## 自动监控与有界恢复

监督器每10秒检查八卡非有限前向、验证持续回退、训练停滞和worker退出。正常AMP拒绝尝试与单batch噪声不回滚。冻结契约锁定三组配置、训练/评测/控制代码和数据manifest，PID及启动时间确定所属进程。

遇到受监控故障，保留原日志/快照/失败后缀，仅停止所属进程，校验最近健康完整checkpoint的SHA、identity、模型/optimizer有限性与八卡RNG/scaler，然后按原步数和相同数据顺序恢复。若5000步前故障，可回到已核验的完整第4步checkpoint。每组最多两次自动回退；重复故障生成只读诊断并停下等待兼容原配方的代码修复，不无限重启，也不宣称能自动修复所有未知代码错误。

禁止以修复为由调整LR、精度、batch、数据顺序或任何training setting；不跳过失败batch，不清空optimizer/RNG/scaler。预检或下游等正式训练之外的失败另行记录诊断，不盲目回滚已完成的训练。三组配置锁定和失败归档/全状态恢复测试见[测试](../tests/test_swa_400m_contract.py)。

## 评测与输出

每组40000步后运行八项全量零样本评测：ARC-Easy、ARC-Challenge、BoolQ、HellaSwag、OpenBookQA、PIQA、WinoGrande、SciQ，21465条文档。综合准确率为八项等权宏平均；每组训练和评测完成才自动接续下一组。

三组全部完成后自动运行`tools/export_swa_400m.py`，验证24份完整checkpoint及三组共24项完整下游，生成原始训练/验证CSV、下游汇总和七组PNG/PDF/SVG。报告目录：`evaluation/swa_400m_ab_20261111T054430Z`。6ND采用与之前相同的共同系数6；重复K/V局部增量按12来源、dim1024计算，不能沿用160M的6来源/768宽度常量。

本地`PIPELINE_SUCCESS`表示三组训练、24项完整下游与自动导出成功。最终图表仍需实际审阅、核对并发布GitHub，不能把后台启动或baseline完成称为三组全部完成。单seed差异只作描述；语料为已批准的固定FineWeb复现配方。

## 只读查看

```bash
cat runs/setup/swa-400m-active.json
cat runs/yoco-swa-400m-ab-8gpu-20261111T054430Z/pipeline-status.json
cat runs/yoco-swa-400m-ab-8gpu-20261111T054430Z/guard-state.json
tail -n 2 runs/yoco-swa-400m-ab-8gpu-20261111T054430Z/baseline/rank0.jsonl
```
