# gRet 400M 方法B：八卡完整训练

2026-11-08用户授权的gRet 400M方法B已完成40000步及八项完整下游。随后用户另行授权的同规格baseline也已接续完成，现有baseline/A/B完整对照：[三组完整报告](../evaluation/gret_400m_3way_20261108T194918Z/README.md) · [九组综合图表](../evaluation/model_comparison_20261111/README.md)。

| 组别 | 最终Valid loss | 下游宏平均% | 相对baseline增幅 |
| --- | ---: | ---: | ---: |
| baseline | 3.005298 | 47.7947 | 0.0000% |
| A | 2.918184 | 48.5270 | +1.5322% |
| B | 2.913688 | 48.3513 | +1.1646% |

B全程11次正常AMP拒绝尝试、0次非有限前向、0次checkpoint回退；每5000步八份完整checkpoint保留。baseline有真实有限梯度尖峰及验证回升，原始图表保留，没有改动setting掩盖波动。三组最终完整checkpoint和24项评测的发布核验见[证据](evidence/gret-400m-final-review-20261111.json)。

以下保留B的初始化、冻结设置和恢复流程。[原始启动快照](evidence/gret-400m-B-launch-20261108.json)记录的是启动时228步，不代表当前进度。[八卡恢复证据](evidence/gret-400m-B-resume-check-20261108.json) · [冻结契约](evidence/gret-400m-B-guard-contract-20261108.json)。

运行目录：`runs/yoco-gret-400m-B-8gpu-20261108T073237Z`。监督器tmux：`yoco-gret-400m-B`。实时入口：`runs/setup/400m-B-active.json`；先检查实际状态，禁止重复启动。

## 模型与共同初始化

沿用已完成400M A的gRet主干：dim1024、24层、FFN2816、前12层self heads=4、attention/KV heads=16、词表100288、tied embeddings及seed42。没有设置sliding_window，前12层实际均为GateRetention。

| 参照/方法 | 参数量 | 新训练 |
| --- | ---: | --- |
| 同规格baseline | 400590848 | 后续授权已完成 |
| 已完成A / learned_kv | 400590860 | 保留原结果 |
| B / conditional_kv | 400591884 | 本轮完整训练 |

B从前12层候选K/V构造一次共享memory，由每个来源的当前token状态生成12个条件softmax权重，K/V共用配比。可学习控制为1024维score向量与12个层bias，共1036个参数；均零初始化，初始12来源等权，temperature固定1。

[CPU预检](evidence/gret-400m-B-preflight-20261108.json)验证222个公共state tensors与A原始随机初始化逐项按位相同；B初始输出与A旧冻结实现差异为0，末来源退化与baseline差异为0，未来token扰动对前缀差异为0。score和bias均有非零、有限梯度，全模型梯度有限。

初始化来自 `runs/setup/yoco-400m-A-initialization.pt` 的公共基座，B初始化另存为 `runs/setup/yoco-400m-B-initialization.pt`。这是同一随机基座上的新预训练，没有使用A的已训练40000步权重。

## 训练配方保持不变

[B配置](../configs/yoco_400m_plan_B_8gpu.json)逐项复用已完成A最终健康run的model、training、data、seed、tied_embeddings和execution，仅memory_fusion改为声明的B结构。

- 8×A100 80GB PCIe；microbatch4、accumulation4、sequence2048，每步262144 tokens，40000步共10485760000 tokens。
- 原FineWeb-Edu sample-10BT固定revision、tokenizer及数据顺序；seed42。
- AdamW：LR1.5e-4，warmup375，线性衰减到0，betas0.9/0.98、eps1e-8、weight decay0.05、grad clip1.0。
- FP16 mixed / FP32 parameters；原AMP初始scale128、growth2、backoff0.5、growth interval2000及连续溢出上限保持不变。
- 每250步验证2048条序列；每5000步保存并全部保留完整模型、optimizer、scaler和八卡RNG。
- fused AdamW开启、activation recomputation关闭、compile关闭、TF32沿用A最终健康执行设置。

已知gRet跨块状态溢出修复和AMP growth tracker回退修复与已完成A最终健康release的源文件SHA完全一致。跨块累积维持原FP32 state，到原RMSNorm后恢复FP16接口；全局精度、模型公式与训练超参数未改。旧release不会原位修改。

## 八卡核验与自动恢复

[控制器](../tools/run_gret_400m_B.py)冻结代码、配置、初始化、数据manifest和评测锁。正式训练前，复用已验证的真实数据八卡0→2→4恢复、连续0→4、独立重复对照，并在任何前向之前逐项检查模型/optimizer/scaler/八卡RNG按位加载。采用原初始恢复容差，训练精度和配方不变。受检前四步计入正式40000步预算。

[专用监督器](../tools/watch_gret_400m_B.py)沿用已验证的进程归属、健康checkpoint选择和失败归档逻辑，只管理本次B作业。每10秒检查八卡非有限前向、验证持续退化、进程退出和停滞；正常AMP退避及普通batch波动不触发checkpoint回退。

故障时保留现场，按PID/start-time仅清理所属进程，校验完整checkpoint的SHA、identity、模型/optimizer有限性及八卡RNG/scaler后，重放相同更新和数据。每组最多自动恢复2次，重复故障输出只读诊断，继续定位兼容原setting的实现问题。禁止改LR、精度、batch、数据顺序，禁止跳过batch或清空optimizer/RNG/scaler。[B配置锁定及回退测试](../tests/test_gret_400m_B_contract.py)已通过。

## 下游与报告

训练结束后自动执行ARC-Easy、ARC-Challenge、BoolQ、HellaSwag、OpenBookQA、PIQA、WinoGrande、SciQ八项完整零样本评测，共21465条文档。随后[导出器](../tools/export_gret_400m_B.py)验证B及历史A的checkpoint和逐样本文档/prompt/target hash、样本数与得分，再输出七组PNG/PDF/SVG、原始训练/验证CSV和逐任务汇总。

B-only历史导出保留在本机 `evaluation/gret_400m_B_20261108T073237Z`。当前发布采用包含已完成baseline的[完整三组报告](../evaluation/gret_400m_3way_20261108T194918Z/README.md)，并通过[九组汇总](../evaluation/model_comparison_20261111/README.md)统一绘图口径。

历史A包含早期编译和修复恢复来源，B全程使用已修复的native路径；两者训练超参数一致，但不能宣称全程执行路径相同，也不能将A/B计时差异单独归因于融合方法。400M baseline已完成，比较采用它的实际40000步指标。

只读进度检查：

```bash
cat runs/setup/400m-B-active.json
cat runs/yoco-gret-400m-B-8gpu-20261108T073237Z/pipeline-status.json
cat runs/yoco-gret-400m-B-8gpu-20261108T073237Z/guard-state.json
cat runs/yoco-gret-400m-B-8gpu-20261108T073237Z/B/heartbeat.json
```
