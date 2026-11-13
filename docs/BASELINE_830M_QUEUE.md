# 830M baseline：SWA → gated retention

用户授权分别训练830M的SWA和gated retention baseline。两者各用八张A100，依次执行完整40k步、八项零样本下游和自动报告导出；本轮不包含A/B训练。

2026-11-14启动状态：SWA已通过全部八卡预检和评测smoke，进入正式40k训练；gRet已通过CPU预检并冻结，持久队列等待SWA全流程完成。两组均未完成训练；[启动与队列证据](evidence/830m-baseline-launch-20261114.json)。

| 档位 | 实际参数量 | 层数 | dim / FFN | self heads / cross heads | 结构 |
| --- | ---: | ---: | --- | --- | --- |
| SWA830M | 838315008 | 24 | 1536 / 4608 | 12 / 12 | window1024 |
| gRet830M | 866737152 | 24 | 1536 / 4608 | 6 / 12 | retention head dim256 |

维度来自[官方Appendix D/Table7](https://arxiv.org/html/2405.05254v2#A4)，两种结构的实际参数量分别报告。新模型各自使用seed42随机初始化，原始未训练权重保留供以后获授权的同结构A/B复用，不加载已训练160M/400M模型。

## 配方和预检

两组training/data完全相同，并沿用已批准的400M配方：FineWeb-Edu固定revision/manifest/tokenizer、FP16 mixed/FP32参数、原AMP策略、AdamW betas(0.9,0.98)、LR1.5e-4、warmup375、线性衰减、weight decay0.05、clip1；8卡microbatch4/accum4/sequence2048，262144 tokens/步，40000步。每250步完整验证，每5000步保存全部完整checkpoint。

execution分别逐项继承同结构400M配置：两者均native、fused AdamW、TF32开启、activation checkpointing关闭。gRet还保留原配置的`emulate_precision_casts=true`及`compile_scope=layers`字段；compile仍为false。这些字段不因换规模而删除或调整。retention与AMP修复源码SHA已与完成的gRet400M冻结版本核对一致。

CPU实际模型有限前后向、因果性、同seed初始权重按位一致、增量cache检查已通过；SWA还验证1024窗口边界。各模型使用GPU之前须完成真实数据八卡0→2→4恢复、连续/独立重复、前向前模型/optimizer/scaler/八卡RNG按位恢复及评测smoke。保持原容差，不为830M放宽。

CPU预检：[SWA证据](evidence/swa-830m-baseline-preflight-20261114.json)、[gRet证据](evidence/gret-830m-baseline-preflight-20261114.json)。冻结契约：[SWA](evidence/swa-830m-baseline-guard-contract-20261114.json)、[gRet](evidence/gret-830m-baseline-guard-contract-20261114.json)。

SWA的[真实八卡恢复检查](evidence/swa-830m-baseline-resume-check-20261114.json)已通过：恢复/连续模型最大绝对差1.8068e-6、relative L2为2.0857e-7，最大loss差1.9074e-5，均小于未改动的原容差；恢复前完整核心状态按位一致。gRet的真实GPU检查会在队列轮到它时执行，不能将CPU预检当作GPU检查已通过。

## 持久接续与状态

- SWA入口与配置：[说明](SWA_830M_BASELINE.md)、[配置](../configs/yoco_830m_swa_baseline_8gpu.json)、[controller](../tools/run_swa_830m_baseline.py)、[guard](../tools/watch_swa_830m_baseline.py)。ACTIVE为`runs/setup/swa-830m-baseline-active.json`。
- gRet入口与配置：[配置](../configs/yoco_830m_gret_baseline_8gpu.json)、[controller](../tools/run_gret_830m_baseline.py)、[guard](../tools/watch_gret_830m_baseline.py)。准备记录为`runs/setup/gret-830m-baseline-preparation.json`；ACTIVE仅实际开始时写入`runs/setup/gret-830m-baseline-active.json`。
- [接续器](../tools/run_830m_baseline_queue.py)读取`runs/setup/830m-baseline-training-queue.json`，等待SWA根/arm的PIPELINE_SUCCESS、40k和八项完整评测/报告证明，并确认8张GPU空闲，才启动gRet监督器。持久进程不依赖聊天窗口继续打开。
- 两个guard各自保留失败现场，核验最近健康完整checkpoint后恢复同一步序；最多两次回退。正常AMP溢出沿原策略重试同一批次，不能改训练参数。超出有界恢复或正式训练外失败时留诊断停止，不无限重试。
- 排队不等于已经训练，后台启动不等于完整结果已完成；最终曲线与下游结果仍需审阅发布。

[SWA契约测试](../tests/test_swa_830m_baseline_contract.py)、[gRet契约测试](../tests/test_gret_830m_baseline_contract.py)、[队列门控测试](../tests/test_830m_baseline_queue.py)覆盖配置/源码防改、完整状态回退及避免未完成前序/忙GPU/重复监督器误启动。
