# YOCO 160M：新 Baseline、A、B 三组对比

更新日期：2026-11-04。Baseline 已替换为本机 **8×A100 80GB PCIe** 完成的原始 YOCO；A/B 保留最新 GitHub 发布的 **8×A800** 运行。三组均完成 **40,000 步、10,485,760,000 个训练 tokens** 和八项完整零样本评测，每组 21,465 道题。

**本次固定 checkpoint 的结果：A 的下游宏平均最高，B 的最终验证 loss 最低。相对新 baseline，A/B 的宏平均分别高 3.28 / 2.24 个百分点。** 本报告中的 baseline、差值、曲线和阈值计算均来自新运行；旧 32×V100 结果仍保留在 Git 历史和原 YOCO/Transformer 报告中。

## 1. 三组基座与设置核查

三组都以同一原始 YOCO-gRet 为基座：12 层、宽度 768、FFN 2304、词表 100288、前六层 self-decoder、后六层 cross-decoder，共享输入 embedding / 输出矩阵。Baseline 从 H6 构造一次 K/V；A/B 仅改变共享 K/V 的六来源融合方式，不更换主干。

| 组别 | K/V 构造 | 总参数 | 比 baseline 新增 |
|---|---|---:|---:|
| Baseline · 8×A100 | 第六层输出生成共享 K/V | 166,708,224 | 0 |
| A · 8×A800 | 六个全局 softmax logits 融合候选 K/V | 166,708,230 | 6 |
| B · 8×A800 | 按 hidden states 生成每 token 权重 | 166,708,998 | 774 |

已直接比较发布的运行配置：`architecture`、`model`、`training`、`seed`、`tied_embeddings` 全部相同。三组均为 seed=42、microbatch=4、accumulation=4、8 ranks、长度 2048、每步 262144 tokens、验证 2048 条序列；AdamW，betas=0.9/0.98，LR=1.5e-4，375 步 warmup、40000 步线性衰减，weight decay=0.05，FP16 mixed / FP32 parameters。

**实际重建完整 160M 模型的核查通过：**同 seed 下，A/B 的全部 114 个公共 state tensors 与 baseline 逐项完全一致；仅多出融合模块参数。A/B 初始六层等权且输出相同；将权重退化为仅选 H6 后，两者都复现 baseline 输出。详见[机器可读核查](comparability_audit.json)。这验证当前实现与发布配置的共同基座；远端 A/B 当时的 step-zero 权重未包含在公开材料中，未宣称直接比对其 payload。

八项评测的 task YAML SHA256、seed、num_fewshot=0、题数和 doc_id 集合均匹配。A/B 原始分词分片哈希和逐题 doc/prompt/target 哈希未包含在此公开导出中，无法在本机重新逐项核对，因此 baseline 差值保持描述性，不给其添加配对显著性结论。A/B 原有的配对统计不变。按本次要求，核查不以软件版本差异作为判定项。

## 2. 最终质量和八项下游

| 指标 | Baseline · 8×A100 | A · 8×A800 | B · 8×A800 |
| --- | --- | --- | --- |
| 最终验证 CE | 3.408906 | 3.160054 | 3.159059 |
| 验证 PPL | 30.232152 | 23.571879 | 23.548433 |
| 最后 100 步平均训练 loss | 3.404486 | 3.151634 | 3.150736 |
| 八项下游宏平均（%） | 43.6631 | 46.9417 | 45.9075 |

![三组最终质量](figures/01_quality_summary.png)

| 任务 | 指标 | 题数 | Baseline % | A % | B % | A−Base pp | B−Base pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arc_easy | acc_norm | 2376 | 42.97 | 45.83 | 45.37 | +2.86 | +2.40 |
| arc_challenge | acc_norm | 1172 | 25.17 | 25.68 | 24.49 | +0.51 | -0.68 |
| hellaswag | acc_norm | 10042 | 29.29 | 31.25 | 31.25 | +1.96 | +1.96 |
| piqa | acc_norm | 1838 | 59.36 | 61.10 | 62.51 | +1.74 | +3.16 |
| winogrande | acc | 1267 | 52.17 | 52.25 | 50.75 | +0.08 | -1.42 |
| boolq | acc | 3270 | 49.85 | 61.62 | 60.49 | +11.77 | +10.64 |
| openbookqa | acc_norm | 500 | 28.40 | 31.80 | 30.00 | +3.40 | +1.60 |
| sciq | acc_norm | 1000 | 62.10 | 66.00 | 62.40 | +3.90 | +0.30 |
| 等权宏平均 | — | 21465 | 43.66 | 46.94 | 45.91 | +3.28 | +2.24 |

BoolQ、WinoGrande 使用 acc，其余六项使用 acc_norm。宏平均对八任务等权，不按题数加权。CSV 保留完整精度和标准误。A/B 原始分数、逐题配对记录、训练曲线均未更改；B−A=-1.0343 pp，既有配对 bootstrap 95% 区间为 [-1.6679, -0.3845] pp，仅反映这两个固定 checkpoint 的样本不确定性。

![八任务与差值](figures/02_downstream_comparison.png)

## 3. 完整训练与验证曲线

现在提供三组各 **40,000 个训练点**，合计 120,000 点；baseline 由 pilot 的第 1–100 步与续训的第 101–40000 步连续拼接，原 optimizer/scaler/RNG 和完整 LR schedule 均保留。正式验证只取每 250 步一次的 160 个点；pilot 的第 50、100 步额外验证不混入正式曲线。

![三组完整训练 loss](training_loss.png)

![三组验证曲线](figures/03_validation_convergence.png)

新 baseline 的最低已观测验证 loss 为 **3.281139（第 20,250 步）**，之后存在回升，最终为 3.408906。因此图中保留全部真实波动，统一使用第 40000 步模型比较，不把最好点替换为最终点；最终分数差异不能单独证明改动的普遍收益。曲线坐标已随新 baseline 调整，避免沿用旧范围隐藏灰色曲线。

## 4. 计算量与同 loss 阈值

定义 `C=6ND`，`PFLOP-days=C/8.64e19`。共享输入/输出矩阵共 77,021,184 个参数：trunk 口径减去一次，output-inclusive 口径计入输出矩阵一次。D 统计实际消费的训练 tokens，含重复采样；训练池为 9,683,617,678 tokens，整个训练约消费其 1.083 倍。

| 指标 | Baseline | A | B |
| --- | --- | --- | --- |
| 总参数 / 计入输出头的 N | 166,708,224 | 166,708,230 | 166,708,998 |
| 主干参数 N | 89,687,040 | 89,687,046 | 89,687,814 |
| 训练 tokens D | 10,485,760,000 | 10,485,760,000 | 10,485,760,000 |
| Trunk 6ND（PFLOP-days） | 0.065308109 | 0.065308114 | 0.065308673 |
| Output-inclusive 6ND | 0.121393224 | 0.121393228 | 0.121393788 |
| 额外五次 K/V 投影代理 | 0.000000000 | 0.004294967 | 0.004294967 |
| Output-inclusive + 重复 K/V | 0.121393224 | 0.125688196 | 0.125688755 |
| 成功 update 活跃 GPU-hours | 64.714 | 60.089 | 60.838 |
| 梯度 overflow 尝试数 | 20 | 11 | 12 |

A/B 对六个来源分别执行共享 K/V 投影。相对 baseline 多五次投影的局部代理为 `6×5×2×768²×D`，在当前 D 下约 0.004294967 PFLOP-days；这仍未包含完整的 retention/attention、归一化、路由、激活重计算和通信成本。**6ND 与局部修正均不是实测设备 FLOPs；A100/A800 的活跃时间不能直接作为架构加速结论。** Baseline 的 wall time 含 pilot 和续训之间的暂停；活跃 GPU-hours 仅累加成功 update 的秒数，不含暂停、溢出重试、验证和存盘。

![不同计算量口径](loss_compute.png)

| 验证 loss 阈值 | Baseline 首次 / 持续步 | A 首次 / 持续步 | B 首次 / 持续步 |
| --- | --- | --- | --- |
| 4.000000 | 2500 / 2500 | 2750 / 2750 | 2500 / 2500 |
| 3.500000 | 8000 / 28250 | 8000 / 8000 | 8000 / 8000 |
| 3.300000 | 18500 / 未达到 | 17250 / 17250 | 17250 / 17250 |
| 3.200000 | 未达到 / 未达到 | 29250 / 29250 | 29000 / 29000 |
| 3.170000 | 未达到 / 未达到 | 36000 / 36000 | 35750 / 35750 |
| 3.408906 | 11250 / 40000 | 11000 / 11000 | 11000 / 11000 |

“持续”表示从该观测点起，之后所有已观测验证点均不高于阈值；阈值分辨率为 250 步。Baseline 有回升，首次命中与持续达到必须区分。它未达到 3.2，图表明确标为 **未达到**，不以零计算量填充缺失值。

![计算量与阈值对比](figures/04_compute_tradeoff.png)

## 5. 图表、数据与复算

[七张图导航](figures/README.md)提供全部 PNG / PDF / SVG。数据包括[三组汇总](summary.json)、[八项分数](downstream.csv)、[训练汇总](training_summary.csv)、[120000 个训练点](training_curves.csv)、[480 个验证点](validation_curves.csv)、[阈值表](loss_thresholds.csv)、[基座核查](comparability_audit.json)。A/B 原始 [80000 点曲线](ab_training_curves.csv)和[21465 条配对分数](ab_paired_scores.csv)保持原样；新增 [baseline 逐题分数与哈希](baseline_sample_scores.csv)，不包含题目、提示或原始模型输出。

Baseline 的最终 checkpoint payload 已重新计算 SHA256，并与八项完整评测 provenance 绑定核对。A/B 沿用此前发布的 checkpoint 元数据与 provenance 校验记录；本机没有下载其 checkpoint payload。公开源文件哈希记录在 summary 中，目录内容由 [MANIFEST.sha256](MANIFEST.sha256) 校验。

从原始本机 baseline 记录更新数值并复做完整模型核查：

```bash
python tools/replace_ab_baseline.py \
  --baseline-run runs/yoco-160m-baseline-8gpu-full-20261104T0114Z \
  --baseline-pilot runs/yoco-160m-baseline-8gpu-pilot-20261104T0109Z
python tools/plot_ab_baseline.py
python tools/plot_ab_comparisons.py
python tools/render_ab_comparison_report.py
python tools/verify_ab_baseline.py --refresh-manifest
```

若只重画图，已有公开 CSV/JSON 足够，无需 checkpoint、原始语料或重新训练。旧导出器 `export_ab_baseline.py` 用于从私有 A/B 原始材料重建初始发布；其历史 baseline 需要再由上面的替换步骤更新。
