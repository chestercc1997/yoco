"""Render the current baseline/A/B report from the verified numerical tables."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evaluation/ab_baseline_20261104'
S = json.loads((OUT/'summary.json').read_text())
T = S['training']
ARMS = ('baseline', 'A', 'B')
AUDIT = S['comparability_audit']
with (OUT/'loss_thresholds.csv').open() as handle:
    THRESHOLDS = list(csv.DictReader(handle))
with (OUT/'validation_curves.csv').open() as handle:
    CURVES = list(csv.DictReader(handle))


def table(headers, data):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |'] +
                     ['| '+' | '.join(map(str, row))+' |' for row in data])


quality = table(['指标', 'Baseline · 8×A100', 'A · 8×A800', 'B · 8×A800'], [
    ['最终验证 CE', *[f"{T[a]['validation_loss']:.6f}" for a in ARMS]],
    ['验证 PPL', *[f"{T[a]['validation_perplexity']:.6f}" for a in ARMS]],
    ['最后 100 步平均训练 loss', *[f"{T[a]['last100_training_loss']:.6f}" for a in ARMS]],
    ['八项下游宏平均（%）', *[f"{100*S['macro'][a]:.4f}" for a in ARMS]]])
downstream = table(['任务', '指标', '题数', 'Baseline %', 'A %', 'B %', 'A−Base pp', 'B−Base pp'], [
    [r['task'], r['metric'].split(',')[0], r['examples'], f"{100*r['baseline']:.2f}", f"{100*r['A']:.2f}",
     f"{100*r['B']:.2f}", f"{r['A_delta_pp']:+.2f}", f"{r['B_delta_pp']:+.2f}"] for r in S['downstream']] +
    [['等权宏平均', '—', 21465, *[f"{100*S['macro'][a]:.2f}" for a in ARMS],
      *[f"{100*(S['macro'][a]-S['macro']['baseline']):+.2f}" for a in ('A', 'B')]]])
compute = table(['指标', 'Baseline', 'A', 'B'], [
    ['总参数 / 计入输出头的 N', *[f"{T[a]['parameters']:,}" for a in ARMS]],
    ['主干参数 N', *[f"{T[a]['trunk_parameters']:,}" for a in ARMS]],
    ['训练 tokens D', *[f"{T[a]['tokens']:,}" for a in ARMS]],
    ['Trunk 6ND（PFLOP-days）', *[f"{T[a]['compute_trunk_6nd_pflop_days']:.9f}" for a in ARMS]],
    ['Output-inclusive 6ND', *[f"{T[a]['compute_output_inclusive_6nd_pflop_days']:.9f}" for a in ARMS]],
    ['额外五次 K/V 投影代理', *[f"{T[a]['extra_repeated_kv_projection_pflop_days']:.9f}" for a in ARMS]],
    ['Output-inclusive + 重复 K/V', *[f"{T[a]['output_inclusive_plus_repeated_kv_proxy_pflop_days']:.9f}" for a in ARMS]],
    ['成功 update 活跃 GPU-hours', *[f"{T[a]['active_gpu_hours']:.3f}" for a in ARMS]],
    ['梯度 overflow 尝试数', *[T[a]['overflow_events'] for a in ARMS]]])
targets = list(dict.fromkeys(float(r['target_loss']) for r in THRESHOLDS))
threshold_table = table(['验证 loss 阈值', 'Baseline 首次 / 持续步', 'A 首次 / 持续步', 'B 首次 / 持续步'], [
    [f'{target:.6f}', *[
        ' / '.join((r['first_observed_step'] or '未达到', r['sustained_observed_step'] or '未达到'))
        for a in ARMS for r in THRESHOLDS if r['arm']==a and float(r['target_loss'])==target]] for target in targets])
best = min((r for r in CURVES if r['arm']=='baseline'), key=lambda r: float(r['validation_loss']))
delta_a = 100*(S['macro']['A']-S['macro']['baseline'])
delta_b = 100*(S['macro']['B']-S['macro']['baseline'])
ci = S['AB_paired_bootstrap']['ci95_pp']
report = f'''# YOCO 160M：新 Baseline、A、B 三组对比

更新日期：2026-11-04。Baseline 已替换为本机 **8×A100 80GB PCIe** 完成的原始 YOCO；A/B 保留最新 GitHub 发布的 **8×A800** 运行。三组均完成 **40,000 步、10,485,760,000 个训练 tokens** 和八项完整零样本评测，每组 21,465 道题。

**本次固定 checkpoint 的结果：A 的下游宏平均最高，B 的最终验证 loss 最低。相对新 baseline，A/B 的宏平均分别高 {delta_a:.2f} / {delta_b:.2f} 个百分点。** 本报告中的 baseline、差值、曲线和阈值计算均来自新运行；旧 32×V100 结果仍保留在 Git 历史和原 YOCO/Transformer 报告中。

## 1. 三组基座与设置核查

三组都以同一原始 YOCO-gRet 为基座：12 层、宽度 768、FFN 2304、词表 100288、前六层 self-decoder、后六层 cross-decoder，共享输入 embedding / 输出矩阵。Baseline 从 H6 构造一次 K/V；A/B 仅改变共享 K/V 的六来源融合方式，不更换主干。

| 组别 | K/V 构造 | 总参数 | 比 baseline 新增 |
|---|---|---:|---:|
| Baseline · 8×A100 | 第六层输出生成共享 K/V | {T['baseline']['parameters']:,} | 0 |
| A · 8×A800 | 六个全局 softmax logits 融合候选 K/V | {T['A']['parameters']:,} | 6 |
| B · 8×A800 | 按 hidden states 生成每 token 权重 | {T['B']['parameters']:,} | 774 |

已直接比较发布的运行配置：`architecture`、`model`、`training`、`seed`、`tied_embeddings` 全部相同。三组均为 seed=42、microbatch=4、accumulation=4、8 ranks、长度 2048、每步 262144 tokens、验证 2048 条序列；AdamW，betas=0.9/0.98，LR=1.5e-4，375 步 warmup、40000 步线性衰减，weight decay=0.05，FP16 mixed / FP32 parameters。

**实际重建完整 160M 模型的核查通过：**同 seed 下，A/B 的全部 {AUDIT['matching_common_state_tensors']['A']} 个公共 state tensors 与 baseline 逐项完全一致；仅多出融合模块参数。A/B 初始六层等权且输出相同；将权重退化为仅选 H6 后，两者都复现 baseline 输出。详见[机器可读核查](comparability_audit.json)。这验证当前实现与发布配置的共同基座；远端 A/B 当时的 step-zero 权重未包含在公开材料中，未宣称直接比对其 payload。

八项评测的 task YAML SHA256、seed、num_fewshot=0、题数和 doc_id 集合均匹配。A/B 原始分词分片哈希和逐题 doc/prompt/target 哈希未包含在此公开导出中，无法在本机重新逐项核对，因此 baseline 差值保持描述性，不给其添加配对显著性结论。A/B 原有的配对统计不变。按本次要求，核查不以软件版本差异作为判定项。

## 2. 最终质量和八项下游

{quality}

![三组最终质量](figures/01_quality_summary.png)

{downstream}

BoolQ、WinoGrande 使用 acc，其余六项使用 acc_norm。宏平均对八任务等权，不按题数加权。CSV 保留完整精度和标准误。A/B 原始分数、逐题配对记录、训练曲线均未更改；B−A={100*(S['macro']['B']-S['macro']['A']):.4f} pp，既有配对 bootstrap 95% 区间为 [{ci[0]:.4f}, {ci[1]:.4f}] pp，仅反映这两个固定 checkpoint 的样本不确定性。

![八任务与差值](figures/02_downstream_comparison.png)

## 3. 完整训练与验证曲线

现在提供三组各 **40,000 个训练点**，合计 120,000 点；baseline 由 pilot 的第 1–100 步与续训的第 101–40000 步连续拼接，原 optimizer/scaler/RNG 和完整 LR schedule 均保留。正式验证只取每 250 步一次的 160 个点；pilot 的第 50、100 步额外验证不混入正式曲线。

![三组完整训练 loss](training_loss.png)

![三组验证曲线](figures/03_validation_convergence.png)

新 baseline 的最低已观测验证 loss 为 **{float(best['validation_loss']):.6f}（第 {int(best['step']):,} 步）**，之后存在回升，最终为 {T['baseline']['validation_loss']:.6f}。因此图中保留全部真实波动，统一使用第 40000 步模型比较，不把最好点替换为最终点；最终分数差异不能单独证明改动的普遍收益。曲线坐标已随新 baseline 调整，避免沿用旧范围隐藏灰色曲线。

## 4. 计算量与同 loss 阈值

定义 `C=6ND`，`PFLOP-days=C/8.64e19`。共享输入/输出矩阵共 77,021,184 个参数：trunk 口径减去一次，output-inclusive 口径计入输出矩阵一次。D 统计实际消费的训练 tokens，含重复采样；训练池为 9,683,617,678 tokens，整个训练约消费其 1.083 倍。

{compute}

A/B 对六个来源分别执行共享 K/V 投影。相对 baseline 多五次投影的局部代理为 `6×5×2×768²×D`，在当前 D 下约 0.004294967 PFLOP-days；这仍未包含完整的 retention/attention、归一化、路由、激活重计算和通信成本。**6ND 与局部修正均不是实测设备 FLOPs；A100/A800 的活跃时间不能直接作为架构加速结论。** Baseline 的 wall time 含 pilot 和续训之间的暂停；活跃 GPU-hours 仅累加成功 update 的秒数，不含暂停、溢出重试、验证和存盘。

![不同计算量口径](loss_compute.png)

{threshold_table}

“持续”表示从该观测点起，之后所有已观测验证点均不高于阈值；阈值分辨率为 250 步。Baseline 有回升，首次命中与持续达到必须区分。它未达到 3.2，图表明确标为 **未达到**，不以零计算量填充缺失值。

![计算量与阈值对比](figures/04_compute_tradeoff.png)

## 5. 图表、数据与复算

[七张图导航](figures/README.md)提供全部 PNG / PDF / SVG。数据包括[三组汇总](summary.json)、[八项分数](downstream.csv)、[训练汇总](training_summary.csv)、[120000 个训练点](training_curves.csv)、[480 个验证点](validation_curves.csv)、[阈值表](loss_thresholds.csv)、[基座核查](comparability_audit.json)。A/B 原始 [80000 点曲线](ab_training_curves.csv)和[21465 条配对分数](ab_paired_scores.csv)保持原样；新增 [baseline 逐题分数与哈希](baseline_sample_scores.csv)，不包含题目、提示或原始模型输出。

Baseline 的最终 checkpoint payload 已重新计算 SHA256，并与八项完整评测 provenance 绑定核对。A/B 沿用此前发布的 checkpoint 元数据与 provenance 校验记录；本机没有下载其 checkpoint payload。公开源文件哈希记录在 summary 中，目录内容由 [MANIFEST.sha256](MANIFEST.sha256) 校验。

从原始本机 baseline 记录更新数值并复做完整模型核查：

```bash
python tools/replace_ab_baseline.py \\
  --baseline-run runs/{S['baseline_replacement']['source_run']} \\
  --baseline-pilot runs/{S['baseline_replacement']['source_pilot']}
python tools/plot_ab_baseline.py
python tools/plot_ab_comparisons.py
python tools/render_ab_comparison_report.py
python tools/verify_ab_baseline.py --refresh-manifest
```

若只重画图，已有公开 CSV/JSON 足够，无需 checkpoint、原始语料或重新训练。旧导出器 `export_ab_baseline.py` 用于从私有 A/B 原始材料重建初始发布；其历史 baseline 需要再由上面的替换步骤更新。
'''
(OUT/'README.md').write_text(report)

figures = '''# 新 Baseline / A / B：七张对比图

Baseline 使用本次完成的 **8×A100** 运行，A/B 使用已发布的 **8×A800** 运行；三组同一 YOCO 基座、相同训练超参数、40000 步、10.486B tokens、八项完整评测。基座和初始公共参数的完整模型核查见[报告](../README.md)与[核查 JSON](../comparability_audit.json)。灰色为 baseline，青色为 A，橙色为 B。

## 1. 最终质量总览

比较最终验证 loss、最后 100 步平均训练 loss、八任务宏平均。当前 A 下游最高，B 的验证 loss 最低；图中 baseline 已替换为本机新运行。

![最终质量](01_quality_summary.png)

[PDF](01_quality_summary.pdf) · [SVG](01_quality_summary.svg)

## 2. 八任务得分与相对 baseline 的差值

左图三组得分，右图 A/B 相对新 baseline 的百分点差值。坐标已按新差值重算；每组 21465 题，无 limit，宏平均为八任务等权。Baseline 差值属于描述性比较。

![八任务](02_downstream_comparison.png)

[PDF](02_downstream_comparison.pdf) · [SVG](02_downstream_comparison.svg)

## 3. 相同 token 预算的验证收敛

每组 160 个未平滑验证点。左图放大 7–10.486B tokens，右图为 A/B 减去新 baseline。完整保留 baseline 的回升，未裁掉灰色曲线；三组均使用最终第 40000 步结果。

![验证收敛](03_validation_convergence.png)

[PDF](03_validation_convergence.pdf) · [SVG](03_validation_convergence.svg)

## 4. Compute–loss 与同 loss 的计算代理

左图 output-inclusive 6ND，中图加上 A/B 额外五次 K/V 投影的局部代理，右图按首次观测达到阈值比较。Baseline 未达到 loss=3.2 时明确标注未达到，不显示为零。该代理不是实测硬件 FLOPs；详细首次/持续命中步数见[阈值 CSV](../loss_thresholds.csv)。

![计算量](04_compute_tradeoff.png)

[PDF](04_compute_tradeoff.pdf) · [SVG](04_compute_tradeoff.svg) · [右图 CSV](compute_at_loss.csv)

## 5. 全程验证与四种横轴

![全程验证](../loss_compute.png)

[PDF](../loss_compute.pdf) · [SVG](../loss_compute.svg)

## 6. 八任务分组柱状图

![八任务柱状图](../downstream.png)

[PDF](../downstream.pdf) · [SVG](../downstream.svg)

## 7. 三组完整训练 loss

现在包含 baseline 的全部 40000 步。展示全程、后期放大、相对 baseline 差值和最后 100 步均值；轨迹使用 trailing-100 平均，最终均值直接复算。

![完整训练](../training_loss.png)

[PDF](../training_loss.pdf) · [SVG](../training_loss.svg)

复画使用 `python tools/plot_ab_baseline.py` 与 `python tools/plot_ab_comparisons.py`；这只读取公开数值，不启动训练或评测。[目录 manifest](../MANIFEST.sha256) 覆盖本目录。按本次核查范围，不以软件版本差异作为比较的判定项。
'''
(OUT/'figures/README.md').write_text(figures)

readme = (ROOT/'README.md').read_text()
start = readme.index('**Completed three-way comparison (2026-11-04):**' if '**Completed three-way comparison (2026-11-04):**' in readme else '**Completed results (2026-11-04):**')
end = readme.index('[七张三组对比图' if '[七张三组对比图' in readme else '[四张专题对比图', start)
replacement = f'''**Completed three-way comparison (2026-11-04):** the original YOCO baseline
completed 40,000 updates on **8×A100**; A/B retain the published **8×A800** runs.
All three consumed 10,485,760,000 tokens and completed all eight full zero-shot tasks.

| Metric | New YOCO baseline · 8×A100 | A | B |
| --- | ---: | ---: | ---: |
| Final validation loss | {T['baseline']['validation_loss']:.6f} | {T['A']['validation_loss']:.6f} | {T['B']['validation_loss']:.6f} |
| Eight-task macro average (%) | {100*S['macro']['baseline']:.4f} | {100*S['macro']['A']:.4f} | {100*S['macro']['B']:.4f} |
| Trunk 6ND (PFLOP-days) | {T['baseline']['compute_trunk_6nd_pflop_days']:.9f} | {T['A']['compute_trunk_6nd_pflop_days']:.9f} | {T['B']['compute_trunk_6nd_pflop_days']:.9f} |

The baseline above is the newly completed eight-A100 run. Full-size model checks
confirm the same common backbone tensors at initialization; A adds 6 parameters,
B adds 774. Published model/training settings and all task YAMLs, seeds and zero-shot
counts match. Remote A/B step-zero payloads and token-shard hashes are not available
for a fresh check, so baseline differences remain descriptive. A has the highest
downstream macro; B has the lowest final validation loss. The new baseline has
late-training loss rebounds, shown in the un-smoothed curves. 6ND remains a compute
proxy. See the [complete report, backbone audit, seven figures and CSVs](evaluation/ab_baseline_20261104/README.md).

'''
readme = readme[:start]+replacement+readme[end:]
readme = readme.replace('[四张专题对比图 / Comparison figures]', '[七张三组对比图 / Comparison figures]')
historical_image = '![2K quality and inference summary](evaluation/figures/04_yoco_transformer_2k_32gpu_inference.png)\n\n'
readme = readme.replace(historical_image, '')
readme = readme.replace('## Earlier reproduction result\n\n', '## Earlier reproduction result\n\n'+historical_image)
(ROOT/'README.md').write_text(readme)
index = ROOT/'docs/README.md'
text = index.read_text()
old = '- **2026-11-04：8×A800 A/B 两组训练和全部下游已完成。**'
text = text.replace(old, '- **2026-11-04：8×A100 baseline 与 8×A800 A/B 的完整三组对比已更新，三组均完成训练和八项下游。**')
index.write_text(text)
print('Rendered the three-way report, seven-figure index and repository overview')
