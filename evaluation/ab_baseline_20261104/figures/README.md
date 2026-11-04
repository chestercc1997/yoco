# Baseline / A / B 对比图

三组均为 YOCO 160M、40,000 步、10.486B 训练 tokens。Baseline 为历史 32×V100 结果，A/B 为 8×A800 完成结果；具体协议与证据边界见[完整报告](../README.md)。统一用灰色表示 baseline、青色表示 A、橙色表示 B。

## 1. 最终质量总览

并列比较最终验证 loss、最后 100 步平均训练 loss、八任务下游宏平均。使用点图展示小差异，避免截断柱状图放大视觉差距。A 的 loss 和下游均略优；B 的 loss 最低，下游却最低。

![最终质量总览](01_quality_summary.png)

[PDF](01_quality_summary.pdf) · [SVG](01_quality_summary.svg)

## 2. 八任务得分与相对 baseline 的差值

左边是三组绝对 accuracy，右边是 A/B 相对 baseline 的百分点差值，正数表示更高。宏平均对八任务等权，不能与按样本数加权混用。历史 baseline 差值是描述性比较，不是配对显著性结论。

![八任务与差值](02_downstream_comparison.png)

[PDF](02_downstream_comparison.pdf) · [SVG](02_downstream_comparison.svg)

## 3. 相同 tokens 下的验证收敛

左边放大最后约 3.5B tokens，右边直接画 A/B 减去 baseline 的验证 loss。曲线未平滑，直接使用每 250 步一次的原始验证点；负值表示 loss 更低。右图从 1B tokens 开始，完整训练曲线见[原始全程图](../loss_compute.png)。

![收敛与loss差值](03_validation_convergence.png)

[PDF](03_validation_convergence.pdf) · [SVG](03_validation_convergence.svg)

## 4. Compute–loss 与同 loss 的计算量

左边使用计入共享输出头的 `6ND`；中间额外计入 A/B 重复执行 K/V 投影的估算；右边比较首次实测达到同一验证 loss 时的计算代理相对差值。

![计算量口径对比](04_compute_tradeoff.png)

[PDF](04_compute_tradeoff.pdf) · [SVG](04_compute_tradeoff.svg) · [右图数值 CSV](compute_at_loss.csv)

只计参数量会漏掉共享投影的重复执行。加上这部分后，在 loss=3.2 时 A/B 的代理计算量分别比 baseline 高约 1.80% / 0.93%；在 baseline 最终 loss=3.163661 时低约 1.64% / 2.29%。这一局部修正仍未完整计入 attention/retention、路由、归一化、激活重计算等运算，因此不能据此认定真实 FLOPs 或训练效率收益。阈值分辨率为 250 步；三组协议限制同完整报告。

所有图都由已发布数值生成，无新训练或评测。复算命令：

```bash
python3 tools/plot_ab_comparisons.py
```

PNG 适合浏览，PDF/SVG 为矢量格式，适合论文或幻灯片。根目录的 [MANIFEST.sha256](../MANIFEST.sha256) 包含本目录文件。
