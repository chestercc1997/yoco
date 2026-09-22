# YOCO 与官方 Transformer baseline 对比

同数据、同 tokenizer、同训练预算，单个 seed=42。YOCO 166,708,224 参数；Transformer 161,975,040 参数。

| 模型 | 验证 loss | 验证困惑度 | 最后 100 步平均训练 loss |
| --- | ---: | ---: | ---: |
| Transformer | 3.1613 | 23.6008 | 3.1544 |
| YOCO | 3.1637 | 23.6570 | 3.1558 |

| 任务 | 指标 | Transformer | YOCO | YOCO 差值（百分点） |
| --- | --- | ---: | ---: | ---: |
| arc_challenge | acc,none | 21.08% | 21.08% | +0.00 |
| arc_challenge | acc_norm,none | 25.43% | 25.51% | +0.09 |
| arc_easy | acc,none | 51.77% | 51.35% | -0.42 |
| arc_easy | acc_norm,none | 47.60% | 46.09% | -1.52 |
| boolq | acc,none | 60.12% | 61.53% | +1.41 |
| hellaswag | acc,none | 28.31% | 28.56% | +0.25 |
| hellaswag | acc_norm,none | 31.00% | 31.33% | +0.33 |
| openbookqa | acc,none | 17.40% | 17.60% | +0.20 |
| openbookqa | acc_norm,none | 29.40% | 30.80% | +1.40 |
| piqa | acc,none | 63.28% | 63.38% | +0.11 |
| piqa | acc_norm,none | 62.51% | 62.46% | -0.05 |
| sciq | acc,none | 76.40% | 74.80% | -1.60 |
| sciq | acc_norm,none | 67.10% | 64.20% | -2.90 |
| winogrande | acc,none | 49.88% | 51.78% | +1.89 |
