# YOCO 与 VertiKV 的末段训练效率

**历史 softmax 对照中，400M gRet 的 A/B 相对各自实验的 baseline 得分为 1.725× 和 1.780×。同档 SWA 的 A/B 分别为 0.950× 和 0.904×。** 新配方使用同一评分公式。 [新版 160M 结果](../160m/results.md) · [新版 400M 结果](../400m/results.md)

## 计算口径

VertiKV 的 final-200 口径先按已消耗 token 数匹配 baseline 的训练交叉熵，再逐点转换成效率倍数。基于损失幂律 `L(C) ∝ C^(-α)`，固定 `α = 0.057`。下面的分数对逐点倍数取算术平均。

```text
score(t) = (train_CE_method(t) / train_CE_baseline(t))^(-1 / 0.057)
final_score = mean(score(t) for t in the last 200 logged training updates)
```

大于 1 表示同 token 数下的训练损失优势。这里用固定指数表达损失差对应的估算计算倍数。本文计算没有重新拟合 YOCO 的指数，也没有实测达到目标损失所需的 GPU 时间、FLOPs 或能耗。[VertiKV 方法说明](https://github.com/chestercc1997/vertikv/blob/8aee2653ef025c95f4a449b039b06f40a212731c/experiments/01_lm_quality_scaling/final200/README.md)

## YOCO

四套完成的三组对照均使用第 39,801–40,000 步的训练 loss，baseline 与 A/B 按累计 token 数逐点精确匹配。末步累计 10,485,760,000 tokens。历次试跑与回退丢弃的后缀由原报告的训练血统处理。计算保留正式血统内所有末段数据点。

| 架构与档位 | Baseline 平均 CE | A 平均 CE | B 平均 CE | A 效率倍数 | B 效率倍数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gRet 160M | 3.170674 | 3.152103 | 3.151212 | 1.109× | 1.114× |
| gRet 400M | 2.991673 | 2.900161 | 2.894959 | 1.725× | 1.780× |
| SWA 160M | 3.173536 | 3.169411 | 3.169232 | 1.023× | 1.024× |
| SWA 400M | 2.930443 | 2.939047 | 2.947388 | 0.950× | 0.904× |
| SWA 830M | 2.770138 | — | — | — | — |
| gRet 830M | — | — | — | — | — |

SWA 830M 只完成了 baseline，gRet 830M 暂停在最后一次成功更新 9792，两种 830M 架构均缺少完成的 A/B。表中的 400M 数字来自历史的 FFN 2816、绑定输入输出权重的对照。现行 400M 的 FFN 为 3072，使用独立输入输出权重。 [830M 队列](../history/BASELINE_830M_QUEUE.md) · [配置审计](YOCO_BASELINE_AUDIT.md)

历史 400M gRet baseline 全程无非有限前向和 checkpoint 回滚，有 12 次 AMP 重试，最大有限 pre-clip 梯度范数 1348.629 出现在第 22891 步。计分窗口内该范数最大为 4.719。方法 A 包含编译及数值修复前的健康训练前缀。400M gRet 的高分描述了这一套单 seed 对照的末段损失差，仍需独立复现来验证幅度。 [400M 原始曲线和训练记录](../../evaluation/gret_400m_3way_20261108T194918Z/README.md)

## VertiKV

下表来自逐点复算。700M Gate-F 的档案提供 192 个对齐点，其余组均有 200 个。Gate-E 在四个档位的得分均最高。

| 档位 | Gate-A | Gate-B | Gate-C | Gate-D | Gate-E | Gate-F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 150M | 1.538× | 1.136× | 1.134× | 1.630× | **1.772×** | 1.568× |
| 300M | 1.489× | 1.133× | 1.144× | 1.574× | **1.605×** | 1.486× |
| 500M | 1.217× | 1.050× | 1.095× | 1.350× | **1.432×** | 1.323× |
| 700M | 1.210× | 0.993× | 1.011× | 1.350× | **1.401×** | 1.251× |

按各项目对自身 baseline 的相对分数比较，YOCO 两种 160M 对照的提升低于 VertiKV 150M 的 Gate-A 和 Gate-E。YOCO 400M gRet 的两种方法得分高于 VertiKV 300M 和 500M 的 Gate-E，而 YOCO 400M SWA 的两种方法均低于本组 baseline。两边使用不同语料、训练预算和模型配置，160M 对 150M 与 400M 对 300M/500M 只提供邻近档位参考。[VertiKV 逐点数据](https://github.com/chestercc1997/vertikv/blob/8aee2653ef025c95f4a449b039b06f40a212731c/experiments/01_lm_quality_scaling/final200/data/final200_efficiency_points.csv) · [VertiKV 训练配置与预算](https://github.com/chestercc1997/vertikv/tree/8aee2653ef025c95f4a449b039b06f40a212731c/experiments/01_lm_quality_scaling)

## 复算

从仓库根目录运行脚本，`--vertikv` 指向 VertiKV 仓库的固定版本 `8aee2653`。脚本验证 YOCO 四组三臂各有连续 40000 步，并核对复算的全部 28 条 VertiKV 汇总与公开结果一致。830M 单基线状态列在上表，CSV 记录具备对照的数据。[复算代码](../../tools/compare_training_efficiency.py) · [完整数值 CSV](../evidence/vertikv-training-efficiency-final200.csv)

```bash
python tools/compare_training_efficiency.py --vertikv /root/vertikv \
  --output docs/evidence/vertikv-training-efficiency-final200.csv
```
