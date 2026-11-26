# YOCO 官方配置与本仓库复现基准

**本仓库 160M 和 400M 的新版配置适合做受控复现。** 架构和训练预算以论文 §4.2、附录 D 为基准，样本构造参照微软公开加载器，语料、采样和运行设置按本仓库的冻结配方执行。400M raw A/B 与 softmax A/B 共用新版 400M Baseline。

## 论文规模实验的设置

论文附录 D 的 YOCO FFN 宽度为 `3 × 模型宽度`，gRet 每头 256 维。§4.2 的 SWA 窗口为 1024。下面的 gRet 头数由模型宽度除以 256 得出。 [论文 §4.2](https://arxiv.org/html/2405.05254v2#S4.SS2) · [附录 D 表 7](https://arxiv.org/html/2405.05254v2#A4)

| 档位 | 层数 | 模型宽度 | 注意力头 | gRet 头 | YOCO FFN 宽度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 160M | 12 | 768 | 12 | 3 | 2304 |
| 400M | 24 | 1024 | 16 | 4 | 3072 |
| 830M | 24 | 1536 | 12 | 6 | 4608 |

这三个档位的训练长度为 2048，每步约 0.25M token，共 40,000 步、约 10B token。AdamW 的 β 为 `(0.9, 0.98)`，峰值学习率 `1.5e-4`，预热 375 步后线性衰减，权重衰减 `0.05`。 [论文附录 D](https://arxiv.org/html/2405.05254v2#A4)

## 微软 GitHub 公开代码的设置

上游固定参考为 `microsoft/unilm` 提交 `ca43e4cd19445a536f133bf2bc25b573b2f0c7c5`。模型入口的 `LanguageConfig` 声明宽度 1024、24 层、FFN 3072、4 个 self-decoder 头和 **8 个注意力头**。构建时可从参数或 `params.json` 读取模型形状。解码器默认使用独立的输入 embedding 和输出投影。要配置论文 400M 档位，注意力头数取论文表 7 的 **16**。 [上游模型入口](https://github.com/microsoft/unilm/blob/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/YOCO/yoco/models/yoco.py#L29-L118) · [解码器构造函数](https://github.com/microsoft/unilm/blob/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/YOCO/yoco/models/decoder/yoco.py#L241-L256)

| 设置 | GitHub README 预训练示例 | 论文附录 D | 本仓库新版 |
| --- | --- | --- | --- |
| 训练长度 / 更新步数 | 2048 / 300,000 | 2048 / 40,000 | 2048 / 40,000 |
| Batch | 单进程，`batch-size=1`，`update-freq=1` | 约 0.25M token/步 | 8 卡，每卡 4 条，累积 4 次，262,144 token/步 |
| 优化器与 β | CLI 为 `adam`，`(0.9, 0.95)` | AdamW，`(0.9, 0.98)` | AdamW，`(0.9, 0.98)` |
| 峰值学习率 / 预热 | `1.5e-4` / 50 步 | `1.5e-4` / 375 步 | `1.5e-4` / 375 步 |
| 衰减 / 权重衰减 | `polynomial_decay` / `0.05` | 线性 / `0.05` | 线性至 0 / `0.05` |
| Adam ε / 梯度裁剪 | `1e-6` / `2.0` | 未披露 | `1e-8` / `1.0` |
| 精度 | BF16 | 未逐档披露 | FP16 混合精度，FP32 参数 |
| 验证 / checkpoint | 示例关闭验证，每 5,000 步保存 | 用验证 loss 报告规模结果 | 每 250 步验证，每 5,000 步完整保存并保留 |

README 用 `yoco_base` 演示启动命令。上表列出的是该命令显式提供的训练参数，本仓库按附录 D 设置论文规模实验。 [固定版本 README](https://github.com/microsoft/unilm/blob/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/YOCO/README.md#pretraining-from-scratch) · [本仓库 400M gRet 配置](../../configs/appendix_d/400m_gret.json)

## 样本构造与采样

1. **微软加载器按文件读取 JSONL。** 每行是一篇文档，正文按段落、行拆开编码，跳过空行和超长行，最后一段加 EOS。官方预训练示例使用 `cl100k_base`。 [分词和读取源码](https://github.com/microsoft/unilm/blob/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/YOCO/yoco/tasks/data/lm_loader.py#L222-L286)
2. **每条样本以 BOS 开头。** 同一文件内可跨文档累计 token。下一段使长度超过 `tokens_per_sample + 1` 时输出定长样本，截去该段余量。文件末尾未输出的部分丢弃。设定长度 2048 时，2049 个 token 错位成 2048 个输入和标签。 [打包源码](https://github.com/microsoft/unilm/blob/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/YOCO/yoco/tasks/data/lm_loader.py#L288-L302) · [输入标签构造](https://github.com/microsoft/unilm/blob/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/YOCO/yoco/tasks/data/lm_loader.py#L144-L148)
3. **微软加载器复用并打乱文件列表。** 多来源按 `weight` 或 `count` 混合。本仓库将相同样本边界用于固定版 FineWeb-Edu `sample-10BT`，按文档 ID 划分训练与验证，再按 seed 42 打乱已打包的样本。论文 §4.2 未披露用于规模实验的确切训练和验证样本。 [上游采样](https://github.com/microsoft/unilm/blob/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/YOCO/yoco/tasks/data/lm_loader.py#L173-L220) · [本地打包](../../tools/yoco_samples.py) · [本地采样](../../runtime/train.py) · [论文 §4.2](https://arxiv.org/html/2405.05254v2#S4.SS2)

## 已校对的本地基准

| 实验 | 应使用的 Baseline 配置 | 已训练的代码快照 |
| --- | --- | --- |
| 160M gRet / SWA | [gRet](../../configs/appendix_d/160m_gret.json) · [SWA](../../configs/appendix_d/160m_swa.json) | `0ab578b` |
| 400M gRet / SWA | [gRet](../../configs/appendix_d/400m_gret.json) · [SWA](../../configs/appendix_d/400m_swa.json) | `ea779aa` |

新版 400M gRet 的 Baseline、raw A/B 和 softmax A/B 使用同一套 3072 FFN、独立权重、训练设置和数据清单。早期 400M 使用 FFN 2816、权重绑定以及整篇编码后连续切块，应在报告中标为历史配方。核对已运行实验时，以对应 run 的 `resolved_config.json`、`release/REVISION` 和数据 manifest 为准。 [历史设置审计](YOCO_BASELINE_AUDIT.md) · [160M 结果](../160m/results.md) · [400M 结果](../400m/results.md)

校对时将上述四组 Baseline 的仓库配置、冻结副本和实际运行配置逐项比较，全部一致。使用同一分词器对比固定版微软加载器和本地打包器，1000 篇训练文档生成的 494 条样本与 13 篇验证文档生成的 8 条样本均逐 token 一致。 [模型与样本边界测试](../../tests/test_appendix_d_baselines.py)

论文中可表述为 **按 YOCO 附录 D 对齐架构和主要超参数，在固定 FineWeb-Edu 上做受控复现**。方法收益以同规模、同架构、同数据的本地 Baseline 为参照。若要比较本地与官方的绝对验证 loss，需要在同一验证样本、分词、上下文长度及计分协议下评测。 [论文验证结果](https://arxiv.org/html/2405.05254v2#S4.SS2)
