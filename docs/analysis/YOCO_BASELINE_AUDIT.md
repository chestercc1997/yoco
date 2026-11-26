# YOCO 160M–830M baseline：预训练设置与 FP16 溢出

**已定位的 gRet 前向溢出已修复。** 这批实验复现了论文的小模型训练规模和大部分优化器设置。影响结果可比性的主要差异是 400M 的前馈层宽度、本地语料与样本构造、输入输出权重共享，以及混合精度和算子实现。只有最后一项已在故障快照上建立了与前向 Inf 的因果关系。

新版 Baseline 的选用、微软 README 的示例参数和论文附录 D 的对照见 [YOCO 官方配置与本仓库复现基准](YOCO_OFFICIAL_CONFIGURATION.md)。

## 历史六组 baseline 与论文如何对应

三个档位各有 gated retention（gRet）和滑动窗口注意力（SWA）两组。下表取自六组实际运行的 `resolved_config.json`，并与下列配置逐项比对。另直接读取六份 checkpoint（五份 40000 步、一份 5004 步），核验每层 FFN 权重形状，以及输入输出权重是否共享存储。小模型的对照依据是论文 §4.2 和附录 D；官方 README 给出的是通用预训练命令，附录 C 则描述另一套 3B 实验。[论文 §4.2](https://arxiv.org/html/2405.05254v2#S4.SS2) · [附录 D，表 7](https://arxiv.org/html/2405.05254v2#A4)

| 档位与本地配置 | 模型宽度 / 层数 | 注意力头 / gRet 头 | 本地 FFN / 论文 YOCO FFN |
| --- | ---: | ---: | ---: |
| 160M [gRet](../../configs/yoco_160m_baseline_8gpu_matched.json) · [SWA](../../configs/yoco_160m_swa_baseline_8gpu.json) | 768 / 12 | 12 / 3 | 2304 / 2304 |
| 400M [gRet](../../configs/yoco_400m_gret_baseline_8gpu.json) · [SWA](../../configs/yoco_400m_swa_baseline_8gpu.json) | 1024 / 24 | 16 / 4 | **2816 / 3072** |
| 830M [gRet](../../configs/yoco_830m_gret_baseline_repaired_8gpu.json) · [SWA](../../configs/yoco_830m_swa_baseline_8gpu.json) | 1536 / 24 | 12 / 6 | 4608 / 4608 |

三档的宽度、层数和注意力头数均对上表 7。每个 gRet 头的维度是 256，三组 SWA 的窗口都是 1024（含当前位置）。**400M 的两组都比论文的 FFN 宽度少 256，即窄 8.33%。** 本地每层有三个无偏置 FFN 投影，按其余结构固定计算，整模因此少了 18,874,368 个参数。这会影响容量和计算量；没有证据将其归因为溢出。[本地 FFN](../../runtime/yoco_official/feedforward_network.py) · [论文附录 D](https://arxiv.org/html/2405.05254v2#A4)

**训练预算与优化器基本对齐。** 六组都配置 AdamW、β=(0.9, 0.98)、峰值学习率 1.5×10⁻⁴、375 步预热后线性衰减、权重衰减 0.05、序列长度 2048、目标 40,000 步。本地每步用 8 卡 × 4 条 × 4 次累积，共 262,144 tokens；完整预算为 10,485,760,000 tokens。论文按约 0.25M tokens/步、约 10B tokens 报告。五组已训练满 40,000 步，830M gRet 在第 9793 次尝试后暂停，最近已核验的完整恢复点为 5004 步。[论文附录 D](https://arxiv.org/html/2405.05254v2#A4) · [八卡恢复核验](../evidence/gret-830m-state-write-exact-restore.json)

| 比较项 | 核对结果与影响 |
| --- | --- |
| 语料与验证集 | 本地使用固定版本 FineWeb-Edu `sample-10BT`，按文档 ID 哈希留出约 0.1% 验证。论文说明各档使用相同训练数据，公开材料不足以重建这些小模型的逐样本语料和验证集。跨项目的绝对 validation loss 不能直接对表。[本地数据来源](../../provenance/fineweb_candidate.json) · [论文 §4.2](https://arxiv.org/html/2405.05254v2#S4.SS2) |
| 分词与切块 | 两边都使用 YOCO 的 cl100k_base 分词方案。本地整篇编码、文末加 EOS、分片内连续切块，按打乱后的块索引取样。官方公开加载器分行编码、文末加 EOS，每个新样本以 BOS 开头。即使用相同原文，模型看到的序列边界也会不同。本地一遍可用训练块约 9.684B tokens，完成 40k 步的运行约有 0.802B tokens 来自第二遍取样。[本地预处理](../../tools/prepare_data.py) · [本地采样器](../../runtime/train.py) · [官方加载器](https://github.com/microsoft/unilm/blob/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/YOCO/yoco/tasks/data/lm_loader.py) |
| 输入输出权重 | 六组均绑定 token embedding 与输出投影。微软公开建模入口调用 YOCO 时沿用默认的独立权重。绑定改变参数量，并让输入和输出两路梯度更新同一张表。附录 D 未单列论文小模型实际使用的绑定开关。[本地入口](../../runtime/model.py) · [官方入口](https://github.com/microsoft/unilm/blob/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/YOCO/yoco/models/yoco.py) · [官方解码器](https://github.com/microsoft/unilm/blob/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/YOCO/yoco/models/decoder/yoco.py) |
| 精度与执行 | 本地为保持既有 A/B 对照，使用 FP16 混合精度、FP32 参数和动态 loss scale，并采用 PyTorch 替代后端。微软公开预训练命令使用并推荐 BF16，附录 D 未逐档披露 dtype。本地六组均用 fused AdamW 并关闭激活重算；正式对照中的 160M gRet 启用了编译，其余五组关闭编译。[官方预训练命令](https://github.com/microsoft/unilm/blob/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/YOCO/README.md#pretraining-from-scratch) · [本地后端](../../runtime/yoco_official/backend.py) |
| 其他超参数 | 本地 AdamW ε=10⁻⁸、梯度裁剪阈值 1.0、末端学习率 0、seed 42。官方 README 示例 ε=10⁻⁶、裁剪阈值 2.0；附录 D 未给出这几项的逐档值。可确认与示例的差别，论文小模型的实际取值仍未知。[官方示例](https://github.com/microsoft/unilm/blob/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/YOCO/README.md#pretraining-from-scratch) · [论文附录 D](https://arxiv.org/html/2405.05254v2#A4) |

## 2816 的来源与后续 baseline 配置

核对仓库锁定的微软提交 `ca43e4cd`，其 `YOCO/` 树中没有随代码发布的 160M、400M、830M 分档配置文件。官方建模入口把调用方的 `hidden_dim`（或外部 `params.json` 的值）传入解码器，FFN 按该值创建权重。论文附录 D 单独规定了 YOCO 的 FFN 宽度为 `3d`。本地首次添加 400M A 和参考 baseline 的提交 `ebf4c7b` 在两个 JSON 中都显式写入 2816；当时的预检比较 A 与本地参考 baseline 的公共配置和初始化，未校验论文公式。之后的 400M baseline 沿用了这个宽度。提交记录可以确定数值的来源，没有给出选择 2816 的计算依据。[官方模型入口](https://github.com/microsoft/unilm/blob/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/YOCO/yoco/models/yoco.py) · [官方 FFN](https://github.com/microsoft/unilm/blob/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/YOCO/yoco/models/decoder/feedforward_network.py) · [本地首次配置](https://github.com/chestercc1997/yoco/commit/ebf4c7b5085268d03a22da33e749f6fa0b8bf2c1)

[六份新配置](../../configs/appendix_d)供从头训练使用。FFN 分别为 2304、3072、4608，输入 embedding 和输出投影采用官方入口默认的独立权重。400M gRet 和 SWA 的实际总参数分别为 **522,160,128** 和 **509,528,064**；“400M”沿用论文规模档位名称。两张权重表的选择会改变总参数量，也改变梯度流向。旧配置和 checkpoint 保留历史定义。

新数据格式在每个来源分片内分行编码、文末加 EOS，每条记录以 BOS 开始。按官方加载器的严格溢出条件截到 2049 个 token，丢弃溢出行的剩余部分和分片末尾待发出的尾样本。训练器整条读取记录，再错位一位构造输入与标签。用锁定的微软分词器和加载器对同一批 FineWeb 原文逐 token 核对，1000 篇训练文档输出 494 条样本，13 篇验证文档输出 8 条，全部一致。两种新 400M 模型均通过 A100 上真实长度 2048 的 FP16 前反向检查。[切块实现](../../tools/yoco_samples.py) · [配置与边界测试](../../tests/test_appendix_d_baselines.py)

这些新配置仍使用本地 FineWeb-Edu 来源、哈希验证集划分、确定性样本顺序和 FP16 配方。微软没有在上述固定版本中提供论文小模型逐样本的语料清单，切块对照验证的是**同一原文对应的训练样本**。新旧权重形状、参数绑定和数据指纹不同，应各用独立训练目录。本次仅做样本及模型验证，未重建全量语料或启动新一轮训练。

在仓库根目录，以下命令用已下载的固定版本原始 parquet 生成六组共用的独立样本 manifest，并校验分片。新格式输出到 `data/fineweb_edu_10bt/yoco_samples_2048/`。从头训练的入口为 `runtime.train --config configs/appendix_d/<规模>_<gret或swa>.json --run-dir <新目录>`，采用既定八卡启动方式。

```bash
python tools/prepare_data.py --offline --config configs/appendix_d/400m_gret.json
python tools/verify_data.py --config configs/appendix_d/400m_gret.json
```

## 为什么前向会出现 Inf

gRet 把许多 token 的贡献相加。即使输入和门值全部有限，**归一化前的和仍可能超出 FP16 最大有限值 65,504**。旧实现过早将结果写入 FP16，产生 Inf，后续 RMSNorm 和 loss 才出现 NaN。已捕获的首次溢出位置有三处：

| 现场 | 首次溢出的位置 | 修复 |
| --- | --- | --- |
| 400M **方法 A**，24935 步 | 读取历史状态的跨块乘积；关闭编译、开关激活重算均可复现 | 跨块乘积与最终求和保留在已有 FP32 累积类型中，归一化后转回 FP16。[故障对照](../evidence/400m-A-retention-fix/before.json) |
| 830M **纯 baseline**，5926、6438、6494 步 | 块内加权求和的输出越界 | 保留原本有限的结果，只对异常元素采用重算值。[三份故障快照](../evidence/gret-state-write-prior-snapshots.json) |
| 830M **纯 baseline**，9793 步 | 写入下一块的状态增量，单个点积约 **65,911**，污染后续 512 个输出 | 同样保护状态写入。八卡回放原 128 条样本，全部前向有限，全局 loss **3.318346**，反缩放梯度范数 **0.756839**。[逐元素核验](../evidence/gret-830m-state-write-9793.json) · [八卡回放](../evidence/gret-830m-state-write-full-update.json) |

用 9793 步本地输入单独调用微软公开的状态写入计算，同一元素在 FP16 下为 Inf，BF16 下为有限值 66048。这说明相同的写回边界足以触发该问题；正式训练仍使用原 FP16 配方。此次官方对照仅覆盖状态写入。纯 baseline 也能复现，首次 Inf 发生在当步 AdamW 更新之前。[官方跨块计算](https://github.com/microsoft/unilm/blob/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/YOCO/yoco/models/decoder/kernel/gate_recurrent.py) · [本地修复](../../runtime/yoco_official/backend.py)

**还有两个历史现场无法定因。** 160M 第 21354 步是三次 AMP 拒绝后出现的有限梯度尖峰。已修复的 scaler 成功步数计数缺陷解释了之后的 scale 过早回升；原始触发权重缺失。400M A 第 14311 步出现非有限前向，原运行与重放从 10001 步起梯度已有差异，重放未复现原故障。两者都不能归入上述三个已证实的位置。[160M 数值审计](../evidence/baseline-numerical-audit.json) · [scaler 回归](../evidence/400m-A-amp-backoff-check.json) · [400M 轨迹对照](../evidence/gret-400m-14311-trajectory.json)
