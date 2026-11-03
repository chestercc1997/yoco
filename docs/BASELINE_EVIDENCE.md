# Baseline 选择与可复现材料核查

核查日期：2026 年 10 月 29 日。本文保留早期候选调研与材料来源；其中的路线建议不代表当前计划。当前采用 YOCO baseline 与动态融合配对，见 [当前总览](DYNAMIC_YOCO_THEORY_AND_EXPERIMENTS.md)。

**我的选择：当前 YOCO 仓库用于快速复刻与架构开发；静态 CED 是论文最重要的直接对照；Pythia 410M 是优先核查的官方训练轨迹参照。** 不建议在一个月内从零复刻某个完整商用小模型的最终分数。

## 1 先看哪些候选真正符合要求

| 候选 | 数据与配置 | 日志和 checkpoint | 本项目中的用途 |
| --- | --- | --- | --- |
| 当前 YOCO 160M 复现 | 固定 FineWeb-Edu revision，完整本地配置 | 仓库内有验证曲线 CSV、结果 JSON；不带权重和完整逐步训练日志 | 最容易先复刻的工程基线 |
| Microsoft 官方 YOCO | 官方实现和论文超参数；论文预训练使用的数据不能直接由本仓库公共数据替代 | 有模型与结果；本轮没有核实到可直接复刻原始小规模训练的数据顺序与完整日志闭环 | 最接近 CED 的架构对照 |
| Pythia 410M 与 1B | 官方配置、预分词数据、顺序重建工具 | 官方给出 W&B 和中间 checkpoint；run 对应关系需要核对 | 优先的官方训练轨迹参照 |
| OLMo 1B | 官方数据、训练代码、配置及相关 provenance | 官方给出 W&B、每千步 checkpoint；不同发布版本必须严格区分 | 完整开放训练配方的备选 |
| SmolLM2 360M | 已找到官方预训练 YAML 和数据配方入口 | 本轮未核实到该尺寸可导出的逐步 loss 与 checkpoint 的完整对应关系 | 暂不作为首选复刻对象 |

“有公开日志入口”与“已经拿到准确 run 的数值日志”分开记录。当前 W&B 链接会重定向，浏览工具未成功读出 Pythia 的数值历史；OLMo 日志页也只返回页面外壳。本轮没有导出它们的 CSV，更没有跑过训练复刻。

## 2 第一轮复刻就选当前仓库的 160M 配对实验

这组实验最符合“先跑起来、能够核对效果”的要求，因为配置、数据锁和验证曲线都已经在手里。

| 项目 | 已找到的具体材料 |
| --- | --- |
| 代码版本 | `25c94928858e1a5945ce653d27b4b53907fc1062` |
| YOCO 参数 | 166,708,224 |
| Transformer 参数 | 161,975,040 |
| 配置 | [YOCO](../configs/yoco_160m.json)、[Transformer](../configs/transformer_160m.json) |
| 训练数据 | FineWeb-Edu sample-10BT，revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| 数据文件和校验值 | [数据 provenance](../provenance/fineweb_candidate.json) |
| tokenizer | 当前实现使用 cl100k_base；复制时还需固定编码资源和 tokenized manifest |
| 训练长度 | 2048 |
| 每次更新 tokens | 262,144 |
| 更新次数 | 40,000 |
| 总训练 tokens | 10,485,760,000 |
| 可核对曲线 | [每个验证点的 CSV](../evaluation/figures/training_efficiency_validation_points.csv) |
| 最终结果 | [training_comparison.json](../evaluation/results/training_comparison.json) |

已有最终 validation loss 为 Transformer 3.1612818241、YOCO 3.1636610031。这是本仓库作者报告的结果，不是此次重跑结果。复刻时不仅看终点，还比较例如 step 1000、2000、4000 的验证曲线。

原配置为 32 卡 × microbatch 4 × accumulation 1。四卡可先使用 microbatch 4、accumulation 8，保持每次更新的样本数和 tokens 不变。仍需保持相同的全局样本顺序和验证批次；改变 GPU 数和数值后端后不要求逐 bit 相同。

若改为 BF16，应记录这是数值实现变化；当前训练器写死 FP16，改 JSON 不能实际切换。先复刻 FP16 参考路径，再单独检查 BF16 的曲线差异更容易定位问题。

**这组适合验证训练器，但不足以独自充当新 CED 论文的全部 baseline。** 后续还需静态 CED 和更大规模的受控比较。

## 3 官方训练轨迹优先看 Pythia 410M

Pythia 提供不同规模的公开模型、预分词数据和顺序重建工具，并发布训练过程中的 checkpoint。官方 README 还链接 loss 曲线项目及小模型多 seed 项目。[官方仓库](https://github.com/EleutherAI/pythia)

入口如下：

- [410M 模型](https://huggingface.co/EleutherAI/pythia-410m)
- [1B 模型](https://huggingface.co/EleutherAI/pythia-1b)
- [官方训练日志项目](https://wandb.ai/eleutherai/pythia)
- [小模型多 seed 日志](https://wandb.ai/eleutherai/pythia-extra-seeds)
- [官方 deduped 预分词数据](https://huggingface.co/datasets/EleutherAI/pythia_deduped_pile_idxmaps)
- [standard 预打乱数据入口](https://huggingface.co/datasets/EleutherAI/pile-standard-pythia-preshuffled)

选 standard checkpoint 必须配 standard 数据；不能为了下载方便换成 deduped。也不能混用 v0 和修订版。数据仓库页面可访问，不代表所有训练文件已经在本机完整下载并验证。

我已保存官方 YAML 的固定版本，来源与 SHA256 见 [manifest](baseline_sources/manifest.json)。其中 410M 配置明确给出：24 层、hidden size 1024、16 heads、长度 2048、peak LR 3e-4、min LR 3e-5、Adam betas 为 0.9/0.95、epsilon 1e-8、weight decay 0.1、143,000 次更新、cosine schedule、warmup 比例 0.01。[本地官方配置副本](baseline_sources/pythia_models_410M_pythia-410m.yml)

### 1B 配置存在需要解决的真实差异

| 项目 | 当前官方 README | 固定提交下的 1B YAML |
| --- | --- | --- |
| peak LR | 3e-4 | 2.5e-4 |
| standard 1B 精度 | 说明改用 BF16 | 配置仍启用 FP16 |

因此不能直接说“复制 YAML 就能完全复刻官方 1B”。正式复刻前需读取准确 W&B run 的 resolved config、对应代码版本和 checkpoint 元数据，解释差异后再开跑。[已保存的 1B YAML](baseline_sources/pythia_models_1B_pythia-1b.yml)

410M 的日志项目同样需要确定具体 run，而不是仅使用项目首页。本轮把它列为首选候选，尚未把“官方 run 已完全锁定”标为完成。

### 只复刻前段曲线时怎样保持可比

Pythia 官方训练全局 batch 为 2,097,152 tokens。由此计算，step1000 是 2.097152B tokens，step4000 是 8.388608B，step5000 是 10.48576B。可以选择一个前段 checkpoint 作为校验目标，不必复刻完整约 300B-token 训练。

但如果目标是对齐原始曲线，必须保留官方长程 LR schedule 的前段，不能把 cosine 终点缩到 10B 后再声称是同一实验。也要核对相同数据前缀、初始化、loss 定义、packing 和验证集。相同 seed 在不同实现里不保证相同初始权重，必要时对照官方 step0 权重。

如果为了研究效率重新设计 5B/10B 的短程 schedule，就为所有架构统一重训，并称为“受控短预算实验”；官方曲线此时只用于实现校验，不是可以直接逐点相减的实验对手。

Pythia 是 GPT-NeoX 风格，不能在复刻时悄悄换成 Llama 的归一化、FFN 或 GQA。后续 CED 研究可独立采用合适组件；直接静态与动态对照需要保持这些组件一致，并补一个同组件的普通 Transformer，隔离 backbone 差异。

## 4 OLMo 为什么作为备选

OLMo 1B 官方模型卡直接给出 Dolma 数据入口、W&B 日志、训练超参数和中间 checkpoint。开放材料较完整，但完整模型训练了 3T tokens，不适合在本项目一个月内从头复刻最终效果。[官方模型卡](https://huggingface.co/allenai/OLMo-1B)；[日志入口](https://wandb.ai/ai2-llm/OLMo-1B/reports/OLMo-1B--Vmlldzo2NzY1Njk1)

当前 OLMo 仓库同时包含 OLMo、OLMo 0724 与 OLMo 2 等配方。已找到的 `configs/official-0724/OLMo-1B.yaml` 不能自动当作最初 OLMo-1B 的精确 run 配置。若选择 OLMo，必须把 checkpoint 名、release、配置、数据顺序和日志绑定到同一版本。它更适合作为第二条复刻路线，不应与 Pythia 一起在第一个月同时重建全部环境。

SmolLM2 的 [360M 配置](https://github.com/huggingface/smollm/blob/main/text/pretraining/smollm2/config_smollm2_360M.yaml) 已找到，但目前没有完成对应日志、训练顺序及中间 checkpoint 的闭环核查，所以暂不推荐它作为“立即可精确复刻”的第一选择。

## 5 我会按什么顺序选定最终方案

1. **先跑现有 160M 配对短程复刻。** 直接对照仓库 CSV，验证数据和训练环境。
2. **核定一个 Pythia 410M 官方 run。** 拿到 resolved config、数值日志和相同 token 位置的 checkpoint；检查数据可下载及顺序可恢复。先不启动几百 GB 全量下载。
3. **只挑一条正式数据与训练路线。** 若官方前缀精确复刻是硬要求，用核定后的 Pythia 路线；若主要目标是一个月内完成 CED 架构论文实验，保留现有 FineWeb-Edu 路线，发布自己的完整日志，Pythia 仅作外部验证参照。
4. **先完成静态 CED，再比较动态版本。** Transformer、YOCO 和 CED 必须在同一正式路线中重训；不同 tokenizer 的 PPL 不直接横比。
5. **400M 做主要证据，1B 做放大验证。** 完整官方最终分数的复刻与短预算架构研究分开表述。

还需补齐的复刻材料只有具体三类：准确 run 的配置与数值日志、可验证的数据前缀、对应 checkpoint 的评测复现。没有这些，就不将一个候选标为“已可完整复刻”。
