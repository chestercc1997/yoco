# YOCO Baseline 配置与复现

本仓库以 YOCO 论文附录 D 配置 160M、400M 的 gRet 和 SWA 骨干，在固定版 FineWeb-Edu 上从头训练 Baseline。每个规模和架构都用自己的 Baseline 对照 K/V 融合方法。当前 400M gRet 的 raw A/B 与 softmax A/B 共用 `ea779aa` 的 Baseline。 [论文与微软代码的设置对照](analysis/YOCO_OFFICIAL_CONFIGURATION.md)

## 选择配置与冻结版本

| 规模 | 层数 / 宽度 | FFN | gRet 头数 / cross-attention 头数 | Baseline 配置 | 训练版本 |
| --- | --- | ---: | --- | --- | --- |
| 160M | 12 / 768 | 2304 | 3 / 12 | [gRet](../configs/appendix_d/160m_gret.json) · [SWA](../configs/appendix_d/160m_swa.json) | `0ab578b` |
| 400M | 24 / 1024 | 3072 | 4 / 16 | [gRet](../configs/appendix_d/400m_gret.json) · [SWA](../configs/appendix_d/400m_swa.json) | `ea779aa` |

YOCO 将前半层作为 self-decoder，后半层共享由最后一层 self-decoder 生成的 K/V。gRet 每头 256 维，SWA 的窗口为 1024。训练使用 seed 42、FP16 混合精度与 FP32 参数。8 张 GPU 每卡每次处理 4 条长度为 2048 的序列，累积 4 次，每步共 262,144 个预测 token，训练 40,000 步。AdamW 的 β 为 `(0.9, 0.98)`，ε 为 `1e-8`，权重衰减为 `0.05`，梯度裁剪为 `1.0`。学习率在前 375 步线性预热至 `1.5e-4`，随后线性衰减到 0。每 250 步验证，每 5,000 步保存并保留完整 checkpoint。 [160M 配方](160m/README.md) · [400M 配方](400m/results.md)

## 新旧 Baseline 有哪些差异

| 项目 | 旧版 | 新版 | 为什么调整 |
| --- | --- | --- | --- |
| 160M FFN | 2304 | 2304 | 两轮都满足论文的 `3 × 模型宽度`。 |
| 400M FFN | 2816 | 3072 | 按附录 D 的 `3 × 1024` 对齐，24 层共增加 18,874,368 个 FFN 参数。 |
| 输入 embedding 与输出投影 | 绑定，输入查表和输出打分更新同一张矩阵 | 独立，两张矩阵各自学习 | 采用微软公开解码器的默认值。400M 增加 `100,288 × 1024 = 102,694,912` 个参数。 |
| 原始语料与分词器 | 固定版 FineWeb-Edu `sample-10BT`，YOCO 的 `cl100k_base` 封装 | 相同 | 14 个原始 Parquet 分片的 SHA-256、文档 ID 划分规则和词表大小均一致。 |
| 样本构造 | 整篇编码，文末加 EOS，再将连续 token 流切块 | 分行编码，文末加 EOS，每条样本以 BOS 开头，按微软加载器规则组成 2049 个 token，再错位生成输入和标签 | 对齐上游样本边界。这会改变上下文、样本顺序和验证序列。 |
| 一遍训练集 / 验证样本池 | 9,683,603,456 个预测 token / 4,703 条 | 9,384,634,368 个预测 token / 4,557 条 | 两轮都训练 10,485,760,000 个预测 token，分别相当于约 1.083 和 1.117 遍。 |

配对检查显示，四组 Baseline 的旧版与新版都沿用各自相同的 `training` 和 `execution` 配置字段。160M gRet 启用编译，其余三组使用 eager。新旧运行代码的指纹有差别，包括 gRet 的数值保护，因此复跑时也要锁定原来的 `runtime` 版本。 [历史配置审计](analysis/YOCO_BASELINE_AUDIT.md) · [新版打包实现](../tools/yoco_samples.py) · [采样与恢复实现](../runtime/train.py)

绑定时，两路梯度共同更新词表矩阵。独立权重增加了参数、梯度和优化器状态的占用，两种配置都只做一次输出投影。400M 的两项结构变化合计增加 **121,569,280 个参数**。结果沿用论文的档位名称，并单列实际总参数量。

| Baseline | 旧版总参数 | 新版总参数 | 旧版最终验证 loss | 新版最终验证 loss |
| --- | ---: | ---: | ---: | ---: |
| 160M gRet | 166,708,224 | 243,729,408 | 3.177219 | 3.151550 |
| 160M SWA | 163,155,456 | 240,176,640 | 3.180143 | 3.166415 |
| 400M gRet | 400,590,848 | 522,160,128 | 3.005298 | 2.916549 |
| 400M SWA | 387,958,784 | 509,528,064 | 2.945771 | 2.943127 |

表中 loss 是各轮在自己验证序列上的最终值。它们记录两套配方的结果，无法单独估计 FFN、权重绑定或切块的影响。比较融合方法时，使用同一轮的 Baseline、相同验证样本和累积 token 预算。旧版 400M gRet softmax 的收益属于旧配方，新版 raw 与 softmax 的对照则固定了这些变量。 [旧版 160M 与 400M gRet 汇总](../evaluation/model_comparison_20261111/README.md) · [旧版 400M SWA](../evaluation/swa_400m_ab_20261111T054430Z/README.md) · [新版 160M](160m/results.md) · [新版 400M](400m/results.md)

## 从头复跑一组 Baseline

1. **准备并校验数据。** 在仓库根目录激活 YOCO 环境。已下载固定版本的原始 Parquet 时，可离线打包。已有打包结果时可直接执行验证。四组使用同一份 `yoco_samples_2048/manifest.json`，固定原始语料 revision 为 `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`。首次下载时省略 `--offline`。 [环境定义](../environment.a100.yml) · [数据准备入口](../scripts/README.md)

   ```bash
   source scripts/activate_local.sh
   python tools/prepare_data.py --offline --config configs/appendix_d/400m_gret.json
   python tools/verify_data.py --config configs/appendix_d/400m_gret.json
   ```

2. **冻结代码并在新目录启动。** 以下以 400M gRet 为例，需要单机八张空闲 GPU。复跑其他档位或架构时，使用上表对应配置和版本。归档的 `runtime`、配置和版本标记都保存在该次运行的 `release/` 中。

   ```bash
   (
     set -eo pipefail
     source scripts/activate_local.sh
     repo="$PWD"
     revision=ea779aa879215fe13010b21caacf3f6040eec4a1
     config=configs/appendix_d/400m_gret.json
     run_dir="$repo/runs/reproduce-400m-gret-$(date -u +%Y%m%dT%H%M%SZ)"

     mkdir -p "$repo/runs"
     mkdir "$run_dir"
     mkdir "$run_dir/release"
     git archive "$revision" runtime "$config" | tar -x -C "$run_dir/release"
     ln -s "$repo/data" "$run_dir/release/data"
     git rev-parse "$revision" > "$run_dir/release/REVISION"

     cd "$run_dir/release"
     export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
     export TORCHINDUCTOR_COMPILE_THREADS=1
     torchrun --standalone --nproc-per-node=8 -m runtime.train \
       --config "$config" --run-dir "$run_dir" > "$run_dir/controller.log" 2>&1
   )
   ```

3. **核对结果与恢复依据。** 用 `resolved_config.json` 核对实际配方和代码指纹，用 `rank0.jsonl` 读第 40,000 步的训练、验证 loss。训练结束时核对 `stage_complete.json`、`latest.json` 和八份完整 checkpoint。续训时从同一 `release/` 使用同一配置和运行目录，训练器会按指纹和 SHA-256 核验 checkpoint，并恢复模型、优化器、AMP scaler 和八卡 RNG。汇报下游时以最终 checkpoint 跑固定的八项全量零样本测试，取八项等权平均。 [训练入口](../runtime/train.py) · [下游评测入口](../evaluation/run.py)

论文可表述为 **在固定的 FineWeb-Edu 配方上，按 YOCO 附录 D 对齐架构和主要超参数，并参照微软公开加载器构造样本**。官方和本地的绝对验证 loss 需要在同一套样本、分词、上下文长度和计分协议下比较。 [官方与本地配置对照](analysis/YOCO_OFFICIAL_CONFIGURATION.md)
