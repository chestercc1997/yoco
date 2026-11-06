# 400M 方法 A 八卡训练

2026 年 11 月 5 日，400M 方法 A 已在本机 8×A100 80GB PCIe 上开始完整训练，目标为 40000 步及八项下游评测。当前尚无最终指标；[启动时状态](evidence/400m-A-launch.json)是发布时的快照，实时状态在本机 `runs/setup/400m-A-active.json`。160M 已完成结果继续保存在[原三组报告](../evaluation/ab_baseline_20261104/README.md)，两种规模的结果分开记录。

2026 年 11 月 6 日，修复版已从原第 10000 步 checkpoint 恢复正式训练，目标仍为 40000 步及八项下游。旧 run 在完成 14310 步后，rank3 下一步前向 loss 非有限而停止。来源 checkpoint 校验和、权重及 optimizer 状态正常，出错卡对应批次没有越界 token。使用旧不可变代码从 10000 隔离重放到 14311，这次没有重现前向 NaN；重放快照的 eager/compiled 运算均有限，见[故障核查记录](evidence/400m-A-failure-review.json)及[快照对照](evidence/400m-A-snapshot-check.json)。这些诊断步数不计为修复版正式训练进度，也不能证明已定位原 NaN 的唯一根因。

排查已确认 AMP 回退逻辑有误：手动调用 `scaler.update(new_scale=...)` 只降低 scale，不会清零连续成功计数。因此原记录中溢出后仅成功 64 步，scale 就再次增长，违背原设定的连续成功 2000 步。当前训练器通过公开的 scaler state API 清零该计数；FP16、FP32 参数、初始 scale、growth/backoff 系数及其余配方均不变。[回归证据](evidence/400m-A-amp-backoff-check.json)核对了原生 AMP 行为、其他 rank 溢出的同步回退及 state 保存/恢复。该缺陷已修复，但尚不能断言它是前向 NaN 的唯一原因。前向出现非有限 loss 时还会保留模型、输入、目标和 scaler 的诊断现场，并停止；诊断文件不是正式 checkpoint，不用于跳过 batch 或自动重试。

修复版已完成真实数据八卡 `10000→10002→10004`、`10000→10004` 及一次独立连续重复对照，见[恢复证据](evidence/400m-A-repair-resume-check.json)。八卡 RNG、scaler、optimizer step 和参数组均精确一致，首步 loss 在三次运行中相同。训练中 checkpoint 的 FP16/Adam 舍入波动比初始化时更大：恢复模型最大差异约 1.058e-4，连续重复自身约 1.103e-4；全模型相对 L2 分别约 6.24e-6 和 4.58e-6。原初始化的逐元素阈值会连独立连续重复也拒绝，因此训练中恢复改用独立重复误差的 2 倍以内，同时保留固定上限（模型最大绝对差 1.5e-4、相对 L2 1e-5；optimizer 一、二阶矩另设上限）。loss 阈值仍为 5e-5，实测最大差异 2.03e-5。这里调整的是有对照证据的恢复验收规则，训练精度和超参数未改；不宣称逐位复现。

## 模型与基座

本次沿用同一 YOCO gRet 主干结构，构造新的约 400M 规格，而非引用官方已发布的 400M checkpoint。基座对应同规格 baseline 的 seed 42 随机初始化；正式训练对象仅为方法 A，从头预训练。该规模尚未另训 baseline/B 对照。

| 项目 | 配置 |
| --- | --- |
| 层数 | 24，前 12 层 self-decoder，后 12 层 cross-decoder |
| hidden dim / FFN | 1024 / 2816 |
| self heads / attention heads / KV heads | 4 / 16 / 16 |
| 词表及 tokenizer | 100288，原 YOCO cl100k wrapper |
| 同规格 baseline 参数量 | 400590848 |
| 方法 A 参数量 | 400590860，比 baseline 多 12 个参数 |
| 方法 A | 前 12 层分别产生候选 K/V，用 12 个全局 softmax 权重融合，K/V 共用配比，初始各为 1/12；后 12 层共享一次构造的 K/V |

[CPU 基座核查](evidence/400m-A-model-preflight.json)验证了 222 个公共 state tensors 与同规格 baseline 的初始化逐项一致，且只选最后来源时输出与 baseline 一致。训练从受检的 400M A 初始化开始，不加载 160M 权重。

## 保持原训练配方

训练仍使用 FineWeb Edu sample 10BT 的原固定 revision、token manifest 和数据顺序；FP16 mixed / FP32 parameters，seed 42，8 ranks × microbatch 4 × accumulation 4，长度 2048，每步 262144 tokens。训练 40000 步，共消费 10485760000 tokens。

AdamW 的 betas=0.9/0.98、eps=1e-8、weight decay=0.05、grad clip=1.0；峰值 LR=1.5e-4，warmup 375 步后按原 40000 步计划线性衰减。验证每 250 步一次，每次 2048 条序列；checkpoint 每 5000 步保存、全部保留。配置见 [400M A](../configs/yoco_400m_plan_A_8gpu.json)和[同规格 baseline 参考](../configs/yoco_400m_baseline_8gpu.json)。

## 加速与数值检查

使用同一初始化和相同的 40 步真实数据前缀比较；测速舍弃前 5 次初始化 update，取剩余 35 次中位数。下表的原执行路径关闭编译，使用原 PyTorch FP16 运算；数值比较均以该路径为参考。

| 执行方式 | 秒每步 | 同批梯度相对 L2 差异 | 是否采用 |
| --- | ---: | ---: | --- |
| 原执行方式，含激活重计算、普通 AdamW | 1.7567 | 0.0000610 | 作为历史执行参照 |
| 关闭重计算、融合 AdamW、无编译 | 1.4006 | 0 | 数值参照 |
| 整体模型和 loss 编译 | 0.8077 | 0.02473 | 未通过数值标准 |
| 整体编译并保留 FP16 中间舍入 | 0.8070 | 0.02473 | 未通过数值标准 |
| 按 decoder layer 编译，保留 FP16 舍入，loss 走原路径 | **0.9108** | **0.0001527** | **采用** |

采用方案比未编译数值参照快约 **1.54×**，比含重计算的原执行方式快约 **1.93×**。40 步 loss 最大差异为 **0.00001335**，同批梯度余弦为 **0.9999999883**，rank0 的 PyTorch 峰值已分配显存约 **28.4 GiB**。完整证据与采用阈值见[数值和性能对照](evidence/400m-A-acceleration-check.json)。

按层编译保留共享 K/V 构造及跨层梯度汇总的原 autograd 路径。`emulate_precision_casts` 让编译器保留原 FP16 算子间的 downcast/upcast，参数和激活 dtype 不变。单独开启这个选项不足以让整体编译通过检查，因此没有采用更快但数值差异较大的整体编译方案。该实验没有将差异归因于某个具体 PyTorch 缺陷。

```json
"execution": {
  "checkpoint_activations": false,
  "fused_adamw": true,
  "compile": true,
  "allow_tf32": true,
  "compile_loss": false,
  "emulate_precision_casts": true,
  "compile_scope": "layers"
}
```

恢复检查比较了 `0→2→4`、`0→4` 和一次重复的连续训练。八个 rank 的 RNG 和 scaler 一致；恢复与连续训练的参数相对 L2 差异约 **1.94e-7**，处于重复运行本身的波动量级，见[恢复检查](evidence/400m-A-resume-check.json)。数值容差通过不等于保证最终指标逐位相同，完整训练继续记录原始 loss、grad norm、scale 和 checkpoint。

## 本次运行与复用

当前 run 为 `runs/yoco-400m-A-8gpu-repaired-20261106`，旧失败 run `runs/yoco-400m-A-8gpu-full-20261105T0310Z` 原样保留。tmux 为 `yoco-400m-A-8gpu`，控制器为 [run_400m_A.py](../tools/run_400m_A.py)。修复代码和配置固定在新 run 的 `release/` 中，`recovery-source.json` 记录第 10000 步来源 checksum 与训练器修复前后 hash。先从该来源完成四步恢复核对，再从受检第 10004 步继续正式训练；模型、optimizer、scaler、八卡 RNG 和原 LR schedule 接续保存状态。

[修复版启动快照](evidence/400m-A-repair-launch.json)逐项核对了模型、训练、数据、方法 A、执行选项和 seed 与原 run 一致，并记录发布时的实际训练步数；该文件不是实时状态，也不包含最终训练或下游结果。

本次恢复入口如下；已在运行时只查状态，不重复执行：

```bash
python tools/run_400m_A.py \
  --run-dir runs/yoco-400m-A-8gpu-repaired-20261106 \
  --recover-from runs/yoco-400m-A-8gpu-full-20261105T0310Z
```

新结果只接续旧 run 的前 10000 步及修复版后续，排除旧失败尾段和诊断重放。第 5000、10000 步正式 checkpoint 仍在旧目录；后续每 5000 步在新目录保存并全部保留。

同一作业恢复时使用原控制器与相同 run 路径；控制器会检查活动记录和锁，避免重复运行。实时查看：

```bash
cat runs/setup/400m-A-active.json
tail -n 5 runs/yoco-400m-A-8gpu-repaired-20261106/rank0.jsonl
```

在新机器上先准备同一数据和环境，按实际位置修改 manifest 路径，再生成受检初始化：

```bash
source scripts/activate_local.sh
python tools/preflight_400m_A.py \
  --output runs/setup/yoco-400m-A-preflight.json \
  --initialization runs/setup/yoco-400m-A-initialization.pt
```

新的独立实验可采用[通用八卡启动方式](BASELINE_A100_ACCELERATION.md#在本机启动新的训练)，将配置改为 `configs/yoco_400m_plan_A_8gpu.json`，使用新的 run 目录，并追加上面的 `--init-model` 路径。`compile_scope=layers` 使用原模块内的编译 forward，不改 checkpoint 参数名称。跨机器先复做性能和数值短测，不直接套用本机耗时。

训练完成后，控制器会记录收敛检查，再按原锁执行八项完整下游评测。`PIPELINE_SUCCESS` 表示训练和评测都完成；收敛情况另见 `training-review.json`，不将进程正常退出当作没有数值波动的证明。
