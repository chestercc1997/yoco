# 400M 方法 A 八卡训练

2026 年 11 月 5 日，400M 方法 A 已在本机 8×A100 80GB PCIe 上开始完整训练，目标为 40000 步及八项下游评测。当前尚无最终指标；[启动时状态](evidence/400m-A-launch.json)是发布时的快照，实时状态在本机 `runs/setup/400m-A-active.json`。160M 已完成结果继续保存在[原三组报告](../evaluation/ab_baseline_20261104/README.md)，两种规模的结果分开记录。

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

本机 run 为 `runs/yoco-400m-A-8gpu-full-20261105T0310Z`，tmux 为 `yoco-400m-A-8gpu`，控制器为 [run_400m_A.py](../tools/run_400m_A.py)。代码和配置固定在该 run 的 `release/` 中，初始化、数值检查和恢复证据保存在 run 内；完整训练从已核对的第 4 步继续到 40000 步，不沿用短程测速试验的权重。

同一作业恢复时使用原控制器与相同 run 路径；控制器会检查活动记录和锁，避免重复运行。实时查看：

```bash
cat runs/setup/400m-A-active.json
tail -n 5 runs/yoco-400m-A-8gpu-full-20261105T0310Z/rank0.jsonl
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
