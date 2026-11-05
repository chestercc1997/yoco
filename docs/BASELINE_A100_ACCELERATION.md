# A100 八卡 baseline 训练加速与复用

本次 YOCO 160M baseline 在 8×A100 80GB PCIe 上完成 40000 步，正式续训约 **3 小时 58 分**，加上八项并行下游评测共 **4 小时 8 分**。训练仍使用与 A/B 发布设置相同的 **FP16 混合精度和 FP32 参数**；加速来自执行优化。最终验证 loss 为 **3.177219**，八项宏平均为 **45.6268%**。

本文的整体编译配置对应已完成的 **160M baseline**。新的 **400M 方法 A** 已单独做数值核对并选择分层编译，配置见 [400M A 训练记录](PLAN_A_400M_8GPU.md)；不要将 160M 的编译结果直接当作更大模型的数值验证。

配置见 [yoco_160m_baseline_8gpu_matched.json](../configs/yoco_160m_baseline_8gpu_matched.json)，逐项测量与恢复检查见[加速证据 JSON](evidence/baseline-execution-acceleration.json)，完整结果见[三组对比报告](../evaluation/ab_baseline_20261104/README.md)。

## 加速来自哪里

同一模型、真实数据、八卡 batch 和 FP16 配置下，从健康的第 100 步开始分别测试 16 次 update，舍弃前 5 次初始化，统计其余 11 次的中位耗时。

| 执行设置 | 秒每步 | 相对原实现 |
| --- | ---: | ---: |
| 原执行方式，开启激活重计算 | 0.7158 | 1.00× |
| 关闭激活重计算，并使用融合 AdamW | 0.5684 | 1.26× |
| 再使用 `torch.compile` 编译模型 | 0.3776 | 1.90× |
| 再编译 cross entropy loss | 0.3273 | 2.19× |

激活重计算通过在 backward 时重做部分 forward 来节省显存。本机显存足够，关闭它可以减少重复计算。融合 AdamW 合并优化器操作；模型和 loss 编译减少部分中间张量读写与算子启动开销。前两项一起测量，表中没有将它们的收益单独拆分。

TF32 加速开关同时开启，但没有单独测量其收益。主要矩阵运算仍由 FP16 autocast 执行，不能把全部提速归因于 TF32。首次编译约 34 秒，后续 update 稳定在约 0.33 秒；这是单独的初始化开销。

这些数字反映本机的执行优化。A/B 原运行使用不同硬件及执行方式，耗时差不能直接解释为架构效率差。

## 保持哪些训练设置

| 设置 | 数值 |
| --- | --- |
| 模型 | 原始 YOCO gRet，12 层，dim 768，FFN 2304，绑定输入输出 embedding，无 memory fusion |
| seed | 42 |
| 精度 | `fp16_mixed_fp32_params`，动态 GradScaler，原初始 scale 128 |
| batch | 8 ranks × microbatch 4 × accumulation 4 × sequence 2048 = 262144 tokens 每步 |
| 训练量 | 40000 updates，共 10485760000 tokens |
| AdamW | betas 0.9/0.98，eps 1e-8，weight decay 0.05，grad clip 1.0 |
| 学习率 | 1.5e-4，warmup 375 步，之后按原计划线性衰减到 0 |
| 验证 | 每 250 步，8 × 4 × 64 = 2048 条序列 |
| checkpoint | 每 5000 步，全部保留，共八份，含第 40000 步 |

保存频率和保留策略是用户指定的例外；A/B 发布配置的保存频率为 250 步。训练超参数、验证规模和精度均未因此改变。分词仍使用原 YOCO cl100k wrapper，复用已经校验的 FineWeb Edu token manifest。

加速项独立放在配置顶层，便于与训练配方分开核对：

```json
"execution": {
  "checkpoint_activations": false,
  "fused_adamw": true,
  "compile": true,
  "allow_tf32": true,
  "compile_loss": true
}
```

`runtime/train.py` 在加载旧 optimizer state 后重新设置融合执行标志，并将 step tensor 放到对应设备；模型参数、optimizer moments、scaler 和 RNG 继续从 checkpoint 恢复。

## 在本机启动新的训练

先确认数据 manifest 指向已准备的数据。当前配置使用本机 `/root/yoco/data/fineweb_edu_10bt/manifest.json`，换机器时应修改为新机器上的实际位置。每次新实验使用新的 run 目录；下列命令建议在 tmux 中执行。

```bash
cd /root/yoco
source scripts/activate_local.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="$PWD/data/cache/torchinductor"

python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m runtime.train \
  --config configs/yoco_160m_baseline_8gpu_matched.json \
  --run-dir runs/yoco-160m-baseline-accelerated-new
```

这条命令从 seed 42 的初始化开始训练。`NCCL_SOCKET_IFNAME=lo` 和 `GLOO_SOCKET_IFNAME=lo` 仅用于这里的**单机八卡**；本机曾因自动选择不可达接口导致 NCCL 启动失败。多机训练必须换成节点之间实际可达的接口。

如果只想先测速度，使用独立目录和 `--benchmark-updates`：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m runtime.train \
  --config configs/yoco_160m_baseline_8gpu_matched.json \
  --run-dir runs/yoco-160m-speed-check-new \
  --benchmark-updates 16
```

结果写入该目录的 `benchmark_summary.json`；该模式不做验证，也不写 checkpoint，不能作为完整训练完成证据。首步可能包含编译，比较速度应看预热后的中位数。

## 续训与恢复检查

同一个 run 中断后，使用**相同代码、相同配置和相同 run 目录**再次执行训练命令即可。trainer 会验证 checkpoint SHA256 和配置指纹，恢复模型、optimizer、scaler、八个 rank 的 RNG，并按已完成步数恢复数据顺序和学习率。

本次完整运行从原 FP16 pilot 的健康第 100 步续训，给上述命令追加了：

```bash
--resume-from runs/yoco-160m-baseline-8gpu-pilot-20261104T0109Z/checkpoint_00000100.pt
```

这个显式入口用于新目录中的受检恢复。trainer 核对模型、数据、卡数、optimizer 超参数、batch、验证和 schedule，记录保存策略及执行优化差异。不要拿后期已出现退化的旧 checkpoint 作为恢复源，也不要在续训中重置学习率或 scaler。

本次做了 `100→102→104` 的中断恢复、`100→104` 连续训练及一次重复连续训练。八个 rank 的 RNG 和 scaler 一致；恢复与连续训练的 loss 最大差异为 **0.00000191**，模型参数相对 L2 差异为 **7.68e-7**，重复连续训练本身为 **7.50e-7**。FP16 加速执行存在舍入与非逐位一致性，不能要求与历史运行权重逐位相同。

当前已完成 run 的代码与评测锁保存在其 `release/` 中，原目录保持不变。`tools/run_baseline_matched.py` 是本次实验的专用控制器，会检查活动记录以防重复启动；新实验可使用上面的通用 `runtime.train` 命令。训练完成后，按原任务锁执行下游评测；本次约 4 小时的总耗时包含八个 GPU 并行评测的结果。

## loss 与数值记录

本次最终验证 loss 为 3.177219，最好观测值为 3.170204，从运行最优值的最大回升约 0.00718，未出现旧 A100 run 在约 21354 步后的持续大幅退化。原始 training loss 与 A/B 的局部波动标准差均约 0.029，详见[未平滑曲线检查](evidence/baseline-training-loss-smoothness.json)。

日志仍记录了 20 次动态 scaler 溢出重试，以及个别梯度范数尖峰；training loss 平滑不等于完全没有梯度数值波动。本轮同时改变了多项执行选项，不能由一次重跑断言某个 PyTorch 版本或某一个内核就是旧异常的唯一原因。今后继续保留逐步 loss、grad norm、scale 和每 5000 步的 checkpoint。
