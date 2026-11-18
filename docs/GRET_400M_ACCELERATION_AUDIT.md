# 400M gRet 的 FP16 溢出排查

**结论（2026-11-18）**　400M 有故障快照的第 24,935 次更新属于方法 A。旧 gRet 后端在第一层 RMSNorm 前将跨块乘积写回 FP16，先产生 Inf，再使前向 loss 变成 NaN。用同一份权重和 token，关掉编译、恢复激活重算、完全不调用 AdamW，照样复现。当前后端通过这份快照的前向和反向。400M 纯 baseline 则在已修复跨块计算的实现上完成了 40,000 步。

## 对照设计

1. 按运行核对 160M、400M 的模型版本、优化器、编译和激活重算设置，区分原始 baseline、纯 baseline 和 A/B 架构实验。
2. 分别核查前向张量、交叉熵、AMP 反缩放后的梯度，以及优化器实际接受的更新。单次被拒绝的反向不能计为一次损坏的权重更新。
3. 用 400M 故障快照对照旧实现和修复实现，检查关闭加速能否消除第一个非有限值，验证生产路径的 FP16 输出和梯度。
4. 沿用既定训练精度和配方，记录证据适用范围及可重跑的回归检查。

## 加速前后实际发生了什么

下表按原始 rank 0 正式训练日志的区段统计。一次八卡同步拒绝只计一次。梯度列是已接受更新的最大**裁剪前**范数。A 的各行有回退重叠，不能直接相加。旧 trainer 遇到非有限 loss 会立即退出，早期没有单独的前向事件日志。

| 模型、运行区段 | 层编译 / 激活重算 / fused AdamW | AMP 梯度拒绝 | 最大有限梯度范数 | 前向与收敛记录 |
| --- | --- | ---: | ---: | --- |
| 160M 原 baseline，101–40000 | 关 / 开 / 关 | 20 | 991.31 | 跑完，约 21354 步起持续退化，最终 valid loss 3.408906 |
| 160M 加速 baseline，101–40000 | 开（亦编译 loss） / 关 / 开 | 20 | 133.98 | 跑完，最终 3.177219 |
| 400M A，首轮，5–14310 | 开 / 关 / 开 | 6 | 39.95 | 尝试 14311 前向失败，从 10000 重放未复现 |
| 400M A，修 scaler 后，10005–24934 | 开 / 关 / 开 | 8 | 102.09 | 尝试 24935，rank 0、microbatch 1，loss 为 NaN |
| 400M A，修 retention 后，20005–40000 | 关 / 关 / 开 | 9 | 33.67 | 该区段没有非有限前向，最终 2.918184 |
| 400M 纯 baseline，5–40000 | 关 / 关 / 开 | 12 | 1348.63 | 没有非有限前向，最终 3.005298 |
| 400M B，5–40000 | 关 / 关 / 开 | 11 | 18.81 | 没有非有限前向，最终 2.913688 |

160M 两组从同一 100 步来源接续，前 100 步 pilot 没有记录 AMP 拒绝。加速前已经出现梯度拒绝和有限尖峰，但 160M 首次退化缺少即时快照，不能与 400M 的跨块溢出合并归因。[160M 原运行](evidence/baseline-superseded-20261104.json) · [160M 加速核验](evidence/baseline-execution-acceleration.json) · [160M 训练曲线](BASELINE_A100_ACCELERATION.md)

400M 的早期纯 baseline 配置标为 reference；未找到使用旧后端、全关加速且完整训练 40k 的 400M 纯 baseline。最接近的旧执行方式对照是 **400M A 的 40 步探针**。重算开、普通 AdamW、eager 时，与重算关、fused AdamW、eager 相比，最大 loss 差 0.00002670，梯度相对 L2 差 0.00610%。它覆盖训练初期。中途归因依赖同一故障快照。[400M 加速探针](evidence/400m-A-acceleration-check.json) · [400M 三组完整结果](../evaluation/gret_400m_3way_20261108T194918Z/README.md)

## 故障快照上的前向和梯度

本次在 A100 上加载第 24934 步权重和失败尝试保存的输入、标签、loss scale 512。四种情况均使用同一快照（SHA-256 见 JSON），运行 eager 前向，无优化器更新。梯度为此 microbatch 对原累积步的贡献，已除去 loss scale，未做跨卡平均或裁剪。

| gRet 实现 | 激活重算 | 第一层 RMSNorm 输入非有限数 | logits 非有限数 | loss | 反向 |
| --- | --- | ---: | ---: | ---: | --- |
| 冻结旧版 | 关 | 94 | 51,347,456 | NaN | 前向已失败 |
| 冻结旧版 | 开 | 94 | 51,347,456 | NaN | 前向已失败 |
| 当前版 | 关 | 0 | 0 | 2.970219612 | 222 个梯度张量全部有限，范数 0.417335 |
| 当前版 | 开 | 0 | 0 | 2.970219612 | 222 个梯度张量全部有限，范数 0.417331 |

逐算子记录显示 Q/K/V、门、块内结果和递推状态均有限。旧跨块矩阵乘法输出有 69 个值溢出，块内项与跨块项相加后共 94 个值非有限。此时尚未执行 loss 缩放、反向、裁剪或当步优化器更新。方法 A 的 K/V 融合也在第一层故障点之后。此前的训练轨迹可以影响权重如何到达这一状态。[逐算子证据](evidence/400m-A-retention-fix/retention-operation-check.json) · [本次旧版回放](evidence/400m-A-retention-fix/replay-old-checkpointing.json) · [本次当前版回放](evidence/400m-A-retention-fix/replay-current-checkpointing.json)

**生产修复已经在代码中。** 跨块乘法和最终求和沿用现有 FP32 状态累积类型，经过原 RMSNorm 后才转回 FP16。这样避免了归一化前超出 FP16 最大有限值 65504 的结果被提前截断。内部算术边界确有调整；训练仍为 FP16 mixed、FP32 参数，输出 logits 为 FP16。本次新增了覆盖跨块溢出、归一化和反向的回归测试，训练代码和配置本次没有再改。[后端](../runtime/yoco_official/backend.py) · [归一化位置](../runtime/yoco_official/gate_retention.py) · [回归测试](../tests/test_gret_400m_cross_overflow.py)

**编译的影响可单独测出。** 旧代码的 eager 与编译执行在故障快照上均产生 NaN。修复后的同一完整更新（128 条样本）eager FP16 loss 为 2.989858，诊断 FP32 参考为 2.989850，梯度范数分别为 0.326817 和 0.347322。原生 FP16 与 FP32 参考梯度相对 L2 差约 30.8%，两者均有限。修复后编译与 eager FP16 的梯度相对 L2 差 **1.1076%**，超过原有 **0.5%** 接受线，因此 400M 后续正式运行使用 eager。[旧版编译对照](evidence/400m-A-retention-fix/before.json) · [修复后完整梯度](evidence/400m-A-retention-fix/fixed-full-gradient.json) · [修复与执行配置](evidence/400m-A-retention-fix/model-repair.json)

## 单独验证 AdamW

AdamW 始终开启。为检查以前一次更新为起点的影响，本次取第 20000 步健康完整 checkpoint，其 SHA 与恢复记录一致。在当前 eager 后端，用原数据顺序的 128 条样本、原缩放和裁剪得到一份公共梯度（裁剪前范数 0.285423）。从相同参数、AdamW 一阶和二阶状态及计数分别执行普通和 fused AdamW 的第 20001 步，随后用相同的下一个 microbatch 测 loss。

| 观测 | 数值 |
| --- | ---: |
| 普通 / fused 更新后下个 microbatch 的 loss | 3.2000544 / 3.2001002 |
| 更新后参数最大绝对差 | 1.19 × 10⁻⁷ |
| 一阶 / 二阶矩最大绝对差 | 5.82 × 10⁻¹¹ / 9.09 × 10⁻¹³ |
| 非有限参数或优化器状态 | 两组均为 0 |

单卡诊断按原样本集合累积，归约顺序与八卡 DDP 有别。这个对照隔离了同一梯度的一次 AdamW 更新，无法量化数万步累积舍入差异。结合故障快照在优化器调用前已失败的结果，没有证据支持通过关闭 fused AdamW 解决第 24935 次前向溢出。[优化器对照结果](evidence/400m-A-retention-fix/adamw-update-20001.json) · [对照代码](../tools/compare_400m_adamw.py)

## 训练时怎样处理

- **前向 loss 非有限**时，保留实际 rank、microbatch、权重、输入、标签和 scaler，停止本次更新。降低 loss scale 不能修复已经溢出的前向。历史 400M A 用完整 20000 步 checkpoint 恢复，同序重跑并通过故障位置，最终到达 40000 步。[八卡恢复核验](evidence/400m-A-retention-fix/exact-restore.json) · [400M A 完成记录](evidence/400m-A-completed-20261107.json)
- **前向有限、梯度非有限**时，既有 AMP 策略拒绝该次更新，下调 scale 并重试同一批数据。另一个已修缺陷是手动回退 scale 时没有清零成功步计数，可能让 scale 过早回升。纯 baseline 的 12 次 AMP 拒绝和最大有限范数 1348.63 是两个独立指标。第 14311 次前向失败的直接原因仍未知。[AMP 计数回归](evidence/400m-A-amp-backoff-check.json) · [14311 重放记录](evidence/400m-A-failure-review.json)
- **适用范围。** 本次验收覆盖已捕获的 400M 现场。另一个 830M 运行在第 9793 次尝试出现非有限前向，两次有界回退后已停止，保留 5004 步完整 checkpoint。830M 此次的第一个异常观测点也是第一层 RMSNorm 输入，单凭该位置不能判断是哪个上游运算。它没有参加本次 400M 训练或诊断更新。

## 复核命令

在仓库根目录、含训练依赖的环境中执行。CPU 回归只需仓库代码。两项 GPU 回放还需要训练机本地保存的原始快照、冻结 release；优化器对照另需健康 checkpoint 和原数据分片。以下输出只写入临时目录。

```bash
python -m unittest discover -s tests -p 'test_gret_400m_cross_overflow.py'
run=runs/yoco-400m-A-8gpu-repaired-20261106
snapshot="$run/nonfinite_forward_00024935_rank0.pt"
python tools/replay_400m_acceleration.py --runtime-root "$run/release" --snapshot "$snapshot" --expect reproduce --output /tmp/400m-old.json
python tools/replay_400m_acceleration.py --runtime-root . --snapshot "$snapshot" --expect fixed --output /tmp/400m-current.json
python tools/compare_400m_adamw.py --checkpoint "$run/checkpoint_00020000.pt" --output /tmp/400m-adamw.json
```
