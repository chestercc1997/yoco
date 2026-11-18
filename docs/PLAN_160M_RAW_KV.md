# 160M SWA 和 gRet 的 baseline、A、B、C 训练计划

每种 self-decoder 从 seed 42 新建四个模型，共八组。前六层产生六个候选来源，后六层共享一次构造的全局 K/V。两种 baseline 优先，以便同一架构内比较。运行目录为 `runs/appendix-d-160m-raw-v1/`。[配置](../configs/160m_raw_kv) · [基线](../configs/appendix_d) · [串行启动器](../scripts/run_160m_raw_kv.sh)

| 组 | 共享 K/V 的构造 | 初始化 |
| --- | --- | --- |
| baseline | 仅用第六层输出投影 K/V | 原始 baseline |
| A | 一组全局可训练系数直接乘各层 K/V，允许负值，系数和自由 | 六个系数各 1/6 |
| B | 每个 token、每个来源的 raw 系数为层偏置加共享向量与该来源 RMS 归一化状态的点积除以 √768，K/V 共用这组系数 | 打分向量为零，六个偏置各 1/6 |
| C | 为共享 K、V 各设一个可训练的深度查询，各自与六层 RMS 归一化状态点积、除以 √768，各自沿深度做 softmax。将两组权重分别乘各层的 K、V | 两个查询为零，两套路由均匀 |

A/B 直接使用参数或打分，训练中可以放大、相消。C 借鉴 Attention Residuals 的可学习查询与深度注意力，在 YOCO 的共享记忆入口做一次双查询读出。三种融合均保持查询流为第六层输出。A/B/C 在初始时刻有相同的混合输出。历史 A/B 的 softmax 语义保存在原模式中。[融合实现](../runtime/memory_fusion.py) · [Attention Residuals](https://arxiv.org/abs/2603.15031)

## SWA 实现核查与待验证的问题

固定版本的微软实现调用因果注意力，左右窗口为 `(window_size - 1, 0)`。本地 1024 窗口在位置 t 直接读取 max(0,t−1023)…t。堆叠局部层可以扩大有效感受野，后半部 cross-decoder 对两种架构都使用全局因果 K/V。gRet 通过带门的递归状态压缩前缀。论文 §4.2 的规模实验报告 gRet 整体优于 SWA；论文研究的是 baseline，跨层融合的相对收益是独立问题。[微软 SWA](https://github.com/microsoft/unilm/blob/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/YOCO/yoco/models/decoder/sliding_window_attention.py) · [论文 §4.2 与图 3](https://arxiv.org/html/2405.05254v2#S4.SS2)

本次用长于窗口的 1030-token 输入，将本地训练注意力与独立构造的稠密因果带状矩阵进行 FP64 前向和 Q/K/V 反向对照。历史预检另含窗口边界及增量缓存一致性测试。它们验证窗口计算；原论文使用的训练语料、验证集和逐步轨迹无法从本地 FineWeb 结果还原，因此官方绝对 loss 达标与否尚无同口径证据。[参考计算测试](../tests/test_raw_kv.py) · [历史 SWA 预检](evidence/swa-160m-preflight.json)

历史 160M baseline 的最终 valid loss 为 gRet 3.1772、SWA 3.1801，差 0.0029。方法 A 相对各自 baseline 的变化分别为 −0.0172、−0.0040。历史 400M 在第 10000 / 20000 / 30000 / 40000 步，SWA A 的相对 loss 差为 +0.0346 / +0.0153 / +0.0101 / +0.0084，同期 gRet A 为 −0.0549 / −0.0955 / −0.0973 / −0.0871。历史 gRet 400M baseline 的收敛波动会影响相对分数，且这些是旧配方的单 seed 结果。浅层局部表示的混合可能稀释较深表示，这是待检验假设。新实验将对照同 token 的验证轨迹、末段训练 loss、梯度与 AMP 事件，并比较已保存 checkpoint 的逐层系数、绝对值和与符号。[历史效率汇总](VERTIKV_TRAINING_EFFICIENCY.md)

另外读取六份历史 400M 最终 checkpoint，在各自冻结代码上用相同的 96 条验证样本做推理干预。保留全部训练权重，仅把融合系数改成只取最后一层。SWA A/B 的 loss 从 2.9632/2.9736 上升到 4.1017/4.4683；gRet A/B 从 2.9292/2.9261 上升到 3.5891/3.7603。独立训练的 SWA/gRet baseline 在这批样本上分别为 2.9526/3.0151。可见两种架构学到的预测都依赖多层 K/V，SWA 的融合通路确实生效。训练完成后的读出干预无法单独确定跨架构收益差的原因。[原始结果与十二层平均系数](evidence/400m-kv-last-source-ablation.json)

## 执行与验收

- 使用 Appendix D 的 12 层、宽 768、FFN 2304、独立输入输出权重。相同架构的四组共用主干初始化、数据、验证集和执行设置。FineWeb-Edu 固定版本按 YOCO 边界生成独立的 2049-token 记录，整库校验成功后开始正式更新。
- 每组 8×A100，4 条/卡 × 4 次梯度累积，序列长 2048，共 262144 tokens/步，目标 40000 步。保留原 FP16 mixed、FP32 参数、动态 loss scale、AdamW 及其学习率计划。使用既有 fused AdamW；SWA 原生执行，gRet 沿用该档既有编译设置。上轮收益很小或尚缺稳定性证明的候选均未加入。
- 启动顺序为 SWA baseline、gRet baseline，然后按 A、B、C 依次跑两个架构。八卡独占、串行运行。冻结代码和配置，每 250 步验证，每 5000 步保存完整 checkpoint。异常退出保留现场并停队，恢复前核对完整状态、数据顺序和配方。
- 新阶段以各组 rank 日志中的真实成功更新判定已开跑。每组到 40000 步并核验最终 checkpoint 后转入下一组。完成后按同 token、同架构比较四组；八组单 seed 结果支持该次运行的描述性结论。

启动前实际检查通过了 61 项自动化测试。八个全尺寸模型用同一批 4×2048 个位置在 A100 上做独立 FP16 前反向，所有参数梯度有限，每个新增打分参数的梯度非零。同架构四组的所有公共参数初始化 SHA-256 相同，A/B/C 初始 logits 逐元素相同。这次全尺寸检查采用 eager 执行；正式编译与 DDP 的表现以训练日志为准。
