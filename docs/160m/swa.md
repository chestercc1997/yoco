# SWA 的窗口、全局记忆与历史结果

固定版本的微软实现调用因果注意力，左右窗口为 `(window_size - 1, 0)`。本地 1024 窗口在位置 t 直接读取 max(0,t−1023)…t。堆叠局部层可以扩大有效感受野，后半部 cross-decoder 对两种架构都使用全局因果 K/V。gRet 通过带门的递归状态压缩前缀。论文 §4.2 的规模实验报告 gRet 整体优于 SWA；论文研究的是 baseline，跨层融合的相对收益是独立问题。[微软 SWA](https://github.com/microsoft/unilm/blob/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/YOCO/yoco/models/decoder/sliding_window_attention.py) · [论文 §4.2 与图 3](https://arxiv.org/html/2405.05254v2#S4.SS2)

本次用长于窗口的 1030-token 输入，将本地训练注意力与独立构造的稠密因果带状矩阵进行 FP64 前向和 Q/K/V 反向对照。历史预检另含窗口边界及增量缓存一致性测试。它们验证窗口计算；原论文使用的训练语料、验证集和逐步轨迹无法从本地 FineWeb 结果还原，因此官方绝对 loss 达标与否尚无同口径证据。[参考计算测试](../../tests/test_raw_kv.py) · [历史 SWA 预检](../evidence/swa-160m-preflight.json)

历史 160M baseline 的最终 valid loss 为 gRet 3.1772、SWA 3.1801，差 0.0029。方法 A 相对各自 baseline 的变化分别为 −0.0172、−0.0040。历史 400M 在第 10000 / 20000 / 30000 / 40000 步，SWA A 的相对 loss 差为 +0.0346 / +0.0153 / +0.0101 / +0.0084，同期 gRet A 为 −0.0549 / −0.0955 / −0.0973 / −0.0871。历史 gRet 400M baseline 的收敛波动会影响相对分数，且这些是旧配方的单 seed 结果。浅层局部表示的混合可能稀释较深表示，这是待检验假设。新实验将对照同 token 的验证轨迹、末段训练 loss、梯度与 AMP 事件，并比较已保存 checkpoint 的逐层系数、绝对值和与符号。[历史效率汇总](../analysis/VERTIKV_TRAINING_EFFICIENCY.md)

另外读取六份历史 400M 最终 checkpoint，在各自冻结代码上用相同的 96 条验证样本做推理干预。保留全部训练权重，仅把融合系数改成只取最后一层。SWA A/B 的 loss 从 2.9632/2.9736 上升到 4.1017/4.4683；gRet A/B 从 2.9292/2.9261 上升到 3.5891/3.7603。独立训练的 SWA/gRet baseline 在这批样本上分别为 2.9526/3.0151。可见两种架构学到的预测都依赖多层 K/V，SWA 的融合通路确实生效。训练完成后的读出干预无法单独确定跨架构收益差的原因。[原始结果与十二层平均系数](../evidence/400m-kv-last-source-ablation.json)
