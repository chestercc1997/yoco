# 400M gRet baseline：队列与完成记录

用户于2026-11-09要求把400M gated retention baseline排在400M方法B之后。已于北京时间2026-11-10 00:27完成40000步、八项完整下游及自动汇总，队列已正常退出，禁止重复启动。最终valid loss3.005297899、宏平均47.7946606%，八份每5000步完整checkpoint全部保留；0次checkpoint回退。见[三组完整结果](../evaluation/gret_400m_3way_20261108T194918Z/README.md)与[九组图表](../evaluation/model_comparison_20261111/README.md)。以下保留原排队机制和复用流程。

## 自动接续条件

原队列在tmux `yoco-gret-400m-queue`中每30秒检查一次，完成后已退出。只有B的根目录和`B/`目录均存在`PIPELINE_SUCCESS`、40000步八卡训练及八项完整评测汇总通过、自动报告导出完成，且八张GPU的显存占用均小于1000MiB时，才启动baseline监督器。不终止B或其他GPU进程。

- B依赖：`runs/yoco-gret-400m-B-8gpu-20261108T073237Z`。
- baseline目录：`runs/yoco-gret-400m-baseline-8gpu-20261108T194918Z`。
- 本机持久队列：`runs/setup/gret-400m-training-queue.json`。
- 队列日志：`runs/setup/gret-400m-queue.log`。
- [排队证据](evidence/gret-400m-baseline-queue-20261109.json)为有时间戳的历史快照，实时进度以本机队列和run文件为准。

进程锁和PID启动时间用于避免重复启动或误认复用的PID。队列重启时会识别已存在的对应监督器；监督器异常退出则记录`needs_diagnosis`，不无限重启。队列依靠本机tmux存活，机器重启后需重新运行队列命令。

## 原配方与初始化

[新配置](../configs/yoco_400m_gret_baseline_8gpu.json)从B的冻结配置复制，仅删除`memory_fusion`，并更新实验名称和授权状态；不覆盖历史reference-only配置。model/training/data/seed/tied_embeddings/execution与A最终健康配置及B冻结配置逐项相等。

| 项目 | baseline设置 |
| --- | --- |
| 结构 | YOCO gated retention，24层，dim1024，FFN2816，self heads4，attention/KV heads16，无SWA窗口 |
| 参数量 | 400590848 |
| 数据 | 原固定FineWeb-Edu sample-10BT manifest/revision、tokenizer及顺序 |
| 训练预算 | 40000步，10485760000 tokens，seed42 |
| batch | 8×A100，microbatch4，累积4，序列2048，全局262144 tokens/步 |
| 优化器 | 原AdamW，LR0.00015，warmup375，线性衰减到0；其余参数保留 |
| 精度与执行 | FP16 mixed / FP32参数，原AMP策略，fused AdamW，compile关闭 |
| 验证/保存 | 每250步验证原2048序列；每5000步完整checkpoint，全部保留 |

初始化来自A/B同一随机基座，**不使用训练后的A/B权重**。CPU核验确认222个公共张量逐位相同，参数量正确，旧A冻结代码移除融合后的baseline初始输出差异为0、因果性差异为0，完整模型梯度有限。[初始化证据](evidence/gret-400m-baseline-preflight-20261109.json)保留初始化和修复代码SHA。

CPU准备和不可变release冻结、真实数据八卡0→2→4、连续与独立重复、前向前模型/optimizer/scaler/八卡RNG按位恢复及eval smoke均已通过；之后正式训练至40000步。原阈值未放宽，受检的4步已计入正式预算。

## 恢复与结果

`tools/watch_gret_400m_baseline.py`沿用已有有界自动恢复机制：监控八卡非有限前向、持续验证回退、停滞及worker退出；保留失败现场，校验最近健康完整checkpoint后重放相同数据，每组最多两次自动恢复。正常AMP拒绝尝试及单batch噪声不触发回滚。配置、冻结代码和数据manifest哈希不允许变化，不改LR、精度、batch或其他training setting。

训练后运行八项完整零样本评测（每组21465条文档），然后`tools/export_gret_400m_baseline.py`汇总baseline/A/B三个完整run、全部24份每5000步checkpoint、训练/验证CSV、逐任务分数和七组PNG/PDF/SVG。输出目录`evaluation/gret_400m_3way_20261108T194918Z`。A保留历史编译/修复前缀，baseline/B全程使用既有修复后的native路径；耗时不能只归因于融合结构差异。单seed差异仅作描述。

自动导出成功后队列保留终态`pipeline_complete_review_pending`，它表示本地自动流程完成。人工渲染审阅与发布核验另存[最终证据](evidence/model-comparison-final-review-20261111.json)，不改写自动队列的历史状态。

## 本机查看与恢复队列

```bash
cat runs/setup/gret-400m-training-queue.json
tail -n 5 runs/setup/gret-400m-queue.log
cat runs/yoco-gret-400m-baseline-8gpu-20261108T194918Z/guard-state.json
```

最后一条仅在baseline监督器启动后存在。若队列进程已退出，可使用同一登记文件重启；文件锁阻止重复队列。不能直接启动baseline来绕过B依赖，控制器自身也检查依赖。

```bash
tmux new-session -d -s yoco-gret-400m-queue \
  'cd /root/yoco && /root/miniconda3/envs/yoco/bin/python -u tools/run_gret_400m_queue.py > runs/setup/gret-400m-queue.log 2>&1'
```
