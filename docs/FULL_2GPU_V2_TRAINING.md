# 160M 两卡 baseline / 动态 v2：完整训练

2026-11-01 用户在查看短程结果后明确要求“先跑完看看效果”。据此继续这组两卡配对训练，不再以短实验未显示收益作为本次不启动的条件。

| 项目 | baseline | 动态 v2 |
| --- | --- | --- |
| 作业 | **788651** | **788652** |
| GPU | 单节点 2×H100 | 单节点 2×H100 |
| 起点 | 对应两卡 pilot 的第 1000 步 | 对应两卡 pilot 的第 1000 步 |
| 终点 | 40000 步 | 40000 步 |
| 原模型/配置 | YOCO `last` | token 融合，原初始权重 5%/5%/90% |
| Slurm 时限 | 24 小时 | 24 小时 |

本轮不重新初始化权重、不重置 optimizer 或 LR；保留原 model、optimizer、GradScaler、各 rank RNG 和原始 40000-step schedule。每组最终累计 10,485,760,000 tokens，其中前 262,144,000 tokens 已在 pilot 完成。根据此前速度估计剩余训练约 16–20 小时，实际受节点负载与保存耗时影响，排队时间另计。

## 续训与历史保留

模型 runtime 源码继续使用原两份不可变快照。恢复前已核对完整源码哈希、配置 fingerprint、checkpoint SHA256、world size=2、step=1000、optimizer 所有参数状态的 step=1000，以及 scaler/RNG 状态存在。GPU 作业开始时再次执行容器内 preflight 和 checkpoint 校验。

新 run 目录与旧 pilot 分开。第 1000 步 `.pt` 通过同文件系统硬链接引用；元数据复制并记录 `resume_origin.json`。训练器只读取旧检查点，随后保存新的文件；新目录的滚动删除只删除链接，不影响 pilot 原件。原短实验日志和已完成验收继续保留。

新的 HPC3 run 目录均在 `/data/share/jzhou945/cchen099/yoco/` 下：

```text
runs/yoco-160m-baseline-2gpu-full-20261101-v2
runs/yoco-160m-fusion-token-2gpu-full-20261101-v2
```

每 250 步按原配置验证并保存检查点，根目录保留最近三份；另将 10k/20k/30k/40k checkpoint 保留于各 run 的 `milestones/`。两组采用一致的 10k/20k/30k/40k 分段恢复边界，动态版在这些节点额外保存路由诊断。中间 `stage_complete.json` 只表示该段完成；必须达到 `step=40000`、`formal_complete=true` 并生成 `full_complete.json` 才算整轮完成。

## 入口与监测

- [续训脚本](../scripts/full_2gpu_inside.sh)
- [部署与提交工具](../tools/deploy_full_2gpu.py)
- 当前提交记录：`runs/setup/full-v2-active-jobs.json`
- 后台状态：`runs/setup/full-v2-watcher.json`

启动沿用已修复的标准 `.sif` 调用，通过容器内 `bash -c` 传递代码目录参数，避免集群封装器将独立目录参数误识别为目录式镜像。没有改动集群封装器。

后台监测每 60 秒核查作业，最多等待 120 小时；两组正常完成后自动进行相同固定样本的 CPU 梯度诊断，回传不含 `.pt` 的日志及元数据，核对 SHA256，并生成：

```text
docs/evidence/full-v2-convergence-review.json
docs/evidence/full-v2-convergence-review.md
```

报告将合并原 pilot 1–1000 步与续训 1001–40000 步，检查步数连续、验证点完整、训练配方和数据身份一致。比较相同 tokens 的验证曲线、同 loss 的采样区间和训练 GPU 时间；GPU 时间不替代 FLOPs，单 seed 也不构成统计显著性或梯度机制因果证明。若作业失败/超时，监测会明确记录，并保留最近 checkpoint 供后续恢复，不假装已完成。
