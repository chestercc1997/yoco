# 数据准备与实验入口

从仓库根目录运行。新 baseline 的六份配置和八卡训练示例见[仓库首页](../README.md)。

| 用途 | 入口 |
| --- | --- |
| 只准备训练数据 | `python tools/prepare_data.py --config configs/appendix_d/400m_gret.json`。已有原始分片时加 `--offline`。 |
| 验证六份配置共用的训练数据 | `python tools/verify_data.py --config configs/appendix_d/*.json` |
| 在本机下载并准备训练和下游数据 | `bash scripts/prepare_local_data.sh`。默认准备 Appendix D 样本。 |
| 合成数据检查及正式训练 | `runtime.train` 接收显式配置和运行目录。合成检查使用单独目录、`--smoke` 和 `--stop-after`。 |
| 运行模型、数据和恢复测试 | [测试说明](../tests/README.md) |

## 160M 八组训练

在单机八张空闲 GPU 上运行。已有锁定版本的原始 parquet 时，从仓库根目录执行下列命令。数据已准备完毕时可直接运行队列脚本。首次联网下载原始数据时，省略 `--offline`。[八组配置、训练预算与顺序](../docs/160m/README.md)

```bash
source scripts/activate_local.sh
python tools/prepare_data.py --offline --workers 14 --config configs/appendix_d/160m_swa.json
bash scripts/run_160m_raw_kv.sh
```

启动器校验八份配置的数据，按顺序独占八卡训练。首次启动从当前 Git 提交导出 `runtime/`、两组配置目录和数据验证器，将提交号写入 `runs/appendix-d-160m-raw-v1/release/REVISION`。同一运行目录再次启动会使用已有快照，跳过带有 `TRAIN_COMPLETE` 的组。已有训练日志但缺少完整 checkpoint 元数据时，脚本会停下并保留现场。各组的步数和验证 loss 见 `runs/appendix-d-160m-raw-v1/<组名>/rank0.jsonl`。[启动脚本](run_160m_raw_kv.sh)

## 400M 六组训练

同一份 packed FineWeb-Edu 数据就绪后，在单机八张空闲 GPU 上运行 `bash scripts/run_appendix_d_raw_kv.sh 400m`。队列顺序为 SWA baseline → gRet baseline → SWA A → gRet A → SWA B → gRet B，运行目录为 `runs/appendix-d-400m-raw-v1/`。

两条 baseline 使用 Appendix D 的 24 层、宽度 1024、FFN 3072 及独立输入输出权重。两种架构均按各自配置采用 FP16 混合精度、fused AdamW 和 eager 执行。A/B 沿用同架构的训练配方，各聚合前 12 层，初始权重均为 1/12。[Baseline 配置](../configs/appendix_d) · [raw A/B 配置](../configs/400m_raw_kv)

## 历史实验

四卡 A/B、两卡 pilot、旧 32 卡基准和 Slurm 文件对应各自的集群与冻结配置。旧四卡 A/B 的数据准备入口为 `bash scripts/prepare_local_data.sh --historical-ab`。本文档涉及的 830M gRet 实验处于暂停状态，其恢复依据见[数值审计](../docs/analysis/YOCO_BASELINE_AUDIT.md)。

早期一次性远端部署、重试和传输工具可按原版本查阅。[历史工具快照](https://github.com/chestercc1997/yoco/tree/ea889a5892034eab5ee37f70cb279dc171ca5371/tools) · [历史实验索引](../docs/README.md)
