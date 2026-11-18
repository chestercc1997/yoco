# 数据准备与实验入口

从仓库根目录运行。新 baseline 的六份配置和八卡训练示例见[仓库首页](../README.md)。

| 用途 | 入口 |
| --- | --- |
| 只准备训练数据 | `python tools/prepare_data.py --config configs/appendix_d/400m_gret.json`。已有原始分片时加 `--offline`。 |
| 验证六份配置共用的训练数据 | `python tools/verify_data.py --config configs/appendix_d/*.json` |
| 在本机下载并准备训练和下游数据 | `bash scripts/prepare_local_data.sh`。默认准备 Appendix D 样本。 |
| 合成数据检查及正式训练 | `runtime.train` 接收显式配置和运行目录。合成检查使用单独目录、`--smoke` 和 `--stop-after`。 |
| 运行模型、数据和恢复测试 | [测试说明](../tests/README.md) |

## 历史实验

四卡 A/B、两卡 pilot、旧 32 卡基准和 Slurm 文件对应各自的集群与冻结配置。旧四卡 A/B 的数据准备入口为 `bash scripts/prepare_local_data.sh --historical-ab`。本文档涉及的 830M gRet 实验处于暂停状态，其恢复依据见[数值审计](../docs/YOCO_BASELINE_AUDIT.md)。

早期一次性远端部署、重试和传输工具可按原版本查阅。[历史工具快照](https://github.com/chestercc1997/yoco/tree/ea889a5892034eab5ee37f70cb279dc171ca5371/tools) · [历史实验索引](../docs/README.md)
