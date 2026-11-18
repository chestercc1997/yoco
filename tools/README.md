# 复现工具

| 用途 | 入口与输入 |
| --- | --- |
| 准备新 baseline 数据 | [prepare_data.py](prepare_data.py) 接受显式配置。[yoco_samples.py](yoco_samples.py) 实现 Appendix D 配置使用的样本边界。[verify_data.py](verify_data.py) 校验数据版本、切块、分片大小和 SHA-256。 |
| 复核 400M 前向溢出 | [replay_400m_failure.py](replay_400m_failure.py) 读取原 release 和 checkpoint。[compare_400m_gradients.py](compare_400m_gradients.py) 对照梯度。[数值审计](../docs/analysis/YOCO_BASELINE_AUDIT.md) 给出已复现的现场和适用范围。 |
| 复核 830M 前向溢出 | [check_gret_830m_repair.py](check_gret_830m_repair.py) 检查早期现场。[check_gret_9793.py](check_gret_9793.py) 定位状态写入，[check_gret_9793_update.py](check_gret_9793_update.py) 回放该步的八卡批次。 |
| 核对恢复 | [check_400m_restore.py](check_400m_restore.py) 在前向前检查各 rank 的恢复状态。[830M 队列记录](../docs/history/BASELINE_830M_QUEUE.md) 给出对应 run 和恢复工具。 |
| 重新生成对比结果 | [export_model_comparison.py](export_model_comparison.py) 汇总九组历史实验。其他规模的输入、输出及评测记录见[报告索引](../docs/README.md)。 |

`setup_local.py` 和按规模或方案命名的 `preflight_*`、`run_*`、`watch_*`、`export_*` 工具与各自的历史实验配套。新 baseline 使用[六份独立配置和通用训练器](../README.md)。
