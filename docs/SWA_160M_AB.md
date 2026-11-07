# YOCO-SWA 160M：baseline、方法 A、方法 B

2026-11-07 用户授权在本机 8×A100 完整训练三组，并开启持续 Goal：每组 40000 步、八项完整下游测试，再导出原始曲线、图表与汇总。新实验为 **滑动窗口 self-decoder**；此前完成的 gRet160M 三组和 gRet400M A 原样保留。当前尚无最终 SWA 指标。

## 结构和共同基座

[YOCO 论文 §4.2](https://arxiv.org/html/2405.05254v2#S4.SS2)的 SWA 窗口是 1024。前六层使用因果窗口注意力，后六层仍通过共享的全局 K/V 读取整个前缀，并非整网只能看 1024 tokens。

三组共同为 dim768、12 层、FFN2304、12 个 self/attention/KV heads、词表100288、tied embeddings、seed42。保持本仓库 160M 档宽度后，SWA 移除了 retention gates，实际参数量如下；论文在跨架构参数匹配时会调整 FFN，这里没有调整，不能称与此前 gRet166.7M 完全等参数。

| 组别 | 共享 K/V 来源 | 参数量 |
| --- | --- | ---: |
| baseline | 第六层输出 | 163155456 |
| A / learned_kv | 六层 K/V，六个全局 softmax 权重 | 163155462 |
| B / conditional_kv | 六层 K/V，由来源 hidden states 生成逐 token 权重 | 163156230 |

三组从新的 SWA 随机初始化开始，不加载已训练的 gRet 权重。102 个公共 state tensors 按位一致；A/B 额外控制参数初始化为零，初始六来源等权。强制只选第六来源时，A/B 输出与 baseline 精确一致。A 的六个 logits 和 B 的 score/bias 都取得非零、有限梯度。

实现沿用 `runtime/yoco_official` 和现有 PyTorch SDPA 后端，使用显式因果窗口 mask；没有安装原生 flash-attn，也不据此宣称获得官方窗口内核的吞吐。新增 SWA 融合支持，并修复增量缓存按默认 `max_batch_size=0` 分配空 batch 的问题。训练无缓存路径的注意力公式不变。闭式窗口边界检查、未来 token 干扰检查，以及三组逐 token / 分块缓存与整段输出的对照均通过，见 [CPU 核验](evidence/swa-160m-preflight.json)。

## 训练设置

复用现有 FineWeb-Edu sample-10BT 固定 revision、分词 manifest 和 cl100k wrapper；这是本仓库固定数据复现实验，未声称取得微软原始预训练语料。所有三组配置的 data、training、model、seed、execution 完全一致，差别仅为 memory_fusion。

- 每组 8 卡 × microbatch4 × accumulation4 × sequence2048，每步262144 tokens。
- 每组40000步，消费10485760000 tokens；AdamW、LR1.5e-4、warmup375及线性衰减、betas0.9/0.98、eps1e-8、weight decay0.05、grad clip1.0沿用原值。
- **FP16 mixed / FP32 parameters**，保留原 AMP scale 策略与已修复的 overflow 计数回退。不改为 BF16 或全 FP32。
- 每250步验证2048条序列；每5000步保存、全部保留模型/optimizer/scaler/八卡RNG。
- fused AdamW 开启，activation recomputation 关闭，TF32 选项沿用原设置，compile 关闭；三组执行设置相同。

配置：[baseline](../configs/yoco_160m_swa_baseline_8gpu.json)、[A](../configs/yoco_160m_swa_A_8gpu.json)、[B](../configs/yoco_160m_swa_B_8gpu.json)。

## 执行和恢复

[控制器](../tools/run_swa_ab.py)使用独占锁；将模型代码、配置、评测锁、初始化与 hash 固定到 cohort 的 `release/`。先让三组分别通过真实数据八卡 `0→2→4` / 连续 `0→4` / 独立重复对照，以及恢复前模型、optimizer、scaler、八卡 RNG 按位核验和 eval smoke。核验通过后，依次运行 baseline→A→B；每组训练结束后八卡并行做八项完整下游，随后释放给下一组。前三组的受检前四步计入各自正式训练，没有缩短总预算。

当前 cohort：`runs/yoco-swa-160m-ab-8gpu-20261107`；tmux：`yoco-swa-160m-8gpu`。

baseline 已通过真实八卡恢复对照、恢复前按位核验及下游接口测试，A/B 核验由同一控制器继续执行。[baseline 恢复证据](evidence/swa-160m-baseline-resume-check.json)与[启动状态](evidence/swa-160m-launch.json)为发布时快照，不是最终结果。

```bash
cat runs/setup/swa-160m-active.json
cat runs/yoco-swa-160m-ab-8gpu-20261107/pipeline-status.json
tail -n 4 runs/yoco-swa-160m-ab-8gpu-20261107/baseline/rank0.jsonl
```

中断后，在确认没有控制器或残留 GPU 作业时，运行同一命令即可核验冻结快照，并从各组最新有效 checkpoint 恢复，保留完整训练状态和数据顺序；已经完成的组跳过，不能重复启动新 cohort。

```bash
python tools/run_swa_ab.py --run-dir runs/yoco-swa-160m-ab-8gpu-20261107
```

非有限前向会保存诊断现场并停止，不自动忽略 batch。`pipeline-status.json` 是实时状态，本文和 GitHub 证据是发布时快照。训练与评测全部完成后，每组及 cohort 均生成 `PIPELINE_SUCCESS`；Goal 还须完成收敛审查、三组图表/汇总和 GitHub 发布才算完成。

全部完成后用[导出器](../tools/export_swa_ab.py)重新核验三组最终 checkpoint SHA、每任务完整样本数、分数、相同文档/prompt/target hash 与相同评测设置，再输出 120000 个原始训练点、480 个定期验证点、八项得分和七张 PNG/PDF/SVG 图。导出器拒绝未完成的 cohort。

```bash
python tools/export_swa_ab.py \
  --run-dir runs/yoco-swa-160m-ab-8gpu-20261107 \
  --output evaluation/swa_160m_ab_20261107
```
