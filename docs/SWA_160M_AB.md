# YOCO-SWA 160M：baseline、方法 A、方法 B

2026-11-07 用户授权在本机 8×A100 完整训练三组，并开启持续 Goal：每组 40000 步、八项完整下游测试，再导出原始曲线、图表与汇总。新实验为 **滑动窗口 self-decoder**；此前完成的 gRet160M 三组和 gRet400M A 原样保留。

## 当前结果

baseline于北京时间2026-11-07 18:45完成40000步及八项完整零样本评测；方法A于2026-11-08 01:39完成相同完整流程。两组每5000步的各8份checkpoint均已核验，B已在同一cohort八卡接续训练，每5000步的五份checkpoint（至25000步）通过核验。三组对比和最终图表尚未完成，Goal保持进行。

| 组别 | 完整训练/下游 | 最终验证loss | 八项宏平均 |
| --- | --- | ---: | ---: |
| baseline | 40000步、8/8项完成 | 3.180143 | 46.3682% |
| A / learned_kv | 40000步、8/8项完成 | 3.176147 | 46.0355% |
| B / conditional_kv | 八卡训练中；25000步已核验 | 待完成 | 待完成 |

[baseline完成证据](evidence/swa-160m-baseline-completed-20261107.json)已独立核对21465条样本、逐样本得分均值、零样本/全量设置、数据锁、任务YAML和最终checkpoint来源。八项分别为ARC-E46.7172%、ARC-C24.3174%、BoolQ61.9878%、HellaSwag31.1193%、OpenBookQA28.8000%、PIQA62.6224%、WinoGrande49.8816%、SciQ65.5000%；BoolQ/WinoGrande使用acc，其余使用acc_norm。

[第40000步完整状态](evidence/swa-160m-baseline-40000-checkpoint.json)的405个模型/optimizer张量均有限，101组optimizer step均40000，配置及scaler超参数未改，八卡RNG完整；全部8份每5000步checkpoint的SHA均重新核验一致。40000个成功更新、160个定期验证点齐全，验证loss持续下降，最大回升为0；全程5次原AMP策略溢出重试、0次非有限前向、0次checkpoint回退。四组退避事件的八卡状态均同步，重试更新均以有限loss/grad norm完成，最终scaler growth tracker962符合最后退避后的成功更新数。所有数值为本次单seed结果。

[A完成证据](evidence/swa-160m-A-completed-20261108.json)独立核对了21465条样本、逐样本得分均值、全量零样本协议、任务YAML、最终checkpoint来源，以及与baseline逐条相同的doc/prompt/target hash。[A最终完整状态](evidence/swa-160m-A-40000-checkpoint.json)的409个模型/optimizer张量均有限、102组optimizer step均40000，冻结配置及optimizer/scaler超参数未改，八卡RNG完整保存；全部8份checkpoint的SHA一致。40000个成功更新和160个验证点齐全，验证loss最大回升为0；累计5次AMP重试、0次非有限前向、0次checkpoint回退。baseline和A从受检第4步之后的39996次正式续训更新/卡，其每一次记录的scale增长及退避均符合原策略，完整时序证据已写入各组第40000步报告。

A的验证loss较baseline低0.003996，但八项宏平均低0.3327个百分点。目前不据此宣称方法A带来整体提升；这是单seed描述性结果，B及三组最终对比仍待完成。

| 任务 | baseline (%) | A (%) | A−baseline (百分点) |
| --- | ---: | ---: | ---: |
| ARC-Easy | 46.7172 | 45.9596 | -0.7576 |
| ARC-Challenge | 24.3174 | 24.7440 | +0.4266 |
| BoolQ | 61.9878 | 57.2171 | -4.7706 |
| HellaSwag | 31.1193 | 31.0197 | -0.0996 |
| OpenBookQA | 28.8000 | 29.2000 | +0.4000 |
| PIQA | 62.6224 | 61.5887 | -1.0337 |
| WinoGrande | 49.8816 | 51.8548 | +1.9732 |
| SciQ | 65.5000 | 66.7000 | +1.2000 |
| 八项宏平均 | 46.3682 | 46.0355 | -0.3327 |

以下checkpoint记录为对应训练步的历史核验快照；各阶段“继续训练”的描述不改变上方两组已经完成的当前状态。

方法B的[第25000步checkpoint](evidence/swa-160m-B-25000-checkpoint.json)通过独立核验：413个模型/optimizer张量有限、103组optimizer step均25000，学习率符合原线性衰减计划，冻结配置及optimizer/scaler超参数保持原值，八卡RNG完整保存。前25000个成功更新和100个定期验证点齐全，验证loss=3.245944500，无验证loss回升、overflow重试、非有限前向或checkpoint回退。每5000步的五份正式checkpoint全部保留并重新核对SHA一致。八卡从受检第4步到25000步的每卡24996次正式更新，其记录的scale均符合原增长及退避策略，最终scale=524288、growth tracker=1000。原八卡B作业继续至40000，完整下游和三组最终图表尚未完成。

方法B的[第20000步checkpoint](evidence/swa-160m-B-20000-checkpoint.json)通过独立核验：413个模型/optimizer张量有限、103组optimizer step均20000，学习率符合原线性衰减计划，冻结配置及optimizer/scaler超参数保持原值，八卡RNG完整保存。前20000个成功更新和80个定期验证点齐全，验证loss=3.291630507，无验证loss回升、overflow重试、非有限前向或checkpoint回退。每5000步的四份正式checkpoint全部保留并重新核对SHA一致。八卡从受检第4步到20000步的每卡19996次正式更新，其记录的scale均符合原增长及退避策略，最终scale=131072、growth tracker=0。B仅完成自身训练计划的一半，原八卡作业继续至40000；完整下游和三组最终图表尚未完成。

方法B的[第15000步checkpoint](evidence/swa-160m-B-15000-checkpoint.json)通过独立核验：413个模型/optimizer张量有限、103组optimizer step均15000，冻结配置及optimizer/scaler超参数保持原值，八卡RNG完整保存。前15000个成功更新和60个定期验证点齐全，验证loss=3.358592510，无验证loss回升、overflow重试、非有限前向或checkpoint回退。每5000步的三份正式checkpoint全部保留并重新核对SHA一致。原八卡B作业继续至40000，完整下游和三组最终图表尚未完成。

方法B的[第10000步checkpoint](evidence/swa-160m-B-10000-checkpoint.json)通过独立核验：413个模型/optimizer张量有限、103组optimizer step均10000，学习率符合原线性衰减计划，冻结配置及optimizer/scaler超参数保持原值，八卡RNG完整保存。前10000个成功更新和40个定期验证点齐全，验证loss=3.474273205，无验证loss回升、overflow重试、非有限前向或checkpoint回退。5000/10000步两份正式checkpoint全部保留并重新核对SHA一致。B仅完成自身40000步计划的四分之一，后续训练、八项全量下游及三组图表仍待完成。

方法B的[第5000步checkpoint](evidence/swa-160m-B-5000-checkpoint.json)通过独立核验：413个模型/optimizer张量有限、103组optimizer step均5000，冻结配置及optimizer/scaler超参数保持原值，八卡RNG完整保存。前5000个成功更新和20个定期验证点齐全，该步验证loss=3.739674807，无验证loss回升、overflow重试、非有限前向或checkpoint回退，首份正式checkpoint的SHA复核一致。条件融合的768维score与6个layer_bias均从零初值学到有限非零值，score的L2范数3.82451、bias的L2范数0.52901；这是模型权重的正常学习，temperature=1及训练设置未改。B采用逐token条件权重，不能用单个全局softmax向量代表其路由。原八卡B作业继续至40000步，完整下游与三组图表尚未完成。

方法A的[第5000步checkpoint](evidence/swa-160m-A-5000-checkpoint.json)已独立核验：409个模型/optimizer张量有限，102组optimizer step均5000，训练设置与冻结release一致，scaler和八卡RNG完整。前5000个成功更新、20个定期验证点齐全，该步验证loss=3.737168550，无验证loss回升、overflow重试、非有限前向或checkpoint回退。六个融合logits已从零初始化学到非零值，对应第1–6层的softmax权重约为0.11704、0.12080、0.12306、0.14952、0.21756、0.27202，均有限、为正且归一化；这是模型权重的正常学习，temperature等训练设置未改。A继续训练至40000步，尚不能据此判断最终优劣。

方法A的[第10000步checkpoint](evidence/swa-160m-A-10000-checkpoint.json)同样通过独立核验：409个张量有限、102组optimizer step均10000，学习率符合原线性衰减计划，配置、scaler超参数和八卡RNG保持原设置。前10000个成功更新和40个定期验证点齐全，验证loss=3.474439144，无验证loss回升、overflow重试、非有限前向或checkpoint回退。5000/10000步两份正式checkpoint均保留并重新核对SHA一致。A仅完成自身40000步计划的四分之一，后续完整训练与下游继续执行。

方法A的[第15000步checkpoint](evidence/swa-160m-A-15000-checkpoint.json)通过独立核验：409个张量有限、102组optimizer step均15000，冻结配置、optimizer/scaler超参数和八卡RNG保持原设置。前15000个成功更新和60个定期验证点齐全，验证loss=3.359063148，无验证loss回升、overflow重试、非有限前向或checkpoint回退。5000/10000/15000步三份正式checkpoint全部保留并重新核对SHA一致。A继续训练至40000步，三组完整对比尚未完成。

方法A的[第20000步checkpoint](evidence/swa-160m-A-20000-checkpoint.json)通过独立核验：409个张量有限、102组optimizer step均20000，学习率符合原线性衰减计划，冻结配置及optimizer/scaler超参数保持原值，八卡RNG完整保存。前20000个成功更新和80个定期验证点齐全，验证loss=3.292289972，无验证loss回升、overflow重试、非有限前向或checkpoint回退。每5000步的四份正式checkpoint全部保留并重新核对SHA一致。A仅完成自身训练计划的一半，后半程、A/B全量下游和三组最终图表仍待完成。

方法A的[第25000步checkpoint](evidence/swa-160m-A-25000-checkpoint.json)通过独立核验：409个张量有限、102组optimizer step均25000，冻结配置及optimizer/scaler超参数保持原值，八卡RNG完整保存。前25000个成功更新和100个定期验证点齐全，验证loss=3.246068954，无验证loss回升、overflow重试、非有限前向或checkpoint回退。每5000步的五份正式checkpoint全部保留并重新核对SHA一致。原八卡A作业继续至40000，A/B完整训练、下游及三组最终图表尚未全部完成。

方法A的[第30000步checkpoint](evidence/swa-160m-A-30000-checkpoint.json)通过独立核验：409个张量有限、102组optimizer step均30000，冻结配置及optimizer/scaler超参数保持原值，八卡RNG完整保存。前30000个成功更新和120个定期验证点齐全，验证loss=3.213108063，无验证loss回升、无非有限前向或checkpoint回退；累计1次AMP梯度溢出重试，八卡退避及重试成功均已核验。保存的scale=1048576、growth tracker=435，符合27565步退避后2435次成功更新及原2000次增长周期。每5000步的六份正式checkpoint全部保留并重新核对SHA一致；原八卡A作业继续至40000，三组全量任务尚未完成。

方法A的[第35000步checkpoint](evidence/swa-160m-A-35000-checkpoint.json)通过独立核验：409个张量有限、102组optimizer step均35000，冻结配置及optimizer/scaler超参数保持原值，八卡RNG完整保存。前35000个成功更新和140个定期验证点齐全，验证loss=3.189705849，无验证loss回升、无非有限前向或checkpoint回退；累计3次AMP梯度溢出重试，27566、31758、34510步的八卡退避及重试成功均已核验。保存的scale=1048576、growth tracker=491，符合最后退避后的491次成功更新和原2000次增长周期。每5000步的七份正式checkpoint全部保留并重新核对SHA一致；A继续至40000，A/B完整训练、下游和三组最终图表尚未全部完成。

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

baseline、A、B **均已通过**真实八卡恢复对照、恢复前按位核验及下游接口测试；baseline 已于北京时间12:14进入完整训练。[baseline](evidence/swa-160m-baseline-resume-check.json)、[A](evidence/swa-160m-A-resume-check.json)、[B](evidence/swa-160m-B-resume-check.json)恢复证据与[启动状态](evidence/swa-160m-launch.json)为发布时快照，不是最终结果。

首个正式第5000步 checkpoint 已于北京时间13:01保存，随后完成[独立完整性核验](evidence/swa-160m-baseline-5000-checkpoint.json)：SHA一致，405个模型/optimizer张量有限，101个optimizer state的step均为5000，实际LR和betas/eps/weight decay符合原配置，参数仍为FP32、训练仍为FP16 mixed，scaler与八卡RNG完整。前5000个更新和20个定期验证点齐全，该步验证loss=3.736929893，迄此未记录NaN/Inf、overflow重试或验证loss回升。训练继续至40000；这是阶段检查，三组最终指标和下游仍待完整流程完成。

第10000步也已保存并完成[同样的独立核验](evidence/swa-160m-baseline-10000-checkpoint.json)：405个张量有限，101组optimizer step均为10000，学习率符合原线性衰减计划，scaler的growth/backoff/interval保持2.0/0.5/2000，配置与冻结release一致。前10000个更新和40个定期验证点齐全，验证loss=3.477878332，迄此没有NaN/Inf、overflow重试或验证loss回升。baseline仅完成自身计划的四分之一；A/B完整训练和全部下游仍待后续阶段，不将此作最终结果。

第15000步完成[独立核验](evidence/swa-160m-baseline-15000-checkpoint.json)：405个张量有限，101组optimizer step均为15000，配置、optimizer与scaler超参数保持原设置，八卡RNG完整。前15000个更新和60个定期验证点齐全，验证loss=3.362773657，迄此仍无NaN/Inf、overflow重试或验证loss回升。5000/10000/15000三份正式checkpoint均保留，训练继续至40000，三组全量训练与下游尚未完成。

baseline半程第20000步完成[独立核验](evidence/swa-160m-baseline-20000-checkpoint.json)：405个张量有限、101组optimizer step均20000，完整optimizer/scaler/八卡RNG和冻结配置一致。前20000个更新及80个定期验证点齐全，验证loss=3.295583487，迄此无NaN/Inf、overflow重试或验证loss回升。5000/10000/15000/20000四份正式checkpoint均保留；这仅是baseline的半程，baseline后半程、A/B及全部下游继续执行。

第25000步完成[独立核验](evidence/swa-160m-baseline-25000-checkpoint.json)：405个张量有限、101组optimizer step均25000，配置、optimizer/scaler超参数及八卡RNG保持原配方。前25000个更新与100个定期验证点齐全，验证loss=3.250399351，迄此无NaN/Inf、overflow重试或验证loss回升。每5000步的五份正式checkpoint均保留；baseline继续至40000，A/B完整训练与全部下游仍待后续阶段。

第30000步完成[独立核验](evidence/swa-160m-baseline-30000-checkpoint.json)：405个张量有限、101组optimizer step均30000，配置、optimizer/scaler超参数与八卡RNG保持原设置。前30000个更新及120个定期验证点齐全，验证loss=3.217280626，无验证loss回升、无非有限前向loss；累计1次AMP梯度溢出重试，未发生checkpoint回退。保存的scaler scale=1048576、growth tracker=1957，与28044重试成功后至30000的1957次成功更新及原2000次增长周期一致。六份正式checkpoint均保留，baseline继续至40000，三组全量任务尚未完成。

第35000步完成[独立核验](evidence/swa-160m-baseline-35000-checkpoint.json)：405个张量有限、101组optimizer step均35000，实际配置、optimizer/scaler超参数与冻结配方一致。前35000个更新及140个定期验证点齐全，验证loss=3.193810701，无验证loss回升、无非有限前向loss。累计4次AMP梯度溢出重试，分别发生在28044、31669及34623步（最后一步重试前退避两次）；八卡退避状态一致，三个更新均以有限loss和grad norm完成，未发生checkpoint回退。保存的scale=524288、growth tracker=378，与最后退避后的成功更新数一致。七份正式checkpoint均保留，baseline继续至40000，三组训练和全量下游尚未全部完成。

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

## 自动检测与恢复：训练设置完全锁定

按用户最新要求，[复现 skill](../.agents/skills/yoco-reproduction/SKILL.md)将“任何训练 setting 参数都不能改动”作为硬约束。已注册到本机全局 skill 发现目录，仓库保存可审阅版本。不能通过降低 LR、改精度/AMP 策略、改 batch/grad clip、缩减预算或验证、跳过失败 batch、清 optimizer/RNG/scaler 来隐藏故障。

实际训练首次AMP梯度溢出发生在成功完成28043步之后、尝试28044步时。[八卡现场核验](evidence/swa-160m-baseline-first-amp-backoff.json)确认：八个rank同步按原backoff因子0.5将scale从2097152退至1048576，growth tracker全部归零，同一28044步重试成功，loss=3.226188660、裁剪前grad norm=0.446501911均有限。配置与scaler超参数未改，没有checkpoint回退。相同步号及未改动的stateless sampler证明数据索引重用；未另外捕获token字节作比对。此处记录梯度溢出重试，前向loss未出现NaN/Inf；后续完整训练与下游仍需完成。

方法A首次AMP梯度溢出发生在完成27565步后、尝试27566步时。[A的八卡现场核验](evidence/swa-160m-A-first-amp-backoff.json)确认：八卡同步按原策略将scale从1048576降至524288，growth tracker全部归零；同一27566步重试成功，loss=3.237055779、裁剪前grad norm=0.434896529均有限。未改训练设置、没有checkpoint回退，前向loss始终有限。此后报告需保留A的overflow计数；第25000步证据中的0次仅覆盖该历史节点之前，不能据此宣称完整A作业从未溢出。相同数据索引的结论来自未改动的stateless sampler和相同步号，未另外捕获token字节作比对。

[外部监督器](../tools/watch_swa_training.py)已附着正在训练的 cohort，tmux `yoco-swa-160m-guard`；没有重启健康的训练作业。它每10秒检查日志：

- 任一卡的非有限前向、非有限验证 loss、worker 退出或训练停滞600秒，触发故障处理。
- 验证 loss 连续3次比历史最佳高出 `max(0.15, 5% × best)`，触发持续退化处理。单 batch spike、单个验证波动或正常 AMP overflow 不触发回退。这些值是监测策略，未写入或改变训练配置。
- 等待训练器完成故障快照，终止本 cohort 的进程树；校验最新健康 checkpoint 的 SHA/fingerprint、模型/optimizer 有限性、scaler 和八卡 RNG。从最近健康完整 checkpoint 重跑所有后续 batch；不足5000步时使用已验收的第4步完整状态，不额外改变正式保存频率。
- `owned-processes.json` 持续记录控制器及其 worker 的 PID/start-time；控制器意外退出后，仍会清理已记录的遗留 worker。信号发送前再次核对 start-time，避免 PID 重用影响其他进程。重载监督器继续沿用归属记录，不重启训练。
- 失败日志、异常 checkpoint 与现场均保存在各组 `recovery-attempts/`，正式曲线只保留恢复点之前的有效前缀及重跑结果。discarded suffix 的训练耗时单独导出，不能隐藏重跑代价。
- 每组最多自动恢复2次。再次失败会输出[只读 FP16/FP32 前向诊断](../tools/diagnose_training_failure.py)，标识首个非有限叶模块，由 active Goal 继续定位并验证兼容原 setting 的实现修复。没有通用算法能保证修好任意未知代码 bug；这里不会盲目改参数或无限重启，也不要求用户人工巡检。

每次启动/恢复前，`guard-contract.json` 校验冻结配置、源代码、数据 manifest 与根目录配置完全一致。不一致时拒绝启动。完整失败证据保留，旧 release 不作原位修改。损坏 checksum、非有限权重、持续/瞬时 loss、LR 偷改、日志回退、优化器/RNG 恢复、仅终止所属进程及控制器退出后清理遗留 worker 的8项检查已通过，见 `tests/test_training_guard.py`。

```bash
cat runs/yoco-swa-160m-ab-8gpu-20261107/guard-state.json
cat runs/yoco-swa-160m-ab-8gpu-20261107/guard-contract.json
# 仅在确认原监督器已停止后重启；同 cohort 独占锁会拒绝重复实例
python tools/watch_swa_training.py --run-dir runs/yoco-swa-160m-ab-8gpu-20261107
```

`monitoring` 表示已附着；`recoveries` 记录实际回退；`needs_goal_diagnosis` 表示有界恢复耗尽或评测/核验故障，需要持续 Goal 处理，不能宣称完整训练已完成。

全部完成后用[导出器](../tools/export_swa_ab.py)重新核验三组每5000步保留的全部24份 checkpoint SHA（含最终 checkpoint）、每任务完整样本数、分数、相同文档/prompt/target hash 与相同评测设置，再输出 120000 个原始训练点、480 个定期验证点、八项得分和七张 PNG/PDF/SVG 图。导出器拒绝未完成的 cohort。

最终 `source_hashes.json` 还会保留冻结 release 清单及其中配置、实现、初始化和评测锁的 hash，以及各组 resolved config、恢复核验、完成状态、summary、全部24份 checkpoint 及其元数据的 hash。导出时重新核对训练数据 manifest 与冻结清单一致；这只补充报告来源记录，不改训练代码或设置。

资源图的时间来自成功更新日志的 `seconds`，不含AMP溢出失败尝试、验证、checkpoint写盘、准备和评测时间，不能当作整个作业的GPU占用时长；回退后丢弃的成功更新耗时另列。溢出重试次数会保留，但现有日志没有逐次失败尝试的精确活动耗时。该说明仅明确计时范围，未修改运行中的训练或日志代码。

```bash
python tools/export_swa_ab.py \
  --run-dir runs/yoco-swa-160m-ab-8gpu-20261107 \
  --output evaluation/swa_160m_ab_20261107
```
