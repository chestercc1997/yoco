# 用户执行约束（2026-10-29）

## 2026-11-06 400M A 故障排查与恢复（当前）

- 用户明确要求修复故障后，从上一个 checkpoint 继续完整训练。原正式 run `runs/yoco-400m-A-8gpu-full-20261105T0310Z` 在完成 14310 步后，于 rank3 的下一步前向 loss 非有限而退出；八项下游尚未开始，不能报告为仍在训练或已完成。
- 最近正式 checkpoint 为第 10000 步，SHA256 为 `778250c85985c341a375ca0e238972c0cd3a9d213837290bac7c67bfd8881a92`。恢复须保留 optimizer、scaler、八卡 RNG、数据顺序及原精度/配方，每 5000 步保存并全部保留，不静默跳过出错 batch。
- **隔离故障重放已完成**：`runs/diagnostics/400m-A-failure-20261106/replay` 用原不可变 release 从 10000 到 14311，未重现前向 NaN；原 13937 步约 39.95 的梯度范数本次约 0.300。重放快照的 eager、按层编译、FP16 完整 reduction 和 FP32 诊断均有限，不据此断言原故障唯一根因。诊断步数不计入正式训练，禁止重复重放。
- `tools/diagnose_400m_failure.py` 已核验第 10000 步权重和 optimizer 全部有限、文件校验和正确；rank3 在 14311 对应的四批数据 token 不越界，使用第 10000 步权重时 eager 和按层编译前向都有限。该检查没有故障现场的 14310 权重，不能据此排除后期内核/数值问题或宣称已修复。
- 原记录还显示 13937 步裁剪前 grad norm 约 39.95（常态约 0.3），但尚未证明与后续非有限 loss 的因果关系。重放脚本在故障处捕获权重及输入，供 `tools/analyze_400m_snapshot.py` 对照；FP32 仅可作为不更新权重的诊断参照，不改正式训练 FP16。
- 已确认并修复 root `runtime/train.py` 的 AMP 回退错误：原 `scaler.update(new_scale=...)` 不清零 `_growth_tracker`，导致 13936 步溢出后只成功 64 步就在 14000 再次加大 scale，而原设定应连续成功 2000 步。`backoff_loss_scaler` 使用公开 state API 正确清零计数，保持 FP16/scaler 超参数及其余训练配方不变；CPU 与原生 AMP、远端 rank 溢出、保存/恢复回归核对通过，见 `docs/evidence/400m-A-amp-backoff-check.json`。不要将该修复冒称已证明是前向 NaN 的唯一根因。
- **修复版已恢复正式训练**：当前 `runs/setup/400m-A-active.json` 指向 `runs/yoco-400m-A-8gpu-repaired-20261106`，tmux `yoco-400m-A-8gpu`。真实数据八卡恢复与连续/独立重复对照、eval smoke 已通过，正式阶段从受检 10004 续训至 40000 并执行八项下游。先查 pipeline-status、heartbeat 与 tmux，禁止重复启动。新结果只衔接旧 run 的前 10000 步，不混入旧失败尾段或诊断曲线。
- 训练中 FP16 恢复的逐元素差异超过原初始化阈值，但独立连续重复也超过该阈值。恢复最大差异 1.058e-4，小于重复自身 1.103e-4；相对 L2 为 6.24e-6 与 4.58e-6。验收采用重复误差 2 倍内与固定硬上限双约束，RNG/scaler/optimizer steps/groups 精确一致，loss 差异 2.03e-5 < 5e-5。完整证据 `docs/evidence/400m-A-repair-resume-check.json`；未改训练精度或超参数，不冒称逐位相同。
- 来源 checksum、训练/模型/执行配置一致性和新不可变 release 的 CPU 准备检查已通过。恢复命令为 `tools/run_400m_A.py --run-dir /root/yoco/runs/yoco-400m-A-8gpu-repaired-20261106 --recover-from /root/yoco/runs/yoco-400m-A-8gpu-full-20261105T0310Z`，controller log 追加写入。不要编辑任何已准备或运行中的 release；`--prepare-only` 不切换 ACTIVE。

## 2026-11-05 进一步加速与方法 A 400M 完整训练授权（最新）

- 用户要求继续测试不改变训练 setting 的加速，并明确要求加速通过精度检查后，正式训练 **400M 方法 A**。仅训练 A；同规格 baseline 用于结构与共同初始化核对，不额外启动 baseline/B 完整作业。
- 沿用已完成 baseline 的数据、FP16 mixed/FP32 parameters、seed、microbatch=4、accumulation=4、8 ranks、global batch=262144、40000 步 schedule、验证 2048 序列，以及每 5000 步保存并全部保留。不得改为 BF16/全 FP32、增加 microbatch、缩短 token 预算或减少验证来宣称加速。
- 仓库原无 400M 配置，现将相同 YOCO 主干扩展为 24 层、dim=1024、FFN=2816、self heads=4、attention heads=16，baseline 参数=400590848，A 参数=400590860。方法 A 对前 12 层候选 K/V 使用 12 个全局 softmax 权重，初始等权。
- `tools/preflight_400m_A.py` 已验证 222 个公共 state tensors 与同规格 baseline 初始化逐项一致、退化到末来源时输出一致，并保存受检 A 初始化。400M 从该新初始化训练，不将 160M 权重当作可直接恢复的 400M checkpoint。
- 当前短程性能测试为 tmux `yoco-speed-audit`，目录 `runs/diagnostics/further-speed-20261105`；先查状态，禁止重复启动。进一步加速候选需做同批 loss/梯度检查，并在 400M A 上验证显存、吞吐及断点恢复后，才能进入正式训练。
- **正式训练已启动**：`runs/setup/400m-A-active.json` 指向 `runs/yoco-400m-A-8gpu-full-20261105T0310Z`，tmux 为 `yoco-400m-A-8gpu`，控制器 `tools/run_400m_A.py`。先读取实际状态与 heartbeat，禁止重复启动或另训 baseline/B。已通过八卡真实数据 `0→2→4` 恢复与连续/重复对照，从受检第 4 步续训至 40000 步，随后自动执行八项完整下游。
- 400M A 整体编译的同批梯度差异约 2.47%，未采用；当前采用 `compile_scope=layers`、`emulate_precision_casts=true`、`compile_loss=false`，保留共享 K/V 构造及跨层梯度汇总原路径。40 步对照 loss 最大差异 1.335e-5、梯度相对 L2 差异 1.527e-4；每步约 0.91 秒（原执行约 1.76 秒）。这不是最终收敛/精度结论，继续检查完整曲线。
- 用户已表达对本任务操作的全面授权，复用已批准工具规则，不重复询问同类 GPU/文件/联网操作；平台强制沙箱策略仍须遵守，不自行绕过。

## 2026-11-04 用户纠正：保持 A/B baseline 配置与精度（最新）

- 用户明确拒绝擅自修改训练精度：baseline 必须与 A/B 发布配置一致，使用 `fp16_mixed_fp32_params`；不再继续全 FP32 训练。旧 FP32 作业已按本次要求停止，保留其记录。
- 用户允许执行加速。可在不更改模型、FP16/FP32 参数精度、数据、microbatch=4、accumulation=4、global batch=262144、seed、AdamW 超参数和 40000 步学习率计划的前提下，测试并记录编译、融合 optimizer、关闭不必要的 activation recomputation 等执行选项。
- 保存策略仍为每 5000 步一个 checkpoint，全部保留；验证仍每 250 步、2048 条序列。不要为了宣称 3–4 小时而减少训练 tokens 或验证规模。
- 新匹配作业使用 `tools/run_baseline_matched.py` 和 `runs/setup/baseline-matched-8gpu-active.json`；启动前核查 GPU、状态和 tmux，禁止重复作业。仅训练 baseline，然后执行原八项下游评测。
- **已完成并核查（2026-11-05）**：`runs/yoco-160m-baseline-8gpu-matched-20261104T1519Z` 已训练至 40000 步并完成八项下游，`PIPELINE_SUCCESS` 存在，最终 checkpoint SHA256 已重新核验。北京时间 11 月 5 日 03:40 完成；训练约 3 小时 58 分，含评测约 4 小时 8 分。最终验证 loss=3.177219391，下游宏平均=45.6268371%。每 5000 步的八份 checkpoint 全部保留。当前任务已完成，禁止自行重复启动。
- 本次原始验证曲线从运行最优值的最大回升约 0.00718，未出现旧 FP16 run 在约 21354 步后的持续大幅退化。仍记录到 20 次动态 loss-scaler 溢出重试，不宣称完全没有数值波动。2026-11-05 用户已要求将本次数据更新 GitHub 并补充可复用的加速 docs；三组报告改绑本次 baseline，A/B 原始证据保持不变，旧结果保留在 Git 历史与 `docs/evidence/baseline-superseded-20261104.json`。

## 2026-11-04 baseline 数值异常排查与修复重跑（最新）

- 用户授权排查旧 A100 baseline 在约 21354 步后的 loss 异常，并在修复后从健康 checkpoint 重跑。旧完整 run 与发布报告保留作历史，不将其正常退出视为健康收敛证明。
- 健康来源为原 pilot 的 step 100；保留模型、optimizer、各 rank RNG、数据顺序、八卡 batch 和原 40000 步 LR schedule。新的精度配置作为显式 fork 记录，不冒充原 FP16 作业无变化恢复。
- 用户最新明确要求每 **5000 步** 保存一个 checkpoint，全部保留；验证仍每 250 步。正式训练不改成每 250 步保存。
- 新控制器 `tools/run_baseline_stable.py`：先通过真实数据八卡恢复核对，再训练与八项下游评测。新作业从 `runs/setup/baseline-stable-8gpu-active.json` 读取；启动前核查状态、tmux 和 GPU，禁止重复启动。本条取代旧 run 完成后的“不重复训练”限制，仅授权本次数值修复重跑。

## 2026-11-04 本机 A100 八卡 baseline 完整训练授权（当前）

- 用户明确要求用本机 8 张 A100 完整训练 baseline 160M，而非仅运行 100 步测试。本条授权本机原始 YOCO baseline 的 40000 步训练及同设置八项下游评测，取代历史 HPC3“不重训 baseline”对本轮任务的限制。
- 从已通过真实数据八卡测试的第 100 步 checkpoint 接续，保留 optimizer、scaler、RNG、40000 步 schedule。global batch=262144 tokens、sequence length=2048、microbatch=4、accumulation=4，验证仍为 2048 条序列。
- 本轮只运行 baseline，不启动 A/B。代码与评测锁保存为 run 内独立快照，保留原 pilot 证据。
- 当前本机作业从 `runs/setup/baseline-full-8gpu-active.json` 读取；先核对该文件、对应状态、heartbeat 和 tmux，禁止重复启动。最终训练加评测完成以 run 内 `PIPELINE_SUCCESS` 为准。
- 本机 8×A100 baseline 已完成 40000 步与八项完整评测，最终验证 loss=3.408905983，八项宏平均=43.6630747%。本轮任务已完成，不重复启动训练。
- 用户要求将 `evaluation/ab_baseline_20261104` 三组报告中的历史 baseline 换成本机新 baseline。七张 PNG/PDF/SVG 图、120000 个训练点及基座核查已更新；A/B 发布数据保持原样。基座核查以完整模型重建和配置为准，不将未取得的远端 step-zero 权重宣称已直接核对。按用户要求，软件版本差异不作为本轮核查判定项。

## 2026-11-03 A/B 四卡完整训练授权（最新）

- 用户要求取消重复 baseline 作业 793493，只训练 A/B；旧作业已取消，日志和 checkpoint 保留。旧两卡 baseline 788651 已完成 40000 步，不重训。
- A 为 `learned_kv`：第 1–6 层各自产生候选 K/V，学习六个全局 softmax 权重；B 为 `conditional_kv`：由各来源 hidden states 生成每 token 六个 softmax 权重。两组均在第 7 层前融合一次，后六层共用 K/V；K/V 共用配比。
- A/B 各 4×H100，从共同主体初始化分别训练 40000 步；同 seed、全局每步 262144 tokens，2048 验证样本，初始六层等权。不使用旧三来源 hidden-state static 实现代替 A。
- 新独立快照 `experiments/plan-ab-full-4gpu-20261103-v2`；单节点四卡 allocation 串行 A→B，再并行两组八项下游测试。两组四卡 smoke/恢复检查均通过后才进入正式训练。动态节点，不指定 nodelist。
- 当前作业从 `runs/setup/plan-ab-active-job.json` 或远端快照的 `submitted.json` 读取。存在作业或 submission-intent 时先核对调度器，禁止重复提交。下文 793493 及旧 baseline/dynamic 配对记录均为历史。
- 详见 `docs/PLAN_AB_4GPU.md`。本轮以 A/B 为主要配对比较，旧两卡 baseline 作为历史参考，不直接宣称跨卡数训练速度优势。

## 2026-11-01 完整训练授权（最新）

- 用户在看到两卡短程结果后明确要求“先跑完看看效果”，已授权 baseline 与动态 v2 从 1000 步续训至 40000 步。本条取代之前“需先有短程收益才启动完整训练”的条件，不要再次询问该授权。
- 完整作业：baseline `788651`，dynamic `788652`，各 2×H100、24 小时时限。当前记录 `runs/setup/full-v2-active-jobs.json`；旧 pilot 作业已完成，不重复提交。
- 新 run 目录与旧 pilot 分离，复用原不可变 runtime/配置和 checkpoint 的 optimizer/scaler/RNG；不更换卡数、重置 schedule 或中途改路由设计。
- 完整训练后台状态 `runs/setup/full-v2-watcher.json`；验收输出 `docs/evidence/full-v2-convergence-review.{json,md}`。检查真实状态后再行动，避免重复提交。
- 细节见 `docs/FULL_2GPU_V2_TRAINING.md`；如因基础设施问题失败，保留证据并按原配置从最近有效 checkpoint 恢复。

## 2026-11-01 两卡配对与动态调整授权（历史）

- 用户明确允许单独提交两卡 baseline，并按两卡动态试跑的问题调整路由设计和参数；同时要求验收“梯度更直接是否让拟合/收敛更快”。这些短实验和诊断无需重复询问许可。
- 当前 baseline 作业为 `788464`、动态 v2 为 `788465`（原 788258/788259 因启动参数触发容器封装器误判，在训练前失败；已仅修复 launcher 并重提），各 2 卡、1000 步，从头配对训练。v2 只将路由初始权重改为 `[0.05,0.05,0.90]`；保留原 v1 证据，不修改运行快照。
- 本地长运行监测进程（执行会话 40443）已安排在两组完成后作只读梯度诊断、回传及收敛验收；状态文件 `runs/setup/paired-v2-watcher.json`，当前作业记录 `runs/setup/paired-v2-active-jobs.json`。先查状态，禁止重复提交现有作业或重复启动监测进程。
- “gradient 拟合更快”暂解释为本设计的梯度传播/收敛假设，不擅自引入新梯度算法。验收以相同 tokens 的 loss 和到达同 loss 的训练量为主；梯度范数只作诊断。
- 下一步以 `docs/DYNAMIC_YOCO_THEORY_AND_EXPERIMENTS.md` 及结果为准；完整 40000 步仍需效果证据支持，不因短作业正常退出就自动开始。

## 2026-11-01 动态融合两卡授权

- 后续用户提出“检查效果可以则启动完整版本”。1000 步效果检查发现动态 v1 验证 loss 5.7308、baseline 5.0690，四个验证点差距扩大；条件未满足，本次未提交完整训练。证据见 `docs/evidence/fusion-token-v1-review-20261101.json`。作业 788159 已正常完成；查询该记录，不重新提交 pilot。
- 用户要求为新实现申请两卡并启动训练。本轮为 YOCO 160M `memory_fusion.mode=token`，单节点两张 H100，先跑 1000 步，保留 40000 步 LR schedule。
- microbatch=4、gradient_accumulation=16，保持全局 batch 262144 tokens；validation_batches=256，保持全局 2048 条验证序列。先通过两卡 synthetic smoke 和 checkpoint 恢复检查，再跑真实数据。
- HPC3 独立代码目录：`/data/share/jzhou945/cchen099/yoco/experiments/fusion-token-2gpu-20261101-v1`。不得在运行中改写此快照或旧 baseline 目录。
- 提交记录：本地 `runs/setup/submitted-fusion-token-2gpu-20261101-v1.json`，远端独立目录下 `submitted.json`。已存在作业时查询该作业，不重复提交；若仅有 `submission-intent.json`，先核对调度器再决定恢复。
- 两卡脚本不指定节点。其他已排队作业保持原调度。

## 2026-10-30 最新授权

- 用户已要求完成跑前数据/训练设置检查，并用四卡试跑 160M；因此允许本次分词、部署和 GPU 试跑，取代下文昨日“不要自动接着启动分词或训练”的限制。
- 当前试跑为 YOCO 160M，HPC3 单节点四张 H100，先跑 1000 步，保留原 40000 步学习率 schedule；对照仓库公开复现曲线，不称为 Microsoft 原始官方训练复刻。
- CPU 准备仍用本地普通进程/tmux。全部数据检查通过后才提交 GPU 作业，作业内先做四卡 smoke 和恢复检查。
- 当前控制器状态：`runs/setup/pilot-controller-20261030.json`；已记录 job_id 时不要重复提交。

## 昨日记录（以上述最新授权为准）

- 下载、校验、数据转换/分词等 CPU 准备任务直接用本地普通进程或 tmux 后台执行，不通过 Slurm/sbatch 提交。
- 用户已要求取消 CPU 数据准备作业 13393990；不要恢复、重新提交或启动同用途的 CPU Slurm 作业。
- 当前授权范围是：复用 HPC2 已下载文件，续传并校验原始数据，然后 scp 到 HPC3 `/data/share/jzhou945/cchen099/yoco/` 对应目录；不要自动接着启动分词或训练。
- 后续 CPU 预处理如另获执行授权，仍用普通后台进程。已有 GPU 实验的调度保持各自计划。
- 保留旧 Slurm 脚本和日志作为历史证据，不再作为 CPU 任务执行入口。
- 普通联网命令遵守 network-routing 技能，优先单条命令直连，不改动 Codex 自身代理。


2026-11-03 用户最新明确要求代码和文档立即推送 GitHub main，不等待排队、训练或评测。原等待全流程后发布的后台任务已停止，旧条件已被本次指令取代。作业 794750 保持原调度；README 与文档明确实验结果尚未产生。tools/watch_plan_ab.py 仅保留为可选工具，目前未运行，勿自动重启发布任务。
