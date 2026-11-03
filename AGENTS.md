# 用户执行约束（2026-10-29）

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


用户要求完整测评后整理代码、文档并推送 GitHub main。HPC2 后台监测入口为 tools/watch_plan_ab.py，状态 runs/setup/plan-ab-publish-watcher.json；仅在作业和完整评测标记成功、回传校验通过、准备提交未被其他本地改动替换时生成结果文档并正常推送，禁止强推。
