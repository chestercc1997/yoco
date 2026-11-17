# gRet830M：块内溢出修复与完整状态续训

用户于2026-11-17授权修复后继续训练gated retention830M baseline。原run为`runs/yoco-gret-830m-baseline-8gpu-20261113T183312Z`；在尝试6438、6494、5926发生非有限前向，两次自动退回5000步后达到重试上限停止。旧release、checkpoint、失败后缀和三个诊断快照均保留。

## 根因与修复边界

第三份快照的第一层retention输入均有限：Q最大19.9233、K最大18.6043、V最大139.25，QK打分最大109.375。块内加权求和的FP16结果为Inf；跨块累积最大48484.7148，仍然有限。首次非有限模块为`self_decoder.layers.0.mixer.subln`，即原RMSNorm收到Inf。此前400M修复覆盖跨块结果，未覆盖此次块内结果。

最终修复保留原有限AMP计算路径。若块内求和结果不能用FP16表示，按原输入舍入边界在现有FP32状态累积dtype重算该加权和，并只替换不可表示的结果元素；随后沿原RMSNorm和FP16激活接口继续。没有裁剪激活、跳过token、改变gate/epsilon或更改训练参数。

这是明确记录的内部算术修复，不能宣称所有中间计算逐位不变。它不构成换成全局FP32/BF16训练。正常范围原路径保持一致；未观测的其他数值故障仍由原有有界监控处理。

## 验证

[故障回放与参考梯度证据](evidence/gret-830m-numerical-repair-20261117.json)：

| 原失败尝试 | 修复loss | FP32参考loss | 绝对差 |
| --- | ---: | ---: | ---: |
| 5926 | 3.414406538 | 3.414433718 | 0.000027180 |
| 6438 | 3.330815315 | 3.330643177 | 0.000172138 |
| 6494 | 3.102327347 | 3.102136850 | 0.000190496 |

三份旧快照均重现旧版本非有限前向；修复后的完整模型前向和反向梯度均有限。独立FP64逐token递推的前向和全部输入梯度核验通过。正常控制批次两版loss均为3.273478985，梯度相对L2差为0。前三个改变正常路径的候选超过原0.5%梯度容差而被拒绝；最终版本没有放宽容差。

[回归测试](../tests/test_gret_830m_repair.py)包含块内闭式溢出、有限反向、正常路径对照和历史前缀合并；[验证工具](../tools/check_gret_830m_repair.py)只读快照，不更新训练权重。FP32只用于诊断参考。

## 注意力反向确定性

首次修复release的训练checkpoint恢复对照超出原容差，已停止且未进入正式训练，记录保留。[独立注意力复测](evidence/gret-830m-attention-repeatability-20261117.json)在相同输入下发现默认反向Q梯度最大差0.0001220703125；启用确定性算法后，三次重复的Q/K/V梯度差均为0。单独指定cuBLAS工作区的检查也保留为未通过/中止的诊断，不将其误记为正式训练。

因此新版本显式开启确定性执行，并在CUDA初始化前指定cuBLAS工作区；这是执行路径变化，不是改变模型FP16 mixed精度、LR、batch或数据配方。加入确定性执行后重新回放三个失败快照并核查正常控制梯度，结果仍满足原阈值。训练恢复检查继续使用原容差，不能放宽。

## 固定梯度归约分桶

仅开启确定性注意力后，连续运行与独立重复结果一致，但分段恢复的模型差仍超限，因此该release也未进入正式训练。对5003更新的捕获显示：恢复前模型和AdamW动量按位一致；梯度最大差约9.31e-10，更新后参数最大差约7.45e-9，随后被FP16路径放大。不能将恢复前状态按位一致当作后续轨迹必然一致。

现有PyTorch的Reducer中，`should_rebuild_buckets`条件为`(static_graph || !find_unused_parameters) && !has_rebuilt_bucket`。显式设`find_unused_parameters=true`、保持原`static_graph=false`可阻止首次迭代后重建分桶，避免恢复时归约分组/顺序改变。所有参数均参与正常训练；该选项增加图遍历开销，不删除参数或改变全局batch。此执行变化与确定性注意力分别记录，八卡对照仍须满足原阈值。

## 续训血统与检查

新run由[准备工具](../tools/prepare_gret_830m_repaired.py)创建并冻结，不修改旧release；准备记录为`runs/setup/gret-830m-fixed-buckets-repair-preparation.json`。原5000步完整checkpoint SHA为`8ec410c76c9eeaf5a7c24600bf58410de4878436b024d1ef3926a8dcca79bc57`。加载其模型、AdamW状态、scaler和八卡RNG，继续相同数据步序；绝不从诊断快照续训。

模型、training、data、seed和tied embeddings与原run相同。execution新增`deterministic_algorithms=true`和`ddp_find_unused_parameters=true`，并使用`CUBLAS_WORKSPACE_CONFIG=:4096:8`，其余执行项不变。40k目标、原FP16 mixed/FP32参数、LR/warmup/schedule、optimizer、batch/累积/长度、验证与每5000步保存全部保持原值。

正式接续前做真实数据八卡5000→5002→5004、连续5000→5004、独立重复以及原5000完整状态在任何前向前按位加载的检查。保留原max_abs2e-5、relative_L2 2e-6、loss5e-5阈值；4个受检步骤计入正式预算。

[Controller](../tools/run_gret_830m_repaired.py)和[guard](../tools/watch_gret_830m_repaired.py)沿原策略监控八卡，保留失败现场，最多两次自动恢复。新的10000步checkpoint之前，以已核验完整5004步状态为回退来源；之后使用最近健康的正式checkpoint。不清空optimizer/scaler/RNG，不改训练setting。

原5000步checkpoint留在旧run，10000及之后的正式checkpoint留在新run。冻结的1..5000步健康日志前缀与新日志后缀由[血统工具](../tools/gret830_recovery_support.py)合并，[导出器](../tools/export_gret_830m_repaired.py)验证完整40000训练点和160验证点、八项全量下游及所有保留checkpoint。旧失败后缀不混入正式曲线。

ACTIVE仍为`runs/setup/gret-830m-baseline-active.json`，实际状态以其中指向的新run为准。修复通过不等于完整训练或下游已完成。

## 八卡恢复验收

固定分桶与确定性执行下，[真实八卡验收](evidence/gret-830m-repaired-resume-check-20261117.json)通过原阈值：分段恢复/连续运行/独立重复的模型最大绝对差和relative L2均为0，最大loss差为0；模型、optimizer、scaler及八卡RNG在任何前向前按位恢复。未放宽容差。完整40k训练和八项下游结果仍待完成。

[正式接续启动证据](evidence/gret-830m-repaired-launch-20261117.json)：run`runs/yoco-gret-830m-baseline-repaired-20261117T065955Z`已接续正式训练，八卡初段loss/梯度有限，监督器正常。后续40k、全量下游和最终报告审阅仍待完成。
