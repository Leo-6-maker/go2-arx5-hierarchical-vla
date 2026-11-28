# 冻结R4 Expert候选：真实冻结语义与保存接口已完成CPU准备

[参数分区前向](V7_B6_PARTITION_COUNTERFACTUAL_RESULT_20261128.md)已显示：在原安全抓持状态，只换回R4私有IO不能消除锚250退化，换回R4 Expert则恢复raw门内输出。按V3的一种Expert更新约束路线，本次准备 **Expert保持R4、只训练各域私有IO** 的独立候选，未启动正式训练或PBS。

这与原官方SLA-PT冻结迁移基线不同：它拟从本体R4-200开始做目标适配，不能称为零目标训练的跨本体结果。混合权重推理也不是该训练候选的效果。

## 已完成的实际工程检查

1. **真实冻结而非只清零梯度。** 原AdamW含非零decoupled weight decay，单把Expert梯度清零仍会改变其权重。新优化器不调用共享分区AdamW，参数、moments和count保持原样；私有IO梯度先组成一个全局范数、按原clip限幅后进入原AdamW。冻结Expert的梯度不压低私有IO的clip比例；任何冻结分区NaN仍作为上游错误拒绝。
2. **CPU合成数组检验。** 使用每分区2×2参数、非零weight decay、共享梯度1e6，私有更新和Adam状态在五个合成逻辑步都与原优化器的对应私有更新完全相同。Expert及其Adam不变；保存/恢复合成状态和下一合成更新相同。缺域、NaN、零活动梯度、错误共享count及非零共享moments均拒绝。[最初机制检查](V7_FROZEN_EXPERT_OPTIMIZER_SYNTHETIC_CPU_ACTUAL.json) · [加入checkpoint时钟检查后的回执](V7_FROZEN_EXPERT_OPTIMIZER_AND_CLOCK_SYNTHETIC_CPU_ACTUAL.json)
3. **全模型形状ABI检查。** 实读已完整恢复的250端点metadata，以真实NNX Param状态结构抽象追踪初始化，全部76个Adam状态叶的路径、shape、dtype与原完整状态一致。11个Expert叶共427,932,672参数；target9、RoboDojo8、LIBERO8个私有叶共6,498,400参数。仅读metadata与抽象初始化，没有加载模型权重值、做前向或实际模型更新。[全形状回执](V7_FROZEN_EXPERT_FULL_SHAPE_ABI_ACTUAL.json)
4. **修复冻结状态与旧保存器不兼容。** 原保存器在写checkpoint后要求所有Adam count等于训练step，真实冻结Expert的count必须是0，因此不能直接复用。新增独立保存/恢复入口，写入前检查共享count/moments与原R4指纹，保留完整原snapshot结构；私有count=step、共享count=0，不伪造优化器时钟。该完整模型保存/恢复路径尚待真实资格，未冒充已通过。

最初抽象形状探针把snapshot的纯字典当作运行时Param树，正确拒绝了路径不一致（缺少moment的`value`节点）；改为NNX Param状态后76叶全匹配。这是CPU探针修正，未重跑科研样本或产生模型更新。Core与Hopper实际原优化器源码SHA相同，`8946ec63…67f11fc`。

上述合成fixture发生的是小数组优化调用，已单独记录次数；**真实模型工程/正式更新均0**。它们只能证明优化器机制与ABI，不能证明H200完整FM/aux梯度、真实更新稳定、吞吐、模型冷恢复或原生能力。

正式worker的准入负例也已实际执行：将本次真实CPU机制/形状回执作为候选资格输入，入口按scope明确拒绝，没有启动模型或更新。[CPU回执不能替代GPU资格](V7_FROZEN_EXPERT_CPU_RECEIPT_NOT_GPU_QUALIFICATION_ACTUAL.json)

## 已固定的候选与待满足条件

拟用R4新建私有Adam，原A93＋R3、128 FM（48旧/46新/2 R3＋16 Robo＋16 LIBERO）、aux4/0.375、原500步学习率前250位置、实际CUDA流、4 worker/2批预取全部不变。唯一主要变量为共享Expert能否更新；不同时加入S2/A2、早期新数据、新loss或新视觉范围。[完整候选登记](V7_FROZEN_EXPERT_CANDIDATE_REGISTRY_20261128.json)

已有B6零系数250可作为同数据/父/架构/噪声的完整训练对照，前提是所有实际输入及课程身份保持精确一致，不再重复训练该健康终态。新候选250为待执行设计，不是已批准新增scope资格或模型结果。固定37格TRAIN面板及原42格源域保持复用原输入和已封存对照输出；原生只有4个余额，尚未新登记原生名册，不能声称足够完成晋级。

当前原工程额度已 **24/24**。新的冻结scope至少仍需2次真实H200工程更新，以及Core独立完整恢复/下一批噪声检查，尚无新增分配。因此本次只交付CPU与代码准备，**不提交PBS、不以合成数组或形状追踪替代GPU资格、不把资格更新改记成正式步绕过上限**。正式训练累计5100/8000、学生46/50不变。D1卡19是另外已登记的Recovery采集，并行进行，不等待本候选。

实现：[优化器](../../scripts/sla/v7_frozen_expert_optimizer.py) · [CPU机制检查](../../scripts/sla/v7_frozen_expert_optimizer_cpu_check.py) · [全形状检查](../../scripts/sla/v7_frozen_expert_full_shape.py) · [独立保存/恢复](../../scripts/sla/v7_frozen_expert_checkpoint.py) · [须真实GPU资格回执才能进入的正式worker](../../scripts/sla/v7_frozen_expert_formal.py)。原运行代码、WBC、L1、安全门和历史端点均未修改。
