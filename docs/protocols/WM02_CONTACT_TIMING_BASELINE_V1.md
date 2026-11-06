# WM02：同信息夹爪时序事件基线 V1

2026-11-06。独立接触支线的 CPU 开发诊断，不是新控制器或动态许可。

已有 mixed46 六个神经模型的 close72 长视野结果为负；不重复训练。当前支持夹持决策中，目标已足以决定 KEEP/RELEASE，不能把事件预测改善直接解释成更好的动作选择。

本次检验固定 ridge=1 的两臂 FIR 事件模型，视野 H1/4/8/16/32/64。输入只取原 WM 九帧历史中夹爪位置、力、内部目标/开度、当前接触事件，以及 ACTION_CONDITIONED 臂在相应视野内的 opening 命令增量。NO_FUTURE_PROGRAM 臂使用当前内部开度持恒，未来增量为零。禁止 phase、绝对时钟、close_start、parent/goal、未来观测以及视野之后的命令进入特征。每臂使用自己的有效 TRAIN 样本归一化。

命令条件沿用原 WM 的 `PREDECLARED_PROGRAM_CONDITIONAL`：静态程序须在 anchor8 已知，已选择 Policy 程序须有真实决策 hash 且在 anchor111 后才可知。这不表示未来 H40 已安装或已消费。`INSTALLED_REFERENCE_ONLY` 未运行。当前接触事件是原生 privileged 测量，与旧 WM 相同，不是纯视觉部署。

先按原始 descriptors 重建纯 TRAIN 输入：C09 common36 中的24 TRAIN episode，加 C10 T01 close56/64 的4条，共28 episode、8 TRAIN 父；不包含 close72。拟合两个模型、保存和精确重载，封存 `TRAIN_FROZEN` 后，才读取完整 mixed46 package、C09 common36、C10 timing12 与旧模型。matched package 的 TRAIN 原生身份必须与实际拟合输入一致。

评价同时报告两个 FIR 臂、持恒、TRAIN 事件频率，以及原 prefix36/mixed46 三 seed×两臂共12个冻结模型；不选最佳 seed。prefix36 模型的训练暴露较少，必须标明，不将其与28-episode基线的差异称为严格匹配增益。事件概率的线性输出显式裁至[0,1]并统计超界值；不裁动作或物理状态。不伪造 TCP/物体预测。

报告逐父、逐时序、逐视野的 Brier、误报/漏检与原网格事件时间指标；保留全部负结果。DEV 和 close72 已曝光，仅为回顾开发诊断，不是独立 TEST。两个模型、12 次闭式头求解、2 次精确重载与 optimizer updates 分开计数；本工具更新次数、新物理转移和物理积分均为0。

实际入口为 `scripts.wm02.fit_contact_timing_baseline --spec … --output …`，由现有 owner/registry/Journal 登记新的有限 CPU role 后执行。该 profile 不授 GPU 或接触执行权限，不复用 CLOSED lease，不修改 WBC、现役 capsule、旧模型或旧评价结果。
