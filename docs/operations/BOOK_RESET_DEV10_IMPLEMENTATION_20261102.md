# BOOK_RESET_DEV10_V1：独立条件入口与当前资格

2026-11-02。依据[PR317协议第2节](../protocols/CONTROLLED_FINETUNING_AND_POLICY_DELIVERY_20261102.md)。本文件记录实现，**不是Core原生放行或新DEV成绩**。

现有`book_policy.py`新增显式`V7_BOOK_RESET_DEV10_CONDITION`模式；每个已登记进程只运行原十条件中的一个，沿用原模型加载器、Student、H40/Prefix8、4步生成、5秒预热、32秒任务、双RGB、原动作门、SourceEngine和原始评分。一个模型的十个条件需要十个有限进程，不能沿用旧模式的一次模型加载费用上界。旧`V7_BOOK_FROZEN_DEV10`仍在硬安全后结束面板，行为不变。

## 何时允许下一条件

当前owner先收集、独立审核并关闭前一worker；新worker启动模型前，按封存`RESET_PREDECESSOR.json`核对前一个兄弟运行目录的manifest/result/CLOSED lease。它再读取真实用户级unit，要求MainPID为0、unit终态、含子cgroup在内没有进程，并检查本uid是否仍有指向旧输出的可写文件描述符。状态文件或一个布尔“已关闭”不能替代这些观察。该过程只读、不终止进程、不修改系统服务。

从第二个条件起，还核对前项属于同一模型、同一新面板、恰好前一个条件，且是有效模型结果、初态/双相机/关闭侧证原字节仍匹配。有效硬安全负结果保留，随后新进程可以重新资格；工程失败、初态失败、关闭未知、残留writer或活跃子进程均阻止推进。每个条件的目录由面板摘要、checkpoint摘要和条件序号确定，不能另起随机run名称重做同一条件。

新进程使用新MjModel、MjData、IK数据和Student，检查共同home、原书本初态、零速度/控制/外力、arm/finger targets、IK缓存/计数及Student观察/规划游标。沿用原双相机视野检查，并要求相机资格仍在零积分时刻。初态、相机和关闭见证作为独立JSON侧证保存，不向原始事件流插入额外事件，也不改变原评分顺序。

**加速度缓存不能简单断言为零。** 实际PC构造表明，原MuJoCo 3.2.7的`mj_forward`会计算非零`qacc_warmstart`。实现另外新建独立MjData，以相同qpos执行构造加snapshot对应的两次`mj_forward`，逐值比较计算缓存；不覆盖原数据，不执行`mj_step`，不改变控制器或阈值。第一次错误的零缓存假设失败和费用保留。

## 固定模型与条件

`book_reset.plan(checkpoints)`只接受开跑前已确定的模型checkpoint manifest摘要，生成同一组原DEV000—009偏移、原`BOOK5_DEV10`种子命名空间和不变任务合同。新面板仍是已暴露DEV，绝非盲TEST；旧Phase、D0、D1、D2与STOP/未运行项不回填、不覆盖。尚未产生的D2/PB正式端点不能用资格10步权重占位。

worker额外封存`BOOK_RESET_DEV10_PLAN.json`、`RESET_NATIVE_QUALIFICATION.json`及`RESET_PREDECESSOR.json`。新模式每条件上界为1次模型加载、0更新、18,500次物理积分、80次生成、642对RGB、1次源模型构造；运行仍受原GPU、磁盘、lease和owner约束。GPU1必须有同checkpoint/同设备的真实CLOSED native96推理证书；训练资格不能代替推理或复位资格。

## 验证与未完成项

PC相关66项检查通过，覆盖实际可写描述符、MainPID/子cgroup、关闭和侧证变化、模型/条件/种子/输入状态、原推理容量合同及现有恢复/诊断回归。另有两个独立PC进程用固定原始机器人/LIBERO资产及MuJoCo 3.2.7实际构造原SourceEngine，初态见证逐值一致、子进程自然退出并回收；零物理积分、零渲染、零神经模型。该PC环境NumPy 1.24.4，与Core 1.26.4分别记录。

**尚未完成Core同运行时的原生复位、双相机及故障注入资格，也未挂载新DEV。** worker必须核对资格的真实主机、uid、NumPy/MuJoCo版本、原模板和六个相关实现文件的完整摘要；PC检查不会自动成为Core许可。未来在GPU冷态空档完成这些资格，再由现有唯一owner绑定固定模型组和逐条件任务；不新建调度器/账本，不热改manager13或当前D2。

当前训练队列仍为D2→固定D2的PB_CONTROL→PB_BOUNDARY；新复位入口发布不改这条已挂载链。下一交付是Core实际资格及新协议行为结果，不能以66项检查代替。

[实现与PC原生静态回执](../../reports/PI0294/design/controlled-finetuning-20261102/RESET_DEV10_IMPLEMENTATION_PC_STATIC_ACTUAL_20261102.json)
