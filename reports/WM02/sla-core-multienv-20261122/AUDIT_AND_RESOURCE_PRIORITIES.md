# 审计依据与资源优先级 · 2026-11-22

本次仅审阅GitHub固定代码/报告并编写计划，未登录Core/Hopper/VANDA，未提交现场任务。执行证据固定于`1fee437d12d9a79deff8e2be0175da3f148dd48e`；发布从当前main `6843163d641eb24a29c3b1557f868765346ba66c`另建文档分支，不合并活动执行树。

## 一、已核对的来源

| 发现 | 固定来源 |
|---|---|
| 最新H2001784/封存1700，七CPU链停止且无后继 | [VANDA_SEVEN_CPU_CHAINS_STOP_ACTUAL.json](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/reports/WM02/sla-core-hopper-parallel-20261121/VANDA_SEVEN_CPU_CHAINS_STOP_ACTUAL.json) |
| H200/Core配对两窗口与原生0，Core插桩改变输出 | [PHASE0_PAIRED_PILOT_ACTUAL.md](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/reports/WM02/sla-core-hopper-parallel-20261121/PHASE0_PAIRED_PILOT_ACTUAL.md) |
| 40布局、共同平移、每布局六指令；split及机器人保护 | [core_hopper_geometry.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/scripts/sla/core_hopper_geometry.py) |
| 当前单world锁、逐条准入、字符串连续失败暂停 | [core_hopper_collection.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/scripts/sla/core_hopper_collection.py) |
| 实际父PID/锁FD/inode/source/scope绑定 | [core_hopper_expert.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/scripts/sla/core_hopper_expert.py) |
| 8格两次接触，仅5格阶段和部件相同 | [TRAIN_CONTACT_PAIRS_036_ACTUAL.md](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/reports/WM02/sla-core-hopper-parallel-20261121/TRAIN_CONTACT_PAIRS_036_ACTUAL.md) |
| 专家12终态轮换与潜在学生饥饿；组合资源门 | [core_hopper_native_queue.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/scripts/sla/core_hopper_native_queue.py) |
| H200 REF16前置导出、旧Core严格证明、真实IPC | [hopper_reference.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/scripts/sla/hopper_reference.py)、[hopper_core_receive.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/scripts/sla/hopper_core_receive.py)、[core_hopper_student_endpoint.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/scripts/sla/core_hopper_student_endpoint.py)、[core_hopper_native_scope.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/scripts/sla/core_hopper_native_scope.py)、[core_hopper_native_campaign.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/scripts/sla/core_hopper_native_campaign.py) |
| 旧N2吞吐与未通过的状态/STOP证据 | [PERFORMANCE_PARALLEL_ACTUAL.md](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4d597259/reports/WM02/sla-d2-teacher-parallel-20261118/PERFORMANCE_PARALLEL_ACTUAL.md)、[机读](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4d597259/reports/WM02/sla-d2-teacher-parallel-20261118/PERFORMANCE_PARALLEL_ACTUAL.json) |
| 原训练/课程/控制/总期限 | [V1原计划](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/171b5bedb290258954aac140157743e92e3b152e/docs/protocols/SLA_CORE_HOPPER_PARALLEL_V1_20261121.json) |
| 原VAL24和早期接触部件分布；320H200小时和绝对期限 | [SERVER_NATIVE_AND_ENDPOINT_ACCOUNTING_ACTUAL.md](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/reports/WM02/sla-core-hopper-parallel-20261121/SERVER_NATIVE_AND_ENDPOINT_ACCOUNTING_ACTUAL.md) |

用户11/21 23:19快照的67TRAIN/38完整、26HOLD/12PLACE、52抓持抬升和24接触失败，在本计划明确作为用户现场报告，不冒充本次新SSH回读。后来的`expert_sealed97`字段不提供同口径成功拆分，不能用它增加成功数。

## 二、为什么资源这样排序

1. 新完整数据未进入梯度、学生没有实际分母时，增加生成速度不能回答训练是否有效。恢复数据接入、后继/备份及Core评价可先打通反馈。
2. 由用户快照计算：38/67=56.7%完整；52/67=77.6%抓持抬升，14条在抓持后仍未完整成功；24/29=82.8%的失败是接触。接触修复和PLACE覆盖比重复易HOLD更值得优先投入。以上是混合Teacher开发数据，不是固定Teacher泛化分数。
3. 当前40布局只改变共同平移，不能靠增加worker变成丰富的相对几何。新480请求显式只称80个独立布局，六指令共享布局有助于区分指令执行，不将每条请求视为独立几何。
4. 旧N2已有实测潜力，因此安排最小修复及双角色隔离；N4的显存/主存/渲染竞争未经证明，置于N2之后。
5. 原七CPU链已停，不能写“后台已挂好”。新版调度必须有真实权限、运行身份、有限期限与断开后的独立证据；不将CPU守候伪装成GPU作业。

## 三、旧短工程容差（仅复用固定输入和阈值，不继承资格）

容差SHA256：`6da63587addd8d9e35d36ce09191593de9dba46b2cee9b47266beeb71df0fc0a`。

| 工程场景 | EEF m | EEF rad | mount m | mount rad | jaw fraction |
|---|---:|---:|---:|---:|---:|
| F1 STABLE | 0.0025152321752068725 | 0.017453292519943295 | 0.002185499738484875 | 0.017453292519943295 | 0.001 |
| F2 STABLE | 0.005999447781465302 | 0.0226721509438131 | 0.011078147106107416 | 0.03166960249425338 | 0.001 |
| F1 SMALL_EEF | 0.0040381210413138425 | 0.017453292519943295 | 0.0031203193680729565 | 0.017453292519943295 | 0.001 |
| F2 SMALL_EEF | 0.01017728480458699 | 0.018870459958318493 | 0.0077012046843688295 | 0.024521885853891402 | 0.001 |

这些容差不用于跨GPU神经输出，不是机器人精度或接触安全阈值；完整接触还要独立核对控制消费、阶段、接触和评分。不能使用时间平移/平均掩盖瞬态超限。

## 四、官方并行背景

MuJoCo官方[Simulation / Multi-threading](https://mujoco.readthedocs.io/en/latest/programming/simulation.html#multi-threading)说明并行采样采用独立动态状态，模型在只读时才适合共享。官方[Rendering / Context and GPU resources](https://mujoco.readthedocs.io/en/latest/programming/rendering.html)说明渲染依赖当前线程的图形上下文和对应GPU资源。因此本项目应隔离每world的状态与渲染，不把增加Python进程误认为整个WBC/ROS/IPC已经隔离。引用用于架构原则，本轮不升级到网页所述最新API或renderer；以现场冻结版本为准。

## 五、文档验证与现场待验

本次静态验证仅验证预算、人口唯一/拆分、seed、链接、分阶段和声明一致性。没有执行CPU物理、GPU、实际IPC或N2/N4资格。详见[PLAN_VALIDATION.json](PLAN_VALIDATION.json)。原生新总上限按实际HOLD/PLACE分别求和，工程为1,074,992tick；没有用全部完整卡都按PLACE上限的宽估值替代真实清单。
