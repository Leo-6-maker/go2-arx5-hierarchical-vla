# Codex启动prompt：SLA真实源域接入与0/10/25%配比

使用本文件时，固定读取它的Git commit，并先核对PR405最新实际进展；计划发布未启动作业。下面指令供具有真实项目访问权限的执行端使用。

```text
继续Leo-6-maker/go2-arx5-hierarchical-vla，但不要打断PR405/V5.2健康执行。执行本计划中的有限真实源数据接入与配比研究，而不是重复已有500步训练或立即启动真机。

先读同一固定计划提交的：
SLA_REAL_SIM_MIX_START.md
reports/WM02/sla-real-sim-plan-audit-20261118/REPOSITORY_AUDIT.md
docs/protocols/SLA_ROBODOJO_REAL_SIM_MIX_V1_20261118.md
docs/protocols/SLA_ROBODOJO_REAL_SIM_MIX_V1_20261118.json

审阅基线PR405@01b4a11c80f95956742379f73404be8fbdf909e1，实际inference/native source2797d0c2，两个500训练端点已完成。00:38:43 SGT快照只是9/24启动、8终态、5成功；接手时查最新真实状态，不把快照当现在。B2已经出现新内部k34拒绝，不能按旧ROLL96通过提前宣布胜者。先保留并完成既定24条件与原固定决定；不改loss/数据/评分/seed，不热改、重开CLOSED或重投科学失败。

我的本次执行范围是该计划的CPU数据审计、必要应用级实现，以及满足依赖后的有界GPU/纯MuJoCo比较；真实site权限、owner、容量与有效期限必须存在，不能自批或延期。旧服务2026-11-18 10:12:18与历史allocation2026-11-19 02:51:23是不同截止。当前聊天没有部署新作业。

R1先定向审计RoboDojo-Real的arx_x5，解析并固定HF revision。首12包含于总60 episode，原始≤32GiB、新数据总≤64GiB。按session/scene/episode簇划分源TRAIN/DEV/TEST，核命令与反馈、时间、frame/TCP、夹爪和双臂依赖。没有command生产者证据，不把字段名action当成执行ACK。不要全库下载或在AMD做重编码。

R2保留SOURCE_NATIVE，并另建可兼容单臂全H40银行。真实未来action序列不是anchor时的一次已执行H40；没有5703tick或P10消费证书就不伪造。30Hz按实际时钟处理，输入不能用未来图像。旧SHIFT7标签把运动硬设2秒，不能直接套真实曲线；只允许同一未改decoder族的独立逆编码，保留raw/retargeted/reconstructed与误差，按计划事件覆盖和容限过门。不能只筛静止片段凑样本。目标H40门不是全部源数据的有效性门，但每条学生部署命令仍严格通过原门。表示不兼容就交付定量BLOCKED，不偷偷换decoder或把future achieved改成issued command。

R3只有兼容性与资源成立且旧健康消费者处于冷态边界后，才训练Real10/Real25各500/B32。共同父始终是原SHIFT7；Real0复用已完成B2（目标FM+endpoint0.1），不是从B2续500。冻结C20统计、参数范围、输入、SHIFT7 codec、LR/noise/time/dropout、cached_actual/10步与WBC。每10update/320槽精确替换32或80槽，嵌套mask；至少12000个共同target槽完全沿用B2银行，随机数用绝对槽、不压缩cursor。objective为FM32+0.1*is_target*edge，全batch平均，不双重乘source比例。

R4按协议完成零更新接缝对照、端点87数组恢复、独立RGB16、372目标开发条件和64源DEV诊断。离线合法率不是任务成功，源动作误差不是RoboDojo真实任务成绩。包身份正确但科学失败的控制仍保留screen。

R5新24条件三臂配对开发比较，原WBC/H40/P10/TTL/2ms/碰撞STOP/评分保持。按完整面板和原门选唯一候选；新确认8→新DEV32→一次TEST40，不占用或改名V5.2封存人口，不换seed刷成功。初始化按物理量分开报告，不能把同scene/reset等同完整WBC状态一致。失败保留成本和部分tick；STOP后积分、owner/冻结/隔离异常停受影响campaign。

最多新增1000优化更新/32000访问，1080离线完整生成，104科学native+最多2接口，3725018tick；所有技术已发生费用计入同一上界。复用现有reader/Journal/owner/worker/配额规划，不新建多层审批或调度器。应用级小修自主完成并定向回归；新方法或新增任务不冒充配比变量。

第一交付是现役E2终态结论＋SOURCE_SEMANTICS/CODEC_COMPATIBILITY，不必等待所有后继。兼容后交付两个真实500端点和24闭环逐条件结果。每阶段提交ACTUAL.md/json、数据/模型/统计/source SHA、实际样本暴露、全部分母、首失败与费用。真实Go2控制、Memory/Spatial Forcing/BASE/RL/新loss/大规模多任务扩训均不在本轮。未过门如实报告，不把数据下载、CPU检查或PR发布写成policy成功。
```
