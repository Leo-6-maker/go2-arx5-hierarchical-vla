# SLA V6当前计划入口 · 2026-11-18

**PLAN_ONLY。** 本分支只发布审核与下一长程计划，没有启动训练、模型、下载、仿真或真机。

先读：[PR407审核](reports/WM02/sla-v6-plan-20261118/PR407_REVIEW.md) · [详细V6计划](docs/protocols/SLA_SOURCE_NATIVE_SHARED_POLICY_V6_20261118.md) · [机读规格](docs/protocols/SLA_SOURCE_NATIVE_SHARED_POLICY_V6_20261118.json) · [Codex启动prompt](docs/handover/2026-11-18/START_SLA_V6.md)。

## 已取得什么

V5.2完整24条件为B0 5/8、B1 6/8、B2 6/8，无合格者；B2对B1只是F2+1/F1−1，没有总体净收益。1384生成、481233新tick、5拒绝、2碰撞、0技术失败；旧500步端点不重训，旧失败/TEST不回填。PR407@`1d78f2b758fb9b0dc5bbc05ecfc5a0aa3b1e63d3`完成真实数据首12条审计，原command＋SHIFT7混合仍BLOCKED。

关键新增解读：源ZOH动态夹爪窗521/661（78.82%）不满足条件性必要误差门；EEF任一违规1078中1076含内部违规。不是只缺dispatch证明或首尾问题。action[t]=state[t+1]不能冒充命令，但可以在新合同下作为未来记录参考监督。

## 下一路线

**T：有终点的可靠性对照。** 冻结B1对照；同B1父的FM500与完整10步生成H40加速度约束500，新24screen及条件确认8。只有这一组，不无限续训。

**S：源域原生学习。** 原双臂/三当前相机/30Hz名义H50，明确FUTURE_LOGGED_STATE_REFERENCE，官方SLA_base固定1000步；源统计/标签与Go2命令隔离。无需伪造dispatch或先证明单臂可移植，禁止未来观测输入。

**M：共享任务与初始化迁移。** 新版本逐时刻G7夹爪，完整HOLD/PLACE六种对象—目的地指令、最多48次Teacher。M-BASE/M-SRC各1000目标步，只改变共享LoRA初始化；source I/O/norm不拷贝。新共享screen、确认及独立DEV60/TEST60。T不过门不阻止合法M准备；S不通过时M-BASE仍可研究。

顺序/记忆/第二本体/真机随后另有具体人口与资源规格，不自动启动。新增任务不重训WBC；严格零目标数据跨本体仍是独立科学问题，不能由目标适配成功替代。

首周期P0/T/S上限2000更新/64000优化访问；条件M另2000/64000。旧Real10/25不恢复，旧C20统计/SHIFT7/历史报告不覆盖。真机、RL、BASE oracle、WBC调参、输出后处理、安全门放宽均不在范围。所有上限服从实际站点许可与截止，不是自动延长allocation或要求耗尽额度。

执行从经复核的PR407后继建立独立worktree，main只作发布/集成入口；接手时核对真实owner，不能把旧exit0当当前机器空闲。代码测试、JSON运行解析与GPU/native资格仍由执行端实际完成；本次只做源码/报告与预算算术审核以及GitHub回读。
