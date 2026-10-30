# 当前入口 · Goal V7 · 2026-10-31

**用户最新要求：减少防御性编程，以推进科学主线为目标，工程问题允许自主修复。** 当前执行计划是[Goal V7](docs/protocols/GR00T_WBC_SCIENCE_GOAL_V7_20261031.md)，[六项任务卡](reports/PI0294/design/science-goal-v7-20261031/TASKS.json)与[AGENTS](AGENTS.md)同步生效；发布不等于远端已执行。

## 现在做什么

| 工作线 | 下一项真实结果 | 不等待什么 |
|---|---|---|
| Core模型闭环 | 已有E1等端点真实加载/生成，首个有效条件，再原面板 | 不等E2训练、不强制D25先完成 |
| Vanda E2对照 | 核清同一队列作业和已退出owner，恢复管理，完成原512 | 不等Core或新技能Teacher |
| 源ARX5任务数据 | 修复放书参考碰撞、诊断柜门失抓，取得各首条完整示范 | 不等E1/E2比较或目标S1成功 |
| 新技能/共享策略 | 合格种子后实际训练、固定DEV闭环，再共享四任务 | 不等凑满300条才开始验证学习 |

Codex可自主处理应用级路径、超时、类型、入口、队列管理、恢复与源域Teacher几何问题。先做受影响检查，再做最小原生验证，不为普通修复反复请求GO。旧源域失败STOP保留；有证据的新仿真开发版本可接续，不得仅改名盲跑。目标S1/T2安全隔离、冻结WBC、真实权限/配额、未知写入不得重放仍然有效。清理由用户负责。

项目内部预算总额和旧计划窗口不是新科研的人工审批门；真实GPU/内存/磁盘容量仍需核实。旧作业不热改、不重开、不回写历史费用/期限。V7允许简化未来运行的重复校验与跨站记账依赖，但这仍需运行代码实际实现，不是已完成修改。

## 最新事实基线（PR217，不是实时监控）

2026-10-31 00:55 SGT：E1共享启动失败已双账结算、三许可关闭，模型/查询/有效评价均0。Vanda完整绑定校验约26.8墙钟秒，接近旧35秒调用上限；不是SSH255根因已证实。E2作业1508345仍Q，管理owner已OS152退出，不能称健康训练链。新技能源域累计6尝试、22046积分、488对RGB、合格示范0；柜门旧分支8.82°后失抓，放书参考自身顶书。新完整端点/有效模型条件/合格示范仍0/0/0。

完整原件见[科学状态](CURRENT_SCIENCE_STATUS.md)、[E1实结](reports/PI0294/design/parallel-science-20261022/E1_SHARED03_DUAL_SETTLED_THREE_CLOSED_AND_BIND_VALIDATION_TIMING_ACTUAL_20261031.json)、[柜门终态](reports/PI0294/design/parallel-science-20261022/ROBOCASA_PORT05_REPEATED_GRASP_FAILURE_CLOSED_STOP_AND_PARALLEL_STATE_ACTUAL_20261030.json)、[放书诊断](reports/PI0294/design/parallel-science-20261022/BOOK_SAVED_PALM_INTRUSION_AND_FIXED512_EVALUATION_ENTRY_ACTUAL_20261030.json)。

## 接手方式

先读本页、V7与当前相关回执；需要实现位置时查[CODE_MAP](docs/CODE_MAP.md)。保持`main`唯一集成入口和`~/Projects/GR00T-WBC`工作区，遵循[仓库工作流](docs/REPOSITORY_WORKFLOW.md)。刷新Issue #11科学结果与#47运行状态，只对当前必要事实查询，不把读完全部历史变成启动条件。

Core单一物理owner，Vanda独立训练；管理端必要授权在计算启动前完成，避免计算进程等待跨站bind。NFS/SSH/挂载/服务/凭据由管理员处理。下一份汇报突出实际更新、有效任务条件、完整示范和失败机制，不以PR/CPU检查数量代替科研产出。

旧[V6](docs/protocols/GR00T_WBC_LONG_HORIZON_GOAL_V6_20261029.md)、[TD10](docs/protocols/TEN_DAY_CLOSED_LOOP_POLICY_PLAN_V1_20261029.md)及[本页更新前快照](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/fad5d0296084a656fb970b02b025c7b13af849af/START_HERE.md)仍保留；其未来调度/普通修复逐项GO规则与V7冲突时以V7为准，原实验身份/数据/评分/历史结果不变。
