# 当前入口 · Goal V7 · 2026-11-01

2026-11-01晨间服务器实测：**uniform4096已完成4096更新/131072次TRAIN访问，完整模型父链/42项AdamW/scheduler/RNG/cursor与全部trace审核通过，02:04 SGT已CLOSED。自动接续实际运行了1024原DEV10的两个条件，0/2成功、8未运行；第一个20秒时参考速度2.445>2m/s被拒，第二个8.074秒书本掉落并与地面穿透7.003mm触发原生安全。02:07队列SAFE_STOP，2048/4096各10条件暂停，事件训练跳过；当前没有继续运行该夜间GPU链。** 固定TRAIN16 H40位置均误16.37→9.03→5.53→4.26mm，尚不能据此宣称4096闭环成功。关闭后只读标签重名的问题已定位，本晨补核验CLOSED通过、未重跑条件；后继标签修复仅验证未部署。源合格书15/自定义柜门1/恢复6，自主完整成功仍0。[实际完成、评价负结果及安全停止](reports/PI0294/design/core32-maintenance-20261031/UNIFORM4096_COMPLETE_CURVE1024_SAFE_STOP_ACTUAL_20261101.json)。

2026-11-01 01:36 SGT实测：**Core服务器端接续已挂载，READY_FOR_DISCONNECT通过**；既有用户级常驻unit `core-unattended-curve-20261101-01.service`在控制连接退出后仍以同一进程身份存活且自主更新状态，运行不依赖PC、Vanda或转发认证。原uniform4096未改，最近实际3764步/120448次TRAIN访问；完整终态/全状态与trace审计/CLOSED后按1024→2048→4096串行原DEV10。事件采样尚未实现资格，本队列明确跳过；原生安全停止后关闭本面板并暂停后续物理，不自动解除STOP。当前新增评价/自主成功仍0，尚未发生训练后GPU接续。摘要和全记录驻Core本地，GitHub在现有发布通道恢复时核账同步。[实际部署与断线生存回执](reports/PI0294/design/core32-maintenance-20261031/CORE_UNATTENDED_CURVE_ACTUAL_READY_20261101.json)。

2026-10-31T23:11:49+08:00实测：同一4096训练已继续到2055步/65760次TRAIN访问；2048完整状态保存及42项AdamW、scheduler/RNG/cursor65536原生恢复证书已独立读回，生产者仍活动。固定TRAIN16 H40位置均误16.37→9.03→5.53mm，初始Prefix8 8.50→9.58→2.99mm；初始首夹爪误差降为0，闭爪锚点48的Prefix8夹爪MAE 0.0966→0.2655→0.0112。初始Prefix8旋转误差仍非单调，闭环成绩未取得。继续原均匀4096；完整终态/审计/CLOSED后依次评价1024/2048/4096。无新增物理尝试，纯书15/柜门1/恢复6，自主完整成功仍0。 [实际2048学习曲线](reports/PI0294/design/core32-maintenance-20261031/M32_ACTUAL2048_RESTORED_MATCHED_TRAIN_LEARNING_CURVE_20261031.json)。

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

## 最新实际与历史基线

2026-10-31 01:53 SGT实际：原E2作业1508345已由新的单一管理角色接管；包装/owner两进程按PID＋proc_start核验存活，原作业新Q态心跳已写出并独立读回。没有qsub、没有改原训练脚本/请求/时钟；尚无训练更新和512端点。首次管理接管因试图超过继承CPU硬限制在入口前退出，后继保留站点硬限制后接管成功，旧失败/关闭记录保留。[实际接管证据](reports/PI0294/design/science-goal-v7-20261031/E2_EXISTING_JOB_MANAGEMENT_REATTACHED_ACTUAL_20261031.json)。E1已真实加载/生成1次并被原动作门拒绝，正式有效条件0；合格新示范0。

2026-10-31 01:42 SGT实际：E1@512已在Core GPU0真实加载并完成1次自由生成（40×10）。原动作门拒绝：最大线速度4.8249m/s、角速度15.7340rad/s、平移路径5.4238m；命令安装/物理为0。工作器27.56CPU/35.47墙钟秒，自然OS0、MainPID0且own lease CLOSED，结果/数组/关闭intent和receipt均独立读回。这是保存观测的推理冒烟，正式有效条件仍0；E2新512端点/合格新示范仍0/0。[实际模型结果](reports/PI0294/design/science-goal-v7-20261031/E1_FIRST_REAL_FREE_GENERATION_ACTUAL_20261031.json)。

2026-10-31 01:21–01:35 SGT独立核验：E1原完整绑定/onceIO在隔离副本用36.7CPU/39.7墙钟秒，超旧30/35限制，正式账本未改；E2原1508345仍Q、原owner已退出；19条放书名义路径均未完成整段静态检查，没有新物理。新端点/有效模型条件/合格示范仍0/0/0。[实际证据](reports/PI0294/design/parallel-science-20261022/INDEPENDENT_E1_BIND_E2_LIFECYCLE_BOOK_CONSTRAINT_ACTUAL_20261031.json)。V7后继将先用同一E1加载器与Processor做一次独立生成冒烟，不冒充正式评价；源技能不等待完整E1/E2面板。

2026-10-31 00:55 SGT：E1共享启动失败已双账结算、三许可关闭，模型/查询/有效评价均0。Vanda完整绑定校验约26.8墙钟秒，接近旧35秒调用上限；不是SSH255根因已证实。E2作业1508345仍Q，管理owner已OS152退出，不能称健康训练链。新技能源域累计6尝试、22046积分、488对RGB、合格示范0；柜门旧分支8.82°后失抓，放书参考自身顶书。新完整端点/有效模型条件/合格示范仍0/0/0。

完整原件见[科学状态](CURRENT_SCIENCE_STATUS.md)、[E1实结](reports/PI0294/design/parallel-science-20261022/E1_SHARED03_DUAL_SETTLED_THREE_CLOSED_AND_BIND_VALIDATION_TIMING_ACTUAL_20261031.json)、[柜门终态](reports/PI0294/design/parallel-science-20261022/ROBOCASA_PORT05_REPEATED_GRASP_FAILURE_CLOSED_STOP_AND_PARALLEL_STATE_ACTUAL_20261030.json)、[放书诊断](reports/PI0294/design/parallel-science-20261022/BOOK_SAVED_PALM_INTRUSION_AND_FIXED512_EVALUATION_ENTRY_ACTUAL_20261030.json)。

## 接手方式

先读本页、V7与当前相关回执；需要实现位置时查[CODE_MAP](docs/CODE_MAP.md)。保持`main`唯一集成入口和`~/Projects/GR00T-WBC`工作区，遵循[仓库工作流](docs/REPOSITORY_WORKFLOW.md)。刷新Issue #11科学结果与#47运行状态，只对当前必要事实查询，不把读完全部历史变成启动条件。

Core单一物理owner，Vanda独立训练；管理端必要授权在计算启动前完成，避免计算进程等待跨站bind。NFS/SSH/挂载/服务/凭据由管理员处理。下一份汇报突出实际更新、有效任务条件、完整示范和失败机制，不以PR/CPU检查数量代替科研产出。

旧[V6](docs/protocols/GR00T_WBC_LONG_HORIZON_GOAL_V6_20261029.md)、[TD10](docs/protocols/TEN_DAY_CLOSED_LOOP_POLICY_PLAN_V1_20261029.md)及[本页更新前快照](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/fad5d0296084a656fb970b02b025c7b13af849af/START_HERE.md)仍保留；其未来调度/普通修复逐项GO规则与V7冲突时以V7为准，原实验身份/数据/评分/历史结果不变。
