# 新 Agent 直接接管 Prompt

你接替已经停止的旧 local-pc owner，成为 GR00T＋WBC 的唯一调度/发布者；不新增竞争owner。阅读同包 `NEXT_AGENT_EXECUTION_PLAN.md` 和 `NEXT_AGENT_TASK_CARD.json`。固定原交接commit：`7e7773a5628d7f119076220ac94bb92a1a9ce120`，集成分支 `pi0267/executor-decoupled-recovery`。该交接HEAD不是科学runtime，原R1 runtime是`021d813fd669561a24ffbf89431a4a9812e0f2ca`。

先读根AGENTS/START_HERE、`docs/handover/2026-10-14/HANDOFF.md`及`CODE_AND_EVIDENCE_INDEX.json`、用户指定的本机私有交接，以及#11/#47最新增量。私有文件路径只在本机解析，不上传。已有166份脱敏owner脚本是带guard的历史参考，不得移除guard后直接运行。通用workspace CLI和通用Hopper入口没有部署资格，复用已经合格的enter/doctor/私有启动器。

当前M512完成512步；48条离线生成和第一条T1的6次新观测查询真实发生，但在2.162秒出现右指原生限位FAIL。P1保持FAIL、selected/deployment=NONE。原两PBS均终态、预留0是旧交接快照，先只读现场刷新。不要盲目启动剩余PBS，不从pristine重训、不切v3.1、不新采数据或清理。

第一阶段只做只读接管，交`OWNER_ADOPTION_READONLY.json`：身份、私有绑定可用性、实际进程/预留、模型/数据现有manifest、当前科学授权和唯一下一动作。此请求不是夹爪修订或新科学预算GO。

若#11已有用户明确采用完整新包，核对其固定release、允许改变的夹爪driver、绝对UTC/SGT窗口和增量预算后，在新run中依次自动推进：CoreLab G0-G3力学归因/2×2短对照/最多两单项候选/8条Teacher回归；通过后同一M512+D1四条新观测P1R；通过后从原完整状态global512续训到2048/4096并跑同一DEV；冻结单一候选后核心确认/旧抓放回归，挑战仅有余量运行。

原安全阈值、500Hz检查、2ms步长、H40/prefix8和语义不变，旧失败不覆盖。所有Teacher和模型用同一D1。v3.0/stats、原确定性numerics、micro2×16、workers0、6144 scheduler horizon和308 warmup不变。模型安全执行与任务成功分开；执行器修复不算模型训练收益。

没有明确新GO则完成只读接管和待采用卡后停在WAITING_SCIENCE_GO，只请求一次合并授权，不反复确认同一阻断。获得完整包后不逐阶段重复索要同项许可。旧截止/余额不自动续用；所有新消耗与旧账分列。

首个有意义的新模型交付是四条原视频与P1R行为评分，不是又一次上传checkpoint。完整本地checkpoint校验后允许同节点评价，远端归档失败不自动终止其他合格阶段。真正的安全、恢复、权限或时间失败保留终态并停止相应范围。最后发布#11科学/#47工程收据并回读，不把计划发布或CPU CI成功写成模型技能成功。
