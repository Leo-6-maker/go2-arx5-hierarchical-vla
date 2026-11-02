# 当前入口 · Goal V7 · 2026-11-02

2026-11-02 18:09 SGT Core实测：**D1完成4096更新／131072访问，1024／2048／4096全状态恢复、独立trace／指标审核和CLOSED完成；GPU1自动完成96诊断后原DEV为0成功／1有效、9未运行，18.346秒joint4=-1.672124rad低于原-1.666rad停止。** TRAIN16 H40位置51.97→3.679mm、Prefix8 41.85→2.990mm，不等于完整行为成功；命令关节解未记录，根因仍待审计。**固定D2的PB两臂已真实挂载：Core常驻PID3718781、分离连接同身份／心跳前进，等待现有D2收尾后各一次native10资格→fresh4096；新物理评价尚未挂载，需独立RESET资格。** 旧D2无计算等待角色已保留原状态并关闭，仅控制生命周期结束，不虚报D2完成；共享owner已实际接管，当前GPU0仅746MiB空闲，正常等待。原Phase1/4、D0 0/1和全部STOP保留，现役配方／外部任务／基础设施未改。[真实端点、负结果、队列READY及边界](reports/PI0294/design/controlled-finetuning-20261102/D1_COMPLETE_NEGATIVE_DEV_PB_SUCCESSOR_READY_ACTUAL_20261102.json)。

2026-11-02 17:25 SGT Core实测：**D1同身份继续3716/4096、118912次TRAIN访问；1024／2048检查点均有42项全状态恢复。共享GPU0后继同PID3077247／InvocationID保持心跳，仍等待健康生产者，不依赖PC或Vanda。** 已实测D1与外部GPU0消费者并存并继续前进；不声称已证明吞吐隔离。原D1评价→D2有限资格／匹配4096及原评价已经在服务器队列；PR317新增RESET／PB／四任务训练尚未挂载。CPU核实D2原15932窗，并对uniform／阶段控制／额外边界各枚举131072次访问：首16接管窗曝光918／1055／9196次，专项直接配额8192不能当总曝光；三组均覆盖所有原窗口，未新增数据／模型／物理。新采样尚未绑定Trainer或通过原生训练资格；现役配方不变。[运行与挂载范围](reports/PI0294/design/controlled-finetuning-20261102/EXECUTION_ADOPTION_ACTUAL_20261102.json)；[实际CPU索引与计划曝光](reports/PI0294/design/controlled-finetuning-20261102/D2_PHASE_BOUNDARY_EXPOSURE_CPU_ACTUAL_20261102.json)。

2026-11-02 15:02 SGT Core实测：**D1同一身份继续到1420/4096更新、45440次TRAIN访问；1024检查点及42项AdamW／scheduler／RNG／cursor32768完整恢复证书已独立读回。** 用户明确设置GPU0共享，已移除后继“必须无其他GPU0计算进程”的额外条件，保留实际24GiB入口、36GiB可用主存和原训练2GiB余量。**共享后继已实际常驻并通过两次分离连接同身份前进心跳；仅替换了无计算、未持有mutation owner的旧等待程序，训练／GPU worker均未重启或收到信号。** 当前仍等待健康D1完整关闭，再自动完成未运行评价和D2有限资格／匹配4096及评价；GPU1可选，GPU0回退不依赖可选离线指标，已有有效面板不重跑。旧等待进程最终CPU统计未保留，保留上界并独立核清，不虚报实测费用；控制效果均已核清。D0原DEV0成功／1有效、9未运行与Phase1/4保留，D2尚无正式4096／有效DEV。共享许可不等于已证明零带宽／吞吐影响，当前训练持续前进。[实际共享队列READY](reports/PI0294/design/science-goal-v7-20261031/GPU0_SHARED_FUTURE_CONTINUATION_READY_ACTUAL_20261102.json)；[原负结果／数据与接管点分析](reports/PI0294/design/science-goal-v7-20261031/MATCHED_FLEXIBLE_EVAL_RESIDENT_READY_ACTUAL_20261102.json)。

2026-11-02 00:26 SGT Core实测：**Phase4096已独立恢复真实1成功/4有效、6未运行；原评分/STOP不改，旧训练、双TRAIN32与DEV均CLOSED。** GPU1真实10更新及42项全状态恢复完成，但最低空闲仅0.57GiB<1GiB，正式训练不准入，采用GPU0串行。新采集/数据阶段常驻manager已通过两次独立连接同身份心跳核验，首个柜门新TRAIN完整安全成功并完成官方导出，累计柜门2；第二个正在运行。**READY仅覆盖采集与数据冻结，完整BC/DAgger训练接续尚未部署，不宣称整条DAG可断线完成。** [实际核对、容量及阶段READY回执](reports/PI0294/design/science-goal-v7-20261031/ISSUE299_PHASE_RECONCILED_GPU1_NATIVE10_SOURCE_DATA_READY_ACTUAL_20261102.json)。

2026-11-01 17:37 SGT实际挂载：**Core源采集后继READY_FOR_DISCONNECT通过，PID1969815在分离控制连接后保持同一身份、独立心跳前进；当前仅等待整条旧训练—双TRAIN32—原DEV关闭。** 原正式训练同一身份1551/4096、49632次TRAIN访问，1024已保存/恢复，采样不改。后继顺序为柜门1→条件通过再3、持夹恢复2→通过再6、新几何放书2→通过再4，最后原生数据导出/联合统计与既有Rerun配置；不自动新训练。CPU原生绑定资格通过，采集注册/新物理/新示范仍0。旧资格失败与费用保留；完整历史账户接续须冷态实际通过，READY不保证未来作业全部成功。[实际常驻接续、资格与边界](reports/PI0294/design/science-goal-v7-20261031/CORE_AFTER_PHASE_SOURCE_COLLECTION_NATIVE_CPU_READY_ACTUAL_20261101.json)。

2026-11-01 17:14 SGT实测：**phase1024已保存并原生恢复42项AdamW/model/scheduler/RNG/cursor32768；同一正式训练已至1281/4096、40992次TRAIN访问。** 同输入/噪声TRAIN16与uniform1024比较，H40位置9.026→8.507mm、Prefix8夹爪MAE0.04533→0.03296；Prefix8位置均误几乎不变，home与部分搬运窗口反而变差，不能称完整任务改善。4096仍为主DEV端点，不改现役采样。**新采集接续仍未通过原生资格、未ARM**：已确认旧源账户约342MB；PC相同历史验证峰值RSS由约5.38降到2.09GiB，冷态CPU反而增加，修复未部署。无新增采集注册/物理/合格示范；全部资格失败效果已独立核清。原服务器训练—双诊断—DEV链继续有效。[1024实际对照与采集接续阻断](reports/PI0294/design/science-goal-v7-20261031/PHASE1024_MATCHED_TRAIN_AND_SOURCE_LINEAGE_COLD_BLOCKER_ACTUAL_20261101.json)。

2026-11-01后继采集CPU实测：**phase4096同一正式训练已至980/4096、31360次TRAIN访问，当前训练—双TRAIN32—原DEV10接续不变。** 新增真实持夹事件接管/仅专家后缀导出、6张独立yaw±3°或抓点±2mm TRAIN卡及既有冷态采集顺序已实现；6/6新书名义整段静态路径通过，3类保存TRAIN持夹状态初始化和MuJoCo完整状态非修改捕获通过，尚无新增动态资格/示范。柜门原4张TRAIN卡保持；10张候选DEV静态8过2失败，归因为大开角把手—底座相交，保留候选失败，不能冒充模型泛化成绩。**新Teacher后继尚未挂载**，原服务器链已挂载；现役数据/统计/模型/队列不改。[后继准备、静态结果与部署边界](reports/PI0294/design/science-goal-v7-20261031/NEXT_SOURCE_COLLECTION_HELD_RECOVERY_GEOMETRY_CPU_ACTUAL_20261101.json)。

2026-11-01阶段采样实测：**BOOK_PHASE4096_V1已在Core真实启动，最近53/4096更新、1696次TRAIN访问；服务器接续READY_FOR_DISCONNECT通过，同一manager在分离控制连接后存活并自主心跳。** 原生10更新资格完成并恢复42项AdamW/model/scheduler/RNG/cursor后CLOSED；正式模型重新从原b248初始化，数据12533窗与book统计不变，连续夹爪不变。训练完整关闭后自动串行uniform4096与phase4096的固定TRAIN32×3噪声诊断，再仅新4096原DEV10；安全STOP保留未运行，不重跑旧面板。另核对现有T1/T2各16条成功导出，PC选各10条形成42条目/25154窗的新共享候选；新统计未用于现役训练，柜门仍只有1条真实成功、4张新增TRAIN卡未执行。新增正式完整端点/有效DEV/自主成功仍0/0/0。[实际启动、断线生存与数据复用](reports/PI0294/design/science-goal-v7-20261031/BOOK_PHASE4096_NATIVE_STARTED_RESIDENT_READY_AND_REUSE42_ACTUAL_20261101.json)。

2026-11-01后继实测：**1024二值夹爪独立诊断0/1成功、9未运行，12.892秒link6接触书架停止；258条已提交命令全为OPEN，未建立双侧抓持，不能把二值化当成已修复。** 保存握持状态的匹配反事实仍精确支持“卸载导致失抓”，但不代表从home到释放的完整任务已解决。柜门已有示范完成官方Processor601窗口资格；共同TRAIN统计候选通过四任务24条目/14336窗口原生编解码及12样例联合批次，无CUDA/模型/优化。候选不替换原书本统计，也未准入共享训练或独立DEV。下一训练干预选择阶段分层采样，尚未实现/启动；不同时加分类头、不加长book-only链。当前相关GPU运行均已CLOSED。[解码负结果与共同处理链实际资格](reports/PI0294/design/core32-maintenance-20261031/BINARY_GRIP_AND_COMMON_FOUR_TASK_CODEC_ACTUAL_20261101.json)。

2026-11-01实际接续：**2048原DEV为0/1成功、9未运行（11.794秒非法接触停止）；4096为0/1、9未运行（2.968秒手指速度停止），两者均已审核CLOSED。** 独立1024夹爪反事实精确复现原前缀与短窗、全状态恢复重复结果逐值一致；保持原EEF，只换专家闭合目标，两条0.4秒窗口均200/200步双侧接触，末端物体相对漂移从53.67/276.08mm降到0.193/0.091mm。不是完整任务成功或合格示范。二值解码诊断代码通过受影响检查，原生结果尚未取得；不新增长训练。所有失败诊断费用保留。[原评价与匹配干预实际结果](reports/PI0294/design/core32-maintenance-20261031/UNIFORM_CURVE_REMAINING_AND_MATCHED_GRIP_COUNTERFACTUAL_20261101.json)。

2026-11-01保存记录归因：**1024两项均出现新规划块夹爪目标超过实际握持开度，接触力减弱，随后持续失去双侧接触**（6.106/7.736秒）；前项书架接触6.494秒在失抓之后，后项无先行书架接触、8.074秒落地安全停止。20秒2.445m/s拒绝位于未执行H40尾部零基索引36→37，Prefix8内部最大0.110m/s，不能称实际执行超速。两DEV仅同场景±6mm位置偏移，距最近TRAIN2mm，属于几何插值；跨资产/背景/任务泛化及共享DiT遗忘尚未测量。4096拟合改善不等于其行为已验证。[500Hz失抓机理与泛化边界](reports/PI0294/design/core32-maintenance-20261031/CURVE1024_FAILURE_MECHANISM_AND_GENERALIZATION_20261101.json)。

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
