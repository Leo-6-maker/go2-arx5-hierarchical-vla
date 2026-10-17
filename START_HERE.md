# Current Entry: E1 Results and Single MP1 Execution

**Latest MP1 native outcome:** recovery children 01/02 are both CLOSED;
671817 F/1 and 671826 F/0 cost 453 conservative H200 seconds, with zero new
updates. Four audited attempts still give no valid full policy result:
3 engineering-incomplete conditions / 5 NOT_RUN, not policy 0/8. An actual
30.301976s Hopper RPC accounting stall crossed the unchanged 20s watchdog.
No third recovery-family PBS; do not train around this fault.
[Timing and qualification](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_REAL_RELAY_QUALIFICATION.json)
and [full attempt/condition matrix](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_E1_M2_RECOVERY_BEHAVIOR_MATRIX.json).
Follow the [evidence-first sprint entry](docs/handover/2026-10-17/MP1_EVIDENCE_TO_SCIENCE_SPRINT_EXECUTE.md)
under the original MP1 T0/budget; independent saved-T2/Vanda lanes remain.
The first subsequent [saved-T2 CPU audit](reports/PI0294/20261016T191445Z-localpc-mp1/M1_T2_SAVED_PROGRESS_AND_ACTION_AUDIT.json)
is complete: both objects entered the original goal zone before rejection,
but TCP clearance remained below 8cm. No new physics/model call, no complete
T2 success or exact geom attribution is claimed.

**2026-10-17 actual owner update:** E1's three PBS jobs and both leases are
terminal/CLOSED, with 19,675 conservative GPU seconds and zero reservations.
M1@1024 retains autonomous 32s T1 success in both seen TRAIN layouts; both
T2 layouts are valid original-S1 rejections. M2-BC/CORR each completed 512
updates with sealed C0, but their autonomous comparison remains incomplete
because of relay engineering failures: five incomplete conditions and three
NOT_RUN, not policy 0/8. No fourth E1 submission or closed-lease restart.
[Actual terminal summary and evidence](reports/PI0294/20261016T150228Z-localpc-e1/E1_RUN_SUMMARY.md).
Repaired C1 was independently consumed and is available for a separately
bound MP1 branch, not hot inserted into E1.

**The user adopted the single MP1 incremental package** at
`c8cdf60e41e31da5796b44a064ef2f22d594a491`. E2 is non-additive reference only.
The same owner registered MP1 T0 once; existing E1/Vanda jobs, original budgets,
negative results and native safety STOPs are preserved. No new MP1 GPU/model/
physics work is established by registration.

Read [CURRENT_SCIENCE_STATUS.md](CURRENT_SCIENCE_STATUS.md), the
[fixed MP1 execution entry](docs/handover/2026-10-17/MAINLINE_ACCELERATION_MP1_EXECUTE.md),
and [actual adoption receipt](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json).
Native refresh takes precedence over the historical snapshots below: Vanda
hardware1462438 and saved-input inference1462451 are F/exit0 with CLOSED
leases; native CPU build1463204 passed. These are not EGL/native WBC or
new-observation closed-loop qualification. Source generalization, T2
competence and target transfer remain unproven; TF-R3 STOP is preserved.
Real-robot deployment is NONE.

## Historical Entry Snapshots (Preserved)

**Vanda 独立闭环建设已获用户 GO，当前环境构建及 GPU 资格排队中。** [实际执行/失败修订及作业状态](reports/PI0294/vanda-bringup-20261016/CLOSED_LOOP_PROGRESS.md) · [执行协议](docs/protocols/PI0294_VANDA_CLOSED_LOOP_BRINGUP_V1.md)。用户指定 WBC 归档已三端 SHA 通过、六项原核心输入匹配；首 CPU 构建因站点容器挂载目录失败，修订后独立 job1462595 已排队。WBC/GR00T真实闭环及任务成功仍 NOT_RUN，不更改 Hopper E0、CoreLab原参考/旧失败或训练。

# GR00T＋WBC — 2026-10-16 E0 与后续代码准备

**E0已实际停止并收尾：PARTIAL，12/12尝试、11/12科学可判定，TRAIN0/3、DEV0/8成功。** [完整终报](reports/PI0294/20261016T083821Z-localpc-e0/E0_FINAL_REPORT.md) · [逐条件结果](reports/PI0294/20261016T083821Z-localpc-e0/E0_MODEL_BEHAVIOR_RESULTS.json) · [原视频索引](reports/PI0294/20261016T083821Z-localpc-e0/E0_RAW_EVIDENCE_AND_PRIVATE_VIDEO_INDEX.json) · [资源终态](reports/PI0294/20261016T083821Z-localpc-e0/E0_RESOURCE_FINAL.json)。11条原S1预测非法接触而预执行拒绝；1条本地中继中断工程不完整，不算策略FAIL、不重跑。12条保存证据/实际视频审计PASS，真实安全故障0；217生成、41000主步/42075影子步、1652双RGB。两PBS均F，原生1708GPU秒/保守1723秒，预留0、两lease CLOSED；无新训练/WBC/S1变化，selected/deployment NONE。以下排队/运行文字均为阶段历史。

**E0最后一次同语义工程提交已实际进入Q：670345.hopper-m-02。** [恢复绑定与提交收据](reports/PI0294/20261016T083821Z-localpc-e0/E0_ALLOCATION_02_SUBMISSION.json)，runtime固定ce1a14a56ddfe9f322e159eef8a2f8054da22525，两端封存入口PASS，15原数值文件及模型generate逻辑不变。2/2 PBS已用，原窗口/总额保持；当前尚无模型结果，以下首次失败/旧排队记录保留为历史。

**E0最新原生增量：首个allocation已F，启动前认证过期，模型/物理均未开始。** [首次实际终态](reports/PI0294/20261016T083821Z-localpc-e0/E0_ALLOCATION_01_TERMINAL.json)：183 GPU秒原生、188秒保守扣账，12条件仍NOT_RUN而非策略失败。正在修复同语义准入并准备唯一剩余工程提交；原T0/截止/总额不重置。下段排队措辞为此前提交阶段历史，不是实时状态。

**E0已采用并实际提交首个PBS，当前仍在排队，不是评价完成。** [本次独立绑定、固定12条件及实际提交收据](reports/PI0294/20261016T083821Z-localpc-e0/README.md)。只评价原v3.0/EEF10/source D0＋原S1的M2048，先TRAIN4再DEV8；TRAIN成功率不作新增门。runtime固定9a699d28c6a585df1abb5aa906e41e8d8c025bdc，报告/测试HEAD不热替换它。旧16项NOT_RUN保留，selected/deployment NONE。

**用户另行授权等待期间准备FlowDAgger代码，不授权新科学执行。** [实际代码/CPU准备报告](reports/PI0294/flowdagger-preparation-20261016/PREPARATION_REPORT.md) · [协议](docs/protocols/PI0294_FLOWDAGGER_PREPARATION_V1.md) · [机器卡](reports/PI0294/design/FLOWDAGGER_PREPARATION_TASK_CARD_V1.json)。独立纠正数据合同、原EEF10 processor桥、冻结模型inversion/steering组件及CPU测试已准备；FD1真实可恢复性、FD2真实模型重构和FD3训练未运行。不能把专家全接管默认例子称作学生状态DAgger，也不将E0 DEV转训练或零填BASE。新代码不部署到在途E0，不重开旧Teacher/WBC任务。

**完整GPU等待期Q0–Q6准备已按新增固定handoff接续。** [CPU准备终报与全部小收据](reports/PI0294/design/gpu-queue-dagger-prep-v1/PREP_FINAL_REPORT.md)：67项定向fixture通过（含2项复用BASE测试），默认827PASS/14显式SKIP；离线Oracle可以INVALID、chunk交易/ACK与split血缘已测试，合成状态不进真实训练。真实学生状态恢复、M2048反演和DAgger训练仍NOT_RUN。初期测试未约束CPU亲和性、历史并发不能认证，后续已固定2核/单线程并披露资源核算限制；不掩盖该偏差。

以下2026-10-15及更早条目是各自阶段历史；最新执行状态以固定收据、#11/#47及原生run/lease为准。

# GR00T＋WBC — 2026-10-15 历史科学入口

**TF-R2已执行S7至真实接触，因接触后夹爪原生安全FAIL停止；尚无成熟Teacher或成功数据。** [本轮报告、500Hz时序图、独立CPU复算与2018原件私有回读](reports/PI0294/tf-r2-20261016/REPORT.md)。用户取消累计资源上限后，执行同源seed0/PC2的0.95s闭合请求；实际6385包、20安装、156双RGB。5.85→6.80s闭合前后至首次接触均无越界；7.662s记录左pad承力接触，7.768s左slide首次低于原0mm，7.770s原registered joint safety停止，最小−0.177230mm。PD全步复算误差0N，其余19关节无越界。source隔离、task物理停止门已实际拒绝；按TF-R2第6节不进入S8/S9。成功seed/TRAIN/DEV/正式consumer仍0，WBC/安全/评分/旧FAIL不改，selected/deployment NONE。下列夜间/R1等条目为历史，不因本次预算更新被改判。


**TF-NIGHT已到原生共享启动安全停止点，数据闭环未成功。** [夜间终局、22步实际calf/ctrl图与私有回读](reports/PI0294/teacher-factory-night-20261016/FINAL_REPORT.md)。T0 2026-10-16 00:23:27SGT；采用354e45ab独立scope。T2原生启动22步/.044s原registered joint safety停止，FL_calf=-2.77245＜-2.7227；Teacher提交/任务RGB/成功TRAIN/DEV均0，T1未运行。未证明task-side可修根因，按协议5/11停止新物理，无追相位重跑、无WBC/安全/score修改。3581原件私有逐文件回读；owner/worker终态、预留0；selected/deployment NONE。下列准备/采用/V3进展为历史记录。

**V3已实际运行至合同停止点：真实单侧物体接触已取得，完整成功seed仍0。** [报告、实际TCP/接触图及662原件私有回读](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/source-demo-access-v3/REPORT.md)。12CPU实际PASS；T1运行5.6s/14安装/113双RGB，右pad真实接触累计.570s；下一未安装窗口原关节加速度4.01285＞1.0门拒绝。独立500Hz另见右slide6点raw超44mm、峰值44.06553mm；无原生stop≠原始机械范围PASS，不能作为成功示范。T2两入口10.346/10.467＞10s、两dense预算用完，001/002未认证/未评，未运行。累计40启动/364111扣账步；整体PARTIAL，停止本V3，无重试/训练，旧FAIL/WBC/安全/评分/窗口不变，selected/deployment=NONE。下列采用/待执行均为历史阶段记录。

**完整V3已获用户明确采用，含累计搜索217bundle/214214dense重分配；沿原窗口/父账执行。** [源绑定人类采用](reports/PI0294/design/SOURCE_ACCESS_V3_HUMAN_ADOPTION.json)。保留V2安全FAIL，新版本实施受影响CPU契约→各task自身源/actual warmup→完整32s；采用不代表运行PASS。下列待采用描述是此前发布阶段记录。

**V3已准备、待明确采用：T1完整任务组件抬高3cm到H47；T2固定首contact直接接入。** [完整有界方案](docs/protocols/PI0294_TABLE_CLEARANCE_AND_CONTACT_ENTRY_SEED_V3.md) · [任务卡](reports/PI0294/design/SOURCE_ACCESS_V3_TASK_CARD.json) · [保存几何/源前缀准备证据](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/source-demo-access-v3-preparation/PREPARATION_REPORT.md)。旧5300步包络含margin最高.399445m，新完整桌底.42m留下20.55mm保存净空；不证明新动态安全。T2三源223/231/223前无pad接触/推送，保留15.50/16.25/16.10s后缀，显式working-closed初态。拟各32s，≤4启动/42012积分/1282双RGB/14000CPU秒，沿原窗口/累计父账，内部累计分配明确217bundle/214214dense、仍在原外层预算内。**本次只准备/发布，authorizes_execution=false、launch_allowed=false、runtime未实施、变更CPU契约0/12，新IK/RRT/渲染/物理/神经/训练0；V2安全FAIL永久保留。**

**V2已实际执行至明确安全停止点，未取得成功种子。** [完整报告、实际TCP图与私有归档收据](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/source-demo-retime-join-v2/REPORT.md)。20CPU检查通过；T2局部时间18.6/19.4/19.1s但三接近端无合格IK；T1锁定000后实际运行5.6s、14安装/113双RGB，原生base_link几何与桌面ILLEGAL_CONTACT。任务物体接触/抓持/成功seed均0，两条完整32s均未完成，不报0/2完整成功率。636原件私有逐文件回读；旧FAIL、WBC/安全/score、时钟/账不变；本scope关闭，不重试、不训练，selected/deployment=NONE。下列采用/未实施是历史阶段记录。

**V2完整方案已获用户明确采用，原窗口与累计预算不变。** [采用收据](reports/PI0294/design/SOURCE_RETIME_JOIN_V2_HUMAN_ADOPTION.json)，采用固定`5a0cceeeb7e79f6f326a5aa17168a453acb65180`。当前接续版本化时间表/首close接入与实际CPU准入，尚未取得V2物理/种子结果；下列“待采用/未实施”是此前发布阶段记录，不能覆盖本次采用，也不能用采用代替执行PASS。

**下一版已准备、待采用：T2局部减速＋T1首close直接接入V2。** [完整有界方案](docs/protocols/PI0294_SOURCE_LOCAL_RETIME_AND_DIRECT_JOIN_V2.md) · [任务卡](reports/PI0294/design/SOURCE_RETIME_JOIN_V2_TASK_CARD.json) · [实际只读准备](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/source-demo-retime-join-v2-preparation/PREPARATION_REPORT.md)。保存数组显示T2保留所有源点、只局部延长超速区段时cartesian时间下界为18.60/19.40/19.10秒；T1原首次close索引30仍张开且有旧合法goal，拟以实际自由连接替换原30个非接触align点、保留原均匀减速及完整抓放后缀。连接/关节/碰撞/动态成功尚未验证。沿原窗口/累计预算，拟直接各32秒、≤4启动，无新短门或FSM阈值搜索。**本次只准备发布，launch_allowed=false，runtime未实施，V2正式CPU准入0/20，新搜索/物理/训练均0。** 原V1负结果与旧WBC/安全/score保持。

**自有示范重定位已执行至明确科学停止点：六个固定TRAIN来源均未准入。** [完整候选/独立复算/归档终局](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/source-demo-retarget-v1/REPORT.md)。T2三源uniform slowdown后需50.49/53.87/61.88秒，超过28秒；T1三源接近端均无满足原限位的登记解，接触端有解但不能组成pair。16项CPU检查及6源提取完成，3bundle/6目标IK/96CLIK，新增RRT/物理/RGB/模型/训练均0，真实接触与成功seed未取得。终局`STOP_FIXED_SOURCE_POOL_NOT_ADMITTED`，整体仍PARTIAL；不是WBC物理FAIL或全局IK无解。75文件私有完整回读，旧FAIL/WBC/安全/评分/窗口/累计账保持，selected/deployment=NONE。下列“待采用/未执行”均为历史阶段记录。

**自有示范重定位完整方案已获明确采用：** [源绑定人类采用收据](reports/PI0294/design/SOURCE_DEMO_RETARGET_HUMAN_ADOPTION_V1.json)，固定采用e7d6537facc3c670cdd197e64302eac913965534，原窗口/父账不重置。执行从真实源提交片段/工具映射接续，WBC、安全、原评分和旧FAIL保留；采用不代表新CPU/规划/物理已PASS。下列“待采用”是此前发布阶段记录。

**固定小样本已下载；自有成功片段重定位方案已准备、待采用：** [实际下载/来源资格](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/source-demo-preparation-v1/PREPARATION_REPORT.md) · [完整有界方案](docs/protocols/PI0294_SUCCESS_DEMO_SEGMENTS_TO_FULLBODY_SEED_DRAFT_V1.md) · [机器卡](reports/PI0294/design/SOURCE_DEMO_RETARGET_TASK_CARD_V1.json) · [采用prompt](docs/handover/2026-10-15/SOURCE_DEMO_RETARGET_ADOPTION_PROMPT.md)。四payload共338566923字节完整SHA通过，6条原TRAIN源身份/成功/成员确认；外部包仅作参考库，不阻塞自有来源。拟直接T2/T1各32秒，≤4启动/42012积分/1282双RGB/14400CPU秒，168bundle及内部IK/RRT/dense逐项封顶，沿原截止/累计余额、冻结WBC/安全/评分，不再修FSM阈值。**本次仅下载/准备已获准并完成；新重定位搜索/物理未授权未执行，launch_allowed=false，运行适配尚未实施。** 原MP-S2真实结果与未启动辅助槽保持，采用新方案后才在停止边界独占接续，旧FAIL不改。

**MP-S2已进入真实物理，当前PARTIAL（并非仍待采用）：** [实际结果与独立CPU审计](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/maniskill-search-v2/PROGRESS.md)。16CPU项通过；T1原首帧无合法规划对，T2完成12秒但无接触，短里程碑有效FAIL；两完整AUTO均未准入。原记录器故障保留，241组RGB从相同保存实际状态恢复并收费；真人辅助尚未启动。新成功整机seed/训练均0，旧FAIL/窗口/预算不改。另已完成[公开成功示范来源与重定位审计](docs/reviews/2026-10-15_SUCCESSFUL_DEMO_SOURCE_AND_RETARGET_AUDIT.md)，锁定官方metadata/revision/hash，未下载轨迹或启动新来源实验。以下“待采用/未实施”保留历史阶段措辞。

**下一版已准备、待采用：MP-S2有界候选搜索＋交互示范备选。** [完整方案](docs/protocols/PI0294_MANISKILL_BOUNDED_SEARCH_AND_ASSISTED_SEED_V2.md) · [机器卡](reports/PI0294/design/MP_SEARCH_V2_TASK_CARD.json) · [接续prompt](docs/handover/2026-10-15/MANISKILL_SEARCH_V2_ADOPTION_PROMPT.md)。每task6几何包、approach/contact成对、单bundle≤12目标IK/192内部初值/8RRT；内部失败继续固定剩余槽。单task独立准入，完整slot互斥走AUTO或真实人工ASSISTED，失败后不换路线重跑。保留原窗口/父账、≤6启动，WBC/安全/最终score冻结，不循环修FSM阈值。**本次只准备发布，authorizes_execution=false/launch_allowed=false；新搜索/物理/seed/模型均0，功能未实施。** 以下旧系列终局永久保留，新方案不自动重开它。

**ManiSkill组件本轮已到有效规划资格停止点：固定T1/T2起点均无满足原限位的登记IK候选，物理未准入。** [实际结果、独立原生FK复算与私有归档回读](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/maniskill-component-v1/REPORT.md)。24项CPU接口检查通过；12个固定保存输入中7个后续输入可规划、5个失败，不能挑后续帧替换两个失败起点。新物理/接触/成功seed/训练均0，终局`STOP_FIXED_START_NO_ADMISSIBLE_REGISTERED_IK`；不是CPU_READY、WBC任务FAIL或全局IK无解。原WBC、安全、评分和旧终局未改，selected/deployment=NONE。下列采用/待采用措辞保留各自历史阶段，不代表继续运行的许可。

**ManiSkill组件方案现已获人类明确采用执行。** [源绑定采用收据](reports/PI0294/design/MANISKILL_COMPONENT_HUMAN_ADOPTION_V1.json)，采用固定13713019077cb128b55090ab6b7a513e162b975e；原窗口/累计预算、冻结WBC/安全/任务评分不变。将依固定CPU接入→两任务短资格→各自完整32秒推进，采用不代表运行或seed已PASS。下列“待采用”是前一发布阶段记录。

**新增已准备/待采用方案：ManiSkill规划组件→整机W0真实接近/接触/成功seed。** [完整有界方案](docs/protocols/PI0294_MANISKILL_PLANNING_COMPONENT_SEED_QUALIFICATION_DRAFT_V1.md) · [机器卡](reports/PI0294/design/MANISKILL_PLANNING_COMPONENT_TASK_CARD_V1.json)。冻结WBC/原T1/T2评分，替换旧cursor与误差阈值阶段前置；采用固定MPlib IK/RRTConnect、隔离NumPy环境。拟2短＋独立条件2完整、最多2任务前工程替换，共6启动/59018积分/1764双RGB/6CPUcoreh，花原父账与绝对窗口；无GR00T/GPU/PBS/扩采。用户本次只授权准备和发布，authorizes_scientific_execution=false/launch_allowed=false；未安装、未启动新物理、未取得成功seed。下列V2是已有实际终局，不因新方案改判。

**最新实际V2已闭合：3条有效12秒短里程碑FAIL＋1条任务前就绪工程不完整。** 无预接触/物体接触/操作，条件两条完整确认未准入；三条角误差最小约0.143/0.097/0.171rad、平移游标仍零。原件2323文件/530436177字节逐文件私有回读通过，旧终局未改，成功整机TRAIN0、训练0、selected/deployment=NONE。见[完整结果、CPU复算、图与终账](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/component-teacher-v2/REPORT.md)。

**后续准备：优先复用成熟规划组件，冻结MuJoCo/WBC。** 已固定读取ManiSkill八个原始文件并核对Panda joint动作、PickCube不含释放/撤离成功条件、实际ARX5与全Go2碰撞资产差异。见[源码到EEF13的具体适配审核](docs/reviews/2026-10-15_MANISKILL_PINNED_SOURCE_TO_EEF13_ADAPTER_AUDIT.md)。仅源码/接口研究，未安装新框架、无新物理/模型/GPU/PBS许可，没有本体成功seed；不再循环补游标阈值。

以下V2“草案/未采用”与V1“最新”措辞是历史阶段记录；后续人类已明确采用V2并得到上述实际终局，不覆盖或改写历史原件。

**最新零物理步补证：44次保存cursor更新严格重放一致，其中34次位置检查在请求平移为零时提前拒绝。** 已准备[分量独立推进V2单项草案](docs/protocols/PI0294_COMPONENT_PROGRESS_V2_QUALIFICATION_DRAFT.md)，CPU fixture／44个独立单步提案检查通过；这是改变Teacher准入合同的未采用草案，authorizes_execution=false／launch_allowed=false，不重开前轮物理或预算。完整示范目标仍未完成，旧终局不改。

**最新已采用的新整机Teacher短系列已终结：2条有效无进展FAIL＋2条时间表示工程中断。** 4个短条件各启动一次，19462含warmup／尾部积分、380实际双RGB；无预接触／物体接触／操作。两条完整确认未准入、成功整机TRAIN0、训练0。CPU已证明并窄修10.2秒负龄期浮点表示，但修复后未再物理运行。保存查询的持久游标未进入目标平移段；Teacher绝对位置包络与旋转推进耦合是具体下一设计问题，未转向WBC调参。原件1302文件／440579732字节私有完整回读，旧0/4／FAIL不改，selected/deployment=NONE。见[实际结果、图与终账](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/persistent-teacher-v1/REPORT.md)。本系列入口已关闭，原科学总目标仍PARTIAL；不把工程中断算科学FAIL，不无条件重跑。

以下为新系列采用前的实际数组准备记录，未授权描述对应当时提案，不覆盖后续直接人类采用及上述已执行终局。

**接近阶段已有真实数组复算：三条位置条件均0/641，推送姿态条件已568/641、561/641成立。** 同刻上一安装参考与实际TCP中位差约3.4mm，但实际距持久阶段目标最终34.8／40.8／46.1cm；原50ms重锚定与prefix8声明不一致，MPC消费身份仍UNKNOWN。见[时间对齐报告和图](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/approach-audit/REPORT.md)。单项持久接近Teacher组件只有CPU草案／10项测试及真实保存输入dry run；新物理、训练0，launch_allowed=false，不因草案或审核自动开放新pilot。

**最新实际任务结果：固定四个整机pilot成功0/4。** V1近场抓放原生安全FAIL保留；V2其余三条完成32秒、真实双RGB／参考／时序及独立CPU一致性通过，但物体任务均FAIL。成功数据0，v4.0／FB13未准入；未运行新训练。已封存全部原件／私有视频及逐文件回读，冻结WBC，不重复有效失败。见[完整结果与收口收据](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/FIXED_PILOT_RESULTS.md)。执行完成与任务成功分开，整体数据—训练—闭环仍PARTIAL，selected/deployment=NONE。

**最新用户决定：不调整WBC，只借用完整环境采集任务数据和训练GR00T。** [主线纠偏与Codex接续规则](docs/protocols/GR00T_TASK_DATA_FIRST_FROZEN_WBC_SCOPE_V1.md)（内含机器规则）。R4B D0已结束，D1及继续控制器归因/补偿/调参不再属于本团队下一步。WBC保持不变。

**新采集采用“真实物体任务成功＋原生安全＋数据完整性”准入；旧保持RMS质量FAIL保留，但不再作为全局数据采集/训练的前置阻断。** 不调高0.05rad、不删除旧FAIL、不忽略实际限位/非法碰撞。应补齐任务夹爪/相机/Teacher，直接检查原样环境能否完成抓放与推送，再制作EEF13数据、并行采集和微调。脚本存在不等于任务或数据已经通过。

本方向已由用户明确，不新增任何资源总量或窗口；owner需绑定有效任务/资源scope，不能自动重开关闭的lease。以下是已发布的事实与历史阶段依赖；其中的“依赖链已停止”描述原阶段终局，不代表新任务还要等待WBC修复或保持精度PASS。

---

**R4B D0实际只读归因已完成：同控制周期FK与力矩证据支持手臂伺服实现偏差主导姿态误差，并有身体预测/实现耦合；尚未证明可自主修的实现bug。** [WBC负责人缺陷包](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/R4B_D0/WBC_OWNER_DEFECT_NOTE.md) · [只读终账](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/R4B_D0/RESOURCE_AND_FINAL_REPORT.md)。新增物理/模型/PBS均0；R4B D1仍未采用，不新开窗口。

**最新实际结果：R4A 原生完整机器人完成10秒世界系EEF保持，姿态RMS0.06965rad超过原0.05rad门；时钟、参考安装及原安全检查通过。整机教师/数据/FB13依赖链已停止，尚无整机学习模型。** [实际报告与CPU收据](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/FINAL_NATIVE_QUALIFICATION_REPORT.md)。原始失败保留；这是有效负结果，不是R4A整体完成，selected/deployment仍NONE。

**最新用户方向：完整Go2＋ARX5环境生产数据和测试，GR00T增加有界BASE速度意图，经原生BASE软位姿接口接入冻结WBC；工程问题在明确范围和既有资源内自主修复推进。**

- [R4A完整长程计划](docs/protocols/PI0294_R4A_FULL_GO2_ARX5_AND_BASE_INTENT_V1.md)
- [EEF13／BASE软参考接口契约](docs/interfaces/EEF13_BASE_SOFT_REFERENCE_V1.md)
- [R4A机器任务卡](reports/PI0294/design/R4A_FULLBODY_TASK_CARD.json)
- [现有Codex直接接手Prompt](docs/handover/2026-10-15/PI0294_R4A_EXECUTE_PROMPT.md)

读取根AGENTS、最新#11/#47与原生运行状态后采用。最新已核验模型是M2048；R4旧v3.1增量在v3.0统计下codec失败保留。固定底座实验是基线，不再是完整整机搭建的永久前置。已运行作业不得热改或重复提交；新scope只使用已授权R4剩余资源和窗口，不新增预算、不重置T0。完整WBC在CoreLab仅作冻结推理/仿真执行器，高层GR00T仍在合法Hopper allocation；不授权WBC重训、实机或历史删除。

完整整机、EEF13与BASE桥在有真实执行收据前是待实施，不因文档发布变成PASS。下列2026-10-14及更早的“当前状态/未授权/未推理”是历史接管记录，不应覆盖后续已发生的R2/R3/R4事实或本次范围增补。

---

# GR00T＋WBC — 唯一仓库与 CoreLab 工作入口（2026-10-13）

**新 Codex 与所有后续工作请先从这里开始。** GitHub 是本团队 VLA 代码、协议、测试、小型执行收据的版本权威；CoreLab 共享 `~/Projects/GR00T-WBC` 是人类统一工作入口，但不能把私有原始数据、WBC、权重都视为已保存在 GitHub。

- **[2026-10-14 新 agent 完整接管与科学推进计划](docs/handover/2026-10-14/NEXT_AGENT_EXECUTION_PLAN.md)** · [机器任务卡](docs/handover/2026-10-14/NEXT_AGENT_TASK_CARD.json) · [直接接手 Prompt](docs/handover/2026-10-14/NEXT_AGENT_BOOTSTRAP.md)：当前 PI0294 仍因真实夹爪原生限位 BLOCKED，修订待 GO；这是计划不是自动执行授权。今后交接由 GitHub 固定提交直接供 Codex 读取，不以本地 ZIP 为主。
- **[2026-10-14 完整工作总结与下一 owner 交接](docs/handover/2026-10-14/HANDOFF.md)**：Oct12-14 已执行事实、代码入口、固定 HF 资产、终局与剩余预算。当前 M512 实际闭环触发原生夹爪关节限位，科学线 BLOCKED；交接不授权重启。夹爪重资格方案仍未获明确采纳。
- **[现行统一入口／GitHub 同步规范](docs/operations/2026-10-13_UNIFIED_ENTRY_AND_GITHUB_SYNC_V1.md)**：host-aware profiles、固定源 SHA、运行 manifest、分支同步、旧目录退役验收。
- **CoreLab 现有可执行入口**：`~/Projects/GR00T-WBC/infrastructure/enter.sh amd-management`（AMD），或在 4080S 使用 `4080-simulation`／`4080-cpu-data`。不提供自动跨主机 SSH；不经许可启动物理／模型。
- **[最新真实维护终局](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47#issuecomment-6137981521)**：已实际退役33源路径；Projects仍7项，有受保护科学依赖。后续统一入口操作只凭新执行收据计为完成。
- **[科学主线 #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11)**：PI0292 以其独立 run capsule、source seal 和当前资源账为准，不要用当前分支 HEAD 自动替代已在执行的源码。
- **Git**：实际活跃科学分支 `pi0267/executor-decoupled-recovery`；`main` 是较早的稳定导航，不是当前仿真或训练启动源。只有作业边界+doctor通过，才允许新 `code/vla` checkout fast-forward，同步本地不改历史 run。

以下是 **2026-10-12 的历史接管文档**，保留供追溯；其中的“尚未授权实验／旧最新状态”不再作为当前实时状态或新 GO。

---

# 先读这里｜2026-10-12 新local-pc接管

**当前是接管与仓库维护，不是新的训练/仿真GO。**

## 唯一入口

- [完整local-pc接管prompt与双服务器/HF规范](docs/handover/2026-10-12_LOCAL_PC_ADOPTION.md)
- [分支清理allowlist及原SHA](docs/handover/2026-10-12_BRANCH_RETIREMENT_PLAN.json) — 尚未实际删除；local-pc先建立兼容tag并回读，再按精确lease删除四个祖先分支。
- [科学ledger #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11)
- [可复用代码/CI/仓库维护 #47](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47)

## 最新用户方向

我们负责独立任务仿真、可信专家数据与上层GR00T的EEF/夹爪行为学习。师兄WBC仅作冻结执行器，不再由上层团队调参修复。TaskAnchor/UNITAS只是设计参考，尚未实现。本轮只读接管、条件化分支维护与下一数据GO草案；不恢复旧实验、不启动新模型或物理。

| 位置 | 职责 |
|---|---|
| local-pc | 唯一owner；分别SSH两端、GitHub审核交互；未来可作闭环obs/action中继 |
| Corelab | 仿真/教师数据/预处理/物理推进/CPU评分/视频，非高层GR00T模型eval |
| Hopper H200 | GR00T训练微调、模型推理与模型eval，须合法PBS allocation |
| GitHub | 代码、协议、有限预算、小收据、审核 |
| HF | 数据/权重/大证据，固定revision、prefix和hash |

两服务器互不连通；不要尝试直连scp/ZMQ。HF用于大资产，不是每0.4秒的实时控制队列。闭环需要当前动作造成的新观测；local-pc双SSH中继尚需另行验证，离线回放不能冒称闭环。

## 最近完整科学状态

PI0285源Panda确认20/32未过26/32。PI0286 C0 TRAIN2/8；C1 TRAIN8/8、DEV5/16；未运行新确认。PI0289安全启动PASS，但首hold姿态RMS0.0652rad>0.05、时钟残差47.8ms>25ms；其余五参考NOT_RUN，目标教师数据0。最近ledger终局是2026-10-10 11:08:34 SGT；接手须刷新最新证据。selected/deployment=NONE。

现役代码分支pi0267/executor-decoupled-recovery。main仍是旧accepted代码＋导航，不能直接当训练runtime。旧README、hpc gate、10月8–10日deadline与PR77不构成当前执行授权。

第一轮交付：ADOPTION_REPORT、RESOURCE_MAP、ASSET_AND_HF_MAP、NETWORK_AND_EVAL_PLAN、实际BRANCH_MAINTENANCE_RECEIPT、NEXT_DATA_GOAL_DRAFT（未授权）；发#11回读后停住等新科学GO。
