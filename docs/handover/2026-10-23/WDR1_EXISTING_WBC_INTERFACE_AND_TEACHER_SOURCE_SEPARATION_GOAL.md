# WDR1：复用已有WBC接口，澄清Teacher执行域，立即推进VLA–WBC解耦科学实验

2026-10-23 Asia/Singapore。Goal ID：WDR1_EXISTING_INTERFACE_SOURCE_TARGET_SEPARATION_V1。
状态：USER_DIRECTION_DEPENDENCY_CORRECTION_PENDING_NATIVE_ADOPTION。
用户本轮明确指令：“WBC接口我们不是已经有了吗？再说我们生产teacher数据也不依赖WBC执行情况而是WBC环境？仔细思考后发布新goal”。
审核基线：76a73b1956cf2e1c1e8790c003917f70aafbffa3；本计划只纠正此前将“缺目标task native qualification”误写成“缺WBC接口/Teacher不能推进”的依赖。它不是新的WBC开发项目、第二科研owner、额外GPU/PBS/physics额度，也不证明当前两域任务已闭环通过。机读任务卡同提交：reports/PI0294/design/WDR1_EXISTING_WBC_AND_TEACHER_PROVENANCE_TASK_20261023.json。

## 0. 立即生效的概念判定：不要混淆三种对象

**A. WBC接口本来就存在。** 仓库已有 TrajectoryChunkV1 -> GR00TToWBCAdapter -> WBCPreviewV1（9点空间preview＋独立GripperCommandV1）的外部参考契约（docs/contracts/WBC_EXTERNAL_REFERENCE_V1.md、src/go2arx5_vla/adapters/groot_to_wbc.py、src/go2arx5_vla/wbc/reference_source.py）。STAGE20S_NATIVE_WBC_CLOSED_LOOP.md 记录真实原生WBC执行：benchmark在98/100控制步内完成参考；另一个已接受的实际GR00T候选进入WBC后以TRACKING_ABORT在25/110步终止。STAGE21同场相机/receding-horizon实际运行过六次gated模型chunk，但旧视觉reach任务未获得全面任务资格。这是“接口存在、native曾执行”，不能写“整个WBC接口尚未交付”，也不能反过来写“本次T1/T2已成功”。

**B. Teacher产生示范可以使用WBC交付的资产/场景知识，而不执行目标WBC controller。** 现有独立PI0290使用 minimal MuJoCo归档只读参照44mm手指/TCP声明，但 Frozen WBC identity 明确NOT_USED；其MuJoCo独立动态工具受有限wrench和指爪驱动。PI0291、PV1、CR2、VAP1现有主训练数据为FIXED_BASE_ARX5_TASK_SIM：scripts/pi0291_arx5_data/scene.py 从交付 go2_x5.xml 派生独立fixed-base scene，移除freejoint和原WBC的旧actuator/sensor，使用真实ARX5关节IK/PD和独立有限驱动、真实MuJoCo接触、双相机与Teacher反馈；scripts/pi0294/scene.py、scripts/pi0294/pv1/capture.py、scripts/pi0294/cr2/runtime.py、scripts/pi0294/vap1/runtime.py 复用这一执行域。正确称“WBC相关资产/机器人几何衍生的独立source simulator示范”，**不是WBC原生策略执行的任务示范，也不是完整free-base WBC运动成功**。

**C. 与B不同，R4A full-body原生 worker是在冻结WBC实际执行下做任务pilot；目标域实验的真实机器人/控制器状态、物体任务和安全链必须按该原生执行器审查。** scripts/pi0294/task_native_worker.py及target_mujoco/reference.py已有原生交付和时钟/参考代码；当前仍需根据同一受检版本查明T1/T2任务资格。若将来自目标WBC的示范混入DEC1高层训练，就不能再把该训练结果宣称为target-controller-disjoint zero-target-task-demonstration。

**重要系统差异：** WBCv1.zip 对应历史IsaacGym/RoboDuet Stage1+Stage2，wbc_mujoco_minimal.tar.gz对应MuJoCo+OCS2+腿RL/臂PD。它们不是同一WBC换文件；旧9点空间preview、MuJoCo时间化target reference和控制器原生动作/时间布局不能互相强制套用。复用现有资产和桥必须标注具体controller_id、runtime和版本，不得因接口名称相似伪造native一致性。

## 1. 纠正执行依赖，而不放松科学与安全

- SOURCE_TEACHER_DATA、R0 A/B训练、源域B0/DEV，以及source-only接口CPU准备，不等待目标WBC的hold RMS PASS、新WBC权重、新EXECUTOR_MANIFEST、target任务成功或WBC负责人调参。原GR00T_TASK_DATA_FIRST_FROZEN_WBC_SCOPE_V1自2026-10-15已明确停止WBC保持偏差归因和参数修理，旧历史FAIL保留。
- Teacher数据准入仍必须满足当次仿真实际物体任务成功、原生安全、真实观测、实际提交的desired EEF/gripper标签、真实时间和split/全reader有效性。不可因跟踪不完美拒绝安全成功数据；亦不可将已达到的EEF轨迹冒充Teacher未来监督。源资产继承不等于可忽略物理验收。
- DEC1目标WBC真机以外的仿真评价则要求实际native controller、正确frame/TCP/gripper、回读state/clock、safe stop和固定物体评分。**已有接口的存在不自动证明当前目标task runtime qualified**。新目标是“重用并补齐可见的原生任务链”，不是“重新发明接口/等WBC改好跟踪”。
- 原CORR/R0权重与源域统计、WBC/plant/gains/solver/腿策略、夹爪PD、S1及任务评分不变。不能以本goal删去原raw/mapped完整H40与实际prefix安全检查、无视ACK/STOP，或用理想控制器替代target。

## 2. 立即复用的具体代码、证据与最小缺口

一次只读 inventory 仅检查以下固定类别，不要求重抄全部WBC:
1. 共同高层/bridge：docs/contracts/{TRAJECTORY_INTERFACE_V1,WBC_EXTERNAL_REFERENCE_V1}.md；src/go2arx5_vla/adapters/{groot_to_wbc.py,execution_progress.py}；docs/architecture/SYSTEM_ARCHITECTURE_V2.md；历史Stage20S和Stage21原生证据。
2. source independent teacher/executor：PI0290的src/go2arx5_vla/task_data/{scene,engine,teacher}.py；PI0291的scripts/pi0291_arx5_data/{scene,engine}.py；PI0294的scene.py、pv1/capture.py、cr2/runtime.py、vap1/runtime.py；D48/CR2真实完成报告和固定HF revision。
3. 目标native候选：scripts/stage21_native_receding_horizon.py与对应旧IsaacGym launcher；src/go2arx5_vla/target_mujoco/reference.py、scripts/pi0294/task_native_worker.py与现有R4A native角色/源身份。读取已注册的私有资产身份、原始执行收据，不能把仅有归档文件替代活跃runtime。
4. 为每个具体执行器只补一份“已有/待接线/未合格”小型证据表：type、actual source+asset/config/weights SHA、参考输入ABI/9点或native targets、frame/TCP/夹爪、时钟/feedback/ACK/stop、task scene/物体、native safety、reset和真实运行证据。已有字段引用固定旧收据，不重复索要整份新WBC包。
5. 结果分类必须至少区分 INTERFACE_EXISTS；HISTORICAL_NATIVE_WBC_EXECUTED；SOURCE_INDEPENDENT_TEACHER_DATA_READY；EXISTING_TARGET_ASSETS_PRESENT_TASK_NATIVE_QUALIFICATION_PENDING；TARGET_TASK_NATIVE_QUALIFIED。当前可据代码与旧收据认定前三种，不可无新现场证据宣称最后一种。

如实际没有可执行目标task bundle，返回唯一“delta-only”缺件清单（例如runtime版本或可达ROS/clock/socket/reset/安全回执），而非写WBC_INTERFACE_MISSING、要求重训师兄WBC、阻止source研究。

## 3. 两条科学工作线同步，但互不冒充

**G：GR00T源域训练（最高资源优先级）。** 复用已冻结源数据/模型与既有A/B配方。最后可证当前 A_CONTROL_HEARTBEAT_RECOVERY2 作业1477127.stdct-mgmt-02、source df6427f791fdbe8f175906f240eb3413bf096cbf，2026-10-23 02:01 SGT原生Q/node_pool；当前真实完整A恢复点2048，旧A2819只算历史花费，B0步。最终当前 grant 已合法采用 DEC1 9987累计更新、12新增PBS、418GiB GPU站点输出与全局485GiB；其余scope/时钟/费用/历史不重置。一个owner对同job轻量观察、认证、release和必要terminal结算，禁止重复qsub、热改旧capsule或从0重跑已保存状态。真正A4096 terminal与两端lease关闭后按现有准入首次提交独立B，B不从A继承。

**S：独立Teacher/source-physics任务评估与DEC1接口工作。** 现有固定数据已成，不追加物理采集：CR2=24/24合格，VAP1 D48=48 attempts/46 admitted/24 groups/22 paired，原139与original126+C1已在Vanda reader PASS且closed。S在CPU上处理已有source task-state闭环和接口资格，不需要WBC owner hold fix。任何新source CPU/物理评价仅沿现有有效PV1/DEC1 scope、实际资源及样本锁，若未准入报准确阻断。

**T：冻结WBC目标任务评价。** 复用已有已验WBC reference path和native runner，在明确目标runtime上按原有限commissioning验证T1/T2场景/夹爪/相机/物理/STOP，最小差异接入；不可将Stage20S visual reach PASS充作当前接触任务PASS。R4A历史hold RMS不作为source data全局前门，但target实际安全、状态和任务成功必须核查。T若缺目标task overlay，只使目标比较标TARGET_TASK_NATIVE_QUALIFICATION_PENDING；G/S继续。

既有两个辅助agent若已合法存在，只允许隔离CPU代码/资料检查，禁止取得原生mutation owner身份；一次报告具体可用接口/阻断后退场。禁止无限审查、不断写新“缺manifest”列表而不复用旧接口。

## 4. 源执行器与WBC目标执行器的ABI应不同，不强造一套WBCPreview

上层统一H40的真实语言+RGB+actual EEF/gripper输入与语义，源和目标下层接线允许不同：
- E_SOURCE是现有FIXED_BASE_ARX5_TASK_SIM，每20Hz提交原prefix8中的真实位姿/夹爪给原single-pose per-tick独立仿真执行器；先校验原H40与映射H40合法性，再沿原source S1精确对8点prefix做native clone/live-shadow或符合原审核的等价完整执行前预测，保存实际每tick、拒绝和重规划。此执行器的single-pose调用**没有**九点WBCPreview9的原生ABI，不得以“九点接口未实现”直接判已有source模拟器缺失。
- E_TARGET沿实际绑定的冻结WBC版本调用既有native外部参考：历史IsaacGym WBCPreview9或MuJoCo OCS2 reference-manager时间化轨迹；必须做显式SE3/rot6d行列、heading/frame、TCP/gripper、时间语义适配。不是把独立source的单pose S1当target安全；也不是把目标WBC的9点preview强迫source engine消费。
- 源 S1 raw/mapped+prefix实际安全和target对应真实native安全都保持硬门。scripts/pi0294/dec1/source_interface.py当前SourcePrefixShadow为NOT_SUPPORTED/fail-closed，是**当前DEC1原生source-prefix安全接线未资格**，不是“项目不存在WBC接口”的证明。其已有单pose predictor不能被声明为9点shadow资格；若最小原生安全接线尚不合格，不能执行物理，只能精准报告缺口。
- I_TIME与I_PROGRESS的唯一实验干预仍是用名义时间或实际EEF进度决定上层参考进度/夹爪事件；两者使用同一模型和各自执行器的相同native adapter/safety，不能为了证明收益调整WBC gains、task geometry、计时、阈值或oracle输入。

## 5. 先做真正能出科学结论的有限评价，不等待所有训练

1. 源域旧CORR B0：原DEV48已有14有效(9成功/5失败)、34未完成；SAVED_DEV48_REMAINING_CONDITIONS.json已CPU reconcilied，不等于原生补跑。该现有模型source-only评价在独立已获准资源下可以与A训练并行；保留相同noise/geometry、原有效结果，须满足原executor与平台数值一致性，不跨未合格GPU平台拼接。
2. DEC1核心固定原CORR@2048/R0而不是新A/B赢家，8独立heldout group×2 tasks×source/target executors×I_TIME/I_PROGRESS=64个任务格；只在两种executor的对应native ABI、安全、时序/feedback、封存场景和完整资源合格后启动原有最多8次commissioning及预注册对照，不在项目别的工作线上反复重采Teacher。若目标暂时没有task runtime，可先交source-only合法条件，但不能填充目标格或计算Transfer Δ。
3. 新A4096和B4096端点用于独立PV1 matched DEV48，按旧完成B0才做baseline-relative model selection/LOCKBOX。中间TRAIN probes只是模型健康资格，不作物体task成功率和checkpoint选优。
4. 新目标原生资格可先使用非物体commissioning定向校验旧bridge，不能宣称完整task成功。源端任务真实执行与目标端冻结WBC真实执行保持不同身份，报告跨执行器迁移包含身体/相机/动力学域差异，不声称纯WBC因果。
5. 若存在合法数据源但目标当期无法任务资格，交出独立source科学结果、确切TARGET_TASK_ABI_GAP和未运行64格，而不是让所有学生实验等待WBC师兄。

## 6. 防泄漏和数据目的防混淆

将实际各数据集合绑定 EXECUTOR_PROVENANCE 与 PURPOSE：
- INDEPENDENT_TOOL_MUJOCO（PI0290）: reference/physics derived, WBC policy NOT_EXECUTED；仅SIM_ONLY data。
- FIXED_BASE_ARX5_TASK_SIM（PI0291/PV1/CR2/VAP1）: frozen Go2-X5资产geometry来源、arm IK/PD独立executor、非原目标WBC策略执行；可以按原scope做源训练。
- FROZEN_FULLBODY_WBC_TASK_SIM（R4A等）: only if actual WBC forwards/controller/robot free base are proven; target-domain data若用于源训练则丧失“no target task demo”实验约束，故当前DEC1严格禁止回流TRAIN。
- PRETRAINED_OR_CORR_ROLLOUT / DEV / DEC1_CONFIRM: never teacher positive TRAIN without separate protocol; object oracle/Teacher phase allowed only teacher offline/score, not student inputs。
数据源与当前执行器同名不说明WBC版本/模拟器一致。已封存统计R0与R1不混；原A/B数据配方不变。Teacher成功率不等于学生策略成功率、controller robustness或自由底座成功率。

## 7. 最小交付、进度与界限

此Goal不增加授权资源、时间或动作。原总96 GPUh/24PBS、DEC1经最新native批准52GPUh/12新增PBS/9987累积updates/418GiB，全球存储485GiB，最终20GPUh与40GiB保护、原训练/物理/关账时点保持；真实used、reserved与native site权限必须从live ledger读取，不能把上一次预算投影当作当下空闲。
- 原停止时间：2026-10-25 02:29:56 SGT不再新作业/候选，14:29:56不再新训练，18:29:56不再新物理；2026-10-26 02:29:56 SGT hard close。2026-10-23 20:00 SGT需真实神经结果或精确阻断。
- 第一交付：WDR1_EXISTING_INTERFACE_EVIDENCE_AND_GAP.json（当前有哪几条桥、哪些历史native通过、task native尚差什么）；WDR1_TEACHER_EXECUTOR_PROVENANCE.json（当前独立Teacher/E_SOURCE/目标WBC的资产/模型/实际执行身份、数据目的，不重新导出视频）。
- 第二交付：最小可运行source S1 prefix/native安全接线+限量commissioning（在原获准物理资源内）；对应native PASS/FAIL。不能仅交空result模板或CPU绿灯冒充物理。
- 第三交付：若现有目标runtime task overlay真实合格，保留64格原预注册表及每格all-attempts、成功、S1/target safety、grasp/push/hold和几何组配对置信区间。若未交付，明确TARGET_TASK_NATIVE_QUALIFICATION_PENDING，不塞入假结果。
- G线同步交真正恢复A4096／B4096或准确原生终局；不得因WDR1阻断健康队列，也不得把目标接口验收变成source训练新门。

任何原资产、独立项目、第三方WBC包、原始RGB/权重/凭据保留在私有注册存储；GitHub只发本Goal和脱敏机器卡/后续真实摘要。单logical native owner、现有两个CPU helper槽位不扩张，GitHub管理HEAD不热切换现役胶囊。本文件只发布判断及已有scope下的步骤，没有SSH、qsub、模型、物理或新WBC安装效果；需原现场owner用WDR1_ADOPTED_OR_EXACT_DELTA_BLOCKER证明实际采用和下一native动作。