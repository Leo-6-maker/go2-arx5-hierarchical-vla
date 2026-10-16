# PI0294 E2：追加预算、持续科研推进与冻结WBC整机迁移 GO V1

**日期**：2026-10-17，Asia/Singapore。
**状态**：USER_AUTHORIZED_BOUNDED_SCIENCE_GO / PENDING_ACTUAL_SINGLE_OWNER_AND_SITE_RESOURCE_BINDING / NOT_LAUNCHED_BY_GITHUB。
**授权依据**：用户直接要求“授权更多预算的goal，跑通实验主线，更新长程计划”。本文件给出有总上限、自动条件分流的一次E2科学GO；不需要对同范围普通工程修复逐项再问人类许可。但GitHub发布不等于站点GPU/PBS/物理配额已获系统分配，唯一owner仍须核对现役leases、账户权限并登记一次真实T0和E2资源。
**固定输入基线**：现役分支审阅8fdb2a179f5db88c294fea4182c2d177a2e2801a；E0永久终局208175f9ccec24edcac4bb50629676c43f2bad3f；E1原GO c851cc791e1fbb3efb293f7075f0e771518cf5c0和其现役run不修改。配套长期路线 docs/roadmap/2026-10-17_GR00T_WBC_SCIENCE_LONG_HORIZON_ROADMAP_V2.md，机器卡 reports/PI0294/design/E2_EXPANDED_SCIENCE_AUTONOMOUS_TASK_CARD_V1.json，Codex接续 docs/handover/2026-10-17/E2_EXPANDED_SCIENCE_EXECUTE.md。

## 1. 采用时必须承认的新事实

E0：M2048在原v3.0/source固定底座D0+S1的11条有效开发评价均S1预测非法接触，TRAIN0/3、DEV0/8，另1条TRAIN中继工程中断；native live safety0，旧run终态CLOSED。
E1：真正训练出M1@1024（原SMALL8、原eef10、VLM冻结）并在预声明TRAIN F0/000 T1取得第一次**独立审计通过的32秒自主成功**；抬升约0.08354m、水平搬运0.17691m、最后41/41 hold、真实安全0、641双RGB；不是T1两布局资格、T2能力、目标WBC成功或DEV泛化。T1 source Recovery Oracle有两独立E0学生episode完整恢复；T2 0/2未获准；C0两独立episode/833去重窗口已实际消费，身份E0_STATE_AGGREGATION_ONLY，绝非新M1 online DAgger。E1@256原科学负例/工程中断与一次scope deviation保留；其余M1@1024、C1/M2和E1终态等待新现场/公开原件，不编造结果。
Vanda：已有被采用的A40/CPU原生构建job，最近已发表1462595 Q、1462438 Q、1462451 H，真实M2048+当前MuJoCo/ROS+冻结WBC fullbody闭环尚未PASS；TF-R3历史目标原生安全STOP仍然有效。
因此新E2的首要动作是**取最新E1终态/未完成条件、保留首成功证据并分流**，而不是“再从M2048开始无目的微调”或将旧Vanda STOP清零。

## 2. E2目标：成功优先、科学分母先于演示效果

最低可用科学成果：
- 源T1第一成功已经有证据，新阶段需使T1在至少2个预声明独立TRAIN布局和原S1下量化重复性，并实际报告T2的独立任务行为；模型选定后给出未被该阶段用于调参的DEV结果。
- 实际学生访问状态上的Recovery Oracle覆盖与可恢复率，至少对一个合格任务采集真正新学生动作前缀后的纠正数据；同数据预算对照普通BC与Action-DAgger。若某task的oracle不合格，跳过其DAgger不阻断另一task。
- 当前完整Go2+ARX5的目标执行平台至少通过真实新观测→模型H40→冻结WBC实际prefix8/ACK→新观测的完整一轮episode，报告真实安全及物体任务结局，不能以推理smoke/模拟数据/单侧接触替代。
目标扩大成果（不承诺可保证成功）：
- target T1/T2各自的真实successful teacher/demo生产管线；每个合格任务至少8独立TRAIN+2独立DEV真实成功、逐文件准入和独立数据消费者，先完成单任务就先交付；
- source-only vs 5/20真实target示范的GR00T迁移与任务成功率曲线；两任务由同一shared high-level policy，在同一冻结WBC和固定任务分集下评价。没有足量真实目标数据时明确NOT_RUN。
- 可选FlowDAgger或FB13/BASE独立研究，但仅在有真实反演合格/BASE非零标签时启用，不消耗前面科学关键路径必需预算。

不根据一次成功决定胜负，不将同一个episode的H40数目当独立示范，不为达到KPI调整原score和安全门。失败实验保留真实分母和可重复诊断。

## 3. E1与现役工程的边界

- E1原T0、8 H200 allocated GPU-hours、3 PBS、全部source积分/CPU/动作预算和2026-10-18 11:02:28SGT硬收尾不可变。已运行第一job670561、第二job670729最后公开R；E2不能代其追加预算、重跑E1有效负例、取消健康GPU job或抢共享source worker。
- E2 owner可以先做只读E1和Vanda库存准备；**第一次E2 GPU/新物理/模型动作执行前**，须现场证明对应E1 job/lease已终态，且任何尚活跃Vanda或CoreLab任务与所需GPU/CPU/ros-domain/仿真state/路径完全隔离。新E2 T0登记一次，不由首轮E1继续计数，也不清除E1账。完成E1的正结果优先作为E2来源，不另选更好seed。
- Vanda现役已采用1462438/1462451/1462595的job/额度先按旧绑定自然完成，E2的Vanda增量不得抢占或把旧累计计作E2新启动/新通过；新硬件/物理资格仍以真正现场结果为准。
- CoreLab TF-R3目标native safety STOP不因本E2预算自动lift。E2可以有**新的、版本化目标任务侧修正科学条件**，但每个候选先登记失败根因假设、隔离独立启动和手指/参考/碰撞/时序再资格，并明确不同于触发STOP的原运行版本；不得在旧停止的worker上直接恢复。任何WBC算法、weights/gains/cost/solver/腿策略、机器人mass/geometry/friction、原力/限位/illegal contact/评分阈值都保持冻结。
- source EEF10 stats和动作对GR00T的语义冻结。TARGET用不同可审核adapter调用冻结WBC；没有真实BASE标签时标BASE_PREFERENCE_ABSENT而不是新建三个监督零动作。EEF13训练仅有明确非零BASE真值时作独立分支。

## 4. 新E2预算：仅增量，按阶段配额，不无限轮回

全部是**E1之外新增的全周期上限**；未使用的预算不能用于重试有效失败或未登记的新科学因素。满足实际账户/调度规则时一次owner登记并自主执行，不需要对文档中阶段间同scope常规修复反复索要routine GO。若站点权限低于此包则采用较小有效配额，不许以聊天批准绕过调度配额。

|资源|E2整个周期硬上限|
|---|---:|
|Hopper H200|累计24 allocated GPU-hours，最多8次PBS、每次walltime <=4h、同时最多1张|
|Vanda A40|累计12 allocated GPU-hours，最多6次PBS、每次walltime <=3h、同时最多1张；旧已采用作业原样|
|CoreLab CPU|176 core-hours|
|Vanda活跃CPU|96 core-hours|
|Hopper活跃CPU|80 core-hours|
|local-PC CPU|16 core-hours；不得在登录节点跑神经/长期physics|
|各执行端活跃CPU|最多8核；计渲染、WBC/ROS、规划、reader、影子；优先单source worker及单target worker，第二仅隔离资格后|
|新增optimizer更新|<=6000，含所有新试验分支；不用更新上限掩盖真实GPU费用|
|新增神经动作推理请求|<=15000，固定真实任务/探测/反演分类记账；训练本身FM前向另记|
|source主仿真启动|<=64，包括资格/学习评价/DAgger/工程恢复；有效科学FAIL不以工程恢复重放|
|fullbody target新启动|<=96，包括启动资格、pilot、Teacher、生产、模型评价；只有合法新版本|
|native主仿真/恢复积分|source <=1500000；target <=6400000；shadow/反事实 <=8000000；所有积分合计<=15900000，先到任一限额停|
|磁盘新增文件逻辑峰值|Hopper<=160GiB、CoreLab<=80GiB、Vanda<=80GiB、local-PC<=8GiB；不清理历史备份或原始科研证据换quota|
|预留收尾|至少H200 2h、CoreLab 16 core-hours、Vanda 8 core-hours、整体最后24h只审计/归档，无新实验|

**分阶段可花上限**（均累计受总上限约束，提前完成不强制用满）：

|阶段|H200|A40|CoreLab CPU|Vanda CPU|新的科学执行|
|---|---:|---:|---:|---:|---|
|S1源学习和行为|<=6h|0|<=40coreh|0|TRAIN2任务、多布局、候选固定后DEV，不重新训练旧E1|
|S2真正source纠错、BC/DAgger对照|<=6h|0|<=40coreh|0|T1优先，T2 oracle合格才并入|
|S3 Vanda当前环境资格+目标WBC任务侧Pilot|0|<=8h|<=56coreh|<=64coreh|原safe adapter+目标资格、source-only、目标Teacher|
|S4目标成功bank、少样本适配和独立结果|<=12h|<=4h|<=40coreh|<=32coreh|target数据达标后才对0/5/20微调，不够则报告缺口|

总H200 6+6+12=24h；A40 8+4=12h；CoreLab40+40+56+40=176coreh；Vanda64+32=96coreh。新source64、target96为共用全周期计数，不能每阶段重置。任务一旦触发安全STOP不得通过转移阶段“刷新”科学条件。

**时钟**：owner核对E1旧job/lease与资源后登记E2唯一T0（不能追溯早于本文件发布/真实采用）。T0+10天关闭新的排队接纳，T0+21天停止一切新模型训练、物理和新科学条件，T0+28天硬收尾，最后24小时仅final审计和关闭lease。队列与实际GPU allocation分开，不能把空排队计为已用GPU，但不能借长期排队无限刷新T0。每7天一个只读里程碑汇总，不要求新的人类routine GO。

## 5. 条件自动衔接的科学Stages

### S0/E1只读交接
- 获取E1最新原生job/lease、M1@1024全预定TRAIN面板、C1与M2是否真正完成，确保原E1终态/负例/偏差及第一次T1成功身份被固定，不宣称一个TRAIN成功即完成源技能。
- 所有旧DEV真实负例已是历史公开结果，不得用作E2新模型调参。冻结候选后方可进行对应锁定DEV独立评价，并披露同家族历史E0 DEV已用于了解基线的限制。
- 若E1仍在运行，CPU静态研发可在隔离路径且不影响原资源下继续；E2无需抢先消费GPU。

### S1：source BC科学能力先成立
- 以E1已完成且固定模型为真实基础，先比较M1和有凭证的M2；T1将第二独立TRAIN布局作为固定确认，T2重点分析模型contact/approach/gripper动作与真正source成功示范的差异。源数据训练使用原v3.0和TRAIN only。新的训练改动每次只改一个因素，优先可证伪的task/phase-balanced采样或追加新source TRAIN合格示范，不直接启动全栈VLM解冻、WING或RL。
- 若TRAIN成功但DEV弱，独立采用v3.1/视觉与语言覆盖分支，保持新数据版本/新stats与等预算对照。若源TRAIN也不行，固定少样本fit/codec/模型输入和控制历史诊断；不能对同一个已证实S1负例无限换seed重试。
- 每个任务至多预登记若干独立布局测试，结果按attempt/valid/engineering分母报告。一个source任务合格先进入后续，不等待T2。

### S2：Recovery Oracle、标准DAgger与可选FlowDAgger
- 复用T1真实2个E0 student episode的C0，但**不能叫online DAgger**。用当前学生新实际访问状态进行S1合法动作→专家同状态建议/接管，记录student实际执行prefix、expert建议和专家接管真正执行的分母；完整H40及末尾mask、actual观测/动作/ACK和任务状态来源验证。不把oracle privileged truth放入学生。
- T2 oracle当前0/2不合格，允许新source task-side recovery方案至多3个实质不同版本，固定任务事件状态及真实安全门；成功前T2不能利用“专家建议”生成伪纠正BC。T1合格不强求T2同晚合格。
- 匹配普通额外成功示范BC与真实student-visited Action-DAgger的专家query预算、TRAIN布局、optimizer更新和模型权重；报告不同episode、窗口、介入比例、真实性及任务成功。FlowDAgger真实反演需要用固定obs的GR00T effective model action以及forward reconstruction数值验证，之后才允许训练steering；有真实更重要主线未完时应跳过FlowDAgger。
- 无法收集独立学生纠正anchor则只交BC及oracle真实结果，不宣称完成DAgger。

### S3：Vanda+CoreLab目标整机执行镜像与source-only
- 原Vanda建造jobs不重复qsub；Stage0 A40/GPU显存、Stage1 M2048推理与原native WBC编译成功后，固定ROS/Jazzy、MuJoCo/EGL、WBC/plant/legs、夹爪、相机、同一20Hz/H40/prefix8/500Hz语义。在Vanda主机合格镜像试完整新观测→GR00T→原SafetyGate→冻结WBC→ACK→新观测，不能仅推理相同保存帧16次就称闭环。
- 与CoreLab冻结参考采用前定合法state/reference及输入身份比较，数值差不强行选有利主机；不同主机若不能证明可比，各按不同实验域报告。TF-R3旧安全STOP不予重启；针对新目标task-side变体先限定root-cause/真实startup gripper/参考资格与原安全无修改，最多3个实质新方案，累计目标start共用96。
- 一个**新**真实目标native安全失败立即停止该worker/版本及相同科学输入；确认独立因果及修复后才可重资格不同版本。E2内部累计第二个新native safety fail时停止全E2新的target物理和采集，转为现有合格source与数据审计；保留旧TF-R3 STOP历史，不以E2计数另造“旧失败不存在”的叙述。
- source-only固定M1/M2可用版本通过原EEF10任务接口、明确BASE preference ABSENT。只测真实原物体/安全，WBC hold RMS只诊断，不当任务硬门。若target全域无安全执行资格，报告阻断并让source研究按原预算完成，不强行迁移。

### S4：目标成功示范→少样本适配→独立评价
- Teacher能从实际目标状态推进并最终通过原task评分，按每task至少8独立TRAIN+2 DEV真实成功和真实H40 EEF/夹爪/BASE如有的标签生产；task通过安全、clock、actual 2RGB、cmd/ACK、原native评分和split才可入success BC。首2–5成功可启动adapter小样本诊断，不用等双任务大包。零成功时不制造目标模型训练资格。
- target DAgger仅在目标Recovery Oracle能从实际学生状态安全完成任务后方可进行，source C0资格不能跨域；新的EEF13/BASE分支需**真实非零BASE动作监督**，源EEF10的三维BASE标签ABSENT与发送零BASE soft preference不同。当前预算首先优先近场EEF10，不要求BASE突破先成功。
- Hopper按同一source checkpoint作source-only、source+5target、source+20target（仅真实拥有该数量时）的独立微调与闭环对照。固定eval scenes/语言/WBC、训练更新和processor/stats，不能按DEV结果选数据或调参；少量demo学习曲线需要独立episode而非H40窗口。最终先报告各task和各host物理结果，再报告整体。
- 数据 bank须真实独立loader成功，区分模型预测的未执行专家plan与真提交命令的BC监督、真实动作达到与高层请求、S1预测风险与live fault。

## 6. 安全、资源、可自动继续/停止

- 正常预执行S1拒绝是科学结果，不构成全包停止；有效科学FAIL不可用工程恢复次数再执行。真实native joint/contact/gripper/clock安全违规或live-shadow完整性破坏停止相关实验分支；如已发生上述第二次新增target native fault，停止整个E2 target物理；不能更换主机/worker/WBC绕过。
- 不修改冻结WBC、原source S1阈值/评分、原物理对象尺寸/摩擦/机器人质量、原夹爪合法范围、不跳过接触或伪造label。方案仅涉及GR00T数据/训练、高层task/skill/geometry/scene合法版本、source/target适配接口、ROS/数据同步/任务侧重规划。
- 用户一次批准的是本协议的**有限上限**。同scope日志、容器、PBS失败恢复、数据codec修正可以按来源保留、在该上限内自动修复，但绝不能盲重试未知神经/动作/原生job或放宽安全；任何新增学习算法、真实机器人、VLM全量解冻、大型人类视频预训练、未批准目标场景/机器人改造需另立新科学GO。
- 不让一个任务/平台永久阻断另一合法科学工作。Oracle不合格则DAgger分支停、BC继续；Vanda不可用则CoreLab保留主目标参考并按既有资格，source训练继续；targetTeacher无成功则不训练伪target数据而交source结论。
- 每阶段按真实任务分母、独立次数、主动模型推理/optimizer、actual physics/S1 shadow、原native安全与任务结果做单页阶段报告；不得用GPU占用/数据准备/降低loss/单触点/首TRAIN成功宣称全部Goal完成。

## 7. 交付与终局

最少文件：E2_ADOPTION_AND_RESOURCE_BINDING、PHASE_LEDGER、MODEL_AND_DATA_REVISIONS、SOURCE_BEHAVIOR_MATRIX、ORACLE_QUERY_AND_RECOVERY、DAGGER_DATASET_AND_COMPARISON、VANDA_CORELAB_WBC_ADMISSION、TARGET_TASK_LEDGER、TARGET_DATASET_AND_SPLIT_CARD、SOURCE_TO_TARGET_ADAPTATION、RESOURCE_FINAL、FINAL_REPORT、RESUME_IF_BLOCKED。所有真实文件仅提交脱敏摘要/hash到GitHub，raw RGB/video/权重/vendor WBC/隐私配置私有。
**明确不同科学终态**：SOURCE_TASK_EVIDENCE_ONLY、SOURCE_BC_DAGGER_COMPARED、TARGET_PLATFORM_CLOSED_LOOP_ONLY、TARGET_SINGLE_SKILL_DATA_CLOSED、TARGET_DUAL_TASK_DATA_CLOSED、TARGET_ADAPTATION_COMPARED、PARTIAL_WITH_EXACT_BLOCKER、STOP_NATIVE_SAFETY、RESOURCE_CLOSED。可以在这些状态中诚实终止，不能把PIPELINE_CLOSED用在未通过真实训练/物理/data consumer的任务上。selected/deployment恒NONE，本协议不进行模型投产。

站点owner在实际E2 T0后按同一固定release接续，普通同范围工程错误自主修并累计费用，不在每个阶段重新向用户请求GO；第7/14/21天小型里程碑可用于判断资源配置，但不能刷新28天截止。**发布该计划并未在本对话启动任何H200/A40/PBS/physics/GR00T推理或训练。**
