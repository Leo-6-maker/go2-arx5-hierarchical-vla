# GR00T＋冻结WBC：科学主线长程路线 V2（2026-10-17）

**用户决定**：新增有界预算，以真实科学进展为先，目标从源示范与任务空间学习，完成闭环行为、DAgger纠正、完整Go2＋ARX5冻结WBC迁移和少量目标数据适配。**这是科学目标，不是任何物理/模型结果已经完成的声明，也不是无限预算。**
**审阅基线**：现役分支 pi0267/executor-decoupled-recovery，已发布头8fdb2a179f5db88c294fea4182c2d177a2e2801a；用户当前时区Asia/Singapore，2026-10-17约02:10。运行实时状态只认唯一owner站点原生凭证，GitHub快照不等于SSH在线核验。
**配套**：新的PI0294_EXPANDED_SCIENCE_AUTONOMOUS_GO_V1协议、机器卡及2026-10-17 Codex handoff。路线与授权卡分开：长程愿景不自动扩大机器卡额度。

## 1. 已证明成果与未解决的科学问题

- E0原M2048在v3.0固定底座D0＋原S1真实评价12/12尝试，11有效S1预测非法接触拒绝（TRAIN0/3、DEV0/8），一条工程中断，真实native安全fault0；原E0已CLOSED。不是0/12有效失败。
- E1当前run 20261016T150228Z-localpc-e1、T0 2026-10-16 23:02:28SGT，原36小时硬收尾2026-10-18 11:02:28SGT；原8 H200 allocated hours、3PBS、2048 optimizer更新、源物理/CPU/存储上限全部保持。最新公开第二job 670729.hopper-m-02仍R；本路线不得重开/加额E1、重试其有效负例或热改active capsule。
- M2048→M1@1024已真实训练1024次Flow Matching更新（原v3.0成功SMALL8、EEF10、VLM冻结）。**第一条完整自主T1成功已独立审计PASS**：M1@1024 CANONICAL F0/000 T1，32s、16000积分、641真实双相机帧、原S1拒绝0/native安全0，抓握、抬升8.35cm、水平搬运17.69cm、释放和最后41/41次hold通过、无专家接管。证据：reports/PI0294/20261016T150228Z-localpc-e1/M1_1024_FIRST_T1_INDEPENDENT_AUDIT.json，#11 comment6221769022。**仅已见TRAIN的单次成功**，不等于另一布局、DEV、双任务、Go2＋WBC、真机、部署。
- 源Recovery Oracle T1在两个不同E0学生episode真实安全状态完整恢复2/2；T2完整恢复0/2，NOT_QUALIFIED。4个tick0短fork因遗漏静态IK历史判为INVALID，不入数据；其余8个非初始短状态及4个完整恢复保持exact history。已封存C0来源纠正6段/两个独立学生episode、833个去重H40窗口，Hopper原processor/CPU collator实际读取，身份E0_STATE_AGGREGATION_ONLY，**不是M1在线DAgger、不等于833条独立episode、不是目标WBC数据**。
- E1@256仍有三条有效负例和一次relay工程中断；TF32上下文不同造成数值可比性限制，曾误启动一条有效负例重复并单独保存协议偏差；历史负例/费用不得洗白。E1@1024其余条件、C1和M2最新公开尚未终局，必须由owner刷新结果。
- Vanda A40已有WBC六项核心源码输入对CoreLab哈希一致，旧GPU硬件1462438为Q、推理1462451等待afterok、CPU容器编译1462595为Q（最近发表2026-10-16 21:22 SGT）；新MuJoCo/ROS＋M2048＋冻结WBC尚无真实完整闭环PASS。旧A40 Stage43/IsaacGym运行不冒充当前系统。CoreLab完整fullbody TF-R3旧native安全STOP仍须尊重，尚未证明任何目标完整成功seed。

## 2. 学术主张分层

架构：真实双RGB＋语言＋可测EEF/夹爪/BASE状态 → GR00T生成H40任务空间EEF/夹爪和可选BASE软意图 → 原SafetyGate → prefix8实际执行 → 冻结WBC与独立夹爪物理 → 下一真实观测。高层不直接控制腿关节/力矩；下层控制器代码/策略/plant/安全/score冻结。

H1 来源域任务学习：模型能否从示范学到真正完成抓放/推送的闭环动作？T1初证已成立一个场景，T2和泛化仍待测。
H2 交互分布纠正：真实学生偏离状态上的有效专家H40能否使Action-DAgger较同等数据量/训练预算的普通BC减少拒绝、提高独立任务成功？
H3 跨执行环境与少样本：同一来源学习权重在free-root Go2＋ARX5冻结WBC上的source-only表现，以及0/5/20条目标真实成功示范带来的改善。
H4 可选FlowDAgger：仅在真实M1动作反演↔正向重构一致时，比较冻结GR00T steering与常规动作模型微调。Microsoft LIBERO 7D映射不等于本项目EEF10/EEF13；其示例默认全专家chunk不能被误称真正on-policy DAgger。
H5 独立EEF13/BASE研究：移动意图必须有真实非零BASE监督；source EEF10缺BASE标签不能填3个零冒称可移动策略。
创新不能仅声称“BC/DAgger/Flow Matching组合新颖”；重点是task-space与冻结wholebody执行分离、目标数据效率和安全可恢复性交互的实证。

## 3. 长程依赖与顺序

### L0 E1先原样收尾
以原T0/预算完成其余预定TRAIN行为、C1纠正生产和M2对照（如现实可行）。即使后续E1失败，已审计T1成功和C0不回退为不存在。保存M1/M2各自checkpoint、真实分母/数值精度/训练上下文、旧scope deviation、数据manifest和原生资源。新预算不替E1恢复旧缺失负例，不能争抢当前活跃H200或共享source worker。

### L1 来源域BC与两任务学习
保留原v3.0 TRAIN/DEV/stats/processor/EEF10的immutable baseline，固定候选与训练/scene/噪声条件。T1再检验另一个TRAIN布局；T2分析真正物体推送/接触阶段和模型动作拒绝，不因T1已成功就称双任务。可以在独立新分支有界使用source TRAIN成功示范扩展、任务/阶段重采样、独立几何损失probe，禁止一次同时改变数据、损失、模型和评估。v3.1如用须独立版本、训练统计与匹配对照；不能热混M2048模型现有stats。决定模型仅用TRAIN诊断，不用原E0 DEV结果调参；锁定以后再做原DEV/独立holdout的一次性泛化报告，披露开发集曾被怎样使用。
科学成果为T1/T2各至少两个预声明不同TRAIN布局的完整结果（成功和失败均归档），然后完整模型源domain DEV报告。一个任务先通过即可进入对应后续阶段，另一任务继续探索。

### L2 来源Recovery Oracle与DAgger
T1 C0已为状态聚合提供真实已执行H40训练来源，但尚非新学生在线样本。先记录M1真正学生产生的合法prefix与被拒绝决策点，合格源Oracle查询同一actual state，提供完整合法H40或INVALID；原S1拒绝动作不安装。按学生/专家/未执行建议/实际ACK及原生物理分别存储；阶段和物体真值仅oracle可见，不得漏给student。
T2 Recovery当前0/2不合格：可在有限task-side动作重规划上修复source oracle并独立核验；不合格时继续普通T2 BC而不是构造虚假T2 DAgger标签。
对照普通BC继续、同专家查询预算下额外成功示范BC、student-state Action-DAgger，并量化自主task success、prefix S1拒绝、专家query/correct率、独立学生episode、原技能保持。只有有真实student-visited纠正来源才叫on-policy DAgger；C0只能叫E0 state aggregation。FlowDAgger是可选后续、非全部项目的额外前置。

### L3 Vanda合格镜像与source-only完整整机
现役Vanda构建/排队原任务继续；只有其资源/模型加载/ROS+MuJoCo/EGL/冻结WBC及真实时钟/夹爪原生资格通过后，才在新独立运行capsule试M2048/M1固定EEF10高层source-only转移。未学习BASE3时明示BASE意图ABSENT，不能称EEF13-trained/移动能力。记录原参考、实际腿/臂/物体、真实双RGB/状态与S1/原目标native安全。Vanda与CoreLab用事先固定合法输入/环境版本做数值/行为一致性；如差异不可接受，单独平台报告，不选择有利结果覆盖CoreLab。
TF-R3旧安全STOP不因新GPU/新预算自动恢复；新目标物理科学条件须独立完成task-side原因界定、startup/gripper/碰撞安全重资格，原WBC/腿策略/plant和阈值冻结。目标语义成功与低层RMS诊断分开，WBC hold RMS不另作总前置。
工程闭环PASS需要：真实新观测→同一模型H40→原安全→原WBC真实prefix8与ACK→新观测，完整episode保存。目标任务SUCCESS还需独立物体评分+安全/完整时钟与最终保持；不得把单段接触称完整成功。

### L4 目标成功数据→少样本GR00T适配
程序化teacher、合格source技能重定位或在原高层接口上的有界人工仿真示范均可提供候选，**必须在目标冻结WBC真实执行成功**才能进入target SUCCESS_BC。分task开发/训练/DEV独立scene family，记录数据/command/ACK/actual/评分和原高频安全；失败独立failure bank，未执行oracle候选不算成功标签。
目标每任务先2–5条真实成功证明来源；再努力达到8 TRAIN＋2 DEV且通过CPU训练loader＝该任务数据管线闭环；长期扩展0/5/20条/任务的配对迁移曲线需真实样本不足时跳过，不能重复一个episode窗口冒充独立示范。T1先成立就先训T1，T2无需成为所有训练的前置。
Hopper同一来源模型做source-only、source+5target、source+20target的独立分支；同数据stats/action encoder及增量更新预算要版本化，DEV不参与选择训练超参。模型真闭环对照用同一个冻结WBC、场景、评分与安全，若Vanda/ CoreLab硬件差异不能忽略，分别报域。

### L5 独立BASE／论文级泛化
仅在实际目标有非零BASE软速度专家动作+匹配观测之后训练新EEF13／FB13分支，absent与zero必须分离；没有则本阶段NOT_RUN，不影响前面EEF10近场科学成果。配对视角、布局、语言反事实、目标示范档位和算法消融报告；只有不同真实机器人/执行器的严格比较才可声称强cross-embodiment。真实部署仍需另行授权，不在本长程goal内。

## 4. 科学停止规则与报告身份

保持E0/E1及TF-R3全部既有历史negative/STOP/预算与peer在途作业，不force push、热改immutable capsule或复用旧run UUID。正常source S1拒绝属于有效策略失败，不应自动停止其他已声明条件；真实native安全fail或shadow不一致立即停止对应新物理/纠正，旧已经安全合格source离线研究仍可完成。新任务原因修复只能是task-side动作/接口/teacher，不能调WBC gains、接触规则、力限、腿policy或评分。
未达成T2 Oracle/target teacher/FB13时，把相应目标标BLOCKED_WITH_SPECIFIC_CAUSE；不向后虚构数据和任务成功，也不让其他任务长期等待。长程预算是新的**有限**授权，具体H200/A40/CPU/物理/科学条件上限和期限见E2机器卡。所有真实消耗按站点原生收据计费、提早资源耗尽直接收尾；GitHub计划发布不意味着任务已启动。

## 5. 统一交付物与退出标准

- MODEL_LEDGER（M2048、M1、M2、source/generalization/target各模型原stats/processor/权重/seed/精度/branch）、TRAIN_DATASET_CARD和raw来源哈希。
- BEHAVIOR_MATRIX（每T1/T2/场景、真正物体任务完成、S1、真实native safety、接触/抓/推/搬/放阶段，成功/有效失败/工程中断/未运行独立）。
- ORACLE_RESULTS（查询和完整恢复两套分母、分布状态、INVALID、真实学生/专家前缀/未执行建议分离）。
- DAgger_C0_C1_NEXT DATASET_CARD（独立episode数量与相关H40窗口数量分开，真实CPU/独立消费者及泄漏审计）。
- TARGET_WBC_PLATFORM_ADMISSION（原冻结plant/WBC/场景身份、Vanda和CoreLab对照、不可比事项）、TARGET_SUCCESS_BANK（各task独立TRAIN/DEV合格数）、TARGET_ADAPTATION_MATRIX。
- RESOURCES_AND_SAFETY（真实PBS/allocated GPU、Core/Vanda/Hopper CPU、source/target主物理/影子、等待、存储、原生STOP及lease关闭）、FINAL_HONEST_REPORT。
- selected/deployment仍NONE；未经不同目标评价/明确审批不自动选择、部署模型，绝不将开发TRAIN单次成功写成生产策略合格。

**最终研究终点**：同一GR00T任务空间策略在来源域真实闭环具备可测能力，纠正学习的增益有受控对照，且在冻结WBC目标整机中以source-only和/或少量目标示范真实运行；双任务及非零BASE作为更高阶目标。所有未成功阶段均交确切实验分母及下个可证伪缺口，优于无证据宣称全线跑通。
