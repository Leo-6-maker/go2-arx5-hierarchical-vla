# SLA_Memory × Go2+ARX5 × OCS2：B2-D1 至 B4 自主推进长程实验协议 V1.5

**2026-11-23 · PLAN_ONLY / NEW PHYSICS NOT_AUTHORIZED。** 本文依赖 [V1.4 实验 PR #433](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/433) @ `cec17aa41b29f6b6b04b2f8d25e17d87e74227e4` 的 `B2_STATIC_BIAS_ATTRIBUTION.md/.json`、`B2_NATIVE_TECHNICAL_CARDS.md`、现场真实二进制与V1.2/V1.3封存源。原V1.3学生H40/G7有效任务准入0/16、教师16/16、V1.4技术卡2/2、12,881个既有原生tick永久保留；不能因为新科学定义改变而重命名历史结果。

**根本目标**：在不修改冻结OCS2/MPC、MuJoCo、原SLA权重以及旧H40/G7基线的条件下，区分(a)参考切换的因果触发、(b)低层关节持续追踪间隙、(c)控制器策略时间与消息消费、(d)任务空间动态可执行域，再对真实OCS2后端单独验证长时域ReferenceManager与冻结SLA操作技能。第一阶段不使用SLA，不启动GPU，不修改底层控制器。

## 0. 证据事实／尚未成立的因果结论

- 2张站立无接触技术卡从tick 5703时把**旧且持续生效的初始参考**改为按该时刻实测EEF/底座绑定的**新静止参考**，随后10Hz持续刷新。新-旧EEF世界系差值（mm）T0=(-12.83,-3.10,-24.28), T1=(-12.70,-3.24,-24.99)；新-旧base世界系差值约(+17.7,+3,+10.2)mm，EEF姿态约2.7–2.8°，base约3.2°。不可把这种试验称作“从无参考进入静止hold”。
- 前100ms真实TCP Z下降-16.71/-15.42mm，其中底座贡献+0.09/+0.16mm，主要来自臂相对base运动；低层六轴位置目标对应TCP先下移，10mm门槛在16/26ms达到，实际TCP在66/76ms达到；初始前25ms内请求已经改变。WBC模式持续，电机输入无饱和。
- tick5703**之前**PD关节位置目标正运动学对应TCP已高于实际TCP24.38/25.46mm；终点差距仍24.16/24.64mm。两次最终actual-target Z约-20.27/-20.33mm；PD-target vs desired +3.89/+4.31mm，actual vs PD-target -24.16/-24.64mm。故必须区分“参考切换造成的新移动”和“预先存在的低层平衡/柔顺偏差”。
- 起点第3轴静止刚体重力项约+5.51N m，实际motor约+5.58/+5.21N m，显式PD torque feedforward=0；不能据此确定缺重力补偿，也不能忽略PD算式最多约1.44/1.32N m残差。
- 约197次/s机械臂请求、200Hz默认定时器，求解器旧日志MPC100Hz/MRT400Hz；频率不相等不是漏掉周期的证明。求解器policy消息有回传，但没有每次 `updatePolicy/Evaluate` 最终选中的policy ID；现有六轴命令↔原生PD的唯一值匹配并非逐命令ACK。
- 归档真实执行后端是 `rl_sim_mujoco` + `go2_x5_ocs2_node` + `mobile_manipulator_mpc_node`，**不是**归档RL-WBC Gamma/TimeLaw。两套科学后端禁止混统计。此前T1仅在**确认暂停边界**测得回退故障后0 tick，不能称运动中急停延迟合格。
- 原B2真实目标回显精确、2个技术卡完成；静止TCP绝对RMS=22.43/23.07mm，但技术完成标准并无精准跟踪门。B2-D0是CPU逐tick重建，新增native/GPU/optimizer=0。

来源：`reports/SLA_MEMORY_V14/B2_STATIC_BIAS_ATTRIBUTION.{md,json}`，`scripts/sla_transfer_v14/b2_static_bias.py`、`B2_NATIVE_READOUT.json`。现场必须回读 **完整原始文件哈希** 和已加载binary SHA，GitHub公开摘要不能代替私有租约/原始命令文件。

## 1. 资源、身份、实验人口和自治边界

- 指定独立Corelab本地owner，保留已注册source sha、controller sha、solver sha、MuJoCo model、WBC模式、2ms physics、原joint/collision/mobile STOP；不允许“修复”其他后端。**旧模式所有原始文件只读**。不使用真实机器人、不访问DEV/TEST、不训练/优化SLA、不复制私有state/weights到GitHub；不接入Agent。
- 自主执行器可读取原始数据、CPU分析、生成新代码和封存回执；**产生新的MuJoCo请求、修改现场ROS/执行节点、更换二进制、启用GPU前向或新的科学人口均必须验证单独的显式准入**。本文的建议预算不是权限。每个物理阶段有独立 `registration.json`、source/model/optimizer哈希、场景/seed/真实租约、resource owner、deadline、GPU UUID（如果需要）、physical_request_cap、native_tick_cap、唯一run ID与STOP。
- 候选 **D1首次4卡、确认无异常后追加一次相同2×2重复，共8个新物理请求**。每卡沿用原同一初始化5703 ticks+无接触保持500 ticks=6203 ticks，D1提议总上限 `8×6203=49624 ticks`；属于新的D1科学额度，**不复用已经消耗的B2 12881 tick**。如果任何一组几何/碰撞不安全就挡住那组，报告不完整因子设计，不能编造缺失数据。
- 后续D3独立科学预算建议 **F1平移/F2纯旋转/F3耦合路径 × ONE_SHOT_INTENT/P10_REVISION × 2个预注册速度档=12卡**，每卡如按5703+1000tick(2s)规划上限6703，总建议上限80436tick，只有D1+D2合格并再次申请后才能进入。具体card tick随现场原native卡和STOP重算，不能凭本建议直接派发；动态条件应先做少量安全smoke，smoke请求计入该12卡分母，不另开免费请求。
- B3新参考编译器的CPU开发可以与D2并行，但新物理安装独立资格；B4冻SLA+OCS2新任务pilot、额外GPU资源与任务分母**本文件不授予、不继承**旧P3十卡候选上限。

状态机：`D0_AUDITED → D1_REGISTERED → D1_WAVE_A(4) → D1_WAVE_B(4 iff safe) → D1_CLASSIFY → D2_PD_AND_POLICY_AUDIT → D3_KNOWN_TRAJECTORY_PAIRS → B3_REFERENCE_MANAGER_SHADOW → B3_NATIVE_QUALIFICATION → B4_FROZEN_SLA_PAIRED → SCIENTIFIC_FINAL`。阻塞类型`SOURCE_NOT_VERIFIED/RESOURCE_BLOCKED/UNSAFE_FACTOR_CELL/CONTROL_ID_UNKNOWN/LOW_LEVEL_TRACKING_BLOCKED/ABI_BLOCKED/SLA_TASK_OOD`；未过门不借新卡绕过失败。

## 2. D1：2×2分离EEF与Base参考切换（最高优先级）

### 2.1 四个固定条件

定义E_old=初始化持续生效的**原静止EEF参考的世界系位姿**（以tick5703之前最后一个合法目标回显核定），E_measured=tick5703从仿真实测TCP得到的EEF静止参考；B_old=初始化持续生效的旧底座目标，B_measured=tick5703实测底座位姿。**不是**将`old`定义成最近测量值，也不是沿用旧包过期TTL。所有4组都用相同连续10Hz刷新、世界系EEF+base有效格式、相同源状态和clock provenance。每次重新发布应有新的真实sequence/有效期并收到相应ACK，但保持old组的空间数值和参考意图不变。

| ID | EEF参考 | 底座参考 | 研究作用 |
| --- | --- | --- | --- |
| D1-OO | old | old | 正控制：不改变原空间目标，排查换包/时钟本身 |
| D1-MO | measured | old | 只改变EEF，估计EEF主效应 |
| D1-OM | old | measured | 只改变base，估计base主效应 |
| D1-MM | measured | measured | 复现原B2新+新边界，并量化交互作用 |

- 每个条件从**独立、可追踪的相同初始化逻辑**恢复；禁用隐藏的物理teleport、无证据的snapshot拷贝或额外稳定时间。固定scene、seed、initialization tick5703、历史128×25tick、gripper工作开度、底座接触、WBC模式、2ms、100ms目标发送、全部原安全门。预先核对tick5703各条件实际qpos/qvel、6轴关节速度/PD gap、TCP/base位姿和上一reference hash的匹配质量；容差先用既有T0/T1初始化差异登记，不看新结果后调整。每张完整卡原生1s测试，结束HOLD和关闭。额外首次失败、非法接触、安全STOP禁止重复计成成功。
- 第一次波次4卡，每条件1次；确认owner/lease/clean exit/STOP=0/原求解器目标合法及无未知低层命令后，才可执行第二个**事先注册**的完全匹配重复波次4卡。2次重复只是工程可复现性，**不代表独立任务成功率或统计显著性**。
- 冻结四种参考（世界系、姿态、base速度/姿态）并先跑CPU预演、FK与扫掠几何；若交叉组合在真实native安全门无法准入，准确记`UNSAFE_FACTOR_CELL`，不得改成“接近值”凑齐2×2。
- 绝不“停止发包”模拟旧old：现场0.4s TTL要求持续刷新；VLA P10高层0.5s与10Hz传输解耦保持原样。

### 2.2 观测变量/因果对比

每张卡前100ms逐tick(2ms)保存目标安装/ACK，选中policy身份如真实可见、六轴目标和速度、原生PD输出、六轴实际q/dq、电机控制、底座和TCP实际、FF/P/D项与限幅、四脚接触。对完整1s记录绝对TCP误差、起点增量误差、100ms Z位移、1s末端偏差、PD-FK相对desired、actual相对PD-FK、目标/实际越过-10mm的时间、求解器回显时间与接收策略age。

主要有符号终点`Y=Δz_TCP(100ms)`及`Y_PD=Δz_FK(q_cmd;100ms)`；记`Y_{OO},Y_{MO},Y_{OM},Y_{MM}`，预注册：
- EEF主效应：`ΔE=[(Y_MO-Y_OO)+(Y_MM-Y_OM)]/2`；
- 底座主效应：`ΔB=[(Y_OM-Y_OO)+(Y_MM-Y_MO)]/2`；
- 交互项：`I=Y_MM-Y_MO-Y_OM+Y_OO`。
其余 `X/Z`、姿态、cmd vs actual间隙与时效也逐项报告，**不能把小样本平均系数解释为显著因果结论**；因果解释条件是四组初始化匹配且只有两种参考内容改变。任何抢占/技术失败保持完整注册分母。

### 2.3 D1科学决策

- 若OO基本稳定但MO/MM触发同方向q_cmd下降，EEF target reset是主要触发候选；若OM/MM才出现，base参考耦合优先；若MM远大于单独分支，则研究MPC联合目标权重/姿态约束。
- 若OO本身也下沉，优先考虑每次重新绑定、控制器状态、PD平衡/策略更新，不可把结果独归EEF。
- 即使OO无新下沉，只要绝对PD-FK/actual间隙仍≈24mm，**D2低层追踪问题独立保留**，不能宣布controller已达毫米级。
- 固定5mm作为**描述性差异重要性**的建议量级，源两卡100ms偏差幅度差约1.3mm；不是机械安全门、不是显著性检验，也不替代完整误差报告。若要使用正式acceptance gate，先在本次D1任何输出产生前签署锁定数值及判据。

产物：`D1_FACTORIAL_4X2_REGISTER.json`、`D1_WAVE_A_ACTUAL`、`D1_WAVE_B_ACTUAL`、`D1_MAIN_EFFECTS_AND_INTERACTIONS`、private full curves/sha。

## 3. D2：低层PD/重力、关节误差、策略消费与时钟归因

**先做现有日志的CPU扩展，不对冻结控制器做盲目改参**：
1. 原生`state.bin.journal + init.trace + PD callbacks`：按每2ms计算`q_cmd-q_actual`、`dq_cmd-dq_actual`、`Kp`比例力矩/`Kd`微分力矩、前馈/限幅前后和`qfrc_bias`的范围，比较reference旧/新各条件；把MuJoCo力矩通道、单位、gear、joint mapping/source SHA锁死。PD目标本体已经存在的24mm TCP间隙不能当成world frame标定错误。
2. 用两种非物理替代算子分别报告：`FK(q_cmd)-FK(q_actual)`（关节追踪/柔顺）与`FK(q_cmd)-EEF_target`（上层reference到下层关节目标的误差），含X/Z分方向与t=0/100/250/500/1000ms。若`qcmd`相对于`qactual`持续有2°级误差，检验是否实际位置控制在负载下达到不同平衡，而非认定硬件不可更好。
3. 通过实际源代码评估`updatePolicy/Evaluate`返回状态、目标引用与被采纳policy的关系。**原封存binary没有policy ID**：先建立上游 target echo→policy message→arm_request→forwarded→PD的时间区间和可辨识比例；不能使用“最新solver结果”代替“已选中policy”。如必须在C++内部插桩，建立**新编译instrumented build SHA + 新资格**作为独立诊断，原冻结binary和旧D1结果不更换，不能称同一WBC。
4. 用零速静止目标调查MPC响应年龄/策略更新（区分模拟时间、单调墙钟、ROS stamp）；首份新参考精确policy回显晚于部分已发生目标变化，并不必然说明旧policy被误用，必须逐层核对同一因果ID。
5. 对稳定的`q_cmd-q_actual`和原生控制力矩，作一个**只读**的负载/PD平衡检验：若检测到重力补偿缺口，提出后续单独控制器改进/实验，不在本研究中自动上新力矩前馈或改PD增益。不要通过给EEF目标统一+20mm来伪装修复。

输出：`D2_LOW_LEVEL_TRACKING_AND_POLICY_AGE`。缺少最终policy身份证据则`POLICY_ID_UNOBSERVED`，不得伪造；若低层控制目标长期稳定不满足抓取精度要求，将本体任务B4保持 BLOCKED，允许继续非接触标定研究。

## 4. D3：非SLA已知运动与一次绑定/P10修订的可执行域

必须在D1定性与D2数值基线充分、原安全和资源资格明确后新立科学预算。不能直接把静态23mm解释为SLA需校正。

- 先评估F1小幅XYZ平移（方向预注册、至少有独立Z和水平轴）、F2小角度纯SO(3)旋转、F3曲率可控SE(3)组合曲线；起点实际状态、路径及障碍扫掠、关节限位、动力学G7/H40/OCS2合法性都在新physical前CPU预检。
- 同一完整意图分别使用`ONE_SHOT_INTENT`与`P10_REVISION`两种**高层行为**。两者始终维持独立10Hz接收器目标刷新，0.4s TTL不超时；P10=0.5s只更新未来意图内容，不重新从t0绑定原技能，不能以停止刷新造成HOLD来比较。
- 两个速度档仅为预登记的低速与较快合法档，具体位移、角度与限速需经D1/D2事实先定义、封存并保持两组一致。三类×两模式×两速度共候选12张独立原生卡（≤80436 ticks理论上限；另需真实授权与卡配置确认）。
- 每个条件必须报告：EOF目标与机器人绝对位置/姿态RMS、增量追踪、延迟与增益、P10边界`Δtarget,Δqcmd,Δqactual`，MPC目标ACK/策略age、底座运动和接触、STOP、时间完成率。要区分“可以沿路径移动”与“能够把40mm小物体精确抓持”；**绝对静止误差不可通过只报增量运动抹除**。

科学结论：若移动动态能力能在长时域/正确reference manager下高质量跟踪，B3研究有意义；若已知简单轨迹都出现大的绝对误差，暂停B4，另立有权限的低层控制优化项目。稳定偏差可能被闭环局部伺服控制改善，但不能用SLA open-loop进行未经资格的补偿。

## 5. B3：后端相关ReferenceManager V2（与D2/D3先行CPU实现并行）

**旧实验原H40/原G7的0/16永久保留**；新协议以 `TASK_SPACE_INTENT_V2` 与 `OCS2_NATIVE_LONG_REFERENCE_V2` 标记，不是放宽旧门后冒称老协议已经PASS。
- 上层SLA官方50×16→选右臂8D arm-base absolute→正确世界系TCP，保存source `t=0..49/30`、50点、gripper 事件和source/task/camera/calibration/quat身份，不把它们直接当作H40。
- 参考管理器把整段source几何路径与**已验证可行**的执行时标分开持有（必要时长于1.95s），持续发布当前MPC绝对时间后2s的未来参考窗口；接收器实际要求固定41×50ms、world EEF+base、单调序列、0.4s TTL/zero lead-in及10Hz刷新；内层OCS2求解器1s优化时域，与发布2s和策略更新0.5s无关。
- 应显式支持`trajectory_id/revision/epoch`、`accepted/installed/consumed`、合法未来替换与当前参考进度、预期安装时刻、command age、HOLD/ABORT；不得每P10把过去整条相对轨迹重置到当前time导致反向目标（旧B1出现-390mm/s错误）。同一10s直线 CPU fixture原来两个行为窗口全字段一致，可作为持续回归。
- 保留夹爪独立`owner/issued/ACK/consumed/actual/contact`，从student-owned reset合法控制，或在非接触合法时间完成明确owner transfer。接触或持物时不自动张开寻找G7通过；不能用measured替换上一consumed。
- 新OCS2安全/可行性契约：必须以D3实际后端能力、原关节/接触/碰撞guard、时效和路径事件为依据预注册；**不允许因为变长接口而绕过原硬安全约束**。任何修改控制器安全限额需要独立科学版本和许可。
- 独立评估延迟：SLA历史推理中位0.293s非新的实时尾延迟保证；用时间戳实测view acquisition/forward/transport/MPC policy age/actuation，明确上层只处理观测→安装，低层保留自身lookahead；避免双重延迟补偿。
- B3 `GATE`：在真实OCS2 ABI shadow下消息字段、39/41/80点区别合法，10Hz never TTL expiry，策略接续/时钟异常/首次HOLD/stop、旧参考替换全部逐字段通过；独立物理无接触资格再申请；不让source artifact替代现场consumer identity。

## 6. B4：冻结SLA任务验证（独立新科学人口）

只有D1-D3证实相关目标区有足够的**绝对**跟踪精度、B3获得正式原生资格、独立GPU0推理owner/lease、学生夹爪权限和新授权native额度，才建立新独立TRAIN held-out条件。
- 场景顺序：源域有单臂充分性的水果抓放局部技能匹配（若现有MuJoCo场景不具备则新协议先做任务科学设计）→StackCube视为OOD，并单独统计→再移动底座，不把全部任务混成一个成功率。
- 配对组：Teacher+现役OCS2、FrozenSLA+Legacy H40/G7（保留拒绝结果）、FrozenSLA+V2 ReferenceManager+同OCS2；固定同一场景/seed、控制状态、物体位置、参考协议与启动权限。
- 指标：reference installed/ack, actual TCP tracking error, planned vs executed motion, gripper close/open, grasp, lift≥20mm, hold≥1s, place/release, task completion、推理p50/p95和墙钟/仿真时间、移动时底座/接触。所有尝试含准入拒绝计入分母；原16TRAIN只是开发数据，不能当独立泛化验证。
- 失败归因分清 `OBS_OR_SKILL_OOD`/`REFERENCE_INFEASIBLE`/`HANDOVER_BLOCKED`/`BACKEND_ABI`/`POLICY_AGE`/`CONTROLLER_TRACKING`/`TASK_CONTACT`。**一旦底层已跟踪准确但SLA不执行抬升，应转向任务本体适配/少样本数据，而非继续重编译时间来“制造”技能**。
- B4无显式新预算即`NOT_RUN`；原V1.3 0/16、V1.4无SLA闭环和旧B2技术2/2永久单列。

## 7. 必需工程模块与测试（不是已实现）

独立分支候选：
- `d1_factorial_reference_matrix.py`：构造/冻结old或measured EEF×base，保持10Hz、真实receiver ACK、正确序号；
- `d1_native_owner.py`：核对host/UID/二进制SHA、owner lock、lease、4+4波次、逐卡上限、STOP，写一次；
- `d1_native_readout.py`：从`state.bin.journal`、control与ROS记录复原目标、PD命令、实际关节与base/EEF，估计主效应、交互与bootstrap描述范围（n=2只描述，不报显著推断）；
- `d2_policy_age_audit.py`：只有现场允许输出policy ID时才能建立完整控制链；缺ID就记不可观测；
- `d3_known_path_runner.py`：保证控制路径/发布频率独立，输出真实绝对与增量误差；
- `reference_manager_v2.py`、`ocs2_target_adapter.py`、`gripper_ownership.py`：只有CPU版本无native资格，不修改正在运行的WBC/solver。

单元/回归：旧old和actual误认、base-only/arm-only FK closure、无法比对初态、非单调时间、400ms TTL、0.5s P10只发一次、ROS新sequence/旧包重播、控制器状态机持旧、新目标失配、角速度数值、PD报文实际ACK与observer伪ACK、非法接触接管、满配额自动重跑、STOP后任意积分、仅报相对改善掩盖绝对偏差。PR #430/433旧结果必须逐项diff=0。

## 8. 隔离与复盘

- 报告层级：`PR433_D0_OLD_EVIDENCE`（原已封存）／`D1_PLAN`（计划）／`D1_NEW_NATIVE_ACTUAL`（未来可审计物理）／`D2_CPU`／`D3_NATIVE`／`B3_B4`。公开PR只放脱敏汇总、SHA和合法代码，私有sim/assets/state/git外。
- 每阶段必须提供`ACTUAL.md+ACTUAL.json`：source/loaded binary/renderer/robot/model SHA、owner lease、注册4×2 roster、issued/accepted/consumed/STOP数字、真实tick, model forward, optimizer updates、每窗时钟来源、失败分类、该阶段是否执行。空/阻塞必须标`NOT_RUN`不能写PASS。
- 新science结果不能更改已关闭B2技术卡，不能修改冻结WBC/MPC、原H40/G7的控制安全门或任何旧统计分母。若出现新的未经授权资源消耗、与其他现役胶囊的ROS_DOMAIN/worker冲突、非零STOP后积分、反馈失时/工具超时、重复请求/伪造ACK、非原命令行为/失支撑，**立即STOP并保留trace**。

**首个可立即推进的工作（CPU、零物理）**：在独立输出目录，回读source原始old/new窗口与第5703tick初始化一致性，封存四组EEF/base参考的精确世界系数值、源SHA、合法器与D1 4×2 roster；检查四种组合是否有无接触/关节安全资格，生成`D1_READINESS_ACTUAL`。没有新的正式资源批准时就停在这里，不能把这个计划等同于已授权物理试验。
