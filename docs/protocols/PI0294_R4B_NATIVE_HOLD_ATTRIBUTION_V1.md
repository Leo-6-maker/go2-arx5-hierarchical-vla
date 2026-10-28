# PI0294 R4B — 原生保持质量失败审核与分层归因

日期：2026-10-15。事实基线：`a6f2e4bc516fd2f4e3d26a9cc9bff161783eec36`，#11 comment6181996104。

**状态：AUDIT_AND_PROPOSED_DIAGNOSTIC_ONLY，authorizes_new_execution=false。** 本次用户报告原生保持FAIL，不等于授权重启已关闭lease、新PBS、调WBC或重试求PASS。沿用已经明确允许的只读代码/证据检查与工程草案工作；新物理诊断和修复后复验须由owner在#11绑定一次明确采用及有效资源窗口。文件发布不是GO。原R4/R4A结果、截止、余量、M2048和92文件原件不变。

## 1. 接受的实际结果

原native plant、free root、12腿＋6臂、冻结MPC/腿策略，5s warmup后10.014s恒定世界系EEF保持。尚未加入合格的活动夹爪/物体任务/机身相机overlay，不是完整操作环境已PASS。

- 物理TCP位置RMS0.012535354m <=0.015；SO3 RMS0.069650907rad >0.05。
- 原生测量FK0.012525667m /0.069599038rad；与物理测量给出一致负结论。
- 33参考接收、0拒绝，安装几何误差0；源时钟残差0.020838534s <=0.025。
- native stops/非法接触/warnings均0：这是**控制质量FAIL，不是本次真实安全事故**，也不是policy训练FAIL。
- 最后2s姿态RMS0.070513483rad；均值旋转向量范数0.069309699、均值周围RMS0.006885810，主要是持续偏置。
- 最后2s平均误差向量在目标坐标中约[0.00669,0.07016,0.00212]rad；不得直接将其第二分量叫世界pitch。
- joint3/4 requested-minus-actual均值约0.035365544/0.035124304rad。两个标量之和接近姿态角不构成因果证明，必须做FK/Jacobian和同刻数据分解。
- 本次BASE偏好省略；不是零速度软保持。WBC实际选择非零身体命令。恒定世界EEF不要求每个关节或base速度为零。

与PI0289约0.0652rad保持偏差具有症状相似性，但初态、源时钟、评分和版本不完全相同，不能宣称同一个已证实根因或直接作配对性能比较。

原先最终wrapper的PAUSE_CLOCK_ADVANCED被保存数组解释为延迟EE消息：物理时间前后均15.186s。保留原异常与独立解释；物理暂停通过不等于完整MPC/腿history恢复已资格。entry.sh未被物理子进程使用的偏差应先做零物理修复/环境验证，不因此重跑或否定原负结果。

## 2. 本轮真正要回答的问题

已正确安装的恒定目标，偏差主要在哪里产生？

1. MPC最优参考本来就有偏差（软成本折衷/模型/解算问题）；
2. MPC预测的BASE运动与腿/身体实际实现不一致；
3. MPC输出经过preview、坐标转换、限速和mode后失真；
4. 实际arm PD因重力/负载/速度前馈/饱和而存在跟踪偏置；
5. 以上有耦合，现有记录不足以唯一归因。

不是再重复确认RMS，不是给VLA加入约0.07rad目标补偿，也不是现在就启动增益搜索。

## 3. 已静态检查的代码线索（不是运行根因）

本审核从用户已上传私有tar读取文本，未执行包内程序/策略；归档源需由owner与真实部署hash对齐。自有源码/小报告可入Git；私有实现不复制到仓库。

- `WbcBridgeCore.cpp:237–270`：base姿态和arm位置有各自的preview，速度由另一采样点的opt_input产生；这是源代码中有意设计的行为，不可只因不同就说是bug。
- 同文件350–366：arm位置clamp后发布，native速度还有显式缩放；追踪后续rl_sar限速后的目标，不用早期request冒充最终applied target。
- `rl_sim_mujoco.cpp:224–229`：执行路径为tau_ff + Kp(qcmd-q) + Kd(dqcmd-dq)。必须读取实际字段、actuator transmission/gear/ctrlrange和最终qfrc_actuator；不能只从一个ROS topic断言实际力矩。
- `FullyActuatedFloatingArmManipulatorDynamics.cpp`的flow map返回input：当前native预测模型是运动学积分结构，不是完整四足接触动力学MPC。
- `task_floating.info`：EEF和BASE通过有限软cost参与优化，不是零跟踪误差的保证。BASE偏好省略时cost条件不同于显式BASE保持。

本地静态task_floating.info SHA256 `0b69bd58db364a01e87331b0e4e777a9684fda18a4f67fa60aee55174f72dcca` 与NATIVE_CLOCK_BUILD相符；不能据此推定所有部署源码/二进制均相同。

## 4. D0：先用现有92文件做零积分归因

先读取已归档 `control.jsonl`、`physics.jsonl`、`ros.jsonl`、`planner.jsonl`、controller/solver日志、初态、TASK_START和build/manifest，不重传整个WBC或重复备份全部历史。

### 4.1 字段覆盖与时间

列清每字段是原始测量、发布值、实际作用值、重构值或UNKNOWN。统一sim timestamp，保留消息接收时刻、policy ID、solver generation/valid horizon、reference ID、preview实际采样时刻。信号按有因果含义的ZOH/插值对齐，不能平移曲线到误差最小后冒称真实跟踪。

记录原sample-RMS不改；可以另外报告time-weighted RMS、整10s/末2s、vector bias与抖动，不能用新窗口选择覆盖原FAIL。R_ref与R_actual使用同一site/world/TCP和同刻源绑定；q与-quat代表同旋转。

### 4.2 误差链，先定位层级再谈控制器修复

在实际可获得且采样语义明确的时刻，构造：
R0=requested/installed reference；
R1=FK(B_plan,q_plan)；
R2=FK(B_actual,q_plan)；
R3=FK(B_actual,q_request)；
R4=FK(B_actual,q_applied_target)；
R5=实际物理TCP（并与measured FK交叉核查）。

R1/R2是明确标记的模型/混合状态诊断，不是实际可实现轨迹。计划sample horizon与命令apply-time不一致时先记录，不能把不同未来时刻的差当传输错误。缺plan数据时R1/R2为UNKNOWN，不凭均值填补。

报告每段相对旋转、位置、signed target-frame rotation vector。旋转复合用(R0^T R1)...(R4^T R5)=R0^T R5核查，不把标量角或RMS直接相加；向量比较必须变换到同一切空间。

对joint3/4分别及联合替换为同刻已作用目标，做离线FK反事实（不积分、不改live qpos），再用旋转Jacobian作局部解释。检查其他关节和base贡献、非线性残差。能说明某关节误差与姿态一致，仍不能单独说明为什么关节跟不上。

### 4.3 力矩与实现路径

以同刻实际状态、控制和原模型检查：
`M(q) vdot + h(q,v) = tau_actuator + tau_passive + tau_external + J^T lambda`。

h=qfrc_bias含重力/科氏/离心，不等于纯重力。可在独立状态副本零速度计算重力项，明确RECONSTRUCTED；不得在live插入额外forward改变warmstart。continuous qacc与离散速度差不强制等同。处理gear、限幅、接触/约束、被动力、模型自由根和arm索引。

分别报告P项、D项、已有feedforward、真实执行力、模型所需保持力、饱和margin，以及qcmd差和dqcmd-dq。若bias主要由有限PD位置误差平衡负载，说明它是控制设计/执行能力问题；不自动把加入qfrc_bias说成修复一个被丢弃的原字段。

恒定EEF且base运动时，非零joint dqcmd可能正是在补偿身体运动，不能直接清零。位置preview与速度前馈不一致也必须结合原设计和实际应用路径判断，不能为了过门同时改时间、gain和feedforward。

### 4.4 自动输出可执行的分流结论

- `CONCRETE_IMPLEMENTATION_DEFECT`：有字段/单位/符号/索引/时间/丢失消息的最小复现、预期规范和修复测试。
- `PLANNER_OR_MODEL_TRACKING_BIAS`：FK计划已偏离；附同刻目标、软cost/solver/约束/运动学依据。
- `BASE_REALIZATION_BIAS` 或 `ARM_SERVO_REALIZATION_BIAS`：附上述链和力矩证据，不只说疑似重力。
- `MIXED_OR_UNRESOLVED`：明确哪一个缺失字段阻止判断。

上述分流不使用事后挑选的阈值来强行指定唯一根因。D0必交一个可交给WBC负责人的简明缺陷包，不让高层团队长期承担盲调低层的工作。

## 5. D1：只有有新增信息时，才申请/采用有界原生诊断

以下属于待采用的完整小包，不自动利用已关闭lease。一次采用后按条件连续完成，不对同一parser修复重复询问；不允许新的gain/cost/weight搜索。

最多4次单worker启动、每次最多20sim秒（含5s warmup和10s评分）、总40000积分，所有原native安全继续首违规停止。无新增政策训练数据/GR00T/GPU/PBS。四个槽位：

1. **Instrumented baseline**：仅D0缺关键字段才跑一次同模式EEF-only保持，新增同步plan/command/force记录。若已有证据足够则跳过，不能重跑求更好RMS。
2. **Declared reference-mode counterfactual**：同一EEF世界目标、相同原始reset/warmup/起始source姿态，改为显式零速度BASE软位姿保持（原有baseTracking，不改权重）。这是新增目标模式/规划条件，不是修复原EEF-only结果，也不是固定根。目标body需绑定合法初态，不得按原偏差反向补偿。先通过实际14D parser/consumer CPU测试；只进行一个预固定模式，不扫描base目标/权重。
3–4. **One evidenced engineering patch verification**：只有证明实现错误，选择一个最小patch版本；在原EEF-only模式先复验同目标，再在预声明第二姿态或BASE模式回归。没有实现错误就不使用这些槽位调PD/重力/solver；交WBC负责人做独立版本决策。

对照2即使通过，也只取得“显式BASE软保持这一工作模式”的有限资格；原7D EEF-only FAIL不变。它与EEF13日后零BASE意图场景相关，但不证明移动BASE或任意EEF参考可执行。不得只选择成功模式后宣称原native controller所有模式都已合格。

判断通过仍用原位置RMS<=0.015m、SO3 RMS<=0.05rad、源时钟残差<=0.025s和原安全。末2s偏置、base命令/位移、joint范围和力矩都另报，不改变分母或选择评分窗口。一个hold PASS不自动启动整机采集：还需原R4A移动/停止/reset以及新夹爪相机plant和Teacher资格。

## 6. 自主工程修复的清晰边界

已允许的工程工作：可复现的时间/坐标/索引/单位/包字段错误、consumer错误、缺失原已承诺信号、入口绑定、只读logger和环境依赖；最小patch+定向CPU测试+固定diff，条件采用后复验。未来native launcher必须经已部署enter/profile，不能再绕过入口；环境问题修复不抹掉原负观察。

不属于工程自动修复：给本来纯PD路径新增未设计过的重力/积分控制、改gain/preview/feedforward scale/任务权重、改solver/摩擦/惯量、切policy-aware或更换WBC、提高限位、给高层目标加固定反偏置。即使它可能有效，也须标为控制器版本干预由用户/WBC负责人明确采用；随后冻结新版本，对所有任务同用并复验，不能声称仍是同一WBC。

若D0证据已经支持设计能力不足，直接交负责人，不要求耗尽4次。若一个最小实现修复仍失败，不做第二/第三个增益候选，不再重复同一保持测量。

## 7. 学习主线并行，但不制造虚假新数据

完整Go2+ARX5/EEF13仍是最终主域，不能退回固定根冒充成功。M2048是仍有效的局部任务模型，FB13和v4数据尚不存在。保持失败与M2048模型质量分开。

CoreLab可独立做EEF13/base API、19D观测/模态、CPU mask/codec测试及Teacher/场景规格，全部标设计或合成工程样本；不能把固定底座样本补3个零当mobile真值，不能把没有执行的Teacher路径当数据。新多worker只有整机单worker取得资格后才测，当前无需占4worker反复跑同一hold。

Hopper的旧R3模型面板因排队跨过截止而NOT_RUN，并未证明M2048失败。若以后明确批准独立模型评价窗口，复用固定底座D0+S1做已封存M2048行为诊断，不与整机hold串为同一门；本R4B不发新PBS、不新增训练，不要求先再训到M4096。发布权限与大权重交付不等于计算授权。

关键交付：把缺陷层级和最小复验交给WBC owner，同时保留可消费的模型和数据。修复本体执行能力后，回到R4A的站立/移动/停止→活动夹爪/相机→Teacher→v4.0→FB13，而不是重新开展无限低层预训练。

## 8. 预算、归档和状态

D0可在仍有效的只读/工程权限内执行；所需远端CPU必须有实际可用lease/profile。旧run终态不被重开。本提案完整D0+D1建议窗口：明确采用后4h（末30min收尾），Core<=8coreh/并发4核、单原生worker、最多4main starts/40000integrations、新增保存训练RGB=0、原生GUI/渲染若实际发生单独计量或UNKNOWN、增量小证据<=2GiB。R4原截止/成本全保留，需明确新的或重分配的有效权限，不默认利用旧余额。GR00T/PBS/GPU/梯度更新/WBC训练/硬件/历史删除/公开raw全部0；诊断中原冻结腿策略CPU前向计执行器工作负载。

每个实跑完整计warmup、被拒绝/未完成、子进程CPU和预留；只有采样推断的积分用保守上界，不能称精确。账号/权限/共享身份错误阻断相关全部任务；普通logger无法补齐则保留UNKNOWN。模型部署仍NONE。

复用92文件私有恢复树 `65d995cd2b549ba4ab1c0b5594731ed41dd83d1201c615c813803184b61604ac`，只发自研工具、去敏摘要/图和hash；禁止重发私有WBC源码/mesh/权重、原始路径和credentials。新重构结果另存，不修改原RESULT/CPU_VERDICT。

必交六类小证据：`SIGNAL_COVERAGE_AND_ALIGNMENT.json`、`POSE_CHAIN_DECOMPOSITION.json`、`JOINT_TORQUE_ACCOUNTING.json`、`WBC_OWNER_DEFECT_NOTE.md`、条件`MODE_AND_PATCH_RESULTS.json`、`RESOURCE_AND_FINAL_REPORT.md`。根因未闭合写UNKNOWN，实验未跑写NOT_RUN；原HOLD_QUALITY_FAIL永不改判。暂停稳定不能提升为完整resume PASS。

## 9. Codex直接接手顺序

1. 读本文件、同release任务卡、根AGENTS/START_HERE、最新#11/#47和本机私有交接；确认旧owner/lease终态。
2. 先执行获准的零积分D0，把92原件中已有信号用完；不为本次重建工作区/下载模型。
3. 发布归因和WBC owner缺陷包。普通可证明代码错误自主写最小修复/CPU测试；新物理按一次明确采用的小包开展，未采用保持停止。
4. 若有新原生资格，只报告该版本/该模式，按原R4A剩余门推进须有相应有效执行权限；没有新资格不采v4、不训FB13。
5. 固定commit和#11评论回读，报告已做与待做，不能以计划发布声称已经执行。

## 10. 来源与审核边界

实际GitHub读取：基线报告、NATIVE_HOLD_CPU_VERDICT、NATIVE_HOLD_SAVED_ARRAY_DIAGNOSTIC、NATIVE_CLOCK_BUILD、score_native_hold.py（ce04b7b3提交），#11 comment6181996104。静态读取用户上传tar内WbcBridgeCore.cpp、rl_sim_mujoco.cpp、base.yaml、task_floating.info与native flow map；没有SSH、未读取/重算92份私有原始数据、未执行任何仿真/神经程序。数值均据固定运行收据，不是本审核独立重测。

官方方法说明：
- MuJoCo 3.2.7 `M vdot + c = tau + J^T f`及bias定义：https://mujoco.readthedocs.io/en/3.2.7/computation/index.html
- 原生数据字段：https://mujoco.readthedocs.io/en/3.2.7/APIreference/APItypes.html
- OCS2有限软cost与reference manager：https://leggedrobotics.github.io/ocs2/optimal_control_modules.html

**下一步的成功标准是“定位误差在哪一层，并给出可复现的最小修复或清楚的WBC能力缺口”，不是把阈值改到能PASS。**
