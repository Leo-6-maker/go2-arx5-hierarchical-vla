# PI0294 — 自有成功示范片段→源绑定重定位→整机真实任务种子 V1

**状态：PREPARED_FOR_EXPLICIT_ADOPTION。** 用户本轮已授权固定小包下载、只读来源准备和本方案发布；四个payload的338566923字节均已下载并完整SHA256核验。**本文件尚未获新重定位物理执行GO，authorizes_scientific_execution=false / launch_allowed=false。** 以下搜索与物理数字是待采用的硬上限，不是已运行结果。沿现役分支，不新开训练配方，不重置原窗口或父账。

机器入口：[任务卡](../../reports/PI0294/design/SOURCE_DEMO_RETARGET_TASK_CARD_V1.json) · [来源清单](../../reports/PI0294/design/SOURCE_DEMO_RETARGET_SOURCE_PLAN_V1.json) · [完整条件/随机流](../../reports/PI0294/design/SOURCE_DEMO_RETARGET_CONDITIONS_V1.json) · [源码与冻结组件锁](../../reports/PI0294/design/SOURCE_DEMO_RETARGET_SOURCE_LOCK_V1.json) · [实际准备与下载收据](../../reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/source-demo-preparation-v1/PREPARATION_REPORT.md)。

## 1. 单项科学干预和保留边界

问题固定为：**把已有成功示范的已提交操作片段接到当前实际整机起点，是否能产生真实接触与原T1/T2完整成功种子？** 本版只把无示范接触路径替换为源绑定的示范片段，不再开发/调整一轮FSM reached、误差游标或WBC补偿。

- 原样使用完整自由根Go2＋ARX5、原MPC/腿RL/WBC/权重/gains/dynamics、实际有限夹爪、原相机/场景/安全/任务评分。旧WBC保持质量FAIL只是独立诊断，不是本任务全局准入门。
- 原MP-S2：T1原首帧无合法pair、T2完整12秒无接触的有效短FAIL与记录器故障全部保留；不把新来源转移写成旧失败的工程重跑，也不声称已获得新成功种子。
- 仅原W0近场、seed0、原source/goal/table/cube yaw。BASE软意图明确为0，不声称学到了BASE协同或移动泛化，不改变native实际自由根/腿物理。
- 无模型/梯度/GR00T训练/GPU/PBS/正式扩采/MimicGen生成/真机/保护TEST/DEV-F/merge/邮件/monitor/广泛清理。selected/deployment=NONE。

## 2. 已下载与可实际依赖的来源

外部四文件固定revision和实际SHA见下载收据：ManiSkill PickCube 29349195字节、PushCube 26480047；RoboTwin ARX-X5 place_a2b_left clean50为190721619；MimicGen source PickPlace为92016062。二进制留在项目数据目录，不进Git，不下载随机500/全部数据集/9.6GB core包。

**物理候选池只用6条已有自有TRAIN示范。** 固定v3.0 manifest `b8ddcb9808e6dfac4f3ac5fa5a4828128fa9644a330c17d1c57bf833ce5889df`，原stats `fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7`只作身份绑定、不拟合或用于本次无模型规划。按task、非legacy/new、TRAIN、id排序，各前3条且不补替：

|task|固定来源优先顺序|
|---|---|
|T2|PI0294-TRAIN-F0-000-T2 → 001-T2 → 002-T2|
|T1|PI0294-TRAIN-F0-000-T1 → 001-T1 → 002-T1|

这6条原numeric文件逐一SHA匹配；实际raw原件存在，每条641个观测、80个H40参考，原SCORE success和AUDIT admitted均为true。导出numeric的601个锚点不是全部641帧，不能用它补造最后40帧。source samples/references/events/card/score/audit哈希已冻结；正式提取前核验这些成员，不重新采原示范，不重复全历史HF回读。DEV/confirmation不读作候选、不拟合统计。

外部只读库本次已按数字id检查各前3条：ManiSkill traj_0/1/2两任务、RoboTwin episode0/1/2、MimicGen demo_0/1/2。它们**不进入本版物理候选排序**，不能在自有来源质量FAIL后热换，亦不作为另一套Teacher搜索：

- ManiSkill实际H5为1000条/task；obs为空，有8D pd_joint_pos、Panda articulation及cube/goal状态。EEF需对应生成版本FK，不能把空obs称为已有RGB或把8D关节动作当EEF13。
- RoboTwin ZIP含50 HDF5＋50 pkl＋50视频，选定HDF5确有两臂endpose、joint_action与三相机；所查HDF5没有物体位姿/时间戳。未执行pkl、未解码图像，不猜端姿态是actual还是target，不把缺物体坐标作为阻断自有来源的前提。
- MimicGen此包实际10条、Sawyer、20Hz OSC_POSE、84×84双RGB；有所测EEF/物体与7D动作，但没有现成DatagenInfo/target_pose字段。不能从类定义推断文件已有标注，也不误写成Panda。

本版优先解决自己已有的成功技能能否接到整机；外部库的后续标定/候选引入须形成另一份固定改动，不能在本版失败后临时扩大来源池。

## 3. S0：只读片段提取与工具契约

采用后最多处理上述6条完整source，各641观测/640实际提交步/80参考。固定读取`execution.actual_tcp`、实际gripper gap、object_pose/contact旁证；动作取`supervision`实际提交及`references`的已安装first8 prefix。按ref_id/point_index/sequence/time重建唯一640步提交时间轴，不能把80份H40重叠全部当独立动作，也不能用未来actual EEF替代原提交目标。原source图像不作为目标整机新观测，privileged对象/stage只给Teacher与审计。

源片段边界来自已保存TEACHER_EVENTS及原提交步，浮点时间以`searchsorted`向首个不早于事件的实际20Hz提交样本绑定，不手调：

- T1：从首个align开始，保存close/verify_grasp/lift/carry/lower/release/withdraw，到首个hold后的第一提交步。例000-T1实际为align0、close1.45、lift2.60、carry4.35、release9.05、withdraw10.15、hold12.80秒；以各自原件为准。
- T2：从首个close开始，保存align/contact/push/detach/withdraw，到首个hold后的第一提交步。例000-T2为close8.25、contact11.10、push11.55、detach20.60、withdraw21.90、hold24.40秒；去掉的是原固定臂前置approach，不伪称整机已经完成它。
- 每源各至多1个完整任务技能片段，阶段索引是该片段的子区间，不另选源或切出多个“最好片段”。缺阶段、源成员/时序/标签不符→该源NOT_DECODEABLE并保留分母，不换第4条。

同时保存实际/提交tool SE(3)、object SE(3)、原时间、source gripper命令及实际gap、阶段边界/证据、原split、所有输入SHA。保存状态与命令分别命名；source“成功”不变成目标成功。

工具变换显式为`T_world_pad = T_world_tcp × C_tcp_pad`。从原交付模型/夹爪配置只读求源与目标的功能pad中心、闭合轴、零开度、开度单位/方向，绑定hash与关节裕度；不能假定两个ARX5同名即同TCP/夹爪。无依据则该源不准入，不标定WBC误差、不扩大开度/力限或调夹爪增益。

## 4. S1：唯一重定位公式与时序

对每task的三个自有来源，按固定顺序生成各一个候选，不增加±角度/新pose/新scene搜索。源functional-pad目标由原提交TCP与源工具变换得到；目标TCP由目标pad目标乘`inv(C_target_tcp_pad)`得到，不复制源joint动作。

固定任务frame：源/目标初始object center分别`o_s/o_t`，原CARD目标中心`g_s/g_t`，`R`仅为把源source→goal水平单位方向映到目标方向的绕世界Z旋转，不含拟合或额外抓姿旋转。零长度任务方向拒绝。对源提交pad位置`p_s(t)`：

`p_t(t) = o_t + R (p_s(t)-o_s) + u(t) [g_t-o_t-R(g_s-o_s)]`。

`u=0`直到T1 carry/T2 push开始；该阶段内按原提交pad位置累计弧长归一化0→1，阶段结束后`u=1`。零弧长却要求输运则拒绝；不按目标运行成绩调整u。这样局部接触保持物体相对几何，运输段绑定新目标，不缩物体/碰撞体。姿态为`R × R_source_pad(t)`，再转目标TCP；原rotation6D/四元数约定与decoder不改。

完整目标参考由两段组成：从**当前实际整机**到该源片段首个提交目标的自由空间连接；随后执行重定位后的源提交技能。当前真实根/腿/臂/夹爪用于规划起点和碰撞检查，绝不teleport、预装专家未来或用原固定臂q覆盖实际状态。目标路径使用初始实测物体/静态目标锚定，不假装知道目标未来物体轨迹。

只做确定性减速，不压缩到更快：按原0.06m/s、0.3rad/s及原规划joint 0.7rad/s/acc1限，计算最小uniform slowdown≥1，旋转用原SO3边界；路径不足则拒绝，不改阈值/速度。自由连接≤10秒，连接＋完整活动片段必须在任务28秒前完成；否则候选TIME_INFEASIBLE。28–30秒原41点保持及完整32秒观测不缩短。源gripper目标按物理开度/原速度约束单调转换；不篡改actual gripper。

活动片段结束后使用原source中已实际提交的hold参考及其夹爪目标保持；H40尾部只延续这个有来源的hold目标，记录HOLD_FROM_COMMITTED_SOURCE。不能用source缺失的后缀/未来actual姿态补出所谓动作证据。目标每一条新H40仍作为本次新提交参考独立记录，不能沿用source的执行成功标签。

阶段进度来自**已安装执行prefix累计时间/索引**，不增加误差达到游标或等到tracking RMS合格才前进。源事件只决定源片段分段，不是目标actual grasp/contact证据。目标状态每0.4秒重新观察，从实际状态对冻结的后续参考做同契约规划/检查；不切源、不回滚源阶段、不把实际输出修成理想值。尚未抓到就进入源后续阶段可能产生有效任务FAIL，须如实保留，而非再次修FSM门槛。

## 5. S2：有界CPU资格与内部搜索（不是物理成功）

先做16项窄CPU契约：6源身份/split；640唯一提交prefix；601/641不混；时间/事件边界；actual/target分离；工具与开度；源/目标frame往返；两锚warp端点与连续性；无joint/root/object写入；速度/减速/28秒可行性；首个完整合法源排序；IK/RRT/dense失败计数；恢复不重抽；task独立；原gate/scorer/native safety冻结；预算/截止/旧FAIL保持。当前这些**尚未执行，0/16**，不得把下载schema审计算作PASS。

6个候选各最多1次保存actual输入CPU bundle；按task独立，选首个满足来源/时序/速度/初始approach-contact合法连接与窗口检查的完整候选，并在新物理前锁定source_id、cut、工具/warp、时参和输入SHA。三个都失败→本task NOT_ADMITTED，另一task继续；不选第4源或外部包追成功。

实际warmup后每task至多1个专用bundle复核同一已锁定源及真正actual起点；不以warmup成绩更换源。任务每次新观察最多1 bundle，每task≤80，共160。**新bundle总≤6＋2＋160=168。**静态预览不证明动态接触/携物碰撞安全，不能复制理想物体未来位置使检查通过；每次实际安装前继续原gate/native安全。

每bundle保留现有固定内部结构：≤12目标IK调用、每调用16 CLiK初值（≤192），合法解按原arm L2/固定index排序，至多两route×两随机流×两连接段＝8个1秒RRTConnect；整体≤60秒，range0.1、simplify=false、fix_joint_limits=false、原时间参数化。失败可继续**同bundle剩余固定槽**，不物理试路后挑胜者。

随机流直接锁定c2ac固定表中从未执行的原MPS2_SEED_W0_T2/T1完整槽：CPU候选依次用对应row0/1/2，warmup用row3，正式decision用原row0..79；固定重复用于配对，不抽新seed。完整168bundle映射已写条件文件。所有CLIK初值、成功/失败/超时/崩溃占用及恢复均持久先扣账；完成子项只读复用，incomplete不免费重抽。

新总上限：2016目标IK/32256初值/1344RRT，密集点≤174174、每点≤5确定性局部修正（≤870870次），均为独立硬cap，不承诺同时用满。6个CPU候选各≤2002密集点、2个warmup各≤1001、160正式各≤1001，合计174174；全部失败路径也计数。不把dense局部IK藏在global IK中。

## 6. S3：直接两项完整整机条件，不再插入新短质量门

顺序：`SDR_SEED_W0_T2`，然后`SDR_SEED_W0_T1`，每task**一次完整32秒**、≤5秒原合法warmup、NON REALTIME。不重跑旧MP-S2短任务，不加一轮短里程碑质量前置。前12秒作为同一episode的进度侧报，合法健康执行自动接着到32秒，不重置物理/history或另算成功episode。

`新实际双RGB/EEF/夹爪/BASE → 已锁定源的重定位与有界规划 → 原SafetyGate → 仅软参考安装 → 执行first8=0.4秒原WBC → 新观察`。H40/20Hz、dt2ms、原source clock/pause/history、native14D安装一致性均保留。模拟等待规划时暂停自有物理/MPC/history，明确不宣称实时。

REJECT/非有限不安装；新物理失败不得换源、换seed或走另一模式重跑。原生安全FAIL停止本版本全部新物理；有效物体任务FAIL只停止本task，另一task独立继续。source缺失/工程不完整与科学质量FAIL分别报告。

只允许2次严格任务前同状态工程替换：每次warmup≤5秒、任务命令0、task RGB0，有明确工程错误并修复；同scene/seed/locked source/代码语义，原件及成本保留。任务开始后的报告/上传/图像故障只恢复原件，不重跑。首条实际RGB前确认父记录进程也有有效display；不能只给内层simulator设置显示环境。

## 7. 原物体评分逐字保留

T1仍要求实际持续grasp、抬升≥2cm/0.3秒、grasp下搬运≥10cm、原goal角点±0.07m/桌面高度±5mm/线速≤0.02m/s/角速≤0.10rad/s/released/TCP清距≥8cm/28–30秒41/41保持/完整32秒及原安全AND。

T2仍要求实际桌面接触水平推动≥10cm、不抬升>5mm、不持续双侧grasp、原goal/高度/速度/released/robot_separated/TCP清距/41点保持/完整高频接触与安全。W0原minimum BASE displacement0不变。

接触=实际接触审计，安装/规划/视频观感不算接触；任务seed=原score＋原安全＋标签/时间/记录完整AND。不同步/不完整不得收成功BC。单task成功只封存该scope一个seed，不声称双任务/自动扩增/GR00T能力或泛化。

## 8. 预算、旧账与独占接续

准备时真实父账38启动/353508积分/12248双RGB/15614.037112470665 Core CPU秒；无本run活跃PID/CPU reservation。只读来源核验的工程失败与CPU阶段均留原账。下载已实耗338566923字节，PC下载CPU未观测，写UNOBSERVED，不估计为0。当前准备没有新增retarget IK/RRT、render、physics或模型查询。

|项目|本版待采用上限与推导|
|---|---|
|完整条件|T2/T1各1×32秒，2科学条件；task独立|
|实际启动|≤4：2正式＋2严格任务前工程替换|
|积分|≤42012＝2×18503＋2×2503，含warmup和每启动3步尾部|
|task双RGB|≤1282＝2×641，无额外UI/视频渲染|
|CPU|新≤14400 core秒（含≥1800留收口），≤4cores/1物理worker；原父账CPU绝对≤36844.79211945514，按实时余额取更小|
|搜索|168bundle/2016目标IK/32256初值/1344RRT/174174密集点，见第5节；先到任一cap即停对应工作|
|存储|新增暂存≤8GiB，含小包/提取/新证据；不复制旧整包Trainer/venv，不删除科学原件腾配额|
|窗口|原T0=1792044600；新工作仍≤1792083759（10月16日01:02:39 SGT）；收口≤1792087800（02:10），不从采用时重新计时|
|额外下载/GPU/PBS/模型/训练/扩采|均0；已完成的小包下载不重复计为另一套新授权|

MP保留资源slice天花板仍43启动/404023积分/13771双RGB。按上述准备快照用满新上限时父账最多42/395520/13530、CPU30014.037112470665，仍小于原slice；预览不是执行或资源预留。

旧MP-S2已用32bundle/94目标IK/1504预留初值/62RRT/31031预留dense均不扣回。新旧合计≤200bundle/2110目标IK/33760初值/1406RRT/205205dense；再加旧V1的11011dense为216216≤原232232。旧V1的96初值/7RRT也永久保留独立账。未用额度不转换成额外source/episode/seed。

本文件发布不关闭或接管原MP-S2。**只有明确采用新固定版本后**，在无健康消费者的边界把旧未启动的ASSISTED槽登记`NOT_RUN_SUPERSEDED_BY_ADOPTED_SOURCE_PLAN`，原PARTIAL/实际FAIL不改；新scope独占继承剩余账，不并行消耗两套剩余额度。采用时再查真实owner/lease/PID/账本/剩余时间。已过截止则只报告精确资源缺口，不暗中续期。

## 9. 自动推进、停止与交付

明确采用后：S0真实提取/工具资格→S1冻结候选与时参→S2窄CPU/有界规划→各自合法则直接S3完整任务→独立CPU复算/原件私有readback/实际标注视频。辅助不是签字依赖，不停在CPU_READY，不等外部库全解码或另一task成功。工程同科学语义最小修复在剩余CPU/窗口/任务前替换槽内自主，记录所有成本。

两task都无合法source-route或有效完整FAIL且无seed→交真实负结果停止本假设，不再加来源/角度/FSM阈值/seed。成功一个→单技能seed封存，另一技能缺口如实报告；不自动MimicGen扩增或开训。

交付：6源原件hash与actual/committed/阶段/工具片段清单；完整候选/IK内部流/RRT/dense/时间可行性与消耗；两完整条件及所有中断分母；实际reference/安装/actual TCP/接触/物体结果分层；独立500Hz/20Hz CPU复算；真实双RGB和接触时间轴视频（FULL Go2＋ARX5、DEMO_RETARGETED_SOURCE、NON REALTIME、实际结果）；私有逐文件回读，小收据及完整负结果写Issue #11 canonical并回读。归档失败不重训/重跑，唯一原件不提前删除。

准备完成与执行完成、执行完成与质量PASS分开。WBC/原安全/原评分/旧FAIL保持，selected/deployment=NONE。
