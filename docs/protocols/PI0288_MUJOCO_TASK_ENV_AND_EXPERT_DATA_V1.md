# PI0288 — MuJoCo目标物体操作环境、冻结执行器与专家数据

Authority: `PI0288_MUJOCO_TASK_ENV_AND_EXPERT_DATA_GO_V1`.

用户要求：在确认新版是部署路线且原生编译/保持闭环已复现后，正式建设所需仿真环境；复现并验收运动/抓放，再在该环境采集专家数据。本协议授权现有Codex owner实现、运行有限目标仿真和教师采集，不只是写设计。GR00T/WBC梯度更新=0，GR00T神经forward=0，硬件动作=0。下一份协议另行固定目标域GR00T微调。

审核基准：`ca4878c34e708cffa77b3f9ab09af26333bca98e`；现役分支 `pi0267/executor-decoupled-recovery`；Issue #11。本文件+机器限额固定commit、Issue #11 canonical发布并回读后激活。发布不等于远端已执行。不创建竞争owner或长期分支，不重启旧Panda训练，不继承过期13:00/14:00截止。本轮没有新日历截止，有限资源和gate先到先停。

## 0. 证据、目的与历史边界

已实际读取的源：`reports/executor_candidates/OCS2_MUJOCO_20261010/REPORT.md`、`NATIVE_REPRODUCTION_RESULT.json`、Issue #11 #6087190102。最新该执行证据时间为2026-10-10 02:49:26 SGT（2026-10-09 18:49:26Z）。

确认的事实：用户确认此包为未来部署路线；archive SHA256 `6dd13098de5a58327fb0a0a5b3ff749e0a5534510375b92fba3f8fb50a322d52`；20 ROS packages+rl_sim_mujoco编译通过；11338 checksum entries通过；原生保持有850 MPC policies、1100 arm和1100 base commands，结束为WBC模式。其资格范围仅NATIVE_REPRODUCTION_PASS，不包含运动精度、外部chunk替换、夹爪接触或任务成功。消息数量不是成功率或跟踪误差。

实际报告的输入链：`/go2_x5/ee/trajectory_json -> measured-state-bound reference manager -> /mobile_manipulator_mpc_target -> OCS2 -> arm PD + RL legs`；EEF为position+XYZW quaternion，reference manager默认10Hz发布、2s窗口、0.05s采样。native floating模型12 state/12 input；不能将其当作历史201D/15D actor或TrajectoryBatch接口。私有JSON精确键/坐标/时间/错误处理尚需owner沿实际源码核对，不能靠topic名字猜测。

native MJCF仅12腿+6臂关节，无独立夹爪actuation和命名body/wrist cameras。必须增加真实任务物理与观测，不把规划URDF的固定夹爪/TCP当成可抓物体的驱动。

旧PI0287历史IsaacGym参考0/6、soft/hard-limit错误原件、缓存问题草案、未采集数据状态均永久保留，不重跑、不改判、不把迁移称旧错误已经修复。PI0286 C0 TRAIN2/8、C1 TRAIN8/8 DEV5/16及源域旧CONFIRM20/32保持原样。切换控制器+引擎是新的工程基线，不能据跨阶段分数推出MPC优于RL。

第一交付：正确教师EEF+gripper参考通过新冻结执行器，在目标Go2+ARX5中完成真实抓放，并形成同步RGB/实际状态/正确命令的可审计数据。附件《从第一性原理审视GR00T＋WBC项目》要求的可观测性、可执行性、接口一致性、闭环修正分别验收；其WAM设想不是本轮授权。

## 1. 复用原生成功，建立两层冻结身份

复用已编译SSD工作副本和合法Corelab-4080S ROS2 Jazzy环境，不默认重新编译20包/下载模型/迁移引擎。交付目录和既有成功smoke只读保存；任务扩展用独立overlay/构建输出。原11338条身份保持原件，新改动逐项diff/hash，不能扩展后还声称整个runtime source无变化。

候选核心身份：profile `native`、`task_floating.info`、`arx5_ac1_floating.urdf`、RL腿policy `N1_s29_175632`及config。继承并记录依赖差异rclcpp28.1.21 vs交付28.1.22；不要为追求表面相同升级全局依赖。保留已有二进制与模型SHA。记录MuJoCo实际版本/步长、MPC模型/代价/约束/时域/solver/频率、臂PD、RL全部预处理、joint/actuator映射、原生安全/超时行为、base/EEF frame与TCP。

**CORE_WBC_PIN**：从采用起禁止更改MPC代价/约束/模型维度、solver参数、臂腿增益、RL权重/normalizer/动作范围、原生安全规则。MPC在线继续求解不属于权重训练。

**TASK_EXECUTOR_FREEZE**：完成必要夹爪物理扩展、相机/时间/参考桥和加载后参考验收后，将核心pin+MJCF/URDF一致性+夹爪驱动+传感器+clock+场景+补丁一起固定为新版本；在锁定pilot和正式采集前完成。教师另有TEACHER_FREEZE。本轮后期不能边采数据边改执行器。

允许的小范围工程：原创bridge/记录/命名映射/末端驱动隔离/renderer/ROS副本与episode管理。原模块必要hook必须保留原行为并通过对照；任何不能证明为同语义修复的控制改变需要新科学决定。私有WBC源码、权重、mesh、build二进制和凭据不上传Git；公共仓库只放原创集成模块、清单与合法最小接口说明。

## 2. G0 — 实际接口与安全时钟清单

沿实际README、build/run脚本、reference manager、MPC consumer、arm/RL dispatch、MuJoCo step核对：完整JSON schema和单位、frame、TCP、quaternion顺序、相对/绝对模式、初点处理、目标插值、joint/action顺序、原生12D state/input含义、错误拒绝、取消/超时、reference ID和实际consumer。

发布链频率必须分开：仿真积分、MPC求解、RL推理、arm/base dispatch、reference-manager 10Hz再发布、教师/VLA 0.4s重规划、20Hz数据采样。10Hz重复发布不代表10Hz新高层决策，不能每次重发都重新锚定相对目标/重置轨迹时钟。

保留原clock/调度，优先统一现有sim time或提供经过测试的单调映射。不得只把一部分节点改use_sim_time；需要改clock adapter时在正式panel之前单独验收并绑定。reset必须同时清理reference、MPC warmstart/时间基准、RL history、夹爪状态和消息队列，episode UUID隔离旧消息；起始warmup<=15s单独计账且不计入32s任务。reset过程中不跨episode消费策略，不宣称仅设置seed就完全确定性。

复用localhost-only ROS domain228，先检查无其他使用者；无公网控制端口、机器人SDK/hardware driver或实机连接。记录实际topic graph、PID/lease和停止路径，原生模式退出/solver fatal/失联保护一律保留。

四个连续可读层级：raw requested -> reference-manager accepted/installed -> MPC/arm+base dispatched -> same-time achieved TCP/object。保存ref_id/序号、obs/issue/receive/accept/install/dispatch/sim时间，不能把相邻tick的目标与实际混为同命令跟踪误差。不存在某级回读时必须先增加只读记录，不能造值补齐。

G0输出SOURCE_MANIFEST、INTERFACE_CONTRACT、CLOCK_CONTRACT和SAFETY_LIMITS。确认新版schema，不复用旧IsaacGym的201D、15D、九点preview或肩部重锚定。

## 3. G1 — 新版外部参考正控（不重跑编译smoke）

原生hold是可复用前置；下述是新增有定量记录的external-reference验收，不拿旧消息计数代替。相同native起态，稳定warmup后t=0；固定6条件各10s，reference最大平移速度0.05m/s、角速度0.20rad/s，完整安全记录：
1. 当前实测TCP hold。
2. 相对名义起点水平+5cm，用2s光滑过渡，之后hold。
3. 垂直+5cm，同样过渡。
4. 工具局部z轴旋转+0.15rad，用2s光滑过渡，之后hold；不测试不可执行随机姿态。
5. 垂直5cm往返：0..2s上移，2..4s保持，4..6s返回，6..10s保持。
6. 平滑往返参考，H40/0.05s的窗口每0.4s真实替换；10s内25个issued chunk、至少24个distinct accepted replacements。保持各窗口共同未来点一致，检查重复发布无时间重置；必须实际经历替换，不以提前终止前缀冒充通过。

每条件要求完整10s、无原生abort/fatal、无非法接触/定义的安全越界。same-TCP最后2s位置RMS<=0.015m、max<=0.030m；SO3 RMS<=0.05rad、max<=0.10rad。平移2/3在2..4s的投影位移>=目标80%；旋转4同期有向角>=0.12rad，终点角差<=0.03rad，防止完全不动也通过。return/hotswap另外报告全程RMSE/P95/max与迟滞，不能仅报告最后静止误差。hotswap对共同未来时间点的安装参考非预期跳变<=0.001m/0.005rad；大于则不通过，不能吸附首点掩盖。

MuJoCo数值限位与安全限位分开：所有正越界都逐关节原值记录，使用未收缩的模型实际限位而非reward soft bounds。此新协议的判别epsilon固定为hinge1e-4rad、slide1e-6m；单步超过hinge1e-3rad/slide1e-4m，或连续超过epsilon>=0.020s，立即停止并quality FAIL；其余微小短暂超过epsilon计NUMERICAL_BOUND_WARNING，不可据此宣称零穿透，正式G1/G2 PASS要求没有此WARNING。原生更严格的安全停止优先。阈值在观察新panel前冻结；不调solver、gain或容差追分，不追改旧PI0287的零穿透判据。模型物理限位、安全约束、原始越界/时长三者分别报。

非法robot/table或robot/object碰撞不得仅靠MPC宣称避免：定义允许的foot-floor和任务finger-object/object-table接触白名单；保存原始penetration/接触力/solver告警，其他contact出现计失败并停止对应任务。编码/单位/TCP一致性CPU门不是控制误差门；初始条件不合法需预执行拒绝并记失败槽，不临时换姿态。

G1必须6/6。失败后停依赖链，不换gain/weight/seed重试；明确的接入错误可按工程替换条款处理，原件保留。这里所有数值是本轮新注册低速接入标准，不是已观测能力或硬件通用阈值。

## 4. G2 — 构建真正的目标物体场景、夹爪和双RGB

可与G0/G1并行做离线实现，正式WBC接触执行在G1通过后。复用原18个腿/臂关节及原浮动base，新增对象free joint、finger joints/actuators/cameras需显式name->ID->qposadr/dofadr/ctrladr映射。MuJoCo nq/nv不是旧18D动作数：禁止固定数组slice，不能把新对象/夹爪state塞进腿RL或MPC12D状态。startup asserts和合成状态单测覆盖base quaternion、18joints、finger/object free-body、controls/sensors；一次reset后所有原生state/reference/history对应新episode。

**夹爪身份先真实**：从本次部署硬件匹配的CAD/URDF或师兄交付规格确定finger几何、行程、闭合方向、质量/COM/惯量、TCP、速度和力上限。旧IsaacGym的14.24cm TCP、0.044m行程不可未经核对直接搬用。若缺可核实规格，仍完成场景/接口和显式PROVISIONAL_GRIPPER诊断，但正式TASK_EXECUTOR_FREEZE/数据门BLOCKED_GRIPPER_SPEC；一次性给出缺项请求，不虚构target-faithful。

独立gripper actuator不改变原18D dispatch和腿策略。规划模型已有固定夹爪时，做质量/COM/惯量与碰撞几何的一致性审计：不能额外重复加同一工具质量；在本轮允许基于真实规格的一次任务MJCF扩展，但不任意改变MPC模型/代价补偿；若必须改变核心模型才能一致，先新决策。开闭位置的几何及等效惯量变化要报告。夹爪机械耦合/tendon可用于真实结构，不允许对象与夹爪weld/吸附/隐藏助力/运动学跟随。

先验证empty open-close、single-pad contact、two-pad lift、release四类正负例。物体夹持需对向有效pad contact和实测相对运动共同确认；闭合命令不等于抓到；无物体/单侧触碰必须不被判稳定抓取。新夹爪驱动参数只在有界空载/仪器校准中按规格确定一次，锁定后不按pilot成绩调力/摩擦。

**首任务 `GO2_MJ_PLACE_V1`**：平地自由浮动Go2+ARX5、桌面、0.04m红色刚体cube、0.16m蓝色平面区域。cube默认质量0.05kg（声明为仿真任务设计，不代表已有真实物体测量）；摩擦/接触参数来自已声明资产或物理规格，缺失则在commission前一次冻结并标注来源，不用成功率拟合。桌面高度与可达source/goal区根据实际名义机器人、工具几何/碰撞计算选择一次；源目标初始水平间隔>=0.15m。移动base允许且记录，不pin/fix；首任务不强制大范围行走，不称为广域移动操作。无其他动态障碍、堆叠/第三任务。

layout生成分开控制object/source、zone、物体yaw及可行名义起态，随机化边界用几何/传感器FOV确定而非根据pilot成绩筛选。所有layout及分布在teacher commissioning之前冻结；确保TRAIN/DEV无重复初态；新确认ID只预留不生成观测。runtime reset不得在任务过程中移动物体到理想状态。

**双相机**：named body/wrist真实MuJoCo render，记录内外参和body挂载变换；使用不改变质量的camera/site附加（不添加虚构负载）。优先复用真实安装规格；未交付时可用明确SIM_PROVISIONAL_CAMERA固定配置做仿真数据，但不宣称sim-to-real，实际部署须另做标定。raw RGB至少256x256、20Hz，当前GR00T resize由processor另定义。renderer在受控线程/快照中运行，不能多进程同时写同一mjData；body/wrist从同一sim snapshot采集，跨双视图/state最大偏差<=0.025s。保存raw sim/ROS/wall时间、frame/drop/age，不复制帧掩盖丢帧。

开启双RGB后重新用相同6个G1条件验收装备化系统，包含真实空载夹爪，计为第二固定panel，必须6/6。不是同版质量重试，而是证明新MJCF/驱动/观测没有破坏执行。额外覆盖reset/clock/丢包/过期参考（非模型查询）的负例：旧ref不能跨episode、超时进入原安全hold、未来RGB/时钟回拨不进入训练样本。

G2输出任务scene manifest、gripper/sensor校准、named map、whole-stack空载运行记录。原生消息通路成功不等于双相机任务环境通过。

## 5. Teacher实现、冻结与pilot（G3）

在原Panda教师成功经验上复用approach/align/close/lift/carry/lower/release/withdraw/hold逻辑，但不调用ManiSkill专属robot API，不复制Panda世界位姿/joint动作。用目标物体相对抓取关系、实测Go2 TCP和目标夹爪几何计算EEF参考；保持物体抓握关系和期望物体终态一致，不能假设Panda/ARX5相同工具。

教师可读sim物体/接触真值，仅用于规划和离线评分；它不是OCS2或腿RL。它输出完整可追踪的EEF reference及独立gripper schedule，由唯一native参考入口和冻结执行器完成动作。不得直接发送臂腿joint oracle绕开OCS2后声称WBC成功。

nominal任务教师每0.4s读取新状态决定新的H40（0..1.95s）EEF参考，20Hz物理/接触检查和夹爪调度；manager仍按原10Hz重发布，原MPC/RL/PD频率不改变。不以40点/2秒/可能41点混为同一索引；JSON精确采样和端点约定通过单测。最大teacher EEF speed0.05m/s、angular0.20rad/s（不超过G1已测值），末端阶段由真实抓持/抬升/落点推进，学习前验证32s内是否足够；不因速度不足延长任务或改gate。

最多16个教师工程开发episode，ID288010..288025，每布局1次、每次<=32s，所有版本和失败计账。在这个开发阶段可修对象相对抓取几何、事件对齐、phase状态机与轨迹时间分配等教师逻辑，最多两个声明teacher版本；不得改已冻结核心controller、scene、physics/safety/task门。没有达到可信任务示范时不进入pilot。

最终TASK_EXECUTOR_FREEZE与TEACHER_FREEZE发布后，固定PILOT8（288000..288007）各一次32s，至少7/8成功才准入。任何valid quality FAIL不补seed/重跑凑分；工程无效必须有实际缺失/错配证据，原件和计费保留。pilot失败只能交付结论/新决策，不再本轮调整老师后重测这个panel。

**统一任务成功**：持续对向夹持>=0.30s；物体比任务起始抬高>=0.02m且持续>=0.30s；水平搬运>=0.10m。完整运行32s，t=28..30s全部41个20Hz样本满足：cube水平四角在蓝区内留0.01m余量、cube中心高度与桌面+half-edge偏差<=0.005m、线速度<=0.02m/s、角速度<=0.10rad/s、不再夹持、TCP到cube中心>=0.08m；无原生abort/安全越界/非法接触。物体进区但未抬起仍FAIL；29.9秒才到位仍不能通过41点。不能oracle成功即停或执行脚本式Student收尾；本轮只有teacher数据，必须正确标其身份。

## 6. G4 — 正式数据与训练输入准入

PILOT通过后同一冻结身份自动采T1 TRAIN64：288100..288163；DEV16：288200..288215。每个布局一个first-valid32s，失败原分母保留；TRAIN>=56/64且DEV>=14/16，成功TRAIN有效标签通过下列检查，才签发`MUJOCO_TARGET_DATA_READY`。预留288300..288331给未来模型确认，不采/渲染/让教师查询。检查历史ID和scene-hash无碰撞；冲突停在命名决策，不随机寻找容易种子。

每条episode保存四类明确隔离字段：
1. PolicyObs：目标body/wrist RGB、实际TCP姿态、实际gripper开度、语言；
2. Supervision：本次实际请求且被执行链接受的EEF/gripper命令、source/issue时刻；
3. Execution：manager安装/MPC结果/arm+base dispatch/实际TCP/base/trunk、clamp/reject/delay/validity；
4. Privileged：物体姿态/接触/teacher stage/成功谓词，仅teacher/score/诊断。

训练标签绝不使用achieved TCP冒充desired action；planned H40和后续actually committed setpoint sequence分别存。默认数据writer从连续实际teacher命令时钟构造20Hz future H40（训练锚定当前实际TCP的relative SE3），保留action_label_source；原requested/accepted/installed三者有差异必须量化、说明，不能静默把裁剪目标改作真值。后继GR00T训练协议必须明确采用何种label来源，此次提供可重建证据。

不能跨episode/reset/abort/接管/失效reference拼H40；每个完整641点32s记录最多602个连续H40窗口，实际数量按索引约定直接统计，不用旧601公式强行补数。每个窗口保留布局/任务/执行器/teacher/clock/语言/数据源和mask，失败样本只作为诊断，不混入成功监督。

source和target统计分开：本任务只用TRAIN生成声明的target statistics/processor，原Panda模型使用原统计量不变；完整raw位姿/矩阵/quat保留。监督encode/decode容差XYZ<=1e-5m、SO3<=1e-4rad、gripper<=1e-5，finite/非退化；preclip数量只做诊断，不再添加float64 preclip_count==0的隐性gate。DEV超范围原样披露、不用于fit、不删失败布局；真实数据损坏/TCP/时间错配仍阻断。

H40 rot6d row/column和XYZW转换做非单位随机旋转正负例；不能沿用旧WBCv1固定TCP偏移。episode独立划分，不把重叠窗口随机分TRAIN/DEV。保留事件/动态阶段标签用于采样与统计，禁止Student读隐藏phase/成功/物体真值。

末步仅在已有现代GR00T CPU环境，用实际缓存官方processor/tokenizer/collator读取真实数据，验证变长语言padding、双RGB、state/action masks、mixed episode批次和设备搬运。必须实际读取至少2个episode/阶段混合batch；代码写好不算READY。没有原生依赖则明确阻断，不修改全局环境。神经网络forward/优化/canary梯度均不在本轮。

输出`TARGET_FINETUNE_INPUT_READY`、数据manifest/失败列表、可直接调用的dataset转换/训练配置草案。下一阶段建议第一臂从官方pretrained+目标EEF接口+native loss开始，Panda-only、Target-only、Source->Target、多任务和反馈/BASE增量分别注册，不在此goal直接展开。

## 7. 工作包、路径与可执行交付

一个原owner+最多3worker：A reference/ROS/runtime；B MJCF/gripper/cameras/task；C teacher/data/scorer/CPU admission。CPU准备可并行，唯一sim-control stack/ROS domain由owner调度，不抢资源，不等辅助签字才能过已满足的gate。

复用旧公共task/scorer/recording的可证明引擎无关逻辑，新增 `src/go2arx5_vla/target_mujoco/` 与 `scripts/pi0288_mujoco/`。旧IsaacGym backend和原报告不覆盖。

要求实际实现的入口（非已存在声称）：`doctor.py`；`run.py --mode interface-tests|native-reference|task-reference|teacher-dev|teacher-pilot|collect`；`validate_dataset.py`；`prepare_finetune.py`。通过私有root和环境local配置找到用户包，不公开私有源码/权重。提供隔离build instructions、manifest、episode reset命令、完整运行示例、CPU单测、真实ROS/native收据。

第一可交付checkpoint是一个真实目标抓放demo及全部raw证据，不等64条全采完才反馈；它不替代固定pilot或数据门。终局必须交付所有成功/失败、同步双RGB原片段、raw->accepted->installed->achieved图表、夹爪/物体/BASE轨迹、gate table、数据集/训练输入准备或精确阻断。视频从保存RGB编码，注明sim时间/播放速度/teacher身份，不用示意图冒充任务结果。

## 8. 有界授权、工程恢复与停止

新预算独立于旧PI0287和native reproduction，历史消耗不清零。env-episode总上限180：setup/calibration/clock/contact48；native+equipped fixed reference12；teacher开发16；pilot8；TRAIN64+DEV16=80；技术替换16。每次reset生成的env-episode包括失败都计账；warmup<=15s、scene/calibration episode<=30s、reference10s、teacher32s。真实物理开始即计，不用只数成功进程。无并行多环境目标采集，最初只有一个sim-control stack。

总cap同时生效：physics integrations<=10000000；leg-RL actual forwards<=2000000；MPC solve attempts<=1000000（attempt/pass/failure分列）；arm/base/gripper dispatch总<=4000000；teacher actual plans/decisions<=120000；RGB pairs<=200000；跨设备GPU reservation/service interval union<=16h，CPU总<=256 core-hours，新owned output/cache<=100GiB。16h是预算不是耗时承诺；未知计数不能填0或all-cap PASS。若关键计数无法直接采集，可在开始前实施上界预留和运行日志，并明确observed vs bound；不能因换统计方式无限运行。

GR00T forward/optimizer=0；WBC optimizer/RL更新=0；新loss/BASE action/head/history/WAM=0；T2采集/训练=0；硬件=0。不申请H200训练。原生在线MPC求解属于允许控制，不算神经训练。

不设旧式3次qsub已受理即永久阻断：预执行build/路径/序列化等同语义问题在上述CPU/资源内可修，单一活跃lease；最多4类有明确根因的停机工程修复轮次，每类无新修复证据不得原样重试超过2次；每次实际物理替换耗16技术槽之一。修复不能改变物理/teacher冻结/算法语义追分。合法质量FAIL停相关下游，但其他只读/CPU模块继续。

本轮不是要求任意控制器都必须通过：G1/G2有效质量FAIL就交付新WBC负责人诊断；G3失败就交付teacher问题；G4数据门失败就交付不合格数据原件，不偷偷扩大布局或放宽门。缺真实夹爪规格不造数据；传感器未知以显式不确定性记录。

保护原始包、Panda所有模型、C0/C1完整恢复与历史证据。使用SSD工作/采集缓存，压缩后原始身份保留；本地完成与小manifest优先，不以HF全量回读作为下一gate必要签字。无任意清盘、sudo系统升级、自动邮件/监控/无限后台goal或PR merge。

Terminal：`TARGET_TASK_DATA_AND_FINETUNE_INPUT_READY`、`ENV_READY_INTERFACE_BLOCKED`、`BLOCKED_GRIPPER_SPEC`、`REFERENCE_QUALITY_FAIL`、`TEACHER_PILOT_FAIL`、`DATA_NOT_ADMITTED`、`TECHNICAL_BLOCKED`、`BUDGET_STOP`。所有均不得冒称GR00T policy成功或硬件资格；selected/deployment=NONE。

## 9. 科学解释与资料来源

成功只证明此已绑定OCS2+MuJoCo+腿RL+夹爪环境能执行并产生可信目标域任务监督，不证明高层已经学会，也不证明本体无关。按目标执行成功准入的银行是target-adaptation数据，独立Panda/任务库保留；后续目标微调不得称为零目标样本迁移。

执行基准材料：原生候选report/manifest（ca4878c...）、#6087190102；旧#6085650439仅历史边界。支持附件的四条件及先执行器正控后模型比较路线保留。OCS2官方reference-manager说明、MuJoCo官方qpos/qvel/约束文档仅用于接口与实现核对，不能替代私有包源码、硬件规格或真实验收。

当前审核只读取了公开证据及附件；没有SSH、读取远端私有JSON实现、运行新仿真或修改controller。上述待核对项必须由owner实际实现和回读，不得在adoption报告提前填写PASS。
