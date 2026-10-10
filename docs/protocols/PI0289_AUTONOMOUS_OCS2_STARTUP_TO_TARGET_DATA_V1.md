# PI0289 — 新版 OCS2＋MuJoCo 自主启动修复、目标操作环境与专家数据

**Authority：PI0289_OCS2_MUJOCO_AUTONOMOUS_BRINGUP_AND_DATA_GO_V1。** 用户明确要求 Codex 自行分析、在师兄最新交付版上跑通、安全验收后冻结，并继续建设目标物体环境和生成可信专家数据。不是只写文档；允许有证据的实现、有限私有 Corelab 仿真和目标数据采集。禁止真机、GR00T 梯度训练和修改原 MPC／RL 训练成果。

**父证据：** PI0288 initialization-repair commit 4f90b59f4f32d0dbad4f61ad3b8cf9357f66d1c8；Issue #11 comment 6091994226；之前 PI0288 原安全 FAIL、原生 hold PASS、所有旧实验不改判。审核起点分支 pi0267/executor-decoupled-recovery。新启动配置和任务扩展必须另记版本，不能覆盖父原件，不得将新 PASS 追溯改写为旧 PASS。

## 0. 依据实测而非猜测的根因范围

真正应用的 home 是合法静态关节初态，但 x5_joint2 和 x5_joint3 恰处于 0 rad 下限（零裕度）。首时刻前后足离地约16.50/26.08 mm；第一足地接触边界在0.056s、实际负接触距离从0.058s出现。保存42步里六个arm ctrl全零，MPC尚未开始；x5_joint3从0.066s开始持续越过原1e-4rad判别线，到0.084s连续20ms触发合法安全stop，峰值0.000661625rad；x5_joint2峰值0.000847217rad但连续4ms。MuJoCo的关节限位采用软约束，不能假设受限关节位置必被严格投影在范围内。

**强机制假设：** 零裕度arm home + 无主动arm保持的Passive/GetUp启动 + 足地触地冲击与关节约束耦合。时间相关性已观察，但单一动力学根因尚未隔离；先分析private source、力项与控制状态。此时不能归咎于MPC跟踪、GR00T或腿RL策略质量。

## 1. Codex自主性与物理安全边界

一个既有owner统一持有写入与仿真lease，可委派至多3个worker做离线代码、诊断、scene/data。只有一个live ROS/MuJoCo控制栈，继续本地/私有Corelab合法入口，不连接硬件或开放公网端口。

**可自行推进，无须为每次小工程修复询问用户：**
- 只读梳理私有最新包的init/reset/FSM/GetUp/DAMPING、motor/channel mapping、MuJoCo contact/limit solver、关节力项和控制器ready时间；增量诊断记录。
- 基于交付硬件/model/config支持的合法名义状态，重新绑定具有关节安全裕度、实际脚地接触解释和正确state/history/mode的启动契约；包括预执行检查、时间清理、实际启动入口/复位修复、只读仪表。
- 使用交付的原生支持模式建立启动阶段机械臂保持或standby序列，前提是使用原生可审计执行命令、未改控制增益/法则且启动状态在物理与安全约束内；任何实质上新增控制器/力矩算法的方案必须标注为新执行版本并另得批准。
- 构建私有overlay、精确source/build hash，执行同契约CPU、短时诊断和有界不同版本的物理验证。每一种新版本只做一次固定首次资格运行；有完整根因及语义变化才创建下一个版本。相同版本无质量复跑、无随机seed/姿态 sweep。
- 成功后按下列Gates继续参考、场景、夹爪、相机、Teacher和数据链，符合既有G1/G2/G3/G4固定原目标门。

**严禁：** 关闭/屏蔽安全observer、跳过启动前N步检查、抬高允许越界幅度或时长、放宽MuJoCo原始关节范围、隐藏reset/重写旧失败、调MPC cost/solver、腿RL权重/normalizer、PD gains或摩擦以追分；不得用Pin BASE、吸附方块、虚构夹爪或oracle动作冒充真实专家。必须保护私有WBC原包及旧资料。若唯一可信方案必须改变核心控制器或物理安全标准，先交证据和最小修改建议并标记 NEED_WBC_OWNER_DECISION；不得自行越权。

## 2. P0 — 做到因果可辨识的诊断

从已保存42步和这版私有源码起步，不重跑原有效FAIL。先做一次CPU/模型审计，把真实关节名称、范围与zero-margin、body site、原生actuator控制模式、joint limit solref/solimp/margin、base与IMU几何、home、RL默认关节位置、MPC启动条件关联起来。

建议只读记录同一个时钟上的：
- x5_joint2/3 q、qd、qfrc_actuator、qfrc_passive、qfrc_constraint、qfrc_bias、arm target/mode、GetUp阶段和controller ready状态；
- base z/roll/pitch、四足distance/contact正常力/冲量、物理solver warning及非线性约束；
- 第一个有效arm active hold时刻（若不存在则明确不存在），实际arm ctrl为何均零；
- home脚底离地与base高度，是否匹配真实预期“站立起始”而非“空中落地”；
- 真实model实际启动数据、state map与probe是否同一个mjData，不再把XML编译自检误当作正式episode。

可以在独立私有模型副本上做极少量小于0.20s、显式标注DIAGNOSTIC_ONLY的对照，用于区分接触冲击、自由臂以及起点限位的机制；不能当作G0/G1 PASS或正式质量重试，必须实际计物理与计算资源。

## 3. P1 — 合法初态与可审计启动修复

按下列优先级，从**实际交付可说明的状态**选择一种确定性方案，而非搜索得分最高姿态：

A. 如果部署包或师兄资料含明确stand/stow/initial-arm名义关节位置与初始站姿，先采用其配置和真实运动学/限位约束。若与旧home不同，给出新名称、来源和完整hash，不声称原包未改变。

B. 如果不存在已声明的安全内侧arm初态，允许从实际URDF/MJCF、关节限位、工具碰撞、工作空间和此前冲击包络**解析构造一次**带非零裕度的初始化候选，并作为SIM_ENGINEERING_STARTUP_VARIANT单独声明；不得在多个角度/高度/seed上穷举选PASS。若需部署级名义姿态权威，向WBC负责人提交一次精确问题，不能声称真实硬件一致性已验证。

C. 如果private source存在受支持的startup arm-hold或standby命令路径，检查它能否在首次足地接触前合法激活，并确保不会与Passive/GetUp和MPC/RL状态冲突。若必须实质改变算法/gain或发送不属于已交付控制合同的直接关节力矩，则BLOCK而非偷偷注入。

在模拟实跑前，CPU证明：完整根部姿态合法、所有受限关节留有与接触冲击相容的可解释margin、非退化quat、命名actor/actuator映射准确、足端接触/重力/碰撞合理、初始qvel/control目标一致、reset与原生状态同步；保存LOAD_INIT_CONTRACT和startup variant ID。所采用的变更只能在隔离overlay/明确源码diff中发生，原始MJB/mesh/weights和PI0288证据不变。

**有界启动判据：** 每个通过CPU的确定性新版本只执行一次固定启动至WBC mode并安全保持至少10秒（可记不计任务的启动warmup）；从第一积分步起启用原安全观察与原异常终止，不允许接触非法、数值警告、超过已冻结的joint判据、丢帧冒充反馈或fail-open。若失败，保留全部原件，先证明新的独立原因再开发新版本（本GO最多4个具可核查因果差异的startup variants）。没有证据则停机交精确根因矩阵，而不是刷seed/修改容差继续跑。

## 4. P2 — 外部EEF参考G1

合法启动通过后，使用已有原生JSON EEF reference manager（不是旧IsaacGym的TrajectoryBatch），保持坐标、时间与初点语义，从已测TCP开始。按PI0288固定六项，各10s：hold、水平+5cm、垂直+5cm、局部z旋转0.15rad、往返、0.4s H40热替换（25个issued chunk，至少24个distinct accepted）。严格使用PI0288的15mm XYZ RMS、0.05rad SO3 RMS、幅度/peak/热替换连续性、原joint安全、非法接触、全程与末窗误差门。未达全部要求不称G1 PASS；必要的source-proven消息/frame/clock bridge修复记新适配版本，不调整MPC/腿RL物理核心追分。

真实记录raw requested -> reference manager accepted/installed -> MPC/arm/base dispatch -> same-time achieved TCP。10Hz manager重发不能重置0.4s高层chunk anchor，旧reference不能跨reset或过期保留生效。若source接口本身不支持TTL，要有明示拒绝/安全hold且真实测试；不能把“未实现”描述为通过。

## 5. P3 — 装备化目标任务G2＋冻结

在G1可执行期间并行做CPU场景、gripper、camera。新任务4cm红块、16cm蓝区、桌面、允许自由BASE；新增独立夹爪控制与真实对向pad-contact、release，两个目标body/wrist相机真实20Hz RGB（至少256x256，场景-状态偏差<=25ms）。命名joint/actuator/qpos/dof映射显式，不把新增自由度塞进旧12D MPC或腿RL输入。不得复制Panda RGB、吸附或焊接物体。

必须确认部署夹爪**具体型号/版本、实际开度、力量/速度、pad几何材质摩擦和TCP/质量**。没有这些资料时允许有明确PROVISIONAL_GRIPPER/SIM_ONLY标签的场景、相机、数据接口诊断，但正式冻结和目标专家训练数据不得声称部署一致。一次精确交接请求即可，其他工作继续。

装备化后再过PI0288原六项参考及接触/空载开合/单侧误判负例，并发布两份pin：CORE_WBC_PIN（送达模型/二进制及控制算法）与TASK_EXECUTOR_FREEZE（全部启动版本、场景、driver、相机、时钟、adapter、补丁/hash）。正式teacher/dataset时期不可漂移。

## 6. P4 — Teacher与真实目标数据，条件自动续接

师兄交付版执行器冻结且可真实抓放后，移植Panda的任务阶段（approach/align/close/lift/carry/lower/release/withdraw/hold），基于目标物体相对EEF姿态/实测TCP/夹爪重新规划，GR00T不控制joint或WBC body/gait通道。Teacher的仿真真值仅用于教师和离线评分，Student训练只见真实RGB、实测TCP、实测gripper、language。

开发最多16条，教师冻结后fixed PILOT8（≥7/8完整32s真实抓放）；通过才自动采固定TRAIN64/DEV16（≥56/64、≥14/16），保留全部first-valid结果/失败，原confirm layouts不使用。完整task谓词和28..30s连续41点按PI0288原标准，不用seed搜索凑成功。保存teacher planned H40与实际accepted/committed setpoints、真实achieved TCP、object真值（privileged单列）、base/躯干反馈和原始同步两相机。不能用achieved TCP冒充desired专家标签；H40不能跨reset/abort；TRAIN-only statistics、rot6d/quat/time/mask严格CPU codec后，使用官方GR00T processor/tokenizer/collator真实数据batch完成CPU训练输入检查。

**本GO只到 TARGET_TASK_DATA_AND_FINETUNE_INPUT_READY；不跑GR00T/WBC梯度更新、GR00T神经forward、H200优化、多任务/BASE-head/WAM和真机。** 数据与执行器资格通过后另订GR00T目标微调对照。

## 7. 自动迭代预算、停止与交付

为避免之前三次尝试后只剩文档的失败模式，允许同一个owner在本GO预算内自主定位/修复有实证的接入错误，**不按提交次数设永久锁**。同时要有严格计量和物理限界：

- 最多180新增env episodes：startup/clock16、scene/gripper/camera24、native/equipped固定G1共12、teacher develop16、pilot8、正式TRAIN64+DEV16共80、技术替换24；总计180。每次实际reset、提前失败、低于0.2秒的物理诊断均记账。每个固定质量条件每版本仅首次valid结果；重新运行必须具有实际证明的接口/状态版本变更，不能同版刷质量。
- 物理积分<=10,000,000；腿RL forward<=2,000,000；MPC solve attempts<=1,000,000；arm/base/gripper dispatch<=4,000,000；teacher decisions<=120,000；RGB pairs<=200,000；GPU reservation/service union<=16h；CPU<=256 core-hours；新增owned output<=100GiB。旧PI0288耗用及FAIL永久另账，不能清零或覆盖。不能从episode推算未观察的actor/MPC开销，UNKNOWN明确写。
- 同刻一个控制栈，localhost ROS、原有租约和端口隔离；不得sudo/升级系统核心环境、破坏其他作业、外发私有模型或密钥、擅自cleanup历史资产，不能创建无限后台任务。若没有合法的private Corelab访问，在公共repo把所有可做的CPU/文档/接口测试实际完成，再明确BLOCKED_REMOTE_RUNTIME，不得伪造“已跑通”。
- 每完成P0、P1、P2、P3、P4之一提交一份compact actual gate receipt；首个真实成功抓放需报告视频/raw/校验，而不是等全部64+16完成才汇报。所有失败原始时序、接触、控制命令和hash保留；工程诊断、有效质量FAIL、未运行须分开。
- 即便未成功，也要交最新能运行的代码、脚本、单一命令、精确失败机制和给WBC负责人只需回答的问题，不停在漂亮的方案文档。

**最终状态只许** SAFE_STARTUP_AND_G1_READY、TASK_EXECUTOR_AND_TEACHER_READY、TARGET_TASK_DATA_AND_FINETUNE_INPUT_READY、BLOCKED_SPEC_OR_INTRINSIC_WBC、VALID_QUALITY_FAIL、BUDGET_STOP、BLOCKED_RUNTIME。中间Gates不能被命名为完成；selected/deployment始终NONE，直到后续神经/硬件资格另行取得。
