# PI0294 — ManiSkill规划组件接入与整机成功种子：有界方案V1

日期：2026-10-15，Asia/Singapore。方案ID：`PI0294_MANISKILL_COMPONENT_SEED_QUALIFICATION_V1`。

**状态：PREPARED_FOR_ADOPTION，不是执行GO。** 用户已授权本方案的准备与GitHub发布，未采用下列新物理系列。机器卡`authorizes_scientific_execution=false`、`launch_allowed=false`；原V2已关闭。本文提出一项固定的组件接入与有限任务资格系列，不启动安装、物理、GR00T或PBS。

配套：[机器任务卡](../../reports/PI0294/design/MANISKILL_PLANNING_COMPONENT_TASK_CARD_V1.json) · [固定条件与预算](../../reports/PI0294/design/MANISKILL_PLANNER_FIXED_CONDITIONS_V1.json) · [源码/依赖/评分锁](../../reports/PI0294/design/MANISKILL_PLANNER_SOURCE_LOCK_V1.json) · [准备收据](../../reports/PI0294/design/MANISKILL_PLANNER_PREPARATION_RECEIPT_V1.json)。

## 1. 要检验的单项路线

使用**ManiSkill式物体技能分解＋MPlib的IK/RRTConnect规划组件**，从当前实际状态产生ARX5可行的Cartesian参考，经已有EEF13/native14D接口由**完整自由根Go2＋ARX5、冻结WBC**执行。先获得安全接近、真实物体接触，最终争取原T1/T2评分下的完整成功seed。不是继续改变FSM误差阈值，也不将Panda关节动作改名成EEF13。

背景事实：固定报告提交`3d2fcc9fd540d9f8e75661a50473b24147160741`、#11 canonical **6190273335**。V2三条有效12秒任务无接触，一条任务前就绪工程中断；游标角度推进但平移为零，成功整机示范0。所有旧FAIL/工程中断、raw、hash和预算保留。历史WBC保持RMS不再作为新任务数据的全局门；原生限位、非法碰撞、跌倒/稳定性、时钟与数据有效性仍是硬门。

这是一次**规划与技能执行架构干预**：旧cursor、逐分量进度拒绝及旧4mm/0.04rad `reached`阶段前置不再驱动新Teacher；它们不是原物体任务评分。没有声称只换库而其他算法逐字相同，也不把新旧结果作严格匹配初态的因果证明。

## 2. 固定组件与环境

- ManiSkill：`mani-skill/ManiSkill@62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`。复用`compute_grasp_info_by_obb`的物体轴/接近/闭合方向方法，以及PickCube/PushCube的规划—执行分解。真实几何来自本项目MuJoCo，工具映射来自实际ARX5 pad/TCP，不能复制Panda的25mm finger常量或world-X推送偏置。
- MPlib：**0.2.1**，上游tag解引用commit **`86376ca87d2c055b792d8cea4d15a38e861d866f`**；Linux CPython3.12 wheel SHA256 **`9d90de48f982f1e7412994ba851b223b411533a69514dc2fa21d655aed761295`**。源码/包是两个独立来源身份，不能凭tag假定wheel逐字等价；安装后要封存实际`mplib/planner.py`及完整依赖锁。
- 该wheel声明`numpy<2.0`，现有原生profile为NumPy2系列。**使用独立planner venv/CPU子进程**，通过版本化JSON/二进制数组协议与原生profile交换数据；不在健康仿真venv降级NumPy，不安装另一套SAPIEN/PhysX运行环境。可在新隔离venv完成必要CPU依赖兼容修复，不换规划器/版本扫参；依赖仍不兼容则交工程阻断。
- 固定主依赖为NumPy1.26.4、SciPy1.13.1、transforms3d0.4.2、TOPPRA0.6.3；相应wheel/sdist哈希已从发行元数据索引。TOPPRA0.6.3未列CPython3.12 Linux wheel，需在隔离环境CPU构建并记录编译器/构建依赖/产物hash，不声称安装或兼容性已经通过。其余传递/构建依赖在P0形成完整artifact lock后才准入物理，不污染原生profile。
- ManiSkill固定wrapper与MPlib0.2.1的方法名不完全相同。接入本卡固定的`IK`＋`plan_qpos`，不盲目执行旧wrapper的`plan_qpos_to_pose`，不引入其两次screw重试或`only-count-success`循环。
- WBC交付包身份仍为SHA256 **`6dd13098de5a58327fb0a0a5b3ff749e0a5534510375b92fba3f8fb50a322d52`**。实际已验收controller/solver/腿policy、暂停/整数时钟接口、有限夹爪工作行程、相机/plant及安全observer均从原私有运行收据绑定并逐文件复核。原始包hash不能替代已部署overlay/binary身份。

Apache-2.0的ManiSkill与MIT的MPlib许可分别索引；实际复制/修改几何helper时保留许可和修改说明。Git只收自有adapter、协议和脱敏收据；原生模型/mesh/WBC及raw保持私有。

## 3. 本体与规划资产合同

1. 保留自由root、12条腿关节、6臂关节和真实有限夹爪；不焊接底座，不teleport真实root/DOF，不重写接触/质量/惯量/摩擦。首批固定**原W0近场两任务**；完整本体仍动态运行，W0不要求额外10cm移动。不增加W1或新场景，本轮不宣称移动泛化。
2. MPC现有ARX5 URDF SHA256`403ce252c525437d66aa361a8a65b90bfe2b69dad0fcb1f7a17109582c7261bc`缺少四腿几何。可用于核对链名/关节顺序，不能单独充当全身碰撞证明。规划资产必须绑定原native完整几何、mount→arm→TCP→pad变换、限位及真实物体/桌面。
3. 若原URDF与native FK不一致，允许从**原模型运动学**导出私有planning-only URDF/SRDF，保留源到导出的逐关节/几何映射；不得拟合tracking偏置或修改物理plant。CPU FK相对native同姿态误差≤1e-5m、≤1e-4rad；这是坐标/模型一致性检查，不是WBC跟踪精度门。
4. 每个规划请求更新actual mount/root、腿与臂位置、夹爪及真实物体。MPlib检查arm自碰撞及场景；另用隔离只读native模型克隆检查全身/桌面/物体。允许在这个**不积分、不渲染训练图像的规划克隆**上放置候选q并做FK/碰撞计算；这些候选不是actual状态或数据标签。
5. 对将安装的2秒Cartesian窗口，按原2ms插值网格核对可行IK/几何，含leg/body/pad/table/cube；保持原允许接触对与穿透规则。规划侧仅按技能显式登记pad↔cube接触，carry附着碰撞体须由实际持续抓握触发，release后解除；不得为了接近而全局移除cube/table或机器人碰撞。保留原feet↔floor、cube↔table关系。
6. 规划时的root/legs快照不是未来WBC全身动力学预测。不得将静态FK/碰撞PASS写成动态全身可执行证明；原500Hz真实安全observer持续裁决，0.4秒后新actual再规划。

## 4. 固定规划与执行算法

每0.4秒一个实际观测包，沿原pause协议暂停自有物理/MPC/history，保留NON REALTIME标记。规划耗时不推进simulation time，不刷新旧包时间戳伪造新鲜输入。

1. 从actual q/mount/TCP、当前物体及固定技能目标建立规划请求。MPlib使用wxyz，native接口使用xyzw，磁盘rot6d仍按原行约定；转换和工具frame只做一次。
2. 一个请求只执行**一次`IK(n_init_qpos=8, threshold=1e-3)`**，其8个内部起点是已定义规划算法预算（首个为actual，后续按冻结RNG），不是8次物理任务。记录所有候选/拒绝，按固定最近actual关节距离及索引tie-break选**一个**合法解；之后只执行一次**RRTConnect `plan_qpos`**，`planning_time=1s, rrt_range=0.1, simplify=false, fix_joint_limits=false, time_step=0.05`。不用失败后screw/RRTstar/第二seed fallback。
3. IK内部候选允许等价角表示，但actual q绝不clip/wrap/替换。先独立检查actual合法；候选仍须映射回原关节限位和native FK。每次请求从`SHA256("PI0294_MP_V1/"+condition_id+"/"+query_index)`前4字节little-endian得唯一规划seed；预先列满220项，分别绑定NumPy与MPlib随机源。物理reset seed始终0；不因分数更换任何seed。
4. 规划关节速度上限取原URDF限制与0.7rad/s较小值，规划加速度上限固定1.0rad/s²；这只约束候选计划，不写入WBC。对计划FK得到Cartesian轨迹，以**全路径统一延时**满足原EEF≤0.06m/s、≤0.3rad/s，T2 contact/push/detach≤0.04m/s；不裁剪actual或偷改目标。若时长超出任务窗，保留真实超时FAIL。
5. 转换成原H40/20Hz world EEF＋有限夹爪＋BASE软意图，走原SafetyGate/接收/安装，执行prefix8=0.4秒；H40末端桥接padding不增加标签。W0明确发出真实提交的零BASE速度偏好，base_valid=true，经原软位姿接口执行；actual BASE可运动，不能将其写成“固定根”或从旧数据补零制造监督。
6. 新窗口起点绑定当前actual、目标绑定物体技能；这是新的**0.4秒实际状态重规划**，不同于旧50ms重新锚定，也不再运行V2 cursor/24mm与0.12rad的整体/分量提前拒绝。原独立SafetyGate及原生安全不变；重规划不是按joint误差表补偿WBC。
7. 每请求总墙钟≤30秒（RRT本身仍≤1秒），超时停止自有planner子进程、保留物理暂停与失败证据，不停止无关进程。IK/路径/碰撞/原gate拒绝时保存原因并停止该条件，不以hold后再次抽样隐藏失败。合法的下一次新actual请求属于正常闭环；同一失败请求不重复。若只是关爪/释放/保持且无位姿移动，使用已验收保持参考，不运行无意义RRT。

IK/RRT只生成候选；**不向执行器发送joint position/velocity、力矩或腿策略动作**。上下游数据报须校验source/condition/query/actual-packet SHA，过期、错位、非有限、缺失full-body碰撞覆盖一律不安装。

## 5. 技能目标与实际事件

使用原桌高0.44m、40mm cube、原source/goal与ARX5 TCP/pad标定。T1物体OBB的接近方向固定world-down，夹闭轴由上游OBB规则和actual pad闭合轴唯一确定；同尺寸轴tie保留确定索引，不枚举抓姿追成功。实际TCP到pad工具变换替代Panda finger常量。预接近/抬升/撤离距离沿原0.08/0.08/0.13m，不能看后续分数再改。

T2方向固定为source→goal，以实际pad支撑距离和cube半边长定位接触侧；沿原0.025m预接近、0.012m对齐、0.001m接触偏移、0.04m分离构造技能目标，不复制上游world-X常量。目标和几何参数在首个物理启动前封存。

新Teacher保留有限的技能顺序；其用途是物理事件驱动，不循环修改“reached”门：

|任务|技能转换|实际依据|
|---|---|---|
|T1|预接近→下降/对齐→闭爪|复用原`precontact_geometry_ready`及`actual_contact_alignment`的实际pad/cube几何，不用reference到点或预计耗时强制闭爪|
|T1|闭爪→抬升→搬运|原物理双侧grasp持续≥0.3s；原score的持续抬升≥2cm/0.3s与真实carry≥10cm，不能看close命令或拟合姿态误差|
|T1|搬运→降低→释放|实际object到原goal角点范围；接触支撑、原桌面高度/速度条件成立再释放。carry/lower用实际稳定grasp时保存的object→TCP变换重定位原物体目标，不做关节误差补偿|
|T1|释放→撤离→保持|actual grasp解除及原有效开度，实际TCP离物体≥8cm；保留完整32s，最终评分独立|
|T2|接近→闭爪→对齐→接触|实际预接触几何、有效闭合开度及实际pad↔cube接触；原几何阈值保持|
|T2|接触→推动→分离→撤离/保持|真实接触推动、实际平移≥10cm及原goal角点范围；解除接触，TCP距离≥8cm。全过程按原高频证据核对不抬升/不持续双侧grasp|

这些是新的技能执行规则，**不改原评分函数**。阶段事件只作Teacher/审计旁路，不作为学生输入，也不能代替物体结果。若某阶段在当前执行器上确实无法完成，保留具体失败，不转回WBC调参或修改门槛。

## 6. 顺序、分母与准入

只用原W0两条场景卡（seed0，source=[0.45,-0.08]，goal=[0.45,0.08]，table_center_x=0.515，cube_yaw=0）。不换颜色、桌面、相机或reset；复制为新条件ID及新scope，旧终局和原件不动。

1. **P0/P1零物理步接入**：固定源码、隔离依赖、FK/工具/全身几何映射、H40/time/gate/标签检查；最多12个已保存actual状态（W0/T1与W0/T2各6个固定前缀点）与24项固定正负接口检查，不跑神经模型或渲染。两个合法起始快照须有合法首段；其余不可规划状态保留原分母，不挑新样本。不要把单纯import或stub标成CPU_ADMISSION_PASS。
2. **P2短开发**：`MP_DEV_W0_T2`后`MP_DEV_W0_T1`，各≤5秒正常warmup＋12秒任务，各一次。两技能独立，不因先跑者FAIL取消另一技能；原生安全FAIL则停止整个版本，保存另一项NOT_RUN_SAFETY_STOP。
3. 每条短任务的准入条件：完整12秒时钟/数据、原生安全与安装标签一致；真实距离缩短≥2cm、实际安全预接触及pad↔cube接触；T1实际物理grasp≥0.3秒，T2真实推动≥1cm且无>5mm抬升/持续grasp。短里程碑不是成功示范。
4. **P3完整种子**：按固定T2、T1顺序分别运行`MP_FULL_W0_T2`、`MP_FULL_W0_T1`，各≤5秒warmup＋**完整32秒任务**，只要求**本任务**短条件合格。不要求两个任务同时PASS，不再等待WBC保持或其他场景通过。每项一次；短任务未过则本项NOT_ADMITTED。
5. 若本任务full通过原评分、原生安全与数据准入，封存**该固定场景一次成功seed**。单任务成功不冒称双任务成功，不从N=1推泛化或成功率保证；另一任务仍如实报告。若均未得到seed，交真实负结果并停止本假设，不新增场景/规划器/seed/姿态/阈值。

## 7. 原任务评分逐字冻结

源码身份见SOURCE_LOCK：原`target_sim/scorer.py::score_episode`、`pi0294/score.py::score_push`与相关几何/安全/clock绑定。新auditor只适配80组查询/640个执行标签的计数映射，不复制旧640-query假设，也不改任何质量谓词。

- T1：实际持续物理grasp、实际持续抬升≥2cm/0.3s、grasp下平移≥10cm；原goal所有cube角点范围±0.07m、桌面高度±5mm、线速度≤0.02m/s、角速度≤0.10rad/s、released、TCP清距≥8cm及原有效性；原28–30秒**41/41**保持、完整32秒/641样本与原安全AND。所有细节以固定源码为准，不额外伪造“原本就要求”的新评分条件。
- T2：真实contact、水平推动≥10cm、不抬升>5mm、不持续双侧grasp；原goal/高度/速度/released/**robot_separated**/TCP清距/有效性，原41/41保持与高频接触完整性AND。
- W0实际BASE位移只报告，卡中minimum=0；不是将W1的10cm要求取消。W1本轮未执行。

报告分开列：规划可达/碰撞候选、参考接受/安装、实际跟踪、实际接近/接触、完整任务结果。ACK不是selected MPC policy消费证明；该项无新证据仍UNKNOWN，不因UNKNOWN重新开展WBC内部研究。

## 8. 拟采用后的唯一预算

|项目|上限/规则|
|---|---|
|任务逻辑条件|2短＋按任务独立准入2完整，共4|
|原生启动|最多6：4科学条件＋最多2次**仅任务前**工程替换；共享总额，不追加科学条件|
|工程替换资格|≤5秒warmup、Teacher命令0、任务RGB0；有直接工程证据并最小修复后，同seed/scene/science重试。任务开始后工程中断保留，但不使用该替换额度重跑；有效质量FAIL不重试|
|原生积分|**59018**，含所有warmup与每启动3步保守尾部：2×8503＋2×18503＋2×2503|
|双RGB对|**1764**：2×241＋2×641；工程替换任务RGB0|
|规划请求|物理Teacher≤**220**：2×30＋2×80；另有P1固定saved-input请求≤12，合计≤**232**。每请求≤8 IK候选、≤1次RRTConnect；总IK候选≤1856；保持阶段可不调用planner|
|密集窗口几何检查|≤232232个2ms窗口点；每2秒窗≤1001；P1只检查不提交，P2/P3只对拟安装窗检查，均不积分|
|CPU/并发|新增≤**21600 core秒（6coreh）**，含安装/构建/规划/物理/审计/视频/归档；至多4 CPU cores，1个物理worker；CUDA隐藏|
|墙钟与父账|执行采用后最多6小时且不越原绝对截止；原T0=10月15日14:10SGT，新工作截止=10月16日01:10，收口=02:10。不得因本方案重置时钟/总账|
|新增临时占用|≤8GiB，位于注册Projects工作根；保存真实小批raw及私有完整回读，不广泛清理|
|GPU/PBS/GR00T/optimizer/正式扩采|全部0；没有FB13、MimicGen扩采或2→4worker权限|

最后只读基线：37启动/345005积分/12007双RGB/15244.79211945514 Core CPU秒，无活动owner/reservation。若花满本卡，算术预览为**43/404023/13771**，CPU≤36844.79211945514秒；不是实际用量或资源预留。采用/dispatch时再次检查父账余额与健康进程，不抢占他人项目；若原窗口已不足/过期，报实际缺口，不偷偷续时。

工程同合同修复在上述CPU/时间/启动范围内自主：路径、包兼容、序列化、FK坐标或就绪消息问题可最小修复并复核后继续；不为普通修复逐项申请。不得借工程修复改变规划算法参数、技能几何、WBC、原评分或有效失败分母。达到预算、有效科学停止点或原生安全FAIL时保存证据停止。

## 9. 必交付与下一依赖

- `COMPONENT_SOURCE_AND_ENV_RECEIPT`：实际source/artifact/transitive lock、许可、native9成员与plant/camera/夹爪/observer身份；不是只写版本号。
- `KINEMATIC_FRAME_COLLISION_BINDING`：arm/native FK、TCP/闭合轴、全身碰撞覆盖与允许pair；明确快照规划不能预测实际腿动态。
- `CPU_ADMISSION`：24项边界/失败检查、12个固定actual状态与首段结果、scalar协议跨进程input/path SHA；8步连续vs4+保存/恢复+4在**已保存计划**上严格一致，不要求重新运行RRT得到逐字同一路径。全部状态未改且无physics/model/render。
- 四逻辑条件的完整分区、6启动总账、每次规划所有失败及固定seed索引、参考提交/安装与prefix标签、20Hz原始双RGB/actual状态、500Hz接触/安全、物体score与独立CPU复算；停止和NOT_ADMITTED也交完整分母。
- 对实际运行任务生成原RGB视频，明确完整Go2＋ARX5/任务结果/NON REALTIME；没有task RGB的工程前缀不造视频。大原件私有逐文件readback，小收据/#11 canonical发布并回读；不因归档失败重跑物理。
- 若成功：`TARGET_BODY_SEED_PASS_SCOPE_W0_T1`或`...T2`及逐项成功身份；仍不称TRAIN64就绪。随后才准备独立的MimicGen式扩采/冻结数据/FB13方案；本卡停在**SEED_QUALIFICATION_COMPLETE_OR_VALID_FAIL**。

selected/deployment始终NONE。旧FAIL、原WBC/安全/任务评分、其他项目进程、邮件/monitor关闭状态不变。不真机、TEST/DEV-F或merge。

## 10. 一次采用方式

用户当前的“准备并发布有界方案”只授权本次文档工作。固定提交发布并回读后，如人类明确采用该固定方案，owner记录源绑定采用收据与剩余原窗口，自动按P0→P3推进，不再等辅助签字或例行GO；不能在COMPILED/CPU_READY/PLANNER_READY处宣称科学完成。未采用前，不将本DRAFT标记转换成launch_allowed=true。

上游依据：[ManiSkill官方规划说明](https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/motionplanning.html)、[MPlib固定planner源码](https://github.com/haosulab/MPlib/blob/86376ca87d2c055b792d8cea4d15a38e861d866f/mplib/planner.py)、[本体适配审核](../reviews/2026-10-15_MANISKILL_PINNED_SOURCE_TO_EEF13_ADAPTER_AUDIT.md)、[冻结WBC/任务数据主线](GR00T_TASK_DATA_FIRST_FROZEN_WBC_SCOPE_V1.md)。外部文档不是本项目科学授权。
