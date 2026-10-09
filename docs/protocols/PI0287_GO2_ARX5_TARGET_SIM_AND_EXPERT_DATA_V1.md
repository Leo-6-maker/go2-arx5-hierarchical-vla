# PI0287 — Go2＋ARX5目标仿真、冻结WBC正控与专家数据搭建

Authority: `PI0287_GO2_ARX5_TARGET_SIM_EXPERT_DATA_BUILD_GO_V1`.
用户请求：先搭建Go2＋ARX5仿真环境，产生专家数据，再基于预训练GR00T微调；由现有Codex owner搭建。本阶段授权实际实现、有限目标仿真、教师正控和数据准入；**GR00T/WBC优化器更新均为0**。目标数据合格后，另行固定目标域微调与评价协议，不自动重启旧Panda训练。

审核起点：`b1f81d260338c4c89aafec67dd4d5bc2da6fcbd9`；分支 `pi0267/executor-decoupled-recovery`；科学ledger Issue #11。PI0286终局 #6075999598（证据 `2f619bd1ed2c6f02aecccadb628300d19f4a9bcb`）原样保留：C0 TRAIN2/8、DEV未运行；C1 TRAIN8/8、DEV5/16；无新确认、无合格policy。旧父模型CONFIRM_A20/32及全部曝光划分不复用为新盲测。

本协议须连同机器任务卡提交为固定commit，在Issue #11发布canonical并回读才激活。发布不是Codex已启动、远端GPU已运行或环境已搭好的证据。本次未设新的日历deadline，旧13:00/14:00截止不适用于这个单独授权；以本阶段有限资源与停止门收口。

## 1. 第一交付与研究边界

第一交付不是多任务VLA，也不是WBC重新训练，而是：

`task teacher -> EEF SE(3) + independent gripper commands -> frozen Stage2/Stage1 WBC -> Go2-X5 physical object manipulation -> synchronized target RGB/actual state/command dataset`。

必须能从一条实际命令追踪到安装参考、实际TCP、夹爪接触和物体结果。先证明正确参考能被执行，再让GR00T学习。Panda数据/模型保留为源域参考；目标域采集和后续微调不称为Panda-only zero-shot或已证明本体无关。

本次明确允许新目标仿真/WBC推理和目标域成功教师示范采集，取代旧阶段“target physics=0”的限制，但仅限本协议预算。**源域通用任务库仍独立保留；按目标执行成功准入本轮目标适配数据，不是把所有任务数据重新定义成WBC可执行数据。** 这是一项范围明确的目标域建设，不改旧研究章程与历史结果。

本轮不增加VLA BASE动作、身体姿态、步态、world model、history输入或神经执行反馈。可记录BASE/躯干与执行反馈供后续研究，缺失字段带valid mask，不把缺失标签填0当作真值。WBC继续拥有本体协调动作的最终控制权。

## 2. 已核对的源码基础，不把设计文档当运行事实

用户Sources中的 `WBCv1.zip`：159225884 bytes，SHA256 `3c59a6d5f8f924df0b838ad2d501d3a853274d741db191a9de453f58212063ee`，687归档条目。只读取文本和文件清单，未运行上传代码/权重。压缩包是私有输入，不能整包、权重、私有WBC源码或资产上传Git。

实际文本检查：
- `go1_gym/envs/config/wbc.py`：go2与go2_x5是不同资产；go2_x5使用 `resources/robots/go2_x5_v3/urdf/go2_x5.urdf`、`x5_link6`及局部TCP偏移[0.1424,0,0.0001057]；默认Stage2 action_mode为end_to_end，同时支持IK相关模式。
- `go1_gym/envs/roboduet/legged_robot.py`：`root_states[:num_envs]`、`rigid_body_state.view(num_envs,num_bodies,13)`、reset/push等有单robot actor布局假设。直接添加对象会影响索引语义。
- 同文件 `_compute_torques()`：18维学习动作补齐到资产DOF数，不能代表任务夹爪已接入。
- `resources/objects/cube.urdf`：碰撞尺寸1.0×1.0×0.05m，不是4cm抓取方块。
- `wbc_env.py::_load_trajectory_for()`：从轨迹bank加载并按肩部重锚定；不能原样用于世界系物体目标。
- `modules/trajectory.py`：可复用Gamma/TimeLaw/TrajectoryBatch；转换包含滤波语义，load存在容量处理，需审查原始/安装差异，不能静默截断目标。
- `scripts/play_by_key_stage2_traj.py`：真实调用顺序是arm policy -> env.plan -> dog policy -> env.step；默认robot=go2而非go2_x5。
- 归档中未识别到足以明确绑定完整Stage2轨迹策略的checkpoint/config组合；这不证明师兄或Corelab没有它。须先查已授权项目manifest、实际已交付文件及新版说明。

上传的AGENTS/设计材料可能属于不同迭代，优先实际import/__file__/source hash、checkpoint参数快照及运行观测/动作布局。不要拿同名历史helper或Stage1 arm JIT替代已训练Stage2。

## 3. 两条可独立推进的轨道

**A — 不等待Stage2权重的工程搭建：** 私有资产定位/哈希、环境入口、显式actor/body/DOF索引、桌面方块、相机、夹爪API、时钟、记录器、scorer和CPU契约测试。可以运行明确标注NO_WBC的静态资产/传感器诊断；不得当作控制器或任务资格。

**B — 必须绑定真实执行器的物理正控：** 完整Stage1+Stage2权重、参数和资产一致后，运行冻结WBC参考fixture、教师pilot及数据集。缺少关键文件时出一份精确的 `WBC_HANDOFF_REQUEST.md` 和 `BLOCKED_EXECUTOR_ASSET`，仍继续轨道A；不反复等待同一份确认，不用随机/未训练arm policy冒充专家。

已有合法Corelab Isaac Gym环境优先，隔离于现代GR00T环境，不在旧环境中升级torch或破坏其他项目。不自动迁移到Isaac Sim/Isaac Lab/SAPIEN；缺少合法仿真依赖时给出具体阻断和版本需求。无硬件SDK初始化、真实机器人连接、公开网络控制端口或sudo安装。

## 4. Codex必须实现的模块与接口

新公共逻辑放 `src/go2arx5_vla/target_sim/`，阶段入口放 `scripts/pi0287_target_sim/`，测试放 `tests/target_sim/`。私有WBC通过环境变量/忽略的本地配置接入。只能发布原创bridge/任务代码、合法的最小接口补丁与哈希；涉及私有源码的完整diff留在受控本地，仅记录patch身份。

### A. RuntimeManifest / doctor
记录Python、Isaac Gym版本/构建、torch/CUDA/driver、真实设备、原生sim dt/decimation/actor频率、资产/mesh/TCP、arm.action_mode、观测/历史/动作布局、两policy及normalizer/参数文件哈希。文件身份以完整bytes/hash为准，不能跨挂载比较inode/device。

权重优先师兄本次明确交付的完整组合；若只有现有历史完整组合，可作为 `HISTORICAL_EXECUTOR_BASELINE` 诊断，不称为新版，结果不混合。开始固定panel前只绑定一套；panel后更换控制器需要单独决策。

### B. TargetActorIndexMap / state / reset
对每个env记录robot/table/object actor、robot rigid bodies、feet、TCP link、arm/leg/gripper DOFs的显式DOMAIN_SIM索引。覆盖root/DOF/body/contact刷新、奖励/终止、Jacobian/body映射、相机、reset、外推力、last_state等所有实际使用路径。不能只修base_pos；不能依赖torch高级索引返回的copy保持实时view或进行原地写回。

先num_envs=1，再num_envs=2检查非相邻robot actor索引；只reset env0时env1的root/DOF/object/reference/history不改变。重置只发生在episode边界，复用任务时不在过程中传送物体/机器人。完整日志校验无state串线、无silent stale gather、无cross-env contact。

### C. Scene / geometry
新任务 `GO2_PLACE_V1`：平地、桌面、4cm红色刚性立方体、16cm×16cm蓝色平面目标标记；先无容器、无额外动态干扰物。指定真实质量、摩擦、惯量、碰撞过滤、solver和夹爪参数，并写入SCENE_MANIFEST。

桌面高度/坐标和抓取/放置区间仅根据师兄名义初态、工具几何及保守碰撞/运动学检查选择一次，在教师开发前冻结，不看任务成绩挑位置。源-目标中心初始水平间距>=0.15m，均位于可测试合法范围；不要求首任务必须显著移动底盘，报告实际BASE运动且不称广域移动操作。

robot始终按原自由浮动基座/冻结WBC运行。固定base的资产诊断必须标NO_WBC_OR_PINNED_BASE且不进入任务成绩/训练数据。不得通过修改臂腿增益、质量、碰撞或摩擦来挑成功。

### D. Separate gripper + physical sensors
按实际DOF名称驱动x5_gripper_joint及配对关节，验证URDF mimic在此runtime的真实效果，不能假定单关节值就是总开口。核对末端几何、开闭方向、joint limits、开度、夹爪力/速度上限。新夹爪驱动参数在空载校准后冻结，不改变原18D policy或臂腿控制。

记录命令/实测开度、两侧接触、物体运动。禁止吸附/焊接物体、隐藏约束、运动学物体跟随、瞬移或使用成功标志代替真实抓握。

body/wrist双相机必须由目标仿真真实render，记录内外参、机身安装变换、RGB时间戳、有效标记。viewer录像不算双相机数据；不复制当前帧伪造历史，不使用Panda图像配Go2动作。允许额外外部相机仅供诊断，绝不静默加入policy输入。

### E. ExternalEEFReferenceBridge
复用实际TrajectoryBatch的Gamma+TimeLaw/preview/progress链；新增明确external_task_reference模式，绕过bank采样和肩部重定位。协议包括frame_id、TCP、rot6d约定、obs/issue/install时间、ref_id、有效期、EEF时间网格、独立gripper schedule、cancel/hold/reject状态。维持WBC实际观测维度/历史与内部plan -> dog调用顺序。

禁止直接把GR00T10D写入WBC15D，也禁止直接覆盖最终dog身体命令。原生空间preview由WBC派生。验证raw -> canonical -> installed目标，容量不足显式拒绝；滤波/投影偏差单独记录。首次策略不增加BASE偏好入口，仅记录真实base及内部nominal/plan供后续设计。

先一次安装固定正确参考；再同一物体任务做每0.4s替换。新chunk锚定观测时刻任务坐标/TCP，不能每个tick叠加移动后的base；不得把progress强行置为时间参考或静默首点吸附。原生WBC控制率按实际训练配置不变，教师/数据采用20Hz时钟并记录多率延迟，不把50Hz低层粗暴改成20Hz。超时/拒绝/取消在边界安全hold，原生安全终止保持有效。

### F. Panda-style target teacher
参考 `scripts/pi0285_execution/teacher.py` 的任务阶段，不直接调用ManiSkill-specific API或复制Panda世界坐标/joint轨迹。按目标物体几何、目标TCP/夹爪重新计算物体相对抓取关系，状态机为approach/align/close/lift/carry/lower/release/withdraw/hold。

教师可读取物体真值进行离线规划和评分；其输出只是正确EEF+gripper参考，由同一冻结WBC执行。采用接触/实际物体变化推进阶段，不仅用发出的命令或时间猜测成功。动态命令受声明的位姿/速度范围限制，阶段记录stage_before/command/stage_after，事件按实际命令与反馈计算。

内部可使用运动学/路径规划器判断可行性，但不能绕过WBC直接驱动臂腿而称为WBC专家成功。若必须使用另一执行器，只交付单列诊断，不进入本协议正式教师数据。连续完整轨迹片段从真实状态生成，不能在物体抓住后重启成默认张开夹爪的approach。

### G. Recorder / scorer / dataset writer
严格隔离：
1. policy_observation：目标双RGB、实际TCP、夹爪、语言；
2. action_target：本次真实发送的未来EEF+gripper命令；
3. execution_feedback：实际base/躯干、installed参考、同一命令对应achieved TCP、拒绝/延迟/有效性；本轮只记录，不自动成为GR00T新输入；
4. privileged_diagnostics：物体位姿/速度、接触、教师阶段/成功谓词，仅教师和离线评分可读。

每个样本有episode/layout/task/asset/teacher/controller/codec身份及同步时间。H40不跨reset、abort或数据接管边界；无未来RGB或事后oracle输入；实际达到的TCP不能替代teacher commanded动作。闭环未来参考必须标明来自后续实际教师命令，并在takeover处截断无效窗口。

采用真实20Hz监督H40/前8点契约，保存native quaternion/matrix与转化说明，rot6d矩阵行列显式单测。TRAIN-only目标数据产生新的声明统计量，旧Panda模型仍用自己的统计量，不能偷换。相机tokenizer/collator用实际缓存官方processor CPU路径；检查双指令变长batch、padding/mask，没实际数据的feedback/BASE channel必须mask，不填0监督。

对真正进入梯度的候选监督要求XYZ回环<=1e-5m、SO3<=1e-4rad、gripper<=1e-5且finite/nondegenerate；float64中间preclip-count是诊断，不能另加count==0门。DEV范围外情况原样披露，不用DEV拟合统计、不将裁剪目标当新真值。

## 5. Gate与真实执行许可

**G0 ASSET_AND_CPU_READY：** 完成archive/import/manifest、CPU正负例、mock接口与缺失依赖报告。mock不是物理PASS。缺少Stage2可继续场景和记录工程，但真实WBC质量分支BLOCKED。

**G1 PHYSICAL_ENV_READY：** n=1/2的显式状态索引/部分reset、真实相机时钟、夹爪空载开合、物体重力/碰撞、独立坐标/FK检查通过；完整controller身份读取通过才启动WBC fixture。

**G2 FROZEN_EXECUTOR_REFERENCE_READY：** 对绑定WBC，固定6个10秒fixture：当前TCP hold、5cm水平平移、5cm竖直平移、0.10rad纯旋转、5cm上下返回、连续参考每0.4s热替换。所有从实际名义TCP开始，记录原始/canonical/installed/achieved链和原终止。建议并在本协议固定的工程门：6/6完整无原生abort/危险接触/实际硬限穿透；同TCP末2s位置RMS<=0.015m/max<=0.030m，SO3 RMS<=0.15rad/max<=0.30rad。多条件平均不能覆盖单条件失败。这是本任务低速接入门，不是完整控制器泛化证明。

冻结WBC行为若不满足门，保存完整证据/师兄请求，不调权重、臂腿增益、progress公式或阈值追分。准确证明的坐标/索引/调度接入错误可在停机边界修复并记为工程替换；禁止质量失败换seed或自动换checkpoint。阻断不取消其他CPU/场景工作。

**G3 TEACHER_PILOT_READY：** G2通过后，最多16个开发episode只用于教师/新任务工程，layout287010..287025。冻结teacher/scene/scorer后，运行固定PILOT8：287000..287007，要求>=7/8完整32秒成功。不得重抽、替换坏布局或用开发成功片段当固定panel。原始所有失败保留。需改变任务几何/碰撞/教师科学逻辑时停在新决策，而不是边评边改。

任务成功：持续真实夹持>=0.30s、相对初始方块中心持续抬升>=0.02m且>=0.30s、水平搬运>=0.10m；最后28..30s全部41个20Hz采样点中，红块水平角点位于16cm蓝区内且留1cm余量，落在桌面（中心高度偏差<=0.005m）、线速度<=0.02m/s、角速度<=0.10rad/s，非夹持且TCP中心距红块>=0.08m。无base/table非法接触、no native abort/actual joint-limit penetration；允许脚地面和任务规定的夹爪/物体/桌面接触。原生grasp检测需先通过无物体、单侧触碰、双侧夹持、释放负例，不拿命令closed当grasp。已达目标也不提前终止32秒。计时容差只允许已验证采样边界误差，不改变物理任务阈值。

**G4 TARGET_EXPERT_DATA_READY：** PILOT通过自动采固定TRAIN64：287100..287163；DEV16：287200..287215。每布局一次；分母保留64和16，成功TRAIN>=56/64、DEV>=14/16，且成功TRAIN所有有效H40标签按上述容差通过，才签发目标数据就绪。未过数据门仍交付完整原件和失败，不无限采集补数。新确认287300..287331仅预留不得采集/渲染/教师查询。

**G5 FINETUNE_INPUT_READY：** 在现代GR00T环境仅用实际processor/tokenizer/CPU collator读取新的目标数据、检查per-episode语言/时间/rot6d/gripper/mask与批次；不加载神经网络、不做canary梯度。输出一份明确的Target-only/Source-to-Target后续方案；建议首个baseline为官方预训练起点+声明目标EEF10接口+native loss，冻结WBC，之后才做带来源标注的Panda混合/恢复/多任务。训练步数、参数范围、噪声、比较和新确认须由后续协议注册，本GO不授权训练。

## 6. 可执行交付，而不是只写文档

Codex必须实现并实际验证对应入口（以下是要求实现的CLI，不是声称现已存在）：
- `scripts/pi0287_target_sim/doctor.py`：只读runtime/资产清单/阻断；
- `scripts/pi0287_target_sim/run.py --mode scene-smoke|reference-fixtures|teacher-pilot|collect`：显式配置、budget、seed和单一lease；
- `scripts/pi0287_target_sim/validate_dataset.py`：数据/时钟/编码/独立scorer；
- `scripts/pi0287_target_sim/prepare_finetune.py`：仅目标训练输入/processor配置检查；
- 原生兼容环境lock、WBC实际start命令、本地路径模板、数据schema、CPU测试和真实运行收据。

至少交付：同步目标双相机真实视频/关键帧、任务成功与失败完整列表、raw/installed/achieved轨迹、物体与夹爪反馈、SOURCE_IDENTITY、GATE_SUMMARY、DATA_MANIFEST、TRAINING_READINESS，以及师兄需补充的checkpoint/config/TCP/相机/夹爪说明清单。只能从保存RGB编码视频，不把示意图或零动作资产展示当任务成功。

## 7. 资源、权限与停止规则

一个现有owner，最多3个worker（环境/bridge、任务teacher/data、CPU验证/训练入口）。同一时刻最多1个合法目标仿真GPU运行进程；n_envs从1/2验证后最多4。不申请H200进行神经训练，不取消其他项目任务。

新增预算独立于PI0286，旧消耗/失败/曝光永久保留：环境/索引/参考/夹爪/热替换测试<=48个env-episode；教师开发<=16；PILOT8；正式采集80；技术替换<=16；累计hard<=168个env-episode。每个并行env、reset的新episode、提前失败均计账；有比分母内的有效质量结果不得同版重试。工程替换必须证实没有改变科学行为并保留原件，最多两轮停机接入修复。

进一步总cap：physics integrations<=2000000，WBC actor forwards（arm+dog分开计）<=600000，教师查询<=70000，RGB pairs<=120000，GPU reservation/service<=12小时、CPU<=128 core-hours、新输出/cache<=80GiB。GR00T forward/optimizer=0，WBC optimizer=0，T2数据/训练=0，硬件=0。所有cap同时生效；不能从episode数伪造未记录细计数，UNKNOWN不得标PASS。

复用合法私有WBC与旧源素材，不重新下载全部HF模型；本地数据先闭合、manifest同步即可，HF批量归档不阻塞工程/数据门。不得删除旧唯一权重、原始数据或其他项目；磁盘不够先报告精确类别和需求，未获授权不扩清理范围。

避免旧过度治理问题：只对科学gate停止；相同语义的依赖/路径/多actor索引/序列化修复可在停止边界继续，累计预算内允许至多3次串行启动尝试、任何时刻最多1个活跃lease，不以“此前接受过一次任务”永久阻断预执行工程恢复。公开readiness不是签字依赖。无新邮件、监控、后台无限goal、PR合并或独立长期分支。

Terminal可为ENV_READY_EXECUTOR_BLOCKED、REFERENCE_QUALITY_FAIL、TEACHER_PILOT_FAIL、PARTIAL_DATA_NOT_ADMITTED、TARGET_EXPERT_DATA_AND_FINETUNE_INPUT_READY、TECHNICAL_BLOCKED、BUDGET_STOP。没有任何状态自动等于GR00T policy成功。selected/deployment=NONE。

## 8. 后续决策

本轮通过意味着目标环境能生成可信任务数据，不能推断GR00T必然可学或跨本体泛化已成立。下一轮优先原始pretrained GR00T在这个目标域上的有限native微调；保留Panda-only、Target-only、Source->Target的对照分开报告。只有基本执行与观测成立后，才单独决定历史/低层反馈输入、BASE软偏好、多任务或WAM，不同时改所有因素。

支持材料《从第一性原理审视GR00T＋WBC项目》明确提出先验证目标执行器正控、再建立跨本体VLA基线；本协议采用这一先后顺序，不把其WAM研究设想当作当前新增授权。第一目标始终是正确参考在真实目标物理中完成一次可审计的物体操作，并在固定布局组上复现。
