# PI0290 — 独立MuJoCo物体任务、物理Teacher与可训练数据pilot

**状态：USER_REQUESTED NEW SCIENTIFIC GO；只授权本文件规定的Corelab数据阶段。** Authority: `PI0290_INDEPENDENT_TASK_SIM_DATA_PILOT_GO_V1`。固定源分支 `pi0267/executor-decoupled-recovery`；执行release由包含本协议及机器任务卡的精确Git SHA与Issue #11 canonical绑定。接手并执行的是**已验证SSH的local-pc唯一owner**，不是GitHub cloud Codex。发布GitHub并不等于远端开始执行；必须有真实runtime/run收据。

## 0. 科学目标、边界与父证据

研究章程 `GROOT_OBJECT_MANIPULATION_MAINLINE_CHARTER_V1` 是本实验前提：预训练GR00T以后学习视觉/语言条件下的EEF SE(3)+gripper任务行为，师兄WBC只负责本体控制并独立推进。本轮科学问题：**不调用目标WBC，是否能用独立的有限驱动动力学夹爪在真实MuJoCo接触下产生完整、可复现、语义可信且支持未来GR00T训练的抓放示范？**

- `object_task`：初始单物体T1，将4cm立方体从源位真实抓起、抬升、搬运到16cm目标区域，释放并撤离，最后稳定保持。32s完整episode，语言明确，不引入多任务结果。
- `high_level_contract`：未来GR00T使用同步body/wrist RGB+instruction+actual TCP/gripper；监督未来H40/20Hz EEF XYZ+row-rot6d(9)+绝对gripper(1)，每0.4s提交最多前8点；这不是腿部/臂部joint训练。当前只有Teacher与工具控制器，GR00T并不运行。
- `frozen_wbc_identity`：**NOT_USED**。归档 `wbc_mujoco_minimal.tar.gz` SHA256 `6dd13098de5a58327fb0a0a5b3ff749e0a5534510375b92fba3f8fb50a322d52` 只作可追溯的MuJoCo、工具几何、TCP及机器人外观参考；旧 `WBCv1.zip` 是Isaac Gym/RoboDuet不同系统。WBC原始代码/二进制/MPC代价/腿RL/PD/安全门不修改、不调用，也不用其G1质量FAIL过滤独立工具示范。
- `single_intervention`：搭建独立物理任务环境+教师/记录/验证一条封闭链；不训练GR00T、不加入BASE head、TaskAnchor、UNITAS完整WAM，不接入真实WBC。
- `claim_boundary`：合格pilot只证明 **INDEPENDENT_DYNAMIC_TOOL / SIM_ONLY / NON_REALTIME** 下物体示范和数据管线成立，不能称Go2+ARX5全身物理、WBC、部署、跨本体或真机成功。selected/deployment继续NONE。

已封存父结果不重写：PI0285 CONFIRM_A20/32 <26/32；PI0286 C0 TRAIN2/8且DEV NOT_RUN，C1 TRAIN8/8、DEV5/16且CONFIRM NOT_RUN；PI0289安全启动PASS但hold SO3 0.06520rad >0.05和clock residual 0.04780s >0.025，另5项G1 NOT_RUN，目标物理Teacher数据0。local-pc接管终局 f906059e9f752f7f6110a28f0847085ce9caf6bf 已通过两端SSH及HF只读核查，四旧分支已保留tag并退役head至5个；旧PI预算不继承。

**唯一执行owner与分工：** local-pc负责代码/运行审核/SSH/小报告GitHub；Corelab完成全部MuJoCo、Teacher、CPU、渲染、录像；Hopper H200本轮仅不带神经负载的既有资产可用性已知，不新提交PBS、不推理、不评价模型。两服务器无法直连；本轮不需要双SSH在线inference relay。

## 1. 实际WBC交付包的使用范围

首选从Corelab已解压的 `wbc_mujoco_minimal_20261009_6dd13098/wbc_mujoco_minimal` 或其不可变私有副本**读取**MuJoCo3.2.7有效模型/几何/单位/TCP/finger声明，不从旧IsaacGym迁移控制结构。最新包在原生profile是OCS2+臂PD+12腿RL运行快照，**不是可直接生成物体操作数据的现成场景**；新场景须另建tool-domain副本。TCP/指爪参数应从交付源复核，不根据之前文字强行填充。

交付模型已声明ARX5双指平行/镜像运动、每指0..44mm行程、gripper_center TCP在link6附近 `(0.1424,0,0.0001057)m`（待绑定实际MJCF/URDF）。这些只说明**模型的SIM几何**；无法推断实物速度、力、pad摩擦或部署认证。正式本轮可声明自己的有限驱动/摩擦/质量假设和来源，并标记 `SIM_ONLY_PROVISIONAL_GRIPPER`；没有实物夹爪规格不是独立工具研究的硬阻断，但绝不能把模拟参数称为部署值。

不得改原始WBC包（包括原model、gain、policy、MPC与controller）；只在独立私有overlay或者新 `src/go2arx5_vla/task_data/` 开发。原本 `target_mujoco.dataset.validate_manifest()` 需要冻结WBC的G1/G2/G3，它的意义不能放宽；独立工具另设 `INDEPENDENT_DYNAMIC_TOOL_DATA_V1` manifest/validator/teacher ID。可提取通用SE(3)、评分、记录逻辑，但别把 `GO2_MJ_PLACE_V1` 或 `FROZEN_NATIVE_OCS2_MUJOCO` 原数据改名冒充独立数据。

Corelab已验证WBC运行环境和私有包存在，**但独立任务MuJoCo Python/可执行runtime并未验明完整安装**。第一步检查实际python import mujoco、library版本、模型资源可读、headless渲染、权限、占用；若需隔离环境可在自己目录建立，不sudo、不更改共享环境、不打断其他GPU任务。先CPU smoke再申请自己合法空闲render资源。绝不报告“包已解压”=“独立任务环境已完成”。

## 2. 必须真实实现的工具/物理场景

建立独立、最小、可审计MuJoCo场景：桌面、物理可动4cm立方体（建议0.05kg、材质/接触参数与来源登记）、16cm目标区、独立工具与双指夹爪。源/目标中心相距至少0.15m。基准不要求移动Go2底盘；如在画面中摆放Go2外观或固定观察相机，应真实标记 `non_wbc`，不计全身移动成功。

**工具控制：** Teacher在task space提供TCP目标和开度；独立executor用有限力/速度的物理控制（例如受限Cartesian力/力矩伺服＋真实指爪actuator）追踪，实际MuJoCo动力学与接触推动物体。首次reset允许显式合法置初态；episode开始后不得通过逐步改 `qpos/xpos`、mocap teleport、隐式约束、weld、snap、磁吸、物体pose assignment将物体移至目标。任何夹爪命令和真正闭合/接触须分别保存；禁止用命令闭合冒充抓住。

工具本体惯量、finger joint映射、qpos/qvel/actuator index、力和速度上限、TCP link、pad geometry/摩擦、桌面/物体碰撞、sim step与solver配置必须构成一个可重现freeze；力学值要有来源/明确的SIM假设并固定在pilot之前。不要照搬规划URDF的“100N/5m/s”声明作实物能力界限。支持单侧finger接触负例、抓空负例、开合极性/实际位移测量和真实双侧contact证据；发生异常穿透/非法碰撞或NaN立即退出并保留。

**相机：**真实MuJoCo渲染 `body_rgb` 和 `wrist_rgb` 两路，至少256×256，目标20Hz；分别记录K、T_world_camera、camera parent/frame、每帧仿真时间、snapshot ID和有效mask；wrist相机真实跟工具移动，body相机若固定则明确标记固定rig，不能冒充Go2已校准部署相机。保存有用的rgb视频和原帧，严禁用Panda或viewer截图冒充训练双相机。真实RGB和状态相对同一物理tick，允许的传感器最大偏差建议<=25ms（事先锁定实现与错误统计）。

## 3. Teacher与动作时序

基于已有Panda物体阶段与 `src/go2arx5_vla/target_mujoco/teacher.py` 的逻辑，重新实现独立task teacher：
`approach→align→close→verify_grasp→lift→carry→lower→release→withdraw→hold`；每次转换读实际物体/指爪/执行反馈。教师离线可读取仿真特权物体状态计算抓取和纠正；Student将来只能看到正常传感器。教师实际作出的决定、实际接收/已commit的控制和物体实现分开记录。教师不可把将来才发生的接触写进当前监督。

高层参考先保留H40、20Hz、`t=0..1.95s`；每0.4s重新从当前实际TCP状态规划，独立executor消费明确的前8个控制点，保留chunk版本和边界。保留 `planned_H40` 和 `committed_prefix`，中途发生恢复/安全停止不能拼接穿过接管与reset的未来标签。按固定行优先rot6d与实际GR00T已有codec做静态往返单元测试，不在本阶段启动GR00T处理网络。若使用不同工具控制频率，显式插值并记录执行频率，不能直接把20Hz理解成MuJoCo动力学步长。

完整每次episode按sim计32.00s，存从0到32每0.05s共641个同步物理/传感器sample（若中途异常，保留真实不完整分母并标FAIL）；物理步长若采用原2ms，则每完整episode 16,000步。界限必须由sim时钟而不是宿主wall时间决定。

## 4. 独立物体任务成功标准（冻结前登记，不因结果调阈值）

基于仓库 `src/go2arx5_vla/target_sim/scorer.py` 的既有T1物理判据提取engine-independent函数、另设独立工具source ID：
- 必须保存实际物体接触：真实左右指与同一物体接触、对象与指爪相对运动和夹爪测量有效；真实抓握连续至少0.30s，不能靠发 `close`；
- 真实持握抬升：物体质心相对初始至少0.02m，并且实际处于抓握状态；真实持握水平搬运：相对初始XY至少0.10m；
- 实际释放、工具撤离至少0.08m（TCP相对物体中心），物体所有八角点在目标区域内且各坐标落入zone center±0.07m内部（16cm目标、1cm margin）；物体中心z相对桌面+0.02m误差<=0.005m；
- 最终物体线速度<=0.02m/s、角速度<=0.10rad/s、有效接触与gripper、无持续抓握；
- t=28.00..30.00共41个20Hz采样点上述条件**全部通过**，且整个0..32s无安全事件、abort、invalid contact/penetration、无reset/takeover或数据中断；
- 评分器仅读取保存的physics与真实sensor，不读取teacher stage/发出的gripper命令/GUI动画作为物理成功证据。重复独立CPU评分及hash核验；保留抓握/抬升/搬运/释放/撤离各失败漏斗。

这些门是T1基线的预注册继承，SIM-only工具可采用不同指爪contact sensor实现，但语义等价，必须登记接触证据如何获得。任何显式数值变化必须在第一条pilot之前登记为新的task/scorer身份，不能看pilot成绩后追改。

## 5. 完整episode数据契约：从第一条开始支持TaskAnchor/UNITAS

`POLICY_OBS`: 两路RGB图像路径+真实frame_hash/snapshot_ID、语言、**actual** TCP SE(3)、actual gripper开度/开度速度、有效mask。
`CAUSAL_HISTORY`: 有序过去合法观测与已经commit的动作、episode/reset/abort/takeover边界；新episode清空，未来H40不能作为过去事件输入。
`ACTION_SUPERVISION`: 每决策的desired world task pose+absolute gripper，H40，row-rot6d/raw quaternion/time grid，实际接收/安装/commit的第一8点，来源与ref_id。Achieved pose不能替代desired标签。
`EXECUTION`: step时序、actuator目标/输出、实际TCP、实际finger/contact、误差、超时与安全。
`PRIVILEGED_LABELS`: 真实物体位姿/速度/刚体局部固定点轨迹、contact、actual current_grasped/ever_lifted/ever_carried/released、teacher阶段、success与failure reason；只用于teacher/离线event与将来监督，禁止进入可部署Student输入。
`GEOMETRY`: 相机K与各帧外参、世界/任务坐标、object frame和固定点IDs+valid/visibility；未来三维后果标签以真实物体运动为准，不能只对EEF再编码。
`PROVENANCE`: 执行源type、asset/tool/scene/teacher/physics/clock/codec/hash、seed/split、sim-only标记、file SHA256，训练统计必须只从未来TRAIN拟合。

所有episode原样保存再按独立layout/split切窗；一个episode里的H40彼此重叠，不得在训练/开发集之间随机混。Pilot数据不得与未来blind TEST混；teacher privileged不随policy_input合并。源Panda 160条与27个纠正suffix可供CPU设计参考，但**本GO不新增训练数据混合、不启动模型**；已曝光旧CONFIRM不可充当新盲评价。

独立manifest ID `INDEPENDENT_DYNAMIC_TOOL_DATA_V1`，分开 `pilot-scored`、`dataset-admitted`、`future-model-input-check`。后两者本GO的正确上限是“原生CPU codec/静态数据形状可校验”，不把无神经加载说成已经通过官方GR00T processor/collator全路径。

## 6. 固定pilot设计与可用性判据

固定一份 `LAYOUT_SEAL.json`：commissioning IDs `C2900..C2903`、pilot `P2900..P2907`，所有随机数由这些ID经过SHA256稳定映射，固定 `task scene/geometry/camera` 合法采样范围，pilot中四个来自见过的布局族、四个为未用于commissioning的布局族。Pilot的8个原始初态/语言/seed在**首次pilot启动前冻结并落SHA**，预检查几何可达性、桌面边缘和遮挡；出现不合法配置按事前规则标不合法/上报，不可看结果替换seed。commissioning允许迭代最多两版有实质区别的Teacher/物理工程修复；pilot开始时冻结其代码和完整参数，八个布局每个只能有一次有效质量尝试。技术启动故障保留真实尝试和资源账目，不能免费无限重新起跑。

**Pilot READY最低要求**：8个pilot layout均真实完成并留下有效的physics+双RGB+动作记录；其中至少7/8具有上述完整持续物理成功，并且至少3/4 held-out成功；输入泄漏/时间错位/数据codec/接触/安全负例全PASS。达不到就如实终止为 `PILOT_VALID_QUALITY_FAIL` 或 `PILOT_DATA_INVALID`；不得“补到7个成功”为止。若遭遇环境/资源阻断，输出 `BLOCKED_RUNTIME` / `BUDGET_STOP` / `BLOCKED_PRIVACY_OR_HF` 中准确一类。commissioning成绩不充作pilot结果。

### 资源上限（仅PI0290新增，不继承旧数值）
- 最多4次commissioning物理启动+8个固定pilot，**总计12次物理episode start**，每条<=32s；不足完整32s的真实启动同样计一次。无同版本quality retry；没有额外免费物理diagnostic episode。
- MuJoCo dt=0.002s时，这12条完整episode最多192,000物理积分；允许额外最多8,000个**事先在账本注册的纯工程预检步骤**，总绝对<=200,000；若使用其他dt必须在首物理运行前按物理秒重算并固定总数，不能为了绕预算单改dt。独立20Hz每完整episode641个成对camera/state sample，共最多7,692对；首次reset/reset前不计为正式episode帧但也不能隐蔽执行训练。
- Corelab渲染/物理的受控GPU服务/预留union<=1h，CPU<=24 core-hours，新产生资产<=10GiB；在用其他用户RTX不抢占/杀进程。真正容量需在运行前check quota与空间；ENOSPC立即停，不清理旧结果换空间。
- 新Hopper/PBS作业0；GR00T model load/forward/backward/optimizer/eval0；WBC调用/训练0；真实硬件0。固定TRAIN64/DEV16和任何更大teacher采集**不在本GO**。HF原始数据上传只有在目的地私有/许可批准、本人write核验、每文件SHA和READY manifest闭合后才允许；否则留私有Corelab，GitHub只发布小型脱敏收据，绝不公开私有WBC/robot/mesh/机密数据。
- 只有一个physical simulation instance；local-pc可用至多3个CPU/代码worker，但实验总账统一owner/lease。不能把其他项目的H200作业、Corelab其他用户的GPU作业视为本实验资源。

## 7. 最小分阶段执行顺序（允许主动解决工程阻断，不必每个小修复反复申请）

**D0：静态源与运行资格**：校验现役SHA、私有资产精确身份、MuJoCo runtime、实际sim engine与XML依赖、render；对需要修复的环境依赖仅在隔离本实验目录完成。安全/隐私/配额/同机资源门失败则BLOCK，其他CPU可继续。

**D1：实际物理与接触/相机资格**：离线CPU契约测试、命名DOF/actuator映射、含真实物理的左右夹爪、抓空/单侧、双RGB校准与时钟，最多commissioning四次。没有真实物体交互不要开始pilot。不得用手工物体跟随或假夹爪PASS。

**D2：冻结Teacher并执行8 pilot**：第一个pilot开始前冻结全部布局/scene/teacher/driver/code/scorer/codec/version及预算文件；发布pilot seal。执行全部原始分母，保存每条视频、物理、事件、refs、RGB原件、日志和失败。

**D3：独立CPU复算与数据读取**：从保存文件独立重算任务结果、摄像机与命令一致性、H40和gripper极性、序列/时间/隐私；实际跑通 *不加载GR00T网络的* tokenizer/codec/shape检查。不要把手写伪batch当官方GR00T processor真实forward。

**D4：归档和返回**：报告真实通过／失败，给下一独立数据扩展/官方预训练GR00T微调的受控设计建议，不提前运行H200。

每完成一个实际阶段提交小型 `D*_RECEIPT.json` 和源码/runner测试；第一条真实物体完整成功立即报告固定证据路径和可观看短视频（若隐私/资产许可允许），不用等全部pilot跑完才给阶段收据。若暂无法访问GPU可先做CPU/engine等价验证，不假装物理pilot已执行。

## 8. GitHub/HF协同与实际交付

只在现役 `pi0267/executor-decoupled-recovery` 当前工作树小范围新建 `src/go2arx5_vla/task_data/`、`scripts/pi0290_task_data/`、`tests/task_data/`、`reports/PI0290/`；不要自动重写main、删除旧历史、覆盖私有WBC或把WBC机密推入public GitHub。所有实际代码需可从固定source SHA复现，公开repo只提交小报告、测试、可发布的脚本/配置。

本GO owner最后提交：
1. `REPORT.md`（实际功能/方法、关节/工具/接触/相机图、失效机理、可重复命令与真实结果）；
2. `SCENE_AND_EXECUTOR_FREEZE.json`（sim-only工具及参数、资产hash、物理引擎、camera、clock、gripper、score）；
3. `TEACHER_AND_LAYOUT_SEAL.json`（全部12个ID、pilot split、固定teacher config与采样规则）；
4. `PILOT_RAW_MANIFEST.json`（每次attempt包括失败、帧/哈希、privileged隔离与取回位置），`PILOT_SCORE.json`；
5. `DATA_ADMISSION_AND_CODEC.json`、`RESOURCE_ACCOUNTING.json`、源/tests/运行命令、至少一段真实双相机与物理证据（在许可的资产通道）；
6. 如完成HF，`HF_TRANSFER_READY.json`含repo_type、visibility、revision、prefix、每文件SHA，双端按需readback。若未完成，必须写具体BLOCKER，不虚构上传完成。
7. 仅设计：`PI0291_B0_PRETRAINED_GR00T_FINETUNE_DRAFT.md`，声明 NOT_AUTHORIZED；H200训练不得包含在本GO。

科学记录：每个阶段可于Issue #11留言实际进展、每次终局有固定commit/receipt和回读；仓库管理Issue #47不混入科学统计。历史PI0286、PI0289原FAIL与未运行保持不变。终局可用 `INDEPENDENT_DATA_PILOT_READY`、`PILOT_VALID_QUALITY_FAIL`、`PILOT_DATA_INVALID`、`BLOCKED_RUNTIME`、`BLOCKED_PRIVACY_OR_HF`、`BUDGET_STOP`。任何本轮实际通过均不把 selected/deployment 改成非NONE。

**特别提醒：** 本GO授权的是在原权限下真正建设和运行独立MuJoCo任务pilot；不是在Issue里仅追加另一份设计文档。缺合法运行入口、存储、GPU/资源或许可时如实给出可验证阻断和可完成CPU产物，不绕权限。