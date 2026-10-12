# PI0291 — 夜间 ARX5 实体模型场景、教师采集与训练数据交付

- 类型：USER_REQUESTED_BOUNDED_OVERNIGHT_GOAL；由现有 local-pc owner 接手执行。
- 日期：2026-10-13，Asia/Singapore。
- 目标截止：2026-10-13 09:00 SGT（2026-10-13 01:00 UTC）；同时受接手后最多 6 小时约束，取较早者。
- 现役分支：`pi0267/executor-decoupled-recovery`。
- 制定时审阅 HEAD：`b3e1df2d8c944d79c3dfd02c8df7624ae2bd8f95`。接手时刷新 #11/#47/真实 HEAD；不能用 main 旧 runtime 代替。
- 科学记录：Issue #11；既有 Corelab Phase-2 维护继续在 Issue #47。
- 本文给现有 local-pc Codex 一个可连续执行的有限任务：满足本文件前置门即可进入下一阶段，不在每个阶段重新等待一份 GO。GitHub 发布只证明任务已交接；实际启动、计数、通过与失败以 owner 运行收据为准。

## 1. 今晚要完成什么

**主交付：一个使用真实 ARX5 机器人资产和关节动力学、具备可驱动夹爪与真实物体接触的仿真抓放场景；一批经过物理与数据双重准入、可由原生 GR00T 数据处理链读取的教师数据；以及已经落地的 Corelab 路径索引与既有清理任务收据。**

“真实 ARX5”在本 goal 中指**真实机器人模型的物理仿真**。默认完成 `FIXED_BASE_ARX5_TASK_SIM`：保留真实六轴手臂、安装关系、夹爪和必要 Go2 外观/碰撞，明确固定底座/支撑约束，由独立、有限驱动的关节控制器执行任务参考。它是目标几何下的固定底座适配数据；不能计为实体机器人采集、移动全身控制成功或零目标数据迁移。

必须保留研究主线：视觉＋语言＋actual TCP/gripper → GR00T EEF SE(3)＋独立夹爪参考 → 未来冻结 WBC。关节 IK/PD 可以作为本轮教师执行工具，但关节/底盘命令不成为高层 VLA 的动作标签或特权输入。师兄的 OCS2/WBC/RL 原件与控制语义继续冻结；本轮 WBC 运行和调参均为零。独立教师数据不以旧 PI0289 WBC 资格通过为前提，另建来源与资格，保留原 FAIL。

优先级：
1. ARX5 实际模型＋真实接触抓放＋原始双相机证据。
2. 首批 TRAIN64/DEV16 尝试队列与可读取的数据交付。
3. 接续现有存储 M0–M3，完成路径解释、允许范围内的回收与回读。
4. 最后收尾封存与次日训练交接。今晚不训练 GR00T，不用长训练掩盖数据缺口。

## 2. 从真实现状继续，不重跑已完成的 PI0290

制定本 goal 时已经读取：
- PI0290 冻结执行源 `c4853f3b7bd9747dff485d58780a0c0abb1e3402`；8/8 完整独立动态工具抓放成功，held-out 4/4；仅限 `INDEPENDENT_DYNAMIC_TOOL/SIM_ONLY/NON_REALTIME`。
- 每个完整 pilot：32 s、16,000 个 2 ms 积分、641 对同步 RGB/状态；原始 commissioning 失败及旧相机版本完整保留。
- PI0290 私有 HF dataset：`leoyuer/pi0290-independent-dynamic-tool`，prefix `PI0290/v1-c4853f3b`；最终完整 revision `0fd9247ba5426eb54ebf152912432302b193b479`。Hopper 只读回 7,511-byte READY；不等于已下载数据或运行训练。
- `scripts/pi0290_task_data/export_native_windows.py` 已实现；此审阅 HEAD 未见 `NATIVE_WINDOW_EXPORT.json` 或 `HF_CODEC_WINDOWS_READY.json` 实际收据。官方 GR00T processor/collator 仍记录 NOT_RUN，不能把静态 codec PASS 写成训练批次 PASS。
- `NEXT_ASSET_REUSE_READONLY.json` 已识别 native MJCF 的手指动力学缺口及第三方 Go2-X5 资产线索，但 ARX5 物理启动与 Isaac runtime 资格仍为零/未通过。
- 既有 Corelab 第一阶段已实际回收 44,017,266,688 allocated bytes（40.994 GiB）/14 个派生 BF16 分片；不能重复记账。
- Corelab 第二阶段 GO 与暂停归档进程补充授权已发布，审阅 HEAD 尚未见实际 M0–M3 终局或进程退役收据。接手必须刷新，不假设它们仍未执行。

原 PI0290 的 D4 如果还在收尾，先读取其真实进度，只补缺的收据/导出/归档，不重新采 8 条，也不覆盖原 revision。PI0290 数据作为工具域的资格证据留存，不改称 ARX5，不混入本轮正式 ARX5 TRAIN/DEV。

## 3. 授权范围、唯一 owner 与资源分工

本 goal 新增授权：隔离 ARX5 场景工程、零积分资产/相机检查、限定物理 commissioning/pilot/正式教师采集、CPU 数据验证/原生 processor/collator、TRAIN-only statistics、私有数据归档与训练交接文件。旧 adoption/read-only/PI0290 的禁止新物理条款不阻止本 goal 明确授权的新 ARX5 域工作；旧实验预算、旧失败和旧数据资格仍不改写。

local-pc 是唯一调度/提交/预算 owner。可让子 agent 并行做场景代码、数据契约、只读路径审计；单一 live physics owner，清理 mutation 也只由 owner 串行执行。禁止另起独立云端 owner 或同名重复进程；已有健康作业/lease 先辨认，不能因它占资源就结束它。

- Corelab：场景、教师/物理、渲染、预处理、CPU 数据验证与存储维护。优先复用已验证 MuJoCo 3.2.7 和隔离运行环境。
- Hopper：本轮只允许现有授权渠道的小 manifest/少量数据交付验证；GR00T 权重加载、神经推理、训练和新 PBS 作业全部为 0。原生 processor 只能运行不构造模型的 CPU 数据处理路径。
- 两服务器不能直连；资产通过 HF 固定 revision/prefix/hash。复用既有认证，不输出凭据。
- GitHub：代码、protocol、任务卡、sanitized 小收据。新机器人 mesh/private WBC/大数据不入 Git。
- 原始硬件动作：0。部署选择：`selected=NONE; deployment=NONE`。

科学与维护是两个有界 workstream：Phase-2 的“无物理工作”仅限定维护授权本身；新 ARX5 物理由本 goal 独立授权。维护源文件缺失只阻断其删除动作，不阻断不依赖这些路径的合法 ARX5 工作。

## 4. 有限预算与时间安排

### 4.1 新 ARX5 域预算

| 项目 | 上限 |
|---|---:|
| commissioning（含物理诊断和负例） | 12 episode starts |
| 冻结 pilot | 8 starts：4 开发族＋4 新资格布局族 |
| 正式 TRAIN 队列 | 64 starts |
| 正式 DEV 队列 | 16 starts |
| 总物理 episode starts | 100；失败/早停同样计数 |
| 单 episode | 最多 32 s 仿真时间 |
| 2 ms 物理积分 | 最多 1,608,000：100×16,000＋注册工程预检最多 8,000 |
| 全部100次物理尝试的双 RGB/状态样本 | 最多 64,100 对；其中正式80次最多51,280对；另记零积分诊断最多256对 |
| 自有 Corelab GPU 服务/预留 union | 最多 4 GPU-hours、最多 1 张已空闲且获准使用的 GPU；没有就用现有 CPU 渲染 |
| 总新增 CPU 使用 | 最多 48 core-hours；核验、渲染、归档与本轮维护均记账；并行最多 8 CPU |
| PI0291 新产物峰值 | 最多 40 GiB，含原帧、导出、临时压缩与已验证下载副本 |
| 新大型引擎/驱动安装 | 0；隔离小依赖总下载最多 2 GiB |
| GR00T load/forward/backward/optimizer | 0 |
| 新 Hopper PBS / WBC / 实体硬件 | 0 |

8,000 工程预检积分也必须预先登记；若它们在单独物理 episode 内执行，该 start 消耗 commissioning 配额，不存在“免费物理诊断”。不同 dt 仅在第一次物理前证明必要、按相同物理秒与积分上限登记一次，不能为绕预算调整。所有计数以实际积分/采样/CPU时间为准；观测不到的利用率写 UNKNOWN，不拿 reservation 冒充利用率。

正式数据预期为 64＋16 个**尝试**，不是预先声称 80 条成功示范。12/8/64/16 额度不可相互挪用，不按失败结果补 seed；达不到则如实输出已采与已准入数量。

### 4.2 时间分配

以 owner 首次实际接手时间 T0 为准。T0、run UUID、截止时间及已消耗计数在断线/换窗口/重连/重新发布后全部保持；恢复时先读原账本，不再给6小时新额度。剩余时间为 min(原T0＋6小时，2026-10-13 09:00 SGT)－当前时间：
- T0–20 min：N0 接管、现状/路径/剩余预算与环境确认。
- 20–90 min：N1 真实模型、夹爪、场景与双相机。
- 90–150 min：N2 有限 commissioning 与冻结 pilot。
- 150–270 min：N3 正式 TRAIN/DEV 分批采集，边采边审计。
- 270–315 min：N4 原生数据链、统计、manifest、HF 交付。
- 最后至少 45 min：N5 最终报告、恢复入口和独立回读。

维护 M0–M3 的 owner 主动操作时间最多 60 min，利用采集等待并行只读检查；大扫描/旧全量归档不能挤占场景主线。所有工作仍受总 CPU、空间和截止约束。每条 pilot 后根据真实耗时估算剩余容量；deadline 前 45 min 停止新增 physics，封存已完成数据。不能到早上自动续一夜。

## 5. N0 — 接续与冻结入口（不重新做整仓研究）

1. 完整读本协议、机器任务卡、#11/#47 最新增量、现有 local goal/lease、PI0290 D4 最新状态。
2. 固定 read SHA、protocol hash、T0/deadline、run UUID；写 `OVERNIGHT_STATE.json` 和 `BUDGET_LEDGER.jsonl`。源代码每个实际运行版本绑定完整 SHA，不以最终报告 SHA 冒充运行 SHA。
3. 尊重 dirty worktree；在当前现役工作流中作增量提交或隔离工作树。只合入本任务相关变更，不 reset/覆盖他人的改动，不重跑旧 HPC owner。
4. 读取真实私有 runmap。NFS 线索 `/home/liuyu-nfs/Projects/go2-arx5/`、4080S SSD 线索 `/var/cache/users/liuyu-nfs/Projects/go2-arx5/` 必须现场核验；新输出用独立 `PI0291/<run_uuid>/`，不覆盖 PI0290/PI0276。
5. 一次有限检查包版本、GPU/CPU占用、已知 mount/allocated/inode/可写路径，测首批真实产物增长并按剩余队列估算峰值。容量需覆盖剩余预测＋10 GiB 余量；个人 quota 不可读则记 UNKNOWN，结合实际写入/错误处理，不声称 df 就是 quota。
6. 不把归档完成当新原件可删的依据。若无需回收即可完成数据工作，先跑数据，维护按本来的窄范围执行。
7. 若接手时截止已过，只作只读/已有产物收尾，不启动新的夜间采集。

输出：`ADOPTION_DELTA.json`、`RESOURCE_AND_PATH_BINDING.json`、真实 next step 和可继续/局部阻断状态。

## 6. N1 — 把真实 ARX5 模型变成能采集的任务场景

### 6.1 资产与引擎选择

优先采用已验证 MuJoCo 3.2.7，加用户交付 native Go2-X5/ARX5 资产的**独立场景扩展**。默认不要在今晚迁移到新 IsaacLab 栈。

必须核对两个附件/服务器副本身份：
- WBCv1 ZIP SHA256 `3c59a6d5f8f924df0b838ad2d501d3a853274d741db191a9de453f58212063ee`。
- 最小 MuJoCo包 SHA256 `6dd13098de5a58327fb0a0a5b3ff749e0a5534510375b92fba3f8fb50a322d52`。
- native `go2_x5.xml`、`arx5_ac1.urdf` 与 SOURCE_MANIFEST 按 `NEXT_ASSET_REUSE_READONLY.json` 的精确 hash 校验。源包不原位修改。

如现有第三方现成资产能减少工作，可在 N1 时间内静态审阅已经固定的 Go2-X5-lab 等来源；不能把另一项目的 joint action、stage_one_hot 或固定腿假 cmd_vel 直接搬成共享 VLA 契约。涉及不同 TCP、相机、夹爪驱动、许可必须逐项核对。只有在本机已安装合格替代引擎、无需大安装且确实解决 MuJoCo 阻断时，才可在首次 physics 前选定一个替代引擎并冻结其身份；不双引擎铺开或在 pilot 后切换。更换后仍须通过本文件同样的物理、相机、数据门与预算。

### 6.2 可复用和必须补齐的部件

已有真实六轴运动链与 mesh，不造抽象悬浮夹爪替代 ARX5：
- native TCP `x5_ee/gripper_center` 在 link6 的声明偏移是 `[0.1424, 0, 0.0001057] m`。
- 另一个 Go2-X5 来源宣称 `[0.08657,0,0] m`，两者不能直接互换。
- 保留实际关节轴/限位、尺寸、惯性与碰撞来源；缺失的接触 pad/摩擦/驱动参数可以作**明确的 SIM_ONLY 建模假设**，冻结值及理由，不称实物标定。

native MJCF 的 link7/link8 当前是无关节、无接触的 visual bodies，18 个 actuator 只有腿＋臂；必须用 URDF 的 opposed prismatic/mimic 结构补独立夹爪 joint、collision、限位、有限力/速度驱动。44 mm 是每指 stroke，不能直接当开口总宽。交付URDF的正向关节位移令两指分开；本新域归一化通道定义g=0闭、g=1开，并以真实双指位置→inner gap的单调标定验证，记录actual与desired。它不代表未提供的硬件SDK极性。

本轮建立独立 fixed-base 场景/driver：通过明确的固定底座或支撑约束固定 Go2/臂安装根，真实手臂关节在积分中运动。使用具名 joint/actuator/sensor ID；不得每步写 arm/object qpos 代替动力学，不给末端施加 free-tool Cartesian wrench 绕过整条 ARX5 臂。

不修改冻结 WBC 的 num_of_dofs 或 18 维 sensor/leg policy 布局来塞进夹爪。旧 rl_sim_mujoco 的 sensor 索引依赖 N=18；场景 extension 不等于原 WBC 已兼容新增 DOF。留好未来 WBC 集成 mapping，今晚不运行原 WBC 做资格补测。

### 6.3 首个任务与传感器

只做 T1：4 cm cube → 16 cm 目标区域，真实抓起、搬运、放下、释放、撤离、稳定保持。桌高、源/目标位置要由 ARX5 workspace、初态、碰撞和可达性确定，不生搬 PI0290/Panda 世界坐标；对象与目标尺寸、物理评分阈值不随失败变更。

两路真实 RGB：`body_rgb` 与 `wrist_rgb`，256×256、20 Hz。body camera 的父 body、固定 rig 或机体安装必须诚实标注；wrist camera 实际随手臂运动，避免掌部遮挡物体。相机初始可见、抓握/搬运/释放关键帧可辨识，不以事后重渲染替代原采集图像。

保存 K、T_world_optical、相机轴约定、相机 parent、TCP 与 pad frame、物体 keypoints 投影、sim tick/snapshot ID。actual TCP 优先来自同一 MuJoCo tick 的 site，另用独立 FK/几何链核对；ROS 到达时间不能代替 sim timestamp。此独立场景不声称解决了原 PI0289 ROS/MPC clock FAIL。

N1 交付 `ASSET_PROVENANCE.json`、`SCENE_BUILD_REPORT.md`、`JOINT_TCP_GRIPPER_MAPPING.json`、`CAMERA_CALIBRATION.json`、可复现 headless render 命令及原图。编译/render 成功还不等于 N2 物体任务通过。

## 7. N2 — 有限教师开发、真实抓放与 pilot

复用 PI0290 的事件记录、通用 T1 物理谓词、CPU 审计和视频工具；替换 executor/robot adapter，不盲改被冻结的旧 validator。

教师可以从真实仿真特权物体状态做 object-relative approach→align→close→verify_grasp→lift→carry→lower→release→withdraw→hold。执行端使用有限关节目标/力矩和独立夹爪控制；高层仍输出 EEF SE(3)/gripper。阶段转换由实际状态决定，不能因指令已发就写“抓住”。

最多 12 次 commissioning。允许在此限额内修复 mesh/模型导入、关节映射、相机遮挡、数据记录、teacher 可达性/阶段逻辑和**独立采集 driver**错误，记录每个版本与原始失败。不能改师兄 WBC，也不能更改成功阈值、隐去安全停止或复跑同一冻结版本的有效质量失败。至少包含抓空和单侧接触负例；有需要可从真实记录作离线 negative mutation，明确其不是新的物理运行。

物体运动必须由真实 contact 产生：不 weld/attach/snap/teleport、不把 object 随 EEF 强制携带、不给物体施加教师隐藏外力。初态/reset 设定只在 episode 开始；出现 reset/接管/abort 必须中断标签与历史。

复用现有独立 T1 scorer 的数值定义，至少验证：
- 双侧同一物体有效持续抓握 ≥0.3 s；
- 在持握中抬升 ≥2 cm、水平搬运 ≥10 cm；
- 真正释放并撤离 ≥8 cm；
- 28–30 s 的全部 41 个采样点满足目标区、速度、释放与保持条件；
- 0–32 s 无安全事件、NaN、限位违规、非法穿透或数据断流。
使用保存 physics/contact 独立 CPU 复算，不依赖 teacher phase、夹爪命令、单张末帧或最终物体位置判成功。ARX5桌面可按workspace合法设置；评分的支撑高度应绑定实际table plane＋cube半高，lift基于实际初始物体高度，保持既有5mm高度容差等物理阈值不变。不能复制旧scorer的world z=0.42常量到不同桌高；新增明确的坐标/支撑面adapter，在新身份下独立复算，不修改原PI0290结果或用事后拟合偏移消除误差。

首次实际成功后立即保存并向本地用户界面交可播放原双相机视频、小报告与 physics hash；不等整夜结束才展示。诊断视频与训练原帧对应可追溯。

commissioning 完成后，在首 pilot 前封存：engine/assets/dynamics/driver/teacher/scorer/cameras/codec/code、8 个固定 pilot IDs 与布局、正式 64/16 ID/布局生成规则、budget。pilot 4 个开发族＋4 个新的资格布局族各只跑一次。8 条必须都具备完整可审计数据，物理成功 ≥7/8 且资格新族 ≥3/4 才进入 N3。它们是教师资格数据，不是 GR00T 泛化成绩或未来盲测集。

pilot 的有效质量不达门时停止该版本扩采，不重采、更换 seed 或降阈值；继续做 N4 能完成的场景/现有有效样本打包、路径维护和精确阻断交付。封存后仅允许证明不改变动作、动力学、时间网格、传感器内容、codec或评分语义的日志/打包修复，并绑定新代码hash和等价证据。上述运行语义任一改变，就不能把不同版本拼成7/8；停止这版扩采、交PARTIAL，不追加新8条pilot。已失败条目始终保留分母。

## 8. N3 — TRAIN64 / DEV16 真实采集

只在 N2 通过、模型与教师/评分/相机冻结后执行。使用新的 PI0291 namespace；正式数据不得复用 PI0290/Panda旧确认、commissioning 或 pilot 的原 episode、seed 和实际初态。同一模板族如何进入正式队列须预注册；不强制另造全新任务分布来凑“新”。先按 episode/layout group 分 TRAIN/DEV，再切窗，重复初态/模板实例作为同一group。TRAIN 与 DEV 的 group 及 seed 不交叉，pilot 已曝光的资格“held-out”不重新命名成盲测。

单次分别执行 64 个 TRAIN、16 个 DEV 的首个尝试，按 8/16 条 chunk 写收据。不可看到 DEV 结果后调 teacher/scorer/driver 或补 seed；DEV 用于这版数据集资格审计，不能称独立最终模型 TEST。

数据质量目标沿用现有首批 teacher 规模模式：
- TRAIN：64 attempts 中 ≥56 个完整物理成功且数据合格 episode。
- DEV：16 attempts 中 ≥14 个完整物理成功且数据合格 episode。
- 所有 80 次尝试的实际原始结果、失败原因、版本和计数保留；安全停止/不完整轨迹不得伪装成功。
- 成功且结构合格的 episode 进入 BC 正样本；失败保留在 `diagnostics/` 或失败 cohort，不能把整条失败示范默认标正。DEV 原始成功率必须以全部 16 次为分母，读取的正示范数单列。
- 规模不足但小包真实有效时可交 `ARX5_PILOT_DATA_READY` 或 `ARX5_DATA_PARTIAL`；不能宣称 TRAIN64/DEV16 已全部准入。

收敛与 GR00T 能力不由这个样本量保证。本轮形成首版可复现实验数据，之后训练配方另行审定。

## 9. N4 — 从 episode 到原生 GR00T 训练数据

### 9.1 每条原始 episode 的必要契约

四类记录分开存，且不能在正式 Student 输入中串入 privileged：

1. `policy_obs`：同步 body/wrist RGB、语言、actual TCP、actual gripper gap/归一化开合/有效 mask。
2. `action_supervision`：desired world EEF/gripper、accepted/installed/committed reference ID 与时序；H40 planned tail 和真正执行过的 prefix 分别存。
3. `execution_history`：actual joint/qvel、实际 TCP、过去观测和已经提交动作、同一 sim clock、reset/abort/takeover/segment 边界、命令和 achieved 的对应关系。
4. `privileged_evidence`：object pose、contact/force、teacher phase、keypoint/任务事件与 score，仅供监督和评分。

所有字段绑定 domain、asset/scene/teacher/driver/scorer、原生 processor/pose codec、git SHA、episode UUID、seed/group/split、单位、frame、time grid 和原始文件 SHA256。真实 joint state 可作执行证据，不能偷偷进入共享 Student state。

### 9.2 时间/窗口与相对动作的两个具体坑

沿用 32 s/20 Hz/2 ms 的已审计时间网格：
- 观测：t=0..32，共 641 个；
- 已执行命令：t=0..31.95，共 640 个；终帧没有虚构新命令；
- H40：从第 i 个已提交命令取 `[i:i+40]`，offset=0..1.95 s；每个完整合法 episode 最多 **601** 个窗口，不是 602；
- 完整且全部准入的 80 条最多 48,080 窗口，其中 TRAIN38,464、DEV9,616。这是推导上限，不是独立样本量或承诺产量。
- 必须检查窗口内 40 个命令确已提交且不跨 reset/abort/takeover/执行身份变更等不连续边界。正常阶段转换和replan不是截断边界；一个H40允许跨多个真实提交的prefix8。planned 未执行 tail 不冒充 committed 标签。本v1只准入完整H40，不足40点剔出训练窗口并保留原件，不用padding凑数。
- 相对表示严格使用 observation 时刻的 actual T_TCP：`T_rel = inverse(T_actual_at_obs) * T_committed_target`；完整 SE(3)、native row-rot6d＋absolute gripper。不得只相减 XYZ 或改成列优先 rot6d。

**官方原生数据链与自制 NPZ 不能直接混用。** 已有 PI0290 exporter 导出相对动作；原生 GR00T loader 的 `use_relative_action=True` 路径会自行把绝对动作转相对。要给它提交绝对 EEF10 commands＋实际绝对 state，或明确解码回绝对后再交原生变换；不能将已相对 NPZ 再相对一次。固定使用项目当前验证的官方源码/pose codec revision，不能从网络 main 随手换实现。

不要盲复用旧 `target_mujoco/dataset.py` 的 WBC-required admission 或 602-window 假设，也不要套用另一 `target_sim` 的 i+1 时间偏移。为 `FIXED_BASE_ARX5_TASK_SIM` 建独立 validator/manifest，复用经过核对的通用部件，旧 FAIL 和旧协议不修改。

### 9.3 必须实际跑完的数据验收

在不构造/加载 GR00T 网络的 CPU 路径运行**项目 DatasetLoader → 固定官方 GR00T processor → 官方 collator**，实际使用后续训练同一数据入口。项目已有合格Dataset wrapper可复用，不强制重造另一种未确认的磁盘格式，不用手写伪batch替代：
- 读取真实导出图像/视频、语言、actual state、action、valid mask。
- 检查 10 个有效 EEF/gripper维、内部 padding/mask 与当前132D原生route匹配；132D不是132个关节。
- 单样本、至少两个不同 episode/多个真实任务阶段的混合 batch、变长 instruction、两个 data worker 的一致性与可重复读取。若原数据只有固定T1 instruction，变长文本仅作CPU collator诊断，不改写原episode language，不计为新任务或新语言数据。
- 从原始绝对命令重算相对动作、rot6d→SO3→rot6d 往返、夹爪极性/单位、时间网格、视频实际帧数与 frame hash；证明没有双重相对变换。已准入TRAIN全部完整H40通过实际官方encode/decode（含本版statistics），最大XYZ误差≤1e-5 m、SO(3)≤1e-4 rad、gripper≤1e-5；分别报告未归一化codec及实际normalization/decode误差、TRAIN/DEV preclip与clip counts，不降低门槛掩盖有损处理，DEV不回流重拟合。
- 当前/过去 obs、committed prefix 与 future label 索引分离；向 policy 输入注入 object/teacher phase/success、未来历史或乱序时间的负例应被 validator 拒绝。
- 相机 K/外参/关键点投影独立复算，检查原始 wrist 图像的物体可见性；不把零积分重渲染写回原 episode。

statistics 只从**已准入 TRAIN episode**计算，使用预先封存且可重现的抽样/加权规则；DEV/PILOT/失败/CONFIRM 不参与。说明按 timestep 或窗口拟合以及重叠窗口权重；存 source train IDs、统计文件与 hash。不得沿用 Panda 或动态工具 normalizer 冒充 ARX5 TRAIN statistics。

输出当前固定 GR00T 版本实际支持的 dataset 布局、modality config、metadata/task map、TRAIN/DEV split、statistics、episode-root路径绑定、真实 loader 命令与 `DATALOADER_SMOKE.json`。`RAW_DATA_READY`、`CODEC_READY`、`NATIVE_DATALOADER_PASS` 分别记录；官方依赖受阻时诚实交 `DATA_READY_PROCESSOR_BLOCKED`，不要靠手写张量形状变成 PASS。

### 9.4 私有数据交付与训练接口

继续使用已验证 HF 认证和 private destination 机制。PI0291 采用单独 private dataset repo 或明确独立 namespace，并在 DatasetCard 写清 ARX5 来源；不得将原 PI0290 repo 名称/说明悄悄改称 ARX5。本条允许在用户现有账号内创建这个任务必要的 private dataset，不产生公开发布。

只上传有权保存的源资产/衍生数据；私有 WBC源码、权重和未经确认可再分发的第三方 mesh 不公开。若资产不能上传，以现有私有原件 SHA＋依赖 manifest 绑定；必要 replay 资产缺失则 READY 状态精确说明，不能谎称外部可复现。

先 upload 数据/必需 metadata，再最后提交 `READY.json`；固定包含完整集合的最终 revision。逐 shard/blob SHA、文件集合和 archive member manifest 核验；另实际下载一个完整 episode/一个导出 shard 到新目录，核对原始 hash 并做读取/codec 回放。Hopper 只需读小 READY/数据描述，不重复下载全库；若进行少量数据验证须受现有合法环境和轻量任务边界限制。

输出 `TRANSFER_MANIFEST.json`（repo_id/type/private/revision/prefix、producer/consumer、files/size/sha、所有来源身份）、`HF_READBACK_RECEIPT.json`、`DATASET_CARD.md`、`TRAINING_HANDOFF.json`。训练交接包括现有官方 pretrained 及源码位置、应读的数据 revision/config/统计、后续最小训练/评估的准确待执行入口；所有训练/模型 forward 标 NOT_RUN，不自动选用旧未过确认的权重或启动训练。

## 10. 并行维护 — 继续原 Corelab Phase-2，预算不重开

必须完整读取：
- `docs/maintenance/2026-10-13_CORELAB_STORAGE_SAFE_CLEANUP_V1.md`（#47 comment6129475191，release6b4cf12e...）；
- `docs/maintenance/2026-10-13_CORELAB_ARCHIVE_PROCESS_RETIREMENT_ADDENDUM_V1.md`（#47 comment6129639251，releaseb3e1df2...）。

它们的候选范围、保护规则和已消耗额度继续有效。本 goal 不授权扩大文件删除范围，也不重做已完成的14片退役。真实维护如已结束，直接核对收据并补新 PI0291 路径登记。

### M0：读真实审计，不按摘要猜路径

local-pc `work/corelab-storage-audit-20261013/` 的四原件：
`AUDIT_REPORT.md`、`FULL_STORAGE_INVENTORY.csv`、`RETENTION_PLAN.json`、`README_STORAGE.md`，加已准备的 path checker。逐文件 hash 固定，再按候选检查，不重扫约181万文件。本 ChatGPT 会话没有这些原件；本协议给出的是路由与规则，不提供猜测的 exact deletion list。

生成 `PATH_ROLE_MAP` 与逐候选 `RETENTION_DECISIONS`：logical path、realpath、consumer host、st_dev/st_ino/nlink、logical/allocated bytes、source/recovery SHA、活跃使用/FD/lease、class、允许动作与原因。NFS 在 AMD/4080S 是共享文件；4080S SSD 是节点本地。AMD 链接指向 SSD 显示断链不等于数据丢失。以 host-aware path registry 和 resolver 表示，不能盲改链接或重复复制大目录。

登记新 PI0291 runtime/raw/export/evidence 到上述真实路径索引；保护旧 PI0290、PI0250/PI0276 依赖。把已退役模型视图登记为 METADATA_ONLY/DERIVED_WEIGHTS_RETIRED，不因目录存在就报告“可加载”。

### M1–M3：继续已经批准的窄动作

- 先验证7个既有小文件 copy/readback；最多24个既有迁移候选只 copy/verify，本轮保留源文件。
- 缓存最多9项、额外派生 model-only view最多2份；**二期累计新释放仍≤50 GiB allocated**，第一期40.994 GiB单列。必须有完整独立恢复、SHA/inode/size不变、无FD/消费者/lease，逐文件manifest unlink。没有合格候选时回收0是合法结果，不凑50GiB。
- 两个历史停止态归档PID3455361/3455362只作定位。按补充授权读取两次独立运行身份（UID/starttime/state T或t/exe/cmdline/cwd/PPID/FD/lease），确认仍为用户的旧PI0276归档角色才执行逐PID、先叶后父的已授权退役；不恢复进程写入、不对进程组/其他用户发信号。保存partial archive/log和退役收据；已退出或身份不符则SKIP，不需要重复询问已给出的窄授权。
- 进程退役不等于可以删除PI0276文件，不等于磁盘自动释放。PI0276约337.6GiB、PI0209约53.9GiB都是规模描述，仍不能整目录清空。
- 不删原始RGB/physics/视频/失败、唯一Trainer/optimizer/scheduler/RNG/processor/stats、private WBC或任何未知权限/依赖/恢复状态的对象。
- 复制/quarantine rename不计释放；硬链接/NFS副本不重复计数；自身释放按allocated与实际对象身份记，df变化另列，不冒充个人quota。
- 新ARX5采集活动路径建立排除名单/owner lease，复制和unlink不与采集争写；大I/O需要节流，额外迁移目标也单独容量预检。

实际收据仍放 `reports/maintenance/2026-10-13-corelab-storage-phase2/<session>/`：
`SOURCE_AUDIT_SHA`、`PATH_ROLE_MAP`、`RETENTION_DECISIONS`、`MIGRATION_COPY_RECEIPT`、`DELETION_RECEIPT`、`PROCESS_RETIREMENT_RECEIPT`、`RESOURCE_BEFORE_AFTER`、`UNRESOLVED_AND_NEXT_ACTIONS`。
私有 exact path/FD/审计表留本机/Corelab；GitHub仅sanitized版本。#47记录真实终局，#11引用相关路径收据。四份源审计任一不可读/不一致时，整个Phase-2为BLOCKED_AUDIT_SOURCE：文件删除全部0、进程退役pending，只做只读诊断/路径草稿；不影响独立ARX5工作。源审计通过后，单个候选权限/恢复/依赖不明只跳过该候选。

## 11. 长程执行方式与局部阻断

owner 持久保存阶段状态、下一命令、已用预算、工作目录、运行PID/lease和最后收据；每30分钟更新一次简短实际进展，阶段变化时立即落账。本地heartbeat/状态文件用于恢复，GitHub不用作高频队列。

工程修复自主进行，优先复用已验证 runtime/scorer/collector，先看具体trace再改。对同类不变错误最多2次诊断重试；未产生physics的安装/导入错误不冒充物理失败，但仍计时/CPU/下载。physics一开始即计start，所有重试计入相应限额。

主线阻断时的可继续工作：
- ARX5接触/teacher未通过：交真实模型/相机/控制/失败证据；整理有效pilot和数据adapter；继续路径维护。不回到抽象工具扩采以冲数量。
- 官方processor依赖受阻：保留physics＋raw＋codec包，完成精确输入/输出契约与缺项；不启动模型试运转。
- HF暂不可达：保留私有本地完整manifest，有限增量重试，记录LOCAL_READY_HF_PENDING；不反复上传全库，不删源。
- 清理原审计缺失/不一致：Phase-2全部delete0且进程退役pending，完成只读路径诊断。审计已通过但单个候选权限/恢复不足则候选SKIP；继续ARX5与其他确已合格的维护动作。
- 资源占用/剩余不足：CPU软件渲染或提前封存；不抢占他人GPU，不占其他项目PBS allocation。
- 时间/CPU/空间/episode预算触顶：停止新增工作并收尾，不自动增加配额、延长deadline或把未运行标PASS。

## 12. N5 — 早上必须可审阅的交付

代码与小文件落 `reports/PI0291/<run_uuid>/`；真正执行脚本、scene adapter 与配置放项目维护位置，原始大资产在绑定的Corelab/HF路径。最少交：

1. `FINAL_REPORT.md`：已完成/失败/未运行、科学结论边界、实际用时/资源/错误，不用设计稿替代实际结果。
2. `SCENE_BUILD_REPORT.md`＋`ASSET_PROVENANCE.json`＋`JOINT_TCP_GRIPPER_MAPPING.json`＋`CAMERA_CALIBRATION.json`。
3. `TEACHER_AND_LAYOUT_SEAL.json`、`ATTEMPT_LEDGER.jsonl`、`PILOT_SCORE.json`、`DATASET_ADMISSION.json`；每个attempt的来源/结果/实际数量。
4. 至少第一条成功的原双相机视频、固定几条成功/失败代表视频与对应physics hash；全量原件可定位。没有真实成功也要交失败视频。
5. `DATASET_CARD.md`、native dataset/config/split/statistics、`DATALOADER_SMOKE.json`、`TRANSFER_MANIFEST.json`与私有HF固定revision/readback。
6. `TRAINING_HANDOFF.json`：下一步可以准确读哪套ARX5数据、配置和统计；GR00T训练/forward均0。
7. 既有维护phase2实际收据链接＋新PI0291 host-aware路径索引；copied/deleted/held、独立释放bytes、进程真实状态和每项跳过原因。
8. `RESUME.md`：真实最后阶段、正确解释器/工作目录、下一条可执行命令、剩余额度（沿用原T0/run UUID，只在本deadline内有效）、需要额外决定的具体问题。

最后在#11发一个完整实际终局，绑定source/run/report SHA、数据counts、scene/teacher/loader/HF状态和维护指针；#47只报告维护实际结果。提交前后回读一致，普通fast-forward，不改main/旧实验结论。

状态采用组件向量，不强迫所有子任务共用一个PASS：
- scene：ASSET_COMPILED / CONTACT_QUALIFIED / BLOCKED；
- teacher：PILOT_PASS / QUALITY_FAIL / NOT_RUN；
- dataset：FULL_BANK_READY / PILOT_READY / PARTIAL / PROCESSOR_BLOCKED；
- transfer：VERIFIED / LOCAL_READY_HF_PENDING；
- maintenance：COMPLETE / PARTIAL_WITH_SKIPS / BLOCKED_AUDIT_SOURCE。

**最高完成态 `ARX5_DATASET_READY_FOR_FIRST_TRAINING`** 只在pilot门通过、正式64＋16次队列全部执行并封存分母、TRAIN≥56/64和DEV≥14/16实际成功准入、原始完整证据/独立CPU验算/原生loader/TRAIN统计/HF回读均PASS时给出。维护若有精确且安全的SKIP应单列，不伪造全清；任何状态都不宣称GR00T已训练、移动WBC已通过或真机已成功。

## 13. 制定依据（固定来源）

以下是计划的证据，不是今晚新结果：
- [现役审阅HEAD](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/commit/b3e1df2d8c944d79c3dfd02c8df7624ae2bd8f95)。
- [PI0290 D2/D3实际终局](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-6129292709)。
- [PI0290私有HF实际交付](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b3e1df2d8c944d79c3dfd02c8df7624ae2bd8f95/reports/PI0290/execution/HF_TRANSFER_READY.json)。
- [ARX5资产审阅与原生夹爪缺口](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b3e1df2d8c944d79c3dfd02c8df7624ae2bd8f95/reports/PI0290/execution/NEXT_ASSET_REUSE_READONLY.json)。
- [PI0290已冻结动态工具身份](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b3e1df2d8c944d79c3dfd02c8df7624ae2bd8f95/reports/PI0290/execution/SCENE_AND_EXECUTOR_FREEZE.json)。
- [PI0290原始协议和数据语义](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b3e1df2d8c944d79c3dfd02c8df7624ae2bd8f95/docs/protocols/PI0290_INDEPENDENT_MUJOCO_TASK_TEACHER_AND_DATA_PILOT_V1.md)。
- [官方固定pose/row-rot6d实现](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/data/state_action/pose.py)。
- [native窗口导出代码](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b3e1df2d8c944d79c3dfd02c8df7624ae2bd8f95/scripts/pi0290_task_data/export_native_windows.py)。
- [研究主线章程](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b3e1df2d8c944d79c3dfd02c8df7624ae2bd8f95/docs/RESEARCH_CHARTER_TASK_SPACE_MANIPULATION_V1.md)。
- [现有双服务器/local-pc规范](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3d3cecaa5a5d8b9c6a17158912b6f5816702b678/docs/handover/2026-10-12_LOCAL_PC_ADOPTION.md)。
- [二期维护GO](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47#issuecomment-6129475191)、[进程补充授权](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47#issuecomment-6129639251)、[第一期真实退役收据](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/70b7c9e7d75d4ea71cb4ddc5c5cd8563e63c2fde/reports/maintenance/2026-10-13-corelab-storage/ACTUAL_RECEIPT.json)。

