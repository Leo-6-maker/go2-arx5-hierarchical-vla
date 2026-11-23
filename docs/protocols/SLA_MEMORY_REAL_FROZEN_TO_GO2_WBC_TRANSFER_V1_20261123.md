# SLA_Memory 真机后训练权重 → Go2＋ARX5 冻结 WBC 迁移协议 V1

> 日期：2026-11-23（Asia/Singapore）。状态：**PLAN_ONLY / NOT_EXECUTED**。本提交只发布研究设计；不启动服务器作业、模型推理、仿真积分或新训练；不自动授权新资源，不更改任何现役 capsule。

## 0. 决策与研究命题

独立实验名：`SLA-MEMORY-REAL-FROZEN-WBC-TRANSFER-V1`。

核心问题：官方在 RoboDojo-Real 的 ARX X5/Piper/Piper X 上联合后训练的 **冻结 SLA_Memory**，能否只通过明确、非学习式的观测/动作接口，迁移局部操作技能到 **Go2＋单 ARX5＋原 WBC**？先研究静止操作，之后另立计划研究底座移动；暂不引入 Agent、继续训练或 OpenWAM。任何成功都只能在明示输入与任务条件下解释，不能直接称为跨本体零样本通用性。

必须区分：A. 官方源域同本体效果；B. 将三相机/双臂观测降为目标双相机/单臂的影响；C. 原生时序/坐标/夹爪到目标 H40/P10 的映射；D. SLA 参考任务正确性；E. 冻结 WBC 的物理执行能力。这是**已后训练权重的跨执行系统迁移**，不是声称未经任务训练的 SLA_base zero-shot。

## 1. 固定的证据起点与身份

### 1.1 上游

- XPolicyLab SLA_Memory：PR https://github.com/XPolicyLab/XPolicyLab/pull/237；审计其 head commit `c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454`（尚未合并；从该提交/源仓库及具体文件 SHA 读取，不能悄然跟踪 PR 最新）。
- 模型：`haofuly/SLA_benchmark_ckpts` 中 `SLA_Memory_robodojo_real_ckpt`，revision `7f6880569c3be777291e646fad1b4951245e8553`；权重与 norm stats 全文件 SHA256、文件数、完整性回读后才允许模型加载；不可误用 `SLA_base`。
- 代码：`train_config_name=train_robodojo_real`、`extra_frames_first=false`、`attention_impl=xla`（先用与公开权重兼容的配置）。不要采用 PR deploy.yml 默认的 sim config/true。独立记录 JAX、OpenPI、预处理版本和 tokenizer。
- 官方真实 ARX X5：3 路当前 RGB（cam_high、cam_left_wrist、cam_right_wrist）＋主相机起始帧/过去 8 秒每秒一帧的 9 张历史；每条渲染/控制观测都要更新 history，而非只在请求动作块时更新。源 ARX X5 记录频率 30 fps；一条动作的源表示为双臂各 [xyz, qw,qx,qy,qz, gripper]（16D）；模型内部 transform 转换/填充，不能把模型内部张量维数当作对外 EE 指令。
- 权重公开文档说模型权重约 12.4 GB，真实占用/峰值 RAM/VRAM 需在目标 host 实测。来源：PR #237 文件 `policy/SLA_Memory/README.md`、`model.py`、`openpi/src/openpi/policies/robodojo_memory.py`、`download_checkpoints.sh`。

### 1.2 我们项目

- 仓库 `Leo-6-maker/go2-arx5-hierarchical-vla` 的 `main` 只是整合发布入口，所查 main SHA `6843163d641eb24a29c3b1557f868765346ba66c`，**不能直接把 main 当现役完整 Core 原生执行栈**。
- 当前最近现役证据：PR #426 execution head `e3a136e2191d14f652bb54960f2b64a89f2af4ef` 与 PR #427 诊断 head `b6869988b90851baf0e4ce207df3985fef1d2dc7`。在实际执行前必须重新读取现场正式 source seal、scope、外部二进制与服务状态，不能自动采用这里的历史快照。
- 原生 Go2＋ARX5 固定 WBC/腿策略/关节与动力学；原生 MuJoCo 2ms 积分，原 EEF10/H40 时间网格 50ms、P10 控制消费、原 G7/SafetyGate/TTL、原接触 STOP 与评分全部不变。PR #403 V5.1 曾取得3/8尝试的已知布局移动抓持结果；PR #427 的 MAIN3.8K 则3/6抓持、0/6抬升，首失败集中夹爪 chunk seam：40mm/s 原门。结果不可相互合并或当新模型的基线。
- 原始 `ExternalEEFReferenceBridge` 在 `src/go2arx5_vla/target_sim/bridge.py`：明确 source/world/TCP 和观察、issue、install 时间，拒绝非法帧和容量，向原 native Gamma/TimeLaw 安装，不得为了迁移私自重写 WBC。

## 2. 全局不变量与禁止项

1. **Frozen**：SLA checkpoint 参数、processor、stats 和固定采样方案保持；本轮 optimizer_updates=0，new_training_visits=0，agent_calls=0，physical_hardware_requests=0。
2. **Safety**：不更改 WBC、SafetyGate、G7、Gripper 40 mm/s 门、P10、接触 STOP、2ms、机器人与场景物理。不能通过裁剪、低通平滑、按成功结果筛选、偷偷改变动作时序或模拟教师接管把失败变成成功。确定性契约转换必须事先版本化，保存 raw→converted→installed→achieved；失败仍入分母。
3. **Isolation**：只读复用现役源码与正式证据，生成新运行身份和独立输出目录，绝不进入活动的 source checkout/模型目录热改；不抢占 Core GPU、Hopper H200 等被使用资源、不假定调度许可。获取现场实际 GPU UUID/lease、主存/显存/磁盘和服务期限后才可能启动。未经新现场授权，本计划不产生任何原生执行。
4. **Science**：不读取、泄露或调整旧 DEV/CONF/TEST 成绩来挑方案。首轮选旧 TRAIN/独立诊断人口，按 scene/episode 先划分再固定条件；不读取 task/Teacher phase 或仿真真值进入正式学生观测。
5. **No robot hardware**：仅隔离 MuJoCo，明确区别 SIM_PAUSED_NATIVE、连续实时与真机资格；不能自动升级到硬件。

## 3. 阶段 P0：模型与运行时阳性核验（零物理积分）

做：在隔离环境下载并校验 pinned upstream 与 HF revision，建立只读 `MODEL_MANIFEST`（每个 params/stat/config/tokenizer SHA、代码 SHA、JAX/torch 环境、输入/输出维度、动作语义、相机顺序、history、source fps）。验证 released checkpoint 的 `train_robodojo_real` 和 `extra_frames_first=false`，仅 CPU 静态检查后在获配 GPU 上进行一次真正的模型前向。禁止重新拟合统计或从其他任务继承处理器。

阳性输入：优先从官方公开 RoboDojo-Real ARX X5 示例/示范中，预注册若干独立 source episode 的真实三相机与双臂数据，在其原生格式上恢复模型观测（源作用为**核验输入、动作分布及编码**；不能仅靠 teacher-forced loss 声称已闭环成功）。使用相同模型/随机种子重复检查输出有限性、物理量级、动作随输入变化和 history reset，无目标权限泄漏。

过门：完整权重与处理链身份一致、真实 sampler 输出非空有限动作、无双臂顺序/相机通道/历史错位；任何失败先定位 source 控制路径。若原生 source 输入本身无法正确推理，**暂停目标 Go2 测试**，不能直接将 target 0% 归咎于跨本体失败。

交付：`P0_MODEL_AND_NATIVE_SOURCE_REPORT.md/json`，实际模型调用数、失败次数、文件 SHA；NOT_RUN 不写 PASS。

## 4. 阶段 P1：观测退化/单臂映射消融（零物理积分）

原模型要求三路当前 RGB 和两侧8D EEF state，而 Go2 只有 body＋wrist 两路 RGB、单臂 EEF；这不是一般 frame 变换。建立三种明确标识的 observation profile：

- `O0_SOURCE_NATIVE`：公开源数据的原生三视角＋双臂，阳性对照。
- `O1_SOURCE_ABLATION`：在同一源输入中，用预注册的单臂选择、缺失侧相机/臂状态处理模拟目标缺失情况。优先使用模型真正支持的 mask；如果代码只接受三路有效图像而无法屏蔽，则明确作为 **surrogate OOD**（固定不可学习图像/状态占位），绝不称为“真实右腕观测”或声称原生兼容。左右手角色作为预注册对照，不根据成功动态选择。
- `O2_GO2_NATIVE`：真实 Go2 body/wrist RGB 与实际 TCP/gripper，仅最小、明确的必要桥接。相机外参、视角、分辨率、时戳和每份替代输入来源全部记录。不能把 scene GT 图像/物体坐标替换给正式 student。

使用相同 source episode 进行 O0/O1 的 paired prediction 与分布差异统计，先判断缺视角/缺手造成多少退化；目标场景 O2 的离线预测必须作为独立 OOD 结果，不冒称原生三视角零样本。

前置门：没有可重现的缺失视角处理、角色绑定不明或重复错误的历史帧时，不进入 WBC 原生物理运行。允许把失败定为本轮科学结论 `OBSERVATION_CONTRACT_BLOCKED`，将学习型视角/单臂 adapter 留给后续独立方案。

交付：`P1_OBSERVATION_ABLATION.md/json`。

## 5. 阶段 P2：动作/时间契约与离线 SafetyGate（零物理积分）

对每一个实际模型输出，保存输出张量、raw world-frame EEF 与 gripper（选定实际单臂）及对应 source 时戳。逐字段核验：

- 模型原始 16D 双臂动作经过官方 inverse transform 的真实含义；任务是否落在选定左/右臂，未使用侧的动作不能暗中成为目标 WBC 的额外控制。
- ARX X5 位置坐标相对双臂 base midpoint 的原点与 Go2 TCP/world 转换；严格校验 SE(3) 姿态、quaternion 的 qw-first/xyzw、rot6d columns/preview rows，转换 round-trip 精度、观测时刻标定；无 double-transform。
- 模型 gripper 的 0–1 开度与我们实际宽度命令/反馈的工作行程，不能把官方归一化值直接等同毫米宽度；记录上一**实际已消费命令**而不是上一模型提案。
- 源 ARX X5 帧率 30fps 与 target H40 的 50ms 不同。官方模型默认 50-step horizon 是否在该 checkpoint 改写，先从完整配置与实际返回 shape 核验；若确为50@30fps，其原生预测时间跨度短于 H40 的0..1.95s。禁止私自拉伸时间或复制后缀伪装完整模型预测；只能用**预先声明、独立命名的确定性参考编译方案**处理不覆盖时段（例如明示 terminal hold，单独报告后缀占比），并用原 WBC/G7 评价。该部分是接口策略效果，不等于原始模型输出。
- 同一输出计算 H40 内部位置/角速度、加速度、首点与当前 achieved TCP 误差、P10 最后已消费与新chunk的 EEF/夹爪 seam。前一模型候选不能代替前一实发命令。特别核对原夹爪 40mm/s 门、原 H40 加速度门（旧 V5.1 0.25m/s²，仅在真正绑定同版代码时适用），不得变更。
- `ExternalEEFReferenceBridge` 原 canonical Gamma/TimeLaw、gripper schedule/TTL、ACK、拒绝保留安全持有引用、P10 消费。回放专家参考作同输入阳性控制。禁止只验证数值数组、跳过 native canonical 校验。

离线计划初版：预先固定最多16个真实目标 TRAIN 观测窗口，覆盖接近、闭合、抓持后第一请求、抬升、搬运、释放/撤离与终末保持；单独记录全部16窗口的实际生成及拒绝类型，不将同一 episode 的重叠窗口当独立泛化样本。

进入物理的资格仅要求：正确的编码/逆转换、完整时间语义可说明、有限输出、对选中条件存在可按原安全规则合法安装的提案、原专家对照能经同门通过。若16/16都拒绝、或转换本身有歧义，则标明离线科学/接口结果并停止原生 SLA 执行，不放宽安全门。

交付：`P2_TRANSFORM_ROUNDTRIP_AND_H40_REPORT.md/json`；必须有 raw/converted/gated 哈希和每项 observed/threshold。

## 6. 阶段 P3：独立 MuJoCo 小面板（需另行确认资源和正式运行身份）

在 Stage P0–P2 实际过门、现场资源/owner/权限核验后方可运行。**冻结 WBC、原场景与相机、原G7/TTL/P10**，只用新注册 `SLA_MEMORY_FROZEN_TRANSFER_PILOT` 人口，不进入旧开发/资格分母。第一轮仅测试当前机器人原生 MuJoCo，Go2 保持稳定站立、无需强制移动底座；如场景自带控制器低层补偿，记录实际底座位移，不冒称移动操作成功。

建议固定四个有边界的任务/场景组合（仅研究建议，正式注册前依据 source 技能与当前合法场景的共同语义确定）：F1红 HOLD、F1红右 PLACE、F2蓝 HOLD、F2蓝右 PLACE，各预先选一个 source-independent 目标 TRAIN 逻辑场景及确定性 reset。若无法和官方 RoboDojo-Real 后训练任务形成足够相似的物体/指令，应把结果明确标记为未匹配任务 OOD，不以“迁移失败”推断官方已学技能无效。

每个条件做一对：原可信 Teacher EEF→相同 WBC、冻结 SLA→相同 WBC；每一对的场景、机器人初态、控制/评分、相机、真值权限一致，Teacher只能用于阳性执行对照而非 student 接管。加两条不会接触物体的接口原生技术卡，测试真正的安装 ACK/P10/cancel/hold/STOP。探索性上限：**2 interface + 4 matched Teacher + 4 SLA = 10 native requests**；按原先技术卡10,703 tick、HOLD35,703 tick、PLACE50,703 tick 估算总上限 **367,030 tick**（需用现场版本化原始卡精确重算，不可把估算当已经授权预算）。每一请求均有明确唯一ID、原请求额、条件和终止状态；技术中断与模型被原门拒绝都保留，不随意重跑找成功。

任务指标：原 native grasp、物体抬升≥20mm、持续保持≥1秒、PLACE进入目标/释放/撤离与完整原任务条件，分别报告；“底座方向移动≥80mm”仅在移动条件的原综合任务成立时计算，不能把静止 HOLD 重新命名移动成功。

关键记录：每次模型查询/全H40编译/拒绝/真正安装 ACK/P10真实消费、首个安全失败及接触体、实际 EEF tracking与物体位姿、初始reset、tick/墙钟/GPU成本、停止后积分计数（必须0）。成功和失败均按所有尝试分母报告；四对是机制性 pilot，不能声称精确成功率或零样本泛化。正式 DEV/CONF/TEST 完全不启动。

交付：`P3_NATIVE_PAIRED_PILOT.md/json`、逐次证据哈希、完整 native stop receipts。若未实际运行就写 `NOT_RUN`。

## 7. 失败归因矩阵

| 实证现象 | 优先解释/下一步 | 不能声称 |
|---|---|---|
| 官方 O0 不能推理 | 权重/processor/源路径错误或模型本身问题 | Go2跨本体失败 |
| O0 好、O1 明显差 | 缺失相机/双臂信息导致输入 OOD | WBC无泛化 |
| O1 尚可、O2 离线失效 | 新视觉视角、状态/坐标或技能分布 | 后训练肯定必要 |
| raw合理、H40变坏 | 编译、时间跨度、末端尾部或单位问题 | SLA动作必然错误 |
| Teacher+WBC好，SLA+WBC差 | 看原参考 vs achieved，区分高层与低层 | “WBC导致全部失败” |
| SLA+理想化执行器可行，SLA+WBC失败 | WBC可执行域、接续/接触跟踪问题 | 两者物理契约完全一致 |
| SLA在已有局部技能成功、长程组合失败 | 后续可研究 Agent Staging/验证/恢复 | 本轮已有Agent收益 |
| 安全拒绝/非法接触 | 原门如实记负结果与STOP；不重试科学ID | 可以通过放宽门修复 |

理想化 EEF 执行器如未来需要，应使用另一个新明确标识的诊断臂，不计 Go2 WBC 成功。

## 8. 提交门、资源与停止条件

- Gate0：GitHub/HF源码和模型 revision 固定，stats/样本与时间契约正确，原 O0 真正推理通过。Fail→停止。
- Gate1：O1/O2 有可解释的观测映射及效果；如源 O1 丧失技能或必须伪造本体特征，转成观测适配课题，不进入物理。
- Gate2：动作 round-trip、两种时间网格、H40/G7/P10、夹爪 seam 与原桥接阳性控制通过。没有任何有效合法学生提案，不能投机性开始原生资格。
- Gate3：独占 Core world 与 GPU/宿主资源有现场合法 owner/lease、截止、预留主存/显存/磁盘/日志，全部现役活动capsule不受扰动。16GB RTX 卡能否加载 12.4GB 权重＋11视图视觉 tokens **未知**，先真实测量；OOM/争用时只报资源阻止，不抢 Hopper 现役 H200、不改其他用户作业。
- Gate4：原生任何非法接触、命令超限、STOPlate、STOP后积分、未知请求/安装状态均执行原停止语义；不能自动重试。纯科学失败终止相应已注册请求，保留下一独立未启动条件，技术 UNKNOWN 先查流水帐不重复派发。
- Gate5：全部10请求或期限/预算/安全/资源门耗尽后冻结结果、退出拥有者服务、确认PID/cgroup/lease无遗留，另列无法运行的条件。后续更广泛 source-matched任务、更换本体、新 Agent、adapter 学习/后训练、新移动任务都须**新版本科学决策**，不自动衔接。

## 9. 实现建议与验证清单

此协议不要求立即实现。后续执行 PR 仅增加**独立版本化**的：`public_sla_memory_client`（复用官方推理服务），`source_observation_replay`，`single_arm_observation_profile`（显式 OOD），`sla_to_eef_reference`（非学习式，时间/位姿/夹爪）、`frozen_sla_transfer_runner`（复用现役原生 world 与 WBC），小范围单测/回归及记录。禁止把新逻辑接入活动 `scripts/sla/core_hopper_*` 的已运行 capsule。

静态/CPU验证：checkpoint identity 错、缺文件、错误配置 false/true、左右手选择错、重复/缺 camera、历史reset泄漏、时间倒退、逆变换与rot6d单位错、夹爪映射/接缝跳变、H40后缀不足、非法 TCP、过期 TTL、非法接触 STOP 和结果去重等负例应被拒绝。实际 GPU 输出与 native 运行另需真实 source/host evidence；mock通过不算科学合格。

产物统一置于新独立目录，必须区分原始模型、接口编译、安全准入、WBC完成度和任务结果。报告至少包括：
`SOURCE_AND_WEIGHT_MANIFEST`、`P0_SOURCE_CHECK`、`P1_OBSERVATION_ABLATION`、`P2_ACTION_AND_H40`、`P3_NATIVE_PAIRED`、`FINAL_TRANSFER_DECISION`（md/json，附完整分母、场景/版本/权限、哈希、实际运行记录与成本）。任何阶段无实绩写 `NOT_RUN` 或 `BLOCKED`，不编造结果。

**结论判定：**成功至少证明“特定官方已后训练技能在声明的输入/接口改造下可被当前 WBC 执行”；失败首先归因到具体门。不能仅凭四对或同布局重播宣称多任务泛化、任务无关适配、连续实时能力或真实硬件可行。