# SLA_Memory V1.4：先验证真实OCS2后端，后升级Task-Space Reference ABI

> 2026-11-23。状态：PLAN_ONLY / NOT_RUN。基于 V1.2 [PR #428](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/428) 以及 V1.3 [PR #430](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/430) 的封存结果。**保留原H40/G7/V1.3 C0–C3所有负结果**；此文档不修改运行WBC、门限、模型、ROS2图或任何私有二进制；新增GPU前向/新物理请求/积分/训练=0。有关现场Corelab私有文件与运行权限必须由现场执行者重新读取、SHA锁定。

## 1. 核心决策与当前事实

1. 六份Corelab教师卡确认的实际执行器是 `rl_sim_mujoco`、`go2_x5_ocs2_node`、`mobile_manipulator_mpc_node`，即 **OCS2/MPC＋MuJoCo**；卡记录二进制哈希前缀 simulator `39e5136603f5`、controller `9e8ce46837d0`、MPC `4a5f5aa6e0b9`，仅为前缀，运行前逐个审计完整SHA。不能把归档 RL-WBC 的 Gamma、TimeLaw 和9点空间预览写作正式Corelab执行输入。
2. 归档RL-WBC的 `ExternalEEFReferenceBridge` / `TrajectoryBatch` / RL Policy 仍作为**独立第二后端**，可做CPU一致性测试与将来独立物理资格；V1.3九点饱和是归档RL系统的科学发现，不构成OCS2预览的已测证据。
3. 现场H40 `FiftyMsProgram.validate_envelope`＋G7阈值（由V1.3已绑定现场文件）：线速 `0.05 m/s`、角速 `0.25 rad/s`、线加速 `0.25 m/s²`、角加速 `1 rad/s²`、夹爪速 `0.040 m/s`。这些是已冻结发布协议包络，不等于已经实测得到的OCS2动力学上限。Teacher16/16通过，SLA C0 0/16；C1位姿门16/16、完整H40 8/16、加G7 2/16、包含非平凡P10位移及事件0/16；C2/C3缺student-ownership接管，16窗BLOCKED。无学生OCS2/RL真实安装、无闭环分母。
4. SLA 50@30Hz 原生总时长1.6333s。V1.3满足其 source 全路径固定发布动态约束的时标为6.65–18.55s（注册算法族内，而非物理最短时长的全局保证），中位9.375s；用单个1.95s H40发布窗只覆盖部分路径，前10点有效位移不够，gripper事件相对路径位置改变。**这里存在“技能全路径时间”与“滚动MPC发布窗口”不应强行相等的问题**。
5. 已核对用户附带的 `wbc_mujoco_minimal.tar.gz`（其版本只作为归档源代码，而非已证明和现场二进制相同）：`src/planning/go2_x5_planning/go2_x5_planning/ee_trajectory_node.py` 公开了 `EETrajectoryManager`：`/go2_x5/ee/trajectory_json` 输入，`/mobile_manipulator_mpc_target` 输出；默认 `publish_rate=10Hz`, `window_horizon=2.0s`, `sample_dt=.05s`, `input_dim=12`，发布 `MpcTargetTrajectories` 包含绝对MPC时刻、7D `[xyz,qxyzw]` EEF或配置底座目标后14D `[EEF,base]`、12D全0目标输入。 `window_times` 默认2秒包含两个端点约41时间点**还会添加路径拐点**，不是必须40点。
6. 该归档的 `on_trajectory_json()` 每次消息都调用 `_arm_trajectory()`，将 `bound_start=None`；下次发布会根据新的测量EEF/当前MPC时间重新bind整条相对时间路径。这对于一次性离线路径合理，却可能使每0.5s P10替换时重置路径的时间原点/绑定。**需要独立连续替换实验，不得未经观察说这是运行时bug**。若mode从HOME/HOLD转到OCS2，`go2_x5_ocs2_node.cpp`本身也存在独立的MPC reset与实测EEF hold接管逻辑。现场的有效`mode` (OCS2仅臂/WBC包括base动作)也须锁定。

## 2. 架构：保留EE任务语义，增加后端相关的时序/控制权契约

新的高层**不学习WBC关节拓扑**，只输出 `TaskSpaceIntentV2` 语义；不是一次性把SLA输出重新命名H40。

```text
SLA_Memory(frozen; 50×16双臂,源30Hz) / other VLA
  → official inverse processor → 选右臂8D arm-base absolute
  → 实际 T_world_armbase * T_armbase_EEF * T_EEF_TCP
  → TaskSpaceIntentV2 (T_world_TCP(t_src), source timestamps,
                        gripper events, observation clock,
                        task/episode id, frame/tool/calibration SHA)
  → ReferenceManagerV2 (preserved path geometry;
                        explicit feasible time law;
                        controller-owning source, gripper ledger;
                        persistent trajectory id + continuity/TTL)
       ├── OCS2Adapter: long bound reference → MPC-time rolling 2s window
       │               → MpcTargetTrajectories → MPC solver → arm (and
       │               if enabled/matched, base) commands → MuJoCo
       └── [separate candidate] ArchivedRLAdapter:
                       Gamma/TimeLaw → actual trajectory progress
                       → native9-point preview → RL arm/leg policy
```

**两个背靠背实验必须独立：**`LEGACY_H40_G7` 保存原协议和0/16；`OCS2_LONG_REFERENCE_V2` 是新候选接口，只有通过后端自身真实性、约束、状态机与闭环资格后，才可与原协议做结果比较，不能用“绕过旧H40检查”充当旧0/16被修复。OCS2长时域计划依然必须有显式速度/加速度/工作空间/接触安全限制，不能因为新发布通道不再固定40点就直接放开原硬约束。任何物理安全限额变更须独立许可。

`TaskSpaceIntentV2` 建议包含：source_model_revision, source_frame, observation_timestamp, source_fps, 6D pose samples with typed timestamps, target_tcp_semantic_id, rigid transform SHA, grip events anchored both to source path progress and task/contact conditions, uncertainty if available, canonical `trajectory_id` + `revision` + `install_at`, required contact/hold semantics, earliest/latest event windows.不得从Teacher真值填充Student端动作或用GT决定夹爪。

ReferenceManager负责跨请求持有：current controller mode+owner, last issued/ACK/consumed gripper command, measured width/contact flag, current installed `trajectory_id`, current materialized MPC times, physical clock and command age; 前一合法参考保持直到显式原生替换/拒绝安全处理。不使用测量开度冒充last consumed命令。把`FEASIBILITY_CHECK`、`ADMITTED`、`INSTALLED`、`CONSUMED`、`EXECUTED`、`TASK_SUCCESS`作为独立状态和不同分母。

## 3. 阶段 B0：绑定后端真值并核查现有材料（只读/CPU）

- 先回读PR#430报告：`reports/SLA_MEMORY_V13/{OFFLINE_ACTUAL_20261123,WBC_REFERENCE_CONTRACT_ACTUAL}.md`，代码/78包及回执SHA。核验Corelab本地V1.3全量原始轨迹、H40/G7首次拒绝、二进制完整SHA和运行的真实源文件对应。验证 `go2_x5_ocs2_node` 的有效OCS2/WBC模式，MPC优化更新周期、目标topic、ROS QoS/latch、MPC observation clock、夹爪控制单独通道、是否有status/consume ACK及是否会丢弃旧target。
- 归档`wbc_mujoco_minimal`若与现场可执行版本SHA不同，只作为参考，不作为已验证执行栈。禁止推断归档RL Pipeline在Corelab当前就能运行。
- 对源域均匀伸缩角速度辅助数值不一致建立单独回归issue；在修正且重新过原有78包回放之前，不允许把该源域诊断用作证明编译器普适通过（目标域16窗口记录仍有效）。
- 产物：`B0_BACKEND_IDENTITY_AND_SOURCE_ACTUAL`，带运行后端SHA、topic关系、实际owner/lease，区分GITHUB_VISIBLE、PRIVATE_LOCAL_VERIFIED、NOT_TESTED。

## 4. 阶段 B1：仅CPU核验真实 OCS2 reference ABI（不触碰物理）

使用真实归档`EETrajectoryManager`及已绑定现场相同实现，构造两条小轨迹/空路径hold，验证：
1. 绝对MPC时钟单调、每次发布窗口以当前MPC observation时间滚动；核查2秒/50ms含两端点默认41点和曲率拐点附加采样；在ROS下比对`MpcTargetTrajectories.time_trajectory/state_trajectory/input_trajectory`长度及所有字段。
2. EEF `[x,y,z,qx,qy,qz,qw]`，如真实启用底座目标则为14D，`input_trajectory` 12D全零只是目标消息并不代表物理执行器只需12关节；报告机械臂/底座实际控制模式。验证`world/world_offset/ee_start`的source-to-world只做一次；clock reset、未使能、HOME/HOLD→MPC过渡时的安全行为。
3. 相同全球几何路径发送一次，与每0.5秒重复`trajectory_json`新版本，回读绑定时间、目标轨迹、起点速度、最远预览的差异；尤其检查`on_trajectory_json→_arm_trajectory→bound_start=None`。如果状态漂移或无合法延续语义，需求是**显式reference replace/continue**（从当前MPC时间下的未来剩余部分发布，不复制古老路径起点），而不是改WBC网络或硬置时间为零。
4. 夹爪不在`MpcTargetTrajectories`中；端到端更新必须有独立gripper channel的 owner/ACK/command state、物理接触证据与与路径进度关联的事件约束。原16 TRAIN窗口全部teacher-owned，不能凭影子回放创造student takeover ACK。
5. 旧H40发行器与新 long source 共存但不同科学ID，不能把旧H40的位姿加速度门当成原MPC的source-native要求，也不能据此删除OCS2的任何实际机械臂速度/力矩限制。

输出：`B1_OCS2_ABI_WINDOW_AND_REPLACE`。若与现场消息不一致，`BACKEND_ABI_BLOCKED`。

## 5. 阶段 B2：不涉及SLA的OCS2真实跟踪能力标定（新物理资格，非已授权）

**核心测试**：固定本体/控制器/场景/边界，给OCS2连续、完全已知的6D EEF参考；不把SLA输出放进去。预注册路径族和控制比较：
- F0：静止/启动/结束保持；
- F1：短距离 XYZ 平移（前、侧、升降选固定合法方向）；
- F2：纯姿态旋转（TCP平移接近零），SO(3)小角度与较大合法角度；
- F3：平移+旋转耦合曲线（不同曲率/速度），预先做碰撞与关节可行性审查；
每族至少有 `ONE_SHOT_BOUND`（整条长路径仅绑定一次并持续2s滚动发布）与 `P10_REPLACE`（每0.5s重复发布相同剩余路径/有计划的新路径）两种参考模式；两个预注册速度档，用于区分对现H40 envelope可行与原SLA时域相近的指令。先对四类做极小接口smoke，再按资格对照扩展。针对移动底座另有`BASE_ENABLED`成对小面板，但只有被测真实mode支持时才另注册，不能把OCS2 Arm-only冒充全身跟踪。

候选**首批不超过2个无接触/安全停机技术卡＋4类×2模式×2速度=16科学卡，总18个新物理请求**；仅建议上限而非批准；tick上限与deadline须按现场真实卡建立。依次执行、前一阶段实证通过才继续，0新训练更新。每一张卡有独立initial conditions/seed/trajectory hash、command issue/ROS receipt/MPC consumption、2ms ticks/STOP后积分、actual TCP+orientation+base movement+joint states+contact、observed limits、wbc/solver status。不得因错误参考而打断正在运行的其他Core活动。

指标：EOF translation/orientation RMS、p95与max、时间滞后、路径完成比、terminal hold误差、接触和STOP、可行与不可行事件、`ONE_SHOT` vs `P10`产生的目标与实际误差跳变、solver infeasibility；命令执行与任务结果两套分母单列。明确这项才足以说当前0.05/0.25/0.25/1.0等门是否保守（范围仅限该后端/状态，不能反推RL-WBC）。

## 6. 阶段 B3：受控升级接口的两个分支（只在B2事实基础上）

`B3-A H40_FROZEN_BASELINE`：原V1.3原样 replay/同一安全门；结果保持C0学生完整门0/16；C1位姿动态16/16、完整H40 8/16、H40＋G7 2/16、联合有效进度/事件0/16。后续若有新物理与新模型状态需单列，不能改写V1.3分母。

`B3-B OCS2_LONG_REFERENCE_V2`：将冻结SLA经过官方inverse processor及右臂arm-base→world/TCP转换后的**完整源技能路径**保存为长时域Reference：

- 时间：不要求在1.95s H40内走完整段；根据B2测得的真实OCS2能力边界和起始真实速度，为整条路径生成可行 `t_{mathrm{exec}}`，在MPC内2s的前瞻窗口滚动发布；不要每个发布周期重新从`t=0`开始。
- 几何：优先保留source path的SE(3)形状；若需要平滑或改变姿态接触点，单独标`GEOMETRY_CHANGED`，不能再称纯时间适配。先完成角速度数值一致性与时间重参数化回归。
- 夹爪：事件以source路径几何进度、实际接触条件双条件关联，`CLOSE/OPEN`按实际可行时刻和真实状态触发；任何接触时自动张开/对错位事件强行执行都BLOCK。只测试student-owned reset或有真实owner移交收据的阶段。
- ReferenceContinuity：采用`trajectory_id/revision/absolute MPC-time`，明确替换是否延续已消费进度；保留旧reference until new target accepted，正确处理timeout/STALE/ABORT/HOLD，不重置实测历史。
- 参考包：OCS2 `MpcTargetTrajectories` 的所有时戳、state/input维度、源/目标frame和实际采纳回执；必须先CPU shadow比对，再原native无接触测试。对长轨迹的新增安全准入基于后端实测能力＋守恒的机器人硬约束，不能只删旧H40阈值；**任何新门／模式的参数与批准都作为新科学协议发布**。

目标是合法的`REFERENCE_ACCEPTED`、`NONTRIVIAL_P10_PROGRESS`、`GRIPPER_EVENT_SAFETY`、`WBC_TRACKED`、`TASK_SUCCESS`分层报告。不允许用P10前10点仅0.2mm的前进/静止路径伪装成功。

## 7. 阶段 B4：仅在真实OCS2能力与新接口均通过后测试冻结SLA

匹配同任务/初态的三列：可信外部Task EEF+原OCS2、Frozen SLA+C0 legacy H40（保留拒绝为失败）、Frozen SLA+V2长参考+原OCS2；若参考接入不合法，不使用teacher代动作。首轮先单臂源域技能相近的抓取/放入容器子任务，其次StackCube OOD；先站立、不强制底座移动；场景/seed/命令权限/安全检查相同。需要独立新的native资格、预算和场景分母，不能直接把本次16个TRAIN研发窗刷成正式held-out结论。
- 记录raw reference / feasible time law / controller target / solver output / achieved TCP / actual grip & object pose / first failure.
- 统计grasp、lift≥20mm、hold≥1s、place/release、task completion、wall time和GPU/CPU成本；另列实时推理median/p95，历史0.293s median不保证紧凑0.5s连续发布周期。
- 失败归因分别为 SOURCE_OBS_OOD、REFERENCE_TIME_OR_GEOMETRY、GRIPPER_HANDOVER、OCS2_COMMAND_ABI、CONTROLLER_TRACKING、TASK_SKILL；不能把所有0%归因为模型能力。
- 最终仍须与独立RL-WBC资格分开，先取得一个真实后端上的可信闭环结果。

## 8. 后端复用与长期科研定义

永久保留`TaskSpaceIntentV2`（EEF＋事件＋时间＋控制权语义）作为共享高层，不编码任何特定WBC的腿部关节参数。

接口层拆成`CommonTaskIntent`、`ReferenceScheduleAndEventManager`、`Ocs2MpcBackend`、未来单独的`ArchivedRlWbcBackend`。后端均独立建立协议一致性和WBC可行域，不把OCS2的MPC消息或时钟直接塞进RL-WBC Gamma/TimeLaw。

最值得证明的不是降低H40包络后的“PASS”，而是：**同一个冻结SLA的单臂任务空间意图，在有明确执行时钟、物理可行时间律、可靠重规划与夹爪接管的情况下，能否被目标本体的一个真实控制后端跟踪完成实际技能。**

## 9. 自主状态机、工件与STOP

状态机：`B0_BACKEND_SEAL → B1_ABI_CONFORMANCE → B2_KNOWN_TRAJECTORY_CAPABILITY → B3_V2_REFERENCE_ABI → B4_FROZEN_SLA_PAIRED → FINAL`。B0/B1仅离线/CPU可以自主执行；B2/B4的GPU/ROS/OCS2/MuJoCo物理积分要预先存在独立owner/lease和预算，否则结束在`WAIT_FOR_AUTHORIZED_NATIVE`。B3的代码与模拟参考可以新隔离PR实现，但实际安装前亦需授权。禁止自动抢占GPU/Core世界、不执行硬件、训练或Agent，不写现役WBC二进制，不复制私有大数据进public GitHub。每阶段`ACTUAL.md/json`：绑定source SHA、原16窗分母、实际请求/物理ticks/STOP回执、完整高低层轨迹、时钟/owner、阶段PASS/BLOCKED，未执行必须写NOT_RUN。

异常停止：MPC模式/时戳/状态量不明，无法校验7D或14D EEF/base语义，source/source_to_world/TCP重复变换，target message消费状态不明，夹爪命令owner无ACK，非法碰撞，STOP后积分非0，实际提交次数与注册不符，控制器超限，场景并行资源冲突，未注册的阈值/路径形状改变。结果统计把安全拒绝、工具失败、物理失败、资源阻塞分列，不删掉失败条件。
