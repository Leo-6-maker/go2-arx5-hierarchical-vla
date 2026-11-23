# SLA_Memory → Go2 冻结 WBC：V1.3 时间包络与夹爪交接归因/优化自治计划

> 2026-11-23 · **PLAN_ONLY / NOT_EXECUTED**。本文件是 V1.2 冻结迁移的**独立后继干预方案**，不修改/覆盖 [PR #428](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/428) 的 V1.2 0/16 原始负结果。当前仅允许准备可审计的、在 Corelab 受控目录内运行的离线分析；不能据此启动新 GPU forward、新 MuJoCo 积分、训练、Agent 或真机。对正式物理测试的预算、owner 和执行栈必须另行注册并核验。

## 0. 新证据与声明边界

用户提供的 Corelab V1.2 封存结论（此 GitHub 计划作者**无法直接读取**主机 /home/liuyu/ISSAC-GR00T/work/sla-memory-frozen-transfer-v1/v12 下的完整私有报告，故本行标记 USER_REPORTED，现场执行前必须逐文件 SHA256 与实际数组回读）：
- 16 个预登记 Go2 TRAIN 窗口，Teacher→同一 H40/G7：16/16 PASS；Frozen SLA/原动作编译：0/16 PASS。P3 未运行，无学生原生闭环成功率分母。
- 16/16 学生首个 50ms 区间超过原 H40 加速度准入；仅源预测覆盖内的内部点也16/16出现线/角加速度超限；不能把全部拒绝解释成尾端 7/40 HOLD。
- G7 又独立拒绝14/16，其中12/16的上一真实消费命令→新块首点超过原 40mm/s。极端记录：上一已消费命令2.00mm，实测夹爪19.08mm，模型首点19.19mm，命令差17.19mm/0.05s=343.8mm/s；与实测的瞬间差只有0.11mm，**不可直接用 measured 替换 last-command，必须分清执行器状态和命令授权**。
- 已存源域 ARX5 专家6/6、O0冻结SLA预测6/6 按**Go2目标时钟和TCP几何**也超原H40内部包络；该结果强烈支持先研究动作源时间/目标执行包络不匹配，而不能说所有SLA动作都是错误。
- P0 source 8/8有效、O1/P1共18、V1.2额外30/30有效；Go2 16窗有完整161/161真实20Hz历史，P2仅离线，GPU0已释放；权重/原processor与stats不动。
- 原 PR #428 GitHub评论已确认 V1.2 O2输入有效，首点与实测TCP相距0.60–1.07mm，且 LIFT 两窗 t=0.45s SLA相对TCP竖直位移+0.33、−0.04mm，同窗Teacher为+1.41、+3.18mm。这提示**即使接入成功，OOD技能的抬升幅度/事件语义还需独立评价**，不得把全部任务问题等同为速度门。

核心区分：
(A) 原始策略及场景语义 (B) 源域30Hz→目标50ms时序投影 (C) 首点边界速度/加速度与内部曲率 (D) gripper command vs measured 状态分离 (E) 冻结WBC真实跟踪能力。现有 V1.2 0/16是有效**原编译器准入失败**；任何后续通过属于新实验 `V13_COMPILER_ADAPTED_FROZEN_POLICY`，不是回头重命名旧冻结结果。

## 1. 禁止变化与可自主范围

- 固定：SLA checkpoint HF revision `7f6880569c3be777291e646fad1b4951245e8553`、官方PR #237源commit `c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454`、选定右臂、O2观测与左臂/左腕代理配置、全部原始16条预测、Teacher 16/16；冻结 WBC本体/动力学2ms、原H40 40×50ms、P10、SafetyGate/G7/40mm/s/接触STOP/TTL、评分以及所有旧封存数据与口径。
- 本轮 **optimizer=0、GR00T/SLA re-training=0、Agent=0、real robot=0**；不读DEV/TEST；旧16 TRAIN诊断集已用作研发，任何新科学性能声明需独立留出scene/episode。
- 允许：在隔离分支与私有本地 scratch 增加独立、版本化的非学习式编译器/CPU诊断与回放记录；保存 `raw→source-time→compiled H40→gated→(如获批)installed→achieved` 的哈希、几何变化量、时钟变化量、事件变动情况。
- 禁止：降低G7/H40阈值、把 last consumed command 私自换成 measured 值、使用 teacher 标签替 student 纠正动作、异常时静默clip/smooth/filter、用保持原地的0运动伪装能力、模型/处理器参数变化、暗自请求原生世界/GPU、以既有源域模型RMSE替代任务成功。

## 2. S0：现场封存审计与输入恢复（CPU，只读）

将 Corelab 本地 `v12/FINAL_V12_DECISION.{md,json}`、`v12/A4_P2_ABI.md`、`v12/A4_SOURCE_ENVELOPE_EXPLORATORY.json`、对应raw源域6/6专家+6/6预测、目标16/16教师+学生、P10 last-consumed、actual gripper、SafetyGate/G7原issue records逐字节验证并建立sha manifest。若缺可逆物理张量或缺previous command timestamps，停止在 `SOURCE_EVIDENCE_BLOCKED`；不能从散列或摘要重构轨迹。

查清活动实际 H40 加速度/角加速度门源代码/阈值/单位/差分公式：历史 `TrajectorySafetyGateV1` 只检查部分速度、姿态、时效，不等于现场完整H40 acceleration guard；必须绑定`native scope`/binary/flags。定位时间量：O2实际观测→推理结束→issue→安装、源50@30Hz→H40@50ms、先验实际已消费速度与夹爪命令、P10生效区间；仅用real action覆盖原生1.633s，不把最后7/40 HOLD当原生策略预测。

输出：`S0_SOURCE_AND_FAILURE_CLASSIFICATION_ACTUAL.json/md`，每窗口的所有observed/threshold比例（中位/P95/max）、first-hard-fail、重叠reject、多重失败分布、不可归因的缺项、真实 SHA。

## 3. S1：分离起点与内部加速度／时间映射（CPU，0模型调用）

同一16源 Go2学生原始物理动作和6+6源域对照（不能对源域重新训练）做四项互斥分析：
1. 原 source 30fps下真实 `v_i=(p[i+1]-p[i])/(1/30)`、`a_i=(v_i-v[i-1])/(1/30)`；姿态用SO(3) Log的旋转角速度/角加速度，比较各窗口步长分布。
2. 显式原 t=0..49/30→目标t=0..39*.05的采样后逐H40差分，分别比较 `raw_source`、`source_resampled` 和 `source_suffix_7`；标记内部有效33点与末7点。
3. 首段 `v_prev,omega_prev` 必须来自真实上一条已消费 WBC参考（另列 achieved 速度作为状态变量），检查初始位姿虽近但 `(v_0-v_prev)/.05` 很大；把 first-step/inside/suffix 贡献分列，禁止只报告一个最大值。
4. 与 matched Teacher 16 PASS同条件对照：不同阶段的路径长度、转角/角度路径、速度和加速度，判断差异来自源域本来节奏较快、模型局部抖动，还是会被重采样放大的几何折角。

一份可审计表格：16行 × {raw_source,interior_raw,first_boundary,suffix,teacher}，给出线/角速度和加速度的 exceed count、max ratio、phase与source episode聚类。不能因target 16/16超限直接断言某一加速度阈值不合理；Teacher16/16能过，说明存在其他可行参考。

## 4. S2：夹爪交接控制权审计（CPU，0积分）

每一窗口读取 `last_delivered/last_acknowledged/last_consumed command`、`actual measured width`、`SLA first command`、设备/模拟执行器实际当前模式、是否接触或持物、控制owner、时间戳/TTL。严禁把 measured width 改写成 previous command 以绕过G7；独立计算：
`seam_speed_cmd=abs(first_sla-last_consumed)/.05` 和 `tracking_gap=abs(actual-last_consumed)`。
例：从2.00mm命令到19.19mm候选至少需要17.19/40≈0.42975秒的**显式**、受限过渡，才能在40mm/s以内；但若正在抓持或有接触，直接执行该过渡可能导致掉物，默认 `HANDOVER_BLOCKED`，不能自动升为`SAFE_TRANSITION`。

预先登记两个科学分支：
- `G0_RESET_OWNED`：完整episode由冻结学生开始控制，reset时由统一控制owner明确建立 gripper初始命令、实际width/反馈、ACK和P10 history，再执行第一块；禁止在中途teacher接管后仍叫无缝学生执行。
- `G1_TEACHER_TO_STUDENT`：只有具备真正owner移交、持续控制历史、可确认非接触/可安全交接、命令状态一致时才允许。若要在空载状态执行显式限速交接，应把过渡作为单独、预算化动作，计入延迟与科学分母；接触/抓持状态不允许自动张开寻找PASS。没有证明就冻结为 BLOCKED，记录夹爪开度匹配实际而命令不匹配的真实结论。

对原16个窗口仅做 shadow review，不自动更改已封存 V1.2 gate结果。新 gripper scheduler 不是 `last_command=actual` 的换变量伎俩，应保留权威命令账本、控制器ACK、真实状态和可否安全改变command的判据。

## 5. S3：分离的非学习式编译器对照（CPU）

保留一份不可修改 `C0_RAW_V12`。针对相同原student16与source12，实施以下固定对照，不可中途新增任意手调版本：

| arm | EEF参考时序 | 夹爪接缝 | 目的 |
|---|---|---|---|
| C0 RAW | V1.2完全不变 | V1.2完全不变 | 复现0/16拒绝 |
| C1 TIME_ONLY | 保留原SE(3)几何路径，显式速度/加速度限制重新参数化 | V1.2不变 | 验证H40动态修复，G7可能继续失败 |
| C2 GRIP_ONLY | V1.2 EEF原样 | 原命令链可验证的安全交接，无法则BLOCKED | 识别G7独立贡献 |
| C3 JOINT | C1相同 | C2相同 | 验证两种独立干预能否同时使全部门合格 |

时间编译器先独立使用非学习式 `s(t)` 轨迹参数化：`p(t)=p(s(t))`、`R(t)=R(s(t))`（SLERP/SE3适当插值）并对**实际冻结的**线/角速度、加速度上限求可行时间；以真实上一已消费参考速度、实际 TCP位姿作为初值限制，不允许突然复位为零。比较全局均匀时标放大(`k>=1`)与有界的边界速度对齐时标，两者固定子实验名；先算可行总耗时和P10执行时长，如果在H40 1.95s末点约束/原持续预览下不可行，明确 `RETIME_INFEASIBLE`，不能将所有点保持原地、再声称已完成动作。实际WBC Gamma/TimeLaw 有路径进度语义，必须在原native canonical化后重新检查s_ref和sdot_ref，而非只给出数组合格。

首选**几何路径不变，仅改进度/时序**；如因源路径折角/姿态尖峰导致在有限时间内无法有可微界的速度/加速度，单独定义`C4_GEOMETRY_CHANGED`研究问题，声明曲线平滑/路径偏差/抓取接触点改变，禁止把C4算作同一路径时间适配。TOPPRA可作为光滑路径约束重参数化方法参考，Ruckig偏state-to-state且中途路径未必不变，不可不审查即插入安全栈。

每个candidate输出 raw vs compiled、空间路径 Hausdorff/最大姿态距离、路径总长、进度/终点比例、抓爪闭合/释放事件相对路径进度、最小可行时间、所需H40续段数/策略更新次数、P10前10点实际进度、位置/角度/夹爪实际速度与加速度、first vs interior vs suffix拒绝及完整 issues。**强制nontrivial progress**：不能把“40个点全保持”计作可迁移成功；预设任务相关的最小独立几何进度/接触事件保留门，并用已有Teacher与物体几何设限，封存在观察候选结果前。

C1/C3即使全通过G7/H40，也只称为 `ADMITTED_COMPILER_REFERENCES`，还不是WBC能够执行或SLA任务成功，且不重写V1.2的原始0/16结果。

## 6. S4：停止准则、独立验证与封存

固定S0/S1/S2为只读，S3最多4条预先登记主方法×16原目标TRAIN窗和6+6源域对照（已有预言动作，不新增GPU forward）；不得根据这16个窗口的PASS率迭代无限候选。提供全量逐窗和stage/seed统计，C0最初0/16需可重现，Teacher16/16不能改变。若模型原始路径本身几何不可行、换成C3也进展为零，报告 `HIGH_LEVEL_PATH_OR_TIMELAW_BLOCKED`；如夹爪接管在接触期间不可合法修复，报告 `GRIPPER_HANDOVER_BLOCKED`；如有效且全门合格，则编译器版本+参数+原观测窗口群立即封存。

建议另立若干独立TRAIN OOD holdout新场景（不能DEV/TEST污染，也不应把已用于C3开发的16窗称为泛化），先仅离线一次性应用封存编译器，记录full denominator。若新场景无source task match，失败单列`SKILL_TASK_OOD`。真实 model forward 若需新增，必须另注册GPU0 owner/lease和次数，不能把“原数组已够分析”当成推理授权。

任何涉及改变原H40/G7门阈值、WBC、具体底盘或新训练的方案都不在此V1.3范围。所有算法失败为负结果，不允许默认使用教师补动作或特权对象位置修复学生方向。

## 7. S5：有条件的原生MuJoCo配对（**未授权，禁止自动开始**）

必须已有S0–S4全部正式收据、C3不作假通过原H40/G7、非平凡EEF进度/事件保留、现役物理/设备/模型owner/sha/lease核实及另行批准固定科学预算。首先进行1–2条无接触 native interface/hold技术请求（STOP后积分=0），然后独立 pre-register 教师与 `frozen SLA + compiled trajectory` 的同场景配对 ≤4对；整个pilot上限 **10个新native请求**，tick上限必须由现场原卡重新计算（旧PR#428曾估计367030 ticks不是权限）。在源相近单臂局部技能先测，StackCube与源 Pack-and-Pour-Fruit差异单独标OOD，不能断言源域能力失效。

追踪 raw/model → compiled → original-gate → installed/ACK/P10 → achieved EEF → object contact/grasp/lift≥20mm/hold≥1s/place/release；真实底盘移动另列，不计为本轮任务。记录policy median/p95 inference latency，源平均0.293s可能触及旧0.20s fresh-age门，只有连续实时有效期内完成才能算真实闭环；不允许靠暂停物理掩盖延迟后宣称continuous real-time。

停止：任何original gate REJECT、未知/重复请求、资源争用、非法接触、非0 post-STOP tick、超过预算/期限，关闭并保留证据。Gate通过但任务失败按完整科学分母继续报告，不能模型补偿修复再算同一策略。

## 8. 自主执行边界/交付

状态机：`S0_SOURCE_AUDIT → S1_TIME_DIAG → S2_GRIPPER_DIAG → S3_FIXED_COMPILER_ABLATION → S4_FREEZE_DECISION → (only if separately authorized) S5_NATIVE_PAIRED → FINAL`。每阶段产物 `ACTUAL.md/json`，含commit/model/data SHA、全部16窗分母、门阈值实际值与observed、CPU/GPU时长、零训练更新/是否发生原生积分、运行主体和exit codes。所有原始轨迹、相机、私有WBC二进制/数据只存Corelab受控路径，不上传公共 GitHub，仅发布脱敏结果与验证哈希。

需使用独立执行PR进行的实现单元（本计划不直接创建）：`frozen_transfer_a4_audit`、`source_vs_target_kinematics`、`gripper_handover_ledger`、`se3_time_parameterizer`、`h40_original_gate_replay`、`pilot_native_pair`。先建立固定的CPU fixtures与对原16窗口的只读重算，附负例测试：非零v_prev、20Hz/30Hz错时钟、夹爪cmd≠actual、非法接触强开爪、末7hold掩盖source peak、错用source midpoint、statusUNKNOWN重复跑、靠静止编译通过G7等。

**最终评价**：这一步研究的是*在不训练模型/不改变执行器安全门下，能否通过显式、可追溯的命令状态与时间参数化，使已有SLA动作合法地进入原WBC*；不能等同于证明SLA最终能抓取或完成放置，更不能拿预注册16训练窗通过率称为跨任务泛化。
