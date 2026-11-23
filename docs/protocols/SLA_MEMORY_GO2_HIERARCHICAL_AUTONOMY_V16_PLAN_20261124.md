# SLA_Memory → Go2＋ARX5：高层任务空间策略与冻结 OCS2/WBC 长程自主推进 V1.6

**2026-11-24 — SCIENTIFIC PLAN / EXECUTION-GATED / PLAN ONLY。** 精确父端点：[#439](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/439) `d6639cb03af1ff98519b2566cee2c9ee29627ba8`。核心前置执行证据：[#437](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/437) F1/F2 `9d498d366d8ad1cc9506a1b13006c9f6faa9b20d`。上游官方：XPolicyLab `SLA_Memory` PR #237 @ `c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454`；官方后训练实机权重 HF `haofuly/SLA_benchmark_ckpts/SLA_Memory_robodojo_real_ckpt` revision `7f6880569c3be777291e646fad1b4951245e8553`。**本PR不授权新的原生MuJoCo积分、GPU forward、梯度更新、夹爪发包或接触操作。**

## 0. 北极星目标与科学身份

**研究目标**：SLA_Memory 在只使用真实可获取的 Go2 视觉、语言和运动状态时，能否产生**自己决定的**、可解释的任务空间 EEF 意图/局部轨迹，配合持续参考管理和冻结 OCS2/WBC 执行、以真实结果驱动下一个动作，完成接近、抓持、抬升、搬运、释放乃至长程组合指令？若官方实机后训练权重无法胜任，能否用保留官方技能的 Go2 多源后训练提高 **held-out 完整任务成功率**？

**科学区别**：
- 官方 SLA_Memory 当前原生输出是 **50×16 局部双臂 EEF+gripper chunk**，并**无原生离散任务阶段/长期规划输出或独立Go2底座移动指令**。不因模型名为VLA就直接宣称它能长期规划；要为长期能力引入可评测的 `Skill/Task Executive`，必要时经新Go2后训练实现阶段预测。**若安全FSM由人工固定决定下一步，只能将成果归因于“局部SLA+脚本化执行器”，不能记作“SLA自主长程规划”。**
- 冻结实验后端是现场 `rl_sim_mujoco` + `go2_x5_ocs2_node` + `mobile_manipulator_mpc_node`；归档RL-WBC的Gamma/TimeLaw/九点preview不是本计划可执行ABI。**原OCS2/MPC、PD、腿部policy、机器人模型、H40/G7、原始安全STOP不修改**；新的 ReferenceGovernor/外环是单独的科学控制条件。
- 不用仿真object GT、教师动作或完整未来episode作为学生在线动作的“提示”。若为了打通演示采用oracle/teacher，它必须作为独立上界组且明确不得称SLA规划。

## 1. 固定证据与剩余阻塞（此处不重新计算历史结果）

- [#439 O0/O1/O3 实绩](../../reports/SLA_OFFICIAL_GO2_TRANSFER/README.md)：官方实机checkpoint 19/19文件、约12.44GB与归一化stats、源代码21文件核验；来源6主窗与Go2 16主窗已存在真实模型前向。Go2当前时钟上按实际2ms tick每隔1s取头历史帧是**正确的**（不再声称现役历史对齐错误）；左腕黑图和固定虚拟左臂仍为明显OOD；三图像mask均为true。
- 官方输出是经官方inverse、反归一化的双臂绝对EEF 50×16；新适配选择右臂8D，遵守来源臂基座、世界TCP和工具外参。原H40/G7学生科学准入0/16；连续世界参考编译器中16/16几何有效但动态准入0/16，且首P10内SLA自身非平凡进展0/16。
- 连续参考接续耗时0.75–0.80s，压过首0.5s P10；保留50源路径结点与8个抓放事件（均未处于首1.95s预览内）。1,718个41点、10Hz离线目标预览中420个超原加速度门，**16/16窗口均有**；其中起于过渡段103窗、起于SLA运动段317窗（滚动窗口可以跨段），所以只修lead-in不够。源1.633s，旧诊断/本次最长编译执行时长约7.40–19.30s，改变时标不等于动作语义/物体接触已经合格。
- 学生合法夹爪接管仍0/16：其中8个旧窗口已持物却在学生首命令要求张开；真正的此前消费命令而不是实测手指宽度才是G7接缝基准；已封存55项CPU测试不构成现场student-owned ACK。
- [#437 F2 静止反馈](../../reports/SLA_FROZEN_FEEDBACK/F2_ACTUAL.md)：A1在4/4个匹配TRAIN01、无接触已知终点达到4mm/2°且连续1s，需9.8–10.0s；每100ms平移修正最大0.40mm，最终参考相对任务目标偏移约34mm；**全程位置RMS仍约16.84–17.13mm，未通过移动抓持/接触资格**。不能将单点慢速停稳当作能在SLA 1.633s/0.5s P10时序内正确追踪。
- 原B2/D1/D3、F1/F2/O1所有分母保持不变；新研究不使用旧16窗作为held-out任务证明。F1/F2 18张新原生请求172,454tick已封存；当前原生/gripper任务真实学生尝试=0，成功率=unknown，不应写“0%”。

## 2. 目标系统与接口职责（唯一端到端设计）

```text
Language task + GO2 camera/body/right-wrist RGB + timestamped actual EEF/hand/base
      ↓ (20Hz observations; exact 1s history from source timestamps)
Frozen/Posttrained SLA_Memory (source50×16; native action around 30fps)
      ↓ official inverse → select right 8D, source arm-base to world TCP
Task-Space Intent Registry [task_id, episode, revision, source_N×SE3,
    requested_gripper_events, observation_clock, target provenance, phase proposal]
      ↓
Task Executive: SLA-derived next subgoal/phase candidate + audited progress predicates;
    safety-only supervisor can veto/hold, not silently inject teacher target
      ↓
Continuous Reference Manager: committed-prefix lock, source→execution progress,
    C2 SE3 motion compiler/geometry qualification, durable trajectory clock
      ↓ (optional separately qualified bounded external TCP feedback; must not override SLA goal)
OCS2 native publish 10Hz, 41×50ms, 2s world EEF+Base, source_mpc_time,
    sequence, epoch, TTL0.4s → receiver → MpcTargetTrajectories (state 14D)
      ↓
Frozen OCS2 MPC/MRT + arm PD + base RL leg policy → MuJoCo2ms
      ↓
Typed receipts: issued/accepted/installed/solver_echo/low_level_applied/achieved
    plus gripper owner/issued/ACK/consumed/contact; feedback to policy/manager
```

**Explicit APIs**:
- `ObservationBundle`: `episode_id,mpc_timestamp,head_rgb,right_wrist_rgb,head_history[(timestamp,pixel_hash)],virtual_left_provenance,right_arm_measured_pose,gripper_actual,base_pose,contact,object_visual_estimates,robot_mode`. Image+eef state from **real observation**; object truth from simulator may be used to score, never to guide student unless explicitly oracle condition.
- `SLAOutput`: checkpoint identity, inference observation hash, `source_actions[50,16]` native official-postinverse, seed, source clock, wall/gpu inference timestamp. **No invented model confidence**.
- `TaskSpaceIntent`: `task_id,skill_id,revision,frame_id='world',tcp_id,source_knots[50,SE3],source_times,execution_times,gripper_events[{source_progress,requested_width,ordering}],requested_goal_source,previous_ref_id`. Event times reproject through actual geometry progress, not merely wall-clock stretching.
- `ReferenceGovernorResult`: `future_ee[41,4,4],future_base[41,4,4],absolute_mpc_start,sequence,valid_until,violation_list,geometric_distortion_mm_deg,committed_progress`. Native encoder emits exact receiver JSON and reference manager's 41-point window; raw model never publishes ROS itself.
- `ExecutionFeedback`: `observation_timestamp,actual_tcp,actual_gripper,contact,base/arm joint state,commanded_reference_id,command_observed_or_applied_provenance,mode,stop_reason`. Differentiate target echo, arm-command observation and real native consumption. Original frozenMRT selected ID remains unrecorded; do not synthesize an ACK.
- `GripperLedger`: `episode,student_owner_from_reset,issued_seq,last_acked_seq,last_consumed_seq,last_consumed_width,actual_width,contact,lease`; cannot useactual_width as previous commanded width. Stage transitions only after independent receiver/consumption evidence.

## 3. 并行推进路径与不可跳过的依赖

```text
V0 Sealed Evidence / Baseline (CPU) ─┬─ V1 Motion Compiler + Shadow (CPU)
                                     ├─ V2 Student-owned Gripper CPU Contract
                                     ├─ V3 Known Dynamic A1 CPU safety & profiling
                                     └─ V4 SLA high-level intent/stage diagnostics (CPU; GPU optional)
V1 + V2 + V3 approved native qualifications → V5 Genuine Frozen-SLA noncontact pilot
V5 + student gripper/contact qualifications → V6 Single-object controlled grasp/lift/place
V6 + evidence of skill/generalization gap → V7 Go2 mixed-source posttraining
V6/V7 → V8 Long-horizon compositional evaluation → V9 mobile-base extension (separate)
```

### V0 — 实证基线统一（CPU/只读，即可进行）

精确核对 `#439 O0/O1/O3`、`#437 F2`、`#435 D3` 有效SHA、模型数据观测/权重哈希、原16窗口、所有初态筛除与成功分母；冻结原安全门数值及后端二进制。生成 `V0_EVIDENCE_MATRIX_ACTUAL.json`：
- 官方权重： `train_robodojo_real` + `extra_frames_first=false`、source frame正确、真实head历史1sec、虚拟左腕/左臂OOD；每窗 source-vs-Go2独立ID。
- O1：0/16有效、420/1718动态拒绝、103/317与lead-in跨窗分类、8源gripper阈值事件、0/16 owner；F2：慢速静止4/4而非动态任务4/4。
- 禁止重跑已具备相同入参/随机种子/检查点的forward。新运行必须有确定的额外信息增益（新scene/prompt/真实摄像头消融）。

交付 `V0_BASELINE_ACTUAL` 和接续至下一PR的source/checkpoint/sim SHA清单。任何文件不可得即`SOURCE_INCOMPLETE`，不以公开摘要替代私有证据。

### V1 — 轨迹编译器（优先CPU，不执行学生原生）

**P0**：旧 `RAW_H40_G7` + V1.3仅重映射时间 `TIME_ONLY`，保持封存拒绝/统计不变。  
**P1**：保持前一条**已消费**世界参考值/速度接缝，不重新绑定当前实际TCP；独立 `SE3_C2_MIN_JERK`/曲率控制方案，在固定世界系TCP和关节/碰撞限制下重新求**可行时间律**，而非仅缩放线性段；同时检查40点50ms原门的速度/加速度和发布窗口内部结点。
**P2**：只有必要时允许带预算、可报告的几何平滑（例如安全通道中的B-spline、min-jerk、分段SO3），记录相对全部原始50结点的最大位置/角度偏差，不得在接触或事件锚点附近改变抓取目标；几何改变的结果不得冒称 `time-only`。
**P3**：分别为来源49/30秒动作、执行绝对时间、0.5s新意图机会、10Hz传输、1sMPC优化、2s滚动窗口建时钟。持有完整技能，发布未来2s，不在每P10把已走路径重新归零。第一0.75–0.80s transition应作为**独立前置对齐阶段**记账，禁止把其进展计为SLA自身进展，也不能强行规定首0.5s必须完成SLA动作。
**P4**：对全16窗每个候选完整重算全部1,718滚动预览及每条加速度拒绝，不筛选通过窗；分解transition、SLA段、跨段、曲率结点、时间接缝，报告 `MAX acceleration/jerk, velocity, distortion, nominal source progress, event timing/path progress, G7 last_consumed`。要证明科学进度，必须有**非平凡SLA路径进展**、至少一个真实任务相关事件未被错置（若源原本没有事件则不得制造），且最终轨迹全程合法。
**资格门**：CPU完整shadow `100%合法原安全门`、物理可达与接缝、不发生前缀覆写；同时轨迹仍忠实表达SLA意图。未通过则明确 `V1_DYNAMICS_BLOCKED`，不换一套更宽安全门掩盖问题。

### V2 — 学生夹爪完全拥有控制权（CPU→新许可native，无接触先行）

CPU复用已封存55个 `gripper_trace` 测试，并添加 `student reset/owner token/clock epoch/lease/last-consumed seq/40mm/s/ACK→native-applied/actual/contact`负例。  
正式新native测试必须从**无接触、student-owned reset**建立真实owner，不从老师已抓持状态直接接管；先做不接触夹爪闭/开/保持，真实双指运动和命令幅度受控，才允许进入接触。所有带老师抓持的8旧窗保持BLOCKED；即使CPU合成owner正例也不能改写历史0/16。
**资格门**：每张测试卡有独立来源真实owner回执、逐命令ACK、消费、实际宽度和contact、超过40mm/s拒绝、安全STOP后0tick；缺一项`V2_GRIPPER_BLOCKED`。

### V3 — 冻结WBC＋A1动态执行资格（CPU→新许可native）

先从非SLA已知EEF/SE3路径开始，任务目标不仅是9.8–10s静止收敛，还要覆盖适配后源路径的曲率、方向、速度、姿态，分别测自由空间 **moving target** 误差、起始过渡、非接触接近后的停稳、意图revision边界。严格保留F2静态4mm/2°保持1s作为**旧工作点**，不能把其结果称动态通过。
预注册A0 no-servo vs A1 feedback，匹配初态/完整分母/两种启动簇；更新频率、单步上限0.40mm/100ms（如要改变，另立独立controller protocol）、原H40安全、base四脚接触、故障停止。动态阶段在执行器能力证明前禁止contact；到达任务对象附近静止/夹爪动作必须等状态门。报告全过程绝对/增量RMS、阶段末端误差、速度、收敛时间、落入容差的dwell、参考偏移、止损次数和被拒次数；**不能只看终点4mm**。
**资格门**：正式运行前按任务几何/碰撞/实际传感器噪声锁定 `epsilon_dynamic`、`epsilon_pregrasp` 和最大容许执行时长，满足所有预登记合法条件、无违法contact/STOP后积分；失败`V3_DYNAMIC_TRACKING_BLOCKED`而不直接修改冻结WBC，必要时单列底层控制研究。

### V4 — SLA真实高层能力与阶段规划能力（可与V1–V3并行，先离线）

测试三个**严格区分贡献归属**的配置：
- `H0_KNOWN_TRAJECTORY`：已知任务意图＋同一WBC/A1，测执行器上限，**不计SLA规划**。
- `H1_FROZEN_SLA_LOCAL`：官方后训练SLA输出50×16动作与夹爪提议，Task Executive仅提供安全拒绝/完成/超时等客观谓词，下一段EEF仍由SLA基于**自身视觉反馈**决定；能成功可称SLA局部闭环技能，不自动称长程规划。
- `H2_SLA_HIERARCHICAL`：在同一冻结SLA视觉/空间表征上独立训练/评估 `subgoal/phase selector`（或在模型可校准的输出上明确证明其真实阶段选择），显式输出 `{skill_id, target_object_from_vision, desired_EEF_trajectory, gripper_event, terminate_condition}`；安全FSM只能 veto/HOLD不能替模型选答案。若以规则决定下一阶段，应另列 `H2_SCRIPTED`基线，不能把任务成功归因于SLA规划。
长时域任务由易到难分开报告：源分布相近单臂水果类局部动作；Go2固定底座单物体接近/抬升/放置；扰动后的重抓与恢复；两物体顺序处理与改语言指令；移动底座参与单列更后阶段。任务进度、阶段边界、下个目标、gripper事件应依赖真实观测反馈，不能按 source50点时间索引自动关夹爪。
离线指标： `object-relative task goal error`（使用仿真GT仅**评分**）、模型选择哪次事件、episode阶段混淆矩阵、source→target OOD输入敏感性、视觉缺失/虚拟左臂消融、新意图在执行反馈后修订的真实性、推理延迟分布。原16窗是开发样本，独立held-out采用新episode/新物体位置/新视觉背景/seed，并与source近域任务分层。
**资格门**：模型确实提供非零任务相关EEF路径、接近/抓放方向正确，事件可被映射到真实空间进度且不会被时间扭曲；无新GPU权限则继续已有16窗CPU审计，待新许可才能采集新预测。

### V5 — 官方冻结SLA真实非接触试验（需V1＋V2相关准备＋V3准入）

仅在学生合法control owner、真实WBC接收器消费与动态精度成功后，新建**独立真实native/TRAIN科学人口**。本阶段可在经过独立安全许可后只动EEF、夹爪保持开/合法持有。对照： `H0 known trajectory` / `FrozenSLA+ReferenceManager(no A1)` / `FrozenSLA+ReferenceManager+A1`。不沿用老师持物的旧窗口、不用学生意图伪造真实反馈。每组固定source checkpoint和场景/初态/相机/prompt/接收时钟，记录安全拒绝、安装、实际消费、动态误差、SLA自身路径进度、修订与停顿；全量注册分母含启动失败与技术失败。
若学生不能产生有目标几何意义的轨迹，则归类`OBS_OOD_OR_SKILL_MISMATCH`，不通过加大A1补偿掩盖；若轨迹正确而执行层不准，归类`REFERENCE_OR_CONTROLLER`并先处理对应门。

### V6 — 夹爪接触、单物体Grasp→Lift→Place

必须另行获准 contact 和运动中故障STOP资格。顺序：独立小幅接触与已知EEF正对照→学生自己reset后关闭/确认抓持→物体抬升（例如≥20mm作为此前常用指标）＋持续保持（例如≥1s）→运输到预定义位置→释放/放置。接近、夹爪关闭、抬升与释放由模型动作/阶段输出提议，但由 `actual TCP error/contact/gripper consumed/object lift` 等证据门允许发生。防止源模型在执行失败时“按照到点时间”直接张开。
A1对 contact 不能默认继续追位置：接触事件触发独立力度/位置与原生接触安全分支。若无合格夹爪owner、动态stop、接触安全，保持`NOT_RUN`。
任务报告： `grasp, 20mm lift, 1s hold, transport, place, full success` 与长尾时延、误抓、非法释放、接触失败，所有条件含拒绝完整计数。不以静止终点成功替代完整任务成功。

### V7 — 条件触发的Go2 Multi-source Post-training（先准备数据，可与V4并行）

**官方实机后训练权重已使用**，不是再从SLA_base训练60k。仅在有证据说明源模型的视觉/任务/阶段意图不对，且已知任务的执行接口可用时，才申请新GPU后训练。基座模型官方 `Pi0Config(pi05=True)`、50动作/32统一内部槽位、真实物理16D、官方checkpoint参数与norm stats钉住；发布real训练`extra_frames_first=false`，源头官方 `train_robodojo_real`混合piper_x/arx_x5/piper。新训练作为不同run/数据定义/归一化版本，不能沿用checkpoint自带optimizer state（release无train_state）。
目标Go2训练数据按**真实**20Hz/2ms视觉时间构建必要历史；左腕缺失有显式camera provenance / dropout适配，虚拟左臂和右臂真实观测区分。Teacher **task EEF intent**、送入 WBC 的command-reference、actual TCP、夹爪previous consumed与实际指位、物体接触成功必须不同字段。永远不把A1额外约34mm参考偏移当作SLA目标动作标签。
小规模严谨消融： `OFFICIAL_FROZEN` / `GO2_ACTION_ONLY` / `GO2_CAMERA_AND_ARM_ADAPT` / `GO2_MULTITASK_WITH_SOURCE_MIX`；优先可审计PEFT/LoRA或小头，再在必要且有计算配额时全参；实现前证实与官方结构和归一化兼容。分源采样权重不盲目使用官方三域等权；单任务探索后至少扩大到 `approach/pregrasp/grasp/lift/transport/place` 等有真实成功数据的技能；TRAIN/VAL/TEST按**episode、布局、任务类别和种子**分离。
按 `successful_demonstration_count`,任务/布局覆盖、独立验证性能优于冻结模型、事件稳定性、OOD泛化及遗忘量决定模型版本，不以训练loss单独放行。每次GPU训练单独报worker/GPU UUID、样本/epoch/优化更新、数据源SHA、参数与norm stats SHA、held-out任务；没有新GPU lease则状态`WAIT_GPU_AUTHORIZATION`。

### V8 — SLA长程任务理解/规划证据

只有V6能合法运行，再比较H1和H2：1) 单阶段局部准确；2) 从任务文字独立选择接近/抓持/抬升/放置；3) 改变物体位置或语义后重规划；4) 两物体顺序、跨阶段记忆、恢复；5) 新指令顺序组合。模型是否真有长程能力，要在不给它具体分步prompt/oracle的位置条件下测 **subgoal accuracy, ordering constraint, error recovery, completion / total elapsed time, unnecessary commands/unsafe proposals**。必须分别报冻结源模型和目标PT、所有episode数与完整科学终态，禁止把执行器的安全FSM当作模型规划。
建议先固定底座任务，成功后单立V9移动底座分支。当前官方SLA没有显式base action；不能让EEF移动解释为已学会Go2导航/移动抓取。

## 4. 自主实验治理：状态机、工件与授权规则

每个阶段必须为独立分支/PR和确定的 `registration.json`（只读内容键、源/模型/二进制SHA、场景、输入感知来源、stage/seed/计划完整卡序、每卡native cap、总上限、GPU/model forwards与training updates、runtime owner独占lease、停止时钟、明确定义科学门）。**这份长期计划不是任何新原生/GPU配额**；具体物理/GPU预算由现场授权后建立后继registration。对于CPU可审计工作，允许按本计划开发和自测，但实际资源作业必须被授权owner通过独立准入。
状态：
```text
V0_READ_ONLY -> V1_CPU -> V2_CPU -> V3_CPU -> WAIT_NATIVE_AUTH
   ├─(V1 dynamics fail) COMPILER_BLOCKED (no native)
   ├─(V2 owner missing) GRIPPER_BLOCKED (no contact)
   ├─(V3 A1 dynamic fail) TRACKING_BLOCKED (no student native)
   └─(all success + approvals) V5_STUDENT_NONCONTACT
       -> [separate gripper/contact/stop approvals] V6_SINGLE_OBJECT
       -> [true policy gap] V7_POST_TRAINING
       -> V8_LONG_HORIZON
```
V4 policy/architecture CPU和数据质量可与V1–V3并行；模型GPU forward需单独批准。遇到失支撑、非预期接触、接收身份错、source clock回退、过期、非法夹爪事件、开爪持物、STOP后任何积分、核心可执行binary/训练SHA漂移、预算超额、owner lease冲突立即`STOP_AND_SEAL`。已确认暂停边界的时钟倒退STOP成功并不等于运动中STOP已验收。
后继可推进到 `READY_FOR_AUTHORIZED_NATIVE` 的算法/registration稿；**不能把“自主”理解成可以自动花GPU资源、自动接触、改真机配置或在后续会话外自发持续运行**。

## 5. 每步必须生成的结果与“成功”定义

- V0：`V0_EVIDENCE_MATRIX_ACTUAL.{md,json}`，所有历史分母/哈希一致；
- V1：`V1_COMPILER_ABLATION_ACTUAL.{md,json}`、逐窗加速度拒绝/路径扭曲、事件相对进度、original gate原样复演，单元/负例；
- V2：`V2_STUDENT_GRIPPER_OWNER_ACTUAL.{md,json}`，先CPU后独立原生；
- V3：`V3_DYNAMIC_A1_ACTUAL.{md,json}`，已知时间变参考、真实时延、动态绝对误差和新的STOP资格；
- V4：`V4_SLA_SKILL_AND_LONG_HORIZON_DIAGNOSTIC.{md,json}`，标注哪些阶段由SLA决定、哪些是脚本/安全门；
- V5：`V5_SLA_REAL_NONCONTACT_ACTUAL.{md,json}`，真实SLA前向/引用/物理消费；
- V6：`V6_GENUINE_GRASP_PLACE_ACTUAL.{md,json}`，独立学生完整任务分母；
- V7：`V7_GO2_PT_MIXED_SOURCE_ACTUAL.{md,json}`，学习更新/验证/防遗忘；
- V8：`V8_LONG_HORIZON_HELDOUT_ACTUAL.{md,json}`，语言组合/恢复/轨迹/任务成功。
所有report必须输出 **新GPU forwards、优化器updates、native requests、physics ticks、native stops、after-stop ticks、租约是否释放、实际任务尝试与成功次数、blocked与technical failure分母**。`NOT_RUN`不得写0%成功率；新接口不改写legacy H40/G7 0/16。

**严格的研究成功条件**：在明确冻结的视觉输入、任务分配、WBC二进制与A1参数下，官方或Go2-PT SLA 的**自生成**运动目标和夹爪动作通过合法接收/消费，实际成功抓放并在未见过的任务/布局上完成多阶段或顺序指令；达到可报告的科学成功率、误差和时间指标，同时安全门/STOP与所有失败分母完整。此前F2的静止4/4是必要工程进展，但不是上述结论。
