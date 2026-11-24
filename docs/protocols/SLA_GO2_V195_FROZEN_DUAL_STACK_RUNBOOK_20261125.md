# V1.9.5 执行清单：冻结双栈，链接层自主推进

2026-11-25，Asia/Singapore。配套：[主计划](SLA_GO2_V195_FROZEN_DUAL_STACK_PLAN_20261125.md) · [登记 JSON](SLA_GO2_V195_FROZEN_DUAL_STACK_REGISTRY_20261125.json)。

本文件是给已有合法现场访问与执行权限的代理的实施交接，不是已存在的新 CLI。文中建议的新模块尚未实现；不要把伪代码当作已经通过原生验证。最新用户已授权本轮链接层工程修复与任务分流，无需再逐小项申请批准。

## W0. 接手即做：实际绑定一次，然后工作

1. 核对现有项目 owner、物理进程、模型 worker 和 lease。只在真实无冲突的合法边界启动；不修改正在运行的 capsule。
2. 读取现场 V1.9.4b 绿色实际 driver、V1.9.4c R01/E0/E1 代码与报告。记录 SHA 与必要 dirty diff hash、真实模块路径、场景/reset、WBC/腿/动力学、receiver 和依赖版本。`#510@719882...` 是复用索引，不是最新现场 runtime 身份。
3. 复用已通过的模型和 processor manifest；可信未变化的大权重不每回合全量重读。发现身份不可信才重新完整校验。
4. 绑定本次 `T0`、`night_deadline=T0+8h`、实际可用资源、输出路径、原终末流程。保留现有 Journal/onceIO；不增加影子账本/审批流程。
5. 先完成最小代码改动和受影响检查，再 Q→该布局学生任务。不要等全历史日志上传、另一个布局或远端无关作业。

现场已报告文件位置仅作查找入口，实际存在性与内容要读取后确认：

```text
/home/liuyu/ISSAC-GR00T/work/sla-v194-direct-frozen/reports/SLA_GO2_V194/ACTUAL.md
/home/liuyu/ISSAC-GR00T/work/sla-v194b-green/reports/SLA_GO2_V194B/ACTUAL.md
/home/liuyu/ISSAC-GR00T/work/sla-v194b-green/reports/SLA_GO2_V194B/ARCHIVE_AND_LANGUAGE.md
/home/liuyu/ISSAC-GR00T/work/sla-v194c-world-reanchor/reports/SLA_GO2_V194C/E0/ACTUAL.md
/home/liuyu/ISSAC-GR00T/work/sla-v194c-world-reanchor/reports/SLA_GO2_V194C/E1/ACTUAL.md
```

发布者本次没有读取上述现场路径。缺少文件时使用已有真实输出定位，不制造哈希或成功回执。正式运行所需的实际源码、机器人和 owner 身份必须绑定；完整旧审计报告的上传不是新运行的额外前置。

## W1. 复用位置与最小改动

以下索引固定在 [719882e9fd154ffbe995484e7a766ada3b3ecd02](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/719882e9fd154ffbe995484e7a766ada3b3ecd02)，实施时先对照现场同名模块：

| 已有代码 | 复用内容 | 注意事项 |
| --- | --- | --- |
| `scripts/sla_go2_v18/world_intent_clock.py` | 世界相对位姿公式、原消费时钟 | 不把原逐提案 world_offset 与新跨查询补偿叠加 |
| `scripts/sla_hierarchical_v16/compiler.py` | 完整 SourceCurve、线性位置/Slerp、源时间、原约束重定时 | `at` 的 source phase 是秒，不是源行号；保留全 50 行 |
| 现场绿色名义 codec / PT driver | 变夹爪、MODEL_FIRST_KNOT、真实消费后查询 | 优先在此新增一个 profile，避免重造 driver |
| `scripts/sla_go2_v19/sea_three_view.py` | 真实三视图 V1、因果 head 历史 | 不能使用旧 worker 黑填第三视图的路径 |
| `scripts/sla_go2_v19/sea1a_panel.py` | `rng_noise`、官方采样配置 | 保留实际噪声 tensor hash，不假设跨硬件重新生成字节相同 |
| `scripts/sla_go2_v19/sea_fruit_h0_native.py` | 合法 P10、owner、接收/消费与终态路径 | F 学生阶段不能调用 teacher.decide/commit/stage |
| `scripts/sla_go2_v19/sea_fruit_receiver.py` | 原接触、开度、速度、站姿及 STOP | 不通过 fake contact、改阈值或 epoch 重置绕过 |
| `scripts/sla_go2_v19/sea_fruit_h0_score.py` | 原实际 grasp/lift/hold 判据 | P/H 使用新交接高度基线并单列，不回写旧 F 评分 |
| `scripts/sla_go2_v19/sea_tool_geometry.py` | 真实 TCP/指垫几何 | 比较物理意图、安装参考、消费参考和真实双垫 |

旧 `SEAStandingStudent` 有固定开爪、旧 interval/query 上限等语义，不可直接拿它当本轮变量夹爪任务 runner。现场 `fusion_trajectory_adapter`、`mobile_boundary_runner` 等依赖若未在远端 tracked tree 内，要记录实际文件 hash；不得假定 Git HEAD 已覆盖全部代码。

建议新增小模块 `scripts/sla_go2_v19/sea_feedback_execution_v195.py`，只保存 profile、入口锚、有限补偿和事件记录；现有 driver 传入真实消费状态。此文件名是实施建议，尚不存在。不要同时替换模型 worker、编译器、owner、receiver 和评分器。

## W2. 冻结接口逐项保持

### 模型与动作

- checkpoint：`haofuly/SLA_benchmark_ckpts/SLA_Memory_robodojo_real_ckpt`；revision `7f6880569c3be777291e646fad1b4951245e8553`。
- upstream：`thriving191228/XPolicyLab@c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454`；`train_robodojo_real`；`extra_frames_first=false`；attention `xla`。
- raw `50×32`，一次官方 inverse 后 `50×16`；每臂 `[x,y,z,qw,qx,qy,qz,g]`，左后右；物理右臂，既有 A1 左臂不变。保存两臂全部输出，不按结果交换手臂。
- state `16D float32`；右臂/夹爪来自实测。源动作时间 `i/30`，最后 `49/30 s`。
- 噪声 `rng_noise([query_index, episode_seed])`，episode 内从 query 0 单调计数；补偿/重发不是新模型查询。
- 官方配置/统计/tokenizer/manifest SHA256 见 JSON；0 次模型、adapter 或 WBC 学习更新。

### 名义位姿与夹爪

沿用 `FLANGE_EQUIVALENT_NOMINAL`，不伪称源工具物理标定已完成：

```text
source_zero = [0.09519999999999992, 0.001000411398944587, 0.15649999265207865] m
T_WV = T_W_target_armbase * Trans(source_zero)
observation = inv(T_WV) * T_W_TCP_actual * inv(T_link6_TCP)
intent      = T_WV * T_V_link6_predicted * T_link6_TCP
```

`T_WV` 按现役名义接口绑定查询观测，不从未来 base 状态重算同一输出；source_zero 只加一次。实际状态输入不回填成目标参考或反馈修正值。

```text
single_finger_command = 0.002 + 0.040*g             # metres
wire_fraction        = single_finger_command/0.044 # EEF10[9]
observed_g           = ((w_left_actual+w_right_actual)/2-0.002)/0.040
actual_pad_gap       = 0.001048+w_left_actual+w_right_actual
```

命令工作范围 2–42 mm，机械实测范围沿原 0–44 mm 与 native valid 规则。实测 `g` 可能略超 [0,1]，不得手工 clip 真观测；命令域外拒绝与官方 inverse 保持原样。42 mm 端点用原 receiver 同一规范表达，验证完整 encode/decode 链；不以扩大上界或通用 epsilon 修掉已报告拒绝。

MODEL_FIRST_KNOT 从最后**已消费命令**接到模型 `g(0)`，使用原五次 ease `10u³−15u⁴+6u⁵` 与单指 40 mm/s 上限/原裕量；记录 `1.875*abs(delta_w)/lead`。夹爪 grant 取当前 atom 对应 `EEF10[row][9]`，不是下一行。连续 epoch/sequence、原接触有效性和释放拒绝规则不变。不能为过旧接缝检查临时声明 `contact=false`。

### 真实三视图与历史

机身 256×256、右腕 256×256、aux V1 480×640，RGB uint8；aux 映射官方 left-wrist 槽。V1 位置/视轴/FOV 保持原值，全部当前图像来自同一真实 packet。历史 head：

```text
[0, max(0,t-4000), max(0,t-3500), ..., max(0,t-500), t]
```

对应 initial、过去 8 至 1 s、current，共 10 帧。来自本回合 native journal，允许原非积分回放渲染；不能用当前帧伪装过去或源录像替换目标图像。P/H/T 前缀也属于真实本回合历史，并在结果披露。相同像素 hash 不等于旧状态；新鲜性由实际 tick 和消费决定。

## W3. BTF-1 精确状态和更新

### 状态

```text
episode_id, profile_id, anchor_native_tick
b0, b_committed, D0
query_id, source_phase_s, fixed_target_phase_s
correction_count_at_target, ramp_id, actual_ramp_end_tick
contact_lock, pending_bias, acquisition_start_tick
last_consumed_ref_id, last_consumed_eef, last_two_consumed_eef
```

无需新建独立事件账本，上述字段加入原 manifest/逐 atom 日志。`b_committed` 表示后续正常 SOURCE 使用的补偿；未实际消费完的候选补偿记录为 pending，不能以安装/ACK 当完成。每个 epoch 的实际命令另存，确保 ramp 内也能回溯。

### 入口

同 tick 实测与消费对计算 `b0=pC-pA`、`D0=RA.T@RC`；`b=b0`。若 `||b0||>40 mm`，本 profile 拒绝，不 clip 入口事实。P/H/T 在明确 handoff 时建立自己的新 episode 控制锚，不能带入教师的未执行 suffix；源记忆仍使用真实因果观测。H/T 若已接触，入口只取一次锚并立即 contact_lock。

整个 episode 内不重置 `b0`，不逐 query 重算 `D0`，不把上一条已补偿曲线当物理意图再次补偿。更新模型状态时仍传实测 TCP，绝不传 `pA+b`。

### 补偿资格与动作

满足全部条件才计算补偿：

1. 原允许接触标志和原 contact journal 表明该段无接触。
2. `s_target` 是已实际消费的固定物理意图端点；源相位已到达，不是未来 horizon 末点。
3. 原编译器形成了合法静止参考端点；incoming 与最后两条实际消费参考一致。
4. ramp 的最后 atom 已实际消费，随后保持同一目标完整 0.5 s native time；不存在未完成安装/时钟错配。
5. 当前目标补偿计数 <3，初始 acquisition 未达到 6 s 总期限，episode/cell 时间端点尚可容纳新完整动作。

以最后 0.25 s 的实测平均计算误差；最后 0.25 s 全部样本峰值已经 ≤4 mm/2°则接受端点，直接继续源动作，不做不必要补偿。位置不合格时使用 L2 单步 ≤4 mm、总额外 ≤12 mm、绝对 ≤40 mm 的候选。任一边界将被超过时，不投影到更有利的目标、不逐轴独立限 4 mm、不保留未应用积分；记录 `BIAS_BOUND_REACHED` 并结束该 acquisition。

新命令从最后实际消费参考经现有 admissible ramp 到 `intent(s_target)+b_candidate`，源相位保持 `s_target`、夹爪保持该相位模型值。原速度、加速度、关节、可达与安全检查覆盖全部 ramp/preview。完成后再 dwell；此时才能提交 `b_candidate`。每次接受/拒绝及实际使用的 b 都写原日志。

姿态误差若不合格，BTF-1 不新增姿态积分，也不忽略它。记录 `ORIENTATION_QUALIFICATION_FAIL`；检查真实旋转约定或登记后继 profile。

### 单调源时钟与不阻塞模型

- 初始 ALIGN 以 `s=0` 到模型首点；最多 6 s 包含所有 ramp/dwell/补偿。超时 `ACQUIRE_TIMEOUT`，使用原安全终末流程；不重新 query 让计时归零。
- 正常 SOURCE 单调推进；只有合法端点允许有限停留。中途停点须先由现有编译器制动至明确 `s_stop`，pose/gripper 同时重定时；每提案最多一个可选中途停点。
- 非零 outgoing velocity 的 P10 边界不能直接当端点。无法安全制动就跳过补偿，以现有 b 继续；不能硬切实际位姿制造停点。
- 每固定目标补偿最多三次；源尾最多处理一次端点 acquisition，之后转下一合法 query/终态，不重复把尾段当新目标重新获得三次额度。
- 补偿 hold、BRAKE、ALIGN 与零源增量 atom 不计作正 SOURCE 消费；更换 query 不重放旧源事件。每份原始 proposal 的 consumed intervals 单独存。
- 不用“闭合后不再前进”“成功后偷偷保持”“跳到50行中第一个闭爪点”来提高分数。

### 接触时的精确处理

正常 SOURCE 中的原允许 pad–fruit 接触：即时锁住后续偏差更新并取消尚未安装的 pending update，模型的源动作、夹爪和合法查询继续。contact_lock 只限制误差补偿，不是抓持阶段 STOP。不得等待物体脱离才允许抬升。

若接触发生在已经安装、尚未消费完的**补偿 ramp** 内，不能将 pending 更新当已完成，也不能假装剩余 ramp 消失。先在真实 contact tick 锁住新的误差更新；到下一合法安装边界时，核算此前仍已实际消费的 suffix，再复用现有原生接口支持的安全取消/制动接缝。制动中固定同一 `s_target` 和模型夹爪相位；制动实际消费完后锁存 `b_held=p_C_brake_end-p_intent(s_target)`。这是命令减物理意图，不是命令减实测，不是新 world_offset；D0不变，b_held仍须满足注册边界。之后在合法边界继续未消费源轨迹与查询。接触即时冻结的是误差更新；连续制动中的有效参考偏差直到制动结束才最终锁存，不能声称瞬间固定了已安装参考。

若现役接口没有这样的原生、可验证取消能力，**第一候选不临时发明它**：用原有安全终末 HOLD/暂停流程结束该 attempt，记 `ADAPTER_CONTACT_CANCEL_UNAVAILABLE`，并保留实际原安全门是否触发的独立字段。不能把这个 adapter 退出伪报成原碰撞门触发，也不能继续沿未知 ramp 顶住物体；终末暂停不算模型合格保持。修复取消接缝须在新的明确版本中复验后接续。

禁止积分接触误差；接触释放后清空所有未应用误差。只有新的无接触端点、完整 ramp+dwell 才能再次更新。原不允许接触和其他原 STOP 始终立即优先。

## W4. 受影响接缝检查，不扩建测试平台

这些检查针对真实风险，不要求重跑所有历史 panel：

| 检查 | 要证明的内容 |
| --- | --- |
| 入口与跨 query | `I=A0` 对应 `C0`；b0/D0 只加一次；新 query 不重置累计补偿/接触锁 |
| 非交换旋转 | `R_I @ D0` 与登记右乘一致，不偷换 SE(3) 左乘 |
| 单步与累计限幅 | 3D L2、4/12/40 mm 和三次计数；拒绝未消费 update 不留积分 |
| 因果和相位 | 未来 knot 不能当实际消费；dwell 无源进度；查询输入来自其自身实际执行 |
| 接触 | 正常抓持后 SOURCE/jaw/query 继续；pending 更新取消；无原生取消能力时受控终止 |
| 端点与 jaw | 42 mm 端点通过原接收器同一表达；0/1 映射、实际 g 不 clip；每 atom 行对应正确 |
| 原约束 | 所有 ramp/preview 走原速度/加速度/接触/站姿门；已被拒 DIRECT 不强行复跑 |
| 终态 | STOP/HOLD 终末后新增物理积分与安装为0；未知写入先查实际效果 |

纯函数/回放检查只能证明实现接缝，不能称 Q 已通过或学生成功。修复普通日志/路径问题只验受影响项；修改共享控制接续时应加有意义的回归。完成这些后运行 Q，不以测试数量代替实际任务。

## W5. Q00/Q01：已知意图资格

在原稳定入口取 `p0,R0`，固定全开。源动作与评分窗在运行前写入本卡 manifest：

| 顺序 | G00/G01 共有目标 | 目标姿态 |
| --- | --- | --- |
| 1 | `p0+(0,0,4 mm)` | R0 |
| 2 | `p0+(0,0,8 mm)` | R0 |
| 3 | `p0+(0,0,12 mm)` | R0 |
| 4 | `p0+(0,0,20 mm)` | R0 |
| 5 | G01 追加：同20 mm位置，世界X转+2°，随后回R0 | 对应旋转 |
| 6 | G01 追加：`p0+(4,0,24 mm)` 且世界X转+2°，随后回20 mm/R0 | 对应旋转 |
| 7 | `p0+(0,0,4 mm)` | R0 |
| 8 | p0 | R0 |

世界X旋转采用 `R_target=RotX(angle)@R0`。每目标采用相同原编译限值和 BTF-1 端点规则，最多三次修正。完整动作在启动前经过原门预检，包含入口及返回双垫净空；不能为增加观测而主动触桌。该序列不取物体 GT 生成目标。

每卡 ≤30,000 tick/1,800 s 墙钟、0模型查询。报告全部过渡/端点误差、同相位物理意图、b0/b、每次修正、饱和、姿态、左右 pad 实际净空、原安全与终态。

每目标最后一次实际 ramp 后 0.5 s dwell 的**最后0.25 s**为固定接受窗，所有样本位置≤4 mm、角度≤2°；至少上/下运动的有符号响应正确，G01 的正旋转方向正确，无饱和/发散/原安全异常。任何目标失败，不允许后来重新挑一个窗口宣布同目标通过。对 Q 的全程峰值另报，不藏过渡大误差。

Q00 通过可立即 F00/F01；Q01 通过可立即 F10/F11。另一布局失败不自动使已通过布局失效，除非暴露共享缺陷。任务可在原无接触资格覆盖内运行，接触负载资格由实际任务验证。H 使用对应布局资格，加实际交接抓持/负载状态和原安全检查进入；若共享执行缺陷仍未解决，不能仅因起点更高就继承资格。可用共享工程后继完成必要的最小针对性资格，不要求重跑整套历史卡。

## W6. 学生任务 driver

```python
# pseudocode only: bind actual driver/owner names in the site workspace
bind_frozen_identities_once()
for registered_card in routed_cards:
    initialize_with_existing_native_owner(registered_card)
    if registered_card.task in {"P", "H", "T"}:
        execute_or_restore_verified_registered_prefix()
        verify_handoff_predicate_and_remove_teacher_control()
    feedback = BTF1.from_same_tick_actual_and_consumed()
    while not original_terminal and within_registered_endpoint:
        packet = actual_causal_observation_at_legal_boundary()
        raw50, decoded50 = frozen_official_model(packet, registered_noise())
        keep_raw_and_nominal_intent_immutable(raw50, decoded50)
        compiler = existing_full50_compiler_with_feedback_profile(feedback)
        consume_by_original_p10(compiler)  # correction atoms are logged separately
        record_object_events_and_actual_consumed_intervals()
    original_terminal_closeout_and_actual_report()
```

上面不表示可直接启动的函数存在。实施者应直接修改现场绿色 driver 对应调用点，保留既有 owner/ACK/receiver。不要让伪代码忽略原 P10、接触取消或计数规则。

### 原生时钟

| 项目 | 固定值/规则 |
| --- | --- |
| MuJoCo/native dt | 0.002 s |
| 一个 atom | 25 tick = 0.05 s |
| P10 | 10 atoms = 0.5 s，不是十个30Hz源行 |
| Preview | 41 knots，0.05 s间隔 |
| 引用刷新 | atom0/7；TTL 0.4 s，沿原 owner |
| 查询资格 | 10个已验证正SOURCE atom，或实际源尾；只在P10合法边界 |
| 正源增量 | `delta_source_s>1e-12`；源尾容差沿原1e-9 |
| 补偿观测窗 | ramp实际结束后0.5 s，末0.25 s测误差 |
| 查询上限 | 下一次新调用之前判断；最后接纳proposal正常消费至下个查询边界 |
| Tick上限 | 每grant前判断`tick+25<=cap`，包括初始化和前缀 |

正常 query 替换未消费旧 suffix 并保留其档案；只消费一条活动 proposal。补偿停点不能改 query counter、noise、packet timestamp，不能改 sampler。同步暂停式推理继续使用，今晚不要求异步实时部署。

### 注册名册与调度

F00/F01=G00 seed0/1，F10/F11=G01 seed0/1。P/H/T 同结构，分别以 P00…P11、H00…H11、T00…T11 命名。每个单独任务的原优先顺序为 G00seed0→G00seed1→G01seed0→G01seed1；布局暂不合格时可跳过等待并记录 `NOT_STARTED_UNQUALIFIED_LAYOUT`，不是换成新seed。

F 为主线，但到 T0+210 min 仍无成功时，下一可运行卡优先 P/H/T。P 若交接后仍无实际闭爪，可继续另布局排除局部问题，再进入 H；不把四张 P 全跑完设为 H 的强制门。正式注册卡因分流未运行应保持未运行；不得把它们换名补进有效分母。

首次任务成功时锁定 profile。C00/C01/C10/C11 使用同任务、新 seed2/3、相同初始化语义；记录 `selected_after_first_success`，是发展性确认，不称全盲独立测试。若只成功H，C也确认H，不把它换成F算新通过。

每卡原任务成功/失败、闭环资格、初始化辅助分别保存。任务成功发生于第一次源段时，不通过停止查询来规避随后模型表现；可在原安全/本卡端点内自然完成后续合法query和消费，直至合法终末仍满足任务且无后续原安全STOP，才标闭环合格。证据不足则保留`first_task_event=true, closed_loop_qualified_success=false`，任务原评分独立保存；后续丢失保持/STOP也要报告。

## W7. P/H/T 前缀与评分实施

H0 对照和 tick 索引只用于寻找候选状态；现场实际行为谓词才决定交接。教师可以在初始化使用目标几何；handoff 后该依赖从学生行动路径退出，GT 只留在原评分/安全路径。

| 字段 | P | H |
| --- | --- | --- |
| 交接指位 | 实际全开/原合法开口 | 真实抓持对应实测指位 |
| 交接目标状态 | 未被夹爪抓住、未受前缀移动；与该reset的物体位姿一致（按原测量精度） | 已实际稳定抓持，不假定只凭g值 |
| 高度基线 | 交接物体高度 | 交接物体高度，不继承原reset高度计分 |
| 学生抓取证据 | handoff后新形成原实际grasp | 说明抓持由教师初始化，不归学生 |
| 新事件 | ≥20mm新增抬升+≥1s合格保持 | ≥20mm新增抬升+≥1s合格保持 |
| 补偿入口 | 无接触入口锚，按BTF-1继续 | 一次入口锚后立即contact_lock |

前缀产生的 tick 和资源属于本次 attempt；准备失败也保留，不能把教师所有尝试隐藏。若使用已有快照恢复，必须含实际物理/控制器完整状态与真实历史，并通过已有恢复机制；纯离线截取图像不构成实体初始化。

P/H 的后续抓持与安全谓词复用原实现；高度参考改变属于明确的新任务定义，不修改旧 scorer 结果。原足支撑、base≤20mm/5°、足位≤5mm等保持原样。原正确对象 internal role 是 `red`，可见对象绿色；语言使用绿色，不能根据内部分数标签再次写红色。

### T：已抓持起点的语言条件搬运

T 使用同一场景中可见的蓝物体，不增加对象或放宽接触规则。原 H0 前缀先建立合法离桌抓持和保持，在稳定交接后完全退出；以交接时刻定义评分基线。无官方支持证据时，它就是探索性任务，不宣传已有源成功。

```text
f0 = fruit_position_at_handoff
b0_score = blue_object_position_at_handoff          # scorer only, not feedback b0
u = normalize((b0_score-f0).xy)
delta = (fruit_position_now-f0).xy
progress = dot(delta,u)
cross_track = norm(delta-progress*u)
distance_reduction = norm((b0_score-f0).xy)-norm((blue_object_now-fruit_position_now).xy)
```

连续1 s内同时要求 progress≥20mm、distance_reduction≥15mm、cross_track≤10mm、原实际抓持成立、物体高度相对原reset≥20mm，且原净空/站姿/接触安全合法。这里继承的初始高度归教师，新增水平位移才归学生。GT与u严格只供评分/安全，模型输入只有原三RGB、实测状态和固定指令；不能据此平移源轨迹、锁Z、限制水平轴或替模型保握。全3D运动仍保存，闭环成功仍需两次实际消费后的查询/输出和合法终末。手写水平轨迹仅可用于独立资格卡，绝不算学生任务。

若现有匹配证据未覆盖朝蓝物体的主要水平轴，用共享预留中的独立QH00/QH01检查，不先扩一套平台：原合法H0前缀到稳定离桌抓持，以交接实测TCP为原点，用已知意图作世界+Y 20mm并返回（当前紧凑场景蓝物体在该方向；现场先核实，若不同则在模型查询前登记对应固定轴）。同一BTF-1、接触锁、原限值、每端点0.5s dwell和末0.25s 4mm/2°验收；0模型前向、每卡≤50,000 tick含前缀。此已知载荷运动独立记为资格，不属于SLA动作。无安全资格则不启动T；通过后重新初始化T，不能沿用已完成搬运来计分。QH占用全夜共享预留，不另加隐形卡数。

T00/T01/T10/T11为两布局×seed0/1。可在210min时依据竖直执行/无闭爪等证据直接选T，无需先跑完P/H。C可确认T，仍单列为辅助初始化任务，不升级成完整抓取。

## W8. 有限自主修复循环

每次工程问题执行以下短循环，而非停机问GO：

1. 用实际日志确定症状与最早错误层；不预设所有问题都是 MuJoCo 或模型。
2. 写一个可证伪预测，例如“端点表达统一后，相同42mm值通过原receiver；其他域外值仍被拒绝”。
3. 修改项目代码，记录 commit/diff；模型/WBC/物理/原安全 identity 不变。
4. 验受影响接缝；涉及物理参考行为则新 profile + 最小Q/取消接缝复验。
5. 以新 attempt 执行一个真实任务增量，或明确合法拒绝；所有旧结果和成本继续保留。

不要求每次 bug 重跑完整两布局/全部历史模型面板。纯序列化修复不能算行为profile优化；改控制律、dwell、几何映射、源相位或任务起点必须先登记不同profile/条件。两个同类失败无新证据时读保存状态或切换独立已登记任务，不用改版本号制造进展。

未知写入/启动先查实际runID、进程、日志和lease；不得盲重放。有效科学失败不能用“技术问题”改名，科学后继必须是实质版本变更或预登记不同任务。预登记单卡正常query/tick/wall终点耗尽记`TASK_TIMEOUT`；全夜窗口或外部资源提前截断记`CENSORED`，不混入同一有效分母。

## W9. 最小产物与终态

建议使用原报告体系下 `reports/SLA_GO2_V195/`，目录名不是新框架要求：

```text
REGISTRATION.json       # actual bound manifest; distinguish from this plan JSON
ACTUAL.md
ACTUAL.json
Q/<attempt>/...
F/<attempt>/...
P/<attempt>/...
H/<attempt>/...
T/<attempt>/...
C/<attempt>/...
```

每实际回合关键字段：task/profile/layout/seed、实际实现/资产身份、是否辅助初始化、query/source phase、原50行动作、intent/installed/consumed/actual、b0/b/D0与补偿、真实夹爪/对象事件、左右净空、最早失败层、原安全原因、技术/截断状态、资源与终态。复用已有日志内容，不为每个字段新建一份回执。

最少报告以下计数：

- 全部已注册条件、已启动attempt、有效科学终态、技术中断、资源/时间截断、未启动。
- F/P/H/T/C分别的原任务成功和闭环合格成功；不同profile不拼接成功率。
- 原生tick、真实新模型前向、精确复用输出、训练更新0；教师初始化消耗单列但纳入总量。
- 原安全STOP次数、adapter自主终止次数、首次失效机制与所有失败；二者不互相冒充。
- 终态后新增安装0、积分0；实际关闭/未关闭资源如实报告。

完整回放需覆盖初始化/交接、所有学生源动作、首成功或失败和终态。公开仓库只保存允许公开的报告/哈希/索引与合适的小图，沿现有视频交付路径，不上传权重、私有数据、vendor资产或凭据。

## W10. 结束后的下一工作窗口

有成功：先完成同profile确认并固定接口，再按主计划M1–M5扩大范围；不立即修改成功参数或更换checkpoint。

没有成功但存在明确工程缺陷：保留首夜报告，在当前授权内登记新有限批次继续修复，旧已封存作业不延长、不复活。没有新机制证据则转已登记条件任务或公布当前边界。

模型在可信执行下仍不生成所需动作：记录冻结迁移范围；不能将训练或WBC调参悄悄纳入“链接层”。完整相机标定、异步部署、移动本体和通用Agent组合都不是本夜抓取/局部技能的前置。
