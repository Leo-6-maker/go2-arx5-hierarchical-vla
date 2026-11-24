# SLA–Go2 V1.9.4：直接冻结接入实施清单

日期：2026-11-24。对应[主协议](SLA_GO2_V194_DIRECT_FROZEN_PLAN_20261124.md)与[机读计划](SLA_GO2_V194_DIRECT_FROZEN_REGISTRY_20261124.json)。

**本文件是待实施工作清单，不是可直接启动的命令包。** 下文 `sea_pt_nominal_*.py` 均为建议新增模块；#510 尚无“真实 3V＋PT＋可变夹爪＋P10”完整入口。不可把旧 reach／T2 命令改个 run 名称就执行本计划。

## 1. 实施者先固定的来源

- 文档出版基线：`main@a718abb5a18a299e116326af59ead3ccb1db4e9a`。
- 可复用执行快照：`719882e9fd154ffbe995484e7a766ada3b3ecd02`，PR #510。
- 在独立 checkout／worktree 准备应用改动；不切换或热补丁运行中的 capsule。
- 模型、manifest、有效配置和统计使用配套 JSON 的精确身份；已验证的不变大文件可复用可信校验，变更或跨边界传输才重验对应成员。
- 先只读核实当前 Core owner／GPU 使用和现有授权；复用现有 lease／日志／关闭机制。不要新建同功能调度器、额度分配器或并行物理 owner。
- 站点模型解释器可参考现有 `$SITE/vendor/XPolicyLab/policy/SLA_Memory/openpi/.venv/bin/python`；`SITE`、本地资产与环境必须从现场真实绑定获得，不能把示例写成已存在路径。
- `fusion_trajectory_adapter`、`mobile_boundary_runner_v1` 等依赖在该 Git tree 中不完整。记录现场源码／二进制 hash、版本和实际 API，针对本次调用核查即可，不扫描整个历史平台。

开始一次新学生 batch 前，在现有运行 manifest 内补齐以下实际值。配套 JSON 中的 null 是待绑定状态，不是通配许可：

| 现场字段 | 如何取得 |
| --- | --- |
| implementation SHA／changed-file hashes | 实施提交和独立执行目录的真实内容 |
| scene／reset／robot／WBC／legs／receiver bindings | 对应紧凑 H0 已封存 manifest；严格逐项比对本次使用值 |
| A1 virtual-left vector／hash | 复用 #510 的同一观测配置，禁止手填近似数 |
| target link6→TCP transform | 已有机器人安装模型，核对旋转顺序和 SI 单位 |
| runtime adapters／native safety identities | 当前现场文件与已资格 binary，不能仅填 Python import 名 |
| owner／resource／output paths | 当前合法资源和唯一路径；日志可持久保存；确认没有活跃同资源 actor |
| compiled nominal profile／noise schedule | 本计划参数加实施哈希，首次正式输出前封存 |

## 2. 最小代码工作包

路径均相对执行仓库；“建议新增”不表示已实现。允许合并相邻小模块，但不得新增一套并行控制架构。

| 工作包 | 建议改动 | 应复用的现有实现 | 必须产出的结果 |
| --- | --- | --- | --- |
| W1 观测／名义 codec | 新增 `scripts/sla_go2_v19/sea_pt_nominal_codec.py`，必要时内含 observation builder | `sea_action.py`、`sea_observation.py`、`sea_three_view.py`、`ObservationStream` 的参考系绑定 | 有实际来源的 16D state、三路当前图及九帧头部历史；双向位姿和 jaw/wire 转换 |
| W2 常驻 PT worker | 新增 `sea_pt_nominal_worker.py` | `sea1a_panel.infer/rng_noise`、`sea1b_panel.verified_processor`、`policy_query.frozen_predictor`；参考 T2 的单次 load 生命周期 | 一个冻结模型服务；每次查询输入、noise、sampler raw、官方 decoded 输出一一绑定 |
| W3 reference 接续 | 新增 `sea_pt_nominal_reference.py` | 全 50 行 `compiler.SourceCurve/Compiler`、现有 H40 gate、SO(3) 与限速工具 | EEF＋夹爪同一时轴；MODEL_FIRST_KNOT；source-phase 消费和 P10 查询准入 |
| W4 原生 driver／owner | 新增 `sea_pt_nominal_native.py`；仅在现有 owner 不能参数化时增加小型 `sea_pt_nominal_owner.py` | `sea_fruit_h0_native.py` 的真实 pause/install/consume/ACK/STOP；同一已资格 fruit receiver | 真实 PT→执行→反馈循环；四格调度；无 teacher action 依赖 |
| W5 最少资格与读出 | 新增或扩展 `sea_pt_nominal_preflight.py`、`sea_pt_nominal_readout.py` | 已有 receiver replay、`sea_compact_p10.py`、`StationaryHoldScore` 和日志 | 受影响接缝检查；四格结果及所有 attempt；原生效果、失败类别和版本边界 |

### 不能误用的旧入口

- `sea_reference.SEAStandingStudent` 固定开爪、查询上限 12、依赖实际位移门；不能直接提高 cap 后称为抓持闭环。
- `sea_reference.ReferenceSegment` 是旧短 prefix 路径，不是本计划的全 50 行曲线。
- `reach_forward.worker` 原来把 left 图填黑、绑定旧 issue/cap，并按次起进程；不能原样当作 3V 常驻服务。
- `sea_target_t2_execute.execute` 是旧八格离线身份；复用内部可重用函数，不借旧 allocation 运行物理 episode。
- `sea_fruit_h0_capture.capture(case)` 要求 H0 lease 已关闭；新 driver 不能伪造 CLOSED 或把正在运行的 student 当作归档。
- `teacher.decide`、`teacher.commit`、`StageClock` 的教师阶段推进不能进入 PT action path。
- 原 `grip_at_grant` 在 `mobile_contact_stage=="approach"` 强制 42 mm；要建立明确 PT 接入语义，不能靠换成任意字符串规避检查。

## 3. W1：实际观测与双向转换

### 3.1 图像／历史

从 owner 当前已暂停的实际 tick 取得 state packet。当前 body/wrist/aux 均从该 packet、同一 episode 生成；renderer 不得积分物理。复用目标相机的实际运行时姿态和 V1 aux 安装。

历史按真实 native journal 精确取 `0,t−4000,…,t−500,t`，负索引按已有规则映射到 0。可在 query 时非积分重渲染必要的真实历史 packet，并在 episode 内缓存；不必为每个 2 ms tick 预先渲染三路 RGB，也不能因为只缓存 query tick 就把“第几次查询”当作秒。

每帧记录 episode、tick、物理 camera、RGB hash、源 packet hash。initial 是真实 reset 帧，不是老师已到水果上方的帧。缺少本应存在的历史 packet 就定位采集问题，不复制当前图补成“完整记忆”。

新鲜观测以实际时间戳和已消费证据定义；静止状态可能产生相同图像／输入哈希，不能仅凭哈希未变化拒绝正常 hold 或 jaw-only 动作。跨 episode、未来帧、错 packet 则拒绝。

### 3.2 位姿／夹爪

实现主协议第 4 节公式。按一组真实、非退化旋转样本验证：

- actual target link6／TCP → source-style state → target command 的往返；
- 米／毫米和 wxyz／xyzw 只在明确边界换一次；
- 正／负 q 表示同一个旋转，不能制造 180° 跳变；
- source zero 只应用一次；query frame 与返回 output 绑定，不拿新时刻 base transform 解旧输出；
- command g 的 0／0.5／1 对应单指 2／22／42 mm，wire 是该宽度除以 44 mm；
- measured state 反映真实左右指平均值，左右差异另留痕，不把最后 command 伪装成测量。

测量机械域保持原 `gripper_state_valid`；#510 明确区分 measured [0,44] mm 与 installed [2,42] mm。G00/G01 已有开端实测约 42.00004／42.00005 mm，说明 observed g 可略大于 1。输入仍为 finite float32；照原官方 processor 处理，记录 OOD，不自行增加、删除或调节官方 normalizer 的 clipping 行为。

## 4. W2：模型返回必须来自本次真实三视图

复用已核实 checkpoint/config/stats/tokenizer 和官方 action inverse。一次 load 后保持 worker；模型服务仅返回候选，不能拥有物理安装权限。

每个 query 的必要关联是：

```text
episode / query_index / query_tick
→ sensor payload hash + processor-input hash
→ rng_noise([query_index, episode_seed]) + sample-key/noise hash
→ raw sampler 50×32 + sampler state
→ official decoded 50×16
→ complete left/right output + admitted right-arm curve
```

核实三路有效 image mask，head 历史顺序、`extra_frames_first=false` 与有效配置一致。必须在 sampler 实际收到的 tensor 处核对，不能只记录发送端 JSON。

GPU 调用异常与科学输出拒绝分开。网络返回不明先查该 query 的 worker receipt，禁止以新随机数盲重发。技术原样重试只在明确未完成原正式条件后按批次规则进行；不能在同一 query 上多抽几次挑能执行的输出。

每次推理返回的左臂动作保存作能力边界分析，主 driver 只消费右臂，不做左右选择器。

## 5. W3：完整曲线、首点夹爪和实际消费

### 5.1 共同时轴与首点接续

复用全 50 行 `Compiler`，固定 `bounded=False, protect_all=False, cruise_from_width_m=w_prev`。anchor 是最后 consumed EEF；incoming 由最后两个连续 consumed EEF 的平移差／相对旋转除以 50 ms 求得。按原安全限值计算 lead、源曲线和预览，保留原 knot，不叠加旧 `world_offset` 的全曲线重锚。可以保留旧 44 mm 跨度的保守时间估算；实际新名义映射为 40 mm 跨度。不得为了加速而放宽 WBC 或接收器限值。

建议新函数 `model_first_knot_widths(...)` 使用：

```text
w_prev = 最后实际 ACK 并消费的 command width
w_0    = 0.002 + 0.040*g_0
h(u)   = 10u^3 − 15u^4 + 6u^5
w(t)   = w_prev + (w_0−w_prev)*h(t/lead)         ; lead 内
w(t)   = 0.002 + 0.040*g_raw(30*source_time_s)   ; lead 后，g_raw按源行索引插值
1.875*abs(w_0−w_prev)/lead ≤ 0.040 m/s           ; 保留原 0.9 余量
```

`Compiler(..., cruise_from_width_m=w_prev)` 可计算保守 lead，但还必须真正包装 lead 内的宽度；不能仅延长时间却立即把命令跳成 g₀。

`Compiler.at(t)` 第三个返回是 source 秒数，不是源行索引。优先直接消费同次返回的 EEF、g、source_time_s；若调用 `SourceCurve.grip`，需传 `30*source_time_s`。终点是 49/30 s；禁止在接口中引入 30 倍的时间／索引错误。

新首点绑定 query ID、输入／decoded hash、row 0、g₀、w₀、w_prev、lead 和 peak rate。不得伪造 `contact=False` 来绕过旧 `alignment_widths` 的拒绝。新函数明确承认模型首点要求开／闭；真实 contact 和 STOP 门照常运行。

命令 seam 以最后 consumed **command** 为起点，实测宽度用于观测和安全，不能用测量跳过 command 连续性。原生 gripper epoch／sequence 从站立初始化延续；不在每次 query 重开 epoch。若确需新 epoch，其首命令必须满足原 `sequence==0` 且与现有 targets 一致的协议。

SO(3) 采用已有 Slerp／RotationSpline；速度用相对旋转。逐实际 50 ms grant 再检查位姿／夹爪差分，不能只凭解析曲线的峰值检查。

### 5.2 P10 的固定语义

实现需要分别维护：

- `policy_query_index`：真正调用模型的次数；
- `p10_window_index`：每 10 个 50 ms atom 的窗口；
- `proposal_id` 与 `proposal_elapsed_sim_s`；
- `source_phase`、本提案实际消费的 SOURCE atom 数；
- 最后 accepted wire、installed trajectory、精确 consumed boundary、gripper ACK/sequence。

P10 的 atom 0/7 进行 EEF 引用 install/reissue；每 atom 都发与已安装 H40 对应的夹爪 grant，25 个真实 tick 后查 ACK 和实际指位。沿用 `action_EEF10[row][9]` 的现役 grant 语义，不自行换成 row+1。

同 proposal 消费至少 10 个正 source-phase atom 或已消费到源曲线终点，才在 P10 边界进行新查询。计数从真实 EEF receipt 与 gripper ACK 更新：atom 起点≥lead，且同 proposal 的 `source_time_end-source_time_start>1e-12 s` 才加一；源终点 49/30 s 的判断容差为 1e-9 s。其他窗口只继续旧曲线和引用刷新。ALIGN 中的首点开／闭是真实模型来源动作，但不记为 SOURCE atom。完全恒定的源轨迹仍按自身位相推进，允许下一次真实反馈查询。

### 5.3 原生接触规则保持原状

已核对的 fruit receiver 要求有有效 command、非全开工作端、pad 对 selected red 的接触、穿透不深于 4 mm；其他原接触／力矩／支撑／STOP 门均保持。

允许模型自然开爪并失抓。如果 opening 达 42 mm 时 pad 仍接触，原接触门会 STOP，记录这一边界；不得关闭该门，也不得给模型增加保握 clamp。

现场 runtime adapter 的 teacher-stage 依赖需在实施时核实。新 PT profile 必须继续经过同一 receiver 和所有原门，不能用“模型输出”标签免除验证。

## 6. W4：单个 episode 的逻辑

以下是设计伪代码，**不是已有 CLI 或已完成实现**：

```text
核实本次既有授权、唯一 owner、绑定与冻结参数
在 attempt 计时前完成冻结 PT worker 预加载，单列其实际资源使用
启动原受控站立初始化，保持原生接收器和 sequence
记录从 reset 开始的实际 journal；完成原稳定站立入口
确认已预加载 worker 的身份与可用状态

while 尚未触发终止条件:
    在真实暂停的 P10 边界：
        若还没有 proposal，或其 SOURCE 实际消费条件已满足：
            先检查 q<120、tick+25<=50000 以及墙钟余量，否则预算终止
            由实际 packet/journal 构造 nominal 3V observation
            query(q, fixed_noise(q, seed))
            核对输入/返回身份；官方 inverse；域与原安全检查
            在最后 consumed command 边界编译新曲线，保存未执行旧后缀
            q += 1
    for atom in 0..9:
        先检查 tick+25<=50000 及墙钟终止条件；不足一个atom则预算结束
        按同一 proposal 的真实 elapsed sim time 生成 preview
        在 atom 0/7 重发原 H40 引用
        用原 grant 语义发当前夹爪 command
        原 owner 积分25 tick；读取实际 ACK/consumed boundary/测量
        更新source phase计数与连续2ms评分
        触发成功、STOP或上限时即终止当前窗口

在原独立HOLD/STOP路径上关闭；确认最终tick与paused状态
保存outcome、全效果计数、未决状态和评分
```

从稳定站立入口前的初始化动作不记成 PT 输出，但计入本 attempt 的物理 tick 和总时间。不得从 H0 的 before-close／after-grasp 锚点启动正式四格；正式条件必须让模型自己完成接近和抓取。

第 120 次返回的提案按同一消费规则执行到下一正常查询边界；需要第 121 次调用时才以 query budget 结束，不把最后一次已取得输出无故丢掉。tick cap 在 grant 前检查；若入口 tick 不与 50,000 同余，结束在最后完整 atom（例如 49,978），不能为凑整触发接收器半段 STOP。若另一个真实 STOP／故障造成部分消费，保留其实际 tick 与原因，不能仅凭 ACK 不完整就划成 technical-invalid。

查询过程中仿真暂停不代表所有 watchdog 暂停；若发生 wall-clock timeout、租约失效或未知命令效果，用既有 STOP／查证路径处理，不能延长引用 TTL 或伪造 heartbeat。

## 7. I2 只验证真实改变的接缝

| 检查 | 最小有意义覆盖 | 通过后的行动 |
| --- | --- | --- |
| codec 往返 | 真实 link6/TCP、非平凡旋转、零位只应用一次、命令端点／wire、实际开端轻微越界 | 不再补源工具实测表 |
| 三视图与历史 | 同 packet 当前三图；initial+八秒历史；错 episode／未来／缺帧拒绝；实际 sampler mask | 不做全相机搜索 |
| 全曲线与夹爪 | jaw-only、恒定 hold、模型首点跨 0.55/0.65、接触时模型明确小幅开／闭、超域 raw 拒绝 | 进入可变夹爪原生 driver |
| 命令接缝 | 首命令 epoch、跨 query 的 last-consumed 起点、同一时轴、离散40mm/s、EEF速度／加速度 | 不靠降低WBC要求通过 |
| 原接收器／STOP | 复用已封存回放；只增加受新接续影响的合法命令与42mm仍接触应STOP等样例 | 保留同一原生binary |
| 消费时钟 | 0/350ms reissue、400ms TTL、50ms/2ms ACK、ALIGN不触发重复query、SOURCE计数 | 首格原生运行确认真实链条 |

CPU synthetic/replay 只证明接口行为，不计模型成功或原生积分。已有精确匹配 H0 不重跑；若机器人、场景、receiver、WBC 或安全版本不匹配，则该现场配置不是本计划已资格配置，应恢复匹配或单列变更资格，不悄悄混用。

必要模型一致性前向 ≤2 次且无物理安装。随后直接 C00，正常则延续同一 episode，再完成其余三格。不得在这些检查之后另起一轮 96 格源域筛选。

## 8. 注册、重试、修复

首版固定顺序 C00/G00/0 → C01/G00/1 → C10/G01/0 → C11/G01/1。每个正式条件一个 episode；最大 120 新 query、50,000 native tick（包括初始化）、1,800 s attempt wall time。按第一次触发的成功／失败／上限结案。

- 科学失败不原样重跑挑结果；模型拒绝也保留在计划四格。
- 技术无效必须有独立根因证据；同格最多一次原样重试，全阶段最多两次，所有尝试留表。
- 不确定写／GPU query／native launch 效果先查原身份；不能“超时即当没发生”。
- 影响输入或动作语义的 bug 修复产生新 profile version，并重新固定同四格；不同版本分开报告。
- 此计划最多两个有证据的后继修复批次。若同症状无新机制证据，转保存状态诊断，不通过不断加版本盲跑。
- 普通路径、类型、序列化、时钟与所有权问题按 Goal V7 自主修复；这些数量不构成逐项人工许可流程。
- 新相机、prompt、映射斜率、目标抓点或 sampler 配方选择属于新实验，不得混入“修接口”。

## 9. 最小产物与读出模板

复用现有 manifest、数组／日志格式和视频工具，避免建立新的多层 receipt schema。可在已有记录中新增下面的必要字段。

### 9.1 公共结果表

| cell | profile版本 | valid状态 | task_success | queries | consumed SOURCE atoms | sim_s含初始化 | wall_s | 最大物体lift_mm | 最长hold_s | 结束原因／首失败 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C00 | 待执行 | NOT_RUN | — | 0 | 0 | 0 | 0 | — | — | — |
| C01 | 待执行 | NOT_RUN | — | 0 | 0 | 0 | 0 | — | — | — |
| C10 | 待执行 | NOT_RUN | — | 0 | 0 | 0 | 0 | — | — | — |
| C11 | 待执行 | NOT_RUN | — | 0 | 0 | 0 | 0 | — | — | — |

另列所有 attempt 的身份、是否技术重试、原始故障、实际已发生 query/tick 和终态；不删除坏 attempt。

### 9.2 能够回答“到底卡在哪里”的单条时轴

至少关联：

```text
query observation／noise
→ raw 50×32
→ decoded 50×16（左右均保留）
→ nominal world TCP＋g/w/f
→ BRAKE/ALIGN/SOURCE/HOLD及source phase
→ accepted H40／wire
→ solver target echo／精确 consumed knot
→ gripper ACK／actual 双指
→ measured TCP／物体lift／hold／STOP
```

保留已有证据边界：installed/echo/consumed endpoint 不等于已导出低层 controller-selected unique policy ID；若现场未提供该字段，不虚构它。

把最早差异归为 observation、decoded policy、frame/jaw mapping、compile/clock、receiver/consumption、tracking/contact 或 task outcome。多个阶段可同时存在，不凭一张失败图直接宣布“相机标定问题”。

### 9.3 批次摘要

必须给出：计划4格、成功数、有效科学结果数、科学失败数、technical-invalid／unresolved／未启动格、总 attempts、实际模型调用、物理 tick、各模型与接口版本。主表给成功/4，同时给成功/有效科学结果及缺失说明。

视频选一个有代表性的完整失败或成功，并链接全部尝试；不能只剪模型最接近物体的几秒。原始场景／模型大资产和私有站点路径不公开入 GitHub；公共报告存可公开指标和引用，详细日志按已有合规路径存放。

## 10. 交付完成的含义

本实施工作完成后，应能够说：

> 在固定官方权重、固定名义接口和冻结站立 WBC 下，四个注册条件中的 X/4 完整成功；其余条件分别止于明确的模型／接口／执行／任务边界，全部过程可以从预测追到实际消费。

如果结果是 0/4，这仍是完成了首批能力试验。只有模型加载、CPU检查或几张闭爪预测，不能记为已完成该执行目标。

本次发布仅创建计划文件；以上代码任务、实际资格和四格运行状态全部为待实施／NOT_RUN。
