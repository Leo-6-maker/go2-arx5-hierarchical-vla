# SLA–Go2 V1.9.4：冻结 SLA-PT，直接接入 VLA＋WBC 的仿真能力试验

日期：2026-11-24，Asia/Singapore。状态：**执行计划；新接口尚待实现，本次发布没有运行模型或仿真。**

用户当前要求：“直接开始让 sla-pt 加入我们 VLA+WBC 系统，测试这个后训练的模型不经过微调只进行接口适配直接用；制定详细执行计划推送到 GitHub。”

**本轮主线：官方 SLA-PT 冻结权重 → 固定、非学习的接口适配 → 现有站立 WBC 闭环 → 用实际抓持结果确定能力边界。** 首批直接做 G00/G01 × 两个固定采样种子的四个条件格。源相机标定、源夹爪物理补测、源域 96 格面板和学习式适配均不再是该首批的前置。

配套：[实施清单](SLA_GO2_V194_DIRECT_FROZEN_RUNBOOK_20261124.md) · [机读计划](SLA_GO2_V194_DIRECT_FROZEN_REGISTRY_20261124.json)。

## 1. 要回答的问题与结论范围

问题是：在明确的名义位姿／夹爪接口、真实目标 RGB 和实际机器人状态反馈下，现成 SLA-PT 能否让现有 Go2＋ARX5 站立系统自行完成“抓起红色水果、抬升、持续保持”？

“只做接口适配”允许单位／参考系转换、官方预处理与动作逆变换、固定相机槽位路由、确定性的夹爪行程映射、满足原限值的轨迹接续和时间重采样。它不包含训练、拟合目标轨迹、语义动作修正或教师接管。

结果统一称为 **FLANGE_EQUIVALENT_NOMINAL 接口下的冻结策略仿真试验**：

- SLA、视觉／语言／动作网络、归一化统计、WBC、腿部控制、动力学均保持冻结；目标训练样本和所有优化更新为 0。
- “零更新”不等于完全未见过目标信息：G00/G01 已用于 H0 和离线诊断，接口也参考了已有目标工程信息。它们是开发条件，不是独立 TEST。
- 名义法兰对应不保证源／目标指垫接触中心等价；归一化行程比例不保证真实源夹爪毫米值等价。
- 首批检验站立操作。源双臂能力、行走操作、实时推理部署、真实硬件能力均不由这四格推出。
- 输入与轨迹接口通过但任务失败，同样是本轮有效交付；不能要求先有正结果才允许记录正式试验。

## 2. 发布基线与已知事实

本计划从当前 `main` 的 `a718abb5a18a299e116326af59ead3ccb1db4e9a` 建立文档分支，PR 目标为 `main`。执行实现与历史证据单独固定为 [PR #510](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/510) 的 `719882e9fd154ffbe995484e7a766ada3b3ecd02`。文档发布不会合并、部署或热修改该执行栈。

| 已有证据 | 对这次路线的意义 |
| --- | --- |
| [源任务 T0/T1/TL](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/719882e9fd154ffbe995484e7a766ada3b3ecd02/reports/SLA_GO2_V193/SOURCE_TASK/ACTUAL.md)：TL 八个输出有闭爪和局部抬升意图；仅来自两个真实 episode 的四个锚点 | 有理由直接试闭环；不是完整源任务成功率 |
| [目标 T2](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/719882e9fd154ffbe995484e7a766ada3b3ecd02/reports/SLA_GO2_V193/TARGET_T2/ACTUAL.md)：before-close 四个输出均未闭爪 | 冻结模型在目标输入上已有负证据；闭环不能预设成功 |
| [T3 交换诊断](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/719882e9fd154ffbe995484e7a766ada3b3ecd02/reports/SLA_GO2_V193/TARGET_T3/ACTUAL.md)：更换视觉／状态会改变输出 | 可支持后续归因，不能直接证明相机外参是唯一原因 |
| [紧凑 H0](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/719882e9fd154ffbe995484e7a766ada3b3ecd02/reports/SLA_GO2_V193/GRIP_H0/COMPACT_H0_ACTUAL.md)：G00/G01 均完成原生抓持，分别 27.906 / 28.406 s 总仿真时间 | 相同目标场景／控制链已有可执行对照；精确绑定一致时复用，不机械重跑 |
| [源坐标与 P10](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/719882e9fd154ffbe995484e7a766ada3b3ecd02/reports/SLA_GO2_V193/FRAME_P10/ACTUAL.md)：3,914 个源观测核实 link6 定义；H0 的 630 个 50 ms atom 有消费证据 | 坐标约定和旧消费链可复用；尚不是 PT 闭环资格 |
| [严格物理工具分支](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/719882e9fd154ffbe995484e7a766ada3b3ecd02/reports/SLA_GO2_V193/PHYSICAL_TOOL/ACTUAL.md)：源实测工具点／开度标定仍缺失 | 保留该事实；另立名义仿真接口，不伪造实测表来通过原分支 |

[原 V1.9.3 计划](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/719882e9fd154ffbe995484e7a766ada3b3ecd02/docs/protocols/SLA_GO2_V193_THREE_VIEW_FIRST_CLOSED_LOOP_20261124.md) 中“所有 PT 接触执行必须先有源实测工具标定”和离线意图通过的未来排队要求，仅在本新名义仿真 profile 内由本计划取代。原严格物理 profile、旧 STOP、已封存分母和真实原生安全门保持原定义。

## 3. 冻结什么

### 3.1 模型与官方变换

| 项目 | 冻结身份 |
| --- | --- |
| 模型 | `haofuly/SLA_benchmark_ckpts/SLA_Memory_robodojo_real_ckpt` |
| HF revision | `7f6880569c3be777291e646fad1b4951245e8553` |
| XPolicyLab | `c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454` |
| 有效配置 | `train_robodojo_real`；`extra_frames_first=false`；attention `xla` |
| 输出 | 原始 50×32；只经官方逆变换得到 50×16 |
| 外部状态／动作顺序 | 左臂 8 维，右臂 8 维；每臂 `xyz,qw,qx,qy,qz,g` |
| 源动作时间 | 第 i 行为 i/30 s，i=0…49；末行 49/30 s |
| 统计与 tokenizer | 复用[已核实 manifest](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/719882e9fd154ffbe995484e7a766ada3b3ecd02/reports/SLA_GO2_V193/SOURCE_TASK/REGISTRATION.json)，精确哈希见配套 JSON |

禁止额外反归一化、重估目标统计、换 checkpoint、改变 sampler 步数／精度／attention 配方，或手动把四元数分量当欧拉角相减。加载一次已绑定的模型服务，完成一个批次；不每次 query 重载模型。

固定虚拟左臂沿用当前 A1 向量及其原始哈希；实际只消费右臂 8 维。完整双臂预测均保存。不得按场景自动交换左右臂、把左臂的更好预测转给右臂，或虚构会动的第二只机械臂。

### 3.2 固定三视图输入

首批使用已有 `THREE_VIEW_V1`，不开展新相机安装搜索：

| 物理输入 | 原始形状 | 官方输入槽 |
| --- | --- | --- |
| body 当前及历史 | 每帧 256×256×3，RGB uint8 | `cam_high`：10 帧 |
| wrist 当前 | 256×256×3，RGB uint8 | `cam_right_wrist` |
| aux V1 当前 | 480×640×3，RGB uint8 | `cam_left_wrist` |

V1 固定在目标 `base_link`：位置 [0.20, 0.22, 0.42] m，固定视点 [0.42, 0, 0.31] m，FOVY 75°。这是已存在的目标采集安装，不是恢复源录制台相机标定。body/wrist 保持实际运行时配置，沿用官方 processor 的图像处理，不额外上采样成“与源原图一样”。

头部序列是 `[initial, t−8s, …, t−1s, current]`；目标 2 ms 时钟上对应 `[0, max(0,t−4000), …, max(0,t−500), t]`。仅早期不足历史按已有 initial-frame 规则重复；缺失真实应有帧不得用当前图、源录像或未来图补齐。三路当前图来自同一实际 packet；跨 episode 清空历史。

源相机内外参继续记 `UNKNOWN`，不制造 K／外参输入。保留目标安装与时间戳用于复现。已有 aux V2 的历史负结果保留，但不把 V1/V2 对比放在首批之前，也不在首批中途换视角。

实现来源：[ThreeViewStream](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/719882e9fd154ffbe995484e7a766ada3b3ecd02/scripts/sla_go2_v19/sea_three_view.py)、[SensorObservation](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/719882e9fd154ffbe995484e7a766ada3b3ecd02/scripts/sla_go2_v19/sea_observation.py)。

## 4. 唯一主接口：FLANGE_EQUIVALENT_NOMINAL

### 4.1 位姿双向对应

沿用已验证源零位平移，单位 m：

```text
z_source = [0.09519999999999992,
            0.001000411398944587,
            0.15649999265207865]

T_WV = T_W_target_armbase · Trans(z_source)
T_V_link6_observed = inv(T_WV) · T_W_target_TCP · inv(T_target_link6_TCP)
T_W_TCP_command   = T_WV · T_V_link6_predicted · T_target_link6_TCP
```

`T_target_link6_TCP` 取目标已有安装几何；`T_WV` 与发起查询的实际状态绑定。源 xyz 已经是零位平移扣除后的 link6 坐标，不再额外加一次偏置。旋转用 SE(3)／SO(3) 正确组合，保持官方 wxyz 接口。

该规则对应法兰。源到真实接触中心的未知偏置没有被“估计为零并声称标定”；它是本试验明确承受的一项名义假设。不得根据水果坐标重新平移／旋转输出以使其靠近抓点。

### 4.2 夹爪：明确区分模型码、目标行程和 wire 比例

```text
policy command g ∈ [0,1]；0=工作行程闭端，1=工作行程开端
single-finger command w = 0.002 + 0.040*g        [m]
observed g = ((w_left_actual+w_right_actual)/2 − 0.002)/0.040
receiver fraction f = w/0.044
target measured pad gap = 0.001048+w_left_actual+w_right_actual [m]
```

`EEF10[9]` 与接收器收到的是 f，不能直接塞 g。测量反馈用真实双指位置，不能回填期望命令。当前 `ObservationStream` 的旧输入编码是 `mean(slides)/.044`，必须在新 profile 内替换成上述逆映射。

**观测域与命令域分开。** g 命令超出 [0,1] 则拒绝；允许最大 1e−6 的浮点 roundoff 归并端点并留痕。实际双指位置先经过原生机械／状态有效性检查，然后按公式如实编码。跟踪误差可能使 observed g 略超 [0,1]，不得把真实状态裁到命令端点；记录其越界程度作为输入域偏移。旧 stream 的输入区间断言不能未经区分地照搬。非有限或原生判定非法的测量仍然拒绝。

不使用两段源录像的最大 g≈0.8903 作重新归一化，不使用未经本源录制核实的 145 mm 工具长或 88 mm 开口作为“实测真值”，不按是否抓到物体选择斜率。

### 4.3 允许的动作接续与禁止的帮助

保留模型原始 50 行，按固定规则建立完整曲线，再只消费本轮实际取得的前缀；剩余部分是未执行预测。位置插值和 SO(3) 插值保持原顺序。限制只来自原接收器速度／加速度、接缝连续性、机械范围和安全门。

首点接续从最后实际消费的 EEF／夹爪 **command** 开始，到模型预测的首点；测量用于模型反馈、跟踪与安全，不替换命令连续起点。夹爪首点变化标记为 `MODEL_FIRST_KNOT`，必须与 EEF 接续共用明确时间轴。可以确定性延长接续时间以满足原 40 mm/s 单指限制，不可改变目标开度、删除闭爪事件或额外维持抓握。

编译参数固定为 `bounded=False, protect_all=False, cruise_from_width_m=w_prev`；anchor 为最后 consumed EEF，incoming 平移／角速度由最后两个连续 consumed EEF 和 50 ms 间隔求得。保留完整原始 knot 几何；不再叠加旧 `world_offset` 对整条曲线重锚。`Compiler.at(t)` 返回的 source phase 单位是秒，`SourceCurve.grip` 入参是行索引；两者转换为 `source_index=30*source_time_s`，优先直接使用同次 `Compiler.at(t)` 返回的 g。

旧 `gripper_alignment.py` 对 contact-at-handoff 和 0.55/0.65 crossing 的拒绝属于旧上游审计语义。新 profile 要把模型首点变化明确归因并做有界接续，不以删掉真实接收器检查的方式放行；原函数和严格 profile 不改。若模型自然张爪、失抓或触发原生 contact STOP，如实记失败，不能用 hold clamp 修好它。

不得注入物体 GT 位姿、教师阶段、抓取点、专家未来动作、定时闭爪、目标投影或按物体位置选 chunk。评分器和安全监控可以使用原有全状态；这些数据不得进入模型输入或动作生成。新 `mobile_contact_stage` 若运行时 API 需要，只能表示由原始命令／实际机械状态确定的安全通道状态，不能恢复 `teacher.decide/commit`。

## 5. 执行时钟：先做同步仿真闭环

首批选可审计的同步模式：物理 owner 在真实暂停边界取图和状态，等待模型返回，安装后再推进。推理／通信墙钟与物理时间分别记录；已有 wall-clock watchdog、owner 锁和 STOP 不暂停。该结果不宣称实时部署。

| 时钟／参数 | 固定值与含义 |
| --- | --- |
| 原生积分 | 2 ms |
| 消费 atom | 25 tick = 50 ms |
| P10 窗口 | 10 atom = 500 ms；不是十个 30 Hz 源行 |
| EEF 预览 | 41 个 50 ms knot，按现有 H40 接口下发 |
| EEF 引用刷新 | 每个 P10 的 atom 0 与 atom 7，即 0 / 350 ms |
| 引用 TTL | 400 ms，沿用原因果时钟和有效性检查 |
| 模型新查询 | 只在 P10 边界、上一提案已满足下述实际消费条件时发起 |
| 源动作时间 | i/30 s；不拿 50/30 s 当整个目标动作的超时 |

**固定刷新规则：**

1. 首个查询来自稳定站立入口的实际观测；没有教师抓取或教师搬运前缀。
2. 编译整条 50 行曲线，区分 BRAKE、ALIGN、SOURCE、HOLD；依据现有冻结限值重定时。所选路径不按任务成功挑段。
3. 同一提案至少实际消费 **10 个具有正 source-phase 增量的 50 ms atom**，或已经实际消费到该提案源曲线末端，才在下一 P10 边界发起新查询。计数只能在同 proposal 的 EEF 消费和 gripper ACK 已证实后增加，且 atom 起点位于 lead 之后、`source_time_end-source_time_start>1e-12 s`；末端为 49/30 s，数值容差 1e-9 s。不能用预测时间代替消费证据。模型输出恒定位姿时，源位相仍可推进；不额外要求 TCP 先移动 1 mm，避免阻断仅夹爪变化或保持。
4. ALIGN 期间继续原提案，不每 500 ms 重置接续。P10 引用刷新继续运行，不能把重新发布误记为新模型查询。
5. 新查询前保存实际消费索引、参考及测量；新提案接到当前已消费边界。上一提案未执行后缀废弃并标记，不能记作执行监督。
6. 无 source 进展、正常跟踪延迟、模型持续空动作或动作拒绝均按原预算／安全规则结案；不无限延长等待，不强行推进到“看起来会抓”的行。

新鲜反馈依据新 native tick 与消费证据，静止图像可以哈希不变。调用上限在下一次新 query 前检查：第 120 次返回的已接纳提案仍按既定规则消费到下一正常查询边界，届时需要第 121 次调用则以预算终止。每个 grant 前先检查 `tick+25<=50000`，余量不足一个 atom 就在最后完整边界结束；不为凑满 100 s 发起半段动作。

复用全 50 行 `Compiler/SourceCurve`；不要把最多前 15 个源 interval、固定张爪的旧 `ReferenceSegment/SEAStandingStudent` 冒充这里的实现。P10 消费证据是已验证基底，**本新查询／替换规则仍须实现并在首批原生轨迹中验证**。

## 6. 从实现到四格闭环的执行顺序

| 阶段 | 要完成的动作 | 结束条件 |
| --- | --- | --- |
| I0 固定输入身份 | 固定 #510 执行来源、现场依赖与资产、模型 manifest、G00/G01 reset、A1 左状态、receiver binary；复用可信缓存 | 身份可重现；缺少现场路径／二进制绑定时只补该项 |
| I1 补齐四个接口接缝 | 名义 codec；真实三视图常驻 worker；可变夹爪 reference；原生 PT driver | 没有黑第三视图、旧夹爪编码、teacher 决策或固定开爪残留 |
| I2 必要资格 | 仅检查改变的坐标／夹爪往返、真实历史、全 50 行变换、P10 消费接缝和原 STOP；复用精确匹配的 H0 | 受影响 CPU／archive 检查通过；不要求源能力面板通过 |
| C00 首个闭环 | G00、seed 0；真实模型→安装→消费→实际观测→再次查询 | 正常则在同一个 episode 继续到成功／失败／有限终点；不另起“更好看”的首跑 |
| C01/C10/C11 | 依次 G00 seed 1、G01 seed 0、G01 seed 1 | 完成四个注册条件格或明确记录批次不完整 |
| R0 读出 | 对齐预测、安装、消费、实际机器人和物体结果 | 四格表、全 attempt 表、失败位置与下一条有证据的建议 |

若 C00 只是有效科学失败，仍完成其余可安全执行的注册条件。原生硬安全 STOP 则先终止相关 owner、保留日志并解释该门；没有排除共享安全缺陷前不启动下一张卡。此时报告未执行格，不能把它们计成成功或已完成试验。

阶段 I2 不再创建源域 96 格查询、相机优化、完整标定恢复、异步执行框架或多层审批账本。新 worker 必要的已知输入一致性前向最多 2 次、单列资格用途；若已有可信身份足够可为 0。不得把资格输出挑选为正式条件的“最佳噪声”。

## 7. 首批注册：四格、一个完整指令、固定预算

场景固定 `FRUIT_COMPACT_STANCE_01`，冻结原 table、red/blue 两物体、动力学、实际站立入口和评分。

主指令固定为：

> Pick up the red fruit, lift it above the table and keep holding it.

旧 TL 指令是 “Pick up the fruit, lift it above the table and keep holding it.”。本次增加 red 以对应原 `HOLD-red` 评分与双物体场景；这是事前固定的新 prompt，不把旧 TL 当成此 prompt 已完成验证，不进行 prompt sweep。

| 顺序／条件 | 布局 | episode seed | 首次查询噪声 key data |
| --- | --- | ---: | --- |
| C00 | G00 | 0 | [0,0] |
| C01 | G00 | 1 | [0,1] |
| C10 | G01 | 0 | [0,0] |
| C11 | G01 | 1 | [0,1] |

第 q 次新查询调用现有 `rng_noise([q, episode_seed])`；其 JAX split、sample key 和 (1,50,32) float32 噪声保持既有算法。保存每次实际噪声哈希。各布局共享对应的查询序号噪声，不能挑 seed 或因失败重复相同条件。q 从 0 到最多 119；技术性原样重试从同一 reset 和 q=0 开始。

**每个 attempt：**

- 最多 120 次新模型查询；
- 从 reset 起最多 50,000 个 2 ms tick，即 100 s 实际仿真时间，包含站立初始化与任务执行；
- 最多 1,800 s attempt 墙钟；模型服务预加载单列，不混作仿真时间；
- 遇任务成功、原安全 STOP、有效动作拒绝或按第 5 节定义将超出下一次 query／atom 预算时结束；墙钟上限按原 watchdog 终止。
- 沿用原 receiver 的 50,000 tick cap，不在看到失败后延长。

四个正式条件最多 480 次查询／200,000 tick／400 s 仿真。明确的基础设施无效允许每格最多一次原样重试、全阶段共最多两次；这个总数不随修复版本重置。因此首批含技术重试最多 6 个 attempt、720 次查询、300,000 tick、600 s 仿真。另有最多 2 次无物理资格前向，均记录实际使用。

这些是本批次有限终点，不是新设的人工作业许可或模型无限时能力定义。普通工程修复沿用 Goal V7 自主执行；若确需新实现版本，先冻结该版本再建立新的同四格批次，不把不同版本结果拼成一个成功率。

## 8. 成功与失败怎么判

主要判据复用 [StationaryHoldScore](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/719882e9fd154ffbe995484e7a766ada3b3ecd02/scripts/sla_go2_v19/sea_fruit_h0_score.py)：选中 red 物体的原生抓握、实际抬升至少 20 mm、连续保持至少 1 s，无错误物体抓握／接触、非法接触、时间缺口或安全违规。原站立约束为 base 平移 ≤20 mm、旋转 ≤5°、脚位移 ≤5 mm、四足支持；其余原安全门保持原值。

评分使用实际连续 2 ms 物理记录。抬升量是物体的实际位移，不是法兰预测上升；保持时长从真实满足原条件的时刻计，不是“输出闭爪的预测时长”。没有教师接管才能记 PT 任务成功。

同时报告三层事实：

1. **输入／推理**：真实三视图和历史进入模型；新鲜反馈查询数、原始 close/open 意图、左右臂分配。
2. **动作／执行**：解码、编译、安装、真实消费、source phase 与测量跟踪；每项拒绝原因。
3. **任务**：是否抓到 red、最大物体抬升、最长连续保持、是否掉落／wrong-object／STOP、完整成功与结束原因。

| 事件 | 计数与解释 |
| --- | --- |
| 正确输入下不靠近、不闭爪、提前松开、抓错物体、抬升／保持不足 | 有效科学失败，保留在计划四格中 |
| 模型输出超域、不可达或被固定安全门拒绝 | 该策略＋接口组合的有效边界失败；不剔除来抬高成功率 |
| 正常重定时／跟踪等待导致有限预算耗尽 | 有效的 budget-limited 结果，不声称无限时也不能成功 |
| 可独立证明的模型未加载、传输中断、历史包损坏等基础设施故障 | technical-invalid；保留所有已发生效果和 attempt，可按原样重试规则补格 |
| 根因还不清楚、动作效果未知 | unresolved；先查既有 run／消费记录，不盲目重发 |

主报告先给 **成功数／4 个计划条件**，同时给有效科学结果数及成功数／有效科学结果数、technical-invalid、unresolved、未启动格数。若只有 3 格有效，不能只报一个没有缺失说明的成功率。四格不计算成四个独立任务，不据此声称稳健总体成功率。

## 9. 根据结果走哪条分支

| 首批结果 | 下一步 |
| --- | --- |
| 接口 bug 有独立证据 | 修正对应代码；原 attempt 保留；影响输入／输出语义则新版本重做同四格，最多两个有证据的后继修复批次 |
| 0/4 成功且接口有效 | 交付固定 profile 下的负结果；用已有日志定位最早差异。没有新机制证据，不自动扫相机、源标定、prompt 或微调 |
| 有成功但仅限一个布局／seed | 交付局部可行性；可准备一个新的近邻场景确认批次，不声称泛化已通过 |
| G00/G01 各至少一个 seed 完整成功 | 进入“未执行场景确认 → 完整 pick-and-place → 第二技能”的后继顺序；该门只用于研究路由，不是统计可靠性保证 |

本阶段整体最多三个接口版本 × 四格，外加全阶段两次技术重试，即 14 个 student attempt；对应最多 1,680 次 episode 查询、700,000 tick。不是要求用满。资格前向每版本最多两次，另记。达到本阶段终点后整理结果；不据两次工程失败永久封禁整个研究方向，也不自动开展未注册的大扫描。

后继每一批在运行前固定场景、prompt、seed、评分和有限预算。优先既有目标执行器已资格的场景；新增放置／释放只补相应执行资格。整段 pick-and-place 必须由一条完整初始指令驱动，不能在抓住后由外部教师改成“现在放下”。相机对照或严格物理标定仅在日志指出相应瓶颈时成为独立假设实验；任何学习适配都是另一条明确命名的非零更新路线。

## 10. 实施边界与交付

遵循 [Goal V7](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/719882e9fd154ffbe995484e7a766ada3b3ecd02/docs/protocols/GR00T_WBC_SCIENCE_GOAL_V7_20261031.md) 和 [仓库工作流](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/a718abb5a18a299e116326af59ead3ccb1db4e9a/docs/REPOSITORY_WORKFLOW.md)：普通工程问题自主修复，复用现有 owner、日志和评分；不新建预算分配器、调度器、影子账本或逐函数审批。

实际执行前核实当前资源、已有授权、唯一物理 owner、资产和站点依赖。现场 `fusion_trajectory_adapter`、`mobile_boundary_runner_v1` 等运行时文件并非全部跟踪在 #510 Git tree 内，实施者需绑定真实版本并核查 stage／hold 语义；本计划不假装已经审过缺失实现。不干预健康作业、不热更新活跃 capsule。目标自由机身隔离和实机限制不由本仿真计划解除。

第一份执行报告必须交付：四格结果表、全部 attempts、原始模型输出到消费命令的对应、实际状态／物体曲线、一个代表成功或失败视频及完整数据索引、首次失败机制、下一项有证据的动作。视频不可替代分母与逐 tick 评分。

本次 GitHub 交付是协议、实施清单和注册计划；新增模型查询、原生启动、物理 tick、训练更新、硬件动作均为 0。
