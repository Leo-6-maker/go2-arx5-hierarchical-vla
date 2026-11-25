# SLA 成功执行 V2：执行、归因与评分合同

日期：2026-11-26（Asia/Singapore）。配套：[主计划](SLA_SUCCESS_EXECUTION_AUDIT_V2_PLAN_20261126.md) · [推进表](SLA_SUCCESS_EXECUTION_AUDIT_V2_REGISTRY_20261126.json)。

这是现有执行端的工作指令。尚未绑定的 runtime／scene／权重／scorer 字段不是可运行 capsule；执行端按实际原件一次性补齐。不要另造调度器、审批账本或替代评分器。

## 1. 现场接手顺序

1. 只读查明自有 run、作业、GPU／world owner 和旧租约终态。不要以旧报告代替实时状态，不启动状态未知作业的副本。
2. 从用户给定目录读取 ACTUAL 和 A/B JSON，再按其中的引用定位原件；记录完整 commit、dirty patch、依赖和实际入口。路径不在本环境时如实记录，不伪造读回。
3. 将本次审计引用的每份证据标成 `ARTIFACT_VERIFIED`、`USER_REPORTED` 或 `UNKNOWN`。从汇总推导的结论单独标 `INFERENCE`。
4. 比较实际 L0/L1 的 decoder、frame、seam、时间、STOP 和 WBC 调用改动；凡与 #516 计划不同处，以实际执行身份为事实，评估其是否改变了科学因素。
5. 按原 evaluator 对齐六对 A 的阶段事件与完整终态，生成全部 PT query／消费表。
6. 选择一个最高信息量的修复或明确无需修复；先重放已保存数值，再做受影响的必要原生资格与有限开发面板。
7. 发布小型结果与引用；大日志、图片和模型留在既有授权路径。关闭实际作业／资源并报告终态。

## 2. 从 SLA 到冻结 WBC 的输入输出

### 2.1 MAIN3.8K 的已核实 provider 合同

以下来自固定 provider／codec 源码，不冒充 d7121cc1 的未读回实现：

```python
predict(
    state10, images, prompt, T_policy_from_mount,
    *,
    noise_id,
    previous_reference,
)
```

| 层 | 输入／输出 | 需要保持的含义 |
| --- | --- | --- |
| 实测观测 | state10；body 与 wrist RGB；语言；episode 固定坐标变换 | 实测 EEF／夹爪与 command 历史不同，不用物体 GT 作学生状态 |
| 模型状态编码 | xyz、SO(3) rotvec、实际 grip 共 7 维，归一化后 pad32 | episode-fixed mount frame；来自同一真实观测 |
| 模型输出 | 40×32；7 个有效量为 delta xyz、delta SO(3)、gripper logit | 每行相对同一个 query 观测锚点，不逐行累积 |
| 原物理解码 | 40×10：xyz3、row-rotation6、nominal grip1 | inverse quantile；grip sigmoid 一次；原 command 语义 |
| 执行内核输入 | 完整 decoded intent、source times、frame/tool/type、最后真正消费的参考与边界速度 | 不以已发未消费或接触后的实际指宽替代上一 command |
| 执行内核输出 | 可预览的连续整段参考与实际 rolling packets；对应原 WBC ABI | 实际 packet 字段、BASE／EEF 路由和时钟从现场绑定；不是神经网络直接输出力矩 |
| WBC／物理输出 | 实际关节、base、TCP、指垫、足端与接触 | 原 OCS2/MPC、arm PD、腿链及安全均冻结，实际二进制／参数身份需要一致 |

位姿的原还原式为：

$$
p_k=p_{\rm obs}+\Delta p_k,\qquad
R_k=\exp([\Delta r_k]_\times)R_{\rm obs}.
$$

40 点、50 ms 节点间隔的首末跨度为 1.95 s。实际插值、第一节点与 P10 边界从原 spec 读回，不按“40×50 ms”猜首末跨度。

`previous_reference` 在 provider 检查与生成后接缝报告中使用，未进入现有 MAIN 神经观测。若要训练时加入历史 command，那是新的模型输入和训练试验。

旧 MAIN command 路径的 nominal grip 采用其原 2–42 mm 命令映射；实际状态有其 44 mm 机械归一化。两者不能互换。最终实际 mapping 以绑定源码为准，不给所有模型强套同一归一化。

来源：[MAIN provider](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/c8642aa602d3e34dd4dcda5a55ff6ae7c7c53ae6/src/go2arx5_vla/sla_adapter/scale20k_package.py)、[codec](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/c8642aa602d3e34dd4dcda5a55ff6ae7c7c53ae6/src/go2arx5_vla/sla_adapter/codec_v1.py)、[gripper](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/c8642aa602d3e34dd4dcda5a55ff6ae7c7c53ae6/src/go2arx5_vla/sla_adapter/gripper_v2.py)。

### 2.2 类型必须显式

`COMMAND_REFERENCE`：MAIN 的目标训练 command；B0/S1/S7 各自原合同单独绑定。默认不叠 PT 的 achieved-to-command 反馈或额外预紧。B0/S1 原 endpoint basis 继续显式记录；MAIN 不能被偷偷改成 endpoint-only。

`NOMINAL_FUTURE_ACHIEVED`：官方 PT 的已声明名义源动作类型。官方输出 H50 双臂 pose/grip，内部 padding 与左右臂槽位保留；源物理工具／夹爪未完全标定之处继续标 nominal。BTF/GRA 属于该 profile 的声明转换，逐项绑定，不能当成对所有模型普适的补偿。

`PT_GO2_COMMAND_ADAPT_V1`：若以目标已消费命令做学习，输出另声明 COMMAND_REFERENCE。不要把学入的教师补偿再次通过 BTF/GRA 叠加。

共享执行内核只统一物理 ABI 和接续，不抹掉上述语义差异。实际 L1 改了哪些数值、如何输出 WBC packet，需要 `INTERFACE_DIFF` 的实现证据；#516 文档本身不是实现回执。

### 2.3 连续接续与时钟

- 接续从最后真正消费到的参考及位置／速度／姿态／夹爪边界开始；保存 query、packet、atom 与 native tick。
- 合法意图和合法接缝应尽量恒等传递。可恢复的速度／加速度问题允许完整接续和有限重定时；范围、非有限值、frame/tool 错误仍硬拒绝。
- 全部节点、几何和夹爪事件次序保留，位姿与夹爪相位同步解释。原 raw 违规仍登记，不因编译后通过抹掉。
- 最大过渡时间、伸缩、source lag、观测年龄和失败出口从实际已资格 L1 读取；缺失则在已知路径开发上确定，在模型面板前冻结。不能为某次失败临时增大。
- P10 与 SOURCE-END 是不同调度 profile。延长 SOURCE 消费若改变查询合同，就另列对照，不能称纯 seam 修正。
- 正常 receding-horizon 替换尚未消费后缀可以保留，但必须记录替换相位；完整预览前删除后缀来通过安全门不可接受。
- rolling 发送的 native 时间保持单调；同一 query／intent 内 source phase 单调，新的 query 可合法开始新的局部相位，跨 query 使用 query_id＋source phase 追溯。不把同一旧轨迹反复从 t=0 重播；不重置站稳足参考；TTL／source age 沿用原值。
- 若涉及 world increment，平移增量与旋转采用各自声明的约定；不要把 `T_previous @ inv(T_actual) @ T_model` 默认当成 world xyz offset。任何公式需用当前 frame 定义核对。

## 3. 四层归因：哪些量才可相减

定义：

- `raw`：原网络输出，先按本模型 codec 还原后才可与物理量比较；
- $I(\tau)$：query 观测锚定、转换到共同 world 的 declared physical intent；
- $C(t)$：native tick $t$ 真正消费到的参考；
- $A(t)$：同一 tick 的实测位姿；
- $\tau(t)$：该 consumed atom 对应的源相位。ALIGN 段没有对应源点时标空，不强塞为 SOURCE。

在对应关系存在的 SOURCE 段计算：

$$
e_{\rm compile}(t)=p_C(t)-p_I(\tau(t)),\quad
e_{\rm track}(t)=p_A(t)-p_C(t),\quad
e_{\rm total}(t)=p_A(t)-p_I(\tau(t)).
$$

三项向量满足 $e_{\rm total}=e_{\rm compile}+e_{\rm track}$，**范数不满足相同加法关系**。姿态分别用 SO(3) 测地角和旋转误差；不要把旋转角绝对值线性相加，也不要把 quaternion 正负当不同姿态。

对于 PT，compile offset 可包含已声明 achieved-to-command 转换；reference 与 achieved 不相等不是自动错误。应分别核对“映射是否按声明实现”和“目标 achieved intent 是否真的达到”。对于夹爪，受物体阻挡时 command 与 actual 差也不自动代表失效。

每张卡输出：

| 组 | 核心量 |
| --- | --- |
| 时序 | sensor/query/return/accept/install/consume/STOP tick；ALIGN/SOURCE；source index 与替换点 |
| EEF | xyz 残差向量、范数、Z；SO(3)；目标方向进展；意图／参考／实测轨迹 |
| 夹爪 | 左右滑块实际值、command、原单位／映射；关闭／张开事件相位与速度 |
| 全身 | 实际 arm q/qdot、WBC 输出的关节／base reference（如原日志存在）、base 相对 SO(3)、逐足位移 |
| 几何 | 两 pad—物体／桌面间隙、腕／link6／其它相关 link 扫掠、物体支撑与允许接触 |
| 任务 | 原 grasp/lift/hold/place/release/retreat 子谓词及其连续窗口 |
| 责任 | earliest divergence layer、first physical failure、upstream contributors、反证与置信等级 |

不从任意新求出的 IK 解推断“EEF 原意图必然导致肘部碰撞”。EEF pose 不唯一决定全臂；尽量使用原 WBC 真正输出的关节／base 与实际轨迹。若参考关节没有记录，标缺失，保留多种构型假说。

源相位比较用于控制任务进度；共同 native 时间用于控制暴露时长。两者都展示。近物最小距离需同时报告发生 tick、source phase 和实际 pad 表面几何，不能用不同时间的最小值解释瞬时误差。

## 4. 两份必填审计表

### 4.1 A 的每个 arm

```text
condition_id / arm / original_task / spec_hash / scorer_hash
runtime_hash / model_hash / link_hash / WBC_physics_safety_hash
reset_residuals / noise_tensor_hashes
first_grasp_tick / first_postgrasp_fresh_query_consumed
lift_hold_intervals / original_full_predicates
last_consumed_query_packet_atom / source_phase
first_stop_tick_reason / earliest_divergence_layer / upstream_contributors
evidence_refs / falsifier / next_action
```

任务对应 A01 红 HOLD、A02 蓝 HOLD、A03 红左放、A04 红右放、A05 蓝左放、A06 蓝右放，最终以原 registry/spec 为准。没有读回的逐卡事件填 null，不把 aggregate 4/6 分配给猜测的四张卡。

对 A06 单独导出抓持前、接缝、最大抬升、保持结束及碰桌的事件窗。将 L0 在 284 ms 被阻断和 L1 更长暴露区别开，不能直接称 L1 引入碰撞。

### 4.2 PT 的每次返回

```text
query_id / source_observation_tick / input_frame_timestamps / noise_hash
raw_hash / codec_and_semantics / declared_intent_hash
return_tick / accepted_packet_ids / installed_packet_ids
unique_consumed_atoms / partial_atom_ticks
align_duration / source_duration / final_source_phase
close_lift_event_phase_raw_compiled / event_actually_consumed
replacement_or_stop_reason / measured_TCP_pads_feet
```

38 份均登记；未消费者说明原因，未知不猜。rolling 重播同 atom 不重复增加 source 消费。将“返回数”“有正消费返回数”“真正基于新观测的消费数”“抓后新观测消费数”分开。

全量计算之后深入看五类事件：第一返回、第一次朝果推进、最近接近、足漂首次持续增大、最后完整及部分消费边界。观察窗按物理事件选，避免只挑看起来最好的图片。

## 5. 最小修复验证

### 5.1 生命周期

若实际 L1 已有同版本 STOP 资格直接引用。否则针对 return 与 STOP 同时、accepted 尚未 publish、部分 atom 消费时 STOP 的三个交错，验证：

`new_publish = new_install = new_integrate_grant = new_physics_tick = 0`，均从真实 STOP 边界后计数。

晚到 receipt 可被记录，但不得伪装成新 install/consume；同一修复用于所有对照臂。历史 #515 的 STOP 后 rolling 不能被改述为已发生 STOP 后实际运动。

### 5.2 planner 与编译器

缺 planner：从准确异常定位环境／锁文件，恢复项目用户态依赖，验证 import、构造和已有静态规划入口。零积分资格中断仍保留，不把 repaired import 称物理路径合格。

编译器：保存异常、版本与首个已返回结果；用同权重、固定真实输入和实际噪声比较修前可用结果与候选路径。数值容差按原 dtype／既有等价标准预先绑定；不得为了通过再改容差。无可比旧结果时只称功能恢复，随后单列新版本验证。

不一并改精度、采样步数、RNG、图像分辨率或权重。若必须改变算法含义，登记独立 profile；旧条件不覆盖。

### 5.3 诊断对照

M_DIAG 每机制一对 control/candidate，只回答机制预测，不做统计显著性承诺。优先对相同归档输出离线单项开关，只有需要物理反事实才启动。

PT_DIAG 一次只选一个主要机制，原合同与候选各 2 seed。若输入／变换错误在离线已经确定，原生只验证修复的实际影响，不再穷举旧 2×2 网格。

对照 arm 的噪声逐 query 精确核验；初态／控制器状态差异如实记录。模型动作不同后，真实后续观测允许自然分叉。不能固定旧视频冒充在线闭环。

### 5.4 有限端点

- M_HOLD、原布局 M_PLACE、B_TARGET、S7_REG：从各自原 spec 导入 physics/query/wall-clock 端点，两个配对臂完全相同，不在新结果后延长。
- 新 PT_DEV 或无法直接继承的开发任务：最多 50,000 个 2 ms tick（100 s，含初始化）、160 次 query、1,800 s 墙钟；较严格的继承端点优先。该有限端点不是成功谓词。
- Q_GAPS：每卡最多 30,000 tick（60 s，含初始化）、0 模型前向、1,200 s 墙钟。
- Q_PLACE／Q_CONFIRM：每卡最多 50,000 tick、0 模型前向、1,800 s 墙钟。
- 机制诊断若只需更短事件窗，启动前登记；不把截断诊断当完整任务。若原已资格轨迹需要其它有限端点，在新输出前依原合同绑定。
- 原安全门始终先于计划端点终止。原 task deadline、source age、相位滞后也不能被较宽的运行上限覆盖。

## 6. 开发与确认的注册

### 6.1 MAIN 四卡 HOLD

使用原红／蓝 HOLD 任务和原场景，固定新 noise 标签 101、102。执行端在首个前向前生成／封存其真实 PRNG 与 query noise bank hash；标签本身不是噪声一致性的证据。

固定次序：红101 → 蓝102 → 蓝101 → 红102。候选身份在首卡之前冻结；四卡期间不因成绩切换接口／权重。不需要为每卡再次核验整个大权重历史。

首张完整成功立即保存完整回放和证据，继续结算剩余卡。出现共享身份／生命周期缺陷则安全终止受影响版本，保留全部结果，再以新版本重新登记未来面板。

### 6.2 MAIN 四卡 PLACE

两个开发条件在新输出前绑定为 P01、P02，各用 101、102。顺序 P01/101 → P02/102 → P02/101 → P01/102。逐布局完整已知路径资格成立才能执行对应卡；不合格的条件保留 BLOCKED，不按学生得分换布局。

若两个布局只代表同场景内两种语言任务，报告任务差异，不能称两个独立几何布局。每条件绑定实际 scene_hash，报告真实独立布局数。

### 6.3 新布局确认

按主计划 §5.4，在完整开发面板边界从就绪系统中优先选择已有完整 PLACE 阳性的任务，否则选择完整 HOLD 阳性任务；同任务内用 MAIN → 官方冻结 PT → 独立适配分支的固定优先级，不按不同人口的成功率排名。任务族确定后，在最终候选封存前固定 N01、N02、N03 的场景输入、真实几何与来源，核对与 TRAIN／历史开发的重叠。候选代码、权重、参数、任务与 scorer 全部冻结后，才进行同版本资格与 201、202 两 seed 确认。

固定次序 N01/201 → N02/202 → N03/201 → N01/202 → N02/201 → N03/202。六卡全部结算；不把换 seed 当新布局，不因失败更换 Nxx。不能混合 HOLD 与 PLACE 后宣称单一任务成功率。

HOLD 或 PLACE 必须在首个 Q_CONFIRM 前根据开发结果确定。若选 PLACE 而 Nxx 资格失败，该布局保持 BLOCKED，本批不降任务。以后另做 HOLD 需新 profile／新批次，保留 PLACE 资格失败，额外资格不隐含在本批容量。没有同期基线的六卡只能支持冻结系统的小样本可行性，不能用于宣称相对旧接口显著提高。

### 6.4 旧 B 未启动人口

B_TARGET 10 卡保留原模型、任务、seed、查询合同和顺序；仅等价工程修复时续作。冻结水果域的原 10 启动、9 有效、1 技术中断，不再用修复后的新 attempt 替换。

若需要完成 PT 技术中断的后继验证，用新的 attempt/version 指向旧 condition 和失败证据。按旧／新 attempt 分层，合并全启动时保留所有尝试。

## 7. 最小统计与结果类型

每个 panel/version 同时给出：

`capacity, enabled, started, valid_success, valid_failure, technical_interruption, not_started, blocked`。

已结束尝试在明确状态下满足 `started = valid_success + valid_failure + technical_interruption`；存在运行中／状态未知时单列，不能强行凑等式。有效分母是 success+failure；全启动包括技术中断。还未启用的容量不属于已登记成功率分母。

配对表给出两者成功、仅候选成功、仅对照成功、两者失败及无法比较。新六对用户汇总中，如逐卡原件确认 A06 为共同抓持阳性，则 grasp 可推出 1 共同成功、4 仅 L1、0 仅 L0、1 双失败；在读回前不把它当已验证逐卡表。

以下都不是任务成功的替代指标：返回数、action consumption、编译通过率、离线 loss、后缀合法率、PR 数、fixture 数。报告这些是为了归因，最终仍看真实物体结果与完整谓词。

技术中断和模型有效失败的界线依据实际错误：模型输出超限、接触、足漂、超原任务时间均属于有效科学结果；依赖缺失、编译异常等才进技术桶。若同时存在，记录最早实际终止及其上下文，不择有利分类。

## 8. 后训练实际启动前的有限绑定

主计划的条件后训练独立登记，不占用冻结面板的 model/version。使用既有训练入口，不在本计划中伪造未核实的 argv。

正式 250 更新 pilot 需绑定：父完整 SHA、zero-B／继承方式、可训练参数精确名与数目、数据／消费 mask／历史帧、TRAIN/DEV 划分、optimizer、LR 日程、batch／有效 batch、RNG、保存／重载、原始生成配置和有限 wall-clock 上限。数值来源是实际可用原 recipe，改动逐项声明。

8–16 窗工程诊断若做更新，计入独立 diagnostic_updates。PT 的正式 adapter 重新从官方父与 zero-B 开始，避免把诊断权重混进“250 更新”谱系。正式端点固定 250；500 需另有事前封存的证据与后继定义。

本次本会话没有训练、生成或新原生资格。上述动作需要执行端已有实际站点访问与 owner 权限；没有这些条件时可完成只读审计、代码修复与数据准备，并如实报告实际阻断。

## 9. 每块结束的交接格式

```text
本块真实新增：完整任务 n/N；阶段事件；真实前向／tick／updates。
归因：已证实的最早偏离、支持证据、未排除因素。
调整：实际改变了哪个变量；验证是否符合事前预测。
决定：保持／修接口／换调度profile／补物理路径／条件后训练。
状态：当前完整代码与模型身份、未启动人口、作业与租约终态。
下一有限块：具体条件、端点、触发与终止；无逐卡人工GO。
```

本轮成功须保留正常入口、完整轨迹、反馈查询与终态证据；人工设置抓持、Teacher 中途接管或仅回放一段成功前缀，都只能按各自诊断身份报告。
