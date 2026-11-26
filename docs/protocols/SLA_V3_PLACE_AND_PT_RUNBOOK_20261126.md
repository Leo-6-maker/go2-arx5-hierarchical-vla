# SLA V3：PLACE 事件归因与消费命令后训练实施合同

日期：2026-11-26（Asia/Singapore）。配套：[主计划](SLA_V3_PLACE_AND_PT_PLAN_20261126.md) · [登记](SLA_V3_PLACE_AND_PT_REGISTRY_20261126.json)。

继承 [V2 Runbook](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/2351fdac40cf09846325a8a723453b2d67cc3fe0/docs/protocols/SLA_SUCCESS_EXECUTION_AUDIT_V2_RUNBOOK_20261126.md) 的四层同步、消费身份和 STOP 合同。本文件只细化 V3 新任务，不能被当作尚未绑定的 runtime 配置。

## 1. 现场读回与基线封存

先读 `NEXT_DECISION.md`；其中已完成且附完整原件的判断直接复用。补读 `CONFIRM_ACTUAL.json`、`RESOURCE_LEDGER.json` 及其指向的 HOLD／PLACE 全部逐卡。不要为复算汇总重复跑模型。

绑定：

- V2 结果完整 commit、实际 runtime commit／dirty patch、L1 实现与配置；
- MAIN3.8K 权重、原神经输入、COMMAND decoder／统计、真实 camera 路由与 prompt；
- WBC、PD、solver、腿策略、physics、完整安全和 BASE profile；
- 原 HOLD／PLACE 各自 evaluator；HOLD 的 mobile 80 mm 进展定义，以及 PLACE 自身原 spec 中实际存在的移动／底座谓词，不从 HOLD 外推；
- N02/N03/N01 与 PLACE 条件实际 scene／reset／noise／query 合同；
- 18 次启动的角色、技术／科学终态及 1,259 前向的实际归属。

N02 成功栈未改时不重复资格。相同框架上的不同任务分别继承原安全约束：MAIN mobile 与 PT standing 不混合。

原始结果 evidence_level 保留 USER_REPORTED，直到实际原件和身份读回。汇总表不能伪造不存在的逐卡字段；未知值填 null。

## 2. PLACE 事件合同

每个事件记录 native tick、源相位、query/packet/atom、物体／机器人几何与原 scorer 对应谓词。各事件可反复进入／退出，记录连续窗口而非只记 ever=true。

| 事件 | 需要的量 | 常见误判 |
| --- | --- | --- |
| G 抓持 | 原接触、双指与物体关系、任务对象身份 | 用 command 关闭代替真实抓持 |
| L 抬升／保持 | 同连续区间的高度、抓持和时间 | 拼接不同区间的高度最大值与最长接触 |
| Z 目标区域 | 实际物体的原区域谓词、进入／离开时刻 | TCP 入区或悬空穿过就称已放置 |
| S 目标支撑 | 物体接触、相对速度、姿态、原容差与任务允许的接触 | 额外发明统一“支撑后才能开爪”来改原任务；离线支撑分析不成为在线 GT 控制 |
| Oi 开爪意图 | 完整 raw／decoded 内开度变化、相位、是否真正 open | 由最终 joint outcome=0 推断所有 raw 都不张开 |
| Oc 开爪消费 | 编译／安装／消费身份，实际到达的 source phase | accepted 当 consumed；重复 rolling 累加消费 |
| R 实际释放 | 实际指宽、pad 脱离、物体脱离机械臂及其后运动 | 夹爪张开就等于物体成功释放 |
| B 稳定承托 | 原稳定时间窗、目标区域、速度／支撑／姿态 | ever-in-zone 代替稳定结果 |
| E 安全撤离 | arm／gripper 路径、物体保持、原安全与终态 | 物体一度稳定却随后被扫落仍算成功 |

G/L/Z/S/O/R/B/E 是离线事件表，不是强制的在线状态机。允许的接触顺序与成功条件完全依原 task；任务若允许先释放再落下承托，应如实分析。

逐卡输出 Oi/Oc/R/B/E 的真值和失败理由，不只给“release_and_retreat=false”。完整不满足时按原 scorer 计失败，不能为首成功临时截到已入区时刻。

## 3. 最早偏离与诊断选择

### 3.1 按相同消费时刻比较

对 SOURCE 段使用源相位映射，把 declared intent $I(\tau)$、consumed reference $C(t)$、actual $A(t)$ 放在同 frame：

- compile 向量残差：`p_C(t) - p_I(tau(t))`；
- actual 跟踪向量：`p_A(t) - p_C(t)`；
- 分别记录 XYZ、SO(3)、夹爪命令和实测，不能把各自最小距离或误差范数相加；
- 对 PT 保留声明的 achieved→command 变换含义；MAIN 不默认叠 PT 补偿。

ALIGN 段若没有源对应点，标明过渡；同 query 内 source phase 单调，新 query 可从新的局部相位开始，跨 query 用身份追踪。

EEF pose 不能唯一确定腕／肘／base。需要实际关节与 WBC 真实参考（若已有），不能另算一个 IK 解就判定原意图必碰撞。

### 3.2 首轮最多两对诊断

DIAG-1 优先对应 PLACE 末段最早可判别缺口；DIAG-2 只有新信息价值时启用，可选第二个末段状态或 N03 接近。

每对在前向前填写：

`hypothesis, source_condition, control_version, candidate_version, single_changed_factor, predicted_observation, falsifier, reset_contract, noise_bank_hash, finite_endpoint`。

保存状态机械反事实需要 simulator、接触、controller/MPC 缓存及消费边界可恢复。缺少隐状态时标 matched reset，不声称完整 clone。Teacher 从中途接管只用于物理诊断，绝不算学生自主完成。

缓存 raw 回放说明执行机制；正常入口闭环开发才说明策略行为。若离线已明确 raw 没有正确终段，不启动“无改动再试一个 seed”的伪诊断。

### 3.3 N01 与 N03

N01 的前次 Teacher 失败先分类：技术未运行、Teacher 参考几何／实现错误、真实原安全门失败。只修已证实问题，保留旧资格失败和两个学生未启动。额外 N01 再资格不隐含在本表 PLACE 资格容量内，若与本批同一条件且完全同任务才可合用，否则单独登记未来有限块。

N03 的两次抓前腕部触块视为开发信息。若场景或抓取几何改变，记录新 scene ID；若用它调参数，不能再称该布局为 V3 独立确认。

## 4. PT 闭爪审计的最后核对

不重新生成 38 份预测。读回已有审计并核实：

1. 38 个独立返回身份，全部 H50 节点覆盖、有效右臂 gripper 槽；
2. 阈值取自哪个源诊断、方向是大值开还是小值开，作用于哪个反归一化阶段；
3. NaN／缺行／变换丢失没有被误记为“没有闭合”；
4. 每个 query 的最低／最高值及 raw→physical→command 映射；
5. 原 source 阈值与目标实际两指接触、nominal GRA 的区别。

确认后，结算假说 `ARCHIVED_CLOSE_SUFFIX_STARVATION_REJECTED`：保形重定时这 38 份固定输出不能补出源闭爪事件。这个标签不等于“视觉是唯一根因”或“任何调度永远无效”。

已验证 checkpoint、处理器、编译路径没有变化时直接复用源阳性；确有变化才补最多两真实源锚点×两噪声、四次离线前向。

## 5. PT target-command 数据构造：必须改的接口

### 5.1 原准备链不能原样当执行标签

固定旧版本的 reader 从 `sensor_action.npz.nominal_action` 取完整名义 H50；实际消费保存在 `task_audit.npz.installed_reference_H40`、`confirmed_native_interval_mask` 和 `native_audit.jsonl.confirmed_native`。右臂 recipe 只有动作维度 mask。

实施前先比对现场后继代码；若已正确修复，复用该身份和验证，不重复改动。若未修，以下两种方案显式选一，首更新前冻结。

### 5.2 A：优先构造连续真实消费 command

按 native tick、query/packet/atom 去重，只从真实 consumed 区间恢复 $C_{\rm native}(t)$。installed 但未消费、STOP 后记录和计划后缀不属于支持集。

对当前观测 $o_i$：

1. 输入只含观测时刻及以前的真实图像／历史、本体状态和任务文本；
2. 标签时刻为 `t_obs + first_label_offset + k/30`，k=0..49；first_label_offset 由新 COMMAND codec 明确声明，检查首节点约定，不默认为源 achieved 下一帧；
3. 可使用后续 query 已实际消费的 command 作为未来监督；后续图像、物体 GT、阶段和结果不得回流到当前模型输入；
4. 所有标签转换到当前 query 冻结的 frame/tool，不能逐段使用各未来 query 的锚点拼成伪轨迹；
5. 保留真实执行的插值／ZOH／gripper 事件规则。无持续支持证据时不跨 gap 插值，不跨 reset／STOP，不复制末点生成“已执行”后缀；
6. 每个有效标签保留原消费区间及变换 provenance。SO(3) 用正确群插值与 quaternion 约定，不线性插值出非旋转；
7. 任何 episode 末尾缺口使用时间 mask。

### 5.3 B：只监督该 query 已消费前缀

若连续命令重建暂不可用，可把当前 query 的真实消费支持映射到 30 Hz 网格。有效行由实际区间、端点和插值支持计算，不能猜“P10 对应前 15／16 行”。

B 更容易实现，但会留下更多未监督后缀。报告有效时间覆盖、终段事件覆盖，不能据此声称完整 H50 的远期行为已被训练。

### 5.4 时间×维度 mask 同时用于 noisy input 与 loss

在官方动作变换、归一化后，模型内部 action 为 `(B,50,32)`。右臂有效维为 `7:14`，令：

`M[i,k,d] = consumed_support[i,k] AND right_arm_dimension[d]`。

示意算法（不是已实现源码）：

```python
safe_target = where(M, normalized_action, 0.0)
x_t = flow_time * independent_noise + (1 - flow_time) * safe_target
u_t = independent_noise - safe_target
prediction = model(observation_at_query, x_t, flow_time)

per_example = sum(M * (prediction - u_t) ** 2, axes=(time, action)) \
              / sum(M, axes=(time, action))
loss = mean(per_example)  # batch 中只允许预先验证过的非空支持样本
```

M 是训练目标的有效性信息，不作为在线 observation、阶段标记或目标物体信息输入模型。不仅 mask loss，也要在 noisy action 构造前清除无证据 target，以防 action token 间耦合泄漏未执行标签。无效槽噪声保持独立原随机约定，不能用未来实际值填充。

只有已实际消费 command 可以进入该分支标签；未来 achieved TCP、名义未消费计划和拒绝命令不能被重新命名为执行成功。

### 5.5 必要验证及通过标准

这是语义改动，须验证实际数据与真实网络：

- 每个有效 row 都能追溯到 consumed 区间，frame/time round-trip 误差沿用已封存数值容差；错误不能用放宽容差掩盖；
- 已知最后半个 atom、STOP 边界、跨 query 接续和 episode 末尾分别有样本；
- 将被 mask 的目标大幅扰动，在固定 observation/noise/time 下 noisy input、有效输出损失及有效参数梯度不变；旧维度 mask 用例同时保留；
- 完整网络一次实际 backward，冻结叶没有更新，可训练集合与 manifest 一致；
- zero-B 初始化时 LoRA A 首步梯度为零可能是预期，不要求每个张量首步都非零；应检查有限值与有效学习路径；
- 保存／重载同一包，对固定真实输入和采样噪声满足原数值一致性；
- 无 clipping、无未声明统计替换；报告真实标签超分位比例；
- 报告独立 episode／布局／事件／消费秒数，不能用重叠 H50 行数冒充新独立样本。

已有 CPU 形状检查不替代这些实际训练前验证。

## 6. 数据采样与训练端点

### 6.1 MAIN 条件后继

先审旧数据和实际 optimizer 访问，必要时补 TRAIN Teacher 数据。首试按 70% PLACE、30% HOLD replay 采样；PLACE 内按接近20%、闭合15%、抓后抬升20%、放置／释放30%、撤离15%分阶段，再在阶段内按 episode、query 分层。只有真实消费支持的样本入池。

这些是 V3 新提案，不是原 MAIN 已使用的配方。阶段缺失不能用重复窗口假装覆盖；若需修改比例，在首更新前记录实际覆盖和新方案，不按 DEV 中途调权。

指定 MAIN3.8K 父、动作侧参数集合、短 LR 曲线、数据与 sampler、optimizer/RNG 全部封存后，固定 250 更新。VLM／视觉语言及其旧 LoRA 冻结，原 L1、输入与 command codec 不变。训练范围变更与数据改变共同披露，不作单因素归因。

### 6.2 PT 首试

四个水果 TRAIN 布局槽 T01..T04、各两个 reset 条件为首批最多八次数据尝试。精确几何在 Teacher 新输出前封存；可精确复用既有合格样本。首试至少四条完整 Pick & Hold、至少两个 TRAIN 布局，不能用失败前缀充数。

均衡采样接近、闭合、抓后首反馈、抬升／保持四组，再按 episode、query 分层；若闭合事件被重复窗口覆盖，只计一次事件。

正式 PT 从官方父＋zero-B＋新 optimizer/RNG 开始，adapter 初始化种子1729与训练种子42分开绑定；250更新、warmup25、peak2.5e-5、decay_steps250、末值2.5e-6；AdamW(.9,.95,eps1e-8,wd1e-10)、clip1、有效batch32、EMA关闭、state dropout0、无新增增强。microbatch 与梯度累积在真实许可设备上绑定后固定，不能用降低分辨率或另换精度偷换比较。

只评价 raw step250，恢复快照不用于挑最好。工程拟合诊断若更新，单列 diagnostic_updates，正式训练不继承它。首启动前绘制／记录实际第0、25、249步 LR，避免代码配置名写250而实际仍用长 warmup。

## 7. 开发、对照和确认的顺序

### 7.1 MAIN PLACE

P01/P02 两个原任务开发条件，noise 标签301/302。固定顺序：

`P01/301 → P02/302 → P02/301 → P01/302`。

实际 PRNG／query 噪声张量 hash 在前向前绑定。标签相同不自动代表张量相同。P01/P02 若同一几何的两个 reset，应如实报告一个几何布局，不能伪称独立布局。

仅一个候选进入本批。无原始意图／执行修复／新训练变化的同栈随机重复不算机制推进。已存在的完全同版本完整任务原件可以复用作事实，不填充新卡。

H_REG 优先使用 V2 N02 两个真实已归档噪声身份，绑定为 N02_A/N02_B；历史与新结果分表。若行为退化且 reset／环境差异无法排除，再另登记匹配基线诊断，不能自动解释为新训练导致。

### 7.2 PT 同 codec 净学习对照

两个未入 TRAIN 的水果 DEV 布局 D01/D02，各 noise501/502；共同新 COMMAND profile 下端点为 CMD0、CMD250。

每个 (layout,noise) 一对。固定平衡顺序：

| 配对 | 顺序 |
| --- | --- |
| D01/501 | CMD0 → CMD250 |
| D02/502 | CMD250 → CMD0 |
| D02/501 | CMD0 → CMD250 |
| D01/502 | CMD250 → CMD0 |

两个端点全部八卡结算；模型真实越界或原门触发为有效失败，不移到技术桶。step0 是新 codec 的零更新对照，**不是原官方部署 profile**。

Teacher 资格只用于同布局可行性，不将 DEV Teacher 轨迹训练进去。两臂共享 processor、统计、decoder、L1、task、WBC、安全、query合同；学习净效应只在这些实际相同时解释。

### 7.3 MAIN 新布局确认

在候选最终冻结前选择并封存 C01/C02/C03 未调参布局，做 TRAIN／开发重叠核查；任务固定为原完整 PLACE。候选选择规则在主计划，不能用确认结果选择模型。

资格只补缺口。确认 noise401/402，顺序：

`C01/401 → C02/402 → C03/401 → C01/402 → C02/401 → C03/402`。

资格失败保持 BLOCKED，不换布局或降成 HOLD。六卡是小样本可行性确认，没有同步旧版基线不声称因果提升或可靠性保证。

## 8. 有限端点、分母与收尾

沿用现有 runner、evaluator、manifest 与生命周期。每个启动只记录必要的版本、输入、终点、实际效果和结果；不新增函数级审批／租约框架。

- 原任务／配对继承原 physics/query/task/source-age/safety 端点；更严格者优先。
- 新数据／完整路径资格若无可继承上限，每次最多50,000个2 ms tick、1,800 s墙钟、0学生模型前向；成功／失败按原task，不因运行cap较宽而延长task deadline。
- 新模型开发若无可继承上限，每卡最多50,000 tick、160次query、1,800 s墙钟；不能按结果延长。
- 每个正式250更新训练job先封存真实墙钟上限，首提案4小时；不足可保存完整端点后以同科学定义和新job身份恢复，累计目标仍250，不重复更新。不绕站点限制或接管他人资源。
- 优先旧资产／报告复用；每个技术失败修复后新 attempt 保留成本，不删除旧失败。
- 分别报告 capacity/enabled/started/valid_success/valid_failure/technical/not_started/blocked；Teacher生成与模型任务分母分开。
- 准确报告逐任务 n/N、完整 episode、独立布局、消费命令支持、训练更新、推理／积分；不把资源总数当成功率。
- STOP 后无新 publish/install/grant/integrate，历史有效结果不重写；实际旧作业终态确认后才启动后继。

V3 的下一次有效进展应是：明确的 PLACE 首失败被修复且原生验证、正常入口完整 PLACE 成功，或 PT 同 codec step250 相对 step0 的可解释行为变化。只有文档、mask 单元检查或 loss 降低，均不能作为任务成功交付。
