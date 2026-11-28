# V7 Codex 执行手册 · 三类失效的最小对照

2026-11-28 SGT · v2。入口：[完整计划](SLA_CONTROL_AWARE_POSTTRAIN_V7_PLAN_20261128.md) · [登记表](SLA_CONTROL_AWARE_POSTTRAIN_V7_REGISTRY_20261128.json) · [PR #529](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/529)。

## 0. 当前已经闭合的事项

| 工作 | 已提交状态 | 下一个动作 |
| --- | --- | --- |
| V6.5 3K | ACTUAL_CLOSED；完整恢复与十张原生评价完成，未晋级 | 保留负结果，停止自动延长观察 |
| V7-A 父选择 | R4-200 两组共同父已确定 | 在完整模型资格中使用真实 R4 |
| A93+R3 上一命令 | 8,031/8,031 原生来源核验 | 保持该数据版本和语义 |
| A 两臂采样 | 500步共同bank已封存 | 验收实际noise、完整图、optimizer |
| A 动作侧接线 | 部分 CPU/实际动作数组台架通过 | 三条全模型路径与冷恢复待实测 |
| 早期 Recovery | 蓝LIFT/红PREGRASP CPU候选通过；0动态准入 | 新前瞻TRAIN动态资格 |
| V7 正式训练 | 最新证据中尚未启动 | AQ通过及现场条件核对后执行已封存A |

证据头：V65 2bf6ab7a663099be27bc85a9b424256eed03920b；Core f67fe887a2cc3e3b9309c122db6773febbefb697；V7-A 3de6774e02b3c266e1b36c4c1843217ed9390f8b。开始时检查是否有更新，只读取变化的相关回执，不用反复重扫整个历史。

## 1. 立即推进的工作包

### WP-AQ：把“动作侧接线”推进到真实完整模型资格

负责人角色：现有 Hopper 训练执行者；Corelab 可完成读回与 CPU 部分。使用独立工作副本和新capsule，避免把文档分支当作完整执行源码。

已有位置：

- scripts/sla/v7_previous_command.py
- scripts/sla/v7_recovery_context_audit.py
- scripts/sla/v7_a93_previous_width_audit.py
- 当前训练 worker、实际 FM/aux 构图和 provider 导出入口，按现场加载模块定位，不从旧文件名猜入口。

建议新增可复用的资格入口名为 scripts/sla/v7_full_model_qualification.py，若项目已有同职责实现则扩展现有入口。这个名称是待实现建议，不是已经可运行的命令。

步骤：

- [ ] 恢复同一 R4 父，核 frozen/trainable 分区、原19个活动推理张量及新投影、源域IO路由。不要以3K动作侧台架替代R4端到端。
- [ ] 按封存bank取真实RGB、prompt、state、支持/attention/state-drop mask；FM noise、tau、aux noise独立按稳定样本/路径身份绑定，实际数组对账。新条件初始化不扰动旧随机流。
- [ ] 完整原图、Control、Treatment在零投影时同输入输出一致。记录同backend/dtype重放基线和预登记数值容差。
- [ ] 检查 FM、部署10步采样、aux完整H40的10步采样全部使用条件；包含cache/JIT闭包；用非零注入验证三条路径的真实敏感性。
- [ ] 验证Treatment投影有限非零梯度、Control零梯度；公共参数裁剪前梯度匹配；记录含新叶全局clip因子和实际Adam更新，不能只看名义loss。
- [ ] 各1–2工程更新内完成新投影、Adam、RNG、cursor、checkpoint导出/恢复及下一更新一致；正式两臂另从共同父新建状态。
- [ ] 首窗暖启动、query−25 tick、未来/过期/跨回合/错误单位/不完整消费负例，源码未启用时保持旧行为。数据无缺失；线上缺失按共同预声明异常处理。
- [ ] 输出资格回执、输入数组哈希、参数树、训练路径调用证据、保存恢复结果、实际工程成本。没有完整模型证据则状态仍为 NEEDS_FULL_MODEL_QUALIFICATION。

完成后才能标 GPU_QUALIFIED。不要为低风险文档或小实现改动新增审批；相关实际资源与owner核对复用已有机制。

### WP-E：定位3K四张PLACE最早分叉

负责人角色：Corelab，优先旧归档和CPU分析，可在训练并行期完成。

- [ ] 逐卡 P01_701、P02_702、P02_701、P01_702 对齐R4/1K/3K，从共同初态找到最早动作/运动分叉。
- [ ] 导出原始normalized H40、完整40点解码、原全块检查/L1、实际P10、accepted、selected推断/真值、arm/base请求、实测TCP/base/q/qdot及首接触；保持原调用顺序，不提前截断。
- [ ] 50ms共同完整区间与2ms细节分开，所有字段带单位、坐标、tick、identity和证据类型。不要以最后250ms的共同差异冒充全过程首因。
- [ ] 将失败按 raw-reference / reference-selection / tracking / whole-body-geometry / grip-seam 分类，可多因，标先后及缺证。
- [ ] 在独立TRAIN做同输入R4/1K/3K交叉前向；保护归档仅归因与原评分，不能变训练。
- [ ] 错蓝块案例增加同图同状态noise的合法红/蓝、HOLD/PLACE文字诊断，区分语义方向与路径/姿态。
- [ ] 若发现参考正确而响应分离，形成A2-TRACK候选；若主要姿态歧义，分别准备BASE/ARM。未证明在线来源的selected或仿真GT只留诊断。

建议输出名：V7_E0_FIRST_DIVERGENCE_ACTUAL_20261128.md/json。报告必须区分零新增积分的归档回放与真实动态反事实；不能写成已经修复。

### WP-D：动态验证早期Recovery，并扩独立起点

负责人角色：Corelab现有N1物理owner，沿原TRAIN名册及资源规则。

- [ ] 首波采集学生固定R4-200完整身份；固定新蓝LIFT、红PREGRASP两卡的合格TRAIN几何/教师、reset/noise、触发和预算。新几何先资格；旧失败卡不可恢复，不能自动用未晋级3K。
- [ ] 原G7预安装拒绝时，确认零当前ACK/安装/消费、上一P10完整、原暂停tick及未硬STOP。
- [ ] 继承上一实耗命令，实际教师H40/P10走原门、真实2ms/50ms动态。终态记录所有尝试，包括G7和接触失败。
- [ ] 成功接管只计assisted；原学生拒绝仍为失败。无有效动态结果不能增加4/10或155窗。
- [ ] 回读目标标签逐原子消费、mask、首窗继承、RGB/状态、R4重锚与split；完整成功后缀按原准入加入新版本。
- [ ] 第一波后按实际缺口再安排2–4卡，优先早期接缝、接近、LIFT；保留资源采下一版学生的新状态。每波冷封存，A的R3与bank不热换。
- [ ] 单列independent_handoff_count、场景族、干预时长/次数与全部attempts。不要把teacher接管后的115窗称为115个学生失败起点。
- [ ] 构造C1共同池和匹配早期槽：48旧+44新+2原R3+2普通早期/学生纠正，源域16+16，aux从共同旧/新池固定。

建议输出名：V7_D0_EARLY_RECOVERY_ACTUAL_20261128.md 及新版本manifest。没有动态准入时只更新候选状态；部分安全纠正另存，不暗中进正式成功后缀。

### WP-LR：训练目标与原能力保持

负责人角色：训练执行者配合Corelab；优先固定权重、零优化更新分析。

- [ ] 固定现场实际loss/reader模块指纹，复用真实TRAIN bank、noise/tau、dtype；分项重构原loss和总梯度。
- [ ] 分旧/新/R3/源域/aux、7物理/25padding、P10/后缀、XYZ/rotation/gripper、phase。
- [ ] 报告梯度范数、cosine、加权合成、global clip，以及已有Adam预条件后的实际更新方向对旧能力目标的影响；范数比例不是可加的贡献份额。
- [ ] 统计G7 max罚的argmax在seam/后缀的分布、非零梯度覆盖及与width/OPEN目标的冲突。delta首点previous抵消，不能称其为状态补全。
- [ ] 建独立RETENTION_ANCHOR_TRAIN和validator，source=R4_student_safe_consumed，旧Recovery前缀仍train_eligible=false。逐条有真实消费及局部阶段成功证据；标raw、L1、selected/consumed，不能以accepted或inferred-selected替代消费。保护评价格、失败CARRY/OPEN、未执行后缀排除。
- [ ] 确认anchor池和未来数据池的scene grouping，按整回合/场景分组，不把重叠query随机拆成train/confirm。
- [ ] 为B6实现独立flag：候选每步2个anchor、阶段/回合平衡、窗内有效支持均值再跨窗归约；物理平移/SO(3)/开度尺度固定。主实验用可回放原noise子集，缺失另记；新noise回归另设计。两臂同anchor读取/noise/前向，λ=0恢复原更新。
- [ ] 只用独立TRAIN标定一个λ，固定模型SHA、样本、noise、共享Expert/Go2 IO/条件scope及含原aux的梯度；θ=R4零保持梯度时使用已合法非零漂移位置，不能除零放大。记录applied≠raw初始偏置；无合法anchor时待采或独立L2-SP。
- [ ] B1/B2/B3/B4/B5仅在诊断支持时进入下一对照；不把多个变量一起改进A。

建议输出名：V7_L0_GRADIENT_AND_RETENTION_ACTUAL_20261128.md/json。脚本可叫 scripts/sla/v7_loss_attribution.py、v7_retention_anchor_audit.py；未实现前不得声称可执行。

## 2. 正式A运行与评价

两臂固定：

- R4-200共同父；同结构、同初始化、同数据/noise/mask/步数、独立新AdamW。
- 各500；B128=Go2 96/Robo16/LIBERO16，Go2 48旧/46新/2R3。
- aux4旧2新2、0.375；peak8.481070829156122e-7、warm50、cosine500至0.1peak。
- 唯一输入差异：Control中性 vs Treatment真实上一消费开度。
- manifest ec5e7436b839e8452b61fc137f4fe6f659af2aabb522165c7604413b12d20024。
- 采样数组 a8fde923b6453be823fd8097b8440a7ad8780d6d484bc3d882df8a5b070e64ff。
- 条件数组 7573a2a63170b2a5bf47d51314e08f0c24724ec5a508bed6c61e9fbb4b85fda2。
- 候选seed529202611仍需实际noise验收。

- [ ] AQ通过后核实际Hopper状态、owner、存储/期限；复用原调度机制，避免重复作业。两臂顺序事先固定。
- [ ] 资格结果不带进正式权重。正式恢复使用各自完整状态，真实失败/未封存更新计实耗。
- [ ] 固定500为科学端点；中间checkpoint用于恢复与预登记TRAIN诊断，不按保护成绩选择最佳步数。不要一臂改到1K另一臂留500。
- [ ] 端点完整导出、Core独立恢复；原24反馈与42源窗按原协议回读。
- [ ] 两臂各原十格完整配对，先核实际名额。若只够HOLD6/臂则如实报告部分评价、未满足完整晋级，不能挑好格替代。
- [ ] 报告R4/Control/Treatment逐卡、原成功格丢失/新增、query暴露及接触STOP造成的G7删失。
- [ ] 原保持门与机制筛选分开。Treatment较Control进步而仍退于R4时，只记机制收益。
- [ ] 无自动延长；根据下面选择一个后续训练。所有失败保留。

建议实际报告名：V7_A_CONTEXT_500_ACTUAL_20261128.md；日期按真实完成日命名，不预造结果。

## 3. A之后的分支决定

| 证据 | 第一优先 | 固定条件 |
| --- | --- | --- |
| 同一R4安全输入上两臂均出现明显坏动作/前段丢失 | B6行为保持；无anchor则独立L2-SP | 同父/输入/原bank/步数/LR，单一保持项 |
| 教师状态正常，当前学生早期状态明显失败 | C1早期纠正数据 | 同父输入；48+44+2+2匹配；aux保持共同 |
| A减少seam但腕碰仍由跟踪分离解释 | A2-TRACK | 同父/数据/训练，仅新增残差信息 |
| 主要arm或base姿态歧义 | A3-ARM或A3-BASE择一 | 先证明部署传感来源，不混两组 |
| R3有FM但生成G7坏，aux未覆盖 | B4 | 同FM，只改变合法完整H40 aux来源 |
| 归因仍不清 | 补E/L/TRAIN诊断 | 不启动组合超参搜索 |

后续是否延长、组合或扩大Recovery比例，另在冷边界固定对应父、数据、输入和原生名册。C1 Treatment为late2+early2，总Recovery4；后续比例实验从总4→8→16另立。C1 Control普通早期分层改变bank，不能直接复用A-Control；相同control只有完整父/数据/步数/noise/损失/优化器等逐项一致才可复用。不同父或不同预算的旧结果不能冒充因果对照。

RTC先固定R4做离线连续生成对照，保持当前物理暂停。训练prefix/异步部署与动作参数化属于独立实验；不占用A配对变量。SOURCE608、RECAP/value/RL保持准备，当前优先前段恢复。

## 4. 现有记账与执行边界

已提交正式实耗3,600；A双500后4,600。在原默认8,000范围内示例余3,400；工程已知17/24、A建议≤4；均以最新现场账本为准。早期学生原生最多余24，A20后最多余4；后续B/C完整原生确认需要明确资源安排，不能默认为有无限名额。Recovery首期最多余10，先2再2–4。原320 GPUh和2026-12-05 12:41:52 SGT截止依现场有效合同核对。

这些是工作分配与成本记录，不创建新的批准流程。可独立的代码、数据核验、只读诊断继续推进；仅资源不足的具体运行待资源解决。不改基础设施，不删用户checkpoint，不改WBC，不热改活跃capsule，不重复调度不明状态作业。

## 5. Codex首个回执和每个科学端点必须回答

1. 当前真实代码/权重/数据与最新账本是否仍匹配；哪些依赖已经完成？
2. AQ的完整模型、真实noise、三路径、梯度和冷恢复分别处于什么状态？
3. E0最早分叉在哪一层，证据是直接观测、被动推断还是静态反事实？
4. D0实际启动几次、成功准入几条、独立学生起点几处？哪些仍只有CPU候选？
5. L0有哪些真实梯度/更新测量？R4合法TRAIN anchor是否足够？
6. A两臂分别真实执行多少更新、原生条件、成本；保持门与相对收益分别如何？
7. 下一个选择为什么是B6/C1/A2或其他单项；未选择项的依据是什么？

每个工作包保留一份简洁实际报告和必要machine receipt，不创建重复调度器或大量无结果计划PR。已有稳定证据按SHA复用；输入/实现变化时核受影响部分。计划、台架、完整GPU资格、实际闭环成功必须分别陈述。GitHub推送本身不表示远端Codex、PBS或MuJoCo已经启动。

