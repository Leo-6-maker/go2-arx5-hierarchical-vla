# GR00T＋冻结 WBC：试验进度审核（2026-10-24）

**文档类型：证据审核与计划依据；本次发布不启动模型、PBS、物理或部署。**

- 完整审核基线：**36925d272095d000ba1ad3ce5e9cdc44373fdce6**，2026-10-24 **01:37:00 SGT**；发布前增量复核至 **5d9f2a82a6aa4830481f7197298f666cdc4aad0c**，**01:46:45 SGT**，纳入新作业1481574实际提交、转R及模型启动放行回执。
- 上次审核：648458a6b19a0454de8069cd51bc4640e957f634，2026-10-23 18:28:41 SGT；本次比较覆盖其后 29 个提交（完整审核26个，加实际提交／启动回执2个、发布时终局契约CPU准备1个）。
- 发布并发复核：**5c9a6de763c6f74b7731809f72b2dd54e6ca47de，2026-10-24 01:59:22 SGT** 新增[1481574共享终局契约及精确身份CPU验证](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/5c9a6de763c6f74b7731809f72b2dd54e6ca47de/reports/PI0294/design/parallel-science-20261022/B_PRECUDA_RETRY1481574_SHARED_TERMINAL_CONTRACT_CPU_20261024.json)。它尚未产生原生终局审计／结算／关闭，未宣称首次更新；运行capsule仍为36925d27，上述科学与训练实际状态不变。
- 现役集成分支：pi0267/executor-decoupled-recovery；main 和 Issue 入口正文不是实时执行状态。
- 方法：GitHub 固定源码／报告／原生回执交叉审阅、关键数字和当前预算复算、上次附件及数据代码审核的差异检查。没有登录集群独立重跑模型／原生闭环；“实际”指被引用原生回执中的运行事实，不表示物理真机。
- 配套：[V3 长程计划](../roadmap/2026-10-24_GR00T_WBC_LONG_HORIZON_PLAN_V3.md)；[当前接续交接](../handover/2026-10-24/GR00T_WBC_NEXT_EXECUTION_HANDOFF.md)。

## 1. 核心判断

项目已经形成独立源域物理示范、GR00T 实际训练和受保护自主任务执行的证据链。当前最完整的学生新条件确认仍为 T1 抓放 10/20、T2 推送 4/20。两任务尚未达到该轮各 14/20 的确认标准。[E01]

新增29个提交推进了失败恢复、Vanda驻留控制、预算结算、目标接口CPU接线、最新一次真实重试提交和启动放行，以及共享终局契约的CPU准备，没有新增完整数据集、A/B终点任务结果或目标学生任务结果。**A4096已完成；B最近两次尝试失败终结，完整恢复点仍为B2048。01:46增量回执确认新作业1481574已转R，服务器驻留owner健康，MODEL_START已放行；FIRST_UPDATE尚未放行，未观察到新训练更新。** 目标任务原生资格仍为false，核心64格为NOT_RUN，不能写成0%成功率。[E02] [E03] [E26]

当前关键路径是将已有学习与接口工程转化为完整行为结果。继续冻结 WBC、保留实际安全与数据约束；源域研究不等待目标 hold RMS 或 WBC 调参。当前阶段不应再因长程愿景增加并行框架、热换模型／统计或重置时钟。[E06] [E07]

### 当前证据状态表

| 工作面 | 已有证据 | 当前缺口／正确结论 |
|---|---|---|
| 独立源域 Teacher | 固定基座 ARX5、独立 IK/PD、有限夹爪与物体接触、双 RGB、独立 raw/codec 审计 | 不是目标 WBC 执行的示范，也不是移动整机策略 |
| 源域学生 | N1 40 个合格条件：T1 10/20、T2 4/20 | 有部分任务能力，未达到该轮双任务确认标准 |
| 新 A/B | 同 CORR@2048 父权重、独立优化器、A 已新增4096；B 完整2048 | B4096 尚未产出；新终点行为差异未测 |
| 新 DEV48 基线 | 旧 CORR 完成14有效条件＝9成功＋5失败 | 34未完成，不能把9/14当完整DEV48结果 |
| 目标任务 overlay | 原机器人属性保留、可动夹爪／物体／两相机构建与渲染 | 最新实际计数为0物理／0WBC forward／0GR00T query |
| 新接口实现 | 观测、安装／反馈ACK、relay、首状态持久化等CPU接线 | 当前源progress学生ACK和目标完整原生资格仍未闭合 |
| DEC1科学主张 | 已登记两个执行器×两个接口×两个任务的配对设计 | 0/64；同模型迁移和progress收益未证明 |
| EEF13／BASE | 已有方向和部分接口／Teacher pilot | 没有已验证的真实BASE监督整机学生 |
| 世界模型／规划 | 已有长期研究策略 | 尚不是当前经验证的策略组成或迁移证据 |

依据：[E01] [E05] [E07] [E08] [E09] [E10]。

## 2. 本次新增事实：B2970 中断与 pre-CUDA 恢复失败

### 2.1 训练终局必须按最后回执更新

| 对象 | 实际结果 | 对下一步的影响 |
|---|---|---|
| A_CONTROL | 新增4096更新、完整终点、自然结束与结算关闭 | 保护原终点，不重训A，不让B继承A |
| B job1480817 | 从2048运行到2970；owner报告 VANDA_PROJECT_TOKEN_EXPIRED，随后 HeartbeatLost，F/Exit1 | 新增922更新未进入完整恢复点；B2970不是可恢复checkpoint |
| B恢复状态 | B1024/B2048经过独立完整文件审计；合法恢复点B2048，cursor65536 | 原始模型、优化器、调度器、RNG、采样游标和R0继续绑定 |
| Vanda recovery2 job1481489 | 服务器驻留owner唯一提交；随后模型释放前存储准入失败，F/Exit1 | 曾经Q的旧快照已被终局覆盖 |
| 1481489模型影响 | 模型加载0、query0、update0、physics0 | 不属于学习失败或新的学生负例 |
| 1481489资源影响 | 265 GPU分配秒、1180 CPU秒、64MiB已结算；两条Vanda lease关闭 | 零更新不等于零成本；旧请求与结算不能重放 |
| 最新重试1481574 | PBS17/28已实际双once采用；新作业唯一提交并转R，驻留owner健康，MODEL_START已放行 | FIRST_UPDATE未放行，未观察新更新；继续原owner，不重复采用、qsub或模型启动放行 |

来源：[E02] [E03] [E04] [E05] [E26]。01:37的“未提交”和01:45的Q均为历史快照；当前以01:46的R／MODEL_START回执为准。模型启动放行仍不等于完成加载或新增训练更新。

1481489 的已知原因是冷态准备仍使用旧生命周期存储峰值计划；prequeue 检查没有覆盖完整剩余输出，原 reserved-runtime 约束随后正确拒绝。后续冷修复用批准的完整未来计划和真实结算成本重新计算，并在 qsub 之前拒绝不足峰值。捕获的实际输入重现了旧计划拒绝／新计划通过。这是具体的存储生命周期准入问题，不宜笼统归因于 GPU 不可调度，也不表示新任务已经运行。[E03]

### 2.2 Vanda-only 是已明确的后续执行方向

用户已要求 Hopper 暂不可用时，训练／微调全部转 Vanda A40，并有效利用已分配资源。新的唯一控制owner应驻留Vanda服务器，不依赖桌面、agent或SSH会话持续存活。此路径先提交1481489并记录其失败；最新1481574已沿同一驻留方向采用、提交、转R并取得MODEL_START放行。当前仍无FIRST_UPDATE放行或新增更新回执，不能写成训练PASS。[E11] [E03] [E26]

旧 Hopper CPU 实体lease仍须按 UNAVAILABLE_FENCED_NOT_CLOSED 披露，不能伪称物理关闭，也不应要求先恢复Hopper登录才允许未来Vanda训练。需要保留原历史、单owner与未知写入仅核对不重发的语义。

## 3. 源域学生结果：可信，但当前能力仍有限

### 3.1 N1完整独立确认

| 任务 | 合格条件 | 成功 | 执行前S1拒绝 | 完整任务失败 | 成功率 |
|---|---:|---:|---:|---:|---:|
| T1抓放 | 20 | 10 | 9 | 1 | 50% |
| T2推送 | 20 | 4 | 14 | 2 | 20% |
| 合计 | 40 | 14 | 23 | 3 | 35% |

42次实际尝试形成40个合格统计条件；两条工程不完整尝试及原允许replacement保留。每任务20条件为10个几何组×2推理噪声，不是20个独立场景。2319次实际模型生成中2296次执行并ACK、23次拒绝；0实际native故障发生在S1保护下。执行使用暂停仿真／远程查询／安装／推进的同步闭环，尚未证明实时吞吐或真机部署。[E01]

上一轮匹配BC/CORR在暴露DEV8上3/8→5/8，改善集中T1的2/4→4/4，T2均1/4。C1仅4独立学生parent／8段。一个训练seed和暴露DEV不足以支持一般化DAgger因果收益。

### 3.2 失败机制证据支持“任务数据／执行余量”方向

FD1：23拒绝中21碰撞预测、1TCP速度、1夹爪／关节有效性；20/21碰撞可重建为接触垫／桌面几何。T2的14拒绝发生于12–28.8s，11次物体已经移动超过10cm；这些常为接触后的持续执行问题。旁观 AUDIT_ONLY_approach 不代表真实失败阶段，预测被拒前缀也不等于已发生碰撞。[E12]

FD2：4个新TRAIN布局的配对Teacher，4mm和12mm名义间隙均4/4成功；实际最小垂直间隙分别约0.323–0.730mm与8.246–8.674mm。证据支持增加Teacher几何余量，但不证明旧Teacher不可行，也不等于学生已学会更安全或更成功的行为。[E13]

源策略弱点仍包括T1最后释放／撤离／保持，以及T2推送终点、分离／保持和禁止抓举的过程要求。完整物体任务成功仍为主要指标，不能以物体移动、部分prefix或loss改进代替。

## 4. 数据、表示与A/B对照

### 4.1 数据用途保持冻结边界

| 资产 | 已知规模 | 用途 |
|---|---|---|
| 原source126 | 126 parent＝96T1＋30T2 | 原始示范来源 |
| C1 | 4独立学生parent、8段、1665合法H40窗口 | 固定纠正来源 |
| PV1新R0 bank | 139 parent＝75T1＋64T2；77几何组、62双任务配对 | 当前B的覆盖干预 |
| CR2 R1 | 12配对布局、24/24准入 | 分离的R1储备 |
| VAP1 D48 R1 | 24新布局、48首次尝试，46准入＝T1 24＋T2 22 | 分离的R1储备；两失败保留 |

46/48描述Teacher数据，不是学生成功率。窗口数／重复采样不增加独立parent数。R1更改相对EEF可表示范围，会改变相同归一化输出对应的物理动作；不得给旧CORR热换decoder后仍称原基线。[E08] [E14]

### 4.2 当前A/B的唯一主要干预

两臂同父旧CORR@2048，独立fresh optimizer，各新增4096步、batch32。B不接续A。

| 每batch32的来源 | A_CONTROL | B_COVERAGE |
|---|---:|---:|
| 原T1 | 8 | 4 |
| 新T1 | 0 | 4 |
| C1 | 8 | 8 |
| 原T2 | 16 | 8 |
| 新T2 | 0 | 8 |

两臂T1:T2=1:1、C1=25%；B把一半普通专家访问换成新增双任务示范。不能将结果单独归因于C1、WBC或新backbone，也不能把A126与B139理解为完全换库。[E15]

### 4.3 代码审阅结论

上次沿capture→实际提交→独立raw audit→export→reader/collator→loss的检查，没有发现明确破坏当前A/B的标签或输入泄漏错误；本次差异没有改变这些数值语义。

- 状态取当前实际TCP／夹爪；监督取实际提交的reference，而非未来实际达到轨迹。
- 导出要求完整episode、独立raw审计、相机／标签／hash一致，拒绝不完整或被修改的动作链。
- 物体真值、Teacher阶段和接触审计与学习输入隔离。
- EEF10、H40、相对SE(3)、前10维mask及零padding语义一致。
- 源场景移除free-base与12腿关节，采用独立IK/PD；body_rgb为固定外部视角，不能把源学习结果解释为完整free-base WBC训练。
- 这些是代码与既有回执支持，不是本轮重新加载私有权重、验证全部原始视频或重跑仿真的声明。[E16] [E17] [E18]

## 5. 目标端：接口工程前移，原生证据没有增加

本次新增5类工作：target time ACK、原relay冷接线、冻结source修复、首durable state绑定、实际观测／既有查询路径绑定。观测读取复用ActualState、二进制state、FrameRecorder；ACK关联安装／MPC／反馈与独立前缀安全。相关精确CI有后续成功回执。[E19] [E20] [E21]

早期relay修改触及冻结source而导致两个CI失败，随后恢复原source字节，以独立target subclass实现扩展。失败被保留，没有通过取消pin检查掩盖。这属于工程修正证据，不是native task通过。

### 原生证据边界

- 历史STAGE20S／21已有GR00T reference进入原生WBC的执行；当前问题不是从未存在WBC接口。
- 当前目标overlay实际构建与双RGB渲染通过，但计数仍为0physics、0policy episode、0WBC forward、0GR00T query。
- Source02四个原commissioning条件均为已知reference/S1拒绝；实际6800live＋7331shadow ticks，prefix数2/14/1/17；第一条STOP ACK确认。Source01工程UNKNOWN和replacement仍保留。
- Source02属于Teacher组件接线，不能填入学生核心64格。新source progress学生ACK亦未合格。
- 12D MPC预测省略腿／接触动态，不能替代完整原生prefix安全。
- 本轮完整原生前缀检查、观测、安装、实际反馈、ACK、STOP及任务评分链仍未资格。[E09] [E10] [E19]

**目标资格应证明执行与测量链有效，不应要求学生先在目标成功才允许打开确认集。** 合法运行中的真实任务失败应保留；不通过筛选成功案例获得“目标ready”。

## 6. DEC1主张的当前状态与统计边界

冻结旧CORR@2048／R0，8新几何×2任务×2执行器×2接口＝64条件。两个执行器为源IK/PD与目标冻结WBC；两个接口为正确名义时间I_TIME与实际TCP／夹爪反馈进度I_PROGRESS。[E22]

主要比较为目标progress−time的任务成功率差，并与源域同类差比较。两接口必须共用正确的坐标／TCP和安全实现，避免把共同bug修复计为progress收益。

| 主张 | 当前支持 |
|---|---|
| 源Teacher不依赖目标WBC策略 | 已由实现及数据provenance支持 |
| 同一共享策略有两任务部分能力 | 源域N1支持，尚未达发布标准 |
| 新覆盖训练优于匹配控制 | 新终点评估未完成 |
| 同一模型可迁移目标WBC完成任务 | 当前主矩阵未运行 |
| I_PROGRESS优于正确I_TIME | 当前主矩阵未运行 |
| 任意本体／纯WBC因果／实时／真机泛化 | 没有对应验证 |

64行具有8个几何block的配对结构；多推理noise不能替代独立训练seed。若源和目标都表现很弱，小transfer gap不代表鲁棒。两域还同时改变基座、相机和动力学，第一层结论应为两个执行域之间的迁移，纯控制器因果需后继控制设计。

DEC1固定旧模型，不等待新A/B赢家或完整B0；PV1正式baseline-relative选模／LOCKBOX则需要完整B0。两线可并行准备，具体运行须分别核算完整资源，不让一线占用另一线保护封套。

## 7. 当前战役成本和完整剩余资源

以下是**当前父campaign累计账本**，不是仓库全部历史研究累计；GPU秒指分配成本，不是模型纯计算时间。未来数为保守封套，不是预期实耗。[E03] [E04]

### 7.1 更新损耗

- 已结算实际更新9666；
- 完整保留进度A4096＋B2048＝6144；
- 未保留更新3522＝1024保存失败＋771 A中断＋805 B早期中断＋922 B2970中断；
- 若本次恢复最终完成，11714实际更新得到8192有效终点更新，额外优化器步数为3522/8192＝42.99%。

当前1214次query是同一父campaign口径，不与旧N1的2319次生成混成全项目总量。

### 7.2 预算表

| 资源 | 已用／结算 | 必需未来封套 | 当前批准上限／保护 | 完整投影 |
|---|---:|---:|---:|---|
| DEC1更新 | 9666 | B再2048 | 11714 | 恰达11714 |
| DEC1 PBS | 14（含已提交1481574） | 原矩阵3 | 17 | 恰达17 |
| Global PBS | 25（含已提交1481574） | 原矩阵3 | 28 | 恰达28 |
| DEC1 GPU分配秒 | 70509 | 109140 | 187200＝52h | 179649，余7551s |
| Global GPU分配秒 | 91003 | 109140，另保留72000 | 345600＝96h | 含最终20h保护为272143s |
| Global GPU站点存储 | 以实际账本及生命周期峰值计算 | 完整未来输出＋最终保护 | 680GiB，内含最终40GiB保护 | 保守预测余约38.1MiB |
| DEC1 LOCAL CPU秒 | 50278.9747（含本次控制计费） | 7320 | 57600 | 57598.9747，余约1.0253s |

109140 GPU秒＝已提交B的预留22560＋三个尚未提交矩阵作业86580；不能再把B的一次PBS计入未来新增提交。提交前13/24＋4与提交后14/25＋3的最终PBS合计相同。原5520site／3600LOCAL未来控制封套保留。旧账本字段名H200_s／Hopper_CPU_s／Hopper_storage_bytes沿用历史schema，不表示A40已成为H200或其小时具有等效算力。矩阵封套为88episode／7040query／2816000ticks；它是预算保留，不重置已经消耗的commissioning槽位，也不自动容纳额外stress32。最终20GPUh／40GiB为global总额内部保护，不再重复加到DEC1上限。[E03]

存储余量精确为39991296bytes。这里的38.1MiB与DEC1 LOCAL的1.025CPU秒均是**完整未来预算投影余量**，不是现场磁盘free或当前机器空闲能力。一次额外重试或未计输出都可能需要重新核算；已有成本不能回扣为免费。

旧B0剩34＋A48＋B48＝最多130源DEV，及后续LOCKBOX80，未自动包含在这份“B恢复＋三个矩阵”的剩余封套内。先检查是否存在仍有效且已覆盖的原授权；未覆盖时把具体增量一次列入后继工作包，不能挤占DEC1或最终保护，也不重复索取已批准PBS17/28。[E23]

## 8. 效率问题：已经定位，尚无新加速结果

实际记录：平均同步更新6.583s，分配CPU36而进程affinity4，DataLoader workers0。scheduler全作业GPU usage23.489%包含初始化／等待，不代表稳态GPU利用率。[E24]

代码发现MP1Trainer override固定worker0，因此只改TrainingArguments无效。新opt-in worker2/4已有小CPU fixture的顺序／tensor／RNG／cursor检查，但完整PV1数据和原生吞吐资格未完成；当前配方仍micro2×accum16＝batch32、worker0。[E25]

近期优先完成原科学终点。worker优化只在冷态、完整数据等价／恢复／总CPU计费都通过且资源确实容纳时采用；不能热改一臂、放宽数值可比性，或把CPU fixture通过写成GPU已提速。真正效率指标应为有效保存更新／分配GPU小时及合格任务条件／端到端小时。

## 9. 固定时钟、停止和本轮诚实出口

全部采用Asia/Singapore：

| 边界 | 原时间 |
|---|---|
| 首真实结果／精确阻断节点 | 2026-10-23 20:00，已到；当前阻断应如实报告 |
| 停新作业／候选 | 2026-10-25 02:29:56 |
| 停新训练块 | 2026-10-25 14:29:56 |
| 停新物理 | 2026-10-25 18:29:56 |
| Hard close | 2026-10-26 02:29:56 |

本文件不重置T0、截止或计数。没有新结果时，应交真实恢复状态、所有失败、成本、实际资格及NOT_RUN矩阵。当前战役可以PARTIAL_WITH_EXACT_BLOCKERS结束；科学无收益、源能力不足、目标执行阻断和资源／站点不完整分别报告。

## 10. 附件与范围说明

本次重新核对两份上传压缩包的SHA256，与上次审核相同：WBCv1为3c59a6d5f8f924df0b838ad2d501d3a853274d741db191a9de453f58212063ee；MuJoCo包为6dd13098de5a58327fb0a0a5b3ff749e0a5534510375b92fba3f8fb50a322d52。

两份上传包属于不同路线：WBCv1为历史IsaacGym/RoboDuet双策略路线；wbc_mujoco_minimal为MuJoCo＋OCS2＋腿RL／臂PD。不能把旧九点空间preview与当前时间化reference当成同一ABI。

上次附件审核保留：MuJoCo包11338个SHA条目匹配、纯Python轨迹测试13/13通过；原smoke仅支持其有限运行范围。WBCv1 benchmark主要为dog-only，arm_only／wbc评估入口未实现，并发现切线旋转基反射和速度缩放被归一化抵消的问题。它们不是当前MPC路径的已证实失败原因。本次没有重新执行附件原生ROS／GPU环境，也不将私有源码、mesh、原始RGB或模型权重写入Git。

审核没有发现足以推翻当前源学习数据链的明确代码错误；尚未闭合的关键科学证据仍是源域终点对照和同模型跨执行器比较。详见[V3计划](../roadmap/2026-10-24_GR00T_WBC_LONG_HORIZON_PLAN_V3.md)的阶段、验收与后继分支。

## 固定证据索引

文中E编号均可直接点击。关键入口：[完整学生结果][E01]、[最新实际提交][E26]、[失败与资源][E03]、[DEC1协议][E22]。原历史文件保持原字节；最新状态按实际回执时序解释。

[E01]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/20261016T191445Z-localpc-mp1/N1_FINAL_SCIENCE_REVIEW.md "N1完整科学结果"
[E02]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/parallel-science-20261022/B_RETRY1_TOKEN_EXPIRED_HEARTBEAT_FAILURE2970_ACTUAL_20261023.json "B2970实际失败"
[E03]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/parallel-science-20261022/VANDA_RECOVERY2_PRECUDA_STORAGE_FAILURE_CLOSED_AND_RETRY_PBS_DRAFT_20261024.json "Pre-CUDA失败、关闭与完整未来预算"
[E04]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/parallel-science-20261022/B_PRECUDA1481489_RETRY_PBS_AUTHORIZATION_20261024.json "PBS17/28明确授权"
[E05]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/parallel-science-20261022/B_PRECUDA1481489_PBS17_28_RUNTIME_CPU_20261024.json "01:37历史CPU接线快照；尚未采用状态已被E26真实提交取代"
[E06]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/docs/protocols/GR00T_TASK_DATA_FIRST_FROZEN_WBC_SCOPE_V1.md "冻结WBC／任务数据优先"
[E07]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/docs/handover/2026-10-23/WDR1_EXISTING_WBC_INTERFACE_AND_TEACHER_SOURCE_SEPARATION_GOAL.md "源与目标依赖纠正"
[E08]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/parallel-science-20261022/DATA_ASSET_AND_PURPOSE_INDEX.json "数据资产与用途"
[E09]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/WDR1_TARGET_ACTUAL_CURRENT_COUNTER_AND_CLOSED_B2_COSTS_20261023.json "目标实际零时间计数"
[E10]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/WDR1_SOURCE02_FINITE_NATIVE_PROGRESS_AND_B2_CLOSED_COSTS_CPU_ACTUAL_20261023.json "Source02实际有限执行"
[E11]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/parallel-science-20261022/VANDA_ONLY_TRAINING_EXECUTION_DIRECTION_20261023.json "Vanda-only用户方向"
[E12]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/20261016T191445Z-localpc-mp1/FD1_DIAG/FD1_FINAL_ROOT_CAUSE_REVIEW.md "FD1失败诊断"
[E13]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/20261018T171916Z-localpc-fd2/FD2_FINAL_SCIENCE_REVIEW.md "FD2配对Teacher余量"
[E14]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/CR2/20261020T160340Z-localpc-cr2/CR2_FINAL_REPORT.md "R1表示与数据"
[E15]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/scripts/pi0294/pv1/sampler.py "A/B采样设计"
[E16]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/scripts/pi0294/pv1/export.py "实际提交监督与导出"
[E17]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/scripts/pi0292/train_adapter.py "输入隔离与相对动作"
[E18]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/scripts/pi0291_arx5_data/scene.py "固定基座源场景"
[E19]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/WDR1_TARGET_TIME_ACK_EXISTING_RESIDENT_CPU_ACTUAL_20261023.json "目标ACK CPU绑定"
[E20]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/WDR1_TARGET_MEASURED_OBSERVATION_EXISTING_QUERY_CPU_20261023.json "观测与查询CPU绑定"
[E21]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/WDR1_EXISTING_WBC_AND_TEACHER_PROVENANCE_TASK_20261023.json "后续精确CI及状态补记"
[E22]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/docs/protocols/DEC1_CONTROLLER_DISJOINT_VLA_SCIENCE_GO_20261021.md "DEC1科学协议"
[E23]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/parallel-science-20261022/EVALUATION_AND_CLAIM_PREPARATION.json "评估与预算边界"
[E24]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/parallel-science-20261022/B2970_CLOSEOUT_STAGING_AND_TRAINING_PERFORMANCE_ACTUAL_20261023.json "实际训练性能"
[E25]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/parallel-science-20261022/PV1_ACTUAL_MP1_LOADER_WORKER_FORWARDING_CPU_20261023.json "worker转发CPU资格"


[E26]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/5d9f2a82a6aa4830481f7197298f666cdc4aad0c/reports/PI0294/design/parallel-science-20261022/VANDA_PRECUDA_RETRY_PBS17_28_UNIQUE_GPU_SUBMISSION_ACTUAL_20261024.json "1481574实际采用、唯一提交、R与MODEL_START放行"
