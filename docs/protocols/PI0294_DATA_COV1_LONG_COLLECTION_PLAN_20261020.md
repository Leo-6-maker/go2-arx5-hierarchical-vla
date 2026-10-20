# DATA-COV1：利用 CoreLab 建设下一版共享任务空间策略数据

日期：2026-10-20，Asia/Singapore。Plan ID：`PI0294_DATA_COV1_LONG_COLLECTION_V1`。
状态：**PLAN_PUBLISHED_AWAITING_EXPLICIT_DATA_GO_AND_NATIVE_ADOPTION**。
用户本轮要求是发布详细长程采集计划；此发布不等于新物理运行已经获准或启动。既有 D1–D4 的 `NOT_AUTHORIZED` 历史记录保留。下文提供可一次性批准的有限数据工作包；若只批准 Pilot，则后续扩采仍不获准。计划、预算批准、现场绑定、实际执行、数据 READY 是不同状态。

## 0. 目标、已有证据和主线保护

目标不是无限增加同类视频，而是解除当前训练数据的方向/场景混杂，生产有完整物理、时序和动作编码资格的 TRAIN-only 数据，并在累计24、72、144个新几何组时交付固定版本。优先支持同一个 GR00T 权重的 T1抓放与T2无抬升推送；不增加任务种类或模型容量。

审核基线：`4b13c60922b1788b1691c73a059dfac4c628f2a5`。
D0真实报告：`reports/PI0294/PV1/20261018T182956Z-localpc-pv1/FUTURE_DATA_D0_ACTUAL_SUMMARY.md` 及同目录 `FUTURE_DATA_D0_ACTUAL_AND_QUALIFICATION.json`。科学回读：Issue #11/6284522342。

- 原44条未编码记录中1条raw审计失败，直接保留排除；43条补审得到32 PASS、11 codec拒绝。PASS为16条T1/4mm、13条T2/4mm和3条T2/12mm。
- **29条新增原参考合格记录**可使未来候选并集达到168 episodes、93几何组、75双任务配对组；这只是资格并集，**尚未组装、封存或用于训练**。原PV1139条银行不动。
- 所有新增组都是F2/+Y；原参考T2/-Y仍16组，F0/-Y和F2/-Y缺口没有解决。全部273条历史attempt最终分类192条codec/官方链合格、79条codec拒绝、2条raw排除；192包含24条12mm行为版本，不能当192个独立布局或直接混入原参考银行。
- D0工作CPU381.0051秒、原scope控制费用和关闭租约均已结算，不重做、不重复收费。原43条补审及68条旧拒绝的所有结果不可覆盖。
- 本次最新已发布B0快照仍12/48、A/B更新0；`policy-b0-09 / 677540.hopper-m-02`为最后观察的Q等待H200。新owner必须刷新现场，不能把快照当实时；该作业和原PV1 B0→匹配A/B→DEV→合格LOCKBOX路径优先。

**冻结项**：WBC代码/权重/gain/cost/solver、腿策略、机器人/夹爪质量/惯量/摩擦/PD/力限、原S1与参考门、任务score、相机标定、原4mm任务Teacher版本、固定红色立方体和蓝色目标语义。原PV1数据manifest `60f7c35a05766fe23974eba3a72f53ca4db5545a3768e6c1cfea164ab4e58294`、统计量 `fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7`、采样器、源胶囊、预算、截止和选模规则均不修改。新数据不热注入任何当前PV1子任务。

## 1. 权限与唯一工作入口

读取当前 `AGENTS.md`、`START_HERE.md`、原PV1协议/资源卡、统一入口和workspace enforcement，以及#11/#47最新实际回执。复用 `~/Projects/GR00T-WBC` 的已部署入口/doctor/registry；不新建顶层工作区、不猜私有路径、账号或session。

发布后默认只登记方案。执行前需要明确的数据采集GO引用、批准级别 `PILOT_ONLY` 或 `FULL_CONDITIONAL_CAMPAIGN`、站点/CPU/存储资源及全局时钟核验、新run manifest/lease/输入输出绑定。若这些缺失，报一次 `BLOCKED_DATA_GO_OR_NATIVE_CAPACITY`，不挪用PV1未花完额度、D0已关闭lease或本文件中的建议预算。

新命名空间：`PI0294/DATA_COV1/TRAIN/...`；输出在已有registry允许的独立run根。新数据账与PV1分开，资产引用不能双计费用。CPU与原B0预约核互斥，原唯一local-PC owner统一调度；独立采集进程不是第二位自主协调agent。

**全程一次性批准后**，达到预先固定阶段门且剩余全部资源仍可支付时，可自动进入下一段，不为每episode再请求GO。不满足数据门则停止扩采并交付已有结果；不得临时修改门槛使阶段通过。

## 2. 科学设计：真正交叉方向、距离与外观

原F0/F1/F2混合了空间和视觉定义，本版不沿用其隐式方向规则。引入独立的 `appearance_profile`、`direction`、`distance_bin`，保留legacy family仅作来源说明，不重新定义历史结果。

| 因子 | 固定设计 |
|---|---|
| 外观AP0 | 原中性桌面/地面/光照，对应原F0的视觉基准 |
| 外观AP1 | 第二个固定中性色调：table RGBA=[0.60,0.62,0.66,1]，floor=[0.70,0.68,0.65,1]，light_scale=1；只是新视觉配置，不声称原F1就是这种外观 |
| 外观AP2 | 沿既有F2范围：table RGB各[0.50,0.76]、floor RGB各[0.54,0.78]、light_scale[0.8,1.2]；具体值按种子预冻结 |
| 方向 | 每一种外观、每个距离层都包含+Y和-Y，不能由family或结果推导方向 |
| 距离 | 近距[0.12,0.15]m；远距[0.19,0.22]m，定义为source-to-goal平面欧氏距离 |
| 几何范围 | source_x[0.425,0.475]m，中心y[-0.025,0.025]m，goal_dx[-0.012,0.012]m，cube_yaw[-0.16,0.16]rad |
| 不变项 | 同一物体、夹爪、4mm任务参考、相机外参/内参与两条原任务语言；不加入相机扰动、新物体、障碍物、新抓取姿态或12mm参考 |

以上是**拟验证的范围，不是已经可达/可接触合格的场景**。给定距离d、横向偏移dx和方向符号s，定义 `dy=s*sqrt(d*d-dx*dx)`；source=(x,cy-dy/2)，goal=(x+dx,cy+dy/2)。因此距离是欧氏距离，不把Y投影当总距离。实际可达性必须通过原静态IK、碰撞和双摄可见性检查。

共12个设计层：3外观×2方向×2距离。每个几何组配对T1/T2，共享几何和具体外观；任务命令可不同，但不可伪造相同模型输入或实测轨迹。AP0/-Y、AP2/-Y是本轮优先补齐的交叉条件。

### 2.1 有限候选池与划分

先冻结完整144个TRAIN逻辑槽位，每层12组；独立种子来自 `DATA_COV1/V1/TRAIN/<profile>/<direction>/<distance>/<replicate>/<candidate>` 的规范哈希。每槽最多32个静态候选，先封存全池，再做资格。顺序中只因几何近重复、原静态IK/碰撞/视野失败选下一个候选，不能按Teacher/学生成功挑场景。

几何排重沿原source/goal XY及yaw规则（默认2mm/2度；读取实际固定实现），忽略外观以免视觉变化掩盖同布局；与所有旧TRAIN/C1/FD2/DEV/CONFIRM/PV1 LOCKBOX以及本版已选组排重。只读取旧隔离工具允许的排重几何，不暴露LOCKBOX图像、结果或用于调整设计。

同时可封存下一轮24个DEV和24个新LOCKBOX的**候选ID/几何池**，每层2组，提前隔离以防训练泄漏；本计划不执行它们的Teacher、策略或物理，也不宣称静态通过。它们的动态资格和策略评价需以后独立批准。所有同物理布局、外观变体、后续恢复fork共享group lineage，不跨split。

一个槽的32个静态候选均失败，则记录 `STATIC_POOL_EXHAUSTED`；不再发明seed或缩范围凑数。全Pilot所需槽位未合格时不得启动该Pilot；后续段不足时保留前段银行并停止后段。

## 3. 累计阶段：24 → 72 → 144新几何组

| 数据阶段 | 累计新组 | 各12层的组数 | 本段新增组 | 累计T1/T2首尝试总数 |
|---|---:|---:|---:|---:|
| D1 / P24 | 24 | 2 | 24 | 48 |
| D2 / P72 | 72 | 6 | 48 | 144 |
| D3 / P144 | 144 | 12 | 72 | 288 |

这些是**计划分母和上限，不是承诺获得同量成功示范**。新组+Y/-Y计划数量各半；成功筛选后的分布可能不均，必须如实报告。

### D1：24组Pilot，先检查最容易暴露问题的四次

先运行预冻结的两个组：AP0/-Y/近距/rep0、AP2/-Y/远距/rep0，各T1/T2，合计4次。这四次包含在48内，不额外增加。先单worker完成，逐次检查完整raw/真实任务结果及所有601个H40窗口codec，避免采完48才发现动作不可表达。

四次进入余下44次前须满足：原native安全故障0、4次记录身份/时钟均可审计且没有unknown、两任务各至少一次全准入成功、合计至少3/4全准入成功。原S1预测拒绝仍是有效负例，不是已发生真实碰撞；如果导致此数据可行性门不足，停止余下采集并报告，不重跑失败。

### D1–D3结束后的扩采/READY门（仅数据工程资格，不是policy指标）

| 指标，按计划累计分母 | P24 | P72 | P144 |
|---|---:|---:|---:|
| 每任务原完整物体成功最少 | 20/24 | 60/72 | 120/144 |
| 每任务最终全准入示范最少 | 18/24 | 54/72 | 108/144 |
| T1/T2都全准入的配对组最少 | 16/24 | 48/72 | 96/144 |
| 每个设计层、每任务全准入最少 | 1/2 | 4/6 | 8/12 |
| AP0/-Y、AP2/-Y分别每任务全准入最少（两个距离合计） | 3/4 | 9/12 | 18/24 |
| 实际native safety fault | 0 | 0 | 0 |
| 未对清副作用、混split、错误动作标签 | 0 | 0 | 0 |

全准入=原任务成功 AND native安全成立 AND 独立raw/时序/图像哈希成立 AND 原动作表示/官方codec成立 AND split及来源成立。一个条件失败，不因其他条件通过而替代。每段计划所有格均需有明确状态；NOT_RUN/工程缺失保留，不计成功。

满足数据门还不够：D2/D3必须在批准级别覆盖该阶段、全部剩余资源和原PV1优先保留通过后才能继续。若只有Pilot批准，P24结束即关账。若全程获准，合格自动扩采；遇S1/codec支持缺口先报告机制，不为跨门同时改Teacher、normalizer或采样器。

## 4. 复用实现，不破坏原PV1强约束

当前 `scripts/pi0294/pv1/capture.py` 明确检查PV1命名空间、96组及旧master_plan。新144组**不能通过伪造PV1 ID或删除len==96检查混进去**。

在现有库上新增独立数据scope/薄调度层（具体模块名由owner实现并回读后确定），复用原 `scene/teacher/episode/s1_preflight/audit`、`pv1/export.py`、官方processor/collator和计数器。旧PV1启动器对错误新scope仍须拒绝。代码准备完成不等于原生collector已部署。

新scope测试至少覆盖：无数据GO/错host/超预算拒绝；跨split与重复组拒绝；错任务/错几何/虚假标签拒绝；PV1 manifest未改；计数器零积分原生状态与双摄像素不变；新组不能进入旧PV1 capture；最后一次binder写入后0400/owner/非symlink/hash/schema/request/input-reference检查；完整bank的RGB/numeric成员存在而不是只验证metadata目录；原S1有效拒绝保留；worker崩溃/unknown/硬时钟正确停止。

共享代码变更采用普通提交、精确source SHA的CI成功和冷态部署；不热换正在排队或运行的B0 capsule。CI不运行远端SSH、模型或真实物理。

## 5. 采集、记录与审计流程

### 5.1 并行与优先级

新数据最多4个独立单核worker，按1→2→4阶梯增加；第一阶段4次用1核。每worker独立MuJoCo实例、renderer/context、随机状态、输出、序号和账本；不能共享可变model、相机缓冲或Python monkeypatch状态。使用原唯一owner，不能另开AI子代理或第二NFS写者。

先读取当前B0保留核、CPU/内存/显卡渲染负载、磁盘和用户quota，选择互斥核号，保留现役心跳和动作通信能力。2核阶段的前两个完整任务仍从原计划选，不增加benchmark episode；4核只在数据身份/图像一致性通过、无主线退化且单位核时效率合理时采用。多核更慢则回2核，不放宽时间门。

B0从Q转R或主线压力上升时，停止派发新的未来数据episode；独立当前episode只能在原安全终态完成或按原停止协议保留partial，不能强杀主线或冻结真实物理时钟。若需要立即让资源，按已验证安全停止；partial仍算已尝试、不得从头重复。

这里的“CPU采集”指不使用神经推理/训练GPU和Hopper PBS。原生双摄renderer若实际依赖CoreLab图形GPU，必须登记其context/显存/占用并保护B0；**不能把EGL硬件渲染声称为完全零GPU使用**。不暗换软件renderer或分辨率改变数据分布；无合法渲染资源则等待。

### 5.2 每条episode

原32s、500Hz物理、20Hz双RGB，完整轨迹目标641组实际图像、16,000 live tick；Prefix8的原S1影子检验逐次计费。失败提前停止按实际时长计，不能padding为32s。全部期望参考、实际安装/提交、actual EEF/夹爪/物体高频状态分别保存。

监督为真实当时状态与实际提交的绝对EEF10参考，经原codec形成所需相对动作；不是未来achieved pose、未执行Teacher计划或S1拒绝后的虚构状态。Teacher可用仿真真值反馈，真值仅在audit/标签生成侧，不能加入policy输入。原固定相机RGB、语言、actual EEF/夹爪输入语义保持。

每条完整原episode先做原安全和完整物体任务评分，再做独立raw/image/时钟/提交动作审计；全601个H40窗口逐元素encode→decode检查，原位置和SO3误差门从固定实现读取，严禁近似替代。每个合格parent另做6个固定锚点的真实图像/标签reader/collator、跨episode混batch与确定性repeat。失败不能仅靠总体RMSE遮掩。

对codec拒绝保留：首次错误window anchor/action index/channel、归一化前后值、clipping mask、XYZ/SO3误差、该窗口对应的真实物理事件。相对codec coordinate2不自动等于world-Z；必须用该样本基坐标变换解释。只测量，不重估统计量、不截断修补标签。

### 5.3 真实事件标注

从实际高频物体/接触/夹爪轨迹提取首次接触、抓持、抬升、位移、入目标区、释放/分离、撤离和最终41点保持；事件定义的阈值版本化。原score阈值与新增解释性阈值分开，不能修改成功率。旁观Teacher phase只保留为辅助字段。

逐parent和逐任务报告：H8、完整H40是否包含上述转移；H8及H9–H40分段速度/动作变化诊断；实际pad-table最低间隙、no-lift、目标误差和hold。不要把过去1.1647%的转移窗口占比解释为整个release阶段比例。此次只建立事件索引，不改PV1或未来模型的采样权重。

### 5.4 失败分类与成本

状态至少区分PLANNED/STATIC_REJECTED/NOT_RUN/VALID_TASK_FAIL/REFERENCE_REJECT/S1_PREDICTED_REJECT/RAW_AUDIT_REJECT/CODEC_REJECT/ADMITTED/ENGINEERING_INCOMPLETE/NATIVE_SAFETY_STOP。

一条可有多诊断标签，但总表必须有唯一终局状态和可追溯判定优先序，不能多重计数。所有首尝试固定分母，物理replacement=0；已失败的真任务不换seed，工程中断也不在本预算下重播。仅可幂等重读已保存工件，不新增物理效果。

数据真实但任务失败可入单独transition/failure archive；不能作为成功BC标签。未执行的S1拒绝没有真实下一状态，只保存拒绝分类及此前真实状态。Teacher-only采集不叫on-policy DAgger。本计划不训练世界模型或恢复策略。

## 6. 提议一次性资源包与停止点

**下表都是尚待明确批准、不是可直接支出的新增额度。** 与旧16核时/12小时Pilot草案相比，当前包增加CPU审计/静态候选/封存余量，并给出后续累计扩采上限；旧草案不覆盖或改写。现场更小的quota/CPU/显卡/存储许可胜出。

| 累计限制 | P24 | P72 | P144 |
|---|---:|---:|---:|
| CoreLab CPU core-hours，含所有worker/审计/打包控制 | 32 | 96 | 192 |
| local协调CPU core-hours | 4 | 8 | 12 |
| 自DATA-COV1首次合法采用起累计wall hours | 24 | 60 | 96 |
| 唯一episode首尝试数 | 48 | 144 | 288 |
| live ticks | 768,000 | 2,304,000 | 4,608,000 |
| shadow ticks | 768,000 | 2,304,000 | 4,608,000 |
| 全部integrations（含任何重建） | 1,536,000 | 4,608,000 | 9,216,000 |
| 静态候选constructors上限 | 2,000 | 5,000 | 10,000 |
| episode/计数资格/clone constructors上限 | 4,500 | 13,200 | 26,400 |
| 总constructors上限 | 6,500 | 18,200 | 36,400 |
| 新输出+原件+打包+暂存峰值GiB | 48 | 128 | 256 |
| 新数据网络传输GiB | 16 | 64 | 128 |
| 新Hopper/A40计算GPU小时、PBS、神经query、optimizer update | 0 | 0 | 0 |

这些是累计而非三份相加的预算。静态32候选×2任务×24/72/144组最多1536/4608/9216次构造；动态以现有88/attempt加worker计数资格为规划参考，实际所有失败构造/clone都计费，上限不足就停。未来DEV/LOCKBOX此阶段只建候选元数据，不占真实构造或积分。

每stage开新任务最晚在累计wall截止前4h，物理在截止前2h结束，最后2h只审计/封存/关账；即P24为T0+20/22/24h，P72为+56/58/60h，P144为+92/94/96h。只有阶段达标、完整采集GO覆盖下段且下段预算保留已成功登记，才可越过上一阶段截止进入下一段。排队、PV1优先让资源、重启和换窗口都不重置新T0。

每次派发先保留本次完整资源和最终关账余量；每段前用首批实测最大值加明确保守系数，更新合格episode/CPU核时、每万积分核时、renderer时间、文件数、allocated bytes、逻辑bytes和I/O峰值。估计不是统计保证；无法覆盖剩余必需工作就交partial，不上调硬上限。

D0既有381CPU秒与原PV1控制费不再计入新scope。共享旧RGB硬链接不算新独立轨迹、不算整份物理副本；禁止通过chmod或写入共享inode改变旧数据。专有块、逻辑大小、全盘quota差额分别报告。无可删失败checkpoint就0删除，不能删除PV1父模型/银行/唯一恢复或原失败证据腾空间。

## 7. 分阶段封存与训练接口

候选版本名（未产生）：`DATA_COV1_P24`、`DATA_COV1_P72`、`DATA_COV1_P144`。以raw member/scene/task/Teacher/reference/processor/stats/source为主键，每级引用以前固定成员加本段新增合格成员；不移动/覆盖以前READY、不把新数据加到PV1既有manifest。

每级另生成可选 `ORIGINAL4MM_UNION` 资格清单：现有139+合格D0原参考29+本级新合格；24条历史12mm替代动作留单独库，不混成覆盖数据。旧+新合并后仍可能方向不均，必须报告而不能称整体50:50。

若全部新尝试均合格，原参考并集的**理论上限**为：

| 数据级 | episodes | 不同几何组 | 双任务配对组 |
|---|---:|---:|---:|
| D0资格并集基准，尚未封存 | 168 | 93 | 75 |
| 加P24 | 216 | 117 | 99 |
| 加P72 | 312 | 165 | 147 |
| 加P144 | 456 | 237 | 219 |

真实数量以去重后的准入成员为准，上述不是保证。新物理数据必须与全部旧组去重后才使用该加法。

每个银行依次经过：RAW_CLOSED → INDEPENDENT_AUDITED → CODEC_ADMITTED → FROZEN_LOCAL_MANIFEST → PACKAGED → FIXED_REVISION_FULL_READBACK → READY_LAST → DOWNSTREAM_NATIVE_CONSUMPTION_QUALIFIED。远端hash成功不代替后续训练环境消费；不能只复制metadata后让RGB指向空目录。Hopper不再重复下载既有相同blob，只有合法关闭、只读、来源hash明确的库可复用。Hugging Face或其他已授权存储按实际隐私和固定revision处理，仓库名称不能证明公开权限。

小型源码、协议、计数/排除原因及hash上Git；raw图像/视频/未公开权重/私有WBC与认证不进Git。视频是索引和复核材料，不替代原帧、物理和真实动作标签。

## 8. 每段必交收据与自动决策

建议原有注册run输出以下小型报告，不另造执行体系：
- `DATA_COV1_ADOPTION_AND_SCOPE.json`：批准引用与级别、T0、原单owner、PV1状态、互斥CPU/渲染/存储预算、源CI和实际部署版本。
- `DATA_COV1_FROZEN_POPULATION.json`：有限candidate池、坐标/外观/方向/距离、split/排重、stage membership及首四次准确ID。
- `DATA_COV1_ALL_ATTEMPTS.json`：全部计划、静态失败、未跑和实际attempt；raw/score/S1/codec/控制作用hash、唯一终局与多诊断标签。
- `DATA_COV1_COVERAGE_AND_EVENTS.json`：12层×task的计划/尝试/任务成功/完整准入、独立组/配对组、真实事件窗口覆盖及first-codec-error明细。
- `DATA_COV1_P24_GATE.json`、`P72_GATE.json`、`P144_GATE.json`：逐门实测/阈值/缺失/是否可扩采；不把数据门当policy成功率。
- `DATA_COV1_DATASET_CARD.md`、`DATA_COV1_READER_REPRODUCE.md`、各级成员manifest/READY及`DATA_COV1_RESOURCE_FINAL.json`：排除和选择偏差、来源与数值版本、读取/未来训练限制、真实CPU/render/tick/构造/字节/租约收口。

若主要失败是codec且新场景物理成功：停止规模扩张，提出独立TRAIN-only统计量/可表达动作范围对照；本计划不重估normalizer。若Teacher动态失败：局部任务参考研究另行版本化，不修改WBC或原判分。若新组准入充足：封存并允许未来受控数据规模比较，不宣称学生已泛化。

PV1获得H200后按原B0→A/B执行，**不等待P144**。未来学习曲线先比较同processor/同父/同更新/同lane配比的P24/P72/P144，按独立episode和scene支持统计；样本数增加与训练步数增加属于不同变量。事件均衡、12mm参考、历史输入、WorldModel、真正on-policy DAgger及更大backbone均保持后续单独科学决策，不能同时混成“多数据更有效”。

## 9. Owner收到本计划后的首次动作

先回报 `DATA_COV1_PLAN_RECORDED_AWAITING_DATA_GO`，核对是否已另有明确的新采集批准。没有批准则仅在已有合法CPU准备范围内整理准入缺口，不启动MuJoCo构造、物理、新租约或新增传输；如完全没有可用准备范围也只记录计划。

有一次性批准后，完成新的正式scope/manifest/lease/site和互斥资源绑定，才可报 `DATA_COV1_NATIVE_ADOPTED`；然后预冻结、静态验证、首四次、P24 gate、条件P72/P144、分段银行交付及关账。原已关闭D0永不复活。

数据科学结果登记#11；CPU/渲染/同步/部署/资源与原PV1不受影响的实证登记#47；每个阶段和真正异常合并报告，不新增永久automation或第二监控owner。本次发布者只审核和发布计划，新增PBS、神经推理、训练、真实物理及远端删除均为0。
