# Core＋Vanda长程推进：BOOK受控对照、共享M32交付、独立监督改进与离线目标适配

日期：2026-11-03（Asia/Singapore）。仓库核对基线：`75e47deb01e32eb49df12c83de1e0ec55c0e8388`（PR337）。

**状态：REVIEWED_EXECUTION_HANDOFF_NOT_DEPLOYED_BY_THIS_DOCUMENT。** 本文把用户本轮六项路线整理为现有执行agent可采用的有限任务图，不提供新的调度器，不宣称已经登录Core/Vanda、切换owner、提交PBS或产生模型。机读任务卡在 `reports/PI0294/design/core-vanda-policy-delivery-20261103/TASKS.json`，是交接清单，不是可以直接送入Night10的运行配置。

## 0. 结论与证据边界

认可用户的主分工：**Core保留BOOK对照；Vanda训练第一版四任务共享M32；监督改进独立准备；Go2适配仅离线。** 最终研究目标仍是GR00T输出EEF轨迹／夹爪目标，由冻结WBC执行。

### 0.1 仓库已核对的事实

- PR337最新公开原生记录截至11月3日11:45 SGT：D2为3266/4096、104512次访问，1024/2048完整恢复；M13及其D2→PB_CONTROL→PB_BOUNDARY队列未改。新增Core接续已真实驻留，原停止新派发时刻为 **2026-11-06T11:43:56.987405+08:00**。[S1]
- 六BOOK候选的新RESET评价与Core raw native10/native16／条件shared8192已有待执行绑定；当时新增有效条件和共享端点为0。关闭Core共享生产节点须针对真实最新状态，不能凭旧PENDING记录操作。[S1,S2]
- 共享采样源码已经固定raw父、46完整专家episode＋13专家后缀、34563窗口、M32 77847684参数、每batch各任务8次、microbatch1×accum32、8192步、warmup256、LR1e-4、WD1e-5及完整端点。[S3]
- **现有Core共享GPU资格代码绑定LOW_RESIDENCY_POLICY，并要求实际CPU copy/restoration见证。** Vanda的no-activation-offload入口不是把同一张证书搬过去即可，须有显式的新内存执行版本、同allocation实测及完整恢复，保留科学配方不变。[S4,S5]
- 原相机／独立reset资格已通过，不重复把旧/proc问题当成未解决障碍；资格只覆盖相机、初态和前驱故障，不代替模型和完整任务成绩。[S2]

### 0.2 用户本轮新增执行报告，需接入原件但不否定其内容

用户报告：Vanda仍在交付输入／入口验证，未切owner、未新qsub、共享更新0；21条BOOK原始记录审计完成，其中一条成功Teacher最小实际关节裕度约4.80mrad；接口＋末6层DiT约2.13亿参数，仅清点未训练；拟冻结12个TRAIN_VNEXT候选。上述作为**USER_REPORTED_CURRENT_STATE**使用；执行agent将其具体source、日志或清点结果绑定到原账本即可，本文未独立读取这些尚未发布的私有原件，不把它们升级为本次服务器实测。

这份4.80mrad记录说明值得检查监督裕度，但不是Teacher导致全部joint4失败的单独因果证明，也不支持删除该轨迹或放宽安全阈值。约2.13亿保留用户口径，不用估值替代最终tensor白名单的精确计数。

### 0.3 高效利用的必要补强

1. 分开**科学配方**与**机器内存执行方式**；A40不卸载激活可以测试，不能隐式改变dtype、batch、loss、样本或参数范围。
2. 完成一次可恢复的跨站唯一owner交接，不以两个站点同时训练同一正式baseline来“提高利用率”。
3. 建立模型交付→Core推理资格→DEV68的接续，不能只挂8192训练、不挂后续消费。
4. 8192为预声明主评价端点；2048/4096/6144保存恢复和固定TRAIN诊断，**不默认各重跑68条件、不按结果挑端点**。
5. 源域采集需真实物理和双RGB；CPU静态搜索不是新增合格VLA数据。Core源域物理owner唯一，不与既有评价争写。
6. 复用同一entry／Journal／onceIO／registry及现有Trainer/evaluator，验证仅覆盖变更接缝；不再造调度系统、预算分层或重复全文校验所有不变资产。[S6]

## 1. 科学不变量与本轮唯一共享baseline

### 1.1 固定共享配方

|项|固定值|
|---|---|
|父模型|现有固定revision的原始预训练GR00T；从raw重新装载，而非全网络随机初始化；不继承book128/b248/Phase权重|
|TRAIN|共同59：T1 10完整、T2 10完整、BOOK 21完整＋13后缀、DOOR 5完整；合计46完整＋13后缀／34563窗|
|状态与图像|当前双RGB、实际EEF与夹爪、语言；state_history_length=1|
|动作|EEF10/H40、Prefix8、原4步生成、原连续夹爪语义|
|参数|category10接口＋DiT30/31，42个可训练张量／77,847,684参数；其他冻结|
|采样|每个有效batch四任务各8次；任务内条目均衡、窗口循环；固定seed与完整序列身份|
|优化|microbatch1、accum32、effective batch32、fresh AdamW、LR1e-4、WD1e-5、8192余弦日程、warmup256|
|访问预算|262144次；每任务65536次；重采样下不假称所有窗口有相同epoch|
|保存|2048/4096/6144/8192的模型、optimizer、scheduler、RNG、采样游标及父模型身份；8192主端点|
|统计|仅现有TRAIN共同统计；不重拟合、不混TRAIN_VNEXT或DEV/TEST|
|首选机器执行|Vanda合法A40 allocation，先验证不卸载激活到CPU；不承诺速度或显存必然足够|
|资格|同一实际GPU allocation内原native10＋42项完整恢复＋原共享推理资格，再从raw fresh启动正式8192；不继承资格更新|

BF16 autocast、参数存储精度、冻结类别表存储位置等沿已审计契约显式记录。“不卸载激活”不等于偷偷将整个模型转换成BF16，也不自动要求所有冻结表占据GPU。关闭activation offload与冻结buffer放置分别命名；保持有效值、训练范围及官方计算语义。

### 1.2 不改写既有研究

Core健康D2/PB、Phase/D0/D1原评分和STOP不改；原E1/E2仍是raw父／12示范／R0／512-of1024定义，不以新共享结果冒充。共享baseline不等BOOK胜出、PB全通过、TRAIN_VNEXT采集完成或Go2安全解除。WBC权重、增益、成本、solver和腿策略冻结；S1/T2自由机身隔离继续；不执行实机、不增加EEF13或BASE监督。

## 2. 资源分工与排程

|位置|本批职责|避免的浪费|
|---|---|---|
|Core原M13|保持D2→PB_CONTROL→PB_BOUNDARY原资格／正式4096／恢复／关闭|不热改、不迁移活动CUDA、不重复提交|
|Core既有72h接续|六模型统一BOOK_RESET、后继共享checkpoint消费、单一源域物理任务|按既有全局神经／物理锁调度；不是有两卡就放行两个模型|
|Vanda PBS allocation|共享输入验证、独立no-offload资格、唯一共享8192、完整保存交付|不占Core神经owner；不把公开可访问或旧资格当当前allocation|
|Core/管理侧获准CPU|局部数据审计、几何/关节裕度搜索、VNEXT索引、Go2离线变换/碰撞检查、失败报告|先低优先级1worker；实测主存/内存带宽和D2吞吐再决定并行，不机械开满核|
|Core源域数据采集|12候选中实际静态合格者，安全资格后首次尝试|需要physics/RGB时进入原物理队列；不得标成CPU静态检查已经采到数据|

新待启动任务的排序：**已有合格checkpoint的预声明完整评价 > 必要输入/模型资格 > 新源数据采集**；同一正式面板按固定顺序完成，不能按成绩换模型。此排序只适用于新冷态可调任务；不改现役M13或活动面板，不因新计划抢停健康训练。

## 3. 任务图与有限交付

|ID|任务|位置|必要前置|交付|
|---|---|---|---|---|
|W00|刷新实际状态与输入索引|CPU|现有合法控制通道|旧E2准确状态、Core三个共享节点、活动writers、冻结版本|
|W10|Vanda输入与no-offload入口准备|Vanda CPU/合法构建环境|W00|raw7＋59数据/RGB/统计/DEV68完整可读；内存执行新版本|
|W11|共享唯一owner冷态交接|既有Core/Vanda owner|W10、Core共享零启动证据|Core RAW_NATIVE10/RAW_NATIVE16/SHARED8192_ADOPT关闭为移交；Vanda唯一授权身份|
|W12|同allocation原生资格|Vanda GPU|W11、站点资源/PBS合法|native10、完整恢复、原共享推理资格及实测速度/峰值|
|W13|shared-M32 fresh8192|Vanda GPU|W12 PASS|主端点及全部保存/恢复，不继承资格状态|
|W14|固定policy包交付Core|CPU/已授权传输|W13完整终态|原始父/训练delta/Processor/统计/解码与SHA、Core独立重建|
|W15|Core本地推理资格＋共享DEV68|Core GPU/物理|W14、现有原生相机/reset、具体checkpoint本地资格|逐任务结果、失败机制与模型包；零成功仍完整交付|
|W20|现役BOOK三训练|Core原M13|既有资格|D2/PB端点或精确闭合失败|
|W21|六BOOK同协议评价|Core GPU/物理|各自完整CLOSED端点＋具体checkpoint推理资格/reset|原固定六模型×DEV10，原评分不回填；缺模型记未运行|
|W30|Teacher裕度与接触审计|CPU|既有原始TRAIN及D0/D1诊断记录|21条BOOK审计闭合到原件、阶段裕度/接触摘要|
|W31|固定12个VNEXT候选及静态审核|CPU|W30|4 BOOK几何＋4 BOOK恢复＋4 DOOR几何，准入/拒绝理由|
|W32|合格候选首次物理采集|Core单物理owner|W31该候选合格＋真实运行资格|每候选最多一次首次物理尝试，总计不超过12；全部attempt保留|
|W33|TRAIN_VNEXT独立冻结|CPU|W32各自实际结果|完整示范、后缀、失败分库；不加入当前baseline|
|W40|Go2离线适配包|CPU|现有冻结WBC/机器人/相机资产可读|坐标/时钟/全身碰撞/稳定站立EEF10与移动后操作静态接口|
|W50|末6层清点报告|CPU|raw tensor headers可读|精确白名单/numel/内存估算；本批GPU更新0|
|W90|综合进度与终局发布|CPU|可读取任务状态，不要求全部成功|真实端点、每任务行为、资源瓶颈、待运行与下一假设|

W30/W40/W50不等待W12或BOOK成绩；W12/W13不等待PB成功；W33不进入W13；W15不依赖W32成功。W20/W21与Vanda训练互不依赖对方活跃心跳。跨站只在明确输入交付／owner交接／输出消费边界核验。

## 4. Vanda最短落地路径

### 4.1 先核旧作业，不无限追溯

只核原项目E2作业 `1508345.stdct-mgmt-02` 的当前／终态、实际owner、是否有计算效果和现有效期限；保留已发生的训练/失败。健康运行不取消，不复制qsub；自有未启动且已失效的旧作业，按原入口完成终态后才能替换。旧E2不可恢复时输出明确状态，不把其成功设为新共享训练前置。没有实际allocation不得在登录节点跑模型。

### 4.2 W10：一次准备，共用固定输入

核对完整raw父七文件、数值数据、实际两路RGB、59条目来源、共同statistics、完整68 DEV定义及训练源码。不是只有manifest就算交付。已可信不可变对象复用校验；新增传输／变更对象全流校验一次。新包建立只读快照，不重复复制整个历史目录。

现有 `shared_training.qualified_entry()` 与 `native_relay.shared()` 强制LOW_RESIDENCY_POLICY。[S4,S5] 因此复用现有Trainer/loss/checkpoint实现，增加一个**明确Vanda内存执行profile**；不用删门或伪造copied_tensors通过旧合同。原Core卸载资格照旧有效但不冒用。原生验证覆盖有效输入、可训练参数、同种子前向/梯度/更新、保存恢复和峰值；跨GPU不同算子允许按预先声明的数值容差检验，不承诺位级相同，也不因设备变更把科学配方改掉。

### 4.3 W11：最小唯一owner交接，防重复与双端空转

准备完成后，使用原registry/onceIO和原读取SHA：

1. 再读Core三个共享节点均未启动、无GPU模型/更新/PBS副作用、无活动writer/UNKNOWN；仅凭PENDING文本不够。
2. 通过现有owner在冷态取消其未来派发资格，并记录 `CLOSED_UNSTARTED_TRANSFERRED_TO_VANDA`。不删除旧状态，不关闭BOOK评价，不重启D2/PB。需要替换CPU relay时保留原截止、已完成工作和其他任务。
3. 在同一既有身份体系登记Vanda唯一共享producer的source/data/recipe与已关闭前驱证明，再提交一次有限PBS。
4. Vanda未能启动时，Core该臂仍明确为已移交／待恢复，绝不自动“保险起见”两端同时运行。任何回退须先核清Vanda未启动或终态，再以新的冷态owner恢复，数据配方不变。

这是一项跨站交接，不建设新的分布式事务平台。别把Core全局神经锁传到Vanda，导致Vanda还要等D2/PB结束。

### 4.4 W12–W13：资格与正式训练

同一实际GPU allocation内完成原native10（320访问、诊断及42项完整恢复）、适用共享推理资格；测allocated/reserved/free、CPU RAM、读写与步时。2GiB等原余量要求如属于当前原生合同应保留，不能只按卡标称48GB放行。

资格权重、AdamW和RNG不继承。显式清理本任务资格模型的对象引用，确认资源恢复后重新装载raw、fresh optimizer与正式种子，启动8192。这里fresh不是随机重置预训练骨干。先核PBS walltime是否覆盖实测计算、四次checkpoint、恢复/导出及有限收尾；不足时明确阻断/另设合法连续恢复合同，不能暗改总步数或声称中间点是8192。

没有通过no-offload资格不立即回头扩参数或降低图像分辨率。先定位原生容量/入口；需要另一内存执行profile时作为新的明确版本验证。资格失败是资源/工程结果，不是四任务policy失败。

## 5. Policy交付与评价：训练结束后不能再断链

### 5.1 一个正式主端点，不默认四倍评价量

首版共享**8192为预声明主端点，正式DEV68只跑一次**。2048/4096/6144为完整恢复与固定TRAIN诊断端点，不默认每个各跑68，即不自动变成272次物理条件。若将来确需学习曲线行为实验，开跑前另固定辅助端点/人口/预算并披露开发选择；不能看结果挑最好端点替换主端点。

训练未达到8192但留下合格中间点时，交付“部分训练端点”，不伪称主端点完成。后续是否评价中间点需明确新诊断身份，不悄悄改变本baseline定义。

### 5.2 最小跨站推理包

传给Core的是重建该确切policy所需的父引用、训练delta或完整推理权重、官方Processor/配置、统计、输入/动作契约、源码与模型SHA、数据manifest、trainable白名单和完整恢复证明。optimizer/scheduler/RNG完整训练档案保留在Vanda已授权持久位置，不必为推理反复传输所有大型优化器文件；父模型已在Core且身份可信时只补缺失项。

Core必须真正重建该8192模型，核固定输入/噪声的生成与解码一致性，并获得**当地该checkpoint的原生推理/容量资格**。Vanda raw的native16不能证明训练后8192在Core可直接运行；也不能将book专属native96资格一概套到共享入口。具体采用共享已有资格合同，不新造重复证书。

### 5.3 68条件的固定执行与统计

按原封存顺序T1 24、T2 24、BOOK 10、DOOR 10；一次模型—条件首次结果保留。独立process/reset、原安全门、原评分和噪声不变。有效安全STOP计入有效失败；工程失败、未知和未运行分别列出。只有前一任务真实CLOSED且独立复位通过才继续下个条件；无法复位就暂停受影响物理支线，不阻断独立训练或只读CPU。

结果至少包括：task/condition、checkpoint、attempted/valid/unrun、success、first_failure_stage、hard_safety_type、grasp/transport/support/release/withdraw、requested/installed/actual参考身份和物体结果。每任务分别给分子/有效/预定覆盖；完成全人口时才直接给成功/24或成功/10。macro与最弱任务均报告，不用BOOK窗口数量或混合平均掩盖其他技能。

现有DEV已暴露，68条件不是最终TEST，也不自动证明跨场景泛化。零成功同样交付负结果与错误分类，不换seed、降阈值、删静态困难条件或反复挑样例。成功学生DEV轨迹不进入专家TRAIN。

### 5.4 发布包最低内容

`POLICY_MANIFEST.json`（确切权重/父/统计/解码/输入版本）、`MODEL_CARD.md`、固定68行的`DEV68_RESULTS`、逐任务汇总、`FAILURE_ANALYSIS.md`与必要Rerun引用。私有原始媒体不新传到公开GitHub。已过期/未准入任务仍在结果表中，不用文件存在代替完整恢复。

## 6. TRAIN_VNEXT：提高信息量，不打断现役对照

### 6.1 W30：复用21条审计，不从头重扫所有历史

用户报告的4.80mrad最小实际裕度先绑定episode/phase/tick/原日志和实际模型限位，区分：实际关节、命令IK解、requested EEF、installed目标、实际跟踪、速度/接触。现有D0/D1诊断只作为设计假设，不能把DEV行为/专家修正直接加入TRAIN。

按真实阶段对齐，报告每关节最小值、低裕度持续段及同段跟踪误差。Teacher路径靠近限位只是风险证据，不自动删除成功示范，也不调WBC/源控制器增益、关节限位或安全容差。

### 6.2 W31–W32：12候选一次首尝试

保留用户固定4＋4＋4：

- 4 BOOK几何／裕度：改变任务可执行路径与抓握/入格几何，检查全路径IK连续性、全臂/手指/场景碰撞及速度；不得用反向控制偏置掩盖跟踪误差。
- 4 BOOK接触事件恢复：起点来自合法独立TRAIN学生状态或明确版本的扰动实验。两者分开命名；扰动状态不称on-policy DAgger。只有真实专家执行才形成标签。若需要新的GR00T rollout，则它是后续GPU查询，不冒称CPU生成。
- 4 DOOR独立几何：仅限已支持的源机械范围，区分位置变化与新机械配置；不得借资产名称声称上游原benchmark成绩。

先固定12个ID/seed/几何/来源/预期机制/静态准入标准。静态拒绝计入候选表，不换seed凑12成功。每个静态合格候选最多一次首次物理尝试，累计≤12；原物理owner、camera、安全与数据门不变。不新增自由机身Go2尝试。

采集是模型/Teacher＋实际500Hz物理/20Hz命令与双RGB的原生过程；CPU静态路径、将未来achieved状态重标、未执行H40尾部都不能成为专家标签。无完整记录/任务成功的尝试保留失败bank。完整成功episode与合格专家后缀分别准入；其他局部安全段不默认为成功BC监督。

### 6.3 W33：独立冻结并等待baseline结果

TRAIN_VNEXT使用自己的manifest/统计候选、来源和分区，复用原Rerun。当前D2/PB/共享59统计不变。后续数据收益实验从同一个明确父模型、相同参数范围和匹配预算比较旧数据与扩充数据；新统计若必需，比较臂采用可解释的匹配版本。

## 7. Go2离线适配与容量：本批不增加物理／训练

W40只检查完整`world→base→arm_mount→flange→TCP`、两相机外参/附着、SO(3)/rot6d约定、观测/提交/安装/执行时钟，以及包含四足的真实碰撞几何。保留旧已验证结果，补当前EEF10版本差异，不从头重复无关WBC诊断。

交付稳定站立EEF10接口和移动→稳定→操作的BASE软位姿**静态可行性**；每个chunk坐标锚定符合既有契约，不使用未来机身姿态或物体真值偷给学生。关节解仅规划见证，不直接送关节绕过WBC。静态通过不证明安装、跟踪、平衡或完整任务。S1/T2隔离不解除，不改WBC、不训练EEF13、不填零伪造BASE标签。

W50仅复核接口＋末6层的tensor名单和精确参数量；用户约2.13亿保留为待原件绑定值。原生末6层native10必须等共享M32主端点与逐任务评价后另作决定；本批不自动启动13亿E2或新的history/grip-loss/LoRA分支。原历史E2可依法核账，仍不成为共享主线前置。

## 8. 故障隔离与长程接续

|终态|处理|
|---|---|
|有效policy失败/安全STOP且完整CLOSED|保留负结果；独立任务继续；下一物理条件仍需原生reset|
|已核清工程失败|保留错误和已发生效果；修受影响接缝后以新版本进入合法任务，不无限原样重试|
|GPU/主存/磁盘不足|有限等待；独立CPU或另一已合格站点继续，不强杀他人进程、不隐式改配方|
|特定设备异常|隔离该设备；不移动活动CUDA，其他已验证独立资源按既有owner继续|
|UNKNOWN/活动writer/不明qsub|暂停受影响写入与资源域，先独立核清；不重复提交、不伪造CLOSED|
|依赖无端点|只阻断该checkpoint的评价；不把未训练模型计零成功，不阻断另一独立臂|
|跨站传输失败|原训练端点保留，Core记录待交付；Vanda其他独立有限任务不依赖Core实时心跳|
|派发截止|不接纳新任务，已合法启动作业按自身有限合同收尾；未运行清单完整交付|

科学结果、进程生命周期和副作用关闭三个字段分开。M13已测已知CLOSED失败续跑的事实复用，不额外宣称所有新跨站分支已实测。源安全阈值、真实资源quota、目标隔离以及未知副作用边界不绕过。

## 9. 截止、验收与后续决策

当前新派发截止保持 **2026-11-06 11:43 SGT**（执行时保留既有原件完整时间戳11:43:56.987405）。本文不重置72h、不新增未来重复运行授权。已开始任务按照各自有限allocation/运行合同保存与关闭；排队作业若有实际开跑截止仍遵守。新PBS需要在接受前估计端点保存和收尾能否装入合法walltime，不承诺所有工作在窗口内完成。

验收按科学里程碑，不按PR数量：

- **M1 交接**：Core三个共享节点确实未启动并已移交；Vanda有唯一合法run/PBS或明确阻断；BOOK健康不变。
- **M2 原生能力**：A40 no-offload新版本的native10/42恢复/推理与性能实测；不是CPU测试通过。
- **M3 训练交付**：一份确切共享8192主端点及完整恢复；中间点、失败或中断单列。
- **M4 行为交付**：同一权重在DEV68的逐任务真实结果；缺条件精确说明，不把零成功隐去。
- **M5 监督与目标准备**：12候选静态与实际尝试表、TRAIN_VNEXT分库、Go2离线适配包、末6层精确清点；不与policy能力混写。
- **M6 发布与决定**：模型包＋结果表＋失败分析。若共享出现能力但不稳，按失败状态选数据干预；若可观测信息不足，再单列历史对照；若共享表示/拟合确有瓶颈，再考虑容量。每次只选一类干预，不自动同时更换数据、采样、网络和loss。

可用policy仍需要冻结候选后的独立确认与无硬安全证据；一次成功或DEV68完成不等于目标整机迁移。Goal V7每任务20条件/目标16成功等后继工程确认标准保持其原性质，不作为本批承诺或总体成功概率保证。

### 最先三个实际动作

1. 读取最新Core/Vanda状态；验证用户报告的新审计/参数清点并绑定原件；保护M13、确认Core共享三节点未启动。
2. 完成Vanda固定输入与no-offload最小应用接线，在cold handoff后获得真实PBS/GPU资格并启动唯一8192；不要再新增文档替代此动作。
3. 同时把共享8192交付→Core具体checkpoint推理资格→DEV68消费接入现有接续，并让CPU完成12候选与Go2离线包；原BOOK评价继续按已有协议。

## 10. 证据索引

- [S1 实际72h与训练性能回执](../../reports/PI0294/design/controlled-finetuning-20261102/CORE_RELAY_EXTENSION_TRAINING_SPEED_ACTUAL_20261103.json)
- [S2 Core实际接续与原生RESET](../../reports/PI0294/design/controlled-finetuning-20261102/CORE_NIGHT10_RESIDENT_READY_ACTUAL_20261103.json)
- [S3 固定共享采样与8192科学配方](../../scripts/pi0294/multitask/shared_task_sampling.py)
- [S4 现有共享训练资格与CPU卸载限制](../../scripts/pi0294/multitask/shared_training.py)
- [S5 原生接续shared/raw证书入口](../../scripts/pi0294/night10/native_relay.py)
- [S6 AGENTS](../../AGENTS.md)、[Goal V7](../protocols/GR00T_WBC_SCIENCE_GOAL_V7_20261031.md)
- [受控微调与交付计划](../protocols/CONTROLLED_FINETUNING_AND_POLICY_DELIVERY_20261102.md)
- [目标标定与新旧接口区别](../protocols/GO2_ARX5_CALIBRATION_AND_TARGET_DEMONSTRATION_PLAN_20261028.md)

本文只发布任务组织、源代码核对与建议；本次未执行GPU、物理、远程作业提交、owner转移或数据删除。当前具体进度以执行owner新原生回执为准，不能把合并本文当READY或完成训练。