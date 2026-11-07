# Vanda目标域GR00T微调：数据准入、真实训练、Core闭环的无人值守接续

日期：2026-11-08，Asia/Singapore。
状态：PLAN_ONLY_NOT_SUBMITTED_NOT_TRAINED。
审阅执行基线：PR #366 `09a8a1a70d546f1e26365d46bc75e388d26afacf`。
关联计划：PR #368 `abf96049bc62760cb19e303b5311668d34587a03`；本文件补充其目标任务训练分支，不回退现役源码。
用户本轮请求：现在是否要在Vanda端继续微调，并给出详细无人值守自主推进任务。

## 0. 决策与适用范围

应立即恢复Vanda训练侧准备，并在合格目标域数据就绪后启动一份**新的目标域GR00T候选**。不是把旧E2／共享M32／S-CORR训练作业直接追加步数，不等待神经WM胜过基线，也不把WM状态转移当成VLA专家示范。

本请求的计划范围为：Core真实目标任务采集、Vanda合法单卡A40配额下的独立GR00T动作侧微调、模型交付、Core原生闭环评价与TRAIN恢复数据接力。只更新新候选的已声明动作侧参数；原GR00T基线、WBC、腿策略、启动规则、安全与2ms步长保持不变。视觉/语言骨干默认冻结；解冻它们不是自动后继。

H200新allocation/排队、GPU物理多世界、分布式、多GPU或替代WBC仍保留既有人工审核停点。Vanda名称本身不代表H200授权。历史H200配额不能当作当前许可。仅在当前站点/用户权限和新实验profile均允许时提交；无权限就记录具体阻断，不伪造授权。

发布本文件不启动Codex、PBS、Core物理或训练；实际执行由既有现场agent和owner完成。无项目内部累计工程预算；每个作业仍有合法有限walltime、端点与收尾。当前Core有效窗口与Vanda allocation分别核实，不把新计划当作延期凭证。

## 1. 已核查事实与尚未知事项

1. G1 Teacher04在冻结0.9rad启动和WBC下实际通过：38703积分、176段Teacher后缀、正向2.732171mm、横向0.737462mm、BASE1.045470mm；是短接近成功，不是抓放。[R1]
2. 该数据明确为`DIAGNOSTIC_NOT_AUTO_TRAIN`，新目标TRAIN/DEV采集器与非零学习入口仍待完成。不能直接把G1、旧DEV、C01诊断改标为TRAIN。[R1]
3. 截至审阅快照，FUSION累计324个原生尝试、30次VLA查询、30800次WM主更新；FUSION的GR00T和Actor新增更新仍0。它不是目标域GR00T微调失败的充分实验。[R2]
4. 历史Vanda路线为合法单A40、PBS、项目scratch、原relay/Journal/onceIO。旧共享M32用46完整示范+13后缀、34563窗口、接口+末2层DiT、77847684参数、8192固定主端点；这些属于旧源域配方，不自动变成目标域新配方。[R3,R4]
5. 原E2 `1508345.stdct-mgmt-02`、共享`1514400.stdct-mgmt-02`只是需要只读核对的历史标识。后者11月3日曾实际运行于A40；当前队列、终态、checkpoint、额度和环境必须现场重新核实。GitHub没有证明它现在仍运行或已经交付8192。[R3,R5]
6. 官方当前N1.7资料将动作侧微调与全视觉语言骨干训练分开，并建议40GB以上训练显存；这仅支持选择A40做实际容量验证，不保证本项目vendor/数据/精度必然适配。不能把上游main直接升级进现役实验。[U1,U2]

结论：Vanda先做真实资产/数据/环境准备，Core并行采集；正式GPU训练由数据与资源门触发，而非“有卡就训练”。

## 2. 分工与依赖图

```
V0 当前作业/权限核对 ──→ V1 版本化目标训练入口与数据CPU验收 ─┐
                                                         ├→ V2 A40原生训练资格
C0 复用稳定起点 → C1 新TRAIN/DEV/TEST与Teacher采集 → D0封存 ┘       ↓
                                                    V3 小样本可学习性
                                                           ↓
                                                    V4 正式目标微调
                                                           ↓
                                       V5 可恢复模型包 → C2 本地一致性+闭环DEV
                                                           ↓
                                      C3 TRAIN恢复采集 → 新数据版本 → V4下一代
                                                           ↓
                                       冻结候选 → 新TEST → 后继WM增量实验
```

Core：真实MuJoCo+冻结WBC、真实双RGB/状态、Teacher与学生闭环、错误分解。一个物理owner；现役健康任务自然封存，不抢停。
Vanda：输入验收、GR00T前后向、checkpoint/恢复、开环诊断和模型包；GPU只在合法PBS compute allocation内，禁止login节点训练，不运行目标物理环境。
现有本地控制agent：唯一任务身份与跨站交付。复用已有Journal、relay、quota检查和传输路径，不增加第二调度平台。跨站只在数据/模型交付边界同步，不让健康训练依赖Core实时心跳。

## 3. V0：先核历史任务，不重复提交

只读核现有项目registry中的E2、共享M32及所有后继jobID；历史两个ID仅为检索起点。使用已安装PBS命令及项目原控制入口，不猜新queue/account/path。

- RUNNING：不迁移活动CUDA、不换数据、不热改；新候选独立排程，不能偷用既有allocation身份。
- QUEUED：绑定实际jobID、输入SHA和实验身份；qsub超时/UNKNOWN不重发，先用PBS和原Journal独立核查。
- COMPLETE：验证完整权重/optimizer/scheduler/RNG/sampler、原端点、传输包及已有评价；可复用有效资产，不重训已完成baseline。
- FAILED：保留已执行步数、失败原因与完整checkpoint。只有同一数据/参数/优化器语义的故障恢复才属于resume；不能用失败旧身份装入新目标数据。
- EXPIRED或无权限：不复活CLOSED，不把旧72h当新授权。只阻断受影响写入/算力域；CPU审计和Core合法采集继续。

旧E1/E2、共享8192的主端点、样本和DEV68不改。它们不是新目标任务的科学前置；唯一冲突是实际资源/所有权，需正确串行或交接。

输出：`CURRENT_VANDA_READBACK`记录实际观测时间、job状态、候选父资产、当前站点额度与现有任务的关系。不得将read timeout记为队列为空。

## 4. C0-C1：目标域数据优先于扩大训练

### 4.1 稳定起点与短任务

复用PR #368当前G1结论与原生入口；不重复完整初始化修复矩阵，不强迫进入源绝对位姿，不修改0.9rad启动、不固定自由底座、不降低安全阈值。

第一任务采用已获Teacher能力的短接近/保持，但新布局必须让学生需要区分至少两种可观察目标关系。不能全为同姿态同物体的恒定+X命令。语言描述与真实课程一致，不能拿完整抓放指令监督一个永远不闭爪的短技能。

新场景在episode开始前声明；全启动扫掠检查、BASE/高度/接触监测与有限退出保持原规则。Teacher可使用明确标记的特权几何；学生仅接收部署可获得的RGB和状态。教师依赖学生不可见的信息时，应改善可观测性，不能期待学生猜测。

### 4.2 首批设计（建议起点，非效力保证或工程预算）

- TRAIN：6个预登记布局 × 2种合法目标条件 × 2次独立重启 = 24次计划尝试。
- DEV：4个不同布局 × 2种条件 = 8个条件。
- TEST：另外4个封存布局 × 2种条件 × 2次独立重启 = 16个最终评价条件；开发期间不开启结果。
- 物理资格尝试单列；不为凑成功分母反复重试同一已完成条件。新修复或新课程另立身份。
- 进入首版正式训练前，建议至少16条合格短episode、至少4个TRAIN布局、两种真实任务差异；不足时继续新的TRAIN采样。此门是本计划选择，不是通用数据量定律。
- 已有诊断只复用软件/能力证据，保持原角色。不得把旧TEST或已暴露DEV改名加入训练。

### 4.3 可用VLA数据的硬条件

每个样本绑定真实同步双RGB、实际state、任务语言、Teacher在政策控制边界提交的命令、安装/消费记录、实际结果与源时钟。图像要覆盖任务全过程，不是每episode仅一张起始图。

监督标签必须是该观察下任务正确、语义明确的Teacher动作；随机偏移、HOLD动力学对照、失败动作与WM预测不可直接作为成功BC标签。完整短episode和经资格验证的专家恢复后缀分别记录。Teacher到执行器之间若有时间重定时/坐标映射，须确定策略训练的那个动作边界；别把WBC反复发布的40点预览全当作独立专家未来。

输出保留四层：`task_intent / commanded / installed_consumed / achieved`。不拿未来achieved pose反标当前专家命令；不使用尚未执行或资格不明尾部。50ms指令网格与2ms物理步分开；视野不跨episode拼接。短尾按现有正确padding/mask实现，不能复制下一episode动作。

## 5. D0：数据封存与原Processor往返验收

优先复用本项目已验证NativeDataset/Trainer与原Processor；若转LeRobot，进行精确字段对照，不为了格式重写整个数据平台。

最小bundle包含：dataset manifest、TRAIN/DEV/TEST布局名单与角色、媒体与数值SHA、modality配置、TRAIN-only统计、相机/状态/动作时钟合同、Teacher与WBC身份、全部采集attempt及失败索引、固定诊断样本。私有媒体、凭据、host绑定和大权重不上传公开GitHub。

在实际CPU环境检查：
1. 双RGB顺序、图像处理、单位、world/base/tool/TCP、rot6d及夹爪开闭含义；不重复Processor解码。
2. 原始物理动作→训练变换→解码的数值往返；容差按已固定FP32/旋转表示原数值误差确定，不凭训练结果放宽。
3. 训练与部署相同的数据模态、dtype、有效mask、语言和时间间隔；视频帧不来自未来状态。
4. 所有统计仅由TRAIN得到，DEV/TEST不读入fit；按物理布局/episode分区，绝不随机拆邻近窗口。
5. 新目标候选可以使用目标TRAIN统计；冻结源基线保持原统计。必须明确“模型+新Processor整体适配收益”，不能将统计变更全部归因权重学习。
6. 新本体配置单独版本化。可显式复制兼容源接口权重作为初始化，但不可伪装成已预训练本体，不填零伪造BASE监督。

数据bundle封存前不提交长GPU任务。无真实目标数据时，Vanda可完成CPU验收、资产缓存及必要的小型资格，状态记`WAIT_TARGET_DEMOS`；不以重复旧源8192代替目标进度。

## 6. V1-V2：Vanda入口与单卡A40真实资格

历史`shared_vanda.py`硬绑定FROZEN59、共享owner、M32及8192，不能直接换路径运行新目标配方。复用其PBS/onceIO/quota/传输/完整恢复接缝，为新科学身份实现窄范围的目标训练profile；原入口与旧capsule不改。

在当前合法A40 allocation内，先测实际UUID/显存/CPU内存/项目scratch配额/持久目录/原软件版本。历史48GB不是当前空闲证明。不在login/AMD上做模型GPU工作、仿真或重编码大媒体。缓存缺文件仅走原授权传输，不安装系统组件或修改共享环境。

同一allocation内完成一个独立资格进程：真实前向、backward、optimizer step、原Processor自由生成、checkpoint保存/恢复。先microbatch1、effective batch32；确认许可与余量后，最多对2/4/8进行有限容量测量，选定配方再开始正式run。峰值需包括参数、optimizer、activation与进程上下文；保持原余量门。不能从4080S推理通过推断反向训练能放入16GB。

首选冻结视觉/语言骨干，训练明确列出的目标state/action接口、projector与action head/DiT。打印参数白名单、精确numel、梯度与至少一个正式step的权重变化；确保确实更新GR00T内部，而非只更新外部adapter。

若完整动作侧不适合实际A40：保留资源失败，按预声明新配方试目标接口+末6层DiT；仍不足则末2层。每个是独立的容量/学习范围变体，不能沿用全动作头名称或悄悄更改正式run。不自动解冻VLM、引入未验证LoRA或上H200。若都不足，返回具体内存分解与`AWAITING_SCALE_REVIEW`。

no-offload、activation checkpointing、microbatch和冻结tensor放置是内存执行版本，需分别测量；不隐式改变模型存储精度、图像分辨率、有效batch、数据或可训练层。上游默认值只是参考，实际命令先核固定vendor的`--help`/dataclass/原trainable实现。

## 7. V3：小样本拟合检查，先回答“是否真的能学”

使用4条已准入TRAIN完整短轨迹，独立DIAGNOSTIC候选200步，必要时按预登记端点到500步。普通loss、固定噪声生成、位置/姿态/夹爪分项、执行前缀与完整H40分别检查，覆盖至少两类目标。GPU资格和此小样本权重不进入正式训练。

训练可学习性门：数据数值/梯度有限、原freeze保持、目标参数实际改变、相对父模型的固定TRAIN动作方向与任务相关误差改善，且没有用夹爪/旋转退化换位置平均数。数值任务容差在数据封存时从Teacher执行要求确定；不要求随机流匹配loss单调，也不以loss下降单独宣布通过。

若不通过，先查动作标签、观测延迟、Processor往返、padding、训练白名单与采样；一次只修一类。修复数据/配方后新身份重做必要检查，不能无限同样500步或直接扩大场景。

## 8. V4：正式目标域GR00T候选配方

### 8.1 模型谱系

冻结源基线B0保留现有checkpoint及原Processor。新的TARGET_FT_V1优先从已经审计的源S-CORR父初始化，使用新目标模态/统计和fresh optimizer；若真实资产核验发现架构或接口不兼容，则明确阻断/建立可验证映射，不能自动换成其他checkpoint。

从预训练权重初始化的新候选不是“从头随机训练”；目标数据/统计改变也不是旧run的精确resume。后继raw基础模型对照可在首个pilot后另立，不同时开无意义的大矩阵。

### 8.2 建议预登记配方（非既有实测）

|项|TARGET_FT_V1|
|---|---|
|位置|Vanda单A40合法allocation|
|任务|一个稳定可观察的目标短技能，尚非完整抓放|
|训练参数|资格确定的action-side精确白名单；视觉/语言骨干冻结|
|数据|封存目标TRAIN；第一代不混旧四任务数据，保留独立源回归集|
|输入|真实双RGB、实际EEF/夹爪状态、正确语言；不加WM特权状态|
|动作|固定EEF10与实际时间语义；默认保留已核实H40，若改变必须重建标签/统计/模型配置|
|优化|AdamW，LR1e-4，WD1e-5，effective batch32|
|数值|固定vendor已验证BF16 autocast/存储策略；无静默全模型cast|
|主seed|预登记20261108；另两个seed仅在首版证明可学习、数据稳定后做确认|
|日程|预声明8000步余弦日程，warmup200；2000/4000/8000是接续端点，不事后改旧scheduler|
|保存|每250更新完整checkpoint；关键端点重载/导出；禁用自动删除历史科研checkpoint|
|首个生产切片|2000更新；完整有效batch访问数64000，与物理独立样本数分开|
|验证|每500步固定TRAIN+DEV开环诊断；Core主要行为比较在2000、选定后继端点，不每250点全量物理|

若固定vendor缺少表中功能，仅实现必要接缝，不假装命令已经可运行。上游CLI与本项目原入口不能直接混抄。

### 8.3 无人值守续训规则

在2000步封存后，CPU/开环诊断与Core小面板确定下一分支，旧端点全部保留：
- TRAIN尚在改善、DEV未恶化：继续同一身份4000，再依据相同规则考虑8000。不得越过许可/walltime。
- TRAIN很好、DEV明显差：停止对同数据堆步数，回Core增加新TRAIN布局/恢复样本；新数据用新generation、新统计与新身份，不篡改原run。
- 开环好、原生差：优先时钟/执行前缀/协变量偏移/接触阶段诊断，不自动加学习率或解冻骨干。
- 数据/权重/数值异常：保存最后完整状态和失败，隔离该臂；必要修复形成后继，不无限重启。
- 第一代目标任务通过后才扩到接触/抓放；不在一次pilot中混入开门、整理书架以及视觉骨干解冻。

任何端点选择属于DEV开发，不将DEV选择收益包装为盲测。最终候选冻结后使用未开启TEST。

## 9. 无人值守PBS/恢复与资源行为

正式提交前固定数据、父模型、配方、源码、授权窗口和唯一run identity。只提交一次；返回不明时查队列/原Journal，不执行第二次qsub。旧job是否存在必须以实际PBS读回为准。

数据齐备后再排GPU，不让已分配A40等Core采样。候选任务在现有owner体系单一producer；不因Vanda排队而在Core复制同一个正式训练。排队期间可做Core数据/CPU工作，不承诺开跑时刻。

每个allocation通过实测step时间+冷载+保存恢复+导出估算下一切片是否装入walltime。至少保留原站点收尾余量；新计划建议不低于20分钟但更严格站点规则优先。不足时先封存完整checkpoint，再经实际允许的resume合同申请后继；禁止直接续期旧CLOSED。

捕捉站点正常预警/终止信号时在完整optimizer step边界落盘。checkpoint包括模型、AdamW、scheduler、scaler（适用时）、RNG、sampler状态、global step、数据/Processor/配方SHA。验证恢复后的固定batch预测及下一步更新；同硬件确定性配方检查应为原要求，否则预先声明跨硬件数值容差。

不删除旧数据释放空间、不杀其他用户作业、不改SSH/NFS/驱动/系统服务。不通过旧scope中`GR00T_updates=0`运行新训练，必须登记允许非零动作侧更新的新profile。本用户请求支持独立候选动作侧训练，不改变原冻结对照或给予额外站点凭据。

## 10. V5/C2：模型交付后必须做Core闭环

交付包包含：精确父revision、完整推理权重或delta、目标Processor/统计/modality、执行时钟/夹爪映射、参数白名单、训练data/recipe/source SHA、恢复证明及模型卡。训练optimizer档案留Vanda持久位置；通过已验证私有路线只传Core所需的推理资产，不重复搬完整父模型。

Core在真实原GPU环境重建候选，固定真实输入和噪声核Vanda/Core解码一致性、推理峰值、完整提案和实际安全。原FP32与新BF16数值策略差异显式资格，不偷偷改变部署推理。

对同一新课程先比较：B0冻结源策略；B1目标微调候选。两者共享实际相机/状态语义、确定性轨迹编译与WBC；B0保留其原Processor，B1使用其训练绑定Processor，结果称“模型包目标适配收益”。可追加只改变Processor的shadow诊断来分解贡献，但不能将故意错配的输入部署成基线。

第一轮DEV使用8个固定条件×B0/B1×2个独立重启，共32次计划评价；新物理条件先单环境。固定顺序/种子，独立重启不称完整MPC隐状态fork，所有拒绝和未运行单列。二模型各自用自己的真实后续观察，不借另一支的未来图像。

H40是预测/预览，不是必须一次盲走完。新的有限闭环先预登记最多3个重规划周期及固定execution horizon，依据本任务实际时延和参考TTL选择；保留每次完整提案、已执行前缀及取消尾部。不得事后选正向前缀、修改旧评分或刷新旧时间戳。暂停辅助与非暂停部署分别报告。

主要报告：完整短任务成功/有效/预定覆盖、首次失败阶段、物体进展、夹爪事件、BASE稳定、原安全/拒绝、参考跟踪、推理延迟与实际GR00T查询。开环loss不是验收结论，Teacher成功不是学生成功。

最终16个新TEST条件只在候选/模型包/执行节奏冻结后开启；B0/B1各2重启设计已含在上述TEST人口里，实际方法评价总数按方法另计并预登记。TEST失败不反复调参；后继用新TRAIN/DEV并保留旧TEST结论。

## 11. C3：反馈数据接力与WM位置

任务意图错误：在新的TRAIN布局中采学生状态，并由合格Teacher/人工提供真实纠正；不要把失败轨迹本身当正确动作，不把暴露DEV/TEST移入TRAIN。

执行误差：保留同一任务意图，审计实际命令/响应，优先简单标定/反馈；可以另训execution adapter，但不称为GR00T微调。

WM转移与VLA示范两库分离。同一次真实采集可同时服务两者，但WM可用的失败转移不自动成为成功BC标签；当前1614→16×8响应模型不是完整可递推环境，不直接用作SuperMimic任意长想象。

第一版目标微调比较不叠加在线WM纠偏。只有B1本身形成可解释短任务基线后，才比较同一个B1+简单反馈与B1+同历史/容量的WM适配；当前线性WM保留为强基线，旧192次负结果和BASE失稳面板不重跑/改写。

## 12. 最小进度表与停止/转向条件

在现有报告树下增量维护，不再造重复账本：
- CURRENT：实际执行源码、jobID/owner、数据/模型身份、正在运行/排队/终态。
- DATA：计划尝试、有效示范、布局/episode/窗口数量、失败和原split。
- TRAIN：实际step、样本访问、TRAIN/DEV分项曲线、峰值内存、恢复与checkpoint。
- EVAL：B0/B1逐条件真实结果、失败机制、覆盖/时延、安全与物体结果。
- DECISION：下一科学分支和第一具体阻断，不用测试数或PR数代替能力。

状态允许：WAIT_TARGET_DEMOS、WAIT_LEGAL_ALLOCATION、WAIT_ACTUAL_CAPACITY、QUALIFICATION_FAILED、TRAINING、CHECKPOINT_SEALED、READY_FOR_CORE_EVAL、DEV_PARTIAL、NO_TARGET_GAIN、TARGET_SKILL_PILOT、AWAITING_SCALE_REVIEW。没有实质变化不刷同一警报。

目标数据尚无有效视觉差异或专家行为：继续采集/修Teacher，不提交大训练。
GPU不足：保留事实，CoreCPU任务继续，禁止抢占他人资源。
正确Teacher也无法稳定执行：暂停该任务族，回已稳定工作区，不让VLA承担不可执行命令。
训练损失下降但闭环没有提高：检查恢复覆盖、执行时间和阶段标签；不宣称成功。
存在有效短技能pilot后：先新布局/独立seed确认，再扩任务长度或模型范围。最终交付不是保证的成功率。

## 13. Codex直接启动指令

读取本文件与PR #368，采用当前最新执行代码，保留活动capsule。把新任务作为TARGET_GR00T_V1独立候选，不改旧E1/E2/M32和冻结FUSION基线。
先只读核Vanda历史job/权限/资产，Core沿已稳定G1采新目标TRAIN/DEV/TEST。数据与Processor验收通过后，在合法单A40新profile中做真实backward/完整恢复与小样本拟合，再正式训练2000并按预声明日程接续4000/8000。训练GR00T内部动作侧，冻结视觉语言骨干与WBC，不把外部adapter或WM更新冒称VLA更新。
提交PBS一次，UNKNOWN先核，不重放；完整checkpoint、数据、统计和源码绑定。候选交Core后完成同任务B0/B1原生DEV，按实际失败补新TRAIN恢复数据。没有目标数据时不为利用GPU而重训旧源模型。H200/GPU物理多世界/分布式/冻结控制变更仍停下审核。
返回真实数据、非零训练与恢复、模型交付、原生结果及worker终态；明确区分本计划、已实现入口、已提交、实际执行与任务成功。

## 14. 依据

仓库来源均以审阅SHA为准，后继现场以最新真实回执为准；不按文档回退代码。
- [R1] `reports/WM02/fusion-night1-20261106/STABLE_START_G1_TEACHER04_ACTUAL_V1.json` @ `09a8a1a70d546f1e26365d46bc75e388d26afacf`。
- [R2] `reports/WM02/fusion-night1-20261106/FUSION_NIGHT1_STATUS.json` 同SHA。旧next/goal字符串可能滞后，以单次实际回执和新scope为准。
- [R3] `docs/operations/CORE_VANDA_POLICY_DELIVERY_CAMPAIGN_20261103.md` 同SHA；这是历史方案，原期限已过，不能当当前准入。
- [R4] `docs/operations/CORE_VANDA_SHARED_M32_20261103.md`、`scripts/pi0294/multitask/shared_vanda.py` 同SHA。
- [R5] `docs/operations/CORE_VANDA_DEPLOYMENT_CPU_RESULTS_20261103.md` 与对应原部署JSON；历史1514400的运行证据不证明今日队列状态。
- [U1] NVIDIA官方 `https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md`（2026-11-08查询；硬件指导不是本项目实测）。
- [U2] NVIDIA官方 `https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/configs/finetune_config.py`（2026-11-08查询；必须绑定实际vendor，不自动升级）。

本文件仅交付计划；未经实际读回，不声称Vanda当前空闲、已qsub、已训练或数据已齐备。
