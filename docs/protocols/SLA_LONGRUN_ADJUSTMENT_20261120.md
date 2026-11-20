# SLA 长程实验审核与接续调整：MIX 异常恢复、D2 课程与学生闭环

日期：2026-11-20，Asia/Singapore。用户现场截面：11:40 SGT。本文是代码与已发表证据审核后的执行调整，**不是本次已完成服务器部署的回执**。

- [机读调整与分母](SLA_LONGRUN_ADJUSTMENT_20261120.json)
- [Codex 接续任务](../handover/2026-11-20/START_SLA_LONGRUN_RECONCILIATION_D2_NATIVE.md)
- [原完整长程计划](SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1_20261119.md)
- [原后台接续协议](SLA_SCHEDULER_FIRST_ASYNC_EVAL_V1_20261119.md)
- [本次审核与发布验证](../../reports/WM02/sla-longrun-review-20261120/REVIEW_AND_VALIDATION.md)

## 1. 决策

继续现役 TGT5K、MIX20K 与一个 ADAPT5K 谱系。TGT 与 ADAPT 健康作业不因 Core 故障、D2 准备或学生评价分数暂停。下一轮优先处理四项工作：

1. **MIX 异常冷态恢复**：核对869完整状态，新增真实的 crash reconciliation 路径；完整时从870接续，不补造旧进程成功、不回到500重练。
2. **D2 与第二阶段后台提前备好**：在 ADAPT 到2000前完成每条新增episode的两站读取验证和2001–5000发放实现；2000冷态只决策一次，不等待Eval成绩。
3. **尽快取得学生 native 分母**：补齐跨GPU数值差异证据，使用明确的新部署兼容版本；随后执行已冻结ADAPT0 DEV24的原顺序，前6只是现有面板前缀。
4. **把新增采集重心放到有效阶段和缺失条件**：HOLD补闭合/抬升，PLACE修预抓取与携物路径；已观察到的同首因失败不原样重复。

保留现役数据/模型/Adam/RNG/LR/参数范围；不默认启用未通过资格的microbatch4。训练、离线Eval、数据CPU工作可以并行；Core只使用一个已合格native world，并同时考虑主存占用。站点有限队列自行运行，Codex不实时轮询训练step。

## 2. 证据边界：哪些已经独立回读

本次读到的PR418固定head为`3670aaca2640449b456bf1ab527c4635a9feeb23`，最新公开现场报告截至约02:35–02:36 SGT；计划母提交为PR416 `ed41d796738b46939a28f4ccb0a34d9410e38a20`。用户11:40现场摘要更晚，两份私有回执未挂载到本会话工作区，未通过SSH读取。下表不把用户报告升级成独立服务器验证。

| 项目 | 本次证据层级 | 审核结论 |
|---|---|---|
| HOLD F1红/F2蓝2次TASK_SUCCESS、46156tick、139H40/1390P10 | 用户11:40报告；算术及现役评分代码交叉核对 | 接受为较新现场报告，完整私有内容和hash仍需执行端回读公开摘要 |
| TGT2000/3500封存；最新4324更新、138368目标访问 | 用户11:40报告；后台JobID与先前公开登记一致 | 4324是运行快照，3500是最新报告的完整封存，不称已完成5000 |
| TGT2000 Eval288+48、48/48恢复、双门93/96 | 用户11:40报告 | 需要固定panel/输入/噪声/统计和逐行结果回读；native仍0 |
| ADAPT750封存、1000后继运行 | 用户11:40报告；公开代码与PBS链独立回读 | 不重新提交同端点，不更换现役亲本或训练器 |
| MIX869训练侧保存、外层oomd终止 | 用户11:40报告；拒绝接续机制由代码证实 | 869完整性、journal与账本尾段须现场核验，不能只凭路径存在恢复 |
| ADAPT0 Core严格输出比较0/16、guard16/16 | 公开报告及`replay_adapt0_core.py`独立回读 | 真实跨GPU误差幅度未留档；不能用后续CPU解码结果替代 |
| ADAPT发放只实现到2000 | `adapt5k_vanda_handoff.py @2d81436b`独立回读 | 2001–5000调度需提前实现，原训练worker已经支持一次2k阶段切换 |

后继Codex先增量发布11:40私有回执的去敏摘要、完整checkpoint SHA和真实JobID/终态，不复制原始数据、主机私有路径或凭据。`f51300df…`只是用户报告的SHA前缀，不能填写成完整身份。[公开最新截面][LATEST] [后台登记][SCHED]

## 3. 科学进展的正确解释

### 3.1 HOLD 已证明这两种条件下的专家执行可行

现役评分器的门为抬升20mm、保持1s、底座定向移动80mm，F2的82.7mm符合当前门；不能把历史V5.1的100mm套到本轮。F1/F2仍只有各1条，F2移动余量仅2.7mm，不能据此估计跨布局稳健成功率。[原评分器][SCORE]

数据计数与现役时钟吻合：`5703 + 72×250 = 23703`，`5703 + 67×250 = 22453`。每份H40是40点专家计划，实际确认消费P10为10点。因此139H40含5560个计划点，**实际执行证据是1390点**；完整计划后缀不能称为全部已执行。139重叠窗口仍来自2个独立episode。

这两条是MuJoCo专家数据。它们证明了数据生产与执行可行性，尚未证明SLA学生成功，也不是硬件真机实绩。新增学生请求和硬件请求都为0。

### 3.2 离线安全性有所改善，任务行为仍缺关键证据

若TGT2000与原TGT500确为同一panel、输入、统计和噪声，双门87/96→93/96相当于净增加6条、6.25个百分点；这是Go2 TRAIN窗口诊断，不能写成闭环成功率，也不能从汇总数推断哪6条改善或是否有其他退化。唯一应闭合样本仍漏闭，说明“保持开爪且平滑”仍可能得到很高门通过率。[TGT500][TGT500]

更有信息量的既有面板是ADAPT0固定88个阶段窗口：24个P10到期闭合正例，10/24及时闭合，但及时闭合与完整H40/接缝同时通过为0/24；普通96窗双门却是92/96。应沿用这88窗在500/2000/5000端点评价，不随D2换题、不复制补到96。[ADAPT0][ADAPT0]

必须逐层区分：预测H40闭合→闭合落在实际消费P10→门通过并真正消费→正确接近和接触→抓持/抬升/保持→完整PLACE。已有闭环长期0分母的问题不能靠重复离线恢复证明解决。22mm是诊断阈值，任务判定仍用原评分器。

### 3.3 规模训练已实际推进，但不能用当前结果否定多源长训练

TGT4324/5000是旧目标数据上的长训练；ADAPT750/5000已在新目标/R200/LIBERO配方中训练；MIX869/20000仍处早期且工程中断。它们不是三个相同条件的纯数据配比实验。MIX与TGT只有同更新、同目标流的固定5k比较能回答原混源对照问题；ADAPT同时改变数据、state统计、域权重和阶段LR，效果只能归于整套适配方案。D2在2k后的改善还与继续训练步数共同变化，不能宣称是两条HOLD的单独因果效应。

## 4. 任务优先级与服务器分工

| 优先级 | 工作 | 启动依据 | 完成证据 |
|---|---|---|---|
| P0，并行 | 保持TGT与ADAPT现有健康PBS作业 | 已存在的真实owner/JobID | 不重投；TGT5k封存、ADAPT阶段持续推进 |
| P0 | MIX869异常核对与一个有限后继 | 旧owner清空、可恢复完整前缀、新实际资源窗 | crash回执、真实870更新及下一完整seal |
| P0，并行CPU | D2两站reader与第二阶段卡 | 已准入新TRAIN候选 | 每episode覆盖、139窗语义检查、stage bank hash、真实2k→5k队列 |
| P0/P1 | ADAPT0部署数值证据与DEV24 | 冻结的兼容V2、真实空闲GPU/主存/world | 完整数值工件、原顺序学生native分母 |
| P1 | TGT5k/ADAPT500、2k、5k独立Eval | 已封存精确端点、唯一eval claim | 原固定面板与阶段漏斗，不重复已完成评价 |
| P1 | 下一批最多12条TRAIN采集 | 新场景/生产者修复证据、单world | 实际成功/失败/first-cause、逐条准入 |
| P2 | 性能候选与N2补证 | 前述工作不被主存/CPU挤占 | 既有有界工程合同；未过仍N1 |

VANDA优先完成已运行TGT/ADAPT；额外实际A40用于短Eval或MIX迁移。TGT5k结束后可按站点许可为后继重新申请资源，不把旧allocation自动当作永久可用卡。Core GPU0优先恢复同谱系MIX，GPU1承担合格推理；模型装载/保存与另一大内存进程的高峰需要错开。GPU分离不代表主存隔离。

已实证的TGT依赖是**PBS afterany + 启动前完整seal门**，不是所有边都afterok。异常前驱可能释放PBS资源，但worker必须在模型装载前拒绝无效状态；完整有界前缀可用于继续训练，正式Eval只接受精确端点。保留已经健康工作的依赖，不为使文档字面一致而改队列。[SCHED][SCHED]

## 5. MIX869：不伪造成功的异常恢复

### 5.1 为什么现有后继没有启动

`closed_predecessor()`和`scale20k_core_continuity.prepare()`需要原外层RESULT、CLOSED lease及trainer完整终态。外层只在`finally`写RESULT/关lease；SIGKILL不会运行该分支。现有handoff等待PID/cgroup消失后仍读旧RESULT，因此缺回执会被拒绝。该保护不能直接删除，应新增“异常后已核对完整前缀”的分支。[恢复入口][CORE_PREFIX] [实际handoff][CORE_HANDOFF]

`systemd-oomd`根据主存压力/交换区相关信号选择cgroup并SIGKILL，不等于CUDA显存分配失败。当前代码有启动时主存检查和24G的服务MemoryMax；reader缓存也有上限，不能未经日志就说是无限worker或batch过大。具体原因需要journal、被杀unit及ancestor slice、memory.events/peak/PSI、当时并发服务。允许的用户态日志不足时明确记录UNKNOWN，不能升权读取或修改主机保护。[systemd官方说明][OOMD]

### 5.2 恢复步骤

1. 保留旧失败与私有目录，核对无活PID/cgroup/GPU进程、无其他owner、STOP及旧期限。公开旧Core后继最晚07:14:32 SGT，11:40已过期；需新真实有限资源卡，不能复用旧时间。
2. 验证最终目录COMPLETE及140数组、模型/Adam/RNG、原worker文件hash、bank与统计、学习率表、global step、完整UPDATES和累计cursor。pending目录或progress文件不是完整checkpoint。
3. 若500→869的seal完整，期望是369条已提交更新，新增访问T11808/R17712/L17712；累计T27808/R41712/L41712。这只是对账公式。中断更新可能多消耗生成，真实H40必须读实际记录，不用4×step掩盖尾段。
4. 独立新增`CRASH_RECONCILIATION`回执，分别表达外层终止、训练seal可恢复、owner已释放；不把旧作业改exit0。已验证死亡的历史lease按现有核对工具释放并保留原状态/事件，不冒充正常完成。
5. 新编排只接受上述完整证据后从870接续，保留Adam/RNG/LR和原数据。若869不完整，找最新完整端点并记录不可恢复尾段/已花成本；回放丢失尾段必须明确为恢复分支，不能称无损连续优化，也不隐去重复计算。恢复资格及重放的全部实际更新、访问、生成先扣对应原账剩余额度，不借用ADAPT或其他谱系预算；若剩余不足完成原终点，不自动超限，封存实际可达前缀并标明原终点未完成，后续版本再单列必要增量。
6. 仅做影响恢复的负例验证：坏数组、短账本、错cursor、活owner、pending-only、STOP、过期资源。随后注册一个有限后继，观测真实870更新及下次seal；不启动无限自动重试。

### 5.3 主存修复与后台可靠性

先根据load/restore/train/save峰值及并发native/缓存占用调整项目内资源准入；不关闭oomd、不移除主机限制、不删数据腾空间。原25k/ADAPT5k优化合同保持。主存仍不足时，允许将同checkpoint和同recipe迁移到真实获配且主存足够的A40，完成原读入/恢复/数值资格并保持跨站唯一训练owner；不得借此采用已拒绝mb4工程权重。

下一段用原runner优雅保存；终态核对放在既有调度机制的独立轻量事件路径，不能只依赖被杀worker的finally。日志和失败事件保留，不需要Codex常驻SSH。正式checkpoint封存分别释放TrainNext与Eval；GitHub发布不在训练关键路径。

## 6. D2准入与ADAPT2000课程

### 6.1 当前候选应怎样计数

公开D1为旧23条/1768窗，加1条F2蓝左完整PLACE/124窗，总24条/1892窗。若本次只加入2条HOLD/139窗，则D2为26条/2031窗，22HOLD+4PLACE；累计新增独立episode为3。配额为`min(48,4×3)=12`新目标槽位，52旧目标槽位，再加32RoboDojo+32LIBERO。不是8槽，也不是139条新示范。[D1与采样][D1] [配额实现][ADAPT_CONTRACT]

这是条件性D2清单：执行时若另有已准入数据，必须明确新的membership、E_new与配额后再冻结，不能在训练中偷偷追加。已有ADAPT亲本仍是MIX500；不得把运行到750的谱系追溯改为新MIX亲本。

### 6.2 14个文件同SHA后还要核验什么

旧预检target只取`[0, original_windows-1, original_windows, len(target)-1]`。在D2上它可能完全漏掉F1新HOLD，更没有覆盖闭合事件。修复CPU预检选择器，不改现役训练worker。[PRECHECK][PRECHECK]

- 全部新增139窗检查ID/episode、时间、nominal H40、上次真实命令、consumed/valid/supervision mask、P10连续性、STOP终态和split排除。两episode全程无STOP仍要按原规则核验。
- 每条新增episode选首窗、闭合前窗、P10到期闭合窗、末窗，去重后在两站用实际production reader读取RGB、相机mask、state、action、语言/分词、前命令及身份；报告实际数，不复用D1的“22条总样本”冒充覆盖。
- 两站重建同seed、同成员、同阶段的三个bank并比较hash；验证新旧配额、阶段抽样、aux轮转以及stage-local/lifetime cursor。
- D1 target state统计、C20 action统计/codec、R48源统计、LIBERO统计保持冻结。D2状态越界只作诊断，不重新拟合后热替换。

### 6.3 现在就完成第二阶段发放，避免2k空转

现公开handoff的TRAIN_LIMITS只到2000、前驱验证step≤2000、Eval仅500/2000。训练worker`1432c8dc`及checkpoint已支持恰好2000时只改变stage范围/target数据/bank，因此提前扩展编排和Eval5000入口，继续调用原固定worker；不要改source_revision、损失或优化器来绕过身份检查。[HANDOFF][ADAPT_HANDOFF] [阶段边界][ADAPT_CHECKPOINT]

2000完整封存后一次性决策：

- D2的准入/reader/bank已完成且确有新episode：phase2001切D2。
- D2未就绪：phase2001沿D1成员继续；仍建立2000–5000的第二阶段bank和游标。训练不等待D2或Eval分数。
- 若核对时已经存在2001及以后的D1更新：不回滚、不热切；本次D2留下一版本。

当前已经排队的至2000作业保持。第二阶段优先按250更新切片准备2500以前的后继，再用同样规则挂至5000；具体片长可依据原同recipe实测p90与保存p95决定，但科学端点、全局LR和总更新数不变。未来parent checkpoint SHA在封存时绑定，不能预造；预排模板持有lineage/期望阶段/真实依赖，启动时解析实际seal。若某个有界前缀未到名义端点，仍从真实s+1推进，Eval等精确端点。

D2切换记录D1固定锚、当前stage membership、bank和新stage-local cursor，并保留累计cursor/Adam/RNG。ADAPT2k Eval使用切换前的封存件，5000 Eval继续固定D1的288+48+88面板；D2训练读取器与固定评价读取器不能混淆。

## 7. 跨GPU部署兼容V2：补证据，保留旧失败

公开Core ADAPT0已有16次真实生成，只有布尔的逐位/门结果；runner没有保存Core数值数组。后续CPU注入同一A40预测得到2.22e-16m解码差，只定位了CPU舍入，不能推断GPU模型输出差异。JAX官方也不保证跨加速平台的精确数值稳定；这不等于任何数值差异都可接受。[部署实现][DEPLOY_CODE] [JAX官方][JAX_NUMERICS]

执行前先清点既有工件。若确有两端原始数组，可零生成计算；如缺失，记录证据缺陷，在原336次技术储备内最多使用32次新生成：原16固定输入/噪声一次，以及同Core独立恢复一次。原已用16次仍计在原48次部署复验额度中，不抹掉。新生成不是重跑科学失败或换样本挑通过。

本文件明确提出并预登记一个**新的DEPLOY_COMPAT_V2**，仅对新的跨硬件部署判定生效。原bitwise gate失败永久保留。以下数值上界取自此前已登记的工程合同，原本不自动适用于deployment；本次将它们作为新版本上界，并增加消费身份与门签名约束。不得看新差异后再调阈值。[既有数值定义][BENCH_CRITERIA]

| 比较项 | V2要求 |
|---|---|
| 权重、统计、codec、语言、RGB、state、mask、噪声 | 身份完全一致；版本与dtype逐项登记 |
| 同一Core环境独立恢复 | ADAPT0对固定backend/worker/dtype/环境做一次双进程资格，输出逐位一致；若不成立，先修实际非确定性/身份 |
| 全H40位置差 | 原`roundtrip_error` XYZ定义≤1mm |
| 全H40姿态差 | SO(3)误差≤0.01rad |
| 全H40夹爪物理宽度差 | ≤0.5mm |
| 闭合事件 | 有无一致；首次时差≤50ms；P10内/外身份必须一致，不允许k10→k11被掩盖 |
| 完整执行判据 | 原guard_signature一致，记录first-reject及阈值余量；原H40/接缝/STOP/任务评分不变 |
| 数值工件 | 保存归一化和解码数组、SHA、逐窗最大/p95/RMS差、全部40点及前命令接缝 |

ADAPT2k/5k各使用原48次部署额度中剩余的16次，核验各自完整checkpoint数组及A40独立恢复，并比较本端点的真实Core输出、guard和P10身份。仅当backend/worker/dtype/环境完全相同时，复用ADAPT0的Core双进程环境资格，不额外要求每端点再生成16次。若环境变化，不能复用；先登记新的有界资格及实际剩余技术额度，本版32次不能被默认为无限恢复预算。

所有条件合取通过才可将新包用于native。物理误差上界是跨实现一致性门，不是碰撞安全距离许可。若任何项失败，继续定位算子/dtype/输入/噪声；使用新的工程版本或原合格环境，在同一技术储备内做有依据的修复，不放宽门、不借CPU注入“通过”。MIX和ADAPT训练继续。

同时保持A40 mb4工程拒绝：其吞吐1.426×、物理输出门27/32，4条XYZ>1mm、1条夹爪>0.5mm；上述V2不能替它追认通过。[ADAPT0][ADAPT0]

## 8. 并行评价与原生执行顺序

### 8.1 离线

- TGT2000已有288+48按唯一eval_key回读，不重复生成；固定5000 Eval复用已登记`1593721.stdct-mgmt-02`并核对实际状态。3500是恢复端点，不另加正式面板。
- 回读ADAPT500已有`1594084`，2000已有`1594215`；只在真实无提交/明确技术失败且完成核对后补缺，不能把“缺公开报告”当“没执行”。
- ADAPT每端点固定288正式+48独立恢复+实际88阶段窗口；三个域分开。模型仍在训练时，Eval只读不可变封存包，独立owner/输出目录/GPU和主存。
- `eval_key=(lineage,checkpoint_sha,panel_sha,recipe_sha,noise_sha)`跨站唯一。没有额外GPU时Eval排队，训练继续；不靠实时Codex查看step来释放。

### 8.2 学生native

公开冻结的ADAPT人口SHA为`cdcd114a712739315d06ee923fe8221142d911ab6b60a64f30eac1b619eec2e0`：DEV24供ADAPT0/2000/5000，另confirm24只用于固定5000，总96科学请求。[LATEST][LATEST]

优先完成ADAPT0部署后执行既定DEV24的前6，再续余18；这6条并不是另一个筛选面板，也不保证F1/F2平衡，不能按前6好坏决定是否选择性报告。A0/2k/5k沿相同顺序、噪声、backend和评分；固定5k确认不依DEV挑最佳checkpoint。只在数据/部署/执行安全证据失效时暂停受影响派发。

TGT5k仍按原SCALE合同自己的DEV24及预算执行，不拿ADAPT人口的结果冒充纯混源配对；MIX5k/20k和其固定确认沿原合同。两条线的预算互不重复。

原生每条输出至少包括：模型完整生成数、H40首拒绝、接缝首拒绝、消费P10数、P10闭合、正确目标接近/接触/抓持、最大抬升/保持、完整PLACE、接触STOP及STOP后tick。先检验专家标签在相同前命令和时钟下是否可同时满足执行门，再区分模型闭合未学会、闭合被推到后缀、门拒绝、接近偏差及释放路径问题。

## 9. 本体数据：下一批最多12条，不等Teacher全格

在原240次采集上界内准备一个新TRAIN批次：4条HOLD优先补F1蓝/F2红及独立扰动，6条PLACE优先五个缺失组合与经修复的携物/开夹链，2条恢复任务仅在有真实可执行恢复生产者时派发。每次最多50703tick/180专家计划，整批最多608436tick/2160专家查询。具体scene/reset/seed/producer和顺序在启动前冻结；尚无生产者的条目保留未派发，不伪造数据填数。

PLACE已有证据分别指向下探腕端碰非目标、持物搬运碰另一块、开夹夹爪垫碰邻物。原`[0,+8]mm`落点候选后来在搬运阶段触STOP，尚未验证开夹修订有效。后继修复应检查从当前真实位姿出发的腕端、两指垫、被抓物体与桌面/非目标的完整扫掠，不只检查TCP终点；H40/P10重规划接缝也须保持连续。[第二批与落点][SECOND6] [LATEST][LATEST]

同cell、同producer连续两次相同首因，停止未修改的重复，转静态/轨迹诊断并记录具体修复，再用新TRAIN身份验证。其他HOLD、源训练和学生评价继续。失败轨迹保留诊断；原方案未准入的partial不混进当前D2完整示范数。成功的阶段前缀如需今后训练，须另建有mask的版本，不能把失败episode改成完整PLACE。

已有两批采集后，学生评价不再排在大批采集之后；在冷态按固定评价块与采集块轮转。N2未获完整资格前维持N1，CPU准备和VANDA离线仍可并行。

## 10. 预算、时间与停止条件

所有数量仍沿原SCALE与原ADAPT长程预算，不新加训练臂、不改5000/20000终点。本轮只重排任务和给出恢复/部署新版本。

| 预算组 | 保持的总上界（不是剩余额度） |
|---|---:|
| 原SCALE两臂 | 25000更新，2720000优化访问，118120模型生成，98原生请求/4399394tick |
| 新长程ADAPT及工程 | 5120更新，655360优化访问，38390模型生成，362原生请求/16824986tick |
| 其中ADAPT科学native | 96请求，4387488tick，15360模型生成 |
| 其中目标采集 | 240请求，12168720tick，43200专家查询另列 |
| 其中技术生成储备 | 336，部署补证最多占用其中32，须先扣真实已耗 |

两条历史接口每条5953tick，是旧D4 E0（`SLA_D4_CORE_NATIVE_INTERFACES`，计划`cf9d1acb…`）的carried证据，非每批新执行，不能反复累计，也不能据此推断新SCALE/ADAPT接口额度已耗尽。是否有另行新增接口按request ID查账；新包确需接口验证时才用其对应未用储备。[采集入口][COLLECT_ENTRY] [E0身份][D4_E0]

在无其他未发布尝试且用户HOLD终态成立的前提下，首6+第二6+落点1+HOLD2可合并为15次新专家尝试、439744tick、3条完整成功（1PLACE+2HOLD）、11接触STOP、1预算耗尽。这是跨报告算术合并，不是本次直接现场读出的总账；旧接口11906tick不加入这15次。执行端按唯一request ID确认后才发布正式累计。

粗略资源量仅用于排队：以公开同recipe暖态MIX35.94s/步，869之后19131步约191GPU小时；以ADAPT phase25的44.12s/步，750之后4250步约52GPU小时；TGT4324之后676步以13.38s/步约2.5GPU小时。均未含排队、启动、恢复、保存和D2数据差异，不能写成完成时刻或当前吞吐保证。A40 mb1公开50.08s/步，迁移不自动更快；稳定性、主存和native资源释放应一起权衡。[性能截面][ADAPT0] [ADAPT25][ADAPT25]

停止/降级按真实原因作用于对应任务：STOP后0积分；缺完整seal阻止该恢复；有效科学失败只记失败不重投；D2未就绪走D1；无额外GPU让Eval排队；N2未合格用N1；权限/期限失效停受影响的新注册，其他有效作业继续。到现役终点根据完整证据制定下一数据/loss版本，不自动无限延长训练。

## 11. 下一接续周期的交付判据

1. 用户11:40回执的可审计摘要已公开，事实/观察时间/私有待核对项分开。
2. MIX有真实异常核对终态、一个新有限后继及870起的实际更新，或具体不可恢复项；仅“已排队”不算恢复训练完成。
3. D2两站reader逐episode覆盖、139窗语义检查、成员/bank及12/52配额（若E_new仍3）完成。
4. ADAPT2001–5000编排已验证并真实排队；2000冷态决策只一次。尚未到2k时清楚标为REGISTERED/PREPARED，不能写已切D2。
5. TGT5000封存和固定Eval终态，或保留其真实运行/排队状态；没有重复TGT2000评价。
6. DEPLOY_COMPAT_V2完整工件及结果；通过后ADAPT0至少原前6条学生native终态，并继续原24条，或明确部署阻塞及实际0分母。
7. 新数据逐条准入，失败/partial保持，评价不再被采集长期挤占。

发布采用原报告分支/已有合法publisher，只在登记、正式封存、Eval终态、错误及谱系终态发简洁ACTUAL。服务器日志/心跳是恢复证据，不改成常驻Codex轮询。不要创建新的ChatGPT定时任务。公开PR418 head并不包含所有ADAPT核心文件：运行使用报告明确固定的训练`1432c8dc`、调度`2d81436b`及各部署源码；不能只checkout报告head就假设全部runtime齐备。补齐可达源码清单和分支关系，保持当前胶囊文件hash不变。

## 12. 本次发布的实际效果

本次完成只读审核、本地文档与算术/链接验证，并更新计划PR。**本次新服务器作业、训练更新、优化访问、模型生成、原生请求/tick和硬件动作均为0。** 上文现役进度来自明确列出的公开报告和用户较新回执，不是这次文档发布产生的结果。

[LATEST]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3670aaca2640449b456bf1ab527c4635a9feeb23/reports/WM02/sla-scale20k-20261119/ADAPT_D1_2K_QUEUE_NATIVE_POPULATION_AND_HOLD_ACTUAL.md
[SCHED]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3670aaca2640449b456bf1ab527c4635a9feeb23/reports/WM02/sla-scale20k-20261119/SCHEDULER_FIRST_DEPLOYMENT_ACTUAL.md
[SCORE]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3670aaca2640449b456bf1ab527c4635a9feeb23/src/go2arx5_vla/world_model/multitask_score_v6.py
[TGT500]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3670aaca2640449b456bf1ab527c4635a9feeb23/reports/WM02/sla-scale20k-20261119/VANDA_TGT500_OFFLINE_TERMINAL_ACTUAL.md
[ADAPT0]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3670aaca2640449b456bf1ab527c4635a9feeb23/reports/WM02/sla-scale20k-20261119/ADAPT0_SAVE_RECOVERY_AND_A40_COMPATIBILITY_ACTUAL.md
[ADAPT25]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3670aaca2640449b456bf1ab527c4635a9feeb23/reports/WM02/sla-scale20k-20261119/ADAPT25_FOLLOWUPS_AND_TARGET_SECOND6_ACTUAL.md
[CORE_PREFIX]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3670aaca2640449b456bf1ab527c4635a9feeb23/scripts/sla/scale20k_core_prefix.py
[CORE_HANDOFF]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3670aaca2640449b456bf1ab527c4635a9feeb23/scripts/sla/scale20k_scheduled_handoff.py
[D1]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3670aaca2640449b456bf1ab527c4635a9feeb23/reports/WM02/sla-scale20k-20261119/ADAPT_D1_R200_AND_A40_PARALLEL_ACTUAL.md
[ADAPT_CONTRACT]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/2d81436be0e4dc02bf67a180ea1cf00e818e29df/src/go2arx5_vla/sla_adapter/adapt5k_contract.py
[PRECHECK]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1432c8dc88c57adf39e8fa363f5438c5e63aa8fd/scripts/sla/preflight_adapt5k_data.py
[ADAPT_HANDOFF]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/2d81436be0e4dc02bf67a180ea1cf00e818e29df/scripts/sla/adapt5k_vanda_handoff.py
[ADAPT_CHECKPOINT]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/2d81436be0e4dc02bf67a180ea1cf00e818e29df/src/go2arx5_vla/sla_adapter/adapt5k_checkpoint.py
[DEPLOY_CODE]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0d3bc1a53bc5066d3590e6022ef2fc6c0cb9116/scripts/sla/replay_adapt0_core.py
[BENCH_CRITERIA]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f3fd3e4d5abce0184021fd77ba03e7d56ec9b6bf/scripts/sla/benchmark_scale20k_outputs.py
[SECOND6]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3670aaca2640449b456bf1ab527c4635a9feeb23/reports/WM02/sla-scale20k-20261119/ADAPT25_FOLLOWUPS_AND_TARGET_SECOND6_ACTUAL.md
[COLLECT_ENTRY]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3670aaca2640449b456bf1ab527c4635a9feeb23/scripts/wm02/target_adaptation_core_launch.py
[D4_E0]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3670aaca2640449b456bf1ab527c4635a9feeb23/src/go2arx5_vla/world_model/multisource_d4_native.py
[OOMD]: https://www.freedesktop.org/software/systemd/man/257/systemd-oomd.html
[JAX_NUMERICS]: https://docs.jax.dev/en/latest/api_compatibility.html#numerics-and-randomness
