# Codex接续入口：PV1 第一版共享多任务policy

2026-10-19，Asia/Singapore。用户最新目标：制定后续推进长程计划，得到第一版多任务policy。
状态：USER_REQUESTED_BOUNDED_GO_PENDING_NATIVE_ADOPTION。任务由现有local-PC单一owner推进；本文件发布不等于已运行。

## 一句话目标

在不改WBC、plant、夹爪PD、S1与任务评分的条件下，从现有CORR@2048出发，完成有针对性的数据扩充、受控微调与新场景独立验收，交付一份同时执行T1抓放和T2推送的共享策略包。不要把任务停留在诊断文档、零散成功视频或checkpoint保存。

第一版是固定底座ARX5源域仿真policy；不称为Go2整机、跨本体、实时或真机部署。两任务不能切换权重/adapter或用Teacher在后台控制。真实机器人deployment保持NONE。

## 必读同一固定提交的三件套

1. docs/protocols/PI0294_PV1_MULTITASK_POLICY_RELEASE_20261019.md
2. reports/PI0294/design/PV1_MULTITASK_POLICY_SCOPE_20261019.json
3. 本 docs/handover/2026-10-19/PV1_MULTITASK_POLICY_EXECUTE.md

先读AGENTS.md/START_HERE.md/统一入口及#11/#47，再固定三文件hash和commit写入PV1_ADOPTION_AND_RESOURCE_BINDING。复用原doctor、registry、RUN_MANIFEST、profile、lease和OUTPUT_BINDING，不新建第二套dispatcher或Home工程根。不把管理HEAD热加载到运行actor。

这是本用户最新目标下的新有限PV1数据/训练/评估范围。FD1与N1的旧停止和费用不重开；FD2原8条pilot的scope也不扩大。过去FD2的future-training=false不授权其自身再训练，但不应成为本次独立PV1授权永远无法训练的理由。实际账户、站点、quota和native准入仍必须成立；只有这些客观权限缺失才汇报精确阻断，常规阶段转换不再逐项请示。

## 最新事实与先手动作

审核基线8745adc74a7177eddef5f477bcd967df1806a0ab：FD2已实际采用，最近发布的是几何审计实现，尚无完整8条件终局；先刷新它的实际结果，不假设4mm或12mm胜出。健康FD2继续其原租约，PV1仅并行准备不抢占；完成则复用收据，不重跑。

旧S-BC/CORR两臂训练与N1已全部完成。CORR唯一旧独立确认T1=10/20、T2=4/20；FD1定位20/21碰撞为pad/table，但未证明名义4mm是神经训练根因。旧42条尝试、失败、视频、8锚点/16输出不反复做。当前policy父为CORR@2048 manifest94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38，不再从头加载官方基模开始。

## 七天关键路径（优先级，不是成功时间保证）

### 0—24h：数据路线决议，终止无界诊断

读取FD2全8条件及真正动态间隙/接触/no-lift/目标成功。按完整协议预设门决定COVERAGE_FIRST（原4mm专家）或CLEARANCE_PAIR（4/12mm同scene公共成功支持）之一。12mm更安全但推不到物体不是有效Teacher。审计缺口只读补一次；FD2不完整而原Teacher仍有有效资格，可走覆盖路线并保留未知，不让资料整理无限阻塞。

必要时允许一次有界Teacher任务反馈/释放/分离修正，至多两个预注册版本、全阶段新校准总额24episode；不改WBC和S1。不把确认图像/状态制成训练标签。

### 第1—2天：新独立数据与评价卡

冻结96个TRAIN组（F0 24/F1 36/F2 36）、两任务，原Teacher最多192条；裕度对照额外同组T2最多96条。前后方向分层，所有采集失败都留账。新T1成功独立支持至少32组、新T2至少64组且两方向各16组；目标各96组。只将真实安全完整成功且label/clock/split通过的示范收入BC。

同时预封存DEV12组×2任务×2bank=48条件/候选，以及LOCKBOX20组×2任务×2bank=80条件。排除所有旧TRAIN、DEV、CONFIRM、C0/C1和FD2近重复。DEV/LOCKBOX不能用Teacher或学生成功筛选，不能看到失败后换seed。先按scene分组再切H40，统计独立parent而非重复窗口。

### 第2—4天：两臂正式共享微调

先给现有CORR B0在新DEV48一次评价。两个新臂共同从CORR@2048开始、全新相同optimizer；各4096更新/batch32，seed20261019，lr1e-5→1e-6/warmup64，其余原flow-matching/projector+DiT/VLM冻结与推理数值不变。

覆盖路线batch32：A原T1=8/C1=8/原T2=16；B原T1=4/新T1=4/C1=8/原T2=8/新T2=8。两任务总曝光1:1，C1访问及保留游标匹配，检验独立示范覆盖而非多训练。裕度路线则两臂相同新布局及T1数据，仅新T2 Teacher版本不同；必须有>=64组公共成功支持，不足不宣称纯裕度对照。

1024检查点只做TRAIN探针；4096才评价DEV48。原2048硬编码不得简单删guard，建立PV1有版本的合法step/session/budget验证并保留旧协议回归。所有样本消费、梯度、冻结参数、完整resume必须有真实记录。

### 第4—5天：最多一次有证据的补强

主对照后若弱任务仍有缺口，按完整协议选择一个方法，再做共同父模型的2048对照：
- 有新TRAIN学生状态且任务对应Recovery Oracle完整5/6合格，最多24个学生episode与24恢复fork，做任务内<=25%纠正替换；T2旧0/2不是资格。
- 没有合格恢复但有真实事件欠覆盖与拟合证据，可做一次任务内事件采样对照，不同时加纠正数据/loss变化。
- 两者证据均缺，跳过补强，直接评价是否能封存候选；不无限增加steps、开新模型或泛化诊断。

### 第5—6天：稳定性复核、唯一候选冻结

按worst-task→macro→S1次数→复杂性→早期ID选择。候选DEV每任务至少18/24，相对B0各任务下降不超1/24且worst-task净改善。对胜出局部recipe用seed20261020复核一次；复核不合格不能声称稳定，资源不足未复核最多BETA。新训练总共最多5次、总更新16384、含B0最多6个DEV候选。

### 第6—7天：最终80条件与可运行策略包

先封存唯一候选的权重/processor/stats/环境/代码/数值/输入动作schema，再打开LOCKBOX。两任务分别40条件，全部attempt审计，原失败不重跑，每格只允许一次确证工程replacement且总cap受控；不得看到结果后换候选或重抽noise。

PV1_VALIDATED：每任务>=32/40，>=14/20场景两bank均成功，每task-family>=50%，每任务S1<=4/40，native fault0、80格完整独立审计、开发复核及交付齐全。
PV1_BETA：每任务>=28/40、>=12/20场景2/2、family>=50%、S1<=8/40、原生安全/审计/交付齐全；这是预设次级结果，不冒充主目标。
否则交BEST_RESEARCH_CANDIDATE_NOT_QUALIFIED，保持真实分母，不再用已开测试集训练和挑模型。

交付一份shared权重+processor/stats+POLICY_MANIFEST/ACCEPTANCE/POLICY_CARD/REPRODUCE、依赖锁、最小推理入口、完整数据训练lineage、DEV与LOCKBOX矩阵、原视频和安全索引。推理入口默认离线/dry-run，不自动连接机器人。原始大文件仍在固定私有存储/HF，Git仅放脱敏索引。

## 工程权限、预算与停止

新PV1 96H200h/24GPU PBS（同时1卡，单作业<=8h及站点更小值）、Core320core-h/Hopper160core-h/local48core-h；0A40/0目标WBC/0真机。最多48000神经生成、1000source episode/fork、40M live+shadow ticks、100000构造/clone，Hopper250/Core300/local20GiB新峰值，300GiB传输。全部失败/idle/恢复计费，实际个人quota/账户更小者优先。最终20H200h、100个episode、9000查询、live/shadow各1.6M ticks及40GiB存储保留，是总额子集。

沿成熟代码自主修复路径/依赖/PBS/IO/时钟/数据codec同语义问题与exact resume；不热改运行source，不放宽原0..300s心跳或20/60/900swatchdog。新allocation保留本作业600s/80cycles/all-fault资格，但不要每个checkpoint重新资格。批量元数据与归档避免卡在动作响应；结束面板不要反复加载。未知qsub/query/install/delete副作用必须先查journal，不盲重试。同一未定位原生故障两次后先修证据机制再新版本续接。

只有一个checkpoint维护lane。新run终态/后继验证/引用解除时精确清理无用failed payload与可重建cache，不逐文件请示；保留当前父/最佳/最新resume+回滚、所有固定比较终点、必要旧witness与原始数据。不能因0/4就删模型，不强删旧BC1631或唯一恢复证据。实际释放bytes单独回执，无对象记0继续工作。

首次采用在完整发布48h内；PV1_T0+144h停新job/candidate，+156h停新训练块，+160h停新物理实验，+168h全部收尾，不随重启排队延长。FD2等旧工作时钟不重置、费用不重复。常规同范围修复或预设阶段转换无需再次GO；真实安全/权限/资源硬阻断则保存并停止受影响lane，不能借新goal绕过。

## 采用与对外汇报

在#11写一次真实ADOPTED（固定三件套SHA/blob、实际T0、owner/lease/资源、首个动作），每最多4个活跃小时报关键进展、24小时一份科学主表。#47只放具体维护/清理收据，不每个helper再发大计划。

本次计划发布没有启动远端Codex、PBS、模型、物理或文件删除。完成状态必须由原生数据、最终验收与完整策略包证明；若硬边界前未达标，交真实候选和失败分析，而不是继续许诺或宣布V1成功。