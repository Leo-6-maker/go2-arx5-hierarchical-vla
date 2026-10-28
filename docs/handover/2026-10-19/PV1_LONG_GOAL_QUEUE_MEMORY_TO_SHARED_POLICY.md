# PV1-LG02｜从Hopper排队内存诊断到第一份共享T1/T2 Policy的单owner长程Goal

日期：2026-10-19，Asia/Singapore。Goal ID：PV1_LG02_QUEUE_MEMORY_TO_SHARED_POLICY_V1。
状态：USER_REQUESTED_EXISTING_PV1_GO_CONTINUATION / NEW_GOAL_HANDOFF_ONLY / PENDING_NATIVE_OWNER_ADOPTION。
发布基线：5a49c50d01ddd0520228d35aba8e2ef1c31f002b。
唯一实际执行者：现有local-PC单一PV1协调owner；同一长期会话可以跨聊天存活，但**本文件不启动第二agent、后台作业或新PBS**。

## 0. 总体目标与权威

**Goal：首先尽快且安全地解决当前675967的225GB排队争议，随后完整结束B0 DEV48、进行同父同配方的A_CONTROL/B_COVERAGE微调、完成开发评价与条件式复核，冻结单一模型，再在新LOCKBOX80上完成独立验收并发布一份真正可运行的共享EEF10多任务研究policy。**

- 这是对**已实际采用的PV1**的工程接续指令，不新授权GPU、PBS、数据、物理、训练、World Model、DAgger扩采或部署，不重置原T0/费用/时钟。PV1原权威提交：9eff29a44a193dd4d3620dc0bfe5764a402eff42；原协议：docs/protocols/PI0294_PV1_MULTITASK_POLICY_RELEASE_20261019.md；原机读额度：reports/PI0294/design/PV1_MULTITASK_POLICY_SCOPE_20261019.json。
- 此Goal只补充现有PV1的**排队根因归档、可用原生资源下的高利用率执行顺序和最终成果交付要求**。任何冲突，以原science/authority/lease/manifest/站点规则中更严格者为准。
- 读AGENTS.md、START_HERE.md、统一入口/doctor/registry、Issue #11/#47、原native campaign ledger及当前request/lease；已有实际证据优先于过时的README日期快照。记录adopted file blob、当前Git固定SHA、唯一owner、当前真实job state与资源占用，再执行下一步。
- 之前单owner已在Issue #11登记STRATEGY_RECORDED：6abe39a353fd655268ef67bdef6d53e90fcd8bff；长期GR00T+DAgger+world-model战略已固定在eecfd0ee3b43dcd73fc3a9b94b2dddd6e2794719。这些**不是新增PV1实验支线**。WBC与plant/腿策略/夹爪PD/S1/评分/GR00T执行配方全部冻结，selected/deployment仍NONE。

## 1. 确认的排队内存事实：不是“PV1默认225GB”

从当前源码 scripts/pi0294/pv1/pbs_owner.py（基线5a49c50d）逐行核实：
- prepare(..., memory_gb=128)，命令行--memory-gb默认128；
- 合法显式取值只有128或225；其PBS生成行是 select=1:ncpus=12:mem={memory_gb}gb:ngpus=1:gpu_model=H200。
- **所以当前qstat显示225GB，不能自动归因于用户选了225，也不能自动证明128足够**：已有同一Hopper历史明证，原本128GB的PBS请求经站点auto->small路径被改写成225GB，并造成可用内存不足；原唯一owner后来用成熟的确切owned queued-job内存下调，在**同job、不cancel、不重投**前提下把资源读取回128GB。参考原封存报告 reports/PI0294/20261016T191445Z-localpc-mp1/N1_SOURCE_CONFIRM_ALLOCATION02_ACTUAL_SUBMISSION.json。
- PBS mem是**计算节点主内存**，不是H200显存；128与225不是GPU计算显存版本。GB/GiB以站点qstat的真实单位为准。
- 最新人类提供状态：675967.hopper-m-02 Q、small因225GB暂不能满足、第四次PBS、原642查询/0新优化器更新/unknown=[]、未占GPU、原本冻结1c4cdc9a source、单owner稳定。本段是**记录时刻的待刷新快照**，不是永久实时结果。

## 2. P0｜内存和队列根因判定：最高优先、先只读

现有single owner进行一次有限的现场收据闭合，不用GitHub推断PBS实际状态：

1. 原生读取准确jobID 675967的 qstat -f、资源select/mem、Resource_List、job_state、Q/running原因、Queue/Project、Owner、submission_nonce、job脚本/JOB_SCRIPT_BINDING/REQUEST、原PBS号及既有campaign ledger；将源码固定SHA与最终native Request身份逐项关联。核对是否已经变R/F、是否有其它所有者作业或未知mutation；**不能因为GitHub最后一条Q快照而执行qalter**。
2. 判断原JOB_SCRIPT的请求究竟是128还是225；若128而qstat变225，标注SITE_REWRITE_128_TO_225_CONFIRMED；若脚本本身225，标注EXPLICIT_225_REQUEST_CONFIRMED；若任一成员缺失标注PROVENANCE_UNKNOWN。对比先前675394及合法早期B0模型加载、HF银行读取时的实际max RSS/highwater/虚拟内存及真实Resource_List，实测数据没有就标UNKNOWN。不要凭6.5GB bank或H200 VRAM推断128GB足够。
3. 对历史成熟memory-only原生作业调整进行源码/站点合同资格核对：只限自己目前仍Q、native job/project/nonce/owner/状态全一致、无已发生启动/分配、调整前后GPU数/CPU数/墙钟/源capsule/原请求hash/调度身份不变；原job mem改变必须得到站点支持，证据可核对，不能让请求脚本签名被篡改。已知128GB RAM峰值＋保守裕度、原实际成功资格和当前缓存/解包路径必要性应支持降配。
4. **只在原PV1工程权限允许且现有成熟owner-only、原生读回合格的情况下，允许一次确切既有队列job的同义memory-only修改**，并记录修改意图、准确jobID、旧新select、提交结果、qstat后读、无新增PBS/重复任务/已修改权限。若与本原生版本的immutable request或site权限冲突，停止于MEMORY_CHANGE_NOT_ADMITTED：不要取消/重投健康已排队job，也不要隐藏排队问题。若qalter效果未知，不盲重试；先按native journal和qstat确认真实副作用。
5. 若225是显式必要申请、128安全内存证据不足，**保留225**，提供真实调度等待与截止条件判断；不以单纯想提速而降低物理内存预算，避免OOM和浪费一个PBS槽。若job已R，按原胶囊运行，严禁热改。下次合法**新**job是否用默认128须由完整原生RAM峰值和核验后的资源门决定，不能照搬旧预算。
6. 把事实存入原PV1报告根新增的小型脱敏 P0_QUEUE_MEMORY_675967_PROVENANCE.json（仅收据，不泄露私有用户名/绝对路径/凭据）；必要时在Issue #47登记一次内存工程结论：DEFAULT128 / SITE_REWRITE或EXPLICIT225 / native peak或UNKNOWN / actual change或NONE / qstat proof / never re-qsub。不要在没做更改时报告“内存已恢复128”。

## 3. P1｜当前已排队作业的原生接续与DEV48完成

原B0第三作业675394已真实终态F/1，原本10 attempts=6 SUCCESS +3 VALID_TASK_FAIL+1 ENGINEERING_INCOMPLETE。9有效身份不能重跑；工程partial F1/002 T2已确认Core RPC原40000上限，唯一replacement合格。原642 draw全部绑定，0新训练。675967是**第四份独立既有PBS提交**，不能从0重新评价48个条件。

在当前job真实获得allocation且通过该allocation自己600s/80cycles/8fault原生通信资格、模型/银行hash与source身份后：
- 按原population继续仅剩的39个条件（包括唯一工程replacement），确保48个有效cell都有原结果、独立raw/causal/score/视频审计、相关历史attempt完整保留。任何valid失败保持失败；任何true native safety STOP立即停止受影响lane。
- 不重测9个原有效条件，不筛噪声，不放宽原参考合法性、S1与任务评分。Teacher不控制学生；完全对齐相机→新观测→模型数组→安装→ACK→物理、任务终末与原timestamp。
- B0完成后先关闭本PBS、两端worker/lease、实际资源结算和unknown reconciliation，才能发布PASS_COMPLETE_ACTUAL_B0_DEV48_EVIDENCE。**B0成绩即使低也不能增加新的成功率训练门**；所需仅真实完整且证据合格。
- 如果同一队列或物理瓶颈迫使7天窗口到达硬停止，实际保留已获得完整条件与失败/工程缺失，不声称48已完成，更不能为了训练绕过完整证据门。明确报告阻断在哪个GATE。

## 4. P2｜同父双臂微调（核心科学目标，不改配方）

前提：完整B0且审计/资源关账、先前训练卡兼容、在每个prepare、submit和每次新优化effect前fresh的全资源预算审核通过，足以保留最终工作。

- CORR@2048唯一父manifest SHA256 94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38；原v3.0统计量 fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7；PV1冻结的新训练银行139独立parent/77几何scene/62双任务配对，COVERAGE_FIRST固定。
- A_CONTROL与B_COVERAGE由同一父模型、分开的全新optimizer开始，各4096正式step，effective batch32（micro2, acc16），BF16、冻结视觉语言主干、同一原生loss和合法sampling。不要让B从A权重接着训练，也不要把所有窗口数称为独立示范。
- 第1024步按原协议做TRAIN-only探针，不能借探针提前筛DEV赢家；全4096终点进入新DEV48。样本消费、实际step、梯度、冻结参数、精确完整resume（包括optimizer/scheduler/RNG/sampler/processor/statistics）逐项落收据。现有checkpoint_reuse.py的CPU合同PASS不等于已完成原生复制。
- 将两臂预封存同一个性能观测计划；前3次真实CUDA比较梯度/checksum、前100步保留原审计、此后只有等价证明后才切批量标量读取；保留同样本游标/precision/语义与完整恢复。**不能只给B实验组单独做性能/数值改动**。不用第二GPU/登录节点训练。
- 当前baseline评估已测RPC阻塞3346.4s、同Hopper模型服务包络203.2s（642 draws），不等于GPU利用率。减少只读控制面IO、合法银行冷态复用、并行CPU预备可以优先；任何尚无native证明的加速不得用于降低预算预测。

## 5. P3/P4/P5｜受控迭代、唯一候选、独立封存验收

- P3仅在原PV1确证弱任务欠拟合、存在对应TRAIN证据、并有合格完整Recovery Oracle或事件采样单变量支持时启动**一次**预注册补强双臂；T2旧Oracle0/2不合格。若无证据、预算或工期不足，跳过而不自动开展新WM、DAgger扩采、codec重估或新horizon。
- P4按原worst-task/macro/S1/复杂度/早序排序，DEV每任务至少18/24、相对B0各任务退化不超1/24、worst-task有净改善；对胜出recipe用seed20261020做原协议许可的独立训练复核。若无稳定性复核，最多BETA，不可称VALIDATED。
- P5先冻结唯一共享权重、processor、统计量、动作ABI/数值、源/执行器身份、模型卡与评价plan，再只对这一候选打开LOCKBOX80：20个scene组 ×T1/T2×2固定噪声。原真正有效失败不得重跑；工程替补最多一次且须有完整前因收据。
- 目标PV1_VALIDATED：两任务分别≥32/40成功、≥14/20 scene两噪声均成功、每task-family≥50%、每任务S1拒绝≤4/40、真实native安全故障0、80格全独立审计＋训练复核＋运行包齐全；PV1_BETA各任务≥28/40、≥12/20 scene 2/2、family≥50%、每任务S1≤8/40，其余证据门同原协议。达不到则BEST_RESEARCH_CANDIDATE_NOT_QUALIFIED，不在已打开测试集调参再挑checkpoint。
- 必须最终交一份可运行的共享 T1/T2 源域EEF10策略包：权重/processor/stats/训练lineage/依赖锁/离线dry-run入口/score及S1矩阵/视频索引/复现说明/准确验收等级；不能把两个单任务adapter冒充一份权重，不把固定底座成绩冒充Go2+ARX5/WBC或真机部署。

## 6. 关键资源、并行方式与绝对时钟

**原PV1统一累计**：H200最多96 allocated GPU-hours、24 PBS、同时仅1张、单PBS≤8h；优化更新最多16384、神经查询48000、物理/构造/CPU/存储/网络按原卡更小者；最终保留20 H200-hours、100episode、9000查询与对应tick/storage。此前三个已结算GPU作业耗11154s（约3.0983h）；第四份675967已提交，**排队reservation不当spent**。不能把多个project总余额当多倍scientific预算。实际剩余由原生Hopper+CoreLab+local三端账本重新核实。

绝对截止，均以2026-10-19 02:29:56 SGT为原T0计算，不因跨聊天会话、排队或这个Goal重置：
- 2026-10-25 02:29:56 SGT：停止新job/新candidate；
- 2026-10-25 14:29:56 SGT：停止新训练块；
- 2026-10-25 18:29:56 SGT：停止新物理试验；
- 2026-10-26 02:29:56 SGT：强制所有工作结算收口。

允许已有唯一owner在等待中做**CPU-only不变更科学语义的准备**：P3/P4/P5必要的代码可达性与单元测试、case审计、峰值内存证据、原生资源预测、已实现checkpoint冷续接必要接口核对及已封存报告增量清理。已完工测试与日志不重复运行/反复计数，不接管第二owner，不热改排队actor。健康队列Q本身不代表真实 GPU占用，不可凭Q开展神经推理、训练或私自开启第二作业。

在给下一份合法GPU作业分配预算前，**必须**测算B0剩余、两臂训练与DEV、随机种子、最终80条件及必须的安全审计/提交/closeout；不能因为训练优先就挪用最终20h。若现有实际时钟/资金让全部强制工作在原门内不可能完成，应明确标为BLOCKED_BY_EXACT_RESOURCE，停止可选支线，保留最强合格研究候选和现有真实分母，向用户报精确所需新增权限；本Goal不默认为用户延长PV1。

## 7. 失败处理、防幻觉验收和连续汇报

- 每个本地/原生阶段的主状态必须属于：PREPARED_ONLY、NATIVE_SUBMITTED_Q、NATIVE_R_QUALIFIED、PARTIAL_EVIDENCE、COMPLETE_AUDITED、ENGINEERING_INCOMPLETE、STOPPED_BY_SAFETY、BLOCKED_BY_EXACT_AUTHORITY、CLOSED；不把Git提交或CI通过当作native运行。
- 保留过去的9有效B0结果、原工程partial、原reference/S1拒绝、STOP、源模型与镜像身份；不拿成功视频替代全部分母。原标准纯读侧误差修复不准更改动作/安全决定。
- checkpoint维护沿当前唯一清理lane：完整后继稳定且解除所有引用后，清理确证废弃无消费者的失败大payload；必须保留原CORR父、各匹配对照终点、最新完整resume、必要回滚、科研负例和未知副作用证据。无可删对象报告0；不反复全盘扫描。
- 每一关键阶段向Issue #11汇报科学结果、向#47汇报必要的内存/工程动作；最多每4个实际活跃小时一份有内容的里程碑，在新submit或修改已注册job等mutating操作后即时保存原生intent/result/readback。远端Codex在读取本固定版本后仅报一次LONG_GOAL_ADOPTED或EXACT_BLOCKER。不能为了汇报而刷屏或频繁ssh polling。
- 用户要求持续推进的是**同一已授权PV1 Goal直到真实终局**，不是另开“世界模型”、更换模型骨干或申请额外预算。长期VLA/DAgger/WM战略按原eecfd0ee文档保持未来候选、另有科学GO才实施。

本文件的GitHub发布仅是用户授权范围内的Codex接续指令：不表示当前675967的225GB已修正、队列已出Q、任一GPU作业已成功、A/B已训练或PV1已发布。实际终局只以唯一原生owner收据判定。
