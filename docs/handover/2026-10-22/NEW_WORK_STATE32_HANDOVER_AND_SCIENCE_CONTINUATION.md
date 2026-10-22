# 新 Codex Work 窗口：安全接手 state32 续训并完成科学主线

日期：2026-10-22，Asia/Singapore。交接 ID：VAP1_NEW_WORK_STATE32_HANDOVER_V1。
状态：HANDOFF_PUBLISHED_NATIVE_TAKEOVER_NOT_PROVEN。用户要求仔细审核进度并让新 Work 窗口接手；本文件落实接手和执行顺序，**不是新科研分支、额外计算额度或第二个调度 owner**。
仓库：Leo-6-maker/go2-arx5-hierarchical-vla；集成分支：pi0267/executor-decoupled-recovery；campaign：20261018T182956Z-localpc-pv1。
审核基线：369644855d475e91d8a95d584bd003eebcf8653a。最新公开工程回执为2026-10-22 12:06:30 SGT；其中最近明确队列现场读取为11:52:14 SGT。**没有直接SSH核验本文件发布时的状态；接手必须先刷新现场。**
机读导航：reports/PI0294/design/NEW_WORK_STATE32_HANDOVER_TASK_CARD_20261022.json。它只索引既有权威，不代替原生准入。

## 0. 第一条指令：接住当前实验，不重新开一遍

不要先qsub、qalter、cancel、重启owner、重新下载数据或从CORR重训A。先只读确认同一既有作业、控制服务、租约、checkpoint与账本。
用户不要在确认新窗口能重连原服务或安全交接之前关闭承载旧owner的会话。新窗口若没有本地环境/SSH/注册入口访问权，报告ACCESS_REQUIRED；仅能读GitHub不等于现场接管。

近期唯一关键路径：**保护当前A1 → 从完整step32真实CUDA恢复 → 1024完整保存与原12条TRAIN探针 → A2048 → A4096 → 同父独立B2048/B4096 → 授权范围内的真实配对评价/DEC1解耦证据。**
数据和Teacher工作已完成本轮可做部分，不再把扩采、额外测试框架或新恢复数据当训练前置。保持WBC冻结。

## 1. 审核结论：三种“进度”不能混淆

| 对象 | 最后可证结果 | 新窗口的处理 |
|---|---|---|
| 旧A / 1474116 | 真实1024更新，但保存前guard误拒绝已批准Cosmos输入alias；无可恢复完整状态，终态已结算 | 保留1024费用与失败lineage，绝不伪称能续到1025 |
| 恢复A / 1474352 | 新lineage完成32更新；成本门9.010780406s超过旧6.1s，随后可选共享报错；**完整checkpoint32已先保存并独立验收** | 只接续这份状态，不能从0再跑32 |
| 当前A1 / 1474599 | 已提交一次，最后Q/node_pool不足；目标32→2048，尚无公开CUDA导入/更新结果 | 先看最新Q/R/F，不重投、不改现役源码 |
| B_COVERAGE | 0更新 | 从原共同CORR父独立开始，不从A开始 |
| 累计更新 | 1056 = 旧不可恢复1024 + 新可恢复32 | 不是一个模型已到step1056 |
| 新模型终点 | A4096=0、B4096=0 | 没有新policy闭环改善结论 |
| PV1 B0 | 14/48，9成功/5有效失败 | 原分母保留；不能把Teacher结果当policy结果 |
| DEC1核心对照 | 0/64 | 原CORR/R0、源/目标执行器、I_TIME/I_PROGRESS矩阵尚待真实执行 |
| 数据 | 原139及Original126+C1已真实读取并关闭；CR2 24条通过；VAP1 D48为48尝试/46准入/24新几何/22配对组 | 固定资产直接复用，不重采重审未变化的数据 |
| Teacher | T0保存TRAIN重演通过；dev/verify=0，未绑定合格完整学生状态 | NOT_TESTABLE，不声称DAgger-ready，不继续无依据物理搜索 |
| selected/deployment | NONE | checkpoint可恢复不等于策略已验证或可部署 |

step32关键身份：
- lineage：A_CONTROL_SAVE_RECOVERY1；arm：A_CONTROL；native step：32；sampler consumed cursor：1024。
- checkpoint相对路径：training/A_CONTROL/checkpoint-32；宿主绝对路径从私有registry/runmap解析，不猜测。
- CHECKPOINT_COMPLETE SHA256：1c50579c172367894fe83921d84e4053803dd451c1c3e693d7881f1ff673e6f8。
- 15成员，25,540,965,550逻辑字节；模型、optimizer、scheduler、RNG、sampler、processor、R0与recipe检查通过。
- 该核验没有加载CUDA张量；真实GPU续接仍是下一项必须验证的效果。

## 2. 既有授权优先级与必须读取的文件

先读AGENTS.md、START_HERE.md、docs/operations/2026-10-13_UNIFIED_ENTRY_AND_GITHUB_SYNC_V1.md、docs/operations/WORKSPACE_ENTRY_ENFORCEMENT_V1.md、infrastructure/workspace/ENTRY_POLICY.json及原生registry。使用实际部署的host-aware入口，不假设CoreLab的enter.sh已经在Vanda部署。
然后按对象阅读以下明确权威；后来的明确修订只替换相应条款，不重写历史：

| 决定 | 固定提交/文件 |
|---|---|
| 原PV1配方与预算 | 9eff29a44a193dd4d3620dc0bfe5764a402eff42；docs/protocols/PI0294_PV1_MULTITASK_POLICY_RELEASE_20261019.md |
| DEC1与训练依赖修订 | a7a7ecd9a6be62a7221af25a5e32a0009b076a84；docs/protocols/DEC1_CONTROLLER_DISJOINT_VLA_SCIENCE_GO_20261021.md |
| VAP1执行与非GPU子帽 | 43f7fe6f86c683f2e38e5c8802eb8cde208e575c；docs/handover/2026-10-21/VAP1_VANDA_AUTONOMOUS_POLICY_DATA_TEACHER_GOAL.md |
| 同项目H200/A40各1张并发 | 4b6ef6e3e6e741e423d7561d67312008b9b68024；reports/PI0294/design/VAP1_DUAL_PLATFORM_CONCURRENCY_20261021.json |
| 旧1024丢失后的唯一A重建、累计9216步 | 1d84e4cfa4f10c84d0cb33c42b06129b048c90c4；reports/PI0294/design/VAP1_A40_LOST_SAVE_RECOVERY_AUTHORIZATION_20261022.json |
| **当前有效分段续接** | **3335c8620c29d8d58f6b5e19075f3943e2dd1904**；reports/PI0294/design/VAP1_A40_SEGMENTED_RECOVERY_AUTHORIZATION_20261022.json |

旧草案中的NOT_AUTHORIZED、旧8192/48h/8PBS子帽、旧必须先B0全48才能训练、旧单卡和两份8h训练默认，不能盖过后来相应明确批准。反过来，后来的文件存在也不自动证明已原生采用；核对两份原生账本的真实grant marker。
**当前已批准的32→2048→4096恢复不需要新窗口再请求一次相同科学GO。**

## 3. 安全接管：一个logical owner，不让换窗口造成停机或重复效果

### 3.1 第一轮仅做只读确认

读取本机现有控制服务/进程及启动身份、当前runmap/manifest/lease、两端完整本人PBS队列、job/request/nonce、更新/查询intent与commit、checkpoint索引、实际used/reserved/unknown及当前认证状态。工作区dirty和本地唯一文件先保留，不git reset/force或重建workspace。

当前已报告服务为vap1-a40-segment-a1-owner-01，最后PID630834、active、Restart=no、Q60s/R30s。PID和active只是旧快照；必须联合进程start identity与原service/run绑定复核，不能按单个PID发kill。

### 3.2 三个接手分支

- **旧服务健康且新窗口可复用同一本地会话**：复用原控制服务继续；新窗口是新的交互界面，不另起调度者。确认心跳、放行和归档确实仍由同一actor负责。
- **旧服务健康但需转移控制**：只使用已经实现并受检的排他交接路径。没有这种路径则保持旧服务，等合法阶段边界；新窗口只读旁观。不能先停健康owner再临时开发接管脚本。
- **旧服务确证终态/已失去控制**：先核清现场job是否仍在执行及所有未决效果；沿既有受检恢复机制，在证明旧actor不会继续写入并取得排他所有权后，建立同一logical owner的接续记录。保留原job、nonce、T0、输入、运行source和账本。租约超时、SSH断开、聊天关闭都不单独证明可抢锁。

Q/R任务若因owner丢失触发原watchdog/STOP，不能伪造心跳或取消STOP；按真实终态/完整checkpoint接续。凭据仅走现有安全认证，不写Git、报告或聊天；MFA/账号撤权无法自助时精准请求必要人工操作。

交接首回执只允许：REATTACHED_EXISTING_OWNER / SAFE_OWNER_HANDOVER_COMPLETED / WAITING_SAFE_BOUNDARY / ACCESS_OR_STATE_BLOCKED。现场未验证时不得写TAKEOVER_SUCCESS。

## 4. 当前A1：新窗口最先保护的确切对象

- PBS：1474599.stdct-mgmt-02；project：CFP04-CF-007。
- request ID：75e12289-7b99-4f46-bd69-bbe64ff9cad0；nonce：8e718e1c-b5d4-46ba-b2f4-00dd8520d2cd。
- request SHA256：b2d088dec7aa9174970b19284b3f4ee85206d3819941899dd65be295d357c651。
- **运行源码**：2b141e9f40babfa749b508f33ea5f6057f59ba33；精确CI39718431394/job119626031301成功。
- wall：22500秒（6h15m）；1A40。申请4CPU/32GiB，实际站点36CPU/240GiB是事前允许capacity；程序4线程、RLIMIT_CPU53760秒、记账上界54000秒。不得再把合法站点profile当资源故障重复qalter。
- 新更新预约2016（32→2048）；另有22560 GPU秒/29queries/54000 CPU秒/70,000,000,000输出字节预约。
- 两端source/lease/manifest及step32身份在实际收据中；最后unknown/pending空，但新窗口必须刷新。

### 状态驱动动作

Q：维护同一job的轻量观察和合法认证，不取消/重投、不启动同平台重复训练，不为了换窗口重置等待。
R：优先同job新鲜身份/认证及原放行链；由原runtime在已付费allocation内导入step32，再验证全成员和完整训练状态。首次CUDA恢复应得到下一次真实committed update，不能重跑32步或重新初始化optimizer。
F/终态：先读取真实effects、完整保存、FINAL与raw日志，确认所有query/update归属；原生费用一次结算、原预约释放、两端lease/registry关闭后，再准入后继。异常外层global_step不能覆盖已验证effects。
H/UNKNOWN：不qrls绕站点、不盲重试mutation；保存准确阻断，其他独立只读准备可继续。过期qstat历史不能单独判定job未发生，复用已有终态证据与当前队列分类。

A1必须：step32完整GPU恢复→继续到1024→完整保存并验证→原12条TRAIN-only探针→保留optimizer/scheduler/RNG/sampler继续至2048→完整终态与关账。读取探针结果，不用探针看DEV选中间权重；不因低loss/单视频跳过失败门。若当前job已完成更远，以原生证据更新，不回滚到本文快照。

## 5. 后续四段训练与checkpoint关键路径

批准顺序为A1(32→2048)、A2(2048→4096)、B1(0→2048)、B2(2048→4096)，四次训练提交包含已经提交的A1；不是接手后额外四次。新的工程失败不自动创造额外重跑或lineage额度。

### A2/B2

后半段intake代码2ba5211e738c24e7848a169687af9f043ce7ecd0及CI39720002227/job119631028448已CPU通过，**没有部署到当前A1**。只在未来冷态scope使用，必须取得源段真实terminal/closed lease/typed settlement，核验同arm的1024和2048完整状态、原12探针数组与结果，update2049前复核而不重复生成查询。没有2048完整状态就不能绑定A2/B2。

### B1

B从原CORR@2048权重建立独立optimizer/seed/sampler轨迹，绝不继承A。A/B固定同一资格通过的平台、精度、loss、effective batch32(micro2×acc16)、原R0、原配方/各4096终点。共享数据支持不等于相同抽样曝光；B的composite identity必须使用已发布的B专属实现，不直接复制A的身份。

### 保存与故障处置

复用成熟完整Trainer保存，保存后验证模型、optimizer、scheduler、RNG、sampler、processor/stats、manifest和实际cursor。可选frozen-shard共享只允许现有严格实现；不能把inference-only父目录当完整训练恢复源，hardlink也不算独立备份。保存成功、共享失败时先核验已有完整状态，不自动宣布进度丢失。

每个未来cold job在模型加载前检查封存的Cosmos输入alias、确切target、output root与实际磁盘瞬态需求；输出越界仍拒绝，不删除全部symlink安全门。不要在活跃A1中热打补丁。必要修复限最小源码/回归/精确CI/未来cold绑定，不新造checkpoint框架。

保留CORR父、唯一可恢复state32、科学1024和4096状态、所有失败与费用。工程32/2048只有在完整后继独立验收、所有消费者关闭、原删除权限覆盖时才可清理其冗余副本；换窗口和存储紧张都不是删状态的理由。

## 6. 科学主线：训练终点是资产，不是论文结论

### 6.1 源域匹配对照

A_CONTROL与B_COVERAGE按原同父/相同步数/任务曝光完成，比较新增独立示范覆盖的收益。DEC1明确允许先训练而不等B0全48；**原PV1 baseline-relative选模、等级发布和LOCKBOX80仍要求完整B0及原DEV门**。旧14格有效结果不能重跑或跨平台无资格混拼。
任何新增DEV48/A/B评价均须已有scope及独立fresh资源准入。当前分段forecast主要覆盖剩余训练和DEC1矩阵；不能在其上无成本再追加96个DEV和80个LOCKBOX。需要同时做时明确列总成本并报告缺口，不偷减分母。

### 6.2 VLA–WBC解耦核心实验

核心问题保持target-controller-disjoint：高层不从目标WBC任务数据学习，目标WBC代码/权重/增益由对方团队冻结。首轮预注册模型仍是原CORR@2048/R0，不因新A/B或目标结果临时换赢家。
DEC1核心64格：8新几何×T1/T2×源/目标执行器×I_TIME/I_PROGRESS。必须先取得真实目标EXECUTOR_MANIFEST、frame/TCP/gripper、状态时间、安全/预测/reset/stop与实际可执行入口，并完成原有限commissioning。目标缺失明确TARGET_EXECUTOR_NOT_DELIVERED，不伪造理想跟踪、焊死底座或改WBC凑结果。

等待A1时只做CPU接口/原生部署资料核对和评价输入/预算预备；没有新的目标资格证据，不声称两个接口的CPU测试已经证明迁移。目标数据不回流TRAIN，不用Teacher phase/物体oracle控制高层接口。评价任务成败、参考拒绝、S1、物体过程约束、跟踪和等待延迟分列，以几何组处理配对相关性。

已批准最多Hopper H200一张＋Vanda A40一张并行，但必须同一个协调owner、各平台无重复链、完整两端队列与预算按GPU秒之和扣。它不授权用另一卡重复A1竞速，也不允许A在A40/B在H200后直接归因数据差异。当前既定续训不因换窗口迁平台。

## 7. 停止反复准备：数据与Teacher只保留已证资产

原139银行及Original126/C1已经通过实际reader并关闭；从原消费回执解析路径，不重新下载或重算所有不变成员来刷进度。C1的8行来自4个学生parent，不宣称8独立示范。
CR1 RANGE_SUPPORTED、CR2 24条R1准入、VAP1 D48的46条准入都是独立研究资产。当前A/B使用R0，**新R1不得热注入**，也不借R1扩采重置本轮对照。
VAP1 D/T已关闭，剩余未使用escrow不能重新当许可。Teacher保存数据T0一致不等于恢复Oracle；未绑定完整学生状态/历史时保持NOT_TESTABLE。无新数据、World Model、大主干、额外DAgger、Teacher物理探索或WBC修改授权。

## 8. 预算与时间：换窗口不重置，当前预测余量很窄

当前已批准：global96GPUh/24PBS/16384updates；DEC1 52GPUh/11个新增PBS/9216累计updates/96GPU-site CPU核时/176GiB输出；最终20GPUh与40GiB输出保护保持。其他CPU/IO/物理上限沿已采用原scope；不得由本文补额度。

最后已报global spent27701 GPU秒、globalPBS16（含当前A1）、DEC1 spent7207 GPU秒/newPBS5、queries1088、累计已花1056updates。当前A1预约22560GPU秒/2016updates等与未来总forecast有包含关系，**不能双加或把预约当已用**。

最近完整forecast（不是保证）：包含当前A1的未来7job，上界176820GPU秒/8160updates/296000GPU-site CPU秒。独立重算：
- DEC1 7207+176820=184027秒，对187200秒子帽仅余3173秒；
- 当前5个DEC1提交已含A1，未来7也含A1，完成后5+7-1=11，**该安排不留DEC1提交余量**；
- global对应16+7-1=22，剩2次全局提交，但它们不是自动取得的DEC1追加权限；
- DEC1 CPU43341+296000=339341秒，对345600仅余6259秒；
- global费用含最终保护27701+176820+72000=276521秒，对345600尚有69079秒。但全局余量不能自动越过更紧的子帽。

重新读取实际剩余CPU、存储峰值、站点计费和队列延迟；上述预算完成日期不是排队ETA。保护已投入进度、减少重复输入检查和新短probe，比新建一套调度系统更有价值。若新实测或排队使计划不可支付，**在下一次submit前**明确给出最小资源/时间缺口与受影响科学结论，不盲重复已经不成立的forecast。

原时间：T0=2026-10-19 02:29:56 SGT；2026-10-25 02:29:56停新job/候选；10月25日14:29:56停新训练块；18:29:56停新物理；2026-10-26 02:29:56硬收尾。分段不会延长它。10月23日20:00报告真实神经结果或准确阻断，不承诺正结果。

## 9. 自主修复范围与升级门

已有同范围授权可自主处理读侧DNS/SSH/NFS短异常、旧PBS历史过期、权限/alias/argv、已知source/data绑定、认证合法续期与有限费用对账。明确zero-effect后才能修复并纠正一次；mutation效果未知先journal核查，不盲重发qsub/release/query/update。
同根因反复失败先停止受影响链，形成最小复现和真实layout回归；不因为CI绿就立即烧下一次GPU。技术测试/真实目录只读检查/真实模型加载/训练/保存恢复/任务成功必须分层报告。
系统hold、MFA/权限撤销、真实安全错误、缺失完整状态、额度或deadline不足必须保留原STOP并报告精确阻断；不能用本交接作为新增预算或第三条重建lineage的许可。
外部独立项目完整可见但不接管/取消/改资源/误计。本计划不批准删除数据或公开私有checkpoint/WBC/凭据。

## 10. 第一份接手回执与最终交付

采用后只发一份NEW_WORK_STATE32_HANDOVER回执：
1. 固定交接SHA、管理HEAD、实际run源码/CI三者分别记录；
2. 接手方式与唯一owner/lease排他证据；
3. 当前1474599真实Q/R/F、request/nonce和已提交效果；
4. latest verified full state、step/cursor、训练lineage；
5. global/DEC1 used、reserved、unknown、剩余clock及预算；
6. 下一项合法动作，或唯一最小阻断。

每个实际里程碑落小型固定SHA报告并在#11科学/#47工程登记。只写必要短导航，不把整个交接反复粘贴START_HERE。未运行写NOT_RUN，未知写UNKNOWN，不用测试数代替实际更新。

最低成功交付：A/B两个4096可复现完整终点及训练消费/故障/费用记录；在相应scope与预算成立时，配对源域评价和DEC1原模型64格结果；真实策略发布仍按原独立验收。不能完成全部时，交付已真实完成的模型、未完成分母、目标ABI状态与准确资源阻断。严禁将局部loss、32步checkpoint或Teacher成功率包装为合格policy。

### 核心证据入口（固定审核基线中可读）

- reports/PI0294/VAP1/20261021T042730Z-localpc-vap1-data/VAP1_RECOVERY32_FULL_STATE_AND_TERMINAL_ACTUAL.json
- reports/PI0294/VAP1/20261021T042730Z-localpc-vap1-data/VAP1_SEGMENT_A1_AUTH_ADOPTION_AND_GPU_SUBMISSION_ACTUAL.json
- reports/PI0294/VAP1/20261021T042730Z-localpc-vap1-data/VAP1_NEXT_SEGMENT_FULL_STATE_INTAKE_COLD_READY.json
- reports/PI0294/VAP1/20261021T042730Z-localpc-vap1-data/VAP1_D48_CLOSED_DUAL_QUEUE_AND_CPU_CACHE_ACTUAL.json
- #11/6332290200：分段已批准并实际采用；#47/6332426815：下一段代码已CI，不是2048已产生。

本文件的GitHub发布没有SSH、qsub、控制权转移、模型、物理或清理效果。只有新窗口现场回执能够证明接手；不能仅凭打开此文档就宣称原生工作已继续。
