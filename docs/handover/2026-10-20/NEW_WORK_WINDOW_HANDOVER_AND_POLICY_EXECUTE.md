# 新 Work 窗口接力：接住现役 PV1，完成共享 T1/T2 policy

日期：2026-10-20，Asia/Singapore。文档 ID：PV1_NEW_WORK_WINDOW_HANDOVER_V1。
状态：HANDOFF_PUBLISHED_NOT_NATIVE_TAKEOVER。本文件是用户要求的新窗口接手指令，不是新实验、新预算、新七天窗口或已完成的 owner 交接。**本文件是单一自包含交接入口，无需另一个新机读附件。**

## 1. 接手对象和阅读顺序

目标：承接同一个 PV1 campaign 的真实现场，尽快完成「基线 → 匹配微调 → 闭环比较 → 合格则独立验收」，不从聊天历史重建工作区或启动第二位调度 owner。

- 仓库：`Leo-6-maker/go2-arx5-hierarchical-vla`。
- 集成分支：`pi0267/executor-decoupled-recovery`。main不是当前科研运行版本。
- Campaign：`20261018T182956Z-localpc-pv1`。
- 原已采用权威：`9eff29a44a193dd4d3620dc0bfe5764a402eff42`。
- 原科学协议：`docs/protocols/PI0294_PV1_MULTITASK_POLICY_RELEASE_20261019.md`。
- 原机读资源卡：`reports/PI0294/design/PV1_MULTITASK_POLICY_SCOPE_20261019.json`。
- LG04已采用：`738a369c25dd24fff408bdedae0120f010623cfd`；本交接只更新快照与接手方式，不修改科学门。

先读本文件，再读 `AGENTS.md`、`START_HERE.md` 最新段落、原PV1协议/资源卡、Issue #11科学和#47工程最新原生回执。工作区仍按 `docs/operations/2026-10-13_UNIFIED_ENTRY_AND_GITHUB_SYNC_V1.md`、`docs/operations/WORKSPACE_ENTRY_ENFORCEMENT_V1.md`、`infrastructure/workspace/ENTRY_POLICY.json` 进入。CoreLab入口为 `~/Projects/GR00T-WBC`；只使用现场已部署的 enter/doctor/registry。不要猜私有路径、账号、session或未实现的命令。

## 2. 最新审核快照：不能沿用child06仍排队的旧信息

审计管理版本：`3bd5d62fcbe96ee1448d9e9b7e0d5763cb6ff214`，2026-10-20 09:05:50 SGT。最后科学评论 #11/6280903573，09:06:22 SGT；工程评论 #47/6280903261。下表是这些收据的状态，不是新窗口启动时的实时状态。

| 项目 | 最新已证状态 |
|---|---|
| 当前工作 | **policy-b0-08 / 677341.hopper-m-02**，PV1第7次PBS，CFP04-CF-007，1 H200 / 12 CPU / 128GB / 8h。 |
| 运行源 | **fb00da44f122c3ce96d0970ba23c2f223470cd33**，不是管理HEAD。精确push Actions **39348602625** / job **118387138067** completed/success已独立核验。 |
| 原生阶段 | 最新收据为R、PBS_READY=true，同一job已release一次，owner活跃，premodel qualification正在执行；**尚无资格通过、MODEL_READY或新任务结果的收据**。 |
| B0 | **12/48有效：8成功、4有效任务失败**；余36固定条件待完成。模型加载前工程故障不计task fail。 |
| 新训练 | **A_CONTROL=0、B_COVERAGE=0 optimizer updates**。现有B0成绩是旧CORR@2048在新DEV的结果。 |
| 数据 | COVERAGE_FIRST，139条合格新示范：T1 75、T2 64；T2 +Y 48/-Y 16；83539个重叠H40窗口，不是同量独立演示。原codec/stats已审计。 |
| 费用 | **15398 H200秒已结算，约4.277小时**；另有28860秒预留，当前在途费用需原生刷新。已用7/24 PBS，累计查询862、更新0、unknown=[]。 |
| 交付准备 | P5_FINAL、LOCKBOX80和离线策略包已有CPU合同实现；LOCKBOX未打开，selected/deployment=NONE。 |

旧N1、FD1、FD2已经结束。旧CORR@2048独立确认T1=10/20、T2=4/20；FD1确认了20次指垫碰桌预测，另一次碰撞重建未合格；这些结果不重新评分、不把原确认转成新隐藏测试。FD2的小型专家裕度结果不等于新策略学习收益。下一任务不是重做它们。

### 最近child的去向

- child06 / 676926：模型加载前，隔离concurrent_replace夹具遇到NFS ENOENT，已两端关闭；新增模型/查询/任务/积分/训练均0，保守计费634秒。修复针对隔离夹具中的有界读取恢复，生产安全门没有放宽。
- child07 / 677332：模型加载前，重新绑定的sealed fixture为0644而不是要求的0400，已两端关闭；新增科研效果0，计费58秒。这是最后绑定产物的静态检查缺口，不是模型能力失败。
- child08 / 677341：沿修复runtime冷绑定，两端最终sealed输入0400已回读；Core零积分模型/双摄资格通过。站点128→225GB改写已由原owner一次原地恢复128GB，现已R，不重复资源修改。

固定证据均在审计管理提交下：
- `reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_B0_06_PREMODEL_NFS_VISIBILITY_FAILURE_AND_REPAIR.json`
- 同目录 `PV1_B0_07_IMMUTABLE_BINDING_PREMODEL_FAILURE.json`
- 同目录 `PV1_B0_08_COLD_BIND_AND_NATIVE_RUNNING_RECEIPT.json`
- 同目录 `PV1_FIRST_FROZEN_BANK_AND_HF_ACTUAL.json`、`PV1_DATASET_CARD_AND_ALL_CAPTURE_ATTEMPTS.json`
- 同目录 `PV1_P5_CPU_CONTRACT_IMPLEMENTATION_RECEIPT.json`，实现历史固定917fcda。

**审核结论：当前最直接的瓶颈是冷绑定、资格与通信可靠性，新增数据的A/B收益还没开始测量。不能把大量工程测试通过当成policy已改善，也不能由这些工程失败否定模仿学习。**

## 3. 新窗口的第一个动作：安全接力，而不是qsub

不要因换窗口关闭旧owner、重启作业、改活跃manifest或再开一位dispatcher。新窗口存在不等于第二位操作者已获得写入权限。

### 3.1 首先只读核对现场

通过实际可用连接和原入口核对：owner的host/PID/启动时间与可重连session；PBS实时Q/R/F、job ID/request/nonce/project；两端run/lease、源版本、当前阶段、最后有效心跳；正在执行的condition/query/install/ACK；已提交训练step/cursor；各站ledger revision、已用/预留、pending/unknown；管理checkout的dirty与未发布修改。私有明细留原注册工作区，Git只存脱敏索引。

不能因聊天关闭、SSH断开或锁文件缺失断言旧owner已死亡；不能把尚未推Git的有效原生结果当不存在。最新实测结果覆盖本文件快照。

### 3.2 接手三分支

**A. 原owner/session健康且可重连：**复用既有受支持会话与同一个调度进程，新窗口只作为交互入口；不创建第二个heartbeat writer、lease或qsub循环。若新窗口没有该连接能力，先只读，不自行启动替代owner。

**B. 确需转移控制：**用已部署、已验证的单owner交接流程，旧owner明确交出控制并持久化状态，新窗口核实现场后接受。排他性必须由实际工具证明。若没有安全的在途交接能力，保持旧owner完成当前安全边界后再接手，不取消健康PBS来制造边界，不热改签名中的owner字段。

**C. 旧owner已死或状态不明：**先核实两端worker/job、最后已提交副作用和原watchdog状态；对未知query/install/qsub按原journal与原生效果对账。核清前不续发动作、不删除锁、不重投或复用失效lease。安全关账后才能按原恢复机制对真正未完成条件冷续接。

新窗口首次反馈使用：`ATTACHED_EXISTING_OWNER`、`TAKEOVER_PENDING_SAFE_BOUNDARY`、`TAKEOVER_CONFIRMED` 或 `BLOCKED_EXACT_ACCESS_OR_STATE`。只有同时证明单一有效调度者、原作业状态完整、账本一致和无重复提交后，才可报告接手成功。

**在确认连续性之前，不要关闭承载现役owner的旧会话。** 本指令允许接力，不授权绕过权限、原停止机制或生产安全检查。

## 4. 后续唯一关键路径

### P0：保住child08，完成B0

现场若仍是原job R，保持其runtime/capsule，不重复release；已完成的本allocation资格不重复，尚待资格则读取真实结果，不把R当MODEL_READY。若已产生新有效条件，先合并原生证据并重新计算剩余集合。

使用现有`remaining_panel`和全部原attempt。正常余36格（T1/T2各18，包含F1/004/T1/N294101唯一获准工程替补），不能把36格再加一条重复replacement。有效失败、参考拒绝和S1拒绝全部保留；以前12个有效条件不重跑。

一份8h分配不保证完成全部余量。按原closeout规则保存有效结果、标明未执行条件，核清效果并关账；后继合法job只跑剩余条件，不重做48格、不复活CLOSED的旧child。

### P1：把未来可在CPU发现的错误挡在qsub之前

不干扰child08。对**未来冷绑定**，在最后一次binder写入之后、下一次qsub之前，用现有校验器核对实际最终文件：0400权限、所有者、非symlink、manifest成员/哈希、schema、fixture/request引用、nonce、稳定quota、project权限、有效lease、输出根、预算和unknown；并检查prepare/submit没有改写sealed成员。不要等拿到GPU才发现0644这类静态错误。

需要修复时，最小复现→针对性CPU回归→普通提交→精确source的CI成功→冷态部署；不重构整套框架，不删除生产安全门，不热改活跃源码。每份新allocation仍保留原600s/80cycle/all-fault原生资格，同一健康allocation不按checkpoint重复验证。

CI以精确`head_sha`的Actions run/check-run为准。只查PR触发的连接器可能漏掉direct push，可GET `/actions/runs?head_sha=<full_sha>`再读实际run；legacy commit status的pending+空statuses不证明Actions失败。当前运行源的成功run已在第2节，**纯文档handoff提交不应成为现役作业的新runtime或重做CI前置**。

### P2：B0完整后立即执行封存的匹配微调

现役硬门是`PASS_COMPLETE_ACTUAL_B0_DEV48_EVIDENCE`在新optimizer之前成立。它是完整性门，不是成功率门。等待期间仅准备A/B冷态输入、增量审计与全流程预算；不采新银行、不重做FD1/FD2、不再建第二套P5。

固定父manifest：`94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38`（CORR@2048）。
统计量：`fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7`。
新银行manifest：`60f7c35a05766fe23974eba3a72f53ca4db5545a3768e6c1cfea164ab4e58294`；READY revision：`e4f603a1b768018077a8b5c9640250be67500902`。

| effective batch32的数据来源 | A_CONTROL | B_COVERAGE |
|---|---:|---:|
| 原T1 | 8 | 4 |
| 新T1 | 0 | 4 |
| 原C1 T1纠正 | 8 | 8 |
| 原T2 | 16 | 8 |
| 新T2 | 0 | 8 |

两臂从同一父权重、各自全新同配方optimizer开始，各4096正式更新；microbatch2/accum16、seed20261019、LR1e-5→1e-6/warmup64，原flow-matching、VLM冻结、原trainable map及数值规则不变。B不能继承A的新权重。

每臂1024保存点做原TRAIN探针，完整恢复optimizer/scheduler/RNG/sampler/processor/stats后继续到4096，不从头重训或把1024作为DEV选模点。各4096终点分别评同一DEV48。银行/完整checkpoint冷复用必须有原生hash和实际消费证明；性能实现两臂一致并通过原CUDA资格，不能借换窗口改batch/codec/loss/horizon。

### P3：先形成科学结论，再考虑附加实验

发布B0/A/B全部48格及所有attempt，按task×scene×bank对齐；报告B-A及各臂-B0的成功差异、最弱任务、参考拒绝、S1拒绝、32s目标失败、工程缺失和真实费用。按scene group处理相关性；新T1/T2同时替换属于联合覆盖干预，不单独归因T2数据。

可选P3补强默认不启动。只有原任务特定恢复/事件证据门与全预算同时成立才执行；旧T2 Recovery Oracle 0/2不是合格监督。WorldModel/扩展DAgger/新codec/历史输入等仍是另行授权的长期方向，不成为当前前置。

### P4：唯一候选、独立验收和策略包

原DEV门：每任务>=18/24、任一任务相对B0下降不超过1/24、最弱任务严格净改善、0原生安全故障。按worst-task→macro→S1→复杂度→早ID选唯一候选。

资格和完整未来资源同时满足后，冻结唯一权重/processor/stats/代码/数值/动作接口，再开LOCKBOX80（20新场景×2任务×2噪声），不得回流训练、筛场景、挑noise或替换第二模型。第二训练seed只有保护最终资源后能容纳才做；没有复核最多BETA。

| 等级 | 原验收门，全部条件均须成立 |
|---|---|
| VALIDATED | 每任务>=32/40；每任务>=14/20场景双bank成功；每task-family>=50%；每任务S1<=4；0真实native fault；80格全审计；训练复核及包齐全。 |
| BETA | 每任务>=28/40；每任务>=12/20场景双bank成功；每task-family>=50%；每任务S1<=8；原安全、80格审计和交付门保持。 |
| 未达标 | BEST_RESEARCH_CANDIDATE_NOT_QUALIFIED，保留完整科学负结果，不称合格policy。 |

交付共享权重、processor/stats、依赖锁、POLICY_MANIFEST/ACCEPTANCE/POLICY_CARD/REPRODUCE、默认离线推理/reset/close/fail-stop、训练lineage及完整结果/视频索引。固定底座ARX5成绩不等于Go2+ARX5冻结WBC或真机成绩。

## 5. 资源、时间与需要人类决策的边界

原总上限：96 H200小时、24 PBS、同时1卡、单job<=8h；CoreLab320/Hopper160/local48 CPU核时；最多16384优化更新、48000神经查询；其余物理/构造/存储/网络按原资源卡与站点更小值。已证7次提交不是8次：child编号与PBS计数不同，child05未qsub。余额必须现场刷新。

原final保护至少20 H200小时、100次episode启动预算、9000查询及原live/shadow/storage保留。科学分母仍80；100是包括工程替补的资源估计。旧case约729秒，100次任务本身约20.25h，未含资格/IO，故20h不能当足够上界。每次新job和优化阶段前用现有全站fresh预算器；优先舍弃可选分支，不能缩减最终分母或假设未经实测的加速。

| 原PV1边界 | 绝对SGT时间 |
|---|---|
| T0 | 2026-10-19 02:29:56 |
| 停新job/候选 | 2026-10-25 02:29:56 |
| 停新训练block | 2026-10-25 14:29:56 |
| 停新物理 | 2026-10-25 18:29:56 |
| 全部收尾 | 2026-10-26 02:29:56 |

换窗口、排队或重启不重置任何边界。若实时预算/排队证明完整B0前置使原目标不可完成，提交一份具体的协议修订建议和缺口：可讨论将B0完整从训练前依赖移至选模前依赖，保留全48比较。这是科学日程依赖而非物理安全门，但本handoff未批准修改，不能自行删guard。新增预算、延长窗口或删评价条件同样需明确决策。

## 6. 冻结项、维护和首次反馈

WBC代码/权重/gain/cost/solver、腿策略、plant、夹爪PD/力限、S1/参考安全门、任务评分、模型数值/动作语义和split保持现役版本。禁止第二GPU、第二调度器、强杀健康作业或盲重发未知动作。

checkpoint清理沿唯一既有维护通道：只删确定无效/废弃、无活动消费者、无恢复/对照/唯一审计引用的临时payload。保留父、A/B终点、当前候选、最新full resume和必要回滚、原数据与失败证据；exact清单、原生删除效果和实际释放字节必须留存。无对象就0删除，不在接手时全盘清理或迁移活跃目录。

新窗口首次应报告：
1. 管理SHA与运行SHA分别是什么；owner已重连、已安全交接或待交接。
2. 实时job/阶段/模型就绪状态、B0真实有效n/48、原attempt及unknown处理。
3. A/B实际已提交step、资源已用/预留、下一合法动作和唯一具体阻断。

建议在原报告根输出一个版本化 `PV1_WORK_WINDOW_HANDOVER_ACTUAL.json`；这是后续**实际接手回执**，本次未产生。只有现场排他控制和状态一致性成立才写TAKEOVER_CONFIRMED；CI、文档或阅读成功不替代它。

随后只报告真实里程碑：B0全48、A/B实际1024/4096、各DEV48、匹配结论、候选、最终80和包。沿原每4个活跃小时报告，不新建永久监控/cron或重复automation；已有监控只报告，不成为第二调度者。科学#11，工程/交接#47；普通提交、完整SHA和回读，不force/reset、不向Git上传私有路径/凭据/模型/WBC。

**发布者本次只审核GitHub并发布接手计划，没有SSH、转移owner、qsub、模型/训练/物理或删除操作。新窗口先核对现场，再沿原授权推进；没有现场连接时明确报告缺少的连接，不声称已接管。**
