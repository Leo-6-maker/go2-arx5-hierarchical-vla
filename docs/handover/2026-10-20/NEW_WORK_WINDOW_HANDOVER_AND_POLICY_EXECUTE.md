# 新 Work 窗口接力：接住现役 PV1，完成共享 T1/T2 policy

日期：2026-10-20，Asia/Singapore。文档 ID：PV1_NEW_WORK_WINDOW_HANDOVER_V1。
状态：HANDOFF_PUBLISHED_NOT_NATIVE_TAKEOVER。本文件是用户要求的新窗口接手指令，不是新实验、新预算、新七天窗口或已完成的 owner 交接。

## 1. 新窗口先读这里

你的目标是承接同一个 PV1 campaign 的真实现场状态，尽快完成「基线 → 匹配微调 → 闭环比较 → 合格则独立验收」，不是从聊天历史重新建立一套工作区或启动另一位调度 owner。

仓库：`Leo-6-maker/go2-arx5-hierarchical-vla`。
集成分支：`pi0267/executor-decoupled-recovery`，不能默认 main 是现役科研代码。
Campaign：`20261018T182956Z-localpc-pv1`。
原权威：`9eff29a44a193dd4d3620dc0bfe5764a402eff42`；原科学协议 `docs/protocols/PI0294_PV1_MULTITASK_POLICY_RELEASE_20261019.md`；原资源卡 `reports/PI0294/design/PV1_MULTITASK_POLICY_SCOPE_20261019.json`。
已采用的 LG04 为 `738a369c25dd24fff408bdedae0120f010623cfd`。本文件整合接手信息、更新过期快照，不改变这些科学边界。

阅读顺序：本文件和配套 `reports/PI0294/design/PV1_NEW_WORK_WINDOW_HANDOVER_STATE.json` → `AGENTS.md` → `START_HERE.md` 最新段落 → 原 PV1 协议和资源卡 → Issue #11/#47 最新实际回执。工作区按 `docs/operations/2026-10-13_UNIFIED_ENTRY_AND_GITHUB_SYNC_V1.md`、`docs/operations/WORKSPACE_ENTRY_ENFORCEMENT_V1.md`、`infrastructure/workspace/ENTRY_POLICY.json` 进入；CoreLab 人类入口仍为 `~/Projects/GR00T-WBC`。用现场已部署的 enter/doctor/registry，不猜命令、用户名、私有路径或 session ID。

## 2. 本次审核的已证事实与尚未证明的事实

审计管理版本：`3bd5d62fcbe96ee1448d9e9b7e0d5763cb6ff214`，2026-10-20 09:05:50 SGT。最后科学采用评论：#11/6280903573，09:06:22 SGT；工程评论：#47/6280903261。这些是已发布现场快照，不是你打开新窗口时的实时 PBS 状态。

| 项目 | 已核实状态 |
|---|---|
| 原 N1 | 训练与独立确认已关闭；CORR@2048 原确认 T1 10/20、T2 4/20。不能重新给旧确认打分或作为新隐藏测试。 |
| FD1/FD2 | 已结束。失败以任务后程指垫碰桌为主的直接几何证据已取得；不是全部归因于初始接近，也未证明 4mm Teacher 是唯一训练根因。不要重新做整轮归因。 |
| PV1 数据 | COVERAGE_FIRST，固定新银行 139 条合格示范：T1 75、T2 64；T2 +Y 48/-Y 16。83539 个 H40 窗口不是独立演示。原统计量与官方 codec 准入不变。 |
| B0 | **12/48 有效，8 成功、4 有效任务失败**；还有 36 个固定条件待完成。预模型工程故障不能计为 task fail；12 个有效条件不可重复。 |
| 新微调 | **A_CONTROL=0、B_COVERAGE=0 optimizer updates**；当前所有 B0 成绩仍是旧 CORR@2048 的表现。 |
| 现役 child | **policy-b0-08 / 677341.hopper-m-02**，第 7 次 PV1 PBS 提交，CFP04-CF-007，1 H200 / 12 CPU / 128GB / 8h。 |
| 当前运行代码 | **fb00da44f122c3ce96d0970ba23c2f223470cd33**，不是管理 HEAD。精确 Actions **39348602625** / job **118387138067**，push/completed/success 已核验。 |
| 现役原生状态 | 最新收据为 **R**，PBS_READY=true，同一 job 已 release 一次，owner 活跃，premodel qualification 正在进行；**尚无资格通过或新 model-ready/任务结果的回执**。 |
| 资源 | 已结算 **15398 H200 秒（约 4.277h）**，另有 **28860 秒预留**；已用 **7/24 PBS**；原累计查询 862、optimizer updates 0、unknown=[]。当前已分配作业的费用要现场刷新，不能把预留当已耗或再次计费。 |
| 策略选择 | selected/deployment=NONE。P5 的单候选/LOCKBOX80/离线包已有 CPU 合同实现，尚未打开 LOCKBOX。 |

本轮最直接的瓶颈是：大量时间与分配机会花在冷绑定、跨节点资格和通信恢复，新增数据的 A/B 学习收益仍未开始测量。这不是已证的 GR00T/模仿学习失败。

### 最近三份 child 的准确去向

- child06 / 676926：模型加载前，在隔离 concurrent_replace 夹具遇到 NFS ENOENT，已关闭；新模型/查询/任务/积分/训练均 0，保守计费 634 秒。修复只涉及隔离测试中有界读取恢复，未放宽生产安全门。
- child07 / 677332：模型加载前发现重新绑定的 sealed fixture 为 0644 而非要求的 0400，已关闭；新科研效果 0，计费 58 秒。它暴露的是最终绑定产物检查缺口，不是模型能力不足。
- child08 / 677341：沿同一修复 runtime 冷绑定，两端最终 sealed 输入 0400 已回读，Core 零积分模型/双摄资格通过。站点将 128GB 改成 225GB 后，原 owner 已一次原地纠正回 128GB；不要再修改该已 R 作业的资源。

固定证据：
- `reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_B0_06_PREMODEL_NFS_VISIBILITY_FAILURE_AND_REPAIR.json`
- 同目录 `PV1_B0_07_IMMUTABLE_BINDING_PREMODEL_FAILURE.json`
- 同目录 `PV1_B0_08_COLD_BIND_AND_NATIVE_RUNNING_RECEIPT.json`
- 数据银行：同目录 `PV1_FIRST_FROZEN_BANK_AND_HF_ACTUAL.json`；完整采集记录 `PV1_DATASET_CARD_AND_ALL_CAPTURE_ATTEMPTS.json`。
- 最终执行器：同目录 `PV1_P5_CPU_CONTRACT_IMPLEMENTATION_RECEIPT.json`，实现历史固定 `917fcda059324961a5c2f10b9f262d2c6711fd03`。

## 3. 新 Work 窗口的第一个动作是安全接力，不是 qsub

**不要因为换聊天窗口而关闭旧 owner、重启现有作业、伪造 owner 心跳或覆盖正在运行的 manifest。新窗口存在不等于第二个 dispatcher 获准。**

### 3.1 先建立只读接手快照

通过当前真正可用的连接和原有入口，核对：现役 owner 的 host、PID 与启动时间、可重连 session；PBS Q/R/F 与实际 job ID、request/nonce/project；两端 run/lease、源版本、当前阶段、最后成功 heartbeat；正在执行的 condition/query/install/ACK；已提交训练 step/cursor（目前预期 0）；各站 ledger revision、已用/预留、pending/unknown；管理 checkout 的 dirty/未发布改动。私有细节仅留原注册工作区，Git 只存脱敏摘要。

不能用旧评论的 Q/R、某个锁文件不存在、SSH断开或聊天关闭，证明旧 owner 已死亡。也不能把还没发布到 Git 的有效 native 结果当成不存在。

### 3.2 只允许下列三种接力方式

1. **原 owner/session 仍健康且可重连：**复用该现有会话与同一个调度进程，新窗口作为用户交互入口；不创建新的 owner heartbeat writer、lease、qsub循环或第二publisher。若连接不支持重连，先只读工作，不盲目启动替代进程。
2. **确需转移控制：**使用已部署、已验证的单owner交接/恢复流程，由旧owner明确交出控制、持久化最后状态，由新窗口核实原生状态后接受；原子性/排他性必须来自真实既有工具而不是文档声称。若现场没有安全的在途控制转移能力，保持旧owner完成当前安全边界后再交接；不要取消健康PBS来强制交接。用户切换窗口不授权热改签名中的owner字段。
3. **旧owner已死或状态不明：**先核实双方worker/job、最后已提交副作用和原watchdog状态；遵守原停止/恢复路径，对未知query/install/qsub先journal与原生效果核清。没有核清前不续发动作、不删锁、不重投、不恢复失效lease。安全关账后只对真正未完成条件建立合法冷续接。

交接成功的证据不是“阅读了文档”，而是同一时刻只有一个有资格的调度者、原作业/心跳未被打断、输入与账本一致且不会重复提交。新窗口首次反馈应使用 `ATTACHED_EXISTING_OWNER`、`TAKEOVER_PENDING_SAFE_BOUNDARY`、`TAKEOVER_CONFIRMED` 或 `BLOCKED_EXACT_ACCESS_OR_STATE`，并给真实 job/阶段/B0有效数/unknown/下一动作。

**在上述确认前，用户不应主动关闭承载现役owner的旧会话。** 没有现场连接时，报告缺少哪一个连接或session，不谎称接管完成。

## 4. 接手后的唯一关键路径

### P0：保护 child08，完成 B0 余下条件

若现场仍为原 job R，保持它的 runtime/capsule，不重复 release、不重复本allocation已完成的资格；仍待资格时按原日志判定结果，不把R当MODEL_READY。资格失败保持工程失败，不算 policy fail；若已产生新有效cell，立即合并原生证据并更新剩余清单。

使用现有 `remaining_panel` 和完整原尝试记录，正常基线余36格（T1/T2各18，其中 F1/004/T1/N294101 只有已审定的工程替补资格），但以新原生证据重新计算。不得把36格再加1次重复replacement。所有有效失败、参考拒绝、S1拒绝都保留。

一次8h分配未必容纳全部余量。按原closeout门保存已完成合法条件，记录未运行格、关账并核清全部效果；后继job只执行未完成的固定条件。不是重新做48格，不复活已CLOSED child06/07。

### P1：阻止下一次可在 CPU 发现的 premodel 失败

不干扰 child08。针对**未来冷绑定**，在最后一次 binder 写入之后、任何下一次 qsub 之前，使用现有校验器检查实际最终文件而非源码模板：0400权限、所有者、非symlink、manifest列举/哈希、完整schema、fixture与request引用、nonce、quota稳定读取、project权限、有效lease、准确输出根、剩余ledger和未知效果。再检查prepare/submit阶段没有修改已封存成员。不能靠GPU allocation来发现0644等静态错误。

需要同语义修复则先最小复现、针对性CPU测试、普通提交和精确source的CI成功、冷态部署；不大规模重构验证框架、不删生产安全门、不热改活跃capsule。不重复已通过的整套测试来替代推进。新增allocation仍按原600s/80cycle/all-fault真实资格；不得取消该要求或每换checkpoint重复一遍。

CI取证要查精确 `head_sha` 的 Actions run 和 check-run。部分专用连接器只列PR触发运行；direct push可用GitHub GET `/actions/runs?head_sha=<full_sha>`，再读 `/actions/runs/<id>`。legacy commit status 的 pending + 空statuses不证明Actions失败。当前runtime的成功run已在第2节，不需要为了本次纯文档handoff重新部署runtime或等待文档commit成为新科学source。

### P2：B0完整后立即进入原封存 A/B 微调

现役协议明确要求 `PASS_COMPLETE_ACTUAL_B0_DEV48_EVIDENCE` 在新优化器更新前成立。这是**完整性门，不是B0成功率门**；新窗口不能用部分结果调整配方，也不能把旧12格表现低当成不训练理由。

GPU运行期间，只做 A/B 的CPU冷态准备、既有结果增量审计、全流程资源预测。不要再采新银行、重跑FD1/FD2或建设第二套P5。

固定父：CORR@2048 manifest `94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38`。
统计量：`fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7`。
新银行manifest：`60f7c35a05766fe23974eba3a72f53ca4db5545a3768e6c1cfea164ab4e58294`；READY revision `e4f603a1b768018077a8b5c9640250be67500902`。

| 每个effective batch32 | A_CONTROL | B_COVERAGE |
|---|---:|---:|
| 原T1 | 8 | 4 |
| 新T1 | 0 | 4 |
| 原C1 T1纠正 | 8 | 8 |
| 原T2 | 16 | 8 |
| 新T2 | 0 | 8 |

两臂从同一个父权重、独立同配方的新optimizer开始；各4096正式更新，microbatch2/accum16、seed20261019、LR1e-5→1e-6/warmup64、原flow-matching、VLM冻结与原trainable map不变。每臂1024保存点做预注册TRAIN探针后，完整恢复optimizer/scheduler/RNG/sampler/processor/stats到4096；不是从1024重新开始一份新训练。不得使B继承A训练的权重。

1024仅是TRAIN诊断，不用来选DEV候选。各4096终点分别评价同一DEV48。银行/checkpoint冷复用必须有完整hash与实际消费证明；性能优化只采用两臂相同且通过原CUDA数值/梯度资格的实现。新窗口不是修改batch、normalizer、loss、H40/Prefix8或采样比的机会。

### P3：先取得匹配科学结论，再决定额外训练

发布 B0、A、B 的各自完整48格及所有实际attempt；按task×scene×bank对齐，报告B-A和各臂-B0的成功差异、最弱任务、参考门拒绝、S1几何拒绝、32s完成但目标失败、工程缺失、实际GPU/CPU费用。按scene group而非两次noise独立样本处理不确定性。由于同时改变T1和T2数据，解释为联合覆盖干预，不宣称单独T2数据的因果效应。

可选P3双臂补强默认不启动。只有原任务特定恢复/事件证据门以及全预算成立时才做；旧T2 Recovery Oracle 0/2不是合格标签。WorldModel/扩大DAgger/新codec/历史输入等留在已登记长期战略，不成为当前前置。

### P4：唯一候选与第一版策略包

原DEV门：每任务>=18/24，任一任务相对B0退化不超过1/24，最弱任务严格净改善，0原生安全故障。按worst-task→macro→S1→复杂度→早ID选唯一候选。

资格与完整未来资源同时满足后，才打开LOCKBOX80（20新场景×2任务×2噪声），不回流训练、不筛场景、不挑成功noise或更换第二模型。第二seed复核只有能保护最终验收资源时执行；未复核最多BETA。

- VALIDATED：每任务>=32/40、>=14/20场景双bank均成功、每task-family>=50%、每任务S1<=4、0真实native fault、80格全审计、训练复核和包齐全。
- BETA：每任务>=28/40、>=12/20场景双bank成功、每task-family>=50%、每任务S1<=8，原安全/完整审计/交付门保持；不是用更少测试换来的等级。
- 未达门：交 `BEST_RESEARCH_CANDIDATE_NOT_QUALIFIED` 与完整负结果，不把最好的checkpoint叫合格policy。

交付共享权重、processor/stats、依赖锁、POLICY_MANIFEST/ACCEPTANCE/POLICY_CARD/REPRODUCE、默认离线推理/reset/close/fail-stop、训练lineage和完整结果/视频索引。固定底座ARX5的源域成功不等于Go2+ARX5冻结WBC或真机成功。

## 5. 资源和时间：换窗口不重置

原总上限：96 H200小时、24 PBS、同时1 H200、单job<=8h；CoreLab320/Hopper160/local48 CPU核时；最多16384优化更新、48000神经查询，物理/构造/存储/网络按原资源卡。最新已证7个PBS并非8个：child编号与真实提交计数不同，child05未qsub。剩余次数先按真实账本刷新。

原最终保护至少20 H200小时、100次episode启动预算、9000查询及原live/shadow/storage保留。科学分母仍80；100是允许的最终工程尝试预算，不是100个独立条件。已有旧case成本约729秒/条件，100次仅任务时间就约20.25h，尚不含资格和IO，因此20h不能当作足额上界。每次新job/优化阶段前用现有fresh全站预算器判断，优先取消可选支线，不能削减最终分母或假设未验证的加速。

绝对截止：
- T0：2026-10-19 02:29:56 SGT。
- 停新job/候选：2026-10-25 02:29:56 SGT。
- 停新训练block：2026-10-25 14:29:56 SGT。
- 停新物理：2026-10-25 18:29:56 SGT。
- 全部收尾：2026-10-26 02:29:56 SGT。

若实时预算/排队证明完整B0前置会使原目标无法完成，向用户提交**一份具体的协议修订建议**及缺口：可讨论将B0完整从训练前依赖移至选模前依赖，仍保留全48比较；这不是物理安全门。然而本handoff未批准该修改，不可自行绕过现役guard。新增GPU、延长窗口或删评价条件同样需要明确决策，不由新窗口自动赋权。

## 6. 冻结项与维护

WBC代码/权重/gain/cost/solver、腿策略、plant、夹爪PD/力限、S1/参考安全门、任务评分、模型数值/动作语义、数据split均保持现役版本。禁止第二GPU掩盖串行IO等待、强杀健康作业、假冒旧owner心跳或盲重发未知动作。

checkpoint清理沿唯一既有维护通道：只删确定无效/废弃、无活动消费者、无恢复/对照/唯一审计引用的临时payload。保留父模型、A/B终点、当前候选、最新完整resume与必要回滚、原数据和失败证据。先列exact文件/理由/占用，后按原授权删除并报告实际释放字节。有效负例不是垃圾；没有对象就0删除。不在接手时做全盘清理或搬移活跃目录。

## 7. 给用户的首份接手报告与后续输出

先交一个短报告：管理SHA/运行SHA分别是什么；原owner是已重连、已安全转交还是待交接；实时job/阶段/模型就绪状态；B0真实有效n/48及原attempt处理；真实训练step；资源已耗/预留/unknown；下一合法动作。私有完整checkpoint/session/路径清单放原注册工作区，公开仅哈希和脱敏索引。

建议输出 `PV1_WORK_WINDOW_HANDOVER_ACTUAL.json`（若已有则新版本，不覆盖），它记录接管事实而非自动创造新锁/租约。只有现场排他控制与状态一致性确认后可写TAKEOVER_CONFIRMED。计划发布、CI通过、阅读文件不能替代它。

之后只报告实质里程碑：B0全48、A/B实际1024/4096更新、两份DEV48、匹配结论、合格候选、最终80和包；原每4个活跃小时进度制度沿用，不为换窗口新建永久监控/cron或第二automation。已有监控只能报告与提醒，不应成为第二调度者。

科学索引 #11，工程/交接索引 #47。普通提交、固定完整SHA、回读验证，不force/reset、不把私有数据/模型/WBC写入Git。

**本文件由GitHub证据审计生成；发布者没有SSH、操作owner、qsub、训练、物理、删除或完成真实接管。新窗口必须先核对现场，再沿上述原授权推进。**
