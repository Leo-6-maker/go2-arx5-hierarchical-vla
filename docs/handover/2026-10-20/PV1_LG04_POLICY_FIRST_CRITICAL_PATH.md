# PV1-LG04｜Policy-first：H200受限下尽快完成共享T1/T2策略的科学闭环

- 日期：2026-10-20，Asia/Singapore
- Goal ID：`PV1_LG04_POLICY_FIRST_CRITICAL_PATH_V1`
- 状态：**USER_REQUESTED_PRIORITY_HANDOFF_ONLY / REMOTE_OWNER_ADOPTION_NOT_YET_EVIDENCED / NO_NEW_SCIENTIFIC_OR_RESOURCE_GO**
- Git审核基线：`85fa1d8d15eae05978c1076d33cf7159b3b69329`，现役分支`pi0267/executor-decoupled-recovery`。最新可见原生收据是Issue #47 `626?`之外的**`6272938836`**（2026-10-20 01:16 SGT）；科学#11`6270506505`、`6272012619`和`6272769694`。本文件**不是**2026-10-20 05:13 SGT的直接PBS查询。
- 权威仍为原PV1已采用`9eff29a44a193dd4d3620dc0bfe5764a402eff42`，原协议`docs/protocols/PI0294_PV1_MULTITASK_POLICY_RELEASE_20261019.md`及机器预算`reports/PI0294/design/PV1_MULTITASK_POLICY_SCOPE_20261019.json`。已采用LG03`42faa3f7bce84877b1fd51c3c31de1d8686a15df`不重做；本Goal仅调整**优先级、剩余作业机会与科学交付证据**，不重写原协议或任何旧结果。
- 唯一执行角色：**现有single local-PC owner**，既有可跨聊天中断的合法本地会话、统一workspace entry、原NFS单mutation owner和集成publisher。用户此前停止的子代理不恢复；没有第二调度器、第二owner或后台新任务。

## 0. 科学目标和最早可接受的完成定义

**第一目标：尽快获得一次可信的新增示范覆盖对照**——同一个旧CORR@2048模型，完整新B0 DEV48；分别从同父同配方训练A_CONTROL/B_COVERAGE各4096真实更新；两份终点各完成相同的DEV48自主物体任务；封存`PV1_MATCHED_AB_SCIENCE_REVIEW`，把两任务success/S1/参考合法性/释放推送终末失败、scene-noise配对、GPU时间与真实可追溯训练样本全部报告。**两臂训练＋两臂DEV是真正的科学闭环，不是训练loss、工程CI或单条成功视频。**

**第二目标：尽快交付一份能运行的共享T1/T2研究policy**。仅在原DEV资格合格、唯一候选按规则冻结且原生剩余资源足够时，执行唯一LOCKBOX80并出`POLICY_MANIFEST`、`ACCEPTANCE`、`POLICY_CARD`、`REPRODUCE`、offline safe entry和唯一原生模型/统计量/数据/评分/视频索引。若VALIDATED/BETA门未过，仍交真实research candidate和完整负结果，但`selected/deployment`不能谎报为成功。

**不得用额外P3、World Model、新DAgger、新codec/历史输入/动作头/horizon修改、WBC归因项目，取代上述两个目标。**

## 1. 本次审核截至最新原生收据的真实状态（允许被新现场结果更新）

| 维度 | 最新来源支持的事实 | 不得误称 |
|---|---|---|
| B0源域新DEV | **12/48有效**，其中8 SUCCESS、4 VALID_TASK_FAIL；T1/T2各6格有效，另有保留的工程中断attempt | 不得把12格外推48格性能 |
| 新权重 | **0 optimizer updates**；CORR@2048 B0仍只是旧权重在新DEV下的基线 | 不能宣称本轮微调已经改善 |
| 数据 | 原封存TRAIN139合格新Teacher轨迹，77场景组、62双任务配对；`COVERAGE_FIRST`路径，原v3.0 codec/stats保留 | 83539重叠窗口不是83539独立示范 |
| 作业 | **child06=`676926.hopper-m-02`**，第5次PBS、项目`CFP04-CF-007`、`B0_DEV`，一个qsub；原脚本128GB被native site改为225GB，原owner已一次in-place qalter回128GB；后续原生Q阻断改为`ngpus` | GitHub最后Q快照不等于当前还Q，更不是已获卡 |
| 新子代source | `85fa1d8d15eae05978c1076d33cf7159b3b69329`，精确push [Actions39286536601](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/actions/runs/39286536601) completed/success；原relay修复在祖先`be900641` | Actions绿不等于原生600s/80cycles资格、模型加载 |
| 旧child05 | quota报告刷新竞态，在prepare/qsub前两端封闭，**新增PBS/GPU/查询/物理/更新为0**；不是可复活活动job | 不得沿用已CLOSED child05租约或覆盖失败证据 |
| P5执行器 | `917fcda`已实现P5_FINAL单候选/80分母/策略包，CPU合同与[CI39260662293](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/actions/runs/39260662293)PASS；真实LOCKBOX仍未打开 | CPU测试不是模型验收 |
| 费用 | 原四份终态PBS保守`14706 H200秒`约4.085小时；第5份队列8h为**预留而非已耗**，更新由原生账本刷新；全部上限96H200h/24PBS/同时1卡 | 不能把排队reservation当已计费，亦不得遗漏失败计费 |
| 选择与部署 | `selected/deployment=NONE`，无Go2+ARX5/WBC整机或真机结论 | 不得称Policy V1已validated |

真实科学阻断目前是完整B0 DEV48的36格，包含对`DEV/F1/004/T1/N294101`已审定的**唯一工程替补**；已有12格有效结果和旧错误原件不得重跑或抹除。此刻以原生最新job/lease/ledger/证据为准，不从本文的Q快照发第二个PBS。

## 2. 当前关键路径与时间风险（按事实制定顺序）

### 2.1 唯一GPU串行主线

`child06 native Q→R及本作业资格 → B0余下36格（必要时保存partial、下个合法分配仅续未做） → PASS_COMPLETE_ACTUAL_B0_DEV48_EVIDENCE → A_CONTROL训练0→1024 TRAIN探针→4096 → B_COVERAGE同样训练 → 两臂各自DEV48 → 匹配科学结论 → 原DEV选模 → 唯一P5 LOCKBOX80（若真正合格） → 共享模型包`。

- 两臂都从同一`CORR@2048`父开启新optimizer，不能让B从A训练的权重继续；全套effective batch32（micro2×acc16）、4096 updates/臂、原相同模型/loss/stats/inference/noise，1024探针仅TRAIN，不提前在DEV选中间checkpoint。
- **现役B0全48在正式A/B训练之前的准入硬门保持有效。** 若排队使它明显妨碍科学闭环，只能报告需要**单独人类科学修订GO**的`B0_COMPLETE_BEFORE_SELECTION_NOT_BEFORE_TRAINING`替代方案；本次hand-off不授权改代码、取消门或看部分DEV结果修改A/B。
- A/B DEV的每一task×scene×noise条件要同一冻结population，无重抽、无valid负例重试，按scenario-group聚类报告；避免把新T1和新T2联合替换误称T2单独因果收益。
- P3额外双臂2048补强**默认不启动**，只有原协议明确的TRAIN学生/真实事件拟合支持及完整fresh资金预算仍可容纳、且直接通往policy更有价值时才考虑。WorldModel、延伸DAgger、更多教师、不必要的重复诊断全部退出本次临界路径。
- 第二训练seed在不影响原P5完整资金与时钟时才做；跳过则不论最终成绩如何**最多BETA**，不能伪造已复核。

### 2.2 真实预算敏感性，不把估计当native准入

已有7条原B0完整case的平均`728.85s`、最大`890.88s`（两者都是小样本观察，不是保证上界），36剩余case在平均值下约`7.29h`，未含新的原生600秒资格、银行hash、模型加载/IO、原900秒stop及closeout；**单次8小时申请不可假设能一次跑完36格**，因此将partial conservation、仅重续真正NOT_RUN当作正常计划，不是工程失败。

训练两个4096端点使用较慢实测`4.6509s/update`并加25%计划余量，约`13.23h`净训练预算，但两臂都要原预注册1024探针/全状态保存与冷态续接，不能把端点总净训练时间假设为两次PBS就够。两个DEV48约`19.44h`按同样的旧case平均值，仅作预算敏感性。

P5的科学分母是**80个有效条件**，原资源预报器还为最终工程替补/收尾按**最多100次实际启动**计画：100×728.85s≈20.25h，且未计真实原生qualification/IO。故原受保护`20 H200h`是**最低保留量，不是已证明足够覆盖整个P5的上界**；可以在原96h内从尚未动用的可选项目转入final，但不能悄悄减80条件或假设一定跑得完。以后预算决策以`scripts/pi0294/pv1/future_budget.py`、所有site fresh snapshot、真实case及training upper和最终受保护账户的**全面PASS**为准，不以此表的均值代替批准。

当前原PBS第5份Q若较长，**日历时间比纯96H200小时更可能成为阻断**。完整计入每个必要下一冷态阶段的queue lead time、<=8h job、资格、closeout、1卡约束、24总提交、站点project余额与原绝对deadline。不得靠更多qsub消除排队，否则只会引入重复/超额与未知副作用。

### 2.3 实际停止时钟，不允许隐形延期

原T0：2026-10-19 02:29:56 SGT；
- 2026-10-25 02:29:56 SGT 起不得开新job/新候选；
- 2026-10-25 14:29:56 SGT 起不得开新训练block；
- 2026-10-25 18:29:56 SGT 起不得开新物理；
- 2026-10-26 02:29:56 SGT前全部证据/资源/租约归档。
这不是新的七天；排队、chat重开或本LG04不推迟任何一天。

## 3. 在child06排队或运行时的唯一可并行CPU工作

由现有唯一owner操作，轻量、有限、可以被抢占，不阻塞心跳/SSH/RPC：
1. **C0健康与账本（最高）**：只读刷新原`676926`真实job状态、是否Q/R/F、request/nonce/lease/STOP、quota/project、current ledger、原生prefix/ack/journal；已经完成的result先判安全/因果/工程身份。对原submit已有一次成功的job不再次qsub、qalter、不重开child05；当前若已R则**只允许当前job自己的正常运行**。
2. **C1 B0剩余路由与证据**：用原真实12个有效条件与一份工程replacement资格，增量生成`remaining_panel`固定36格；每个结果原adjudication/raw/hash/video/独立two-site chain可进入原汇总，无效工程attempt保留；不预览LOCKBOX，不重评成功或有效失败，不为已有原数组做第二轮FD1/FD2。
3. **C2 A/B的冷态ready而非GPU启动**：复用已实现的`bank_reuse`、`checkpoint_reuse`、`performance`、`future_admission`，CPU-only确认两个合法相互独立的prepare/dry-run的source父权重、数据manifest、statistics、sampler lane/parent/window游标、1024→4096全状态恢复/finited gradient判定、每臂输出根及严格0重启效果。输出`A_AND_B_COLD_READY_NOT_SUBMITTED`；**不在未得B0全48前写出会使神经任务启动的真实PBS intent**。
4. **C3 最小CPU P5接线**：P5已在`917fcda`完成CPU contract/CI；仅检查从A/B DEV最终胜者冻结后，能否调用现有`p5.py`及`offline_policy.py`在CPU测试fixture下生成准确包。**不重写P5，不读取任何LOCKBOX真实观测或执行，实测native P5资格留到合法时。**
5. **C4预算分叉**：实时用原有资源预报器做`required path`与`replica omitted resource-critical`两种独立方案；P3只列`OPTIONAL_NOT_RESERVED`。对必需工作若无法全部保留充分计算/物理/CPU/查询/提交/存储及final allowance，则标`BLOCKED_BY_EXACT_RESOURCE`和缺口，不默默剪掉DEV/LOCKBOX或偷偷延长停止点。向人类报告可行选择。

完成原C0-C4后，**停止制造更多准备代码，等待/接续真实作业**；工程准备不替代新的模型更新。旧数据银行无需重新采集，旧模型不用从基础GR00T重训。实际CPU/Hopper/Core/local源配额按原PV1执行许可计算。

## 4. 每个阶段的具体输出与科学门

### G0：B0全48且资源收口
- 第五job自己的600s/80cycles/all-fault传输资格、CUDA/model CORR@2048、原139银行全hash/官方processor及新增每格原审计PASS才算。保留R/F及未知分支，绝不根据队列Q推断生效。
- 完整48个预注册格，合法失败保留，工程中断只能在确证+限定次数下替补，历史重复不计独立样本。闭合所有peer leases/PBS/H200花费和unknown effects，发布`PASS_COMPLETE_ACTUAL_B0_DEV48_EVIDENCE`，**不按B0成功率额外挡训练**。

### G1：正式双臂微调（两端点全部）
- 固定`CORR@2048` manifest `94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38`，stats `fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7`，HF新银行READY revision `e4f603a1b768018077a8b5c9640250be67500902`。
- A batch32旧T1 8/C1_T1 8/旧T2 16；B batch32旧T1 4/新T1 4/C1_T1 8/旧T2 8/新T2 8，T1:T2均1:1。每臂4096真实更新，训练步准确计费，原8/40动作/参考合法性与安全不改。完整保存optim/scheduler/RNG/sampler、源数据成员、冻结视觉参数与trainable梯度，安全通过后才允许依据元数据冷续接。
- 两臂第1024步只执行预注册TRAIN探针。当前CUDA性能计划若用于下一合法训练需同臂严格验证并在A/B同一配方执行；新方案尚未实测的加速不抵扣预算。

### G2：两臂共同DEV48并封存对照结论
- 同两个任务、12独立场景组、2噪声，共48cell/arm；不能只跑B或先看B结果重改A配方。完成旧B0、A和B各自原始结果与独立审计；按原预注册最弱任务→macro→S1→复杂度→ID排序。
- 输出一份`PV1_MATCHED_AB_SCIENCE_REVIEW.md`+机读result：B比A、各臂比B0的逐任务完整成功、scene成组区间、F0/F1/F2、正反方向、完整H40参考拒绝、S1 pad/table、完成但物体任务失败、原工程缺失，以及GPUhrs/所消耗的真正独立parent。
- 当两个臂完整DEV结果不能证明有效改善，也要发布科学负结论和完整共享research candidate，**不自动伸出P3、WorldModel、DAgger或把同一DEV当新TRAIN**。

### G3：最短路径得到唯一共享策略
- 原DEV门：每任务>=18/24，原B0每任务退化不超1/24且最弱任务严格净改善，native fault0；单次checkpoint同时服务T1/T2，不能混合单任务adapter或偷换decoder。满足才按固定身份冻结唯一候选。
- 只有fresh预算、单卡以及时钟全部PASS才打开80格LOCKBOX（20独立场景×两任务×两噪声）；两bank不能当40独立scene；原S1/score不变。
- `PV1_VALIDATED`:每任务>=32/40、>=14/20场景双bank成功、每task-family>=50%、S1<=4/task、0原生fault、80审计、复核seed与完整包。若跳过复核最多`PV1_BETA`（每任务>=28/40、>=12/20配对场景、S1<=8及其余原门）；未达即`BEST_RESEARCH_CANDIDATE_NOT_QUALIFIED`。
- 交付唯一共享权重/processor/stats/epoch/code+数据身份、依赖锁、离线单次推理/reset/close/fail-stop入口、`POLICY_MANIFEST.json`/`ACCEPTANCE.json`/`POLICY_CARD.md`/`REPRODUCE.md`、全条件表/音视频hash/科研失败与资源账。默认不连WBC/真机；本轮源域固定ARX5不叫Go2+ARX5成功。

## 5. 仅有两条真正需要上报人类的未来决策

**D1原生资源/安全硬阻断**：如果`child06`及以后合法分配无法在原时钟、site/quota、24PBS/96h及final保留内完成至少`B0 + matchedA/B + DEV`，提交**当场精确账本与不可行性证明**，不因焦虑另开H200或让CPU改安全。必须时发布真实阶段最好的研究候选、源trace和`CLOSED_RESOURCE_BLOCKED`。若需要新时间/预算，它只能是新的**明确人类授权**而不是本Goal自身赋权。

**D2科学依赖修订的备选提案（当前未批准、不得实施）**：当且仅当新鲜排队+剩余时间预测证明`B0全48在A/B训练之前`是完成policy的关键障碍，而且A/B训练数据/父模型/seed/配方已提前封存、单卡资源可支付，可向人类提交版本化`B0_COMPLETE_BEFORE_SELECTION_NOT_BEFORE_TRAINING`修订以降低串行依赖。**仍要求B0全48在任何选模/公开综合成绩/打开LOCKBOX之前完成**，绝不改分母、S1/score、选模门槛或总GPU预算。明确审批前原`B0_COMPLETE_BEFORE_TRAIN`硬门保持原样，既有stage source不能热改。

其余常规同权威工程修复、已诊断relay/quota/site排队、合法新allocation的冷态续接无需重复逐次请示，但必须固定SHA/CI/owner/nonce/output/lease、原生提交及读取回执、未知query/install/qsub效果先journal对账。

## 6. 发布与验收要求

- **发布新Goal不是新一轮GPU GO。** 本文件及配套机读卡只在GitHub登记优先级，无SSH、qsub、模型加载、训练、物理、删除、配额或WBC更改；不能声称child06已出Q或策略改善。
- 仅原有唯一local-PC owner在刷新`AGENTS.md`、`START_HERE.md`、#11/#47、真实Hopper/Core+local/站点权限和原manifest/lease后采用，发一条`POLICY_FIRST_LG04_ADOPTED`或`EXACT_BLOCKER`；不得创建第二owner、重启已关闭child05或改已排队child06 source。
- 给owner的**单一当前工作单**是：先确定原676926的最新Q/R/F及真实资格与已完成cell；未开跑就继续监控**该作业**，已经开跑则保住原本代码与S1直至安全终局；并行CPU只完成C1-C4中尚缺的短工作。下一份科学收据要体现真实`B0_valid_n/48`和新模型有效更新，而不是仅报告又新增了几百个tests/文件。
- #11只报告真实科学任务完成/失败与A/B结果，#47只报告必要工程、认证与原生账本；单个实质里程碑合并披露，避免以大量readme和新helper刷屏。每个提交普通快进、固定SHA、逐文件blob读回；保持现役源胶囊、WBC、plant、夹爪PD、腿、S1、物理评分和已开放/未开放split不可变。任何未满足安全、source或站点门的情况准确报`PENDING`或`NOT_RUN`，不填PASS。

**核心判据：第一版policy必须来自同一GR00T权重对T1/T2真实闭环改善与独立验收，而不是来自未核验的额外模型、重复示范、工程PASS数或降低的科学分母。**
