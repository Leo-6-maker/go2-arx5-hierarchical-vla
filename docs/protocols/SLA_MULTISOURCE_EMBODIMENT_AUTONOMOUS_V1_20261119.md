# SLA 多源后训练与 Go2+ARX5 本体适配：长程自主推进 V1

> 2026-11-20更新：接续先读[实验审核与计划调整](SLA_LONGRUN_ADJUSTMENT_20261120.md)及其[机读清单](SLA_LONGRUN_ADJUSTMENT_20261120.json)。本页11月19日状态是历史截面；新调整明确MIX异常恢复、D2双站读取和2001–5000编排、部署兼容V2与学生native优先级。原预算和现役训练配方保持。

日期：2026-11-19，Asia/Singapore。状态：研究执行计划；本次发布没有提交服务器作业或产生训练、生成、原生积分。

- [后台排队接续与并行 Eval（最新用户调度决定）](SLA_SCHEDULER_FIRST_ASYNC_EVAL_V1_20261119.md)
- [机读任务与预算](SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1_20261119.json)
- [Codex 启动 prompt](../handover/2026-11-19/START_SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1.md)
- [证据、算术与发布核验](../../reports/WM02/sla-autonomous-longrun-plan-20261119/REVIEW_AND_VALIDATION.md)

## 1. 决策与最终交付

用户要求制定详细长程自主计划、充分利用 Corelab/VANDA A40 资源，并推送 GitHub。本计划采用四条并行工作线：

1. **完成现役 SCALE-MIX20K 与 SCALE-TGT5K 连续优化**，取得可恢复检查点、源保留面板和真实目标闭环曲线。
2. **立即扩充目标本体数据**：逐条准入，第一批争取24–40条，中期目标新增120条；Teacher全矩阵、D5成功和J全48完成均不是启动前置。
3. **第一批新增目标数据可用后启动一个 ADAPT5K 候选**，目标batch64、RoboDojo32、LIBERO32，预登记一次可选的数据课程切换；当前MIX继续。
4. **利用空闲/实际获配资源提高吞吐并验证双world**；已有单卡、单world科学路线持续工作，工程资格不成为整个项目的新前置。

交付是完整可恢复的模型与可运行的目标推理包、独立示范及覆盖清单、固定条件的闭环结果和失败归因。代码检查数量、PR数量、排队状态不算能力交付。低成功本身不停止健康的既定长训练；到正式端点依据数据、拟合和闭环证据选择下个有界版本。

本计划是既有合同的上层调度和明确后继。现役PR418仍使用[92622691固定SCALE合同][SCALE_CONTRACT]，PR417仍使用[cf9d1acb固定D4/D5/J合同][D416_CONTRACT]。本次研究数量是有限科学端点和可审计记账口径，不是每次小修再向用户请求批准的流程。常规项目用户态修复、数据准备与已有资源范围内接续由执行者自主完成。

## 2. 审计基线与当前问题

固定读取PR418 `93b3165f785a4c14a27c432f18b454c52571178a`、PR417 `afaf35970e56a0827afb092654ce6731f4723fe2`；计划母分支为PR416 `cc96f1709170e47298463337d050c79a24b7c7d9`。发布前再核对PR418 `39d8f6e341d2c974aa2baed31588b115e0c2c463` 与PR417 `4c9ccb9cc1db2dd5187152259762739bdbd98ee9`，以下进度表已采用补读结果。这些是GitHub已发布快照；执行时必须刷新现场，不将提交时间当作训练观察时间。

| 事项 | 已核验事实 | 下一动作 |
|---|---|---|
| Core MIX | 22:36:12 SGT观察到452更新；完整step200已封存；after500守候器active但后继训练尚未启动 | 核实当前端点和owner，沿同一谱系接续500/2k/5k/10k/20k |
| VANDA TGT | 22:36:12 SGT为898更新；完整500已封存，PBS1593003已从500恢复 | 保持同一前5k目标流与原LR，完成2k/5k |
| TGT500离线 | 288正式预测、48独立恢复均完成；48/48输出一致；A40墙钟5m58s | MIX500使用同一面板，择一站评价一次 |
| 目标数据 | 20 HOLD+3 PLACE、1768窗口；3PLACE仅F1蓝左/红右 | 补六个缺失组合、接近/闭合/释放/恢复 |
| 源扩充 | RoboDojo TRAIN200、210398窗口已到Core，304输入指纹一致 | 仅供新版本；不替换现役R48 |
| 新本体采集 | 首6条件预检和独立守候已就绪，仍0/6请求、0新合格示范 | 当前有限world段结束后优先采集 |
| D5 GRASP24 | 24/24完成，抓持/抬升/完整HOLD均0；2406 H40仅2个后缀闭合，P10闭合0 | 保留终态；不重跑选成功、不启动原CONFIRM32 |
| J | 最新10/48、417689tick、1446 H40，抓持/抬升/完整任务0/10 | 当前已派发段正常收尾；未派发尾段在新调度中明确排队 |

证据：[Core守候][CORE_WATCH]、[最新两站账本与microbatch就绪][LATEST_TRAIN]、[TGT500][TGT500]、[目标身份][TARGET]、[R200][R200]、[六条件][FIRST6]、[D5终态][D5_FINAL]、[J十条][J10]。

D416查询漏斗按当时D4 28+D5 24+J7共59条审计，6977份H40只有2个后缀阈值跨越，69557个实际消费原子未达到22mm。J七条的925份P10最低名义开度仍为41.952mm。因此优先问题是目标接近与动作选择、接触阶段覆盖、关闭时序；不能把无抓持主要解释为执行门拒绝。[漏斗证据][FUNNEL]

TGT500的6个H40首拒绝为5个线加速度、1个夹爪速度，另3个接缝拒绝；D5的全部H40首拒绝为夹爪速度。每个模型分别归因，不能将旧首因套给新SCALE。TGT500固定96仅1个窗口内闭合正例且漏闭，不能沿用D5的20正例分母。[TGT500][TGT500]、[D5联合诊断][D5_JOINT]

## 3. 两站资源调度

### 3.1 实际资源与默认分工

| 资源 | 已发布身份/容量 | 主要任务 |
|---|---|---|
| Core GPU0 | 站点报告RTX4080SUPER，total32760MiB；MIX已实际运行 | 唯一现役MIX训练owner，持续至20k |
| Core GPU1 | 站点报告RTX4080SUPER，total16376MiB | native推理、目标包资格；与world任务配合 |
| Core CPU/world | 目前只有一个正式可用world | 目标示范采集、固定学生闭环、有限并行工程 |
| VANDA已获A40 | TGT单A40训练；实际GPU容量/空闲须重读 | 连续TGT5k；有限allocation间恢复 |
| VANDA额外实际allocation | 本计划不假设已拥有第二/第三张GPU | ADAPT5K；短离线评价；有限吞吐工程 |
| 两站CPU/合法存储 | 使用现有项目用户态目录和数据路线 | 解码缓存、manifest、统计、场景准备、回读评分和报告 |

Core GPU0的容量是站点记录值，不据厂商常规规格推断；Core不是A40。GPU1能装旧LoRA推理包，不等于已经证明能装完整SCALE推理。完整包必须测量推理显存；不通过临时量化、降分辨率或替换decoder解决。[资源记录][RESOURCE]

默认同时维持Core的MIX和VANDA的TGT；新增目标可用后申请/使用额外一张实际A40运行ADAPT。短评估使用实际空闲allocation或正常训练段切换窗口。没有额外卡时串行排队，健康在训进程继续，CPU准备照常推进。取得更多既有许可内的资源时可并行工程验证，正式训练臂数量仍以本计划为准。

VANDA现有SOURCE_DEV旁件就绪不证明源TRAIN已就绪。A40上的ADAPT需要D1目标、R200、LIBERO1261；同配方MIX benchmark只需要其父端点的原23条目标、R48、LIBERO1261，不能等待新D1或R200。两种工作分别按自身固定manifest完成缺失文件有界增量传输、继承统计/模型绑定、逐文件hash和真实三域CPU窗口核验；复用已核对成员，不反复全库复制。此准备不阻塞Core训练。

同一GPU同一时段一个训练进程；同一训练谱系只有一个优化owner。跨站两个独立Adam不能称为同步数据并行。GPU训练不持有native world锁，两站运行不依赖对方在线心跳。按实际cgroup检查CPU affinity，数据worker避开physics核心；预取深度先2、解码worker先2再实测4，禁止重扫描共享登录存储。提前按实测checkpoint大小、保留端点数和首批raw日志估算临时/持久容量；磁盘不足时报告，清理由用户负责。

### 3.2 单world轮转与旧J尾段

不热停当前健康段。当前注册的有限native段关闭、未决请求核清后，按以下固定优先级排队：

1. 首6个新TRAIN采集条件（本版240次采集预算内）。
2. 已到正式里程碑且完成推理资格的学生固定评估批。
3. 后续新TRAIN采集批，每批最多12次尝试，逐条审计准入。
4. 旧J未派发尾段，保持其原顺序与预算，安排有界批次。
5. 不占科学关键时段的N2工程验证。

连续两个采集批后，如存在就绪正式评价，则下一批给予评价，避免评价饥饿。已启动的正式面板不依据成绩改人口或换模型；可在预登记批边界/有限allocation结束后冷态续其原顺序。当前J全48结束不是新采集的前置；延期的是未派发尾段，已有失败分母保持。

复用现有owner/lease/Journal/onceIO和调度入口，只增加这份任务优先级表及必要冷态接续；不另建影子预算器或多层审批系统。先核对现有watcher，避免两个接续者竞争同一资源。首6采集的独立守候已在WAIT_PREVIOUS_WORLD_OWNER，最迟开工11/19 23:39:04，原采集截止11/20 00:12:04；不能再创建第二个首6守候。若原窗到期仍未派发，保存真实0/部分请求终态，在旧守候和world实际冷态、取得新的有限站点时段后只重绑未执行卡；不延长旧期限、也不把旧窗到期当永久禁止采集。[已运行首6守候][FIRST6_WAIT]

### 3.3 有限allocation之间连续运行

每条训练使用既有链路：发现实际状态→保持RUNNING→到有限时段末完整保存关闭→核验端点→绑定下个真实allocation→从下一更新恢复。

- 记录PID/cgroup或PBS、GPU、lease、实际期限、模型/Adam/RNG、global及phase step、bank/cursor/LR表。
- 未知启动结果先查实际效果，不重复提交。
- 下一里程碑来不及完成时，保存实际step s并在下段从s+1继续。Core已具备非里程碑冷态接续基础。[现有接续][COLD_RESUME]
- 保存余量取实际保存/关闭p95加余量；启动时段不足时先做CPU准备，不冒险追求整齐里程碑。
- PBS使用真实队列和已有账户范围。已确认自有且未运行的无效队列可以按既有机制修复；RUNNING或UNKNOWN不得复制。
- 过期凭证按现有合法机制续接；确实需要用户登录时只报告该阻塞项，其他资源线继续。
- 本审计时Core原first500截止11/19 23:49:33，J截止11/20 00:12:04，TGT后继外层截止11/20 05:02:00且受实际6h PBS限制。这些仅是历史卡片身份，不是执行时可复用的授权期限。

## 4. 工作包、依赖与可验收结果

| ID | 优先级/资源 | 工作 | 完成证据 |
|---|---|---|---|
| L0 | P0/两站只读 | 刷新现场与封存快照，接管现有健康任务 | 唯一owner、实际step/端点、有效期限、后继任务表 |
| L1 | P0/CoreGPU0+VANDAA40 | 原MIX20k/TGT5k连续训练 | 原端点、实际更新/访问/生成和完整恢复 |
| L2 | P0/Coreworld+CPU | 新TRAIN本体采集、有效片段准入 | raw/命令/时间/结果链和实际覆盖；首条新示范 |
| L3 | P0/CPU+独立GPU资格 | 可扩展target reader、正确loss权重、统计与课程bank | 真实128窗口梯度/路由，0更新step-zero独立恢复；完整包输出 |
| L4 | P1/额外A40 | 一个ADAPT5K连续候选 | phase25恢复、500/2k/5k端点、4aux/更新 |
| L5 | P1/A40+Coreworld | 固定离线/阶段诊断及ADAPT0/2k/5k闭环 | 实际同条件分母、阶段失败链、独立确认 |
| L6 | 并行工程/空闲GPU+CPU | microbatch/缓存/同配方A40 benchmark | 暖态时间分解、数值差异、端点恢复、净吞吐 |
| L7 | 有界工程/Coreworld | 修复N2两个具体资格缺口 | 原容差全通过、双向STOP后同伴继续积分 |
| L8 | 持续/CPU | 汇报、代码集成、下一版证据驱动决策 | 固定SHA报告、失败与未派发清单、可执行handoff |

L1不等待L2/L4/L7；L4需要实际新增有效目标数据、L3成立和真实资源，不等待Teacher全格或D5成功。L5的native需要对应模型资格与world，但离线无需world。L6/L7工程失败只影响各自加速路径。

## 5. 本体数据建设与逐条准入

### 5.1 建设目标与有限尝试

新增目标TRAIN目标120条：完整PLACE80（F1/F2×红/蓝×左/右8格各10），HOLD20，纠偏/恢复20。第一批争取24–40，但数量不是开训硬门。首6缺口为F1红左/蓝右和F2四格。

最多240次采集尝试，每次最多50703tick、180次专家计划捕获；总上限12168720tick、43200专家计划查询。首6已有预检包含在240内；若执行端已经采过，按真实runID扣减，不能再计为新6次。到120条目标达到即停止无必要重复；到尝试端点仍不足则报告覆盖缺口及新机制方案。

Teacher、实际可执行规划和人工遥操作均可生产数据。自主执行不依赖人工随时在线，优先复用现有可用生产者与规划修复；人工仅在其实际可用时补充。通过逐条监督与执行核验的记录即可入库。完整成功、局部有效纠偏、失败诊断分别标识。

同一producer配置、同一条件格连续两次同首因失败，停止原样重复该生产路径，读取保存状态定位，作有预测的几何/规划修复或切换生产者后，以新版本继续剩余尝试；全局训练不停。修复可调整Teacher接近/腕姿/净空/落点路径，保持实际机器人、WBC、原安全门和评分。版本改名本身不是修复。

历史评估失败只决定待解决问题类型，不回流旧DEV/TEST轨迹为TRAIN。恢复从新TRAIN场景或已核准TRAIN状态重新建立episode；STOP后仍0积分，不在旧STOP后偷偷接着动作。

### 5.2 监督语义与数据字段

现役Go2监督是策略名义层nominal_policy_H40；编码为相对当前TCP的EEF动作，解码仍回名义层，consumer执行原机械映射。新数据必须把名义、installed/received/selected reference及实际状态分别保存。[codec][CODEC]、[gripper][GRIPPER]

最小schema：episode/scene/reset/session/producer身份与hash；TRAIN/DEV归属；指令与对象；RGB及相机掩码和曝光/捕获时钟；EEF/关节/夹爪反馈；底座位姿速度；TCP/相机/安装标定；名义命令、上一实际消费命令；每点时间与supervision/consumed/valid mask；实际控制参考；接触、抬升、释放、STOP和结果。

label_kind分开：

- expert_query_plan_H40：同次专家给出并经审核的计划40点，可监督后缀，实际消费仍P10。
- executed_future_command_H40：从同一名义层连续专家命令流构造的未来窗口，保留真实时间和有效性。
- partial_verified_recovery：局部有效恢复，需新reader/loss支持mask；不能填复制终态来伪造完整H40。

未来实测状态不能自动当成下发命令。RoboDojo继续native future-logged-reference语义和私有I/O，不能称为已证明Go2命令等价。阶段与对象GT可用于采样/诊断，不能偷偷进入student输入；底座字段先记录，扩展模型状态维度是另一个表示版本。部分片段的FM与有标签宽度监督使用label-valid mask，预测的完整H40可执行性项仍审核全部40点；未完成该真实mask路径核验时，部分片段保留隔离，已合格完整标签继续训练。

### 5.3 目标reader、统计与源200

现役ScaleTargetDataset绑定C20及最多3PLACE，新增一个可扩展版本，不修改活动进程。[现役reader][TARGET_READER]

初始ADAPT数据D1取冻结时已逐条准入的新TRAIN与旧23条，split先按scene/session/episode划分，再切窗口。至少一个已经由本版reader/标签/mask路径核验且实际进入训练成员的新独立episode即可形成D1；隔离partial不计入。实际覆盖不足必须报告，不虚构24条齐备。

初版ADAPT只重估Go2状态统计：用D1目标TRAIN拟合state_stats_v2，检查每维token边界；保持原Go2动作统计/codec、原R48动作与状态统计、LIBERO统计，以减少继承端点物理含义漂移。R200新量化统计保留为独立诊断，不自动替换当前源head绑定；reader显式绑定“R200成员＋R48统计”，并报告超界分布。若实际证明旧动作表示不能覆盖，则单独形成表示迁移版本，不伪称无损续训。

Go2已有PLACE状态越界66.3%，连续动作没有裁剪；优先修状态覆盖。更换action stats会改变旧输出对应的物理动作，不能靠简单改head宣称整个flow matching/噪声/Adam等价。[分布实绩][STATS]

## 6. ADAPT5K：唯一默认新训练臂

### 6.1 初始化、参数和学习率

选择D1就绪时已完整恢复核验的最新MIX正式端点，优先级5k→2k→500；仅按可用时间选择，不按DEV成绩挑父模型。若尚无MIX500，继续现役MIX和数据准备，取得首个正式端点即启动。不等待20k。

继承共享参数、目标/源私有I/O、兼容Adam槽位和RNG。保留父谱系；记录parent_global_step、phase_step和lineage_total_updates=parent_global_step+phase_step。原MIX仍在自身owner下继续，ADAPT是新优化谱系。

保持冻结视觉、VLM LoRA、完整action expert、三域私有I/O、现有混合精度、25% state-token dropout与raw端点身份；首版不加EMA、memory、OmniVGGT或视觉解冻。三个I/O只收本域梯度，未参与域的参数/Adam/decay保持不动。

phase共5000更新。新阶段LR预先固定为从5e-6到1e-6的cosine，j=0..4999：

    lr[j] = 1e-6 + 0.5*(5e-6-1e-6)*(1 + cos(pi*j/4999))

这是显式新阶段schedule，不继承原20k的索引，也不暗中重新warmup。AdamW沿现场β1=.9、β2=.95、eps=1e-8、weight_decay=1e-5、global_clip=1；不因分段、换站或数据D2切换重置。技术训练前缀计入5000，phase25保存并独立验证非零恢复，正式评价端点为0/500/2000/5000。

### 6.2 batch与损失：同时明确两个权重

每更新64T+32R+32L，有效batch128。域内先平均，FM为：

    L = 2*mean(FM_T) + mean(FM_R) + mean(FM_L) + L_target_aux

L_target_aux保留原XYZ/夹爪速度/物理宽度定义和绝对系数，每更新仍4次完整十步H40；各项内部严格保留原归约和绝对系数，4次样本aux再平均。XYZ加速度/夹爪速度超限保留原全轨迹最大超限归约，宽度监督保留逐点均方均值；部分标签按有效点数归一化，不将峰值约束改成40点均值。当前首版不加D5-T delta。共享梯度应为2*gT_FM+gR_FM+gL_FM+gT_aux；目标私有组为2*gT_FM_private+gT_aux_private；源私有组仅对应源FM。

现役Engine对shared target直接赋值且FM/aux已合并；只改DOMAIN_WEIGHT[target]=2会造成shared/private不一致并误缩放aux。新版本应拆开FM/aux聚合。同时检查target32硬编码的aux_slots、frozen_draws、访问银行和cursor；64槽位要全部可被抽到，维持4aux/步。新aux槽位采用预封存的16步轮转，每步4个槽位，覆盖0–63；batch内部按固定随机键打散，防止只监督同一子组。[训练实现][TRAINING_CODE]

L3采用step-zero真实128窗口梯度、私有路由、合并算子与step-zero独立恢复，0次实际优化更新。首次Adam apply是L4的phase1，非零完整恢复使用phase25，两者计入同一5000更新，不形成先完成phase25才允许L4开始的循环门。inactive Adam保持原合同；不为文档或机械字段创建大量镜像测试。分域梯度范数/方向只能量测，不能将batch占比称为实际梯度贡献。

### 6.3 目标采样与一次课程切换

令E_new为已经通过本版reader/schema、标签与mask核验、实际进入D1/D2成员的新增独立episode数。隔离partial、预检条件和失败诊断不满足启动所需的一个新episode，也不进入E_new。每更新新增目标配额为min(48,4*E_new)，其余补旧目标回放，总计64；不同片段来自同一episode仍只计一个E_new。达到12个新增episode后配额为48新+16旧，避免仅一条新示范立即承担48个窗口。

目标内部按任务/条件→episode→阶段分层。50%配额用于整体轨迹覆盖，50%用于接近/闭合/释放/恢复邻域；关键阶段不足时回填有效全轨迹，记录实际分布，不伪造事件。闭合正例以专家真实事件是否在当前P10到期定义，不把所有窗口改成立即闭合。源域按任务/episode分层，R200与L1261不会因轨迹长短自然支配。

预登记一次数据课程切换：phase0–2000使用D1；封存phase2000并交付独立评价队列后，在冷态只检查一次D2；不等待评价成绩。若D2已准入且有新独立episode，则phase2000–5000使用D2；否则保持D1。两段的状态统计、动作统计、codec、模型、LR表和Adam均不改变。D2新bank、成员、阶段mask、配额与新的stage-local cursor在2001更新前封存；历史累计cursor保留，不能将新bank第0项冒充旧bank接续。后来才形成的D3留待下一版本。

因此结论表述为“本体强化课程方案”的整体收益，不能称为纯步数增长或单一源数据因果效应。本版不默认再开比例网格或新2k对照；需要单因素归因时，先提交同父、同源/统计/学习率/目标访问的有限配对设计。

## 7. 评价与部署

### 7.1 离线与关键阶段面板

ADAPT0指继承权重接入本版state_stats_v2后的零更新端点，不冒充原MIX旧输入表示。0/500/2k/5k每个端点固定三域96，共288次预测，再独立恢复每域16，共48次；四端点正式合计1344次生成。底层原始样本固定，输入统计身份明确。

另冻结最多96个目标TRAIN阶段窗口：P10到期闭合正例32、不应闭合负例32、接近16、释放/恢复16。报告实际可得窗口与独立episode数；不足不复制凑数。在ADAPT0首次预测前封存去重的(window_id,label_kind,phase_bucket,noise_seed)清单；每个窗口只能归一个桶，四端点复用同一噪声与判定公式。P10到期按专家原命令时钟/消费范围定义，时序误差采用原分析器的首次22mm跨越时间差，≤100ms为单列诊断；缺跨越记失败而非零误差。面板从D1选定后不随D2换题；每端点最多96次，合计384。它是TRAIN拟合诊断，不冒充DEV泛化。

离线同实例只联合报告阶段/闭合时机、P10到期动作、完整H40和上一真实命令接缝判定。实际消费、目标接触、抬升、保持/完整PLACE由对应native query/episode产生并按实际分母报告；专家历史执行只证明标签来源，不能证明本次离线预测已执行。22mm是当前诊断阈值，抓持和任务成功继续使用原评分。TGT500只有1个正例的旧事实保留。

### 7.2 新科学人口与确认

预先冻结四个独立新DEV布局、每布局6指令，共24条件；ADAPT0/2k/5k各跑同24条件。每24条件为8HOLD+16PLACE，最多1096872tick、3840模型H40。固定新确认另4布局24条件，仅ADAPT5k使用。总计96科学请求、4387488tick、15360模型调用。

确认使用固定5k端点，不按DEV选最佳checkpoint；符合部署/安全资格即可测，即使DEV低成功也如实得到确认结果。若数据身份、输出路由或安全执行证据有缺陷，冻结受影响派发并修复，保留未派发分母。所有场景与新TRAIN、旧留出均核对身份；旧TEST不读取。

仅当新包/路由确需接口复验时，预留原F1/F2两个无接触接口，共11906tick，每接口最多一份模型H40（合计2）。若沿用确定性参考则模型调用为0。正式失败不重跑选成功。

父端点/2k/5k部署各预留16条Core输出复验，共48生成。完整模型包已存在reader与CPU路由基础，下一步是实际GPU输出/显存和native资格，不重写一套框架。GPU1显存不足时：CPU数据采集继续；在CoreGPU0原训练自然冷态窗口先做固定native块，再恢复训练；或在已通过同配方迁移资格且取得A40 allocation后迁移MIX以释放GPU0。上述fallback先与原after500 watcher核对并使用同一个既有handoff入口，不能另起调度器抢GPU0；复训从当时最新正式冷态端点继续。未经验证的WAN实时推理不作为默认方案。[现有部署][DEPLOY]、[现有native路由][NATIVE_ROUTE]

### 7.3 原SCALE端点评价不重复消费

原MIX500/2k/5k/10k/20k和TGT500/2k/5k沿原面板和预算。一个封存端点的正式288+48只选一个站点运行一次；已成功的TGT500结果直接复用。跨站只改变评估执行位置。技术失败消耗保留，不能用“转移站点”重新置零。[跨站评价基础][CROSS_EVAL]

## 8. 提速与多worker

### 8.1 先测同配方，再选择站点

MIX实测暖态窗口p50为36.29–45.77s/更新；20k的量级约202–254GPU小时，队列/恢复/保存/评估另计。TGT5k按已测13.19–14.18s约18.3–19.7GPU小时。A40上的TGT速度不能代表同硬件MIX速度。[吞吐][WARM]、[快照][TRAIN_SNAPSHOT]

记录data wait、FM反传、4次aux反传、optimizer、保存/恢复、GPU利用/显存、CPU和IO。旧50步区间数据等待占11.57%，最新区间13.19%；即使全部消除，单靠它的理想吞吐收益约1.13–1.15倍。优先有界预取、RGB/视频缓存、可信版本的一次性校验缓存；保持样本顺序、随机键、有效batch和loss。

microbatch1/2/4的真实host_batch/domain_gradient接口CPU核验已发布，使用轻量可控loss；该结果不等于完整模型GPU资格，应复用其已核验部分。[CPU实绩与剩余GPU工作][LATEST_TRAIN]

独立工程预算最多4个实现候选，每个从同一封存父端点运行30个更新，最多120更新：优先复用已有Core baseline，实测A40 microbatch1、2、4；已有baseline仅在冻结父端点、样本和计时口径相容时复用，否则第4候选用于Core同条件microbatch1。每个30更新候选预先固定前10步编译/暖态、后20步统计p50/p90并单列保存成本；预编译不产生未记账优化更新。每候选额外至多96个前后/独立恢复预测。固定为原32T/48R/48L配方，工程更新不进入正式MIX或ADAPT谱系；采用新执行版本时，从正式谱系交接时最新已冷态封存的完整checkpoint恢复s+1，继承其Adam/RNG/cursor/LR，不能回退benchmark旧父端点，绝不把benchmark权重导入科学臂。

工程数值比较在跑前固定合同并保留逐位比较和原门结果，不能见到差异才放宽旧判据；归约顺序/后端变化使用新的执行版本。如无法证明兼容，就保留原科学实现，工程结果单列。达到可用兼容合同且暖态净吞吐改善至少15%、实际显存有余量时，可在冷态边界采用经资格的新后继执行版本；这15%是工程采用阈值，不是科学训练门。相同配方在A40实测更优时再作整谱系冷态迁移，原owner已关闭才允许新owner。

不把多worker解码、多个独立训练臂和同步多GPU训练混称。跨站WAN同步不在本版；真实FSDP/DDP需另有实现、路由与实际多卡资源验证，不阻塞现有训练。

### 8.2 N2工程独立推进

历史N2吞吐1.99倍，但8个容差只有7个通过；F2瞬态EEF差7.658mm超过5.999mm，第二个方向的STOP缺同伴后续积分。原结果保持，科学默认legacy N1。[N2实绩][N2]

最多24个新工程请求，每个10703tick，总256872tick；预分配8N1+8N2匹配卡、4个双向STOP角色请求、4个瞬态定位储备。使用确定性参考，模型生成0。两个STOP方向都要求同伴预留足够未积分时长，停下world后续0tick，同伴真实继续；完整记录selected-reference身份、同钟状态和隔离资源。

每world独立CPU集合、ROS domain、临时目录、cache、output、owner和STOP作用域，原WBC/物理步长/容差保持。N2全项通过后，只在新采集批/新评价人口冷态边界启用，人口到worker的映射提前冻结；已经开始的正式比较保持原backend。N2未通过则自动沿N1继续，不暂停L1/L2。event-ready等待缩减同样需要实际消费证据，selected-reference仍UNKNOWN时不删原固定等待。

## 9. 预算与分母

### 9.1 继承预算：沿用，不重新加一份

原SCALE总25k更新、800000目标访问、1920000源访问、2720000总访问；训练完整生成100000、离线2760、native15360，共118120生成；96科学+2条件接口，共98原生请求、4399394tick。原D4/D5/J/M2分别记账；J已发生10条及其尾段不计入下表新增预算。

### 9.2 本版新增上界

| 类别 | 更新/优化访问 | 模型完整生成 | 原生请求/tick |
|---|---|---:|---|
| ADAPT5K | 5000更新；T320000/R160000/L160000 | 20000训练aux | 0 |
| 吞吐工程≤4×30 | 120更新；T3840/R5760/L5760 | 480训练aux+384复验 | 0 |
| 新配方梯度资格 | 128梯度窗口，0优化更新/访问 | 4目标aux | 0 |
| ADAPT四端点正式离线/恢复 | 0 | 1344 | 0 |
| 阶段诊断≤96×4 | 0 | 384 | 0 |
| phase25非零独立恢复 | 0 | 48 | 0 |
| Core三包复验 | 0 | 48 | 0 |
| 技术失败恢复储备 | 0 | 336 | 0 |
| ADAPT科学DEV/确认 | 0 | 15360 | 96 / 4387488 |
| 条件接口 | 0 | ≤2 | 2 / 11906 |
| 新本体采集 | 0 | 0学生模型调用；≤43200专家计划查询单列 | 240 / 12168720 |
| N2工程 | 0 | 0 | 24 / 256872 |
| **新增上界合计** | **5120更新；T323840/R165760/L165760，总655360优化访问；资格128梯度窗口另计** | **38390模型完整生成；专家计划43200另计** | **362 / 16824986** |

技术恢复储备仅用于确认过的技术中断，不重试有效科学失败。若额度未用不强制消耗。任何已发生副作用未知时先查账，不能重新开run绕过唯一事件身份。teacher planned H40、学生模型生成、优化访问、CPU验收和native tick分列；新源码上线前的CPU检查不计作科学进度。

本版不默认再启动配比网格、40k、EMA/alignment/memory或真机动作。它们可由末期分析形成下一份明确有界的后继规格，已有普通工程自主权限不需要逐小操作GO。

## 10. 自主推进、异常与科研决策

- 健康的既定训练继续到端点；TRAIN/H40或闭环低成功只触发定位与并行数据建设，不重置训练或增加Teacher门。
- 非有限梯度、错误参数路由、数据/留出污染、无法恢复的状态、错误动作语义影响对应训练谱系：先封存当前状态与实际消耗，修复后以明确版本接续；其他独立谱系继续。
- native STOP立即保持0后续tick。原安全拒绝是有效科学结果；流程/证据断裂是技术问题，二者分别计数。
- 同症状无新证据的原样重试停止；从保存状态改变诊断/实现。普通依赖、路径、序列化、冷态恢复、Teacher几何和用户态入口修复自主处理，不反复问用户。
- 资源不足、队列或凭证阻塞仅暂停该资源线，继续CPU、另一站和现有健康作业；不占用他人GPU，不修改基础设施或删除用户checkpoint清空间。
- 当前模型输出尚未到目标且近似全开时，优先补正确接近/事件监督和检查表示；若名义参考正确而实际偏离，再沿执行链核验，不能先改WBC补偿。
- 2k后保留固定训练，采集可按失败类型补新TRAIN；5k/20k端点后以实际闭环、数据覆盖和曲线决定下一版本。全程区分暂停时钟仿真、连续实时和真机能力。

到本版端点应交付：完整端点/恢复、全部实际分母、独立示范覆盖、实际GPU小时与world小时、主要首失败、未派发项目及下一有界研究建议。低成功如实结束本版评价，不包装成通用能力通过。

## 11. 时间安排与资源成本解释

采用事件里程碑，时间是排程参考，不是承诺：

| 时段 | 希望取得的增量 | 依赖/不确定性 |
|---|---|---|
| 接手首个工作单元 | 刷新现役任务、一个可恢复MIX正式端点、首6采集排程、R200/新reader准备 | 当前owner正常结束、实际端点和队列 |
| 第一个24小时资源窗 | MIX/TGT接续、首批新增有效目标示范、短评估和同配方A40计时 | 示范不保证每次成功，A40新增卡须实际获得 |
| 后续2–4天量级 | ADAPT首端点与2k反馈、目标数据持续扩充、N2有界修复 | 新ADAPT秒/步以实际30步更新估算 |
| 约5–14天量级 | 主20k曲线、ADAPT5k与固定确认、数据/吞吐总结 | 按当前MIX量级，纯计算约8.4–10.6天；排队/保存另计 |

ADAPT5k先按当前128窗口配方量级预留约50–64GPU小时，但H40/H50配额改变，必须以新配方实测修正。不得把多个卡数量直接乘成加速，也不得为填满GPU重复跑相同无信息实验。

## 12. 汇报与仓库交付

执行端复用现有发布分支/报告机制，不为每个checkpoint另开永久分支。每个正式端点、新数据版本、原生批次或技术终态发布一次简洁ACTUAL；长程任务交由Corelab用户级后台服务与VANDA PBS队列/已核验的事件后继独立运行；不依赖Codex SSH会话或每6小时实时状态轮询。只有后台作业真实注册、正式检查点封存、独立Eval终态、技术阻断及训练谱系终态时，才由既有合法发布机制发布简洁ACTUAL；GitHub不可用不阻断健康训练。本次文档不创建自动化任务。

每次汇报只列真实step/完整端点、三域优化访问、训练/离线/native生成、实际native分母/成功/首失败、合格完整/局部示范数、GPU与world占用、下一可执行动作。使用Asia/Singapore；观测时间与提交时间分开。

计划PR继续以main为集成目标，执行源码来自明确PR417/418固定栈，不认为main已包含runtime。发布本计划不热改现场。固定失败、WBC/腿控/增益/solver、H40/P10/50ms/2ms、安全和评分、旧TEST保持。真实Go2数据本版仅设计schema/标定/离线adapter；硬件请求上限0。

## 固定证据与实现索引

[SCALE_CONTRACT]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/926226914a0c08ee1ebb143e0edea1434728107b/docs/protocols/SLA_SCALE20K_DIRECTION_20261119.md
[D416_CONTRACT]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/cf9d1acbf72e96b1bd158322ff8cbaf8fe28fabb/docs/protocols/SLA_D4_CLOSED_LOOP_AND_GRASP_20261119.md
[CORE_WATCH]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/reports/WM02/sla-scale20k-20261119/CORE_MIX_AFTER500_DETACHED_WATCH_ACTUAL.md
[TRAIN_SNAPSHOT]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/reports/WM02/sla-scale20k-20261119/SCALE_TGT500_OFFLINE_SITE_AND_LIVE_ACTUAL.md
[TGT500]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/reports/WM02/sla-scale20k-20261119/VANDA_TGT500_OFFLINE_TERMINAL_ACTUAL.md
[TARGET]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/reports/WM02/sla-scale20k-20261119/TARGET_AND_CORE_HANDOFF_ACTUAL.md
[R200]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/reports/WM02/sla-scale20k-20261119/ROBODOJO_TRAIN200_CORE_CUSTODY_ACTUAL.md
[FIRST6]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/reports/WM02/sla-scale20k-20261119/TARGET_ADAPTATION_FIRST6_NATIVE_PREFLIGHT_ACTUAL.md
[D5_FINAL]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/afaf35970e56a0827afb092654ce6731f4723fe2/reports/WM02/sla-d4-next-plan-20261119/GRASP24_COLD_TERMINAL_ACTUAL.md
[J10]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4c9ccb9cc1db2dd5187152259762739bdbd98ee9/reports/WM02/sla-d4-next-plan-20261119/J_SCREEN48_BLUE_LEFT_PAIR_ACTUAL.md
[FUNNEL]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/afaf35970e56a0827afb092654ce6731f4723fe2/reports/WM02/sla-d4-next-plan-20261119/D416_CLOSED_QUERY_FUNNEL_ACTUAL.md
[D5_JOINT]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/afaf35970e56a0827afb092654ce6731f4723fe2/reports/WM02/sla-d4-next-plan-20261119/D5_TRAIN96_JOINT_GATE_ACTUAL.md
[RESOURCE]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/reports/WM02/sla-scale20k-20261119/DATA_AND_RUNTIME_ACTUAL.json
[COLD_RESUME]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/reports/WM02/sla-scale20k-20261119/SCALE_CORE_COLD_CONTINUITY_ACTUAL.md
[CROSS_EVAL]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/reports/WM02/sla-scale20k-20261119/SCALE_MIX500_CROSS_SITE_EVAL_STAGED_ACTUAL.md
[STATS]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/reports/WM02/sla-scale20k-20261119/GO2_PLACE_C20_DISTRIBUTION_ACTUAL.md
[CODEC]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/src/go2arx5_vla/sla_adapter/codec_v1.py
[GRIPPER]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/src/go2arx5_vla/sla_adapter/gripper_v2.py
[TARGET_READER]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/src/go2arx5_vla/sla_adapter/scale20k_target.py
[TRAINING_CODE]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/src/go2arx5_vla/sla_adapter/scale20k_training.py
[DEPLOY]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/reports/WM02/sla-scale20k-20261119/GO2_DEPLOYMENT_READER_ACTUAL.md
[NATIVE_ROUTE]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/reports/WM02/sla-scale20k-20261119/NATIVE_PANEL_ROUTE_CPU_ACTUAL.md
[WARM]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/reports/WM02/sla-scale20k-20261119/CORE_MIX_WARM_AND_SOURCE_HOLDOUT_ACTUAL.md
[N2]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4d597259/reports/WM02/sla-d2-teacher-parallel-20261118/PERFORMANCE_PARALLEL_ACTUAL.md

[LATEST_TRAIN]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/39d8f6e341d2c974aa2baed31588b115e0c2c463/reports/WM02/sla-scale20k-20261119/SCALE_OPTIMIZER_AND_MICROBATCH_READINESS_ACTUAL.md
[FIRST6_WAIT]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/39d8f6e341d2c974aa2baed31588b115e0c2c463/reports/WM02/sla-scale20k-20261119/TARGET_ADAPTATION_FIRST6_DETACHED_WAIT_ACTUAL.md

## 13. 2026-11-19晚间后续决定：Scheduler-first 训练链与 Eval 并行

用户明确要求Codex将长程接续训练**挂载到Corelab/VANDA后台**，无需实时查看训练step，Eval可以并行。执行细节、真实站点验收、状态机、任务依赖、截止与STOP详见[Scheduler-First协议](SLA_SCHEDULER_FIRST_ASYNC_EVAL_V1_20261119.md)。此最新决定替换第12节旧“每6小时摘要”要求，保留server自身训练/失败日志，不建立常驻LLM/SSH轮询。

- Core：使用已存在的用户级systemd/after500冷态接续，核对真实当前unit与owner后按封存端点success事件串接有限后继；只允许一个同谱系训练owner。旧有限窗已过期时按现场新授权绑定，不重复旧服务。
- VANDA：在实际PBS支持且配额/凭证有效的前提下，预挂载带`afterok`等真实依赖的有限后继作业；否则采用既有可验证的站点终态事件提交一次。必须回读JobID/状态/依赖；不为尚未分配的未来6小时窗口伪造队列成功。
- Eval：正式checkpoint seal同时触发下一训练段与独立离线Eval，不使训练等待评价成绩；优先独立可用A40，若没有空闲GPU则Eval独立排队。Native Eval依然需要Core唯一world，不能因为允许并行而与另一个native作业竞争。
- ADAPT与原MIX/TGT分别排队；ADAPT仍只在D1数据真实准入、父MIX正式端点及独立GPU allocation成立后启动。计划中的更新/访问/生成/native总上界不变，排队与checkpoint转移不额外生成模型调用。
- Codex须输出两个站点的**实际登记回执**及独立Eval claim，明确REGISTERED/QUEUED与PLANNED/BLOCKED；发布文档本身不是任务挂载成功。

