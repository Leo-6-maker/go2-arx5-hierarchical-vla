# MP1-N1 后续推进执行卡：完成独立确认、增量清理、形成下一阶段决策

日期：2026-10-18，Asia/Singapore。
状态：USER_REQUESTED_CONTINUATION_WITHIN_EXISTING_MP1_N1；本卡不是远端采用或执行完成收据。
用户要求：把后续的推进计划推送到 GitHub，让现有 Codex 读取推进。
仓库：Leo-6-maker/go2-arx5-hierarchical-vla。
集成分支：pi0267/executor-decoupled-recovery。
本次事实基线：a1460412f5890335b2286c9f43e6afdedc432944，2026-10-18 13:22:25 SGT；科学 #11/6245896148，维护 #47/6246068229。
现役确认执行源码：6b291000858612423b67838275efa52caf8b633f；不是本卡发布时的管理仓库 HEAD。
唯一执行者：EXISTING_SINGLE_LOCAL_PC owner；不创建第二个科研调度器、第二套清理工作流或新自治时钟。

## 0. 先读、接续、不重启

读取 AGENTS.md、START_HERE.md、统一入口规范和最新 #11/#47；原生 job/owner/lease/ledger 与有日期的实际收据优先于旧文档的历史状态。遵守以下已有合同：

- docs/protocols/PI0294_MP1_N1_OVERNIGHT_AUTONOMY_20261018.md
- reports/PI0294/design/MP1_N1_OVERNIGHT_AUTONOMY_20261018.json
- docs/protocols/PI0294_SOURCE_CONFIRM_V1_FINITE_IMPLEMENTATION.md
- reports/PI0294/design/SOURCE_CONFIRM_V1_FINITE_SCOPE.json
- docs/protocols/GR00T_TASK_DATA_FIRST_FROZEN_WBC_SCOPE_V1.md
- docs/handover/2026-10-18/CHECKPOINT_RETIREMENT_DURING_SOURCE_CONFIRMATION.md（唯一规范清理卡）
- docs/handover/2026-10-18/MP1_N1_LIVE_CONFIRMATION_CHECKPOINT_HYGIENE.md（仅为保护身份补充，不另开任务）

本卡整合既有工作的优先级与验收，不能扩大已冻结实验条件、模型/物理/删除权限或重置预算。常规同范围工程修复仍按已采用 N1 自主执行，无需逐项再询问用户。当前 source-confirm-01 若健康，直接承接；禁止为读取新计划而取消673388、重建场景、重新加载候选或热切执行源码。

## 1. 已经发生的进展：不要再当成待办

### 1.1 训练与开发对照已经完成

S-BC@2048 与 S-CORR@2048 各2048更新、65536实际访问，同一M1父权重与固定配方。两臂实际32768个T2游标及16384个保留原T1游标匹配；CORR另外16384次T1纠正访问来自4个学生parent/8个片段，不是独立演示数。

| 已核验面板 | BC2048 | CORR2048 |
|---|---:|---:|
| TRAIN4 | 3/4 | 3/4 |
| 暴露DEV8 | 3/8 | 5/8 |
| DEV T1 | 2/4 | 4/4 |
| DEV T2 | 1/4 | 1/4 |

全部24个终点条件已独立核验。保留原32格中期矩阵中的耗尽工程缺口；不重训BC/CORR，不重跑有效结果。两条新增DEV成功只在T1；T2减少一条S1拒绝但增加一条完整运行后的物体目标失败，不能称T2成功率提高。既有固定八个BC512状态的16个终点推理与影子诊断已经完成，不重复生成。

### 1.2 确认已经冻结并运行

唯一候选：S-CORR@2048。
完整checkpoint manifest SHA256：94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38。
原M1 parent manifest SHA256：7dba1c4c8abacdb57e2faa0e6d7563ed5dcef2e5473cb13c281eb0e57d5a5150。
冻结confirmation plan SHA256：6e5a45e121aa493e1de32977c00d3f20ecc3c1f9cb7a6b9698438cbb16028e97。
10新场景组（F0=2/F1=4/F2=4）×T1/T2×固定噪声bank29201/29202，共40条件、8个固定5条件session。场景静态资格、泄漏排除与双端绑定已经完成，不重新筛选。

最后现场收据：673388.hopper-m-02为R，confirmation allocation1/2，source家族11/12，父Hopper14/24，8h allocation且费用OPEN；这不是持续实时查询。当前formal training为0新增，原两臂4096累计更新不变。

已独立核验两条正结果：F0/001 T1 bank29202与F1/002 T1 bank29202。每条均32s/641双摄帧/80新观测—生成—ACK/41保持，全程Teacher OFF。它们不是确认成功率分母。
重要负例：同一F1/002的bank29201安全完成32s，抓起/抬升/搬运成立，但released=0、hold=0/41，仍是任务失败。初始图像相同不等于后续轨迹相同，不能直接唯一归因于随机噪声。原S1负例和工程partial全部保留。

### 1.3 新的checkpoint盘点已经完成，实际删除0

CHECKPOINT_RETIREMENT_DURING_CONFIRMATION_ACTUAL.json已于本次基线发布：Hopper905/CoreSSD2309/AMD1425目录，31个checkpoint形态容器，9个完整大型学习状态保留；合格失败payload0、删除0、释放allocated bytes0。原生确认仍健康，candidate/manifest/plan/lease未改变。
这是新的现场检查，不是沿用旧CHECK_01。不要为了本卡再次扫相同目录并产出相同零结果。BC1631属于被替代的有效resume，当前推理不加载它，但其历史精确恢复复现是否脱离原payload尚未证明；不能标记为失败后删除。

## 2. 立即执行的主线：READBACK → FINISH40 → AUDIT → CLOSE → DECIDE

### P0：一次现场接续核对

在现有owner账本中记录本卡固定commit/文件hash、实际读取时间、已完成事项与剩余条件，不新建一套adoption系统。核对原生jobID和进程起始身份、leases、候选/plan/source/stats、未决query/install/ACK、GPU/CPU实际与预留、个人quota及原截止。

按当前真实状态分流：
- job仍R且无阻断：继续其原计划；观察和审计不能占用动作关键路径。
- job已终态且仍有未运行或合格可补条件：先闭合全部已知副作用、结算旧作业；只有原confirmation剩余第二次allocation及N1额度/时钟/站点准入全部成立，现有owner才可提交一次有界续接。不得开第三次confirmation或第13次source家族作业。
- 有未知副作用、错身份、真实安全STOP或失效权限：只阻断受影响lane，先取证；不假设旧进程已停、不重复qsub或动作。
- 确认已完成：跳过评估调度，直接进入审计/封存，不因本卡重复运行。

### P1：优先完成原40条件，保持确认有效性

从固定plan与原生registry逐condition计算remaining，不根据当前成功率或噪声效果变更顺序、候选或样本。两组噪声均需保留；不选择只成功的bank29202，不增加seed sweep，不使用Teacher接管。原H40/prefix8、20Hz命令/500Hz物理、32s任务、原S1与评分均不变。

每个正式条件最多原规定一次工程replacement；SUCCESS或VALID_TASK_FAIL不得复跑。有效S1拒绝不是工程缺口。ENGINEERING_INCOMPLETE先核对真实执行、未安装响应和pending effects，只有确证原结果与effects闭合且尚有原replacement才可恢复调度；replacement保持原场景和噪声身份。机会耗尽则保留缺失，不要求把40格全变有效才能退出。

使用已成熟的有限owner/metadata observer/transport/审计消费者。常规SSH、路径、解释器、锁、quota或元数据同步问题在N1已有权限内自行诊断、回归测试、普通commit、冷态部署和续接。保持0..300s heartbeat与20/60/900s watchdog；新allocation仍须本作业自己的600s/80cycle/故障资格。已完成资格的健康allocation不重跑准入；不得热部署新管理代码到正在执行的actor。

只允许已确定幂等metadata读和已验证可重入计数边界作有界重试；query/install/start/qsub/未知删除效果不盲重发。原生安全STOP不通过工程新epoch解除。健康作业不得为了发布视频或清理checkpoint取消。

### P2：审计全量终态，不只审计成功视频

复用confirmation_evidence及已有raw/video/noise/ACK检查，必要时在独立只读CPU scope运行，禁止把源码已存在当作实际部署。每个原condition应记录：task、scene、noise bank、原attempt及replacement、实际runtime/checkpoint/plan身份、模拟时长、物体谓词、S1/原生安全、raw/video/hash、generation/实际ACK/未安装动作，以及审计状态。

区分：
- 实际策略SUCCESS；
- VALID_TASK_FAIL/S1预执行拒绝；
- VALID_TASK_FAIL/完整安全运行但物体目标未达；
- ENGINEERING_INCOMPLETE；
- NOT_RUN；
- 已有raw判定但独立审计仍缺失。

原判定与审计资格并列保留：缺审计不能擅自改成成功、科学失败或补跑授权。全40格和全部实际attempt必须可追溯；early rejection保留真实短片，不补帧，不模拟补齐。任何数据缺口都在最终矩阵显式列出。

最终汇总至少给出：
1. 每任务20个预设条件中的success/有效失败/工程缺失/未运行/待审计数；同时报告原计划覆盖率与有效条件内成功比例，不能删掉工程缺失后只展示更漂亮的分母。
2. task×10个scene×2个bank的完整配对表，以及每任务在每个scene的0/2、1/2、2/2成功分布；40次不是40个独立场景。
3. 同场景两个bank的success-discordance、释放/撤离/保持差异；仅描述噪声条件下的行为变异，不能把后续状态分岔唯一归因于某个网络内部机制。
4. T1/T2失败阶段、真实物体终点和S1分类；“推远了/举高了”不是目标完成。released=0、hold=0等具体谓词单列。
5. 每任务14/20是否达到原研发里程碑；未完成面板记无法充分判断/原计划成功下界，不降低门槛或改评分。零native fault不证明不需要安全门。

只有一个候选接受本轮确认；最终结果主要检验该候选的源域可靠性，不能当作BC/CORR在新场景的匹配因果对照。保持已暴露DEV与SOURCE_CONFIRM分开。

### P3：原生收口后归档，数据与算力结算同时闭合

优先复用已准备的SOURCE_CONFIRM_FINAL_VIDEO_DELIVERY.json与confirmation_video_delivery.py，不另造publisher。只有全部相关native allocations终态、已执行/未执行动作和query/optimizer intents核对完毕、所需进程/lease/reservation已按真实状态关闭后，才采用最终delivery scope。

单一HF writer、固定新prefix、原视频字节、READY最后上传、固定revision回读。全量成功、负例、工程partial和零步记录按实际内容归档；缺失视频显式列出，不为完整度重渲染或重跑。归档不是再次完成任务，HF上传失败不使已审计本地结果失效。

复用现有N1_FINAL_SCIENCE_REVIEW.md/N1_RESOURCE_FINAL.json及原报告根；若同名已有版本则按owner已有版本规则更新导航并保留历史。最终列出累计used/reserved、每次PBS实际wall/charge、CPU scope费用、清理释放、全部未知effects、实际模型与目标域资格。旧E1费用不搬入新账，不把queued或OPEN成本当0，也不提前结算运行作业。

## 3. Checkpoint清理：已完成盘点转为生命周期触发的增量检查

仍只有既有CHECKPOINT_RETIREMENT_DURING_SOURCE_CONFIRMATION工作流与CHECKPOINT_RETIREMENT_DURING_CONFIRMATION_ACTUAL回执；本卡不新建第二清理任务。

触发条件：新的checkpoint生成run终态、完整后继验证、真实引用解除、明确废弃run，或实际quota风险。没有这些变化时不重复全盘扫描。优先小范围metadata和consumer增量检查，不做影响在途IO的大型重复hash或归档。

允许直接清理已有授权内明确无效/废弃、无活跃消费者、无恢复/对照/唯一复现义务的大payload或可重建cache；无需逐文件再问用户。先核对本人登记根、路径和link关系、PID+start/lease/writer/reader、精确成员/已有hash、当前与既定后续引用，再执行唯一精确删除意图。未知身份或引用保留；不kill消费者以获得删除资格。

必须保留原M1父、BC512/2048、CORR512/2048、当前confirmation候选/processor/stats、必要完整resume、数据、唯一失败证据、STOP/journal、活动环境和WBC/他人资产。BC1631只有在单独证明历史精确恢复不再需要原payload且N1原冗余partial退役条件全部成立时，才可作为SUPERSEDED_VALID_PARTIAL重新审议，不冒称失败。一个hash不能替代仍需复现的权重/optimizer本体。

执行后记录实际unlink文件数、逻辑bytes、可归因allocated bytes/quota观察及保留原因；硬链接、打开文件或quota滞后导致无法归因时如实写明。无可删对象就继续科学工作并保持deleted=0，不为了交差删除模型。版本化保留本次已经完成的零删除回执。

## 4. 确认完成后的科研分流（本卡不自动启动新训练）

当前N1方法决议NONE_NO_OPTIONAL_NEW_TRAINING继续有效，新增formal updates=0。先交付完整确认，再选下一阶段唯一主要问题；不在看到确认成败后换候选、重抽seed、修改loss/夹爪/S1或把确认状态、图像和新标签放进TRAIN。

| 完整确认结论 | 本窗口接下来做什么 | 后续实验准备 |
|---|---|---|
| 两任务达到原研发目标且数据完整 | 封存候选和适用范围；仍非真机部署批准 | 用既有授权准备冻结WBC目标域任务资格，不再重训源BC/CORR |
| T1改善但T2明显较弱 | 完整记录任务差距及T2目标/退出失败 | 只读核对已有TRAIN失败bank与Recovery Oracle缺口，准备一种新的TRAIN-side目标反馈/安全恢复实验草案；无合格T2完整恢复标签则不训T2纠正 |
| 两bank同场景成功分歧较大 | 报告每任务0/1/2成功分布与释放/保持差异 | 准备新的TRAIN-side稳定性/任务终末行为诊断；不选择幸运噪声、不把当前确认当调参集 |
| 工程缺失影响结论 | 如实PARTIAL并定位实际故障 | 仅按剩余原补测/作业机会修复；机会耗尽则闭合，不扩大分母或无限申请 |
| 未达到目标且机制仍不清楚 | 输出负结果与一个最有信息量的下一实验 | 不因有剩余预算自动开S-EVENT、DAgger、容量诊断或更长训练 |

新的模型对照、数据收集和学习干预若以后开展，必须先在已有适用权限范围内独立封存TRAIN来源/单变量/预算/验证计划；本卡只授权接续已有工作与准备建议，不从确认结果推导额外计算权限。未来模型需要新的独立评价，不在已打开确认集上反复选模。

Vanda lane可在原N1合法scope内独立修复EGL/依赖/相机身份和任务接口，但不挤占源确认与最后收尾预算。旧camera03失败和TF-R3真实安全STOP保留；不通过换host或修改WBC权重/增益/solver/plant/夹爪PD解除。WBC保持RMS不是全局数据前置，但真实安全仍是硬门。没有目标任务成功，不宣称Go2+ARX5或真机闭环已完成。

## 5. 资源与时钟：沿用N1，不重新发一份额度

父MP1-N1累计上限继续为H200 80h/A40 48h，source家族H200 36h/PBS12，最终6H200h与4A40h保留。当前source-confirm-01已冻结的子卡更紧：总2次新PBS/最多16H200h/6400新draw，单次最多8h且站点更小限制优先；总160静态+policy starts、live与shadow各1.28M均计入父资源。

审核快照已用到confirmation第1次/source家族第11次；673388正在运行，旧source已结算42833s/父47333s，当前28800s reservation并非新增免费额度。下一动作前刷新实际used/reserved和剩余原条件，不沿用快照代替现况。若需要最后一次续接，必须有实际终态/效应闭合、剩余场景、站点权限、原预算与本作业前置资格；无确定结果不重发qsub，禁止第3次confirmation/第13次source。

固定N1 T0=1792264081；停止新提交/候选为2026-10-19 09:08:01 SGT，停止新增实验block为13:08:01，硬收尾15:08:01。原2700s allocation收尾、20/60/900s约束不变。本卡不新开36小时，不能把新allocation跨过对应截止；实际站点/权限/余额更小者优先。

## 6. 对外汇报与验收

现有owner在下一自然协调边界向#11写一次本固定卡的ADOPTED/ALREADY_IN_PROGRESS/ALREADY_COMPLETE及原生状态，重大科学终局和每最多4活跃小时更新；#47只记录具体新工程/清理收据。不要每个小helper再发一份大计划。

核心交付按原报告系统复用：
- 固定40格confirmation结果矩阵和全部attempt/noise/原始审计索引；若未完成，给每格精确状态和原因。
- 每任务与每scene配对结果、S1与目标未达分开、阶段/释放/保持/工程缺失分析。
- 最终video/archive与固定revision回读，缺失项明确。
- 已有checkpoint清理回执的必要增量或UNCHANGED_NO_NEW_ELIGIBLE_OBJECTS，不重复算零盘点或旧释放量。
- N1_RESOURCE_FINAL.json及N1_FINAL_SCIENCE_REVIEW.md：真实native/effects终局、实际资源、是否达到原研究目标、唯一后续实验建议、仍缺的目标WBC证据。

这些是执行时应产生的交付，不是本文已经完成的结果。若所有lane受硬阻断或原截止到达，报告真实PARTIAL/NEGATIVE并正常停止；不静默等待常规逐项GO，也不虚构后台运行。

## 7. 基线证据与本次发布范围

以下文件在a1460412f5890335b2286c9f43e6afdedc432944可回读：
- START_HERE.md
- reports/PI0294/20261016T191445Z-localpc-mp1/MATCHED_SOURCE_PAIR_FINAL.json
- reports/PI0294/20261016T191445Z-localpc-mp1/N1_SOURCE_CONFIRM_ACTUAL_BINDING.json
- reports/PI0294/20261016T191445Z-localpc-mp1/N1_SOURCE_CONFIRM_SECOND_INDEPENDENT_SUCCESS.json
- reports/PI0294/20261016T191445Z-localpc-mp1/CHECKPOINT_RETIREMENT_DURING_CONFIRMATION_ACTUAL.json
- reports/PI0294/20261016T191445Z-localpc-mp1/N1_MECHANISM_AND_NEXT_BRANCH.md（其中未冻结确认等历史状态由后来的实际收据更新）

本次发布仅创建一个后续接续文件并登记现有Issue入口；不启动远端Codex、不SSH、不运行模型/物理/PBS、不删除服务器文件、不更新在途源码或候选。不将GitHub发布视为已被现场owner读取执行；实际采用应由owner回执确认。
