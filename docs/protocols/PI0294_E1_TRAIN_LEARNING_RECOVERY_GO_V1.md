# E1 GO — TRAIN限定的学习诊断、Recovery Oracle与一次纠正学习

日期：2026-10-16，Asia/Singapore。
状态：**GO_BY_CURRENT_USER_REQUEST / OWNER_BINDING_REQUIRED / NOT_YET_EXECUTED**。
依据：用户已明确要求“仔细分析失败原因，以推进科学主线作为第一目标，放行下一goal”。本文件据此发布一个完整、有限的新E1执行包，不是再次要求用户确认的草案。唯一现役owner登记一次E1_ADOPTION_AND_BINDING及实际资源许可后，可自动推进下述阶段；普通同范围工程修复不另问routine GO。GitHub发布不是已经启动服务器的收据，不产生站点原本不存在的账户权限。
审核基线：1cd38388571f25e9438663ef3c5ba2fcdeed4f03。E0原终局208175f9ccec24edcac4bb50629676c43f2bad3f不可变。机器卡：reports/PI0294/design/E1_TRAIN_LEARNING_RECOVERY_TASK_CARD_V1.json。

## 1. 唯一主线与本轮终点

保留GR00T任务空间EEF10 -> 原source D0执行与S1 -> 实际反馈的学习诊断。最终目标仍为GR00T -> 冻结WBC -> 完整Go2＋ARX5，但**E0已经说明来源域能力本身未成立，当前不将全部精力投到新整机数据、换GPU或低层控制**。

本轮要实际完成：
1. 用已保存TRAIN证据确定预测非法接触的具体位置、时间与动作因素；
2. 在原v3.0的小型成功示范子集上实施真正的Flow Matching微调，并完成自主任务评价；
3. 在安全、实际学生访问的TRAIN状态上验证Recovery Oracle；
4. Oracle合格时自动采集一轮干预式DAgger数据、完成一次受控纠正训练及自主评价；不合格时仍完成独立BC学习诊断，不让Oracle成为所有工作的前置。

最低有价值交付：明确的碰撞归因表、可加载的固定小集、真实训练checkpoint及行为矩阵、真实Oracle覆盖/恢复结果。理想交付：至少一个源域任务出现可重复自主完整成功及一个可实际消费的纠正数据包。本轮不能承诺成功，不以CPU_READY、loss下降或单次接触代替策略任务结果。

## 2. E0事实及不可越过的解释边界

- 尝试12/12；有效11，全部S1_ILLEGAL_CONTACT；1条TRAIN F0/000 T2因中继进程消失工程不完整。TRAIN0/3、DEV0/8、T1 0/6、T2 0/5，不能写0/12有效失败。
- 217次真实新观测/生成意图，216次提交，205个prefix实际执行/ACK；11段被拒绝未live执行，1段已生成未提交。原S1没有修正动作或替换Teacher。
- TRAIN F0/000 T1与F1/000 T1都在任务1.2秒后第4次生成被拒；影子首次违规分别发生在127和156个2ms步，即下一prefix内0.254/0.312秒。TRAIN F1/000 T2运行12.4秒，第32次生成，影子69步即0.138秒违规。**三条TRAIN拒绝均落在拟执行0.4秒prefix之内，不支持仅以“远期未执行预览过于保守”解释。** 不缩短原S1检查或跳过碰撞。
- 真实live安全fault为0；最大live-shadow差约2.28e-13。预测碰撞与真实安全违规分开；影子一致性强，但具体碰撞几何对尚需原件诊断，不能据摘要指定为桌面/手臂/物体某一项。
- T1多数很早停止，T2多数在9.2–12.4秒停止。DEV F0/000 T1存在接触与双侧抓握flag、约23.48mm水平位移，但最高抬升约0.986mm；不是持续可靠抓取/完整任务。该DEV信息仅用于既有结果解释，不进入本轮诊断标签或训练。
- M2048真实续训1536更新期间T1/T2各24576样本，并非简单的任务采样失衡；first64/last64更新平均loss约0.06808/0.03261，不能由loss下降证明闭环技能。阶段覆盖、误差发生维度、on-expert与on-student分布差距仍待测。
- 第一allocation认证过期是独立工程失败。第二allocation的中继中断解释1条不完整，不能解释其余11条有效模型负结果。原报告未唯一确定relay消失的OS/进程原因，不编造其他timeout或文件复制根因。
- E0两作业/租约已关闭，GPU原生1708秒、保守1723秒；E1独立T0与预算，不将它称作E0第三次恢复，也不补写原缺失结果。

证据：E0_FINAL_REPORT.md、E0_MODEL_BEHAVIOR_RESULTS.json、OWNER_INTERRUPTION_RESOLUTION.json、E0_RESOURCE_FINAL.json位于reports/PI0294/20261016T083821Z-localpc-e0/；训练证据为reports/PI0294/20261014T121640Z-incoming-pi0294-r3/M2048_TRAINING_RECEIPT.json。

## 3. 冻结项与可自主修复项

冻结：原v3.0数据/统计/processor与上游源码、M2048原件、EEF10/20Hz/H40/prefix8/原四步推理、source D0/原S1的数值规则、任务几何/原32秒成功判据、WBC/腿策略/plant/增益/代价/solver/原生安全。禁用v3.1混入、BASE伪零、量化、改分辨率、推理轨迹平滑/投影/重抽噪声追通过、DEV/HOLDOUT调参、FlowDAgger真实反演/steering训练、RL、人类视频新预训练、目标整机新物理条件。

允许：读取TRAIN原件、独立源域影子复算与冻结S1调用、现有Teacher的显式状态查询/事件初始化adapter、真实分叉恢复、数据转换/持久collector、下面固定配方的优化器更新和新版本自主评价；同语义的路径/IPC/认证生命周期/计数/持久化修复。

原受保护S1或D0数值规则如确有缺陷，不静默改变后混记模型结果：保留证据，停止依赖该规则的物理/纠正标签；独立的已合格源BC和只读诊断可继续。全局WBC保持RMS不是新的前置。Vanda已经采用的建设许可不被取消，但本E1不替它扩大范围。

## 4. A：先消费已保存的TRAIN失败，不再全库重审

A1 输入只取E0四条TRAIN的实际观测、raw/generated动作、已提交/执行prefix、MuJoCo全状态、ctrl/act/history、实际物体状态与现有S1原件；DEV只保留发表的汇总，不再次生成DEV动作、Oracle标签、候选或训练样本。

A2 三个TRAIN拒绝逐一在**隔离影子副本**重算原动作，保留原判据，额外记录：geom/body pair、接触允许类别、接触点/法向/穿透、首次违规offset、requested TCP/R/gripper、actual TCP/关节/物体、该时刻源专家目标、chunk边界和单位/参考系。缺字段标UNKNOWN；不能将额外日志视为碰撞规则修改。

A3 每条拒绝在同一完整状态比较原模型、当前状态专家，以及最多三种分量替换（仅XYZ／仅姿态／仅夹爪改为专家）共最多15个主假设影子检查。混合动作只作离线反事实、不能安装到主环境；保留耦合效应，不以单分量减少碰撞就宣布修复成功。

A4 对四条TRAIN各选择初态、中间已ACK态、最后安全决策态，去重后最多12个真实学生状态；按时间与ID规则确定，不按Oracle表现挑状态。中断T2的已有安全前缀可入池，但最终task_success保持null。三条拒绝锚点优先保留。恢复全状态需包含执行器/控制历史及随机状态；如果只能恢复qpos而缺关键历史，不能称精确fork。

A5 创建固定8条成功source TRAIN学习子集：F0/000、F1/000已准入双任务对，加每个family在v3.0中按ID排序的下一个双任务均准入组，共4组8episode。若某组没有完整双任务，按原manifest事先确定的排序取下一组并在第一次新GPU生成前封存，不看模型成绩选择。原数据字节与stats不重拟合。

A6 原expert轨迹上的probe选择每episode合法窗口起点[0,80,160,240,320,400,480,560]，再加最多4个原TRAIN真实操作事件邻近合法窗口，去重，共不超过96窗口。和12个on-student状态分开报告。核查真实观测/动作codec、归一化一次、gripper开闭、SO3、未来索引、任务语言；扫描阶段采样占比而不自动改loss/阶段权重。

A产物：TRAIN_COLLISION_ATTRIBUTION.md/json、TRAIN_STATE_POOL.json、SMALL8_BINDING.json、PROBE_MANIFEST.json。可用原源码定位碰撞，不要求再下载外部数据或重新回读全部12视频。

## 5. B：Hopper实际学习诊断，不停在准备或更多旧M2048评价

B0 GPU作业前备齐固定输入、source服务和故障测试。M2048为只读M0。在封存96 source probe与最多12 TRAIN状态上做固定采样的基线；on-student已有完整raw可直接复用，不额外重抽到通过。报告XYZ/SO3/gripper误差、prefix S1可执行率、整块/前缀连续性与原source阶段。单一随机种子和每观测固定噪声表，全模型对照复用；此为开发诊断，不是部署统计。

B1 新模型M1从M2048权重warm-start，**新optimizer/scheduler/sampler，不叫原训练精确续训**。只训练历史projector＋action diffusion/DiT模块，冻结整个原VLM，原Flow Matching目标/有效mask/precision维持；8episode按任务1:1、任务内episode均匀、episode内合法窗口均匀采样。总1024更新。

明确配方：AdamW；peak learning rate=1e-5；weight decay/betas/eps、gradient clipping、有效batch与微批/累积设置从原RECIPE.json精确复用，首新GPU前输出解析数值和hash；M2048有效batch由原回执为32，不能将32随意改为1。新scheduler cosine，32-update warmup，末LR=1e-6；训练seed=20261016。支持原有同语义gradient accumulation，不换优化器/offload/量化。原RECIPE须能从已注册完整状态/哈希恢复；缺件先只读恢复而非猜测。

B2 固定保存/检查点M0、M1@256、M1@1024；每次使用同一source probe，@256和@1024各做4条新的TRAIN诊断episode（F0/000和F1/000两任务），每条最多32秒。它们是新学习版本/诊断条件，不能填补或覆盖E0原缺失行。教师接管OFF、原S1 ON；有效拒绝计策略失败，不在同版本重复到成功。M1@1024是后续固定父模型，不选“最好看的256”替代。

B3 学习检查同时记录trainable-gradient有限性、非零梯度/参数delta、freeze hash、loss曲线、decode error、task事件。若数值错误/数据错位，允许一次有证据的实现修正和新明确编号的训练attempt，累计更新仍计入本包上限；不能扫描学习率/seed/解冻层。

B阶段不需要Oracle先成功。即使M1任务0成功，也应完成已定可信的探测和终局，不宣称“再多训练一定会好”。Oracle若有有效纠正，可进入下述D/E，不要求M1先取得成功；双方都无有效监督时不盲训纠正模型。

## 6. C：CoreLab源域Recovery Oracle资格，可与B并行

复用已经发布的Q0–Q6代码，但承认其原测试仅synthetic/mock，production persistent collector尚需实现。只适配原source T1/T2 Teacher：从实际TCP/夹爪/物体/执行历史推断task phase，使用物体相对目标和合法轨迹；阶段真值仅专家可见，不能输入学生。需要INVALID/UNRECOVERABLE，不能将旧时间戳拼成假纠正。

C1 对A封存最多12个真实状态查询专家H40，以原S1检查拟执行prefix。记录合格率、不合格原因及样本覆盖。不得以只有prefix8安全声称整个H40物理成功。

C2 从这些完整状态建立明确标记的SOURCE_RECOVERY_FORK，原E0运行和原件不改。最多12个2秒短fork，使用原S1逐prefix检查并实际执行专家5个prefix，记录真实40条提交命令以及新观测；再选择每个原TRAIN条件一个固定终态锚点，最多4个fork接续至原任务仿真时钟32秒，按原源域判据检验是否真正恢复任务。不能把新增32秒接在原32秒后却仍称原benchmark成功。

恢复全状态/控制历史可信、同状态回放符合原S1匹配准则才允许继续。标准reset专家原资格可复用；部署身份有疑问时允许最多T1/T2各一条原source Teacher控制检查，计入live启动预算，不再重建完整WBC或新教师项目。

C3 每项任务独立：至少一个真实完整fork恢复成功，且来自至少两个独立student episode的有效状态/前缀，才对该任务标RECOVERY_QUALIFIED_FOR_E1。未覆盖任务继续INVALID/NOT_QUALIFIED，不阻断另一项或B。安全的2秒轨迹不是完整恢复成功；来源于E0的状态确实是student-visited，但独立fork不是新模型在线访问。

C数据分为oracle计划、已S1认证前缀、实际专家执行命令、fork恢复结果，不能把反事实混同target成功示范。用于本轮训练的H40优先取实际expert续接40条命令；只有8条就不伪填32条未来，不支持mask则丢弃尾窗并记账。

## 7. D：条件式一轮干预DAgger，不等待目标整机Teacher

只有C合格的source任务参与。最多8个全新收集episode（双任务合格则每task4个，单任务合格仅其4个、不用另一任务额度反复追成功），使用SMALL8的固定TRAIN布局、固定M1@1024，任务与噪声列表预先封存。

学生先生成，原S1先检查；合法prefix8则学生实际执行。第一次学生被S1拒绝时，在同一真实状态查询合格专家，并在专家prefix被原S1接受后接管余下episode。原非法学生动作永不执行。专家无合法动作则结束，反事实拒绝不算真实安全fault。单episode只从学生切专家一次，避免反复切换新复杂性。

监督从真实expert执行段提取，保存first_intervention_observation和所有后续obs/command/ACK；其后expert-state不能全部冒称student-state。若整个episode无student动作被实际执行，标EXPERT_ASSISTED_DATA，不称该episode是on-policy DAgger；至少有2个不同episode出现真实student prefix和合格纠正anchor后，才将该任务的新增数据标为E1_INTERVENTION_DAGGER。

每个合格任务至少8个完整H40纠正窗口、至少2个不同episode/anchor来源，才能进入E；窗口相关性与独立episode数分别报告。整条episode最终失败不必否定之前有实际认证的局部专家标签，但不得称整段成功示范。默认本轮BC只消费通过局部S1/时钟/来源验证且来自已证明可恢复专家的命令；fail学生动作不做正标签。dataset分叉origin、student/expert执行者、privileged_audit_only严格分离。

D1持久数据实现属于已放行工程工作：正式writer/ingestion、原processor roundtrip、实际CSV/JSONL/LeRobot转换、去重、原TRAIN-only stats复用、独立reader。全空bank不能报DATA_READY。此新纠正bank名E1_SOURCE_CORRECTION_V1，不改v3.0原件。

## 8. E：纠正更新与对照，一次完成而非无限训练

条件：D满足实际有效样本要求，或C提供同样数量且来自至少2个E0 student episode的完整已执行专家续接窗口。后者可以开展E，但必须标E0_STATE_AGGREGATION_ONLY，不能声称进行了新M1在线DAgger。两者均不足时E跳过并交付B/C的真实结果。

从同一M1@1024权重分别新建两个分支：
- M2-BC：512更新，仅原SMALL8 BC；
- M2-CORR：512更新，batch样本50%原SMALL8、50%合格E1纠正，纠正池内episode均匀/合法窗口均匀，合格task内均衡。只有一个task合格就明确为该任务纠正，不给另一任务造标签。

两分支同初始权重、seed、AdamW设置和有效batch32；cosine、16-update warmup、peak1e-5、末1e-6，VLM冻结；总新增更新上限1024+512+512=2048。此对照控制额外优化步数，**不等价于专家查询成本匹配或完整论文级DAgger优势证明**；后续若需额外普通专家示范对照另立实验。

各分支固定末checkpoint在同一4条TRAIN诊断面板自主运行，专家接管OFF、S1 ON，必须报告T1/T2各自结果、干预数据来源和原技能保留。不得利用DEV选checkpoint/调参。单次seed与4条TRAIN面板仅支持学习/纠正可行性，不支持泛化或部署可靠性。

## 9. 资源、新时钟、调度与工程自治

这是用户要求放行的新E1资源包，原E0两次PBS和1708s原账保留：
- 一次owner绑定T0，排队接纳至T0+24h；所有计算/归档硬截止T0+36h；重启进程不刷新。进入分配不早于准备完成，不为等CPU数据空耗GPU。
- Hopper最多1张H200同时，累计8 allocated GPU-hours，最多3次PBS，每次walltime<=4h且不超余额；这些是整个B/E/评价的共同上限，不是每阶段额度。第2/3次仅后续已列学习阶段或同语义恢复，不重复有效失败。无新额外Vanda GPU申请，本包不让未经资格A40替代数值后端。
- CoreLab CPU<=48 core-hours、Hopper活跃CPU<=32 core-hours；各端活跃<=8核（含渲染/影子/解码）；保留至少6 CoreLab core-hours和最后45分钟作收尾。站点实际更小许可取小。
- source单live worker；main/live环境启动最多32（含最多2条专家控制、8条M1诊断、8条互动收集、8条M2评价、最多6个明确工程中断恢复）；隔离恢复fork最多16（12短+4至原32秒）。有效任务负例不占“工程恢复”名额重跑。
- 实际live/fork积分合计<=800000；S1/反事实/clone比较积分合计<=1200000，**所有物理积分总计<=2000000**。warmup、失败和检查实际积分计入，不能重复扣漏或视shadow免费。
- 新GR00T推理请求<=3200，优化器更新<=2048；训练FM前向与预测/采样请求计数分开，不能用此请求上限吞掉正常训练。只读原动作复算不计新model query。
- 新输出峰值Hopper<=120GiB，CoreLab<=40GiB，local-PC<=8GiB；复用原模型/数据只读cache；每分支末checkpoint完整状态、M1@256仅推理必要文件，原M2048绝不覆盖。节点临时中间件可在记录持久化后清理本轮新建tmp，不能清理任何旧资产来腾空间。存储不足先减少可选检查点/日志，不丢必需模型和数据。

中继健壮性修复在GPU前完成：有限run内可监督worker/relay存活，持久状态记录request/obs/model/action/ACK；atomic写入与完整性校验；客户端进程退出和响应部分写入的故障注入。只能重取同ID已完成响应或只读查询，未知模型/动作提交不得盲重试；断开后确认零悬挂执行再恢复到下一个未运行条件。保留原source时钟与timeout安全语义，不把认证/中继错误归模型。

认证采用现有有效合法方式，凭据不入Git/日志，执行前完成项目权限检查。网络/路径/容器/事务/数据writer等同语义问题在本包内自主修复及测试，不新建一系列人工小GO。明确资源不足/数值契约改变只报告一次具体阻断。

## 10. 安全、继续与停止逻辑

- S1正常拒绝是有效模型/Oracle结果，继续剩余预定实验或按本包专家接管；不降低限位/碰撞/力/时间标准。
- 任何真实live/fork原生安全违规或live-shadow关键状态不一致：立即停止本E1所有新source物理及纠正采集，隔离原因；已独立确认的旧source成功数据上的离线学习/归档可继续，不用其他主机绕过安全STOP。
- 没有Oracle恢复成功：C标不合格，D/E对应支路跳过；B继续。M1没有任务成功：不因0分重抽seed，C合格仍可测试E。
- 所有阶段都可在既定预算内形成负结果；达到结果/预算/截止即收尾，不为了“完成goal”造成功。不得在同一阻断上连续三轮重复只报等待，必须输出唯一失败层和下一具体可证伪问题。
- 本轮不读取DEV原件生成新训练监督；已发表DEV汇总可以讨论，但正式DEV/确认后续新版本评价留到另一个冻结协议。原E0缺失T2行保持工程不完整。

## 11. 各端职责与Vanda关系

Hopper：M0/M1实际source probe、真正微调、两个M2分支和固定模型服务；不再仅提供更多离线loss。
CoreLab：原source D0/S1影子归因、精确state fork、Recovery Oracle、真实source互动/自主结果、纠正dataset独立消费。这里不是运行完整Go2/WBC。
Vanda：依据其已经采用的闭环建设GO继续硬件/容器/编译/原生观察与独立开发；不等待E1结束、不暂停健康构建、不重新申请routine GO。新平台成熟不直接替代E1固定source数值域，也不自动恢复TF-R3的安全停止。实现通用adapter可在源/目标间共享，但实际科学结果分域。

## 12. 终报与成功定义

首屏同时给出：3个TRAIN预测碰撞的几何/offset；8条BC数据绑定；M0/M1/M2更新/梯度/动作误差与自主任务表；Oracle查询valid/完整恢复分母；真正student-prefix数、干预anchor、expert已执行窗口、fork与online来源；数据loader实际通过；各端资源终态。

终态可为：
- SOURCE_LEARNING_AND_CORRECTION_LOOP_CLOSED：BC+合格纠正bank+训练+自主比较均真实完成；无须任务分数全通过，但不得称可用policy。
- SOURCE_TASK_COMPETENCE_OBSERVED：一个固定checkpoint在该任务预定2个不同TRAIN布局均安全完整成功；仅开发可重复性，不是DEV泛化。
- BC_DIAG_COMPLETE_ORACLE_NOT_QUALIFIED：真实B已完成，纠正来源失败；
- PARTIAL_WITH_EXACT_BLOCKER / STOP_ACTUAL_SAFETY / RESOURCE_CLOSED。
selected/deployment保持NONE；模型仅登记development候选，不自动部署或扩大到FB13。

交付固定GitHub代码/小报告，私有raw/模型/视频仍在授权存储：E1_RUN_SUMMARY.md、TRAIN_COLLISION_ATTRIBUTION、SMALL8_BINDING、ORACLE_RECOVERY_RESULTS、CORRECTION_DATASET_CARD、MODEL_BEHAVIOR_MATRIX、RESOURCE_FINAL及新checkpoint/data revision。不要只报test数量，也不要重做已合格Q0–Q6 mock当作新科学成果。

原始方法参考：Ross et al. DAgger, https://proceedings.mlr.press/v15/ross11a.html；NVIDIA自定义本体教程 https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md。本文不升级项目固定runtime到官方latest；标准DAgger和普通Flow Matching均非新算法贡献。本轮优先证明source动作学习与可恢复纠正数据链。
