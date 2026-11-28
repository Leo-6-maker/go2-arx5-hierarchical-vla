# SLA V7.1-A · 无人值守长程自主推进与迭代 Teacher Recovery / DAgger
2026-11-29 SGT · PR #529 · 在既有V7.1/V3.1上增加执行编排，不改写已完成实验或重新制定安全门。用户要求长程无人值守计划、推送和Codex启动入口。**本文件是有界自主执行协议，不是新的PBS、集群配额、资源授权或已完成实验回执。**
继承固定[原PLAN](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/c4364780327924a0db71f84082b7c6b4a54f3bb8/docs/protocols/SLA_CONTROL_AWARE_POSTTRAIN_V7_PLAN_20261128.md)、[原RUNBOOK](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/c4364780327924a0db71f84082b7c6b4a54f3bb8/docs/protocols/SLA_CONTROL_AWARE_POSTTRAIN_V7_RUNBOOK_20261128.md)、[原REGISTRY](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/c4364780327924a0db71f84082b7c6b4a54f3bb8/docs/protocols/SLA_CONTROL_AWARE_POSTTRAIN_V7_REGISTRY_20261128.json)；执行证据 07c433dd24ad359933a0b4c7381a10e12373abf2。配套[机器登记](SLA_V71_UNATTENDED_DAGGER_REGISTRY_20261129.json) · [Codex启动prompt](SLA_V71_UNATTENDED_DAGGER_CODEX_START_20261129.md)。以后先读PR实际head及执行分支增量，现场有效授权、实耗、租约和已封存终态优先；不得以旧文档倒退真实进度。历史实验始终采用其原SHA。

## 1. 最终目标和当前快照
目标：R4旧能力保持 + 学习真实纠正 + 因果控制状态补足，在**冻结原WBC/OCS2/PD/腿策略/L1**的前提下，实现学生自主完整PLACE，并用独立布局验证。数据采集时Teacher可用登记的仿真特权信息；部署学生仅合法RGB/语言/本体状态/控制接口，无GT、任务阶段特权或未来状态。Teacher辅助成功、raw门内、物理标签一致、训练loss下降均不等于自主成功。
当前主线 Frozen Expert → C1 → A2；分清冻结网络参数（学习方式）、Teacher接管（数据来源）、A2（策略观测）。Frozen Expert不是DAgger；D1是intervention-based DAgger采集，当前214窗尚未参与真实训练。后继合格学生再采集、聚合和更新才能完成新迭代。

| 已核验事实 | 状态/约束 |
|---|---|
| V65 1K/3K、V7-A双500、B6双250 | CLOSED负结果；B6不续500；S1直接换统计不再执行 |
| Expert回置全37格 | 66新前向+8复用混合+222复用旧输出=296；原非法5/37同成员，新增0；Query72首缝30.407、H40最大36.232mm/s；Query84 3.851 vs R4 3.880mm；Teacher XYZ+2.08%。混合权重仅诊断，不原生准入、不冒充真实冻结训练 |
| A93/R3 | 93完整示范7916窗+3晚期纠正115窗=8031；全96条315480有效点物理复演通过 |
| D1 | 红15、红19成功，蓝17接触STOP；2/3教师辅助成功、学生自主0；新增214窗（100+114）、两个真实reset同场景家族，独立布局0 |
| Recovery全部库存 | 17/20次、6条369窗；C1只纳入原R3+两条新红，不暗混另外40窗 |
| C1 | 250行配对bank、8245窗/98回合；原94共同槽、原aux/源/rates匹配；组合RGB训练reader及真实CUDA/模型资格未完成，实际训练访问0 |
| A2 | 原8031+新增214=8245×6因果Accepted Reference残差、时间/顺序/上一实耗开度对齐；不等于直接selected reference或模型接线 |
| C2 | 计划500纠正访问中HANDOFF11、(0,2]s27、后续462、LOWER/OPEN/WITHDRAW0；首接管和教师后续不是同一状态分布 |
| Frozen Expert | CPU机制/ABI/保存接口已准备，真实H200资格和独立完整模型恢复仍缺；原24/24工程额度耗尽 |
| 实耗 | 正式5100/8000、工程24/24、学生原生46/50、Recovery17/20；历史320GPUh及2026-12-05 12:41:52 SGT截止仍有效，实际GPU小时余量须现场核实 |

固定结果：[回置37格](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/07c433dd24ad359933a0b4c7381a10e12373abf2/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V71_POSTRESET_EXPERT_FIXED37_RESULT_20261129.md) · [A2/C2实绩](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/07c433dd24ad359933a0b4c7381a10e12373abf2/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V71_EARLY214_CAUSAL_CONTEXT_20261129.md) · [C1/第二条D1](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/07c433dd24ad359933a0b4c7381a10e12373abf2/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_D1_SECOND_RED_ACTUAL_AND_C1_PREPARATION_20261128.md) · [冻结CPU](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/07c433dd24ad359933a0b4c7381a10e12373abf2/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_FROZEN_EXPERT_CPU_PREPARATION_20261128.md) · [分区反事实](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/07c433dd24ad359933a0b4c7381a10e12373abf2/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_B6_PARTITION_COUNTERFACTUAL_RESULT_20261128.md) · [蓝卡失败](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/07c433dd24ad359933a0b4c7381a10e12373abf2/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_D1_BLUE_CLOSE_FAILURE_20261128.md) · [B6停止](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/07c433dd24ad359933a0b4c7381a10e12373abf2/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_B6_250_PARAMETER_FUNCTION_DISSOCIATION_20261128.md) · [全量标签复演](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/07c433dd24ad359933a0b4c7381a10e12373abf2/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_FULL_PHYSICAL_LABEL_REPLAY_20261128.md)。
本次发布不产生模型前向、更新、积分或数据访问，不能用发布当实绩。

## 2. 无人值守的任务调度规则
### 2.1 启动先核对，而不是再次派发
读取PR #529当前head、原V7.1 registry、执行分支 experiment/sla-v7-control-context、Core/V65实际head；如有后继只比较增量。检查当前工作目录是否干净、已有本地未推提交、真实git代码版本、site/owner/lease、PBS job ID/状态/真实更新时间、模型/数据/噪声/score/split指纹、实际资源与磁盘个人quota。只使用已配置/获准的访问方式，不改凭证、权限、网络、调度基础设施或其他用户任务；保留用户本地工作，不reset/force push。已有健康作业接续监控；未知状态先读现场，不因客户端断线、轮询超时或无Git更新就重启。只有确认前一job/owner终止且合法新capsule可用才接力。部署只限仿真，不迁移真机执行。
不要再次跑已封存37格measure、原96条标签复演、原8245 context、原B6训练、旧蓝卡或失败旧卡；只有相关输入、源码、合同或新数据发生变化才核受影响部分。旧证据按SHA复用。
### 2.2 两站点分工和并行
- **Hopper**：真实H200资格、正式训练、必要的真实训练梯度诊断。主正式实验一次只选一项，复用已有调度与单owner，等待CPU或现场资源时释放H200；不申请GPU空等授权，不为凑满时长重复训练。
- **Corelab/本地授权计算节点**：C1 reader、合法固定权重评价、独立恢复、Teacher data工厂、所有原生评价、A2/S2和C2 CPU接口、红蓝执行归因。按实际神经/物理owner锁和显存/RAM/SSD/HDD容量串行互斥工作；CPU任务可并行，不把未验证N2/N4当并发能力。Core原有数据生产可继续在不冲突的现有配额内，Vanda暂停，SOURCE608准备不派发长训。
- 有多个Codex/subagent时由一个协调者管理同一状态账本，分给子代理的工作限定为不同只读/CPU包或隔离工作树；子代理不能自行派发、抢owner、改原计划门或重复写同一工件。没有子代理则同一Codex按READY优先级切换，不等用户每一步指示。
### 2.3 持久接力与退出状态
在现有执行机制中维护一个append-only接力记录（建议每轮既有报告目录下 V71_AUTONOMY_WAVE_*.json/md；不新造后台守护进程或第二套调度器）。每个包记录 id, state, parent, run_id, code/data/roster/bank/inputs SHA, owner/lease, job/terminal, started/finished SGT, actual/reserved/remaining, dependencies, decision/evidence, next_action。状态用 PREPARED / CPU_VERIFIED / BLOCKED_AUTH / BLOCKED_RESOURCE / BLOCKED_DEPENDENCY / GPU_QUALIFIED / RUNNING / SEALED / CLOSED_PASS / CLOSED_NEGATIVE / CLOSED_TECHNICAL / UNKNOWN_RECONCILE / COMPLETE；科学未通过与技术失败分开。已完成的数据准备不能标记训练通过。
每波真正完成一个包或发生关键异常就封存实际回执、更新实际执行分支并远端SHA核验；若长任务仍在运行，以现有调度方式低频状态检查（建议10–20分钟、或合法job事件驱动），最长约12小时提交有实际变化的阶段回执，不创建大量空提交。不以报告作为停止条件；报告之后自动选择下一READY包。若客户端/会话重启，先读最新记录和现场lease/job，不依赖聊天记忆，不重复已完成工作。未知或无法访问标UNKNOWN，继续其他独立工作。持久化只是计划中的Codex工作方式，本次没有安装守护进程、计划任务或改变客户端电源/审批设置。
### 2.4 调度循环（有限预算，非无限重试）
1. reconcile实际状态、有效额度和期限；分类READY/RUNNING/BLOCKED/CLOSED。
2. 优先完成活跃合法作业及收尾，随后选最高价值且依赖齐全的READY任务；等待Hopper时继续Corelab/CPU包。没有新授权也不要空转：完成所有可独立推进的实现、输入和诊断。
3. 每次派发前检查该项实际合规授权、实耗+已有reservation+最大新增成本≤对应硬上限、实际GPU小时/磁盘个人配额、owner、完整身份及预登记的有限终点和安全退出；留原deadline−900s保存余量。仅通过这些门才在已有调度入口派发，不重复提交。现场更严格限制优先。
4. 完成时先真实终态/完整封存/独立回读，再做固定科学门；PASS、NEGATIVE、TECHNICAL、UNKNOWN不同。一个科学失败不自动多步/改λ/放宽门；同类技术失败最多一次有明确修复且原授权允许的新资格（旧失败、实耗保留；不是自动重试权）。不重开已硬STOP或关闭原生卡。
5. 自动选下一证据支持的候选；同父/数据/输入/noise/步数/优化器/损失及评价逐项匹配，唯一变量明确；不依赖逐步骤人工批准，但资源/权限/不可逆操作需要现有明确授权。达到目标、原期限、明确撤销、硬安全/数据身份无法修复或所有READY工作耗尽且外部阻塞时，安全收尾并给最终状态；不能伪称成功。无合法可做工作时输出一次合并阻塞清单和具体恢复条件，不无限空轮询或重复索取授权。

## 3. 任务图：从当前起点到完整闭环
P0-Q为主关键路径；P0-PANEL、P0-READER、P0-WB、P0-A2/S2、P0-L0并行，不等Hopper。以下数值是原V7.1已固定门或本次候选设计，后者必须在看新输出前登记、核资源，不能用于追溯重评旧实验。
### P0-Q：Frozen Expert真实资格与正式250
当前blocker是工程24/24，而非实现概念或正式步数。真实资格至少2步，现无新增分配。先核实际后继授权；若未分配，准备资格入口、完整真实FM/aux/deploy图、父权重/Adam时钟、snapshot和负例；不要用合成数组、shape ABI、未保存更新或正式步名义绕开24。
- R4-200新建参数与私有Adam。共享Expert427932672和原VLM/LoRA逐字节固定，原共有Adam count=0/moments不动，不调用共享AdamW/weight decay；私有Go2/Robo/LIBERO共6498400更新。原优化器clip=1对活动私有梯度范数，报告原全梯度范数、活动范数、两种clip比例和实际Adam proposal；梯度裁剪变化是本候选组成，不声称隔离纯参数冻结的所有效应。保留完整状态结构，记录真实速度和内存，不按参数量猜吞吐。现役中性previous-command投影保持中性，不加入新纠正、A2、S2或视觉改动。
- 使用真实A93/R3、原bank SHA a8fde923b6453be823fd8097b8440a7ad8780d6d484bc3d882df8a5b070e64ff，Go2 48/46/2，Robo16/LIBERO16，aux4(0,1,48,49)权重.375，原500日程前250（peak8.481070829156122e-7 warm50 final.1peak），真实逐步FM noise/tau/aux。Step0原FM、部署H40、aux生成完全一致（预登记同backend/dtype容差），两次真实有限更新证明实际活动分区非零/冻结值与moment/count不变；保存完整权重/Adam/RNG/cursor，Core独立全文件回读恢复与下一批数组、噪声和LR。独立恢复若无需更新不计工程更新；实际任何新更新按原规则计数。资格端点不得带入正式父。
- E0通过才派发正式250新步，原Dense B6零250仅在完整匹配时复用；未能证明匹配则暂停对照，重新登记配方和成本，不假设可无条件复用或另跑控制。100仅恢复端点，不用保护集选步；250完整封存。中断恢复必须验证各自完整snapshot、cursor、真实下一行、账本interval；若仅完整端点可恢复则保留未封存实耗，不能跳过或冒充连续。存储按实测checkpoint/个人quota预留两保存+一有限中断容量及15min收尾，未获许可不得删除/转移关键端点；scratch与hpctmp独立。
### P0-PANEL：在新端点结果出现前完成评价名册
原37格回置已完成不重复。准备真正Frozen Expert候选的原37格评价（R4和Dense/L2SP已存实际输出核身份后复用）及额外12状态/4个未入原37格的完整PLACE TRAIN回合，CLOSE/LIFT/OPEN各1；三份实际不同noise，R4/候选72个新前向为待分配设计（原37格候选所需前向和源域42格另计）。尽可能按不同scene家族分组、排除保护布局与原37选择状态；这里是未用于选择的TRAIN诊断，不是独立未见训练数据。先封名册和noise，不看候选输出后挑状态。缺真实噪声/资源/目标几何记缺测，不称通过。
沿用V7.1原REGISTRY数值门：
- 对同输入同noise的R4原合法条件新增raw非法=0；检查原全H40最大速度≤40而不是只看首缝；原5个R4非法条件逐项保留/改善/恶化报告。
- Query72原安全抓持、原R3 OPEN在40门内；Query84上移≥R4的0.95；四个Teacher LIFT每格上移≥R4的0.90。
- 30 Teacher条件XYZ均值≤1.05×R4；各phase≤max(1.10×R4,R4+0.02mm)；SO3均值≤max(1.05×R4,R4+0.005°)；jaw均值≤max(1.05×R4,R4+0.05mm)。不以平均掩盖新非法或任务停滞。
- 源域原42格/臂、原10%告警不触发；Ville固定旧私有IO仅共享表示诊断，未测LIBERO闭环保持。保留语言条件化：在原合法独立TRAIN几何的审计层核对指令目标响应方向，GT只作离线评分，不进学生；无目标几何/方向证据=MISSING，不是PASS。独立多噪声按每状态/噪声匹配检查新增raw非法和阶段动作，报告R4本就非法，不能替代原生。
- E1通过且资源与模型原生资格齐全，冷登记第一保持候选E2。旧学生最多4张：Dense及候选各 N03_202/P02_702，完全相同原reset/noise/合同，只方向筛选；若不够4不得挑单臂/好格。如先前有新合法分配则按原完整每臂10格。混合回置诊断不能替代新正式端点或直接使用名额。E2不劣于Dense且未新增硬安全失败、存在保持/阶段改善，可进入后续**机制实验**；R4始终部署基线直到完整原门+独立确认通过。仅E1阳性但E2资源不足，不能晋级；可准备C1，只有原授权明确允许该层级的机制研究且配对评价计划事前封存才做C1，标PROVISIONAL，不把离线/少量样本称完整保持。
- 若真实冻结候选失败：保留负结果。先在固定TRAIN上对其IO action input/output/time MLP单分区回置诊断，另预封小面板；根据结果只选一种 Output/IO范围、Expert低秩或函数保持候选，不盲目解冻或恢复B6。合法真实消费P10保持需直接selected/consume/成功阶段原证据；父raw函数保持另命名、仅已验证安全阶段/实际噪声，不能当教师真值或保护失败OPEN。无资源只做准备；不等待本包而停止独立A2/S2/数据工作。若保持无法建立，不以C1/A2为同一候选补多个因素，另登记机制单项并明确对旧能力不能承诺。
### P0-READER：C1训练数据真正接入
当前bank 14ffa20aecb55753fc3b8c315699595f2ceeb1e931d9e01027c529ec26251a11，context上一实耗命令 e8937bd193216db09290ff12b9ff53f6cdd29137d456826d116e03ad94aad37a，8245×6 A2 context f2228bc99bd6b80636029a9c38beeda8423a87ce007a95948b0c39e35b01042e。将原真实A93/R3 reader按旧顺序8031行与两冷manifest100/114行拼接，不替换原读取器或现役bank；逐行核source/scene/tick/RGB/state/prompt/mask/实际动作label/width，抽查全部新214及原边界、尾窗/无效位，验证索引98回合、8245行；优先全新行+250×实际采样行hash，重复行只算访次。共同94实际模型输入完全一致；末2差异槽同scene/task/phase/full-H40；aux实际0/1/48/49仍指原样本。CUDA noise/tau/state-drop/aux随机数组按相同逻辑slot绑定，不仅比较seed。验证两个首接管row的原实际学生RGB/state和教师第1动作支持；不得把未执行后缀或失败学生prefix当专家标签。
C1的配对bank、索引和causal context已完成，不重复生成；reader、实际组批与资格未完成。只要原scope未选定，先准备通用reader与完整负例，不让代码阻塞正式首个候选。
### P1-C1：冻结保持方式下的早期Teacher纠正
冻结方式通过上述机制筛选后，重新登记C1两臂**同一个R4-200父与各自新Adam**、相同已资格冻结范围。选择R4 fresh是为了与原冻结scope机制对照隔离数据变量，不把刚训练的Frozen Expert250仅作为其中一臂父；若未来选择不同共同父，需单独登记并不再与旧端点直接因果比较。使用原250行C1 bank、每步48旧+44共同新+2原R3+2差异+16+16=128，aux4/.375原行；Control普通教师、Treatment新纠正；两组各250（新增500），原500日程前250、actual noise与支持配对。不复用旧A/B6/Frozen Expert250作C1 Control，因为bank不同。数据准备完成不等于GPU资格，至少按新reader/scope实际需要的资格更新与Core冷恢复；原工程额度不足时只准备。
对照意义：当前Control1个episode/61独立窗，Treatment2个reset/96窗，两臂500计划访问分别max repeat45/25；差异不只是纯学生分布，而是纠正数据包（含多样性）净效应。仅214新窗中早期100窗被计划采样，后段LOWER/OPEN/WITHDRAW不在两个差异槽；C1不是直接释放修复。完整C1每臂32000 FM=24000Go2+4000Robo+4000LIBERO；Treatment早期500/24000 Go2，原晚R3 500，aux1000每臂。记录实际不同episode、window、stage、handoff距时、原安全/失败，不能把计划索引次数当真实访问。
评价用原37格和预封独立TRAIN（包括合法教师及实际学生纠正附近、非保护集），相同R4保持门、源域原告警、相对匹配Control真实近端意图/接缝/任务阶段改善；必须有预封实际评价名册与合法资源。E2少量方向、E3完整配对另列。若有新学生合格，才进入P2-DAGGER采集。C1仅红色同一场景家族阳性不能宣称蓝色/独立布局泛化。若双方都退化，先返回保持/控制状态诊断；若两臂保持且Treatment无方向性收益，不盲目扩步，检查首接管曝光与数据条件歧义。
### P1-A2：独立控制状态候选
全8245×6 Accepted Reference−实测TCP上下文已经可用：固定episode mount内XYZ m、SO(3) log rad，有因果上一消费wire、时钟、有效位；不是WBC内部selected真值。部署入口同拍取实际receiver可读参考、测量、frame、age，首窗合法暖启动或共同mask，未来、跨回合、过期、错单位或不完整消费拒绝。教师物体/阶段只能审计，不进模型。原离散state和action统计均保持；新增Go2私有零初始化6维投影，固定物理尺度和显式valid、与既有state-drop策略共同登记。Control同结构同参数量/中性条件，Treatment真实残差，step0完整输出相等、非零注入真实影响FM、部署10步采样、aux全H40，真实梯度/保存/独立恢复；公共源不误用Go2条件。检查是否真正利用平移和旋转信息、固定同观测合法残差变化时输出敏感性、阶段条件和原安全。可在TRAIN上先做条件分布歧义/尺度诊断。
优先在C1结算后，使用**固定同一已选择数据版本/采样bank**（C1若阳性用已选Treatment bank，否则若C1无收益用原A93/R3）和同一R4父及保持scope，从共同父fresh训练两臂各250（500）。这是本次候选设计；若另选共同已合格父，须明确改登记，禁止混用父/优化器/噪声。同结构中性Control不能用不同结构旧训练替代，除非完整功能和训练身份逐项等价并预登记（默认不复用）。保持原母日程、原FM/aux；不同时加入S2/BASE/ARM/HISTORY。若C1因资源阻塞，A2继续实现和零更新诊断；只有已有明确保持scope、数据、独立实验评价/资格和合法资源，才允许独立机制A2，不将其冒充C1结算。
### P0-WB：低层执行约束与Teacher纠正
蓝17接管后最后250ms参考−0.450mm、TCP−11.129mm、安装座−17.991mm；FK Shapley底座−16.488/arm+5.359mm。红15/19完整辅助成功不是蓝色泛化。配对红蓝的PREGRASP/DESCEND/CLOSE/LIFT实际时间、近端参考与L1、accepted/selected证据、arm/base request、真实q/qdot/PD/支撑足、桌面/指垫净空；缺直接selected标UNKNOWN。不能把静态FK归因当动态反事实，不改WBC权重、L1、PD、支撑足或加固定高度补偿。通过合法现有可用状态研究教师接近/停滞/全身可行性，优先只读归档和静态首包资格；如有明确新候选，最多使用1个灵活Recovery名额，在新TRAIN场景/reset/noise上真实验证。原蓝17终态不可复活。新路径仍通过所有原安全门。
### P2-DAGGER：把采集—学习—再次采集真正闭环
不是在正式训练过程中让Teacher直接替代学生或使用特权输入。每一轮固定学生checkpoint+provider/代码/动作合同和新TRAIN roster；优先最近通过E1、真实原生方向筛选且无新增硬安全失败的候选（仅采集资格，非部署晋级）。若无合格候选，至少保留2次给下一版，不把R4重复红卡充当新策略迭代。
1. 学生实际生成H40并执行P10，真实2ms/50ms。采集Teacher在完整P10、未STOP、无未确认安装、原评分/时钟/租约一致的边界观察风险。D1 PRECONTACT：合法抓持前、0<腕-目标净空<20mm；LIFT_STALL：已抓持、连续3个查询/1秒进展<2mm且未达20mm抬升，严格沿现役适用条件；原G7预安装拒绝路径另保留。20mm是采集预警而非安全门，成功Teacher也可能触发，不能当失败分类器。新阈值/触发只在独立TRAIN上预登记；教师GT只供教师用。
2. 继承学生实测状态、当拍accepted参考、上一真实夹爪序号和开度，教师从真实偏移状态生成后续动作；首包序号=上一已消费+1，全部经过原H40检查/L1/40mm/s及真实WBC。教师接管后原学生自主success=false，辅助结果单列。硬STOP之后零积分/安装/放行，不从失败点恢复；原生attempt包含失败、技术中止、不触发，未启动不计实际。
3. 采集阶段完整记学生prefix、实际H40/消费P10、教师handoff intent与时钟、raw/accepted/selected/consume（区分直接/推断）、状态/图像/mask、真实成功与安全。当前准入只含合法成功教师实际消费后缀，逐点逆重锚定并保留尾窗mask；失败suffix和学生prefix只诊断，短安全纠正若研究须另建schema、资格及独立实验，不能混成功BC。Teacher生成但未执行动作不当实际标签。
4. 对每新回合冷封存文件SHA/终态/实际序号与25tick原子、TRAIN split/scene家族、reader和物理回放、上一实耗开度和A2因果残差。新dataset version只增不热换当前bank，保留原数据及失败；窗口数与独立handoff count分开。真实新学生来源/episode grouping/行为策略identity写manifest，不称独立布局除非确实不同家族。
5. 重新训练时保留原示范、旧R3与已准入纠正，按有限bank做匹配对照；默认C1先结算。后继C2时间分层、Recovery比例或新的学生诱导样本各自独立变量，不能一起改变。每轮先固定一个训练配方与评价，真实端点通过才替换采集学生；部署/晋级仍要求完整E3。最多按实际Recovery余额2次留给新学生（可依实际两个不同故障/场景安排），不默认每轮追加资源。
指标：每个学生版本独立attempt、介入前阶段/首次失败、介入率、成功Teacher恢复率、无介入自主成功率、每次干预长度和实际消费监督点、不同场景/阶段/首接管状态覆盖、同版本固定对照收益。教师辅助不计学生成功；episode和scene分母独立，重叠窗口不独立。后继完整自主任务阳性出现前不派发全模型RL/RECAP；迭代DAgger采集可在此前逐步进行。
### P2-C2 / S2 / Loss / 视觉扩展：有证据再择一
C2（接管距离分层）当前只准备：固定C1 Treatment为Control，若C1保持但纠正净收益弱、首接管曝光不足，另登记C2 Treatment仅重采两个新早期槽。候选500次差异访问中HANDOFF25、(0,2]s75、其他早期400（5%+15%=20%，原7.6%）；先按两个episode均衡、原四阶段/有效H40与共同槽、同scene/task逐slot校验可行性，不能改变C1原bank或phase support；两首接管与8个早期window独立人口很小，报告最大重复/真实多样性与可行性，若分层不可行则先登记可行单一比例，不强行凑配额。真实C1 Treatment端点仅在同父/scope/step/noise/其余全部匹配时复用，候选新增250否则配对500并重排预算；不改变原C1科学问题，不将同一窗口重复当新状态。后段LOWER/OPEN/WITHDRAW另立过渡采样/aux对照，不混进C2早期变量。
S2：旧state token保留、连续原7维物理状态/valid无裁剪、TRAIN固定尺度零投影；原统计S1负结果保留。两臂各250；只有状态分箱歧义和独立资格支持才正式。
L0：预封3–5个可复现真实batch/实际Adam邻接端点，分旧/新/R3/源/aux、P10/suffix、7/25、XYZ/SO3/jaw、phase，重构真实合成clip与Adam方向；仅已有实际合法数据的梯度/前向诊断可以推进，任何真实新optimizer更新均按原口径扣费（不以shadow隐藏）。只有重复证据才选B2（P10权重2其余1，按mask有效权重归一；全H40原检查不删）或B4（只换一个合法完整H40 aux槽，原FM/总aux4/.375不变；Recovery vs关键过渡独立）。B3/B1/B5依原方案低优先，不能单批断言源域干扰。
SFA/EMA/RTC/Source608：继续读已核官方组件、license、固定真实teacher feature及梯度路径、视觉输入对齐，准备F冻结/L开放LoRA/LA+OmniVGGT分离对照；官方π0.5离散state、align .5、EMA .999并非我们实证最佳，不混入首FE/C1/A2。EMA短程读出另验，RTC只因果历史同步暂停不强制重复已消费P10或伪造异步延迟。SOURCE608只准备，正式扩源、RL或更长训练须新证据与有效资源，不自动延长至5K/10K，不改WBC。
## 4. 科学晋级与失败分支
E0工程资格、E1固定TRAIN与额外多噪声、E2方向原生、E3完整匹配原生与独立确认相互不可替代。原R4门：N02 2/2、独立HOLD≥2/4、PLACE grasp4/4、lift≥3/4、in-zone≥2/4、无新增硬安全STOP、原源域MAE告警未触发；完整PLACE≥1/4优先或前段不退化下有真实阶段进步才可讨论候选保持，**最终完成仍须自主完整PLACE阳性及独立布局确认**。逐卡报retained/gained/lost，安全失败不被总成功数抵消。原生完整H40、P10、2ms、50ms、G7 40、原接触/支撑脚/硬STOP、零STOP后积分/安装/放行不变。接触早停删失后续G7，报告episode级分母和查询暴露/阶段到达，未到达阶段不是条件失败。独立布局每臂至少6个事前固定配对（拟12卡，非授权）；若未获得或结果不足，状态INCOMPLETE，不称泛化。
| 结果 | 自动下一步（资源合法时） |
|---|---|
| FE真实资格失败 | 具体技术根因、CPU负例、在原合法预算内至多一次修正资格；无额度则阻塞且其他工作继续 |
| FE E1新增非法、阶段退化 | 不花原生名额，不加步；IO子模块诊断/一种函数保持候选；C1/A2继续准备 |
| FE E1/E2方向阳性 | 冷登记C1同scope两臂；FE完整保持未验证时标PROVISIONAL，R4仍基线 |
| C1保持且纠偏/抓持更好 | 进入A2固定所选bank；如有新合格学生，在原剩余名额内新TRAIN DAgger |
| C1保持但无收益 | 检查首接管曝光、有效多样性和A2缺信息；若相符择C2或A2，不两者一起 |
| C1前段改善后段仍不进展 | 阶段条件诊断，B2/B4关键过渡或A2择一，不能用R4失败释放作保持目标 |
| A2改善但蓝色仍发生全身下降 | 优先BASE/ARM独立候选与P0-WB可行性，不改硬安全 |
| S2/B2/B4或C2单项阳性 | 另登记单因素组合配对，完整原生评价与独立布局；无证据不组合 |
| 资源/权限阻塞 | 其他READY包继续，集中一次提交具体缺口；全部可做任务完成才停 |
## 5. 预算与无人值守授权边界（必须现场对账）
目前有效上限（如后继合法更新则使用现场新值）formal8000、engineering24、student50、Recovery20，已知5100/24/46/17。剩余2900/0/最多4/最多3；其中Recovery至少2保留后继学生、1灵活。原320GPUh和2026-12-05 12:41:52 SGT截止，实际remaining GPUh、Hopper quota、时间、站点owner需实读。计划不得自行将工程24升26/34或student50升118。仅用户明确的研究内部配额批准加真实集群项目award、站点权限及实际余额都具备时才改变本轮ledger。
- 核心正式预算：Frozen Expert250（严格匹配复用Dense）+ C1双250=500 + A2双250=500，总1250，条件完成累计6350，余1650。计划可选C2新增250（严格复用C1 Treatment；否则500）、另1–2项证据支持的S2或B2/B4双250；若核心1250+C2 250+两个500共2500，累计7600，余400作为拟议技术/确认缓冲。不得自动跑满；需要重复Control、失败实耗或额外资格则重新分配，保留相同配对，不能突破8000。优先省重复科学控制而非删原始证据。
- **一次性可选的有界后继资源提案（未授权）**：仅FE先+2工程=26；若用户明确希望整条主线无人值守，可提出分阶段FE2/C1≤4/A2≤4合计≤10工程（新内部cap34）；C1/A2额度仅在相应已登记依赖通过、实际资格需要时生效，不能挪给其他scope或技术重试超过各项上限。后续S2/C2/B候选资格若现有有效资格不能复用，另留BLOCKED_AUTH。新增资格实际用量按现场，不得预扣为已执行。
- **原生提案（未授权）**：4现有筛选槽若完全匹配可计入FE完整双20格，尚需16；C1完整20、A2完整20、独立确认12，共额外68（拟student cap118，实耗46+4+68=118）；若旧4筛选不符合完整合同则完整FE需20新卡，额外缺4，不在118内自动追加。阶段按阳性与合法资源释放；不为完成计划而运行失败候选。额外12×3×2 TRAIN前向72为预登记上界、原37候选和源42另按合法实际计算资源；不是工程优化或物理名额，不能靠换口径绕开其他约束。
- 无额外授权的默认无人值守范围：合法CPU reader、控制入口、固定权重/只读诊断及真实已获准现有资源内的工作；如果FE资格仍无法派发，先做全部独立任务和一次合并资源缺口报告（工程最低2、真实GPU/储存/确认名册），不能直接跳到未经资格正式训练或宣称长程完成。不要为普通代码修复重复索取批准。无合法待做工作时正常结束，不忙等，待用户或合法新执行者按最新账本续接。
## 6. 72小时到原截止的执行目标（不是实际耗时承诺）
T0=Codex真实接手时刻。按现场资源/作业状态推进，事件优先于固定时钟；所有任务有有限成本与终态。
| 时间窗 | 关键主线 | 并行可交付 |
|---|---|---|
| T0–6h | 现场对账、核授权、准备/条件执行FE E0；没有额度就记录BLOCKED_AUTH | C1组合reader、12状态/3噪声名册、A2/S2接口、C2分层候选、红蓝全身执行归因、L0真实批固定 |
| T+6–24h | 若FE E0合法通过，正式250及独立恢复；E1+源域；合格时预登记E2 | C1真实组批/恢复资格准备、部署A2因果检查、合法Teacher新前瞻准备 |
| T+24–48h | 条件FE原生方向/完整评价、C1双250与固定评价；资源不足不自动晋级 | 新策略TRAIN roster、合法纠偏触发、C2索引、实际阶段监督 |
| T+48–72h | 按C1结果择一A2或C2（默认A2），真实资格后有限训练/评价 | 有资格新学生的DAgger新数据冷封存，S2/L0/B2/B4独立准备 |
| 第4天至原截止 | 有证据选后继单变量/组合、完整自主PLACE+独立确认；条件新迭代 | 保持版本/失败、独立布局、所有账本与可复现交付；不自动扩大资源或延长期限 |
原截止前留足保存、独立回读、job/lease关闭；若达到目标先完成封存与实际确认，不继续消耗。若时间/授权不足，准确交付完成部分、哪些科学问题仍无法回答、确切资源缺口、下一个安全起点。**不中途只输出建议就结束，也不在全部合法任务完成后无限运行。**

## 7. 每波自动验收清单
- 原父/模型/代码/输入/noise/有效mask/scene/split/统计/训练步与资源已登记，实际哈希一致；控制组复用须逐项匹配而非同名。
- 实际工程/正式/学生/Recovery/GPUh、预留与未封存实耗、每个实际job和terminal，未启动不报实耗。
- 原生原安全与全块检查、接管前后时钟/序号/消费、首次STOP后零；保护集不进训练、调参或选步。完整成功/局部成功/教师辅助/科学失败/技术失败分别分母。
- 新冷数据逐文件、原始消费标签、reader、因果上下文、独立episode/布局和phase覆盖；真实访问≠计划访问。
- 结果与原V7.1数值门比较，给出PASS/NEGATIVE/MISSING/TECHNICAL、相对Control和R4分别、下一READY动作；保留所有负结果。每个波次提交实际报告与必要机器回执到执行分支，再GitHub远端回读核验；不得只改文档制造已运行假象。
- 最终交付：现役/历史代码SHA、所有端点/数据/评估回执、累计预算、明确达成/未达成的E层级、进程/owner/lease收尾、下一冷启动入口。禁止自动merge main、force push、修改权限/密钥、删除用户资产、热改正在运行capsule。本次仅计划与prompt推送。
