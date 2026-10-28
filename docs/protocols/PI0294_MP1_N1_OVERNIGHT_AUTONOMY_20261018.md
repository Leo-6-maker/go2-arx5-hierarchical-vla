# MP1-N1：夜间无人值守工程与科研接续协议

日期：2026-10-18，Asia/Singapore。
Decision ID：MP1_N1_OVERNIGHT_ENGINEERING_AUTONOMY_V1。
状态：GO_BY_CURRENT_USER_REQUEST_PENDING_COMPLETE_BUNDLE_AND_SINGLE_OWNER_BINDING。
授权依据：用户本次明确要求“仔细审核目前试验进度，给出夜间无人值守长程计划，给足工程预算，自主修复”。本次是有界资源与工程自治的新修订，不是再次要求用户逐项批准的建议稿；它不能创造实际站点/账户权限。

## 0. 唯一入口、效力和交接

仓库：Leo-6-maker/go2-arx5-hierarchical-vla；分支：pi0267/executor-decoupled-recovery。
科研与原生证据基线：403704527d876bf36b7c122d0f59384a4693c19d，#11/6238428190，2026-10-18 02:16:51 SGT。
发布前补读工程增量：e418bb790d291c6c936ce3878366314f4f5af83e，已增加第五次冷态准入；这是代码证据，不是第五次已执行证据。发布者未 SSH、未独立复跑、未提交 PBS。
唯一父资源族仍为 MP1，原 authority c8cdf60e41e31da5796b44a064ef2f22d594a491、parent run 20261016T191445Z-localpc-mp1、原 MP1 T0=2026-10-17 03:14:45 SGT；source-pair-01 的科学身份、累计费用、样本游标和尝试记录不重置。
N1 是此父族的授权修订和有界自治窗口，不是第二个 owner、独立可叠加的资金池或 E2 goal。

完整发布包必须在同一固定可达提交中具有以下三份文件，owner 实际回读后才采用：
- 本协议；
- reports/PI0294/design/MP1_N1_OVERNIGHT_AUTONOMY_20261018.json（可机读上限与停止条件）；
- docs/handover/2026-10-18/MP1_N1_OVERNIGHT_EXECUTE.md（接手入口）。

优先级：真实已经发生的结果/费用/安全事件不可改写；本修订只替换明确列出的未来资源和工程调度限制；其余科学合同沿用原 MP1、fb16d101 审核以及 GR00T_TASK_DATA_FIRST_FROZEN_WBC_V1。旧额外第五槽卡被本修订纳入同一 source 上限，不是再加一份额度。若第五作业在发布期间已经入队/运行，保留原绑定至合法边界并计入同一累计账；不得重复提交第五份。

## 1. 审核结果：科研已经前进，近期主要消耗在执行可靠性

### 1.1 应保留的科学成果
S-BC@2048 全状态封存，实际完成 2048 更新/65536 访问，T1/T2 各32768；共享同一模型的 TRAIN4 为3成功/1有效失败，暴露DEV8为3成功/5有效失败。DEV T1=2/4、T2=1/4；四条预执行S1拒绝与一条安全完成32s但目标区保持失败要分开。DEV是开发集而非独立确认；源域为固定底座ARX5，不是Go2+ARX5 WBC或真机成功。
S-CORR@512 已真实封存，和BC同一原M1初始化/配方，实际16384访问中8192 T2与4096保留原T1逐游标一致；4096纠正T1来自4学生parent/8段。当前固定512的96专家窗配对没有显示XYZ均值改善，不据此判定终点纠正无效。不要再重训已完成的BC或重开旧M2恢复。

### 1.2 最新原生边界已经不同于旧Q快照
source04/672907 已F/1，381原生墙钟秒、391保守H200秒；source累计24119秒、父MP1累计28619秒、当前GPU预留0。Source已用4次PBS，父Hopper7次；CORR仍512/游标16384，组合更新2560，pending_updates=0。
CORR512 F0T1/F0T2是有效原S1负例；F1T1两次工程partial已耗尽原replacement，不能第三次补跑；F1T2新增8.4s工程partial，尚有原合同唯一replacement。矩阵21attempts/18有效policy cells/2工程cells/12未尝试cells，共32条件。最终允许以明确不可补缺口收口，不要求把32格全部变成有效成功/失败才准许推进训练。
Source04的21个已执行ACK、22个生成、一个生成未安装动作和1970个mutation intents已闭合。旧STOP字节、原件、视频、消费与失败都保持。

### 1.3 已定位且须优先验证的工程问题
最近心跳年龄为-0.0678687096秒，并非真正超过300秒：调用先采时钟，再读取文件；文件读取期间owner写入较新心跳，旧时钟减新心跳产生假未来。07b04707已改为每次读取之后采时钟，包括强制重读，保持0<=age<=300；406CPU测试/CI通过，但最新原生持续资格尚未证明。真正跨主机时钟偏差是另一问题，不能用abs(age)、夹到0或放宽负值容忍来掩盖。
02/03还出现短暂SSH/元数据同步导致owner退出、compute随后按watchdog停机。NFS、锁等待、传输、clock race必须按已实测机制区分；不宣称一个补丁解释全部历史中断。
源码审核还发现旧session_spec仅接受01..04。发布期间e418bb79已联动第五次准入和部分身份检查；N1扩至12后，必须统一验证所有request/launcher/label/ColdGuard/矩阵解析/registry/计数路径，不得只修改一个JSON或全局删除assert。
现有neural调度在续跑时仍会走已结束BC面板的load/evaluate路径。允许在加载前依据固定面板与原生已闭合记录跳过零剩余工作；不能把尚未完成的probe目录因exists就当完成。

## 2. 目标与成功顺序

首要交付：完成既有同父、同配方的S-BC2048/S-CORR2048终点对照，获得完整训练消费证明、逐任务/逐布局结果，以及对工程缺失的诚实处理。
第二交付：把心跳/中继从“单次smoke通过”变成实际长程稳定测量；日常工程错误自主定位、测试、发布、冷态部署、续接，不等待逐次GO。
第三交付：用终点和已保存锚点证据选择最多一种后继学习干预，并在有候选时进行原独立确认。
第四交付：在不阻塞源域主实验、不更改WBC的前提下推进Vanda的连续原生资格。禁止用更多保持RMS/控制器调参取代任务数据和闭环。

## 3. 资源修订：累计新上限，不是把旧余额清零

下表的“新上限”均包含此前MP1实际使用、当前预留、第五次单槽授权和以后N1工作；不是原上限加整列新上限。站点/用户项目余额、实际quota、已批准并发与更早截止取更小值。

| 资源 | 旧上限 | 新累计上限 |
|---|---:|---:|
| MP1 H200 allocated GPU-hours | 48 | 80（新增32） |
| MP1 A40 allocated GPU-hours | 32 | 48（新增16） |
| MP1 Hopper GPU PBS次数 | 12 | 24 |
| MP1 Vanda GPU PBS次数 | 12 | 20 |
| MP1 Vanda CPU build提交 | 8 | 16 |
| source-pair-01 H200小时 | 12 | 36 |
| source-pair-01 PBS次数 | 原4，后单槽修订5 | 12（第五槽计在内） |
| source-pair-01累计CPU core-hours，Core/Hopper | 60/48 | 180/144 |
| source-pair-01实际模型生成上限 | 4000 | 12000 |
| MP1 CPU core-hours，Core/Vanda/Hopper/local-PC | 320/240/160/24 | 640/480/320/48 |
| MP1累计网络传输GiB | 500 | 750 |

并发保持1张H200、1张A40；Core/Vanda/Hopper每站最多16活跃CPU核、local最多4；本机站点实际申请更小者优先。Source新分配默认请求最多8小时、站点只准4小时则使用4小时；每份墙钟不得穿过N1/MP1截止，保留原2700秒allocation收尾，不改20/60/900秒RPC/action/episode约束。新的更长分配必须通过版本化准入，不热改已经运行的4小时job。
父存储峰值仍沿原Hopper400GiB/Vanda250GiB/Core200GiB/local20GiB；source软输出估计180GiB可据实际checkpoint计算，不是强制预先分配。原个人quota不足不能靠修改估计伪造余量。

不扩展：source主比较4096总更新（剩1536，不是从0再训）、原32条件与每条件至多一次工程replacement、source主启动64、live/recovery与shadow各2048000积分；父24576总更新、source400/target320启动和64000000总积分保持。新增工程fixture/缺失原probe/后继诊断生成须独立绑定并计入新的12000或适用父子账；不能用多query给正式条件安全重采样。

资源保护：未完成CORR终点与12项终点评价前，source剩余额度中保留至少6H200小时给最终封存/评价；父最终验证/收尾保留6H200小时和4A40小时，source保留是父保留的子集而非另计。source余额不够此门则不启动可选新模型。所有allocated idle、失败、测试分配均收费；queued未分配不算GPU实际用量但保留请求reservation。PBS确定拒绝也计提交attempt；返回未知先原生查重，不盲qsub。

旧birth card/原manifest只读保存。实现一个显式有效授权解析器：原身份+本N1固定card hash+本次ADOPTION_DELTA，返回修订后的有效上限；没有N1时旧规则逐字保持，有N1时仅上述字段变化。对子stage、RLIMIT_CPU、launcher和ledger一起测试，禁止只改文档却让旧硬编码48小时/4次/01..04在夜间再次拦截。所有权限完整性校验仍硬失败。

## 4. 无人值守窗口与调度

既有单一owner首次采用完整N1包时记录 autonomy_started_at，最长36小时；新提交/新候选截止为采用+30小时，新实验/训练block截止+34小时，+36小时硬收尾。各时间均不晚于原MP1的10月23日03:14:45、10月23日19:14:45、10月24日03:14:45 SGT。断连、重启和换allocation不重置任何时钟。
本窗口执行器在10月18日10:00 SGT写一次晨报（采用晚于此点则立即补一份，不追溯伪造），此后每最多4个活跃小时或出现重大结果/硬阻断写已有#11摘要；#47只记录新工程证据。该要求是Codex已启动执行器的职责，不是本聊天已经创建的后台任务。

同一owner控制调度。CPU诊断、数据准备和Vanda资格可以在隔离lease内并行；一个registry mutation publisher，不允许两个manager抢同一source、同一条件或同一GPU。固定期内有限supervisor可以在用户许可的既有节点/调度器运行；不得在login节点跑模型/物理、创建永久cron/daemon，或在丢失原owner真实性后伪造心跳。
每次状态转换持久记录intention -> native receipt -> observed effect -> reconciliation。原生job/process身份用jobID及进程起始身份，不单看PID；旧owner实际终态及pending effects闭合后才能原子交接/fencing。健康job保持到正常边界，不因发布新方案而杀掉重来。

## 5. 夜间实际执行顺序

### N0：一次采用与原生对账
读取AGENTS、START_HERE、统一入口与ENTRY_POLICY、原MP1/card、最新#11/#47。刷新Hopper/Core/Vanda账户、原生job/owner/lease、pending draw/install/optimizer、存储quota、已用和预留。若第五次已在途，将其并入本账，先维持健康运行，不新建同号作业。
发布N1_ADOPTION_AND_EFFECTIVE_LIMITS.json：完整包固定SHA、全部原identity、修订字段、原累计费用与pending、自治绝对截止、可以立即开始的工作。旧401/403/配额不明不能自动当有权限；凭据仅用已授权凭据管理和刷新流程，禁止猜测密码、绕MFA、把密钥写Git/日志。无法无交互认证时转向本地合格CPU任务并报告一次阻断，不反复向睡眠用户询问每个工程细节。

### N1：修复心跳、资源解析和调度，再做持续原生资格
先复用07b04707和e418bb79的已定位补丁，避免重新发明NFS helper。版本化扩充授权解析，不再为06、07分别写硬编码cold_boundary文件；同一通用实现验证连续原生终态链、已用资源、旧STOP hash和新的epoch/fence。测试范围包括合法01..12、越界13、未经N1的06、别run、超额、并发重复owner、重复settlement、跨模型poison和禁止复跑有效条件。
在新合法allocation加载模型之前，至少600秒持续读取真实compute看到的owner心跳，覆盖并发replace和强制重读，记录每次读取后时钟、heartbeat值、host与clock-source。单次初始9秒年龄PASS不足以建立资格。记录真实跨机clock偏差；不改系统时钟、不abs、不截断负值、不放宽0..300。额外负例在独立fixture namespace验证真过期/真未来/错误run/owner退出会停止，不能给正在训练的真实heartbeat注入故障。
运输fixture使用真实生产编码/文件尺寸和接近真实RGB负载，至少80个有限请求/应答cycle，注入延迟回复/元数据读失败/被占用统计锁/进程退出后验证：无重复install、无错query消费、无未知effect重发。fixture不是神经推理成功或任务结果。原20秒限制继续，单纯把超时变长不是修复。
统计遥测/上传/视频编码/整树hash不放在每个动作回复的同步关键路径；生成、安装、物理计数和mutation journal仍立即持久。只在固定边界复用已有核验，不跳过必需身份校验。有限幂等metadata读/确定可重入counter采用最多3次/120秒重试，权限、数据完整性、动作副作用不套此策略。
同一未定位故障连续两次原生复现时不再直接启动第三次同配方任务；回到CPU/真实传输诊断，必须有新根因证据与回归测试才允许下一版本。每个组件最多6个实质native修复版本；达上限仅阻断该lane，其他合格任务继续，所有费用照计。
交付ENGINEERING_ENDURANCE_AND_FAULT_INJECTION.json，分别写CPU_PASS、NATIVE_LONG_READ_PASS、REAL_MODEL_LOOP_PASS，不以测试数量代替实际效果。

### N2：完成既有CORR，不让中间面板吞掉整晚
冷态采用已验证工程代码，不动学习/数据/物理字节。模型加载前计算剩余工作集：所有BC面板已经终态，零剩余则直接跳过，不重复load/probe/serve；@512已耗尽F1T1保留工程缺失，不循环等待第三次；F1T2只允许原唯一replacement，成功/有效失败/再次工程耗尽均终态登记后继续原正式训练。安全STOP与未知副作用仍阻断，不能跳过。
对原probe逐draw核对journal与arrays：完整原draw复用；确定NOT_RUN且原manifest允许的缺项可以在原限额补齐；未知生成不重发；目录存在或半个ON_EXPERT_RESULTS不是完整证据。
从有效完整CORR512恢复optimizer/scheduler/RNG/sampler，继续到2048（总组合4096）。训练到allocation边界可保存完整partial并按新累计上限自动续接；不重开optimizer、不重复已提交样本、不根据512负例临时加训到4096/8192。若最新原生状态已经超过512，直接使用最新合格完整状态，不能倒退重训。
@2048固定原96专家+12历史学生probe、TRAIN4、暴露DEV8执行；Teacher OFF、相同D0/EEF10/H40/prefix8/噪声与数值、原评分与S1。原每条件一次工程replacement不变。所有有效结果不重跑，不能用预算换best-of-seeds。
最终完成65536实际CORR访问核账，与BC逐游标对齐T2、保留T1及纠正T1；核验同一M1初始化、全部正式更新和冻结参数。固定矩阵32cells允许一格以上不可补工程缺失，但必须所有格有精确终局/未执行原因；原终点比较24cells的覆盖与中期512缺口分别报告。

### N3：有限的同状态诊断和鲁棒性决策
优先复用现有8个S-BC512失败锚点（4parent，非8独立episode）、静态恢复证据与实际噪声receipt。缺失终点draw最多16（BC2048和CORR2048各8），总三模型24；只有两终点封存且输入/动态前缀恢复资格通过后在单独NI/shadow子卡执行。专家/XYZ-only/SO3-only/gripper-only仅作通道归因，不成为自主成功或未验证正标签。恢复与shadow真实积分全部收费。
当前BC2048的F1接近失败、T2安全完成但目标未达是不同机制。DEV仅作冻结开发评价/描述，禁止从DEV生成教师标签或把DEV状态加入训练。后继数据只从预先登记TRAIN场景和当前策略访问状态采集，目标域数据准入不改原判据。
两臂终点结果和锚点证据出来后，owner无需再问GO，按原MP1选择最多一种改进族：
- 专家态关键事件误差和实际曝光与失败一致：原S-EVENT一次、2048更新，原M1父权重与其他配方固定，任务内50%uniform+50%event-neighborhood，事件定义与窗口在训练前封存。
- 专家态合理而当前学生态恢复差：原MP1合格Recovery/DAgger，最多原总2轮，每轮<=12个collection episodes/每任务<=6，每轮同父BC512/CORR512；T2必须先具有完整安全恢复Oracle，短prefix通过不够。实际episode/parent去重，失败动作不当正标签，C1旧T1来源不得冒称当前T2纠正。
- 两类证据都不足：不自动新训；交付机制未决。单独4条T2/512容量诊断只在原触发条件满足且尚有资源时选择，视为本窗口唯一可选学习分支，不与上述同时开。
原总更新24576与物理上限不增加，不启动RL、FlowDAgger、新backbone、全VLM解冻、无真实BASE标签的FB13或损失大改。

### N4：独立确认与Vanda原生lane
共享候选按原worst-task -> macro-success -> 低复杂性从固定开发结果选择；预先冻结SOURCE_CONFIRM_V1的10scene-group x2task x2noise=40条件。每任务14/20仅为原研发目标，不是部署可靠性证明；同scene两seed不当独立场景。一次确认不回流训练，未达到候选条件不为刷里程碑强开TEST。
Vanda使用独立最多1A40；刷新实际队列/资格，不重开旧过期camera lease。优先把新合法窗口定成贯穿排队与运行的合理有界资格卡，避免15分钟测试却配一个过短排队TTL造成始终NOT_RUN；请求、lease与N1绝对截止在提交前封存，不追溯延长旧job。
先现有SIF/环境/双摄EGL与指定模型数值输入，再原生WBC启动/时钟/合法参考/ACK/新观测，最后任务pilot。几何保持RMS不是全局前置，但TF-R3真实安全STOP仍有效；只有原允许的有证据任务接口/时钟/启动修复及原版本次数内的冷态重新资格可推进。若同一未解决真实安全原因再次出现，封存目标lane并继续源研究，不能换host规避。
不修改师兄WBC权重、代码、增益、cost、solver、腿策略、plant、夹爪PD/力限；不把fixed-base成功称整机成功。原样WBC真正安全完成任务后，才可准入target示范；本夜不以控制器归因工程取代GR00T任务主线。

## 6. 自动修复与存储整理的明确权限

允许在已登记工作区内自主实现、测试、普通提交、冷态部署：路径/解释器/bootstrap、PBS请求与节点资源字段、quota预检、checkpoint原子落盘与完整resume、SSH/文件传输/元数据、时钟采样、身份绑定、可重入调度、审计publisher、模型服务与仿真反馈接口的同语义修复。所有科学参数与消费者哈希变化必须声明并审计；不能仅因为叫engineering就改变动作语义。
测试和小收据按一个预先注册的工程子scope合并，不为每个fixture另向用户申请；同一owner内部最多2个只读CPU辅助者，不新建科研调度owner。正式32cell raw审计可以单一serial worker流水收集；本地合格数据消费不等待HF上传，视频公开可延后到GPU空闲/终态后集中发布。
用户已有失败checkpoint清理授权可继续执行：仅删除已经明确无活动消费者、无resume/对照/基线价值的本项目失败临时payload或可重建cache，不要求再为每候选询问或重复大备份；必须保留轻量manifest/原因/删除清单。M1父权重、BC512/2048、CORR512/2048与最新完整resume、原始数据、唯一失败证据、WBC/环境/他人文件不得删除。只有新检查点独立完整校验后，才可退役不再引用的冗余partial大payload；保留审计witness。没有可删除对象就写freed=0，不能把全quota已使用解释为本项目可删。
同一queued本人job的安全内存/节点字段修改允许，先查Q且无执行、单意图qalter、回读并记账；未知返回先查询。同scope确定无副作用的拒绝可修后再提交，已运行健康job不得为抢卡取消。禁止账号/项目权限绕过、秘密外传、外部付费服务或新云实例购买。

## 7. 硬停止与从错误中恢复

真实native安全、错误身份/输入/统计、权限失效、不可证明的pending mutation、重复owner、资金/时间/个人quota上限仍立即阻断受影响动作。S1预测拒绝是有效策略负例，不是绕过理由，也不自动是整个工程系统故障。
工程STOP不可删除。修复后在旧job/process全终态、effects闭合、旧STOP hash保留和新工程版本合格条件下，版本化解除其对未来工程epoch的阻断；不是改写原结果或解除TF-R3。owner未知不得自发抢锁或伪造新心跳；以确定性fencing交接后方可恢复。
N1资源修订只给更多机会，不给无限同错误重试。只要某lane被阻断，自动切到其他已经合格且无依赖的只读诊断/数据准备；所有lane均阻断或截止/上限达到时写终局并停止，不能沉默等待睡眠用户批准常规操作。

## 8. 必交产物与结论边界

在当前MP1 run报告根发布并交叉链接，不新建竞争report系统：
1. N1_ADOPTION_AND_EFFECTIVE_LIMITS.json：累计费用、适用新旧授权、绝对截止、代码部署/运行身份、原生owner/job快照。
2. ENGINEERING_ENDURANCE_AND_FAULT_INJECTION.json：真实compute持续心跳/运输与故障注入，代码测试和native效果分开。
3. MATCHED_SOURCE_PAIR_FINAL.json + BEHAVIOR_AND_FAILURE_MATRIX.json：完整两臂更新/样本/parent/32cell结果与所有attempt；不可补缺失保留。
4. N1_MECHANISM_AND_NEXT_BRANCH.md：失败按布局/阶段/S1/目标未达/工程缺失分类；选一方法或明确不选及原因。
5. N1_MORNING_REPORT.md：10月18日10:00的成果、实际运行、余量、阻断和下一自动动作；不得只报代码数量。
6. N1_RESOURCE_FINAL.json + N1_FINAL_SCIENCE_REVIEW.md：所有native终态、累计与增量资源、未知effects、已选/未选候选、SOURCE_CONFIRM与目标WBC实际资格、可点击固定证据。

版本有代码、CI通过、已部署、native启动、真实训练、完整任务成功、独立确认分别记状态。未到最后层不得升级措辞。零实际安全故障不意味着模型无需安全门，已见双任务成功不意味着成熟或真机可部署。
本次发布只有文档与机读卡，不是已经启动夜间执行器；必须由既有Codex owner回读完整包并记录实际采用后执行。后续不需要为本协议范围内每个修复、合格续跑或已规定条件实验重复获得常规GO。

## 9. 固定证据索引

以下原文件保持原始字节；优先按本协议首部的审核提交读取：
- reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE04_ACTUAL_TERMINAL_AND_CLOCK_REPAIR.json
- reports/PI0294/design/SOURCE_PAIR_ADDITIONAL_PBS05_AUTHORIZATION_20261018.json
- reports/PI0294/20261016T191445Z-localpc-mp1/S_BC2048_COMPLETE_DEV8_AND_CORR_INDEPENDENT_START.json
- reports/PI0294/20261016T191445Z-localpc-mp1/S_CORR512_ACTUAL_MATCHED_CONSUMPTION_AND_FIRST_VALID_NEGATIVE.json
- reports/PI0294/20261016T191445Z-localpc-mp1/SAVED_BC512_CORR512_PROBE_COMPARISON_AND_STUDENT_IDENTITIES.json
- scripts/pi0294/mp1/owner_heartbeat.py、plans.py、science.py、neural.py，以及07b04707/e418bb79的版本化工程diff。

本包没有取消健康作业、启动新模型/物理、删除数据或授予控制器修改权。当前selected/deployment沿最新原生事实记录，不能仅因计划发布设置为PASS。
