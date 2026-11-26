# SLA Hopper ARX5-rich Source Expansion V6：扩大公共源域、自动接力与本体联合适配

2026-11-26（SGT）。用户同意先加入一批ARX5公共数据，以较大规模源域知识扩充、随后少量Go2本体联合适配。**本PR为执行计划与数据登记；尚无新增下载、reader准入、GPU优化更新、原生评价或PBS提交。**

配套：[执行手册](SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6_RUNBOOK_20261126.md) · [机器登记](SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6_REGISTRY_20261126.json)。继承 [V4 #519][v4]、[Core数据工厂V5 #521][v5]；现役V4完整科学配方与旧结果不变。原WBC/OCS2/PD/腿策略、物理2ms、原任务安全与评分冻结；公开固定臂数据用于源域监督，不能直接冒充移动本体数据。普通工程缺陷自主修复，有限工作包自动接续；实际授权和资源先行，不无限重试，不另建调度平台。

## 1. 最新事实与目标

实际回读#520最新可见commit `d175383b07cabb701492222bc333809d57ca5830`（17:09现场564/1000，非当前实时SSH），H200正式job735204，热态5.087s/步、p95 5.106s，250/500完整中间点。三域FM 2.360/1.201/1.148s，aux0.329s；64/32/32/4整批已实际训练。R0已部署，原1K正常终态后从STEP0250恢复27个工程更新（3 warm+24 trace），不进入科学谱系。未读到1K完整终态或2K后继提交，不能假定自动接力已生效。[现场回执][actual]

原MAIN3.8K完整三域端点继承2500祖先更新，实际目标243200、RoboDojo121600、LIBERO121600访问；源TRAIN为RoboDojo200条/210398窗、LIBERO1261条/209146窗。目标当前52条/4450窗、4294完整支持H40，69条旧候选仍需新合同准入。已有MAIN3.8K移动HOLD阳性；MAIN_PLACE250的6张回归/PLACE在抓持前拒绝，不能以loss拟合替代行为。Core V5持续生产Adapt100/200+Recovery40，不等源域数据或Hopper端点。[源账本][ledger] [Core报告][core]

**工作假说：** 单臂ARX5源数据缩小单双臂及末端/夹爪语义差距；共享SLA从多源学习通用操作，Go2实际命令与移动视觉/本体经验负责目标适配。不是已证明的技能迁移。新增源域阶段保留少量Go2 replay，避免纯源域长训覆盖既有移动能力。源域单独适配不重新预训练整个SLA，不需要训练腿/底座低层；显式BASE动作不在本轮主线。

## 2. 数据源：先小批、再500、再2000，按实际许可与兼容性推进

本次重新读公开数据卡/官方文档和旧Table30审计。下面的数字都是去重后的**TRAIN准入目标**，不含留出，不是可下载或实际消费数量。200条既有RoboDojo计入400目标，新增ARX5理论1800；LIBERO1261另计，不凑满才开始。

| 来源 | 已读事实、优势 | 首轮风险/行动 | ARX5-Mix2K目标 |
| --- | --- | --- | ---: |
| RoboDojo-Real ARX X5 | 已有200条实际训练；同源6任务；公开库标Apache-2.0，官方曾修正RGB顺序/帧差 | 复用实际已封存reader、统计和旧成员；按实际real/arx_x5路径增量，确认不是simulation/leaderboard eval；完整新revision不混旧 | 400 |
| Table30 / Table30v2 | 官方单臂ARX5与多个本体混合；旧v1 30任务元数据实际11项ARX5，v2需另读。v1 XYZ+RPY、v2 XYZ+xyzw；宽度m，独立相机 | 按真实robot_id和episode metadata选；具体数据许可未核实，不因公开网页推定许可。旧同步/collector来源缺口保留，见§3 | 500（两版合计） |
| OpenNeoData arx5_single | 数据卡全库200k+轨迹，LeRobot3、30Hz、单臂7D关节/10D EEF，明确区分实测与命令 | gated、CC BY-NC-SA 4.0；当前访问未证实，核使用/派生/共享条款与真实账号授权，不能自动接受条款或绕过gate；只取单臂meta、RGB及数据文件 | 500 |
| RoboTwin 2.0 ARX-X5 | 社区Traly转换卡报24500 ARX双臂episode、全库约20GB，native 30Hz、next-frame action_ee xyzw | 核验原始版本/转换许可与数据来源，优先demo_clean实际manifest，原官方新默认下载是ALOHA而非ARX，不能使用默认全量脚本替代筛选 | 400 |
| Ville ARX5 | 单臂真实双RGB；pravsels汇总卡1704轨迹/860076帧，1TB+，含eval/fail/DAgger | 优先原作者LeRobot2.1具体小库，而非巨型HDF5聚合；逐库许可、实际action/state、FK及成功身份；仅正常训练正例，纠正另表 | 200 |
| LIBERO | 既有1261 TRAIN及独立H10 OSC合同 | 原独立stats/reader保留、固定源留出 | 不计入2000 |

来源：[RoboDojo][dojo] [官方代码][dojo_code] [Table30][table] [Table30v2][table2] [OpenNeoData][neo_readme] [RoboTwin转换][twin] [官方采集文档][twin_doc] [Ville原作者][ville] [汇总卡][ville_agg]。[SLA官方Memory][official]用于源语义参考，不将官方PT权重当当前MAIN父。

### 2.1 首批执行与资源排序

- P0并行：已有RoboDojo清单回读与追加真实未见episode，Table30旧小样本/许可核对，OpenNeoData只读可见元数据/授权状态，RoboTwin社区ARX目录与原manifest、Ville原始小库；不等待gated源解锁。每候选族最多8条真实pilot（每可访问任务至少2条、优先接近抓放/搬运/释放/Push/Drawer/Insert；Table30两个版本合计8，既有可验证pilot复用）；五族最多40条新/复用pilot，失败保留。CPU工作在获准的Core/Hopper存储与计算环境，不在AMD共享登录盘做大量解码/扫盘。Core原生生产优先，Vanda保持暂停。
- ARX5-Mix500目标：已有200 + RoboDojo新增100 + RoboTwin100 + 合格单臂源100（OpenNeoData优先于未知command的代理；Table30/Ville哪项先READY先用，单臂总100可跨源）。如果合法来源数不够，不虚构500；第一个可训练封存允许已有200＋至少100条真正新增TRAIN、覆盖至少3个技能/任务组，且至少1个新来源/新增有效episode群；首选至少一个单臂源。若只有新增双臂可用，允许明确dual-only新课程并保留LIBERO/Go2，不阻断健康任务；不能称已单臂扩充。完全没有新准入则不启动V6 source扩充，转现有V4/目标数据工作。
- Mix500后的目标是Mix2K。正常份额400/500/500/400/200是目标而非硬门；某源不具备许可/语义/存储时标BLOCKED，优先同组其他合法来源。任何配额再分配要记录实际数量/理由，不跨训练边界暗换source。Table30先查已确认的open_the_drawer、put_cup_on_coaster，其余必须按真实robot_id确认；RoboTwin优先pick_diverse_bottles、place_object_basket、place_a2b_left/right、move_can_pot等实际存在且数据合格的任务；不把双臂合作动作删左臂当单臂。Ville可选原始pick_coffee_prop_20Hz、drop_footbag_into_dice_tower、rice-cooker_step系列，先核用途/成功，不将eval/fail/dAgger_test混入教师。
- 旧TRAIN/VAL/TEST保持。新源在读模型效果前固定完整episode/session/scene分组，建议每任务约80/10/10 train/dev/test（小任务至少2个独立留出组才声称有留出；否则标TRAIN-only或暂缓外推），目标训练配额不含留出。原作者→镜像→转换按episode来源、轨迹/视频哈希去重，跨版本/数据源的同一轨迹归一个split。原MAIN已见数据不冒充新留出。保存失败/未知标签及实际不准入数。
- 数据量约束：先metadata/API树/大小，不clone整个仓库或snapshot全库。pilot所有新源总在途下载/解码暂存预算初始10GiB（资源不足更低），Mix500初始40GiB新增工作区上限（不是保证足够）；每个文件下载前实际size/partial/extraction/compact及两个新完整训练状态与现有安全余量综合核对。既有完整历史文件不删除。历史Hopper约150GB余量不是实时容量。大HDF5逐episode按有权限路径读取，优先官方压缩视频/小分片、复用已缓存校验；OpenNeoData按meta/episodes指向的真实文件取parquet+third/wrist视频，不取tactile或整repo。共享文件覆盖多个episode时按真实下载总字节计，不能以抽样帧数估算下载体积。允许在已授权数据盘分波暂存、无损/原MP4保存、内容寻址；不主动清理用户数据或使用未批准挂载。每波预留明确字节；不足则缩小本波/报告用户，仍可进行其他就绪源。仅真实READY不可变版本进入Hopper。

## 3. 不同来源的动作合同与最小准入

原则：**统一任务空间语义，不强行统一原始数组。** 每个数据源/版本独立domain codec、时域、mask、stats；原有Go2和两源分支参数、codec、统计不变。新增source IO只在实际形状和物理语义相同才初始化兼容叶；否则确定性新初始化、显式mapping，不能将旧Go2 IO改写成新源归一化。共享模型、视觉输入处理和动作专家按已验证实现复用。源域不能直接作为native命令或安全通过证据。

| 域 | 观测 / 动作来源 | 时域/有效维（候选，以真实reader绑定） |
| --- | --- | --- |
| Go2 | 当前真实body/wrist/EEF/grip；实际消费COMMAND重建，上一实际夹爪命令 | 原H40×32，50ms，7个有效动作维，真实时间×维mask；四个完整支持aux |
| RoboDojo | 旧actual reader的双臂EEF，保留源nominal/future语义 | 原H50×32；旧实际30Hz及有效双臂维、独立统计 |
| LIBERO | 原OSC | 原H10×32，原10Hz及有效维 |
| Table30 v1/v2 | states记录的base-frame EEF和gripper_width；未来记录位姿序列作为源参考，非Go2 command | 逐episode实际fps/timestamps，候选H50@30Hz、单臂7维；v1 RPY→R、v2 xyzw→R，必要TCP标定，v1/v2各独立合同 |
| OpenNeoData single | observation.eef_pose实测，action.eef_pose为文档所称命令；rot6d列顺序需显式核对 | 候选H50@30Hz、单臂7维，输入third/wrist，tactile不进模型，不把命令当实测 |
| RoboTwin | 实际state_ee与action_ee下一帧位姿，卡声明xyzw、grip 1=open；双臂数据 | 候选H50@30Hz、双臂有效14维，真实camera masks，不因16D padded joints判成单臂 |
| Ville | 原关节command/observations；验证6关节+grip映射，实际FK/TCP生成EEF | 以真实20Hz等采样率，候选H40@20Hz；单臂7维，无法证明命令仅使用有明确来源的记录参考 |

实际时间以原timestamp/PTS/episode offsets为准，不强改源clock。每个窗口anchor为当前实测/明确记录观测，动作位置delta与rotvec按真实固定frame、约定乘法侧转换；旋转检查正交/det、quat顺序与连续性、TCP/法兰/基座转换、FK数值一致性。夹爪分清全间隙/每侧滑块/角度/归一化，开闭方向与实际反馈独立；源不明的夹爪保留MASK或拒绝该样本，不直接套Go2 2–42mm、logit或sigmoid两次。新源归一化只用TRAIN，保存q01/q99或经实际原domain定义确认的stats及有效支持，不裁剪高分位；数据零填充只作mask，不把缺臂/缺BASE当零标签。缺相机使用原Processor原生image mask，不复制当前/未来图像当历史。

**Table30特例：** [旧审计][tableaudit]中v1 8条5837状态行FK最大XYZ差约0.647mm、旧11个ARX任务、1条双视频PTS与行时间关系，仍不能证明原writer反馈来源或快门同步；旧SOURCE_NOT_ADMITTED保留。v1 README timestamp称毫秒而审计样本实测秒，逐轨核验单位/顺序。优先取得版本关联的原始collector/维护者说明；若只具备官方逐帧记录语义、实测文件对齐和合理FK/时间，但缺原快门同步证明，可明确登记独立 **SOURCE_RECORDED_REFERENCE_WEAK_PROVENANCE** 新domain：只训练源记录未来参考，不称实测反馈/实际command；首批仅≤100条、按来源分项报告，不能代替旧#375严格准入或Go2真实消费。前提是具体许可已确认，真实数据时间/位姿/语言语义足够解释；无法排除错配、单位/命令语义矛盾则继续BLOCKED。后续从可信证据修订新版本，不将旧拒绝改PASS。优先更清楚的OpenNeoData命令型数据。

**OpenNeoData** 的rot6d是R前两列按列展开，不直接塞入Go2 rot6d；读取info/features及实际每episode文件时偏，按真实RGB key选择third_view/wrist，禁止下载全部9TB/其它embodiment。CC BY-NC-SA的使用范围、署名和衍生共享须确认，未确认不纳入可共享模型。**Ville** 不靠汇总文件名推断成功，读原库元数据、控制语义和FK链；DAgger只有明确Teacher correction段才另入源纠正池。**RoboTwin** 社区版与官方原始是同源，不重复入库；原默认下载ALOHA不能充ARX。**RoboDojo** 旧200版本保持，新增同一原始episode更新版本按原身份去重，若RGB/state修复导致不兼容则另domain版本不静默混合。

### 3.1 实际测试（以最小可用为先）

每个新domain真实8个固定window（含接近/接触/抓持后/结束）做state/动作、真实RGB、原Processor后张量检查、维/时间mask、FK/回转、实际stats、相机顺序；8个不是训练成功分母。检查mask掉的标签改变不影响noisy input/有效loss/grad；拒绝跨episode、STOP/gap、无支持后缀、末点重复。两臂需要真实两臂监督；单臂padding不产生虚拟动作。CPU读出样本、固定访问银行、实际resize/token/shape；H200在旧job冷边界从选定父真实初始化新domain，至少每个新domain有2次真实梯度/优化覆盖，整配方6步预热/吞吐短测（2warm+4timed），冷恢复+下一固定真实步/推理验证；若覆盖不足按新domain轮转补齐，本批最多24工程优化更新，均独立不进科学起点。冻结叶不变、有限loss/grad、实际label/mask/weight、global clip一次Adam、完整恢复。保留旧严格数值等价结果作为诊断，不再因正常有限batch浮点差阻断新profile；如有数据/梯度归约错误必须修。4 CPU worker、2预取、整域设备batch按真实VRAM，JIT持久cache；分域吞吐/峰值/RSS/编译/保存实测，不盲建每个task独立图。OOM回退同数学配方microbatch，健康旧作业不打断。

## 4. Hopper自动接力：保留原1K/2K，不让新数据等到完全齐备

先核真实PBS和owner。原735204运行中则继续；若结束，确认1K完整Orbax manifest、参数/Adam/RNG/cursor/访问、实际退出与Go2导出。R0已部署从STEP0250复现27工程步，原作业正常终态后在H200独立执行一次；不拿R0端点当父。若已执行则复用，工程失败有据后修一次或关闭，不无限阻塞正式训练。

原V4 1K→2K仍用同课程同optimizer/LR/visit bank，从1000恢复剩余1000（**不是**再加2000）。遵从原V4续训条件：已有行为改进或TRAIN/独立DEV可解释未收敛且无明显HOLD/源退化；不能只凭有限梯度或低训练loss自动认定。如果明确满足原门，可在有限独立评价继续时接力；若关键证据未到，不冒称已达资格，先完成可用离线/必要原生对照，同时做CPU数据准备/R0。若原2K已健康提交或运行，不重复/取消。若2K不合格，则保留已完整1K与旧阳性父，基于失败机制选择新源分支；不要重复失败1K。

**自动调度状态机**：复用现有提交intent/Journal/onceIO、已授权PBS用户服务/后继脚本，在合法环境部署唯一接力owner：
`READBACK -> VERIFY_COMPLETE -> HANDOFF_EVAL -> ELIGIBLE_NEXT -> SUBMIT_ONCE -> RECONCILE_JOB_ID -> ACTIVE -> SEALED`。
每一阶段以实际持久化事件与哈希触发，不依赖本地Codex聊天在线；需要site允许的执行位置。无实际授权/可用预算或状态UNKNOWN则停止重复提交并读回，保留终态。只有旧任务确知结束或合法无冲突才占H200；原有效quota/账户/旧总截止2026-12-05 12:41:52 SGT、320 GPUh不能通过改名重置。每capsule≤6h，预留≥15min保存、按当前p95和实际保存成本分段，每100步或30min先到及端点封存，完整临时→fsync→atomic COMPLETE，核哈希并合法备份。Vanda暂停，不重启旧CPU守候；无跨站授权通道则保持原READY并报告，不能把私有权重公开。必须回报实际后继jobID/owner/依赖与独立读回；文档不等于部署。

在V4健康2K终态或已结算不合格后，新课程选择**首个新正式更新前**已完整、原生有依据的父：若M2000通过回归及行为选择门→M2000；否则M1000若实际通过→M1000；否则MAIN3.8K（已历史阳性）。不得选择工程端点、MAIN250失败、未封存3.9K或仅29个推理数组。若已有更新更好的完整候选，需先登记独立父选择依据。保存各新域初始化、参数映射、原域完全一致初值；新树/新采样课程从新optimizer和RNG开始，不拼旧Adam叶冒充resume。新source阶段更新计数从0，历史谱系单列。

## 5. 新课程：先ARX5-rich，再Go2 joint adaptation

### 5.1 共享配置

保留单H200 learner，BF16冻结基座、原FP32 trainable/grad/Adam、原共享action expert及VLM LoRA、独立domain IO；视觉/语言基座冻结，原state-token dropout0.25（按真实实现核对）。固定effective FM128，4个Go2完整真实H40 aux槽，原10步全反传。新域分组轮转，不把不同H/codec/stat直接stack；一个group内部每step默认选一个已经READY且匹配domain的子源，按预生成round-robin/任务分层访问银行选择，尽量每个native shape整批并行；若有必要分子domain，按各自样本数加权后再group均值。固定全部选源概率、任务/episode/query、噪声/tau/dropout与完整aux支持，数据多样性优先，不让最长episode垄断。按任务→episode→query分层，平衡接触和关键阶段，记录每个source实际曝光。弱来源≤100条独立候选，初期在single组采样最多25%，绝不挤占可信全部单臂。

| 阶段 | Go2 | ARX5 single | ARX5 dual | LIBERO | Aux | 目标 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V6 SOURCE primary | 16 | 64 | 32 | 16 | 4 | 先扩大近本体源知识，少量Go2 replay |
| V6 SOURCE dual-only fallback | 16 | 0 | 96 | 16 | 4 | 仅新增双臂READY时，单独标明不含单臂 |
| V6 ADAPT primary | 64 | 32 | 16 | 16 | 4 | 使用Adapt100/200和源回放 |
| V6 ADAPT dual-only fallback | 64 | 0 | 48 | 16 | 4 | 单臂还未准入时有界回退 |

新目标loss明确为各组每样本有效time×dim归约后group mean的加权和，权重为B_group/128（SOURCE primary 1/8,1/2,1/4,1/8；ADAPT primary 1/2,1/4,1/8,1/8），子域按其在group实际B比例，辅助项独立。以实际旧目标FM系数和旧aux有效系数核对并保留其相对比值：若旧域FM归一化为2:1:1→Go2 0.5，SOURCE 0.125时新aux有效系数为旧λ×0.25，ADAPT恢复旧λ；若实际旧实现不同，先把旧目标/aux相对量写成实际公式，再按新Go2权重换算，不能把旧2/1/1未归一化数值直接当λ。4个aux仍来自当步不同、完整支持的目标slot，16目标也须预先满足；mask后无支持不冒充有效。梯度在设备端FP32归约，跨域合并后一次global clip+Adam；未参加的domain私有叶及Adam计数/decay不更新。无跨域虚构动作平均。

### 5.2 SOURCE阶段与端点

- 数据READY最低门见§2.1（不等2K源配额齐）；优先Mix500实际版。新SOURCE5K日程：新optimizer/RNG，peak = 原V4已核实声明峰值的0.5倍（保守预设，实际数值首发前封存）；warmup100，cosine至0.1peak at 5000，第一正式里程碑累计1000；检查点100/250/500/1000/2000/5000。原1K/2K课程不热换。正式SOURCE前1000必须实际消费新源，单臂不存在时只允许显式fallback，不隐瞒。第一波数据固定到当前有界阶段终点；新READY包可在1000自然边界开启**独立版本**，明确旧父/新optimizer或合法同树同日程resume、新银行游标/权重统计；不能改变数据却声称原5K固定对照不变。若改变新域或统计，重新绑定新谱系及预算；总新SOURCE实际正式更新累计至5K（不是每个数据版本再加5K）。
- 1000端点做预固定新源独立DEV、原Go2 HOLD/PLACE及安全raw指标、源保留；若有正迁移/源改善且目标无明显回归或可解释未收敛，自动继续至累计5000（可按新READY冷边界切换）；若源明确改善但Go2下降，优先启动目标ADAPT以恢复，不盲长训。若同症状反复、无新数据/证据或评估显示退化，先改变数据/归因，不以源训练loss下降作充分条件。可在不冲突的已登记评估并行后继。
- 5K→10K仅在新源数据覆盖明显增加且阶段效果有证据、真实资源许可时登记新LR后继（低峰值、不复活已衰减日程，具体数值首发前绑定）。不是默认自动训练到20K。条件SOURCE总上限10000，其中首轮默认计划5000，不能将V4的2K计入SOURCE的5K。
- 以当前V4 5.087s/step外推，1K约1.41h、5K约7.07h、10K约14.13h，只是原三域热态情景；新增源horizon/分支、batch布局会改变吞吐，先实测，不以此承诺时间。

### 5.3 ADAPT阶段与纠正

Core V5保持Adapt100→200＋Recovery40。SOURCE阶段固定父完成后，不要求所有200条就绪；至少有新兼容Adapt100则采用该版本（不足但确有高价值新技能，另登记更小真实版，不冒称100）。若数据仍未扩充，先用52/69小型Go2 replay作有界1K适配/回归验证，等待V5可继续SOURCE已登记段，而不是反复在旧52上跑5K。选择完整SOURCE端点（原1K/5K按预定条件，不挑中间最佳权重）作为新ADAPT父。新optimizer、warmup100、peak取SOURCE声明峰值或更低的已核实安全值（默认0.5×SOURCE峰值，首发前锁定），cosine到0.1peak at 3000；初始1000→条件3000，之后明确新版本条件追加至本轮ADAPT累计5000；保持源回放，评估Go2成功/恢复、新技能与源保持。Adapt200/Recovery新增数据通过新冷边界进入，不改健康训练的bank。纠正片段保持真实支持mask及来源，采样按本体旧/新/纠正阶段；无Recovery时不凑比例。后续可另注册250–1000步有针对性的纠正，不自动作为本轮新额外容量。当前训练能力仍是EEF-only+冻结WBC，未验证导航或BASE head。

## 6. 科学评价：不把源域扩充当作已证明迁移

延续V4原有PARENT/M1000/T1000/M2000与V5独立目标未见技能评价（Drawer/Insert/组合），原生旧条件不重复、新模型用原受控接口、真实噪声/场景/评分。第一批V6原生最小panel：父、SOURCE1K、SOURCE5K（若达）、ADAPT1K、ADAPT3K（若达）各4原PLACE和2原N02 HOLD（共最多30个，若原同端点同条件已有实绩引用）。固定使用V4 DEV配对场景/噪声身份701/702及旧N02两份，不拿最佳中间模型替代预定端点。源域固定DEV/TEST按episode/group分离；每新source至少16真实DEV窗口（有多个episode/阶段，含接触/后段；不足报告）检查源实际输出、归约mask、阶段/动作准确，不把MAE写成闭环成功。原源评价尽可能复用真实旧环境/统计，新增源无物理backend就只报离线。新目标任务仍先Teacher真实资格，已封存V5 CONF/zero-shot使用同条件，不把目标未见技能数据提前加入Go2。公共数据已在SLA base见过与否可能未知，称目标域zero-shot而非全训练零样本。若新增源TRAIN任务与holdout同名，这是有意测试跨本体技能，不是目标数据泄漏。

优先实际工程主线V6 SOURCE→ADAPT；另保留一个成本有限的控制：从同一已封存V6父，匹配目标数据/同目标采样访问/动作语义及LR的ADAPT-only（1K）与源扩充后ADAPT1K，比较源知识扩充的增量效应；额外source计算不同，报告而非声称等计算量纯因果。若要严格比较先源后目标 vs 从SLA-base仅目标 vs joint，另登记共同SLA-base、相同target曝光、等source曝光的两阶段/混合对照，不能把已有MAIN混合祖先当从未见源的模型。当前主线不等全部对照。原V4 TARGET_CONTROL1000不重复、真实资源足够才执行。科学结果以独立任务真实完成、移动HOLD保留、安全STOP、原H40/消费和迁移数据效率为主要指标。

## 7. 有界资源与交付

本轮新正式SOURCE≤5000默认，条件≤10000；ADAPT≤3000默认，条件≤5000；可选匹配控制1000；工程新domain/全配方资格≤24（R0已登记27单列，不重复）。各阶段在真实授权、配额、存储和有限截止内自动接续，不凭本文件提高旧GPU授权。Core新原生最小panel≤30，复用旧V5评价人口；源设备评估若另启需独立有界登记。无新增实际执行的文档不能作为达标。

每波必须交付：逐数据源许可、revision/episode/raw file哈希、实际训练/留出/拒绝数量、真实独立domain reader与统计/mask、训练入库manifest、目标/源实际访问账本、GPU throughput和完整checkpoint/恢复、后继PBS ID及调度状态、真实原生任务与失败分母。没有合法数据访问就跳过该源，不等待新源齐全；没有可用H200则继续获准数据准备与CoreV5。不得自动删除用户文件、重开暂停Vanda、改变WBC/safety或公开私有资产。

[v4]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f72f3d57b470de3e62821cd5e1a33d938a669cb7/docs/protocols/SLA_HOPPER_AUTONOMOUS_V4_PLAN_20261126.md
[v5]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/2cbce375a4f5a01bedc4f065d5ad6581fe7cbb05/docs/protocols/SLA_CORE_EMBODIMENT_FACTORY_V5_PLAN_20261126.md
[actual]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d175383b07cabb701492222bc333809d57ca5830/reports/SLA_EXPERIMENT_PROGRESS/V4_HOPPER_HIGH_BATCH_20261126.md
[core]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d175383b07cabb701492222bc333809d57ca5830/reports/SLA_EXPERIMENT_PROGRESS/V4_STEP100_CONTAINER_20261126.md
[ledger]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1250f3a70bc591051a19140e9daf089eca85d999/reports/SLA_EXPERIMENT_PROGRESS/SOURCE_EXPOSURE_ACTUAL.json
[tableaudit]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/a21d8ece1748ed4dc1dcbc305a70bbf08c77c12f/docs/reviews/2026-11-12_NATIVE_ARX5_PROGRESS_AND_SOURCE_PROVENANCE.md
[engine]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b6869988b90851baf0e4ce207df3985fef1d2dc7/scripts/sla/hopper_engine.py
[codec]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/c8642aa602d3e34dd4dcda5a55ff6ae7c7c53ae6/src/go2arx5_vla/sla_adapter/codec_v1.py
[official]: https://github.com/XPolicyLab/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/README.md
[dojo]: https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo
[dojo_code]: https://github.com/RoboDojo-Benchmark/RoboDojo
[table]: https://huggingface.co/datasets/RoboChallenge/Table30
[table_old]: https://huggingface.co/datasets/RoboChallenge/Table30/blob/1eda76b8935b5c7a59cd635ca9e3fab882943e11/README.md
[table2]: https://huggingface.co/datasets/RoboChallenge/Table30v2
[neo]: https://huggingface.co/datasets/NeoteAIEmbodied/OpenNeoData
[neo_readme]: https://huggingface.co/datasets/NeoteAIEmbodied/OpenNeoData/blob/main/README.md
[twin]: https://huggingface.co/datasets/Traly/Robotwin2.0-lerobot
[twin_doc]: https://robotwin-platform.github.io/doc/usage/collect-data.html
[ville]: https://huggingface.co/villekuosmanen
[ville_agg]: https://huggingface.co/datasets/pravsels/arx5-robot-dataset
[sla]: https://arxiv.org/abs/2611.08373
[hf_dl]: https://huggingface.co/docs/huggingface_hub/guides/download
