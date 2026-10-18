# PV1：第一版共享多任务 Policy 的七天有限推进协议

日期：2026-10-19，Asia/Singapore。Decision ID：PI0294_PV1_MULTITASK_POLICY_RELEASE_V1。
状态：USER_REQUESTED_BOUNDED_GO_PENDING_NATIVE_ADOPTION；不是已经启动、通过验收或部署的收据。
用户目标：制定后续推进长程计划，得到第一版多任务policy。
研究章程：GROOT_OBJECT_MANIPULATION_MAINLINE_CHARTER_V1；参见 docs/RESEARCH_CHARTER_TASK_SPACE_MANIPULATION_V1.md 和 docs/protocols/GR00T_TASK_DATA_FIRST_FROZEN_WBC_SCOPE_V1.md。
审核代码基线：8745adc74a7177eddef5f477bcd967df1806a0ab。最近已读FD2采用收据：794e793db8abfcafbb9cb27007acca4e498a3e56；FD2完整终局尚未发布，不把已准备的几何审计代码当结果。
唯一协调者：现有local-PC Codex owner；一个共享checkout一个mutation publisher。沿原统一入口，不新增第二调度系统。
配套：reports/PI0294/design/PV1_MULTITASK_POLICY_SCOPE_20261019.json；docs/handover/2026-10-19/PV1_MULTITASK_POLICY_EXECUTE.md。

## 1. 目标、授权与不变边界

本用户的新目标授权本协议内有界的：数据生产、条件式Teacher任务参考改进、共享GR00T微调、真实闭环评估、一次有条件的补强、独立验收和研究策略包交付。它不是只允许继续诊断的FD1，也不继承已关闭N1的剩余预算；实施仍须真实站点权限、账户额度、native owner/lease/manifest/输出绑定成立。常规同范围工程修复与阶段转换不再逐次请求GO。没有现场许可则汇报精确阻断，不绕过站点控制。

第一版范围明确为 FIXED_BASE_ARX5_SOURCE_MULTITASK_POLICY_V1：同一份共享权重，仅凭双摄RGB、任务语言与实测EEF/夹爪状态，生成EEF10/H40，每次执行Prefix8并重新观察，完成T1抓放和T2推送。保持原32秒任务、20Hz参考、500Hz物理、四步推理与原S1。任务不同只能由合法语言条件体现；不得按task-id切换模型、adapter、手写轨迹或特权控制器，不采用任务成功真值来帮策略决定下一动作。

WBC代码/权重/增益/cost/solver、腿策略、plant/质量/摩擦、关节/力限、夹爪PD、原生安全规则与任务评分完全冻结。原TF-R3 STOP保留。不做WBC归因/补偿/hold-RMS优化。PV1只生产源域策略，目标Go2+ARX5/WBC、实时控制、真实机器人和跨本体主张仍分别未证；不让Vanda/EGL或目标WBC重新成为源域学习前置。本窗口不启动目标域物理/真机。

Teacher的task-space参考、成功数据覆盖与受控数据采样可以按本协议版本化改进，但绝不将固定Z补偿插在学生输出后、放宽S1、加best-of-N采样或使用oracle恢复来提升自主成绩。相机/物体真值/Teacher阶段可以用于生成和诊断，不输入被评估策略。任何未来新结构、RL、VLM全解冻、EEF13/BASE补零、执行前缀改变均不在本轮自动权限内。

## 2. 已完成证据：复用而非重跑

- N1完整训练：S-BC与S-CORR各2048更新/65536访问；同M1父、匹配T1/T2曝光。TRAIN均3/4；暴露DEV为3/8和5/8，差异集中T1，T2均1/4。
- CORR唯一新场景确认：T1=10/20、T2=4/20，40条件42尝试；23次S1拒绝、3次安全完整目标失败，真实故障0。没有达到旧14/20目标；所有旧成绩保持。
- FD1：20/21碰撞的首发几何为pad/table，另1未合格；其余速度和finger-validity限位拒绝被定位。14个T2拒绝都在真实接触/移动之后，不使用AUDIT_ONLY_approach当作真实阶段。向下参考与碰桌有关，但未证明名义4mm裕度导致训练失败。
- 既有固定8锚点/16模型输出及影子分析已经完成，不为了新计划重复诊断。
- FD2是独立已采用的4个新TRAIN布局、4/12mm共8条专家pilot；承接其原始结果和一次机会规则，不因PV1扩大其原分母或重启健康任务。新几何测量要核对finite table、pad footprint、原生query和解析间隙区别。
- 历史模型、N1/FD1终局、失败与清理零释放收据保持；旧窗口均不滚动续时。

策略主父模型为已完整封存CORR@2048，而非再次从M1或官方模型重来：manifest 94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38。M1与BC/CORR旧checkpoint用于身份/机制证据，旧实际结果不重跑。当前冻结CORR在新的PV1开发集可以评价一次，以建立新人口上的参照，不复用旧CONFIRM当新测试。

## 3. 预先冻结“第一版”验收标准

所有等级均为研究仿真范围，不是部署安全认证；既有14/20门槛不改写。新最终评价用20个全新scene-group × T1/T2 × 两固定噪声bank，共80条件，每任务40条。

### PV1_VALIDATED（本轮主目标）

必须同时满足：
1. 唯一共享checkpoint与processor/stats/runtime封存；两任务各至少32/40完整自主成功（80%）。
2. 每任务至少14/20场景组在两bank均成功；预先分配F0=4组/F1=8组/F2=8组，各task-family至少50%完整成功，避免整体掩盖某一族崩溃。
3. 每任务原S1拒绝不超过4/40；真实native安全故障0；80条件均有独立合格审计，工程缺失不删除分母。
4. 当前真实图像—模型输出—安装—实际ACK—下一观测连续因果链成立，Teacher OFF；同scene/task对的初态图像、EEF与配对噪声身份核验，策略输入不泄漏task-id选择器或特权状态。
5. 可重建/加载/运行的策略交付包和完整失败、安全、版本、数据声明齐全。

### PV1_BETA（预先声明的次级阶段结果，不冒充主目标）

若两任务各至少28/40（对应旧70%研发标准），每任务至少12个场景2/2、各family至少50%、每任务S1拒绝最多8/40且所有审计/原生安全/交付要求成立，标记PV1_BETA。否则只交付BEST_RESEARCH_CANDIDATE_NOT_QUALIFIED和负结果。不得看结果后降低上述门槛、重抽场景/噪声或换第二候选。

20组而非40条独立场景；报告task×scene×bank原始表、0/1/2成功分布，以及按scene成组重采样的95%区间，保留两任务/两bank配对。小样本和0故障不支持真实高可靠安全保证。研究selected可指向实际通过相应等级的固定版本，real_robot_deployment始终NONE。

## 4. 关键路径与阶段准入（有出口，不做无限诊断）

### P0：采用、复用FD2、冻结唯一首轮干预（软目标T0+24h内）

先刷新实际HEAD/#11/#47、FD2 native状态、原source/asset模型、old jobs/leases、Hopper个人quota与真实权限。健康FD2继续原范围；PV1可并行准备metadata/采样器/评估卡，不能抢占旧owner。FD2完成则读取一次完整结果；只做缺失且影响决策的最小只读核验，不重新跑FD1/FD2有效例。

FD2只能改变下一步的分支选择，不能无限阻塞：
- 4mm普通专家动态可行且12mm未显示合格收益：选择COVERAGE_FIRST，继续原4mm Teacher，检验新的独立任务几何覆盖。
- 12mm完整任务成功数不少于4mm且至少3/4，至少3个可比较pair接触相关的实际最小pad-table裕度增加>=4mm，原no-lift/目标/退出未恶化：允许CLEARANCE_PAIR候选，不宣称四组证明总体更优。下一数据对照须在同scene双Teacher完整成功的公共支持集比较。
- FD2审计不充分：只对已有读侧补缺；超过软24h仍无可采用结果，若原普通4mm成功Teacher可复用，则走COVERAGE_FIRST并保留FD2未知；不得伪称12mm已更优。
- 原Teacher在本轮新校准布局也大面积不能完成：允许一次有界TRAIN-only任务Teacher改进，最多两个预封存版本、总24个新episode，改动限实际对象/目标反馈、释放/分离/保持或4/12mm参考，不动底层。每版用不同预定开发布局，全部attempt保留；没有合格可执行数据就DATA_GENERATOR_BLOCKED，不烧GPU训练。

P0所有新Teacher校准最多24 episode；旧FD2成本不重记。仅凭单例/安全但不接触物体，不得选择Teacher版本。4mm与12mm只做原定两值，不进行隐藏连续搜索。判据使用相同物理事件阶段、有限桌面正确距离、原task成功，而不是全程飞行阶段的最大间隙。

### P1：建立新数据银行和独立评价人口（软目标第1–2天）

在任何新数据/模型结果前封存PV1_DATA_PLAN：
- 96个新TRAIN几何组：F0=24/F1=36/F2=36，每组T1/T2；F1正反方向各半；按源XY、目标XY、距离、yaw和外观分层覆盖原任务族，不额外引入F3或第三任务。
- 12个新DEV组：F0=2/F1=5/F2=5，每任务两固定bank，即48条件/候选；仅开发用途。另有20个LOCKBOX组：F0=4/F1=8/F2=8，即80最终条件。新TRAIN、DEV和LOCKBOX彼此排重，同时排除原TRAIN、C0/C1 parent、FD2 pilot、全部已打开旧DEV/CONFIRM几何及appearance siblings。
- 使用原空间/相机/物体分布、静态IK/碰撞/视野规则和已验证2mm/2degree几何排重。封存候选池及拒绝原因；DEV/LOCKBOX不按Teacher或策略成功筛选。不在结果出现后重抽seed/换场景。
- 同几何T1/T2初态与视觉尽量严格配对，语言指令不同；实际hash与状态相等性必须核验，不能使用task-specific初态暗示策略。
- 普通路线新采96×2=192条第一尝试。CLEARANCE_PAIR至多另采同96组T2另一裕度96条，最多288条第一尝试。所有失败保留；最多总16条可证明工程无结论的替代，合法失败不重跑。成功示范可筛入BC，但完整采集人口的失败率/筛选偏差必须报告。
- 训练数据准入：原安全+完整物体task成功+真实提交动作/观测/时间一致+完整H40及prefix标签+split/hash。记录state实际值与Teacher期望动作，不能用未来achieved状态替代动作label，不能把failed action当正监督。
- 最低新独立有效支持：T1>=32组、T2>=64组，T2正反方向各>=16组；目标96/任务。达不到不无限补采；走一次已规定Teacher修正或交付数据阻断。CLEARANCE_PAIR的双Teacher公共成功支持至少64组，不足则不进行纯裕度学习因果比较，可退回原Teacher覆盖路线（若原成功bank够用）。
- 原v3.0 normalizer、EEF10、rot6d顺序、相对动作与时间语义默认固定。对新增数据做逐元素encode/decode/crop/clip审计，误差沿既有数值门；不能静默用旧range截断正确新动作。超范围或非法标签先修正同语义数据接口；需要更改normalizer/表示而无法保持比较时报告专门阻断，不一边训练一边refit。
- 原始RGB/高频物理/参考/夹爪/phase-events均保存；统计独立scene/parent数，不用切窗数冒充示范数。先按scene切split，再切窗。

复用现有采集器，CPU多worker可在独立engine/render/输出/随机状态/lease验证后2→4扩展，计入总核数。先冻结64组以上合格小银行即可准备模型；后续数据不热注入已开始的固定训练。

### P2：第一组正式共享策略对照（软目标第2–4天）

先将旧CORR@2048作为B0在新DEV48评价一次（不训练），用于判断进展和T1保持。随后两个新臂均从同一旧CORR@2048权重开始、全新相同optimizer，不串行从对照臂继续：

默认COVERAGE_FIRST：
- A_CONTROL：原专家+C1，任务T1:T2=1:1。每effective batch32：T1原8+C1 8；T2原16。
- B_COVERAGE：相同task/C1访问与保留游标；T1原4+新T1 4+C1 8；T2原8+新T2 8。只改变独立新示范覆盖，C1仍为总25%，不把4个纠正parent扩称独立规模。解释为两任务覆盖的联合干预，不单独归因某任务。

有证据选择CLEARANCE_PAIR时：
- A_4MM、B_12MM使用相同新scene公共支持与完全相同T1/C1数据、相同T2旧样本/新样本比例；只改变新T2专家参考标签及其相应真实示范。不同Teacher实际运行造成的状态轨迹差异属于这项Teacher干预，不声称同状态标签替换。采集失败全部留账。

两臂固定各4096正式更新、effective batch32=micro2×accum16、native flow matching、VLM冻结、原projector/DiT训练图谱、seed20261019、lr 1e-5→1e-6、warmup64、其余optimizer参数继承已封存Source Pair配方。训练BF16/原TF32上下文，推理沿原四步BF16/FP32-TF32 false上下文。预注册1024/4096保存点；1024只做廉价固定TRAIN探针，不额外打开新DEV或作为最终选模候选。每臂131072真实样本访问，全游标/parent/阶段/梯度/冻结参数/resume状态可审计。

旧代码2048上限必须以新PV1 scope版本扩展，旧回放仍拒绝越权；不能改旧N1预算字段、删除guard或只改计划不改dispatcher。model head tensor数量不是参数总数，也不把VLM冻结误称标准LoRA。

末端4096才进入新DEV48。所有候选用同一DEV48与配对noise；每个有效条件只一次。T1、T2、family、真正物理事件和安全/完整目标失败分开。loss仅用于训练完整性及失稳预警，不据总体RMSE选择policy。

### P3：最多一轮有条件补强，立即转化为训练（软目标第4–5天）

用P2 DEV选择最弱任务及机制；不读LOCKBOX。最多选择以下一条方法，并再做一对共同父模型的2048更新对照（C_CONTROL与D_INTERVENTION）：

路线R：实际学生状态纠正。前提是TR AIN-side原/改进Recovery Oracle在预登记6个锚点中至少5个从同状态完整安全成功、按任务独立合格。锚点来自新TRAIN学生rollout而非DEV/LOCKBOX，失败及未恢复均保留。最多24个独立学生episode，两任务各最多12；对应Teacher恢复fork最多24。仅使用实际成功恢复的监督，原S1拒绝动作绝不当标签。资格不足的任务不制造纠正数据。对照两臂同父/同更新/同任务曝光；task内纠正访问最多25%，其余 replay留T1能力。新增parent/接管状态/后续专家窗口分开统计。短前缀安全不能充当Recovery Oracle资格。

路线E：物理事件均衡。若学生接触/目标后失败与**独立TRAIN事件级**拟合差且有效前八点曝光不足共同出现，而完整恢复Oracle不合格，可只改任务内anchor采样；接触/脱离/释放/withdraw/hold事件取自真实数据而非旁观Teacher假阶段。先写精确概率表并固定，同组对照其他因素不变，不同时加入新纠正数据或新loss。

路线D：若两类证据都不足，跳过这对训练；不要自动开更多诊断/新backbone/RL。把P2最优者送后续门槛，未达到开发资格则给明确负结果，不能靠同一数据不断延长更新。

若需要小幅Teacher任务参考/恢复代码以实现R，只允许在P0/P3总体有限校准名额内验证、版本化并计费；不改底层或自主评价里的Teacher OFF。此轮干预只能一次，不做每次失败后重新起一套goal。

### P4：开发选模、一次训练随机性复核与候选冻结（软目标第5–6天）

最多新训练5次：P2两臂4096；P3可选两臂2048；一次胜出recipe的不同seed复核（seed20261020，若胜出P2则4096，若胜出P3则2048）。总optimizer更新最多16384，最多评估6个checkpoint（包括旧B0）。复核只重训胜出局部recipe，若共同父为已训中间模型，则不声称整个多阶段训练都跨seed重复。

新DEV排序预先固定：先worst-task完整成功，再macro完整成功，再更少S1拒绝/更低复杂度，最后更早candidate-ID；不以视频或最高单次noise选模。开发候选至少每任务18/24成功、native fault0，才打开最终测试。预设B0保留门：新candidate T1或T2不得比B0低超过1/24，且worst-task应有净改善；若达不到不谎称稳定第一版。

同训练配方复核若无法重复到每任务18/24，则报告训练敏感性，不能仅挑幸运seed称稳健；可以交付候选但不打开主级验收。资源临界时优先终点评估，若省略复核，则最多可评为BETA，不得称PV1_VALIDATED。

达到开发门的唯一候选完整冻结：weights、processor、stats、runtime、precision、action semantics、task语言、plan和环境hash。LOCKBOX场景卡可提前封存但执行与结果在候选冻结前不可打开；不看部分test后改变candidate。

### P5：全新80条件独立确认及研究包交付（软目标第6–7天）

执行第3节预声明的80条件；每任务20scene/40trial。全部成功、拒绝、完整目标失败、工程缺失保持分母。每个condition最多一次工程replacement，全campaign有限总额；有效task fail/S1拒绝不得重跑。无best-of-seed、无Teacher接管、无评价期训练、不提前oracle结束episode。

最后确认只选一个候选；若不达门，封存真实BETA/NOT_QUALIFIED并结束，不用剩余预算在已打开test上找第二模型。统计scene clustered区间与两bank一致性，不伪造95%部署可靠性。

交付POLICY_V1目录/固定私有HF或现有模型存储revision，Git只放脱敏card和hash索引。必须有：
- 一份可加载的shared weights、exact model/processor/stats/embodiment/动作schema、依赖锁和最小推理入口；提供dry-run离线加载与输入schema测试，以及在批准仿真scope运行的配置，默认不触发真机。
- POLICY_CARD.md，明确SOURCE_FIXED_BASE_ARX5、T1/T2任务、适用geometry/语言、已知失败、S1依赖、paused simulation而非real-time、无WBC修改、非真机可部署。
- 完整训练lineage、actual consumption、复核seed、DEV选择轨迹、80条件表/全部attempt/真实视频与raw索引、precision、审计、安全及UNKNOWN清单。
- REPRODUCE.md、POLICY_MANIFEST.json、ACCEPTANCE.json、冻结资源最终账、checkpoint退役清单、加载/单次生成/重置/失效停止回归测试。原始模型/视频/凭据不得进入Git。
若未达门也交付同样可运行的candidate包，但醒目标记NOT_QUALIFIED，不把完成训练包装成第一版合格。

## 5. 有界自主修复与预算

这是新PV1工作窗口；N1/FD1已关成本与FD2独立成本不得重复计费。首次采用不晚于完整发布后48h；记录PV1_T0一次，全部排队/修复/采集/训练/审计包含在168小时内。T0+144h停新job/候选，+156h停新训练块，+160h停新物理实验，+168h硬收尾。阶段日程是优先级与预算边界，不是成功或耗时保证。

新累计上限：
| 资源 | 上限 |
|---|---:|
| Hopper H200 allocated GPU-hours | 96（同时最多1卡） |
| Hopper GPU PBS submissions | 24（每次最多8h，站点更小者优先） |
| A40 / 目标WBC / 真机 | 0，本轮不增加旁线 |
| CoreLab CPU | 320 core-hours，同时最多8核 |
| Hopper CPU | 160 core-hours，不超native请求 |
| local-PC协调CPU | 48 core-hours，同时最多4核 |
| 优化器更新 | 16384（5次训练硬上限，不含梯度累积微步） |
| 新神经生成 | 48000，包含probe/DEV/CONFIRM/工程失败 |
| source真实episode/恢复fork starts | 1000，计采集/评估/失败/替代 |
| live+shadow+重建总物理ticks | 40000000 |
| engine构造+S1 clones | 100000，另与episode计数并列 |
| 新驻留峰值 | Hopper250GiB/Core300GiB/local20GiB |
| 新传输 | 300GiB，复用cache和单writer，禁止整模型重复下载 |

最终验收硬保护：20 H200h、1600000 live ticks、1600000 shadow ticks、100个episode starts、9000次神经生成和40GiB存储是上述总额的子集，不是额外额度；除最终资格/确认/收尾外不得动用。软件应在每次新训练和提交前核算未来必需评估预算；保护不足立即停止新候选。

软分配H200：训练36h、开发/学生诊断24h、工程恢复16h、最终验收20h。owner可在前76h内部调整，不增总上限、不拆出新GO；并发采集与Hopper训练允许独立数据版本，跨机动作仍使用已验证通道。

复用原成熟transport/teacher/dataset/trainer/safety/audit工具；只新增PV1 namespace与必要采样、状态及role配置。遇到旧硬编码2048/固定session/PBS次数，增加有版本的PV1验证器与旧协议回归，而非关闭所有guard。每新GPUallocation的跨机原600s/80cycle/all-fault资格保留；同一健康allocation只验证一次，不每换checkpoint重跑。批处理元数据统计与归档，不能把非关键fsync放在每次动作响应前；即使改IO，原关键动作/unknown-effect日志仍持久。

自主修复权限覆盖：读侧/NFS/SSH、路径/依赖、PBS合法字段、quota、数据时序/codec同语义错误、exact resume、日志统计/批归档、scope兼容。每次变更测试后冷态部署，固定code/source/CI/manifest；不热改正在执行的actor，不改模型或安全科学身份。对可确定幂等metadata操作做有界重试；query/install/qsub/未知删除效果先核对journal和原生进程，绝不盲重发。

同一未定位工程故障原生复现两次后回到证据诊断，不继续烧第三张同版本作业；诊断建立新原因与回归测试后可在本总额度内恢复，不需要逐项叫醒用户。真实native安全/完整性/越权错误只停受影响lane并取证；不得借新scope解除原安全STOP。

## 6. Checkpoint生命周期清理，不阻断训练

沿用一个维护owner、一个consumer/ref图和版本化retirement回执。仅在新完整恢复状态封存、run真实终态、引用解除或真实quota风险时增量检查，不反复全盘扫同一批空目录。

长期保护M1父、旧BC/CORR固定对照和必要复现witness、当前PV1父、全部已用于选模的终点推理权重、当前最佳完整状态、最新resume+一个回滚点、冻结最终候选、数据/视频/唯一失败和安全证据。无active consumer、无未来resume/对照/审计依赖的失败临时payload和可重建重复cache可按已有授权直接删，不逐文件再次确认；保留轻量manifest、失败原因、精确删除intent、实际unlink/allocated bytes/quota差额。

有效中间checkpoint不因任务0成功而变垃圾。旧BC1631只在明确payload不再承担精确resume/审计时作为SUPERSEDED_VALID_PARTIAL重新评议，不强删。原始控制/数据/S1/WBC或其他用户资产不动。不通过杀活跃消费者制造清理资格；无合格对象就0删除继续科学工作。

## 7. 报告、停止与实际完成的定义

读取最新固定卡后现有owner在#11发布一次真实ADOPTED并给下一native动作，#47仅维护收据。活跃时每最多4小时写简短里程碑，每24小时一份主表：数据独立覆盖、训练steps/consumption、每任务成功/失败、安全/工程缺口、资源used/reserved、清理真实字节、下一决策；健康运行无需每个helper新起goal。

在统一注册根新建PV1有限run与reports/PI0294/PV1/<run>/小型报告，不覆盖N1/FD1/FD2。所有新collector/trainer/eval/publisher在原RUN_MANIFEST/profile/lease/OUTPUT_BINDING准入下派发。GitHub只发布协议、脱敏证据、模型card/索引；私有绝对路径/token/raw权重/WBC/vendor不上传。

收尾必须真实native jobs和effects终态、租约关闭、future reserve归零、训练与评价总量相符。计划成功并不保证策略成功；若到上限或deadline未达标，必须交完整候选包、失败边界与唯一下一实验，不宣布PV1通过或无限自动续期。

## 8. 基线证据与出版边界

- N1结果：658b432fa1ee37182945e84b7d4ec4d830bee67e / reports/PI0294/20261016T191445Z-localpc-mp1/N1_FINAL_SCIENCE_REVIEW.md。
- FD1归因：ba40c713b483a02be5387b5254d045174dc6493b / reports/PI0294/20261016T191445Z-localpc-mp1/FD1_DIAG/FD1_FINAL_ROOT_CAUSE_REVIEW.md。
- FD2范围：318445f17ed2093595e9df252cc56bb24280f805 / docs/protocols/PI0294_FD2_DYNAMIC_EXPERT_CLEARANCE_PILOT_V1.md；采用794e793d，最新管理代码8745adc7是几何审计准备而非最终成绩。
- 原训练实现：scripts/pi0294/mp1/train.py、dataset.py、neural.py；事件、安全与Teacher参考按原文件实际hash封存。

发布此文仅表达用户新目标对应的有限PV1科研执行范围；实际站点权限和owner采用须有回执。本次发布没有SSH、GPU/PBS、模型训练、物理执行、checkpoint删除或WBC修改。