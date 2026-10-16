# MP1 GO：从首个自主成功到双任务策略、纠正学习与冻结WBC迁移

日期：2026-10-17，Asia/Singapore。
状态：**GO_BY_CURRENT_USER_REQUEST / SINGLE_OWNER_BINDING_REQUIRED / NOT_LAUNCHED_BY_PUBLICATION**。
授权依据：用户明确要求“授权更多预算的goal，跑通实验主线，更新长程计划”。本文件给出现役Codex可连续执行的新增长程科学与资源范围，不是等待第二次常规确认的草案。站点账户/项目/调度器许可仍须实际有效；本文不能创造集群权限。
审核基线：`8fdb2a179f5db88c294fea4182c2d177a2e2801a`。最新已读科学评论截至2026-10-17 02:08:50 SGT；没有在本对话SSH实查服务器。当前原生任务状态以owner接手时的只读核实为准。
机器卡：`reports/PI0294/design/MAINLINE_ACCELERATION_MP1_20261017.json`。
接手入口：`docs/handover/2026-10-17/MAINLINE_ACCELERATION_MP1_EXECUTE.md`。

## 1. 当前事实已经改变：承接已有成功，不重新从零制定Teacher计划

- E0仍为12次尝试、11条有效S1拒绝、1条工程不完整，TRAIN 0/3、DEV 0/8，真实执行安全故障0。旧报告、统计、关闭租约和所有费用不变。
- E1 M1已从M2048实际完成1024次更新、32768个样本消费。2026-10-17 02:05发布的独立审计证实：**M1-1024-CANONICAL在source TRAIN/F0/000/T1自主完成32秒抓放**，16000积分、641双RGB、无Teacher接管、S1拒绝0、原生安全0；实际抬升83.536mm、水平搬运176.906mm、末段41/41保持子判据全部通过。不是Teacher回放。
- 这是一个已见来源域条件，不是双任务能力、两个T1布局通过、DEV泛化、Go2/WBC或真机成功。最新评论中同版本F0/000/T2已为可信S1拒绝，F1两条件尚未收口。不要提前报完整面板成功率。
- 三个E0 TRAIN拒绝已复算为table/right_pad、table/right_pad、table/left_pad，位于即将执行的0.4s前缀内。原同状态专家和仅XYZ替换通过，单独姿态或夹爪替换未通过。支持优先调查位置/接触几何误差，不证明唯一原因，更不授权固定目标偏置或放宽桌面接触规则。
- Recovery Oracle仅T1取得当前资格；C0有833个去重完整H40窗口，来自2个E0学生episode，实际独立reader通过。其身份是`E0_STATE_AGGREGATION_ONLY`，不是833条独立示范、不是新M1在线DAgger、不是目标整机数据。4个初始恢复历史无效样本明确排除，不复活。
- 推理TF32继承问题已揭示同输入/权重的输出差异，CANONICAL上下文须固定；旧不同精度结果、一次意外重复的scope deviation不改判或混入公平对照。不能把首次成功的原因完全归于额外训练或精度修复中的任一个。
- 最后发表记录中Hopper E1第二作业670729在运行；D/M2是否已有新结果须现场刷新，不能根据准备代码宣布完成。Vanda最新已发表快照仍为1462438硬件Q、1462451依赖H、1462595构建Q；六项输入匹配不等于新WBC二进制或闭环已合格。

固定证据（均相对审核基线）：
`reports/PI0294/20261016T150228Z-localpc-e1/M1_1024_FIRST_T1_INDEPENDENT_AUDIT.json`；`A_INPUT_PROGRESS.md`；`CORRECTION_DATASET_CARD_C0.json`；`ORACLE_HISTORY_ADJUDICATION.json`；`FIXED_PROBE_PROGRESS_AFTER_M1_1024.json`；`PROTOCOL_SCOPE_DEVIATION_01.json`；`reports/PI0294/vanda-bringup-20261016/CLOSED_LOOP_PROGRESS.md`；#11 comments6221515047、6221769022。

## 2. 总目标、成果等级与优先级

保持原架构：**GR00T(RGB/语言/实测状态) -> 任务空间EEF/夹爪轨迹及可选BASE软意图 -> 冻结WBC -> 真实物体反馈**。不改WBC/腿策略/plant/控制增益/solver，不让环境建设重新吞掉VLA研究。

主要交付按顺序递进，但独立工作并行：
1. 收完E1已有规范面板与同父M2对照，固定可复用checkpoint及纠正数据；不覆盖旧结果。
2. 得到一个**共享权重、语言条件的source T1/T2策略**，真实独立评价并解释弱任务；不以loss替代物体行为。
3. 跑通真实source DAgger数据生产->数据集->训练->自主比较；普通BC对照保留。专家未合格的任务不阻断另一任务和普通BC。
4. Vanda完成当前MuJoCo/ROS/冻结WBC的同节点闭环资格；在安全可执行前提下完成source-only整机迁移，独立记录失败层。
5. 目标域成功数据出现后，以5/20条每任务的嵌套TRAIN子集微调，必要时扩至50条每任务；完成source-only与few-target的固定比较及一次独立确认。

研发里程碑：source或target最终确认各10个未训练scene-group × 两任务 × 两个固定采样seed = 40个尝试。每任务20个，预设期望至少14个完整安全成功（70%），且两任务使用同一checkpoint；这是**本轮工程研究目标，不是文献保证、统计可靠性认证或真机部署标准**。报告所有尝试/有效/工程缺失、按scene-group的相关性与不确定性，不能把重复seed当新场景。
- 仅接口/数据链成立：PIPELINE_READY_NOT_POLICY_SUCCESS。
- 仅来源域达成：SOURCE_MULTITASK_RESEARCH_CANDIDATE；目标线继续或如实BLOCKED。
- 目标整机双任务满足固定确认：TARGET_SIM_MULTITASK_RESEARCH_CANDIDATE。
- 未达：PARTIAL_WITH_MEASURED_BOTTLENECK，不改门槛、删难条件或靠重试挑成功。
历史selected/deployment保持；新模型可登记带domain/scope的研究候选。**真实机器人deployment始终NONE**。

## 3. 新增预算：7天主计划，不重置旧账

下面是**本次新增加的上限**，不是必须花完的指标，也不是完成时间承诺。

|资源|新增硬上限|
|---|---:|
|Hopper H200 allocated GPU-hours|48，最多1卡同时|
|Vanda A40 allocated GPU-hours|32，最多1卡同时|
|新增GPU PBS提交|Hopper最多12；Vanda最多12|
|Vanda CPU构建作业|最多8次；计入Vanda CPU总额|
|单GPU作业walltime|不超过12h及站点更小上限；优先使用适合队列的较短作业并精确续接|
|CoreLab CPU|320 core-hours，活跃最多16核|
|Vanda CPU，含构建/渲染/控制|240 core-hours，活跃最多16核且不超过实际PBS请求|
|Hopper活跃CPU|160 core-hours，活跃最多16核且不超过实际PBS请求|
|local-PC准备/管理/审计CPU|24 core-hours，最多4核|
|新增优化器更新，所有模型/分支合计|24576；梯度累积不是多次optimizer update|
|source主episode或真实恢复fork启动|合计最多400；不是400条成功承诺|
|target主episode或真实恢复fork启动|合计最多320，含资格/采集/评价|
|所有新增MuJoCo积分|64000000，含live、fork、预检shadow、失败、warmup|
|本轮新增驻留峰值|Hopper 400GiB、Vanda 250GiB、CoreLab 200GiB、local-PC 20GiB；实际quota更小则取小|
|新增网络传输|合计500GiB，按实际计；先复用固定本地cache|

实际owner登记一次MP1 T0；T0+144h停止新增排队提交和新增实质候选，T0+160h停止新增实验/训练块，T0+168h硬收尾。排队不计allocated GPU使用但计日历期；不因重启或新子任务续时。首次绑定应在本发布后48h内，超过则只汇报过期，不能数周后启动旧授权。

资源分配指导，不额外制造阶段小门：H200 6h接续E1、14h源域学习/评价、10h纠正比较、12h目标适配、2h可选FlowDAgger、4h最终评价/收尾；A40 8h平台与等价验证、16h目标交互/评价、4h小适配、4h收尾。owner可在不增加总额、不扩大科学范围且保留两端各4GPUh最终评价余量时重分配，写一条ledger原因即可，不再逐项问GO。CPU保留至少15%给数据审计与收尾。

旧E0、E1、Vanda已发生/已预留费用各归原账，不追溯转移、抵扣或重置。健康在途作业按原capsule完成到正常边界。若后续需要新增资源完成E1规定的剩余更新/未完成条件，创建MP1子运行并链接父ledger；只计算真实剩余量，原M1/M2更新数不重跑。一个作业的费用只能归一个ledger。当前job已被原范围预留的容量不能同时算MP1的空余量。

不取消健康排队作业去“试另一台”；同一科学条件不得两边竞速择优。owner可在尚未产生模型/物理动作前，根据已取得平台资格选择一个后端并固定；超时后只取消本项目明确无执行且状态已核实的旧候选任务，留下取消/未执行证据，不能误操作他人任务。

## 4. 执行A：先把E1做完，而不是因为新GO重启

读取最新E1各阶段实际收据、pending request/step、checkpoint的optimizer/scheduler/RNG/sampler游标及原始精度上下文。先让正在执行的规范M1@1024面板自然完成，保留已审计T1成功与已发生T2拒绝。D/M2已开始就接续原已封存方案；未开始则在T1资格与真实纠正bank条件成立时按原方案完成，不把T2额度挪为无限T1重试。

对于原@256的有效负结果及意外重复，原身份/科学排除保持。明确许可的未完成条件可在新子run继续，不能把新运行覆盖旧不完整行。原结果尚未充分确认的字段写UNKNOWN/NOT_RUN，不通过模型名推断更新完成。

本轮E1收尾至少交：规范末checkpoint两任务实际结果、M2-BC/M2-CORR等更新比较或准确跳过原因、真正干预anchor数量与C0/C1各自来源、数据独立消费、所有native资源终态。**首条成功不是停止整个主计划的理由。**

## 5. 执行B：建立共享source双任务能力和可解释的学习曲线

### B1 数据和候选冻结
保持原v3.0与其TRAIN126/DEV32成员、stats、processor。SMALL8与C0原件不重写。扩展学习先使用原126条TRAIN，不急着重采或混v3.1。v3.1只做库存/覆盖审计；本轮主比较不换其归一化统计或数据定义，避免额外混杂。

基准父权重固定为审计通过的M1@1024；M2结果是独立纠正对照，不按某一次漂亮视频随意更换父模型。M1与M2都保留。若发现实际checkpoint不完整，先恢复已注册原件，不能猜数值或把weights-only加载说成完整resume。

新增两个主要候选：
- S-BC：M1权重warm-start，原v3.0 TRAIN126，2048更新。
- S-CORR：同一父权重、配方、2048更新；50%原TRAIN、50%当前已封存合格source纠正数据。纠正数据不足则该分支不伪造，S-BC正常继续。只有T1数据时明确任务不对称，按任务先1:1抽样，在T1部分做纠正混合，不能让T1淹没T2；同时报告实际混合占比。

任务先1:1、任务内episode均匀、episode内合法窗口均匀；原原生Flow Matching/有效mask，VLM冻结，历史projector/action DiT可训练，BF16及已验证数值路径。新优化器使用E1已解析的AdamW betas/eps/weight decay/clip、effective batch32、microbatch2/accum16；peak LR1e-5、cosine到1e-6、warmup32。每个新分支自己的optimizer/scheduler/RNG/cursor明确，seed20261017；不叫精确续训。

每分支@512和@2048只做固定TRAIN probe与开发面板；末checkpoint是候选，不从密集快照挑最好值。检查真实梯度、freeze hash、输入/输出单位、训练/推理TF32上下文隔离（推理显式恢复CANONICAL，异常路径也恢复）。不同精度结果另列，不能由BF16外层包装推断所有FP32运算一致。

### B2 有界后继，不是无限扫参
如果T2依旧明显弱且TRAIN事件覆盖显示关键接触窗口不足，允许一个S-EVENT分支：同一M1父权重、同配方2048更新，仅改变训练窗口分布，任务仍1:1，每任务50%原均匀窗口+50%原TRAIN接触/推动/释放邻近窗口（事件无此类则回退本任务均匀并记录）。阶段信息仅用于训练采样，不输入学生。该分支是采样干预，不能与其他设置一起改变。

若所有共享候选在TRAIN都没有目标技能，可做一次T2固定4episode、512更新的capacity诊断，不作为最终共享policy；结束要报告原因，不能把它偷偷替换双任务模型。最多上述3个共享学习候选+1容量诊断，不新增全参数微调、量化、不同基座或新world model。

### B3 开发与一次独立确认
E0的DEV现是**已暴露development集合**，允许在本新范围内对冻结候选评估一次，明确它不是未触及test；不在其中生成纠正标签或训练数据。候选按每任务较低成功率优先、再宏平均、再相同结果时选择复杂度更低/新增计算较少者；native失败候选不能优先。

另封存SOURCE_CONFIRM_V1：10个新scene-group、两任务共用可比初态、每组两个固定动作采样seed，总40次。使用原已定义任务/场景分布及静态可行性规则，不根据模型成绩筛场景。与所有source TRAIN、恢复父episode和外观兄弟组去重；不使用Teacher成功率筛选confirmation难度。父模型及选择规则写入清单后才开放确认。确认只评一个共享候选，不依失败重抽seed或再选checkpoint；若未达到目标，保留结论，本轮不以该确认轨迹再训练。

## 6. 执行C：真正的source纠正学习，可与B和Vanda建设并行

复用T1已资格Oracle/C0，不重复833窗口的全链建设。C0来自两个E0实际学生episode，fork历史审计限制始终披露。对T2只做有界的Recovery Oracle修订，最多两个实质版本，从TRAIN学生实际安全状态推断真实接触/进度，允许INVALID，不调S1/plant来取得标签。

最多两轮新增在线采集，每轮最多12个预定TRAIN episode、每任务最多6个。初始学生为该轮明确冻结的共享source候选。原S1合法则执行学生prefix8；第一次拒绝时，从同一实际状态查询合格专家，专家前缀也须原S1通过，随后专家接管余下段。未合格任务只记录学生结果，不盲用无恢复能力的专家。

记录：实际学生访问的anchor、专家建议H40、谁实际执行、安装/ACK、真实新观测、局部安全/任务结果、来源domain与split。后续expert状态不冒称student访问；全专家episode不叫on-policy DAgger。学生被拒动作不作为正确label；仅8点不能补32个零当H40。固定统计与实际有效维mask，完成真实writer、去重、视频引用和独立processor/collator读取。整个任务最终未成功不自动否定有效的局部纠正，但必须分开标注。

每轮数据封存后，从同一轮前父模型做512更新BC-only和512更新BC+correction，匹配训练步数并自主评价（专家接管OFF）。若采用不同数据量，报告expert query/有效anchor/独立episode/窗口，不能把增加数据或额外算力全归给DAgger。两轮中某任务连续无有效纠正则停止该Oracle支路，继续其他任务和BC。

## 7. 执行D：Vanda形成真正闭环，不等待所有source任务和DEV完美

继续已采用的硬件/容器/编译任务；先刷新1462438、1462451、1462595真实状态，保留已有失败和旧额度。新原生二进制须来自固定包并完成版本/动态库/时钟资格；六项输入一致不代表二进制数值等价。

顺序：硬件/驱动与EGL -> resident模型固定观测推理 -> 原生WBC非任务启动/观测/参考 -> 同节点新观测模型闭环。模型进程与WBC进程隔离；同A40避免同时全负载训练与渲染推理，默认分时。source学习不等Vanda；Vanda无需等T2学会才验证安全任务接口。

跨H200/A40检查相同checkpoint、processor、stats、动作有效维、精度开关、工具坐标/夹爪方向；保存同观测固定噪声输出和延迟/显存分布，不要求不同GPU必然bitwise相同。实质数值差异须单列平台因素，不能把A40较好的成绩冒充H200的相同实验。Hopper主训练；A40通过训练配置与内存smoke后可以承担小适配/steering或排队时的备用学习，新后端必须在新分支开始前冻结，不在一条精确恢复链中默默换数值实现。

当前目标TF-R3真实安全STOP保持。**追加预算不解除该STOP，不允许原样反复执行已知不安全参考。** 本MP1明确允许一个单独的TASK_INTERFACE_REQUALIFICATION继任系列：只修我们自己的启动、事务、参考时序/插值、任务连接、夹爪请求曲线和模型codec错误；先给原件定位、最小修订、受影响CPU测试与隔离预测证据，再做低激励原生资格。不得改WBC、夹爪PD/力限、质量/摩擦、限位、碰撞、安全宽容值，不能回放旧腿命令或瞬移qpos。旧sentinel不删除，新lease绑定被测试的新capsule。

最多3个有证据的任务接口版本，每版最多2个预定资格条件，全部计target启动/预算。一旦真实原生安全违规，停止该版本所有相关物理；只有证据证明属于上述允许的自有接入错误且有新修订时，才可进入下一预声明版本。同一未解决根因出现第二次真实违规，或需要改变WBC/plant，目标物理支路本周期BLOCKED，不换主机绕过；source离线学习与合法source仿真继续。

不新增独立保持RMS合格门。合法接触、物体成功、原生安全和真实时钟/参考/数据完整性仍不可省。较慢推理只采用已经验证的sim-lockstep，不延长看门狗或混用墙钟来伪造实时控制。

## 8. 执行E：完整Go2＋ARX5迁移和目标数据，不把造Teacher当唯一入口

平台通过安全资格后，至少一个任务已有source自主成功即可开始该任务的source-only开发迁移；不等另一任务。使用同一固定source checkpoint、原EEF10状态/动作契约，BASE preference明确absent（不是零保持），原WBC自行跟踪。原S1是source保护，不假称它已证明fullbody安全；target使用其已资格的原生约束/预检。

先封存目标开发任务几何、初态/目标与相机合同、实际32秒任务时间；可在**独立slow-seed域**采64/96/128秒Teacher示范，运行前固定，保持末段时长和空间/安全标准。慢速成功不改判32秒标准。最终target确认默认32秒，慢速结果单独报告，不失败后续时。

允许三种有来源标签的数据生产者：已有source成功技能的任务空间连接Teacher；已证明可在target实际成功的冻结source策略（标policy-generated/filtered，不冒称外部专家）；人在可用时通过相同高层接口给出的仿真辅助示范（不是无人值守必需）。已有源示范可用于训练，无须逐条先在target执行；只有标target-success的数据才须target真实成功。没有原生成功就不得离线变换文件后说目标示范已采好。

目标数据目标：每任务50条独立TRAIN成功+10条DEV成功，合计100 TRAIN/20 DEV；这是采集目标不是启动次数承诺。首批每任务5条即可进行相应目标微调，20条时做扩大对照，不等50条全齐；每个规模通过不可变版本交付。目标生产最多160个预定尝试，失败全入失败库并计attempt，不能丢弃分母。每任务长期无成功的块停止重复，转归因/另一任务，不耗完160次同类失败。

每个任务的5/20子集按初始确定ID顺序嵌套，TRAIN-only；数据图像/动作/实际位置/时间/执行者/接触结果齐全，完整H40来自实际已提交的同一标签语义，统计未验收不训练。不同视角/物体位置/基座起态按scene-group去重。开发和确认不因反复失败回流TRAIN。

目标学习优先原projector/DiT部分微调，不全量解冻。target-5与target-20各从相同冻结source父模型warm-start、1024更新、原数值配方，在目标开发面板评价；样本足够且仍有增益才增加target-50的2048更新。不足任务单列，不伪造配对数据。保留source数据回放，明确比例并在第一次训练前封存；默认每batch50%source、50%target，target内任务均衡。

当目标坐标/范围不适合原归一化时，先作为新数据/动作适配版本审计，禁止仅换stats使同输出变成另一物理动作；需要可测试的旧物理动作往返一致映射。超出本体状态/动作定义的架构变化不在本轮临时试。

最终source-only与few-target比较使用相同冻结WBC、任务、运行平台和独立TARGET_CONFIRM_V1。先在目标DEV选择一个适配候选，再将source-only和该候选在同一40条件配对评价；这共80个target尝试。训练后不得根据确认结果再换候选、改Teacher/场景/时长。目标确认的两模型各自成功率和配对变化均报告；模型贡献、执行器问题和平台差异分开。

## 9. BASE13与FlowDAgger是条件式扩展，不再阻塞主线

EEF13方向保留。只有target已获得含非零BASE软意图、对应可观测状态与实际安装记录的数据，才启动一个新FB13分支：从合格EEF10权重暖启动，新维的参数/缺失mask明确，旧EEF10不补零当移动监督；原WBC参考ABI不改。最多2048更新、一个预声明移动开发面板；不把free-root漂移叫VLA移动规划，也不把该实验混入32秒近场主结果。条件不足则交付近场EEF10整机结果和BASE缺口，不强制本周期实现移动泛化。

FlowDAgger仅在普通纠正数据/比较可用、主要source/target评价资源仍有预留时启用，最多新增2GPUh。固定已审计官方代码，不直接使用LIBERO7D encoder。先最多32个TRAIN anchor做真实model-space encode->inverse->forward->decode，统计XYZ/SO3/gripper有效维误差、有效率、耗时和同原SafetyGate的可执行性；mock PASS不代替真实反演。反演不足则结束，不花预算强训steering。合格时最多512次steering更新及一个固定开发对照，主GR00T冻结，不改WBC。

既有参考默认100%专家接管不能冒称student-on-policy；真实学生前缀比例、expert请求/有效率/纠正成本必须披露。视频预训练、RL、替换大模型、全参数VLM训练不在本MP1范围。

## 10. 工程自治、继续/停止与数据可信度

同范围路径、构建、IPC、认证生命周期、持久writer、read-only重取、计数与CPU资源调度修复自主实施，记录改变和相关测试即可，不再反复要求人工GO。禁止热改在途capsule、重发未知执行intent、删除历史结果或凭据入Git。模型输出与物理安装使用唯一事务ID；已完成响应只读重取可以，未知神经/物理动作不能盲重试。原失败进程日志保留。

正常策略失败/S1拒绝是研究结果，不关闭无关任务；同一候选同条件有效失败不重复到成功。新候选、预声明不同seed或修订科学变量是新实验，记录全部分母。工程中断可在状态清算后补一个明确REPLACEMENT行，原行仍存在；一个条件最多一次工程replacement，不以安全失败为工程中断。

安全/完整性故障停止受影响domain，保护已合格独立分支。硬预算/截止到达则持久化当前模型/data/ledger并收尾，不自增预算。资金规模变大不意味着推迟每个检查点的真实行为评价。

不再把完整public HF交付、全部任务成功、四worker或每份旧文件重新回读作为第一项合法训练/评价的前置；私有固定revision或已授权本地不可变数据经过实际reader即可消费，远端归档另列。数据采集与审计/导出并行，模型读取冻结的vN，采集只写vN+1。source环境并发1->2->4、target1->2，仅在完整进程/ROS/时钟/输出隔离与实际CPU可用后提升；GPU推理服务串行化/排队并记录实际延迟，不把旧固定臂2.86倍速度外推整机。

任何专家privileged对象/阶段字段只能用于监督与审计，不能进入学生输入。保留teacher接管ON的结果与最终policy自主结果的边界。低loss、833窗口、独立视频解码、一次T1成功分别是不同层级证据，不互相替代。

## 11. 首次绑定、报告与结束

现役唯一owner先读AGENTS/START_HERE/已部署统一入口、最新#11/#47，登记`MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json`：实际T0、旧run/job清单、旧used/reserved、MP1新增余额、site许可、各lane负责人/lease、固定模型/data/precision/capsule。共享NFS checkout只有一个修改者；可并行读和隔离运行，不建立第二套入口。

第一项产物为`CURRENT_SCIENCE_STATUS.md`，把START_HERE顶部的过期E0排队文字明确降为历史，新增真实E1首成功与当前模型/数据状态，保留原文和所有历史链接；不以本文发布替代actual adoption。

阶段完成或原生job状态改变才发关键进展；至少每6个活跃工作小时汇总一次实测进度/累计余额/下一已放行行动，不在相同阻断上三轮重复查证。不要求ChatGPT在当前会话后台执行；监督、检查和通知仅是现役Codex在合法运行窗口内的工作，不创建永久daemon或cron。

最终首页给出：每个固定模型的source TRAIN/DEV/独立确认结果、target source-only/few-target配对结果、数据版本/成功episode/纠正anchor数、是否真正DAgger、WBC是否冻结、真实安全故障、工程缺口、GPU/CPU/物理/空间账及全部自有lease终态。模型和数据私有，GitHub只交自研源码、脱敏报告、hash/revision与视频逻辑索引。不能以新计划、更多测试或新GPU作业替代研究成果。

**本次发布新增三份授权/执行文档；服务器GPU、训练、物理是否执行必须由后续原生收据证明。本计划优先承接第一条已经成功的策略，把预算用在重复性、双任务、纠正收益和目标迁移，而不是重新启动一个无期限Teacher工程。**
