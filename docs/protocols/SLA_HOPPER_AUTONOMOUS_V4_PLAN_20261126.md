# SLA Hopper 自主长程推进 V4：高吞吐后训练、完整任务与多技能迁移

2026-11-26（SGT）。**计划发布；未迁移数据、未提交 PBS、未启动训练或仿真。** 用户最新要求：迁回 Hopper 后训练，采用多 worker 加速微调。执行端自主修工程问题、推进有限科学工作块，不逐步等待人工 GO。

配套：[执行手册](SLA_HOPPER_AUTONOMOUS_V4_RUNBOOK_20261126.md) · [机读登记](SLA_HOPPER_AUTONOMOUS_V4_REGISTRY_20261126.json)。继承 [V3][v3]、[V3 执行合同][v3run] 及 [V3 原47格][v3registry]；后者成绩、权重、任务、噪声和已启用人口不变。新工作包不覆盖历史失败，不将旧计划容量当实绩。旧 [Hopper 20K][oldplan] 和 [Core N2/N4][n2] 作为现有实现/资源约束来源，V4 调整后续优先级，不重开 CLOSED 作业。

## 1. 事实基线与科研决定

本次从 GitHub 固定版本实际读回截至 13:36:28 SGT 的 V3 汇总/JSON、12:40 H200 现场回执、原 H200 工程与源码。没有本轮实时 SSH/PBS 访问；以下不是现在仍在运行的断言。

| 证据 | 实际结果 | 决定 |
| --- | --- | --- |
| V2 MAIN3.8K+原 L1+冻结 WBC | 移动 HOLD 开发2/4，独立确认启动2/4，N02双噪声成功；PLACE抓持4/4，lift/hold3/4，入区2/4，完整0/4 | 已具备基础技能；只一条兼具抬升与入区，不能称完整PLACE已成功 |
| PLACE 逐卡归因 | P01/101、P02/101搬运不到位；P01/102入区28ms无承托；P02/102最高10.85mm，承托44ms，脱离2ms再夹回；两布局教师完整通过 | 优先学完整搬运/放置阶段，不靠消费延长强制开爪 |
| V3 MAIN250 | 52条（16 PLACE/36 HOLD）、4450查询、44500个50ms消费区间；真实反传已过；19可训练数组；两次GPU1中止0更新，GPU0后继待资源；第二次账本terminal=false | 先对账，再迁唯一待运行的250试验；保留已有数据/优化配方 |
| V3 PT250 | 4条水果TRAIN覆盖3几何、127查询、6068有效H50标签；6张新教师（含2 DEV）全过；实网络mask/reload通过；159/250更新、5088访问；8卡未启动 | 健康任务完成原端点，后续PT迁移按实际证据；不另开重复250 |
| 原 H200 MAIN | 3900 COMPLETE更新，最高完整phase03800（140/140）；3.9K有127文件、126长度完整、无COMPLETE；NumPy写入0字节，事后仅5.14MiB配额 | 不以3.9K恢复；保留已付100后缀与失败文件；不自动清理 |
| H200 A/B | 各从3.8K独立500更新、两份4.3K完整端点、Go2推理包均到Core；旧接口跨布局A抓持1/6、B0/6、完整均0/6 | 可低成本同L1复测，不重复训练，不串成4.8K |
| 原三域课程 | Go2 TRAIN-052已实际消费；TRAIN-069的新增17示范在MAIN/A/B访问0；公共源历史TRAIN为1309条/36任务 | 扩充实际训练课程与目标技能评估，不只扩大方块位移 |
| H200/站点状态 | 12:40项目账号无可见运行/排队；当前余额UNKNOWN（认证过期） | 恢复真实授权/配额后开工，不猜余额、队列或主机配置 |

来源：[V3进度][progress]、[机器账本][progress_json]、[视频事件][video]、[H200只读][hopper]、[3.8K与A/B][actual]、[公共数据][scale]。

**方向：SLA-base多源后训练仍为长期主线，官方PT目标适配为独立快速迁移研究线。** 完整PLACE是近期目标，但不是新技能数据、零样本测试、训练性能工程的共同前置。已有 HOLD 与冻结 WBC 阳性意味着该架构可行，不证明可靠泛化或实时部署。1×视频里的53–60秒入区是仿真时间，不是GPU推理延迟。

## 2. 三站部署：优先恢复一条高效H200 learner

| 站点 | 职责 | 不做 |
| --- | --- | --- |
| Hopper | 单卡H200训练、数据管线、短性能试验、正式端点、原生推理包导出 | 不同时开多个独立optimizer争同一GPU；不默认迁MuJoCo/ROS/WBC |
| Corelab | 现役PT结算、V3学生、教师数据、N1→N2→N4采集/评价、轨迹归因及回放 | 不因迁移中断健康任务；不在GPU1反复尝试相同OOM |
| Vanda | 经授权的数据/父端点/备份中转、完整性验证；独立有真实GPU负载的评价可选 | 不重启七条已停止的CPU-only PBS守候、不申请闲置GPU伪装CPU任务 |
| AMD共享基础设施 | 仅既有合法登录/存储 | 不在AMD做训练、重物理、批编码或扫全盘 |

首项并行做Hopper权限、存储和完整父包准备；Core继续原PT和已就绪原生评价，不等待跨站长训。新任务按可行性逐批推进。各站真实权限、实际资源与单一训练身份owner必须成立；不通过绕过站点、sudo、NFS/SSH系统改动解决问题。

### 2.1 迁移 V3 MAIN_PLACE250（第一正式任务）

- 核对现场 V3 01/02/03 的进程、更新、封存、实际调度与 terminal。若03已健康开始或250已封存，不强行中断、不重复训练；优先完成该身份。只有确证无活跃/未知owner，待启动后继才迁Hopper，技术尝试保留原账本。
- 复用已封存52条兼容命令轨迹、4450查询、44500真实消费区间、19参数数组及step0；传实际V3 reader/codec/mask/采样/optimizer代码、数据哈希、固定随机序列、250步LR曲线与新optimizer初始状态。旧MAIN140数组作为父身份/完整资产，不将历史Adam误带入V3新optimizer。
- 训练范围不改变：动作专家+Go2目标IO（以实际19叶为准），冻结VLM/视觉/语言/其LoRA、L1、WBC。有效batch32，250次正式更新，8000次样本访问，PLACE/HOLD=70/30；PLACE子阶段1120/840/1120/1680/840访问，HOLD2400。继承原任务/消费标签合同。
- 性能工程权重独立隔离；正式从原绑定step0重新开始，若已产生合法正式更新只能从完整正式封存恢复剩余更新。无完整端点则按真实旧后缀成本/新attempt处置，不能从0重跑冒充原进度。
- 选择最快合格后端（§3），不因未取得大batch资格而延迟短训；参考路径可落回原batch。原250端点、4卡PLACE、2卡N02回归、条件3个新布局资格+6卡确认全部照V3结算。PLACE/HOLD移动80mm等原谓词保持；PT站姿足漂门不混用。

### 2.2 PT 路线

从现场回读159之后的实绩，健康Core任务在原处完成250，原CMD0/CMD250八卡在Core运行。若中断且存在完整可恢复状态，先恢复当前合法后继；确需迁Hopper则迁该完整PT状态与原随机/游标/配方，不能从MAIN包推造PT。PT新后继仅在同codec闭环、raw意图与TRAIN覆盖证据支持后登记；不默认追加步数、不重复叠BTF/GRA。后续新PT训练优先Hopper，复用性能方法但单独核实模型合同。

## 3. H200 加速：多CPU worker、单GPU learner、真正batch

旧H200 E1的FM/aux microbatch=1/1，16更新、暖态p50 16.26s、p95 17.56s；E2–4的FM4/8/16总梯度差0.789/0.806/0.813%，target分项约6%；E5/6 aux2/4总差5.366/4.672%，均0正式更新、数值REJECT。原容差是分项loss相对1e-4、分项及总gradient相对L2 1e-3；输入已匹配，不能靠总梯度抵消掩盖单域差。[旧工程实绩][engineering]、[分项JSON][engineering_json]、[原计划][oldplan]。

已存在 [HopperEngine][engine] 的UpdateInputs：4个线程、每域锁、两步预取；批FM还在每microbatch做device_batch，辅助每组同步。**不是从零新造调度器。** 先剖析实际GPU利用、host准备、H2D、FM、aux、同步、保存、等待；若实际V3入口不同，按真实代码复用对应机制。

### 3.1 Pipeline 设计

1. 单一 sampler/learner持有global step、domain/phase、绝对visit、RNG和optimizer。主进程提前分配固定样本序号/seed，worker只读数据、解码、tokenize、构造mask，不能自己抽样或更新模型。
2. 先4个CPU worker和2个完整batch预取；比较1/4/8个worker，资源允许最多8、队列最多4。避免同一个mutable reader被串行锁或线程不安全抵消收益：按只读shard独立reader/tokenizer/解码器，按原序归并；避免fork已初始化CUDA/JAX进程。原数据字节/PTS/mask/样本身份、消耗顺序、跨恢复游标不变。
3. 每域整batch一次上传，设备resident ring buffer双缓冲；只缓存当前/下一batch和固定资产，不将全部数据盲塞显存。固定真实noise/tau与state dropout（短训按V3、长期按原配方）。减少逐microbatch scalar readback，仍在提交边界等待真实设备完成；不把异步enqueue时间当吞吐。
4. 单一learner按域JIT真实batch前/反向，短训effective32，长期64/32/32=128；保持exact logical batch、loss、mask、全局clip与一次Adam。异构H40/H50/H10各自处理，不能把源achieved/OSC和目标command同一归一化。
5. 冻结VLM前缀缓存是条件性加速：只缓存真正独立于action noise、tau、可训练参数、dropout/augmentation和prefix/suffix交互的计算。短训冻结VLM，优先8真窗验证loss/梯度/固定推理与reload。长期若VLM LoRA参与更新，禁止缓存依赖它的输出，只复用经证明独立的冻结视觉子图；缓存预算超过收益则关闭。任何统计/相机/权重/processor/dtype变化均失效。
6. 优先JIT持久缓存、固定shape、合法buffer donation/重计算、受控数据预取；FP8、缩短采样步数、删减aux、增大effective batch、多个独立GPU learner/MIG都不是本次默认加速。若另有真实第二卡资源，可跑独立预登记对照，不同owner、数据/模型身份，绝不超配一张卡。

### 3.2 有界性能资格与推进顺序

同配方同父同输入同实际随机张量，先用当前真实参考路径；旧H200性能拒绝保留。每个profile最多8个工程更新（首步及次步warm，后6步计时），首步不合格0提交，实际尝试及成本全记；缓存先固定8真实窗无更新验证，再进入候选。短训最多5 profile/40更新：REF、FM4、FM8、FM16、最快合格+cache；长期最多7/56：另加aux2、aux4。短训原实现若有额外loss组件也逐项检查。worker 1/4/8各16真实logical batch只读准备，不计训练更新；实际资源可缩减，保留分母。

采用原1e-4/1e-3数值等价门（短训新工程门在本计划预注册，不事后改旧门），分别查有效mask、归约/梯度与有限值；至少一次完整保存/冷恢复、下一固定输入更新和推理数值资格。性能胜出需相对**同配方同H200参考**暖态p50至少1.25×，目标1.5×；同时报告p95、峰值、端到端含I/O时间、真实提交更新/小时。候选不合格不能因速度快被选入V3。没有达标就用当前最快已合格且稳定的参考，不阻塞正式短训/长期。选取只看工程、非DEV成绩。

若已证实是正常浮点/编译路径而非科学语义变化，但旧门仍失败，可以另立NUMERIC_BATCH_VARIANT：从相同父做匹配固定短前缀，先封存容差和实际loss/梯度/学习/原生比较；不在已登记V3或原20K中无声放宽，不把旧拒绝改成PASS。至少证明稳定训练与任务行为再考虑推广。该分支不占上面等价工程容量，另登记有限后继。

## 4. 长期多源后训练：不机械地从3.8K续跑旧16.2K

### 4.1 近期端点与候选筛选

完成原V3两短训的独立闭环结算，继续冻结成功L1。并行复用已有MAIN3.8K、A4.3K、B4.3K做同L1小面板：两个原PLACE开发条件×新噪声601/602，每模型4格、总12格；旧跨布局负结果保留。三个候选权重均冻结，同一新噪声银行与实际输入合同，后续实时观测自然分叉。若该结果已经被新同条件实绩严格覆盖就引用，不重跑。原V3面板优先，不等12格全部完成才开始新任务数据或训练。

选择长期父：在首个长期正式更新前，优先已完整结算V3且PLACE至少一次完整成功、N02回归2/2的MAIN_PLACE250；否则可用完整开发阳性且同N02两噪声新回归2/2的A或B（A先于B；若尚未就绪不等待）。都未就绪时直接从已验证MAIN3.8K开始。若首个端点后发现更好父，不中途换权重，另建后继。候选父身份必须有完整共享/各域参数可恢复。A/B如被选须额外最多2张N02回归。3.8K沿用V2阳性。

若选择V3目标短训父：将已绑定19个改变叶按精确映射装回完整三域父树，核其余所有叶未变；不能从仅29数组的Go2推理包恢复三域训练。无法精确重建则回落完整MAIN3.8K。新增V4课程使用新optimizer/RNG，从选定父重新建立完整所有域训练态，不能把V3或3.8K旧Adam与不同参数树拼接冒称原谱系恢复。

### 4.2 正式长期主线 MAIN_MULTISKILL_V4

首个1000更新端点，条件延续至同谱系累计2000更新（不另加2000）；后续5K/10K/20K是未来有证据的端点候选，不自动提交。选择新训练目标：以当前实际兼容数据、TRAIN-069及以后实际READY新技能/纠正示范提高任务与几何多样性，保留公共源能力。若新技能尚未合格，可先从现有兼容数据启动，不宣称已多技能目标训练。

| 项 | 新V4首批提案 |
| --- | --- |
| Global FM | Go2 64 + RoboDojo 32 + LIBERO 32 |
| 各域FM均值权重 | 2/1/1，保留旧配方定义及外部aux系数 |
| Go2辅助 | 同每步4槽、十步生成与全程反传；仅用完整真实消费支持的监督窗，按实测标签重建 |
| 参数 | 原共享动作专家、原VLM LoRA与各域IO；视觉骨干/语言原基座/WBC冻结 |
| 精度 | 继承原BF16冻结权重/FP32训练参数、梯度/Adam；原25%状态token dropout |
| LR | 新optimizer，peak取3.8K账本最后核实LR，warmup100步，cosine至0.1peak，总2000步；固定序列，1000后不重新warmup |
| 数据 | 固定同一批2000的成员/采样/统计；新数据在新明确版本冷边界接入 |
| 目标采样 | 有合格新技能时旧任务60%（内部HOLD40/PLACE60），新技能40%按任务均分；旧/新池内按真实阶段→episode→query分层；没有新技能时旧任务100%按同内部分配，实际变体先绑定 |
| 新目标标签 | 真实消费COMMAND、同当前query固定frame/time；尾缺时间mask进入noisy input和loss；aux整段需实际支持，不把旧nominal后缀重命名执行 |
| 源语义 | RoboDojo名义future reference，LIBERO原OSC；各自decoder、clock、stats、mask与真实episode边界 |
| 端点 | raw 1000、条件raw 2000；完整训练态、推理包与固定离线/原生评价 |

首1000的正式访问64K/32K/32K、4K完整aux；累计2000为128K/64K/64K、8K aux。不得把工程/失败或旧祖先访问当本轮成果。新配方明确改变了目标数据、采样、optimizer和标签支持；相对历史端点的变化只解释为整个新课程收益。

对于完整H40辅助：先建立可证明真实连续消费的40点窗；从已登记目标batch的64条中选四个完整支持槽，采样银行预先确保有四个互异合格槽（明确其实际影响），不能按模型分数挑样本、用末点复制、删后缀或把不完整窗当全监督。若缺少完整支持，先修reader/补连续轨迹，或者另立显式仅FM配方；不冒称旧辅助原样完成。其余部分支持窗的mask不影响有效目标。来源见V3 Runbook。

### 4.3 多源学习的真实对照

从与MAIN_MULTISKILL_V4完全相同的已封存父及新optimizer起点，启动条件性的 **TARGET_CONTROL1000_V4**，相同1000个目标batch、64K目标访问、4K aux、相同目标随机张量/时间/阶段分布、LR前缀、目标参数集合与新COMMAND标签；不加入公共源损失，source-only IO不更新、Adam/decay按实际闲置规则不动。两个1000端点用同L1、同任务、同噪声评估。源评估按真实源域统计/任务，不将源误差当Go2原生成功。

这回答的是**在共同已有多源祖先上继续加入源监督的增量效果**，不是“从未接触公共数据的纯目标模型 vs 公共预训练”的净历史因果。总样本/计算量不同，单列成本；若要等计算量或从raw-base比较，另建匹配研究，不拿旧TGT5K的不同codec/配方冒充干净对照。控制臂不是长期主臂开训前置，只有真实可用资源才调度；没有配对不宣称多源因果增益。

## 5. 新任务：从三个技能开始，而非扩大同类方块

当前公开多源资产已有RoboDojo 48 TRAIN/50979窗、LIBERO 1261 TRAIN/209146窗、合计1309条/36任务；历史TGT2K/5K源域MAE变化不一致，都是离线指标。[数据实绩][scale]、[源域误差][source_eval]。先读实际source任务清单与已消费索引，把新任务分别标SOURCE_SEEN、UNSEEN/UNKNOWN（不能由LIBERO名称假定抽屉/推已见）。Go2 TRAIN必须确认无对应新技能示范才称目标零样本。

| 技能 | 原生任务目标 | 首批独立评价 |
| --- | --- | --- |
| PUSH | 在允许接触定义下将指定物体推入目标区并稳定，撤离 | 2个几何×2噪声 |
| CONTAINER | 从未抓持起点抓取不同形状物体、放入指定容器、稳定释放、撤离 | 2个几何×2噪声 |
| DRAWER | 从关闭或明确定义初态将抽屉打开规定行程，安全撤离 | 2个几何×2噪声 |

Teacher 先用同冻结WBC完整路径资格（最多每技能2个零学生模型DEV资格）；实际scene、原生资产、动作空间、允许接触、目标物体/关节位移/稳定窗/截止和移动/站姿判据必须在首次学生输出前封存。新任务允许声明任务合法接触，但原WBC、安全硬门不动，不放宽旧任务。无合适抽屉资产或完整物理路径则保留BLOCKED；其它技能继续，不能偷偷换更简单任务算同一面板。

冻结MAIN3.8K为第一新任务目标零样本候选；条件复测其他已有权重（官方PT仅在同物理任务与自身合法codec支持时启用；新适配模型须核无目标任务训练）。候选最多3个×12格=36格；固定选择为原MAIN3.8K、原V3 PLACE250（可用且未见该技能）、官方冻结PT（兼容才启用）；不因测试成绩更换候选。若不同移动/站姿profile不匹配，另标栈比较、不得声称配对。独立教师资格先做，每个新模型从正常入口单初始任务指令，至少2次真实返回消费及自身运动后fresh观测，禁止Teacher接管算成功。

零样本失败也可推进少样本：每技能首批最多10次TRAIN教师尝试（预封存布局/reset，失败全保留），优先至少5条合格完整示范且≥2个独立TRAIN几何，随后按需要扩至10/20条合格示范（每波最多再20次尝试，另注册；不把尝试当成功数据）。新技能的TRAIN/DEV/CONF按几何与episode分组，不能仅换seed；不复用已评选轨迹作训练仍称独立。已存在兼容资产可减少新采集。先用一套共享多任务政策，避免每技能独立模型；少样本比较若开启，固定5/10/20嵌套TRAIN子集、匹配端点与目标暴露，先小样本250更新，样本效率与真实成功分别报告，控制/端点/容量另注册。

若初始新任务已能完成，记录冻结跨技能迁移并扩独立布局；若raw行为不足且教师可行，补真实命令监督；若raw有合理动作但消费/物理错误，回到实际接口/构型归因；若Teacher不通过，修任务的物理开发，不让训练承担不可执行路径。

长程组合从两个已有单技能阳性开始：同一初始语言指令下连续两项操作，加入同场景改变指令顺序和目标的对照，报告阶段正确、顺序、错误对象、重复、终止、完整任务及fresh feedback。先测试单SLA端到端，不用硬编码阶段或Teacher GT推进；仅当独立技能已可执行而顺序/记忆持续失败，另立显式层级planner/历史输入对照。长程多技能不是短期PLACE成功的前置。

## 6. 多环境加速：采集和评价，不在监督训练里多开物理

旧N2八张匹配工程卡N1 1128.48s、N2 566.24s=1.99×，状态7/8通过，F2瞬态7.658mm超5.999mm，交换STOP第二组缺peer继续证据；**未资格生产**。[N2实绩][n2actual]。已有[Core计划][n2]的36核心N2/28条件N4资格，不重新造64卡。现场先读回是否已有后继；复用已验证部分，只补当前版本实际缺口。N1继续收数据/学生评价，不等N2或N4。

N2通过同钟状态、原控制实际selected-reference、全任务接触、STOP/故障隔离、world间路由和有界资源后才扩大采集；N2≥1.5×N1、N4≥1.25×N2且合格完整轨迹/小时不下降为旧性能目标。N4仅真实收益/容量支持后启用。每个world独立WBC/MPC/ROS与非ROS IPC、渲染、状态、RNG、STOP、存储、owner；统一catalog单writer。停止一个world后该world新publish/install/grant/integrate均0，peer必须真实继续。若用户态无法隔离，保留N1，不能放松安全或复制锁名。

学生多world后继采用**单SLA policy server**按同checkpoint/codec/shape批推理，独立world query/obs tick/noise/TTL，有限等待、实际返回匹配、禁止跨world状态或history缓存；batch推理先在固定真实输入核对同卡原profile，再独立原生对照，教师N2不自动授予学生batch资格。新H200训练只做offline supervised learner，不要求加载多个MuJoCo；后续DAgger从TRAIN布局采真实学生偏离与可执行教师纠正，保留失败前缀及mask，策略版本固定在episode边界，不在线改健康worker权重。若要把actor迁Hopper，另做实际站点CPU/渲染/权限与隔离资格，不默认占用训练卡或改变backend。

## 7. 速度评价、资源、自动后继

V3录像的1×sim-time缓慢要区分模型墙钟推理、参考编译/ALIGN、SOURCE推进、WBC真实运动。复用逐tick日志，报告reset至首grasp、grasp至入区、原完整task时间、fresh queries、SOURCE实际时长/占用、ALIGN与实际替换比例、路径长度和与同任务Teacher路径时间差；同时报端到端wall、冷启动/热态推理、GPU等待和RTF。先保留原P10/TTL/2ms/安全/任务deadline；速度调整若改变查询或相位合同必须独立profile和配对，不能加快播放/缩短成功窗冒充加速。

工作包首轮排序：T0现场对账与Hopper资产/授权 → H200短训性能及MAIN250 → 原V3完整评价；并行Core新技能资格/数据/N2缺口与已有4.3K小面板；H200长期1K在合格数据和实际资源就绪后启动，不等待其他线。每个完整端点回传Core评估，H200可在已登记下一段继续，不等全部评价。2000条件延续需要已证实的行为改进或TRAIN/独立DEV显示可解释的未收敛且无明显HOLD/源域退化（首1K阶段固定面板；不按最好中间步挑权重）。若raw改好、native未改善则修相应层，若相同失败反复无新证据则换数据/诊断，不盲加步数。PT结果独立判断。

原20K剩余逻辑容量16200以3.8K计；真实3900中的未恢复100仍计成本。V4的新250及1K/2K不是原20K续步，旧320 GPU小时/2026-12-05 12:41:52 SGT期限不能靠改名重置。现场复核原剩余实际授权和用户最新范围，不自行增加站点余额/延长期限；无有效资源时照实报告并推进Core。每个PBS capsule有实际合法walltime，旧最多6h且保存预留15min，按实测p90和保存成本决定分段；完整状态每100步或30min先到者、端点必封存；无足够配额停止新启动而不删用户历史文件。详见Runbook。

首批V4新增正式长训容量：MAIN 2000累计+可选匹配控制1000=3000更新（V3两个250另计且不重复）；性能工程短训≤40、长期≤56，实际发生逐项记录。新增Core条件槽位共118：4.3K比较12+候选HOLD2，长期端点面板32（含同父基线6、三个训练端点各6，以及两个1K端点各4张同新技能配对，另见Runbook），新技能资格6+冻结学生36+首批TRAIN教师30。实际执行只启用对应就绪条件；机读表逐格登记。V3原47格与旧N2/N4资格分别继承，不把它们重复计入本批。所有新增容量为条件性，不是阻断其他就绪科研的新人工审批制度。

## 8. 自主执行与交付

执行端按 [Runbook](SLA_HOPPER_AUTONOMOUS_V4_RUNBOOK_20261126.md) 的开工指令复用已有入口，工程修复按V7自主执行。每块只需：实际输入版本、唯一变更和预测、真实训练/物理/访问/资源结果、原任务成功与首失败、下一有限工作块。旧结果、STOP、所有技术尝试保留；不要从汇总伪造原件。只发布可公开的小报告/索引，私有数据/权重/日志沿原批准通道保存。用户负责清理，无全盘扫描/删除。

首要可交付：真实PBS/owner与完整恢复；H200 worker与batch性能实测（或说明回退）；V3 MAIN250端点及真实完整PLACE、HOLD回归；原PT250与八卡净学习对照；新技能首批冻结迁移成绩；新长期1K真实完整端点和阶段/源域数据访问。发布文档、性能fixture、Teacher成功或loss下降都不是学生完整任务成功。

本次仅GitHub计划更新；没有新SSH/PBS、迁移字节、模型前向、积分、optimizer更新、真实机器人动作。最新站点实绩仍须执行端读回。

[v3]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/a0eea81a7c2b7989a3acfac2a3c419cc8cd073bf/docs/protocols/SLA_V3_PLACE_AND_PT_PLAN_20261126.md
[v3run]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/a0eea81a7c2b7989a3acfac2a3c419cc8cd073bf/docs/protocols/SLA_V3_PLACE_AND_PT_RUNBOOK_20261126.md
[v3registry]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/a0eea81a7c2b7989a3acfac2a3c419cc8cd073bf/docs/protocols/SLA_V3_PLACE_AND_PT_REGISTRY_20261126.json
[progress]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b1c5ae70e1d36ccddf2805d1ffa3391916113d3e/reports/SLA_V3_PLACE_AND_PT/PROGRESS_20261126.md
[progress_json]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b1c5ae70e1d36ccddf2805d1ffa3391916113d3e/reports/SLA_V3_PLACE_AND_PT/PROGRESS_20261126.json
[video]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/95712ce3ecb3c37ff0785cde6f105e29e12dbc66/docs/media/sla-v2-native-20261126/README.md
[hopper]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/95712ce3ecb3c37ff0785cde6f105e29e12dbc66/docs/media/sla-v2-native-20261126/H200_READONLY.md
[actual]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b6869988b90851baf0e4ce207df3985fef1d2dc7/reports/WM02/sla-main3800-native-diagnostics-20261122/ACTUAL.md
[engine]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b6869988b90851baf0e4ce207df3985fef1d2dc7/scripts/sla/hopper_engine.py
[engineering]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/reports/WM02/sla-core-hopper-parallel-20261121/ENGINEERING_SELECTION_AND_MAIN_MOUNT_ACTUAL.md
[engineering_json]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/reports/WM02/sla-core-hopper-parallel-20261121/ENGINEERING_SELECTION_AND_MAIN_MOUNT_ACTUAL.json
[oldplan]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/171b5bedb290258954aac140157743e92e3b152e/docs/protocols/SLA_CORE_HOPPER_PARALLEL_V1_20261121.md
[n2]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4fe2c8cf1b69c168b2e3079e061bb7c702089cd3/docs/protocols/SLA_CORE_MULTIENV_COLLECTION_V1_20261122.md
[n2actual]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4d597259/reports/WM02/sla-d2-teacher-parallel-20261118/PERFORMANCE_PARALLEL_ACTUAL.md
[scale]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/c8642aa602d3e34dd4dcda5a55ff6ae7c7c53ae6/reports/WM02/sla-scale20k-20261119/DATA_AND_RUNTIME_ACTUAL.md
[source_eval]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/c8642aa602d3e34dd4dcda5a55ff6ae7c7c53ae6/reports/WM02/sla-multisource-goal-v2-20261120/TGT2K_5K_PAIRED_AND_MIX_PUBLIC_WATCH_ACTUAL.md
