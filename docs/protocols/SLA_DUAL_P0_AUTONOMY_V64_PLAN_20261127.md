# SLA V6.4 · 双P0自主接力：Hopper ADAPT与Corelab多环境数据工厂

2026-11-27 SGT。用户确认 Core 多环境资格提升P0，与 ADAPT-A250并行；要求长程自主推进。本文为[V6.3 #526](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/526)的增补，继承[#521](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/521)、[#425](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/425)、[#522](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/522)、[#523](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/523)、[#524](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/524)的原人口、身份、安全、预算与真实权限，不重新批准旧失败或扩大现场权限。现场入口 [RUNBOOK](SLA_DUAL_P0_AUTONOMY_V64_RUNBOOK_20261127.md) · [REGISTRY](SLA_DUAL_P0_AUTONOMY_V64_REGISTRY_20261127.json)。该PR只发布执行合同，不启动现场任务，不覆盖健康capsule。

## 1. 最新固定证据

[Corelab 采集实绩 e45d857](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/e45d857e2528808882389c6e0c1ea4b1bb7a22c7/reports/SLA_CORE_EMBODIMENT_FACTORY_V5/FIRST_SITE_ACTUAL_20261126.md) · [选参考证据 d2dc185](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/commit/d2dc18589f0b8f9226e871d99d31608dcf34e7d9) · [Hopper实绩 d6f21ab](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d6f21ab0b3eee835a1d5bae931b04aa37df2104f/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V63_ADAPT_A250_ACTUAL_20261127.md)；实际提交/worker修订也已读回。此为GitHub快照，不是实时qstat/SSH；以现场更新的真实终态覆盖旧快照，不重复已完成条件。

| 角色 | 实绩 | 决策 |
|---|---|---|
| Hopper 736456.hopper-m-02 | H200 143771MiB/12CPU/申请225GB/2h，6工程更新，最近正有限梯度1.80679，正式0；资格STEP0006约6.82GB，初始化到Orbax提交354.05s（SOURCE资格52.40s），冷恢复进行中 | 原job终态和独立恢复通过后复用资格，不重跑六步 |
| A74 | 52旧+22修订，新增2094窗/2028完整H40/82440支持点，44组边界图像实读；总6544窗，五条最新全纳入，真实tokenizer形状通过 | 固定A74/3K银行；75号不热换 |
| Core | 75/100，最新005蓝左放成功，132窗/129完整H40/5220支持点；006 F2下探link6碰红色目标，-0.1003mm，STOP后0tick | F1蓝左放扩充多样性，F2根因重查，不盲搬干扰物 |
| N1吞吐 | 前四波8请求5准入，HOLD 7.64条/封存h，PLACE 3.20，合计4.92；含整个19952.165s跨度0.90/h；PLACE含两波间空档0.55/h | 主KPI为每计划生产小时合格数，分解主动占用/封存/排程空档；不要把波内速率当全天产量 |
| 资源 | 20可用核、可用RAM约52.35GiB；20s全cgroup抽样0.883平均核、峰值3.97GiB | 潜力足，先量全回合peak与调度；不拿短采样外推N8 |
| selected reference | 3720实际50ms消费原子，1860有两种候选安装内容；实际冻结观察点地址绑定、两条真策略消息41/80结点哈希对，冷态零积分 | 先一张有界N1真实观察卡，确认实际选择/时钟/干扰，再N2 |
| Recovery / SRC608 | 仍以V6.3最后实绩为边界，原生接管0/准入0；608库存、SC-VLA H30实际训练资格未证明 | 并行准备，按实际新回执更新 |

新第三波2启动/1完整、57549tick，三波PLACE累计3/6合格，第三波未报告精确wall，不能重新估算三波合格/h。A74新增实际读取不等优化访问。N1观察证明的是低层MRT实际活动命令/解身份，不能由此倒推历史标签全部错误：原R4真实发出/已消费50ms命令与低层MPC选择是两个层级，分别对账；历史歧义只标不确定，不选一个看似合理的solver消息伪造低层证据。

## 2. 双P0调度与有界推进

P0-H：Hopper先完成A250正式短训（如现场已经正式启动则核状态、健康继续），端点Core评估后按V6.3门推进A500/A1000；SOURCE500保持HOLD。P0-C：Core多环境资格，当前优先一张N1实际观察技术卡，N2双卡1+1，条件N4 2+2；合格N1持续生产，P0-C技术卡使用有限world窗口，不另开竞争owner。P1并行：完整操作、Recovery、608 reader/bank、冷场景/编码。P0优先不等于抢占健康回合；已运行正常封存，按原教师与学生公平调度，避免工作互相阻塞。

执行窗口从现场实际接续起先按0–6h（阶段I）、6–24h（II）、24–72h（III）规划检查点，非强制完成时间或无条件启动许可。原Core V5 72h有界工作窗/原#425协调期限、H200原截止2026-12-05 12:41:52 SGT及320GPUh/实际更严格配额不续期；所有后继只在原剩余授权内运行。阶段IV在72h真实复盘后、原剩余期限内走已放行后继；不足权限/配额时封存并明确下一项申请，不自动扩大。普通可逆工程修复在既有许可内自行推进，不逐项请求人工；遇真实安全/权限/泄漏/预算硬阻塞停止受影响角色，其他合法工作继续。Vanda保持暂停。Core负责数据、评价、并行仿真；Hopper只GPU训练与直接训练资格，不派CPU-only守候。实际可写/配额/封存余量硬约束，原V6.2取消项目自设固定储备门仍有效。

## 3. P0-H：从已完成资格到正式A250

### H0 收尾与唯一后继

固定A74增量manifest e3487b9322a6ce7e795a69a596c762a775e7475af3752d6d0142ae5f37d3f0fd，Core读回 9197b9cf3c85e2a6d5f72fa702672df5c34cfd823236b10a1ff30f1145cb098e，Hopper入口 bed65589f9986f8f3ef4d29e51a3a01da4cdb98896ff0ab99cfda31a18853f19，3K bank 1ce0abe57d403f93fa25f71ad8268389c80a4cfbb8bbaa5b499e3db3820b0fa0。实际冻结原R4-200全父/processor/codec/stats、新Adam/独立RNG、固定bank与微批，首250预定涵盖74条、5482不同Go2窗；实际唯一窗/episode/事件曝光单独报。75号新蓝左放标签清单 955ad71fe4ddeb8e47fc184b2a3c3b8fc49678a8b4d546dfd73f64102d52af69，只进入后续冷版本，A74不热换。 

先回读736456 qstat/终态、原A250 capsule及 qualification/TERMINAL.json、UPDATES.jsonl、STEP0006成员、实际GPU内存峰值/limit与恢复预测等价；不能把写完checkpoint等同PASS。若原作业已进入正式run，不申请重复A250；若已部分正式训练，封存并从最近完整科学端点合法恢复，核不可恢复消耗，不用资格入口重跑0→250。若原作业F且因余时不足在合格冷恢复后退出、正式run目录不存在、严格符合[d6f21ab提交入口](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d6f21ab0b3eee835a1d5bae931b04aa37df2104f/scripts/sla/v63_adapt_submit.py)的after-probe条件（原job名/状态、QUALIFICATION_INTERRUPTED、6条日志/同bank/restore prediction_equal、确切余时错误文本及所有成员封印），才使用该修订已支持的--after-probe入口和独立code-formal-r1/A250-FORMAL-R1 capsule。实核worker完整course匹配、前资格峰值<0.9实际容量、归一化和父绑定；不能绕过任何require。若恢复不成立或实际失败是别的原因，明确归因，不假造stderr/回执；若剩余工程更新支持且确有新证据才有限修复/补资格，否则暂停此角色。复用的是工程资格，不是6步工程权重。正式从原R4科学step0开始；工程序列6步只记一次。新增提交回执、PBS ID、现场启动后独立读取，禁止未知状态重投。SOURCE250原晋级false不变。

### H1 写入瓶颈和正式步数

354.05s对52.40s说明本次资格checkpoint边界慢约6.8倍，但不是独立磁盘带宽测量；初始化/写入/提交/冷恢复各分段计时，绑定实际卷、Orbax shard数/大小、P50/P95、峰值/临时重叠、CPU I/O/PSI，比较相同完整保存合同。首正式PBS沿已授权2h建议、原最多6h和至少15min保存余量；根据实际剩余wall、最新稳态更新、完整save+restore时间和最坏合理尾延迟在每次定期保存前判断，不靠旧SOURCE耗时外推。若不足完成下一科学段，停于已登记完整端点并释放GPU；不得为了赶时间减少应保存的optimizer/RNG、推迟关键封存或删旧权重。原每100步或30min先到及250完整端点，恢复时验证实际step/cursor/Adam/LR/RNG、全叶与固定输入。只在冷边界改变存储策略，复用合法本地卷和原安全封存；不得并行两optimizer或热改活动代码。新保存加速若有实质瓶颈，作为有界单变量工程验证，不占首A优先级。

### H2 正式课程、端点评价和晋级

保持128 FM +4真实完整H40 aux；Go2 64（旧32/其余修订24/最新8），Ville16，RoboDojo32（旧16/扩16），LIBERO16；aux旧2/其余修订1/最新1，aux系数1/4；micro Go2/RoboDojo32、Ville/LIBERO16、4 reader/2预取；peak LR 8.481070829156122e-7、warm100、cosine3000。原state dropout、FM/aux物理含义、冻结模型/执行器、L1、2ms/50ms和原门不动。首250计划32000 FM+1000 aux，其中16000 Go2；不把工程曝光、bank预定或重复窗口计独立任务。

端点以V6.3固定24反馈、42旧源DEV、N02两张、PLACE四张原配对（P01/P02×701/702）独立回读；A250→A500→A1K每段6张，同门；A1K追加原独立HOLD四张。保留旧R4与SOURCE250，不重复已完整回执。目标是HOLD2/2、PLACE正确抓持4/4、无新增安全STOP及重复抓前拒绝、抬升≥3/4、入区≥2/4，且完整≥1/4或一个明确阶段进步；其余阶段不退化。旧源同统计回合等权MAE相对A-init恶化>10%暂停查原因（告警非物理门，来源初始化差异单列）。逐窗区分raw H40/P10、接缝40mm/s与块内40mm/s、实际消费、实际释放及撤离；不能以预测开度/目标到达替真实成功。没有实质进步或安全退化则不自动延长训练；保留负结果，选明确的数据/事件/观测修复，不能盲扫LR、改阈值或换seed。若A250过门但A500/A1K未过，最近已通过端点可作为独立后继父，失败端点不可自动晋级；如果任何原生卡未授权或缺失，仍未结算，不用离线代理替代。

A250/500/1K是同一冻结A74 3K bank的累计更新，后继用完整Adam/RNG/LR cursor，非三个独立从头短训。A75新数据不修改已资格银行，封存入下一冷版本。A1K后若稳定：优先评估原独立HOLD/已资格CONF及新数据是否真正提供新的任务/恢复监督。若新增A100/Recovery首10达到真实reader与实际支持，优先明确A100-R10冷版本，固定父、目标曝光/源回放和事件比例；计划改bank/stats/树时重新登记，树/统计不兼容显式新optimizer与日程，不伪装同一精确恢复。原A3K/条件A5K不是默认跑完；只有证据支持、实际旧预算有余且Core门通过才申请。采用A100-R10等新谱系的正式更新也累计入同一V6总量。若只采集更多同质HOLD而Recovery无进展，则不自动把额外训练当科学收益。SOURCE250初始化的匹配ADAPT-S仅在旧可选控制预算内另作必要对照，非优先PBS。

## 4. P0-C：真实选参考 → N2 → N4

### C0 一张真实N1技术观察卡

原冻结二进制9e8ce46837d09bc39ae04d4b49f1ce4dd8c8c2d31bea39eac56c5806d108c1f3；台架ef6a7636f34d8546b8b0c8740a582fa373a016a289b6c6bb8b28c5c3232d9e79、实际地址绑定b5d04e0128fdce1f7986aa0c43f73eecd632295d758fc6d9b097bade956012f1，歧义只读068ab13fb87534c202acee0a36f7324f7447d150a14a17b3544d4132b34aa5b8，真实两策略台架96e5574bcdc30da1f4d7edcafeaa9628bbeb5df4c25420d7ffbd837321fed575。现场实际符号/二进制/ABI/地址再次匹配；不得将台架地址或内部结构偏移用于其他版本，不写控制器内存/寄存器、不改变策略选择。使用原合法owner、原已登记适合短工程的固定N1卡（优先原#425 N1匹配基线剩余槽，必要时独立有限登记；非新额外人口）。一张技术卡、按所映射原最大tick/wall，先固定相同已知参考、合法初态、对照/阈值与输入哈希，零学生模型前向/零训练，不计TRAIN。没有合法槽或可比原对照则先补授权/基准，不自造成功。旧回调等待不变。

在实际evaluatePolicy调用点采集活跃命令与解的身份；每个实际调用/实际控制步以world/episode/reset_epoch/控制cycle/仿真tick、host monotonic_ns、安装/发布序号、selected payload hash、解时间轴、执行输入/实际消费、STOP关联。2ms积分与50ms原子分别对齐，明确控制调用cadence；不是每2ms都一定有新policy，保持原生语义，复用活动身份必须有实际证据。全目标范围内零无法解释的未匹配/跨world/逆序/误归属、零非法安装/STOP后积分/新增发送；如有边界未观测，单列不通过，不能用visualizer或最近消息填补。保留两条真实候选同时存在时选中哪一份及因果顺序。对比未插探针的同一固定历史N1基准（身份/任务/硬件有可比性才用，缺失则另按旧槽取对照），原状态/消费/终态/时钟门不变。完整记录每次断点暂停、p50/p95/max、全卡wall、额外延迟、调度抖动/TTL、实际cgroup CPU core-s/峰值RSS、观察丢失与开销。建议生产观察开销≤10%同工作量墙钟且不引发任何原控制门退化；这是开发性能目标，不放宽原门。如果探针只证明真实身份但开销过大，认定诊断可用/生产未资格，开发低干扰只读trace或有限缓冲，单独同合同复核，不能直接把重断点扩到四world。原始WBC冻结；任何改二进制或控制计算路径的方案另协议。观察者异常必须可安全停止，不允许因日志写失败影响其他world或隐藏缺证。该卡结束后正常N1采集继续，P0工程不无限占用生产。

### C1 N2双卡1+1资格

只在C0实际同钟/干扰通过并核原36请求预算余量后，复用[#425原资格人口](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4fe2c8cf1b69c168b2e3079e061bb7c702089cd3/docs/protocols/SLA_CORE_MULTIENV_COLLECTION_V1_20261122.md)与V5 1+1拓扑；历史1.98x、6/8状态和两组STOP记录保留，只有同拓扑/版本/实际证据等价才映射，不自动认定通过。未通过的同钟安装点/mount差异必须在现役实现按原容差归因。每world独立进程树/cgroup、4个互不重叠核（其余CPU为scene/encoding/审计）、MuJoCo/MPC/WBC/leg/ROS及非ROS IPC/共享内存/cache/tmpfs/时间/RNG、渲染EGL UUID、lease/STOP/ACK/output；只读资产共享，单一pool/catalog writer，不复制锁绕限制。原8张短匹配、原N1基线、角色交换的4 STOP/4故障隔离、4完整接触及原其他绑定按原名册，不重开已完成结果；检查全状态序列（EEF、mount、base、qpos/qvel、object、gripper、控制与真实选参考），原短卡同钟容差、实际接触与任务顺序；不得平滑、时间平移或改阈值。STOP后被停world新增publish/install/grant/integrate均0，peer至少按原登记继续；故障注入仅针对本登记worker，禁止影响其他用户进程。原N2合计36/645308tick上限，已消费/可映射槽现场扣减；任何缺口标NOT_QUALIFIED，不能增新seed凑数。完整数据合同和实际selected-reference均通过才测收益。原端到端吞吐门≥1.5×同期匹配N1；完成后可开始有界N2生产，不等N4。

### C2 N4与后续扩容

N2稳定且真实收益达门、全回合资源余量足才2+2，建议CPU4×4=16核、保留4核后台，2 GPU各两个renderer；按实际NUMA/affinity/cgroup/UUID、整机CPU/峰值RAM、VRAM、PSI、磁盘写入与所有在途封存余量确认，低平均核不等峰值。继承N4 28请求/429684tick（8短匹配、16STOP角色覆盖、4完整接触），先映射实际旧额度；端到端≥1.25×同期N2且原质量/公平窗口不下降，才能N4生产。N2/N4混合学生推理另按V5 P4/P5原有限资格，不用Teacher-only通过来推断混合可用。失败退回最后合格拓扑，不影响Hopper；全局串扰、STOP异常或失去真实可写条件则关闭受影响池。普通科学接触STOP仅终止其隔离world，peer按原门继续。N8/更大只作为后续条件研究：N4证明CPU/IO/吞吐瓶颈后，测完整cgroup peak和CPU调度尾延迟，尝试2核/world的事前配置；必须另登记有限拓扑与真实安全/资源额度，N4的合格不授权N8。不以冷态128 MjData渲染替物理world。

### C3 生产性能与调度

以实际N1 PLACE 3.20条/封存h为当前参考，N2门1.5×即约4.80、N4相对N2门1.25×即若N2恰达门约6.00；只有匹配同类任务/布局难度及时间口径才比较，8–12/h为后续stretch而非已实现。合格/波内封存h、合格/整个排程h、各任务成功/请求、独立几何、有效事件数分别报。第一批真实匹配生产至少20尝试或V5各族首4的原规则，在实际原人口余额内，报告配置、全树CPU p50/p95/max core-s/peak RSS、GPU/EGL与推理占用、WBC/advance/ACK/等待/渲染/编码/标签/传输各独立耗时与等待原因；不把多个worker耗时相加当并行wall。生产阶段用统一catalog写入、预编译待派发名册、每world有界缓冲，CPU离线准备并行。保持原学生等待≥30min或12教师终态等公平让行规则，未获混合资格则排空Teacher后单world DEV6，不因追吞吐长期延迟A250验收；混合资格过后1S+1T再条件1S+2T。调度应减少无意义空档但不把实际必要队列等待/站点限制归零。先收集端到端trace，依次单变量：真实selected ACK event-driven代替可证冗余150ms等待、完整warm reset（进程/solver/时钟/STOP/RNG/缓存/场景都重置，不能遗留控制状态）、异步双RGB编码/标签（2→4 CPU worker、每world≤2捕获缓冲，背压不丢帧）；每项遵循V5已有≤4匹配工程启动、总≤12原预算，未验证不进入正式采集，不同时改N2根因、物理步长或评分。不要为了更高产量省略真实WBC或接触检查。

## 5. 数据工厂：75→100→200 + Recovery40

75来自69 legacy+3 Mobile HOLD+3完整PLACE；Wave100剩25=Reach/Orient8、MobileGrasp5、DiversePlace5、M_FREE Container4、Push3。当前第三波蓝左放132窗/129完整H40/5220点的准确标签回执为 955ad71fe4ddeb8e47fc184b2a3c3b8fc49678a8b4d546dfd73f64102d52af69，现场必须从原报告实际完整回执复核；A74固定。优先补任务、颜色、目的区、相对几何及恢复事件，不靠容易HOLD冲数量。F2本次蓝块偏移后碰撞对象从干扰蓝块改为目标红块，旧597状态3.400mm冷筛只证明反事实静态条件；先从真实新失败前缀归因下探手腕姿态、全臂/指垫/目标净空与控制/参考实际消费，分开搬运持物扫掠；在原V5可用资格槽做单变量合法新Teacher候选，保持原碰撞体和安全门，不继续重复旧失败几何。已有合法F1成功模板继续，不因局部F2停全局。M_FREE Container按原独立任务合同先验证释放后目标-容器支撑的合法接触角色及真实终态，不修改原通用安全或站姿失败；Push先原任务资格。Reach/Orient不充完整PLACE，完整HOLD与恢复片段分别计。

Recovery先原首10目标/最多20尝试中的两张独立TRAIN试点，真实学生prefix、安全偏离、教师从当下状态重规划接管、实际消费、完整或局部纠正终态；历史62候选/2预览只诊断，不复用历史DEV或从STOP续积分。先抓后搬运/已入区未支撑/支撑后仍闭爪/释放后撤离；按原V5组扩到40目标/80请求总额，保留所有失败与技术终态。只有已实际消费的教师纠正为标签，原生接管成功不等学生自主成功；仅部分纠正标RECOVERY_PARTIAL。所有数据按source episode/场景group/任务/teacher版本去重，原双RGB、实测EEF/上一消费夹爪、真实H40 mask与R4逆重锚、低层真实参考记录独立，实际读回后才计READY，TRAIN与DEV/CONF/TEST不混。每4新条或6h有变化或波末封存，Core/Hopper readback/实际优化分别记账。A75→A100→A200及Recovery快照可并行准备，不改A74运行bank；旧V5 424完整请求、80恢复、各族QUAL/VAL与旧#425人口实际映射、不重复计。200之后的500/1000只是科研规划候选，需基于新吞吐与任务覆盖另行申请，不在本轮自动授权。

## 6. 608源课程、后续科学长程

SRC608=499 RoboDojo H50/25Hz/14有效动作+58 Ville H40/20Hz/7+51 SC-VLA H30/15Hz/7，109单臂/六类；544694窗，固定合并指纹165f388dabc4f6d5ecd147f709c1749ee726daa820a828dc560ef985f936877f。保持原native语义与各域TRAIN统计，SC-VLA未来实测状态不是实际消费命令。H30当前固定H40 role拒绝，先Core CPU实际reader/tokenizer原生B×30×32/noise/mask/位置编码/逆变换/真实有效loss归约，H40/H50及LIBERO原H10回归，确认source版本/许可/去重/受保护split。然后仅在首A正式端点结束且实际剩余工程预算与授权允许时H200有限真实梯度/冷恢复；不把CPU形状检查或6次试读当GPU资格、不以假后缀绕过H30。V6工程24更新中已知SOURCE6+ADAPT6=12，最多剩12（如有其他消耗现场扣减）；只做最小必要资格，不重复A74六步。新bank仍128 FM+4 aux，SOURCE608=16 Go2/32 Ville/32 SC-VLA/32 RoboDojo/16 LIBERO，ADAPT608=64/16/16/16/16；旧SOURCE500永不因队列空闲自动恢复。源新课程必须等已验收A端点（优先A1K，或V6.3允许有明确行为进步的A250/A500冷边界）与608真实资格，另立SOURCE608首250新父、新optimizer/RNG/LR、实际配比及源DEV固定名册；各源独立≥16真实DEV窗并复用旧42窗，做同统计/同输入配对。首250过相同目标保持门和可解释源进展才续500、1K；更长至5K/条件10K不是默认。若A阶段失败则先补目标任务/恢复数据，不以新source掩盖。若源学习提高但目标任务退化，停止SOURCE608接力，选择最近已验收目标父或匹配控制，在真实预算内研究联合/交替课程。正式长程目标：同一权重在Go2 Reach/Grasp/Place/Container/Push保持，独立Drawer/Insert/组合先zero-shot（源见技能仅称目标域未见），不得用CONF训练或脚本串联充语言组合。完成后再条件少样本5/10/20与多任务独立验证；严格跨本体结论另需第二本体冻结高层权重，不由Core任务数量推出。保留原冻结WBC/L1/动作语义/无GT学生，低层训练不在此计划。

## 7. 预算与自主状态机

已知主线旧SOURCE250正式250，A74资格6另计，正式A0（现场更晚即更新），原V6默认8000/条件15000总更新不重置；最多7750/14750余，工程24已知12→最多12，旧可选控制1000、历史R诊断800独立保留，实际项目GPU小时、截止、租约/原生科学槽都从最新账本扣减。主线学生原V6.3条件剩50（若无更晚消耗），A250/500/1K各6+独立HOLD4=22槽；SOURCE608三个短端点各6=18，理论合计40，余≤10给原已登记其他最终/CONF映射，实际名册不一定能逐项映射，不能因算术可容纳就自动授权；3K/5K/10K或新A100-R10需要同一余额内明确槽与评价预算，否则停并申请。完整生产/Recovery/N2/N4/性能优化原容量分开，N1观察卡优先映射原36；N2 36/N4 28不得以新计划重置；所有尝试/拒绝/技术停止/已支付/未启动记录原分母。例原250+A1K+SOURCE608 1K=2250，默认余5750；若A3K且SOURCE608 5K，累计8250超过默认上限，最多4750 source或真实条件扩大后才续；绝不把端点步数相加当多次预算。

唯一owner维护只增Journal：READBACK→READY→SUBMIT_ONCE→RECONCILE→RUNNING→SEALED→INDEPENDENT_EVAL→DECISION；各角色独立状态：BLOCKED_LOCAL/WAIT_RESOURCE/WAIT_EVAL/STOPPED_SCIENCE/COMPLETE；提交unknown先核qstat/lease/capsule，绝不复制提交。自主执行只在原合法有限范围，运行健康job不取消，不因一支失败停全部。每4新示范/端点/波末及最多12h发布事实；每6/24/72h检查依赖、吞吐、累计GPU/工程/科学/原生槽、后续实际授权。72h复盘：至少交付A250完整端点与六卡结算或真实阻塞，N1观察/N2资格/生产或具体未通过项，75→100实际新数据与Recovery试点，608真实reader/训练资格状态，实际job ID与封存/剩余任务。不是承诺在时间点完成。结束前安全封存、关闭不再需要的角色lease、交接原期限内尚需执行的合法后继；不创建无限守候或自动扩大范围。

本次仅计划与代码入口审阅，没有直接启动/取消PBS、训练、原生仿真、数据上传/转移/删除或扩大任何站点权限。未实时读取服务器，所有更新须现场对账。
