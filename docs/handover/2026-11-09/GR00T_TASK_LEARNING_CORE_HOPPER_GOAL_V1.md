# GR00T + frozen WBC：CoreLab先验证、Hopper训练的长程任务学习Goal

日期：2026-11-09，Asia/Singapore。
Goal ID：`GR00T_TASK_LEARNING_CORE_HOPPER_V1`。
发布状态：`PLAN_ONLY_NOT_DISPATCHED`。发布文档不创建现场scope、不提交PBS、不启动模型或物理。
审阅基线：现役PR #366，`cad80fbc2a9e3f70288b56967f196731612c452d`。
关联：PR #368稳定起点、PR #369目标GR00T训练。保留其有效代码、资产、历史结果与未完成作业；本文替代“21个起点窗口自动长训”和“单卡Hopper仍必须另等H200科研审核”的后继优先级，不回退执行源码。

## 0. 用户指令、授权变化与交付目标

用户明确要求：制定详细长程goal，Hopper已恢复可用，先CoreLab调通，再提交训练；计划推送GitHub并提供启动prompt。

本轮允许现场Codex在合法资源和新有限scope内完成：用户态必要修复、CoreLab真实采集/训练接口验证、单卡Hopper GPU资格与目标GR00T动作侧训练、完整恢复/交付、Core闭环评价，以及下文门控的DAgger与轻量RL后训练。无需每次常规代码修复、合格数据批次或同协议checkpoint接续再次询问用户。

**单卡Hopper/H200训练已纳入本轮用户方向，不应继续被旧`AWAITING_H200_SCALE_REVIEW`科研停点循环阻塞。** 但“服务器恢复”不证明当前项目token、队列权限、额度或GPU allocation有效；必须现场核实。只更新新实验的许可声明，不能手改旧lease或绕过站点规则。

仍须人工审核：修改冻结WBC/腿策略/控制增益/solver数学/启动规则/安全阈值/机器人动力学/2ms步长；新增GPU物理多世界、超过既有两CPU环境、分布式/多GPU；实机/自主行走；解冻视觉语言骨干或切换基础模型家族。单卡Hopper上的合法神经训练与普通数据并行处理不属于GPU物理扩量。

不设项目内部累计工程预算，也不承诺耗时或成功率。每个job仍须有真实许可、固定端点、合法walltime、存储余量和收尾；长期goal由合法新job接续，不是无限job或复活CLOSED。没有覆盖续作的有效站点权限时，仅阻断受影响分支并报告，不能虚构授权。

最终交付不是“更多更新/更多PASS”，而是：
1. 一项有任务意义的目标接触技能和可重复环境；
2. 覆盖连续决策及真实恢复状态的数据；
3. 一个完成非零GR00T动作侧学习、完整恢复和Core重建的候选；
4. 冻结源策略、SFT及一个证据驱动后训练方法的真实独立对照；
5. 明确结论：收益来自示范覆盖、可观测性、参数适配、交互监督还是回报优化。WM不作为上述链路前置。

## 1. 本次核查到的事实：不要重复已完成工作

以下是00:15 SGT封存记录，不是当前服务器实时状态。[R1]

| 项目 | 已有证据 | 不能声称 |
|---|---|---|
| 目标数据 | 21 TRAIN决策窗口、7 Teacher DEV、TEST未开；数据/统计身份固定 | 数百张RGB不是数百个独立策略决策 |
| Core07 | 32微批次、1次optimizer调用、checkpoint1；该次LR=0；第二步Qwen投影OOM | 不能声称已验证有效学习或完整恢复 |
| Vanda1532527 | 一张A40、FULL_ACTION_SIDE 1,304,913,028参数、2调用/cursor64、保存1/2 | 后训练推理被训练噪声审计hook误拦；PBS F/exit1，恢复未通过 |
| 钩子修复 | `0aa0ab5e7963be5a5e770b4d0c1ddf8fb68f3779`有CPU回归 | 还没有真实GPU恢复资格 |
| Hopper | scratch依赖和19个模型资产已准备，CPU导入通过，约25.95GiB逻辑存储 | 尚无目标数据/Processor、有效Hopper训练profile、GPU加载或PBS提交 |
| Hopper站点 | 上次项目token过期，PBS后续连接失败；另一个同账号job曾Q | 用户说恢复后必须新读回，不猜旧job终态或取消它 |
| 冻结执行 | Teacher双周期短接近有成功；独立启动在任务前已出现差异 | 不能把初态失败归因GR00T，也未证明完整抓放 |

固定身份：
- GR00T父manifest：`94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38`。
- vendor：`51d4c89f72fda44cbf77285c6a8114b52676b8a1`；Cosmos revision：`9ce19a195e423419c349abfc86fd07178b230561`。
- 旧目标数据：`cf80dd4c40f93bf6a7e904c70809237b108e7aca3c46d38df18deacbece006e3`。
- 旧目标统计：`08d0f1b79cd60d5773c928b1f4231edad00fa19300ea7ac24f97a31f03a50420`。

保留旧训练、旧192次WM负结果、源域S-CORR任务成功及BOOK失抓反事实；不要把历史多轮学习改写成“从未训练”。既有21窗口现在明确用作短任务接口/FIT诊断，不能改旧角色或冒称新接触技能数据。

## 2. 科学假说与不再进行的无效循环

H1：在稳定执行器上，连续决策和关键事件覆盖比重复起点窗口更有效。
H2：单帧观察若无法区分接触/支撑/恢复状态，增加训练步数不能可靠消除歧义；应先检验可观测性。
H3：已具备部分能力的SFT策略，可以通过学生状态纠正或实际回报进一步改善，而不是无限离线BC。
H4：WM只有在同信息、同实际交互预算下增加任务收益/学习效率时才部署；线性响应模型也是模型，不要求神经WM获胜。

禁止：同一21窗口无新证据地自动2000→4000→8000；反复复制37xx源码文件或重下载不变父模型；混算optimizer调用/实际参数变化/任务成功；用未执行后缀伪造动力学标签；改变相机、统计、参数范围、时钟和奖励后仍称单因素对照；通过放宽安全或筛掉失败获取成绩。

## 3. 总体DAG与角色

```
G0 唯一owner/新scope/真实站点读回
  ├─ G1 Core代码/内存/非零更新/保存恢复验证 ─ G1H Hopper单卡canary ─┐
  └─ G2 稳定环境/合格Teacher/可观测性 ─ G3 连续决策数据封存 ────┤
                                                              ↓
                           G4 一次有解释力的SFT与完整模型交付
                                                              ↓
                           G5 Core冻结基线/候选真实闭环归因
                                      ├─ G6A 学生状态DAgger
                                      └─ G6B 冻结GR00T轻量RL
                                                              ↓
                           G7 新TEST确认/最终决策/下一任务
                                                              ↓
                           G8 可选WM增量（不是前置）
```

- local-PC/既有现场agent：唯一控制owner，协调manifest、job、数据/权重交付。复用原Journal/registry/onceIO，不再建设一套调度平台。
- CoreLab：真实MuJoCo+冻结WBC、稳定性/Teacher/学生rollout、实际RGB与状态、CPU审计、局部训练接口和原生评价。新物理任务先1环境；已有2环境只有当前隔离和资源资格仍有效时才用于同协议采集。
- Hopper：合法单GPU神经训练、保存恢复和离线评价；不默认迁移物理仿真。数据与模型只通过已获准私有传输路线交接。
- Vanda：保留原终态与资产，只读核旧job；不与Hopper重复训练同一run，不抢停别的任务。若实际资源条件要求回退，登记新作业归属，不能两个站点同时写同一checkpoint。

## 4. G0：启动与新授权绑定

1. 读取最新PR #366/后继执行PR、AGENTS及实际代码diff。文档SHA只定位计划，不能把最新执行工作树reset到审阅SHA。
2. 核所有相关owner/worker/job/lease的真实终态。1532527最新证据为F；若现场更晚结果不同，以真实回执核对。旧日志中的705946不属于本goal自动可取消对象。
3. Hopper SSH别名从现有私有配置发现，不猜host/account/queue；凭据只复用已有授权，不打印、不提交、不索取secret。
4. 新读回项目资格、queue/account允许项、剩余额度、walltime上限、compute节点GPU型号/UUID、scratch配额与CPU内存。`df`剩余量不等于个人/项目quota；无法验证额度时走站点官方接口或报告确切缺项。
5. 为单Hopper训练实现窄范围`HOPPER_SINGLE_GPU`新profile（名称为设计，不是现成CLI）；当前profile只允许Core/A40，必须明确新增站点并测试，不把Hopper伪装成Vanda。
6. 单卡Hopper神经训练的旧科研审核停点由本轮指令解除；保留站点准入与新finite scope。原WBC、安全、数据split、其他项目任务不改。
7. 现场选用允许的有限操作窗口并记录绝对截止；跨窗口通过既有合法renewal/新scope接续。最后预留至少20分钟或站点更严格余量封存。授权/计划/站点许可/实际dispatch分别记录。

输出：在已有报告树创建一个本goal状态文件，包含EVIDENCE_ASOF、实际执行SHA、当前owner/jobs、授权差异、next_gate。不要每个小测试生成新的campaign。

## 5. G1：CoreLab先调通，不把16GB训练能力作为无限工程课题

### 5.1 必须修复的具体问题

- 训练专用`NativeNoiseAudit`只在原训练loss路径启用；generation前解绑/暂停并用finally恢复，异常时不遗留hook。测试训练→生成→训练、保存→重载→生成，以及两个模式RNG互不污染。保留原失败，不只是吞掉assert。
- 验证Core07第一步LR=0这一事实；资格必须跨过warmup零学习率，至少一次非零LR且目标权重确实改变。optimizer调用数不是有效更新数。
- 16GB上的Qwen未使用语言logit投影仍分配约594MiB。先核输出确实未被动作路径使用，再研究feature-only路线或冻结特征缓存；必须保留原视觉隐藏层、位置编码、dtype、mask与动作输出语义。不得直接升级vendor/main或全局monkeypatch。

### 5.2 Core分级资格

C1：真实数据/Processor/动作编解码及CPU梯度/模式测试；复用既有已验收哈希，不重复全量传输。
C2：真实GPU固定输入/噪声H40推理与原路径比较。声明相同设备/图精度时要求原有精确规则；更换运算图/硬件不能临时用更松容差过关，应在运行前单独冻结数值资格。
C3：在真实已部署网络接口上做`4连续step`与`2step→完整保存→新进程恢复→2step`对照，包含非零LR、optimizer、scheduler、RNG和sampler；检查冻结参数不变与目标参数变化。不把fixture更新记为GR00T更新。
C4：旧FIT4仅做200步，必要时既定500步，验证已知数据上的动作方向/夹爪/姿态，而非宣称新任务泛化。诊断权重不自动作为正式父模型。

Core可用接口+末2层配方验证训练逻辑；完整动作侧参数范围在Hopper另行资格。冻结backbone缓存只在输入/增强/模型/精度身份固定且已证明输出、动作侧梯度一致时使用；这不证明原始全模型驻GPU训练可放入16GB，也不能用旧缓存替代在线新RGB。

若真实Core全路径在16GB仍OOM：记录峰值与失败，完成可执行的feature-cache/小动作侧真实更新恢复及原始推理/物理接口验证；不再同样OOM重试。Hopper只先提交单卡容量与恢复canary，正式主训练仍等待canary通过。用户要求“Core先调通”指逻辑和最小可执行链，不要求1.3B动作侧AdamW强行塞入16GB。

**禁止为了通过Core资格删减图像、任意改动作维度、静默BF16全量存储或降低安全门。** 合法无损内存实现需独立版本与数值对照；不同参数范围必须独立命名。

## 6. G1H：Hopper交付、单卡资格与恢复

### 6.1 复用已有资产

复核已准备19文件及scratch环境的内容身份，不重下载17.47GB。核原vendor32块DiT/H40与父模型；上游main或RLPT当前结构不能替换它。将FIT4、旧21窗口及其原Processor按私有固定revision+prefix+SHA封存交接，确认数据/统计两端匹配。现场文件和大型权重不放公开GitHub。

### 6.2 提交规则

- GPU只在合法PBS compute allocation使用；login节点只做站点允许的小型预检/提交，不大规模CPU训练、渲染或GPU探测运行。
- job名称带goal/generation/seed/phase唯一身份。qsub只一次；未知返回先核PBS和原Journal，不盲重发。
- 首Hopper作业可只含canary+FIT4，不能等待新接触数据占卡空转。主训练须G3完成。
- 实际请求CPU/RAM/scratch/GPU/walltime由Core测量与站点允许项确定，不照抄旧project token或H200显存假定。一个goal至多一个主学习writer；Core物理可同时独立运行。

### 6.3 H200 canary必须验证

加载固定父权重、正确target category、32块架构、TRAIN-only统计；打印精确trainable名字/numel。先microbatch1，再最多对2/4/8做一次容量扫描，effective batch固定32。训练采用FP32可训练参数/优化器与已验证BF16 autocast策略，不以全局cast偷省内存。

跑真实非零更新与完整4-versus-2+2恢复；推理hook修复真实生效；导出后在Hopper和Core同输入/噪声核模型包。相同设备恢复与跨设备浮点一致性分开：跨GPU不承诺bit-exact，数值容差在CPU/Data/Teacher精度要求和同权重资格试验前登记，偏差不得改变安全/任务判断。

运行末尾保存：模型/父delta、AdamW、scheduler、scaler（适用时）、全部RNG、数据sampler游标、global step、Processor、参数白名单及源码/数据/配方SHA。用临时文件+原子完成标记；manifest齐全不等于恢复已实测。

## 7. G2：建立可重复、可观察的接触任务基准

T0：现有4mm接近/撤离和21窗口继续作为接口诊断，不作为最终科研能力。
T1：本goal主任务是**在稳定工作区内接近、抓持、抬起并保持一个物体**；T1通过后才扩到有支撑放置。开门/整理书架属于后继目标，不与首个训练批次混合。

### 7.1 新T1任务定义（设计，不是已有成绩）

优先使用当前真实已建模的物体/抓持几何，目标建议抬起20mm并稳定保持1s；正式运行前根据几何可行性和原安全检查冻结精确高度/时间/接触/姿态/BASE/掉落判据。若20mm不可行，必须在看到学生结果前另立任务版本并说明，不能事后降低目标。旧SHORT/BASE2mm等既有评分和失败不改；新任务也不放宽原机械安全。

成功要求物体而非仅TCP达到要求，抓持/相对运动稳定且没有非法碰撞、关节/机体安全违反。有限运行后按原允许的安全收尾处理物体；不得以成功标记后立即不受控退出产生掉落。

### 7.2 初态与Teacher门

预登记10个课程布局×2次独立启动，共20次资格；每个计入attempt，不能重复seed直到成功。建议门：初态合格至少18/20，合格初态上Teacher至少90%成功，硬安全违反0。门槛是本goal的研究选择，不是现有证明；同时报告两个分母和全请求成功率。

将任务前差异与任务动作后的差异分开。记录初始qpos/qvel、solver/control版本、参考年龄、BASE速度、高度、关键接触、controller readiness和首个偏离时间。位精确恢复未证实时只称匹配独立重启，不称反事实fork。

若因冻结启动/控制本身无法达到上述门，停止该任务族并提交定位证据；可继续CPU和Hopper既有有效数据诊断，但不把环境失败包装为VLA失败。不得反复加EEF反馈追源绝对姿态、写qpos或固定自由BASE绕过。

### 7.3 可观测性门

Teacher可使用特权物体/接触状态；学生部署输入默认真实双RGB、EEF/夹爪状态。对接近、已抓未稳、稳定抓持、滑移风险、支撑/释放边界建立TRAIN诊断集合，核近似观察是否需要相反动作。

若只有单帧不能区分，先做独立的短历史输入/可部署反馈表示：建议8个50ms本体/上一动作样本，必要时两帧RGB；保持缺失mask、真实时钟与训练部署一致。先训练和核这个新版本，不往旧state padding/语言里偷偷塞接触真值。历史版本一旦选定，SFT、DAgger、RL比较臂共享同一观察能力；不能将多传感信息收益归因算法。

视觉条件化检查采用预登记同状态换目标、同目标换位置、必要的遮挡诊断。它们是诊断，不把打乱图像的OOD成绩当任务泛化结论。若策略只输出常量小位移，不能晋级“视觉技能”。

## 8. G3：连续决策数据，而非零散起点窗口

### 8.1 起始规模与分区

| 人口 | 预登记设计 | 用途 |
|---|---|---|
| TRAIN | 25个新布局×4次计划轨迹=100次尝试 | 完整任务、阶段变化、初态变化 |
| DEV | 20个不同布局×2重启=40条件/方法 | 开发与checkpoint选择 |
| TEST | 20个封存新布局×3重启=60条件/方法/训练seed | 一次冻结后的确认 |

这是起始设计与科学端点，不是工程总预算或统计把握度保证。所有失败保留。首版正式SFT要求建议至少80条合格完整/合格恢复轨迹、20个TRAIN布局，且接近/闭爪/稳定保持/恢复等关键决策均有覆盖；不足时新增预登记TRAIN区组，不重试旧条件凑成功。新TEST不复用已经看过的T06、C01或旧DEV父场景。

80/100不能取代覆盖检查：Teacher动作必须随可观察目标/状态改变，不能只是同一脚本平移后复制；场景变化一次只按声明维度，不能无约束随机化制造不可学分布。

### 8.2 每次真实高层决策都记录，不伪造窗口

建议先沿已验证H40/50ms表示，选择固定、经过Teacher资格的执行前缀和重规划节奏。每次真实观察后Teacher重新决策，记录状态、图像、语言、raw intention、编译参考、安装/消费、真实结果、阶段/干预标记。不能只记录初始Teacher提案，后面27张图没有动作监督；也不能把控制器每次参考续约当新Teacher决策。

**标签必须绑定明确policy边界与时钟。** 优先将连续实际Teacher提交的名义意图映射到同一50ms策略边界，构成当前观察之后的H40行为序列；未来序列可含后续反馈重规划，明确为实际专家行为序列，不声称这些动作全在t时刻已确定。训练只使用当前及过去观察，绝不输入未来状态。

若编译器有重定时/偏置，必须保留逐动作逆向身份映射，不能将低层重复命令重新当高层原始动作。无法证明该映射时，先固定或重新定义新版本的时间接口，禁止直接拼训练标签。现有目标collector只能anchors==0的旧数据保留原角色。

另一种合法的DAgger标签是当前状态下合格oracle提出但未执行的动作；这必须独立标为`ORACLE_INTENT_LABEL`，前置核oracle能力和安全，不声称真实消费/成功，不用于WM动力学标签。不能把未执行的学生H40、模型预测或参考续约当专家标签。主SFT优先使用已核实的执行序列；是否加入oracle标签须预登记混合比例并单列来源。

H40跨episode禁止；短尾按真实mask处理，不能填下一episode。临界抓持/支撑状态的后缀若未执行，不能标为发生。所有时钟、图像顺序、rot6d、夹爪归一/工作行程、world/base/tool坐标只转换一次。

### 8.3 采样与统计

按完整episode/物理布局划分，TRAIN-only拟合统计。新候选绑定新统计，冻结源基线保持原统计。采用一个事前固定的episode与阶段平衡采样方案，关键边界/恢复建议占访问的25%，其余覆盖所有任务阶段；报告实际曝光，过采样不冒称新增独立数据。

若数据并未包含接近失抓/卸载的纠正，就不能把稳定握持后缀称为解决失抓。基础数据与DAgger新增数据分开封存；评分器、split、语言和学生观察在比较前固定。

## 9. G4：Hopper正式SFT，只运行能回答问题的模型

### 9.1 父模型和训练范围

B0：当前冻结源CORR2048，原Processor与统计保留。
SFT1：同一父初始化、新目标数据和统计、fresh optimizer，冻结视觉语言骨干，训练目标接口+完整动作侧。实际1.3B规模以运行参数名单为准，不靠开关名称。不是旧E2或book128的resume；不继续用历史失败父链替代未经说明的初始化。

先只跑一个主seed20261109。证明学习与原生效果后再顺序跑20261110、20261111确认，不能根据DEV改seed选最优。单卡Hopper容量不足时可另立末6层/末2层配方；不同容量配方不是同一run，且不同时展开大网格。

### 9.2 预声明配方（待原生验证，不是已取得结果）

- 官方原masked flow-matching主loss；首轮不加新物理loss、不混WM奖励，保留强基线。
- AdamW，LR1e-4、WD1e-5、effective batch32；实际microbatch从已测1/2/4/8固定。
- 8000步连续日程、warmup200；2000/4000/8000为合法查看/接续端点，不重置scheduler。
- 每250更新保存全态；每500在固定TRAIN/DEV上开环诊断；真实DEV只在2000及按规则选定的后继端点做，不为每个小checkpoint跑全物理面板。
- 所有H40/执行前缀、XYZ/SO3/夹爪、阶段边界、跨chunk同未来时刻一致性分别报告；不以一个平均loss晋级。
- 基础模型的历史任务回归只读评价，不把旧TEST拿来训练或选目标checkpoint。

旧FIT4/21起点样本只允许有限学习性试验，不能因Hopper空闲直接当80条级接触任务数据训练8000步。若G3未过，Hoppercanary结算释放GPU；Core数据工作继续。

### 9.3 接续规则

2000时TRAIN仍改善且DEV不退化可沿原日程到4000/8000；TRAIN已很好而DEV差，停止同数据堆步，进入覆盖/观察诊断；TRAIN拟合不过则只做一个隔离原因检查，不同时换数据/精度/头/统计。新数据generation固定新统计和配方，不能假称精确resume；故障恢复仅限同data/model/schedule。

Hopper的完整训练不依赖Core在线或PC持续SSH：由合法PBS脚本运行。数据齐备和code capsule固定后才提交长作业；网络断开保留job，独立读回不重放。墙时不足时完整step边界保存、退出，新合法job恢复，不挤占收尾或硬改deadline。

## 10. G5：Core重建与真实闭环，先区分任务错误与执行错误

模型包必须包括父链/权重、Processor、统计、模态、观察历史配置、动作时钟、训练白名单和dataset/recipe/code SHA。通过既有私有交付路线传Core；复用父模型只传合法delta。Core固定输入/噪声验证解码一致性、显存和实际推理时延后才进行原生执行。

开发面板：共同新DEV40条件，B0与SFT1各一遍；独立重启但初态分布匹配、顺序交错，所有拒绝保留。既有21窗口的旧DEV协议保持自己的名称，不将新T1结果填入旧任务分母。

H40为预测/控制预览；执行前缀和重规划周期在G2固定。每次新query必须有新真实观察，旧后缀取消与新参考安装清楚记录。暂停辅助先用于行为验证，非暂停实时性另立测量；不得刷新旧时间戳掩盖1秒级查询延迟。

同时给出：初态合格率、合格初态任务成功率、全请求成功率、首失败阶段、物体状态、BASE/关节/接触、安全拒绝、部署时延。只观察到动作朝正确方向不等于抓取成功，Teacher阳性不能记为GR00T阳性。

晋级后训练的建议机制门：DEV上至少出现20%实际完整技能成功，且原生安全/状态接口有可解释覆盖。这个20%只是本goal避免盲目冻结策略RL的门，不是算法普适条件或高可靠性声明。低于门且提案显著错误优先G6A；任务本身不可行回G2；不硬上DSRL期望凭空产生动作语义。

## 11. G6A：真正的交互式模仿，不再零碎加几条就重训

固定SFT1父与任务协议，在新的TRAIN布局运行学生；以任务进度停滞、夹爪卸载/失抓前兆、专家与学生分歧等已定义触发器查询Teacher。不得先让机器人越过原安全再纠正；实际安全始终由原机制执行。

一个DAgger批次预登记32个新TRAIN学生episode，至少覆盖8个TRAIN布局；累计至少40个不同真实接管/专家查询状态后再形成完整更新批，不每四条重训。若该策略几乎无干预，按实际记录下调“需要DAgger”的结论，不人为制造错误。

保存学生动作、oracle标签、接管前真实状态、实际执行主体、恢复结果。Teacher标签来自学生实际访问状态；未执行oracle标签与实际专家后缀分别统计。旧DEV/TEST状态不进入优化。

D-UPDATE：固定原基础数据重放并加入新纠正；初始建议原数据/纠正访问各50%，两类各按episode与阶段平衡。新generation训练2000步，fresh已声明日程和数据身份。

D-CONTROL：同SFT1父、同2000更新，仅原数据，作为“多训练本身”的对照。若要声称比同数量新专家示范更高效，必须另做等专家标注/交互成本的对照，不能仅凭此两臂声称采样最优。

首批有增益且失败类型仍可由oracle纠正，可按同协议再做一个32-episode批次；两批无关键行为/DEV改进时封存`NO_DAGGER_GAIN`，转向可观测性/执行归因，不继续无尽小数据迭代。

## 12. G6B：冻结GR00T的轻量RL后训练，WM不是前置

### 12.1 官方代码复用范围

将`nvidia-isaac/isaac-rlpt`的接口和DSRL/SAC实现作为参考，获取并固定实际commit，保存许可/依赖身份。它是research preview；其G1动作维度35、LIBERO维度7、执行视野或奖励不能直接复制。当前项目vendor32层与上游版本差异必须独立适配；不升级整个现役环境来凑依赖。若该版本没有原生验证所需接口，应如实标`RLPT_PORT_NOT_QUALIFIED`，不能以mock通过冒称可运行。

初期只实现`reset / observe / propose / step_chunk / reward / terminated / truncated / close`窄环境adapter；Core原MuJoCo+WBC为唯一物理来源。每次返回实际执行k、原生时间、原任务评分、安装/消费与失败原因。不要同时迁移Isaac Lab或改变机器人动作合同。

### 12.2 首选DSRL，动作残差仅作为明确后继

以同一个冻结SFT1为基础（不混用DAgger最好checkpoint），学习小actor选择flow noise；GR00T、Processor与WBC冻结。先核真实噪声张量形状/有效10维mask/其余padding/全部flow步，不能直接拿G1的noise_dim=35或简单声称随机种子控制等价于DSRL。保持实际有效噪声映射和未编辑padding的原规则。

每个动作chunk只执行G2规定的合法前缀。噪声编辑不保证物理编辑小，所有提案须原安全认证，拒绝和回退计入结果；不挑多个噪声直到成功后只记录一个。

首个新Core原生RL smoke：至少32个真实决策transition、真实actor/critic更新、完整保存恢复、旧策略不变及实际命令消费核对；fixture不计。然后预登记2000个高层决策pilot。若真实回报/任务覆盖有效而未收敛，可沿同协议至10000；不能按物理2ms tick计作10000个策略决策。三seed仅在pilot有可解释信号后顺序确认。

若冻结策略的可用候选明显不足，停止DSRL并回任务适配；不得靠无限noise搜索。SAC-edit/有界残差可作为下一命名实验，先对空间/姿态/夹爪/时间编辑范围取得真实资格。直接PPO动作头更新需先对固定vendor的Flow-SDE/log-prob/old-policy/实际变换进行独立验证，不用MSE伪造PPO ratio；该直接大模型RL不是首轮必跑项。

### 12.3 奖励与回放

主目标仍为实际任务成功，奖励配方在TRAIN/资格阶段冻结。保留原生成功/碰撞/掉落/BASE指标作为独立原始通道；训练奖励可使用明确标记的接触/物体特权真值，Actor不得偷用。

可用的起始设计：成功+1，非法接触/掉落/安全终止-1，其他0；零稀疏成功时可另立带势函数进度差的版本，而非在原run静默调奖励。势函数由接近、抓持、抬升等任务阶段真实量构成，终止处理防刷分；对无限闭爪、抖动、反复接近、停住避免终止、未抓抬TCP等投机轨迹做负例测试。训练return上升而任务不升判失败。

transition至少含`obs_history / nominal_chunk / latent_or_edit / installed_and_consumed / actual_k / raw_reward / shaped_reward / next_real_obs / terminal_reason / policy_version`。同一chunk内奖励按50ms基准累计，bootstrap折扣为`gamma^actual_k`；time-limit truncation、真正terminal、安全终止、初始化失败分开，不能把工程丢包当负任务回报。

### 12.4 不假定Core与Hopper可实时互联

默认Core本地冻结GR00T推理和采样，Hopper只训练learner；32～128决策一个带身份的rollout shard，learner更新后在episode冷边界回传小actor。使用已许可固定revision传输，监测策略版本和staleness；跨站稀疏更新是本项目异步off-policy变体，不宣称精确复现官方默认运行。

SAC可用回放但必须记录collector policy；队列或传输延迟令版本过旧时暂停新增采样/释放闲置GPU。无合法低延迟连接时不暴露新公网服务或改SSH。若Core小learner已能完整运行，就留本地，不为使用Hopper增加网络复杂度；重型SFT仍放Hopper。不能在离线旧数据上做几次actor更新就声称完成在线RL。

## 13. G7：独立确认、路线选择与成功/失败出口

最终候选只从DEV选择，冻结模型包/任务/观察/执行节奏/奖励后开启新TEST。主要比较B0、SFT1、一个有效DAgger候选、一个有效DSRL候选；未达到前置门的方法记NOT_RUN，不伪造全矩阵。强Teacher/简单状态反馈作为独立能力/信息基线，特权输入明确标注。

每方法每训练seed：20新布局×3重启=60实际条件；有3训练seed时分别报告，不能只展示最好seed，B0只需相应共同评价一次并共享其原始结果。物理布局是聚类单位，60次不当60独立布局。给出逐布局成功/失败、初始化覆盖、安全终止、任务成功率和布局聚类不确定区间。

研究目标建议：达到至少80%目标短接触技能成功，同时在共同评价上相对冻结源基线有有意义增益；这是目标，不是预期保证。方法增量以事前登记的实用差（建议10个百分点）和置信区间报告；小样本不足以支持优势时记`INCONCLUSIVE`。若开发中基线已接近上限，必须在开TEST前声明改测恢复/鲁棒性子集，不能看TEST后换主指标。

保留`P(reset_ready)`、`P(success|reset_ready)`和`P(success per requested episode)`。硬安全触发按原机制停相关动态族、保留未运行，不降低约束完成面板。独立root/复位未通过时暂停物理比较；已经完成的有效负例不重试。

有稳定T1后再开启T2有支撑放置，复用已学能力与真实事件监督；开门需要单独门轴/接触Teacher、任务定义和工作区资格。不得将T0、T1、T2成绩混算。

## 14. G8：WM只在清楚的应用场景上做增量

本goal前七阶段不等待新WM；旧线性/神经checkpoint和正负结果保留，不再重跑18k同类训练。

可选出口：同一合格任务策略、同一历史/候选/交互成本下，WM能否预测简单反馈看不到的抓持丢失、支撑转换或状态相关全身响应，并改进实际任务。先前1614→16×8不是完整可递推状态，不能直接用作SuperMimic长想象。WM预测不能当实际传感器，特权状态不进入部署Actor。没有增量则不部署，不影响GR00T+WBC主线交付。

## 15. 最小实现清单、成本与无人值守规则

### 15.1 修改范围

优先复用：`scripts/wm02/train_target_gr00t_v1.py`、`src/go2arx5_vla/world_model/target_gr00t_profile_v1.py`、`target_gr00t_dataset_v1.py`及现有target runtime/采集/回执函数；入口名称必须现场确认，本文没有声称新CLI已经存在。

需要的窄新增能力：Hopper实际site profile与PBS包装；training-audit hook生命周期测试；Core内存资格；连续决策Teacher数据版本；任务reset/step/reward API；独立SFT/DAgger/DSRL recipe与评价器。所有新接口先CPU/fixture，再真实最小原生验证；不同层结果分开。

不改AGENTS全局约束、不另建庞大自动调度框架、不复制历史报告。一个goal、一份状态索引、一份增量科学结果表、原始证据按实际run存放即可。实际运行分支基于最新代码创建；计划PR不自动合main。

### 15.2 运行规则

- 每个自然冷边界保存QUESTION / CHANGED / ACTUAL / NEGATIVE / DECISION / NEXT_GATE；PR只摘要重大结果，不以测试数量代替任务能力。
- 软件调用、optimizer call、非零权重变化、完整恢复、episode启动、初态通过、原生安全、完整技能成功独立计数。
- 每次保存统计GPU峰值（Torch与进程口径分开）、wall/CPU/GPU时间、数据量、实际吞吐；并行时间不相加成总墙钟。
- 不变源码/模型资产只交付一次；长时本地writer不能依赖聊天会话。源码变更只在新capsule和冷边界生效，运行中不热改。
- 相同确定性失败没有代码/数据/资源证据变化，不自动重试；连接/提交UNKNOWN先核效果。两次同根因新修复仍失败且无新增信息，暂停受影响分支并给出最小再现实证，其他独立分支继续。
- 不删除旧数据/模型释放空间，不改共享NFS/SSH/驱动，不杀他人或其他项目作业；不在公开报告写token/私有连接细节。
- 训练到达有限端点并完整封存后，满足门才登记后继。达到站点绝对截止但goal未完成，报告剩余依赖，不把ACTIVE写成许可自动延期。

### 15.3 状态机与不可冒领的出口

`READBACK → CORE_LOGIC_READY → CORE_MINIMAL_TRAIN_RESTORE_READY → HOPPER_CANARY_READY → DATA_TASK_READY → SFT_CHECKPOINT_SEALED → CORE_DEV_SCORED → DAGGER_OR_RL → FINAL_TEST_SEALED`。

可并行状态：`WAIT_SITE_AUTH / WAIT_ALLOCATION / RESET_BLOCKED / OBSERVABILITY_BLOCKED / DATA_INCOMPLETE / NO_SFT_GAIN / NO_DAGGER_GAIN / NO_RL_GAIN / INCONCLUSIVE / TASK_SKILL_PILOT`。

G1/G1H通过只叫工程训练资格；旧FIT4拟合通过不是T1策略；DATA_READY不是训练完成；回报增长不是任务成功；发布PR不是现场执行。

## 16. 第一批应实际完成的动作（不用等大目标全部就绪）

1. 核最新现场终态和Hopper当前合法资格；将本轮单卡授权绑定新profile，不再等待旧H200科研审核。
2. 复用scratch环境/权重，传目标FIT4/Processor；同时在Core验证hook修复和最小非零更新恢复，避免重复原OOM路线。
3. Core准备连续决策Teacher与任务前稳定性记录；现有21窗口有限诊断继续，但不启动其8000步生产训练。
4. Core最小链通过后，在Hopper一次提交单卡canary/FIT4，完整恢复/导出；新T1数据工作与之并行，不占GPU等待。
5. G2/G3成立后提交正式SFT；交付Core做共同DEV；依实际失败走G6A或G6B，不只返回新计划。

本文件没有启动上述工作。Codex应在实际运行后回报jobID/实际命令/退出码/数据与checkpoint/原生分母/下一阻断，不能只复述计划。

## 17. 来源与证据分层

内部事实均以审阅commit `cad80fbc2a9e3f70288b56967f196731612c452d`固定；后继使用最新真实回执，不被文档快照倒退。上文所有新样本数量、阈值、seed、训练端点和阶段门均是本goal设计，不是既有实测或官方保证。

[R1] `reports/WM02/fusion-night1-20261106/HOPPER_SCRATCH_AND_FIRST_TARGET_OPTIMIZATION_ACTUAL_V1.json`：实际Hopper准备、Core/Vanda失败、3次optimizer调用和未恢复资格。
[R2] `src/go2arx5_vla/world_model/target_gr00t_profile_v1.py`：当前仅Core/A40站点、FULL/LAST6/LAST2、固定父与日程；Hopper需新增真实profile。
[R3] `src/go2arx5_vla/world_model/target_gr00t_dataset_v1.py`及`TARGET_GR00T_FORMAL_DATA_AND_BASELINE_INTERFACE_ACTUAL_V1.json`：一episode一个真实Teacher决策、21窗口及旧标签语义。
[R4] `TARGET_GR00T_TEACHER_TWO_PREFIX_AND_START_DIVERGENCE_ACTUAL_V1.json`（同报告目录）：Teacher双段阳性和任务前启动差异。
[R5] `reports/PI0294/20261016T191445Z-localpc-mp1/N1_FINAL_SCIENCE_REVIEW.md`：源策略T1 10/20、T2 4/20及纠正范围，非目标成功。
[R6] `reports/PI0294/design/core32-maintenance-20261031/UNIFORM_CURVE_REMAINING_AND_MATCHED_GRIP_COUNTERFACTUAL_20261101.json`；`reports/PI0294/design/science-goal-v7-20261031/LATE_HELD_TRAIN7_TAKEOVER_COVERAGE_ACTUAL_20261102.json`：夹爪卸载因果贡献与恢复样本局限。
[U1] https://github.com/NVIDIA/Isaac-GR00T/blob/main/FAQ.md （2026-11-09检索）：数据量只是指导；困难任务推荐HG-DAgger，N1.x无持久记忆。本文没有承诺100条必成。
[U2] https://github.com/nvidia-isaac/isaac-rlpt/blob/main/docs/pages/concepts.md （2026-11-09检索）：冻结策略steering、DSRL/SAC-edit、动作头PPO及chunk决策折扣；不能当作Go2已支持。
[U3] https://github.com/nvidia-isaac/isaac-rlpt/blob/main/README.md ：research preview；执行者先固定实际commit和适配版本，不自动升级现役vendor。

只读研究与新计划均未证明当前站点凭据有效、GPU空闲、作业已提交或任何新任务成功。
