# SLA Corelab Embodiment Factory V5：双卡多环境、开源场景与持续本体数据生产

2026-11-26（SGT）。用户最新要求：Hopper 后训练/等待端点期间继续 Corelab 数据采集，尽可能使用两张 GPU、多环境并行，复用开源场景设计。**本文件为新执行计划，不是现场启动或数据产出的声明。**

入口：[执行手册](SLA_CORE_EMBODIMENT_FACTORY_V5_RUNBOOK_20261126.md) · [机器登记](SLA_CORE_EMBODIMENT_FACTORY_V5_REGISTRY_20261126.json)。本计划是 [V4 #519][v4] 的数据生产后继，原V3/V4训练谱系、既有科学人口、任务评分及全部历史失败保持。复用旧 [Core多环境计划][n2plan]，不重复整个平台建设。普通工程修复由现场Codex自主完成，不逐步等待人工批准；保持真实授权、有限任务、原控制与安全。

## 1. 已核实事实与决策

本次实际回读 #520 最新可见提交 `d175383b07cabb701492222bc333809d57ca5830` 的增量报告；无新的SSH/PBS或Corelab实时资源查询。早期CURRENT正文是14:50历史快照，以下以最新分节回执为准。

| 项 | 已核实实绩 | 决策 |
| --- | --- | --- |
| Hopper 正式1K | 作业735204；17:09回读564更新，p50 5.087s、p95 5.106s；250/500完整中间点；原1K尚未在读到的报告结算 | 健康训练继续，R0设备profile依旧有界且在原job自然结束后；不为采集停止/等待H200 |
| 训练瓶颈 | Go2/RoboDojo/LIBERO FM分别2.360/1.201/1.148s，aux 0.329s；当前已64/32/32/4整批 | Core数据和评价独立推进，不再用Core大型训练争资源 |
| 目标数据 | 52条当前兼容旧轨迹（4450窗、4294完整支持H40）；TRAIN-069封存69条，46 HOLD/23 PLACE，新增17需新标签/reader准入 | 先去重准入已有数据，再填缺口 |
| 源域 | RoboDojo200条/210398窗，LIBERO1261条/209146窗；原MAIN3.8K实际源监督各121600次，推盘、抽屉、容器、insert均有实际暴露 | 复用任务及资产，不把源数据动作改称Go2执行命令 |
| 容器 | 9冷候选3过；6原生启动、4技术中断、2有效完整失败，均真实抓持抬升38.85mm；D02搬运姿态5.00069°越原5°，D01足漂5.617mm越原5mm，释放未开始 | 原站姿2格完整0/2保留；不反复同因重试；另建M_FREE容器任务、完整移动教师资格 |
| N2 | D3旧匹配8卡，N1 1128.48s，N2 569.57s（1.98x）；最新归档同容差6/8；两组STOP后被停0积分、peer各3500tick | 非接触工程吞吐不等于生产；补同钟状态/真实selected-reference及接触/故障隔离 |
| N2最新定位 | F1 R2世界TCP峰差7.041mm、安装点差6.731mm、安装点系TCP差2.068mm；参考差约0.00034mm；R1类似 | 优先排查实际底座/安装点、控制器状态和参考选择身份，不靠时间平移、改容差 |
| Core产量历史 | 11/21 16:55 TRAIN23终态/7准入，21:02 TRAIN48终态/25准入；增量18条/4.12h，同时新增12 VAL终态 | N1历史4.4合格TRAIN/h仅同类任务规划基线，重新实测新任务 |
| Core GPU历史 | 两设备记录均称RTX4080 SUPER，index0 32760MiB、index1 16376MiB；前者非标准规格，可能为站点特殊报告/识别问题 | 按当前真实UUID、总/空闲显存、实际EGL设备和可用资源绑定，绝不猜卡型号/容量 |
| Vanda | 最新CURRENT明确用户安排暂停 | 不重启Vanda或旧CPU-only PBS守候；只用现有合法Core→Hopper或已授权中转通道，若无合法路线保存READY并报告 |

固定依据：[Hopper][hopper]、[新容器与N2归因][container]、[N2回读][n2readback]、[源域][source]、[GPU][gpu]、[早期采集][earlier]、[后续采集][later]。

## 2. 两条并行生产线：不等Hopper，不停Core

Corelab负责物理任务与数据生产，Hopper负责高吞吐训练。Core主协调者只在真实资源冲突时切换已有工作；任何已就绪N1、离线场景编译、数据复核、Teacher开发不等待1K完成。学生评价获得有限公平窗口；已健康的任务不取消。V4父模型正式DEV6已经启动，原人口继续，不拿新工程卡代替正式结果。

目标状态不是两个GPU各运行一个训练模型，而是两个卡为独立物理world提供真实渲染，GPU0另外保留一个共享SLA推理服务。仿真/OCS2/Teacher主要使用CPU；GPU负责渲染与学生推理，CPU/RAM/IPC/磁盘可能先饱和。

| Profile | GPU0 | GPU1 | 总物理world | 用途 |
| --- | --- | --- | ---: | --- |
| P1 已有合法N1 | 按当前合法绑定渲染/评价 | 可做CPU离线准备（无需占GPU） | 1 | 立即继续原采集/已就绪新任务 |
| P2 双卡N2 | 1 Teacher渲染 | 1 Teacher渲染 | 2 | 首选并行资格与生产 |
| P3 双卡N4 | 2 Teacher渲染 | 2 Teacher渲染 | 4 | N2已合格且CPU/主存/渲染实测有余 |
| P4 混合 | 1 Student及SLA服务 | 1 Teacher | 2 | 独立混合负载资格后 |
| P5 混合扩展 | 1 Student及SLA服务 | 2 Teacher | 3 | P4及对应N2同卡资源/隔离已证明后 |
| P6 后续 | 单SLA服务，多个学生批推理 | Teacher渲染 | 按实测 | 单独登记，不是本轮必要条件 |

GPU0/1为逻辑角色，现场可按实际设备调换；GPU0历史报告较大，不等于当前可用。Teacher在GPU0有评价等待时不新派发；已开始回合正常安全结束，若混合尚未合格则先排空所有冲突world再交接原学生独占窗口。沿用旧30分钟等待或12个Teacher终态让行规则。N2/N4工程未过就维持N1，其他CPU工程与Hopper不阻断。

## 3. 轻量Scene Factory：只复用有用的开源资产/任务，不换控制后端

复用现有 [core_hopper_geometry.py][geometry]、[core_hopper_collection.py][producer]、现有MJCF编译器、admission、catalog、Journal/OwnerLock；旧geometry只平移两块，旧producer绑定F1/F2及单world，需针对真实缺口扩展对象/任务/布局接口，不能复制目录改锁名冒充泛化。不要新建大型调度或仿真平台。

| 模板 | 来源和范围 | 第一用途 |
| --- | --- | --- |
| S1 已有桌面 | 原红蓝块及目标区，独立物体位置/yaw、真实机器人初态 | legacy补缺、Reach/Orient、移动抓持 |
| S2 多物体桌面 | 优先简单几何体＋少量许可清楚的LIBERO mesh/XML | 不同大小/形状/负载、完整Place |
| S3 Push | LIBERO push plate语义/物体与现有桌面 | 非抓持接触、目标区、撤离 |
| S4 Container | LIBERO basket/bowl、已有近源水果/蓝碗资产 | 移动接近、完整入容器与释放 |
| S5 Drawer | LIBERO柜体或RoboCasa许可明确的单个活动fixture | 保留独立未见技能评价，不进Adapt100/200 |
| S6 Insert/组合 | 源域实际insert/顺序任务、最小局部资产 | 条件性独立迁移；非首批采集前置 |

LIBERO优先，RoboCasa只导入必要局部资产，RoboDojo主要复用任务语义/近源物体与轨迹结构，不迁其Isaac全栈；不引入MJX/Warp/新物理/新渲染后端。RoboCasa项目公开许可说明代码MIT、资产/数据CC BY4.0，具体固定版本/第三方资产仍逐项核实，保留署名。LIBERO也须固定release/commit及具体mesh/texture许可，不将代码许可自动套所有资产。下载小白名单，不整库盲下载；不公开项目私有数据/原模型/主机路径。公开来源：[LIBERO][libero]、[RoboCasa][robocasa]、[许可][robocasa_license]、[MimicGen][mimicgen]。

Scene manifest 必含 asset source/revision/license/member hash、物理单位/坐标/尺寸/质量惯量/摩擦、visual与collision mesh、joint/limit、物体初态及支撑、桌面及环境几何、初始底座及臂构型、两相机定义、目标区域/合法接触、任务语言/评分器、所有随机种子与split。不能只在JSON填姿态字段不验证实际编译后的MuJoCo状态。保留Go2/ARX5模型、关节映射、TCP标定、WBC、2ms及安全版本。桌高/可达布局为目标本体适配，原任务的语义及关系保持；改变布局/移动模式须新任务身份，不能称原站姿任务成功。

冷态先检查mesh尺度/质量、关节限位、支撑、初始无重叠、全臂/指垫/底座净空、相机真实可见与工作空间；仅静态筛查不能宣称可行。每个新任务模板从有限教师资格取得完整正常初态→目标→终止成功；Teacher可用GT规划，学生不能。只在实际可行工作包进入自动生产。原站姿容器0/2不放宽5mm/5°；新M_FREE容器任务从新预定初态和明确移动需求执行，冻结通用硬安全/接触，单独评价。优先紧凑、可达的basket或低边容器；不得通过移除碰撞体、虚假支撑或强制释放达标。

### 3.1 自动轨迹生成

借鉴MimicGen的object-centric subtask思路，从已准入的真实Go2示范及可核实源任务事件提取接近/闭爪/抬升/搬运/下放/释放/撤离阶段，按物体或目标局部frame迁移关键意图，再用实际冻结WBC执行。对位姿变换用实际标定、完整SE(3)，不同夹爪方向/单位/clock分开处理。LIBERO原OSC action不假称EEF command；源future-achieved不作目标执行标签。先静态全路径筛选，后真实轨迹、评分和消费验证。绝不直接将重定向轨迹、未来实测状态或未执行后缀作为Go2 command。新scene每个物体真实相对位置/姿态改变；同场景不同指令有意复用但按同group划split，不能拿多个seed当独立场景。

## 4. 数据目标：Adapt100→Adapt200＋40 recovery

完整示范表示完成自身已声明技能的全episode；Reach不是完整PLACE。恢复片段独立计数，不混入完整200。现有52为当前真实兼容基线，69为待完整准入候选；若不足69，只补实际缺失的legacy task/phase，不以易HOLD替代稀缺PLACE。按目标桶去重，移动Place不同时计入MobileGrasp与Place。

| 训练桶 | Adapt100累计目标 | Adapt200累计目标 | Wave100实际请求上限 | 全两波累计请求上限 |
| --- | ---: | ---: | ---: | ---: |
| LEGACY | 69 | 69 | 51 | 51 |
| REACH_ORIENT | 8 | 20 | 16 | 40 |
| MOBILE_GRASP | 8 | 25 | 24 | 75 |
| DIVERSE_PLACE | 8 | 40 | 24 | 120 |
| CONTAINER | 4 | 26 | 12 | 78 |
| PUSH | 3 | 20 | 9 | 60 |
| 合计 | 100 | 200 | 136 | 424 |

其中legacy目标69包含46 HOLD/23 PLACE，51个补缺请求仅在真实缺口时启用（最多17缺失各3次不同预封存条件，不重复正式评估）。若现有69全兼容则legacy新增0，Wave100其他请求容量85，累计其他373。表为本批**最多请求/逻辑槽位**，不是承诺成功数量。任何已存在且符合合同的完整TRAIN示范可直接计入相应桶；不为凑数重采。若新增17全部兼容，达200需131条新完整；否则最多148条（假设现有52仍兼容）。

每波在首个新正式请求前封存实际布局、种子、任务、容量与分割。采用机读slot范围，展开具体manifest，使用已存在V4 PUSH/CONTAINER TRAIN的合法未启动条件/结果优先映射（各10个上限，不另算同一尝试），旧#425 TRAIN480及历史TRAIN240只映射实际相关未启动/已合格条件。未开始条件保持身份；已完成失败不重跑作同一科学条件。原V4 DRAWER目标训练10格若未开始暂缓，保留其独立科学人口及结果，不将Drawer/Insert纳入Adapt100/200；若已经有目标技能训练则标非zero-shot，保留未见子技能/组合。当前已结算站姿CONTAINER DEV失败不可映射为新M_FREE DEV成功。

新增131条优先覆盖至少40个新独立数值TRAIN布局（提案；各桶至少5个、覆盖不足则报告，不以重复seed补足），每布局完整示范通常不超过4条；只有有意的同scene多语言对照按组预封存并说明。物体类型至少4类（箱/柱/球或果实/不规则或瓶状，实测可抓取为先）；有干扰物、不同桌面高度/视角、同scene不同目标、不同初态，真实相对几何去重。移动先8–12cm已验证附近，再条件探索12–20cm及更远、不同方向/yaw，先教师资格，不假定冻结WBC全方向可达。搬运后释放需稳定支撑、无再抓持、撤离；不能仅因入区算成功。

恢复数据额外40条：接近构型偏离10、抓持/抬升偏离10、搬运/放置/释放偏离12、移动后跟踪偏离8；每组最多2倍请求，合计80，Wave100先10条目标/20请求。优先从TRAIN学生实际访问的安全可恢复状态，独立复建真实初态/原生前缀，记录状态偏差与实际Teacher纠正。不可从触发STOP状态直接继续积分，不伪造恢复控制器内部状态；使用合法独立reset和可复现前缀，记录差异。若只完成局部纠正，标RECOVERY_PARTIAL，保留真实mask，不写完整任务成功。旧评估几何和日志只诊断不作为训练。每个恢复窗口明确学生/教师控制来源，只用实际Teacher接管后消费的有效命令作标签。

### 4.1 评价与泄漏控制

原V4 PARENT/M1000/T1000/M2000六卡、AB/L1及PUSH/CONTAINER/DRAWER零样本人口照原注册结算。初始冻结模型可在无目标数据新技能上先测试，不需要等200。新M_FREE容器不混入原站姿V4人口，单独绑定相同WBC/原通用安全和新合理任务谓词。新增训练族先最多2个独立非TRAIN资格场景/族（6桶中新增5族共10；旧可用资格复用，站姿原失败不代替移动）；每族2个独立VAL几何（共10）只做数据质量与Teacher可行性，不进TRAIN；Drawer/Insert/组合各3独立CONF几何、两noise，最多9教师资格＋3候选模型×3技能×3几何×2noise=54学生卡，组合仅在组成技能已有阳性时启用。CONF任务/几何/噪声/成功谓词先封存，固定候选MAIN3.8K或预先绑定V4父基线、Adapt100、Adapt200；比较各候选相同零样本合同。无新本体示范才称目标零样本；公共源实际见过的技能与从未见/组合区别报告。已通过V4同条件可引用，不重复计费。

Adapt100/200之间比较目标数据效率时，预先固定同父和相同目标访问的控制，不能将序列训练更多更新自动归因于更多示范；实际延续训练作为工程主线，单独报告。少样本5/10/20学习在零样本评价结算后以独立TRAIN/DEV/CONF另登记，不改旧测试分母。

## 5. 双卡、多world工程：最短路径到生产

旧N2两组STOP证据已有，但同钟差仍未通过；真实controller-selected reference缺失。先读现场最新进程/旧slot/代码，复用有效旧证据，不重做已通过部分。对变更后的拓扑实际检验，不把同一历史结论自动授予双卡。

1. P1立刻运行已资格N1（如原学生在途，先完成它），并行CPU场景编译/可行性冷筛/数据入库和转运；按新任务模板有限资格推进，不等待所有新场景。
2. 现场确认两GPU UUID、PCI bus、实际EGL/renderer mapping、CUDA/JAX可见设备、CPU可分配核/cgroup、RAM/PSI、tmpfs、磁盘可用及quota。GPU0异常32GiB记录只做历史提示，按真实现场值。不同卡的EGL渲染绑定不能只依赖CUDA_VISIBLE_DEVICES；验证每worker渲染上下文、设备和图像身份。保留原相机图像分辨率/采样。
3. 复用旧[N2/N4工程协议][n2plan]的36/28条件上限，优先补D3实际缺口：统一初态/控制器暖态，记录实际selected reference/policy/序号与控制消费tick，定位F1同钟安装点偏差；交换worker/CPU/GPU/domain映射，短匹配、完整接触、peer STOP和worker故障隔离。旧状态阈值不事后放宽；旧两组STOP仅在相同实现拓扑证据可适用时复用，否则补受影响组。双卡1+1作为明确profile单独资格，原单卡N2失败不能改成成功。
4. 每world独立MuJoCo mjData、WBC/MPC/leg/ROS domain、所有非ROS IPC及共享内存、可写cache/tmpfs、RNG/时钟、EGL context、输出、owner lease、STOP/ACK；唯一pool协调者与catalog writer，不能通过复制锁或配置规避单world限制。底座和机械臂安全门仍逐tick有效，STOP后该world新publish/install/grant/integrate=0，peer真实继续。
5. 资格完成后生产P2，并在真实完整任务衡量“合格示范/墙钟小时”，目标≥1.5x同期匹配N1；P3 2+2在N2稳定且CPU/主存/渲染余量足够才启用，目标≥1.25x N2且任务质量不下降。没有收益保留低并发。资源预算全池统一预留，避免每个worker单独通过旧内存门而总体OOM；实际不重叠CPU核心、固定WBC线程数，禁止线程过量。旧N1/N2/N4资格只补当前版本所需，原上限/分母不重置；新拓扑确需额外条件先版本化有限补充。
6. 混合Teacher+Student另行资格：P4 1T+1S两组（正常/停止一方，互换STOP），合计最多4工程启动；P5 2T+1S两组（一次停学生/一次停教师，其他继续），最多6工程启动，合计10。已有实际同拓扑合格证据复用。共享SLA推理只加载一份权重；当前正式学生N1不能直接混跑。若混合不合格，在原公平规则下交替Teacher与学生，生产不长期占卡。多学生batch推理另登记，输入/噪声/历史/输出按world/query ID隔离，禁止迟到结果安装。
7. 若旧同钟数值差证实是独立初态/正常执行随机性而非串扰，可另建新“独立数据采样profile”在首新结果前固定真实状态、任务和隔离统计标准，保留旧失败；不能以“只是采数据”为由跳过真实控制消费、状态归因和安全资格。

MuJoCo的多线程采样和独立动态状态原则见[官方simulation][mujoco_parallel]，每个GL context/渲染线程的资源管理见[官方rendering][mujoco_render]。实际固定后端版本不因引用最新文档而升级。

## 6. 单world效率优化：并行开发，单变量验证

历史短卡138.89s中warmup53.02s，固定150ms回调处理49.53s，advance-to-pause44.72s，install/H40 26.07s，RGB9.81s；部分计时嵌套不能相加。[原始profile][perf]。优先实测当前完整任务的冷启动、物理/WBC、实际参考选用、RGB、写盘与admission，按互斥区间报告。

- 常驻world/控制进程＋warm reset：只在真正重置仿真、控制器/solver内存、时间代次、参考/夹爪/STOP/RNG、历史/渲染/日志后使用。未能证明完整reset保留冷启动；不能靠重用不干净状态加速。先同场景冷/热匹配，之后完整接触和STOP。不同场景改变mjModel时要真实重新绑定，允许仅复用只读资产与进程环境，不假定model热换可靠。
- Event-ready：找出实际selected-reference/ACK与消费，确保原回调和确认语义；仅取消可证实冗余的等待，保留WBC必要响应、2ms、H40原时标及STOP；不擅自缩短150ms或加快模拟时间。必须先获得真正参考选择身份，不以收到消息等价控制器采用。
- 2→4 CPU离线编码/审核workers，独立缓冲，按真实world/tick/相机复制字节，队列最多每world2份捕获，满则背压不丢帧；同步保留关键控制和STOP记录；不改变训练图像字节或静默降采样。将必要审计移到封存边界，复用可信不可变校验而非每tick重扫历史。模板只读mesh缓存，scene编译/静态筛查和真实物理独立。
- 顺序验证 warm reset、event-ready、异步I/O，不能与N2根因同时混改；每候选最多4个匹配工程启动（冷/候选两类条件各一对），包含完整任务或用旧已合格原件作精确基准，若需要更大样本按证据另登记。3项最多12新工程启动；新实现若影响隔离/控制另补对应真实资格。先按同条件端到端吞吐和入库质量选，不改变评分/图像/轨迹；不承诺可实现倍率。
- 单次完整任务失败按原终态保留，同一cell连续2次相同机制先暂停该cell并查真正首偏离；改Teacher几何、路径或任务定义属于新版本，其他就绪cell继续。不要给易HOLD无限重试、不要为了成功调整旧科学评分。

## 7. 样本合同、交接与实际训练访问

每query保存当前真实body/wrist RGB及原观测时间、实测EEF/夹爪、原指令；额外保存底座pose/twist、臂关节/速度、原世界到episode-fixed mount变换、控制器参考/实际消费、接触/物体/任务phase用于审计和未来独立BASE/历史研究，GT不进student。以当前query固定frame/anchor、50ms时钟、真实连续已消费命令重建H40，缺失尾段时间×动作维mask，FM噪声与loss合同保持；仅有完整实际消费支持的窗用于完整H40辅助。保存实际上一夹爪命令（不能用预测/测量替代），不跨reset/STOP/gap拼接，不把nominal未消费后缀标成监督。源域各自原H50/OSC等语义与归一化保持。

每4条新准入或6小时/阶段收尾（先到者且有变化）封存不可变小快照；首100/200与recovery另封，manifest含原件/scene/split/teacher/version/实际消费/哈希/有效窗数、实际完整H40支持与阶段计数。统一catalog去重旧52/69/后继，连续消费窗重建版本与当前V4 reader兼容。单writer atomic publish、分批跨站hash与真实reader读入、Hopper独立READY；实际进入优化器后报告访问数，不能仅以数据上传宣称学习。使用既有合法交接；Vanda暂停，不擅自恢复，若只有未授权通道则保持Core READY并报告。私有数据不自动公开。

现役1K及条件2K固定已封存课程，不热换dataset/采样/LR或训练树。Adapt100就绪后可以安排新的明确冷边界训练版本（同父对照与实际续训分开）：建议1K–3K；Adapt200+Recovery40建议3K–5K。保留64/32/32/4整批、2:1:1域均值权重、旧冻结WBC及实际训练合同；目标旧技能/新增技能/恢复分层、实际完整支持aux另核，采样/统计在新阶段预封存；不因短训loss下降自动扩大。采集不等待训练，不重复健康H200作业；新长期授权/空间不足时仍继续Core可行工作。旧V4累计2000与5K等新后继不能混作同一已锁定日程。

## 8. 真实产量、资源预算和阶段收尾

先以20个实际新生产尝试（或各族首4条可行请求的合计，记录精确工作量）重新测完整任务平均墙钟、准入率和合格示范/h；每日给每类任务与布局的尝试/有效/成功/准入/唯一几何/实际消费窗/已到Hopper/实际被访问，分别列完整与纠正，工程卡与VAL/CONF不进TRAIN。不要仅按tick/s、渲染fps或GPU占用报成功。

N1历史约4.4/h，规划3–5/h；N2实际工程1.98x不代表完整产量，条件估计6–9/h；N4仅情景8–12/h。若69全兼容，131条新增完整在持续生产时N1约26–44h、N2约15–22h、N4约11–17h；若仅52兼容，148条约30–49h/16–25h/12–19h。均不含新任务教师开发、评价、40条恢复、资源等待或停机；按实测更新预测。历史24条传输训练包841457381 bytes≈35MB/条不是完整raw空间；原始工程短卡写入约315MB，不据此断定长期磁盘够。现场测每任务峰值与失败原始文件占用；保留原采集/学生存储门＋全池在途最大写入/下一快照/partial传输安全余量，配额不足提供具体需求给用户，不删除原件。尽量内容寻址、同文件系统硬链接、无损编码，不移动历史数据或改系统设施。

本轮计划从现场真实挂载起先运行有限72h数据生产，服从实际项目授权/截止及旧Hopper Goal的独立期限（不延长旧320GPU小时/12月5日边界）；每12小时或每波结束核对实际进展及剩余资源，不为时间限制停止正常安全收尾。允许普通工程自主修复及有证据的新有限后继，不要求每个cell用户批准；在失去真实授权、全局控制串扰/STOP异常、无法安全封存或任务端点完成时关闭受影响池。若目标未达且还有合法资源，现场报告实际缺口与下一有限批次，不能无限后台。

首批资源/容量：完整采集目标200（最多424完整请求，含映射继承的新尝试）、恢复40（最多80请求）、新增技能资格≤10、VAL≤10、保留未见技能资格≤9、CONF学生≤54；旧N2/N4资格64容量不重复，混合资格≤10、单world优化≤12。新增技能任务每条有实际有限deadline；无原继承规格时按50,000个2ms tick、1800s wall、学生160 queries的旧V4 fallback先绑定，教师学生前向0；更长任务在首发前单独明确端点，不为完成而延长运行中条件。每次技术/科学失败都计真实效果，未启动容量不作分母。

**第一交付不是更多设计文档：** 真实Core资源与单world继续回执、首批新场景教师完整成功及实际数据入库、N2双卡真实合格产量（或明确缺口/N1回退）、首个新READY快照及Hopper实际读回；然后Adapt100、200及40恢复。同步原V4评价、新技能零/少样本结果与已封存模型。

本次仅发布GitHub文档；无新增现场GPU/物理/训练/传输、无已取得N2/N4生产资格或资产导入的断言。

[v4]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f72f3d57b470de3e62821cd5e1a33d938a669cb7/docs/protocols/SLA_HOPPER_AUTONOMOUS_V4_PLAN_20261126.md
[v4registry]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f72f3d57b470de3e62821cd5e1a33d938a669cb7/docs/protocols/SLA_HOPPER_AUTONOMOUS_V4_REGISTRY_20261126.json
[hopper]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d175383b07cabb701492222bc333809d57ca5830/reports/SLA_EXPERIMENT_PROGRESS/V4_HOPPER_HIGH_BATCH_20261126.md
[container]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d175383b07cabb701492222bc333809d57ca5830/reports/SLA_EXPERIMENT_PROGRESS/V4_STEP100_CONTAINER_20261126.md
[source]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d175383b07cabb701492222bc333809d57ca5830/reports/SLA_EXPERIMENT_PROGRESS/SOURCE_EXPOSURE_ACTUAL.json
[n2readback]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d2561743ecc507bdb279ca12cc5ff30e4e6e3ff7/reports/SLA_EXPERIMENT_PROGRESS/V4_FOLLOWUP_20261126.md
[n2plan]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4fe2c8cf1b69c168b2e3079e061bb7c702089cd3/docs/protocols/SLA_CORE_MULTIENV_COLLECTION_V1_20261122.md
[geometry]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/scripts/sla/core_hopper_geometry.py
[producer]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/scripts/sla/core_hopper_collection.py
[earlier]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/reports/WM02/sla-core-hopper-parallel-20261121/ENGINEERING_SELECTION_AND_MAIN_MOUNT_ACTUAL.md
[later]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/reports/WM02/sla-core-hopper-parallel-20261121/TRAIN024_AND_H200_CONTINUATION_ACTUAL.md
[gpu]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/c8642aa602d3e34dd4dcda5a55ff6ae7c7c53ae6/reports/WM02/sla-scale20k-20261119/DATA_AND_RUNTIME_ACTUAL.json
[perf]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4d597259/reports/WM02/sla-d2-teacher-parallel-20261118/PERFORMANCE_PARALLEL_ACTUAL.json
[libero]: https://github.com/Lifelong-Robot-Learning/LIBERO
[robocasa]: https://robocasa.ai/docs/introduction/overview.html
[robocasa_license]: https://github.com/robocasa/robocasa#license
[mimicgen]: https://mimicgen.github.io/docs/modules/overview.html
[mujoco_render]: https://mujoco.readthedocs.io/en/latest/programming/rendering.html
[mujoco_parallel]: https://mujoco.readthedocs.io/en/latest/programming/simulation.html#multi-threading
