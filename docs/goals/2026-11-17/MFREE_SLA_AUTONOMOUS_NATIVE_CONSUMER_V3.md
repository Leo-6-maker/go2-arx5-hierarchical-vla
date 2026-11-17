# M_FREE SLA 自主工程推进 Goal V3：从离线96/96到安全移动抓持

**制定日期：2026-11-17（Asia/Singapore）**  
**状态：PLAN_PUBLISHED_NOT_EXECUTED；旧离线实验完成，原生重新准入未获得。**  
**现役执行事实：** [PR #393 终态](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/393)，运行证据固定于提交 7afbc0bd953c9b87821858bd70738a287ec082fc 的 [ACTUAL_MATRIX.md](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/7afbc0bd953c9b87821858bd70738a287ec082fc/reports/WM02/sla-eef-grip-decoupling-20261117/ACTUAL_MATRIX.md)。  
**此计划延续：** [计划PR #394](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/394)的V2，但**取代V2中尚待SHIFT7、组合或CN38结果的启动顺序**。不覆盖V2原文件，不修改冻结任务回执。本次授权是**工程问题可在指定边界内由Codex自主修复并推送GitHub**，不等于自动解除隔离、分配新GPU配额或允许原生仿真。

## 1. 已冻结事实、当前阻塞和完成定义

| 层级 | 实测结果 | 科学含义 |
|---|---|---|
| 冻结WBC专家 | CN38：41次原生请求、38条合格（F1 24、F2 14），2次ILLEGAL_CONTACT、1次共同隔离中止 | 有限场景专家可抓持，但已有不可忽视的安全负结果 |
| C20父模型 | 20条成功专家数据／1376窗，真实1000更新／32000访问；旧学生抓持0/13，另11未启动 | 新本体后训练已真实执行，旧策略失败不能回填 |
| TIME500 | CLOSE20/20、LIFT12/20、总88/96 | 时间表示解决CLOSE但出现LIFT曲率回退 |
| SHIFT7 | **500更新／16000访问、TRAIN96/96、87项恢复、RGB16一致**；k26峰值由0.308411降至0.080901m/s²，k33升至0.199283（阈值0.25） | 参数布局影响XYZ时间曲率，但不能定位具体网络层 |
| 两固定组合 | FM500 EEF＋TIME500 Grip **96/96**；FM500 EEF＋SHIFT7 Grip **96/96** | 共享模型和双分支均有已见专家状态上的合法参考 |
| CN38−C20 | **18真实episode、180/180**（F1 140/140、F2 40/40），闭合事件36/36及时 | episode-heldout专家状态诊断，不是盲TEST、学生rollout或成功抓持 |
| 实际共享推理 | 冻结父模型1份／adapter2份，新增376完整分支生成，16次独立输出逐位一致；热态双分支p95 **355.31ms**；缓存响应8正常＋56故障案例通过 | GPU推理和非物理响应接口成立，但缺相机／IPC／native接缝与真正的消费生命周期 |
| 物理准入 | F1/F2专家与学生、M_ASSIST共同隔离，新增native请求／积分**0** | 任何学生或专家新物理任务都必须先独立批准 |

当前离线候选首选固定为 **FM500 EEF＋TIME500 Grip**，模型包manifest SHA256为 **7aacd184941c6980154935833d97a1975e3cba3d1c1dd6c8a4878d38abbbd1ce**；TIME/SHIFT备选不能按query动态切换。原研究来源：SHIFT ef584be687d9f3c94e1f79e0e9f44ad12f32e92a／spec c5faece3…，解耦 e3fb6005597a9329fa886c3015e9c5fa7f92afbe／spec 1f086443…；终态报告提交 7afbc0bd953c9b87821858bd70738a287ec082fc 不等于现场源码热更新。

**目标不是继续做100%离线合法率：** 必须使同一冻结WBC和原有H40/P10/50ms、40mm/s及全部原安全约束下的SLA真正完成M_FREE F1/F2移动接近、原生抓持、物体抬升≥20mm并连续保持≥1s；沿指定任务方向底座净位移≥8cm另列。DEV开发目标≥24/32（各族≥12/16），新一次性TEST目标≥32/40（各族≥16/20），0硬安全STOP。仅当原生实测完成这些条件才标GOAL_MET。以上阈值为**拟议目标和待审面板**，不是当前成绩，也不是本计划授予的物理请求。

## 2. 工作主线：尽快完成非物理工程闭环，停止多余训练

### A0 — 先做一次不可变源／终态核对，不重做离线科学实验

- 核对PR393最新HEAD、Core执行source/spec、CN38和冻结包SHA、注册表、unit/PID/cgroup、lease CLOSED；确认没有正在运行的同一native-consumer任务，UNKNOWN按现役Journal独立核查，不删除锁重投。
- **复用** SHIFT7 96/96、两组合各96/96、CN38 180/180、共享RGB16、缓存响应64/64与所有原始失败；不得将其重新当作待跑任务或重复优化训练。
- 以新Goal唯一ledger分开记录：已存在／实际新增模型生成、优化更新、原生请求、积分、GPU墙钟、失败／未启动。历史PR392的“blocked”仍成立：真实移动目标未完成。

### A1 — 真实请求链的确定性协议（禁止物理积分）

基于PR393现有 hybrid_request_v1.py、hybrid_eef_grip_v1.py 和resident_close_pair_v1.py，**补齐真实服务／请求／原编译器之间的接口**，而非重写GR00T、SLA、MuJoCo或WBC。至少证明：

1. Query身份绑定 episode、sequence、native step、单一输入RGB＋实测EEF＋语言＋固定frame的哈希、父模型／两adapter/codec/statistics SHA；禁止不同query的输出被合并。
2. FM500产生EEF位置／rot6d，TIME500独立正确解码夹爪。两分支同一有效输入与noise来源；融合先发生在同一**物理nominal EEF10 H40**下，再走原 nominal/reference完整编译器，不拼接不同语义的归一化第7维，不裁剪／平滑／补专家动作。
3. previous_reference使用**上一条实际发出**夹爪命令的工作开度及source_native_s、query_native_s，精确检查50ms；不得用受到物体接触限制的实测开度、预测未执行后缀或已停止旧episode的参考替代。这个接口只给确定性解码器，不引入教师stage/真值。
4. 在EEF与夹爪两次推理之前、之间、完成后、编译/安装前检查同一request有效性、取消／STOP／TTL、原子ACK和当前native时钟；只允许**一份**完整H40安装，绝不独立写入夹爪控制。
5. H40生成不推进持久状态。只有实际receiver确认已发出参考及真实消费/ACK才更新命令历史；P10可完成10点或发生7点部分消费，后者不冒充完整0.5s。STOP/重置/隔离清理旧查询与cache，延迟响应弃用并留下审计。
6. 保留原 M_FREE 7D EEF-only、不携带旧BASE hold或oracle；不得暗中启动M_ASSIST或固定底座评分。
7. 核对训练／部署task-frame、TCP、world转换及nominal→mechanical只映射一次；与已保存真正安装参考做定点等值对照。

交付：一份带明确消息schema、原子状态转换和实际路径的适配器版本、定向单测及故障注入结果、普通RGB到联合H40再到原编译器的**非物理端到端回放**。仅mock receiver/录制packet的CPU接缝不能写作“原生消费资格”。

### A2 — 真实Core GPU+非物理IPC试验及延迟准入

- 在有效站点allocation、owner锁、内存/显存余量允许的条件下，**最多32个统一请求、每请求2分支，新增完整模型生成≤64**；已完成CN38 376次一律复用。无新训练、无物理请求。若资源不合格，WAIT_RESOURCE并报告，不抢他人GPU。
- 覆盖不同query、同query重复、两分支延迟、相机读入、参考身份与time stamp、服务重启与cache清理，以及原生接收器的**非积分替身**。录制native消息格式与原编译器输入输出，但不连接真实MuJoCo积分/独立新native worker。
- 对录制的已知8个正常和56个故障案例复用结果，重点另测**真正IPC传输**中的竞态、部分消费回执、STOP立即发生在两分支之间、TTL失效和超时后不安装动作。
- **延迟风险需明确计量**：现有仅双分支网络热态p95=355.31ms，已经略超过历史≤350ms续期目标，还不含相机／IPC／编译。测普通响应真实端到端p50/p95/p99/最大、模型冷载、JIT缓存、超时拒绝。仿真可暂停等待推理时标记SIM_PAUSED_ONLY；不能据此声称连续实时运行可守住0.4s TTL。若无法在现有真实时间合同下续期，停止升级，允许工程优化（缓存/预取/正确驻留/串行化），但**不延长TTL、不跳过ACK或编译器**。
- 优先清楚地区分模拟时钟/墙钟，不能把0.5s原生P10与一次GPU生成实际墙钟混为同一计时。任何优化必须在相同输入/noise得到可解释的同版本结果；改变计算数值的优化需新candidate身份和离线回归。

验收：每条请求的身份、时钟、两个模型输出、融合H40、原编译器结果、回执/拒绝、实际耗时与资源可重演；全量故障注入0误安装/0漏STOP。无法达标则出可行动的非物理阻塞报告，不启动原生。

### A3 — 学生策略离线反事实状态审计（有源才运行）

在旧13条学生真实rollout的已保存普通RGB、原生时钟和历史命令可完整核验的前提下，对**冻结新候选在学生曾到达的状态**做一次性one-step离线预测。不得把“换了策略后的下一状态”仍当作新策略产生的真实环境反馈。若缺同步RGB/之前实际发出命令则标SOURCE_MISSING，绝不事后重渲染冒充采集。

- 报告当前学生状态的H40合法、开闭时机、EEF进度、接触前预测风险；同旧专家状态的180/180分开，不能把历史754个query当作754次独立任务或新TEST。
- 这是**off-policy single-state**分析，不能证明student-on-policy连续性；它可以指导获准后pilot的诊断采样，不自动允许新训练或伪闭环回放。
- 如果输入充分，优先复用旧episode固定锚点与缓存，新增调用应包含在A2不超过64的预算；超出则需独立新分配，不隐式扩量。

### S0 — 原生安全风险与停止合同，和A1/A2并行

- 三次既有碰桌的原生STOP、contact pair、TCP/右pad/BASE的2ms状态、reference age及WBC实际选定的reference身份分别核查；若只有receiver见证，写UNKNOWN_CONTROLLER_SELECTED，而非自动推断控制器内部原因。
- 当前保存2ms监测在3/3碰桌前12、34、316ms发告警，但**1/38成功专家被误拦**；原零误拦资格失败不能重命名为PASS。现有C++只在1204保存状态静态核对，native hook位置、传感/IO延迟、最坏STOP时延和真实预防行为均未证明。
- 独立准备至多一种新的高层安全接近／净空合同或一种保护器新合同（不得混合后宣称单因素）；在只读数据上声明检测延迟、允许误停代价、原失败回放、对照分母和新的场景支持域。若希望调整误停验收标准，必须由独立责任方**事前签署新风险接受标准**，不可基于历史1/38事后改分。
- Codex可以独立修复**安全监测实现中的工程缺陷**（路径、超时、时钟读数、无效输入fail-closed），并提交PR及被动测试；不得自行安装到native生产进程、修改原硬安全门、扩大任务安全范围或签署自己的READMISSION。
- 有权限的独立责任方签发绑定**执行源SHA、双adapter/参考包SHA、scene MJB SHA/layout、冻结WBC/config SHA、保护/任务profile、有限面板、STOP/hook/失败处理**的明确READMISSION之前，F1/F2专家、学生及M_ASSIST仍隔离，**新增native请求/积分=0**。

### P — 只有独立安全重新准入后才开展真原生

在签发READMISSION且当前资源/owner重新核验合格以后，按批准身份逐级执行，候选预先冻结；不自动运行228矩阵、不恢复C60/C100采集、不做真机或RL。

| 原生阶段 | 条件性上限 | 继续条件 |
|---|---:|---|
| P0专家安全 | 8（2族×2新审定布局×2reset） | 无新硬安全STOP，已审安全profile生效；有STOP立即隔离相应范围 |
| P1学生有限pilot | 8（F1 4/F2 4），前2各族1次包含在8内 | 至少各族1次完整移动抓持：native grasp、抬升≥20mm、连续保持≥1s、方向底座≥8cm；0硬安全STOP |
| P2开发DEV | 32（两族各16） | 成功≥24/32、各族≥12/16且0硬STOP；不针对每个失败偷偷改变模型/测试种子 |
| P3一次性TEST | 40（两族各20） | 成功≥32/40、各族≥16/20且0硬STOP；TEST失败仍报告失败 |
| **总上限** | **88** | 当前只是待审批预算；未签发时仍是0 |

每次记录candidate，scene，seed/reset，native trace完整消费/部分消费，首次拒绝k/通道/峰值、WBC跟踪、BASE位移、物体抓持与接触、STOP及原生积分。新安全保护停止是安全事件，不算任务成功。正式pilot/DEV/TEST有一项技术入口失败可经核实做**新技术接续身份**，但有效科学失败不能重投挑成功。

### L — 凭真正闭环失败选择下一学习研究，禁止惯性继续训练

- A2通过且P1/P2真实可成功：冻结双分支版本，继续已批准评价而不引入第三模型。
- 闭环第一个阶段异常但所有H40均合法：归因视觉目标/任务阶段、EEF目标和实际追踪、夹爪与抬升协调、base瞬态；追加可部署的历史或学生状态专家纠正只有在独立模型/data预算批准后才开始。
- 若student-on-policy状态暴露新的H40失效：记录准确的首失败通道/位置、对应预测和上一命令，不简单重新用早已达100%的专家状态TRAIN调参。优先基于新的学生失败分布提出单一变量候选。
- 高层表示或WBC/安全profile的更改属于**新的科学身份和独立批准**，不能作为工程修复悄悄上线。

未来F4持物搬运、F5抓持运输、F6放置、开门与书架整理须在实际移动抓持具备可信结果后另立数据、任务和原生准入，**不在本V3自动采集／训练额度内**。

## 3. 自主工程修复：Codex无需为每个小故障等待新GO，但必须推送可审PR

依据仓库[AGENTS.md](../../../AGENTS.md)和[REPOSITORY_WORKFLOW](../../REPOSITORY_WORKFLOW.md)的既有V7工程自主原则，Codex具备**项目自有用户空间、非安全更改**的修复自主权。自主修复不等于自动合并、放宽科学资格或无限消耗资源。

| 类别 | 允许自主修复并提交GitHub PR的事项 | 完成证据 |
|---|---|---|
| E0：纯工程 | Python导入/依赖pin、部署入口、路径/权限错误（不提权）、类型/JSON/NumPy序列化、原子写/owner lock/lease、JAX缓存所有权、超时/日志/manifest/恢复、旧服务冷接续、测试夹具 | 复现错误→最小补丁→定向单测/错误注入→不改变输出语义的数值回归→PR |
| E1：通信／时钟合同 | 请求schema、native tick/50ms历史、取消/TTL/ACK/部分P10生命周期、双adapter合并的真实race、提交后状态推进、非物理接收器接口 | **新合同版本及独立source SHA**＋前后数值/负案例、0误安装故障注入；只获非物理资格 |
| E2：学习／任务科学 | action编码、量化/统计、Loss/优化步数、训练数据成员、模型权重与行为、阶段阈值、EEF/抓点、评分、候选选择、GPU新scientific run | **不得以工程修复名义擅改**；提出单变量新计划、明确科学预算和原面板保留，经授权再执行 |
| E3：安全／物理 | WBC/MPC/腿/臂PD、原H40或夹爪40mm/s门、碰撞阈值、STOP hook在真实进程装载、解隔离、场景危险范围、实机 | **一律不可自批**。只可提交候选代码/只读诊断与重新准入申请；独立授权前新增物理0 |

**操作循环**（Codex主动完成而非只报告）：

1. 发现具体错误后先记录症状、期望观测、失败命令/输入/身份与业务影响；在有真实所有者的项目目录内做最小定向复现，先调查该任务是否已RUNNING/UNKNOWN/CLOSED，不假设进程已终止。
2. 仅当缺陷属于E0/E1时，在与现役胶囊隔离的worktree/new source branch里修复；**主动运行相关单测、故障注入与数值对照**；单测没有覆盖的真实接缝一律标未验证。
3. 生成代码diff、源SHA、测试范围、预期不变项和失败记录。在GitHub提交普通feature/ops PR，审阅回读远端files/commit。以**PR #393执行源码 7afbc0bd953c9b87821858bd70738a287ec082fc 为代码参照**，先查明当前PR基线与main包含哪些模块；main是集成入口，但不能直接从旧main复制出不存在的runtime。必要时对现役执行分支开目标明确的stacked PR，最终按仓库正常集成流程进入main。**禁止force-push/硬reset/自动merge、改动历史证据或密钥。**
4. 已封存、正在跑的科学任务永不热改。若修复会改变真实通信时序或协议，采用全新不可变执行版本和资格；旧失败保留。修复E0/E1成功后**继续下一项已获准的非物理工作**，不为每个小补丁停下索要GO。
5. 只有对已证实未启动的项目自有技术任务才能在原预算内用新技术身份冷接续，保留原失败与耗时；UNKNOWN要核process/cgroup/Journal；不重复运行有效科学失败、不能重复提交同一身份。
6. 同一个症状重复两次却无新增解释时应改诊断策略，而不是无限重试。缺少代码权限、站点allocation、依赖来源或安全放行时记录BLOCKED和最小缺失条件，不编造成功。禁止sudo、Docker/socket、改系统包/NFS/SSH/权限、删除其他用户数据、干扰健康GPU任务。
7. PR中明确报告实际补丁状态：**PATCH_PUSHED / TESTED_CPU / TESTED_GPU_NONPHYSICAL / READY_FOR_REVIEW / BLOCKED_NATIVE_SAFETY**，不能用“PR通过”冒称真实GPU/原生物理成绩。涉及安全候选的PR应单独明确标NOT_INSTALLED_NOT_ADMITTED。
8. 如Codex没有真实Core或GitHub写权限，可在本地形成完整补丁、测试报告及阻塞说明；**不得宣称已推送**。本Goal的GitHub文档不会自动启动Codex，需由有对应工具访问权的运行端接手。

## 4. 资源与无人值守执行范围

**本Goal默认新增训练0、模型权重不动、原生物理0。** 首轮只允许在**既有有效资源许可和真实独占owner**下申请以下非物理工程量；这不是自动站点GPU配额：

| 工作 | 新优化更新 | 新神经生成 | 新原生请求 |
|---|---:|---:|---:|
| A0复用历史矩阵 | 0 | 0 | 0 |
| A1非物理通信/编译与fault injection | 0 | 0（优先复用缓存） | 0 |
| A2 Core真实非物理两分支请求 | 0 | **≤64（32次统一请求×2）** | 0 |
| A3旧学生状态one-step诊断 | 0 | 优先从A2的≤64共享；若需更多另外批准 | 0 |
| S0三次碰桌只读安全审计 | 0 | 0 | 0 |
| P0–P3真实原生 | 0 | 逐获批episode实际计数 | **待独立准入≤88；当前0** |
| L科学新训练 | **未授权** | 未授权 | 0 |

双adapter热态网络p95=355.31ms超过曾用≤350ms续期目标，且不包含相机/IPC/编译；从“离线可采样”到“真实不中断周期”存在必须解释的工程风险。固定原0.4s TTL与≤0.35s续期要求，先量真实路径，不靠放宽期限或省略原检查获得资格。如果必须暂停仿真等待推理，只报告SIM_PAUSED_SIMULATION，和实时部署明确分开。

真实全链应运行于Core等实际授权计算节点的server-managed单位，不依赖前台agent、用户电脑、SSH隧道或聊天唤醒；但**必须先真实创建/核验单位和断线存活回执**，不能因为有本文件就宣称挂载完成。当前既有HYBRID与SHIFT7服务皆CLOSED、PID/cgroup清空，不存在可自动续期的后台执行。GPU独占owner、allocation、配额、OOM预检、原deadline（2026-11-19 02:51:23.855488 SGT）和日志/stop入口沿现役站点政策核验；不得静默延长allocation。私有RGB、模型、原生完整trace只保留受控路径，GitHub仅发布脱敏证据/SHA。

## 5. 终态、执行交接与验收

状态机：SOURCE_RECONCILED → A1_PROTOCOL_PATCHING → A1_NONPHYSICAL_QUALIFIED → A2_GPU_IPC_DIAGNOSTICS → A2_COMPLETE / A2_TECHNICAL_BLOCKED → S0_SAFETY_DOSSIER_READY → WAIT_INDEPENDENT_READMISSION → [READMISSION_SIGNED] → P0_EXPERT_SAFETY → P1_STUDENT_PILOT → P2_DEV → P3_TEST → GOAL_MET；任何hard STOP → SAFETY_QUARANTINE。ENGINEERING_PATCH_PR_OPEN可以和A1/A2并存；WAIT_RESOURCE、BLOCKED_GITHUB_WRITE、BUDGET_EXHAUSTED均为有效**阻塞终态**而非科学成功。

每轮自主提交应更新一个精简goal_state（实际来源和当次最新事实）、一个append-only effect ledger与GitHub PR正文的**四个字段**：新增源码/测试，完成的证据和SHA，真实新成本及旧失败，下一项获准工作或明确阻塞。Issue #11科学，#47运维，避免所有消息重复巨幅堆叠。若不满足物理独立准入，交付完整A1/A2及S0分析后报告BLOCKED_SAFETY，而不是通过无限CPU检查拖延或假装任务在跑。

**下一次有价值的交付应包含**：真实双adapter请求→完整原编译器的非物理端到端追踪；全64或实际更少的有限GPU/IPC样本的正确性和端到端延迟；完整命令历史/ACK/STOP/TTL/部分P10的故障注入；三次碰桌的可审核预防方案与安全误停代价；已推GitHub的工程补丁PR及代码SHA。真实移动抓持必须在独立READMISSION以后才报告新的科学分母。

本文不改变旧PR392/393/394结果，不提交/重训/复采、不批准新的native请求。**全程以真实可用工具为边界，行动与发布事实必须有运行回执。**
