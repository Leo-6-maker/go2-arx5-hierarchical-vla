# M_FREE / SLA V4：SHIFT7单模型数据泛化 → 持续非物理GPU → 安全准入 → 闭环学习曲线

**2026-11-17（Asia/Singapore）** · **文档状态：PUBLISHED_EXECUTION_PLAN_NOT_RUNNING**  
**基线：** [PR #396](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/396) @ `3c576f5fa29afcccc5ca143a4a4ab6d40d67f8d3`（Draft、Core零GPU工程终态）；[PR #395](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/395) @ `7930e9a6ccac4d0dff0c2bb402323b22d2ca9100`（最新GPU+IPC实测）；[PR #393](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/393) @ `7afbc0bd953c9b87821858bd70738a287ec082fc`（SHIFT7及CN38离线实测）；[自主推进Goal #394](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/394) @ `bcc2c67be723149e38998a36802f218caa666917`；[历史C20及事故 #391](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/391) @ `65bd544d8ed3bf6a717783f8691fcb95c8d12165`。  
**执行源码参照**：PR396实际Core运行源码 `ec71edb0038f3c9d8c6543053d74bd2b6e1054d7`；PR395实际GPU源码 `3c95a04a70d7772e347763c6f4541046ff13d2b0`。报告commit不是现场热更新；`main`是PR集成入口，**尚不能假设包含上述执行模块**。

## 0. 为什么本计划替代旧推进顺序

本项目目标不是再获得一条合法H40预测，而是在冻结Go2+ARX5 WBC/MPC/腿RL/臂PD、原H40/P10/50ms、2ms物理和所有安全门的前提下，交付可以在**M_FREE F1/F2**安全地连续移动、抓取、将物体抬升≥20mm并保持≥1s的SLA。任务方向底座净位移≥8cm必须另外计分；不使用教师阶段、物体真值、BASE oracle、强制闭爪或任务接管。

**已实测并冻结，禁止重做/重新评分：**
- C20：20成功专家episode／1376窗，C20父模型1000真实optimizer更新×batch32=32000训练访问；旧学生原生抓持/任务0/13，11未启动；CN38专家38合格／41真实物理请求，2次自身非法接触+1次共享隔离中止。
- TIME500：CLOSE20/20、全96为88/96；SHIFT7单模型500新更新/16000访问，固定TRAIN96/96；FM500-EEF+TIME500-Grip和FM500-EEF+SHIFT7-Grip均TRAIN96/96。
- **仅HYBRID首选**在未用于C20训练的CN38额外18EP上离线180/180（F1 140/140，F2 40/40）。这是**记录的专家状态／相近布局／相关窗口**，非新盲TEST，更不是学生闭环。**SHIFT7单模型尚无CN38实测**。
- PR395新增32联合非物理请求／46实际分支生成，普通GPU 16/16 H40+替身ACK，F1/F2各8/8；旧+新两scope共64请求/98生成、0训练/原生。原热态15条GPU全链p95=525.730ms、模型p95=354.802ms；RGB首读48–166ms，ACK后串行归档热态中位298.725ms；原400ms TTL与≤350ms续期目标仍失败。
- PR396已在独立Core CPU完成**R0/R2**：32/32缓存完整H40/EEF wire逐字节一致、272/272非物理atom、32次归档624文件SHA一致、73项定向检查，队列最多4条，部分P10/STOP/UNKNOWN保护保留。**0新增GPU、0训练、0native**；文件路径可能页缓存预热，内存路径仅已录RGB，异步归档自身耗时没有消失。PR396的优化**尚无GPU持续2Hz资格**。
- 三次已保存native碰桌、监测误拦1/38成功专家、最短被动提前告警12ms、controller-selected reference未知、STOP hook最坏响应未测；F1/F2专家与学生、M_ASSIST共同隔离，原RP8/JOIN10/DEV_TEST失败不变。
- 既有SHIFT7、CN38、PR395/396执行服务均CLOSED；未注册新的R1/R3 GPU工作、未签署独立READMISSION。原allocation截止2026-11-19 02:51:23.855488 SGT，本文件**不续期**。

**研究优先级：R1（单模型能否替代双模型）→ R3（获选架构是否能持续2Hz/满足TTL）**，与S0安全归因/独立准入并行。已完成的R0/R2只接受新源引用，不再重新执行。关闭“再训练几千步来修复推理时间”的错误路径。

## 1. R0（已完成，不重复）：工程输出回归与资源对照

- 固定PR395的同输入/noise普通16条，PR396的文件／已交付内存双路径各16回放结果归档为基线。
- R0保存原热态15条逐行配对延迟、ACK＋归档串行下界p50=682.908ms/p95=843.030ms，以及PR396 CPU缓存input→ACK结果（文件p50/p95=12.818/46.021ms；已交付内存3.739/3.888ms）；**不可将GPU/CPU和相机路径混报同一统计样本**。
- PR396运行内容不重复：VerifiedObservation单次SHA+解包／只读内存H40／异步ArchiveReserve→Seal→Queue→Durable SHA→COMPLETE，最多4条预留/队列/执行，64MiB总live空间。
- 归档线程实际p50/p95=348.908/436.401ms，是被移出请求串行路径，**不是磁盘速度变快**。后续GPU实验必须同时计ACK与下一次派发间隔，检查队列背压、UNKNOWN与故障停止，不允许将归档等待藏在测量之外。

## 2. R1：零训练的SHIFT7单模型CN38 episode-heldout资格（独立GPU额度，尚未授权启动）

**科学问题：** SHIFT7已在TRAIN96/96，能否在CN38未用于C20训练的18EP（但仍是相近布局）保留完整H40、闭合时序和EEF几何，从而从双adapter的两次串行forward降为一次？

**锁定身份与输入：**
- 新候选固定PR393的SHIFT7_500完整包、parent、codec、冻结C20统计、原10步Euler/40×32、noise_ids=[0,1]、两相机RGB+7D状态+语言+policy frame+部署可得的上一真实发出夹爪参考及50ms时钟。禁止重新训练或加入CN38到TRAIN。
- 对照使用**相同输入/noise的已保存**FM500 EEF＋TIME500夹爪CN38输出，按实际冻结文件SHA读取，不重新生成HYBRID；不能以专家phase/未来真值作为学生输入（phase只用于事前选锚与评分）。
- 在18EP中先按固定ID排序：F2取全部4EP；F1从14EP的下中位分层四分位取4EP，具体位置0-index=[1,5,8,12]；每EP固定5个阶段的原lower-median query锚点、每锚点noise0/1。**首波8EP×5×2=80条生成**（F1=40、F2=40；CLOSE=16）。
- 首波读所有完整nominal/reference H40各通道、真实参考接缝、夹爪crossing≤100ms、终末开度、EEF时序/对专家位姿误差、旧接触净空shadow指标。对比HYBRID在**这80个完全相同条件**的指标，而不是与全180混比。
- **首波开发GO门**（非安全准入）：80/80完整H40、16/16 CLOSE且crossing及时、无漏开闭、首点夹爪参考速度≤40mm/s，EEF位置误差分布p95不劣于同条件HYBRID超过0.5mm、单点最大不劣于同条件HYBRID超过1mm。0.5/1mm是**事前定义的相对开发阈值**，不是真实抓取误差容许值；任一失败保留负结果，不自动补种子或调门。
- 仅首波GO才扩至其余10EP×5×2=100条，**R1总模型生成上限180条（80+100，不是80+180）**。预定义全银行优选门为180/180合法、36/36 CLOSE事件≤100ms、F1 140/140、F2 40/40及同条件EEF误差不大幅回退。未达到则保留HYBRID作为唯一部署性能候选；不得擅自调整阈值或按query择优。
- 16/16 SHIFT7老RGB新进程恢复已经通过，不需要为“同模型重载”无意义追加生成；新运行只在独立会话绑定、完整身份/时钟/输出SHA和普通RGB重放中计入实测。
- R1输出完整逐EP、family、stage、noise、编译通道与首超限k、峰值/阈值、EEF动作与专家几何误差、事件时间、latency/显存、所有实际生成/成本；即使180/180也**仅为专家状态离线资格**，不能宣称学生on-policy成功。

## 3. R2（已完成，不重复）：PR396内存输入／异步归档

- 不重建另一个scheduler/ledger，不重新消耗原已使用完的PR395 32请求额度；将PR396`verified_observation_v1.py`、`async_request_archive_v1.py`、缓存Receiver等作为新不可变源的**显式依赖**。
- 保证所有提交安装的H40原编译器、M_FREE不发送BASE、只一次夹爪机械映射和参考绑定。允许Codex处理项目内路径、导入、IPC/消息、Journal、序列化、锁竞争、有限背压等**E0/E1工程问题**，必须定向故障注入和GitHub PR；禁止热改已封存执行者。
- 需保留`reserve`先于任何模型作用、异步归档完成后哈希回读、崩溃/丢失未归档attempt记UNKNOWN且不自动重试；队列限4且容量满足原64MiB上限。非物理异常停止不删除原成果。

## 4. R3：选定唯一架构的连续GPU非物理请求实验（另一项独立新GPU额度，尚未授权启动）

**入场条件：** R1已封存**实际**终态或有界失败，冻结全局选择SHIFT7单模型（R1通过）或原FM500 EEF＋TIME500 Grip HYBRID（R1未通过）。**不做两架构并行调参或按query切源**。

**固定面板与计算预算：**
- 在Core持有新的项目自身owner/lease和实际站点allocation后，最多**32个新联合请求**；F1 16条＋F2 16条，事前选定合格历史录制RGB连续片段（每族16个固定50ms/0.5s边界的状态、命令参考需核真实时钟），固定noise身份，单一不可变source。
- 若获选**单模型**：最多32次完整SLA生成；若获选**HYBRID**：最多64次分支生成；**两种情况都是同一个32请求上限，不能相加为96或自动补跑第二种方案**。R1最大180+R3最大64=**本计划新GPU生成不超过244次**（条件上限，不是现有许可）。
- 真实GPU/IPC＋PR396的verified-memory、内存H40、最多4条异步归档；使用原`compile_proposal`、40点全门、原wire和非积分receiver mock ACK。不得直接调用MuJoCo/WBC动力学、生成新物理tick或以SLA预测后缀充当真实已消费原子。
- 面板包括一次冷启（单列首次import/JIT）、后续热态连续名义2Hz发布；**不能事前读取未来RGB作为实时相机优化**，记录专家捕获历史源与实际送入内存边界；保存RGB流不是现场相机，不能冒称相机到WBC闭环。
- 分开比较`SIM_PAUSED_ONLY`非物理语义与严谨墙钟4百毫秒TTL评估。所有结果要报真实input-available→request→model→compile→ACK、实际wall等待、归档背压/队列占用、下一query可派发时间。仿真时钟冻结并不能把墙钟失败写作REALTIME_PASS。
- **预注册工程目标**：热态全链p95≤350ms、最慢有效请求<400ms，严格wall TTL/剩余50ms检验无失效安装或未保护的续发；名义500ms间隔中无漏发/背压阻塞，最多4个归档任务，所有真实完成branch和32请求能哈希封存；与各自冻结模型在同输入/noise上的解码/H40原门结果一致（改变计算图则新数值/候选身份）。
- **足够精确的失败分流**：A) 单模型通过离线但实时失败：量单模型计算与输入/IPC/归档；只做不改变模型数值/时钟安全的独立工程对照；B) 单模型R1失败：HYBRID性能测量，若仍超过阈值则明确BLOCKED_REALTIME，可另申请SIM_PAUSED_NATIVE协议的安全审核，**不能自动部署**；C) 满足2Hz但墙钟严谨TTL仍不合格：分开报告吞吐与deadline，不拿一个代替另一个；D) 背压/UNKNOWN或部分P10/STOP失败：立刻结束新派发，按Journal真实成本封存，不回放未知effect。
- 32只是有限工程资格分母，不是稳定性或硬实时认证；即使全通过仍未取得实际相机/native消费/碰撞安全资格。不要以“32/32”作为GOAL_MET。

## 5. S0：与R1/R3并行的**独立**安全准入，原生预算仍为0

- 保留旧三次碰桌：旧学生右pad与桌接触，两次新专家`ILLEGAL_CONTACT`。原被动监测对3/3有历史提前告警12/34/316ms，但**误拦成功专家1/38**；旧零误拦筛选FAIL。旧C++仅冻结状态数值核对，并未安装或通过真实native STOP hook/最坏延迟资格。
- 汇编每次contact pair、pad/TCP/base、真实reference age与controller-selected reference（若无法证明，必须标UNKNOWN）、2ms历史、原STOP后积分0、实际MJB SHA及WBC/solver/gains/profile SHA；S0独立责任者而非模型开发者决定风险接纳。
- 可以自行在**非物理**代码环境修复监测候选的路径、时间绑定、异常fail-closed和审计，实现并推送普通PR；**不能**自签READMISSION、装入native hook、放宽40mm/s/H40、屏蔽碰撞、改变WBC/MPC/腿/臂PD、重启隔离family。
- 如果拟议停机策略接受少量安全误停，必须事先由有权限审查者明确写出新错误拦截成本、停止延迟要求和样本面板，不能对已保存1/38进行事后重评分。
- 真原生仅在签发绑定**代码SHA、模型/codec、MJB与布局SHA、冻结WBC+安全配置、STOP/hook资格、责任人、有限request面板、硬停止处置**的独立`READMISSION`后，可条件性申请专家安全8、学生pilot8、DEV32、一次TEST40，最多88个native episode，**当前全部0**。任何硬STOP立即按获批profile隔离；未签发时结束为`BLOCKED_SAFETY_WITH_EVIDENCE`。

## 6. L0：闭环数据量／训练步数研究（未来决策，不属本轮可执行预算）

用户真正关心的是新本体闭环成功率对示范数量与optimizer步数的影响。当前**不存在该函数的实测学习曲线**：C20 1K/32K及后继500/16K中动作表示也变化，不能将不同candidate差异归因于数据量或单纯训练时间。当前CN38的额外18条**未进入现役模型训练**。

只有获得真实原生学生pilot/DEV失败归因且独立获得新训练预算，才登记**单一冻结动作表示/模型初始化/统计/训练schedule**的实验；不要同时变更action head、LoRA范围、EMA、空间loss、normalizer和任务场景。

建议条件性设计（**未授权、无自动GPU训练**）：
- 数据因素：C20原20EP vs C38（包含原20＋额外18EP），审计实际sceneSHA/episode，不把38当38种独立布局。
- 时间因素：从相同冻结C20父权重开始，两数据臂与**两个训练seed**分别新增1000优化更新（batch32，B32训练访问各32000），在**新500和1000更新**处检查：共2数据组×2seeds=**4条训练轨迹**、实际新增**4000更新／128000访问**、8个500/1000检查点。这里原C20父1000另列，单后继累计1500/2000更新，绝不把4臂相加当单checkpoint步数。
- 相同旧C20 quantile统计冻结以隔离仅“多加18EP”；训练和推理均用一条事先锁定的表示（若SHIFT7胜出就选SHIFT7），共享LR/访问规则并明确`C20`和`C38`的阶段采样等量规则。不能靠新的独立normalizer混淆数据量效果。
- 一旦C38被用于训练，旧CN38的180专家状态**不再是held-out**；必须准备新的独立DEV/TEST并记录闭环学生状态分布、抓持/抬升/保持/8cm移动与硬STOP。物理准入仍独立。
- 如果pilot显示碰桌/跟踪/接触不可控，先解决安全/动态跟踪，不能将其归为“训练步数不够”；如果模型合法却抓空或时序冲突，优先记录实际学生可见状态/纠正数据，后登记最小数据增益实验；不得自动C60/C100、SHIFT14、RL、真机或复杂技能。

上述4K/128K只是**条件性科学预算示例**，不是本V4授权的训练额度，当前本轮新增训练必须为0。

## 7. 自主工程修复与Codex状态机

遵照仓库`AGENTS.md`及`docs/REPOSITORY_WORKFLOW.md`，Codex可以在**项目自有用户空间**自主定位并修复：Python路径/依赖兼容、序列化与hash、输入内存缓存、非物理IPC、GPU worker生命周期、Journal/owner/UNKNOWN、JAX编译缓存、限时背压和编译器调用接口；每次最小修复在独立worktree和新source SHA，跑相关CPU故障注入、必要的有界非物理GPU回归，然后推送普通GitHub PR，不需为每个工程小故障等待用户再次GO。

**不属于工程小修：**新模型训练/数据更改、推理动作表示或数值语义变更、删掉完整H40后缀、放宽安全阈值、WBC/腿/PD/求解器/物理参数、原生hook真实装载、解除隔离、运行新native面板或未经许可超出资源/训练预算；这些必须有新的独立科学或安全批准。

状态机：`READ_LIVE_REGISTRY` → `R1_SOURCE_VERIFIED` → `R1_GPU_AUTHORIZED`（或`WAIT_GPU_AUTH`）→ `R1_80` → `R1_180_CONDITIONAL`/ `R1_NEGATIVE` → `ARCHITECTURE_FROZEN` → `R3_GPU_AUTHORIZED`（或`WAIT_GPU_AUTH`）→ `R3_32_NONPHYSICAL` → `R3_REALTIME_PASS`/`BLOCKED_REALTIME` → `SAFETY_READMISSION_PENDING` → 只有独立签发`READMISSION_GRANTED`才进入物理P0/P1/DEV/TEST。安全事件从任何阶段进入`SAFETY_QUARANTINE`；历史有效负结果不补做成功。

失败处理：UNKNOWN先核对PID/start/cgroup、owner、Journal与磁盘真实effects，绝不盲重投；一个已完成的科学失败不重做；若只修未执行的入口错误才可用新不可变技术身份冷态接续，并保留失败及成本。禁止提权、修改系统包/SSH/NFS/其他用户GPU、force-push、自动merge、删除旧原始数据；新PR只能用正常stacked branch或正常main集成方式。**PR发布、CPU回放、GPU离线与原生安全资格分层计分。**

## 8. 精确预算、验收与交付

| 研究工作 | 允许的新optimizer/访问 | **建议申请、非当前授权**新神经生成 | 新原生物理 |
|---|---:|---:|---:|
| 已完成R0/R2的复用 | 0/0 | 0 | 0 |
| R1 SHIFT7单模型CN38 | **0/0** | 80，若GO则再100，**总≤180** | 0 |
| R3选定架构32请求 | **0/0** | 单模型≤32，双分支≤64；**一个架构** | 0 |
| S0独立安全审查/被动历史回放 | 0/0 | 0 | 0 |
| L0未来数据量×步数实验 | 未授权（可申请4K/128K） | 未授权 | 0 |
| 新native能力评价 | 0/0 | 依已准入科学面板记账 | **独立READMISSION之后条件性≤88，当前0** |

因此R1/R3计划条件性**GPU模型生成总封顶244次**，不是已获准计算；未获准则只做只读/CPU安全资料并标`WAIT_GPU_AUTHORIZATION`。R3请求上限32不能因为只使用32/64分支预算就自动多发第33个请求。所有资源、原截止、GPU负载必须实际重新核验；老Core进程CLOSED不等于可以免注册启动新任务。

每个阶段交付：`goal_state`及append-only effects（source/spec/模型/数据/统计SHA、实际注册与scope、更新/访问、神经生成、请求与物理tick、全部任务/异常/未启动分母、CPU/GPU内存/墙钟、latency p50/p95/p99/max、所有原始失败），再发布脱敏`ACTUAL_RESULT.md/json`及独立重放SHA。R1需要按18EP/2family/5phase的完整结果；R3需要32请求逐条ACK/归档/背压和冷热时延；S0需要签发与否的清晰结论。GitHub仅发布普通PR与脱敏摘要，私有RGB/权重/原生厂商资产留在私有Core。

**最后一句执行定义：拿到一个带来源和严谨时钟的可推理策略、真实墙钟与2Hz证据、独立安全放行后，才可以进行学生原生闭环；任何一项没有通过，就在该层终止并交付可行动的阻塞。**
