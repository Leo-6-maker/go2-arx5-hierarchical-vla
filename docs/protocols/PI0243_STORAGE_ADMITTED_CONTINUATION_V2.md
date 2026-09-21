# PI0243 — 清理后的存储准入与训练续行 V2

日期：2026-09-21（Asia/Singapore）  
审核基线：`991911b00c3f504913498d9b702814f0cdd7d671`  
决定：`GO_STORAGE_ENGINEERING_THEN_CONDITIONAL_STAGE_A_CONTINUATION`  
授权：`PI0243_STORAGE_ADMISSION_AND_CONTINUE_GO`  
发布状态：`PLAN_PUBLISHED_NOT_EXECUTED`  
Issue #11 canonical；PR #77 仅 pointer，不 merge。

## 0. 本次决定

历史清理已经收口，不再以继续清理旧模型作为默认下一步。现在完成真实写入位置、容量、完整checkpoint持久化和消费路径的闭环，再按原PI0243继续Stage A。工程问题自主修复，没有固定次数/qsub上限；科学预算和阶段标准不变。

这是PI0243的工程补充，不是PI0244或新的科学recipe。原协议commit `5d6d843235625c416d372cdbed01b185a1e6da5f`、Issue5755883470的A→B→C→selection→TRAIN sim条件推进继续适用。通过本次工程准入后明确允许提交新的、唯一的Stage A有效执行，不必再等“是否可以重提”。不修改旧626184或覆盖其证据。

**立即可执行：**本地/Corelab代码与CPU测试、限定路径存储盘点、现存checkpoint身份核对、已授权项目存储接收链路、既有账号下HF写读探针/上传恢复、必要的薄I/O修复、生成新执行binding与admission。

**条件可执行：**工程准入通过后，恢复经过核验的同一Stage A完整Trainer状态，或在没有有效恢复点时从原initializer/seed重新执行原5000步。依旧没有额外训练arm、seed、loss/LR/模型/精度/ABI/gate变更；后续阶段只按原科学门推进。

**本次发布未执行：**GPT未SSH、未qstat、未测试实际磁盘、未读远端权重、未上传/删除HF内容、未提交训练。以下事实来自GitHub账本/源码；具体机器/容量/凭据/恢复点须由PC agent核验。

## 1. 接受的证据和必须分开的结论

来源：
- 清理canonical：Issue #11 `5758674618`，PR pointer `5758677756`。
- `reports/PI0243_STORAGE_CLEANUP_CLOSURE_20260921.json`（审核基线）。
- 前次执行 `5757845729`、存储授权 `5758072436`。
- `hpc/pbs/pi0243_stage.pbs`、`scripts/pi0243_staged_learning_to_sim.py`、保留修复 `0debbc0895d9000445b7b851fde8c4404a5a8c73`。

清理报告记录：PI0241三份完整checkpoint已在HF固定revision `2f9fb89b0788890a01391fec9c4abce8c37e76d9`验证；指定本地副本已删除；回收39,955,988,480 bytes，约37.2 GiB。该数字属于清理基线的观测，不是当前训练盘剩余配额，也不证明另一个节点的/hpctmp已释放空间。

报告的存储身份是 `hpfs.nus.edu.sg:/hpfs-main` 挂载到/scratch；另有不同df视图，不能把系统总空闲、项目/用户配额、compute mount和删除量混成一个可用容量数。失败626184在hopper-29创建 `/hpctmp/e1192548/pi0243_stage_a_runtime_r2_20260921` 时即No space left，Exit1/Stageout1，Python/model未开始。日志SHA `d9c7bb71dbf262f32ce4cfc32bf7c4e777a50f3d0de4efb510cd0b3e31684cc5`。

625381/625384/625949当前保存在Project_Storage；HF写入HTTP401，不能改称HF已归档。小closure不是每个历史payload的全量复验：例如625949 source_bytes与archive_bytes不同的含义由已有逐文件manifest解释，不能仅凭两个数字断言损坏或完全等价。缺少说明时补小型证据索引，不为此重下载全部历史权重、扫描全盘或扩大清理。

“未完成checkpoint-1000”在本次小closure中没有足够的具体job/path/state证明，不能凭名字当恢复点。保留，不删不自动加载。

计数分开：本轮cleanup和626184的训练/model计数为0；历史625949实际做过250 updates但无有效checkpoint，其工程消耗保留，不能将整个PI0243历史计算重写成0。

## 2. 已确认的代码问题：为何清理后仍不能原样qsub

当前PBS：先mkdir输出目录，再用test -w；没有完整checkpoint峰值/配额验证。随后运行整个train/screen，结束后才调用 `archive_output()` 执行一次rsync。这个结构无法证明中途checkpoint写盘安全，也无法保护全部长作业进度免受节点/工作盘故障。

该PBS还明确写明不同compute节点的Project_Storage挂载不一致，所以不能把默认输出直接改成/Project_Storage并假定每个节点可见。应先解析并验证实际挂载，必要时使用已有合法数据传输路径。

`train_stage()` 当前拒绝复用已存在的training目录，直接调用official训练入口，没有公开的已验收resume参数；最终核验又要求1000/2500/5000全部在output/training下。screen同样直接从该目录载入模型。因此：

1. 不将“把max_steps改小/把initializer指到旧权重”当exact resume。
2. 不盲目套用PI0241的1000步恢复脚本，避免max_steps、cursor和save steps被旧常量覆盖。
3. 未修复并测试训练finalizer、恢复loader、screen三类消费者前，不实施“上传成功就删除当前checkpoint”。
4. 保留0debbc的sealed save gate，只保存A的1000/2500/5000。用CPU控制流测试确认所有实际callback组合不会额外写step250；compile-only不够。

只修正在调用的路径，不新建通用训练框架或复制G0–G3。

## 3. Goal 1：冻结三个存储角色，而不是继续换目录碰运气

采用唯一 `storage_binding.json`：

- `work_root`：实际训练进程能够创建/写入的工作盘；
- `durable_root`：节点失败后仍可读取的已授权项目存储/其他既有持久存储；
- `hf_publisher_host`：持有可用HF写权限、能够读到durable数据的PC/Corelab/获准传输主机；不默认就是Hopper compute node。

同时列出TMPDIR、HF_HOME/HF_HUB_CACHE/HF_XET_CACHE、Torch缓存、Python缓存、日志、模型缓存、screen恢复区及传输临时区的真实写入位置。不要把满盘问题从output挪到默认/home/.cache；只迁移实际会写的缓存，不复制整套原runtime/模型。训练保持offline，HF uploader是独立online进程/环境，不全局关掉训练offline限制。

每个主机/路径记录：hostname、canonical realpath、mount source/type、filesystem/fileset标识、用户/项目容量和inode可用量、采集命令/时间、保留/清理政策、是否compute可见、是否重启后持久。`df`和`du`都不是完整quota证明。findmnt/stat等仅提供实际可得的事实；站点规则/配额接口不可得则标UNKNOWN，不编造NUS统一政策。

选择顺序：
1. 优先选择已验证compute可写、站点允许、能容纳本阶段完整输出的持久或共享work_root。经验证可用的Project_Storage可以直接写；没有compute可见性证据就不能选。
2. 若只能通过传输host访问durable_root，work_root必须有足够checkpoint缓冲并完成逐checkpoint持久化。默认保留本地A三份直到screen结束，路径消费最简单。
3. 只有前两者容量不够且下面的rolling消费者测试通过，才允许本PI新产物滚动迁出。不能为了维持不停训练先删恢复点。
4. /hpctmp只是候选，不再享有“名字不同所以容量足够”的默认假设。若无任何合法可用工作空间，报告具体fileset/缺口并请求站点解决；不申请H200反复试哪个目录能mkdir。

本地/Corelab完成轻量盘点和代码测试；Hopper登录入口仅按站点允许做轻量检查/传输/调度。共享work_root的资格尽量在提交前建立；只有实际分配节点才能判断的挂载，于下一次正式作业的shell前缀复验，失败在模型加载前退出并留下结构化原因。该复验不是额外科学实验，也不是绕过scheduler SSH计算节点。

## 4. Goal 2：从真实checkpoint大小计算容量和故障余量

从已封存同配方/同trainable范围的manifest读取完整Trainer大小、模型大小及optimizer等构成，不为测大小重新训练。若没有完全可比的样本，依据实际权重/optimizer结构建立保守上界，并在首份真实checkpoint后更新。记：

- C：完整checkpoint的保守bytes上界；包括所有shards、optimizer、scheduler、RNG、TrainerState及必要配套文件；
- T：除最终checkpoint目录外，序列化/复制额外临时峰值；
- R：screen、日志、缓存和传输缓冲等的新增预算；
- I：同fileset不可避免的新输入/恢复副本。

**A阶段retain-all布局**预算约3C+T+R+I，缺乏T实测时取T=C，即4C+R+I。headroom沿原PI0243：max(10GiB, 20%新增峰值)。比较实际quota free时只计未来新增部分，已存在同盘输入不重复计费；不同主机/fileset分别计算，不用一个总数覆盖所有盘。相同device/inode硬链接只算一次真实占用，说明逻辑size和allocated blocks差异。

**经过测试的rolling布局**：work_root至少容纳上一份可恢复完整C、新写C、额外T以及仍需驻留的screen/model copies、R、I与余量；在T很小、无额外model副本时也约2C，绝不能默认为C。durable_root仍需要保存A全部3C，加其临时文件与余量。HF上传变慢/中断时仍须有可持久化的接收区；不能以未验证吞吐假定下一checkpoint到来前一定清空。

小文件create/write/fsync/rename/read/hash/lock验证是功能证明，不是容量证明。容量来自实际quota/站点分配/已批准可验证的阶段容量证据；可在站点允许的CPU/数据传输资源里做受控大文件I/O或reservation，不在login做未经允许重写盘。不能用sparse truncate充当占用证明，reservation释放后的余量也不是永久保证。禁止故意写满全盘。

形成明确的 `CAPACITY_PASS` 或 `CAPACITY_NOT_PROVEN`，不写“37.2 GiB已清，所以应该够”。下一job在加载模型前复验同一binding；每次协议save前再次核验未来峰值。不足时在合法边界暂停/封存已有进度，不能跳过协议checkpoint后继续计算。

## 5. Goal 3：持久归档先于删除，HF认证独立解决

### 5.1 最低训练条件不是“Hopper能上传HF”

首要验收是：完整checkpoint能从writer送达已验证、容量足够的durable_root，被另一个合法consumer按manifest读取。只要这条路径可靠且有完整stage容量，Hopper HF401可以保持 `HF_REMOTE_COPY_PENDING` 而不阻止Stage A。

HF上传任务移到PC/Corelab/已经授权的传输host执行，优先复用其既有凭据。不能说“配置了HF所以节点拥有写权限”；也不能因为公共repo读取成功认定认证成功。

### 5.2 定位401，不打印秘密

在真正uploader进程检查：SDK版本、实际账号/endpoint、token是否存在及来源（explicit/env/token-file）、目标repo和写权限。HF_TOKEN可能覆盖磁盘token；HF_HOME改变可能改变token文件定位；这两项只是待检验原因，不预设为本次根因。命令输出仅保留脱敏账号/角色/错误，不输出token、完整环境或credential文件。不将token放在Git/PBS argv、-v参数、日志或聊天中。

在独立允许写入的probe前缀上传无敏感内容小JSON，按返回commit/revision读回比对hash；再用受控小binary验证与真实大文件相同的上传后端。小JSON成功只说明控制面可写，不保证大文件路径/空间/吞吐成功。复用现有SDK能力，不升级训练runtime。probe内容不覆盖canonical；新repo默认private，既有可见性不自动改变。

无有效写权限且PC/Corelab也没有合法凭据时，输出 `HF_AUTH_REQUIRED` 和最小人工操作；不能索要在聊天粘贴token或绕过认证。已有durable_root足够则训练仍可继续，HF复制保留pending；不能删除最后一份持久源。

### 5.3 checkpoint发布流程

`SAVING → LOCAL_COMPLETE → DURABLE_VERIFIED → HF_VERIFIED（可后补） → RELEASE_ELIGIBLE`。

LOCAL_COMPLETE必须由实际save结束事件、完整文件清单/hash、Trainer/global_step/max_steps一致支持，不只是目录存在或两次size不变。seal/receipt放sidecar，不更改原payload。writer停止修改后再复制；消费者拒绝.partial及缺receipt目录。

每个协议checkpoint完成即送达durable，不等全训练exit。目标端先写临时前缀，文件齐全后核验并发布ready receipt；不同filesystem不声称rename是跨机原子事务，原子发布仅在接收端自身目录内。最新完整恢复点始终保留；恢复、保存或screen正在持有的文件不删除。

当前代码需要的最小改变：
- save完成后的持久化hook/外层协作；不得改变模型随机流、优化器或step时序；
- 同一个checkpoint resolver供train finalizer、resume和screen使用，解析路径+immutable content身份；
- rolling模式下在消费前恢复到已验证本地cache，模型文件只是可重建读缓存，不代表完整Trainer状态；
- 只有本PI新产生的、已全量durable验证且无reader/恢复依赖的冗余work副本，才允许按原PI0243删除权限回收。旧initializer/runtime/data/旧sealed权重不扩大清理。

优先retain-all可靠布局；不要为了“架构完善”强制引入rolling。无论哪个布局，完整科学checkpoint总集和screen时点不变。

## 6. Goal 4：先完成无模型存储端到端，再提交训练

最小CPU fixtures在现有脚本周围完成：

- callback组合在0..5000只保存1000/2500/5000，测试异常step250不得写；
- 小型synthetic Trainer-like payload含model/index/optimizer/scheduler/RNG/state，各required字段、shape/header按真实格式测试，无需本项目模型forward/backward；fixture不得进入科学记录；
- save→durable接收→完整性校验→resolver→screen/readability与resume-state读取；
- 任意缺shard/截断optimizer/错误max_steps/错family必须失败；
- HTTP401、低容量、传输中断、正在读取与重复publisher均不删除唯一有效恢复点；
- 新进程/不同cwd仍解析到同一文件；on-save发布不消耗训练RNG；
- checkpoint存在但无完整性证据不升级为resume；
- 训练return code与传输return code分别保存，后者不能覆盖原先失败原因。

数据移动测试使用已授权路径；大规模校验在Corelab或站点允许的CPU/传输资源运行，不使用H200来调通普通Python/SSH/HF错误。

改PBS：最早可行位置设置错误封存；对mkdir失败在stderr与已验证的小型证据路径保存host/path/filesystem/errno。保留站点批准的PBS spool/stageout方式，不重现未经验证的直接输出路径问题。rsync/SSH/HF调用设连接/无进展超时，状态写盘；失败保持源文件，修复后可恢复传输。工程尝试总次数不设上限不等于一次传输无限挂住GPU allocation。

发布独立 `storage_admission.json`：protocol、代码、旧/新card映射、科学配置digest不变、mount/quota/容量表、CPU end-to-end、durable receipts、HF status、恢复决策、剩余steps/checkpoints、真实启动命令。只有 `WRITER_CAPACITY_PASS + DURABLE_HANDOFF_PASS + CONSUMER_READABILITY_PASS + SCIENTIFIC_IDENTITY_UNCHANGED` 才放行模型加载。HF_VERIFIED不是刚性前提，但HF是唯一durable路径时必须通过，且本地故障缓冲有证明。

更改I/O代码、路径binding或实施resume时重封执行身份，保留旧card及变更映射；不要在sealed代码原地打补丁后继续声称旧SHA有效。

## 7. Goal 5：明确决定resume还是restart，然后完成原Stage A

先读取已有运行ledger，排除并发job和新supersession。625949、626184已终态；本授权允许在上述门通过后提交新job，不再继续“等626184”。

### 7.1 唯一合法恢复决策

保留用户提及的未完成checkpoint-1000并核对其具体path/job/family/card。必须同时证明：

- 属于PI0243 A、原16cohort/schedule/initializer/recipe一致、max_steps5000；
- 模型index、全部shards及optimizer/scheduler/RNG/TrainerState完整、内容hash稳定；
- global_step、schedule cursor、实际消费及有效attempt lineage可追溯；
- 官方Trainer resume能正确载入这些状态，不重复手动skip、不重新初始化scheduler；
- 保存数值语义和科学有效性无冲突。

全满足：从最高完整、合法且未被科学否决的A checkpoint做exact resume，保留0/1000/2500/5000 screen集合和first-valid结果。必要时薄适配已有官方resume入口及恢复日志，不使用旧PI0241硬编码max_steps/step列表。

只具备部分weights/日志/目录名，或属于PI0241/其他family：`NOT_RESUMABLE`。不从partial weights继续，不拼文件“补齐”checkpoint。若不存在任何合法A恢复点，授权从PI0226 baseline和原seed/schedule重启A；是同一冻结科学试验的工程恢复，不是新增recipe/seed。历史无效updates和失败job全部计账，不把626184称从step250恢复。

代码复用与修复已授权；缺必要证据不能将不确定性写成exact。明确矛盾涉及科学配置时回PI；仅无有效恢复点时按上述restart分支继续，不额外等PI。

### 7.2 后续执行

第一验收点是原协议step1000的完整LOCAL/DURABLE receipt与可消费性，不是再汇报一次“排队已提交”。然后同一已批准A继续2500/5000，不人为把1000增加成新的PI科学门。

按原协议完成64个唯一A screen cells（0/1000/2500/5000×16；同身份可复用），最终step5000仍四项联合标准：REJECT0、median XYZ RMSE≤0.020m、p95≤0.035m、median SO3 RMSE≤0.050rad。没有因空间问题缩小训练步数、只保存权重或删除候选。

A_PASS自动进入原B/C/selection/one-chunk/receding门；本补充不扩大后续科学范围，不提前读DEV/F。A有效FAIL仍停原科学门。存储/认证/权限失败按工程或外部基础设施分类，不伪称模型科学失败。

## 8. 实施与交付边界

只在独立execution分支做最小存储/启动/恢复适配，优先修改成熟PI0243/PI0241保存与消费者入口；不用新科学编号或通用orchestrator替代任务。工程不限次数，单stage一个active owner，first-valid结果冻结；无新monitor。

输出尽量集中在4项小证据：
1. `storage_binding.json`和每fileset容量表；
2. `durable_handoff_test.json`（HF状态及测试、CPU回归摘要）；
3. `resume_decision.json`与`storage_admission.json`；
4. 每checkpoint receipt及Stage A原训练/screen closure链接。

报告未完成checkpoint的身份决定、实际work/durable/HF publisher、有效容量而非df猜测、HF写入是否恢复、job/steps、canonical与engineering计数和下一动作。不把“HF pending但项目存储已验证”写成HF已完成，也不把它误当必须阻断一切。

Issue #11先canonical，PR #77只pointer，不merge。旧cleanup关闭事实不改写，不重新扩大删除。必要时只补已有failure inventory解释，不进行全盘寻宝。

## 9. 直接给PC agent的启动正文

执行PI0243_STORAGE_ADMISSION_AND_CONTINUE_GO。完整读取本文immutable commit、清理closure5758674618、原分级goal5755883470和最新状态。复用991911b的证据与0debbc save修复，不重跑旧科学诊断、不继续盲删旧模型。

先做三根路径storage binding和真实容量表；不把37.2GiB回收当/hpctmp已可写。修成熟PBS/runner的mkdir-only预检和exit-only归档；完成无模型CPU save→完整发布→durable校验→resolver/read/恢复验证。

优先用已证明compute可写的持久/共享工作空间，不能未经验证硬设Project_Storage。HF uploader放PC/Corelab或合法传输host，查真实凭据来源并做脱敏写读测试。保持训练offline。Project_Storage等durable已可靠且容量足够时，HF401标pending不阻塞A；不删除唯一持久源。容量/传输不成立就修工程或报告确切外部资源缺口，不反复qsub赌节点。

核对保留的checkpoint1000所属job/family和完整Trainer状态。满足原A同身份与exact-resume条件才恢复；否则保留为NOT_RESUMABLE，并在无其他有效恢复点时从原PI0226 initializer/seed/schedule重新跑原A5000步。不要把partial权重或旧PI0241作为A续训点。

准备准入全绿后明确允许提交一个新的Stage A job；626184不再等待、不原样重提。只保存1000/2500/5000，每份完成即校验并持久化，HF复制可异步于持久存储但所有状态如实记录。当前三个checkpoint消费者未经完整测试不得滚动删除；优先retain-all布局，容量不足才用已验收resolver/restore的rolling方案。

工程自主修复不限次数；一stage一active执行，禁止为成绩重试、改recipe/阈值/ABI/WBC。A通过后继续原PI0243条件路线；科学FAIL才停PI，真实权限/容量无合法解决路径则交精确基础设施blocker。小代码/tests/receipts推execution分支，Issue11 canonical、PR77 pointer，不merge、不新建monitor。

## 外部技术依据（不替代项目事实/固定runtime）

HF官方环境变量文档说明HF_HOME/token定位、HF_TOKEN优先级及offline变量：
https://huggingface.co/docs/huggingface_hub/main/package_reference/environment_variables

HF官方token文档区分read/write/fine-grained权限：
https://huggingface.co/docs/hub/security-tokens

这些说明用于定位认证，不能由此推断本次401的具体根因；使用已安装SDK支持的接口，不为修归档升级训练依赖。本文没有引用外部资料来假定NUS各目录的真实容量或挂载政策。
