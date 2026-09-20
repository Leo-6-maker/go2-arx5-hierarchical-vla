# PI0240 — 本地开发、Corelab 优先、Hopper H200 按需放大的自主执行 Goal

日期：2026-09-20（Asia/Singapore）  
决定：`LOCAL_TEST_CORELAB_FIRST_H200_SCALE_OUT_GO`  
发布状态：`PLAN_PUBLISHED_EXECUTION_NOT_STARTED`  
代码导航基线：`af5ad2815085e65f190d81a826de396c8b95c7b9`。  
Canonical：Issue #11；PR #77 只作 pointer，不 merge。

## 0. 授权、证据和对旧规则的覆盖

本计划落实用户的最新资源指令：本地 PC 作为 agent/SSH 控制端；小任务在 Corelab 4080s 跑通；只有大规模正式计算或 Corelab 单卡显存/资源实际不足时才上 NUS Hopper H200。不再把 H200 计算节点当成反复发现 Python、路径、FFmpeg、manifest 或测试框架错误的地方。

沿用 PI0239：工程定位、修复、测试、必要恢复不设固定次数或累计工程 walltime 上限；同一科学阶段不得并发重复提交，first-valid-result freeze，不因成绩不好重试；真实科学失败停止交 PI。此资源计划不自动授权新的训练 recipe。

本文件明确取代旧 PI0238/R1/R2 的工程次数限制，以及将 G3 或小型 G4 默认绑定在 Hopper/PBS/H200 的资源安排。原科学 cohort、模型 family、seed、训练/动作/评价契约、已有结果全部不变。允许在已授权 Corelab 上建立只读、内容可核验的输入副本和新 host binding；不是修改 sealed dataset 或 model bytes。

必读：
- PI0238-R2 结果：Issue #11 [5748879506](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5748879506)。
- PI0239 canonical：[5748982568](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5748982568)。完整协议位于 commit `451bbd231b60f7534bb7c08c654f81d7bea322ee` 的 `docs/protocols/PI0239_AUTONOMOUS_ENGINEERING_UNTIL_SCIENCE_FAIL.md`。
- 实际代码与路径来源：`scripts/pi0238_audit_to_sim.py`、当前 PI0239 执行 wrapper、`configs/pi0239_resolved_launch.json`；不要依据旧 PR head 或摘要猜路径。
- FFmpeg6 补丁 `a30905e03d4009abc419cb7b60b39da3032cfef8`；最新可读分支 head `af5ad2815085e65f190d81a826de396c8b95c7b9` 包含 clock-control 复用安排。

已由 GitHub 支持的状态：G0/G1/G2 已产生证据；全量专家 census 为 10880 windows（324 ACCEPT、10396 WARN、160 REJECT）；G3 曾受 FFmpeg8/torchcodec 阻塞，修复切向既有 Stage45B FFmpeg6；旧 PI0236 仍为 NO_ADMISSIBLE_CHECKPOINT，selected checkpoint=NONE。

用户最新报告：624609 的嵌套 metric comparator 已修复，12 个 same-process clock cases 通过；624616 正在 auto→small 等待 ngpus。GPT 本次未登录 Corelab/Hopper、未查询 live qstat、未重跑回归或实测显存。因此 job 状态、clock artifact 原值、SSH 可达性、硬件容量由执行 agent 接手时核验，不能将用户报告写成 GPT 实测。

## 1. 固定资源分工

| 层级 | 机器/访问 | 应做 | 不应做 |
|---|---|---|---|
| L0 开发与控制 | 本地 Windows PC，既有 SSH | git、代码修改、CPU pytest、manifest/schema/比较器/路径/状态机单测；调度和证据索引 | 假定 Windows PASS 证明 Linux 动态库兼容；保存明文凭据到 Git |
| L1 集成与小规模计算 | Corelab 已授权 CPU + 4080s | Linux 真实 loader/video/processor/collator/decode；G3；显存允许的 G4；小规模正式评估；获准后的 native sim/WBC；未来获批 recipe 的训练 smoke/pilot | 为迁就显存擅改分辨率/H40/精度/模型/训练语义；把 mock 当科学结果 |
| L2 接入与准备 | 本地 PC 经既有 `ssh nus-hopper` 到 Hopper 登录入口 | 站点允许的轻量 shell/import/路径检查、文件传输、配额查询、qstat/qsub、结果回收 | 模型加载/训练/sim、大量视频解码、全量数据扫描等重负载；绕过调度器 SSH 计算节点 |
| L3 正式放大 | Hopper H200，经站点批准 PBS | 已在低成本层验证的显存不足任务，或获批大规模正式训练/评估 | 通用工程代码迭代、pytest、G3 CPU debug；以反复 qsub 代替本地集成测试 |

任何层都遵守当地使用规则。Corelab 如有自己的 scheduler/预约机制，则使用它，不假设可无视共享资源政策。Hopper 登录节点是否允许某个轻量检查以已知站点规则为准；不确定或负载较大时在 Corelab 做，不擅自升级权限。

### 1.1 实测硬件，不把主机内存或多卡总量当单卡显存

先在允许的 Corelab 主机记录：

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,memory.used,driver_version --format=csv
```

同步记录 CPU、可用 RAM、磁盘 bytes/inodes、GPU 现有进程、选用 GPU UUID 和 site 预约。不得终止别人的进程。

用户提到的 32GB 是待核验的资源描述，不是程序常量。NVIDIA 标准 RTX 4080 SUPER 公开规格为 16GB；Corelab 是否为扩容设备、其他型号或多卡，必须以单卡实际配置和 nvidia-smi 为准。两卡显存不当作单进程自动可用的相加值。若实测该设备单卡为32GB，按该容量规划。

## 2. Goal A — 接管当前状态，解除 CPU 诊断对 GPU 排队的依赖

本地 agent 读取已有 SSH config/项目 handoff 中的 alias 和路径，复用 `nus-hopper` 及实际 Corelab alias；不发明主机、端口、用户名，不发布私钥/token。优先本地分别连接两端；若两远端不能直连，使用允许的本地中转传输，不要求新开放端口。

先查 624616 的真实状态、拥有者、Job_Name、脚本 hash、阶段和现有输出：
- 仍为 Q 且只做 G3 CPU 工作：保存 qstat/脚本身份，将 G3 转移 Corelab 后取消这一个冗余 job；先确认自己的 job，绝不批量 qdel。取消前不启动相同科学阶段的并行副本。
- 已为 R 且在执行正确冻结版本：不原地改 staged 文件，不同时重算相同阶段；让该次有用工作完成并复用结果，后续迁移 Corelab。不能因为新策略删除正在形成的证据。
- 已终止：读取实际结果并恢复下一未完成阶段；不要按旧聊天继续等队列。

资源盘点、本地单元测试、数据传输等不重复科学结果的准备，可与既有作业并行。不得自动新建定时 monitor；agent 会话中的查询和任务日志不是承诺 GPT 在后台工作。

验收：`resource_inventory.json`、`active_execution_state.json` 与机器分工表。完成即继续，不需要用户再次确认“可否用 Corelab”。

## 3. Goal B — 最小跨机镜像和可重定位 launch，原证据不变

### 3.1 只搬必要内容

先优先复用 Corelab 已有 code/model/data/runtime/cache 的内容相同副本，避免重复下载或拷贝。缺失时按允许清单做增量传输：
- G0–G2 已封存 JSON/CSV/inventory、cohort、clock-control artifact；
- G3 固定128cohort与原12reject/2excess去重并集（最多142个已授权窗口）的必要 Parquet、两路原视频、元数据、TRAIN stats/relative stats、modality、processor 配置与源代码；
- 只有 G4 需要时再传 baseline-1000、623593 fresh-10000 的必要完整模型 shards/config/processor 文件；不为 inference 搬全部 optimizer payload；原完整 Trainer checkpoints 留在 Hopper，不删不重seal；
- 若之后完整五profile评价获准，再补齐其余 checkpoints 和完整1088 CALIB，不提前无目的复制全部 archive。

稀疏诊断 bundle 不是完整 TRAIN/CALIB replica，不得声称其 tree hash 等于原全集。保持原 episode/source IDs、顺序、window/视频字节；原 loader 若依赖完整 metadata/索引，使用受测的只读定位映射或按清单复制完整所需数据；不得重新编号、重新编码视频、重新生成 stats、改变 sampler 来适应子集。

### 3.2 以内容证明迁移，不靠路径看起来相似

保存 source URI/path、destination path、relative file path、bytes、SHA256、原 seal/inventory 来源和 scope，目标端核验。原含绝对路径的 seal 不改写；另存 `host_bindings.json` 和 `replica_manifest.json`。新 binding 必须逐文件绑定原 authority，不能把根路径检查全部关掉。

迁移只改变 locators，helper 的参数传入 host binding；预检和运行共用同一个 resolver。不得新造一套数据、训练或 WBC 框架。保留原 scientific result 与 backend identity；新 Corelab 结果不能冒称原 Hopper tensor 的逐bit重放。

### 3.3 存储检查针对实际输出主机

Corelab 当前阶段只读消费镜像并写新 output/cache/trace；检查 Corelab 实际写入预算和可用容量。Hopper quota 未知不是 Corelab G3/G4 的全局 blocker，前提是运行时不会偷偷在 Hopper scratch 写 cache/结果。输入源的可读性和完整性仍需验证。

Hopper 正式计算前才验证其实际目标 fileset 的容量：模型已驻留，计算新增 checkpoints、原子临时文件、日志、cache 和 archive staging 峰值，不重复复制大模型。使用实际 quota；若无 quota 工具，沿 PI0239 做有界容量证据/已授权新输出区迁移，但不能把释放后的 reservation 当作长期保证。保留实际分配或分批释放/写入并持续监测，明确 concurrent writer/TOCTOU 风险。禁止删正式权重、sealed data 和 canonical evidence。

验收：副本身份PASS、host binding可重定位、输出本地化、存储证据明确。新小量manifest允许写Git，大tensor/权重留已授权存储。

## 4. Goal C — Local pytest → Corelab真实CPU集成 → G3完成

### C1 本地快速循环

工程错误先在本地 CPU 测试复现并修复：manifest/checkpoint/inventory解析、nested numeric comparator、标准库路径、cwd、真实dispatch、schema、missing/nonfinite、路径越界、PBS自任务识别、phase复用与准入分类。使用真实函数和与旧故障对应的 positive/negative fixtures；compile只是语法检查，不是功能验收。

检查现有 `test_pi0238*`/`test_pi0239*` 和适用的 repository instructions，调用已发现的 CPU test 子集；不要无差别执行会启动GPU或sim的全仓库测试。mock与fixture只用于软件验证，不进入科学scorecard。

### C2 Corelab Linux实际兼容集成

使用既有独立 GR00T Python 环境，保留 pinned GR00T source、torch/torchcodec、modality、processor 与 sampler配置。复用 sealed Stage45B FFmpeg6 库在兼容 OS/ABI 下的已校验组件；不直接把 Hopper venv 目录当可随意跨机复制的便携环境，不安装随机最新版替代。

检查实际动态库加载路径，而不只看 ffmpeg CLI version。沿原正式 loader读取预定固定样本的两路视频→真实 processor→真实 collator→实际target tensor与mask→独立eval decoder→SafetyGate→artifact write/read。此CPU环节不需要GR00T/VLM权重，不能因为import构造副作用加载模型。

Corelab GPU0可承载后续小模型推理，但G3本身主要是CPU工作。先用固定cohort的第一个样本验证完整IO，再做原≤142窗口G3，保持样本集合和原时序/mask语义，不用大batch堆积全部图像造成RAM压力。

### C3 复用已经完成的工作

G0/G1/G2 原身份有效则复用，不为迁移主机重复10,880专家census或5440历史模型调用。Clock12例的已修正同进程回归，读取status、输入/实现hash和实际值后复用；不能仅因 `test -s file` 就写“已PASS”。受代码bug影响的子诊断另写revision，不覆盖历史。

G3必须证明真实训练目标的[B,40,132]字段、active slots/mask、state anchor、train/eval有效配置、XYZ/SO(3)/gripper物理parity和normalized excess解释。数据/processor真正需变才能通过属于science fail；FFmpeg/import/路径/API/比较器错误继续在Corelab自主修。

验收：`LOCAL_CPU_TESTS_PASS`、`CORELAB_REAL_CPU_INTEGRATION_PASS`、真实G3结果或可审核科学失败。不要交付一个只有import PASS的“成熟”结论。

## 5. Goal D — 小规模真实GPU诊断默认在Corelab完成

G3科学准入通过后，不默认申请H200；先盘点模型/输入显存需求，在Corelab执行固定G4。

G4沿PI0239：baseline1000、623593 fresh10000 × 64TRAIN+64CALIB，共256个唯一科学cells；原seed=22301、language、dtype、sampler、H40/StateV1/frame/TCP/dt不变。first-valid-result freeze，不因gate REJECT或高误差换样、换seed、换设备挑成绩。工程失败的未完成cell可恢复并完整记账，取消工程次数限制不扩大科学集合。

两模型顺序加载，默认复用既有 batch=1 query路径；不要同时常驻baseline和fresh，不开启optimizer或梯度，不把大量训练state加载到显存。记录实际单卡峰值allocated/reserved、模型load峰值、CPU RAM、任务latency、host/GPU UUID。

第一条按原顺序的固定G4 cell兼作真实推理集成/显存测试，其结果如完整合法就直接纳入G4，不能再跑一遍挑结果。只有GPU程序加载或推理发生实际capacity错误、尚无有效结果，才恢复该cell。始终按固定模型/样本计划完成，其科学负结果本身不归为工程错误。

**一个smoke能装下且同契约全cohort运行可行，就在Corelab完成整组G4，不为了“正式”二字搬H200。** 最好两模型同平台形成对照；确需中途跨平台时标注backend分层和解释限制，不能把硬件与训练效应混为一谈，不替换已合法结果。

输出raw [1,40,132]、decoded [1,40,10]、原输入hash、expert/mask、gate、逐点XYZ/SO(3)/gripper与速度、TRAIN/CALIB分层、source-rollout统计。目标距离不替代拟合，reference不称achieved TCP。G4若只有汇总差异且尚不足以认定根因，保留不确定性并形成PI决策包，不能无限增加科学样本把结论试出来。

## 6. Goal E — H200升级门：容量或规模证明，而非调试方便

升级H200的合法理由：

1. **单卡显存不足**：在目标完整模型、原dtype/resolution/H40/sampler、单模型/无梯度/无泄漏条件下，实测峰值或OOM以及静态模型容量证明无法在Corelab完成。明显静态容量已不可能时，不要求故意反复OOM。
2. **已获批准的大规模训练/计算**：Corelab小规模真实执行已验证功能；训练的forward/backward/optimizer/save/reload只在对应recipe授权后验收。固定大任务工作量与实测throughput说明H200能合理缩短正式计算，预计开销与quota均已准备。

无效理由：代码还没debug好、G3 CPU test排不上、数据在Hopper、pytest尚未过、FFmpeg还没绑定、不想传小bundle、想改变结果或需要“正式”标签。

不能为避免升级私自量化、降低图像分辨率、缩短H40、改dtype/采样步数、裁剪模型、改变effective batch/LR/optimizer或替换attention实现。相同代码允许清理自身未使用缓存、顺序加载、改IO/logging；涉及科学数值/训练配置的memory优化需要原recipe已批准或回PI。

`h200_admission.json`最低包含：计算阶段/科学授权、低成本层测试与真实sample报告、固定code/input/model/processor/FFmpeg依赖身份、已测显存/规模理由、确定命令与配置、输出fileset容量、剩余cells或训练步数、已有结果freeze、没有同阶段活跃重复任务。

Hopper通过本地`ssh nus-hopper`接入既有合法PBS；登录入口只做站点允许的轻量环境/路径/库核验、传输与调度，不把CPU重活转到login node。H200第一次正式作业内，第一条既定cell/已授权训练smoke用于硬件集成核验，完成后直接继续，非新增seed或科学样本。真正H200特有kernel/runtime问题可以最小定位并自修，但通用逻辑回本地/Corelab回归，不循环提交单用途debug作业。

不设工程qsub次数总上限；只允许与已验证任务匹配的站点资源，不投机并发副本、任意加卡或无依据增大资源。每次恢复必须针对新证据且保留上次输出；平台权限/节点不可用且无合法替代才报EXTERNAL_INFRA_BLOCKER。

## 7. Goal F — 从模型诊断到冻结WBC仿真闭环

沿用PI0239的条件科学授权，不因为本次更换资源层而退回每个PASS都等PI：

### F1 归因和必要的完整选择

G4若确认真实TRAIN拟合失败、稳定TRAIN→CALIB问题、真实标签/模型契约冲突，停止SCIENCE_FAIL，提交证据和下一recipe建议，不启动新训练。若只发现实现偏离已批准契约，最小修复并核验后按新执行身份完整评价baseline+4 fresh，各1088 rows，unchanged selector选唯一checkpoint或返回NO_ADMISSIBLE_CHECKPOINT。

没有任何相关实现修复时，不能重跑完全相同的旧5440 cells碰运气；旧NO_ADMISSIBLE_CHECKPOINT持续有效。完整五profile选择的各候选采用同一冻结backend/runtime，不把不同平台的最佳profile拼成selection。该评价若Corelab容量/吞吐可行就在Corelab，容量/规模门成立才H200。

### F2 Corelab的one-chunk与receding

有唯一admissible checkpoint且没有science fail时，按PI0239进入同一权重/processor/runtime的TRAIN A–D C0 one-chunk，再在全部预注册任务通过后执行progress-preserving receding。冻结WBC、原0.035m scorer、target边界和原已批准cadence/tick cap/termination保持。未获批准的参数不得临时填值。

复用当前dense-relative/PI0226已验证的native执行入口；旧Stage40代码可参考独立model-worker/native-worker的环境分离，但其历史StructuredAction路径不能代替当前dense-relative ABI。

Corelab上native IsaacGym/WBC与GR00T保持既有独立环境，不将模型侧FFmpeg/LD_LIBRARY_PATH覆盖native的CUDA/Python依赖。单卡能同时容纳则用已经验证的进程间观测/reference交接；若两进程合计显存不足，先确认是共驻问题，不把各自单独能跑误称组合也能跑。

不能为了省显存改WBC算法、reset仿真、缩短轨迹、用旧观测或反复reload模型改变未批准的时序。共驻确实不足可提出已批准的分离执行机制；若将模型置于H200、sim置于Corelab，必须是已支持的暂停仿真/跨进程协议，绑定episode/query ID/观测时间/timeout/禁止stale结果，不能声称实时控制。若实现不支持同契约分离，保留资源事实回PI，而不偷偷改cadence。

必须保存：reference vs achieved world-TCP、每cell结果、gate、native termination、execute→new observe→infer→gate→reference update→execute因果trace。工程bootstrap/环境/IO错误继续修；正确执行后的任务失败即SCIENCE_FAIL，不换模型/seed/阈值追PASS。

### F3 DEV与最终完成定义

PI0239已条件授权的DEV-E只在TRAIN闭环成功且checkpoint/runtime完全freeze后一次执行；不能据此调参再用同DEV。Final-F与硬件仍锁定，不能借资源计划解除；PI0239文件中关于final后续路线不视为本次自动开放。

当前成功终点：有证据的shared reaching policy通过冻结TRAIN闭环及条件DEV，提交完整review package；这不是不同技能的general model。不存在admissible模型则不绕过F1进入sim。

## 8. 新一轮大规模训练的资源路线（只有科学recipe批准后才执行）

现有10k训练已完成，本资源指令不是再跑一次10k的科学授权。若诊断证明需要新训练，停止交PI；提交一份明确initializer/data/loss/LR/optimizer/scheduler/batch/mask/trainable modules、样本与评价规则的run card。

获批后由同一PC agent连续执行：

`本地训练IO/保存恢复tests → Corelab原recipe少量真实forward/backward/optimizer → 保存/恢复/解码检验 → 获批small pilot和早期TRAIN free-running检查 → H200规模训练（若符合升级门）→ full checkpoints→统一selection→Corelab frozen-WBC one-chunk/receding`。

smoke的权重和optimizer是独立产物，正式run从批准initializer/seed重新建立或按已批准exact resume恢复；不把调试updates混入正式family。不能为Corelab拟合不同模型再在H200训练另一模型却称同recipe。新recipe的预算、成功/停止条件和checkpoint family在结果前冻结；工程恢复不设次数上限，科学失败不继续加steps。

H200只是显存/吞吐资源，不是默认GPU选择，也不是模型科学问题的解法。

## 9. 执行纪律、成果和状态

同一阶段一个owner，Corelab与Hopper不同时重做同一批科学cells；不同阶段的纯准备工作可并行。first valid scientific result固定，所有engineering attempts/cancellations/资源/修复hash单独记录，不把“GPU模型调用=0”写成“GPU资源申请=0”。

只在新分支/worktree及新output写文件；不reset用户脏工作区，不force-push/merge，不删除正式数据权重。工程副本/自身cache可在证据封存后受控清理，其他用户进程和文件不碰。

任务恢复索引要包括：current_goal、host/SSH alias（不含秘密）、actual device/memory、code/runtime/model/data identities、active process/job、completed phase/artifacts、remaining scientific cells、科学与工程计数、下一动作。运行中持续落盘日志和manifest；`-s`/文件存在不能代替artifact status/hash核验。

报告三个维度：工程链是否成功、科学结果是否通过、下一任务在哪台机器。合理终态为RUNNING/READY_NEXT_STAGE、SCIENCE_FAIL_*、COMPLETED_REVIEW_PACKAGE或真实EXTERNAL_INFRA_BLOCKER。禁止ENGINEERING_BUDGET_EXHAUSTED；也禁止因没有确定新recipe而偷偷开训。

小代码、测试、协议、manifest和报告推GitHub；Issue #11先canonical，PR #77只pointer。复用而非复制巨量旧结果；不要每修一处创建新的科学协议。

## 10. PC agent启动正文

现在执行PI0240：本地PC负责开发与SSH控制，Corelab负责Linux集成、CPU G3及能装下的小GPU诊断/sim；NUS Hopper H200仅在已成熟代码的大规模任务或实际单卡显存不足时使用。完整读本文件immutable commit与PI0239 canonical 5748982568及最新执行状态。

先查624616真实状态：仍Q且只做G3时保存证据并取消该冗余job，转Corelab；已经R则不改staged、不重复科学执行，复用其有效结果。不要继续把G3 CPU运行与auto→small的GPU队列绑定。

通过已有SSH配置连接Corelab和nus-hopper，不发明主机或凭据。Corelab运行nvidia-smi，记录实际单卡UUID/总显存/空闲显存、RAM和磁盘；不要把32GB当未经核验的显存常量。

复用G0–G2和已通过的12-case clock-control，status/input/code/hash有效即不重跑。按清单将G3所需窗口、metadata/TRAIN stats、modality/processor和必要环境组件同步Corelab，逐文件hash核验；用host binding保留原seal，不改内容或source IDs。

执行本地CPU tests→Corelab真实视频decode/processor/collator/decode/write-read→完整固定G3。沿sealed Stage45B FFmpeg6检查实际loaded libs；复用成熟pi0238 runner，wrapper只做host/launch/admission/report。工程问题自主最小修复、回归、恢复，不限制工程次数，不在H200上反复pytest和debug路径。

G3科学准入后在Corelab尝试固定G4，baseline1000与623593 fresh10000顺序加载，固定64TRAIN+64CALIB、256 unique cells、原seed/dtype/processor/H40。第一条预定cell兼作显存/真实推理smoke，合法结果直接封存，不重复挑结果。装得下就完整跑Corelab；实测原配置单卡装不下，或获批正式大规模计算有规模证明时，封存h200_admission后才经本地SSH/PBS上H200。不得靠量化/改分辨率/H40/模型/recipe适配显存。

Corelab输出用Corelab容量证据，Hopper quota未知不阻断不会在Hopper写入的工作。真正Hopper job仍验证其输出容量。Hopper login只做站点允许轻量检查/传输/调度，重CPU工作放Corelab，不绕过scheduler。

科学有效但失败则停止SCIENCE_FAIL并交PI；不把模型差当工程错误重试，不自主新增训练。若只有原契约实现偏差且修复后有依据继续，则沿PI0239完整五profile评价→unchanged selector唯一freeze→Corelab frozen-WBC A–D C0 one-chunk→progress-preserving receding→条件单次DEV-E，阶段通过自主继续。原Final-F/硬件/PR merge继续锁定。新训练只在PI recipe批准后按Corelab smoke/pilot→必要时H200规模放大执行。

没有固定工程重试预算，但同一科学阶段只能一个active执行；已有合法结果first-valid freeze、阶段恢复不从头重做。外部权限/设施确无合法路径才EXTERNAL_INFRA_BLOCKER。持续记账，Issue #11 canonical、PR #77 pointer；不要只回复已读、不要重建monitor。

## 外部事实参考（不替代项目pinned runtime）

- NVIDIA RTX4080家族规格：https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4080-family/ 。标准4080 SUPER为16GB；实际Corelab硬件仍需测量。
- PyTorch reproducibility说明：https://docs.pytorch.org/docs/stable/notes/randomness.html 。不同平台/版本不能默认逐bit相同；本计划不升级项目runtime，新host身份必须明示，不能用该事实豁免真实语义不一致。
