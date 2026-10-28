# PI0238-R2 — 共享 manifest 解析、功能预检与有界 CPU 恢复

日期：2026-09-20（Asia/Singapore）  
决定：`MANIFEST_FIRST_ENGINEERING_CONTINUATION_GO`  
发布状态：`AUTHORIZATION_PUBLISHED_EXECUTION_NOT_STARTED`  
审核代码：`e4817bbae47919687b3748eea5e536b41e04c972`。  
Canonical scientific ledger：Issue #11；PR #77 仅 pointer，不 merge。

本文件是 PI0238 / PI0238-R1 的工程恢复补充。它不追加训练预算、不修改旧实验结果、不重新定义 SafetyGate，也不代表文档发布时已经运行作业。CPU 工程恢复按本文件生效；未明确改动的科学边界继续遵守原 PI0238 与 R1。

## 1. 必读依据与已接受的终态

- PI0238 原协议 commit `044e4d5ef8b4c1effb8236190c0eb083fa49979b`，`docs/protocols/PI0238_AUDIT_TO_LARGE_SIM_CLOSED_LOOP_GOALS_V1.md`；Issue #11 comment `5747200805`。
- PI0238-R1 commit `b940fc9c1186ce01e14c13638f251f1e5e051b76`，`docs/protocols/PI0238_R1_ENGINEERING_RECOVERY_AND_CONTINUATION.md`；canonical `5747515056`。
- 最新 closure [5747853063](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5747853063)、PR pointer [5747853482](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/77#issuecomment-5747853482)。
- `e4817bb` 的 diff 与固定版本 `scripts/pi0238_audit_to_sim.py`、`scripts/pi0238_r1_engineering_recovery.py`；委托修改 `b08e2c3`，runner SHA 封存 `d08d0c9`。实际执行仍须核验 staged 文件和调用路径。

最新账本记录 R1 三个作业：624262（干净环境探针破坏 stdlib 搜索路径）、624278（排队取消）、624283（hopper-16/small/Exit 1，G0 误判 checkpoint family 不完整）。624283 未到 G1–G3，模型 load/forward/get_action、训练、WBC/sim、DEV/F 都为 0。正确分类为 `PRE_SCIENCE_ENGINEERING_BLOCKED / R1_SUBMISSION_BUDGET_EXHAUSTED`，不是新的模型或泛化结论。

S2 的实际路径是：

`/scratch/e1192548/pi0228_h200_large_adapt_v1_623593.hopper-m-02/training/checkpoint-{2500,5000,7500,10000}`

这是账本对已封存路径的说明，不是让执行者重写这个模板生成路径；权威定位仍为 `fresh_family.checkpoints[*].path`。旧 runner 漏掉 `training/`。`e4817bb` 已改读逐 checkpoint path 并校验根目录包含、step/name、candidate 与目录存在；同时修正 qstat 自任务 ID 截断比较。此修复方向接受。账本仅提供 exact-runtime compile-only 通过，没有证明修后整条 G0 已通过。

GPT 已读取以上 GitHub 评论、diff 与实际代码片段；没有登录 Hopper、运行测试、取得四个目录原始文件或复算远端 hash。目录存在、零模型活动和 runtime compile 结果依据 canonical 账本，不能把本次静态审核写成独立的远端验收。

R1 三个作业额度已经用完，不能直接使用旧授权提交第四次。本文提供新的、有限且附功能预检的 R2 预算。此前 624206/624209/624229 及 R1 624262/624278/624283 全部保留，不从累计日志删除。

## 2. 本轮唯一工程主线

复用 `scripts/pi0238_audit_to_sim.py` 的 G0–G3；R1 wrapper 仅承担启动、预检、准入与报告。不得再写一份独立 G0–G3 来代替成熟 runner，不在 wrapper 的未调用函数里修复后声称实际路径已修好。

本轮重点不是继续增加协议层或检查器，而是让预检与正式运行使用同一解析函数、同一配置、同一 schema：

`sealed manifest → shared resolver → resolved launch → functional preflight → actual G0 → sealed CPU diagnostics → evidence-based admission`

只做必要的函数提取、调用点统一、功能测试与最小 wrapper 修复。保留封存的 selector、provider、gate、modality、数据、权重和统计量；任何科学语义修改仍返回 PI。

## 3. 提交前必须通过的五组功能检查

### 3.1 Checkpoint 的唯一解析入口

将已修复的 sealed-checkpoint 解析提取为成熟 runner 内的一个纯函数，预检与实际 G0 都调用它。不要在预检验证 A 路径、G0 再重新构造 B 路径。wrapper 中未调用的旧 `fresh_root/training/checkpoint-*` 逻辑不能变成第二套 authority。

最低验证：

- 清单恰好四条，steps 恰好 2500/5000/7500/10000，无 duplicate step、无 duplicate resolved path；在转 dict 前检查，防止重复项被静默覆盖。
- `selection_candidate is True`，不要使用 `bool("false")` 一类宽松转换；若实际 schema 不是布尔类型，报告实际 schema，不靠自动 coercion 掩盖。
- 清单 path 必须符合其已声明 absolute/relative 语义；目前 canonical 是绝对路径。目录、真实根目录包含和 step/name 一致；拒绝根目录外路径及未授权 symlink，不能通过名称匹配当作完整身份。
- 报告四个原始 path、resolved path、step、sealed tree/index/shard inventory 来源。目录存在只叫 `PATH_RESOLUTION_PASS`；没有验证内容时，不能叫 full checkpoint integrity PASS。
- G4 前按原协议核验 baseline/fresh10000 的真实权重身份；CPU G1/G2 不加载模型。不得为了 CPU 标签分析复制/重建/重新 seal 任何 checkpoint。

测试必须实际调用共享函数：合法 nested `training/checkpoint-*`、路径打乱仍可正确映射、缺一份、重复 step、重复路径、错 step/name、错误/字符串 candidate、根目录外路径。旧坏实现应在合法 nested fixture 上失败，新实现通过。fixtures 只证明程序行为，不是生产数据的科学证据。

### 3.2 Inventory 的预检与 G0 必须共用相同基目录规则

静态发现：R1 `inventory_parse()` 按 `base_dir / relative_entry` 解析；成熟 G0 在 checkpoint 检查之后仍使用 `Path(raw_path)`，因而相对条目依赖进程 cwd。不能靠预检已通过推断实际 G0 一定能读取同一文件。实际远端条目若都是绝对路径可能不触发，但必须验证真实格式并消除这种不一致。

统一规则：相对条目以该 inventory 声明的 artifact base 为准，当前 PI0237 约定是 `p237_root/evidence`；绝对路径保留并经过既有 allowlist。清单为空、格式错、重复条目、越界、文件缺失或 hash 不符必须失败；禁止 `all([])` 得到成功。不得跳过不认识的行，也不扫描全盘寻找同名文件。

使用同一共享 parser，避免 wrapper→legacy 的循环 import。至少在两个不相同的 cwd（含空格目录）运行相同 fixture，得到相同路径和摘要；绝对路径同样测试。原路径/hash不符仍是 blocker，不准修改旧 inventory 的 expected values。

### 3.3 PBS 自任务识别与实际子进程环境

保留 `e4817bb` 的边界匹配方向，但实际测试完整 `624283.hopper-m-02` 和显示截断的同一任务，以及 `6242830.*`、另一个 job、header/空输出。只能排除确定匹配的当前作业；未知 qstat 格式或命令失败不等于“无活动作业”。若站点使用 array/server 特殊格式，按原始输出明确解析，不能以宽松前缀排除别的任务。

不取消其他作业。重复提交检查围绕本协议的实际活动 job/输出根和资源冲突；若出现未分类任务，先读取详情，不盲重提。

`modality_probe` 成功发生在子进程，并不会自动把 import 绑定带给下一次实际 G3 进程。实际 G3 在导入前也必须绑定同一个已验 hash 的 modality，并记录 `__file__`。保护 stdlib 和 site-packages；禁止清空 `sys.path` 只留下项目路径。子进程各自使用同一 resolver/启动配置，禁止凭父进程已 import 推断子进程也正确。

### 3.4 预检不得破坏正式启动的前提

现有 `preflight()` 会创建 cache/tmp/output probe，而开始时又要求这些目录不存在。把它在提交前和 PBS 内原样各调用一次会产生新的机械阻塞。

最小处理：分离无副作用 `verify-only` 与一次性的新 attempt 初始化；或为同一 launch 提供有严格 owner/launch/protocol/hash 绑定的 re-verify。只能复用本 attempt 自建且身份可核验的目录，未知/不同 attempt 目录仍拒绝。不得使用宽松 `exist_ok=True` 绕过证据隔离。

cache/output/tmp 的共享合法父目录不应因为子目录创建顺序互相冲突。预检输出与缓存写入均计入 R1 CPU 1GiB 上限。提交前轻量路径/小文件/清单/环境验证先行；全数据 tree/shard hash 和 dataset CPU 计算仅在合法资源中执行，不搬到登录节点。

用同一 `resolved_launch` 驱动预检和真实命令，记录实际 argv、cwd、解释器和运行模块 hash；不要把完整 R2 文件伪装成 R1 协议 hash。保留原 R1/PI0238 identities，在扩展字段显式绑定 R2 authority 和实际 implementation commit。

### 3.5 整条调用链与 schema 的功能冒烟

编译只检查语法。本次额外要求一项 endpoint smoke：在小型 synthetic fixture 中，经实际 wrapper dispatch 调用实际 mature G0（允许以 fixture/mock 隔离重型生产 hash 和站点查询），验证其确实使用共享 resolver并生成可读结果。不要只测试函数副本或 wrapper 中已经不使用的 G0。mock 结果标 `SYNTHETIC_CONTROL`，不进入科学证据。

在真实远端环境做轻量 resolved-input 检查，明确四个 checkpoint 路径、inventory 所有条目、modality 实际原点、脚本/hash 与本 attempt 的 cwd。记录为功能预检，不冒称完整 G0。

用 fixtures 验证 G1/G2/G3 的实际输出可以被 G4 admission/G5 读取；`PASS_READ_ONLY` 只说明读取/测量完成，不直接授予 parity PASS。缺字段或 evidence 未覆盖为 `NOT_ESTABLISHED`，而不是自行填 PASS。

当前 mature G3 已增加 mask 必需、active slot 观测、SO(3) 物理误差和 numerical 状态，相比旧版有进展；但仍需按原协议核验真实训练配置如何用于实例、字段/mask 语义和数值解释。源码中的 `float64_cast_control_max_abs=None` 不代表已完成 float64 对照。不得为了让本次 CPU 脚本通过而更改动作语义或放宽数值条件。

## 4. 避免永久不准入和假科学分类

静态发现 R1 `quota_snapshot()` 将 `admission_capacity_pass` 固定为 False；这是当前 fail-closed 的行为，不是已实现实际 quota 验证。G4 读取预检文件中的该值；即使以后取得真实 quota 输出，也不能只修改该布尔值。

必须有独立、带原始报告/hash/单位/被测fileset与时间的容量验证结果，经同一准入函数计算可用 byte/inode、预计峰值和余量。quota 工具不可用时继续 UNOBSERVED；不能依据文本出现数字/limit字样、df 或小写入探针得出容量 PASS。G4 准入时重新读取足够新的容量证据，不能永久依赖启动时的旧快照。

G5 当前 `classification = BLOCKED_ENGINEERING_OR_RESOURCE if blockers else SCIENCE_BLOCKER` 会把各种 blocker 一概归工程，同时把无 blocker 归科学失败。修正为依据有类型的证据分类：工程/资源、真实 parity/numeric 科学问题、尚缺实现/证据、或诊断已就绪但未执行。阶段运行失败不能被误报为模型科学失败；正常专家/模型 REJECT 也不能被重试消除。

`g4()` 在当前 wrapper 中是 admission 检查，不是实际 `get_action` evaluator。允许在原 PI0238 的范围内完善最小有界 evaluator及其测试，不能把 admission JSON 当作完成256次推理。尚未实现则如实 `INFERENCE_RUNNER_NOT_IMPLEMENTED`，保留已完成 CPU 归因。

## 5. 新预算：R2 只补 CPU 工程恢复，不重置科学计数

### 5.1 新增最多2个审计作业ID

从本授权起最多 **2个新增、串行 G0–G3 审计 job ID**（一次主作业，最多一次满足下列条件的工程替代），每个申请 walltime≤2h，合计≤4节点小时。取消后的 ID 也计数。qsub 结果不明先查询，不盲重提。旧 PI0238/R1 六个已知 ID 单独列出，不能改名重置历史计数。

资源允许已知 `auto→small` 标准节点，项目 `CFP03-CF-004`，不超过1 GPU/12CPU/225GB；CPU-only计算，CUDA禁用，零 GR00T/VLM 权重实例化/load、零模型 forward/get_action。合法既有 CPU-only 资源可用，但不为寻找新队列另提试验作业，不因平台按已授权标准配 GPU 而取消。记录申请的 GPU 节点资源与实际零模型调用的区别。

本轮不创造第三个 qsub 用于 preflight。轻量无作业预检在站点允许的环境完成；重型 CPU 核验纳入主作业。若站点不允许预检的某项操作，将其作为主作业起始步骤，失败也如实计入预算，不假装已经测过。

只对已定位的路径/import/日志/解析/资源启动错误允许一次替代：失败证据封存、最小补丁、旧失败fixture复现并在新代码通过、全部入口预检通过后再提交。科学语义不一致、真实数据 hash 漂移、越权读写、模型/采样预算超限不属于工程替代条件。

### 5.2 CPU存储与阶段恢复沿用R1

quota未知可在原R1的 `CPU_STORAGE_LIMITED_DIAGNOSTIC_ONLY` 条件下继续：专属≤16MiB fsync写入探针、df观测余量至少max(2GiB,3×预测新增字节)、正inode、CPU新增输出/缓存总量≤1GiB、运行时上限检查；这不是quota证明。只删除自建探针，不删输入、权重或历史产物。

G0实际身份闭合后先封存证据，再连续 G1→G2→G3。某一阶段的工程故障不抹掉此前的完整结果；恢复时按输入/代码/协议hash验证并引用已有phase，算法改变则另写revision。不要每次修后都无条件全树重算、重做已完整census。

每phase独立起止日志、输出inventory与failure；不依赖全部命令结束才有报告。G0失败时G1–G3不得冒进；G3科学检查未通过时保留G1/G2结果并交G5，不需要把所有结果等到模型推理后才报告。原 expert/data/codec 结论的缺口不能用“成熟”命名抹去。

### 5.3 G4保持原有条件授权，零新增科学预算

PI0238/R1已经授权、但尚未准入的 G4 条件继续有效。必须同时满足实际数据/权重/模型输入/132mapping与mask/数值解释/存储/预算独立准入，才可使用既有H200额度。CPU代码的任何PASS不自动解除CUDA禁用。

固定64 TRAIN+64 CALIB、baseline1000及623593-family fresh10000、合计至多256 attempted policy queries，所有PI0238/R1/R2合计计数不重置。首条预算内canary、原seed/dtype/processor/sampler、专家REJECT保留分层、新同契约测量不冒充旧tensor等规则不变。

G4仍为1个H200诊断job（1GPU/12CPU/225GB/walltime≤1h）；只有原规则定义的首次零query工程失败可替代一次，总G4≤2 IDs/≤2h申请。此额度独立于R2 CPU额度且不是新增一套。quota未知或parity未闭合时保持NOT_ADMITTED，交付CPU诊断与精确blocker，不用R2 CPU额度试跑模型。

监控已关闭的状态不改变；不自动重建定时monitor。作业正在运行时按现有执行环境能力查询状态；窗口中断则交接真实job ID/状态，不声称后台持续工作。

## 6. 交付和硬停止门

本轮完成定义不是 compile PASS，也不是“修好等下一次授权”，而是：

1. 一个最小修复 commit 和功能回归表，实际命中的源码函数与被修复旧错误可追溯。
2. 同一 resolved launch 的功能预检 + 实际G0身份结果，或明确哪个真实输入未满足条件。
3. 在G0有效基础上的G1/G2/G3全部可完成CPU结果；有证据的parity/numeric状态，不填充无测量PASS。
4. G4完成证据，或独立可验证的未准入原因（quota、真实语义、数值未解、实现未闭合分别列），不得只有泛化的blocked字符串。
5. G5给出已确认事实、仅假设、缺失项、主线下一实验/修复run card；证据不足则注明尚无法决定recipe，不空填科学结论。

工程或资源未闭合：`BLOCKED_ENGINEERING_OR_RESOURCE`；新CPU额度耗尽：`R2_ENGINEERING_BUDGET_EXHAUSTED`；真实科学归因或完整诊断交付：`STOP_FOR_PI_RECIPE_DECISION`。可兼容引用旧停止token，但reason必须准确。全程 `selected_checkpoint=NONE`。

保持PI0236 `NO_ADMISSIBLE_CHECKPOINT`、PI0237负对照和R1失败历史不变。不允许：训练/backward/optimizer/scheduler、DAgger/记忆训练、WBC/sim/receding/hardware、DEV-E/F/Final-F内容读取、修改selector/gate/threshold/action representation/frame/dt/StateV1/H40/target输入/loss/LR/容量、平滑/重定时/模型替代、重新seal/删除历史权重、PR merge。原大规模训练与sim长程路线依然需要其对应科学GO。

在新输出根写入，不改用户脏工作区；execution修复可在现有工作分支延续但不得改已排队/已运行attempt的staged文件。小代码/测试/manifest/报告推送GitHub；Issue #11先写canonical closure，PR #77仅pointer。权重和大数据不推Git。必须报告所有旧新job IDs含取消、实际资源、每phase状态、attempted/completed/reused计数、禁区计数及证据路径/hash。

## 7. PC启动正文

现在执行 PI0238-R2，不再等待是否恢复CPU额度的确认。完整读取本文件的固定publication commit、R1协议及Issue #11最新canonical。以 `e4817bbae47919687b3748eea5e536b41e04c972` 为审核基线，保留所有旧数据、结果和六个历史job记录。

继续使用成熟 `scripts/pi0238_audit_to_sim.py`，R1 wrapper只做launch/preflight/admission/report。先统一预检与实际G0的sealed checkpoint resolver和inventory基目录解析，补严格重复step/candidate/root边界检查；验证PBS完整/截断自任务ID；修复预检初始化与复验的副作用冲突。实际子进程同样绑定modality原点且保留stdlib。用同一resolved_launch进行功能级预检，不能只compile，不在未调用的wrapper副本里修复。

统一实际phase输出与admission/report schema；quota必须有真实容量validator，G4 admission不是推理结果，缺证据不填PASS。G5区分工程/资源/科学/未实现，不使用blockers空不空来猜分类。

预检通过后R2最多2个新增串行CPU审计job，≤2h/job，允许既有auto→small标准资源但CUDA禁用、零模型load/forward/get_action。一次替代仅用于已定位+回归修复的工程失败，不能重跑追结果。quota未知按R1小写入预算继续CPU，不伪装容量PASS。

封存实际G0后连续推进G1/G2/G3；按identity复用完整phase，真实科学blocker交G5。仅全部原PI0238条件准入后才执行既有G4，累计256次queries和原H200预算不增加。G5后停止，selected_checkpoint=NONE。训练/WBC/sim/DEV/F/Final-F/科学契约修改及PR merge仍禁止，不重建monitor。

执行代码/测试/证据推送，Issue #11先记账、PR #77只pointer。回报实际功能测试和phase结果，不只回复“修好/compile通过/等待授权”。
