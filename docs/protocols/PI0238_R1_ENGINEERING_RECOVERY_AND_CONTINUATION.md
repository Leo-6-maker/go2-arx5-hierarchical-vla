# PI0238-R1 — 工程恢复放行与 G0–G5 连续执行

日期：2026-09-20（Asia/Singapore）  
决定：`ENGINEERING_RECOVERY_GO_WITH_PREFLIGHT`  
发布状态：`AUTHORIZATION_PUBLISHED_EXECUTION_NOT_STARTED`  
审核基线：`d553ca3a9f1145f72e131f2d66aefd5c32200f1e`，分支 `pi0238/audit-to-sim-execution`。  
本文件是 PI0238 的工程执行补充，不是新训练 recipe，也不重新定义 PI0236/0237 的科学结果。Issue #11 继续 canonical；PR #77 只作 pointer，不 merge。

## 1. 审核依据与当前决定

必须读取：

- [PI0238 原协议](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/044e4d5ef8b4c1effb8236190c0eb083fa49979b/docs/protocols/PI0238_AUDIT_TO_LARGE_SIM_CLOSED_LOOP_GOALS_V1.md)及 canonical comment `5747200805`。
- [最新执行账本 5747407663](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5747407663)、[PR pointer 5747408922](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/77#issuecomment-5747408922)。
- 审核基线下 `reports/PI0238_EXECUTION_STATUS_20260920.md`、`scripts/pi0238_audit_to_sim.py`、`tests/test_pi0238_audit_to_sim.py`、`hpc/pbs/pi0238_audit_cpu.pbs`。
- 修复 commit `67bda86f38c8d1c2bd1fb1797f087f5b8b856944`、`b31874ccac95d3144b11cda9fdd90e5829a9dba1`。

GPT 本轮读取了以上 GitHub 账本、提交差异、当前源码和原协议；没有登录 Hopper，没有直接重读远端原始日志、实际执行测试或复算远端数据 hash。以下源码缺口是静态审核发现，不能当作已执行 G1–G3 的新结果。执行 agent 需读取原始终态与 failure artifacts 做接手核验。

### 1.1 结论

这次是 `PRE_SCIENCE_ENGINEERING_BLOCKED / SUBMISSION_BUDGET_EXHAUSTED`，不是新的科学失败，也没有得到可以决定训练 recipe 的新证据。原 `STOP_FOR_PI_RECIPE_DECISION` 保留为历史协议停止 token，但报告必须另有工程 blocker 原因；不能因此推断欠拟合、不泛化或需要再训练。

历史作业逐一保留：

| Job | 账本记载 |
|---|---|
| 624206.hopper-m-02 | 站点注入标准资源后、诊断脚本输出前取消；账本称无科学读取 |
| 624209.hopper-m-02 | auto→small，hopper-27；G0 的 tree hash 实现排序与 canonical 不符 |
| 624229.hopper-m-02 | 唯一替代，auto→small，hopper-15；G0 的 PBS modality 路径不存在 |

两个已执行 attempt 均报告模型 load/forward/get_action=0。不能把零科学调用等同零资源申请，也不能漏记 624206：累计提交、取消、实际运行、资源申请和科学调用是不同计数。新 R1 预算从本授权起单列；旧作业不被改写成未发生。

**立即授权工程修复和 CPU 预检；满足下文提交条件后，恢复 G0→G1→G2→G3。完整证据满足原 PI0238 G4 准入时，直接进入已授权的有界 G4，再交付 G5。无需为常规阶段转换再次请求 PI。** 发布本文档不等于已运行任何作业。

## 2. 已接受的两项补丁，以及提交前必须补齐的工程检查

### 2.1 Tree hash 修复保留，增加真正的顺序回归

`67bda86` 将目录遍历时即时 digest 改为先收集 `(relative_path, bytes, file_sha256)` 再按相对路径排序，与既有 PI0228 canonical tree 算法方向一致。不要修改 sealed expected hash，不重新 seal，不把首次误报写成实际数据变化。

现有测试只创建一个普通文件并验证 symlink 拒绝，无法捕捉顺序错误。新增多目录、多文件、不同创建/枚举顺序的 fixture，与 pinned canonical implementation 或独立固定 golden digest 比较；测试 prefix、NUL/newline 格式、relative POSIX path、文件内容/大小变化、symlink 和排序。TRAIN prefix `train`、CALIB prefix `train_calib` 不得混用。

轻量测试通过后，再在合法 CPU 资源中执行真实 tree 验证。不得为发现另一处缺路径先反复遍历整份数据；路径与小文件身份检查应排在昂贵全树 hash 前。

### 2.2 Modality 的文件路径和实际 import 必须是同一个身份

`b31874c` 修正了 PBS 字符串；固定 artifact 为：

`/scratch/e1192548/pi0227_h200_prep_20260919/inputs/go2_arx5_modality.py`

SHA256：`244f37f28e62b7b392a1598b78e78f00e67693f6c08a606b3c5358830ada8b25`。

但当前 G3 在建立相关路径绑定前仍执行裸 `import go2_arx5_modality`。文件存在和参数正确不等于 Python 实际加载了它。必须在干净子进程用原训练所要求的注册方式，将唯一已核验文件绑定为该模块，记录 `__file__`、SHA256、公开配置符号及注册结果；同名模块已从别处载入时拒绝，不以全盘搜索/随便加 PYTHONPATH/选择任意同名副本掩盖问题。文件内容与 modality 科学定义不得修改。

用一次共享的 `resolved_launch.json` 驱动预检和 PBS 真正 argv，不再维护两套不一致的路径常量。最少包含：协议链、脚本/test/PBS hash、解释器、TRAIN/CALIB、PI0237 evidence、PI0236 run、prep、VLA source、Stage10 GR00T、modality、baseline、fresh root、Cosmos、本次新 output/cache/tmp。所有参数以完整 argv 数组记录，含空格路径不靠未引用的 shell 拼接。

新输出根由 run-id 唯一生成；当前 PBS 写死的 `...T034100Z` 属于旧失败 attempt，不能继续写入。不得手改已提交或排队作业的 staged 文件。

### 2.3 增加入口级预检，而非只有工具函数单测

在现有薄诊断脚本增加安全的 `--preflight-only` 或等价入口，默认不执行 g0/g1/g2/g3，不实例化网络、不全树扫描、不读视频/Parquet 内容、不计算科学指标。测试文件不得 `source` 会直接执行 phase 的 PBS 来冒充预检。

预检使用**将要提交的同一份配置、解释器和 staged scripts**，汇总全部轻量失败，而不是修一个再排队发现下一个。执行：

- shell syntax/argv schema、所需脚本和配置存在可读、输出根与输入根隔离、write/cache 目录新建检查；
- 全部小文件 hash、modality/processor 配置、relative_stats、cohort/manifest/inventory 的必需键；
- 必需路径 resolve/allowlist，禁止 `..`/symlink 跳出；仅从已批准 manifest 定位文件，不扫描整个 scratch；
- inventory 相对路径以被声明的 inventory/evidence 基目录解析，不能使用不确定的当前工作目录；hash line malformed/重复/越界均失败，inventory 为空不能因 `all([])` 成为 PASS；
- pinned 解释器可用、模块 import 原点与实际 loader/collator 的方法签名；若登录节点政策不允许重 import，则先静态解析和轻量规格检查，worker 最前面完成同一 exact-env import 预检，不在登录节点偷偷运行重负载；
- 相关 safe imports 不生成 stats、不写 sealed dataset、不加载 GR00T/VLM 网络，CPU 工作明确设置 CUDA 不可见；
- 当前同协议活动作业与锁检查。已提交未回收 job 不另发新 job。

新增回归必须让原两种错误真实失败：乱序 tree fixture、不存在/错误 hash/错误 import 原点的 modality、错误 PBS argv/旧 OUT、相对 inventory 在不同 CWD 下解析、quota 错误输出、stage admission 不完整。只有“7/7 tests”不构成 launcher 资格。

## 3. 不要让修复后脚本生成假准入：G1–G3 的具体完善

这些要求是落实原 PI0238 契约，不允许修改模型、数据或评价定义。

### 3.1 G1

当前专家关联/额外线速度汇总只遍历 `by_step[1000]`；扩展到五个已存在 profile，每个都有完整 1088 行专家 join 和分母。专家实际速度应取 sealed gate metrics，不因专家没有 LINEAR_SPEED_REJECT issue 就记为数值缺失或零。保留全行/非拒绝、rollout-balanced、分层/相邻 checkpoint 配对与 missingness 区分。

### 3.2 G2

当前 clock control 只把 `now` 设为1、未移动 observation/point timestamps；这仅改变 age，不能称为统一 clock-origin shift 对照。新增独立 sidecar 控制：同一数据整体平移 now/observation/point timestamps，保持 pose、indices 和 dt，比较原 gate 的 age/速度/首点角度；不改原专家记录或生产 provider。

原 rot6d 退化/正交性应在正交化之前度量；不能仅检查 Gram-Schmidt 后矩阵的正交性来声称原表示健康。12例 casebook 应有原 source/state/desired/sync 对应证据；缺项标 UNOBSERVED/UNRESOLVED，不用写死字符串代替实际来源。完整 TRAIN/CALIB gate census 继续按原预算执行，可复用通过身份核验的已有记录。

### 3.3 G3（必须补全，不能仅修 import 后运行现版本即宣称合格）

静态发现：当前两个 processor 均由 Stage21 builder 构造；training loader 另行手选参数；action/mask 用“首个形状匹配张量”查找；mask 可缺失；active slots 写死0–9；只算 XYZ/gripper、不算 SO(3)，且不 enforce 物理误差容限；`processor_train_eval_resolved_equal=True` 与 `PASS_READ_ONLY` 是写死值；数值 casebook 只是复制原两条 excess；三种 negative controls 没有实际实现。

必须改为证据驱动的判断：

1. 只读追踪真实 `623593` training resolved config、dataset/step sampler、processor factory、augmentation 和实际 loss 消费的 tensor key；不是把 eval builder 新建两次就当训练/推理相等。允许从原 training code 安全提取纯配置/数据函数，不调用训练入口或会重算 stats 的 qualification main。
2. action/mask/embodiment key 来自 pinned loader/模型 loss 源码所证明的命名与规则；不能静默选择第一个 shape 相同的张量。mask 若实际是其它 shape，则按真实且已证明的 broadcast 规则记录；若真实 loss 用隐式 mask，记录该执行规则。缺少这类证据一律 BLOCKED，不人为塞全1 mask。
3. active/inactive slot 集、每维有效数、time padding mask 由真实 mapping+实际张量观测导出，再与契约比较；不把0–9写成“observed”。实际 tensor、有效 mask 和独立 decode 必须真实经过本次测量。
4. 固定128窗口与12reject/2excess并集最多142不变；CPU audit 可用1–2样本微批逐个调用真实 collator，记录batch形状/索引与 padding，避免一次攒142份图像处理结果。此微批仅是审计执行内存安排，不改变训练 batch size、模型推理 batch 或样本定义。
5. 同时输出并 enforce XYZ、SO(3)、gripper 物理差异；沿用原1e-5m、1e-4rad、1e-5容限。normalized原1e-5超差仍保留；按原PI0238 §5.2测实际分母、cast、ULP、clip predicate/影响和误差界，不把复制旧表称为“已解释”。
6. 真正运行 wrong slot、wrong mask、wrong anchor、非单位姿态的阴性对照；检查应能拒绝被注入的错误。测试与真实评估输出分离。
7. `execution_status=COMPLETE`、`parity_status=PASS/FAIL/UNRESOLVED`、`numerical_status`、`g4_admitted` 分开；只有完整准入检查计算出的 conjunction 可以设 `g4_admitted=true`。G0里的 `quota_observed` 不能单独放行G4。

当前 CLI/PBS 只实现并调用 G0–G3，没有 G4/G5 执行器。不得报告“脚本会自动完成全部”。允许在本次范围内实现薄 G4/G5 wrapper，复用原 `_run_one`/provider/gate，加入测试、输入白名单、预注册计数和独立 admission；验收完成再提交 G4，而不是为网络诊断重新发明 runtime。

## 4. Quota：分清探测到信息、容量合格、诊断有限可写

`returncode==0 and stdout非空` 不足以证明实际输出 fileset 的 quota 合格；其他挂载、只有表头、配额0/不限额含义未解析、inode耗尽或已超限都不能 PASS。未知持续记为 `FILESYSTEM_QUOTA_UNOBSERVED`，不能用 df/free 替代。

应按站点已有可用工具/帮助和既有记录，定位实际 user/project/fileset/mount，解析 byte/inode 的 used、limit、available/grace，保存原始输出、单位、时间、命令exit与被测路径。site无已知合法工具则不猜管理员命令，不升级权限，不改配额。

**本R1允许 G0身份核验、G1–G3 的有界只读CPU诊断在 quota 尚未可观测时继续**，但必须单列 `CPU_STORAGE_LIMITED_DIAGNOSTIC_ONLY`：

- 实际新输出根做一次不超过16MiB、写入+flush/fsync的专属探针，验证新增文件基本可写；只删除该自建探针，不触碰输入；这不是容量/配额充分性证明。
- 明确CPU输出及新cache预测量，新增总写入硬上限1GiB；只写小报告/CSV/所需审计张量，不复制权重、视频或整份dataset，不下载缓存。超过这个硬上限需停止或提交新预算，不默认扩容。
- 保持 `df` 的观测性余量至少max(2GiB,3×预计新增字节)及正inode，另有写入上限、运行时容量/error检查。它降低本轮小写入风险，但不产生quota PASS。若探针/余量不满足则停止对应计算，先提交现有证据。
- 身份通过而quota未知的状态必须拆分记录，不能变成全局 `G0=PASS/G4=true`。

G4仍要求原PI0238的真实quota/输出峰值admission。CPU阶段得到完整科学诊断后，即使因quota不能启动G4，也要交G5已有证据和精确缺项，不只回复“无权限”。quota未知不授权占用GPU做试写，也不授权删除/搬迁历史模型。

## 5. 新执行预算与平台 auto 资源适配

### 5.1 本次只更新工程重试与CPU工作的资源规则

旧 CPU 提交额度确已用尽；本R1显式提供**最多3个新增、串行的G0–G3审计作业ID**（首次+最多2次满足条件的机械失败替代），每作业walltime不超过2h，申请时限合计不超过6节点小时。取消、自动路由、作业进入队列后未执行也计入3个ID；不能通过改run名或再发新PI子号重置本额度。qsub未返回ID的失败请求也记账，最多每份已修正提交配置重试1次；未知是否已入队时先查询确认，不能盲重提。

允许复用已知的平台 `auto→small` 标准申请：项目 `CFP03-CF-004`，1节点、站点配置不超过1GPU/12CPU/225GB/2h。G0–G3仍只用CPU、禁用CUDA，模型loads/forward/get_action=0。这是**CPU计算占用带GPU配额的标准节点**，不能称为scheduler的CPU-only allocation或gpu-resource=0。若已有合法可用CPU-only资源可直接承载相同环境，可优先使用；不得为了重新寻找不存在的队列另提交探测作业。

站点分配高于以上资源上限需停止并回报，不自动接受。仅因资源上有GPU、但与本段已授权配置一致，不能再取消并重提。绑定真实queue/node/resource_list与请求值，申请GPU时间和实际GPU模型调用分别记录。

### 5.2 允许自主替代提交的必要条件

必须是非科学故障：错误路径/import、诊断hash实现、日志/序列化、站点零模型活动的启动错误等。保留失败原因、已读取/已输出范围、counter、scheduler终态；给出最小修复commit，并让针对该错误的回归先失败于旧代码、通过于新代码，再做全部入口预检。不能复用同一个已知坏配置重试。

即使G1/G2已生成只读科学描述，未改变数据/模型/契约、零新policy query的机械失败也可恢复；不再用“曾读过一个标签”阻止合法的工程恢复。已经得到的完整结果按精确输入/helper/脚本身份校验后复用；阶段算法修改则产生新的phase revision并保留旧结果。不得选择更好结果、删除已发现的真实不一致、跨不同科学身份拼接。

实际数据 hash变更、训练/推理mapping不一致、未解释的物理或数值误差、目标输入污染、非法数据读、模型调用预算超限等不是工程重试：停止G4并交G5科学决策包。未知原因先定位，不能贴“工程”标签继续。

### 5.3 G4额度不追加、不降门槛

原PI0238的G4条件授权仍有效：固定64TRAIN+64CALIB，baseline1000和623593-family fresh10000，最多256 attempted policy queries，第一条在预算内兼作canary。没有新增种子/样本/候选，专家REJECT仍保留，不强制逐bit复现旧tensor；执行安全/selection规则不变。

G4默认1个H200诊断job，沿用已验证站点请求：1GPU/12CPU/225GB、walltime≤1h。仅首次job `get_action=0` 的明确工程失败可按原协议有1次替代，总G4提交ID≤2、申请时限≤2h；已进入真实query后异常不自动重提。所有PI0238/R1的G4合计≤256，不能按新run根重置。

CPU标准auto节点不是自动获准G4的节点；不要为了复用它悄悄解除CUDA禁用。G4须有完整独立admission与正确H200/runtime证明。若站点不接受原G4资源约定，报告具体资源冲突，不把CPU的2h授权挪给G4。

## 6. 防止每个机械修复都从零跑：阶段恢复和日志

薄wrapper须在每phase起止写append-only state与原子manifest，保存input/implementation/protocol hashes、状态、所处理唯一行数、输出inventory、attempt编号。真实终态由工具/记录核验，不从旧聊天推测。

不覆盖旧`g0_failure.json`；新的attempt使用新根，旧结果可通过只读path/hash引用。恢复G1–G3前检查依赖G0身份和source契约仍有效；修改tree verifier/import binding会使相应G0子检查失效，必须重做，不用缓存掩盖真实变更。已完整核验的数据tree可在同次稳定输入/受控write边界下引用已有证书并做完整inventory/变化检查；检测到不一致或无法证明未变时再做必要的完整核验。

避免把输出巨大JSON当唯一日志。stdout/stderr与每phase摘要持续落盘；failure路径封存traceback、exit code、实际argv、当前阶段和已完成行的计数。不要靠最后一行“PASS”或任何硬编码0代替真实执行边界。

## 7. 短期完成定义、G5与仍锁定事项

按原PI0238完成G0–G3以及合格时的G4，并提交G5。G0入口测试通过不是交付终点；不能完成某一条修复就再次回复“准备就绪、等待授权”。

G5最少包括：修复diff与回归表；三项旧作业和新增作业完整attempt账；真实quota状态与资源；五profile专家/模型逐键分离；完整专家census/12例溯源；actual-batch/slots/mask/XYZ-SO(3)-gripper/数值解释；G4真实数据或严格NOT_ADMITTED原因；只有证据支持的归因；一条主线训练/修复run card及必要备选。

通过CPU但存在真正parity/numeric科学问题时，停止于`STOP_FOR_PI_RECIPE_DECISION`并明确`SCIENCE_BLOCKER`；只有工程/资源失败则`BLOCKED_ENGINEERING_OR_RESOURCE`，兼容保留旧token但不伪称已做科学决策。额度用尽则`ENGINEERING_BUDGET_EXHAUSTED`。完成有界模型诊断后仍是`STOP_FOR_PI_RECIPE_DECISION`，不得自动训练。

全程保持 selected_checkpoint=NONE。训练/backward/optimizer/scheduler、DAgger、小窗口记忆实验、WBC、simulation/receding、硬件、DEV-E/F/Final-F内容读取继续禁止。不得改变gate/阈值/selector、action representation、StateV1/H40、frame/dt/target输入、loss/LR/模型容量；不得smoothing/retiming、重选最不差checkpoint、改sealed数据/统计量/权重、清理旧模型或merge PR。

原PI0238 G6–G9大规模训练与sim长程路线继续有效，但它们仍需相应的新科学授权。本R1解决的是进到科学测量之前的工程阻塞，不将训练或仿真授权混入预算放行。

## 8. 执行启动 prompt

现在按 PI0238-R1 执行工程恢复，不再等待是否恢复CPU预算的确认。

先完整读取本文件的immutable publication commit、原PI0238协议 `044e4d5ef8b4c1effb8236190c0eb083fa49979b` 和Issue #11最新canonical。以 `d553ca3a9f1145f72e131f2d66aefd5c32200f1e` 为审核基线，保留 `67bda86`、`b31874c` 及旧624206/624209/624229证据。

先做完整launch preflight和回归：canonical多文件tree digest；modality文件/实际import同SHA；一份resolved argv；新OUT；inventory基目录；quota解析与独立G4 gate。补全G3真实训练配置、132/mask/active slots、SO(3)、数值casebook及阴性对照，不能使用当前写死的PASS字段。薄G4/G5 wrapper可在既定范围内实现，不改科学源代码。

入口预检通过后，按R1新增预算最多3个串行CPU审计job（≤2h/job），允许既有auto→small的标准GPU配额节点但全程禁用CUDA/零模型调用。只针对已定位且回归修复的机械故障替代提交；复用经身份核验的完整phase，不重新抽样或重做已完整的模型实验。quota未知只按R1有界CPU写入条件继续，不伪装quota PASS、不放行G4。

连续推进G0→G1→G2→G3；全部原PI0238新诊断准入满足后直接做G4，保持原2模型×128窗口、累计≤256 attempted queries和原H200预算。G5交完整决策包后停止；训练/WBC/sim/DEV/F/Final-F与merge继续禁止。

代码/测试/报告推新execution修复commit；Issue #11先记账，PR #77只发pointer。回报phase/status、全部job IDs含cancelled、实际resource、queries/禁区计数、证据hash、归因或精确blocker，不只回复“已读、等待确认”。
