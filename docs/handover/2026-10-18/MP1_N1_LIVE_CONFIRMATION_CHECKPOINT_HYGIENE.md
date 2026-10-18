# MP1-N1：在 SOURCE_CONFIRM 持续执行期间清理明确失败的中间 Checkpoint

日期：2026-10-18（SGT）
状态：**USER-DIRECTED CHECKPOINT HYGIENE / EXISTING SINGLE-OWNER HANDOFF / NO DELETION RECEIPT / NO NEW COMPUTE GO**。
适用仓库：`Leo-6-maker/go2-arx5-hierarchical-vla`，现役分支 `pi0267/executor-decoupled-recovery`。
基线审计 commit：`27e47eb80e692b7ac319db892c185cd5d3f9cc1b`（2026-10-18 12:14:42 SGT）；现役执行 actor `6b291000858612423b67838275efa52caf8b633f`，不要将 GitHub HEAD 热加载到在途进程。
用户新增要求：**“在推进实验的同时，需要清理明确失败的中间checkpoint”**。本卡只把已允许的失败 checkpoint 退役整理为随实验持续运行的安全流程；没有通过本卡直接访问或删除 Hopper/CoreLab 文件。遵守 `AGENTS.md`、`START_HERE.md`、统一工作区入口，原 MP1/N1 权限、单 owner、clock、财政账、安全与数据合同全部保留。

## 最新科研阶段事实（不要让清理耽误确认）

- `S-BC@2048` 和 `S-CORR@2048` 两臂各完成 2048 updates/65536 访问、全部 24 终点条件；BC 暴露 DEV 3/8，CORR 5/8。CORR T1 DEV 4/4 vs BC 2/4；T2 都是 1/4。样本访问匹配，但 C1 实际独立 student parent 仅 4，不能宣称普适因果 DAgger。
- 唯一 SOURCE_CONFIRM 候选是 **S-CORR@2048**，full-state manifest SHA256 `94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38`。10 个新场景组 × 两任务 × 两固定噪声银行，40 条冻结条件；不因单个失败改变 scene/noise/checkpoint。
- `source-confirm-01` native `673388.hopper-m-02` 在最新已发布收据中 **R**，allocation1/2，family PBS11/12；GPU 费用还 OPEN，不能提前结算、结束、缩短或重启。已实际独立审计的新 F0/001 T1、bank29202 自主32s成功，80 fresh ACK/641 paired RGB/41/41 hold；这只是一条确认正结果。另有明确的原S1负例与一条T2工程中断的原件，剩余40格不得外推。
- 现有原生600秒/80次传输/8隔离故障资格已通过；actor 维持原始固定 S1/Teacher OFF/D0/EEF10/H40/prefix8。Vanda 目标域 Go2+ARX5 WBC 尚未证明完整任务，TF-R3 native safety STOP 不变。
- 上一份 **`reports/PI0294/20261016T191445Z-localpc-mp1/FAILED_CHECKPOINT_RETIREMENT_CHECK_01.json`** 已实地审查 Hopper907 + CoreLab SSD1751 + AMD Home1412 个登记目录，当时大型合格删除候选**0，删除文件0、实际释放0 bytes**。不能把这个历史结论当成本日所有新产物的永久分类，也不能假称已经清理出了空间。

## 执行优先级与最小非阻塞工作

现有唯一 local-PC Codex owner 在不抢占/取消/热改 `673388` 和任何健康 physics/model/audio(?) 进程的前提下，开启**只读、低优先级、登记 CPU 子工作**，在 Hopper、CoreLab-4080S、AMD 中仅检查本人/本项目实际登记的 `runs/models/archive` 根。先读取最新 source-confirm 原生 job/leases/worker PID+start identities、持久 mutation journal、checkpoint manifests、model selection/endpoint references、quota 和 registry。关闭后再考察退役；若相关平台不允许无干扰扫描，就在正常作业边界执行。清理不新建 owner/第二调度器、不提交 PBS、不做模型推理/物理仿真，也不重置 N1 时钟或预算。若过了 N1 hard close，停止新增清理操作。

### 必须保留的核心集合

| 身份 | 固定 checkpoint manifest SHA256 | 退役规则 |
|---|---|---|
| 原 M1@1024 | `7dba1c4c8abacdb57e2faa0e6d7563ed5dcef2e5473cb13c281eb0e57d5a5150` | 永久保护，实验父模型 |
| S-BC@512 | `1cf83ad430c03c1d953266562cf90f3d0475f49c806701448fe5750d494c4d8c` | 固定中期有效负例，对照必留 |
| S-BC@2048 | `ac0f46924d540ec5519f8c64b1ec0ea1c7f37bba96df0a21744e6a6a8e88efc2` | 完整基线/终点必留 |
| S-CORR@512 | `31f0514beb4003e42afc3437aa4781b5df47fc35a6270ba5c4fa081b31102d02` | 固定中期资格/恢复与对照必留 |
| S-CORR@2048 | `94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38` | 当前唯一 SOURCE_CONFIRM 权重，**在活跃确认期间绝不可删除** |
| S-BC@1631 | `a54ab1ab635f7ea221ffd982e71b503752c40d7c6574d9db16e309d4facbec0c` | 合法完整恢复中间证据，不是“明确失败”，先保留 |
| 其余已发布 full-resume witness | 以最新原生 manifest/审计 SHA 为准 | 只有确认无任何 resume/replay/科学证明义务后才可单独重新审议 |

另外，所有已验证 C0/C1 纠正 bank、源 TRAIN/DEV/CONFIRM manifests 与实际 RGB/raw/ACK/物理数据、原生安全及失败证据、最终行为矩阵、权重与统计处理器、HF 已发布身份清单、WBC/vendor/环境/他人文件均受保护。一次任务 VALID_TASK_FAIL 或 512 检查点 0/4 **不是**可删除 checkpoint 的判据。

### 允许成为清理候选的类别（严格取交集）

1. 本项目本人登记的 **INCOMPLETE/INVALID** 临时 checkpoint payload：例如原子保存未完成、成员缺失/哈希不一致，已由 ledger 与完整后继证明不是任何合法 resume 或科研对照；不存在未结算的写入或原生消费者。
2. **明确被废弃**且证明无引用的失败 trial 临时权重、大型缓存或重复工件；不是最终候选，不是唯一本体/训练中间见证，不是任何有效负例或失败归因的唯一证据。
3. 独立完整后继封存、所有控制与消费者终态、科学证明已另有完整来源后，明确可再生且未被引用的多余 scratch/shard/cache。有效 checkpoint 的旧中间版本 **默认不删**，除非另有逐对象的可复现实验与引用证明。

对于每个候选，必须满足以下 **全部** 条件才可触碰：

- 验证 canonical owner、register root、目录解析及设备身份，拒绝 symlink/hardlink 冒认、跨根逃逸、无权目录、整个 Home 枚举/删除；不跟随任何符号链接目标。
- 原生 job/PID+start/worker/lease/owner 真实终态，对候选执行 `no active consumer/open handles/no pending writer` 检查；未知状态不删。正在执行 SOURCE_CONFIRM 的 runtime/model/中继/产物根保持不可变。
- 读取所有 RUN_MANIFEST、checkpoint / parent lineage、model/processor/stats、evaluation/probe、fullstate resume、S-CORR selection、HF revision、审计和失败 witness 引用。至少一次 exact sha/path/consumer 双向查证。任何引用或无法判明引用都保留。
- 该对象确有已核实的大型 bytes，且移除不会使唯一 scientific replay、真实 resume、审计诉讼原件或已授权收尾失去资格；不能仅凭名字含 `failed`、`checkpoint`、`smoke` 断言有可删模型。
- 先生成 **exact enumerated DRY_RUN deletion intent**，包括对象路径安全摘要、来源和身份、目录/文件哈希、引用证明、原状态、候选 allocated bytes、排除原因、最小必要原件保留方式。由同一 mutation owner 仅对已证明的冷态 exact files 操作；不得做模糊递归 `rm -rf` 或不审计的通配符清理。
- 一个对象删除以后立即核实文件实体不存在、原生 quota/allocated blocks 变化、实际释放 bytes、保留的小型 member/manifest/科学索引仍可回读；`logical file size`、压缩预计值或旧历史退役 bytes 不得算成本次实际释放量。若删除是否生效未知，先只读核对，绝不盲重发。
- 事后再次检查健康 job/owner/leases/ledger/GC 安全状态；若异常立即保留证据并停止该清理 lane，不修改 S1/WBC/plant、原数据、配额、当前 checkpoint 或任务判定。

如果没有符合条件的候选：**按之前正确实践写 `eligible=0, deleted_files=0, actual_freed_allocated_bytes=0`，不要为了显示清理效果删除一份有价值的模型。**

## 现场回报与科研线路

将此次新的现场清理审核记为 **`reports/PI0294/20261016T191445Z-localpc-mp1/FAILED_CHECKPOINT_RETIREMENT_CHECK_02.json`**（名称为应交产物，**当前未产生**），至少包含 actual observation time、唯一owner、各根候选数和理由、protected identities、all excluded items、删除意图/实际作用与对应哈希、实际磁盘/个人quota前后、释放字节、全部受影响/未影响 job/lease、资源CPU实际/保守计费和运行安全。不上传真实私有绝对路径、token、原始模型权重或图片到 GitHub。

优先向 **Issue #47** 登记清理的真实技术收据；只有确实影响候选资格/原生模型执行或科学资源时，才在 **Issue #11** 交叉记录。不要以清理报告替代确认评估报告，也不要以“视频发布完成”冒充 40 条任务成功。

当下实验目标不改变：持续使 `source-confirm-01` 的 40 条预先冻结条件走向有效终态，在缺失与有效安全失败上不偷换分母；完整确认前不解封新模型/不回流 CONFIRM 训练。成功候选仍只限于 S-CORR@2048；原 S1、任务评分、冻结 WBC 与 TF-R3 停止保留。用户本次不要求再次改动采样器、重训旧模型或新增训练预算。

**本文件是用户要求在实验同时开展合格 checkpoint 清理的 GitHub 接续指令，不是 CoreLab/Hopper 实际删除凭据，不承诺扫描会发现可删除对象。**
