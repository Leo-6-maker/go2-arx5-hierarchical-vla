# PI0239 — 无固定工程重试预算的自主推进，直到首个科学失败

日期：2026-09-20（Asia/Singapore）  
决定：`AUTONOMOUS_ENGINEERING_UNTIL_SCIENCE_FAIL_GO`  
发布状态：`AUTHORIZATION_PUBLISHED_EXECUTION_NOT_STARTED`  
仓库：`Leo-6-maker/go2-arx5-hierarchical-vla`  
审核起点：`a30905e03d4009abc419cb7b60b39da3032cfef8`。  
Canonical scientific ledger：Issue #11；PR #77 继续只作 pointer，不 merge。

本协议取代 PI0238/R1/R2 对“工程作业 ID / 工程替代次数”的固定总量限制。它不放宽科学样本、模型选择、SafetyGate、WBC、数据边界或任务成功标准。核心规则：

> **工程问题自主修复并继续，不因工程重试次数停止；只有得到需要改变科学假设、数据/模型/评价契约的真实科学失败，才停止交 PI。**

唯一例外是执行者权限之外、且在所有允许的非破坏性工程路径用尽后仍无法继续的外部基础设施/授权阻塞；这种情况必须报 `EXTERNAL_INFRA_BLOCKER`，不能伪装成科学失败。

---

## 1. 已确认的当前状态

必须先读取以下 canonical evidence，不得只依赖聊天摘要：

- PI0238-R2 closure：Issue #11 comment `5748879506`。
- PR pointer：`5748880095`。
- R2 protocol：commit `9f766201420680956967c6ffc5ffea71cb4990d4`。
- 最新工程修复：commit `a30905e03d4009abc419cb7b60b39da3032cfef8`。
- execution branch：`pi0238/manifest-first-recovery-r2`。
- 原 PI0236/0237/0238/R1/R2 evidence 全部保留。

当前 canonical 事实：

1. G0：`PASS_WITH_QUOTA_UNOBSERVED`。
2. G1：五份 profile 各 1088/1088；旧 selection 结果仍为 `NO_ADMISSIBLE_CHECKPOINT`。baseline 在 1088 个 CALIB 窗口中全部存在 model-extra `LINEAR_SPEED_REJECT`，即对应 expert 行本身没有该 reject、模型行有。
3. G2：完整 expert census 共 10,880 个窗口（TRAIN 9,792 + CALIB 1,088），ACCEPT=324、WARN=10,396、REJECT=160。这个结果是真实的描述性负证据，但 reject 的科学含义仍需按 code/source semantics 分解，不能把 160 个 expert reject 自动解释成模型 5,431 个历史线速度 reject 的根因。
4. G2 的 clock-origin control 当前记录为 `speed_or_initial_rotation_invariant=false`。在将其作为科学结论前，必须先排除诊断实现误差：同进程、同 chunk、同 poses/indices/dt，对 original 与整体平移后的 observation_timestamp/point_timestamps/now 使用数值容差比较；不能将跨历史 artifact 的 exact-float 比较当成时钟因果证据。
5. G3 尚未完成。R2 实际阻塞在 torchcodec 0.8.0 + FFmpeg 8 不兼容，发生于模型加载/forward 前。
6. `a30905e` 仅绑定既有 sealed Stage45B FFmpeg6 native runtime；账本报告 exact Hopper CPU import regression 已通过 `torchcodec.VideoDecoder + ShardedSingleStepDataset`。该补丁属于工程环境绑定，不改变科学契约。
7. G4 未准入、policy query=0；训练/backward/optimizer/WBC/sim/DEV/F 均为 0；selected checkpoint=NONE。

因此当前终态应理解为：

`ENGINEERING_CONTINUATION_READY / SCIENTIFIC_ATTRIBUTION_INCOMPLETE`

而不是新的训练失败、泛化失败或模型能力结论。

---

## 2. 新治理规则：工程预算不设固定总次数

### 2.1 不再因为“重试次数用完”停止

从本协议开始：

- 不设 G0–G3 工程 job ID 总量上限。
- 不设因为 path/import/ABI/library/PBS/queue/storage/logging/serialization/cache/runtime binding 等工程错误导致的固定替代次数上限。
- 不设纯工程修复的累计 walltime 总预算。
- H200/CPU 作业的**单次资源仍使用完成任务所需的最小已验证资源**；不因“预算无限”扩大单次资源、并发多份作业或开启 speculative replicas。
- **任意时刻同一科学阶段最多一个 active job。** 未确认旧 job 已终止/取消前，不得重复提交。
- scheduler 资源不足可以等待、取消后按同语义重新提交、或使用已知兼容且项目允许的队列/节点；这些属于工程恢复，不需要 PI 再授权。
- 每次替代必须保留 failure evidence、根因、最小修复 diff、针对性 regression 和新 attempt lineage。禁止原样重复坏配置。

“工程预算不限”不等于“科学采样不限”。科学样本集合、模型集合、seed 和评价规则继续冻结。

### 2.2 工程自主修复 allowlist

PC agent 可以自主修改并继续：

- PBS/queue/resource syntax、job routing、launch wrapper、cwd、env、LD_LIBRARY_PATH、FFmpeg/torchcodec/动态库绑定；
- Python import/path、manifest/path resolver、hash/inventory、staging、resume、atomic IO、logging、traceback、serialization；
- offline cache/tmp/HF/XDG/Torch cache 重定向；
- fileset quota/bytes/inodes 的观测与容量证明；
- 当前 attempt 自建的临时文件/cache/失败输出清理或迁移，前提是先封存 evidence inventory；
- 将新输出/evidence/cache迁移到**同项目已授权且不会改变数据/权重身份**的可写 fileset；
- CPU/H200 作业因机械错误的恢复、续跑和阶段级 resume；
- 针对工程 bug 的 unit/integration/exact-env regression；
- 性能优化、日志压缩、分阶段 checkpoint/evidence 写入，只要不改变科学计算结果。

不得自主修改：

- sealed TRAIN/CALIB/DEV/F 数据内容、split 或 target authority；
- model checkpoint bytes、initializer、seed、language、model inputs、target injection；
- official dense-relative H40、StateV1、frame/TCP、dt、rot6d semantics；
- SafetyGate/scorer/selector 阈值与 ranking；
- loss、LR、optimizer、scheduler、trainable modules、模型容量、训练数据配比；
- WBC source/checkpoint、success radius、termination/reward；
- 任何为了“让结果通过”而做的 smoothing、retiming、clipping patch、least-bad checkpoint substitution；
- PR merge。

上述科学项需要新的 PI 科学决策；出现证据要求修改这些项时，就是 science stop。

---

## 3. 立即阶段 A — 从 G3 继续，不重跑已完成 G0–G2

### 3.1 Evidence reuse

如果下列身份未变化，则 G0/G1/G2 直接复用，不重跑 10,880 expert census：

- R2 G0/G1/G2 evidence SHA；
- TRAIN/CALIB tree/split；
- PI0236 profiles；
- PI0237 cohort；
- selector/provider/gate/SO(3) helper；
- mature runner 的与 G0–G2 相关代码 hash。

仅对**受实现 bug 影响的子诊断**做 revision。例如 clock-origin control 可以只重算 12-case control 并形成 `G2_CLOCK_CONTROL_REV2`，不覆盖 G2 census。

### 3.2 G3 runtime continuation

使用 `a30905e` 的 sealed Stage45B FFmpeg6 runtime，正式 G3 job 必须同时封存：

- `LD_LIBRARY_PATH`；
- FFmpeg runtime root/manifest/hash；
- torchcodec version 与实际 loaded shared libraries（可用 ldd/loader diagnostics）；
- Python/torch/CUDA runtime identity；
- actual modality `__file__`/SHA；
- actual `ShardedSingleStepDataset`、collator、processor/decode source identity。

exact-env import regression 只是 admission；真正 G3 仍必须经过实际 batch construction。

### 3.3 G3 必须回答的科学问题

在固定 PI0237 128 cohort + 12 expert rejects + 2 prior normalized excess 的去重集合内：

1. 真实 training dataset/collator 是否产生预期 `[B,40,132]` target/action representation；
2. active slots、mask、padding/time anchor 是否与真实训练配置一致；
3. 真实 train-side target 经独立 eval decode 后，XYZ/SO(3)/gripper 是否在既定物理容限内；
4. normalized excess 是可解释的数值 roundoff/scale 效应，还是实际 mapping/clip/anchor 语义差异；
5. modality/state/camera/language/input construction 与 evaluator 是否同契约。

工程错误（FFmpeg、import、loader API、路径、缓存、内存、序列化）自动修复直到 G3 能给出**有效科学结果**。

### 3.4 G3 的 science stop

若在工程环境、真实配置与数据身份已闭合后，仍观察到以下任一情况，则停止并标 `SCIENCE_FAIL_TRAIN_INFER_CONTRACT`：

- 实际 training target/mask/active slots 与冻结 action contract 不一致；
- 独立 decode 后物理 XYZ/SO(3)/gripper 超出既定容限，且不能由纯数值误差解释；
- state anchor/frame/rotation semantics 存在真实不一致，需要改数据、ABI、processor 或 evaluator 科学语义才能解决；
- expert reject 的根因被证明是 sealed label/reference 与冻结 controller contract 的真实语义矛盾，需要改标签或 gate。

这类结果不得通过工程重试“试到 PASS”。

---

## 4. 阶段 B — Storage/quota 作为工程问题自主闭合

`FILESYSTEM_QUOTA_UNOBSERVED` 不再因为重试预算导致整个项目停止。PC agent 应自主闭合容量证明。

优先顺序：

1. 使用站点已知/可发现的合法 user/project/fileset quota 工具，记录 bytes/inodes used/limit/available/grace。
2. 若站点不暴露 parseable quota，但科学作业的最大新增写入可以精确上界，则允许使用**同一 fileset 的非稀疏 reservation proof**作为工程容量证据：
   - 先计算本阶段最坏新增 bytes/inodes（NPZ、JSONL、logs、cache、临时原子写入）；
   - 加固定安全余量；
   - 写入真实非 sparse reservation、flush+fsync，并验证实际 allocated blocks/inodes；
   - admission 封存成功后才释放 reservation 给同一作业使用；
   - 这证明的是本阶段写入容量，不冒称账户全局 quota 已知。
3. 或将**新产生的 output/cache/evidence**迁移到同项目已授权且容量可证明的 fileset；输入数据和 checkpoint 保持原 sealed identity/read-only。
4. 可以删除当前协议生成且已封存 hash/inventory 的 transient cache、失败 attempt 临时文件；不得删除 sealed datasets、正式 checkpoint、canonical evidence 或历史科学结果。

Storage engineer path 可无限恢复；只有平台拒绝访问、无任何授权 fileset 且无法在权限范围解决时，才报 `EXTERNAL_INFRA_BLOCKER`。

---

## 5. 阶段 C — G4 固定 cohort 模型拟合诊断

### 5.1 科学集合不扩张

模型固定：

- baseline-1000；
- fresh-10000，仅来自 623593 family。

样本固定：

- PI0237 64 TRAIN；
- PI0237 64 CALIB。

seed、processor、dtype、sampler、language、action contract 全部保持。

科学集合最多是 **2 × 128 = 256 个唯一 policy-query cells**。这不是工程重试上限。

### 5.2 工程失败时的 query 恢复规则

- 对每个固定 cell，第一份**合法、有限、身份完整**的结果一旦产生即冻结，不因数值差而重跑。
- 如果 job 在某 cell 之前或该 cell 未产生合法结果时机械失败，可以无限次工程恢复该 cell，所有 attempt 记账。
- 不得重复已有合法 cell 来挑更好的随机输出。
- 若 deterministic seed/runtime 下为了验证工程恢复必须重放，重放结果标 `ENGINEERING_REPLAY`，不替换首份合法科学记录；异常差异本身进入诊断。
- 断点恢复从剩余 cell 继续，不重新跑完整 256。

### 5.3 G4 输出

对 expert-vs-model 报：

- XYZ RMSE/mean/p95/max、first-point、valid endpoint；
- SO(3) geodesic；
- gripper；
- linear/angular velocity segment；
- model-extra SafetyGate issues；
- TRAIN/CALIB、A–D、phase、jitter/source-rollout 分层；
- baseline vs fresh 同-window paired comparison。

target min-distance 仅副指标，不能替代 expert-action fit。

### 5.4 G4 science stop

完成固定 G4 后，根据证据分类：

- `SCIENCE_FAIL_TRAIN_FIT`：fresh-10000 在 TRAIN 上仍产生系统性的 expert-action/kinematic mismatch，无法归因于工程路径；
- `SCIENCE_FAIL_GENERALIZATION_GAP`：TRAIN 明显符合 expert/准入，而 CALIB 出现稳定、分层一致的模型额外失败；
- `SCIENCE_FAIL_MIXED_MODEL_DATA`：数据/label 与模型问题并存，必须修改科学 recipe/data contract 才能继续。

任何上述结果立即停止，提交 PI；**不自动再训练、不自动 DAgger。**

若 G4 证明历史近全量 reject 可由已确认的工程 evaluator/runtime bug 解释，且修复不改变预注册科学语义，则这仍是工程修复，不算 science fail：继续阶段 D。

---

## 6. 阶段 D — 工程修复后的完整重评与唯一 selection

仅在 G3/G4 没有 science fail、且需要的修复被证明是“实现恢复到原科学契约”时执行。

1. 固定修复 commit/runtime identity；
2. baseline comparison + fresh 2500/5000/7500/10000 全部 1088 CALIB profiles；
3. 不混用旧/新 profile；
4. unchanged selector 只对四个 fresh candidate；
5. 正常 REJECT 完整保留，不重跑挑成绩；
6. 冻结唯一 selected checkpoint 或返回 `NO_ADMISSIBLE_CHECKPOINT`。

若在已验证正确的科学契约下仍 `NO_ADMISSIBLE_CHECKPOINT`，这是 `SCIENCE_FAIL_NO_ADMISSIBLE_CHECKPOINT`，停止。不能选择 least-bad。

---

## 7. 阶段 E — 无 science fail 时自动进入 frozen-WBC sim

若存在唯一 frozen selected checkpoint，则不再因工程问题停下。

### 7.1 One-chunk A–D C0

同一个 model/runtime，TRAIN A/B/C/D C0：

`fresh RGB + StateV1 → GR00T → dense-relative H40 → original SafetyGate → persistent native reference → frozen WBC → achieved world-TCP`

保持原 0.035m task scorer。

记录 model reference、gate、WBC tracking、base/body/dog telemetry、achieved TCP initial/min/final/first-hit、native termination、physical cleanliness。

工程错误（sim launch、Isaac runtime、IPC、path、logging、render、GPU binding）自主修复并继续。

科学停止：

- accepted/model reference 本身明显不到目标：`SCIENCE_FAIL_MODEL_TASK_TRAJECTORY`；
- accepted reference 正确但 frozen WBC 系统性无法跟踪：`SCIENCE_FAIL_WBC_TRANSFER`；
- 任一预注册 A–D C0 cell 在正确执行后任务失败：保留四 cell 完整结果并停 PI，不换 checkpoint/seed。

### 7.2 Progress-preserving receding

只有 A–D C0 one-chunk 通过预注册 success 时自动继续：

- 每次 query 必须来自执行后的新 RGB + StateV1；
- 多次真实 `get_action`；
- reference lifecycle/progress/time 单调；
- 不把每个 H40 reset 到 s=0/t=0；
- 不平滑/重定时隐藏输出；
- 因果 trace：execute → observe → infer → gate → update reference → execute。

工程故障自主修。

若 lifecycle invariants 都通过但任务仍失败，标 `SCIENCE_FAIL_CLOSED_LOOP` 并停。

---

## 8. 阶段 F — 覆盖与后续大规模训练路线

若当前 frozen family 在 one-chunk + receding 下通过，则可在**已有 TRAIN perturbation family**内继续固定协议覆盖，不把 A–D reaching 称为多技能 general model。

DEV-E/Final-F 与新训练的治理：

- DEV-E 仍应作为一次性评估集合，只有 TRAIN 闭环成功、checkpoint/runtime 完全冻结后才读取；本协议允许在达到这些条件后进行一次性 DEV-E，不用因工程问题另停。
- DEV-E 一旦读取，不允许据此回调当前 model/selector 后再把同一 DEV 当 blind test。
- Final-F 保持最后锁；只有完整 TRAIN + DEV 证据包均通过后再进入一次性 final evaluation。
- 硬件继续锁定。

如果任一阶段出现 science fail，需要新训练/数据/模型 recipe，则立即停 PI。新的大规模训练可以作为下一科学授权的主线，但**不是工程修复**。

未来新 large-adapt family 获批后，同样沿用本治理：
工程故障不限重试；scientific cohort/seed/checkpoint rule 固定；第一次真实 science fail 停止，而不是烧算力追 PASS。

---

## 9. Science fail 与 engineering fail 的判定原则

### ENGINEERING — 自主修，继续
例：路径、manifest、hash implementation、PBS、scheduler、资源不足、import、FFmpeg、torchcodec、CUDA binding、cache、tmp、quota observability、output capacity、serialization、logging、OOM/磁盘（可在不改科学语义下修）、sim bootstrap、dependency binding。

### SCIENCE — 立即停止
满足以下任一：
- 需要改数据标签/split/target authority 才能解释结果；
- 需要改 action/state/frame/TCP/dt/rot6d contract；
- 需要改 SafetyGate/scorer/selector；
- 真实 train/eval mapping/anchor 不一致且不是实现偏离 intended contract；
- 当前模型在正确 TRAIN 数据上拟合失败；
- TRAIN 拟合但 CALIB 出现稳定 generalization gap；
- 正确 evaluator 下 fresh family 无 admissible checkpoint；
- frozen WBC 正确执行但 one-chunk/receding 任务失败；
- 需要新 loss/LR/optimizer/model capacity/trainable modules/additional training/DAgger。

**判定 science fail 需要有效执行和证据；“作业报错”永远不是 science fail。**

---

## 10. 必须维持的操作纪律

即使工程预算不限制，也必须：

- 一次只跑一个同阶段 job；
- 不以 metric 好坏决定是否重跑；
- 不丢弃失败 attempt；
- first-valid-result freeze；
- phase-level resume，不从头反复烧已经完成的 G0/G1/G2；
- 所有修复 commit、runtime、job ID、resource、stdout/stderr、evidence SHA 记账；
- Issue #11 先记 canonical，PR #77 只 pointer；
- selected checkpoint 未经正式 selector freeze 前始终 NONE；
- 不重建低频 monitor，除非用户另行要求。

---

## 11. PC agent 启动 prompt

现在执行 PI0239：工程问题自主修复并持续推进，**不再因为工程 job/retry 次数达到某个预算而停止**。只有得到需要改变科学数据、模型、评价或控制契约的真实 science fail 才停止交 PI。

先完整读取：
- Issue #11 comment 5748879506；
- commit a30905e03d4009abc419cb7b60b39da3032cfef8；
- 本 PI0239 immutable protocol；
- 原 PI0238/R1/R2 scientific contracts。

接手现有 G0/G1/G2 evidence；identity 未漂移则复用，不重跑完整 10,880 census。先修正/复核 12-case clock-origin control 的 same-process comparison，再使用 a30905e 的 sealed Stage45B FFmpeg6 runtime 继续 G3。正式 G3 必须实际构造 training batch/collator 和 [B,40,132] mapping/mask，封存真实动态库、modality、processor、dataset identity。

任何 path/import/PBS/FFmpeg/torchcodec/queue/storage/quota/cache/logging/serialization 等工程失败：保留证据→最小修复→回归→继续。没有固定工程重试总次数；但任意时刻仅一个 active job，不原样重复坏配置，不 metric-driven retry。

quota 无法由站点工具直接观测时，按 PI0239 使用同 fileset 的最坏写入上界 + 非稀疏 fsync reservation proof，或将新 output/cache/evidence 迁移到同项目已授权可证明容量的 fileset；不删 sealed 数据/checkpoint/evidence。

G3 科学契约闭合后进入固定 G4：baseline1000 + 623593 fresh10000 × 固定64TRAIN+64CALIB。科学 cell 集合固定256，不增加样本/seed/model；工程失败可以恢复同一 cell，已有合法结果 first-valid freeze，不因表现重跑。

G4 若证明 TRAIN fit deficit、TRAIN→CALIB generalization gap、真实数据/合同冲突，立即以 SCIENCE_FAIL_* 停止，不自动训练。若只发现工程 evaluator/runtime 偏离且修复恢复原科学契约，则自主修复后完整重评四个 fresh checkpoints，使用 unchanged selector freeze 唯一 checkpoint。

如产生唯一 admissible checkpoint且无 science fail，直接按协议进入 frozen-WBC TRAIN A–D C0 one-chunk；工程 sim/runtime 问题自主修。正确执行后的任务失败才作为 SCIENCE_FAIL 停止。A–D one-chunk 全部成功则继续 progress-preserving receding closed loop；工程问题继续自主修，真实闭环任务失败再停。

训练/backward/optimizer/DAgger、新 recipe 只有在 science fail 后由新 PI 科学决策授权；不能把它们伪装成工程修复。PR #77 不 merge。

每个阶段实时写 execution state 和 evidence inventory。最终只在：
1. SCIENCE_FAIL_*；
2. 或权限之外且无合法工程路径可继续的 EXTERNAL_INFRA_BLOCKER
时停止。
