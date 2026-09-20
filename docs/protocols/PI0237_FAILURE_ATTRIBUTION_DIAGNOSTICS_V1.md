# PI0237 — PI0236 failure-attribution diagnostics v1

日期：2026-09-20（Asia/Singapore）  
状态：`PROTOCOL_PUBLISHED_EXECUTION_NOT_STARTED`  
性质：冻结输入上的只读分析、CPU 阳性对照与有界模型推理；不是训练计划的执行许可。  
仓库：`Leo-6-maker/go2-arx5-hierarchical-vla`  
Canonical scientific ledger：Issue #11；PR #77 仅作 execution pointer，禁止 merge。  
文档基线：`7d3caca79f85dd9fc5e2faa4f81542920414a0a7`，来自 `pi0236/runner-reject-fix`。

## 0. 依据、证据边界与本次授权

### 0.1 依据

本协议落实用户上传的《制定训练计划.txt》中的四步顺序：S5 细粒度只读诊断 → 专家标签同 SafetyGate 阳性对照 → 训练编码/推理解码 round-trip → 固定 TRAIN 拟合诊断。该附件 SHA256：`b9038a639574e312ea1201d7b7c8765dc47d075d4849ce6f45cd9b255f2fb67a`。

直接证据：

- [PI0236 canonical closure](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5746476491)
- [PI0236 PR pointer](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/77#issuecomment-5746483934)
- 基线源码：`scripts/pi0232_train_calib_eval.py`（其内容为 PI0236 outer runner）、`scripts/pi0228_train_calib_checkpoint_selection.py`、`scripts/pi0228_h200_large_adapt_v1.py`、`scripts/stage20s_infer_groot.py`、`scripts/stage21_same_scene_groot_inference.py`。

本协议中的 64+64 样本、256 次推理上限、作业预算、数值核验容限和分层抽样算法，是本次新增的预注册执行约束，不是 PI0236 已观测事实，也不替代旧的科学准入标准。Hopper 原始文件、完整调用链和运行时配置仍须由执行 agent 在 P0 核验；不能把本文件写成已完成诊断的报告。

### 0.2 本轮要回答的三个问题

1. 正确的专家 H40 标签经当前 provider/chunk/SafetyGate，能否满足原准入要求？
2. 真实训练编码与真实推理解码是否保持相同动作含义、统计量和状态锚点？
3. 若前两项成立，fresh-10000 在固定 TRAIN 窗口上是否已学会对应专家动作，且与 TRAIN-CALIB 存在怎样的差距？

不预设答案为泛化失败、归一化 bug、WBC 问题或训练步数不足。`NO_ADMISSIBLE_CHECKPOINT` 是可信的、针对已执行组合的负结果；新诊断只能增加解释，不能覆盖原结果。

### 0.3 明确的授权矩阵

| 操作 | 本协议范围 |
|---|---|
| 读取下文 allowlist 内的既有证据、TRAIN/CALIB 元数据和标签、已授权源代码 | 允许 |
| 编写最小诊断脚本、CPU 单元测试、生成新的 JSON/CSV/Markdown/NPZ 证据 | 允许；只能写入新目录 |
| P0–P3 的 GR00T/VLM 权重加载、get_action、模型 forward | 禁止；均为 0 |
| 无模型的 processor/tokenizer 初始化、实际 action transform/decoder 调用 | P3 允许；必须离线，不能隐式加载网络权重或运行模型 |
| P4 无梯度模型推理 | 仅在全部前置条件通过且用户发送启动 prompt 后允许；最多 256 个 policy queries |
| backward、optimizer/scheduler step、参数更新、训练、DAgger、记忆实验 | 全部禁止 |
| WBC、仿真环境 step、receding loop、机器人硬件 | 全部禁止 |
| DEV-E/F、Final-F 原始数据、结果内容或样本读取 | 全部禁止；不得用来挑选或解释样本 |
| 修改 SafetyGate/阈值、selector、动作时间、frame、StateV1/H40 ABI、processor 语义、数据与权重 | 全部禁止 |
| 新诊断分支的脚本/报告提交、Issue #11 记账、PR #77 pointer | 允许；不改旧评论，不 merge，不 force-push |

这里“只读”指输入数据、权重与科学契约不被修改，不等于全流程零计算。P0–P3 严格零模型推理；P4 明确包含有界新推理。仅发布本协议不会启动任何作业。随附启动 prompt 是交给 PC agent 的明确执行请求，不留空白 `My request:`，也不要求其重复询问同一授权。

## 1. 冻结的起点

PI0236 evaluator `624097.hopper-m-02` 在 `hopper-13` 以 Exit 0 结束。5 个 profile，各 1088 行，共 5440 个 policy queries：

| Profile | ACCEPT | WARN | REJECT |
|---|---:|---:|---:|
| baseline-1000 | 0 | 0 | 1088 |
| fresh-2500 | 0 | 0 | 1088 |
| fresh-5000 | 0 | 0 | 1088 |
| fresh-7500 | 0 | 7 | 1081 |
| fresh-10000 | 0 | 0 | 1088 |

S3 provenance lock PASS；未修改的 S4 selector 返回 `NO_ADMISSIBLE_CHECKPOINT`；selected checkpoint 为 NONE。WBC、训练、DEV/F 均未执行。7 条 WARN 只属于 fresh-7500，不是候选选择理由。

### 1.1 必须核验的文件哈希

| 证据 | SHA256 |
|---|---|
| PI0236 run manifest | `4b9ceff21be16a8a2c15b789cdde7b5ecc598af1cb9f9112f295d4481f2236e4` |
| S3 provenance lock | `101c92b8f9f835ebaa5e9d1b25c844653106d58ace02ed0a2f850d97fe131dd8` |
| S4 selection freeze | `e06f876630e56894a6776e83082f29374fdbe2503598f050db5b76c653ffd8f9` |
| S5 diagnostics | `7a0ad71c1b3a677078f84f75ef08d63035d3af7d397fd6c2f5681c60167581cd` |
| terminal trace | `ffbbb98ab2fd8f9b54d73d7abe4f583b90a621f35c3ad5cf88a189e4c9797f9c` |
| baseline-1000 profile | `24a9feb7dd88650f464ee435aab466b0f843c8ba7e4186e2c425c3ff74363746` |
| fresh-2500 profile | `afb7f7c3a553fe3873f9732020fcb02bd27a03834c40e12970f0ccdee2f6f051` |
| fresh-5000 profile | `b0f76b59a7d911e18f9cf653b7195fbbeaa328ce5d3064f247913b17a2ba3890` |
| fresh-7500 profile | `43f67250f9ca04b35961677edae53cd8ce1691f64a1d7468eeb8d2449c4af38f` |
| fresh-10000 profile | `b24cbc32095be03218a7453b1e641110a7965dacaecab25102f3d587d5349c27` |

证据根：`/scratch/e1192548/pi0228_h200_prep_20260919/evidence/`。具体 run/profile/trace 文件位置从已核验的 manifest/lock 解析；不能因为只有哈希就虚构文件名或认为文件已下载。

### 1.2 输入 locator；最终身份以 sealed manifest 为准

- TRAIN：`/scratch/e1192548/pi0227_large_adapt_v1_inputs/train_dataset`
- TRAIN-CALIB：`/scratch/e1192548/pi0227_large_adapt_v1_inputs/train_calib_dataset`
- baseline：`/scratch/e1192548/pi0227_large_adapt_v1_inputs/checkpoint-1000`
- fresh family：`/scratch/e1192548/pi0228_h200_large_adapt_v1_623593.hopper-m-02`
- GR00T checkout：`/scratch/e1192548/stage22_h200_runtime_deb920_20260902/src/isaac-gr00t`
- GR00T source commit：`51d4c89f72fda44cbf77285c6a8114b52676b8a1`
- modality：`/scratch/e1192548/pi0227_h200_prep_20260919/inputs/go2_arx5_modality.py`；SHA256 `244f37f28e62b7b392a1598b78e78f00e67693f6c08a606b3c5358830ada8b25`
- Cosmos revision：`9ce19a195e423419c349abfc86fd07178b230561`；使用 PI0236 既有本地 snapshot，不下载。
- TRAIN 预期 9792 episodes，tree SHA256 `66559a759c187fdf4d55c175a0808417369c033323e35710b6ca8aeda3786e81`；CALIB 预期 1088 windows。运行时重查 seal、split-lock、tree 算法与内容，不能只比较目录名。

旧 run `623299` 的未完成/已封存 checkpoint 不属于 fresh family，不得混用。checkpoint 的 index hash 不代替所有权重 shard 的内容身份；沿用 archive/manifest 的 shard inventory 做只读验证。

## 2. 全程不变量

保持 `official_dense_relative_H40`、H=40、StateV1=10、seed=22301、action dt=0.05s、语言 `move the end effector to the red target`。沿用 provider 的既有时间网格；40 个从 0 开始的采样点最后是 1.95s，不得重解释为另一个时长。

保持 `SOURCE_FRAME=NATIVE_WBC_RELATIVE_REFERENCE`、`TCP_SEMANTIC_ID=x5_link6_gripper_center`。不能根据变量名把 native-relative reference 擅自当作 world frame，也不能未经审定增加 anchor 变换。P2/P3 必须报告标签、state、decoded 的实际坐标含义。旧 selector 中 reference-target distance 的计算只按旧口径复现；坐标可比性未闭合时加注限制，不能据此断言 world-space 到达。

模型观测只含 body RGB、wrist RGB、state，以及固定语言；target XYZ/ID、未来动作、专家标签、episode/phase ID、split 标签不进入模型。诊断标签与模型 observation 由不同对象/通道携带，调用 get_action 前断言白名单。

沿用 PI0236 的模型推理 dtype、采样器/flow steps、embodiment ID、universal action mapping、processor kwargs、stats、相对变换与 rot6d 行约定。禁止平滑、clipping 新补丁、重定时、覆盖首点、padding 修补或替换 decoder。已有有损变换只能测量，不能删掉。

SafetyGate 使用 PI0236 实際实例配置与 hash；不得用本文件中列出的几个数值自行重建一个缩水 gate。预期关键 reject 阈值为初始位置 0.75m、初始旋转 1.5rad、线速度 2m/s、角速度 4rad/s；完整 warn/reject/arc-length/path-length/aging 配置从原实例记录。若实际不符，先报告 provenance/config mismatch。

## 3. P0 — 只读 admission、实际调用链与样本预注册

### 3.1 检查当前状态，禁止重复执行

先读 Issue #11 的 PI0236 closure、PI0237 canonical 与后续相关评论，以及 PR #77 pointer。若已存在本协议的活动 job、sealed P0–P4 或后续 PI supersession，先核对并复用；不要重新 qsub、重跑已完成样本或覆盖旧文件。有授权冲突则 `BLOCKED_AUTHORITY_CONFLICT`。

使用独立 worktree/新诊断分支，不重置用户的脏工作区，不把 PR #77 的旧 head 当作当前 evaluator 运行基线。记录本协议 immutable commit/blob hash、执行脚本 commit/SHA256、运行时 imported module 的 `__file__`/SHA256。代码文字写了什么与实际 import 了什么必须分别核验。

### 3.2 allowlist 和证据完整性

只允许访问上述 TRAIN、CALIB、PI0236 evidence/权重、锁定 runtime 源码和必要的训练 manifest/log/stats。通过解析已有 TRAIN/CALIB manifest 可加入明确列出的源轨迹标签文件；不得扫描整个 scratch/reports/datasets 树来发现数据。

路径判断用 resolve/realpath 和祖先边界，拒绝指向受限数据的 symlink；不能仅依赖文件名是否包含 `dev`/`final`。不得读取 DEV/F 样本、视频、parquet、指标报告内容；允许读取不包含其结果的协议和源代码。记录 allowlist、被拒绝的路径请求与可审计的读边界。计数为 0 必须有执行路径/日志支持，不能靠硬编码声明代替检查。

核验 1.1 全部哈希，5×1088 行数、唯一键、相同 CALIB IDs、authority 绑定与拒绝统计。source rollout 身份从 split-lock/episodes/grid manifest 解析；TRAIN 与 CALIB source-rollout 集必须不相交。统计量应来自 TRAIN；不能为 CALIB 重新拟合统计量。

创建新输出根 `pi0237_failure_attribution_<UTC run-id>`，不得位于 sealed 输入目录中。HF/XDG/Torch/tmp/pycache 等可写缓存重定向到新根，网络下载关闭。检查实际输出 fileset quota、inode 和预计写入峰值；预算至少保留 `max(2 GiB, 3×预计新增证据字节)` 的可用空间。不复制模型，不删旧 checkpoint，不触发自动清理；不够则 `BLOCKED_STORAGE`。

### 3.3 必须阅读的代码路径

先读本协议和仓库适用的 README/目录说明，再读：

- `scripts/pi0232_train_calib_eval.py`：实际 PI0236 wrapper，fresh family 与 runtime locator；staged 文件可能命名 `pi0236_train_calib_eval.py`。
- `scripts/pi0228_train_calib_checkpoint_selection.py`：`evaluate`、`_build_chunk`、`select`、source-rollout/authority 解析。
- `scripts/pi0228_h200_large_adapt_v1.py` 及其实际调用的 Stage28 配置/数据路径；只读或安全导入必要的纯函数，不执行训练入口。
- `scripts/stage20s_infer_groot.py`：`_build_processor`；其 `_run_one` 是 canonical Stage21 路径的兼容导出。
- `scripts/stage21_same_scene_groot_inference.py`：`_run_one_raw`、`_run_one`、`decode_action` 的真实 state 锚点。
- `scripts/stage40_b0_model_control.py` 和实际使用的本地 Cosmos binding。
- `src/go2arx5_vla/safety/trajectory_gate.py`、provider/chunk 类、SO(3) helpers。
- 从既有 manifest 定位实际 materializer、官方 dataset/window sampler、processor/action transforms、stats loader 与相关单元测试。不要猜它们的位置，也不要用相似但未生效的旧脚本代替。

GitHub 固定基线只是导航起点；PI0236 S3/S2 中实际 helper/import hashes 与 staged runtime 是复现身份依据。发现语义分叉时冻结并回报，不能自动 cherry-pick/升级依赖修复。

### 3.4 在查看细粒度结果之前固定 P4 cohort

只使用 TRAIN/CALIB 元数据、source IDs 与时间索引选样；不使用 gate、loss、reference distance、专家误差或“7 条 WARN”作为选样条件。

预定每个 split 64 个 window，总共 128 个：4 个目标族 A–D × 4 个时间阶段 × 每层 4 个不同 source rollouts 的 window。

确定性算法：

1. 按原 manifest 映射 A–D、jitter family、source_rollout_id 和原 source observation offset。缺失映射不能按字符串猜测。
2. 对每条 source rollout 的有效观测起点按原时间排序，令 rank 为 r、有效起点数为 n，阶段 `min(3, floor(4*r/n))`；不是按目标距离划阶段。
3. 对每个 split/目标/阶段，把 rollout 按已记录 jitter family 分桶。family 按其规范字符串排序；桶内按 `SHA256("PI0237-v1|" + split + "|" + target + "|" + phase + "|" + rollout_id)` 升序，按 family round-robin 选 4 条不同 rollout。
4. 对选中 rollout 在该阶段的 windows，按 `SHA256("PI0237-window-v1|" + split + "|" + source_episode_id)` 升序取 1 条。哈希相同以原 ID 字典序破平。
5. 禁止重复 window、跨 split 同源 rollout、因难度或缺失标签临时换样。某层不足 4 条合法 rollout 时标记 `BLOCKED_COHORT_COVERAGE`，不补重复、不悄悄改为 32/96 等规模；P1 与可执行的 CPU 归因仍可完成，但 P4 不放行。

保存完整 cohort CSV/JSON，包含原 episode/window/offset/rollout/目标族/jitter/phase/split、输入文件定位与 hashes。抽样脚本 hash 和 cohort hash 在 P1 前冻结。已有分析者看过汇总结果不改变此算法；必须明确这是针对已暴露 CALIB 的诊断，而不是新的盲测。

产物：`pi0237_p0_admission.json`、`pi0237_cohort.json`、`pi0237_cohort.csv`、`pi0237_runtime_identity.json`、`pi0237_read_boundary.json`。

P0 数据/身份失败：停止相关下游并输出 blocker；不得加载模型。

## 4. P1 — S5 细粒度只读诊断

对原 5440 条 records 做只读分析；不补 forward，不重运行 selector，不改变 profile status。

### 4.1 必须输出

- 每 checkpoint 的 ACCEPT/WARN/REJECT 核对；每 issue code 的 row prevalence 与 occurrence count 分开；多 issue overlap/co-occurrence。每个比例有分母。
- gate metrics 的 min、p50、p90、p95、p99、max、finite/missing count；用 NumPy quantile 的 linear 方法，记录软件版本。
- 同一 metric 的 observed/原 warn threshold、observed/原 reject threshold；只有正的有限标量阈值才计算倍数。双侧 workspace 与缺失阈值单独呈现，不能硬套比值。
- 按目标族、jitter family、source rollout、阶段、checkpoint 分层；源 rollout 作为主要相关性单位。5 个 checkpoint 在同一窗口上的值是配对观测，不是 5440 个独立任务。
- 同窗口各 checkpoint 的指标变化；相邻 checkpoint 与 baseline 的配对差值。不能从“非单调”直接认定训练发散。
- all-row reference minimum distance 的描述分布；selector 的 non-REJECT 子集统计另列，ACCEPT 与 WARN 数也另列。没有非拒绝行时保留 null/N=0，不填 0，不用 all-row 数字覆盖 ranking 字段。
- 7 条 WARN 的 IDs/issue/已保存 metrics 单独列出，只解释现象，不进入 P4 特选名单，不获得模型候选资格。

已有记录只能支持其中实际保存的量。逐 tick 尖峰位置、频谱、专家误差、clipping/saturation、achieved TCP、WBC tracking 若缺少对应 tensor/日志，标记 `NOT_OBSERVED_IN_SEALED_ARTIFACT`；哈希不能还原 tensor。对已有 array 的再计算标注 derived、来源 hash 和算法，不能冒称原 runner 输出。

产物：`pi0237_p1_gate_diagnostics.json`、`pi0237_p1_issue_table.csv`、`pi0237_p1_metric_table.csv`、`pi0237_p1_observability.md`。

P1 出现真实 REJECT 不算执行异常，必须保留完整分布并继续 CPU 对照。格式、身份或计数不一致则停止相关下游。

## 5. P2 — 专家物理标签通过原 SafetyGate

### 5.1 范围与提取

对完整 1088 条 CALIB windows 和固定 64 条 TRAIN windows 做 CPU 对照（cohort 未能建立时先做完整 CALIB，TRAIN 部分记 BLOCKED）。使用原数据/官方 sampler 在原 window 起点定义的专家 H40 与同观测 StateV1。

不能无条件假定 `table.iloc[:40]` 就等于真实训练 target：先核对 delta indices、是否包括当前时刻、episode 尾部 padding/mask、action 时序、observation offset 与 frame。若真实定义恰为前 40 行，记录该证明；若不是，使用真实定义但把与 evaluator 的差异显式列为待审问题。

禁止跨 source episode 拼接、不记录地补末帧、把 action 当 state、把 ground-truth 首点强制改成 achieved pose。记录 H40 中有效/填充位置；gate 仍按原 H40 口径，拟合误差另报 valid-only 与全 40 点口径。

### 5.2 两条独立的 CPU 证据链

- `E0 physical-label gate`：从标签语义还原专家物理动作，不经过待验证的 normalization round-trip；若物理标签只有通过待测 decoder 才能得到，则标记非独立/缺失，不能伪造阳性对照。
- 使用原 `_build_chunk` 与原 `TrajectorySafetyGateV1`，以原 evaluator 的 now=0、观测 timestamp、StateV1[:3]、同一 rot6d→quat 方法调用。保持 provenance 参数与 dt，诊断身份放在外部 sidecar，不随意改 helper。

检查 shape、finite、valid mask、首点位置/旋转误差、逐段线/角速度、路径长度、全部 gate issues。记录 label hash、state hash、provider/chunk hash、gate config hash 与 source-window identity。

`EXPERT_NONREJECT_PASS` 条件：全部要求的行/映射完整且 REJECT=0；ACCEPT、WARN 仍分别保留。这个标签仅说明专家参考满足原非拒绝条件，绝不证明 WBC 可执行或任务成功。任何 expert REJECT 都是 label/window/gate-consistency 负证据，不能因此放宽 gate。

CPU 合成单元测试必须证明测试线本身有效：已知平滑合法轨迹、单次高速跳变、已知初始跳变、单位旋转、非法形状、NaN。纯静止轨迹可能触发原 zero-arc-length 检查，不能默认把它当必过的阳性例子。

产物：`pi0237_p2_expert_gate.json`、`pi0237_p2_expert_rows.csv`、`pi0237_label_alignment.json`。

P2 实质性负结果不抹掉；允许继续 P3 的纯 CPU 定位以区分物理标签与 codec，但立即封锁 P4。若数据损坏/语义无法唯一确定，P3 只做不依赖该缺失的检查，随后停止。

## 6. P3 — 真实训练编码与真实推理解码的 round-trip

### 6.1 先绑定训练与评估的完整策略身份

分别记录 fresh 训练和 PI0236 evaluator 实际读取的：`processor_config.json`、base `statistics.json`、TRAIN `stats.json`、TRAIN `relative_stats.json`、modality、embodiment tag/数值 ID、132 维映射、mask、相对动作 state anchor、time indices、decode implementation、软件版本与 dtype。

同时比较文件 hashes 和实际解析/注入后的配置内容；文件顺序/包装差异不自动等于语义错误，最终生效值必须能追溯。

特别核查 evaluator 从 `args.train_dataset / meta` 构造 processor 的事实：这可能对 fresh 完全正确，不能单凭路径宣布 bug。baseline 也使用当前 TRAIN stats，故需只读追溯其原 PI0226 processor/statistics。若历史身份不可得，标记 `BASELINE_NATIVE_PROCESSOR_COMPARABILITY_UNRESOLVED`；baseline 仅代表 PI0236 as-evaluated 组合，不代表其原生最佳能力。不为 baseline 自行切换统计量并重新挑选。

同一套错误 encoder/decoder 可以自洽地 round-trip；所以“自循环误差小”不替代与真实训练调用链、标签物理语义和实际 evaluator 的 cross-path parity。

### 6.2 对 P2 同一组标签做三层记录

- `E0`：P2 的独立物理专家标签。
- `E1 lossless/cross-path`：真实训练 action transform/normalizer/mapping/mask 的输出，传入真实 evaluator inverse/decode；逐 stage 记录张量 shape/dtype/hash、有效 slots 与 anchor。不得用手写 encoder/decoder 冒充真实路径。
- `E2 as-configured-lossy`：若真实 pipeline 含 clipping、padding 或 dtype 量化，额外保留该路径，并分开报告有意不可逆操作的损失与实现不一致。没有这些操作就标记 NOT_APPLICABLE，不默认存在。

训练动作编码所需的 processor 模式/参数必须按真实路径执行，但不运行 experiment.run、Trainer、模型网络或任何 optimizer；图像 augmentation 不得污染动作 parity。计算 action transform 可以使用 torch CPU 张量，不需要 GPU。

### 6.3 数值规则与病态情况

本 PI 的 float32 无损核验上限：XYZ 最大逐点 L2 误差 1e-5m；SO(3) 最大 geodesic 误差 1e-4rad；gripper 最大绝对误差 1e-5（按原单位）；归一化有效 slots 最大绝对差 1e-5。旋转角在 float64 计算、trace 输入合法截断；必须同时报告原 rot6d 数值、正交性/退化程度，不能让 SO(3) 正交化隐藏 malformed representation。

这些容限用于数值一致性，不是任务精度/训练合格阈值。运行前若 pinned 实现已有不兼容的既定容限/语义，记录冲突并停止放行，不能看过结果后扩大容限。先用无模型合成输入验证测量精度。

BF16/其他实际量化单列：对同一 tensor 执行相同 cast 的无模型控制，量化误差与 encode/decode mismatch 分开，不将 float32 epsilon 机械套到 BF16，也不自动以 BF16 为借口豁免语义差异。

有意 clipping 必须输出 per-dimension/tick 的裁剪率、边界、相对于原标签的 XYZ/SO(3)/gripper 误差及 gate 决策变化；对实际 clipped 数值的预期逆变换再核验实现一致性。合法但超过上述无损精度、引起 gate 变化或尚无法量化的失真标记 `LOSSY_TRANSFORM_REQUIRES_PI_REVIEW`，不能把其总差全部称为 decoder bug，P4 不自动放行。

必须验证非零 state anchor、非单位姿态、slot 顺序、mask/padding、relative composition 次序、degree/radian 与 m/mm。使用实际 transform 约定；不要因为动作名包含 relative 就擅自改为 SE(3) 乘法或逐元素减法。

### 6.4 P4 admission 的必要条件

P0/P1 完整性通过；P2 完整 expert 非拒绝；fresh 实际训练与推理配置一致；cross-path 无损容限通过；损失/量化已有解释且不引起未解释的 gate/语义差异；输入/标签时间与 frame 同义；cohort 完整冻结；无受限读/参数变化。

如某项尚缺证据，就没有 P4 admission。baseline 历史 processor 缺失本身不构成 fresh 配置失败，但限制 baseline 结论，必须写入 P4 report。

产物：`pi0237_p3_codec_roundtrip.json`、`pi0237_p3_tensor_stages.csv`、`pi0237_p3_processor_identity.json`、`pi0237_p3_lossy_transform.json`、`pi0237_cpu_tests.log`。

## 7. P4 — 有界、无梯度的 TRAIN/CALIB 拟合诊断

### 7.1 只允许这两个 as-evaluated 模型

baseline-1000 与 fresh-10000（必须属于 `623593` family）。不因为 fresh-7500 有 7 条 WARN 改选它；2500/5000/7500 仅做 P1 旧记录分析，不加载、不额外 forward。

两个模型均使用 PI0236 同一 evaluator processor 身份，并明确 baseline 历史可比性限制。固定 64 TRAIN + 64 CALIB，每模型每 window 一次 policy query：**最多 256 个 get_action 请求**。模型内部 flow/denoiser 次数另记，不能与 256 个 policy queries 混淆；采样器保持不变。

不做 teacher-forced model forward、不做随机种子 sweep、不看完结果换模型/窗口、不加测试扰动、不重复抽样找成功。

### 7.2 能复用的先复用，重放身份要明确

若既有 sealed raw/decoded tensors、输入身份和完整 processor/checkpoint lineage 均可验证，则在 P4 row-action-plan 中标记 REUSE，免掉对应 query。仅有 SHA 而无 tensor 不算可复用。

P4 开始前冻结所有 row 的 REUSE/FORWARD 选择、模型顺序、预算与预期 hashes。CALIB 新推理视为对旧观测的 identity replay：记录 raw/decoded SHA 并与 PI0236 同窗口同 checkpoint 原 SHA 比较。任何不一致都保留新 tensor 和差异，标记 `REPLAY_IDENTITY_MISMATCH`，停止后续 queries；不得重新播种、重试到相同、静默改浮点容差或把新 tensor 宣称为原 artifact。需要新的数值复现裁定时回 PI。

固定顺序：baseline CALIB（episode_index 升序）→ baseline TRAIN → fresh-10000 CALIB → fresh-10000 TRAIN。第一条预定 CALIB query 即内嵌 canary，计入 256；禁止另加 canary/暖启动模型 query。模型初始化不计 query，但实际 get_action 在入口处先记 attempted 计数，异常仍占预算。

使用原 `_build_processor`、canonical `_run_one`、原 `_build_chunk`、原 gate；外层只增加只读 instrumentation 与证据保存。允许将原 `_run_one` 返回的 raw/decoded 直接写出，禁止在计算前修改。

### 7.3 必须保存与计算的量

每 window 保存原 input state、专家 H40、valid mask、原输入图像 hash、raw `[1,40,132]`、decoded `[1,40,10]`、model/processor/code identities、seed、原 gate、关联 PI0236 record hash。NPZ 不含 pickle，不把 labels 注入模型输入。

模型对专家的指标必须独立于 target-hit：

- XYZ：逐点 L2，整窗 RMSE、mean、p95、max，首点误差、最后有效点误差；全 40 点与 valid-only 分开。
- SO(3)：同一 rot6d-row 约定下 geodesic mean/p95/max，首点与最后有效点；同时报告无效/近退化 rot6d。
- gripper：MAE/max（原单位）；不要与 XYZ/SO(3) 拼成无单位解释的总分。
- temporal：实际 dt 下的预测/专家逐段线速度和角速度、速度向量差、超限 segment 数、峰值 segment index；derived 指标必须标注口径。
- gate：ACCEPT/WARN/REJECT、每 issue，始终统计所有 row，不用 gate 先筛出好样本再报拟合误差。
- 拒绝后 reference 仍可计算离线误差；不运行 WBC。

逐目标/阶段/jitter 和 source-rollout 宏平均报告；提供 row-weighted 与 rollout-balanced 两种聚合。TRAIN vs CALIB 不是同一 source rollout 的配对实验；不得伪造 row-level pairing。checkpoint 比较才在同一 window 配对。可给分层描述与置信区间，但样本量单位须为 source rollout，不能把 tick/window 当独立样本放大显著性。为保持本轮小而确定，不强制增加 bootstrap 或显著性筛选。

不能要求所有 H40 窗口在窗内到达目标；首尾专家动作也不是 achieved TCP。target distance 仅继承参考路径描述口径，不是拟合主要指标。

### 7.4 运行时安全与作业预算

P4 前生成 `pi0237_p4_inference_admission.json`，绑定 P0–P3、cohort、原 PI0236 identities、执行脚本、PBS、固定 row-action-plan 和预算的 SHA256。完整验证后才允许 qsub。

最多 1 个新的 H200 诊断作业：项目 `CFP03-CF-004`，1 GPU，12 CPU，225GB，walltime 上限 01:00:00；继承已证明可用的 PI0236 queue/GPU request/login-shell 写法，不猜新队列，不降级换硬件。walltime 是调度资源限制，不是完成时间承诺。若原站点契约不支持此短作业配置，停报资源问题，不自行改用其他平台。

P0–P3 优先在现有允许的 CPU 环境运行；如 HPC 政策要求提交，则最多 1 个 CPU-only 短作业（2 CPU、16GB、walltime 上限 01:00:00），使用已知合法的 CPU queue；不存在合法环境时记录 BLOCKED，不占 H200 来解决 CPU 预检。登录节点只做轻量预检，不运行重负载。

禁止重复提交、自动重启、隐式 rerun。若有先前活动 job 则先查询并复用状态。stdout/stderr 直接分文件、逐 query 原子保存 ledger 和 manifest；完整 REJECT 不是异常，不中止后续固定样本。OOM、磁盘、import、身份、预算或 malformed output 失败立即封存；修复后重试需新执行授权，不能本轮自动再跑。

模型 eval + inference_mode；记录训练调用计数、optimizer 创建/step、参数版本/原 checkpoint shards 前后 hash，证明权重未修改。新文件不能伪装为 checkpoint，也不能调用 checkpoint saver。

## 8. P5 — 归因矩阵与停止

P5 是 CPU 汇总，不增加 query。允许多个解释并存，不强制选一个根因。

| 证据组合 | 可报告的结论 | 禁止推出的结论 |
|---|---|---|
| E0 专家物理标签被同一 gate 拒绝 | `EXPERT_LABEL_GATE_MISMATCH_OBSERVED`；定位 label/window/time/frame/gate 的不一致 | 直接认定模型不会学；直接放宽 gate |
| E0 通过，cross-path codec 不一致或出现未解释失真 | `TRAIN_INFER_CODEC_MISMATCH` 或 `LOSSY_TRANSFORM_REQUIRES_PI_REVIEW` | 用修后结果覆盖 PI0236；直接重训 |
| 前述通过，固定 TRAIN 上模型仍明显偏离专家 | `TRAIN_FIT_DEFICIT_EVIDENCE`，提交连续误差和 gate 数据供 PI 审定 | 自动归因纯欠拟合、特定层没更新或纯泛化问题 |
| 前述通过，TRAIN 相对 CALIB 有明确较低误差/较好准入，且分层结果支持 | `TRAIN_CALIB_GAP_OBSERVED_ON_FIXED_COHORT` | 宣称独立盲测已失败、证明 GR00T 普遍不泛化 |
| 缺标签、replay identity mismatch、样本不足或不同指标冲突 | `INCONCLUSIVE` / `MIXED_EVIDENCE` / 相应 BLOCKED | 把缺失写成 0、把相关性写成确定根因 |

本协议没有新设“TRAIN 学得足够好”的任务阈值。是否已经充分拟合以及差距是否有科学意义，由 PI 根据原目标、连续指标和分层覆盖审核；agent 不得自行定一个百分比来宣布泛化失败。诊断用过的 CALIB 不能再包装成独立最终测试。

每个 finding 包含：证据路径/hash、observed fact、inference/hypothesis、alternative explanations、证据缺口和唯一建议的下一类实验。建议不等于执行授权。

终态统一：`STOP_FOR_PI_ATTRIBUTION_REVIEW`，附 attribution_status 或 blocker。保持 `selected_checkpoint=NONE`；禁止调用旧 selector 重新挑选或提交 WBC。

下一 PI 才可能授权：单一语义修复后的新身份重评、数据版本修订、8–16 窗口有界记忆训练、或针对已证实差距的数据覆盖实验。PI0237 不执行任何一项。

## 9. 最小实现、测试、证据交付与 GitHub 记账

实现只增薄诊断层，例如 `scripts/pi0237_failure_attribution.py`、`tests/test_pi0237_failure_attribution.py`、必要的单个 PBS 文件。优先调用既有 helper，不新造 orchestrator/ledger 框架。所列文件名是建议产物，不代表它们已经存在；先查仓库，避免覆盖用户文件。

CPU 测试至少覆盖：计数/unique key/哈希失败、坏 split/受限 realpath、确定性抽样与缺层、阳性/高速/起点跳变 gate、shape/NaN、非单位姿态/非零 anchor round-trip、slot/mask/time 错位、clipping 与量化分账、missing/null、REJECT 后继续、query 上限与 attempted 记账、重复 job 拒绝、原子写入以及禁止训练/WBC 入口。允许 mock 单元测试；mock 结果不得进入真实科学 scorecard。

必须有以下终结产物（未执行阶段保留 status=NOT_RUN 与原因，而不是虚构文件内容）：

- `pi0237_protocol_binding.json`：本协议、源附件、基线、实际执行代码、各阶段输入/输出的哈希；
- P0–P3 产物与 cohort；
- 若 P4 获准：inference admission、row-action-plan、逐 query JSONL、raw/decoded NPZ、profile/fit tables、PBS/trace/stdout/stderr；
- `pi0237_final_report.md`：已确认/待验证/不能判断三类结论，归因矩阵和下一决策；
- `pi0237_run_manifest.json`：phase status、attribution status、开始/结束 UTC、job ID、证据 inventory、observability 缺口和全部计数。

计数分开列：旧 PI0236 queries=5440（历史，不计本轮）；本轮 reused records/tensors、expert-gate evaluations、CPU transforms、attempted/completed get_action、内部 denoiser calls（能观测则计数，不能则明确未观测）、模型权重 loads、训练/backward/optimizer/scheduler steps、WBC/simulator steps、DEV-E/F/Final-F reads、job submissions/retries、selected checkpoint。P0–P3 get_action=0；P4 attempted<=256；所有训练/WBC/受限读取计数必须为 0。

GitHub：将小体积协议/代码/报告/manifest 或 inventory 提交新诊断分支；大 tensor 留既有授权存储并发布明确路径/bytes/hash，不把权重或大量视频上传 Git。Issue #11 发布一个 PI0237 canonical closure（或 blocker），写 phase table、实际 query 数、主要证据与终态；PR #77 仅发该评论的 pointer。不得 edit/delete PI0236 原评论，不改变其 freeze，不 merge PR，不把文档发布说成已经运行。

执行 agent 开始后直接进行 P0；每到科学负结果或机械 blocker，按本协议处理并交付已有证据。无需反复询问是否可以做已经授权的只读诊断；也不得因为计划列出了未来训练方向就越界开始训练。
