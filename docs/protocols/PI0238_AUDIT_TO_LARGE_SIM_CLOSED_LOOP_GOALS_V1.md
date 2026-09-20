# PI0238 — PI0237 审核、归因补全与大规模训练—sim 闭环长程 Goal

日期：2026-09-20（Asia/Singapore）  
发布状态：`PLAN_PUBLISHED_EXECUTION_NOT_STARTED`  
项目：`Leo-6-maker/go2-arx5-hierarchical-vla`  
Canonical scientific ledger：Issue #11。PR #77 仅作 pointer，禁止 merge。  
审核基线：`998eba5dc571a600a6c25a350b4a0b5dd2dad40a`，`pi0237/failure-attribution-execution`。  
本文件同时包含近程执行授权、科学停止门和长期训练/sim 路线。列入路线不等于自动获准执行。

## 0. 审核依据、已确认结果与证据限制

必须完整读取以下 canonical 材料，不能只读用户摘要：

- PI0236 结果：Issue #11 comment [5746476491](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5746476491)。
- PI0237 协议：commit `6e83c5494772190f5c8082f730f06723d9c9bfda` 下 `docs/protocols/PI0237_FAILURE_ATTRIBUTION_DIAGNOSTICS_V1.md`，canonical comment [5746790269](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5746790269)。
- PI0237 结果：Issue #11 comment [5747055582](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5747055582)，PR #77 pointer [5747056426](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/77#issuecomment-5747056426)。
- 本次审核基线下：`reports/PI0237_FAILURE_ATTRIBUTION_DIAGNOSTICS_V1.md`、`manifests/pi0237_run_manifest.json`、协议 binding/inventory、`scripts/pi0237_failure_attribution.py`、`tests/test_pi0237_failure_attribution.py`。
- 真实 gate/provider/数学与数据路径：`src/go2arx5_vla/safety/trajectory_gate.py`、`src/go2arx5_vla/providers/groot.py`、`src/go2arx5_vla/trajectory/so3.py`、`src/go2arx5_vla/data/materialization.py`；实际训练 adapter、Stage28 dataloader/processor、Stage20S/Stage21 inference helper 必须沿 manifest 追踪。

本次 GPT 已读取 GitHub canonical、报告和上述关键代码；没有登录 Hopper、执行 CPU 测试或复算远端逐行 artifacts。远端根 `/scratch/e1192548/pi0237_failure_attribution_20260920T014727Z` 的内容须由 PC agent 核验。GitHub 报告支持的数字和源码支持的行为，与未观察到的远端细节必须分开。

### 0.1 接受的进度

PI0236 的 5×1088 profiles 和 `NO_ADMISSIBLE_CHECKPOINT` 保留。PI0237 发现了专家标签问题并停止 P4，没有因为进度目标越过训练/WBC/DEV/F 边界；这一停止是正确的。

| 项目 | 来源支持的结果 |
|---|---|
| TRAIN/CALIB source rollout | 报告为 612/68，交集为 0；固定 cohort 64+64 |
| 历史模型 records | 5440；`LINEAR_SPEED_REJECT=5431`，`ANGULAR_SPEED_REJECT=228`，初始旋转 REJECT=25，初始位置 REJECT=12；issue 非互斥 |
| 专家 CALIB 1088 | ACCEPT=37，WARN=1039，REJECT=12 |
| 专家 TRAIN 64 | ACCEPT=2，WARN=62，REJECT=0 |
| 专家 REJECT 原因 | 12 条均为 `INITIAL_ROTATION_DISCONTINUITY_REJECT` |
| 当前 10 维 processor 回环 | XYZ 最大 1.51470672531452e-7m；SO(3) 最大 1.42503412487299e-5rad；gripper=0 |
| normalized 回环 | 最大 1.65253877639771e-5；2 条超过旧 1e-5 检查 |
| 新 policy queries / 权重 loads / 模型 forward / H200 | 均为 0；没有训练、WBC、仿真、DEV/F 读取 |

不能将 12/1152 的专家起始姿态问题直接解释为 5431/5440 的模型线速度问题。精确的“模型额外线速度错误”应在同 CALIB 行上关联专家 gate 后确认，不能用 1152 与 5440 不同分母直接相减。

### 0.2 审核发现：保留测量，但收窄完成声明

**A. P3 是有价值的 processor 自循环证据，不是完整训练—推理 cross-path 证明。** `run_p3()` 用 Stage21 `_build_processor` 构造同一 processor，调用其 `state_action_processor.apply`，拼接 `(40,10)` 并直接交给 `decode_action`，再用同一 processor re-encode。`raw` 实际为 `[1,40,10]`，没有通过实际训练 dataset/collator 的 `[B,40,132]` 输出、active slots/mask、实际训练实例与实际 evaluator 实例的独立对照。代码把 `padded_universal_dim=132` 和 `cross_path_semantics_exercised=True` 写入报告，不能替代这条缺失的执行证据。修订解释为 `PROCESSOR_10D_SELF_ROUNDTRIP_OBSERVED / FULL_TRAIN_INFER_PARITY_NOT_ESTABLISHED`，保留旧文件，不追改历史 PASS。

**B. `LOSSY_TRANSFORM_REQUIRES_PI_REVIEW` 是旧协议的审查状态，不是已经确认 clipping 的因果结论。** 旧容限确实超过了；物理回环误差却远小于旧 XYZ/SO(3) 数值容限。当前没有 clip-outlier 测量，不能把没有测到 clipping 写成 clipping=0，也不能认定误差来自裁剪。下一步区分 float32 舍入、归一化尺度放大、量化和真正有损操作；不得直接改旧容限为 2e-5 让旧 P3 PASS。

**C. P1 汇总执行了，但原协议要求的诊断尚未全部闭合。** 已有 issue CSV 带分层字段，不等于完成 row prevalence/co-occurrence、各组分母/分位、相邻 checkpoint 的配对报告。`_numeric_stats` 过滤非数值后永远返回 `missing_count=0`，使全缺失输入也被记录为零 missing；外层还固定 `observability=OBSERVED`。必须新增正确摘要，不覆盖原表。`warn_rows` 是 warning issue 行，不一定是 decision=WARN 的 7 个窗口；必须分清。

**D. P0 声明不是全部约束已自动执行的证明。** 代码对 S3/S4/S5 和各 profile 有实际 SHA 核验，但 run manifest 只是计算/记录 SHA，未对 pinned 值做等式断言；数据 tree 主要比较 seal 中的字符串；实际 import hash 主要记录而未逐项对 admission enforce。allowlist/禁区计数是声明，未见统一 realpath 开文件拦截；quota 命令输出虽保存，容量 gate 实际使用 `disk_usage.free`。这是审计覆盖缺口，不是发现了实际数据泄漏或实际 quota 不足。补全证据，不重做已封存的真实 forward。

**E. 3/3 tests 的范围仅为 ID 解析、cohort 确定性/唯一性、缺失值输出 null。** 尚无 gate 阳/阴性、非单位旋转、132 槽位/mask、真实 cross-path、路径/哈希拒绝等要求的测试。不能用 3/3 表述完整 CPU 契约验证。

**F. 时间原点不能独立解释首点旋转拒绝。** gate 的初始旋转是 `angle(R_state, R_chunk[0])`；线速度使用相邻 XYZ 差和相邻 dt。统一平移时间戳而不改索引/采样，不改变这两个量。时间错配仍可能改变所选 state/action 行，但必须证明“选择了错误行”，不能仅以绝对时间未归零作原因。provider 已明确将输入 row-rot6d 转为 canonical quaternion/column-rot6d，不能因两种命名不同就再次转置。

### 0.3 不预设唯一根因

当前支持：专家与 gate 有少量起始姿态不一致；模型参考的线速度违规广泛存在；完整训练—推理 parity 尚未证明；拟合与泛化仍不确定。任何确定归因必须具有 case-level 原始值与独立对照。`INCONCLUSIVE` 不允许被改写成成功，但也不应成为无限审计、永不测量模型拟合的理由。

## 1. 总 Goal 与权限边界

长程交付目标：一个 shared checkpoint + 一个冻结 processor/runtime，经原 SafetyGate 与 frozen WBC，在 TRAIN A–D C0 得到实际 achieved world-TCP 任务成功，并完成 persistent、progress-preserving、使用新观测的多次真实预测闭环；随后再按冻结协议测试覆盖与 DEV。A–D reaching 不是不同操作技能，不直接命名为通用多任务模型。

近期必须交付：`CPU_GATE_AND_FULL_PATH_AUDIT` + 条件满足时 `FIXED_COHORT_MODEL_FIT_DIAGNOSIS` + 下一轮具体可执行 training/repair run card，而不是重复交付“已准备好”。

| Goal | 动作 | 本次权限 |
|---|---|---|
| G0 | 接手、证据核验、最小诊断层修复/测试 | 立即 GO，CPU/只读输入 |
| G1 | 补全历史 S5 分层与同窗口归因表 | 立即 GO，无模型 |
| G2 | 专家起始旋转 casebook、源标签/状态对应和 TRAIN 全量专家检查 | 立即 GO，无模型/WBC |
| G3 | 真正 training-batch→132 slots→eval-decode parity、数值归因 | 立即 GO，无模型权重/forward |
| G4 | 固定64 TRAIN+64 CALIB、baseline1000/fresh10000 的有界诊断推理 | G0–G3 的诊断准入通过后 GO，最多256 attempted policy queries |
| G5 | 归因决策包；准备小规模训练、修复或大规模训练的精确 run card | 现在可写代码草案/CPU tests；提交后 STOP_FOR_PI_RECIPE_DECISION |
| G6 | 16-window 有界训练自检/单变量 pilot；再大规模训练 | 本次不执行；新的 PI recipe GO 才能训练 |
| G7 | 新 family 完整 selection freeze、frozen-WBC one-chunk | 训练 run card 内预注册；WBC 必须新 PI release |
| G8 | progress-preserving receding 与 TRAIN 覆盖 | one-chunk 合格且新 PI release 后执行 |
| G9 | 受控 DEV、最终测试和真正多技能路线 | DEV 单次新授权；Final-F/硬件继续锁定 |

### 1.1 明确的 PI0238 协议修订：诊断准入不等于执行准入

本文件对 **PI0238 的 G4 非执行诊断**，显式替代 PI0237 §6.4 中“所有专家必须无 REJECT 才允许推理”的条件。理由：测量模型在同一已知样本上为何产生高速输出，不需要把任何轨迹交给控制器。12 条已知专家拒绝继续保留，不能删掉或改成通过；G4 全 cohort 纳入并分层报告。旧 PI0237 的 blocked 状态正确且不变。

归一化旧 epsilon 超出本身不再与任何形式的“危险输出”画等号，但 G3 必须对真实 train/eval parity、物理误差和数值失真给出可验证解释；本协议不是免查 processor。

PI0238 G4 是 **新的、同一冻结配置下的诊断测量**，不是重建 PI0236 原 tensor。与原 raw/decoded SHA 相同可标 `EXACT_REPLAY_VERIFIED`；不相同但权重/输入/processor/runtime/sampler 身份核验通过时，标 `NEW_MATCHED_CONTRACT_DIAGNOSTIC / NOT_BITWISE_REPLAY`，保留差异并按预注册顺序继续新诊断，不重试追哈希。输入/权重/配置身份不符、非有限/非法张量则仍停止。这一变化仅避免把 bitwise identity 当成科学语义的充分必要条件，不授权修改原 profiles/selector 或声称新结果属于旧 run。

原模型选择的 zero-REJECT 规则、SafetyGate 全配置、WBC、task scorer、阈值、StateV1/H40、语言、target 边界、seed=22301、dt=0.05、模型容量/损失/LR 全部不变。改变其中任何内容，先回 PI。工程自主权只适用于不改变上述语义的诊断脚本/IO/环境绑定/日志与测试。

## 2. G0 — 接手与最小工程修复

1. 先查 Issue #11 最新 PI supersession、同协议 active job 和已有 artifacts；接手真实状态，不从旧消息推断，也不重复 qsub。只读获取当前工作区；有未提交用户代码时用新 worktree，不 reset/clean。
2. 固定本协议发布 commit、执行代码 commit、原 `998eba5...`，实际运行 helper/import 的 `__file__` 和 SHA。不根据 PR #77 head 替代运行基线。
3. 复用 PI0237 原 cohort JSON（已公布 SHA `dfd60b94bbdd82cd07af2bbb93185e8cec720f6f4a88a33bb2cfe3a2eabce9e6`）；核验原算法、grid metadata 与所有 IDs。禁止重抽样或为了覆盖/成绩换样。ID parser 与 canonical grid 对照，不能仅从正则成功推出 authority 成立。cohort 没有视频/state/action 指纹时补入新 sidecar，不改 cohort membership。
4. 对 PI0236 run manifest、S3/S4/S5、五 profiles 和 PI0237 P0/P1/P2/P3/inventory 按原 canonical 的具体 SHA 实际核对。原 PI0236 run manifest expected SHA 为 `4b9ceff21be16a8a2c15b789cdde7b5ecc598af1cb9f9112f295d4481f2236e4`。manifest 文件路径从 verified inventory 解析，不猜。seal 字符串不代替 file inventory validation；无需重拷贝权重，但 G4 load 前必须核验实际 shards 而不只 index。
5. 对 dataset、source labels、model/runtime 和输出路径实施 resolve/realpath allowlist；不扫描整个 scratch，不读取 DEV/F 内容。合法训练/数据脚本可能 import 常量，不得因模块名字含 Validation 自动调用或读取其数据。静态确认无副作用；如需隔离纯函数，保留调用来源证明。
6. 修复 missing/finite/non-numeric/NaN/Inf accounting，区分 `total_count`、`finite_count`、`missing_count`、`nonfinite_count`；禁止 filtered count 冒充总分母。完整缺失量标 NOT_OBSERVED。issue occurrence、unique affected row 和 decision 计数独立。
7. 对 issue co-occurrence 与 checkpoint pair join 使用 `(source_episode_id, episode_index)` 显式键，不仅 `zip`；缺键/重复/错序可重排核对但不能无声丢行。
8. 原 binding 将 40 位 Git blob ID 命名为 `protocol_blob_sha256`；新增更正字段 `protocol_git_blob_id` 与实算 64 位 `protocol_file_sha256`，保留旧值来源，不覆盖历史 binding。
9. 输出新根，例如 `/scratch/e1192548/pi0238_audit_to_sim_<UTC_ID>`。未获授权不删/改旧 checkpoint、数据与 provenance。真正的 fileset byte/inode quota 才是容量依据；站点工具按既有证据使用，不猜新工具或用 df 代替。证据预算测量后冻结，余量至少 max(2 GiB, 3×预计新增 evidence bytes)。不得因 quota 命令不存在却 df 很大自动 PASS。
10. 配置 HF/torch/tmp/pycache 等指向新目录，离线运行；不下载模型、升级依赖、替换全局 runtime 或新造 orchestration 框架。

CPU 测试应新增且实际运行：路径跳出/symlink、错误 SHA、错误 family、错误/缺失行键、全缺失/nonfinite 分母、co-occurrence、clock-origin invariance、速度/初始旋转阴性例、非单位姿态 row/column 正反例、非零 anchor、132 active slots/mask/permutation/padding、lossy accounting、禁止训练入口和 forward budget。纯 mock 结果不能进入科学 scorecard。

完成证据：`g0_identity_and_access.json`、`g0_cpu_tests.*`、`cohort_binding.json`、精简 review correction 表。不得重新运行 PI0236 五份评估来补日志。

## 3. G1 — 两种失败分离，补齐已有 records 的信息

对原5440 records 输出完整、可重算的 per-checkpoint/per-target/per-jitter/per-phase/per-rollout 分母、issue prevalence/co-occurrence、metrics 的 p50/p90/p95/p99/max 和原阈值倍数。同窗口比较 baseline、2500、5000、7500、10000，既有 baseline 配对也有相邻 checkpoint 配对。报告 row-weighted 与 rollout-balanced，不能把重复窗口/ticks当独立任务。

与 P2 的完整1088 CALIB专家逐键 join：输出 expert-vs-model gate confusion table、模型额外 LINEAR_SPEED_REJECT、专家有/无初始姿态REJECT两组中的模型速度分布。确认有无专家速度拒绝，不能只从汇总漏项推断。

P1 已报告 all-row reference min distance median：baseline 0.258416m、fresh2500 0.759239m、5000 0.664165m、7500 0.668037m、10000 0.713394m。复核数值与坐标可比性后可说当前参考几何描述更差；不可称为实际world TCP误差或证明训练发散/灾难遗忘。非拒绝集合的ranking统计继续独立。

7条 decision=WARN 原样列出，但不选择7500为候选，也不把它们加入固定诊断cohort。逐tick频谱/尖峰/专家误差如果没有tensor，保留未观测；G4新tensor解决新诊断的问题，不冒称恢复历史数据。

完成证据：`g1_failure_decomposition.json/csv`、co-occurrence、paired deltas、missingness与observation coverage。无新增模型forward。

## 4. G2 — 专家起点旋转 casebook 与数据可学性

### 4.1 精确复核12条，不先“修好”

导出12条CALIB rejects及其split/episode/rollout/目标/jitter/state realization/phase/offset、原state rot6d、action[0] rot6d、两者矩阵/quaternion、原gate observed/threshold和文件hash。包含同rollout原时间相邻可用窗口和元数据确定的同phase非拒绝对照；这是标明 outcome-selected 的归因casebook，不是新的performance cohort。

至少关联以下链：原 source observation/state → desired reference → synchronizer indices → materialized parquet → provider输出quaternion → gate角度。区分 state 是 achieved pose 还是 desired reference，action是 persistent reference 的哪一采样点。证明首点差异在何处首次出现。

检验互斥/可并存假设：
- 来源中确有 measured-vs-reference 姿态偏差；
- state/action 索引或观测对齐错误；
- row/column、四元数 xyzw/wxyz、base/world/reference frame被误用；
- 已经发生转换又被重复转换；
- 冻结 gate 对真实专家窗口的连续性要求与标签语义不相容。

用合成非单位旋转和独立矩阵比较验证假设，不得枚举所有转换后挑使gate通过的版本当作真值。纯 `q` 与 `-q` 等价应产生相同角度，不能用符号翻转解释旋转差异。

做只读clock-origin对照：统一平移timestamp、observation timestamp和now但保持indices/poses/dt，速度和起点旋转不应改变。若改变，先查诊断实现。索引错位可做标注的离线反事实对比，但不写回数据或进入训练。

### 4.2 检查整个TRAIN标签，而不以64条代替训练集

授权对9792 TRAIN及1088 CALIB的全部已封存专家H40窗口做CPU只读gate统计，总unique window上限10880，已有可验证P2记录可复用。64TRAIN全部非拒绝不能推出其余9728条都非拒绝。每窗口同时记录原rot6d退化/正交性、首点角度、速度、frame/validity标识；不能因WARN多就自动解释为错误。

对真实窗口采样的 t0、内部起点、末端与padding/valid mask做有界抽检：从原64TRAIN中每目标/phase取metadata-hash最小窗口（16个），每个检查实际loader合法的首/中/末起点，最多48个额外CPU样本；若实际loader仅允许唯一完整H40起点，就记录不适用，不人为扩成新训练样本。检查相同RGB+state是否因采样产生不同label/尾部padding，是否存在真实有效样本数与materialized episode数混淆。不能把shape=40就当训练采样语义已证明。

不重新生成/重seal数据，不删除12条，不向模型加入phaseID/targetXYZ。若需要标签或采样修复，输出最小diff提案、影响范围和数据v2迁移计划；实际应用等待G5后的PI决策。

完成证据：`g2_rotation_casebook.*`、`g2_full_label_gate_census.*`、`g2_source_window_alignment.*`；归因可以是UNKNOWN，但必须列出已经被排除的具体机制。已知专家REJECT不再单独阻止G4纯诊断；state输入自身或标签坐标含义不可确定时仍禁止解释相应物理指标。

## 5. G3 — 完整训练/推理parity与数值裁定

### 5.1 必须新增的独立路径

在 pinned `51d4c89f72fda44cbf77285c6a8114b52676b8a1` GR00T source 与真实PI0228训练resolved config下，以只读方式构造实际dataset/step sampler和processor/collator；不得调用生成stats写入sealed目录的qualification函数、experiment.run、Trainer.train、AutoModel权重加载或backward。

对固定128cohort，加casebook中12reject/2excess对应窗口的并集（最多142unique，如已在cohort则不重复），采集真实训练路径处理后的state、action、mask、embodiment数值ID和全部active universal slots。必须实际观察 `[B,40,132]` 或 pinned runtime 真正输出的结构；结构与协议不符则报告，不靠手工padding成132伪装。将真实训练target tensor按实际映射交给独立的PI0236 evaluator processor/decode，比较物理专家标签。

分别采集：训练实例的resolved kwargs与统计量、eval实例的resolved kwargs与统计量、相同观测在训练评估模式及部署模式的processed inputs差异、语言/两摄像头通道/resize/crop/颜色约定、训练augmentation（仅观测既有配置，不更改）。action labels不进入模型观测侧。输入parity的合理差异应有配置来源，不要求training随机augmentation与eval图像逐bit相同。

验证active slots、loss mask有效count、padding horizon、state timestep anchor和SO(3)组合顺序；新增故意swap slots、wrong mask、wrong anchor的negative controls，保证测量会抓错。只读读取Trainer logs、resolved config、optimizer param group名、checkpoint tensors可用的更新证据；CPU阶段不实例化GR00T/VLM网络，也不创建optimizer。需要昂贵权重比较时只读选择已明确的tensor/shards，超预算则在G5提案，不能遍历整个HF archive。

完整parity证据不能仍来自同一processor的apply/decode自循环。旧10维loop可作为E0，新的actual-batch cross-path为E1；二者并列保存。

### 5.2 两条normalized超差的数值归因

先从已核验P3表定位两条记录及具体dimension/tick，而不推断它们一定与12条旋转reject重合。观察active dimension的实际normalization公式/分母、zero-range保护、cast次序、值域和中间张量。

在纯CPU sidecar用相同输入做float64参考/float32实际cast控制，记录ULP、绝对/相对误差与条件数。若实际公式是仿射归一化，可用 `delta_z = scale * delta_a` 分析小物理误差的放大；实际scale必须来自真实实现，不假设一定为q01/q99。

instrument实际clip/mask/quantize步骤或以其源代码定义的条件独立计数；不改变生产processor。至少报告clip predicate命中数、裁剪前后差异、每维分母范围与原gate决定是否改变。`not separately exposed`不能永久代替可测量的因果诊断。

旧float32数值容限保持记录：XYZ1e-5m、SO(3)1e-4rad、gripper1e-5、normalized1e-5；旧2条excess仍为excess。新的诊断准入可接受 `NORMALIZED_TOLERANCE_EXCESS_EXPLAINED_DIAGNOSTIC_ONLY`，但仅当：
1. 真实训练与eval路径的配置/mapping/state anchor一致，无未解释的语义差异；
2. physical XYZ/SO(3)/gripper全部符合上述既定容限；
3. 没有未解释的实际有损clip，或其影响已明确且不被当作无损；任何会改变专家gate决定的codec损伤必须回PI；
4. 超差被与真实运算绑定的roundoff/scale分析解释，误差界在新增全量检查前封存并在其它窗口验证，不把容限直接设成观察最大值；
5. 多种case与synthetic negative control表明该检查能区分真实mapping错误和数值误差。

如这些条件不满足，输出 `FULL_PATH_MISMATCH` / `NUMERIC_EFFECT_UNRESOLVED`，停止G4并提交G5修复提案。不可凭“1.65e-5很小”自行放行。

### 5.3 G4诊断准入清单

必须有完整数据/权重/配置身份、实际132/mask路径核验、真实输入构造、已冻结cohort与模型顺序、可靠数值解释、可区分的expert-invalid strata、访问边界和预算。G2专家原始REJECT可以保留并分层，不要求修成全PASS；疑似label语义不一致但模型输入正确时，可只报告与该歧义无关的速度/归一化数值指标，不虚构专家物理误差。存在模型输入state/camera/统计量身份歧义时不准G4。

完成证据：`g3_actual_batch_trace.*`、`g3_effective_train_eval_config.*`、`g3_132_slot_mask_tests.*`、`g3_numerical_casebook.*`、`g4_diagnostic_admission.json`。无需因每项PASS再问一次授权。

## 6. G4 — 固定cohort的新诊断推理（本次条件GO）

模型：baseline-1000与fresh-10000（仅623593 family），不加载7500挑好结果。窗口：PI0237固定64TRAIN+64CALIB，不删除专家reject，不替换样本，不追加target/jitter。仍用PI0236 as-evaluated processor；baseline原生processor身份缺失时限制结论，不擅自切换。

最多256 attempted `get_action` policy queries，内部denoiser calls与policy query分别记账。顺序：baseline CALIB episode升序→baseline TRAIN→fresh CALIB→fresh TRAIN。每窗口一次，首条就是预算内canary，不加warmup/seed sweep。旧tensor可复用仅限实存、全部输入/配置身份可核验；要作同run的TRAIN-CALIB对照时明确哪些是复用与新测量，不将混合来源伪装成单一执行。

`eval()` + `inference_mode()`，不建optimizer，不调用teacher-forced网络训练forward。模型输入只用原body/wrist/state与固定语言；专家动作、targetXYZ/ID、phase、split、source ID不能进入模型。raw `[1,40,132]`、decoded `[1,40,10]`、state、expert、mask、input hashes和实际slot信息都封存。

报告：逐点XYZ L2/整窗RMSE/mean/p95/max/首点/有效endpoint，SO(3) geodesic/rot6d退化，gripper误差，预测与专家的逐段速度/角速度/峰值与超限segment数；valid-only与full-H40分开。原gate全量记录不先筛好样本。专家有/无REJECT两组并列，不能只报可通过组。target min distance只是副指标，不代替专家拟合，不要求每个中途窗口命中目标。

同窗口baseline/fresh配对；TRAIN vs CALIB仅分层可比，不是假配对。报告source-rollout宏平均及其sample count，不能将128窗口等同128独立rollouts；输出phase/jitter支持范围，不能将此cohort当全体泛化证明。

如新raw/decoded SHA不同于历史值：先核验实际input/model/config身份；身份相同则按1.1的新诊断身份规则标记，不重试追哈希、不中止仅因非bitwise。保留新结果与旧record并分开科学归属。非有限/错误shape/身份不符/目标注入/预算超限立即停止。

算力：复用站点已验证PBS写法，项目CFP03-CF-004，1×H200/12CPU/225GB/单作业walltime上限1h；不猜queue、不换平台。默认1次提交；仅在第一次attempt的get_action=0且明确为工程错误时，允许最小修复+exact-env测试后1次替代提交（最大2次submission、最多2h申请GPU时间、任何时刻只1活动job）。已进入推理后的job异常不自动再提交；封存已完成诊断并报告缺失，不凑满256或挑重复结果。正常REJECT不是工程失败。

CPU侧优先现有合法环境，必要时单个CPU-only作业2CPU/16GB/walltime≤2h；只允许1次零科学读取/输出前的机械失败替代提交，不在login node跑重负载。CPU预算不足先给已有证据与明确缺项。工程修复不改源数据/runtime科学配置；测试/日志/IO补丁可在此边界自主处理。

输出：逐query append-only attempt ledger、NPZ、input fingerprints、原gate、诊断CSV/JSON、PBS终态、quota与read-boundary日志。get_action在入口前记attempted，失败也占预算。所有训练/backward/optimizer/WBC/sim/DEV/F计数为0且有代码路径支持。

## 7. G5 — 近期完成定义与下一科学决策包

完成G0–G4（或明确机械/语义blocker）后，不只回复“等待审批”。提交能决定下一轮的单个package：

- 审核correction表：旧结论、真实覆盖、新证据、未覆盖；保留PI0236/0237原文和hash。
- G1专家/模型错误分离，G2起点旋转来源，G3真实path/numeric结论，G4拟合结果与覆盖限制。
- failure分类允许共存：`REFERENCE_STATE_DISCONTINUITY`、`DATA_OR_SAMPLER_SEMANTICS_MISMATCH`、`TRAIN_EVAL_MAPPING_MISMATCH`、`NUMERICAL_EXCESS_EXPLAINED`、`TRAIN_FIT_DEFICIT_EVIDENCE`、`TRAIN_CALIB_GAP_ON_FIXED_COHORT`、`INCONCLUSIVE`。不能只依据loss下降认定动作拟合，不把snapshot问题一概归WBC。
- 只推荐一条主线和一个备选；每条给原因、最小变化、哪些结论会被证伪、预算、统计量来源、精确输入与版本、可执行命令、输出计划、停止条件及是否需重新评估旧family。
- 需要改变label/frame/ABI/processor/mask/训练recipe时，提供可审的patch和CPU tests，但未获后续PI科学GO不应用到训练/evaluator生产线。

本次最后token：`STOP_FOR_PI_RECIPE_DECISION`，`selected_checkpoint=NONE`。若G4没准入，报告精确原因而不是泛称“读过但没有授权”；本文件已授权满足条件的G4。

## 8. G6 — 从诊断到训练：后续PI批准的具体路线

以下为长程目标及下一份run card的必填内容，不是本次训练许可。训练前只保留一次完整recipe审核，批准后同卡片内常规阶段可自主连续推进，不为每个smoke/保存点重复申请。

### 8.1 有语义问题时，先选对“修哪里”

- 仅评估路径错：修复提案与独立阳性/负性测试→新eval身份→同一checkpoint family有界probe→必要时全量5 profiles；不因评估bug重训。旧结果保留为旧pipeline结论。
- 训练loader、标签或物理语义错：创建dataset/processor v2，明确受影响样本/变更前后清单，source-rollout split不变，TRAIN stats仅TRAIN拟合；新权重lineage从预注册initializer开始。不能覆盖sealed v1、删除难例来制造通过或把新坐标contract仍称同ABI。
- 专家本身与现有gate不相容：先决定是修采集/参考连续性还是科学修订评价。不能通过数值容限修复1.5rad级跳变；12例不修成新训练标签前不能无条件宣称controller-compatible dataset。

### 8.2 TRAIN拟合不足时的16-window自检（提案默认预算）

固定样本来自原64TRAIN：每A–D×phase层按source ID的PI0238固定salt hash取1条，共16条；不得依据已观察loss/gate误差挑选。若任一label语义无效，先解决数据问题，不能替换成容易样本。

建议单次1000 optimizer updates上限、保存250/500/1000；使用批准的大规模训练实际trainable modules/loss/LR/precision与TRAIN统计量，不同时改容量/损失/LR；initializer默认原PI0226 checkpoint-1000而非事后挑选的fresh7500。该默认必须在新PI run card中批准。batch/accum/optimizer/scheduler/warmup/seed等全部填实数并绑定实际resolved config，不在agent实现时隐式重置。

step0/250/500/1000对固定16TRAIN做真实free-running action queries，64个诊断queries预算；TF loss仅辅助。必须验证正确embodiment/active modules的gradient与参数变化、mask有效比例、实际sample exposure。通过标准在看新结果前由PI结合G4物理误差和原gate固定；卡片缺标准则不得开训。这个实验只证明学习实现具备记忆能力，不证明泛化，也不作为最终候选。

失败分支：参数没更新→训练实现定位；loss降而free-running轨迹乱→模型目标/采样/训练推理关联定位；16窗可拟合而大TRAIN失败→优化暴露/多模态冲突/容量或数据配比的单变量pilot。不能每次失败都把steps翻倍。

### 8.3 TRAIN好但CALIB差时

先控制phase/target/jitter/state分布与source-rollout支持，再做覆盖或regularization的单变量pilot。输入RGB包含target证据是否真正被利用须独立设计对照；不准直接注入targetXYZ。数据变大但有效state/trajectory支持不变，不等于新增泛化证据。DAgger仅在已经得到可执行的on-policy rollout且专家纠正来源合法时讨论，当前不得启动。

### 8.4 大规模large-adapt-v2训练

前置：实际数据/loader/codec gate兼容性闭合；小规模学习可行性支持；一份完整科学run card获批。

默认提案上限仍为10k optimizer steps，full Trainer checkpoints 2500/5000/7500/10000；不是现在让PC agent直接再跑10k。明确initializer、weights/optimizer/scheduler/RNG/dataloader ordering、sample exposure、active parameters、loss、LR、seed、数据比例、checkpoint和archive保留策略。改变旧recipe的项目仅限已经获批的单一归因干预，避免一次改多个因素失去解释能力。

新增**固定TRAIN mini-screen**：在run card预注册的早期step进行free-running物理动作检查，不只是import/runtime smoke或平均loss。行ID、预算、指标和abort规则先冻结；不使用CALIB/DEV来在线调LR或挑seed。发现与当前相同的近全量高速违规时，按预注册规则早停，而不是坚持烧完10k再发现全reject。

早停是新实验的合法终态，若family尚未产生四个checkpoint，不声称完整selection；不得为了凑family在未知原因下强行续训。

存储按实测full checkpoint字节和最大同时占用计算：未来保留checkpoints+当前原子写入临时文件+archive staging+optimizer/trainer payload+runtime缓存+evidence+安全余量；不是只看最终四份总和。文件系统quota与inode至少满足峰值加max(20GiB,20%预计新增峰值)余量。保留必要旧证据；需要删除/迁移权重时列exact path、bytes、archive hash、恢复验证和用户/PI清理授权，不盲删共享cache。

允许exact resume仅在checkpoint完整、loader顺序/RNG/optimizer/scheduler/config条件成立时；否则明确new-start lineage，不声称科学等价。保存完整data/model/runtime provenance。

## 9. G7 — 完整selection与真正的one-chunk sim

### 9.1 单一checkpoint freeze

所有预注册full checkpoints封存后，在统一冻结TRAIN-CALIB协议下评估并选唯一模型；原v1 pipeline不变时沿用原selector。若数据/表示/评价必须v2，先新PI协议，不修改旧selector让v1过关。保存每row raw/decoded或可验证tensor证据，避免再次只存SHA导致关键归因缺失。

正常REJECT是负结果并继续收集；全部候选不合格就STOP，不挑least-bad，不按A–D分别选权重。训练完成、selection完成、任务成功三个status分开。

### 9.2 frozen-WBC A–D C0

需要新的PI WBC release，绑定唯一checkpoint/runtime、具体existing runner、seed、起始状态、原target/scorer和action-provider版本。复用PI0226已证明能实际传递模型轨迹的成熟链，不重新建设GR00T/WBC接口。

四个cell均执行和保存，不删失败cell。每cell记录真实RGB/StateV1、模型query→原gate→WBC input→native trace→achieved world-TCP scorer，reference与achieved分开。target坐标仅用于场景/专家/评分，不进入模型生成旁路。拒绝轨迹不执行；runtime fallback/hold明确计数并不得算成功。

使用原0.035m任务定义及其既定hit/termination语义，不用碰巧一帧经过或reference min距离代替。controller gains、WBC、threshold/ABI/cadence/retiming不改。成功证明同shared模型具备明确范围的物理任务能力，不证明多技能。

失败时按reference几何、reference速度、tracking、起点/状态和scorer对齐拆解；不自动换模型、target或seed。完整证据回PI决定下一阶段。

## 10. G8 — progress-preserving receding closed-loop

需要one-chunk review和具体cadence/horizon/tick预算的PI release。提前CPU静态准备入口/trace，当前不启动sim。

同一continuous simulation保持base/joints/controller memory/reference progress。query时抓取新的RGB与StateV1，用同一冻结模型做新预测，再按已批准progress/time continuation更新reference；不能每H40 reset s/t、用旧观测、缓存预测重复播放或专家动作替代。

每次更新封存因果链：前次WBC执行→新achieved state/新RGB hashes→新模型输出→gate→reference更新→后续物理轨迹。记录new-query counts、old-reference carryover、gate reject/fallback、停止/碰撞/invalid-contact、native termination、min/endpoint achieved target distance与first-hit。逻辑闭环与实时硬件可运行性分开；若推理时暂停sim，如实报告。

在批准的完整tick/chunk预算内复现A–D shared policy。若后续加入扰动与recovery，先冻结TRAIN-only cohort和专家采集机制；DAgger新数据不含DEV/F，不重用CALIB做训练。必须有baseline/no-update对照才能归因闭环feedback收益，具体增加的runs需预注册，不能在看到结果后挑成功control。

完成标准：完整persistent多次真实query/update与原任务success，并有failures/coverage全量记录；不是单段视频或25次调用日志本身。

## 11. G9 — 泛化、DEV与真正多任务路线

TRAIN覆盖与未见条件泛化分开；source rollouts、目标/jitter/state组合的独立性及支持范围明示。当前CALIB已被分析用于研发，不能作为未接触最终盲测；后续独立验证方案需要新协议，且不擅自解锁Final-F。

DEV-E只在checkpoint/runtime freeze且TRAIN meaningful success后，按新PI授权一次性运行。DEV失败不允许当前line就地调参再把同一DEV称盲测。Final-F与硬件继续锁定，直到明确最终授权。

真正多任务需要可区分的任务/语言目标、对应expert与物理环境、shared checkpoint选择规则和每task独立成功定义。A–D reaching只是多目标同技能。接触/抓取等任务若需要改变action semantics、WBC或benchmarks，独立charter；不把这份reaching闭环计划自动扩大。

## 12. 自主执行、交付与启动prompt

工程自主修复：只改诊断层的IO/path/import binding/logging/testing，先保留失败attempt，再最小修复、exact-env CPU测试、绑定新script identity、按本预算继续。不能移除失败断言、换样、放宽阈值、升级runtime、改数据/模型语义来继续。G4一旦开始的机械异常按§6停止；其它CPU小修不必逐项申请。活动作业的staged依赖只读，修复在独立新目录。

轻量execution-state sidecar记录protocol commit、当前goal、input/output hash、active job、attempted/completed/reused query、已知blocker、下一动作。恢复先查真实状态，复用完成结果，不重跑G0/G1全套或重提交已排队job。Issue #11为唯一科学账本，不新建并行ledger。

交付至少：审核correction表、G0访问/身份/tests、G1统计、G2 casebook/全量label census、G3 actual-batch trace/数值归因、条件完成的G4 tensors/fit/终态、G5具体run card和建议主线；`phase_status`、`scientific_status`、`missingness`、`selected_checkpoint=NONE`与完整计数。未执行阶段记NOT_RUN+原因，不造产物。

小报告/JSON/CSV/tests/scripts推新execution分支；tensor留授权存储，Git只存明确locator/bytes/hash和小样本展示。不要将token、私有download URL、auth文件、权重或大视频推到Git。先Issue #11 canonical closure/blocker，再PR #77 pointer，不merge。

### PC启动prompt（用户发送此段即为G0–G5范围的执行请求）

> 现在执行PI0238。完整读取本文件的固定发布commit、Issue #11本轮canonical、PI0237结果5747055582，以及实际源代码，不只看摘要。先接手真实job/artifact状态，保护998eba5与旧cohort/数据/权重。连续完成G0工程审计补全、G1历史错误分离、G2专家12例旋转casebook及全量TRAIN标签检查、G3真实训练batch/132槽位/mask到eval-decode的parity和数值归因。G0–G3零模型load/forward。按本协议新的诊断准入，满足条件后自主运行G4固定64TRAIN+64CALIB、baseline1000/fresh10000、最多256attempted queries；不要求专家全量无REJECT来做非执行诊断，不因同配置新输出不逐bit等于旧SHA而伪造复现或重复试跑。所有已知异常保留并分层。工程问题在预算与语义边界内自主最小修复，不重复询问已授权步骤。交付G5完整归因与下一训练/修复的精确run card后STOP_FOR_PI_RECIPE_DECISION。G6–G9是长期路线，未获新的PI GO不训练、不WBC/sim、不DAgger、不DEV/F、不硬件，不改gate/selector/threshold/ABI/representation/target输入或PR merge。Issue #11先记账，PR #77只pointer；不要仅回复已读等待确认。

### 外部背景与本项目事实区分

PyTorch官方数值精度说明：[Numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)。有限精度与不同执行路径不保证逐bit相同，这是数值解释背景，不证明本项目两条超差一定来自舍入；具体结论必须来自G3控制。该文档不是更换pinned软件、修改precision或抹掉旧epsilon失败的授权。
