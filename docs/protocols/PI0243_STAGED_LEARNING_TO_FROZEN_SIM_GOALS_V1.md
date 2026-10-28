# PI0243 — 分级学习到 frozen-WBC sim 闭环的自主 Goal

日期：2026-09-21（Asia/Singapore）  
决定：`GO_GATED_16_TO_64_TO_FULL_TO_SIM`  
授权 token：`PI0243_STAGED_LEARNING_TO_SIM_GO`  
文档发布状态：`AUTHORIZED_NOT_STARTED`  
审核基线：`3ea3614444ddb76ebf1abe91be31744f0a6e6ded`  
仓库：`Leo-6-maker/go2-arx5-hierarchical-vla`  
Issue #11 canonical；PR #77 仅 pointer，不 merge。

## 0. 这次明确放行什么

用户要求的是长程推进并跑通sim闭环，不是再做一轮泛泛归因。本PI接受PI0242-R1的原生一致性闭合，授权一条**有条件、单路线、分级支持规模**的学习路径：

`准备 → M16-5K → M64-5K → FULL-10K → 五份完整CALIB profile/唯一selection → A–D C0 one-chunk → progress-preserving receding → TRAIN_SIM_CLOSED_LOOP_COMPLETE`。

首个科学干预是：在原16个TRAIN窗口、相同初始化和native objective下，将训练预算及cosine总时长从1000增至5000；不同时调整loss、peak LR、模型容量、精度、noise schedule或动作表示。后续M64和FULL是预先声明的课程式支持扩展，只有前一级达到下文标准才自动启动，不是三个并行备选arm。

**阶段PASS后无需再等PI逐项确认。工程错误自主修复，不设固定重试、qsub、token或累计工程时间上限。** 但一个已有效完成阶段的科学FAIL必须停止，不加steps、换seed或改recipe追PASS。资源/权限确实不可取得时报告真实外部blocker，不把它称科学失败。

本文件取代旧PI0241/0242“本轮零训练”的限制，但仅限这里列明的三个训练阶段及条件sim。它不开放无界训练、备选loss/LR/精度、DAgger、DEV-E/F、Final-F、硬件、生产部署或PR merge。旧实验结果保持不变。

成功终点定义为**同一冻结模型/processor/runtime在既有TRAIN A–D C0任务上具有真实WBC one-chunk与receding任务成功证据**。这不是宣称multi-skill general model，也不是独立泛化或实时硬件验证。

## 1. 审核依据：现在有理由测试预算，不等于预算一定能解决

必读且复用：

- PI0242-R1 canonical [5755602153](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5755602153)、PR pointer5755603048；审核基线的 `reports/PI0242_R1_NATIVE_FLOW_PARITY_CLOSURE.md`。
- PI0241 canonical5754335876、固定16/schedule/run card、`scripts/pi0241_fixed16_learning.py`、其恢复及reconciliation证据；不要重跑原1000步。
- 原PI0240 G3/G4和资源协议、PI0239的原生one-chunk/receding契约；只沿相关调用链读取，不重新扫描全库。
- `scripts/pi0228_h200_large_adapt_v1.py`、实际Stage28 config builder、pinned GR00T训练入口。
- `scripts/pi0228_train_calib_checkpoint_selection.py` 的provider/evaluate/select；实际采用的native sim入口与已批准参数从canonical运行清单解析。

R1最新canonical记录的有效矩阵是625607/hopper-04 H100，science runner `29ec76af09437c2a4f7bd36e88223bbebedcf8dd`，后处理CPU finalize单独记账。原始root为 `/scratch/e1192548/pi0242_r1_native_parity_20260921_v3`。关键SHA256：

- run card `1e3df7dbe72ca003e7d8709c4d46537cc8188c1584753f7209aad67941224a62`
- closure `d2038b6b6df2d48f095e802bdbc05a6f18d5b9f703eced23f4315c2d4e01bff1`
- corrected direct records `dce079966f7f91b67c85bbcd73bbe57bb59fb2fc1c97916699f569b84e7cdabd`
- parity records `ce6ca46222a1ea2dc49bc492d3f5b6dc1e3b9af57f7c4db361ddf08d0480a9f7`

已支持的结论：32/32 native/helper exact；旧F(F(h))偏差已纠正且效应可测；修正后final1000在bridge4上的local XYZ重构仍约0.23–0.41m、SO(3)约0.16–0.17rad；stock仍16/16REJECT。此前模型相对initializer已显著数值改善，但尚不充分。这里“改善”是描述，不宣称显著性检验。

据此测试更充分暴露是一个合理的**下一假设**，不是已确定根因。改变cosine总时长会改变之前各step的LR，因此M16-5K是“预算和schedule horizon干预”，不是严格保持前1000步不变的额外4000步，也不是PI0241 exact resume。

C_T/C_I只有较小的已测pixel处理效应。本路线明确以**实际训练processor C_T**作为新冻结训练/评估部署身份；不声称旧C_I为bug，不与旧C_I分数混为同一实验。该选择和5K预算都事前声明，不能声称新旧差异完全由步数单因素导致。

GPT本次审核了GitHub源码、报告和ledger，未SSH、未加载远端模型、未验证大文件实际hash、未运行训练/仿真。执行agent核验并引用既有封存证据即可，不把这份计划写成已经完成的实验。

## 2. 不变量与真正的训练配方

所有阶段继承同一pinned GR00T source `51d4c89f72fda44cbf77285c6a8114b52676b8a1` 和实际已验收运行时；工程更新只有必要的薄stage/sampler/输出绑定，不能引入新版模型实现。

| 字段 | 冻结值或唯一来源 |
|---|---|
| 动作/状态 | official dense-relative H40；StateV1=10；universal132，active0:10；原mask/row-rot6d/frame/TCP |
| 时间 | dt=0.05s，40点原provider时间网格，最后点1.95s；不重定时 |
| 语言 | `move the end effector to the red target` |
| 输入边界 | 两路原RGB、原state、语言及模型原有非特权字段；target XYZ/source ID/phase/expert未来不进入conditioning |
| loss | 原native masked flow-matching objective及原分母；不加velocity/smoothing/target loss，不重加权XYZ，不处理inactive122噪声 |
| trainable | projector + diffusion/action head；visual/LLM/VLLN冻结；保持实际模型原有dropout配置 |
| optimizer | `adamw_torch`，peak LR1e-4，weight_decay1e-5，max_grad_norm1.0；betas/epsilon/decay groups沿PI0241实际resolved来源并写成数值 |
| batch | 1 GPU；batch1、gradient accumulation1、workers0 |
| schedule | 官方cosine，warmup_steps10、warmup_ratio0；max_steps为该阶段5000/5000/10000；每阶段新optimizer/scheduler |
| precision | bf16=True、fp16=False、tf32=True、load_bf16=True、backbone_trainable_params_fp32=False、gradient_checkpointing=False；不得以memory workaround改变 |
| processor | C_T真实pipeline与最终生效kwargs；原full-TRAIN stats/relative_stats；state dropout0，原关闭的颜色/旋转augmentation保持，shortest_edge96/crop_fraction1等按实际已验证来源绑定 |
| RNG | train/data seed22301；官方实际Python/NumPy/torch/CUDA种子和flow随机流记录；不每step重播种 |
| inference | 原stock四步sampler、原noise/time schedule、每固定screen cell seed22301；不扫步数/seed |

R1修好的raw/processed-once特征与角速度公式必须保留；不要在新screen或direct sidecar重新引入可变别名。原native loss已按有效mask求均值，不将122个padding维误称loss稀释问题。

实际模型/optimizer参数清单、dtype和membership是证据；配置布尔值、前4096元素抽样或常数字段不是。记录有效embodiment行的变化，并对冻结组做内容核验；不能重用已被推翻的projector假阴性。

## 3. 准备 Goal：一份stage plan，先把后续参数绑定好

### 3.1 当前状态与内容身份

先核对最新canonical是否有superseding决策及当前活动job；同一科学stage最多一个active owner。使用独立execution worktree/分支，不reset用户脏工作区或修改在跑作业的staged代码。

直接复用原16TRAIN、原64TRAIN、full TRAIN9792窗口、CALIB1088窗口及其split/seal/target authority。原16是原64子集。核对实际内容与原规范一致，不能靠目录名或shape成立。数据根可以通过已验收的host binding迁移，不改seal本体或source IDs。

已知专家census10880里存在160个REJECT，其中12在CALIB；不得删掉这些行、改首点或放松gate来制造全通过，也不以已知专家有少量REJECT为由禁止所有训练。它们意味着“完美模仿所有标签”与“每条gate合格”不能自动等同。保留标签和全量评估，策略能否通过必须由真实输出决定；如需修订标签/gate，回PI另立数据/科学版本。

### 3.2 三阶段曝光表事前固定

每阶段只调用既有episode loader的原window observation index0和对应H40，不在window内部随机换起点。这是明确的window-addressed课程方案，不冒称等价于原PI0228全部内部采样方式。

- M16：原PI0241原16IDs，canonical顺序(a→d,phase0→3)。
- M64：原PI0237/0240的全部64TRAIN，按(target_family, phase, source_episode_id)排序。
- FULL：sealed TRAIN全部9792windows，按original episode_index、source_episode_id排序。

对每阶段独立建立 `numpy.random.Generator(PCG64(22301))`，每轮排列全部N条，拼接到该阶段max_steps。生成和封存完整schedule JSON/hash，不消耗模型RNG；实际DataLoader不得叠加第二层shuffle。M16每窗312/313次，M64每窗78/79次；FULL一遍9792加下一遍208条，前一级权重已作为warm start。不能把一次FULL低于充分覆盖的风险隐藏为训练保证。

通过薄Dataset/Sampler调用既有官方Trainer；记录“yielded”和“真正送入training_step并关联optimizer update”的区别。可以在wrapper侧携带source元数据，但必须在model输入边界剥离，禁止将episode/phase当模型特征。恢复阶段也追加真实消费日志，不再禁写后直接填常数总数。

### 3.3 sim运行清单提前准备，不能到最后临时改任务

只读解析PI0226及后续已批准的dense-relative native one-chunk/receding入口、A–D/C0/seed、原成功scorer、cadence/prefix/tick cap/termination/clock-progress规则。把源码SHA、artifact authority与实际命令逐字段填入 `sim_binding.json`，不是重建接口。

本计划没有资料支持的具体cadence、tick cap不编造数值：必须机械继承已批准的canonical值，并引用其来源。合法查找/路径修复agent自主完成；若存在两个科学含义不同的版本或从未批准的必要值，记 `SIM_SCIENTIFIC_BINDING_REQUIRED`，训练可按已批准阶段继续，但sim不得自选值启动。禁止引用历史Stage40 StructuredAction替换当前dense-relative ABI。

### 3.4 提交前测试

本地CPU测试覆盖stage转移、采样顺序、hash/路径、数据访问边界、真实loss/mask、classifier、恢复cursor、原生快照不被二次处理、存储与日志失败封存。Corelab做真实固定样本video→processor→collator→decoder→写入/重读。

基于已完成G3/R1，只验证新stage/view/config边界，不重跑旧142/10880/5440/256诊断。首个正式optimizer update兼作真实训练集成并计入该阶段，不另外混入调试updates；核心逻辑问题必须在本地/Corelab回归，不拿H200当pytest机器。

保存一份 `pi0243_plan.json`，其中包括本protocol hash、stage配置、schedule、stats/模型/代码身份、资源、科学预算、验收条件、sim_binding和阶段恢复索引。准备通过即启动M16，不再询问“训练是否授权”。

## 4. Stage A — M16-5K：一次5000步预算检验

**立即授权，前置为准备完成。**

- 从同一PI0226 adapted baseline checkpoint1000权重初始化；不是通用BASE，不从失败的PI0241-final1000续训。
- 新optimizer/scheduler/RNG；max_steps5000。相同16窗口、native loss、peakLR、precision、模型范围，按上表执行。
- 保存full Trainer checkpoints1000/2500/5000；screen在0/1000/2500/5000，原16窗全量，最多64个唯一policy screen cells。
- 主判定是step5000，不选中途最优。早期未达最终标准不提前叫科学失败；真实非法训练、NaN/非有限梯度、已确认的错误目标/无有效更新才可早停。
- 使用C_T训练和C_T的label-free推理构造；新processor/runtime身份封存，不能默认复用旧C_I step0。只有weights/conditions/runtime/noise完全一致且原tensor可验时才复用已有cells。
- 独立screen进程读保存后的weights，不污染训练RNG或在训练回调里reset推理seed。训练完成后依序screen也允许；不可更改训练schedule。

定义每窗XYZ_RMSE=sqrt(mean_t(||pred_xyz-expert_xyz||²))，SO3_RMSE=sqrt(mean_t(geodesic(Rpred,Rexpert)²))。全部40点有效为原契约，真实padding/mask冲突先报告不偷偷删点。跨窗口percentile用NumPy linear。

**A_PASS必须同时：**step5000的16/16 rows完整、身份/更新/曝光审计有效；原gate REJECT=0；median XYZ_RMSE≤.020m；p95 XYZ_RMSE≤.035m；median SO3_RMSE≤.050rad。WARN保留但不新设排除。gripper、首尾误差、max速度、最差window、target距离描述均侧报；target距离不替代专家RMSE。

为失败时不再回到泛泛归因，本阶段另授权32个唯一direct-flow记录：step0和5000 × 原R1 bridge4 × 原t0/q10/q50/q90 × C_T。使用R1已闭合的native/helper单次特征路径，记录actual head/backbone调用；不加paired种子/条件。它们仅诊断，不参与A_PASS或选模。原条件完全相同可复用对应R1数据，不能把跨runtime值强行作逐bit配对。

A_PASS封存后自动启动B；有效最终FAIL则 `SCIENCE_FAIL_M16_BUDGET — STOP_FOR_PI_SINGLE_RECIPE_DECISION`。不额外开10000步M16、不改loss、dtype、dropout或LR。

## 5. Stage B — M64-5K：支持扩展与记忆保持

**仅A_PASS后自动授权。**

- 初始化为A-final5000的weights；新AdamW/scheduler，max_steps5000。这是课程下一阶段，不声称exact resume。
- 用原64TRAIN完整固定support和stage B schedule，其他配方同第2节；不读取新CALIB结果，不按A成绩挑64条。
- full checkpoints1000/2500/5000；screens0/1000/2500/5000×64，共256个唯一cells。A/B共同16若实际身份相同可复用边界结果；不得仅因权重相同就忽略processor/device差异。
- step5000必须满足A同四项质量标准，统计范围为64窗，且旧16子集仍满足A四项以检查遗忘；全部身份/曝光/权重审计有效。各A–D/phase侧报，不把相关窗口当独立rollout夸大证据。

B_PASS只说明固定64支持上的可生成性，不证明泛化；自动进入C。B有效FAIL则 `SCIENCE_FAIL_SUPPORT_EXPANSION — STOP_FOR_PI_SINGLE_RECIPE_DECISION`，不得用A memorizer跳过B送WBC。

## 6. Stage C — FULL-10K：全量controller数据适配

**仅B_PASS后自动授权。**

- 从B-final5000 weights初始化，重建optimizer/scheduler；FULL max_steps10000。
- 数据为原sealed9792TRAIN windows；stats不重拟合；不碰CALIB标签优化、不DAgger。
- 保存full checkpoints2500/5000/7500/10000，四个均归属新的 `PI0243_FULL_V1` family；禁止混入旧623593/M16/M64权重。
- 保持旧S4所需的step family，避免改selector。旧baseline1000不是课程候选，只作新C_T评价下的比较。
- 固定64TRAIN screens在stage C的0/1000/2500/5000/7500/10000，共384个唯一cells。1000只需冻结的model-only临时screen快照，不称full resume checkpoint；归档身份后可受控回收临时副本。
- 早期失稳预注册：如果step1000和2500两次完整64TRAIN screen均64/64出现原 `LINEAR_SPEED_REJECT`，记 `SCIENCE_FAIL_FULL_TRAIN_COLLAPSE`，在2500后停止。单个早期不合格window或一时未达最终质量不触发此专门早停。完整性/非法值或其他真实契约错误另行fail-closed。
- 不以训练loss下降或Exit0代替free-running检查；不根据CALIB改变LR/数据/epochs。

若没有早停，完成四个full checkpoint与screen并进入D。FULL末点不是自动winner；是否可以执行由完整family selection以及winner自己的TRAIN结果决定。这个区别允许原预注册selector选中5000/7500，但不允许根据WBC成绩再选一次。

资源最大科学上限：A5000+B5000+C10000=20000 canonical optimizer updates，前级不通过则不消耗后级。它不是保证需要或足够20000步，更不是不限步数。

## 7. Stage D — 一次完整五profile选择和runtime freeze

**仅FULL四checkpoint完整、无科学终止后授权。**

固定順序：baseline1000 → FULL2500 → FULL5000 → FULL7500 → FULL10000。每份原1088CALIB行，共5440个唯一profile cells。保留所有gate负结果。正常REJECT不能由outer runner抛异常导致后续profile不跑；坏身份/坏文件/缺行必须停止纳入结果并修工程。

C_T为本family唯一processor/runtime：从训练生效配置和stats建立label-free评估obs，stage0已封存；沿原provider、frame/TCP、dt、target authority和SafetyGate。可以新增薄profile入口绑定C_T和新family，但**原selector的admissibility/ranking函数不改**，其源码hash单列。不得把旧C_I scores、不同GPU的最佳profile或其他family拼接进选择。

排名仍是：完整无REJECT的候选 → median reference min-target-distance → worst source-rollout-cell median → 较低step。baseline不参与选模，WARN保留。原指标称native-boundary reference metric，不能冒称已实现的worldTCP结果。所有候选不合格：保留 `NO_ADMISSIBLE_CHECKPOINT` 并科学停止，不选最不差。

S3核对五份profile的模型/数据/行集合/runtime/processor/noise契约，S4原规则选唯一winner，S5只读诊断。C_T为新评估条件，旧PI0236负结果不覆盖、旧结论不冒称被直接推翻。

**额外sim准入，不改变排名：**选出的唯一winner在C阶段已有对应step的64TRAIN screen上，满足B的质量条件和旧16保持条件。若winner不满足，则 `SCIENCE_FAIL_SELECTED_TRAIN_SCREEN`；不能移除该winner后挑runner-up。缺完整同身份screen是工程补证，在已列cells范围恢复，不偷偷新增seed。

通过后写不可变 `selected_policy_manifest.json`：weights所有shards、processor有效配置/TRAIN stats、语言/输入接口、sampler/dtype、代码与runtime、原gate/provider、唯一step、selection文件hash。离线模型freeze后将其部署到Corelab，记录部署host/device binding；换host不称逐bit等价、不因表现再挑模型。

## 8. Stage E — 原生A–D C0 one-chunk，真实WBC任务真值

**D完成且唯一policy及sim_binding完整后自动放行，无需再次申请例行PI GO。**

只使用原canonical A/B/C/D C0各一个冻结seed/episode，调用已验证的dense-relative→TrajectoryBatch→frozen-WBC入口。一个共享checkpoint/processor，不做按任务换权重。每cell一个首次有效trajectory query，经原SafetyGate后执行；预计4个唯一one-chunk任务结果，技术失败按既有记录恢复。

原gate REJECT不得执行；记录该cell失败并停止科学推进。原scorer依据实际achieved world-TCP以及既定.035m成功半径、原native termination/cleanliness判定；不把reference命中、expert误差或离线过gate当任务成功。

记录input RGB/state/hash，target只在合法环境/scorer侧，raw/decoded/reference、gate及原因、实际worldTCP min/final/first-hit、tracking误差、tick与termination、原生控制日志。环境合法目标用于scorer/场景不等于可把target XYZ灌进GR00T或用oracle EEF替换输出。

四个cell全部满足原任务定义才 `ONE_CHUNK_PASS_4_OF_4` 并自动进入F。合法执行但失败是 `SCIENCE_FAIL_ONE_CHUNK`，不能改threshold、seed、滤波、dt、首点、WBC权重、target injection或重试挑轨迹。无效的import/IO/native启动错误继续工程恢复。

## 9. Stage F — progress-preserving receding闭环

**仅E四cell全通过后授权。**

使用同一冻结policy、同一WBC和sim_binding中的既有cadence/prefix/tick cap/query cap。A–D C0仍各一次完整任务；不新增任务family或benchmark。query数量由预先解析的任务cap与cadence决定，启动前run card写出每cell和全程最大值，不把工程不限次数理解为无限模型闭环调用。

必须保持：同一episode连续仿真；每次新query读取当前实际RGB/StateV1；真实模型forward；原gate；正确frame/TCP与动态anchor；reference进度/时间连续衔接。不每个H40重置s/t，不缓存旧观测，不把一次forward重复播放，不在失败后偷偷reset环境继续同一cell。

用事件ID绑定 `execute → fresh observation → infer → gate → reference update → execute`；记录observation_time、request/resultID、被执行的reference和真实worldTCP。使用已支持的同步暂停仿真方式时明确标注，不宣称满足实时硬件latency。跨Corelab/Hopper的model/sim分离仅在原协议已支持相同时间与过期输出处理时使用，否则不能把新的通信架构当无语义变化的小修复。

四个任务均按原scorer成功且handoff/clock/progress完整，终态：

`TRAIN_SIM_CLOSED_LOOP_COMPLETE — STOP_FOR_PI_GENERALIZATION_REVIEW`。

交付同一模型完成多目标reaching闭环的证据，不命名为不同操作技能的general model。当前计划终点不运行DEV-E/F或Final-F；后续单次DEV、盲测或多任务扩展需依据新冻结模型另行决定。本条明确收窄旧PI0239的DEV条件授权到本次TRAIN sim终点，防止跨协议误解。

有效receding失败立即 `SCIENCE_FAIL_RECEDING`，保留全部轨迹与失败阶段，不用DAgger或奖励改造继续。同一有用的scientific attempt只取first-valid，不因任务失败重试。

## 10. 资源、存储与恢复：工程自主，但不靠H200反复测代码

- 本地PC：代码、CPU pytest/fixture、git、SSH控制；使用已有Corelab和nus-hopper连接，不发明主机/凭据。
- Corelab：Linux依赖/真实数据集成；能容纳的训练和全部小screen；原native sim。实测GPU UUID/单卡总与空闲显存、RAM、CPU、磁盘。不要把用户说的32GB或两卡相加作为未核实单卡容量。
- Hopper：经站点批准PBS使用单H100/H200承担确实超Corelab容量或已获批的大规模训练/评估。H100/H200间选择只按容量/可用性及站点规则，不为追更好分数；同一阶段成组评估统一backend。
- Hopper login只做站点允许的轻量检查、传输、quota、qstat/qsub；重CPU、视频扫描、模型/仿真不在那里运行。

训练容量必须包含grad/Adam状态/activations，推理能跑不证明训练可跑。只用已批准batch1/原精度；静态容量已证明不可能时不必反复OOM，Corelab可完成的非GPU准备先通过再上大卡。不为了省显存改量化、optimizer精度、H40、分辨率、gradient checkpointing、模型容量。

Corelab模型和IsaacGym/WBC使用已分离环境，不把FFmpeg/LD_LIBRARY_PATH串入native环境。检查同时驻留显存，不能以两者各自能跑推断共驻可行。不能终止其他用户进程。

### 存储避免再次成为训练结尾故障

所有阶段的科学checkpoint均需**逻辑完整保留**，不要求所有副本在同一scratch同时驻留。使用已有授权archive存储及既有经过验证的归档工具；若需新repo，采用已批准的访问范围，默认不扩大为公开，不能把秘密/runtime上传。

每stage启动前按实际bytes计划：最新可恢复full checkpoint、当前原子写临时payload、必须的screen权重、archive staging、cache/log/NPZ、20%或10GiB取大的新增峰值余量。原fileset实际容量/配额或已有批准的等效阶段容量证据不可用时，自主寻找合法输出存储或等待资源，不盲跑到最后才靠删除抢救。

仅可在校验远端逐文件hash/full inventory、归档可恢复且非当前restore/save/infer依赖后，回收**本PI新建的冗余本地副本**。旧PI0241/623593/initializer/数据/seal不在本轮清理权限内。当前恢复checkpoint只有模型、optimizer/scheduler、RNG和TrainerState都加载完且有远端恢复源时才可释放；禁止再次重现先删后restore竞态。

### 恢复与科学预算

同stage exact resume需model/optimizer/scheduler、Python/NumPy/torch/CUDA RNG、global_step、sample schedule cursor、配置和max_steps一致，保存首个恢复训练样本到日志。缺少证据不称exact；可以同冻结seed/config重放丢失前缀并单独记工程重复计算，但不得拼接不同attempt最优weights或重复有效模型失败。

stage间A→B→C是明确的weights warm start和新optimizer，不是exact resume。单stage一次有效科学trajectory；不限制工程修复/job次数，但没有新增scientific arm、sweep、额外预算。

日志/每cell/phase state运行时落盘，原子小manifest；不要所有结果只放内存到最后。阶段已有效完成则核验复用，不从头重跑。稳定的公开source/statics引用hash，不每轮耗费大量token再次全库阅读。

## 11. 科学计数、交付与终态

最大新canonical optimizer updates：20000，按条件递进，不必全部执行。

固定离线unique policy cells上限（不含后续sim的已冻结query cap）：
- M16屏幕64；M64屏幕256；FULL训练屏幕384；五profile5440；合计6144，身份相同的边界复用会减少actual calls。
- M16 direct-flow额外32个唯一记录，只作诊断；内部backbone/head次数分别计账。
- one-chunk4个任务；receding4个任务的query cap由原sim_binding在执行前数值化。

技术失败的attempted/completed/reused、损失重放updates、scheduler allocation与取消均另列。原PI0240/0241/0242历史计数不加进新科学样本量。禁止用常数声明零受限读取或有效更新，需实际代码边界/日志支持。

建议只新增一个薄 `scripts/pi0243_staged_learning_to_sim.py`、必要的stage config和CPU tests；复用PI0241 fixed view、PI0228官方config/run、成熟screen/原selector/native入口、R1已验证direct helper。新脚本是待实现交付，不是已经可运行的软件。

每阶段完成写小 `stage_manifest.json` 和统一 `execution_state.json`，stage gate由实际指标计算并经PASS/FAIL/missing/duplicate/WARN-only/中途PASS末点FAIL测试。更改运行配置或生成新run card不能悄悄修改本文标准。

Issue #11在A/B/C/D/E/F关键终点先canonical记录结果与下一自动动作；PR #77只pointer。不必每处工程补丁新造科学协议，也不要求每PASS回PI。最终交付模型/processor/runtime与数据lineage、full checkpoint inventory、训练曲线和真实更新、全部screen与S3/S4/S5、one-chunk/receding worldTCP/event trace、实际资源/计数/缺口。

合理终态：
- `ENGINEERING_RECOVERING`：有根因的工程修复，继续；
- `EXTERNAL_INFRA_BLOCKER`：当前授权范围无可用资源/权限，保留进度；
- `SCIENCE_FAIL_* — STOP_FOR_PI_SINGLE_RECIPE_DECISION`：预注册终点真实未达标或科学契约冲突，停止整个下游；
- `TRAIN_SIM_CLOSED_LOOP_COMPLETE — STOP_FOR_PI_GENERALIZATION_REVIEW`：目标达到且证据完整，停止于TRAIN范围。

所有阶段都不承诺必然成功。真正的长程自主不是无限调参直到PASS，而是**当前假设有效则持续放大并接上真实闭环；无效则带着可定位的证据停止**。

## 12. PC agent启动正文

现在执行PI0243_STAGED_LEARNING_TO_SIM_GO。完整读取本immutable协议、R1 canonical5755602153和审核基线3ea3614444ddb76ebf1abe91be31744f0a6e6ded相关代码。训练现在仅按这条新路线获准，不再受旧PI0242零训练限制；不扫描全库、不重做已闭合G3/native parity。

本地测试/Corelab集成优先；显存或规模证明成立再上H100/H200。工程问题自主最小修复、回归、恢复，不设次数上限，不在H200反复debug通用Python。先封存三stage配置/样本schedule、C_T processor、存储及已有sim参数binding。

按协议顺序：
A 原16TRAIN+PI0226 baseline1000新optimizer→5000步；
B 仅A通过，A-final weights→原64TRAIN新optimizer5000步；
C 仅B通过，B-final weights→全9792TRAIN新optimizer10000步；
D 完整FULL四checkpoint后，baseline+2500/5000/7500/10000各1088CALIB，原selector唯一freeze；
E 唯一winner自己的64TRAIN screen合格、sim_binding闭合后，自动frozen-WBC A–D C0 one-chunk；
F 四个one-chunk通过后，自动原cadence的progress-preserving receding。

只改本协议明确的训练支持/预算/课程和C_T绑定，不改loss/peakLR/精度/capacity/gate/scorer/ABI/target输入。A/B最终标准沿原四项且B检查旧16保持；C早停、D无候选或winner训练不过、E/F真实任务失败按协议停止。早期普通screen未达最终标准不误判立即失败。

所有PASS自动进下一阶段，不逐项等PI；完整科学FAIL不靠新增steps/seed/平滑或改阈值继续。保留first-valid、原负结果与完整checkpoint归档，不换最差候选，不混family。small code/tests/manifests推execution分支，Issue11先canonical、PR77只pointer不merge。

终点TRAIN_SIM_CLOSED_LOOP_COMPLETE或明确SCIENCE_FAIL/真实infra blocker。DEV-E/F、Final-F、DAgger、硬件、多技能训练仍不授权，不重建monitor。

## 来源说明

这是PI预注册的新实验与条件授权，不是对源报告中尚未存在的5K/64/full/sim成果的摘要。5000/5000/10000及早停与阶段保持标准均在新结果前指定。原报告及源码支撑当前失败和既有接口；后续有效性由实际运行决定。

cosine schedule的总时长影响LR轨迹：Hugging Face官方说明 https://huggingface.co/docs/transformers/main_classes/optimizer_schedules 。仅用于解释干预，不升级pinned运行时。
