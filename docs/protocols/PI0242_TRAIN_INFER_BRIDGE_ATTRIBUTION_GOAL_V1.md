# PI0242 — 固定样本训练输入、局部流场与自由生成的对照归因

日期：2026-09-21（Asia/Singapore）。

Decision: `GO_READONLY_TRAIN_INFER_BRIDGE_ATTRIBUTION`  
Publication status: `AUTHORIZED_NOT_STARTED`  
Authorization token: `PI0242_READONLY_BRIDGE_GO`  
Audited/execution parent: `c1b2bd4634cc8cd5b756c9e3d9423d70be88b397`  
Issue #11 canonical; PR #77 pointer only, no merge.

## 1. PI决定与本轮唯一goal

接受PI0241封存的终点负结果和停止：该checkpoint在原screen路径下未达到fixed16标准，不升级WBC，不扩大数据，不追加训练。但不将其扩大为“优化器、输入一致性、loss到sampler的关系均已排除，因此只能加步数/改loss”。

**本轮goal：用已经封存的PI0241权重和样本，区分训练/推理条件输入或目标编码偏差、局部流场目标拟合不足、局部拟合向自由生成的转化不足；形成一张有直接证据的下一配方决策表。**

授权一次有界的只读诊断campaign，不授权任何新的training/backward/optimizer step，不建立新训练checkpoint。不重新开启全库审计或G0–G4，不自动执行下一训练recipe。本goal的完成是归因证据交付，不是强行得到模型PASS。

## 2. 本次审核实际依据与边界

已读取：

- PI0241 canonical issue comment `5754335876`，其中报告训练625395和screen625411均Exit0；旧授权comment `5751592991`。
- parent下 `scripts/pi0241_fixed16_learning.py`：选样、schedule、config、真实dataset factory、参数审计、screen和分类器。
- `scripts/pi0241_resume_and_finalize.py`；修复commit `518f2b13f959d8bb016437d81bef132aecbd37bd`。
- `scripts/pi0241_reconcile_update_integrity.py`，commit `02e9f4765f02350d45d3fb6b172138e5b6fa3b7a`。
- `scripts/stage21_same_scene_groot_inference.py` 的_build_processor/_run_one_raw/_run_one；PI0228/Stage28继承配置（此前审核）。
- `reports/STAGE23_PHASE_C_PROTOCOL.md`和`reports/STAGE33_DUAL_TRACK_BOTTLENECK_MAP.md`仅作旧实现/设计导航，不把PREPARED文档当旧实验结果。

GPT本次没有登录Corelab/Hopper，没有读取远端原始closure JSON、64个NPZ、optimizer.pt或完整loss日志，没有独立重算指标/验证大文件SHA，也没有运行测试/模型。HF连接仅确认归档仓库metadata可见，不代表原始证据已被读取。以下数字来自canonical；代码发现与机制假设分开表述。

### 2.1 接受的报告事实

PI0241 step1000：REJECT16/16；median逐窗XYZ RMSE=0.20030531962816311m；p95=0.23236484491746884m；median逐窗SO(3) RMSE=0.09790447437553179rad。四项均失败。selected_checkpoint=NONE。

档案身份：

- HF repo: `leoyuer/pi0241-fixed16-learning-v1`。
- archive manifest SHA256: `b4dd42125191445459744f13af326e0241cc8ed8f51197651e492a6dd766ce7f`。
- step250 tree: `746d498b0ff4e7ee2c935c588290d34b35d5662093e85abc5e7fed74de6e04cd`。
- step500 tree: `5c693760152fe803638c0eb8da7f23011a3e0168e32762dfbbf51609a8297688`。
- step1000 tree: `bbf0d48a39b7433f3a650ac3a34a69a7deb5058f89cb3eab65897aa6a94b6180`。
- training manifest: `c808c9e89838d9024ca5d81b4684300c7f27ad2b0b025637afd715431ddb2b50`。
- recovery manifest: `221f2970aa6ccfcd5e7d21eb9945aef8847c703ce234e127c2ea9d068a7c5336`。
- reconciliation: `cab2e02ad52c5d182066a85128b1b65ed142e53130dd0f52336a5e9cce49babf`。
- closure: `e3c54d8dad07bcee373cc11c85fb7e4f06338524eda69381db5dcc1a304a18be`。
- run card: `d7d9f0aa6186aea6279b09dedd6ffa6be7f3010d1286568f3d730a704218aa95`。

15/15 projector tensors变化推翻了“projector完全未更新”的原抽样假阴性，不证明有效embodiment的所有相关参数更新充分、loss已经拟合或自由生成应当成功。保留原false-negative记录，不重开同一个无更新结论。

### 2.2 源码中明确可见的证据缺口，不冒充已证实根因

**A. 两个processor构造入口。** 训练由Gr00tN1d7Pipeline的实际config建立processor；screen调用Stage21 builder，从baseline/processor_config.json的processor_kwargs重建，再补TRAIN statistics。训练继承shortest_image_edge=96等配置。functional_preflight只检查首个训练样本的action/mask形状、active slots与ID；没有证明全部16窗的实际训练条件张量和screen条件张量相同。不同builder不自动意味着实际不一致；必须比较输出和配置才能定论。

**B. 恢复消费证据。** resume wrapper的resumed_iter显式抑制sample_consumption和sample_contracts写入；恢复后manifest写入actual_sample_consumption=1000。_combined_audit又合并initial/recovery事件。不能只凭这个常数字段或旧失败attempt的1000行日志，证明625395的501–1000真实绑定正确。518f2b1已修复手动skip与Trainer skip叠加，不预设仍然有错误。先核实canonical lineage、实际Trainer版本/源码、状态和日志；区分OBSERVED、DETERMINISTICALLY_RECONSTRUCTED、NOT_AVAILABLE。缺日志不是错恢复的证据，也不靠重训“补历史证据”。

**C. screen载入身份。** screen按training/checkpoint-N或screen_models/checkpoint-N路径载入。必须把真正加载的model shards/config绑定到封存manifest中的内容hash；文件夹名不是身份。step0复用的是PI0240/Corelab输出，其他screen在H200；旧曲线只作描述，不把跨硬件seed标签当作新paired试验的noise身份。

**D. 缺少局部流场实测。** 现有wrapper主要记录参数、最终动作和gate，没有本轮fixed16的固定noise/time局部流场对照。不能声称teacher-forced loss已经很低，也不能仅凭20cm终点误差断言BF16、容量或四步sampler是根因。

不能把本次median RMSE20.03cm和PI0240的mean error15.15cm直接比较为退步：统计量与cohort不同。只在相同16窗上统一口径做描述性比较。

## 3. 冻结的科学边界

沿用official dense-relative H40、StateV1、active slots0:10/universal132、原TRAIN stats/relative_stats、原frame/TCP/row-rot6d、dt0.05s、原语言和SafetyGate。

语言：`move the end effector to the red target`。

模型只能接收原RGB/state/language与模型本来需要的非特权条件。target XYZ、source ID、phase、expert未来不得成为模型观测。专家动作只用于训练目标重构和离线诊断，不能用作自由生成input或初始化。

只比较两个权重：PI0241初始化的PI0226 adapted baseline1000（不是通用BASE），以及PI0241最终step1000。step250/500仅读取已有日志、metadata或已存NPZ，不增加它们的新模型查询，不选最佳中间点。

不增加seed/LR/loss/precision/denoising-step-count/active-dim/noise干预臂；不平滑、限速、裁剪、重定时、零化inactive122或改阈值。BF16等运行方式按各冻结路径真实实现记录，不能为结果更好擅自统一成FP32。两条条件路径的模型执行模式均固定eval；两条路径的原生预处理差异作为本实验对照对象保留。

## 4. P0 — 一次性恢复证据与真实训练输入闭合，0模型forward

### 4.1 先读取小证据，不复制全历史权重

从原run card解析具体output/path，按上节SHA核验closure、training/recovery/reconciliation/archive，提取原16窗cohort和schedule。不得使用历史无效625384 checkpoint250或625381 partial1000替代canonical权重。

用现有64个screen NPZ/records重算原指标和每步summary：逐窗XYZ/SO3 RMSE、gate issue codes、gripper、速度、首尾误差。step0/250/500/1000全保留，不选最好。分解每窗XYZ误差的bias与去bias残差：RMSE²=||mean(error)||²+mean(||error-mean(error)||²)。输出速度尖峰所在segment及专家对照。历史日志已有native loss/LR/gradient则整理并标明断点和attempt；不得补造不存在的值。

恢复轨迹只检查影响canonical归因的部分：checkpoint500→625395→最终1000的model/optimizer/scheduler/RNG、TrainerState.global_step、原max_steps=1000、真实optimizer step计数及LR、实际恢复源码。把初次attempt501–1000已丢失计算和有效recovery501–1000分开，不能拼成1500 canonical updates，不能把合并audit中同step事件当同一轨迹。

允许CPU-only、无模型的deterministic cursor replay检验pinned Trainer/dataloader skip语义和501/1000对应ID。它是当前代码重构证据，不称历史真实消费记录。证据缺失时保留UNKNOWN；只读checkpoint行为诊断可继续，但不宣称已独立证明exact resume。发现正面的错权重/错数据/错状态证据则停止受影响比较，交PI，不自动重训。

### 4.2 对原16窗比较实际条件与目标，而不再做shape-only检查

分别建立：

- `C_T`：从PI0241真实resolved config、官方pipeline和fixed dataset生成的条件张量；不是调用eval builder后改名train。
- `C_I`：原Stage21/PI0241 screen builder生成的条件张量。

用同一原始window、RGB、state、语言比较实际观测时间/视频帧、camera次序、RGB通道/范围、图像变换结果、pixel_values/分辨率/grid、token IDs/attention mask、state归一化与mask、embodiment ID及其有效投影行、relative-action开关、stats身份、delta indices、action slot布局。记录实际存在的keys；不要假定未验证的字段名。比对离散张量用exact；连续张量报告shape/dtype、max/RMSE差异和cast前后差异，不能用宽松容差掩盖归一化/分辨率差异。

将C_T真实collator action/mask同PI0240原16窗expert_normalized_132/mask比较，再按独立既有decode回到physical expert；沿既有容差与slot定义，不自写codec。核验每窗400个有效action元素（40×10）；如实际不符先定位，不改mask使其通过。

保留专家未来和metadata在诊断侧，但从模型条件中严格移除。由pinned模型split-input路径验证label改变不会改变观测conditioning；get_action不能读取expert。train-mode随机augmentation若在实际配置中存在，记录为与原卡的冲突，不暗中关闭后冒称历史等价。

冻结完整resolved config和effective processor kwargs。旧summary的Adam beta/epsilon含字面值；用真实TrainingArguments/optimizer param_groups核对，不能把重复抄写当运行实证。记录actual trainable weights、grad/Adam moment dtype和有效embodiment行的已有full-tensor变化；只读，不新做backward。

P0完成后：若两路径完全等价，明确排除本组样本上的条件差异；若不等价，列出最早分叉。不同但无证据违反目标语义时称DIFFERENCE，而非BUG。无需等待再次授权即可执行以下已固定P1，只要权重/目标/输入可解释且未发现会使比较无效的来源冲突。

## 5. P1 — 固定4窗、2权重、2条件路径的配对诊断

### 5.1 样本和预算（结果出来前冻结）

从PI0241原16窗中，每个target_family a/b/c/d取1窗。排名键：SHA256(UTF8("PI0242_BRIDGE4_V1|"+source_episode_id))升序，source_episode_id字典序破平。不得按PI0241 gate/RMSE、phase、图片内容挑选。输出cohort和hash；这4窗只支持本轮机制诊断，不是16窗通过率估计。

矩阵：4窗 × {initializer, PI0241-step1000} × {C_T,C_I}。

- 最多16次stock free-running sampler/get_action查询，捕获完整轨迹内的向量场调用；没有新seed和step-count sweep。
- 每个上述cell最多4个固定teacher-forced local-flow条件，共64个unique direct-flow cells；与sampler首步完全相同者可复用已记录值。
- 条件输入完全一致、权重/噪声/runtime全同的cell可通过内容身份去重，但保留完整矩阵和复用指针。
- backbone特征每个window×condition至多计算一次并复用，前提为冻结权重和处理语义相同。统计policy queries、backbone forwards、head forwards分别计数，不把16条sampler称16次总forward。

### 5.2 噪声与运行身份

两个模型、两条件在同一设备/同一运行profile内比较，不拼接旧Corelab step0和新H200输出形成配对实验。固定seed22301；对每窗先按原C_I stock路径捕获一次完整初始[1,40,132]噪声，后续对应cell复用该exact tensor。首次C_I查询计入16次预算。通过作用域明确的initial-noise注入/截获实现配对，不更改任何迭代公式、dtype、time embedding、更新步数或inactive维反馈。hook外行为不受影响。仅“相同seed”而未核对noise hash不算配对。

原stock路径所用step数、time grid、连续t与离散embedding从pinned源码/实际config读取后封存，禁止拿最新版GR00T替代。若与旧四步N1.7描述不同，先查该checkpoint真实config；不为了凑四步而改它。

### 5.3 局部流场诊断

复用Stage26的已审计direct action-head图或等价薄只读hook，严禁重新发明模型/训练实现。noise保留完整132维，active mask仍0:10。按照本次pinned源码确认：

x_t=(1-t)epsilon+t*a_star；v_target=a_star-epsilon。

4个t为：0，以及该训练时间分布的q10/q50/q90。旧pinned文档描述Beta(1.5,1.0)、noise_s=.999，变换t=(1-B)*noise_s，对应约.0677624/.3696694/.7837720。实际字段值、精确数值与离散化必须从本次源码/模型确认；不无声沿用文档数值。发现语义冲突先停，不挑其他t。

对每个固定probe报告：

- 原native loss定义和分母；active10、XYZ3、rot6d6、gripper1分别的velocity MSE；inactive只描述能量，不计入active拟合。
- epsilon、a_star、x_t、t、discrete embedding、predicted v、mask、conditioning hash。
- 可用epsilon+v_pred构造teacher-forced implied endpoint并按原decoder回到物理量，报告XYZ/SO3误差，明确其t>0输入包含专家信息，只是oracle诊断，不是policy success。
- 对两模型/两条件给逐窗逐t配对差值，不只报告一个总loss。

如果原loss是在mask后对132维mean，报告其与active-mask-normalized loss的关系，不因比例因子就直接认定loss bug或擅自重标定。

### 5.4 自由生成trace

在stock sampler内部捕获每步x、v、t/time embedding、初始noise、最后raw action；不额外增加采样步数。最后分别保存原C_T/C_I decoder结果；若decoder不同，补CPU交叉decode同一个raw以分离conditioning和后处理效应，不以更优decoder替换历史成绩。

报告每步active/inactive能量和末端误差、最终原gate、逐窗RMSE/首尾/速度尖峰。可报告生成轨迹与给定noise–expert直线的距离，但**不得把任意自由生成状态上的a_star-epsilon当作唯一真实velocity标签**，从而伪造off-trajectory oracle误差。local-flow target比较只在定义的teacher路径上进行。

不通过修改noise的inactive122维来救结果，不用teacher-forced endpoint作为WBC参考，不把低loss当自由生成成功。

## 6. 解释与停止规则

最终必须交付同一矩阵：输入一致性／训练目标回环／恢复证据等级／有效参数更新／按t和action组的local-flow误差／原自由生成误差与gate。

按证据给非互斥分类，不能硬编码唯一根因：

| 证据 | 可支持的结论与下一配方方向（本轮不执行） |
|---|---|
| 真实target/normalization/embodiment/观测语义与原冻结意图冲突 | CONTRACT_MISMATCH_CONFIRMED；定位具体字段。先修对应构造路径，再另开编号复验；只有训练输入/目标有错才讨论重训。 |
| C_T/C_I实际不同，且同weights/noise下输出有配对差异 | CONDITIONING_PATH_EFFECT_MEASURED；报告效应大小和例外。不能仅见不同builder便称根因；效果未解释全部失败时保留剩余拟合问题。 |
| 等价条件下local-flow物理重构和free-running均远未达到原尺度 | LOCAL_OBJECTIVE_FIT_DEFICIT_SUPPORTED；下一轮优先有证据的最小优化/目标干预，不能直接指定BF16或容量。 |
| 相同条件下local probes接近专家，而stock自由生成仍差 | LOCAL_TO_FREE_RUNNING_GAP_ON_PROBED_SUPPORT；优先检查采样路径、训练t覆盖和有效动作组，不直接扫4/8/16/32，不声称已证明唯一sampler bug。 |
| 只有部分t、部分组或部分窗好 | MIXED_BOTTLENECK；保留分层，不强归一类。 |
| 必要原始证据无法读取/某比较无法合法构造 | ATTRIBUTION_INCOMPLETE；写明已测、未知、最小缺项，不编造训练曲线、不重训补证据。 |

所有local-good/poor表述须附实际物理误差和原PI0241标准作参照；四窗及几个t不是对整个分布的证明，不新增任意百分比门槛冒充机制因果定理。

终点：`PI0242_ATTRIBUTION_COMPLETE — STOP_FOR_PI_SINGLE_RECIPE_DECISION`，或上述typed blocker。即使某诊断C_T结果很好，也不promotion、不selected_checkpoint、不WBC。本轮目的不是再拿一次笼统SCIENCE_FAIL，而是给出能决定下一次最小干预的证据及仍未区分的假设。

只读发现明确工程实现错误可自主修诊断工具并继续；改变原训练/模型/数据/推理语义的科学修复仅作为具名对照或建议，不冒充等价工程修复。已产生的first-valid结果保留，不因成绩重新选样或改矩阵。

## 7. 资源、工程自主与产物

Corelab优先。P0在本地/CPU执行；P1为无梯度推理，应先复用PI0240的Corelab成熟环境。只有真实必要显存/运行依赖证据才用H200。不得在H200调路径/跑普通CPU测试，不结束他人进程。

只物化本轮两个权重的推理必需文件，checkpoint250/500优先读取现有metadata/NPZ；不复制三套optimizer，不新增checkpoint输出。需要读取历史optimizer证据时单独按需读取/流式处理，不能成为无理由复制全历史的任务。验证当前fileset与输出峰值，原sealed权重和数据不删除，不新建定时monitor。

工程修复无任意次数上限，但每次记录根因、最小diff、针对性回归；同一阶段最多一个活动执行。IO失败可恢复到缺失的固定cell，既有first-valid结果不覆盖；算术/语义错误必须保留旧结果及更正来源。诊断科学预算不变，不新增表现驱动arm。

建议只新增一个薄入口 `scripts/pi0242_train_infer_bridge.py` 和针对性CPU tests，复用官方pipeline、Stage21、PI0241 metrics、Stage26 direct-flow及原gate。准备、诊断、归因依次连续执行，不在数据目录/配置等已授权工程点反复停PI。

CPU测试至少覆盖：两个builder相同/不同检测；离散ID差异；mask/分母；未来label不进入conditioning；重复/缺失cell；固定noise配对；observer hook不改变stock输出；未知恢复证据不被硬编码PASS。GPU hook parity如需额外模型调用必须纳入上面固定矩阵预算，不能借smoke另开hidden cells。

交付集中为：

1. `pi0242_run_card.json`：输入证据hash、bridge4 IDs、精确矩阵/noise/time/runtime/代码身份。
2. `pi0242_evidence_index.json`：原64结果重算、恢复证据分级、16窗tensor/label parity、有效更新和loss日志索引。
3. `pi0242_bridge_records.jsonl`及小型trace NPZ：全部配对结果、ledger、artifact hashes。
4. `reports/PI0242_TRAIN_INFER_BRIDGE_CLOSURE.md`：source事实、实测、推断分列；机制对照表；唯一优先的下一干预建议/不能决定的最小缺项。下一训练run card必须标明NOT_AUTHORIZED。

Issue #11先canonical，PR #77只pointer。不得上传token、完整私有runtime或大权重到GitHub。没有新增backward/optimizer/training、CALIB/DEV-E/F、sim/WBC/DAgger/硬件；selected_checkpoint始终NONE。

## 8. PC agent启动正文

执行PI0242_READONLY_BRIDGE_GO，以本文为唯一新科学授权。先读取PI0241 canonical5754335876与parent c1b2bd4634cc8cd5b756c9e3d9423d70be88b397的相关脚本，不扫描全库、不重跑旧G0–G4。

连续完成P0→P1→归因closure：原16窗真实training-pipeline与screen的条件/目标tensor对照、恢复证据分级及旧64 cells重算；再按本文hash选每个A–D一窗，在initializer/step1000与C_T/C_I上做固定noise的stock sampler和teacher-flow矩阵。最多16 sampler查询、64 direct-flow cells；统计内部forward，不新增seed或sampler步数。

工程自主修复继续；不得用常数字段当恢复实证、不得把两个相同eval builder冒称train/eval parity、不得把参数changed当已经fit。发现确切语义冲突和仍然未知的证据要分开。Corelab优先，不新训练、不改loss/精度/ABI/gate、不进入WBC/sim/CALIB/DEV/F。交付机制证据与一个下一配方建议后STOP_FOR_PI_SINGLE_RECIPE_DECISION。此goal不需要再询问是否授权只读模型诊断。
