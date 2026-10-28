# PI0248-R1 — 保留训练配对失败，放行现有模型的固定质量评估

日期：2026-09-27（Asia/Singapore）。

Decision: `GO_FIXED_CR_SCREEN_WITH_EXPLICIT_TAU_PAIRING_LIMITATION`  
Authorization: `PI0248_AS_EXECUTED_QUALIFICATION_GO`  
发布状态：`AUTHORIZED_NOT_EXECUTED_BY_GPT`  
审核执行源码：`c2dc1eb000750ad39ed5c0548f6d15a0cccc5598`。  
原协议：`81e3faa4eb23e6ce0ed521e6b8dd5b7206506771` / `docs/protocols/PI0248_M64_POSE_REPAIR_PAIRED_PILOT_V1.md`。  
Issue #11 为 canonical；PR #77 仅 pointer，保持未合并。

## 0. PI决定：不重训以修配对，不改原FAIL，用现有权重回答质量问题

C/R两臂均已经执行2000更新，但C在1000点恢复后的CPU Beta tau流与R不再配对。原严格配对执行失效，原停止判断正确。现有证据没有给出C/R模型质量结果：原512个screen cells仍为0/512。

本文件作出明确的**科学分析范围修订**，不是把tau差异当作可以忽略的工程同义修复：

1. 永久保留原 `STOP_UNPAIRED_NATIVE_TAU_IDENTITY_MISMATCH`；不生成伪造的 `PASS_ALL_2000_STEPS`，不重写tau/RNG日志，也不把C的完整恢复称为已经证明exact stochastic replay。
2. 新的近期问题是：**这两份按实际过程训练完成的模型，在原固定64及old16、原B0/B1上是否达到原质量门？** 这与“新增旋转loss的纯单因素因果效应”分开。
3. 在逐臂执行完整性复核通过后，放行已有C1000/C2000/R1000/R2000的原512格screen及独立CPU复算。只使用原模型、原输入/noise、原评价器计算和原阈值。
4. 本轮新增训练、backward、optimizer updates均为0。已执行的4000更新不重置；512格是原来尚未执行的评估额度，不新增另一组512。基线128格按内容身份复用。
5. 配对输入/noise下的模型输出仍可逐source比较；final2000的R−C差异必须披露native tau流不同，不能全部归因于旋转项。checkpoint1000保留配对前缀的研究价值，但仍只是中期观察，不能替代final2000晋级。
6. 全部固定评估完成后，有合格产物则封存 `AS_EXECUTED_M64_QUALIFIED` 候选并提交FULL准入审查。没有合格产物则如实停止。**本文件不开放FULL、CALIB、selector或WBC/sim。**

这项授权不降低任何位置、姿态、gate或保持门，也不保证会得到合格模型。它允许在不知道screen成绩之前，使用一项透明的分析修订，避免因原因对照不完整而继续搁置已经训练出的权重。

## 1. 审核来源与当前状态

必读且复用：
- [原PI0248授权5847759659](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5847759659)及原不可变协议。
- [C/R终态配对停止5850988664](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5850988664)。
- [RNG/恢复源码审计5851180653](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5851180653)。
- [checkpoint种子的CPU Beta重放5851224835](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5851224835)。
- [只读tau位移分析5851302733](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5851302733)。最新读取时点为2026-09-27 00:34:07 UTC，即08:34:07新加坡时间；本轮未查询live scheduler。
- 执行源码 `scripts/pi0248_cr_train.py` 的 `_preverify_training`、`_write_training_integrity` 与resume路径；`scripts/pi0248_cr_screen.py` 的 `check_card`、`check_evidence` 及其继承的M64 evaluator。
- `reports/evidence/PI0248_C_EXACT_RESUME_PROGRESS_20260927.md`；此文件中的较早“exact resume PASS”应结合后续tau证据解释，不删改历史文本。
- 公开证据包：HF repo `leoyuer/pi0246-gate-pass-campaign`，固定revision `c7ad3f747c36ecd45025ed7156c61d9352580032`，prefix `evidence/PI0248/m64-pose-repair-20260927/paired-identity-stop-634405`。执行侧报告五个证据文件读回通过，bundle SHA `14c5b152632288b331e87a26567e457a12a6ac8be2e3617981e72c818c36e843`。

GPT本次读取了上述GitHub账本、协议及相关源码，没有SSH、实时qstat、模型加载、HF权重下载、原始tensor复算或远端作业操作。这里接受的是来源支持的状态；执行agent对必要原始证据作只读核验。未亲自重新验证HF完整大文件。

### 1.1 已确认的区别

| 对象 | 最新证据支持的状态 |
|---|---|
| C | 原前缀1000＋恢复后1000，共2000；有效终点job634402；旧in-run integrity曾PASS，但完整随机轨迹exactness已被后续tau证据否定 |
| R | job634405完成2000；update audit COMPLETE；原finalizer在C/R配对检查处停止，未写R训练完整性PASS |
| 样本与Gaussian噪声 | 两臂2000条sample sequence/64 contracts一致；native epsilon与stock initial noise各2000/2000同index一致 |
| native tau | 前1000配对；首差异step1001，C=.31640625、R=.203125；后1000中995条不同 |
| RNG checkpoint1000 | 两臂rng_state.pth byte-identical，14645 bytes，SHA c425f3be6f7af031fcf2ada4e6e59c9eb4cd89cd3ca7e9914ef0e1954e6512a9 |
| CPU replay | saved1000 CPU状态重放Beta(1.5,1.0)、原noise_s=.999与BF16路径，后1000与R全部一致，和C仅5条同index一致 |
| 位移侧报 | 后段存在与R+8相合的长连续区间，但具体CPU RNG consumer仍未记录；不能声称单一额外调用已确证 |
| 质量结果 | C/R新screen 0/512；没有这两臂的正式模型PASS/FAIL；selected_checkpoint=NONE |

这说明“R没有旧PASS manifest”不等于R没完成训练；当前旧入口把逐臂完整性与跨臂tau匹配耦合了。也不能反向推论R所有独立完整性项目都已经验证完成：新逐臂核验必须覆盖旧异常之后尚未运行的检查。

### 1.2 必须固定的模型来源

以下均属于repo `leoyuer/pi0246-gate-pass-campaign`，路径前缀由各固定revision的durable receipt解析，禁止凭名称选择最新版。

| arm/step | 固定HF revision | 完整checkpoint tree SHA256 |
|---|---|---|
| C1000 | 3d817c29f972735475c2e5fde3b244aede6f6d79 | 8c0cfdd6116cdfc0e295091900539b438fcd56d87423a4cbcc7a28a5fc8c5e46 |
| C2000 | 001d86cc3492304e4df2816b3182a5e826daa0b7 | d72cb37db3f2f3931a7adacf24c2ff8736e4c1dc4836e3a4c75dd3087758425b |
| R1000 | b32f423d6dc58f00a05c246d409ac594f02fd78b | b3077d1c25c40271ec27fdd031b7bcc192df39b10f4801ab0153015a702ed3fd |
| R2000 | d089c09380c568e720c3dfa6bec5aa064acd7c4f | 4422a24d2117206681c423a0f8d3137dff7eb614c990cb0334a8cdaf73900298 |

C/R共同初始化仍是原PI0247 M64-final5000，revision `093c450fd49052c8fd3285bf38111a30a1f59009`、tree `1811bfa5c29fb7ce2e911ade0a0b476607e32ac1c1322cce68b8dff04449d776`。原U-M16合格模型及原M64 FAIL都保持。

## 2. G0：逐臂独立收口，保持原配对失败可见

这是零新模型调用的窄核验，不重新训练，也不重新构建全数据树。

为C和R各生成一个**新文件/新schema** `PI0248_AS_EXECUTED_ARM_INTEGRITY_V1`，另有一个共享deviation manifest。以原始card、更新记录、checkpoint receipt及实际source为输入，至少确认：

- 2000实际更新与原2000样本顺序完整；原64 source/observation/action-mask契约和固定初始化不变；无重复计入恢复前缀。
- C仅为原U目标；R保持native+.1位置+.1增量+.1旋转的原定义，实际总loss与各分项一致且有限。不得以原tau失配为由豁免其他损失/模式异常。
- 每臂原S4 forward/全四步梯度、mode恢复、实际参数更新、冻结模块与optimizer membership的原必要检查。报告来源/实测，不用期望常数伪装观测。
- 完整1000/2000 checkpoint与durable inventories；配置/模型/optimizer/scheduler/RNG和训练来源绑定。已读回收据可复用，推理文件物化时另核内容。
- C恢复的optimizer/scheduler/source cursor等按已有证据核对；完整CPU随机流连续性单独记录为不成立，不把“加载了RNG文件”升级成消费顺序exact。
- 保存每臂**实际**tau序列与源hash；合法域/finite等检查沿原实现。不得将R的tau拷进C日志，也不得通过offset重排两臂样本/噪声来补配对。

共享deviation manifest必须含：

```text
original_paired_execution_status = STOP_UNPAIRED_NATIVE_TAU_IDENTITY_MISMATCH
training_pairing_scope = PREFIX_1_1000_ONLY
training_tau_full_pair = false
native_epsilon_same_index_matches = 2000
stock_noise_same_index_matches = 2000
tau_same_index_matches = 1005
tau_mismatch_first_update = 1001
tau_mismatch_count = 995
C_full_stochastic_resume_equivalence = false
original_pair_failure_preserved = true
final2000_single_factor_causal_attribution_admitted = false
```

这些计数以原记录复核为准，若发现不同要报告而非强行填期望值。逐臂核验通过后的新状态可为 `PASS_AS_EXECUTED_ARM_INTEGRITY`；它不等于原paired PASS，也不等于质量PASS。

特别注意：旧 `_preverify_training` 中R的tau比较会阻断后面的共同finalizer。新只读工具应拆开/复用独立checks并生成新receipt，而不是删除比较后用原schema写PASS。不要直接在原目录调用会重写overlay或旧manifest的normalizer；必要小文件视图放在独立输出目录。所有旧hash、旧失败和原C/R card不覆盖。

**只允许这一个已披露的tau配对/连续性例外。** 模型身份错误、目标输入泄漏、数据/decoder冲突、非法值、训练实际未完成、缺失完整checkpoint等仍按真实问题停止受影响profile。若独立核验发现额外缺项，先仅用已有证据补齐；不能虚构PASS。

## 3. G1：薄screen准入适配，不改变预测或评分函数

现有 `scripts/pi0248_cr_screen.py::check_evidence` 对R要求 `PASS_PI0248_CR_TRAINING_INTEGRITY` 及 `paired_noise_identity=PASS_ALL_2000_STEPS`。在原协议下它正确拒绝执行；这次是PI显式修改评估准入目的，不是让agent自行绕过检查。

保留旧入口/旧失败的默认行为；增加明确的 `as-executed` 评估scope或独立薄adapter，只在以下全部身份成立时启用：
- 本次不可变protocol+canonical授权；
- 对应逐臂as-executed完整性receipt及共享deviation；
- 原训练card/family/checkpoint和scope之间一一映射；
- 原HF pinned inventory、processor、TRAIN stats、H40/S4和真实B0/B1条件/noise身份齐全。

不要修改原训练card的authorization以适配新常量。外层evaluation card引用原card和新scope即可；receipt、screen card、report命名均不能冒充旧paired成功结果。原RNG配对检查仍应在测试中返回失败。

只做CPU fixture回归：匹配新scope可准入；无授权/缺deviation/错误checkpoint/缺独立完整性仍拒绝；旧scope仍因tau拒绝；实际预测、decode与评分函数hash/参数不变；false/空结果不能被all(empty)判成功。使用既有validator，不重搭通用执行平台。

评估入口本身不需要新增真实模型parity；首个原定cell承担集成并计入512。若既有协议内模型parity确实还需执行，先核对原4次额度的实际余额；本文件不新增parity或梯度smoke预算。

## 4. G2：执行原512格，把权重转化成科学质量结果

在首个screen前封存四份模型、原输入/noise与新分析scope。顺序：C1000 → C2000 → R1000 → R2000。每份原64窗口 × 原M64 B0/B1，共128格；四份合计512。每个noise组分别报告，不合并成128个独立样本。

原M64-final5000的128格baseline按完整身份复用，不再查询baseline。原old16的source集合固定；新48/A–D/phase只作预先可解释的分层，不加隐藏资格门。

执行要求：
- Corelab优先无梯度screen；HF直接到Corelab，只拉必要model/config视图，原冻结processor/stats另绑定。不下载optimizer到推理GPU，不经本机PC中转大权重。
- S4仍原四步，H40/dt=.05、132维/active0:10、原图像/state、decoder/rotation行约定、gate/scorer不变。不加M padding路径、平滑、裁尾、retime或新seed。
- B0/B1使用原封存实际tensor，而不只复用seed数字。两臂输出可以按相同评估condition/noise配对，但不得由此宣称训练tau也配对。
- 以完整source_id+checkpoint+noise+runtime为cell key，first-valid保存。合法REJECT/大误差是有效结果，不重抽或重试；未形成有效结果的工程错误才可恢复。
- 一个profile只有一个执行owner；先检查actual job/产物，不假定旧ledger实时；无培训作业的场景也不应盲投GPU探测。
- 独立CPU复算128/份：身份、文件hash、shape、finite、全H40 decode、XYZ/SO3、线角速度和全部gate issues。报告/上传故障只修传输或报告，不重跑正确cells。

## 5. G3：原质量门不变，解释分两层

每臂仅final2000决定模型是否合格；以下四组分别全部过门：全64 B0、全64 B1、old16 B0、old16 B1。

| 条件 | 固定要求 |
|---|---:|
| 有效数据 | 原窗口集合、全40点及原mask，数值有限 |
| 原SafetyGate | REJECT=0；WARN保留 |
| median per-window XYZ RMSE | ≤.020 m |
| NumPy linear p95 XYZ RMSE | ≤.035 m |
| median per-window SO3 geodesic RMSE | ≤.050 rad |

### 5.1 模型资格层

全部512格完成并独立复核后：
- 仅一臂final2000四组通过：记录该实际训练产物为候选；同时注明其训练执行范围和配对限制。
- 两臂都通过：沿原预先选择，优先较简单C；不是事后改变排名以偏好R。记录R同样合格，二者都不因此成为正式selected。
- 两者都未通过：`PI0248_AS_EXECUTED_NO_QUALIFIED_MODEL — STOP_FOR_PI_NEXT_HYPOTHESIS`。它是这两份实际权重的有效质量负结果，不是一个恢复严格配对后的旋转loss因果否定。
- 任一臂缺有效证据/有效screen：标明确incomplete/invalid，不自动把它当FAIL或从完整两臂中消失；不自动替换新权重。

通过终态：`PI0248_AS_EXECUTED_M64_QUALIFIED — STOP_FOR_PI_FULL_ADMISSION`。

候选manifest包含：weights/tree/revision、原recipe、所有执行子card/恢复来源、实际tau/noise证据、analysis_scope、原paired fail、四组screen和独立复核，以及 `selected_checkpoint=NONE`、`sim_admitted=false`。C候选的随机轨迹偏移必须显著标注；这份screen不证明该已偏移训练过程可被原recipe逐bit重现。

### 5.2 机制解释层

- **1000点**：源码/条件/训练完整性核验支持的同初始化、同样本、同epsilon/tau前缀仍可作同期中期对照，报告R1000−C1000。只对该前缀和一次训练实现解释，不外推普遍显著性，不提前选1000。
- **2000点**：比较R2000−C2000、每臂相对自身1000及共同M64起点。评估输入/noise配对，但C恢复后的tau不同，是明确混杂因素。不能将全部改变量归于新增旋转项，也不能声称tau差异必然无影响或是全部影响。
- 报告位置median/p95、姿态、全部gate原因、线角速度、old16/new48保持、全H40误差和已有saturation/bias分解。只从已有NPZ计算，不新增模型诊断。

不选择较好noise，不池化四组、不删问题窗口、不用1000替代2000。原U-M16合格及M64 FAIL保持，不因新评估重新定义。

## 6. G4：CPU恢复回归并行，不能再阻塞当前模型资格

精确CPU RNG consumer尚未定位，+8的后段匹配也不能证明“只要补/跳八次sample就修复”。本轮不继续无界offset扫描，不重新生成历史tau或补训后1000。

允许在原实际库版本上，用小型CPU fixture检验恢复生命周期：checkpoint读取、迭代器建立、已消费batch跳过、日志/预检与首次Beta样本之间的RNG状态；对比不中断与恢复后的确定样本/tau后缀。fixture必须模拟实际入口顺序，而不只是独立Beta从rng_state重放。不要加载GR00T/权重、做真实backward或占H200测试此事。

已有checkpoint-seeded Beta replay结论直接引用。记录 `ROOT_CAUSE_UNRESOLVED` 或 `CPU_REPRODUCTION_PASS` 等准确状态；发现调用点后仅在未来执行代码中做保持语义的最小修复与针对性回归，不改现有模型或日志。通用保存了RNG文件不足以声明未来exact；必须验证消费顺序。

这条CPU修复不作为本次512格前置条件，除非发现另一个会使模型/评价无效的真实契约冲突。任何新的项目训练和因果重做必须另获PI预算，不套用旧L工程重放授权。

## 7. 边界、预算和工程权限

- 本轮新训练/项目backward/optimizer预算=0。已有C/R4000实际更新永久保留，配对无效不等于零计算，也不自动转成免费可重跑额度；其他历史工程attempt按原账本另计。
- 原screen尚未消费512格，本次显式重新准入这同一批模型×source×noise。不是追加到1024；接手如发现已有有效cells则按完整身份核对并扣减，不覆盖。
- 原baseline128复用；512个完整S4 query内部至多2048次head evaluations，分别记逻辑/新增/复用/工程attempt。CPU复算不算新模型查询。
- 不开放新seed、loss、solver、sampling比例、precision、statistics、target条件、WBC/gate/scorer修改。
- 不开放FULL/CALIB/selector/WBC/sim/DEV/F/DAgger/硬件/merge；若有合格候选，下一PI决定扩量资格和恢复回归，不把本文件解读为自动越过原PI0248终点。
- 工程问题可自主定位、最小修复、回归与恢复，不设任意次数上限；不能伪造或覆盖原配对FAIL，不能用新manifest改写原训练历史。
- 继续邮件关闭、无新monitor；无扩大home/scratch清理或删除权限。保护唯一完整Trainer源及已有合格U；沿成熟HF固定revision/推理视图路径，报告失败不重训。
- 旧文件不覆盖。新screen-admission代码、CPU测试、receipt和结果在独立execution worktree提交。不要为了读本文件修改健康作业staged目录。Issue #11追加新canonical，PR #77仅pointer。

## 8. 最少交付和启动正文

交付集中为：一个deviation manifest、两份as-executed arm receipts、一个固定screen card、512格结果及独立CPU复核、final四组质量表和候选/无合格结论；CPU恢复回归另附状态。不要求先建设新调度/存储框架，不再仅以“继续等待PI”结束。

### PC agent启动正文

执行 `PI0248_AS_EXECUTED_QUALIFICATION_GO`。完整读取本文件、原PI0248协议及5850988664/5851180653/5851224835/5851302733。

原C/R tau配对失效永久保留；本PI明确放行新的as-executed模型资格分析，不是原paired PASS。C/R已有各2000，不再训练或补跑后1000。

先零模型逐臂复核实际4000更新、原samples/loss/梯度/mode/冻结与固定checkpoint inventories，生成新schema receipt及deviation。R原finalizer被跨臂比较拦住，不得伪造旧PASS；必须检查其余独立完整性。C完整CPU随机轨迹连续性false，不用加载过RNG文件冒充exact。

只做薄screen准入分支：原strict入口保持失败，新scope需本授权＋as-executed receipts＋deviation＋原card和模型/条件绑定。不得删旧tau检查或改旧manifest。部署计算与指标函数不变，CPU fixture证明边界后直接执行，不等唯一RNG consumer定位。

Corelab执行原C1000/C2000/R1000/R2000×64×B0/B1共512格；原M64 baseline128复用。使用固定HF revision及实际noise/condition packets，不重建noise，不选别的checkpoint。first-valid保留，工程恢复只补缺失有效cell。

只final2000判资格，全64/old16的B0/B1四组分别零REJECT、medianXYZ≤.020、p95≤.035、medianSO3≤.050。两臂都完成；两者均过沿原简单C优先。1000仅中期配对观察，2000差异不得纯归因旋转loss。

通过封存AS_EXECUTED_M64候选并停FULL准入；都不通过记录现有产物质量FAIL并回PI。selected_checkpoint=NONE；不FULL/CALIB/WBC/sim/DEV/F/DAgger/真机/merge。

CPU恢复生命周期可并行做小fixture回归，不增加项目forward/backward、tau搜索或重训。工程自主小修不限次数，旧失败/计数保持，邮件关闭且不建monitor。下一成果是实际模型质量表，不是又一份RNG排查或准备状态。
