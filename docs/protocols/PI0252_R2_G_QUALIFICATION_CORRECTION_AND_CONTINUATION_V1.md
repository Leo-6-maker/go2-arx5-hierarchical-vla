# PI0252-R2 — 纠正跨臂停止判断，冻结合格G，继续TRAIN闭环与FULL微调

日期：2026-09-28（Asia/Singapore）。

Authorization: `PI0252_R2_G_QUALIFIED_SIM_FULL_CONTINUATION_GO`
Decision: `CORRECT_AG_ADJUDICATION_FREEZE_G_CONTINUE_EXISTING_SIM_FULL_CHAIN`
发布状态：`PI_REVIEW_AND_AUTHORIZATION_ONLY_NOT_REMOTE_EXECUTION`。
Issue #11为canonical scientific ledger；PR #77仅追加pointer，保持draft/unmerged。

## 0. PI裁决：A失败不否决G，旧总STOP与预注册规则不符

本次读取最终报告canonical 5866137482及PR pointer 5866140827，并回读原PI0252-R1不可变协议：

- Commit: `cef7dadc2fb60304d39057dc3e720bc5b6210061`
- Path: `docs/protocols/PI0252_R1_SPEED_MARGIN_TO_FULL_FINETUNE_V1.md`
- 授权canonical: 5862818190。

该协议第4节明确规定：

> 只有一臂四组全部PASS则取该臂；均PASS预先优先简单A；均有效FAIL则 PI0252_R1_SPEED_MARGIN_HYPOTHESIS_EXHAUSTED。

报告中的A all64/B0有效FAIL保持；G all64/old16的B0/B1四组全部PASS也保持。二者不能互相改分，但原协议允许从合格臂中取唯一候选。因此，报告以“A失败、G不能抵消A”为由将整阶段判成 `PI0252_R1_M64_VALID_FAIL_STOP`，属于跨臂晋级规则的误用，不是新科学负结果。

本PI接受已报告的逐臂质量结果，并依原规则明确指定：

- `arm_A_quality = VALID_FAIL`
- `arm_G_quality = VALID_PASS`
- `qualified_arms = [G]`
- `development_checkpoint = G2000`，完整manifest由已有原件机械封存。
- `formal_selected_checkpoint = NONE`，正式FULL/CALIB选择尚未执行。
- 正确阶段决定：`PI0252_R2_M64_G_QUALIFIED — GO_PRE_FULL_TRAIN_SIM`。

这不是放宽阈值、池化两臂、豁免A、挑选1000点或根据新sim结果追选。原规则早于本次结果已经指定“单臂通过即可选该臂”。本文件纠正总裁决并确认续行；不新增M64训练或评估预算，不发明新loss。

原R1第5–10节的成功路线继续有效：G的扩量前TRAIN闭环通过后自动FULL，不再要求A也质量PASS。本文件只修正总停止和候选绑定；R1的有效科学失败、完整性门和资源边界仍适用于各自依赖阶段。

## 1. 本次依据、数值复核与边界

最近执行canonical发布时间：2026-09-28 08:17:18 UTC，即16:17:18 SGT。随后Issue增量读取未返回更晚记录。本次核对的PR元数据为open/draft/unmerged。这些是GitHub观察，不是实时PBS或Corelab进程查询。

已读取的执行证据：
- 5864689020：A2000完成及逐臂训练完整性、完整HF读回。
- 5865706270：A128格有效结果；G作业636296完成、2000更新、2000 paired-identity/step-RNG/mode记录及完整归档。
- 5866137482：完整A/G最终8组、256 first-valid cells、配对评估输入闭合和两臂独立CPU复算。
- 5866140827：旧总STOP的PR指针。
- R1原不可变协议第4–10节；实际G0归因与gate分层约定继续继承。

### 1.1 公布的最终数值，不改变任何一项

| 臂/组 | valid | REJECT | median XYZ m | p95 XYZ m | median SO3 rad | 按原门复核 |
|---|---:|---:|---:|---:|---:|---|
| A all64/B0 | 64 | 1 | .0146616122 | .0309444734 | .0157844238 | FAIL |
| A all64/B1 | 64 | 0 | .0140486770 | .0295327830 | .0165282802 | PASS |
| A old16/B0 | 16 | 0 | .0127084264 | .0240943934 | .0129820949 | PASS |
| A old16/B1 | 16 | 0 | .0131468068 | .0243767301 | .0161644703 | PASS |
| G all64/B0 | 64 | 0 | .0148080560 | .0314542320 | .0162741860 | PASS |
| G all64/B1 | 64 | 0 | .0140809919 | .0305526907 | .0172145644 | PASS |
| G old16/B0 | 16 | 0 | .0125458324 | .0251005573 | .0141178820 | PASS |
| G old16/B1 | 16 | 0 | .0125797434 | .0264270528 | .0171580431 | PASS |

单臂四组依然分别要求：完整有效、零REJECT、median XYZ<=.020m、NumPy linear p95<=.035m、median geodesic SO3<=.050rad。G全64的p95余量分别3.545768mm与4.4473093mm。old16是all64重叠子集，两组噪声不是独立训练重复。

本GPT使用上述已公布十进制数复核了8组资格和跨臂真值表；没有重新计算256个原始NPZ、加载项目模型、执行项目runtime测试、SSH/PBS、HF完整权重下载或sim。原件/独立CPU通过由执行账本支持；下一执行agent复用已有收据完成定位，不凭本文数字倒填原件。

本次没有读取实际最终汇总代码的运行字节，故确认的是“报告总裁决违反规则”；究竟是人工解释还是程序用了错误AND/any-fail，要由第2节定位，不冒称已发现某一行实际源码bug。

### 1.2 候选及证据固定身份

G2000：
- PBS job: `636296.hopper-m-02`，已报告Exit 0，2000 canonical updates。
- HF revision: `efecea485ae77000599d19171b02491f8d514f42`
- 完整tree SHA256: `cfbfeb54bea2b5ac266047dfd764cfa6b041a8a77b9688737ea70b56935d5226`
- G run card SHA256: `98c0cd42ee7d2ff522634539047bc5933dba8929d839a768909d4de79d9cdc84`
- Screen report SHA256: `99acd5fa3282960f862be210f09088d5d670912c494fd048a253ea0c1dc06262`
- Score card SHA256: `4eb44674e7cb1d82ff1ddc7d3c397f5e65e6db4f1068cf509eacfde5c49fb1e4`
- CPU replay SHA256: `e645d33c96e657c0fd20198ebc5af429bf2a8b93ef28893422031e7a3f487307`

A2000对照保留：revision `2a7945d49d17f3ec1801f2b2af6dba5e5bd6c100`；tree `f58ec02d44f1c250936d1b9687930669f12b4c2f6044c0df2c38f39dd36f379c`；screen `d43072c7a0ad2e52a983409fbbeedfd0d6c740b5095cf181b2c3ac8223787583`；CPU replay `2be2a7c3cf359d00f04fada2e930668355d65f48b1f3d44ecac0e9941402edb3`。

旧总verdict：JSON SHA256 `15434f070b9e27610e78ebf0a17c951736d9cac20e2d1ec3c318563b2c3bde47`；MD SHA256 `8d5ba388f5941e9ce4c540ba0530a76f2cce320f01cc5a2d006590bd435d5ecf`。Corelab证据根 `/home/liuyu-nfs/Projects/go2-arx5/PI0252/evidence/`。

repo/prefix和各文件路径从固定durable receipt解析，不凭revision或文件数猜下载路径。若获准存储迁移，保留旧到新固定revision的全内容映射。

## 2. P0立即执行：只纠正裁决和自动推进入口

### 2.1 零新模型调用的收据闭合

从已有原件核对旧verdict、A/G各自完整性、G实际2000步配对/guard收据、两臂128格CPU结果、共同source/noise grid及G模型/processor/统计身份。已通过且未改变的原始数组验证直接复用，不再跑旧256格或重做整库审计。

明确区分：训练sample/tau/epsilon/stock配对收据，与评估source/B0/B1 packet配对收据是两件事，不用其中一个冒充另一个。5865706270已报告G有2000 paired-identity行；补可读定位和hash即可，不要求重新做2000次forward。缺少摘要字段可从实际日志追加；真实输入/恢复/权重冲突则准确停完整性门，不伪造PASS也不自动重训。

独立写：
- `PI0252_R2_ADJUDICATION_ERRATUM_V1.json/md`
- `PI0252_R2_G_DEVELOPMENT_CHECKPOINT_V1.json`

erratum记录 `supersedes_decision_only`、旧JSON/MD hash、原R1第4节引用、8组不变结果、A_FAIL/G_PASS、唯一G和后续准入。旧文件/comment不覆盖、不删除；追加更正并保留历史STOP作为审计事件，而非继续有效的科学结论。原5865706270中“A失败故不能进入下游”的解释一并指出适用限制；不得改写其实际数字。

### 2.2 恢复实际控制流程，不能只写一份说明

定位真正生成总verdict或阻止下一phase的命令/函数/agent规则。若程序正确而错误只是人工解释，修正判读与continuation状态；若程序要求A和G均质量PASS，按以下原规则薄修，提交实际使用的代码与测试。不重新设计统一评估平台。

概念逻辑：

```python
# Before this point: both arms are complete, valid and satisfy existing integrity receipts.
qualified_A = all(four_frozen_group_passes_A)
qualified_G = all(four_frozen_group_passes_G)
if qualified_A:
    chosen = 'A'       # sole A or pre-registered both-pass tie break
elif qualified_G:
    chosen = 'G'
else:
    chosen = None      # both effective quality FAIL
```

四组内用AND；跨臂候选存在性用OR。`any(valid_group_failed)`不能触发整个campaign失败。完整性/原件有效性与模型质量分开；INCOMPLETE/INVALID不能默认为质量FAIL并进入上述选择。

CPU回归应调用实际汇总/路由函数，至少覆盖：A_FAIL/G_PASS->G；A_PASS/G_FAIL->A；两PASS->A；两有效FAIL->原假设耗尽；任一原有完整性未闭合->INTEGRITY_STOP；缺失/重复组、NaN/非有限、数量不足不得PASS。用当前报告的真实8组复现A_FALSE/G_TRUE/CHOSEN_G。原阈值精确边界<=仍沿原函数，不另引入容差让越限通过。

这不是增加新的模型资格门；是确保原门和原路由被正确执行。P0完成且已有原件一致后自动进入P1，不再等待一个“允许G抵消A”的审批。

## 3. G已具备什么、还缺什么

将G2000固定为唯一开发候选。原128格是训练支持上的资格证据，不是正式CALIB选择或新任务泛化。保留WARN、最大值和初始连续性记录；零REJECT不证明所有速度都<=1.8m/s，因为1.8只是训练margin而非验收门。

现有配对结果支持“这份G产物达到原联合资格，而A没有”。G部分位置/姿态聚合数值略高于A，不能宣称所有指标均优。是否具体修好了历史obs-0073、第7->8段，以及A的现存REJECT是否仍来自它，可从已有A/G record窄读侧报；不增加查询，不把历史峰值直接当当前值，也不把机制侧报作为新的模拟前置研究。

第一次学习策略sim前封存：G weights/revision/tree、原C祖先、实际loss、S4与原processor/TRAIN stats、完整离线gate profile、实际输入/噪声身份、四组质量与训练配对证据。不得切换到A、旧C/E/X、1000点、权重平均或新的solver。

G完整loss固定：

`J_G = L_native + .4*ell_pos + .1*ell_delta + .1*ell_rot + .1*ell_speed_margin`

`ell_speed_margin = relu(max_j(sum_xyz((p[j+1]-p[j])**2)/(2.0*dt_j)**2) - .81)`。

保持实际dt=.05、39段最大值、完整S4四步梯度、原rotation镜像、统计/精度/冻结范围。当前阶段不再优化G；这是描述已完成配方并为FULL继承，非M64续训授权。

状态分别记录 `development_checkpoint=G2000`、`formal_selected_checkpoint=NONE`、`pre_full_sim=NOT_RUN`。不得因formal selected仍NONE而循环阻止已授权的development sim。

## 4. P1：现在运行扩量前TRAIN闭环，而不是再训练小模型

沿原R1第5节和PI0226 canonical5733559806恢复A–D C0场景/seed、实际WBC源码/权重、frame/TCP、StateV1、持续reference入口、cadence/prefix/tick/query cap、物理时钟、termination/cleanliness及world-TCP scorer。复用已有场景和环境，不扩成新benchmark。

离线screen、CALIB和sim各自绑定原作用域profile。原R1已纠正离线初始拒绝绑定为.75m/1.50rad；不能将它复制到本来不同的sim gate，也不能把旧错误.10m/.70rad强加到sim。同profile身份一致即可；真正无法恢复的sim配置冲突只阻塞依赖的物理阶段，CPU full准备可以继续。

先进行最多4个原TRAIN native expert控制episode，完全相同runtime身份已有有效控制可复用。控制失败先辨认环境/接口问题，不改WBC或阈值救模型，也不将其写成G模型失败。

然后唯一G按任务A->B->C->D各一次one-chunk，共最多4次模型查询。注意任务A–D不是训练臂A/G，不能复用`arm`字段混淆。真实RGB/StateV1 -> S4 dense-relative输出 -> 原sim gate -> TrajectoryBatch -> frozen WBC。REJECT不下发、不重采样。

4/4 one-chunk全部达到原actual world-TCP距离<=.035m及完整时长/cleanliness才进入progress-preserving receding。A–D各一次，每任务模型查询<=64且继续遵守原更严格tick/query上限，总<=256。每次用实际执行后的fresh观测，reference时钟和进度连续，不reset、不重复播放旧预测、不teacher prefix/RTC/平滑/blending/retiming。

同次记录保存实际noise张量、模型reference与achieved TCP、tracking/first-hit/min/final/termination及视频。既看生成reference是否能到目标，也看WBC执行是否跟上，但不增加评分重试或根据结果换候选。同步sim时钟不等于真机实时性能。

科学有效的G任务FAIL停原 `PI0252_R1_PRE_FULL_TRAIN_SIM_FAIL`，不救A或改选别的模型。工程环境/传输/报告故障只按原语义修复并保留第一次有效物理结果。全部通过封存 `large_scale_init_checkpoint=G2000`、`PRE_FULL_TRAIN_SIM_PASS`并直接进入P2；无需新的PI批准。

## 5. P2：单个FULL_G family已获条件续行，不重置原预算

只在P1通过后启动 `PI0252R1_FULL_G_V1`；不启动FULL_A，不新建第三个M64臂。原R1第6节精确配方继续有效：

- 从同一G2000 weights warm start；新AdamW/scheduler，不将此称为G旧optimizer的exact resume。
- 既有seal的9792 TRAIN窗口，原source-rollout split和A–D范围；1088 TRAIN-CALIB仍隔离，不用于梯度、统计或改配方。
- 排序后PCG64(22301)逐轮排列，9792完整一轮+下一轮208，共10000行；原mask/表示/dt与目标输入边界不变，不重加权old16或困难source。
- Canonical 10000更新，batch1/accum1/workers0；AdamW LR1e-5、WD1e-5、betas(.9,.999)、eps1e-8、maxgrad1；cosine10000/warmup10。
- 完整继承J_G，包括.1速度margin项；不丢rotation项，不加E超额项，不改S4、参数dtype/autocast或冻结模块。
- 保持已批准FULL RNG key：`PI0252R1|22301|stage=FULL|optimizer_step=<1-based i>`，按原SHA256前8bytes大端/63bit规则，原随机隔离及恢复cursor。不因R2改名重置namespace。单FULL臂不伪造跨臂pair receipt。
- 正式完整Trainer checkpoints为2500/5000/7500/10000，全部durable保留。1000仅原model-only观察，不是选模候选或完整恢复点。

可以与P0/P1并行进行纯CPU的FULL card/路径/表/容量准备，但P1未通过不能开始FULL optimizer。真实入口要验证10000终点、G initializer/J_G、保存回调和恢复逻辑，不能沿用硬编码A/G2000入口只换名称。completed M64 cards/日志不改；本continuation authority用独立字段/伴随收据绑定，不触发旧card失配重训。

FULL保持矩阵仍为640逻辑格：B0在0/1000/2500/5000/7500/10000各64，正式四候选B1各64。step0/B0的64从本次G有效screen精确身份复用，最多576新查询；old16直接提取。Corelab screen/归档与Hopper训练可按原规则并行，2500保留既定barrier。

普通中期质量FAIL不得误用M64路由来提前中止FULL：只按原1000和2500的B0均64/64 LINEAR_SPEED_REJECT的系统失稳规则停；非有限、真实训练语义/恢复完整性问题另按对应门停止。不得看到较好中期结果就删候选或改变训练终点。

## 6. P3/P4：正式选择和post-FULL闭环

继承原R1第7–8节。FULL四候选完整后，原PI0226 adapted1000 baseline仅作对照，加2500/5000/7500/10000，每份原1088 CALIB，共最多5440；基线只有完整身份一致时可复用。原source集合、target authority、S4、真实noise、processor/stats和gate profile封存；目标XYZ用于scorer，不进入模型。

成熟selector仍对每个候选分别检查完整且零REJECT，再按median reference min-target-distance、worst source-rollout-cell median、较低step排序。必须恰好1088唯一source及有限完整结果；不把expected count改小。一个FULL候选FAIL不否决其余合格候选；只有无admissible candidate才NO_ADMISSIBLE_CHECKPOINT。不可改成“所有四候选都必须PASS”，也不可用本次G开发选取代替正式selector。

S3 provenance lock、S4 formal selection freeze、S5只读诊断。正式winner自己的all64/old16 B0/B1保持门通过才post-FULL sim；保持失败则保留选择记录并sim_admitted=false，不改选runner-up。

post-FULL使用同一TRAIN任务/控制器/评分，最多4 one-chunk及256 receding；4/4 one-chunk后才receding。模型已改变，不能复用G的学习策略轨迹冒充FULL输出；expert环境控制可严格按相同runtime身份复用。有效物理FAIL不换checkpoint/seed/控制器。

最终分层交付：`FULL_TRAINING_COMPLETE`、`FULL_SELECTION_COMPLETE`和原 `PI0252_R1_FULL_FINETUNE_SELECTED_TRAIN_SIM_PASS`。前两者不替代最终闭环；成功后停泛化审核，DEV-E/Final-F、DAgger、真机及merge仍未授权。

## 7. 预算与工程自主边界

A/G训练4000 canonical更新和M64 screen256已经消耗并封存；本R2不增加或重发它们，P0新增项目模型查询/优化更新为0。

从当前未进入下游的状态起，剩余：

| 阶段 | canonical optimizer | 新模型评价上限 |
|---|---:|---:|
| 裁决更正、CPU测试、manifest | 0 | 0 |
| pre-FULL TRAIN sim | 0 | 260；expert<=4 episodes |
| 唯一FULL_G | 10000 | 训练native/S4/head/重计算实测另记 |
| FULL保持screen | 0 | 576新格/640逻辑格 |
| CALIB | 0 | 5440，合法精确复用扣减 |
| post-FULL TRAIN sim | 0 | 260；expert<=4，优先复用 |

即剩余canonical训练最多10000、离线scored query最多6016、sim model query最多520；不是另加原14000和6272全量预算。FULL非选分parity至多原批准的2次未用额度；不得重发已完成A/G的parity或新optimizer smoke。所有旧attempt、实际updates/queries/head calls保留raw账本。

工程最小修复、路径/报告修复、同完整Trainer状态恢复自主，不设任意重试次数上限；但阶段终点固定。有效FAIL和first-valid结果不重跑，不重新抽noise追成功。已完成训练的上传/finalizer/元数据问题不重训。真实缺证或配置冲突不靠虚假PASS放行。

PC控制与代码；Corelab做CPU、screen和sim；确需大显存训练才使用已验证Hopper合法PBS auto资源。用合法GPU型号资源满足已封runtime，不pin节点、不重复已知4080S训练OOM、不在H200反复调CPU脚本。qsub之前查真实job状态；查询不可用记UNKNOWN，不能当无作业；提交超时先去重。

保持现有scratch Projects布局、兼容链接、X未封存保护源、G完整源和唯一证据。没有新大清理授权。按原可恢复全内容验证和滚动归档规则退役本Goal新副本；100GiB是准备目标，不是每次保存后必须维持的free下限。每次保存按实际next-write/重叠temp/cache/log+reserve>=10GiB规划，quota未知如实标注，不以df冒充quota。

HF权限/公开可读与模型质量分别记状态，当前durable内容验证不被旧HEAD/metadata代替。邮件及monitor保持关闭；Issue #11追加canonical，PR #77只pointer、不merge。

## 8. 下一交付与启动指令

下一个科学交付应是G在原TRAIN场景中的实际reference/achieved TCP和闭环结果；不是再次A/G训练、又一轮scorecard或仅宣布另一份授权已准备。P0只是窄裁决修复，完成即可进入已授权P1。

启动正文：

执行 `PI0252_R2_G_QUALIFIED_SIM_FULL_CONTINUATION_GO`。完整读取本文件及原R1第4–10节。A2000有效FAIL保留、G四组PASS保留；纠正旧总STOP，按原单臂合格规则固定G2000，不是放宽门或用G把A改成PASS。

复用既有A/G训练完整性、实际训练随机配对、256格及CPU复算。只补原件索引、erratum及真实路由CPU回归，不新模型查询、不重训。确认实际汇总函数能处理A_FAIL/G_PASS->G，完整性失败另停。G revision `efecea485ae77000599d19171b02491f8d514f42`及tree按第1节封存，repo/prefix从receipt取。

冻结唯一development_checkpoint=G，formal selected仍NONE。立即按原scope执行pre-FULL TRAIN one-chunk4/4->receding；合格后自动FULL_G9792/10000，继承完整J_G与原key/时程。再原CALIB选择、winner保持、post-FULL闭环，无需新的例行GO。

不把“所有训练臂/所有候选都PASS”当晋级条件，也不把局部group失败掩盖。不同层的规则严格分开：单候选四组AND、候选可用性OR、物理任务A–D各阶段4/4。科学有效FAIL停对应后续链，工程同状态修复自主。禁止新M64分支、改gate/WBC、补采样、换seed/候选救失败、DEV/F、DAgger、真机、merge或新monitor。
