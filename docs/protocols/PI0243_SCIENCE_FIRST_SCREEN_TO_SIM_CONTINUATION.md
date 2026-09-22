# PI0243 — 以科学结果为终点的 screen → support expansion → sim 自主续行

日期：2026-09-22（Asia/Singapore）  
Decision：`GO_A_VERDICT_THEN_CONDITIONAL_B_TO_SIM`  
Authorization：`PI0243_SCIENCE_FIRST_CONTINUATION_GO`  
发布状态：`PLAN_PUBLISHED_NOT_EXECUTED`  
审核基线：`c5348fe6348aad5976e8ba790188c2a37610bfa7`  
Canonical scientific ledger：Issue #11；PR #77仅pointer，不merge。

## 1. 这轮goal是什么，什么不是完成

**近期必须交付：现有Stage-A四组权重在原固定16TRAIN上的完整64-cell结果、step5000的原标准判定，以及通过后真实推进Stage B，或失败后的定量单配方决策包。**

长程沿既有唯一路线：`A verdict → B64/5K → FULL9792/10K → 原selector唯一冻结 → frozen-WBC A–D C0 one-chunk → progress-preserving receding → TRAIN_SIM_CLOSED_LOOP_COMPLETE`。每个有效科学PASS后自动执行原条件授权的下一阶段，不需要反复等待PI。任何有效科学终点FAIL停止下游；不换seed、阈值、模型或配方继续追PASS。

“又做了一次审计”“代码compile通过”“新job已提交”“等待H200”可作为中间状态，但不算goal完成。确无合法可用设备/权限/空间时可以交付精确外部blocker；不得假装能控制队列、无限占用其他用户资源或绕过站点规则。

本文件是PI0243执行补充，不另起科学编号，不扩大训练或评价预算。原分级协议commit `5d6d843235625c416d372cdbed01b185a1e6da5f`（Issue5755883470）、存储续行 `5644d54d2f23ae412a2c4d57441563ed73d66d8e`（5759512929）、screen验收 `3b28b83899eae366136b988dc0e81a1cb9c627ce`（5772413116）继续适用。此补充取消把“627716先取得H200”作为唯一推进路径的执行解释；不取消准入和first-valid约束。

## 2. 已审核证据及诚实边界

已读取最新Issue5773244422、PR pointer5773247689、c5348fe完整参数审计diff，以及该commit下：
- `scripts/pi0243_stage_a_cpu_audit.py`；
- `scripts/pi0243_staged_learning_to_sim.py`的screen/metrics/classifier；
- `hpc/pbs/pi0243_stage.pbs`；
- `scripts/pi0242_train_infer_bridge.py`中实际训练条件构造与条件张量比较实现，供窄补证复用，不重跑PI0242。

账本支持：Stage-A训练627454完成5000步；完整比较projector15/15和diffusion456/456变化，frozen backbone493/493和VLLN66/66未变；5000条消费记录与16个action/mask契约已核验。HF三份checkpoint每份18文件在固定revision内容验证通过。旧627524为NEW_EMBODIMENT重复注册导致的forward前工程失败；627716在账本时点为Q/small、Insufficient ngpus；screen仍0/64。

GPT本次没有SSH、实时qstat、运行项目测试、加载模型、读取远端大文件或NPZ，没有提交/取消PBS。上面是ledger报告，源码核验不冒充实测；执行agent核对原始小证据及live状态后复用。

必须复用的身份：

| 证据 | SHA256或固定身份 |
|---|---|
| A run card | `569b5d4e0e221422621a1b9b05b9fa18970d399f7febed7d278685a0b89fe8e7` |
| training manifest | `d7b719c989192c7d403c1ffdc3f766d99657d7401a9a73a958f5031e527f8be0` |
| 原C_T admission | `830137af8c57dcd7bcb15d6c17b85613dca6946daed3aba026b660a06271f548` |
| HF full-content verification | `ab7b982d42e2dc338f7529869289aaf1a5d6aab09a5485f5d22de81535c1d7af` |
| HF repo / prefix | `leoyuer/pi0243-stage-a-admitted-v1` / `pi0243-stage-a-hf-20260922-retry1` |
| step1000 revision | `814ec7989805adffe153d27cea019c078a55c28f` |
| step2500 revision | `bbeaac695145c5b01faf8c40e12860292eda4b9e` |
| step5000 revision | `2b078bba96e121eaa1b8fdaf4c94c2a5b41db71c` |

完整更新证明优化发生过，不证明已经拟合；HF归档证明保存，不证明任务成功；loss0.0397只引用训练记录。没有eval前，不能判A_PASS、恢复旧失败结论或宣称泛化好坏。

## 3. 一项必要的窄补证：action/mask parity不等于整个conditioning parity

### 审核中实际发现

`audit_screen_contract()`由Stage21 builder创建processor，给dataset设置这个processor，然后把collator的action/mask hash与训练期记录对比。它记录state/image的shape、config及stats身份，但没有逐叶比较真实训练路径与无标签screen路径的pixel/grid、token/attention mask、state及embodiment张量。

因此保留原报告和`PASS_C_T_TENSOR_PARITY`历史字段，接受其**16/16 action/mask数值一致性**，但不能把它扩写为已经完整证明所有conditioning输入一致。这不是观测到C_T错误，也不否定已有训练/归档。无需重新做参数、恢复、全数据树或旧native/helper审计。

### 唯一追加任务（CPU、原16窗、零模型forward）

先检索已有当前binding的更完整conditioning证据；若已经有可验证的真实pipeline对照，直接引用，不重算。否则复用PI0242中的真实pipeline构造和leaf inventory/diff方法，对当前A实际resolved config做一次窄检查：

1. **C_T来源**：当前PI0243实际训练config/官方pipeline产生的processor与batch，不是再调用一次Stage21 builder。不原样运行PI0242/PI0241 hardcoded1000步配置，不调用model构造/Trainer.train。只适配当前run-card字段，沿当前5000步配置提取无标签条件。
2. **screen来源**：当前实际screen入口由相同原RGB/state/language生成的label-free batch。按照pinned模型真正消费的条件keys比较；训练action/target字段是诊断标签，不进入推理。不能把全部训练batch直接传给get_action。
3. 对原16window、同帧/观测offset比较实际pixel_values与grid、token IDs/attention mask、state/其mask、embodiment ID、time/camera ordering及kwargs。相同dtype的离散与确定性张量比较hash，浮点同时记录dtype/cast与差异，不通过扩大容差掩盖真实预处理不同。训练目标action/mask直接引用本轮已通过证据。
4. 生成一份小sidecar `stage_a_conditioning_scope_check.json`，明确FULL_CONDITIONING_MATCH、差异字段或真实缺项；不把action shape/pass硬编码成全部输入PASS。

等价则直接沿现有screen。若确认部署入口偏离原批准的C_T，允许最小修复恢复原意图，冻结新screen execution binding；不改已训练weights/stats/标签/旧card。只有明确非契约结果才标NONCONFORMING并保留、从受影响cells恢复；不能因为成绩差决定它“无效”。若原C_T本身无法唯一确定而需要新的科学语义，停止相关promotion，报精确差异，不自主另选preprocessor。

这项检查与设备准备/B的CPU准备可并行。它只有确定的16窗与输出字段，不得扩张成新一轮helper/teacher-flow矩阵，也不得再次重跑10880专家census或全checkpoint参数比较。

## 4. Goal A1：取得能执行screen的设备，不把等待某个job当科学前置

### 4.1 先接手唯一执行

经本地已有SSH配置核对627716的live状态、owner、脚本/card/hash、已产出records。Q是旧状态，不能假定此刻仍Q。若有更新PI决定或已完成科学结果，先按身份复用。

- **正确版本正在R**：不改staged、不重复科学执行，让其完成；可并行做CPU证据核对与B配置。若发现真实契约偏离，保留输出并阻止错误结论。
- **仍Q且0有效cells**：检查Corelab既有成熟环境和真实可用单卡显存/RAM/磁盘；优先在Corelab完成小规模screen。若准备已齐且能运行，先合法hold/取消这个自己拥有的冗余queued job，并确认不会启动，再开始Corelab第一条正式cell。不得先重复推理再决定保留哪套结果。若取消时变R，以真实状态防竞态。
- **Corelab确实装不下/不可达/被合法占用**：记录具体设备/容量或权限证据。可以沿站点已授权的H100/H200路径申请适配推理负载的资源，H200不是模型科学必要条件。没有合法替代就继续唯一队列并回报外部约束，不新开并发重复job。
- **已有终态或部分cells**：读取真实产物；有效结果冻结，缺失的按剩余列表恢复，不从头重跑。

现有PBS源码统一使用12CPU/225GB/48h，并限制H100/H200。它只是模板，本次无法从GitHub证明627716最终qsub是否覆盖了资源。执行agent应读取实际`Resource_List`。screen用一个已获准GPU、无backward/optimizer，所需资源依据既有PI0240/0242推理遥测和真实load峰值规划；不得未经测量指定某个“小内存必够”。也不能因为模板写了H200就否认Corelab路径。

只调整科学计算外的合法资源/host binding，不改BF16、分辨率、H40、sampler、attention实现、量化或model容量来凑显存。第一条原定screen cell兼作实际显存/推理集成，计入64；不新增试运行seed/window。完整有效但REJECT仍是有效结果，不因为失败换设备再跑。

### 4.2 数据与存储不再阻塞一个纯推理screen

训练和完整HF归档保持不动，不复制optimizer到GPU，不重新上传已有匹配权重。优先复用目标主机已有完整树/模型副本。若缺失，按固定HFrevision与已验证full inventory仅物化模型、配置、processor及必要16窗输入；scope标为MODEL_ONLY，不冒称full Trainer恢复点。

当前`verify_receipt()`要求full Trainer目录，因此model-only路线必须使用或最小增加基于同一完整manifest的推理resolver，明确验证index中每个shard/config的内容；不删除检查、伪造optimizer或以任意local路径替代权重。若现成full tree可用，直接沿原读取，不为了“完善架构”强制改resolver。

CPU单测和真实轻量I/O在本地/Corelab，Hopper login只做站点允许的轻量检查/传输/调度。不在H200反复调Python、NEW_EMBODIMENT注册或FFmpeg。registry修复只允许同一受绑定定义幂等复用，定义冲突仍fail；不要全局忽略注册错误。

没有在Hopper写入的Corelab评估，不因Hopper旧quota问题阻断；实际输出主机的容量/缓存仍需保证。不得扩张清理旧模型、覆盖canonical或删除reader依赖。

## 5. Goal A2：一口气完成64个原定cells并生成判定

固定顺序0→1000→2500→5000，每组原16TRAIN。一个模型顺序加载，完整原stock推理，seed22301、语言/StateV1/dense-relative H40/132slots/原stats/row-rot6d/dt0.05不变，target/专家未来不进入模型。

首个有效cell前绑定backend；尽量整组64在同一设备/runtime完成。跨平台数值不自动逐bit相同；只为资源选择backend，不为成绩选backend。确有运行中基础设施故障需要迁移时遵守原first-valid规则，保留host差异及推断限制，不选两套中的更好结果。没有真实受影响内容变化，不再重复训练/参数/HF整体验证。

每个cell运行时保存raw、decoded、专家动作、state/RGB、mask、gate全部issues、权重/processor/代码/backend/cohort身份和NPZ/hash。复用记录之前、在dict按source_id折叠之前做duplicate/conflict/身份/finite/完整key集验证；额外键、缺行、坏NPZ不伪装通过。通用mock只验证程序，不进入科学统计。

早期REJECT/高RMSE正常记录并继续后面checkpoint。不能把baseline失败、1000未达标或2500变差当A最终失败。形状/数据损坏等为工程或契约异常，按具体证据处理；不吞掉异常，也不把模型差当工程理由重试。

### A唯一最终标准，完全沿原协议

每窗：XYZ_RMSE=sqrt(mean_t(sum_xyz((pred-expert)^2)))；SO3_RMSE=sqrt(mean_t(geodesic(Rpred,Rexpert)^2))。全40点，跨16窗使用NumPy linear percentile。

A_PASS条件：四组64cells完整合法、训练/输入身份成立，且**step5000**同时满足：
- 原SafetyGate REJECT=0；WARN保留，不另加否决；
- median逐窗XYZ_RMSE≤0.020m；
- p95逐窗XYZ_RMSE≤0.035m；
- median逐窗SO3_RMSE≤0.050rad。

固定生成4行结果表：checkpoint/N/ACCEPT/WARN/REJECT/各主要issue/median与p95 XYZ/median SO3/最差窗；附16窗配对变化和A–D×phase侧报。均值XYZ误差不代替RMSE；reference距离不代替expert误差或achieved TCP；loss下降不代替任务评价。不做最佳中期选择。

原32个direct-flow侧报（step0/5000×原R1 bridge4×四t×C_T）只在既有预算内补或按完整身份复用，使用corrected processed-once/native-helper路径，禁止新诊断arm。64cells完成后可立即计算A主判定；sidecar是独立并行的解释产物，不参与阈值、排名或选模，不能成为无边界诊断工程。完整A主判定及身份成立即按原授权推进B；sidecar不得被取消，也不得通过训练进程重置RNG生成。

## 6. Goal B-ready：排队/CPU阶段并行准备，不提前开训

只准备当前下一步，不阻塞A screen去实现所有远期架构：
- 从原cohort锁定64TRAIN及16子集；固定5000步B曝光表，不按A结果选样。
- 起点绑定A-final5000的完整内容hash；新optimizer/scheduler，不复用A optimizer。建立独立B family/HF prefix/输出根；修正当前通用代码里A-family字面值的stage绑定，不能覆盖A archive。
- 原native loss、projector/action head范围、冻结模块、peakLR1e-4、batch1/accum1、原BF16、cosine5000/warmup10、seed22301、C_T/statistics不变；未列字段按原PI0243实际resolved值。
- B保存1000/2500/5000，screen0/1000/2500/5000×64；原16记忆保持标准不变。资源按真实训练峰值而非推理峰值；Corelab适配原recipe则优先，实际不够再H100/H200。
- 保存/发布/收据/恢复工具复用已解决的路径与runtime；不因为需要B归档重新触发A扫描、上传和清理。

A尚未判定时，B最多`READY_NOT_STARTED`，没有B backward/optimizer。

## 7. Goal B onward：科学PASS就继续，不回到逐项等授权

### A通过

将A判定与64cells小型manifest封存到Issue #11，随后自动运行原B64/5000。A-final是B initializer，不是WBC候选，selected_checkpoint仍NONE。不要只回复“PASS，等PI确认”。

B-final整体64与原16子集均满足原四项标准→原FULL9792/10000；FULL独立新family的2500/5000/7500/10000→baseline+4候选各1088CALIB的一次完整评价→原selector唯一winner→winner自身TRAINscreen准入→原A–D C0 one-chunk→progress-preserving receding。

C启动前才必须验证train view9792/screen view64分离、真实1000 model-only snapshot、1000与2500早期检查能够在继续训练前实际触发，不是在10K结束后追认“早停”。这是未来stage实现原契约，不是现在加做全量测试。不得将A/B质量门省略以追求尽快进sim。

原selector/轨迹gate/WBC/0.035m任务scorer/cadence/termination不变；sim实际参数必须从既有canonical继承并预先绑定，不现场调参。正确物理任务失败停止；不得多seed试到成功。终点仍为`TRAIN_SIM_CLOSED_LOOP_COMPLETE — STOP_FOR_PI_GENERALIZATION_REVIEW`，不声称multi-skill general model。

### A有效失败

完成并封存`SCIENCE_FAIL_M16_BUDGET — STOP_FOR_PI_SINGLE_RECIPE_DECISION`，停止B/C/sim。只用已完成screen、现有训练日志与原授权direct侧报给定量归因：
- 位置/姿态整体仍有较大expert误差；
- 位置已改善但速度/起点等gate仍失败；
- 少数window或组的尾部主导不合格；
- native objective / free-running指标是否同向改善，以及无法确定的机制。

这些分类可重叠，不硬编码“欠拟合/泛化”。新5K的step1000与旧PI0241-1000具有不同cosine时程/运行身份，不能把相同step号当严格单因素对照。只在同cohort同度量且解释条件差异时给描述性比较。

交付一个具体、唯一、`NOT_AUTHORIZED`的下一科学干预建议；不得预先认定需要平滑loss、重加权、FP32、更大head或更多步数。不追加A训练、不改阈值、不选2500代替5000。没有新增科学证据前，不再发布下一训练recipe。

### 工程与外部blocker

路径/import/依赖/IO/调度/报告代码错误：自主最小修复→与该根因对应的CPU或真实轻量回归→恢复缺失cells，无固定修复/qsub/token上限。不能无限重复同一失败而不新增根因证据，不能新增科学样本。健康的stage结果按身份复用。

资源真不可取得时报告Corelab现状、Hopper合法路径、具体阻塞与下一可执行动作，状态`EXTERNAL_RESOURCE_BLOCKER`；排队不是科学FAIL，也不是goal完成。无需重建monitor或让GPT承诺后台工作。

## 8. 最小执行产物与禁止事项

产物尽量归并为：
1. 既有四份manifest的hash引用＋唯一conditioning范围补证；
2. 一份`screen_execution_binding.json`（host/实际资源/owner/剩余cells/已验证代码和输入），不用新通用orchestrator；
3. 64cells、4-row科学表、A判定；
4. PASS后的B实际运行记录，或FAIL的原预算侧报和单配方建议。

不反复全库扫描；不重跑旧142、10880、5440或256诊断；不重新训练已完成A；不重新上传/下载全部optimizer；不扩大旧权重清理；不为了出新报告改变sealed evidence。缺失确有影响的内容可以按需恢复，不以“推进科学”为理由跳过实际损坏。

同stage只有一个有效owner；first-valid scientific结果保留。PR77不merge，DEV-E/F、Final-F、DAgger、硬件仍锁定。CALIB与sim仅到达原D/E/F阶段的条件门后才可执行；不是本screen自动提前解锁。

本文件发布仅新增文档和ledger，不修改真实screen或训练代码，不证明runtime可达、不取消/提交job。agent按此goal执行时的每项实际操作需写入Issue #11，PR #77只pointer。

## 9. PC agent启动正文

执行PI0243_SCIENCE_FIRST_CONTINUATION_GO。先读本文固定commit、Issue5773244422、原PI0243分级goal5755883470。目标是完成A科学判定并按PASS自动推进B，不是再交一次“审计通过/仍在排队”。

复用c5348fe及已有完整参数、消费、action-mask和HF证据，禁止重训A或重复大审计。当前C_T报告在所审源码中只核验action/mask；若没有独立condition证据，只用原16窗CPU比较实际A训练pipeline与实际screen的label-free条件。复用既有PI0242方法但使用当前A真实config，不运行旧诊断主程序。必要时最小修screen恢复原C_T，保留旧记录，新execution binding，不改训练/配方。

核对627716 live状态与产物。正确R就复用，不重复；仍Q/0cells则优先准备Corelab成熟单卡推理路径，能跑时先hold/取消自己的冗余queued job并确认不再运行，再执行唯一screen。Corelab不够则用已授权H100/H200合法推理资源，检查实际Resource_List，不照搬225GB/48h训练模板或声称H200是唯一科学入口。条件检查与设备/B的CPU准备并行。

按0/1000/2500/5000×原16完成64cells。首个预定cell兼smoke，合法结果冻结；原seed/dtype/H40/sampler/gate保持。记录与复用先核对完整身份、duplicate/NPZ/hash，不以source_id相同就skip。早期REJECT继续，不重复挑成绩。只按step5000原四项标准判A：REJECT0、median XYZ RMSE≤.020m、p95≤.035m、median SO3 RMSE≤.050rad。完成原32 direct侧报或引用，侧报不改变判定，不新建矩阵。

A_PASS及主证据完整→直接启动原B64/5000，A-final5000 weights、新optimizer/scheduler、独立family/prefix；B/C/selection/one-chunk/receding按原条件自动推进。A有效FAIL→完成既定证据、给一个未授权的下一配方建议并停科学门。不追加A步数、不改loss/LR/精度/阈值、不挑中期。

工程自主修复无次数上限，本地/Corelab调试，不在H200反复pytest。真实外部资源缺口如实上报；不以已完成审计为由重新找问题扩展范围。selected_checkpoint在A/B保持NONE，DEV/F/DAgger/硬件/merge不开放。Issue #11先canonical、PR #77只pointer，不重建monitor。

## 通用外部约束（不替代项目冻结实现）

PyTorch官方Reproducibility说明不同平台/版本不能保证完全相同的结果：https://docs.pytorch.org/docs/2.13/notes/randomness.html 。本计划只据此要求明示host/runtime身份，不升级本项目pinned版本、不因设备变动改评分或许可成绩驱动重跑。
