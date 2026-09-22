# PI0243 — Stage-A screen验收与后续连续推进

日期：2026-09-22（Asia/Singapore）
决定：`GO_EXISTING_STAGE_A_SCREEN_WITH_CONTRACT_VERIFICATION`
授权：`PI0243_STAGE_A_SCREEN_ACCEPTANCE_GO`
发布状态：`REVIEW_PUBLISHED_NOT_EXECUTED`
审核基线：`4d5c173757502a664629937dfaef7ea87d5ce33e`
Issue #11 canonical；PR #77只作pointer，不merge。

## 1. PI决定和本次证据边界

接受Stage-A 5000步已完成的账本事实，不因后续HF/PBS异常再跑训练。下一目标是把现有四组权重的64个原定screen cells变成可信的A阶段验收，然后按原PI0243条件继续B。训练loss约0.0397不决定PASS，不据此改阈值、预判成功或扩大数据。

原科学协议：commit `5d6d843235625c416d372cdbed01b185a1e6da5f`，Issue comment5755883470。存储续行协议：commit `5644d54d2f23ae412a2c4d57441563ed73d66d8e`，comment5759512929。两者继续适用，本文件不新开科学实验、不追加A训练步数、不改变原验收标准。

本次已读取：Issue5771889442、PR pointer5771892480、训练/归档过程5771391642和5771735878、审核基线的archive closure、`scripts/pi0243_staged_learning_to_sim.py`、`scripts/stage21_same_scene_groot_inference.py`、`scripts/pi0243_checkpoint_persistence.py`、`scripts/pi0243_hf_archive_tree.py`、`hpc/pbs/pi0243_stage.pbs`。Hugging Face connector确认公开仓库metadata可见；网页端未能读取固定revision文件。因此GPT没有独立读取HF大文件、远端run card、完整训练manifest或NPZ，也未SSH/qstat、运行pytest/模型、取消/提交作业。作业Q是账本时点状态，不是GPT实时scheduler观测。

### 冻结定位信息

- 有效训练：`627454.hopper-m-02`，账本报告hopper-22，5000/5000；后续sidecar运行时失败与训练完成分开记账。
- 原screen：`627524.hopper-m-02`，账本报告auto→small/Q。
- run-card SHA256：`569b5d4e0e221422621a1b9b05b9fa18970d399f7febed7d278685a0b89fe8e7`。
- HF：`leoyuer/pi0243-stage-a-admitted-v1`；prefix `pi0243-stage-a-hf-20260922-retry1`。
- 原cohort16与原seed22301不变；selected_checkpoint=NONE。

| step | 报告完整payload bytes | canonical tree SHA256 | HF manifest revision |
|---|---:|---|---|
| 1000 | 20889599778 | fbc0131175b72bfe8cd30647962c71de7e490ee39792f39ecbf717836f322101 | 814ec7989805adffe153d27cea019c078a55c28f |
| 2500 | 20889807848 | b0cbe2a53c41eae999a701cf5d3e986c283e518765dac66c24b60a904aa48ee8 | bbeaac695145c5b01faf8c40e12860292eda4b9e |
| 5000 | 20890156805 | e2da0e2460f003a4a178e36a05cb80394ca55a4748413eff334e414a0ffc6393 | 2b078bba96e121eaa1b8fdaf4c94c2a5b41db71c |

这些是ledger/closure的值，不是本次重新测量。模型baseline使用原PI0226 adapted initializer，不是PI0241-final或旧623593。

## 2. 审核发现：只补三个窄缺口，不重新开大审计

### 2.1 训练完成与训练finalizer完成需要分别核对

当前`train_stage()`在`run_official()`返回后，依次核对consumption/contracts、checkpoint records、durable receipts，然后才读取update audit并写`pi0243_training_manifest.json`。HF sidecar曾失败，因而存在“训练已完、收据后补，但最后manifest尚未生成”的可能。当前archive closure不是这个manifest的替代品。没有证据证明manifest一定缺失，也不得把该可能性当训练无效。

先查已有训练manifest、日志和收据；齐全有效直接复用。若只缺事后finalizer，则新增薄CPU只读finalize入口，复用原核验逻辑，不调用`run_official`、`Trainer.train`或`train_stage`，不生成新optimizer/backward。核对5000步、原schedule的真实消费、16个action/mask契约、optimizer membership与原参数审计；用原完整tensor reconciliation处理已知抽样假阴性，不把常数字段或前4096元素当完整更新证明。保留实际观察/可重构/缺失的区别。文件缺失是证据缺口，不预设科学失败；确有错误数据/权重/配置则阻止promotion并报PI。

### 2.2 C_T标签不能代替实际screen条件一致性

原PI0243明确使用C_T实际训练processor的label-free条件。当前screen源码仍调用Stage21 `_build_processor(processor_base_model, TRAIN meta, modality)`；该builder复制所给base目录里的processor_kwargs。路径不同本身不证明输出不同，processor_base_model也可能已经正确导出C_T。现有公开card摘要不足以确认。

复用已闭合PI0240/0242/R1证据，只在原16窗口做**CPU、零模型**的当前binding核对：真实训练生效kwargs和screen kwargs；两路帧/时间/顺序、pixel值与grid、token/mask、state、embodiment、stats、action mask及物理标签。明确从实际训练构造路径得到C_T，不用两个eval builder冒称parity。模型观测不得带expert未来、targetXYZ、sourceID或phase。原训练标签必须对应相同window，不能为了小RMSE替换专家。

若当前输出与原批准C_T等价，封存证明并保留现有screen，不改builder。若实测偏离C_T，允许最小screen-side修复恢复原批准契约，新建screen execution binding并记录旧→新实现；不改训练权重、源数据、已seal训练代码或旧card来追认。已生成的偏离契约结果标`NONCONFORMING_SCREEN`并保留，不能用作A_PASS或SCIENCE_FAIL_M16_BUDGET；替代仅因为事先确定的契约不符，不因为分数差。任何无法由原C_T唯一裁定的科学差异返回PI，不自动设计新预处理。

### 2.3 HF归档验证强度需要如实闭合

当前`pi0243_hf_archive_tree.compare`在远端sha缺失时跳过sha比较；control readback只读取config和model index，list/download也未显式指定revision。这说明本次代码保证的范围不自动等于“所有文件在固定revision均已读回”。它不是归档损坏证据。

按上表的immutable revisions复用原manifest/逐文件inventory核对file-set、bytes、所有内容hash。大文件有内容SHA则使用；无SHA的小文件或其他对象在固定revision流式读取hash，缺失校验标UNVERIFIED，不默认PASS；Git blob SHA不冒充文件SHA256。沿原tree标签/排序规则计算，不比较不同prefix的tree而误报。

不需要把三份约58.37GiB payload全重下载，不重新上传已匹配文件，也不删除/改写公开HF历史。不打印token，不复制credential。

**分开两道门：**已有本地模型payload内容经原receipt/manifest核验，且screen其他条件齐全时，可以继续screen，同时后台工程流程由agent补齐归档证据。HF全量内容核验应在唯一持久源删除或B-stage启动前闭合；归档补证不触发训练/推理重跑。任何实际模型内容不符则立即停止该比较。

## 3. 接手627524：一个active owner，不盲目重排

先在PC已有SSH通道查实际job状态、owner、参数、staged代码SHA、card和输出；不根据旧Q消息判断还在排队。

- 正确版本已R：不原地改文件、不并发另跑同cells。只读准备与finalize可并行；确认其科学binding不符时保留产物，不promotion。
- 仍Q且screen admission成立：可继续原任务；不因为本次review重新提交重复job。
- 仍Q但已证明缺本地payload/依赖或C_T契约不符：保留qstat/script证据，使用合法hold或仅取消自己的这一个作业，修复和本地/Corelab回归后恢复唯一执行。不要让已知无效版本占GPU。
- 已终态：先读取已有records；合法完整结果复用，机械失败只补缺失/无效cells。模型真实REJECT不属于技术失败。

Corelab优先执行实际显存容纳的小screen；Hopper原合法队列可保留，不能在无Corelab容量/可达性证明时反复搬迁或重复提交。迁移在首次有效cell前冻结host/runtime；不得挑选不同设备上的最好结果。已有合法结果按原protocol first-valid规则保留。Hopper不用于普通pytest/import/debug；不创建monitor。

目前screen源码仍要求本地`training/checkpoint-*`及完整receipt。已归档不代表本地目录存在。运行前确认真正consumer路径；只保留/恢复需要的文件，若改为model-only读取必须有基于原full inventory的受测resolver和显式scope。不能删除当前reader依赖再指望HF自动恢复。

## 4. screen执行与验收：四组各16条，共64个唯一cells

固定顺序：initializer(step0)→A1000→A2500→A5000。所有窗口和四组模型都完成；早期REJECT正常记录并继续。不选checkpoint2500替代5000。

当前runner读取`records.jsonl`时以source_episode_id建dict，已有行会直接skip。新增外层CPUvalidator必须在dict折叠前检查duplicate/conflicting rows、完整key集合、stage/step、model/processor/runtime/cohort身份、raw/decoded/tensor hashes、有限值和gate结构。相同source ID不足以证明来自同一实验；不能复用旧PI0241/旧C_I记录。完整同身份结果直接复用，避免新增forward。

每cell保留原raw[1,40,132]、decoded[1,40,10]、实际state/RGB、expert、mask、gate全部issues和artifact hash。模型身份与C_T有效配置可通过独立screen admission绑定，不强迫改旧record schema。gate/provider/row-rot6d/dt/scorer不变。

逐窗定义：
- XYZ_RMSE = sqrt(mean_t(sum_xyz((prediction-expert)^2)))；
- SO3_RMSE = sqrt(mean_t(geodesic(Rpred,Rexpert)^2))；
- 全40点、NumPy linear percentile，原有效mask契约不变。

stage5000唯一主判定：16条完整且身份/训练审计有效，REJECT=0，median XYZ_RMSE<=0.020m，p95<=0.035m，median SO3_RMSE<=0.050rad，四项联合通过。WARN不另设新否决。安全通过不替代拟合，平均XYZ error不替代RMSE，参考目标距离不替代专家误差，更不等于achieved worldTCP。

报告一张4行结果表（step、有效N、ACCEPT/WARN/REJECT、线/角速度/首点issue数、median/p95 XYZ_RMSE、median SO3_RMSE），附16窗配对变化、A–D/phase侧报和最差窗口。loss0.0397仅作原日志字段，不与归一化local-flow MSE或物理RMSE混比。此阶段是TRAIN记忆能力验证，不声称泛化。

原协议另有32个direct-flow sidecar cells（step0/5000×R1 bridge4×原四个t×C_T），有同身份证据则复用，缺失则只按原预算补齐。R1单次特征与native/helper证明保持，不扫seed/step数；该sidecar用于终局失败解释，不参与A_PASS排名，不因已有结果差无限加诊断。

## 5. PASS / FAIL / 工程异常的下一步

### A通过

封存`PASS_STAGE_A`及其证据后，无需新增PI确认，自动执行原Stage B：A-final5000 weights warm start，新AdamW/scheduler，原64TRAIN、5000steps，其他原配方不变。不是从A optimizer精确续接，也不是选中途最好模型。

B启动前的CPU准备可并行但不可提前训练：绑定stage B独立family/prefix（当前make_card仍有A-family字面值，不能照抄造成跨stage覆盖）、真实5000项schedule、原C_T和stats、原64以及16子集、同stage checkpoint/screen集合、实际容量和健康归档sidecar。只修改薄stage绑定并新seal，不修改已训练A card来伪装身份。

B最终需64整体及原16子集都满足原标准。B通过→FULL10K→新family统一五profile原selector→winner的TRAIN准入→原A–D one-chunk→progress-preserving receding。各科学门和资源规则继续来自原PI0243，不由本文件扩权。

在将来实际启动C之前，机械检查C的training view是9792而screen view是原64；早期1000/2500 screen应在规定时间触发而非训练结束才补算，step1000 model-only snapshot必须真实可消费。当前通用CLI代码不自动证明这些未来阶段已经实施。提前CPU检查，不让这类后续任务准备阻塞当前A验收。

### A有效失败

终态`SCIENCE_FAIL_M16_BUDGET — STOP_FOR_PI_SINGLE_RECIPE_DECISION`。保存所有中期与最终负结果和已授权direct sidecar；不转B、不用最佳中期代替、不追加A10000、不改loss/LR/精度/阈值。给定量分支：总体位置/姿态仍偏离专家，或位置较好但时序/首点gate失败，或分层尾部失败；以已测数据为据，不预先把原因指定为欠拟合、容量或泛化。新recipe只提交建议，不执行。

### 工程/证据不完整

明确`SCREEN_ENGINEERING_BLOCKED`、`TRAIN_FINALIZATION_INCOMPLETE`、`SCREEN_CONTRACT_UNRESOLVED`或`ARCHIVE_CONTENT_VERIFICATION_PENDING`及实际影响范围；不是科学FAIL。工程自主定位→最小修复→针对性CPU/真实轻量回归→恢复，不设固定次数上限，但不增加唯一科学cells，不并发重复任务，不重训A来生成报告。

## 6. 最小交付和执行纪律

只需复用/补齐四类小证据：
1. 原训练manifest或CPU补全manifest（原日志、schedule/update、三份checkpoint、archive receipts）；
2. `stage_a_screen_admission.json`（C_T/模型/输入/代码/设备/剩余cells以及现有job身份）；
3. 原64 cells、CPU完整性validator、4-step表和A判定；
4. PASS则B run-card和实际启动记录，FAIL则封存失败及单一下一配方建议。

不得反复全库扫描、重跑旧10880专家census、5440 profiles、PI0242诊断或A训练。不创建通用orchestrator，不每修一处新增科学编号；保留旧证据、独立screen代码版本和可恢复ledger。模型输入边界、gate/selector/WBC、DEV-E/F/Final-F、DAgger、硬件与PR merge限制保持。A/B阶段selected_checkpoint=NONE。

Issue #11先canonical，PR #77只pointer。publication不等于执行，GPU/scheduler实时事实由PC agent报告。

## 7. PC agent启动正文

执行PI0243_STAGE_A_SCREEN_ACCEPTANCE_GO，完整读取本固定文档、原5755883470及5771889442/5771735878。Stage-A5000已完成，禁止因sidecar/PBS退出或manifest缺失重训A。

接手627524真实状态和staged身份，不重复qsub，不原地修改排队/运行脚本。并行补三个窄点：已有训练finalizer是否完整（缺则CPU只读补全）、当前16窗screen是否真正绑定C_T（零模型tensor核对）、HF固定revision逐文件内容校验（缺sha不当PASS，已验证本地权重可先screen）。复用既有证据，不重开大审计。

契约合格即完成原四组×16的64cells；已有合法记录按完整identity/hash核验复用，重复/混源/坏NPZ不折叠成PASS。REJECT保留并继续。最终只看step5000，原四项门不变；完成原32 direct侧报或引用同身份结果，不新增诊断矩阵。

A_PASS且审计完整则自动从A-final5000 weights启动原B64/5000，新optimizer/scheduler、独立family/prefix；B准备仅做CPU配置检查直到A通过。A科学FAIL按原门停止，不挑中期、不加步数、不改recipe。工程自主修复不限次数，不重训补报告。

Corelab优先小screen，Hopper仅合理容量/现有合法执行；不重复搬迁、不创建monitor。所有后续FULL/selection/one-chunk/receding严格沿原条件门，DEV/F/DAgger/硬件/merge继续锁定。Issue #11记账，PR #77pointer。
