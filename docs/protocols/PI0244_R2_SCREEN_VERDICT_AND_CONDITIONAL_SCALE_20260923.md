# PI0244 R2 — 现有权重筛查、科学判定与条件扩量

日期：2026-09-23（Asia/Singapore）  
Decision：`GO_COMPLETE_EXISTING_R2_SCREEN_THEN_BRANCH`  
Token：`PI0244_R2_SCREEN_VERDICT_CONTINUATION_GO`  
发布状态：`REVIEW_PUBLISHED_NOT_EXECUTED`  
审核基线：`83120c6fd09626f015252ec031b5b3ecd6d1d07b`  
Issue #11 为 canonical scientific ledger；PR #77 只放 pointer，不 merge。

## 1. 决定：收口现有科学实验，不重新开训或扩大审计

近期唯一主交付：现有 R2 initializer/2500/5000/7500/10000 在原16TRAIN上的80个唯一有效screen cells，以及final10000按原四项联合门的判定。通过后自动执行此前已批准的XYZ4 M64→FULL→selection→frozen-WBC路线；有效失败则探索树耗尽，停止新配方决策。

本文件是执行补充，不新增实验臂、seed、screen样本、训练步数、准入阈值或loss。原章程 `4117f401778067e183385508c668686d1798120f` / Issue5782066001，R2放行 `984565af45097b547354e681fc0c63fc7fa304af` / Issue5787874192继续适用。R0有效FAIL、R1/P32不适用、旧P1条件重放不足等结论保留。

`PASS_R2_TRAINING_CLOSED` 只表示训练执行/收据/收口审计满足既定检查，不表示模型通过质量门。R2训练已完成，不得为报告、归档、代码version或screen错误重训这条已完成轨迹。

## 2. 本次审核依据与证据范围

已读取：Issue5796145774（训练closure）、5796151220（hash erratum）、5796235163（screen提交）、PR5796236692；查询最新后续ledger未见新的screen结果。源码审核包括上述commit diff，`scripts/pi0244_r2_recover_posttrain.py`，`scripts/pi0244_xyz4_m16_10k.py`的loss/hook/配置绑定/训练收口/screen验证与判定，及原条件授权的内容。

GPT没有SSH、查询live qstat、读取远端NPZ/完整权重或原始训练日志、运行pytest/模型、修改PBS、提交/取消作业。以下执行状态和hash来自canonical，不称为GPT独立远端实测。本次只在独立文档分支发布计划，不改变已提交629335的staged代码和绑定。

### 2.1 当前接受并复用的身份

| 对象 | 固定身份 |
|---|---|
| 唯一screen owner（ledger时点） | `629335.hopper-m-02`；提交时Q/small，Insufficient ngpus |
| 实际执行child card SHA256 | `c772fc39ea37bb944b3394f7edf01b02434a7622fa3a57374699f1d271f2ff0d` |
| execution binding | `3a3f443ffd43b967c285091d0000f525e855a24a8c6e0b0c7c4df55e89502ee3` |
| screen wrapper SHA256（ledger） | `2c585990aaf6556bdca1caa6426a54a14d2e5f84bc551b825e0c3cd8bfe846a4` |
| training closure SHA256 | `6c627eee8fb28868e5a3c02ba668983a3e27bb5ae17562f6d09bd599bbd8d111` |
| training manifest SHA256 | `3ec77a175cdf15d690debae749665bfee6000da5cc9c1076ebf41ca19ee8ecf3` |
| posttrain reconciliation SHA256 | `05e468169bb027867361db3425804606d80b20a9ca403c1e4b6d58cf2c2a3105` |
| projector V2 report SHA256 | `e21bf4d88c69cd2f27ff8a7efa8c351ec29b4137056be28a4beaaad9253f8330` |
| V1 erratum SHA256 | `6660ad98d49603615954ce1ef33d0217818e5a665b6b0e38c6a9ff975a59e42d` |

R2四份完整checkpoint tree：
- 2500：`c770dba2f45842593a9d7d61fef5b1df98d25d9177f97572d93a1739a0d8e49f`
- 5000：`bd8ff46965e0b099cdb0943e1fa279755615fa859a833baa394c5c70f00d760c`
- 7500：`cf0fdb14652ee0720e65b78f17748751800f641543cff95190ccdc9c2abd4273`
- 10000：`7735a27eaf6a9f99d7d1ec7ecd17eba0f9a48e6a14cacb5a81e8139b99b5b752`

具体本地/归档路径、HF fixed revision和逐文件inventory从已验证child card与receipts解析，不凭basename替换。R0、PI0243-A、PI0241同名checkpoint均不得混入。

### 2.2 已闭合事项不反复重做

1. V1误绑preflight parent；V2及erratum转为实际child card和binding，保留旧文档与原始测量。接受append-only更正，不重训修报告。
2. update audit的INCOMPLETE/expected_end=1000由import-time default解释；实际global_step10000、10000条完整消费与schedule、16份action/mask contracts、runtime loss-forward计数、配置与receipt的联合核验支持训练收口。不能只把INCOMPLETE文本替换成PASS；现有reconciliation正是需要保留的证据链。
3. Projector V2比较15个张量的有限F32前缀，其中action_decoder.layer2.b、position_embedding.weight前缀变化，支持原any-nonzero-projector-parameter谓词。它不证明全组L2、全部15个都变，也不证明其余13个整个张量未变；没有新矛盾时，不为了把前缀审计扩成全量而阻塞screen。
4. 冻结组与diffusion的结论仅沿现有审计的覆盖范围引用；不把代表性证据夸大成全权重逐元素证明，不重新引入已经纠正的projector“零更新”假阴性。
5. “本次reconciliation模型/forward/backward/optimizer=0”是只读收口新增活动计数，不是整条R2计数。R2有10000 canonical training updates；screen执行前是0个完成cells。各阶段分开记账。

### 2.3 XYZ4确实在实际优化路径上，而不只是日志字段

源码 `patch_action_head` 调用一次original.forward，使用其返回的可微action_loss/action_mask计算加权值，然后替换output['loss']。权重XYZ0..2为4、其他1，分母为weighted mask sum加原epsilon。`_install_training_interventions` 将其装到官方训练入口；posttrain核验要求loss-runtime TRAINER_RETURNED且forward_calls=max_steps，并关联first-native-forward和named optimizer moments。

这支持按既定XYZ4实验进行screen；不在screen安装新的训练loss hook，也不改sampler/decoder补救成绩。运行期原始证据由agent根据closure引用核对，不用本次源码阅读冒充模型结果。加权loss与R0原loss不能直接比大小；更高XYZ权重不保证位置和速度都会改善，姿态也可能回退。

## 3. Goal S0：接管已有screen，保护真实身份

先经PC既有SSH/login-shell路径核对629335实际owner、状态、Resource_List、脚本与child card、输出/records。不能把ledger中的Q当实时事实；若qstat只在特定login-shell/module路径可用，复用已验证入口，不在错误shell找不到命令后猜状态。

- 正确R：让它继续，不改staged代码/当前execution checkout，不并发另一套同cells。
- Q且尚无有效cell：原则上保留已准入任务。Corelab确有同契约可用设备和可读模型时，可沿既有Corelab-first授权受控迁移：先hold/cancel唯一自己的冗余Q任务并确认无竞态，再执行。没有新的容量/可达性证据，不机械地搬来搬去。
- 终态或部分完成：读实际产物，冻结first-valid cells，只恢复缺失或被明确技术证据证明无效的部分，不按成绩重跑。

本次没有新脚本bug证据要求取消629335。普通CPU的校验与M64准备可在独立worktree并行。`_execution_binding`含git branch/commit和source hash：不要将本文件分支直接merge/pull到正被验证的execution checkout，哪怕只改文档也可能改变绑定。只读读取协议，新阶段另建identity。

GPU分配前不做新增模型warmup。第一条既定cell兼作真实推理集成；合法REJECT也保留，不换设备重新抽样。实际scheduler资源根据已测推理需求，不能未经证据声称必须H200，不能改量化/精度/分辨率/H40来迁移。

## 4. Goal S1：一次完成固定80 cells

顺序保持initializer(step0)→2500→5000→7500→10000；每组原16source IDs和原顺序。原PI0226 initializer不是R0-final。当前ledger未发现既有R2screen记录，默认运行该已提交的完整计划；不为了省baseline重新引入未绑定的历史records。

若已有完整同身份baseline可按旧协议复用，必须把模型/输入/processor/backend/noise/record的对应关系写明；不得仅凭source ID相同复制R0结果。80指唯一预注册cells，不强制掩盖复用或工程attempt的实际查询计数。

冻结：原seed22301、已批准的训练C_T-derived label-free eval条件、TRAIN stats、两路RGB/state/language、dense-relative H40、132维输出/active0:10、row-rot6d、frame/TCP、dt0.05s、原stock sampler/精度、原gate。目标XYZ、expert未来、episode/phase标识不进入模型条件。

只顺序加载一个模型，不把optimizer载入GPU。现有完整本地tree可用则直接使用；缺失才根据fixed-revision manifest恢复必要输入/模型scope，不重新下载全部四份optimizer、不重传已验证HF。

早期模型REJECT或误差大正常收集，继续原剩余cells；不提前挑终点、不把step7500好于10000当更换候选的依据。第一批16/16不是整个R2完成，也不是提前晋级门。

## 5. Goal S2：轻量CPU结果验收，不用报告字符串替代数据

### 5.1 源码已具备和仍需补足的范围

当前R2 `_verify_screen_records` 已检查重复/foreign IDs、step、NPZ路径和file hash、raw/decoded hash；`_complete_screen`明确要求每组16个原IDs，并有显式四项criteria，不是all(empty)。这些检查保留。

当前完成判定从JSON的xyz_rmse_m/so3_rmse_rad聚合，只对gate_decision计数REJECT；所审函数没有完整校验decision枚举、全部array shape/finite、JSON metrics与NPZ重算的一致性。因此新增/复用一个薄CPU结果validator，在promotion前给出完整性结论。不是已发现本轮数据坏了，不因此暂停健康screen。

### 5.2 Validator最小范围

- 在dict去重前核验5×16唯一(step, source ID)集合，不允许额外/冲突/错family记录；profile expected-binding与actual-binding绑定actual child card、recipe、权重tree和source cohort。
- 核对raw[1,40,132]、decoded[1,40,10]、expert[40,10]、原StateV1及action_mask；按原mask契约检查值与对应专家来源，保持shape/dtype证据。不要把不同dtype的相同mask字节hash强行判标签变化；只复用已经批准、显式记录的等值cast。
- 验证原始数组、指标及关键gate数值finite，gate_decision只能为ACCEPT/WARN/REJECT，record与gate.as_dict的decision/issues相符。保留全部WARN/REJECT原因和observed/threshold。
- 从NPZ重算逐窗XYZ RMSE、SO3 RMSE、gripper、相邻动作线/角速度及首尾误差，使用现有受测helpers；与JSON/summary逐项对照。缺字段/不一致为typed数据或实现问题，不靠赋零补齐。
- 数值差異先区分同定义计算舍入和真正公式差异；不能因少量舍入改变阈值。原模型实际产生nonfinite是模型/科学无效输出，不作为“坏文件”偷偷重新采样；只有传输截断、错binding等可证明机械原因允许修复相应产物。
- 如需重算原gate，只用原输入/state/provider/timestamp/config在CPU确定性核对，不改变阈值和输出轨迹。保存原结论与validator结果，不覆盖原record；实际评分程序错误需append-only更正，不能按成绩选择采用哪个评分版本。

无须为此再申请GPU、重建G3/PI0242/native parity或读取新数据。尽量在一次遍历80个现有NPZ内完成，small JSON和CSV足够。

### 5.3 唯一final-10000联合门

前提：80个合法cells、5份绑定、训练closure/执行身份有效。每窗全部40点：

`XYZ_RMSE_i = sqrt(mean_t(sum_xyz((pred_i,t - expert_i,t)^2)))`

`SO3_RMSE_i = sqrt(mean_t(geodesic(Rpred_i,t, Rexpert_i,t)^2))`

跨原16窗口使用原NumPy linear percentile。**只有final10000**同时满足下列四项才PASS：

1. 原SafetyGate REJECT=0；WARN保留，不另加ACCEPT比例门。
2. median逐窗XYZ RMSE <=0.020m。
3. p95逐窗XYZ RMSE <=0.035m。
4. median逐窗SO3 RMSE <=0.050rad。

不得使用训练loss、reference-target min distance、平均点位误差或teacher-forced端点替代。零REJECT并非充分条件；SO3中位数通过也不代表所有姿态/角速度项通过。M16 PASS仅是训练支持上的拟合和准入，不是已泛化、物理任务成功或可直接上机。

## 6. Goal S3：用R0对照回答XYZ4是否有帮助

不增加新forward，复用封存R0与本轮R2相同16窗口数据。交付：

- R2五行screen表：step、有效N、ACCEPT/WARN/REJECT、主要issue频数、XYZ median/p95、SO3 median、max-speed分布。
- R0-final与R2-final的逐source配对表、目标/phase分组；位置整体bias与去bias残差、速度峰值segment；姿态是否因XYZ相对权重提高而回退。
- 说明checkpoint/processor/sampler/backend/noise身份是否可比较。历史只存seed或hash时不把它夸为完整同噪声受控因果实验；跨设备差异明示，不为追严格bit相同重跑旧模型。
- R0已公布参考：REJECT5/WARN11，XYZ median0.047100m、p950.054982m、SO3 median0.037452rad。新R2数值未生成前不预测优劣，不把大部分安全或相对改善当PASS。
- 若记录有native和XYZ/rotation/gripper loss分项则并列，不将加权总loss与R0native数值直接比较。缺失分项标NOT_OBSERVED，不为补日志重训。

原新condition packet和48/32 direct-flow辅线继续沿R2放行规定：有同身份证据可复用，缺失如实标NOT_RUN_NONCRITICAL_DIAGNOSTIC。历史P1 exact replay不可恢复的结案不重开。辅线不阻塞screen、主判定或PASS后的M64，不用新观测偷改正式screen，也不新增本轮科学矩阵。

## 7. Goal S4：PASS自动扩量；FAIL结束当前探索树

### 7.1 PASS_R2_XYZ4

先封存screen closure与fit_recipe_manifest（arm/final10000权重、XYZ4公式、R0实际precision、C_T/statistics、stock runtime、四项证据）。`selected_checkpoint`仍为NONE；这是训练配方晋级，不是WBC policy选择。

随后无需新的PI确认，运行原已批准的 `PI0244X_XYZ4_M64_5K_V1`：

- 从**本R2-final10000**权重warm start，新optimizer/scheduler，原64TRAIN、5000步；不是接着R2优化器跑到15000。
- 保持XYZ4、R0实际parameter/moment/autocast精度、其余原recipe；peak LR1e-4、warmup10、cosine5000，seed和既定64曝光顺序不变。原16是固定子集。
- 完整保存1000/2500/5000；screen0/1000/2500/5000×64。final5000整体64和旧16子集都满足原四项门。
- 新family/output/HF prefix，不覆盖R2。复用成熟loss、loader、publisher和reader；当前R2入口只支持fixed16/10K，不能直接`--stage b`就声称实现扩量。只加薄stage preset/启动绑定，不另造训练框架。
- 显式传入expected_end_step，测试5000与10000等不同stage，不继续依赖import-time默认1000然后反复事后修报告。这是未来stage配置修复，不重写R2旧audit。
- 若读取旧R0辅助metadata作身份校验，仍只读既有TRAIN/归档记录，不把其family当新模型。

M64 PASS→继承XYZ4的FULL9792/10K→完整familybaseline+四候选各1088CALIB→原selector唯一winner→winner自身TRAIN准入→原A–D C0 frozen-WBC one-chunk→四任务通过后progress-preserving receding。原scorer/cadence/tick/termination/旧16保持及FULL1000/2500失稳门继续适用。FULL的train view与screen64应真实分离，早停须发生在继续训练之前，而不是事后补写。未到前一级PASS，不提前训练下一级。

未来sim是单一冻结策略的实际achieved world-TCP任务评价，原0.035m任务成功半径不是专家RMSE阈值的替代。DEV-E/F、Final-F、DAgger、硬件及merge仍锁定；不声称多技能/独立泛化。

### 7.2 R2有效科学FAIL

R0已FAIL、P32不适用、XYZ4若有效FAIL，则：

`PREAPPROVED_RECIPE_SET_EXHAUSTED — STOP_FOR_PI_NEW_HYPOTHESIS`

停止B/FULL/sim。只用已有screen、日志、可用sidecar指出：XYZ仍未达标、速度仍违规、姿态退化、或尾部窗口问题，允许多因子并存。给一个具体但NOT_AUTHORIZED的新假设；没有P32+XYZ4、权重2/8/16、额外20K、seed sweep、平滑/retiming、修改sampler/gate的自动授权。未运行侧报不能被当作负机制证据。

### 7.3 工程或来源冲突

报告工程、外部资源或科学来源冲突的准确类型，不将队列等待/坏输出文件/未完成profile判成模型FAIL。合法机械故障自主定位→最小修复→低成本回归→恢复缺失工作，不设任意修复次数上限。已正确生成的负结果不重试；模型真实非法输出和未授权训练变化需科学处理，不能靠“恢复”清除。

## 8. 排队期间应做和不应做的工作

允许并行CPU准备M64 run-card草稿、stage-specific loss绑定、sample IDs/schedule、5000-step classifier、容量、保存/恢复测试，状态READY_NOT_STARTED。只有R2主PASS、完整性与身份成立才启动backward。

不要因当前消息而重建监控或新建通用orchestrator；消息里的“已安排任务”不视为本次额外定时授权。既有PC agent单实例监控若获授权则只跟踪真正active owner及产物增量；没有变化不反复拉全库，终局停止。此GitHub文档发布不是ChatGPT后台运行声明。

资源：本地PC做git/CPU tests/SSH；Corelab做Linux集成、输出复核、小screen与后续native sim；H100/H200只用于有容量或规模依据的正式任务。不在H200上反复测试路径/import，不重复已知OOM。当前healthy queued/running screen不无理由取消。

存储：旧R2四份完整checkpoint和HF receipts保持，screen只需加载一个模型、不需要优化器上GPU；新M64三份full checkpoint重新核对实际输出主机容量、临时峰值和余量。复用可见scratch mirror、已成功publisher以及独立HF prefix；旧Project_Storage可读性不冒称所有compute节点可读。上传或报告失败不重训，不删除唯一durable源和当前reader依赖。

## 9. 最小交付与启动正文

交付四项即可：
1. 本次review authority＋原closure/reconciliation/V2/child binding的引用，不重跑大审计。
2. 实际629335或合法后继状态、五份80-cell输出及薄CPU完整性/重算报告。
3. final10000联合结论＋R0/R2配对表，辅线明确COMPLETE/NOT_RUN和解释限制。
4. PASS后的fit_recipe与M64实际启动证据，或FAIL后的树耗尽closure及一项未授权新假设。

Issue #11先canonical，PR #77只pointer。不得仅回复“PASS_TRAINING_CLOSED/仍排队”作为完成，也不得承诺无法保证的GPU排队时间。

### PC agent启动正文

执行PI0244_R2_SCREEN_VERDICT_CONTINUATION_GO。完整读本固定文档、R2放行984565af及章程4117f401，核对5796145774/5796151220/5796235163。复用83120c6的V2更正和已完成10000步，不重训R2、不改旧audit、不重开projector全量/旧P1诊断。

接手629335实际状态和child card c772fc39…、binding3a3f443f…；保留一个active owner，不改staged文件、不重复qsub。完成既定0/2500/5000/7500/10000×原16的80cells，first-valid固定，早期REJECT继续。

独立CPU复核5组身份/ID/NPZ hash/shape/finite/gate枚举和metrics重算，不把JSON的坏字段当科学PASS/FAIL；真实模型非法值不能重采样掩盖。不新增模型查询或全面审计。只按final10000：REJECT0、median XYZ RMSE≤.020m、p95≤.035m、median SO3 RMSE≤.050rad，连同完整证据判定。给R2五行表和R0-final配对变化，不把加权loss直接对比native loss。

旧48/32direct packet按既定规则非阻塞处理，无法完成明确写NOT_RUN；不阻塞R2主判定和PASS后的晋级。

排队期间CPU准备XYZ4 M64/5K，但不提前backward。R2主PASS后自动从其final10000权重启动原64/5K，新optimizer/scheduler、独立family/prefix、保持XYZ4和原precision，检查64整体和原16保持；按原条件继续FULL/selection/frozen-WBC one-chunk/receding。当前R2preset只支持16/10K，后续以薄preset实现，expected_end_step显式传值，不借stage C冒充扩量。

R2有效FAIL研究树耗尽停PI_NEW_HYPOTHESIS，不加步数/权重搜索/seed/新loss，不跑B或sim。工程自主修复无次数上限；本地/Corelab测试、小screen，容量需要才上H100/H200。选模前selected_checkpoint=NONE，DEV/F/DAgger/硬件/PR merge仍锁定。Issue #11canonical，PR #77pointer，不新建monitor。
