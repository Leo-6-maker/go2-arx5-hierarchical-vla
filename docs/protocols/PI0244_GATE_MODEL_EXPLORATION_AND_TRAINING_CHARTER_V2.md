# PI0244 — Gate-pass model 长程探索、训练与 sim 推进章程 V2

日期：2026-09-23（Asia/Singapore）  
Decision：`GO_CURRENT_R0_THEN_PREREGISTERED_EXPLORATION_AND_SCALE`  
Token：`PI0244_GATE_MODEL_EXPLORATION_GO`  
文档状态：`PUBLISHED_NOT_EXECUTED`  
源码导航基线：`5169dffaab888c9f5ec1953f541d0dc306f1a00c`  
Canonical scientific ledger：Issue #11；PR #77只作pointer，不merge。

## 0. 总目标、承诺边界和本次新增授权

用户要求agent长程探索并训练出能够通过gate的模型。目标不是把检查改成PASS，而是得到**在固定原SafetyGate上零REJECT、同时准确拟合专家动作的共享策略**，再扩到64窗口、全量训练、唯一选模和frozen-WBC仿真闭环。

本章程是一张有限实验树，不是保证成功的承诺，也不是无界自动科研。保留正在执行的PI0244 M16-10K（R0），不改变其权重/recipe/预算；R0通过直接扩量。R0有效失败后，新增最多两个具名、单因素、各10K的候选训练配方（R1/P32、R2/XYZ4），只在下述条件满足时串行执行。它们是本次PI明确批准的科学干预，不能冒称工程修复。

**覆盖范围：**仅把旧PI0244“R0最终FAIL立即等待新的配方授权”改为“封存R0 FAIL后进入本章程中满足准入的下一具名arm”。一个arm失败不会被覆盖、延长或重命名为通过。未列明的科学变更、已列探索耗尽、上层support/selection/sim科学失败，仍回PI。工程修复不设任意次数上限。

当前R0、未来R1/R2、后续B/C各有独立card、output、checkpoint family和HF prefix。绝不在运行中的R0上套新loss或精度补丁。发布文档本身不启动/取消任何job。

## 1. 当前事实及只需阅读的依据

### 1.1 最新账本，不把旧Q当实时状态

2026-09-22 18:13:19 UTC（新加坡2026-09-23 02:13:19），Issue评论5781596538记录：`628370.hopper-m-02`在hopper-17取得一张H200、GPU preflight通过、trainer进入初始化；当时没有完整checkpoint、receipt或screen结果。该信息不是GPT实时qstat。接手时查询实际状态和产物，不新提交同一R0。

此前5780966374和5781433200记录了Project_Storage在计算节点不可见，以及resolved_diff没有走metadata mirror的机械问题；当前实际mirror为 `/scratch/e1192548/pi0244_m16_10k_v1_r1_compute`。R0 card SHA：`12e09718d5499aece46800b348f733d1bb506400d0468cb128f233f571f75191`；resolved_diff SHA：`cd7fe87750e0f9aabc8ad90088355246ead7c9f4d2f75dc7e3f4adace31a2ee2`。从card/ledger解析路径，不手打猜目录。

PI0243已封存负结果：原16窗、5K终点14REJECT/2WARN、XYZ median0.095847251653m、p950.116609711754m、SO3 median0.047693562034rad。更新/动作契约/固定revision归档审计已闭合。R1 native/helper 32/32 exact和F(F(h))修复保持。不能重新拿已推翻的projector无更新或旧重复特征处理当当前根因。

必读且只沿相关调用链：
- PI0244原科学授权：Issue5776019972；协议commit `d23727c2b7764589573f6c7a7195d091980a9bbc`，`docs/protocols/PI0244_FIXED16_10K_BUDGET_TEST_AND_CONDITIONAL_SIM_V1.md`。
- PI0243原闭环门：Issue5755883470，commit `5d6d843235625c416d372cdbed01b185a1e6da5f`；负结果5775639138；已完成screen/参数/HF证据直接引用。
- 最新R0执行5781596538及后续评论；当前 `scripts/pi0244_fixed16_10k.py`、成熟PI0243/Stage28、persistence和原stock screen。
- pinned NVIDIA GR00T `51d4c89f72fda44cbf77285c6a8114b52676b8a1` 的 `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`，仅为R1/R2精确确认参数/原loss调用。

本次GPT核对了GitHub实时可读取的ledger、PI0244 wrapper和pinned native loss，没有SSH、读取远端原始大文件、运行模型、测试、上传HF或查询scheduler。以下R1/R2、分支条件、预算是**新增预注册研究设计，不是已有实验结论**。

### 1.2 科学解释边界

已有数据支持模型学到了一些动作，但位置与运动学质量尚不合格；不证明BF16是根因，也不证明XYZ必须重加权。原native loss为 `sum(mask * (pred_velocity-target_velocity)^2)/(sum(mask)+1e-6)`，不是对132维简单求均值；不能把122个inactive维称为已证实的loss稀释。

训练mode与label-free eval mode存在已记录pixel处理差异。保持原设置；相同16个source windows不等于每步看到相同的增强pixel tensor。本轮不悄悄关augmentation来制造理想memorization。

## 2. 完成层级：gate-pass不等于任务成功或泛化

| 里程碑 | 必须证明 | 不能宣称 |
|---|---|---|
| M16_FIT_GATE_PASS | 原16TRAIN全非拒绝，XYZ/SO3联合标准合格 | 泛化、多技能或可直接上WBC |
| M64_FIT_GATE_PASS | 原64整体合格，原16子集合格 | 独立未见分布验证 |
| FULL_CALIB_ADMISSIBLE | 同family完整评价，原selector选唯一且winner自身TRAIN合格 | CALIB是未使用过的盲测 |
| TRAIN_SIM_CLOSED_LOOP_COMPLETE | 同一模型在原one-chunk和receding的真实achieved TCP任务成功 | 硬件实时性或跨任务普适性 |

TRAIN-CALIB是模型选择用的校准集，而且此前已经反复报告过结果；不能将它包装成最终独立泛化测试。DEV-E/F、Final-F、硬件仍锁定，最后提交PI_GENERALIZATION_REVIEW。当前A/B `selected_checkpoint=NONE`；可记录 `training_recipe_candidate`，但不是WBC候选。

**M16及M64共同最终质量门不变：**在完整有效记录与更新/输入审计前提下，原gate零REJECT（WARN保留）；median逐窗XYZ RMSE≤0.020m；p95≤0.035m；median逐窗SO3 RMSE≤0.050rad。每窗使用全部40点，XYZ为sqrt(mean_t(sum_xyz(error²)))，SO3为sqrt(mean_t(geodesic²))，percentile linear。M64还要求原16子集同样通过。平均误差/训练loss/目标参考距离不替代这些量。

禁止靠静止输出、复制expert或target注入、挑两个WARN、平滑/重定时、改首点、放宽阈值或换seed满足目标。原SafetyGate包含哪些具体检查，完整沿原配置，不只复制线速度阈值。

## 3. Goal G0：保护当前R0，跑完科学结果

### 3.1 一次接手，不重新开始

核对628370或其有证据的合法后继：actual host、job ID、phase、card/source SHA、已完成updates/checkpoints/cells。正确R状态不改staged代码；已结束先读真实结果；partial checkpoint不作为resume。以前预启动失败不计canonical optimizer steps，但实际资源消耗保留。

R0继续原PI0244：同PI0226 initializer、16TRAIN、10K+cosine10K/warmup10、seed22301、native loss/原精度；full checkpoints2500/5000/7500/10000；screen0/2500/5000/7500/10000×16=80cells，只判10000。先完成旧计划已授权48个direct侧报，身份相同可复用；侧报不参与PASS。

### 3.2 把剩余wrapper边界一次修到位，不改训练

源码中PI0244的train/screen外层仍可能在委托成熟runner后按sealed而非runtime-mapped output_root找manifest/closure。只让finalizer、screen收口和publisher一致使用已有runtime_card/path mapping，另立execution binding；不能因为报告找错路径重训。

final classifier须显式核对所需criteria key集合、finite数值、80完整唯一cells及final16，不用空字典的 `all()` 推出PASS。loader、manifest、NEW_EMBODIMENT幂等注册和PBS变量名的修复命中实际调用链；不要新增第五套orchestrator。

实际最终qsub由同一resolved launch生成并检查PI0243_*变量（不能再混成PI0234_*）、可见mirror、publisher解释器、stage/steps/输出/归档参数。站点只能在计算节点验证的部分，在正式job模型加载前列出所有可检查项和结构化错误；不通过一次次H200作业逐个猜路径。轻量本地/Corelab测试先完成。

如果当前job已进入正确训练，非破坏性的CPU代码/报告准备可在独立worktree进行，不中断有效训练来追求新框架。

## 4. Goal G1：把已有结果变成分支输入，而非再做泛泛诊断

R0结束后，从原NPZ/records/native audit和48direct读出：每cell gate issue/observed/threshold、XYZ bias和去bias残差、速度尖峰segment、姿态及gripper、各t/group的局部误差、实际trainable parameter/grad/Adam moment dtype及full update。缺量注明NOT_OBSERVED，不补一个无限诊断矩阵。

读取原full checkpoint中的optimizer state和已有model parameter inventory，区分：checkpoint磁盘dtype、运行时trainable参数dtype、exp_avg/exp_avg_sq dtype、计算autocast dtype、screen加载dtype。不能从config的bf16=True直接推断所有状态都是BF16。若运行时没有日志，则依据pinned创建路径与状态给出证据等级，不能伪称实测。

分支决策只使用固定TRAIN数据，不读新CALIB或DEV/F来选recipe。原16的目标/时序标签身份已闭合，除非新证据直接矛盾不再重验全数据。数据/ABI矛盾属于停止条件，不临时修标签并继续。

## 5. Goal G2：预注册有限研究树，普通arm FAIL不再每次等prompt

### 5.1 决策顺序

1. **R0最终PASS：**锁定R0为训练配方，跳过R1/R2，进入G3。
2. **R0有效FAIL：**封存原FAIL；若实际trainable权重或Adam一/二阶moment是低精度，进入R1。两者已经是FP32则记录R1=NOT_APPLICABLE，不能重复同配方；身份无法确定则R1=NOT_ADMITTED。
3. **R1最终PASS：**锁定R1，跳过R2，进入G3。
4. **R1有效FAIL/不可适用：**仅当R0的XYZ median或p95超原门槛且active XYZ槽位/labels可信时，进入R2；R0只有rotation/gate失败而XYZ已合格，则R2不具备这份授权的依据。
5. **R2最终PASS：**锁定R2进入G3。R2FAIL，或R1/R2都不适用且R0FAIL：提交 `PREAPPROVED_RECIPE_SET_EXHAUSTED — STOP_FOR_PI_NEW_HYPOTHESIS`。

这是先后顺序的first-PASS方案，不是跑完所有arm再挑最好。合法分支失败保留；不把错误归为代码问题后重跑找好成绩；不把R1/R2组合为第三个隐藏arm。任一明确NaN、非法优化、数据/ABI矛盾、权限/安全问题停止受影响计算并报告，不能擅自切下一arm掩盖异常。

### 5.2 共用控制条件

R1/R2均从**同一PI0226 adapted baseline1000**重新初始化，不从R0/R1失败终点续训；新optimizer/scheduler，固定原16、同一10000项曝光表、seed22301，10K cosine/warmup10、原peakLR1e-4、WD1e-5、maxgrad1、batch1/accum1及真实beta/epsilon。其他数据、模型容量、冻结模块、augmentation、采样器和输入/输出契约沿R0。

full saves2500/5000/7500/10000；screens0/2500/5000/7500/10000×16=80cells，唯一定点10000作主判定。相同身份baseline可复用；实际backend/条件不符则按既定matrix执行，不增加额外smoke。第一正式更新/第一原定screen兼作真实集成，计入budget。中期正常差成绩不改变终点或提前挑模型。

每个arm上线前发布有限字段完整的run card/resolved diff、唯一science patch SHA和对应unit tests到Issue #11；这是执行记录，不是再申请新GO。代码不应依赖旧PI0244 ALLOWED_CONFIG_DIFFS默许新变化：R1/R2使用独立显式allowlist，旧R0验证器不放宽。

### 5.3 R1：P32 — trainable权重与优化状态精度干预

Family：`PI0244X_P32_M16_10K_V1`。

唯一科学因素：在optimizer创建前，把已有requires_grad的projector/action-head浮点参数提升并保留为FP32，让对应Adam exp_avg/exp_avg_sq为FP32；冻结模块权重/precision不变。保留BF16 autocast下的原训练算子路径、原loss、原noise/time采样、train/eval预处理；不同时全模型FP32推理、改attention或修改loss reduction。

这是标准混合精度“参数/优化状态存储”与“前向算子autocast”分离的候选方案，不证明当前故障由BF16导致。actual grad dtype依pinned实现记录，不私建第二份不同语义optimizer。不要误用只管backbone的配置开关冒充action-head FP32。

必要验证：转换前后float数值相同，requires_grad名单相同，冻结内容不变；第一次optimizer update后实际moments/参数dtype正确；save/reload/断点恢复维持FP32 master状态。screen仍沿原全模型BF16加载/原sampler，原四项标准不改；如果改善只在未经授权的FP32 screen出现，不算PASS。

若必须改变原输入噪声dtype、mask、模型图、优化器种类或训练算法才能实施，超出本arm，记录精确冲突回PI。FP32状态增大训练显存和checkpoint尺寸必须先计容量；不得为适配显存偷改effective batch、8bit optimizer、gradient-checkpointing或其他recipe。

可从已有状态做只读量化诊断来解释结果，但不要求预先证明“多少百分比更新被舍入”才有资格试验；该比例不是已知事实。收益由固定screen和局部对照判断。

### 5.4 R2：XYZ4 — 只改变native flow loss的XYZ权重

Family：`PI0244X_XYZ4_M16_10K_V1`。

本arm相对于R0仅改变loss对有效动作维度的权重；使用R0原参数/优化状态precision，**不叠加R1**。原native方法已经返回保持autograd的逐元素 `action_loss = mse(pred_v, target_v)*mask` 和 `action_mask`；优先复用同一次原生forward的输出，不第二次采样noise/调用另一个模型。

令w在XYZ slots0..2为4，在其他slots为1（inactive仍由mask为0）；新loss：

`L_XYZ4 = sum(w * action_loss) / (sum(w * action_mask) + 1e-6)`。

因此w全为1时回到原native loss，而不是把总loss无解释放大四倍。w=4是本PI事前固定的候选，不是未来结果调参；不搜索2/4/8/16，不把state/gripper/rot6d映射猜成XYZ。保存native原loss及XYZ/rotation/gripper组分，训练日志中的不同加权loss不能和R0的0.0397直接比较。

不改flow target `action-noise`，不平滑predicted velocity（该量包含noise，不等于物理轨迹速度），不加oracle target位置，不改decoder/clipping/time。位置拟合权重增加不保证速度gate改善；原零REJECT门会继续检验这一点。

CPU/小synthetic autograd验证：w=1与native值/梯度一致；inactive无贡献；目标权重梯度比例符合解析式；有穷梯度；归一化分母正确；不detach后重新包tensor。项目真实模型的第一训练步纳入本arm，不另加科学调试updates。

### 5.5 科学量预算和比较规则

R0是已授权10K，不重开。最多新增R1+R2共20K optimizer updates；每arm最多80stock cells。加原R0最多240stock cells，不含已有历史实验；同身份复用单列。每个实际执行的R1/R2仅补32direct侧报（initializer/final×原bridge4×四t×固定eval-mode C_T），使用R1 native parity已修helper。和R0原48direct合计最多112本轮direct records；内部backbone/head calls另计。

不强制把所有预算花完，首次有效PASS即关闭其余候选。工程失败的无效计算/IO恢复单独记账，不增加唯一科学cells和训练终点；exact resume保持optimizer/scheduler/RNG/消费cursor。不以重新seed、重新初始化或换数据恢复一个有效失败。

## 6. Goal G3：只扩大第一个通过联合门的recipe

配方锁定后输出 `fit_recipe_manifest.json`：arm、terminal step、完整weights、processor/统计、训练precision、loss公式及权重、screen runtime/精度、已完成metric与gate证据。它是训练配方，不是selected WBC policy。

M64阶段名 `PI0244X_<CHAMPION>_M64_5K_V1`；若R0直接PASS，原PI0244 M64 family可继续使用，不能同时创建两条B。起点为champion-final10000权重，fresh optimizer/scheduler，原64TRAIN5000steps、原schedule与16保持门，其他字段继承champion。R1胜出则保持trainable FP32；R2胜出则保持XYZ4，不在扩量时悄悄改回native或叠加另一arm。

M64 full saves1000/2500/5000，screens0/1000/2500/5000×64=256；final5000整体64及原16子集都满足原四项才通过。M64 FAIL不能绕过送WBC，也不能自动回头改探索顺序或多训一轮；提交support-expansion决策。

FULL阶段 `PI0244X_<CHAMPION>_FULL_10K_V1`，M64-final warm start＋fresh optimizer/scheduler，sealed9792TRAIN10K；仍继承champion精度/loss，原data/stats/表示不变。保存完整2500/5000/7500/10000；固定64screens0/1000/2500/5000/7500/10000=384。1000 model-only snapshot不是resume点。1000及2500两次都64/64 LINEAR_SPEED_REJECT，必须在继续2500之后训练前落实原早停，不能到10K后追认。

既有wrapper目前只实现fixed16的prepare/train/screen；B/C需要薄preset和实际screen调度适配。只在其前一级通过后完成必要实现/测试，不能拿stage C默认full view或PASS分支充当新M16。训练view9792和screen view64分开。每个阶段正确PASS后自动推进，不再请求例行GO。

整个研究树最高条件训练预算：R0 10K＋两个可选arm20K＋仅一条M64 5K＋FULL10K，共45K canonical updates；只有失败探索和通过扩量按树消耗。这不是无限科研许可，也不是预计必须花45K。

## 7. Goal G4：统一校准选择，再进入真实sim

FULL family完整后，以同一冻结processor/runtime/backend完成原baseline＋四候选、各1088CALIB（共5440），原selector准入及排名代码不改，baseline不候选。只对champion这条FULL family做一次统一选择，不把不同arm/checkpoint/设备最好的结果拼成一个family。

全候选无资格：`NO_ADMISSIBLE_CHECKPOINT`并停科学门。唯一winner还必须在其对应FULL step的64TRAIN及16保持门合格；winner不合格不能剔除后换runner-up。首次真正设置selected_checkpoint在这一步，此前都是NONE。

沿旧科学约束保留专家census中已知160个REJECT（含12CALIB），不删除难例或重写标签。若匹配标签与全量gate冲突成为实际主因，提出新数据/任务协议回PI；这不在agent工程修复内。

冻结winner完整weights、实际部署processor、raw/decode、gate/provider、frame/TCP、dt、输入边界、model-noise/sampler后，按既有canonical参数进入Corelab原A–D C0 one-chunk。四个原任务首次有效执行都必须按原achieved world-TCP成功半径0.035m、termination/cleanliness规则通过；gate拒绝不得执行。专家RMSE达标不等于world target达标。

one-chunk全通过→progress-preserving receding：同一model、persistent sim、新RGB/state、多次真实query与有因果ID的reference更新；保持原cadence/prefix/tick/query预算，不每H40归零、不重置机器人、不重播旧观测。原sim参数从canonical绑定，不编造新参数；缺科学定义时停该门，不阻止已批准离线训练。

模型与native WBC环境隔离；实际显存共驻不足按已有合法机制分离或回PI，不修改WBC/time语义。同步暂停sim等待推理的方案如实报告，不能称实时机器人控制。

成功终点 `TRAIN_SIM_CLOSED_LOOP_COMPLETE — STOP_FOR_PI_GENERALIZATION_REVIEW`。DEV-E/F、Final-F、DAgger、硬件和PR merge继续锁定；多任务/未见环境能力需独立协议，不能将多目标TRAIN reaching改名general model。

## 8. 资源、存储、节奏和自主恢复

本地PC做git/测试/SSH控制。Corelab做Linux集成、CPU数据/诊断、装得下的无梯度screen及native sim。Hopper login只做站点允许轻量核验、传输和调度；H100/H200用于已成熟、显存或规模需要的正式计算，不作为Python路径/debug/pytest平台。复用实际训练显存证据，R1状态变大会重算需求；不重复已知OOM证明。

“工程不限次数”不等于每修一行就qsub。同一resolved_launch生成预检与实际调用；CPU功能覆盖完整mapper/launcher/聚合路径；GPU第一原定cell/update检查必须计入原预算。一stage只有一个active owner；现有job正在正确训练时不创建竞品同阶段、不给同目录写新代码。

每arm完整checkpoint逻辑保留；HF按独立prefix和immutable revision发布并核验。工作盘容量必须计入当前完整Trainer大小、optimizer精度变化、序列化/传输临时峰值、screen、缓存和余量，不能用旧BF16的20.89GB当FP32状态上界。当前活跃恢复点和reader引用不删除；旧initializer/data/runtime/历史证据不扩大清理。

屏幕推理顺序加载一个模型，不向GPU加载optimizer。Hopper quota不阻塞完全在Corelab读写的任务，但目标主机容量须真实证明。仅更新自身新产物的可重建cache，不删唯一持久副本，不在日志/卡片/聊天里暴露token。HF失败只恢复上传，不重训。

恢复状态仅一份轻量execution_state，记录current_arm/stage、science card、actual host/device、active job、complete checkpoint receipts、remaining cells、下个动作、科学与工程计数。聚合PASS依据显式key、完整性和实际metrics，不依赖exit0、非空文件、loss小或all(empty)。

每个checkpoint与阶段milestone记账；无状态变化不大量重复拉取全库。已闭合参数/数据/HF审计按hash引用，只有新内容或真实冲突才重算必要部分。未来arm可以CPU准备，不能在R0运行时提前backward。普通REJECT合法保存并继续该阶段预定样本。

不新建/复制monitor；已有执行agent监控按其授权运行，不把本章程发布当GPT后台服务。确实无法取得合法设备、权限、空间时记录EXTERNAL_INFRA_BLOCKER，不伪装科学FAIL、不绕过站点规则。

## 9. 停止条件与最终交付

- 单个已列arm有效FAIL：封存后按G2分支继续，不每次等新prompt。
- 配方树耗尽或不适用：停PI_NEW_HYPOTHESIS，不再20K/50K延长。
- M64/FULL/selection/one-chunk/receding有效FAIL：停对应科学门，没有自动的数据/控制/任务修订。
- 模型/数据/ABI/目标注入/冻结模块错误、未授权参数、非有限优化或无法解释的identity冲突：立即隔离，交PI。
- 工程IO/import/launcher/report错误：最小修复、低成本回归、从已封存边界恢复；不重新跑合法负结果。

集中交付：一份experiment_tree.json（采用本固定规则而非临时决策）、每arm run_card/resolved_diff、training/dtype/update/消费证据、checkpoint HF receipts、stock/direct records、一个逐arm结果表、fit_recipe_manifest、后续阶段closures及sim trace。保持少量薄代码，不创建新的通用训练框架。

Issue #11先写canonical，PR #77只pointer。所有新增科学配方明确标新授权；不能声称来源文档已经证明FP32或XYZ4会奏效。本次文档发布并未运行这些试验。

## 10. 给PC agent的启动正文

执行PI0244_GATE_MODEL_EXPLORATION_GO。完整读取本文immutable commit、原PI0244授权5776019972和最新执行5781596538及后续状态。

先接手628370或有证据的合法后继，不重复训练、不改正在运行的R0。修复仅命中实际launcher/runtime-mapped manifest/screen收口，CPU测试显式criteria和完整cells；不再让H200逐次暴露普通路径问题。

先完成原R0 16TRAIN10K、四checkpoint、80stock和原48direct、原联合门。R0 PASS直接锁recipe扩64/全量；FAIL保留原结论，按本章程有限研究树继续：实际权重/Adam低精度时可运行独立R1 P32；R1 PASS就不运行R2；R1FAIL/不适用且R0位置门失败时，运行独立R2 XYZ4。两者都从同一PI0226 initializer重训10K、同schedule/seed，不warm-start失败模型，不叠加变更。公式、dtype语义、预算和准入必须按正文，不自由挑值。

每arm final10000只按原零REJECT＋medianXYZ≤.020/p95≤.035/medianSO3≤.050判定，80cells完整；中期不挑最优，不重试好成绩。工程自主修复无限次不增加科学arms、seed或steps。合法arm FAIL封存后进已授权下一arm，不要求例行PI确认；树耗尽或语义/优化异常才返回相应门。

第一个联合PASS的recipe自动进入原64/5K、旧16保持→全9792/10K→同family原selector→winnerTRAIN准入→Corelab frozen-WBC one-chunk/receding。保留champion的loss和训练精度；不能只warm-start权重却悄悄改回别的recipe。每阶段只有一条主线，不同时跑旧新B/C。选模前selected_checkpoint=NONE，不能memorizer直上WBC。

本地/Corelab处理测试、CPU和小screen；H100/H200只做有资源依据的正式训练。保护原gate/WBC/scorer/ABI/frame/dt/target边界；不改标签、采样器、平滑输出，不读DEV/F、不DAgger、不上硬件、不merge。HF使用独立family固定revision，保留恢复状态，报告归档错误不重训。

目标交付是可验证的联合PASS模型与按门推进的训练/sim结果，或者穷尽已授权分支后的机制证据，不是又一次compile或排队状态。Issue #11先canonical，PR #77pointer。

## 外部方法说明

PyTorch官方AMP示例区分原精度创建的模型/optimizer与autocast计算范围：https://docs.pytorch.org/docs/main/notes/amp_examples.html 。这是R1设计的通用方法依据，不证明本项目低精度就是根因，不要求升级pinned runtime。

本次已核对pinned native loss源码：https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/model/gr00t_n1d7/gr00t_n1d7.py 。R2仅用同次原生forward的action_loss/action_mask形成显式新加权目标，不另造teacher图。