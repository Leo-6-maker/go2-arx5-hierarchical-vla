# PI0241 — 固定16窗口学习可行性：一次1000步训练与自由生成验收

发布日期：2026-09-21。审核的是2026-09-20封存的PI0240结果。  
决定：`GO_FIXED16_MEMORIZATION_1000_WITH_UPDATE_AUDIT`。  
发布状态：`AUTHORIZED_NOT_STARTED`。本次发布本身未执行训练或任何远端作业。  
审核/实现导航基线：`f536aa359e265fe8e225faa29090fee5922e5baa`。  
Issue #11继续canonical；PR #77只作pointer，不merge。

## 1. 审核依据与结论边界

必须读取以下固定版本，而不是重新扫描整库历史：

- PI0240 canonical：[5750516982](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5750516982)，PR pointer `5750517767`。
- 审核基线下 `reports/PI0240_CORELAB_G3_G4_G5_CLOSURE.md`。账本给出的报告SHA256：`396726c65a1a2c846ad87c8ed82d311e8fb823ceb174dca01700219513c9a466`。
- `scripts/pi0240_g4_fixed_cohort.py`：真实query、原gate和逐窗metric_bundle；`scripts/pi0240_g5_attribution.py`：原汇总与未授权fixed16建议。
- `scripts/pi0228_h200_large_adapt_v1.py` 的build_config/run_official；`scripts/stage28_h2_fresh_base_2000_fixed_warmup10.py` 的_build_training_config；pinned GR00T实际训练入口和resolved config，只沿这些调用点读取。
- PI0240资源协议 `ae57d2824d2e97770382393d265dce57c472a334`；工程自主/first-valid规则继续沿用PI0239/0240。

GPT已核对GitHub报告和上述代码，未登录Corelab/Hopper、未重算原始NPZ或独立运行测试。执行agent复用既有封存证据核验内容身份，不把这份计划当作新的已完成实验。

### 1.1 接受的实测结果

G3的142个固定窗口实际经过dataset/collator `[142,40,132]`、slots/mask/state anchor和独立decode；报告XYZ最大误差1.5305370383095253e-7m，SO(3)最大1.425034124872992e-5rad。G4完成256个first-valid cells，baseline/fresh各64TRAIN+64CALIB：

| 模型/split | mean XYZ error (m) | mean SO(3) error (rad) | mean of per-window max speed (m/s) | LINEAR_SPEED_REJECT |
|---|---:|---:|---:|---:|
| baseline1000/TRAIN | 0.496990 | 0.510579 | 4.4814028 | 64/64 |
| baseline1000/CALIB | 0.516374 | 0.489551 | 4.4951143 | 64/64 |
| fresh10000/TRAIN | 0.151497 | 0.075521 | 3.0721757 | 64/64 |
| fresh10000/CALIB | 0.150062 | 0.077615 | 2.9845185 | 64/64 |

因此接受 `SCIENCE_FAIL_TRAIN_FIT` 作为**这组固定TRAIN上的动作拟合/运动学质量尚不足**的操作性结论。模型已学到部分内容，不是完全没更新的证明；但“学到一些”不能转成可执行性。

不能据此证明唯一根因是模型容量、步数不足、学习率、BF16、某个loss、或泛化绝对不存在。TRAIN/CALIB相近的均值不证明全分布等价；分位数、分层、未见任务仍可能不同。当前不支持把泛化差距当首要修复方向；WBC尚未执行，不能把当前速度拒绝归因给WBC。

### 1.2 汇总与代码注意事项

原G5把 `status=SCIENCE_FAIL_TRAIN_FIT` 写在特定报告构造中，并非通用条件分类器。本PI接受负结果基于实际64/64拒绝和专家误差，而非那个字符串。新结果必须根据下述布尔条件计算PASS/FAIL；合成全通过/单项失败fixture须覆盖分类器。

原closure的mean XYZ error不是RMSE；其p95来自跨窗口统计口径，不能直接当逐tick p95。新实验明确计算逐窗RMSE再聚合，不用旧mean字段代替。原minimum-target-distance在126/128窗变差继续保留，不因专家误差下降而删除；它不是本记忆实验的优化/选模标准，也不是achieved TCP。

G3动作codec通过不自动证明训练参数确实更新、训练输入与部署输入完全相同或采样器能生成所拟合的动作。本轮把这些关键证据嵌入训练，而不重新开启全套G0–G3审计。

## 2. 本次科学授权与不授权事项

**现在授权一个独立的FIXED16_FREE_RUNNING_MEMORIZATION_FEASIBILITY实验**：固定16个TRAIN窗口，最多1000个canonical optimizer steps，保存250/500/1000完整Trainer checkpoints；step0/250/500/1000在同16窗口上做自由生成，共64个唯一screen cells。

明确解除过去“training/backward/optimizer=0”禁令仅限本实验。授权真实forward/backward/AdamW/scheduler/update/save/reload和必要更新诊断。不授权另一套LR、loss、精度、容量、数据或seed搜索，不授权第二条表现驱动的训练run，不授权10k续训/重启、DAgger、CALIB新推理、WBC/sim、DEV-E/F、Final-F或硬件。selected_checkpoint仍为NONE。

这是学习可行性实验，不是原10k的exact resume，也不是纯粹“只改数据量”的因果消融：样本支持/暴露顺序和1000步cosine总时长均在本卡明确改变。它回答“原模型/损失体系能否在固定少量观测上生成足够准确、连续的动作”，不直接解释原10k全部失败机制，更不证明通用策略。

## 3. 固定科学run card

| 字段 | 本次冻结值/解析规则 |
|---|---|
| 实验身份 | `PI0241_FIXED16_V1`；独立output及checkpoint lineage |
| 初始化 | PI0226 adapted checkpoint-1000，与PI0240 baseline权重同内容；只加载权重，**不加载其旧optimizer/scheduler/RNG**。不是原始通用BASE，也不是fresh10000/7500 |
| 数据 | 原PI0237固定64TRAIN中A–D×phase0–3每层1窗，共16个；只取原window的observation index0及对应完整H40 |
| 表示/语言 | official dense-relative H40；StateV1；dt0.05s；原frame/TCP/rot6d；`move the end effector to the red target`；target XYZ不得入模型 |
| TRAIN stats | 与PI0240/623593训练绑定的完整TRAIN stats及relative_stats；不在16窗或CALIB重新估计 |
| 可训练参数 | tune_projector=True，tune_diffusion_model=True |
| 冻结参数 | tune_visual=False，tune_llm=False，tune_vlln=False；以实际参数名/optimizer membership验证，不仅检查配置布尔值 |
| 优化器 | 原`adamw_torch`；LR=1e-4，weight_decay=1e-5，max_grad_norm=1.0；betas/epsilon/parameter-group规则继承pinned PI0228官方resolved TrainingArguments，准备阶段将实数封存，不由agent选择 |
| LR schedule | 官方cosine，max_steps=1000，warmup_steps=10，warmup_ratio=0；按官方scheduler步序，记录前12步实际LR。此schedule不是旧10000步scheduler的继续 |
| batch与设备数 | global_batch_size=1；gradient_accumulation=1；1GPU；dataloader_num_workers=0 |
| 精度/性能参数 | 沿PI0228：bf16=True，fp16=False，tf32=True，load_bf16=True，backbone_trainable_params_fp32=False，gradient_checkpointing=False；不因显存/效果擅自改FP32 master、8-bit optimizer、量化或attention实现 |
| 预处理 | 沿pinned Stage28配置：state dropout0、color jitter/random rotation关闭、crop_fraction1、shortest_image_edge96及其实际processor实现；不新增augmentation。与PI0240部署processor对照，发现不一致先区分预期配置与真实偏离，不能手改到看起来一致 |
| 随机性 | data/train seed22301；封存Python/NumPy/torch/官方Trainer实际seed设置与版本；每步不得重新播种 |
| 步数/保存 | 最多1000 canonical optimizer steps；full-state250/500/1000，save_only_model=False，至少保留三份；不通过改max_steps反复调用250步训练来拼接cosine |
| 自由生成screen | step0/250/500/1000×16窗；使用PI0240同一_run_one、processor、sampler/dtype/语言、每cell推理seed22301；无新seed/样本/扰动 |
| 主判定时点 | **step1000**，不能选表现最好的250/500替代；早期screen只报告趋势，不据此调参 |

上述继承字段有唯一代码/封存配置来源，不是交给agent自行补参数。实际构造config后保存完整resolved config、optimizer默认与每组值、dtype、trainable参数清单和逐字段diff。除了本卡明确的16窗view、顺序、步数/保存及seed设置，不引入其他科学变化。缺失来源先沿已知训练manifest/源码机械解析；确实存在未裁定科学冲突时返回PI，不能以默认最新版库填空。

## 4. 准备阶段：一次完成，不重做PI0240

### 4.1 样本选择和固定顺序

旧G5建议只描述“PI0238 salt-hash”，未给可执行字节串。本PI明确为新预注册规则，避免不同实现随意选样：

1. 读取已封存PI0237 cohort的64TRAIN元数据，验证与PI0240 TRAIN records的ID/文件身份对应。
2. 分组键为原 `target_family` 的a/b/c/d与整数phase0/1/2/3。每层按 `SHA256(UTF8("PI0241_FIXED16_V1|" + source_episode_id))` 升序取1条；hash碰撞以source_episode_id字典序破平。不读取误差/gate来排序，不按是否容易记忆替换。
3. 输出16窗original episode/source-rollout IDs、原Parquet/视频定位和hash、state/action/mask/语言指纹及层级。可有同rollout不同phase，不能声称16独立rollouts。
4. 在查看这16窗的细粒度fit前封存cohort。用已有PI0240专家gate核验：任何选中标签出现原始REJECT/非法mask/来源不一致，不删除换样，停止报告目标与准入矛盾。此前64TRAIN专家无REJECT是预期，不替代实际绑定。
5. 固定1000条训练暴露顺序：先按(a→d,phase0→3)建立16-row序列，再用单独的 `numpy.random.Generator(PCG64(22301))` 每轮生成0..15的permutation，连续拼接截取前1000项。预先写schedule及SHA，不消耗正式模型训练RNG。共62个完整轮次加8项；每窗暴露62或63次。官方DataLoader必须顺序消费该schedule，不再另加随机sampler。

实现用一个薄fixed-view Dataset/Sampler适配层调用现有官方episode loader/get_datapoint(...,0)/processor/collator；不自写动作encoder、不重写Trainer/loss。记录完整TRAIN与16窗view的区分；不将filtered view说成原full-tree相同。不重新编号原source IDs，不改原视频、Parquet、stats或sealed manifest。每step日志绑定原window ID，自动断言实际1000项与计划一致，防止名义fixed16实际仍混入全部TRAIN。

### 4.2 复用证据，只增加低成本、影响决策的检查

复用PI0240 G3、模型/数据replica manifests、FFmpeg6和已通过loader/processor；不再重跑142/10880/5440/256历史实验。核验改变涉及的文件和依赖，其他引用旧证据。

在16窗上无模型地核对训练实际processor与PI0240推理构造的图像/state/语言/mask语义。训练默认预处理来自上述真实config，不是调用两次eval builder便称train/eval一致。数据augmentation默认已关闭，不新增随机crop。输入与label严格分开，专家未来只用于训练target/离线评分。

复用这16窗已有baseline/fresh NPZ做只读误差分解（不新增query）：每窗位置bias、去bias后的残差、39个segment的预测/专家速度、最大尖峰所在tick、首/尾误差，分别记录位置偏差与高频不连续。不得把这份诊断转成平滑惩罚或重新选样；它只为解释新训练结果。

准备阶段产物集中为一个 `pi0241_run_card.json`，链接cohort/schedule/config/输入hash/测试/实际host/存储证据。通过这些既定条件后**直接开始已经授权的训练**，不再停在“卡片准备好，等待批准”。

## 5. 真实训练与参数更新审计

直接复用PI0228 build_config→Stage28 _build_training_config→pinned官方training入口。旧Stage28/PI0228的CLI含历史BASE/TINY1/10000步保护，**不要原样执行旧--train**，也不要修改它的封存常量使其假装本实验；新薄入口应用本卡已批准override并验证完整resolved config。不得把历史StructuredAction trainer或projector-only路径替代dense-relative projector+diffusion配方。

只运行一条科学训练轨迹；正式前不另起一条任意optimizer smoke。第一个正式batch/forward/backward和首次正LR optimizer更新兼作训练端集成与峰值显存检查，成功即计入本次1000步。Corelab显存可容纳时继续；容量不够按第7节恢复/迁移。

在不改loss的前提下记录：
- trainable/frozen参数名称、数量、optimizer group归属、实际dtype，尤其projector和action_head各自的更新证据；
- 实际action/mask字段与每sample有效位置/维度数，不用“shape为132”代替loss mask正确；
- 每step sample ID、global step、实际LR和loss；保留原loss reduction，不事后把loss除数改成另一个定义；
- 前12步、step32、250/500/1000的组级有限gradient norm、固定代表参数的before/after差异、optimizer state/dtype。不要只凭loss下降判断参数已更新；
- 若正LR下代表参数长期不变，先核查实际optimizer membership并做对应组的完整/分块更新确认；不能从少量抽样零delta就断言整组未更新。需要改变trainable set/精度/optimizer才能恢复的，作为科学决策返回，而非偷偷改配置；
- 冻结backbone参数不得更新；训练mode/dropout与eval mode分别记录，不将“frozen”误解为可以任意替换backbone计算。

warmup首步LR可能为0，不能据此报“没有学习”。不新增teacher-forced model probes；训练本身的forward/gradient证据足够。单次异常、OOM、IO和loader错误先按工程根因恢复；有效数值中非有限loss/grad、已确认错误优化对象或无有效更新需封存，不能以换seed/LR追成功。

## 6. Screen、判定公式与完成条件

### 6.1 不让评估污染训练RNG

优先完成训练和三份checkpoint封存后，在独立评估进程加载已封存checkpoint依次screen；这样现有_run_one的逐cell seed重置不触碰训练RNG/样本顺序。可以逐保存点在独立进程评估，但同卡顺序使用且训练状态/RNG不被改动。不要把seed-resetting get_action直接塞进训练callback。

训练总schedule始终1000步；不能为了在250/500评估而重建optimizer或warmup。若step0的同16个baseline first-valid NPZ、input/processor/seed/backend完全匹配，可引用PI0240已有结果，不再forward；不匹配时明确新step0诊断身份，所有本轮screen使用同一冻结评估backend，不挑更好版本。

screen独立于训练batch，调用原模型的free-running get_action→原decode→原SafetyGate。不能把训练target张量送给decoder冒充模型预测。每cell保存raw/decoded/expert/state/mask、原输入指纹、全部gate issues、metrics。成功/失败都保留。

### 6.2 四项联合判定，仅看step1000

对第i个窗口，t=0..39：

`XYZ_RMSE_i = sqrt(mean_t(sum_xyz((pred_i,t - expert_i,t)^2)))`，单位m；

`SO3_RMSE_i = sqrt(mean_t(theta_i,t^2))`，theta用原row-rot6d/原SO(3) geodesic，单位rad。

跨16窗percentile统一NumPy linear方法；每窗40点在准备阶段已确认全部有效。若真实mask有padding，不悄悄填补或删点改分母，应先暴露契约冲突。

step1000必须同时满足：
1. 固定16窗原SafetyGate **REJECT=0**；WARN原样记录，不临时改为必须ACCEPT。
2. 16个 `XYZ_RMSE_i` 的median <=0.020m。
3. 16个 `XYZ_RMSE_i` 的p95 <=0.035m。
4. 16个 `SO3_RMSE_i` 的median <=0.050rad。

这些是采纳并明确化G5建议的**新实验事前标准**，不是PI0240已达成事实，不改变WBC的0.035m世界目标成功半径；专家轨迹RMSE与world-target-distance是不同量。

gripper MAE/max、位置与姿态首点/尾点、线/角速度及其专家对照、worst-window、每目标/phase结果全部侧报。gripper没有在本轮新增任意硬阈值，不能把四项通过宣称完整抓取技能。目标min-distance仅描述，不参与loss或checkpoint选择。

执行完整性、参数更新证据、16/16 rows、checkpoint/输入身份必须通过；这些缺失时是INVALID/INCOMPLETE，不是模型科学PASS或科学FAIL。模型原始输出若真有NaN/非法旋转，在正确运行契约下保留为数值/模型失败，不无限重复相同cell。

step0/250/500不满足最终质量标准是预期可观测的学习曲线，不提前以此中止；也不能在250/500偶然通过后选它代替1000。只有明确的数据/语义/数值异常或有证据的无有效训练更新可提前停止。1000结束时未满足任一联合条件，停止，不加步数、不改阈值。

### 6.3 状态必须由检查计算

- `PASS_FIXED16_LEARNING_FEASIBILITY — STOP_FOR_PI_SCALE_DECISION`：满足完整性、更新审计与四项最终标准。说明能记忆这16个固定观测，不代表大规模/闭环/泛化已通过。
- `SCIENCE_FAIL_FIXED16_LEARNING — STOP_FOR_PI_RECIPE_DECISION`：有效1000步实验未满足标准。
- `SCIENCE_FAIL_OPTIMIZATION_OR_CONTRACT`：有证据的loss/优化对象/精度更新/输入语义冲突，需要科学修改；附确切实测项，不猜唯一原因。
- `ENGINEERING_RECOVERING`：路径、library、IO、资源、序列化、测试实现错误；自主修复继续，不触发工程次数用尽。
- `EXTERNAL_INFRA_BLOCKER`：在现有授权和非破坏性路径内确实无法获得必须资源/权限。不能冒称科学失败，也不能无限无信息重提。

新分类器测试至少包含：四项全过、一项分别失败、missing/nonfinite/重复或错ID、WARN-only可通过、早期通过但1000失败不能代选。不能复用原G5硬编码的SCIENCE_FAIL字符串决定新结论。

## 7. 算力、存储和恢复

本地PC做代码/CPU fixtures/SSH；Corelab做真实数据准备、已有运行环境集成及**能容纳的真实训练和screen**；Hopper H200仅在本配方真实backward+Adam state显存不足或已证实规模理由下使用。G4 inference已能在Corelab跑完，不证明相同GPU一定能容纳训练；实际峰值要包含gradient、optimizer和activation。

查询实际GPU UUID/单卡总空闲显存、RAM和输出fileset容量，不猜32GB，不终止他人进程。优先按本卡batch1和原dtype完整运行；不为显存擅自降低模型容量、分辨率/H40、优化器状态精度或修改gradient checkpointing。静态容量已足够证明不可能时无需故意OOM，Corelab能完成的准备/输入/screen先通过后再上H200。H200不是pytest/路径调试平台，登录节点不做重CPU/训练。

存储按实际模型/full Trainer payload bytes估计三份保留checkpoint、当前原子写临时payload、必须的恢复状态、cache和证据的最大同时占用，留至少max(10GiB,20%新增峰值)余量；这项只影响资源不影响科学配方。只读取训练所需initializer，不为本实验复制整套历史四模型。旧sealed数据/权重不删除。Corelab工作不在Hopper写文件时，不受Hopper quota未知的全局阻塞。真正Hopper写入时，沿PI0240实际fileset阶段容量证明及持续检查。

不设工程修复/提交次数总上限，但同一训练轨迹/科学阶段最多一个活动执行。每次失败保存根因、最小修复、针对性回归；不原样重提坏配置。已经完成的模型screen first-valid结果不可因成绩差替换。

训练恢复只从完整checkpoint开始：模型、optimizer、scheduler、Python/NumPy/torch/CUDA RNG、采样schedule位置/global_step、实际config/dtype全部一致，且max_steps仍1000。无状态证明不声称exact resume。可按同一冻结配方重放丢失的计算前缀，重复计算单独记账；不拼接不同attempt最优权重/记录，不为失败结果新建seed或科学arm。

本卡科学预算是一条最多1000 canonical updates和64个唯一screen cells（可含已存在step0的16个）；不是工程失败的物理尝试次数。已完成有效1000步后的失败不能用“工程预算无限”重训至过。若换主机后的恢复需新科学数值语义，先交PI，不把资源迁移当权限豁免。

## 8. 产物、执行入口与精简工作方式

建议只新增一个薄入口 `scripts/pi0241_fixed16_learning.py` 和对应CPU测试；按需要增加一个小config/PBS包装。函数复用优先：PI0228 config/官方run、PI0240 metric_bundle/Stage21 _run_one/原gate。不新造训练框架、数据体系、第二套evaluator或多层protocol runner。

该入口是agent要实现的交付，不是已存在/已测试的脚本。建议固定CLI三步：

```text
python scripts/pi0241_fixed16_learning.py --prepare --run-card <new_run_card.json>
python scripts/pi0241_fixed16_learning.py --train --run-card <sealed_run_card.json> --authorization PI0241_FIXED16_1000_GO
python scripts/pi0241_fixed16_learning.py --screen-and-close --run-card <sealed_run_card.json>
```

一份run card供三阶段使用；准备后的token有本文授权依据，不是等待另一个空白My request。不得调用旧PI0228 --train误启动10000步。无副作用CPU测试通过、真实数据链闭合后连续执行上述三步。

核心产物仅为：
- run card/cohort/schedule/resolved config及输入/实现hash；
- training log、参数更新审计、完整250/500/1000 seals；
- 64-cell screen ledger/NPZ（step0复用时引用原hash）、四条件判定表；
- 一份closure，说明哪个环节首先失败/通过、替代解释和下一条最小科学决策。

针对用户报告的861365 tokens/约87分钟，不能从该数字单独推断账单或哪段代码低效；但本轮应避免重复全库读取、重复旧证据、无变化的轮询和长日志反复输入上下文。入口工作表列“已复用/新改动/待执行”，仅失败时按依赖定点读取源码。原G0–G4已封存，本轮不重新审计整段项目历史；一次工程修复只在同execution分支提交diff，不再要求新PI编号。终端全日志保留文件，给GPT的是阶段摘要和证据路径/hash。

完成后先Issue #11 canonical closure，再PR #77 pointer。推送小代码、tests、config、报告与inventory；权重/大tensor保留授权存储。不merge，不重建monitor。

## 9. 后续到大规模sim闭环的科学路线（本卡不自动开训下一阶段）

fixed16 FAIL：利用本轮保存的真实更新/gradient/dtype/loss/free-running误差区分：优化对象没有更新；teacher-forced loss下降但采样器生成不对应；更新与loss均正常但此配方记忆预算不足。允许解释并提出一条最小对照，但不得直接加LR、改FP32、加速度loss或延长训练。

fixed16 PASS：学习链具有有限可行性，下一步先比较原623593实际sample exposure、窗口抽样/尾部mask与本固定view，提出中等规模TRAIN-only coverage/pilot方案，再决定large-adapt-v2。不能从16窗成功立即推断继续10k就会好，也不能把MEM16_step1000当通用checkpoint。

新中等/大规模recipe需要独立批准的数据/预算/早期TRAIN free-running检查/abort规则；届时仍按本地测试→Corelab可容纳的pilot→必要时H200规模训练→完整candidate family/冻结selection→Corelab frozen-WBC A–D one-chunk→progress-preserving receding推进。旧12/160条expert与gate问题不能通过本记忆实验抹去，扩大数据/正式全CALIB选择前需按来源解决。

本轮停止于1000步最终学习可行性决策，是科学范围边界，不是工程次数限制。WBC、DEV-E、Final-F、硬件和下一轮大规模训练在本卡均不执行，selected_checkpoint=NONE。

## 10. PC启动正文

现在执行PI0241：本文已授权一次固定16TRAIN窗口、最多1000 canonical optimizer steps的真实学习可行性实验，不再等待训练授权确认。完整读固定版本本文、PI0240 canonical 5750516982及f536aa3基线的closure、G4/G5和PI0228/Stage28实际训练config；禁止重新扫描整库或重跑旧G0–G4。

复用PI0240 Corelab环境/FFmpeg6、数据与模型身份。按本文明确hash规则从原64TRAIN每A–D×phase层取1窗，固定16和1000项暴露schedule；保留原TRAIN stats、H40/StateV1/frame/dt与target边界。initializer是原PI0226 adapted baseline1000的权重，非fresh10000，optimizer/scheduler从新run开始。

复用官方训练路径，只新增薄fixed-view/launch与更新审计。原projector+diffusion可训练，visual/LLM/VLLN冻结；AdamW LR1e-4、WD1e-5、maxgrad1、batch1/accum1、cosine1000/warmup10，精度与其他字段按本卡绑定。先封存resolved config及diff，数据/身份/接口条件通过就直接训练，不停在准备就绪。

Corelab优先；训练backward/Adam state确实超单卡容量才经SSH/PBS上H200，H200不做通用代码反复调试。第一正式更新兼作集成测试，不另起任意smoke训练。验证实际sample顺序、mask、LR、gradient和参数delta，不能只看loss。工程问题自主定位/修复/回归/恢复，无固定工程次数；科学配置不随失败更换。

保存250/500/1000完整Trainer state；独立进程screen step0/250/500/1000×16，64 unique cells，step0同身份已有tensor可复用。避免get_action重置污染训练RNG。step1000以原gate零REJECT、median XYZ RMSE≤.020m、p95 XYZ RMSE≤.035m、median SO3 RMSE≤.050rad联合判定；不是mean error，不选最好的中期checkpoint替代，不加步数追PASS。

有证据的优化/语义/数值科学问题停PI；有效1000步后无论PASS或FAIL都提交本文规定的科学决策包。不进行新CALIB查询、WBC/sim/DAgger/下一轮训练、DEV-E/F/Final-F或硬件。MEM16不作为正式候选，selected_checkpoint=NONE。

小代码/tests/config/closure推execution分支，Issue #11先canonical，PR #77只pointer不merge。回报训练与screen实际计数、参数更新证据、四项判定、资源与hash；不要仅回复已读或卡片待批准。
