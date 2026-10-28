# PI0244 — 一次固定16窗口的10K预算检验及条件sim续行

日期：2026-09-22（Asia/Singapore）  
Decision：`GO_SINGLE_FRESH_M16_10K_THEN_CONDITIONAL_SCALE`  
Token：`PI0244_FIXED16_10K_GO`  
发布状态：`AUTHORIZED_NOT_STARTED`  
审核/执行导航基线：`69b587fe4edc3a7a0c9ff99240ba0214bc6e6e86`  
Canonical scientific ledger：Issue #11；PR #77仅pointer，不merge。

## 1. 决定及证据边界

接受PI0243 Stage-A的有效科学FAIL，不修改原结果，不启动原失败分支的B。本次是新的、事前固定终点的科学实验：从原PI0226 adapted initializer重新开始，在原16个TRAIN源窗口上运行一次10000 optimizer updates的native recipe，cosine总时长同步改为10000。只有下文最终门通过，才允许以新终点权重替代课程入口，进入原64→FULL→selection→frozen-WBC路线。

目的：检验在现有曲线仍明显改善的情况下，更多有效曝光与较长cosine时程是否足以满足原精度和动力学准入；不是宣布“根因就是步数不够”，不是把旧失败改成PASS，也不是无限1K→5K→10K→20K翻倍。

本轮训练不同时修改native loss、XYZ权重、平滑项、precision、容量、图像augmentation、sampler步数、数据/statistics或gate。若10K最终仍FAIL，本授权不会再自动加长预算；下一PI根据本轮定量诊断选择不同机制干预。

### 已读取的依据

- PI0243 closure：Issue #11 [5775639138](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5775639138)，PR pointer5775644004；查询该时点之后的ledger未见新的结果。
- 原分级协议：commit `5d6d843235625c416d372cdbed01b185a1e6da5f`，`docs/protocols/PI0243_STAGED_LEARNING_TO_FROZEN_SIM_GOALS_V1.md`，canonical5755883470。
- 审核基线的 `scripts/pi0243_staged_learning_to_sim.py`（stage/view、exposure_schedule、resolved_config、family绑定），`scripts/pi0228_h200_large_adapt_v1.py`；已有PI0241/Stage28及screen、persistence调用链沿原协议复用。
- 原screen验收/资源/存储补充和已闭合的G3、full-parameter、HF、PI0242-R1 native/helper证据按内容身份复用，不再重做。

GPT本次读取GitHub账本和上述源码，未SSH、未查询live qstat、未读取远端records/NPZ/原始loss日志、未下载HF权重、未运行模型/训练/测试。下文表格来自canonical，百分比是对表格的计算；未公布的逐窗口gate构成和当前A的local-flow表现不能虚构。

### 冻结的历史负结果

| PI0243 A step | N | ACCEPT/WARN/REJECT | median XYZ RMSE m | p95 XYZ RMSE m | median SO3 RMSE rad |
|---|---:|---|---:|---:|---:|
| 0 | 16 | 0/0/16 | 0.506671580441 | 0.564693884064 | 0.448743674131 |
| 1000 | 16 | 0/0/16 | 0.311820713052 | 0.369875026787 | 0.178970995322 |
| 2500 | 16 | 0/0/16 | 0.212737239805 | 0.256367304391 | 0.142770537726 |
| 5000 | 16 | 0/2/14 | 0.095847251653 | 0.116609711754 | 0.047693562034 |

2500→5000的XYZ中位数和p95分别下降约54.95%、54.51%，SO3中位数下降约66.59%。这支持“已测后半段仍在改善”的描述，不能证明每窗单调、没有更细的平台期或保证10K达标。最终XYZ中位数仍为原2cm限值的4.79倍，p95仍为3.33倍；14/16拒绝不能解释成只差一个异常值。SO3仅中位数通过，不代表所有旋转或角速度都通过。当前摘要未列14条拒绝的全部issue频次，不能把旧5431次线速度错误机械套到新14条。

Current A sidecar未被本closure交付：PI0242-R1的EXACT只证明其对应helper/旧权重；不得把其final1000局部误差写成当前A5000的实测。新计划会用固定的小对照补足，而不重做原PI0242完整矩阵。

### 必须复用的历史定位

- 科学screen：627934，canonical报告Exit0/hopper-21；全部64cells有效。
- parent训练card SHA：`569b5d4e0e221422621a1b9b05b9fa18970d399f7febed7d278685a0b89fe8e7`。
- 有效screen card SHA：`9306e37af7ab370206b9cc519b0b770402853b37da66fdcecb1fb3e564c931a6`。
- screen binding manifest：`359f061b4ada892cb02395a99561aceefefc9c7bba034db0935ace859dca96ae`。
- 原16cohort：`88f35ad5bc176d13ae998b88fa97c1daf0f7c6abc37b74b3989af131d2a36ed9`；source cohort `dfd60b94bbdd82cd07af2bbb93185e8cec720f6f4a88a33bb2cfe3a2eabce9e6`。
- 原5K曝光表：`4cf272d488c81f8bcb3ffb4298d90473656a235cba82190dd7cd5cdce05d2936`。
- 原训练manifest现值：`09d6572f670ef23641f4a76d11eef3482bcfb58109d2f6352655454900e39297`。
- conditioning范围报告：`4b2d6ba67b1ca6bd71a72a33292b42ff2de11a5903c77e4a11cad8e267fc92e1`。
- 原stage closure：`4f0fc92c5ff8c37e5cf1be0ba72f54e94e1d0a653ef6be123fd1ae2584897378`。
- 原A5000完整tree：`e2da0e2460f003a4a178e36a05cb80394ca55a4748413eff334e414a0ffc6393`，HF `leoyuer/pi0243-stage-a-admitted-v1`，原prefix `pi0243-stage-a-hf-20260922-retry1`，固定manifest revision `2b078bba96e121eaa1b8fdaf4c94c2a5b41db71c`。
- 原records.jsonl和NPZ路径/hash从canonical明确的原root `/scratch/e1192548/pi0243_stage_a_hf_20260922_retry1` 及对应清单定位；不猜本地Windows文件存在，不扫描全盘。

## 2. 单一新配方：M16-10K，不是旧模型exact resume

| 字段 | 本次冻结值 |
|---|---|
| experiment/family | `PI0244_M16_10K_V1`；独立output、receipt、HF prefix，不覆盖PI0243 |
| initializer | 与PI0243 A相同的PI0226 adapted baseline checkpoint-1000全部权重；不是旧A5000、PI0241-final或623593 |
| optimizer起点 | 新optimizer/scheduler/RNG；不是重置旧A后继续，也不是exact resume |
| TRAIN support | 原16窗口、原顺序、各window原observation index0和原H40；不删14个拒绝窗口，不只选2个WARN |
| steps/schedule | 10000 canonical optimizer updates，原官方cosine半周期、warmup10、warmup_ratio0、原peak LR1e-4；总时长10000 |
| optimizer其余 | 原adamw_torch、weight_decay1e-5、max_grad_norm1；betas/epsilon/parameter groups由PI0243实际resolved config/optimizer记录读取并写成数值，不能重新挑默认 |
| batch/loader | 原1GPU、batch1、accumulation1、workers0；原fixed loader/mask |
| loss/模型 | 原native masked flow objective及实际分母；projector+diffusion训练；visual/LLM/VLLN冻结；无新增项/重加权 |
| precision | 原BF16/TF32/load_bf16及gradient_checkpointing设置不变；不能为装入Corelab改量化/精度/容量 |
| 输入处理 | 原实际训练C_T及其train-mode；评估为已验收的label-free eval-mode；保留原augmentation/dropout/shortest edge96/crop等真实配置，不新增或关闭变换 |
| 动作/观测 | dense-relative H40、StateV1=10、universal132、active0:10、原mask、row-rot6d/frame/TCP、dt=.05，原语言/target边界 |
| RNG/采样 | seed22301，原flow-noise/time分布和stock sampler；不扫seed、采样步数或attention后端 |
| full saves | **2500/5000/7500/10000**；每份完整Trainer state；不额外写probe250/1000 |
| stock screens | **0/2500/5000/7500/10000 × 原16**，80 unique cells，最终只判10000 |

从原 `exposure_schedule` 构造10000条PCG64(22301)顺序：每轮完整排列16条，625轮，每窗恰625次；前5000条的字段顺序需与原5K曝光表完全一致。不是说新旧权重在5000步相同：cosine total_steps改变，之前每步LR就可能改变；训练噪声/增强实际消费也须记录，不由“同seed”宣称跨设备逐bit相同。

官方cosine依赖num_training_steps并在该时长内降至0。本试验是“训练预算＋对应cosine时程”的联合干预，不是单独多5000次相同LR的更新。以原5K终点继续、同时重启scheduler，会是另一配方，本轮不做。旧A1000与新1000、旧A5000与新5000也不能包装成相同训练轨迹的断点。

前次conditioning报告显示pixel_values因train/eval模式不同而有差异，其他已列条件叶一致。这里保留该实际模式差异；“16个源窗口”不等于训练始终仅见16份完全相同pixel tensor。本轮不声称在关闭全部正则/augmentation的理想记忆问题上测试理论模型容量。不要把mode差异悄悄消掉来制造PASS，也不再完整重复16窗条件审计。

## 3. Goal 0：窄准备，做完直接开训

### 3.1 复用与新增边界

接手时核对最新ledger与active jobs，避免与后续superseding计划冲突。原PI0243 FAIL封存，不重跑原训练/64 stock，不重扫全量数据/HF归档/全部参数。

新增只包括：新preset/run card、10000项曝光表、实际resolved差分、save/screen配置、独立family/storage binding及CPU分类器测试。复用PI0243 trainer、PI0241 fixed loader、Stage28、修好的registry/processor/native helper、checkpoint persistence与HF sidecar。不要从头搭训练/evaluator框架。

**禁止借用stage C仅为了得到max_steps10000。** 当前stage C会把view设成full；本次必须是view16，明确family PI0244_M16_10K_V1，最终判定分支必须执行fixed16四项门，不能落入通用stage C只看有final_rows的PASS分支。可以增加一个薄preset/显式stage-role，不修改sealed PI0243 preset的默认科学含义。

CPU回归验证：view16、原cohort、10000 schedule/前5000一致、save只四个step、screen80唯一cells、classifier final10000、错误family/欠缺mask/NaN拒绝、已存在记录的identity检查、恢复不双skip、HF sidecar正确5000→10000参数。日志/归档/报告失败不得触发重训。

运行前写 `resolved_diff.json`：除预算/相应LR时程/屏幕与保存时点/新identity和host-storage locator之外，所有科学配置沿原actual值。发现非预期漂移先恢复原契约，不把新随机默认偷偷带入。

### 3.2 用旧结果生成一次CPU误差分解

仅从原64个NPZ/records做：各checkpoint全部gate issue及重叠，原14条reject的精确类型、observed/threshold；逐窗XYZ bias、去bias残差、专家/预测相邻速度与尖峰segment；逐目标/阶段配对趋势。利用RMSE²=||mean(error)||²+mean(||error-mean(error)||²)分开整体偏位与抖动。

只重算有真实数组支持的量。缺失就标UNKNOWN，不为这个小报告重跑旧stock。完整性矛盾会影响结果有效性时需报告；普通结果差异/混合原因不改变已封存新recipe。这份sidecar可与资源/配置准备并行，不是新增的无限诊断门。

## 4. Goal 1：执行唯一10K科学轨迹并形成完整screen

通过上述薄准备与已有存储准入，直接运行新训练，不停在“READY等待确认”。首个正式update兼作原recipe的集成检查并计入10000，不能额外预热训练后把权重继续当initializer。

训练逐步记录sample ID、optimizer_step、真实LR、原native loss、有效mask与更新审计。沿已修好的全组/有效参数审计证明更新与冻结；不重用前4096元素假阴性，也不在每step做全模型比较。所有5K旧日志保留作为历史，不混入新10K lineage。

保存2500/5000/7500/10000的全部Trainer payload（模型、optimizer、scheduler、RNG、TrainerState和必要processor/provenance）。save结束才发布receipt，按实际容量和现成HF持久化链归档；一份未完成checkpoint不是resume点。工程中断从最高合法本family checkpoint exact resume；无有效新checkpoint可从原initializer/seed重启并保留工程消耗，禁止从某个成绩更好的失败attempt选起点。恢复用原10000 scheduler/cursor、无手工叠加skip，真实消费日志不能写常数代替。

stock评估使用真正保存后的权重和冻结eval-mode，独立进程，不污染训练RNG。按0→2500→5000→7500→10000完成每组16条。训练后一次顺序screen允许；若在保存边界screen，也不得用中间好坏调LR、选模型或提前解锁B。完整有效的中期REJECT不是工程异常，不重试；早期未达终点阈值不终止本10K计划。

first-valid cell按family/step/source/input/model/processor/runtime/seed身份冻结；复用必须验证完整证据，不只按source_id跳过。依赖/路径/IO坏结果可修复，数学上有效但分数差的结果不得丢弃。model/input非有限、真实优化器无有效更新或实际目标语义错误按科学/契约失败处理；抽样审计或工具错误先修清楚，不伪称science fail。

## 5. Goal 2：绑定正确权重的48个局部目标诊断

不重跑PI0242全矩阵，不重复32个native/helper parity。复用PI0242-R1已修复的processed-once路径和原生前向一致性证据，仅对改变的权重做固定诊断：

`{原initializer, PI0243-A5000, PI0244-A10000} × 原R1 bridge4 × 原t0/q10/q50/q90 × 单一冻结eval-mode C_T = 48 unique direct-flow cells`。

这项新授权将此前缺交的当前A sidecar明确补齐，不能用PI0242 final1000冒充。只能复用权重/输入/特征处理/time/noise/runtime均内容可证的已有格子；否则按新矩阵执行。三个权重顺序加载，尽可能同设备；跨backend标记限制，不能从不同设备挑最好。统计真实backbone/head调用和缓存复用，48不是总网络forward次数。

每个window固定同一完整132D初始noise，沿旧R1已验证的精确time值/映射和native target定义，保留active XYZ/rot6d/gripper分组误差、native loss与物理重构、原mask。hook不二次处理features、不改sampler、不零化inactive122。只为获已批准格子所需的backbone/head forward；无额外stock/noise/seed/step-count sweep。

teacher-forced重构是oracle诊断，不是policy成功；非teacher路径上的a_star-epsilon不能冒充唯一真实流场。此sidecar不改变A10K判定或挑中期，只保证有效FAIL后能报告local objective与free-running的关系。允许与主screen共享只读模型/特征，但不得让诊断种子影响训练。

完整stock gate已足以判定主结果；direct暂有工程问题不触发重训，也不能抹掉主FAIL。在当前固定预算内修完它再交失败机制包。主PASS且所有主身份有效可继续条件B，sidecar继续执行且不得无声取消。

## 6. 唯一最终科学门：仅step10000

80cells完整、身份/曝光/更新与持久化成立后，step10000的16条需同时：

1. 冻结SafetyGate `REJECT=0`，WARN保留、不新增WARN否决；
2. median(逐窗XYZ_RMSE) ≤0.020m；
3. p95(逐窗XYZ_RMSE) ≤0.035m；
4. median(逐窗SO3_RMSE) ≤0.050rad。

XYZ_RMSE=sqrt(mean_40(sum_xyz((pred-expert)^2)))；SO3_RMSE=sqrt(mean_40(geodesic(Rpred,Rexpert)^2))；跨窗percentile用原NumPy linear，全40点与原mask契约。不是mean distance，不是参考目标命中，也不是achieved TCP。gripper/最差旋转/逐段速度/首点问题单独侧报，不能借侧报新设结果后的主阈值。

报告五行科学表及16窗配对曲线；不挑7500或最优checkpoint。旧5K FAIL原样保留，不因为新试验成功而撤销。这里只有训练支持内的可生成性，不能声称独立泛化或多技能模型。

FAIL：`SCIENCE_FAIL_M16_10K_BUDGET — STOP_FOR_PI_OBJECTIVE_OR_OPTIMIZATION_DECISION`。保持selected_checkpoint=NONE；B/C/sim不启动。不得自动20K/50K、换seed、关闭augmentation、改dtype/loss/采样器。根据实际bias/temporal、局部目标和自由生成的证据，提交一项NOT_AUTHORIZED下一机制干预；不能再次只说“再翻倍”。

PASS：`PASS_M16_10K`，封存主判定和完整权重身份后，按第7节自动进入支持扩展，无需再申请一次例行PI GO。

## 7. PASS后的科学主线与family替换，明确授权而非暗中绕过旧FAIL

旧PI0243 A未通过，旧B仍不被该旧结果放行。本文件**新增条件授权**：若PI0244 A10K-final通过上述门，将该唯一权重作为新课程入口；原B/C/D/E/F科学定义与预算继承，但新建family并明确parent变更：

- `PI0244_M64_5K_V1`：A10K-final weights warm start，新AdamW/scheduler，原64TRAIN/5000updates；不是A optimizer的exact resume。保存1000/2500/5000，screen0/1000/2500/5000×64=256cells，整体64及旧16子集联合原四项通过。
- `PI0244_FULL_10K_V1`：仅M64通过，从其final5000 weights开始，新optimizer/scheduler；原9792TRAIN/10000updates，原TRAIN stats不重拟合。full saves2500/5000/7500/10000；原64 screens0/1000/2500/5000/7500/10000=384cells。实际train9792/screen64隔离、1000 model-only snapshot及1000/2500失稳检查必须在继续训练前实现，不在训练后追认早停。
- FULL四份完整后，沿原顺序baseline+四候选各1088CALIB=5440cells，以该新family同一C_T/runtime执行；原selector准入/排名函数、gate/threshold不改。原baseline只比较、不参与选择。全部候选拒绝则NO_ADMISSIBLE_CHECKPOINT，不选最不差。
- 唯一winner仍需对应step的64TRAIN及16保持门通过；不因winner差移除后选runner-up。
- 原A–D C0 one-chunk→四个任务全部通过→progress-preserving receding。使用原dense-relative→TrajectoryBatch→冻结WBC、原0.035m achieved-world-TCP scorer、原seed/cadence/tick/termination。必要参数只能从既有canonical绑定，不猜值、不现场调参。模型对专家RMSE≤.035不等于WBC任务成功。

原PI0243协议中所有阶段数据/窗口/失败规则保持。新family下游阶段不是重复实施两条B/C分支；旧分支不再并行启动。最高条件科学预算是本轮A10000+B5000+C10000=25000个新增canonical updates；只按PASS链消耗，旧PI0243已完成5000不混入新计数。

任何支持扩展、选模或物理任务有效FAIL即停止下游。成功终点仍为`TRAIN_SIM_CLOSED_LOOP_COMPLETE — STOP_FOR_PI_GENERALIZATION_REVIEW`。当前A/B selected_checkpoint保持NONE，只有最终FULL统一selection后才能建立候选。DEV-E/F、Final-F、DAgger、硬件继续锁定，PR #77不merge。

## 8. 资源/存储与工程自主

本地PC负责git、CPU tests、SSH控制；Corelab负责工程/真实Linux集成、能装下的screen和后续native sim；大显存训练优先复用此前已证明需要的H100/H200路径，不能为表示Corelab优先再重复一次已知OOM。新的科学精度/attention/量化变更不属于工程适配。资源依据真实峰值与固定工作量，不强制沿用48h训练模板给1–2分钟screen，不承诺队列何时可用。

HF/Project_Storage继续用已成功的sealed runtime和固定revision验证流程；token不入Git/日志。新prefix/family隔离，旧checkpoint不删除。复用已验证副本，screen顺序加载模型，不加载optimizer到GPU。

本新阶段四份full checkpoints容量要重新计算：设C为原同结构checkpoint实测上界（旧A约20.89GB是历史观察，不是新上界保证），retain-all至少4C+实际额外write/copy峰值+cache/screen/日志+原余量；约83.56GB只是四份payload估算，不含temporaries。现成rolling已通过consumer/receipt/恢复测试才沿用；不重新设计归档工具。每个协议checkpoint本地完整→durable核验后发布；不因HF失败重训。当前stage只有一个owner，恢复不能丢唯一有效点。

工程修复/恢复不设任意次数、累计token或walltime上限；每次须有根因与针对性低成本回归，单stage不并发重复，first-valid结果固定。无限工程权限不扩大科学样本/seed/steps，也不允许占用他人GPU或绕过站点策略。真实外部存储/权限/设备不可取得时报具体blocker，不叫science fail、不把排队当goal完成。不创建新monitor。

## 9. 最小交付和PC启动正文

只需一套新run card+resolved diff+schedule+storage binding，沿原训练/screen文件格式，新增旧64-cell CPU诊断表、48 direct records、主closure和条件B启动记录。小文件代码/测试/报告推execution分支，Issue #11先canonical，PR #77只pointer。文件存在/compile通过不是实际验收。

启动正文：

> 执行PI0244_FIXED16_10K_GO。完整读取本固定协议、PI0243 closure5775639138及执行基线69b587f。接受旧A科学FAIL，不重写、不启动旧B。新授权是一轮从PI0226 baseline1000重新初始化、原16TRAIN/seed/目标与native recipe不变、10000updates+cosine10000/warmup10的预算试验；不是从旧A5000续训。薄preset必须view16，不借stage C切到full。
>
> 复用已闭合输入/参数/HF/R1证据；只冻结新10000曝光表（前5000与旧表一致，每窗625次）、resolved差分、四份2500/5000/7500/10000保存、80个0/2500/5000/7500/10000×16 screen和独立family/storage。CPU读旧NPZ给gate分类及bias/时序分解，无额外stock。准备完成直接训练，不等重复确认。
>
> 独立screen按原gate和RMSE标准只判10000，不挑中期。完成initializer/旧A5000/新A10000×R1 bridge4×四t的48个direct对照，复用processed-once/native helper；旧PI0242权重不能冒充当前A。原train/eval mode差异保留，不擅自关闭增强、改loss/LR/精度/sampler或输入target。
>
> 新A10K主PASS自动以其final权重进入本协议明确映射的新B64/5K→FULL10K→统一选模→冻结WBC one-chunk/receding条件链；有效FAIL停OBJECTIVE_OR_OPTIMIZATION决策门，不自动再翻倍。工程自主修复不限次数，真实保存/恢复与first-valid证据完整，不重训补报告。DEV/F/DAgger/硬件/merge仍锁定。目标是产生终点科学结果并按条件前进，不只报排队或编译通过。

## 通用外部说明（不替代pinned项目源码）

Hugging Face官方scheduler说明：https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules#get_cosine_schedule_with_warmup 。cosine总时长是调度函数输入；本计划只据此解释为何10K是新时程实验，不升级项目依赖、不从最新文档引入其他scheduler。
