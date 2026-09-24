# PI0246 — P1已完成后的C/T科学收口与失败归因执行计划

日期：2026-09-24，Asia/Singapore。

Decision: `CONTINUE_EXISTING_C_T_THEN_FROZEN_BRANCHES`  
执行授权沿用：`PI0246_ENGINEERING_TO_GATE_PASS_CONTINUE_GO`。  
发布状态：`PLAN_PUBLISHED_NOT_EXECUTED_BY_GPT`。  
本轮代码阅读基线：`45d16c1b164ffe6a4e3de5c92f4a365cb3c8cc83`。  
Issue #11为科学账本；PR #77仅pointer，不merge。

## 0. 现在应做的事

**不要再从P0或P1开始，也不要重新提出一套10K配方。先完成现有C/T各2000步和配对评估，按已经批准的条件运行P/PT，取得第一版真实联合PASS模型或关闭本研究树。**

本文件只是现有PI0245/PI0246的执行补充，不增加loss、系数、训练步数、模型、seed、样本或科学门。原章程 `PI0246_GATE_PASS_CHAMPION_CAMPAIGN_V1.md`（blob `c17a7e6f40586b56b64b455995efafd6c1e6df52`）和R1 `847fb93acfe97851ce283aca5e261280c3cf2d49`继续适用。

当前最近读到的状态：

- [P1完成：5809200003](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5809200003)：2026-09-24 14:47:08新加坡时间，Corelab完成32/32首次有效queries；Hopper独立重验closure一致。P1不是0/32，不要重复。
- [C/T准备及C提交：5811477934](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5811477934)：2026-09-24 17:25:44新加坡时间，C唯一作业`630367.hopper-m-02`处于Q/auto→small；当时C/T更新与主screen均为0，T未提交。
- 最新execution head包含新paired screen实现；代码变更不等于已运行screen。不得把head自动视为630367的staged版本。

GPT本轮没有SSH、live qstat、原始远端NPZ或checkpoint读取、项目测试、GPU查询或训练。状态和数值来自已读取canonical；本文件没有修改630367或运行任何远端命令。

## 1. 已有失败的解释框架

以下是按证据和现有实验可识别性排列的假设，不是宣布唯一根因。

| 假设 | 已有支持与限制 | 现有计划如何检验 |
|---|---|---|
| 归一化目标与物理误差尺度不一致 | P0证明逐时间(40,3)XYZ反归一化尺度真实存在；原decoder重算一致。尺度会放大某些残差，但尺度存在不等于统计或decoder有bug | T用原物理decode后的增量残差；条件P/PT补物理位置项。statistics保持 |
| 点位拟合和时间连续性是两个剩余问题 | 原R2尾35..39贡献31.514451% pooled SSE；去尾诊断median仍42.731mm；旧R0/R2均没过位置门 | 主指标保留全H40；同时报告bias、去bias残差、39段速度和尾部贡献，不裁尾 |
| 低学习率继续微调可能有益 | 旧预算实验有改善，但尚无本次R0-final→低LR 2K对照结果；不能从loss低或参数变动推导已学好 | C是必要对照，T与C比较，不将T相对旧R0的全部改善都归因于新loss |
| 采样噪声与位置/尺度交互 | 新P1已多噪声配对完成，R0为11WARN/5REJECT，R2为4WARN/12REJECT；R2 roll11四窗仍REJECT。不能据此证明峰值如何移动或完全排除noise | 只分析已存P1 trace，不新增P1；主screen保存实际condition/noise，最终原32 noise复核完整报告 |
| 局部teacher目标改善未必转化成stock生成 | 现有局部估计和stock sampler不是同一种输入分布；当前C/T尚无结果 | 实际stock的主门决定结果，aux loss/局部重构仅解释；不把低aux loss当通过 |
| clip饱和可能限制辅助项梯度 | 新物理decoder镜像保留原clamp；旧32条stock XYZ未clip，不代表训练局部a_hat也不clip。尚无本轮饱和率曲线 | 同一次T forward记录saturation和各loss，不改clamp/ST梯度或自调lambda |
| 输入信息不足、观测歧义、head容量、attention/sampler差异 | 当前证据不足以优先定位到这些机制 | 本树耗尽后再选择一个新假设，不现在解冻backbone、换表示或扩大诊断 |

旧R0/R2的有效FAIL保留。模型在TRAIN上未达门，不能称“仅泛化失败”。曾经的路径、HF、配额故障不解释已经完整生成并独立验证的坏轨迹；它们只解释进度中断。已修复的F(F(h))及projector抽样假阴性不能反复当根因。

P1的16个cell/模型来自4窗口×4噪声条件，不是另16个独立TRAIN窗口，不能与原fixed16通过率当成同一个统计量。新P1也不是旧screen的精确噪声重放。

## 2. G0：接管现有作业，立即推进而不是重做准入

1. 查询630367真实owner/状态/节点和其后继，以及staged trainer/PBS/card hashes。最新GitHub head与提交时版本分开记账。
2. 正确运行则保护，不原地pull/修改运行文件；排队则保留单一合法owner。仅有新的明确工程缺陷时处理受影响执行，不因为发布本文件取消健康作业。
3. job已结束则读取实际训练产物，不把旧Q状态继续当现状。只有归档/finalization失败时，用既有权重和日志只读收口，不重训。
4. P0、P1、输入/模型/HF小证据直接复用；不存在新内容冲突不重做全量审计。

固定已知身份：
- P1 closure SHA256：`77712a2598ae7f8d8a076139c4160ec480c4d9ff624b2e45be9a0a51a55cc70b`。
- C card：`f3a9fdffffce91d97891dc0b919c5b64cb12a8819292eb2ad78e34c7c3188601`。
- T card：`469c0a29571e05c30d50cf63ab0a13d8e3495a91469044bcf14feb840600e938`。
- cohort：`3d679d9cf5814d6aca63ca52e5953e1a50fb6097e3d8d67a7b875ab1178e36cd`。
- 2000 schedule：`b8d0fe1b262f70c35753c3cecba5d13a75e07a5e2f0a3f112a368da9751523bf`。

若发生合法host/staged binding修订，保留父子关系、全部旧hash和修订原因；不把原文件覆盖成新版本。

## 3. G1：C臂必须产出训练与动作质量的完整答案

C是`PI0245_C_NATIVE_M16_2K_V1`：R0-final10000 weights-only初始化，新AdamW/scheduler；原16、同2000表、每窗125次；LR1e-5、WD1e-5、maxgrad1、cosine2000、warmup10、batch1/accum1/workers0。原精度/冻结范围/augmentation/tau-noise分布保持。

- 第一正式更新确认实际LR、loss、参数成员、梯度与更新；首步计入2000，不增加smoke更新。已有审计覆盖什么就报告什么，不夸成全模型逐元素证明。
- 保存checkpoint1000和2000完整Trainer状态，并建立持久收据。两份checkpoint应绑定同family、同card、真实step和文件内容。
- 2000消费记录要与sealed表逐行核对`optimizer_step/fixed_index/episode_index/source_episode_id`等原有字段，不只比较总数。没有的历史字段标UNOBSERVED，不填常数冒充观测。
- 1000只是检查点，不择优、不中途换seed/LR，也不因普通未达最终门取消2000。
- C结束后，不等PI再次GO即可执行已授权T；C有限的质量FAIL是对照结果，不是停止T的理由。真实模型非法值/来源矛盾另处理。

当前源码再次用代表性projector delta是否>0判完整性。若这个代表性探针为0，不直接标“训练完全没更新”或重训；沿已批准的只读reconciliation检查精确有效embodiment参数，保留原audit并限定证据范围。只有获得真正无更新/错误optimizer等正面证据才作科学判定。没有冲突则不额外扫描全部权重。

当前trainer入口只有prepare/train，且拒绝已有training目录。工程恢复若需要resume或finalize，复用官方Trainer恢复和既有只读finalizer，新增薄入口并CPU测试。不能把删目录重跑当exact resume，也不能加载旧optimizer做跨arm暖启动。

## 4. G2：T臂按已冻结的物理目标运行，先确保可解释性

T为`PI0245_T_PHYSDELTA_M16_2K_V1`。与C同起点、同2000表和全部训练配置，只增加以下项：

`a_hat = x_tau + (1-tau)*v_hat`

`E_k = D_xyz(a_hat)_k - D_xyz(a_star)_k`

`L_delta = mean_valid_transitions || ((E_(k+1)-E_k)/0.05s)/(1m/s) ||_2^2`

`L_T = L_native + 0.1*L_delta`。

D_xyz使用P0封存的逐时间scale/offset、相对参考姿态和原clip。三坐标平方求和后按有效39个transition平均，不额外除3。所有量来自同一次native forward；不从平方action_loss推断符号，不重复采noise/tau、处理backbone或新增teacher前向。

### 4.1 日志增强不是新实验门

当前repository overlay只保留first_forward和调用数；它不等于原协议要求的全程各分项/饱和证据。T启动前在独立worktree补一个只观察、无随机调用的薄记录器，在原1/10/100/1000/2000里程碑持久保存：

- L_native、L_delta、0.1L_delta、总loss和tau；
- 39个有效transition及scale binding；
- 局部a_hat越过clamp边界的XYZ比例；主screen raw裁剪比例另列，二者不得混写；
- 从现有梯度/audit可读取的有限性、实际LR、参数/moment dtype与sample身份；
- 可合法获取的同forward noise/输入hash，不能为了日志另一次forward/backward或重新采样。

C若已提交/运行，不能为日志缺失原地改图或重训C。能从现有产物获得就整理，缺失则披露；新的T记录不自动赋予“C/T训练每个随机张量严格相同”的论断。两臂共同screen的实际输入/噪声配对仍必须成立。

新增观察代码先在本地/Corelab用synthetic/既有CPU fixture证明lambda0、原输出/RNG后缀、梯度和单次调用等价。只添加已有授权的日志，不增加模型查询或debug optimizer steps。若T已运行，按同样规则保护既有执行，缺日志不追溯重训。

### 4.2 不做在线调参

即使辅助loss很大、饱和率高或native loss下降较慢，也只记录；不能动态调lambda、解除clamp、改p_ref/v_ref或替换loss。非有限量及确切实现错误按科学/工程证据分类。有限的较差表现继续到已批准终点，不能以成绩不好判“实现有错”重跑。

C→T两臂都必须完成，C先PASS不能省略T这个对照设计；后续P/PT才受“首个冠军关闭未执行arm”规则约束。

## 5. G3：在Corelab完成配对主screen，而不是等所有训练和归档都结束才准备

新screen源码`pi0246_paired_screen.py`已在本次基线出现，但没有运行结果，不能把提交代码记成screen完成。只读审查和CPU fixture可与C训练并行；原协议预算中的共享step0也可在Corelab合法执行，独立进程避免改变训练RNG。同一profile不能并发重复。

主80逻辑cells：S0（R0-final）×16，加C1000/C2000/T1000/T2000各×16。优先顺序S0→C1000→C2000→T1000→T2000；已封存同身份结果可恢复复用。

- S0必须保存实际label-free condition张量，以及stock产生的完整初始noise与实际注入dtype；后续checkpoint真正读取同一packet，不再次调用随机builder冒称相同。
- 模型get_action可原地修改输入容器，调用侧按既有语义使用独立副本并保护封存源；CPU fixture检查无分支污染。
- 对所有profile绑定精确checkpoint/receipt、family、step、来源source IDs、processor/stats/metric/gate、noise及condition；仅源ID或文件名相同不足以复用。
- 同一paired screen采用同一已验证backend/profile；Corelab已证明可运行P1，但训练和推理容量不同，不能外推训练必然装下。旧Hopper的成绩与新Corelab成绩不作为逐bit同噪声历史重放。
- 使用现有CPU validator复算全40点RMSE、SO3、39段速度、gate决定/issues以及NPZ/hash/shape/finite，分量报告不替代原主门。
- 机械中断仅恢复缺失固定cell，先核对attempt/已有输出；不覆盖first-valid。模型真实nonfinite不是可重抽种子的工程失败。

最终C2000/T2000 ×原bridge4×原N0/N1/N2/roll11的32个noise复核仍执行；不把roll11当第四个独立随机种子，不混入主16门，也不隐瞒其他noise失败。

原P1不再新增查询。可从已保存trace做CPU位置/noise/尺度贡献描述，但不是C/T启动或主判定的新前置条件，不声称四窗就能确定唯一根因。

## 6. G4：联合判定、因果边界与下一分支

只按每臂final2000判定：

1. 原SafetyGate REJECT=0（WARN保留）；
2. median逐窗XYZ RMSE≤0.020m；
3. p95逐窗XYZ RMSE≤0.035m；
4. median逐窗SO3 RMSE≤0.050rad；
5. 完整训练、输入、权重、噪声、16窗及独立复核证据成立。

逐窗XYZ=sqrt(mean40(sumXYZ squared error))；SO3按40点geodesic RMS；quantile沿原linear。不得用截尾、mean error、loss、teacher-forced局部估计或reference-target距离替代。

必须交付同一表：S0/C1000/C2000/T1000/T2000的有效N、gate counts、XYZ median/p95、SO3 median、峰值线速度分布/segment、首尾/clip与尾段SSE。另报C2000相对S0、T2000相对S0，以及T2000相对C2000的同source差值。

解释矩阵：

| 结果 | 可以支持什么 | 不能自动声称什么 |
|---|---|---|
| C与T相对S0都改善而差异小 | 该低LR/新调度/额外2K微调组合有收益 | 物理delta项必不可少 |
| T相对C减少物理增量误差和拒绝，XYZ/SO3保持 | 所测固定配方支持时序项贡献 | 已泛化、跨seed普遍有效 |
| T速度更好但位置仍过门失败 | 连续性和位置目标仍未同时解决 | 已取得pass-gate model |
| T更差、辅助项饱和/主项权衡明显 | 此系数和实现下存在权衡线索 | 所有时序目标都无效或可当场换系数 |
| 主16通过而噪声复核弱 | 有限TRAIN主screen成功，稳健性弱 | 稳定部署或通过独立泛化 |
| C/T都失败 | 当前两个终点失败，执行下述已批准条件树 | 直接无限加步数或改sampler |

同一权重、实际condition/noise配对降低了抽样混淆；没有完整训练RNG对应或重复训练seed，仍不声称统计显著性或唯一因果。只有分量loss日志不能判断整个模型是否学好。

### 6.1 保持既有条件树

- C/T任一主PASS：按照原章程比较p95 XYZ，再峰值速度median，再XYZ median，再C的确定性规则冻结冠军；保存另一臂，不运行P/PT。
- C/T均FAIL，且至少一臂XYZ median或p95仍FAIL：自动运行P，`L_native+0.1 L_pos`。
- P有效FAIL：自动运行PT，`L_native+0.1 L_pos+0.1 L_delta`。
- 若C/T的XYZ都通过而只剩其他问题，则P/PT不适用，停新假设门。

`L_pos=mean_valid_points ||E/(0.05m)||²`。P/PT均从同一R0-final10000重新开始，新optimizer/scheduler，各2000、同曝光表/超参数、保存1000/2000，不接前一失败arm、不加XYZ4。P/PT各增加32主cells；若P或PT成为冠军，另执行原16个固定噪声复核。

本文件没有新增这两个arm；当前C/T入口只支持C/T，未来需薄preset与单次forward公共loss组合、各自sealed card及CPU tests，不能仅改arm字符串。未满足科学准入之前只能CPU准备，不backward。

预算继承：C/T/P/PT至多8000 canonical updates；整个PI0245/0246至多224 logical stock cells，其中已完成P1的32包含在内，不重新充值。实际内部forward、工程attempt和资源消费另计。

## 7. 不让工程故障再次吃掉科学结果

### 调度与机器

本地PC做代码与SSH；Corelab做CPU、已容纳的推理/screen；大训练使用已有已验证配置。当前snapshot的train()确实有`exactly one H200`检查，而不仅是PBS auto。因此不能只改队列/让auto派到H100就宣称同入口可跑。630367保持已封存配置；未来若调整为能力准入，先依据实测显存/兼容性和配对runtime要求进行独立工程绑定，不能删除检查后盲投。不要在GPU上反复测路径、HF登录、pytest。

### 存储

17:25账本的约90GiB scratch/358GiB Project_Storage只是该时点观察，不是长期配额保证。C与T两份完整checkpoint各自要加optimizer/RNG、原子临时写入、reader/归档占用。C完成后、T启动前重新核对实际写入fileset和并存峰值；不能把只够C的容量证明直接复用为C+T。

计算节点不能访问Project_Storage时继续scratch工作、由可访问的登录/传输端持久化，记录真实receipt。不要跟随账本中指回Project_Storage的tar symlink进入计算依赖。HF/归档失败只修传输，不重训；无receipt不删除唯一源。新screen仅需推理权重，不搬全optimizer到Corelab GPU。

### 审计、恢复与代码身份

C运行源码SHA、T源码SHA、screen SHA独立绑定。不要将本文件的文档commit直接pull进已封存执行checkout。代码head不等于已部署文件，也不等于运行结果。

工程问题自主定位→最小修复→本地/Corelab定向回归→恢复；没有任意工程重试上限。真实权限/MFA/设施不可达可准确报外部blocker，不绕过权限、不承诺GPT后台执行。连接中断后先核对qsub是否受理，禁止双投。

不为缺少漂亮日志重训或新增模型诊断；已运行证据不足则披露，而不是补造。必须区分传输损坏/错绑定与真实科学失败，不能用工程重试替换坏成绩。

## 8. 最终交付与停止门

近期最小交付：
1. 630367或合法后继的实际状态和C完整2000更新/1000&2000收据；
2. T同规格训练结果与可取得的loss/saturation证据；
3. 主80cells及32noise复核、独立校验、C/T效果表；
4. 按原条件P/PT的实际推进或冠军完整冻结。

冠军交付包括完整checkpoint、fixed HF revision/inventory、可读恢复说明、fit_gate_champion_manifest、精确配方和实际condition/noise、所有逐窗及噪声结果。不只交付PASS token。

成功：`M16_GATE_PASS_MODEL_FROZEN — STOP_FOR_PI_SCALE_CONFIRMATION`。研究树耗尽：`PI0246_GATE_PASS_TREE_EXHAUSTED — STOP_FOR_PI_NEW_HYPOTHESIS`。不能保证该树必然产生冠军。仍保留`selected_checkpoint=NONE`；M64/FULL/正式CALIB选模/WBC/sim/DEV/F/DAgger/硬件/merge继续锁定，不借长程goal跳过新的科学门。

Issue #11先canonical，PR #77只pointer，不新建monitor。复用一个campaign_state/ledger，不新增通用调度框架或新科学编号。

## 9. 可复制给PC agent的启动正文

执行原`PI0246_ENGINEERING_TO_GATE_PASS_CONTINUE_GO`及本执行清单。先读5809200003（P1完成）、5811477934（C提交）、原PI0245/PI0246章程和R1；45d16c1仅是本次阅读基线，先核对实际job/source身份。

P0/P1已完成，禁止重跑。接管630367或其合法后继，不假设仍Q、不重复提交、不改健康staged文件。首先完成C2000训练、1000/2000完整保存与只读收口；然后按原授权完成T2000，不能因C有限FAIL停下，也不能C通过就省略T。

T使用原同forward物理delta公式，原scale/decoder/clip/系数/单位不变。T启动前只补已有授权的loss、tau、saturation和实际更新日志；无额外随机函数、模型forward或backward。不为C历史日志缺失重训或假称严格训练RNG配对。

并行在Corelab准备/执行预算内共享S0和paired screen：S0、C1000/C2000/T1000/T2000×16共80；实际condition/noise冻结复用，完整模型/文件/指标/gate核验；终局32noise照常。全部记录但只判各final2000原联合门，不选中期。

C/T主PASS按原规则冻结冠军。两臂都FAIL且XYZ仍FAIL则自动P，再按P有效FAIL执行PT；全部从同一R0-final重新开始，各2000，原系数和2000曝光表不变。无新lambda/seed/额外step/XYZ4/sampler更改。最多8000updates/224总logicalcells包含已完成P1，不重复充值。

实际资源使用已验证入口：当前C代码硬检查H200，别只改queue盲投H100；screen/Corelab已可运行。T前重核C+T并存存储，不误把90GiB当永久quota。上传/报告错误只读修复不重训；代表性参数probe零变化先按旧规则核实，不直接宣布整个projector没更新。

工程自主最小修复不限次数，当前研究树有下一分支的普通FAIL直接前进。冠军后停scale确认，树耗尽停新假设；selected_checkpoint=NONE，M64/FULL/选模/WBC/sim/DEV/F/硬件/merge未解锁。Issue #11记实际结果，PR #77只pointer，不新建monitor。