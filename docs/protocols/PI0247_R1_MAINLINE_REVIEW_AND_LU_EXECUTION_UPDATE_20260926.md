# PI0247-R1 — 科学主线审核与 L/U 执行更新

日期：2026-09-26（Asia/Singapore）。

Decision: `GO_REPAIR_EXECUTION_THEN_COMPLETE_FROZEN_LU_TO_SELECTED_SIM`  
执行标识：`PI0247_R1_MAINLINE_LU_CONTINUATION_GO`  
原科学授权：`PI0247_GENERATION_TO_SELECTED_SIM_GO`  
原release：`50e66efe6ccfbe814d04caf12499493821be7957`。必须同时读原主协议和execution binding notes。  
代码阅读基线：`7dbb5eeff362a358f9a0a852d0f5908cc5ab4040`；准备证据：`1f49235904ee349092b414f2062889fe82a4498b`；本次最后查到的execution head：`40a5b7499bd6c46a8294463ceadf10be5dd83961`。  
Issue #11 canonical；PR #77 pointer only，保持不merge。

## 0. PI结论与本次授权范围

**D0→D1→D2→L/U的科学方向没有偏离主线。D1未找到足够收益的runtime，正确保留R*=S4；D2两noise组均FAIL，正确进入已经批准的L/U，而非跳过M16送sim。**

目前的风险主要在执行实现：准备测试尚未覆盖真实PBS/在线归档边界、U的dtype/device与原stock一致性、训练后收口、临时checkpoint失效与恢复，以及无条件afterany链式启动。不能把这些风险说成模型科学FAIL，也不能用不断追加诊断或新loss回避已批准的L/U对照。

本文件维持原loss、系数、初始化、数据、seed、4步S4、H40、精度策略、2K终点、两噪声验收、后续扩量/selector/sim条件及总预算。**不新增科学arm或采样变体，不重新开放M4/M16，不增加模型诊断矩阵。**

允许并要求修复本文指出的实现问题，做针对性本地/Corelab回归、追加子card/runtime绑定，然后继续原L/U。工程修复不限任意次数；真实科学语义冲突或已列科学终点FAIL仍停止对应分支。完成数值/模式/保存恢复等窄准入即可自动开训，无需再次请求例行PI GO。

GPT本轮只读取GitHub账本、源码、报告及外部官方库文档，并发布文档/ledger；未SSH、未查实时qstat、未下载项目权重、未运行项目或GPU测试、未提交/暂停/取消远端作业。下面的代码问题是静态审阅发现；不冒称它们已经在Hopper发生。

## 1. 本次真实状态与证据

### 1.1 已读取来源

- [当前PI0247 canonical 5836195300](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5836195300)。此comment已包含后续执行更新，不再只是发布时的NOT_STARTED摘要。
- 原协议 `docs/protocols/PI0247_GENERATION_CONSISTENCY_TO_SELECTED_SIM_V1.md` 和 `PI0247_EXECUTION_BINDING_NOTES_V1.md`，固定于原release。
- execution准备head下的 `reports/evidence/PI0247_D1_MECHANISM_REPORT_20260926.md`、`PI0247_D2_FULL16_REPORT_20260926.md` 及其机器报告身份。
- `scripts/pi0247_lu_train.py`、`scripts/pi0247_u_rollout_overlay.py`、`hpc/pbs/pi0247_lu_train_h200.pbs`，以及commit `7dbb5ee` 和准备commit `1f492359`。
- [最新提交证据](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/40a5b7499bd6c46a8294463ceadf10be5dd83961/reports/evidence/PI0247_LU_SUBMISSION_20260926.json)。
- PR #77元数据：仍open/draft、merged=false。

### 1.2 状态不能再退回D1/D2未开始

| 阶段 | 最新来源支持的状态 |
|---|---|
| PI0246旧树 | 8000 updates、208 logical cells、全部旧FAIL保留；不重训 |
| D0 | 已复用既有证据；16窗结构性零padding经执行报告确认 |
| D1 | 64 logical cells：56新query＋8精确R0/S4复用；608新增head denoising evaluations；R*=S4 |
| D2 | 32 logical cells：28新query＋4精确D1/B1复用；B0/B1均FAIL；L/U条件成立 |
| L/U准备 | 同一P-final2000、同16窗/2000表、新optimizer/scheduler、LR1e-5；原card已seal |
| L提交 | `633185.hopper-m-02`，提交记录为Q/auto→small，等待ngpus |
| U提交 | `633190.hopper-m-02`，提交记录为H，依赖`afterany:633185.hopper-m-02` |
| 新训练结果 | 最新提交证据中的model_loaded=false、optimizer_updates=0；不等于此刻实时状态 |
| selected/sim | selected_checkpoint=NONE；没有新selected或sim成功证据 |

最新提交commit时间为2026-09-26 02:52:49 UTC，即10:52:49新加坡时间。接手必须再查actual owner/state，不能假定还在Q/H，也不能把旧“尚未提交”继续当当前状态。

D1 report SHA：`a7e5eb96a93e8726577b740286ab6115d01938cb5f1219941fd2cf103cd1b6b3`。
D2 report SHA：`3316e2b50b742fade18da34225af531d9716aa4f0c3b765695a52fe03ed0ad28`。
原L card：`5e37af9db3e3c209090442fb34a5432579f740046e6f67c3be33d196e7589bb8`。
原U card：`078cda5ee1341f98a527a5a17d6f48611610e07a041e6f4a111496ffc0a239fd`。
这些是已提交身份，不要在原位覆盖。修复用子card和新的immutable执行目录，并保留原父子映射。

### 1.3 D1/D2的科学解释

P-final2000的D1 S4中位XYZ为43.498mm；M4为41.172mm，配对8/8改善，但改善幅度没有达到预注册10%门；其p95为60.538mm，仍远高于最终35mm门。S16/M16中位数约65.053/64.364mm，且SO3中位数超0.05rad。

因此：padding路径有小幅效应，但本次不足以晋级；更细16步未改善所测模型。**不能说padding“完全无效”，也不能说更多去噪步普遍更差。** 这些是本矩阵的证据，不外推唯一根因。不得事后降低10%门、启用M4、扫8/12/32步或追加noise。

D2原16的结果：

| noise组 | valid | REJECT | median XYZ m | p95 XYZ m | median SO3 rad |
|---|---:|---:|---:|---:|---:|
| B0：原S0实际逐source packet | 16/16 | 1 | 0.0405363 | 0.0520768 | 0.0333199 |
| B1：共享P1 N1 | 16/16 | 1 | 0.0487148 | 0.0618682 | 0.0436456 |

B1比B0更差提示采样实现对质量有影响，但两组都不合格，不能挑B0宣布可用。两组是同16个训练窗口，不是独立泛化数据。由此进入L/U检验“局部目标 vs 真正生成终点目标”仍是直接的科学主线。

## 2. 必须修复的执行问题及证据范围

本节不是新科学停止树：只修受影响入口，不重做D0/D1/D2或整个仓库检查。

### E1. PBS离线训练与在线HF预检冲突（共同入口）

当前PBS导出`HF_HUB_OFFLINE=1`；`train()`在模型加载前调用`_verify_hf_prefix()`，最终在同一进程执行`HfApi.whoami()`和`model_info()`。这与HF离线API契约冲突。准备阶段在在线shell的认证PASS不能证明正式PBS环境也PASS。

修复：模型/processor进程保持离线；把需要联网的认证、目标prefix核验和上传放入明确联网的独立preflight/sidecar进程，且在import HF库前设置其环境。以内容绑定的小receipt向训练入口传递结果。不能为此把整条模型运行改成任意在线取最新版，也不能打印token或认证头。

CPU回归必须模拟实际wrapper导出的环境，验证离线训练不触网、在线sidecar独立访问、401/403/断网可区分。身份登录成功不是已经证明写权限；如使用既有写权限/归档收据，要准确写其证据范围。

### E2. 完成2000步后的未绑定常量与假阴性（共同收口）

阅读基线的`pi0247_lu_train.py`使用`LAMBDA_POS`、`LAMBDA_DELTA`核验总loss并写manifest，但本文件没有相应定义或导入。这类运行时名称错误不由py_compile保证发现，可能在训练完成后才显现。

修复：系数仍严格0.1/0.1，从一个明确的受测常量/冻结card来源绑定；用合成完成状态调用真实finalizer，覆盖loss核算和manifest写入，不增加项目模型训练。

同一finalizer再次以代表性projector delta是否大于零判断更新。若唯一失败是已知抽样探针为零，沿已有只读全张量reconciliation闭合，保留原audit。不要再把已知probe不足当作重新训练的理由；也不要虚称未检查的所有参数均更新。

把完成manifest的验证从“必须删除了tmpfs checkpoint”中解耦。checkpoint已完整持久化但本地仍保留是合法状态；退休/删除不是训练质量条件。

### E3. U的真实stock一致性尚不能由CPU toy代替

有利证据：U实现确实从捕获的纯noise开始、使用四次Euler更新、保留autograd，并使用非reentrant激活重计算；未见在该loop中注入专家前缀。CPU finite-difference toy和L/U物理loss等值检查是必要准备，但不是实际模型的S4一致性证明。

需要在首个正式更新前闭合以下具体点：

1. **dtype链**：原stock噪声/迭代dtype来自其实际生成路径；U当前直接使用native训练epsilon，parity只在请求dtype和epsilon.dtype完全相同时替换。训练action dtype是否等于推理状态dtype，当前preparation未证明。必须记录native action/epsilon、VL features、stock initial、每步Euler输入/输出dtype。若不同，保留native分支原样，只把共享epsilon以明确转换映射到真实stock入口dtype；保存转换前后张量身份，两条对照使用同一转换后noise。不额外抽noise，不把S4悄改为全FP32去满足parity。
2. **设备链**：trainer以CPU tensor创建scale/offset。U一般loss处有`.to(expert.device)`，但`verify_parity()`对CUDA动作调用decode时直接传入安装时scale/offset。要在两条parity分支一致使用与动作同device、规定FP32的统计张量。CPU-only self-check不能覆盖此CUDA路径。
3. **哈希**：当前`_tensor_sha256()`直接`.cpu().numpy()`。不要假定BF16可被当前NumPy无损读取；用保持原dtype字节的序列化/byte-view并连同shape/dtype记账。若另存FP32数值视图，必须与原字节hash分开命名。不得通过修改参与计算的tensor来修日志。
4. **完整parity**：原协议要求逐step/最终数值界以及原gate decision/issues一致。当前U parity主要检查raw/XYZ/每步差；补齐同一两次完整rollout的rotation/gripper decode和冻结gate比较，不加新的选分cell。阈值保持原raw≤1e-4、XYZ≤1e-5m及gate完全一致；原算法可bit-exact时优先exact。
5. **mode/随机/梯度**：保持raw与processed-once特征独立；不F(F(h))；U eval mode但梯度开启，退出及重算后恢复native模式。不得detach Euler状态、二次处理teacher-conditioned hidden state或触发RTC。第一正式backward记录全四步梯度与参数更新，不能把只是requires_grad=true当链已证实。

共享noise的dtype转换是对原“复用epsilon且重现部署S4”要求的明确实现绑定，不是新noise分布搜索或精度arm。若真实运行无法在原parity界内匹配，不放宽界限，停止受影响U并给出差异；不得用错误U结果去否定H3。

全campaign非选分kernel parity总额仍为8个完整rollouts，先查已消耗量；U参考/候选两次由余量扣除。CPU测试不限次数但不偷偷新增项目模型parity。已有有效parity不为了新报告重复执行。

### E4. tmpfs不是持久checkpoint，恢复入口目前不完整

当前实现强制把完整checkpoint写到`/dev/shm`，scratch仅要求5GiB以上；on_save同步上传并重试，成功readback后删除tmpfs副本。若网络/权限异常导致进程退出或节点回收，所谓“local copy retained”可能只剩临时内存文件。另有训练入口要求空prefix和空输出，CLI只有prepare/train，尚没有适用恢复/只读finalize的路径。这容易把一次归档异常变成无法恢复的训练重复。

工程更新优先复用成熟持久spool：每个协议checkpoint先完整写入或复制到节点退出后仍可访问的可靠接收端，再做HF发布/读回；输出、receipt、日志留在持久fileset。`/dev/shm`只可作为有容量和进程内存预算的临时加速层，不能是上传失败后的唯一源。没有合法持久容量时明确`PERSISTENT_CHECKPOINT_STORAGE_BLOCKED`，不赌16h作业和网络。

如果使用tmpfs，还要把tmpfs占用计入host RAM/cgroup预算；只看`statvfs`可用字节不证明进程内存或用户quota充足。Hopper的具体tmpfs生命周期须按实际节点/站点证据确认，本review未实测。

恢复必须区分：fresh run要求空科学prefix；resume允许且验证同family/step/原card派生关系的既有完整archive；finalize只读既有训练与归档，不执行Trainer。保留optimizer/scheduler/RNG/真实消费cursor和第一有效结果。不能删除目录重训来修收据，不能把volatile残片标成exact resume。

只需薄的恢复/收口适配和保存→接收→读回→恢复的synthetic测试，不造新的通用存储/调度框架。已有安全持久源在时，推理视图可先进入screen；完整归档未核验前不能删除唯一完整Trainer源。

### E5. afterany只代表进程终止，不能单独代表科学准入

最新记录确实已提交U，依赖`afterany:633185.hopper-m-02`。预提交一个被hold的后继本身不等于并行训练违规；危险在于L因输入冲突、实现缺陷或未完成而退出后，U未经分类自动运行。

允许保留afterany作调度唤醒条件，但必须加**类型化准入检查**；或待L训练收口后再提交U。不能只改成afterok后又让收口Exit1永远卡住科学有效的L。

- L2000已完成且训练/来源有效：U可以继续，不必等L的任务质量PASS，L普通质量FAIL或PASS都不取消U。
- L2000已完成，仅上传/报告失败：修复和核验来源/共享资源后可继续；不重训L，不将Exit0当唯一标准。
- L尚未完成、仅工程故障：先恢复原L或可靠地保存其同一轨迹状态，按串行要求执行；不由afterany盲启U。
- L暴露共享decoder/输入/目标/非法算术冲突：停止受影响两臂，不能靠U遮住共同问题。

## 3. 立即行动：先保护现有owner，再修最后一公里

接手先读最新canonical/commit和live scheduler及实际staged字节。

1. 若仍是633185 Q、633190 dependency-H，**先给U加独立人工hold，再处理L**，避免hold/delete L触发afterany自动放U。对已识别的共同入口问题，允许hold未开始的L，完成小修后恢复或单一替代提交；不是取消健康作业重跑。
2. 若L已经运行，不原位改源码/卡。核对它是否真的使用上述版本、是否已越过相应故障点、是否有真实更新。仅隔离明确不合规路径；健康的合法L运行保留，其报告缺陷事后只读修。U在修复验收前不得意外启动。
3. 如果已结束，读实际更新、checkpoint与错误类别；不要沿过时Q状态再提交一份。
4. 在独立worktree对E1–E5做窄回归，追加parent/child card与source hash差分。没有科学变化时工程准入可由PC agent自动确认。不得覆写老card、在正在执行的checkout里pull文档分支，或给旧结果换身份。
5. 用一份`pi0247_lu_execution_admission.json`列出pass/fail、actual owner、未解决点、原预算使用量和当前动作；不是新的全库audit。网络/MFA需要用户操作时只指出准确外部前置条件，不猜IP、不关闭主机密钥校验。

这些修复是为防止又花GPU发现普通Python/IO错误。修复有效即恢复L→U，不再停在“新的计划已读、pytest通过、等待GO”。

## 4. 科学主线压缩成一个近期交付：完成L/U判定

### 4.1 不变的研究问题

原共同P-final2000，fresh AdamW/scheduler，原16/2000表、每窗125次，seed22301，LR1e-5、WD1e-5、maxgrad1、cosine2000/warmup10，原精度/冻结范围/augmentation/statistics。

`L = L_native + 0.1*ell_pos(a_local) + 0.1*ell_delta(a_local)`。

`U = L_native + 0.1*ell_pos(a_free_S4) + 0.1*ell_delta(a_free_S4)`。

物理position/delta的单位、mask和decoder沿原协议；区别只在局部估计与完整四步生成终点。U的专家仅用于监督loss，不进初始化、prefix、state修正、RTC或模型条件。U生成全132维且使用原S4；D1未选M，故不能在U训练中顺手加padding固定。

L/U两臂各2000，先L后U，两臂都完成。U每update一native forward加一个完整四步可微rollout，activation重算单独计数。没有更多lambda、solver、precision或noise arm。实际first-update集成计入预算，不额外训练smoke。

### 4.2 同步准备screen，不等训练完成再修路径

D2的P/R*=S4/B0/B1共32逻辑cells是共享起点，**按完整身份直接复用**。只新增：

`L1000/L2000/U1000/U2000 × 原16 × B0/B1 = 128 logical cells`。

每个stage/profile单一owner，既有合法cells不重跑；B0、B1必须是D2实际tensor，不重新采样。更新screen支持L/U、1000/2000、family/源文件映射，推理端只需完整验证的推理文件而非optimizer。按原CPU validator核验identity、finite、shape、raw/decoded、metrics和原gate所有issues。

两noise组分别按原四项判定：零REJECT、medianXYZ≤.020m、linear p95≤.035m、medianSO3≤.050rad。不池化、不挑中期、不只报主noise结果。只有final2000决定各arm。checkpoint1000的正确坏成绩不阻止后半段训练。

最小科学报告为P起点/L2000/U2000的每组结果，及U−L同source/noise差值；辅助loss、saturation、逐步轨迹只用于解释，不替代stock结果。

### 4.3 结果驱动的既定后续，不新增等待门

- L/U中有合格者：原规则选唯一fit recipe，锁定weights/loss/S4/processor/noise，自动进入M64，不再请求例行scale GO。
- 两臂均有效FAIL：`PI0247_M16_HYPOTHESIS_SET_EXHAUSTED`，禁止M64/FULL/sim；交付当前对照支持和不支持的机制，不继续偷偷加训练。
- 真实实现/资源问题：工程修复不改变科学轨迹；错误U不得作为H3负结果。完整合法梯度图仍无法容纳则报`ROLLOUT_GRADIENT_CAPACITY_BLOCKED`，不能截断梯度/降低N冒充同实验。

## 5. 保留selected checkpoint与sim的条件通道，但不让远期准备挡住L/U

原PI0247第7–9节及绑定说明继续有效，未新增更严格或更宽松的数字门：

1. 单一合格配方 → 原64TRAIN、5000步，final两noise组及旧16子集均过门。
2. M64通过 → 原9792TRAIN、10000步，新FULL四候选2500/5000/7500/10000；沿原LR1e-5与既定early-collapse barrier。
3. FULL family完整后才解锁原1088CALIB×五profile；原selector admissibility/ranking不改，无合格者不选最不差。
4. S4选择阶段得到唯一FULL checkpoint才把selected_checkpoint改为该模型；runtime_id=S4与selection_stage=S4用不同字段。
5. winner自身TRAIN准入及原sim_binding闭合后，自动原A–D C0 one-chunk，4/4通过再progress-preserving receding。原WBC、gate、.035m world-TCP任务真值与clock/progress不变。
6. 最终`PI0247_SELECTED_POLICY_TRAIN_SIM_PASS`。不依据sim成绩换候选，不把离线成功称闭环成功。

当前不要为未开始的FULL/选模重建全部工具链，不做提前CALIB评估或WBC查询。M64准备可以在L/U末段用CPU完成；sim_binding在扩量阶段按原artifact机械恢复，真实缺失值不能猜。DEV/F、Final-F、DAgger、真机和PR merge继续锁定。

## 6. 资源与存储原则

本地PC控制与轻量测试；Corelab真实集成和已容纳的screen；Hopper用合法PBS auto承载已有容量依据的训练。当前请求设备guard是H200，auto并不取消该guard；不要求某一hostname，也不为节省显存盲删设备/精度检查。旧4080S全训练AdamW OOM不重复，U多步图不以推理能跑证明可训练。

只修实际依赖链的code/card/processor/cache/FFmpeg路径；不重新初始化全量视频树做无关验证。归档可用HF或已验证持久接收端；工作PC不下载大模型作为新中转，沿现有server-to-HF/服务器间路径。凭据值不进Git/log或回复。

上传/校验失败不重训，临时文件与持久checkpoint分开记账，host RAM/显存/磁盘峰值分别计算。不反复跑几十GiB写盘探针来代替容量规划；复用同fileset有效证据并核对新增占用。工程权限不包含删除旧initializer、数据、原结果、用户其他作业或安全检查。

## 7. 计数与ledger纪律

本次没有新科学预算。PI0247原上界19000 optimizer updates、离线query上界6696、sim上界和parity8保留；已经完成的D1/D2不充值。

D1为64逻辑/56新增；D2为32逻辑/28新增。D2的128 head evaluations包含4个复用cell对应的16次历史head执行；28个新S4 query对应112次新增head。报告区分逻辑完成、实际新增、复用与工程attempt，不把内部forward数等同query数。

D2复用后，L/U主screen完整表仍160逻辑cells，但当前只余128新逻辑cells。U两个非选分parity rollouts由原8总额扣除；实际重算计数写实测，不把期望常数当观测。

将原5836195300授权说明和后续执行分开：后续重要执行结果使用新的canonical comment，引用固定commit；不要不断改同一条授权正文使旧阅读时点失效。历史comment不删、不恢复覆盖；当前状态快照带observation时间、source commit、job IDs和sources，UNKNOWN就写UNKNOWN。

本文件只发布计划，没有代PC agent执行qhold/qsub/训练。下一有效里程碑应是受测L/U运行和final联合判定，不是再发布一个prepared/pytest/排队摘要。不上新monitor。

## 8. PC agent启动正文

执行`PI0247_R1_MAINLINE_LU_CONTINUATION_GO`，一并遵守原`PI0247_GENERATION_TO_SELECTED_SIM_GO`。完整读取本文件、原release两文件和最新ledger。

D1/D2已经有效完成，R*=S4；不重新评估M4/M16，不扫采样步数，不重跑D0/P1或旧科学结果。L/U的科学条件已经成立。最近提交证据为L633185 Q、U633190 H(afterany L)，先核对真实状态与staged代码，不能照过时状态重复qsub。

优先保护U免于盲目afterany启动。对未开始且确受本文缺陷影响的作业允许hold→最小修复→独立子card/runtime→恢复；健康运行不原位改代码或重训。afterany只能唤醒，U需类型化准入，L有限质量FAIL不阻止U，但共享语义/实现冲突不能盲过。

修共同入口HF离线/在线preflight冲突、finalizer未绑定lambda与旧projector探针假阴性；补持久checkpoint与同轨迹resume/read-only finalize，删除不是训练收口条件。tmpfs不作网络失败后的唯一源。

修U的真实stock dtype/device链、BF16无损哈希、同noise逐步parity与gate/issues；保持native训练不变，只按实际S4入口绑定U noise表示。保留全四步autograd、纯观测、原模式恢复和无teacher前缀。CPU toy不是GPU模型parity，后者在原剩余8预算内执行；第一正式update核验完整梯度并计入2000。

准备收口后连续L→U各2000，不再请求GO，不加loss/seed/预算。原P-final、同16/2000表、fresh optimizer、LR1e-5不变；L局部PT目标，U完整纯噪声S4终点的同一物理目标。

Corelab并行准备L/U screen；D2共享起点32复用，只新增四checkpoint×16×两noise=128。只final2000，两noise分别原四门，不择优中期。合格按原规则唯一fit recipe，自动M64→FULL→原selector→winner TRAIN准入→frozen-WBC one-chunk/receding；两臂有效FAIL按原协议停。

工程自主最小修复不限次数，但不重跑有效科学结果、不在H200测试普通路径、不变更WBC/gate/ABI/dt/target边界。正式FULL选择前selected_checkpoint=NONE，DEV/F/DAgger/真机/merge未授权。Issue #11追加新canonical，PR #77只pointer，不建monitor。

## 9. 外部官方依据（不是项目实测）

- HF环境变量：<https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables>；HF_HUB_OFFLINE下HfApi调用的限制。实际安装版本继续锁定，不升级runtime。
- Linux tmpfs说明：<https://kernel.org/doc/html/v6.10/filesystems/tmpfs.html>；tmpfs文件的内存/临时性质。具体Hopper生命周期与quota仍需本地证据。
- PyTorch BF16/NumPy互操作维护记录：<https://github.com/pytorch/pytorch/issues/109873>；本次只要求在现有runtime验证并采用dtype字节哈希，不以外部文档代替实际dtype检查。
