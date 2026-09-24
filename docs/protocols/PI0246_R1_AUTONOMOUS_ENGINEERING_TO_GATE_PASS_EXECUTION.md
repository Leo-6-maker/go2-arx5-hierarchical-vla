# PI0246-R1 — 从工程恢复连续推进到 Gate-Pass 模型交付

日期：2026-09-24（Asia/Singapore）  
Decision：`GO_AUTONOMOUS_ENGINEERING_AND_FROZEN_CAMPAIGN_CONTINUATION`  
Authorization：`PI0246_ENGINEERING_TO_GATE_PASS_CONTINUE_GO`  
发布状态：`PLAN_PUBLISHED_EXECUTION_NOT_STARTED`  
审核基线：`4d2348843680aec26a58c6f4f0199ac66e07fe0e`  
Canonical：Issue #11；PR #77 仅 execution pointer，禁止 merge。

## 0. 总目标、权威与完成定义

本文件落实用户最新授权：工程问题由 PC agent 自主定位、最小修复、测试、恢复并继续，完成已经批准的训练微调全流程，争取交付第一版真实满足原联合门的 fixed16 模型。这里的完整流程包括输入绑定、实际训练、checkpoint 保存/恢复、自由生成评估、独立复核、持久归档和最终模型说明，不以 compile、import、作业提交或 loss 下降作为目标完成。

**本文件不新增科学配方或预算。** 保留并连续执行：

- `docs/protocols/PI0245_SCALE_AWARE_TEMPORAL_DIAGNOSTIC_AND_PAIRED_PILOT_V1.md`，原发布 commit `f33f345a2419b2eb59a542c2117834a799a2d567`，canonical `5799851237`；
- 当前基线内的 `docs/protocols/PI0246_GATE_PASS_CHAMPION_CAMPAIGN_V1.md`，Git blob `c17a7e6f40586b56b64b455995efafd6c1e6df52`；
- 最新进度 canonical `5807158828`、PR pointer `5807161634`；
- P0已完成证据 `5800283247`；旧R0/R2科学结果保持不变。

PI0246已将PI0245双臂终点后的普通失败处理扩为固定C/T→条件P→PT研究树；不要在C/T普通终局FAIL后误停回旧PI0245的例行确认门。本文件补足工程执行和持续推进规则，不改变该研究树。

成功终点：

`M16_GATE_PASS_MODEL_FROZEN — STOP_FOR_PI_SCALE_CONFIRMATION`

必须同时满足：固定16窗与模型/输入/噪声/训练证据有效；原SafetyGate零REJECT；逐窗XYZ RMSE中位数≤0.020m；p95≤0.035m；逐窗SO(3) RMSE中位数≤0.050rad。WARN保留，不能另改为全ACCEPT，也不能将零REJECT单独视为充分条件。

目标不构成保证。有限研究树全部有效失败则交付已完成证据和一项新假设，不能为了目标文字必须出现PASS而降低标准或无限训练。M16冠军是训练支持上的拟合模型，不是泛化证明或WBC候选；`selected_checkpoint=NONE`，M64/FULL/CALIB/selector/WBC/sim/DEV/F/DAgger/硬件仍未放行。

## 1. 本次审核确认与未确认事项

已实际读取最新canonical/pointer、4d23488 diff、完整P1 PBS、P1 loader/路径映射/模型预检入口、PI0246章程与P0证据。本次查询未返回上述进度之后的新科学结果。GPT没有SSH、live qstat、远端磁盘访问、项目pytest、模型运行、训练或调度操作；文档发布不等于这些操作已经执行。

账本支持：

- `629669`在hopper-22于Python之前退出；项目补充组存在，但该节点的根文件系统中没有`/Project_Storage`。这只支持该次节点的挂载事实，不扩大为所有计算节点都没有该挂载，也不改称用户目录普遍无权限。
- `4d23488`将模型/输入/输出定位改为scratch，按模型拆分，使用`ARCHIVE_PENDING`等待登录端归档。R0固定revision的12文件和4个标签回环已通过记录的预检。
- 6/6 exact Stage10测试发生在最后一次log-label修改之前。不要声称最终推送字节也已运行这些测试；补一轮受影响的轻量预检，不重做全项目。
- P0尺度、decoder镜像与CPU数学证据已完成。P0原文明确“真实训练图准入与P1 stock hook仍待完成”；不能将CPU数学PASS扩写为已经执行了实际模型更新。
- 最新记录中P1=0/32，训练=0，无活动owner；这是ledger时点信息，恢复控制后重新核对。DNS故障是本机无法解析目标名的观察，不等同于Hopper服务宕机。

### 1.1 必须最小修复的一项残留依赖

当前`hpc/pbs/pi0245_p1_stock_h200.pbs`虽然只执行`${MODEL_NAME}`，却仍分别要求R0和R2两份`pi0245_inference_restore.json`可读。只准备R0时，这可能制造与当前阶段无关的启动阻塞。

修复为：当前作业仅要求`${MODEL_ROOT}/pi0245_inference_restore.json`及该模型真实所需文件；完整P1汇总时仍必须要求R0与R2各16个合法cells。不能删除当前模型的内容/身份核验，也不能将“另一模型尚未暂存”误报为科学数据损坏。重复NOISE赋值可以顺手清理，但不将其夸大为已证实故障原因。

以实际wrapper调用链做fixture：只有R0存在时R0阶段可通过无模型预检、R2阶段应准确失败；反向同理；非法MODEL_NAME、缺card/噪声/代码/模型shard、错误hash仍fail-closed。生产PBS的GPU段用显式preflight-only分支或测试stub隔离，不为测试意外调用模型/调度器。

## 2. 工程自主权：做什么可以直接继续

无需再次向PI申请的操作包括：

- 诊断既有SSH alias、DNS、已授权网络/VPN/jump路径与login-shell环境；修复可回滚的用户态别名/本项目配置。
- 修复定位器、host binding、路径/参数/模块导入、shell、日志、序列化、读取器、恢复与归档等机械实现，保持科学语义和内容身份。
- 建立/复用模型与输入的校验副本，迁移本次新输出到已授权存储；恢复HF上传和receipt，不扩大删除范围。
- 实际代码的CPU回归、Corelab真实集成、合规调度申请、已定位机械故障后的重新提交或checkpoint exact resume。
- 补齐已授权C/T/P/PT所必需的薄训练入口、同一次native forward观察/损失overlay、screen、独立验证和收口，不另造通用框架。

工程次数不设固定上限，也不设累计工程walltime的人工终止门；但每个PBS作业仍必须申请合法walltime/资源并遵守站点配额。每次重试必须有新证据或针对性修复，禁止重复相同失败命令赌节点，禁止并发同阶段副本，禁止将低成绩称为工程异常重跑。

需停止受影响科学路径的事项：必须更改loss/权重/LR/seed/表示/H40/dt/stats/frame/目标输入/WBC/gate、当前标签或模型身份真实矛盾、实际模型/优化器非有限且没有已证实的机械原因、研究树耗尽。

真实不可自行解决的MFA、授权、站点故障、网络或存储问题，可以交付`EXTERNAL_ACCESS_BLOCKED`或相应外部blocker；必须列明已查到的事实、合法替代路径尝试和最小缺失条件，不写“重试预算用完”。不得承诺没有执行能力的后台继续。

## 3. G0 — 恢复控制端，不把DNS问题带到GPU

1. 在本地读取当前使用的OpenSSH配置与可用的既有alias。先`ssh -G <existing-alias>`查看实际HostName、User、Port、ProxyJump/ProxyCommand和IdentityFile来源；不要凭聊天中的名字创建新账号或连接随机IP。OpenSSH的-G输出解析后的配置并退出，并不是远端连接测试。
2. 对解析出的真实HostName做本机DNS查询（Windows可用`Resolve-DnsName`），检查当前已有网络/VPN路径。分别记录DNS失败、TCP不可达、host-key错误和认证错误，不能混写。
3. 优先复用本机已配置且已授权的校园/VPN/jump路径。允许在既有可达Corelab上检查其已获授权的Hopper访问路径；不能把Corelab作为未授权绕过网关。临时IP替代仅能来自当前可信DNS或管理员确认，并保留正确主机身份校验。
4. 禁止关闭StrictHostKeyChecking、清空known_hosts、拷贝私钥到其他机器、将token写日志、批量扫描IP、擅改系统DNS/hosts或提权挂载存储。必要的用户MFA/VPN授权无法自动完成时，明确外部前置条件。
5. 连通后读取host/uid/groups以及已有合法login-shell/module下的scheduler命令。找不到qstat不等于没有作业；不能仅依据旧ledger再次qsub。
6. 核对629669终态及日志，归档失败输出；查询同project/current campaign的真实owner，确认没有其后已提交的任务。SSH中断导致某次qsub结果未知时，先按job name/提交记录查是否受理，不能直接重投。

控制端阻塞期间，继续本地和Corelab能够独立完成的CPU代码测试、loss与reader准备、manifest整理；不需要等DNS恢复才开始所有工程工作。只有在无法确认其他owner时，不得启动会重复科学cells的替代执行。

最小产物：一份不含秘密的access诊断摘要与`active_execution_state`。可合并进已有campaign state，不新增多套ledger。

官方命令参考：OpenSSH ssh/ssh_config手册与Microsoft Resolve-DnsName文档；它们不证明项目实际网络可达或NUS站点权限。

## 4. G1 — 将路径、容量与归档变成一次可验证的执行契约

### 4.1 输入/工作/持久存储分开

- `source_root`：原sealed数据、权重及HF fixed revision身份；原文件保持。
- `compute_input_root`：已验证的当前计算节点可读副本。
- `work_root`：当前任务写tensor/checkpoint/log的真实文件系统。
- `durable_root`或HF publisher：任务结束后仍可取回的持久副本，通常由拥有挂载和凭据的登录/传输端执行。

不把同名`/scratch`默认为跨所有节点同一存储。记录host、findmnt/文件系统标识、realpath、读写探针、bytes/inodes、capacity证据等级。前次节点能读不能代替下一节点启动复验。

### 4.2 完整依赖检查，不只检查PBS里的字符串

在同一resolved_launch中列出card/seal、cohort、schedule、noise bank、4个P1观测及source records、模型index/shards/config、processor/meta/relative_stats、实际import源码、Python/动态库、cache/tmp/output/log路径。解析依赖时保持原card只读，用单独host overlay和逐文件bytes/hash绑定副本。

预检和正式运行共用同一resolver；不要全局字符串替换Project_Storage，也不要仅grep PBS没有这个词就宣布所有Python依赖都已可见。含绝对路径的旧证据无需覆盖，可由受测host binding定位同内容文件。P1仅读取4个封存观测，不为此重新初始化/扫描9792条视频。

提交前验证：脚本最终字节/hash、Unix换行、shell语法、实际argv、MODEL_NAME单模型隔离、import身份、来源allowlist和无模型端到端fixture。既有local和Stage10结果内容一致可复用；只测试此次受影响部分。

模型载入前的计算节点启动块：确认当前host上的所有已解析输入可读及小hash匹配、真实输出容量/写入可用后直接继续原定计算。这是正式任务内必要的节点验收，不另外排一个GPU只做通用CPU探测。若节点特有挂载异常，保存证据、在控制端修好绑定后再恢复；不得猜一个新路径盲投。

### 4.3 Archive pending不是已持久化

4d23488的`ARCHIVE_PENDING`仅是提示，不是归档完成凭证。PC执行循环必须接手：读取已完成cells的manifest→复制到durable目标→目标端内容核验→写receipt→更新状态。部分作业退出也先保存已完成合法cells和失败日志。

P1可分R0/R2作业，每个只暂存所需推理文件，不复制optimizer到GPU。输出按model/source/noise完整隔离；汇总时内容身份一致。未获receipt前，scratch结果不是可删除的唯一副本。

训练前按实际full Trainer state大小，计入当前和下一checkpoint、临时序列化、缓存、上传和screen峰值；不能继续用目录可写代替容量。沿已批准的实际容量验证方法，quota未知就准确保留证据等级；旧释放量或已释放reservation不提供永久保证。持续检查余量，checkpoint写出后及时持久化，不等整个训练结束才复制唯一结果。

只可清理由本campaign新建、可重建且已无reader/恢复依赖的缓存；旧initializer/失败模型证据/数据和唯一durable副本不删除。不为修HF、report或stage-out重新训练。

## 5. G2 — 完成32个P1逻辑cells，而不是只让作业启动

保留原sealed bank与原bridge4：R0-final/R2-final × A–D × N0/N1/N2/N0_roll11，总共32。已有bank的实际张量/hash复用，不重新生成一次假称相同。

执行顺序：R0的16个cells→验证封存→R2的16个cells→两模型完整性与原输入/噪声配对检查。每个模型一个驻留进程顺序运行其cells；已固定的噪声显式注入。按模型拆分是资源安排，不是两项新实验，不使query预算翻倍。

每个cell保存真实condition、初始noise、stock step trace、raw/decoded、专家/mask、原gate和指标。R0/R2条件/hash不一致时先区分允许的机械dtype记录差异与真实输入变化；不能只凭seed相同宣布配对。

正常有限REJECT是合法诊断结果，继续其余cells。first-valid条目按完整模型/输入/noise/源代码身份核验复用，不能用sample ID或非空文件代替；真实模型非法输出不得重抽噪声掩盖。失败cell保留attempt，只有机械原因修复后按同定义恢复缺项。

Corelab环境和显存确实可用时可按既有政策运行P1；Hopper为必需环境/容量时只运行成熟入口。两模型的运行profile尽量相同；变化必须明示且不得挑设备成绩。无新增noise/seed/sampler-step实验臂。

P1完成不要求先证明唯一根因。只要没有实际label/decoder/ABI/输入身份冲突使训练设计无效，进入C/T，普通诊断结论不成为又一次PI等待门。

## 6. G3 — 实际完成C/T双臂训练和评估

以下全部沿PI0245，不作新调参：

| 配置 | C | T |
|---|---|---|
| Family | PI0245_C_NATIVE_M16_2K_V1 | PI0245_T_PHYSDELTA_M16_2K_V1 |
| 初始化 | 同一sealed R0-final10000 | 同左 |
| 优化状态 | 新AdamW/scheduler/RNG | 同左 |
| 数据/曝光 | 原16，同一2000项表，每窗125次 | 同左 |
| 超参数 | LR1e-5、WD1e-5、maxgrad1、cosine2000、warmup10、batch1/accum1/workers0 | 同左 |
| Loss | L_native | L_native+0.1 L_phys_delta |
| Full保存点 | 1000/2000 | 1000/2000 |

其他R0实际parameter/moment/autocast精度、冻结/可训练范围、augmentation/dropout、sampler、TRAIN统计与H40不变。相同数据顺序不等于自动保证完整RNG配对；保存实际source、tau/noise、条件/augmentation证据并报告差异。

物理损失严格继承已验证公式：同一次native forward的`a_hat=x_tau+(1-tau)*v_hat`，`E=D_xyz(a_hat)-D_xyz(a_star)`；`L_phys_delta=mean_valid_transitions ||((E[k+1]-E[k])/0.05s)/(1m/s)||²`，三XYZ平方求和，不额外除3。原逐时间尺度、anchor和clip在D_xyz内保持。不能退回normalized差分，也不能从平方action_loss猜测有符号残差。

已经完成的CPU decoder与数学parity复用。尚缺的真实训练图检查在第一正式update前向中完成：验证同一次noise/tau、正确pred残差、39有效transition、native/auxiliary/总loss、梯度有限及优化器成员。第一次optimizer更新纳入该臂2000预算；不能为“smoke”在正式initializer上先偷偷更新再算step0。无需新增训练GPU来重跑通用CPU测试。

C→T串行。即使C最终通过也按原双臂设计完成T及配对报告；“首次PASS关闭后续arm”对P/PT适用，不允许省略已经预注册的对照实验一半。C的普通有限成绩FAIL不阻止T。

actual trainer max_steps、save steps、schedule cursor和audit expected_end_step全部显式传值，未来恢复也绑定它们，不再依赖旧1000-step默认值。完整checkpoint是恢复点；只有模型文件、目录名或partial保存不能exact resume。

报告/PBS尾部上传失败但训练完整时，用只读finalizer补齐，不重新训练。恢复时使用本arm的optimizer/scheduler/RNG/进度与真实sample消费；不同arm之间只从指定R0权重warm start。

P3主screen：共同step0×16＋C1000/C2000/T1000/T2000×16=80逻辑cells。保存step0实际噪声和条件，此后真正复用这些张量，不以新seed重建代替；P1的CPU noise bank与主seed实际噪声分开标识。两臂final×原bridge4×P1四noise=32诊断cells照常完成。没有新增旧direct-flow复跑。

## 7. G4 — 条件成立自动继续P/PT，直至冠军或研究树耗尽

### 7.1 两个物理量定义保持

`L_phys_pos = mean_valid_points ||E/(0.05m)||²`，完整H40三XYZ求和；原统计、decoder及mask不改。

### 7.2 严格按现有PI0246树

1. C/T任一final2000主16通过，按原章程确定性规则选champion；保存另一臂和噪声复核结果，不再运行P/PT。
2. C/T均有效FAIL，且至少一臂XYZ median或p95仍失败：自动准入P；两臂XYZ都已经通过而只剩其他失败，则P/PT没有自动准入，停新假设门。
3. P：`PI0246_P_PHYSPOS_M16_2K_V1`，同一R0-final10000重新开始，fresh optimizer/scheduler，和C/T同一2000曝光/超参数；loss=`L_native+0.1 L_phys_pos`。full保存1000/2000，新增32主cells，共同step0复用。
4. P有效PASS：关闭PT并冻结champion。P有效FAIL：按原树执行PT，不把普通FAIL再次变成例行确认。
5. PT：`PI0246_PT_PHYSPOS_PHYSDELTA_M16_2K_V1`，仍从同一R0-final10000，而非P-final；loss=`L_native+0.1 L_phys_pos+0.1 L_phys_delta`；同2000预算/保存点/32主cells。
6. PT有效FAIL：`PI0246_GATE_PASS_TREE_EXHAUSTED — STOP_FOR_PI_NEW_HYPOTHESIS`。

P/PT同一次native forward共用a_hat/decoder，不重复采样或第二次forward；mask、p_ref/v_ref和系数固定。lambda=0、常量physical bias、oracle、clip/scale/finite回归在本地/Corelab先通过，未有科学准入不提前backward。

C/T同时PASS的选择沿原章程：较低p95 XYZ；相等到1e-6m时比较median per-trajectory峰值线速度，再median XYZ，再选C。完整保存原数值及比较过程，不按噪声诊断临时改排名。

每臂只判final2000，不能挑1000或最少REJECT的中间点。各臂要求原四项联合门以及完整证据。科学表中的普通FAIL只有在本树有下一分支时继续；工程FAIL只修机械部分，不消耗或重置科学arm。

既有预算上界：最多C/T/P/PT四臂各2000，总8000 canonical updates；P1 32＋C/T主80＋C/T噪声32＋可选P主32＋可选PT主32＋P或PT冠军噪声16，总最多224个逻辑query cells（同身份可复用）。这是整理原PI0245/0246上界，不新增预算。内部backbone/head calls、失败attempt与实际资源另记。

## 8. G5 — 交付一个可读取、可复核的模型，而不止PASS字符串

每臂结果由独立CPU校验器复算现有NPZ：固定ID集合/模型/condition/noise、dtype/shape/finite、gate枚举及issues、逐窗全40点XYZ与SO3 RMSE、linear percentile以及源hash。不得通过截尾、限速、输出平滑、改dt或删除困难样本取得PASS。

champion最终需保存：

- 完整checkpoint及HF/持久存储的fixed revision、inventory、逐文件内容验证和恢复说明；训练产物和部署推理文件的scope明确；
- `fit_gate_champion_manifest.json`：arm、initialization、loss/单位/系数、训练参数和dtype、processor/stats/decoder/frame/TCP/H40、runtime/code身份、主16实际条件/噪声、各项实测门与独立校验、原gate全部issues；
- 原16逐窗结果、训练曲线与更新/冻结证据、clipping和尾部误差、P1/noise结果及局限；
- 原固定noise复核：C/T沿既有32，P或PT冠军另用已批准16；不得以主seed通过隐藏其他noise失败。

原章程允许主16通过但噪声不稳时标记`NOISE_ROBUSTNESS_WEAK`，不修改主门也不称稳健。即使主PASS，后续scale仍由PI审核，不自动M64或sim。

若没有冠军：交付每个已执行arm的主表、相同source配对及已测机制，提出一项NOT_AUTHORIZED的新假设；不扫lambda、改LR、延长20K/50K、改sampler或解冻模型。

## 9. 持续推进的操作纪律

只维护一个轻量`campaign_state.json`，复用原机制；字段至少有：authority/code、current phase/arm/host、job/process owner、submitted/running/terminal、逻辑cells及actual attempts、checkpoint receipts、next action、精确blocker。每个milestone落盘并写Issue；不为每次小修创建新科学编号，不重新拉取全库。

提交前写入intent与launch digest，qsub返回后立即保存job ID。连接丢失且提交状态未知先恢复scheduler确认，禁止双投。日志/model/query/optimizer计数分开：Python前退出可为模型活动0，但scheduler占用不应写成0。PID或job完成不自动代表artifact有效；副本完整性与科学结果单独判定。

断线后恢复到最近已封存的step/cell；prepared不等于ran。仅有命令行、compile或queue状态不是任务结束。相同有效结果固定，工程恢复不因成绩替换它。

本地/独立worktree可以并行准备未来臂的CPU代码，但不能改变正在运行/排队作业封存的checkout、card或数据。原Git分支/commit参与binding时，文档读取与运行目录隔离，避免pull新文档本身使执行身份失效。

不新建定时monitor。PC当前执行会话可以按既有单实例机制查询真实owner和增量产物，避免每次扫描全库；等待和重试有退避且遵守站点规则。真正外部不可达时准确停BLOCKED，不假称科学完成或承诺GPT后台工作。

## 10. PC agent启动正文

执行`PI0246_ENGINEERING_TO_GATE_PASS_CONTINUE_GO`。先完整读取本固定协议、PI0246_GATE_PASS_CHAMPION_CAMPAIGN_V1和PI0245原协议；核对5807158828/5807161634、P0证据5800283247及最新后续状态。4d23488是审核起点，不默认等于当前live head。

目标是完整执行已冻结P1→C/T→条件P/PT→评估/归档/冠军交付。工程问题自主修复、回归、恢复，不设次数上限；不在DNS、路径、编译、排队或普通branch FAIL处重复请求已有授权。

先恢复既有SSH/DNS/合法网络路径，查真实scheduler owner；不猜IP、不关闭host-key校验、不绕过权限。失联时继续不依赖远端的CPU准备，但不在不能确认owner时重复提交。

保留629669失败证据。修正单模型wrapper仍同时要求两模型restore manifest的残留依赖；只要求当前MODEL_ROOT，完整P1汇总仍检查两模型。最终字节做真实preflight-only/缺文件/错hash/另一模型缺失等回归。计算节点只读compute-visible scratch镜像；原card用受测overlay，不裸改sealed路径。节点启动前复验所有真实依赖和输出容量，再直接原定计算，不再盲投GPU探测。

复用P0和已封存noise/R0模型，不重跑旧大审计。先R0的16个P1cells，校验/登录端归档；再R2同16，形成32完整配对。ARCHIVE_PENDING必须有实际接手与receipt，不是归档成功。无真实科学契约冲突即继续C/T，不等P1证明唯一原因。

C/T各2000、同R0-final起点和2000schedule、LR1e-5；保持物理尺度loss的原公式。先C后T，完成原80主screen和32noise复核。第一实际训练图/更新审计计入正式budget，不额外调试更新。

两臂均FAIL且至少一臂XYZ门失败，按原章程自动P，再按条件PT；各从原R0-final重新开始，不能接上一个失败arm。P= native+0.1 phys-pos；PT再加0.1 phys-delta。工程恢复不改loss/seed/LR/统计/精度/H40/gate。

每臂仅final2000判：零REJECT、median XYZ≤.020m、p95≤.035m、median SO3≤.050rad。C/T双臂设计必须完成；第一个符合原选择规则的冠军关闭后续P/PT。完全跑通保存/恢复/独立screen/固定revision归档，产出fit_gate_champion_manifest和全部噪声结果。

成功停M16_GATE_PASS_MODEL_FROZEN；树耗尽停PI_NEW_HYPOTHESIS。selected_checkpoint仍NONE，M64/FULL/CALIB/WBC/sim/DEV/F/DAgger/硬件/merge未开放。容量需要才H100/H200，本地/Corelab先成熟代码；HF/报告错误不重训、原唯一持久源不删。

Issue #11先canonical，PR #77只pointer，不重建monitor。不要仅回复已读、compile通过或等待GPU；交付实际训练/评估结果，或已经穷尽合法路径的精确外部阻塞证据。

## 官方命令参考（非项目实测）

- OpenSSH ssh：https://man.openbsd.org/ssh （-G查看解析后的配置，-J跳板连接的语义）。
- OpenSSH ssh_config：https://man.openbsd.org/ssh_config （HostName、ProxyJump、HostKeyAlias及主机校验）。
- Microsoft Resolve-DnsName：https://learn.microsoft.com/en-us/powershell/module/dnsclient/resolve-dnsname （DNS查询；不代表TCP或SSH认证成功）。
