# WM02 AUTO1：自主跑通目标 WBC 采集 → World Model 训练入口

日期：2026-11-04，Asia/Singapore。基线为未合并 PR #352 / `76bb8115c3b588b70fd55bdde13f10926c60bce7`。这是新授权范围的执行设计，不是目标动态验收或实验结果。本文件、Codex交接及配置一起使用；历史PR、原始数据、STOP和CLOSED lease不改。

## 0. 本轮目标与事实边界

用户本轮直接要求：**“给codex更多授权，制定一个长程自主推进计划，跑通WBC采集到WM的训练入口”。** 本计划把该要求限定并落实为：项目自有用户空间内，完整Go2＋ARX5 MuJoCo＋ROS2/OCS2冻结执行器的仿真调通、证据驱动工程修复、有限真实顺序采集、目标域Replay和WM训练/恢复/评价。不是实机授权，不是无条件放行旧失败版本。

唯一主交付：**原生目标初始化/保持 → 合法EEF参考实际消费 → 真实全身与物体转移 → 目标域数值Replay → WM至少200次真实更新 → 完整checkpoint重载/续训 → 再采独立批次并评价。** 每个箭头必须有实际输入输出，不以源域S数据、合成fixture或计划文件代替。

基线事实：PR352的M3完成200主更新＋1恢复探针；H1/H8/H16动作差门槛通过，但H1绝对物体误差1.947mm差于M1的1.151mm，建立/释放H16抓持Brier约0.6921/0.7816，整体FAIL。H1物体MSE中99.9844%是候选共同误差。目标候选已构建/加载检查，但冷启动、保持、实际STOP均NOT_RUN；目标积分和Actor更新为0。来源：`reports/WM02/temporal-response/EXECUTION_REPORT.md`及PR352机器回执。

关键判断：配对差监督不约束候选共有偏差；当前M3不宜驱动Actor，但**源域模型没有胜过M1/规则不是目标采集或训练入口的前置门**。首次目标WM训练成功只证明入口可用，不证明模型可用于规划。源域的OLS、M1和M3仍保留为诊断，不把它们当目标动力学。

## 1. 授权范围：一次注册，分级执行，不逐小操作再问GO

### 1.1 已授权的项目工作

执行Codex可凭本次直接请求，经现有 `EXISTING_LOCAL_PC` / 当前项目实际owner和原profile/registry/Journal/lease机制建立**新的** `WM02_AUTO1_SIM_ONLY` 范围，完成以下工作，无需为每次工程修复、阶段通过或下一有限批次另问用户：

- 创建独立worktree、提交代码/测试/文档，更新一个连续工作PR；不强推、不合并main、不切换正在运行的源码。
- 复用或构建R1同步、停止锁存、时序记录器、wrapper/receiver/数据接口的候选副本；修复有证据的初始化顺序、历史填充、pos/vel命令版本配对、时钟/坐标/序列化和保存恢复缺陷。
- 新候选的有界原生初始化和保持；通过对应检查后自动进入小EEF动作、夹爪、受支撑接触与站立操作数据采集。
- 完整目标顺序采集、失败与拒绝记录、目标schema/Replay实现；不要求先有任意中途完整restore。
- 在已有本地CPU与获准Core GPU资源上运行真实目标WM训练、恢复、预测诊断；数据和模型按已有授权路径分批传输。GPU1优先，已有GPU0共享许可与现场余量允许时可顺序回退，按UUID而不是卡号猜测身份。
- 在有限端点内自动继续可行动的下一阶段；物理/优化成功后不以“计划完成”提前停止。

### 1.2 本次授权实际改变什么

V7和旧交接中的“没有新的目标动态范围”是过去的事实。**本次用户请求是新仿真范围的授权来源；Codex应实际尝试通过现有owner注册它，而不是再次只引用旧CLOSED lease并报告缺少用户GO。** 保留旧S1/T2隔离和旧失败版本；只对新范围内的新候选进行分级commissioning。旧安全检查、STOP、限位、接触白名单和原生ACK仍全部执行。

必须分开三个状态：`USER_AUTHORIZED`（当前请求）→ `REGISTERED`（现场owner/权限/资源实际绑定）→ `NATIVE_STAGE_PASSED`（实测结果）。本文件只提供前者和执行计划，不伪造后两者。若真正的站点管理员/其他资产owner权限不属于用户、凭据不足或原owner机制拒绝，保留实际拒绝原因，停止该动态分支并报告**具体缺失的外部权限**；不得写一个permission=true绕过。

### 1.3 始终不在授权范围内

实机/电机通信；WBC或腿策略重训；控制增益、成本、solver、物理dt、机械限位或碰撞规则调宽；更换控制器冒充原WBC；强杀其他用户任务；系统/网络/NFS/SSH/权限/凭据修改；删除数据或checkpoint；公开私有资产；无界后台循环；未经本地入口验收就提交A40/H200长队列。固定底座S成功不算目标成功。当前不授权正式Actor梯度训练、主动行走或移动抓取。

工程修复可以改变运行时包装实现，但须明确 `executor_version`；不得用“权重没变”假装执行动力学没有变化。命令配对修复只在已证明版本错配风险及对应测试下实施；不能把它预先宣称为唯一腿部根因。

## 2. 长程执行与停止规则

默认一次自主campaign最多72小时，**从实际执行者注册开始计时，不从文档发布时间起算，也不是完成时长承诺**。到期停止新派发并安全结算在途工作。每个worker沿用现有有限端点/RuntimeMaxSec/STOP，最长2小时；初始化按仿真tick和墙钟双重限额。使用现有owner持续接续或重新接管，不另建平台/daemon。Chat发布计划不意味着Codex或服务器已开始工作。

现场只允许一个目标物理owner/单目标实例；先不并行目标环境。CPU默认4个允许的核，模型单卡；GPU allocator默认4GiB，测量后可在8GiB至最多13GiB内调整，须留设备实际余量并尊重已有站点规则，不抢占。仿真和训练可在episode边界串行驻留；不因共享GPU暂忙而停止可进行的CPU工作。AMD只作已有存储/中继。

允许最多6个有实质修复的初始化候选版本。每候选按0.1/2/5秒各两次独立冷启动，最多6次常规验收；任一次原生安全失败立即中止该次运行及该版本向后升级。下一候选必须有新增证据、明确修复、针对性回归，不能只改run名盲重跑。相同故障无新证据重复时改诊断，不循环启动。达到candidate上限或物理安全无法解释时暂停目标动态，继续可做的离线诊断并提交最窄阻断；不把普通一次构建失败永久封禁全研究。

健康作业不因新模型/源码出现而中断；只有本campaign自有已确认任务可安全停止。未知执行状态先查原Journal/进程/产物，不重放不确定写入。失败优化器调用、构造/影子积分和提前终止全部累计；预算是本campaign操作范围，不改历史额度或现役期限。

## 3. 工作包与自动推进条件

### G0 — 现场接管与范围注册（先实际执行）

读取根AGENTS及本文件；固定基线，保留本地dirty工作与旧封存树。仅核对相关当前owner、私有资产、二进制/动态库、已关闭队列和资源，不能全盘扫描或反复重核所有历史。将本次用户请求原文/日期与本协议SHA绑定新scope，用现有host-aware registry/lease真实注册。

检查当前目标入口：基线 `scripts/wm02/qualify.py --domain T` **无条件返回78**，并非有授权就能执行；`world_model/runtime.py`也只支持旧S/合成有限profile和精确200端点。Codex获授权在新版本实现受控target commissioning/collection profile与通用有限target-neural profile，复用原owner/Journal，不删除必要验证。旧入口可保持历史语义，新增薄入口更清楚。不得仅更改spec字符串就称目标adapter完成。

交付 `AUTO1_RESOLVED.json`：scope注册状态、owner/UID/host、源SHA、候选hash、WBC/模型/物理参数版本、现场资源、first runnable command；所有尚未执行状态为NOT_RUN。无需等上游PR全部合并，用独立固定worktree执行。

### G1 — 有界初始化、修复与保持（最高优先级）

优先复用PR351已构建并经PR352复核的候选；只有变化时重建。选定**一条**外层：R1 task_actuation → observer → 冻结MuJoCo，绑定同一个trace，不能叠加两套outer。部署到新运行目录，验证实际ldd/dlsym字节与冻结输入，明确过程/ROS domain隔离且不触达实机。

新scope下先两次独立0.1秒（50×2ms）冷启动。记录观测/历史、推理开始结束与发布版本、PD实际消费pos/vel版本、q/dq/kp/kd/tau、ctrl、模式、sim和monotonic时间、每个tick与STOP。目标是确定正常推理时PD是否持续更新、启动历史是否就绪、pos/vel是否成对；首个坏输入/命令/状态才是根因证据。trace容量/丢包和记录扰动单独检验，不用0事件doctor冒充实际记录。

两次短启动均通过，自动各做两次2秒、两次5秒独立保持。原joint/contact/fullbody安全、参考为空/保持和合法夹爪初态不变；无Actor/GR00T、抓放、主动底座指令。通过需要所有预定启动结果可见、不挑成功seed、无原生硬安全违规、模式/控制时序可解释。

受控停止必须在稳定短运行中验证：使用现有合法STOP/停止入口并确认锁存后0额外物理积分；不为触发停止而故意推关节越界。若operator STOP没有覆盖安全锁存同一分支，应明确区分两项证据并在模拟测试hook/原harness中测试安全分支，不能把未发生的实际安全停止记PASS。任何安全停均冻结现场，原scope结算后用新修复身份继续。

可自主修复：既有R1同步、启动ready握手、正确原历史初始化、命令pos/vel原子配对、只读记录与wrapper返回语义；不改变控制数学/参数。需要改控制律才能站稳时超出本轮授权，报告可复现低层问题，而不是让WM补偿。

### G2 — 目标参考消费与顺序转移adapter

保持通过后，在同一scope自动完成至少8个有限条件：2个保持、2个小自由空间EEF运动、2个夹爪/受支撑抓持条件、2个受支撑释放与撤离条件。命令幅度不超过既有原生范围；先极小动作，再按实测资格增加。任何原生REJECT/STOP都是结果，不能放宽门或换计数口径。

复用 `task_native_worker`、`task_state`、原receiver和V2 compiler，不搬OSC替代WBC。真实核对proposed→accepted→installed ACK→实际消费/reference version→achieved；每50ms应有25个原2ms有效tick，末尾不满50ms另存partial，不能补齐。考虑H40预览、41点wire、绝对时间、world/rot6d约定、夹爪机械范围和controller消费时钟。仅有发布日志不算消费证明。

新增 `TargetSequentialAdapter`（名字为拟实现接口）：reset/start/step/observe/close复用原执行链；从正常冷初始化连续采集。**完整任意中途MPC/WBC恢复不是这里的门槛**。缺hidden变量时保留可观测历史/命令历史与缺失mask，模型称history-conditioned/POMDP近似；不伪造Markov状态或零填目标BASE/腿。顺序采集不声称严格同状态反事实。

初期可交付FREE_SPACE_TARGET入口，但如果接触条件未通过，必须单列 `CONTACT_COLLECTION_NOT_QUALIFIED`，不能声称操作WM已可训练。保留初始/失败/终止和所有预定条件分母。

### G3 — 目标数据首批与分批扩量

先8次pipeline尝试，每次最多8秒（上限1280个50ms转移，实际按停止计数），使用已通过G2的有界策略/几何参考，不依赖未合格M3或Actor。包含保持、不同方向EEF、小夹爪和已合格接触；失败尝试保留，不要求完整任务成功才学习动力学。

首批pipeline数据与后续科学面板分开。后续先预登记32个实际物理父布局/初态，24 TRAIN、8 DEV；同父episode、恢复/重放与相邻窗口不得跨分区。模型归一化只用TRAIN。最多累计32次32秒顺序尝试（不含首批8次）；需要更多事件覆盖或新策略数据时可以自动扩至累计128次，不是看到好结果才换布局。每次扩量前先从TRAIN覆盖统计说明新增价值；安全失败停止该版本/条件，修复后另版本且不覆盖旧分母。

拒绝动作没有实际转移；未执行H40后缀不作标签；模型预测状态不混入真实Replay。构造前缀、episode、物理tick、50ms完整转移、partial、安全事件分别计数。建立/释放事件应有明确native标签；缺少support字段不能使用illegal_contact代替。真实reward可后补版本化计算，但缺Reward不阻断WM入口，缺真实接触记录必须显式报告。

### G4 — 目标Replay/WM训练入口（核心验收）

确定 `WM02_TARGET_NATIVE_STATE_V1`：从实际模型/packet/命名映射导出维度、单位、坐标、quat/rot、时间/版本及可用性；不要照抄S50或合成93维。原始状态完整保存，训练tensor可选择充分可观测子集+历史。记录executor fingerprint，含WBC/腿策略、模型、wrapper、queue/reset/timing、场景参数；不同执行器版本不无标记混训。旧失败版本原件只作诊断，不自动进入最终版本TRAIN。

复用非pickle数值格式、split/parent检查、checkpoint工具和existing owner。输出序列长度显式为burn8+predict16，需要连续24转移/25状态；较短段可以单步训练，不拼跨reset的假序列。目标历史不足时可以预声明burn16/32对照，不为了凑窗口延长失败运行。

新目标训练从可靠绝对状态监督起步：现有relative-reference WM、独立匹配的小型线性/MLP残差对照即可。首先不使用M3的效应差项作为主loss，更不继承其checkpoint冒称目标预训练。目标已采数据不具备同状态真实配对时，不启用pair loss。

loss至少报告真实状态/增量、每变量物理误差、单步及FREE H1/H8/H16、事件与终止；连续与事件保持独立统计。若共享干路使事件退化，可在新配方中分离事件网络/优化，不能只冻结head却忽略干路仍变。归一化/权重只基于TRAIN预定；不以某个DEV指标通过掩盖其他阶段失败。M3的共同偏差可用绝对监督或同锚点平均残差约束，但新目标无配对时仅用可得真实标签。

真实目标loader首个batch检查后，1步/重载，然后固定200次主更新、1次独立续训探针。必须保存model/optimizer/normalizer/RNG/cursor/schema/executor/data/code身份；实际resume和不中断参照在已声明同环境容差内一致。probe另计，不重跑200冒充恢复。不把loss下降当控制能力。

通过pipeline后，Codex可自主选择最多3个明确配方、每次先200端点；TRAIN/DEV证据支持继续时可到1000或5000端点，保存预定checkpoint，不逐次请求“扩到5000许可”。5000不是默认；200也不是永久硬编码批准上限。每配方单seed工程调通，后续正式多seed研究另报告。模型失败允许调整归一化/表示/容量/损失和数据覆盖，但一次消融明确主要改动，真实记录负结果。

### G5 — 新批次、加载与二次训练

加载首个目标WM checkpoint，另运行4次最多8秒的新目标采集；使用同一已验证几何/参考生成器，不要求WM选动作，更不要求Actor。评价旧模型对新批次H1/H8/H16与事件结果，然后将符合预定TRAIN身份的新数据入新Replay版本，进行100次有界后继更新并再次恢复。原DEV不入TRAIN。

由此证明 **collect A → train WM → reload → collect B → evaluate/update** 的真实链路。此时Actor仍为0。可输出一段实际目标回放/视频用于PC查看，viewer只读取实际状态，不为画面重新积分或捏造完整任务；渲染不得阻塞PD。

### G6 — 完成定义与后继研究界限

`TARGET_WM_ENTRY_VERIFIED`必须同时满足：G1真实冷启动/保持、G2真实参考消费和安全记录、G3非零目标有效转移、G4目标200更新及完整恢复、G5第二批及后继加载/更新。报告模型预测好坏，数据可分FREE_SPACE与CONTACT资格。`TARGET_CONTACT_WM_ENTRY_VERIFIED`还要求实际通过接触采集及接触事件数据，不能由空中EEF运动推断。

`MODEL_USEFUL_FOR_CONTROL`是另一项科学结论：新目标DEV的绝对状态/时间响应/动作效应/事件和实际决策增量。它不是上述工程交付的必要条件。Actor正式训练、复杂开门/放书、主动行走/移动抓取、视觉学生和跨本体结论均为后继；本campaign不为追这些目标扩大权限。

## 4. 模块复用、拟实现CLI与测试

已有实现：`review_temporal_round.py`（只读复核）、`build_init_candidate`、`inspect_init_trace`、原task worker/receiver/记录器、V2 compiler、源Replay/WM/checkpoint、原runtime owner。旧T qualify是阻断说明，不可伪装已存在可运行target CLI。

Codex应按需新增薄接口，以下名称是**工作清单、不是已经实现的命令**：

- `scripts.wm02.commission_target`：当前scope＋候选＋phase，受控原生启动/保持。
- `scripts.wm02.collect_target`：合格executor、预登记episode与原生顺序记录。
- `scripts.wm02.build_target_replay`：只读真实目标原件→schema/分区数值集。
- `scripts.wm02.train_target_wm`：有限目标训练、重载/续训、后继更新。
- `scripts.wm02.evaluate_target_wm`：冻结模型与实际转移预测评价，不隐式启动物理。

所有--help/import不得启动模型/仿真；许可拒绝必须在构造前；关键边界不用可被-O移除的assert；测试普通与-O均跑。覆盖scope错host/失效/终态、并发owner、未知Job拒绝、pos/vel配对、stop锁存、动作/时钟/夹爪、跨episode窗口拒绝、TRAIN/DEV泄漏、native域与维度、事件命名、checkpoint恢复。合成测试只是组件验证，不冒充原生PASS。

禁止新增重复调度器、预算allocator、审批平台或每函数回执。私有现场一个最小 `AUTO1_STATE.json` 索引阶段、实际run、累计量和下一命令，引用既有Journal而非另记一套效果账本。源码/数据验证在版本变更边界做，非每50ms全历史扫描。

## 5. 自主决策与汇报

阶段PASS自动推进；普通工程FAIL先复核效果、定位、修复、测试，再新身份重试。GPU容量不足转CPU工作或等待现有容量机制，不能强杀其他进程。真实安全失败冻结当前现场并停止该候选升级；有实质工程修复后在本scope内可再commission，不每次问GO。不能自动修复的真实外部权限、需要修改低层控制数学、硬件/系统变更、连续6候选仍无进展、72小时到期或用户STOP才升级给用户。

每个里程碑更新一份简短状态：实际目标积分/完整转移/父场景/模型主更新与probe/完整checkpoint/首个失败/下一命令。运行跨交互连接时，仅在已实际部署并验证存活的原owner/worker机制下报告“正在运行”；计划不承诺未来交付。一个阶段完成不是自动结束研究，达到G6或硬停止条件才停止本campaign。

首次Codex返回应至少有：当前用户授权已如何登记、候选实际身份、首个真实初始化尝试或具体外部拒绝、已实现目标入口的真实命令。不得仅复核历史文件后再次返回“无init-only授权”。最终返回可复现目标采集及训练命令、完整SHA、checkpoint位置、原生安全/接触范围、真数据统计、负结果及成本。

## 6. 参考源与本次发布状态

- 根 `AGENTS.md` / V7：自主修复与既有owner规则；本AUTO1对新目标仿真范围作窄补充。
- `reports/WM02/temporal-response/EXECUTION_REPORT.md`：M3结果及共同偏差/事件退化。
- `reports/WM02/response-init-candidate/INITIALIZATION_CANDIDATE.json`：已有目标候选、加载链和队列观测。
- `reports/WM02/decision-initialization/TARGET_STARTUP_AUDIT.json`：历史初始化失败，不是新动态结果。
- `scripts/wm02/qualify.py` / `world_model/runtime.py`：当前T入口缺口与旧固定profile；不要把描述当运行实现。

本发布只写授权解释、执行计划和入口提示；没有改WBC/运行时实现、没有注册实际主机lease、没有新目标积分或GPU更新。当前runtime资格与目标状态仍以实际Codex执行为准。
