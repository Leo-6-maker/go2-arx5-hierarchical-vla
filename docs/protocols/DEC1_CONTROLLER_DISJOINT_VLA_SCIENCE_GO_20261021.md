# DEC1：Controller-Disjoint VLA / Frozen-WBC 解耦证据与新科学授权

日期：2026-10-21，Asia/Singapore。Decision/Goal ID：DEC1_CONTROLLER_DISJOINT_VLA_V1。
状态：USER_AUTHORIZED_NEW_SCIENCE_PLAN_PENDING_NATIVE_ADOPTION。
授权依据：本轮用户原话「你思考确定一下我们的创新、贡献点，我们focus在VLA和WBC的解耦，确定后给出新的授权计划，尽快拿到科学结论」。这是对科研重点、实验变量和依赖关系的明确新决定；不是再次把所有新实验标成NOT_AUTHORIZED。发布不等于现场已采用或运行。
审核基线：2d6c4d68380d22a284412167aa465609903fdf35。原PV1 authority：9eff29a44a193dd4d3620dc0bfe5764a402eff42。现有workspace、单owner、站点准入与原始安全规则继续适用。

## 1. 确定论文主线：不是首次分层，而是可检验的控制器隔离迁移

工作标题：Controller-Disjoint VLA: Execution-Grounded Task-Space Trajectories for Frozen Whole-Body Control。
核心问题：不使用目标WBC产生的任务示范来训练高层，不联合训练或调参目标WBC，同一份语言条件EEF策略能否通过明确的任务空间接口，在源执行器和独立冻结的目标WBC间复用？实际执行进度反馈能否降低这种迁移中的失效？

贡献候选（实验前均为待验证，不在论文中写成已证实）：
1. 数据/学习解耦：高层学习来自源域的视觉-语言-实际状态到提交EEF参考，不学习目标关节/力矩、目标WBC latent或目标任务轨迹；目标控制器独立交付冻结。准确称target-controller-disjoint training，不称所有数据与所有执行器无关。
2. 接口/执行解耦：将任务空间几何、执行进度、夹爪事件和实际反馈明确对齐；比较原时钟锚定与实测进度锚定，而不是假设期望EEF已经实现。新增方法只在高层与WBC之间，不修改WBC内部控制。
3. 证据与边界：同一高层权重跨两个冻结执行器的全交叉对照，分开报告语义任务正确性、接口合法性、跟踪响应与物体任务结果；用实际交换执行器检验可替换性，不以模块图或RMS代替任务结果。

CR1的范围修复是可靠接口和数据表示的支撑证据/消融素材，不能把min-max扩范围单独包装成主要算法创新。当前两个固定任务只支持双任务语言条件控制，不自动支持开放词汇、任意任务、灵巧力控制、任意本体或实时部署。

## 2. 最近相关工作与不能主张的“创新”

- HAMSTER (2025)：高层VLM输出2D EEF路径、低层3D策略执行，并利用off-domain数据。https://arxiv.org/abs/2502.05485
- 3D HAMSTER (2026)：已经直接研究高层metric 3D轨迹指导低层控制。https://arxiv.org/abs/2606.31329
- UMI (RSS 2024)：相对轨迹、推理延迟匹配与跨机器人部署已是明确贡献。https://roboticsproceedings.org/rss20/p045.html
- LeVERB (2025)：视觉语言latent与RL whole-body control分层已存在。https://arxiv.org/abs/2506.13751
- H-VLA (2026)：关键动作与运动生成、统一相机坐标动作空间已存在。https://arxiv.org/abs/2609.22895
- Opt2VLA (2026)：单多任务VLA输出几何/接触力、任务特定RL WBC执行。https://arxiv.org/abs/2609.23968

因此不声称“首次VLA+WBC”“首次EEF动作”“首次层次化”“首次机器人无关表示/延迟处理”。预期差异是：在目标任务示范与目标控制器训练都不参与高层学习的约束下，研究冻结控制器可替换性的失效及反馈接口。上述组合的学术新颖性与实证价值仍需结果和持续文献核对；不能保证录用或优于已有全部方法。

## 3. 已有证据和当前阻断（不是实时服务器查询）

- CR1 e710f150：独立TRAIN诊断7条旧FAIL恢复、6个几何lineage；原139条codec无回归。R1统计SHA256=50d9672397fd0b5353da2011b9f9f568dc9c65c26fb271458f87d60b2d497fa8。无模型/物理/训练改善证明。
- PV1最后B0=14/48（9成功、5有效失败），A/B optimizer=0。原共享CORR@2048权重已经存在，可以立即研究接口/执行器迁移，不必等新模型训练或通过80%发布门。
- 2d6c4d68新收据：child12 / 679029.hopper-m-02为H/Hold_Types=s、execution hook rejected、exit -16，而非普通Q。没有PBS_READY、START_NEURAL或新模型/任务效果；约20–21秒prelaunch分配仍需记录。原H200 spent20473秒、reserved28860秒、PBS11/24尚是快照。
- 同收据显示原local owner因只读SSH临时DNS失败退出，与站点hook失败是否相关未知。系统hold需要站点管理员处理，不能把客户端重试当作释放系统hold的权限。
- CR2已独立批准并采用；12组零积分静态资格完成，首四条采集开始，尚无终局数据READY。CR2执行其原48h/32Core核时上限，不自动扩采。
- src/go2arx5_vla/adapters/groot_to_wbc.py当前是离线转换器，使用SE3弧长preview，但锚定controller_timestamp、gripper按时间取值；不是已验证的目标任务闭环。已有预览/相对表示必须复用并正确归属，不假称新发明。

## 4. 本轮新授权：明确替代旧依赖，保留旧结论

### 4.1 现在批准的改变

A. 移除“B0全48是任何新源域优化器更新的前提”。已封存的PV1 A_CONTROL/B_COVERAGE配方可以在独立训练数据资格、全状态恢复、站点预算和安全满足后先训练；它们不读取部分B0表现调配方。B0全48仍是任何使用B0相对门槛的正式选模、PV1_VALIDATED/BETA和原LOCKBOX80的前提。旧14格和原协议历史不重写。
B. 允许DEC1独立仿真研究：将已有CORR@2048/R0在源执行器及WBC团队交付的一个冻结目标执行器上执行T1/T2；允许本文两个版本的高层参考适配器。所有变化只进入新DEC1 scope/冷源码，不能偷偷热改旧B0。
C. 允许为上述新scope建立最小任务overlay、真实相机/夹爪/状态/记录连接和CPU测试；仅复用已有合格物理资产。不得改原机器人质量/摩擦/惯量、夹爪PD、安全阈值，或私自修理WBC。
D. DEC1结果独立报告，不需要先达到旧PV1的18/24或最终80%门才允许测量。它是科学实验，不是降低PV1发布等级。
E. 本轮同语义工程修复、故障证据、CI、冷态续接、预算内阶段转换无需再次请求每一步GO。真正的站点权限、未知副作用、物理安全和不可支付资源仍阻断对应lane。

### 4.2 未获准的改变

不改WBC代码/权重/增益/cost/solver/腿策略，不引入force head、task-specific关节脚本或目标WBC示范训练高层；不训练world model、增加DAgger或扩CR2规模；不热换R0/R1，不改原任务成功谓词/32s截止/安全门，不将EEF10补零称为学得BASE意图。不运行真实机器人。

本次授权只改变训练前置关系和新增有限解耦测量，不把所有旧门删除。增加版本化admission flag及测试：旧authority仍拒绝未完成B0的新更新；新DEC1 authority允许独立source训练，但没有完整B0时仍拒绝旧baseline-relative selection与LOCKBOX。

## 5. 方法规格：一份VLA、两个接口模式、两个冻结执行器

高层继续使用原RGB/语言/实际EEF与夹爪输入，生成H40 EEF10，原2.5Hz重规划/20Hz参考/Prefix8等现役频率不变。第一轮固定权重CORR@2048，manifest=94ad4d0eaaa5bf7df0c55a7b0c9c5f9242aa9798b1只是错误示例，禁止使用；正确manifest必须为下述完整校验值。
正确CORR manifest SHA256：94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38。
原R0 stats：fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7。
原新数据银行：60f7c35a05766fe23974eba3a72f53ca4db5545a3768e6c1cfea164ab4e58294。

### 5.1 两种执行器

E_SOURCE：当前已注册、固定底座ARX5任务执行器。
E_TARGET：WBC团队独立交付、版本/权重/配置冻结的Go2+ARX5真实动力学仿真执行器。第一轮只研究已声明共同可达的近场抓放和推送，不要求学得导航/BASE；free-root平衡由原WBC承担。不能换理想跟踪器、焊死base或覆盖关节状态冒充WBC。若团队尚未交付可执行目标，标TARGET_EXECUTOR_NOT_DELIVERED，source训练与接口CPU工作不停止。

两者不是必然仅控制器不同，视觉、身体、可达性与动力学也可能不同；结果应叫executor-domain transfer。只有控制其他因素后，才可归因为单纯WBC差异。纯控制器更换实验可由WBC团队以后交付，但不能VLA团队自己改gain造第二WBC。

### 5.2 I_TIME对照

使用现役语义正确的时钟锚定参考接口：相同输出坐标、相同观测/重规划频率，几何preview按原方式转换，gripper按原时间序列。
所有已确证的坐标/身份软件bug必须在I_TIME与I_PROGRESS两臂共同修复。尤其当前_prepare_chunk的expected_source_frame==chunk.source_frame且与wbc_frame不同分支，应检查是否真的应用了frame_transform；不要给基线故意保留漏变换。不能给基线错误frame/旧观测使新方法虚假获益。

### 5.3 I_PROGRESS实验：Execution-Grounded Trajectory Interface

只改变锚定和阶段推进依据：以实际测得TCP姿态/夹爪状态，在当前已接纳chunk的局部、单调有界SE3路径上估计执行进度，按原空间preview offsets取参考。参数只能用source TRAIN轨迹/无物体接口fixture确定，提交前固定；不按target成功率校准，不读取物体真值/接触oracle/任务ID来决定动作。

概念：s_hat=argmin over a bounded forward neighborhood of d_SE3(T_actual,gamma(s))；高层新观测仍参与每次重规划，WBC仅跟踪已发布preview。不要把desired pose当actual pose。相邻chunk的进度要重新对齐，不能把上一chunk的归一化s直接当下一条曲线的进度。

几何/夹爪事件的约束必须一并满足：只能保留原预测的开合顺序和幅值，不增加task-specific抓/推脚本；gripper转换在同一几何进度上发生。零位移、纯旋转、重复waypoint、停留和原地开合要有明确局部时间/索引规则，不能因最近点投影永远停在首点或跳过抓持。不得根据隐藏Teacher phase/物体到区真值触发释放。

冻结实际实现前需通过：身份映射、刚体frame等变、反馈=理想进度时两臂一致、局部不倒退/不过度跳跃、静止/纯旋转/夹爪平台段、chunk切换/reset、stale/NaN/reject安全停止。具体局部窗口/权重/停留规则形成一个固定配置，不在确认集上搜索。

此方法是待验证的高层接口干预，不是已经成功的新模块，更不是WBC控制律改进。若现有WBC已经内部推进相同路径，必须核对接口避免重复相位估计；以实际ABI为准，必要时报告方法在该控制器下等价/无额外收益。

### 5.4 安全和可比性

两臂都先检查原始完整H40参考合法性；不能通过重新计时将原来非法的H40包装成合法绕过旧速度门。实际映射/重参数化后的完整参考和每个将执行的prefix仍按同一原数值门及对应执行器的冻结native安全检查验证。当前S1若只支持source模型，不能直接宣称它保护target；target必须有WBC团队提供的等价安全/停止链和对应仿真预测，算法/阈值不放宽。
两臂相同推理精度、噪声张量、推理次数上限、任务物理时间32s、20Hz观测、持久保持谓词和总预算。不延长到成功、不挑seed、不best-of-N。通信等待wall-time与模拟时间分开，不能声称实时控制。WBC不接受语言或T1/T2 ID切换到两个任务脚本；夹爪仍由高层输出控制。

## 6. 最小实验：先用已有模型拿到结论，不等待新训练

### 6.1 预实验与数据隔离

在source TRAIN/专用commissioning几何上最多8次真实任务/接口资格：2个几何×2任务×2执行器。target的这部分仅为设备就绪与接口资格，不训练高层、调target WBC或按结果选容易确认场景；如有target特定适配/参数选择必须披露且降为calibrated transfer，不能声称零目标适应。所有物理启动计数。

确认集在读取新结果前封存8个独立新几何组：2个既有外观×2方向×2距离。准确slot/32个有限静态候选池/排重规则，复用CR2已修正合法几何域NEAR0.15–0.18、FAR0.19–0.22。需与旧TRAIN/C1、CR1/CR2/COV1、PV1 DEV/CONFIRM/LOCKBOX和所有同几何兄弟排重。只能按接口共同工作域、物理静态安全/可达性/视野/评分输入合同选，不依Teacher/task成功筛选。池耗尽说明任务域不可行，不改分母或无界换seed。

### 6.2 固定64格核心矩阵

8几何 × T1/T2 × E_SOURCE/E_TARGET × I_TIME/I_PROGRESS × 1预定噪声bank = 64格。
每个执行器/接口有16条任务，逐任务8条；一个几何内的两任务及所有对照成组相关，不能把64条叫64独立场景。固定一个新的噪声bank并在四单元对照复用相同事前张量，噪声随着已发生决策按索引取；不能为失败轨迹追加样本。实际初始物体/目标在统一task frame匹配、真实相机/robot状态各自渲染，不伪造两执行器后续观测完全一致。

首轮矩阵只用已存在CORR@2048/R0，不选中间checkpoint，不等待新数据/BC loss/B0全48或目标Teacher高成功率。目标接口未ready时不能将source重复四次凑64；产出明确缺失矩阵，训练lane可继续。

运行顺序按预注册几何block交错，避免先跑完一个方法后基于失败调另一方法。source固定训练数据独立于target确认；高层和接口配置的hash在打开确认集前封存，target闭环结果不回流训练。

### 6.3 主要指标与结论规则

主要结果是完整物体任务成功，分别报告两个任务/两个执行器/两个接口；安全/参考拒绝、释放/无抬升/撤离/保持等原谓词作为机制分解。
Delta_target = SR(E_TARGET,I_PROGRESS)-SR(E_TARGET,I_TIME)。
Delta_source = SR(E_SOURCE,I_PROGRESS)-SR(E_SOURCE,I_TIME)。
Interaction = Delta_target-Delta_source：只作为任务/执行域上的交互证据，不冒充纯动力学因果。
同时报告源到目标绝对下降、目标示范训练数=0的来源审计、目标WBC/高层hash不变、实际跟踪残差与进度滞后；源成功率低时不以很小迁移差宣称鲁棒。

按8个几何block做成组重采样区间/配对结果，单个噪声bank不支持随机种子鲁棒性主张。若主要区间跨0，称PILOT_SIGNAL_OR_INCONCLUSIVE，不把点估计包装成显著改善。无足够source能力/target接口不可行/缺数据分别为SOURCE_POLICY_LIMITED、TARGET_EXECUTOR_BLOCKED、INCOMPLETE，不“证明解耦无效”。

初步继续门（不是发表/部署门）：I_PROGRESS在两任务均有target完整成功、无native安全故障、无测试泄漏；若Delta_target为正且两任务没有明显相反趋势，则值得安排后续更大新确认。完整64格即使负结果也必须交付，不依结果裁减样本。旧PV1 BETA/VALIDATED不由此获得。

### 6.4 后续消融只选一项且受总预算约束

核心64完成之后，若剩余预算完整可支付，最多增加32格高层反馈消融：同8几何×2任务×2接口，仅target，使用预封存的额外100ms执行接口延迟（不是改变WBC参数/实际物理积分频率）。必须在最初plan中预登记并通过独立安全/时序fixture；未能保持原安全语义则不运行。它是已公开核心几何上的预注册stress，不是新的独立测试集。不同时增加新语言/物体/WorldModel。

## 7. 源域训练与第一版policy：可以先做，但不等它才测解耦

新authority明确允许原A_CONTROL/B_COVERAGE先于B0全48更新。两臂仍同CORR父、新独立optimizer、原R0、原139新银行/原C1配比、4096更新、1024 TRAIN-only探针、原BF16/TF32区分、有效batch32。没有新统计量/动作head/样本配比；真正的更新/消费/完整checkpoint与精确resume逐项落账。可在同一合法allocation内按可支付时间完成多个预登记阶段，资格不因每个checkpoint重复；不得超单job8h。

调度的事前规则：两执行器与新接口已ready且有合法GPU时，优先固定CORR的64格核心；target未交付而训练全部资格具备时，优先产生两份新4096端点，不因B0不完整空转。不能根据target成绩决定先训练哪臂或重新挑配方。

这些源域新端点是研究资产，不自动替换本次64格的预选CORR；要用新赢家验证迁移，须在下一预登记确认集上比较，不能看过DEC1 target结果后替换本矩阵模型。完整B0/两臂DEV可在新计算预算余量内继续，原baseline-relative发布仍须全48和原80格最终门。若资源不能同时容纳所有旧可选计划，暂停P3/额外seed及重复旧指标，报告哪些旧承诺本轮未完成；不虚构PV1已闭合。

CR2沿原授权完成固定R1小银行，不热混入DEC1。旧CORR+新decoder不是原模型；R1神经适配作为以后独立问题，不加入本轮解耦干预。

## 8. 本轮有限预算和原预算关系（实际授权，不是无限追加）

DEC1是原PV1预算内的明确科研重分配：总H200/可等价核准GPU累计96小时、24PBS、单卡和旧截止不增；原20小时final保护不挪用。新目标仿真/接口试验允许从原未用source episode预算转为DEC1明细，原每次作用与费用不能漏记或重置。原CR2单独已批准的资源不拿来暗补DEC1。

本轮DEC1执行上限：48 GPU分配小时（含允许提前的两臂8192更新，不额外再加一份训练预算）、8次新增GPU提交、8192新optimizer updates、14000模型生成、136物理starts（8commissioning+64核心+32可选stress+16原TRAIN探针+16确证工程替补）、5,000,000总live/shadow/reconstruction ticks、12000构造、96 CoreLab核时、96 Hopper CPU核时、8local核时。最多4个Core CPU，始终一个科学调度owner。新输出Hopper/ Core各64GiB、local8GiB、总新传输64GiB，均还必须满足原累计更小余量与站点配额。

每格工程替补至多一次、总16，仅原生确证的工程不完整且副作用闭合；有效策略/安全门拒绝不能重试。target实际WBC成本未知时先commissioning实测；不能用source速度假定完整目标矩阵可支付。若136项不是都能支付，先完整核心64，省略明确可选stress，而不是把核心裁成有利子样本。

实际支出门=min(本卡剩余,原PV1所有已用/预留/final保护后剩余,站点限制)。旧held job预约未结清时继续占用预算和单卡约束，不能凭无当前GPU占用当作不会再次启动。原clock：10月25日02:29:56SGT停新job/候选；14:29:56停新训练；18:29:56停新物理；10月26日02:29:56全部关账。DEC1不得重置T0；到10月23日20:00SGT交第一份实测结果或明确阻断表，到硬截止交完整或诚实partial科学包。这些是决策时点，不是成功保证。

同一新目标研究若需要不同GPU资源：只允许在用户已有合法使用权、项目注册许可的资源上先做有界兼容性资格，所有GPU时/费用仍纳入本48/原96，单卡；不购云卡、不转移秘密到未授权主机。必须证明held作业不可能并行恢复、旧占用与reservation可核清；若不能，就只准备备用方案，不实际启用第二GPU。平台/精度变化必须同一核心四单元一致并单列，不能把旧Hopper结果直接当匹配结果。

## 9. WBC团队的交付与高层团队边界

WBC团队交付EXECUTOR_MANIFEST：机器人/控制器版本、权重/config/checksums、实际参考ABI/frame/TCP/夹爪约定、原preview offsets、实际EEF与时间反馈、reset/reject/stop与原安全链、通用可行工作域及运行入口。说明WBC训练数据/任务依赖；独立冻结不等于整个系统没用过机器人数据。
高层团队交付POLICY_MANIFEST：共享权重、processor/stats、图像与state模态、source数据lineage、submit-vs-achieved标签、推理精度/噪声/频率与接口版本。校准只限必要rigidframe/传感器同步和task-blind接口资格，任何target task表现驱动调参必须披露。

不要求WBC团队在本计划内修改控制器。其交付未到，不以理想执行或某个source gain扰动冒充第二控制器。若只有一个真实执行器，论文只能报告source task-space控制，不能把软件两个函数名称为VLA-WBC解耦验证。

## 10. 工程自助与当前站点hold的正确处理

先刷新679029准确H/Q/R/F、owner和两端lease、tracejob、pending效果及现有CR2 worker。恢复唯一owner的只读DNS/SSH有限退避与状态机，不启动第二dispatcher；故障指纹相同的重复密码/submit不能称修复。
系统hold不在普通owner释放权限内：准备包含准确job/time/hook-rejected/无launcher证据的脱敏管理员请求，沿用户已有站点支持渠道按其权限处理；不擅自qrls、改hook/认证配置或伪造proof。管理员未授权解除/处置前不重复qsub绕过同一问题。若站点允许删除确切自有held job且原owner完成无科学作用与计费归档，可在本新调度授权内安全终止该已阻断作业并核对真实终态后冷换下一作业；不取消健康R科学作业、不删除旧证据。无法获得站点许可则SITE_HOLD_REQUIRES_ADMIN，不无限折腾研究代码。

未来提交前静态检查final binder权限/哈希、RGB与numeric全成员、模型alias、state/frame/codec、sourceCI、quota/认证、FD关闭及自身资源余量。未知qsub/release/install/update先journal对账，不能盲重试。普通同语义修复自主，最多3轮有实质变化的CPU诊断后报告最小阻断；不得烧H200来测试已可CPU复现的文件错误。

## 11. 交付、决策与论文节奏

最少固定工件：DEC1_NATIVE_ADOPTION_AND_BUDGET.json、DEC1_EXECUTOR_POLICY_INTERFACE_MANIFESTS.json、DEC1_SOURCE_ONLY_INTERFACE_FREEZE.json、DEC1_EVALUATION_PANEL_SEAL.json、DEC1_ALL_ATTEMPTS_AND_PAIRED_RESULTS.json、DEC1_FINAL_SCIENCE_REVIEW.md、DEC1_RESOURCE_FINAL.json。必要私有raw数组/视频留注册存储，Git只有脱敏指标/完整hash/来源。

首次owner回DEC1_AUTHORITY_ADOPTED或准确native blocker：明确旧B0门已按本新决定改为选模前、剩余预算、目标WBC是否可执行、source训练或固定CORR矩阵下一动作。不能把本计划再次标成全局等待新GO；也不能把发布当成已经使用GPU。每次新科学结果合并回#11，工程/站点阻断回#47，不继续创建LG05/LG06等平行owner文档。

结果分类：SUPPORTED_PILOT、NO_ADDITIONAL_INTERFACE_BENEFIT、SOURCE_POLICY_LIMITED、TARGET_EXECUTOR_BLOCKED、INCOMPLETE_RESOURCE_OR_SITE。无显著收益就明确说没有额外接口收益；有目标成功但无反馈优势，只能主张初步可迁移性；没有两个执行器真实任务数据，就没有主要解耦证据。

CVPR2027官方注册11月10日AoE、全文11月16日AoE（对应新加坡11月11日/17日19:59附近）：https://cvpr.thecvf.com/Conferences/2027/CallForPapers 。本阶段先取得有限真实科学结论，10月26日以后扩大20+几何、多噪声/训练seed、语言改写、第三执行器或硬件的论文确认实验，需要独立有资源的后继范围；它们不是本次自动追加支出。论文写作/相关工作/接口团队沟通现在并行，不需要等policy发布。

最终原则：贡献主张必须随证据降级或升级。训练数据不含关节标签不证明动力学完全无关；冻结不证明控制器可替换；没有target示范训练不等于没做target校准；两任务语言区分不等于开放世界语义。我们要交付的是在这些明确边界下可复现的VLA-WBC解耦结论，而非新的模块示意图。