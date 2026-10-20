# DEC1：Controller-Disjoint VLA / Frozen-WBC 解耦证据与新科学授权

日期：2026-10-21，Asia/Singapore。Decision/Goal ID：DEC1_CONTROLLER_DISJOINT_VLA_V1。
状态：USER_AUTHORIZED_NEW_SCIENCE_PLAN_PENDING_NATIVE_ADOPTION。
授权依据：本轮用户原话「你思考确定一下我们的创新、贡献点，我们focus在VLA和WBC的解耦，确定后给出新的授权计划，尽快拿到科学结论」。本文件将这次新科研决定限定为下述可执行实验、依赖修订和预算重分配；不是再次把所有新实验标成NOT_AUTHORIZED。发布不等于现场已经采用或运行。
审核基线：2d6c4d68380d22a284412167aa465609903fdf35。原PV1 authority：9eff29a44a193dd4d3620dc0bfe5764a402eff42。单owner、原生站点权限、已有workspace与运行安全继续适用。

## 1. 论文定位与三个待验证贡献

工作标题：Controller-Disjoint VLA: Execution-Grounded Task-Space Trajectories for Frozen Whole-Body Control。
核心问题：不使用目标WBC产生的任务示范训练高层、不联合训练或调参目标WBC，同一份语言条件EEF策略能否跨源执行器和独立冻结的目标WBC复用？实际执行进度反馈能否降低这种迁移的失效？

C1 数据/学习解耦：高层从源域视觉、语言、实际状态学习提交的EEF参考，不学习目标关节/力矩或目标WBC latent。目标控制器由另一团队独立交付并冻结。准确称target-controller-disjoint training，不称所有数据都与任何执行器无关。
C2 接口/执行解耦：把任务空间几何、实际执行进度和夹爪事件对齐；比较时钟锚定与实际反馈锚定。新增方法位于高层接口，不修改WBC控制律，且不假设期望EEF就是实际EEF。
C3 可替换性的实证：同一权重在两个冻结执行器上的全交叉任务对照；区分任务语义、接口合法性、执行跟踪和物体结果，用真实交换执行器而非模块图证明复用边界。

这些是实验前贡献假设，不是已证创新。CR1范围研究属于接口可靠性支撑证据/消融，不把min-max扩范围单独包装为主要算法创新。两个固定任务不能自动支持开放词汇、任意本体、力控制、学得移动意图或实时部署。

## 2. 最近相关工作与新颖性边界

HAMSTER：off-domain数据训练高层2D EEF路径、低层3D策略执行。https://arxiv.org/abs/2502.05485
3D HAMSTER：高层metric 3D轨迹指导下游控制。https://arxiv.org/abs/2606.31329
UMI：相对轨迹、推理延迟匹配和跨机器人部署。https://roboticsproceedings.org/rss20/p045.html
LeVERB：视觉语言latent与RL whole-body控制分层。https://arxiv.org/abs/2506.13751
H-VLA：关键动作与运动生成、统一相机坐标动作空间。https://arxiv.org/abs/2609.22895
Opt2VLA：多任务VLA输出几何与接触力、任务特定RL WBC执行。https://arxiv.org/abs/2609.23968

不声称首次VLA+WBC、首次EEF、首次分层、首次机器人无关表示或首次处理延迟。预期差异是在目标任务示范与目标控制器训练不参与高层学习的约束下，研究冻结控制器迁移的失效及执行反馈接口。方法组合的新颖性、性能价值和CVPR竞争力仍需实验和文献检验，不能保证优于已有全部方法或保证录用。

## 3. 当前证据与主线阻断

- CR1 e710f150：诊断7条旧FAIL/6个独立于FIT的几何组恢复，旧139条codec无回归。R1 SHA256=50d9672397fd0b5353da2011b9f9f568dc9c65c26fb271458f87d60b2d497fa8。不是模型/物理成功的改进。
- PV1最后已证B0=14/48，9成功/5有效失败，A/B新更新0。共享CORR@2048权重已存在，可直接做接口迁移实验，不必等新权重。
- 2d6c4d68收据显示child12/679029.hopper-m-02为H、Hold_Types=s、execution hook rejected/exit -16，不是普通Q。无PBS_READY/START/模型/任务效果；约20–21秒prelaunch分配需保留。原spent20473 H200秒、reserved28860秒、PBS11/24仍为待实时刷新快照。
- 原local owner另因只读SSH临时DNS失败退出；与站点hook拒绝是否因果相关未知。系统hold需管理员处理，不可由普通owner擅自qrls或绕过hook。
- CR2独立批准且已采用；12组零积分静态资格完成，首四条实际采集开始，尚无终局READY。它保留原48h/32Core核时上限，不自动扩采。
- 现有groot_to_wbc.py只是离线参考转换：使用SE3弧长preview，但controller_timestamp锚定、gripper按时间取值；不能称已具备目标WBC物体闭环。复用已有实现并正确归属。

## 4. 本轮明确授权的变化与保留项

### 4.1 新决定

A. 解除“B0全48必须先于任何新源域训练”的依赖。原先封存的A_CONTROL/B_COVERAGE可以在数据、真实样本、完整恢复、站点预算与安全合格后先训练；不得根据部分B0成绩改变配方。完整B0仍是任何baseline-relative正式选模、原PV1等级和LOCKBOX80的前提。
B. 新增DEC1仿真scope：已有CORR@2048/R0在源执行器和WBC团队交付的一个冻结目标执行器上运行T1/T2；批准下述两个高层接口模式。新代码与结果独立版本化，不热改旧B0胶囊。
C. 允许最小任务接线、相机/状态/记录器与CPU接口测试，仅复用已有合格物理资产；目标为模拟环境，不是真实机器人。
D. DEC1可以在旧B0不完整、新权重未完成或未达到旧发布成功率门时运行，结果单列。这是取得科学证据，不是降低旧PV1发布要求。
E. 工程修复、精确CI、冷绑定、预算内阶段转换无需逐项另请GO；真实站点权限、未知副作用、安全和预算硬限制仍有效。

### 4.2 不允许改变

WBC代码/权重/增益/cost/solver/腿策略、plant质量/摩擦/惯量、夹爪PD、原安全阈值/物体评分/32s任务时限保持。无force head、目标任务关节脚本、目标示范训练高层、world model、新DAgger、CR2扩采、EEF13补零假标签或真实机器人。
建立显式新admission版本：旧authority仍按旧B0训练前门测试；DEC1授权可先训练，但完整B0缺失时仍禁止原baseline-relative选模/LOCKBOX。原历史14格、失败、scope和时钟不重写。

## 5. 固定策略身份、接口方法和两种执行器

第一轮权重固定CORR@2048，checkpoint manifest SHA256：94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38。
原R0 statistics：fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7。
原新数据银行manifest：60f7c35a05766fe23974eba3a72f53ca4db5545a3768e6c1cfea164ab4e58294。
输入/输出保持真实双RGB、语言、实际EEF/夹爪状态→H40 EEF10；原20Hz参考、Prefix8、2.5Hz模型重规划和推理精度。WBC自身频率按冻结交付披露，不伪装统一频率。

E_SOURCE为现有已注册的固定底座ARX5任务执行器。E_TARGET为WBC团队独立交付、版本权重配置冻结的Go2+ARX5真实动力学仿真执行器。第一轮只研究共同可达近场T1/T2；不要求学习导航/BASE，free-root平衡由原WBC承担。不得以理想跟踪、焊死base或覆盖关节状态假装目标WBC。目标未交付则TARGET_EXECUTOR_NOT_DELIVERED，只阻断target测量，不阻断source训练。

源/目标可能还同时改变身体、相机和动力学，因此准确称executor-domain transfer。纯WBC因果结论需要其他因素固定的后继对照；不能通过改source gains造一个“新WBC”。

### 5.1 I_TIME：正确的时钟锚定对照

使用现役语义正确的时间锚定接口：相同frame/TCP/动作与观测频率，原preview转换、gripper时间序列。所有确证软件bug对两臂共同修复，不能保留错误基线。特别核对_prepare_chunk在source等于expected_source_frame、却不同wbc_frame时，传入frame_transform是否真的应用；一旦证实遗漏，两臂用同一修复。

### 5.2 I_PROGRESS：Execution-Grounded Trajectory Interface

唯一研究干预是以实际TCP/夹爪反馈而非名义时间锚定当前已接纳轨迹的执行进度。概念上在局部单调有界SE3路径上投影实际状态：s_hat=argmin d_SE3(T_actual,gamma(s))，再使用原固定空间preview offsets。高层仍按同频率从新观测重规划。

进度局部窗口、SE3权重与停留规则只由source TRAIN/无物体CPU接口fixture固定，target结果不能调参。禁止读物体真值、接触oracle、Teacher phase或task ID来决定轨迹/夹爪。新chunk必须重新锚定，不把旧chunk的归一化s直接沿用。

夹爪事件保留原模型开合顺序与幅值，只与相同路径进度一致绑定，不加task-specific抓取/释放脚本。静止段、纯旋转、重复waypoint、原地开合和停留必须有明确局部时间/索引规则，避免最近点投影死锁或跳过抓持。推理次数、最长物理时限不增加。

源侧冻结前通过：恒等/刚体frame变换，actual progress等于名义进度时两臂一致，局部不倒退/过跳，静止/旋转/夹爪平台段，chunk reset/stale/NaN/reject停止。所有实际参数形成一个固定配置，不在确认集搜超参。若WBC已内部做同样的路径推进，核对ABI避免双重进度估计；结果可能是接口等价或无额外收益。

### 5.3 安全不能由重参数化绕过

两臂先检查原始完整H40合法性，再检查实际映射后的完整参考和将执行prefix。不能用平滑/重计时使原非法H40绕过原速度门。source沿原S1；target必须使用该真实模型对应、由WBC团队提供并冻结的安全/停止与预测链，不能把source专用shadow误称target安全。

两臂同推理精度、噪声张量/次数上限、实际32s任务时长、20Hz观测和原保持谓词；不延长到成功、不best-of-N、不修改夹爪/物理。wall-time通信等待与sim-time分开，不能声称实时。WBC不能接受语言/T1-T2 ID选择两套任务脚本。

## 6. 首轮64格核心对照：无需新训练即可取得解耦结论

### 6.1 Commissioning与封存

最多8次接口任务：2个source TRAIN或专用commissioning几何×2任务×2执行器。target仅用于输入/设备/安全资格，不训练高层、调WBC或挑容易确认场景。任何target任务驱动的接口调参都应披露并将“零目标适应”降为calibrated transfer。

随后在任何新确认结果前封存8个独立新几何：2个既有外观×2方向×2距离。NEAR0.15–0.18m、FAR0.19–0.22m；每槽32个事前静态候选。与所有旧TRAIN/C1、CR1/CR2/COV1、PV1 DEV/CONFIRM/LOCKBOX及同几何兄弟排重。只用获准几何索引，不读评价图像/结果；仅按共同声明工作域、静态安全/可达/视野/评分输入条件选，不按Teacher或policy成功筛选。池耗尽报告域不可行，不无界换seed。

### 6.2 64格矩阵

8几何×2任务×2执行器×2接口×1预定噪声bank=64格。
每个执行器/接口16次，逐任务8次。同几何内相关，不能称64独立场景。固定噪声bank在四单元复用事前张量，按决策序号读取，不给失败额外抽样。

第一轮只用已固定CORR@2048/R0，不选择新中间checkpoint；不等待新训练、更多数据或B048。真实初态物体/目标在统一task frame配对，但各执行器用自己的实际机器人/相机观测，不能伪造后续RGB相同。以几何block交错执行，避免先看一方法结果再改另一方法。

目标未ready不能用source重复凑矩阵。完整64即使负结果也交付；有效失败不重跑，工程未完保留且按有限规则替补。新target确认数据不能回流训练。

### 6.3 主要结果和允许的结论

报告任务成功矩阵、逐任务/方向、H40参考拒绝、S1/target-native拒绝、抓持/无抬升/释放/脱离/保持、实际跟踪残差与进度延迟、模型/控制器hash、target任务训练示范数和实际GPU/CPU成本。
Delta_target=SR(target,progress)-SR(target,time)。Delta_source同理。Interaction=Delta_target-Delta_source，仅说明这两个执行域上的交互，不能排除相机/身体等改变。

以8个几何block做配对与聚类重采样区间，不把两任务当独立机器人场景。源策略弱时不能以“小迁移差”宣称鲁棒；单噪声bank不支持随机性稳健结论。主要区间跨0则PILOT_SIGNAL_OR_INCONCLUSIVE，不宣称显著。仅目标成功但接口无增益，结论是初步可复用/无额外接口收益；无目标执行器数据则没有主要解耦证据。

早期继续线索不是验收成功率门：两任务均出现target完整成功、无native故障/泄漏，Delta_target为正且无明显相反任务趋势，则值得扩大确认。否定或不确定结果同样发布，不事后调整标准。

### 6.4 可选32格stress

仅在最初plan已预登记、核心64完整且预算充足后：同8几何×2任务×2接口，仅target加入100ms执行接口延迟，共32格。不是调WBC、改物理频率或放宽watchdog；须有明确的延迟buffer/当前状态安全检查CPU资格。无法保持原安全语义则NOT_RUN。它是已打开几何的预注册stress，不叫新盲测。不同时追加语言/物体/world-model实验。

## 7. 原源域A/B训练可以前置，但不污染首轮解耦对照

新authority允许原A/B先于B0全48更新：同CORR父、独立同配方optimizer、R0、139新数据银行/原C1配比、4096步/臂、1024 TRAIN-only探针、原BF16/TF32设置与batch32，全状态checkpoint/真实消费/恢复保持。源域配方不能看部分B0或DEC1目标结果改变。

事前调度规则：两个执行器和接口ready且有合法GPU时，优先已有CORR的64格矩阵；target未交付而训练资格ready时，优先两份源域4096端点。不能因为B0不完整让GPU空转，也不依target成绩选择先训哪臂。可在同一合法allocation内衔接已预登记的阶段，完整预算与8h墙钟必须满足；健康同allocation不因每个checkpoint重复资格。

新端点不替换本次预选CORR矩阵。未来新赢家的迁移需新预登记确认；完整B0和两臂DEV仍是原baseline-relative发布条件。原旧可选P3/额外seed暂停；剩余预算不够时明确哪些旧承诺未完成，不能把DEC1结果称为原PV1已验收。
CR2按其独立GO继续且不挤占活跃主线；R1神经适配以后单独研究，不热混入DEC1。

## 8. 本轮有限预算：既有预算内重分配，不无限追加

全项目原PV1累计96 GPU分配小时、24PBS、单卡、旧截止与20小时final保护不增加。DEC1新增目标仿真/接口实验可从原未用source episode/CPU配额重分配，作为显式新分类记账，不删除任何已用或预留费用；CR2独立预算不借用。

DEC1上限：48 GPU分配小时（包含提前的A/B合计8192步，不另加训练预算）、8次新GPU作业提交、8192新optimizer updates、14000神经生成、136物理starts（8commissioning+64核心+32可选stress+16原TRAIN探针+16工程替补）、5000000 live/shadow/reconstruction总ticks、12000构造、96Core核时、96Hopper CPU核时、8local核时。Core最多4核，单科学owner；Hopper/Core新输出各64GiB、local8GiB、新传输64GiB。始终取本上限、原累计扣除spent/reserved/final保护后余量、站点权限三者最小。

每格工程替补最多一次、总16，必须确证工程而非策略失败且已闭合副作用。target WBC真实CPU成本未知则先用commissioning计量，不能把source耗时当target上界。预算不足先省stress/可选工作，不缩核心64到有利子集。

旧held作业reservation和可能恢复的占用需核清；不是无当前GPU就可并行开第二卡。备用GPU仅限用户已有合法使用权、已注册许可的资源，有界数值/精度/模型兼容后才可使用，计入本48/原96且全程单卡。不得购云卡、发出新的付费承诺或向未授权主机转移私有模型；平台不同则四单元全部匹配该平台，不混用旧Hopper结果。无法排除held恢复并行则只准备备用方案，不实际使用。

原时间不重置：10月25日02:29:56SGT停新job/候选；14:29:56停新训练；18:29:56停新物理；10月26日02:29:56关账。10月23日20:00SGT提交第一份真实对照或精确阻断表，硬截止交完整或诚实partial包；这是决策时点，不是成功保证。

## 9. 两团队交付与工程自助

WBC团队提供EXECUTOR_MANIFEST：机器人/权重/config/checksums、参考ABI/frame/TCP/夹爪、固定preview offsets、实际EEF/时间反馈、reset/reject/stop、安全链、通用工作域与运行入口；说明底层训练任务/数据依赖。高层团队提供POLICY_MANIFEST：共享权重、processor/stats、source数据lineage、真实提交参考标签、观测与推理/噪声/频率、两个接口配置。

必要rigidframe/相机同步校准与task-blind接口资格允许，target任务性能调参不属于零适应主张。WBC未交付时不理想化替代，不在上层改控制器；一个执行器的两个函数名字不构成controller-disjoint测试。

当前工程先刷新679029的H/Q/R/F、两端lease、tracejob和pending，有限恢复唯一owner的只读DNS/SSH状态机，不开第二dispatcher。系统hold不在普通owner权限：准备准确job/time/hook拒绝和无launcher证据的脱敏管理员请求，通过已有合规支持渠道按权限处理；不擅自qrls/改hook/伪造认证/重复qsub绕过问题。若站点允许删除准确的自有held作业，且无科学效果、费用和原件可保留，可按本新调度授权安全终止该已阻断job，确认终态后再冷续接；不取消健康R任务。

未来提交前核对最后写入后的权限/hash、完整RGB/numeric、模型alias、frame/codec、精确CI、quota/认证、FD和预算。未知qsub/release/install/update先journal，不盲重试。相同故障最多3轮有实质变化的CPU修复后报告精确阻断，不用昂贵分配测试能CPU发现的静态错误。

## 10. 交付、采用与论文节奏

固定小工件：DEC1_NATIVE_ADOPTION_AND_BUDGET.json、DEC1_EXECUTOR_POLICY_INTERFACE_MANIFESTS.json、DEC1_SOURCE_ONLY_INTERFACE_FREEZE.json、DEC1_EVALUATION_PANEL_SEAL.json、DEC1_ALL_ATTEMPTS_AND_PAIRED_RESULTS.json、DEC1_FINAL_SCIENCE_REVIEW.md、DEC1_RESOURCE_FINAL.json。原始数组/视频留注册私有存储，Git只发布脱敏指标、hash与索引。

owner首次回DEC1_AUTHORITY_ADOPTED或精准native blocker，明确B0门修订、实际余量、target交付和下一source训练/固定CORR矩阵动作。不要再次把整个计划置为等待新科学GO，也不能声称文件发布就是原生采用。重要新结果回#11，工程/site阻断回#47，不再新建重复LG调度体系。

结果分类：SUPPORTED_PILOT、NO_ADDITIONAL_INTERFACE_BENEFIT、SOURCE_POLICY_LIMITED、TARGET_EXECUTOR_BLOCKED、INCOMPLETE_RESOURCE_OR_SITE。没有两执行器结果就不声称解耦；不确定区间和负结果照实保留。

CVPR2027官方注册11月10日AoE、全文11月16日AoE，约为新加坡11月11日/17日19:59：https://cvpr.thecvf.com/Conferences/2027/CallForPapers 。先用本阶段产生有限实证，10月26日后20+几何、多噪声/seed、语言泛化、第三执行器或硬件等论文确认实验需独立资源安排，不自动超出本预算。方法/相关工作/论文表格和两团队接口沟通现在并行。

贡献随证据升级或降级：无joint标签不证明动力学完全无关；冻结不证明可替换；无target示范训练不等于无target校准；两任务条件化不等于开放世界。目标是尽快交付这些明确边界下真实的VLA-WBC解耦科学结论。