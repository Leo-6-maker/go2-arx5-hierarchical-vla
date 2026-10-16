# PI0294 | GPU排队期CPU准备：从Teacher数据到BC、DAgger和FlowDAgger的完整研究流水线

日期：2026-10-16，Asia/Singapore
状态：USER_REQUESTED_CPU_PREPARATION_HANDOFF；论文/工程路线及有界CPU准备，**不授权新的GPU/PBS、模型推理、优化更新、MuJoCo积分/渲染、Teacher科学条件或WBC变更**。
审核GitHub基线：bf3d1e793130d14713e64e1bde550d1b5ba68bb3（active branch pi0267/executor-decoupled-recovery）。
使用既有唯一local-PC owner、统一入口、不可变running capsule；不新建协调者，不覆盖已采用E0/现役安全停止。当前文件及GitHub Issue评论不是远端Codex或服务器实际启动收据。

## 0. 已确认事实、证据层级与并行原则

- 当前E0已经由人类明确采用，固定实验版本22e5f0baeed0af662a7cf1c97edae1f4060947fe；run 20261016T083821Z-localpc-e0；T_adopt 2026-10-16 16:38:21 SGT。原E0是M2048/原v3.0/EEF10/source-D0＋S1，TRAIN4→DEV8，不按TRAIN成功率新增停止门，不训练。
- 原E0 Job 670049.hopper-m-02的最新**已发表**原生快照为Q（排队）；这是GitHub证据，不是本次SSH实时状态。不要再qsub第二份或撤销/重排健康job。E0时钟/资源/已签批权限保持；若实际状态R/F，由现有owner优先维护/收尾E0。
- v3.0已有158条准入固定底座示范，TRAIN126/DEV32；v3.1共有295条且含v3.0，不能热混。M2048确实由M512全状态续训至2048更新、累计65,536 TRAIN样本；历史16条件均NOT_RUN，不可误写0/16。
- 整机TF-R3最新发表含有效失败和新反应式Teacher推进，但完整fullbody成功seed仍为0；E0人类采用记录另称当前TF-R3 E2具有原生安全STOP，所有历史FAIL/lease/执行版本必须现场只读确认，不能靠旧进展文字声明目前仍RUNNING或自动恢复。WBC/腿策略/任务分数/原生安全冻结。
- FlowDAgger官方GR00T backend目前针对LIBERO-90任务57、7D OSC动作，模型内部反演为padded动作张量；**默认intervention_probability=1.0，训练rollout实际全部expert chunks**。它提供算法/代码原语，不自动证明针对我们EEF10/EEF13的真正on-policy DAgger。
- 本文阶段用两条平行线：Lane E0原固定评价由既有采用控制；Lane CPU-PREP仅静态研发/保存状态测试/新数据结构/离线模拟接口，在GPU排队也可完成。两线共享owner和资源，prep不得用E0的GPU/PBS及现役native租约。

主要来源：
- reports/PI0294/20261016T083821Z-localpc-e0/README.md、E0_ADOPTION_AND_BINDING.json、E0_SUBMISSION_ACTUAL.json、E0_PANEL_MANIFEST.json
- reports/PI0294/20261013T142457Z-localpc-pi0294/DATASET_CARD.md
- reports/PI0294/20261014T121640Z-incoming-pi0294-r3/M2048_TRAINING_RECEIPT.json、S1_TEACHER_QUALIFIED.json
- docs/RESEARCH_CHARTER_TASK_SPACE_MANIPULATION_V1.md、docs/protocols/GR00T_TASK_DATA_FIRST_FROZEN_WBC_SCOPE_V1.md
- https://github.com/microsoft/FlowDAgger/tree/3426f246510fd67a9e39c3c08bd5b02d586ad688/flowdagger_gr00t
- https://github.com/microsoft/FlowDAgger/blob/3426f246510fd67a9e39c3c08bd5b02d586ad688/shared/intervention_handler.py

## 1. 总体研究问题：把任务学习、执行、数据迁移分开证明

保留视觉/语言/实测EEF和夹爪 → GR00T task-space高层轨迹 → SafetyGate／时间身份 → 冻结WBC＋独立夹爪通道 → 物体/EEF/BASE真实反馈。现有M2048输出EEF10；未来目标FB13含EEF姿态/夹爪10维＋BASE软意图3维，**BASE软意图不是直接腿速度控制**，缺少BASE标签时absent和zero不得混淆。

三项分开的科研假设：
1. Behavior learning：当前GR00T是否已在source D0学会T1/T2操作？先完成E0。
2. Domain/embodiment adaptation：source-only策略接入fullbody Go2＋ARX5、冻结WBC后损失多少？5/20等少量target示范如何修复？不能把固定臂到自由根等同第二个机器人本体。
3. Interactive correction：在学生实际访问的偏离状态上，Recovery Oracle+DAgger是否以同等标签预算提高成功率？FlowDAgger latent steering与正常Flow Matching更新相比如何？
论文或报告分别披露上述证据；不能用CPU测试、Teacher计划或训练loss替代任务成功。

## 2. 完整闭环：P0 → P8，各阶段数据/训练/准入与失败路由

### P0 数据生产环境的最小可执行契约

source-domain：固定底座ARX5旧D0/S1及原真值/运动学/物体评分，用于source BC/E0/以后source-DAgger；已经有v3.0数据，不因新Goal而重新造一套。
target-domain：free-root Go2、12腿＋ARX5机械臂、真实有限夹爪、机身和腕相机、原生WBC、20Hz观测与H40前缀8步、真实接触与任务评分。与source独立版本化。
每domain锁定对象/目标/语言/相机标定/坐标轴/工具TCP/任务成功/安全/step、ROS时钟与参考安装语义。每个命令保留raw proposed / gated / accepted-installed / actually scheduled prefix / achieved state / outcome，禁止用未来实际轨迹替代提交动作标签。
source/target均用原合法安全门，禁止把WBC保持RMS设置为新的全局任务门；但机械/接触/时序硬门不能放松。

### P1 示范与数据归属：四类bank分开

A Source Successful BC：原v3.0的已准入158条，先锁定旧TRAIN/DEV和原stats；v3.1只有在独立实验中使用，不能混原统计。
B Source/Target Intervention：学生实际访问时的专家建议，分oracle有效且未执行 / expert takeover实际执行 / student executed三种；不要求整个rollout成功才能用合格纠正标签，但**不能标成成功示范**。
C Target Successful BC：独立完整free-root WBC物理task真正成功、500Hz安全/接触/时间/数据验收合格，才标target success。单侧接触或静态IK不算。
D Failure/Diagnostic：安全失败、计划拒绝、无接触/任务未完成、无效专家建议独立保留；不能混入成功BC。人类视频、其他机器人和MimicGen源轨迹可独立贡献技能表征/来源域动作监督，但其目标域任务成功需要实际目标执行。

### P2 基础BC/Flow Matching训练与源域E0评价

已存在M2048：冻结VLM的既有训练及原Source v3.0/EFF10 stats，E0先TRAIN4后DEV8，评估真实生成、S1拒绝、接触、任务动作、释放及结果。E0已采用且排队，不运行第二份模型作业。
根据E0结果**另立后续实验**：
- TRAIN接近成功且DEV弱：研究覆盖/语言反事实/视觉域，必要时把v3.1作为单独stats/revision的新增数据。
- TRAIN也不会操作：先5～10条固定TRAIN子集拟合、动作codec/语言/夹爪边界/训练优化诊断，而非直接增加训练步数。
- source能力成立：优先评估同一冻结策略对target WBC的source-only迁移，再按target示范量适配。
- E0遇真实安全或身份故障：按E0原协议停止相关实验，不因本prep方案修改S1或者复跑有效负例。
所有新训练/模型查询/PBS需要独立新的科学和资源采用，本CPU方案仅准备条件化配方。

### P3 构建两个职责不同的Teacher

Demonstration Teacher：在标准可行场景给出成功任务动作；它可利用物体真值、源轨迹/物体相对技能、IK和路径规划，然后经原执行器做真实成功确认。优先复用已合格source D0专家，target fullbody另修但不作为source训练的总前置。

Recovery/DAgger Oracle：从学生**当前实际访问状态**(measured TCP/gripper/base、物体真值仅限专家、实际事件与动作历史)推断当前phase和目标，生成一次完整H40合法未来task-space参考，附质量、有效区间与不可恢复原因。阶段切换必须依赖真实接触、物体进展、抓/放、分离，而非仅复读源时间戳。可以返回INVALID/UNRECOVERABLE，不强制造出修正。
接口建议：
    query(state_real_t, privileged_task_truth_t, task, executed_history, horizon=40)
        -> {valid, expert_H40, active_dim_mask, phase, reason, action_semantics, version, certificate}
动作/观测坐标和时间绑定；禁用joint控制绕过WBC；source专家动作维度EEF10，target最终EEF13且BASE维缺监督不得补零。
**当前S1 source教师资格仅证明标准source起点，绝不等价于偏离状态恢复能力。** 在GPU前只可用已保存实际状态/CPU复制状态做接口fixture；人为扰动的状态标为SYNTHETIC_CPU_FIXTURE，不是student-on-policy。

### P4 DAgger在线闭环

在已经验证学生/专家可用的domain中：
1. 学生根据本体真实RGB、任务语言及实测状态生成H40 proposal（含固定采样身份）。
2. 同一actual observation state上查询Recovery Oracle的H40标签；专家独享privileged sim truth但不进入学生输入。
3. 原SafetyGate验证学生计划的合法执行前缀；额外风险/专家可用性仲裁：安全时可执行学生prefix8并记录label；风险需专家接管时执行专家prefix8；双方都不合格就停止该episode。**不能先执行有安全拒绝的学生计划，只为了多收DAgger偏离状态**。
4. 经20Hz实际推进0.4s取新的真实观测，重复。每次记record parent state id、原样expert_H40、实际执行者、动作安装、实际 prefix8/时钟/物体结果。专家未执行的H40是合格建议标签，不是目标已执行成功动作。
5. 以真实student-visited state的纠正数据与来源示范聚合，固定有效mask和sample weights，训练下一轮。专家输出不能因结果不好临时换seed；E0期间不做步骤1—5的真实模型/物理循环。

科学DAgger要求student确实有非零比例独立行动、专家纠正采样来自其访问分布。仅用既有教师全程执行样本不是DAgger。beta schedule／action disagreement／risk takeover的具体阈值要在后续单独固定实验中采用；不能把风险门变成新放宽原机械安全。评价时expert接管=OFF，安全器继续ON。
若expert在偏离状态无合法label，应记录query invalid与coverage，而非把学生错误动作当正样本。完整H40未得到验证时不可把仅有8点的补丁尾随32零，必要时使用有理论/实现支持的mask并单独模型实验。

### P5 两种不同的DAgger学习方法

Baseline Action-space DAgger：训练数据合并后，仍使用原GR00T Flow Matching BC损失拟合专家H40动作，控制模型架构与更新步数；与“同量额外普通成功示范BC”对照，隔离访问状态数据的收益。

FlowDAgger alternative：冻结GR00T本体，观察/动作专家样本通过本体processor变换到真实有效的padded model action space，使用逆向Flow Matching (per-step fixed-point reverse Euler) 求能生成专家动作的初始latent w*；训练小型steering network g(obs)->w*，推理从g(obs)初始化原forward flow sampler。
官方example使用LIBERO_PANDA OSC_POSE 7D→内部padded H40×132；我们的M2048是EEF10，不可直接复用其gripper变换、绝对/相对动作假设或132有效维映射。EEF13新分支也不能在未学习BASE时凭latent添加3维能力。
**先做CPU-only源码/shape/processor/mock反演结构检查，GPU到位且另获授权后才用真实checkpoint运行encode→invert→forward→decode round-trip。** 反演必须在真实同个obs和数值设置下验证，报告有效维的XYZ/SO3/夹爪重构误差、invalid比例与模型前向数，才能考虑训练steering。
官方GR00T example的默认intervention_probability=1.0，因此不能把运行其默认命令视为包含足量student-visited DAgger状态；可参考shared/intervention_handler里的beta-decay/disagreement设计，必须与原风险限制、oracle validity适配。
对照：M2048、普通BC同量新增专家示范、Action-DAgger、FlowDAgger（若反演合格）；相同train/eval scenes、真实专家查询预算与观察来源。报告任务成功、干预率、专家查询/有效率、碰撞拒绝、未见布局、数据量及计算成本。

### P6 来源→目标WBC迁移与少样本微调

目标首先是source-only M2048模型（以及确需的显式EEF10→目标参考适配，且不宣称BASE已学）在完整Go2＋ARX5上的近场T1/T2评价。所有真实安全未通过的执行器行为单独报告，不用WBC质量RMS作为任务成功的替代。
当target能安全生产成功参考后，目标示范建议每task 2～5条来源证明、5～10条小样本诊断、20～50条适配初步学习曲线；100～200/任务只有当学习曲线支持时才扩。数量只是规划档位，不是当前批准资源。训练source-only与source+target(0/5/20)配对，任务/语言/场景/低层版本相同；同一目标执行器不为任务调WBC。
后续在target稳定的Recovery Oracle上再做target-DAgger。新的FB13/BASE非零软意图训练须另有非零合法BASE标签与观测、自己的stats/normalizer/masks/版本，不把source EEF10补3个零称为移动专家。
人类视频/WING作为后续固定机器人数据量的独立交互表征对照，不能直接生成未测的米制EEF13/BASE动作真值，当前GPU排队不启动视频模型。

### P7 研究和论文的有效证据标准

(1) 源域BC确有T1/T2操作行为（实际episode/失败阶段，不仅loss）。
(2) 相同VLA权重到目标WBC后能否执行，统计source-only。
(3) 随目标示范量增加的样本效率/迁移曲线，以及action-space DAgger是否超过同量普通BC。
(4) 若FlowDAgger加入，明确测试是否保留已有任务、反演误差和每次专家纠正代价；FlowDAgger不因冻结主干而必然优于普通BC。
(5) 如未来证明跨本体，需不同实体机器人/执行器及固定策略对照，不能仅用两个任务或固定/自由根称为严格embodiment-independent。
成功/有效科学任务FAIL/真实安全FAIL/工程未完成/未运行五类保留，归因字段不跨层偷换。

## 3. GPU排队期 Codex 可立即做的CPU工作包：顺序、合格收据

### Q0 | 单一owner与冻结输入 (P0 priority，前置)
- 读取START_HERE/AGENTS/统一入口、当前#11/#47、新E0采用与排队原件、TF-R3原生停止/owner lease；按现场实际状态登记，不把GitHub快照当实时。
- 不改/不取消队列中的670049.hopper-m-02，不重复qsub，E0本身不新增GPU/时间；不能占用正在RUNNING的source D0 physics及共享lease。若E0进入RUNNING，priority高于prep；只在其正常episode边界避让。
- 输出 PREP_INVENTORY.md：固定旧数据/model/processor与版本键，资源身份与有哪些CPU预备工作未做；不复制raw/重新封数据。

### Q1 | 两个动作ABI和数据血缘 (最高优先，可完全CPU)
- 检验原v3.0原stats/EEF10 D0/S1的编码/解码与H40和prefix8时钟；再写target EEF13未来schema与BASE absent-vs-zero及维度mask测试。不能更改E0原数值路径。
- 保存状态/原source数据小fixture做input/output roundtrip：6D旋转正交性、mm/m、gripper正反方向、relative-to-absolute恰好一次、20Hz实际command/observation对齐、H40跨prefix8。
- 产物 ACTION_CONTRACT_MATRIX.md，tests/test_pi0294_dagger_action_contract.py（可按现有test位置适配），清晰标PASS_CPU_FIXTURE而非模型PASS。至少含5条负例：单位、TCP、夹爪反向、重复relative、BASE伪零。

### Q2 | Recovery Oracle只做离线接口，不跑新物理
- 在原source已保存标准状态和已允许的离线clone状态上调用既有专家/Task Planner包装接口。先检查可否从已有状态给出完整安全H40和INVALID原因；优先只读source D0历史，不改S1安全逻辑；没有现成source expert query函数则做adapter和mock fixture，不启动数值搜索的大批量任务。
- 测试source标准输入/合法小偏差/受限状态/不可恢复情况与未来标签的obs timestamp；人为构造的偏差状态必须标SYNTHETIC_CPU，不当作on-policy DAgger。
- 输出 ORACLE_QUERY_CONTRACT.md + ORACLE_FIXTURE_RECEIPT.json，注明valid/invalid覆盖与NOT_PHYSICALLY_EXECUTED，不声称已证明在学生轨迹上完成恢复。
- target fullbody沿TF-R3现有owner单独推进，本prep不激活隔离版本或解除native stop。

### Q3 | DAgger记录与干预框架：mock/saved-state CI
- 声明统一记录：episode_id、domain、task、split、actual_observation_id/time、measured_EEF/finger/BASE、RGB hash、language、student_H40+noiseID、oracle_H40+validity/mask、safety gate、executor {student,expert,none}、installed_ref_id、actual8prefix/timestamps、task outcome、privileged_truth_audit_only。
- 构造纯CPU的step/chunk状态机测试（fake model/fake expert/fake env；禁止MuJoCo mj_step/render），验证学生访问/专家查询同状态、expert拒绝/接管、8点执行、H40不足的mask、数据隔离、重复chunk拒绝、边界早停、failure-bank不进success-bank。
- 产物 DAGGER_RECORD_SCHEMA.json + DAGGER_COLLECTOR_CPU_FIXTURE.md + 定向测试；不声称DAgger真实样本已经采到。

### Q4 | FlowDAgger兼容性审计及CPU-only mock
- 从官方固定3426f2465...只读小源码，不将LIBERO 7D拷贝进生产；列出encoder/processor/inverter/forward sampler/steering/接管handler映射表。遵守官方MIT声明，外部代码如引入需许可和commit引用。
- 核查我们真实M2048 model config、processor action keys/normalization、H40有效维与内部padded tensor。构造小模型或无模型依赖的mock forward/inverse数值fixture，不访问H200、下载新weights或把CPU shape pass当真实Flow inversion PASS。
- 标出FlowDAgger官方默认100% expert-chunk导致缺on-policy student states；提出具体测试的混合/查询设计，但只作为后续科学提案，不在本包确定并运行新的干预率。
- 产物 FLOWDAGGER_PORTABILITY_AUDIT.md、MOCK_INVERSION_ROUNDTRIP.json（确实跑mock才标PASS）、RUN_WITH_REAL_M2048_LATER.md。明确正式model-space invert、steering训练、实际closed-loop均NOT_RUN。

### Q5 | 数据集/评估导出合同及E0结果后路由
- 参考原source v3.0 TRAIN/DEV，无重抽scene、无重新统计。编写对以后DAgger buffer的TRAIN/DEV family隔离、状态/动作来源类型、expert-only vs actual-executed、全事件窗口/不足尾窗mask、训练输入不得含privileged object/phase。
- 编写E0结果决策表：TRAIN和DEV均有技能→source-to-target; TRAIN可拟合而DEV差→data/vision/language; TRAIN弱→小样本拟合/codec；安全/接口损坏→先修受影响链。即使E0是有效0/12，也不能造E1已跑。
- 输出 DAGGER_DATASET_AND_SPLIT_SPEC.md、E0_TO_DAGGER_DECISION.md。不得写真实stats或empty success-bank PASS。

### Q6 | 最终CPU交付和handoff
- 在原repo现有测试位置做新增最小CI（不改旧fixture来让测试绿），所有实际mock/saved-state结果区分PASS/FAIL/NOT_RUN。
- 交付reports/PI0294/design/gpu-queue-dagger-prep-v1/下的 PREP_FINAL_REPORT.md 与证据清单，只报告本次新增代码/测试、固定SHA、源码/模型适配界面、阻断和下一决策；不要再重复数千旧归档成员回读。
- 更新#11科学与#47工程，全部只发布脱敏小文件与hash。原始RGB、模型weights、私有WBC/路径、凭据、未公开视频永远不推Git。
- E0若在期间出实际科学结果，另开只读结果补丁或下一版计划，不热改原E0运行代码/既有评估分母。

## 4. CPU准备范围、资源、所有者和硬边界

该请求直接要求Codex做后续准备，可立刻做仓库审阅、脱敏文档、接口代码以及已有本地环境中的有界**非物理CPU unit/fixture**。发布文件本身并不创建服务器任务、自动唤醒Codex或增加Hopper/CoreLab配额。现役唯一owner先检查项目现场CPU规则，通常本地准备上限建议8 core-hours、4并发核、2GiB新产物；若现场现有资源许可更小取小，不能占E0正在执行的计算份额。Q0到Q6可并行不需要外部网络服务的内容；任何新仿真积分、神经模型加载、optimizer更新、GPU/PBS、长期daemon/cron、下载大数据或目标Teacher rollout一律需后续独立科学授权。

严格保持：source v3.0/v3.1内容与stats、M2048、原E0 12条件/权重/采样、source D0/S1保护数值、TF-R3失败/终局、冻结WBC/腿政策/机械和评分阈值。禁止test-double PASS覆盖真实质量FAIL。只允许新增prep文件/可选版本化CPU库和不影响runtime的测试；在shared checkout遵守单一mutation owner，不force push或热改运行capsule。

初始静态prep成功标PREP_CPU_READY_NOT_DAGGER_EXECUTED；E0结果、源DAgger on-policy、目标Teacher成功、target fullbody适配、FlowDAgger真反演/训练分别标NOT_RUN直到真凭证。若需要GPU/物理/新Teacher科学条件，只在最终报告合并提出下一项决策和已核算增量，不能从本文继承授权。

## 5. 最终给研究负责人看的整体推进门

当前：E0已在队列；Codex CPU Q0→Q6并行完成。
E0真实完成：决定是否需要更多source BC或已有source策略可迁移；不以“train0成功”阻断DEV8。
source Recovery Oracle从有效学生偏离状态成功标注：才可能建立真正DAgger收集计划。
target Teacher在冻结WBC上完整物体成功并从有限偏差状态可恢复：才准许把target-DAgger称为可行，不能用single-pad接触替代。
Source-only vs target少样本 vs DAgger vs FlowDAgger：固定同一评估、示范预算、冻结WBC，逐项改变一个研究因素；最终主张以真实任务分母和样本效率支撑。

源DAgger或FlowDAgger是否值得上线并不妨碍当前E0 GPU排队任务；本预备包避免两线在同一资源/科研条件上相互阻塞。
