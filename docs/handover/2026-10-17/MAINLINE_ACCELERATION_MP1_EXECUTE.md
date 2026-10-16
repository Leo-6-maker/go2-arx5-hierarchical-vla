# Codex接手：MP1追加预算，连续推进科学主线

**用户已明确授权更多预算并要求跑通主线。MP1是有界执行GO，不是等待第二次常规确认的建议。** 当前对话只发布文件，服务器真实运行必须另有原生收据。现役唯一local-PC owner核实站点许可、run/job/lease后登记一次，随后自主执行，不为正常工程步骤反复询问。

## 先读：防止同时出现的E2/MP1重复执行

固定回读发现同一用户请求期间并发新增了E2协议和长期路线。它们不能与MP1叠加出两份预算、两个owner或两套相同实验。**本次统一以MP1为新增资源口径，48 H200 GPUh、32 A40 GPUh只计算一次，不是再加E2的24/12小时。** 其他作者文件保留为历史/方法参考，不覆盖、不删除。

从本handoff同一固定commit按顺序读取：
1. `reports/PI0294/design/MP1_SINGLE_BUDGET_PRECEDENCE_V1.json` —— 并发计划优先级与单一预算。
2. `docs/protocols/PI0294_MAINLINE_ACCELERATION_MP1_20261017.md` —— 完整科研协议。
3. `reports/PI0294/design/MAINLINE_ACCELERATION_MP1_20261017.json` —— 阶段、数据、预算与停止规则。

若E2已在文件发布期间实际采用/运行，先保留其原记录和健康任务到正常边界，再核对同一请求的used/reserved纳入同一个新总包，不能追溯改科学身份或热换配方。未来新派发统一使用MP1或其明确登记的E2别名，不建第二份可花余额。旧E0/E1/既有Vanda预算另行保留，不重复计算既有预留。原生安全STOP绝不因本协调表解除。

## 1. 承接真实结果，不要从零重来

已审计M1-1024-CANONICAL在source TRAIN/F0/000/T1自主抓放32秒SUCCESS：16000积分、641双RGB，原S1拒绝0、native安全0、Teacher接管OFF，抬升83.536mm、水平搬运176.906mm、41/41保持。证据为审核基线8fdb2a179f5db88c294fea4182c2d177a2e2801a下`reports/PI0294/20261016T150228Z-localpc-e1/M1_1024_FIRST_T1_INDEPENDENT_AUDIT.json`。

这不证明双任务、DEV或Go2/WBC成功。同版本F0/000/T2已有效S1拒绝，剩余面板/D/M2按新实际收据更新。C0有833个完整H40窗口、来自2个E0学生episode，只有T1 Recovery合格；它不是833条独立示范，也不是新M1在线DAgger。旧E0负结果、无效fork、不同推理精度、一次scope deviation和TF-R3安全STOP保持。

## 2. 登记一次真实状态和新增资源

读AGENTS/START_HERE/统一入口及最新#11/#47。核实Hopper E1、Vanda1462438/1462451/1462595是否仍排队/运行/终态，不照抄历史快照。登记`MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json`，包含实际T0、旧used/reserved、MP1新增余额、E2别名处理、site许可、各lane固定输入和资源。

MP1：新增48 H200h/32 A40h，各最多1卡并发、各12个新GPU PBS、单job<=12h及站点限额；CoreLab320/Vanda240/Hopper160/local24 core-hours；source启动400、target启动320（含真实fork），所有积分含shadow<=6400万，新增optimizer更新合计<=24576。空间/网络和更小站点quota按机器卡。

实际T0+144h停止新排队和实质候选，+160h停止新增实验/训练块，+168h硬收尾；首次绑定在发布后48h内。旧已运行任务按原capsule自然完成，不kill或热改；需新增资源完成剩余工作就建MP1子run，只做剩余更新，原统计不重置。同条件不双服务器竞速挑结果。

更新START_HERE顶部和`CURRENT_SCIENCE_STATUS.md`，明确真实E1首成功、当前data/model与Vanda进度；旧文字/链接保留为历史。不要新建另一个统一入口。

## 3. 立即执行的三条并行线

### Hopper：已有E1收口 -> shared source policy -> 纠正对照 -> target适配

先让现役规范M1@1024面板自然完成；已有成功/有效负例不重跑。D/M2进行中则按固定配方与真实optimizer/scheduler/RNG/cursor续接；未开始且数据资格成立，完成原BC512/CORR512。未完成行保留原缺口，新结果另行命名。

再从固定M1@1024父权重做S-BC2048（原v3.0 TRAIN126）及条件式S-CORR2048，同预算/原FM/EEF10/冻结VLM。任务先1:1，T1独有纠正只在T1内部混合。原统计不换、v3.1本轮只库存审计；CANONICAL推理上下文显式控制，训练全局TF32不能污染评价。@512/@2048固定检查，最多一个证据触发的事件采样分支，不无限扫超参。详细LR/batch/采样和父子身份按协议。

目标数据5/20条每任务实际可用后，各从同一source父模型1024更新；50条及学习曲线支持才加2048。边生产vN+1边读取冻结vN，单任务先有数据就先适配。BASE13与FlowDAgger均为有条件后继，不抢主线资源。

### CoreLab：复用T1/C0 -> 真实source DAgger -> 数据独立消费

不重做已合格833窗口管线。T2 Recovery最多两个有证据版本；专家可返回INVALID。最多两轮在线采集，每轮12个预定TRAIN episode、每任务最多6个。学生原S1合法才执行；首次拒绝时由合格专家同状态接管，专家也必须过原S1。记录最初student anchor与后续expert状态、实际提交/安装/ACK、真实新观测和局部结果。全专家不叫on-policy，学生失败动作不当正确标签，8点不得补零造H40。

每轮封存数据后同父BC-only512/CORR512，并自主评价、Teacher接管OFF。无合格纠正的任务跳过，不阻断另一任务和普通BC。source1->2->4、target1->2只在完整隔离与CPU余量合格后增加；采集、审计、导出并行。

### Vanda：完成现有建设 -> 安全原生资格 -> 同节点闭环与迁移

继续原包/容器/编译及已有GPU资格作业，不另建重复任务。硬件/EGL、固定模型推理、原生WBC启动/时钟/参考/RGB逐级通过后连接新观测->H40->原安全门->prefix8/ACK->新观测。单A40训练与渲染推理默认分时。跨GPU差异需要记录，不要求bitwise一致，也不混为同一数值后端结果。

旧target安全STOP不解除。只有证据定位到自有启动/事务/参考时序/任务连接/夹爪请求或codec错误，完成最小修订/CPU和隔离预测后，才在新TASK_INTERFACE_REQUALIFICATION scope做原生资格。最多3版本、每版2个预定条件；不改WBC/腿策略/plant/夹爪PD/力限/限位/碰撞/评分。同一未解决根因第二次真实安全违规，target物理本周期停止；source独立研究继续，不换主机绕过。

安全target执行器成立后，一个source任务已经学会即可做该任务迁移，不等双任务或全部DEV。BASE preference absent不等于零填监督。目标数据每任务50 TRAIN+10 DEV为目标、生产最多160预定尝试，失败保留；可用合法Teacher、明确标记的成功policy rollout或人工可用时高层仿真示范，均需target实际安全任务成功。slow-seed 64/96/128秒独立标记，最终32秒评价不改。

## 4. 最后必须做独立确认，而非只训练和存checkpoint

E0 DEV已暴露，属于development，不生产训练标签。SOURCE_CONFIRM和TARGET_CONFIRM各预封10个新scene-group×2任务×2采样seed=40条件/模型，与训练、纠正、兄弟场景去重。source选一个共享candidate确认一次；target选一个适配candidate与source-only做同40条件配对，共80尝试。每任务期望14/20完整安全成功是研发目标，不是承诺或真机部署许可。未达到如实记录，不换seed重来；本轮不从确认结果回流训练或Teacher调参。

FlowDAgger只在普通纠正对照成立、评价资源有预留后启用，最多2GPUh，先32个TRAIN anchor真实反演，不合格即停；合格后最多512 steering更新。BASE13只有真实非零BASE标签时做独立<=2048更新，不伪造旧EEF10能力。不开RL、WING式预训练、全量VLM或新基座。

## 5. 工程自治与收尾

同范围构建、路径、IPC、认证生命周期、持久writer、atomic写入和已完成ID只读重取可自主修复并留测试，不逐项问GO。未知model/physical effect不重发。一个shared checkout一个mutation owner；所有运行capsule固定。正常S1拒绝继续其他预定实验，真实安全错误停止相应domain。

两端各保留4GPUh最终评价/收尾，CPU至少15%收尾。预算为上限，提前达标不强行花完。每6个活跃工作小时和重要原生状态变化汇总一次实测进度/余额/下一行动，不反复三轮只报blocked，不创建永久daemon。

最终发布真实source/target任务矩阵、数据revision/独立episode/纠正anchor、模型/processor/stats、无专家接管评价、视频逻辑索引和各端资源终态。Git仅自研源码/脱敏收据，raw/权重/WBC/凭据仍私有。登记domain限定的研究candidate，真实机器人deployment始终NONE。

**直接采用单一MP1资源包并连续执行。不要再生成一个并行E3预算；不要因新计划重做已有成功，也不要把更多预算理解为清除旧安全停止。**
