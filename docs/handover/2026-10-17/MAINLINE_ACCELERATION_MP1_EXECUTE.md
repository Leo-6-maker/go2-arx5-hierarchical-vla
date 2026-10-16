# Codex接手：MP1追加预算，连续推进科学主线

**用户已明确授权更多预算并要求跑通主线。MP1是新的有界执行GO，不是还要用户再次确认的建议稿。** 当前对话只发布文件，不代表服务器任务已启动。由现役唯一local-PC owner核实站点许可、原生run/job/lease，登记一次MP1绑定后连续执行。不要转交另一个工作模式或重新要求用户批准每个正常步骤。

从本handoff同一固定commit读取：
1. `docs/protocols/PI0294_MAINLINE_ACCELERATION_MP1_20261017.md`
2. `reports/PI0294/design/MAINLINE_ACCELERATION_MP1_20261017.json`

## 第一项工作：承接真实进展，不能继续报告“一条policy成功都没有”

已审计的M1-1024-CANONICAL在source TRAIN/F0/000/T1自主抓放完整32秒成功：16000积分、641双RGB、原S1拒绝0/native安全0、Teacher接管OFF、抬升83.536mm、水平搬运176.906mm、41/41末段保持。固定证据在审核基线8fdb2a179f5db88c294fea4182c2d177a2e2801a下的`reports/PI0294/20261016T150228Z-localpc-e1/M1_1024_FIRST_T1_INDEPENDENT_AUDIT.json`。

这不是T1两个布局都通过，不是T2/DEV/Go2/WBC成功。同版本F0/000/T2已有效S1拒绝；其余面板和D/M2以实际新收据为准。C0已有833个有效H40窗口、2个E0学生episode来源，仅T1 Recovery合格；不是833条示范或新一轮在线DAgger。旧E0结果、精度偏差、scope deviation、无效恢复fork、TF-R3安全STOP和原账全部保留。

## 接手顺序

### 0. 登记真实状态和新增预算
读AGENTS/START_HERE/统一入口/#11/#47，检查实际Hopper E1与Vanda硬件/模型/构建作业。只读查询后列明每个任务是RUNNING、QUEUED、CLOSED还是未知，不能把历史Q快照当实时。

创建`MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json`和`CURRENT_SCIENCE_STATUS.md`，登记实际T0、旧used/reserved、MP1新增额度与站点允许额度、父子run、固定输入、上下文与各lane资源。更新START_HERE最上方当前摘要，旧文字及链接保留为历史；不要重建一套入口。

MP1新增48 H200 GPUh、32 A40 GPUh；各最多1卡并发、各12次新GPU PBS，单job不超过12h和站点限额；CPU CoreLab320/Vanda240/Hopper160/local24 core-hours；source启动400、target启动320（含恢复fork），所有物理积分含shadow最多6400万；全新optimizer更新合计24576。详细空间/网络和收尾预留以机器卡为准。

T0+144h停止新排队提交与新实质候选，+160h停止新增实验/训练块，+168h硬收尾；首次绑定须在发布后48h内。旧job照旧账，后续超出旧范围的真实剩余工作归新的MP1子run，不追溯改账/重复计费/重置老时钟。健康在途job不可kill或热改；相同实验不可两边竞速挑结果。

### 1. 先收完E1的规范面板和M2
正在执行的M1@1024其余条件先自然完成；已成功或有效失败的条件不重跑。D/M2开始了就按已有冻结配方和真实cursor续接；未开始且C0或新C1条件成立，完成原BC512/CORR512及自主比较。缺监督的任务如实跳过，不造标签。精确resume需要optimizer/scheduler/RNG/sampler及pending update=0；weights-only不是精确恢复。

### 2. 并行两条研究线，不等目标Teacher才能训练
Hopper：从固定M1@1024父权重，原v3.0 TRAIN126做S-BC2048更新；纠正bank合格则同父同预算S-CORR。先任务1:1，只有T1纠正时在T1内部混合，避免吞掉T2。根据TRAIN证据最多再开S-EVENT一次，仅改变事件窗口采样，其他模型/优化/精度不变。原VLM冻结，原FM、EEF10、H40/prefix8与stats保留。按照协议固定@512/@2048检查，不密集挑checkpoint。

CoreLab：复用C0/T1，扩真实学生访问状态的Recovery和最多两轮source DAgger；T2 Oracle最多两个有证据版本。原S1合法学生prefix实际执行，首次拒绝且专家合法才接管。区分原学生anchor与后续expert状态，实际提交标签/ACK/数据reader完整。每轮BC-only512和CORR512同父对照，评估没有专家接管。Oracle无效不能阻断普通BC和其他合法任务。

### 3. Vanda继续原生闭环建设，不另起无期限移植项目
复用已上传冻结包和原构建记录；实际硬件/容器/编译->固定模型推理->原生启动/时钟/RGB/参考->同节点实际新观测闭环。一个平台不要求跨GPUbitwise相同，但模型/processor/stats/精度/差异必须明确。阶段工程成功不是物体任务成功；单A40默认训练与渲染推理分时。

旧target native STOP不因预算追加清除。只有私有证据定位到自有接入错误，并有版本化最小修订/CPU与隔离验证，才能进入MP1独立TASK_INTERFACE_REQUALIFICATION；最多3版、每版2个预定条件。允许修启动/事务、参考时序/插值、任务连接和夹爪请求，禁止改WBC/腿策略/plant/夹爪PD/力限/机械限位/碰撞和评分。同一未解决根因第二次真实安全违规，目标物理本周期停止；source独立工作继续。不能换主机绕过。

### 4. 有安全target执行器后，尽早做迁移和小样本适配
一个source任务已成立就可在其安全target开发场景做source-only，不等全部DEV或T2；记录BASE preference absent，不伪造旧EEF10的BASE3。

目标数据来源可为经target实际验证的源技能Teacher、source策略成功rollout（明确policy-generated标签）、人工可用时的高层仿真示范。最终都要真实物体成功和原生安全，不能重命名离线轨迹当target数据。

每任务目标50条TRAIN+10条DEV成功；5条即可固定target-5，20条固定target-20，Hopper边采边训冻结版本。生产最多160个预定尝试，失败全计且相同已知无进展不无限复制。target-5/20各从同一source父权重1024更新，target-50条件成立才2048；默认source/target 50:50，target任务均衡。原32秒目标评价不改；64/96/128秒slow-seed单独命名，不改旧FAIL。

### 5. 先固定候选，再做独立确认
E0 DEV已暴露，作为development，不生成训练标签。SOURCE_CONFIRM和TARGET_CONFIRM各封存10个新scene-group×2任务×2固定噪声seed=40条件/模型，与所有训练/恢复/兄弟场景隔离。

source先只选一个共享checkpoint做一次确认；target先DEV选择一个适配候选，再与source-only做相同40条件配对，合计80尝试。每任务期望14/20完整安全成功只是研发目标，没达到如实报告，不能改门/换seed重来。所有attempt、有效/工程缺失、scene-group依赖和不确定性保留。确认结果本轮不用于进一步训练或Teacher调参。

### 6. 可选工作不能抢主线
FB13仅在真正有非零BASE动作和可观测状态标签后做独立<=2048更新分支；缺条件就交付近场结果，不用零padding造能力。

FlowDAgger只在标准BC/纠正比较成立、主线评价资源已预留后启用，合计最多2GPUh，先32个TRAIN anchor真实反演；不合格就停。合格后最多512 steering更新。不能把mock或默认100%expert执行当真实on-policy改进。不开RL、视频world-model、全量VLM微调或换基座项目。

## 工程与报告约束

同scope工程修复、原子写入、完成ID只读重取、认证和资源调度无需再次routine GO；未知model/physical intent不重发，shared NFS只一个mutation owner，运行capsule不可变。source1->2->4、target1->2环境并行仅隔离合格后启用；采集写vN+1、训练读冻结vN，数据不边读边改。

两端各保留4GPUh用于最终评价/收尾，CPU至少15%收尾。预算是上限不是花完目标。每6个活跃工作小时及重要实际阶段改变汇总一次，遇阻断切可独立推进的lane，不连续三轮只重复“等待GO”。既有私有数据在本地不可变版本经独立reader通过即可消费，不为公共HF上传或全库重回读暂停学习。

最终交付固定模型/processor/stats/data版本，真实source/target双任务表、source-only/few-target/DAgger对照、纠正anchor和episode数量、视频逻辑索引、冻结WBC证据与完整资源终态。source成功和target成功分开；只准登记domain限定研究candidate，不自动真机部署。

**现在直接登记MP1，完成已开始的E1，随后按条件自动进入源域扩展、纠正、Vanda闭环和目标适配。不要只把本指令再改写成一份准备计划；也不要把新增预算理解为抹除旧安全停止或无限实验权限。**
