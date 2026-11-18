# SLA D2：Teacher完整路径、原生吞吐与目标适配长程计划

日期：2026-11-18（Asia/Singapore）。状态：**PLAN_ONLY / NOT_EXECUTED**。本计划接续PR409截至 `b29bd2badc976e2288f9d009e4bf14cc41b769af` 的终态，替代PR410中“继续原Teacher48未启动后缀”的未来派发要求。旧实绩、旧失败、旧科学门不变。

配套：[机器计划](SLA_D2_TEACHER_AND_PARALLEL_20261118.json) · [证据与问题分析](../../reports/WM02/sla-d2-audit-20261118/ANALYSIS.md) · [Codex启动prompt](../handover/2026-11-18/START_SLA_D2_AUTONOMOUS.md)。

## 一、目标与当前决策

本轮首先建立**可重复完成全部HOLD/PLACE的目标Teacher和可靠的双world执行能力**，再用完整目标数据检验源初始化能否帮助SLA适配Go2+ARX5。本轮不扩大同分布训练步数、不启动RealMix、不以源离线误差替代目标闭环。

当前原Teacher18/48已终止：9完整成功、8接触STOP、1预算耗尽、0技术失败；526198科学tick。F1红物体左放0/3、剩余1，最高1<2，原剩余30条（含#24）不得继续派发。M原更新、访问、生成、学生请求均0。9条正例属于不同Teacher版本的开发累计，不是冻结策略50%成功率证明。

新工作命名：

| 阶段 | 职责 | 正式能力结论 |
|---|---|---|
| A0 | 存量证据归因、现场冷态/资源核验 | 无新任务成功 |
| E | 性能计时、事件就绪、world隔离与IPC | 有限工程资格 |
| TEACHER_DEVELOPMENT24 | 新人口上修Teacher完整路径 | 开发证据，不能回填资格 |
| TEACHER_TRAIN_QUAL48 | 同一冻结Teacher/后端的新48 | 目标TRAIN建设最低覆盖 |
| M2 | 同目标数据的BASE/SRC适配比较 | 目标闭环screen/confirm/DEV/TEST |

A0、离线几何开发与性能工程可并行；实际吞吐测量须独占预登记资源时间窗或使用不重叠资源。Teacher开发无需等event-ready全部成功，可继续使用legacy单world。正式QUAL48首发前冻结最终backend。

## 二、立即执行的A0：用已有证据减少盲目原生试错

1. 回读PR409最新head、D0/D1终态与原始本地receipt，若前进则列出差异；只读核实Core当前owner、进程/cgroup、lease、实际allocation、磁盘和内存。旧CLOSED不重开，健康他人/本项目任务不抢占。
2. 用现有第12、13、15、18条封存qpos/qvel、mount、TCP/object transform和参考链，构建“状态→名义H40→submitted→accepted→selected或UNKNOWN→actual”的逐tick表。
3. 复核第18条LOWER净空收缩、第15条DESCEND pad/table路径，以及F2原红物体腕部冲突。完整SO(3)、双pad、腕、全臂、目标/非目标物体、桌面都在几何检查范围。
4. 用完整第18条长历史离线回放，测accepted_boundary、identity lookup、JSON/归档随atom数量增长。离线重放不能计新增原生成功/模型生成。
5. selected仍UNKNOWN时不强行归因WBC、不伪造日志，也不阻止独立Teacher几何修复。D0的0/34不支持把新增历史输入作为当前首要工作。

交付 `D2_CAUSAL_AUDIT_ACTUAL.md/json`，包含具体缺陷、可检验预测、实现改动范围；不能只是再写一轮计划。

## 三、Teacher修复：从抓取构型开始规划完整放置

### 3.1 修改范围

只修改Teacher目标、候选选择、阶段逻辑、可行性预检与只读证据记录。原WBC/legs/MPC/solver/gains、2ms积分、50ms动作、H40/P10、TTL、原接触STOP和评分保持。

主要改动：

- 抓取前联合筛选抓取位置/朝向、搬运姿态、落点和退出方向；候选需覆盖接近→抓持→抬升→搬运→下放→打开→撤离，不仅检查搬运段。
- 几何变换从实际TCP到物体抓持变换出发；保留完整SO(3)与mount姿态，不把绕物体中心的静态远端旋转当作真实全臂运动。
- 在每次LOWER查询、授予下一P10前，依据当前实际状态重新检查剩余下放、打开和撤离路径；按阶段检查合法目标接触，同时保留腕/pad/桌面/非目标物体原安全规则。
- 使用显式正余量和提前减速/安全中间目标。余量应包含下一反馈区间、连续参考过滤器的剩余位移、实测误差敏感性与几何采样误差。不能把少量数据的最大误差当作全局安全界；不能仅把旧>0改成未经验证的常数。
- 通过Teacher生成连续合法参考完成重规划，不在SLA输出端clip/smooth/retime；不能为了避免STOP而过滤碰撞观测。若已没有合法动作或余时不足，按预登记安全终止/任务失败保留分母。
- 每个PLACE在出发前估计完整时间，包括转腕、额外抬升、进入LOWER、OPEN至少2s、撤离/稳定。HOLD仍60s、PLACE仍90s，不自动延长。优先减少不必要的高度、等待和转向，不能通过加速WBC解决。
- 新数据逐条保留完整名义H40计划与真实消费P10的区别；未执行尾部不能标成已经执行的命令。失败前缀不自动成为成功BC标签。

### 3.2 有界开发人口

最多24次新科学开发请求，分2组，每组12（F1/F2各一个新场景簇×6指令）。第2组使用另一组新场景。最多两个实质Teacher候选版本，只有上组全部终态、服务冷态后才可切版本；无需为了使用第二组而制造补丁。

默认先完成第一组12。若同一签名已达到12格各1次完整成功，可直接冻结，不为了耗尽预算重复。否则根据已证实机制作至多一次实质修复，再执行第二组；未修改签名时，两组可合并判断，全部失败保留。**不同版本或backend的成功格不能拼接。**

开发晋级门：同一最终Teacher/几何/评分/原生执行协议签名在全部12格各至少1次完整动态成功，并通过G7。该门仅是进入更完整资格人口的最低准备，不是总体成功率或安全概率证明。未达到则交付已定位失败与剩余问题，M2不启动；不能用更改版本号、补相同seed或自动加第三批取得通过。

旧18不重跑；开发24不加入首轮M2新TRAIN，不进入QUAL48分母。已发布9个正例可作离线回归，不计新候选资格。

## 四、性能工程：先定位，再改等待

### 4.1 计时内容

当前端到端RTF约0.13。第18条1702任务atom+128warmup对应274.5s固定150ms处理窗口，占714.09s的38.44%，并非全是空等；完整H40固定根shadow证书等开销仍待计时。

新增只读计时：startup、warmup、advance/quota到pause ACK、post-atom callbacks、prepare_install/证书、reference/solver witness、几何、render、图像队列背压、score、seal/audit。报告次数、总exclusive/inclusive wall、描述性分位数及超时。统计完整cgroup的sim/solver/controller/planner/Python CPU核时、PSS/RSS、I/O、线程、tmpfs及EGL/GPU占用；不能用父Python process_time估CPU余量。

当前policy hash缓存、增量PlannerEventLog、常驻相机/模型及16帧异步npz已存在。优先补真正缺失的ROS证据/参考identity增量索引。不得删安全门、降低H40认证分辨率、将shadow成本未经计时说成主因。

### 4.2 事件就绪候选

借鉴native_collection_performance_v1.settle_atom，另建移动版合同，保留legacy固定150ms路径。提前返回必须证明精确native tick/pause ACK、当前安装参考身份、gripper实际消费、观察/EE freshness和实际控制器消费条件就绪；仅有accepted/solver匹配不足以声称最终控制器已选中。

如果无法得到足够消费证据，events候选不晋级，Teacher和legacy双world仍可推进。不能放宽旧construction角色限制或直接删spin。

### 4.3 固定工程矩阵与预算

先注册4个不接触物体的短工程场景（F1/F2×稳定参考/小幅连续EEF运动），每场景2次，共8卡。每卡初始化≤5703tick、任务≤10s，合计≤10703tick。明确它只覆盖消息、时钟、H40、图像与资源隔离；完整接触和释放覆盖由新Teacher开发/资格完成。

| profile | 卡数 | 用途 |
|---|---:|---|
| legacy N1 | 8 | 冻结基线 |
| event-ready N1 | 8 | 等待机制资格 |
| selected backend N2 | 8 | events通过用events，否则独立命名legacy N2 |
| 条件selected backend N4 | 8 | 工程扩展，不用于本轮正式科学人口 |

legacy/events单world按预登记交错顺序；N2/N4第二轮轮换场景与worker/domain绑定。重复是工程对照，所有分母和失败保留，不是科学失败重投。硬门包括身份隔离、原tick/配额/H40/P10/G7、freshness、无非预期STOP/技术失败、预期STOP后零积分。状态差异容差由执行端依据冻结模型单位和legacy重复变异，在查看候选输出前登记；没有足够依据就不宣称等价，不追求无依据的bitwise一致。

另登记**2组双world STOP隔离试验，共4个原生请求**：一world按固定task tick发项目STOP，另一world按自己的配额继续至登记终点；第二组交换角色。预期STOP是工程测试结果，不是物理任务成功。普通科学接触STOP只终止所属world；隔离破坏/STOP后积分等工程故障停止全部受影响自有消费者。

**全部性能、接口、隔离和技术效应合计最多64个原生请求**，其中比较≤32、STOP隔离4，其余≤28覆盖实际需要的接口及技术成本，均≤10703tick。接口不能凭旧不同domain/source的通过移植。执行前形成具体manifest；若N4接口需求装不进该有限范围，直接保留N2，不新增临时配额系统、不挤占科学24/48。没有发生的条件不计作完成。

短probe初始化比10s任务还长，必须分开端到端吞吐和任务稳态RTF；不以短probe证明完整抓放成功率等价，少量样本的p99只描述，不用于强统计结论。

工程收益目标：N2批次吞吐≥对应N1的1.5倍；N4≥N2的1.2倍。它们决定是否值得更多资源，**不是安全或科学真假门**。语义合格但收益不足可保留少worker；不换场景追逐提速。所有批次同总任务工作量，墙钟从首worker启动到最后封存，不将子进程wall相加。另报完整成功/G7合格且满足覆盖的episode每小时、CPU核时/合格episode。

## 五、多world实现与学生队列

### 5.1 原生world

复用旧corelab_world_owner_v3与target_parallel的隔离经验，建立移动双物体的新协议。每world有独立OS进程树、ROS_DOMAIN_ID、真实owner锁、CPU集合、输出/控制/tmpfs/cache、quota/ACK/STOP和trace；协调者只拥有其batch，不删除旧单world锁校验。模型等不可变资产可共享。

以实际spec、sched_getaffinity、cgroup cpuset/cpu.max交集和其他owner占用确定资源，不写死“机器还有几个空核”。旧环境出现过注册CPU与service affinity不一致。原OMP/BLAS线程控制可复用，但不等于原生求解器只有单线程。每world8+768MiB tmpfs设计量与实际进程内存、磁盘及EGL占用分别核查。

首轮正式科学人口最多2个在途；N4只做工程资格。不得直接切8world，不迁移物理引擎作为本轮捷径，也不复用未被完整重置的有状态控制器跨episode。

### 5.2 学生模型

现有REQUEST.json单槽必须先改为有界请求队列，唯一身份包含environment_id、episode_id、capture_id、native_step、request_id、model/package SHA；一请求一个不可串流的响应。支持确定性去重、公平排队、取消、旧响应拒绝、超时后不重复生成。

初版用**单个常驻GPU worker、batch1顺序推理**，各world在自己的暂停点等待；等待期间不得发额外tick、复用前一个响应或强制全局物理栅栏。queue wall timeout、model warmup和计数对两臂一致，原原生TTL不改。真实load/restore与必要生成检查纳入原M离线1600，不另外隐匿。动态batch2/4、多个模型副本及新GPU仿真后端不在本轮默认范围。

## 六、冻结QUAL48与数据准入

1. 冻结同一Teacher与backend后建立8个新TRAIN场景簇（F1/F2各4）×6指令=48，均与开发、模型screen/confirm/DEV/TEST及历史TEST隔离。所有人口在模型训练前冻结；不根据学生表现选数据。
2. 每格至少2/4完整成功，即至少24完整正例。这只是训练覆盖最低门，不声称可靠部署。逐次G7复核RGB/状态同钟、名义H40、实际P10、上次真实发出的gripper命令与完整物理成功。
3. 首发前固定顺序与world映射。每次派发在现有统一账本锁内检查s+r≥2；r包含该格全部尚未终态、已经在途和未派发条件。任何一格s+r<2触发不可逆停止新增派发。
4. 数学资格失去时，至多两个合法在途请求按原cap/STOP/截止完成并全部封存。隔离破坏或STOP后积分则立即停止受影响消费者；不能借在途继续。既不删不利结果，也不追加补seed。
5. 中途不改Teacher、几何、评分或等待协议。只有已事前资格的同一backend N1/N2调度可因资源降低并发；否则暂停至有限截止。不能按分数切legacy/events。
6. 新目标TRAIN只来自该冻结资格人口的完整合格回合；旧C20依原20%访问保留。TRAIN-only共同统计只拟合一次；旧18和开发24不入首轮新数据。实现新population reader/version，不修改原历史48已失败的准入结果。

## 七、条件M2：保留科学比较，迁移未用预算

启动前核实原M仍零效应，将其未使用预算唯一关联到M2，避免原M与M2同时消费；旧M状态不改成已经训练。S1000原端点复用，不重训S/T。

| 项目 | M2-BASE | M2-SRC |
|---|---|---|
| 初始化 | 官方SLA base | 同base＋S1000已审核20个共享LoRA数组 |
| 其余目标参数 | 67个数组初值完全相同 | 同左；不拷源IO/normalizer/optimizer |
| 目标更新/访问 | 1000 / 32000 | 相同 |
| 数据访问 | C20 6400＋新QUAL48合格数据25600 | 相同 |
| codec/统计/loss | 原G7、共同TRAIN统计、原V6 loss | 相同 |
| 学习率/噪声/dropout/采样/optimizer | 原V6匹配设置、fresh optimizer | 相同 |
| 正式端点 | 固定1000，不挑中间最佳 | 相同 |

20/80是旧目标/新目标访问比例，不是真机/仿真比例。M2-SRC额外源1000/32000历史成本另报；本比较匹配目标适配预算，不匹配总预训练算力。

离线/恢复门沿用V6；训练内生成合计8000、离线含queue/restore≤1600。固定screen48（每臂24），每臂≥18/24、每指令≥3/4、每family≥9/12、hard STOP=0、技术失败=0。按原排序选唯一候选，完全平手选BASE；confirm24同门，失败不切第二候选再刷。通过后DEV60，再一次TEST60；各≥48/60、每指令≥8/10、每family≥24/30、STOP/技术失败=0。DEV不过不开TEST。

两臂交错并平衡world/CPU slot、同queue/后端、同条件和噪声；报告实际初态差异，按场景簇与episode都统计。任务60回合来自10簇×6指令，不能当60个独立场景。原M不重启，旧TEST不复用。

源模块2×2诊断只继承PR410最多224次SOURCE_DEV生成的未用余额，计入原T/S离线1600。它不是Teacher或M2前置，不改S1000迁移端点；实际GPU空闲且机制问题明确时才执行。

## 八、预算、时间与自主终止

| 类别 | 条件上限 |
|---|---:|
| 工程含所有接口、STOP隔离、技术成本 | 64请求；每请求≤10703tick |
| 新Teacher开发 | 24请求 |
| 新Teacher冻结资格 | 48请求 |
| M2学生screen/confirm/DEV/TEST | 48+24+60+60=192请求 |
| 所有新原生请求总上界 | 328 |
| 保守原生tick总上界 | 14,070,584（科学均按PLACE上限估算，实际HOLD较低） |
| M2训练 | 复用未用2000更新、64000访问 |
| M2生成 | 训练内8000，离线≤1600，原生≤34560 |

以上是所有条件通过时的上界，不是要求跑满，不重置历史效应。技术失败、接口与部分积分均收费/计量，不能把批次数当原生请求数。

已知旧allocation截止2026-11-19 02:51:23.855488 SGT，必须现场核实实际授权截止，服务≤12h、campaign≤24h，取最早有效截止；不自动延期。按当前单world5–12min/完整episode量级，全部264条科学条件可能需二十多至五十多小时墙钟，且不含工程和训练，不能承诺本allocation全部完成；这是历史耗时范围的粗估，学生推理成本还不同。

优先完成：A0/Teacher实质修复 → 同时推进双world → 最终候选开发通过 → QUAL48 → M2训练和已允许的筛查。N4和源模块诊断低优先级，不挤占关键路径。每批发车需给当前观测尾延迟、close/seal预留时间；不在截止前抢发不可收尾任务。期限到达封存真实进度与未启动后缀，不伪造“长期运行”或自动新lease。

## 九、验收交付与具体改动位置

交付最少四组文件：D2_CAUSAL_AUDIT、PERFORMANCE_PARALLEL、TEACHER_DEVELOPMENT、TEACHER_TRAIN_QUAL；条件到达才增加M2_TRAIN/SCREEN/CONFIRM/DEV/TEST_ACTUAL。每组md/json包含真实source、population、模型/数据身份、请求和tick、成功/失败/技术分母、首失败、G7、资源与关闭状态。未启动写0及原因，不用单元测试数量证明能力。

相关实施入口：

- `world_model/multitask_teacher_v6.py`、`multitask_carry_geometry_v6.py`、`mobile_continuous_teacher_v2.py`：在新版本中修完整Teacher路径。
- `scripts/wm02/mobile_boundary_runner_v1.py`、`world_model/native_collection_performance_v1.py`、`mobile_boundary_clock_v1.py`：新移动profile/事件就绪/增量索引。
- `world_model/corelab_world_owner_v3.py`、`target_parallel.py`、`target_sequential.py`、`native_live_control_v3.py`：复用真实world隔离，不削弱旧角色准入。
- `scripts/wm02/sim_only_campaign_v51.py`、`multitask_stage_cards_v6.py`：新移动并行父子合同，科学STOP与工程故障分开。
- `world_model/sim_only_mobile_v51.py`、`scripts/sla/hybrid_ipc_worker_v3.py`：学生有界身份队列。
- `sla_adapter/multitask_data_v6.py`、`multitask_initialization_v6.py`：新人口 reader和未用预算迁移；保留原G7/初始化语义。

针对改动做有意义的单元/接口检查：LOWER余段检查、完整候选路径、world身份与STOP隔离、请求去重/错路由、futility含在途；不得将fixture作为原生资格。普通文档发布不等于任何runtime已通过。无需新调度系统或多层审批框架。

## 十、发布与执行边界

本次仅发布审计、计划、图表和启动prompt，未改runtime、未连接Core执行、未新增训练/生成/原生积分。执行者在实际有权限的工作环境内自主实施上述有界工作，不需逐项再次询问GO；真实资源/owner/截止无效则封存阻碍，不能伪造授权或绕过锁。

普通GitHub feature branch/PR发布到main；执行源码从固定PR409栈在独立worktree接续，不假定main包含该runtime，不对活跃胶囊pull/checkout/hotpatch。保持WBC、安全门、RealMix封锁、历史失败和TEST、真机禁令不变。

