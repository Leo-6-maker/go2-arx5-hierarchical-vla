# PI0293 — 夜间多任务 ARX5 Policy：并行场景数据工厂、原生微调与闭环验收 V1

日期：2026-10-13，Asia/Singapore。状态：USER_REQUESTED_BOUNDED_OVERNIGHT_GO_FOR_EXISTING_LOCAL_PC_OWNER。
审阅起点：43784131815bc8691c3faa7f8d43bac330d67e71；集成分支 pi0267/executor-decoupled-recovery。
本文件是执行授权与设计，不是采集、训练或服务器部署已完成的收据。科学记录 #11，基础设施/路径协调 #47。

## 0. 科学目标与今晚的取舍

遵循 GROOT_OBJECT_MANIPULATION_MAINLINE_CHARTER_V1：学习视觉、语言和实际 EEF 条件下的物体操作行为；本团队不重新训练师兄 WBC。今晚的主要交付是**一份共享 GR00T checkpoint，在同一固定底座 ARX5 仿真中，依据语言完成两种不同物理技能、三个任务条件，并保存逐任务闭环结果**。

核心任务：PLACE_BLUE（抓起红色方块放入蓝区）、PLACE_YELLOW（抓起同一方块放入黄区）、PUSH_BLUE（沿桌面推入蓝区、不得抓起搬运）。前两者是同一技能的不同语言目标，不能独自称为两个不同操作技能；第三者提供真正不同的接触行为。三条指令必须在每个场景组的**完全相同初始物理状态与双相机画面**上分别采集和评价。一个 checkpoint、一个 processor、一个 EEF10 输出接口；不按任务路由三个模型或隐含选择三个 adapter。

数据量、CPU CI、目录整齐和 checkpoint 存在都不是最终目标。必须预留模型生成与真实新观测闭环时间。第一版多任务数据一旦封存立即启动训练，CoreLab 继续生产下一批；不得等待所有扩采或所有旧目录整理完再开始神经工作。

## 1. 最新起点与复用清单

- PI0292 bank v2：64 TRAIN/16 DEV，38464/9616 个完整 H40；私有 HF 固定 revision 6075aac0cdc44b8d252d7540efbac208701c30b3，repo leoyuer/pi0291-arx5-fixed-base。两端原始 RGB 与官方 CPU 数据接口已通过。复用原64 TRAIN，不重采、不改旧 PI0291 PARTIAL/原统计/失败记录。
- 官方 GR00T 源码：51d4c89f72fda44cbf77285c6a8114b52676b8a1；官方基模 nvidia/GR00T-N1.7-3B revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495；Cosmos 缓存来源 9ce19a195e423419c349abfc86fd07178b230561。实际本机版本/完整成员哈希仍须读取迁移后的 registry 验证，禁止回到已经删除的九月路径。
- 已有 scripts/iteration/ 的两个 spawned 合成 worker 原型与 decoded-RGB LRU 原型，尚未实现/验收 live physics adapter，未实测提速。接续56b914f8844e93985739ed26fb34952e55725699及其准备收据，不从零重写调度器，也不把 synthetic PASS 冒充并行物理 PASS。
- 已有 PI0292 真实数据 Trainer adapter、完整 checkpoint/恢复代码、observation-only 输入、ACK_CHAIN_V2 relay endpoints。9c00bffe9619a3213589b2e8add484bffd40cc8f 的 endpoints 只完成 CPU 组件准备，尚非实际双 SSH/物理资格；原训练 loader 的监督 action 不得进入在线请求。
- 最新 PHASE_STATUS：S3–S6仍未运行，GPU/model updates=0；CFP04-CF-007先前已核验但实时 account-manager 登录重新阻断。个人容量最近原始字节报告已恢复到约316 GiB可用，不再沿用下午83 GiB的旧结论。认证和存储必须分开判断。

## 2. 既有 PI0292、所有权和入口的衔接

一个 existing local-pc 集成/调度 owner；允许独立子任务分别写场景/采集、数据/训练、评价代码，但同一文件、run root、Git checkout只能一个写入owner。复用 ~/Projects/GR00T-WBC、AGENTS.md、统一入口规范和 ENTRY_POLICY，不再创建 Home pi* 根或顶层新仓库副本。每个 collect/train/eval/audit/pack 在副作用之前封存 RUN_MANIFEST、host profile、授权、输出根、source seal、数据/model/stats和lease。

这次用户明确允许新的多场景并行采集及多任务微调；因此 PI0292 的“本轮不扩任务/只单物理owner”等限制不适用于本 PI0293 新run。**不编辑旧 guard、旧账本或冻结运行源码来偷用新预算。**

采用时落 OLD_NEW_HANDSHAKE：
1. 若 PI0292 的合法训练已实际在运行，保持其原配方/截止/账本，不取消健康作业。CoreLab先做 PI0293 数据；PI0293神经任务排在本团队该作业之后，不能变成账号第三个GPU分配。
2. 若与当前已读收据一致、PI0292尚无运行/排队神经作业，则明确记录旧线 DATA_READY_NEURAL_NOT_RUN_HANDOFF_TO_PI0293。停止旧调度器的**新提交意图**并释放冲突调度lease，不动原数据/收据；新goal优先直接训练共享多任务模型。**完成旧单任务4096步不是本次前置条件。**
3. 旧已有 checkpoint 若存在，只登记为带其来源的诊断对照，不自动成为多任务训练母本；本次共享模型默认仍从上述 pristine 权重开始。
4. 新goal有自己的T0与额度，旧PI0292历史计数/截止不重置。T0后本团队仍在继续的旧神经作业资源，同时纳入本次“夜间团队神经资源总上限”的视图，以防叠加超用；账本原件仍各归各。
5. 当前同账号独立 pi05_relative/CFP04-CF-009 工作不取消、不借GPU、不用其产物；沿用明确的账号最多两份独立GPU分配。本团队PI0292/PI0293合计同时最多一个single-H200分配。

若统一workspace高级CLI尚未部署，可在本次首小时内给已有 enter/doctor 增加最小、已测试的manifest薄包装；不等待完成整个工作区重构。实际运行仍绑定独立源码快照，不能热 pull。

## 3. 固定夜间预算及暂停原则

首次真实采用记T0。deadline=min(T0+14小时, 2026-10-14T12:00:00+08:00)，最后60分钟停止新增任务、只封存/归档。若采用时已过绝对截止，不能自动解释成下一天。重连、认证等待和子agent重启不重置时钟。以下是上限，不是吞吐或完成保证：

| 项目 | 新任务上限 |
|---|---:|
| CoreLab live workers | 默认2，实际资格通过后最多4；每worker独立CPU集合，合计同时≤16个获准CPU核，包含审计/编码/管理进程及本团队并行旧科学占用 |
| CoreLab CPU累计 | 192 core-hours；所有worker/渲染线程/审计/视频/失败计入 |
| Hopper | 本团队1张H200，T0后团队累计allocated≤12 GPU-hours；本次最多4次qsub调用，失败也计；不借其它作业 |
| Hopper CPU / local-pc CPU | 96 / 12 core-hours；不用login节点作神经计算 |
| 正式多任务训练 | 一个配方、一条训练链，最多4096 optimizer steps，有效batch32 |
| 真实模型工程smoke | 最多16个实际optimizer updates，测试权重丢弃，不作为正式母本 |
| 新物理启动 | 最多384次，每次最多32sim秒/16000积分/641对观测；总≤6144000积分、246144对物理观测 |
| 物理配额分解 | commissioning24；pilot24；串并行8；正式数据180；relay6；模型开发12；旧任务回归8；确认108；工程中断诊断14 |
| 神经生成 | 离线≤2048次，闭环/工程≤9000次，其它≤32次，累计≤11080次；完整生成调用与内部denoising步数分别计数 |
| 新增峰值磁盘 | CoreLab160 GiB、Hopper180 GiB、local-pc8 GiB，包含原始/缓存/打包/上传并存 |
| 新删除/GPU仿真/WBC/真机 | 全为0；已有独立维护仍按其授权，不挪用此goal清理原件 |

Hopper给评价预留至少3小时allocated预算，给checkpoint封存预留实际测得所需时间；不能把12h全用来训练。物理工程中断重做只能消耗14次技术配额，保留原attempt并引用明确工程错误；禁止因质量FAIL重试同一正式layout、重新抽噪声或改阈值。所有计数、CPU和字节在开始前原子预留，完成后核销；UNKNOWN继续占用保守预留，不释放后再次启动。

## 4. 双线并行工作图与关键路径

| 时间带（相对T0，非必须等待的时间点） | CoreLab / AMD | Hopper / local-pc |
|---|---|---|
| 0–1h | 入口/迁移后资产核验；复用并行原型，开始串并行物理资格；构建双目标场景 | 检查有效project、本人quota及peer；依赖导入/数据复用；认证受阻只通知一次并继续独立工作 |
| 1–3h | 新teacher commissioning及4场景族pilot；冻结scene/teacher/scorer | 可以使用既有v2 TRAIN完成真实原生smoke/恢复；完成真实SSH消息通道与无物理端点检查 |
| 2–5h | 优先DEV12组与TRAIN-A24组，组内三指令公平轮转；边采边审计打包 | Drop-A合格且全成员到位立即正式训练；不等TRAIN-B |
| 5–9h | 继续TRAIN-B24组；修只影响下一数据版本的问题；预留闭环CPU | 同一配方到M2048；已冻结开发面板可做短闭环，随后同链继续至M4096 |
| 9–13h | 给真实relay/模型评价优先CPU，扩采在episode边界降并发或暂停 | 固定M4096评价；逐任务新场景闭环、语言反事实对照、旧任务回归；保存完整checkpoint与policy包 |
| 最后1h | 停止新episode，等待自己已启动的有限worker或记录不完整；归档所有失败 | 不再新qsub/新训练，保存resume、终局资源和回读报告 |

**认证阻断不再让整个goal停止。** qsub前必须有合法当前项目/调度资格；需要用户交互 amgr login 时，给一次明确提醒，不索要密码/token，不伪造认证或自动尝试别人的项目。每次有认证变化再重试；无变化时最多每45分钟只读检查一次，拒绝循环提交。若用户睡前未恢复认证，GPU目标可能无法完成，仍必须交付多场景数据和实际并行吞吐，而不能把重复准备文档当policy。

已经被调度器合法接纳且正在有效allocation内的作业，不因某次account-manager查询token过期就推断授权已撤销、反复登录或取消健康作业；以真实调度状态/账号/项目/节点/GPU绑定为准，任何真实撤销/终止立即服从。任何新的提交仍需当前权限，不能借此绕过认证。

## 5. 多任务、多场景的必做规范

详细数值设计见同目录 SCENE_DATASET_SPEC.json。该规范是新v3域，不修改v2场景或旧评分。

### 5.1 场景共同部分与三指令配对

固定底座原生六轴ARX5，原关节限位、有限力矩、重力、夹爪几何与摩擦基准保持。桌面top=0.44m，4cm红方块、两块16cm边长的蓝/黄无碰撞目标标记。已有body_rgb是world-fixed外部相机，wrist_rgb随腕部；不称为移动Go2机身相机。

同组重置的初始qpos/qvel/夹爪开度、物体位姿、两目标、照明、相机必须一致；只有语言和teacher内部任务选择不同。预先保存三任务首帧 decoded RGB SHA及状态哈希，必须相等。选中目标不许高亮、消失另一个目标、改变相机朝向/初始手位或在RGB叠文字。task_id/group_id/phase/seed/目标坐标只在审计侧，不能进入学生请求。

蓝/黄区位于源方块两侧；其颜色和north/south位置在每个TRAIN-A/B场景族内平衡。禁止蓝色永远只在一个方向。PLACE_BLUE和PUSH_BLUE在同场景对同目标要求不同接触行为，可区分单纯目标选择与技能条件化。

### 5.2 四个场景族

F0 NOMINAL：接近现有几何、固定相机/外观，但有两个目标和语言分歧。
F1 SPATIAL：更宽的物体x范围、双目标偏移、正反Y移动和物体yaw；仍在同一保守可达桌面。
F2 APPEARANCE：F1几何，改变中性桌面灰度与灯光强度；红方块及蓝/黄语义颜色不变，不增加可交互遮挡物。
F3 VIEWPOINT：F1几何，外部相机小幅三维位移并重新look-at，保存真实K/extrinsics；腕相机安装固定。

今晚不要求全新仿真引擎、不改成GPU vector simulator，不强行加入开门/插入/堆叠等高工程成本任务。若前三任务及交付提前完成，也不能使用剩余额度临时开未登记新任务。

### 5.3 Teacher与独立评分

抓放复用现有feedback teacher的接近、抓握验证、抬升、搬运、下降、释放、撤离；根据私有场景真值选指令对应目标。仍要求0.3s持续双侧抓握、至少2cm抬升、至少10cm搬运、撤离至少8cm，28–30s的41点保持均通过且全32s安全。终点选错颜色判失败。

推物teacher是本轮**必须实际实现的新技能**：关闭夹爪，移动到方块侧方合适接触面，基于真实方块运动持续修正平面推送参考，停止后撤离。不能把现有pick-place轨迹改语言就叫push；不能改物体qpos、weld、外力、摩擦或低层gain来保证成功。首次接触面、高度、有限速度须在commissioning验证。

PUSH_BLUE评分由保存的物体/接触高频证据复算：robot-object有效接触累计≥0.3s、平面搬运≥0.10m、全程物体底面离桌≤0.01m、不出现持续≥0.1s双指抓握、最终方块全角点在蓝区且41点保持、工具撤离≥0.08m、全程安全。不能以teacher.phase或teacher.done作为成功判据。底面高度由实际方块姿态和角点算，不用TCP高度代替。

新记录器应保存500Hz的物体姿态/必要接触flags/力摘要及安全flags，支持推物不抬升和接触历史独立审计；原v2数据字段和源不回写。所需数值必须来自同次物理积分/原生反馈，不事后用一个无warmstart的forward猜接触力。

Commissioning最多24次，允许反馈式任务teacher工程开发；最多两个teacher/scene/scorer封存版本。每个版本pilot为4个独立场景组×3任务=12次，全部通过才进入该版本正式采集。最多两轮pilot共24次；第二版用新的pilot ID，保留第一版结果。正式采集一旦启动，不再调teacher/几何/评分追分。若push无法资格：交付抓放语言条件化的数据/诊断，状态降为PLACEMENT_ONLY，不宣称两种技能政策，不能换个简单任务暗中计作第三项。

## 6. 数据规模、分集与两次封存

在任何新正式物理调用前，用固定SCENE_DATASET_SPEC中的整数哈希生成规则生成并封存72个场景组：TRAIN48（每族12）、DEV12（每族3）、CONFIRM12（每族3）。每组PB/PY/PUSH三条任务，共144 TRAIN、36 DEV；CONFIRM仅留评价，不进入采集训练池。

同组所有语言/任务、同一初态的渲染变体、任何派生窗口必须在同一split。先按group分集再切H40；不要随机拆窗口。与旧TRAIN/DEV/确认布局做精确初态及来源冲突检查；发现碰撞按事先生成的ID拒绝并记录，不临时抽更简单seed。公开seed保存十进制字符串，避免JS uint64精度损失。

**Drop-A（本夜训练集）**：原64 TRAIN + TRAIN组G000..G023三任务（72新episode），合格时136 TRAIN/81736 H40；新DEV12组36episode（21636 H40），另保留旧16 DEV作独立回归面板。全部首批计划必须有终态；新任务每种至少20/24 TRAIN与10/12 DEV准入，并且四场景族各至少4条新TRAIN/2条DEV/任务，才能称多场景训练bank合格。未达门不换split或补seed；可交部分数据和单任务诊断，不主张完整多任务资格。

**Drop-B（并行扩充资产）**：TRAIN组G024..G047再72新episode；本夜新正式采集上限180。全部成功时与旧64合计208 TRAIN/125008 H40。Drop-A模型训练期间B只能进入独立staging和新revision，不能热加入其DataLoader、更新stats或替换正在读取的manifest。B即使全完成也不必启动第二次训练；先保证M4096评价与交付。

调度交错DEV与TRAIN-A、三个任务和四场景族，不再先采完整大TRAIN队列再挤掉DEV。worker完成顺序不决定split、数据子集或采样权重。失败和工程不完整数据保留但不作为成功BC标签；quality rejected记录也进attempt分母。

## 7. 原始数据、编码和学习条件

每episode保留：真实同步双256×256 RGB，实际TCP/夹爪，语言全文，641观测/640条真正committed命令，planned H40与actual committed prefix分离，物理/接触/教师审计/原视频、source/scene/task/group/seed哈希。正常窗口601，H40时间偏移0..1.95s；仅prefix8执行到0.4sim秒。零命令终帧不补标签，不跨reset/abort/接管边界。

政策输入只含原双RGB、指令、实际EEF9+实际夹爪1。task_id、物体真值、目标坐标、phase、成功标志、未来动作、episode时钟不作为模型条件；合法时钟/标识只留协议封套。训练标签是绝对committed EEF10，官方processor按观测时实际TCP做一次SE(3)相对变换，夹爪绝对。10有效维/132padding与row-rot6d不变。

Drop-A仅在准入后从TRAIN拟合一次共享新statistics，按原合法anchor/H40规则；不是按任务分别归一化。与旧v2 stats另版本保存，DEV/CONFIRM不参与。保留TRAIN数值roundtrip门（XYZ≤1e-5m，SO3≤1e-4rad，gripper≤1e-5，SE3 matrix≤1e-8）；DEV preclip/反变换误差完整报告，不能用重拟合消除越界。新场景相机审计按真实投影/几何可见性检查，不能固定只识别旧单红方块画面而忽略两个目标。

大数据经私有HF新增PI0293前缀、固定revision、全成员hash、真实各任务/场景样本回读，READY最后写。原v2只按固定引用复用不重复全量上传/下载。Hopper必须拥有本次实际训练所需全部RGB/numeric，不以READY或NPZ存在冒充数据完整。输出dataset card列训练/未使用的Drop-B、失败、来源、示范数量和窗口数。

## 8. 真正的多进程采集与审计流水线

实现 live adapter 接入已有 scripts/iteration 原型，保留preparation-only原测试，新增独立PI0293 authority。不能只是移除 LIVE_ADAPTER_NOT_ADMITTED 拒绝而绕过资源保护。

Coordinator为唯一全局账本写者，原子预留episode ID、积分/帧/CPU/空间。每worker是spawn进程，自己构建MjModel/MjData/renderer/teacher/RNG，拥有唯一输出目录及局部journal；不能共享可变MjData/OpenGL上下文、原PI0292 PHYSICS_OWNER.lock或中央JSON。启动前mkdir(exclusive)、清晰READY/FAILED终态；崩溃保留in-flight reservation并先核对PID/start identity，不自动重放。

先2worker，每个2个不重叠获准CPU核；LP_NUM_THREADS=2，OMP/OPENBLAS/MKL=1，auditor另给CPU。修正旧collector总选affinity前4核的行为，使用coordinator分配的cpuset。所有本团队任务含审计/视频在核上限内；不把逻辑核数量当独占授权。

8次诊断物理配额：两个既有安全layout串行各一次；同两个layout双worker各一次；若双worker已过门，可用四worker各运行这两个layout的独立诊断副本，共4次。全部排除TRAIN/DEV/CONFIRM。比较相同source/seed下所有物理数值（atol1e-9、rtol1e-9）、首末与关键事件、decoded RGB像素、接触/评分和安全完全一致；身份/墙钟字段不比较。失败不放宽；退回串行继续有界采集。

实测wall/CPU、raw与准入episodes/hour、每阶段physics/IK/render/PNG/SHA/fsync/audit秒数、RSS和磁盘吞吐。只有双worker的准入吞吐实测≥串行1.3倍且资源合法才默认保留双worker；四worker须比双worker≥1.15倍才升级，否则选择更小并发。这些是本轮工程选型门，不承诺倍速。

物理worker只负责按原32s动力学推进与原始记录；完成后审计/导出/观看视频编码由有界队列独立处理，不占其物理槽。RGBA/PNG写入可有界异步，但队列满必须backpressure、不丢帧；结果最终manifest等所有写入完成后才能封存。审计积压最多8episode，超限暂停新发单。高频安全检查、2ms积分、20Hz观测、真实接触和641终帧均不为提速削减。

默认4080S本地SSD执行；AMD只做经profile资格的管理/索引/后处理，不假定有同样模型/CPU仿真环境。若要把物理worker分配AMD，必须在上述诊断配额内证明相同版本/字节/数值，且不改变总核与CPU预算；不是必做条件。

## 9. Hopper：尽快训练一条共享策略，不变成新的框架项目

### 9.1 最小神经资格

复用已核验本机N1.7/Cosmos和现有官方Trainer/observation-only adapter；只补新task bank与sampler，不重装整套CUDA、不换模型版本。缺依赖先判断是否本路径必须；确需安装的最小Python依赖可在隔离环境受控安装并记录锁定版本，不改全局环境/驱动；未经验证不能偷偷改变flash-attention或精度。

有合法allocation且Drop-A尚未到齐时，可以用原64TRAIN做最多16updates总量的隔离smoke：真实loss/梯度/冻结骨干不变、4步连续与2+2恢复；保存完整optimizer/scheduler/RNG/processor/stats/consumed cursor。若复用实际已通过且同代码语义的PI0292smoke可避免重做，记录适用范围。训练母本重新加载pristine。

Drop-A到齐后跑跨任务、两种worker配置的实际Trainer loader/张量对照及一次真实多任务batch forward/backward（仍计smoke16）。fixture通过不代替真实模型。

### 9.2 单一正式配方

初始化固定pristine，不继承历史Panda失败链或smoke权重；冻结视觉/语言骨干，训练官方projector与完整动作DiT/动作头所需官方可训练子模块。采用PI0292原生flags的明确展开值，包含tune_vlln等实际字段；保存named_parameters trainable清单、各模块参数数与来源，不能只看四个布尔旗标就声称冻结范围正确。不是全3B微调，不新增LoRA、目标坐标head、GR00T-RL或辅助几何loss。

masked flow-matching；AdamW lr1e-4，betas0.9/0.999，eps1e-8，wd1e-5；bf16；gradient clip1；cosine4096、warmup5%；有效batch32。正式最多4096步（131072样本曝光）。主训练前microbatch按2→4→8的顺序做有界吞吐/显存比较并保持accum16→8→4，选择显存余量≥15%、模型smoke等价的最快配置；测试一共仍计入smoke16上限。选好后不得中途变batch、lr、网络或输入。固定train seed29300，两个评价noise banks29301/29302。

每optimizer step32个样本固定构成为：旧v2 TRAIN4、PLACE_BLUE6、PLACE_YELLOW6、PUSH_BLUE16，物理技能总体抓放/推物1:1。各pool先均衡场景族和episode，再选合法anchor。阶段分层目标为接近20%、接触/抓取20%、搬运/持续推送25%、释放/撤离25%、稳定保持10%；用确定性weighted sampler与消费cursor实现，保存实际曝光。元数据只指导TRAIN采样，不作为模型条件。不存在的task-stage bucket必须在主训前报缺项并按预声明合法映射处理，不能静默回落全hold。新push的release bucket对应stop/withdraw。

新采样、场景和技能是本次联合的工程学习设计，不能将收益归因为单一采样算法。Drop-A统计不按重采样权重重拟合。无随机图像增强/state dropout=0延续已验收处理；F2/F3变化来自实际场景，不是偷偷变processor。

### 9.3 数据加速与checkpoint

接入已有numeric缓存/decoded-RGB有界LRU原型；首次读取验证内容hash，缓存键至少含dataset revision/episode/frame/camera/processor，输入像素不变。优先本机只读data staging、numeric mmap/LRU、persistent workers/prefetch/pinned memory；workers2起、最多4，证明样本身份/consumed cursor/RNG不因prefetch改变。不能把sha检查简单取消后依赖可变路径。

本夜**不实施VLM feature缓存、不换仿真引擎、不多GPU训练**。这些需要新的等价资格，不能拖慢第一版policy；尤其不能按图像缓存忽略语言条件。

保存完整M2048/M4096和必要恢复点；未完成4096则如实发布M<steps> DEVELOPMENT_ONLY，不能重命名为M4096。禁止训练时自动删除唯一完整恢复点。checkpoint落盘/上传后台CPU可与后续计算重叠，但必须不可变且完整成员回读；policy-only与完整Trainer分开标。存储需求按真实smoke checkpoint大小+临时并存+20GiB余量核算，不坚持未经测量的固定80GiB估计，也不无依据降低门；预算内可改变保存暂存布局，不能丢optimizer/RNG来冒充可恢复。

## 10. 尽早接入测试：不能训练完才发现无法下发动作

Hopper认证等待时即可部署CPU消息通道与CoreLab端点；复用ACK_CHAIN_V2，使用现有local-pc双SSH合法路由，不开公网端口/新隧道。6条teacher relay资格（每任务两条开发layout）在真实物理中走同一观测/参考安装/ACK路径，必须6/6完整成功，才允许相应模型控制。Teacher端点是工程阳性对照，不算神经policy。

模型输入构造必须是已验证observation-only路径，禁止把训练sample的未来action、privileged或task_id携入在线batch。Hopper驻留权重与processor，CoreLab暂停等待新query，返回H40后原SafetyGate判断，只安装prefix8推进200积分/0.4sim秒，再用新实际RGB查询。每query绑定run/episode/tick/obs hash/实际TCP anchor/checkpoint/processor/stats/ref/ACK，重复/过期/错episode拒绝。ACK丢失先查持久状态，不重发物理命令；未知进度记UNKNOWN。网络延迟算wall和GPU allocated，不算sim时间。

模型服务串行处理请求；CoreLab可有2个独立评价环境并行推进/等待。每query采用独立、可记录的噪声身份，不让到达顺序改变随机样本；没有通过batch等价验收则不引入GPU批量生成。单query60s、单episode20min墙钟上限，到限保留失败/不完整，不能teacher接管后计model成功。

M2048一生成即可在预封存开发面板做每任务2条真实短评（仍完整32s成功标准），另4条旧v2 DEV回归。与主训练在同一个allocation中分阶段串行，不用一份可变模型同时训练和服务。评价完成恢复同一训练链继续；M2048是早期开发产物，不触碰CONFIRM。M4096复用同一开发面板各2条与4条旧回归；模型某任务开发至少1/2成功、无真实安全违规才进入该任务CONFIRM，其它任务可独立继续。开发失败不是隐藏，必须交付首失败阶段的视频/请求/执行证据。

原SafetyGate/关节/力矩/穿透/工具速度门保留；拒绝计失败，不自动投影/平滑/重定时/换noise。没有真实安全事故与“动作被门拒绝”分开；真实安全违规停止对应模型/场景的后续物理执行。

## 11. 多任务可行policy的验收，不用一个总平均分掩盖失败

CONFIRM为12个全新场景组，四族各3组，每组3任务；三臂Teacher/B0 pristine/M4096，各36次，共108次预算。所有臂同初态、同相机、同executor、同32s评分。Teacher先运行；每任务至少10/12且无真实安全违规才给该任务环境资格，失败case仍在分母，不筛掉。两模型按预封存组顺序进行固定整块评价，减少重复加载；初态噪声按group/query定义，与模型与任务到达顺序无关，实际bytes留档。

候选固定M4096，不在CONFIRM上从多checkpoint选优。最终报告至少逐任务/逐场景族列：attempted/planned/success、wrong target、grasp/lift/carry/release/withdraw或push/no-lift谓词、SafetyGate reject、真实安全违规、reference→installed→achieved误差、wall latency。语言反事实报告同组PLACE_BLUE与PLACE_YELLOW是否均到正确目标，以及PLACE_BLUE/PUSH_BLUE是否分别满足抬升/不抬升历史。

**本夜最高状态 SIM_MULTITASK_POLICY_WORKING_V1**：同一固定checkpoint三个任务各≥9/12完整成功；四族每任务各≥2/3；PLACE_BLUE与PLACE_YELLOW同组双成功≥8/12；每个placement任务wrong-target≤1/12；两种技能均满足各自接触历史；模型真实安全违规0；视频/数据/权重/processor/stats/执行命令可以回读。B0对照实际差值单独报告，不预设微调必胜；若B0已工作，不能把全部能力归因于本次训练。

这是小样本、固定底座、两种技能的开发里程碑，不是统计显著、广泛多任务泛化、实时控制或移动Go2/WBC/真机资格。只有PLACE_BLUE/PLACE_YELLOW成功时标LANGUAGE_CONDITIONED_PLACEMENT_ONLY；只有TRAIN成功标DEVELOPMENT_ONLY；只有数据/CPU PASS标DATA_READY_POLICY_NOT_RUN。不得为了给用户一版work policy降低原物体成功标准。

## 12. 失败时继续做什么，停止做什么

- Hopper认证缺失：继续live多场景采集、CPU审计/打包和relay teacher资格；提醒必要登录，不能以继续写CPU准备文档代替神经进展。
- 并行不等价或不提速：退回已合格串行/双worker；数据语义不变，不以强上4worker作为任务目标。
- push teacher未过门：停止该技能正式采集，保留负例；可交两目标抓放数据/开发策略，但不伪称两种技能。不要临时改成reach来冲多任务数。
- Drop-A缺任务/场景覆盖：在原180次正式名单内完成尚未尝试项；不能用B中更简单条件替换A的失败。未达完整门只交部分bank/明确开发结果。
- 数值拟合差：先核对单个真实batch、mask/标签/梯度/归一化及观测歧义，不盲目全模型解冻/加loss。本夜没有第二个配方自动搜索。
- 离线好、闭环差：分别报告观测条件、生成参考、解码/安装、actual跟踪和接触阶段；不把WBC拉进来掩盖固定底座政策失败。
- 一个网络或路径局部失败只暂停依赖它的工作；正式训练bank冻结，独立Drop-B可继续。真正deadline/CPU/GPU/空间上限则停止新增，保留明确部分结果。

## 13. 必交产物与GitHub回报

1. ADOPTION/OLD_NEW_HANDSHAKE/RESOURCE_LEDGER/WORKSPACE_HANDSHAKE及已验证可执行status/resume命令。
2. SCENE_REGISTRY_SEAL：72组固定cards、场景XML/asset引用、task/instruction、camera K/T、teacher/scorer版本、每次commissioning/pilot/串并行正负结果。
3. Drop-A/Drop-B独立DATASET_CARD、attempt ledger、group splits、TRAIN-only stats、全成员manifest、固定私有HF revision、实际Hopper读取资格；标明B是否用于训练。
4. 实测BENCHMARK：raw与admitted episodes/hour、CPU core-hours/episode、GPU step time及DataLoader wait，不用CPU秒直接推断wall加速。
5. 真正训练的MODEL_IDENTITY/TRAINABLE_PARAMETERS/TRAINING_RECEIPT/EXPOSURE_LEDGER/RESUME_EQUIVALENCE；M2048/M4096或准确部分checkpoint的完整Trainer及model-only包。
6. 可直接加载的POLICY_BUNDLE：权重+processor+stats+embodiment mapping+camera/action contract+entry/source SHA+固定数据revision+推理及仿真eval命令；不依赖可变latest或teacher任务路由。
7. 开发/旧任务回归/三臂CONFIRM逐episode原视频、请求/ACK/物理记录、TASK_SUCCESS_MATRIX.csv、语言反事实与失败归因；最终必须明确哪些任务真正work。
8. #11发布真实各阶段状态、已用/剩余额度和固定文件引用；#47只发布路径/并行基础设施部署事实。一个集成发布者正常fetch/merge/push保留并发提交，逐文件回读，不force/reset。状态文件的PASS必须绑定实际收据，不以本GO存在当执行成功。

研究主线维持任务空间高层学习＋冻结执行器。今晚WBC代码/权重/运行0，真机0。下一轮再用正确任务参考验证冻结WBC并接入同一政策；不把该项作为本次固定底座policy的前置障碍。

## 14. 固定参考

- 研究章程：docs/RESEARCH_CHARTER_TASK_SPACE_MANIPULATION_V1.md。
- 最新审阅状态：reports/PI0292/20261013T025824Z-localpc-pi0292/PHASE_STATUS.json @43784131815bc8691c3faa7f8d43bac330d67e71。
- 并行准备：reports/engineering/2026-10-13-iteration-preparation/PREPARATION_REPORT.json；synthetic资格不是live资格。
- 当前共享入口：AGENTS.md、START_HERE.md、docs/operations/2026-10-13_UNIFIED_ENTRY_AND_GITHUB_SYNC_V1.md及WORKSPACE_ENTRY_ENFORCEMENT_V1.md、infrastructure/workspace/ENTRY_POLICY.json。
- MuJoCo 3.2.7 per-state并行与OpenGL约束：https://mujoco.readthedocs.io/en/3.2.7/programming/simulation.html 与 https://mujoco.readthedocs.io/en/3.2.7/programming/index.html。
- PyTorch DataLoader文档：https://docs.pytorch.org/docs/stable/data.html。执行者核对本机版本，不能把最新库文档支持的选项当成本机已安装能力。
