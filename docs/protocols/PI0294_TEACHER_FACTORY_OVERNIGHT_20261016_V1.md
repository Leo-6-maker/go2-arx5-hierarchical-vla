# PI0294 TF-NIGHT：无人值守 Teacher 数据生产闭环

日期：2026-10-16，Asia/Singapore。用户本次请求：**“制定一个夜间无人值守自主推进goal，跑通teacher数据生产闭环”**。

状态：READY_FOR_CODEX_ADOPTION_FROM_CURRENT_USER_REQUEST。本文是交给现有 local-pc Codex 的完整有界执行目标，不是已开始运行的收据。执行 owner 读取本次用户请求和固定发布，登记一次 NIGHT_ADOPTION 后按阶段自动推进；不把每个普通工程修复、规划候选失败或一个任务成功后进入扩采变成再次等待用户签字。账号权限、真实资源可用性、硬件/公开分发等越界事项仍不能自行扩大。

## 0. 成功定义：要交付数据，不是再交一套准备文件

核心目标：在完整自由根 Go2＋ARX5、冻结原 WBC 的仿真中，获得真实完整成功示范，形成可重复的自动采集器，扩出小型冻结数据集，并由独立消费者通过原生数据加载/解码验收。

按真实成果分别判定：
- NO_TARGET_SUCCESS：无完整成功；必须交最小可复现阻断，不能叫数据闭环完成。
- TARGET_SEED_ONLY：至少一条完整成功，但尚未形成有独立场景的可加载bank。
- SINGLE_TASK_PIPELINE_CLOSED：至少一个任务有 >=8 TRAIN、>=2 DEV 个独立成功episode，冻结manifest/stats/成员完整、独立消费者读取通过。
- DUAL_TASK_PIPELINE_CLOSED：T1/T2各 >=12 TRAIN、>=4 DEV 个独立成功episode，满足数据及消费者验收。另列 SHARED_RESET_PAIRED：至少4个已准入场景组的两任务使用相同机器人/夹爪/物体/目标/相机初态且均成功；未满足时明确两个任务来自不同初态域，不能声称验证了语言条件能力。
- 本夜目标为32条合格episode（每任务12 TRAIN＋4 DEV），不是32次尝试保证32成功；允许一项技能先形成闭环，另一项失败不使已取得的单技能成果归零。

不启动 GR00T/FB13 神经训练、推理或 PBS。数据封存及独立读取合格才得到后续微调入口。本轮避免为GPU排队等待；Hopper至多承担允许的短CPU消费者检查，不能在login节点做长计算。独立消费者也可位于local-pc/CoreLab另一干净进程，必须实际读取字节且披露位置。

## 1. 必须承接的最新事实，而不是重开旧实验

本次读取仓库到 `68a6da3c40e01e60cd7dd8bbf0e3a326b8c9b71a`（2026-10-16 00:00 SGT附近快照）：
1. 源重定位V1六源未准入：T2全局减速50.49/53.87/61.88秒；T1接触端有合法解而接近端无合法见证。它没有新物理失败。
2. 后续V2已将T2局部时参降到18.60/19.40/19.10秒。T1删除非接触前缀后已实际运行5.6秒，发生 base_link–task_table 非法接触；该原生安全FAIL完整保留，不能引用更早的“只有CPU未运行”。
3. V3 `60cef8283f4a44baa3367d6f264db5250f52f5bb` 已明确采用。T1整体任务桌/方块/目标/参考升高到H47；T2改接首contact并用working-closed初态。采用记录在 `reports/PI0294/design/SOURCE_ACCESS_V3_HUMAN_ADOPTION.json`，代码/CPU检查在推进，尚不能据此宣称任务成功。
4. 当前M2048、旧固定底座v3.x全部保留。成功整机数据数量以owner刷新后的实际结果为准，本次快照未看到完整成功。

入场只刷新最新#11/#47和原生进程/lease/账本。若V3已有健康进程，先让其按原source/授权到正常episode终点，不kill、不热改、不重复实例。本goal采用后只向新的 TF-NIGHT ledger 启动后续任务；采用之前已经启动的V3工作仍记原账，本夜汇总只引用它，绝不同时记为新物理两次或重写旧T0。

V3已生成的合格结果直接复用资格/种子，不为了跑新计划重复全部检查。旧有效FAIL、已关闭系列标记和原件一律保留。只追加新run、修订和attempt身份。

## 2. 时间和资源：一个新的夜间目标，不是修改旧截止

现有用户夜间请求对应这一次新的有界goal。采用时记录真实新T0和新预算分录；原累计ledger永久保留，所有既有有效并发任务先纳入资源视图。新goal不能通过改旧scope里的deadline或删除closure实现。

- 硬收尾：**2026-10-16 10:00 SGT**；09:00后不启动新场景、新版本或新物理episode，最后1小时仅完成已在途且能及时结束的工作、审计、封存、传递、报告。
- 如果owner到10:00以后才读到本文，仅只读交接，不自动延到下一夜。
- 新CoreLab预算：**64 CPU core-hours，总并发最多12核**，包含冻结腿策略、MPC、MuJoCo、渲染、规划、审计和压缩。先核实主机许可/实际空闲，不占用其他人的资源。
- 物理最多72次启动，主积分最多4,788,216步，真实双RGB记录最多184,392组；早期开发/失败/重置/warmup和并行资格均计入。任何一项先耗尽则停止新物理。
- 每次任务最长128秒，正常warmup最多5秒；每次启动最多66,503积分（含保守尾部），超出不得继续。短早退实际计数，不能拿理论上限当真实成本。
- 若沿现有dense/clone检查，另计最多8,000,000个静态/影子检查点；所有CPU仍在64coreh内。静态FK不是实物理，但也不是免费计算，更不能称已动态验证。
- 最多24,000个高层Teacher提交决策；重规划bundle最多2,000、实际CLIK初值最多384,000、RRT调用最多8,000。单bundle最多60秒wall、192实际IK初值、8个RRT（每次<=2秒）；全部内部失败计数，预算是上限不是必须搜索完。
- 新峰值存储CoreLab<=96GiB，local-pc<=8GiB，消费者Hopper<=2GiB；保留至少20GiB或站点更严的可用余量。私有目标无容量则不盲传、不改public、不删旧数据。
- GPU/PBS/GR00T查询/梯度/真机/WBC改动/历史删除均0。冻结腿RL的原生CPU推理是已知执行器负载，计CPU，不冒充高层模型运行。

建议72启动分配：开发<=12，资格/重复性/串并比较<=12，生产<=48。可以在总上限内把未用开发槽转为生产，不能把已用失败删掉。最后至少6coreh留审计/交付；剩余不足最坏episode+收尾时不再dispatch。

## 3. 永不变化的边界

- 使用完整自由根、十二腿关节、六臂关节、有限夹爪和真实足地接触；不固定root/关闭腿/隐藏支撑力/拖动物体。
- WBC/MPC/腿策略代码、权重、gain、cost、solver、动力学与原机器人安全限制冻结。不开新的保持RMS归因课题，不加重力/积分/目标误差补偿。
- 原生500Hz安全、夹爪行程/力限/碰撞规则、任务空间速率、任务的抓握/抬升/推动/释放/稳定性空间阈值不放宽。碰撞或关节违规不能因为数值小就忽略。
- 源实际状态、原提交命令、目标环境新提交命令和目标实际结果分别保存；不能把未来actual伪称专家目标，也不能把外部RGB/理想FK当目标实际观测。
- 本夜只用已核验自有T1/T2各000/001/002六源和现有V3工具/技能组件。不下载更多库，不迁移仿真器，不执行未知pickle，不把需要人类操作的备选设为无人值守依赖。
- 不使用DEV/TEST/确认集为Teacher修复提供反馈。所有调试场景在development域。原WBC保持FAIL保留，但不作为成功物体数据的全局阻断。

## 4. G0：先在30分钟内接对系统，不再重建基础设施

读取AGENTS、START_HERE、冻结WBC主线规范、V3及最新实际结果。复用 `~/Projects/GR00T-WBC`、现有enter/doctor/私有runmap、sdr/sdr2/源接入V3、现有MPlib隔离环境与dispatcher。只测被改动的接口，不重装MPlib、不重读所有历史大件、不重做全部旧CPU测试。

建立 NIGHT_ADOPTION、资源账、唯一coordinator、child worker lease、当前开发版本和截止。操作者不可用是正常条件，本夜路线全部自动；凭据不足只阻断对应外部服务，其他本地有效工作继续。

把代码改动放新的不可变source capsule。现役checkout管理目录与运行源码分离；一个NFS修改owner，不给多个worker并发git pull。文件发布保持并发commit，禁止force/reset/清理旧树。

## 5. G1：自主拿到成功种子——承接V3，允许有限任务侧分支

### 默认路线
先看V3实际结果，合格种子直接采用；否则从已固定的V3场景/源入口起步，不再测试旧uniform+不可达approach组合。

可用场景仅H44与H47两种已定义完整桌面高度，桌/物体/目标一起在初始化时正确放置；不缩小桌子/挖洞/改摩擦/碰撞。H47是针对已有机身–桌面证据的任务场景版本，非控制器补偿；新旧域分别记录。

初始站位仍用原生实际reset。BASE高层偏好明确近场zero，有或无BASE字段语义不混；真实root始终自由。由此得到的是完整身体近场EEF13示范，**不能宣称学到了非零BASE移动策略**。

### 允许owner自主选择的有限任务方法
1. 接入点默认使用V3已核验的首close（T1）／首contact（T2），使用源中的完整接触后缀，不强制重现旧不可达的非接触align首点。
2. 目标入口规划器可以构造沿声明源接近方向退让1/2/4cm的有限预接触候选，加上直接接触入口，共最多4种；工具/碰撞/开度保持真实。预接近与接触必须组成合法路径，不是有一个IK解就直接送动作。
3. 六源仍按固定task/source排序。CPU阶段可以搜多个合法候选，首个满足完整路径+时间+安全契约者选中；这些是内部规划，不把一次IK失败当成整个夜间停止。完整源一旦在episode启动前锁定，任务中不热换来源。
4. 优先采用已验证局部时间法，保留源pose/gripper/顺序，使用阶段原速度上限；不能因少量边界峰值把整段统一拖长，也不静默剪掉接触事件。可复用现有时间参数化检查实际关节速度/加速度；不可只凭cartesian下界宣称可执行。
5. 若原32秒时序可行，优先32秒原评分。否则允许显式慢速seed域，见第6节；不继续为28秒限制浪费整夜。源几何变化/拼接属于新的目标Teacher，保存完整变换及新标签，不声称是源原样回放。
6. 进度由实际已执行prefix推进；源事件用于分段，目标实际接触/grasp用于任务判断。必要事件缺失时最多合计12秒等待/最多3次有根据的任务级重规划，在已声明路径族内执行，不能无限重复接近或闭爪。等待/replan时间和新命令全部在本episode记录与既定时长内。32秒模式不能偷加时间。

最多6个有实质差异的Teacher开发版本，最多12次开发物理启动。每个修订必须先写 CHANGE_NOTE（原因、允许改动、受影响测试、复验条件），再启动新attempt；不能更名重复同一失败。至少一个任务成功就进入自己的重复性和数据生产线，另一任务可以在开发预算内独立继续。

### 普通失败与安全失败分流
- IK/RRT/时间不准入：保留候选证据，继续本bundle剩余有界候选或下一个开发版本；不等用户批准每个内部搜索。
- 安全但无接触/无任务成功：该episode记真实FAIL，不能加入成功bank；可以按已授权任务方法写明原因后继续新的development attempt或其他任务。
- 真正native安全失败：立即终止该episode并冻结同场景/同Teacher版本的全部物理worker；正在运行其他worker按已合格安全暂停或终止流程处置。没有证据前不重跑。只允许既定任务几何/路由/初始化范围内的有因最小修订，经CPU及单worker重资格后新版本再启动；不修改WBC、限位或接触规则。全夜最多2次真实native安全FAIL，第二次后停止所有新物理，保留已有安全成功数据继续只读审计/交付。无法归因或只能靠WBC改动解决时立即停止该物理线。
- 进程/记录/序列化错误：有原journal可恢复则补做记录审计，不重跑物理冒充原件；无法恢复的attempt保留INCOMPLETE，新attempt也计数。只有同语义工程修复才可自动继续。

## 6. 32秒与慢速种子域：明确授权的任务时间参数，不改历史成绩

定义 SEED32 和 SEED_SLOW64/96/128。每个scene family/Teacher版本在第一次任务前，根据静态连接+完整技能时间+必要等待余量+末尾4秒，选择能容纳的最小长度，随后冻结。开发后不能因为某条失败在episode内延长；生产中不能按成功快慢临时选择结束时刻。

32秒仍完整沿旧28–30s保持与32s观察。慢速域使用同一组物体/安全/释放阈值，保持窗口移到T-4至T-2（仍41个20Hz点），并观察到T；活动需在T-4前完成。这是显式新的seed时间评价合同，不冒充原32秒通过。原scorer原件不覆盖；新参数化wrapper只绑定时间窗口，必须对T=32做等价回归，对晚失败/假接触做负例。

N_cmd=20T、N_obs=20T+1、N_H40=20T-39。T=128时为2560/2561/2521，仍是一条episode不是2521条独立示范。最大128秒+5秒warmup，20Hz观测、0.002s物理、H40、prefix8=0.4s和原速率不变。无意义停滞不会因延长变成成功；有限无进展规则仍记录具体失败阶段。

## 7. G2：重复性→共同初态→冻结Teacher与场景

每任务首成功后，使用同版本最多3个正常reset中的预先指定独立开发条件，至少2条完整成功、无真实安全FAIL，才开放该任务生产。已有V3符合本条件的结果可引用，不重复计为新数据。

双任务优先构成共同初态：选择一个已资格的公共桌高/真实机器人和夹爪reset，T1/T2同场景仅语言不同。默认检查H47+working-open能否同时服务两任务；T2所需闭爪在真实任务中执行并计时间/标签，不能reset时偷偷闭爪而仍称两任务同初态。

如果共用初态无法在本夜通过，允许各自合格域继续产出任务特定数据，明确 H47/open 与 H44/closed 等区别，paired=false；不因此抹掉单任务闭环，也不宣称模型可以仅靠语言区分任务。不能把两个不同reset域混成语言反事实数据。

在任何正式TRAIN/DEV场景生成前，冻结 TEACHER_RELEASE、SCENE_FAMILY、tools/normalization schema 和成功评分版本。DEV一旦运行不得拿结果修改该release后再声称该组未见；若必须改release，旧DEV变development旁证，新版本必须用剩余预定新组。

## 8. G3：合格后多worker生产，采集/审计/导出成流水线

优先12个TRAIN场景组＋4个DEV组，每组T1/T2（各16条，共32条目标成功），T1/T2条件可独立缺失。开发场景及原旧DEV/确认集不在新生产split。原始seed/source共享不等于场景组独立，所有近重复派生与task配对归同一split。

小范围参数只在已资格近场包络内：物体与目标XY围绕基准各<=1cm、方块yaw<=5度，必要的中性光照强度0.9/1.0/1.1倍。先几何/视野/碰撞检查，再运行；不改变camera mount、物体尺寸/质量/摩擦，不引入W1/复杂移动。每个场景在task开始前登记参数、seed、split和哈希；拒绝的场景也留候选分母，不换编号直到凑全。

冻结一个包含最多24 TRAIN组＋8 DEV组的备用池，task优先轮转、成功较少者优先但不依据DEV质量调Teacher；实际生产最多48次启动，各组各task最多一次。达到每任务12TRAIN/4DEV成功便停止额外生产，不能重跑同组冒充独立样本。剩余不足时输出真实partial bank。

1→2→4 worker资格：每个worker必须独立ROS domain/namespace、MPC/reference、腿history、MuJoCo状态、仿真时钟、渲染、输出、CPU affinity。复用旧dispatcher/CPU管线，不复制固定底座worker假装fullbody。用独立诊断输入检查无串台、时序、任务/安全结果与可实现的数值一致性；不要要求不确定渲染字节必须与不同后端相同而阻塞。原同后端一致性要求保持并披露实际覆盖。

默认2worker，各最多4核，余下核用于审计；4worker仅在CPU/RAM/ROS隔离和实际吞吐通过时开启（例如每worker2核，另4核后处理）。整项12核不超售，设置OpenMP/BLAS/Torch/渲染线程上限。历史固定底座2.86倍不是本夜保证。比较的是完整准入episode/小时及coreh/成功，不只比较积分帧率。

Coordinator原子分配ID/CPU/积分/磁盘预算；worker写独立临时目录，结束时封存manifest/READY；审计只读完成件。PNG/hash/压缩可有界队列，满则阻塞，禁止丢帧。一个episode通过即允许exporter增量处理，不等整批；冻结数据版本后不热改。异步打包错误不能杀正在正常进行的采集。

## 9. G4：训练就绪的数据契约

逐episode必有：真实双RGB/K/外参/时戳；实际EEF/夹爪/BASE；原生qpos/qvel/ctrl/接触及安全journal；Teacher提出、接受/安装、实际执行的命令；完整任务事件与失败原因；source/scene/Teacher/plant/WBC/clock/评分版本与成员hash。

动作仍为EEF13：EEF XYZ+row-rot6d+gripper+BASE vx/vy/heading偏好。近场base监督可以真实为零，但必须标明BASE_INTENT_NEAR_ZERO，不冒称有移动训练覆盖；actual base即使因WBC自主运动也不能被回填成目标速度。旧EEF10数据不补3零冒充完整新域label。

模型输入只有实际可观测RGB/EEF/夹爪/BASE/语言；object真值、FSM阶段、成功布尔值只给Teacher/审计。绝对高层命令落盘，官方processor只做一次EEF相对转换；BASE速度单独模态，不能再当pose变换。动作索引与实际提交prefix保持；未执行的H40预览不能冒充执行标签。

独立审核分三层：
1. 实际完整物体成功 AND 原生安全；高频abort覆盖低频采样PASS；缺证据为UNKNOWN，不默认PASS。
2. 输入/标签/观测时钟/安装/已执行语义一致，完整长度和camera来源真实，source/target状态不混。
3. scene-group split有效、only TRAIN统计、有效维度mask与浮点encode/decode，在固定容差内。不得因DEV裁剪而拿DEV重拟合；新数据域用新统计，旧stats不改。

先冻结最小单任务bank（>=8 TRAIN/2 DEV），然后双任务bank（各>=12/4）。可以边生产后续版本边消费已冻结bank，不边训边改样本池。本夜只有CPU数据资格；不得加载VLM/DiT/Adam或称已完成模型smoke。

## 10. G5：独立消费者实读与交付闭环

输出版本化 LeRobot/本项目官方N1.7兼容数据包、manifest、TRAIN stats、modality config、语言/scene分布与可执行的CPU验收命令。

独立消费者使用另一个权限/进程清晰的输出根实际读取固定成员，至少对全部manifest成员完成存在性/大小/hash检查、全部数值窗口codec验证，并跨task/stage读取真实图像走官方processor/collator；不得只读README或把文件格式转换当已训练。消费者环境必须允许相应CPU工作；Hopper未就绪则用local-pc/CoreLab隔离消费者，不等待GPU。

优先使用现有授权私有HF repo/prefix或独立私有恢复存储，禁止本夜新开public数据repo、改可见性、上传raw到已有public checkpoint库、要求用户在夜里输入token。远端容量/认证失败时：合格本地bank和独立local消费者继续完成，状态明确 LOCAL_LOOP_PASS_REMOTE_DELIVERY_BLOCKED；不要谎称异地灾备/完整远端交付已通过。至少保留唯一原件和member hashes；已有源不删除。

记录LOCAL_BANK_READY、INDEPENDENT_CONSUMER_PASS、REMOTE_PRIVATE_READBACK分别的状态。远端失败只阻断远端交付，不回头重跑任务或删除已合格数据。

## 11. 无人值守运行机制与停止规则

建立持久coordinator状态机：ADOPT→SEED_DEVELOP→REPEATABILITY→COLLECT→AUDIT→FREEZE→CONSUME→CLOSE。计划、request、attempt和worker身份原子落盘；每30秒轻量heartbeat，每5分钟聚合一次队列/资源状态，不每步fsync大文件。断SSH不是run终态；先查真实PID/start/job再决定恢复，不盲目启动第二owner。重启coordinator只恢复未消费的stage/slot，不重放已安装动作。

每个工程问题使用有界故障工单：证据→最小patch→受影响CPU测试→单worker资格→新固定release；同一根因最多90分钟主动修复，超过后换本goal内另一独立任务/来源方法或收敛阻断。最多6个任务Teacher版本，不能不断改名突破上限。

普通failure不关整个goal；全部候选或本任务工程范围耗尽，仅关闭该task lane。成功lane照常扩采。只有真实安全累计上限、未知共享安全缺陷、唯一owner/输入身份损坏、账号权限/机器资源政策阻断、时间/总预算耗尽，才停止相关全部执行。维护/上传类阻断不传播到无关健康lane。

本夜不依赖人工辅助，不通知等待人来点下一步，不新增monitor/mail/定时服务到用户账户。owner自己的进程监督只存在本goal窗口，收尾后关闭；本文件不是ChatGPT在后台运行的承诺。

## 12. 交付与实际进度报告

阶段改变、发现根因、取得首成功、bank冻结、结束时提交脱敏小报告到GitHub；正常运行合并为约60–90分钟一次，勿把大量归档/重复回读置于关键路径。原raw/视频/WBC仍私有，Git只存索引/自研代码/统计/声明。

必交：NIGHT_ADOPTION.json、CHANGE_LOG.jsonl、ATTEMPT_LEDGER.jsonl、TEACHER_RELEASE.json、SCENE_SPLIT_MANIFEST.json、FIRST_SUCCESS_REPORT.md、PARALLEL_THROUGHPUT.json、DATASET_CARD.md、DATASET_MANIFEST.json、MODALITY_CONFIG、statistics、INDEPENDENT_CONSUMER_RECEIPT.json、PRIVATE_DELIVERY_RECEIPT.json、RESOURCE_FINAL.json、FINAL_REPORT.md、RESUME.md。

最终报告第一屏必须回答：两任务各真实接触/完整成功/失败/工程未完整多少；多少独立TRAIN/DEV/配对组；属于32秒还是慢速域；worker实际用到几路、生产和审计重叠多少；独立消费者是否实际读过；下一次训练可以直接用哪一版数据。没有新的任务RGB就不伪造视频；有视频但无成功不算完成。

硬截止前停止dispatch、完成可结束的在途工作、停止本goal自有worker、结清预留、保存原件和恢复说明，#11科学/#47工程结果各发一次并回读。selected/deployment始终NONE，本goal从不冒称GR00T已学会任务。

## 13. 当前审查依据

- `START_HERE.md` 与 `SOURCE_ACCESS_V3_HUMAN_ADOPTION.json` at 68a6da3c40e01e60cd7dd8bbf0e3a326b8c9b71a。
- V3方案：`docs/protocols/PI0294_TABLE_CLEARANCE_AND_CONTACT_ENTRY_SEED_V3.md`（准备文字是历史；采用以human-adoption为准）。
- V2实际安全终局：#11 comment6195800940；早期源V1终局：#11 comment6194051557。
- `docs/protocols/GR00T_TASK_DATA_FIRST_FROZEN_WBC_SCOPE_V1.md`：冻结WBC、真实任务数据优先。

本次发布没有实际启动服务器任务。现有Codex按当前用户请求登记新goal、确认真实资源/原run边界后执行上述完整有限范围，无需每次同范围工程修复重新取得一个GO。越界或新增硬件/控制学习/公开分发不属于本goal。
