> **当前事实（2026-10-25 09:29 SGT）：** 原 B24 已在有效窗口内实际启动，但所选 GPU 上有其他计算进程，入口在原生绑定、模型和物理前明确拒绝。该新失败已一次结算、GPU／物理双 lease CLOSED 并独立回读：61 GPU 分配秒、1 进程 CPU 秒，0 模型／查询／物理／有效条件，控制完整 upper 保留。人类随后明确批准共享 GPU0 做同一 B24，GPU1 仅在实际显存资格成立后可作为后备；不停止其他任务。详见[真实关闭与共享接续](reports/PI0294/design/parallel-science-20261022/V4A_B24_PREBIND_ZERO_CLOSED_SHARED_GPU_COLD_ACTUAL_20261025.json)。新共享边界尚未原生采用／启动；完整剩余资源复算无缺口。原 B24 最晚启动仍为10:00，其他原时钟／caps／matrix／final不变。A24=11/24，B有效0，B−A未知，完整goal未完成、core64=0。

> **当前授权（2026-10-25 09:22 SGT）：** 人类已明确批准仅原 B4096／N294101 全 24 条件在今天 10:00 SGT 前登记并启动，以及此前已批准的新冷态边界；原 UNKNOWN 不重发。精确授权见 [10:00 单 B24 授权](reports/PI0294/design/parallel-science-20261022/V4A_B24_ONLY_1000_EXACT_HUMAN_AUTHORIZATION_20261025.json)。原 T0、全局停新、停止训练／物理／硬关闭、全部 caps、矩阵与最终保护不变。真实闭合双账本的完整剩余资源离线复算无缺口，失败准备费用保留；896 项 PV1 CPU 测试通过。代码与授权尚未原生采用，尚无新 GPU／模型／物理。A24=11/24，B 有效0，B−A未知；完整 goal 未完成、core64=0。以下旧状态按时间保留。

> **当前事实（2026-10-25 02:34 SGT 独立原生回读；传输只读验证随后完成）：** A4096 原 N294101 面板已完成并审计、结算关闭，11/24 成功（T1 7/12、T2 4/12）；B4096 有效结果仍为 0，B−A 未知。没有现役 B GPU 或后台科学 owner。超时部署的悬挂准备进程已精确关闭，双账本全文未变，原 UNKNOWN 保留。811/329 GiB 两项存储 cap 已采用。原 02:29:56 停新时间已过，[仅原 B24 的一次接续例外草案](reports/PI0294/design/parallel-science-20261022/V4A_B24_ONLY_LATE_SUBMISSION_KNOWN_ZERO_COLD_BOUNDARY_EXCEPTION_DRAFT_20261025.json) **待明确批准，未采用**。完整 goal ACTIVE／未完成，core64=0。以下先前“健康后台／A22”等叙述是历史快照，不能据此重启旧 actor 或恢复旧请求。

新增[原完整 T1 失败的 H40 夹爪／实际 ACK 诊断](reports/PI0294/design/parallel-science-20261022/V4A_A_T1_F1_004_HOLD_GRIPPER_DIAGNOSTIC_ACTUAL_20261025.json)：F1-004在原28–30s保持窗口中41个提交点的夹爪目标0.0106–0.0533，6次draw的完整H40最大0.0688；原生抓握41/41、物体高度0.529–0.546m高于原0.460m放置目标，完整放置／释放／撤离未完成。以原冻结0闭合／1张开语义解释，不改门、不将实际开度差直接判为执行器故障，不从DEV生成恢复标签或推出训练干预。原参考完整hash与原manifest／ACK／score已核，仅新读终态参考CPU0.0201s，无模型／物理。另同一健康后台原队列已自行A22＝11成功11失败（参考1／S1门7／完整任务3），live原生安全0，B0／配对增益未知；原A16图谱保留。原caps／费用／matrix／final／截止不变，S1 core64=0，完整goal ACTIVE。

新增[实际 C1 单锚点重复曝光与 A19 终态补充](reports/PI0294/design/parallel-science-20261022/V4A_C1_MATCHED_SINGLE_ANCHOR_EXPOSURE_AND_A19_ACTUAL_20261025.json)：4个短导出各单anchor，合计每臂16365次（全分支12.486%、C1通道49.942%），来自2个学生父轨迹；原冻结uniform-export→uniform-anchor配方及A/B完整C1消费直方图均匹配，不宣称数据错误／新增独立示范，不改sampling/loss。新增3条已审计终态为2成功＋1预测S1拒绝（shadow56／拒绝段live0），A19为10成功9失败；B有效0、配对增益未知，A16原图谱保留。全部是已有保存文件的只读分析，未新训练／采集／模型／物理／预留或热改健康owner，原A→B后台自续；S1 core64=0，完整goal ACTIVE。

已按人类最新优先级交付 [Failure Atlas、真实 TRAIN 曝光与 24 条件候选准备](docs/handover/2026-10-25/V4A_FAILURE_ATLAS_TRAIN_COVERAGE_AND_CANDIDATES.md)：冻结 A16 为 8 成功＋8 有效失败（1 参考门／4 S1 预执行／3 完整任务失败）；发布前后台已自行到 A18＝9＋9、B0，配对增益未知。7 段 retained 消费链逐 cursor／forward 核对，A/B 各 131072 anchors，B 新库49152／A0；原T2 detach Prefix8 在两臂均覆盖30/30父轨迹，新T1对应物理事件Prefix8为67/75而H40为75/75，不能据此宣称必须过采样或更多训练。24个任务条件来自原冻结TRAIN的12个几何，仅候选元数据／历史静态资格，动态Teacher与标签NOT_RUN；DEV不转TRAIN标签。可执行笔记本及3图已用原保存证据复核，无新模型／采集／训练／物理／预留，健康服务器A→B继续自运行。原caps／全部历史费用／matrix／final／截止不变，S1 core64=0，完整goal ACTIVE。

已将同一半面板的5条有效失败按原始保存证据分解：[A4096 失败条件诊断](reports/PI0294/design/parallel-science-20261022/V4A_A4096_HALF_PANEL_FAILURE_PREDICATES_ACTUAL_20261024.json)。1条原参考门拒绝保留 TEST_ONLY 速度阈值出处，2条 S1 拒绝均为影子预测非法接触且被拒前缀 live=0，2条完整 T2 分别未达到水平推动条件、违反不抬升条件，二者均未达最终目标区保持。原始评分、安全门、时钟审计和全部失败标签不改；这些是 A 臂原测量的局部描述，尚不能归因为示范覆盖或据此启动训练。健康服务器 A→B 接续保持，完整目标仍未完成。

A4096 原首噪声面板已完成一半：[12条有效结果与后台阶段审计](reports/PI0294/design/parallel-science-20261022/V4A_A4096_HALF_PANEL_12_VALID_RESULTS_ACTUAL_20261024.json)。有效12/24＝7成功＋5有效失败；T1已测6为4成功2失败，T2已测6为3成功3失败，原参考门1／S1预执行拒绝2／完整任务未成功2，live原生安全故障0。每条原raw审计与动作—实际ACK因果链已核，12条captured bytes复用既有只读汇总通过；900次推理完成、错误0／pending0为该快照，不当最终结算。当前顺序前6几何并非完整12几何面板，不能用7/12宣称整体效果；B仍原队列有效0、配对0／B−A未知。健康服务器owner／GPU同PID自行继续剩余A，再审计、一次结算关闭并按原门派发B，无前台agent保活／控制；无新请求／重启／热改／重复资格或有效失败。只读回收复用旧9条、新取3条，没有新审计worker、模型或物理。原caps／费用／模型R0／安全／评分／matrix／final／截止保持，S1 core64=0，完整goal ACTIVE。

后台原Eval已独立推进到A4096有效9/24：[9条原DEV结果与阶段只读汇总](reports/PI0294/design/parallel-science-20261022/V4A_A4096_NINE_VALID_DEV_RESULTS_BACKGROUND_STAGE_ACTUAL_20261024.json)。A共5成功＋4有效失败，T1已测5为3成功／2失败、T2已测4为2成功／2失败；失败分为原参考门1、S1预执行2、完整32秒物体任务未成功1。全部原raw审计及两角色因果链已核，原实时native安全故障0；700次查询已完成、失败0／unknown0。第10条实际在途、live预留保留，账本快照不冒称终局或退款。B保持原已接受队列，GPU／模型／有效条件尚0，配对0／B−A未知；原顺序partial不当整体成功率或选模。原9条完整小原件以本地旧4条＋精确远端新5条回读，无重复审计、物理或查询；既有只读汇总实际bytes解析通过。服务器owner和A GPU同PID健康、退出登录驻留不变，自行完成A审计结算关闭后接续B，无前台保活／逐条控制。原caps／历史成本／matrix／final／模型R0／安全／任务／截止不改，不自动额外面板或训练，S1 core64=0，完整goal ACTIVE。

A4096 已取得首条完整32秒T1任务成功：[首个真实成功、独立审计和三条DEV现况](reports/PI0294/design/parallel-science-20261022/V4A_A4096_FIRST_COMPLETE_T1_SUCCESS_AND_THREE_VALID_DEV_ACTUAL_20261024.json)。F0-001／T1／N294101实际16000live＋16000shadow／641双RGB，80次原模型draw和80前缀执行ACK；抓取／抬升／搬运、释放／退离／目标区及41个hold采样的原完整评分均通过，原S1拒绝0／live原生安全故障0、raw独立审计及两角色因果链PASS，无Teacher接管。此前两条有效负结果不改；现A有效3/24＝1成功＋2有效失败，T1已测2为1成功1失败、T2已测1为失败；B有效0／配对0，B−A仍未知，不据顺序前3条宣称整体效果。已用原三终局bytes通过现有只读配对汇总，无mock／新模型或物理。服务器健康owner按原A24→审计结算关闭→B24自行接续，退出登录驻留已验证，不需前台保活／逐条控制；不重启／热改／重复失败或资格。原全部caps／已用成本／matrix／final／时钟不变，S1 core64=0，完整goal ACTIVE。

已按人类要求落实[服务器后台自动接续与退出登录后驻留](reports/PI0294/design/parallel-science-20261022/V4A_SERVER_BACKGROUND_SERIAL_CONTINUATION_AND_LOGOUT_PERSISTENCE_ACTUAL_20261024.json)：现役唯一owner和A GPU进程真实active，原已接受A24→B24队列保持；补齐原用户manager Linger=no→yes并独立回读，同PID／proc-start未变、标准输入null／无TTY，无新GPU请求／模型／物理、无重启或热改。正常流程由服务器自行心跳、原driver/core/audit、实际结算／关闭及B前置核验后派发，不需前台agent保活或控制；未知写仍不盲目重发。首次大只读超时未推定作业终止，随后精确轻读验证同一活进程。原caps／成本／模型R0／任务／安全／评分／matrix／final／截止均保持，不自动新训练／采数／其他面板；S1 core64=0，完整goal ACTIVE。

A4096 首几何的T1／T2均已独立审计为有效负结果：[两条真实失败与配对结果只读汇总资格](reports/PI0294/design/parallel-science-20261022/V4A_TWO_A_DEV_NEGATIVES_AND_PAIRED_READ_SIDE_CPU_ACTUAL_20261024.json)。T1在28.8秒因原参考速度门拒绝；T2在21.2秒时原S1 shadow于下一前缀第61tick预测ILLEGAL_CONTACT_OR_PENETRATION而拒绝，预测段live0、状态未改，累计10600实际live／10661shadow、425双RGB；53个前缀有执行ACK、54次原模型draw，两条raw审计及两角色因果链都PASS，实际live原生安全故障0。现A有效2/24（成功0／失败2），B仍队列有效0、配对0，不能判断B−A。新增只读汇总复用原baseline全字节审计，按原12几何／两任务配对；64项CPU／72子测试通过，真实两终局captured bytes无mock解析通过。无终局证据不推定NOT_RUN／失败，partial不出CI；完整首面板才按12几何block重采样，非正式DEV48／独立确认或选模。汇总代码未部署／未热改现役7cc capsule，无新GPU／模型／物理／账本写；健康后台继续原A24/B24，全部旧费用／caps／matrix／final／截止保留，S1 core64=0，完整goal ACTIVE。

A4096 已取得第一条独立审计的原DEV有效负结果：[首条件原参考速度拒绝与真实动作／反馈证据](reports/PI0294/design/parallel-science-20261022/V4A_FIRST_A4096_DEV_VALID_NEGATIVE_AND_REFERENCE_SPEED_REJECT_ACTUAL_20261024.json)。同bank294101／F0-000／T1实际执行28.8秒、14400 live＋14400 shadow、577双相机帧；73次原模型draw中72个前缀有执行ACK，第73份原H40参考因最大线速度2.042921m/s超过冻结2.0m/s门而未安装。原始阈值的TEST_ONLY来源如实保留，不把参考拒绝称为真实机器人安全故障。独立raw审计全部通过、原两角色因果链通过，native安全故障0；曾抓取／抬升／搬运和部分终态hold不等于完整32秒成功，结果为VALID_TASK_FAIL，不能重跑或改门。现A有效1/24（成功0／有效失败1），原T2已接续；B仍原已接受串行队列、有效0，配对0，B−A尚不可估。健康服务器owner不依赖agent保活，原48条件继续；不自动新训练／采数／其他面板，所有成本／caps／matrix／final／截止保持，S1 core64=0，完整goal ACTIVE。

Corelab 已产生新的真实执行进展：[A的原600s／80轮PASS、原4096模型加载及DEV首条件运行](reports/PI0294/design/parallel-science-20261022/V4A_CORE_ACTUAL_600S80_PASS_A4096_MODEL_READY_AND_DEV_STARTED_20261024.json)。同一现役进程完成600.0086秒真实心跳、80轮通信及隔离故障检查，随后两次新鲜原生放行均通过；A4096原权重／R0加载完成，冻结VLM参数hash通过。首条原DEV任务已真实启动，已有19次原生draw完成、0失败／unknown／pending；实际物理计数及仍在途live预留如实保留，不把在途计数当结算。独立终局有效条件仍0，不能计算B−A或称任务成功；B保持原已接受队列、未开GPU／模型／物理。复用健康后台owner，原600s资格已完成、不得重跑／重启／热改；下一项原driver/core/audit及原24条件。原全部caps／成本／25PBS／11714收费更新、matrix／final和截止保持；S1 core64=0／target未资格，完整goal ACTIVE。

Corelab 后台原生评估已实际接续：[两模型迁移关闭、原A/B顺序队列接受及A加载前资格](reports/PI0294/design/parallel-science-20261022/V4A_CORE_MODELS_DELIVERED_SERIAL_PAIR_ACCEPTED_A_NATIVE_PREMODEL_ACTUAL_20261024.json)。A/B两份原4096推理模型已完成HF固定revision上传／Core下载／完整SHA和原配置恢复，迁移费用一次结算、有限服务自然Exit0；完整Trainer保留。现役独立服务器owner已接收同bank294101的A24/B24两个精确请求，A原生GPU进程已通过boot/PID/32GiB设备／4核48GiB和完整资源身份，正在原600s／80轮加载前资格；B只排队、无GPU进程。不是模型／任务PASS，A/B DEV仍0、core64=0。原LOCAL辅助凭据路径保护导致的单侧GLOBAL预留已精确核验，随后仅补齐原LOCAL proposal，未重复reserve／request；两次前写／大stdin失败UNKNOWN和原件全保留，冷接续未改15数值、任务、评分、R0或安全。62项私有CPU/原once隔离测试通过，未冒称CI；成熟model.load/serve、driver/core与独立audit直接复用。后台服务无需agent保活，健康时不重启／热改；仅这两原面板，其他面板／采数／训练不自动启动。原caps、全部旧费用／25PBS／11714更新、matrix／final及截止保持，完整goal ACTIVE。

Corelab模型迁移已实际后台运行：[双账本预留与A4096固定HF提交](reports/PI0294/design/parallel-science-20261022/V4A_CORE_BACKGROUND_MODEL_DELIVERY_ONCE_RESERVATION_AND_A4096_HF_ACTUAL_20261024.json)。A的9份推理文件、12576324408bytes已全tensor前后SHA及远端逐成员核验，HF revision `fbff9c0c3fd0cd5bfbadf345c88a20be00fd970c`；Core正在下载，第一shard已校验、第二inflight，B未上传。原GLOBAL/LOCAL一次采用15000站点控制CPU、187800CoreCPU及25.22GB模型输出／50.31GB传输hold，未注册GPU请求／未开GPU或模型／物理。两次上传前控制失败及全部原件保留，精确已确认预留不重做，原LICENSE和Cosmos说明全文保留；28项私有CPU/原once临时测试通过，不冒称CI或Eval。现役有限后台进程无需agent保活；后继原评估owner适配仍在测试准备。A/B DEV仍0，core64=0，原全部矩阵／最终保护、caps和截止不变，完整目标ACTIVE。

Corelab GPU Eval profile已通过原once IO实际加入标准入口，并独立回读；旧profiles/enter/doctor均不变：[实际入口与原评估器接缝](reports/PI0294/design/parallel-science-20261022/V4A_CORE_STANDARD_ENTRY_ACTUAL_AND_EVALUATOR_HOOK_CPU_20261024.json)。既有pv1.neural现可接Core原生身份、完整资源、原600秒/80轮及同请求两次owner release，再复用原model.load/serve/physics/audit；没有新训练器或控制器。805项PV1/268子测试通过，含原17历史读取和原processor冻结哈希case，已准备原GLOBAL/LOCAL真实终态结算接缝，未当作实际结算。当前只新增一个入口profile元数据写，尚无新GPU/模型/物理/账本或科学lease；A/B DEV仍0、core64=0、目标ACTIVE，下一步实际封存/登记、双hold及HF传输。

Corelab 顺序队列及600秒/80轮通信身份接线已通过 CPU 验证：[本轮代码与服务器复算记录](reports/PI0294/design/parallel-science-20261022/V4A_CORE_SERIAL_QUEUE_AND_PREMODEL_BRIDGE_CPU_ACTUAL_20261024.json)。只允许原A24/B24两份完整面板、单卡顺序执行；队列意图不算原生接受，必须有截止前真实驻留owner ACK、实际boot/PID/GPU及完整准入。774项PV1及268子测试通过，Core服务器原once IO仅在临时文件中通过；原25PBS/11714更新、矩阵/最终保护和全部时钟保持。现有Core→Vanda连接健康、GPU无计算进程；原CORR数据清单/R0小元数据已精确回读。真实owner/双hold/登记/HF传输/600s80轮及闭环仍待，未启动GPU、未写科学账本或lease。A/B DEV仍0、core64=0，目标ACTIVE。

Corelab GPU 后端的冷态身份、原 GLOBAL/LOCAL 扣费与完整资源重算已接线：[代码、真实 OS 限额及服务器 CPU 记录](reports/PI0294/design/parallel-science-20261022/V4A_CORE_NATIVE_BACKEND_ACCOUNTING_SCOPE_AND_OS_CPU_PREPARATION_ACTUAL_20261024.json)。Core 用 boot/PID/proc-start/GPU UUID，未伪造 PBS；旧25条作业及11714收费更新保持。模型副本已同时纳入 Core 与 GPU 站点存储，完整未来/矩阵/最终保护无缺口（captured CPU 复算，不是 fresh 准入）。729项PV1、268子测试通过；实际 Core 临时 once IO 与4核/48GiB内核限额通过，应用后传输失败的 UNKNOWN 不重发。后台顺序派发、600s/80轮、模型传输及真实闭环仍待；没有新 GPU/PBS/模型/物理或原账本/registry/lease 写。A/B DEV仍0，core64=0，原所有截止保持，目标ACTIVE。

Corelab接续实查：原 CUDA 推理环境、官方51d4源码及模型/processor CPU导入已通过；已有 HF token 认证有效，公开仓库可读。A/B各9份推理文件、每臂12576324408bytes的精确传输/私有原配置恢复计划已生成，原完整Trainer保留，尚未上传/下载。AMD仅CPU；4080节点现场报告32/16GiB两卡，查询时无计算进程，未分配GPU。[实际环境、认证、传输准备与保守复算](reports/PI0294/design/parallel-science-20261022/V4A_CORELAB_NEURAL_ENV_AUTH_AND_MODEL_TRANSFER_PREPARATION_ACTUAL_20261024.json)。Core进程分配/原账本once会计、fresh准入及原600s/80轮资格仍待；不伪造Vanda PBS身份。A/B DEV仍0，core64=0，不自动训练/采数，目标ACTIVE。

V4A接续：当前提交门只允许A/B同bank294101各24条件，最多2份原面板，后续B0／第二bank不自动派发；已采用大额cap保持。[冷态接线与现场终点元数据](reports/PI0294/design/parallel-science-20261022/V4A_PAIRED_STAGE_NATIVE_AND_CONTROL_COLD_CPU_20261024.json)。原matrix／final及截止保持；两份完整阶段和第二次提交时钟均检查，真实标准入口／驻留owner与同job资格仍待，尚无新GPU或Eval结果。

> **2026-10-24 两份并行修订已统一优先级（防止重复派发）：** [V4A：现行可执行的最小原DEV配对阶段](docs/protocols/GR00T_WBC_V4A_EVAL_FIRST_STAGED_EXECUTION_20261024.md)＝同一个原噪声bank294101，A/B各24条件、最多2个原生PBS，**仅在两个完整面板的真实资源与截止均通过后**执行；不自动申请/使用130条评估。此前[V4.1：16次小诊断备选](docs/handover/2026-10-24/SCIENCE_FIRST_MINIMUM_SPEND_EVAL_V4_1.md)因现有runner强制24格完整面板，当前状态为**备选设计、未获子面板原生准入，不是第二个可并行派发的实验**。不得两方案同时提交、重复有效DEV或提前训练。global121GPUh/34PBS是已采用cap非消耗目标；两张主表仍待行为结果。

> **最新执行优先级修订 V4A（2026-10-24）：先用现有 A/B4096 在同一原封存 DEV 噪声面板完成配对闭环 Eval，再由真实 B−A 结果决定是否扩展评估或提出新的训练干预。** [V4A 分阶段最小评价执行计划](docs/protocols/GR00T_WBC_V4A_EVAL_FIRST_STAGED_EXECUTION_20261024.md)；原121GPUh／34PBS／213CPUh／810GiB global 上限已真实采用，但仅是安全预算天花板，绝不要求先消耗完整130条件或注册全部六个S2面板。没有新增GPU作业／资格，也不撤销原已采用账本；先保证必要 native 安全、B4096 精确复用准入、矩阵／最终保护和原截止。下方旧 S2“完整执行包”只是当时批准的最大作用域，不是当前自动派发清单。

> **历史并发V4.1低成本诊断备选（不属于当前原生派发）：[先做小规模配对 Eval，再决定扩大评估／修改训练](docs/handover/2026-10-24/SCIENCE_FIRST_MINIMUM_SPEND_EVAL_V4_1.md)。** 已批准并实际采用的global121GPUh／34PBS等是上限，**不是必须花完的执行目标**。新A/B DEV仍0/48、旧B0 14/48、core64=0；优先冷态资格下的4几何×2任务×A/B＝16条新行为诊断，结果出现前不扩训练。原完整DEV和S1/最终保护不降低；8格子面板目前**尚未经原生支持/准入**，不允许从旧24格作业偷偷截断。本次GitHub发布不启动任何作业或重算已采用账本。

S2整包已真实一次采用：[服务器＋LOCAL独立回读](reports/PI0294/design/parallel-science-20261022/V4_S2_ACTUAL_DUAL_ONCE_ADOPTION_20261024.json)。global121GPUh／34PBS／213站点CPUh／810GiB及精确B4096只读DEV例外已双账本落地；25条closed原记录、全部失败成本／训练进度、DEC1／其他限额／matrix／final／时钟均未改变。连接消失与首次非空队列只读拒绝保留，任何未知写未重发。账号另有1481491、CFP04-CF-031双GPU作业，人类已明确确认属其他独立工作；只精确区分该作业，不使用／修改／停止或计入本实验，其他未知作业仍拒绝。**无新GPU／PBS／模型／物理**；继续fresh typed native admission与独立DEV release，core64=0，B0有效14（9成功5失败）、A/B各0，goal ACTIVE。

S2整包已获人类明确批准：[精确授权](reports/PI0294/design/parallel-science-20261022/V4_S2_GLOBAL_RESOURCE_AND_SAVED_B4096_REUSE_AUTHORIZATION_20261024.json)／[冷态采用与单面板修复](reports/PI0294/design/parallel-science-20261022/V4_APPROVED_SCOPE_AND_SINGLE_PANEL_COLD_CPU_20261024.json)。仅global GPU121h／PBS34／站点CPU213h／output810GiB及精确B4096只读DEV例外；DEC1、其他限额、原matrix／final和截止保持。原25条真实closed账本字节纯采用／临时once验证通过，全部费用、训练进度及owner Exit1保持；24新增CPU、完整PV1 568／249子测试通过。成熟DEV循环只运行该请求封存的完整remaining噪声面板，保留全部有效负结果。**当前仍未真实native采用、未新reserve／PBS／模型／物理**；下一项双账本一次采用及独立回读、fresh完整准入。core64=0，B0有效14（9成功5失败），A/B DEV各0；goal ACTIVE。原保守全量上界不能在原剩余时限内容纳，未运行项保留NOT_RUN，不延时。

V4完整remaining-work重算与原global once reserve已冷接线：[CPU及真实原件复算](reports/PI0294/design/parallel-science-20261022/V4_FULL_RESOURCE_AND_ORIGINAL_ONCE_RESERVE_CPU_20261024.json)／[S2整包资源与精确B4096复用草案](reports/PI0294/design/parallel-science-20261022/V4_S2_COMPLETE_RESOURCE_AND_B4096_REUSE_SCOPE_DRAFT_20261024.json)。解析全17历史，真实global／LOCAL原字节给出paid／reserved下界，只允许精确已bound R请求的实际GPU／query／CPU／output／PBS抵扣一次；不接受caller cap、旧快照或UNKNOWN重试。原manifest／request双封存的history文件、完整项目余额／队列／queue上界／全部未来输出quota，在原owner lock＋once CAS前验证；临时成功／重复拒绝／post-intent UNKNOWN保持与只读reconcile均过。51新增CPU，完整PV1 544／249子测试通过；实际25条已结算global＋17原件复算四缺口仍为89288GPU秒／6PBS／189889.659站点CPU秒／139546445824bytes，真实reserve正确在intent前拒绝，历史capture在当前时间被拒绝。**没有真实新reserve／PBS／模型／物理或native账本写；尚未部署。**四项拟增global cap与B4096只读Exit1复用例外均仍未批准／未采用，DEC1／其他上限／原matrix与final／原全部截止保持；全保守上界不容纳于原剩余窗口，不延时或承诺全量完成。付费control credit、fresh producer与完整native admission／独立DEV release仍待；core64=0，B0有效14、A/B DEV各0/48，goal ACTIVE。

V4的H200硬编码启动阻断已冷修：[A40评估bootstrap／实际设备与CPU上限接线](reports/PI0294/design/parallel-science-20261022/V4_A40_EVALUATION_NATIVE_BOOTSTRAP_AND_DEVICE_CPU_20261024.json)。独立Vanda DEV请求走完整自有队列＋同global已绑定row，复用原始作业字段检查，不制造bootstrap admission PASS；加载前按typed receipt严格核对单张真实A40，训练／H200冒充拒绝，原H200路径仍保持。80000CPU秒accounting upper落实79760进程上限并取实际remaining较小值。50新增CPU用例；相关212／8子测试、完整PV1 493／249子测试通过；真实25条旧账本全保留的隔离临时fixture中，原三项matrix仍被PBS上限正确保护。完整17次历史已于b1bc5ac0现场回读并解析，14有效＝9成功＋5失败／3工程原件保持。**尚未部署／未提交／无新模型、物理或native账本写；不是完整资源或原生准入。**下一项为完整remaining-work fresh重算／精确credit、原global once reserve及native admission／独立DEV release；四项S2 scope未批准／未采用，B4096复用边界待。core64=0、A/B DEV各0/48；原时钟／matrix／final保护、LOCAL已用一分钟和两SDK UNKNOWN保持，goal ACTIVE。

V4完整历史原件已装配：[17次原始attempt与剩余工作实际回读](reports/PI0294/design/parallel-science-20261022/V4_COMPLETE_ORIGINAL_PRIMARY17_HISTORY_ACTUAL_20261024.json)。三个原CLOSED child逐目录库存精确17；原RESULT／adjudication／audit／CARD／model-ready共180353bytes现场只读回读、Core CPU0.009881秒，控制与关闭metadata未变。全部17通过同一逐attempt解析；14有效＝9成功＋5失败，3工程失败原件保留；两个既有replacement已有有效结果，F2/000/T1仅保留原REPLACEMENT-01资格且未派发。旧V2诊断只增加有来源的命名digest索引视图，原字段／原件不改；9项矛盾拒绝。由全部17推导B0剩余34、A/B各48、原面板10/24/24/24/24/24，共130；完整DEV48训练／正式选模仍拒绝partial，未重跑14有效条件。LOCAL一分钟已采用并消费，不重复采用；两SDK UNKNOWN、原时钟／matrix／final保护保持。**新增科学条件0、core64=0、A/B DEV各0/48；无新模型／物理／PBS／native账本写。**下一项为原global冷owner的完整fresh资源／once reserve及A40原生准入，四项S2 scope未批准／未采用、B复用边界仍待；goal ACTIVE。

V4独立评估的剩余工作推导已有CPU实现：[原始partial history与完整regular work](reports/PI0294/design/parallel-science-20261022/V4_PARTIAL_PRIMARY_HISTORY_AND_COMPLETE_REGULAR_WORK_CPU_20261024.json)。原完整DEV48训练／正式选模入口继续要求48；逐attempt校验抽为同一实现，新V4历史入口接受合格partial但不授予training／selection／native权限。旧published实现与重构后对B0／A／B三份完整负结果fixture的canonical返回逐字节一致；两份真实旧B0原件＋原LOCAL causal joined结果通过新解析，旧训练门仍因不满48拒绝。当前真实B0仍14，2仅是缓存原件子集资格；完整17attempt库存仍待装配，不把2误写为全量。41用例（初次39＋修正fixture字段1＋新增耗尽工程例1）／69子测试通过；regular剩余10/24/24/24/24/24共130的全部费用上界和四项缺口与原scope一致，矩阵及final保护不减，proposal fit不等于授权。**未部署／无新native probe、job、模型、物理或账本写**；fresh全资源／精确credit／原生owner与B复用待，core64=0、A/B DEV各0/48，原截止和全部历史保持，goal ACTIVE。

V4评估接续的扣费／终态链已接到成熟runtime：[CPU与临时once验证](reports/PI0294/design/parallel-science-20261022/V4_A40_EVALUATION_ESCROW_DEBITS_AND_ONCE_TERMINAL_CPU_20261024.json)。同global按request/job/actor记唯一effect，运行时转移query／CPU预留，F结算只加未收费差额，GPU／模型／输出与原失败成本保持；重复bind／settle和once写拒绝，UNKNOWN query保留hold／marker，旧UNKNOWN不清／不重发。83针对性及115回归／72子测试通过；原25条真实账本记录在未mock旧grant验证的临时once生命周期中完整保留，6次临时CAS／只读reconcile均过。原H200路径与模型／query journal不改；LOCAL和两SDK UNKNOWN真实字节不变。**没有真实新reserve／qsub／模型／物理／账本写，未部署**；V4独立S2全剩余资源／项目／quota／queue准入、owner reserve、A40原生bootstrap与B复用仍待，完整B0不重新成为A/B科学结果的前门。core64=0，B0有效14/48、A/B DEV各0/48；四项S2 scope未采用，原所有截止与最终保护不变，goal ACTIVE。

V4源评估冷接线新增两项完成：[Vanda冷scope与同global非训练分类CPU结果](reports/PI0294/design/parallel-science-20261022/V4_A40_COLD_SCOPE_AND_GLOBAL_EVALUATION_CLASSIFICATION_CPU_20261024.json)。标准entry现识别独立Vanda评估profile并保留原source／15数值pin／数据／租约／registry；必须使用原556723byte完整人口而非公开摘要，原DEV48、既有模型与封存prior不能改成方便子集。A40 DEV单列global分类，旧compat与训练chain上限不增，不借DEC1子额度、不另建账本，GPU／CPU／query／output完整hold和单owner保留。138用例／38子测试通过；实际封存人口原字节未mock纯读验证、原25条native账本记录全部保持；临时26序号的一次一致性检查不构成准入，加回三项matrix时仍由原PBS28上限拒绝。LOCAL与两SDK UNKNOWN字节不变。**未部署／无新job／模型／物理／账本写**；实际allocation与fresh全剩余资源、once生命周期、GPU bootstrap／严格CPU上限和B复用仍待，四项S2完整scope仍未采用；core64=0、B0有效14/48、A/B DEV各0/48，原截止和所有保护不变，goal ACTIVE。

V4评估接线又完成一个必要组件：[A40实际作业身份与成熟资格入口CPU验证](reports/PI0294/design/parallel-science-20261022/V4_A40_EVALUATION_CURRENT_JOB_IDENTITY_CPU_20261024.json)。新分支独立标识Vanda DEV，封存原B0／A4096／B4096与R0，核对完整自有队列、实际R作业／project／nonce／host／capacity／start-end；训练allocation和H200不能换标签通过。75用例（53身份用例；原46分项计数已纠正）／26历史子测试通过，LOCAL账本与两SDK UNKNOWN字节不变；原600s／80cycle不放宽且未重跑。LOCAL一分钟已双端采用、零步IO已关闭，不重复采用或重放。**组件仅CPU验证，未部署／无新PBS／模型／物理／账本写**；完整冷profile、实际资源准入、同global非训练分类及GPU bootstrap／B复用仍待。S1 core64=0，S2 B0有效14/48、A/B DEV各0/48，四项S2完整scope草案仍未采用；原时钟／所有保护保持，完整goal ACTIVE。

V4评估冷接线新增一项已修复阻断：[原24序号硬编码的CPU修复](reports/PI0294/design/parallel-science-20261022/V4_EVALUATION_ADOPTED_PBS_NAMESPACE_COLD_FIX_CPU_20261024.json)。现有全局账本已25且精确批准到28；transport普通／FAULT namespace与current-admission查找现在读取已采用的上限，未增PBS或预算。29用例／26历史子测试通过；临时文件系统用原生完整账本原字节、未mock授权验证，25–28通过、29／31／34及伪造proposal／篡改marker拒绝。原H200／A40资格区别、真实600s／80cycle、模型／安全／截止门未放宽。**这只是序号接线，完整Vanda评估桥仍未完成／未部署**；下一项按同一成熟runtime补cold profile与A40实际作业身份，当前不提交新作业。S1 core64=0；S2 B0有效14/48，A/B DEV各0/48；四项S2全scope增补仍只是未批准草案。

V4两张主表仍未完成：[S2逐条件真实partial](reports/PI0294/design/parallel-science-20261022/S2_DEV48_ACTUAL_PARTIAL_COMPARISON_20261024.json)／[完整130条件费用上界草案](reports/PI0294/design/parallel-science-20261022/S2_DEV130_CLOSED_COST_AND_FULL_UPPER_SCOPE_DRAFT_20261024.json)。S1 core64=0，source当前mapped-prefix／ACK与target完整prefix／安装未资格。S2 B0按任务为T1已测7＝5成功＋2有效失败、T2已测7＝4成功＋3有效失败；A/B DEV各0/48，不能计算B−A或相对完整B0保持效应。服务器只读已核14份原RESULT／WORKER_RECEIPT／adjudication，9份完整32秒的最大wall890.981s／Core447.637CPU秒／输出90.367MB；这些是旧平台观测，不是A40全链吞吐。已关闭1481574的17个原TRAIN回执表明A40首次10.507s、warm中位0.306s／最大0.312s，未重跑，不能据此宣称GPU利用率或闭环资格。新鲜Vanda账本确认global25已终态／无pending-update或global unknown；原两SDK UNKNOWN字节保持。按6个原噪声面板各完整8h的保守方案，保持全部matrix／final保护后，global仅四项超额：GPU89288秒、PBS6、站点CPU189889.659秒、站点输出139546445824bytes；向上整定草案96→121GPUh、28→34PBS、160→213站点CPUh、680→810GiB，DEC1上限不改。该方案不是最少作业数／不是授权，S1＋S2上界72.15GPUh也不被当前原窗口完整容纳；原截止不延长，到期未运行项如实NOT_RUN。下一动作只做成熟评估入口的最小Vanda冷绑定与实际可容纳阶段核验，不再扩展vendor逆向，不新训练／新数据／重放旧IO。

V4已实际接续：[两条科学线与必要缺口](reports/PI0294/design/WDR1_V4_SCIENCE_FIRST_ADOPTION_AND_BOUNDED_BINDING_ACTUAL_20261024.json)。S1仍core64=0，目标完整前缀／参考安装和source当前资格未闭合；S2保留B0有效14/48（9成功、5失败），A4096／B4096训练已完成、DEV各0/48。剩余130条件需10400查询及4160000 live+shadow ticks，不能借原核心矩阵7040查询。global已用25/28 PBS，余3份全属原matrix；保护matrix86580GPU秒和最终72000后另余83872GPU秒，独立源评估完整CPU／IO／平台／提交覆盖仍须一次核清，当前不是准入。此前已派发的只读链接检查实际完成：83函数／3275字段，3251局部地址／常量或PLT-GOT名字已核对，0已检查不一致，24局部引用未解析；动态DSO实现和完整状态ABI仍未资格。停止通用链接／vendor逆向扩展，只补直接解除原生prefix接缝的必要项。LOCAL一分钟已用且原6600未来不减；两SDK UNKNOWN和全部历史保留，无新模型／物理／PBS／账本写，完整goal ACTIVE。

## 当前优先 Goal：科学主线优先 V4（2026-10-24）

**[执行本轮长程 Goal：先交科学结果](docs/protocols/GR00T_WBC_SCIENCE_FIRST_LONG_HORIZON_GOAL_V4_20261024.md)**。当前最多两条在制科学线：固定旧CORR/R0的DEC1核心64格，以及独立的B0／A4096／B4096源域DEV比较。两执行器和接口合格时优先核心64；目标受阻不暂停已合格源域工作。完整负结果照常交付，18/24是PV1发布门，不是科学结果前门。

事实复核至 **24a2bc794bcbfcb23f769a4a62696051d8e85816（2026-10-24 13:05:24 SGT）**：A/B训练终点已完成；B保存审计及资源关闭完成，后续复用边界仍待闭合。目标真实零步feedback／双相机／STOP已完成并关闭；零步未进入WBC工作模式，安装／完整prefix／T1T2尚未资格，core64=0；部分实际链接绑定不替代此资格。下一项是复用已有接缝补齐必要动作执行链，取得原科学矩阵，已关闭零步请求不重跑。

当前收尾窗口暂停新训练候选与扩采；world model、BASE和更多泛化移出当前日程。LOCAL一分钟已采用并消费，原时钟与所有已用成本保持。V4更新优先级，不新增计算额度；旧V3与01:46交接保留为历史和设计背景。

下方按原时间保存历史记录；接续时使用最新同一身份链的原生回执。

---

完整目标继续推进：[原RL编译库真实重定位绑定只读结果](reports/PI0294/design/WDR1_FROZEN_RL_EXACT_LINK_BINDING_ACTUAL_20261024.json)。新source542fbbd1已普通push／四份全文回读，实际47个含重定位函数／2446字段中，858个具名链接目标精确匹配、已解析项0不一致；1587个外部／局部section与常量目标未解析、1个GOT类未支持，实际具名常量完整字节验证0。不能把旧52个屏蔽重定位函数体匹配或当前858局部绑定当成完整header/state ABI、腿历史snapshot／prefix安全。10初始／11当前CPU测试包含错误call仍过masked-body却被精确绑定拒绝、常量内容替换拒绝，以及NOBITS不冒充文件常量。新Core只读CPU0.066413277秒，控制metadata／Core／LOCAL账本与两SDK UNKNOWN均不变，模型／构建／物理／原生写0；未重跑已关闭零步IO。下一项精确解析PLT/GOT和局部section／常量，再核验原状态ABI；不猜offset、不重编WBC，不用12D MPC替代完整全身prefix。原生零步feedback／双相机／STOP已真实通过／关闭；安装ACK／完整prefix／T1T2／CORR-R0 core64仍未完成，完整goal ACTIVE。

原冻结目标的服务器零步IO已真实跑通并一次关闭：[实际反馈／双相机／STOP与独立关闭](reports/PI0294/design/WDR1_ORIGINAL_NATIVE_ZERO_IO_FEEDBACK_STOP_CLOSED_ACTUAL_20261024.json)。source802886e4／request45c7a884／manifestfafa2a0b；311封存成员现场回读、新鲜完整8D与全未来预算通过，单一有限后台monitor自运行，worker自然Exit0。实际700条odom／699条arm反馈，原生暂停step0/time0、双相机、持久STOP及真实ACK，observer已停止且0积分／157快照；7构造尝试、worker+children14.299281312CPU秒。Core614.299281312／LOCAL60及原IO金融upper一次结算，lease CLOSED、独立actor0／全部IO hold0，旧两次IO失败和SDK两UNKNOWN保持。人类批准的LOCAL57600→57660已真实双once采用／独立回读，其余预算与原时钟不改；已用LOCAL51058.974682509，保留原未来6600后余1.025317491秒。原WBC／求解器数值不改，原107依赖封存；Actions账单恢复不再是前置。**本轮actual_mode_WBC_before_pause=false，policy-prefix0／reference-install未资格；完整全身prefix／安装ACK／T1T2／固定CORR-R0 core64仍未完成**。不得重跑已关闭零步请求或把零步组件PASS当任务PASS；下一项沿现成原生IO接缝补齐完整目标安全与策略安装闭环，完整goal未达成。

人类已明确批准仅增加1分钟LOCAL CPU：[精确授权](reports/PI0294/design/WDR1_ONE_LOCAL_CONTROL_MINUTE_AUTHORIZATION_20261024.json)。新冷合同仅将DEC1 LOCAL57600→57660，原VAP1授权原件和其他额度／时钟／全部历史费用不改；须真实Core/LOCAL一次采用并回读。新的第三次零步IO只绑定两份精确CLOSED工程失败，保留原source／request／manifest／result／actor／独立关闭回执，禁止第四次或安全失败重放；原八维预留和全部三项矩阵／控制上界保持，现场须再次核对真实旧lease/result。原107份ELF依赖目录与文件在新的输入封存中逐项绑定，不重编WBC。129针对性、全DEC1 1861＋647子测试通过（13私有parser skips）；临时完整once生命周期、精确source回读／服务器CPU验证、实际双账本采用和全新fresh准入仍待，尚无新注册／模型／物理或GPU。完整prefix／任务／core64仍未资格，Actions账单恢复不再是依赖。

按用户直接指令，Actions账户限制已不再是原生执行前置：[服务器实际接续与动态库冷修复](reports/PI0294/design/WDR1_SERVER_QUALIFICATION_ACTUAL_NATIVE_IO_ELF_REPAIR_20261024.json)。bc26运行源码260份完整tree/blob绑定，服务器标准entry实际18CPU用例通过；新请求b1cf63f5一次完整准入／注册309文件、服务器有限后台monitor已实际运行。ROS/rclpy问题已修复，receiver已启动；原sim／solver／controller随后因原库搜索路径缺GLFW／Pinocchio／OctoMap退出，物理0，未获feedback／STOP或任务资格。五次constructor尝试与实际worker+children1.903223959CPU秒如实保留；Core601.903223959／LOCAL60和原IO上界一次计费／lease CLOSED，独立actor0／hold0／两次SDK UNKNOWN不变。精确只读已确认补原目录后107份依赖全部解析，不缺安装包、不重编WBC；冷修复53针对性CPU测试通过，尚未新注册或原生采用。原LOCAL57600秒减已用50998.974682509和全未来6600后仅余1.025317491秒，不能暗删原未来upper来容纳再一次完整60秒控制预留。Actions失败仍保留；完整prefix／T1T2／CORR-R0 core64未资格，目标ACTIVE，全部原截止保持。

用户已直接要求先搁置GitHub Actions账户限制并优先服务器端执行：[验证方式调整](reports/PI0294/design/WDR1_USER_SERVER_VALIDATION_ACTIONS_DEPENDENCY_CHANGE_20261024.json)。当前冷态零步IO接续允许用精确源码全文回读、完整本地CPU回归及真实标准entry服务器CPU验证，替代Actions启动依赖；旧CI失败保留，不能伪造成功。此调整不增实验资源、不放宽原生安全／完整prefix、不授权重发两次SDK UNKNOWN，也不修改原时钟或WBC。新入口验证接线已通过34针对性CPU用例及完整DEC1 1844项／647子测试（13私有parser skips），现场验证仍须实际通过；当前未新注册／预留／启动IO，完整任务资格仍false、core64=0。下方旧“必须恢复Actions”是历史记录，由本次直接用户指令覆盖。

最新针对性只读ABI检查：[精确冻结产物与原生CI门记录](reports/PI0294/design/WDR1_FROZEN_TYPE_LAYOUT_DEBUG_READ_AND_NATIVE_CI_GATE_ACTUAL_20261024.json)。冻结sim与对应对象、四份原RL／history／inference／motion库的4个ELF成员均无类型布局调试段／外部调试链接；仅覆盖这些hash-bound原件，不推断全站资产缺失。Core只读CPU0.011294069秒，控制metadata及Core／LOCAL账本不变，两次SDK UNKNOWN保留，构建／模型／physics／GPU／PBS／原生写0。52个屏蔽重定位的函数主体匹配仍不能证明完整header-state ABI，原state journal不能冒充腿历史完整snapshot，不猜私有offset或重编／替换WBC。已准备的5671冷态接续仍被精确CI失败（40052632537／120748430019，无runner／steps）拒绝，账号侧启动限制连续三轮确认且须账号owner恢复；本地CPU验证不代替CI或原生资格。下一真实步骤仍为精确CI及fresh完整准入后的原生feedback／STOP，再补齐完整prefix／安装ACK／T1T2／原CORR-R0 core64；完整目标未达成，原预算／时钟／历史失败／checkpoint与三项矩阵额度保持。

冷态接续的完整captured临时文件系统验证已完成：[实际CPU记录与CI启动阻碍](reports/PI0294/design/WDR1_ZERO_IO_COLD_CAPTURE_AND_ACTIONS_STARTUP_BLOCK_ACTUAL_20261024.json)。runtime5671da2b已普通push，29份全文／Git blob／本地commit一致；259源码／307封存准备成员，原完整native registration＋onceIO在/tmp注册308文件，重复register拒绝；真实本地无root标记的良性进程组仍能阻止close且无账本改变，已wait；原完整settle／lease close＋onceIO在/tmp通过、重复结算拒绝、八维hold0。全部是假时钟隔离CPUfixture，CI原本failure保留，未伪造CI成功／native准入；真实LOCAL与两次SDK UNKNOWN字节不变，真实新seed/request／registry／预留／lease／模型／physics／GPU／Vanda写0。保留全部原三项matrix与controller future LOCAL6600及新IO完整60，captured纯预算剩余1.025317491秒，不能当成fresh准入。新CI40052632537／job120748430019 failure、无runner／steps。另已从用户已有登录会话确认原19bf Actions账号侧启动限制，需账号owner恢复Actions；不再需要抄错误文字，未修改任何账号设置，账号私有细节不进入仓库。完整prefix／安装ACK／T1T2／CORR-R0 core64仍未资格，目标ACTIVE，原全部预算／时钟／历史与checkpoint不变。

新增首次零步IO已闭合、实际构造0之后的冷态接续合同：[CPU准备与真实原件解析](reports/PI0294/design/WDR1_ZERO_CONSTRUCTOR_IO_COLD_CONTINUATION_CPU_20261024.json)。只接受原e7cf请求／8e2源码／cb78manifest的五份精确原件（dual／lease／result／actor／独立双回读），真实85341byte packet已用未改公共hash解析；不接受UNKNOWN、已构造模型或安全失败，保留原收费与CLOSEDlease，只容许一个新冷接续、最多一个active、禁止第三次与旧request-child重放。新manifest及row保留完整predecessor，native导入前须再次读取原实际lease/result；完整8D、三项原matrix及controller上界、新鲜全剩余预算不减，IO无credit。新冷wrapper复用既有finite monitor／onceIO，绑定原ROS环境并计入旧IO output，不读取无关proc environ、不新增低层runner/controller。17新／82原相关及全DEC1 1826＋647subtests通过（13私有parser skips）；实际failed-CI proof正确在动作前拒绝。未创建新seed/request／BASE／预留／registry／lease／模型／physics／GPU或Vanda写。精确新CI与完整captured临时文件系统接续仍待；此前19bf CI无runner的启动阻碍继续等待页面错误信息，两次SDK UNKNOWN保持不重发。完整目标ACTIVE，target/core64资格0，原限额／时钟／失败／checkpoint不变。

最新精确CI启动阻碍：[实际记录](reports/PI0294/design/WDR1_ROS_SDK_CPU_REPAIR_EXACT_CI_STARTUP_BLOCK_20261024.json)。runtime19bf9505已普通push，26份全文／Git blob／本地commit一致。CI40051437887两次job120744780171／120745181895均failure且runner空、steps空，未执行任何测试；已知失败job只重试一次，日志404，无可用启动注释，具体原因仍UNKNOWN，不能归因代码或账单、不能拿本地测试替代CI。已向人类请求检查该任务页面的具体启动错误。两次SDK UNKNOWN及原字节／无intent-receipt-temp／actor0证据完整保留，不重发；CPU修复后的第三冷边界仅prepared，须新的明确例外及精确CI后才能一次执行。ROS冷修未原生采用，首次零步IO已真实一次结算／独立双关闭，实际模型0／physics0。完整目标仍ACTIVE，target／core64资格0，预算／时钟／checkpoint及全部历史不变。

最新SDK新冷边界执行也返回255，独立精确回读确认双账本原字节、原生intent／receipt／temp均不存在及actor0，两次UNKNOWN全部保留，绝不自动重发。[实际CPU限额诊断、修复及新的单次例外草案](reports/PI0294/design/WDR1_SDK_SECOND_UNKNOWN_CPU_LIMIT_REPAIR_AND_ONE_COLD_EXCEPTION_DRAFT_20261024.json)：删去锁及全部写段的真实只读复现，在原生context已用8.72CPU秒后设置整个进程8秒绝对限额，SIGXCPU返回88；不是据SSH255推断网络失败。新纯转换保守地计入先前进程CPU，追加软／硬时间仍不超过原8／10秒、RPC40秒与所有预算不变；11CPU用例及7新／7历史captured guard通过。新helper须明确新例外及精确CI才可执行，尚未原生采用。另已精确定位原Jazzy／Python3.12环境，六个ROS包metadata均可从原setup环境解析；[冷修代码与证据](reports/PI0294/design/WDR1_ORIGINAL_ROS_ENVIRONMENT_COLD_REPAIR_CPU_20261024.json)显式pin原路径／ABI／module及扩展库，完整准入后同解释器一次exec，不执行setup shell、不改WBC。13新／176相关CPU及完整DEC1 1798＋647subtests通过（13私有parser skips）；代码CI待核验，没有新IO重试或原生任务资格。首次零步IO已一次结算／独立双关闭，0实际构造／physics／GR00T。完整WDR1仍ACTIVE，core64=0，原限额／时钟／全部成本与checkpoint不变。

当前零步目标IO已真实进入标准entry，但在模型构造前因原ROS rclpy环境未绑定而失败：[真实失败、一次结算及关闭](reports/PI0294/design/WDR1_NATIVE_ZERO_IO_ROS_IMPORT_FAILURE_ONCE_CLOSED_20261024.json)。source8e2bcd7d精确CI40049244582／job120737940405全部12步success，20份全文／blob一致；305份现场输入回读。实际constructor0／physics0／GR00T0／GPU-PBS0，自有observer编译完成、worker及children CPU0.470118811s；Core600.470118811／LOCAL60CPU及原金融constructor8／IO上界已一次计费，不能称8个实际模型。lease CLOSED、独立Core/LOCAL八维hold0和真实登记actor0；原环境读取权限错误保留，终态由PID/start/group和命令回读确认，不靠超时推断。继续精确绑定原ROS环境、冷修工程问题，不重放原request／热改capsule。人类已明确批准SDK已知零效果的新冷边界一次分类：[精确授权](reports/PI0294/design/WDR1_SDK_KNOWN_ZERO_ONE_COLD_CLASSIFICATION_AUTHORIZATION_20261024.json)，原UNKNOWN仍保留，新分类尚未执行。完整prefix／T1T2／CORR-R0 core64仍未资格，完整目标ACTIVE，预算／时钟／历史及checkpoint不变。

冷入口47c1bde5精确CI40048601126／job120735920361全部12步success，20份全文／Git blob／本地commit一致。进一步限定复用原Xvfb/Xauth dependency PATH，并要求observer实际stopped回执才可判零步组件通过；61相关CPU测试通过。新的精确只读绑定确认14份MuJoCo headers和SDK05已闭合的30份原数学缓存（63084bytes），无现场编译／模型／物理／写入。初读工具PATH错误原样保留，已按既有构建的精确工具路径修正。当前仍未注册／预留／启动新IO；最终冷绑定、全资源新鲜准入和现场STOP必须实际完成，完整prefix／T1T2／core64仍未资格，目标ACTIVE；原SDK UNKNOWN不重发，全部预算与时钟不增。

当前目标零步冷入口已实现：[入口与原冻结进程图CPU结果](reports/PI0294/design/WDR1_CURRENT_NATIVE_ZERO_IO_COLD_ENTRY_CPU_20261024.json)。标准entry／封存源码／当前lease／完整双账本8D及全剩余资源重算均先于native导入；复用原sim、solver、clock controller、receiver、ActualState和FrameRecorder，不调用旧R4 run，不发送ENTER或正配额。进程均有一次intent及真实PID/start，有限清理不依赖agent在线。原task会重编数学库，因此明确绑定已资格d0c485bc缓存加载派生（仅recompile true→false，原0b69及数值参数分别保留）；全部数学缓存／实际327 header closure只读封存，仅编自有observer。16新入口／180相关及完整DEC1 1785cases＋647subtests通过（13私有parser skips）。代码未原生采用，无新child／模型／物理／GPU／PBS；现场feedback／STOP仍待实际准入，安装ACK／完整prefix／T1T2／CORR-R0 core64未资格，目标ACTIVE。原SDK UNKNOWN例外仍待明确答复，不重发；预算／时钟／历史和checkpoint不变。前e399文档CI40047047639已completed/success，当前新代码CI待发布后实查。

精确资源代码53c6061a已CI40045534775／job120726262927全部12步成功，13份源码／依赖／文档全文及Git blob一致。新的原冻结构建证据定位到实际pause链接使用的4份小型原RL／history／inference／motion编译库：[原编译库接续证据](reports/PI0294/design/WDR1_EXISTING_FROZEN_RL_LINK_ARCHIVES_ACTUAL_20261024.json)。52个导出数值函数的指令主体在屏蔽原重定位字段后与冻结binary一致，0 mismatch；这尚未验证重定位目标／常量和完整header-state ABI，不能冒称整个controller一致、full-state replay或prefix资格。下一项沿已有编译库资格化IO接缝，不重写低层controller、不以未证vendor源重编WBC。两次精确只读Core／LOCAL账本及控制metadata均不变，构建／模型／物理／GPU／PBS／原生写入0。原SDK UNKNOWN仍保留，不重发；当前资源代码和零步组件尚未原生采用，完整prefix／T1T2／CORR-R0 core64仍0，目标ACTIVE，所有原预算／时钟／历史和checkpoint不变。

零步目标检查已补齐完整8D双账本预留／实际终态结算的纯合同：[完整资源合同CPU结果](reports/PI0294/design/WDR1_NATIVE_ZERO_IO_FULL_RESOURCE_CONTRACT_CPU_20261024.json)。Core3600／LOCAL60CPU秒、金融constructor8、Core512MiB／LOCAL64MiB／transfer128MiB，episode／physics／GR00T／GPU0；仍须原cap内全剩余预算，不加额度。保留全部原矩阵及三项控制上界，不给新IO任何CPU或其他维度credit；旧SDK只用已资格same-work CPU credit，未镜像费用显式入算、已镜像marker精确去重，不能假称落账。24预算＋42预留终态新用例、179相关CPU用例通过。新的Core／Vanda完整只读均成功，LOCAL九namespace实际1840586752bytes；协调器在成功读取后误读invoke返回结构，已修复并仅续完本地pool核算，未重复两次原生读取，原UNKNOWN不变。不是fresh原生准入PASS：当前source CI／完整冷入口／fresh八维及真实现场ACK-STOP仍待。完整前缀／T1T2／CORR-R0 core64=0，目标ACTIVE，预算／时钟／checkpoint及全部失败不变。

目标零步观察组件已实现：在原冻结暂停 ACK 的 rename 返回前、原 native sim mutex 仍持有时复制完整 GRTSTATE／contacts；不从轮询线程读取 live mjData，不猜 C++ 对象布局。只接受原 main executable 的暂停方法／内联 C ABI caller，独立拒绝6个公开积分入口，删除状态前必须停止 observer，不改原 task_actuation 或 WBC。20项O2编译隔离mock及176相关CPU用例通过（12私有parser skips）；测试假时钟只在隔离fake库，不是现场输入。新精确只读证据确认冻结binary链接MuJoCo3.2.7、导入mj_step／rename且暂停方法1183bytes，10个函数签名匹配实际header；Core／LOCAL账本和控制metadata不变，模型／物理／现场构建／worker／写入0。详见[原生零步观察组件与实际ABI证据](reports/PI0294/design/WDR1_ZERO_INTEGRATION_NATIVE_OBSERVER_CPU_AND_ACTUAL_ABI_20261024.json)。尚未原生采用：全8D、全剩余资源、真实header／binary seal、receiver反馈／安装ACK／STOP与一次结算仍须接续；原SDK UNKNOWN例外仍待答复，不重发。原046ff302精确CI40040948760全部12步成功／10份全文blob一致，当前新source CI须发布后实查。完整prefix／T1T2／CORR-R0 core64未完成，目标ACTIVE，全部预算／时钟不增。

目标原生接线新增零物理 quota／STOP CPU 组件：只能一次 arm 配额0，拒绝全部正向配额；逐次核验封存 request，STOP 仍需真实零步完整状态、原冻结 clock、FSM／腿历史／gait 不变、新鲜 ACK 与持久回读。29新用例及156相关用例通过（12私有 parser skips），旧正向配额语义保留，旧 pending／intent 拒绝新 owner。它不是当前原生准入，尚无新 Core child／构造／物理／GPU；还须完整双账本8D及全剩余资源、原生逐入口零积分保护和现场 receiver／反馈／安装 ACK／STOP 接线。完整 prefix／T1T2／原 CORR2048-R0 core64 均未通过，不用零物理检查替代。原SDK费用UNKNOWN及已发新冷边界例外继续等待明确答复，不重发、不重复询问。原4b5edda1五文件全文／blob回读及CI40039484020全部12步成功，当前新组件精确CI待发布后核验。详见[零物理配额接线CPU准备](reports/PI0294/design/WDR1_TARGET_ZERO_PHYSICS_QUOTA_STOP_CPU_20261024.json)。完整目标ACTIVE，原预算和截止不变。

target四条件代码22eb5217精确CI40037256890／job120699670827全部12步成功，9文件（含依赖）全文与Git blob回读一致。新的实际只读接缝核验：[冻结目标历史与prefix差额](reports/PI0294/design/WDR1_FROZEN_TARGET_HISTORY_AND_PREFIX_DELTA_ACTUAL_20261024.json)。原clock纯派生的3份pause源码与原生封存hash逐字节一致，无重编译。冻结腿策略为88维／30帧历史；原binary真实导出InitRLFromBundle、ComputeObservation、Forward、ObservationBuffer get/reset/insert及暂停入口，但现有state journal不包含完整控制器历史，已检查当前bundle范围内未绑定合格的全身prefix／完整snapshot-replay producer。不能从有限查找推断全站缺失，也不能用12D MPC、猜测C++对象偏移或替换冻结binary冒充资格；不是“WBC接口缺失”，不加holdRMS或Source成功门。Core／LOCAL账本及控制metadata未变，模型／物理／登记写入0。SDK费用投影原UNKNOWN和已知零效果证据保持，之前的新冷边界例外仍待答复，不重发。完整目标ACTIVE，target未资格／core64=0，原全部预算／时钟不增。

原剩余4个target commissioning条件已用真实封存TRAIN population字节完成纯CPU冷态身份绑定：[四条件绑定](reports/PI0294/design/WDR1_ORIGINAL_TARGET_FOUR_COLD_IDENTITY_CPU_20261024.json)。复用既有几何／静态回执解析，保持F0/F1、T1/T2、I_TIME/I_PROGRESS平衡及固定CORR2048/R0，Source成功不是输入，不重开原8条件池或旧pilot。明确原2500tick warmup加16000task tick，不能漏计；10新用例／25几何用例／158相关用例通过（12私有parser skips）。这仍是CPU准备，未注册／预留／构造／物理或GPU；实际target原生prefix、feedback、ACK、STOP与完整准入仍待完成，core64=0。SDK费用投影原UNKNOWN及精确零效果证据保留，已向人类提交新冷边界例外，未收到批准前不执行其写入。完整目标ACTIVE，不重置时钟或预算。

SDK费用投影source285b2252精确CI40035481583／job120693768186全部12步成功、11文件全文及Git blob回读一致，10项临时原生程序测试通过。但实际写调用SSH返回255／无stderr，原Journal UNKNOWN保留；精确回读global与DEC1均为原字节，原生intent／receipt／temp均不存在，/proc实读原调用候选进程0且无不可读项，不能由超时推定终止。尚未落账、不自动重发：[已知零效果的新冷边界例外草案](reports/PI0294/design/WDR1_SDK_COST_TRANSPORT_UNKNOWN_KNOWN_ZERO_EXCEPTION_DRAFT_20261024.json)。新边界及7项捕获guard测试已准备，仍需人类明确例外；不增加预算、PBS、SDK／模型／物理或重放旧结算，只拟一次补齐DEC1金融constructor10／transfer335544320，global原费用不变。B4096完整审计／财务双关闭和SDK真实READY保持；target原生prefix／feedback／ACK／STOP／T1T2仍待资格，core64=0、完整目标ACTIVE。

SDK 五次已闭合费用的一次投影已完成纯 CPU 实现和真实双账本捕获字节核验，尚未原生采用：[闭合 SDK 费用投影](reports/PI0294/design/WDR1_CLOSED_SDK_COST_PROJECTION_CPU_20261024.json)。只补 DEC1 缺少的金融 constructor10 与 transfer335544320，Vanda global、GPU 站 CPU、全部训练费用和旧 Source/C0 marker 均不变；Core 输出2.684GB在原 Source/C0 base后仅计一次，LOCAL仍含原8GiB。SDK CPU费用已真实收费，不二次计费。114 fresh-import相关用例通过，完整回归2782／1649subtests／13private skips（完整回归在纯rows抽取前启动）；精确新CI／全文回读和fresh原生边界仍须通过。完整目标ACTIVE，target未取得原生任务资格，core64=0。

原冻结SDK链已实际跑通：[SDK READY／数学缓存／一次结算和关闭回读](reports/PI0294/design/WDR1_ORIGINAL_SDK_ACTUAL_READY_CACHE_AND_DUAL_CLOSURE_20261024.json)。source049beb63精确CI40031995146／job120682234978全部12步成功、26份绑定回读；SDK05真实编译／链接／动态依赖／11mesh／初始化PASS，收到modelType3、state12、input12的真实READY，SDK自然Exit0。30份只读AD缓存共63084字节及自有客户端binary已独立全SHA回验。实际worker+children8.2222CPU秒；Core608.2222／LOCAL120及原IO上界一次结算、lease CLOSED／actor0／8D holds0。前四失败全保留，五次累计Core6031.94373／LOCAL600CPU秒、金融constructor10非MuJoCo实际，episode／physics／GR00T／PBS均0；尚未把SDK费用镜像到Vanda。原WBC参数、控制器binary和资产均不改，AD缓存是原SDK按原task配置生成的数学缓存。当前仍0个真实policy-prefix sample、target_task_native_qualified=False、core64=0；下一项为SDK闭合费用一次投影、当前DEC1 target-native完整准入与真实feedback／ACK／STOP和prefix安全接线。B4096真实审计／结算／双关闭已完成，原owner读取Exit1继续保留；完整目标ACTIVE，预算／时钟／最终保护不增。

SDK04已实际通过编译／链接／清空环境的完整动态依赖检查，并进入原SDK配置与约束初始化；随后因CPU入口ROS包目录未绑定，无法解析URDF的package://网格URI，尚无SDK READY／物理／查询。Core607.904754／LOCAL120 CPU秒及原IO上界已一次结算、lease关闭、独立回读／8D holds0。原URDF全部11份网格已真实回读、0缺件；实际安装Pinocchio rosPaths说明读取ROS_PACKAGE_PATH／AMENT_PREFIX_PATH，当前两者均空。冷修复只在封存清单中绑定原包目录与mesh hash，编译前逐URI／文件预检，客户端使用固定包目录且清空非绑定AMENT路径，不改URDF／mesh／WBC。110相关CPU通过；最多6次有限工程尝试、同一原矩阵200000／3600 CPU内一次计账（内部组件10800／720上界），外部预算／PBS／原64格／三项矩阵／时钟不增。新修复仍需精确CI、全文回读和fresh完整准入。完整目标ACTIVE/core64=0。

SDK03已实际编译／链接成功，但动态加载器因间接HPIPM库未找到而在main前退出；进程启动尝试1，SDK初始化未进入（据真实loader诊断），物理／查询0。Core607.859879／LOCAL120 CPU秒及原IO上界已一次结算、lease关闭、独立回读／8D holds0。实查清空环境仅缺HPIPM，使用全部7个原封存库目录后ldd完整解析0缺件。冷修复只给自有客户端使用可传递RPATH及构造前有界ldd检查；真实CPU小库fixture验证RUNPATH失败／RPATH成功，109相关CPU及21预算用例通过。最多4次有限工程尝试仍在同一原矩阵200000／3600 CPU封套一次计账，不增外部预算／PBS／时钟、不重编WBC，不重放旧请求。新修复尚未原生采用，精确CI与fresh完整准入仍须通过。B4096已审计／结算／双关闭，原owner读取Exit1保留；完整目标ACTIVE/core64=0。

B4096已实际一次结算／GPU与当前Vanda CPU lease关闭并独立回读；累计11714更新、1231查询、103148 GPU秒，训练预留0，owner读取失败Exit1仍保留。SDK02已通过编译但链接缺原libcoal的两个OctoMap依赖，实际SDK构造／物理0；一次计入Core607.956897／LOCAL120 CPU秒及原IO上界、lease关闭、独立回读／8D holds0。修复只给自有客户端显式绑定已回读的原共享库及链接／运行搜索路径，不重编WBC；有限工程尝试最多3次，均从同一原矩阵200000／3600 CPU封套一次计账，外部预算、原64格／88条件／三项矩阵和时钟不变。新修复尚未native采用，需精确CI／全文回读与全新完整准入。完整WDR1 ACTIVE/core64=0。

B4096三份15成员完整状态及原12probe已实际独立审计通过，4096 SHA be7ab71dee0fb68e17efae06a04ff8dde9cd776ff429c1d6c408ae01323a9c19／cursor131072；真实训练F0与owner读取退出1仍分开。财务失败路径只允许结算／关闭，不返回checkpoint复用或新科学许可，原成功owner0谓词保持。修正了财务校验应使用不可变PBS训练结束时间，而非审计器后来生成的summary读取时间；120项相关CPU通过，仍未原生结算／关闭。

最新实际：[B4096真实F0、owner读取失败与SDK入口失败收尾](reports/PI0294/design/WDR1_B4096_F0_FAILED_READER_AND_SDK_ENTRY_FAILURE_ACTUAL_20261024.json)。1481574训练F/Exit0、实际2048新增更新／17查询／2加载；owner在训练完成后一次只读错误退出1，原成功owner0边界仍未满足，不重启。原有限独立三份完整状态审计正在运行，尚未结算。SDK01误绑旧仿真Python，在标准CPU入口拒绝，编译／构造／物理均0；全额费用已一次计入、lease关闭、独立回读／全部8D holds0，不重放该请求。当前仅增加明确财务失败路径与原矩阵准备内最多两次SDK工程尝试界限，所有外部预算、原64格／88条件／三项矩阵、最终保护和时钟保持。2638 CPU＋1499子测试通过；新代码CI待发布后核验。完整目标ACTIVE/core64=0。

SDK入口 a55e5e56 精确CI40021832581／job120648037331已全部12步成功，16份全文／Git blob回读一致。实际192项SDK链接与资产输入已完整只读回验；新Core child／编译／SDK构造／物理仍0。接续时修复了准备时间必须与稍后原生重算时间浮点相等的工程错误：封存输入时间不变，原生用真实当前时间重算并保留120秒新鲜度；77项相关CPU及10子测试通过。新修复CI仍须独立验证。

原 SDK 的[完整 CPU 资金接线和当前标准入口](reports/PI0294/design/WDR1_SDK_COMPLETE_FUNDING_AND_CURRENT_STANDARD_ENTRY_CPU_20261024.json)已实现：同一矩阵准备 CPU 一次抵扣，全部原64格／88条件上界／三项矩阵作业及其他资源保持；当前双账本、role、lease、源码与标准入口全检查后才允许独立 SDK 有界编译／构造。70项针对性CPU验证通过，真实安装SDK及新Core child仍未执行。训练同一1481574在04:34:45 SGT实查B3802（新增1754）、17查询／2加载／unknown=[]，owner健康；未宣称完整保存4096。完整WDR1 ACTIVE/core64=0。

Latest material progress (2026-10-24 03:54:24 SGT): job1481574 has actual first-update release and native B3346 observed (1298 new committed updates from2048),17 real queries/2loads/unknown[]. Healthy server-resident owner/CONFIRMED heartbeat. Original runtime36925d27 unchanged. Independent cold terminal helper5c9a6de7 exactCI39999597863 all12 success/full7files; original finite audit/once settlement/dual close/LOCAL projection connected and temp-tested, refuses live owner before transport. No actual terminal audit/settle/close. Source and task qualification remain separate; wholeWDR1 ACTIVE/core64=0. Actual: `reports/PI0294/design/parallel-science-20261022/B1481574_FIRST_UPDATES_AND_COLD_TERMINAL_CONNECTED_ACTUAL_20261024.json`.

Current target delta: [owned quota and paused STOP IO](reports/PI0294/design/WDR1_TARGET_OWNED_QUOTA_PAUSED_STOP_CPU_20261024.json) now uses the explicitly pinned original82004c31 clock, a permanent single-owner latch and full native-state/fresh-pause readback. Cold code/CPU only; no native adoption, new Core child or physics. Original target prefix safety and actual task qualification remain required; do not call the expired R4 pilot run.

The [actual frozen SDK link-input readback](reports/PI0294/design/WDR1_ACTUAL_FROZEN_MPC_SDK_LINK_INPUTS_20261024.json) locates the original static implementations and matches selected constraint/prediction function bodies to the frozen executables. Header/class ABI and destructor variants remain unqualified; this is no native target admission or safety PASS.
[Original SDK constraint reader](reports/PI0294/design/WDR1_ORIGINAL_SDK_CONSTRAINT_CLIENT_CPU_20261024.json) now dispatches the actual SDK box-cost and self-collision APIs with exact native-policy wire binding. 32 new CPU cases pass; native SDK build/construction and task qualification remain pending. Box cost is not a feasibility margin, and native-policy knot samples grant no physical quota.

[原 SDK 有限客户端及构建计划](reports/PI0294/design/WDR1_FINITE_ORIGINAL_SDK_CLIENT_AND_BUILD_PLAN_CPU_20261024.json)已完成独立二进制管道、同一进程复用、当前请求／输出绑定、编译器禁用及未知不重发接线；新增31项CPU测试，相关63项通过。真实安装SDK的编译／构造及当前DEC1双账本准入仍待完成，未新增Core child或物理运行。

## 2026-10-24 进度审核与长程计划 V3

证据截至 **5d9f2a82，2026-10-24 01:46:45 SGT**（原生读数01:45:53；回执01:46:11）。A4096完整；B恢复点2048；**新Vanda作业1481574已唯一提交并转R、server-resident owner健康，MODEL_START已放行，FIRST_UPDATE未放行且未观察新更新**。PBS17/28已实际采用，当前已用global25／DEC1 14；保持同一owner，禁止重复采用或qsub。目标任务资格仍未完成，core64=0；旧学生完整确认仍T1 10/20、T2 4/20。

- [进度审核：实际结果、失败、成本与证据边界](docs/reviews/2026-10-24_GR00T_WBC_PROGRESS_AUDIT.md)
- [长程计划V3：阶段、验收、分支与完整预算](docs/roadmap/2026-10-24_GR00T_WBC_LONG_HORIZON_PLAN_V3.md)
- [当前接续交接：保护现作业、最小目标接入、原时钟收尾](docs/handover/2026-10-24/GR00T_WBC_NEXT_EXECUTION_HANDOFF.md)

此次为文档更新，未改变运行capsule、科学GO、预算或截止。10-25 02:29:56 SGT停新作业／候选，10-26 02:29:56硬收尾。下方旧状态按其原时间解释；真正接续仍刷新唯一owner原生状态。

发布并发复核：**5c9a6de763c6f74b7731809f72b2dd54e6ca47de，2026-10-24 01:59:22 SGT** 新增[1481574共享终局契约及精确身份CPU验证](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/5c9a6de763c6f74b7731809f72b2dd54e6ca47de/reports/PI0294/design/parallel-science-20261022/B_PRECUDA_RETRY1481574_SHARED_TERMINAL_CONTRACT_CPU_20261024.json)。它尚未产生原生终局审计／结算／关闭，未宣称首次更新；运行capsule仍为36925d27，上述科学与训练实际状态不变。

---

Cold terminal preparation for the actual live job1481574: `reports/PI0294/design/parallel-science-20261022/B_PRECUDA_RETRY1481574_SHARED_TERMINAL_CONTRACT_CPU_20261024.json`. Existing source90 checks extracted into shared pure predicates and retained; new binding pins actual source36925d27/request/manifests/owner instances. 2456 PI0294 +1499 subtests pass. Actual full captured request identity passes, and live-owner terminal path refuses before transport. No native audit/settle/close yet; active training capsule unchanged, WDR1/core64 remain incomplete.

Latest actual: job1481574 R, MODEL_START_RELEASED observed, FIRST_UPDATE_RELEASED not yet observed; healthy existing server-resident owner/heartbeat. This supersedes the earlier Q snapshot below. Runtime36925d27 remains sealed and unchanged; exactCI all12 success. Actual report: `reports/PI0294/design/parallel-science-20261022/VANDA_PRECUDA_RETRY_PBS17_28_UNIQUE_GPU_SUBMISSION_ACTUAL_20261024.json`. No replay/prepay/register/restart; all original caps/clocks/science and remaining three matrix jobs retained.

Latest actual: new Vanda GPU job1481574 uniquely submitted and bound, Q with healthy server-resident owner; no model/first-update release observed. Exact 17/28 grant adopted once with independent readback. Runtime36925d27 exact CI39996540379 all12 success; full native1019-file readback. Actual report: `reports/PI0294/design/parallel-science-20261022/VANDA_PRECUDA_RETRY_PBS17_28_UNIQUE_GPU_SUBMISSION_ACTUAL_20261024.json`. All original costs/clocks/science and three matrix jobs remain; WDR1 ACTIVE/core64=0.

Latest PBS-only implementation: `reports/PI0294/design/parallel-science-20261022/B_PRECUDA1481489_PBS17_28_RUNTIME_CPU_20261024.json`. Human 17/28 grant is published; 2425 PI0294 CPU tests + 1499 subtests and 13 actual captured-byte cases pass. Native adoption and new GPU submission remain pending; no running GPU task. Original three matrix jobs, all costs and clocks remain unchanged.

Human approved exactly DEC1PBS16→17/globalPBS27→28 after actual pre-CUDA1481489 closure: [fixed authorization](reports/PI0294/design/parallel-science-20261022/B_PRECUDA1481489_RETRY_PBS_AUTHORIZATION_20261024.json). All costs9666updates/1214queries/global91003GPU-s and old805/922 losses remain; seedB2048→4096 and original3matrix jobs retain every other cap, T0/deadline and final20GPUh/40GiB. No repeated permission for this scope. Record/adopt this exact grant once and complete cold repair qualification/fresh native admission before a new request. Current native adoption and new retry submission are absent; old1481489/leases are terminal and never replayed. FullWDR1 ACTIVE/core64=0.

Actual recovery2 return supersedes the earlierQ snapshot:1481489 F/Exit1, original pre-CUDA release wait timed out after its owner rejected a stale storage lifecycle peak. Zero model/query/update/physics, preservedB2048. Original pre-CUDA failure path once settled265GPU-s/1180CPU-s/64MiB, both Vanda leases independently CLOSED and LOCAL once projection confirmed; totals9666updates/1214queries/global91003GPU-s/globalPBS24/DEC1PBS13. A qdel attempt was rejected before a native intent because the job was alreadyF; its local UNKNOWN remains historical, no resend. Cold fix derives peak from the exact approved draft plus actual closed costs and rejects insufficient peak beforeqsub;12new/37relatedCPU+9subtests,2411fullPI0294+1499subtests/13private skips and6private settlement temp cases pass. Exact source90 original arithmetic reproduces old-plan rejection and corrected-plan full captured reservedR PASS without mocks. Final4096 helper2daacc61 exactCI39992252264/job120546681665 all12success/6fullraw+Gitblob matches. Only proposed retry counts DEC1PBS16→17/global27→28; other caps and clocks/final protection fit full remaining review, including a separately priced1800siteCPU/15LOCAL/256MiB/64MiB controller without refunding any old cost or reducing5520site/3600LOCAL future ceilings. Current retry is unapproved/unregistered/unsubmitted; new repairCI pending. [Actual failure, fix and full review](reports/PI0294/design/parallel-science-20261022/VANDA_RECOVERY2_PRECUDA_STORAGE_FAILURE_CLOSED_AND_RETRY_PBS_DRAFT_20261024.json). FullWDR1 ACTIVE/core64=0.

Vanda GPU recovery2 is now actually submitted once:1481489.stdct-mgmt-02/Q, real paired binding and live server-resident original watch/CONFIRMED heartbeat. New ordinary authenticated connection succeeded after the human fresh-login evidence; old timeout groups/masters are preserved, no root cause inferred. Exact90d357a7 runtimeCIall12success; both standard capsules independently fully read1006/972sealed members,1012unique files. New control1800site/20LOCAL/256MiB/64MiB once prepaid, original future envelopes/final protection retained. New full payload correctly imports3actual closed sealed training bindings; the earlier local assembly assertion preceded any registration intent/native write. Current9666settled updates/globalPBS24/DEC1PBS13, new2048updates reserved; no model/first-update release yet. No hot changes/restart/repeatedqsub/prepay. [Actual submission](reports/PI0294/design/parallel-science-20261022/VANDA_RECOVERY2_SERVER_RESIDENT_UNIQUE_GPU_SUBMISSION_ACTUAL_20261024.json). Pure final4096 contract:41new/160related+65subtests,2399PI0294+1499subtests/13private skips; exact17compatibility intent/array census rejects replayed probes while finance retains actual bounded counts. New helperCI/readback and private finite terminal dispatch remain pending, GPU runtime unchanged. FullWDR1 ACTIVE/core64=0.

Pure final4096 recovery2 contracts now bind the future actual request, full saved states, nativeF0 and real owner OS exit0. Full17compatibility query intent/array census rejects original-probe replay; financial counts remain actual within the unchanged envelope.41new/160related CPU+65subtests,2399PI0294+1499subtests/13private skips pass;10temporary real-file reader cases pass. Exact helperCI/readback and remaining private finite audit dispatch are pending. The already qualified90 GPU runtime is unchanged and its actual resident submission is reported separately above. [CPU contract evidence](reports/PI0294/design/parallel-science-20261022/VANDA_RECOVERY2_FINAL4096_TYPED_CONTRACT_CPU_20261023.json).

GPU-priority native boundary: all8authority cards and current environment pin match; new roots/roles are absent and paired ledgers are unchanged after approved cap adoption. Queue was NOT observed by that metadata-only read. A second qualified cold qselect read again timed out10s; a distinct own-client diagnostic observed qselect sleeping in poll, without child or stdout/stderr. Preserve UNKNOWN; no root cause or empty queue inferred. Cold owner now reuses the original foreground-shell exit capture, recording real OS return code after the owner returns; real dummy exit0/exit7 and19temporary integration cases pass. Before any new control cost, the cold bootstrap requires fresh project/quota evidence covering all future GPU work plus protected20GPUh; that new native evidence has not yet been obtained. No new native prepay/register/resident/request/qsub. Runtime90d357a7 remains exactCIall12success; final new-lineage4096 terminal binding is still required. FullWDR1 ACTIVE/core64=0. [Evidence](reports/PI0294/design/parallel-science-20261022/VANDA_RESIDENT_DETACHED_WATCH_AND_TERMINAL_HISTORY_CPU_20261023.json).

Resident cold bootstrap now has exact runtime90d357a7 CI39983917781/job120518029162 completed/success/all12,26fullraw+Gitblob readbacks,966source/35small prepared and16temporary integration cases pass. Captured once accounting, full original prequeue recomputation, standard cold registration/full SHA readback and actual detached OS launch/parent-exit semantics were exercised without native PBS/model. Preserve026failed fixture CI. Fresh native own-qstat25s and own-qselect10s still timed out; correct scheduler route/TCP connectivity does not establish a complete queue. No new GPU request, prepay, registration, resident or qsub exists. FullWDR1 ACTIVE/core64=0. [Actual qualification and queue boundary](reports/PI0294/design/parallel-science-20261022/VANDA_RESIDENT_DETACHED_WATCH_AND_TERMINAL_HISTORY_CPU_20261023.json).

Detached CPU fixture CI repair: exact026fd575 CI39982253385 failed only the new child import fixture (missing explicit repository path after launcher detachment); preserve that failure. The fixture now pins its own repository imports and is tested with PYTHONPATH removed:20selected tests pass. Production runtime is unchanged; new exact CI/readback pending. Vanda own-account qstat again timed out25s at1792771479; queue is UNKNOWN and no new GPU request/submission/prepayment exists. [Evidence and repair](reports/PI0294/design/parallel-science-20261022/VANDA_RESIDENT_DETACHED_WATCH_AND_TERMINAL_HISTORY_CPU_20261023.json). FullWDR1 ACTIVE/core64=0.

Resident owner terminal and launcher independence: [CPU actual](reports/PI0294/design/parallel-science-20261022/VANDA_RESIDENT_DETACHED_WATCH_AND_TERMINAL_HISTORY_CPU_20261023.json). A separate exact fresh qstat-history proof now resolves a registered job leaving the complete current queue, with same request/nonce and actual F/Exit status; no fabricated queue, settlement or success inference. Detached real-OS CPU fixture proves original watch keeps heartbeating after launcher exit and returns on matching terminal without resubmission. Optional ordinary-login child CPU fence preserves hidden credentials/default callers.23new/103focused CPU+5subtests;2467fullCPU/1649subtests pass,13private skips. Vanda five caps remain actually adopted; no new GPU request/owner/prepay/submission. PBS reads remain last observed timed out; private cold binding and fresh admission pending. FullWDR1 ACTIVE/core64=0. New exact CI/full readback pending; prior2d323e55 all12success remains historical qualified code.

Current Vanda recovery2: [actual caps adopted, project login confirmed, native queue reads interrupted](reports/PI0294/design/parallel-science-20261022/VANDA_RECOVERY2_CAPS_ACTUAL_AND_NATIVE_QUEUE_READ_TIMEOUT_20261023.json). New11714updates/16-27PBS/614-680GiB caps are actual paired once adoption with independent receipts; costs9666updates and failed2970 remain. Runtime2d323e55 exactCI39976861380/job120493606263 all12success/4fullreadbacks;2444CPU/1644subtests pass. Ordinary account login and fresh project eligibility/balance readback succeeded. Three complete qselect reads each hit10s; distinct original qstat-own route hit25s. Queue remains UNKNOWN, never treated empty or terminal. No new GPU request/qsub/prepay/owner/model/training. Local966source/35small cold inputs prepared; resident binding remains private WIP, not deployed. Full goal ACTIVE/core64=0; original clocks/final protection unchanged.

Actual recovery2 five-cap native adoption is independently confirmed on Vanda: [report](reports/PI0294/design/parallel-science-20261022/VANDA_RESIDENT_CAMPAIGN_AND_APPROVED_RECOVERY2_CPU_20261023.json). DEC1 updates11714/PBS16/storage614GiB; globalPBS27/storage680GiB. Full costs9666updates and old2970 lineage remain immutable; new RECOVERY2 seed2048 registered. Accounting6b1b266e exactCI39975513066/job120488937481 all12success/23fullraw+Gitblob readbacks. A predeployment heartbeat-progress regression was reproduced and fixed;2444CPU/1644subtests pass,13private skips. Resident GPU owner is not yet deployed or submitted; its exact fix CI/readback, cold registration and fresh admission remain required. FullWDR1 ACTIVE/core64=0.

Vanda resident campaign and approved recovery2 runtime: [CPU actual report](reports/PI0294/design/parallel-science-20261022/VANDA_RESIDENT_CAMPAIGN_AND_APPROVED_RECOVERY2_CPU_20261023.json).2443 CPU tests/1644subtests pass,13private parser skips; exact captured native ledgers pass pure cap/lineage adoption with9666paid updates and old2970 retained. New explicit Vanda current queue/authority and local Journal transport reuse original watch/300s watchdog,30/40s RPC limits and no mutation retry. Five caps are human approved, but native adoption/new resident owner/GPU submission remain pending exact CI/readback and fresh cold admission. Worker2 full native data qualification still pending; no speedup claimed. FullWDR1 ACTIVE/core64=0.

Human approved exactly five recovery2 cap changes: [binding authorization](reports/PI0294/design/parallel-science-20261022/B2970_RECOVERY2_FIVE_CAP_AUTHORIZATION_20261023.json). DEC1 updates11714; DEC1/global PBS16/27; GPUsite storage614/680GiB. Preserve9666actual charged updates, prior805+922paid losses, both old B lineages, all checkpoints, original clocks/other budgets/final protection. New B_COVERAGE_HEARTBEAT_RECOVERY2 imports auditedB2048/cursor65536 to4096 on Vanda under a new resident owner and fresh admission. Authorization is recorded; native adoption, new request and GPU submission remain pending. FullWDR1 ACTIVE/core64=0. Earlier pending-approval entries are historical.

Actual Vanda closeout complete: [failed2970 once settlement, GPU closure and concrete recovery2 scope draft](reports/PI0294/design/parallel-science-20261022/B2970_VANDA_ACTUAL_SETTLED_GPU_CLOSED_AND_RECOVERY2_SCOPE_DRAFT_20261023.json). Both native ledgers and independent once receipts confirm9666updates/1214queries/global90738GPU-s/DEC170244GPU-s; all reservations0/unknown[]. GPUlease CLOSED, completeB2048 preserved,922paid loss retained. Old unavailable Hopper CPU lease remains fenced/not physically CLOSED; currentVanda adminlease ACTIVE. LOCAL projection now explicitly derives fromVanda. Runtime c0fe1fec exactCI39970390264/job120471199144 all12success/6fullreadbacks. A new final2048→4096 continuation plus original3matrix jobs requires only proposed10792→11714updates,DEC1/globalPBS15/26→16/27,storage549/615→614/680GiB; all other budgets/clocks/final protection fit full conservative review. This is a draft awaiting human approval, no native cap change or new train/PBS. Resident owner/auth renewal/full-data loader2 qualification and fresh admission still required. FullWDR1 ACTIVE/core64=0. Earlier snapshots remain history.

Vanda failed-job financial closeout now has an explicit accounting-only evidence path: [prior real F1 plus current exact native terminal files](reports/PI0294/design/parallel-science-20261022/B2970_KNOWN_FINALITY_CURRENT_NATIVE_FILES_AND_ONCE_SETTLEMENT_CPU_20261023.json). Original PBS timestamp1792756412 stays immutable; three later25s history timeouts remain failures. Current actual files/effects922/17/2 and inactive owner/auditor match the full independent B1024/B2048 audit.41new/104combined CPU +13subtests, actual captured-ledger arithmetic and5temporary once-IO cases pass; new distinct proof cannot become training admission. No settlement/closure yet, reservations retained. Prior loader f31c7fb8 exactCI39966977551/job120459277216 all12success/5fullreadbacks; full PV1 worker qualification still pending. FullWDR1 ACTIVE/core64=0; new exact CI/readback/native adoption pending. Earlier entries remain history.

Actual PV1 loader bottleneck corrected in CPU engineering: [MP1 worker forwarding](reports/PI0294/design/parallel-science-20261022/PV1_ACTUAL_MP1_LOADER_WORKER_FORWARDING_CPU_20261023.json). PV1 invokes MP1Trainer.get_train_dataloader, whose override hardcoded worker0. New opt-in2/4 now constructs real spawned readers with bounded threads and prefetch2; legacy0 and existing PV1 arguments remain unchanged.16new/39selected CPU +14subtests pass, including actual worker2, exact ordered CPU tensors, original resume cursor and main Python/NumPy/torch RNG. Tiny fixtures are not full PV1 native qualification or GPU speedup evidence.7ac5617d CPU-partition code has exactCI39964074288/job120449236499 all12success/5fullreadbacks; distinct server-resident600s audit started under the original54000CPU hold, actual saved-state audit now independently completed Exit0 with both15-member B1024/B2048 states, original12probes and53unique files independently fully rehashed. No new training, settlement or closure is claimed by this snapshot. FullWDR1 ACTIVE/core64=0. New loader code CI/readback pending; earlier entries remain history.

Actual Vanda8MiB administrative output supplement is once charged and independently reconciled; exact7d8bd384 CI39961479891/job120440284722 is all12success/5fullreadbacks. [Finite audit failure and held-CPU partition](reports/PI0294/design/parallel-science-20261022/VANDA_FINITE_READONLY_AUDIT_FAILURE_AND_RESERVED_CPU_PARTITION_20261023.json): the server-resident auditor independently persisted a CPU-limit failure at108.2125s and later its exact actor handle is absent. No full-state PASS, settlement or closure is claimed; the failed actor/intent stays immutable. Preserve the prior480s group and qualify a distinct finite600s readonly allocation within this same failed job's already-held54000CPU-s; whole conservative job cost11023s, no new CPU cap/prepay or reset.18new/93selected CPU and actual captured paired-ledger arithmetic pass; new native audit not started. Last settled8744/1197 and all failure reservations remain. FullWDR1 ACTIVE/core64=0. Earlier entries remain history.

Vanda control authority is now actually registered and adopted: [native adoption, full1910source readback and actual storage discrepancy](reports/PI0294/design/parallel-science-20261022/VANDA_CONTROL_AUTHORITY_NATIVE_ADOPTION_AND_STORAGE_RECONCILIATION_20261023.json). Exact68ac CI39957384368/job120426504285 is all12success/5fullreadbacks. Both955-source capsules match; native paired ledgers preserve all old8744updates/1197queries and current failed2970 reservations. Old Hopper origin stays unavailable/fenced, never called fresh. Actual metadata5,885,952bytes exceeds prior2,113,536-byte headroom; preserve the miss and qualify an additional8MiB once charge inside existing549/615GiB caps, retaining original64MiB plus2MiB finite audit output.18new/72selected CPU and7private temporary bootstrap checks pass; supplemental charge not native yet. No new training/settlement/closure, server-resident audit and worker qualification pending. FullWDR1 ACTIVE/core64=0. Earlier snapshots below remain history.

Vanda-only controller handoff now has exact actual-byte preparation: [complete stopped-owner journal and paired-ledger preservation](reports/PI0294/design/parallel-science-20261022/VANDA_CONTROLLER_HANDOFF_CAPTURED_JOURNAL_CPU_ACTUAL_20261023.json).1599files/799intents are pinned by full catalog hash:542reads including1preserved failure,255heartbeats,2once Vanda releases, no unresolved effect-bearing call or post-snapshot Hopper write. Captured actual paired ledgers pass pure handoff with every old cost/cap/reservation preserved; neither native adoption nor new training is performed. Fresh Vanda metadata confirms parent adoption/global+DEC1ledgers/manifest/binding are absent, while original qualification/native-build/neural-training profiles exist; a standard registered parent/controller bootstrap is required.18newCPU/83combined+55subtests pass. Original10792/15/26/549/615, T0/deadlines/final protection remain; WDR1 ACTIVE/core64=0. Exact new CI/readback pending.

New explicit human direction: [all fine-tuning and background training control to Vanda, within existing caps](reports/PI0294/design/parallel-science-20261022/VANDA_ONLY_TRAINING_EXECUTION_DIRECTION_20261023.json). Hopper is unavailable: do not repeat its login or require it as a future training control endpoint. Cold authority transfer must preserve exact last-known ledger bytes and prove/reconcile all subsequent mutations; an old snapshot is not a fresh Hopper read. Existing one-A40 training had36allocated CPUs but4process-affinity CPUs and worker0; qualify ordered-loader2/4, CPU/thread accounting, full resume and any microbatch change before deployment. No new budget, T0/deadline, recipe/math/WBC change or new training submission. Prior saved-state audit child is now authoritatively missing; its287.49CPU-s upper plus34s prior attempt remains charged, no native audit completion receipt recovered. New readonly hash helper15CPU/36related pass; all cached bytes are tentative until independent full final rehash, including same-tick mutations. FullWDR1 ACTIVE/core64=0. Exact helper CI/readback pending.

Latest B2970 closeout staging and actual throughput: [audit recovery, fresh Vanda project login and existing training performance](reports/PI0294/design/parallel-science-20261022/B2970_CLOSEOUT_STAGING_AND_TRAINING_PERFORMANCE_ACTUAL_20261023.json). Runtime8eb0 failure contract exactCI39950226552/job120402591906 is all12success/5fullreadbacks. New pure closeout join50CPU/55subtests and actual captured-ledger failure arithmetic pass, with no native settlement/closure or retry. Full state audit remains incomplete: first scheduler read timed out before hashing; saved-state-only child remains independently confirmed live after the local controller hit its CPU limit, never restarted. Fresh Vanda project membership is verified; Hopper connection still unavailable. Existing922-update performance averages6.583s/update; worker0 has138.23s loader wait in first100, scheduler whole-job usagePerc23.489 is not steady utilization. No worker/microbatch/backend or active capsule change. All old caps/clocks/final/history preserved; fullWDR1 ACTIVE/core64=0. New exact CI/readback pending.

Latest training failure overrides old R snapshots: [actual token-expired owner stop and native B2970 failure](reports/PI0294/design/parallel-science-20261022/B_RETRY1_TOKEN_EXPIRED_HEARTBEAT_FAILURE2970_ACTUAL_20261023.json). Owner exited on VANDA_PROJECT_TOKEN_EXPIRED after last CONFIRMED heartbeat; exact healthy-Vanda-mux read at1792756412 confirms1480817 F/Exit1/HeartbeatLost,922actual updates from2048 to2970,17queries/2loads/no pending or unknown intents. No4096 completion, owner restart, qsub or success-terminal execution. Last settled8744 plus922 would be9666 after once settlement; all current22560GPU-s/54000CPU-s/2048updates/29queries/70GB reservations remain held. Hopper master disappeared and normal authentication was rejected; manual original connection restoration requested, settlement/closures NOT executed. Failure contract27CPU/65subtests and exact captured effects checks pass; full saved-state/probe audit is compiled only until precise CI qualification. If only originalB2048 survives, a further2048updates needs11714cumulative, beyond current10792; no new scope is inferred. Original caps/clocks/final/history preserved, WDR1 ACTIVE/core64=0. Independently,068b64a4 measured-input code exactCI39948666350/job120397480955 all12success/5fullreadbacks; CPU qualification only.

Target measured input seam: [actual-state/camera records to existing query CPU integration](reports/PI0294/design/WDR1_TARGET_MEASURED_OBSERVATION_EXISTING_QUERY_CPU_20261023.json). New target_observation.py reuses original34/32/20 binary parser and frozen ActualState read-only forward kinematics; owner-pinned inverse rigid transform gives original policy EEF10 while actual-world feedback retains real TCP/gripper/time. Original FrameRecorder native index/step/absolute calibration clocks and both256RGB PNGs bind the same packet; object truth/Teacher phase are excluded.31new CPU tests/2214fullPI0292+PI0294/1644subtests pass (13private skips), including actual-format observation→existing e1 single draw→original H40 gate→WAIT_ACK, never fake physics ACK. Reader update/model draw are explicit CPU doubles, not native qualification. Original worker/state/relay source unchanged; no Core child/physics or running-capsule edit. Current1480817 R/step2931/883newupdates/17queries/2loads/unknown[] at1792755974. FullWDR1 ACTIVE/core64=0; exact new CI/readback pending. Prior de86223f CI39947320770/job120393043936 is all12success/5fullraw+Gitblob readbacks.

Target cold first-state durability: [owner contract pinned in the first durable state](reports/PI0294/design/WDR1_TARGET_COLD_FIRST_STATE_DURABILITY_CPU_20261023.json). The target overlay now constructs only a fresh locked state with its target pin already present; original relay operations and sealed39b19c67 glue bytes remain unchanged. A real temp-state write followed by a lost-return fixture verifies that RAW reopen rejects it and target reopen refuses any existing state: exactly one write attempt/no unknown replay.17target route cases and268related/155subtests pass. This is CPU durability only, no native target child/task safety/core64; current B4096 training untouched. Prior a8e865d9 has7full-file/Gitblob readbacks and exactCI39946989977 last observed in progress; current new exactCI/readback pending. FullWDR1 remains ACTIVE.

Correction to prior target relay publication: [frozen-source-preserving cold target overlay](reports/PI0294/design/WDR1_TARGET_RELAY_FROZEN_SOURCE_OVERLAY_FIX_CPU_20261023.json).7919c972 CI39943324436/job120379813087 failed2 original source pin tests because relay_protocol.py is one of the sealed source glue members; that failure is retained and was never native deployed. Its original exact39b19c67 bytes are restored, pin tables/guards unchanged. The independent TargetRelayExchange only binds a fresh owner-injected target contract, inheriting original action/intents/ACK/stop operations; relay_transport selects it explicitly without packet/case flags or source-S1 wrappers.52focused tests pass;2179fullPI0292/PI0294 tests/1644subtests pass with13private skips, plus the final16target route tests pass including3later factory negatives. No native task/safety qualification, target child, model, physics or matrix result. Existing1480817 remains R/2821 at1792755109, heartbeat CONFIRMED/unknown[]/entry unchanged. New exact fix CI/readback pending; fullWDR1 ACTIVE/core64=0. Earlier7919 pending text is history.

Current target cold route: [original relay now accepts the explicitly pinned target ACK contract](reports/PI0294/design/WDR1_TARGET_ACK_ORIGINAL_RELAY_COLD_ROUTE_CPU_20261023.json).12new CPU route cases plus2final-generation79 boundary cases pass;281related/155subtests pass with12private-parser skips. The original relay and current resident share full ACK bytes across two synthetic prefixes, preserving separate known neural generation vs unknown physical-send outcomes; no resend/resampling, contract drop/replacement, or packet/case-flag selection. This is CPU integration only, no native target dispatch, safety qualification or core64 result. f9d6acca target ACK exactCI39942004910/job120375451293 is completed/success/all12 with6fullraw/Gitblob readbacks; its earlier pending record is historical. New route commit CI/readback remains pending. Existing B4096 training and648terminal guards stay unchanged; fullWDR1 ACTIVE.

Current target CPU integration: [target time ACK connected to the existing resident consumer](reports/PI0294/design/WDR1_TARGET_TIME_ACK_EXISTING_RESIDENT_CPU_ACTUAL_20261023.json). The cold owner must pin original CORR2048/R0, the frozen82004c31 clock and original2500 warmup ticks; all8 installed time references, native float32 policy echoes/independent observations, physical tick links and separate original target-prefix verdict are bound. This is pure binding/CPU integration only: no new controller/runner, native target child, safety qualification or matrix dispatch. Source S1 and RAW defaults remain distinct; unknown model generations cannot resample.41new CPU cases and153related pass. Combined root/PI0292/PI0294 run has3097passes/1704subtests/27skips and one publisher Git-worktree environment failure; that unchanged test passes separately with explicit worktree. Current1480817 remains R/step2513 with17actual queries/2loads/unknown[]; active capsule unchanged. Native target qualification=False/core64=0, fullWDR1 ACTIVE. Exact new-code CI/readback remains pending.

Actual training progress: current unique B retry1480817 is R/GN-A40-039; model and first-update full native admission released; actual17 compatibility queries/2loads passed; see reports/PI0294/design/parallel-science-20261022/B_RETRY1_NATIVE_RELEASE_AND4096_TERMINAL_CPU_PREPARED_20261023.json. New4096 terminal contract and cold wrapper are CPU prepared only, not native terminal evidence. Requires F/Exit0, natural owner success, all3x15 states/original12probes, once settle and both closures. Cumulative successful expectation10792; all old costs and clocks preserved. b788 document exactCI39932574926 now all12success. FullWDR1 still ACTIVE/core64=0.

Latest actual: human PBS-only grant e48db225 is once dual adopted (DEC1/global15/26). Unique retry job1480817 is nativeQ under one active owner; whole dual queue PASS, heartbeat CONFIRMED, no model/first-update receipt yet. Resume original B2048 to4096; old1480412 failure fully charged/closed. Cumulative8744 updates/82257GPU s/1197queries; PBS23/12, new reserve22560GPU s/54000CPU s/2048updates/29queries/70GB. Same original control1800Hopper/180LOCAL/256MiB/64MiB shared; no second prepay. See reports/PI0294/design/parallel-science-20261022/B2_PBS15_26_APPROVED_UNIQUE_RETRY_NATIVE_ACTUAL_20261023.json.

Latest actual: B2 1480412 F/Exit-3/startup21 failure has been once conservatively settled (484 GPU s /15384 CPU s), both GPU/CPU leases CLOSED and independently read back. Cumulative8744 updates/82257 GPU s/1197 queries; no new model/update effect. Human approved only PBS15/26 via e48db225; retry runtime qualification/cold binding/fresh native admission still required, no retry submitted. NUS hourly PBS mail opt-out completed. See reports/PI0294/design/parallel-science-20261022/B2_STARTUP_FAILURE_ACTUAL_CLOSED_AND_PBS_ONLY_RETRY_CPU_20261023.json.

> **PBS-only retry adjustment approved:** [Exact human authorization](reports/PI0294/design/parallel-science-20261022/B2_SYSTEM_STARTUP_RETRY_PBS_AUTHORIZATION_20261023.json) approves DEC1PBS14→15/globalPBS25→26 for one unique original B2048→4096 retry, preserving the original three matrix submissions. No other budget, checkpoint/history, scientific gate, T0 or deadline changes. Do not repeat this approval question. Grant still needs qualified code and native paired-ledger adoption, real old-failure settlement/closure and full fresh admission before retry.

> **Auth restored and email disabled; retry scope prepared:** [Actual authentication/email and truthful startup-financial CPU qualification](reports/PI0294/design/parallel-science-20261022/B2_SYSTEM_STARTUP_FAILURE_FINANCIAL_AND_RETRY_SCOPE_CPU_20261023.json) records successful normal masked `amgr login` and sufficient fresh project balance. The NUS hourly PBS email opt-out marker is created and independently read back. Exact old1480412 AMS transaction is unreconciled;21acquires/20releases are preserved separately from physical costs. New administrative proof keeps recorded_process_terminal=False, charges a conservative484GPU-s/15384CPU-s across the full424-second lifetime/all21attempts, zero model/update/query effects, and no duplicate output charge (both existing roots41.9MB inside prepaid64MiB). Pure full captured-ledger settlement passes;6new/17subtests and full2003/1432 pass. No native financial settlement/closure/retry yet; exact code CI/readback still required. Full original remaining-work review has only one PBS shortfall per ledger; a concrete PBS14→15/25→26 adjustment would fit all other unchanged caps/clocks,16297DEC1GPU-s and36LOCALCPU-s margins. No cap is silently increased; new setup must share/enforce the existing180LOCAL/1800Hopper/256MiB/64MiB cost envelope. Prior pending-login statements are history; goal incomplete.

> **Human retry direction received:** [amgr/project check and retry intake](reports/PI0294/design/parallel-science-20261022/B2_SYSTEM_HOLD21_RETRY_DIRECTION_AND_AUTH_BLOCK_20261023.json) records the direct instruction to check via `amgr login` and retry. Actual native query returned TOKEN_EXPIRED; no project-balance value is available yet. A masked terminal login is requested once; no credential is requested in chat. Old1480412 stays F/Exit-3, all reservations/leases remain pending lawful settlement, no new request/registration/PBS yet. Original full remaining matrix/final work is retained; one new retry plus three matrix submissions would require DEC1/globalPBS15/26 versus current14/25, so no cap is silently increased. Fresh auth/quota, exact failure accounting and complete retry resource scope precede dispatch. Goal remains incomplete.

> **Latest B2 site failure:** [System-held21 failure and once withdrawal](reports/PI0294/design/parallel-science-20261022/B2_SYSTEM_HOLD21_WITHDRAWN_ACTUAL_20261023.json) supersedes the healthy-Q snapshot. PBS reported H/system hold,run_count21,Exit-3 before any PBS_READY/model/first-update/effect receipt; owner stopped on original STOP_UNKNOWN_CAMPAIGN_STATE, never restarted. Exact owned job1480412 was withdrawn once; fresh native qstat confirms F/Exit-3. No unhold/requeue/newqsub. Actual underlying server/MOM cause and charge are unknown: user-side tracejob/error logs unavailable; one administrator-record request is pending. No fabricated process-final or success settlement: all GPU/CPU/update/query reservations remain, GPU+CPU leases not closed. Settled8744/81773GPU-s/1197queries and PBS22/11 remain; full goal incomplete/target-qualified=False/core64=0. Prior Q reportf6c0 exact document CI39923083495 succeeded, but is historical health.

> **Latest approved B2 actual:** [Unique native submission](reports/PI0294/design/parallel-science-20261022/B_APPROVED_SAVED2048_TO4096_UNIQUE_NATIVE_SUBMISSION_ACTUAL_20261023.json) is job `1480412.stdct-mgmt-02`, request9895f812, exact runtime07937041. Human saved-B2048 reuse grant918c is natively adopted in both ledgers and independently read back.935 code/972 sealed GPU members match; owner is healthy, actual Q/heartbeat confirmed. Cumulative settled8744updates/81773GPU-s/1197queries; global/DEC1PBS22/11 and reserved22560GPU-s/54000CPU-s/2048updates/29queries. No model/first-update release yet. New finite180LOCAL control is charged once; old1200LOCAL and full3600future-controller upper remain, no Source/C0 mirror replay. Exact cross-site cold failures were read back before separate known-state completion; original UNKNOWN journals remain. No restart/replay/hot edit. Prior F/Exit1 and all costs remain; expected10792 is conditional on actual success. All caps/clocks/final20GPUh40GiB unchanged; WDR1 target-qualified=False/core64=0. Earlier pending/no-native statements below are historical.

> **Approved saved-B2048 cold glue:** [CPU qualification](reports/PI0294/design/parallel-science-20261022/B_FAILED_SAVED2048_REUSE_RUNTIME_CPU_20261023.json) adds an exact human-grant/failed-closure binding, pure paired-ledger marker adoption and the existing segment intake exception. Full local1996tests/1412subtests pass; latest focused6/24 and actual captured dual-byte2/6 tests pass. No native adoption or B2 prepayment/registration/request/submission has occurred. Existing shared1200LOCAL cost and all original3600future-controller ceilings remain charged/retained; a separate finite180LOCAL control envelope (owner hard100, setup/terminal80) fits with36seconds conservative headroom and still needs enforcement qualification. Exact source CI/readback, cold bindings and fresh whole-work native admission precede training.

> **Human decision received:** reuse of the exact independently audited and fully closed failed-job B2048 (`db2b2be8…`,cursor65536) is explicitly approved for the original remaining B2048→4096 segment. [Narrow authorization](reports/PI0294/design/parallel-science-20261022/B_FAILED_SAVED2048_REUSE_AUTHORIZATION_20261023.json) preserves F/Exit1, all costs/history/checkpoints, existing caps/clocks and original science. No repeat approval is needed. Cold source/metadata integration, CPU tests, exact CI/readback, native once adoption and fresh all-remaining-resource admission still precede a unique new B2 request; no B2 native action has occurred. Earlier pending-decision text is history.

> **Current B recovery terminal:** job `1479559.stdct-mgmt-02` remains F/Exit1/HeartbeatLost and its owner remains failed/MainPID0. Independent full streaming SHA, original optimizer/RNG/sampler/recipe validation and original12probe reads prove complete B2048/cursor65536 (`db2b2be8…`). Actual1024new updates/17queries/2loads are settled once: cumulative8744updates/81773GPU-s/1197queries. GPU and both shared-prepaid CPU leases are CLOSED, independently read back; all reserves0/unknown[]. No owner/request/mirror replay or B2 native action. See [actual closed failure](reports/PI0294/design/parallel-science-20261022/B_RECOVERY_HEARTBEAT_FAILURE_SAVED2048_ACTUAL_20261023.json). The original F/Exit0/natural-owner-success B2 boundary is unsatisfied; [one narrow saved-state reuse exception](reports/PI0294/design/parallel-science-20261022/B_FAILED_SAVED2048_REUSE_EXCEPTION_DRAFT_20261023.json) is a review draft, not approval. It requests no cap or clock increase; pure all-remaining-work arithmetic fits, and fresh native admission still remains. Goal incomplete; target_task_native_qualified=False/core64=0. Earlier Q/healthy-owner entries are history.

> **Current native owner:** B recovery job `1479559.stdct-mgmt-02` is uniquely submitted and bound for existing request `185541e4-2348-4a3c-bee7-1a5703369780`. Actual Q, no model/first-update release yet, owner heartbeat confirmed. Service `vap1-a40-b-recovery1-preqsub-resume-01` reuses existing CPU prepayment; keep it running unchanged. Frozen GPU source2ed, fixed cold control e4bc; exact pure pre-qsub math2d4c CI39900376419 succeeded/all12/6full files. Old failed actors and UNKNOWN journals remain historical. See [actual submission](reports/PI0294/design/parallel-science-20261022/B_RECOVERY_SAME_REQUEST_UNIQUE_NATIVE_SUBMISSION_ACTUAL_20261023.json).

> **B recovery actual update:** grant09ae and closed Source02/C0 projections are adopted. Exact e4bc CI39899093916 passed all12 steps. The recorded split reservation has now been completed once and independently read back for the same request; both original UNKNOWN journals remain unchanged. No qsub/model/physics has occurred. Pure fresh pre-qsub arithmetic now credits the real reserved envelope without inventing a PBS job ID and preserves queue delay; exact CI for this change is pending. See [actual completion report](reports/PI0294/design/parallel-science-20261022/B_RECOVERY_KNOWN_PREQSUB_COMPLETION_ACTUAL_20261023.json).

# Current continuation: B recovery grant adopted and cold inputs sealed; pre-qsub reservation is being reconciled

[Actual adoption, cold readback and reservation boundary](reports/PI0294/design/parallel-science-20261022/B_RECOVERY_ACTUAL_ADOPTION_COLD_SEAL_AND_PREQSUB_RECONCILIATION_20261023.json): five approved caps are now adopted in both native ledgers. B1024/cursor32768 retains all15full-state members and original12probes; all964native sealed files match fixed2ed8aea9, whose exact CI39896527594/job120228160003 passed all12steps/28full-file readbacks. Closed Source02 and both C0 costs were projected once with no second global charge. No new GPU job or model has run.

The new cold owner stopped before qsub when the legacy total-source guard rejected its ninth retained allocation. Exact independent reads prove the global ledger still equals the original reservation intent and DEC1 equals its planned reservation; no qsub intent, reply, PBS script or model effect exists. Preserve the unknown journal and both actual bytes; do not replay reservation, restart the failed service or create another request. A narrow fix validates the exact B grant for at most10source rows and validates a complete proposal before any caller writes a ledger; old7/8limits and one active chain remain. Qualify the fix and explicit partial-transaction completion before resuming this same request through a new cold control boundary. The current2edcapsules remain immutable. All805lost paid updates, all checkpoints, original caps/clocks and final20GPUh/40GiB remain. WDR1/DEC1 goal ACTIVE, target task/core64 still incomplete. Earlier entries below are history.

# Current continuation: five B recovery changes approved; cold runtime prepared, native adoption pending

[Exact human authorization](reports/PI0294/design/parallel-science-20261022/B1_HEARTBEAT_RECOVERY_AUTHORIZATION_20261023.json) is published at09ae7896. DEC1 updates10792, DEC1/global PBS14/25 and GPU-site storage549/615GiB are approved; other resources, costs, checkpoints and clocks remain unchanged. Do not ask again. [Runtime CPU qualification](reports/PI0294/design/parallel-science-20261022/B1024_PRESERVED_COST_RECOVERY_RUNTIME_CPU_20261023.json) covers preserved-cost dual-ledger adoption, original full B1024/cursor32768 and12probes import, new B_COVERAGE_HEARTBEAT_RECOVERY1 and B2048 split-origin continuation. Full PI0294 CPU suite:1978passed/1350subtests,13private-parser skips. Captured actual closed ledger/terminal bytes pass pure adoption and original global validation with all used costs unchanged. This commit still needs exact CI and file readback; no new native grant adoption, registration or training has occurred.

B1 failure1829 and both closed leases remain immutable, charged7720updates/75658GPU-s/1180queries. Next: qualify the cold owner, adopt the exact grant once, project only unmirrored closed Source02 and both C0 costs at the actual failed-B inactive boundary, then fresh full native admission and unique1024→2048 request. Real B2048 full audit/settlement/dual closure precedes2048→4096. Original remaining matrix/final work remains, with336GPU-s DEC1 margin. Target counter qualification is real; target prefix/feedback/ACK/STOP/task and core64 remain incomplete. Goal ACTIVE. Earlier entries below are historical.

# Current training: B1 failed at1829; exact audit, settlement and both closures completed

[Actual B1 failed-state audit and closure](reports/PI0294/design/parallel-science-20261022/B1_FAILED1829_ACTUAL_AUDIT_AND_DUAL_CLOSURE_20261023.json): the existing Hopper qselect returned162 during a read-only snapshot. Its retained diagnostic has now been found; the underlying scheduler reason is unknown. The original owner exited; Vanda then reached F/Exit1 with HeartbeatLost at1829,29queries/4loads and no unknown effects. Independent streaming audit passed all15B1024 members/cursor32768 and original12probes. One14196GPU-s/15060CPU-s/70GB-bound settlement yields cumulative7720updates/75658GPU-s/1180queries. Both leases are CLOSED with independent byte readback; no old owner/request replay, no recovery or B2 submission.

[Exact recovery scope draft](reports/PI0294/design/parallel-science-20261022/B1_FAILED1829_RECOVERY_SCOPE_DRAFT_20261023.json) preserves805paid updates after B1024 and proposes a new B recovery lineage1024→2048→4096. Required five changes are updates9987→10792, DEC1/global PBS13/24→14/25, DEC1/global storage483/550→549/615GiB. Full original remaining work and final20GPUh/40GiB remain; all dimensions fit proposed arithmetic, with only336GPU-s of original DEC1 headroom. No approval/adoption/native admission is inferred. A narrow queue-read guard has17CPU cases: scheduler errors retain diagnostics and remain unknown queue state, use the existing bounded read-failure policy, and never make mutation retries. It is not natively deployed. Native grant/new-lineage integration/fresh full admission still precede training.

Actual target zero-time frozen scene and two RGB counter PASS remains valid, both counter requests closed; it is not task/prefix/ACK/STOP qualification. The prior2e25e709 document's R1786 is now historical; its exact CI39891948136/job120213527115 passed all12steps. Goal ACTIVE. Earlier entries below are history.

# Current target progress: actual frozen scene and dual RGB passed at zero simulation time

[Current registered target counter and B2 costs](reports/PI0294/design/WDR1_TARGET_ACTUAL_CURRENT_COUNTER_AND_CLOSED_B2_COSTS_20261023.json): a fresh DEC1 child actually loaded the frozen full-body plant and original task overlay, checked identical native robot properties and rendered both RGB cameras with software EGL. MuJoCo3.2.7, nq34/nv32/nu20; simulation time remained0, with no WBC/policy forward or physical episode. The first child failed module lookup because the standard entry replaces PYTHONPATH; its original failure and full financial upper remain closed. A separate admitted child fixed only the capsule bootstrap and passed. Both requests were settled once, actors terminal, leases CLOSED, all holds0 and independently read back. Counter PASS does not qualify native prefix safety, feedback/ACK/STOP, T1/T2 or the64cells.

B1 remains under its original healthy owner and immutable0cf86cbc. New cold B2 accounting includes both source closures and both counter costs:45isolated CPU cases pass. At actual B2048 closure, apply the original Source02 mirror once, then the separate counter mirror once, and only then original full fresh B2 admission. Global costs are already charged; retained source output7.012GB and counter output128MiB enter Core storage once, LOCAL8GiB remains inclusive of all eight namespaces. No Hopper mirror or B2 submission has run. Qualified counter runtime06afc60c exact CI39889132414/job120204381902 passed all12steps/12full-file readbacks. Prior a6bd document CI39885603759/job120192948801 is also all12success. Goal ACTIVE; all budgets, original clocks, science and checkpoints remain. Earlier entries below are history.

# Current target work: bind the provided MPC prediction and preserve native float32 transport

[Actual native prediction seam and float32 transport report](reports/PI0294/design/WDR1_NATIVE_MPC_PREDICTION_BINDING_AND_FLOAT32_TRANSPORT_20261023.json): installed MpcFlattenedController includes the solver's actual target, initial observation and predicted state/input trajectories. A thin binding now rejects stale target/observation identities, wrong model dimensions, malformed float32 vectors and incomplete0.4s horizons. It uses the supplied MPC output, not a new controller or source S1. This12D model excludes leg/contact dynamics, so binding a plan does not qualify full-body safety or physical dispatch.

Actual installed MpcState/MpcInput are float32[] while times are float64[]. The prior2dcb ACK checker compared to double poses and could falsely reject real ROS output. It now reproduces original transport serialization without widening safety thresholds. An actual closed native41point/14D EE+BASE trace replay through the frozen parser independently confirms exact float32 equality and maximum2.98e-8double discrepancy. Historical quality FAIL/closure remains unchanged; no pilot rerun.182related CPU cases pass, including11paused-planning and12actual-parser changing-SO3/frame/clock cases and one-native-ULP substitution rejection. A separate PausedReferenceCandidate now lets the native solver plan only under the original frozen pause barrier and fresh unchanged native step, while explicitly retaining prefix validation pending and no physical quota. This avoids requiring a prediction before its candidate exists. Current native policy, prefix safety, STOP and task qualification remain pending; no new physics/model/GPU/PBS. Prior36b364c3 code CI39884421946/job120189140121 passed all12steps/8full-file SHA readbacks. The latest comparison also reproduces the frozen82004c31 mjData-to-ROS nanosecond clock rule, retains both raw timestamps and the same25ms limit; fixed20139662 exact CI39885088506/job120191290395 passed all12steps/8full-file SHA readbacks. Its clock comparison return expression exactly matches the frozen source AST, and the actual original require_pause function passed the CPU planning fixture. Prior document eef24f6b CI39882400419/job120182589505 passed all12steps. B1 continues under its original owner/source; goal ACTIVE. Earlier entries below are historical.

# Current preparation: shared DEC1 reference connected to existing target time receiver

[Target time-reference transaction CPU report](reports/PI0294/design/WDR1_TARGET_SHARED_REFERENCE_TIME_TRANSACTION_CPU_20261023.json): both I_TIME and I_PROGRESS now pass through the existing V2 mapper, full raw/mapped H40 gates, explicit target-native prefix validation callback, original gripper law and EEF-only41point converter. Exact wire/clock/TTL, original native time grid and interpolated installed poses are checked against accept+installed trace rows. No new controller or runner.30new CPU cases/132related cases pass, including6checks against the byte-pinned private parser at0/12.5/25ms for both modes. Prefix callbacks and trace rows remain synthetic CPU fixtures, never native qualification or permission to advance physics. Original shared .01m lookahead is preserved.

Actual target-native prefix/feedback/install ACK/STOP qualification and current admitted resource binding still remain. B1 is protected under unchanged0cf86cbc; real B2048 full audit, once settlement and both closures precede inactive Source02 cost mirror and fresh B2 admission. Fixed2dcb019d code CI39882060203/job120181476997 passed all12steps and5full-file SHA readbacks; prior447bc1f5 document CI39880696923/job120177101553 now passed all12steps. Goal ACTIVE. Earlier entries below are historical.

# Current target evidence: frozen runtime located and byte-verified

[Actual target assets and native time-wire CPU qualification](reports/PI0294/design/WDR1_ACTUAL_FROZEN_TARGET_READBACK_AND_TIME_WIRE_CPU_20261023.json): actual registered metadata,356sealed code members,9native config/weights/executables,5clock/pause/adapter binaries and all14startup pins including the74MBplant match. The geometry root differs from the actual native execution root in the startup config; no missing WBC runtime is inferred from geometry-only paths. Current process view has0matching native workers. Original task pilot4/0success and expired window are preserved, never retried or relabeled current DEC1 results.

The actual private pure parser passed local CPU tests using the existing H40→41point world converter and original receiver: first action preserved, final hold, no BASE padding,11invalid clock/identity/grid/tail cases rejected. No ROS node, model, controller or physics started. Fixed5447d639 CI39880036692/job120174926979 passed all12steps/5full-file readbacks. Actual Core standard-entry CPU revalidation also PASS:41poses/11invalid cases,0.157063CPU s; control manifest/lease/registry/DEC1ledger unchanged, no model/simulator/ROS packages or external/write attempts. This is CPU wire qualification; native ACK/stop/prefix/task qualification remains pending. Future target bindings must preserve actual frozen task_clock82004c31 rather than silently use management HEAD9f4286bd. B1 remains protected and independent.

# Current preparation: existing target identity and future B2 cold entry

[Existing target identity and B2 cold entry CPU report](reports/PI0294/design/WDR1_EXISTING_TARGET_IDENTITY_AND_B2_COLD_ENTRY_CPU_ACTUAL_20261023.json): existing full-body MuJoCo WBC task entry, native41-point world-reference converter, gripper/clock/ACK/STOP seams are identified.17new identity cases and36related cases/46subtests pass. The historical task window remains closed; source FD2 metadata is not target WBC qualification. No new controller, manifest, role or native target execution occurred. Current target task qualification remains pending and does not block source training.

Future B2 cold context/entry9cases and explicit launcher14cases plus9inactive-mirror cases pass; actual immutable dual-closure bytes are checked separately from fresh native admission. Every B2 action rejects before original owner import while actual B2048 terminal receipts are absent. The new wrapper preserves the qualified dual-source storage callback in its future serve child without replacing fixed0cf numerical modules. Actual B2048 full audit/once settlement/both closures and natural owner exit, one inactive-Hopper Source02 cost mirror, then full fresh resource/site admission still precede B2. No B2 native write or B1 hot change occurred. Fixed53698763 CI39877207308 passed all12steps/9full-file readbacks; old5de failure retained. Target inventory code fixed36e794ee CI39877705945/job120167250877 passed all12steps/6full-file readbacks. The private inactive mirror passed actual Journal/once-CAS temp fixtures including lost-return reconciliation; no native mirror ran. It must precede any new B2 prepay/registry/request after real B2048 closure. Goal ACTIVE. Earlier entries below are historical.

# Current source continuation: four known safety rejections, actual dual closure

[Source02 actual finite results, independent closure and cold B2 cost/report contracts](reports/PI0294/design/WDR1_SOURCE02_FINITE_NATIVE_PROGRESS_AND_B2_CLOSED_COSTS_CPU_ACTUAL_20261023.json): all four fixed Source02 conditions ended with independently audited known S1 rejection and real returned STOP ACK; task successes0. The owner naturally exited, actual Core/LOCAL settlement ran once, leaseCLOSED, all8pending0. Independent actual readback verified10terminal actor records/138attempt files and preserved the old Source01 UNKNOWN/13failed files/16000historical liability unchanged. New actual live6800/shadow7331 were charged separately (14131ticks), never relabeled the old16490financial upper. Do not restart or replay Source02.

New source cost/storage and post-settlement aggregation contracts passed302related CPU tests/19subtests. Initial5de4ae8c CI39876083728 failed35 tests because the new test audit hook affected unrelated tests in the same process. The scoped-fixture fix now passes the full workflow suite locally:1828tests/1284subtests; fixed53698763 CI39877207308/job120165586286 now passed all12steps and9full-file readbacks. No production or active runtime changed. Actual dual receipt/intent/after-ledger and audit bytes aggregate5historical attempts:1retained engineering UNKNOWN+4known S1 rejections, exactly1engineering replacement. Actual closed-source projection prepares only the missing4episodes/14131ticks/44constructors/1GiBtransfer and3.506GBnew retained Core output for the future B2 inactive boundary; no native Hopper or global write occurred. B1 remains R/476returned updates under unchanged0cf86cbc with no restart/hot change; source success is not a B gate. Existing frozen WBC task/student/core64 qualification remains incomplete. New cost/storage/report code is qualified at fixed53698763. Earlier entries are historical.

# Current source continuation: native metadata and zero-integration counter passed

[Source02 actual metadata/counter and known-terminal CPU contracts](reports/PI0294/design/WDR1_SOURCE02_NATIVE_METADATA_COUNTER_AND_KNOWN_TERMINAL_CPU_ACTUAL_20261023.json): Source02 was registered once with937sealed files fully read back and both resource holds bound. Original zero-integration native counter passed with2constructors, identical native fingerprint/two RGB cameras and Exit0. Four source conditions remain NOT_RUN. The original failed Source01 UNKNOWN,13failed files, closed lease and all financial costs remain immutable.

Known-terminal pure proposal now has21CPU cases;261related cases and7private metadata-only owner test methods pass. Actual live/shadow costs stay separate; old16490financial upper stays unchanged. All actual attempt PID/start/exit pairs and independent byte audit are required; unknown writes never retry. New code still requires exact CI/full readback before adoption. B1 continues R with296actual returned updates, unchanged0cf86cbc/owner567538; no restart/hot change. Source success is not a B gate. Native component dispatch/terminal close and target/student/core64 qualification remain incomplete. Earlier entries below are historical.

# Current source continuation: actual read-only check and full cold input package

[Actual closed-source inventory and offline continuation inputs](reports/PI0294/design/WDR1_SOURCE_COLD_HISTORY_NATIVE_READONLY_AND_OFFLINE_INPUTS_ACTUAL_20261023.json): qualified47dc6001 passed exactCI39871262132/job120145986402/all12steps and25full-file readbacks. Actual Core read-only check verified26assets,13failed artifacts, exact old UNKNOWN ledger and CLOSED lease,0active source workers/0other unknowns. No native resource or lease writes occurred.

New Source02 request081c5299-5594-4ff6-8676-d48b9652cf52 has913code/937sealed inputs and passed the original complete pure intake plus dual reservation proposal using the fresh actual snapshots. It remains OFFLINE, without native registration/reservation/lease/counter/components. Cold once owner, active-hold intake and truthful terminal closure must be completed before physical dispatch. B1 remains healthy under unchanged0cf86cbc, with actual172returned updates/17queries/2loads/unknown[]; source qualification is not a new B gate. Frozen target task/neural/core64 qualification remains incomplete. All original scope/clocks/costs remain. Earlier snapshots below are historical.

# Current source engineering: exact closed-history cold intake implemented

[Closed-source history, slot and financial-upper cold integration](reports/PI0294/design/WDR1_SOURCE_CLOSED_HISTORY_COLD_INTAKE_CPU_ACTUAL_20261023.json):240 related CPU tests pass/0external or native attempts. Intake, once dual reservation and runtime preserve the actual old requested4/UNKNOWN/costs while counting charged commissioning1, remaining originals3 and at most one separately bound engineering replacement. Five actual historical byte pins and the exact tested STOP repair/report are required; financial upper16490 is counted once, never relabeled actual live. Expired/UNKNOWN state remains read-only metadata closeout only. No new source child or physical continuation has run.

B1 **1478518** remains healthy R under unchanged0cf86cbc/owner567538; same-receipt returned78updates/17queries/2loads/unknown[], with an active next update. A4096 remains fully closed. Source-owner cold metadata/cross-domain holds/fresh native admission and separate frozen-target task qualification remain required; budgets, clocks and frozen science are unchanged. Earlier entries below are historical.

# Current experiment: A4096 fully closed; independent B1 running

[A4096 actual audit, settlement, dual closure and B1 adoption](reports/PI0294/design/parallel-science-20261022/A4096_ACTUAL_CLOSURE_AND_B1_NATIVE_ADOPTION_20261023.json): A1477527 is F/Exit0, original owner naturally exited; three15-member states and original12probe records independently verified. One settlement retained all history/checkpoints, cumulative5891updates/61462GPU seconds/1151queries, both leases CLOSED. Closed Core costs were projected once into inactive Hopper DEC1 with no second global charge.

B1 **1478518.stdct-mgmt-02** is actual R on GN-A40-016, independent original CORR0 start0→2048 under unchanged0cf86cbc, request22cdd5d4-c735-4dfc-901c-3b9dda2ca246. Unique service vap1-a40-zero-update-b1-owner-01/PID567538 is healthy;888 sealed members read back, full remaining budget/clock admissible, final20GPUh/40GiB protected. Native compute-ready identity verified; original owner released model start, training updates not yet observed. Reuse this owner and request; actual B2048 audit/settlement/both CLOSED leases precede B2.

Two existing helpers also prepared finite source continuation classification and multi-child aggregation:15+14 pure CPU tests pass/0external attempts. Intake financial-upper and historical slot accounting still require cold code integration/native qualification; old Source UNKNOWN and three NOT_RUN remain. Existing WBC interfaces are reused, student/target task qualification and64cells remain incomplete. All original caps/clocks/freezes persist; goal ACTIVE. Earlier entries are historical snapshots.

# Current training boundary: A native F / Exit0; independent saved-state audit in progress

[Cold B fee-catchup and snapshot CPU preparation](reports/PI0294/design/parallel-science-20261022/B_CLOSED_SOURCE_COLD_NATIVE_AND_SNAPSHOT_PREPARATION_CPU_20261023.json):10 private synthetic/filesystem tests with25subcases pass,0external/native attempts. The existing B1 journal gets a once inactive-Hopper metadata CAS; active A/CPU/GPU holds/open lease/changed bytes reject before write, lost write return cannot be replayed. Only closed Core-side cumulative costs are projected; global neural ledger remains byte-identical.

B1/B2 preserve original0cf86cbc numerical source and mature once owner. Closed Core output is included in the existing storage peaks before original full-budget recomputation; LOCAL8GiB remains inclusive only with fresh known-namespace allocation readback. Explicit task-specific cold launchers retain this callback in actual serve. Missing real predecessor receipts reject before historical owner import/configure/transport. No B registration, Hopper cost catchup or submission has run.

A1477527 is actual native F/Exit0 at4096, and the original service naturally exited successfully with MainPID0. The original once-only independent full-state audit is now running; settlement and both lease closures have not yet run. Priorc2ac3ff6 exactCI39864981215/job120124778381/all12steps and8file readback passed. Source financial closure remains confirmed; raw source UNKNOWN and failed artifacts remain. All budgets, freezes and clocks are unchanged. Older entries below are historical.

# Current source continuation: actual dual financial close; B cost catchup prepared

[Actual dual financial closure and cold B cost projection](reports/PI0294/design/WDR1_SOURCE_DUAL_FINANCIAL_CLOSURE_AND_B_COST_PROJECTION_CPU_ACTUAL_20261023.json): Source request23da1121 was once settled at1792707417.8988905 on Core and LOCAL; original source lease isCLOSED. Independent actual readback confirms all8 source holds0, both source actors terminal,13 original failed files and the original child ledger unchanged. The child still retains UNKNOWN and its16000 live reservation as historical evidence. Scientific state remains1 ENGINEERING_INCOMPLETE_UNKNOWN+3 NOT_RUN; no STOP ACK or qualification is invented.

Financial charge is1039.537288565 CoreCPU/720 localCPU,1 attempt/6constructors and16490 total tick upper (16000 unresolved live+490 known shadow). Unstarted work was unreserved only after actual actors were terminal and lease closed; no failure charge was refunded. Old global costs, A runtime and all checkpoints remain. Financial source751e0d62 passed exactCI39863169664/job120118710154/all12steps and all7file readbacks.

The cumulative closed-Core cost projection is CPU prepared:40 tests/19subtests with0external/native attempts. It adds only cumulative Core-side nonGPU costs to the inactive DEC1 projection without charging global parent again or touching GPU-site CPU/models. It requires actual A4096 F/Exit0/full-state audit/once settlement/both CLOSED leases/natural owner exit and rejects active GPU reservations. No Hopper projection or B registration/submission has executed. Same A1477527 remains R/step3925/unknown[] under0cf86cbc/owner431568. Next: exact projection publication/CI and once cold native adoption at the actual A boundary; original CORR/R0/clock/budget/safety and separate target scientific requirements remain. Older entries below are historical.

# Current source continuation: terminal financial bounds prepared; scientific UNKNOWN preserved

[Source financial closeout CPU and actual readonly preparation](reports/PI0294/design/WDR1_SOURCE_FINANCIAL_TERMINAL_BOUND_CLOSEOUT_CPU_PREPARATION_20261023.json):135 combined CPU tests pass with0 native/external attempts, including24 closeout cases;7 private fake-backend cases cover once settlement, live/stale evidence rejection and lost Core/lease write returns. These are not actual settlements. The proposed financial charge covers the complete16000 uncertain live bound plus490 known shadow ticks,1 actual attempt/6 constructors, measured worker CPU with original registration/control/audit upper allowances. The original failed child ledger/UNKNOWN and source files remain immutable; no STOP ACK or qualification is synthesized. Only proven unstarted work may be unreserved after actual terminal actors and original lease closure.

Source repair c296d730 passed exactCI39860658998/job120110199920/all12steps and all21 files full readback. Repair is still not natively deployed. Fresh Core read confirms4 actor/exit records all terminal,13 original files hashed and original400x45 highrate, with Source lease stillACTIVE and original holds intact. Native financial execute has not run. Current A1477527 remains R/step3754/unknown[] under unchanged0cf86cbc/owner431568. Next: current exact publication/CI, once Core charge+lease close then confirmed LOCAL settlement and independent readback, with complete Core-source cost classification retained for B. Actual A4096 full-state successful close still precedes B; all scopes/clocks/freezes remain. Older entries below are historical.

# Current source continuation: native counter passed; first component stopped with engineering uncertainty

[Actual native registration, counter, first attempt and CPU STOP repair](reports/PI0294/design/WDR1_SOURCE_COMPONENT_NATIVE_FIRST_ATTEMPT_AND_STOP_READBACK_CPU_ACTUAL_20261023.json): fixed55e73c7b passed exactCI39858937721/job120104294749/all12steps; all24 code/report/runtime files were fully read back. The actual CPU source child,928 sealed files and parent/Core resource holds were registered once and independently verified. Original zero-integration counter passed with2 constructors, identical native model state and both cameras, Exit0.

First fixed T1/I_TIME component exited1 after400 live and490 shadow integrations, with2 complete actual prefixes. A third-shadow native S1 rejection entered STOP, whose mutating snapshot readback failed the original fingerprint; the later error is ENGINEERING_INCOMPLETE_UNKNOWN. The original16000-live liability and all resource holds/unknowns remain; other3 conditions are NOT_RUN. No STOP ACK, task success or student/target qualification is fabricated, and no condition is replayed.

A cold STOP metadata fix now observes native data.time/steps without calling snapshot(), which invokes mj_forward. It retains frozen-function identity, complete integration/Python/RNG fingerprint, owned latch and once durable readback.107 related CPU tests pass, including3 new regressions; repair has not been natively deployed or rerun. Continue precise read-only effect/accounting reconciliation and same-scope cold engineering. Same A1477527 remains R/step3547/unknown[] under unchanged0cf86cbc/owner431568; A4096 real successful close still precedes independent B. Budgets, original four-source/four-target plan, CORR/R0, WBC/S1 and clocks remain. Older entries below are historical.

# Current source continuation: once resource reservation and independent terminal evidence

[Once-reservation and terminal CPU receipt](reports/PI0294/design/WDR1_SOURCE_ONCE_RESERVATION_AND_TERMINAL_CPU_ACTUAL_20261023.json):375 combined CPU cases and9 subtests pass with0 external/native attempts; four additional private metadata-controller tests use temporary files only. The typed parent/Core proposal preserves all historical costs, uses formal original CPU/episode/tick/constructor holds and keeps the complete IO index. Partial writes retain reservations, forbid dispatch and permit read-only reconciliation, never automatic replay.

Source component finish now pins actual owner-confirmed event offset/sequence/hash after close, reconciliation and returned STOP. Independent audit checks full H40 gates, actual prefix ACK/physics, original native counters and STOP; missing/truncated evidence remains UNKNOWN. Teacher task outcome stays diagnostic; no student or target qualification follows. Fixed source4/target4 and the original plan65f9d193 remain.

Prior80df3458 exactCI39854880873/job120090381766 passed all12steps, all7changed files and11runtime dependencies fully read back. Fresh actual Core input read verifies26 assets, original entry/doctor and0 active registered workers. Full eight-dimensional offline budget is prepared, including closed VAP1 CPU retained once. No native source reservation, child, counter or episode has yet run; old80df payload deliberately cannot be registered without this new tested code. Same A1477527 continues R/step3392/unknown[] under immutable0cf86cbc/owner431568. Next: exact current publication/CI, fresh complete resource intake, once native metadata registration, original native counter and finite source components with independent audit/closure. A4096 actual successful closure still precedes independent B; original CORR/R0, WBC/S1, all budgets and clocks stay fixed. Older entries below are historical.

# Current source continuation: reuse the exact existing VAP1 CPU/IO amendment

[Existing adopted caps CPU receipt](reports/PI0294/design/WDR1_SOURCE_EXISTING_VAP1_CAPS_CPU_ACTUAL_20261023.json):338 combined CPU tests and9 subtests pass, including14 amendment cases, with0 external/native attempts. A stale legacy96h Core/8h local/64GiB Core+transfer cap was found in new intake before native use. Exact previously adopted43f7fe6f bytes (SHAa26aeb50), fresh native readback and explicit sealed-manifest amendment now select the already approved160h/16h/128GiB cumulative caps. Old costs remain counted; no adopted delta means original caps. This creates no new budget or permission request.

Previous392b9a62 shared-entry/factory/runtime passed exactCI39854114371/job120087722050, all12steps; all18 changed files and3 additional runtime dependencies fully read back. Fresh Core read at1792703230.7024837 confirms the same adopted bytes, actual native CPU/IO caps,540s Core DEC1 cost andunknown[], with0 mutations. No source CPU child/physics or B was started. Same A1477527 continues R, local watcher step3049/unknown[] at1792703263.532339; runtime and owner remain unchanged. Next: complete current resource classification and once cross-ledger reservation/registration, actual original native counter and four bounded components, independent terminal audit. Helper reservation proposal remains text-only and unadopted; source/target neural64 remains incomplete. All other grants, clocks and freezes stay fixed. Older entries below are historical.

# Current source continuation: original shared entry, CPU runtime and Teacher component factory wired

[Shared-entry/runtime/factory CPU receipt](reports/PI0294/design/WDR1_SOURCE_SHARED_ENTRY_FACTORY_RUNTIME_CPU_ACTUAL_20261023.json):324 unique tests and9 subtests passed under external/network/native-import denial, with0 denied attempts. The new explicit CPU component role reuses existing PV1 scope/runtime/core/ledger and original episode; old roles retain original15-file checks. Only the new role accepts the exact13 original files plus2 recorded glue deltas. The fixed four-condition plan and source4/target4 slots remain unchanged.

Original Teacher factory, one shadow/prefix, actual live counter, full raw/mapped H40 gates, durable ACK/S1 and returned STOP are connected. Both modes derive the .4s stale limit from original ProgressConfig. Compound interruption cleanup retains original interruption and actual lower bounds. Only component metadata closeout can pass expired worker/hard deadlines and actual cost overruns; no new constructor/physics permission follows. Tests use synthetic physics, not native qualification.

Previousb0e7b674 passed exactCI39851076725/job120077087239, all12steps/all7files fully read back. Same A1477527 remains R, latest local watcher step2992/unknown[] at1792702953.7995377, original owner431568 active and runtime0cf86cbc unchanged. No B or source CPU native child was registered/dispatched. Next: exact publication verification, complete fresh parent/site budget, once native reservation and existing child registration, original zero-integration counter/native qualification, bounded component attempts and independent terminal audit. Source/target neural64-cell qualification remains incomplete. Budgets, clocks, original CORR/R0 and all WBC/scoring/S1 freezes stay fixed. Older entries below are historical.

# Current source continuation: pure CPU intake and real component-ACK IO prepared

[CPU intake and component ACK receipt](reports/PI0294/design/WDR1_SOURCE_CPU_INTAKE_COMPONENT_ACK_CPU_ACTUAL_20261023.json):264 combined CPU tests passed, including33 pure-intake cases and50 independent-Teacher component ACK cases. The exact four-condition seal remains65f9d193/owner80854c8d; future runtime release is separate. Intake requires complete parent/DEC1/site resources and exact13+2 source identity; a healthy active A does not block a zero-GPU source CPU role. Its return explicitly is NOT_NATIVE_ADMISSION.

The original Teacher's full raw H40 is durably prebound before shadow/live; actual adapter mode/final ACK, one200-tick shadow,200 actual rows,8 actual reference ACKs, original S1/parity and persisted source events are bound and read back before receipt return. No neural identity/relay pending is fabricated. Two tests use the actual durable event-chain reader with the ACK callback; all physics in fixtures is synthetic. Original CORR/R0 neural qualification remains separate.

Source5e344282 passed exactCI39849902290/job120072955878, all12steps/all7files full readback. Fresh Core inventory has0 active registered workers, parent STOP absent, no child unknown effects or SOURCE_PHYSICS_STOP. All seven historical policy-child STOP markers are preserved: their leases are actually CLOSED, reservations/unknowns empty and SOURCE_PHYSICS_STOP absent. Old failures and171600+38000 recorded live ticks are not relabelled or replayed. No new source child or physical worker is registered/dispatched. Next: existing-entry runtime/role validation, complete native reservation and original episode factory/S1/stop. Same A1477527 immutable0cf86cbc continues; all budgets/freezes/clocks and later B ordering remain. Older snapshots below are historical.

# Current source continuation: four slots sealed, original ledger callbacks and durable events CPU tested

[Actual component seal and ledger/IO receipt](reports/PI0294/design/WDR1_SOURCE_COMPONENT_SEAL_AND_EXISTING_LEDGER_IO_CPU_ACTUAL_20261023.json): the four original source slots are now privately sealed once, SHA65f9d193239105ca3ff78916bb468856d002938981c1d75cb65bdedb4c6307ad, from fixed80854c8d owner predeclaration. Its exactCI39848664403/job120068649927 passed all12steps and all11 changed files were fully read back. A later runtime code release is pinned separately; conditions are not reselected.

35 CPU tests passed for existing Core ledger callbacks and durable event IO, including actual Ledger.steps fixtures: live remains one outer debit; known partial shadow rejection settles actual counts; unknown effects retain both original bounds and are not retried. Append records require fsync, exact byte readback and complete sequence/hash-chain validation. These are component preparations, not native execution evidence. Class/clone compatibility is statically consistent; actual snapshot fingerprint idempotence remains native qualification.

Healthy A1477527 remains under the same immutable0cf86cbc owner; no B or source component native worker was started. Continue the explicit CPU role/intake, original entry registration/fresh reservation and original episode factory. Source4/target4, no GR00T in this component stage, original CORR/R0 matrix and all science/resource/clock boundaries remain. Older snapshots below are historical.

# Current engineering continuation: source component plan, stop and interruption accounting

[Source plan/stop/interruption CPU receipt](reports/PI0294/design/WDR1_SOURCE_COMPONENT_PLAN_STOP_INTERRUPT_CPU_ACTUAL_20261023.json):146 tests passed with zero external/native attempts. Four predeclared Teacher-driven source commissioning slots use two existing TRAIN geometries and balanced I_TIME/I_PROGRESS assignment; four target slots remain reserved. This qualifies components only, with zero GR00T queries and no new training data. Synchronous stop and actual partial-step accounting are CPU verified; native source compatibility and dispatch remain pending.

[Original source pins and actual Core parent read](reports/PI0294/design/WDR1_SOURCE_ORIGINAL_PINS_AND_CORE_PARENT_READONLY_ACTUAL_20261023.json):13 original files remain byte-identical; episode.py and relay_protocol.py have two explicitly recorded interface-glue deltas. New DEC1 CPU intake must verify those deltas without weakening old PV1 admission. Core parent control lease is active, STOP absent and DEC1 physics0. This is control preflight, not a newly registered scientific child or complete resource admission.

Prior source4893b721 passed exactCI39845583235/job120057874835 (all12steps), all10 changed files fully read back. Same healthy A1477527 continues immutable0cf86cbc; no B or new component worker dispatched. Next: publish/read back this source, seal the private component plan, then complete the versioned CPU role and existing parent/DEC1 admission. Original clocks, costs, checkpoints and freezes remain. Older snapshots below are historical.

# Current engineering continuation: mature PV1 seams and existing EEF wire converter

[Mature PV1 mapped ACK seams](reports/PI0294/design/WDR1_MATURE_PV1_MAPPED_ACK_SEAMS_CPU_ACTUAL_20261023.json) passed34 CPU tests/5subtests: actual resident.QueryReservations, e1.neural.serve original sealed packets and existing relay_transport owner route now have explicit optional contract injection. Default RAW and old admissions remain strict. This implements the precise previously identified mature-route seam; actual DEC1 evaluation admission/factory and source stop/meter/persistence remain pending.

[Existing target EEF10 wire converter](reports/PI0294/design/WDR1_TARGET_EXISTING_EEF10_WIRE_CPU_ACTUAL_20261023.json) passed8 CPU tests/6subtests, with1 explicitly skipped private-native-parser test. The existing world41-point EEF serialization is shared with original EEF13; EEF10 requires no fabricated BASE labels. Frame/TCP qualification, original safety/gripper/install ACK/clock/stop and frozen target T1/T2 commissioning remain mandatory. No native qualification is inferred from conversion.

Healthy current A job1477527 keeps immutable0cf86cbc runtime and original owner/grant/clock. Both existing helpers are reused only for source commissioning/seal intake and synchronous stop preparation. No B/native evaluation submission occurred. Older snapshots below are historical.

[Exact mapped-ACK CI/readback and mature PV1 route gap](reports/PI0294/design/WDR1_MAPPED_ACK_EXACT_CI_READBACK_AND_MATURE_ROUTE_GAP_20261023.json): source641acd1be36d4525846830a208e0badf06e3d988 passedCI39844023081/job120052390525;14 changed files fully read back. Generic pi0292 query consumer and RelayExchange are CPU verified. Mature PV1 actually uses resident.QueryReservations through e1.neural.serve; that resident seam and relay_transport.run_episode cold injection remain unfinished. The existing science helper is preparing this precise delta. Actual sealed evaluation panel/condition and native commissioning remain pending. This is not a source/target task PASS.

# Current continuation: A running; B chain and mapped ACK consumers CPU verified

[B2 current-scope cold owner and terminal](reports/PI0294/design/parallel-science-20261022/B2_ZERO_SCOPE_OWNER_TERMINAL_CPU_ACTUAL_20261023.json) passed10 isolated CPU tests, following [B1 terminal/B2 full-state metadata](reports/PI0294/design/parallel-science-20261022/B1_ZERO_SCOPE_TERMINAL_B2_METADATA_CPU_ACTUAL_20261023.json), also10 tests. Current B1 V5 preparation passed17. No B registration, submission or training has occurred. Actual A4096 full audit/once settlement/dual closure precedes B1; actual B2048 closure precedes B2. Cumulative endpoints remain5891/7939/9987, old costs and all checkpoint identities preserved.

[Mapped ACK consumer preparation](reports/PI0294/design/WDR1_MAPPED_ACK_CONSUMERS_CPU_ACTUAL_20261023.json) passed168 CPU tests/35 subtests plus a separate clean-import assertion. Both optional consumers retain the original full response SHA and complete previous ACK, while binding mapped actions, eight actual reference ACKs, original S1 and physics receipt. Default RAW protocol remains strict. Actual sealed condition, cold owner factory/persistence/stop bindings and finite native commissioning remain pending; this is not source or target task qualification. Existing frozen target WBC interfaces remain a separate qualification path. Prior episode implementationcbc02254 passed exactCI39842134018/job120045666146, all11 changed files fully read back.

Same job1477527 and healthy service `vap1-a40-zero-update-resume-owner-01` continue with fixed0cf86cbc source; no restart, hot edit or duplicate submission. Grant871094ee remains adopted, updates9987/DEC1 PBS13/storage483GiB/global550GiB; original clocks and other caps unchanged. The two existing helpers have delivered text, independently integrated and tested by the sole owner. Older snapshots below are historical.

# Current native entry: approved successor1477527 running under one healthy owner

[B1 zero-scope V5 cold preparation](reports/PI0294/design/parallel-science-20261022/B1_ZERO_SCOPE_V5_COLD_PREPARATION_CPU_ACTUAL_20261023.json) passed17 isolated CPU tests, with27 small inputs/856 fixed0cf sources, independent original CORR0 and the unfinishedA block retained. All native entrypoints require actual A4096 audit/once settlement/dual closure; no B registration or submission occurred. B1 terminal/B2 migration and mapped-ACK consumer binding continue in the same two isolated helpers.

[Actual first training updates](reports/PI0294/design/parallel-science-20261022/ZERO_UPDATE_RESUME_FIRST_UPDATE_NATIVE_ACTUAL_20261023.json): same1477527 has returned native step2057 after complete same-job premodel and17-query compatibility qualification,2 model loads and actual first-update budget admission. Global settlement remains3843 until terminal closeout; newest full checkpoint remains2048. This is ongoing training, not A4096 or task success.

[Existing source episode wiring](reports/PI0294/design/WDR1_SOURCE_EPISODE_WIRING_CPU_PREPARATION_20261023.json) passed92 combined CPU tests: one shadow/prefix, original outer live tick/scorer, raw planned H40 and actual mapped committed labels. Typed raw→mapped ACK rejects no-op/raw-only receipts and retains unknown ACK state. Cold owner callbacks, protocol-consumer binding and original finite native qualification remain pending; this is no native/source-task PASS.

[Source single-pose Prefix8 CPU implementation](reports/PI0294/design/WDR1_SINGLE_POSE_PREFIX8_CPU_PREPARATION_20261023.json) passed20 new/76 combined existing progress contracts, including the actual original full-H40 gate. Published implementation757a5817 passed exactCI39839902342/job120037843920 and5-file full readback; this adds a CPU regression case without changing its numerical implementation. It uses one200tick shadow and200live, full raw/mapped H40, original native parity and8 actual ACKs. Existing episode wiring and native qualification remain pending; no physics/model/PBS was run by the tests, no active training capsule changed.

[Actual authorization, cold registration and unique native submission](reports/PI0294/design/parallel-science-20261022/ZERO_UPDATE_RESUME_NATIVE_ADOPTION_ACTUAL_20261023.json): job **1477527.stdct-mgmt-02** is R, request56d0211f-751f-44b9-945a-27367cf6c7ab, immutable runtime **0cf86cbc67a3a9639488d1b5080284f45ddad40e**, owner `vap1-a40-zero-update-resume-owner-01` active. START_NEURAL confirmed; no independently audited new training updates yet. Both-site authenticated mux-only transport preserves original RPC/watchdog limits. Do not restart, duplicate-submit or hot-edit. Grant871094ee is adopted in both ledgers: DEC1 PBS13/storage483GiB, global550GiB, updates9987. Exact CI39837369784/job120029092603 passed all12steps;14-file full readback,260 CPU contracts/421subtests,5 private owner and9 terminal tests. Partial registration stopped on a missing GPU grant mirror; all856 source files/891 sealed inputs and generated bytecode were read back before a separate missing-tail completion. Original registration was not replayed. Spent3843updates/49117GPU seconds remain; globalPBS19/DEC1PBS8. Old1477127 zero-update failure and old1024/2819 histories/charges/files remain closed and preserved.

A2048→4096 must actually finish and pass independent full-state audit, once settlement and both lease closures before independent B(CORR0)→2048→4096. Expected successful cumulative updates5891, then7939/9987; CPU fixtures are not native endpoints. Continue source single-pose Prefix8 and existing target WBC T1/T2 qualification separately. All freezes, original clocks and final20h/40GiB remain. Older snapshots below are historical.

# Current recovery authority: three narrow changes approved; cold source verified locally

The human explicitly approved [zero-update preserved-cost recovery authority871094ee](reports/PI0294/design/parallel-science-20261022/ZERO_UPDATE_RESUME_AUTHORIZATION_20261023.json): DEC1 PBS13, GPU-site output483GiB and global550GiB; cumulative updates9987 and all other caps, checkpoints and original clocks unchanged. No further approval is needed for this same scope. [Complete cold-call-chain CPU preparation](reports/PI0294/design/parallel-science-20261022/ZERO_UPDATE_RESUME_CALL_CHAIN_CPU_20261023.json) passed260 contract tests/421 subtests plus4 isolated private owner tests. This is not native adoption, registration or a new job. The old1477127 request and both leases remain closed, with0 new updates and preserved cost. Next: exact-source CI/readback, immutable cold code, native prepay/register, once grant adoption, fresh full resource admission and one new A2048→4096 request. B still follows actual successful A4096 close. Older pending-approval statements below are historical.

# Current entry: recovery2 failed before model load; settled and both leases closed

Job **1477127.stdct-mgmt-02** is native **F/Exit1**, failed with HeartbeatLost in premodel intake at2048. New updates/queries/model loads are all0. The watcher exited after three Hopper DNS failures; private diagnostics retain exact stderr. [Actual failure, once settlement and independent dual closure](reports/PI0294/design/parallel-science-20261022/RECOVERY2_PREMODEL_HEARTBEAT_FAILURE_CLOSEOUT_ACTUAL_20261023.json) preserves754 GPU seconds/259 CPU seconds and the full70GB output bound. Global updates3843/GPU49117s/PBS18; DEC1 PBS7. All reservations0, unknown effects empty; both leases CLOSED. Original audited A2048 remains the seed. Never restart/replay the closed job or submit B1 before actual A4096 closure.

[Reviewable narrow recovery draft — NOT AUTHORIZED](reports/PI0294/design/parallel-science-20261022/RECOVERY2_ZERO_UPDATE_RESUME_SCOPE_DRAFT_20261023.json) keeps updates9987 and every other compute/science/clock cap, requesting only DEC1 PBS12→13 and DEC1/global storage418/485→483/550GiB. It retains all failed charges, all checkpoints, original three remaining training segments and matrix. The old sealed storage peak is not accepted as proof that the expanded retained history fits. No native registration or submission occurred. [Both-site mux preparation](reports/PI0294/design/parallel-science-20261022/RECOVERY2_BOTH_SITE_CANONICAL_MUX_CPU_AND_READONLY_ACTUAL_20261023.json) passed4 CPU tests and actual readonly authenticated channels; future cold-owner adoption/CI/full-budget and seed validation remain required. Original30/40s calls and300s watchdog are unchanged.

[B2 independent cold owner and4096 terminal preparation](reports/PI0294/design/parallel-science-20261022/B2_ISOLATED_OWNER_FINAL_TERMINAL_CPU_PREPARATION_20261023.json) passed5 isolated CPU tests; this is not native B evidence. B1/B2 entrypoints must be rebound to a future actually closed successful A successor before use. [Existing WBC/independent Teacher distinction](docs/handover/2026-10-23/WDR1_EXISTING_WBC_INTERFACE_AND_TEACHER_SOURCE_SEPARATION_GOAL.md): interfaces and historical native bridges already exist; current source prefix8 and target task qualification remain executor-specific. No new capture, physical evaluation or model run is admitted by these preparations. All entries below are historical.

**Preparation verification:** exact source cbf6b610 passed CI39834415253/job120018814525, all12 steps successful, all8 published files fetched and compared in full. [CI/readback receipt](reports/PI0294/design/parallel-science-20261022/RECOVERY2_CANONICAL_MUX_EXACT_CI_READBACK_20261023.json). The three narrow PBS/storage adjustments have been requested once and await explicit human approval; this is not a new training authorization. [Existing interface evidence and exact task gaps](reports/PI0294/design/WDR1_EXISTING_INTERFACE_EVIDENCE_AND_GAP_20261023.json) and [Teacher executor provenance](reports/PI0294/design/WDR1_TEACHER_EXECUTOR_PROVENANCE_20261023.json) are bounded static inventories with code hashes and historical receipts. They qualify no new physical episode.

# Historical entry: approved recovery adopted; A2048→4096 job queued

**Latest live watcher:** service `vap1-a40-heartbeat-recovery2-watch-resume-02`, observedPID382102, is continuing the same1477127 Q job after the previous watcher exited on a read-only native-program returncode1. [Actual second watch handoff and diagnostic receipts](reports/PI0294/design/parallel-science-20261022/RECOVERY2_NATIVE_READ_STOP_DIAGNOSTIC_RESUME2_ACTUAL_20261023.json) prove prior actor absence, known write state, unchanged request/leases/costs, fresh full-work check and a new confirmed heartbeat. The prior error's stderr was unavailable; its cause is unknown. The wrapper now retains private stderr without changing deadlines, classification or retry policy. Canonical SSH master remains available. No new PBS/model/update or numerical-source change. [B1 isolated owner preparation](reports/PI0294/design/parallel-science-20261022/B1_ISOLATED_COLD_OWNER_CPU_PREPARATION_20261023.json) passed6 isolated CPU tests but has not been deployed; A4096 actual full-state audit, settlement and dual closure still precede B. Earlier watcher snapshots below are historical.

**Earlier watch reconnection:** the first recovery2 watcher exited after three bounded read failures. Job1477127 remains Q, with both leases active and no model release. [Actual watch-only reconnection](reports/PI0294/design/parallel-science-20261022/RECOVERY2_WATCH_READONLY_STOP_RECONNECT_ACTUAL_20261023.json) verifies actor absence, exact request and source, unfiltered two-site queues, known effects and unchanged reservation before continuing the same job under service `vap1-a40-heartbeat-recovery2-watch-resume-01` (observedPID369896). New heartbeat is CONFIRMED; no new PBS, registration or scientific source change. Use this live service for future terminal absence/closure checks. [B1 current-scope V4 metadata](reports/PI0294/design/parallel-science-20261022/B1_RECOVERY2_V4_METADATA_CPU_ACTUAL_20261023.json) passed5 isolated CPU tests; actual closed A4096 and fresh B1 admission remain required.

Human-approved recovery is now natively adopted. New job **1477127.stdct-mgmt-02** is Q, with one active cold owner and fixed runtime **df6427f791fdbe8f175906f240eb3413bf096cbf**. [Actual adoption, registration, reservation and submission](reports/PI0294/design/parallel-science-20261022/HEARTBEAT_RECOVERY2_NATIVE_ADOPTION_ACTUAL_20261023.json) records exact CI39819312716 success, 250 CPU tests and byte readbacks. A registration reply timeout was resolved by complete read-only evidence; no write was replayed. Queue reason: insufficient node_pool resource. Training has not started at this observation.

[Exact human grant](reports/PI0294/design/parallel-science-20261022/A2_HEARTBEAT_RECOVERY_AUTHORIZATION_20261023.json) sets DEC1 updates9987/PBS12/storage418GiB and global485GiB. Historical spent3843 updates and failed2819 lineage remain charged; the new lineage starts at the original full2048 checkpoint. Existing owner handles the queued job; do not restart, duplicate-submit or hot-modify it. A4096 must finish, pass full-state/terminal audit and settle/close both leases before independent B(CORR0). Original clocks, other caps and science freezes remain. Native preview9/rawS1/target WBC qualification is still missing; this is not scientific closure. All snapshots below are historical.

# Historical entry: A2 Failed2819, Settled and Closed; Cold Recovery Draft Pending

A2 job1475719 ended with HeartbeatLost after771 new updates. The latest full
checkpoint is2048; A4096 does not exist. Once-only settlement and both GPU/CPU
lease closures were independently read back. The failed owner has MainPID0.
[Actual closeout](reports/PI0294/design/parallel-science-20261022/A2_HEARTBEAT_FAILURE_CLOSEOUT_ACTUAL_20261022.json),
[timeout forensics](reports/PI0294/design/parallel-science-20261022/A2_OWNER_TIMEOUT_FORENSICS_20261022.json), and
[recovery resource draft — NOT AUTHORIZED](reports/PI0294/design/parallel-science-20261022/A2_FAILED2819_RECOVERY_SCOPE_DRAFT_20261022.json).
The draft preserves all spent costs and files: DEC1 updates9216→9987, newPBS11→12,
DEC1 storage352→418GiB and global419→485GiB. GPU/CPU caps, original clocks,
final20GPUh/40GiB, R0/WBC/S1/recipes remain. This is offline arithmetic, not
native resource admission or authority. Cold timeout/readback engineering is
still pending. Do not restart the settled A2 request or submit B1 early.
[Helper isolation incident](reports/PI0294/design/parallel-science-20261022/PARALLEL_HELPER_ISOLATION_INCIDENT_20261022.json)
is contained; the new connection draft is not qualified. No causal attribution
to A2 is established. Evaluation patch a51c732e passed CI39795202586 but native
preview9/rawS1/target ABI qualification remains missing. Older snapshots below
are historical and do not supersede these receipts or the latest goal card.

# Historical entry: A1 Complete2048; A2 Storage Admission Pending

**A1 job1474599 is now F/Exit0, fully audited and once settled.**
[Actual2048 full-state verification and both lease closures](reports/PI0294/VAP1/20261021T042730Z-localpc-vap1-data/A1_2048_FULL_STATE_TERMINAL_SETTLEMENT_AND_CLOSURE_ACTUAL_20261022.json).
A_CONTROL_SAVE_RECOVERY1 reached2048/cursor65536; complete1024 and2048
states and the original12 TRAIN probes passed independent byte checks.
GPU and CPU-control leases are CLOSED, reservations0 and unknown effects empty.
Original lost1024 costs remain charged; this is not a4096endpoint or task success.
[Updated active goal and parallel science preparation](reports/PI0294/design/parallel-science-20261022/GOAL_SUPPLEMENT_AND_PARALLEL_TASK_CARD.json)
includes actual A2048 continuation metadata, independent B preparation, frozen
data inventory and saved evaluation conditions. A2 is not registered/submitted.
[Current storage admission assessment](reports/PI0294/design/parallel-science-20261022/A2_STORAGE_ADMISSION_PREPARATION_20261022.json)
requires a concrete remaining-work lifecycle before further dispatch. No files
have been deleted; original budgets, deadlines, scientific states and freezes remain.
Queue/running snapshots below are historical.

**Human-approved segmented continuation is adopted; job1474599 is nativelyQ.**
[Actual authorization, exact CI, cold binding and one native submission](reports/PI0294/VAP1/20261021T042730Z-localpc-vap1-data/VAP1_SEGMENT_A1_AUTH_ADOPTION_AND_GPU_SUBMISSION_ACTUAL.json).
Fixed authorization3335c862 changes only DEC1 GPU48->52h, PBS8->11 and
GPU-site output160->176GiB inside the unchanged global budget and clock.
Runtime2b141e9f passed exactCI39718431394/job119626031301(all12steps),
14-file full byte/blob readback and1180local PI0294 tests/1110subtests.
Both cold capsules have836members; old capsules and closed states unchanged.
Job1474599.stdct-mgmt-02 is the once-submitted A32->2048 segment underCF007,
1A40/6h15m, nativeQ at epoch1792641134(node_pool insufficient). Site capacity
36CPU/240GiB is cold-declared; actual process4threads, CPUhard53760 within54000.
No qalter/cancel/resubmit. GlobalPBS16/24, DEC1newPBS5/11, globalGPU27701s,
updates1056 remain spent:old lost1024 plus full-state32, never reset or replayed.
Current reservations22560GPU seconds/2016updates/29queries/54000CPU/70GBoutput.
Complete remaining pair+matrix-ceiling+final20h forecast passes; target runtime
feasibility remains unknown. New state copy/model/CUDA/training have not run at
thisQ snapshot. Same owner service vap1-a40-segment-a1-owner-01/PID630834 is
active, Q60s/R30s checks; no second owner. Once allocated, verify native release,
copy and fully recheck saved32 inside its funded own root, then original1024
fullsave/12TRAINprobes before2048. WBC/R0/recipe/score freezes, original clock,
global96h/24PBS/final20h and selected/deploymentNONE remain. Earlier draft
NOT_AUTHORIZED and oldQ snapshots below are historical, not current permission.

**Recovery job1474352 is F/Exit1, not queued or healthy training.**
[Actual32 updates, full-state audit, typed settlement and both closed leases](reports/PI0294/VAP1/20261021T042730Z-localpc-vap1-data/VAP1_RECOVERY32_FULL_STATE_AND_TERMINAL_ACTUAL.json).
The first32 measured cost gate stopped at9.010780s with headroom versus the
sealed6.1s ceiling. A subsequent optional sharing step incorrectly expected a
full parent manifest in an inference-only consumer. Unlike the old lost1024,
native checkpoint32 is complete:15members/25540965550bytes passed full hashes
and mature optimizer/RNG/sampler/recipe/statistics resume checks; cursor1024.
CUDA reentry has not run. Preserve this state; do not restart its32updates.
The job is settled once:1026GPU seconds/736CPU seconds/32updates/17queries;
global GPU27701s/PBS15/updates1056/queries1088, DEC1 GPU7207s/newPBS4.
Original lostA1024 remains charged in its old lineage; recoverylineage is32.
GPU/CPU/update/query/output reservations are0, both leases CLOSED, owner MainPID0.
Fix84f8c32c passed exactCI39712534631/job119606981111 and1165local tests,
with both changed files byte/blob read back; no hot edit or newqsub occurred.
[Segmented continuation and precise resource-gap draft: NOT_AUTHORIZED](reports/PI0294/design/VAP1_A40_STATE32_SEGMENTED_RECOVERY_DRAFT_20261022.json).
The prior two8h job forecast no longer fits measured costs. The draft does
not increase any cap or admit a job. Original clock/freezes/final20h remain,
selected/deploymentNONE. Queue/running snapshots below are historical.

**Bounded lost-save recovery is natively adopted; a new A40 job is actually Q.**
[Actual authorization adoption, cold binding, complete forecast and native submission](reports/PI0294/VAP1/20261021T042730Z-localpc-vap1-data/VAP1_LOST_SAVE_RECOVERY_ADOPTED_AND_A40_SUBMITTED_ACTUAL.json).
Human authorization1d84e4cf changes only DEC1 cumulative updates8192->9216.
Old A1024spent updates, costs, closed failure and original training_runs entry
are preserved. New lineage A_CONTROL_SAVE_RECOVERY1 starts0 from the same CORR
parent, not an invented1024resume; A/B remain4096each on original R0/recipe.
Job1474352.stdct-mgmt-02 was submitted once underCFP04-CF-007 and read backQ
at epoch1792635039 (node_pool insufficient). Request1A40/4CPU/32GiB/8h was
rewritten to the cold-declared36CPU/240GiB capacity; no qalter/cancel/resubmit.
Actual process affinity4, CPU hard79760/accounting80000. Runtime78427dd1 passed
exact CI39708750610/job119594810704,11-file byte/blob readback and1163local tests.
Both native ledgers adopted the grant once with no cost reset; full source-pair,
matrix-ceiling/final20h forecast and fresh native auth/quota/queue checks passed.
Same owner service vap1-a40-lost-save-recovery-owner-01/PID563461 is active and
polls the same queued handle every60s. No new model/update/physics at snapshot.
Original clock/budgets/final20h and all freezes remain; selected/deploymentNONE.
The unapproved-recovery and failed-training snapshots below are historical.

**A_CONTROL reached1024 committed updates, then failed before checkpoint write.**
[Actual failure, independent absence-of-state audit, typed settlement and closed lease](reports/PI0294/VAP1/20261021T042730Z-localpc-vap1-data/VAP1_A40_1024_SAVE_FAILURE_AND_SETTLEMENT_ACTUAL.json).
Job1474116.stdct-mgmt-02 isF/Exit1; owner03 is inactive. The new storage guard
incorrectly rejected the approved Cosmos input alias. Native and fallback saves
both failed; no complete checkpoint exists and no1024-state resume is possible.
Actual1024updates/17queries/5861GPU seconds/5969CPU seconds are now settled;
all its reservations are released, leaseCLOSED, unknown effects empty. Global
updates1024/PBS14/GPU26675s; DEC1 updates1024/newPBS3/GPU6181s. No rollback,
deletion or resubmission occurred. The cold engineering fix93899748 passed153
local DEC1 tests and1155 PI0294 tests; exact CI39704727441/job119581956584
is completed/success (all12steps), with four changed files byte/blob read back.
The same published guard subsequently passed a read-only check on the actual
Vanda filesystem (epoch1792632268); old receipts/capsule were unchanged. This
is not a full native checkpoint save or a recovered1024-state claim. A stricter
enforced CPU-time profile is cold-prepared to fit the existing96core-hour cap;
no job or active ledger has adopted it. Sourceb2663d88 passed exact complete
CI39706081300/job119586316785, nine-file byte/blob readback and1159 local tests.
The80000CPU-second envelope includes a79760process hard limit and240seconds
for wrapper/observer/accounting; four threads, recipe and site-billing gates
are unchanged. It is a stricter enforceable bound, not added CPU permission.
A new matched4096+4096 pair would require9216 cumulative DEC1 updates, exceeding
the unchanged8192subcap. Replacement lineage/subcap authorization is requested,
not assumed. Fresh complete resource forecast and site gates remain required.
Originalclock/final20h and WBC/S1/R0/recipe remain frozen; selected/deploymentNONE.
The running-training snapshot below is historical, not current job state.

**A_CONTROL is actually training on its own A40 allocation; first32 passed.**
[Actual data closure, allocation, model qualification and optimizer evidence](reports/PI0294/VAP1/20261021T042730Z-localpc-vap1-data/VAP1_A40_FIRST32_AND_REAL_TRAINING_ACTUAL.json).
Job1474116.stdct-mgmt-02 isR onGN-A40-034 underCFP04-CF-007. Runtimeb2a82480
passed exactCI39652926789/job119405661124. Both fixed CPU readers are CLOSED
and settled; original126/C1 and139bank identities/R0 are unchanged, noR1 mix.
Seventeen saved-TRAIN draws passed;32real A_CONTROL updates met the presealed
A40 cost gate and training continued (56committed at epoch1792613090).
This is not a4096endpoint, A/B result, task success or target qualification.
The running job retains its sealed source; newer report commits are not hot
updates. Originalclock/budgets/final20h and selected/deploymentNONE remain.
Older queue/CPU snapshots below are historical, not the current GPU state.

**VAP1 first2 parents are actually R1 admitted; saved-TRAIN Teacher T0 is closed.**
[Raw successes, both codecs/readers, videos, Teacher replay and claim boundaries](reports/PI0294/VAP1/20261021T042730Z-localpc-vap1-data/VAP1_FIRST_PAIR_AND_TEACHER_T0_ACTUAL.json).
Both T1/T2 completed32s, raw auditsPASS, R0FAIL->R1PASS; all601windows and real
dual-camera/mixed-parent readers passed, no safety/unknown effects. Not D24 READY.
T0 reproduced15384saved observations/1920H40 blocks and all original events
across24CR2 Teacher parents without physics; no student-state recovery claim.
Two next single-core first-batch captures are live; current A40 remainsQ.

**VAP1 is now natively adopted; the first new TRAIN pair is collecting.**
[Actual adoption, nonadditive budget migration, owner recovery and frozen48-card population](reports/PI0294/VAP1/20261021T042730Z-localpc-vap1-data/VAP1_ADOPTION_AND_DELTA_BUDGET.json).
The same A40 job remainsQ/4CPU/32GB, with no model effects or resubmission.
Owner03 resumed only after owner02's DNS-only exit was proven terminal; the
original neural capsule is unchanged. Data static24groups/48cards passed with
48constructors and zero integrations; its lease is closed. First-pair capture
is live on one isolated Core CPU, not a completed bank. Teacher saved-TRAIN
state availability is only a precheck, not recovery or DAgger qualification.

**DEC1 separately approved corrected qalter succeeded on the same queued job.**
[Actual correction and independent native readback](reports/PI0294/DEC1/20261018T182956Z-localpc-pv1/DEC1_SAME_JOB_RESOURCE_CORRECTION02_ACTUAL.json).
At epoch1792550964 the one additional explicitly approved call used the exact
`select=1:ngpus=1:ncpus=4:mem=32gb` argument and returned0. Immediate and separate
epoch1792550985 native reads show1471903.stdct-mgmt-02 stillQ/1GPU/4CPU/32GB/30min,
with all nonresource job/input identities unchanged and no model/effect records.
The old rc187 syntax failure below is preserved; total actual qalter calls2,
each separately approved. No cancellation, resubmission, new owner, hot capsule
change or extra budget. Native am_job_cache still mentions36CPU; actual billing
and allocation-time resource stability remain unverified, not overwritten here.
Same owner service02/PID3934448 is active and continues low-frequency watching.
This removes the observed queued select mismatch only, not model qualification,
training admission or scientific completion. All original reserves/frozen items
remain; selected/deployment=NONE.

**DEC1 one approved same-job resource restore was attempted, not successful.**
[Actual one-call error, unchanged native resources and control-reader recovery](reports/PI0294/DEC1/20261018T182956Z-localpc-pv1/DEC1_SAME_JOB_RESOURCE_RESTORE_ATTEMPT_ACTUAL.json).
Job1471903.stdct-mgmt-02 remainsQ/36CPU/240GB with no neural process at the
independent readback. The singleqalter returned187 because our control command
omitted `select=`; this is not evidence of a site policy rejection. No second
qalter, cancellation or new submission occurred. A corrected call is only
proposed, awaiting a separate explicit reply. Do not erase the failed attempt.
The earlier user-filtered queue parser also failed before any qalter; its
zero-effect witness is preserved. Reader597a97594278932d9b7ac825fd8131ab138144ec
reuses existing qselect/individual-json-qstat, passed exact CI39531282509
/job118997503156, full GitHub readback and both native site views. Only the
read-only control watcher was handed over, oldPID3776138 terminal before new
PID3934448/service dec1-vanda-compat-owner-02; same logical owner/session,
original neural capsulea2caf6a5/inputs/leases/T0/budget and PBS job unchanged.
No model readiness, training, physics, scientific result or DEC1 completion.

**DEC1 current CORR cold checks passed; one A40 job is queued with a resource mismatch.**
[Actual cold binding, native request, owner and same-budget reservations](reports/PI0294/DEC1/20261018T182956Z-localpc-pv1/DEC1_A40_COMPAT_COLD_AND_SUBMISSION_ACTUAL.json).
Runtime a2caf6a54f37cb76f64f8d7f42abd5f0e3252339 passed exact Actions39501594922
/job118900166184 and full changed-file GitHub readback. The registered candidate
actually hashed all current CORR/Cosmos inputs and decoded its original saved
dual-camera TRAIN input without loading a model. Only one job1471903.stdct-mgmt-02
was submitted, inside the same original budget:1860 GPU seconds and17 queries
reserved, no training or physics. Its native Q record rewrote4CPU/32GB to
36CPU/240GB; cause is unconfirmed. The fixed native launcher requires4CPU,
so this is NOT a healthy model-ready queue or numerical qualification PASS.
The same sole owner watches; it cannot release a mismatched allocation and
does not qalter/cancel/resubmit automatically. One explicit same-job original
select restoration question is pending. Local prepaid CPU is near its original
8h bound. No new learning result, target WBC qualification or CR2 expansion.
[Subsequent native historical comparison](reports/PI0294/DEC1/20261018T182956Z-localpc-pv1/DEC1_A40_HISTORICAL_ALLOCATION_RESOURCE_COMPARISON.json)
shows the old successful A40 probe also had36CPU/240GB. The additional strict
native4CPU gate is new; mature affinity4 is not allocated-ncpus4. This is an
admission-contract conflict, not a model failure or proof the site is broken.
Whether normal same-job resource restoration is allowed/effective is untested.

**DEC1 fixed CORR assets are now actually delivered to Vanda, not model-qualified.**
[Full native handoff, preserved failure, costs and closed leases](reports/PI0294/DEC1/20261018T182956Z-localpc-pv1/DEC1_CORR_VANDA_ASSET_HANDOFF_ACTUAL.json).
HF READY68afb6503982613ad47fe60d8a1ae66a580dec60 follows actual independent
Vanda full-byte readback of the CORR@2048 delta and its reused frozen shard.
It is not old M2048 or E1 M2-CORR, nor a full-resume export. Only two private
Cosmos locators are relocated; weights/index/R0 are unchanged. All four CPU
children are CLOSED, including the preserved pre-client missing-locator failure.
Its replacement uses explicit anonymous fixed-public reads, no credential copy
or repeated weight upload. Actual runtime e0792964/869f3100 passed exact CI.
That completed asset-only stage added no PBS/model/query/update/physics or policy improvement. The
same original budget includes 15313692088 logical transfer upper bytes and
8329724022 new GPU-site output upper bytes across both sites; future storage
forecasts must include them. Cross-platform accounting is now implemented above;
A40 numerical/precision remains pending, as do source native interface qualification
and the frozen target task ABI. Do not run the expired old standalone probe.

**DEC1 source cold inputs now use actual saved bytes, but are not native-bound.**
[Actual identity/timing/storage evidence](reports/PI0294/DEC1/20261018T182956Z-localpc-pv1/DEC1_SOURCE_COLD_INPUTS_AND_STORAGE_IMPLEMENTATION.json) ·
[Narrow storage implementation and remaining gates](docs/protocols/DEC1_SOURCE_TRAINING_COLD_INPUTS_AND_STORAGE_V1.md).
Both unchanged source arms have CORR/R0/139/C1/population/recipe/probe inputs;
no full-B0 PASS was invented. A_CONTROL still has zero new-bank optimization
visits. Native source registration and a complete fresh budget/lifecycle seal
are pending. The opt-in Trainer wrapper shares only an exact fully frozen
shard after a full save and preserves original parent bytes/mode. It does not
retire a checkpoint or constitute a native storage/backup receipt. Actual
parent logs show maximum5.64634s/update; the new data has not been trained.
Hopper's own queue was empty and all registered campaign children closed at
the cited fresh read. Hook cause remains unconfirmed; no blind same-hook job
was submitted. Existing H200 guards, final reserve and original clock remain.

**Latest CR2 is fully CLOSED and its fixed-R1 bank is delivered, not collecting.**
[All24 real attempts and final HF/readback/resources](reports/PI0294/CR2/20261020T160340Z-localpc-cr2/CR2_FINAL_REPORT.md).
T1 12/12 and T2 12/12 Teacher/data success, 12paired new TRAIN geometries,
14R0PASS->R1PASS/10R0FAIL->R1PASS/0regressions; all children/control CLOSED.
HF READY9435ba48fa98336b6e88431ed145382724b16b44 was published only after
full new-payload/member readback. No GR00T learning or target transfer is
claimed. R1 is not hot-injected into DEC1/PV1's original R0/139bank.

**DEC1 CPU reference implementation now exists; native qualification is pending.**
[Paired interface rules and explicit boundary](docs/protocols/DEC1_EXECUTION_PROGRESS_CPU_REFERENCE_V1.md) ·
[CPU scalars fixed, NOT a native-qualified freeze](reports/PI0294/DEC1/20261018T182956Z-localpc-pv1/DEC1_SOURCE_ONLY_INTERFACE_FREEZE.json).
Both I_TIME/I_PROGRESS use frame-corrected V2, shared original-pose stationary
duration semantics, complete raw H40 before mapped full/prefix checks, and
post-action actual feedback/ACK. CPU fixtures are not native S1 qualification
or 64-cell task results. Target ABI, source native binding/commissioning and
evaluation-panel seal remain absent; no WBC/plant/S1/scoring changes.
Fresh source-training budget bridge a9eac69e passed exact dependency-complete
CI39474334901/job118806274229; see its implementation report. Native budget
inputs, cold source-training binding and legal GPU qualification still remain.
The older snapshots below are preserved as historical observations, not the
current CR2 status or a reason to wait for already successful exact CI.

**DEC1 is natively adopted inside the original PV1 window, not a second budget.**
[Adoption and actual nonadditive limits](reports/PI0294/DEC1/20261018T182956Z-localpc-pv1/DEC1_NATIVE_ADOPTION_AND_BUDGET.json) ·
[Actual source identity and missing target delivery](reports/PI0294/DEC1/20261018T182956Z-localpc-pv1/DEC1_EXECUTOR_POLICY_INTERFACE_MANIFESTS.json).
**Latest native job state: child12 F and both leases CLOSED, not still H.**
[Exact owned-held disposition and original pinned settlement](reports/PI0294/DEC1/20261018T182956Z-localpc-pv1/DEC1_HELD_JOB_DISPOSITION_AND_SETTLEMENT.json).
Official installed PBS permits owner deletion. The exact H/noexec/noSTART job
was first user-held to exclude recovery, then deleted once; no system-hold
release, healthy-R cancellation or hook bypass. Original obit interval20s plus
one-second uncertainty upper gives21charged H200 seconds, total20494,
reserved0, PBS11/24. Both child12 leases and Core43200-second reservation
are actually closed; no new model/query/task/physics/update. Hook cause is
stillUNKNOWN: no same-hook blind retry. [Current budget reconciliation](reports/PI0294/DEC1/20261018T182956Z-localpc-pv1/DEC1_CURRENT_RESOURCE_RECONCILIATION.json)
also corrects initially omitted old child episode/tick/constructor reservations;
the effective DEC1 caps were unchanged and no scientific dispatch used that
projection. Original adoption bytes and all history remain preserved.
Both sites read back identical DEC1 adoptionSHA21e5a6fafafd73873c5ba0c8e06c2245be37e1d8e0068b69d09f10caa8450433,
through the existing parent control roles/leases. Original T0/cutoffs/final20h
and old costs/reservations remain; no second owner or new science from adoption.
DEC1_SOURCE_TRAIN_STAGE_V1 explicitly admits only unchanged A/B source stages
before full B0. Legacy stages still require complete B0; native scope/recipe/
resume checks remain. The published fresh DEC1 future-budget bridge passed
exact CI; its native budget inputs, cold binding and allocation qualification
are still required before training. CPU fixtures are not admission.
The confirmed frame-transform omission is corrected in explicit
GR00TToWBCAdapterV2 for BOTH new DEC1 interfaces. Historical V1 bytes and
Stage35's frozen-file guard remain unchanged; no wrong V1 baseline is allowed
in DEC1. Source7328828e CI39466956612 failed the old V1-byte guard, so it is
not a deployable success. The V2 correction retains2red-to-green regressions
and adds same-frame equivalence; no WBC or running capsule was patched.
I_PROGRESS native source commissioning and the64-cell matrix are not run yet.
Source CORR/R0/139bank manifests and15numerical/26asset hashes read back.
Core/Vanda registries contain no qualified DEC1 task executor delivery; old
R4/TF-R3 leases are expired despite ACTIVE text, no recorded worker is live,
and TF-R3 native safety STOP hash22d554d5 remains. Do not reopen those scopes.
Target absence does not block the newly authorized source training direction.

**Historical CR2 snapshot: first four closed, remaining twenty then collecting.**
[Actual paired codec, readers, closed leases and resource snapshot](reports/PI0294/CR2/20261020T160340Z-localpc-cr2/CR2_FIRST_FOUR_ACTUAL_PAIRED_AND_REMAINDER.json).
The first4 original attempts (2geometry groups, T1/T2 each) all completed
real32s Teacher tasks and fixedR1 admission. Every episode has601actual H40
windows checked under both frozen ranges, six actual dual-camera reader
anchors per variant, and actual cross-parent batches. Transitions:2PASS->PASS,
2FAIL->PASS,0PASS->FAIL; no range refit/native fault/unknown effect.
Capture/export leases are CLOSED; their costs are settled once. The remaining
20first attempts are running on2single-core workers under the original CR2
clock/limits, not twenty completed results. Bank/HF READY remainsfalse;
Teacher/data evidence is not GR00T learning or target-WBC migration evidence.

**DEC1 is the latest scientific authority; native implementation is pending.**
[Fixed a7a7ecd9 protocol](docs/protocols/DEC1_CONTROLLER_DISJOINT_VLA_SCIENCE_GO_20261021.md)
and [machine limits](reports/PI0294/design/DEC1_CONTROLLER_DISJOINT_SCIENCE_GO_20261021.json)
explicitly allow unchanged source A/B training before B0 completes, through
a new versioned admission only. Old authority replay and baseline-relative
selection/LOCKBOX still require full B0. First DEC1 matrix remains fixed
CORR@2048/R0; CR2 is independent, not a prerequisite or hot data injection.
This paragraph is not a native adoption, GPU allocation or target delivery
receipt. Older B0-before-training statements below describe their old scopes.

**Historical pre-disposition observation: child12 was system-held before launcher.**
[Native hook rejection, owner DNS interruption and CR2 static/data effects](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_CHILD12_SITE_HOLD_AND_CR2_STATIC_ACTUAL.json).
679029 is H/Hold_Types=s,exit-16,run_count1. tracejob says an execution hook
rejected the request; noPBS_READY/START/model/queries/episodes appeared.
The local owner separately exited on transient SSH hostname resolution.
No qrls/qdel/qsub/release retry; the installed official qrls manual requires
manager/admin privilege for a system hold. Hook identity/cause remainUNKNOWN.
Keep28860H200seconds reserved; the20-21s prelaunch allocation is not zero cost
or a terminal settlement. B014/48,updates0,selected/deploymentNONE unchanged.
CR2 has native newT0=21Oct00:03:40SGT,stopdispatch22Oct16:03:40,hardclose
23Oct00:03:40. D0 all12groups pass with32constructors/0integrations,
45.184645CPU seconds; staticleaseCLOSED. Its first-four single worker is
actually collecting, not yet qualified data. Fresh same-job H/noexec/noREADY
and zero-effect evidence establish no current GPU allocation; any R/unknown
state blocks new CR2 dispatch. Original PV1 and closed COV1/CR1 stay untouched.

**CR2 now has explicit user GO; do not reopen DATA-COV1 or change PV1.**
The user approved the full fixed5d139315 CR2 package in reply
call_656187cffb8b43519e678e7561b78575. [Exact new scope](reports/PI0294/design/CR2_FIXED_R1_BANK_PILOT_SCOPE_20261020.json):
12new TRAIN groups/24first attempts, fixedR1, original4mm Teacher, paired
saved-raw R0/R1,32Core/4localCPUhours,48h,2single-core workers maximum.
Native T0/lease and exact sourceCI are still required before dispatch.
This authorization does not claim new data, a READY bank, training or policy
improvement. PV1 has priority; its R0/139bank/B0-before-training stay frozen.

**Native authentication recovered; child12 is actually submitted once, not running science yet.**
[Fresh project, cold binding and actual native continuation](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/NEXT_NATIVE_ALLOCATION_AND_B0_PROGRESS.json).
Fresh same-context CFP04-CF-007 active/member/cluster/positive-balance PASS.
All old child leases CLOSED, own queue empty and unknown[] before binding.
Exact runtime30e2e152 passed Actions39436335491/job118674212836; both cold
capsules757members match that commit. Original139bank/R0/parent unchanged.
679029.hopper-m-02 is Q at23:39SGT20Oct:1H200/12CPU/225GB/8h, PBS11/24,
spent20473H200seconds/reserved28860. Sole finite owner service is active,
Q-auth health enabled; credentials stay memory-only, no identical login loop.
B0 remains14/48(9success/5valid fail),34remaining,queries1054,updates0.
No new model/physics or training is claimed. Original175s release and own
allocation qualification still required. B0 must finish before A/B; final20h
and original deadline remain. CR1/COV1 CLOSED,CR2 NOT_AUTHORIZED,selectedNONE.
Authentication-blocked paragraphs below are preserved historical snapshots.

**Official-guide/session round3: no client-method mismatch found; no password retry or PBS.**
[Official-source access scope, four native session checks and CPU-only request proof](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/AUTH_OFFICIAL_GUIDE_AND_SESSION_AUDIT_ROUND3.md).
Fresh/reused SSH and clean/inherited Python environments all return401/15006
through the administrator-configured gateway. A network-denied official-CLI
fixture verifies Basic current-user/dummy-password login, not cached Bearer
reuse; the real nested SSH/PTTY dummy-input fixture also passes. These are
not real authentication success. Vendor default10-second server auth timeout
is consistent with the earlier10.04-second login failure, not a proven root
cause or NUS server setting. Full Hopper presentation is behind NUS SSO and
was not read; public NUS guides and the matching vendor manual were read.
No global auth/config/cache edits, new password attempt, GPU/PBS or science.
Native auth must pass before any continuation; original PV1 gates stay fixed.

**Native auth round2: official login API returns HTTP401; no new PBS.**
[Actual transport/cache/logout evidence and bounded cost](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/AUTH_NATIVE_SERVER_DIAGNOSTIC_ROUND2.json).
Same-context normal login reached the official service and received401 after
10.04seconds; subsequent project query remains15006. The CLI-opened auth cache
is owned/readable/writable,0600,parent writable. Supported logout also exit1;
no manual token edit, credential collection, MFA bypass or unchanged login loop.
The CLI's timeout prose alone is not evidence of a lost SSH connection, bad
password or mandatory human challenge. Diagnostic error classification is
corrected with27 focused tests; real auth renewal remains unproven. E0 stays
CLOSED, H200spent20473/reserved0, PBS10/24, B014/48,updates0,selectedNONE.
Neither another PBS authorization nor CR2 approval fixes this native auth gate.

**POST-CR1 E0 is actually CLOSED; E1 CPU repair is tested, native auth still blocked.**
[Child11 actual settlement and both-site closure](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_CHILD11_SETTLED_AND_CLOSED.json) ·
[Authentication lifecycle repair and native readback](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/AUTH_LIFECYCLE_REPAIR_AND_NATIVE_READBACK.json).
678230 is F/exit1, settled once through its original pinned owner path:
198 conservative H200 seconds, campaign total20473, GPU reservation0,
PBS10/24. Both child11 leases CLOSED; Core43200-second reservation released.
B0 remains14/48(9success/5valid task failures),queries1054,updates0,unknown[].
The same-context normal masked login did not restore project admission:
fresh query still15006; a connectivity/timeout hint is not proof of a bad
password or a human-only challenge. No password/token/raw CLI output is stored.
26 focused and967 PI0294 CPU tests pass (836 subtests). Q-health/fresh-R
authentication, original175-second deadline, sealed exclusive publication,
partial-proof recovery and exact unknown-reply readback are CPU-tested.
The mature private owner adapter passes CPU injection; no new native owner
loop, child binding or PBS has started. Exact push CI and real native auth
remain prerequisites; do not claim engineering recovery or policy improvement.
PV1 original R0/139-bank/B0-before-training/budget/deadline and all frozen
numerical modules stay unchanged. CR1/COV1 stay CLOSED; CR2 awaits science GO.
The earlier child11 pending-settlement and queued paragraphs below are history.

**CR1 R0-R4 is actually complete and CLOSED: RANGE_SUPPORTED, offline only.**
[Full review and next scientific decision](reports/PI0294/CR1/20261020T105832Z-localpc-cr1/CR1_FINAL_REVIEW.md) ·
[Paired results and actual official readers](reports/PI0294/CR1/20261020T105832Z-localpc-cr1/CR1_PAIRED_RESULTS_AND_READER.json) ·
[Independent final resources](reports/PI0294/CR1/20261020T105832Z-localpc-cr1/CR1_RESOURCE_FINAL.json).
The explicit e7593fdd GO was adopted once at18:58:32SGT on20Oct. Runtime
8e2097ed passed exact Actions39412546118/job118593595499 and a755-member
immutable CPU capsule.194saved TRAIN parents:141FIT/50DIAGNOSTIC/3known
challenge; all601windows and six real dual-camera reader anchors per variant
were checked. Seven diagnostic FAILs across six disjoint lineages recover;
all139retention parents stayPASS. One sealed PAD5 relative-EEF9 range only;
no feedback refit, no state/gripper/clamp/gate change. Independent JSON QA
and native seven-file/lease readback PASS. Core4328.4369CPU seconds including
600control upper; local2400seconds upper; CR1leaseCLOSED/reservations0,
unknown=[]; model/physics/GPU/PBS/updates0. Frozen15numerical files,
26native assets, oldDATA13closed leases,139bank and originalstats unchanged.
This is not a new policy, bankREADY, BF16 qualification or rollout success.
The candidate must not be installed on oldCORR as the original B0 decoder.
Old DATA-COV1 remains2/4FAIL andCLOSED; a new small bank/taskpilot needs GO.

**PV1 child11 is now F, NOT the queued/healthy snapshots below.**
[Exact model-before-auth engineering terminal, pending settlement](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_B0_11_FRESH_AUTH_PREMODEL_TERMINAL_PENDING_SETTLEMENT.json).
678230 allocated19:37:53SGT, then failed exit1 after fresh project auth15006
prevented owner release; wrapper models_loaded_by_launcher=0. Original owner
service exited. B0 remains14/48(9success/5valid task failures),updates0,
queries1054,unknown=[]; this is not another valid task failure. Both child
leases remainACTIVE and28860H200/43200CoreCPU seconds are still reserved,
pending original-PV1 settlement; do not claim zero GPU charge or released
resources. CR1 did not retry, re-release, alter or settle this job. Preserve
B0-complete-before-training and all PV1 freezes/budget/deadline. The following
paragraphs are historical receipts, not current queue or owner state.

**DATA required saved-parent audit now PASS; collection stays CLOSED/failed gate.**
[Final dataset card and actual cross-episode consumption](reports/PI0294/DATA_COV1/DATA_COV1_DATASET_CARD.md) ·
[Resource/render/storage addendum](reports/PI0294/DATA_COV1/DATA_COV1_RESOURCE_FINAL.json).
Exact audit source16a4c2d6 passed Actions39400354340/job118553219899;746-member
cold capsule, originalCPUentry, separately registered finite saved-data lease.
Both distinct qualified episodes/all2566numeric+RGBmembers were read;12real
anchor samples collate/repeat PASS, no model/physics. Full event times for3raw
parents are saved. Audit4.6209CPUsec; its lease alsoCLOSED. Original collection
STOP/terminal/ledger/11child leases remain immutable andCLOSED, T0 unchanged.
CumulativeCore5242.0977CPUsec/local3000upper, reservations0/unknown0. Stage gate
still2/4<3; no remaining44, noP72/P144, no bank/HFREADY or PV1 data injection.
Native render timings/current inode-aware bytes are reported; historicalI/O
peak is explicitly unmeasured, not invented. Follow-up range draft is
NOT_AUTHORIZED. Continue the independent PV1 B0 job, not this closed scope.

**17:09 SGT: DATA first-four is terminal2/4 admitted; independent window CLOSED.**
[Dataset qualification card and exact native evidence](reports/PI0294/DATA_COV1/DATA_COV1_P24_DATA_CARD.md).
Three known full task successes, one unknown engineering result; two parents
pass raw/all601codec/officialCPUchain, one succeeds but fails frozen codec.
First-four requires3admissions, so gateFAIL. All11child leases and parent
control leaseCLOSED; reservations0/unknown0. All44remaining P24 rows are
NOT_RUN; noP72/P144, no stage bank/HF READY or PV1 data injection.
OriginalT0/counters/failures remain. Do not restart this closed scope or count
per-parent data success as GR00T/Go2 success. PV1 child11/678230 remainsQ at
the latest17:10read, B014/48,updates0, one existing owner and225GB request.
The queue paragraph below remains its binding; earlier DATA-running snapshots
are historical. Continue PV1 B0 before A/B, with all freezes unchanged.

**DATA first revised T2 succeeds physically but fails the unchanged codec gate.**
[Actual task/raw/codec distinction and first-four support limit](reports/PI0294/DATA_COV1/DATA_COV1_FIRST_REVISED_T2_TASK_SUCCESS_CODEC_REJECT.json).
32s/16000live/641RGBpairs/raw audit PASS,41/41hold,0native faults. All601H40
windows checked: first SO3 error0.00013817rad exceeds frozen0.0001rad, at
anchor209/action31 with channels4/6 inverse-clipped. Not admitted; official
chain not run after codec reject. No stats/decoder/threshold change or retry.
Together with the preserved first engineering failure, even two later passes
cannot reach3/4 admissions. Complete only the remaining fixed first-four
diagnostics; do not dispatch44 more, expand stages or claim a READY bank.
PV1 child11 remains the independent queued B0 continuation described below.

**16:40 SGT: child10 effects and leases are CLOSED; child11 is queued once.**
[Actual closure and native225GB continuation](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_B0_10_EFFECT_CLOSED_AND_CHILD11_QUEUED.json).
678230 isQ, oneH200/12CPU/225GB/8h, globalPBS10; sole existing owner attached.
Exact runtime d7dc38ef passed Actions39393487629/job118530396143. The37/36
engineering tail is independently accounted as generated but uninstalled,
with0live effect and no unknowns. Child10 STOP/raw remain preserved; its
two leases and reservations are closed. B0 remains14/48 (9success/5valid fail),
updates0. No new model/query/physics is claimed for queued child11.
[Revised DATA static qualification and bounded first-four continuation](reports/PI0294/DATA_COV1/DATA_COV1_NEAR_V2_STATIC_AND_FIRST_FOUR_CONTINUATION.json):
24groups static PASS,24new constructors,33.7242CPUsec,0integration. Original
first failure stays in48; second physical task is active, not yet admitted.
No bulk dispatch/stage bank/HF READY. Original clocks and PV1 priority stay
fixed. All following snapshots are historical, not the current native state.

**Child10 is terminal after two valid results and a separate ENFILE interruption.**
[Exact engineering terminal, resource settlement and pending effect closure](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_B0_10_ENFILE_TERMINAL_PENDING_EFFECT_CLOSURE.json).
678041 isF/exit1; owner settled3074H200s and exited. Total20275H200s,
reservation0, PBS9,queries1054,updates0. B0 stays14/48:9success/5valid failures.
F2/000/T1's14.4s interruption is ENGINEERING_INCOMPLETE after a login-side
file-table exhaustion error. Preserve37generated draws/36executed prefixes;
close the generated-but-uninstalled tail explicitly before any replacement.
No live native safety fault is recorded. DATA new dispatch waits for exact
engineering-effect closure/priority reconciliation, with originalT0 unchanged.
Do not treat the R/healthy-owner paragraphs below as current native state.

**B0 now14/48:9successes and5valid task failures;34conditions remain.**
[Second child10 condition and independent causal evidence](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_B0_10_SECOND_AUDITED_CONDITION.json).
F1/004/T2/N294101 stopped at29.6s under originalS1 predicted-contact rejection,
75draws/74executedACKs, rejected-prefix live ticks0, native safety flags0.
Raw audit and both-site causal chain PASS. It is a valid task failure, not an
engineering retry or initial-approach diagnosis. Do not repeat this condition.
Native child10 remains on its pinned source; no new training has begun.
[Stage-bank packaging contract implementation](reports/PI0294/DATA_COV1/STAGE_BANK_IMPLEMENTATION.json)
is tested source only, not actual bank/HF READY; DATA new physics still waits
for PV1 priority and original first-four/stage gates. Historical counts follow.

**16:07 SGT: child10 model is ready; B0 has13/48 trustworthy conditions.**
[First new audited DEV success and exact progress](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_B0_10_FIRST_AUDITED_SUCCESS_AND_PROGRESS.json).
678041 remainsR. F1/004/T1/N294101's preauthorized REPLACEMENT-01 completed
32s autonomously, raw/physics audit PASS,80 fresh camera/action/ACK joins PASS
on both sites,41/41 hold,0S1 rejection/0native fault. B0 totals9success/4valid
task failures;35 remain. No A/B optimizer updates, selection or deployment.
DATA first-four continuation still waits for PV1-R priority; originalT0 and
all frozen components stay fixed. The model-not-ready/Q paragraphs below are
historical snapshots, not current runtime state.

**15:37 SGT actual update: child10 is R, not Q; future-data physics is waiting.**
[Native amendment, first-attempt witness, resource and allocation receipt](reports/PI0294/DATA_COV1/DATA_COV1_NEAR_V2_NATIVE_AND_FIRST_ATTEMPT_RECEIPT.json).
678041 started onhopper-19 at15:30 with225GB; original premodel qualification
is in progress, model readiness is not yet claimed. AmendedDATA runtime34079e41
passed exactActions39387023857/job118509187552 and is natively appended with
originalT0 unchanged. Saved-byte witness closed with641 camera pairs/16000
clock rows checked, no invented score or admission. StaticV2/first-four remain
pending; finite one-worker continuation waits for PV1 priority, with no bulk
dispatch, retries, replacements or new clock. Historical snapshots follow.

**PV1 child10 is submitted; DATA-COV1 NEAR revision is explicitly approved.**
[Actual submission and narrow geometry amendment](docs/handover/2026-10-20/DATA_COV1_NEAR_AMENDMENT_AND_PV1_CHILD10.md).
678041/globalPBS9 uses exact-CI runtime79161e32,225GB,oneH200; last observedQ,
sole owner attached, no new model/query/physics/update. DATA's first attempt
is an unscorable engineering failure, not a task result; its raw and32k ticks
remain counted. The user approved unexecuted NEAR0.15-0.18m only, with no
replacement/new budget/T0. New static qualification and first-four codec gate
are pending. Do not reuse the following historical queued/adoption snapshots.

**DATA-COV1 is natively adopted; PV1 child09 is closed before model loading.**
[Independent data window and actual static qualification](reports/PI0294/DATA_COV1/DATA_COV1_ADOPTION_AND_SCOPE.json) ·
[Prefrozen crossed-factor population and24 static-qualified TRAIN groups](reports/PI0294/DATA_COV1/DATA_COV1_FROZEN_POPULATION.json) ·
[PV1 model-alias failure and two-site closure](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_B0_09_MODEL_ALIAS_FAILURE_AND_CLOSURE.json).
Explicit FULL_CONDITIONAL_CAMPAIGN approval is recorded at#11/6284943603.
DATA-COV1 T0 is2026-10-20 15:04:42 SGT; runtime03860679 passed exact
Actions39383323045/job118497224191. P24 static24 passed with54 constructors,
74.9464 CPU seconds and0 integrations; renderer is actual Mesa llvmpipe.
The original four-task raw/codec barrier is still pending; no full data-stage
success or new READY bank is claimed. PV1's139 bank remains unchanged.
677540 passed its600s/80cycle qualification and fully hashed178341 members,
then failed B0_CORR-LOAD because a required Cosmos symlink was dereferenced
by the cold binder. No model/query/episode/update ran. Both leases are CLOSED;
1105 H200 seconds settled, campaign17201 seconds/PBS8, B0 still12/48.
The frozen loader is unchanged; the cold repair creates only the exact existing
snapshot alias and checks it before prepare and submit. No new PBS is claimed
by this entry. The queued paragraphs below are historical snapshots.

**Future-data D0 is closed; PV1's frozen bank and critical path are unchanged.**
[Actual 43-row CPU qualification and prospective coverage](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/FUTURE_DATA_D0_ACTUAL_SUMMARY.md):
32 PASS/11 codec exclusions; one additional raw-audit FAIL was excluded before
reading. The 29 qualified original-reference rows could expand future coverage
77 to93 groups, but are not injected into PV1. No new physics/GPU/PBS/training;
the independent lease is CLOSED. The 24-group future pilot is NOT_AUTHORIZED.

**PV1-LG04: child09 is queued once; the existing sole owner is attached.**
[Actual continuation with complete-bank binding and native queue receipt](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_B0_09_COMPLETE_BANK_COLD_AND_NATIVE_QUEUE.json) ·
[Actual owner continuity](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_WORK_WINDOW_HANDOVER_ACTUAL.json).
Runtime2ac48b1a passed exact push Actions39364285377/job118437383591.
Both cold identities, final immutable inputs and139numeric/RGB bindings pass;
the full bank is directly reused from closed child03 with unchanged hashes.
677540 is PV1 PBS8, CFP04-CF-007,1H200/12CPU/128GB/8h, nativeQ for ngpus.
The site225GB rewrite was corrected once within the same queued job; no replay
or cancel. Its own600s/80cycle/fault and full payload checks remain pending
allocation. B0 remains12/48,8success/4validfailure; updates0,queries862,
selected/deployment NONE. Settled16096H200s/reserved28860s, no unknown effects.

**PV1 child08 is closed; the full frozen bank remains intact in child03.**
[Qualification, data-binding diagnosis and actual two-site closeout](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_B0_08_DATA_BINDING_FAILURE_AND_CLOSURE.json):
677341 passed its own600s/80cycle/fault qualification, then failed before model
load because child04's metadata-only directory was bound as the full bank.
The original child03 bank has the exact frozen metadata; no data reconstruction
or new download was required. Child08 charged698 H200 seconds; both leases
are CLOSED and all CPU/GPU reservations are0. Source2ac48b1a adds checks for
complete bank metadata, actual raw/numeric paths and final immutable inputs
before prepare/qsub. 894PI0294 tests/818subtests and827repository tests PASS;
15numerical hashes unchanged. B0 remains12/48, updates0, queries862,
selected/deployment NONE. The following running paragraph is historical.

**PV1-LG04 is adopted; child08 is the sole active B0 continuation.**
[Cold identity, native submission and running snapshot](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_B0_08_COLD_BIND_AND_NATIVE_RUNNING_RECEIPT.json):
exact runtime `fb00da44` passed Actions39348602625/job118387138067 and both
native child identities. Core zero-integration model/RGB parity passed. The
single native submission is677341 under CFP04-CF-007, global PV1 PBS7. Its
site-rewritten225GB request was changed in-place once back to the immutable
128GB request with every non-memory identity unchanged. It reached native R on
hopper-24 and the sole local owner is attached at low frequency. Fresh project
identity passed and the same job was released exactly once; its original native
premodel qualification is running, with no qualified model state at the report
snapshot.
B0 remains12/48 (8success/4valid task failures), selected/deployment=NONE.
A/B training remains gated on independently audited B0=48/48.

**PV1 child07 is closed as a zero-science premodel engineering failure.**
[Immutable-fixture mode failure and resource closure](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_B0_07_IMMUTABLE_BINDING_PREMODEL_FAILURE.json):
job677332 exited before model load because a sealed fixture was rebound0644
instead of required0400. It used58 conservative H200 seconds; both leases are
CLOSED and GPU reservation is0. There were no new queries, physics, task
episodes, optimizer updates or unknown effects. This is not a valid task
failure and does not change the12/48 B0 science baseline.

**PV1-LG04 is adopted; child06 is a closed premodel engineering failure, not a task result.**
[Native failure, zero-science closeout and bounded NFS visibility repair](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_B0_06_PREMODEL_NFS_VISIBILITY_FAILURE_AND_REPAIR.json):
job676926 completed the real80-cycle transport fixture and both process faults,
then exited before model load when a rapidly replaced isolated heartbeat was
temporarily invisible. It charged634 conservative H200 seconds; both leases are
CLOSED and Core/Hopper reservations are0. The fixture-only ENOENT/ESTALE
recovery is fixed at `fb00da44`, with exact Actions39348602625/job118387138067
success. Production heartbeat/watchdog and raw-member/hash gates are unchanged.
B0 remains12/48 (8success/4valid task failures), updates0,
selected/deployment=NONE. Next work is fresh cold admission for only the36
unrun B0 conditions; no A/B training before complete B0.

**PV1 full-state checkpoint reuse now has an executable cold path, not a native copy receipt.**
[CPU implementation receipt](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_FULL_STATE_CHECKPOINT_REUSE_CPU_IMPLEMENTATION.json):
the existing closed-sibling validator now performs an exact member copy through
a content-bound staging directory, reuses already verified partial members after
an interruption, atomically exposes the completed checkpoint, and reruns the
native resume reader before writing a receipt. Source bytes and campaign update
counters stay unchanged; changed or extra partial members fail without cleanup.
879 tests/818 subtests PASS. No real checkpoint was copied, no GPU/PBS/model was
started, and current B0 job675967 remains on its immutable source. Actual use
still requires complete B0, registered cold peers and fresh resource admission.

**PV1 evaluation efficiency: actual service and serial-control cost measured.**
[Original642 service envelopes and65411 local RPC metrics](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_ACTUAL_EVALUATION_SERVICE_AND_RPC_COST.json):
same-Hopper inclusive service mean0.31655s/total203.2243s; serial RPC blocking
3346.3777s versus9612s original allocation. Service includes preprocessing/IO;
intervals overlap, so neither is a GPU utilization or kernel-time measurement.
Cold opt-in owner-only heartbeat avoids RPC renewing owner liveness on every
poll while retaining the original300s stale-owner stop and actual60s owner
renewal. Both manifest/stage must opt in; old04 stays on1c4cdc9a unchanged.
876CPU/816subtests PASS. No native deployment or speedup claim, no new queries,
updates, physics or budget. Full B0 and matched A/B remain the scientific path.

**PV1 TRAIN event audit: all139 frozen parents read, no sampling intervention.**
[Actual500Hz/20Hz event exposures and all parent raw identities](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_ADMITTED_TRAIN_PHYSICAL_EVENT_EXPOSURE.json):
all139 original scores recomputed exactly; T1 transport-then-native-grasp-end
appears in all75 parents but only1.1647% of Prefix8/6.4892% of H40 windows.
T2's12419 raw contact-bit transitions in64 trajectories are contact chatter,
not independent interactions. Categories overlap; new-bank exposure is not the
combined actual A/B/C1 sampler consumption. No fitted-error/causal-underexposure
or P3 qualification claim. Frozen raw+full-audit hashes select the admitted
saved-audit view without overwriting original failed audits; card canonical
digest and file-byte hashes stay separate. 871CPU/816subtests PASS,0new
queries/updates/physics,15 numerical files unchanged. Continue original B0/A-B.

**PV1 GPU-efficiency implementation: actual CPU findings, not a measured speedup.**
[Execution boundaries and first100-update instrumentation](docs/operations/2026-10-19_PV1_GPU_EFFICIENCY_SAME_SEMANTICS.md) ·
[Actual642-array attribution,537-tensor CPU equivalence and bank units](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_GPU_EFFICIENCY_ACTUAL_CPU_AND_COLD_IMPLEMENTATION.json).
The original2.102m/s T1 veto is at H40 points24->25, not Prefix8; it stays
REJECT. This does not establish tail-loss causation or bypass full-H40 safety.
537 gradients match exactly across four native CPU dtypes;1074 scalar reads
become1, not yet a CUDA speed measurement. The opt-in cold performance plan
keeps legacy audit through100 local updates, requires3 exact actual CUDA/RNG
checks before later batched scalar transport, and adds no updates/queries.
Both arms require the same plan; B0 remains on its pinned1c4cdc9a capsule.
868CPU/816subtests PASS,15 numerical hashes unchanged. Actual P2 profiling,
CUDA equivalence, full B0_48 and native cold primary bindings remain pending.
New statistics/loss/history/workers/features/multi-actor changes are not adopted.

**PV1 B0 third allocation actual effects and resources are now closed.**
[All ten original attempts, nine valid chains and one engineering closure](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_B0_03_ACTUAL_CLOSED_EFFECTS_AND_RESULTS.json):
6success/3validfailure/1engineering-incomplete; all10 raw audits PASS,
all9 valid independent two-site chains PASS. All642 native draws classified:
639 executed prefixes,1 uninstalled reference veto,2 zero-live S1 vetoes.
The partial T2 has29 real draws/ACKs plus1 unused observation, unknown effects[].
Both old leases CLOSED/all workers terminal/original STOPs retained. Local
projection repaired once from saved native receipts, without remote closure replay.
Exactly39 conditions remain, including one diagnosed engineering replacement;
no completed valid condition repeats. New cold binding is not qualification
or a completed continuation; each next allocation must pass its own real checks.
0optimizer updates; cumulative11154H200s/3PBS before any new allocation.

**PV1 B0 third allocation ended on a diagnosed finite-control cap, not training.**
[Actual native termination, preserved outcomes and cold repair](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_B0_03_FINITE_RPC_EXHAUSTION_AND_COLD_REPAIR.json):
675394 F/1;6success/3validfailure/1engineering-incomplete. Core RPC exited
normally at its original40000call cap; the interrupted T2 recorded29prefixes
and5800live ticks. Raw audits PASS/nativefault0, but independent partial-effect
closure/replacement eligibility still pending. STOPs/raw evidence retained.
9612H200s charged here,11154 cumulative,0GPUreserved/0updates/642draws.
New250ms/150000call finite PV1 control envelope is CPU-tested preparation,
not a native fix claim or resubmission; legacy40000/50ms defaults remain.
846CPU/778subtests PASS;fresh same-job qualification required after cold binding.

**PV1 actual seven-condition snapshot and next-cold training admission.**
[Seven raw/causal-audited B0 outcomes and terminal cost evidence](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_SEVEN_ACTUAL_B0_CONDITIONS_AND_COST_EVIDENCE.json):
5success/2validfailure,0nativefault,all7 independent chains PASS;not complete48.
The second negative is original S1 pre-execution veto at11.2s,29draws/28ACKs,
zero live execution of its rejected prefix. Earlier reference-speed veto retained.
[Cold full-state continuation and fresh mandatory-work budget admission](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_COLD_RESUME_AND_FRESH_FUTURE_ADMISSION_PREPARATION.json):
838 CPU tests/773subtests PASS,source integrated but NOT native deployed.
Prepare/submit recomputation and fresh same-job pre-training/DEV release are
required for the new cold code. Private binder/coordinator integration and actual
B0 completion still pending. Subagents stopped;one original owner continues.

**PV1 parallel CPU preparation, not an early training launch.**
[Probe recovery, P2 second-seed dispatch and future-work forecast](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_PARALLEL_TRAINING_READINESS_PREPARATION.json):
821 CPU tests/742 subtests PASS;15 frozen numerical hashes unchanged.
Five actual B0 conditions independently audited (4success/1validfailure), not48.
Current675394 stays on e6cd88c4,0optimizer updates. Preparation has no native
deployment claim; forecast integration and full actual B0 admission remain required.
One publisher/native owner and one GPU; no extra budget or clock.

**PV1 actual update: first new-DEV T2 success and T1 valid failure; B0 still incomplete.**
[First two actual outcomes and independent two-site causal audits](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_FIRST_TWO_B0_ACTUAL_RESULTS_AND_CAUSAL_AUDITS.json):
same CORR2048/F0-000/bank294101, T2 completes32s/641pairs/80draws-ACKs/41hold;
T1 stops28.8s at original H40 linear-speed gate2.102>2.0m/s,73draws/72ACKs,
zero executed rejected prefix,0S1preflight/nativefault. Both raw and real
camera/action/ACK/noise/array chains PASS; actual initial state and both cameras
equal across tasks. Only2/48 reported,not aggregate reliability or new training.
Coverage bank unchanged; actual two-row admission correctly rejects incomplete48.
[Current qualification, consumed-bank identity and first live DEV snapshot](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_NATIVE_QUALIFICATION_BANK_CONSUMPTION_AND_B0_STARTED.json).
675394/policy-b0-03 actually passed its own600s/80cycles/all8faults and loaded
CORR2048 after full fixed-revision139-parent bank hashes/official collator checks.
First DEV condition has real observation/action/ACK progress, not a terminal task
result or complete48. Running source remains immutablee6cd88c4; this report/helper
publication does not hot-update it. Earlier qualification failures remain.
The new baseline-admission helper is CPU preparation only: all48 actual audited
conditions required, no added baseline success-rate gate. Training, replica,
unique-candidate80 and runnable policy package are still pending. The following
first-capture/read-side/preparation paragraphs retain their historical snapshots.

**PV1 ADOPTED: one source-domain shared T1/T2 policy, independent seven-day scope.**
[Fixed execution entry](docs/handover/2026-10-19/PV1_MULTITASK_POLICY_EXECUTE.md) ·
[Actual adoption](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_ADOPTION_AND_RESOURCE_BINDING.json) ·
[FD2-based route](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_FD2_AND_TEACHER_DECISION.json).
Authority9eff29a44a193dd4d3620dc0bfe5764a402eff42; T0=1792348196,
hard close2026-10-26 02:29:56SGT. CLEARANCE_PAIR meets the predeclared
contact-gap gate, but training still requires the new admitted data bank.
Other eligible projects do not increase96H200h/24PBS or one-GPU concurrency.
Final20H200h and physical/query/storage reserves remain protected.
[P1 first actual capture and read-side recovery](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_FIRST_CAPTURE_AND_READSIDE_RECOVERY.json):
128 groups statically sealed; first2 Teacher T1 successes independently audited
without physics replay. Four initial T2 cards remain unattempted in this snapshot.
Official fixed-statistics codec/data-bank admission and new GPU work are pending.
[P1 four-worker actual expansion](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_FOUR_WORKER_CAPTURE_ACTUAL.json) and
[first6 official codec results](reports/PI0294/PV1/20261018T182956Z-localpc-pv1/PV1_FIRST_OFFICIAL_CODEC_ADMISSION.json)
supersede that first2 snapshot: all6 first Teacher episodes pass physical/raw
audits, but only4 pass the unchanged official codec. Both12mm references
have clipped relative-Z labels beyond the original numerical gate; raw task
success is retained. The finite ordered CPU stream checks every new parent,
including4mm; it freezes a bank only after all independent/directional gates.
Fallback to original-reference coverage needs the predeclared support evidence,
not an assumption that all later12mm fail. No normalizer/decoder refit or GPU
work has occurred. Actual capture source stays26d86625 and reader ff430330;
management HEAD is not a hot update to those immutable consumers.
Hopper cold admission/campaign-ledger code is prepared and CPU-tested only,
not an actual allocation, premodel qualification or complete neural dispatcher.
The full policy-release goal remains active; selected/deployment NONE.

**FD2 CLOSED: all8 actual Teacher trials independently audited.**
[Final review](reports/PI0294/20261018T171916Z-localpc-fd2/FD2_FINAL_SCIENCE_REVIEW.md) ·
[All8 results](reports/PI0294/20261018T171916Z-localpc-fd2/FD2_ALL_EIGHT_ACTUAL_RESULTS.json) ·
[Public actual videos](reports/PI0294/20261018T171916Z-localpc-fd2/FD2_ACTUAL_VIDEO_ARCHIVE.json) ·
[Resource final](reports/PI0294/20261018T171916Z-localpc-fd2/FD2_RESOURCE_FINAL.json).
Original4mm4/4 and12mm4/4, four paired fresh TRAIN layouts,0S1/nativefaults.
Qualified contact-phase minimum gaps improve8.002-8.027mm in4/4pairs;
whole-episode minima and native collision-query values are separate metrics.
256000live+shadow ticks/684starts; lease CLOSED, future reserve/unknown0.
No GR00T inference/training/GPU/PBS occurred in FD2, no neural benefit proved.
Its own no-training scope and budget stay closed; PV1 has separate authority.
WBC/plant/gripperPD/S1/score frozen, selected/deployment NONE, TF-R3 STOP retained.

**FD1 CLOSED (2026-10-19 SGT): original40/42 preserved; no training or WBC change.**
[Final failure-attribution review](reports/PI0294/20261016T191445Z-localpc-mp1/FD1_DIAG/FD1_FINAL_ROOT_CAUSE_REVIEW.md) ·
[Exact geometry and one explicit gap](reports/PI0294/20261016T191445Z-localpc-mp1/FD1_DIAG/FD1_S1_FIRST_GEOMETRY.json) ·
[FD1 resource final](reports/PI0294/20261016T191445Z-localpc-mp1/FD1_DIAG/FD1_RESOURCE_FINAL.json).
Actual22/23 S1 rule/tick reconstructions qualified:20 pad/table,1 TCP velocity,
1 finger-validity JOINT_LIMIT branch;1 spent T2 geometry remains UNQUALIFIED.
All14 T2 vetoes followed contact/motion; Teacher approach was not actual phase.
This identifies immediate triggers, not a unique training root cause or WBC fault.
FD1 used177340 shadow ticks/45 starts,0 live episodes/NN queries/GPU/PBS/training.
New FD1 lease CLOSED; old N1 leases, raw scores and STOPs unchanged.
[One TRAIN-only dynamic expert-clearance pilot](reports/PI0294/20261016T191445Z-localpc-mp1/FD1_DIAG/FD1_NEXT_EXPERIMENT_DECISION.md)
was NOT_AUTHORIZED in that historical draft; its later, separate FD2 authority
has now closed as described above. Selected/deployment NONE.

**MP1-N1 CLOSED: complete audited40, T1 10/20 andT2 4/20; both14/20 R&D targets NOT MET.**
[Final science review](reports/PI0294/20261016T191445Z-localpc-mp1/N1_FINAL_SCIENCE_REVIEW.md) ·
[All40 cells/all42 actual attempts and fixed video URLs](reports/PI0294/20261016T191445Z-localpc-mp1/N1_SOURCE_CONFIRM40_FINAL_RESULT_AND_EVIDENCE.json) ·
[Final native/resource/24closed leases](reports/PI0294/20261016T191445Z-localpc-mp1/N1_RESOURCE_FINAL.json).
All42rawaudits PASS,actualnativefault0;23S1rejections and3safe32sgoalfailures
remain failures. All2319draws=2296executedACKprefixes+23S1zero-liveprefixes,
alloriginalnoise/arrays/freshpreviousACK-camera chains verified,unknown0.
Last674009F/0,12680H200seconds;confirmation21136/2PBS,source63969/12PBS,
parentH68469/15PBS,V22A40seconds. GPU/CPU/networkfuture reservations0;
allownedscience/control consumers terminal;oldSTOPs/clock/data/models retained.
[All42actualvideos fixed-revision readback](reports/PI0294/20261016T191445Z-localpc-mp1/N1_SOURCE_CONFIRM40_FINAL_VIDEO_ARCHIVE.json)
atHF2ceea49321f5806a9312c7bcdeea06335ccf4fc4,READYlast,no missingaudit/clip.
Protocol complete is NOT policy-milestone success or Go2/WBC/realrobot proof.
MethodNONE/0newconfirmationtraining;selected/deploymentNONE,TF-R3STOP unchanged.
[One nextTRAIN-onlyT2coverageBC draft](reports/PI0294/20261016T191445Z-localpc-mp1/N1_NEXT_T2_TRAIN_COVERAGE_DRAFT.md)
isNOT_AUTHORIZED. No automaticextraPBS/oldbalanceorclock reuse. All following
dated running/preparation entries are historical snapshots,not currentliveness.

**Actual: FIRST new-source T2 confirmation success independently audited.**
[First T2 success and finite paired progress, retaining all negatives](reports/PI0294/20261016T191445Z-localpc-mp1/N1_CONFIRMATION_FIRST_T2_SUCCESS_AND_FINITE_PAIRED_PROGRESS.json).
F2/000T2bank29201:32s/641actualpairedframes/80draws-ACKs/41hold,realcontact
and0.183699m horizontalpush,no sustainedgrasp/lift,0S1/nativefault,TeacherOFF.
18raw/score/safety/video checks plus separate80freshobservation/action/ACK/
fixednoise receipts PASS. Same94adsharedweights now have actualnew-source
successes inBOTHtasks;not T2reliability,RecoveryOraclequalification orGo2proof.
Finite28frozen-attempt snapshot:T1success6/qualified14/planned20,T2success1/
qualified12/planned20;T1F0/001 andF1/003 each2/2banks. Incompletepairs excluded
from0/2counts. Evenallremaining successes allow atmost12T1/9T2,beloworiginal
14/tasktarget;finish originaldesign,no early-stop/reselection/thresholdchange.
Originaltwoengineeringpartials retained,their sole replacements nowaudited;
laterinflight work excluded. 674009stillR/GPUcostOPEN,0training,newbudget0,
candidate/plan/S1/WBC/noise/clock/STOPs frozen;final40/video/closure pending.

**Actual: allocation02 qualified; first T1 scene succeeds under BOTH fixed banks.**
[Own native qualification, full independent audits and exact paired-scene evidence](reports/PI0294/20261016T191445Z-localpc-mp1/N1_CONFIRMATION02_NATIVE_QUALIFICATION_AND_FIRST_COMPLETE_SUCCESS_PAIR.json).
674009 nativeR/source887e6562:own81reads/600.084s/80realcycles/all8faultsPASS
BEFORE model load,raw5625953d...,originalwatchdogs preserved. NewF0/001T1
bank29201 completes32s/641pairs/80draws-ACKs/41hold,0S1/nativefault,TeacherOFF;
18raw/score/safety/video checks plus fresh saved-camera/action/ACK/fixed-noise
chainPASS. With originalbank29202 success,this is the first2/2T1scene,
not twoindependent scenes,full40reliability or Go2/WBC/realrobot proof.
T2F0/000bank29201 sole engineeringreplacement is a valid originalS1rejection
at20.4s/409pairs/19auditchecksPASS,not success;oldpartial retained,no retry.
Exact22frozen-attempt snapshot:T1qualified10/success4,T2qualified10/success0;
6incomplete scene-pairs/task are NOT0/2pairs. Laterinflightwork excluded.
Samecandidate94ad/plan6e5a;no newtraining/candidate/clock or safetychange.
NativeGPUcostOPEN,remainingconditions continue;finalall-effect/video closure
pending. Source12of12/confirmation2of2,noautomatic extraPBS.

**Actual: LAST original confirmation allocation02 submitted once, native Q.**
[Exact submission, queue memory readback and original resource limits](reports/PI0294/20261016T191445Z-localpc-mp1/N1_SOURCE_CONFIRM_ALLOCATION02_ACTUAL_SUBMISSION.json).
674009/source887e6562,CFP04-CF-007,auto->small,1GPU/12CPU/128GB/8h.
Native site rewrote the128GBrequest to225GB;the mature exact-owned queued
memory decrease was applied once and128GBactually read back,no cancel/re-qsub.
Confirmation2of2/source12of12/parentHopper15of24;28800H200seconds reserved,
not consumed. Prior55789parent/51289source/8456child settledseconds retained.
Fresh project/member/quota/peer/resource/import gates PASS;no otherownedGPUpeer.
Own600s/80cycle/all-fault nativequalification NOT_RUN whilequeued;no newmodel,
query,physics ortraining. Stale673388MODEL_READY is NOT current readiness.
No thirdconfirmation/13thsource or automatic extra submission. OriginalN1
clock,final6H200hreserve,candidate94ad/plan6e5a,S1/WBC/score/STOPs unchanged.

**Actual V2 cold deployment and real derived-reader profile now PASS; no new PBS.**
[Both-site V2 acceptance and preserved original identities](reports/PI0294/20261016T191445Z-localpc-mp1/N1_CONFIRMATION02_COLD_V2_AND_REAL_OBSERVER_ACCEPTANCE.json).
Both nodes now bind immutable887e6562/exactCI39114861370SUCCESS,623members,
with all15numerical+12learning hashes unchanged. The actual Hopper derived
reader manifest passed its real existing runtime profile without launching
an observer/model or fabricating MODEL_READY. Original b379 prototype/V1
inputs and all6b science/results remain; fresh supplemental readback corrects
an old-binding receipt field without overwriting original activation receipts.
Source11of12/confirmation1of2,930draws/20attempts/GPUreserve0 unchanged.
Repeated human call_d8b8e55b8bd84f34be89bf13f7c02e20 already funded673008;
it adds NO new slot. Only original separateN1 authority governs current work.
Fresh auth/peer/quota/resource gates and own600s/80cycle/all-fault native
qualification are still required before new model work in the last allocation.
No third confirmation/13thsource, new training, clock or safety override.

**Observer preflight caught a derived-manifest incompatibility BEFORE the last PBS.**
[Actual legacy rejection and prepared strict V2 repair](reports/PI0294/20261016T191445Z-localpc-mp1/N1_CONFIRMATION02_OBSERVER_MANIFEST_COMPATIBILITY.json).
The rootb379ce20 cold entries below remain valid,but its qualified read-only
observer clone was actually rejected by the root-authority comparison.
No observer/model/PBS/physics was launched in the preflight. Strict V2 repair
keeps identical source/seal/profile/authority/data/limits and only admits the
exactvisibility subcapsule plus immutable session-bindinghash. CPU622tests/
569subtestsPASS;actual repaired derived-profile andV2deployment still pending.
OriginalV1inputs/capsules stayattheir paths;CPU-onlyb379 prototype mustremain
zero newnative/model/case consumption. Original40/candidate/noise/clock/STOPs
unchanged;source11of12/confirmation1of2,not a policyfailure ornewbudget.

**Actual: both confirmation02 cold entries PASS; last original allocation not yet submitted.**
[Actual two-site cold adoption and preserved resource/effect identities](reports/PI0294/20261016T191445Z-localpc-mp1/N1_CONFIRMATION02_ACTUAL_COLD_ADOPTION.json).
Both nodes now use immutableb379ce20/sourceCI39103735777SUCCESS;623members,
15numerical+12learning hashes unchanged. Original52Core/14Hopper recorded
processes terminal;old673388F/8456GPU seconds/930draws remain accounted.
Both approved entries andimmutableworker manifests PASS after atomic pointer
advance. OldSTOPab07 retained,new02STOPabsent,candidate94ad/plan6e5a unchanged.
Originalprivateowner/clientbytes exactly recoverable;no oldcapsule overwritten.
Sourcefamily11of12/confirmation1of2 still;newGPU/PBS/models/physics/training0.
Next:fresh project/auth/quota/peer gates andthe lastoriginal boundedallocation,
then own600s/80cycle/all-faultqualification before anymodel. Notnewnative
qualification orcomplete40;selected/deploymentNONE,originalN1clockunchanged.

**Prior preparation: confirmation02 cold guard implemented and CPU-tested.**
[Exact preparation scope](reports/PI0294/design/SOURCE_CONFIRMATION02_COLD_REPAIR_PREPARATION.json)
and [cold-boundary requirements](docs/protocols/PI0294_SOURCE_CONFIRM_V1_FINITE_IMPLEMENTATION.md#confirmation02-cold-repair-preparation)
preserve oldSTOPab07 atits originalpath androute onlyfuture02 faults to a new
STOP. Native safety cannot be resolved. Requires exactprior20rows/930effects/
8456GPUseconds,alloldprocessesclosed,originalplan/candidate/27protectedhashes,
both approved entries andownfresh600s/80cycle qualification. Originalbinding
artifacts remain;allocation02 owner now refuses the old6b firstbinding artifact.
CPU620tests/566subtestsPASS;actualcoldactivation/newPBS/model/physics0 atpreparation.

**Actual CPU transport diagnosis: redundant inflight progress fsync removed; native continuation pending.**
[Actual80-cycle recorded-byte route and controlled before/after proof](reports/PI0294/20261016T191445Z-localpc-mp1/N1_ACTUAL_RECORDED_TRANSPORT_AND_PROGRESS_JOURNAL_REPAIR.json).
160RPCs over the original dualSSH route,one95081-byte original double-camera
packet,all80checksums match;median cycle0.03879s/max1.24919s. This is recorded
transport,notfresh-observation physics or a newallocation qualification.
Controlled old written-progress fsync21.022s consumes unchanged20s watchdog;
new batched progress returns the same isolated request in0.576s without that
inflight fsync. Durable pre-wire intent and durable response/failure remain;
no action/query/install retry or watchdog extension. HistoricalACK timeout's
unique cause still NOT PROVEN. Allprobeprocesses terminal;NN930/cases20 unchanged.
Old fault close did not observe SSH exit;later PID/start disappearance verified,
exitcodeUNKNOWN retained. Bounded discard-only close is CPU-pipe tested,
not yet realSSH/native accepted. Actor remains6b291000,engineeringSTOPs retained.
Next:publish/CI,coldbinding,then own600s/80cycle qualification before anymodel
in the last legalallocation. No newGPU/PBS/training/physics from these probes.

**Current: confirmation allocation01 terminal; whole recorded effects closed, fixed40 incomplete.**
[Actual all40-cell partial coverage, all20 raw audits and resource terminal](reports/PI0294/20261016T191445Z-localpc-mp1/N1_SOURCE_CONFIRM_ALLOCATION01_TERMINAL_PARTIAL_COVERAGE.json).
673388 is nativeF/1,8453 wall/8456 conservativeH200seconds settled;
source51289seconds/11of12PBS,parentHopper55789seconds/14of24PBS,GPUreserve0.
T1:3success/6validfail/1engineering/10unrun;T2:0success/9validfail/1engineering/10unrun.
Both tasks9of20qualified;all20actualattempts independently audited.
930originalgenerations=917executedACKprefixes+13S1rejectedzero-liveprefixes;
allarrays/fresh-observationchains andall930fixed-noise receipts PASS,unknown0.
Three positiveT1conditions do not establish full40reliability or targetWBCsuccess.
Both engineeringSTOPs retained;stale residentREADY is not native liveness.
Latest readonlyACK call hit unchanged20s watchdog;serverprocessed it,but lacked
absolute timestamps,so timelyreply/unique cause are NOT proven. New diagnostics
are CPU-tested only,not deployed/nativequalified;no automatic nextPBS.
Remaining20originals+2sole engineeringreplacements require diagnosis/regression,
coldbinding andfreshown nativequalification before the last legalallocation.
[Follow-up card54bc85da](docs/handover/2026-10-18/MP1_N1_CONFIRMATION_TO_NEXT_STAGE_EXECUTE.md)
ADOPTED in the existingN1clock/budget;no newtraining,selected/deploymentNONE.
All following dated snapshots retain their original observation states.

**Prior snapshot: second different new-source T1 condition independently verified.**
[Second success and its retained other-noise negative](reports/PI0294/20261016T191445Z-localpc-mp1/N1_SOURCE_CONFIRM_SECOND_INDEPENDENT_SUCCESS.json).
F1/002 bank29202:32s/641 real paired frames/80 draws and actual ACKs,
18 independent raw/score/safety/video checks plus separate dual-SSH chain and
fixed-noise receipt audit PASS. Lift0.079012m/carry0.187603m/hold41/41;
S1 refusals0/nativefault0/TeacherOFF. The same scene's bank29201 remains a
safe32s object-task FAIL: grasp/lift/carry true but hold0/41/released0,
despite identical initial camera bytes. Both outcomes remain, not a claim of
stable scene success or a unique noise-cause attribution. Two positive
conditions are not the full40-condition rate denominator. Same actor6b291000,
job673388 still running/zero training/GPU costOPEN;selected/deploymentNONE.

**Actual checkpoint maintenance, without blocking confirmation:**
[Current scoped inspection receipt](reports/PI0294/20261016T191445Z-localpc-mp1/CHECKPOINT_RETIREMENT_DURING_CONFIRMATION_ACTUAL.json).
Hopper905/CoreSSD2309/AMDshared1425 directories;31 checkpoint-shaped
containers,9 intact large learning states retained,eligible failedpayload0,
deleted0/freed0. BC1631 is valid superseded state, not a failed checkpoint:
current inference does not load it, but actual historical resume witness
and payload-replay independence remain distinct; no unsafe retirement.
Core/Hopper original manifest/plan/candidate/leases unchanged and nativeR
after observation;no model/physics/PBS/cancellation or historical bytes counted.

**Current: first new-source confirmation success independently audited.**
[Exact single-condition receipt](reports/PI0294/20261016T191445Z-localpc-mp1/N1_SOURCE_CONFIRM_FIRST_INDEPENDENT_SUCCESS.json).
CORR2048, T1 F0/001, fixed bank29202: autonomous32s/16000ticks/641 paired
frames; original S1 refusals0/nativefault0, lift0.083652m/carry0.165017m,
hold41/41. All18 original raw/score/safety/video checks PASS. Separate
dual-SSH readback joins80 actual draws/ACKs/fresh observations and original
arrays/camera hashes/fixed-noise receipts; no model or physics rerun.
One condition is not complete40-condition confirmation or target-WBC success.
673388 continues on immutable6b291000; GPU costOPEN, zero training,
selected/deploymentNONE. Actual decoded641-frame video exists locally;
public video upload is NOT yet performed. Pure archive inventory code is
CPU-tested/prepared, not deployed in the actor;597 PI0294 tests PASS.

**Current: own native qualification PASS, confirmation evaluation in progress.**
[Actual qualification and selected engineering-effect closure](reports/PI0294/20261016T191445Z-localpc-mp1/N1_SOURCE_CONFIRM_QUALIFICATION_AND_SELECTED_CLOSURE.json).
673388 own81reads/600.0083296215162s/80realcycles/all8faults independently
verified;models/queries/physics0 duringqualification,rawreceipt1a3938d73a0021393d0b4ab06099ba229ea68c9ae529f3da30e25aa2546dc450.
Actor remainsimmutable6b291000 andnativeR atreceipt;no newtraining.
One T2 condition is preservedENGINEERING_INCOMPLETE at15.6s,notpolicyfailure.
Its39actualdraws/39ACKs/7800integrations andallrawarrays joined,selected
worker andbothRPCs terminal/no pendingdraws. Only one provenidleorphan
controlRPC wasterminated afteritsSSHclient exited255;noGPU/physicsworker
cancelled. Connectionfailure's unique cause remainsunproven. This isnot
whole-runclosure or finalindependenttaskaudit;rawcoverage is explicitly
preliminary. Purecoverage/noiseauditcode isCPUtested/prepared,notactorcode.

**Prior snapshot: first native allocation running, qualification pending.**
[Actual one-job native receipt](reports/PI0294/20261016T191445Z-localpc-mp1/N1_SOURCE_CONFIRM_ALLOCATION01_ACTUAL_SUBMISSION.json).
673388.hopper-m-02 is actualR/oneGPU/8h,sourcefamily11/12,parentHopper14/24.
It has its own600s/80cycle/all-fault qualification before model load;models0
andqueries0 at this snapshot.28800H200seconds reserved,actual costOPEN,
not settled or a task result. Runtime6b291000,originalN1clock/final6hreserve
unchanged,no automaticnextsubmission. One local import-pin rejection occurred
before any native request;retained and repaired with immutable package lookup
and the mature visibility observer,zeroPBS/GPUcost for that local rejection.
6000preparationCPUupperclosedonce;active3960controlupper separatelyreserved.

**Prior boundary: source confirmation actually frozen on both sites, before PBS.**
[Actual static qualification and both native bindings](reports/PI0294/20261016T191445Z-localpc-mp1/N1_SOURCE_CONFIRM_ACTUAL_BINDING.json).
One CORR2048 candidate,10 new groups/2 tasks/2 fixed noise banks=40 conditions.
11 original static candidates/22 engine initializations/0 integrations;one
static-only failure retained. All20 selected task qualifications PASS.
Fixed HF756ee34c4139bb1acf8930a3861c49da2356e73a actually consumed onHopper;
63payload files+READY,allhashes readback,no payload reupload or modeldownload.
Both immutable runtime6b291000 entries/15numerical+12learning hashes PASS;
exact CI39091231819SUCCESS. Static-only policy execution explicitly rejected.
New confirmation PBS/GPU/models/queries/policy conditions all0 at binding.
Original source42833GPU seconds/10PBS/2406draws/4096updates andN1clock retained.
Next finite native admission and fresh own600s/80cycle qualification;no new
training,optional method NONE,WBCfrozen/TF-R3STOP retained,selected/deploymentNONE.
Older snapshots below retain their original times and are not current job states.

**2026-10-18 morning checkpoint before confirmation binding:**
[N1 morning report, actual09:48:50SGT snapshot](reports/PI0294/20261016T191445Z-localpc-mp1/N1_MORNING_REPORT.md).
The [actual10:00native follow-up](reports/PI0294/20261016T191445Z-localpc-mp1/N1_MORNING_1000_NATIVE_READBACK.json)
has empty ownedHopperqueue/all delivery+shadowworkers terminal,sourcecosts
unchanged. [Confirmation scope preparation](docs/protocols/PI0294_SOURCE_CONFIRM_V1_FINITE_IMPLEMENTATION.md)
was pureplan/cardcode only at that snapshot,not native admission.
Complete paired endpoints,16matched prefixdiagnostics andall12CORRvideos
are actual. Nooptionalnewtraining;oneCORR2048confirmationcandidate ranks
firstunderoriginalrule. ConfirmationwasNOT_FROZEN/NOT_RUN then,goalACTIVE/incomplete.

**Current N1: actual matched anchor diagnosis and all12 CORR endpoint videos delivered.**
[Closed NI16/HF/Core matched shadows](reports/PI0294/20261016T191445Z-localpc-mp1/N1_CURRENT_EIGHT_ENDPOINT_MATCHED_SHADOWS_ACTUAL.json)
and [all12 actual video fixed revisions](reports/PI0294/20261016T191445Z-localpc-mp1/N1_CORR2048_ALL12_ACTUAL_VIDEO_ARCHIVE.json).
Original8 TRAIN states/4BC512parents,not current2048 on-policy rollouts:
BC T1prefix2/4PASS,CORR3/4PASS;T2both4/4PASS. All16reference gates ACCEPT.
Fresh original-history/native-copy parity passed again at1e-10;4starts,
29202revalidation+2823endpoint=32025shadowticks,196.409127034CoreCPUledger
seconds settledonce,newleaseCLOSED,0live/NN/labels/updates/GPU/PBS.
HF5378b8e8 fixed outputs actually consumed onCore;all12CORR2048clips
sequentially published and READY/hash read back. Old16HFlinks unchanged.
[Mechanism and next branch](reports/PI0294/20261016T191445Z-localpc-mp1/N1_MECHANISM_AND_NEXT_BRANCH.md):
nooptionalnewtraining;CORR2048 ranks first by originalworst-task thenmacro
rule for a single futureconfirmation,notdeploymentselection. SOURCE_CONFIRM
notyetfrozen/run,selected/deploymentNONE. OriginalN1clock/caps/STOPs retained.

**Current actual NI16 terminal: 673318 native F/0, all16 diagnostic draws verified.**
[Terminal and independent same-job qualification](reports/PI0294/20261016T191445Z-localpc-mp1/NI16_ACTUAL_TERMINAL_AND_NATIVE_QUALIFICATION.json).
81 actual post-read samples/600.027s/78 distinct owner epochs,80 real transport
cycles and all8 isolated faults PASS before model load. Original8 TRAIN
observations/noise tensors reused for BC2048 and CORR2048;16 original query
journals/array hashes and own PBS_PROCESS_FINAL reconciled,not policy results.
741 native wall/745 conservative H200s settled once;source42833,parent47333,
GPU reserves0,sourcePBS10/parent13. Combined4096updates unchanged,new physics0.
HF handoff and matched Core shadows NOT_RUN at this snapshot;no automatic11,
no new training or optional method selected. Old snapshots below are historical.

**Current N1: missing16 endpoint-anchor draws have one actual bounded H200 job.**
[Actual NI16 cold binding and single submission](reports/PI0294/20261016T191445Z-localpc-mp1/NI16_ACTUAL_COLD_SINGLE_SUBMISSION.json).
673318 nativeR/7200s/oneH200,source0ad0cc3a/exactCI39085543322SUCCESS,
120oldprocessrecords closed/35outcomes/2390oldqueries/4096updates unchanged.
Actual both-site cold scopes and fresh Hopper eight input/noise receipts
PASS beforeqsub. Only BC2048/CORR2048 each8draws on original8TRAINanchors;
oldBC512draws reused,no repeatedpolicyconditions/108historicalprobes.
At firstsame-job ready snapshot:models0/queries0/physics0,new600s/80cycle
qualification pending. Old09modelmarker/qualification are not reused.
No new training/live/shadow/WBC;no automatic11. Source42088,parentHopper
46588settled+same7200reserve,sourcePBS10/parent13,currentactualGPUcostOPEN.
All original N1clock/caps/final6H200h reserve remain;not new neural results.

**Current N1: all eight original TRAIN anchors now have qualified native-history restoration.**
[Actual eight-anchor gate and closed resource receipt](reports/PI0294/20261016T191445Z-localpc-mp1/N1_EIGHT_NATIVE_HISTORY_COPY_ACTUAL.json).
New d2eee965/CI39084263586SUCCESS,not a reopening of old failed gates:
four complete acknowledged histories and eight initial saved states match
exactly;all four rejected fullstate traces and four accepted highrate traces
match exactly,next saved fullstate maxima2.665e-15<unchanged1e-10.
4parents/8states,4starts/29202shadow/194.95881514CPUledgerseconds settled
once(includes180registrationupper),zero live/NN/labels/updates/GPU/PBS.
NewleaseCLOSED,alloldleases/STOPs/codeunchanged. Future shadow consumers
must reconstruct histories and revalidate this route,not reinject the old
integration-only state.16endpoint draws stillNOT_RUN and require a separate
binding;no optional learning method or SOURCE_CONFIRM selected yet.

**Current N1: both2048endpoint TRAIN/DEV panels complete and independently audited.**
[Actual source09 terminal and complete paired endpoint evidence](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE09_ACTUAL_TERMINAL_AND_COMPLETE_ENDPOINT_PAIR.json).
BC TRAIN3/4,DEV3/8(T1=2/4,T2=1/4);CORR TRAIN3/4,DEV5/8(T1=4/4,T2=1/4).
Two paired T1 DEV improvements,no success-status regressions;T2 success
membership unchanged. CORR has1T2 S1reject+2safe32sobject-goalfailures,
versus BC2rejects+1objectfailure. Exposed4layout DEV,one training seed,
not hidden confirmation or proof of a general causal/onlineDAgger benefit.
Same M1/recipe,matched actualT2 and retainedT1 consumption remain audited;
C1 windows derive from4parents/8segments,not independent demonstrations.
All24endpoint conditions qualified;fixedmatrix31/32 across35actualattempts,
one exhausted CORR512F1/T1 midpoint gap retained,no third attempt.
All33priorattemptrows,4exactgeometries and16fixedHFlinks preserved.

Source09 actual673298 nativeF/exit0,1953nativewall/1956conservativeH200s
settled once:source42088,parentHopper46588,reservation0,sourcePBS9/parent12.
All2390native draws/originalarrays,1941executedACKs,14originalS1zero-live
installs independently reconciled;allrecorded related processes terminal,
unknown/pendingeffects0. Same600s/80cycle qualification remains actual.
No new training,no automatic10,no sourcepolicy/WBC/safety changes.
Separate [native-copy vs integration-restore ablation03](reports/PI0294/20261016T191445Z-localpc-mp1/N1_NATIVE_COPY_ABLATION03_ACTUAL.json)
is now actuallyCLOSED:5200originalhistoryticks exact;nativefullcopy reproduces
all63rejectedshadowrows exactly,while fresh-engine restore and native-copy
plus restore both reproduce the3.4268961e-9gap. This identifies a sufficient
native-history/copy route at one state,not a unique cache/forward cause or
full8anchor qualification.2starts/5452shadow/183.625143285CPUledgerseconds
settledonce,zero live/NN/PBS,original1e-10. Runtime d3aed7fc,
exactCI39083647224PASS;original leases/T0/caps/STOPs unchanged.
SOURCE_CONFIRM and16endpoint-anchor NN NOT_RUN;selected/deploymentNONE.

**Previous source09 admission snapshot and read-only component decode.**
[Actual read-only native state-component decode](reports/PI0294/20261016T191445Z-localpc-mp1/N1_TRACE02_COMPONENT_DECODE_ACTUAL.json)
locates the only above-1e-10 trace difference at live mjSTATE_WARMSTART:
3.4268961e-9. Decoded highrate physical maximum1.3173e-14,steps/safetyexact,
controller tail0. MuJoCo3.2.7 native APIs,zero integrations/NN/PBS;
180.546847382CPUledger seconds settled once including180registrationupper.
This is NOT full8anchor restoration PASS;threshold unchanged,unique cause
unproved,16endpointanchorNN remainNOT_RUN. Old closed leases preserved.

[Actual cold binding,single submission,real qualification and RPC journals](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE09_ACTUAL_COLD_SINGLE_SUBMISSION_AND_RPC_JOURNAL.json).
673298/source976b9584 passed independent same-job raw readback:81actual
post-read samples/600.021045657s,79distinct owner epochs,80real94080byte
RGBtransport cycles,all8isolatedfaultsPASS. Models/NN/physics were0at the
qualification boundary;only afterward did the originalCORR2048remainder
start. No new complete condition is claimed yet. OnlyF2/T1sole engineering
replacement and F2/T2original may run;no training,no automatic10.
Nine cumulative sourcePBS/twelve parentHopperPBS,source40132settled+
28800reservedGPU seconds,parent44632+same reservation. Current actualGPU
cost isOPEN,notzero.109priorprocess records closed;all33outcomes/2230old
draws/4096formalupdates/oldSTOPs and firstN1clock retained. No hot switch.
New local-client prewire/failure journals keep20/60/900s and no-retry rules;
80CPUechoes/actual20.0016stimeout/exit72are local evidence,not site proof.

Single-state trace02 is alsoCLOSED:63shadow ticks reproduce the original
S1reject,but savedfullstate mismatch is3.4268961e-9 at row5/column42 versus
unchanged1e-10. Exact arrays are preserved;unique cause is unproved. Its
180.60674454CPU ledger seconds include180registrationupper+0.60674454
measuredworker seconds. No live action/NN/labels/PBS. OldB1/dynamic01remain
CLOSED;16endpoint-anchorNNremainNOT_RUN. A separately scoped read-only API
decoder is published/CIpassed,not yet dispatched at this snapshot.

**Previous source08 terminal: six CORR DEV cells qualified, one engineering gap and one unrun.**
[Actual terminal,unused-action closure and STOP-inventory repair](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE08_ACTUAL_TERMINAL_EFFECT_CLOSURE_AND_STOP_REPAIR.json).
673235 ended07:38:41SGT after the original20sFILE_RPCtimeout. F2/T1 stopped
at12.4s,31ACKs/6200live ticks,249real dual-camera frames,zero live native
faults. Its32ndmodel draw was generated but never installed;all2230cumulative
draws and original arrays are now reconciled. The partial18check/raw/video
audit passed,but this is not a qualified task result. F2/T2 was not started.
Actual3525GPU seconds settled once:source40132,parentHopper44632,reserved0.
The failed generic settlement was a recursive ancestor inventory bug,new08
STOP not changed old bytes. New5876ce18/CI39080331271SUCCESS preserves all
STOPs and passed the actual frozen partial auditor;current08STOP still blocks
execution. No network-root-cause fix is claimed,no09bound/submitted.
The earlier live-R publication used a prior snapshot and is superseded here.

**Past same-job native qualification PASS; six audited CORR DEV results retained.**
[Same-job native proof](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE08_ACTUAL_NATIVE_QUALIFICATION.json)
and [two new audited DEV results](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE08_FIRST_DEV_PAIR_ACTUAL.json).
673235/source eb851e82 is the existing eighth allocation, not a new adoption:
81 actual reads over600.022s,80 real transport cycles and all8 isolated
fault tests passed before model load. CORR2048 DEV/F1-000-T1 then completed
32s SUCCESS;F1-000-T2 was legally rejected by originalS1 at23.2s. Both have
independent raw/video/fresh-input-action-ACK audits and zero live safety
faults. [Second new pair](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE08_SECOND_DEV_PAIR_ACTUAL.json):
F1/001T1 is another full32sSUCCESS;F1/001T2 safely completes32s but fails
the object goal,not an S1 rejection. Both have independent raw/video and
fresh input/action/ACK audits. Matrix29/32qualified;two DEV cells and the exhausted midpoint
F1/T1 gap remain. All30prior attempt rows,4exact geometries and16fixed HF video
links are preserved. No full DEV rate,causal DAgger gain or deployment claim.

[Actual saved-anchor dynamics gap](reports/PI0294/20261016T191445Z-localpc-mp1/N1_ANCHOR_DYNAMIC_PARITY_ACTUAL_GAP.json):
separate CPU-only child1acfe240 passed5states,then stopped on the sixth
full-observed-trace mismatch at the original1e-10 tolerance. Its S1 decision
and63tick rejection offset matched,but exact failed trace maximum/column
was not persisted;unique cause is unproved.6starts/851shadow ticks/
182.218476217CPU seconds settled once,new leaseCLOSED,oldB1stillCLOSED.
No live action,neural query,training,Teacher label or GPU/PBS. The16new
endpoint-anchor draws remain NOT_RUN;no same-lease retry or threshold change.
This diagnostic gap does not invalidate or interrupt the independent DEV panel.

[Vanda camera03 actual native failure and closed lease](reports/PI0294/20261016T191445Z-localpc-mp1/N1_VANDA_CAMERA03_ACTUAL_FAILURE.json):
1464745 nativeF/exit1,22conservativeA40seconds/10CPU seconds,zero images,
model queries,WBC launches or physical integrations. File visibility passed;
OpenGL identity read failed. Unique root cause and hardware camera quality
remain unproved. Old camera02 stays CLOSED;no automatic camera04.
Source work remains independent,but no GPU job is currently active. OriginalN1T0=1792264081 and cumulative
budgets/costs/STOPs remain. Formal updates4096/pending0;selected/deploymentNONE.

## Historical Snapshots

**N1allocation06 failed before model load; exact sampling mechanism identified.**
[Actual native failure/resource and repair ledger](reports/PI0294/20261016T191445Z-localpc-mp1/ENGINEERING_ENDURANCE_AND_FAULT_INJECTION.json).
673223 nativeF/exit1,653GPU seconds;source cumulative36164,parent40664,
reservations0. Zero new models,NN queries,updates or physics. Capture
measured600.095s and75distinct valid owner readings,but required81samples:
reader/guard/fsync/pump overhead reduced density before duration-only exit.
The strict gate then stopped compute;72/80fixture cycles completed and the
next reserved checksum expired at the original60s limit. Not task failure.
Repair requires BOTH81actual reads and600s,within the same900s capture cap;
blocking native faults now wait until all80nominal cycles and owner closure.
No weakened watchdog,no copied reads,no hot patch,no automatic re-qsub.
Existing25qualified cells and six remaining DEVconditions are unchanged.

**Fifth allocation ended normally; N1 cold continuation submitted once.**
[Actual paired results and resource receipt](reports/PI0294/20261016T191445Z-localpc-mp1/MATCHED_SOURCE_PAIR_FINAL.json).
673008 nativeF/exit0,11392GPU seconds;source cumulative35511,reservation0
before continuation. Matrix25/32qualified:CORR2048TRAIN3/4 and individually
audited DEV/F0-T1,F0-T2 full32sSUCCESS. SixDEVconditions still unrun;one
exhausted midpoint engineering gap remains. No fullCORR DEV rate claimed.
55c0610d exact-head CI SUCCESS. Both real cold scopes passed with84old
process records closed and all1899NN outputs reconciled;13installed S1
zero-live rejections retain original intents without fabricated ACKs.
673223 is the single N1allocation06 submission,auto/1H200/8h maximum,
nativeR/released at latest readback,pre-model entry not yet ready. No training
or repeated valid cases;native600s/
80-cycle qualification must pass before any model load. OriginalN1 T0 and
all cumulative costs/final reserves remain;no automatic next submission.

**N1 cold binding/effect closure code prepared, not yet activated.**
Generic immutable staging/activation now requires the actual recorded transport
plan while retaining all earlier bindings and STOPs. Read-only closure joins
saved RGB,commands,executed ACKs,physics receipts and every native action
array;executed,rejected,generated-uninstalled and inbox-only effects remain
distinct. All resident/fixed-probe draws must match the cumulative ledger.
506-test local suite and21post-guard focused tests pass. The private existing
owner's finite06-12 adapter is prepared,not launched;no automatic next PBS.
Two CORR2048DEV/F0 conditions now independently audited SUCCESS at32s,
zero native faults;remaining six DEV conditions not yet run. No complete DEV
rate or new native transport qualification claimed. Healthy fifth preserved.

**CORR2048 TRAIN4 complete: 3/4 actual successes, all four independently audited.**
[Paired endpoint report and raw-score/video index](reports/PI0294/20261016T191445Z-localpc-mp1/MATCHED_SOURCE_PAIR_FINAL.json).
T1=1/2,T2=2/2,zero native safety faults,TeacherOFF. F0/T1 is a valid
originalS1 rejection at24s;F0/T2,F1/T1,F1/T2 complete32s successfully.
Every condition passes full raw/video and fresh observation/model/action/ACK
chain audit. BC2048TRAIN4also3/4,but its T1success isF0 rather thanF1:
no aggregate TRAIN gain or causal DAgger/generalization claim. Matrix now
23/32qualified;CORRDEV8 is in progress,not a complete DEV rate. Fifth673008
continues originalsource/request;no newPBS. Transport implementation245446c2
has exact-head GitHub CI SUCCESS;native600s/80cycles remainNOT_RUN.

**N1 transport producer and owner IO phase now CPU-tested, not site-qualified.**
[Required endurance/fault ledger with actual local mature-client fault bench](reports/PI0294/20261016T191445Z-localpc-mp1/ENGINEERING_ENDURANCE_AND_FAULT_INJECTION.json).
Recorded real-size dualRGB uses the existing stdio/sealed codec,compute
checksum pump and80once-only fixture installs/ACKs,not model actions or
physics. Actual raw input/output/intent/wire/ACK hashes,isolated faults and
RPC process closure must join the600s native heartbeat before model load.
Pre-qsub rejects a missing/unsealed reference plan.496localPI0294 tests/
372subtests pass;local mature-client20.019s timeout and controlled72exit
both refuse a second request. CPU/local only,not600s/80cycle site evidence.
Private site adapter prepared;no cold binding/deployment/newPBS. General
finite manager and actual prior-effect closure/site qualification still
required. Healthy fifth673008 now evaluates CORR2048DEV8;sourcee418bb79,
4096formal updates,pending0,original4h request and all scientific rules
unchanged. See subsequent actual science receipts for task outcomes.

**Science milestone: both2048 endpoints sealed and actual matched consumption audited.**
[Matched endpoint report; behavior comparison still IN_PROGRESS](reports/PI0294/20261016T191445Z-localpc-mp1/MATCHED_SOURCE_PAIR_FINAL.json).
CORR2048 complete manifest94ad4d0e,combined4096updates,pending0. Both arms
have65536actual visits,32768T2cursors and16384retainedT1cursors exactly
matched;CORR16384C1visits are repeated windows from4parents/8segments,
not independent demonstrations. Both original1030-tensor M1 inventories/
recipes match;2048positive projector/DiT gradient records per arm and frozen
tensors verified. Same-input/noise96TRAIN expert windows plus12historical
student draws and actual event exposure independently CPU-audited;no new
model queries. These probes are not task/generalization evidence or the
new current-eight-anchor NI draws. CORR TRAIN4 is now audited;DEV8 still executing;
matrix19qualified/32fixed,spent midpointF1/T1gap retained,no final success
rate/causalDAgger gain/selected model claimed. Same fifth673008/source
e418bb79 remainsR,original4h/no sixth.84e7cee1 actualCI39066444935 SUCCESS,
8blobs exact readback;its nativeN1 transport/endurance still NOT_RUN.

**N1 heartbeat capture implemented and CPU-tested; native qualification incomplete.**
[Endurance/fault-injection ledger, including explicit missing native evidence](reports/PI0294/20261016T191445Z-localpc-mp1/ENGINEERING_ENDURANCE_AND_FAULT_INJECTION.json).
Future N1 premodel entry now captures at least600real monotonic seconds/
81post-read samples with native job/request/host checks and raw immutable
journal joins. Same-job model-ready/terminal,stale/future/wrongrun,missing/
tampered raw evidence and failed capture restart are rejected.473PI0294
tests/365subtests pass locally;not native evidence. Full80cycle transport
producer/router and isolated native faults remain pending;gate stays closed
without all proofs. No deployment/newPBS/hot switch;old673008 continuesR,
CORR1944/2048 at frozen readback,not endpoint completion. All12learning/
15numerical protected source hashes still match e418bb79.

**Latest live readback: fifth science job continues; audit reader03 is terminal.**
[Native progress and explicit correction of the reader's earlier live snapshot](reports/PI0294/20261016T191445Z-localpc-mp1/N1_READONLY_READER_TERMINAL_AND_SOURCE05_PROGRESS.json).
Same673008 remainsR,CORR1828/2048 with one legitimate in-flight update;
five submissions,original4h request,no sixth or controller/model restart.
Reader03 stopped on an unclassified read-only SSH process SIGPIPE(-13),
not a policy failure;zero new audit intents,540CPU upper settled,reserve0.
Underlying pipe cause is not established and no replacement was launched.
A subsequent independent Core read succeeded and preserved the already
audited F1/T2 negative;it is not endurance proof. F1/T1 remains spent.
The reader-live paragraph below is a historical snapshot,not current state.
CORR endpoint,paired evaluation and nativeN1 endurance remain pending.

**Latest actual control repair:read-only serial audit pipeline restored.**
[ESTALE terminal,prestart import rejection and verified replacement reader](reports/PI0294/20261016T191445Z-localpc-mp1/N1_READONLY_SERIAL_AUDITOR_ESTALE_RECOVERY.json).
Original reader01 exited on actualNFS Errno116;its completed F1/T2 audit and
540CPU upper remain settled. Reader02 import-path guard rejected before
START/new audit;private original bytes retained,360CPU upper settled. One
reader03 is now PID+start verified live,using isolated immutable metadata/
retry imports from2cf7bf1e;no second science owner. NativeR read after START
shows CORR1605updates,not2048. Only remaining12fixed endpoint audits allowed,
old helper deadline unchanged. Classified metadata only gets3attempts/120s
including blocked read;no heartbeat writes or action/query/install/start/
qsub retries.463CPU tests/343subtests and actualCI39064129714 PASS;two source
blobs exact readback. This is not N1native600s/80cycle qualification. Science
job/capsulee418bb79 unmodified,no newPBS/model/physics;540reader CPU reserve
stays held until its terminal. Full endpoint/paired science still pending.

**Latest engineering:N1 runtime layers integrated in source,not native-admitted.**
[Generic cold admission,fixed-probe resume and fresh site readbacks](reports/PI0294/20261016T191445Z-localpc-mp1/N1_GENERAL_RUNTIME_ADMISSION_CPU_AND_SITE_READBACK.json).
One contiguous06..12cold format retains all original native costs/outcomes/
STOPs;request/queue/CPU/counters/session/Core/stdio/registry/matrix consume
the sealed first N1 authority. Unbound06/13,wrongrun,budget reset,live prior
consumer,unknown effect,duplicate owner/settlement and valid-result replay
are rejected. Zero-work panels skip before load only with complete fixed
probes;known-unstarted draws resume through unchanged E1 primitive,unknown
intents never resample. Fresh synthetic108array+aggregate parity with the
original is exact.457localPI0294 tests/328subtests PASS;not native evidence.
Hopper masked authorized auth refreshed;active project/member/positive
balance and fresh personal quota read back. Nativeauto->small singleGPU
wall cap supports future8h;current fifth stays4h and was not altered.
Actualfifth673008R,CORR1345updates at report snapshot,not2048. Runningcode
e418bb79 and all12learning/15numerical files remain unchanged. General
finite manager and actual600s/80cycle qualification producer are still
pending;no cold deployment/newPBS. New N1 model gate fails closed without
same-job native endurance/transport/fault proof. Old initial smoke is not
substituted. OriginalN1 start/cutoffs/MP1 T0 and selected/deploymentNONE persist.

**Latest authority:N1 adopted once;healthy fifth job stays on its original binding.**
[Actual adoption,three-site control readbacks and cumulative limits](reports/PI0294/20261016T191445Z-localpc-mp1/N1_ADOPTION_AND_EFFECTIVE_LIMITS.json).
Fixed926f9ef6 complete bundle read back;N1 began2026-10-18 03:08:01SGT.
New submissions/candidates stopOct19 09:08:01,experiments13:08:01,
hardclose15:08:01SGT,all before originalMP1 cutoffs. OriginalT0/costs/cards,
STOPs and fifth673008 remain unchanged. At adoption source24119settled+
14400reservedH200seconds,parent28619+14400;sourcePBS5,parentHopper8.
New authorized cumulative ceilings:H20080h/A4048h,source36h/12PBS,
not a fresh pool. Current fifth still has its old4h/no-sixth binding;future
N1 cold consumers are NOT deployed/admitted. General resolver foundation
has6CPU tests/15subtests;600scompute endurance/80transport cycles NOT_RUN.
Fresh Hopper project query failed,new-submission lane not yet admitted;
healthy current training continues,not cancelled/requeued. CORR already
beyond512,endpoint2048/full paired science still pending. No extraPBS,
model,physics or deletion attributable to adoption. Selected/deploymentNONE.

**Latest CPU evidence:CORR512 true event exposure and reused supervision audited.**
[16384actual visits,raw events and exact executed-label membership](reports/PI0294/20261016T191445Z-localpc-mp1/CORR512_ACTUAL_EVENT_EXPOSURE_AND_SUPERVISION_REUSE.json).
CORR512T1/T2=8192each;4096C1visits split2062student anchors/2034subsequent
expert states,3115oldE0aggregation/981actualE1intervention. All4qualified
historicalT1diagnostic expert-label windows exist in C1 and were consumed2060
times. This is exact executed-supervision reuse,not heldout generalization;
entire GR00T processor input tensor identity was not compared. T2label C1matches0.
Retained originalT1close/open/withdraw prefix-uncovered counts52/49/57 versus
BC51227/26/36 reflect half originalT1visits,not a demonstrated sampler bug.
All30T2episode event/visit summaries exactly match BC512. C1close7/open4 are
segment events,not independent demonstrations;4close events have only later
H40coverage. C1detach/withdraw/stages remain UNKNOWN_NOT_ZERO,no imputation.
All96saved CORR expert draws now join actual raw-event strata,not newNN/S1.
Two new CPU workers terminal,17.5037metered processing seconds,one shared180s
local upper charge,all1800Hopper/Core CPU reserves cleared.426tests/285subtests
and actualCI39059634999 PASS;five code/test/card blobs verified. Runtimee418bb79
unchanged,fifth673008 stays healthy/formal training;no newquery/physics/PBS/
teacher label/training update attributable to this audit. Full pair pending.

**Latest:CORR formal training actually resumed;F1/T2 replacement audited negative.**
[Actual12s originalS1 negative,31draw chain and resumed533step snapshot](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE05_CORR512_F1_T2_AUDITED_AND_EXACT_TRAINING_RESUME.json).
Fifth673008 remainsR,originalCORR512fullstate31f0514b resumed optimizer/
scheduler/RNG/sampler/stats,actual533updates/cursor17056 at fixed snapshot.
Not initialized from BC,no recipe/data/S1/WBC change;formal stop remains2048.
F1/T2 sole replacement12s/6000ticks/241pairs,19raw checks and31new observations/
draws,30ACKs+1zero-live rejected prefix PASS,native faults0. This is VALID_TASK_FAIL,
not another engineering missing result,no further retry. Original8.4s partial
retained. Matrix22attempts/19qualified cells/1exhaustedF1T1engineering cell/
12unattempted finalCORR cells,all16public links retained. CORR512 has3valid
negatives/1engineering gap,not0of4validresults. Formal training proceeds under
original authorization without adding a success gate. SourcePBS5/5,no sixth
under current binding;originalT0/caps/closeout preserved,full pair incomplete.
**Probe caveat now resolved:**both historicalT1probe parents occur in actual
C1 training consumption,3115of4096correction visits at512;T2parent overlap0.
The observedT1error gain is on training-parent lineages,not heldout evidence.
Parent overlap alone does not prove exact anchor/H40window overlap. New N1
documents were merged/preserved without force;expanded limits not adopted or
used by this healthy old-bound job. TF-R3STOP,VandaCLOSED_NOT_RUN,selectedNONE.

**Latest:673008 actually R;new saved-label joins complete,not endpoint results.**
[Native pre-model post-read qualification and real-prefix snapshot](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE05_NATIVE_ADMISSION_AND_REAL_PREFIXES_IN_PROGRESS.json).
Forced compute/server heartbeat read PASS with post-read clocks and unchanged
0..300s gate;original CORR512F1/T2 sole replacement is actually executing.
Initial qualification/prefix ACKs are not a completed condition or proof of
sustained transport. SourcePBS5/5,no sixth;CORR2048 still NOT_RUN at snapshot.
[Three-model actual historical student-label comparison](reports/PI0294/20261016T191445Z-localpc-mp1/THREE_MODEL_HISTORICAL_STUDENT_LABEL_COMPARISON.json).
Same12historical states/4parents,8qualified H40 labels/4invalid initial labels
excluded;no new NN/physics/teacher labels. Same512 T1 prefix XYZ macro mean
BC6.954->CORR2.502mm;T2 XYZ12.550->11.793mm but gripper absolute error
0.2675->0.2726. This is not uniform improvement,full recovery or independent
generalization:only2TRAIN parents/task,C1 overlap not independently resolved.
BC2048 T1/T2 XYZ6.346/12.591mm is a different-step learning-curve description.
All actual ACKed labels,input/noise/model identities and16private HF members
per new model checked on Core;no copied BC512 error labels. Specific S1 and
collision margins remain NOT_RUN/NOT_MEASURED. Four new CPU workers terminal,
8.0704actual CPU seconds settled,all their2400CPU reserve cleared;no model
weights downloaded. Audit codef29f748a passes418tests/276subtests/CI39058058261;
scientific runtimee418bb79 unchanged. Full matched pair/mechanism pending.

**Latest:human-authorized fifth native job673008 submitted once,Q,not resumed.**
[Actual two-site cold adoption and single native submission](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE05_ACTUAL_COLD_AND_SINGLE_NATIVE_SUBMISSION.json).
Exact runtimee418bb79,416tests/274subtests/actualCI39057571128 PASS;ten changed
code/test blobs verified. Both native entry checks and immutable560-member
capsules PASS;15frozen numerical/12learning-data-plant modules unchanged.
Both original STOPs,all prior capsules/manifests/outcomes and original birth
PBS4 binding retained. Only exact sealed human card permits effectivePBS5;
source5/5,parent8/12,no automatic sixth. CFP04-CF-007/auto/1GPU/max4h,
same finite owner live,source24119s/parent28619s settled plus14400s reserved.
Queue is not model readiness/training/native sustained heartbeat qualification.
CORR512/cursor16384 unchanged;remaining1536updates,original F1/T2 sole
engineering replacement and final2048 panels only. F1/T1 no third attempt,
valid negatives not repeated,T0/caps/45min closeout/final4h reserve unchanged.
Full matched scientific outcome remains incomplete,selected/deploymentNONE.

**Latest native state:source04 is terminal F/1;one fifth job directly authorized,not submitted.**
[Actual partial closure and clock-race evidence](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE04_ACTUAL_TERMINAL_AND_CLOCK_REPAIR.json).
Native672907 used381wall/391conservative seconds;source24119H200seconds,
parent28619,reservations0,updates2560,CORR512/cursor16384 unchanged.
F1/T2 is engineering-incomplete8.4s,not a policy negative:21executed prefixes,
22native draws,one generated-but-uninstalled action;all1970mutation entries
closed.18raw checks/169decoded paired frames PASS,native faults0. Both STOPs
retained. Observed negative67.9ms age reproduces a pre-I/O clock race;07b04707
samples after each read,keeps0..300s unchanged,406tests/CI pass,not yet native
qualified. [Direct human bounded fifth-slot authority](reports/PI0294/design/SOURCE_PAIR_ADDITIONAL_PBS05_AUTHORIZATION_20261018.json)
permits only one extra<=4h job within unchanged12H200hours/T0/MP1 caps;
1536updates remain,F1/T2 original sole replacement,F1/T1 no third. No sixth.
Cold binding required before dispatch. Matrix21attempts/18qualified cells/
2engineering cells/12unattempted;16existing public links retained. Prior Q
snapshots below are history. TF-R3STOP,VandaCLOSED_NOT_RUN,selectedNONE.

**Latest video delivery:four actual CORR512 attempts are public,not a complete panel.**
[Two valid negatives,two engineering partials and fixed-revision receipts](reports/PI0294/20261016T191445Z-localpc-mp1/S_CORR512_FIRST4_AUDITED_VIDEOS_PUBLIC.json).
READY9ba0fe42,1132real paired frames,no padding/resimulation. Core full-byte
readback and Hopper token-free READY/index checks PASS;no movies/weights fetched
to Hopper. Four attempts cover only3conditions,2qualified negatives;F1/T1 has
two spent engineering attempts,F1/T2 still NOT_RUN. Matrix adds4exact URLs,
preserves12BC links and every non-URL scientific field.16public clips total
are not16successes. Publisher9cf27435 passes402tests/259subtests/CI39055006711,
but original video execution03646938/current source04 runtime8e50bcc2 stay
distinct. Delivery/readback terminal,reservations0,no new NN/physics/PBS/update.

**Latest saved-evidence audit:BC512/CORR512 fixed expert pair verified.**
[Actual96-window pair and24additional student-output identity audits](reports/PI0294/20261016T191445Z-localpc-mp1/SAVED_BC512_CORR512_PROBE_COMPARISON_AND_STUDENT_IDENTITIES.json).
Same real inputs/noise/precision,96windows from8true TRAIN episodes,not126.
T1prefix XYZ mean2.703->2.817mm,T2 2.986->3.010mm;no descriptive mean XYZ
improvement or causal correction benefit established. Per-episode,stage and
tail summaries retained;macro p95 is not global pooled p95. BC2048/CORR512
each12 historical student draws now identity-audited,36outputs across3models
but only12same states/4parents. All trajectory gates ACCEPT;these are not S1
collision passes or recoveries. Existing8qualified BC512 labels are not new
models' errors;new-model same-state label join remains pending. Three auditors
terminal,4.821Hopper CPU seconds settled,reservations0,no newNN/physics/PBS.
Source04 still Q at last actual read;full paired endpoint/mechanism incomplete.

**Latest:allocation04 submitted once,native672907 is Q,not resumed.**
[Actual single submission,capacity,controller and queue receipt](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE04_SINGLE_NATIVE_SUBMISSION_AND_QUEUE.json).
CFP04-CF-007/auto/1GPU/12CPU/128GiB/max4h;Q comment is GPU unavailable.
SourcePBS4/4,parent7/12,no fifth source submission. Source23728s/parent28228s
settled plus14400s reserved for the same job;45min allocation closeout and
parent final4h remain protected. Same finite owner is live,heartbeat fresh.
Original140GiB output estimate suffices;no new180GiB rebinding or deletion.
PBS_READY/release/native compute heartbeat qualification are NOT_RUN. Old
MODEL_STATE allocation03 is stale,not current readiness. CORR stays512/cursor
16384;13unattempted conditions pending,spentF1/T1 not replayed. Queue is not
training or task evidence. Original budget/T0/S1/WBC/selectedNONE unchanged.

**Previous boundary:cold allocation04 adopted on both sites before submission.**
[Actual cold pointer/proof readbacks and retained original budgets](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE04_ACTUAL_COLD_BOUNDARY_ADOPTION.json).
Code8e50bcc2 passes400pytest tests/254subtests and actualCI39052726122.
Both sites staged immutable558-member capsules and qualified approved entry
checks before advancing their same-root current manifest pointers. Original
manifests and STOP713fef5c are sealed and retained,old scopes still blocked.
Fifteen numerical and twelve learning/data/plant modules remain byte-identical;
no new budget,T0,attempt right,model query,physics,training orPBS. Source23728s,
parent28228s and sourcePBS3/4 unchanged. CORR512/cursor16384 still not resumed.
Native compute heartbeat/NFS qualification is NOT_RUN;CPU checks are not that
qualification. Original300s gate unchanged,last original slot04 only. No third
F1/T1 engineering attempt,no valid-negative replay. Failed-payload inventory
still yields actual deletion0/freed0;required parents and fixed states retained.

**Previous terminal:allocation03 is F/1;partial effects closed,old engineering STOP retained.**
[Actual terminal,26 real ACKs,orphan closure and heartbeat uncertainty](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE03_TERMINAL_PARTIAL_EFFECT_CLOSURE_AND_HEARTBEAT_DIAGNOSIS.json).
Native672829 consumed409wall/412conservativeH200seconds,26draws and0updates.
Source23728s,parent28228s settled,reservations0;finite owner actually completed
settlement and exited. Original CORR512/cursor16384 is intact,not continued.
F1/T1 sole replacement is ENGINEERING_INCOMPLETE10.4s/5200ticks,not a policy
negative or success;no third attempt. Both original22.4s and replacement10.4s
raw audits pass18checks/full449+209paired frames,native faults0. Independent
partial closure joins26native draws/26ACKs/54camera hashes;sequence26orphan
has no draw,response,command or physics. All1926mutation intents COMPLETE.
Old STOP713fef5c remains on both sites;no fourthPBS submitted. Compute guard's
actual heartbeat value was not recorded,NFS visibility is a hypothesis,not
proven root cause. New helper8f689fe3 passes382CPU tests but is NOT deployed
or compute-NFS qualified;original300s gate unchanged. Cold qualification and
cumulative-family binding are required,not a blind retry of the stopped branch.
Current matrix:20attempts,18verified policy cells,1engineering cell with both
attempt rights spent,13unattempted. Original caps/T0/safety/selectedNONE persist.

**Previous queue milestone:continuation03 was submitted/Q before its run.**
[Native672829 queue,373 tests/CI and cold management qualification](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE_CONTINUATION03_ACTUAL_QUEUE_AND_ENGINEERING_QUALIFICATION.json).
Same owner,CFP04-CF-007/auto/1GPU/max4h. SourcePBS3/4,parent6/12;
source23316s/parent27816s settled plus14400s reserved,original clocks unchanged.
Scientific launch stays03646938;management1b1806d9 is separately sealed and
both entry validations PASS. Original manifest unchanged;full CORR512/cursor
16384 preserved for exact continuation,not claimed to have resumed while Q.
Output estimate140->180GiB is evidenced inside unchanged parent400GiB cap,
not a new storage/science budget. Bounded idempotent metadata-sync recovery
does not resend action/query/start/qsub or relax heartbeat/safety watchdogs.
Native queued memory225->128GB decrease read back,no cancellation/resubmission.
[Updated32-condition matrix](reports/PI0294/20261016T191445Z-localpc-mp1/BEHAVIOR_AND_FAILURE_MATRIX.json)
now preserves19actual attempts,18verified policy conditions,1engineering gap
and13unattempted cells;all12 BC endpoint public links remain. CORR F1/T1 has
only its one original-contract replacement pending;two validF0 negatives stay.
Finish remaining fixed CORR panels and terminal exposure before mechanisms.

**Previous native boundary:allocation02 is F/1;full CORR512 state is preserved.**
[Native terminal,error chain and exact resource settlement](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE_ALLOCATION02_NATIVE_TERMINAL_AND_PRESERVED_RESUME.json).
A transient CoreLab SSH loss interrupted the finite local controller;Hopper
then enforced its unchanged owner-heartbeat watchdog. Connectivity recovered,
all original physical handles are terminal,reservations0. Source H20023316s
and parent27816s are actually settled;sourcePBS2/4,updates2560=BC2048+CORR512.
CORR F0/T1 andF0/T2 are audited valid negatives and will not be replayed;
F1/T1 is an engineering-incomplete22.4s attempt,not a policy negative;F1/T2
and finalCORR2048 remain pending. No newPBS or safety/recipe/clock reset.
[Direct-user failed-checkpoint retirement check](reports/PI0294/20261016T191445Z-localpc-mp1/FAILED_CHECKPOINT_RETIREMENT_CHECK_01.json)
found no eligible large failed payload in three registered storage areas:
actual deletion0,freed0. Required parents/fixed checkpoints/resume state remain.
The following learning milestones are preserved history,not a current-running
job claim. Full pair/final deliverables remain incomplete,selectedNONE.

**CORR512 is sealed and actual paired consumption passes;the fixed panel is in progress.**
[Actual CORR512 learning,matched T2 cursors and first valid negative](reports/PI0294/20261016T191445Z-localpc-mp1/S_CORR512_ACTUAL_MATCHED_CONSUMPTION_AND_FIRST_VALID_NEGATIVE.json).
Same originalM1 initialization and actual recipes verified;512 real updates,
537 changed trainable tensors,frozen tensors unchanged,full state31f0514b.
Each arm consumed16384 visits,T1/T2=8192each. All8192 T2cursors and4096 retained
originalT1cursors match exactly;CORR actually consumes4096 T1correction visits
from4student parents/8segments,not4096 independent demonstrations.2062visits
are student-anchor states,2034 are subsequent expert states. FirstCORRF0/T1
is originalS1 refusal at1.6s:19raw checks,33decoded pairs,5native draws,4ACKs
and1zero-live rejection independently PASS,native faults0. No negative rerun.
352 CPU tests/CI39043547480 PASS;read-only consumption auditor terminal1.397
CPU seconds,reservation0. No actual2048CORR,full panel rate or correction gain
claimed. Native672471/same owner continue;original03646938 runtime unchanged.

**S-BC2048 endpoint is complete:TRAIN3/4 and exposedDEV3/8.**
[Complete DEV8,actual original-M1 initialization and remaining work](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC2048_COMPLETE_DEV8_AND_CORR_INDEPENDENT_START.json).
[All32 fixed conditions,17 verified and15 pending](reports/PI0294/20261016T191445Z-localpc-mp1/BEHAVIOR_AND_FAILURE_MATRIX.json).
DEV T1=2/4,T2=1/4 across four exposed layout groups,not hidden confirmation.
Four failures are originalS1 pre-execution refusals;one T2 safely completes32s
with real pushing but misses the goal-zone hold. Zero actual native faults or
engineering gaps,TeacherOFF.148 raw checks/3360 decoded pairs/423 real draws/
419 executedACKs+4 zero-live rejections PASS.
[All eight actual DEV videos are now public](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC2048_ALL_DEV8_VIDEOS_PUBLIC.json)
at fixed READY448ad9de;Core full-byte and independent Hopper token-free
READY/index readbacks PASS. All12 BC endpoint clips,successes and negatives,
are public;no earlier TRAIN prefix or original task outcome was changed.
The earlier77update snapshot is superseded by the actual CORR512 proof above.
Metadata-only32-cell aggregator85c4a013 does not change scientific runtime.
No newPBS or clock/budget reset;the current running allocation is not settled.
[Actual serial audit harvesting and measured closed-TRAIN transport](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE_ALLOCATION02_SERIAL_AUDIT_AND_TRANSPORT_MEASUREMENTS.json)
reuse real receipts;the RPC residual is not measured GPU idle/inference time.
Complete independentCORR and matched exposure before mechanism/candidate
selection. Final four review deliverables remain incomplete;TF-R3STOP,
VandaCLOSED_NOT_RUN and selected/deploymentNONE remain unchanged.

[BC2048 final raw-event audit is complete](reports/PI0294/20261016T191445Z-localpc-mp1/BC2048_FINAL_RAW_EVENT_EXPOSURE.json):
all65536 actual visits join126 hash-verified raw timelines. T1 sampled prefix8
covers96/96 close,95/96 open,95/96 withdraw events;all events occur in sampled
H40 supervision. T2 detach/withdraw both30/30,so wholesale missing event
supervision does not explain its1/4 DEV result. Coverage is not action accuracy
or safety margin. No sampler change/new query/physics;auditor terminal20.477
Hopper CPU seconds,reservations0. CORR comparison and mechanism choice remain.

## Previous Complete TRAIN4 And Video Milestone

**S-BC2048 TRAIN4 is complete:3 successes,1 valid original-S1 failure,no gaps.**
[Complete panel,all four public videos and eight original anchor noise joins](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC2048_COMPLETE_TRAIN4_AND_PUBLIC_VIDEOS.json).
T2 succeeds on both seen TRAIN layouts at32s/41-of-41 hold;F0/T1 also succeeds,
butF1/T1 is rejected at1.2s before unsafe execution. Zero actual native faults,
TeacherOFF. All73 raw checks/1948 decoded paired frames/244 native generations
and243 executed ACKs+1 zero-live rejection independently PASS. Two seen layouts,
not generalization. All four original videos,including negative,are actually
HF-public at READY53d71ad9;Core full-byte hashes and Hopper unauthenticated
READY/index readback PASS. New334-test publisher/CI PASS,scientific runtime
03646938 unchanged. Eight fixed512 action receipts match actual input/cameras/
weights/precision/noise-hook fingerprints;no new draw,raw noise tensor not saved.
Reuse those eight original outputs;future missing endpoint draws still await
both sealed endpoints and their own original-MP1 NI/shadow binding.
Native672471/same finite owner now runs exposed DEV8;independentCORR remains.
Original budget/T0/closeout/WBC/S1/TF-R3STOP remain,selected/deploymentNONE.

## Previous FirstT2 Success And Static Restore Milestone

**The fixed S-BC2048 model now also has an independently audited T2 success.**
[Actual F0 push success,F1 early valid negative and eight static anchor restores](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC2048_FIRST_T2_SUCCESS_AND_ZERO_STEP_ANCHOR_RESTORE.json).
F0 T1 and T2 each complete32s/641 camera pairs/41-of-41 hold with noTeacher,
zero originalS1 rejects/native faults. T2 actually pushes15.685cm with3903
robot-contact ticks,no bilateral grasp;80 real fresh generation/ACK chains pass.
F1/T1 instead has an independently audited validS1 rejection at1.2s,zero native
faults;no negative rerun or two-layout/generalization claim. F1/T2 is still active
at this snapshot;DEV8 and independentCORR fromM1 remain pending.
Eight fixed512 full live/IK/controller/history snapshots actually restore with
max difference1.78e-15,zero integrations/queries. Static-only,not dynamic-prefix
qualification.182.450CPU seconds and8 conservative starts charged once,reserves0.
Initial CPU-entry interpreter import failure preserved and same-bound existing
runtime repair passed.330 CPU tests/CI pass;scientific runtime03646938 unchanged.
Project auth is currently valid;native672471/finite owner healthy,no newPBS.
Original MP1 budget/T0,TF-R3STOP,selected/deploymentNONE remain unchanged.

## Previous First2048 Success Milestone

**S-BC2048 is sealed and its first F0/T1 is an independently audited success.**
[Complete training,actual65536 visits and first32s autonomous success](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC2048_TRAINING_AND_FIRST_AUTONOMOUS_SUCCESS.json).
Checkpointac0f4692 has full optimizer/scheduler/RNG/sampler state;three completed
partitions verify65536 visits,T1/T2=32768 each,126 original TRAIN IDs,C10.
Final417 resumed updates have537 changed trainable tensors,493 frozen tensors
unchanged,positive finite gradients and continuing scheduler through2048.
F0/T1 actually completes32s/641 camera pairs/41-of-41 hold,lift8.136cm,
carry16.978cm,zero S1/native faults and noTeacher.18 raw checks,full video
decode and80 fresh model-generation/ACK chains independently PASS.
This is ONE seen TRAIN scene,not fullTRAIN4/DEV/T2/CORR or target success.
Strict512-vs2048 saved probe inputs/noise/weights match;prefix8 XYZ means
T1/T2 improve2.703/2.986mm to2.221/2.506mm on8 expert episodes,not causal proof.
Native672471 and same finite owner remain active;remaining fixed panels and
independent CORR fromM1 continue,no new budget/clock/recipe or selected model.
The four valid512 negatives and current B1 geometry evidence remain unchanged.

## Earlier Current512 Failure Attribution

**B1 current-failure histories and first illegal geometry are actually verified.**
[Four current512 failures,eight full-history anchors and terminal resources](reports/PI0294/20261016T191445Z-localpc-mp1/CURRENT_SBC512_FAILURE_HISTORY_FIRST_GEOMETRY.json).
All27600 original live highrate rows and wire/installed references reproduce
with zero difference. Original rejected-prefix offsets64/124/63/150 also match.
F0T1/F0T2/F1T1 first contact is right_pad/table;F1T2 is left_pad/table.
These are diagnostic-shadow predictions,not original live collisions;native
faults remain0. Frozen ARM_TABLE_OR_FLOOR rule stays,not replaced by4mm margin.
Eight noninitial current S-BC-induced states have full live/IK integration and
controller/Python/teacher histories saved and hash-read back. Original hidden
RNG identity was not recorded and is not invented;independent saved-snapshot
restore and actual cross-host noise receipt joins await future model queries.
Diagnostic child CLOSED:4 charged starts,28001 new shadow ticks,194.195 charged
CPU seconds including180 control upper;zero live/NN/PBS/GPU,all reservations0.
326 CPU tests/native CI passed. Healthy source job672471 continues unchanged;
fixed pair endpoints/panels and causal channel attribution remain incomplete.

## Earlier Source Resume And Event Snapshot

**Source continuation02 is actually running and exact1631 resume has updates.**
[Raw event exposure,per-episode coverage and native resume](reports/PI0294/20261016T191445Z-localpc-mp1/RAW_EVENT_EXPOSURE_AND_ACTUAL_SOURCE_RESUME.json).
672471 is R,observed1743/cursor55776;first resumed microbatch52192,
gradient1631 and optimizer1632 match the complete1631 state and continuing LR.
No512 valid condition rerun;fixed2048/CORR/C1/DEV remain pending.
B2 independently joins actual512/1631 traces to126 raw installed-command
timelines and teacher events,not coarse manifest phases. At512,T1 close/open/
withdraw events outside all sampled first8:27/96,26/96,36/96;all nevertheless
occur somewhere in fullH40 supervision. At1631 those first8 gaps are1/1/2.
T2 detach first8 coverage30/30,withdraw29/30 at512;supervision not absent.
Four expert open-event windows have first8 gripper error0.21353 despite2.32mm
XYZ. Small expert-state sample,not a unique causal explanation of live failure.
All96 saved probe errors stratified by actual raw stage/event and true episode.
No sampler intervention chosen;fixed pair and B1 first-geom work remain first.
Audit20.136 CPU seconds,terminal/reservation0,new inference/physics0.
Numeric export640 commands/601 window states was checked against641 original
observations;auditor-only repair preserved the original1200s deadline/budget.
320 CPU tests/native CI success;runtime03646938/WBC/safety remain frozen.

## Previous Continuation Submission Snapshot (Before GPU Allocation)

**Authentication restored; Source Pair continuation02 was submitted/Q.**
[Same-state student label audit and native continuation](reports/PI0294/20261016T191445Z-localpc-mp1/B0_SAME_STATE_LABEL_AUDIT_AND_SOURCE_CONTINUATION02.json).
672471 is queued for one H200/max4h under CFP04-CF-007,not yet loaded or
training. Original full1631/cursor52192 state will resume; fixed2048 remains.
Source2/4 PBS,MP1 allocated16295s/reserved14400s; no clock/budget reset.
Finite same-owner controller handles this one allocation and incremental
settlement only. Native queued memory decrease225GB to128GB read back.
B0 now actually joins all12 existing historical student draws:8 qualified
noninitial expert H40 labels,4 initial exclusions retained,two parents/task.
Prefix8 episode-macro XYZ means T1/T2=6.95/12.55mm,not RMSE or task success.
Actual ACK commands,exact history and matching first dual-camera pixels PASS.
S-BC-specific S1/clearance remain NOT_RUN/NOT_MEASURED; no older-model safety
borrowed,no new queries/physics. Only16 small private HF files transferred.
311 CPU tests/CI pass; scientific runtime03646938 and frozen WBC unchanged.
B1/B2 and fixed pair endpoints/TRAIN/DEV remain incomplete;target STOP stays.

## Previous Native Boundary (Superseded By Continuation02 Above)

**Newest review adopted; first Source Pair allocation is terminal with a1631 partial.**
[Review adoption, real multi-session consumption and12 historical student draws](reports/PI0294/20261016T191445Z-localpc-mp1/REVIEW_ADOPTION_STUDENT_PROBES_AND_ALLOCATION01_PARTIAL.json).
672080 isF/0,not running:11795 conservative H200 seconds,zero GPU reservations.
Full checkpoint1631/cursor52192 is saved; independent two-partition readback
verifies26096 visits per task,126 original TRAIN IDs,correction0. No2048 claim.
Both child leases stay ACTIVE for legal continuation; physical workers/owner
and helpers are terminal. Current project query15006 requires fresh auth
before the next original-budget allocation; no new PBS submitted.
The12 E0 student-state draw input/noise/weight identities passed independently;
qualified-label RMSE and S-BC-specific original-S1 comparison remain pending.
Follow reviewfb16d101: finish the fixed pair,then evidence-directed one-variable
intervention. No extra budget/clock,WBC/S1 change or replay of valid negatives.

## Earlier Source Pair Snapshots (Not Current Native State)

**Current source milestone: S-BC512 TRAIN4 is complete and exact resume is running.**
[All four actual conditions, independent chains and resume engineering](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC512_FULL_TRAIN4_AND_RESUME_AUDIT_ENGINEERING.json).
This intermediate model has0/4 task successes: four valid original-S1
rejections at1.6/20.8/10.4/22.4s,zero native safety faults or engineering gaps.
All four raw audits,1108-frame video decodes and142 fresh cross-host chains
passed. No valid negative is repeated or replaced by an old model's success.
Native full-state resume has actually reached515/cursor16480,starting at
cursor16384,under the same job672080 and runtime03646938. Original fixed2048
criterion remains; no new success-rate gate,recipe change,CORR/C1 or DEV claim.
The [complete512 consumption receipt](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC512_CHECKPOINT_CONSUMPTION_AND_FIRST_VALID_NEGATIVES.json)
verifies16384 balanced original visits across126 TRAIN IDs. Read-only resume
partition auditor d2c05a19 passed289 CPU tests and real single-partition readback;
actual multi-session qualification remains pending. Vanda camera stays CLOSED
NOT_RUN. Human engineering autonomy adds no budget or clock; TF-R3 STOP stays.

**Saved expert-state probes are now independently verified:**
[96 windows / eight true episodes and parent-provenance limit](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC512_SAVED_PROBE_AUDIT_AND_PARENT_PROVENANCE_GAP.json).
S-BC512 prefix8 XYZ means are2.70/2.99mm forT1/T2,not a replacement for
its0/4 live successes. Actual inputs/noise/weights and all metrics/gates match;
no new inference. Strict M1 comparison remains NOT_PROVEN because its first
legacy array lacks input fingerprints; no metadata is invented to fill them.
New auditor297 CPU tests/CI passed. Same-job training is now observed668,
cursor21376,not terminal2048. Original scientific capsule remains unchanged.

## Earlier MP1 Snapshots (Not Current Native State)

**Source learning is now actually running; Vanda camera queue is closed NOT_RUN:**
[native admission, real optimizer/sample-prefix audit and camera terminal receipt](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE_LEARNING_STARTED_AND_VANDA_CAMERA_QUEUE_CLOSED.json).
672080 entered one H200 under CFP04-CF-007; S-BC has real updates, with the
first27 committed updates consuming864 TRAIN samples,432 each task and finite
positive projector/DiT gradients. This is not a new task-result claim.
1463728 never started before its original deadline; after an exact-owner queued
node_pool correction still left it Q,only that expired queue was withdrawn.
NativeF/zero allocation and CLOSED lease/zero Vanda reservations are verified.
No camera/EGL/WBC PASS,deadline extension or second camera submission is implied.

**All ten actual recovery videos are now public at a fixed HF revision:**
[delivery and exact queue resource delta](reports/PI0294/20261016T191445Z-localpc-mp1/PUBLIC_RECOVERY_VIDEOS_AND_QUEUE_MEMORY_DELTA.json).
Core verified every published hash; Hopper independently read READY/index.
Ten attempts include three successes,two valid negatives,five engineering
interruptions,including two initial-frame-only clips; no missing result filled.
The same queued source job672080 now requests128GB,not the original225GB:
one native owner-only memory decrease,zero newPBS or recipe changes. It is
still Q for GPU capacity,not a claimed training result. Runtime03646938 is pinned.

**Matched Source Pair is now submitted, not trained yet:** [native first job672080](reports/PI0294/20261016T191445Z-localpc-mp1/MATCHED_SOURCE_PAIR_FIRST_SUBMISSION.json)
is Q under CFP04-CF-007/auto, one H200 and at most4h. Both native immutable
entries and fresh project/output quota passed. Runtime03646938 uses the
fixed M1@1024 parent, TRAIN126 and qualified C1, matched T1/T2 exposure,
new optimizers,2x2048 planned updates and32 fixed development conditions.
Original MP1 budget/T0 remains; no fourth old-M2 recovery or new owner.
Actual training consumption and behavior are pending, not inferred from code.

**M2 recovery is CLOSED:** [actual eight-condition panel and terminal resources](reports/PI0294/20261016T191445Z-localpc-mp1/RECOVERY_FINAL_ACTUAL_PANEL_AND_RESOURCES.json).
CORR T1 succeeds in both seen TRAIN scenes; BC T1 succeeds in F1 while F0
remains engineering-missing. Overall: 3 SUCCESS, 2 valid original-S1 failures,
2 engineering-missing and 1 NOT_RUN. All ten family attempts audited; three
complete successes have 240 independently matched fresh observation/ACK chains.
671899 is F/0, 4,047 conservative H200 seconds; family total 4,500 seconds.
Both child03 leases and all recovery workers/reservations are closed, including
the separate metadata reader. No fourth recovery PBS or new training.
Old C0/exposure confounding remains; no causal DAgger gain or T2/DEV/WBC success.
Vanda camera job1463728 remains Q in the last native read; its separate
reservation is not released. Original MP1 clock continues.

## Historical MP1 Progress Snapshots (Not Current Native State)

**CORR also has an independently verified full T1:** [original M2-CORR512 F0/T1 SUCCESS](reports/PI0294/20261016T191445Z-localpc-mp1/M2_CORR_FIRST_COMPLETE_T1_AND_RAW_AUDIT.json),
32s/641 camera pairs/41-of-41 hold/zero S1 or native faults.18 raw checks,
full video decode and80 actual cross-host ACK chains passed. This is C0-only
seen TRAIN, not freshC1 learning or causal correction gain; BC's verified
success is F1,so these two rows alone are not a paired comparison. Remaining
CORR conditions continue in671899; original budgets and all safety STOPs stay.

**BC session actual terminal panel:** [one SUCCESS / one valid S1 failure / two engineering-missing conditions](reports/PI0294/20261016T191445Z-localpc-mp1/M2_BC_FIXED_TRAIN4_ACTUAL_PANEL.json).
F1/T2 is a valid original-S1 rejection at21.2s,425 camera pairs,zero native
faults; its19-check independent audit/video decode passed. It will not be
repeated. BC's208 generated actions reconcile to206 full ACKs,one75-tick
engineering partial andone S1-rejected zero-live prefix. CORR is now loaded
and evaluating its remaining fixed conditions in the same job; no new training.

**First independently verified complete M2 behavior:** [M2-BC original512 F1/T1 and real relay qualification](reports/PI0294/20261016T191445Z-localpc-mp1/M2_BC_COMPLETE_T1_AND_CLOSED_LOOP_QUALIFICATION.json)
is autonomous32s SUCCESS:641 actual camera pairs,41/41 hold,zero S1/native
faults. All18 raw checks and80 distinct observations/generations/actual ACK
chains match, including160 input-camera hashes. This qualifies one explicitly
metadata-reader-assisted paused-sim transport, not unassisted/real-time
performance. Scientific source4d430556 stayed pinned; no new training.
Remaining eligible conditions are still running on671899. Full M2 comparison,
DEV, T2 and target success are not implied; prior engineering failures remain.

**Current native run progress:** [third recovery first partial / disclosed metadata observers](reports/PI0294/20261016T191445Z-localpc-mp1/THIRD_RECOVERY_FIRST_PARTIAL_AND_DIRECTORY_OBSERVERS.json)
records BC F0-T2 reaching 29.35s/588 camera pairs, then stopping at the unchanged
900s episode wall bound. Its 28/28 observed hold checks do not make a complete
task SUCCESS. BC F1-T1 is now running. Two separately registered CPU directory
readers may affect NFS lookup visibility; their source/time identities are
disclosed, scientific capsules untouched. No unassisted performance or full32s
qualification is claimed yet. Job671899 is now R, not its earlier Q snapshot.

**Later explicit human revision:** [one additional bounded recovery submission](reports/PI0294/20261016T191445Z-localpc-mp1/THIRD_RECOVERY_ADDITIONAL_GO_AND_NATIVE_SUBMISSION.json)
changes only the recovery-family PBS cap from two to three, with one new
allocation at most 7,200 seconds. Both prior costs/closed leases and all
per-condition attempt counts remain. New child03/native job671899 is Q;
BC F0-T1 is exhausted and excluded, remaining BC3/CORR4 only. No training,
new T0/budget or claim of relay/behavioral success. No fourth family submission.

**Latest independent Vanda lane:** the [sprint adoption / native camera binding](reports/PI0294/20261016T191445Z-localpc-mp1/SPRINT_ADOPTION_AND_VANDA_CAMERA_BINDING.json)
records job1463728 Q for a bounded A40 EGL/frozen-task-camera check only.
Original container/assets are reused, not rebuilt; no WBC/model/action/physics
is launched. The first request was rejected without a job and is preserved.
Queued is not runtime PASS; 900 GPU seconds and 3,600 CPU seconds remain
reserved. TF-R3's actual ILLEGAL_CONTACT STOP remains unchanged.

**Latest MP1 native outcome:** recovery children 01/02 are both CLOSED;
671817 F/1 and 671826 F/0 cost 453 conservative H200 seconds, with zero new
updates. Four audited attempts still give no valid full policy result:
3 engineering-incomplete conditions / 5 NOT_RUN, not policy 0/8. An actual
30.301976s Hopper RPC accounting stall crossed the unchanged 20s watchdog.
The original two-submission boundary was sealed; the later explicit human
revision above permits exactly one further bounded submission, not training around the fault.
The [scoped RPC telemetry repair](reports/PI0294/20261016T191445Z-localpc-mp1/RPC_REPLY_LOCK_ISOLATION_CPU_RECEIPT.json)
has passed actual two-host CPU lock-isolation probes: successful replies no
longer take the metrics ledger write lock. Scientific charges remain durable;
this does not qualify SSH/compute-node NFS or a full32s model loop. Closed
scientific capsules were not hot changed; that CPU-only repair submitted no GPU job.
[Timing and qualification](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_REAL_RELAY_QUALIFICATION.json)
and [full attempt/condition matrix](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_E1_M2_RECOVERY_BEHAVIOR_MATRIX.json).
Follow the [evidence-first sprint entry](docs/handover/2026-10-17/MP1_EVIDENCE_TO_SCIENCE_SPRINT_EXECUTE.md)
under the original MP1 T0/budget; independent saved-T2/Vanda lanes remain.
The first subsequent [saved-T2 CPU audit](reports/PI0294/20261016T191445Z-localpc-mp1/M1_T2_SAVED_PROGRESS_AND_ACTION_AUDIT.json)
is complete: both objects entered the original goal zone before rejection,
but TCP clearance remained below 8cm. No new physics/model call, no complete
T2 success was claimed by that saved-only stage.
The subsequent [exact-history shadow diagnosis](reports/PI0294/20261016T191445Z-localpc-mp1/M1_T2_FIRST_REJECTION_ATTRIBUTION.json)
has now resolved the contacts: F0 both finger pads/table, F1 right pad/table,
at the original 56/52 shadow ticks. Expert/XYZ-only short prefixes pass while
SO3/gripper-only still reject; no unique Z fix, relaxed S1, full Recovery or
new task success is claimed. Child CLOSED; 19,123 new shadow steps, no GPU.
The subsequent [original T2 TRAIN30 stage/window audit](reports/PI0294/20261016T191445Z-localpc-mp1/T2_TRAIN30_EVENT_AND_H40_COVERAGE.json)
passed: detach/withdraw appear in all 30 demonstrations. Detach occupies
4.19% of anchors; no pure-detach H40 exists because detach lasts less than
2s, but 1,926 valid cross-stage windows include it. No automatic S-EVENT or
training recipe change; sampler forecast is not actual consumed training.

**2026-10-17 actual owner update:** E1's three PBS jobs and both leases are
terminal/CLOSED, with 19,675 conservative GPU seconds and zero reservations.
M1@1024 retains autonomous 32s T1 success in both seen TRAIN layouts; both
T2 layouts are valid original-S1 rejections. M2-BC/CORR each completed 512
updates with sealed C0, but their autonomous comparison remains incomplete
because of relay engineering failures: five incomplete conditions and three
NOT_RUN, not policy 0/8. No fourth E1 submission or closed-lease restart.
[Actual terminal summary and evidence](reports/PI0294/20261016T150228Z-localpc-e1/E1_RUN_SUMMARY.md).
Repaired C1 was independently consumed and is available for a separately
bound MP1 branch, not hot inserted into E1.

**The user adopted the single MP1 incremental package** at
`c8cdf60e41e31da5796b44a064ef2f22d594a491`. E2 is non-additive reference only.
The same owner registered MP1 T0 once; existing E1/Vanda jobs, original budgets,
negative results and native safety STOPs are preserved. No new MP1 GPU/model/
physics work is established by registration.

Read [CURRENT_SCIENCE_STATUS.md](CURRENT_SCIENCE_STATUS.md), the
[fixed MP1 execution entry](docs/handover/2026-10-17/MAINLINE_ACCELERATION_MP1_EXECUTE.md),
and [actual adoption receipt](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json).
Native refresh takes precedence over the historical snapshots below: Vanda
hardware1462438 and saved-input inference1462451 are F/exit0 with CLOSED
leases; native CPU build1463204 passed. These are not EGL/native WBC or
new-observation closed-loop qualification. Source generalization, T2
competence and target transfer remain unproven; TF-R3 STOP is preserved.
Real-robot deployment is NONE.

## Historical Entry Snapshots (Preserved)

**Vanda 独立闭环建设已获用户 GO，当前环境构建及 GPU 资格排队中。** [实际执行/失败修订及作业状态](reports/PI0294/vanda-bringup-20261016/CLOSED_LOOP_PROGRESS.md) · [执行协议](docs/protocols/PI0294_VANDA_CLOSED_LOOP_BRINGUP_V1.md)。用户指定 WBC 归档已三端 SHA 通过、六项原核心输入匹配；首 CPU 构建因站点容器挂载目录失败，修订后独立 job1462595 已排队。WBC/GR00T真实闭环及任务成功仍 NOT_RUN，不更改 Hopper E0、CoreLab原参考/旧失败或训练。

# GR00T＋WBC — 2026-10-16 E0 与后续代码准备

**E0已实际停止并收尾：PARTIAL，12/12尝试、11/12科学可判定，TRAIN0/3、DEV0/8成功。** [完整终报](reports/PI0294/20261016T083821Z-localpc-e0/E0_FINAL_REPORT.md) · [逐条件结果](reports/PI0294/20261016T083821Z-localpc-e0/E0_MODEL_BEHAVIOR_RESULTS.json) · [原视频索引](reports/PI0294/20261016T083821Z-localpc-e0/E0_RAW_EVIDENCE_AND_PRIVATE_VIDEO_INDEX.json) · [资源终态](reports/PI0294/20261016T083821Z-localpc-e0/E0_RESOURCE_FINAL.json)。11条原S1预测非法接触而预执行拒绝；1条本地中继中断工程不完整，不算策略FAIL、不重跑。12条保存证据/实际视频审计PASS，真实安全故障0；217生成、41000主步/42075影子步、1652双RGB。两PBS均F，原生1708GPU秒/保守1723秒，预留0、两lease CLOSED；无新训练/WBC/S1变化，selected/deployment NONE。以下排队/运行文字均为阶段历史。

**E0最后一次同语义工程提交已实际进入Q：670345.hopper-m-02。** [恢复绑定与提交收据](reports/PI0294/20261016T083821Z-localpc-e0/E0_ALLOCATION_02_SUBMISSION.json)，runtime固定ce1a14a56ddfe9f322e159eef8a2f8054da22525，两端封存入口PASS，15原数值文件及模型generate逻辑不变。2/2 PBS已用，原窗口/总额保持；当前尚无模型结果，以下首次失败/旧排队记录保留为历史。

**E0最新原生增量：首个allocation已F，启动前认证过期，模型/物理均未开始。** [首次实际终态](reports/PI0294/20261016T083821Z-localpc-e0/E0_ALLOCATION_01_TERMINAL.json)：183 GPU秒原生、188秒保守扣账，12条件仍NOT_RUN而非策略失败。正在修复同语义准入并准备唯一剩余工程提交；原T0/截止/总额不重置。下段排队措辞为此前提交阶段历史，不是实时状态。

**E0已采用并实际提交首个PBS，当前仍在排队，不是评价完成。** [本次独立绑定、固定12条件及实际提交收据](reports/PI0294/20261016T083821Z-localpc-e0/README.md)。只评价原v3.0/EEF10/source D0＋原S1的M2048，先TRAIN4再DEV8；TRAIN成功率不作新增门。runtime固定9a699d28c6a585df1abb5aa906e41e8d8c025bdc，报告/测试HEAD不热替换它。旧16项NOT_RUN保留，selected/deployment NONE。

**用户另行授权等待期间准备FlowDAgger代码，不授权新科学执行。** [实际代码/CPU准备报告](reports/PI0294/flowdagger-preparation-20261016/PREPARATION_REPORT.md) · [协议](docs/protocols/PI0294_FLOWDAGGER_PREPARATION_V1.md) · [机器卡](reports/PI0294/design/FLOWDAGGER_PREPARATION_TASK_CARD_V1.json)。独立纠正数据合同、原EEF10 processor桥、冻结模型inversion/steering组件及CPU测试已准备；FD1真实可恢复性、FD2真实模型重构和FD3训练未运行。不能把专家全接管默认例子称作学生状态DAgger，也不将E0 DEV转训练或零填BASE。新代码不部署到在途E0，不重开旧Teacher/WBC任务。

**完整GPU等待期Q0–Q6准备已按新增固定handoff接续。** [CPU准备终报与全部小收据](reports/PI0294/design/gpu-queue-dagger-prep-v1/PREP_FINAL_REPORT.md)：67项定向fixture通过（含2项复用BASE测试），默认827PASS/14显式SKIP；离线Oracle可以INVALID、chunk交易/ACK与split血缘已测试，合成状态不进真实训练。真实学生状态恢复、M2048反演和DAgger训练仍NOT_RUN。初期测试未约束CPU亲和性、历史并发不能认证，后续已固定2核/单线程并披露资源核算限制；不掩盖该偏差。

以下2026-10-15及更早条目是各自阶段历史；最新执行状态以固定收据、#11/#47及原生run/lease为准。

# GR00T＋WBC — 2026-10-15 历史科学入口

**TF-R2已执行S7至真实接触，因接触后夹爪原生安全FAIL停止；尚无成熟Teacher或成功数据。** [本轮报告、500Hz时序图、独立CPU复算与2018原件私有回读](reports/PI0294/tf-r2-20261016/REPORT.md)。用户取消累计资源上限后，执行同源seed0/PC2的0.95s闭合请求；实际6385包、20安装、156双RGB。5.85→6.80s闭合前后至首次接触均无越界；7.662s记录左pad承力接触，7.768s左slide首次低于原0mm，7.770s原registered joint safety停止，最小−0.177230mm。PD全步复算误差0N，其余19关节无越界。source隔离、task物理停止门已实际拒绝；按TF-R2第6节不进入S8/S9。成功seed/TRAIN/DEV/正式consumer仍0，WBC/安全/评分/旧FAIL不改，selected/deployment NONE。下列夜间/R1等条目为历史，不因本次预算更新被改判。


**TF-NIGHT已到原生共享启动安全停止点，数据闭环未成功。** [夜间终局、22步实际calf/ctrl图与私有回读](reports/PI0294/teacher-factory-night-20261016/FINAL_REPORT.md)。T0 2026-10-16 00:23:27SGT；采用354e45ab独立scope。T2原生启动22步/.044s原registered joint safety停止，FL_calf=-2.77245＜-2.7227；Teacher提交/任务RGB/成功TRAIN/DEV均0，T1未运行。未证明task-side可修根因，按协议5/11停止新物理，无追相位重跑、无WBC/安全/score修改。3581原件私有逐文件回读；owner/worker终态、预留0；selected/deployment NONE。下列准备/采用/V3进展为历史记录。

**V3已实际运行至合同停止点：真实单侧物体接触已取得，完整成功seed仍0。** [报告、实际TCP/接触图及662原件私有回读](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/source-demo-access-v3/REPORT.md)。12CPU实际PASS；T1运行5.6s/14安装/113双RGB，右pad真实接触累计.570s；下一未安装窗口原关节加速度4.01285＞1.0门拒绝。独立500Hz另见右slide6点raw超44mm、峰值44.06553mm；无原生stop≠原始机械范围PASS，不能作为成功示范。T2两入口10.346/10.467＞10s、两dense预算用完，001/002未认证/未评，未运行。累计40启动/364111扣账步；整体PARTIAL，停止本V3，无重试/训练，旧FAIL/WBC/安全/评分/窗口不变，selected/deployment=NONE。下列采用/待执行均为历史阶段记录。

**完整V3已获用户明确采用，含累计搜索217bundle/214214dense重分配；沿原窗口/父账执行。** [源绑定人类采用](reports/PI0294/design/SOURCE_ACCESS_V3_HUMAN_ADOPTION.json)。保留V2安全FAIL，新版本实施受影响CPU契约→各task自身源/actual warmup→完整32s；采用不代表运行PASS。下列待采用描述是此前发布阶段记录。

**V3已准备、待明确采用：T1完整任务组件抬高3cm到H47；T2固定首contact直接接入。** [完整有界方案](docs/protocols/PI0294_TABLE_CLEARANCE_AND_CONTACT_ENTRY_SEED_V3.md) · [任务卡](reports/PI0294/design/SOURCE_ACCESS_V3_TASK_CARD.json) · [保存几何/源前缀准备证据](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/source-demo-access-v3-preparation/PREPARATION_REPORT.md)。旧5300步包络含margin最高.399445m，新完整桌底.42m留下20.55mm保存净空；不证明新动态安全。T2三源223/231/223前无pad接触/推送，保留15.50/16.25/16.10s后缀，显式working-closed初态。拟各32s，≤4启动/42012积分/1282双RGB/14000CPU秒，沿原窗口/累计父账，内部累计分配明确217bundle/214214dense、仍在原外层预算内。**本次只准备/发布，authorizes_execution=false、launch_allowed=false、runtime未实施、变更CPU契约0/12，新IK/RRT/渲染/物理/神经/训练0；V2安全FAIL永久保留。**

**V2已实际执行至明确安全停止点，未取得成功种子。** [完整报告、实际TCP图与私有归档收据](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/source-demo-retime-join-v2/REPORT.md)。20CPU检查通过；T2局部时间18.6/19.4/19.1s但三接近端无合格IK；T1锁定000后实际运行5.6s、14安装/113双RGB，原生base_link几何与桌面ILLEGAL_CONTACT。任务物体接触/抓持/成功seed均0，两条完整32s均未完成，不报0/2完整成功率。636原件私有逐文件回读；旧FAIL、WBC/安全/score、时钟/账不变；本scope关闭，不重试、不训练，selected/deployment=NONE。下列采用/未实施是历史阶段记录。

**V2完整方案已获用户明确采用，原窗口与累计预算不变。** [采用收据](reports/PI0294/design/SOURCE_RETIME_JOIN_V2_HUMAN_ADOPTION.json)，采用固定`5a0cceeeb7e79f6f326a5aa17168a453acb65180`。当前接续版本化时间表/首close接入与实际CPU准入，尚未取得V2物理/种子结果；下列“待采用/未实施”是此前发布阶段记录，不能覆盖本次采用，也不能用采用代替执行PASS。

**下一版已准备、待采用：T2局部减速＋T1首close直接接入V2。** [完整有界方案](docs/protocols/PI0294_SOURCE_LOCAL_RETIME_AND_DIRECT_JOIN_V2.md) · [任务卡](reports/PI0294/design/SOURCE_RETIME_JOIN_V2_TASK_CARD.json) · [实际只读准备](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/source-demo-retime-join-v2-preparation/PREPARATION_REPORT.md)。保存数组显示T2保留所有源点、只局部延长超速区段时cartesian时间下界为18.60/19.40/19.10秒；T1原首次close索引30仍张开且有旧合法goal，拟以实际自由连接替换原30个非接触align点、保留原均匀减速及完整抓放后缀。连接/关节/碰撞/动态成功尚未验证。沿原窗口/累计预算，拟直接各32秒、≤4启动，无新短门或FSM阈值搜索。**本次只准备发布，launch_allowed=false，runtime未实施，V2正式CPU准入0/20，新搜索/物理/训练均0。** 原V1负结果与旧WBC/安全/score保持。

**自有示范重定位已执行至明确科学停止点：六个固定TRAIN来源均未准入。** [完整候选/独立复算/归档终局](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/source-demo-retarget-v1/REPORT.md)。T2三源uniform slowdown后需50.49/53.87/61.88秒，超过28秒；T1三源接近端均无满足原限位的登记解，接触端有解但不能组成pair。16项CPU检查及6源提取完成，3bundle/6目标IK/96CLIK，新增RRT/物理/RGB/模型/训练均0，真实接触与成功seed未取得。终局`STOP_FIXED_SOURCE_POOL_NOT_ADMITTED`，整体仍PARTIAL；不是WBC物理FAIL或全局IK无解。75文件私有完整回读，旧FAIL/WBC/安全/评分/窗口/累计账保持，selected/deployment=NONE。下列“待采用/未执行”均为历史阶段记录。

**自有示范重定位完整方案已获明确采用：** [源绑定人类采用收据](reports/PI0294/design/SOURCE_DEMO_RETARGET_HUMAN_ADOPTION_V1.json)，固定采用e7d6537facc3c670cdd197e64302eac913965534，原窗口/父账不重置。执行从真实源提交片段/工具映射接续，WBC、安全、原评分和旧FAIL保留；采用不代表新CPU/规划/物理已PASS。下列“待采用”是此前发布阶段记录。

**固定小样本已下载；自有成功片段重定位方案已准备、待采用：** [实际下载/来源资格](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/source-demo-preparation-v1/PREPARATION_REPORT.md) · [完整有界方案](docs/protocols/PI0294_SUCCESS_DEMO_SEGMENTS_TO_FULLBODY_SEED_DRAFT_V1.md) · [机器卡](reports/PI0294/design/SOURCE_DEMO_RETARGET_TASK_CARD_V1.json) · [采用prompt](docs/handover/2026-10-15/SOURCE_DEMO_RETARGET_ADOPTION_PROMPT.md)。四payload共338566923字节完整SHA通过，6条原TRAIN源身份/成功/成员确认；外部包仅作参考库，不阻塞自有来源。拟直接T2/T1各32秒，≤4启动/42012积分/1282双RGB/14400CPU秒，168bundle及内部IK/RRT/dense逐项封顶，沿原截止/累计余额、冻结WBC/安全/评分，不再修FSM阈值。**本次仅下载/准备已获准并完成；新重定位搜索/物理未授权未执行，launch_allowed=false，运行适配尚未实施。** 原MP-S2真实结果与未启动辅助槽保持，采用新方案后才在停止边界独占接续，旧FAIL不改。

**MP-S2已进入真实物理，当前PARTIAL（并非仍待采用）：** [实际结果与独立CPU审计](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/maniskill-search-v2/PROGRESS.md)。16CPU项通过；T1原首帧无合法规划对，T2完成12秒但无接触，短里程碑有效FAIL；两完整AUTO均未准入。原记录器故障保留，241组RGB从相同保存实际状态恢复并收费；真人辅助尚未启动。新成功整机seed/训练均0，旧FAIL/窗口/预算不改。另已完成[公开成功示范来源与重定位审计](docs/reviews/2026-10-15_SUCCESSFUL_DEMO_SOURCE_AND_RETARGET_AUDIT.md)，锁定官方metadata/revision/hash，未下载轨迹或启动新来源实验。以下“待采用/未实施”保留历史阶段措辞。

**下一版已准备、待采用：MP-S2有界候选搜索＋交互示范备选。** [完整方案](docs/protocols/PI0294_MANISKILL_BOUNDED_SEARCH_AND_ASSISTED_SEED_V2.md) · [机器卡](reports/PI0294/design/MP_SEARCH_V2_TASK_CARD.json) · [接续prompt](docs/handover/2026-10-15/MANISKILL_SEARCH_V2_ADOPTION_PROMPT.md)。每task6几何包、approach/contact成对、单bundle≤12目标IK/192内部初值/8RRT；内部失败继续固定剩余槽。单task独立准入，完整slot互斥走AUTO或真实人工ASSISTED，失败后不换路线重跑。保留原窗口/父账、≤6启动，WBC/安全/最终score冻结，不循环修FSM阈值。**本次只准备发布，authorizes_execution=false/launch_allowed=false；新搜索/物理/seed/模型均0，功能未实施。** 以下旧系列终局永久保留，新方案不自动重开它。

**ManiSkill组件本轮已到有效规划资格停止点：固定T1/T2起点均无满足原限位的登记IK候选，物理未准入。** [实际结果、独立原生FK复算与私有归档回读](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/maniskill-component-v1/REPORT.md)。24项CPU接口检查通过；12个固定保存输入中7个后续输入可规划、5个失败，不能挑后续帧替换两个失败起点。新物理/接触/成功seed/训练均0，终局`STOP_FIXED_START_NO_ADMISSIBLE_REGISTERED_IK`；不是CPU_READY、WBC任务FAIL或全局IK无解。原WBC、安全、评分和旧终局未改，selected/deployment=NONE。下列采用/待采用措辞保留各自历史阶段，不代表继续运行的许可。

**ManiSkill组件方案现已获人类明确采用执行。** [源绑定采用收据](reports/PI0294/design/MANISKILL_COMPONENT_HUMAN_ADOPTION_V1.json)，采用固定13713019077cb128b55090ab6b7a513e162b975e；原窗口/累计预算、冻结WBC/安全/任务评分不变。将依固定CPU接入→两任务短资格→各自完整32秒推进，采用不代表运行或seed已PASS。下列“待采用”是前一发布阶段记录。

**新增已准备/待采用方案：ManiSkill规划组件→整机W0真实接近/接触/成功seed。** [完整有界方案](docs/protocols/PI0294_MANISKILL_PLANNING_COMPONENT_SEED_QUALIFICATION_DRAFT_V1.md) · [机器卡](reports/PI0294/design/MANISKILL_PLANNING_COMPONENT_TASK_CARD_V1.json)。冻结WBC/原T1/T2评分，替换旧cursor与误差阈值阶段前置；采用固定MPlib IK/RRTConnect、隔离NumPy环境。拟2短＋独立条件2完整、最多2任务前工程替换，共6启动/59018积分/1764双RGB/6CPUcoreh，花原父账与绝对窗口；无GR00T/GPU/PBS/扩采。用户本次只授权准备和发布，authorizes_scientific_execution=false/launch_allowed=false；未安装、未启动新物理、未取得成功seed。下列V2是已有实际终局，不因新方案改判。

**最新实际V2已闭合：3条有效12秒短里程碑FAIL＋1条任务前就绪工程不完整。** 无预接触/物体接触/操作，条件两条完整确认未准入；三条角误差最小约0.143/0.097/0.171rad、平移游标仍零。原件2323文件/530436177字节逐文件私有回读通过，旧终局未改，成功整机TRAIN0、训练0、selected/deployment=NONE。见[完整结果、CPU复算、图与终账](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/component-teacher-v2/REPORT.md)。

**后续准备：优先复用成熟规划组件，冻结MuJoCo/WBC。** 已固定读取ManiSkill八个原始文件并核对Panda joint动作、PickCube不含释放/撤离成功条件、实际ARX5与全Go2碰撞资产差异。见[源码到EEF13的具体适配审核](docs/reviews/2026-10-15_MANISKILL_PINNED_SOURCE_TO_EEF13_ADAPTER_AUDIT.md)。仅源码/接口研究，未安装新框架、无新物理/模型/GPU/PBS许可，没有本体成功seed；不再循环补游标阈值。

以下V2“草案/未采用”与V1“最新”措辞是历史阶段记录；后续人类已明确采用V2并得到上述实际终局，不覆盖或改写历史原件。

**最新零物理步补证：44次保存cursor更新严格重放一致，其中34次位置检查在请求平移为零时提前拒绝。** 已准备[分量独立推进V2单项草案](docs/protocols/PI0294_COMPONENT_PROGRESS_V2_QUALIFICATION_DRAFT.md)，CPU fixture／44个独立单步提案检查通过；这是改变Teacher准入合同的未采用草案，authorizes_execution=false／launch_allowed=false，不重开前轮物理或预算。完整示范目标仍未完成，旧终局不改。

**最新已采用的新整机Teacher短系列已终结：2条有效无进展FAIL＋2条时间表示工程中断。** 4个短条件各启动一次，19462含warmup／尾部积分、380实际双RGB；无预接触／物体接触／操作。两条完整确认未准入、成功整机TRAIN0、训练0。CPU已证明并窄修10.2秒负龄期浮点表示，但修复后未再物理运行。保存查询的持久游标未进入目标平移段；Teacher绝对位置包络与旋转推进耦合是具体下一设计问题，未转向WBC调参。原件1302文件／440579732字节私有完整回读，旧0/4／FAIL不改，selected/deployment=NONE。见[实际结果、图与终账](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/persistent-teacher-v1/REPORT.md)。本系列入口已关闭，原科学总目标仍PARTIAL；不把工程中断算科学FAIL，不无条件重跑。

以下为新系列采用前的实际数组准备记录，未授权描述对应当时提案，不覆盖后续直接人类采用及上述已执行终局。

**接近阶段已有真实数组复算：三条位置条件均0/641，推送姿态条件已568/641、561/641成立。** 同刻上一安装参考与实际TCP中位差约3.4mm，但实际距持久阶段目标最终34.8／40.8／46.1cm；原50ms重锚定与prefix8声明不一致，MPC消费身份仍UNKNOWN。见[时间对齐报告和图](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/approach-audit/REPORT.md)。单项持久接近Teacher组件只有CPU草案／10项测试及真实保存输入dry run；新物理、训练0，launch_allowed=false，不因草案或审核自动开放新pilot。

**最新实际任务结果：固定四个整机pilot成功0/4。** V1近场抓放原生安全FAIL保留；V2其余三条完成32秒、真实双RGB／参考／时序及独立CPU一致性通过，但物体任务均FAIL。成功数据0，v4.0／FB13未准入；未运行新训练。已封存全部原件／私有视频及逐文件回读，冻结WBC，不重复有效失败。见[完整结果与收口收据](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/FIXED_PILOT_RESULTS.md)。执行完成与任务成功分开，整体数据—训练—闭环仍PARTIAL，selected/deployment=NONE。

**最新用户决定：不调整WBC，只借用完整环境采集任务数据和训练GR00T。** [主线纠偏与Codex接续规则](docs/protocols/GR00T_TASK_DATA_FIRST_FROZEN_WBC_SCOPE_V1.md)（内含机器规则）。R4B D0已结束，D1及继续控制器归因/补偿/调参不再属于本团队下一步。WBC保持不变。

**新采集采用“真实物体任务成功＋原生安全＋数据完整性”准入；旧保持RMS质量FAIL保留，但不再作为全局数据采集/训练的前置阻断。** 不调高0.05rad、不删除旧FAIL、不忽略实际限位/非法碰撞。应补齐任务夹爪/相机/Teacher，直接检查原样环境能否完成抓放与推送，再制作EEF13数据、并行采集和微调。脚本存在不等于任务或数据已经通过。

本方向已由用户明确，不新增任何资源总量或窗口；owner需绑定有效任务/资源scope，不能自动重开关闭的lease。以下是已发布的事实与历史阶段依赖；其中的“依赖链已停止”描述原阶段终局，不代表新任务还要等待WBC修复或保持精度PASS。

---

**R4B D0实际只读归因已完成：同控制周期FK与力矩证据支持手臂伺服实现偏差主导姿态误差，并有身体预测/实现耦合；尚未证明可自主修的实现bug。** [WBC负责人缺陷包](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/R4B_D0/WBC_OWNER_DEFECT_NOTE.md) · [只读终账](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/R4B_D0/RESOURCE_AND_FINAL_REPORT.md)。新增物理/模型/PBS均0；R4B D1仍未采用，不新开窗口。

**最新实际结果：R4A 原生完整机器人完成10秒世界系EEF保持，姿态RMS0.06965rad超过原0.05rad门；时钟、参考安装及原安全检查通过。整机教师/数据/FB13依赖链已停止，尚无整机学习模型。** [实际报告与CPU收据](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/FINAL_NATIVE_QUALIFICATION_REPORT.md)。原始失败保留；这是有效负结果，不是R4A整体完成，selected/deployment仍NONE。

**最新用户方向：完整Go2＋ARX5环境生产数据和测试，GR00T增加有界BASE速度意图，经原生BASE软位姿接口接入冻结WBC；工程问题在明确范围和既有资源内自主修复推进。**

- [R4A完整长程计划](docs/protocols/PI0294_R4A_FULL_GO2_ARX5_AND_BASE_INTENT_V1.md)
- [EEF13／BASE软参考接口契约](docs/interfaces/EEF13_BASE_SOFT_REFERENCE_V1.md)
- [R4A机器任务卡](reports/PI0294/design/R4A_FULLBODY_TASK_CARD.json)
- [现有Codex直接接手Prompt](docs/handover/2026-10-15/PI0294_R4A_EXECUTE_PROMPT.md)

读取根AGENTS、最新#11/#47与原生运行状态后采用。最新已核验模型是M2048；R4旧v3.1增量在v3.0统计下codec失败保留。固定底座实验是基线，不再是完整整机搭建的永久前置。已运行作业不得热改或重复提交；新scope只使用已授权R4剩余资源和窗口，不新增预算、不重置T0。完整WBC在CoreLab仅作冻结推理/仿真执行器，高层GR00T仍在合法Hopper allocation；不授权WBC重训、实机或历史删除。

完整整机、EEF13与BASE桥在有真实执行收据前是待实施，不因文档发布变成PASS。下列2026-10-14及更早的“当前状态/未授权/未推理”是历史接管记录，不应覆盖后续已发生的R2/R3/R4事实或本次范围增补。

---

# GR00T＋WBC — 唯一仓库与 CoreLab 工作入口（2026-10-13）

**新 Codex 与所有后续工作请先从这里开始。** GitHub 是本团队 VLA 代码、协议、测试、小型执行收据的版本权威；CoreLab 共享 `~/Projects/GR00T-WBC` 是人类统一工作入口，但不能把私有原始数据、WBC、权重都视为已保存在 GitHub。

- **[2026-10-14 新 agent 完整接管与科学推进计划](docs/handover/2026-10-14/NEXT_AGENT_EXECUTION_PLAN.md)** · [机器任务卡](docs/handover/2026-10-14/NEXT_AGENT_TASK_CARD.json) · [直接接手 Prompt](docs/handover/2026-10-14/NEXT_AGENT_BOOTSTRAP.md)：当前 PI0294 仍因真实夹爪原生限位 BLOCKED，修订待 GO；这是计划不是自动执行授权。今后交接由 GitHub 固定提交直接供 Codex 读取，不以本地 ZIP 为主。
- **[2026-10-14 完整工作总结与下一 owner 交接](docs/handover/2026-10-14/HANDOFF.md)**：Oct12-14 已执行事实、代码入口、固定 HF 资产、终局与剩余预算。当前 M512 实际闭环触发原生夹爪关节限位，科学线 BLOCKED；交接不授权重启。夹爪重资格方案仍未获明确采纳。
- **[现行统一入口／GitHub 同步规范](docs/operations/2026-10-13_UNIFIED_ENTRY_AND_GITHUB_SYNC_V1.md)**：host-aware profiles、固定源 SHA、运行 manifest、分支同步、旧目录退役验收。
- **CoreLab 现有可执行入口**：`~/Projects/GR00T-WBC/infrastructure/enter.sh amd-management`（AMD），或在 4080S 使用 `4080-simulation`／`4080-cpu-data`。不提供自动跨主机 SSH；不经许可启动物理／模型。
- **[最新真实维护终局](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47#issuecomment-6137981521)**：已实际退役33源路径；Projects仍7项，有受保护科学依赖。后续统一入口操作只凭新执行收据计为完成。
- **[科学主线 #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11)**：PI0292 以其独立 run capsule、source seal 和当前资源账为准，不要用当前分支 HEAD 自动替代已在执行的源码。
- **Git**：实际活跃科学分支 `pi0267/executor-decoupled-recovery`；`main` 是较早的稳定导航，不是当前仿真或训练启动源。只有作业边界+doctor通过，才允许新 `code/vla` checkout fast-forward，同步本地不改历史 run。

以下是 **2026-10-12 的历史接管文档**，保留供追溯；其中的“尚未授权实验／旧最新状态”不再作为当前实时状态或新 GO。

---

# 先读这里｜2026-10-12 新local-pc接管

**当前是接管与仓库维护，不是新的训练/仿真GO。**

## 唯一入口

- [完整local-pc接管prompt与双服务器/HF规范](docs/handover/2026-10-12_LOCAL_PC_ADOPTION.md)
- [分支清理allowlist及原SHA](docs/handover/2026-10-12_BRANCH_RETIREMENT_PLAN.json) — 尚未实际删除；local-pc先建立兼容tag并回读，再按精确lease删除四个祖先分支。
- [科学ledger #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11)
- [可复用代码/CI/仓库维护 #47](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47)

## 最新用户方向

我们负责独立任务仿真、可信专家数据与上层GR00T的EEF/夹爪行为学习。师兄WBC仅作冻结执行器，不再由上层团队调参修复。TaskAnchor/UNITAS只是设计参考，尚未实现。本轮只读接管、条件化分支维护与下一数据GO草案；不恢复旧实验、不启动新模型或物理。

| 位置 | 职责 |
|---|---|
| local-pc | 唯一owner；分别SSH两端、GitHub审核交互；未来可作闭环obs/action中继 |
| Corelab | 仿真/教师数据/预处理/物理推进/CPU评分/视频，非高层GR00T模型eval |
| Hopper H200 | GR00T训练微调、模型推理与模型eval，须合法PBS allocation |
| GitHub | 代码、协议、有限预算、小收据、审核 |
| HF | 数据/权重/大证据，固定revision、prefix和hash |

两服务器互不连通；不要尝试直连scp/ZMQ。HF用于大资产，不是每0.4秒的实时控制队列。闭环需要当前动作造成的新观测；local-pc双SSH中继尚需另行验证，离线回放不能冒称闭环。

## 最近完整科学状态

PI0285源Panda确认20/32未过26/32。PI0286 C0 TRAIN2/8；C1 TRAIN8/8、DEV5/16；未运行新确认。PI0289安全启动PASS，但首hold姿态RMS0.0652rad>0.05、时钟残差47.8ms>25ms；其余五参考NOT_RUN，目标教师数据0。最近ledger终局是2026-10-10 11:08:34 SGT；接手须刷新最新证据。selected/deployment=NONE。

现役代码分支pi0267/executor-decoupled-recovery。main仍是旧accepted代码＋导航，不能直接当训练runtime。旧README、hpc gate、10月8–10日deadline与PR77不构成当前执行授权。

第一轮交付：ADOPTION_REPORT、RESOURCE_MAP、ASSET_AND_HF_MAP、NETWORK_AND_EVAL_PLAN、实际BRANCH_MAINTENANCE_RECEIPT、NEXT_DATA_GOAL_DRAFT（未授权）；发#11回读后停住等新科学GO。


### 2026-10-25 A24 terminal / B zero-effect startup closure

A N294101 is complete and independently audited: 11/24 success (T1 7/12, T2 4/12), 13 valid failures, 0 actual native safety faults. A is once settled and dual-closed. Automatic B startup failed pre-model at the strict read-only predecessor evidence gate; B remains 0 valid, B−A unknown. B was charged once (70 GPU seconds, 1 process CPU second; original conservative bounds retained) and dual-closed with independent readback. One fresh cold B24 replacement is in CPU preparation only, preserving all original rows/costs, A24 and clocks; no native adoption or new training is implied. [Actual report](reports/PI0294/design/parallel-science-20261022/V4A_A24_COMPLETE_AND_B_ZERO_START_CLOSED_ACTUAL_20261025.json). Full goal ACTIVE; core64=0.

Fresh B24 cold replay of actual captured full ledgers found only two storage gaps while retaining every failed cost and all future bounds: GPU-site peak 810.58833 GiB against 810 GiB; Core peak 328.51620 GiB against 300 GiB. A [two-cap draft](reports/PI0294/design/parallel-science-20261022/V4A_B_ZERO_START_REPLACEMENT_EXACT_STORAGE_SCOPE_DRAFT_20261025.json) proposes 811/329 GiB. This is not adopted authority or native admission; no new B registration or dispatch has occurred.


### 2026-10-25 — human approved exactly the two storage changes

The human directly approved only GPU-site storage 810→811 GiB and Core storage 300→329 GiB for the one original B24 cold replacement. [Exact authorization](reports/PI0294/design/parallel-science-20261022/V4A_B_ZERO_START_TWO_STORAGE_CAP_AUTHORIZATION_20261025.json). The captured-byte full all-resource proposal now has zero shortfalls, retains the 70-second failed startup and its full output bound, original three matrix jobs, and final 20 GPUh/40 GiB protection. GPU/CPU/PBS/query/physics/transfer caps, all clocks and scientific gates are unchanged.

This approval is not yet actual dual-ledger adoption, registration, native model permission or an Eval result. The cold-only helper and related tests pass 152 cases. A broader local environment attempt is not represented as an all-green suite (missing Torch/safetensors, old pytest fixture compatibility and separate target ABI tests); no CI success is claimed. The next action is exact dual-ledger once adoption, fresh unique request/resources, standard registration, and the original native qualification before B model load.


### 2026-10-25 — B24 cold backend is actually running

The two approved storage changes are now adopted once in both existing ledgers (811 GiB GPU-site, 329 GiB Core; all other caps and original clocks unchanged). A single new original B4096/N294101 24-condition request is registered and bound to its real Core GPU process under a healthy server-resident owner; automatic service restart is disabled. Source a6991fca99f82e8dc413816e8bde6c8f0e1552ea has 1070 files fully read back on both sites, and all original 15 numerical files match.

The original native identity/resource admission passed with zero full-forecast shortfalls. At the actual readback the 600-second/80-cycle communication qualification was running: zero models, queries and task physics. Subsequent fresh model release, evaluation, independent audit and terminal closure are handled by the finite server owner; a foreground agent is not required. Native communication qualification, model permission and valid task outcomes remain separate. The original failed B row and all costs/STOP files, the original two-row queue, and the completed A24 are retained. No A replay, new noise bank, data capture or training is scheduled.

[Actual backend binding and complete resource proof](reports/PI0294/design/parallel-science-20261022/V4A_B24_COLD_BACKGROUND_OWNER_NATIVE_BOUND_ACTUAL_20261025.json). A is 11/24; B has no valid outcomes yet, so B−A is unknown. Full goal ACTIVE, core64=0.


### 2026-10-25 — the cold B owner is terminal, not healthy

The c90d backend-running receipt is a historical snapshot. The same B cold request subsequently failed during zero-model qualification: Vanda qselect timed out after its original 10-second bound; the old owner issued STOP. Core GPU hardware was not the diagnosed cause. The actual owner and GPU unit are terminal Exit1; zero models, queries, episodes and physics. Actual costs (678 GPU seconds, 3 process CPU seconds plus original control upper) are once settled and both leases independently closed.

Unique newly allocated output is measured from stable actual files, with 64 MiB per-role metadata allowance. Only exact immutable hardlinks to already-paid source/model input inodes are excluded; all original A/first failed B full output charges and every historical counter remain. A fresh captured full forecast fits the already adopted 811/329 GiB caps with zero shortfalls and all original future/matrix/final bounds.

The narrow cold engineering repair separates scheduler-read uncertainty from a live Core process: at most three original bounded read attempts; raw fresh ledger reads may keep an actually live owned Core process heartbeat, explicitly labelled QUEUE_UNKNOWN. They never mean empty queue, native admission or model release. Fresh full queue and full resources remain required for model release. There is one exact additional known-zero read-failure boundary, not a retry pool; no old actor/request is restarted. A24=11/24 is unchanged; B−A remains unknown. [Actual closure and concrete repair](reports/PI0294/design/parallel-science-20261022/V4A_B_COLD_QUEUE_READ_FAILURE_KNOWN_ZERO_CLOSED_AND_REPAIR_20261025.json). Full goal ACTIVE, core64=0.


### 2026-10-25 — B cold deployment is UNKNOWN; no new GPU is running

The approved 811 GiB GPU-site / 329 GiB Core caps remain adopted once. A24 is audited and dual-closed at 11/24 (T1 7/12, T2 4/12); B has zero valid outcomes. The new Core control exited before reserve/registration because five new Vanda account jobs lacked exact human identity confirmation; complete GLOBAL and LOCAL ledger bytes remained unchanged. The human then confirmed all five as independent work. Source `fccad14fa3edb307491be589068d2aaca9314eef` binds exactly those IDs/name/project/GPU resources, preserves the full queue, and does not modify/adopt/charge them; both sites fully read back 1074 files and the original 15 numerical files. Nine focused identity checks and the captured full-resource proposal passed; CI is not claimed.

The subsequent cold deployment returned a 40-second SSH timeout. An exact read found no target control directory. This UNKNOWN write stays latched, is not treated as success, and is not automatically re-sent. No new scientific request/lease/GPU/model/task physics was created. Original stop-new remains 1792866596 (25 October 02:29:56 SGT); no new job is permitted after it. Full goal remains ACTIVE/incomplete and core64=0. [Actual evidence](reports/PI0294/design/parallel-science-20261022/V4A_B_FIVE_JOB_IDENTITY_CONFIRMED_PRE_RESERVE_EXIT_AND_DEPLOYMENT_UNKNOWN_ACTUAL_20261025.json).


### 2026-10-25 — post-cutoff deployment cleanup and one B24-only draft

The exact pending preparation process (PID/proc-start verified against its SSH parent and blocked output pipe) was still live after timeout. It has now been terminated once and independently verified missing; the new control directory, both role roots and GPU unit are absent, and full GLOBAL/LOCAL bytes remain unchanged. The old UNKNOWN stays latched; no deployment or scientific request was re-sent. This corrects any inference that timeout itself proved termination.

A data-only stdin frame now avoids compiling a 14 MB single-line payload. Six focused test methods passed, including truncated/corrupt/trailing-data rejection; the real 57.5 MB packet completed an actual read-only Core round trip in 13.2 seconds with the original 40-second limit, zero writes/models/queries/physics. The exact old remote error cause is still unknown; no GPU or scheduler fault is inferred. The source and tests are an offline/control transport repair, not scientific admission.

[One B24-only exception draft](reports/PI0294/design/parallel-science-20261022/V4A_B24_ONLY_LATE_SUBMISSION_KNOWN_ZERO_COLD_BOUNDARY_EXCEPTION_DRAFT_20261025.json) would permit one new cold boundary after exact zero-effect reconciliation and only the original B24 submission by 25 October 04:29:56 SGT. It does not reset T0/global stop-new, extend the other deadlines, add caps/training/panels, or consume the original matrix/final protection. Its full-future arithmetic also retains an extra 7200-second control upper and 64 MiB per-site output/256 MiB LOCAL+transfer bounds for failed preparation; all dimensions fit existing caps. This is **not approved or adopted**. Original stop-new is currently binding; no new job is running. A24=11/24; B valid=0; full goal ACTIVE/incomplete; core64=0.
