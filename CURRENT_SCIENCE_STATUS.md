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

> **当前唯一原生执行优先级（V4A）：** [完整原噪声bank294101的A/B匹配评估](docs/protocols/GR00T_WBC_V4A_EVAL_FIRST_STAGED_EXECUTION_20261024.md)，若实际全链成本/截止可容纳则A24+B24（最多两PBS）再决定其它。V4.1的A8+B8小诊断只是[需要新增合法子面板冷准入的备选](docs/handover/2026-10-24/SCIENCE_FIRST_MINIMUM_SPEND_EVAL_V4_1.md)，**不得当作现有成熟24格eval程序已支持的执行请求**；不得同时排两种，结果不可重复计数。当前A/B DEV仍0/48，DEC1 core64=0；已采用大cap不等于消费指令。

> **V4A 最新研究执行更正：** [先评估微调效果、再决定训练](docs/protocols/GR00T_WBC_V4A_EVAL_FIRST_STAGED_EXECUTION_20261024.md)。截至修订前最新原生回执（2026-10-24 16:10 SGT）：A/B4096训练完整；B0 DEV有效14/48（T1 5/7、T2 4/7）、A/B DEV均0/48，DEC1 core64=0。较大global上限121GPUh／34PBS／213CPUh／810GiB已经被人类批准并双账本一次采用，新增Eval运行/模型/物理/PBS仍为0；该上限**不是全部运行的调度授权**。当前优先冻结第一阶段：同一原DEV bank294101、A与B各完整24条件（共48个新条件，2个原噪声面板），先发布有效配对结果；旧B0完整、第二噪声bank及新增训练按证据后置。执行仍须新鲜原生资源、单owner、600s／80cycles、原安全与截止全门，不因文档自动提交；真实状态以更新的同一身份链回执为准。

> **历史V4.1小面板备选（当前未原生准入）：** 先读[小规模配对Eval计划](docs/handover/2026-10-24/SCIENCE_FIRST_MINIMUM_SPEND_EVAL_V4_1.md)，按冻结名单P0（8条件/模型）取16条新A/B闭环行为；复用8条B0，不预订130条全部预算。已批准/采用的global上限不是消耗目标。旧S2 DEV结果B0 14/48，A/B 0/48，S1 core64 0。P0诊断不能代替正式DEV48；新分段入口须先严格冷态准入，任何真实任务/资源效果为NOT_RUN直到原生回执。

S2整包已真实一次采用：[服务器＋LOCAL独立回读](reports/PI0294/design/parallel-science-20261022/V4_S2_ACTUAL_DUAL_ONCE_ADOPTION_20261024.json)。global121GPUh／34PBS／213站点CPUh／810GiB及精确B4096只读DEV例外已双账本落地；25条closed原记录、全部失败成本／训练进度、DEC1／其他限额／matrix／final／时钟均未改变。连接消失与首次非空队列只读拒绝保留，任何未知写未重发。账号另有1481491、CFP04-CF-031双GPU作业，人类已明确确认属其他独立工作；只精确区分该作业，不使用／修改／停止或计入本实验，其他未知作业仍拒绝。**无新GPU／PBS／模型／物理**；继续fresh typed native admission与独立DEV release，core64=0，B0有效14（9成功5失败）、A/B各0，goal ACTIVE。

S2整包已获人类明确批准：[精确授权](reports/PI0294/design/parallel-science-20261022/V4_S2_GLOBAL_RESOURCE_AND_SAVED_B4096_REUSE_AUTHORIZATION_20261024.json)／[冷态采用与单面板修复](reports/PI0294/design/parallel-science-20261022/V4_APPROVED_SCOPE_AND_SINGLE_PANEL_COLD_CPU_20261024.json)。仅global GPU121h／PBS34／站点CPU213h／output810GiB及精确B4096只读DEV例外；DEC1、其他限额、原matrix／final和截止保持。原25条真实closed账本字节纯采用／临时once验证通过，全部费用、训练进度及owner Exit1保持；24新增CPU、完整PV1 568／249子测试通过。成熟DEV循环只运行该请求封存的完整remaining噪声面板，保留全部有效负结果。**当前仍未真实native采用、未新reserve／PBS／模型／物理**；下一项双账本一次采用及独立回读、fresh完整准入。core64=0，B0有效14（9成功5失败），A/B DEV各0；goal ACTIVE。原保守全量上界不能在原剩余时限内容纳，未运行项保留NOT_RUN，不延时。

V4完整remaining-work重算与原global once reserve已冷接线：[CPU及真实原件复算](reports/PI0294/design/parallel-science-20261022/V4_FULL_RESOURCE_AND_ORIGINAL_ONCE_RESERVE_CPU_20261024.json)／[S2整包资源与精确B4096复用草案](reports/PI0294/design/parallel-science-20261022/V4_S2_COMPLETE_RESOURCE_AND_B4096_REUSE_SCOPE_DRAFT_20261024.json)。解析全17历史，真实global／LOCAL原字节给出paid／reserved下界，只允许精确已bound R请求的实际GPU／query／CPU／output／PBS抵扣一次；不接受caller cap、旧快照或UNKNOWN重试。原manifest／request双封存的history文件、完整项目余额／队列／queue上界／全部未来输出quota，在原owner lock＋once CAS前验证；临时成功／重复拒绝／post-intent UNKNOWN保持与只读reconcile均过。51新增CPU，完整PV1 544／249子测试通过；实际25条已结算global＋17原件复算四缺口仍为89288GPU秒／6PBS／189889.659站点CPU秒／139546445824bytes，真实reserve正确在intent前拒绝，历史capture在当前时间被拒绝。**没有真实新reserve／PBS／模型／物理或native账本写；尚未部署。**四项拟增global cap与B4096只读Exit1复用例外均仍未批准／未采用，DEC1／其他上限／原matrix与final／原全部截止保持；全保守上界不容纳于原剩余窗口，不延时或承诺全量完成。付费control credit、fresh producer与完整native admission／独立DEV release仍待；core64=0，B0有效14、A/B DEV各0/48，goal ACTIVE。

V4的H200硬编码启动阻断已冷修：[A40评估bootstrap／实际设备与CPU上限接线](reports/PI0294/design/parallel-science-20261022/V4_A40_EVALUATION_NATIVE_BOOTSTRAP_AND_DEVICE_CPU_20261024.json)。独立Vanda DEV请求走完整自有队列＋同global已绑定row，复用原始作业字段检查，不制造bootstrap admission PASS；加载前按typed receipt严格核对单张真实A40，训练／H200冒充拒绝，原H200路径仍保持。80000CPU秒accounting upper落实79760进程上限并取实际remaining较小值。50新增CPU用例；相关212／8子测试、完整PV1 493／249子测试通过；真实25条旧账本全保留的隔离临时fixture中，原三项matrix仍被PBS上限正确保护。完整17次历史已于b1bc5ac0现场回读并解析，14有效＝9成功＋5失败／3工程原件保持。**尚未部署／未提交／无新模型、物理或native账本写；不是完整资源或原生准入。**下一项为完整remaining-work fresh重算／精确credit、原global once reserve及native admission／独立DEV release；四项S2 scope未批准／未采用，B4096复用边界待。core64=0、A/B DEV各0/48；原时钟／matrix／final保护、LOCAL已用一分钟和两SDK UNKNOWN保持，goal ACTIVE。

V4完整历史原件已装配：[17次原始attempt与剩余工作实际回读](reports/PI0294/design/parallel-science-20261022/V4_COMPLETE_ORIGINAL_PRIMARY17_HISTORY_ACTUAL_20261024.json)。三个原CLOSED child逐目录库存精确17；原RESULT／adjudication／audit／CARD／model-ready共180353bytes现场只读回读、Core CPU0.009881秒，控制与关闭metadata未变。全部17通过同一逐attempt解析；14有效＝9成功＋5失败，3工程失败原件保留；两个既有replacement已有有效结果，F2/000/T1仅保留原REPLACEMENT-01资格且未派发。旧V2诊断只增加有来源的命名digest索引视图，原字段／原件不改；9项矛盾拒绝。由全部17推导B0剩余34、A/B各48、原面板10/24/24/24/24/24，共130；完整DEV48训练／正式选模仍拒绝partial，未重跑14有效条件。LOCAL一分钟已采用并消费，不重复采用；两SDK UNKNOWN、原时钟／matrix／final保护保持。**新增科学条件0、core64=0、A/B DEV各0/48；无新模型／物理／PBS／native账本写。**下一项为原global冷owner的完整fresh资源／once reserve及A40原生准入，四项S2 scope未批准／未采用、B复用边界仍待；goal ACTIVE。

V4评估冷接线新增一项已修复阻断：[原24序号硬编码的CPU修复](reports/PI0294/design/parallel-science-20261022/V4_EVALUATION_ADOPTED_PBS_NAMESPACE_COLD_FIX_CPU_20261024.json)。现有全局账本已25且精确批准到28；transport普通／FAULT namespace与current-admission查找现在读取已采用的上限，未增PBS或预算。29用例／26历史子测试通过；临时文件系统用原生完整账本原字节、未mock授权验证，25–28通过、29／31／34及伪造proposal／篡改marker拒绝。原H200／A40资格区别、真实600s／80cycle、模型／安全／截止门未放宽。**这只是序号接线，完整Vanda评估桥仍未完成／未部署**；下一项按同一成熟runtime补cold profile与A40实际作业身份，当前不提交新作业。S1 core64=0；S2 B0有效14/48，A/B DEV各0/48；四项S2全scope增补仍只是未批准草案。

V4两张主表仍未完成：[S2逐条件真实partial](reports/PI0294/design/parallel-science-20261022/S2_DEV48_ACTUAL_PARTIAL_COMPARISON_20261024.json)／[完整130条件费用上界草案](reports/PI0294/design/parallel-science-20261022/S2_DEV130_CLOSED_COST_AND_FULL_UPPER_SCOPE_DRAFT_20261024.json)。S1 core64=0，source当前mapped-prefix／ACK与target完整prefix／安装未资格。S2 B0按任务为T1已测7＝5成功＋2有效失败、T2已测7＝4成功＋3有效失败；A/B DEV各0/48，不能计算B−A或相对完整B0保持效应。服务器只读已核14份原RESULT／WORKER_RECEIPT／adjudication，9份完整32秒的最大wall890.981s／Core447.637CPU秒／输出90.367MB；这些是旧平台观测，不是A40全链吞吐。已关闭1481574的17个原TRAIN回执表明A40首次10.507s、warm中位0.306s／最大0.312s，未重跑，不能据此宣称GPU利用率或闭环资格。新鲜Vanda账本确认global25已终态／无pending-update或global unknown；原两SDK UNKNOWN字节保持。按6个原噪声面板各完整8h的保守方案，保持全部matrix／final保护后，global仅四项超额：GPU89288秒、PBS6、站点CPU189889.659秒、站点输出139546445824bytes；向上整定草案96→121GPUh、28→34PBS、160→213站点CPUh、680→810GiB，DEC1上限不改。该方案不是最少作业数／不是授权，S1＋S2上界72.15GPUh也不被当前原窗口完整容纳；原截止不延长，到期未运行项如实NOT_RUN。下一动作只做成熟评估入口的最小Vanda冷绑定与实际可容纳阶段核验，不再扩展vendor逆向，不新训练／新数据／重放旧IO。

V4已实际接续：[两条科学线与必要缺口](reports/PI0294/design/WDR1_V4_SCIENCE_FIRST_ADOPTION_AND_BOUNDED_BINDING_ACTUAL_20261024.json)。S1仍core64=0，目标完整前缀／参考安装和source当前资格未闭合；S2保留B0有效14/48（9成功、5失败），A4096／B4096训练已完成、DEV各0/48。剩余130条件需10400查询及4160000 live+shadow ticks，不能借原核心矩阵7040查询。global已用25/28 PBS，余3份全属原matrix；保护matrix86580GPU秒和最终72000后另余83872GPU秒，独立源评估完整CPU／IO／平台／提交覆盖仍须一次核清，当前不是准入。此前已派发的只读链接检查实际完成：83函数／3275字段，3251局部地址／常量或PLT-GOT名字已核对，0已检查不一致，24局部引用未解析；动态DSO实现和完整状态ABI仍未资格。停止通用链接／vendor逆向扩展，只补直接解除原生prefix接缝的必要项。LOCAL一分钟已用且原6600未来不减；两SDK UNKNOWN和全部历史保留，无新模型／物理／PBS／账本写，完整goal ACTIVE。

## 当前科学主线与长程 Goal V4（2026-10-24）

以 **[科学主线优先 Goal V4](docs/protocols/GR00T_WBC_SCIENCE_FIRST_LONG_HORIZON_GOAL_V4_20261024.md)** 作为当前优先级入口。事实复核至 **24a2bc794bcbfcb23f769a4a62696051d8e85816，13:05:24 SGT**。

| 主线 | 当前状态 | 下一科学交付 |
|---|---|---|
| 固定CORR/R0的DEC1 | core64=0；零步真实feedback／双相机／STOP已通过，WBC工作模式下prefix／安装和任务资格未完成 | 最小必要接缝及原commissioning之后，优先完整64格配对结果 |
| 源域覆盖比较 | A/B4096训练终点完成；B复用准入待闭合；B0已有14/48，新A/B完整DEV未产出 | 有覆盖即独立推进，完成匹配源域行为表；不等待目标成功 |
| 科学完成标准 | 完整有效的正向、负向或不确定结果均交付；NOT_RUN不当迁移失败 | 任务成功／拒绝／完整失败／工程缺失分开，按几何配对分析 |
| 当前资源 | 11714更新；global103148GPU秒；PBS已用25／14，原剩余matrix3；LOCAL cap57660、used51058.974682509、future6600 | 先核完整比较覆盖；源130不自动挤入原7040query封套 |
| 当前截止 | 10-25 02:29:56停新job／候选；10-25 18:29:56停新物理；10-26 02:29:56硬关闭（SGT） | V4不重置旧clock，后继第1／3／6／12周另按有效范围推进 |

既有授权内的普通必要修复直接接续；工程项只服务上述两张表。当前不新训、不扩采，不重新建设已通过的零步组件。后续扩展由结果触发。

### 以下为历史与实际回执记录

旧段落中的B2048、R、首次更新未放行及“零步尚未执行”按原时间解释，不覆盖上方当前状态。

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

# Current Science Status

## 2026-10-24 实际训练增量

同一作业 **1481574** 已获 FIRST_UPDATE 放行；03:54:24 SGT 的原生回读确认 **B3346（从2048新增1298次更新）、17次实际查询、2次加载、unknown=[]**，驻留owner和心跳正常。B4096及目标资格仍未完成，core64=0。运行胶囊36925d27未改变；冷态终局接线已独立资格且只允许真实F/Exit0和owner自然成功退出后使用。此后新增进度以同一身份链原生回执为准。

[实际首次更新和终局接线](reports/PI0294/design/parallel-science-20261022/B1481574_FIRST_UPDATES_AND_COLD_TERMINAL_CONNECTED_ACTUAL_20261024.json)。下方审核保留01:46基线及后继计划，不覆盖本段更晚的实际更新；计划不重置本轮预算和截止。

目标接线新增[同一 owner 的 quota／暂停 STOP IO](reports/PI0294/design/WDR1_TARGET_OWNED_QUOTA_PAUSED_STOP_CPU_20261024.json)，显式绑定实际冻结82004c31时钟；未知写不重发，完整状态／FSM／leg history不变且新鲜pause ACK后才返回停止回执。当前仅冷态代码与CPU资格，尚无新目标child、物理执行或任务资格；core64仍为0。

[冻结SDK实际链接输入](reports/PI0294/design/WDR1_ACTUAL_FROZEN_MPC_SDK_LINK_INPUTS_20261024.json)已定位并核对部分约束／预测函数体；它们可用于原实现薄封装的接续，尚不证明header/class ABI、完整链接来源或目标前缀安全。未重编／修改WBC，未新增目标模型、child或物理执行。

[原 SDK 薄约束读取器](reports/PI0294/design/WDR1_ORIGINAL_SDK_CONSTRAINT_CLIENT_CPU_20261024.json)已按实际 SoftBox 成本／自碰撞余量 API 接线，32项新CPU测试通过。修正了将 jointLimits 误当通用约束向量的接线假设；惩罚成本不作为安全阈值。实际SDK编译／独立客户端生命周期与目标native资格仍待完成，未修改原控制器或启动新目标物理。

[原 SDK 有限客户端及构建计划](reports/PI0294/design/WDR1_FINITE_ORIGINAL_SDK_CLIENT_AND_BUILD_PLAN_CPU_20261024.json)已完成独立二进制管道、同一进程复用、当前请求／输出绑定、编译器禁用及未知不重发接线；新增31项CPU测试，相关63项通过。真实安装SDK的编译／构造及当前DEC1双账本准入仍待完成，未新增Core child或物理运行。

## 2026-10-24 审核摘要

完整审核基线36925d27，增量复核至 **5d9f2a82a6aa4830481f7197298f666cdc4aad0c（01:46:45 SGT）**。原生读数01:45:53、回执01:46:11；本页是固定时间快照，不是实时集群监控。

| 项目 | 最后可证状态 |
|---|---|
| 源域学生确认 | N1：T1 10/20、T2 4/20，尚未达到该轮各14/20 |
| A/B | A4096完整；B完整恢复点2048，历史2970不是完整checkpoint |
| 当前作业 | Vanda1481574已转R，server-resident owner健康；MODEL_START已放行，FIRST_UPDATE未放行，未观察新增更新 |
| 失败与成本 | 保留B2970的922丢失更新及1481489 pre-CUDA失败；当前campaign9666更新，3522未保留 |
| PBS | 17/28已实际采用；已用DEC1/global14/25，余原3matrix提交位置 |
| 源终点评估 | 旧B0 14/48（9成功5失败），新A/B完整终点评估未完成 |
| 目标与主矩阵 | 目标本轮原生资格未完成；core64=0，NOT_RUN |
| 最近截止 | 10-25 02:29:56 SGT停新作业／候选；10-26 02:29:56硬收尾 |

[详细审核](docs/reviews/2026-10-24_GR00T_WBC_PROGRESS_AUDIT.md) · [长程计划V3](docs/roadmap/2026-10-24_GR00T_WBC_LONG_HORIZON_PLAN_V3.md) · [当前接续交接](docs/handover/2026-10-24/GR00T_WBC_NEXT_EXECUTION_HANDOFF.md) · [最新实际运行／启动放行回执](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/5d9f2a82a6aa4830481f7197298f666cdc4aad0c/reports/PI0294/design/parallel-science-20261022/VANDA_PRECUDA_RETRY_PBS17_28_UNIQUE_GPU_SUBMISSION_ACTUAL_20261024.json)

沿用冻结WBC／任务数据优先、Vanda-only及原科学协议。当前运行源码仍为36925d27，GitHub文档更新不热切capsule；已批准且实际采用的范围不重复索权，后继提案不自动增加计算额度。

发布并发复核：**5c9a6de763c6f74b7731809f72b2dd54e6ca47de，2026-10-24 01:59:22 SGT** 新增[1481574共享终局契约及精确身份CPU验证](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/5c9a6de763c6f74b7731809f72b2dd54e6ca47de/reports/PI0294/design/parallel-science-20261022/B_PRECUDA_RETRY1481574_SHARED_TERMINAL_CONTRACT_CPU_20261024.json)。它尚未产生原生终局审计／结算／关闭，未宣称首次更新；运行capsule仍为36925d27，上述科学与训练实际状态不变。

## 历史状态记录

以下原文保留；其中“Latest”、R/Q、未采用／未提交等字样只描述当时快照，不覆盖上方带日期的最新回执。

**Latest: N1allocation06 failed its pre-model engineering gate,not learning.**
[Exact raw-sample/resource receipt and repair](reports/PI0294/20261016T191445Z-localpc-mp1/ENGINEERING_ENDURANCE_AND_FAULT_INJECTION.json).
673223F/exit1,653GPU seconds;source36164,parent40664,reservations0.
Zero new models/NN/updates/physics.75actual owner reads spanned600.095s,
but81were required;duration-only collection under-counted real I/O overhead.
Native abort stopped the pump at72/80cycles;next checksum timed out at60s.
Repair keeps both81reads/600s,900s cap and all watchdogs;no seventh submitted.
Task matrix remains25qualified/32with sixDEVunrun and one exhausted mid-gap.

**Actual fifth terminal and first N1 cold continuation:** [paired result/resource receipt](reports/PI0294/20261016T191445Z-localpc-mp1/MATCHED_SOURCE_PAIR_FINAL.json).
673008F/exit0,increment11392GPU s,source35511settled. Matrix25/32qualified;
CORRTRAIN3/4,DEV/F0-T1 andF0-T2 individually audited32sSUCCESS. SixDEV
conditions unrun,no complete DEV rate;exhausted midpoint gap retained.
55c0610d actualCI SUCCESS,both cold native scopes pass,84old process
records closed,1899actual NN outputs accounted,13installed S1 zero-live
rejects preserved with no synthetic ACKs. Single N1job673223 submitted
auto/1H200/max8h,nativeR/released at latest readback,pre-model entry not yet
ready;zero new training,no valid-case replay.
ActualN1 600s/80cycles stillNOT_RUN;must pass before loading any model.
WBC/S1/TF-R3STOP unchanged,selected/deploymentNONE,originalN1clock preserved.

**Cold-bound N1 continuation plumbing prepared, no new job yet.**
Immutable common06-12 staging/activation and full recorded effect closure
are CPU-tested;live consumers,unknown effects or unsealed transport plans
reject advancement. Private same-owner finite adapter prepared,not launched.
506-test full suite plus21post-guard focused tests pass;native600s/80cycles
stillNOT_RUN. CORR2048DEV/F0-T1 andF0-T2 both independently audited32sSUCCESS,
zero native faults. Other six DEV conditions not run;no fullDEVrate claim.
Fifth673008 still originale418/request;cold activation awaits actual closure.

**CORR2048 TRAIN4 actual audited result: 3/4 success, T1=1/2,T2=2/2.**
[Matched endpoint task outcomes and complete raw-score/video index](reports/PI0294/20261016T191445Z-localpc-mp1/MATCHED_SOURCE_PAIR_FINAL.json).
All4 raw/video and closed fresh-input/model/action/ACK chains pass;native
safety faults0,TeacherOFF. F0/T1 rejects under originalS1at24s;the other
three complete32s successfully. BC2048TRAIN4also3/4with opposite T1layout
success;this is no overall TRAIN benefit,causalDAgger or DEV claim.
Matrix23/32qualified,CORRDEV8 inprogress,irrecoverable512F1/T1gap retained.
No newPBS or active-source mutation.245446c2 exact-head CPU CI SUCCESS;
actualN1 600s/80cycles/sitefault qualification stillNOT_RUN.

**N1 full transport producer/owner IO phase CPU-validated, native evidence pending:** [endurance/fault ledger](reports/PI0294/20261016T191445Z-localpc-mp1/ENGINEERING_ENDURANCE_AND_FAULT_INJECTION.json).
Actual-size recorded dualRGB,existing wire codec,compute checksum pump,
80fixture-only ACKs and all native fault producers now have execution paths.
Full raw journals/wire/CoreACK/process closures must join600s before load;
no inline PASS substitute.496tests/372subtests and local mature-client real
20s/EOF poison-channel bench PASS;NOT site native qualification. Existing
owner site adapter prepared,finite manager/cold effect closure still pending.
No newPBS/hot source switch;fifth673008R at CORR2048DEV8,4096updates,pending0.
Source-only task results remain separate from qualification and WBC success.

**Actual scientific milestone, not yet a complete behavior result:** [both2048 endpoints and actual matched consumption](reports/PI0294/20261016T191445Z-localpc-mp1/MATCHED_SOURCE_PAIR_FINAL.json).
CORR2048 sealed94ad4d0e,combined4096formal updates,pending0. Each arm has
65536actual visits;32768T2+16384retained originalT1cursors match exactly.
CORR16384C1visits=8306student-anchor+8078subsequent expert-state visits,
4parents/8segments,T1only. Same originalM1/recipe/frozen parameters and
2048positive-gradient records per arm verified. Fixed96expert/12historical
student outputs and actual event exposure audited,zero newNN/physics.
Expert-fit metrics are not hiddenDEV or closed-loop success. FullCORR
TRAIN4 audited/DEV8 executing;matrix23qualified/32,irrecoverable midpoint gap stays.
Fifth673008R unchanged;no sixth. Goal and nativeN1 qualification incomplete,
selected/deploymentNONE,WBC/TF-R3STOP unchanged.84e7cee1 actualCI succeeded.

**Native heartbeat producer implemented, not yet deployed or qualified:** [N1 endurance and fault ledger](reports/PI0294/20261016T191445Z-localpc-mp1/ENGINEERING_ENDURANCE_AND_FAULT_INJECTION.json).
Future premodel entry requires600s/81real post-read samples,exact same-job
native identity and immutable raw-journal join;an inline PASS cannot replace
missing/tampered raw evidence.473PI0294 tests pass locally,not native proof.
Full transport producer/router and isolated native fault evidence remain
required;no newPBS/deployment/model or physical execution. Same healthy
fifth673008R,CORR1944at snapshot;endpoint and paired science still pending.
Original data/recipe/physics/S1/WBC and MP1/N1 clocks unchanged.

**Latest native progress and reader correction:** [same fifth job still running; reader03 terminal](reports/PI0294/20261016T191445Z-localpc-mp1/N1_READONLY_READER_TERMINAL_AND_SOURCE05_PROGRESS.json).
673008R,CORR1828/2048,one legitimate in-flight update,fivePBS,original4h.
No sixth/restart/hot code change. Reader03's previous live snapshot is
superseded: actual read-only SSH child SIGPIPE(-13),empty output,no blind
retry,zero audit intents;540CPU upper settled,reserve0. Pipe cause remains
unknown. Independent Core metadata read later succeeded;not endurance.
F1/T2 midpoint replacement remains an audited S1 negative,F1/T1 spent.
Endpoint/full paired science and nativeN1 qualification remain pending.

**Latest actual control repair:** [read-only serial auditor recovered without restarting science](reports/PI0294/20261016T191445Z-localpc-mp1/N1_READONLY_SERIAL_AUDITOR_ESTALE_RECOVERY.json).
Reader01 had actualNFS116 metadata failure,not a policy or neural failure;
its one complete audit retained. Reader02 failed local import protection
before START,zero audit effects,cost settled. One isolated immutable
replacement reader03 verified live and same native673008R read after START,
CORR1605at frozen snapshot. Only12remaining fixed endpoint audits permitted;
old deadline and all valid/engineering outcomes preserved. Main science
controller/codee418bb79 unchanged,no newPBS/model/query/physics. Metadata
recovery excludes mutation retries and heartbeat writing. NativeN1endurance,
CORR2048/full paired science remain incomplete;selected/deploymentNONE.

**Latest actual science/engineering boundary:** [N1 generic admission and site receipt](reports/PI0294/20261016T191445Z-localpc-mp1/N1_GENERAL_RUNTIME_ADMISSION_CPU_AND_SITE_READBACK.json).
Original fifth673008 stillR;CORR1345 at frozen readback,not endpoint2048.
No new model or physical attempt;current e418bb79 unchanged. Public N1
runtime layers/zero-work skip/known-unstarted probe resume pass CPU tests,
including108array exact equivalence to unchanged original probe primitive.
Native600s/80cycles,qualification producer,general finite manager and cold
deployment remain pending;newN1 model load is fail-closed until actual proof.
Project auth was autonomously refreshed and independently read back valid;
prior failed query below is history,not a current request for user login.
Nativeauto/small queue supports the authorized future8h upper,current4h
job untouched. No newPBS,new method,science-budget reset or WBC change.

**Current authority/runtime boundary:** [N1 single adoption and effective cumulative ceilings](reports/PI0294/20261016T191445Z-localpc-mp1/N1_ADOPTION_AND_EFFECTIVE_LIMITS.json).
N1 control adoption03:08:01SGT Oct18;three registered sites read back the
same fixed bundle/delta without changing active manifests. Source05/673008
is healthyR and CORR training beyond512,not endpoint completion. NewN1
source36h/12PBS and parent80H200h/48A40h include old used/reserved costs;
current fifth remains old-bound4h/no-sixth until terminal. Future N1native
code/deployment/600s heartbeat/80transport qualifications NOT_RUN. New
Hopper dispatch project check failed;in-flight training unaffected. No new
PBS/query/physics by adoption. OriginalMP1 T0/S1/WBC/STOPs unchanged.
N1 cutoffsOct19 09:08:01/13:08:01/15:08:01SGT,not reset on resume.
F1/T2 sole replacement now audited originalS1 negative;F1/T1 spent gap retained.
Known T1 saved-label supervision reuse is not heldout generalization.
Older "latest" paragraphs below are preserved historical snapshots.

**Latest public evidence:** [four CORR512 actual attempt clips,partial panel](reports/PI0294/20261016T191445Z-localpc-mp1/S_CORR512_FIRST4_AUDITED_VIDEOS_PUBLIC.json).
Fixed READY9ba0fe42,1132paired frames;Core full bytes and Hopper token-free
READY/index PASS.2valid negatives+2engineering partials cover3conditions,not
completeTRAIN4. F1/T1 rights spent,F1/T2 NOT_RUN. Matrix4new+12preserved URLs,
all non-URL science unchanged,not16successes. Publisher9cf27435/402tests/CI
39055006711 does not relabel original video source03646938 or source04 runtime
8e50bcc2. Both metadata publishers terminal/reservations0,no newscience/PBS.

**Latest diagnostic:** [BC512/CORR512 actual saved pair and student identities](reports/PI0294/20261016T191445Z-localpc-mp1/SAVED_BC512_CORR512_PROBE_COMPARISON_AND_STUDENT_IDENTITIES.json).
96expert windows/8true TRAIN episodes,actual identical input/noise/precision
and recomputed metrics verified. Prefix XYZ means T1 2.703->2.817mm,T2
2.986->3.010mm;not improved descriptive means,no causal mechanism decision.
Each additional BC2048/CORR512 student set has12original E0 states verified;
36outputs across3models are only12unique states/4parents. All trajectory gates
ACCEPT do not establish S1 clearance or recovery. New-model same-state expert
label joins remain pending;four invalid initial-label exclusions retained.
Three read-only auditors terminal,4.821Hopper CPU seconds settled,reservation0,
no model/physics/PBS/update. CORR2048 and full final comparison remain pending.

**Latest submission:** [actual native672907 Q,last original source slot](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE04_SINGLE_NATIVE_SUBMISSION_AND_QUEUE.json).
One qsub,CFP04-CF-007/auto/1GPU/12CPU/128GiB/max4h;GPU unavailable,Q not R.
SourcePBS4/4,parent7/12,no automatic fifth. Source23728s/parent28228s settled
plus14400s same-job reservation;45min closeout and parent final4h protected.
Same finite controller is live with fresh heartbeat;PBS_READY/release/compute
heartbeat qualification NOT_RUN. CORR512/cursor16384 not resumed,13conditions
unattempted,spentF1/T1 and valid negatives not replayed. Original140GiB output
estimate used,no new180GiB rebinding or required deletion. Full goal incomplete.

**Previous boundary:** [actual same-root source04 cold adoption before submission](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE04_ACTUAL_COLD_BOUNDARY_ADOPTION.json).
Both native entries PASS with immutable8e50bcc2 capsules;400pytest tests and
254subtests/actualCI39052726122 PASS. Current manifest pointers cold-advanced,
old bytes sealed in history,old STOP713fef5c retained,old scopes blocked.15frozen
numerical+12learning/data/plant modules byte-identical;no new budget,T0,attempts,
GPU/PBS/model/physics. Source23728s,parent28228s,sourcePBS3/4,CORR512 unchanged.
Compute heartbeat qualification NOT_RUN;unchanged300s gate requires native
measurement before model load. Last slot04 only,no automatic fifth submission,
no thirdCORR512F1/T1 attempt or negative replay. Failed-checkpoint inventory
found no eligible large failed payload:actual deletion0/freed0,valid states kept.

**Previous terminal:** [source03 F/1,partial-effect closure and unresolved heartbeat diagnosis](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE03_TERMINAL_PARTIAL_EFFECT_CLOSURE_AND_HEARTBEAT_DIAGNOSIS.json).
Native672829:409wall/412conservativeH200seconds,26draws,0updates;finite owner
settled and exited. Source23728s,parent28228s,reservations0. CORR stays512,
cursor16384. SoleF1/T1 engineering replacement ends10.4s/5200ticks,0nativefaults;
both attempts raw-audited with18checks/full449+209frames,not qualified policy
outcomes. All26actual draw/action/ACK chains and known sequence26inbox-only
orphan close without resend;1926mutation intents COMPLETE. Original STOP
713fef5c retained on both hosts,no fourthPBS. Compute-read heartbeat value
unrecorded,NFS cause only hypothesis;8f689fe3 helper382CPU tests PASS but NOT
integrated/deployed/native-qualified. Original300s watchdog unchanged. Matrix
20attempts/18qualified/1engineering cell(exhausted)/13unattempted. SourcePBS3/4,
originalfamily caps/T0 remain;cold qualified scope still needed for final work.

**Previous queue milestone:** [native672829 submitted/Q and actual engineering qualification](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE_CONTINUATION03_ACTUAL_QUEUE_AND_ENGINEERING_QUALIFICATION.json).
SourcePBS3/4,parent6/12,1GPU/max4h/auto;not allocated or resumed at this snapshot.
Same full CORR512/cursor16384 and original03646938 scientific runtime;new cold
management1b1806d9 passed373CPU tests/CI39047585619 and both native entry checks.
Only output estimate and finite metadata recovery changed,no scientific recipe,
safety/heartbeat or original MP1 budget/T0 reset. Exact owned queued memory
225->128GB decrease read back,no cancellation or additional submission.
Latest native matrix has18verified policy cells,1engineering-incomplete cell,
13unattempted cells across all32. ParentGPU27816s settled/14400s reserved;
source23316s settled,3960CPU seconds finite-control upper remains reserved.
No original valid-negative replay;remaining CORR512/F1 and2048 still required.

**Previous allocation status:** [allocation02 F/1 and preserved exact resume](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE_ALLOCATION02_NATIVE_TERMINAL_AND_PRESERVED_RESUME.json).
CoreLab SSH loss interrupted the finite owner;Hopper enforced the unchanged
heartbeat watchdog. Both controller and read-only adjunct are terminal;CoreLab
connectivity recovered,physical workers terminal,all current reservations0.
Source GPU23316s,parent Hopper27816s settled;BC2048+CORR512=2560updates.
Two CORR512 F0 negatives are audited;F1/T1 is ENGINEERING_INCOMPLETE22.4s,
not a policy failure. F1/T2 andCORR2048 remain pending,original budgets intact.
[Failed-checkpoint check](reports/PI0294/20261016T191445Z-localpc-mp1/FAILED_CHECKPOINT_RETIREMENT_CHECK_01.json):
three registered storage areas inspected,no eligible large failed payload,
actual deletion0/freed0;required parents/fixed models/exact resume preserved.
No newPBS or model execution in this reconciliation. Prior milestones below
do not mean the old allocation is still running. Final goal remains incomplete.

**CORR512 actual learning and matched consumption:** [sealed state and first audited negative](reports/PI0294/20261016T191445Z-localpc-mp1/S_CORR512_ACTUAL_MATCHED_CONSUMPTION_AND_FIRST_VALID_NEGATIVE.json).
512actual updates,537 changed trainable tensors,frozen tensors unchanged;
full state31f0514b. Both initial1030tensor inventories equal originalM1 and
actual recipes match. Each arm16384actual visits,T1/T2=8192each;all8192T2
cursors and4096retained originalT1cursors match exactly.4096CORR T1correction
visits come from4student parents/8segments,including2062student-anchor and
2034later expert-state visits,not4096independent demonstrations. F0/T1 is
validS1 refusal1.6s:19raw checks/33decoded pairs/5actual draws/4ACKs+1zero-live
rejection PASS,native faults0. Other fixed conditions remain in progress.
352tests/CI39043547480 PASS;consumption audit terminal1.397CPU seconds,
reservation0. No newPBS,budget/clock/recipe/runtime change or candidate chosen.

**Current mainline:** [complete BC2048 endpoint and independent CORR start](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC2048_COMPLETE_DEV8_AND_CORR_INDEPENDENT_START.json).
BC TRAIN3/4,exposedDEV3/8:T1DEV2/4,T2DEV1/4 over four development groups.
DEV has four originalS1 pre-execution refusals and one safe full32s object
goal-zone miss,not five executed collisions. Zero engineering gaps/native
faults,TeacherOFF.148 raw checks,3360 decoded camera pairs,423 actual draws,
419 executed ACKs and4 zero-live rejected prefixes independently verified.
[All eight actual DEV clips are now public](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC2048_ALL_DEV8_VIDEOS_PUBLIC.json)
at fixed READY448ad9de,with Core full-byte and independent Hopper token-free
READY/index readbacks PASS. Earlier four TRAIN clips stay at READY53d71ad9:
all12 BC endpoint videos,including every negative,are available without any
old-prefix reupload or task-outcome change. No failed condition was repeated.
[Full32-cell behavior matrix](reports/PI0294/20261016T191445Z-localpc-mp1/BEHAVIOR_AND_FAILURE_MATRIX.json)
joins actual Core/Hopper registries,episode IDs,checkpoints,raw audits and
chains:16verifiedBC conditions plus1verifiedCORR negative,15pending conditions,
not a final matched comparison. The earlier77update snapshot is superseded
by the actual512proof above. Pure aggregator85c4a013 is not deployed into
the immutable03646938 science capsule. Native672471 is now terminal as recorded
above;no newPBS or resource-clock reset,actual GPU settlement is complete.
[Serial raw-audit harvesting and closed-session transport measurements](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE_ALLOCATION02_SERIAL_AUDIT_AND_TRANSPORT_MEASUREMENTS.json)
are real,read-only and separately accounted. Residual wall time is not measured
neural inference or guaranteed recoverable GPU idle. CORR endpoint/exposure,
mechanism decision and final resource closure remain. TF-R3STOP,VandaNOT_RUN,
WBC/S1 and selected/deploymentNONE stay unchanged.

**B2 terminal BC evidence:** [actual2048 raw-event consumption](reports/PI0294/20261016T191445Z-localpc-mp1/BC2048_FINAL_RAW_EVENT_EXPOSURE.json).
All65536 visits join126 verified raw timelines. Sampled T1prefix8 covers
close96/96,open95/96,withdraw95/96;all events have H40 supervision. T2detach
andwithdraw cover30/30 each despite DEV1/4,so missing event supervision alone
is not established as the mechanism. No error/safety margin inferred from
coverage,no sampler change. Auditor terminal20.477CPU seconds,reservation0,
0newqueries/physics. CORR512 paired exposure now passes;terminal2048 remains.

## Previous Complete TRAIN4 And Video Milestone

**Current mainline:** [complete S-BC2048 TRAIN4 and actual public videos](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC2048_COMPLETE_TRAIN4_AND_PUBLIC_VIDEOS.json).
3SUCCESS/1VALID_TASK_FAIL/0engineering gaps/0native faults:both T2 layouts and
F0/T1 complete32s/41of41 hold,whileF1/T1 is originalS1 rejection at1.2s.
All73 raw checks,1948 actual paired frames,244 fresh native draws and243ACKs
plus1zero-live rejection PASS. Four conditions/two seen TRAIN layouts,notDEV.
All four success/negative videos ACTUALLY HF-public at fixed READY53d71ad9,
Core full-byte readback and Hopper public unauthenticated READY/index PASS.
Eight fixed512 anchors join original actual inputs,cameras,arrays,weights,
precision and native-hook noise fingerprints;0newqueries/physics. Original raw
noise tensors not saved;no invented rehash. Same-state endpoint comparison
still waits CORR seal and separate original NI/shadow scope.334 CPU/CI PASS.
Native672471/same finite owner now evaluates exposedDEV8;CORR from original
M1 remains pending. No newPBS,budget/T0/WBC/S1/TF-R3 change,selectedNONE.

## Previous FirstT2 Success And Static Restore Milestone

**Current mainline:** [S-BC2048 F0 succeeds at both tasks;F1/T1 is a valid negative](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC2048_FIRST_T2_SUCCESS_AND_ZERO_STEP_ANCHOR_RESTORE.json).
F0/T2 has independent18-check raw/video and80 fresh observation/generation/ACK
chains PASS:32s/641pairs,15.685cm push,3903 contact ticks,zero bilateral grasp,
41/41 hold and zeroS1/native faults,TeacherOFF. Same fixed2048 model also has
the previously auditedF0/T1 success. F1/T1 instead rejects before execution
at1.2s:19 raw checks,3ACKs+1 rejected prefix,zero native faults;valid negative
preserved. F1/T2 active at snapshot,DEV8/CORR pending,not completeTRAIN4.
Eight current512 saved full-state/history anchors actually restore with max
difference1.78e-15,no integration/query;static-only,not dynamic-prefix parity.
182.450 chargedCPU seconds/8 conservative starts,reservations0,B1 lease stays
CLOSED. Initial interpreter import gap preserved,same-bound existing-runtime
repair passed.330 tests/CI pass. Native672471/finite owner live,auth valid,
no newPBS,T0/recipe/WBC/S1/TF-R3STOP unchanged,selected/deploymentNONE.

## Previous First2048 Success Milestone

**Current mainline:** [S-BC2048 sealed / first autonomous F0T1 audited SUCCESS](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC2048_TRAINING_AND_FIRST_AUTONOMOUS_SUCCESS.json).
Complete2048 checkpointac0f4692 with full state;independent three-partition
audit65536 actual samples,T1/T2 each32768,126 original TRAIN episodes,C10.
The417 final resumed updates have finite positive gradients,unchanged493
frozen tensors,537 changed trainable tensors and original final scheduler.
F0/T1 completed32s,641 pairs,41/41 hold,lift8.136cm/carry16.978cm,zeroS1/native
faults,teacherOFF.18 checks,full641-frame video and80 fresh actual model/ACK
chains PASS. One seen layout only;remainingTRAIN3,DEV8 andCORR still pending.
Paired96-window/8-episode expert probe comparison is now strict512-to2048,
not legacyM1:identical raw input/noise/precision verified,prefix8 XYZ means
2.703/2.986mm to2.221/2.506mm. Not generalization or unique mechanism proof.
Mainline672471 still R,original finite owner/recipe/reserves intact;do not
restart2048 training or initializeCORR fromBC. Original512 valid negatives
andB1 newly verified table/pad shadows remain immutable;target STOP stays.

## Earlier Current512 Failure Attribution

**Current512 failure attribution:** [exact original history and first geometry](reports/PI0294/20261016T191445Z-localpc-mp1/CURRENT_SBC512_FAILURE_HISTORY_FIRST_GEOMETRY.json).
Four current S-BC failures now reproduce all27600 actual ticks with zero
highrate/control/reference difference and original64/124/63/150-tick rejects.
First contact is right_pad/table forF0T1/F0T2/F1T1,left_pad/table forF1T2.
All contacts are predicted shadows;original native faults0 and valid S1
negatives remain. No oldM1 cause substituted,no safety/plant/WBC change.
Eight current-policy-induced noninitial anchors saved full live/IK/controller/
teacher history,not justqpos/qvel. Original unseen RNG fingerprint not claimed;
independent snapshot restore/noise-receipt joins remain before new model draws.
B1 child is CLOSED,194.195 charged CPU seconds/28001 shadow ticks/4 conservative
diagnostic starts,zero new live conditions/queries/PBS/GPU and reservations0.
326 tests/native CI PASS. Source672471 remains independently active;fixed pair
and final review deliverables still incomplete,no optional intervention chosen.

## Earlier Mainline Resume And Event Snapshot

**Current native boundary:** [actual full-state resume and raw event audit](reports/PI0294/20261016T191445Z-localpc-mp1/RAW_EVENT_EXPOSURE_AND_ACTUAL_SOURCE_RESUME.json).
672471 is R,observed1743/cursor55776;native full1631 resume has first
microbatch52192/gradient1631/optimizer1632 and continuing scheduler. No2048
claim yet,512 valid negatives not repeated,CORR and DEV pending.
B2 actual raw-command/event joins cover126 TRAIN episodes and completed
512/1631 traces. T1 at512 misses close/open/withdraw events in sampled first8
for27/26/36 of96 episodes,but all events are in some sampled fullH40 target.
T2 detach first8 coverage30/30,withdraw29/30;do not call supervision absent.
Per-episode counts/hashes and96 expert probe strata are published;open-event
gripper errors illustrate why low overall XYZ is insufficient,not causal proof.
Audit is terminal20.136 CPU seconds,zero reservations/new inference/physics.
Original1200s CPU/wall family preserved through actual-schema repair,320 CPU
tests and native CI PASS. B1 exact-history reconstruction and current first
geom remain pending;no new24-query or sampler branch self-authorized.
Original MP1 T0/reserves/fixed endpoint,WBC and TF-R3 STOP remain unchanged.

## Previous Continuation Submission Snapshot (Before Native R)

**Previous native boundary:** [B0 saved-label join / source continuation02](reports/PI0294/20261016T191445Z-localpc-mp1/B0_SAME_STATE_LABEL_AUDIT_AND_SOURCE_CONTINUATION02.json).
Project authentication restored and actual CFP04-CF-007 membership passed.
672471 is Q,one H200/max4h/native128GB,not training yet;same native request
had one owner-only queued memory decrease from site-rewritten225GB. Source
PBS2/4,MP1 H200 actual16295s/reserved14400s,original clock/recipe retained.
Same-owner finite controller will resume full1631 state and handle remaining
fixed panels without repeating the four valid512 negatives;no2048 claim yet.
B0 all12 historical E0 draws joined,8 qualified noninitial expert H40 labels,
4 initial exclusions and two true parents/task. Prefix8 mean XYZ T1=6.95mm,
T2=12.55mm;these are action errors,not RMSE,tracking or causal proof. Same
input pixels/history/ACK commands independently checked;S-BC S1 NOT_RUN and
clearance NOT_MEASURED remain. Zero new inference/physics;private16-file HF
delivery verified on both hosts. Both CPU workers terminal,reservations0.
311 CPU tests/native CI success. B1/B2,BC/CORR endpoints,DEV and all four final
review deliverables remain incomplete. Runtime03646938/WBC/TF-R3 unchanged.

## Previous Native Boundary (Before Auth And Continuation02)

**Previous native boundary:** [adopted review /1631 partial /12 existing student
draws](reports/PI0294/20261016T191445Z-localpc-mp1/REVIEW_ADOPTION_STUDENT_PROBES_AND_ALLOCATION01_PARTIAL.json).
672080 isF/0,11795 conservative H200 seconds,not currently training. Complete
state at1631/cursor52192 and exact two-session consumption passed independent
audit; T1/T2=26096 each,correction0. Both child leases remain ACTIVE but no
workers or GPU/physics reservations remain. Next original-budget submission
awaits fresh project authentication (last query15006); no second source PBS.
Reviewfb16d101 is adopted without changingruntime03646938,clock or recipe.
The12 historical E0 student draw identities all passed; actual S-BC S1 and
qualified same-state expert-error joins are NOT_COMPARABLE/pending,not filled
from old-model evidence. Four new failures' geom and actual event joins await
bounded diagnosis. Fixed pair/terminal evaluation remains the mainline.

## Earlier Source Pair Snapshots (Not Current Native State)

**Latest fixed panel and real continuation:** [S-BC512 TRAIN4 complete / actual
full-state resume](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC512_FULL_TRAIN4_AND_RESUME_AUDIT_ENGINEERING.json).
S-BC512 has0/4 task successes: four valid original-S1 failures at
1.6/20.8/10.4/22.4s,zero native faults and zero engineering-missing rows.
All four independent raw audits,1108-frame video decodes and142 fresh
observation/native-generation chains passed,including138 executed ACKs and
four rejected zero-live prefixes. These valid negatives will not be repeated.
Same-job exact resume is actually at515/cursor16480; first resumed cursor16384,
finite gradients and continuing original scheduler are verified. Fixed2048
criterion,recipe and budget stay; no full2048,CORR/C1,DEV or target result yet.
The read-only resume-partition auditor passed289 CPU tests and real512
single-partition readback; real multiple-session audit awaits completed state.
672080 is R,runtime03646938 unchanged. Vanda camera stays CLOSED NOT_RUN.
Original MP1 clocks/caps and TF-R3 STOP persist. selected/deployment NONE.

**Independent saved-probe evidence:** [S-BC512 expert-state audit and legacy
parent gap](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC512_SAVED_PROBE_AUDIT_AND_PARENT_PROVENANCE_GAP.json).
All96 windows from8 TRAIN episodes have exact raw inputs,noise,weight hashes
and independently recomputed metrics. Prefix8 XYZ means T1/T2=2.70/2.99mm;
all96 trajectory-gate ACCEPTs are not S1 or live task success. Four T2 H40
windows have spurious gripper transitions,only one inside the executed8-point
prefix; these do not uniquely explain live failure. Strict M1 comparison is
NOT_PROVEN: its first legacy NPZ lacks actual input fingerprint fields.
No resampling or fake metadata; M1's prior task successes remain untouched.
Both CPU auditors are terminal,7.017 CPU seconds total under their original
600-second family bound. Native training observed668/cursor21376 continues.

## Earlier MP1 Snapshots (Not Current Native State)

**Live source / terminal camera:** [actual learning and original-window camera closeout](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE_LEARNING_STARTED_AND_VANDA_CAMERA_QUEUE_CLOSED.json).
Source672080 is R with real S-BC updates and verified balanced committed sample
prefix; CORR/new policy outcomes are not available yet. Camera1463728 expired
without allocation,was withdrawn at that boundary and is CLOSED/NOT_RUN,zero
GPU and zero reservations. Source learning continues independently of target.

**Recovery videos delivered:** [fixed public HF byte verification and queued-memory delta](reports/PI0294/20261016T191445Z-localpc-mp1/PUBLIC_RECOVERY_VIDEOS_AND_QUEUE_MEMORY_DELTA.json).
All ten attempts/3405 decoded paired frames are indexed,including failures and
two initial-only clips. Same source job672080 remains Q; memory-only request
was lowered225GB to128GB with native readback,no newPBS or training recipe change.

**Current mainline:** [matched Source Pair first native submission](reports/PI0294/20261016T191445Z-localpc-mp1/MATCHED_SOURCE_PAIR_FIRST_SUBMISSION.json),
job672080 Q, CFP04-CF-007/auto, one H200/max4h; no training result yet.
Both bindings passed, fixed parent/C1 and task-balanced sampling retained.
This is a new MP1 source study, not reopening E1 or the closed M2 recovery.

**Recovery-family final:** [eight-condition actual panel and resources](reports/PI0294/20261016T191445Z-localpc-mp1/RECOVERY_FINAL_ACTUAL_PANEL_AND_RESOURCES.json).
CORR seen-TRAIN T1 is 2/2 successful scenes; BC has one success and one
engineering-missing T1. Fixed total: 3 SUCCESS, 2 valid original-S1 failures,
2 engineering-missing, 1 NOT_RUN. No complete T2 or DEV/WBC success.
All ten attempts independently audited; 240 complete fresh observation/ACK
chains and 480 actual model-input camera hashes verified for three successes.
671899 is F/0; all three recovery children are CLOSED, 4,500 conservative
H200 seconds, zero updates/reservations. No fourth recovery PBS. Read-side
metadata assistance remains explicit; original MP1 budget/T0 and TF-R3 STOP
persist. Vanda camera qualification remains separately queued, not PASS.

## Historical Progress Snapshots

Observed 2026-10-17 after E1 and both MP1 recovery-child lease closures by the existing single local-PC owner.
This is an actual-state summary, not a replacement for immutable run receipts.

**Second independently verified complete model case:** [M2-CORR512 F0/T1](reports/PI0294/20261016T191445Z-localpc-mp1/M2_CORR_FIRST_COMPLETE_T1_AND_RAW_AUDIT.json)
also completed32s,641 pairs,41/41 hold,0 S1/native faults,18 raw checks and80
cross-host actual ACK chains. This oldC0 model's success on F0 and BC's F1
success do not establish a paired gain, freshC1 learning or generalization.
Remaining CORR evaluation is still running under the single7200s allocation.

**Latest actual qualification:** [original M2-BC512 F1/T1 full32s SUCCESS](reports/PI0294/20261016T191445Z-localpc-mp1/M2_BC_COMPLETE_T1_AND_CLOSED_LOOP_QUALIFICATION.json)
has641 camera pairs,41/41 hold,0 S1/native failures,18 independent raw checks,
and80 cross-host observation/generation/actual ACK chains with160 input-image
hashes matched. P0 is qualified for this explicitly metadata-reader-assisted
paused-sim transport, not unassisted/real-time performance. Scientific runtime
remained4d430556. Other M2 conditions remain in progress; no complete T2, DEV,
generalization, correction gain or target result is implied.

**Subsequent actual third-run progress:** [first 29.35s engineering partial and metadata observers](reports/PI0294/20261016T191445Z-localpc-mp1/THIRD_RECOVERY_FIRST_PARTIAL_AND_DIRECTORY_OBSERVERS.json).
671899 is R; first BC F0-T2 stopped at the original 900s wall bound, not a
complete32s success despite 28/28 observed hold checks. Next BC F1-T1 is running.
Directory-only CPU observers have explicit cache observer effects and separate
sealed sources; no model/action/physics/watchdog code was hot changed. No P0
PASS or unassisted end-to-end performance claim is established by this snapshot.

**Latest direct human scope revision:** [third recovery-family submission](reports/PI0294/20261016T191445Z-localpc-mp1/THIRD_RECOVERY_ADDITIONAL_GO_AND_NATIVE_SUBMISSION.json)
is explicitly permitted once, at most two hours, without changing MP1 total
budget/T0/deadline or prior attempt counts. Child03 passed both native entries;
671899 is Q, 7,200 GPU seconds reserved, no model loaded yet. BC F0-T1 is
excluded as exhausted; BC3/CORR4 remain eligible. Previous children01/02 stay
CLOSED and their engineering conclusions/costs remain. No fourth submission.

**Later independent native admission:** [sprint adoption and camera binding](reports/PI0294/20261016T191445Z-localpc-mp1/SPRINT_ADOPTION_AND_VANDA_CAMERA_BINDING.json)
records a new camera-only A40 job1463728, currently Q (site node_pool capacity).
This does not reopen the two closed recovery children or TF-R3 physical STOP.
No new model/physics/WBC work has run; 900 GPU seconds and 3,600 CPU seconds
are reserved until native terminal reconciliation. Runtime EGL remains unproven.

## Active Authority

**MP1 is the only new incremental resource family.** The user directly adopted
release `c8cdf60e41e31da5796b44a064ef2f22d594a491`.
The concurrently published E2 documents remain non-additive references; there
is no E2 owner, second budget or 28-day clock.

- MP1 run: `20261016T191445Z-localpc-mp1`.
- T0: 2026-10-17 03:14:45 SGT. New submissions/candidates stop October 23
  03:14:45; new experiments/training blocks stop October 23 19:14:45;
  hard closeout October 24 03:14:45 SGT. Restarts do not extend these times.
- New upper bounds: 48 H200 GPU-hours, 32 A40 GPU-hours, 12 new GPU PBS
  submissions per site, 24,576 optimizer updates. Native site limits win.
- Old E1 and Vanda jobs retain their original capsules, budgets and leases.
  Their reservations are not spendable MP1 balance.
- [Actual adoption and three-host readback](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json).
- [Fixed execution entry](docs/handover/2026-10-17/MAINLINE_ACCELERATION_MP1_EXECUTE.md).

MP1 has completed CPU-only source-loader qualification and one Vanda native
build, then actually ran recovery jobs 671817 F/1 and 671826 F/0 using the
existing M2 weights with zero optimizer updates. Both recovery children are
now CLOSED. Total conservative H200 cost is 453 seconds; four actual starts,
600 live plus 600 shadow integrations, five generations, three executed ACKs,
two completed draws not installed, pending/reservations zero. All four raw
evidence audits passed, but there are zero valid full policy results:
three engineering-incomplete conditions and five NOT_RUN across eight fixed
conditions. No 32s relay qualification or M2 behavioral gain is established.

The first native failure was a heartbeat-file read in the recovery entry;
the mature bounded control reader was reused in a new immutable child.
The next attempt actually located a 30.301976s stall inside Hopper RPC ledger
accounting, exceeding the unchanged 20s watchdog. Lock wait versus
read/write/fsync is not yet separated. The branch is stopped, with both of
its two PBS opportunities used; no third equivalent recovery submission.
The controller also advanced to an initial-only CORR case before the owner
stopped the cross-model branch; this charged deviation remains in the matrix.
[Actual phase timings](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_ACTUAL_RPC_PHASE_TIMING.json),
[eight-condition matrix and four audited attempts](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_E1_M2_RECOVERY_BEHAVIOR_MATRIX.json),
[real qualification failure](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_REAL_RELAY_QUALIFICATION.json),
[closed child resources, not whole-MP1 closure](reports/PI0294/20261016T191445Z-localpc-mp1/E1_RECOVERY_RESOURCE_FINAL.json).
Continue independent saved-T2 diagnostics and separately qualified Vanda
work under the original MP1 clock. Do not train new weights to bypass this
evaluation fault. Fresh permission and capacity remain actual admission checks.

The subsequent registered CPU-only saved-T2 audit actually completed with
zero new physics, model queries or training. Both original M1 T2 trajectories
already entered the unchanged all-corners goal zone (F0 16.70s / F1 16.15s).
Last saved TCP-object distances were 0.04690m / 0.04240m, below the original
0.08m clearance requirement; objects remained in-zone and nearly stationary.
This supports investigating post-push disengagement, not declaring complete
T2 success or a unique collision cause. Original scores recomputed exactly.
That first saved-only receipt left the illegal geom unknown; the subsequent
separately admitted exact-history shadow result below now resolves it.
No new autonomous negative episode was run.
CPU was conservatively charged at its 600s process cap, not advertised as
exact whole-process usage. [Actual saved-state/action audit](reports/PI0294/20261016T191445Z-localpc-mp1/M1_T2_SAVED_PROGRESS_AND_ACTION_AUDIT.json).

**Exact-history T2 attribution is now complete, not just prepared.** Child
`saved-m1-t2-shadow-01` reconstructed all 18,000 saved historical integrations
with zero maximum difference, including the original static IK history.
The two model prefixes reproduced their original 56/52-tick S1 rejections:
F0 table-left_pad and table-right_pad; F1 table-right_pad. Maximum forbidden
contact overlaps were 0.03726mm / 0.01903mm. These trigger the original
any-arm-table-contact rule, not the separate 4mm penetration threshold.
Both same-state expert and expert-XYZ-only 0.4s prefixes passed; SO3-only
and gripper-only substitutions still rejected. This distinguishes the tested
Cartesian path, not a unique Z bias or permission to calibrate/relax S1.
The audit-only expert remains in approach history; short-prefix PASS is not
qualified full T2 Recovery, a model-intent label or new correction supervision.
[Actual geom/component receipt](reports/PI0294/20261016T191445Z-localpc-mp1/M1_T2_FIRST_REJECTION_ATTRIBUTION.json)
and [event/contact figure](reports/PI0294/20261016T191445Z-localpc-mp1/M1_T2_TIMELINE_AND_CONTACT_DIAGNOSTIC.png).

The child is CLOSED with worker terminal, original input hashes unchanged
and reservations zero. New shadow cost: 19,123 integrations (18,000 exact
history + 1,123 diagnostic), Core CPU meter 189.659s including the 180s
management allowance; new GPU/PBS/model/optimizer/live/autonomous-eval zero.
Its two shadow reconstructions were conservatively charged to the MP1 source
start allowance. MP1 source-budget starts are now 6 (4 actual new evaluation
attempts + 2 shadow reconstructions), total integrations 20,323, H200 453s;
the original MP1 clock and target TF-R3 STOP remain unchanged.

**Original T2 TRAIN30 event coverage has also been actually audited.** All
30 admitted demonstrations / 30 distinct scene groups (F0/F1/F2 = 8/11/11,
not 30 OOD categories) contain every original push stage. Their 18,030
overlapping H40 anchors include detach 756 (4.19%), withdraw 1,500 (8.32%),
and 1,926 cross-stage windows containing detach. No window lies entirely in
detach because those segments last only 1.10-1.35s, shorter than H40's 2s;
this is not missing detach supervision. Raw stages exactly match the frozen
manifest buckets. All 640 supervised commands plus the unlabeled terminal
observation were explicitly distinguished and original input hashes rechecked.
[Actual episode/event/window coverage](reports/PI0294/20261016T191445Z-localpc-mp1/T2_TRAIN30_EVENT_AND_H40_COVERAGE.json).

The fixed Source Pair sampler forecast gives 1,324 detach + 2,678 withdraw
visits among 32,768 planned T2 visits per branch. This is a deterministic
future schedule, not actual training consumption or evidence of sufficiency.
No S-EVENT trigger or main recipe change follows automatically from these
counts. One CPU audit failed its initial 640-versus-641 observation check;
the bounded explicit-terminal repair passed. Both workers are terminal,
charged at 600 CPU seconds each as conservative process upper bounds,
reservations zero; new physics/neural/training/GPU/PBS zero.

**The measured RPC accounting hot path has a scoped code repair.** Runtime
`4fda127a` keeps successful-reply telemetry in memory and settles RPC count
plus process CPU once at exit; neural/optimizer/physics charges, journals,
guards and original 20/60/900s contracts stay durable and unchanged.
Poisoned channels now publish a separate engineering stop on both child
roots; native recovery/Source Pair guards reject it before the next model.
An existing marker prevents new panel binding. This is not a native safety
STOP and does not relabel previous policy or engineering outcomes.

Both registered Core/Hopper CPU probes actually held a synthetic ledger lock
in another process while emitting 80 memory replies, then made one final
shared-ledger transaction. All workers and lock holders are terminal, and
uncertain/repeated settlement is rejected. Native probes use no real model,
neural/physics counter or RPC wire; they prove this telemetry-lock isolation,
not actual SSH/NFS/32s qualification or the cross-model model-stop behavior.
Heartbeat and guard reads still do I/O; complete-loop speedup remains unknown.
A killed session requires registered CPU-bound reconciliation, not zero cost.
[Implementation, actual probes and limits](reports/PI0294/20261016T191445Z-localpc-mp1/RPC_REPLY_LOCK_ISOLATION_CPU_RECEIPT.json).

The first CPU probe on each site failed before lock setup on a Path expression;
the explicit role-entry test repair passed on the sole bounded CPU retry.
All four attempts are terminal and conservatively charged 600s/site/attempt
(1,200s per site), reservations zero. Native queue recheck finds no own
PI0294 running/queued GPU job; old recovery leases remain CLOSED, 453 H200s
unchanged, no third recovery PBS. A new legal end-to-end route is still required
before claiming P0 or a completed M2 behavior comparison.

**Campus connectivity is restored and the old E1 native states are reconciled.**
Job 670729 is F/exit 1 after the original owner-heartbeat timeout, not a
model/safety result. Both old jobs are terminal: 16,363 conservative allocated
GPU seconds charged, their old reservation released only after native proof.
The final third E1 job **671655** is now F/exit 0. All three allocations
are natively reconciled: 19,675 conservative GPU seconds charged and zero
GPU reservation. No fourth E1 submission is permitted even with unused
seconds. All source/neural/controller/stdio workers were terminal and
physical reservations zero before the separate CPU artifact-closeout stage.
Both E1 leases and registry entries are now
CLOSED after native/resource/artifact readback. E1 is a bounded closed
execution with explicit scientific comparison gaps, not a successful full
M2 behavioral comparison or a reopened resource window.
[Final resources and closed leases](reports/PI0294/20261016T150228Z-localpc-e1/RESOURCE_FINAL.json).
[Complete execution summary](reports/PI0294/20261016T150228Z-localpc-e1/E1_RUN_SUMMARY.md).
[Actual final allocation, training and consumed exposure](reports/PI0294/20261016T150228Z-localpc-e1/LAST_NATIVE_TERMINAL_AND_ACTUAL_M2_EXPOSURE.json).
[Restored-network reconciliation and exact final allocation](reports/PI0294/20261016T150228Z-localpc-e1/NETWORK_RESTORED_AND_LAST_NATIVE_ALLOCATION.json).
The preceding interruption remains preserved:
[Actual interruption and resumption order](reports/PI0294/20261016T191445Z-localpc-mp1/LOCAL_NETWORK_INTERRUPTION_01.json).

Offline E1 checks repaired the last-allocation controller's middle-hop cache
path and bounded pipe-write timeout, plus final condition/attempt reporting.
All 228 CPU contracts and a local OS-pipe controller fixture passed; these
are not native pipeline qualification or a measured speedup. The saved M1
T1 audit metadata still reconciles to two seen-TRAIN successes, not new runs.
[Checks and exact native readmission preconditions](reports/PI0294/20261016T150228Z-localpc-e1/LAST_ALLOCATION_ENGINEERING_CHECK_01.json).
No third E1 submission, native source switch, new MP1 GPU or physical action
was made during the disconnected interval. That historical network block
is resolved, not deleted or treated as completion.
[Blocker and explicit completion gaps](reports/PI0294/20261016T150228Z-localpc-e1/OWNER_BLOCKED_NETWORK_AUDIT_01.json).
Original clocks and total caps remain unchanged. The final immutable scope
retains all M1 valid outcomes, reuses completed M2-BC weights without training,
and admits three settled engineering replacements plus the unstarted fourth
BC condition and fixed C0 M2-CORR branch. It does not hot-insert C1 data.

**BC recovery did not produce a valid comparison.** Its first replacement
stopped after 6 simulated seconds on a 20-second file-RPC timeout; a late
response was then incorrectly reused on the same channel, causing the next
three conditions to fail response-ID checks before physical integration.
All four remain engineering failures, not valid policy negatives. No more
BC replays are made in E1. Both SSH file endpoints subsequently exited.
CORR completed its fixed 512 updates/16,384 samples in the same allocation,
with 537 changed trainable tensors, frozen parameters unchanged, and 512
finite positive gradient records. Its first autonomous case stopped at
10.8 simulated seconds on another original 20-second RPC timeout, with
zero S1 rejection/native fault. The poisoned-channel latch then prevented
the other three physical cases from starting: NOT_RUN, not three failures.
No valid complete M2 comparison exists. Both model weights are preserved.
The single local controller was replaced only at the settled BC/active
CORR-training boundary, using the same owner lock and original deadlines.
An uncertain channel now latches invalid before any later physical case.
The finite direct-open metadata helper actually ran and ended, costing
1.160 CPU seconds, no model queries or physics. It did not establish a
complete pipeline repair; original timeouts and scientific sources stayed
unchanged. Independent main-evidence audit accounts for all 22 actual
attempts, including three zero-integration engineering attempts. Audit
PASS is not task success or complete policy-outcome coverage.
[Actual failure chain, timing caveats and controller continuity](reports/PI0294/20261016T150228Z-localpc-e1/BC_REMAINDER_ENGINEERING_FAILURE_AND_CORR_CONTINUITY.json).

## Source Results

**M1@1024 autonomously completed T1 in both predeclared seen TRAIN layouts.**
Both runs lasted the full 32 seconds with Teacher OFF, original D0/S1/scorer,
no native safety fault, 41/41 terminal hold observations, and independent
raw/action/ACK/contact/score/video audit PASS. T2 was validly rejected by the
original S1 in both layouts. This is not DEV, Go2/WBC or real-robot success.

- [Complete M1 TRAIN4 and independent audits](reports/PI0294/20261016T150228Z-localpc-e1/M1_1024_COMPLETE_TRAIN4_AND_T1_AUDITS.json).
- Terminal E1 H200 job `670729.hopper-m-02` completed M2-BC's actual 512 updates,
  with 537 changed trainable tensors and 512 finite nonzero gradient records.
  Its three attempted evaluations are engineering-incomplete, not task outcomes;
  the first reached 25.6 simulated seconds before the original 900-second
  relay limit. M2-CORR training is now complete but its autonomous panel is
  engineering-incomplete. Both branches have the fixed M1@1024 parent and
  their previously sealed data choices; valid outcomes were not rerun.
- Actual consumed traces verify BC T1/T2=8192/8192 and CORR=12288/4096.
  CORR consumed 8192 original and 8192 C0 samples, from six correction
  segments/two independent student parents. Equal updates are not
  task-matched exposure; no causal correction-gain claim is supported.
- All 12 predeclared terminal-model conditions are explicit in the
  [actual behavior matrix](reports/PI0294/20261016T150228Z-localpc-e1/MODEL_BEHAVIOR_MATRIX.json):
  four valid M1 policy outcomes, five engineering-incomplete conditions,
  three NOT_RUN conditions. All 22 actual attempts have accounted evidence,
  but not all have valid task results. The six fixed probe contexts now have
  648 original-prefix S1 shadow checks, without new neural queries.
- On the noninitial fixed student states, CORR T1 prefix8 XYZ error is
  2.19 mm versus M1 6.90 mm, with original S1 prefix execution 6/6 versus
  4/6 across all fixed student states. These overlap C0's same two TRAIN
  parents and do not prove independent generalization or autonomous gain.
  [Actual student-state diagnostics](reports/PI0294/20261016T150228Z-localpc-e1/ON_STUDENT_DIAGNOSTIC_COMPARISON.json).
- [Saved M1 T2 object progress and original final rejection](reports/PI0294/20261016T150228Z-localpc-e1/M1_T2_SAVED_REJECTION_AND_OBJECT_PROGRESS.json):
  actual horizontal motion 13.28/14.61 cm before original S1 rejection,
  without lifting; no complete task success. That historical saved-only
  receipt did not know the specific illegal geom pair and did not inherit it
  from M2048. The separately charged MP1 shadow receipt above now resolves
  those pairs; the old E1 evidence and result remain unchanged.
- E1 C0: 833 full H40 windows from two E0 student episodes, T1 only. It is
  state aggregation, not new M1 on-policy DAgger.
- Four real M1 collection episodes are terminal. Two required qualified
  expert takeover after actual student prefixes; two produced no correction.
  Saved-data repair produced 832 new windows, 1665 including C0, with
  independent source and actual Hopper processor/loader checks passing.
- [Repaired C1 fixed revision and actual consumption](reports/PI0294/20261016T150228Z-localpc-e1/C1_INTERVENTION_REPAIR_ACTUAL_DELIVERY.json).
- **E1 M2 still uses C0**, frozen before the repaired C1 became ready.
  The new C1 is available to a later explicitly bound MP1 branch, not hot
  inserted into E1. T2 Recovery remains unqualified (full recovery 0/2).

The original M1@256 precision comparability gap, one disclosed unintended
repeat, invalid initial fork histories, old E0 negatives and all costs remain
preserved. No claim of a flawless full E1 comparison is made.

All four fixed inference checkpoints and 38 decoded main/fork clips now
have immutable public HF revisions and READY readback. The private upload
hit quota and its empty prefix was reconciled; the private repository was
not made public. Only two Cosmos metadata paths were sanitized; weights
and statistics stayed unchanged, with complete upstream license notices.
Public payloads exclude optimizer/RNG and private paths/assets. M1@1024
and both M2 full native training states remain on Hopper; M1@256 was
inference-only originally. Public inference/resume qualification is not
claimed. [Actual fixed artifact index](reports/PI0294/20261016T150228Z-localpc-e1/ARTIFACT_DELIVERY.json).

## Target Platform

Native Vanda refresh: hardware job `1462438` and saved-input inference job
`1462451` are F/exit 0, with CLOSED leases. Native allocated GPU walltime is
58 and 137 seconds respectively, charged to their old scope, not MP1.
The former passed actual A40 CUDA checks; the latter loaded old M2048 and
made 17 saved-observation queries (one warmup, 16 timed), median 0.2356s,
peak allocated 13.35 GB. This is neither a live loop nor H200/A40 numerical
equivalence, and does not qualify M1/M2 or native EGL rendering.
[Original native terminal receipts](reports/PI0294/20261016T191445Z-localpc-mp1/PRIOR_VANDA_ENGINEERING_NATIVE_TERMINALS.json).
Failed old builds are preserved.
MP1 CPU build `1463204` is F/exit 0: container tests, native compilation,
before/after controller-source checks, and linked-library checks passed.
The simulator and two controller binary hashes are recorded in the
[actual build receipt](reports/PI0294/20261016T191445Z-localpc-mp1/VANDA_CPU_NATIVE_BUILD_ACTUAL.json).
The build used 1,533 native CPU seconds and zero GPU, physics, model queries
or WBC launches. It does not establish runtime safety or task success.
Healthy queued/held jobs were not cancelled. Container/package network
downloads were not pre-metered: the ledger retains an explicit pending
audit and 32 GiB conservative reservation, not a measured zero-transfer claim.

The old TF-R3 safety STOP is unchanged. MP1 permits only evidence-backed,
bounded owned task-interface requalification, not WBC/plant/PD/safety changes
or a cross-host bypass. Source research does not wait for target construction.

## Next Admitted Work

After E1 closure, GitHub CI at d30354e5 failed four fixtures because pytest
loaded a second module alias and string-based mocks patched the other alias.
Tests now patch their actual imported module objects. Remote CI at
0218b645 passed; default local pytest is 827 PASS/14 explicit SKIP, PI0292
108 PASS and PI0294 243 PASS. No native scientific capsule was hot patched.

An earlier MP1 engineering action actually ran on both hosts: 96 same-host
synthetic packet/heartbeat/ledger filesystem cycles each, with no model,
PBS or physical start. The maximum component latency was 15.74ms Core
and 24.71ms Hopper. No 20s stall was reproduced; this is not a cross-node
NFS, SSH or 32s closed-loop test. Direct open is not a proven speedup.
Guard/invoke/accounting/output phase tracing was then used by the two real
recovery children and located the accounting stall described above, without
watchdog or mutation-retry changes.
Initial CPU entry/path errors and a pre-dispatch immutable-request collision
are preserved; failed cost is conservatively charged, not silently zeroed.
All probe processes are terminal, reservations settled, E1 still CLOSED.
[Actual timings, limitations, CI and costs](reports/PI0294/20261016T191445Z-localpc-mp1/ACTUAL_RELAY_CPU_DIAGNOSTIC_01.json).

E1 cannot submit again or reopen its closed leases. Continue within the
already adopted MP1. Both existing-model recovery PBS slots are now spent;
the failed shared relay branch is stopped, not queued for a third equivalent
attempt. A future usable evaluation route still requires actual qualification,
not another same-host smoke or training new weights to bypass the fault.
No valid M1 outcome is replayed.
Then bind S-BC2048 on original v3.0 TRAIN126 and the fixed M1@1024 parent;
use the matched S-CORR branch from qualified frozen correction data. Preserve 1:1
task exposure, including when only T1 has correction. Actual TRAIN126 has
96 T1 and 30 T2 episodes; the task-first sampler preserves 1:1 exposure.
The real source-reader and exact-cursor-resume CPU checks passed. With only
T1 correction qualified, S-CORR uses 50:50 original/correction within T1,
giving 25% correction overall, not a task-unbalanced 50% overall mixture.
Next Vanda work is runtime qualification, not another compilation-only claim.
No new DEV labels, WBC tuning or BASE padding.

The user additionally permits HF public storage for new owned artifacts.
This does not make existing private repositories public or grant rights to
redistribute private WBC/vendor assets. Credentials stay private; publication
requires sanitization and license review and does not block legal local use.
Historical selected remains NONE; real-robot deployment remains NONE.
# Latest Actual Source04 Terminal and Direct Fifth-Slot Authority

Source04 native672907 is F/1,381wall/391conservative seconds;source24119,
parent28619H200seconds,reservations0. CORR remains512/cursor16384,combined
updates2560. F1/T2 engineering partial8.4s/4200ticks/169paired frames passes
18raw checks,not a policy result.21ACKs,22draws,one known uninstalled action,
1970journal entries closed;both engineering STOP bytes preserved.
Observed negative67.9ms heartbeat age identifies a pre-read clock race.
Post-read repair07b04707 passes406tests/CI,keeps original0..300s predicate;
native sustained qualification remains pending. Direct human reply authorizes
sourcePBS4->5 only,one extra<=4h within original12H200h/T0/MP1 caps and1536
remaining updates. No sixth,no F1/T1 third attempt,no valid-negative replay.
Cold binding and single submission are NOT_YET_EXECUTED.
[Exact terminal and audit receipt](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE04_ACTUAL_TERMINAL_AND_CLOCK_REPAIR.json).
[Exact additional-slot card](reports/PI0294/design/SOURCE_PAIR_ADDITIONAL_PBS05_AUTHORIZATION_20261018.json).
Matrix now21attempts/18qualified cells/2engineering cells/12unattempted,
16prior public clips retained. Previous source04Q snapshots are historical.
Full matched pair remains incomplete;TF-R3STOP,VandaCLOSED_NOT_RUN,selectedNONE.

# Latest Actual Fifth Submission:673008 Q

Human-authorized fifth job submitted exactly once,CFP04-CF-007/auto/1GPU/max4h.
SourcePBS5/5,parent8/12,no sixth. Same finite owner live. Both native cold
entry checks PASS at e418bb79,560source members,15numerical/12learning-data-
plant modules unchanged;both prior STOPs and old histories retained. Original
birthstage4 remains byte-identical,effective5 requires sealed direct human card.
416tests/274subtests and actualCI39057571128 PASS,ten exact blob readbacks.
Source24119s/parent28619s settled plus14400s reserved,not actual GPU use.
CORR512/cursor16384 not yet resumed,remaining1536updates. Q is not native
sustained heartbeat qualification or task completion. Only original F1/T2
replacement;no F1/T1 third,no valid-negative replay,no T0/budget/S1/WBC reset.
[Actual cold and single native submission receipt](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE05_ACTUAL_COLD_AND_SINGLE_NATIVE_SUBMISSION.json).
Full paired science incomplete;TF-R3STOP,VandaCLOSED_NOT_RUN,selectedNONE.

# Latest Actual R and Historical Student-Label Comparison

Fifth673008 is actually R. Forced compute/server heartbeat qualification before
model load PASS with real post-read clocks,original0..300s gate unchanged.
CORR512F1/T2 original sole replacement is executing real prefixes;its terminal
raw/chain audit and sustained transport outcome are still pending. SourcePBS5/5,
no sixth,remaining1536updates,no F1/T1 third or valid-negative replay.
[Actual native admission/prefix snapshot](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE05_NATIVE_ADMISSION_AND_REAL_PREFIXES_IN_PROGRESS.json).
[Actual three-model same-state saved-label comparison](reports/PI0294/20261016T191445Z-localpc-mp1/THREE_MODEL_HISTORICAL_STUDENT_LABEL_COMPARISON.json):
12historical states/4parents,8qualified labels/4invalid initial exclusions.
At matched512,T1 prefix XYZ macro BC6.954->CORR2.502mm;T2 XYZ12.550->11.793mm,
gripper0.2675->0.2726. Only2TRAIN parents/task,C1 overlap unresolved,not heldout
generalization,full recovery,live S1 clearance or a final paired comparison.
BC2048 is descriptive at a different update count. New four CPU-only workers
terminal,8.0704actual CPU seconds charged once,2400reserve cleared;all16private
HF members/model verified on Core. No newNN/physics/teacher labels/model-weight
downloads. Auditf29f748a passes418tests/276subtests/actualCI39058058261;running
sourcee418bb79 unchanged. CORR2048/full pair/mechanism incomplete,selectedNONE.

# Latest Actual CORR Resume and Audited F1/T2 Negative

673008R resumed exact originalCORR512 fullstate. Fixed observed progress533,
cursor17056;optimizer/scheduler/RNG/sampler/stats preserved,stop2048 unchanged.
F1/T2 sole replacement is valid originalS1 failure12s/6000ticks/241pairs,
19raw checks and31draws/30ACKs/1zero-live rejected prefix verified,native faults0.
Original8.4s engineering partial retained,no further F1/T2 retry;F1/T1 no third.
Matrix22attempts/19qualified cells/1engineering gap/12unattempted finalcells.
[Exact audit and resumed formal-update receipt](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE05_CORR512_F1_T2_AUDITED_AND_EXACT_TRAINING_RESUME.json).
HistoricalT1probe parent overlap with actualC1 consumption now verified2/2,
3115of4096correction visits at512;T2overlap0/2. Error improvement is explicitly
not heldout generalization;exact anchor/window training overlap not asserted.
Current sourcePBS5/5/no sixth,originalT0/caps/45min closeout unchanged.
Concurrent N1 docs preserved by ordinary merge,expanded budget not native-bound
or consumed by current healthy job. Full pair/mechanism incomplete,selectedNONE.

# Latest CORR512 Raw Events and Supervision Reuse

[Actual event and executed-label reuse audit](reports/PI0294/20261016T191445Z-localpc-mp1/CORR512_ACTUAL_EVENT_EXPOSURE_AND_SUPERVISION_REUSE.json)
joins16384committed visits,4096C1=2062student-anchor+2034expert-after states.
Four historicalT1qualified expert-label windows exactly recur in C1 and were
consumed2060times;T1probe error gain is training-supervision fit,not heldout
generalization. Whole model processor tensor byte identity is not claimed.
OriginalT1prefix-close/open/withdraw uncovered52/49/57 versusBC51227/26/36;
coverage changes alone do not prove a sampler bug or license a recipe change.
All30T2episode event/visit summaries match BC512 exactly. C1stage/detach/
withdraw remain unknown,not zero.96saved CORRexpert draws join raw-event
strata;no newmodel/query/physics/teacher labels. Two workers terminal,
17.5037metered CPU seconds,one shared180s local upper,1800CPU reserves cleared.
Audit83405c41 passes426tests/285subtests/actualCI39059634999;runninge418bb79
and healthy673008 unchanged. CORR2048/full paired final still pending.



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
