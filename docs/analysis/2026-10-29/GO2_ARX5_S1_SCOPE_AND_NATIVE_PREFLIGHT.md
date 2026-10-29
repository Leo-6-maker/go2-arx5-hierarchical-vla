# S1批准、十天计划衔接与实际冷态核对

**2026-10-30后继实际：恢复01已关闭并安全隔离。** [两次首次结果及独立核账](../../../reports/PI0294/design/parallel-science-20261022/GO2_ARX5_S1_RECOVERY01_TWO_ACTUAL_FAILURES_CLOSED_20261030.json)优先于下文准备时点。T1实际执行首个Prefix8/0.4秒，存8对RGB；第二次查询因已执行旋转历史误用列解码而拒绝。实际保存状态用正确行解码，原1051点证书通过，最大关节速率0.225rad/s、加速度0.332rad/s²；没有新物理或改原阈值。T2初始化第24步真实FL小腿越界，未运行任务，停止原件与隔离标记保留；其控制在第4步与相同初态的T1分歧，原因未定。累计5启动/7736计费步/8对RGB；新scope/lease及旧run均关闭，0条合格新示范，不重试。46项CPU检查通过，冷修未部署。

**后继授权：** 用户已直接批准[两次S1恢复范围](../../../reports/PI0294/design/parallel-science-20261022/GO2_ARX5_S1_TWO_TASK_RECOVERY_HUMAN_APPROVAL_20261029.json)。T1/T2各32秒＋最多5秒初始化，无重试；累计上限5启动/42012计费步/1282对RGB。旧关闭账本与lease精确hash承接至新immutable run，旧run不重开，原CPU/存储/传输/截止不变。独立审计核对80个已提交/安装H40、640个执行Prefix8区间、18500个原始状态及641对RGB，沿用原评分；47项定向CPU检查通过，实际两次仍未运行。

**当前以[实际开发终态](../../../reports/PI0294/design/parallel-science-20261022/GO2_ARX5_S1_FINITE_DEVELOPMENT_CLOSED_INTENT_SERIALIZATION_ACTUAL_20261029.json)为准。** 原完整胶囊已经实际采用；三次启动包括预约前身份错误、T1在固定暂停处误等旧控制日志、T2在首条意图日志处NumPy布尔值编码失败。累计实际5000积分/计费5006，任务图像0。T2原生安装/取消/HOLD/WBC回切已通过；发布调用曾发生但接收端无接受回执、无执行任务区间，原失败不升级为成功。own scope/lease已关闭，条件确认均不合格。修复先持久记录完整意图再发布；同一真实保存状态已验证，未再运行物理。[新的两任务恢复卡](../../../reports/PI0294/design/parallel-science-20261022/GO2_ARX5_S1_TWO_TASK_ENGINEERING_RECOVERY_PROPOSAL_20261029.json)原待批状态已由上方精确授权接续。下文入口资格、0/4与未部署文字保留其准备时点，不代表当前运行状态。

**最新原生结果：** [原标准入口本地资格已实际通过](../../../reports/PI0294/design/parallel-science-20261022/GO2_ARX5_S1_ORIGINAL_ENTRY_NATIVE_QUALIFICATION_ACTUAL_20261029.json)。使用已接受main `2c9ad348`内的固定运行源`20911ba3`；原三文件字节不变。原canonical registry仅追加本次资格角色，移除该角色后可逐字节重建原registry hash；没有创建第二登记系统。原onceIO登记、实际解释器/cwd探针、关闭资格lease均完成，20个原件全文独立回读，后续只补保守成本结算且其余19原件hash不变。入口实际CPU0.0703核秒；准备总保守Core80＋PC120核秒、传输1,336,576字节，均计入S1原批准上限，不增加新额度。

这是**入口与工作目录资格**，其源码胶囊只含本次管理依赖，不冒充完整Teacher运行胶囊。物理仍0/4，RGB、GR00T、训练、PBS均0。后继`task_path_worker.py`完整EEF10工作器和`task_path_stage.py`冷态采用程序已实现但未原生启动；其新scope lease/源绑定、全链本地输入及实际ACK/反馈/STOP/安全资格仍需完成。原已关闭资格lease不重开。

## 新原生工作器接线（冷态代码，尚无新物理结果）

- `task_path_teacher.py`固定T1负30度、T2正30度几何选择，从真实home/反馈开始；每个新运动阶段至多一次固定seed294/20初值/2秒RRT，单attempt至多6次。20Hz观察、每8观察一H40；新阶段保留全局序号和已执行区间，普通查询不重新锚定阶段路径。T1/T2真实物体/接触谓词换阶段，不按时间强制进入抓取或推送。
- `task_path_model.py`把抓持/推送运动假设仅施于私有规划MjData。抓持假设需要实际native_grasp；推送假设需要真实未抓持pad接触。实际机器人、物体与标签不被这些假设改写。每个H40保留原500Hz局部IK/碰撞及20Hz关节速度0.7rad/s、加速度1rad/s²门，并把最近两个实际执行的参考点纳入衔接检查；不修补/clip被拒的值。
- 新worker复用原完整机器人、冻结sim/controller/solver、原有限夹爪和记录生命周期。首个已预约启动内，5秒预热后在step2500原地检验当前反馈、hold参考安装、cancel/原生HOLD、WBC重入；该检验额外积分0，不另开第五个物理启动。ROS参考从序号2接续，夹爪独立从序号0开始，符合原严格包序列。首对实际RGB使用本episode第0帧，保存成功后才推进任务，不额外消耗RGB对数。
- 原task.info的`recompileLibraries true`保持。原63,084字节/30文件codegen seed已从旧真实solver日志定位并只读核查；solver/controller各用本次输出下独立缓存复制件，保留原生成行为，不写旧cache，不改WBC数学代码/权重/gain/cost/solver。现有独立planner环境已实际导入NumPy1.26.4、MPlib0.2.1和原runtime的MuJoCo3.2.7，无安装/环境修改。
- 原标准三文件原样用于新的不可变完整runtime子布局；旧两文件资格胶囊保留。有限冷程序只通过原registry/onceIO更新本run自己的角色、原scope准备费用及新lease，不能启动sim/worker。源及私有输入完整封存，控制/日志/缓存/工作目录均检查节点本地。每attempt使用原用户级有限进程单元，800秒运行、10秒停止、KillMode control-group、最多4核；原共享native互斥锁继续使用，旧系列/费用/UNKNOWN不动。
- 4张card事前固定：SHORT T1/T2均source[.45,-.08]→goal[.45,.08]、yaw0、20秒；CONFIRM T1/T2均source[.455,-.07]→goal[.455,.09]、yaw.03、32秒；都另计5秒预热。确认只在同任务短开发完整性及四项里程碑（至少2cm接近进展、安全precontact、实际接触、T1连续0.3秒真实抓持或T2接触下至少1cm未抓举位移）独立复核通过后进入。确认不是正式泛化TEST，不合并为新总体成功率。
- 内部构造器完整预测由64修正为96，含冷态副本及固定阶段规划；这是原共享构造器余量内的工程预测，不是增加人类批准的4启动/62,012积分/2,084RGB或CPU/传输/存储额度。实际全14维仍须在采用前按新原件重新核对。

当前28项CPU检查通过（原23项及5项新阶段/假设/衔接检查）。新检查器在原保存的完整机器人状态上，两段初始H40各1001个500Hz点通过，CPU约1.185秒、积分0；这些仍不能证明动态WBC执行。下一步为固定源码发布后的实际冷采用、独立回读和首个S1短开发。新整机成功示范仍0。

**后继实际：** [完整14维重算与原入口封存方案](../../../reports/PI0294/design/parallel-science-20261022/GO2_ARX5_S1_ENTRY_AND_FULL14_PREPARATION_20261029.json)已用新原生全文hash/大小核对复用的完整GLOBAL/LOCAL字节，再调用原`first512_forecast`保留所有旧工作及matrix/final。原传输池仅余26,335字节；用户批准的是新S1最多8GiB传输，作为仅S1可用的追加范围单独记账，不用它偿还旧工作或扩大十天计划。旧14维义务均在原cap内，加入该S1范围后无预测缺口，尚未实际预约。总4CPU核时分为Core最多13,800核秒＋PC最多600核秒，未双重增加PC额度。

已定位同字节的本地XML候选。原`enter.sh`、`doctor.py`和完整profile配置可原样放进新run的`entry/infrastructure`，其既有相对路径规则自然把工作目录解析到`entry/code/vla`；无需重写入口或改变全局profile。新增组件只准备这一封存布局、验证原registry精确readback及真实挂载类型。有限入口资格程序仅调用原入口的解释器/工作目录探针，通过原onceIO登记原canonical registry并关闭自己的资格lease；不能启动Teacher、solver、simulator或模型。实际执行和完整原生worker资格另行核证。

两次完整原件只读请求因回复上限终止，未重发原操作；原CPU/传输上界全额保留，只有这两条已知只读操作的状态经原onceIO核账，其他UNKNOWN未动。随后用小型全文身份读数核对已有完整缓存，避免再次传回大原件。

用户已直接回复“批准该 S1 有限开发范围”。[授权原件](../../../reports/PI0294/design/parallel-science-20261022/GO2_ARX5_S1_HUMAN_APPROVAL_20261029.json)固定原PR #143提案及全部上界，不重复申请同一许可。登记采用为2026-10-29 19:31:39.836045 SGT，24小时窗口于10月30日同一时刻结束；原stop-new、硬截止、历史费用和保护资源保持。

PR #144的十天计划已从main `672e1858369ee3b4ca9ebd725800a4dd3f15ef67`正常合入本工作分支。S1批准的4次上界只是TD06的一部分，最多2次完整确认不等于其8条件资格。十天新增训练、查询、容量和11月延期仍未采用。TD00/TD06标RUNNING仅表示本页的实际只读核对，未声称已启动物理。

## 当前原生事实

[SCOPE_AND_CAPACITY.json](../../../reports/PI0294/design/ten-day-policy-20261029/SCOPE_AND_CAPACITY.json)记录各自观察时刻与原件hash。Core当时没有匹配的目标物理进程或GPU计算进程，但旧task lease仍写ACTIVE且已过旧截止；未重开、删除或把它当新权限。D25独立读数同样显示服务不存在/主PID为0，而原lease仍ACTIVE；不能仅凭GPU空闲宣布原owner资源已释放。

原代码、解释器、模型、native库、clock/pause二进制均能在节点本地定位。旧配置的入口、registry、lease和XML资产仍指向共享存储。用已有ROS/ELF绑定顺序重新做只读解析后，三项冻结程序全部无缺库，107个实际解析库位于ext4；九个Python模块实际导入成功，CPU 0.196秒。默认未绑定环境下的ldd缺库不等于需要安装软件。本次没有安装、编译、基础设施修改、运行迁移或原生账本写。

下一步是冷态绑定新scope的节点本地入口/profile、已登记输出与本地控制/lease/onceIO，以及精确原资产。继续复用原registry和owner；不新建竞争登记系统。完整14维余额及保护资源尚未在本收据闭合，不能声明native admission PASS。

## 更晚Teacher结果必须覆盖陈旧“正在运行”摘要

前面的12次清单只覆盖三个早期系列，不是全部目标历史。已复用的[SDR3独立接触复算](../../../reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/source-demo-access-v3/TASK_CONTACT_RECONCILIATION.json)确认285个受力tick、累计0.57秒、只有右pad；没有抓持或成功，原关节加速度失败保留。因此不能概括为所有版本“从未接触”。

通过原registry定位TF-R3并读回四份原RESULT：midpoint、reactive V1、E1均为规划失败；E2运行源码`6af24da77cc11951366acd07c75939c9648a7fdb`在18,847积分处原ILLEGAL_CONTACT安全停止，654观测/82安装intent。旧REACTIVE_PROGRESS中的“E1正在运行”已被其真实终态覆盖。原安全失败版本不重跑，完整任务成功仍0；不把这四条与早期不同人口合并算策略成功率。

## 四段新参考的保存状态复核

复用已有TF-R3的500Hz Cartesian插值、最多5步局部IK及原0.7rad/s、1rad/s²界限，检查四个已经封存的候选，不新增规划或物理。使用独立MjData，base固定在保存构型，仅用于运动学复核。

| 参考 | 时长s | 500Hz点数 | 最大关节速度rad/s | 最大关节加速度rad/s² |
| --- | ---: | ---: | ---: | ---: |
| T1接近 | 6.15 | 3076 | 0.22435 | 0.31497 |
| T2接近 | 11.20 | 5601 | 0.19987 | 0.27555 |
| T1下降 | 2.30 | 1151 | 0.22802 | 0.46707 |
| T2下降 | 2.75 | 1376 | 0.27209 | 0.46269 |

四段均通过保存构型下的全模型接触/限位复核，CPU共5.936秒；[逐段首结果](../../../reports/PI0294/design/parallel-science-20261022/GO2_ARX5_DENSE_SAVED_REFERENCE_CHECK_20261029.json)保留。两任务接近都使用原V2 W0/T1实际首态；下降起点是明确的未来计划构型，不是新反馈。此结果排除了这些固定参考在该保存构型下的原速率/局部IK问题，不能证明自由机身动态跟踪、原生安装消费、真实接触或成功。新物理启动、GR00T调用、训练和PBS均0，任务资格仍待实际验证。
