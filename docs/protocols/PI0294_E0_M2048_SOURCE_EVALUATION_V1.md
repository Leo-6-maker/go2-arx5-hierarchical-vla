# E0：M2048 固定来源域行为评价

日期：2026-10-16，Asia/Singapore。
状态：**READY_FOR_EXPLICIT_ADOPTION / PROPOSED_NOT_LAUNCHED**。
`authorizes_execution=false`；当前文档发布不是人类采用，也不是服务器启动。
研究基线：`b408ee2882c9d517557be8aa0236719163475f40`。
机器卡：`reports/PI0294/design/E0_M2048_SOURCE_EVALUATION_V1.json`。

## 1. 唯一目标与结果边界

**只评价已有 M2048：先原固定源域 TRAIN4，随后同模型 DEV8。** 不训练、不选新checkpoint、不改变v3.0、EEF10、原D0或S1，不增加BASE标签，不开展目标整机迁移、Teacher开发或人类视频学习。

本轮要回答：M2048是否学到了源域T1抓放/T2推送，以及失败发生在生成、预执行拒绝、接触、操作或最终物体结果的哪一层。完成评价可以得到12条有效负结果；**E0完成不等于policy成功**。

用户报告已有本地五项准备材料：源绑定、行为矩阵、目标数据卡、动作契约、迁移计划。发布助手未访问用户本机这些字节，不宣称逐项重新验收。现有owner应直接复用原件，提交脱敏公开副本或其已公开依据索引，私有绑定保持私有，不重新编写五份同名方案。

## 2. 仓库事实与准确身份

- M2048使用原v3.0，从M512完整续训至2048更新，累计消费65536个TRAIN样本；本轮不做任何optimizer更新。
- 原R3面板包含M512 TRAIN4、M2048 TRAIN4、M2048 DEV8，共16条件，实际全部NOT_RUN/UNOBSERVED。E0只补其中M2048的12条，不把原16条改为已执行，不补跑M512来扩大范围。
- S1资格版本 `PI0294_R3_D0_S1_COPYDATA_V1`，资格source `90c90d784dd702f10c79c719d717e80aeb50e31e`；已有四条完整Teacher通过及八个clone context通过。资格是source D0上的特权仿真预检，不是整机WBC资格。
- 资格文件：`reports/PI0294/20261014T121640Z-incoming-pi0294-r3/S1_TEACHER_QUALIFIED.json`；规范文件同目录`S1_SPEC.json`。
- M2048固定交付文件同目录`HF_PUBLIC_M2048_RECEIPT.json`：公开repo `leoyuer/pi0294-public-artifacts`，prefix `PI0294/20261014T121640Z-incoming-pi0294-r3/public-M2048-0449992fddbc`，最终revision `4c3097652bb81e43ea40a2b0b423086ec6ea7091`；实际全内容回读revision `8331f51bf0765ffae290347c78782f9c4f2ab10c`。不同revision的用途分开，不根据HEAD猜模型。
- 以上固定报告可复用；主机文件是否存在、近期运行/锁是否占用仍须owner现场读取。已发表Teacher失败或过期Teacher资源不能被本卡继承成E0权限。

## 3. 一次采用，一次绑定，不再另造准备循环

需要用户明确采用本E0计算包；研究路线认可与本条准备汇报本身不推导GPU/PBS/物理许可。由现役唯一local-pc owner记录一次`E0_ADOPTION_AND_BINDING.json`，包括采用原文、固定release、源数据/模型/processor/D0/S1的hash、原面板ID与随机表、新资源窗及独占关系。

只读/CPU准备与运行授权分离，但不要再次重建旧统计、重新收集数据或全量历史恢复测试。五份准备文件若有私有路径，仅其脱敏副本、hash和逻辑asset ID进入GitHub。不得上传credentials、raw RGB、机器人资产或模型权重。

本轮固定来源域D0，不调用最新fullbody Teacher worker、不导入其夹爪/时参/起点采样变更。计划commit、实际runtime source、资格source三者分别登记，不以最新分支HEAD自动替换数值实现。

## 4. 精确评价面板

TRAIN4复用原F0/000与F1/000，每组T1/T2配对；S1资格对应初态hash分别为：
- F0：`2a6b2ea49bd823c321f049b3c115350f95f0e935d3c432c03fecfb02925f28f5`
- F1：`b74d0f1ed334ba5d6bcccceaedeaea165d16e1f8042fde55908be048933583cb`

DEV8复用原已封存的四个DEV scene-group各T1/T2，具体ID、初态与随机表从已准备私有面板和原run capsule绑定。本文不猜测DEV编号。首次模型生成前发布实际12条件映射的脱敏manifest；无法恢复原面板时先报告确切缺项，不能用方便的新seed替代。

同组初态、视觉和物体条件一致，语言不同。固定M2048、原推理采样/denoise配置与原随机规则；每条件一次有效科学评价，不多抽动作噪声挑成功。查询失败的恢复不得将一次已返回动作重新抽样并择优。

**TRAIN4 -> DEV8的条件是执行链可信且无未解决的真实安全/完整性故障，不是TRAIN必须成功。**
- 0/4真实任务成功、模型动作被原S1正确拒绝、超时或未接触：均可形成有效负结果；原链健康时继续DEV8。
- actual live安全违规、S1副本/主仿真不一致、错误权重/processor/坐标/帧时钟：停止受影响的执行并保存证据，不把它写成普通task失败后继续冒险。
- 工程同语义缺陷可修复、测试、在episode边界封存新wrapper；旧有效结果不重跑。若修订改变D0/S1/动作语义，本E0不自动授权新科学配置，不能混合结果。

## 5. 运行顺序与最低工程工作

### P0：提交PBS之前

复用统一入口、用户已有五项材料、现有双SSH新观测中继。核对自有任务/lease及其他Teacher任务，不取消、kill或抢占它们。源D0与整机Teacher是不同执行路径，由同owner协调CPU/输出/端口。

确认推理所需文件已在Hopper本地或有可校验的传输来源；检查原stats、相机、EEF坐标、rot6d、夹爪语义、H40时间和prefix8的既有回归。

**这不是训练恢复。** 只加载模型权重、config、processor、原stats以及必需的本地tokenizer/backbone资产；optimizer/scheduler/sampler文件保留但不加载，不做恢复训练smoke。无需为推理重下约25.54GB的完整训练状态；缺少资产时只从同一固定revision补齐实际推理需要的成员，逐文件核验。不能改用网络latest或新模型弥补缺件。根statistics与processor/statistics分别按收据核对，字节hash不同不自动表示语义不同，也不允许重拟合强制对齐。

S1/D0/physics/依赖完全匹配即可复用已有资格；可运行保存状态/CPU契约回归，不强制重跑整套四条Teacher。若现场需要source环境部署smoke，本E0预留最多T1/T2各一条Teacher检查，单列非模型结果；不得将其作为完整新S1版本资格的替代。身份不匹配且无法恢复原实现时停止，不迁就latest。

### P1：单一resident推理服务

Hopper获得合法allocation后只加载一次M2048，使用原数值推理设置、eval/inference模式，不创建优化器。以固定观测作最多32次接口/离线数值查询，包括需要的序列、输出finite与维度、原解码及无重复相对变换；这些不是任务成功。无需再跑此前48条无关smoke。

CoreLab source D0读取真实双RGB与实际EEF/夹爪，经现有local-pc双SSH中继给Hopper，取H40，经原SafetyGate/S1检查后仅执行前8点；每个新模型请求绑定实际新的观测，不能离线复播Teacher观测伪装closed-loop。

原S1只能接受或拒绝，不修改模型动作、不重抽噪声、不接管为Teacher。记录原始H40、安装/前缀执行、影子拒绝理由与物体结果。神经resident与结果落盘分离，远端归档失败不能中断合格本地评价。

### P2：TRAIN4然后DEV8

32秒、20Hz、H40、prefix8保持原定义：完整条件640条执行命令、641观测、最多80次在线模型查询、16000主积分。影子预检的仿真积分另计。以原S1条件早停的轨迹保留原始长度，不能伪补完整视频或虚构640条命令。

TRAIN4结束写一次中期矩阵，按上节规则自动进入DEV8，不再问例行GO，不因暂时0成功而在本轮改模型/提示词/噪声/阈值。不同表现用于下一轮研究决策，不用于本轮调参。

## 6. 每条必须交付的行为证据

一行对应一个固定条件：checkpoint、scene-group/split/任务、初态hash、动作随机性、attempt身份、是否开始、是否得到可判定任务结果、实际时长、生成次数、实际接触/物体位移或抓举、释放/分离/保持、原S1/参考拒绝、真实安全、首次失败阶段、视频私有索引和原件hash。

区分五类终局：`SUCCESS`、`VALID_TASK_FAIL`（含模型导致的原预检拒绝）、`ACTUAL_SAFETY_FAIL`、`ENGINEERING_INCOMPLETE`、`NOT_RUN`。模型输出非法但原安全链正确拒绝与公共接口损坏分开；任务尾部未执行不强记为完整32秒。实际安全失败计入该条件失败并触发执行停止；系统性工程中断不当成0成功率样本。

主要报告T1/T2及TRAIN/DEV各自的 successes/科学可判定数，同时另报固定计划12条的覆盖数；若12条齐备，TRAIN分母4、DEV分母8。未运行不填0，拒绝不从有效策略分母删掉。还应报告H40/raw到installed、installed到actual、actual到物体结果的层次，以判断学习不足还是执行接口问题。

不声称真实部署、整机迁移、BASE能力、独立确认或未见技能泛化。结果名称包含`M2048 + SOURCE_D0 + S1_PRIVILEGED_SIM_PREFLIGHT`；单次开发面板不是部署可靠性估计。selected/deployment仍NONE。

## 7. 新E0资源提议：避免排队耗尽原实验时钟

以下是待采用的新资源包，不取Teacher授权或旧过期余额，也不恢复后来取消的Teacher累计小门。

- 一次记录`T_adopt`。排队接纳窗截至`T_adopt+24h`；不得在截止后启动新allocation。健康且仍在窗内的E0排队不重复qsub。
- 以实际进入RUNNING计allocated GPU用量。最多1张H200同时、总计6 allocated GPU-hours、最多2次PBS（第2次仅同协议工程恢复/剩余未评条件，不能重复有效失败）。首作业建议申请不超过4小时；备作业walltime不超过剩余总GPU额度，并受站点真实限制。
- 将预备工作完成在qsub前；RUNNING时两端重新核验健康和锁。CoreLab源端不可用时保存状态并尽早结束自有推理allocation，不能驻留空耗配额。
- 24小时排队接纳窗到期，若E0自有queued作业仍未开始，按采用卡停止其提交并撤回该自有queued作业，记录`QUEUE_NOT_RUN`；不能取消他人或Teacher作业。已在接纳窗内启动的作业可在自己的walltime与总额度内完成。全局收尾不迟于`T_adopt+31h`，进程重启不刷新这些时间。
- CoreLab最多24 core-hours、总8核，其中一个D0 physics worker最多4核；Hopper CPU最多24 core-hours、最多4核活跃使用。均包括审计/解码/渲染，不能忽略影子成本。站点/账户并发限制从现场真实权限取得，不沿用历史数字。
- 最多16主环境启动：12模型条件＋最多2独立部署smoke＋最多2明确同语义工程恢复；未知/安全/真实任务失败不可作恢复借口。每条已有效判定不重试。
- 主积分最多256000，影子积分最多256000；额外warmup/不一致检查也包含在总额内，若实际旧环境需要额外积分不得默认为免费。
- 最多10256组真实双RGB，模型查询最多1056（12×80在线＋32预备＋64工程恢复）。全部补跑消耗同总额；不以新run-ID扩张。
- 新输出峰值CoreLab16GiB、Hopper24GiB、local-pc2GiB；若现有只读模型/输入缓存可复用，则不重复传输，不清理历史腾空间。新数据私有保存，不公开RGB/视频或权重。

这不是完成时间或资源充足保证。预算/站点不足时一次报告缺口；不要扩大范围或把没有运行的条件改为任务失败。最后30分钟allocation优先结束当前条件、落盘/终态并不再开启无法完成的新条件。

## 8. 与现役Teacher及后续实验的边界

E0不等待TF-R3成功；也不因本文件取消TF-R3。两条线共享CoreLab时先在安全episode边界排定资源，单owner调度，不竞争lease、ROS/渲染/输出。

本轮不训练、不混v3.1、不拟合统计、不生成新TRAIN、不补BASE零标签、不用WBC、不改S1、不运行M512/zero-shot对照、不做目标迁移。必要真实评价RGB不算新增训练数据。

E0结束后才决定：TRAIN不成立则提议小样本拟合/接口诊断；TRAIN成立DEV弱则提议覆盖/语言/视觉适配；source能力成立则提议固定WBC迁移。不能在看到TRAIN4后即用同一E0权限自动续训或新开模型比较。

## 9. 唯一owner可直接使用的接手消息

读取本文件及同release机器卡，复用用户已有五项准备材料。用户明确采用E0后，登记一次新E0资源和12条件manifest；不要继承Teacher授权、重新审计全部历史或再生成一轮路线建议。提交脱敏准备依据到GitHub，私有路径留本机。用原M2048/v3.0/EEF10/D0+S1运行TRAIN4，链路可信时即使0/4也继续DEV8。普通同语义工程问题在本卡边界内自主修复；真实安全或数值域变化停止相关执行。完成后交真实12条件覆盖/成功/拒绝/安全/失败阶段/原视频索引和资源终态，随后停止，不擅自继续训练。

本次发布只新增协议/机器卡，不含五份本地准备原文，不修改现役源码或旧结果。发布者没有SSH/模型/物理/PBS执行；是否采用和实际开始分别以人类采用与owner原生收据为准。
