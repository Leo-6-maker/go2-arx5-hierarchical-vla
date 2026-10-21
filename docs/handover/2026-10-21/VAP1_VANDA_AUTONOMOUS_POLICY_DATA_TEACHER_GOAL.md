# VAP1：Vanda自主接续 + CoreLab数据生产与Teacher验证

日期：2026-10-21，Asia/Singapore。Goal ID：VAP1_VANDA_POLICY_DATA_TEACHER_V1。
状态：USER_REQUESTED_BOUNDED_AMENDMENT_PENDING_NATIVE_ADOPTION。
用户本轮指令：「把更新的vanda端长程自主推进计划写入github，等待GPU的同时继续推进数据生产和teacher验证开发，你仔细思考流程，加快速度」。
本文件落实该指令：原DEC1神经主线继续，新增下述有界、源域TRAIN-only的数据/Teacher工作，及非GPU子预算重分配。不是重复申请同一科学GO；不是无限加算力。现场权限、真实余量、原安全、单owner和冷态绑定仍须通过。
审核固定基线：abe6c0605c1308321721e134618369aa1034b811。原DEC1：a7a7ecd9a6be62a7221af25a5e32a0009b076a84；原PV1：9eff29a44a193dd4d3620dc0bfe5764a402eff42。运行源码与管理HEAD分开，本文件不热改任何现役capsule。
配套机读卡：reports/PI0294/design/VAP1_VANDA_POLICY_DATA_TEACHER_SCOPE_20261021.json。

## 1. 立即采用的决策

1. 保住现有Vanda作业，不再处理已经修复的问题。已有一次成功的资源纠正，禁止因旧聊天再次qalter、cancel或qsub。
2. GPU等待不等于全项目等待：CoreLab复用CR2成熟链，先做24次新数据、通过门后自动扩至累计48次；Teacher开发/验证另列来源和结果。
3. GPU获配有最高放行优先级；纯GPU训练不占Core物理控制时可以有限并行采集，闭环评价需要Core时数据在安全边界让位。不再用“任何GPU R就全CPU停工”的一刀切规则。
4. CR2已完成并CLOSED，不复活其lease、重复24条或重做已合格codec研究。新工作新namespace/manifest/lease，旧结果与预算原样保留。
5. A/B源训练不等待完整B0或新增数据。仍是同父CORR/R0、原139新银行与旧C1配方，各4096步；完整B0仍约束原PV1相对选模和LOCKBOX。新R1数据/Teacher结果绝不热注入该对照。
6. 原48h DEC1 GPU子帽/96h总GPU、8次DEC1/24次总PBS、全程单卡、原20h最终保护与10月26日硬截止不增。仅本文件第8节明确的CPU/存储/传输子帽调整生效；新数据与神经主线在父账本中各记一次。

## 2. 审核所得最新事实（不是实时SSH）

- #47/6306809214及abe6c060：1471903.stdct-mgmt-02在10月21日10:49左右的两次原生读回均为Q/1GPU/4CPU/32GB/30min。一次纠正命令已成功；前次漏select=返回187的失败原件保留。累计qalter2次各有独立批准，未取消/重投。当前神经capsule仍a2caf6a54f37cb76f64f8d7f42abd5f0e3252339，owner-02仍active。
- am_job_cache尚记job.ncpus=36；不能由4CPU显示宣称计费已同步。实际allocation、site资源规则和费用需再核对，不能盲计零或按错误CPU数预报。
- 该在途作业仅获准30min/17个保存TRAIN输入推理seed的A40兼容检查：不是服务、训练或机器人闭环。保持其原scope，不临时把它变为训练job。
- 原CORR@2048/R0完整推理资产已在Vanda独立全成员回读；这不是完整Trainer恢复包。历史2048优化器不能被假称已搬迁；新的A/B从同父权重各自新optimizer开始，后续恢复必须保存完整状态。
- 最新已报总GPU spent20494秒，PBS12/24（DEC1新增1/8），当前reserved1860 GPU秒/17queries/7440 GPU-site CPU秒；B0=14/48（9成功5有效失败），源A/B更新0，DEC1核心矩阵0。均须采用时刷新。
- 原DEC1本地CPU预付27923.9747/28800秒，约876秒余量；旧64GiB GPU-site输出子帽不足以轻松容纳4份约25.54GB完整checkpoint及现有约36.66GB输出。不能继续把它们写成“已就绪”。
- CR2已实际完成24/24任务成功及固定R1准入，12新几何双任务配对；R0→R1为14 PASS→PASS、10 FAIL→PASS、0回退。HF READY revision 9435ba48fa98336b6e88431ed145382724b16b44，bank manifest 3b818f7c53e155a6b47cc2e2116abc34b921a73ea26d6665a623f2bb9e699f44。范围不再拟合，CR2不再派发。
- I_TIME/I_PROGRESS已CPU实现（7bde40e1），共同V2 frame修复及驻留点规则已有测试；源端原生安全/ACK资格、目标冻结任务ABI和64格真实对照仍未完成。CPU freeze不是native freeze。

关键固定身份：CORR parent manifest=94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38；R0=fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7；R1=50d9672397fd0b5353da2011b9f9f568dc9c65c26fb271458f87d60b2d497fa8；PV1 bank=60f7c35a05766fe23974eba3a72f53ca4db5545a3768e6c1cfea164ab4e58294。从原件取完整bytes，不把摘要重建为processor。

## 3. 一位owner、三条可并行工作线

G：Vanda作业守护/资格→训练或64格解耦；D：CoreLab源域数据；T：Teacher验证开发。G是神经与论文关键路径；D/T不成为G的前置，也不暂停G等待“大数据”。

只有原single local-PC coordinator掌握qsub/release/源发布/共享账本写入；采集/导出/分析worker是有界子任务，不是新调度者或自主agent。复用enter.sh、doctor、registry、现有launcher和CR2导出，不建第二套总控、永久cron、workspace或全新的数据格式。

调度器使用缓存的已验证静态身份＋原生动态状态，区分：Q、R_COMPAT、R_TRAIN、R_CLOSED_LOOP、H、F、UNKNOWN。动态证明超时/未知只阻止相应新派发；现有物理episode遵守原安全终止规则，不能凭猜测抢占。

| G状态 | D/T允许的并行工作 |
|---|---|
| Q或确证无GPU执行 | 初始2个独立单核collector；实测吞吐/隔离后最多4核，总Core并发不超过原8核 |
| R_COMPAT（保存输入，无Core物理） | 最多2个单核collector，暂停大传输；放行路径优先，不阻塞175s等原deadline |
| R_TRAIN（真实训练） | 最多2个单核collector，CPU集合/磁盘输出隔离，训练数据只读；出现IO/心跳压力就暂停新派发 |
| R_CLOSED_LOOP或source/target commissioning即将开始 | 先停新采集、当前任务安全收口，确认Core/渲染/通信资源已让出后启动神经闭环；不杀健康任务强抢资源 |
| F且效果结清 | 结算/关闭本job，按预注册下一阶段准入一次新提交；无人工重复GO循环 |
| H/UNKNOWN/身份不符 | 不自动qrls/重投/放行；保留证据，独立不冲突的CPU工作继续 |

raw生产与导出组成有限流水线：最多2条已完成raw等待导出；超过则暂停新的collector而非持续堆数据。导出、打包、视频解码也计Core核时和并发。优先使用一个dispatcher定时回收，不为每条数据跑全仓测试或反复SSH扫描全树。

## 4. G轨：获卡后必须从“文件齐备”走向真实模型结果

### G0 保护现有1471903

先只读核验最新Q/R/F、job/project/nonce/请求、source、lease、owner进程及所有unknown。若已完成则消费结果，不重跑17次；若Q正常则维护同principal的合法认证，不再次qalter。若R则沿现役capsule完成其自身硬件/身份/一次fresh release与17次固定保存TRAIN推理；禁止热换统计量、扩大种子或开始optimizer/物理。

明确回报4层：ASSETS_READBACK；MODEL_LOAD_AND_FORWARD；NUMERICAL_CONTRACT；CLOSED_LOOP/TRAINING。17次单保存观测即使通过，也只证明该probe范围，不证明所有观测、A40训练显存或与H200位级一致。

### G1 自动合法接续，不等待一次新的长程计划

当前probe终态/效果/费用/lease闭合后，按原DEC1授权准备下一冷态job。目标WBC与两个接口原生ready时优先固定CORR/R0的64格；否则优先同父源域A/B训练。不能继续用旧“B0全48训练前门”阻断新版DEC1；不伪造B0 PASS，旧发布门照旧。

A40与H200使用相同明确推理精度/4步采样/R0/真实双摄/噪声身份；不同架构不要求任意bitwise相同，也不能因样本差异事后扩大容差。兼容比较标准来自既有固定数值门和事前TRAIN资格卡。没有相同输入噪声的旧Hopper原数组则标CROSS_DEVICE_MATCH_UNMEASURED，不编造结果；所有新科学比较在同一资格通过的平台/版本上匹配完成，旧Hopper B0不得与A40新增格悄悄拼成平台无关结论。需要合并旧B0时先取得既定可比性资格，否则分平台保留。

源训练先验证真实参数可训练图、BF16/native loss、mask、冻结参数、finite梯度、优化器/调度器/RNG/sampler完整保存恢复、最大显存/host RAM。前32个真实更新同时做吞吐与内存观测，记入A_CONTROL的4096而不是额外免费试训；成功后同一状态继续，不任意丢掉已计费更新。失败按完整恢复/未知效果规则处理。

保持原micro2×acc16/effective32和两臂固定recipe，A/B同平台同实现。若A40显存或时限无法承载，不静默改变loss、精度、batch或用CPU offload冒充原配方；形成精确成本/内存阻断，合法备用平台或单独数值改动需版本化对照。不得套用H200的6.3s/update为A40实测速率。

每臂1024只运行原12条TRAIN探针，保存完整state，再继续4096。若同一预登记合法allocation的剩余墙钟、资源和完整续接合同允许，可在该allocation内保存/验证/探针后继续，避免为逻辑阶段人为再排一次队；不能临时扩展当前30min probe。正式DEV仅4096终点。两臂准备与执行交错可行，但A不作为B父模型。

### G2 目标接口未交付时，不全局停机

并行整理固定EXECUTOR_MANIFEST：WBC/机器人/config hash、frame/TCP、夹爪、原安全/预测、实际状态/时间、reset/stop和任务入口。已有交付就读取核验，不重复索要同一资料。缺一项精确报给WBC负责团队，不在上层调WBC或用理想执行器替代。

I_TIME/I_PROGRESS的原生source commissioning复用DEC1原8次资格配额及原136starts/500万ticks总帽；不是本计划新增免费机器人试验。目标数据仅用于明确的DEC1评价/资格，绝不进入D/T训练库，以保留target-controller-disjoint的论文约束。

## 5. D轨：新48条源域示范，24→48有界生产，不重开CR2

### D0 零物理准备

使用新namespace PI0294/VAP1_DATA/TRAIN。冻结3个既有AP×2方向×2距离×2独立replicate=24个全新几何；每组T1/T2，共48次首尝试。第一批replicate0共24次，第二批replicate1另外24次。近距0.15–0.18m、远距0.19–0.22m，沿CR2现有XY/yaw、相机、物体、4mm Teacher、plant/夹爪PD/S1/原32s评分。每槽至多32个事前静态候选；只按评分输入域、可达/视野/静态安全及几何排重选择，不看任务成败重抽。

与旧TRAIN/C1、CR1/CR2/COV1、PV1 DEV/CONFIRM/LOCKBOX及DEC1 sealed panel几何索引排重。同几何兄弟统一TRAIN。不读受保护评价图像/结果，不用target/WBC任务示范训练高层。若DEC1评价尚未封存，先原子登记D/TRAIN几何，后续DEC1必须排除；不能双方并行各声称对方已排除。

一次封存全部48卡、顺序、原Teacher hash和R0/R1统计字节。R1固定PAD5，不重拟合；R0配对只作同raw的离线诊断。不要让新Teacher开发混入这48条覆盖生产：该批只使用CR2同一合格Teacher版本。

### D1 批次生产与质量门

复用CR2 capture/export/事件索引/真实reader/pack，新增薄scope适配；不删除旧CR2身份guard，不重写采集系统。单worker先验证完整一对T1/T2管线，计入首批24而不是额外benchmark；合法失败保留，有效安全STOP立即停止相应执行域。通过真实隔离及吞吐资格后2→4worker，任何时刻受第3节资源优先级约束。

首批24继续门沿CR2可用性定义：两任务各>=10/12任务成功且>=10/12最终准入；>=9/12几何两任务均准入；每AP×方向×任务至少1/2准入；无R0 PASS→R1 FAIL；native fault0、unknown0、全部计划状态与原始证据明确。通过自动进入第二批，无需再问GO；未过就封存真实PARTIAL/FAIL且停止新增覆盖，不以换seed/范围/阈值救门。

累计48银行门：每任务>=20/24任务成功且>=20/24全准入；>=18/24双任务配对；每AP×方向×任务>=2/4准入；每距离层每任务>=9/12准入；相同零回归/安全/unknown与证据要求。这是数据库可用性，不是VLA成功率。

每实际首尝试最多一次，物理后工程替补0；效果为零且独立证实时可修复输入后启动原未尝试卡。任何真实失败不重播。所有成败和未跑都保留原分母。

### D2 完整导出、固定版本交付

每条成功数据保存实际双RGB/EEF/夹爪、实际提交参考与时间、raw接触/物体结果/失败、601全部H40原codec门、两范围6锚点真实官方reader及跨parent混batch。state preclip单列；仅R1通过的原R0失败仍保留。事件索引覆盖接触、抓持、抬升、释放、脱离和hold，按独立几何/parent和事件曝光报告，不能把窗口数当独立示范。

首批门过且实际完整银行可用就发布VAP1_R1_D24；第二批结束另发VAP1_R1_D48。采用raw closed→audit→codec→full members→fixed revision全读回→READY-last。复用已配置且权限合格的私有存储写者，公开发布只使用已经获准的项目namespace/内容类别；未公开新资产默认私有，不因存在公开writer就扩公开范围。失败提交先查效果，不盲重发。

所有数据worker在CoreLab；Vanda只准备已冻结训练包、将来合法消费，不在login或未经授权GPU上物理采集。不要复制R1到现役R0推理目录。可单列CR2+新48的最大72条候选并集，但去重/全部成员验收前不声称并集READY；不建立更大的CR1归档并集作为新前置。

## 6. T轨：Teacher验证开发，与规模采集及学生改进分开

### T0 先做保存TRAIN的事件与恢复条件测试

用CR2与获准旧TRAIN原记录核对Teacher是否依赖实际观测、抓持结束/无抬升/退出/hold是否与物理事件一致、边界是否被错误时钟触发。修复范围为源域任务Teacher反馈/阶段逻辑、日志、重启状态与评分输入契约，不改WBC、夹爪PD、物理或评分阈值。没有证据就不改Teacher。

明确记录状态来源：ACTUAL_STUDENT_TRAIN、TEACHER_ON_MANIFOLD_RESTART、SYNTHETIC_INITIAL_CONDITION。教师分布内中途重启不是DAgger；合成初态成功也不是对真实学生恢复能力的证明。旧DEV/CONFIRM/DEC1数据不能借Teacher研究回流训练。

### T1 有界开发，最多12次物理，不让Teacher试错吞掉主线

在source TRAIN/专用DEV_ONLY但归训练开发域的状态上，最多一个基线Teacher及一个候选修复版本，合计12次开发首尝试（建议T1/T2各6），每条从原先封存的安全状态/布局出发，完整执行不超过原32s物理时限。开发版本逐次记hash，不无限搜索动作裕度或任务路径。源状态不足或没有完整MuJoCo状态/历史就仅做T0，不能用pose重置伪装同状态恢复。

真实恢复fork必须绑定完整模拟器状态、时间、接触/warmstart/ctrl、夹爪/Teacher必要状态、随机源及累计任务历史；有界重建成本另计ticks。禁止episode中传送物体、覆盖机器人状态或执行已被S1拒绝的动作。合成初态只允许在reset明确声明且通过静态/原安全资格，不计入真正学生状态恢复。

候选在开发域完整成功且通过raw/codec后冻结，才能进入T2。开发失败数据不作为正BC监督；发现接口或Teacher bug先CPU最小复现，最多3轮实质不同修复，已执行尝试不可重写。

### T2 独立验证12次，给出任务特定结论

事前封存12个与T1开发几何/状态lineage不交叉的TRAIN-heldout恢复/重启条件，T1/T2各6；同一候选Teacher、不调参、不重试。每任务至少5/6完整安全任务成功、原raw与R1全准入，才标该任务和该来源类别的ORACLE_QUALIFIED_PILOT；没有actual-student状态就只能称restart/initial-condition robustness，不能标DAgger-ready。

评分保留原全过程谓词/累计历史与终末保持，不能只判恢复后局部窗口或重置no-lift历史。源episode先前已有不可逆违规，保留不可恢复失败而不改分数。两任务分别判，不能T1过就宣称T2恢复合格。

开发和独立验证各最多12，总24次；与D的48次合计最多72物理starts，无物理replacement。若没有真实可资格的状态，明确T1/T2_SCOPE_NOT_TESTABLE，D/G继续。验证合格片段单列SOURCE_TEACHER_VALIDATION_BANK，不混入名义覆盖统计、原C1或当前A/B；后续训练利用另作明确科学比较。

## 7. 性能与可靠性：少等待、少搬运、少重查

- 队列守护保持轻量：Q沿当前已准入60s节奏，R/READY走原受检短路径；一个logical owner复用会话。重计算/大传输不能阻塞release。无变化只持久化本地小状态，重要变化再发#11/#47，不把每次轮询变成Git提交。
- 冷态检查缓存到不可变source/model/data哈希；文件身份/权限变化才失效。新allocation或原协议要求的独立资格仍做，不能省掉原native safety/身份验证。未来stage不要反复下载已经完整回读的CORR、Cosmos和139银行。
- 新源码只跑改动关联测试＋一次必须的完整CI；通过后复用，不重复成百上千测试计成果。最终绑定后检查0400、model alias、完整RGB/numeric、实际reader、输出空间与writer权限，防止CPU能发现的问题带上GPU。
- 生产速度报告合格独立parent/Core核时、闭环完整条件/GPU小时、optimizer update/GPU小时；区分模型服务时间、CPU仿真、RPC、排队及GPU空闲。不可用分配R代替GPU利用率。
- Checkpoint优先简单完整保存。第8节增加存储子帽后，不把冻结片hardlink压缩作为训练前置；保留完整optimizer/scheduler/RNG/sampler，使用已有受检保存路径。hardlink不独立备份，未经原生证明不用它解决主线。
- 每份新checkpoint前按实际resident/保存瞬态/临时归档/失败保存/4份状态的峰值核验。旧父模型、在途恢复点、A/B终点和唯一科研证据不删；只在既有清理权限下对已闭合、无引用、已有完整独立副本的中间重复payload清理。

## 8. 本次明确资源重分配（上限，不是额外购买）

保持全球PV1原上限：96 GPU小时、24 PBS、同时1卡、每job<=8h或站点更小、320Core核时、160 GPU-site核时、48local核时、1000episode/4000万ticks/10万constructors、GPU-site250GiB/Core300GiB/local20GiB新增峰值、300GiB传输，最终20 GPU小时和相应物理/存储保护不变。GPU时间按物理设备分别报告，A40保守1:1扣原GPU秒，不叫它H200实测。

为消除已观察到的人为子帽瓶颈，本次仅在全球真实剩余额度内批准新VAP1子账：
- DEC1/VAP1 local累计上限8→16核时（含旧已付，不重新给16）；新增数据协调也在其中。GPU-site输出累计64→160GiB（Hopper+Vanda合并、含旧现存输出）；Core本组输出64→128GiB；传输累计64→128GiB；Core本组累计96→160核时。旧全球上限不增加，生效前显式迁移账本并保留旧bytes。
- GPU侧DEC1上限48h/8次新增提交/8192更新/14000queries/136个原DEC1任务starts及其500万ticks不增加。当前兼容作业已用的1次提交与所有reserve均包含在内；不可再算“新增8次”。
- 新D/T源数据子账最多64Core核时、72物理starts（D48+T开发12+T验证12）、live<=1152000、shadow<=1152000，所有积分含重建总<=2800000、构造<=12000、最多400个保存parent的有限复核、Core新增峰值32GiB、传输24GiB。它们计入上面VAP1新160Core/128GiB等子帽及全球原余量；不占原DEC1核心64格的136starts子帽，也不能偷用关闭CR2余额。新数据新profile版本需要一项明确迁移收据，不能靠删除旧guard执行。
- 新D/T窗口为实际采用T0后48h且不晚于原PV1各停止点；+40h停新派发、+44h停大计算、+48h关账。任何暂停/重启不重置。若global剩余不足，只执行可支付已声明子阶段，先保障神经主线，不能挪用final保护或无限等预算。
- 每次prepare/submit/start/save前以真实used+reserved+必要未来成本检查，预算单位明确区分实际CPU秒、预付上界、allocated slots和站点计费。site node36CPU不等于进程真消耗36核，也不允许只按affinity4伪称账单4CPU；am_job_cache与最终usage不一致则保留费用保守上界并查清。

本次扩子帽不代表Vanda宿主真实磁盘quota一定足够，也不使A40能装下训练显存；不足时准确报告，不伪造准入。无新云购买、无第二GPU、无自动跨账号。旧有效费用、失败费用和closed CR1/CR2账本不回拨。

## 9. 失败处理权限：同范围自助，不再逐个小问题请GO

普通读侧DNS/NFS/FD/稳定quota/合法认证维护、参数拼装、最终权限、模型alias、manifest绑定、公开存储token=False的只读路径、已有合格writer定位等允许有界修复。小补丁→针对性fixture→一次精确push CI→未来cold capsule；不能热改1471903或正在训练的数据/精度/控制器。

当前1471903资源已修好，不再动它。若将来站点再次改写资源：先记录Q/R与真实请求/分配/计费差异，现役固定guard不绕过。在未来冷scope中可支持已核实的站点分配profile（例如4CPU/32GB或36CPU/240GB）与明确process affinity/内存上界，只有单GPU、项目/身份/时限、过程limit及保守费用全满足才接受；不能把“>=请求”当任意多GPU/多节点授权。若CPU/readback合同不一致且必须换scope，先安全终结确切自有非健康运行job、核清效果/费用，再一次冷提交；不取消健康R，不用重复qsub绕站点规则。

命令先CPU验证argv（包括select=）和只读前置，再写intent，做一次原生修改，独立回读。明确zero-effect语法错误可在本次同范围自主修复权限下纠正一次，日志保留；未知效果先对账绝不盲重试。站点系统hold/MFA/权限撤销/真实安全/预算不足不可“自动绕过”；准备最小支持材料，按已有权限联系管理员，不假称工单已发送。

## 10. 时间、交付与采用回执

全球原时间不变：2026-10-25 02:29:56 SGT停新job/候选，14:29:56停新训练，18:29:56停新物理；2026-10-26 02:29:56硬收尾。新D/T若更早结束以更早者为准。2026-10-23 20:00前报告真实神经结果或最小准确阻断；不是正结果/耗时承诺。

采用后先发一份VAP1_ADOPTION_AND_DELTA_BUDGET：当前1471903状态、旧owner连续性、所有spent/reserved/unknown、新子帽迁移、D/T冻结人口和允许的并发模式。原生job状态刷新后立即接续原任务，不因新文档重建整个系统。

最低交付：
- G：现有A40 probe的真实模型/数值/费用结果；后续真实source更新或DEC1配对task结果，不以文件搬迁/CI代替。
- D：VAP1_R1_D24及有条件D48实际准入银行、全attempt/双codec/事件/覆盖表、固定存储回读与视频索引。
- T：Teacher版本差异、12dev/12verify或明确NOT_RUN、分任务分状态来源的恢复结论与所有失败；不伪称DAgger训练。
- 一份合并终局VAP1_RESULTS_AND_RESOURCES，列出至少首个实际optimizer update、数据有效parent、已完成科学对照cell和原始安全/资源成本；未运行就写0。

复用现有#11科学/#47工程，重要里程碑合并发布固定SHA；原始图像/权重/私有WBC留注册存储。只新增本计划和机读卡，不往160KB的START_HERE持续粘贴全文；确需导航只加短链接，历史记录不重写。

原owner读取固定版本后回VAP1_ADOPTED或精确native blocker。本用户请求已覆盖上述有界并行数据、Teacher验证和工程自助，不再把全计划标NOT_AUTHORIZED；现场许可不足只阻止对应动作。发布本文件没有远端qsub/qalter/物理/模型/训练效果，是否实际采用与运行必须以新原生收据确认。
