# E1／MP1 实验证据审核与下一步优先级

日期：2026-10-17（Asia/Singapore）。
审核固定基线：295da3ec612d296d19f7381ba1ea6e348b908372；最后实际执行者现场收据约2026-10-17 11:42 SGT。
性质：**EVIDENCE_REVIEW_AND_MP1_PRIORITY_HANDOFF；NO_NEW_BUDGET、NO_NEW_OWNER、NO_CLOCK_RESET、NO_HOT_RUNTIME_CHANGE**。
本文件响应用户“仔细审核目前的试验进度，分析失败、成功原因并给出下一步计划”。沿用已采用的单一MP1：c8cdf60e41e31da5796b44a064ef2f22d594a491及MP1_SINGLE_BUDGET_PRECEDENCE_V1.json。E2只作非叠加参考。本次仅审阅GitHub与相关源码/收据，没有SSH、native任务查询、模型推理、训练或物理积分。下述Q/R/F为执行者有日期的现场收据，不是发布者实时测量。

## 0. 最新状态已经覆盖旧网络阻断：不可重复要求恢复已恢复的VPN

最初审阅基线3ed7a5a记录2026-10-17 04:57:45 SGT校园VPN路由缺失、本地DNS失败，670729远端状态UNKNOWN且保留预留。审核期间新提交295da3ec确认：
- 校园网络已恢复；670561与670729均原生F/exit1。第二次因E1_OWNER_HEARTBEAT_LOST结束，非模型任务或原生物理安全失败。
- E1前两次累计保守allocated GPU16363秒已核销。最后第3次原E1 job **671655.hopper-m-02** 已实际提交，最后状态Q，预留12420秒；16363+12420=28783 <= 原28800秒，3/3次提交已经全部使用。
- CoreLab无active worker、无剩余physical reservation、无source STOP。**这不清除另外的target TF-R3安全STOP**。
- 最后scope只补3个结算后的M2-BC工程不完整条件+1个未开始条件，复用已训练完的M2-BC权重；M2-CORR从固定M1@1024首次训练512步并评价固定TRAIN4。M1四个有效结果不重测；原C0数据固定，不热换C1。
- MP1已实际采用，不需要再批一套预算：run20261016T191445Z-localpc-mp1，T0 2026-10-17 03:14:45；10月23日03:14:45停新排队/新候选，23日19:14:45停新实验/训练块，24日03:14:45硬收尾。MP1神经/GPU/物理仍无实际新执行，CPU输入检查与Vanda构建已发生。
证据：CURRENT_SCIENCE_STATUS.md；reports/PI0294/20261016T150228Z-localpc-e1/NETWORK_RESTORED_AND_LAST_NATIVE_ALLOCATION.json。

## 1. 科学结果矩阵：能力、数据、工程和未知分开

|对象|真实已完成|证据仍不支持|
|---|---|---|
|M2048 E0|12尝试，11有效S1拒绝；TRAIN0/3、DEV0/8，一条工程缺失；live native安全0|不是0/12有效失败，也不是目标WBC故障|
|M1@1024|SMALL8追加1024步、32768样本；T1两个已见TRAIN布局均自主32s SUCCESS、T2两个布局S1拒绝（17.6s、18.4s）；两T1独立原件/视频/score审计PASS|不能称100%稳健、DEV泛化、语言grounding已证、双任务可用、Go2/WBC成功|
|M2-BC|实际512步、16384样本、537训练张量改变，冻结参数不变；完整训练状态保留|三条评价工程不完整、一条未开始；不是退化到0/4，也未证明优于M1|
|M2-CORR|同父M1、原C0、固定配方及最后剩余计划已绑定|还没有可核实完成的512更新或完整行为结果；不能宣称纠正训练收益|
|C0|源T1、2独立E0学生episode、833去重H40，实际reader已通过|不是833独立示范，也不是M1在线采集|
|C1修复版|新M1干预两段新增832窗口；并C0共1665窗口/4独立学生episode，私有固定revision与Hopper实际loader PASS|不是1665个独立学生纠错点；专家接管后的状态不是学生访问；未用于旧E1 M2|
|Vanda|新CPU job1463204 F/exit0，容器测试、仿真器+两个控制器编译/链接、输入源码检查通过；1533native CPU秒|没有当前A40模型查询、WBC运行/新物理或完整闭环资格；旧1462595失败保留|
|完整Go2+ARX5|冻结资产与合法接口研究保留|尚无完整安全成功种子/目标数据，TF-R3 STOP有效|

来源：E0_FINAL_REPORT.md；M1_1024_COMPLETE_TRAIN4_AND_T1_AUDITS.json；M2_BC_512_AND_FIRST_EVAL_ENGINEERING_TIMEOUT.json；C1_INTERVENTION_REPAIR_ACTUAL_DELIVERY.json；VANDA_CPU_NATIVE_BUILD_ACTUAL.json。均相对上面的固定审核基线。

## 2. 成功原因：有证据与未证实因果

### 2.1 M1确实学到了可执行的源域抓放
两条T1从实际新观测生成H40、执行原prefix8；Teacher OFF，原D0/S1/score无变化。抓握、约8cm抬升、约17cm搬运、释放/撤离/终段41个hold全过。这排除了“只有loss下降”“其实是专家回放”“靠放宽S1取胜”等解释；支持现有GR00T EEF10+原动作生成方法至少能学会该source任务。

在已固定专家观测probe上，M2048→M1@1024前8点XYZ平均误差：T1约5.65→2.58mm、T2约5.62→2.87mm；SO3和夹爪误差亦改善。这是分布内动作拟合证据，不是闭环成功率。T2误差下降但任务仍失败，说明均值误差不足以反映接触阶段的关键失误。

不能将改善唯一归因于“只多训1024步”。M1同时将数据集中到8条已见成功示范、使用新optimizer/较低学习率和新scheduler/seed/采样；还需要控制推理TF32上下文。原配方peak1e-4，M1新peak1e-5/end1e-6。合理的工作解释是小集上的更好任务拟合，但学习率、数据覆盖、时间与精度各自贡献没有单因素隔离。

推理精度：同输入/权重/seed的12个保存student probe在继承训练TF32与canonical间有实测差异，最大XYZ分量0.63565mm、rotation6D分量0.003485、夹爪fraction0.082764。不同单位不可相加；不能把夹爪fraction当厘米，也不能把精度修复单独说成所有成功的根因。旧@256精度缺口和scope deviation保留，不重跑原有效负例。

### 2.2 Recovery Teacher成功来自当前状态重接，而非源时钟回放
T1从2个独立E0学生状态完整恢复，并在新M1两个不同TRAIN布局的10s和1.2s真实学生前缀后接管。原S1仍约束专家，且被拒学生动作未执行。证明此域有限偏离状态可恢复；不推广到T2或目标整机。

C1导出错误已经通过只读原scalar tick_before/tick_after及200个500Hz时间戳修复。新数据没有重跑物理，已独立NPZ/JSONL/RGB读取、固定HF revision和Hopper processor消费；不能继续写“C1等待导出”。C1版本ce5faf453266689f7f889b8e452dd4f0a9c61200，manifest a6c029e3e07f264a08ab47a757eb899b752e350e7a93f1db2dd87c03b88f0052。旧M2继续使用C0是控制实验变量的正确决定，不是数据丢失。

## 3. 失败原因分层

### F1 科学失败：T2仍不能给出完整合法任务轨迹
E0三个TRAIN拒绝已经重现：table/right_pad、table/right_pad、table/left_pad，分别在即将执行prefix内0.254/0.312/0.138秒触发原ARM_TABLE_OR_FLOOR；不是必须超过独立4mm穿透阈值才拒绝。同状态expert和仅XYZ替换通过，SO3-only和gripper-only未通过。这支持检查位置/接触几何，但反事实分量替换并不证明唯一原因，也没有授权固定坐标补偿。

**M1 T2新拒绝在17.6/18.4秒，不能直接照抄E0的碰撞对和时间作为M1根因。** 下一步用两条M1原保存TRAIN状态/动作进行既有shadow复算，给出首次geom pair、phase、物体实际位移、接触之前/之中/之后、最小桌面距离及有符号XYZ/姿态偏差。比较同状态T2 expert；原T2 recovery0/2不合格，合法短前缀不等于整段恢复可完成。

任务1:1采样解决曝光平衡，但原TRAIN126实际为96条T1/30条T2；重复抽样不增加T2独立示范多样性。应审计30条T2的几何/接触阶段覆盖和合法轨迹裕量，不能仅凭样本数不等判定原因。按已批准MP1先用原TRAIN126；v3.1不是主比较的热替换数据。

### F2 工程失败：评价时限/通信阻塞不能写成策略FAIL
M2-BC首T1执行到25.6s时已经持续抓住/抬升/搬运（lift84.73mm/carry176.69mm），但末段hold尚未观察；约902.9秒wall触发原900秒episode relay限制，64次prefix已ACK、S1拒绝0/native fault0。它既非完整成功，也非有效模型失败。另两条件到20.4/21.6s工程中断；不能根据它们断言M2退化。

直接由收据计算：32s×20Hz/8=80个prefix；900秒总墙钟预算平均每个prefix最多11.25秒（尚未扣startup/closeout），而该次约902.9/64=14.1秒/已完成prefix。粗略平均只是吞吐诊断，非所有prefix都相同或线性耗时模型，但已足以解释“动作正常仍来不及看终态”的风险。

已测单次文件RPC有Core中位0.876s、Hopper中位1.753s/最大32.35s；还叠加模型、shadow/物理、图像及校验等待，不能把总14.1秒全归给GPU或网络。下一次需分项记录模型、传输、gate、物理、写盘与idle时间；减少重复SSH连接/控制文件检查，保留原20s RPC/60s动作/900s episode的已采用语义，不热改时限去“补成功”。现228项CPU/OS-pipe测试仅证明逻辑，不证明实际全链提速。

### F3 控制平面单点与网络失败
校园VPN/DNS断连发生在本机到远端执行之前，owner退出；670729之后原生因owner heartbeat lost结束。现网络已经恢复，原生账也已核销，最后671655 Q。不要再要求新科学GO或不断重复检查旧阻断，更不能假装网络失败等于所有物理无安全。

中期建议：单一逻辑owner不必让个人电脑充当每个prefix的必经数据中继。优先复用有限persistent stdio/RPC并实测；Vanda同节点模型+仿真合格后减少跨机路径。把有截止的执行worker移到合法计算节点属于需要明确部署/host/lease权限和故障接管验证的工程设计，**本审核不自动授权第二owner、私自daemon、打开网络权限或热迁在途进程**。

### F4 Vanda编译问题解决了，但运行资格没有被编译替代
旧container mount/ROS可选变量在严格shell下的setup问题已通过自有容器修订解决；1463204取得当前源码/二进制编译结果。下一项是合法allocation中实际启动、EGL/双摄、时钟/参考和native安全测试，而不是再次编译一遍或声称仍等1462595。A40硬件1462438 Q、依赖推理1462451 H只代表最后现场状态，派发前刷新。

## 4. 下一步执行顺序：只细化已采用MP1，不创建MP2或新预算

### P0：让已提交671655完成最后的E1剩余量
网络恢复和原生核销已经完成，不再从VPN恢复开始。现役owner按真实队列/账户维护唯一671655生命周期，不第四次提交E1、不取消健康作业、不在另一端复制同条件。
- 复用M2-BC完整权重，不重新训练已完成512更新。
- 三个已结算工程replacement+一个未开始BC条件；M1原四条有效结果全部保留。
- M2-CORR按原同父M1/C0首次512步及TRAIN4；不热插已修复C1。
- 先正常收全模型/score/数据/视频与原生费用；若原E1额度/期限到达仍缺，仅在既有MP1允许的正常边界另立剩余子run，不重置E1或重复有效科学条件。
本次第三预留12420s约3h27，旧账16363s，整体距8h上限仅17s未分配；没有隐含额外E1重试空间。所有结果必须以终态证明，排队不是完成。

### P1：P0执行的同时，准备但不抢占现役GPU的固定Source Pair
复用已实际reader通过的原v3.0 TRAIN126、原stats和合格C1，固定M1@1024为同父，S-BC与S-CORR各2048更新；原VLM冻结/projector+DiT可训练/原数值条件，MP1@512/@2048预定评价，不额外开新世界模型、RL或全量微调。
抽样按当前scripts/pi0294/mp1/dataset.py：先T1/T2 1:1；仅T1有纠正时T1内部原示范/纠正1:1，所以整体为25%T1原示范+25%T1纠正+50%T2原示范。**不能写成整体50%纠正**；发布真实microbatch任务/来源trace，不能只验证配置文本。
E1旧M2若名义整体50%混合只含T1纠正，其真实任务曝光需要按原sampler trace披露。不得事后改E1采样以制造公平；新的Source Pair负责更严格的任务平衡比较。

### P2：T2学习诊断，禁止将0/2只归结为少训练
先只读/既有预算内shadow分析当前两条M1 T2拒绝，而非重跑已有效失败。对30条TRAIN T2统计阶段长度、接触/推送关键窗口、正常末端高度和物体变化；与学生到拒绝之前的轨迹匹配。利用结果决定是否触发已经批准的唯一S-EVENT采样分支或四episode容量诊断，原主配方不同时改变多个变量。
报告逐阶段带符号误差和最小接触裕量/首个拒绝，而不只给全H40平均MSE；若出现“安全但没完成任务”的学生轨迹，也记录为不同失败类。T2 Oracle未合格前不掺其拟议动作进入专家BC。

### P3：真实纠正学习消费已到位数据，而非先再采大量数据
C1已有1665窗口/四父episode，其中832新窗口来自两次学生接管；先用于Source Pair看自主效果，再按原MP1最多两轮新采集和512/512匹配训练推进。收益比较必须Teacher OFF，S1 ON；辅助完成率不是策略成功率。
训练采样审计需区分接管当刻student anchor和其后expert states：832窗口不等于832次独立纠正决策。若模型只学到后续平稳expert轨迹而最初恢复仍弱，这只是待检验假设；可先报告anchor与后续分段错误，新的anchor加权或无进展触发接管属于另一个科学因素，**本review不将它默默加入当前固定DAgger触发规则**。
M2只是SMALL8 BC vs C0状态聚合，Source Pair的S-CORR才可消费冻结C1；后者即使胜出，也需要把数据增加、状态分布和额外算法因子区分，不能自动归功于FlowDAgger。FlowDAgger尚无真实反演/训练，仍低优先级。

### P4：Vanda由编译转入当前运行资格，目标主线与source并行
使用1463204的固定SIF/仿真器/两个控制器manifest，先刷新原GPU afterok链，阶段独立地检查A40实际推理、EGL、原WBC合法startup、夹爪/接触和参考时钟，再用真实新观测做同节点闭环。不要求source T2或新CONFIRM先全部成功才测试合格T1迁移。
如果GPU仍排队，可以整理已有编译产物、依赖与日志并做合法CPU验证；无GPU的native测试只在**已资格且明确允许的后端**上执行，不私自换软件渲染或设备/plant。旧TF-R3 STOP保留，重复已知unsafe输入不能作为新资格。
source-only先固定M1/后续已选source模型、EEF10、BASE preference absent；短演示/平台闭环PASS与32s物体task SUCCESS分开。target若不安全，独立source训练/采集继续。只有真实target成功数据出现，才进入MP1原0/5/20/50目标样本适配，不用离线文件变换代替真实成功。

### P5：独立确认与科学可解释性
现T1 2/2只代表两个已见布局。按MP1开发集规则选择冻结共享候选，再用封存SOURCE_CONFIRM_V1做一次新scene-group确认；不重复抽样直到70%。两采样seed共享scene，其相关性要披露，不能当20个独立场景。task/语言配对条件及未见布局要保持同一checkpoint；T2仍未成则来源域双任务candidate尚不成立。
较强的因果论点必须来自配对的S-BC/S-CORR及最后target source-only/few-target对照；当前证据不支持WBC已经可迁移、更不支持任何真机部署。当前controller不修改也不另设hold RMS前置。

## 5. 下次汇报必须优先展示的实际成果

1. 671655原生终态、M2-BC/CORR固定行为矩阵、每条valid/engineering/NOT_RUN和剩余成本；不能用全测试PASS替代。
2. 当前两个M1 T2拒绝的具体geom/phase及真实物体行为，不沿用旧M2048根因。
3. S-BC/S-CORR真实更新与实际任务/来源采样比例，以及自主T1保持/T2改善；未训练则NOT_RUN。
4. C1固定数据确实被新训练读取的证明、原student anchor/后续expert状态比、真正独立父episode数；不能仅再交HF传输。
5. Vanda运行（不是编译）资格和首个真实闭环结果；目标unsafe与source成功分域报告。
6. 修订中继后的端到端时间分解与完成32s的实测证据；管道mock不是实测吞吐。

既有单一MP1总量48 H200h／32 A40h和2026-10-24 03:14:45硬收尾不变。普通同范围工程修复按原授权进行，不需要再开新预算；未知动作不重发、有效负例不洗白、无site permission不执行、不建立第二套入口或协调owner。本review只上传新增文档和审计交接，无新GPU/PBS/模型/physics/原始数据变化。

## 6. 主要固定证据索引

均在审核基线295da3ec612d296d19f7381ba1ea6e348b908372可读：
- CURRENT_SCIENCE_STATUS.md；START_HERE.md（其顶部仍有旧运行字样，新的CURRENT与原生收据优先）。
- reports/PI0294/20261016T150228Z-localpc-e1/NETWORK_RESTORED_AND_LAST_NATIVE_ALLOCATION.json
- 同目录：M1_1024_COMPLETE_TRAIN4_AND_T1_AUDITS.json、A_INPUT_PROGRESS.md、M1_RECIPE_RESOLVED.json、FIXED_PROBE_PROGRESS_AFTER_M1_1024.json
- 同目录：M2_BC_512_AND_FIRST_EVAL_ENGINEERING_TIMEOUT.json、C1_INTERVENTION_REPAIR_ACTUAL_DELIVERY.json、LAST_ALLOCATION_ENGINEERING_CHECK_01.json、ORACLE_HISTORY_ADJUDICATION.json
- reports/PI0294/20261016T191445Z-localpc-mp1/VANDA_CPU_NATIVE_BUILD_ACTUAL.json
- scripts/pi0294/mp1/dataset.py；scripts/pi0294/e1/correction.py。

**总体判断**：方向没有被否定——来源域T1真正学会、状态纠正数据真正产生；但主要科学短板仍是T2、独立泛化和目标整机，而近期效率瓶颈是跨主机评价的可靠完成。优先把已经产生的数据和模型转成闭合比较，不再靠更多计划、无限Teacher修补或不断扩算力替代这一步。