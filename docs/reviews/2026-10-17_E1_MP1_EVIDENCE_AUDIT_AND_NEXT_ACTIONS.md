# E1／MP1 实验证据审核与下一步优先级

日期：2026-10-17（Asia/Singapore）。
最终审核输入：295da3ec612d296d19f7381ba1ea6e348b908372的网络恢复/E1最后分配收据，以及2cece6ab4af96ca3898ae1e95a259c30aecad6c1中新增的Vanda原生终态。Hopper现场快照约11:42 SGT；本发布者没有SSH或直接测量实时队列。
性质：**EVIDENCE_REVIEW_AND_MP1_PRIORITY_HANDOFF；不新增预算、owner、T0或运行权限，不热改在途代码。** 沿用已采用的单一MP1及MP1_SINGLE_BUDGET_PRECEDENCE_V1.json，E2仅非叠加参考。用户要求的是审核与接续计划，不是重新从零启动一个Teacher项目。

## 0. 最新结果：两项旧阻断已改变，不要继续照抄旧快照

**网络已经恢复。** 04:57 SGT记录的校园VPN/DNS断连是真实历史；新的原生核对确认670561和670729均F/exit1，后者因为E1_OWNER_HEARTBEAT_LOST结束，不是模型任务失败或真实物理安全故障。旧预留在核对原生终态后才释放。

**E1最后作业671655已提交，最新快照Q。** 已用保守GPU16363秒，新预留12420秒，总28783/28800秒；3/3次E1提交已用完，没有第四次E1提交。CoreLab已无active source worker、无physical reservation、无source STOP；这不解除另一域的target TF-R3 STOP。最后scope只补M2-BC三个结算的工程不完整条件和一个未开始条件，复用已有BC权重；M2-CORR从固定M1/C0首次512步并评价TRAIN4。M1四个有效结果不重测，C1不热换进M2。

**Vanda并非仍停在GPU排队。** 最后读回PRIOR_VANDA_ENGINEERING_NATIVE_TERMINALS.json确认原1462438硬件job和1462451推理job均F/exit0、lease CLOSED。A40 CUDA matmul通过；M2048真实加载一次、1次warmup+16次相同保存TRAIN观测查询，中位0.235583秒、p95 0.237365秒、峰值allocated显存13345840128 bytes。旧scope费用仍归旧账，不重复计算MP1。
这只是M2048保存输入推理，**不是M1/M2、不是在线双摄、不是H200/A40 canonical精度等价，也不是完整WBC闭环**。EGL实际渲染和WBC runtime仍未合格。

**Vanda原生编译也已通过。** 新CPU1463204 F/exit0，容器测试、仿真器与两个控制器编译/链接、原控制器源码前后检查通过，1533 native CPU秒、WBC启动/新物理0；旧1462595构建失败保留。下一步应验证运行，而不是又编译一遍。

MP1已在03:14:45 SGT实际采用：run20261016T191445Z-localpc-mp1；10月23日03:14:45停止新排队/候选、23日19:14:45停止新实验、24日03:14:45硬收尾。新MP1 GPU/神经/物理尚无已发表执行；CPU source reader和构建已完成。无需再批新包或重置时钟。

## 1. 科学进度矩阵

|对象|实际证据|尚不能声称|
|---|---|---|
|M2048 E0|12尝试、11有效S1拒绝，TRAIN0/3、DEV0/8、一条工程缺失；真实安全0|不是0/12有效策略失败，不是目标WBC故障|
|M1@1024|SMALL8追加1024更新/32768样本；T1两个已见TRAIN布局自主32s SUCCESS，T2两个布局S1拒绝（17.6/18.4s）；两T1原件/视频/score独立审计PASS|不证明100%可靠、DEV泛化、语言grounding、共享双任务、Go2/WBC或部署|
|M2-BC|实际512更新/16384样本、537训练张量改变、冻结参数不变、完整状态保留|三条评价工程不完整、一条未开始；不是退化到0/4，也未证明优于M1|
|M2-CORR|同父M1、原C0、固定配方与最后剩余scope已绑定|尚无完成512更新或完整自主结果；不能宣称纠正收益|
|C0|T1源恢复，2个独立E0学生episode、833去重H40，实际reader通过|不是833独立示范，也不是M1在线数据|
|C1修复版|两段新M1干预新增832窗口；并C0共1665窗口/4个独立学生episode，私有固定revision和Hopper loader实际通过|不是1665独立学生纠错点，也没有进入旧M2|
|Vanda当前栈|编译通过；A40硬件及M2048保存输入推理通过|还缺EGL、WBC runtime/原生安全、真实新观测闭环|
|完整Go2＋ARX5|冻结资产及已记录任务接入研究|完整目标安全成功示范仍未证实，TF-R3 STOP保持|

## 2. 成功原因：支持什么，不支持什么

### 2.1 源域T1任务学习确实成立
两条T1均由实际新观测生成H40并执行prefix8，Teacher OFF、D0/S1/score不变；约8cm抬升、17cm搬运、释放/撤离/末41次hold全部通过。这不是专家回放、只看loss或放宽安全所得成功，说明原GR00T EEF10动作学习路线可用。

固定专家状态probe中，M2048→M1@1024前8点XYZ均值：T1约5.65→2.58mm，T2约5.62→2.87mm，SO3和夹爪误差也改善。但是T2均值下降而任务仍失败，平均预测误差不是接触任务成功的充分条件。

不能把改善唯一归因于“多训1024步”：M1改成8条成功小集、新optimizer、较低学习率/新scheduler、seed和采样方式；原名义peak LR1e-4，M1为1e-5到1e-6。最合理解释是更好拟合这批已见任务，但数据集中度、LR和训练时长没有单因素隔离。

canonical推理精度必须固定。12个同权重/输入/seed探测在继承训练TF32与canonical间存在最大XYZ分量0.63565mm、rotation6D分量0.003485、夹爪fraction0.082764差异。不同单位不混同；此差异既不能忽略，也未证明是原失败的唯一原因。旧@256精度缺口和scope deviation保留。

### 2.2 Recovery成功是有限真实偏离状态的恢复能力
T1从两独立E0学生episode完整恢复，又从两条M1真实学生前缀（10s、1.2s）之后接管完成。专家也受原S1约束、被拒学生动作没有执行。它证明此source域部分偏离可恢复，不推广至T2或完整整机。

C1导出首次误将高频安全列当tick；保存数据的scalar tick_before/tick_after及200个500Hz时间戳复算后已修复，**没有重跑物理**。C1最终revision ce5faf453266689f7f889b8e452dd4f0a9c61200、manifest a6c029e3e07f264a08ab47a757eb899b752e350e7a93f1db2dd87c03b88f0052，真实reader/processor已消费，不能再写等待导出。
M2仍用旧C0是保护固定实验比较，不是新数据不可用。新C1应供后续MP1明确版本使用。

## 3. 失败原因分层

### 科学失败：T2还没有完整可执行策略
E0三个TRAIN拒绝已复算为table/right_pad、table/right_pad、table/left_pad，在下一0.4s前缀内0.254/0.312/0.138s触发原ARM_TABLE_OR_FLOOR；无需超过另一4mm穿透阈值。同状态expert和仅XYZ替换通过，SO3-only/夹爪-only不通过，支持调查位置与接触几何，但不证明唯一原因，更不授权固定XYZ补偿或放宽桌面接触规则。

**M1 T2新拒绝在17.6/18.4s，不能照抄旧M2048碰撞对作为本次根因。** 用保存状态/动作和现有shadow预算查首个geom pair、实际阶段/物体位移、是否已推到物体、最小桌面距离与带符号XYZ/姿态偏差，再对同状态专家。合法短前缀与T2完整恢复0/2不是同一能力。

TRAIN126实际96条T1/30条T2，任务1:1采样解决曝光，不自动补出T2独立轨迹多样性。数据覆盖是待检验因素，不因数量不等即判根因。下一主比较仍使用原TRAIN126/stats；v3.1不热混。

### 工程失败：M2评价超时，不是策略退化
首条M2-BC T1在25.6s已抓住/抬升/搬运（约84.73mm和176.69mm），但没有观察最后保持；约902.9s墙钟触发原900s relay期限，64个prefix有ACK、S1拒绝0/native fault0。另两条件到20.4/21.6s工程中断。均不能标完整成功或有效策略FAIL。

32s×20Hz/8=80个决策块；900s意味着平均每块最多11.25s，还未扣startup/closeout。该次902.9/64约14.1s/已完成块。这是端到端粗略吞吐诊断，不是假设所有块耗时相同。已有文件RPC Core中位0.876s，Hopper中位1.753s、最大32.35s；总时间还含模型、S1、物理、图像和校验，不能全归GPU或网络。

新的持久stdio、中间SSH cache路径及whole-call写入超时修复有228项CPU/OS-pipe测试，但不是实际提速。下一合法评价需分解模型/传输/S1/物理/写盘/等待耗时；在原20s RPC、60s动作、900s episode约束内减少重复连接/校验。不热改时限或把25.6s已搬运当成功。

### 网络故障：单一逻辑owner与个人电脑转发单点应分开
校园VPN/DNS断连导致owner退出，远端随后原生heartbeat停止。这一故障**已恢复并核销**，不能再次要求用户授权或恢复同一已恢复网络。长期应减少个人电脑处于每个prefix数据路径中的必要性：先实测已实现的有限持久通道，再在已获部署/host/lease资格的Vanda上同节点闭环。此设计不是授权新owner、永久daemon、站点权限绕过或热迁当前run。

### Vanda部署：硬件/旧模型推理可用，控制闭环仍未知
A40 16次保存输入推理约0.236s中位、0.237s p95，说明有本地低等待推理的潜力；不能断言加上双摄渲染、WBC和物理后仍满足0.4s墙钟。不能把M2048检查视作M1/M2已在A40等价，也不能将13.35GB allocated显存推断为可同时训练/渲染的安全余量。
运行前补齐明确canonical推理精度、M1/指定模型processor与stats、当前SIF/二进制身份、EGL/真实相机、WBC startup/时钟/native安全，才有资格做target source-only。原TF-R3安全STOP不因GPU变快自动解除。

## 4. 已有MP1内的接续优先级

### P0 完成已提交671655的最后E1余项
网络恢复和原生核销已完成。唯一owner维护现有job，不第四次qsub、不双机复制同条件。
1. 复用M2-BC已完成权重，不重训512更新。
2. 完成三个已结算工程replacement+一个未开始BC条件；M1有效结果不重跑。
3. M2-CORR按固定同父M1、C0首次512更新及TRAIN4；C1不热插。
4. 收齐比较、独立评分/视频与原生费用；原额度/期限先到则真实收尾，只在已采用MP1允许的正常边界另立剩余子run，不改旧账。
第三预留12420s约3h27，加旧16363s，总距E1上限只有17s未分配，不能假设额外重试空间。

### P1 源域共享策略：S-BC/S-CORR并行准备、按资源顺序训练
继续已准备的原v3.0 TRAIN126、原stats和已合格C1；同父M1@1024，两个候选各2048更新，冻结VLM/projector+DiT原方案。使用原MP1@512/@2048计划，不额外做密集checkpoint选择。
当前scripts/pi0294/mp1/dataset.py先T1/T2 1:1；T1内original/correction 1:1，所以总体25%原T1、25%纠正T1、50%原T2，而不是整体50%纠正。必须核对actual consumed trace。旧E1 M2若整体50%只含T1纠正，任务曝光可能不匹配，应按原采样trace披露，不事后改旧配方。新的Source Pair负责更严格的任务平衡比较。

### P2 T2定向归因与数据覆盖
复用两个M1 T2失败的保存证据，先查首次非法几何/阶段/实际物体进展，不重跑有效负例；再审计原30条T2的接近、接触、推送和脱离阶段窗口。仅在TRAIN证据支持时，触发原MP1允许的S-EVENT一次采样分支或固定4episode容量诊断。原S1/WBC/任务评分不改，T2 Recovery资格不成立前不得伪造纠正标签。

### P3 先消费真实C1，随后再扩DAgger
先用Source Pair检验C1的自主改善，再按原MP1最多两轮新学生收集和512/512匹配比较。评价Teacher OFF，S1 ON；专家辅助完成不等于学生成功。
832新窗口来自两次新接管，大部分是接管后的expert状态。报告接管anchor与后续expert段的独立数及误差，而不是声称832个独立学生纠错点。如果需要anchor加权或对无进展而非仅S1拒绝触发接管，那是新的科学变量；本审计不默默更改现行固定规则。FlowDAgger仍无真实反演和steering结果，保持低优先级。

### P4 Vanda下一项是运行，不是继续排队/编译叙述
原hardware/inference均终态，首先保留其费用/成功边界，不重复这些smoke。复用1463204 SIF与二进制，在合法已批准allocation中核验EGL/双摄、指定checkpoint及canonical数值、冻结WBC实际启动、夹爪/接触和参考时钟，然后新观测→H40→原安全→真实prefix8/ACK→新观测。
本地推理可行但完整0.4s墙钟未证实；合法sim-lockstep可独立论证，不修改原watchdog来伪造实时。不同GPU数值不要求bitwise完全一致，但差异必须固定记录，不能挑有利host取代E1。
已合格T1可先作source-only目标迁移，不必等T2完美。源EEF10缺BASE监督则明确BASE preference absent；不补零造FB13。target平台/任务安全没通过时，source研究继续；只有目标真正完整成功才进入target success bank及0/5/20/50少样本适配。

### P5 泛化与科研结论
T1 2/2是已见布局证据，不是可靠策略。按MP1开发集规则选择冻结共享模型，封存SOURCE_CONFIRM_V1再一次性评价。相同scene多个seed不算独立新场景，confirmation不回流训练/Teacher。T2未成就不宣称双任务candidate；source成功不证明target WBC迁移。真实模型部署仍NONE。

## 5. 下一次最有价值的六项交付

- 671655原生终态与M2-BC/CORR完整行为矩阵（有效、工程缺失、NOT_RUN分开）。
- 当前M1 T2两个拒绝的真实geom/阶段和物体轨迹，不用旧M2048原因代替。
- S-BC/S-CORR实际训练/采样比例和自主表现；尚未训练就明示。
- C1进入新训练的真实消费及student-anchor/expert后续段分布。
- Vanda运行和首闭环结果（不是另一次编译或重复保存帧推理）。
- 修复后端到端时间分解和完整32s合法评价能否完成。

MP1预算48H200h/32A40h和2026-10-24 03:14:45硬收尾不变。沿原入口/单一owner/不可变运行源使用；普通同scope工程自主修复，未知intent不重发，真实native安全与原TF-R3 STOP不绕过，不建另一个预算/长程计划。发布本审计不启动任何GPU/PBS/模型/物理，也不修改现场原数据或源码。

## 6. 原始证据索引

主要在295da3ec612d296d19f7381ba1ea6e348b908372可读：
- CURRENT_SCIENCE_STATUS.md、START_HERE.md（旧入口文字如有冲突，以有日期native收据优先）。
- reports/PI0294/20261016T150228Z-localpc-e1/NETWORK_RESTORED_AND_LAST_NATIVE_ALLOCATION.json
- 同目录M1_1024_COMPLETE_TRAIN4_AND_T1_AUDITS.json、A_INPUT_PROGRESS.md、M1_RECIPE_RESOLVED.json、FIXED_PROBE_PROGRESS_AFTER_M1_1024.json
- 同目录M2_BC_512_AND_FIRST_EVAL_ENGINEERING_TIMEOUT.json、C1_INTERVENTION_REPAIR_ACTUAL_DELIVERY.json、LAST_ALLOCATION_ENGINEERING_CHECK_01.json、ORACLE_HISTORY_ADJUDICATION.json
- reports/PI0294/20261016T191445Z-localpc-mp1/VANDA_CPU_NATIVE_BUILD_ACTUAL.json
- scripts/pi0294/mp1/dataset.py、scripts/pi0294/e1/correction.py。
最后Vanda增量见2cece6ab4af96ca3898ae1e95a259c30aecad6c1中的reports/PI0294/20261016T191445Z-localpc-mp1/PRIOR_VANDA_ENGINEERING_NATIVE_TERMINALS.json。

**总体判断**：方向没有被否定，源域T1能力、真实纠正数据、A40推理和原生编译都已有证据。科学短板是T2、泛化、纠正训练增益和目标整机；眼前优先把已经存在的模型/数据变成可靠完成的对照，再用Vanda削减跨机执行风险。不要以重复准备、更多预算或新的Teacher大工程替代可比较的行为结果。