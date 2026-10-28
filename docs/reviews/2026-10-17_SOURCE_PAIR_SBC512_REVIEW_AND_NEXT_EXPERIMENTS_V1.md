# Source Pair 审核与下一实验决策 V1

日期：2026-10-17，Asia/Singapore。
状态：EVIDENCE_REVIEW_AND_EXISTING_MP1_CONTINUATION_PLAN；不是新的增量预算、服务器运行收据或全新科研GO。
审核固定输入：e7153154c197dab20456fa8de922bee94244cc7e；最新读取科研评论 #11/6232817564，2026-10-17 18:28:18 SGT。
现役分支：pi0267/executor-decoupled-recovery；真正正在运行的 Source Pair capsule 仍为 03646938ece5c6fa5f09b0fd00732f8159a51468，不等于本文发布HEAD。
沿用唯一已采用 MP1：release c8cdf60e41e31da5796b44a064ef2f22d594a491、parent run 20261016T191445Z-localpc-mp1、原 machine card/precedence。本文更新 aff9a9d2 执行卡中已过时的优先级判断，不覆盖任何原始结果、运行合同或健康在途任务。
本次仅审核/发布文本，没有SSH、提交PBS、训练、推理、物理积分、WBC变更或资源清理。现役owner执行前刷新实际native状态、有效权限、used/reserved、pending effect和固定子manifest；不得把本文快照当实时队列。

## 1. 审核结论

源域已经具有可核验的T1物体任务闭环，不应再从零建立Teacher或重复旧M2恢复。当前最重要的问题变为：为什么由同一M1父权重扩展到TRAIN126后，中间模型在专家状态上误差较小，却不能在实际闭环保持原来的T1成功？

这不是已证明的灾难性遗忘，也不是已证明的模型容量不足。当前证据支持区分：关键阶段拟合、学生访问状态覆盖、接触裕度/执行可行性以及多任务干扰。先完成固定Source Pair终点，再按归因选择一次有界干预。

最新状态：
- 旧M2恢复8条件：3 SUCCESS、2 VALID_TASK_FAIL、2 ENGINEERING_INCOMPLETE、1 NOT_RUN；CORR的已见TRAIN T1为2/2，BC为1成功+1工程缺失。旧训练曝光BC 1:1/CORR 3:1且仅用C0，不能作为纠正因果收益证据。
- 已资格的运输是明确元数据读侧辅助的暂停仿真锁步，三条T1完整32s/240条实际新观测-生成-ACK链。不是实时/无辅助运行，原20/60/900s限制不变。旧恢复3个PBS全部终态、lease CLOSED，不开第四次恢复。
- S-BC@512：四个预声明TRAIN条件全部有效S1拒绝，0/4任务成功、0工程缺失、0原生实际安全故障。原始预测违规不等于真实执行已碰撞。
- 源作业672080最近已发布状态R。探针报告较早快照668updates；后发布#11评论690updates/cursor22080。均为带时间快照，不外推当前实际步数。已实际完整状态恢复，S-BC@2048、S-CORR/C1和DEV尚无终局。
- Vanda1463728未获分配至原截止，NOT_RUN/CLOSED、零实际GPU/相机/WBC/模型运行；不是EGL实验FAIL。target TF-R3安全STOP保持。

## 2. 科学完整性：已经确认和仍有缺口的内容

### 2.1 训练真实；但数据消费有效不等于行为有效
前512更新具有537个changed trainable tensors、493个冻结参数核验不变、有限正梯度，以及完整optimizer/scheduler/RNG/sampler checkpoint。独立消费审计验证16384样本访问、T1/T2各8192、126个原始TRAIN episode访问、无纠正样本、游标唯一连续。不是只改日志而未训练。

四条实际闭环：
| 条件 | 原S1拒绝时刻 | 结果 |
|---|---:|---|
| F0 T1 | 1.6s | 未产生物体移动，VALID_TASK_FAIL |
| F0 T2 | 20.8s | 已有约15.864cm位移，非完整任务 |
| F1 T1 | 10.4s | 抬升约7.678cm/水平18.317cm，非完整任务 |
| F1 T2 | 22.4s | 已有约18.374cm位移，非完整任务 |
四条19项原件审计、1108帧视频解码、142个新观测-生成身份全部通过；138实际prefix ACK+4拒绝。不能将这些负例改成通信缺失。新四条首次非法geom仍未独立重建，不用旧M1的pad/table归因代替。

### 2.2 低专家状态误差与0/4并不矛盾
S-BC@512的96个预设on-expert窗口来自8条真实episode（每任务48窗口/4episode），不是TRAIN126的全面覆盖，也不是96个独立实验。
Prefix8 episode-macro XYZ：T1 2.702812mm、T2 2.986323mm；96个trajectory gate接受。该gate不是原S1碰撞影子检查，不意味着接触可行；宏平均也不约束失败时刻的尾部误差。
另有12个已生成学生状态探针未在本次96窗审计中核验。先消费已有数据，不自动再做推理；这些是历史学生状态，不能冒称本轮S-BC诱导的新状态。

### 2.3 任务均衡并不等于事件均衡
实际T1访问：approach371、contact339、transport1759、withdraw1025、hold4698，总8192。hold约57.35%，approach约4.53%，contact约4.14%，后二者合计约8.67%。这来自原manifest粗phase，不是逐原始接触事件重新标注。
这是值得检验的采样偏重，不单独证明训练不足或根因。不能仅按hold占比自动开启S-EVENT。必须看关键事件在执行prefix8中的实际次数、对应误差和闭环失败阶段；anchor phase不能替代事件发生时点。
旧T2 TRAIN30审计已证实每条均有detach/withdraw、跨阶段窗口存在，不再声称撤离监督完全缺失。

### 2.4 夹爪未来事件不能直接解释当前执行失败
S-BC的4个T2伪转换窗口来自4条专家episode，其中点25/28/32均在当前执行prefix8之外，只有点7在其中。应报告时间位置并追实际安装动作，不能把未执行的远期预测当成四次现场失败的原因；也不能从这点推出远期学习误差永远无影响。

### 2.5 M1严格探针对照缺口不否定M1成功
首次检查的旧M1 NPZ没有实际输入/noise/precision/weight指纹字段，结果JSON仍报告部分元数据。不得用期望值补成实际证据，不能据此作严格逐项遗忘/改善因果对照。也不得把一份旧数组字段缺失扩大成所有旧实验无效。
可审计原有真实独立journal/观测/权重/数组跨绑定收据是否足够建立派生证据链；若不够，保留历史描述性比较即可，不重写原件、不重启E1、不为补格式重训母本。下一严格比较优先使用当前已完整记录的S-BC512/2048及S-CORR同批探针。

## 3. 唯一首要主实验：完成现有固定Source Pair

不中断672080健康执行，不把512时0/4作为临时停止或重配的质量门。完成原定S-BC到2048及固定TRAIN4/暴露DEV8；再从同一M1@1024重新初始化S-CORR，各自新optimizer，不能从S-BC终点初始化CORR。当前neural.py在每一臂用固定parent载入，仍需用实际trainable与冻结参数inventory验证。

冻结项：v3.0 TRAIN126/stats、EEF10/H40/prefix8、原生FM、冻结VLM与既有projector/DiT训练范围、seed20261017、microbatch2/accum16/effective32、peakLR1e-5/end1e-6、warmup32、原训练与canonical推理数值上下文、D0/S1/评分、元数据辅助运输身份。

预期完成消费（不是当前已完成）：
| 臂 | 更新 | 原T1 | 纠正T1 | 原T2 | 总访问 |
|---|---:|---:|---:|---:|---:|
| S-BC | 2048 | 32768 | 0 | 32768 | 65536 |
| S-CORR | 2048 | 16384 | 16384 | 32768 | 65536 |
两臂合计4096更新/131072访问。T2消费按当前确定性sampler应逐cursor一致；T1在指定位置被纠正数据替换。该实验估计的是任务曝光匹配下替换部分T1监督的效果，而不是简单增加训练量。

C1目前只有T1资格，所以本对照不能检验直接T2纠正的收益。T2改善若发生是共享参数迁移候选解释；T2不改善也不能否定合格T2 DAgger。C1的1665窗口只对应4个独立学生parent，须分开报告接管anchor与随后专家访问状态。

保留固定@512/@2048检查，不新增密集checkpoint选择。总32个原计划开发条件，当前仅4完成；以原每条件replacement、source safety STOP、native资源与截止规则运行。报告SUCCESS、有效S1拒绝、实际安全事件、工程缺失和NOT_RUN，不能只报告有效子集成功率。

接近allocation收尾时，按原代码保存完整partial state和消费游标；原生终态核对后才在已有额度内登记下一分配续接，不重置scheduler/optimizer/总曝光、不跳过未评条件、不因剩余GPU秒而突破submission cap。不得为了赶完两臂修改45分钟收尾或900s评价约束。预留终点评价资源优先于任何可选新模型。

## 4. 与训练并行的最小诊断：先用现有保存证据

### B0：12个历史学生状态探针的独立审计（优先零新推理）
读取已有ON_STUDENT_RESULTS、实际payload/双RGB、生成数组/noise/precision/checkpoint绑定及已保存的原S1证据。没有实际合格专家标签的状态不计算虚构动作RMSE；原无效initial-restoration仍排除标签对照而保留分母。报告actual-state、on-expert两个分布的差异，不把历史E0状态冒称S-BC自身分布。
输出：student-parent/anchor资格、每任务prefix8误差（仅合格标签）、原S1结果（仅实际运行证据）、缺失项；新query默认0。没有足够来源时给NOT_COMPARABLE，不把缺字段修复升级为新模型实验。

### B1：当前四条S-BC失败的状态归因
固定四条已判定失败，不复跑自主负例。每条取首次拒绝前实际状态及紧邻上一个ACK后的实际状态，最多8个预先登记anchor；保存来源必须包含原模型输入、语言、时钟、IK/控制器历史与全部影响动力学的状态。只有qpos/qvel快照不足以证明恢复等价。

先对原动作作exact-history重建并对齐已记录高频结果，再查询首次非法geom、触发规则、位置/姿态/夹爪方向、请求与实际差异、物体实际阶段。T1 F0在1.6s就失败，不能把全部新问题归为T2终末撤离。

随后若原MP1诊断范围与实际额度允许，在终点权重封存后，将S-BC512/S-BC2048/S-CORR2048放到同一固定anchor/输入/噪声下比较。缺失输出最多8×3=24个诊断draw、零新自主live episode，已有可用输出复用；shadow与历史重建仍按真实积分收费。具体模型/NI/shadow/CPU上限、实现和子manifest必须在dispatch前绑定，不能用该24上界自建额外预算；未能映射到既有有效scope则该新增draw部分只保留为提案。

专家/XYZ-only/SO3-only/gripper-only混合参考只作通道归因。专家需从同一真实状态生成可信动作，不能拿原示范相同时间戳硬配；短prefix通过不构成完整Recovery Oracle资格，不可自动进入正样本库。privileged phase/geom可用于离线审计，不进入policy输入。

### B2：阶段与事件曝光审计
在已有committed trace上连接原始事件，按task/episode/origin/学生parent，区分first8内close/open/detach/withdraw开始、H40后段事件和长hold。报告每个独立episode的实际事件访问与未覆盖比例，而不是只给总窗口数。
把对应的专家态和学生态误差按阶段分层，同时报告均值、尾部、signed XYZ/角误差、几何最小裕度与S1结果；不同单位不相加。该分析只描述现有事实，不在训练中途修改采样。

## 5. 终点之后的实验选择：每次只选择一个可辨识问题

| 固定终点和诊断结果 | 下一动作 | 不可声称 |
|---|---|---|
| S-BC恢复T1，T2也有稳定任务结果 | 固定候选，按原DEV/确认规则检验；不无条件再训练 | 512负例可删除、已经普适泛化 |
| S-CORR优于BC且T2未明显退化 | 报告单seed、同task曝光的纠正试验结果；固定候选再确认 | 4个parent足以证明普遍DAgger收益 |
| 两臂专家态关键prefix较好，学生态持续差 | 在合格任务上扩当前学生访问状态的恢复；复用原MP1最多两轮、每轮至多12episode及BC512/CORR512同父比较 | 旧C0/C1已经覆盖新策略所有偏差 |
| 专家态关键事件本身误差突出，实际曝光与相应失败提供一致支持 | 原MP1已有S-EVENT仅一次：任务内50%uniform+50%event邻域，其他项固定，事先定义事件和窗口 | hold57%本身已证明采样是根因 |
| T2连成功专家状态也拟合不好，尚不能区分覆盖/容量 | 原MP1固定4条T2 episode/512更新容量诊断；不是共享候选 | 小样本拟合成功代表泛化，诊断checkpoint可自动部署 |
| 同实际状态可信专家也不能安全恢复 | 结束该恢复标签来源，记录Teacher/任务接口限制；不放松安全 | 可以用短安全前缀造完整成功标签 |
| 仍有共同工程中断 | 保留缺失，不据此判断BC/CORR胜负；修运输而非加训练 | 未运行就是0%策略成功率 |

S-EVENT与新的学生纠正不要在同一个实验同时加入；证据不足时不启动可选模型。更换学习率/损失/VLM训练范围/输入历史/codec/分辨率/推理步数也是新变量，不能隐蔽加入当前对照。若固定两臂均失败，本轮仍须形成明确负结果，不自动续到4096/8192。

## 6. 独立确认、语言与预训练主张

两臂@2048按原TRAIN4+暴露DEV8报告。DEV是开发数据，非未接触的测试；诊断/Teacher标签只用TRAIN。按原worst-task→macro→较低复杂性选同一共享候选；预先固定SOURCE_CONFIRM_V1，10scene groups×2tasks×2noise=40条件、每任务20，不改变已有70%研发里程碑为部署认证。统计不把同scene两noise和重叠窗口当独立scene，报告场景聚类和小样本限制。

同场景T1/T2指令对提供语言条件的必要检查，但仅输入文本不证明依赖语言。若未来增加指令交换/同义改写或新对象组合，应固定合法任务成功标准与不可混入训练的条件，另行界定在已有scope还是后续研究，不在本次确认中临时加条件择优。

原官方初始化与长期继承初始化的匹配对照、数据量曲线与任务级基线，是证明GR00T预训练价值的后续论文实验。当前Source Pair只比较纠正数据，不承担这些主张；不因为旧parent探针缺指纹就强迫重训官方基模。

## 7. 目标Go2＋ARX5 lane：一次连贯资格，独立于源策略

P0现已对明确辅助运输有实际32s资格，不再把无辅助实时优化当源域全部训练的永久门槛。同理，目标lane不以完美EEF hold RMS或T2先达到100%作为额外全局前置。

Vanda相机上一份是排队未运行，不能复制成新失败证据。下一合法target分配先完成已构建栈的EGL/真实双摄和固定模型数值/输入资格；若这些通过且已有原生安全准入，再连续推进冻结WBC实际startup、时间/参考安装/ACK、新观测与任务。仅相机初态渲染不算动作后的闭环。

TF-R3真实STOP保持；后续只能采用原MP1允许且有证据的任务接入修正并独立重新资格，不能跨host绕过同一故障。冻结WBC/腿策略/plant/PD/机械限制/原评分，不改夹爪力或制造关节传送。重复同一未解决原生安全故障按原卡停止target lane；source主实验不等待它。

目标最小科学比较是同一任务、同一冻结WBC下的可信参考执行与固定源策略执行，分别保存raw reference、installed reference、actual achieved EEF/物体结果。T1已在source有能力，可优先做T1目标资格；真实target成功示范具备后才启动原少样本适配。缺BASE标签就标preference absent，不补零称BASE13训练。真实机器人deployment仍NONE。

## 8. 工作量、预算和结束规则

唯一MP1原T0：2026-10-17 03:14:45 SGT；新提交/新实质候选截止2026-10-23 03:14:45；新实验/训练块截止10-23 19:14:45；硬收尾10-24 03:14:45。H20048h/A4032h及各站点CPU、提交数、400source/320target启动、6400万积分、24576更新、存储和网络上限均沿原卡，不加E2、不重置子时钟，不因发布本文件再登记新owner。

先保护当前两臂和已有终点评价。32条原计划开发condition按每条900s墙钟上限预算，单评估即可占用至8小时分配等待；加训练、载入、checkpoint与每次原45min收尾后，不能声称一份4h allocation能完成整个pair。用真实资源及各分配余量安排原生续接，不对用户保证完成用时，不修改原deadline凑成功。保留原Hopper/Vanda末4GPUh和收尾CPU；已用与预留不能混成余额。

读侧辅助固定为当前已资格算法/角色/来源，持续记录可见性/锁等待/IO/推理/物理耗时；其元数据缓存观察者效应公开披露。无需为旧根因叙事无限做smoke；只有影响当前可用性或可比性的工程修复进入关键路径。

## 9. 本轮结束必须交付的四件内容

1. MATCHED_SOURCE_PAIR_FINAL：两臂各2048（或如实partial）、每臂65536实际访问、真实source/task/origin/parent、冻结参数与完整恢复证据，终点模型与processor/stats绑定。
2. BEHAVIOR_AND_FAILURE_MATRIX：全部固定condition状态、工程缺失/未运行、当前四条失败的可用首次geom、成功与负例视频、同一元数据辅助运输说明；不以历史成功填新condition。
3. MECHANISM_DECISION：96专家窗与12历史学生窗资格、当前学生anchor诊断、事件曝光与prefix8错误，说明支持/未排除哪些假设以及仅选择哪一个后继实验。
4. RESOURCE_AND_TARGET_STATUS：source/target分开、原生used/reserved/lease终态、目标资格的PASS/FAIL/NOT_RUN、当前是否存在共享候选；不可用的独立确认和真机部署仍明确NONE。

最小可信成果是完成任务匹配的比较并定位失败；不要求必须跑完所有可选分支。不得用更多准备文件、CPU PASS或总loss下降代替以上交付。

## 10. 固定证据与源码索引

以下项目文件均以审核基线e7153154c197dab20456fa8de922bee94244cc7e读取；实际训练代码另固定03646938。
- reports/PI0294/20261016T191445Z-localpc-mp1/RECOVERY_FINAL_ACTUAL_PANEL_AND_RESOURCES.json
- 同目录 S_BC512_CHECKPOINT_CONSUMPTION_AND_FIRST_VALID_NEGATIVES.json
- 同目录 S_BC512_FULL_TRAIN4_AND_RESUME_AUDIT_ENGINEERING.json
- 同目录 S_BC512_SAVED_PROBE_AUDIT_AND_PARENT_PROVENANCE_GAP.json
- 同目录 SOURCE_LEARNING_STARTED_AND_VANDA_CAMERA_QUEUE_CLOSED.json
- scripts/pi0294/mp1/dataset.py、train.py、neural.py；scripts/pi0294/e1/model.py
- reports/PI0294/design/MAINLINE_ACCELERATION_MP1_20261017.json
- #11/6232675864、#11/6232817564（后者较JSON快照晚），#47的对应实际工程收据。

方法参考：Ross et al., AISTATS 2011, A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning，https://proceedings.mlr.press/v15/ross11a 。其问题设定说明专家分布拟合不自动等于策略诱导分布表现；不是本项目已发生某种唯一机制的证明。NVIDIA Isaac-GR00T官方README区分open-loop与closed-loop评价，https://github.com/NVIDIA/Isaac-GR00T 。公开最新文档仅用作方法背景，不替换项目固定runtime/revision。
