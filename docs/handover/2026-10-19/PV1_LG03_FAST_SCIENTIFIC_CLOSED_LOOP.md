# PV1-LG03｜GPU工作时CPU准备，最短路径形成共享双任务Policy科学闭环

日期：2026-10-19，Asia/Singapore。Goal ID：`PV1_LG03_FAST_SCIENTIFIC_CLOSED_LOOP_V1`。
状态：**USER_REQUESTED_SAME_PV1_GO_CONTINUATION / HANDOFF_ONLY / REMOTE_ADOPTION_NOT_YET_PROVEN**。
审计源提交：`917fcda059324961a5c2f10b9f262d2c6711fd03`；最新本Goal编写可见的科学收据：Issue #11 `6268758561`（2026-10-19 21:47 SGT）；工程收据：Issue #47 `6268432885`。
现役权威：`9eff29a44a193dd4d3620dc0bfe5764a402eff42`；执行协议 `docs/protocols/PI0294_PV1_MULTITASK_POLICY_RELEASE_20261019.md`，机读预算 `reports/PI0294/design/PV1_MULTITASK_POLICY_SCOPE_20261019.json`。
原已采用长程Goal：`docs/handover/2026-10-19/PV1_LONG_GOAL_QUEUE_MEMORY_TO_SHARED_POLICY.md`（固定127253737ae0a5ba0e4fed349b3745b9c87beace）。**本LG03只是科学闭环优先级补充，不代替LG02和PV1原协议，不发放新科研或计算授权。**

## 0. 真正的目标：结果优先，不以工程准备代替科学结果

**最小可信科学闭环**：同一旧CORR@2048父模型、两任务相同曝光、相同原始codec与原物理安全/评分下，完整B0 DEV48，A_CONTROL和B_COVERAGE各4096真实optimizer updates，分别完成DEV48原生闭环。发布全部候选的逐task×scene×bank成果、真实安全/参考/物体失败类型、原始证据身份、训练消费与资源账，以及“新增T1/T2独立示范覆盖是否有效”的有限、可证伪结论。两任务联合数据替换的结果不能冒称“只改T2”因果结论。

**若唯一候选通过原DEV资格且剩余资源/时钟合法**，再冻结其唯一权重/processor/stats/代码/动作ABI，打开LOCKBOX80，按现役等级标准完成PV1_VALIDATED或PV1_BETA和可运行研究策略包。若候选不达资格或资源硬不足，**仍须交真实匹配科学结果、失败机制/缺失分母与BEST_RESEARCH_CANDIDATE_NOT_QUALIFIED研究包**；不能用低门槛或事后重抽场景宣称V1成功。

这里的科学闭环≠训练loss下降、CPU合同PASS、GPU allocation变R、单个成功视频，亦≠Go2+ARX5整机/真机或WBC部署资格。

## 1. 本次Goal发布时的最新可核实现场快照（不是实时状态）

- 旧N1确认：S-CORR源域T1=10/20、T2=4/20；FD1、FD2都已闭合，不重跑。新PV1采用COVERAGE_FIRST，冻结TRAIN139条示范、77不同几何scene、62双任务配对组；旧CORR@2048是唯一共同父。
- 原第三B0分配675394已真实结束：9个有效条件（6 SUCCESS/3 VALID_TASK_FAIL）＋1个确证工程中断，642旧model draws。B0完整48是原P2训练硬准入，不加成功率门。
- 当前第四B0作业`675967.hopper-m-02`在2026-10-19 21:28 SGT已被原生qstat观察到`R`，仍原同一job/单H200/12CPU/128GB/8h；128GB排队内存纠正已经原地受控完成（历史证据在#11/#47），**不要重复qalter/cancel/qsub**。
- 2026-10-19 21:47 SGT科学回执#11/6268758561已明确：该作业原生`PV1_NATIVE_MODEL_READY`、B0_CORR、0新训练updates；诊断合格的`DEV/F1/002/T2/N294101-REPLACEMENT-01`已真实32s SUCCESS、641双摄、16,000ticks、S1=0/nativefault=0且独立raw audit PASS；保留被替代的原工程partial。该条件**只补齐一个旧分母cell**，使已确认有效B0至少10/48（由既有9再加1），后续尚余至多38个待合格cell。下个`DEV/F1/003/T1/N294101`已经开始，不推断其结果。
- P5_FINAL/LOCKBOX80、单候选冻结、分级、可运行离线入口等已有**CPU合同实现并在固定917fcda CI39260662293成功**；但**native P5未执行、LOCKBOX未打开**，不能重造第二套。
- 当前B0`selected/deployment=NONE`，未有任何PV1新神经权重训练结果。最后一次公开费用基准是11154已结算H200秒＋当前第四job实际费用待结算；此前28860秒为预留不是已消耗。当前真实ledger必须每次重要阶段从原生刷新。

凡本段与随后的实际原生结果不一致，**以原native hash/ledger/独立audit/owner回执为准**。现役code capsule与管理HEAD不同，禁止热替换。

## 2. 强制优先级：一条临界GPU主线＋必要CPU并行准备

### GPU唯一临界主线

`B0剩余独立DEV → 合法B0全48/全部审计与资源结算 → 同父A_CONTROL/B_COVERAGE各4096 → 各自DEV48 → 固定选模决议 → （合格且资源可容纳时）唯一LOCKBOX80与包`。

**不是**：世界模型、P3默认额外微调、重复FD1/FD2、重新采集新银行、训练一个模型就宣称A/B、重新制作P5框架。

### 当前B0正在R期间：CPU可做，但只做以下四项

优先级C1（必要）：小成本检查原唯一owner/lease/会话持续存活，核对原B0当前case与已持久化ACK/原子文件；已完成的合格case只读审计可以按原许可结算并准备下一轮结果拼接。禁用第二publisher、第二mutation owner、已停止的子代理。**不要在前台同步做慢IO/大量全树扫描以致漏掉现役owner heartbeat**；远端健康优先。

优先级C2（最重要的未来节省）：只在独立管理checkout/既有合法工作根，提前完成A/B `cold readiness` 合约：
- 核对固定原CORR@2048 checkpoint manifest `94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38`、原statistics `fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7`、新HF READY revision `e4f603a1b768018077a8b5c9640250be67500902`；
- 两个臂共用同一新银行/split/codec/loss/训练步数与单GPU配方，各自独立optimizer、RNG、sampler cursor和output，不重复使用另一臂训练过的checkpoint；
- 将`0→1024 TRAIN探针→完成原全状态checkpoint→1024→4096`所需的正式测试、恢复、银行冷态复用、现有性能计划绑定、预算输入和合法冷态递交入口准备完整。**1024只做原固定TRAIN探针，不打开DEV筛选，不发新模型查询**；
- 运行现有CPU合约/CI的有针对性失败注入与dry-run证明：当前B0不完整时必须继续拒绝实际训练；缺run lease/source/data/effect时拒绝；不得用CPU验收冒充native完整恢复。对已通过的CI不要为刷测试数反复跑。
- 私有RUN_MANIFEST/资源证据留在批准路径，Git只发布脱敏proof。C2结果应是两个明确`PREPARED_NOT_SUBMITTED`的合法原scope冷态任务定义，**不是两个新PBS或新GPU reservation**。

优先级C3（必要）：复用已实现`future_admission`+实际已审计case成本，在B0运行阶段更新**只读**的完整资源情景，明确：当前B0剩余＋所需B0后续分配、两臂各两个冷态阶段及DEV48、原生资格/load/checkpoint/close、受保护最终80/审计/档案、站点quota/权限和原绝对时钟。必须区分`observed`、保守规划余量、`UNKNOWN`，队列reserved与allocated绝不双计；实际阻断精确落`BLOCKED_BY_EXACT_RESOURCE`而不是给未知成本填0。B0完整之前**不签发训练准入**。

优先级C4（仅有剩余合法CPU资源）：**复用已发布P5 CPU合同**，按固定源码`scripts/pi0294/pv1/p5.py`、`offline_policy.py`和本次CI检查native cold ABI/未来资源接口缺口及错误拒绝fixture；只做CPU、虚拟布局/元数据，不读真实LOCKBOX图像、不开任何真实条件。不再开发另一个P5、重复实验统计、world model或其他GPU性能方案。若C1-C3或现场任务受影响，立即停止C4。

C1-C4都不得启动物理/GR00T推理/训练或新预算；不得触碰正在运行的675967 capsule、活跃文件、固定S1或控制器。CPU-only工作只在原PV1准入额度内；不足则保留不完整并停止非关键任务。

## 3. G0：先完成B0 DEV48（已活跃GPU的最高优先级）

单owner保持原`675967`，只使用本job已通过的原生资格/模型身份和固定case清单，不能因本计划重新资格、热改或重启：
- 已有至少10个有效cell不可重评。先前F1/002 T2合法工程replacement已成功，保留原中断前缀和工程审计，**不重复replacement**。
- 对剩余未审计条件按原两个噪声bank、两任务全固定先后推进；每个完成后按原raw、动态物体任务score、相机/指令/ACK/噪声、S1/参考拒绝、完整原始失败与视频hash记录。有效S1拒绝与动作参考拒绝都是合法负例。
- 一份8h job未必覆盖所有剩余条件。到真实本job尾端必须在原预留closeout时间内安全保存完整已有结果、处理unknown并终结lease；未执行的case标`NOT_RUN`，不记为任务失败。后续若原预算/排队窗口合法，再通过**下一份全新受检合法cold allocation**仅执行剩余cell，绝不对675967续跑/重投。
- 本轮完整B0条件必须恰好48格有独立完整审计，保留全部原物理尝试（含工程中断和允许的replacement）；只有那时才发布`PASS_COMPLETE_ACTUAL_B0_DEV48_EVIDENCE`并允许P2训练。没有baseline成功率阈值。
- 任何真实原生安全STOP、未知install/query/qsub/optimizer效果、过期站点授权或时间/额度硬门，须停对应lane、保留原证据，不能因长Goal掩盖。

## 4. G1：让A/B科学对照完整跑通，而不是只跑出一个checkpoint

B0完整并结算、真实全资源fresh admission通过后，由同一owner按原PV1顺序在合法H200上启用**原先就允许的**P2：
- 共同父 CORR@2048，COVERAGE_FIRST路线，`A_CONTROL`只用原T1+原T2+C1；`B_COVERAGE`以预冻结配比替换一部分为新T1/T2，所有task与C1曝光严格匹配。
- 两臂各4096 optimizer updates、effective batch32（micro2, acc16）、BF16、原projector/DiT、VLM冻结；seed20261019、原LR1e-5→1e-6/warmup64、同一个性能计划。A不成为B的父模型。真正优化更新数、按task/lane/parent的真实样本消费和梯度检查、scheduler/RNG/optimizer/完整checkpoint全部落原生证据。
- 每臂`1024`探针走原TRAIN-only资格；中断时通过已实现的原子复制+完全恢复验证和已计费cursor精确继续，不因代码准备完成就声称已native resume。
- 只有两臂4096终点完整，才分别评价原`DEV12场景×2任务×2噪声=48`，同时固定B0原baseline。不得只运行有希望的一臂，不用失败模型的尾声当作成功模型的样本，也不得改变scene/noise。
- 发布`PV1_MATCHED_AB_SCIENCE_REVIEW`：每任务成功数/24、task×family×方向×bank、scene为cluster的区间，模型身份、训练loss诊断、参考H40合法性拒绝、S1碰桌与其它S1、完成32s仍未达目标、工程缺失、TeacherOFF；分别报告B相对A和B相对B0的变化。因同时替换新T1与新T2，不能宣称单独T2机制因果已证明。
- 科学闭环结论可以是有效改善、无明确改善或证据不足。**不为提高成功率更改统计量、S1、任务score、候选选择噪声或临时放宽时间条件**。

## 5. G2：满足门才验收 V1；默认跳过额外支线

依原worst-task success→macro→原S1→复杂度→ID顺序，候选DEV每任务至少18/24，不能比B0任一任务下降超过1/24，worst-task必须有净改善。无法成立时，先交真实A/B科学终局，不打开LOCKBOX。**不将选模资格降为“有一个成功视频”**。

P3一轮2048成对补强：**默认跳过**；除非原PV1同任务证据门、原资格和fresh全预算同时成立并证明它比直接收尾更有价值，才能执行。T2旧Recovery Oracle 0/2未达资格，不能私自追加DAgger/WM实验。

胜出局部recipe的第二seed（20261020）复核仅在预算足以保护最终工作时执行；没有复核则即使最终成功最多评为BETA，不能宣称VALIDATED。无理由不许用仅有一次seed夸大稳定性。

P5已经实现并CPU测试，**只在冻结唯一已资格通过的checkpoint、原DEV证据、processor/stats/动作ABI、数值/模型来源、资源封存后**才打开LOCKBOX80；一个scene中两个noise视为相关重复，不伪装40条独立场景。P5每任务40条件，最终按原VALIDATED/BETA/NOT_QUALIFIED门评估，真实失败不重跑，工程替补仅限另行结清原因且每格至多1次。预算不足则不开已封存测试集、报告精确缺口。

研究策略包：共享权重（不是T1/T2两套）、processor/stats、POLICY_MANIFEST/ACCEPTANCE/POLICY_CARD/REPRODUCE、依赖与数据训练lineage、最小离线单次推理及reset/close/fail-stop入口、完整实验审计/视频索引、准确等级。只声称固定底座ARX5源域，真实机器人deployment始终NONE。

## 6. 资源、deadline与线程/owner守卫（完全沿现役PV1）

现有PV1上限：96 H200 allocated hours、24个PBS、同时1 H200、每job≤8h、16384 optimizer updates、48000神经生成、CPU/CoreLab/local、物理、构造、传输、存储等均按原scope和站点更小值；**保护原20 H200小时、最终100 episode/9000 queries及必要ticks/存储**。当前第四job真实计费需要后续原生结算；排队时间不当allocated GPU时间但仍消耗原七天日历截止。各阶段都要原生准确账本，不能用旧published常数充当最新余额。

绝对PV1 T0是2026-10-19 02:29:56 SGT；T0+144h=2026-10-25 02:29:56 SGT停止新job/candidate；T0+156h=10-25 14:29:56停止新训练；T0+160h=10-25 18:29:56停止新物理；T0+168h=10-26 02:29:56结算关账。不会因为GPU排队、Code review、本Goal或会话重启而延长。

**效率与活性**：
- 唯一local-PC owner继续既有可跨聊天中断存活的正常会话，使用原registered workflow和immutable capsule。用户此前已叫停多子代理，不恢复子代理/开第二dispatcher/NFS writer/管理publisher。
- Core/Hopper只在有实际授权角色和原manifest/lease/input时做工作；不在login运行新模型，不调配额外GPU。
- 60s owner renewal、300s lost-owner STOP和原动作/查询/响应/900s watchdog不削弱。不以频繁SSH/RPC轮询制造伪活跃；GPU job优先，CPU准备任务可以安全中断/恢复。
- 已实际CPU证明的gradient标量批量传输及性能仪表若要进入下一冷态训练，仍须前3次真实CUDA等价校验、前100步原审计后才切换；A/B同一冻结plan。当前675967不热改。
- 复用固定HF银行和checkpoint仅在注册源closed+no unknown、完整hash/processor与正式cold消费资格成立时；不得从活跃worker目录偷读可变缓存。
- 只在明确定义的已关闭无引用临时payload上执行原许可的增量checkpoint清理；保留父、固定对照、当前完整resume/必要回滚、raw失败与全部原效果证明。禁止为节省空间删除唯一可复原状态。

## 7. 自动接续状态机、验收回执与STOP准则

合法状态流：
`CPU_PREP_ONLY`（GPU仍Q时）或 `B0_NATIVE_R_ACTIVE`（当前阶段） →
`B0_PARTIAL_PRESERVED`（分配末端未完成） →
`B0_COMPLETE_48_AUDITED` →
`A_1024_PROBES` → `A_4096_COMPLETE` →
`B_1024_PROBES` → `B_4096_COMPLETE` →
`A_B_DEV_COMPLETE` →
`SCIENTIFIC_CLOSED_LOOP_RESULT` →
（仅资格及真实费用允许）
`ONE_POLICY_FROZEN` → `LOCKBOX80_COMPLETE` → `RESEARCH_POLICY_RELEASED`。
实际A/B顺序由原唯一owner的合法已冻结阶段安排决定；任何prefix在途的原PBS不得因新文档重新提交。

以下是硬分叉：
- `STOPPED_BY_SAFETY`：真实原生安全/控制完整性STOP，禁止绕过或修改历史。
- `UNKNOWN_EFFECTS_BLOCKED`：qsub/install/ACK/query/update/delete副作用无法核清，先原生journal对账，不盲重试。
- `CLOSED_RESOURCE_BLOCKED`：不能在剩余时钟、GPU/PBS/Core/查询/存储和最终保护内完成全部必要下一阶段，停止可选工作并保留现有真实科研资产，向人类报告缺少的确切额度/授权，不自行延期。
- `SCIENTIFIC_NEGATIVE_CLOSED`：完整匹配A/B没有DEV资格或未显示期望改善，输出全部真实负结果与有条件下一实验，不以尚未验收的checkpoint叫成功V1。
- `EXACT_AUTHORITY_AMENDMENT_REQUIRED`：若B0旧门使主线不可能及时完成，可**另行提议**将“B0全48训练前门”移至“选模前门”，但原协议及现役代码在单独科学审批前绝对不能改；本Goal不默认同意该修订。

每个重大里程碑：先原生RESULT/ledger/worker/lease/完整模型与数据hash，后脱敏固定Git SHA＋独立readback；Issue #11发布新科学实测和A/B结论，Issue #47只记录确切工程/资源收据。活跃工作按原每≤4h有内容汇报；无事不刷屏、不以CI测试数替代实际优化更新。

远端唯一owner读取固定版后仅一次写 `FAST_SCIENCE_GOAL_ADOPTED` 或 `EXACT_BLOCKER`，同时说明：当前B0已认可有效cell/总48、现役原PBS/lease、GPU累计已结算/预留及未知作用、CPU C1-C4中真正缺失的内容、下一个合法native动作。这只是恢复并聚焦原已授权PV1，不增加物理、神经、GPU、PBS或训练额度。

**长期世界模型/更大DAgger/新codec/horizon/history/新WBC/目标Go2域全部继续由原长期战略记录为未来工作。** 本Goal不授权也不做其准备为本次主线的新前置。

---
本文件发布或Issue评论本身不意味着owner已读取、B0变48/48、A/B新权重已训练、P5通过、资源减少或部署成功。所有这些只以将来的原生回执和独立评价为准。
