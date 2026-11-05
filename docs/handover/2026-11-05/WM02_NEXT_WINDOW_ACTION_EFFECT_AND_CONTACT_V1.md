# WM02 NEXT1：新有效窗口的动作效应辨识、结构化WM与接触资格

日期：2026-11-05（SGT）。**这是新窗口计划，非已经开工或获得现场权限。** 从 [PR #356](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/356) NIGHT1结案 `3d62dc098eadf789aa92f962484e1bdc442c7a99` 叠加；上一夜所有worker terminal/lease CLOSED，NIGHT1的派发截止08:36:08.657、最终截止08:56:08.657已经过去。不能使用旧lease、沿用旧计时器、热改旧运行或假设旧scope仍可派发。后继须通过**现有站点owner/registry/Journal/实际资源锁**获取新的有效simulation-only有限窗口；外部许可不足时仅离线工作，不自行放宽。

## 0. NIGHT1冻结事实和明确科学问题

- NIGHT1新增66个合格生产episode、4176个50ms转移；实际703269原生2ms tick，700次WM主更新+4恢复探针；199原生PASS、17原生FAIL保留、1准入拒绝。较早AUTO1另有336转移和300+2更新，不并入NIGHT1的新效果；**不同模型run的更新数不代表单个模型训练1000步。**
- C08执行源码`93362d53497fa1aaee58ff03e091153e6c893a6c`，原生二进制SHA256 `29b261541b9dc2b2ed5c6df065026b33a11c3019181b9a8cfa980a7878d91df2`；48/48原生PASS、生产24episode/576atom、8个已知物理父（6 TRAIN、2 DEV，**各父完成3/4动作**）。新ReplayV3 `3493aa2ed00ceb2f9cd2c9f74a8c5a224f73be8da595be3d6abc4e74424a9a9a`，**尚未训练新C08模型**；剩余8条旧面板是PENDING，不擅自归入完成集。
- 最后已训练目标C06模型checkpoint `c32cd16ac30782d3e1efe135c922f3134ae1dd100c4515a48de49dcfb20f438f`。仅在对应两个生产DEV父的H16 TCP坐标RMSE .273214mm，强ARX .435573mm；H1则模型不及ridge/ARX。**不得跨executor静默热启或把该指标冒充C08上的动作辨识能力**。
- C08真实观测到任务TCP最大运动42.103857mm、BASE最大19.960516mm，而任务动作候选仅±X/+Y1mm，且所有66合格生产episode物体contact/grasp/motion=0。**42mm是从起点的运动幅度，不一定是动作跟踪误差**；不能直接用全task峰值与1mm命令做等号。冻结模型未来目标Jacobian数值有效但最大测试坐标约1.12e-4m/m，尚无native动作因果验证。
- 根因有进展：C06 q/dq独立队列产生17个错配APPLY；C07成对同步后仍发生RL calf限位；C08物理时钟绑定5ms PD/20ms策略/2ms物理，原生48/48PASS及0重复策略bucket，未改变冻结WBC数值。历史失败与C07、C08各候选必须保持不同执行器版本。
- 静态目标接触方案：从已封存全身pose出发，朝立方体中心下降在第25路点腕部碰桌；**中心上方20mm**的26路点和13步夹爪几何检查PASS，但未有动态接触力、抓取、受支撑释放资格。
- 基准报告：`reports/WM02/night1-20261105/{NIGHT1_FINAL_CLOSEOUT_07.md,C08_SEALED_REPLAY_07.json,FINAL_ATTEMPTS_07.json,FINAL_COST_TERMINAL_07.json}`。只读hash与原文件身份优先，不复制未经验证的私有路径到公开库。

### 优先研究问题

**Q1：** 在给定C08冻结执行器、已知参考、实际观测历史下，0mm HOLD与±1mm动作的**独立动作效应**能否从自然settling、BASE运动、参考重锚、续约与测量噪声中区分？**Q2：** action-conditioned模型是否比NO_ACTION及TRAIN-only ridge/ARX在新父场景上预测这些增量更好？**Q3：** 独立于此，在合法且实际稳定的目标初态下，是否能得到最初有效接触正例？

## 1. NEXT1总范围、顺序与资源守则

- 优先级：`R0→S0→S1→D0→M0→V0`主科研线；`C0→C1`接触线在资源合适时独立推进；`B0`并行基准可选；`FINAL`无论成功与否执行。
- **R0实际新scope有效登记后**，才允许任何新的T域构造和mj_step。新window通过原owner登记有限端点；不重复NIGHT1旧warmup/cold资格，除非新的执行器/scene/运行条件需要重新验收；不复活CLOSED worker。新文件不等于准入；失败必须标`OWNER_BLOCKED`。
- 默认**最多两个原生隔离slot**（ROS220/221需现场重新核实是否可复用），新接触先1slot；一条管理owner和一个合法GPU模型worker。现场实际CPU、RAM、GPU、cache、物理锁的优先级高于本计划；不扩4/16、不迁MJWarp、不改变系统/SSH/驱动、不抢其他用户作业。
- 每个新native worker有原scope限定的tick与墙钟上限、专属scene/config/cache/trace/STOP/lease；已有控制器权重、增益、成本、solver、关节及碰撞安全、接触限位、物理dt和腿policy均冻结。用户只授权**仿真**，无真机、主动行走、正式Actor/PPO/GR00T训练。任何真实原生安全停立即终止受影响批次并保存最早完整trace；不修改安全门。
- 建议新scope的申请额度（**申请值，不是已有权限**）：4–6小时有限研究窗口，最多2slot、总计≤64新原生尝试（包括资格/拒绝/失败）、≤400000 native tick，单任务段≤8秒；两个matched WM对照最多各200主更新+1恢复探针，额外NO_ACTION matched arm最多200+1；是否增加至事先登记的2000端点另取资源准入并在新DEV和动作可辨识性支持后进行。所有失败中实际发生的calls与ticks计数；站点更严格的限额优先，离窗口结束预留≥20min结算。**不自动设置下一日期、续时或后台守护**。

## 2. R0：接管、冻结输入与离线复核（0新物理、0更新）

1. 检查目前真实branch HEAD、NIGHT1 canonical index / run terminal /原owner registry/lease状态；独立验证所有worker终态和最后一次真实实际结果，确认没有别人正在使用同一物理锁。旧任务不能未知时重复派发。核验C08真实binary/scene/controller/ROS/trace SHA与被记录的内容，明确区别代码提交SHA与实际载入.so SHA。
2. **C08只读分组诊断**：原24episode按6TRAIN/2DEV、3动作/父，输出每50ms的world TCP、BASE、TCP-in-BASE、关节、夹爪、任务阶段、实际ref/solver payload、来源时钟、mode、初始速度和时间序列。报告0～8预决策HOLD与第8～24的时间曲线，不用全段最大幅度冒充跟踪误差。
3. 对每个合法运行计算：(a) `||TCP_t-TCP_t0||`、(b) `||TCP_t - T_ref(t)||`、(c) `||T_BASE^-1 T_TCP||`漂移、(d) 参考重新锚定差、(e) action发生后的投影位移；分解BASE共模和EEF相对BASE的变化，画图/表保留每父每动作及前缀。检查延迟周期和TTL余量，`controller_selected_policy_id`仍DERIVED时不能宣称直接确认。
4. 检查C08原面板是否在独立重启间能达到可比前决策状态：按父计算`t_decision`处TCP/BASE/姿态/速度与接触，测量自然run-to-run散布；**没有完整MPC/WBC/ROS restore，不叫严格同状态反事实**。根据TRAIN侧数据**冻结**未来稳定起点筛选条件（不可从DEV动作结果调条件），输出`STEADY_PREFIX_SPEC`与拒绝分母。
5. 复查现有模型C06对C08身份是否具备合法只读迁移资格；不允许静默加载跨executor训练。若仍需对C08独立预测，明确使用C08新fresh模型或只读兼容性诊断，分开结论。

**R0 Gate**：时间与动作效应计算合同、每父/动作实际行及重启偏差已经明确；原权重/数据SHA不变，至少一份`STABILITY_CAUSALITY_AUDIT`有实际运行退出码。源域/S数据不可替代目标。即使本阶段认为当前1mm动作不可辨识，也要输出为什么和最小修改假设。

## 3. S0：建立“预决策稳定前缀”，先单环境、再双环境

目的：使任务中动作注入前的settling小且可观测，或证明当前执行器在该初态不可达到条件。**不能把任意等待增长当作已经改善**。

- 保留原冻结WBC数值及C08队列/回调逻辑；只允许改变**adapter的合法参考安装顺序、正常HOLD稳定时间、预登记任务初态与观测gate**，每个变化设新executor/collection协议身份。不能在安全超限时自动重试同一原始状态。
- 首先使用已有同一安全workspace的真实测量EEF/BASE pose hold，按原引用、time/TTL、350ms renew执行；单环境在固定上限内（建议初次最多8s合法任务前缀）收集0-action时间序列。分别报告开机到稳定、base高度、足端contact、TCP世界/相对BASE，以及达到门槛的比率。持续HOLD也应维护消息/时钟，不得用停物理作弊。
- 建议**科学筛选目标**（不是原生安全限制，需在TRAIN预先冻结阈值）：决策前连续0.4s的TCP-in-BASE peak-to-peak≤2mm，世界TCP与BASE变化分别统计；同段max绝对速度及5/20ms实际控制时序不出现未解释异常，所有关节和接触保持原安全合规。该2mm是试验门槛，可以与TRAIN噪声水平比较再预登记，不在DEV上调参挑成功episode。未达门槛记`UNSTABLE_PREFIX`并保留计数，不能隐去失败。
- 只有单环境正常安全达到可观察稳定，才以相同spec检查2环境；单环境失败先分析base/arm耦合、初始化与相对坐标误差，**不得直接增加EEF位移幅度或改冻结WBC**。

**S0 Gate**：至少2次独立合法初始化达到预登记的稳定窗口，且实际参考消费者证据完整；2slot资格另计。否则停在解释性基础数据，不宣称可训练Actor。

## 4. S1：动作辨识（Action-effect identifiability）——先有真实信号

- 预登记4个动作：HOLD、+X1mm、−X1mm、+Y1mm；保证从同类稳定前缀出发，并使动作仅在**burn8之后的decision atom8才首次出现在安装或actor可见的参考**，严格检查H40后缀、时钟与安装状态。用每个合法episode的观测建立前缀；不将未来完成的TCP/BASE当输入。
- 第一小面板建议4个已有TRAIN父各4动作共**16次目标尝试**，顺序和slot分配随机但预登记。现场只能合法顺序重新初始化，不得从qpos/qvel快照伪造MPC/WBC恢复。完整分母包含未稳定、REF拒绝、非法接触和原生STOP，安全失败停止该条件。
- 按**父×动作**对齐，计算从decision时刻起的`ΔTCP_world`、`ΔTCP_BASE`、BASE运动、相对安装目标的误差和`Δx^{a}_{h}-Δx^{HOLD}_{h}`（h=1,8,16，按同父不同episode比较）。这是*跨重复重启的匹配对照*，不是物理严格同起点反事实。提供bootstrap以父为重采样单位的区间，样本少时报告宽区间而不说显著。
- **诊断性晋级标准**（在TRAIN面板先注册）：每父真正动作效应与HOLD散布可辨；理想状态为沿方向符号≥75%父一致、H8/H16有`|effect|≥3×HOLD重复标准差`。这只是继续模型对照的研究门槛，不等于跟踪精度/任务成功，也不等于必须达成1mm实际移动。若1mm无法辨识，不凭空增加3mm动作；先查自由度、参考消费、BASE耦合和时钟，未来幅度梯度必须按原安全和新预登记资格另批申请。
- 如果仍存在长程自然漂移，但匹配差分效应稳定，可以继续研究**增量模型**；但不能把绝对漂移为零当训练必要条件，也不能只靠差分准确宣称全状态模型可信。

**S1 Gate**：真实作用方向、有效值、对照噪声、置信范围均可审计；未观察到辨识信号则转向执行响应建模，不训练高阶Actor。

## 5. D0/M0：完成现有C08覆盖，建立真正的动作条件WM对照

- **历史8条缺失C08动作**：先审查与原C08已运行24条是否在**相同执行器、稳定前缀和预登记协议**下仍具有可比意义。若新S0调整了决策前稳定策略，则收集成新版本面板，不能填入旧C08的32个格子冒充同protocol。旧C08保持24/32和8条PENDING；不能补数据造满面板。
- 先保留P01–P06 TRAIN、P07–P08 DEV的已有父隔离。训练来源只用有实际合格行动的完整train windows（burn8＋predict16）及合法观测/动作，保证actor在决策前看不到未来选中的动作；归一化只用TRAIN。
- 冻结**三种配方**：RAW动作、同参数/种子的SCALED动作，以及**NO_ACTION消融**（另报告ridge/增强ARX/persistence），建议每个fresh200次主更新＋1恢复探针，全部同训练样本/初始化/归一化/有效batch/optimizer；不得在新DEV上回看择优后称其盲测。现有V4中3×2000的长端点是另一项预登记；**不擅自把200结果叫2000**，若S1证据足够，可按新有效scope在匹配身份下延长至预定端点。
- 同时报告FREE与TEACHER-FORCED（特权只诊断）、H1/H8/H16的TCP、BASE、执行关节、夹爪、物体、接触；一开始对象静止，persistence在object位置可能精确为0；不能人为忽略。必须额外报告**动作差的预测幅度/符号、与NO_ACTION的差距、真实有限差分方向、Jacobian敏感性**。
- **M0晋级标准**（建议值、不是历史结论）：存在真实可辨动作对照后，action-conditioned在新DEV上H8/H16动作差RMSE优于NO_ACTION至少20%，同一父≥75%方向正确，绝对TCP各H不比合理ridge/ARX基线显著恶化，安全／事件无虚假正例；分父报告，2父不能推显著统计。若未达，回到状态／动作表示与显式执行响应模型，不能靠多次DEV调权。
- 模型版本WBC/clock/scene不可混，严格保存optimizer、normalizer、RNG、cursor和命令采样身份；每次1恢复探针单列。C08旧模型尚未在C08数据上训练、C06 checkpoint不能无合同迁移。

## 6. V0：新父场景前瞻与有限选择（严格区别预测和控制）

- P09–P16保留为**在模型、基线和阈值全部冻结后**的8个新父场景；先按原RootSafe物理与scene核验，不复制一个seed来冒充物理新父。原预登记P09–P14/P15–P16若被当作训练/DEV，需要重新标注其角色；**本NEXT1默认只做冻结模型的新父零更新前瞻**，禁止把同一场景先用于fit再称其完全未见。
- 每父至少同样4类动作的可比较、有界合法顺序尝试；先保存模型的四个动作预测和不确定性，再执行，完整分母包括拒绝／安全失败。没有完整控制状态恢复就不声称同状态oracle regret。
- 只有动作响应诊断可信、任务评分与安全事件有合法定义且模型前瞻排序超过规则存在实际机会，才另行设计小规模**模型选动作**与原强规则的独立合法执行。当前不是正式Actor训练授权。模型数值输入梯度非零不代表物理梯度可信。
- 若新父场景表现不佳，保留独立负结果，优先定性系统辨识/执行器领域漂移，而非在同批P09–P16上训练再宣称盲测。

## 7. C0/C1：接触路线从静态IK走向真实动力学

- 在**已完成S0单环境稳定性**、与新执行器/场景指纹相符时，重新计算静态26路点并检查TCP/BASE/物体位置，不能未经复核直接重放旧`center+20mm`路径。原center路径第25点腕部/桌面干涉原件保留。
- 按分级：小步自由空间接近→接近目标上方→逐步接触并验证左右pad、TABLE/object支撑力与非法接触→有界闭爪→短保持→受支撑释放→撤离。每级必须记录真实关节/接触力、运动和原安全状态；**静态几何重合不是抓持成功**。
- 未资格的接触条件**最多1slot**。某一级原生安全停止则保留首次失败；不得为接触放宽腕部与桌面碰撞规则、4mm接触条件、joint limit或WBC gain。先在离线几何调整路径并用新scene/参考身份试另一条件。
- 有合法接触正例后，单独输出contact-labeled replay与事件覆盖（left/right pad、grasp、support、release、illegal/stop）；此前66个自由空间episode仍明确“无物体接触”，不能把重复0标签当作正例覆盖。
- 不用接触失败阻断稳定执行响应的独立研究；反之不能仅凭自由空间WM通过就授予复杂抓放的world-model控制资格。

## 8. B0可选吞吐对照和FINAL

- 真正的1／2环境比较必须在相同executor/scene/任务/工作量下预登记，包含完整启动/预热/收尾与实际数据质量；旧描述性2.02×不充当普遍基准。在S0/S1任务推进优先且资源允许时做，严格不扩4slot或MuJoCo Warp。
- 每次结束输出`NEXT1_STATUS`：当前有效scope/owner/lease、注册尝试+成功+失败+拒绝、真实物理tick、有效转移、父集合、动作条件与采集版本、GPU主更新+探针、两套baseline/WM的逐父误差、Q0–Q3判断、接触资格、worker终态、所有完整SHA/实际CLI/退出码/成本、未完成下一条已实现命令。
- 最关键的判定写三个独立标签：`EXECUTOR_STEADY_QUALIFIED`、`ACTION_EFFECT_IDENTIFIABLE`、`ACTION_CONDITIONED_WM_PREDICTIVE`；若未满足则明确FAIL/PARTIAL和根因。正式Policy准备为另一个科学Gate，不能因为模型拟合而自动宣称就绪。

## 9. Codex自治规则（只在合规scope/endpoint内）

先读本文件和对应JSON，再核现场index；在独立新run/capsule自主修普通路径、序列化、切窗、引用时钟和报告漏洞，跑相关测试、按新身份有界验证，不逐小修询问。一次只改一条物理因果机制并保留负结果；不因产出报告、单次200更新或GPU暂忙就结束所有离线合法工作。资源/外部许可/安全阻断时停止动态分支，继续只读审计、基线分析、静态路径与文档，不改系统网络。到实际有限截止先stop/close own workers, seal, report，不允诺或启动后台自动续期。**本次GitHub交付仅计划，不构成新scope或实际数据。**
