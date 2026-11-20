> 2026-11-20接续请先执行[新版启动prompt](../2026-11-20/START_SLA_LONGRUN_RECONCILIATION_D2_NATIVE.md)。本页保留原配方/预算；涉及当前状态、D2边界、异常恢复和部署门的未来动作，以新版明确调整为准，不热改现役胶囊。

# Codex 启动：SLA 多源后训练与本体适配长程自主推进 V1

把下面的完整任务交给已具备本项目 Corelab/VANDA 合法访问能力的 Codex。执行前记录本文件实际提交SHA；计划JSON不是站点allocation或已经完成的资格回执。

[完整计划](../../protocols/SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1_20261119.md) · [机读任务/预算](../../protocols/SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1_20261119.json)

---

**最新执行优先级（2026-11-19晚间）**：先按[后台训练与并行Eval协议](../../protocols/SLA_SCHEDULER_FIRST_ASYNC_EVAL_V1_20261119.md)实际部署Corelab用户级systemd有限任务链和VANDA PBS受支持的依赖作业链，并将独立Eval挂到每个正式checkpoint封存事件。**不做Codex实时训练step轮询或每6小时主动汇报。** 本文件以下其它训练/数据/评估科学合同仍有效。只有实际systemd unit/PBS JobID独立读回才能声称已挂载。

你正在继续 Leo-6-maker/go2-arx5-hierarchical-vla 的 GR00T+WBC / SLA 项目。用户要求自主持续推进多数据集长步数后训练、把新 Go2+ARX5 本体数据逐批加入，并充分利用 Corelab 和 VANDA A40。请实际执行下面的有限工作包，完成有信息价值的科学增量；不要只返回新计划或不断等待小模型/Teacher高成功率。

## A. 先读取合同和现场，不重复启动

1. 读取本prompt同目录实际提交对应的完整计划与JSON，记录plan SHA。读取AGENTS.md与现役SLA入口；11月11日旧GR00T优先级由用户11月19日SLA主线覆盖，安全/所有权/数据/历史边界继续有效。
2. 用GitHub/API和既有站点访问刷新PR417/418最新ACTUAL及实际服务器。发布前补读基线是PR418 39d8f6e341d2c974aa2baed31588b115e0c2c463、PR417 4c9ccb9cc1db2dd5187152259762739bdbd98ee9；它们不是实时进度。不要用陈旧PR正文的“7更新、GRASP0、J1未启动”覆盖实际报告。
3. 发现现场PID/cgroup/PBS、owner/lease、GPU名称/显存、允许CPU集合、代码/环境/数据/检查点路径、真实期限及未决派发。使用既有入口/Journal/onceIO，实际路径从站点配置读取，不能凭本文编造。
4. Core已存在MIX first500与after500守候器，以及首6目标采集WAIT_PREVIOUS_WORLD_OWNER守候；VANDA已有TGT从500续到2k的PBS1593003；J已发布10/48，仍可能持有world。先核清这些身份，健康进程继续观察，不并发提交同一谱系后继。旧绝对期限仅供定位，不能延长旧卡时间戳当新allocation。首6守候最迟开工23:39:04；若原窗错过，封存真实终态，冷态后用新有限时段接续未执行卡，不另建重复守候或永久冻结采集。
5. 两站现役合同分别为SCALE 926226914a0c08ee1ebb143e0edea1434728107b、D4/D5/J cf9d1acbf72e96b1bd158322ff8cbaf8fe28fabb。活动胶囊不pull、不checkout、不热改。新适配实现使用隔离checkout和明确runtime SHA；不假设main包含现役代码。

## B. 保持三条科研线并行

- Core GPU0保持唯一MIX优化谱系，连续完成20k；VANDA首张实际A40完成原TGT5k。模型/Adam/RNG/bank/LR连续，遇有限时段结束就完整保存实际step，即使不是500/2k等整齐端点，下段从s+1接续。
- 原固定端点的288预测+48独立恢复择一站执行一次，优先VANDA短allocation；已完成TGT500直接复用。MIX500跨站出口与Core冷态非里程碑接续已有实现，复用已有代码，不另造恢复框架。
- Core当前有限native段正常冷态结束、未决请求核清后，优先首6个新TRAIN采集。不要等J全48；未派发J尾段明确延期后按原顺序轮转，已执行分母保持。每采集批最多12尝试，连续两批后给就绪正式评价一个批次。
- 现役MIX/TGT的开始与继续不依赖Teacher矩阵、D5胜出、J全48、N2、24/120条示范或某个高成功率。

## C. 让新增本体数据真正可用

1. 从F1红左/蓝右及F2四格开始，逐条采集。第一批目标24–40条，中期80PLACE+20HOLD+20恢复；它们不是开训门。整体最多240次尝试、每次50703tick/180专家计划；首6包含其中。所有失败保留，120是有效数据目标，不是保证成功数。
2. 用实际可用Teacher、可执行规划和新TRAIN上的纠偏；遥操作只在实际有人可用时加入，不把等待人工设为自主路线前置。同一producer/cell同首因连续两次，停止原样重跑，读取保存状态作实质修复/换生产路径，再用新身份消耗剩余尝试。WBC、安全与评分不动。
3. 主标签沿名义策略层nominal_policy_H40；installed/received/MPC-selected参考和实测TCP/开度分开。expert_query_plan与executed_future_command分型。完整计划后缀可做计划监督，不能称已执行；STOP/缺失后的点mask失效。原DEV/TEST失败不回流TRAIN，使用新独立TRAIN场景。
4. 新target reader替代当前硬绑定C20/3PLACE的入口限制，但不修改活动版本。逐episode准入、明确mask与split，至少一个通过本版reader/schema/标签/mask并实际进入训练成员的新独立episode即可封存D1；隔离partial/仅预检不计E_new。partial路径未真实核验就隔离partial，已合格完整数据继续使用。
5. 初版D1目标state stats只用TRAIN拟合；Go2动作统计/codec、RoboDojo原R48统计与LIBERO统计继承。源成员可用已核验R200，显式绑定R200成员+R48统计；R200自带新统计仅诊断，不偷偷换head输出含义。报告状态token边界与动作超界分布。

## D. 实现并启动唯一新ADAPT5K

- 从D1冻结时最新可用正式MIX端点5k/2k/500按此顺序取父模型，选择依据可用性，不按DEV分数。保留父模型/Adam/RNG，fork新谱系；原MIX继续。
- batch128=64T+32R+32L；FM为2*mean(T)+mean(R)+mean(L)。原目标XYZ/速度/W宽度aux绝对权重与各自内部max/mean归约不变，4次完整十步H40/更新再平均，无delta；不能把峰值超限改成40点均值。
- 重点修改现役shared target直接赋值和FM/aux合并的问题，不能只改DOMAIN_WEIGHT[target]。同时检查aux_slots/frozen_draws/bank/cursor中的32硬编码；64个target槽位按预封存16步轮转覆盖，每更新4aux，batch固定键打散。
- 新目标配额min(48,4*E_new)，其余旧目标回放，总64；按任务→episode→阶段分层，50%整体覆盖、50%关键事件，缺事件回填并记实际数。只有专家事件本次P10到期才是及时闭合正例，不强迫所有窗口立即闭合。
- 参数范围/混合精度/state dropout .25/raw身份保持。AdamW .9/.95/1e-8/wd1e-5/clip1；phase LR显式cosine 5e-6→1e-6，j0..4999，无新warmup。保存phase与父global step、bank和阶段局部游标，不能复位Adam。
- L3完成step-zero真实128窗口梯度/路由/合并及4次目标aux、ADAPT0固定离线和独立恢复，实际优化0；首次Adam apply为L4 phase1，phase25非零保存/恢复均计在5k内，不另做资格更新。CPU fixtures不能替代实样验证。
- 连续5000更新，正式端点0/500/2k/5k。phase2k封存后独立评价并行，冷态只检查一次D2：已封存且有新独立episode则2001起切新bank，否则保持D1。统计/codec/LR/Adam不变，课程切换不等待评价成绩；之后新数据留下一版本。
- 先完成VANDA全量D1/R200/LIBERO1261 TRAIN、继承统计和模型的manifest/缺失成员增量传输/真实CPU窗口核验，SOURCE_DEV旁件不代表TRAIN就绪。在实际额外A40运行ADAPT；若无额外allocation则排队或在TGT正常完成后接续，CPU准备与Core训练不停止。

## E. 评价与资源提速

1. ADAPT0是新state表示下的继承权重，明确与原MIX输入身份不同。四端点各三域96+独立恢复各16，共1344生成；另固定最多96个D1 TRAIN阶段窗口/端点，共384，实际不足不复制凑分母。在ADAPT0首次预测前固定去重window/label_kind/phase_bucket/noise列表与时序公式，各窗口仅归一桶、四端点同噪声。离线预测不能借专家历史消费声称已执行，接触/抬升只能来自新native结果。
2. 冻结新DEV四布局×六指令；ADAPT0/2k/5k各24，固定5k另24新确认，共96科学请求。原双门、P10消费、2ms评分、STOP后0tick保持。Core三端点各16输出核验，完整SCALE包在GPU1实测能装入后再派发。
3. GPU1不够时先继续CPU采集，在GPU0原训练自然冷态窗安排固定native块后恢复；或在A40同配方迁移验证成立且实获allocation后冷态迁移MIX。先与after500 watcher通过同一既有handoff协调，避免抢GPU0；不能静默量化、减图像或改decoder。
4. 空闲GPU上最多4候选×30更新做同一MIX配方性能比较，优先复用Core baseline，实测A40 microbatch1/2/4；benchmark使用父端点原目标23/R48/LIBERO1261及其bank，只准备该配方所需数据，不等待新D1/R200。工程权重不得进入正式谱系；采用新实现时从正式谱系当时最新冷态端点继续s+1，不回退benchmark父端点；分解data/FM/aux/optimizer/save与显存，旧逐位失败保留，数值合同提前固定。兼容且净吞吐提升≥15%才采用明确新后继执行版本。
5. N2最多24请求×10703tick，修复F2瞬态容差和双向STOP同伴继续证据；独立CPU/ROS/目录/owner。原容差全通过才在新冷态人口使用，否则N1持续。不要把此前1.99倍吞吐称为已资格。

## F. 计数、修复与汇报

原SCALE总预算保持25k更新/272万访问/118120生成/98原生/4399394tick；旧D4/D5/J分别记。新V1最大5120更新（5000科研+120工程）、655360优化访问、额外128资格梯度窗口、38390模型完整生成、362原生请求/16824986tick；43200专家计划查询另列。完整明细以同commit JSON为准，未用储备不强制消耗。

低成功不暂停健康长训练；错误标签/梯度路由/不可恢复状态只停止受影响谱系。未知副作用先核清，不无限原样重投；常规用户态工程问题自主修复，实质修复后接续。现有所有权、原安全门和站点配额真实有效是运行条件，不另造逐步人工审批。

每个正式端点、数据版本、原生批次或技术终态发布固定SHA ACTUAL；训练接续改为后台依赖事件驱动，无需持续Codex/SSH监督；仅后台登记、checkpoint封存、Eval完成、失败和终态发布变化。只汇报真实step/完整端点、域访问、生成、示范/尝试、闭环成功/首失败、占用与下一动作。历史失败与旧TEST保留，硬件请求0。本任务不启动新的ChatGPT自动化，不修改基础设施、他人作业或删除用户数据。

首个工作单元优先取得实际新增更新/可恢复端点、第一条新本体有效示范或明确的物理失败证据、一个正式离线端点结果；不要止步于再次写计划。若某条线受实际账号/资源阻塞，说明具体阻塞并继续其余就绪工作。

## G. 本轮优先实作：后台依赖链与Eval并行（覆盖之前的实时监督措辞）

1. Core只读复核旧first500/after500与first6守候的PID、cgroup、lease和到期时间。直接使用已存在且功能符合要求的用户级systemd后继机制；其已到期/已终态时，只有当前新分配的真实有限时段才能派发新的后继。**不热改运行中的worker，不同时启用两个同谱系scheduler。**
2. 将Core训练拆成可恢复、有限终点的自托管service段；每个成功封存checkpoint后释放下一训练段及异步Eval边。每段具有完整step/模型/Adam/RNG/bank游标/实际deadline，失败门禁止后继，不允许只看unit退出码。
3. 在VANDA实际测试PBS是否支持/允许afterok依赖。若支持且有合法资源，按实际≤6小时walltime、保存余量及真实吞吐把TGT的剩余5k划成有界连续JobID队列；同样ADAPT在真实D1+父MIX端点+额外A40可用后独立入队。若依赖不支持，优先复用站点允许的唯一终态回调；不建立远端分钟级轮询和无限重投。
4. 每个正式端点独立安排离线Eval（288生成+48恢复的原面板，ADAPT按其独立预算），在合法空闲GPU/A40与下一段训练并行，或保持QUEUED不抢卡。已完成TGT500结果不重测。模型通过SHA不可变复制并隔离统计/处理器/推理版本；同一eval_key只claim一次。原生Eval与GPU训练可并行，但Core现有唯一world继续串行。
5. 部署结束后从**断开控制连接的独立连接**验证Core后台unit和VANDA PBS JobID/原样依赖/owner；分别封存`SCHEDULER_FIRST_DEPLOYMENT_ACTUAL.md/json`，记录REGISTERED、RUNNING、PLANNED、BLOCKED的真实差别。前驱完成后的后继释放/一次Eval触发至少有一项实证；如果暂不可触发，则记录可运行的受控依赖资格与明确待完成的证据，不虚构。
6. 训练不等待Eval结果；Eval异常不停止健康训练。公开发布在非训练控制关键路径，仅在事件终态触发，GitHub断连时保留站点持久回执。旧费用、失败、安全和任务身份不得重置。

**立即执行的首要成果不是再次修改总目标，而是Core和VANDA各自一个实际后台任务登记回读，以及同一checkpoint上的下一训练与独立Eval两条依赖边。** 没有合法allocation或鉴权时，可交付实际已完成部分，另一站继续执行，并如实标记阻断；不能把仅写了脚本称为已排队。

