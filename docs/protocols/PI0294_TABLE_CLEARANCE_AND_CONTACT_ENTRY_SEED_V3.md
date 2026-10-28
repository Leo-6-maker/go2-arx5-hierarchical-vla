# PI0294 V3 — 桌面净空与首contact接入：有界整机成功种子方案

**状态：PREPARED_FOR_EXPLICIT_ADOPTION / NOT_AUTHORIZED。** 用户本次授权准备和发布。本文件不启动新IK/RRT/渲染/物理/模型/训练；`authorizes_execution=false`、`launch_allowed=false`，运行适配NOT_IMPLEMENTED。采用本固定提交后，才可按本方案的科学变更和下列累计资源继续；不得删除V2关闭或安全标记。

[任务卡](../../reports/PI0294/design/SOURCE_ACCESS_V3_TASK_CARD.json) · [条件/逐槽随机表](../../reports/PI0294/design/SOURCE_ACCESS_V3_CONDITIONS.json) · [来源和冻结锁](../../reports/PI0294/design/SOURCE_ACCESS_V3_SOURCE_LOCK.json) · [实际只读准备证据](../../reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/source-demo-access-v3-preparation/PREPARATION_REPORT.md) · [采用prompt](../handover/2026-10-15/SOURCE_ACCESS_V3_ADOPTION_PROMPT.md)。

## 1. 固定负结果与本版研究问题

基线为`3b607bbbf4373f61a53280d5b06afc3651e35f54`、Issue #11 canonical **6195800940**。V2 `STOP_V2_VALID_NATIVE_SAFETY_FAIL_NO_TASK_SEED`、636文件归档、全部旧FAIL和原件不改。T1原5.6s前缀的机身—桌面安全FAIL不是工程无效，也不是完整32s质量结果；T2接近端无合格IK不等于接触端无解或WBC不能做任务。

本版分别检验：

|task|唯一主要干预|仍然固定|不能先声称|
|---|---|---|---|
|T1 `W0_H47`|完整桌面、方块、目标及源参考共同抬高30mm；桌高0.44→0.47m|桌面长宽/50mm厚度、XY/yaw/源索引30..257、原uniform时间、working-open初态|旧机身包络净空意味着新动态安全/抓放PASS|
|T2 `W0_CONTACT_JOIN`|去掉原不可行接近pose前缀，直接连接固定首contact源目标；为此匹配working-closed夹爪初态|原0.44m桌面、XYZ/姿态接触目标、contact→push→withdraw→hold全部后缀及V2局部时间法|源phase名称或旧contact IK意味着新实际接触/成功|

T2并非只删除一个数学前置而让开爪直接推送：其接入合同显式包含下述合法闭爪初态。两task不是2×2消融；不把T1新场景或T2组合接入的成败纯归因某个孤立量。旧场景不改判，新结果只限各自新scope。

完整Go2＋ARX5自由根/腿/臂、原合法native启动、MPC/腿RL/WBC及gain/solver/dynamics、机器人/桌面/方块质量惯量摩擦、所有碰撞体尺寸/接触margin/允许对、SafetyGate、native安全、H40/dt2ms/20Hz/first8=0.4s、参考安装及任务成功阈值不变。BASE软意图仍0，不冻结实际根、不调WBC、不加补偿，不换相机/seed/目标方向或搜索FSM阈值。

## 2. T1：固定抬高3厘米，不裁小桌面或放开碰撞

实际只读准备扫描V2全部**5300个已保存root包**，用原模型base_link直属碰撞geom11/12/13的local AABB经实际root旋转和平移取保守world-Z上界。首末状态与原生`mj_forward`的几何中心核对最大差0，无新`mj_step`。包含原margin后的最高包络 **0.3994451305469343m**；原桌面下沿0.39m，与该包络重叠。

按一次性几何规则 `new_top = ceil((saved_max_Z_plus_margin + 0.020 + 0.050)/0.010)*0.010` 得 **0.47m**。新下沿0.42m，旧保存包络到新下沿净距 **20.554869mm**。候选在本发布前冻结为0.47，**执行阶段不得再尝试0.46/0.48或扫描高度**；公式只是选择这一个设计的来源，不是执行时自适应升桌。

版本化新scene overlay只允许：

- task_table中心Z `.415→.445`，完整half-size仍`.17 .24 .025`，XY仍`.515,0`，厚度、摩擦、碰撞mask/margin不变；不挖洞/缩小/隐藏桌面。
- 方块初始化中心Z `.46→.49`、目标区渲染Z `.4402→.4702`；source/goal XY仍`[.45,-.08]→[.45,.08]`，yaw/边长/质量不变。仅在新scene初始创建/reset实际数据时设置，episode内不teleport。
- T1源V2的全部228个参考位置统一`+[0,0,.03]`，rotation/gripper/source index/time原值逐项保持；连接仍是当前actual→源首close索引30，绝不平移或伪造机器人actual。原uniform后缀15.633403722/15.252646531/15.395897036s不变。
- 原robot XML、18个原生actuator、根/腿/臂初态及WBC代码不改。新builder在原task_plant输出上按命名字段施加白名单变更，原task_plant文件与历史compiled模型不改。

这是**新场景假设**，不是把原碰撞当作不存在。只覆盖旧保存root包络的几何理由；新执行中root可能走得更多、发生pitch/arm/腿碰撞，必须由原全身规划/native安全实时检验。旧轨迹反事实净空不得报为新物理通过。

### 原任务评分如何严格保持

T1继续调用字节冻结的原`score_episode`，显式参数`table_height_m=.47, goal_center_z_m=.49`；相对抬升2cm/0.3s、持物搬运10cm、goal角点±.07m、桌面高度±5mm、速度、释放、TCP清距8cm、28–30s的41/41保持、完整32s和全部安全AND均不变。公共函数原本支持table_height参数；这不是提高高度容差或忽略接触。

旧`.44`场景包装/identity拒绝仍保留，旧入口仍必须拒绝H47；新scope适配器只能接受固定H47参数，不能把旧wrapper改成任意桌高放行。变更CPU检查须证明共同Z平移后上述任务谓词等价、旧入口仍拒绝新scene、真实场景/参考/评分三者采用同一高度。原world坐标raw保持真实.47/.49，不把actual记录移回.44伪装旧场景。

## 3. T2：固定首contact直接接入，保留真实推送技能

采用原source contact事件经原`searchsorted`映到的已提交目标，不搜索后续更容易点：

|source|新起点索引|保留至首hold|保留点数|V2时间表保留后缀|
|---|---:|---:|---:|---:|
|000|223|488|266|15.500000000s|
|001|231|504|274|16.250000000s|
|002|223|496|274|16.100000000s|

保存source的原geom ID解析确认，在起点之前及起点采样时均无pad–cube接触，无grasp/lift/carry；物体水平位移0，只有cube–table沉降接触。原close和align是非接触准备，不能把它们说成已经完成推送。原contact→push/搬运→detach/withdraw→hold的pose、gripper、阶段顺序与所有原时间间隔保留。新时间仅减去该contact节点的V2时间原点，保留局部限速，不重新平滑/加速或另做时参搜索。

### 与闭爪工具状态一致

T2新scope的**任务夹爪**合法初态/每次reset固定为`q_left=q_right=.002m`（原working-closed，mechanical fraction `.002/.044`）。不是机械0下限，原2mm安全margin、motor速度/力/gain/damping、机械44mm范围均不变。仅新任务夹爪两DOF、其qvel=0及最初command/previous target绑定相同working-closed；根/腿/臂的原生初始化完全不改。实际创建/reset的`mjData`、CPU audit与首个真实packet必须一致，不修改observer只读副本。初始cube仍原source，夹爪不能假设已抓住物体或添加attached关系。

这个初态替代被删除close阶段的准备结果，随后保持保留源首点的closed命令；不另插闭爪物理search或新的短质量门。若实际reset路径不能一致绑定，先同语义工程修复；真实限位/碰撞失败不是工程重试。

从原完整actual状态，以原MPlib自由连接到固定contact目标，保留原native/contact允许对：仅pad–cube合法，任何table–robot、非法self碰撞仍拒绝。**T2连接按明确的contact-entry阶段使用既有contact规则**，允许连接中发生原本合法的首次pad–cube物理接触；不是增加碰撞白名单，也不改变4mm原penetration界。所有接触/物体偏移都是真实任务历史，不能在安装后重置或把源未来物体pose塞给目标clone；不因为已经触碰便切换更容易的source。

旧CPU T2 contact各1个去重goal只可在**原保存actual包＋原closed planning view**消费者严格等价时引用；它不是当前新reset的actual解。CPU输入显式标记保存状态/规划view。新actual warmup/live状态、finger q/packet/world或约束一变，就用扣账的新goal，不改旧input_sha、不免费重抽旧96初值。初态改变后的当前观测必须重新渲染/采样，不能旧RGB配新state。根/腿始终真实自由，规划中的closed工具view不得冒充实际测量或回写actual。

## 4. 有限CPU来源资格与内部搜索

继续只用自有TRAIN/new的T1/T2各000/001/002，不新增第4源、外部下载、姿态/scene/yaw/seed搜索。固定000→001→002；每源内部原arm L2/index规则，首条完整合格即锁定，不跨源按成绩排名。T1新Z目标不能当旧goal cache hit；T2只在前述精确保存消费者等价时复用contact。

**2个CPU batch，仍6个科学候选分母**。T1最多3个新join目标×16CLIK；T2新IK为0，3个旧contact引用。各batch总≤6RRT/60秒、每源最多2原RRT流；各最多2×1001 dense点。原MPlib0.2.1、range.1、1秒/RRT、无simplify、fix_joint_limits=false、原关节限位/速度.7rad/s/acc1及cartesian .06m/s/.3rad/s不变。未轮到的源写NOT_EVALUATED_INTERNAL_BUDGET，不伪造科学FAIL。

CPU只做标注清楚的新scene初始化/保存状态**静态规划view**，不制造actual RGB或训练样本。T1根据新scene初始化方块、使用保存root/arm预测，不能声称已实际执行；T2引用的旧open-actual/closed-tool规划view也不声称等于新closed实际状态。后续每task专用actual warmup bundle必须对**新实际packet、同一锁定source、新真实world**重新求1目标×16CLIK/≤4RRT/1001dense，无缓存伪装；准入失败不换源。

仅执行任务卡列出的**12项变更相关CPU契约**，保留20项旧有效契约及来源，不为本方案重做旧全部诊断。12项尚未执行，代码/计划存在不代表PASS。新source/geometry/scorer参数、实际初始化、cache身份、预算/停止边界全部闭合后，直接运行各自完整任务，不增加短里程碑质量实验。

## 5. 实际物理：T1→T2，每task最多一条32秒

两条科学条件独立，物理顺序固定T1 `W0_H47`→T2 `W0_CONTACT_JOIN`。各task自己的CPU源路线和actual warmup通过即直接进入32秒；不等另一个task PASS、不等辅助签字或例行GO。各warmup≤5秒，不修改原GetUp/WBC时序。

自由连接≤10s、连接＋保留源活动后缀≤28s；剩余至32秒保持原终点。固定H40/20Hz/first8、fresh actual RGB/EEF/gripper/BASE→原gate→reference-only安装→真实0.4秒WBC→新观察，所有g20Hz命令与原ABI一致。不能沿每次actual误差重新锚点、停源clock追成功、装RRT关节轨迹、夹爪接触后重置history、复制source actual或控制根/物体。等待CPU冻结本run sim/MPC/history，NON REALTIME，不宣称实时策略。

保留planned reference、installed reference、same-tick tracking、actual contact、actual object task各自证据；跟踪RMS仍只是诊断，不恢复WBC保持前置。只有完整原task成功＋原native安全＋数据/时钟/标签完整性，才能称本新scope成功种子；一次触碰不是抓放/推送成功。固定32秒失败保留晚期失败，未完成前缀不报完整成功率。

普通有效任务FAIL不重跑本task、不换source/seed；另一task仍可独立执行。**任一次有效native安全FAIL停止本V3全部新物理**，包括warmup；不因未跑另一task或还有额度继续。目标仍无seed时明确真实数据缺口，不转入WBC/FSM调参。单task成功也不自动开放正式扩采、GR00T训练或新评价集。

## 6. 明确预算、累计重分配与原截止

最新只读准备后父账：**39启动/358808积分/12361双RGB/15856.836595361141 Core CPU秒**；本轮准备实测远端1.675332356 CPU秒，新增IK/RRT/渲染/物理/模型均0，PC编辑/传输未测部分UNOBSERVED。V2结算/终局文件不改；准备增量另存。

|本V3拟上限|硬推导|
|---|---|
|2科学条件；≤4启动|2完整＋最多2次严格任务前工程替换；有效安全/任务失败不算工程替换|
|≤42012积分|2×18503（5s warmup＋32s＋3尾步）＋2×2503任务前替换|
|≤1282双RGB|2×641；任务前替换task RGB=0|
|≤164bundle|2CPU＋2actual warmup＋160正式decision|
|≤165新goal/2640CLIK|3CPU＋2warmup＋160live，每goal16|
|≤660RRT|12CPU＋8warmup＋160×4live|
|≤166166dense/830830local迭代预扣|4004CPU＋2002warmup＋160160live；每point≤5迭代|
|≤14000 CPU秒|含最后≥1800仅收口；4CPU/1物理worker，按更小剩余额度扣|
|≤8GiB新增暂存|不复制历史Trainer/全raw/venv，不删唯一原件|
|新GPU/PBS/神经/训练/扩采/下载|全0|

每次库调用前持久预扣；actual与reserved分开；失败/崩溃/incomplete占槽，预扣未用部分不返还再抽。RNG只引用固定条件表中的原子流，换版本不当免费重试；新目标/新场景每次实际计算仍正常计费。

**不隐瞒局部搜索重分配**：MP-S2＋源V1＋V2现有实际53bundle、122goal/1952CLIK、78RRT、48048已预扣dense。V3最大后为**217bundle、287goal/4592CLIK、738RRT、214214dense**。原source后继的局部200bundle/205205dense保守界不足以容纳本完整32s新方案；本方案拟在明确采用时改为217/214214，仍不超过更早完整MP-S2外层222bundle/221221dense授权。加旧ManiSkill V1另存11011dense为**225225≤原232232**。其他历史调用另存，不删账；旧V2本身上限/结果不改，不借“batch”把内部调用藏掉。

物理累计最坏**43启动/400820积分/13643双RGB**，≤原source-slice总天花板43/404023/13771；CPU最坏29856.836595361141≤父账36844.79211945514。原source CPU14400 allowance起点15614.037112470665，截至准备已花242.799482890476，加14000为14242.799482890476，仍有约157.20秒保守余量。采用时刷新实际更大消耗，按`min(14000,14400−source累计已花,36844.79211945514−父账)`降低新CPUcap，保留1800收口；若不足完整固定合同所需资源就统一报告具体缺口，不能免费重开。

原T0 **1792044600**、新工作截止 **1792083759（10月16日01:02:39 SGT）**、收口 **1792087800（02:10 SGT）** 均不重置。采用/断连/排队不续时；过期不发起新物理，仅收口或明确报告一次资源缺口。新scope有独立人类采用、source seal、版本化scene、manifest/owner/lease及namespace `source-demo-access-v3`，沿现有父账；旧V2安全关闭标记永久保护。没有本固定提交的明确采用，仍不得执行。

## 7. 交付与停止点

采用后主owner端到端：版本化CPU/scene与实际reset→自身来源CPU/actual warmup→直接32s→独立CPU→私有逐文件回读/实际双RGB视频→Issue #11 canonical全文回读。不中止于CPU_READY、编译、提交或静态IK。

两task都无合格源路线：`STOP_V3_FIXED_SOURCE_ACCESS_NOT_ADMITTED`；原生安全FAIL：`STOP_V3_VALID_NATIVE_SAFETY_FAIL`；完整任务均FAIL：`COMPLETE_V3_VALID_TASK_FAIL_NO_SEED`；有成功：按task/scene列精确`SCOPE_LIMITED_SUCCESS_SEED`。工程不完整与有效科学失败分开；失败来源、未执行分母、晚期失败全部保留。不临时增高桌面、改T2contact点、加新source/scene/seed、裁路径/调FSM、调整WBC或换控制器。

必须交六source hash/切片/时间及工具身份；T1高度白名单、完整编译robot属性/接触尺寸、原评分参数与阈值；T2初态/reset实际应用、原geom-ID前缀证据、旧goal消费者身份；所有内部查询/资源实际账；500Hz物理/接触和20Hz实际RGB/参考/installed/actual TCP/原score独立复算；明确FULL Go2＋ARX5/V3/scene/NON REALTIME/真实PASS或FAIL水印的视频。仅保存成功任务为候选成功示范，失败另存；raw/WBC不公开Git/HF。

本文件只交付准备与发布。尚无新scene动态PASS或成功数据，12项新契约未执行、runtime未实施；不新建Issue/PR/monitor/mail，不merge，不真机/保护TEST/DEV-F/GR00T训练。**selected/deployment=NONE。**
