# PI0294 — 有界候选搜索与交互示范备选：真实接触到完整成功种子 V2

日期：2026-10-15，Asia/Singapore。唯一新scope：`PI0294_MANISKILL_SEARCH_AND_ASSISTED_SEED_V2`，简称 **MP-S2**；不是此前FSM component-progress V2。

**PREPARED_FOR_ADOPTION / NOT_AUTHORIZED。** 本次用户授权准备并发布方案，没有采用新的科学执行。机器卡 `authorizes_execution=false`、`launch_allowed=false`；没有新安装、IK/RRT、渲染、物理、模型或训练。未来明确采用本固定版本后，owner按下列确定路由连续执行；不在CPU_READY或等待辅助签字处结束。

配套：[任务卡](../../reports/PI0294/design/MP_SEARCH_V2_TASK_CARD.json) · [候选与搜索合同](../../reports/PI0294/design/MP_SEARCH_V2_SEARCH_POLICY.json) · [条件/种子表](../../reports/PI0294/design/MP_SEARCH_V2_CONDITIONS.json) · [源码与冻结身份](../../reports/PI0294/design/MP_SEARCH_V2_SOURCE_LOCK.json) · [准备事实](../../reports/PI0294/design/MP_SEARCH_V2_PREPARATION_RECEIPT.json) · [接续prompt](../handover/2026-10-15/MANISKILL_SEARCH_V2_ADOPTION_PROMPT.md)。

## 1. 问题、已有证据与此次改变

旧组件系列实际报告固定于 `4de048a66e5a10611180e61025ddacb2c54aa2ce`，Issue #11 canonical **6191161535**。24项CPU接口检查通过；12个固定保存输入中7个可规划、5个失败；两个起点各8候选均没有原限位合法解。真实物理/接触/成功seed均0。旧状态 `STOP_FIXED_START_NO_ADMISSIBLE_REGISTERED_IK`、输入/结果/归档和父账永久保留，不能改判或重开旧scope。

完整native导出的身体、腿、臂、夹爪及碰撞资产已经建立；本次不重复声称缺腿几何。已保存候选原生FK复核支持frame一致，只说明旧固定目标/候选集未合格，不证明全局无解，也没有证明WBC不能完成物体任务。

本次检验：**预先定义少量物体/工具几何候选＋多起点IK＋有限RRT连接，能否通过同一冻结执行器取得真实接近/接触，进而完成原T1/T2？** 失败候选属于内部搜索分母；完整episode属于外层分母，两者不混用。候选和搜索预算是新的科学合同，不伪装成旧系列的工程修复。自动短任务失败后允许按预定路由尝试一次人工辅助完整示范；不把全自动Teacher作为唯一数据来源。

不引入新场景、物体、颜色、reset seed、规划器、WBC或任务评分。不以旧保持RMS为全局前置，不继续调整FSM的reached/cursor阈值。结果不构成新旧严格配对因果对照：新几何/内部搜索及可能的人类输入均需如实披露。

## 2. 冻结边界与复用

- 完整自由根Go2＋ARX5、原有限夹爪、原MuJoCo场景、双RGB、腿RL、OCS2、原生安全observer、原SafetyGate、reference-only接口和时间协议保持。W0真实提交零BASE速度偏好，不焊接根、不假称BASE固定；本版不增加站位搜索。
- WBC原交付包SHA256 `6dd13098de5a58327fb0a0a5b3ff749e0a5534510375b92fba3f8fb50a322d52`；复用已部署source/build/config成员，不能仅凭tar包名当实际身份。源码锁另列原scorer/gate/plant/clock等逐文件SHA。
- 复用已安装的独立MPlib0.2.1环境、完整依赖artifact锁、私有planning-only URDF/SRDF与TCP/pad校准；原生NumPy2环境不变。主源ManiSkill `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`，MPlib `86376ca87d2c055b792d8cea4d15a38e861d866f`。不整体安装SAPIEN/PhysX，不转关节动作执行。
- 只修改新scope的自有候选生成、搜索编排、候选局部几何事件、交互输入和旁路审计。保护旧源/结果；不在健康runtime热改。新版本必须独立RUN_MANIFEST、lease、源码seal、预算和输出绑定，经已有enter/doctor/launcher进入。
- 源码已发布不等于已部署；本版上述新功能均 **DESIGN_NOT_IMPLEMENTED**。实施时复用旧代码及24项不受影响收据，仅核对变更面，不重装成熟环境或重读历史大包。

## 3. 固定候选：六个任务几何包，预接近与接触成对

每个actual包含原native root/legs/arm/finger状态、物体pose、TCP、同帧RGB和时间。所有计算只在规划克隆中；不得改变真实qpos、物体或实际观测。单位m/rad，world坐标；native xyzw、MPlib wxyz、磁盘row-6D转换沿旧合同。

### T1 抓放：2个水平闭合轴 × 3个接近倾角

1. 从实际cube OBB取两个水平轴 `u,v`（原局部X/Y，按轴索引，不按分数挑选）。闭合轴 `c` 的正负只按与本次初始actual TCP-Y点积为正确定，零tie取原正轴；每个episode冻结这两个有向轴。
2. 基准接近 `a0=(0,0,-1)`。倾角按 **0,+30,−30度**固定顺序；`a=Rot(c,theta)*a0`，TCP旋转列为 `[a,c,a×c]`，保持右手正交。六包索引为先u后三个角，再v后三个角；不是运行失败后手调角度。
3. 用原native两个pad在**零机械行程**的工具坐标中心均值 `b`，定义接触位姿 `p_contact = actual_object_center − R*b`；这是新明确的几何中心对齐合同，区别于旧单目标的高度构造，不是WBC误差补偿。开度沿原1=open/0=closed机械映射，不二次缩放。
4. `p_approach=p_contact+(0,0,0.08)`。同一包的approach和contact都须检查原限位、native FK、完整碰撞及开放pad对cube的空间容纳；仅预接近可达而接触不可达不准入该包。保持真实pad不对称，不复制Panda finger常量。
5. 夹爪闭合依据**实际pad与物体包围/重叠几何**，不是reference到点或时限到达。为适配u/v及倾斜，新的局部几何helper在所选闭合轴/正交平面投影实际pad和cube角点：两pad在闭合轴两侧包围cube区间，另外两轴有实际重叠，并满足原安全/接触允许规则；原不等式不加松弛。不能继续使用硬编码“物体Y就是闭合轴”的旧helper拒绝合法u候选。该helper在首个任务前冻结，不按任务成绩改阈值；它不是最终scorer。

### T2 推送：固定推送方向，六个工具滚转

1. 方向 `d=normalize(goal_xy−source_xy)`不变。以旧 `push_geometry` 的正确工具旋转为 `R0`，新旋转 `Rot([d_x,d_y,0],phi)*R0`；角度按 **0,+30,−30,+60,−60,180度**固定顺序。
2. 每包重新从真实closed-pad几何计算沿d的front support和world-Z bottom support；不得旋转TCP但沿用旧支撑常量。cube方向支撑来自actual OBB。
3. 接触TCP水平坐标为 `object_xy − d*(cube_support+pad_front_support) +0.001*d`；高度 `table_top−pad_bottom_offset+0.004`。approach再沿−d退0.025m、向上0.08m；align沿−d退0.012m。推送、分离0.04m、撤离0.13m沿原距离与阶段规则。不能改推送方向或缩小cube。
4. 几何不成立、桌面/全身碰撞、pad高度不合适的包登记拒绝，不为了保住六包改变物理资产。必须保留全部六包分母，即使构造后重复或无效。

### 包选择与后续执行

开放间隙、物体/桌面接触关系按原阶段显式定义；carry附着只能由actual持续grasp触发。非允许接触、非有限、原限位越界均不可安装。

选择第一个通过下述固定排序、RRT及密集窗口检查的包，**锁定到该episode**；后续根据新actual更新同一几何包的物体相对目标、实际grasp变换和连接，不在停滞后切角度追成功。carry/lower参考由actual grasp时物体→TCP变换构造；释放/撤离/保持仍按原任务。阶段事件沿真实接触/抓握/物体结果，不能由规划成功或预计耗时冒充。

## 4. 内部搜索、计数和选择规则

一个外层bundle绑定 `(scope,condition,query,actual_SHA,stage,mode)`，同一输入最多执行一次；工程重启读取已封存子结果并接续未消费的子槽，不从头重新抽样。

|层级|单bundle上限与规则|
|---|---|
|任务候选|6包；approach/contact各一端点，共≤12个不同目标pose；已锁定阶段只计算所需目标，未用槽不能转新角度|
|目标IK|每pose最多1次固定MPlib IK，`n_init_qpos=16, threshold=1e-3`，actual为首初值；≤12 IK调用、≤192个实际CLIK初值。重复解、未收敛、限位/碰撞拒绝全部保存|
|几何包排序|先过滤非法；各包最优合法approach/contact配对按原6臂关节欧氏距离 `||q_a−q_actual||+||q_c−q_a||` 排序，tie依次包索引、两个IK解索引；无wrap距离、无关节clip|
|连接与fallback|取排序前2个合格包/分支配对，各最多2个预定RRT随机流，按包0流0→包0流1→包1流0→包1流1；每route最多actual→approach及approach→contact两段，合计≤8次RRTConnect。单目标阶段相应只有一段。失败是内部候选证据，不立即终止外层|
|固定RRT|每次`planning_time=1s, rrt_range=0.1, simplify=false, fix_joint_limits=false, time_step=0.05`。不用新算法/screw、参数扫参或额外seed。返回轨迹每段最多1024行，超出登记预算拒绝，不截短伪装完整路径|
|选择|依上述固定顺序，首个完整连接通过全身检查、时间参数化、H40及原gate的route获选；不物理试跑多个route再挑优。其余标UNUSED，不记成FAIL或完成搜索|
|密集窗口|最多对2条候选窗口各检查1001个2ms点；继续使用旧每点≤5次确定性局部IK修正、无随机重启，单bundle≤10010次局部修正迭代。它们另记账，不能藏在192个全局IK初值中；密集失败可转下一个预定route但全局点数上限仍生效|
|超时|bundle≤60秒墙钟，含IPC/IK/RRT/认证；各内部子结果先原子落盘。超时记TIME_BUDGET_INCOMPLETE，不当全球IK无解；不延长时限重试|

种子派生：`uint32_le(SHA256("PI0294_MP_SEARCH_V2/"+condition+"/"+query+"/"+component+"/"+slot)[0:4])`。component分别IK/RRT；IK slot0..11，RRT slot0..7。条件表封存所有222个外层槽的20个流种子；16个IK起点及实际CLIK次数需由proxy记录，不能只写理论n_init。接触端IK起点仍绑定同actual；配对/碰撞验证需同时检查两端，RRT从选定approach q连接contact q，不假称检查了未执行的全局联合优化。

物理reset seed始终0。内部搜索seed不根据成绩变更。重复数值目标且输入/版本/seed完全一致时复用first-result；跨版本、不同种子或actual字节不同不得伪造缓存命中。历史V1仍为独立负结果。

## 5. 执行与交互示范备选

### 共同执行

每0.4秒新actual观测→planner→原SafetyGate→reference-only安装→原WBC执行prefix8→新观测。规划中沿原pause协议冻结自有sim/MPC/history，标注NON REALTIME；保持原actual timestamp，不能用墙钟刷新旧包。H40/20Hz、原dt2ms、原速度/角速度与统一延时方式不变：EEF≤0.06m/s、≤0.3rad/s，T2 contact/push/detach≤0.04m/s，原规划关节上限≤0.7rad/s、加速度1rad/s²。关节轨迹只作规划/FK，不发送q/dq/torque到执行器。

### 人工辅助范围：备用的真实参考输入，不是自动成功补丁

只在第6节确定路由到ASSISTED时启用。拟本机loopback交互面板复用已有同帧body/wrist RGB、actual TCP、物体几何预览；只编辑**目标ghost**位姿/开合，不拖actual关节、root、物体，不回滚/teleport，不重置episode/history。面板/控制链当前未实现，不宣传已有teleop。

- 操作者必须是实际人类；没有操作者时标`NOT_AVAILABLE_OPERATOR`，不能让LLM/agent替代人并写成human。无需为自动lane等待操作者。
- 每完整episode≤128次位姿/开合编辑，≤80个0.4秒决策；下一决策只提交当前一个目标，期间可有多次未提交ghost编辑。规划/拒绝也消费该次决策/bundle预算；同一包内手动目标不再枚举额外六包，单pose仍≤16 IK起点，沿同一RRT上限。
- 单次位移编辑≤0.02m、旋转≤0.15rad；可重复有限编辑形成waypoint，但所有编辑日志和未提交预览保存。这个界面步长不提高原执行速度上限。开合沿原机械映射；BASE意图固定零，不开放额外站位参数。所有提交需相同fullbody规划/原gate/时钟/原安全认证。
- 操作者只依据当前及历史actual观测，不接触保护TEST/DEV-F，不看未来反事实成功。每episode人工思考暂停累计≤900秒，原绝对窗口仍优先；暂停只复用旧RGB，不追加未记账渲染。
- 输入超过上述边界、暂时断连或面板异常：保留证据并安全停止该lane；任务开始后不使用工程替换额度重跑。用户取消或长期无操作不是有效任务成功；记录具体incomplete类型。
- 记录`demonstration_source=HUMAN_ASSISTED_SIM`、operator事件、实际提交/安装/执行标签；AUTO记录`AUTOMATIC_PLANNING`。成功仍必须原物体scorer和原生安全AND，不将人类判断当评分。

本版不下载公开示范，不安装MimicGen/SkillGen、不扩采；这些保留为后续独立方案。交互示范可成为后续来源，但不假装已有成功轨迹。

## 6. 四个逻辑条件与唯一确定路由

只用原W0/T2、W0/T1，原seed0、source `[0.45,-0.08]`、goal `[0.45,0.08]`、table center x0.515、cube yaw0；新条件ID，不覆盖旧cards。

1. **P0工程**：复用已闭合24项收据；只实施搜索/候选/新scope/交互的窄边界核验。准备完整源、合同、环境、实际成员identity和预算seal。必须验证单臂独立路由、内部失败→下一候选、预算耗尽拒绝、计数跨恢复不重抽、ghost编辑不改变actual、禁止joint命令及原score晚期失败保留。仅CPU fixture，不运行上游Panda物理当额外对照。
2. **P1有界零物理步搜索**：只取原12个保存输入中的**两个原首帧**，各执行一次新bundle。旧10个后续输入/原V1结果直接复用旁报，不再全部重规划。两任务**各自**合法首段准入各自AUTO短任务，不再以“两者同时通过”阻塞合格单任务。单个内部候选失败继续其剩余固定搜索；穷尽才标该任务P1 FAIL。
3. **P2 AUTO短任务**：固定顺序`MPS2_DEV_W0_T2`、`MPS2_DEV_W0_T1`，各≤5秒正常warmup＋12秒任务，各至多一次；P1未合格者NOT_ADMITTED。真正fresh warmup结束actual须重新规划并认证，保存首帧不是新episode的伪actual。
4. 短里程碑沿旧标准：完整12秒/数据/时钟/安全/安装一致，actual接近≥2cm、原实际安全预接触及pad↔cube接触；T1 actual grasp≥0.3秒；T2 actual水平推动≥1cm，未抬升>5mm或持续grasp。短PASS不是完整成功seed。原生安全FAIL停止本版本所有新物理，不转交互绕过。
5. **P3每任务唯一完整seed槽**：固定顺序`MPS2_SEED_W0_T2`、`MPS2_SEED_W0_T1`，各≤5秒warmup＋完整32秒任务，各一次。自身AUTO短PASS→AUTO；自身P1穷尽或非安全的短FAIL/incomplete→如CPU交互合同通过且真实操作者可用则ASSISTED，否则NOT_ADMITTED/NOT_AVAILABLE。mode在该slot启动前落盘并不可变。人类首个目标的合法首段须先在live暂停actual上认证，再开启task时钟；不能把未准入的非法参考安装到物理。
6. AUTO完整任务若有效FAIL，该task完整槽已消耗，**不能再切ASSISTED或换候选重跑**；ASSISTED完整FAIL同理。短任务的失败与随后预注册不同来源的完整任务均保留，不能只统计成功full。两任务各自推进，不等另一任务PASS。
7. 任一完整slot按原score、安全和数据准入通过，即封存该W0任务单条成功seed；另一任务继续自身固定槽。停止于`SEED_QUALIFICATION_COMPLETE_OR_VALID_FAIL`，不等待双任务完美才认可单技能数据。没有本版训练权限；可旁路准备codec合同，但不能自动启动小样本训练。

两条短+两条完整共四个逻辑条件；AUTO/ASSISTED是完整槽的互斥路由，不额外加episode。最多另有2次仅任务前工程替换：≤5秒warmup、Teacher命令0、task RGB0、有同语义工程证据；同scene/seed/mode替换，原件/账保留。任务开始后工程中断不是免费重试，有效安全或质量失败不替换。

## 7. 原评分及证据层级

原`target_sim/scorer.py::score_episode`与`pi0294/score.py::score_push`逐字冻结，scene/camera、task_contacts、SafetyGate及native observer身份绑定。只新增旁路索引/来源字段和计数映射，不能改scorer接受谓词。

- **T1**：actual持续grasp、抬升≥2cm/0.3秒、grasp下搬运≥10cm；原goal角点±0.07m、桌面高度±5mm、线速≤0.02m/s、角速≤0.10rad/s、released、TCP清距≥8cm、原28–30秒41/41保持、完整32秒/641样本和安全AND。
- **T2**：actual接触/水平推动≥10cm，不抬升>5mm、不持续双侧grasp；原goal/高度/速度、released/robot_separated、TCP清距、41/41保持和高频接触/安全AND。原W0 minimum BASE displacement0不变。
- 报告依次区分几何候选、合法IK、连接/窗口、gate接受、reference安装、实际tracking、接触/操作、完整任务结果。规划路径不保证WBC采用同一joint分支；原真实安全持续裁决。无selected MPC消费证据仍写UNKNOWN，不因此开启WBC归因项目。
- 全部内部候选/实际成本、所有逻辑条件/实际启动/工程中断/外层FAIL分母保留。成功BC seed bank可只收成功，但attempt ledger不删失败。AUTO和ASSISTED不合并成自动Teacher成功率；本组是见过的W0开发任务，不称独立泛化测试。

## 8. 资源与绝对窗口：不叠加旧额度

本次只发布；以下是**待采用的新搜索合同上限**，不是已获准执行。2026-10-15 19:50:24 SGT只读现场确认：旧scope仍关闭、CPU reservations空、父账 **37启动/345005积分/12007双RGB/15270.13819312814 Core CPU秒**；源 `59045165b518837fc1cf8fb369f52752caa31ee5`。此快照不是未来launch证明，采用时刷新owner/PID/lease和父账，不接管其他进程。

|项目|本版拟新增硬上限|
|---|---|
|逻辑条件/实际启动|4条件，≤6启动（含2任务前工程替换）；AUTO/ASSISTED共享|
|积分|≤59018，含warmup及每启动3步尾部；2×8503＋2×18503＋2×2503|
|双RGB|≤1764：2×241＋2×641；无额外UI/调试渲染|
|外层规划bundle|≤222：2个固定首帧CPU＋≤220个物理决策；hold/cached不新抽样，原12bundle仍保留独立历史，不重置累计|
|全局IK/RRT|新≤2664个目标IK调用/42624个实际CLIK初值/1776次RRTConnect；分别为222×12/192/8。旧96初值/7RRT保留；这是明确的新内部搜索配方，不沿用旧1856/232上限来暗中扩量|
|密集点/局部IK|新≤221221点，旧11011保留，合计≤原232232；每点≤5次确定性局部修正，新≤1106105迭代。单bundle≤2002点，所有候选与工程重复也消费总账。不能按222×2002假称全部有额度|
|CPU|新≤**21574 core秒**，不超过旧21600减已实际25.346073673秒；父账科学CPU≤36844.79211945514秒。至少保留1800core秒作只读评分/封存，剩余不足先停止新规划/物理|
|并发/存储|≤4自有CPU cores、1物理worker、CUDA隐藏；新增暂存≤8GiB，不复制完整旧环境/历史大包，不广泛清理|
|窗口|原T0 `1792044600`（10月15日14:10 SGT）不变；新工作沿更早已绑定 `1792083759`（10月16日01:02:39），父账绝对新工作门01:10不延长；收口≤`1792087800`（02:10）。不从新采用时重给6小时|
|GPU/PBS/GR00T/训练/扩采/公开权重|全部0；raw/WBC私有，不安装其他仿真栈或下载外部demo|

积分/启动/RGB花满的父账算术预览仍为43/404023/13771，**不是已执行或预留**。V1 CPU以真实增量扣除，不能把移动文件/缓存命中写成释放资源或扣减旧预算。

内部上限是逐项独立的硬门，不承诺都能花满：CPU/密集点/时间先到，未执行的槽标`NOT_RUN_RESOURCE_LIMIT`，已开始任务按incomplete原件报告；不把资源不足写科学质量FAIL。每次dispatch先预留自有bundle最坏所需资源，子项执行前扣账，缓存恢复不得重复收费或免费重抽。接近截止不启动无法留下安全停止/收口余量的新阶段；过期只交精确缺口，不暗中续时。

## 9. 工程自主修复、停止与交付

同科学合同的路径、IPC、序列化、counter/恢复、UI ghost和依赖兼容问题可在剩余CPU/窗口/既定任务前替换额度内自主最小修复；逐次记录，不设额外任意修复次数门。不得改角度集合、代价、种子、限位、gain、接触允许表、速度、teacher阶段阈值或最终score。真实非法初态/原生安全FAIL停止所有新物理，保存原因；不得通过人工输入规避。

两个task全部候选/允许路线穷尽且无seed→保留真实负结果停止，不继续增加FSM阈值、场景、姿态或预算。一个有seed另一个无→单技能seed明确封存，另一个真实缺口报告；不夸称双任务成功、泛化、实时、TRAIN64就绪或策略部署。

必交：
1. 新scope采用/源/环境/资产/clock/gate/scorer成员收据，复用与变更逐项列明。
2. 两个首帧的六包完整目标、所有IK初值/解/碰撞/限位、RRT随机流/失败/耗时、first-result SHA、排序与选择依据；真实计数含恢复/超时和未用槽。
3. 四逻辑条件、每次实际启动/积分/CPU/RGB/人类事件、0.4秒新观测/提交/安装/执行标签与实际接触/物体score；独立CPU从保存数组重算。没有物理就明确0，不以CPU_PASS代替交付。
4. 原始双RGB和actual TCP/object/contact时间轴视频，标完整Go2＋ARX5、AUTO或HUMAN_ASSISTED_SIM、NON REALTIME、真实结果。无task RGB不造视频，不用理想FK/上游Panda或旧视频代替。
5. 大原件私有逐文件固定身份readback，小收据/完整负结果到GitHub Issue #11 canonical并回读；归档失败只恢复封存，不重跑任务。保留唯一源直到完整回读。

明确采用后自动从变更面接入推进到本任务实际contact/full seed或真实停止点，不在READY或辅助审核处暂停。当前发布仅是可审核方案；无本次执行结果。selected/deployment始终NONE，邮件/monitor、真机、保护TEST/DEV-F、WBC训练与merge关闭。

## 10. 原始依据

- [MPlib固定0.2.1源码](https://github.com/haosulab/MPlib/blob/86376ca87d2c055b792d8cea4d15a38e861d866f/mplib/planner.py)：IK/连接是不同内部操作；本版预算和排序由本协议明确规定，不冒称上游默认最优。
- [ManiSkill规划说明](https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/motionplanning.html)：现成机器人示例与可复用规划工具；不把Panda动作层或成功定义直接迁移。
- [ManiSkill交互示范说明](https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/teleoperation.html)：目标ghost/规划/开合的交互思想；本项目面板仍需适配验证，不承诺直接运行其Panda脚本即可用。
- [项目跨本体研究](../reviews/2026-10-15_MANISKILL_TRADEOFFS_AND_CROSS_EMBODIMENT_VLA_WORKFLOWS.md)、[旧组件实际终局](../../reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/maniskill-component-v1/REPORT.md)。

下一次采用应引用本方案发布后的固定提交，不引用可变HEAD。若原绝对窗口已过，只有单独明确的时钟/资源授权可改变它，本文件不自动赋权。
