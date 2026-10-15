# PI0294 — 局部减速＋源close直接接入：有界整机任务种子 V2

**状态：PREPARED_FOR_EXPLICIT_ADOPTION。** 本次用户授权基于V1负结果准备、发布下一版；未采用本文件的科学执行。`authorizes_scientific_execution=false`、`launch_allowed=false`，运行适配 **NOT_IMPLEMENTED**。本轮新增物理/IK/RRT/渲染/神经查询/训练均0；只有保存数组算术和源前缀只读检查。不得删除或重新开放V1关闭标记。

入口：[任务卡](../../reports/PI0294/design/SOURCE_RETIME_JOIN_V2_TASK_CARD.json) · [条件及逐槽随机表](../../reports/PI0294/design/SOURCE_RETIME_JOIN_V2_CONDITIONS.json) · [来源/冻结锁](../../reports/PI0294/design/SOURCE_RETIME_JOIN_V2_SOURCE_LOCK.json) · [准备证据](../../reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/source-demo-retime-join-v2-preparation/PREPARATION_REPORT.md) · [采用prompt](../handover/2026-10-15/SOURCE_RETIME_JOIN_V2_ADOPTION_PROMPT.md)。

## 1. 依据及唯一两项改动

V1固定结果为 `770989874a59447449d649ed3401340b016b66c1`，Issue #11 canonical **6194051557**。旧终局 `STOP_FIXED_SOURCE_POOL_NOT_ADMITTED`、原及R1审计、75文件私有归档不改：

- T2三个完整源片段在uniform slowdown后需50.494535/53.866355/61.882846秒；峰值跨原first8参考边界，在原submitted轨迹中已存在。
- T1原align首点三源均无满足原生限位的登记解；首次close目标各有1个去重合法goal。接触端有解不等于整条连接或任务通过。

**V2只改：T2时间分配；T1非接触接近前缀。** 不重新调整FSM reached/误差游标、Teacher门槛、WBC或score。两task独立，分别与保存V1形成配对诊断；本版没有两项同时作用于同一task，也不另跑2×2物理消融。

固定完整Go2＋ARX5自由根、W0/seed0、原桌面/物体/source/goal/yaw、BASE软意图0。MPC/腿RL/WBC/质量/惯量/摩擦/gain/dynamics/solver、原SafetyGate/native安全/参考安装/dt2ms/H40/20Hz/first8=0.4秒、原任务score及41点保持全部不变。源未来actual不替代submitted动作，源成功不变成整机成功。

不新增source/外部下载/角度/scene/seed/loss/模型/训练/GPU/PBS/扩采/真机/保护TEST/DEV-F/邮件/monitor/merge/广泛清理。selected/deployment=NONE。

## 2. 原件复用与已完成的只读证据

继续固定六源：自有v3.0 **TRAIN/new**，T2/T1各 `PI0294-TRAIN-F0-000/001/002-T?`，按000→001→002；不补第4条。manifest `b8ddcb9808e6dfac4f3ac5fa5a4828128fa9644a330c17d1c57bf833ce5889df`、stats `fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7`仅身份绑定，不重拟合。复用V1六源641/640/80真实重建、工具矩阵、双锚warp、original成员hash及固定actual输入；不用重采教师或重下载历史包。

本轮只读计算实际结果：

|源|T2局部限速20Hz时间表|T2剩余连接时间上界|T1保留close→hold、沿原uniform factor|
|---|---:|---:|---:|
|000|18.60 s|9.40 s|15.633404 s|
|001|19.40 s|8.60 s|15.252647 s|
|002|19.10 s|8.90 s|15.395897 s|

T2表格只考虑XYZ/SO3速率，是**未含关节/碰撞与连接的时间下界**，不是可行路径PASS。T1表格只表示保留后缀时长，不证明RRT、后缀或动态WBC执行可行。

T1三源close事件均为1.450000000000001秒，原`searchsorted`规则映到实际提交索引 **30/1.50秒**，仍为logical opening=1；首个关闭命令在后续索引32。只读检查原0..30共31帧：无pad–cube接触、无grasp/opposing_contact/lift/carry；已有接触仅cube–table，物体最大位移0.125127mm为原桌面沉降，不能写成绝对静止或全无接触。新切片不会删除已完成的抓取/搬运。

这不是新的canary、IK或物理准入；20项V2运行契约目前 **0/20，待采用后执行**。

## 3. T2：保留全部路径点，只局部延长时间

保留V1原cut、全部重定位pose/rotation、所有gripper值、原source索引及阶段顺序。禁止平滑、平均、裁掉边界峰、重锚actual、抽稀或重定义接触姿态。

相邻源提交点`i→i+1`：

`h_i = max(0.05, ||p[i+1]-p[i]||/0.06, SO3(R[i],R[i+1])/0.3)`

`m_i = max(1, ceil(h_i/0.05))`，`t[0]=0; t[i+1]=t[i]+0.05*m_i`。

只允许与原certifier相同的浮点表示误差处理（比例1e-10），结果必须再次按原0.06/0.3核验，不能把实际超速clip成合格。每个原pose为一个按原顺序保留的时间结点；结点间位置线性、姿态原shortest-SO3插值。源时间只能减慢，不能加速任何区段；大多数区段仍50ms，速率峰值占用更多ticks。source阶段/gripper切换以同一单调映射绑定，gripper命令按原索引零阶保持，经原working-stroke转换及原实际速度/力限制驱动，不篡改actual开度。

原0.7rad/s、acc1、关节限位和原gate仍硬检查。局部cartesian时间表**不保证**角点关节加速度合格；后续原规划/原生证书若拒绝，则保留有效规划FAIL，不追加平滑、样条、姿态搜索或多轮时参优化。不得只验XYZ/SO3后直接称为完整安全轨迹。

完整自由连接必须`entry≤min(10,28-active_duration)`；000/001/002分别≤9.4/8.6/8.9秒。CPU资格锁定源及时间表hash后，warmup实际起点重新验同源，不能换较易源。后续按冻结表消耗源进度，不因跟踪误差/接触成绩暂停或回滚时钟。

## 4. T1：用真实自由连接替代align前缀

唯一接入点取该源**首次close事件后的首个实际提交目标**，仍用原warp/tool，三源均索引30。完整活动片段由旧`[0..257]`改为`[30..257]`，228个提交结点；保留首次close、verify_grasp、lift、carry、lower、release、withdraw及首个hold。只去掉源0..29的非接触align前缀，不在多个时刻中按解或成绩挑接入点。

当前actual完整根/腿/臂/夹爪→原首close目标，由原MPlib/原完整碰撞模型规划自由连接。连接期间夹爪保持原working-open，`approach/align`原pad–cube禁接触规则不变；不得穿过物体“接上轨迹”，不得teleport、冻结根/腿或复制源q。端点必须满足原native FK/限位/碰撞，连接完成按冻结计划时间进入源close，不以新reached阈值作游标。

T1后缀**沿用各源原uniform slowdown factor**，不同时引入T2局部时间法；表中后缀15.63/15.25/15.40秒。自由连接≤10秒、连接＋后缀≤28秒；连接路径合格不代表WBC已实际到达。实际未到、未抓住或后续失败须真实计FAIL，不能再修FSM阈值补救。

### 已有合法goal如何复用

保存V1 `SDR_CPU_T1_i/0000/IK-01.private.json`及R1 native复算引用，不重跑96个CLIK。复用依据逐项是actual packet、root transform、原joint向量、world objects/允许碰撞、目标矩阵、开度/closed标记、约束mask/solver参数、资产/环境hash和原goal seed；endpoint标签从contact改叫join不改变这些语义字段。

新请求记录旧response SHA和消费者等价检查，**不得改旧input_sha冒充新响应**。证据缺失先从75文件归档恢复；identity不等价则本保存输入不能复用，不免费重抽或把旧解当新actual解。CPU阶段只复用相同保存状态的goal；warmup/live状态一变，必须走已扣账的新goal与route。

接入点有解仅解决旧align硬前置，不证明当前actual→该点有无碰撞的RRT，也不保证保留后缀都可执行。实际接触前不得把source grasp或已知未来物体位置写进目标clone；attached碰撞只可基于当前实际native grasp及实测相对变换。

## 5. CPU、候选顺序与固定内部预算

采用后先版本化适配和20项窄CPU契约。新scope `PI0294_SOURCE_LOCAL_RETIME_DIRECT_JOIN_V2`，实际入口版本 `FULLBODY_SOURCE_LOCAL_RETIME_DIRECT_JOIN_V2`，独立注册输出`source-demo-retime-join-v2`；旧scope和hash不动。现有标准RUN_MANIFEST/profile/authority/lease/账本/owner/输出绑定先过，再进入CPU或物理。只在无健康消费者的边界更新未来source capsule。

### 保存输入CPU：2个batch bundle，仍是6个候选分母

每task的三个来源共用同一冻结actual输入，放入**一个**bundle，来源固定000→001→002，不按跨源L2成本重排。T2最多6目标（每源首点approach＋原contact目标）；T1最多3个已保存首close目标引用，新增IK为0。每源内部按原arm L2/固定index取其最佳一个goal/pair。

- CPU T2：总≤6新目标IK、每目标16初值；来源按序，每源最多两原RRT流×两连接段，总bundle **≤8次RRT/60秒**，不是每source各8次。剩余额度不足的后源写`NOT_EVALUATED_INTERNAL_BUDGET`，不能伪造它的质量FAIL。
- CPU T1：复用三源goal，每源最多两原RRT流×一个连接，总≤6RRT/60秒，无新IK。每task CPU bundle至多2次×1001点dense检查，共4004点，不把batch包里的6个候选说成只有2个科学分母。
- 第一条同时满足来源/时间/合法连接及原证书的源锁定；已锁定后不继续搜成绩。未锁定则本task不准入，另一task继续。RRT随机流、无simplify、range0.1、1秒/call、fix_joint_limits=false、MPlib0.2.1/native隔离环境保持。

T2次级contact连接只保留原存在性/几何检查，不偷装第二段RRT代替冻结源技能。完整连接时间来自实际会安装的首段完整route；CPU证书说明原FCL完整规划路径和有限native preview覆盖，不能将first2秒dense称作整个32秒动态保证。后续实际每个新H40继续原gate/native检查。

### 实际warmup和正式循环

每task最多1专用warmup bundle，以真正actual状态复验**同一锁定源**。T2≤2新目标IK/8RRT，T1≤1新目标IK/4RRT；各≤1001dense。不可用warmup成绩换source/pose/seed；合法规划FAIL不是工程替换。

两个正式条件各最多80新观察decision、每decision一个冻结源目标：≤1目标IK×16初值；同一目标最多两个合法joint分支×两原RRT流＝≤4RRT，≤1×1001dense、每点最多5确定性局部IK。新观察必须来自同tick真实RGB/EEF/夹爪/BASE；未来phase/源索引/privileged对象仅给Teacher和审计，不给GR00T输入。

每个调用前持久扣global/子槽账；崩溃/超时/incomplete占槽，不免费重抽。已有实际输入及完整结果只读恢复，输入变化的新调用正常计费，不伪装成cache hit。batch、复用、goal初值、RRT、dense/local IK各自有数，不能靠外层RPC少把内部搜索藏掉。

随机表完整固定于条件JSON：CPU每源用V1原row0/1/2的对应goal与RRT子流，T1已消费goal仅引用；warmup仍row3；正式仍row0..79。无需新增随机数。每task CPU RRT从8个共享槽顺序扣，原seed来源以source/stream/link显式存档，不能对耗尽槽重画。

## 6. 两项完整物理任务与验收

各task自身CPU和actual warmup通过，直接运行原 **T2→T1，各一次32秒**，无新短质量门；原安全warmup≤5秒。健康12秒侧报后继续同episode，不重置history/physics。`fresh observation → 冻结V2源参考/有界规划 → 原gate → reference-only安装 → 原WBC执行first8 0.4秒 → fresh observation`。固定H40，末端hold来自原提交，不用source未来actual或预装task未来代替初始观测。

初始自由连接和源技能用统一执行时钟，接入点落在chunk内部也必须正确推进；时间只计实际消费的prefix，不因暂停等待CPU推进，不在每次0.4秒重置源phase。等待推理/规划暂停自有sim/MPC/history，明确NON REALTIME。

原原生安全、finite、合法碰撞、限位、跌倒、gate REJECT及数据身份保留；REJECT绝不安装。运行有效任务FAIL不重跑本task、不换source/seed，另一task独立；真实native安全FAIL停止V2全部新物理。工程问题同语义自主最小修复，所有raw成本保留；任务已开始后的记录/报告/归档故障恢复原件、不重跑任务。

任务score原样：T1实际grasp、抬升≥2cm/0.3秒、持物搬运≥10cm、正确goal角点±0.07m/高度±5mm/速度、released、TCP清距≥8cm、28–30秒41/41保持及完整32秒安全AND；T2实际桌面接触推动≥10cm、不抬升>5mm、不持续双侧grasp、goal/分离/速度/同保持/安全AND。完整原定义/阈值以冻结score锁为准，不用这段摘要覆盖代码。

WBC跟踪误差只报告，不恢复旧0.05rad保持前置。分别交代planned reference、实际安装、actual tracking、物体接触和完整task成功。没有任务执行则不报0/2成功率；单task成功只能称本scope一个seed，不自动进入扩采/训练。

## 7. 原窗口与剩余预算，不叠加两套额度

读回快照：38启动、353508积分、12248双RGB、15619.141820450666 Core CPU秒，无活动reservation；V1实际3bundle/6目标IK/96初值/0RRT/0dense保留。V1已占原14400 CPU allowance中5.104707980000967秒；本轮只读前缀model metadata已测CPU1.270403526秒，尚未写原科学父账，采用时一次性按来源收据登记，不能重复扣。PC数组算术CPU另见准备收据，未观测的传输/编辑/导入CPU标UNOBSERVED，不声称总体CPU完全测得。

新执行CPU向下取整限 **14390 core秒（含≥1800收口）**，仍受原14400总allowance剩额和父账绝对36844.79211945514的更小者约束；采用时若已有额外消耗再降低。原source-slice天花板43启动/404023积分/13771双RGB不变。

|新V2上限|推导|
|---|---|
|科学条件2、启动≤4|两完整＋最多2次严格任务前同状态工程替换|
|积分≤42012|2×18503＋2×2503，warmup/每启动3尾步收费|
|双RGB≤1282|2×641，任务前工程替换task RGB=0|
|bundle≤164|2保存输入batch＋2warmup＋160正式decision|
|新目标IK≤169；CLIK≤2704|CPU T2 6＋CPU T1 0＋warmup3＋正式160；每目标16|
|RRT≤666|CPU8＋6＋warmup8＋4＋正式160×4；每次≤1秒|
|dense≤166166；local IK≤830830|CPU2×2002＋warmup2×1001＋正式160×1001；每点≤5|
|CPU≤14390、并发≤4CPU/1物理worker|最后≥1800只收口，按实时更小余额|
|新增暂存≤8GiB|不复制旧Trainer/venv/全raw、不删除唯一原件腾配额|
|新GPU/PBS/模型/训练/扩采/下载|全部0|

MP-S2＋SDR V1＋V2最大累计：**199bundle/269目标IK/4304初值/728RRT/197197dense**，分别低于原200/2110/33760/1406/205205；再加旧V1 11011dense为208208≤原232232。旧ManiSkill V1的96初值/7RRT仍单列保留，不扣回。两个CPU batch没有增加每bundle的12IK/8RRT/60秒或每goal16初值上限；所有6个source候选留分母。

用满新物理上限时父账42启动/395520积分/13530RGB；新CPU14390及已知准备1.270403526扣入后，按当前快照父账≤30010.412223976667，不是预算重开。原T0 **1792044600**，新工作截止 **1792083759（10月16日01:02:39 SGT）**，收口 **1792087800（02:10）**。采用/排队/断连不重置；过期只能报告统一资源缺口，不自动续时或重开已关闭lease。

## 8. 实施、停止与交付

采用后唯一owner：刷新Issue #11/#47、真实PID/lease/父账→封存V2人类采用及20项窄CPU检查→T2两级时间与T1直接接入实现/身份→各task自身CPU/actual warmup准入→直接两条完整任务→独立CPU和私有逐文件readback/真实视频。辅助不是签字依赖，不停在CPU_READY，不等待外部包解码或旧WBC保持PASS。

20项待执行检查详见任务卡，重点包括T2所有源结点/速率/夹爪时序不丢、T1索引30/未删除接触技能、旧goal消费者身份、source优先级、跨chunk执行时钟、原gate/score/hash/随机流/累计预算、不重抽及父记录器display。代码断言存在不能替代实际测试收据，CPU通过不能替代动态执行。

如两task均无合法源连接，或已准入完整运行均有效FAIL，封存真实负结果停止本V2；不追加source/姿态/seed/样条或FSM搜索。预算/未完成工程和科学有效FAIL分开；一个task合格不等待另一task。原件唯一源不提前删除，归档故障不重跑科学。

必须交：六源/切片/phase与工具hash，T2逐区段`m_i`时间表及旧uniform对照，T1去掉的30点和完整保留后缀索引，所有缓存引用及消费者等价，IK/RRT/dense/预算完整流，两条件分母、500Hz/20Hz接触/安装/actual TCP/原score独立复算，实际双RGB视频（FULL Go2＋ARX5、V2、NON REALTIME、实际结果）、私有全文件readback、小报告与Issue #11 canonical全文回读。没有新物理就明确无本轮视频，不用固定臂/源示范替身。

**本文件发布只完成准备和发布；未执行新搜索、物理、训练或部署。** 人类采用固定提交后，工程同语义修复在原剩余预算内自主推进，无需每项例行GO；若需新增科学设置/资源/延期，形成一个具体缺口，不能暗改本方案。
