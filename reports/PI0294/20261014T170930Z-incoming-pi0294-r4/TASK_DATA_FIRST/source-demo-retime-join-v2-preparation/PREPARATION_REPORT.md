# V2准备：T2局部时间分配，T1首close直接接入

**已准备，待明确采用固定提交；新科学执行未启动。** [完整方案](../../../../../docs/protocols/PI0294_SOURCE_LOCAL_RETIME_AND_DIRECT_JOIN_V2.md) · [任务卡](../../../design/SOURCE_RETIME_JOIN_V2_TASK_CARD.json) · [逐槽条件/随机表](../../../design/SOURCE_RETIME_JOIN_V2_CONDITIONS.json)。

## 来自本轮负结果的两项最小改动

|任务|保留|只改变|本轮保存数组提供的依据|尚未证明|
|---|---|---|---|---|
|T2|全部submitted位姿/夹爪/源索引与阶段，原warp和速度上限|全局uniform slowdown→逐区段只延长超速区间|20Hz局部限速表18.60/19.40/19.10秒，原uniform需50.49/53.87/61.88秒|完整连接、关节加速度、碰撞、WBC实际接触与成功|
|T1|首次close起全部228点、原uniform factor及完整抓放目标|用当前actual到原首close的自由连接替换前30个align点|三源索引30仍张开；旧保存IK各有1个去重合法goal；原0..30帧无pad–cube接触/grasp/lift/carry|当前起点到该点的RRT、后缀可执行性、实际抓放|

V1终局和Issue #11 **6194051557**永久保留。新scheme没有用后续actual替代submitted，没有把源成功当整机成功，也没有把T2局部时间下界写成路径PASS。

## 实际只读工作

- 从已封存V1读取六个retarget NPZ，传输后逐文件SHA匹配；未重提取整个raw bank、未复制完整WBC或模型。
- PC用纯NumPy复算逐边SE(3)位移/旋转、最少时间及50ms量化、T1事件切点/保留后缀。算术源码：[source_retime_v2_analysis.py](../../../../../scripts/pi0294/source_retime_v2_analysis.py)。这是分析工具，不是运行Teacher，也没有新增dispatcher或部署。
- 只读三份T1原samples的0..30帧；读取其交付source XML建立零物理步模型，只查geom名字来区分cube–table接触与pad–cube接触。三者只有cube–table，最大物体沉降0.125127mm，无实际抓举。没有mj_step、renderer、IK、RRT、神经查询。
- 当前主科学父账读回38启动/353508积分/12248双RGB/15619.141820450666 Core CPU秒，CPU reservations为空；未修改旧runtime、source capsule、数据、stats、原响应或关闭收据。现场状态必须在采用时再刷新。
- 21个冻结公共源码成员按Git原始blob核对一致；Windows工作树CRLF不用于冒充Linux原始字节hash。私有native身份复用V1已发布的逐项回读，下一次采用前仍需现场核验，不因新文档出现就声称已重新运行WBC。

准确算术在[SAVED_ARRAY_ARITHMETIC.json](SAVED_ARRAY_ARITHMETIC.json)；源前缀检查在[T1_SOURCE_PREFIX_AUDIT.json](T1_SOURCE_PREFIX_AUDIT.json)。PC算术process CPU与remote model metadata process CPU分别在收据记录；后者1.270403526秒，尚未写原父账，采用时有来源地登记一次。传输/编辑/未计时导入CPU为UNOBSERVED，不能称全部计算为0。

## 预算准备与执行目标

新最多两条完整32秒任务、≤4启动/42012积分/1282双RGB、14390新增远端CPU核秒（含1800收口）、1物理worker/≤4CPU。保留原工作截止10月16日01:02:39 SGT及02:10收口，不从采用时重新计时。

用同一个保存actual输入的每task三个候选做一个CPU batch，但**6个来源仍是6个科学分母**。每bundle原12IK/8RRT/60秒硬cap保留；T1只引用原相同输入下的goal，不重做旧CLIK。连同warmup和正式80×2，新增最多164bundle、169目标IK/2704初值、666RRT、166166dense点；详细逐项推导在协议和任务卡。不能把每个source另给8RRT，耗尽后未检查项单独记录，不能写FAIL。

采用后，两task各自合法即从CPU/warmup继续完整任务；不加新短质量门、不等另一task、WBC保持RMS或辅助签字。物理任务有效FAIL按原规则停止，未开始的另一task独立；原生安全FAIL停该版本全部新物理。只允许原有限的同状态任务前工程替换，任务开始后的报告/图像/HF故障不重跑。

## 当前发布状态

|事实|状态|
|---|---|
|六源保存数组及三段原前缀只读准备|已完成|
|V2完整协议/任务卡/源锁/条件及采用prompt|本次发布|
|V2 runtime适配/部署|NOT_IMPLEMENTED / NOT_DEPLOYED|
|V2正式CPU准入20项|0/20，尚未执行|
|新增IK/RRT/physics/render/神经/训练/GPU/PBS|全部0|
|新真实接触/完整成功种子|未取得 / 0|
|科学执行授权|authorizes_scientific_execution=false，launch_allowed=false|

**发布目标是一个可采用、可核查的有限任务，不是又一轮FSM阈值搜索。** 采用后推进到原物体任务的真实接触、完整成功种子或明确有效停止点；目前仅完成准备，selected/deployment=NONE。
