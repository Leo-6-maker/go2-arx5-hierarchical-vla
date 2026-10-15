# ManiSkill固定源码 → 冻结WBC EEF13：具体适配审核

## 结论与执行边界

承接用户提供的研究材料与固定研究提交 `d2379367385a8669f354b5c5c1307b96037de2d0`。**优先复用ManiSkill任务几何与运动规划组件，不替换当前MuJoCo/WBC执行器，不继续无边界扩展自研FSM。** 本文是源码与接口审核，不增加物理／GPU／PBS／GR00T／安装权限；没有运行ManiSkill、没有获得或下载成功HDF5种子。

当前已采用V2独立实跑：三条有效12秒短任务均未预接触／接触，W0/T2在任务开始前就绪中断，非质量结论。V2保存的转向参考累计推进约1.32／1.42／1.80rad，实际角误差最小0.143／0.097／0.171rad；平移参考累计仍为0。两条W1实际BASE前移0.176／0.178m、最终距声明站位约0.03m，仍没有操作。旧V1／旧pilot0/4及WBC保持FAIL不改。

这为“从actual状态计算可执行的接近路径、与站位共同规划”提供了具体需求；不证明外部专家可直接在本体成功，也不提供新的物理GO。

## 1. 实际读取的上游身份

上游 **`mani-skill/ManiSkill@62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`**。只读取8个源码／许可文件，共45095字节，逐文件记录Git blob和SHA256，见 `reports/PI0294/design/MANISKILL_PRIMARY_SOURCE_INVENTORY.json`。未克隆整个仿真栈、未下载数据／权重。当前CoreLab profile中`mplib`和`mani_skill`均未安装；没有更改该环境。

官方[运动规划说明](https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/motionplanning.html)明确示例面向Panda，规划由mplib提供，任务仍需要工程化分解。已用Context7对照该文档和控制器／planner接口；具体行为以以下固定源码为准。

## 2. 不能忽略的源码事实

### PickCube并非本项目完整抓放验收

固定[PickCube solver](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/examples/motionplanning/panda/solutions/pick_cube.py)从物体OBB及夹爪朝向构造抓取姿态，依次接近、下降、闭爪、移动至goal姿态。**它没有在goal处释放、撤离并持续保持的完整序列。** `FINGER_LENGTH=0.025`是该Panda示例参数，不是ARX5实际夹垫几何。

对应[任务evaluate](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/envs/tasks/tabletop/pick_cube.py)要求物体到goal阈值及机器人静止，没有要求释放／机器人分离；goal还可在桌面上方。它可作为接近／抓取／搬运的种子设计，**不能把其success标志映射为我们T1完整成功**。我们原释放、撤离、物体支撑／速度及保持评分全部保留。

### PushCube含固定场景假设

固定[PushCube solver](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/examples/motionplanning/panda/solutions/push_cube.py)先闭爪，以物体位置加`[-0.05,0,0]`作为接近点，以goal加`[-0.12,0,0]`作为后续目标，使用当前TCP朝向。它不是任意推送方向／ARX5夹垫支撑面的通用算法。

适配时必须由实际物体—目标方向、ARX5 TCP和pad支持距离重建接触侧，不直接复制world-X常量；最终仍按本项目真实接触、水平位移、不抬升、不持续抓握、释放分离和保持评分。

### 位姿目标接口背后仍执行关节轨迹

[两指夹爪planner的follow_path](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/examples/motionplanning/two_finger_gripper/motionplanner.py)读取规划器返回的joint position／velocity，加gripper标量后调用`env.step`。[run入口](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/examples/motionplanning/panda/run.py)固定`control_mode="pd_joint_pos"`。

因此不能把返回action改名成EEF13，更不能把Panda关节输出发给师兄WBC。我们需要的是**规划层的Cartesian技能目标与本体可行路径**，输出经原EEF13→native14D软参考接口提交。学生仍不学习joint action。

### 规划可行性不自动等于本体碰撞通过

[BaseMotionPlanningSolver](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/examples/motionplanning/base_motionplanner/motionplanner.py)由URDF/SRDF、move group、link/joint名建立mplib模型，并设置base pose。`use_point_cloud`初始False；示例Pick/Push没有显式调用`add_box_collision`加入我们的桌面／完整Go2碰撞体。不能笼统宣称直接搬来就有目标环境完整碰撞证明。

`move_to_pose_with_screw`在失败时会再尝试一次；`run --only-count-success`会递增seed并丢弃未成功轨迹。**这两项上游默认行为不能继承为本项目无限尝试／挑成功权限。** 新适配须冻结提案／seed／尝试分母，保留全部规划与物理失败。

## 3. 目标模型的实际兼容性

已零物理步读取目前受pin保护的 `arx5_ac1_floating.urdf`，SHA256 **403ce252c525437d66aa361a8a65b90bfe2b69dad0fcb1f7a17109582c7261bc**。其中15个link，6个arm revolute、2个gripper prismatic及固定连接，**没有四足腿部关节／完整腿碰撞几何**。这是现有MPC使用的模型资产，不能单凭它向外部planner宣称完整Go2碰撞可行。

后续adapter应复用原ARX5运动学，同时以当前完整native MuJoCo几何复核身体、四腿、手臂、桌面与有限夹爪；必要的碰撞表达转换属于规划资产工程，不能静默删除腿、锁死base或更改质量／惯量／WBC。`MPC模型存在`、`mplib可以导入`、`全身可执行`是三个独立事实。

## 4. 最小组件适配清单

| 上游组件 | 保留的思想／数据 | 本项目必须实现的接口 | 当前状态 |
|---|---|---|---|
|`get_actor_obb`／`compute_grasp_info_by_obb`|基于物体OBB和接近／闭合方向构造抓取候选|从真实物体pose与现有pad geometry导出，不复制Panda finger常量|SOURCE_REVIEWED，未接入原生Teacher|
|`setup_planner`|明确URDF/SRDF、joint名、move group和root pose|ARX5实际限位、TCP／mount变换、完整Go2/桌面碰撞覆盖；实际移动base时更新frame|TARGET_MODEL_GAPS_IDENTIFIED，未安装mplib|
|`move_to_pose_with_screw`／RRT接口|从**实际**机器人状态到固定技能目标规划有限路径|返回Cartesian参考候选，明确失败；不得把其joint action送给WBC|NOT_IMPLEMENTED，不宣称已有可执行种子|
|`follow_path`|计划—执行—记录的边界|使用原H40/prefix8=0.4秒；新actual输入后再规划；保持20Hz双RGB及原500Hz安全|现有native接口已实跑，外部规划未连接|
|gripper open/close|显式技能动作|归一化方向与原有限行程、actual接触判断绑定；不复制±1／迭代6次作为成功谓词|现有任务gripper有效，外部标量语义待映射|
|源demo／FK|可追溯的计划TCP或技能片段|区分source planned target与actual observation；固定repo/revision、坐标／工具frame、时间、来源机器人|未下载HDF5，不虚构已有seed|
|success与扩采|物体任务结果决定成功示范|沿本项目原T1/T2评分；失败独立bank，MimicGen仅在本体成功seed之后|当前本体成功seed=0|

推荐接入点是**替换任务空间规划组件**，复用现有collector、有限夹爪、双相机、native14D安装和独立CPU评分。不会把另一套SAPIEN/PhysX环境当成当前MuJoCo实际执行，也不整体替换冻结执行器。

## 5. 数据与坐标合同

- 原始来源保持`ManiSkill/Panda`。joint计划若要导出TCP，需源URDF/FK及明确工具frame；不得仅取张量前10／13维或从无关RGB拼状态。
- 两个本体的TCP／pad几何不等价。物体相对SE3公式只在已明确刚性相对锚点的片段使用，工具frame还需显式校准。推送接触、carry/goal、释放不能整条套同一物体刚性变换。
- Panda无Go2 BASE监督；不能零填充3维而称有真实BASE标签。目标BASE意图必须由目标平台规划器实际提交并记录，actual BASE仅作真实输入与执行证据。
- 目标RGB来自目标MuJoCo实际状态；只有真实任务成功、安全、输入／提交动作一致、完整性与split通过的轨迹进入成功BC bank。源Panda成功不替代该门。
- 未来GR00T仍是本体匹配EEF／夹爪／BASE意图，物体oracle／FSM阶段只留审计，不泄漏学生输入。旧M2048保留作初始化资产，改变模态后不伪称exact optimizer resume。

## 6. 下一项科学决定应回答什么

先选**一个有限的W0技能与一套固定规划组件**，定义真实预接触→接触→操作的原生验收、规划失败分母、source revision、full-body collision/TCP绑定和episode／CPU预算，再采用对应新scope。当前V2的原授权不会自动变成“无限迁移ManiSkill”的授权。

本审核不再提出第三轮游标阈值调参，不替换正在执行或已封存版本，不启动新physics／GR00T／PBS。Apache-2.0许可文件已核对并索引；若后续实际复制／修改上游组件，应随文件保留相应许可与修改说明。当前Git只收原创审核与provenance，不收第三方源码、数据、权重或私有WBC。
