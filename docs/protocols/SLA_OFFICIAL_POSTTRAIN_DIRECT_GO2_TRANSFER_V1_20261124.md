# SLA_Memory 官方后训练权重 → Go2+ARX5 冻结 OCS2/WBC：直接迁移验证 V1

2026-11-24 · **READ-ONLY ASSESSMENT + PLAN ONLY**。依赖目标项目 [PR #428](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/428) / [#430](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/430) / [#437](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/437) 已封存结果。本独立分支不更新SLA权重、冻结OCS2/MPC/WBC/PD、H40/G7/夹爪原门，不创造未经许可的GPU/物理任务。本轮实际新增GPU前向、原生请求、物理tick、训练更新均为0。

## 0. 最重要的事实：之前测试的已经是官方后训练 checkpoint

上游固定来源是 [XPolicyLab SLA_Memory PR #237](https://github.com/XPolicyLab/XPolicyLab/pull/237) head \`c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454\` 的 \`policy/SLA_Memory/\`；官方 [README](https://github.com/XPolicyLab/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/README.md) 明确官方RoboDojo real后训练来源：\`haofuly/SLA_benchmark_ckpts\` 子目录 \`SLA_Memory_robodojo_real_ckpt/\`、Hub revision \`7f6880569c3be777291e646fad1b4951245e8553\`，初始化于 \`haofuly/SLA_base\`（base revision \`5bbdf3012ab643554aa2eb83ddeae9ace3dce476\`）。官方导出含参数和checkpoint自身norm stats，禁止把 \`SLA_base\` 当成已后训练模型。
- 对已发布real后训练checkpoint，\`train_config_name=train_robodojo_real\`；\`extra_frames_first=false\`，历史token位于当前三视角之后；\`action_type=ee\`；\`attention_impl=xla\`可以运行在非Hopper CUDA，\`xla_cudnn\`要求Hopper。缓存没有因GPU限制而证明模型不兼容；不要用xla_cudnn在不支持的GPU上运行。
- 公开训练配置提供 \`train_robodojo_real\`/ \`train_robodojo_sim\` 两种后训练**recipe**；官方脚本直接下载/评估的**已确认release**为上述real checkpoint。尚未核实任何另一独立官方、可直接加载的 \`Go2\` 或 \`SLA_Memory_robodojo_sim_ckpt\` 权重，不能自行发明。
- 模型数据：三路RGB当前相机 \`cam_head/cam_left_wrist/cam_right_wrist\`，九张head历史（episode首帧+过去8秒每秒一帧），双臂状态16D；每臂 \`[xyz,qw,qx,qy,qz,gripper]\`，左先右后；50步×16D输出，实机\`arx_x5\`按30fps构建历史。每臂姿态基于其各自的来源臂基座，而非Go2真实世界。官方 \`RobodojoOutputs\` 会在output transforms完成反归一化／增量动作→绝对动作／来源frame逆变换；不要再把网络原始训练空间或两次反归一化当EEF物理动作。
- 项目历史P0/P1/P2确实已经让**同一官方real后训练权重**在目标Go2观测上生成动作：V1.2 source/target model forwards有实际回执；旧H40/G7学生原始0/16；V1.3 retime 16/16 pose dynamics,完整H40 8/16, H40+G7 2/16,有效任务/事件0/16；B3目标13/16有加速度窗问题、夹爪0/16合法student owner。**不能重新下载同一份checkpoint当作“新模型后训练测试”，也不能把未安装学生轨迹说成“WBC抓取失败”。**
- PR#437冻结OCS2＋有界task-space feedback 在**固定TRAIN01无接触已知目标**达4/4配对、位置<=4mm、姿态<=2°持续1s；收敛需要9.8–10.0s，终点发出的EEF参考相对任务目标约34mm。**仅认证该静止终点工作点，不能外推到1.63s SLA技能的移动物体、接触或抓放。**

## 1. “直接用于本体”必须分层判断

| 命题 | 判定 | 依据 |
| --- | --- | --- |
| 无需改变官方后训练权重，即可加载和执行真实SLA forward | **YES, previously verified** | 原P0/P1，官方加载/processor契约 |
| 无需任何观测适配、直接将Go2传感器字典喂给官方model.Model | **NO** | Go2两路视觉/单右臂/20Hz，不符合三相机/双臂/按30Hz帧计历史 |
| 无需EEF/时间/GRIP适配，直接将50×16输出送入OCS2 | **NO** | 现场接收器要求41×50ms世界系EEF+base，固定会话seq/0.4s TTL；夹爪另通道 |
| 使用来源官方real后训练权重+已核验的Go2 Adapter而不做新后训练 | **MODEL_FORWARD+POSE_MAPPING PASS, TASK UNKNOWN** | 原前向存在；没有学生闭环操作 |
| 在冻结OCS2+F2 A1下完整抓取/搬运/放置 | **NOT_RUN, NOT_QUALIFIED** | 仅静止终点A1，缺动态接触资格、student-owned gripper ACK |

## 2. O0：零新GPU/零物理的身份与现有证据复核（首项立即执行）

现场按完整SHA锁住实际 checkpoint \`params/\` + \`assets/train_robodojo_real/norm_stats.json\`，加载source commit与model asset一致。不要凭文件名推断同权重；若有多个加载目录逐个比较完整的model artifact manifest。锁住 \`extra_frames_first=false\`、\`attention_impl\`、\`env_cfg_type=arx_x5\`、\`action_type=ee\`、native fps 30与Go2真实fps20的区别。
- 重用以前真实的source肯定性推理、Go2右臂50×8物理动作，六条source专家和目标16窗；比较同一source checkpoint SHA。能用原已算结果就不要重跑新GPU。
- 对官方model.Model的 \`update_obs\` 核验：缺任一路 \`vision/cam_*/color\` 会抛出异常；直接Go2没有左腕图像。提供虚拟左腕仅是**明确标识的OOD替代**，不是获得真实左腕观测。原source侧消融显示缺左图右臂路径有显著偏离。
- 官方 \`RobodojoMemoryBuffer\` 依赖接收帧计数：\`fps=30\`、当前Go2视频为20fps，若不做逐物理时刻重采样，历史步距会从真实1秒变成1.5秒。必须用带真实timestamp的历史采样纠正或提供真实30fps历史；不得简单复制并称为30fps真观测；保留相机来源和缺帧掩码。
- 记录 \`OBS_SLA_DOMAIN_SHIFT\`（左腕缺失+虚拟左臂状态）、\`LANGUAGE_TASK_SHIFT\`（水果/源域任务与StackCube差别）、\`TIME_MAPPING\`（原生1.633s与新执行时钟）、\`GRIPPER_OWNER\`、\`ACTION_SEAM\`，不允许忽略任一轴。

## 3. O1：官方后训练权重标准重放与非接触Go2 shadow

先使用过去已生成的官方后训练模型输出；对同一16个封存TRAIN候选，在不发生新模型前向时复用source checkpoint、状态/相机、pose/timestamp、previous consumed command、gripper和contact的私有源数据。分别出报告：
- \`DIRECT_OFFICIAL_RAW\`：保持官方后训练输出、官方inverse、单右臂EEF/TCP映射；原旧H40/G7照样评分，保留历史0/16。
- \`OFFICIAL_WITH_CONTINUOUS_REF\`：只修改未来command reference anchor/时间编译，严格保留原source几何和抓放事件，不让teacher真值补SLA动作；报告Path preservation/动态/非平凡 P10移动与事件误差。使用新版41点世界系接收格式10Hz shadow，亦保留原legacy H40统计。
- \`OFFICIAL_WITH_FEEDBACK_ELIGIBILITY\`：**离线**把原动作意图的目标/轨迹提交给已通过F2静止已知点控制器接口的静态适用域判定；任何时变段、接触段、动中抓持段都标 \`A1_DYNAMIC_UNQUALIFIED\`，不能把F2停点成功伪装成移动中运动成功。
- 输出按来源/source文档分类 \`WELL_FORMED\` / \`FRAME_VALID\` / \`OBS_IN_DOMAIN\` / \`DYNAMICS_PASS\` / \`GRIP_SAFE\` / \`NONTRIVIAL_PROGRESS\` / \`TASK_LEGAL\`，每个有独立分母。按source checkpoint报表，不能因为某样本reject就删掉。

## 4. O2：官方模型的新真实 Go2 forward（**需要独立GPU owner/lease**）

只在确认已有source P0/P1回执缺乏需要的新任务/观测条件时新发起。同一 **official post-trained** checkpoint，锁住source revision/norm stats/config、推理随机种子、精确robot/cam/历史状态、episode reset和推理GPU身份。
- 阳性来源对照：同checkpoint、source标准三视角/双臂/30Hz/head历史、训练分布相近的单段抓取动作；不引入teacher action作为student输出。
- 目标Go2：两真实相机+显式虚拟左腕和左臂；观测输入保持视觉时间/来源透明；对真实右臂选择官方8D后输出世界系TCP；摄像头与状态仍有domain gap，实验名称不得称'no adapter direct use'。
- 一次forward可生成50×16；全程保存模型输入hash（包含逐时标历史帧身份）、raw/decoded动作hash、source→world rigid calibration、gripper events、真实推理延迟；最多可建议8个非接触Go2 inference窗口+两个来源阳性对照，**建议上限10次前向**不是资源许可。按本模型已有0.293s历史median只作为参考，不是当前p95。
- 没有CPU审计pass与GPU资源owner许可则停\`WAIT_GPU_AUTHORIZED\`，不能在GitHub工具里把实际服务器任务说成已经执行。

## 5. O3：学生自己的 gripper ownership（单独的先决硬门）

不能从teacher持物阶段继承夹爪；8/16老师持物学生提议张开，按上一条已经消费的teacher命令算起存在显著非法命令跳变。合法入口优先 \`STUDENT_OWNED_RESET\`：episode从reset开始，控制器模式/机器人contact已核验，夹爪有真正receiver owner ACK；旧老师持物历史不自动转owner。仅非接触合法时才可经explicit transfer。必须保存 \`issued/ACK/consumed/actual/contact\` 五种分离字段。
- 先执行独立CPU state-machine/时效/伪ACK测试，后带独立新权限做不接触夹爪张闭与安全停机物理资格。任何没有真实student owner receipt/接触危险/持物开爪，标\`GRIPPER_BLOCKED\`。不替换原H40/G7门，绝不以actual width顶替previous_consumed。

## 6. O4：真实冻结WBC+官方后训练SLA (条件性原生)

F2已通过的是**慢速已知终点**，不是可沿运动轨迹自动追踪的动态闭环。正式学生任务前还需两个独立资格：
A. \`A1_DYNAMIC_NONCONTACT\`：已知无接触动态EEF路径（含外环介入）在真实机器人反馈下满足预注册的绝对轨迹误差/相位/速度；保持所有old H40/Safety原限与一个新servo自己的安全包络。
B. \`GRIPPER_OWNER_AND_CONTACT\`：独立证明student合法接管/由自己reset拥有夹爪，精确验证touch/grasp/release，移动中故障安全处理有资格。
- 新正式学生实验优先source相似的单臂接近/非接触、再合法接触grasp/lift/release，最后StackCube OOD、移动底座分开报告。不能直接拿V1.3 TRAIN16窗当held-out能力。
- 对照组至少\`TeacherSameBackend\`、\`OfficialPosttrainedSLA+FrozenOCS2+A1\`、\`OfficialPosttrainedSLA+FrozenOCS2+A0\`，严格匹配相机/初态/场景/owner；若资源许可，可另有\`SLA_base+same\`作为后训练权重贡献的正当对照，但必须使用其**自身**正确处理与归一化，不能默认base可直接给出符合实机协议的EEF。
- 同时报告\`model_loaded\`、\`forward\`、\`reference_admitted\`、\`native_installed\`、\`consumed\`、\`grip_owner\`、\`grasp\`、\`lift>=20mm\`、\`hold>=1s\`、\`place/release\`、\`task success\`、wall-time，拒绝和技术异常都不能从分母删除。
- 任何新physical/native/GPU要单独owner/lease/native_tick_cap/stop/safety/version SHA授权；本文件不继承PR#437过去租约，没有执行许可证，且B4旧NOT_RUN原状态保留。

## 7. 这次能诚实报告的结果

**当前已有证据支持：官方实机后训练SLA权重可直接作为推理权重、经过合法观测适配和任务空间映射输出Go2右臂运动意图。**  
**当前已有证据不支持：原生即插即用到Go2传感器或OCS2接口、动作直接可被安全安装、可在F2外环下实时移动抓持，或者已实现Go2抓放成功。**

只有O4真实闭环成功，才能回答“官方后训练版本不进行Go2额外后训练是否完成跨本体技能迁移”。若O4失败且低层能跟踪应按 \`OBS_OOD/GRIP_TASK_EVENT/EXEC_TIME/POSE_GEOMETRY\` 分类；若观测缺失或双臂单臂gap显著，才有因果证据转入Go2后训练或训练多源联合数据。对泛化提案应按已知source任务与目标StackCube分别计算。
