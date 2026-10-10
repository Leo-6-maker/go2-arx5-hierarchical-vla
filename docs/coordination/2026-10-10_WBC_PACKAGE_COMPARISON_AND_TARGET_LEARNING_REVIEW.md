# WBC版本、接口变化与目标域GR00T学习路线审查

**性质：REVIEW_AND_PROPOSED_ROADMAP_ONLY，非新GO。** 本文件不启动任务、不调整PI0289安全/质量门、不解除其下游停止、不新增训练或物理预算。用户请求比较新旧上传WBC及思考教师数据和GR00T后续实验。本轮只做归档静态阅读、哈希核验、公开运行证据刷新与方法建议；没有SSH、编译、仿真、模型forward或权重加载。

## 1. 实际可见版本与审核依据

当前可见两份归档：
- `WBCv1.zip`：159225884 bytes，SHA256 `3c59a6d5f8f924df0b838ad2d501d3a853274d741db191a9de453f58212063ee`。旧IsaacGym/RoboDuet双策略路线。
- `wbc_mujoco_minimal.tar.gz`：215194129 bytes，SHA256 `6dd13098de5a58327fb0a0a5b3ff749e0a5534510375b92fba3f8fb50a322d52`。新部署路线OCS2+MuJoCo+腿部RL。此哈希与此前已复现的交付包相同，不能据本次“最新版”称呼虚构一次新的MuJoCo源码变更。

本轮实际计算两归档哈希、静态提取新版140个选定文本成员并追踪18个关键文件；没有执行上传库/二进制。完整交付11338项校验属于此前owner运行证据，本轮不冒称重新逐文件校验全部payload。新tar有11948个条目、11339个普通文件，与11338个校验payload不是相同口径。

运行证据固定SHA `c62335b9319df79b4a3711867fcf0f64c14ecf90`；最新科学收口Issue #11 #6093151230，2026-10-10 11:08:34 SGT。主报告 `reports/PI0289/execution/REPORT.md`。必须区分交付归档、项目启动/参考overlay、实际控制资格三种身份。

## 2. 控制结构：不是同一WBC换权重

| 层次 | 旧WBCv1 | 新MuJoCo交付默认native |
|---|---|---|
| 物理/运行 | IsaacGym、Python/PPO训练和play | MuJoCo3.2.7、ROS2 Jazzy、C++/Python运行快照 |
| 臂与身体协调 | Stage2 arm actor，轨迹模式常为6臂+9身体/步态输出；具体obs/history按checkpoint配置 | OCS2在线优化BASE与arm参考，再经WbcBridgeCore派发 |
| 腿部 | Stage1 dog PPO | 已交付N1_s29_175632腿RL，12腿动作 |
| 参考管理 | Gamma(s)、TimeLaw、TrajectoryBatch、实际进度/空间preview | 时间化EEF/可选BASE pose，ROS reference manager、MPC TargetTrajectories |
| 高频执行 | 旧混合臂/腿控制 | arm PD + RL腿；夹爪独立通道待完成 |
| 模型身份 | Stage1+Stage2配套权重、history、资产 | MPC配置/模型/solver/bridge+腿权重+arm PD+启动与任务overlay |

旧源码依据：`go1_gym/envs/config/wbc.py`的end_to_end/IK模式；`go1_gym/envs/roboduet/wbc_env.py::plan()`（约426–527行）及preview（约2400–2442行）；`modules/trajectory.py`；`scripts/play_by_key_stage2_traj.py`。201D和15D是历史特定配置，不是外部统一接口或新MPC维度。

新版`README.md`明确该包是运行快照，包含模型、策略、专用库和源码，不含GR00T/任务示范训练、真机SDK或完整操作环境。默认profile=native、task_floating.info、arx5_ac1_floating.urdf，不能自动切换到另带的policy-aware配置后继续声称相同控制器。

### MPC的实际预测模型

`task_floating.info:17–18,26–27`：state12为base x/y/z/yaw/pitch/roll + arm q1..q6；input12为base运动分量+arm速度。`FullyActuatedFloatingArmManipulatorDynamics.cpp:45–49`的flow map直接返回input。因此当前native使用的是简化全驱动浮动臂运动学积分模型，不是包含四足接触力、负载与全身动力学的完整MPC。实际物理仍由MuJoCo、arm PD和腿RL实现。

`task_floating.info:132–139`的MPC时域1s、期望MPC100Hz/MRT400Hz是配置，不等于实测频率；reference manager的2s是另一层窗口。包内policy-aware模型属于另外的配置/假设，不能把其存在当作已经启用或验证。

`MobileManipulatorInterface.cpp`将多项关节、BASE和自碰撞限制放入soft-constraint penalties；不能把“有MPC约束”当成实际物理不可能违约。环境碰撞模块在默认profile未见启用，不代表桌面/任务障碍已被优化器感知。

## 3. 新的高层接口契约

### 3.1 输入输出维度不能混用

实际入口 `/go2_x5/ee/trajectory_json`，类型std_msgs/String。JSON允许旧顶层waypoints或嵌套ee对象，以及可选base对象。每个waypoint含time、position[3]、quaternion_xyzw[4]。eef-only输出为7D pose，eef+base为14D pose；`ee_trajectory_node.py:239–246`构造MpcTargetTrajectories，input_trajectory为12维零向量。

所以：GR00T有效10D（xyz+rot6d+gripper）→ 解码SE3 → 7D EEF pose +独立gripper schedule；14D pose不是14个关节，内部12D state/input也不是VLA动作格式。当前这个JSON路径提供BASE位姿参考，不是任意BASE速度字段直接下传的接口。

位置单位米、时间秒、姿态XYZW；MuJoCo freejoint本身通常WXYZ，转换要明确。当前TCP是gripper_center/x5_ee，必须按该交付模型绑定，不仅依据同名旧URDF。

### 3.2 BASE软参考确实已实现，但只对部分分量生效

`task_floating.info:293–307`，baseTracking.activate=true，权重x/y/yaw=5/5/2，z/pitch/roll=0。`FloatingBaseTrackingCost.h:12–15,38–47,69–83`：7D EEF-only时BASE reference cost为零；14D时增加0.5*e^T W e。

这直接支持“高层给操作站位/朝向偏好、WBC允许为其他目标偏离”的结构。它不是BASE六轴硬跟踪，也不能推断高层命令身体高度/俯仰已有效。EEF自身同样是软cost（position10/orientation5，task_floating.info:309–319），不能描述为绝对硬优先。

特别注意：BASE-only路径保持EEF相对BASE的初始关系，让手随BASE走；它不是让手固定世界位置后移动身体。需要手世界系固定又移动BASE时，应显式给EE和BASE两条参考。依据`ee_trajectory.py`的TaskTrajectory绑定/采样与`ee_trajectory_node.py:249–257`。

`WbcBridgeCore.cpp:90–111`显示MPC BASE位置是机械臂安装平面，来自IMU位置加旋转后的mount_offset，并非直接是IMU或trunk原点。人体躯干/机器人base轨迹必须先转换坐标，不能原值直写。

### 3.3 时钟/参考锚定变化

frames支持world、world_offset、ee_start（BASE还接受base_start别名），有start_delay、lead_in、align_orientation。完整JSON校验之后，参考在下一次MPC观测绑定实测起始姿态；不是默认绑定GR00T拍摄图像时的TCP。

建议异步GR00T适配时：根据观测时刻的真实TCP解码相对H40，一次转换成固定world参考，再处理声明的时钟映射和过期动作；不要让接收端较晚的TCP悄悄改变动作锚点，也不能通过整体平移掩盖推理延迟。

teacher/VLA新chunk可0.4s一次；manager10Hz只是再发布，不是10Hz新决策。native2s窗口的0.05s闭区间产生41点，且可能合并额外knots；GR00T H40为0..1.95s。writer和bridge须明确端点与保持补齐，不机械认作同shape。

原生parser有有限性/四元数/时序检查，但没有完整episode/ref-ID/ACK/TTL/cancel协议。参考到期停止再发布不代表OCS2旧目标已清空；每个新JSON重新绑定，而同一chunk重发不应重锚。PI0289的session/expiry guard是项目新增overlay，不是原包自带；真实负例已回到HOLD，但时钟准入仍失败。

### 3.4 可读取反馈与夹爪

当前已有odom、arm/state、ee/state、MPC observation/policy、arm command request/target、base command及arm mode request/state。可记录完整requested→accepted/installed→dispatch→achieved链，但消息可读不代表同步有效、延迟可忽略或适合直接新增VLA输入。

原始native MJCF只有12腿+6臂关节，未提供完整独立gripper驱动与named body/wrist相机。PI0289已声明SIM_ONLY parallel mimic pair选择，但工作速度/力、pad几何/材料/摩擦和硬件修订未认证。场景/相机/夹爪工程不能被当作已采得真实目标示范。

## 4. 最新实际进度及必须先处理的两项问题

PI0289 11:08:34 SGT终局：
- V3安全启动真实WBC hold10.000178504s，原首步安全生效，无关节越界、native warning或禁止接触；S1 PASS。
- 2001 native反馈，末2sXYZ RMS0.011672954m（门0.015，PASS），SO3 RMS0.065199717rad（门0.050，FAIL）。其余五参考未运行，不能报0/6完成或6/6通过。
- native EEF和MuJoCo实际site晚期差异≤8.24微米/2.51e-5rad，不支持用大静态TCP偏移解释全部0.0652rad误差。
- joint3 endpoint actual q=.589266、qd=-.000471；收到target q=.625330、qd=.194336；actuator/bias5.44713/5.44435Nm，tau_ff=0。时间戳不完全相同，只支持保持/命令实现问题的假设，不是唯一根因证明。
- V4接收时刻拟合残差0.047803157s>0.025s；估计发布origin后的更小残差不能替代未保存的真实源时间。
- 目标教师/数据/GR00T训练0；selected/deployment NONE。

源码线索：`WbcBridgeCore.cpp:237–270,350–366`分别对位置预览与速度分量采样，native arm velocity scale默认0.4；`rl_sim_mujoco.cpp:220–229`实际tau=tau_ff+kp*(qcmd-q)+kd*(dqcmd-dq)，原base.yaml中joint2/3为kp70/kd15。静止目标仍有非零desired dq和位置误差，可能是简化运动学MPC、预览及无独立重力扭矩的PD实现共同造成。应由WBC负责人核查现有受支持的保持/前馈约定，不能由VLA学习固定偏置替代低层修复。

下一可审计动作应是只读对齐同一时间的qcmd/dqcmd/tau/solver state和源端sim/ROS时间戳，然后单独注册必要修复/复验。不能修改gain、preview、cost、阈值来直接追分，更不能因此将原失败重写。

## 5. 如何生成专家数据：模型是工具，教师是任务层

MPC不是自动的抓放专家；包内trajectory_library的字母/圆/曲线是参考测试，不是物体操作示范。推荐链路：

任务与物体几何 → object-relative多阶段teacher → 正确EEF+gripper（必要时BASE偏好） → frozen参考管理/MPC/PD/腿RL → MuJoCo真实物体交互与同步RGB → 物理/标签/时间准入 → GR00T监督。

教师可用特权物体状态、运动学/抓取规划或优化工具决定接近、抓握、抬升、搬运、释放、撤离。Panda提供任务阶段经验，不复制Panda关节、工具朝向、世界坐标或图像。源相对抓取关系必须按ARX5工具几何重新计算；target状态变化后重规划，不按专家时间索引贴错标签。

如果以GR00T/其他大模型生成候选计划，只能视为proposal：必须经过几何/可达/物理验证或teacher纠正再成为监督。不能用未合格的Student自输出循环当expert。MPC优化出的joint/base动作和实际achieved TCP也不能自动当高层正确标签。

每episode同时保存PolicyObs（真实双RGB/实测TCP/夹爪/语言）、ActionTarget（明确来源的desired EEF/gripper）、Execution（accepted/installed/dispatch/achieved/BASE/时钟）、Privileged（object/contact/phase/score）。未来planned chunk与actually committed命令分开；所有label与obs同一执行时钟，H40不跨reset/abort/失效reference。物体真值/教师phase/success不进入Student。

按执行成功准入目标数据属于target-domain adaptation，独立source任务库仍保留；不能因此称零目标数据迁移。缺BASE/feedback标签用mask，不把Panda没有移动底盘解释成所有BASE通道应为零。

## 6. 建议路线（没有新增执行授权）

A. 完成真实保持偏差与时间同步修复/验收，再做剩余运动、热替换。复用已成立安全启动，不继续盲目换站姿或重跑旧编译。
B. 在冻结核心上补独立夹爪、接触、目标双相机和记录；任务overlay重新验收后形成TASK_EXECUTOR_FREEZE。先SIM_ONLY可行性与硬件一致资格分开。
C. 首个任务仍用宽目标抓放。依既定gate先小规模teacher开发、固定pilot，然后初批TRAIN64/DEV16；这是可控试验规模，不保证收敛。保留失败、抓握/抬升/释放/撤离漏斗和held-out布局，不能用总窗口数代表独立数据量。未经准入不得开始当前下游采集。
D. 目标数据就绪后首个baseline从真正pretrained GR00T出发，沿已核验native flow-matching/EEF10、target TRAIN-only统计、原生processor/collator；先保持VLM冻结和明确动作侧训练范围，不叠加新BASE head/世界模型/历史。独立短闭环优先于H200连续长训；模型身份包含weights+processor+stats+decode+clock+executor。
E. 做Source-only、Target-only、Source→Target的目标域对照，目标数据预算/冻结WBC/布局/评价一致。新模型使用旧Panda失败链不是强制起点；预训练收益和目标适配收益应分开。
F. 首任务有可信能力后，两任务共享使用同一个checkpoint、同场景不同语言及每任务分别的成功率/旧任务保留。先抓放+堆叠，再任务域扩展；不要无限在固定8个条件上续训。
G. BASE/反馈后续增量：先测试正确规划器给出的BASE软偏好是否优于EEF-only；有价值再学习BASE head。低层反馈先按真实可部署信号标准化并加入历史对照。不能同时改变reference contract、控制权、输入和损失后把收益归给某个模块。

**应向师兄核对：** 实际native/备选policy-aware profile；arm保持及位置preview/速度前馈/重力力矩职责；发布端时间戳与reset/expiry接口；BASE arm-mount参考坐标及14D意图；夹爪工作规格、pad和双相机安装。核对不是要求重新训练整个WBC。

## 7. 公共外部参考（仅方法背景）

- OCS2官方Optimal Control Modules：ReferenceManager、cost与soft constraint概念 https://leggedrobotics.github.io/ocs2/optimal_control_modules.html
- NVIDIA官方new embodiment微调入口 https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md
- NVIDIA官方数据准备 https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md

官方示例的joint动作/16步/2000更新不是本项目应直接照搬的配方。当前项目EEFH40需要自己的modality配置、统计和真实闭环评价；文献与框架功能不替代本次包内代码或物理收据。

最终判断：新版为统一EEF与可选BASE软参考提供了更明确的接口，但当前native模型不是全身物理专家，已运行证据仍未通过姿态与时钟门。先得到合格执行器和目标域教师，再学习GR00T；结构性解耦可保留，未经适配跨本体共享必须另测。本审查没有新canonical GO、没有启动Codex/远端作业或更改已封存结果。
