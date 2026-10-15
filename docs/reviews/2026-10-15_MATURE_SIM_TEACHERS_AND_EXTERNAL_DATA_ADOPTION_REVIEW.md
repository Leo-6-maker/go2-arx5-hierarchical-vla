# 成熟仿真 Teacher 与公开数据集：GR00T＋冻结 WBC 适配研究
日期：2026-10-15；文档类型：外部方法研究与建议，**不授予新的物理/GPU/PBS/Teacher 版本执行权限**。唯一现有 local-PC Codex owner 按最新 #11、已采用实验及真实 lease 继续，不热改在途 run。历史固定整机 pilot 0/4、整机成功训练数据0、FB13尚无已验收结果保持原账。当前/后续新 Teacher 已发布代码不等于已物理成功。本文件不修改 WBC、不重写任一 FAIL。

## 直接回答
**现有 Teacher 不是从 MimicGen、ManiSkill、Isaac Lab Mimic、RoboCasa 或其他成熟专家自动移植的实现。** 实际仓库证据：
- `scripts/pi0291_arx5_data/teacher.py` 的 PickTeacher 为本项目FSM；
- `scripts/pi0294/teacher.py` 的 PushTeacher 是本项目的几何+接触反馈FSM；
- `scripts/pi0294/task_native_worker.py` 实际创建上面两个Teacher，向固定 WBC 的 EEF+BASE 参考接口发命令；
- `scripts/pi0294/persistent_teacher.py` 和 `persistent_teacher_worker.py` 是2026-10-15才加入的自研持久接近补丁，不是外部框架port，不能因为CPU测试就认定任务专家已成立。
旧固定底座场景曾用IK预放TCP到物体上方；新整机从native home出发，致原pilot连续3条32秒停在approach且无物体接触；这说明源专家迁移的**起点分布和接近规划缺口**，不是“没有WBC权重”或“N1.7学不会”。事实来源 #11 comment6187275435、6187515183、6187967298、6189405130及固定pilot报告7a05ad279349e59e89e51920beec187ac08cdc88。

## 成熟外部解决方案：分清「无自有真机示范即能种子生成」与「有示范后扩增」

| 方案 | 原始原理和必要输入 | 本项目建议 | 限制 |
|---|---|---|---|
| **ManiSkill 3 scripted motion-planning experts** | 提供 PickCube-v1 和 PushCube-v1 的motionplanning演示/BC例子；以物体几何、IK、路径规划、抓取/放置事件构造专家，**不要求我们自己有真机示范** | **第1优先**：移植其任务分解、目标采样、规划—执行—重观察及失败逻辑到自己的EEF13 Teacher | 示例控制Panda、PhysX/MANI环境，不可直接执行到Go2/ARX5或把Panda关节动作当EEF13 |
| **MimicGen (NVlabs, CoRL2023)** | 需要少量源成功示范（可下载公开源）；将示范拆为物体中心子任务、变换至新物体位姿、插接并重执行；公开48k+数据 | **第2阶段**：等本项目有几条成功、带物体真值的seed后，建自定义Environment Interface并扩增 | 不能从0个本体可执行seed凭空产生成功；新仿真WBC需要写wrapper/片段目标映射，全部变换重播重新评分 |
| **Isaac Lab Mimic / SkillGen** | 物体相对源示范片段；SkillGen又用cuRobo在片段间求运动学/碰撞可行路径；官方展示Franka堆叠、要求已注释来源 | **设计参考**：借用可行段拼接、collision-aware桥接和源片段注释；若独立单项优势明显再评估真正迁移 | 不是零源示范魔法；整体导入Isaac Sim会重建已封存MuJoCo+ROS+WBC runtime，不作为近期默认 |
| **RoboCasa365** | 大规模任务/场景、按帧原子skill/stage/语言注释 | 任务层级/语义与可观测阶段标注的参考 | 厨房和其它本体动作/相机不直接映射我们的EEF13 |
| **UMI on Legs (Stanford)** | 真正用任务空间EEF轨迹桥接可移动四足WBC；任务演示来自手持UMI**真实人类示范** | 四足操作系统架构与EEF/task-frame接口的高相关基线 | **不能作为「没有真机/人类示范的现成仿真Teacher」**，不能把WBC训练和任务示范混淆 |

官方来源：
- https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/motionplanning.html
- https://github.com/mani-skill/ManiSkill/blob/main/examples/baselines/bc/examples.sh （PickCube、PushCube）
- https://mimicgen.github.io/docs/introduction/overview.html
- https://github.com/NVlabs/mimicgen/blob/main/docs/tutorials/getting_started.md
- https://github.com/NVlabs/mimicgen/blob/main/docs/tutorials/datagen_custom.md
- https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html
- https://github.com/robocasa/robocasa
- https://github.com/real-stanford/umi-on-legs

## 可以利用但不可以直接当Go2-ARX5示范的公开数据

| 数据 | 证实范围 | 推荐用途 |
|---|---|---|
| **ManiSkill PickCube/PushCube motionplanning** | 开源脚本规划轨迹/BC数据，任务与T1/T2最接近 | **算法/动作时序种子**；从原环境提取任务相对TCP/接触段，再在我们MuJoCo冻结WBC重执行 |
| **BridgeData V2** | 公开资料约5–6万条任务轨迹，其中公开原版另记9,731条scripted pick-place；多任务语言/真实照片/多场景 | 示范分段、物体中心路径、真实相机视角参考；只选择有可恢复EEF/相机/动作语义的子集 |
| **DROID** | 76k真实操纵轨迹、350小时、564场景、86任务 | 多视角、接触前后动作和语言先验；可做小样本视觉/阶段对照，不是GO2 BASE标签 |
| **Open X-Embodiment** | >100万轨迹、22种本体，RLDS统一组织但源动作定义不同 | 广义视觉/语言先验与数据格式对照；N1.7本来具多本体先验，不能不经对照就把异构action混成EEF13 |
| **Mobile ALOHA** | 完整身体移动+双臂、语言、相机、真实遥操作数据；TFDS范例train约276段 | 研究手+底盘协同和何时开始/结束移动，**不直接学其16D动作** |
| **RoboCasa365 / RoboTwin 2.0** | 大规模仿真任务、官方stage标注/脚本库/开放演示 | 找成熟任务定义、失败分类和可复用片段，而非复制对方关节动作到我们的WBC |

来源：
- https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html
- https://bridgedata-v2.github.io/ 及 https://rail-berkeley.github.io/bridgedata/ （release口径不同，引用条数时固定版本）
- https://droid-dataset.github.io/
- https://robotic-transformer-x.github.io/
- https://github.com/tensorflow/datasets/blob/master/docs/catalog/aloha_mobile.md
- https://robocasa.ai/
- https://robotwin-platform.github.io/doc/usage/collect-data.html

使用前逐版本检查许可/归属/相机/机器人/动作表示；**有公开真实机器人数据**不等于我们自己有同本体/同相机的标注。原数据必须保留自己的provenance与机器人标签，不允许直接重命名成FULL_GO2_ARX5_NATIVE_WBC数据。

## 无自有真机数据时，真正可实施的 Teacher 生产管道（建议，并非执行许可）

**0. 冻结未改变的执行边界。** 保留原Go2自由根、原腿策略、原WBC权重、MPC/reference parser、机械限位和物体任务成功谓词。另行版本化**我们自己的专家任务规划器**，且需要授权与独立物理收据。不要把0/4历史质量失败改判。

**1. 从ManiSkill借「任务与路径规划」，而不是试图整仓移植仿真引擎。**
- 在独立研究环境审阅其PickCube和PushCube脚本、demonstration HDF5样本，记录object pose、EEF目标、gripper动作、物理事件、task stages、参考帧和时间语义。
- 借用对象中心 grasp/approach/retreat/placement/推送段、可达性检查、collision-aware waypoint connection和失败恢复策略。
- 如果采样外部片段有物体参考源姿态 `T_WO^src`、源EEF `T_WE^src`，新物体姿态 `T_WO^dst`，目标EEF世界姿态由
  `T_WE^dst = T_WO^dst * inverse(T_WO^src) * T_WE^src`
  得到。**只适用于与该物体刚性相对定义的相应片段**，移动目标、释放、接触滑动、站位和其它锚点应重新规划，不可生硬整条变换。
- 插接和轨迹可行性在我们平台里重新验收（绝对时间、采样、碰撞、camera/FOV、gripper几何），不可声称Panda/外部demo的成功自动转移。

**2. 真正考虑完整Go2的接近。** 离散采样base站位和朝向，先用原ARX5运动学、任务空间和碰撞几何筛掉明显不可达，再以冻结WBC原生BASE软参考执行站位；保持可观测的actual base/tcp反馈。固定独立全身Teacher先解决站位/预接近，再执行靠近物体的6D EEF预抓/预推。
- W0近场先做**单技能真实到达预接触**，然后真实接触/短推动、握持/抬升；W1移动只在W0闭环成立后作为第二层证据。不是把base锁住，也不是改低层WBC。
- 给技能节点设置**持久目标、真实几何触发、允许真实状态纠偏、有界无进展退出**。不能按时间强迫进入close/contact；不允许反事实qpos瞬移/固定joint偏置。
- 使用H40、prefix8=0.4s和现有20Hz记录的显式标签时间。不同步/参考未被实际提交的动作不能用于训练。

**3. 由真实MuJoCo执行而非离线轨迹本身决定数据是否合格。**
- Teacher读仿真物体位置和接触状态（特权），提出EEF13；
- 冻结WBC+夹爪执行，录**实际机身/腕双RGB、actual EEF/夹爪/BASE、当时发出并安装的Teacher高层参考**；
- 评分真实物体抓举/推送与原高频安全、任务完整、scene-group分集。成功轨迹→成功BC bank；失败→独立failure bank，不改分母。
- 不用ideal EEF未来、外部Panda关节动作、WBC关节输出或actual未来位移偷偷覆盖教师目标标签。学到的策略只接收RGB/语言/可测状态，不接收object oracle/FSM阶段。

**4. 首批本体成功seed成立后，才采用MimicGen式扩增。** 对每个物体中心子任务保存origin物体姿态、相对EEF/夹爪/base意图、段边界和真实阶段事件；在原TRAIN可用的场景组内改变源物体位姿、任务目标和base站位，重新检查可达性与完整执行，重新过真实成功门。不是仅图片变换或轨迹插值。1 worker成功资格→2 worker隔离/数字等价→4 worker吞吐，后者不继承旧固定臂2.86×。

**5. 与GR00T真正匹配。**
- 物理训练目标是本体EEF13 (EEF10＋BASE vx/vy/heading preference)，通过官方N1.7 custom embodiment modality 规则声明视频/state/action/language及动作变换。注意视频/状态`delta_indices=[0]`，H40未来动作索引与stats完全匹配；新增BASE缺失项必须有真实label，旧EEF10不可零填充当监督。
- 先在极小成功训练样本上做open-loop overfit/codec round-trip，再做**完整32s模型新观测闭环**；离线loss下降不能代表物体任务成功。
- 优先保留目前已训练固定底座M2048作为模型warmstart/比较组，但新输入/动作模态与数据域意味着新分支，不是假称exact optimizer resume。
- 源：https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_config.md
- 源：https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md
- 源：https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md

## 优先顺序与门（全部为拟建议，不构成旧GO）
1. **STOP/GO A：无需物理的外部框架可用性评估**：明确对照ManiSkill原Pick/Push源示范、坐标/动作/几何、公开许可和最小适配成本；若只借概念，无须安装复杂GPU运行栈。复核仓库中的新Teacher源和最新终态，保护同一owner/lease。
2. **STOP/GO B：我们自己的新Teacher单场景物理检查（新scope/预算才允许）**：先真实W0接近进度，再真实接触、短推/抓握，再完整一次T1和T2；每个里程碑保留真实安全与旧任务评分，零成功不启动四worker。
3. **STOP/GO C：本体成功种子扩增**：先保证两种技能均有完整成功执行；若少于一方，保持相应数据缺口，不用其它机器人label替代。MimicGen式扩增要针对可迁移片段和实际接触重新评分。
4. **STOP/GO D：GR00T首版训练**：有冻结全身、完整EEF13合格bank才正式训练FB13，并立刻排真实模型开发评价；CoreLab在其间独立扩下版数据。无本体成功数据不以更多optimizer步数“消除数据问题”。

## 需要正面披露的三项限制
- 外部教程的Panda/Franka/移动ALOHA轨迹不是可直接发送到当前Go2 native WBC的命令；硬件、frame、观测和夹爪差异必须通过自研适配与真实MuJoCo回放解决。
- MimicGen/SkillGen需要源成功示范及任务环境适配，因此**更适合作为第二阶段扩增器**，而非现在的第一条成功demo来源。
- 成功只在仿真证明，不可假称无需真实数据就能零差距部署真机；如以后要求真机泛化，需额外的数据/视觉/动力学随机化及真实验证。

本研究不启动/撤销已采用的新Teacher执行系列，不取消在途任务。该文档仅给当前Codex一个成熟方案参考；若想在其上真正替换Teacher或增加物理实验，必须检查先前授权或取得新的具体范围。Git仅保存原创研究文字和官方链接，不存第三方权重、原始机器人数据或私有WBC。
