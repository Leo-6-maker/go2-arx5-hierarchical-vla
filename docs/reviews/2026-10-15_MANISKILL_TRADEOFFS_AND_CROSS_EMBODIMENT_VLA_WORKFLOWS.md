# ManiSkill 接入取舍与成熟跨本体 VLA 工作流

日期：2026-10-15。类型：RESEARCH_REVIEW_AND_RECOMMENDATIONS_ONLY。

本文件回应用户关于 ManiSkill Teacher 优缺点、替代方案和不同本体 VLA 微调成熟度的问题。不采用新实验、不增加资源、不替换当前已采用规划组件方案，不启动仿真/模型/PBS，不改 WBC、已有数据、失败或任务评分。引用文献见同目录 `2026-10-15_CROSS_EMBODIMENT_VLA_REFERENCES.bib`；外部 main/latest 文档是此次研究所读版本，不替代项目已冻结软件版本。

## 1. 核心结论与两项纠正

**不同本体的 VLA 监督微调、数据转换、模态注册和推理服务已有成熟且可运行的开源流程；但给任意新机器人和任意任务自动构造成功专家，不是这些微调框架自动提供的能力。**

当前项目同时承担了：目标完整仿真接入、第一批成功示范生产、EEF13/BASE新接口、已有VLA的目标域适配。不能把完成这四项的难度与在LIBERO已有数据上运行官方finetune脚本等同。

需要修正以前过强的表述：
1. ManiSkill提供的是现成任务环境、规划/记录工具和特定机器人上的专家示例，不是可直接接入任意WBC的通用Teacher。MPlib/OMPL的规划模块比Panda示例更通用，但仍要求准确模型和执行契约。
2. MimicGen需要**源示范**，不要求源一定来自我们自己的真机或目标机器人。它明确研究不同robot arms/初态/物体上的生成。可从公开源演示开始；但新的目标框架wrapper、TCP/action映射、子任务信号和目标执行仍须成立。建议先获得目标本体少量seed是工程降风险方案，不是MimicGen算法的严格先决条件。

## 2. 本次实际项目与上游源码证据

私有项目读取基线：`3ac0bd5ec541433bc5632c8533fd6e2f1e2176c5`；最新读取#11评论6190273335和6190497455。V2终局仍没有安全预接触/物体接触/成功整机示范。已有有界ManiSkill规划组件提案和固定源审计；本研究不改它的采用状态，执行owner应读取之后的真实授权与收据。

项目审计文件：`docs/reviews/2026-10-15_MANISKILL_PINNED_SOURCE_TO_EEF13_ADAPTER_AUDIT.md`。本次还直接读取了上游固定 `mani-skill/ManiSkill@62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`：
- `mani_skill/examples/motionplanning/panda/solutions/pick_cube.py`
- `mani_skill/examples/motionplanning/two_finger_gripper/motionplanner.py`

证实：PickCube solver以OBB构造抓取、接近、闭爪和移动到goal，没有我们要求的完整release/withdraw/hold；follow_path实际发送qpos/可选qvel及gripper到env.step。不能把这些joint动作改名EEF13，也不能把上游成功标志当本项目完整任务成功。

已有项目审计进一步记录PushCube固定world-X偏移、默认碰撞场景覆盖不足，以及ARX5 MPC URDF缺四足腿几何。完整碰撞判断必须包含当前native MuJoCo的身体、腿、桌面、夹爪和物体；不改控制器原参数。

## 3. “接入ManiSkill”有三种完全不同的含义

| 路线 | 复用内容 | 收益 | 必须承担的代价 | 本项目建议 |
|---|---|---|---|---|
| A. 只复用规划/技能组件 | OBB、抓取候选、IK/RRTConnect/螺旋运动、时间参数化、任务片段 | 不重写规划器；保留现有真实目标环境与WBC | ARX5/TCP/碰撞、BASE站位、EEF参考执行适配 | 当前默认优先 |
| B. 在ManiSkill产生源示范再retarget | 外部Panda轨迹与物体相对操作片段 | 可获得明确技能种子和阶段参考 | 换机器人/夹爪/相机/控制模式；必须在目标MuJoCo重新执行采集 | 第二来源，不能直接改数据标签 |
| C. 把Go2+ARX5和WBC整套移入ManiSkill | SAPIEN/PhysX、GPU并行、场景/录制/训练工具 | 若移植成功可扩大仿真吞吐 | 接触/惯量/驱动/腿policy/ROS时钟/同步/资产/验证重建 | 不是首批seed的默认前置 |

**A不会继承C的GPU仿真吞吐。** 只导入MPlib后，MuJoCo和原OCS2执行仍按当前CPU调度；旧固定底座4worker约2.86倍也不能保证新fullbody/规划吞吐。

## 4. ManiSkill能给我们的真实优势

1. 从当前机器人q出发求IK、规划可达路径，代替无界修补“抬高/转向/接近”游标逻辑。
2. 对象几何和抓取候选相分离，接近、接触、搬运、放下、撤离可以复用可测试原语。
3. MPlib提供IK Failed、RRT Failed等错误区分和带时间/速度的轨迹，方便定位规划失败、接口失败和任务失败。
4. 已有演示、HDF5、视觉录制、模仿学习例子可以当阳性工程对照，不必自研所有外围工具。
5. 公开源码与许可允许固定版本和归属；任务数据生成可用仿真真值，不需要自有真机示范。

官方motionplanning文档同时明确：示例写给Panda，需要人工工程化编写/适配，对动态任务有局限。它不是保证自动成功的专家模型。

## 5. 最重要的不足：规划器路径与冻结WBC执行不是同一件事

### 5.1 不是传一个target pose就自动有全身执行

MPlib接收pose但通常在关节空间搜索，结果是q(t)、dq(t)。原Panda wrapper由关节伺服执行。我们必须沿原高层EEF+BASE软参考接口执行，不能转而把q(t)注入WBC底层。

**推论：即使使用本体FK将q(t)映射成EEF轨迹，也可能丢失原规划的姿态冗余/关节分支信息。** 原生WBC可能选择不同关节或身体运动实现同一EEF参考。所以“规划器检查过路径”只证明该规划解的属性，不证明实际WBC实现路径也无碰撞。保留实际反馈重规划、高频安全和真实任务评分即可，不要求重新研究/调WBC保持精度。

### 5.2 抓放与接触不由自由空间规划保证

RRT的无碰撞路径不能单独证明夹爪能可靠握住物体、推送摩擦合适、释放后稳定。计划本身应明确允许的finger-object接触阶段、携带物体的占用、放置支撑与撤离；不能全局忽略机器人碰撞。

上游PickCube终点和我们完整T1不同；PushCube中的world-X常量与我们的Y方向/移动场景不同；工具长度和归一化开度不同。这些需要任务侧适配，不是WBC增益修复。

### 5.3 工具不能替代实际目标域的数据

Panda RGB+Panda actions可以做先验/源技能，但不能贴Go2标签；把Panda joints截取13维、把缺失BASE写零、把外部frame误当我们WBC mount都是错误。目标训练数据应来自目标环境实际执行后的图像、可测状态和真实提交的高层动作。

若生成源示范后做重定位，刚性对象相对片段可用
`T_WE_dst = T_WO_dst * inverse(T_WO_src) * T_WE_src * C_tool`
其中C_tool只有在源/目标semantic TCP关系被明确标定时才定义。推送滑动、目标容器、BASE站位和释放要按相应锚点处理，不能整条轨迹套单一物体刚性变换。公式是几何候选，不是动力学成功证书。

### 5.4 搜索预算和科学分母应分开

多次IK初值/抓取候选/有限RRT搜索是规划算法内部合理操作，不等于重新运行实验挑成功。应固定内部候选/时间/种子预算、保留失败与总成本，不能因任何内部失败就把成熟算法人为退化为永不恢复的脚本。

另一方面，上游only-count-success不能成为最终成功率分母；成功BC bank可以只收成功，所有采集尝试及失败仍保留。确认集从未见任务开始按完整episode评分，不能调过该组scene后再称held-out。

## 6. 成熟VLA项目实际怎么接新本体

### GR00T
官方教程顺序是已有目标机器人视频/state/action数据→GR00T LeRobot格式→NEW_EMBODIMENT模态/时间/动作表示→统计→训练→open-loop→实际机器人闭环。示例cube_to_bowl_5的5条用于流程/拟合验证，不是5条数据保证任务泛化。其API需要用户提供新数据和控制接口，不从URDF自动生成成功Teacher。[R5,R6]

### OpenVLA/OpenVLA-OFT
OpenVLA用目标Franka数据做新setup适配；原模型在某些预训练机器人上的out-of-box表现不等于任意机器人zero-shot。OFT在LIBERO使用benchmark示范，在ALOHA使用目标任务数据，再改chunking/连续动作/并行解码；这些解决模型学习和推理，并不生成我们缺失的整机seed。[R7,R8]

### OpenPI
官方依次示范LIBERO数据转换为LeRobot、Inputs/Outputs、data/train配置、归一化统计、训练、policy server和仿真评估。ALOHA说明零样本在新站点仍属实验性质，推荐目标机器人数据。开源的LIBERO/ALOHA/UR5流程值得复用，不表示pi0/0.5掌握当前ARX5动作语义。[R9]

### Octo
灵活tokenizer/动作头允许新camera/proprio/action接口，论文在目标任务约100示范上做新输入/新动作空间finetune。此数是其经验条件，不是我们的通用样本复杂度门。[R10]

### LeRobot
把标定、teleoperate、record、replay、train/eval连接起来；机器人驱动负责get_observation/send_action，数据层负责一致格式。对我们可复用schema与record/replay思想，不建议无必要整体重写已合格的native collector。[R11]

### NVIDIA Isaac Lab Arena（特别值得阅读的完整身体示例）
release/0.2.1的G1 static apple流程提供200条预录仿真示范，用户可不自行teleop，转换LeRobot后微调N1.7并闭环。记录时用AGILE+PinkIK，保存的processed_actions是**43维关节目标**；评价使用joint embodiment，而不是学习PinkIK输入的EEF pose。文档还明确不混用不同WBC数据/评价域。这是完整流程成熟的好例子，但与我们坚持的EEF13 task-space策略不同，不能直接复制其标签或认定覆盖Go2四足执行器。[R12]

## 7. 新本体适配真正需要统一的合同

至少记录以下字段并用于训练与推理同一实现：
- 实际camera名、图像shape和曝光时间；state来自测量，不来自目标值。
- 动作物理单位/正负方向、TCP/世界/BASE frame、absolute或relative、四元数/rot6d约定。
- gripper为机械行程、逻辑开合或归一化量；标签和执行器不做第二遍缩放。
- H40与20Hz意味着2秒展望；其他例子的H40/50Hz是0.8秒，不能只比较数组长度。
- 预测H与执行前缀k分开；记录实际提交动作，不把未消费的规划预览全部当真值。
- 训练统计仅用TRAIN，保存输入/输出变换和每维loss mask；缺BASE监督不等于零BASE目标。
- 正式episode split按场景组和task，派生样本/窗口不泄漏到DEV。
- action head、normalizer、dataset与机器人接口必须一起版本化；换动作语义不假称相同训练链。

所有机器人采用同一种finetune冻结方式也不是事实。OpenVLA原论文的Franka实验发现只训最后层或冻结vision表现差，而LoRA更好；GR00T项目有自己的默认；Arena特定配置训练vision/projector/diffusion并冻结LLM。当前先保留已通过的N1.7基线，等真实数据和闭环对照证明视觉/语言问题，再做单变量适配实验，不立即改所有模块。[R7,R12]

## 8. 替代方案与优先级

### A. 现有环境内的仿真人工/交互式示范——最直接的备选

“没有自己的真机数据”不等于“不能由人提供少量仿真种子”。ManiSkill有鼠标拖拽+按键触发规划的采集；robosuite有MuJoCo示范/输入设备。可在我们既有环境实现类似交互层，仍只发送EEF/夹爪/BASE意图，经同一冻结WBC真实执行。[R13,R14]

这条路线需要人参与，不能宣称自动Teacher；但它可先回答同接口下任务能否被示范，并为后续MimicGen提供种子。不要通过拖动实际qpos/物体或绕过WBC取得假演示。若自动规划受阻，少量人工辅助种子可能比继续构造完整自动专家更直接；此为建议而非已实现功能。

### B. MimicGen/SkillGen：有源片段后扩增

MimicGen按物体相对子任务变换，在目标环境重新执行。新的环境接口要提供实际EEF、action-to-target/target-to-action、gripper、物体pose、subtask信号；可使用公开源demo和跨臂迁移，不限定来自本体真人示范。[R2,R3]

SkillGen把自由空间连接交给motion planner，把接触技能保留成可变换片段，并处理持物/接触阶段碰撞。这个“规划自由空间、模仿接触技能”的拆分比不断加一个复杂FSM更值得借鉴；但仍有源数据与环境适配成本。[R4]

### C. RoboTwin 2.0：参考可扩展任务接口

论文给出50个双臂任务、5种本体，使用带操纵注释的对象资产、任务程序生成及simulation-in-the-loop修复。说明已有跨本体数据生成工作，但机器人/grasp/solver适配已由其框架提供，不是LLM直接产生普适正确动作。借任务和对象API、域随机化、失败分类；整体换引擎/双臂任务非近期默认。[R15]

### D. Isaac Lab Arena / Mimic：完整官方样板，不是默认迁移目标

可作为一个独立known-good对照验证“示范→模型→闭环”工具链，借输入输出单一配置和训练/仿真环境隔离。更换我们整个MuJoCo/WBC到Isaac/PhysX带来新的执行器与物理验证成本；不把换引擎作为第一份目标成功seed的前置。[R12]

### E. UMI on Legs：层次接口最相关，但数据来源不同

该工作用真实手持UMI人类示范训练操作策略、仿真训练tracking WBC，通过任务frame EEF轨迹接合，并展示部分跨本体迁移。它没有声称从零仿真任务示范自动生成高层专家；其训练WBC不属于我们的工作范围。可借相机/EEF/时间接口与任务跟踪分离，不把公开架构成功当本项目资格。[R16]

### F. 特权状态RL Teacher：后备研究，不是眼前的默认

可在冻结WBC之外训练小型任务级状态策略，输出同EEF13并蒸馏到RGB VLA；这并非重训WBC。但需要reward、探索、接触成功和仿真吞吐，低层可执行性没有自动消失。当前无成功seed时它未必比规划或仿真交互更省成本，暂不挤占主线。

## 9. 我建议借鉴的实际工作方式

1. 继续当前有界规划组件研究，只复用ManiSkill/MPlib任务与规划层，不整体换仿真器。先读现有owner终局，不把本研究当新GO。
2. 明确一条完整可行行为来源：自动规划优先；如果不能在合理开发边界内给出成功种子，使用拟议的目标仿真交互示范备选，而不是重复FSM阈值补丁。
3. 先取得某一种完整任务的真实成功数据就可准备该技能最小模仿验证；另一技能继续采集，最终共享双任务目标不变。不要规定“两个技能都完美才允许任何学习”。5–10条只作管线/过拟合测试，不宣称泛化。
4. 用成熟示范扩增形成目标环境数据，按语言同场景配对；保留Teacher特权状态但不输入学生，真正训练图像/状态/参考来自当前场景实际执行。
5. 先跑原生N1.7小样本拟合和模型闭环。若同数据下一个小型state-BC/ACT/DP诊断能成功而VLA不行，再定位模态、优化或视觉/语言条件；基线作用是排错，不擅自替换GR00T科学主线。
6. 有首批行为后，CoreLab再开已资格多worker扩下版数据，Hopper读冻结当前版训练；评价优先，不以归档或全场景制作作为永久前置。

本研究不重新建立WBC保持RMS全局门，不调其gain/cost，不取消失败分母。规划内部多候选可有界记录，资格确认不可挑seed；真实安全继续生效。

## 10. Zotero使用说明与可复现引用

本会话已读取安装的Zotero技能说明；它通过用户本机Zotero Desktop的local API在Codex运行。本会话未连接/检索/更改用户本地库。提供独立BibTeX只是可导入文献清单，不是Zotero同步完成。

在具备该插件的本机Codex中，先用插件helper `status --json`，然后按论文标题/DOI只读检索和去重。只有用户明确要求导入并确认目标collection后，才import-bibtex；不要因此重启正在运行的实验。本文献key不是Zotero item key。

## 11. 原始来源（访问日2026-10-15）

[R1] ManiSkill motion planning: https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/motionplanning.html ; ManiSkill3 paper https://arxiv.org/abs/2410.00425 .
[R2] MimicGen paper https://arxiv.org/abs/2310.17596 .
[R3] MimicGen custom framework interface https://mimicgen.github.io/docs/tutorials/datagen_custom.html .
[R4] SkillMimicGen paper https://arxiv.org/abs/2410.18907 ; Isaac Lab implementation https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html .
[R5] NVIDIA NEW_EMBODIMENT https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md .
[R6] NVIDIA modality schema https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_config.md .
[R7] OpenVLA https://openvla.github.io/ ; https://arxiv.org/abs/2406.09246 .
[R8] OpenVLA-OFT https://arxiv.org/abs/2502.19645 .
[R9] OpenPI https://github.com/Physical-Intelligence/openpi ; target-data recommendation https://github.com/Physical-Intelligence/openpi/blob/main/examples/aloha_real/README.md .
[R10] Octo https://octo-models.github.io/ .
[R11] LeRobot workflow https://huggingface.co/docs/lerobot/en/il_robots .
[R12] Isaac Lab Arena0.2.1 https://isaac-sim.github.io/IsaacLab-Arena/release/0.2.1/pages/example_workflows/static_apple/step_3_policy_training.html .
[R13] ManiSkill simulation teleoperation https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/teleoperation.html .
[R14] robosuite simulation demonstrations https://robosuite.ai/docs/algorithms/demonstrations.html .
[R15] RoboTwin2 https://arxiv.org/abs/2506.18088 ; https://robotwin-platform.github.io/ .
[R16] UMI on Legs https://arxiv.org/abs/2407.10353 ; https://umi-on-legs.github.io/ .
[R17] MPlib plan API/collision limitations https://motion-planning-lib.readthedocs.io/latest/tutorials/plan_a_path.html .

以上网页/论文用于方法与实施对照，不直接赋予第三方数据再发布许可；代码、数据、模型和资产许可应按实际下载版本分别登记。没有引用网络转述的巨大提升百分比作为我们成功率或提速预期。
