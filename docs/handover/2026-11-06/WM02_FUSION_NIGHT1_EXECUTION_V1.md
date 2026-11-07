# WM02 FUSION-NIGHT1：冻结 GR00T × 本体 WM × 冻结 WBC

**计划日期：2026-11-06，Asia/Singapore。状态：PLAN_ONLY_NOT_EXECUTED。**
用户本轮要求夜间长程无人自主推进；承接“冻结WBC、不设项目内部工程总预算、本地小试先行、H200/GPU物理多环境前人工审核”。本文件提出新科学实验，不是新现场凭证，也不表示已启动任何进程。

## 0. 唯一入口、接管方式和本夜要回答的问题

从 `WM02_FUSION_NIGHT1_START.md` 开始，再读取本文件与 `experiments/wm02/fusion_night1_v1.json`。
执行基线：未合并 PR #364 的固定提交 `ad72c0e1779f015f18780dd173eef9fe3f40125e`。它包含 PR #362/#363 的已有代码和PILOT2实测，不假设main已有这些变化。当前PILOT2报告仍有活动接触timing任务：**不打断健康worker、不热改capsule、不重跑已完成矩阵、不创建第二个物理owner。** 在现有管理器的下一个封存批次冷边界，以独立新worktree/capsule接续；接管前读取实际现场状态，不能把GitHub的ACTIVE当实时探针。

### 科学主线

> 在视觉/动作语义已经可解释的前提下，本体WM能否让同一个冻结GR00T提出的参考，在Go2＋ARX5＋冻结WBC上产生更接近期望的实际运动？

主链为：

`实际观测 → 冻结GR00T → 名义动作 → 语义/坐标/时间解码 → 有界WM参考校正 → 原SafetyGate → 冻结WBC → 实际状态反馈`。

**先证明参考校正有物理作用，再证明对同一冻结VLA有收益；不要求小Actor先超过规则，也不等待所有PILOT2面板结束。** 原PILOT2学习与接触工作保留为独立证据与可复用支线，不以新计划抹去或重训。

### 本夜交付等级（不是预先宣称成功）

- **M0：执行校正实测。** 至少一个合法新TRAIN诊断组中，有未经校正、简单校准和WM校正的实际运行、预测前置、真实误差及失败分母。正结果和负结果都算有效科研交付；单组不算控制增益通过。
- **M1：真实VLA接入。** 一个实际可用的冻结GR00T checkpoint，在真实/合法已存双RGB及机器人状态上实际生成动作，保存身份、原始输出和动作语义；无物理也可先完成shadow，但不能叫融合控制成功。
- **M2：冻结VLA融合闭环。** 在相同任务/观察/动作与安全范围，比较同一GR00T的direct与WM-corrected原生闭环。必须是真实在线观测更新；保存轨迹回放不能冒称在线VLA。
- **M3：反哺学习。** 把真实新TRAIN转移用于WM/小adapter训练、保存恢复，再用新的冻结评价验证。离线拟合、教师模仿和真实控制收益分开。

缺少checkpoint或动作接口未知时，明确 `VLA_BLOCKED_<REASON>`，继续M0和已有接触学习，不整夜空等。没有真实VLA调用就不得将校准轨迹改名为GR00T输出。

## 1. 已查证的基础与未知项

以下仅是固定HEAD中的报告事实，非本计划作者远端复验。

1. `reports/WM02/pilot2-20261105/PILOT2_ACTUAL_PROGRESS.md` 的最后冷边界记录：PILOT2已封存83次原生PASS、363949 tick、192000个多模型累计Actor主更新＋22探针、8000个WM主更新＋4探针。活动timing尝试另列，不能预计PASS。
2. 已完成24/53历史×cyclic/shuffle×3seed匹配。8k端点shuffle的DEV教师一致率约95.65%–97.66%；后续117 TRAIN/原8 DEV的三seed达到98.44%–98.68%。这是教师模仿，不是新原生Policy收益；不要再从头重复整个矩阵。
3. 两批8个新TRAIN父提供64个动作episode、1536条新可用转移；117/8 Replay：`9ce72009c7bf3505e196627c6ee170941d823cbe43fc3c30455cbb826b6f0791`。旧TEST仍不能加入训练。
4. 接触多视野模型已经有小样本真实学习：1 TRAIN父/1 DEV父、6个episode、954个相关anchor、5010个有效视野标签。DEV32步grasp Brier动作条件0.06197、动作盲0.43165、持恒0.25；一步仍不及持恒。单父、正常化/优化因素和脚本相关性仍限制因果与泛化结论。后续normalizer对照没有证明根因已解决。
5. 已有实际受支撑夹持/保持/释放脚本正例。目标条件接触Policy入口已经实现，但固定报告中真实新Policy更新/原生评价仍为0。必须先查现场，避免覆盖已经完成的后继。
6. 冻结原生C08二进制：`29b261541b9dc2b2ed5c6df065026b33a11c3019181b9a8cfa980a7878d91df2`；已验证局部ARX：`c69ae25164b023cb005e17aaa0c6a45b9d75f756963a1887c782597e7acf55d1`。
7. `src/go2arx5_vla/world_model/target_arx.py` 是历史条件、固定goal和原时钟的H16递推模型。当前四候选为HOLD/+X1mm/−X1mm/+Y1mm。**它不是任意H40/Z/旋转/接触轨迹的已资格模型，也不是完整MPC隐藏状态。**
8. 尚未核验有可直接在新本体上工作的GR00T checkpoint、目标双RGB适配与动作语义；也没有“冻结GR00T＋WM校正”控制增益证据。NVIDIA仓库HEAD不是本地已安装版本，必须读取实际checkpoint/processor/config并固定版本。

参考：`docs/CODE_MAP.md`、上述PILOT2报告、`reports/WM02/pilot1-20261105/PILOT1_MATCHED_COVERAGE_AND_NATIVE_TEST.md` 与 `PILOT1_FIXED_TEST_AND_SUPPORTED_CLOSE.md`。

## 2. 项目授权、夜间时钟与资源

### 2.1 无项目内部累计工程预算，仍保留真实站点和安全边界

不设项目累计CPU/GPU小时、token、储存、尝试次数、修复次数、物理tick、优化更新或总运行小时封顶。下文面板/seed/checkpoint是**有限实验设计**，可在合法范围依据新证据登记后继，不是全goal预算。

每个worker仍必须使用原owner/registry/Journal/资源锁、有效有限lease、实际deadline、明确终点/STOP与完整收尾。检查现有授权是否覆盖冻结VLA推理和新reference-adaptation profile；不足则通过原机制登记真实新范围，不改字段伪造许可。不复活CLOSED任务，不重放副作用未知的运行，不改旧正式面板。资源/权限阻断时转只读工作。

计划在2026-11-06夜间启动语境下编写：**当地09:00是晨间报告检查点，不是新增工程总预算或站点续期。** 执行者绑定实际启动时间和时区；如果这个检查点已过，使用实际启动后的首个当地09:00并明确写日期。到检查点在安全批次边界封存短报告；有合法后继则继续，否则关闭所有自有worker并交接。实际site/parent deadline更早时优先正常收尾，预留至少20分钟；不能为了09:00报告超期，也不能未经许可自动延长旧scope。本文件不创建自动化通知或远端常驻进程。

### 2.2 默认计算配置

- 一个canonical physical owner，复用已资格1/2个CPU MuJoCo＋ROS2/OCS2原生slot；新轨迹/接触模式先单环境。
- 本地一个实际有余量且已授权GPU。GPU0历史上有他人任务，但不能把旧占用快照当实时事实；新运行前只核资源，不动他人进程。
- GR00T推理与小WM/adapter训练优先串行共享同卡。冻结VLA单次推理可能比小模型占显存更多，先测全进程headroom；失败保留，用已有合法资产/离线分支继续，不偷偷切H200。
- 原生ROS Python不安装Torch，不导入训练环境；重用独立推理进程和NumPy/IPC桥。不能让VLA推理阻塞腿Policy/PD或原STOP响应。
- bulk文件使用已验收非根本地数据盘；检查字节/inode与关键fsync/rename行为，不重建存储平台、不自主删除历史证据/模型。

### 2.3 冻结与规模审核

**GR00T全部权重/processor统计在融合对照内冻结；WBC、腿Policy、增益、cost、solver、机器人质量惯量/摩擦/关节参数、2ms dt、原生碰撞/限位/STOP全冻结。**

允许的“物理纠正”仅是**外部高层参考适配器**，必须版本化、受同一SafetyGate约束且在已验证动作支持域内；这是对旧“禁止补偿项目”条款的窄范围更新，不允许调WBC或用大偏置强行补旧68mm下降。早期无目标数据迁移协议不被回写。

任何H200 allocation/排队/作业、GPU物理multiworld构造或试跑、DDP/多节点、GPU批量替代/蒸馏WBC前，停止为 `AWAITING_H200_SCALE_REVIEW`，交付实际瓶颈、科学必要性、等价性/新域迁移与回滚包。不能把试跑叫离线准备绕过审核。

禁止真机、主动行走、系统/网络/驱动/凭证/他人任务修改、云端私有资产发布、自动合并main。现役健康作业自然封存后才接续；不要求另行关闭所有尚未执行的旧计划来获得空闲owner。

## 3. 统一科学合同：三种轨迹绝不混淆

每条样本分别保存：

- `vla_raw_action`：冻结模型真实输出及checkpoint/processor/seed/观测身份；
- `nominal_reference`：已解码但未经过本体WM修正的命令；
- `desired_outcome`：期望真实EEF/夹爪/物体后果及其来源；
- `submitted_reference`：通过证书并真正安装的H40；
- `consumed_reference`：原生实际时间/索引/命令证据（derived字段仍标derived）；
- `achieved_state`：真实TCP/BASE/夹爪/物体/接触结果。

`desired_outcome_provenance`只能取以下已证明的一种：

**INTENT_POSE**：动作标签经核实表示希望到达的任务空间位姿；不能只因为字段叫EEF就认定。

**SOURCE_OUTCOME**：源策略命令先经已验证源执行器/源WM或真实源记录得到期望结果，再作已知坐标映射。源WM未资格时不能拿它产生目标真值。

**CALIBRATION_INTENT**：独立、预登记的解析局部目标，用于物理纠正辨识；不能改称GR00T任务。

来源未知标 `SEMANTICS_UNRESOLVED`，只允许shadow/数据审计，不强行选其一。若源命令已经补偿了源控制器，再把命令直接作为实际位姿目标可能二次补偿，必须检查。

动作合同覆盖：world/BASE/camera/tool frame，TCP到夹指变换，相对当前观测vs相邻点增量，左/右乘旋转、quat/rot6d约定与单位，夹爪逻辑开度/物理开度/内部限速目标，预测长度与执行前缀、point dt、source时间/有效期/preview。

配置/shape可读不等于适配完成。不能伪装embodiment tag、补虚假关节或静默换normalizer。若只能用oracle/真物体状态完成桥接，应标privileged诊断且所有比较臂使用同信息，不能称双RGB部署结果。

## 4. 阶段F0：冷态接续与资产清单（不全仓重审）

读取实际现有worker/lease/磁盘/GPU状态与PR #364最新短报告。对已完成矩阵、已采物理episode、训练和timing后继按真实manifest去重；保留每一个正在运行capsule的源码/终点和原结果。新代码在独立worktree开发，只在新worker绑定。

输出 `FUSION_ASSET_MANIFEST`，记录：

1. 原生executor/robot/scene/checkpoint/ARX与时钟合同，真实可复用路径；
2. 可用GR00T候选的checkpoint SHA/父版本、embodiment、modality、processor/normalizer、state/action keys、已知源任务结果、相机分辨率/坐标标定、实际服务入口；
3. 候选选择只依据已有TRAIN/DEV证据与接口兼容性，禁止挑正式TEST成绩最高的模型；
4. 是否存在真正源技能可用、仅推理可用、资产不可用三种状态。

复用：`contracts/`、`trajectory/`、`adapters/`、`target_mujoco/`，GR00T层优先 `scripts/pi0292/train_adapter.py`、`scripts/pi0294/generalization/`、`scripts/pi0294/pv1/`；目标执行复用 `auto1_native.py`、`target_sequential.py`、`pilot1_native_worker.py` 与 `target_parallel.py`。实际文件/函数仍由Codex先读，不凭名称调用旧绑定命令。

**源任务成功不是首次真实推理的前提，但“源技能跨本体迁移成功”的必要证据。** 不为融合试验等待共享四技能全部训成；也不重新开展GR00T微调。本夜先冻结最早具备有效兼容入口的候选。

## 5. 阶段F1：动作语义、时间和支持域

实现一个薄decoder/contract，CPU正负例涵盖：非零旋转、非单位BASE变换、相对位姿累积、夹爪两种方向、不同execution horizon和不连续时间；未知单位/维度/坐标必须拒绝。官方网络文档仅帮助查术语，以现场实际源码/元数据为准，禁止自动升级依赖。

本地参考文档可核对：
- https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md
- https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md
这些不是冻结运行版本，不预设通用新本体zero-shot能力。

建立 `MODEL_VALIDITY_ENVELOPE`：状态/历史/动作族/幅度/旋转/夹爪/场景/时钟/H范围、训练分布距离、有效参考年龄、不确定性/拒绝规则。现有ARX只用于原固定goal、H16、burn历史与既有支持域。H40后缀即使未执行也可能被WBC preview使用，应保存并纳入相应模型条件；但不能将未执行后缀当真实监督。

**小幅修正叠加到大VLA动作后仍然可能是OOD。** 检查最终提交的整个参考，而不仅检查delta。不能通过clamp或nearest-neighbor投影隐藏VLA越界；必须保存原动作、投影大小、覆盖率，并标 `PROJECTED_VLA_DIAGNOSTIC`。

如果nominal或corrected不在模型支持域，优先：对该方向/幅度开展新、有限、真实安全资格与任务无关TRAIN采集，然后重新拟合/验证；否则保持/拒绝、转其它合格条件。不能用已有XY1mm模型直接逆解Z、旋转、任意H40或68mm误差。

## 6. 阶段F2：先验证“本体WM能否纠正已知意图”

### 6.1 目标、比较与原生执行

使用 `CALIBRATION_INTENT`，从TRAIN侧真实可达局部结果与独立任务定义生成目标，不用WM自己的预测同时当目标和真值。优先保持夹爪打开、旋转和BASE参考不变。初版采用原合格动作集合内的候选选择，明确称局部选择式校正；连续补偿另行资格。

固定比较臂：
- **B0 direct**：统一decoder和原SafetyGate，无学习型修正；
- **B1 static calibration**：只用TRAIN估计的常量/线性平均响应校准；
- **B2 feedback-only**：相同观察、重规划机会和合法候选，使用预先冻结的简单误差反馈参考规则，不改WBC内部增益；
- **B3 WM correction**：历史条件预测＋受限选择/优化；所有状态、时钟和任务信息与基线对齐。

若B2需要多次决策而现有协议仅允许一次，则本夜先做所有臂共同的单决策版，B2记与对应简单规则等价，不额外赋予它更频繁控制。多决策完整公平对照另行资格，绝不静默提高一方控制频率。

先在一个新TRAIN父完成四臂的最小实际组，立刻封存。然后建议展开4新TRAIN父＋2新DEV父、每父2种预声明意图、4方法、2次独立重启，共96次方法评价，资格另计；这是起始设计不是预算。按方法顺序和slot交叉随机化，数据/模型版本在面板前冻结。

所有参考必须经过原SafetyGate；同一分布被拒绝的条件计入coverage/失败，不能只对WM恰好支持的子集报告整体成功。资格检查、native PASS、任务PASS、证据无效分开。起点独立重启，不是完整MPC恢复分叉；记录前缀散布。

### 6.2 求解器边界

建议目标：`predicted outcome error + reference deviation + smoothness + calibrated uncertainty`，约束为当前真实合格动作域/速度/加速度/关节/接触/连续性/TTL。初版枚举小集合即可，不为“可微”强迫优化连续动作。

若研究梯度法，只能在模型autograd与数值差分一致、物理局部有限差分方向也经新TRAIN验证后启用；小步trust region、显式停止和失配检测。ARX逆映射病态时用正则/阻尼或退回已合格候选，不放大指令。未知安全成本不能用模型伪造硬安全保证。

不能在控制回调中耗时求解。推理/校正晚于部署时限时取消未安装建议，沿原合法HOLD/续约路径处理；记录超时和动作覆盖。暂停仿真上的推理只可标 `PAUSE_ASSISTED`，不冒称实时。不能为过期数据伪造新观测时间；必须基于新的实际源观测重新编译/安装并保留链路。

### 6.3 评价与下一步判定

报告实际world和BASE-relative TCP、姿态、BASE运动、夹爪、轨迹/端点误差、命令与实际事件时延、reference修正量、拒绝/回退/超时、native安全、推理wall和GPU/CPU成本。只做XY时明确不等于SE3全任务。

实际效果阈值由TRAIN噪声和任务容差事先冻结。按父/意图汇总连续代价差与区间，不以相关50ms帧扩大样本量，不以4次试验宣称统计可靠优势。

- 物理执行通过且无增益：保留负结果，继续F3 shadow与已就绪接触学习，不强迫继续调参。
- B1/B2足够：明确WM额外价值未证实，保留简单修正做实用baseline。
- B3在DEV有可复现实际改善且无新增安全/覆盖代价：冻结后进入F4；不是仅模型分数低就晋级。
- 接口错配导致所有臂异常：仅修该数据/合同问题，独立新版本，不重写旧成绩。

## 7. 阶段F3：冻结GR00T的真实推理与shadow接入

F3可与F2离线实现并行，GPU资源按原owner串行。实际调用固定checkpoint，使用已验证的真实图像/状态，保存SHA、模型/配置身份、seed、原始动作、decoder结果、所有候选和预测。图像/模型不进公共Git。

优先级：实际当前双RGB → 合法原始封存观测与真实模型重新推理 → 既有真实VLA动作回放。后两种必须分别标离线推理/动作回放，不能顶替在线闭环。

核查视觉和状态输入是否匹配；不使用机器人真物体pose、手工任务phase或未来状态偷渡到VLA。若融合修正器使用额外可测状态，应披露且所有对照可用同样信息；只在仿真可得的信息标privileged。不得通过错误embodiment tag“让接口能跑”来声称zero-shot。

**源命令≠意图时**，选择SOURCE_OUTCOME路线：用合法已验证源执行器/源模型预测相同源命令的真实后果，明确frame映射和源模型误差。没有可信源响应则shadow止步，CALIBRATION_INTENT线继续。不能编造源结果、不能将目标未来实际状态作为期望轨迹。

一次真实模型推理就应产生可审计交付；不以新GUI、全部技能、GR00T再训练或完整latent WM为前置。没有资产则输出具体缺项，不扫描整个Home/拉取未授权大权重/安装新系统依赖。

## 8. 阶段F4：同一冻结VLA的物理对照与在线闭环

依赖F1语义确定、F2相应支持域有真实证据、F3真实checkpoint及输入可用。源任务能力弱时可以做动作级实验，但不得标任务迁移成功。

分两类证据：

**F4a paired proposal replay**：固定由真实VLA产生的名义轨迹/输入，比较B0/B1/B2/B3。独立重启和起点差异量化，不伪称完整隐状态fork。目标是隔离校正层，而非在线VLA闭环。

**F4b live closed loop**：同一个checkpoint、同一任务/起始分布/seed日程/感知接口，分别独立运行direct、简单校正、WM校正；每一步反馈各自真实图像与实际本体状态。不同策略导致后续观测不同是闭环的正常结果，不能强迫复用反事实图像。

先一条新TRAIN诊断，再建议至少2个新DEV父×2任务条件×3核心方法×2重复。冻结checkpoint、WM、适配器、阈值和选择规则后，才登记新确认TEST。所有正式TEST至多按原协议执行，不为成功重试；当前P23–P26及历史所有已消耗TEST永久保留，不进入fit。

指标同时包含原生任务成功、实际轨迹误差、接触/支撑/释放结果、拒绝比例、最大修正、执行/推理时延与成本。更小的EEF误差不能冒充物体任务收益。视觉选错/语义阶段错、名义参考不可行、执行响应错、物体交互错四类分开。

**zero-shot声明矩阵：**
- GR00T权重冻结，不等于目标数据为零；
- 目标WM用一般辨识数据、没有目标任务示范：可研究 `FROZEN_VLA_TARGET_TASK_DEMO_FREE`；
- 若使用SC07B同目标任务示范训练adapter/交互WM：标 `TARGET_ADAPTED`；
- 若手工把VLA输出投影成四动作：只称 `PROJECTED_VLA_DIAGNOSTIC`；
- 若无真实VLA调用：`NON_VLA_CALIBRATION_ONLY`；
- 不声称官方新本体原生zero-shot，也不把单任务成功当跨任务泛化。

## 9. 阶段F5：数据回流与轻量适配器训练

保持GR00T冻结。新TRAIN记录必须同时保存nominal、desired provenance、installed/consumed、actual history/outcome、真实任务回报、安全/时间、模型不确定性与拒绝。有效物理失败用于相应负例；无执行/日志破损记录不能伪造监督。

两种有限新学习可自主开展：

1. **目标响应模型更新**：冻结旧ARX作为对照，用新校正分布下TRAIN辨识或零输出小残差/直接多视野模型；新动作族需要明确独立训练合同，不能静默改原ARX行为。
2. **适配器蒸馏**：`adapter(history, VLA_nominal) -> bounded correction / legal candidate`，初始网络保持小型。优化器求解的建议是模型标签，实际验证结果单列，不当作真实反事实最优动作。需要时用真实连续代价加权，不对偶发阈值失败强行惩罚。

首个端点建议2k，按预登记TRAIN/DEV曲线可延续8k/20k，非总上限。shuffle与sampler/generator/cursor完整恢复复用PILOT2已验证实现；不再重做12臂。每个checkpoint保存normalizer/模型/优化器/RNG/sampler和所用VLA/WM/动作合同身份。

训练完即做一个合法小原生组与冻结baseline比较，再收集新TRAIN、继续学习、评价；不要只给出更低loss。大规模VLA微调、视觉域适配训练和连续接触BPTT不是本夜默认工作，必须另立明确科学配方与资格，不能破坏冻结VLA对照。

## 10. 独立接触线C：继续现有进展，不成为融合的万能阻断

自然结算正在运行的PILOT2 timing组；复用已通过共同close56前缀与5303tick决策边界的实现。核现场资格而非只看本报告；新受支撑策略的动作以实际已验证KEEP/RELEASE为准。

已有多视野接触WM可评价不同program的后果，但它是**partial direct forecaster**，不能把7个输出补成qpos/ctrl/force再自由展开。保留真实后缀mask，只有在动作实际选定后才把对应未来program作为条件；不能提前把自适应阶段结果泄漏。

先完成目标条件“保持支持夹持/释放并支撑”的真实小Policy训练—恢复—原生评价—TRAIN反馈，与原目标FSM、经验回报教师、WM教师同信息对照。状态/目标不变化导致恒定动作足够时如实报告，不能强迫复杂WM胜出。

融合到GR00T夹爪时，先确认其开度与阶段意图语义，仅在已有接触模式/相机/动作程序范围运行；真实grasp/support作为可用测量或privileged诊断的角色明确。新布局按完整初始化扫掠与原native安全检查，禁止live object weld/瞬移/固定Go2底座。

这条线在F3资源受阻时继续；不重复旧C2大幅下降，不为了night成果尝试新开门/插书/抬物/行走。完整抓放仍需后继单独资格。

## 11. 自动接力与停止

状态机使用现有管理器：`READY → RUNNING → SEALED_PASS / SEALED_NEGATIVE / BLOCKED_EXTERNAL / NEEDS_SCALE_REVIEW`。只补一个科学任务清单与短状态文件，不新建多层调度/预算平台。

执行优先：当前健康批次自然完成 → F0/F1 → F2首个实际组；F3可独立准备；F4依资格接续；F5闭环学习；C线复用空闲资源。F3阻断不阻断F2/C；F2无增益可继续shadow但不得宣称真实融合收益；某一方法失败不重复挑成功。普通工程问题自主定位、单变因修复和新run，不逐小步骤询问。

必须停止相关动态分支：原生安全故障、有效owner/期限/资源不足、不可解释数据污染、未知动作语义、原模型/权重改变、未验证的连续动作外推、物理实时性受阻、H200/GPU物理规模触发。停止后保留证据并转其它合法离线支线；重复同一症状没有新假说时不盲跑。

**允许持续自主推进，不等于无限运行同一个worker或无证据反复修改。** 如某一科学假说被否定，输出负结果和替代解释，不通过降低评分/删除失败达成PASS。

## 12. 实现与报告合同

复用现有执行器、模型恢复、两槽采样、场景构造、观察和可视化入口。建议新增以下薄模块，名称是待实现建议，不是已存在CLI：
- `fusion_contract`：源/目标动作语义和零样本声明；
- `fusion_corrector`：identity、静态校准、反馈、WM候选/受限修正；
- `fusion_proposals`：冻结模型推理/回放/desired来源封存；
- `fusion_native`：通过现有native worker在新profile下执行、评价与回流。

先导出实际可执行`--help`/测试与真实命令再列“下一条CLI”；不要提供虚构已存在的运行命令。数据处理保持NumPy层与Torch层分离。单测覆盖坐标/单位/时钟、identity等价、支持域拒绝、过期HOLD、接触模式、TEST隔离、恢复、原生/Replay输入一致；单测不替代物理证据。

每个冷边界写一份短 `FUSION_NIGHT1_STATUS`：当前有效source/owner、各M0–M3状态、真实VLA次数/采样/更新、下一READY、阻断；晨间/终态写详细报告，至少含：
- frozen VLA/processor、WM、native、scene、数据与checkpoint SHA；
- 各臂实际命令/退出/attempt分母/原生PASS/任务结果/拒绝与失败；
- nominal/desired/installed/consumed/achieved的不同含义与实际来源；
- 每父效果、回退/不确定性/域外覆盖、latency与成本；
- 新TRAIN/DEV/TEST、teacher建议与真实标签分列；
- 活跃与终态worker清单，晨间报告不伪称全部已关闭；
- `CALIBRATION_CORRECTION_GAIN`、`REAL_VLA_INFERENCE_DONE`、`FROZEN_VLA_NATIVE_COMPARISON`、`TASK_TRANSFER_GAIN`独立标签；
- 一个真实已实现的接续命令或明确缺失项。

本计划发布效果：native tick=0、WM/Actor/VLA更新=0、VLA推理=0、新现场scope=0、worker=0。GitHub文件的发布不会自动启动Codex或后台任务。
