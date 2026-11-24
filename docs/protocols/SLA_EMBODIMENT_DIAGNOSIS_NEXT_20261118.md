# SLA 移动本体适配：问题分析与下一步实验计划

日期：2026-11-18（Asia/Singapore）。状态：**PLAN_ONLY / NOT_EXECUTED**。基于 PR409 在 11:38 SGT 回读的固定 head 31db982450bd879a5f88f28fe5009206528e0544；最新已发布 M 终态为 11:15:21 SGT。本次完成 GitHub 报告与代码审阅，未登录 Core、独立读取私有原始 trace、运行训练或原生仿真。

本计划细化 PR408 的 Teacher 修复及后继 M 比较；不覆盖原 V6 负结果，不改变已封存人口、统计、端点和停止条件。配套：[机读设计](SLA_EMBODIMENT_DIAGNOSIS_NEXT_20261118.json) · [启动 prompt](../handover/2026-11-18/START_SLA_EMBODIMENT_NEXT.md) · [入口](../../SLA_EMBODIMENT_NEXT_START.md)。

## 1. 当前判断

当前最优先的工作是建立**目标本体可执行的完整示范和可解释的执行误差链**。已有结果证明学习程序有效、边界约束有局部收益；它们没有证明任务闭环可靠，更没有完成源域到 Go2+ARX5 的配对迁移评价。

| 线路 | 已完成事实 | 当前结论 |
|---|---|---|
| T：单任务可靠性 | B1 5/8、FM 3/8、H40 6/8；合计14成功、9加速度拒绝、1接触STOP、0技术失败 | H40减少已知拒绝，但出现碰桌，无合格者；按原计划停止本轮T追加训练 |
| S：源域未来参考 | 1000更新、32000访问；DEV动态误差相对S0下降52.49%；TEST96动态误差相对保持当前状态下降56.94% | 源episode留出预测有收益；不是源真机任务SR或Go2迁移成绩 |
| M：目标共享任务 | Teacher 8/48，3完整成功、5接触STOP；学生更新、生成和请求均0 | 当前阻碍是完整示范资格；尚未做M-BASE/M-SRC目标比较 |
| 旧RealMix | 命令来源、frame/TCP、同步语义未建立；SHIFT7表示不能忠实覆盖大量源夹爪曲线 | 继续BLOCKED_LABEL_SEMANTICS；S原生未来参考路线与直接混命令是不同实验 |

T三臂真实初态没有完整WBC内部状态逐位配对：query0最大TCP差0.876mm、底座差4.932mm。因此6/8与3/8是固定小面板结果，不能当成精确、稳定的因果效应；固定离线输入能更好隔离模型输出差异。

来源：[T完整终态](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/reports/WM02/sla-source-native-shared-v6-20261118/TARGET_SCREEN_ACTUAL.md)、[S终态](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/reports/WM02/sla-source-native-shared-v6-20261118/SOURCE_ACTUAL.md)、[M终态](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/reports/WM02/sla-source-native-shared-v6-20261118/M_WORKER_SETUP_ACTUAL.md)、[RealMix兼容性](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4b7040d3b9f43bdc0514609d564d91ecff67410e/reports/WM02/sla-robodojo-real-sim-v1-20261118/CODEC_COMPATIBILITY.md)。

## 2. 阻碍、证据强度与解决路径

### 2.1 Teacher 的几何可行性与真实移动执行：已经发生的首要阻碍

三个完整正例全部属于蓝物体：F2 HOLD-blue、F1 HOLD-blue、F1 PLACE-blue-left；红物体完整成功为0。三条正例各占不同的“指令×family”单元，每格只有1条。不能把280个相邻H40窗口当作280次独立任务经验。

| 机制 | 已保存证据 | 对应修复和验证 |
|---|---|---|
| F1红物体下降/闭合碰桌 | HOLD-red及PLACE-red-right均STOP；HOLD末段提交参考垫桌净空约12.6–12.9mm，实际TCP比参考低10.7–11.1mm | 同时核对参考身份、实际姿态漂移和抓取构型；从可夹持姿态集合选择有余量的端点 |
| F2红物体抓取构型碰腕 | 原静态目标腕端与蓝物体距离−16.62mm；末H40名义22/40、提交40/40重叠 | 改变Teacher抓取位置/朝向，验证完整接近、闭合、提起；不能靠增加学生更新学习不可行标签 |
| F1红物体搬运时垫碰蓝物体 | 固定姿态扫掠无碰撞；提交点间隙约2.6–2.9mm，实际发生接触 | 将姿态与跟踪误差纳入腕、垫、持有物扫掠；不能只检查物体中心路径 |
| F2蓝物体搬运碰红物体 | 持有物横移2.72mm即扫掠相交，缺约19.08mm垂向净空 | 先抬升到实际净空成立再横移；需把持有物、夹爪、腕及整臂一并验证 |

当前8mm搬运余量候选只覆盖持有物与另一物体的XY扫掠，不能视为整臂安全保证。第八条在CLOSE失败，未进入CARRY，该条没有动态验证搬运修复。F1目标升高16mm只是静态候选；不能将其变成全局WBC偏置或学生输出补偿。F2的16个候选施加已测STOP位姿差后虽仍无禁止碰撞，但0/16保留双侧夹持：避碰与抓持必须同时成立。

**决策：**先在存量状态上定位问题、修必要Teacher；用原未启动条件进行动态验证。若冻结执行器的实际误差使所有可达、可夹持构型都失效，应报告执行能力限制，另立明确问题的执行研究，而非继续同分布训练。

来源：[M实绩](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/reports/WM02/sla-source-native-shared-v6-20261118/M_WORKER_SETUP_ACTUAL.md)、[Teacher实现](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/src/go2arx5_vla/world_model/multitask_teacher_v6.py)。

### 2.2 H40约束合格与接触成功之间仍有缺口：已经证实

T-H40沿真实10步自由生成反传，500更新中有2000次辅助反传，104条非零。这个数不证明梯度损坏：训练anchor大多已经在0.20m/s²余量内。H40相比FM减少拒绝，但一条新状态仍在末点达到0.276150m/s²，另一次碰桌。

原约束采用：

    v = diff(p) / 0.05
    a = norm(diff([0, v, 0]) / 0.05)
    L_dyn = mean_batch(max_time(relu((a - 0.20) / 0.25)^2))

首尾补零对应原执行门约定，不是机器人真实速度的接缝动力学。首端0.25m/s²限制相当于第一相邻位置差不超过0.625mm。该loss不包含腕部、夹爪、持有物与环境的接触几何。

移动模式下，固定根IK/接触检查在代码中为shadow_only；它不是完整移动体的可行性证书，也不能简单升格为硬门，因为成功回合也有固定根关节加速度旁路拒绝。正式H40与原生STOP按原规则执行。

**决策：**完整H40和STOP保持；把轨迹包络、参考传递、实际跟踪和任务成功分别评价。按PR408结束本轮T，不追加第三个同数据loss/系数/500步。

来源：[完整生成loss](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/src/go2arx5_vla/sla_adapter/full_h40_v6.py)、[移动提交与旁路](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/src/go2arx5_vla/world_model/mobile_wbc_reference_v1.py)、[原T停止规则](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/ae3d3c0dceade437b0104a150cd0402fef650052/docs/protocols/SLA_SOURCE_NATIVE_SHARED_POLICY_V6_20261118.md)。

### 2.3 参考选择与实际跟踪的因果链还不完整：已知证据缺口

现有代码证明名义H40、提交参考、receiver接受和solver返回匹配目标，但明确没有导出最终controller selected policy/reference。字段为DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED。

F1 HOLD-red最后210ms的TCP下降29.86mm，可按刚体运动分解为安装座平移−18.60mm、安装座旋转−2.19mm、臂端相对运动−9.07mm。这说明移动安装座运动不可忽略；该运动学分解不证明WBC算法出错。参考选择、参考龄期、安装转换和真实跟踪仍需分离。

**决策：**优先用已有只读接口补齐selected policy ID、时间和目标。若导出必须改变冻结控制器，先记录UNKNOWN，按单独工程版本处理；不能虚构字段或默默改变WBC身份。没有selected证据时，仅称“相对提交参考的误差”，不称“相对实际选中参考的跟踪误差”。

来源：[边界证据实现](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/src/go2arx5_vla/world_model/mobile_boundary_clock_v1.py)、[实际模型输入身份](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/src/go2arx5_vla/world_model/shift7_native_input_v5.py)。

### 2.4 当前输入对移动动态与控制历史可能不足：待验证机制

学生输入为当前body/wrist RGB、当前EEF pose/gripper和episode固定坐标变换。当前base twist、关节速度、TCP速度、上一段EEF参考、实际参考跟踪误差未进入策略。记录了qvel不等于网络看到了qvel。固定坐标变换具有刚体roundtrip检查，不能把动态信息缺失说成已发现坐标变换错误。

Teacher保留连续速度状态、闭合开始时间和阶段记忆，H40预测使用当前实际与参考偏移来近似未来，P10每0.5s才刷新一次。若相近静态观察对应不同运动状态或Teacher历史，标签可出现多值性。

M的夹爪接缝loss使用上一实际命令，但该命令仅为provider/训练元数据，未进入模型生成函数。0.038m/s训练余量对应每50ms可改变1.9mm；执行门0.04m/s对应2.0mm。若上一命令不同而当前可观测状态相近，首点安全区间可能不同。单次measured−command误差大不能证明不可学习；需要检查同观测邻域的标签和可行区间是否冲突。

**决策：**先复用已存数据做观测歧义诊断。只有动态状态/上一命令能够解释明显条件差异时，才另立可测动态输入或短历史对照。Teacher阶段、物体真值不进入部署输入；现役M的输入不能中途更改。不要同时改变数据、Memory、解冻范围、loss和学习率。

来源：[输入白名单](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/src/go2arx5_vla/world_model/shift7_native_input_v5.py)、[G7 provider](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/src/go2arx5_vla/sla_adapter/provider_multitask_v6.py)、[G7速度loss](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/src/go2arx5_vla/sla_adapter/dense_g7_v6.py)、[连续Teacher](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/src/go2arx5_vla/world_model/mobile_continuous_teacher_v2.py)。

### 2.5 目标数据覆盖狭窄，重复优化不产生新经验：已知限制

C20为20条episode、1376个窗口、单一抓取保持操作。每臂16000访问不等于16000条独立示范。T-FM的3/8不足以证明普遍退化，却已经不支持“再训练同一分布自然解决”的预期。

完整H40监督来自当前Teacher计划；真实消费仅P10，后30点是未执行计划尾部。数据合同诚实保留了这个区别，离线分析也应分开P10和尾部。失败回合保留作诊断，未经同状态合法Teacher纠正不能自动成为成功BC标签。

M的六指令是HOLD/PLACE两类操作的颜色和目的地组合，适合作为共享策略第一阶段。当前场景主要共同平移物体，未交换红蓝相对位置；未来泛化需单独新增位置×语言交叉条件，不能改旧TEST。

**决策：**先补完整任务、每个family和关键事件的数据。之后若学生在自身偏离状态失败，再做同状态可执行纠正；最后才扩外观、更多物体与长程顺序。

来源：[C20 reader](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/src/go2arx5_vla/sla_adapter/mfree_training_v1.py)、[G7数据准入](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/src/go2arx5_vla/sla_adapter/multitask_data_v6.py)、[M场景人口](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/scripts/wm02/multitask_population_v6.py)。

### 2.6 源数据语义、动作表达及可迁移参数：不同的三项问题

RoboDojo已核验action[t]=state[t+1]，支持未来记录参考，尚无dispatch证据。源双臂14维、三相机、局部坐标、名义30Hz H50，与目标单臂G7、双相机、50ms H40、移动安装座/WBC不同；同属ARX5不能消除这些差异。

SHIFT7含单次夹爪事件结构，不能忠实表示任意多次开合。M的逐点G7是合理新方法，但它尚未取得目标训练/闭环结果。RealMix的10/25%不能通过换decoder继续伪装为同一实验。

S1000同时训练20个共享LoRA和8个action/time接口数组；M只移植20个LoRA，8个接口、normalizer和Adam不复制。目标I/O从同一官方base保留，不是随机重新初始化。因此源整体误差改善不等于共享可迁移参数取得了同幅度收益。

**决策：**主实验保持M-BASE/M-SRC同目标数据、同目标优化预算；必要时在SOURCE_DEV做2×2模块互换，定位收益是接口、共享参数或共适应。源闭环、目标闭环与离线参考误差分别报告。

来源：[源训练](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/src/go2arx5_vla/sla_adapter/source_training_v6.py)、[源类型合同](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/src/go2arx5_vla/sla_adapter/source_native_v6.py)、[迁移白名单](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/src/go2arx5_vla/sla_adapter/multitask_initialization_v6.py)。

### 2.7 预训练与当前后训练的作用边界

SLA官方论文明确在持续预训练整理中排除底座运动轨迹，其统一EEF表达主要解决机器人之间状态/动作表示差异。它不能自动提供Go2腿式移动平台的动态执行经验。我们从SLA_base做LoRA/FM适配，没有在当前T/S启用Spatial Forcing或Memory，不能称为完整复现论文所有下游配方。

当前M_FREE只输出EEF意图、不显式输出BASE路径，底座运动由WBC产生。因此现有成功是移动抓持能力证据，不是自主底座导航或长程任务规划证据。长程判断需要物体事件、顺序、子目标和恢复评价；不能仅增加episode持续时间。

来源：[官方论文§3.3.2、§4、附录B](https://arxiv.org/html/2611.08373v1)、[M_FREE提交](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/31db982450bd879a5f88f28fe5009206528e0544/src/go2arx5_vla/world_model/mobile_wbc_reference_v1.py)。

## 3. 下一轮目标与顺序

### D0：复用存量证据，完成执行与训练诊断

新增优化0、模型生成0、原生请求0、积分0。读取全部T24及M8的既有输出、所有失败邻域和成功对照，复核原文件身份。不能用一次新推理替换旧动作。

产物只需一份诊断表和可复算脚本，复用原报告/Journal结构：

- 同一native tick对齐nominal、submitted、accepted、selected（未知则UNKNOWN）、actual。
- P10与H40尾30点分别报告；将EEF平移/旋转、gripper、mount运动、参考龄期、最小几何间隙与阶段对齐。
- 既有T训练日志中的active7/padding25误差与参数组梯度，检查state分位数越界/量化饱和；没有记录的量不能补造。FM平均32维不等于已经证明padding主导。
- 既有Teacher的上一夹爪命令与实测宽度差、连续Teacher隐藏状态和标签差；统计以episode/条件分层，窗口不能充当独立试验。
- 六指令×两family覆盖表、每格已尝试a、成功s、剩余4−a；检查 s+(4−a)≥2。任一格不可能满足时，不继续耗尽请求以期待原资格自动成立。
- 计算当前已知净空与实际姿态误差的关系；少数失败姿态只能做敏感性分析，不宣布99%稳健保证。

第一交付标准是能把失败归入明确的几何不可行、参考传递/选择未知、实际执行偏离、或模型生成约束失败。selected尚不可得不永久阻断独立可行的Teacher几何修复。

### D1：修复Teacher，先验证发生故障的阶段

先做零积分几何检查，范围包括腕、两夹爪垫、整臂、持有物、非目标物、桌面。固定旧场景与成功谓词；抓取构型必须同时满足双侧夹持、可达、完整接近/闭合/抬升和已测误差敏感性。将抓取姿态选择、扫掠避障与低层跟踪区分，不能用统一高度补偿混在一起。

动态验证只从**原48中尚未启动的40条**接续，不重新运行已失败的8条作为替换成绩。优先级作用于修复工作，不擅改已经封存的执行顺序。先消费原序列下一段至多8条作为检查点，包含在40条内；每一条启动前固定Teacher版本/行为差分与条件身份。运行中不热改，变化只适用于后续未启动请求。

检查点记录：该条件是否实际进入修复阶段、碰撞对/阶段是否变化、完整成功、无抓持假成功、保留全部成本。如果相同机制再次出现，先核对其证据，再决定下一项修复；没有新证据不重复派发同一机制的“碰运气”请求。不得额外新建8条诊断EP绕过48总账。

允许的修复是有证据的Teacher构型、完整扫掠目标或项目内传递/日志工程缺陷；原WBC/legs/MPC/solver/gains、H40/P10、50ms/2ms、TTL与STOP保持。原物理评分不随候选变化。

### D2：建立目标训练bank

必须达到原规则：6指令×2family，每格≥2完整成功，至少24完整episode；每格总尝试仍≤4。当前3个正例可保留，前提是满足原合同；必须保留每条Teacher版本，不能将8条都说成同一代码。若新修复改变标签合同，需新schema而非伪装为兼容。

逐episode核验实际RGB与状态同步、完整名义H40、真实P10消费、gripper上次实发命令和成功谓词。失败前缀不补成完整成功。C20与新数据仅用于各臂共同目标TRAIN统计一次拟合，DEV/TEST不参与。

在训练前复核所有标签完整XYZ余量和G7夹爪0.038m/s余量，包括上一实际命令到首点；系统性冲突时报告具体来源，不能看模型结果后调系数。保留CLOSE/OPEN/撤离等关键事件覆盖，防大量接近/静止窗口稀释少数事件。

### M：执行原计划唯一源迁移主比较

| 项目 | M-BASE | M-SRC |
|---|---|---|
| 共享初始化 | 官方SLA_base | 同一base＋S1000的20共享LoRA |
| 目标I/O、统计、Adam/RNG初始规则 | 同一 | 同一；87数组中非共享67逐位一致 |
| 目标训练 | 1000更新、32000访问 | 相同 |
| 数据 | 6400旧C20＋25600新目标多任务 | 同一访问、noise、time、dropout与LR |
| 表示与loss | G7；FM32＋0.1完整XYZ＋0.1夹爪速度 | 相同 |
| 正式端点 | 固定1000；独立RGB16恢复 | 相同 |

20/80是旧目标数据与新目标数据构成，不是真机/仿真配比。M-SRC已额外使用1000源更新，只能评价相同目标数据/优化预算下源预适配收益。若未来要证明真实源数据优于额外算力，另做总算力控制，不能改变这两臂含义。

固定48学生screen（每臂24）全部按事前人口和交错次序执行。资格仍为每臂≥18/24、每指令≥3/4、每family≥9/12、0hard STOP和技术失败，并满足原离线/恢复门。唯一候选确认24，同门；通过后DEV60，再一次TEST60，分别要求≥48/60、每指令≥8/10、每family≥24/30、0hard STOP/技术失败。DEV不过不打开TEST。排序和完全平手选M-BASE的规则沿用PR408，不强行选源模型。

报告每个操作事件、正确物体/目的地、H40拒绝、STOP和真实初态差，按scene簇及episode两种单位统计。60回合为10个scene簇×6指令，不能当作60个独立场景。

## 4. 条件性源模块诊断：最多224离线生成，0新训练

触发：M-SRC在固定screen没有目标收益，或者在Teacher等待期间需要澄清共享迁移机制且GPU实际空闲。此诊断不改变S1000已封存门，不作为M-BASE启动前置，也不根据结果更换现役M初始化。

在原SOURCE_DEV96固定输入/噪声/统计上比较：

| 组合 | 共享20 LoRA | 源8个I/O/时间数组 | 生成 |
|---|---|---|---:|
| S0 | S0 | S0 | 复用既有 |
| S1000 | S1000 | S1000 | 复用既有 |
| 诊断A | S1000 | S0 | 新96 |
| 诊断B | S0 | S1000 | 新96 |

两新组合各额外16条恢复复现，共224。恢复16取各自SOURCE_DEV96中事前固定的16个窗口，使用相同输入和noise，对照该组合第一次预测逐位核验；新诊断入口不能自动取原恢复脚本的TRAIN16或另开TEST。source统计、输入、标签和14维有效维度的评价口径不变，所有交换叶子逐名逐shape核对；Optimizer不用于推理。不开SOURCE_TEST，不把新混合包替换S1000或直接移植Go2。

T/S此前已发生1331离线生成；若没有其他新增，1331+224=1555≤原1600。45余额不是新任务承诺。若现场累计不同，重新计算真实余额；无法容纳就交付未执行诊断，不借M预算隐匿超额。

分别报告动态/低变化、各任务、XYZ/SO3/gripper，以及四组合误差。只解释模块依赖和交互：混合组合变差可能是接口共适应破坏，不足以证明LoRA没有技能；源DEV收益也不能替代Go2闭环。

## 5. M后按结果选择下一项，不同时打开所有改动

| 观察 | 下一项有区分力的实验 |
|---|---|
| Teacher尚不能稳定完成 | 继续必要几何/执行诊断；不开始学生更新 |
| 两臂TRAIN拟合也差 | 检查active/padding梯度、统计/标注、事件覆盖和训练margin冲突；证据支持后才比较解冻范围 |
| Teacher状态好、学生偏离状态差 | 采集同一真实偏离状态下合法Teacher纠正；另立同父同loss、只变数据的比较 |
| 两臂出现相同实际漂移或gripper接缝问题 | 先检查执行与可观测性；必要时同父同数据比较当前输入和增加真实动态/上一命令输入 |
| M-SRC与BASE相当或更差 | 上述源2×2模块诊断；检查迁移收益位置与共适应，不直接加公开数据集 |
| M-SRC更好 | 固定目标预算结论；下一周期再做等总算力或源规模比较 |
| 六指令通过、位置交换失效 | 增加颜色×相对位置×目的地交叉数据与新独立人口 |
| 原子抓放通过、顺序任务失败 | 整体指令、oracle子目标诊断、可部署高层分别评分；有历史歧义证据后才匹配Memory |

观测增强、纠正训练、等总算力第三臂、位置交换、Memory和真机不属于现役M预算；该表是后续选择规则，不是自动执行所有分支。触发后先形成具体输入/数据/population/预算版本，保留本轮结果。

实际上一命令、current base/EEF twist、当前mount或短历史都可以成为部署可获得信息；不能一次全部加入再称单因素实验。优先选择能解释现有误差的一类。恢复任务与长程顺序需真实事件评价，不能用持有状态误差替代。

## 6. 本轮成本和停止边界

| 工作 | 新增优化上限 | 新增离线生成 | 原生请求 |
|---|---:|---:|---:|
| D0存量诊断、D1静态几何 | 0 | 0 | 0 |
| Teacher未启动后缀 | 0 | 0个SLA生成 | 剩余≤40，含首段至多8；原总48 |
| M两臂 | 2000更新/64000目标访问 | 原M≤1600；训练内8000另计 | screen48＋条件确认24＋DEV60＋TEST60 |
| 源模块诊断 | 0 | ≤224，列入T/S原1600 | 0 |

不新增接口请求；已通过M两项接口可在身份/有效性保持时复用。若执行边界改变导致资格失效，先报告需要的新验证范围，不能复用不再匹配的旧凭证。所有额度都是条件上限。

T/S训练2000/64000已发生；T24screen已结束，确认0；M8科学请求/206782 tick已发生，接口2/11906 tick另计。所有失败和已发生效应保留。原V6全路线4000更新/128000访问等条件上界没有因本计划重置。

实际资源/owner/截止须执行端读取。已知外层allocation截止为2026-11-19 02:51:23.855488 SGT，不能用本计划自动延期。每service≤12h、campaign≤24h且不越实际allocation；不足时保存已完成端点和未启动后缀。当前纯仿真直接授权及正常工程修复规则沿用，无需每个小修复重复请求GO。未知或健康现役任务不抢占；现役胶囊不热改。

当前计划不发真机命令。真实机器人能力仍需目标本体采集、命令/测量/时钟与TCP标定、控制执行资格和实际任务评价；外部RoboDojo文件的存在不等于这些步骤已经完成。暂停仿真不代表连续实时资格。

## 7. 实施复用与交付

优先复用以下模块，不新建调度器、审批框架或重复账本：

- world_model/mobile_boundary_clock_v1.py、mobile_wbc_reference_v1.py：参考身份和提交证据。
- world_model/multitask_teacher_v6.py、multitask_teacher_episode_v6.py：最小Teacher修复和真实阶段验证。
- scripts/wm02/multitask_teacher_campaign_v6.py：仅未启动后缀的既有有限执行。
- sla_adapter/multitask_data_v6.py、multitask_initialization_v6.py、multitask_training_v6.py：完整数据资格、白名单初始化、匹配训练。
- scripts/wm02/multitask_student_campaign_v6.py：原固定学生面板与门。
- sla_adapter/source_evaluation_v6.py：条件性模块诊断，明确新模型组合身份。

执行端第一交付应包含：D0机制表、每格Teacher覆盖/剩余可能性、具体Teacher差分、下一段原条件与有限成本；之后发布真实Teacher终态和M训练是否启动的理由。若进入M，再交付两个端点、共享初始化逐位对照、24/臂逐条件结果和原门决定。

报告重点是成功完整任务、第一失败机制、真实数据/生成/优化/积分和未决问题。合并数、测试数、脚本预检和服务正常退出不作为能力指标。受影响实现做必要定向验证；只有文档发布时，不把文档检查标成native/GPU资格。

## 8. 证据和审计范围

审阅固定PR409 31db982450bd879a5f88f28fe5009206528e0544、PR407 4b7040d3b9f43bdc0514609d564d91ecff67410e、原计划PR408 ae3d3c0dceade437b0104a150cd0402fef650052；原V5.2终态14886acdf739484cac914578e72736e63cedef62保持。正文已给出逐项永久链接，运行source SHA与报告commit不可互换。

本次发布只新增计划、机读设计、入口与启动prompt；本地仅检查文档、JSON、预算与引用。未取得新的GPU/native/CI结果，未改执行代码或历史结果。wbc_mujoco_minimal.tar.gz附件未能读取；本审阅依据GitHub现有证据，不依赖它；需要将它纳入后续独立WBC复现时再补传。

