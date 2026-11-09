# GR00T + WBC 实验进度独立审核 V2

审核日期：2026-11-09，Asia/Singapore。  
审核对象：Leo-6-maker/go2-arx5-hierarchical-vla。  
冻结阅读版本：执行 PR #371，`8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6`；现计划 PR #370，`c3e47ff2f25894cb28eba807b5ff4c5a7bd5a179`；历史实验 PR #366，`cad80fbc2a9e3f70288b56967f196731612c452d`。

配套：[长程执行计划 V2](../../../docs/handover/2026-11-09/GR00T_WBC_LONG_HORIZON_EXECUTION_V2.md)。本报告只读检查公开仓库的源码、实验合同、实际回执和提交变化；未登录现场、重新运行实验或为未来实验签发通过。源码存在、软件检查通过、现场执行完成、科学目标达到分别判断。

## 1. 审核结论

**项目已经把冻结 WBC 下的新接触任务做成了可重复的 Teacher 与数据生产路径，并在 Core/Hopper 实证打通原 GR00T 动作侧训练、完整恢复和 Core 推理重建。当前仍处在“正式任务数据即将准入”的位置，尚无新任务正式 SFT 或学生完整闭环成绩。**

本轮最需要推动的是现役 TRAIN100 → 原 Processor → 新任务 SFT2000 → Core 模型包 → B0/B1 全 DEV 的实际接续。继续增加同类 FIT4/资格 canary 的收益已经有限；采集性能、H200 吞吐和 GPU 物理扩量可以并行，不能替代第一版目标 policy 的交付。

代码还存在一个明确的长程执行缺口：现有公开入口能运行已经登记的训练 job 和单个 DEV group；没有已发布证据证明“训练终态 → 模型交付 → 所有 DEV 分组汇总 → 科学决定 → 下一训练/数据端点”全链已经自动接续。计划应要求沿既有 owner、Journal、registry 补薄入口并取得真实接续回执，不能把一份新 prompt 当作已经实现的协调器。见第6节。

## 2. 新鲜度与现役状态

以下时间是不同证据的观测时刻，不是本报告发布时的统一现场快照。

| 证据 | 最后记录时间（SGT） | 可确认的状态 |
|---|---|---|
| PR #371 源码/报告发布 | 11月9日 15:48:43 | 新性能候选及测试已提交 |
| GOAL_STATUS / 后50采集 | 15:10:30 | 后50恢复组 ACTIVE；首条推进至7883 ticks；组完成0 |
| H200 scan | 13:24:10 | `707720.hopper-m-02` 为Q，ngpus不足；扫描更新0 |
| 旧采图内部时间复查 | 15:29:35 | 15次capture内部累计18.631s |
| 八条件性能后续流程 | 15:43:34.906 | 等全TRAIN100和Core导出/Processor；新性能积分0 |
| 该性能流程原有限截止 | 21:43:34.906 | 文档发布不延长此期限 |

相对此前阅读版本 `aeefdbc6b0d82591dab6cb33a9e7805d1d95165b`，最新三次提交修改11个文件，集中于性能候选、调用计时、相机生命周期、检查与等待回执；未发布新的后50终态、正式 SFT 或学生 DEV 结果。启动端必须先读更晚现场状态，不能把旧 ACTIVE/Q 照抄为当前状态。[S1] [S2] [S3]

必须继承的现役身份包括：

- TRAIN collector：`dc418f48243bd1939d631d3b82e67213fd747e82`；binary：`82c135c1bf5bb093cac832554b953d0017ec33e35e584865d08ca14327f3aa29`。
- 后50 run：`task-learning-continuous-teacher-20261109-02`；原 scope：`task-learning-pilot1-20261109-04`。
- H200 scan：`707720.hopper-m-02`。
- 已有数据到首seed2000的有限后续流程；已有独立八条件性能等待流程，源码身份 `cdd75715ab37e5e073714c6e9eb8f3c92b4af255`。

这些是接管时的核对线索。健康在途作业继续；UNKNOWN 独立查询，不盲重发；新的长程计划不得生成同 run 的第二个 writer 或第二个物理协调者。

## 3. 已取得的实验进展

### 3.1 接触任务与稳定性：已有实质进展

新任务是独立的 under-hand `GRASP_RED_LIFT20MM_HOLD1S`：40mm、50g红方块，实际物体抬升至少20mm，连续原生抓持保持至少1s，保持速度不超过20mm/s。一次任务有15次真实观察/Teacher决策，每次H40×50ms；共600个消费命令，另有20点终止保持，post-readiness共31s，物理仍为2ms。[S4] [C8]

原第一次20请求资格只执行10次：9次任务成功、1次关节硬安全停止，10次未运行；整批失败保留。诊断定位到跨线程传感向量复制竞争。修复采用49项原子快照交接，在占用原时钟槽前核完整性与同一时刻，未改腿策略权重、控制数学、增益或安全。新的独立20请求资格得到20/20 ready、20/20完整任务成功、硬停止0、技术失败0。旧9次成功没有拼入新20。[S1] [S5]

新4次 under-close 恢复资格也得到4/4：真实未抓稳 → Teacher纠正 → 真正抓持并完成抬升保持。它证明特定欠闭爪恢复，不证明滑移、释放或任意失败恢复；故意干预的H40不作为专家标签。[S6]

评分边界必须保持清楚：新任务 standing score 为BASE位移≤20mm、转角≤5°、足平面位移≤5mm、至少3足支撑。旧SHORT的2mm/0.5°诊断继续保留，仍可能失败。新任务得分通过不等于旧严格短任务也通过，更不等于放宽了原生硬安全。[S4]

### 3.2 数据：50条真实成功，不等于正式TRAIN已全部合格

正式人口为25布局×4独立启动=100请求，前50为普通轨迹，后50为恢复轨迹。前50已CLOSED/exit0，50/50 ready和完整成功，覆盖25布局、硬停止0、技术失败0；后50最新回执尚无完成条目。固定4条DEV Teacher参考4/4完成，单独封装；它们不是学生成绩，也不进入优化。[S1] [S2] [S7]

当前准入实现要求全部100请求终态，至少80条合格轨迹、20成功TRAIN布局、20次真实恢复成功且覆盖10布局，硬停止与技术失败均为0，外加图像、动作、时钟和消费的真实审计。`qualified_training_demonstrations=0` 指整个人口和Processor准入尚未完成；不能解释成没有真实数据。报告中的50/100为进度，已执行成功率是50/50；剩余未运行既不是失败，也不是成功。[S1] [C1]

有效数据量不能按物理tick或采样器重复数膨胀：

- 普通轨迹15个真实观察—H40标签窗口；恢复轨迹排除1个干预chunk后14个。
- 如果100条全合格，最多为50×15+50×14=**1450个不同监督窗口**。它们仍在episode/布局内相关。
- 60000是消费waypoint数，不是60000次高层决策。
- 每episode32次平衡访问产生3200个采样行；首2000更新×batch32=64000次访问，相当于每个不同窗口平均约44.14次，实际关键阶段权重不均。
- TRAIN25只覆盖7mm×7mm局部XY，同一个40mm/50g物体；当前DEV/TEST属于局部未见位置验证，不能据此声称大范围工作区或新物体泛化。[S4] [S8] [C2] [C8]

审核判断：数据生产已越过“是否能获得完整示范”的关键困难；下一风险是恢复标签、统计与目标Processor真实一致，以及小而相关的数据是否足以学到可观察的接触策略。

### 3.3 训练与推理：工程资格通过，目标任务学习尚未开始

| 项目 | 实际证据 | 解释边界 |
|---|---|---|
| Core完整恢复 | 连续4步对独立进程2+2，参数/Adam/scheduler/RNG/cursor精确一致 | 证明训练与恢复路径 |
| Core FIT4 | 200次optimizer调用、199次非零LR；XYZ 2.197→0.165mm | 4条件有限拟合；非接触技能 |
| Hopper原Trainer | FULL_ACTION_SIDE共1,304,913,028参数、537张量；外部adapter0、骨干冻结 | 原动作侧训练，不是只练小外接头 |
| Hopper恢复 | `707196.hopper-m-02` F/exit0，4对2+2全态一致 | 正式新配置仍需自己的恢复/准入 |
| Hopper FIT4 | `707481.hopper-m-02` F/exit0；200调用、6400访问；XYZ 2.202→0.155mm | FIT4权重不得作新任务父 |
| Core普通GPU重建 | 4 probe全部132维H40及物理解码一致；峰值分配13.301GB | 新任务包40相位probe仍待完成 |
| 正式任务训练/学生 | G4–G7为NOT_RUN | 不是学生成功率0%，也没有已证实成功 |

Hopper FIT4墙钟27分39s、训练峰值分配约32.27GB。首正式2000更新保守预测约6.82h、拟8h分配；8个全checkpoint的存储下界约125.28GB。这些是有限诊断和资源预测，不能等同正式训练已发生。[S1] [S9] [S10] [S11]

当前正式配方从原CORR2048父初始化，原masked flow-matching，动作侧全参数，AdamW LR1e-4/WD1e-5、effective batch32；8000步连续日程与warmup200，首科学端点2000，后续4000/8000依据证据决定。先完成当前真实跨站Processor、microbatch扫描及选中配置完整恢复。不能因硬件空闲跳过准入或自动使用旧21窗口长训。[C3]

## 4. 当前推进的性能工作与真实边界

headless独立任务已经完成真实双RGB与完整Teacher任务，但318.802s对原318.350s没有墙钟加速。viewer CPU占比或CPU消耗下降不能替代回合时间收益。[S12]

最新性能候选有两个机制：

1. **事件等待**：在新construction角色的task窗口，用原150ms超时内的证据满足即返回；仍核pause ACK/packet、WBC/TTL、0–25ms观察年龄、原target→solver消费join、arm请求与夹爪消费。准备和preaction保持仍走原等待，不增ENTER/tick或失败重试。
2. **相机缓存**：一回合内复用模型/renderer，逐packet重置MjData、核实际哈希与相机设置、新query身份和原wrist字节，顺序独占并finally关闭；仍是原Mesa EGL渲染。[C4]

相关软件检查189项通过；实际缓存像素一致性、原生加速均尚无结果。已经存在一个有限等待流程：全TRAIN100与Core导出/Processor CLOSED0之后，按 profile/events/cache/combined/combined/cache/events/profile 的固定八条件执行。带cache条件之前先核全部15个真实保存状态的原图/新鲜渲染/缓存渲染PNG和标定字节一致。不能再造同一八条件副本，也不能提前热切TRAIN100。[S3]

最新计时揭示：15次capture内部18.631s；Teacher query边界间隔44.544s；差额25.913s包括采图前身份等处理，不能全称render self-time。748×150ms=112.2s是源码计数的名义满等待，不是已测省时，新路径也不移除所有这些等待。[S3]

单H200扩量已由后续用户指令授权，旧内部scale-review停点不再构成同范围重复审批；但GPU物理实际积分0、GPU批量WBC未qualified。报告核查的Warp后端与MuJoCo3.2.7/elliptic存在兼容缺口；不能静默换friction cone或动力学来声称4096支持。正确首交付是同任务单环境全链资格，再逐级规模化、测端到端吞吐。[S13]

## 5. 必须保留的历史负结果与方法定位

旧C10 B1已完成144条件，6父布局×4目标×3方法×2重启；原生失败与硬停止为0。简单TRAIN平均响应规则21/48成功（43.75%），ARX7/48（14.58%），训练Actor9/48（18.75%）。Actor相对规则的等父布局代价改善为−0.065085mm，层级95%区间[−0.134147,−0.011369]mm，未建立控制增益。这是有模型、有完整评价的负结果，不能说项目一直没有训练过Actor。[H1]

FUSION固定预测验证为1布局、4独立重启：首次安装H16线性模型RMSE0.665669mm，action-blind1.903335mm；主神经模型0.974452mm，blind1.722979mm。说明在该局部域完整动作计划有预测信息，线性基线很强；其他神经seed的结果也必须保留。[H2]

真实参考选择仅完成原计划6条件中的2个，线性选择+X0.5mm，误差由独立DIRECT重启3.596422mm到3.225069mm，但BASE位移43.666645mm并触发任务失败。不是精确状态反事实，也没有安全整体控制收益；其余4条件未运行。1614特征→16×8响应头也不是递归闭合的完整世界状态模型。[H2]

审核判断：WM应后置到“同目标、不同可部署状态确实需要不同动作”的问题；先保留简单反馈和线性模型。只在匹配policy/数据/观察/候选/成本下证明预测增量及真实任务增量，才把WM放进执行链。继续旧数据上堆神经更新不能代替任务适配。

## 6. 代码级缺口与科学风险

### 6.1 已登记入口不等于端到端自主推进

检查结果：

- `task_learning_hopper_sft_v1.py` 执行已登记配置；bootstrap要求真实PBS_SUBMITTED绑定，不负责qsub。
- `task_learning_student_dev_v1.py` 与student batch消费已登记serial group/assignment_slice，未证明全80人口跨scope聚合及后继训练已接通。
- training profile支持2000/4000/8000；>2000读取endpoint_decision描述符，admission验证其绑定。它们不自行计算科学续训结论。
- 已部署有限流程明确只延伸至首seed2000，不能据此宣称之后的全链已完成。[C3] [C5] [C6] [C7]

**优先修补**：沿既有owner/Journal/registry贯通训练终态、完整包、Core40probe、DEV分组、完整聚合、结果绑定决策及下一合法job。每个副作用有独立读回；最小真实接续成功后再称无人值守。部分DEV只能接未执行条件或技术修复，不能据部分好结果晋级或选择模型。

### 6.2 主要科学风险及下一项判别实验

| 风险 | 已有证据与尚缺证据 | 最小有效判别 |
|---|---|---|
| 单帧接触可观测性 | Teacher有stage/lift_origin/native_grasp；学生只有当前双RGB+EEF9+grip1 | 从真实正常/恢复/失败状态检查当前观察歧义；必要时同数据/父模型/曝光的因果短历史臂 |
| 固定动作模板可完成局部任务 | 局部视觉probe0.299mm只证明图像有信息 | 同指令不同位置、同观察不同目标的真实行为；换图仅作机制诊断 |
| 过拟合小且相关的数据 | 最多1450窗口、25 TRAIN布局、64000训练访问 | 按布局保留DEV；阶段误差、首次失败和完整成功共同判断，不只loss |
| B0/B1归因混合 | B0原统计，B1目标统计+新权重 | 主结果称整体target adaptation；必要时另登记0更新C0统计诊断，不阻塞首SFT |
| 暂停辅助混同实时 | 当前2s/H40全执行，查询时仿真暂停 | 先交付暂停研究policy；后继另测端到端时延/年龄/超时与实际非暂停闭环 |
| 失败恢复超出Teacher域 | under-close4/4未证明任意学生状态可纠正 | DAgger先验证真实访问状态上的oracle恢复，再收32episode/40查询批次 |
| TEST复用与统计独立性 | 20布局×3重启；重启不是新布局 | 冻结候选后开TEST；布局聚类区间；暴露后另设新确认分区 |

不能预判上述风险必然造成失败；它们是选择后续实验的候选机制。正式SFT和共同DEV是最有信息量的下一步。

## 7. 后续优先级与可执行出口

| 顺序 | 实际动作 | 端点/失败出口 |
|---|---|---|
| 现在 | 核现役owner、后50、707720、两个后续流程真实状态 | 接续健康工作；不重复派发 |
| 第一主线 | 全100人口审计→Core/Hopper Processor→原父SFT2000 | 交付真正新任务checkpoint与全态恢复；数据不足则新TRAIN区组 |
| 第一策略 | Core包/40probe→共同B0/B1各40条原生DEV | 第一版可评分policy、完整分母与失败阶段 |
| 有证据后 | 同代4000/8000，或新覆盖/短历史/批量DAgger | 每个选择绑定完整DEV和预登记规则；不无限同数据加步 |
| 有可用SFT后 | 合格noise actor/DSRL：真实接口→32transition→2000决策pilot | 奖励和任务共同改善才扩大，冻结SFT/WBC |
| 候选冻结后 | 独立TEST60/方法/seed；后续seed确认 | ≥80%为局部研究目标；区间不足则INCONCLUSIVE |
| 长程扩展 | 工作区/恢复→支撑放置→新物体→可达柜门/书架 | 每个任务独立可达性、数据/观察、split、评分与失败出口 |
| 并行 | 既有八条件性能、合法独立单卡吞吐、单H200物理兼容资格 | 实际收益/最大合格规模；不阻塞第一策略 |

具体配方、后继决策、观察与动作合同、有限job衔接和交付定义见配套计划。新建议阈值不冒充仓库已有协议；在对应新结果读取前冻结。没有内部累计GPU/CPU秒、作业数或总工程预算，不等于可以复活CLOSED、修改健康capsule或绕过站点配额。后续同范围授权已继承，不需逐项重复GO。[S14]

## 8. 本次发布的验证范围

本次变更只新增审核、计划和启动文本。检查内容为证据链接、来源SHA、数字/分母一致性、现役保护、授权继承、计划阶段依赖和GitHub提交读回。没有运行物理、训练、额外资格或CI；文档验证不构成任何实验PASS。

后继执行必须返回实际模型、原生闭环、完整人口和结果绑定决定；只有“代码已写、测试通过、下一步待做”的报告不满足第一策略交付。

## 9. 固定版本证据索引

文中的S编号链接到实际状态与合同，C编号链接到实现，H编号链接到历史实验；全部绑定不可变提交。

[S1]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/reports/WM02/task-learning-core-hopper-20261109/GOAL_STATUS.json
[S2]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/reports/WM02/task-learning-core-hopper-20261109/G3_CONTINUOUS_TEACHER_SECOND50_ACTUAL.json
[S3]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/reports/WM02/task-learning-core-hopper-20261109/COLLECTION_PERFORMANCE_CANDIDATE_V1.json
[S4]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/reports/WM02/task-learning-core-hopper-20261109/G2_TASK_CONTRACT_V7.json
[S5]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/reports/WM02/task-learning-core-hopper-20261109/G2_TEACHER_QUALIFICATION02_ACTUAL.json
[S6]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/reports/WM02/task-learning-core-hopper-20261109/G3_RECOVERY_QUALIFICATION01_ACTUAL.json
[S7]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/reports/WM02/task-learning-core-hopper-20261109/G3_CONTINUOUS_TEACHER_FIRST50_ACTUAL.json
[S8]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/reports/WM02/task-learning-core-hopper-20261109/G3_CONTINUOUS_TEACHER_COLLECTION_CONTRACT_V1.json
[S9]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/reports/WM02/task-learning-core-hopper-20261109/HOPPER_FIT4_ONLY03_ACTUAL.json
[S10]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/reports/WM02/task-learning-core-hopper-20261109/CORE_ORDINARY_FULL_CATEGORY_GPU_ACTUAL.json
[S11]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/reports/WM02/task-learning-core-hopper-20261109/G4_FIRST2000_RESOURCE_FORECAST_V1.json
[S12]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/reports/WM02/task-learning-core-hopper-20261109/HEADLESS_NATIVE_AND_DUAL_RGB_ACTUAL.json
[S13]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/reports/WM02/task-learning-core-hopper-20261109/HOPPER_PARALLEL_SCALE_ACTUAL.json
[S14]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/c3e47ff2f25894cb28eba807b5ff4c5a7bd5a179/docs/handover/2026-11-09/GR00T_TASK_LEARNING_CORE_HOPPER_GOAL_V1.md
[C1]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/src/go2arx5_vla/world_model/task_learning_collection_v1.py
[C2]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/src/go2arx5_vla/world_model/task_learning_native_dataset_v1.py
[C3]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/src/go2arx5_vla/world_model/task_learning_training_profile_v1.py
[C4]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/src/go2arx5_vla/world_model/native_collection_performance_v1.py
[C5]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/scripts/wm02/task_learning_hopper_sft_v1.py
[C6]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/scripts/wm02/task_learning_student_dev_v1.py
[C7]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/src/go2arx5_vla/world_model/task_learning_training_admission_v1.py
[C8]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8f1c7dc0f54e4b93c9a0bfb2a96651beb54b50b6/src/go2arx5_vla/world_model/task_learning_scene_v1.py
[H1]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/cad80fbc2a9e3f70288b56967f196731612c452d/reports/WM02/fusion-night1-20261106/B1_C10_COMPLETE_NEGATIVE_ACTUAL.json
[H2]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/cad80fbc2a9e3f70288b56967f196731612c452d/reports/WM02/fusion-night1-20261106/WINDOW_CLOSEOUT_ACTUAL_V1.json
