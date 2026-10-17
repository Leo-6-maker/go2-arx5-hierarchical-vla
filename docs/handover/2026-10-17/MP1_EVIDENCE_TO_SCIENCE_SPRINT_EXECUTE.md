# MP1 Evidence-to-Science Sprint — GR00T＋WBC 科研主线接续执行卡

日期：2026-10-17（Asia/Singapore）。
状态：**PRIORITY_AND_EXECUTION_HANDOFF_ONLY / NOT_AN_ADDITIONAL_SCIENTIFIC_GO / NOT_AN_EXECUTION_RECEIPT**。
工作区：Leo-6-maker/go2-arx5-hierarchical-vla；现役研究分支 **pi0267/executor-decoupled-recovery**。
本计划审核基线：**e77011b918bee38504d1664e0f5ce0c33480a3de**（提交时间 2026-10-17 13:49 SGT 左右）；执行时必须刷新真实分支 HEAD、Issue #11/#47、原生站点、所有活跃 owner/lease/job/intent，不得把本基线称为持续实时状态。
研究章程：**docs/RESEARCH_CHARTER_TASK_SPACE_MANIPULATION_V1.md**。
资源与执行唯一权威：已被用户采用的 **MP1**，固定 release **c8cdf60e41e31da5796b44a064ef2f22d594a491**；其完整协议 **docs/protocols/PI0294_MAINLINE_ACCELERATION_MP1_20261017.md**、机器卡 **reports/PI0294/design/MAINLINE_ACCELERATION_MP1_20261017.json**、precedence **reports/PI0294/design/MP1_SINGLE_BUDGET_PRECEDENCE_V1.json** 及已登记的 parent run **20261016T191445Z-localpc-mp1** 优先。本卡只重新组织在现有权限内的执行顺序，不追加预算、owner、T0、训练/物理安全权限。并行 E2 是非叠加参考，绝不双启动。

## A. 一句话研究目标与唯一首要验收

**使同一共享 GR00T EEF10 权重在真实连续反馈下安全地完成 T1 抓放和 T2 推送，形成能够区分普通行为克隆、真实纠正数据和执行器可行性的闭环比较；在独立场景确认后尝试冻结 WBC 的目标域组合。**

这不是要求本轮一定证明泛化，更不是用失败回放填充成功率。若预算/截止前只取得部分结果，必须产生可审计的 PARTIAL/NEGATIVE 结论，而不是持续重命名 goal 或扩大单任务训练。

科学成果按四层严格分类：
1. 数据和训练合同有效（源/统计/实际消费/参数更新）。
2. 已见 TRAIN 上有完整自主任务行为（Teacher OFF，原 D0/S1，原评分和真实物理）。
3. 暴露 DEV 与独立 SOURCE_CONFIRM_V1 上的双任务泛化（同一权重、同一语义、预设条件）。
4. 同一高层＋冻结目标 Go2＋ARX5 WBC 的完整物理任务表现。
不得从低层层级自动宣布高一层成功。

## B. 本轮最新事实：不要重复已完成工作

- 旧 E1 **全部 CLOSED**：三次 H200 原生作业已终态，保守 19,675 GPU 秒；不得第四次 E1 PBS、重开旧 lease 或将旧余额滚入 MP1。E1 最终记录 **reports/PI0294/20261016T150228Z-localpc-e1/E1_RUN_SUMMARY.md**。
- M1@1024 在两条预先指定的 **seen TRAIN T1** 布局上自主完整 32 秒 SUCCESS；T2 两条在 17.6/18.4s 被原 S1 拒绝。M1 的有效结果不重跑；尚无独立 DEV 成功或 Go2/WBC 成功。
- 旧 M2-BC / M2-CORR **各真实完成 512 updates/16,384 samples**；BC/CORR 的完整自主评价因 20s 文件 RPC、迟到响应与 poisoned channel 未得到有效面板。不得把工程不完整当成 0/4 模型失败。
- 旧 M2 的实际消费不匹配：BC 的 T1/T2=8192/8192，CORR=12288/4096。因此旧 M2 无法单独估计纠正数据的因果任务收益。
- C0=833 H40 窗口/2 个独立 E0 学生 episode；修复后的 C1 新增 832 窗口/2 个 M1 学生 episode，合计 1665 窗口/4 个 student parent。绝不可把切窗数当成独立纠正状态数。C1 未进入旧 E1 M2。
- 2026-10-17 最新 Issue #47 回报：默认 pytest 827 PASS/14 SKIP，Core/Hopper 各 96 次合成同主机文件诊断完成、未复现 20s 停顿；新分阶段 RPC tracing 代码已准备，**尚无跨节点完整闭环修复收据**。不要再把 CPU smoke 当作 P0 完成。
- 当前现役代码新增 **scripts/pi0294/mp1/recovery_config.py** 与 **recovery_neural.py** 等既有 M2 模型评估恢复支持：固定原 checkpoint、0 更新、在 MP1 内子账执行。**代码已存在 != 原生运行已成功**。
- Vanda 原生 CPU build 与旧 M2048 保存帧 A40 推理已完成；EGL/双摄/目标 WBC 实际 startup、真实参考安装/ACK及完整目标任务仍未通过。TF-R3 原生安全 STOP 保持。

## C. 工作顺序：先取得能区分科学假设的完整数据

### P0 — 跨机中继真实资格（所有新增行为比较的实际硬门）

Owner：唯一现有 local-PC coordinator；CoreLab 物理、Hopper 神经均只是受其管理的分工，不产生第二 owner。

1. 刷新站点账户、原生作业、RUN_MANIFEST、lease、registered worker、pending query/ACK、CPU/GPU/物理保留，以及工作分支与已运行 capsule 的不可变身份。若检测到活跃健康任务，等待其合法终态/正常边界；不热更其数值或二次 dispatch。
2. 保留已通过的 CPU 同主机文件诊断结论，下一步在**合法 MP1 子 run** 中实测完整运输/环境交互。沿用现有 RPC phase tracing（guard / invoke / accounting / emit），单独统计模型 forward、跨机写盘/可见性、查询、S1、物理积分、RGB、ACK、checkpoint/ledger sync、等待和长尾。
3. 保留原 20s RPC、60s action、900s episode、32s 模拟任务和 20Hz×prefix8（500Hz 物理）合同；不改 watchdog 使失败变 PASS。不缓存/重复使用不同 query_uuid 的迟到答复。不自动重发任何 unknown-effect 操作。
4. 在每个决策边界保留 source state、raw H40、S1 decision、installed prefix8、ACK、actual achieved EEF/夹爪/物体状态及每阶段耗时；不得把 25s 搬运或部分接触记为完整成功。
5. 退出门：有至少一项**实际合法完整 32s 的模型端到端链路**且可复核身份、计时、物理、S1、原评分/视频；或留下明确的工程不可达/超时证据并停止在该阻断上继续烧录等价评价。不要再以 CPU 准备状态宣布 P0 PASS。
6. 如果出现原生安全停止、身份/事务不一致或 uncertain effect，立即保存、停止相应后续物理；跨节点/跨 host 不得绕过同一已确认安全 STOP。

交付：**MP1_ACTUAL_RPC_PHASE_TIMING.json**, **MP1_REAL_RELAY_QUALIFICATION.json**, 失败点、受影响 condition、已收费次数与有效性分类。新文件名是预期产物，不代表已存在。

### P1 — 既有 E1 两份 M2 权重的固定行为恢复（零新训练）

现有代码已将 M2-BC 与 M2-CORR 固定为真实 E1 checkpoint manifest：
BC **edc0f390776056f9b22cd07d152c8fd5e273556f1d242f8e4f220b3034f9ffe6**；
CORR **5275bb0c188a1d2148418d0cd7593dd712d2563e74b1907148563e5582a01f9a**。

1. 独立 MP1 recovery child，采用 **MP1_E1_EVAL_RECOVERY_STAGE_V1** 身份，必须通过真实合法 run/lease/站点预算和当前代码 CI/静态/数值资格；不把原 E1 内缺口重新标记为已完成。
2. 只读加载上述两份现有 512 权重；**optimizer_updates=0**、不重新采样训练、不读取新 DEV 标签、不新增 C1 到旧 M2。
3. 原四个 TRAIN 场景任务卡 × 两模型，共 **8 条固定评价 condition**。当前源码 recovery_config 限制 <=16 main starts（包含允许的工程 replacement）、<=256,000 live/recovery 积分、<=256,000 shadow 积分、<=1,280 model queries、<=6 H200 GPUh、<=2 MP1 recovery PBS、<=0 更新；**这些是该子 stage 的上限，仍需以 parent MP1 真正剩余资源和站点更小上限为准**。任何生产 dispatch 前检查 stage/card/代码 SHA 不变。
4. 只允许原明确工程不完整/NOT_RUN 条件进入新的独立恢复面板；M1 四条有效结果不得重跑。新恢复 attempt 有新 lineage，旧失败与全部费用保持。
5. 每条完整判断：SUCCESS / VALID_TASK_FAIL（含原 S1 拒绝）/ ENGINEERING_INCOMPLETE / NOT_RUN / ACTUAL_SAFETY_FAIL，并把工程缺失留在分母账中单列。
6. 即便两模型得到完整可比较行为，仍在结论中说明旧 BC 与 CORR 训练的 **T1/T2 曝光混淆**，不能用这组面板直接宣称纠正数据因果收益。

如果远程中继再次导致不可诊断的共同工程中断，**不自动重复、也不以训练新权重绕开工程问题**；直接封存 failure taxonomy，把主线资源转向不依赖该 relay 的 T2 只读诊断和 Vanda 独立运行资格。

交付：**MP1_E1_M2_RECOVERY_BEHAVIOR_MATRIX.json**、逐尝试证据索引、0 更新验收、具体失败耗时分解、与 E1 原 12 条条件矩阵的关系、各站点实际成本。

### P2 — T2 科学归因（可与 P0/P1 的合法 CPU 工作并行）

不重新跑已经科学可判定的 M1 T2 两条物理负例。从 17.6s / 18.4s 的固定保存包中复算：
- 首个被 S1 标记的非法 geom-pair、shadow tick、几何类别和侵入/接触阈值，保留所有重叠 issue；
- 被拒的 raw H40 与实际 achieved state、前后已 ACK prefix、姿态/XYZ/夹爪及时间坐标；
- 被拒前物体移动 13.28cm / 14.61cm 的任务阶段、是否已满足推送目标谓词、是否处于退出/撤离阶段；
- **同一合法状态**的专家与仅作归因的 XYZ / SO3 / gripper 对照；不能冒充 GR00T 独立成功，不能拿 E0 的旧 geom pair 代替 M1 的未知值；
- 原 TRAIN126 中 30 条 T2 episode 的独立 scene/事件/接触和完整窗口覆盖，区分“任务曝光已均衡”与“独立几何覆盖足够”。

分流：若证明接触几何/特定阶段覆盖缺失，按 MP1 已有资格考虑 **S-EVENT 单次、固定4-episode容量诊断**；若是共同的参考安装或解码错误，单独绑定工程修复、旧结果原样保存；若专家参考在冻结规则下也不能通过，不通过放宽 S1 追成绩。T2 Recovery Oracle 未合格前不得制造 T2 正标签。

交付：**M1_T2_FIRST_REJECTION_ATTRIBUTION.json**＋源/目标/动作/碰撞的保存时序图；每个解释标注“确认/支持/未排除”。

### P3 — 匹配训练曝光的共享 Source Pair（本轮主要模型比较）

此项是本轮主要科学新信息，必须严格采用已经冻结的 MP1 配方，而非临时 loss 搜索：

- 共同初始化：已审计 **M1@1024** 权重；每臂全新 optimizer（不是彼此 checkpoint 续训）。
- 两臂：**S-BC2048**（只原 v3.0 TRAIN126）及 **S-CORR2048**（原 TRAIN126 + 固定合格 C1）。
- 训练：各 2048 updates；effective batch32 = microbatch2×accum16；seed 20261017；native Flow Matching、原 action/EEF10/H40、原 stats、VLM 冻结、仅原项目已许可 projector/DiT 训练；同一身份的训练数值与 canonical inference 精度必须明确区分，checkpoint identity 不得只看 weights。
- sampler：先 T1/T2 各 50%；在 T1 内原数据/纠正各 50%，则 CORR 整体 original-T1 25% + corrected-T1 25% + original-T2 50%。**不得让只存在 T1 的纠正数据吞掉 T2 曝光**。
- 原 TRAIN126 包含 96 T1 / 30 T2 独立来源 episode；T1/T2 平衡不意味着 T2 轨迹分布充足。原 stats 不用 DEV refit；v3.1 本主比较不热插。
- 实际消费强检验：每臂 **65,536** 样本访问、每个 batch/task/origin 的顺序、episode、事件阶段与 source_parent lineage；检查逐步更新、梯度、trainable/冻结、resume cursor、RNG/scheduler。
- 模型/评价：按固定 512/2048 checkpoint 和 TRAIN4 / 暴露 DEV8 面板执行，遵守 **scripts/pi0294/mp1/plans.py** 规定。不得根据密集 checkpoint 或曝光 DEV 成绩反复重配；最多一次明确工程 replacement，不重跑有效任务失败。
- 统计：每任务真实完整任务成功率、S1拒绝、工程缺失、阶段成功、接触/物体结果、教师动作与模型物理偏差；按独立 episode/scene 分层，呈现 worst-task、macro-success、场景置信区间和样本不足限制。不把重叠窗口当独立试验。

若 P0 尚不能支撑完整行为评价，训练后的 loss 降低和 checkpoint 保存只能记 **TRAIN_COMPLETE_EVALUATION_BLOCKED**，不能提升候选或浪费剩余预算不停加训。若 P2 证据支持特定事件覆盖不足，只使用已预设的单变量可选 S-EVENT 路线；否则直接收口为有解释力的负结果。

交付：**MATCHED_SOURCE_PAIR_ACTUAL_EXPOSURE.json**、训练原生收据、同一闭环判据下的 **S_BC_VS_S_CORR_BEHAVIOR.json**、原 C1 是否真实消费及 anchor/expert 后续段分解。

### P4 — Vanda/冻结 WBC 的目标域组合（独立 qualification lane）

不要求 T2 先完美才允许做已验证 T1 的目标域 source-only 资格，但任何原 TF-R3 安全 STOP 均不得因本计划或新预算解除。

先复用已经 F/0 的旧 A40 CUDA/保存帧推理与新 CPU native build，不重复把同一 smoke 当成新成果；在独立合法 allocation 与固定工件身份下依次确认：
1. 原生 EGL、双路真实 camera、source obs/action processor、模型数值身份及内存/时间；
2. 冻结 WBC/腿 policy/plant/PD/安全的真实启动、时钟和机械限制；
3. raw EEF10 → 原合法 S1 → installed reference → 实际 achieved EEF、物体/BASE 状态、真实 ACK → 下一 observation 的连续管道；
4. 专家正确任务参考与同一冻结 VLA 参考分别运行以区分 WBC 能力和高层输出问题；在共同可行目标上比较，真实失败不被过滤；
5. 如果目标已有真实安全成功 T1，再在授权 scope 内申请/执行 source-only 到少样本 target-adaptation 的固定比较；缺合法 target 成功示范则记 DATA_UNAVAILABLE，不补零造 BASE13，不把 0/5/20/50 视作既成数据。
6. 独立披露 source fixed-base ARX5 成功、target Go2＋ARX5 WBC 物理成功及真实机器人部署；三者不能互代。

交付：**VANDA_NATIVE_RUNTIME_QUALIFICATION.json**、**FROZEN_WBC_TARGET_SOURCE_ONLY.json**、raw/installed/achieved/物体结果交叉表、原生安全与所有 target 尝试的完整分母。若 target 因安全或环境受阻，源域科学工作继续而不是改 WBC 寻求捷径。

### P5 — 选择、确认与实际 Recovery / DAgger（有条件）

1. 在可重复的开发面板上按原 MP1 规则选定**同一份共享 checkpoint**：先 worst-task SUCCESS、再 macro-success、再较低复杂性；历史 checkpoint 的有效负面证据保留。
2. SOURCE_CONFIRM_V1 必须在冻结候选和规则后只使用一次：10 scene group × 2 task × 2 noise = **40 条/候选**。目标每任务 ≥14/20 是原 MP1 的研发里程碑，**不是当前已达标、不代表部署可靠性或可据此偷调的阈值**。
3. 未见 scene/object/episode 与训练 lineage 应隔离；旧暴露 DEV 不冒充独立确认。若双任务未达要求，以 **PARTIAL_SOURCE_SINGLE_TASK** 或明确失败总结，不写 MULTITASK_GENERALIZATION_PASS。
4. 只有实际 Recovery Oracle 合格的学生状态可以产生纠正监督。最多沿已批准 MP1 两轮，每轮最多 12 条 student collection episode（每任务最多 6），与同源 BC512/CORR512 匹配比较；评价仍 Teacher OFF、S1 ON。T2 未取得 Oracle 资格时不可由 T1 结果外推。
5. FlowDAgger、BASE13、全 VLM 解冻、换新 backbone、RL、另起世界模型均非本轮关键路径；不能为赶进度在当前身份里引入。

交付：候选冻结文件、**SOURCE_CONFIRM_V1.json**、每任务/scene 成功和工程缺失、独立数据资格表、下一轮研究分流建议。

## D. 资源、截止与停止规则（不重置、不叠加）

MP1 唯一 parent T0 **2026-10-17 03:14:45 SGT**；禁止新 submission/candidate 从 **2026-10-23 03:14:45 SGT**；禁止新增实验/训练 block 从 **2026-10-23 19:14:45 SGT**；硬关闭 **2026-10-24 03:14:45 SGT**。旧时钟不随恢复/重建而延长。

父 MP1 上限：新增 H200 48 GPUh、A40 32 GPUh、每站最多12新 GPU PBS、总新增24576 optimizer updates、source main/recovery 最多400启动、target 最多320启动、总64,000,000物理积分（含shadow/失败等），以及机器卡中各站点 CPU、空间、网络上限。E2 不可相加；已核销 E1/Vanda 不二次记账。子 stage 的更紧上限优先。保留规定的末端评估与收尾容量；若站点许可/quota 更小则更小者生效。

硬停止包括：原生真实安全 STOP、WBC/plant/限制变化、无效模型/processor/统计/归一化/时间身份、未知 effect 盲重发、越权读取 DEV/confirmation、伪造训练或标签、资源或截止耗尽。同一工程故障若重复而不能证明根因/修复，则停止在该支路消耗新的行为条件，提交技术终局；不将 ENGINEERING_INCOMPLETE 合并成策略 0% 或 100%。

## E. Codex 唯一 owner 的具体执行协议（直接可接手）

1. 从现役统一 **START_HERE.md / AGENTS.md / 2026-10-13 workspace enforcement** 进入，刷新 **Issue #11/#47**、现役 HEAD、所有 run manifests/native jobs/leases、MP1 累计实际 used/reserved、HF 固定 revision 和源代码 capsule，检查工作树，保留他人进行中修改，不 force/reset、不开第二 owner 或 parallel E2。
2. 输出一次简明 **GOAL_ADOPTION_READBACK**：有哪些条件已完成、目前哪个子 stage 能合法开始、哪个受阻、精确资源和未知 intent；仅在已获 MP1 的合法边界派发。
3. 按 **P0 → P1 → P3** 形成首条关键链；**P2** 仅使用合格保存证据可并行 CPU 执行；**P4** 在独立合法 target 资格边界推进，不等源 T2 完美；**P5** 必须在对应数据/模型资格后启动。
4. 每次工程修复只修改被证据定位的运输/采集/调度边界，在新 source SHA 与新子 run 上自测；不热更正在执行的模型、SafetyGate、WBC/PD/plant/计时/评分；保留失败原始 run、收据与费用。
5. 产生可核验 native 收据、模型版本、实际 consumed trace、任务时序、raw hash/视频与完整条件表。所有报告显式标识 **NOT_RUN / ENGINEERING_INCOMPLETE / VALID_TASK_FAIL / SUCCESS / NATIVE_SAFETY_STOP**。
6. 现役工作时按 MP1 原规定每不超过6个活跃小时更新已有 Issue #11 的关键阶段状态；同一工程问题在 Issue #47 指向具体收据即可，不创建新的竞争 research-control issue。中间无结果时如实写 blocker，不自称新 GO/完成。
7. 截止前给出单份 **MP1_FINAL_SCIENCE_MATRIX.md + MP1_RESOURCE_FINAL.json**：每个科学 claim 的强/弱/缺失证据、每任务独立成功分母、实际 GPU/CPU/物理/网络/存储核账、所选模型状态、目标 WBC 安全状态、所有残留问题与下一阶段唯一最有信息量的实验。旧 selected/deployment 保持 NONE，除非另有适用的明确 qualification 决议。

## F. 不在本轮偷换成的新研究主张

- **M1 的已见 T1 2/2** 不等于 T1 泛化，也不等于多任务能力。
- **M2 的训练和非初始学生状态误差改善** 不等于真实纠正数据已经提高自主任务成功率。
- **同一模型接口输出 EEF10** 不等于相同高层权重能跨本体泛化。
- **GR00T 的官方预训练权重被继承** 不等于已用公平初始化对照证明预训练收益；原始官方初始化 vs 历史继承初始化应列为后续独立对照，不在固定 MP1 配方中偷偷更换。
- **源固定底座完成 T1** 不等于冻结 Go2 WBC、第二本体或真机完成操作。
- 师兄 WBC 的跨轨迹分布泛化属于独立执行侧资格与未来系统主张；上层团队不在 MP1 下重新训练或改它，也不把源数据预先按目标 WBC 成功与否筛选。

## G. 关键固定证据与源码

- E1 完整终局：reports/PI0294/20261016T150228Z-localpc-e1/E1_RUN_SUMMARY.md
- E1 原始 12 条矩阵：reports/PI0294/20261016T150228Z-localpc-e1/MODEL_BEHAVIOR_MATRIX.json
- E1 实际更新和 1:1/3:1 曝光：reports/PI0294/20261016T150228Z-localpc-e1/LAST_NATIVE_TERMINAL_AND_ACTUAL_M2_EXPOSURE.json
- M1 T1 实际结果：reports/PI0294/20261016T150228Z-localpc-e1/M1_1024_COMPLETE_TRAIN4_AND_T1_AUDITS.json
- M1 T2 已保存拒绝证据：reports/PI0294/20261016T150228Z-localpc-e1/M1_T2_SAVED_REJECTION_AND_OBJECT_PROGRESS.json
- 近期双机 CPU 文件诊断：reports/PI0294/20261016T191445Z-localpc-mp1/ACTUAL_RELAY_CPU_DIAGNOSTIC_01.json
- 恢复评估限制：scripts/pi0294/mp1/recovery_config.py 与 recovery_neural.py
- 共享任务采样：scripts/pi0294/mp1/dataset.py 与 train.py
- MP1 真正执行合同仍以冻结 protocol/card、有效原生身份、lease/owner/run manifest 为准；本文不取代。

**本文件由 GitHub 连接直接发布的是科学优先级与 Codex 接续卡；本次发布没有 SSH、未启动 PBS、GPU、仿真物理或模型，也没有对任何旧结果作成功追认。**
