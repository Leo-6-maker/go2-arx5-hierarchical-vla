# WM02 长程自主推进委托 V1：找清根因、搭好全身仿真环境、验证上层 WM→Actor 价值

**执行对象：Codex；任务提出者：项目用户；创建：2026-11-04（SGT）。** 本文是**用户项目级扩展委托**，不是目标原生平台的权限凭证；不能取代原 owner/registry/lease/native-safety admission。Codex 应在本委托可执行范围内连续推进多阶段，做到明确 PASS、可复现 FAIL 或 OWNER_BLOCKED，**不应每完成一轮200步就停止等待新指示**。禁止声称未运行的动态项目已完成。

## 0. 基线、待解决的问题、绝不可混淆的证据

- GitHub：\`Leo-6-maker/go2-arx5-hierarchical-vla\`；执行基线为未合并草稿 **PR #352 head \`76bb8115c3b588b70fd55bdde13f10926c60bce7\`**。上游 #344–#351 为叠加草稿，不假设 main 已合并或现场部署。必须保留现有源代码、checkpoint、原始原件、错误和历史审计的字节身份；始终独立 checkout、独立 immutable capsule 和新 run ID。
- 原生目标是 \`GO2_ARX5_MUJOCO_OCS2\`：**自由底座 Go2＋ARX5，冻结 WBC／腿 Policy＋ROS2 Jazzy／OCS2／MuJoCo**；名义物理步长2ms、上层原子步50ms、Actor每8原子步决策、完整H40 EEF参考。固定底座S域动态 ≠ 目标T域动态；S的完整restore ≠ MPC/WBC/receiver恢复。
- **已知首要目标阻断**：原 S1/T2 隔离仍有效，旧lease CLOSED；尚无原owner有效新init-only动态范围。目标冷启动0、2s/5s保持0、实际STOP资格0、目标WM转移0。新R1同步+observer停止锁存+trace的候选已编译/链接/载入校验，不等于原生验收。
- **目标已发现异常**：旧T2 step4首个控制差异，step4–23力矩可能保持旧值，step24关节限位越界；原POS/VEL两队列不同版本消费风险未证明为唯一根因；旧STOP后3次多余物理积分，候选仅在假内核验证无后续积分。不得将“编译成功”写为“WBC修复”。
- **S域最新结果**：M1、M2、M3各为18,905,387参数的50维S域模型。M3 DEV8配对物体差H1/H8/H16预测0.216/7.063/6.926mm、真实0.200/7.952/7.906mm，逐父动作差门槛PASS；但M3 H1绝对物体RMSE1.947mm > M1 1.151mm；M3 H16 preclose/release抓持Brier .6921/.7816，M1 .0734/.0548；M3 H16绝对TCP RMSE13.676mm。**开发门槛FAIL**。
- M3 H1物体误差MSE **99.9844%为不同候选共享的common-mode bias**，pairwise差分损失不能纠正；M2 H1已在第一步偏，H16 FREE2.534mm而单步诊断.427mm，不能只归于展开误差。原只用FIT0–3×5候选拟合的局部OLS在DEV8 H8物体端点池化RMSE约54.31μm，已是该狭窄稳定抓持源域的重要强基线，不能忽略。
- 旧ONCE_HOLDOUT因之前的评价/重评分已被看过；未来禁止将其再当“全新盲测”。原训练 \`c12293bc877758254a14d19ec112f147fe19c93ffacf4c8399ae8ebfcb42c808\` 不修改，TEST未授权读取；训练从旧FIT24/DEV8始，任何新场景必须**先冻结scene+parent+episode分组**，所有共同前缀分叉在一个split。

报告源：\`reports/WM02/temporal-response/{EXECUTION_REPORT.md,DEV_GATES.json,ERROR_DECOMPOSITION.json,TARGET_STATUS.json,TEMPORAL_CURVES.json}\`。先验证hash和本PR head，再使用其他信息。

## 1. 授权矩阵——增大自主性，但边界真实有效

### A. 用户在本委托中直接授权 Codex 自主推进（无需每个子步骤再询问）
1. 在独立分支、用户项目私有run目录中**调查、增删新代码、编译候选、编写测试、修复离线逻辑和非冻结用户层接口、构建隔离CPU/GPU/PC可视化环境**；可持续产出多个细粒度草稿PR、引用上游实际SHA、提交证据与说明；禁止热改现役capsule及旧提交。
2. 读取项目已经授权的**非TEST**私有数据／原始故障日志，校验hash、导出新的只读派生特征与局部回放；保留原件不覆盖。CPU合成、无动力学的\`mj_forward\`、有限编译、ABI/pose/state/ACK合同测试可自主执行。
3. 在**已有用户本人有效资源owner/lease/队列许可和资源余量条件满足时**，继续执行有界 **S域** 真实物理诊断和独立神经GPU实验；允许对算法进行机制性版本对照、从失败自动派生新的修复候选（必须写明新假设、冻结参数与唯一变因）。
4. 对自己新建、且确属本任务的worker/lease，在原Journal/onceIO/owner锁规则下合法停止、读取终态并关闭；不改变其他项目/账户的作业，不重启已CLOSED旧队列。
5. 提前实现PC **只读3D重放+可选实时状态订阅器**、自带许可检测与隐私保护；GUI永不作为目标动态许可的一部分。

### B. 只有原站点/本体owner新有效范围才可自动执行
- 原生**T域**MuJoCo/WBC/OCS2/receiver的模型构造、冷启动、物理积分、真正的STOP、2s/5s保持、EEF/夹爪/接触/采集：必须由原\`EXISTING_LOCAL_PC\`或其他有效现场owner在**其原有profile、registry、lease及安全机制内**绑定新有限scope。**此聊天授权、Codex指令、GitHub草稿或GPU空闲，不是准入。** 用户级委托允许 Codex 准备并提交完整的scope申请材料、检查新scope；不允许用metadata=true或新脚本暗中绕过。
- 获准也必须阶段匹配：只准init-only时不得偷偷进入EEF/抓放；正常顺序采集的授权不代表可任意midpoint恢复/同状态反事实。

### C. 仍需单独明确的新批准（不包含在本委托）
- 改冻结WBC/腿策略权重、MPC求解器核心、增益、物理模型惯量/碰撞/摩擦、机械限位、安全阈值、原隔离机制；执行真实硬件；全局ROS/CUDA/驱动/系统包升级；杀其他用户任务；任何删除旧artifact/隐藏负结果；大规模A40/H200/PBS长期训练；超出下述预算。若根因指向冻结控制器，先建**独立非部署候选**及迁移评估方案，提交实证后请求扩权。

### D. 任务级资源授权上限（**仅在原资源管理机制允许时**）
- **累计新研究预算建议上限**：独占GPU实际占用 **≤8 GPU小时**；CPU进程合计 **≤80 CPU核小时**；新私有产物 **≤60 GiB**；新增S域受控50ms真实转移 **≤20,000**，只为新盲测/失效辨识或强基线必要性服务。所有成功、失败、探针与被拒绝开销计入；未开工范围不预留空闲资源。
- **单个GPU worker** 优先沿用已通过owner约束：4 CPU、≤10GiB RAM、4GiB PyTorch allocator、固定UUID、\`Restart=no\`、≤2400秒、同一项目owner锁、开始前live headroom确认；私有缓存，不改原训练venv，不影响其它用户。CPU/S域worker不得超过现有允许cpu/内存/并行上限（当前证据是4进程测试，而非自动授权64进程）。
- 新科学神经实验最多**3个预登记机制候选×200主更新**（各加最多1次严格续训探针），在开发数据上证明必要且改进后，最多再**1个候选延长至800主更新**；无新独立验证之前**禁止5000/百万步、禁止正式Actor优化**。拒绝/失败不重启同一immutable capsule，必须新身份且预算仍计。
- 每个有界实验都包含时限、停止文件、lease、实际命令及原生安全检查；资源冲突时可延期进入其它已授权离线工作，但不无限循环重试。

## 2. 总目标与自主执行状态机

**最终完整交付**：A) 可在PC实时或录播查看完整自由底座Go2+ARX5场景的**安全、只读**Viewer；B) 可复现且在原站点合规的目标仿真初始化与5秒安全站立（如果owner有效范围可得）；C) 原生实际ACK/消费、腿Policy/PD/积分时序、限位/STOP的高率证据；D) 对接的目标真实顺序Replay合同；E) 有清晰增量价值的分阶段S域WM/控制基线；目标WBC稳定后才训练目标域WM并评估Actor。

采用\`READY→RUNNING→PASS | FAIL_ACTIONABLE | OWNER_BLOCKED | BUDGET_EXHAUSTED\`阶段状态；若一阶段FAIL，**先在该阶段找可观测根因，改变一项机制，执行新的有界对照**，而不是复制上轮的200步；若T域\`OWNER_BLOCKED\`，不要空等、不要反复跑exit78，继续所有离线可执行Track B/C，最后交付明确owner决策包。每阶段保留原文档SHA、实际测试、成本、正负结果。**失败不是必须立刻终止整个项目；安全／未授权必须立即阻止该分支的动态动作。**

## 3. Track A：整机原生环境与WBC初始化根因（最高工程优先级）

### A0：本地/ Core read-only环境完全落地，无目标积分
- 复核原冻结23个目标核心/初始化文件与当前候选20项bytes SHA，站点现有Python3.12/ROS2 Jazzy/MuJoCo3.2.7/OCS2/rl-sdk依赖、ABI\`nq34/nv32/nu20\`、两个外层库的**实际dlsym链**。冻结的基线候选\`rl_sim_mujoco\` sha256 \`7828f9f29e174ac7bf9e82edd897de577a2d365ab22e6dec438f0565b09ff488\`；选用\`task_actuation_r1.so → observer.so → frozen libmujoco.so.3.2.7\`＋唯一trace。若要变更，必须新的受控候选SHA，不改旧包。
- 对T1/T2原始日志做**时钟因果时间线**：物理step0–27、前两版Policy推理、观测/历史、Policy版本发布、独立pos/vel消费、PD实际频率/目标、最终ctrl、first limit、STOP。原trace稀疏不能证明PD没更新，应标\`UNKNOWN\`；枚举并预先建立可区分证据的假设：推理阻塞PD、pos/vel版本错配、历史/输入版本错配、初始化模式切换、控制器与物理步长不一致、模型动力学/接触。
- 记录器采用现有有限队列、环形内存和异步写盘，审核丢失计数、事件时间和原生IO副作用；缺证据就先完善**仅记录的候选**及无物理单测，禁止为取得日志而松限位/安全。
- 交付\`ENV_NATIVE_MATRIX\`、\`T1_T2_0_27_TIMELINE\`、\`FIRST_CAUSAL_DIVERGENCE\`（有证据的结论与仍待动态验证项分列）、\`TARGET_SCOPE_REQUEST\`（列明预期具体owner/机器/源码/二进制SHA/所需CPU与有限tick/STOP的原安全机制、风险及回滚）。

### A1：现场原owner实际签发有限init-only范围后才能运行
- **仅单个目标env，无Actor/GR00T，无抓取/底座主动运动，无更改冻结安全**。新独立run/lease；两次冷启动各至多50个2ms tick（0.1s，先逐次验收），均PASS才开展2s（1000tick）及5s（2500tick）站立；总预定上限**3600物理tick**，任一安全错误提前结束且不再自动重试同阶段；owner的更严格限额优先。
- 必须在真实运行验证：原关节/滑轨范围所有tick均合法；Policy输入/历史/结束与实际消费版本可追踪，pos/vel版本一致或有明确合法异步原因；PD有明确更新时序；接触/ROS/仿真时钟映射合法；trace dropped=0且不挤压原控制；原STOP触发后实际积分0（若未触发则\`NOT_TESTED\`，不能记PASS）。
- 若失败：保存**首次**真实违规前后的trace、SHA和最小复现；在**未获另一个新有效窗口之前**只做CPU时间线分析/替代机制设计。不要在已违规目标上反复试错。
- 若通过：经相应新增scope批准才进入A2的**单次**小EEF轨迹、夹爪、受支撑物体接触和40行参考实际安装/ACK/消耗验证。局部任务通过不代表完整T1/行走通过。

### A3：目标数据落地（在A2许可通过后）
- 先从正常安全初态顺序采集真实目标\`(state,action,installed/consumed reference,contact,event,terminal,clock)\`；明确时钟、坐标、全身安全及表面接触，\`TRAIN\`新layout按scene/parent分层。**不能把S域缺BASE字段补零冒充目标；不能把qpos/qvel快照当完整控制状态恢复。** 尚无完整midpoint MPC/WBC/receiver快照时可做合法**顺序**采集，但禁止同状态分叉及迁移资格宣称。
- 首批T数据以**合法有限序列**和数据合同为主，采用当前owner允许的最小批量；合格再开始新的目标域WM拟合/模型比较。WBC未来改变，必须版本化\`WBC_identity\`与对应的\`WM\`；迁移验收看TCP/BASE/接触、H1/H8/H16和Actor排序，不能假设旧S checkpoint泛化。

## 4. Track B：S域WM本质失效、可靠基线与新模型（无需T动态）

### B0：0更新、0新物理的原因调查（必须先完成）
1. 将M1/M2/M3在FIT24/DEV8的**每一阶段**与H1..16逐时刻对齐：真实／预测\`TCP\`、object、relative object-to-TCP（含SO3）、grasp/left/right/contact/terminal。分别显示 FREE 与 \`TEACHER_FORCED_PRIVILEGED\`，后者绝不用于可部署输入；第二块原生参考从**预测TCP**重锚。
2. 分解端点/多步误差为共同偏差（across candidates）和动作差，验证当前 M3 H1共同偏差MSE 99.9844%、事件Brier退化；做stage-balanced loss/gradient冲突及预测事件概率时序对照、起点当前事件标签可观测性测试，区分**状态锚点不准／预测输入错误／mode switch／event head竞争／长程累积**。
3. 将20个FIT训练样本OLS、状态持恒、恒速、**真实未来TCP的刚性 oracle（明确特权）**与基于预测TCP的刚性baseline隔离；只在真实持续抓持样本用SE3刚性近似，不将其用在释放/接触建立。
4. 对物理单位、完整H40、world/body相对参考、group/field role、mask/terminal、第二决策时钟、batch/候选排列、原float64 raw动作身份做严格回归；源码\`neural.py\`仍有V1合成Actor和zero tracking-residual占位，不能拿它作为目标合格Actor。
5. 输出每个假设的**可否证预测**，优先选最小改动模型。连续三轮未改善开发表现就停止黑箱调参，公布机制限制。

### B1：优先最小模型/损失机制，而不是直接扩训练
候选必须提前冻结在\`experiments/wm02/\`并绑定同一FIT、数据SHA/seed、原模型初始化、等效batch、评价代码和输出身份，**一次只变一项**，按下列优先序（最多3个200更新）：
- **M4-A（首选）**：在M1相对参考上保留真实单步/绝对状态对齐，增加或重标定持续时间的**绝对TCP、object及事件损失**，防止共同偏移被配对差相消；通过对比完整loss/梯度确认新的联合损失未吞掉preclose/release事件。先只使用原FIT。
- **M4-B（若A无法达标）**：条件结构化模型：\`M_execution(s, H40, WBC_id)\`预测实际TCP／base（S域无base时仅TCP）；\`M_interaction(TCP_actual/estimated, gripper, contact_mode, object)\`预测接触/物体。稳定抓持可用OLS+原生SE3小残差作为强结构先验；接触建立/释放另设模式分类/转移，不把oracle阶段交给Actor。
- **M4-C（仅有明确证据）**：修复时钟/真实\`ack/consumed\`参考路径偏差或滚动内存状态的根因；保留外部物理行动合同及源代码证据。
- 在已有DEV上预设**联立验收**：H1/H16绝对物体mean-parent RMSE不劣于M1（1.151/1.958mm）；H1/H8/H16真实动作差逐父正向且误差不超过.5/1.0/1.5mm；preclose/release H16 Brier不劣于M1（.0734/.0548）；报告H16绝对TCP误差和稳定抓持相对位姿；候选排列和单例输出完全一致；**无stage可被严重牺牲来换H8差分**。
- 未通过时，不扩更新；若所有神经版均不及强OLS/物理基线，应允许结论\`NEURAL_WM_NOT_JUSTIFIED_FOR_THIS_S_STAGE\`，将稳定搬运使用结构/解析执行模型，集中模型学习于复杂接触，停止追求19M全域模型的表面胜利。

### B2：开发通过才配新盲测/在线建议
- 先冻结新至少8个真实未见父layout、parent分区和候选动作/预算；只在**已允许S域**采样，TEST数据不读，旧ONCE_HOLDOUT不重当新盲测。重复/尾部对照/拒绝/失败均计分母；样本不足时报告可信区间限制。
- **纯WM、OLS/规则、受约束hybrid**分别报告选择效果、回退比例与成本；当前已有DEV规则regret=0不能宣称模型增量。使用实际oracle改善空间并预登记显著区分的场景，先选动作、保存建议，再真实执行有限闭环，不用事后重打分代替兑现。
- 模型需对部署动作输入有合法轨迹/事件安全资格，**只有通过可靠动作影响/真实奖励/安全、在新场景表现优于或至少明确有增量的规则**，才申请正式Actor H16更新的另一个阶段。非零输入梯度≠梯度物理正确。无需提前训练5000步或启动PPO百万转移。

## 5. Track C：PC“像Isaac Sim”的可视化环境（独立于T动态准入）

- 立即以**用户私有PC**建隔离\`mujoco==3.2.7\`和可选Rerun（原PC工具\`scripts/pi0294/replay_viewer\`固定rerun-sdk0.40.0），不修改控制/训练环境。保留机器人/桌面/物体模型\`task-scene.private.mjb\`和同episode\`state.bin.journal\`的原SHA；从\`scripts.pi0294.task_state.parse_state\`校验\`GRTSTATE\`、\`nq34/nv32/nu20\`及接触，非原包拒绝。
- **第一优先实现真正mesh的历史3D播放**：MuJoCo被动Viewer，逐帧\`qpos/qvel/ctrl/time\`赋值、\`mj_forward\`、\`viewer.sync\`，**绝不调用\`mj_step\`**，显示Go2自由底座、四足、ARX5、夹爪、桌子/物体和可选BODY/WRIST RGB；可暂停、拖动、旋转与改变速率。确保高率包按原t播放；缺帧/失败段显式标注，绝不插值/合成“行走抓取成功”。
- 第二优先实现**仅订阅/只读 live state bridge**和PC显示（可由同机已有文件/独立localhost feed实现，不需要新控制服务或公开网络）；消息携带\`source_sha,scene_sha,monotonic/physics_t,step,sequence,qpos,qvel,ctrl,contact\`，一个原生源一个订阅者，不产生任何动作、改变物理dt或接管WBC。**无T动态许可时仅用录播/合成测试桥**；只有已合法运行的owner进程允许输出真实状态。
- 完成后交付README+命令、真实模型截图/私有视频位置、30fps渲染实测、CPU/GPU开销、无积分断言，以及PC侧\`SOURCE_REPLAY\`/ \`TARGET_FAILED_REPLAY\`/ \`LIVE_TARGET_AUTHORIZED\`清晰标签。私有模型/视频不得上传Git。

## 6. Codex 自主执行循环与分支决策

每个独立阶段：
1. **DISCOVER**：读本阶段最新HEAD与原证据、owner/lease/资源现状，不依赖老绝对路径必然存在；
2. **HYPOTHESIS**：先写可否证的2–3个假说、expected observation、已知负结果、最小变因；
3. **SPEC & CHECK**：冻结有限规格/预算/输入输出hash/测试，准备独立checkout。旧记录写\`PRESERVED\`、不动；
4. **RUN**：在准入范围内使用原owner锁、一次性run、硬deadline、止损、测试；run期间不热改代码；
5. **VALIDATE**：原生状态/动作/时钟/安全及神经一阶/多步/检查点按规格回读；普通/-O单测、适用集成、全仓CI能跑时报告；不把被跳过测试计PASS；
6. **CLASSIFY**：\`PASS\`、\`FAIL_ACTIONABLE\`（定位并进入新的假设实验）、\`OWNER_BLOCKED\`（转并行线，不重试）、\`BUDGET_EXHAUSTED\`（停止所有新有成本动作）；每次严格记录真实效果0或>0；
7. **PUBLISH & CONTINUE**：提交仅新增/可审阅的草稿PR，列原固定SHA、真正运行的源码SHA、数据/checkpoint/binary SHA、实际updates/integrations、正常/负结果、资源用量、当前阻断。可在剩余额度内自动推进下一阶段，不因“交付第一个PR”自动停止。

**停止/请求升级触发**：任何违反安全/owner门槛、其他用户资源冲突、冻结文件hash变化、未解释数据污染、原生STOP后有积分、模型利用未来真实输入、预算超限、需改WBC/安全/大规模队列、或实验假说连续失败且不能缩小根因。停止时保留checkpoint、run与负结果，最终给出最接近根因的证据与下一条真实CLI，绝不创建“伪PASS”。

## 7. 最小返工成果物（不接受只交路线图）

- 一个可运行、可回放的**私有PC Go2＋ARX5 3D Viewer**（静态+真实失败轨迹），含不积分测试和实际启动验证；没有私有授权资产则明确BLOCKED_ASSET而非伪造画面。
- \`TARGET_INIT_ROOT_CAUSE\`：旧trace因果对齐、已证实/未证实候选、已签/未签owner scope；若有有效范围，则真实2×0.1s＋2s/5s验收日志与停止证据；若没有，则可直接交owner的**具体、有界范围申请**，而不是“继续等待”。
- \`WM_STAGE_DIAG\`：与M1/M3及OLS/刚性基线对比的绝对/common-mode/动作差、preclose/release、H1..16；如有新模型，完整冻结spec、更新数、梯度/保存/恢复及真正独立开发评价；从不将DEV调参结果冒称TEST。
- \`RESOURCE_AND_PROVENANCE\`：每个run/lease/bin/checkpoint/data完整SHA、失败码、元数据、累计GPU小时/CPU核小时/私有磁盘/真实S/T转移、下一条**已实现**安全命令；不将“计划动态运行”表述为已经运行。

**阶段主优先级**：A0＋C先完成可复现目标环境/观测及可视化；B0并行锁定common-mode与接触mode；A1只在owner合法动态准入下进行；B1仅围绕一个明确机制按预算试验；A2/A3或B2依各自通过条件推进；其余工作不应被重复合成GPU烟测占据。
