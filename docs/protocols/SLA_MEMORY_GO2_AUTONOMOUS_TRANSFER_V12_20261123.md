# SLA_Memory → Go2 单臂观测/动作迁移：自主推进协议 V1.2

> 日期：2026-11-23。状态：**PLAN_ONLY; 本文件本身没有启动 GPU 或 MuJoCo**。上游冻结权重不训练、无 Agent、无真实硬件、无 DEV/TEST、无修改 WBC/2ms/P10/H40/G7/40mm/s/TTL/STOP。对应 [PR #428](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/428) V1/V1.1，若有冲突，以本协议中基于新回报的 V1.2 说明为准。

## 0. 当前事实、证据边界、科学目标

用户 2026-11-23 回报：Corelab GPU0（现场报 RTX 4080 SUPER，32760 MiB）真实 O0 通过，源权重19/19校验共12,441,551,063字节；8/8次12图/双臂 forward 输出有限50×16，2/2 reset 重放逐位一致；load 4.44s、第一次12.83s（XLA编译11.66s），其后7次中位0.293s；整卡VRAM峰8624MiB、进程RAM峰14.89GiB；六个源样本右臂相对专家RMSE 4.6–11.0mm，两个放置样本均产生正确开爪事件。P1 18次源域消融，正确 time-aligned 20Hz历史对 O0 中位偏差0.23mm，错误30帧索引7.01mm，联合替代左腕图和左臂状态12.27mm，且六个源窗专家误差都增加。合计26/26真实源域 forward。O2真实 forward=0；旧六个Go2 TRAIN窗只存2Hz RGB，8秒完整20Hz时间网格应有161帧，只找到17个时刻且缺episode reset首帧；P2/P3 NOT_RUN，MuJoCo积分=0、训练更新=0；GPU0已释放、回归120MiB。

**来源纪律**：以上结果为用户转述并指向 Corelab 私有本地报告；本轮 GitHub 计划制定者无法直接读取 `/home/liuyu/ISSAC-GR00T/work/sla-memory-frozen-transfer-v1/{P0_REAL_GPU0_O0_REPORT.md,P1_OBSERVATION_ABLATIONS.md,evidence/P1_O2_ARCHIVE_CHECK.json,FINAL_TRANSFER_DECISION.md}` 的字节。现场自主执行器必须首先只读核验这些报告、原始前向记录和 SHA 清单，并在本地追记其 SHA256；未经核验，GitHub 只能记录 `USER_REPORTED`，不能写 `INDEPENDENT_AUDITED`。不得覆盖既有报告或改变原26次分母。

科学目标：证明/否证不再训练的 `SLA_Memory_robodojo_real_ckpt` 在原始正确输入→缺一侧状态/相机→Go2双RGB 20Hz原生输入→冻结单臂EEF/WBC的阶段性能力保留。分开估算：观测域偏移、缺臂依赖、动作编码/时序问题、WBC可行域、最终任务技能。静止任务在前、移动底座任务另立科学版本。

## 1. 固定身份与职责

- 上游 XPolicyLab PR #237 commit `c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454`；HF revision `7f6880569c3be777291e646fad1b4951245e8553`；config `train_robodojo_real`、`extra_frames_first=false`、`attention_impl=xla`、`env_cfg_type=arx_x5`、`action_type=ee`，复用已经校验的模型/processor/stats/原source；不得重新下载/加载另一份无SHA的权重。
- Target：Go2＋单 ARX5，body/wrist 20Hz 双相机、实际TCP/夹爪状态，原native控制和SafetyGate。因源 O0 右臂表现良好，**在 O2 之前预注册以官方 RIGHT slot 代表唯一实际执行臂**。左臂虚拟状态/左腕图分别冻结一种明确的 source-derived 合法静止替代，作为 OOD；不能根据 Go2成功率动态换手/挑图。
- 输入 SLA 的 shape 保持双臂状态16D、三路当前RGB（image mask都true）+ 9张主相机历史 =12图；网络保持输出双臂 50×16（以已回报 O0 为当前source identity）；外部只选择右臂动作，不删模型权重或改动内部14D训练槽。
- 输出 source real-policy inverse transform 后是**每只臂自身base系下绝对 EEF**，不是 model 内部中点坐标；P2使用 `T_W_TCP=T_W_armbase T_armbase_EEF T_EEF_TCP`，绝不再叠加双臂 midpoint 0.6m offset。
- 实验独立本地根：`/home/liuyu/ISSAC-GR00T/work/sla-memory-frozen-transfer-v1`。原有 Core collector、Hopper 训练、旧封存 TRAIN/DEV/VAL、WBC/私有资产及 PR #426/#427 活动 capsule 只读。公开 GitHub 只提交协议、脱敏统计和哈希，不上传权重、私有场景或视频。
- 本协议是 **bounded autonomous runner specification**，并非本身批准新的 MuJoCo 请求。现场 owner、GPU0 lease、科学预算与任务人口必须写入新执行 ledger 后才能自动跨越需要资源的 gate；缺任何一项记 `AUTHORIZATION_BLOCKED` 并停，不自行申请别的GPU/偷偷复用旧PR物理预算。

## 2. 自治执行状态机与预算

状态机：`LOCKED → PREFLIGHT → O1_DISENTANGLE → O2_CAPTURE → O2_FORWARD → P2_ABI → P3_PAIRED → SEALED`；每个阶段只在实际收据 PASS 后走下一阶段；其他终态 `BLOCKED`/`FAIL_SCIENCE`/`TECH_UNKNOWN`/`NOT_RUN`。

- **CPU/Git阶段**：只读事实和配置比对、单测、静态几何校验，0 native physical tick。
- **GPU0前向预算**：O1拆分最多6窗×2额外分支=12；O2首轮最多6个真实Go2窗口+2个reset重复=8；P2如需补齐至16个离线窗口，新增最多10次；总计**新增不超过30次**真正模型 forward。实际有已封存输出则复用、不重复。单卡GPU0、batch1、预注册采样seed；每次记录 model SHA、调用/编译时长、GPU UUID与VRAM/RAM峰，所有异常计入分母。26次源域旧结果不得混入新调用数。
- **O2历史采集（新物理预算建议，非既有权限）**：优先尝试已有合法高密度源轨迹/原生复演（需包含可复现 reset、所有已消费控制与WBC内部初态）；不能从2Hz图像插值“填造”20Hz输入。无可复演源就独立注册最多**4条TRAIN native observation-only采集请求**，每条不超过真实绑定任务卡上限，建议总硬上限 `202812 ticks`，也不得超过4×50703（仅是原PLACE卡历史参考，不是许可）。如真实卡不等价取较小值/另行封存预算。到期/超预算停止，已有合法序列照样保留；不从未授权 DEV/TEST 获取图像。
- **P3物理（需独立明确授权）**：最多2个无接触接口卡+4对冻结场景 Teacher(4)/SLA(4)，共10独立native请求；旧V1估算上限367030×2ms tick，必须现场按真实卡重算。**O2采集请求与P3评估请求分别计数，绝不能互相充抵或重复计成成功**。无正式物理授权即停在 P2 已完成的证据上。
- optimizer updates=0、model parameter delta=0、Agent calls=0、real robot=0、SafetyGate/WBC/physics change=0。若某一步的实际成本高于注册，止损并向 owner 报告，不能向上“自动扩容”。

幂等：每次请求唯一 `run_id / phase / scene_id / seed / query_id / model_sha`，原始写一次；重启先读取 journal、owner lock、实际 native request count、STOP receipts。对技术 UNKNOWN 先审阅现存 receipt，不自动重复发起科学请求；STOP 后积分必须=0。

## 3. Step A0：把 Gate0 与 P1 实证正式接管

现场读取上述四份原始本地报告、原 O0/P1 raw predictions 和对齐记录；计算 SHA256，核对 19文件的本地内容、GPU0有效型号/VRAM 32760MiB 与 UUID、26 forward 账本、不完整旧2Hz O2 archive 中的17/161以及真实 scene/seed 的 TRAIN 所属。注册 `P0=PASS_SOURCE`、`P1=PASS_SOURCE_ABLATION_WITH_RESIDUAL`；不能把 P1 自动提升成 `PASS_GO2_OBSERVATION`。

验收：每一数量有现存数据和哈希；当前如无法读取私有源数据，则 `EVIDENCE_UNAVAILABLE`，不启动新模型。

产出：`A0_SOURCE_RECEIPTS_AND_SCOPE.json/.md`。不重复 O0 8/8 以“证明结果”。

## 4. Step A1：继续分解缺图与缺臂

锁定原6个 O0 源窗口/相同随机密钥，保留原 O0、错误历史和联合替代结果。在总30次新增预算内，优先只补两组各6次：
- `O1_VISION_ONLY`：左腕图固定源域合法占位，**左臂仍是真实源域状态**，其余完全相同；
- `O1_STATE_ONLY`：左臂为预注册合理 idle state，**左腕图仍是真实源域图**；
- 原 `O1_JOINT`（替代左腕图和左臂）与 `O0_NATIVE` 直接复用。

每一组报告目标右臂 EEF位置（paired median、p95、每窗口值）、SO3角度、gripper事件时刻与是否转变、专家误差、来源状态/相机哈希。做包含交互项的2×2分析，报告 sample n=6，并明确这只是模型输出扰动而非物理任务指标。不能通过选择更接近专家的占位图来定义唯一合格Go2配置；一旦锁定原方案再测试Go2。

验收：每组实际数据完整，左/右臂不颠倒，time-aligned history固定。模型对单臂缺失非常敏感并不意味着“立即训适配器”，而是记录潜在转移损失。

## 5. Step A2：O2原生20Hz观测采集（第一关键阻塞）

现有Go2 archive 2Hz只能提供8秒回看中的17/161个时刻，而且无reset首帧；**不能用补零、视频插帧或线性复制伪造“真实 O2 20Hz”**。必须增加独立、无损的 `go2_o2_dense_observation_capture` 路径：

1. 每条独立TRAIN episode 在 reset **同一sim tick** 捕获：reset cam_body、cam_wrist、实际 TCP/armbase/夹爪、base pose、T_world_armbase/T_EEF_TCP、source scene/control/renderer SHA、相机内外参与像素分辨率（如相机不支持前置读取，要明确reset后的首个有效帧时刻）。
2. 从reset到各预定候选查询时刻，按照实际 MuJoCo 时间，每 0.05s 即25个2ms tick记录完整同步双摄、实际TCP／gripper反馈、观测日期/时间索引、episode phase及对应的已消费控制标识；同时保留任务说明与设备id。**原真实RGB**应在本地受控空间存储并校验checksum，不替换现有2Hz证据和不直接发布到公开PR。
3. 每次生成O2时从实时帧时间戳选出 episode首图与 t-8…t-1秒历史点，目标应有 `t=0..8s` 共161帧（含端点，适用完整8秒时段）；若指定查询t<8s，按官方 episode-first padding 语义处理并注明，不与完整8秒条件混淆。同步误差对20Hz采样必须小于半帧（25ms），建议实际仿真目标≤1个2ms tick。采样缺口、重复、失序、丢帧、曝光延迟、不可复原 reset首图均单列，不伪装PASS。
4. 给每条帧写 `cam_body/cam_wrist` 的来源和 RGB/BGR/像素变换日志；body对官方head、wrist对官方right_wrist，缺少的left_wrist与虚拟left_arm遵循A1已注册固定 surrogate，mask全true但记录 `OOD`。官方resize224px不代表Go2原始图像包含224px细节。
5. 20Hz→源域历史**按时间戳选样**，不要直接每20Hz frame调用官方 fps30 memory.update；两种实现可选且需单独命名：从已采集20Hz帧建立12图Observation直接传入相同冻结processor；或30Hz虚拟索引点选择距离最近的**真实**20Hz图像（产生重复图必留证），再交官方Buffer。首选直接时间戳builder，须验证O0 30Hz的等价性；不能声称新builder是未经修改的上游Buffer。
6. 与 O2 capture 相关的物理积分明确记在独立 `OBS_CAPTURE` population；这些是采集请求，不是Student闭环。优先复现原六TRAIN窗口的合法reset及已消费控制序列；如缺少完整物理/控制历史，则注册新TRAIN episode与相似阶段，**不可声称精确重放了旧六窗**。
7. 数据本地索引按 `episode/scene/seed/reset_tick/observation_t/query_t` 排列，保存含完整20Hz历史、reset frame和标定的可复原六个查询锚点；如采集面板未覆盖6个，报告真实不足而不是加随机源造样本。

验收：O2历史真实/原frame hash、采样tick和时钟、两路同步、TCP/夹爪和标定可回读；至少一个独立 episode 中可还原完整12图O2语义，下一步目标6查询；每个条件记录161帧覆盖情况及起始帧，缺项标记 `O2_HISTORY_BLOCKED`。该验收不要求SLA动作已经成功。

产出：`A2_O2_DENSE_CAPTURE.md/json`、私有frame manifest、source replay receipts、新增physical ticks。

## 6. Step A3：真实 O2 forward 与冻结单臂策略行为

冻结源后训练checkpoint+sample RNG，先用2个真实 O2 window 做离线 model-input smoke，确认模型处理3路当前RGB/9历史/16D状态、输出 finite 50×16；再按已注册最多6窗做右臂动作、gripper事件/目标方向评估，预留最多2次reset replay（这两次与用户P0旧2次不是一回事）。严格使用 actual Go2 TCP、world ← arm-base标定构建右臂状态；左侧保持已注册 surrogate。所需对象位置仿真真值只供评分，**不能输入policy或据其生成动作目标**。

对原生O0、O1、Go2 O2分别记录相同指令/任务phase、右臂物理轨迹大小、角速度/夹爪开合及首点；OOD条件的专家RMSE不是成功率，且 O2和 O0非同布局不直接比较RMSE差当成“损失12.27mm”。Go2物体/任务语义若与源Pack-and-Pour-Fruit不同，明确`SKILL_TASK_OOD`；优先选择一手可独立完成的抓取+放入容器局部技能，如当前场景不支持同义任务，不能虚构源域匹配。

验收：真实 O2至少一份完整合法输入+一次有限动作+来源完整报告，是 `PASS_O2_FORWARD`；任务方向/夹爪阶段失常则记录 `O2_SKILL_WEAK`，不冒称源模型无法操作。Go2输入尚未准备好不能因为时间紧继续 P3。

## 7. Step A4：P2动作与时间ABI，离线优先与A2并行

可在A2等待数据时，先用O0的既有50×16输出开发**纯确定性**`right_arm_real_eef_to_go2_h40`，不需要目标GPU/物理。源右臂输出已是 arm-base-local **absolute pose**，组合 `T_W_TCP = T_W_ARXbase * T_ARXbase_EEF * T_EEF_TCP`，对 quaternion wxyz→rotation matrix→项目 row/column rot6d 逐项 roundtrip；测试合法零姿态、90°角、旋转跨±π、base移动但eef相对不变、左右槽误选、TCP offset反序、gripper 0..1/实际毫米宽映射。
- 来源30Hz、输出50点0..49：由真实O0确认时间零点，目标H40 40点@50ms；如重合t0，则目标index33..39 7/40（17.5%）处于源1.633s以外；只允许一个**预注册、显式命名**的`HOLD_SUFFIX_7`方案作为主ABI候选；hold尾部是reference compiler而非模型原预测。不能偷偷拉伸、改SLA长度，或以预测外后缀计算模型RMSE。模型下一次重规划仍按原 P10 消费/反馈/调用时序处理。
- 全程保存raw16→selected8→worldSE3→EEF10→H40→G7和原native bridge接收/拒绝；检查full chunk内和上一实际已消费P10动作→新chunk的线/角速度、首点位置、夹爪40mm/s及原SafetyGate所有observed-vs-threshold；不使用 smooth、额外clip或有选择地改变gripper时刻去取得PASS。
- P2离线最多16个预注册 TRAIN窗口，A3已有6可复用，余不足可追加≤10个真实Go2 O2 forward（受总30个GPU请求预算、真实高密度数据覆盖约束）。若无足够真实O2窗口，P2将输出已完成的子集与BLOCKED，禁止用源O0伪装成目标O2。
- 原native教师参考通过同一门；记录不可行性/可达性与实际源frame标定。原 bridge 测试可用mock/CPU静态路径，但不能把mock合法当成已完成物理。

Gate P2：数值逆变换及历史/时间语义经双向校验，且至少存在原门合法学生轨迹及匹配Teacher合法轨迹，才能申请/执行 P3。全部16/16拒绝、错位或未知单臂角色，停在 `ACTION_ABI_BLOCKED`；不降低门。

## 8. Step A5：P3原生冻结WBC有界配对（需独立物理GO）

与A2 capture互不充抵的新科学人口 `SLA_MEMORY_FROZEN_TRANSFER_PAIRED`。前置：O2验证、P2 GatePASS、任务语义匹配、当前正式source seal/owner、GPU0 lease和world owner/物理预算都有效。绝不能自动重用旧VAL/DEV/F或把源域teacher phase给学生。

首轮优先站立不强制移动Go2；至少匹配一类单臂抓取并放入目标容器的任务，如果现有F1/F2 stackcube与pack_and_pour_fruit并非相同技能，要分开报告 OOD 与 Task-Matched，不能偷偷更换成功定义。
- 两条无接触native技术请求：检测安装ACK、取消/STOP后的0积分、WBC初态可重复。
- 四个pre-registered scene/seed配对条件：Teacher EEF→同WBC(4次)，Frozen SLA右臂EEF→同WBC(4次)；共最多10请求，按历史建议上限367030 ticks，必须新注册真实卡和owner。
- 保留双方相同场景reset及合法观测，但Teacher可以读取其合法特权源用于阳性控制；SLA绝对不能读取物体GT／Teacher phase。所有学生尝试（含模型原Gate拒绝）都按注册ID计入，不允许只挑“合法安装成功”的统计分母。
- 记录 GRASP、LIFT≥20mm、HOLD≥1s、PLACE/RELEASE/物体终态、实际base移动、场景相同reset、WBC achieved TCP 与 raw/installed轨迹差、首次拒绝门、非法接触、STOP后积分=0，显式区别模拟暂停运行与连续实时硬件。
- 完成4对只能证明机制可行/失败及其分类，不能推断可靠成功率。任何任务技术 UNKNOWN 或原G7/接触STOP先封存，不暗中重复；新运行必须另立预算。

## 9. 决策规则（不自动训练）

| 观察结果 | 首要归因与后继研究 |
|---|---|
| A0证据缺失/设备无授权 | BLOCKED；不重算旧成功、不申请其他GPU |
| O1 only-vision影响大，only-state影响小 | 单臂 camera surrogate/visual adaptation 是优先候选，不立即训练 |
| O1 only-state影响大，vision影响小 | arm-slot/state virtualisation 先做确定性研究 |
| O2数据不完整/无法复原reset | OBS_HISTORY_BLOCKED；回到可信数据采集，不运行P2目标任务 |
| 源O0正常、O2输入完备但动作行为失常 | O2_CAMERA_OR_EMBODIMENT_OOD；后续可以新协议研究轻量adapter |
| O2动作合理但H40/G7全拒绝 | ACTION_TIME_OR_SEAM_FAILURE；只修查明的确定性接口错误并新版本化 |
| Teacher＋WBC成功，SLA＋WBC不能跟踪 | 冻结WBC与reference可执行域/衔接诊断 |
| Teacher＋WBC成功，SLA正常跟踪但任务失败 | 高层技能匹配、目标/接触/夹爪事件问题；不要归咎WBC |
| Frozen SLA在声明任务下完成实际抓放 | 先对独立TRAIN新scene扩大验证，再另立移动底座/Agent或少样本适配对照 |

以上不允许以一个 RMSE、一次preview视频或一条成功重试代替最终结论。本轮 **不修改权重、不训练单臂adapter、不自动新增任务或Agent、不改变底层物理/门**。

## 10. 交付、状态更新与验收

每阶段至少写一个小的真实 `ACTUAL.md/json`；包含：固定SOURCE_SHA/HF revision、source episode/scene/seed、observer profile与mask、所有调用/请求/物理ticks/训练更新的实际分母、结果/失败分类、原始哈希、GPU/CPU/磁盘使用和最后确认的资源释放。私有源文件不入公开GitHub，只能提交脱敏证据摘要及校验哈希。每次过门才向PR #428追加一条简短的事实评论；**报告永远区分 USER_REPORTED / LOCALLY_AUDITED / GITHUB_SUMMARY 与 NOT_RUN**。

强制STOP：任何未授权 native 使用、抢占现役GPU、不可解释的旧请求重复、模型/processor stats SHA变化、源train/test泄漏、frame history越过episode、未处理的未知TCP/quat、SafetyGate阈值变更、非0 post_STOP_ticks、非法自碰撞、原始证据被删除/覆盖，均立即停止并保留足迹。无新物理GO不能自动解锁A2 native采集或A5 P3。

建议最早一份新里程碑为 **A2_O2_DENSE_CAPTURE**：至少一条可追溯的 Go2 reset episode 的20Hz双RGB、同步EEF/gripper/标定，以及可以回放的 t-8..t 历史。它将替代本轮2Hz档案作为真实O2源，不依赖先训新模型。

## 11. 后继实施组件（不污染现役模块）

建议将来执行PR独立新增 `go2_o2_dense_observation_capture`、`o2_time_aligned_history_builder`、`single_arm_ood_profile`、`sla_memory_target_forward`、`right_arm_real_eef_to_go2_h40`、`transfer_gate_runner`、`transfer_native_pair_runner`；这些只是概念模块名，尚未确认现存代码可直接提供相同入口。由执行者复用已封存的 Core world/bridge 和 RoboDojo policy server，在独立运行目录构建，不改现役源码/worker。

提交针对性的 unit tests：时间戳缺/乱/重复、reset首帧缺失、16/161误归有效、以20Hz当30Hz历史、错误RGB/BGR、左/右臂错选、invalid quat、base-midpoint双加、EEF/TCP错序、row/column rot6d错、夹爪接缝/40mm/s、模型输出时间尾部7点不标记、预测外插值、G7拒绝、STOP后物理tick、非TRAIN数据混入和重启重复请求。

**本文件不是“已运行”的证明**。下一项能够改变状态的实绩是Corelab现场原始证据核验和新20Hz高密度观察数据，而非再做普通模型 smoke 或训练。
