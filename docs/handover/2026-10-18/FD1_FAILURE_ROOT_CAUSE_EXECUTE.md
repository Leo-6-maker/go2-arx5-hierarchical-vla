# Codex 接手：FD1 不改 WBC，查清现有 SOURCE_CONFIRM 的真实失败机制

**2026-10-18，Asia/Singapore。当前用户直接决定：不要修改 WBC；先查清失败原因。**
目标：从 **40条件/42原尝试**的已封存数据中解释失败形态，必要时用**有上限、仅CPU的原动作原历史 shadow 重建**辨别首次非法几何。**不重新训练、不重复确认、不更换模型或调整WBC、安全、评分。**

## 读取顺序与身份

1. 沿用现有 single local-PC Codex owner，从 CoreLab 统一入口 `~/Projects/GR00T-WBC` 按 `AGENTS.md`、`START_HERE.md`、统一入口工作区规范/registry/doctor 与 #11/#47 刷新最新现场事实。旧 MP1/N1 已完全结束，**不能将旧lease/预算/owner时钟解冻**；新的CPU诊断child采用新有限资源卡与实际 native site 权限。
2. 从**同一个固定、可达、经回读的 GitHub commit**读取三件：
   - `docs/protocols/PI0294_FD1_FAILURE_ATTRIBUTION_CPU_20261018.md`（完整科研与工程协议）；
   - `reports/PI0294/design/FD1_FAILURE_ATTRIBUTION_CPU_SCOPE_20261018.json`（机读许可与上限）；
   - 本 `docs/handover/2026-10-18/FD1_FAILURE_ROOT_CAUSE_EXECUTE.md`（执行入口）。
   将commit SHA、三文件blob SHA、owner、实际adoption时刻与新FD1 manifest/lease登记入 `FD1_ADOPTION_AND_SOURCES.json`。GitHub发布不等于原生adoption或执行。
3. 以真正完成的原证据 `658b432fa1ee37182945e84b7d4ec4d830bee67e` 为科学基线，重点看原run的 `N1_FINAL_SCIENCE_REVIEW.md`、`N1_SOURCE_CONFIRM40_FINAL_RESULT_AND_EVIDENCE.json`、`N1_RESOURCE_FINAL.json`、最终归档42视频、冻结 S1/Teacher/scene/score 代码。不要按历史中途快照把完成项目当待办。
4. 固定唯一已评价的 `S-CORR@2048` manifest `94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38`、原确认plan `6e5a45e121aa493e1de32977c00d3f20ecc3c1f9cb7a6b9698438cbb16028e97`。原TF-R3安全STOP保持，WBC/腿策略/plant/夹爪PD/S1/score不动。`selected/deployment=NONE`。

## 需要解决的问题（一次只检验一个链条）

**已确认数据：**T1 10/20，T2 4/20，两者原14/20目标未达；23次S1拒绝（21预测非法接触/穿透、1速度、1关节限位），3次32秒物体目标失败。S1拒绝的违规前缀没有真正执行，0原生实际安全事故不能证明去掉S1仍安全。

**关键解释陷阱：** T2的14个 `AUDIT_ONLY_approach` 全在**12–28.8秒**拒绝，其中实际物体已经被推进的案例仍写approach；这是旁观Teacher阶段而非物理真实phase。先用500Hz接触和物体位移确认实际时序，再说接近/接触/推送/退出哪个阶段有问题。

**待证机制：** 21个非法接触首次精确geom pair未知；不能冒称都是指垫碰桌。推送Teacher名义 `+4mm` pad-table裕度不是实际全程最小间隙。T1完整目标失败是未释放；T2两次完整目标失败均未进目标区且违反原≤5mm的no-lift判据。T2成功只在F2，family混杂空间和外观，不可直接归因视觉。双bank成功不一致可能是后续状态累积/随机动作差异，不是“挑成功seed”。

## 工作状态机

**FD0 — READBACK / IDENTITY**：现场检查没有旧活跃科学actor、确切owner/lease/manifest、原四十个条件＋四十二次attempt及两个工程partial、所有 hash 和原生源身份。只有新范围真实登记+有效权限后才能做CPU shadow。原始文件始终只读。

**FD1 — ZERO-NEW-PHYSICS ATLAS**（先做、耗资低）：从原 `S1_PREFLIGHTS`、真实physics/highrate/contact、object/EEF/gripper、SCORE、original action/ACK/noise以及结果构造 **40-cell×42-attempt**事件图谱。23次拒绝分开看拒绝前真正接触与物体位移、原S1类别/决策时间/预测offset；3次完整任务失败分别核对zone、release、no-lift、41 hold。凡原件没存 geom pair 写`FIRST_GEOMETRY_NOT_IN_ARCHIVE`，不编造。

**FD2 — FIRST GEOMETRY**（仅FD1报告确实缺少必要物理几何时）：按协议允许的**新FD1 CPU-only shadow**复用原动作数组及完整native history-copy，先过原`<=1e-10`完整状态回放parity、原前缀拒绝类与tick；再记录首次原违规 contact pair、signed distance、机器人/物体几何、joint/TCP速度、请求XYZ/SO3/grip与实际状态。原结果不改、无新模型查询、不把shadow称为真实发生接触。无法恢复就保留UNQUALIFIED，不降低判据。旧BC512/M1“table-pad”不自动等于现CORR2048的21例。

**FD3 — HYPOTHESIS TESTS**：按任务、10独立scene、2固定bank分析：
- 真正发生事件和旁观Teacher标签错位；
- 违规前真实物体是否接触、已推多远，首次contact与安全裕度；
- 原T2 4mm**名义**裕度、signed geom clearance与XYZ/SO3/夹爪几何，不能直接用全局RMSE减4mm；
- 原T1未release及T2两个no-lift/goal失败的终末谓词；
- F0/F1/F2的几何和颜色/光照混淆、四个两bank不一致的scene条件，不能只展示成功bank、不能按bank调模型。
有可信同状态专家的旧TRAIN原件才可做固定通道归因；CONFIRM不能产生新TRAIN标签，不自动进行恢复Oracle/新Teacher。

**FD4 — SCIENCE GATE / CLOSE**：将每个候选原因分为 `OBSERVED`、`SUPPORTED_NOT_CAUSAL`、`HYPOTHESIS`、`UNMEASURED`。注明每个拒绝首次违规几何覆盖率及缺失率，不要求为显示进展而补齐不可靠假数据。最终只选择**一个未来**TRAIN-side方向：稳健专家裕度、独立T2几何覆盖、完整合格Recovery Oracle或任务终末观测/状态历史。若无证据则明确`MECHANISM_UNRESOLVED`；这些未来训练/新动态Teacher试验均**不在FD1权限内**。关闭新CPU scope，结算原生站点/CPU/物理计数/费用。

## 资源与禁止越界

本用户新指令只覆盖受限FD1诊断，原N1余额不能继续使用。所有新增开销严格照机读卡：
- CoreLab累计CPU ≤48 core-h；local-PC≤8 core-h；每站并发≤4核；CPU源域shadow物理ticks≤500,000、starts≤64，真实新episode/NN调用/训练更新/新Teacher示范/GPU/PBS全部**0**；
- 新scratch≤12GiB，登记网络≤8GiB；实际quota/site许可更小者优先，未核验manifest/租约/有损副作用时不执行；任何失败/重建也要收费；
- 新FD1 owner采纳≤发布后24h；采纳+40h停止新shadow，+48h必须真实收尾；排队/重启不延长；**不等于重新开N1**。
- 不许修改 WBC代码、权重、增益、代价、solver、腿策略、plant、手爪PD、S1安全阈值、评分、GR00T checkpoint/推理配方；不放宽拒绝以追求成功；不热切原来代码；不得用私人token/raw模型/视频更新GitHub。
- 只允许批准的读侧CPU工程与必要同语义诊断修复，实际CPU shadow必须严格使用注册CPU launcher和可重复native proof；无法取得site权限则发布`BLOCKED_BY_EXACT_AUTHORITY`，不绕过权限。
- 当前WBC是被冻结借用的执行环境，不搞WBC质量追责，不重做hold RMS；源域失败不能归给尚未在Go2+ARX5完成任务的目标WBC。

## 必交验收与汇报

在现有原MP1报告根新建有限`FD1_DIAG/`小收据索引，不覆盖已终态N1原件：`FD1_42_ATTEMPT_COVERAGE.json`、`FD1_FAILURE_EVENT_ATLAS.md`、`FD1_40_CELL_ACTUAL_PHASE_MATRIX.json`、`FD1_SOURCE_DATA_GAPS.json`、`FD1_SHADOW_NATIVE_PARITY.json`、`FD1_S1_FIRST_GEOMETRY.json`、`FD1_LAYOUT_NOISE_TERMINAL_ANALYSIS.md`、`FD1_EVIDENCE_GRADE_MATRIX.json`、`FD1_FINAL_ROOT_CAUSE_REVIEW.md`、`FD1_NEXT_EXPERIMENT_DECISION.md`、`FD1_RESOURCE_FINAL.json`。

向GitHub **Issue #11**登记固定提交、采取`ADOPTED`或精确`BLOCKED`的真实owner回执、科学归因和最终下一分支；Issue #47仅记录新的工程修复与运行资格。**本卡已发布≠Codex已读取≠CPU诊断已经运行**。实际步骤与消耗以owner原生收据为准。

成功定义：保住全部40/42分母与原失败，给23次S1和3次完整目标失败建立真正物理阶段/规则/几何证据，辨别哪些原因有足够证据、哪些仍缺测，并给出一条可测试的科研决策。**本阶段新任务成功率无需改变，也不能伪造。**
