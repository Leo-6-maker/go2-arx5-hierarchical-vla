# PI0294-FD1 — 源域失败根因鉴别协议（WBC 完全冻结）

日期：2026-10-18，Asia/Singapore。Decision ID：`PI0294_FD1_FAILURE_ATTRIBUTION_CPU_V1`。
状态：`USER_REQUESTED_BOUNDED_DIAGNOSTIC_SCOPE / SINGLE_OWNER_NATIVE_BINDING_REQUIRED / NO_EXECUTION_RECEIPT`。
用户决定：“**不要修改 WBC，继续先查清失败原因，计划推送到 GitHub**”。
科研目标：解释现有 GR00T/EEF10 固定底座 ARX5 的 **23 条 S1 拒绝、3 条完整但未达任务目标的失败**；给下一阶段的 **一个** 有辨别力的科研方案。**本阶段不训练、不收集新的任务示范、不调/修/归因 WBC，也不重新打开 N1。**

## 0. 证据基线、权威和边界

- 原样封存的最终科学报告：`658b432fa1ee37182945e84b7d4ec4d830bee67e`，`reports/PI0294/20261016T191445Z-localpc-mp1/N1_FINAL_SCIENCE_REVIEW.md`；
  原始条件与全部尝试：同目录的 `N1_SOURCE_CONFIRM40_FINAL_RESULT_AND_EVIDENCE.json`；
  原生结算：`N1_RESOURCE_FINAL.json`；
  旧方案草案（**NOT_AUTHORIZED**）：`N1_NEXT_T2_TRAIN_COVERAGE_DRAFT.md`。
- 证据：40/40 固定确认条件、42/42 实际尝试独立审核、42 段原片完整存储；T1 10/20、T2 4/20，未达到每任务14/20；T1 9个S1拒绝+1个32秒目标失败，T2 14个S1拒绝+2个32秒目标失败。23次 S1 拒绝分为21次 `ILLEGAL_CONTACT_OR_PENETRATION`、1次 `TCP_VELOCITY`、1次 `JOINT_LIMIT`；它们是**预测前缀拒绝（零违法前缀实际执行）**，实际 native safety 故障0。
- 10个独立源 scene-group，每任务每组2个固定噪声bank：T1 0/2、1/2、2/2组数4、2、4；T2为7、2、1。T2四个成功均位于F2；但F0/F1/F2的场景生成同时涉及几何/颜色/光照差异，**不能把 family 直接解释成视觉因果**。
- 本轮基于最终确认状态，**原 N1 完全收口**：全部 MP1 作业终态、24 lease CLOSED、future reserved=0；原计算余额**不是新 GO**。要对封存数据做新的受限 CPU 诊断，必须由原 local-PC **单一协调 owner** 以新有限 FD1 child / RUN_MANIFEST / authority / lease / output / native site 权限绑定，精确累计新成本。不能将旧 N1 lease 激活，也不能让第二调度 owner 抢占 shared NFS。
- 本协议与 `docs/protocols/GR00T_TASK_DATA_FIRST_FROZEN_WBC_SCOPE_V1.md` 一致。**WBC、腿策略、Go2 plant、机械/摩擦、夹爪PD/力限、Solver、gain/cost、所有安全限值和评分公式原字节保持不变**；不开展R4B/D1或hold-RMS补偿归因、不注入固定XYZ/Z补偿。FD1只分析**源域固定底座 ARX5**，不以该域结论替代 Go2＋ARX5/WBC 或真机结论。
- 模型不变：唯一被分析的 CORR@2048 manifest `94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38`；训练父和数值设定、BF16/TF32、EEF10/H40/Prefix8、20Hz/500Hz、Teacher OFF、安全门S1、原task-success规则不变；无需新模型加载。旧行为、视频、hash、STOPS、失败/工程partial、HF revision 都只读。
- 数据泄漏边界：SOURCE_CONFIRM40 **已经打开，只能用于失败审核/假说发现**；不将其中的新scene、图像、状态、物理响应或教师参考变成训练/纠正标签，不在这些原确认条件上做补丁后重新优化/重新选择 checkpoint。今后验证要另建新独立人口。已有实验成绩不能重标或改成功阈值。
- 协议与机读约束见 `reports/PI0294/design/FD1_FAILURE_ATTRIBUTION_CPU_SCOPE_20261018.json`；现有 owner 不应以 GitHub 发布或聊天文本代替实际权限、scope adoption 或执行收据。

## 1. 审核后待验证的机制：按可辨别证据而不是按直觉排序

### H1. S1 是主要**失败形式**，但“非法接触”的几何根因未定
当前最优先证据缺口不是平均 XYZ RMSE，而是21条非法接触预测中**首次违规时的精确 geom/body pair、signed contact.distance、法向、IK/关节姿态、实际TCP与请求参考、指垫相对桌面的最低裕度**。先验证是pad/table、其他arm/table、物体异常穿透还是别的几何，再研究最先偏离的参考通道；不把旧 M1/BC512 的 table-pad 结果套在当前 CORR2048 全部确认负例。
`scripts/pi0291_arx5_data/engine.py` 的非法接触规则包括**任何 arm/table 或 arm/floor contact**，以及 `contact.dist < -0.004` 的额外路径，两者**不是同一阈值**。不能把4mm penetration stop改成“允许手指碰桌4mm”。`scripts/pi0294/s1_preflight.py` 原 S1 只在shadow预测，无安全失败前缀的真实执行。保留零live事实。

### H2. T2 的 `AUDIT_ONLY_approach` 标签可能与真实任务阶段严重不符
42次实际尝试终局显示，T2的14次S1拒绝发生于**12.0–28.8秒**（从固定收据逐行读取，非估计）；旁观 `RelayTeacher` 仍附 `AUDIT_ONLY_approach` 标签，因为它沿自己的专家状态机观察学生，不控制学生动作。20秒后已实际推动物体的轨迹依然可能显示approach。必须从高频真实接触/物体轨迹、夹爪与TCP状态重新推断**实际物理事件**，不能按标签宣称“主要早期接近失败”。

### H3. 普通推送专家的名义几何裕度可能过窄，但还不是根因
`scripts/pi0294/scene.py:push_geometry` 通过实际pad相对TCP盒几何设定 `tcp_height_m=table_top-bottom_offset_m+0.004`，即设计上的**4mm静态离桌名义裕度**；并非动态全过程实测间隙。对比专家态约3–4mm H40位置均误差**不能直接证明**发生垂直方向碰桌；必须把当前位置XYZ、姿态SO3、开合、接触几何和动态servo响应分开量化。任何未来更稳健Teacher路径都属于未来独立TRAIN方案，不在本轮偷偷调整参考或真实运行阈值。

### H4. 模型终末事件也不可靠，不能只修S1
T1一条完整32s任务抓取/抬升/搬运成立，但**released=0、hold=0/41**；T2两条32s目标失败既未进目标区，又有约**7.54–7.89mm**物体抬升，违反原 **no-lift <=5mm** 的高频判据。需要量化“首次正确接触→最大推送→物体相对目标→结束接触→打开/分离→终末保持”全过程；不能把移动距离≥10cm冒充完成推送。

### H5. 与噪声、布局相关的失稳可能来自感知或长程状态偏移
每任务仅10个场景组×2 bank：T1与T2各有2组discordant。T2仅F2成功，是**描述性关联**，不区分几何、视觉、方向、接触裕度；same-scene不同noise初始渲染像素相同不意味着执行后观测/策略状态相同或可归因于模型噪声。应定位**第一次相同状态的生成差异、随后的真实状态分叉**；有相同输入+noise的真实数组才比较，同步历史不完整时明示不可归因。

### H6. 普通专家动态可行性在新布局上并未得到完整证明
此前独立确认仅在模型评价前使用静态IK、静态碰撞和视野资格；**静态合格不等于原任务Teacher在该布局上全程安全成功**。不能根据学生失败直接断言模型容量或数据不足。先从既有TRAIN专家和合格历史评估可执行裕度；若需要**新的动态专家 pilot**，将其列为未来单独批准的科学分支，不在FD1中用已打开CONFIRM布局制作正标签。

## 2. FD0：只读身份与证据封存（第一个、强制）

由现有 local-PC owner 读取最新 `AGENTS.md`、`START_HERE.md`、Issue #11/#47、上述固定完整报告和安全/工作区规范。读取真实CoreLab注册根、数据绑定、lease/job/完整收尾事实、现有CPU权限、个人quota/隐私目录；对比固定模型/plan/stats/source/数值和终局工件的hash。
任何最新文件缺失或不匹配，停在 `EVIDENCE_GAP`，不能自行重跑原40条件。原40 cell/42 attempt 原始身份、两个工程 partial、两个原native失败、0未决action effects 均保持。将成功/拒绝/目标失败/工程partial分开建立 provenance join，绝不复制敏感 raw/hash以外路径到Git。
输出：`FD1_ADOPTION_AND_SOURCES.json` 与精确的 `FD1_42_ATTEMPT_COVERAGE.json`。一项不可归因不阻断其他独立可读证据。

## 3. FD1：零新增物理、零神经推理的全量失败事件图谱（最高优先）

对**全部40个固定条件/42次原尝试**做已存数据的只读连接，不删工程尝试、不重复有效负例：
1. join `CARD` / 原始 `RESULT` / `SCORE` / `S1_PRELIGHTS` / 原 `physical`、`highrate`、`contact_audit` / actual RGB snapshots / model command arrays / installed ACKs / original noise fingerprint。高成本整视频无需为同一案例重复解码，优先读取原audited索引和有界局部帧。
2. 每次拒绝定位**决策时刻、真实已执行200-tick prefix历史、第一次预测违规offset、预测类别、影子结果摘要**；只报告存档已有的 geom pair / signed clearance。如原始S1仅保存类别/offset而**没有准确geom pair**，明确 `FIRST_GEOMETRY_NOT_IN_ARCHIVE`，不能凭标签或另一模型猜测。
3. 重新标注“真实物理事件”，例如首次真实碰物体、第一次达到预登记1mm/10mm物体位移、是否超过任务规定10cm位移、物体是否进目标区、grasp/lift/release/detach、TCP-clear和终末41点保持。**事件阈值要分成‘原任务判据’与‘新诊断性的早期阈值’，后者不得改评分。**对于S1拒绝，**只利用拒绝前真实历史**，不得把预测shadow当真实已发生事件。
4. 23条S1/3条完整任务失败逐一出`actual_phase`，并同时保留原 `AUDIT_ONLY_*` 和二者不一致标记。首先按任务、几何族、原bank、是否已有物体接触或达到目标、拒绝实际时刻（T1早于约12s、T2在12–28.8s）拆开，不用一个“approach”类掩盖。
5. 两个工程partial在42 attempts保留；但原唯一replacement已给有效结果，**不将工程partial计为新失败条件或模型任务失败**。保存全部40固定cell作为分母。
输出：`FD1_FAILURE_EVENT_ATLAS.md`、`FD1_40_CELL_ACTUAL_PHASE_MATRIX.json`（脱敏、仅小型聚合/摘要；逐帧私有存档）以及 `FD1_SOURCE_DATA_GAPS.json`。目标为42/42身份闭合、23/23拒绝时刻原样映射、3/3终末谓词拆分；缺几何等字段应明确missing率，不伪造100%测量完成。

## 4. FD2：冻结物理的首次违法几何重建（限额CPU shadow；无真实新任务）

仅在FD1证明存档缺失确切首发接触几何且实际本用户诊断授权、单owner、site CPU权限、独立fresh diagnostic lease、资源控制和完整历史状态校验通过时，允许启动本地**严格只读的源域MuJoCo shadow**：
- 先按先前已合格的“完整历史＋native copy”路径，在**原始确认状态**重建到S1决定前，不用早先已失败的integration-only shortcut。不推新GR00T动作，只使用封存的原模型输出数组。原S1决策、first-reject offset、原真实已执行动作，参考当前物理、shadow和安全结果必须在**未改原始数值阈值**下逐项一致；`max(fullstate_error) <= 1e-10` 只是原已采用的状态奇异阈值门，超出则停止该case并标注 `RESTORE_UNQUALIFIED`；不能放宽通过率。
- 对原23个S1拒绝首先严格重建原决策之前的因果轨迹，最多每个case一次封存重建；记录**首次原规则违规几何体/Body名称/接触距离（m）/法向/时间（s）/关节边界裕度/TCP速度，及其对应请求的XYZ/SO3/夹爪与实际状态**。只有原原生代码、资产、时钟和录制历史一致时输出`QUALIFIED_ACTUAL_CAUSE`，否则`UNCERTAIN`。显示预测违规而非实际发生碰撞。
- 归因仅限于同一保存状态的passive geometry、原已录动作；如要做仅XYZ/SO3/gripper替换，**先使用原已合格的TRAIN历史锚点及其原合格专家**，对照必须在固定状态下同时有合法数据身份。不能用已打开确认的场景构造未来TRAIN标签，不能更改原策略/评分，也不能把替换的安全短前缀升级成完成任务/恢复成功。需要新训练或Teacher运行则停止提交单独新GO。
- 允许有界解释已发生失败，但不发新模型请求、不执行新的源域物体任务、不从zero-live拒绝动作在真实live继续rollout，不修S1/plant/控制器/夹爪。
- 如原始shadow全量复现可能超过CPU/tick限额，则用预先固定的代表子集（T1早期/T1中期/T2后程/三类S1规则，各含双方噪声及成功对照），**先抽完整名单后读模型结果**；采样记录不允许依改进程度挑例。未覆盖的拒绝如实保留，不外推为全部。
输出：`FD1_S1_FIRST_GEOMETRY.json`、`FD1_SHADOW_NATIVE_PARITY.json`，以及表格按原因分列：table-pad / arm-table-other / table-floor-other / penetration-other / TCP velocity / joint limit / UNPROVEN。总和需按23原拒绝对齐，任何新额外类别及未测数应保留。

## 5. FD3：跨布局、噪声与终末行为的机制证据（纯已有结果/已合格shadow）

制作三组互相独立的图谱与比较：
- **几何—裕度**：根据实际SOURCE/TRAIN/CONFIRM原始CARD计算推送方向/距离/X-Y/旋转范围、固定`push_geometry`名义+4mm与实际模型终端接触裕度，分别看`S1 collision`和`goal-fail`；不要直接将XYZ整体RMSE减去垂直裕度。明确F2只有4/8 T2成功是相关性，不能宣称视觉导致。
- **相同场景双bank**：每任务10组分别报告0/2、1/2、2/2；四组discordant的第一个原生实际观察/动作分岔时刻、最大/最终目标误差、释放/保持/接触指标；对图片完全相同的初始时刻和原已录噪声可提出精确关联，但**不能宣称因果噪声单因**。
- **终末谓词**：T1唯一完整目标失败未release；T2两次完整失败未进区、实际抬升约7.54–7.89mm > 原5mm no-lift门，严格重报各41 hold子谓词、首次越限时刻和物体路径。不做“视觉上基本成功”打折计分。
- 按**scene-group（10个）**汇报描述性聚合、配对discordance与不确定性；不能把两噪声bank当20个独立scene。交叉family与姿态/方向的统计只能作为描述，避免多重筛查后声称根因。
输出：`FD1_LAYOUT_NOISE_TERMINAL_ANALYSIS.md` 与小型`FD1_EXPLANATORY_TABLE.json`。

## 6. FD4：结论评级、未来实验分流与正式停机

结论分 `OBSERVED`、`SUPPORTED_BUT_NOT_CAUSAL`、`HYPOTHESIS`、`UNMEASURED/UNQUALIFIED`。每一条必须引用完整 caseID、模型/源/资产/存档哈希、原生S1规则、触发时刻与误差判据。不得给一个“唯一根因”除非同状态、same physics、可控单变量证据确实足够。

**下一轮优先选择一条科学假设，FD1本身不训练**：
- 如果专家原路径可行裕度小且学生在同一位置小偏移就撞桌：下一轮考虑**TRAIN-only的稳健专家路径与普通BC受控对照**；先单独确认专家动态成功，不调WBC/S1。
- 如果专家在有裕度的状态良好、学生实际状态显著偏移：未来考虑**有合格完整Recovery Oracle的TRAIN-only on-policy纠正**；T2旧Oracle仍0/2，禁用伪专家标签。
- 如果主要是场景/方向覆盖欠缺而原专家有安全完成能力：优先固定场景类型的**TRAIN-only独立 T2 ordinary-expert coverage BC**，继续沿 `N1_NEXT_T2_TRAIN_COVERAGE_DRAFT.md` 的单变量思想。
- 如果失败在后程目标/释放/分离/保持，而非几何临界：未来单独研究真实可观测状态与任务终末条件/历史；不向策略加入Teacher privileged `stage/object_pose`。
- 如果没有充分证据支持任何上述机制：将缺失证据列为最小下一实验，不凭未花完预算训练模型。
- 新的**动态专家pilot、神经训练、SOURCE_CONFIRM复用、任何新目标WBC运行**都不由本协议自动授权。

产出：`FD1_FINAL_ROOT_CAUSE_REVIEW.md`、`FD1_EVIDENCE_GRADE_MATRIX.json`、`FD1_NEXT_EXPERIMENT_DECISION.md` 和 `FD1_RESOURCE_FINAL.json`。保留`selected/deployment=NONE`，关闭FD1新lease，结算真实CPU、shadow积分、物理启动及未知副作用。汇报未来方案需要的**一条**明确新科学GO及预算卡，不悄悄开下一轮。

## 7. 新授权与资源（仅本用户“先查清失败原因”的受限诊断；不是N1剩余额度）

允许在原本权限有效、注册新的诊断 child、采用完整发布包且通过manifest/lease/site身份后，以**独立、有限的CPU诊断资源**实施FD0–FD4；不创建第二科研manager，不依赖用户在夜间逐次确认同范围的读侧缺陷修复。

| 资源 | FD1新增硬上限 |
|---|---:|
| CoreLab（4080S/AMD合并计费）CPU | **48 CPU core-hours** |
| local-PC CPU协调与资料核对 | **8 CPU core-hours** |
| 每个站点同时在用核数 | **最多4核**（实际站点更小者） |
| 已保存行为的原生 MuJoCo CPU **shadow** | **最多500,000个物理ticks**（包括历史恢复、S1影子、失败/中断的全部有效ticks） |
| 全新shadow物理进程/starts | **最多64次**；一case最多一次原始重建，任何额外control需有独立诊断ID及余量 |
| 全新 live task episode / Teacher示范/GR00T生成 | **0 / 0 / 0** |
| H200/A40 allocated GPU/PBS | **0 / 0 / 0** |
| 新优化器更新、纠正标签、新训练数据 | **0** |
| 新增scratch输出峰值 | **最多12GiB**，原始数据/模型不复制全量 |
| 新增登记网络传输 | **最多8GiB**（原已有cache优先） |
| 原训练、评分、安全、控制器、模型参数更改 | **0** |

计数已发生值、预留值必须分别登记；失败CPU shadow仍算真实tick/start消耗，读侧CPU计实际+明确保守上限，用户quota/站点硬限更小者胜出。不靠删除历史检查点腾配额；此前checkpoint盘点eligible0/deleted0/freed0不再反复全盘重扫。
FD1首次实际合法adoption记录独立 `FD1_T0`，**最晚在本协议完整发布后的24小时内**；adoption之后 **40小时停止新 shadow**、**48小时必须关闭诊断**，排队/中断/重启不延长。未能取得合法新scope/站点绑定时，只能做可访问的GitHub小型报告整理并输出一个`BLOCKED_BY_EXACT_AUTHORITY`，**不得复活旧N1的任何lease、工作命名空间或提交余额**。
允许CPU审计程序、hash readback、读侧分析、必要同语义bug修复与原安全门负例测试；发布前run/test/CI与immutable capsule核对。脚本不能在import时直接调MuJoCo执行。真实simulation必须通过注册CPU launcher、guard、source/asset精确封存与新source identity校验，不能用普通shell绕过admission。
禁止新GPU登录节点推理、隐式SSH、永久cron/daemon、权限绕过、未知effect重试、热切已完成源capsule、改WBC/Go2/plant/夹爪/安全或将模型训练和诊断混作同一事务。

## 8. 标准反馈与固定证据交接

GitHub上只存脱敏的协议、机读上限、聚合结果、排除原因、校验通过/缺失摘要、可追溯固定ref；私有绝对路径/真实图片/权重、敏感工作区配置、原生令牌不得提交。原证据目录与只读HF revision持久保留，不覆盖N1原报告和原32/40面板。
将上述FD1小型报告置于同一父MP1报告根下**单独`FD1_DIAG`子目录**，而不把它写进原有终局JSON充当新的闭环结果；用Issue #11索引科学归因，Issue #47只记录修复与新scope的实际工程收据。没有真实 native 资格的产物只能写`CPU_LOG_AUDIT_PASS`或`PREPARED`，不能写`NATIVE_SHADOW_PASS`。

成功准则不是新的任务成功率，而是**23/23拒绝原身份/时间/类别映射、3/3完整任务失败终末谓词与实际事件归因、40/40条件+42/42原attempt原分母保留、可证明的首次违法几何覆盖率/缺失率，以及只有一条证据支持的后续实验建议**。如某关键几何不可在原预算/状态复现下确认，如实报告UNQUALIFIED/UNMEASURED，不凭旧Teacher标签或视觉猜测填空。

> 此协议是提交给现有Codex owner的**新有界诊断任务卡**；文件的创建**不是**现场fd1 child已经登记、CPU/影子模型已运行或用户追加授权训练的收据。
