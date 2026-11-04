# WM02 NIGHT1：今晚实际采集优先的自主执行计划

版本 V1；编写日期 2026-11-05，Asia/Singapore。用户要求：“制定详细的自主推进计划，今晚目标是走通仿真环境并真实开始采集物理动力学数据用于WM policy，计划推送github，给出简短启动prompt”。

**本文件是给实际具备现场工具的Codex的执行委托，不是已启动的作业。** 目标平台为完整自由底座Go2＋ARX5、原生MuJoCo＋ROS2/OCS2＋冻结WBC。不要回退到固定底座S域替代目标成果。今晚不追求正式Actor训练、行走或复杂任务成功；必须真正新增目标物理数据，构建可用于后续WM/Policy研究的Replay并做有限模型更新。

## 0. 唯一入口、已知成果与证据边界

执行代码基线：PR #355 / `35c7954cad86e952cce8b4b8e0234c17b7e31ed5`，分支 `research/wm02-autonomous-execution-20261104`。本计划是其叠加的文档分支，不假设main已合并。首读根AGENTS.md；再读本文件和 `experiments/wm02/night1_collect_first_v1.json`。对今晚的优先级，以本文件为准；原AUTO1现场权限、冻结控制与安全契约不变。

已核实的来源：
- `reports/WM02/auto1-dynamic-g2/EXECUTION_REPORT.md`、`TARGET_WM_METRICS.json`、`PARALLEL_ACTUAL.json`；
- `src/go2arx5_vla/world_model/target_parallel.py`、`target_replay.py`、`target_training.py`；
- `experiments/wm02/auto1_parallel_pilot_v1.json`。

已完成：目标A192＋B96＋C48＝336条50ms任务转移；A+B288条进入训练，C48尚未更新模型；小型目标GRU为74,986参数、106维原生状态，已完成300主更新＋2独立恢复探针。完整本体、WBC参考保持及极小自由空间采集、双环境试跑均有真实证据。它们不是今晚新增成果。全部旧数据来自同一物理父场景、DEV=0；接触/抓放未合格，Actor=0，完整任务=0。selected-policy身份关联仍为DERIVED，不得改写成直接controller导出。

固定模型：`9a0ef30ba0e79412312c434de504f94b7476611ffd3b6d484b929926445dfd51`；A+B Replay：`c29390ff49c66164f46d49b941e410fd276c95f4472635a256a140e65e2c515c`；C Replay：`a10d88a63d7e738a32f7130aa9ec8a097ea490a0d1e9c4ba4e3bf9bf9626ea77`。从私有已登记路径解析，不将标题猜成文件路径。

**今晚不要重新从G0开始。** 已有双环境和训练资格能复用时直接复用；不重复全部历史审计、合成烟测或原G1六次保持。依赖/代码/执行器变化才做对应回归。旧`qualify --domain T`返回78不是当前AUTO1入口，禁止再用它代表目标不可运行。

## 1. 夜间范围与自主性

### 1.1 时间与接管

本夜指2026-11-05凌晨，默认 `effective_deadline = min(实际启动+8小时, 2026-11-05 09:00:00+08:00, 现有合法campaign截止)`。这是运行上限，不是完成时间承诺。若启动已超过本夜截止，只做只读接管报告，不私自平移日期或延长原lease。预留最后20分钟用于退出、封存、恢复核验和发布，不再派发不能在剩余窗口内安全结束的任务。

先查现役owner和原Journal；使用原AUTO1 campaign和唯一physical owner。若已有合法worker运行，读取现状并在其episode/批次冷边界接续，不重复派发、不抢锁、不改活动capsule。旧CLOSED lease不重开；每个新的有界worker使用新run身份，原现场机制登记。此用户委托授权普通工程修复及有界仿真推进，不能伪造外部站点凭据、容量或安全通过。

### 1.2 可自主进行

在独立worktree/capsule中修复非冻结路径、数据、时间映射、批次/长度、序列化、checkpoint和退出问题；普通/-O回归后用新身份接续。可实施原AUTO1范围内的已合格自由空间采集、单环境分级接触资格、目标Replay、有限WM训练及草稿PR交付。阶段通过后自动进入下一合法阶段，不因发出一份报告或完成200更新就停在聊天等待GO。

今晚不允许：真机、主动行走、修改冻结WBC/腿权重、增益、成本数值、solver或物理dt、机械/碰撞/安全阈值；删除旧数据/失败、强杀其他用户进程、系统/网络/ROS全局升级、自动合并main、正式Actor/PPO/GR00T训练、A40/H200长队列、MuJoCo Warp迁移。

### 1.3 资源与有限端点

默认2个已资格目标环境；新接触条件先单环境。今晚不扩4/16环境。每slot沿用现场独立CPU集合、ROS域220/221和私有cache/state/quota/trace/STOP；不要根据旧空闲量假设当前资源仍可用。原生采集CUDA隐藏；仅一个合法空闲GPU模型worker，GPU1优先、GPU0仅实际获准且不影响他人时可用。不要求双卡，也不因没GPU而停止CPU采集。

native worker沿用原批准有限端点（默认不超过600s）；模型worker默认不超过2400s、4GiB allocator，原站点更严值优先。整夜最多96次新目标尝试（含证明、失败、接触和生产），每次任务段不超过8s；累计原生积分预算上限650,000 tick，包含预热、被拒绝前验证和失败；最多10,240条**新合格学习转移**，原336不重复计入。最多1个200主更新及其1次恢复探针＋1个100主更新及其1次探针，共302次新优化调用；失败中已发生更新计入，不能为了恢复重跑200。建议私有新增存储不超过20GiB且始终保留现场所需余量；临近容量则停新任务，用户负责清理。

## 2. 今晚结果定义：先保证有新数据，再追求扩展

- **第一实际产物**：用已经合格的1.2s双环境配置采到至少一个全新成对批次（48条上限），封存并构建Replay。不得先改长序列数小时而一条新数据没有。
- **最低可验收工程目标**：至少16个新完整采集episode，至少384条新有效50ms转移；可在现有单父场景与原1.2s范围完成。计全体预登记尝试，不能挑成功重试凑数。未达到则返回PARTIAL及首个确定阻断，不保证必达。
- **主要目标**：预登记8个真正新增父条件，6 TRAIN＋2 DEV，每父4个预定episode，合格时按4.8s采集，最多3,072条。父场景变化必须是真实物理布局/合格初态变化，不能仅改seed/run_id；只改命令的面板应标ACTION_CONDITION_PANEL、parent泛化未验证。
- **扩展目标**：前述完成且资源/时间允许，再增加不重叠的8父（6 TRAIN＋2 DEV）；总16父×4次×8s最多10,240条。8s长度须先通过资格，不能为达到数量跳过。
- **接触目标**：完成或明确定位剩余受支撑抓持/释放四项，保留真实接触转变；至少观察到一次接触不等于抓持/完整任务成功。接触未合格时完整交付自由空间数据，不能冒称操作动力学已覆盖。
- **学习目标**：新TRAIN数据至少一次200更新＋恢复，独立新批次先评价后100更新＋恢复；模型不优于简单基线也交付负结果，不以科研分数阻挡已合格采集。

## 3. 顺序执行：N0→N1先落地，后续按分支继续

### N0 — 只读接管与C批冻结评价

核查实际PR head、dirty worktree/运行owner、有效scope、代码/二进制与设备身份；若head已前移，先读新报告，保留并发提交，用实际执行SHA封存，不覆盖它。读取现有预制C评价spec和输出状态：已完成则核验，不重放onceIO；未执行则调用**已实现** `scripts.wm02.evaluate_target_wm`，只用CPU、固定C和后继checkpoint，零更新零物理。可用现有 `core_control.py` 私有入口，但必须核对原现场路径/命令，不把本文件当作新执行器。

C评价是新执行但同父TRAIN的工程前瞻评价，不是新父DEV。评价失败若是软件/身份错误须修；评价指标差不阻断安全物理采集。接管后不做全Home扫描、不清理未知进程、不重装环境。

### N1 — 立即使用已资格2环境采集

新run ID下复用原双slot资格、MEASURED_EEF_AND_BASE_POSE_HOLD_V2、原50ms/2ms频率、350ms续约、source+400ms TTL。先至少一组已合格极小±x/保持参考，2×1.2s；新数据从真实WBC消费的参考及状态得到。核验每slot各24 atoms、25状态、完整退出，父摘要与Replay计数一致，actual selected-policy关联继续标DERIVED。

沿用源代码中已修正但未额外物理重跑的批次摘要/失败关闭逻辑前，做针对性普通/-O测试与本次双环境实际验收；不要复写旧错误摘要。成功数据马上封存并报告SHA。这一组可计最低工程目标，但仍属于原物理父TRAIN；不能计进新父DEV。

若2环境资源或并发控制质量不满足，关闭本批自己的进程并使用已资格单环境继续；不要以扩大CPU/杀进程换速度。若出现原生安全STOP，停止受影响批次，保存首次状态与日志；仅有明确修复或新的合法条件才重新启动，不直接重复故障动作。

### N2 — 仅做采集所必需的薄接口扩展

当前不是通用采集器：`target_parallel.py`只接受两个固定phase/端点；`target_replay.py`固定600任务tick、24转移/25状态且单MJB；`target_training.py`依赖固定1.2s时钟和特定B4名字。不得只放宽验证字符串或移除hash检查。

按单变量实现：
1. 已登记有限episode manifest驱动批次，仍2slot，一个父owner继承canonical锁，不建立新队列/daemon。
2. 变长完整序列与显式窗口索引：burn8＋predict16需要24连续转移/25状态；长度L可产生L−23个滑窗，但按父/episode保留依赖，重叠窗口不冒称独立数据。
3. 机器人/执行器结构指纹与每scene资产SHA分开；多场景必须预登记白名单、固定robot joint/actuator映射与控制器，不删除身份校验。场景上下文必须能由观测/已知任务取得；隐藏参数只给合法训练模块并明确标记。
4. 时钟从真实窗口起点产生，支持已声明续约；未来source age未知时保留missing mask，不能填真实未来值。原固定目标模型只能使用预声明的完整目标/执行规则；任意变化H40需新因果动作schema，今晚不默认扩大。
5. 原接口仍可复现旧1.2s数据；新格式版本化，原文件/元数据/checkpoint不改。checkpoint加载在状态/动作schema变化时必须显式迁移或fresh model，禁止静默非严格加载。

自由空间任务段按单环境2.4→4.8→8.0s逐级检查，其间不同时扩大动作幅度；完整身份不变后做双环境同长度验证。持续保持也是真实动力学，但要统计动作方向和响应覆盖，不能将纯等待堆成操作数据。

**防止本夜只剩重构**：最多两轮针对同一扩展缺口的有依据修复仍未解决，或扩展工作累计占本夜可用时间25%，就将其标DEFERRED，立即使用原1.2s collector继续最低采集目标。不得因为长episode、多场景或漂亮GUI未完成而停止已合格的数据生产。

### N3 — 先冻结场景，再持续两slot采集

主面板：8新父（6TRAIN/2DEV）×4次；按原合法站立工作空间对物体/目标布局或允许初态做有界变化，静态构造/可达/净空检查后逐父有限动态资格，未资格项保留拒绝。不要瞬移到悬空已抓状态，不改变robot惯量/摩擦/控制增益，也不把移动桌面以绕过碰撞当成修复。

同父所有reset/重放/窗口只能落在一个split。旧A/B/C父永久TRAIN；新父分区先于物理生成结果。任务在slot间轮换，不让slot0固定TRAIN、slot1固定DEV。对已通过父使用保持及合格极小方向参考；新增方向/夹爪模式先单环境资格后加入生产。episode失败不换父挑选结果。

先完成Wave1，即新8父或已实现的保守子集；封存一次Replay后马上进入N5，不需要等待扩展16父。若多父构建受阻，继续旧父工程采集但标SAME_PARENT_ONLY，不宣称DEV。

并行加速来自同时运行完整CPU原生链，不是MJWarp或显存并行：各slot独立ROS、可写codegen/cache、状态/日志/trace/配额/停止。只共享只读资产；一个Replay写入者只读取已封存episode。每50ms无全slot屏障；episode终态后可派发下一个有限任务，不能在活动episode换代码或模型。接触诊断期间先收尾双slot，使用同一owner单环境，不偷偷加第三个实例。

### N4 — 接触资格支线：单环境、有边界、不压住主线

补原G2剩余四项：两项受支撑抓持/夹爪、两项支撑后释放/撤离。先核真实关节/滑轨范围、BASE/table净空、对象/指垫碰撞对、支撑与grasp标签。使用从正常初态合法执行的接近前缀；不因缺完整MPC快照强造中途reset。

只推进已预检路径中的一个极小增量，原native safety/STOP保持；不因曾有base_link/table非法接触而允许该碰撞对，不改变安全阈值。不故意越限制造STOP；原真实ILLEGAL_CONTACT的STOP后0额外积分是历史证据，新条件未触发则写NOT_TRIGGERED。

合格条件才进入CONTACT_COLLECTION子面板，每个合格条件最多4次新前瞻尝试，记录左右接触、稳定抓持、支撑、实际夹爪、释放与撤离；CONTACT_TRANSITION标签需来自原生事件，不用几何代理或illegal_contact替代support。最多16个接触生产episode计入总96次上限。无接触资格就返回准确首个拒绝/失败与下一工程缺项，继续FREE_SPACE_TARGET。保持/几何Teacher不是正式Policy，不把局部接触写成整任务成功。

### N5 — 用新目标TRAIN实际训练，并作后继更新

在第一批满足schema/时钟/分区的目标序列形成后，先用冻结旧模型评价新批次；同父评价、独立新父DEV和训练后拟合分开。

默认保留小型history-conditioned目标WM；不搬S50/M4-B checkpoint，不因数据变多就上19M模型。若schema相容，明确继承optimizer/RNG/cursor与TRAIN-only normalizer；若新表示/normalizer需改变则从新配方开始并比较，不能假装严格续训。模型仍是缺完整控制隐状态的POMDP近似。预测中的qpos quaternion与rotation6d不得未经合法解码就当有效SO(3)控制输入；今晚模型仅离线预测，不驱动控制。

仅TRAIN入训练、scaler和ridge拟合，DEV从不参与。第一批200主更新＋1独立恢复探针，完整保存模型/optimizer/normalizer/RNG/cursor/metadata/schema/executor/data/source身份；探针不污染发布端点。第二批是预登记新TRAIN尝试，先评价后最多100主更新＋1探针；严格恢复与不中断参照一致，在固定软件下记录实际容差。若GPU不可用，继续CPU数据工作；CPU小模型训练只能经独立资源测量、版本化后执行，不能抢占原生控制CPU集合。

H1/H8/H16逐字段报告TCP、BASE、关节、夹爪、物体，位置用m/mm、角度用SO(3)/rad或明确标numeric quaternion-component诊断，不把混单位qpos平均loss称物理准确度。保留persistence、TRAIN拟合的ridge reference-residual、明确的CV；静止物体持恒误差0时，直接承认WM不占优。接触头仅有真实合格正负事件时才能训练/评价。无需赢过这些基线才算数据/工程入口完成，但不能据loss下降宣称控制收益。

### N6 — 收尾、发布和精确终态

目标输出目录建议 `reports/WM02/night1-20261105/`。至少产出：实际代码/二进制/模型SHA、已执行命令与退出码、预登记任务表和全分母、原生tick/有效atom/partial/拒绝、每父每动作接触覆盖、Replay/schema/split、模型checkpoint与恢复结果、失败和资源成本、下一条已实现命令。

同一个最小进度索引复用原AUTO1_STATE/Journal，记录当前阶段、实际worker/run、计数和next action。每个批次封存更新；若需要有限工作链在Codex工具返回后继续，由现场已有合法机制登记并实际启动后才声称运行，不创建循环daemon，不把Chat计划当后台实验。发布继续使用正常feature PR或在实际执行分支追加，不自动merge或force-push。main与旧数据不变。

终态分开：REAL_TARGET_DATA_INCREMENT_VERIFIED、FREE_SPACE_DATASET_READY、NEW_PARENT_DEV_AVAILABLE、CONTACT_COLLECTION_QUALIFIED、WM_REFRESH_RESTORED。只有对应真实证据才能PASS；Actor和完整任务本轮仍单独计0。不以97项旧测试代替新代码回归，不以计划目标计成实际数据。无新物理数据则不得将今晚写为采集完成。

## 4. 验证和失败处理

必要回归覆盖：普通/-O、import/--help零积分；双slot域/路径/CPU冲突、继承锁、父退出子清理、同行失败STOP；原生TTL和350ms续约、跨window参考与clock；可变长度/partial/reset/scene分区拒绝；TRAIN-only normalizer、冻结新批次评价、完整恢复。原安全触发后停止该候选及当前兄弟slot，封存有效prefix并标失效，**本夜默认不将安全失败prefix加入生产TRAIN**；后续可在显式失败数据配方中使用，绝不删除。

对同一故障只有新证据/实质修复才能新身份重试，最多3个修复候选；无修复不重放正式失败面板。不将未结束PID或不确定作业标CLOSED；只结算确属自己且核验终态的进程。全局用户STOP、权限失效、冻结字节改变、失控积分、数据泄漏、磁盘不足或到期立即按原机制停止新派发并安全收尾。

今晚取消非必要前置：完整三轮N=1/2严格性能benchmark、扩4环境、WM架构大搜索、更多源域面板、全仓历史扫描、重建GUI不再阻塞首批数据。可附带测每批有效atom/s、预热/任务/关闭/序列化墙钟，约2倍只作已有描述性观察，不写成保证。严格benchmark在数据目标完成后另排。

## 5. 真实入口与未实现项

当前已实现并有历史实际执行记录：`scripts.wm02.parallel_target`、`commission_target`、`collect_target`、`build_target_replay`、`evaluate_target_wm`、`train_target_wm`。运行必须使用现场合法spec/run身份，不能直接重放旧输出或CLOSED lease。

今晚须按需新增/扩展而非假称已存在：通用有限episode manifest、可变长度Replay与窗口、预登记多scene兼容、后继任意批次训练/评价、接触采集资格。保留现有只读核验作为检查，不将它当作新科研成果。

**下一条真正有价值的成果：一份全新目标双slot采集回执和Replay SHA。其后再交长度、父场景、接触和训练；不能整夜只返回计划或复核报告。**
