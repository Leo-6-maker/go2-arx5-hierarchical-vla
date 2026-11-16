# M_FREE SLA 长程 Goal V2：EEF／结构化夹爪解耦、移动闭环与独立安全准入

**日期：2026-11-17（Asia/Singapore）**  
**状态：REVIEWED_DOCUMENTATION_PLAN / NOT_DEPLOYED_BY_THIS_PR**  
**适用仓库：** `Leo-6-maker/go2-arx5-hierarchical-vla`  
**实施参照（不可变）**：执行 [PR #393](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/393) HEAD `e3fb6005597a9329fa886c3015e9c5fa7f92afbe`；旧Goal [PR #392](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/392) `74264d5b56cf0cf201713d407152a36ecd798983`；事故基线 [PR #391](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/391) `65bd544d8ed3bf6a717783f8691fcb95c8d12165`。

> 本文件是对**已经实现而尚未取得实测的** `docs/goals/2026-11-17/MFREE_EEF_GRIP_DECOUPLING_AND_SAFETY_V1.json` 的**长程目标、预算、科学分流与验收补充**，并不替代其中已提交的执行代码与封存SHA。发布文档不创建Core任务、GPU lease、解除隔离、批准新物理试验或延长allocation。执行前读取最新现场事实，任何新进程必须是新的登记身份。禁止热改现役SHIFT7胶囊、重启旧CLOSED任务或将main误认为最新执行源码。

## 1. 目标与可核实的当前进度

### 1.1 最终研究目标（GOAL_MET的定义）

在**同一冻结MuJoCo／OCS2 WBC、MPC、腿策略、臂PD、2ms物理、原碰撞与完整H40门**下，训练或组合出一个能够独立普通RGB部署的SLA高层策略，完成Go2＋ARX5的 **M_FREE、F1/F2移动抓取—物体抬升≥20mm—连续持有≥1s**；真实任务方向底座净位移**≥8cm**单独计分。SLA不得读取专家stage、物体真值、未来参考、专家接管或BASE oracle；高层不控制腿关节。旧0/13有效学生任务失败、旧RP8、JOIN10和原生碰撞绝不回填。

- **科学里程碑A**：固定TRAIN48×2的完整40点原门/事件/命令接续通过，并有CN38 episode-heldout验证，证明至少存在跨视觉状态的一致候选。
- **工程里程碑B**：真实共享父模型双adapter推理、原native参考/ACK/TTL/部分P10和STOP生命周期逐层有证据（未准入前只能非物理测试）。
- **安全里程碑C**：独立责任方绑定代码／场景／profile及有限预算签发重新准入；原生安全停止0不是从离线数据自动推得。
- **能力里程碑D**：获准的pilot、DEV和一次独立TEST实际完成任务，预设DEV≥24/32、F1/F2各≥12/16；TEST≥32/40、F1/F2各≥16/20，两个面板均0硬安全停止。数量为目标，不是当前成绩或工业安全保证。

### 1.2 本轮只读核查到的事实（不同证据级别必须分开）

| 研究块 | 可核实事实 | 未证实或禁止称作 |
|---|---|---|
| 冻结WBC＋专家 | CN38=38合格/41专家物理请求，F1=24、F2=14；2次本身ILLEGAL_CONTACT、1共享隔离中止 | 不是C60/C100完成，不代表可恢复生产；14条scene_lineage须按实际模型SHA校验 |
| 原C20学生 | 20个成功episode／1376窗；真1,000更新／32,000优化访问；原生抓持0/13，另11未启动 | 不是“没训练”；也不是新策略成功 |
| 754次旧query | 742合法、12拒绝，12/12均P10后缀finger_velocity | 不是754个独立物理episode；不能归因joint3或整体EEF无法表示 |
| 固定TRAIN96 | 原C20 CLOSE 0/20＋非CLOSE 76/76；FM续训500 0/20＋76/76；物理时序监督500 7/20＋76/76 | 离线合法不等于任务成功 |
| TIME500 | 真实500更新/16K访问；CLOSE20/20、其他68/76；8条LIFT后缀k26线加速度0.2524–0.3084m/s²（限0.25） | 不得删掉后缀、平滑掩盖或在单一样本上宣称因果 |
| SHIFT7 | **PR393最新可核实阶段为50/500更新／1,600访问（2026-11-17 00:36:39 SGT）**；既有server任务有限截止08:26:30.531913 SGT；后续实际终态尚未在本轮GitHub检索中证实 | 500不等于已完成；不得重投或热改 `ef584be687d9f3c94e1f79e0e9f44ad12f32e92a` |
| HYBRID V1代码 | PR393已有 `hybrid_eef_grip_v1.py`、`hybrid_saved96_v1.py`、`cn38_hybrid_bank_v1.py`、`hybrid_cn38_infer_v1.py`、`hybrid_decoupling_control_v1.py`、共享父模型`resident_close_pair_v1.py`；机读状态`IMPLEMENTED_NOT_DEPLOYED` | 尚无组合96通过、CN38泛化或完整native链成功的证据 |
| 安全 | 原学生+两新专家碰桌；2ms被动监测提前12/34/316ms，**误拦1/38成功专家**；候选C++仅1204保存状态数值核对、未安装 | 不是可用保护器；F1/F2共同专家／学生和M_ASSIST隔离仍有效 |

读取最新GitHub的执行来源和Core原生回执时，以最新**实际**结果为准。报告SHA、运行source SHA和数据SHA分别记录；GitHub PR更新时间不等于Core训练进度更新时间。

### 1.3 当前明确的科研判断

TIME500的结构化夹爪时间表示稳定解决CLOSE超速，却在共享action expert上出现8个LIFT空间曲率回退。专家LIFT标签在整段加速度峰值只有约0.15m/s²、余量约0.10m/s²；k26专家接近零、FM500仅0.0194–0.0377m/s²，新TIME500为0.2524–0.3084m/s²。该加速度由
`(p[27] − 2*p[26] + p[25]) / 0.05²`直接产生；0.6–1.4mm轨迹误差并不能保证导数合法。时间参数槽原分割[0,14),[14,27),[27,40)，异常接近槽27是**待检验机制**，非已证明因果。

SHIFT7只改变三参数位置布局（全体循环右移7点，原边界27移到34，新增环绕边界7），并保持原物理解码和监督；无论异常迁移与否，它都不是可直接自动授予native权限的策略。下一优先不是再扫SHIFT14/21，而是用已经保存的FM500空间输出及TIME/SHIFT夹爪输出做**固定通道组合**，验证消除共享参数干扰的可行性。

## 2. 执行顺序（每个阶段必须有明确的终态和分流）

### H0 — SHIFT7终态摄取；0新计算

1. 只读核验SHIFT7真实unit/PID/start/cgroup/lease/Journal/source/spec；区分RUNNING、CLOSED_SUCCESS、CLOSED_TECH_FAIL、UNKNOWN。**正在运行就只等待原任务封存，不提交替身、不覆盖同身份文件**。
2. 科学成功必须有**实际500更新／16K访问、250/500保存和87数组全态重载、96 TRAIN预测、16独立RGB replay、全500访问及noise绑定、全20 LIFT的原k26/新k33以及k6/7/20/21/34曲率**。若缺失则标记不完整/技术故障，不把50步快照外推。
3. 用同一噪声、同一输入、同一物理H40全门比较原TIME500与SHIFT7：CLOSE、非CLOSE各分阶段、所有first-exceedance、terminal width、crossing、last-emitted-command seam。迁移k26→k33只支持位置耦合，不单独构成训练收益。
4. 只有在完整匹配SHIFT7科学终态成立时，复用现有混合V1执行合同；如果SHIFT7未产生有效500端点，**不得悄悄把现有要求两个端点的控制器改成TIME-only后仍用同一个spec**，应单独登记缩减范围且保留技术终态。

### H1 — 原96条件的两套固定通道组合；新增0优化／优先0模型调用／0物理

调用已有 `scripts/sla/hybrid_saved96_v1.py`，但先核对source pin、同条件配对、保存预测路径和新版本的CPU单测。两个**预登记的全球固定**组合：
- `FM_EEF_TIME_GRIP`：**FM500解码后的EEF位置＋旋转**，TIME500正确解码后的夹爪；
- `FM_EEF_SHIFT7_GRIP`：相同FM500 EEF，加SHIFT7正确解码夹爪。

对48 TRAIN窗口×2固定noise=96条件，每组合检查96次，共**192次组合H40检查**；对FM500原始96条的所有四项空间通道也做shadow，不能仅依据原first-reject掩盖空间失败。分别用原`FiftyMsProgram`的nominal/reference全H40五通道，保持H40=40／50ms、P10=10／0.5s、原40mm/s和原平移加速度0.25m/s²。

**绝对禁止**：合成归一化第7通道（两个codec语义不同）；按query/seed/是否过门动态选来源；利用专家phase或目标挑选输出；在评估中限速裁剪、接续平滑、删掉k≥10后缀；重计旧模型为新优化。

**预先承接PR393的全局择优**：若主要`FM_EEF_TIME_GRIP`达到CLOSE≥18/20、非CLOSE**76/76**、20个已标注事件全部≤100ms、终末宽度误差≤1mm、上一命令至首点夹爪参考速度≤40mm/s、且没有任何空间通道不合法，则固定该候选；否则只有已登记的SHIFT7组合满足同条件时才选SHIFT7；两者都不满足时封存`NO_TRAIN_CANDIDATE`，**不自动继续CN38或新模型**。即使得到96/96也只报告`TRAIN_OFFLINE`，不是成功抓持。

同时记录每个query的夹爪接近目标时刻、EEF首次抬升≥2mm时刻和相对差；这是**非准入的协调诊断**，不是新的放行动作门。必须报告“EEF先抬、夹爪尚未关”的风险，即便速度门已通过。

### H2 — CN38−C20的独立episode离线诊断；条件性0训练／≤376完整生成／0物理

仅在H1已有**一个全局固定**候选时复用PR393的`cn38_hybrid_bank_v1.py`和`hybrid_cn38_infer_v1.py`，不重复实现或训练。

1. C20成员严格为20条；CN38有效成员38条，去重后**新增18条（F1=14、F2=4）**，按原native result SHA、实际MJB SHA/layout与同步RGB／状态／时间身份检查；14条旧`scene_lineage`字符串仅作审计，不能替代真实scene。冻结C20统计、原action表示、训练父权重。
2. 每episode在PREGRASP、DESCEND、CLOSE、LIFT、HOLD各选**一个事前固定的下中位锚点**，各用noise 0/1：18×5×2=**180个条件**。每条件两个分支生成，合计**360次完整生成**；另外8个相同旧RGB条件×2分支=**16次**独立双adapter重放，总新生成上限**376**。每条预测保留来源及模型输出。
3. 报告每个family/episode/stage的五通道合法率、原历史接缝和夹爪事件时机、末端宽度、对专家EEF/夹爪差异；不得把180相关条件当180个独立场景，亦不称“盲TEST”。CN38数据不得事后热追加进C20模型或重新拟合统计。
4. **预登记开发目标（不是安全标准）**：全180中CLOSE≥32/36、非CLOSE≥140/144，F2 CLOSE≥7/8；且无严重漏闭合/异常先抬再关集群。达不到时报告具体family和阶段，不挑选更幸运noise或改阈值；停在离线问题，暂缓native衔接申请。

### H3 — 双adapter统一普通RGB、参考历史与生命周期；无新增物理

复用PR393的`resident_close_pair_v1.py::attach`，同一个不可变SLA父模型一次加载、两套绑定adapter；实际测显存、同步延迟、模型/统计/codec/hash、同RGB输入、同noise身份、完整H40编译器检查。现有代码已实现双adapter离线推理，但这不等于native通信和动作参考生命周期已经验证。

离线/模拟故障注入必须覆盖：上一**实际发出**的夹爪参考开度和source/query native时钟相距50ms；失效／过期／未来时间；跨episode清空缓存；H40读写一次映射；0.4s TTL与≤0.35s续期；ACK、取消、STOP、partial P10只提交实际消费，不让夹爪参考孤立运行；实体`BASE`持续M_FREE不提供oracle。模型不能使用专家stage、物体真值或未来命令补齐夹爪状态。任何端点不具备这些字段就fail-closed。

验收：固定8个旧RGB输入×2noise身份全部两分支输出与各自独立加载逐数组一致，恰好16次新分支生成的上限已在H2计算；共享父模型一次加载，服务资源峰值与延迟可核实。**原生STOP／TTL／partial P10的动力学资格仍需独立物理准入**，不能拿CPU故障注入冒充。

### S — 独立并行的安全重新准入，不受H1离线合法率自动解锁

既有1学生＋2专家的`ILLEGAL_CONTACT`完整保留；保存状态2ms被动检查在三STOP前分别提前12／34／316ms发出告警，亦会误拦**1/38**个成功专家。已有C++候选只在1204保存状态做数值核对，**未安装至live native hook**；receiver见证也不等于控制器选中的实时参考。不得因“检测到三次”便称保护器合格，更不得把改变安全阈值或篡改旧成功计分作为修复。

并行形成独立安全档案：
- 三次碰撞完整contact pair/时间、真实base/TCP/right-pad、reference age、solver所选参考身份（若无法证明，标`UNKNOWN_CONTROLLER_SELECTED`），和原STOP后积分0。
- 风险控制候选：更高净空的**新高层接近/抓取合同**或明确额外保护；任何新增改变独立命名，所有对照共同应用。不把自动偏置加给SLA仍声称其原输出成功。
- 保护候选需提交误拦代价的**事前接受准则**、动态STOP hook位置、观测年龄、最坏延迟、fail-closed策略、实际native资格计划；原1/38误拦筛选失败不可在事后改为PASS。
- 有权限的独立审核者签发与**代码SHA、MJB/布局SHA、WBC配置SHA、监测/执行profile、有限原生面板、停止与隔离范围**绑定的`READMISSION`，才允许任何新原生实验。签发前**新增专家/学生物理请求=0**。

### P — 独立安全签发后，才逐层执行原生能力评价

所有数字为**条件性申请上限**，本PR不提供运行许可；整体最多88个新原生episode，执行身份和数据分母逐项锁定。

| 顺序 | 最大episode | 进入下一阶段的条件 |
|---|---:|---|
| P0：专家安全资格 | 8（F1/F2×2审核布局×2 reset） | 0硬安全STOP；冻结WBC+新审定安全合同真实可执行；若出现STOP，按审定范围立即隔离 |
| P1：**一个已固定**的混合学生pilot | 8（F1/F2各4） | 真实抓持、≥20mm抬升、连续≥1s；至少各family 1个合格移动任务，0硬安全STOP，才考虑DEV |
| P2：新DEV | 32（每族16） | ≥24/32，且每族≥12/16，0硬STOP；选择与调参仅用TRAIN/DEV |
| P3：冻结模型＋一次新TEST | 40（每族20） | ≥32/40、每族≥16/20、0硬STOP；失败不在同TEST补seed或重试 |
| **总上限** | **88** | 所有未启动单列；未签发准入时仍为0 |

每个episode必须独立报告raw/组合H40全门、真实安装/消费点、EEF轨迹与跟踪偏差、真实base方向≥8cm、native grasp、抬升≥20mm和保持≥1s、碰撞/跌倒/停止、query与积分。辅助监测触发/防护停止不得当成功。通过本面板仍不是通用实机安全认证。

### F — 条件性新学习研究，不自动启动

若H1没有足够合法的固定组合，或H2暴露持续的事件/时序冲突，先报告失败和需要的数据，不以增加SHIFT14/SHIFT21替代归因。**下一优先模型研究**为冻结已证明较稳定的FM500 EEF分支、单独训练一个夹爪事件/目标头，阻止夹爪损失通过共享action expert改变EEF。训练时教师的stage与close_start仅可做标签，不得输入部署。以同RGB/noise下EEF与FM500逐位相同作为解耦证据。

若有新的单独批准，建议仅一个500更新／batch32／16,000访问、C20原统计的候选；真实预算不得从旧Goal剩余额度推导，也不得自动开第三、第四模型。若离线合格而pilot仍抓空，优先获得闭环学生状态下的可部署反馈/纠正证据；不能只看速度门继续调模型。CN38额外18条可作固定episode诊断，不以数据量增加直接宣称泛化。

长期扩展到F4持物移动、F5抓持搬运、F6放置和复杂门/书架任务，**需要在M_FREE真实抓持闭环已通过后另立技能数据与安全范围**；本V2未授予新技能采集/训练预算。

## 3. 可执行的数字预算与禁止重复的成本

| 本V2阶段 | 新优化更新／访问 | 新完整神经生成 | 新原生请求 |
|---|---:|---:|---:|
| H0摄取SHIFT7原task | 0新增；原已登记总500/16K（尚未证实终态） | 0新增；原任务已登记最多112 | 0 |
| H1两个保存96组合 | **0/0** | **0**（只有192次纯组合/原门检查） | 0 |
| H2 CN38＋共享重放 | **0/0** | **360+16=376上限** | 0 |
| H3接口非物理核验、S被动安全档案 | **0/0** | 复用H2，新增调用须单独登记 | 0 |
| F明确新学习分支 | **尚未授权**（建议500/16K） | 尚未授权 | 0 |
| P独立准入后的分阶段面板 | 0 | 跟随面板记录，不混入H2 | **条件性≤88，不授权** |

**本次GitHub文档动作的真实新增**优化/生成/原生均为0。SHIFT7后台和以前所有实际成本独立记账，不把各模型从共同父模型开始的500更新相加冒充一个checkpoint被训练了几千步。

## 4. Codex状态机、真正的最小下一步和交付格式

- `READ_LIVE_STATUS`：先读Core现场所有者/lease/最新PR，校验是否已有同一HYBRID任务。已有受管作业保持，禁止重复launch或重新计数；未知原生身份先核查，不能推测未启动。
- `WAIT_SHIFT7_TERMINAL`／`SHIFT7_RESULT_INCOMPLETE`：只允许已有任务收尾；无有效终态时禁止改变固定H1规格伪造二候选。
- `SAVED96_OFFLINE` → `SAVED96_PASS`或`SAVED96_NEGATIVE`；保存192个包络与控制端空间检查；不因负结果自动开训练。
- `CN38_EPISODE_HELDOUT` → `OFFLINE_MODEL_DIAGNOSTIC_COMPLETE`或`BLOCKED_SOURCE`；保存180条件/376调用真实分母，模型耗时/GPU峰值/全部failure。
- `SAFETY_REVIEW_PENDING`：离线可以有明确获准的工作，但不能触发物理；独立批准之前`NATIVE_REQUESTS=0`。
- `READMISSION_GRANTED`后仅按获批原生面板从P0启动；任何hard STOP进入`SAFETY_QUARANTINE`，绝不自动补跑。
- `GOAL_MET`仅在冻结候选DEV/TEST真实成功与安全证据完成后；`BLOCKED_WITH_EVIDENCE`、`BUDGET_EXHAUSTED`或`WAIT_RESOURCE`均为独立合格终态报告，但不是科学成功。

每个阶段写入单个append-only effect ledger/terminal evidence：**计划与实际执行source SHA、绑定模型/数据/统计、旧/新增训练更新和访问、生成及缓存分母、原生请求/tick、RUNNING/STOP/UNSTARTED、原门first rejection和峰值/阈值、任务/底座成功、readmission文件和审核人、日志及停止入口**。固定必要原始证据SHA在PR；私有模型/RGB/原生机密文件不上传。发布入口放在新的docs-only PR，执行实现仍以PR393实际封存分支为准；PR的创建/CI/静态测试都不是模型或物理成果。

## 5. 精确的下一次交付

1. SHIFT7真实终态（或明确运行中/未知）：500更新和固定96＋RGB16证据、k26/k33曲率、旧/新全阶段对比；
2. 已有`hybrid_saved96`是否已执行（真有则复用，不重复）及两组合的192条H40结果；原FM全部空间四通道shadow；
3. 若H1通过，CN38 18条episode／180条件／最多376次实际神经调用、按族/阶段/episode分布与跨query动作协调；
4. 独立可核对的三次碰桌归因、监测误拦1/38和未准入状态；提交给安全责任方的是`READMISSION_REQUEST`而不是已签发的放行；
5. 只有获准后才给真实移动pilot/DEV/TEST成绩；未获准就写`BLOCKED_SAFETY`和具体尚缺的证据。

**优先级：真实终态 > 已存预测组合 > 新episode离线 > 双包部署验证 > 独立安全准入 > 真实任务 > 新模型训练（仅在必要且获授权时）。**

参照实现：[PR393 HYBRID规格](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/e3fb6005597a9329fa886c3015e9c5fa7f92afbe/docs/goals/2026-11-17/MFREE_EEF_GRIP_DECOUPLING_AND_SAFETY_V1.json)及对应已提交代码；[SHIFT7部署事实](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/e3fb6005597a9329fa886c3015e9c5fa7f92afbe/reports/WM02/sla-time-layout-shift7-20261117/STARTED_ACTUAL.md)；[TIME500真实终态](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d96ae066d21d84abf901da5c8de83e3b32d1a9f4/reports/WM02/sla-close-safe-mobile-20261116/TIME500_BOUNDED_FAILURE_ACTUAL.md)；[安全候选负结果](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d96ae066d21d84abf901da5c8de83e3b32d1a9f4/reports/WM02/sla-close-safe-mobile-20261116/SAFETY_CANDIDATE_IMPLEMENTATION_ACTUAL.md)。
