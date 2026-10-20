# POST-CR1 长程 Goal：自助修复执行链，推进共享策略，验证固定新范围的数据试点

日期：2026-10-20，Asia/Singapore。Goal ID：`POST_CR1_POLICY_DELIVERY_AND_SELF_REPAIR_V1`。
审核基线：`e710f1507de15f365ea0142c9c5f9242aa9798b1`。本文件是单一自包含接续入口，不创建第二套调度体系。

**用户本轮指令：制定长程计划，写入GitHub，工程阻碍自助修复。** 现有PV1范围内的工程诊断、修复、验证、合法冷态续接不再逐项请求批准；原科学规则、站点权限、累计预算与截止不因此扩大。CR1/COV1已关闭；使用新范围开展新的物理/训练属于新的科学工作，下文给出一次性可批准的CR2包，未有可核验明确批准前不得执行。不要把整个Goal标成等待GO而停止已经获准的PV1工程修复。

## 0. 三条轨道及完成定义

| 轨道 | 当前权限/状态 | 目标 |
|---|---|---|
| E：工程恢复 | 用户已要求自助修复，使用原PV1有效工程权限和剩余额度 | child11结算；认证从排队到release可靠；下一合法allocation产生新的有效B0结果 |
| P：现役PV1 | 原GO继续，139条新银行/R0统计/配方与分母不变 | B0全48、A_CONTROL/B_COVERAGE各4096真实更新与匹配DEV、合格唯一候选的原验收及可运行包 |
| D：CR2_FIXED_R1_BANK_PILOT | **PROPOSED_PENDING_EXPLICIT_SCIENCE_GO_AND_NATIVE_BINDING** | 锁定CR1唯一候选，在12组/24次新TRAIN任务中验证数据生产与独立新银行；不训练模型 |

CR1不是新的全局前置，也不再复跑R0–R4。PV1有GPU即优先推进；不等CR2结束，不向PV1热注入R1。科学闭环可以给出负结果，但必须有真实训练/执行/对照证据；CI、提交数、CPU检查和Teacher成功不等于policy改善。

## 1. 已核实证据与必须保留的不确定性

来源：固定CR1七工件目录 `reports/PI0294/CR1/20261020T105832Z-localpc-cr1/`；科学#11/6290698383、工程#47/6290699139（2026-10-20 20:20 SGT）。这些是发布时的原生收据，不是当前实时SSH/qstat。

- CR1结论 `RANGE_SUPPORTED`：191个原4mm、安全完整成功且raw合格的旧TRAIN parent，96几何lineage；141 FIT、50 DIAGNOSTIC，另3个已打开的KNOWN_CHALLENGE。DIAGNOSTIC恢复7个旧FAIL、覆盖6个与FIT不重叠组；旧139条新数据银行的codec检查无回归。139是重叠retention角色，不另计139条独立样本；不代表全部旧预训练数据/C1/任意未来任务都已验证。
- 主档案R0 168通过→R1 191通过，已知挑战另有1个恢复；两版本各检查116594个H40窗口。R1最大XYZ约1.08455e-8m、SO3约4.71216e-8rad，action inverse-clip 76→0；state preclip 1424保持，不隐藏该输入分布风险。
- 原AP0/-Y/NEAR/T2的anchor209/action31复现：float64+原clamp仍失败，float32+隔离no-clamp通过；active近常量mask为空。支持该案例的clipping损失，不将所有旧碰桌、任务失败或其他codec问题归为同一原因。no-clamp只用于已完成诊断，不进入新生产。
- 唯一新统计候选 `R1_TRAIN_ENVELOPE_PAD5`，**序列化statistics SHA256 `50d9672397fd0b5353da2011b9f9f568dc9c65c26fb271458f87d60b2d497fa8`**，只变 `new_embodiment.relative_action.eef_9d.min/max`。候选报告路径 `CR1_CANDIDATE_STATISTICS_AND_EFFECTIVE_PARAMS.json`；必须从注册原件取得对应完整统计文件并比对，不能把参数摘要当完整processor文件。
- 原R0 stats SHA256 `fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7`；PV1银行manifest `60f7c35a05766fe23974eba3a72f53ca4db5545a3768e6c1cfea164ab4e58294`；共同父CORR2048 manifest `94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38`。原官方代码 `51d4c89f72fda44cbf77285c6a8114b52676b8a1`。均不得改写。
- CR1运行源 `8e2097ed2513aecc205d71532f49b72f5cf0b08f` / Actions39412546118成功；交付源e710f150 / Actions39421021426成功。CR1已关账，无新物理/神经/训练，不能复活其lease。
- PV1 child11=`678230.hopper-m-02` 已F/exit1，不是仍在排队。原native stime19:37:53、obittime19:41:11、walltime00:03:08；owner报 `PV1_PROJECT_AUTH_EXPIRED_15006_EXISTING_AUTH_ONLY`，wrapper报 `PV1_OWNER_FRESH_PROJECT_RELEASE_MISSING_NO_IDLE_MODEL`。0模型加载/0新query/0物理/0更新，但有GPU实际分配，不能记零费用。
- 两个child11 lease仍ACTIVE待原PV1结算；H200预约28860秒、Core CPU预约43200秒待核销。此前已结算20275 H200秒、10次PBS、1054 queries只是旧基准，不含本次待结算实际消耗。B0仍14/48=9成功+5有效失败，剩34；两臂优化更新0。
- 原DATA-COV1 2/4<3的FAIL、原四次尝试和44 NOT_RUN、关闭账本与所有收尾结果保持。此前NEAR0.12–0.15与评分器至少0.15输入合同冲突属于计划遗漏，已批准新NEAR0.15–0.18；不能回评分或删旧工程未知让旧门通过。

## 2. E0：立即完成旧作业结算和单owner恢复

这是下一步，不是先提交child12。复用AGENTS、START_HERE、统一入口/doctor/registry、原PV1协议和本用户自助修复授权。先读取#11/#47最新native结果，若已有后继运行，接续它，不按本文快照重复提交。

1. 核对678230的job owner/project/request/nonce、真实F/Exit、起止时间、wrapper/ready、未出现START_NEURAL及原生成/安装账本、两端进程/租约、所有pending/unknown。原owner曾failed，不能假定健康；只有确证旧调度者已退出或完成排他交接，才恢复既有唯一owner角色。认证helper不是第二位scientific dispatcher。
2. 通过原 `scripts/pi0294/pv1/pbs_owner.py:settle` 或其现役等价路径，按真实walltime与stime/obittime及原保守计费规则一次结算。资源读侧证据足够时自主恢复结算；不要手工把used清零或直接删除reservation。Core控制/CPU消耗单独结算，释放已终态旧预约，关闭两端lease。
3. 旧STOP、失败源、日志、原费用和manifest保留。输出 `PV1_CHILD11_SETTLED_AND_CLOSED.json`，含真实已用/释放/剩余、各lease哈希、unknown为空及无重复效果。只有这些通过，才允许prepare一份新的合法续接。
4. 如站点查询/认证暂时失败，先进行E1允许的认证修复再核销；不将CR1 CLOSED当成PV1也已关闭，不用CR1资源替PV1买单。

## 3. E1：修复认证生命周期，而非只在提交前登录一次

代码依据：`pv1/pbs_owner.py:project`在原生amgr返回15006时拒绝；`release`要求准确owned R job、PBS_READY时间小于175秒、fresh项目证明和唯一START_NEURAL。这个原安全身份门保留。问题是认证有效性未可靠覆盖Q→R的时刻，并非模型容量问题。

### 3.1 必须实现或核实的运行行为

- **提交前**：在站点允许的登录环境，使用用户自己既有、仍获授权的认证方式，验证project active、成员资格、cluster、余额和稳定个人quota。记录认证有效期/剩余期（站点可提供时），否则明确UNKNOWN并使用站点支持的健康检查；不要编造TTL。
- **排队期间**：原owner保持生产心跳；允许维护同一项目的可续期会话。续期频率由真实有效期与站点限流确定，采用有界退避，不能等获卡后才第一次检查。确认helper和实际release进程使用同一principal/同一有效认证上下文；“helper登录成功但release读另一个缓存”必须在本地流程检查中被捕获。
- **收到R/PBS_READY时**：进入高优先级的短路径，立即读准确job身份和新鲜项目资格，在原175秒deadline内生成本job证明并唯一release。不要同步做全仓CI/大文件哈希/全树du而错过放行。账户证明仍须现场新鲜，不能把Q期间缓存证明冒充R时证明。
- 认证失效但既有站点方式能够自助续期时，允许有限一次修复循环并再次fresh查询；只有仍有足够时间通过原release门才继续。超时或权利失效按原规则停止，不延长175秒、删除guard或在compute节点执行禁止的amgr登录。
- 只使用用户既有合法凭据/密钥管理和站点支持流程。不读取无关秘密、不把密码/token放命令行、源码、Git、日志或普通run manifest；不绕过MFA/CAPTCHA或伪造证明。如只能由用户完成一次交互认证，报告 `HUMAN_AUTH_ONCE_REQUIRED` 和准确待操作环节，其他已授权CPU工作继续。真实账号权限撤销/余额不足不是可“修好”的软件bug。
- 明确处理进程重启和证明已写但START_NEURAL未写的崩溃窗口：按原nonce、文件hash、owner与时间证明幂等恢复或安全终止；不能盲重调release、覆盖现有证明或生成第二次放行。

### 3.2 不新增H200的回归检查

使用依赖注入的clock/project/query与CPU fixtures，覆盖：Q等待跨认证过期、续期后R正常放行、helper与owner认证上下文不一致、ready接近175秒、认证失败后owner仍可安全结算、重复callback/重启不重复release、资格被撤销仍拒绝。fixtures不能调用真实qsub/模型/仿真。

在真实授权登录环境做一次有限健康往返，证明实际release进程读到的身份有效；不能把mock PASS当真实认证。不要申请一整份H200只为了重新测这个错误，下一份合法续评同时完成其本allocation原生资格和B0任务。

### 3.3 修复循环与防止无效重投

同一工程故障指纹最多3轮有实质差异的CPU诊断/修复，每轮保留复现、patch、针对性测试、精确CI和冷态部署证明；达到上限后输出最小复现及具体阻断，不重复烧PBS。同一根因第二次导致原生premodel失败时，停止该根因下的新qsub，直到增加了能覆盖它的回归证据。原科学负例不适用工程重试。

候选源码普通提交、direct-push精确head_sha Actions+check-run核验；不要只调用限PR事件的workflow查询误判CI缺失。完整发布/readback后才冷绑定新source，运行胶囊不热换。下一child名称/nonce从真实registry分配，本文不预占child12或jobID。

## 4. E2：自助修复权限清单及边界

| 可在现有权限/预算内自主完成 | 必须保留的约束 |
|---|---|
| 登录会话续期、同用户认证上下文、quota稳定读取、项目资格读侧恢复 | 不绕过认证/成员/余额、不开新项目，不在禁止位置登录 |
| 路径、依赖、final binder权限、完整银行、官方模型alias、元数据可见性 | 验证原件身份，不用重新算hash掩盖变更；一般输入按schema防symlink，必需模型alias则保留正确的合法symlink，不能统一复制解引用 |
| NFS/文件读取瞬时异常、关闭FD、缓存/控制面I/O和ENFILE读侧修复 | 有界、幂等，原watchdog不放宽；未知qsub/install/query/optimizer效果先journal核清，不能重发 |
| 全状态checkpoint/采样游标恢复、CPU-only性能实现、原子日志写入 | 原optimizer/scheduler/RNG/数据顺序和实际消费不变，A/B同配方；不能只保存weights冒充resume |
| 原作业终态计费、关闭lease、单owner连续性、合法剩余条件cold续接 | 单卡、原截止/配额/24PBS；有效任务失败与已完成条件不重跑 |

不在此工程授权内：WBC/腿/plant/夹爪PD/摩擦/solver/gain/cost，参考门/S1/评分/统计量/H40/采样配比、B0训练前门、测试分母、额外GPU/并发/预算/截止，新的物理研究或R1模型训练。

派发前按**最后一次写入后的实际目录**检查0400等封存权限、owner、完整RGB/numeric而非metadata-only、原绑定hash/别名、可用FD与可写输出根。小错误在CPU解决，不带到新H200分配后。没有失败checkpoint清理必要就0删除；任何清理只针对已授权、无活跃消费者/引用且非唯一恢复或科研证据的产物。

## 5. P：沿原PV1尽快产出真实模型

E0结算、E1/E2验证、fresh所有站点/预算/单owner门通过后，允许原owner在原已批准PV1内自行prepare→最多一次qsub→原生job回读→其自身资格→仅剩34个B0格；若现场已有更多有效格则按最新remaining清单减少。原工程替补资格逐条件核验，不因新job重复已允许的替补。未运行格保留NOT_RUN。

B0完整48并独立审计/计费关账是现役A/B训练前硬门，不增加B0成功率门。训练前核对完整后续成本而不以“还有GPU小时”替代资源可完成性。

- 同父CORR2048、R0统计、相同seed/原flow loss/BF16/effective batch32，A_CONTROL/B_COVERAGE各4096实际更新；1024原TRAIN-only探针、完整冷态resume，两臂独立optimizer。VLM冻结，当前139条新银行/原C1配比和各任务曝光不变。
- 两个4096终点分别完成同一DEV48，给出B比A、各自比B0的task/scene/方向/噪声、H40参考拒绝、S1、完整任务终末和成本；不以单条视频或loss选模。
- 原DEV资格每任务至少18/24、不相对B0任一任务下降超过1/24、最弱任务净改善，按原排序选唯一候选；合格且预算足够才打开LOCKBOX80。P3默认跳过，独立seed仅在保护最终验收后可支付时执行；无复核最多BETA，不叫VALIDATED。最终32/40或28/40等全部原分级门引用现役协议，不事后降低。
- 原PV1预算96H200h、24PBS、同时1卡、单job≤8h、至少20h final保护。所有工程失败照实收费，queued reserved不当spent。原T0=2026-10-19 02:29:56 SGT；10-25 02:29:56停新job/候选、14:29:56停新训练、18:29:56停新物理，10-26 02:29:56全收尾。新Goal不延长。
- 若准确预测证明原B0训练前串行门使主线无法按时完成，一次提交有实测成本的狭义科学修订提案（B0完成移到选模前、数据/配方提前冻结），**未经单独批准不能自行实施**。也不要隐瞒可预见的日历阻断，直到最后才说没有policy。

## 6. D：CR2 固定R1新银行/任务Pilot（新的科学包，待一次批准）

### 6.1 只回答一个问题

在不再拟合候选、不改变Teacher与真实任务执行的情况下，固定R1能否支持新的方向/外观/距离覆盖，并成为完整可读取、可封存的数据版本？

**物理任务由同一个原4mm Teacher执行，统计量只用于保存后的数据转换。** 所以不能把Teacher成功率变化称为R1改善控制；同一raw分别走R0/R1，不需要执行两遍物理。CR1已完成的194个parent不重新作为盲测；只复用其证据和固定数据身份。

CR2_STATUS=`PROPOSED_NOT_AUTHORIZED`。发布本文件/工程自助修复不等于新科学GO。一次批准 `CR2_FIXED_R1_BANK_PILOT` 后，D0→D3在原定上限与阶段门内自动推进，无需每条示范再请示；未批准则轨道E/P正常进行，CR2只保持准备提案，不开新lease/构造/物理/传输。

### 6.2 D0：先封存数值和全链路输入合同

- R1固定到上述50d967...完整统计字节和有效norm_params，R0对照原字节不变；原clamp保留，state/gripper/模态/EEF10/H40/语言和相机不变。禁止重拟合5% padding、扩大到全新范围或用新pilot结果修候选。
- 新namespace `PI0294/CR2_FIXED_R1/TRAIN/...`，独立manifest/数据版本/新lease；旧PV1的STATS/96组/命名空间guard保持，不能给旧export函数传新stats却沿用旧身份。新增薄scope适配复用原capture/export/processor/audit；helper必须显式区分R0/R1，继承字段不能把R1写成“原v3.0未改”。
- 在零积分CPU上检查source-goal评分输入域：原0.15m边界两侧/等值、合法schema、目标可达/接触路径、Teacher阶段目标、原安全/评分与图像域。采用已批准COV1 NEAR-v2范围0.15–0.18m，FAR0.19–0.22m；不能再次拿0.12m进入冻结评分器。几何生成与评分使用同一欧氏距离定义和数值单位。
- 保存数据的新R1银行完整成员/hash/实际图像reader和跨parent混batch验证，复用CR1已合格数值证据而不重复整个研究。191主档案+3已知挑战最多194个候选；旧139 retention并不覆盖所有original/C1训练来源，未来要使用的额外旧TRAIN/C1必须另行进入完整资格清单。最多检查400个保存parent，真实缺失不填PASS。
- CR1 DIAGNOSTIC若进入未来训练库，明确promotion为TRAIN并保留原CR1角色，不再用来声称未来未见泛化。PV1 DEV/CONFIRM/LOCKBOX不进入拟合、训练或候选选择，新未来模型评价人口必须另行封存。

### 6.3 D1：12个全新几何组、24次预注册任务

设计为3个既有AP外观profile ×2方向(+Y/-Y)×2距离层(NEAR/FAR)，每层1个独立新组，各T1/T2，共24次首尝试。沿已批准COV1-v2外观与XY/yaw包络，不加入新物体、障碍、相机扰动或12mm参考。准确坐标/seed先封存，以新namespace规范hash生成，每槽最多32个静态候选，只允许原静态IK/碰撞/视野/评分输入域或几何排重失败换下一个；不根据Teacher或policy成功重抽。

与旧TRAIN/C1/FD2/CR1/COV1和受保护评价split几何排重；只用已允许的几何排重索引，不读取评价图像或结果。一个槽静态池耗尽，报告可行域不足，不改WBC、缩数据分母或无限搜索。所有同实际几何兄弟保持同一split。

前四次从AP0/-Y/NEAR和AP2/-Y/FAR的新组各T1/T2取，计入24内，单worker先做，不另加benchmark。**新协议将工程有效性与数据产出分开**：只有真实安全/身份/时序/unknown硬故障阻止继续；有效任务失败/codec拒绝保留，可继续预注册面板以估计产出，不再把一次工程输入错误当成Teacher失败概率。若预定最终门已数学不可能满足，可提前停止并将其余标NOT_RUN。旧COV1 3/4门和FAIL绝不改写；新规则只有CR2明确获批才生效。

### 6.4 D2：同一raw的R0/R1转换与完整银行资格

每条原32s/500Hz/20Hz双摄、640提交参考与641实际观测，S1 shadow照实计数；期望提交参考不等于未来achieved pose。真实物体成功、native安全、独立raw/时序/图像、全部601窗口原XYZ1e-5m/SO3 1e-4rad/gripper1e-5/relative-SE3 1e-8门全部通过才可准入。原门从固定实现读取；不靠打印四舍五入或更有利角度公式判PASS。

新raw分别以固定R0/R1编码，逐parent保存PASS→PASS/FAIL→PASS/PASS→FAIL/FAIL→FAIL及首错anchor/channel/clip、实际物理事件；state preclip单独披露，不隐式修改输入normalizer。R1六锚点真实reader、至少两个独立parent的mixed batch和确定重复须通过。不以纯数值检查代替实际帧读取。

预注册CR2数据READY门：24个计划条件均有明确状态；每任务至少10/12完整任务成功且至少10/12最终全准入，至少9/12实际几何双任务配对；每个AP×direction组合每任务至少1/2准入；新原PASS不得在R1下变FAIL；native故障0、unknown0、全员原始证据合格。该门仅为小数据可用性，不是policy成功率或总体可靠性估计。缺失/工程未完不计成功，但不叫有效任务失败。

每个实际开始最多一次，不重播已执行失败，不拿旧COV1未跑44格当本次续跑。工程在构造/积分前失败且原生证明0效果时可以修复后启动该未尝试格；一旦产生物理，保留原attempt，额外物理replacement上限0。所有失败成本计入本次。

### 6.5 D3：固定交付，不自动扩采/训练

如门通过，封存 `CR2_FIXED_R1_P12` 新24任务的实际合格子集，以及单列的可选 `CR1_ARCHIVE_R1_UNION`。候选归档194+新24的绝对上限218个parent，不是218新场景，更不是已获得218成功；实际数量与去重几何以新manifest为准。

RAW_CLOSED→AUDITED→CODEC_QUALIFIED→BANK_MANIFEST→PACKED→固定私有存储revision全成员回读→READY_LAST；不存在远端完整银行时只能报LOCAL_FROZEN。metadata-only不是可训练库，后续Hopper实际消费另需资格。旧hardlink inode不chmod/写入，原银行/统计只读，秘密、RGB、WBC私有资产与未公开权重不入Git。

若新数据门未过，交付全部真实attempt、误差与覆盖，不缩小面板伪造迷你成功版本。CR2最多到新数据版本，不自动进入72/144扩采、BF16兼容、GR00T训练、DAgger或world model。

### 6.6 一次性待批上限（不从旧关闭scope转账）

| 资源 | CR2_FIXED_R1_BANK_PILOT累计上限 |
|---|---:|
| CoreLab CPU，含静态/采集/审计/打包/控制 | 32核时 |
| local协调CPU | 4核时 |
| 新独立日历窗口 | 48小时，批准并实际采用时记录T0 |
| 并发采集 | 先1后最多2个独立单核worker，原单owner |
| 新物理任务首尝试/物理replacement | 24 / 0 |
| live / shadow ticks | 384000 / 384000 |
| 全部积分（含重建） | 768000 |
| 静态构造 / 动态及counter/clone构造 | 1000 / 2400，总3400 |
| 保存数据parent检查 | 最多400，不拟合新范围 |
| 峰值新增输出/银行/打包/暂存 | 48GiB |
| 新传输 | 16GiB |
| 新Hopper/PBS/神经forward/query/optimizer | 全部0 |

T0+40h停止派发，+44h停止物理/大计算，+48h关账；PV1优先暂停和会话重启不重置。现有实际renderer若为Mesa llvmpipe则照实披露，若改变或使用图形GPU必须有站点授权/计量，不能暗换渲染后宣称数值不变。PV1获卡时停止新数据派发并在安全边界让资源，不抢占原B0保留CPU/磁盘/通信。quota/全局控制权限更小或更早者优先。

批准必须明确包含新的R1数值数据版本及这24次物理，不把工程自助修复变成新的科学授权。一次有效批准覆盖D0–D3和同范围必要工程修复，无需逐parent/子租约重复询问。

## 7. 从新数据走到Policy：只选一个下一训练问题

CR2或当前PV1完成后，根据实际结果提出**一份**后继神经GO，不马上同时做四臂实验：

- 若先要确认R1数值坐标适配：同一批两版本共同可表达的TRAIN数据、同父权重warm-start、独立同配方optimizer/更新量，比较R0与固定R1。此干预也改变了归一化target和物理误差对应的loss权重，不能把全部收益叫“消除clip”；不使用R0无法表示的标签来故意削弱对照。
- 若要确认新增覆盖收益：两臂共同用已经合格的R1处理器/统计量、同父初始化与训练量，仅改变数据支持，分别报告旧能力保持和新交叉方向场景。处理器适配与扩数据收益不能混为一谈。
- 旧CORR权重直接装R1不是原B0，也不是零训练修复。必须验证真实BF16/flow loss、梯度finite/非零、mask、模型加载、C1及全部计划来源覆盖、完整恢复及独立闭环。float32 codec PASS不能替代这些证据。
- 新模型仍是一份共享T1/T2权重。使用预冻结新DEV和独立最终评价，原S1/评分/WBC不变；交付模型卡、统计量版本、完整失败/成本与offline入口。CR1的DIAGNOSTIC/known challenge、已打开PV1结果不能包装成新盲测。

如果当前PV1首先达到原policy门，应优先发布，不等待新R1路线；如果当前PV1失败，已完成的CR1/CR2为下一轮提供具体数据与数值证据，而不是再开始泛化的无边界分析。

## 8. 自助执行循环、资源与回执

原唯一owner的循环：刷新实际状态→选择一个最关键阻断→CPU最小复现→最小同语义patch→针对性测试/精确CI→cold绑定→一次合法原生效果→独立回读/结算→继续。无需每个普通工程修复再请示；没有实际进展时不靠反复运行整仓测试、制作重复文档或新增监控器刷成果。

可自动处理失败，不能自动认定成功。账号需交互认证、原权限撤销/预算不足、真实安全、未知动作副作用、新科学变量和停止时钟，是准确上报人类的有限边界。区分 `RETRYABLE_READ_ONLY`、`WAIT_NATIVE_CAPACITY`、`HUMAN_AUTH_ONCE_REQUIRED`、`RESOURCE_BLOCKED`、`SAFETY_OR_UNKNOWN_STOP`、`NEW_SCIENCE_GO_REQUIRED`；某一路blocked不全局停止其它独立获准工作。

现owner首次回 `POST_CR1_GOAL_ADOPTED` 或实际访问阻断，附：child11结算/lease状态、真实当前job/owner、B0有效n/48、实际优化器更新、spent/reserved/unknown、认证生命周期修复方案与下一合法动作。CR2未批只需一行 `CR2_AWAITING_SCIENCE_GO`；不能把它当不做E0/E1的理由。

建议新增的最少收据：`PV1_CHILD11_SETTLED_AND_CLOSED.json`、`AUTH_LIFECYCLE_REPAIR_AND_NATIVE_READBACK.json`、`NEXT_NATIVE_ALLOCATION_AND_B0_PROGRESS.json`；若CR2获批则加 `CR2_SCOPE_POPULATION.json`、`CR2_R0_R1_PAIRED_ATTEMPTS.json`、`CR2_BANK_GATE_AND_READER.json`、`CR2_RESOURCE_FINAL.json`。沿原run根和#11/#47报告，不另建第二owner或永久cron。每个重要新结果及原生异常合并发布固定SHA，不频繁刷无变化快照。

## 9. 发布声明

本文件依据上述固定源、CR1报告和原 `pv1/pbs_owner.py:project/release/settle` 制定。原 `GR00T_TASK_DATA_FIRST_FROZEN_WBC_SCOPE_V1.md` 的职责界限继续成立。本次GitHub写入只发布Goal，不表明远端认证已恢复、旧预约已结清、后继PBS已提交、CR2已获批或policy已改善；这些都需原owner实际收据。

**当前立即执行：工程E0→E1→E2→原PV1；未来数据执行：只有一次明确CR2科学GO后D0→D3。最终以真实共享权重、匹配任务结果和独立验收交付，不以更多计划替代模型。**
