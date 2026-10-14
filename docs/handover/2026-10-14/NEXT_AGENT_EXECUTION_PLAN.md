# GR00T＋WBC 新 Agent 接管与科学推进计划 V1

日期：2026-10-14。审核基线：`7e7773a5628d7f119076220ac94bb92a1a9ce120`。

**状态：接管计划已形成；科学执行待用户明确采用。** 本文不是夹爪修订的采用记录，不启动模型、物理、PBS或清理。它整合既有章程、交接和待采用方案，不重写旧 FAIL、不自动续用旧截止或余额。

## 0. 给新 Agent 的核心指令

你是替换已经停止的旧 local-pc owner 的唯一接管者，不是新增并行调度 owner。最终目标是同一份 GR00T checkpoint 在固定底座 ARX5 仿真中，按语言分别完成抓放 T1 和推送 T2；随后才研究接入师兄提供的冻结 WBC。

**当前优先顺序：只读接管 → 获得明确科学 GO → CoreLab 有界夹爪归因/修订/资格 → 同一 M512 四条新观测闭环 P1R → 原 v3.0 完整续训 M2048/M4096 → 固定 DEV 选型 → 核心确认和旧任务回归。**

不从头重建场景，不重采全部数据，不重训 pristine，不改用 v3.1，不把源码归档当作可运行部署，不重新整理全仓库来替代科学进展。

## 1. 科学问题与本轮成功定义

研究章程：`docs/RESEARCH_CHARTER_TASK_SPACE_MANIPULATION_V1.md`。

高层学习 RGB＋语言＋实测 TCP/夹爪 → 任务空间 EEF 轨迹与夹爪命令；不学习目标机器人 joint action。任务数据不依赖目标机器人遥操作或目标 WBC 成功筛选。同一冻结低层是否能承接多个新任务是待验证的系统命题，而非当前已有结论。

当前两技能必须由同一 checkpoint、同一 processor/stats、同一执行器完成。任务 ID/教师阶段/物体真值不能作为学生隐藏答案。不同位置或颜色不是不同操作技能。Teacher 成功、离线 SafetyGate PASS、CPU CI SUCCESS 均不等于学习策略成功。

本轮实际执行器是独立固定底座 ARX5，**不是师兄 WBC**。夹爪 D1 修复若成立，属于执行器资格改进，不能归为神经微调收益，也不证明跨本体、移动 Go2 或真机成功。

## 2. 已核实起点：直接复用，不退回起点

| 项目 | 固定证据与状态 |
|---|---|
| 仓库 | `Leo-6-maker/go2-arx5-hierarchical-vla` |
| 集成分支 | `pi0267/executor-decoupled-recovery`；main 不是现役运行源 |
| 交接基线 | `7e7773a5628d7f119076220ac94bb92a1a9ce120` |
| 实际 R1 科学 runtime | `021d813fd669561a24ffbf89431a4a9812e0f2ca` |
| 神经父实现 | `a171997a704072e127d16410ba584c8aa0ddd573` |
| 原物理实现 | `f1fdf3acd481687cb275334ba512ce038eba51e9` |
| 保存夹爪审计 sidecar | `aa9664cd7e55773ad7c6a91a561944f3df8bc04f`，不等于新 runtime |
| 原模型 | M512，512 更新、16384消费样本，T1/T2各8192 |
| v3.0 | 158 episodes；TRAIN 75726 / DEV 19232 个H40；HF交付和CPU资格通过；M512只用它 |
| v3.1 | 295 episodes；TRAIN 149048 / DEV 28247 个H40；本地CPU合格，未HF交付、未训练 |
| 数据生产 | 224新首次尝试全部终态，215准入、9拒绝保留；v3.1包含v3.0 |
| CPU并行 | 串行/2/4 worker约17.194/31.600/49.171准入episode/h，4worker约2.86倍，不是GPU训练基准 |
| 模型行为 | 48条真实离线H40；首个T1闭环6次实际查询，5个完整prefix ACK，第6块部分执行后原生限位停止 |
| 当前科学状态 | `BLOCKED_REAL_NATIVE_FINGER_JOINT_LIMIT_NO_P1_GATE`；selected/deployment=NONE |
| 未完成 | T2模型行为、剩余开发case、M2048/M4096、DEV选型、独立确认、旧任务回归、R1完整优化器恢复 |
| 工程 | 762测试通过、14显式跳过、58subtests；CPU CI38521034429成功，不替代物理或神经资格 |
| 166份owner源码 | 带脱敏路径和停止保护的`.py.txt`参考；不可移除guard后直接运行 |

上一轮 R1：`20261014T025730Z-localpc-pi0294-r1`。交接时2次PBS均终态、890 GPU allocated秒、55 query意图（1 pre-forward失败＋48离线＋6在线）、新增训练0；CoreLab 1 start/1081积分/44 RGB pairs，charged CPU 5240.044481548秒（含保守上界）。全部预留为0。以上是交接快照，接管时仍须现场刷新；不要将它写成新agent已经SSH复核。

原父 PI0294：`20261013T142457Z-localpc-pi0294`，4 PBS、16工程更新、512正式更新、3669 GPU allocated秒。旧PI0292无模型训练，旧PI0291仍为56TRAIN的PARTIAL。新任务不篡改任何旧账。

## 3. 阅读顺序与最短代码地图

### 3.1 先读这些，不要从所有旧PI开始全文重审

1. 同目录 `HANDOFF.md` 与 `CODE_AND_EVIDENCE_INDEX.json`。
2. 根 `AGENTS.md`、`START_HERE.md`，以及 `docs/operations/2026-10-13_UNIFIED_ENTRY_AND_GITHUB_SYNC_V1.md` 和 `WORKSPACE_ENTRY_ENFORCEMENT_V1.md`。
3. R1目录中的 `FINAL_REPORT.md`、`RESOURCE_LEDGER.json`、`M512_POLICY_EVAL.json`、`SAVED_GRIPPER_TRACE_ACTUAL.json`、`GO_TASK_CARD.json`。
4. `docs/protocols/PI0294_R1_GRIPPER_SAFETY_REQUALIFICATION_PROPOSAL_V1.md` 和 `reports/PI0294/design/GRIPPER_SAFETY_AMENDMENT_TASK_CARD.json`，均仍未获采用。
5. 父PI0294的 `TRAINING_RECEIPT.json`、`STRICT_REQUALIFICATION_ACTUAL_PASS.json`、`FIXED_MODEL_EVAL_PANEL_RECEIPT.json`、`DATASET_MANIFEST.json`、`CHECKPOINT_MANIFEST.json`、`RESUME.md`。
6. 用户提供的本机私有交接文档。精确路径仅在本机/对话中取用，不复制到GitHub、公共HF或公开日志。
7. #11/#47交接之后的增量评论；按实际分页读完。最新明确人类决策优先于过期说明，未知则保留UNKNOWN。

`RESUME.md` 的历史段落可能仍写“未提交/认证待完成/没有模型推理”，以最新终态收据覆盖，不能照抄旧“下一执行”命令。

### 3.2 修复时只沿相关模块进入

| 功能 | 入口 |
|---|---|
| 夹爪控制/安全 | `scripts/pi0291_arx5_data/engine.py`、`config.py`、`scene.py` |
| 现役场景/继承关系 | `scripts/pi0294/scene.py` → 原Engine.step |
| 物理/原始记录 | `scripts/pi0294/episode.py`、`audit.py`、`score.py` |
| 保存命令算术 | `scripts/pi0294/saved_gripper_trace.py` |
| 已验证模型/训练 | `scripts/pi0294/native_train.py`、`neural.py`、`numerics.py`、`sampling.py` |
| 模型驻留与真实反馈 | `resident.py`、`relay_endpoint.py`、`relay_transport.py` |
| 作业/版本/预算 | `allocation_admission.py`、`run_admission.py`、`continuation_session.py` |
| checkpoint | `local_checkpoint.py`、`checkpoint_session.py`、`public_checkpoint.py` |
| 早期正式Trainer适配 | `scripts/pi0292/trainer_adapter.py`、`train_adapter.py` |

不重放历史init/freeze/delete/submit/closeout helper。需要复用其中逻辑时先读源码、核对副作用，抽取最小函数到新版本；不直接运行166份归档脚本，不复制里面的旧时间与私有绝对路径。

## 4. 机器、入口与权限

| 位置 | 职责与限制 |
|---|---|
| 新local-pc owner | 唯一调度/发布/SSH中继；替换旧owner，非同时新增owner |
| CoreLab AMD | NFS管理和可用CPU检查；不是4080S本地SSD的替代路径 |
| CoreLab 4080S | 合格MuJoCo环境、物理、Teacher、评分、视频和原数据 |
| Hopper | 仅合法PBS allocation内进行神经加载/推理/训练；不在login或CoreLab偷跑模型 |
| GitHub | 自有源码、协议、CPU测试、小型脱敏证据 |
| HF/私有存储 | 按明确权限保存数据/模型；公共M512不意味着raw、WBC、视频也可公开 |

CoreLab人类入口保持 `~/Projects/GR00T-WBC`。已有 `infrastructure/enter.sh`、doctor及 `amd-management`、`4080-simulation`、`4080-cpu-data` profiles 可查验复用。通用 `workspace doctor/prepare-run/sync` 公共包和通用Hopper入口**仍未完成部署资格**；不要调用不存在的CLI或把实现它作为科学前置工程。

沿用已合格、私有绑定的Hopper启动器，合法授权后用独立新run封存相同数值实现。Core/Hopper不直接通信；两条既有SSH由local-pc中继。不新开端口、服务或隧道。未知主机返回HOST_REQUIRED，不猜路径。HF/GitHub/PBS认证互相独立，Git SSH成功不等于amgr有效。

## 5. 模型和数据的固定身份

| 对象 | 固定身份 |
|---|---|
| 官方GR00T源码 | `51d4c89f72fda44cbf77285c6a8114b52676b8a1` |
| 官方基模revision | `2fc962b973bccdd5d8ce4f67cc63b264d6886495` |
| Cosmos revision | `9ce19a195e423419c349abfc86fd07178b230561` |
| v3.0私有HF | `leoyuer/pi0291-arx5-fixed-base`；prefix `PI0294/20261013T142457Z-localpc-pi0294/bank-v3_0-2fd28e00` |
| v3.0 READY | `6aa94f4f40d791bedd6b4e75489e4ede6504d10d` |
| v3.0 manifest SHA256 | `b8ddcb9808e6dfac4f3ac5fa5a4828128fa9644a330c17d1c57bf833ce5889df` |
| v3.0 stats SHA256 | `fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7` |
| 原私有M512 inventory | `6fdc10c2cfb13841bbabd444aa0ab8a4a6aef09adfbffae7d5fb4dee24747a8a`；16成员，25541023437 bytes |
| 公共M512仓库 | `leoyuer/pi0294-public-artifacts`，dataset类型 |
| 公共prefix | `PI0294/20261013T142457Z-localpc-pi0294/public-M512-bc28535755c4` |
| 公共READY revision | `c698f54d0fc422ebfdd1e2ce1fc96552b6930c52` |
| 公共manifest SHA256 | `bc28535755c4034c8c5f0cc10cb127cf46893f205f49aa663d6b1e417e5421f2` |
| 公共payload | 25541086052 bytes（不含READY），24最终成员含16完整Trainer成员 |

优先使用已在Hopper、按16成员全hash合格的私有M512，不重复下载全部公共大件。公版三份metadata经脱敏，数值张量不变，但身份不同。公版恢复要使用公版manifest，在本地新副本中仅重绑允许路径，记录变更，不改公版原件；独立公版神经恢复尚未证明，不作为第一次安全复验的前置条件。

本机文件“存在/大小吻合”、远端LFS OID一致、实际下载hash通过、模型加载、完整优化器恢复是不同证据级别；报告分别写。不要在只读接管时执行来源未确认的pickle。

## 6. A阶段：立即开展的只读接管

目标是产出一份 `OWNER_ADOPTION_READONLY.json`，不是新一轮大规模审计。

- 确认旧owner确实退出、新owner独占发布/调度；读取锁的host/PID/start identity，不能按PID数字杀进程或删除“看起来陈旧”的锁。
- 刷新Git分支HEAD、dirty/untracked/worktree与实际部署模块来源；`git fetch`允许只读更新远端引用，不checkout/pull正在被消费者使用的目录。交接commit和科学runtime分开登记。
- 用既有认证在每台机器只读核对自己的进程、PBS终态和预留；若有意外活跃任务则先按其身份保护，不能依据旧“0作业”快照终止。
- 读取本机私有路径/环境绑定、M512/v3.0现有验收manifest和失败视频，确认资产可访问；只补缺失或失效的检查，不盲跑全盘hash或重下模型。
- 查看个人quota与当前project权限。已有授权项目是CFP04-CF-007，须以实时资格为准；若登录过期，仅请求用户在原生终端登录，不索取凭据。没有科学GO也不试投PBS。
- 阅读待采用夹爪方案，形成一张合并任务卡，清楚列出固定项、候选修改、验证门、预算和新窗口。不要再次从零提出研究方向。

只读接管不能解除P1，也不能进行MjModel/mj_forward/mj_step、模型forward、训练或新物理重放。元数据检查本身的时间/CPU同样如实记账；适用的已有read-only许可不确定时先保留本地文档阅读范围。

## 7. 授权模型：一次明确采用，阶段之间自动过门

用户当前要求的是接管计划，不构成夹爪修改或新计算GO。现卡 `authorizes_execution=false` 保持不变。新agent可完成A阶段和方案整合；B至E必须由用户明确采用合并任务卡，并由owner在#11登记固定release、允许改动、绝对UTC/SGT截止和增量预算。

推荐采用一个新的连续执行窗口，不拼接旧余额：旧R1即便账面还余一次PBS，也不能绕过FAIL；旧cutoff是2026-10-14 19:57:30 SGT，deadline20:57:30 SGT，接管不自动重启它。没有新GO不得进入下列阶段。

**建议新授权包（提案，不生效）：** 首次实际采用时新T0；总墙钟12h，T0+11h停止新阶段，T0+12h终结；其中夹爪诊断/资格子预算至多4h墙钟、8 CPU core-hours。按新的独立continuation run_id登记，旧父/R1账本永久保留；新老账分列并显示历史累计，不能同时叠用旧余额。

| 资源 | 整个新窗口上限 | 说明 |
|---|---:|---|
| GR00T GPU | 1张H200同时、8 allocated小时 | 账号最多2独立allocation，不挤占别的任务 |
| PBS新增调用 | 3 | 失败计数；每次共享总GPU余额，不是3×8h |
| 正式新增更新 | 3584 | global512→2048→4096，不自动6144 |
| 隔离恢复工程更新 | 4 | 只在恢复缺口真实存在时使用，不重跑整套16步 |
| CoreLab CPU | 48 core-hours，最多8并发核 | 安全诊断子阶段最多4核/8coreh |
| Hopper CPU | 16 core-hours | 按层级去重，不与scheduler重复叠计 |
| 物理启动 | 128 | 安全诊断/Teacher≤32；模型评价与对应Teacher≤96 |
| 物理积分 | 2048000 | 每attempt≤16000；原2ms不改 |
| RGB pairs | 82048 | 128×641上界；短诊断可不渲染 |
| H40神经查询意图 | 5200 | 预forward失败也计意图；完成生成另记 |
| 新增峰值存储 | Hopper120GiB/CoreLab36GiB/local1GiB | 个人实际空间另留20GiB；不得重复计已有文件 |
| 公共新checkpoint衍生物 | 最多2份、合计60GiB | 只有用户同时确认延用脱敏checkpoint发布权限才启用；不含raw |
| 禁止 | WBC/硬件、新训练数据采集、历史删除=0 | v3.1只读准备不等于发布raw或换训练数据 |

B–E之间验收门通过且资源/时间成立后自动推进；不要求每小阶段再签一次。新的真实安全失败、需第三种driver干预/扩大范围、权限不足或窗口结束才停。至少预留2 GPU-hours给闭环，不把剩余时间全部花在训练/归档；预计无法容纳下一阶段时保存部分结果。

## 8. B阶段：CoreLab先处理夹爪，暂不占GPU

### B0 已知事实与待证明假设

旧D0：有效范围[-0.001,0.045]m，名义范围[0,0.044]m。失败tick1081/2.162s右指0.045004271727m，超有效门4.27微米，已超过名义上界约1.004mm，不是仅超名义4微米。所有原发送opening在[0,1]内；目标经原0.04m/s限速，motor±8N。失败时目标0.04352m、右指速度+0.005746m/s，motor -0.589112N。原算术重构误差0。

这支持动态超调，不证明模型单位错误、唯一根因是急闭、缺阻尼或接触。已有MJCF被动damping=3、implicitfast和2ms步长；不要因motor公式未用`finger_kd`就断言系统无阻尼。

机械臂有bias补偿而手指P力没有同样项，可作为待检验假设。MuJoCo中的bias、passive、constraint和惯性耦合是不同项；bias不包括所有臂加速度耦合。不得将单项补偿描述为全身逆动力学。

### B1 / 原G0：保存证据的力学与时序归因

复用1081行高频记录、初态、6个prefix和44已启动20Hz命令。对齐pre-step→ctrl→post-step，明确发送48点和实际启动44点的区别。补查左右轴/gear、位置速度、bias/passive/constraint及其保存覆盖。

必要的零积分模型计算也要先获得本科学GO；把原记录和重构字段分开。若缺warmstart、内部状态或ctrl history，不能用2.15s的qpos/qvel宣称恢复了完整原仿真状态。以原t=0、相同调用顺序复现为首选。

输出`FORCE_BALANCE_AND_TIMING.json`，列事实、支持假设、未能识别项。原已通过的范围检查不重复三轮。

### B2 / 原G1：2×2因果对照

每项最多2.4sim秒/1200积分；原native abort一出现即停，记录为FAIL而非通过。

| 组 | 机械臂 | 夹爪 |
|---|---|---|
| A | 保存原参考 | 保存原参考 |
| B | 保存原参考 | 全开1 |
| C | 原controller保持初始TCP | 保存原参考 |
| D | 原controller保持初始TCP | 全开1 |

不weld机械臂，不逐步覆写物体/关节状态；这是诊断重放，不冒称新观测Policy成功或训练数据。A未复现则先核对状态/程序次序，最多一次有依据的工程修正，不随机重试至成功。

### B3 / 原G2：最多两个单项候选，最终一个D1

依据B1/B2选择：finger bias feedforward，或额外耗散速度反馈。每次只变一个任务无关motor控制机制，参数先按动力学依据封存，保留原±8N、0.04m/s、几何/质量/摩擦/被动阻尼/步长/安全阈值。不能同时调kp、solver、夹爪命令平滑和阈值。

每候选最多6个4秒压力用例：开/闭保持、开到闭、闭到开、高端附近变化、原失败prefix加显式终点保持。首个通过全部压力测试且符合诊断的候选封存D1；任何真实安全失败淘汰该候选。不为追分无限扫描增益。

禁止qpos/qvel钳制、禁用limit、放宽容差、增加力上限、隐藏外力、目标吸附、Teacher接管、直接把模型夹爪改成0/1。候选范围不足时提出下一项明确变更而不是绕门。

### B4 / 原G3：8条Teacher回归与高频安全聚合

预登记F0/000、F1/000及v3.0 F2中按ID最小的两个双任务均准入组，每组T1/T2，8条32秒完整Teacher任务。算法/任务评分保持，只改独立driver为D1。8/8持久成功、原高频安全全通过才得`GRIPPER_EXECUTOR_REQUALIFIED_D1`。

任何500Hz abort/安全flag覆盖20Hz最后一帧；缺高频尾部为UNKNOWN而非PASS。用新聚合wrapper和负例测试实现，不回写旧评分。8条回归失败不继续用同一面板调driver。

B阶段未通过：交付因果报告、精确失败候选与差异；不申请GPU继续训练，不启动P1R。通过不等于P1R或多任务policy通过。

## 9. C阶段：冻结M512＋D1，先完成四条P1R

先校验本地M512同一成员、processor和v3.0统计，复用原48条离线诊断，不重新大规模生成。模型驻留在合法PBS，保留已修复NumPy32 seed兼容和原Torch63-bit噪声身份。

原四个开发条件：`PI0294/TRAIN/F0/000`与`PI0294/TRAIN/F1/000`，各T1/T2。新attempt ID标明D1和来源原case，不覆盖D0失败。

每次真实RGB＋实际EEF/夹爪→H40→同观测anchor解码一次→原SafetyGate→执行prefix8→200次积分/0.4sim秒→实际ACK→新观测。只读固定模型身份，taskID不是policy selector。Teacher和学生评估用同一D1；无教师接管，不重抽到接受。

P1R资格要求：4条完成新观测闭环、零真实安全违规、无消息回放/身份/解码错误。任务成功数另报，早期M512任务失败不等于安全/工程失效。安全拒绝仍保留在分母；若不能完整执行，P1R不假称通过。需要放开离线续训与物理门的耦合时必须独立提出修订，不在本包暗中旁路。

C阶段的第一关键交付是四条原始视频、每任务结果、首次失败阶段，以及requested→installed→achieved的误差。不能只报告“模型加载成功”。

## 10. D阶段：原完整训练状态续训，不换配方

C通过且时间/GPU预算成立后，恢复model＋Adam＋scheduler＋所有RNG＋消费sampler/cursor；global_step从512开始、已消费16384不清零。原训练完整恢复工程资格已成立，但M512独立optimizer恢复尚未跑过，要实际验证恢复到正确step与下一批；最多4个隔离工程更新只在必要时使用，隔离权重不作为候选。

固定：官方N1.7、冻结VLM、训练projector/动作DiT，确定性CUDA math SDP，BF16，micro2×accum16，workers0，lr1e-4，scheduler horizon6144、warmup308，官方masked flow matching、原任务/阶段平衡采样、v3.0和它的统计。不要重新warmup、把scheduler改成4096或因显存空闲就换microbatch。旧48条离线PASS不是本次恢复完整优化器的凭证。

1. 新增1536步至M2048，完整本地checkpoint/恢复索引通过即进入固定DEV8。
2. 再新增2048步至M4096，运行相同DEV8。
3. 只选M2048/M4096，不自动扩6144，不把M512诊断临时纳入候选。

原DEV四组：`PI0294/DEV/F0/000`、`F1/000`、`F2/000`、`F1/001`，各T1/T2。先最大化min(task success)，再macro，再更低拒绝率，再更早step。配对噪声与初态遵循原封存规则。评估前后恢复训练RNG、模型mode和数据cursor；预取不记实际消费。

**LOCAL_CHECKPOINT_READY和REMOTE_ARCHIVE_READY独立。** 完整本地状态已验证则可同节点评估/续训，HF403只阻断归档；跨机器读取仍校验文件身份。不得使远端上传失败再次提前结束健康模型进程。前两个checkpoint足够，远端归档在CPU路径异步排队但仅由既有owner和实际运行进程管理，不作本对话背景承诺。

## 11. E阶段：独立双任务确认与回归

DEV选型后封存唯一candidate。独立确认在此之前不解封、不用于D1调参，不因为Teacher失败而替换布局。

核心8场景组×T1/T2：Teacher先验证同一D1；每任务至少7/8且无真实安全违规才运行模型核心确认，所有原定case留在分母。模型每任务≥6/8、同组两指令均成功≥5/8、旧抓放回归≥6/8、零真实安全违规，才报告`MULTITASK_SIM_WORKING_V1`。小样本仅支持探索性固定底座双技能能力。

8条旧抓放回归优先于挑战；有剩余时间才运行8组挑战×两任务及相应Teacher资格。不在确认后再挑新checkpoint、补采seed或改driver。核心或挑战失败仍交全分母与失败视频，未运行写NOT_RUN；安全性未知写UNKNOWN。

性能改善若来自D1修订，身份必须是`M512+D1`，不能记作训练收益。M2048/M4096同用D1才可比较学习进展。D1下的状态分布不假定与D0完全相同；保留接触、实际夹爪误差和饱和率。

## 12. 并行资源计划：优化关键路径，不制造新项目

| 阶段 | CoreLab | Hopper | local-pc |
|---|---|---|---|
| B安全归因 | 单worker因果对照/压力/Teacher回归 | 不占GPU；可只读核对原资产和有效权限 | 唯一账本/版本/权限协调 |
| C P1R | 真实物理/双RGB/高频安全 | 同一模型驻留推理 | 双SSH ACK中继 |
| D续训 | 评分/视频、只读整理既有v3.1归档差量 | M2048/M4096训练 | 发布小收据、检查里程碑 |
| E确认 | 优先服务模型评价，减少I/O竞争 | 固定候选推理 | 全分母/噪声/身份核验 |
| 收尾 | 失败/成功视频与环境状态 | 完整状态保存和终态 | 报告、Git/HF索引、预留归零 |

本轮新采训练数据0，v3.1不热加入、不重新拟合stats。四worker数据工厂是已有能力；只有模型失败归因表明覆盖缺口后，才另做v3.0/v3.1对照或恢复数据实验。大规模LeRobot迁移、VLM缓存、升级依赖、统一CLI重写、全盘清理不占关键路径。

## 13. 工程规范与收尾

只修阻断本阶段的最小文件，列allowlist diff；旧D0/数据/训练统计/runtime快照不可改写。先本机相关CPU测试，再固定Git提交/CI/源hash回读，再在空闲节点部署新capsule；CPU CI不证明D1有效。不要每个文档更新都重跑全部历史仿真或重新安装环境。

新控制器封存后，该轮所有Teacher和模型引用同一driver SHA。主代码管理checkout可以fetch；活跃源快照不能pull。共享NFS只有一个修改owner。只用普通提交/合并，保留并发提交，不force/reset/clean。通用runtime源与166份历史参考目录分开。

HF只发布已明确授权和去敏的checkpoint衍生物；M512已归档不重复上传。v3.1 raw没有公共发布授权。模型、视频、WBC、资产、环境、凭据不得随Git add -A上传。私有来源与metadata路径也不进入Git报告。

一次外部阻断明确通知一次，在状态变化或约定复查点再检查；不连续三轮重复同一FAIL、不重复qsub。物理因果诊断预期重现FAIL仍执行原生停止；后续安全违规则停止该候选/模型臂，不使用预算余额作为绕过门的理由。

## 14. 必交文件与状态机

按实际证据逐级推进，不能跳级：

`TAKEOVER_VERIFIED` → `WAITING_SCIENCE_GO` → `GRIPPER_CAUSE_BOUNDED` → `D1_TEACHER_QUALIFIED` → `P1R_SAFE_RUNTIME` → `M2048_DEV_EVALUATED` → `M4096_DEV_EVALUATED` → `CANDIDATE_FROZEN` → `CORE_AND_REGRESSION_EVALUATED`。

`MULTITASK_SIM_WORKING_V1`只在任务成功门成立时出现。`selected`只在合法DEV选型后更新；`deployment`本轮始终NONE，因为不存在真机部署资格。

必交：
- `OWNER_ADOPTION_READONLY.json`、`AUTHORITY_ADOPTION.json`、新`RUN_MANIFEST.json`（后两者无新GO时只能DRAFT）。
- `FORCE_BALANCE_AND_TIMING.json`、`DIAGNOSTIC_MATRIX.json`、`DRIVER_DIFF_AND_SAFETY_INVARIANTS.json`、`STRESS_RESULTS.json`、`TEACHER_REGRESSION.json`。
- `P1R_MODEL_RESULTS.json`、`CHECKPOINT_RESUME_RECEIPT.json`、`TRAINING_RECEIPT.json`、`DEV_RESULTS.json`、`CANDIDATE_SELECTION.json`。
- `CORE_CONFIRM_RESULTS.json`、`REGRESSION_RESULTS.json`、可选`CHALLENGE_RESULTS.json`、原视频的私有路径角色/哈希。
- `RESOURCE_DELTA.json`、`FINAL_REPORT.md`、准确可复现的下一步/恢复命令。不存在的产物标NOT_RUN，不生成空PASS。

#11发布科学实际结果、#47仅发工程/资源/路径变化；回读固定commit/文件/评论。每个新checkpoint有local content ID、对应processor/stats、driver、数据/采样身份和恢复性；不把local ID伪装HF revision。

## 15. 后续WBC接入（路线，非本轮授权）

固定底座双任务能力成立后再开展：正确Teacher参考→师兄固定WBC；同一高层→当前独立ARX5；最后同一高层→同一冻结WBC。把EEF/rot6d转换到实际WBC接口所需位姿与时钟，夹爪独立通道明确绑定；旧不同版本WBC接口不能混用。

先独立确认保持、源时钟、参考热替换、夹爪和正确任务参考，后跑学习策略。固定外部body相机与移动机身相机属于不同观测域；需要独立评价。WBC/真机0调用这一事实不因有了D1而改变。我们不把重新训练通用WBC列为高层任务成功的无限前置。

## 16. 给新 Agent 的第一条回复要求

第一条接管回复应简短列出：固定交接/runtime身份、私有绑定是否可访问、旧作业/预留是否已终态、当前新GO是否存在、唯一阻断和下一动作。不要复制整段历史汇报，不写“我已开始训练”除非有实际提交/运行收据。

无新GO时完成只读接管和合并授权卡即停在WAITING_SCIENCE_GO；用户一次明确采用完整B–E包后按门连续执行，不逐阶段重新请求同一许可。不得为了避免询问而把当前规划请求解释成真实动力学或控制器修改授权。

## 17. 固定证据索引

以下是本方案的审查依据，不是本方案新增执行结果：

- E1 固定完整交接： https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/7e7773a5628d7f119076220ac94bb92a1a9ce120/docs/handover/2026-10-14/HANDOFF.md
- E2 代码/证据索引：同固定commit的 `docs/handover/2026-10-14/CODE_AND_EVIDENCE_INDEX.json`。
- E3 科学终态： https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-6162189824
- E4 工程终态： https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47#issuecomment-6162190095
- E5 夹爪待采用卡：同固定commit的 `reports/PI0294/design/GRIPPER_SAFETY_AMENDMENT_TASK_CARD.json`。
- E6 父训练/双任务数据/评估面板：同固定commit的 `reports/PI0294/20261013T142457Z-localpc-pi0294/`。
- E7 原R1失败与保存轨迹：同固定commit的 `reports/PI0294/20261014T025730Z-localpc-pi0294-r1/`。
- E8 CPU CI： https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/actions/runs/38521034429 （本轮已读取job结果SUCCESS，非模型/物理测试）。
- M1 MuJoCo3.2.7力学/状态说明： https://mujoco.readthedocs.io/en/3.2.7/computation/index.html
- M2 MuJoCo3.2.7仿真状态与计算流程： https://mujoco.readthedocs.io/en/3.2.7/programming/simulation.html

本计划制定过程为GitHub收据/源码与官方文档审查；没有SSH、重新下载25GB权重、复算私有全量数据、运行MuJoCo或模型。发布与下载计划文件不授予科学计算权限。
