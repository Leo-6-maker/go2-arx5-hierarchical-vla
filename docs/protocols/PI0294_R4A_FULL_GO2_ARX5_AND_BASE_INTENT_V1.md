# PI0294 R4A：完整 Go2＋ARX5 数据、BASE软参考与自主工程推进

日期：2026-10-15。事实审核基线：`ca2b738575e3c37a753454f3b59f227331045408`；发布期间另见父R4 `PROGRESS_0220.json`，应以owner现场刷新为准。

## 0. 用户方向、权限和正在运行的任务

本次直接用户指令：
> 我们的policy是否应该在完整go2+arx5环境中生产数据并测试。之前我们探讨过policy还需要输出base软限制的速度，这与目前版本的WBC接口相通，请你仔细分析后更新长程实验计划，工程问题自主修复推进。

状态：**USER_REQUESTED_SCOPE_AMENDMENT_CONDITIONAL_ON_EXISTING_RESOURCE_AUTHORITY**。本文件据该指令更新目标域和自主工程范围，只使用已授权R4剩余资源；不是新增资源预算，不是实际运行收据。

现有incoming local-pc继续为唯一owner，读取#11/#47最新增量、原生Q/R/F和预算后，登记用户原文、固定scope版本、实际余额与阶段边界。普通parser、坐标、时序、依赖与构建修复不再逐项申请同一许可。

**发布中发现R4已有4个正式采集worker在运行。** 不kill、不热改它们；让已开始的episode完整结束。仅在调度边界停止分配尚未开始的可选固定臂新数据，并将相应未尝试项记为SCOPE_SUPERSEDED_NOT_RUN。仍健康的R3评价/已有Hopper训练按原版本和原授权完成，不重复qsub或取消其他项目。

**新范围允许私有已交付冻结WBC及腿部RL在CoreLab完整仿真中推理、跟踪、生成数据。** 旧WBC_calls=0只对本增补子阶段被替代，旧记录保持0；高层GR00T仍只在合法Hopper allocation运行。腿部冻结CPU策略是执行器负载，不能称作偷偷在CoreLab运行GR00T。

原WBC权重、成本、约束与增益保持冻结，不授权重训WBC、暗换native/policy-aware模型、实机/SDK、关闭安全、删除历史数据。需改变这些科学因素时提交具体证据和另项决策。

资源继承R4：T0 2026-10-15 01:09:30 SGT，新工作截止12:09:30，终截止13:09:30；3PBS/10H200allocated小时、至少3小时留评价、项目1GPU/账号3个独立allocation。所有已用和在途预留先扣减；过期后只做阅读/代码准备，不自行新开远端窗口。

优先级正式改为：**完整Go2＋ARX5为主验收域，固定底座是基线，不再要求它完美才启动整机建设。** 未启动的旧可选C1/固定臂扩采份额转给移动数据/EEF13，不能两路重复领取额度。

## 1. 已核验事实与实际WBC接口

M2048已真实训练；固定底座D0/S1和Teacher资格不等于整机资格；v3.1增量在v3.0统计下codec拒绝保留。`PROGRESS_0220`记录旧R4生产在进行、C1数据视图受阻、模型评价等GPU，这些不是本对话SSH实时结果。

旧场景固定了Go2根/腿，body_rgb是世界固定外部相机。它能诊断局部任务学习，不能验证足地支撑、身体运动、移动视觉和整机任务成功。加载Go2外观不等于完整自由动力学。

本轮实读归档 `wbc_mujoco_minimal.tar.gz` 的高层 `/go2_x5/ee/trajectory_json` 输入是EEF7D或EEF+BASE14D pose；baseTracking软跟踪x/y/yaw，默认权重5/5/2。节点把12D input_trajectory填零。下游机身速度由MPC/bridge生成，不是VLA直接控制腿的入口。

新链路：
`GR00T EEF13 = EEF10 + (vx_pref,vy_pref,heading_rate_pref)`
`→ 固定速度/加速度/区域/TTL边界`
`→ 从观测时实际BASE姿态SE2积分得到软位姿参考`
`→ 显式EEF+BASE原生14D目标`
`→ 冻结OCS2/WBC + arm PD + 腿RL + 独立夹爪`
`→ 全身物体交互及真实新观测`。

“软”是优化器对偏好的允许偏离，不是安全限制可以忽略。速度意图与速度上限不同；EEF本身也不是绝对硬优先。完整ABI、坐标、origin偏移、零与无BASE、40/41点、时钟/TTL见 `docs/interfaces/EEF13_BASE_SOFT_REFERENCE_V1.md`。

若现场包比本次实读版本更新，先实际追踪API parser→consumer→cost/constraint→执行者，测正负号与失效；不得根据名字假定有直接速度接口。不得给零12D输入或腿cmd_vel填入VLA速度。

## 2. 科学来源与边界

D_fixed保留原独立工具/固定底座数据和M2048，作为初始化与诊断基线；D_full是完整整机/WBC产生的目标仿真适配数据；E_full是独立整机场景评价。

D_full无需目标机器人遥操作，但其准入受本体和WBC能力影响，应明确叫**目标整机仿真适配**，不能继续宣称该分支严格独立于目标WBC。保存任务规格、全部Teacher请求、执行失败和拒绝，不能把无法完成的场景从任务定义删除。保留D_fixed，供后续分离数据来源/执行器偏差。

BASE意图不是joint action，仍然是任务空间接口；但新增base观测/意图会增加本体相关性。跨本体、真实硬件、冻结WBC承接任意任务仍是待验证命题。

## 3. 里程碑与并行职责

M0：固定package/API/资源身份。
M1：完整Go2＋ARX5能通过正确Teacher参考站稳、移动和完成操作。
M2：整机v4.0-mob13和EEF13原生数据链成立。
M3：同一FB13模型在完整整机完成两任务开发面板。
M4：独立含移动场景的双任务确认成立。

CoreLab A建设整机与接口；CoreLab B并行准备scene/split/Teacher和数据工程，通过环境门后2→4worker生产。Hopper收齐已有M2048面板/健康训练、准备新模态，在v4.0冻结后训练，不等v4.1。M1未成立不能用固定臂数据改名交差。预算优先M1/M2/M3，M4按资格与余量执行。

## 4. S0：最短接管

读AGENTS/START_HERE、#11/#47、私有runmap和真实PBS/worker；复用 `~/Projects/GR00T-WBC`、既有enter/doctor与Hopper启动器，不重建通用CLI、不运行166份历史owner档案。

固定当前WBC包/模型/mesh/腿策略/库/ROS/profile及来源hash，现有M2048/v3.0stats和仍在运行的source capsules；新child namespace为PI0294_R4A_FULLBODY，对应父R4总账。

阶段边界后把旧未启动的固定臂C1/扩采重分配；已进行的C0不能热换EEF13数据。新包是OCS2+MuJoCo+腿RL，不是旧201D/15D双actor接口，禁止混装。

## 5. S1：真正的全身物理场景

必须有free root、12条腿关节、6臂关节、有限力矩夹爪、重力、原质量/惯量和足地接触。不得weld根、禁用腿、逐步覆写base、隐藏悬浮力或沿理想路径拖动模型。

原包缺任务夹爪/命名相机时，允许版本化仿真plant overlay补齐真实末端安装、有限驱动/碰撞pad、body-mounted与wrist相机；记录质量/惯量/几何增量并重新资格。视觉与collision分离，避免重复geom挤压。Teacher/model用同plant；不能把修改plant后的分数同旧域直接混合。

机身相机固定到Go2实际body/site，随身体移动；手腕相机随腕移动。额外外部机位只作审计。每帧保留真实K/外参/仿真时间/hash。旧世界固定body_rgb保持单独域名。

0.002s物理、20Hz观测、H40/prefix8；OCS2/MRT/腿策略/PD/夹爪的实际调度和采样点写入manifest。等待高层时所有sim-time组件冻结，不能只停MuJoCo让ROS wall timer/腿history继续推进；恢复保持同一时钟。实际延迟单列，不靠重锚定掩盖。

先做最多8个有界native检查：站立、EEF世界保持、正负平移/偏航、EEF移动+BASE软保持、BASE移动+EEF世界保持、停止/TTL/reset。所有起立/warmup/settling计积分。沿用已有位置RMS0.015m、姿态RMS0.05rad、源时钟残差0.025s和原native安全门；正确参考下仍不合格，给同刻request/installed/achieved/力/clock证据，不降低门槛。

范围内允许修源时间、quat/mount、前馈字段绑定、模式切换与记录正确性；不靠VLA学固定控制偏置。WBC问题阻断整机物理线，不阻断独立已有离线/CPU工作。

## 6. S2：整机数据与任务

T1抓取—搬运—放置—释放—撤离，T2桌面推送—分离—保持。每组两任务同机器人/物体/目标/相机初态，仅语言不同。

W0：free-standing近场，臂可达，腿RL保持，BASE偏好近零。
W1：受控前移/侧移接近后操作；尽可能使目标超出初始固定base合法臂工作空间。
W2：改变朝向再接近，或低速身体协同搬运/推送。

移动候选从0.12–0.25m、偏航0.15–0.30rad范围选择，由实际native资格进一步收窄；不是已验证可行域。W1/W2在运行前记录静止base可达性和安全几何，不能把一次局部IK失败当严格不可达。只能证明参与时标BASE_PARTICIPATING，能给任务几何证据才标BASE_REQUIRED。不得用随机漂移充当执行意图。

Teacher使用任务/物体真值产生正确EEF与BASE意图，可几何/IK/确定性优化选站位；不得把WBC纠正速度或actual速度反填成高层监督。保持/释放需正常零速度参考，移动样本覆盖非零平移/偏航及正负方向。精细接触优先靠近后进行，不首夜上复杂地形或未知物体形状。

**整机资格pilot新增上限4条（2组双任务），从父R4的main-start/积分/CPU余额重分配；与已执行的旧固定臂4条pilot分开记。这里明确细化原pilot子分配，不能因旧pilot做完而跳过新域资格，也不增加父160启动/总积分/CPU/GPU上限。** 全部持续成功/零native违规才进入正式数据；pilot不训练。

v4.0-mob13：12TRAIN组（W0/W1/W2各4）×2=24条；4DEV组×2=8条。每任务TRAIN>=11/12、DEV>=3/4且移动类型有覆盖，所有首次尝试终态后冻结小包即开始训练。
v4.1-mob13：累计24TRAIN组×2=48条、8DEV组×2=16条，包含v4.0；各任务TRAIN>=22/24、DEV>=7/8。旧R4已采TRAIN扣减父48条总量，余额不足则缩小/保留扩展NOT_RUN，优先启动包，不伪造全量完成。

每条默认32s：16000任务积分、641实际观测、640提交、601 H40；reset/warmup积分额外记录但在总预算内。不得插值伪造未采帧或标签。所有物理、安全、接触、codec、失败记录保留。分集按scene-group，配对/相邻变体不跨split；旧DEV/确认集不能用于新TRAIN。

记录EEF13、推荐实测state19、移动机身/手腕RGB；保存requested/installed/WBC-selected/achieved四链。旧EEF10不能补3个zero监督冒充整机，不可由EEF速度推导base命令。D_full的WBC能力过滤需在数据卡明确披露。

## 7. S3：fullbody多worker和数据制作

复用 `scripts/pi0294/dispatcher.py`、`cpu_pipeline.py`，实现新的fullbody worker插件，不能给旧fixed-base worker换标签。

每个worker独立ROS domain/namespace、OCS2/reference manager、腿history/gait phase、MuJoCo可变状态、arm/gripper目标、时钟、渲染/输出。只读权重和mesh可共享。一个coordinator原子分配资源/attempt，不能多个worker写同topic或总JSON。

1worker资格后2worker与串行同seed比较；4worker须资源/隔离/数值/任务和安全对照成立才开启。旧2.86x不当作新整机收益。起步每worker最多4核，2worker给审计留余量；整项16核包括旧任务、MPC、RL、渲染和审计。设置BLAS/Torch/OMP上限，不超售。

记录worker/optimizer真实时间交集、吞吐/coreh/合格episode与MPC/渲染/PNG/I/O分项。评价优先，在episode边界降采集并发。v4.0本地冻结校验后Hopper可训练，v4.1晚到数据只归档不热加入。

## 8. S4：从EEF10暖启动新的EEF13模型

M2048 padding槽10:13没有已学会的BASE含义，必须有新监督/mask/codec。官方内部132D宽度可复用，但输入state、输出13D、物理单位、模态和loss需要实际验证。

FB13-Uxxxx从固定M2048模型参数warm-start，不从pristine重训；保留前10维语义和原动作DiT，新增base/state适配做有界初始化。冻结VLM，训练projector/动作DiT/必要适配。零初始化或归一化中心必须经测试证明对应零物理BASE意图，而不是正向漂移。

**新动作/观测/数据域是显式科学干预，不是假称原完整optimizer连续恢复。** 保存父M2048全部状态，FB13使用独立optimizer/sampler/scheduler和parent_model hash，记录WARMSTART_NEW_ABI_DATA_INTERVENTION。若原C0继续，它保留原exact-resume链。

v4.0 TRAIN独立拟合新版本归一化；BASE速度可用固定安全包络定义对称缩放，使物理零与符号明确，所有尺度写入schema，不使用DEV。原v3.x统计不改。第一版默认只训练有效完整EEF13数据；旧数据若以后混合需逐维缺失label mask/状态处理真实资格，不能零填充监督。

旧v3.1在v3.0统计下的codec失败继续保留，不放宽它；新域新stats/ABI是独立版本。检查state/action clipping、13D mask、SE3一次变换、base速度绝不能再次当pose做relative转换。不是shape正确就PASS。

训练固定官方masked flow-matching、BF16/mathSDP、effective32、lr1e-4；FB分支自身2048步schedule、5%warmup，保存U512/U2048。新正式最多2048步，替代未启动旧可选C1；R4所有臂累计新增上限4096不变。

DataLoader0/2/4与有界cache限时验证；实际端到端收益>=5%且消费/张量/恢复一致才采用，否则workers0。隔离工程更新与父上限16共享，不另扫microbatch/attention/loss。远端归档错误不阻断合格本地checkpoint评价。

## 9. S5：整机评价和可证实的结论

先Teacher+fullbody；旧M2048 EEF-only/no-BASE-preference作为迁移基线；新FB13完整输出经同一WBC执行。任务正确性、参考安装、实际跟踪与足地稳定分别评分。

开发消融用同一FB13、同初态/噪声，启用与省略BASE软偏好比较。省略不等于velocity=0。若WBC无偏好也成功，诚实归因，不强行宣称base-head有收益。

实际整机安全模式：原固定臂S1不能自动覆盖OCS2/腿history。允许在完整Teacher/时钟/接口资格后进行**单worker、仅仿真FB_NATIVE_ABORT**有界模型测试：原500Hz限位/接触/稳定性检查，首违规即时冻结并终止该条，保留分母；不是旧S1保护，也绝不是真机急停保证。只有整个MuJoCo+MPC+history+控制器时钟/RNG可复制并通过matched-prefix资格才采用可选S_FB；未实现则明确NOT_QUALIFIED。新native安全失败关闭该版本live臂，独立无共同输入问题的离线工作可继续。

必须报告有符号base位移/偏航、足地接触、身体姿态、EEF误差、base期望与WBC选择差异、物体行为及移动相机观测；根固定、腿没执行或隐藏拖动不计整机。

开发门：8条DEV两任务各>=2/4、移动类有成功、无真实安全违规，方可固定一份候选。
核心门：独立8组×两任务；Teacher每任务>=7/8且零native违规才开模型；模型每任务>=6/8、配对均成功>=5/8。至少4组是预注册移动类，移动类每任务>=3/4，且完整目标/释放/保持成立。近场8条回归单列。资源不足优先真实DEV和移动证据，未运行核心如实NOT_RUN。

达标才报告 `FULL_GO2_ARX5_MULTITASK_SIM_WORKING_V1` 并附实际guard模式。可选S_FB/无预测原生停止的分母不混用；部署NONE，不声称无保护、真机或跨本体资格。

## 10. 工程自主修复与范围边界

允许在既有权限和预算内自主修复：构建/局部依赖/路径、ROS隔离、时钟/坐标/mount/quat、parser/shape/mask/codec、reset/模式、有限夹爪与相机overlay、headless渲染、ACK/TTL/cancel、原子资源账和checkpoint I/O。

步骤固定：最小复现→最小patch→相关CPU契约→隔离短资格→冻结source/plant/bridge→正常边界采用。每子系统主动修复最多2h、3个有理由的工程attempt；其他独立线继续。源码和所改plant是否数值干预分别标记，不以工程名义隐瞒。

不允许：改WBC学习参数/成本/gain/摩擦追分、改安全或评分、关碰撞、钳制实际qpos/qvel、隐藏外力/teleport、换失败seed、公开raw/mesh/WBC、无权限系统安装、重置窗口和预算。发现明确视觉/碰撞资产错误，可做最小版本化plant-overlay修复并重新资格；原与新plant分数不混用。要重训或更换WBC科学配置时另提具体范围，不无限追分。

固定臂未完美不阻止整机建设；WBC尚未资格不阻止CPU schema/旧模型离线；数据扩展慢不阻止冻结小包训练；上传失败不阻断本地评价。共同身份/权限/总预算错误才停止相关全部范围。

## 11. 父预算、优先级和收尾

R4/R4A统一扣账，所有已用/在途先减：3PBS、10H200allocated h且至少3h评价、项目1GPU/账号3需实际权限、4096新增正式更新、16隔离工程更新、Core96coreh/并发16核、Hopper32coreh、160 main启动、2560000主积分、102560真实RGB对、2000000 shadow积分/10000shadow调用、6000高层生成意图。Core96GiB/Hopper180GiB/local4GiB新增峰值、Hopper个人余量20GiB。额外整机pilot重新分配的是main启动份额，不增加这些总额；旧4pilot和新4pilot分别列明。

新TRAIN累计仍<=48（旧已采扣减），新DEV<=16来自原main/eval剩余份额；全身warmup/settling和WBC/腿RL CPU全部记账。目标数据/核心数量是有条件目标，不保证在余额内全部完成。须削减时先停v4.1和未启动旧可选分支，保v4.0、Teacher/模型真实行为与恢复状态。

超过原窗口仅交PARTIAL和准确恢复命令；不自主加PBS/下一夜。新资源或WBC训练/硬件范围需要真实决定，普通工程fix不重复GO。公共脱敏checkpoint仍最多2份/60GiB并扣旧额度；不公开新的raw/RGB/视频/mesh/WBC或私有路径。

## 12. 交付与证据

必交：R4A_SCOPE_ADOPTION（用户原文/父授权/实时余额/旧未启动项如何转配）、WBC_INTERFACE_RECEIPT、FULLBODY_MODEL_MANIFEST、BASE_INTENT_BRIDGE_TESTS、FULLBODY_TEACHER_QUALIFICATION、MOBILE_DATASET_CARD/MANIFEST、PARALLEL_WORKER_QUALIFICATION、PIPELINE_TIMELINE、MOBILE_CODEC_AND_MASK、FB13_WARMSTART/TRAINING_RECEIPT、FULLBODY_DEV_RESULTS、BASE_PREFERENCE_ABLATION、条件CORE/回归、原视频私有索引、RESOURCE_DELTA、FINAL_REPORT。

普通Git提交保留并发更新并回读，#11科学、#47工程。只发自研代码/小收据；模型和原始数据留授权存储。没有运行写NOT_RUN，脚本存在不等于整机PASS，不重复上传M2048充当进展。

## 13. 审核来源

静态archive/API/坐标证据见 `docs/interfaces/EEF13_BASE_SOFT_REFERENCE_V1.md`；两归档哈希均在本次本地重新计算并与原审查一致，未运行私有库或权重。
既有审查：`docs/coordination/2026-10-10_WBC_PACKAGE_COMPARISON_AND_TARGET_LEARNING_REVIEW.md`；章程：`docs/RESEARCH_CHARTER_TASK_SPACE_MANIPULATION_V1.md`。
父R4采用：#11 comment6173988141，`reports/PI0294/20261014T170930Z-incoming-pi0294-r4/AUTHORITY_ADOPTION.json`；数据视图失败：同目录V31_V30_STATS_CODEC.json；并发进度：PROGRESS_0220.json（只是快照）。
方法背景仅说明分层可能，不证明本项目成功：https://umi-on-legs.github.io/ ；https://leggedrobotics.github.io/ocs2/optimal_control_modules.html 。

**本次只有静态审核与GitHub范围更新。完整场景、EEF13、BASE桥和fullbody worker在真实执行验收前均为待实施，不能将新计划提交计为策略或整机成功。**
