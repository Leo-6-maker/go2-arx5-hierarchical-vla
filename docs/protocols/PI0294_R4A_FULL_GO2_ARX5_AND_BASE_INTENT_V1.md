# PI0294 R4A：完整 Go2＋ARX5 数据、BASE软参考与自主工程推进

日期：2026-10-15。事实审核基线：`ca2b738575e3c37a753454f3b59f227331045408`。

## 0. 用户方向与执行权限

本次直接用户指令：
> 我们的policy是否应该在完整go2+arx5环境中生产数据并测试。之前我们探讨过policy还需要输出base软限制的速度，这与目前版本的WBC接口相通，请你仔细分析后更新长程实验计划，工程问题自主修复推进。

状态：**USER_REQUESTED_SCOPE_AMENDMENT_CONDITIONAL_ON_EXISTING_RESOURCE_AUTHORITY**。本文件按该指令更新科学方向、允许范围内自主工程修复，并在既有 R4 有效余额内分配整机仿真工作；不是增加一份资源预算，也不是已实际运行收据。

- 现有 incoming local-pc 是唯一 owner。采用后在 #11 登记这条用户原文、固定文件版本、实时资源与阶段边界，不要求用户对同一范围的 parser、坐标、时序、构建修复再次确认。
- R3 已排队/运行的评价和 R4 已实际启动的作业按其原版本完成或在正常任务边界交接；不 hot patch、不重复 qsub、不取消其他研究作业。
- **本次明确扩大场景范围：允许私有已交付冻结 WBC/腿部RL在 CoreLab 整机仿真中推理、跟踪及生成数据。旧方案的 WBC_calls=0 对本增补子阶段被替代；旧实验仍永久记0。** 高层GR00T仍只在合法Hopper allocation运行；腿部冻结CPU策略属于执行器工作负载，不是偷偷把GR00T搬到CoreLab。
- WBC权重、原任务成本/约束/控制增益冻结。不授权重新训练WBC、暗中切native/policy-aware模型、实机/SDK动作、关闭安全或删除历史数据。需改变这些科学因素时交付具体证据和另项提案。
- 资源沿 R4 原授权：T0 2026-10-15 01:09:30 SGT，新工作截止12:09:30，终截止13:09:30；账户3个独立allocation/本项目1GPU、R4累计最多3PBS/10H200小时、至少3GPU小时留评价。若现场已经过期，只读接管/代码准备可继续，新的远端运行需要真实新窗口；不能凭本文件自造T0。
- 规划优先级发生变化：**完整 Go2＋ARX5 成为主验收域，固定底座降为可归因基线，不再要求固定底座先完美才启动整机建设。** 已有有效固定底座模型面板优先收齐；未开始的可选固定底座C1/额外48条扩采额度转为本增补的移动数据与EEF13分支，不能两边都领取同一份额度。

## 1. 审核结论与不能混淆的接口

实际当前成果：M2048已训练；固定底座ARX5的独立D0、S1和Teacher通过部分资格，完整学习策略仍需实际面板；v3.1增量在v3.0统计下有codec拒绝。最新远端小收据不证明当前作业状态，owner必须刷新。

固定底座仿真曾冻结Go2根/腿，body_rgb为世界固定外部相机。这可以验证局部操作学习，却不能证明腿部稳定、身体移动、移动相机分布或完整Go2＋ARX5任务成功。仅加载一个Go2外观模型不叫整机闭环。

本轮静态实读私有 `wbc_mujoco_minimal.tar.gz` 的原生高层入口为 `/go2_x5/ee/trajectory_json`：EEF 7D pose，可选BASE7D pose，总14D；baseTracking默认x/y/yaw权重5/5/2。`input_trajectory`由节点填12D零向量。WBC下游的机身速度命令由MPC/bridge生成，并不等于VLA可直接写腿的速度端口。

因此用户提到的BASE软速度合理落地方式是：

`GR00T EEF13 = EEF10 + (vx_pref,vy_pref,heading_rate_pref)`
`→ 固定速度/加速度/区域/TTL安全边界`
`→ 从观测时实际BASE位姿积分为BASE软位姿轨迹`
`→ 原生EEF+BASE 14D参考`
`→ 冻结OCS2/WBC + arm PD + 腿部RL + 独立夹爪`
`→ 完整MuJoCo物体/身体运动和真实观测`。

BASE“软”指优化器允许偏离偏好，不是允许物理安全违约；速度意图与速度上限是两种不同字段。EEF自身也不是绝对硬优先。完整ABI、坐标、速度积分、13/14/12维区别、零偏好与无偏好、40/41点、失效处理见 `docs/interfaces/EEF13_BASE_SOFT_REFERENCE_V1.md`。

该说明适用于实际读到的固定包。现场若有更晚的速度reference API，先追踪解析器→代价/约束→执行者并做正负号/无效值/失效测试；不能凭名称或旧讨论假定已实现。不得为了找速度接口直接改为下游cmd_vel控制。

## 2. 科学设计：保留三类来源，不重写旧结论

D_fixed：原独立工具/固定底座ARX5数据和M2048，作为局部技能、初始化及回归基线。
D_full：任务Teacher通过完整Go2＋ARX5与冻结WBC产生实际图像、状态和监督，用于本轮整机仿真适配。
E_full：从未参与训练/选型的整机任务场景，用于最终评价。

D_full不需要真实机器人遥操作，但它被本体与WBC执行能力筛选；应称为**目标整机仿真适配数据**，不能继续宣称该分支严格独立于目标WBC。每个Teacher请求、实际执行失败和质量拒绝都保留。保留D_fixed和独立任务规格，才能后续分析执行器选择偏差；不把WBC失败场景从任务定义中删除。

BASE偏好不是腿关节控制，因此仍保持任务空间/低层控制分工，但显式base观测与意图带来额外本体相关性。跨本体是后续实验，不是加了13D接口就成立。

## 3. 三条并行工作线与里程碑

| 线 | 工作 | 不必等待 |
|---|---|---|
| CoreLab A：整机环境与接口 | 固定包核验、独立namespace、全身自由动力学、相机/夹爪、EEF+BASE桥、原生时间 | 不等M2048在固定底座全部成功 |
| CoreLab B：数据 | 场景组和Teacher规格、split、完整整机资格后2→4worker生产、原生CPU数据验收 | 可先做规格/静态检查；不可在环境未资格时伪造数据 |
| Hopper：模型 | 收齐已许可M2048面板/正在运行C0；准备EEF13状态与动作适配；v4.0冻结后开始移动任务分支 | 不等待v4.1扩采、目录重整或通用CLI重写 |

M0：实际高低层接口和固定资源身份成立。
M1：完整Go2＋ARX5教师能站稳、转向、平移并完成物体操作。
M2：整机v4.0数据、EEF13 processor和Teacher闭环通过。
M3：一份FB13共享checkpoint在两任务上完成受控开发评价。
M4：同一WBC/同一checkpoint在独立场景，包括确需底盘移动的场景，通过任务验收。

本轮优先M1→M2→M3；M4只在有资源且开发资格成立时执行。没有M1就不能把固定底座数据/视频改标签冒充整机成果。达到预算仍未完成的项写NOT_RUN，保留可直接恢复的版本。

## 4. S0：最小接管与版本封存

刷新AGENTS/START_HERE、#11/#47及原生Q/R/F、实际余额。复用 `~/Projects/GR00T-WBC` 已合格入口、私有runmap和Hopper启动器，不建立第二owner、第二顶层checkout，不运行历史owner档案。

记录：当前WBC包/mesh/模型/腿策略/ROS2/编译库/运行profile hashes；原VLA M2048、v3.0统计、R3/R4仍在用的source capsules；新增child namespace `PI0294_R4A_FULLBODY` 与父R4预算映射。

阶段边界后仅取消尚未启动的可选固定底座扩采/C1计划项，保留其旧计划与NOT_RUN原因。已启动的有效episode完整结束再收口。已有C0训练不能直接换13D数据；新动作分支另建manifest。

最新固定包是OCS2+MuJoCo+冻结腿RL，不是旧WBCv1的201D/15D actor接口。不要混用两套checkpoint/观测或以旧TorchScript策略凑出一个似是而非的整机环境。

## 5. S1：搭建真正自由底座的Go2＋ARX5场景

必须使用free root、12腿关节、6臂关节、实际夹爪动力学与接触、重力、原质量/惯量、真实足地接触。禁止把腿固定/禁用、weld根、逐步覆写base pose、隐藏悬浮力或沿理想底盘轨迹拖拽模型。

原包缺少任务夹爪和命名相机，允许在**版本化仿真资产overlay**中补齐：原生臂末端安装关系、有限动力夹爪、真实碰撞pad、真实body-mounted和wrist camera。视觉mesh与collision geom分离；不能用重复碰撞几何挤压物体。新增质量/惯量/碰撞必须记录，所有Teacher与模型共享同一plant版本；不能称修改plant后仍与旧数据完全等价。

body相机固定到Go2机身的实际body/site，随姿态变化；wrist相机固定到实际手腕。可额外渲染外部审计视角，但不能替代policy的机身视角。每帧保存K、W_T_camera、实际曝光仿真时间和原图哈希；原世界固定body_rgb另保留域名。

同步采用同一仿真tick调度：0.002s物理，20Hz观测，H40/prefix8。OCS2、MRT、腿RL、arm PD、夹爪与相机各自实际频率和采样点写入manifest。推理等待期间所有sim-time状态冻结，不让ROS wall timer把WBC独自向前跑。恢复时继续同一sim clock；不能只暂停MuJoCo却让参考到期/腿history继续更新。

保留原先PI0289保持和时钟FAIL证据。允许对源时间、参考锚定、quat顺序、mount transform、前馈字段和模式切换做正确性修复，但必须对照原意图、封存diff再复验。不得用VLA学习固定偏置来掩盖一个恒定目标也跟不住的低层问题。

### 原生整机资格

先完成不含新高层模型的短测试：站立保持、EEF世界系保持、正负平移、正负偏航、EEF移动且BASE软保持、BASE移动且EEF显式世界保持、停止/失效/重置。实际分配最多8个有界启动；warmup、站立过渡及settling都计物理积分。

基线门沿已有位置RMS0.015m/姿态RMS0.05rad/源时钟残差0.025s及原native安全，不能因整机难就放宽。运动命令初始取经native资格允许的保守包络，详接口文件；检查实际有符号运动，而非仅看topic在发布。零意图应能减速保持，无BASE字段允许WBC自主选择，两种模式分别测试。

若现役WBC在正确参考下仍不合格，停止该整机执行线并给出同刻request/target/actual/力/solver/clock证据；不重训WBC、不扫gain。独立旧模型评价或CPU接口准备可继续，但不把M1写PASS。

## 6. S2：完整整机的数据与场景要求

保留T1抓取—搬运—放置—释放—撤离，T2桌面推送—脱离—保持。两任务由同一初态、相同物体/目标/相机条件配对，仅语言不同。

三类任务布局：
- W0 free-standing near：臂可达范围内操作，根仍自由、腿RL支撑，不等于固定底座；Teacher BASE软偏好近零。
- W1 translate-required：物体或目标在初始固定base合法臂工作空间之外，需要受控前移/侧移再操作；至少一半移动数据应包含平移，不能用随机漂移计作底盘参与。
- W2 turn-and-approach：要求改变朝向再接近，或低速底盘协同搬运/推送。先在有地面/桌边安全余量的短行程场景，不引入台阶/复杂避障/新物体形状。

W1/W2必须在采集前用原几何和完整碰撞约束记录静止base可达性依据（不能把一次局部IK失败等同严格不可达），声明完成所需base位移/偏航。初始移动目标范围从0.12–0.25m、偏航0.15–0.30rad内选取，由实际资格进一步收窄。若只能证明base有运动、不能证明任务必需，则标BASE_PARTICIPATING而非BASE_REQUIRED。

Teacher为任务规划器，使用物体真值产生EEF与独立BASE意图；可用几何/IK/确定性优化选站位，不把WBC输出速度当目标真值。尽量先靠近再做精细抓取；保持/释放阶段具有非零比例的零速度参考。合法非零base命令、正常停止和转向方向都必须覆盖，不能所有label为零。

### 分阶段小包

资格pilot：2个独立场景组×T1/T2=4条，覆盖静态支撑与含平移/转向的移动操作；全部原任务持续成功、无native违规，方可进入正式数据。pilot不混入TRAIN。

`v4.0-mob13`启动包：12 TRAIN场景组（W0/W1/W2各4）×2任务=24条，4 DEV组×2=8条。每任务TRAIN至少11/12准入、DEV至少3/4准入且每种移动类别有有效覆盖，全部首次尝试终态后冻结。未完成不能凑出不存在的H40窗口。

`v4.1-mob13`扩展目标：累计24 TRAIN组（各8）×2=48条，8 DEV组×2=16条；包含v4.0，不额外再收一份。每任务TRAIN至少22/24、DEV至少7/8准入；较晚数据只归档，不热加入已开始训练的v4.0。

正式单episode默认32秒、20Hz观测，641观测/640提交/601完整H40。起立和reset后的warmup单独记录且计预算，不伪装为任务帧。需要更长任务时先封存新episode时长并按N_command-H+1计算窗口，本子阶段不暗改时长。所有失败与接触事件完整保存，不能用只保留成功轨迹的输出隐藏WBC能力过滤。

split在场景组级固定；两任务配对、同站位近重复、同物体/相机种子留同一split。旧DEV、确认集不用于新TRAIN构造。Teacher失败不能替换seed直到分母好看。

### 动作与观测

按新接口保存EEF13：EEF10保持物理语义，BASE三维为有界速度偏好。policy输入补可观测base状态，推荐state19；不输入Teacher阶段、期望base/物体真值/未来MPC答案。记录requested/installed/WBC-selected/achieved四条链及实际机身相机图像。

数据只有实际全身动力学产生才标FULL_GO2_ARX5_NATIVE_WBC。原v3.x仍标FIXED_BASE_ARX5，不能补三零改名。原完整任务规格和失败账继续独立保留。

## 7. S3：真实多worker与数据管线

复用已验证 `scripts/pi0294/dispatcher.py`、`cpu_pipeline.py` 的调度/预算/原子输出思想，增加真正的fullbody worker插件；旧fixed-base worker不改标签复用。

每个worker有独立MuJoCo data、ROS namespace/domain、OCS2 solver/reference manager、腿RL history/gait phase、arm/gripper state、时钟、CPU affinity、渲染上下文与输出目录。只有只读模型权重/mesh可共享，不能多个worker写同一MPC目标topic或live总账。

native环境比旧固定臂复杂，先1worker通过，再2worker与串行的同seed诊断比较。确定性数值和任务结果/安全必须一致或满足冻结的紧容差；不能用统计吞吐掩盖串台。4worker只有实测资源与隔离资格通过才启用；不承诺旧2.86x自动适用于全身MPC。

整项CoreLab总并发仍16核/96coreh，与原R4审计/旧任务共用。起步每worker最多4核（MPC+RL+渲染），2worker运行时留核给审计和通信；4worker须削减单worker或暂停其他活动，不能16核采集外再开未计的审计线程。BLAS/Torch/OMP设置线程上限。

记录真实worker与optimizer时间区间、episode/h、coreh/合格episode、MPC计算/仿真/渲染/图像/打包耗时。评价优先，采集在episode边界降并发；训练必须有本地已验证数据，不能实时读仍写入的文件。

原R4的48条固定臂新增额度尚未用完的部分优先转给本整机TRAIN，累计新TRAIN上限仍48，pilot累计上限仍4；已有旧pilot/采集消耗扣减，无法覆盖完整目标时先交启动包或partial，不能追溯清零。

## 8. S4：EEF10模型怎样变成可学习BASE的EEF13

不能直接把M2048的padding槽10:13当已学会的base命令。新schema需要真正的标签、输入处理、mask与损失。官方132D padded内部宽度可能不需改，但新增3维的输出/归一化/模态语义必须全部显式验证。

默认新分支：`FB13-Uxxxx`，以现有M2048模型参数为warm-start，不从pristine重训；为新增BASE接口做有界初始化，冻结VLM，训练原projector/动作DiT及新增state/action适配参数。用当前fixed checkpoint做被mask前三10维的数值兼容检查，不能让旧接口读出任意base动作。

**这是动作/观测/数据域干预，不是假称原optimizer精确续训。** 原M2048及它的Adam/scheduler/RNG保持；FB13建立自己的优化器/采样/scheduler和`parent_model_sha`，标签为WARMSTART_NEW_ABI_DATA_INTERVENTION。旧C0若还继续则保持原完整恢复链，两者不相加成一个global_step。

新mobile TRAIN拟合独立`v4.0-mob13`统计量，只用TRAIN；保留旧v3.x统计原件。第一版mobile训练默认只用完整EEF13 mobile数据。旧EEF10没有base监督，不补零做target；后续确需混合时仅在逐维loss mask和缺失state处理真实合格后新增显式masked-replay设计。

这也正面处理了R4发现的v3.1在v3.0统计下codec失败：不扩大旧门、不把新stats热塞旧训练；新域用新合法ABI/统计与独立分支。所有state/action clip、单位、relative一次转换、旋转与速度round-trip都独立验收。不能只检查shape(40,132)。

冻结算法：官方N1.7原生masked flow matching、BF16、确定性math SDP、VLM frozen、有效batch32。默认lr1e-4，FB分支目标2048更新、warmup占5%、自己的2048步schedule；保存U512和U2048。新branch累计正式更新最多2048，属于替代原可选C1的额度，R4所有分支总新增上限4096不变。

CPU DataLoader0/2/4与有界cache只能在新增schema成立后限时测试，真实端到端快至少5%且语义/消费/恢复一致才采用；最多16隔离工程更新与原R4共享，不搜索microbatch/attention/新loss。旧v4.1在CoreLab扩采时，Hopper只读冻结v4.0训练。模型已有本地完整checkpoint即可评价，HF归档失败不杀健康训练。

## 9. S5：完整系统如何测试，防止看起来像整机

先把原M2048＋无BASE偏好的EEF-only参考接入完整WBC，作为迁移基线；这不是声称M2048能预测base。新FB13输出完整13D，桥接后EEF+BASE都由同一冻结WBC实现。

开发面板至少有同一场景两个指令，W0/W1/W2覆盖；报告任务成功、实际base有符号位移/偏航、脚地接触、身体姿态、碰撞、力/关节、EEF跟踪、base意图与WBC选择差异、停止/释放和全时长。随机漂移不能算完成移动意图。

重要对照：同一FB13、同一噪声/输入，启用base preference与省略base preference比较。省略不是把速度置零；只在训练开发布局做首轮4条ablation，不用确认集调参。若没有base偏好也成功，要如实说明WBC本身已提供移动；不强求编造base-head收益。

### 安全评价模式的范围变化

固定臂S1的完整状态副本包含的是旧Engine，不能直接宣称它覆盖分布式OCS2、腿history、MPC policy和reference manager。整机必须重新验证时间、状态及控制器一致性。

本用户要求推进完整仿真，允许在Teacher资格成立后开展**有界单worker整机原生安全停止**模式 `FB_NATIVE_ABORT` 的模型开发：500Hz原生限位/接触/稳定性检测，首违规即时冻结仿真并终止本条，保留全部分母；仅仿真，绝不映射到硬件。其结果不是旧`D0+S1`模式，不能合并成功率。这个显式模式避免因尚未实现可克隆的整个异步WBC系统而无限阻塞模型测试。

可选 `S_FB` 只有整个MuJoCo+MPC状态/腿history/控制缓存/clock/RNG均能复制且matched-prefix检查成立后启用；否则标NOT_QUALIFIED，不伪造影子安全保障。原固定底座评价仍按旧S1约定。模型的真实native失败关闭该新live臂进行原因审计，离线数据训练没有共同输入损坏时可继续。

### 成功层级

`FULLBODY_TEACHER_READY`：原生完整body+WBC、4pilot两技能成功，含有意图的base移动，实际动力学与时间合格。
`FULLBODY_POLICY_DEVELOPMENT`：同一FB13模型在8条DEV中两任务各至少2/4，移动类有成功且无真实安全违规，才选定一个候选进入后续核心。
`FULL_GO2_ARX5_MULTITASK_SIM_WORKING_V1`：固定模型/同一WBC，独立8组×两任务，T1/T2各>=6/8、同组均成功>=5/8；至少4组属预先声明移动必要/参与类且移动类两任务各>=3/4，目标区域/释放/保持成立；零实际安全违规。运行模式（FB_NATIVE_ABORT或合格S_FB）写入名称/报告。

Teacher同核心场景各任务>=7/8且无安全违规才开启模型确认；不按Teacher失败更换seed。保留8条近场/旧局部技能回归；预算优先DEV与移动证据，资源不足不虚构核心结果。全部候选/部署身份分开，deployment本轮NONE。独立确认未开封前不用于数据/控制修复。

## 10. 工程问题自主修复与停止边界

允许自主：可重复构建/缺失依赖诊断、原环境路径适配、离线缓存、ROS domain隔离、仿真/ROS时钟、包格式/shape/mask、坐标与四元数、reference identity/ACK/TTL、完整reset、夹爪/相机资产集成、native模式切换、只读记录、渲染headless、worker入口/原子账、checkpoint文件I/O。小型修复在合适测试与固定diff后继续，不每项询问。

循环：复现最小错误→定位→最小patch→CPU契约→隔离短仿真/资格→封存新source/plant/bridge→在正常阶段边界采用。修复失败是有信息的记录，不连续三轮仅重复同一状态；同一子系统主动修复时间盒最多2h，预算内最多3个独立工程attempt，成熟其他线继续。

不允许借工程名义：调WBC权重/任务成本/控制增益/摩擦以追分、删除真实失败、放宽安全/评分、剪切真实qpos/qvel、隐藏外力、换teacher答案/场景seed、用未执行的轨迹当真数据、公开原始资产/视频、用新T0归零。若诊断证明真实资产模型错误，产出最小且版本化修复，作为本场景overlay的plant修订重新跑资格；不得把物理模型改变后的分数同旧域直接合并。

### 对过去串行阻断的纠正

固定底座策略不完美不阻止完整场景建设；完整WBC尚未合格不阻止官方codec/旧模型离线工作；数据慢不阻止已冻结包训练；归档错误不阻止本地评价。共同输入/权限/总预算损坏才停止相关全部范围。动态实际失败不可以靠持续增加预算或篡改STOP变PASS绕过。

## 11. 资源不是新增一套预算

所有R4/R4A消耗统一扣父账，已花/在途预留先减。总上限仍：3PBS、10H200allocated小时（项目1GPU、账号3需真实已有权限）、4096新增正式更新、16隔离工程更新、Core96coreh/并发16核、Hopper32coreh、160个main/Teacher/eval启动、2560000主积分、102560真实RGB对、2000000shadow积分/10000shadow调用、6000高层生成意图；Core96GiB/Hopper180GiB/local4GiB新增峰值、Hopper20GiB个人余量。

新增子项建议占用而非叠加：完整WBC/桥资格<=8启动，pilot<=4，新TRAIN<=48和新DEV<=16，串并等价<=4启动；余量优先真实模型DEV/移动对照/核心。reset/settling积分及所有WBC CPU和低层推理都记账。目标总量是上限，不保证所有完整包能在余额内同时完成；需缩减时先停止v4.1扩采和可选旧C1，再保留v4.0与实际模型行为。

如果现在R4剩余窗口不够完成移动训练，只交完整环境/Teacher/数据/准确恢复清单；不宣布双任务成功，不另外自增PBS或下一夜窗口。用户只需在真正资源扩容或WBC学习/实机等范围改变时决定，不需要每个普通工程修复都再GO。

## 12. 必须提交的证据与最终报告

`R4A_SCOPE_ADOPTION.json`：引用本次用户原文、父R4授权和实时余额；不是助手自行新增资源。
`WBC_INTERFACE_RECEIPT.json`：实际package/consumer链、7/14/12含义、baseTracking开关/权重、mount/clock、受支持profile。
`FULLBODY_MODEL_MANIFEST.json`：free root、腿/臂/夹爪、质量/碰撞、实际相机、固定controller和leg-policy、overlay差异。
`BASE_INTENT_BRIDGE_TESTS.json`、`FULLBODY_TEACHER_QUALIFICATION.json`、`MOBILE_DATASET_CARD.md`、`MOBILE_DATASET_MANIFEST.json`。
`PARALLEL_WORKER_QUALIFICATION.json`、`PIPELINE_TIMELINE.jsonl`、`MOBILE_CODEC_AND_MASK.json`、`FB13_WARMSTART_RECEIPT.json`。
`FB13_TRAINING_RECEIPT.json`、`FULLBODY_DEV_RESULTS.json`、`BASE_PREFERENCE_ABLATION.json`、条件核心与回归、原视频私有索引、`RESOURCE_DELTA.json`、`FINAL_REPORT.md`。

代码/小证据正常提交现役GitHub分支并回读；#11科学事实、#47工程/资源。只有明确允许的两个脱敏checkpoint/60GiB仍在父授权余额内可公开；新增原始整机数据、机器人mesh、WBC源码/权重、私有路径不得公开。不要重传M2048来充当新的进展。

最终首先回答：真正全身自由动力学是否跑了？BASE偏好被哪一层消费？是否有超出静止臂可达范围的真实任务？同一个checkpoint能否按语言完成两技能？预检/执行器干预是多少？多worker与Hopper实际优化有多少重叠？未完成项和独立结论逐项列清，不以代码编译/asset存在替代物理和策略效果。

## 13. 审核来源

- 固定私有归档哈希及模块静态核查：见 `docs/interfaces/EEF13_BASE_SOFT_REFERENCE_V1.md`。
- 本项目既有WBC审查：`docs/coordination/2026-10-10_WBC_PACKAGE_COMPARISON_AND_TARGET_LEARNING_REVIEW.md`。
- 研究章程：`docs/RESEARCH_CHARTER_TASK_SPACE_MANIPULATION_V1.md`，本增补改变目标域适配与BASE意图范围，不改写过去的独立数据声明。
- 最新R4采用：#11 comment6173988141，`reports/PI0294/20261014T170930Z-incoming-pi0294-r4/AUTHORITY_ADOPTION.json`。
- 最新R4旧统计数据视图失败：同目录`V31_V30_STATS_CODEC.json`。失败原件继续保留。
- 公开方法背景仅说明可分层，不证明本项目已达成：https://umi-on-legs.github.io/ ；https://leggedrobotics.github.io/ocs2/optimal_control_modules.html 。

**本文件更新的是实际研究终点和执行范围。整机环境、EEF13策略、BASE桥与新多worker适配在有执行收据前均为待实施，不因本次GitHub发布自动变成PASS。**
