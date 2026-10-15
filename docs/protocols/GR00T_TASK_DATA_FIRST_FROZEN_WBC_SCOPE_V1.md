# 主线纠偏：冻结 WBC，按物体任务采数据和训练 GR00T

日期：2026-10-15。
Decision ID：`GR00T_TASK_DATA_FIRST_FROZEN_WBC_V1`。
状态：`ADOPTED_USER_DIRECTION_AND_GATE_DEPENDENCY_CORRECTION`。
证据基线：`c5f6da4e8bed83e4971152ebee2e7ed51a422bf2`，#11 comment6183939636。

## 1. 用户的最新决定与本文件效力

> 我们不要调整WBC，我们只是想要借用这个环境去采集数据，这不是我们训练gr00t需要考虑的

本决定明确终止上层团队继续开展WBC保持偏差归因、增益/前馈/重力/积分补偿候选、控制器设计或等待WBC负责人修到某个RMS的工作。R4B D0已完成、缺陷包保留；R4B D1不作为下一任务，不因还剩诊断槽位就执行。

**WBC是固定的环境执行组件，GR00T团队负责任务Teacher、真实同步数据、语言条件学习和任务闭环。** 不再把控制器性能研究当作GR00T采集/训练的主线或永久前置。

本文件据用户直接指令修订R4A/R4B的依赖关系：原生恒定EEF保持的0.015m/0.05rad质量门继续用于描述历史控制实验，但不再自动阻断新的物体任务pilot、成功示范准入或GR00T训练。它不是把0.05改成0.08，也不是改写旧FAIL。

修订仅限WBC几何跟踪质量与任务数据准入的关系；原生物理安全、输入/参考身份、源时间、数据完整性、资源和权限门继续有效。WBC质量未合格不等于环境已经能完成物体任务，需要直接进行任务验证。

本次不新增CPU/GPU/PBS/物理次数或新的时间窗口，不重开已关闭lease。执行owner刷新现有任务卡/实际资源，记录本方向修订；有仍有效的对应采集/训练权限则按本修订接续，无有效窗口时合并报告一次所缺资源，不为每项普通工程修复再请求GO。旧未采用的D1权限不能转为新的计算许可。

## 2. 当前证据的正确解释

- R4A原native free-root/12腿/6臂保持：位置RMS约12.54mm、姿态约0.06965rad，安装/时钟及原native安全通过。历史质量FAIL保留。
- R4B D0：joint3/4离线替换到记录目标使姿态RMS从0.07043降至0.00956rad，但位置升至17.58mm；这是反事实，不是真实修复。没有证明传递丢失、力矩饱和或可自主修复的WBC实现bug。
- 这些结果尚未证明“原样WBC不能完成抓放/推送”，也未证明“可以完成”。不能用一个孤立保持实验替代真正的任务pilot。
- 完整活动夹爪/物体/移动机身相机尚未取得任务资格，v4.0/FB13还没有。M2048仍是已有固定底座EEF10模型，不是整机13D模型。

上述信息只决定我们保留哪些风险和源身份，不要求继续研究WBC内部根因。下一项有价值的证据应是原样环境中的真实物体操作，而不是更多FK或重力表。

## 3. 三类门必须分离

| 类别 | 判据 | 对采集/训练的影响 |
|---|---|---|
| 运行安全与数据有效性 | 原生限位/非法碰撞/跌倒和稳定性保护、finite状态、合法参考/时钟、实际观测与命令一致、ACK/reset有效 | 仍为硬门；失败停止相应运行并保存原件 |
| 物体任务结果 | T1真实抓起/搬运/释放/撤离/保持；T2真实桌面接触推动且无抓举、分离/保持；移动任务有实际BASE参与 | 决定是否可作成功行为克隆示范，以及策略成功率 |
| WBC跟踪诊断 | EEF requested-versus-achieved位置/姿态RMS、base意图偏离、joint target偏差 | 记录并报告，**不再单独作为全局采集/训练阻断** |

成功示范的基本规则：
`ADMIT_SUCCESS_DEMO = integrity_ok AND native_safety_ok AND task_success AND label_schema_ok AND split_ok`。
这里不包含`native_hold_SO3_RMS <= 0.05`或`WBC_owner_patch_ready`。

例如一条episode仍有0.07rad参考跟踪偏差，但物体被正确抓起、完整放置且安全/记录成立，应按任务判据判断；不能仅因该参考误差拒收。反过来，末端RMS很好但没有抓起物体，不是成功示范。

任务若确实需要特定姿态（以后插入等），在任务规格中预先声明相应功能要求；不得为了接纳失败事后删去任务标准。现有T1/T2物体成功谓词及原生安全阈值不放宽，不把推到目标伪装成抓放。

## 4. 立即结束和保留的工作

停止继续安排：
- R4B D1的instrumented hold、BASE-hold对照或所谓最小控制器patch复验；
- WBC gain/cost/preview/velocity-scale/solver/重力/积分或腿策略调参和训练；
- 为GR00T加入根据本次0.07rad误差标定的固定反偏置、选择性joint correction；
- 等待WBC负责人回复或整改，作为其他独立GR00T工作的许可前置。

保留：
- 已交付且固定的WBC/腿策略/plant基本参数，以及实际已采用的参考接收与源时钟修复；
- 92份原始证据、R4A FAIL、R4B D0结论及缺陷包，不改写、不删除；
- 完整Go2+ARX5作为主数据/测试域、EEF13与BASE软意图接口；
- 原高频安全、真实状态反馈、有限夹爪动力学和完整视频/数据证据。

本决定不是要求换成理想跟踪器、冻结根、复制反事实关节状态，或让Teacher在背后直接控制关节。WBC仍是实际执行者，所有结果按其固定版本披露。

## 5. Codex接下来做什么：回到现有数据—训练流水线

### A. CoreLab：任务环境与Teacher，不是WBC性能改造

1. 复用统一入口、原生完整机器人和已经正确安装参考的桥。补齐真实有限驱动夹爪、任务物体、机身/腕相机和记录器；只做任务接入必要的工程工作，记录overlay并检查安全/时间/数据契约。不得调原WBC参数，或借overlay改原机器人质量/惯量/摩擦来追分。新增真实夹爪自身的物理参数必须版本化并全流程一致。
2. 直接开展原R4A的小规模T1/T2任务pilot，而非再跑保持质量门。先少量已声明近场/移动场景，全部attempt留分母；不假称已通过未运行的平移/转向或装配资格。基础运行/停止/失效语义仍需测试，但没有额外的全局精确EEF保持要求。
3. Teacher根据实际EEF、BASE和物体状态更新任务阶段、接近点和已声明可行的任务路径。正常的任务反馈/重规划允许；不根据joint3/4误差表偷偷校准WBC或注入固定姿态反补偿。Teacher调试与正式数据版本分开，禁止用重复换seed把正式失败变成成功。
4. 若原样WBC能完成任务，就采集并准入；不能完成的场景保留失败/不可收集原因，继续其他独立已合格场景，不重新进入控制器调参循环。若某一技能完全没有有效示范，必须报告该技能数据缺口，不能声称双任务数据就绪。
5. 合格单环境与隔离检查后，复用dispatcher/cpu_pipeline，从2worker到有资源且合格的4worker。独立ROS/MPC/history/clock/renderer/output，真实fullbody吞吐重新测，不能引用旧固定臂2.86x作为新实测。

### B. 数据：保存真实执行与正确监督的区别

输入：同一时刻真实机身/腕RGB、语言、实际EEF/夹爪及可观测BASE状态。
监督：算法Teacher在当时状态提出并实际提交执行的未来EEF/夹爪/BASE意图；EEF13按已定义模态转换。
旁路证据：请求、安装、WBC输出、实际达到状态、接触、物体结果、Teacher阶段/真值、来源和失败。

**实际状态不能替换成期望状态；已达到的未来轨迹不能悄悄替换原目标动作标签。** 一个期望动作有跟踪误差，不自动使该动作标签无效，关键是输入是实际观测、标签是实际提交且具有已验证任务意义的控制参考，时间与语义一致。

只把通过任务/安全/数据门的成功episode用于本轮成功行为克隆；失败进入独立failure bank，不混成正确示范。失败可以供分析，未来如做纠正学习须有真实纠正标签。

优先复用R4A数据规模：启动包24TRAIN+8DEV，小包真实准入并冻结后即可训练；资源与窗口允许再扩累计48TRAIN+16DEV。这些不是新增授权数量。不能把旧fixed-base EEF10补三个零冒充BASE示范；旧数据按真实来源保留或使用明确缺失维度mask的独立混合方案。

新fullbody统计只从该版TRAIN建立；原codec、时序和字段误差检查继续执行。移除的是WBC几何跟踪精度前置，不是取消原生数据正确性。

数据卡如实称为通过固定目标仿真执行器采集的任务适配数据；成功筛选受该执行器能力影响，不假称严格WBC独立或跨本体已证明。

### C. Hopper：学习任务行为，不学习低层控制修复

复用已有M2048作为模型资产。原固定底座M2048评价若另有有效窗口，与整机WBC保持诊断解耦；不因R4A hold FAIL就宣称M2048失败。

新fullbody EEF13/状态/统计按R4A建立显式FB13分支：模型参数warm-start，VLM冻结，学习projector/动作DiT和必要接口适配，新的optimizer/sampler身份明确。没有真实有效的BASE标签和小包，不制造FB13训练资格；但CPU接口/数据工具可并行准备。

CoreLab扩采后续冻结版本时，Hopper只读当前已冻结训练包，不热混入、重拟合统计或切换WBC。训练完成尽早在同一冻结环境做真实新观测闭环，评价T1/T2成功、失败阶段、拒绝/安全及旧能力，而不以WBC的独立hold指标选择checkpoint。

保持v4.0训练/v4.1生产的两路重叠；不要求把所有场景采完再开始训练。远端归档失败不阻止经过验证的本地模型评价。

## 6. 现有代码门的最小调整要求

不要简单删除`NATIVE_VALID_QUALITY_FAIL`文件、绕过validate_scope，或把所有FAIL都改成非阻断。应增加明确scope版本和门分类：
- 旧协议回放仍得到原`VALID_NATIVE_HOLD_QUALITY_FAIL`；历史判定及hash完全不变。
- 新任务采集scope识别该项为`DIAGNOSTIC_ONLY_HISTORICAL_WBC_HOLD`，不要求其变PASS。
- 任务capture gate仍独立检查权重/环境身份、原native安全、相机/夹爪/时钟/动作格式、物体任务pilot及真正的有效运行权限。
- 当前无有效lease、数据损坏、真正原生安全失败、错误主机或任务未成功，仍被对应门阻断。

增加针对性CPU单元测试：质量历史FAIL不能永久锁死新任务capture；native安全FAIL仍硬停止；任务成功与质量诊断可以独立报告；任务失败不能准入；旧报告不变；缺资源不能启动；Teacher和模型使用同一个固定WBC/接口。

这些是数据/启动器依赖修正，不改WBC控制律。普通构建、路径、时间、相机、字段和队列问题自主修复并小测试后继续；已识别的WBC设计偏差不再是本团队的自动修复任务。

## 7. 机器可读规则与一次性交接

```json
{
  "decision_id": "GR00T_TASK_DATA_FIRST_FROZEN_WBC_V1",
  "status": "ADOPTED_USER_DIRECTION",
  "WBC_role": "FROZEN_ENVIRONMENT_EXECUTOR",
  "WBC_model_parameter_gain_cost_solver_changes": 0,
  "WBC_training": 0,
  "additional_WBC_hold_attribution_or_D1_runs": 0,
  "historical_hold_verdict": "FAIL_PRESERVED",
  "historical_hold_is_new_capture_prerequisite": false,
  "hold_XYZ_SO3_tracking_metrics": "LOG_ONLY_NOT_GLOBAL_TASK_ADMISSION",
  "native_physical_safety": "UNCHANGED_HARD_GATE",
  "reference_clock_and_data_integrity": "UNCHANGED_HARD_GATE",
  "task_success": "REQUIRED_FOR_SUCCESS_BC",
  "demo_targets": "ACTUALLY_COMMITTED_TEACHER_EEF13_NOT_ACHIEVED_OR_COUNTERFACTUAL",
  "inputs": "ACTUAL_OBSERVATIONS_NOT_DESIRED_STATE",
  "keep_all_attempts_and_failure_evidence": true,
  "primary_domain": "FULL_GO2_ARX5_SIM",
  "CoreLab": "TASK_TEACHER_REAL_DATA_AUDIT_AND_QUALIFIED_MULTIWORKER",
  "Hopper": "FROZEN_DATASET_GR00T_ADAPTATION_AND_TASK_EVALUATION",
  "new_resource_budget": false,
  "closed_lease_auto_restart": false,
  "runtime_requires": "EXISTING_VALID_OR_EXPLICITLY_RENEWED_RESOURCE_SCOPE",
  "physics_train_PBS_performed_by_this_document": 0
}
```

现有owner直接读取本固定文档和根AGENTS，刷新#11/#47。一次登记本方向与任务gate切换，复用R4A已定义的接口、数据规模、模型和并行安排，不再创建新WBC诊断项目。任务权限仍有效时按新scope进入物体任务pilot；若已过期，只准备代码和统一报告所需采集/训练窗口，不把WBC负责人批准或hold精度PASS列为所缺条件。

下一报告应优先展示：任务环境必要工程完成情况、实际T1/T2 pilot/数据、已准入episode、模型实际训练/行为和CPU/GPU重叠。WBC内部诊断不再占用主线交付。

## 8. 历史与安全声明

历史R4A/R4B文件、旧goals、FAIL、checkpoint、私有92文件/增量证据和预算不改。此前没有整机数据与策略的事实仍成立。本次方向改动不等于已经采到数据，也不能使未运行的FB13获得PASS。

本文件及AGENTS/START_HERE发布为计划/工作范围更新，没有在本对话SSH、物理、模型、PBS或删除。GitHub仅存自研规范与小收据，凭据、私有原件、视频、模型和WBC资产保持原授权存储。

来源：用户本轮明确指令；R4B D0 `WBC_OWNER_DEFECT_NOTE.md` 与 #11 comment6183939636（证据固定c5f6da4e8bed83e4971152ebee2e7ed51a422bf2）；R4A原生实际报告a6f2e4bc516fd2f4e3d26a9cc9bff161783eec36；已有EEF13/BASE接口和R4A协议。该文件是研究职责和准入依赖纠偏，不是控制器修复结论。
