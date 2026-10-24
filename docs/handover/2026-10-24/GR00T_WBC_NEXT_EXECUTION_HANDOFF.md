> **现行执行版本已统一：** [V4A 原生完整面板配对eval](../../protocols/GR00T_WBC_V4A_EVAL_FIRST_STAGED_EXECUTION_20261024.md) 优先（A24+B24，原noise294101、最多2PBS、仅在两个完整作业可准入并遵守截止时执行）。[V4.1 小诊断备选](SCIENCE_FIRST_MINIMUM_SPEND_EVAL_V4_1.md) 的16条新增条件当前缺少原生子面板支持，**不是现行已准入任务**。两份不能同时提交或把一份视作另一份的追加授权；先出匹配真实结果再考虑训练。

> **最新owner接续请先读 [V4A：Eval-first 分阶段执行修订](../../protocols/GR00T_WBC_V4A_EVAL_FIRST_STAGED_EXECUTION_20261024.md)。** 本交接下方B2048／运行中恢复和原矩阵顺序为历史快照：A、B4096现均完成训练，B4096只读DEV例外已被精确批准并双账本采用；B0旧有效14/48、A/B DEV各0/48、S1 core64=0。当前先完成A/B同原封存bank294101各24条件的真实配对Eval，在审查结果后再决定第二bank、补齐B0或后继训练。不可借现有大型global cap自动派发6面板，旧valid失败不重跑，维持单owner与所有真正原生资格、原截止、DEC1及最终保护。

> **历史V4.1备选：** [V4.1 Eval-First/Minimum-Spend](SCIENCE_FIRST_MINIMUM_SPEND_EVAL_V4_1.md) 覆盖下文旧「整包130条」执行预期。先按原DEV固定四几何八task条件测试A/B各一次，回收16条新行为结果，再根据配对差异及实际成本考虑下一阶段。已采用全局额度只是cap，实际不得提前为六个完整面板全部占用；S1矩阵/最终保护、原时钟及历史成本不改。本交接下方旧训练／重试状态属于历史，不可据此重新启动B。

# 当前接续交接：GR00T＋冻结 WBC（2026-10-24）

> **当前接续入口已更新为[科学主线优先 Goal V4](../../protocols/GR00T_WBC_SCIENCE_FIRST_LONG_HORIZON_GOAL_V4_20261024.md)。** A/B4096已完成训练，B完整状态审计及资源关闭已完成，后续复用准入仍待闭合；目标零步feedback／双相机／STOP已实际完成并关闭，完整prefix／安装／任务资格仍待完成。下文B2048恢复、1481574运行和重复准入步骤仅保留为01:46历史，不能据此重启已关闭作业或零步请求。当前按V4的两条科学线、原预算与原截止继续。

**用途：把进度审核与V3计划交给现役owner；本文件不创建新科学GO、作业、资源或部署。** 已明确批准的PBS17/28不重复申请；后继研究的拟议规模不被本文件自动采用。

完整审核基线：36925d272095d000ba1ad3ce5e9cdc44373fdce6（01:37 SGT）；**发布前增量快照：5d9f2a82a6aa4830481f7197298f666cdc4aad0c，2026-10-24 01:46:45 SGT**。实际接续前读取唯一owner的当前ledger／job／lease；本文是快照，不是持续在线状态。

发布并发复核：**5c9a6de763c6f74b7731809f72b2dd54e6ca47de，2026-10-24 01:59:22 SGT** 新增[1481574共享终局契约及精确身份CPU验证](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/5c9a6de763c6f74b7731809f72b2dd54e6ca47de/reports/PI0294/design/parallel-science-20261022/B_PRECUDA_RETRY1481574_SHARED_TERMINAL_CONTRACT_CPU_20261024.json)。它尚未产生原生终局审计／结算／关闭，未宣称首次更新；运行capsule仍为36925d27，上述科学与训练实际状态不变。

- [详细进度审核](../../reviews/2026-10-24_GR00T_WBC_PROGRESS_AUDIT.md)
- [长程计划V3](../../roadmap/2026-10-24_GR00T_WBC_LONG_HORIZON_PLAN_V3.md)
- [最新1481574实际采用、唯一提交及R／MODEL_START回执][S11]
- [PBS17/28实现与历史原生边界][S1]
- [已批准的精确两项PBS调整][S2]
- [上一次失败结算、存储修复与完整剩余预算][S3]

## 1. 先按最新终局纠正状态

| 项目 | 最后可证状态 |
|---|---|
| A_CONTROL | 4096完整、已关闭；不从0重训 |
| B_COVERAGE | 完整B2048／cursor65536；2970为失败作业末步，不是完整恢复点 |
| B1480817 | token到期／HeartbeatLost，922更新未保存，已结算 |
| recovery2 1481489 | pre-CUDA存储准入失败，F/Exit1；0模型／query／update／physics；265GPU秒、1180CPU秒、64MiB已结算，Vanda双lease关闭 |
| 新重试1481574 | PBS17/28已实际双once采用；新request与唯一qsub已完成，native R且Vanda驻留owner健康；MODEL_START已放行，FIRST_UPDATE未放行，未观察新更新 |
| 当前科学证据 | N1 T1 10/20、T2 4/20；新A/B终点评估未完成；B0 14/48、9成功5失败 |
| 新source progress／target任务链 | 学生原生资格未闭合；core64=0，NOT_RUN |

后续若已有更晚的原生回执，以同一身份链的实际回执更新状态，不重新执行已完成采用／提交／结算，不把旧Q、旧草案或CPU PASS当作当前运行。

## 2. 近程优先级与并行关系

当前批准必需剩余工作是：

1. 保护并跟进已提交的1481574，从完整B2048恢复至4096；保持原配方、原数据／统计／模型父身份。
2. 复用现有source／target路径完成原有限commissioning；具有完整资格后执行固定旧CORR／R0的DEC1核心比较。
3. 在原时钟内输出真实终点／部分结果、未运行条件、失败和结算。

B训练优先使用Vanda A40；现场控制驻留Vanda服务器，由单一owner执行。源／目标CPU接线可在其既有有效范围内并行准备，不阻断健康训练。DEC1固定旧CORR，不等待新A/B赢家或完整B0；PV1选模／LOCKBOX才需要完整B0。

**不在本轮末期插入新的R1训练、DAgger、EEF13、世界模型或多候选规划。** 后继阶段已在V3给出单独工作包与条件。

## 3. B恢复的有限接续路径

### A. 先确定现场属于哪种状态

- 已有同一合法请求的活动作业：沿既有owner／watcher继续，不创建第二个dispatcher、不重复qsub、不热改capsule。
- 已明确终局：保留原失败和费用，按其原关闭流程确认一次结算／独立回读；不重放历史请求。
- 只读返回不完整或可能写入的结果未知：保存UNKNOWN，使用原精确身份回查，禁止把未知当不存在或再发同一写入。
- 本快照的“1481574为R、MODEL_START已放行”不替代接续时的新鲜原生查询；不能由此推断模型已加载完成、FIRST_UPDATE已放行或实际新增更新。

### B. 继续已提交1481574，禁止重做准入／提交

最新实际回执已确认双账本17/28采用、原始source36925d27精确CI与1019文件完整原生回读、新request及唯一qsub。原storage冷修复的658827016192bytes生命周期峰值已经绑定；旧588617300992bytes计划仅作失败历史。[S11]

1. 核对同一job1481574、原request／nonce、source、双lease、后台owner实例与write journal。实际健康时继续原watch，不重启、不创建第二owner，不再采用PBS标记或重新qsub。
2. 保留9666已结算更新、1214父campaign查询、91003global GPU秒及全部旧损耗。当前PBS已计global25／DEC1 14；B预留22560GPU秒、54000siteCPU秒、2048更新、29query和70000000000bytes输出。
3. 保持B2048完整状态与cursor65536，R0、sampler、optimizer／scheduler／RNG、batch32和独立父权重不变。新源码发布不热切正在使用的capsule。
4. 由Vanda驻留owner按原release／heartbeat／STOP路径继续。MODEL_START已放行，不重复释放；FIRST_UPDATE仍须原流程及对应模型／梯度／更新回执。Q、R或启动放行本身均不等于实际新增更新。
5. 按原有限观察与认证处理规则跟进；SSH或返回超时不证明任务已死。对未知写入只做精确回查，绝不重复提交／释放／结算。
6. 实际到4096后，按原typed terminal路径核对完整checkpoint、消费游标、兼容probe及query/load差额、原生terminal与owner退出，再完成一次结算、独立关闭和索引发布。
7. 若新作业失败，保留准确终局与恢复点，按原范围处理；剩余三次PBS已保留给矩阵，不能自动再挪一次重试。下一动作只由当时有效范围与完整预算决定。

复用现成launcher与typed contracts，不发明新CLI或额外审计架构。完整PV1 worker2资格未取得时保持worker0；小CPU fixture不构成热切数据或数值路径的依据。[S1] [S2] [S3] [S4]

### C. 本轮恢复的停止出口

原预算和截止无法容纳、原生输入身份不成立、实际安全失败或不可消除的UNKNOWN，均按对应原规则停止依赖动作，留下单个精确阻断。不能用新长程计划重新累计额度或自动增加重试。原已批准17/28范围内部的普通冷修复不再逐项索取重复许可。

## 4. source／target任务接入只补真正缺失部分

### 固定身份

- Source：固定基座ARX5、独立IK/PD及原S1，single-pose路径按20Hz参考网格依次执行prefix8，500Hz物理推进，共0.4秒／200tick；不强套九点WBCPreview ABI。
- Target：原自由基座Go2＋ARX5、冻结MPC/WBC／腿策略／臂PD、现有可动夹爪／物体／相机overlay与原生reference manager。
- 两接口共同使用正确frame／TCP／rot6d／夹爪／时间实现；唯一干预为I_TIME与I_PROGRESS的进度锚定。
- 本轮学生固定旧CORR@2048／R0，不用新A/B终点替换，不混target任务示范入TRAIN。

### 最小闭环证据

在原有限commissioning内，用已实现的ActualState／FrameRecorder、target observation、mapped ACK及relay冷接线，取得：

当前实际观测与时钟 → 当前策略query → raw及mapped完整H40 → 原生prefix安全回执 → 安装确认 → 获准物理推进 → 实际反馈／ACK → 下一观测及可验证STOP／任务评分。

每条连接应有实际对应的query／参考／安装／tick／state身份。CPU合成fixture、12D MPC预测、renderer PASS均不能替代全身prefix安全或学生执行证据。资格检查验证链路可运行、可测量、可停止；不以学生必须先完成目标任务作为打开确认集的筛选条件。[S5] [S6]

### 已用条件不能重置

Source02的四个原commissioning条件已有有效reference/S1拒绝；Source01工程UNKNOWN与replacement保留。原最多8个commissioning条件的剩余额度必须由当前ledger核对，不写“重新开始8次”。这些组件试验不填入核心学生64格，合法拒绝不因想取得PASS而重跑。[S7]

## 5. 科学执行与验收

### DEC1

有资格时，完成8新几何×2任务×2执行器×2接口的64格。保留配对noise和全部attempt，以几何block分析目标progress−time、源域同类差及交互；报告成功、S1／原生拒绝、完整任务失败、工程不完整和NOT_RUN。

若目标仍未资格，保留源侧合法结果，目标标TARGET_EXECUTOR_BLOCKED／NOT_RUN，不计算虚假的transfer差。若源策略两接口都弱，报告SOURCE_POLICY_LIMITED，不把低迁移落差解释为鲁棒。原可选stress32只有核心完整、原先登记并且全部资源仍容纳时才做；当前88episode矩阵封套不自动意味着可再加32条。[S8]

### PV1

完整B0、A4096和B4096的匹配DEV是后续重要任务，但当前剩余B＋DEC1封套没有自动包含额外最多130条源DEV。先检查仍有效的既有范围与资源；未覆盖的工作按V3工作包一次核算，不挤占已保护矩阵或最终验收。

正式DEV资格沿原协议：每任务≥18/24、native fault0、相对完整B0每任务回退≤1/24且worst-task净改善。之后按原seed复核和唯一候选LOCKBOX80授予VALIDATED／BETA，不能降低门槛或反复打开锁箱。[S9] [S10]

## 6. 固定资源与时钟

### 当前未完成工作封套（B已经提交）

| 工作 | GPU秒上界 | PBS | 更新 | 查询 | Episode | ticks |
|---|---:|---:|---:|---:|---:|---:|
| B2048→4096 | 22560 | 1 | 2048 | 29 | 0 | 0 |
| 原DEC1矩阵＋commissioning | 86580 | 3 | 0 | 7040 | 88 | 2816000 |
| 合计 | 109140 | 4 | 2048 | 7069 | 88 | 2816000 |

当前提交后DEC1/global PBS已为14/25，表中B的一次PBS已计used，不再额外提交；剩余三个matrix作业后恰达17/28。DEC1更新9666＋原B预留2048恰达11714；global另保留20GPUh／40GiB，包含于总上限。完整未来预算投影的global GPU站点存储余约38.1MiB、DEC1 LOCAL CPU余约1.025s；这些不是现场free容量。预算不因文档发布或已失败作业零更新而退款。[S3]

### 原停止时点（SGT）

| 边界 | 时间 |
|---|---|
| 停新作业／候选 | 2026-10-25 02:29:56 |
| 停新训练块 | 2026-10-25 14:29:56 |
| 停新物理 | 2026-10-25 18:29:56 |
| Hard close | 2026-10-26 02:29:56 |

到期按原协议收尾并提供实际恢复点／未运行条件；V3的8–12周参考窗口只适用于后继已成立的阶段，不能改写当前时钟。

## 7. 下一次报告的最小内容

在现有报告／ledger中补齐，不再新建平行管理系统：

| 类别 | 需要的实际内容 |
|---|---|
| 训练 | A/B完整恢复步数、当前原生job／owner／lease、实际新更新和保存状态 |
| 目标接入 | 本轮source／target前缀、观测／ACK／STOP资格，最后实际物理／query计数 |
| 科学 | core64各格状态；源DEV完整性；成功与有效失败及工程缺失各自分母 |
| 资源 | 当前used／reserved／完整future／final保护、旧损耗、一次结算与真实关闭 |
| 结论 | ACTUAL_ADOPTED_AND_CONTINUING、SUPPORTED_PILOT、接口收益不确定、SOURCE_POLICY_LIMITED、TARGET_EXECUTOR_BLOCKED或INCOMPLETE_RESOURCE_OR_SITE；仅当区间已排除预声明最小有用增益才报告无实用增益，并附证据 |

无需把每个CPU测试数或旧追加记录再次抄入首页。更新简短状态入口和固定证据链接，保留历史原文。代码发布、实际部署、运行资格和科学结果分别记录。

## 固定证据

优先读[当前实际提交][S11]、[完整资源][S3]和[原DEC1协议][S8]；其余编号在正文相应位置可点击。

[S1]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/parallel-science-20261022/B_PRECUDA1481489_PBS17_28_RUNTIME_CPU_20261024.json "最新PBS接线"
[S2]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/parallel-science-20261022/B_PRECUDA1481489_RETRY_PBS_AUTHORIZATION_20261024.json "精确批准"
[S3]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/parallel-science-20261022/VANDA_RECOVERY2_PRECUDA_STORAGE_FAILURE_CLOSED_AND_RETRY_PBS_DRAFT_20261024.json "实际失败与完整未来资源"
[S4]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/parallel-science-20261022/PV1_ACTUAL_MP1_LOADER_WORKER_FORWARDING_CPU_20261023.json "worker资格边界"
[S5]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/WDR1_TARGET_MEASURED_OBSERVATION_EXISTING_QUERY_CPU_20261023.json "观测／query接线"
[S6]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/WDR1_TARGET_TIME_ACK_EXISTING_RESIDENT_CPU_ACTUAL_20261023.json "原生ACK绑定"
[S7]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/WDR1_SOURCE02_FINITE_NATIVE_PROGRESS_AND_B2_CLOSED_COSTS_CPU_ACTUAL_20261023.json "source历史尝试"
[S8]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/docs/protocols/DEC1_CONTROLLER_DISJOINT_VLA_SCIENCE_GO_20261021.md "DEC1协议"
[S9]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/reports/PI0294/design/parallel-science-20261022/EVALUATION_AND_CLAIM_PREPARATION.json "源评估资源边界"
[S10]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/36925d272095d000ba1ad3ce5e9cdc44373fdce6/docs/protocols/PI0294_PV1_MULTITASK_POLICY_RELEASE_20261019.md "PV1验收"


[S11]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/5d9f2a82a6aa4830481f7197298f666cdc4aad0c/reports/PI0294/design/parallel-science-20261022/VANDA_PRECUDA_RETRY_PBS17_28_UNIQUE_GPU_SUBMISSION_ACTUAL_20261024.json "1481574实际采用、唯一提交、R和MODEL_START放行"
