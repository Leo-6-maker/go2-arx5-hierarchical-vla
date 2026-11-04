# WM02：完成新接触数据、匹配200更新与真实选动作闭环

本轮已实际执行，不是计划交付。新增 **23,424条真实S域转移，WM主更新400次＋续训探针2次，Actor/Critic更新0，目标积分0，完整任务0**。原PR349未合并；独立分支从固定提交 `e3dc384738127d8d24b4ceadf6aeb6debf0cb4fb` 接续，未切换活动source。

结论仍为 **WM相对同观察规则的增量价值未建立**。新模型真实闭环抓持8/8，无动作模型8/8，规则8/8，随机5/8；这些是0.8秒局部抓持，不是完整任务成功。搬运分叉提供了有效考试，但条件模型8/8选择背离目标的动作。目标线已缩小初始化故障范围，并修复停止包装器；原S1/T2隔离仍有效，未启动目标。

可先运行已实现的只读检查入口：

```bash
python scripts/wm02/review_decision_round.py \
  --report-dir reports/WM02/decision-initialization
```

该入口核验本目录证据哈希并返回实际数量，不创建任务、训练、GPU上下文或物理环境。

## 实际执行身份与命令

每个运行使用独立源码快照、原有registry、Journal和有限lease，结束后均CLOSED。全部私有路径不进入Git；实际绝对argv保存在本机 `EXECUTED_COMMANDS.private.json`。以下是路径变量化的真实调用，不将入口描述冒充已执行shell命令。

| 运行 | 完整执行SHA | 实际结果 |
|---|---|---|
| 冻结B0成功运行02 | `d39e75e0c9dc7bb437df0d0efb8eb0dd027fa0a0` | 原20条DEV分叉推理，exit0，0积分/更新 |
| 保存的目标初始化审计、原模型推理副本 | `584b8cc7887aee577bc6caf1abbfa4297f2d1db9` | exit0，0目标积分/腿策略查询 |
| 三类物理锚点B1 | `47ca4fbc2a0e6293ae24f1f7d1b63f4c4b8aeaf2` | 12/12合格，1,344转移，exit0 |
| 原V3在线对照 | `8e23bf9a7ea414ede9ee90b4dc0f42fcf37954d9` | 320转移，exit0；对照缺陷保留 |
| 修正V4在线对照 | `c3550fd80a31a921641dbf6e21dc80a9d4b21b5c` | 320转移，exit0；WM/盲模型/规则均4/4 |
| 新40父布局数据 | `7e41cd69c3c215e91cb2f85688e5426d4a7f4d2c` | 20,800转移，exit0 |
| 原FIT损失/梯度诊断 | `7a1b677dfe5e493c4362d69efad0fdf98e4fd786` | 640窗口，exit0，0更新 |
| GPU1尝试 | `70965d3c8710225045ec6da74ad0784e17207bb5` | exit1，显存准入拒绝，0模型更新 |
| GPU0匹配训练 | `c7dd382654eb0e0caa4573dff0299ac4070b3e9e` | 两臂各200＋1续训探针，exit0 |
| 新8DEV真实闭环 | `d2b1dc065573dd3b4418267c0842d1c17715a357` | 640转移，exit0 |
| 支撑＋原物理动作身份修正评分V3 | `4881ecebb025b6fb8572a6779e62e81d4e82e441` | 冻结模型重评分，exit0，0积分/更新 |

最后一行完整SHA以 [COMMANDS_AND_SOURCES.json](COMMANDS_AND_SOURCES.json) 的实际快照值为准；该文件由运行spec读取，不由提交简称推断。

本机各实际worker调用：

```bash
CUDA_VISIBLE_DEVICES='' "$WM02_CPU_PYTHON" "$RUN/worker.py"
```

worker实际调用下列已实现入口的 `main(argv)`，不是重复运行已完成Journal：

```text
scripts.wm02.qualify_decision_anchors
scripts.wm02.run_frozen_selector
scripts.wm02.collect_decision_data
scripts.wm02.diagnose_contact_loss
scripts.wm02.evaluate_supported_decisions
--spec "$RUN/EXECUTION_SPEC.json" --output "$RUN/collected-01" --run-id collected-01
```

Core原有systemd用户worker实际调用 `scripts.wm02.train_decision_model.main`：

```text
--spec "$RUN/EXECUTION_SPEC.json" --output "$RUN/matched-200-01" --run-id matched-200-01
```

GPU0绑定原实际UUID、4GiB allocator上限、4 CPU、10GiB RAM、2400秒端点、Restart=no和现有neural owner锁。启动返回0；最终MainPID=0、ExecMainStatus=0、NRestarts=0、lease CLOSED。GPU0/1当时都有其他Unix账户任务；没有停止或改变它们。已知旧队列727048/3892803/1425203及原M13/book owner没有重启。

## A0：任务前故障定位与修复边界

[目标保存日志审计](TARGET_STARTUP_AUDIT.json) 重新核对了原高率journal、初始化、控制和安全记录的SHA。

* 两次冷启动初始q/qdot/ctrl相同；映射后12腿初态与冻结默认站姿一致，初速度全0。XML前18位置传感器和18执行器顺序一致，FL/FR/RL/RR策略映射正确。[映射/reset证据](JOINT_MAPPING_RESET_AUDIT.json) 排除了这条已检查的静态错配，不能证明所有运行期输入正确。
* 初始处于RLLocomotion/HOLD，腿kp及目标缓存全0；T1暂停控制调用435次、T2仅15次。T1首次非零腿力矩在step10，T2在step4。同一初态的首个q/qdot/ctrl偏离也在step4。
* T2的第一组腿力矩保持step4–23，共20个2ms tick；step24切换到峰值约61.98的力矩，并发生首次限位失败。原任务pause补丁把RunModel的sim mutex扩大到所有策略，会在N1推理时阻塞PD。此时序与日志吻合，但没有每次推理的真实输入/历史/完成戳，**不能宣布唯一根因已证实或目标已修复**。
* 复用仓库已有R1 pause修复，对实际冻结clock三份源码离线应用成功。完整RunModel计算体、SetCommand控制律不变；实际编译执行原C++并发harness通过：PD不被正常推理阻塞、pause等待推理、嵌套锁正常。[结果](PAUSE_REPAIR_OFFLINE.json) 不包含原生动态资格。
* 夹爪初始目标已是.042m，并非.044机械全开端点。前100步用冻结400/3 PD及力限重建ctrl误差0。steps38–40右滑轨超程时，控制力已向内；不能将其归因为首条全开命令，也不能声称内区间reset已经解决耦合瞬态。
* T2首次停止step24后仍保存25–27三个物理包，即额外6ms。新增原生observer停止锁存，两个外层actuation包装器仅在真实时钟前进时记录转移。实际C++包装器＋故意忽略SIGINT的假内核夹具：旧版6次调用/6包，修复版1次/1包；普通和R1外层均通过。夹具真实积分0；没有替换现役二进制。[证据](STOP_LATCH_FIXTURE.json)

实际冻结parser还完成零积分复核：非单位姿态、完整H40、43.15秒绝对时钟、41点wire、receiver首点与过期/重复/未来/取消检查。编译误差1.11e-16，parser姿态误差2.33e-15。[编译器](COMPILER_PARSER_ACTUAL.json)、[receiver](NATIVE_RECEIVER_ACTUAL.json)。未运行ROS2/OCS2节点，也没有证明实际ACK、参考消费或全身任务通过。

目标lease保持CLOSED，SHA仍为 `db2c30f675acb9e8dc280e3fc69d277e1bf91db10cfc2476881eeac2895be407`。首个确定动态阻断是缺少原owner的新初始化专用有效范围；GPU授权不解除隔离。MPC/WBC/receiver的任意中途完整恢复、站立T1原生评分和Reward五类真实验收仍未完成。

## B0、B1：真实考试与旧模型缺陷

[原冻结B0](FROZEN_B0_ACTUAL.json) 使用原四个DEV预闭爪状态、20个真实候选，H8/H16，0新积分/更新。条件模型H16抓持Brier约.4202，盲模型约.1242；条件模型过早且过度预测抓持，延迟闭合的实际失败没有得到正确概率。真实当前状态teacher-forcing也没有消除事件误差。四个DEV仍是开发数据。

B1以当前接触、TCP/物体相对位置、速度和夹爪观测寻找锚点，不以Teacher阶段或固定秒数作资格。实际首先匹配的时刻恰为1.6/3.6/9.2秒。12/12锚点合格：闭合前出现真实抓住/未抓住分化；搬运物体效果差8.31–8.34mm；释放均有原生桌面支撑。12次完整恢复重复和12次未消费尾部对照误差0。五候选顺序预先哈希，第二400ms块固定重锚并保持。[B1](PHYSICAL_ANCHOR_QUALIFICATION.json)

原V3规则只看最终开度，延迟与立即闭合并列；盲模型候选批量浮点差也产生虚假的排序。V4改为平均命令开度，盲预测单次广播，分数1e-6内按实际raw字节哈希确定并列动作。旧V3结果保留，不能使用“规则0/4”证明WM优越。[V3](ONLINE_V3_ACTUAL.json)、[V4](ONLINE_V4_ACTUAL.json)

## B2、B3与新模型真实闭环

40个全新物理布局与旧24布局不重合，预先固定24 FIT/8 DEV/8一次性留出；仍是原F0局部工作空间、同物体/摩擦/控制器，不声称广泛泛化。120/120新锚点合格，600/600主分叉完整16步；120重复＋120尾部对照均完整状态相同。9,600主转移、1,920重复、1,920尾部、7,360前缀分别计数，无拒绝/提前终止/UNRUN。前缀和控制不混入主训练曝光。[数据](NEW_DATA_ACTUAL.json)、[哈希审计摘要](NEW_DATA_AUDIT.json)

训练replay SHA256：`c12293bc877758254a14d19ec112f147fe19c93ffacf4c8399ae8ebfcb42c808`。两臂同720个FIT窗口、同seed/初始化/normalizer/effective batch256/micro32及固定200端点；两臂各 **18,905,387** 参数。所有normalizer来自FIT；DEV不重采样，留出在两臂端点固定后才评价。

| 新模型 | checkpoint200 SHA256 | 主更新 | 续训探针 | 1/200重载与199→200误差 |
|---|---|---:|---:|---|
| 条件 | `530a6e5b395e7eca76f12efcc6179b6af745b063faff4f6c0bdbc23b18b21f6f` | 200 | 1 | 0 |
| 无未来动作 | `0e60f92f5c133382f9e3244dcfa578094619ff4f43b1c8dc8f562f2e7b0dee0d` | 200 | 1 | 0 |

原FIT损失/梯度诊断使用全部640原窗口计算分阶段均值，每阶段4个预先哈希窗口做梯度探针。旧数据抓持建立为0；稳定阶段的加权事件梯度远小于连续项。释放阶段事件梯度较大且可能与连续项冲突。探针不是全FIT精确梯度，也不能证明只提高事件权重即可解决问题。[诊断](FROZEN_FIT_LOSS_GRADIENTS.json)

必须保留的两项评分修复：V1将事件3误当支撑，但schema事件3实际是illegal_contact；V2的离线并列动作哈希用了神经float32近似值，与原生float64命令不一致，会让盲策略的“选对/选错”翻转。两版均保留、没有重训或重采。V3从原生table/cube接触力恢复支撑，并保留原命令字节身份。修正后的留出属于**同一已看过集合的工程重评分**，不能再称一次新盲测。[V1保留](NEW_WM_V1_RETAINED.json)、[V2保留](MODEL_EVALUATION_V2_RETAINED.json)、[V3有效评分](MODEL_EVALUATION_V3.json)

支持标签独立于四个事件头；释放模型只给“已支撑锚点上的非抓持”条件分数，不声称预测了原生支撑概率。实际释放Q仍要求真实支撑。所有这些Q均为机制诊断，不是正式Actor奖励。

| 修正留出结果 | 条件WM | 无动作WM | 同观察规则 |
|---|---:|---:|---:|
| H8物体末端RMSE | 3.190mm | 2.223mm | 不适用 |
| H16物体末端RMSE | 5.107mm | 2.747mm | 不适用 |
| H16抓持Brier | .1160 | .1306 | 不适用 |
| 预闭爪H16 regret | 0 | 0 | 0 |
| 搬运H16 regret | .07907 | .03467 | 0 |
| 受支撑释放H16 regret | 0 | 0 | 0 |

不同阶段Q单位不同，不用跨阶段平均值冒充统一任务收益。搬运失败可进一步归因到冻结预测：8个DEV中“朝目标/背离目标”的真实距离均值为153.832/161.737mm，而预测为159.8155/159.8125mm；真实7.905mm差异被压到反号3.06微米，抓持概率几乎相等，导致模型选择背离。此处已超出单纯“没有超过强Teacher”的解释，确有动作效应预测不足。[分解](TRANSPORT_FAILURE_DECOMPOSITION.json)

真实闭环在新8DEV父布局中每400ms重观察并重建候选，执行前将完整提案、分数、选择fsync落盘，再通过原gate/S1后积分。32策略episode＋8次条件重复，共640真实转移；8次完整恢复重复误差0，未修改权重，未使用未来真实状态或Teacher阶段。条件/盲/规则抓持均8/8，随机5/8。[闭环](ONLINE_V5_ACTUAL.json)

## 数量、成本、测试及下一步

本轮合计1,004段完整16步分叉/局部rollout＋7,360前缀步，23,424真实转移；585,600 live＋585,600 shadow tick。主分叉及主策略11,584、重复/尾部4,480、前缀7,360。完整抓放、目标成功、Actor/Critic均0。[总表](SUMMARY.json)

原生源域五阶段CPU合计1654.391秒、wall合计429.552秒；GPU0 unit CPU130.301秒、wall115.823秒，主训练程序wall111.489秒。GPU0自有进程采样峰1512MiB，Torch reserved峰1,021,313,024字节，在4GiB上限内。此时驱动报告GPU0总量32,760MiB；本轮新训练不是在GPU1的16GB设备上完成，不能伪报设备验收。GPU1曾因实测余量不足在分配前拒绝，CPU3.066秒、wall3.284秒、0更新。CPU损失诊断、推理、支撑派生与重评分另有各自日志；传输、staging、只读SSH及发布未完整计费，不估计货币成本。[成本/终态](COUNTS_COSTS_AND_TERMINALS.json)

相关world_model **43项**与neural **24项**普通/`-O`通过；实际C++双包装器夹具、原生parser/receiver和原并发harness通过。全仓CI未运行。这些测试不构成目标原生动态资格。

已保存两项无新物理的工程终止：最初B0元数据tuple/list传输不兼容，修复后独立02运行通过；一个V4 worker在Journal INTENT后、必需STARTED目录创建前消失，经独立检查确认无原生效果，保留INTENT并关闭，不重放。同样保留尚未生成spec/lease的两份不完整本地staging目录。评分失败和显存拒绝均不称为算法训练失败。

目标下一步应由原owner绑定独立、有限的init-only窗口：先两次冷初始化，仅检查原限位、输入/历史/模式/时钟和首次控制输出；通过后分级2秒、5秒保持。新停止锁存及已有R1同步修复只进入新候选构建，不热改旧capsule。任一原生失败立即按原契约停止；不直接执行抓放，不修改WBC、腿策略、增益、求解器或限位。该窗口尚未获准，本轮目标0积分。

源域下一项应检验未来参考的尺度、输入利用率与搬运动作效应监督。当前DEV没有证明增加到800或5000更新、或启动Actor梯度优化的收益；本轮停在已完成的固定200端点和真实闭环，保留负结果。
