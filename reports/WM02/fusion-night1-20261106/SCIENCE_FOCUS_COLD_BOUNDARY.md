# FUSION-NIGHT1：科学优先级接续与已封存实测

PR #367 的补充优先级已在自然冷边界采用，未合并计划分支、降级执行源码或改写活动胶囊。补充提交是 `c7804e338f40d6c5acb2f7272720c9a78dea90a3`；本批新增原生入口与完整历史基线的实际执行源码是 `5103ae60520939d9a9d399b20f783a15b144f670`。原 FUSION-NIGHT1 主 goal 继续 ACTIVE，原 owner、截止、冻结 WBC、2ms 物理步长与安全机制保持。

本批完成的是两次真实冻结 GR00T 推理、一次新参考程序的原生资格检查、两项完整历史闭式模型拟合，以及复用既有数据的误差/语义审计。没有正式 Actor/Critic、弱 ARX 教师继续蒸馏或新 GR00T 训练。完整 SHA、各预测时域、失败与计数在 [实际结果 JSON](SCIENCE_FOCUS_COLD_BOUNDARY_ACTUAL.json)；历史 [192 次 DEV 面板及旧负结果](FROZEN_NATIVE_STRUCTURED_TIMING_AND_CAPTURE.md) 保留。

| 本批项目 | 实际执行 | 退出 / 终态 | 可支持的结论 |
|---|---:|---|---|
| S-CORR2048 源观察推理 | 1 次模型加载、1 次真实 query | 0 / CLOSED / MainPID0 | 完整提案已取得；原源域 Gate ACCEPT |
| 同一模型的目标保存观察推理 | 1 次加载、1 次真实 query | 0 / CLOSED / MainPID0 | 使用实际保存的双 RGB；仍是 shadow |
| 完整 927 维历史接触基线 | 2 个闭式模型、12 个 head 求解、2 次精确重载 | 0 / CLOSED / MainPID0 | 可与既有暴露 DEV 预测比较；0 次优化器更新 |
| +Z 0.5mm 新参考程序 | 4,103 个原生积分步、64 个任务原子动作 | 0 / CLOSED / MainPID0 | 仅此固定程序、单 TRAIN 场景合格 |
| R1.2 既有均衡数据分解 | 6 个 NumPy 参数读回、12 次前向批次 | 0 / 只读 | 0 新采样、拟合、优化或物理积分 |

FUSION 累计原生 **263 次尝试 / 263 PASS / 1,079,089 步**；连同继承的 PILOT2 为 **917 次尝试 / 912 PASS / 5 次历史失败 / 4,066,382 步**。这些是工程执行计数，不能当作完整任务成功率。新 Z 检查的 64 个任务原子动作不等于初始化阶段在内的 164 个完整 25 步区间，也尚未作为新 WM 训练样本消费。

累计 FUSION 已完成真实 VLA query 为 3 次，GR00T / Actor 更新均为 0。WM 主优化器调用仍为 12,200：12,000 次已完成模型更新、200 次历史失败后未保存的更新；6 次恢复探针单列。本批没有重复这些训练。新的闭式求解与 NumPy 前向不计入优化器更新。

**R0：参考意图已经澄清，目标执行条件仍需补齐。**

冻结 S-CORR2048 模型 manifest 为 `94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38`。新版本标签审计绑定原 TRAIN126、统计与 C1 修复数据，核对初始训练 `03646938ece5c6fa5f09b0fd00732f8159a51468`、最终训练 `e418bb790d291c6c936ce3878366314f4f5af83e`、冻结源确认 `887e65629291f16f73863e2b111123007542d268` 的生成/导出/读取链。11 项核心文件逐字节一致。

原 Teacher 按阶段几何目标，从实际 TCP 生成限速平移/SO3 参考；原 episode 提交同一参考，导出器只取已提交并获得 ACK 的命令。C1 也进入相同原 Teacher 链，没有发现 TCP 逆跟踪或偏置补偿。因此新入口可以使用 `INTENT_POSE / INTENT_POSE_REFERENCE`：希望跟踪的任务空间参考位姿。它不表示实际 TCP 等于命令，不是 `SOURCE_OUTCOME`，也不证明模型的源技能或本体迁移成功。H40 训练窗跨每 8 点实际反馈重规划，不能称一次不可变的开环 Teacher 计划。新审计 SHA256 为 `2424f8379259cb5c13763792b7ba877525a5f543dcf9bc27fd609466902948c7`；旧 `SEMANTICS_UNRESOLVED` 收据未修改。

| 真实新提案 | 源观察 | 目标保存观察，映回目标 world |
|---|---:|---:|
| 最大离观察锚点距离 | 82.131mm | 17.179mm |
| 完整 H40 waypoint 路径长度，不含锚点至首点 | 127.060mm | 281.624mm |
| 首点跳变 | 0.943mm | 9.004mm |
| 最大 waypoint 速度 | 0.1153m/s | 0.3430m/s |
| 最大离锚点转角 | 0.01568rad | 0.04250rad |
| 原始机械夹爪指令越目标 working stroke | 40/40 点 | 40/40 点 |

两条提案的原源域 Gate 都 ACCEPT；该 Gate 的通过不等于目标碰撞、完整 preview 或 WM 支持资格。现有 XY1mm 模型不覆盖这些完整轨迹。目标 shadow 仍使用借用的源 T1 语言，对应旧采集任务不匹配；相机、本体与在线时延资格也未由此次推理建立。审计仅读取已由原 Processor 解码的 `physical_H40`，没有再次反归一化、相对累加或新增裁剪；原 inverse 内部已有的 36 / 41 个越单位范围分量保留。旧 CORR 的 8 个源后果没有附给新 query。

两次真实 query 分别耗时 1.166917s / 1.384693s；冷加载、校验等使整个阶段为 251.512s / 173.795s。两次都实际运行在报告 32,760MiB 的 GPU0，峰值 allocated 13,320,805,376B、reserved 13,524,533,248B，不能据此声称已在 16GiB 卡完成资格。N=1/条件不产生延迟 P50/P95，wall 推理时间大于 0.4s 也不能直接当成已暂停的仿真 TTL 过期。

**R1 / R2：有动作响应信号，尚无绝对轨迹或真实控制增益。**

复用原 48 个均衡窗口：32 TRAIN / 16 DEV，按完整物理父分为 4 / 2，每父每动作两次独立重启。不是隐状态完整恢复分叉。没有再采样，也没有重新训练六个响应模型。数据 SHA256 为 `e5139d792d8d2f28e49ea28d7f57fbabbd5e79378e5358b747b2896c851ed795`。

| pooled H16 的 XYZ 坐标 RMSE，mm | 训练均值 | AC seed366301 | AC seed366302 | AC seed366303 |
|---|---:|---:|---:|---:|
| TRAIN 绝对轨迹 | 0.081103 | 0.005523 | 0.007022 | 0.005820 |
| DEV 绝对轨迹 | 0.077067 | 0.230533 | 0.266060 | 0.261671 |
| DEV 实际 HOLD 轨迹 | 0.073908 | 0.122995 | 0.169280 | 0.167722 |
| DEV 动作减 HOLD 的独立重启组均值增量 | 0.088312 | 0.084842 | 0.083791 | 0.083818 |

模型已经能拟合 TRAIN；不能把问题解释为单纯无法拟合。DEV 绝对轨迹、HOLD 和 HOLD 向动作前缀的 transport 误差明显变差；但动作增量 pooled XYZ/XY 略好于均值。H8/H16 的增量只在 XY 更好、XYZ 更差，因此不能宣称均值在每个组件或每个维度都赢，也不能把有限增量预测信号升级为控制收益。组均值绝对 bias = HOLD transport bias + 增量误差在代数上闭合，RMS 项不能相加；独立重启的前缀散布仍保留。

原 192 次 DEV 面板不重跑：WM 相对强静态基线的 objective gain 为 **−0.06658793009705134mm**，95% 区间 **[−0.138153263009148, −0.0011761912374917724]mm**，正向父为 **1/6**，原 TRAIN 实用门槛为 **0.08124448952310876mm**。本次事后诊断不改写该负结果。

**接触支线：完整历史线性基线已完成，没有新增接触物理面板。**

FULL927 使用原 927 维历史及每个 head 时域以内的动作程序前缀；28 TRAIN episode / 8 父 / 4,452 anchor，TRAIN-only 归一化、固定 ridge=1，先冻结重载再读取已暴露 DEV。模型 SHA256：AC `6809038e56faff8c12e451cfa6bd30dc9c83d5c99319e2bc38c980ccc3ee2ce5`；blind `d1ffdfa4c951be857bfa153acbe689949daac72be5dafc2f4ca4277de9a32be9`。冻结描述 SHA256 为 `bff4502f6a42656ad25b5d2185843938afa259cc589fb8953408f86a956be064`。

C10 DEV H64 grasp Brier：FULL927 AC **0.03996824**，blind **0.24344293**；原匹配神经 AC 三 seed 为 **0.00962187 / 0.02661809 / 0.00499281**。完整历史基线改善旧 FIR90 的 C10 H1–H32，H64 略差；不能沿用“每 seed、每时域均优”的泛化说法。C09 H64 则由 FULL927 胜过这三个神经 seed。C10 grid 漏报由旧 FIR90 的 3 降至 0，误报仍 28，检测成功条件下时序 MAE 由 0.172233s 降至 0.065619s。该时序是六个预测时域网格上的首次变化，不是原生首次接触时刻；C09/C10 有两条 episode 重叠，DEV anchor 不是独立试验。没有独立 TEST、原生控制收益或状态依赖策略选择结果。

**新 Z 资格：真实物理响应，不是 GR00T 短技能。**

单 TRAIN 场景固定 INIT2503 reference，decision3703，endpoint4103；参考锚点与期望实际目标锚点分别保留，没有静默重锚。原生 binary SHA256 为 `f30450e669236724a63c125cc88df86bb28aad9136ee644b1c8328e088b0adea`。64 个任务原子动作都有原 receiver ACK 与 solver target 消费证据，控制器最终选中 policy ID 仍仅为 derived；trace 无丢弃，原安全告警 0，四个 owned child 均正常退出。STOP 本轮没有触发，不能写成新的 STOP 资格通过。

期望 +Z0.5mm，实测相对 decision 的位移为 **(+0.441184, −0.287433, +0.347196)mm**；相对期望的三维终点误差为 **0.548279mm**。这不是完整 Z 域、旋转、夹爪变化、GR00T 执行或 WM 收益资格。实际结果 SHA256 为 `fd163e866a1c82873c3a5e044e60c3295b7696655b4af28580321a7cfad9b9c0`。

| 成本口径 | wall s | CPU s | 其他 |
|---|---:|---:|---|
| CORR 源 worker | 253.139483 | child 22.651371 | query GPU event 1.166753s |
| CORR 目标保存观察 worker | 175.696380 | child 23.071330 | query GPU event 1.380500s |
| FULL927 拟合函数 | 57.964454 | 53.523822 | wrapper wall 61.590975s，RSS 1,140,912KiB |
| Z 原生 worker | 57.044681 | own 13.794880 / child 73.472050 | unit CPU 87.835645s，峰值 1,171,075,072B |
| R1.2 只读分解函数 | 0.764216 | 0.346356 | wrapper wall 0.848256s |

函数、worker 和 unit 口径有重叠，不能相加；GPU query event 不含冷加载，也不是 GPU 账单。共享 GPU 的独占利用积分和货币账单未知。

本批真实 worker 命令形态如下；其对应 run 已 CLOSED，**不得重放**。完整实际 argv、私有路径和 wrapper 退出码已保留在原 worker 收据。

```bash
"$WM02_GR00T_PYTHON" -B -m scripts.wm02.infer_fusion_vla --spec "$SOURCE_SHADOW/EXECUTION_SPEC.json" --output "$SOURCE_SHADOW/fit-01"
"$WM02_GR00T_PYTHON" -B -m scripts.wm02.infer_fusion_vla --spec "$TARGET_SHADOW/EXECUTION_SPEC.json" --output "$TARGET_SHADOW/fit-01"
"$WM02_NATIVE_PYTHON" -B -m scripts.wm02.fit_contact_timing_full_history --spec "$FULL927/EXECUTION_SPEC.json" --output "$FULL927/fit-01"
"$WM02_NATIVE_PYTHON" -B -m scripts.wm02.pilot1_native_worker --spec "$Z_PLUS_0P5/EXECUTION_SPEC.json"
```

下一研究问题是完整、任务相关轨迹的支持域，不继续扩大旧四动作弱教师蒸馏。正在实现共享的有版本轨迹适配：保留全部 40 个解码位姿与原始夹爪，明确坐标映射、工作行程映射、重定时、初始 reference transport 与每项改变量；先做单环境分级资格，完整尺度实际通过后才讨论整个提案族。分级几何检查标 `NON_VLA_CALIBRATION`，不冒充实时 GR00T 或短技能对照。只有后续相同冻结 GR00T、相同轨迹适配、相同信息和评分下的真实有/无 WM 对照，才能支持融合收益；蒸馏轻量 adapter 仍等待可复现物理收益。H200、GPU 物理多环境和分布式扩展保持审核前停止。
