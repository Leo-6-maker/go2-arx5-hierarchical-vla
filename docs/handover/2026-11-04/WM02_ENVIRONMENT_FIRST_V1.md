# WM02 P1.5 — 失败归因与环境优先恢复（Codex 执行交接）

2026-11-04 · 新一轮设计，不是已经完成的目标动态资格或模型训练。起点：草稿 PR #345 / 固定源码 \`8161d61c322852a2e997abd5e93af7ed369f051f\`，上游 WM02 PR #344、既有 V7 规则均保留。本计划以**先完成可复现环境，再做本地神经训练，最后有资格时完成目标物理**为顺序，不复活旧作业，不重训冻结 WBC。

## 0. 事实与“失败”的正确归类

必须首先阅读 [PR345 原始执行报告](../../../reports/WM02/wm02-20261104-p0-p1/FINAL_REPORT.md)、[脱敏目标身份](../../../reports/WM02/wm02-20261104-p0-p1/RESOLVED_SPEC.redacted.json)、[本地4080S现状](../../../reports/WM02/local/LOCAL_4080S_READINESS.md) 与原始交接协议。

| 现象 | 已有证据 | 结论 |
|---|---|---|
| \`qualify --domain T\` 返回78 | T分支明确拒绝，目标积分0；S1/T2仍隔离、无当前动态准入 | **有意的授权/实现门**，不是MuJoCo/ROS2训练崩溃；不能手动改permission=true绕过 |
| 目标后端 | 现场报告实际 MuJoCo＋ROS2/OCS2 Go2–ARX5；9核心及14初始化成员上轮SHA吻合 | **不能使用误认的Isaac Gym或第三方MuJoCo Warp/OSC作为目标** |
| 目标restore | MPC solver/policy、WBC history、ROS receiver状态缺导出/恢复 | 完整目标分叉与因果梯度评价尚不可进行；只恢复qpos/qvel不足 |
| 目标standing scorer | 原全身评分有BASE相关条件；局部站立T1尚未绑定独立版本 | 不能静默删旧判据或拿旧分母声称T1成功 |
| S域固定底座 | 8接触诊断、16分叉、872个真实50ms转移，restore最大误差0、硬安全0；释放只覆盖0.4s | **S域工程已通过部分资格，不是目标动态结果；完整释放撤离/32s任务未证明** |
| 1/4/16并行 | 源CPU 20.7486/47.9711 atomic/s，4进程2.312x；16因为CPU/RAM不足未运行 | 这是源域CPU真实吞吐，不是目标GPU/WBC吞吐 |
| GPU1 | 用户停止727048，lease CLOSED，旧1024/2048完整端点不变，2154中106步未形成checkpoint；历史空闲15899/16376MiB | 空闲只支持准备模型训练；不是长期GPU reservation，更不是S1/T2许可 |
| WM/Actor | 模型/Actor优化更新0、无模型checkpoint、无训练显存峰值 | **尚未实现/启动，并不存在学习不收敛的结论** |

原727048的最终\`FAILED_BOOK_FINITE_OPTIMIZATION/TimeoutError\`与用户主动STOP一致；不当成WM02失败，不重启旧lease或拿2154作为可恢复端点。上轮私有\`ACTUAL_COMMANDS.private.md\`只在执行者机器，本文不称已读取其原文。

## 1. 先把三套环境分开：E-S、E-T、E-G

**E-S / source CPU**：已有Python3.10.12、NumPy1.24.4、MuJoCo3.2.7固定底座环境，在合法S域用于接口/抓持/奖励诊断。保留原solver/dt/Teacher/S1，不因为WM要求升级、改场景或冒称WBC。

**E-T / target native**：实际Go2–ARX5 **MuJoCo＋ROS2／OCS2**和冻结腿策略、MPC及gripper。上一轮 \`4080-simulation\` Python3.12。需要复核现役运行依赖、ROS distro、C++ ABI/链接依赖、ROS receiver与WBC所有状态；旧已冻结二进制、配置、权重不重编译/升级，不启动服务或动态目标而未取得原有站点授权。

**E-G / GPU neural**：在可授权的本地4080S 16GB用独立项目环境装载PyTorch/CUDA、结构化WM、Actor/Critic；**不要在E-T原生环境上全量\`uv sync\`、\`pip install -U\`、重装ROS2、系统CUDA或修改全局Python**。不同Python环境通过有schema、单位、hash、split/parent的数值数据交换；不要跨环境盲反序列化动态Python对象/未审计pickle。GPU1空闲须重新探测，不据前次读数声称仍有资源。

## 2. E0：只读环境与冻结文件核验（当前第一条实际任务）

从固定PR345 HEAD派生独立管理分支，不在活动source/hot capsule上checkout。优先执行**新增的已实现只读脚本**：

\`\`\`bash
cd "$WM02_SOURCE"
PYTHONPATH=src "$WM02_CPU_PYTHON" -m unittest discover \
  -s tests/world_model -p 'test_env_preflight.py' -v
PYTHONPATH=src "$WM02_CPU_PYTHON" -O -m unittest discover \
  -s tests/world_model -p 'test_env_preflight.py' -v

# 变量由实际owner在本机核对绑定，不公开私有绝对路径。
# 输出文件必须是新路径，不能覆盖旧回执。
PYTHONPATH=src "$WM02_TARGET_PYTHON" -m scripts.wm02.env_preflight \
  --spec "$WM02_PRIVATE/RESOLVED_SPEC.json" \
  --target-root "$WM02_FROZEN_TARGET_ROOT" \
  --output "$WM02_PRIVATE/ENV_PREFLIGHT_NEW.json" --gpu-query
\`\`\`

**该脚本只读取CPU软件版本、ROS环境变量、可选nvidia-smi及9个冻结目标文件实际SHA**；未提供真实目标根时返回NOT_CHECKED，而不是虚假MATCH。它**从不授权动态执行**：\`target_dynamic_authorized=false\`、\`native_target_admission=NOT_PERFORMED\`，不会加载WBC/MuJoCo/ROS2进程。只有已验证的私有当前spec才能用于现场检查；从GitHub取的脱敏spec不能替代活体准入。

同时使用原现场现有工具只读确认14个初始化成员每份**完整路径**，不要以重复basename误配；检查\`python -V\`、\`ROS_DISTRO\`、\`command -v ros2\`、精确rclpy/目标节点依赖、冻结二进制依赖（若有\`file/readelf\`），以及\`nvidia-smi\`、Torch CUDA版本。禁止未经许可修改NFS、SSH、系统服务、ROS网络、系统包、GPU0他人任务、文件权限。

**交付**：公有脱敏\`reports/WM02/environment/ENVIRONMENT_MATRIX.md\`、私有\`ENV_PREFLIGHT.json\`；逐项状态\`OFFLINE_MATCH|MISSING|INCOMPATIBLE|NOT_CHECKED\`，独立报告目标仍\`TARGET_BLOCKED\`与最窄实际缺口。E0没有动态物理、没有模型更新。

## 3. E1：先建立本地GPU神经训练环境（不需目标动态许可）

在与ROS2/OCS2控制环境隔离的项目环境检查Torch/CUDA和GPU1容量；冻结训练env安装记录、Python/Torch/CUDA/driver/依赖版本，确认当前授权owner和GPU进程。**先用synthetic明确标记样本实现并验证训练器**，再读取合法TRAIN真实数值文件。

- 先实现WM02结构化状态增量模型：初版5×2048 SiLU/LayerNorm MLP＋256 GRU历史编码器，状态/接触/终止输出按当前WM02 schema。模型未来只能吃预测、解析更新和固定context，绝不能读取回放未来WBC缓存。
- 用H1/H8/H16自由展开测试、SO(3)合法增量、terminal/truncation与分段400ms Actor节奏；模型FP32小批8/16/32测实际GPU peak，必要时累积保持有效序列batch256。GPU最低余量用真实profile判断，别把17M权重大小当BPTT整图大小。
- 完成**真实200步WM优化**、保存并重载model/optimizer/normalizer/RNG/replay cursor；优化前后loss/数值与恢复一致性。没有足够真实TRAIN数据则只能称\`SYNTHETIC_ENGINEERING_SMOKE\`，不能写\`TARGET_WM_TRAINED\`。
- Actor/Critic做至少一次16原子步想象的真实梯度更新：WM参数冻结而动作输入梯度非零；Actor只在步0与8重规划，16×50ms=0.8s，不可每50ms偷给Actor新的未来观察。保存Actor/critic与完整恢复信息。

**交付**：\`WM02_GPU200_ACTUAL.json\`、\`ACTOR_H16_ACTUAL.json\`、本地VRAM/壁钟/输入身份、checkpoint与恢复结论。若目标仍隔离，可以继续在合格S域真实数据上调通整个神经流程，绝不混入T域Replay或称目标学会抓放。

## 4. E2：目标控制器与评分接口（先CPU分析，动态部分等待许可）

**动作时序**：PR345只证明冻结parser CPU 41个点和11种错误包正确拒绝：H40经receiver成为41节点，保留动作0及2.0s末端保持；**尚未证明reference安装ACK、空间进度或真正消费了哪段**。在只读状态下完成\`ACTION_TIME_MAP\`字段清单，明确真实运行时观察的ACK/安装/消费/achieved时钟及预览影响。

**控制状态**：列出MuJoCo完整积分、WBC observation/history、MPC solver/policy/warmstart、ROS reference receiver、gripper、随机数、任务monitor。每字段记录数据类型/时钟、snapshot/restore API或不可恢复证据、\`PREDICT|ANALYTIC_UPDATE|CONTEXT|RESTORE_ONLY\`，形成\`TARGET_CONTROL_STATE_MAP.md\`。可以实现不触发服务的offline serializer测试。若必须用重新实例化＋确定性prefix replay替代快照，须在未来有资格时以同命令独立跑出可测的状态/接触/终止一致性，不能现在假定成功。

**评分**：在原版原生安全和碰撞逻辑不变基础上，另行定义并冻结\`WM02_STANDING_T1_V1\`的物体抓取→运输→放置→释放→撤离/稳定保持的原生结果；记录与旧BASE位移目标的差别，新任务得分不直接当旧全身任务分数。用已封存只读状态构造成功/失抓/未释放/碰撞的CPU反例，不能为成功降低threshold或改写历史。

**交付**：\`TARGET_CONTROL_STATE_MAP.md\`、\`TARGET_SCORER_BINDING.md\`、是否真实具备分叉恢复的证据；目标动态仍单独标\`WAITING_VALID_ADMISSION\`。

## 5. E3：仅在原站点准入有效时做目标原生物理

**本计划、新分支、GPU1空闲均不解除S1/T2或真机限制。** 只有原有效站点准入、scope/owner、冻结执行和安全检查实际通过才允许目标动态。缺少许可就继续E0–E2及合格S域神经训练，不重复返回78去碰运气。

先只开**单环境**，做目标8个有限条件：hold2、small EEF move2、grasp/lift2、supported release＋retreat2；不能把S域0.4s释放冒充完整撤离。至少两个已恢复状态分别用同一命令独立重放并比较接触、实际TCP/BASE和终止；独立核验安装ACK/消费轨迹与task score。记录50ms转移/s、CPU、GPU真实峰值、ROS/OCS2延迟及失败。进程/topic隔离、RAM/GPU余量不满足时**不强制并行4/16**；不能把源域47.97 atomic/s当目标吞吐。

目标恢复若仍无可审计方案，禁止把目标反事实梯度诊断P3标成有效；可继续收集**经许可的顺序真实转移**用于WM训练，但不伪造精确同状态分叉。

## 6. E4：最小完整科学循环与排队决定

仅在合格域内完成：\`真实TRAIN采样 → WM200步/重载 → Actor H16想象更新 → 新Actor再采样\`。首先用1–4条完整TRAIN episode作管线验收，准确记录完整失败/中止；872条已收集S域诊断（含恢复分叉和重复profile）不是872个独立训练episode，更不是目标真值数据。

取得本地真实模型GPU峰值/吞吐、独立控制器与任务评分、可恢复的checkpoint及实际目标50ms转移/s之后，才提交A40/H200的**单个有界native smoke**，完成实际存储/恢复和站点配额资格后再提交正式百万步训练。正式配方和TRAIN/DEV/TEST划分沿WM02固定计划，改变任何科学定义须单列新版本。站点间只允许已授权批次数据/模型交付，不能每50ms做跨站RPC。

## 7. Codex第一轮完成标准

最先返回：原branch及完整SHA、上述三环境的真实版本/依赖、冻结9＋14文件复核覆盖、\`env_preflight\`命令/退出码/结果、现役目标隔离的准确状态、最窄可行动的工程缺口、是否已创建隔离GPU训练环境。然后给出能真正执行的CPU serializer/scorer检查命令，再推进WM200＋H16梯度。运行没发生的项用\`NOT_RUN\`，不能引用PR345的旧CPU测试作为本轮新增成绩。

**本轮原则**：环境不等于授权；软件构造不等于目标物理；源域S1不等于WBC；模型梯度不等于任务能力。普通工程缺陷按V7自主修复，无需为每个小修再建立审批/调度/预算框架；安全、数据真实性、原STOP和一切完整历史保持。
