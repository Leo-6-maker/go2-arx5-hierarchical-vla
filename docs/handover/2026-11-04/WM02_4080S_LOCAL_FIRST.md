# WM02补充交接：RTX 4080 SUPER 16GB 本地先行，再提交A40/H200队列

日期：2026-11-04，Asia/Singapore。适用：WM02 V1；作为原交接的**资源/执行顺序补充**，不修改其科学模型、正式训练步数、任务、原生安全或现役运行。本补充文件所在的完整提交由实际Codex接管时从GitHub固定，不用变动branch HEAD冒充immutable身份。

## 最新用户方向及状态

用户明确采用：**先在本地RTX 4080 SUPER 16GB完成环境、代码、数据、梯度和短循环的实际调试；仅在达到排队准入标准后，提交A40/H200的PBS训练任务。** 4080S可以跑小型WM和Actor训练；不能推断4096环境、完整WBC GPU后端或正式百万步训练必然适合16GB。WM02当前仍只有设计文件，新的WM02运行CLI尚未实现，任何测试/性能数值必须来自接管后的实际测量。

此指令的优先级：WM02原“training首选A40”改为“**开发调试首选本地4080S，正式规模训练首选可用且有许可的A40/H200**”。P0-P7科学目标、原始seed/评估分母、action/时钟/奖励/安全不改。核心目标仍是目标Go2+ARX5的**原生冻结WBC**，不能把源ARX5+OSC的成功包装成目标系统通过。

## 一、能在4080S完成的范围

| 阶段 | 本地操作 | 16GB判断 |
|---|---|---|
| L0/WM02-P0 | 只读profile/私有资产/Isaac Gym或当前真正后端/WBC及身份绑定，已有任务/owner/lease保护；PyTorch CUDA、驱动、扩展/依赖检查 | 不需要A40；依赖本地有合法版本和资产 |
| L1/WM02-P1 | action compiler、H40首点与控制区间、row/column rot6d、尾部preview、GRU递推、真实save/restore、记录与split；CPU合成单元测试 | 本地CPU即可，必须先完成 |
| L2/WM02-P1 | **只有目标原生动态许可有效时**单环境real physics/WBC+gripper+安全真实检查；再测n_env=1,4,16，环境不足或OOM及时停在能力边界 | 可能能跑，须实测显存；不能以OSC代替 |
| L3/WM02-P2 | 使用TRAIN真值转移或合成fixtures，对1个5×2048结构化WM进行200-step优化/保存/完整恢复；再在真实TRAIN replay上做5000-step预训练，如本地吞吐允许 | 单独模型大概率可行；全套sim+WM不必同时驻留 |
| L4/WM02-P3 | 8–16个TRAIN状态动作差异/梯度烟测；正式CALIB64分叉在有目标许可、恢复和固定条件后做，不用CALIB调参 | 本地可调；任何分叉需真实目标准入 |
| L5/WM02-P4 | Actor/critic优化BPTT图的单步和16步、梯度冻结、50ms/400ms周期、简短collect→fit→imagine→update→collect端到端pilot | 可小批次；正式每臂1M真实转移×3seed一般排A40/H200 |
| P5/P6 | 少量DEV候选推理、双RGB数据保存、冻结小视觉编码器的forward/小批微调 | 本地适合冒烟；完整600次TEST及视觉正式训练另计划 |

独立WM权重~17.2M参数。FP32 AdamW的参数、梯度和2组moment粗略下界~17.2M×16B=275MB；**不包括**网络激活、16步BPTT计算图、CUDA库/分配器、Physics/WBC/渲染等开销。上游A100 80GB在4096并行环境下报告27–45GB，不能据此直接判16GB不能跑单环境，也不能据此保证本地能跑。

## 二、严格的本地调通顺序（未实现的CLI不得假装可用）

L0：检查是否真有可用的目标原生后端。原公开仓库的`target_sim/environment.py::build_native`使用Isaac Gym+私有WBCEnv；上游`emoPointer/supermimic_manipulation`使用MuJoCo Warp+OSC。检查Linux/CUDA/PyTorch、实际Isaac Gym或被现役profile选定的原生仿真后端、资产和冻结WBC checkpoint/依赖、GPU当前使用者、现役任务/许可。禁止升级或覆盖现役运行环境；优先隔离项目虚拟环境。不能读取private资产时只记录准确缺口，不虚构本地完整Go2 WBC通过。

L1：先在CPU通过组件测试，输入轨迹和时间戳逐项绑定；`--help`/import不启动物理；轨迹变换与参考安装同一个compiler；save/restore要包含模拟积分、controller hidden/history、参考进度/夹爪/RNG。确保想象时第2步起绝不读取真实未来WBC隐状态，400ms Actor周期与真实一致，安全/终态前状态记录准确。

L2：有原生许可才做目标单环境8项诊断：hold2/free-space2/grasp/lift2/supported release2。先1环境且不渲染，必要时4、16、最多64（后两档只在显存/资格允许时），逐档记录GPU真实peak allocated/reserved、`nvidia-smi`峰值与其他负载、CPU/RAM、physics/s、**实际控制50ms转移/s**、contact/safety、冷启动/重置用时。设工程保守预警：16GB卡自身稳定峰值**≤13GB**且有其他任务共享时要按实际可用显存再降低；13GB仅为初始余量建议而非科学门槛。超限时减少num_envs/训练microbatch、拆开采样与学习，不减少WBC安全、修改物理dt或热改旧任务。

L3：先不用原生场景，在合成/历史TRAIN数据上独立训练结构化WM（WM02原5×2048，FP32），先200更新＋恢复。真实数据存在后按P2执行正式5000更新；local建议microbatch 8/16/32逐档测量，并累积到**effective sequence batch256**，改变小批大小要核查loss缩放和batch依赖语义。不要把随机输入可反传视为真实模型能力。checkpoint应含模型、优化器、normalizer、RNG及数据索引。

L4：Actor/critic先做GPU纯神经想象的BPTT smoke：H=1，再H=8和16原子步；`start_batch`先8/16，有必要按相同归一化规则累积到原formal64；检查Actor梯度非零、WM权重完全不变、finite状态、done mask、原子0.05s及Actor0.4s严格一致。接着选8–16个TRAIN真实状态做原生反事实smoke，再按冻结CALIB做正式P3。WM图本身能运行不代表真正有用，必须实测原生反馈。

L5：当采样与学习无法同时装进16GB，可采用**串行分阶段单卡工作流**：在同一合法执行环境中 collect真实短批→保存原件/卸载模拟器→加载WM/Actor fit与imagine→保存checkpoint/卸载训练图→重新加载模拟器并恢复合法训练reset，继续真实采样。避免把`--empty_cache`误认为可以释放仍被引用的张量。不应重启无法独立恢复的活动episode；批次与episode边界停顿。

本地最小端到端验收：一次实际目标物理episode及联系正确的50ms转移→WM200步/重载→一轮Actor想象更新→回到实际物理环境再次采样，结果有完整SHA、现场GPU峰值和真实task/safety结果；模型应能在保持输入相同的情况下通过重载获得一致的预测/采样结果（在可声明的浮点容差内）。若没有目标运行资格则只做到offline/获准S域，不得称“目标native完整调通”。

## 三、何时再申请/提交A40/H200 PBS

必须能提供以下**实际结果**，并写入`reports/WM02/local/LOCAL_4080S_READINESS.md`和机器可读`LOCAL_4080S_PROFILE.json`（私有路径/资产不进Git）：

1. P0：具体backend、WBC checkpoint/config SHA、frame/time/actuation合同、目标资格/当前owner/合法site角色。
2. P1：CPU合同测试含不加速assert禁用（普通与`-O`）、真实或明确未获准的原生保存/恢复、8项接触诊断结果；如部分未运行说明为什么。
3. P2：WM200-update实际训练及保存/读取、TRAIN预测误差、状态/接触覆盖；5000-step是否本地可完成及真实wall/peak。
4. P3/P4：H16想象的真实Actor gradient、WM冻结状态、完整batch占用，短回环至少一次真实交互后有新的可加载Actor；若当前安全门阻断须原样记录。
5. 资源：n_env=1/4/16成功/拒绝/OOM记录，atomic transitions/s，GPU峰值/RAM/CPU/渲染与实际WBC负载。给队列资源估计但不从单点假设线性外推。
6. 集群PBS只提交一个已核实身份的有限`native smoke`，完成checkpoint重载与数据读写之后，才提交对应正式多步作业；不要预提交百万步任务、跳过现场quota/证据门、抢占外部GPU。Core物理和Vanda训练若跨站，按已授权固定revision交换**批次**，不每50ms远程调用。

**不能把本地GPU内存不足等同算法不可行**，也不能把纯WM拟合等同Go2+ARX5执行成功。正式结果保留一贯的native真实评分、交互预算和全成本。若目标S1/T2或真机禁止仍在，说明阻断且继续完成合法的离线与源域部分；本补充不解除安全隔离。

## 四、Codex第一次应交付

1. 最小本地profile脚本和CPU测试，打印精确依赖/模型shape、action-time映射、可用GPU。
2. 真正执行的最小模拟/WM/actor smoke命令与exit code（不存在则先实现，不编造）。
3. `LOCAL_4080S_PROFILE.json`含可重复命令、peak VRAM、吞吐、科学输入/输出、样本数。
4. 最窄阻断及下一条可执行命令；在本地准入前不向A40/H200提交PBS。

此补充只改变执行次序与设备策略，不修改WM02的科学方案或既有高低层控制契约。