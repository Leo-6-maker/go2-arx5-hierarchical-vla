# WM01 — 小型世界模型辅助任务空间策略学习 V1

日期：2026-11-03，Asia/Singapore。实验ID：`WM01`。仓库读取基线：`0a27753bbda8ed75ab6b74aeb64755e0427362c7`（PR341合并）。

**状态：READY_FOR_CODEX_IMPLEMENTATION；尚未实现或运行WM实验。** 用户要求整理代码库，随后由Codex读取并执行新计划。本次发布不代表远端启动。现有执行agent采用交接后，沿用V7的自主工程修复与有效站点权限推进源域实验；真实缺失的站点许可不能伪造。无需每个工作包再请求人工GO。

## 1. 研究目标与范围

核心问题：在相同任务、冻结执行器、相同task-space动作合同下，小型结构化动力学模型是否带来真实的动作决策价值，并提高上层策略的任务成功／学习效率？

主线：`原生源域转移 → 小WM → 反事实/梯度验证 → 小状态策略 → 原生任务评价 → 有条件的视觉/VLA转移`。

首个任务固定为现有源域T1抓放（以当前原生card/score核实含义）。复用固定底座ARX5物理链；明确不是Go2 WBC。第二任务为现有无锁DOOR；BOOK/PUSH、多任务合训、视觉JEPA、GR00T端到端BPTT和目标整机不混入首轮。WBC权重、腿策略、gain、力矩/速度限制、IK/solver和安全判据不改。目标S1/T2隔离和真机限制不变。

本项目借鉴SuperMimic“从真实模拟器转移学习模型，在短想象中优化策略”的方法，不复现其G1运动库、训练规模或速度倍数；小WM不是天然JEPA，也不自动拥有语言能力。

### 可检验假设

H1：动作条件模型在同状态不同动作的原生分叉中，优于持恒/动作打乱基线。
H2：模型建议的局部动作改动，在原生仿真中优于不改动与同范数随机改动。
H3：相同小策略、BC初始化、reward、任务和真实交互终点下，PPO+WM优于PPO，或以更少真实交互达到相当成功率；同时公开全部计算成本。
H4（后继）：WM产生的状态教师能提供真实视觉监督，使VLA/小视觉学生闭环受益。H1/H2/H3成立不等于H4成立。

## 2. 与现役实验的隔离

最新报告显示D2有完整4096端点、PB_CONTROL进行中、M32在Vanda核验输入。接管时刷新实际状态；不重放旧作业，不修改活动capsule或旧时钟，不把shared59/DEV68装入新replay。旧结果保留。代码开发和已有闭合记录的只读审计可先做，原生运行在已有owner与实际空闲资源下按完整episode边界安排。

V7取消旧项目预算作为永久科学阻塞，不取消实际站点配额、有限作业终点、唯一owner、disk/VRAM和安全。WM不新增预算allocator、跨站心跳、审批系统或另一套scheduler。数据清理由用户负责。

## 3. 必须复用的代码与初始审计

- `scripts/pi0291_arx5_data/engine.py`：源ARX5真实有限力矩驱动、`snapshot/step`。
- `scripts/pi0294/multitask/source_engine.py::make_engine`：BOOK/DOOR源任务；原T1/T2从当前pv1源入口绑定，不误用target `night_score`。
- `contracts/trajectory/adapters/safety`：原frame/TCP/time与row-rot6d语义。
- 当前pv1/multitask对应task scorer：W0写出实际完整路径、callable、SHA、任务几何/32秒持续成功定义。未核准前不猜一个同名历史score。
- 官方GR00T adapter及generalization训练链：仅W5使用。

W0产物一份`RESOLVED_SPEC.json`：实际engine/driver/teacher/score/asset身份、字段和单位、原参考首点时间约定、dt、profile、环境与设备；一份`DATA_AUDIT.json`：哪些原始记录有完整转移、哪些缺状态、使用split与累计原生成量。只读一次相关不可变源，不扫描整个Home或重哈希全部历史。

## 4. 状态、动作和转移合同

### 4.1 三种状态分开

WM privileged state可包含qpos/qvel、实际TCP/夹爪及速度、物体位姿/速度、真实接触摘要、arm/finger targets、上次命令、IK/控制记忆、任务计时与必要监测记忆。常量几何/质量等仅来自该TRAIN环境已知参数，单独context；不得从未来帧推断当前输入。

Actor task state只使用当前任务空间量：实际TCP/夹爪及速度、物体位姿/速度、目标关系/任务ID、必要历史与剩余时间。禁止robot ID、关节状态、WBC隐状态、Teacher phase、oracle成功标志进入actor。这里仍是**特权状态教师**，不是视觉学生；几何目标可以用于这一标明的状态实验。

Critic可用WM privileged state＋goal，两个比较臂完全相同。以后视觉学生仅接受原合法双RGB、语言、实际EEF/夹爪与声明的历史，不能沿用教师物体真值。

`Engine.snapshot()`不包含所有驱动器记忆；必须通过薄adapter保存/恢复原native需要的完整积分状态、ctrl/act、warmstart、arm/finger targets、IK缓存、RNG、监测/计时/历史。不能只复制qpos/qvel就宣称同初态。恢复只用于实验reset/分叉，不能在持续任务内teleport纠错。

### 4.2 新actor参数化，不更改旧外部ABI

新actor每0.4秒输出8×7=56个数：8个50ms增量的task-frame平移3、旋转向量3、绝对夹爪开度1。每步平移范数≤0.01m、角增量范数≤0.05rad、夹爪命令[0,1]。用SE(3)合法组合与原row-rot6d编码，生成外部EEF10/H40；前8点是待执行prefix，后32点保持最后目标，仅用于预览。首点时间和anchor从现役contract绑定，不能假设t或t+dt；全臂使用同一compiler。

这是WM01新policy的研究参数化，不是对旧GR00T输出平滑补丁。原生SafetyGate仍可拒绝，拒绝计任务失败，不重采噪声直到通过。原驱动器内部限速/夹爪力不改。源driver不消费预览尾部时可只建模prefix8；将来目标WBC使用预览时必须记录其看过的整段参考与进度，重新验证模型适用性。

BC标签来自真实提交且执行的前8个物理命令，第一点相对当时actual anchor，其后相对前一命令；不得用未来achieved代替命令。超出新参数化范围的标签不裁剪成“专家成功”，单列并保留在WM/失败记录；若没有足够可精确表示的BC样本，两臂统一随机初始化，并在任何结果前固定该选择。绝不按成功率选择BC子集或动作限幅。

### 4.3 转移记录

每条记录至少含`episode_id/scene_id/parent_id/split/executor_id/t0/t1/physics_ticks/state_before/state_after/action_raw/reference_installed/reference_consumed/controller_memory/context/terminal/truncated/safety_reason`，并区分`planned_dt`与`actual_dt`。

记录50ms原子区间；训练model以8个连续实际消费区间组成0.4秒macro transition。跨reset/takeover/丢帧/不同executor不能拼接。H40未消费后缀不可当作已执行监督；上一chunk被新chunk覆盖时必须按真实命令时间线组成标签。

安全停止的部分macro保留，事件/termination头使用真实终止标签；无完整0.4秒状态目标的记录不伪造后续状态，不参与完整状态回归。失败和成功都可用于WM；BC成功数据另标。训练不把安全停止后的未知结果补成零状态。

## 5. 数据与留出

优先复用真正含上述字段的闭合TRAIN记录，不把34563个VLA窗口直接宣称WM转移。缺驱动器记忆等关键字段就给缺口，补小规模原生采集，而非凭推测填零。

初始目标：64条最多32秒的TRAIN episode（最多5120个完整macro转移）。来源先32条原生Teacher、再32条受约束扰动/恢复；所有尝试和失败都计数，不为凑64条成功而无限补采。扰动只作用于合法task-space命令，不改物体状态、WBC或接触力。若旧数据足够，可复用等量且记录原始采样成本。

先按独立scene/episode划分，后切窗；同一初态分叉、恢复后缀和数据增强始终与parent同split。TRAIN64、CALIB16、DEV20、TEST50布局在学习前固定；排除原shared59、DEV68与已被反复查看场景的精确重复，几何接近也应报告。已有长期开发条件只叫DEV，不改名盲测。统计量只来自TRAIN；TEST文件只在最终端点评价阶段读取。新数字布局需native几何检查；非法布局拒绝并留在定义账，不替换更简单seed凑分母。

旧Teacher有成功不等于本轮源adapter已验证；W0固定4个TRAIN canary同一teacher通过原task predicate，所有失败保留。若primitive/接口使可信teacher无法完成，只修明确定义的转换问题，不通过放宽score替代。

## 6. 模型与损失：小而可归因

首版3成员ensemble，每成员3层、宽512、SiLU残差MLP，独立bootstrap与初始化；实际参数数在绑定字段后打印，不预写论文参数量。按状态类型预测连续变化/速度、合法SO(3)更新，以及接触/抓持/terminal logits。基于TRAIN scale归一化，位置/角度/速度分组宏平均，防止q维数掩盖物体；SO(3)用几何距离，不对raw quaternion直接MSE。

优先FP32；AdamW lr=3e-4、weight_decay=1e-5、grad_clip=1、batch256。一次启动先跑200更新smoke，再同配方到5000次模型更新作首个诊断端点。若实际显存不够，学习前缩microbatch并用累积保持effective256；不因结果改变优化目标。单步与K=4多步状态损失系数各1，接触/终止BCE系数0.1；multi-step不能teacher-force未来真实状态来冒充rollout。

事件样本不足时报告未校准，不把未见过碰撞的低预测概率当安全证明。长期模型误差、ensemble分歧与真实失败率分别验证；不确定性仅作拒绝/缩短想象的信号，独立硬安全仍执行。

最少对照：状态持恒、动作打乱、单步/多步误差；逐接近/抓持/运输/释放/恢复阶段报告，这些阶段只用于离线分析，不进入actor。

## 7. 必做的决策与梯度诊断

从CALIB中固定64个实际访问状态，使用完整native快照恢复，分别测试原动作、WM梯度方向、小幅反方向、同范数随机方向、action-blind建议。每分叉预测/真实执行最多4个macro（1.6秒）；相同前缀外的后续动作由同一固定规则提供。计入真实分叉成本。接触/非接触分层；拒绝和提前停止不能从分母删除。

报告：候选进展排序相关性、top-choice实际regret、动作梯度与有限差分的方向一致性、真实奖励改善比例和置信区间；另外报告原生物体结果与安全。

进入actor BPTT的预设pilot条件：梯度建议真实改善比例至少60%，且高于等范数随机；state-conditioned模型排序优于动作打乱；无新增硬安全违规。阈值是本项目决策规则，不是文献保证。若未达到，仍交付负结果；可修一个有证据的数据/实现问题并独立版本复测，不能无界改模型直到“通过”。纯梯度不可靠时不启动BPTT，保留WM为预测诊断用途。

## 8. 策略实验与公平对照

共同Actor：3×256 SiLU、tanh/squashed动作分布，输出上述56参数；共同Critic：3×256。相同数据上最多2000次BC初始化更新，保存同一个初始化文件供两臂使用；不是继承旧GR00T或偷偷给予WM臂更强教师。

A：BC→PPO（仅真实源环境）。B：同BC→相同PPO＋WM短想象梯度更新。BC-only是参考，不替代PPO。两臂相同state白名单、primitive、reward/score、安全、初始布局及随机seed；不能拿状态B与视觉GR00T直接宣称WM算法优越。

共同PPO起点：lr3e-4、gamma0.99（每macro）、GAE0.95、clip0.2、4epoch、batch/rollout256、minibatch64、entropy0.001、grad_clip0.5。有限尾批也记录。先pilot seed1103，两臂各5000个新真实macro；检查预算端点按真实交互而非想象次数。满足W2且无确定工程错误后，三个seed1103/1104/1105各到20000真实macro（同seed的pilot是原运行续训，不从新母本覆盖）；保存1000/5000/10000/20000端点。两个臂6次正式训练，共最多120000新真实macro，不含公共初始数据、诊断、评价。

B的每256真实macro：在该臂TRAIN replay拟合WM100个minibatch，然后最多4次actor想象更新，start batch64、H=4 macro、gamma0.99，actor想象lr3e-5。WM更新与actor更新交替；在actor更新内冻结WM参数，但不能用`no_grad`切断对动作/状态的梯度；冻结critic参数但保留输入梯度。ensemble各成员独立rollout，目标取平均回报减0.5×成员回报标准差。critic从真实rollout TD/GAE目标训练，不能仅用想象回报自证正确。

想象回报`sum gamma^k r(s_hat,a)+gamma^H V(s_hat_H)`；terminal掩码和时间截断分开，真实失败不bootstrap，正常time-limit按声明规则处理。每轮从真实replay state重启，禁止长时间只在自造状态中循环训练。通过终止头提前结束不能获得额外奖励或规避安全惩罚。

H=1/H=4和移除想象梯度是后继消融，不在首轮同时穷举。B多做更新的成本必须报告；若只在等交互下优胜，不称等算力优胜。确认有提升后另做A的等墙钟/计算扩训或matched-update控制，区分额外计算与模型梯度贡献。

### 奖励和原生评分

W0在原任务几何上实现一个公共、可微的训练reward；相同函数分别作用于真实状态和WM预测状态。建议结构：任务目标距离/抓持与抬升/到位释放与静止的有界势函数差`gamma*Phi(s')-Phi(s)`，加动作代价与安全/终止惩罚。具体各项单位、归一化、常数和可观测门控写入`RESOLVED_SPEC`，在任何优化前固定，不输入teacher phase。

这部分尚未实现：Codex须先用真实成功/失抓/未释放/碰撞记录检查reward顺序、有限差分及两臂逐值一致，然后冻结；不能凭本计划虚构reward已经足够。原native持续成功predicate完全独立、逐字节保留，reward高不是任务PASS。零安全违规是实验运行要求，不等于证明策略普遍安全。

## 9. 原生评价与决策规则

DEV20仅在固定学习端点评价，供诊断不用于挑最终checkpoint；最终端点固定20000，未到端点就报告partial，不把最好中间点冒充。终局TEST50×3seed×2臂=300个完整episode，最长32秒，冻结初态与执行次序。formal有效失败不重试选成绩；启动错误可在修复后同定义重启但保留工程次数，区分工程和策略失败。

报告每任务每seed成功率、宏平均、最弱seed、拒绝/硬安全、恢复成功、超时、完整episode完成率、真实环境交互数、积分数、总训练GPU/CPU/墙钟（含初始采集/WM/actor），并分别列评价成本。以scene与training seed为层级作配对bootstrap；3seed只能探索性结论，不把重叠窗口当独立样本。

值得继续的设计门槛：B相对A终局成功率宏平均提高≥10个百分点且无安全退化；或达到预设80% DEV成功率所需真实交互≤A的70%，终局不低于A。置信区间覆盖0时写“有希望但不确定”，不声称显著；未达阈值写INCONCLUSIVE/NO_BENEFIT并定位模型、梯度或任务能力，不自动加步数。正式科学结论必须依据实际原生结果。

## 10. 闭环、硬件与跨站运行

训练期允许非实时源仿真：采样完成一个本地episode/batch，模型训练可暂停物理；明确不是real-time qualification。执行policy只读取当前真实观测与history，输出chunk，原driver执行prefix8，再重新观测；不会用预测状态替代真实反馈。

A40为首选单卡模型/actor训练；H200用于后续规模/视觉对照，不先加载GR00T；4080S可在现有允许角色和空闲slot做小模型推理/本地原型。硬件建议不改变站点权限，不能抢占现役GPU。重型编码/训练/仿真不在AMD登录/存储节点运行。

优先沿已可用的Core物理＋Vanda训练：每个小轮次通过授权固定revision传输状态转移与小actor/WM checkpoint，站点运行无跨站实时RPC/心跳依赖。已有单站profile若明确允许同站源物理＋单GPU训练，可减少传输，但需新run绑定且不抢旧任务。PBS每个作业有有限steps/批次终点和站点walltime；正常保存恢复，不以旧11月6日时间自动授权新任务。

## 11. 后继：何时与VLA结合

W0–W4没有真实收益，不扩到大视觉模型。有效后，先在DOOR重复小范围机制验证，再让状态教师在原生仿真中生成真实RGB＋真实提交动作，最后才做视觉学生。

三者必须分开交付：特权state policy；仅合法视觉/EEF/语言的student；同权重冻结WBC目标迁移。状态教师特有的关节/执行器补偿不能包装成跨本体泛化。部署可不运行WM；在线reranking是可选后继，不是本轮必须实现。

向GR00T转移复用官方数据/processor/action约定，监督只来自真实成功/恢复执行，不从WM虚构未来视频。新的teacher数据、初始权重、样本曝光和对照学生各自固定。目标WBC相关数据使用应声明target adaptation；跨本体结论需另一实际执行器/本体，不把固定底座源域结果当整机成功。

## 12. 工作包与交付

W0：薄native adapter、恢复/分叉一致性、字段/动作合同、数据审计与split。
W1：replay、ensemble、训练/恢复、持恒/action-shuffle基线。
W2：真实分叉、梯度方向与排序报告。
W3：共同BC/PPO与B臂imagination、单seed pilot，再固定三seed。
W4：完整原生评价、配对统计、成本和机制结论。
W5：仅在前述有效后单独制定视觉/多任务/目标迁移配方。

代码拟放`src/go2arx5_vla/world_model/`，薄CLI在`scripts/wm01/`，测试在`tests/world_model/`；原生driver和scorer只import。不假装CLI已经存在。每个阶段只保留最小必要的config/source/data身份、checkpoint（含优化器/RNG/replay游标）、结果/失败/用量、接续命令；避免一项小操作生成十层回执。

## 13. 来源与证据边界

- SuperMimic官方项目： https://supermimic.github.io/ ；论文： https://arxiv.org/abs/2610.28480 。本计划只借鉴交替学习、短想象与梯度方法；未采用未经核对的社区实现作为官方代码，未承诺论文算力收益。
- MBPO： https://arxiv.org/abs/1906.08253 。参考短模型rollout与真实数据起点，本文PPO+BPTT不是MBPO原算法复现。
- 项目基线：[实际回执](../operations/CORE_VANDA_DEPLOYMENT_CPU_RESULTS_20261103.md)、[V7](GR00T_WBC_SCIENCE_GOAL_V7_20261031.md)、[代码地图](../CODE_MAP.md)。

文中模型、超参数、样本量和门槛是本项目预设设计，不是已有实验结果，也不是从论文直接迁来的保证。
