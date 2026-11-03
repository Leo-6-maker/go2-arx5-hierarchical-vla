# WM02 — Go2＋ARX5 冻结 WBC 上的世界模型辅助 Actor 学习

版本：V1；日期：2026-11-04，Asia/Singapore。状态：**READY_FOR_IMPLEMENTATION；训练器与目标运行尚未实现/执行**。

本仓库审阅基线：`0a27753bbda8ed75ab6b74aeb64755e0427362c7`。参考实现：`emoPointer/supermimic_manipulation@eda7da480c6d92a68cc41c9628b7efdae08cae23`。本文数值是新实验预设，不是文献保证或实测成绩。

## 0. 决策与不变边界

用户已选择研究“Actor→冻结 WBC→实际物理反馈→WM→想象优化 Actor”的直接路线。WM02 将它作为新科研主线；**不必加载 GR00T，不必先学视频生成，不重训 WBC**。WM 与 Actor 交替更新，不是等待离线数据收齐后只训练一次 Actor。

目标成果分开：T（目标执行器上的状态 Actor）、V（仅合法视觉/传感器输入的学生）、M（同权重多任务）、X（固定上层权重跨执行器）。T 不自动证明 V/M/X。目标环境数据训练属于 target adaptation；可以主张无需新增目标本体人类遥操作，但不能声称没有目标本体交互数据。

本次请求授权制定/发布计划，**不等于已经发起远端计算，也不自动解除目标 S1/T2 或真机限制**。Codex 接管后核实实际授权、现役 owner/lease、可用设备及目标资格；缺少目标许可时推进代码、离线与已获准源域工作，报告一个准确阻断，不绕过隔离。V7 的普通工程自主修复、真实站点权限、数据真实性、唯一执行 owner 和固定安全仍适用。不得抢停健康的 PB/M32/consumer 链、重开 CLOSED 运行、热改活动 capsule、删除数据/检查点或重建调度平台。

新方案不要求完成旧 GR00T 比较才开始；旧正式面板/数据/权重/失败完整保留。WM01 PR #342 在审阅时仍未合并。WM02 新建独立 feature branch，不依赖该 PR，也不覆写其协议。

## 1. 科学问题与主要对照

H1：模型能预测由真实冻结 WBC 执行的动作差异，而非仅预测常见轨迹。
H2：沿模型提供的动作改进方向，在实际物理仿真中有收益，尤其在接触建立/释放附近。
H3：相同目标环境、Actor 观察、动作编译、奖励、reset 分布和原生评价下，WM 辅助 Actor 相比 PPO 提升任务成功或真实交互效率。
H4：视觉学生能在不读取物体真值的条件下保留状态 Actor 的主要能力。

正式首任务仅 T1 抓取—搬运—放置—释放—撤离，以实际现行任务 card 与 native scorer 为准。先做全 Go2＋ARX5 自由机身、局部站立操作；不焊死底座或移除腿部碰撞后仍称完整本体。此阶段不要求行走，不称移动操作成功。DOOR、BOOK、共享多任务和主动移动在首任务结论后逐项增加。

主要两臂：
- A：任务空间 Actor＋PPO，在真实物理环境学习。
- B：相同均值网络的 Actor＋WM 短想象 BPTT，Actor 不接受 PPO 更新。Critic 使用真实数据校准并补充想象目标；这是 SuperMimic-style 适配，不是原论文逐参数复现。

PPO＋WM 混合是后继消融，不默认成为第三条正式臂。完整 GR00T 可作系统参考，但不能拿视觉 GR00T 与特权状态 B 的差异当作 WM 算法收益。

## 2. 三个域与实际后端选择

| 域 | 用途 | 不允许的混称 |
|---|---|---|
| U：第三方固定 ARX5＋OSC | 可选安装/方法复核 | 不是本项目 WBC，也不是目标能力 |
| S：本仓库固定 ARX5 源执行器 | 接口/奖励/数据诊断，可并行开发 | 不是 Go2 全身结果 |
| T：实际 Go2＋ARX5＋冻结 WBC | 主要训练和完整系统结论 | 不允许暗换 OSC、腿策略或控制增益 |

**先确定后端，而非强制迁入 mjlab。** 基线 `target_sim/environment.py::build_native` 明确导入 Isaac Gym 与私有 WBCEnv；`target_mujoco/` 还保留另一套原生接口。文件存在不证明当前运行使用哪一套。P0 从现役 profile/配置确认唯一 runtime、WBC、仿真后端、驱动、gripper、资产和 scorer 的完整身份。保留当前已资格后端；不能为了复用上游 MuJoCo Warp 训练器替换冻结 WBC 或宣称两个后端等价。

上游的固定底座状态、OSC、reset bank 和 checkpoint 不直接当 T 域初始模型或数据。跨后端迁移需要独立数值/接触验证，不能作为本轮隐藏依赖。第一轮不同时改仿真后端与学习算法。

## 3. 系统接口与两级时钟

物理执行：`Actor(o,g) → command compiler → installed EEF reference → frozen WBC + gripper → native simulator`。
学习：`真实 TRAIN replay → WM fitting → 短 imagined rollout → Actor/Critic update → 新 Actor 真实采样`。

采用两级时钟：原生积分/WBC频率保持现场绑定值；记录与 WM 原子步为 0.05s；Actor 决策间隔为 0.4s（8个原子步）。本轮默认非实时仿真学习，暂停物理等待训练可行，但不得报告为实时部署资格。

### 3.1 动作与完整参考

Actor 输出8×7=56个数：每个区间的平移增量3、SO(3)旋转向量3、绝对夹爪开度1。平移在声明的固定 task frame 中、范数≤0.01m；旋转增量范数≤0.05rad；夹爪范围[0,1]，开闭方向/机械宽度由原生标定确定，不能复制上游二值符号约定。

命令由同一个可测试的 compiler 转为现有 EEF10/H40。使用有界连续映射；旋转先构造合法旋转矩阵/四元数，再按既有 boundary 转码，不对 rot6d 六个数随意加噪声。初版尾部32点保持最后指令位姿，**这是明确的规划选择，不声称尾部无影响**；两个臂使用同一尾部规则。

P1必须输出 `ACTION_TIME_MAP`：56维参数、40个参考节点、8个控制区间、实际 first-anchor/binding time 的逐项对应。原生首点可能替换为 measured start，节点数量不等于区间数量。必须用已知小轨迹验证消费时间；不能擅自假设首点在 t 或 t+0.05。若当前 ABI 无法无歧义地表达这一方案，在任何训练前修订 compiler 版本并固定，不用静默补点掩盖。

WBC可能依据空间进度和完整未来预览控制，不等于按时间机械播放前8行。每次决策仅推进0.4s后重规划，记录实际上安装/看过/消费的参考、进度和时间。H40剩余部分可作为条件输入，但不能伪造其未发生的未来物理标签。

### 3.2 训练、想象、部署使用同一个决策节奏

WM每0.05s预测一次，但 Actor只在0.4s边界更新。16步想象覆盖0.8s，最多两次新 Actor 决策；其间保持该块已提交参考。禁止在每个预测小步重新查询Actor、却在真实环境执行整块，否则会给模型内策略额外反馈能力。

Gamma按物理时间定义：gamma_atomic=0.997。一个含n个真实原子步的决策，折扣为 `0.997**n`，块内奖励也按该折扣聚合；不把原子步和块步同时写成0.997。提前终止使用实际n，不能用补零步骤凑完整块。

### 3.3 拒绝与停止

保留原native SafetyGate、接触/关节/速度/稳定性门与已有保持/停止流程。拒绝记录为拒绝，不反复采噪声直到通过。若拒绝后未推进物理，作为决策/准入事件存储，不伪造0.05s状态转移。实际安全终止保存terminal前状态与已发生转移；不记录自动reset后的观察为其后继。不得把松开正在抓持的物体作为默认超时行为，应沿原合格停止契约。

## 4. 状态设计：可递推模型与可部署策略分离

WM输入包括机器人基座姿态/速度、臂腿关节与速度、TCP/夹爪、物体位姿与速度、必要接触摘要、已安装参考及执行进度。场景几何/质量/摩擦等已知常量作为context；它们仅在仿真训练中可用，不自动提供给部署Actor。

Actor白名单：实际TCP/夹爪及速度、物体/目标与TCP的任务空间关系、必要任务参数、上次动作和实际跟踪残差、最近4个决策时刻历史、剩余时间。首版物体状态来自仿真，明确是state policy。禁止关节角、腿部状态、WBC隐状态、reset类别、Teacher phase、未来状态或oracle成功标志进入Actor。所有几何坐标与速度单位在P0固定。限制输入不等于已经证明本体无关。

Critic可以使用特权状态、目标和时间，但两个臂的白名单一致。后续视觉学生只读取实际双RGB、合法TCP/夹爪反馈及目标/语言，不沿用仿真物体真值。

### 4.1 完整保存状态不等于必须预测每个缓存

为了分叉恢复，需要保存原生积分状态、控制目标/历史、WBC观察栈或RNN状态、参考/进度、随机数、gripper/监测记忆、场景动态状态；不能仅复制qpos/qvel。

WM是可微的约化模型，不要求复原所有solver warmstart。每个模型字段必须归入：`PREDICT`连续/事件量、`ANALYTIC_UPDATE`时钟/参考队列/历史移位、`CONTEXT`常量或`RESTORE_ONLY`缓存。

**任何在下一步再次作为WM输入的量，都必须能在想象里递推。** 不可在第一步用真实WBC隐藏状态，第二步继续偷用回放中的未来真实隐藏状态。默认使用256维GRU处理过去8个原子步的状态/命令历史，随后由预测状态与命令更新其belief；显式执行进度如不能解析更新，作为预测输出。未来WBC缓存不是外挂oracle。

若能证明完整可观测状态已足够，可在预结果修订中取消GRU；不要把这一变动混入主实验。对递推模型，历史burn-in来自真实TRAIN前缀；burn-in以后必须自由展开预测。

## 5. P0–P1：最先完成的工程与物理证据

P0只读当前owner、相关#11/#47、身份和已有许可；一次完成 `RESOLVED_SPEC.json`，不扫描全部Home或重读全部历史模型。填写实际backend、engine、frozen WBC、gripper、资产、task card、scorer、频率、动作/状态schema、原生安全和合法device/profile。

P1复用而不重写WBC/评分器，实现薄环境adapter、compiler、完整save/restore、50ms recorder与状态投影。CPU测试通过后，在有效目标资格下执行8个固定TRAIN诊断：保持2、自由空间小轨迹2、抓持/小幅抬升2、支撑后释放/撤离2。部分任务初态可来自经物理验证的reset，不冒充完整任务成功。必须能取得必要接触链的有效反馈、无真实硬安全违规；完整任务Teacher成功不是训练Actor的强制前提。

同状态/同指令恢复两次，比较轨迹、接触、参考消费与终止。不能bitwise时在训练前固定合理误差容限并报告实际误差，不能根据策略成绩调整容限。若后端无法可靠restore，采用固定reset与相同物理前缀重建，并验证重复性；否则不能作“严格同状态反事实”声明。

吞吐profiling按实际可用并行数1→4→16逐级测试，后续最多64只在权限/容量允许且P1通过时考虑。测全周期 transitions/s、原生WBC耗时、峰值VRAM/RAM、记录/传输/恢复成本。不得为提高吞吐改物理dt、decimation、WBC增益或移除接触。上游4096并行不是目标默认值。

目标隔离未解决时，P0、CPU实现及S/U域有限诊断可以继续；报告 `TARGET_BLOCKED`，不能把S/U域PASS改名T域。

## 6. 数据、Reset Curriculum与样本定义

一条原子记录至少包含：run/episode/parent/layout/split/executor/actor_version、t0/t1/实际ticks、原始Actor动作、编译参考、installed引用、实际消费/进度、pre/post物理状态、控制记忆引用、reward/native success、terminated/truncated、有效mask/安全原因。颜色图像在状态阶段非必需；关键诊断可记录，视觉阶段才批量渲染。

训练数据包括成功、失败、抓持滑脱和未完成片段；数值发散后的无效状态剔除回归但保留次数。硬安全停止不授权继续同一危险尝试。teacher执行、student执行、未执行专家查询三类不能混标。

### 6.1 划分

预先固定 TRAIN512、CALIB32、DEV32、TEST100个独立场景/初态layout。按scene与parent episode划分，再切窗；同一父轨迹的分叉/恢复/增强始终同split。不可把CALIB/DEV/TEST放入replay、归一化、Actor想象或教师蒸馏训练。CALIB是开发校准，不称盲测；TEST仅正式固定端点评价读取。

排除现有shared59/DEV68等受保护实验数据；旧长期看过场景不能改名TEST。记录几何距离和资产复用范围，精确不重合不代表分布外。生成规则先固定，非法布局保留拒绝及原因；不得结果出来后换简单seed。正式100个合法TEST应在学习前从固定候选列表按既定几何规则确定；未凑齐即实际少于100，不能声称完成预设600次。

### 6.2 Reset Bank

四类：R0正常home/未抓取；R1物体附近张开夹爪；R2实际抓持/已抬起；R3实际抓持且接近放置区。采用前64个TRAIN布局，每类64个候选，共256；每候选最多2次构造尝试，全部成本记录。R2/R3来自实际物理抓持路径及完整控制状态，不能穿模摆位、weld物体或伪造夹持。每个保存状态用原控制器保持1s，并从序列化状态重建再保持1s验证；两次保持都计物理采样成本。

初期以四类均匀训练，正式交互进度到50%后，R0占50%，其他三类均分其余50%。这是固定时间表，不按测试成绩调权。正式主评价全部从R0开始；阶段初态成功率另表，不能平均后声称完整任务成功。reset之前已完成的阶段仅是继承历史，不重新发成功奖励。

### 6.3 初始数据与探索

先64条最多32s尝试，作为管线/事件覆盖诊断，不作为“足够学习”的保证；然后同一数据链累计到256条（不是再加256）。每类64次，初始探索混合50%可用程序教师/已验证局部参考、50%有界扰动。不存在可信教师时使用物理有效reset加有界动作，并如实标注，不伪造示范。

完整256条的上限是163840个50ms转移/20480个0.4s块；提前停止实际会更少。所有尝试计数，不只统计成功。WM使用失败，主Actor不做BC：两臂共享同一随机均值网络初始化，避免“教师给某一臂额外优势”。受约束探索在归一化动作空间加std0.1噪声再经共同compiler，原生限制不变；夹爪事件覆盖可单列有界开闭探索，不能对持重对象无条件随机松爪。

## 7. WM：结构、训练与必须通过的诊断

初版采用上游可复用MLP形式：5×2048、LayerNorm、SiLU、归一化状态增量；加第4节的history encoder与任务特定输出头。不是复用OSC权重，不承诺参数恰好17.2M，字段绑定后打印真实总数/可训练数。主要单模型；ensemble仅作为后继消融，不为了不确定性先训练三套大模型。

预测连续机器人/物体状态、合法SO(3)增量、必要执行进度，以及接触/抓持/失败事件。确定性时钟/参考缓存解析更新。用TRAIN scale分组归一化，姿态采用SO(3)误差而非raw quaternion MSE；机器人和物体分别报告，防止腿部维数掩盖物体误差。

预训练：AdamW lr2e-4、wd1e-5、grad clip1；有效batch256个序列；burn-in8，free-rollout8原子步。先200个optimizer updates工程smoke并恢复，再从正式初始化独立到5000更新；smoke不混作主母本。状态/多步组均权，事件BCE权0.1。默认FP32完成smoke；随后BF16-MLP/FP32状态积分仅在数值/梯度回归通过后、正式训练开始前固定。微批次可因容量调整，保持有效batch；不根据DEV优化结果改精度或batch。

多步训练只使用真实动作序列，不输入未来真实状态/物体真值/控制隐藏状态。训练跨Actor边界时使用记录的真实installed序列；想象跨边界则由当前Actor生成新参考。

P3诊断：64个CALIB真实访问状态，覆盖接近/接触/搬运/释放（不足如实报告）。每状态比较原动作、模型梯度改进、反向改进、同范数随机改进、action-blind建议，共最多320个物理分叉，每分叉最多16原子步/0.8s；最大5120原子步。恢复成本另计。

报告单步/8步/16步分层误差、事件校准、action-shuffle与持恒基线、真实候选排序/最优动作遗憾、梯度方向与有限差分一致性。BPTT启用门：可度量的梯度改进在全部预定状态中至少60%为正、优于同范数随机，排序优于action-blind，且无新增真实硬安全违规。平局/拒绝/未运行不能从分母删除；同时给按parent分组的区间。不通过则交负结果并改一个有证据的模型/数据问题，另立版本；PPO基线与独立开发不因此永久停止。

## 8. 任务奖励：原生成功与可微训练目标分开

原native scorer、持续时间和安全阈值不变。训练用统一 `reward_train(state, action, next_state, goal, process_state)`，同一实现用于A的真实状态与B的预测状态；**不加入只给B的隐藏想象奖励**。参考项目存在任务定制的想象shaping，不能直接搬来再称奖励匹配。

首任务默认形式：`r = 10*success_surrogate_onset + gamma*Phi(s_next)-Phi(s) - 0.01*action_cost - 0.02*action_change_cost - 5*unsafe_onset`。Phi由接近、稳定抓持、抬起、物体到目标、到位释放/支撑五项构成，权重[1,2,2,4,2]。位置核尺度0.10m，抬起尺度0.05m；目标区域、夹爪开度、物体几何/稳定速度从实际task card绑定。

抓持/释放的几何或连续代理必须显式实现；它们不是物理接触真值。P1先用真实成功、空手搬运、过早释放、碰撞、抓住不放五类记录核验排序/梯度与native predicate差异，冻结全部公式/单位/门控/持稳计数到RESOLVED_SPEC后才学习。若默认公式错误，必须在结果前版本化修订，不称本文已证明奖励充分。success-onset只奖励一次；reset继承历史不获新奖励；正式任务评分独立报告。

连续代理若需straight-through导数，必须注明是surrogate gradient，而非真实接触物理导数。训练前在真实分叉核验其方向。terminate掩码在Actor梯度内detach；禁止Actor通过预测“提前成功/提前停止”消掉未来惩罚。安全屏障仍由真实执行器负责，模型低风险概率不是放行依据。

## 9. 交替训练与Critic

每个周期先用固定actor_version收集最多2048个新的真实50ms转移，再训练；完整块通常对应256次Actor决策，提前结束按实际记录。PPO只使用当前收集策略的真实on-policy样本；不得拿旧教师或混合版本replay做PPO更新。B的WM可使用本臂全部TRAIN replay，Actor想象从真实TRAIN片段起点开始。

A：PPO lr3e-4、clip0.2、GAE lambda0.95（决策尺度）、4 epochs、minibatch64决策、entropy0.001、grad clip0.5；Gaussian分布在共同动作变换前计算log_prob。B使用同一均值网络而外加相同物理有界探索。

B：每收集周期WM拟合50个batch256/window8；Actor想象更新4次、start batch64、H=16原子步、Actor lr3e-5、grad clip0.5。每8步调用一次Actor，用预测观测重建history；不调用真实仿真器或读取未来replay状态。

Actor/Critic均为3×256 SiLU。Critic预测决策边界上的未来回报，输入白名单两臂相同。真实块奖励按0.997原子折扣聚合；真实TD(lambda)目标使用target critic，EMA tau0.01。B每周期4次真实critic更新，再4次想象TD(lambda)更新（想象loss权0.5）；A value更新沿共同PPO。目标全部detach。真失败/真成功及任务本身32s终点不bootstrap；人为collector切片而任务继续则bootstrap，不能把它当任务失败。

B优化 `J = discounted imagined rewards + terminal target value - 0.01*mean_square(action_mean - last_collect_actor_mean)`，该正则仅约束本周期内更新幅度，原生安全仍独立。WM/Critic参数在Actor更新期间冻结，但保持对输入/动作的梯度；`torch.no_grad()`不能包住想象传播。未来history、坐标变换与动作compiler必须保持可微路径，并与原生实际命令数值对照；不要经NumPy转换切断梯度。

### 9.1 有限科学终点，不用无限重训替代结论

seed固定20261104/20261105/20261106。单seed工程pilot：每臂100000个新真实原子转移。通过数值/接触和执行验证后，该seed原样续到1000000，并运行其他两seed；不因为seed成绩差而换seed。正式检查点：100000/250000/500000/1000000。总计两臂三seed最多6000000个新真实原子转移，初始公共采集、reset构造、校准和评价另计。

正式端点固定1000000，不在TEST挑最佳检查点。到资源/站点walltime时保存续训状态并报告partial，不把未到终点的最好结果改称最终结果。续训保存model/actor/critic/target critic/optimizer、normalizer、RNG、replay成员与游标、actor_version、环境/控制状态或精确可重放前缀。做不到精确恢复则另报restart，不宣称无缝续训。

端点是科学定义，不是旧式逐操作预算审批。继续现有版本不需每小步再问GO；改变主任务、奖励、动作时钟、模型规模或终点须先留新版本，不能在看到TEST后改配方。

## 10. 资源与单卡执行

WM/Actor训练首选一张A40；H200适合容量/并行需求较高的后继；4080S用于实际允许角色的推理、低并行调试。参数小不意味着全身WBC＋仿真便宜；没有测量前不承诺显存或训练小时数。

默认遵循实际站点角色：Core原生物理/推理，Vanda合法allocation中训练，固定revision传输小批状态与actor/WM checkpoint。一次批次产物完整后再消费；控制周期中无跨站RPC/SSH/heartbeat依赖。若已有合法单站profile能同站物理＋学习，可在P0预先选定更低开销拓扑。硬件型号本身不产生权限；AMD不跑重型任务，不改网络/系统/他人进程。

P1后报告：每1000000原子步的预测采样墙钟=1000000/实测聚合transitions_per_second，加实际训练/IO；分列64/256初始尝试与诊断成本。不要把1M/20Hz换算出的累计仿真时间当墙钟。并行数在正式前固定，后续资源变动必须说明并保留累计交互与成本。

原生全身链不支持4096并行时，不要求先把WBC重写为GPU可微程序；先得到低并行有效结论，再决定是否值得专门优化吞吐。不能为了速度替换成理想EEF执行器后仍报告T域。

## 11. 原生评价与进退标准

DEV32只在固定端点做完整R0 episode，诊断不选最终checkpoint。TEST100×3seed×2臂=600个预定完整R0 episode，最长原32s；这是设计分母，实际缺失/未运行另列，不当成功或删除。有效策略失败不重跑选成功；工程启动失败可按同定义修复重启并保留次数。硬安全事件按原生流程终止并定位，不为凑600而绕过隔离。

主要结果：每seed完整任务成功率、宏平均/最差seed、有效运行率、拒绝/真实硬安全、抓持/释放失败、恢复、超时；实际底盘运动和稳定性另报。恢复从阶段初态测，与R0任务独立。不要拿完美状态输入的结果当视觉部署。

算法继续门：B相对A最终成功率宏平均提升≥10个百分点且无安全退化；或达到80%DEV成功所需真实交互≤A的70%，最终成功率不低于A。它是项目决策目标，不是效果保证。计算95%配对层级bootstrap（layout和training seed），3seed仅探索性；区间含0则写不确定，不能声称显著。样本数不是独立窗口数。

同时报包括warm-up、reset、WM、Critic、Actor、物理和IO的总GPU/CPU/墙钟；等交互收益不能冒充等算力收益。若B有收益，再新增A的matched-compute或matched-update实验；它不自动从原正式对照中免费获得数据。

若没有收益，按“模型预测不足/梯度不可靠/探索缺接触/奖励代理错误/执行能力不够”定位，交 `NO_BENEFIT` 或 `INCONCLUSIVE`；不自动不断加步数。可靠Actor本身也可交付，勿将算法无提升等同于系统无成果。

## 12. 视觉路线：不强制绕回GR00T

状态Actor确认有效后，P6使用同一目标执行器生成真实双RGB＋实际TCP/夹爪＋目标＋动作记录。相机标定、时间、物体可见性、运动模糊、延迟/丢帧是数据属性，不把body/wrist两相机默认当固定基线立体相机。

先分三档测同一状态Actor：真值、注入已声明的位姿/延迟/遮挡误差、真实视觉估计。检测不可观测的接触信息，必要时增加历史或合法传感器；不能在视觉评价时补回隐藏物体状态。

视觉学生主线：冻结已固定revision的视觉编码器＋两视角融合＋4个决策时刻历史＋小Actor头，输出同56维参数。第一候选DINOv2 ViT-S/14；它仅是待验证的工程选择，不保证优于V-JEPA。不生成视频、不自动加载GR00T。初始200条物理教师尝试，保留全部结果；完整成功与恢复后缀分开标注。BC有效batch32、lr1e-4、10000 updates固定端点。

随后2轮各50条TRAIN视觉学生尝试，聚合实际访问状态的教师查询或接管监督；每轮5000 updates，最终共20000。`expert_query_not_executed`是反事实动作建议，不冒充已执行专家转移；执行接管后缀单独标注。不将学生失败从尝试分母剔除。成功Teacher未知隐藏信息若在RGB历史中不可辨，不能指望蒸馏恢复信息；报告ambiguity并修改可观测性设计。

视觉TEST另封存100个新布局，state teacher与最终visual student配对共200条，目标学生成功率距teacher≤10个百分点且无安全退化。此门槛为设计值；P6不能用P5已揭示TEST调参。首先交付一seed的视觉机制结果；跨seed可靠性后继复核。P6拟执行量不属于本次已经发生的计算。

## 13. 多任务与移动操作的后继

首T任务成立后，按DOOR→BOOK逐个增加task adapter/context，分别核奖励代理与接触执行范围。每个任务先单任务能学，再训练共享WM/goal-conditioned Actor，统一目标表示、mask和TRAIN/TEST划分。至少与各任务独立Actor比较，报告平均与最差任务，不把三个checkpoint叫共享policy。

主动移动进一步要求free-base运动、足地接触和基座工作空间覆盖；不能把站立机械臂任务称移动操作。不要自行新增BASE动作通道或改WBC步态。完整目标不能执行时，高层WM不承担修复底层缺失力控/顺应的职责。

P7在P5/P6后另封存多任务/移动/跨本体配方，当前不自动扩为无限任务队列。语言接口可先用结构化goal，后接VLM解析对象/目标；无需完整GR00T动作专家。

## 14. 工作包、接口与最小交付

| 包 | 实施内容 | 必需产物/完成定义 |
|---|---|---|
| P0 | 现状/后端/许可/来源核对 | RESOLVED_SPEC；正确target身份，缺口不伪造 |
| P1 | native薄adapter、compiler、restore、recorder、reset与吞吐 | 合同测试＋8原生诊断＋profile；真接触链可用 |
| P2 | 64→256尝试、WM200-smoke/5000正式 | 有效replay、事件覆盖、可恢复WM及物理单位误差 |
| P3 | 64状态×5方案真实分叉 | 排序/梯度/事件差异和所有失败/成本 |
| P4 | A/B同初始化pilot→三seed正式 | 连续真实计数、端点与完整恢复；不选seed |
| P5 | 原生R0测试与统计 | 预定600条件的实际覆盖、成功/失败/成本和结论 |
| P6 | 视觉学生与观测差距 | 真实视觉输入、独立200条配对测试；无真值泄漏 |
| P7 | 后继DOOR/BOOK/共享/移动 | 单独明确方案，不自动执行 |

拟放 `src/go2arx5_vla/world_model/`、`scripts/wm02/`、`tests/world_model/`；若WM01已有实现就适配并保留版本，不平行重复。建议模块：native_adapter、state_schema、action_compiler、replay、recurrent_dynamics、task_reward、imagination、actor_critic、train_loop、native_eval、visual_student。上游算法代码仅在license/依赖审阅后复用，不将上游OSC或私有WBC一并vendoring。

必须实现的薄CLI（目前均NOT_IMPLEMENTED）：resolve、qualify、collect、fit、diagnose、train、evaluate、distill。每个CLI只有现有launcher/profile绑定，无新scheduler/ledger。import与--help不得启动模型、仿真或下载。科学选择进spec/config，路径与凭据使用现有私有绑定，不能写入公开Git。

测试重点：时间/坐标/旋转/夹爪；raw-installed-consumed-achieved区分；完整恢复；跨episode/terminal/autoreset mask；未来控制记忆不泄漏；50ms WM与400ms Actor节奏；动作梯度穿过compiler/WM但不更新WBC；critic/target/normalizer/replay恢复；split不穿透；两个臂奖励与动作相同；视觉缺失与延迟。只跑变更相关回归，重大共用修改再扩展。

每阶段保留一份run manifest、完整checkpoint与数据引用、metrics/failures、费用、下一条真实命令。长证据进 `reports/WM02/<run_id>/`，根入口仅摘要。代码实现、CPU通过、原生物理、模型学习、任务成功分别报告。

## 15. 来源与本次发布边界

[上游算法/状态说明](https://github.com/emoPointer/supermimic_manipulation/blob/eda7da480c6d92a68cc41c9628b7efdae08cae23/docs/supermimic.md)；[上游任务定义](https://github.com/emoPointer/supermimic_manipulation/blob/eda7da480c6d92a68cc41c9628b7efdae08cae23/docs/tasks.md)。这是第三方实现，不是SuperMimic原作者的操作扩展；本文没有复现其成绩。

[本项目AGENTS/V7](../../AGENTS.md)、[现役后端实现](../../src/go2arx5_vla/target_sim/environment.py)、[原生参考桥](../../src/go2arx5_vla/target_sim/bridge.py)。后端存在不代表已部署/资格；实际记录以固定运行证据为准。

[MBPO](https://arxiv.org/abs/1906.08253)仅支持参考短模型展开/真实数据起点的研究思路，不是本文算法等同MBPO；[DAgger](https://proceedings.mlr.press/v15/ross11a.html)支持学生访问状态数据聚合的动机，不保证我们的视觉迁移结果。

本次仅发布详细设计与配置，没有训练器实现，没有GPU/仿真/远端作业，没有任何任务成功率新结果。运行环境在本会话不可用，因此CPU测试、程序化JSON/链接校验、全库测试和CI均NOT_RUN；不可复制旧WM01的17项测试成绩作为WM02验证。发布后需按固定SHA回读文件，Codex再实现并执行真实检查。
