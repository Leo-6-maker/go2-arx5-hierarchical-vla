# SLA × Go2+ARX5：冻结 WBC 后的接口闭环伺服验证及 WBC 优化决策

**2026-11-23 · PLAN_ONLY / PHYSICS_NOT_AUTHORIZED / GPU_NOT_AUTHORIZED。** 独立后继于 [PR #435](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/435) 冻结结果；不改写V1.5停止规则、旧H40/G7、D1/D3任何原始分母，亦不修改正式OCS2/WBC、MuJoCo和底层机械臂PD可执行文件。第一步只做CPU分析与隔离脚本，**当前新增native请求/tick/训练/冻结SLA前向=0**。

## 1. 本计划建立在什么事实之上

- **D1匹配初态**：固定40次新启动，18次通过原初态匹配并执行，22次筛除；完整4组2×2对照中，OO继续旧EEF/base目标100ms的TCP平均Δz≈+0.04mm，EEF改为当刻实测位姿的主效应≈−15.6mm，底座主效应≈−1.1mm。**现有直接证据确认参考重设是主要短期触发，而非启动后必然自由下坠。**
- **D3冻结OCS2已知路径**：12/12真实无接触MuJoCo卡完成；绝对TCP位置RMS=28.36–30.68mm，增量位置RMS=1.66–3.97mm；4mm X、4mm Z指令只得到约2.68–2.73mm和2.53–2.58mm，纯旋转1.75°得到约1.48–1.51°。说明冻结系统仍有响应，但不是单位增益或毫米级准确servo；曲线平移拟合增益≈0.35–0.37且延迟到300ms搜索边界时不能认定物理延迟就是300ms。
- D3 3张B2匹配卡绝对RMS仍在28.71–30.05mm；**0/6个ONE/P10配对**通过初态资格，不能宣布两种更新策略性能等价。结束时关节命令FK与真实TCP高度差24.15–24.63mm；D2几何归因第3/4轴约16mm/8mm。D2全部1005个PD回调的电机力矩可以在正确传感器相位下精确重算，无电机限幅，显式重力前馈为零，但不能单凭这些证据证明增加重力前馈就能修复全部偏差。
- V1.3原生SLA→旧H40/G7科学准入为0/16；B3长时域消息CPU回放仍有13/16意图存在至少一窗原包络问题。教师持物中有8/16个学生窗口要求打开夹爪；学生owner/ACK=0/16。**不能因D3完成而开放原B4**。用户已批准V1.5停止规则得到满足；本PR不是任何额外科学物理权限。
- 既有D1+D3新科学资源52 requests / 317,556 ticks（原首波OO初始化另计），均已关闭，原B2历史租约不可复用。原始证据：[D1](../../reports/SLA_MEMORY_D1_MATCHED/README.md) · [D3](../../reports/SLA_MEMORY_D3_KNOWN_PATH/README.md) · [控制归因](../../reports/SLA_MEMORY_V15/D2_POLICY_AND_PD_CLOSURE.md)。

## 2. 回答“是否一定要修改WBC”：预登记两条不可混淆的路线

**A：FROZEN_OCS2** = 冻结已加载OCS2/MPC、末端PD、腿部policy、硬安全/碰撞/关节限位、2ms物理和模型。上层允许进行：
1. `A0_REFERENCE_CONTINUITY` 仅保持前一条已消费世界系EEF/base参考，按10Hz发布新序号合法2s/41点窗口、0.4s TTL、保持绝对MPC时间与轨迹ID，不把 measured TCP当作新的command anchor；这部分属于纯接口。
2. `A1_EXTERNAL_TASK_SERVO` 在A0之上新增显式**外环控制器**，从经验证当前TCP和任务期望 `T_des_world` 构造有界SE(3)修正，更新上一次已ACK/consumed的EEF参考，非暴力重置。尽管WBC权重不变，A1不能再称为“纯格式转换”；它改变闭环动力学，需独立新实验、安全资格和对照分母。不能以D1的旧目标≈27mm偏差证明它必然收敛。
3. `A2_SLA_OFFLINE_ONLY` 对官方冻结SLA与目标后训练SLA做相同场景的纯离线意图/事件/动作分布比较，训练若新开GPU资源必须另行授权；**不得在A1精度与gripper owner未获准前使其发真实控制命令**。

**B：LOW_LEVEL_SERVO_RESEARCH** 是**另立控制器科学版本**，不能称为冻结WBC A组：仅在A1不能达到任务精度或受约束不可行时，测试第3/4轴PD负载平衡、增益/速度目标/显式力矩前馈、MPC任务权重或低层joint-reference定义；原冻结可执行文件与B分支必须有不同SHA、场景/物理预算、完整安全/停止独立资格。拒绝未经诊断将WBC改为“增加+20mm TCP偏置”；这不是动力学补偿的替代。

## 3. A0/A1的合法 Reference Manager 语义

```text
VLA / known geometric task target T_des_world(t)
  -> timestamped TaskSpaceIntent (pose+task event+source frame/TCP)
  -> ReferenceManager {current measured TCP, old acknowledged/consumed EEF/base reference,
                       episode/trajectory revision, policy/result age, gripper command ledger}
  -> A0: old reference continuity + explicit future versioning
  -> optional A1: bounded external task-space servo on old reference
  -> existing original H40/G7/safety checks & separate new-servo envelope
  -> 10Hz 41-point,2s native OCS2 target window, TTL400ms
  -> frozen OCS2/MRT/arm PD/leg controller -> 2ms MuJoCo -> achieved feedback
```

- 位姿输出相对**实际任务目标**而非把TCP measured开度/姿态冒充上一次command；如参考绝对为世界系，仅做一次armbase→world及工具TCP变换。
- A1候选更新在世界坐标位置可写 `p_cmd^{k+1}=p_cmd^k+sat_C(Kp (p_des-p_actual))`；姿态在SO(3)上做左右约定正确的群增量，使用源到目标相同TCP/姿态；所有更新依据当刻已ACK/consumed旧参考并带过期时刻，若实际状态滞后则等待或HOLD。
- `sat_C`不仅限单次位移，还必须满足原线/角速度、加速度、连续旧命令接缝、夹爪独立G7、base关节接触/工作空间安全，控制器应有防积分饱和和稳定判据。**不能拿现有D3的4mm→2.7mm增益直接做全局1/0.67倍率补偿或跳变28mm**；频率、增益、作用时序须先在小幅、无接触CPU预演并预登记。
- 参考MPC发布10Hz与SLA新推理/P10 0.5s独立。时间重映射不能切断靠近物体前的抓持事件；不能重置源轨迹为过去；夹爪只允许学生拥有的reset或真实owner/ACK、非接触安全交接，原8个抓持中张开候选明确BLOCKED。
- 每个阶段记录`raw_intent → time_law → previous_ack_ref → bounded_compiled_ref → original_gate → installed/consumed → policy_joint_goal → achieved_TCP → object_contact`，不同分母；A1绝不可报告为纯SLA权重改进。

## 4. 分阶段推进和独立许可

### F0 已完成的全部数据回读 + 新CPU诊断（可立即独立做）

- 固定V1.5 12张D3原始known-path、18张D1匹配、22张筛除、PR#435 SHA；在同一工作区逐条件审计`p_actual-p_cmd`、`FK(q_cmd)-FK(q_actual)`和EEF/body-relative的时序，拟合局部**增量**响应与残差，分离初态两类簇。
- 用闭合误差恒等式及非SLA已知目标，模拟一个**不调用真实物理**的A1外环模型，报告预测收敛可行域、不确定性和受安全约束的上界；标记`OFF_POLICY_MODEL_ONLY`，不能当实际成功。对错时戳、伪ACK、命令大跳、重复控制积分、夹爪误接管设置负例。
- 交付`F0_ACTUAL.md/json`、A0/A1源码SHA、闭环候选预登记；原D3 0/6严格匹配配对结论不变。

### F1 A0纯接口无接触能力复验（**需要新增原生许可**）

- 独立后端契约预检通过后，以单次合法旧参考持有与安全刷新确认不再人为重设EEF；按既有初态匹配规则登记完整启动分母，禁止事后只筛成功卡。
- 以实际任务需求定义预注册绝对位置/姿态容差、目标工作区和可接受的命令幅度。固定现役WBC、全部安全限制、稳定计时，测起点稳态误差和已知小运动的绝对/增量跟踪。若与原D1/D3不一致先停查ABI，不能把对照合并。
- 这一步验证A0不消除约24mm低层gap而可能仅消除15.6mm额外重设瞬态。

### F2 A1小幅有界闭环外环资格（**需要独立新的原生许可；不自动运行**）

- 预注册同目标/seed/初态的A0 vs A1，先短、无接触、小增量参考，收敛稳态与动态；记录输入、反馈有效时刻、下一次真实ACK、每次增量大小/速率、运动幅度、策略年龄、是否震荡/限幅，准确度同时用绝对和增量RMS。
- 要求在**所有预登记**初始条件下稳定朝目标收敛，明确消除系统性偏置的幅度；以物体抓取净空、传感器精度和接触余量推导具体`epsilon_task`，在观察效果前写入注册。可将≤5mm作为**待论证的精细抓取工程目标示例**，不是既有科学PASS门。
- 如果新参考持续命中原门、收敛不足、出现周期震荡、重新触发初态差异、接触或超限，记录`FROZEN_INTERFACE_TRACKING_BLOCKED`，不允许手调后继续当原实验；下一步转B分支独立控制器改进。

### F3 需要时启动B：WBC低层优化分支（仅方案）

- 非SLA、非接触、冻结机器人/任务环境，以本体低层定位问题为目标，对第3/4轴施加独立可审计的低层servo/重力项/PD或MPC目标权重更改，在相同已知参考上与A0和A1比较，按修改版本锁定新binary/config SHA。
- 优先按加载的物理/关节传感器相位确认原已复现的1005/1005 PD输出及是否处于非零负载下位置平衡；不能把显式FF=0直接叫做错误，也不能用新的物理控制器与冻结组混算学生模型能力。
- 只有新控制器通过完整安全回归/资源授权后才能提议另一次SLA集成，不将其改善计入冻结WBC的VLA跨本体迁移结果。

### F4 SLA post-training与B4研究的分离门

- 可离线、并行研究SLA对Go2真实20Hz双RGB历史、右臂绝对位姿、源官方processor/stats、单臂技能OOD的后训练效果；保存teacher **command**、achieved TCP、原场景接触/夹爪owner作为不同字段。比较`FrozenSLA+FrozenWBC+interface` vs `PosttrainedSLA+同一FrozenWBC+same interface`，并保留旧H40基线，不把WBC模型权重变动混入。
- 只有F2证实目标任务精度、安全及夹爪所有权，才在新独立TRAIN holdout进入抓取/抬升≥20mm/持握≥1s/放置闭环；StackCube当前仍可为OOD单列。任何新GPU资源/原生任务额度须新单独授权，**不能继承旧B4未运行的候选budget**。

## 5. 决策与交付口径

`F0_CPU_ONLY → F1_A0_NATIVE_QUALIFIED → F2_A1_NATIVE_QUALIFIED → (if fails) F3_WBC_RESEARCH_FORK OR (if passes) F4_SLA_PAIRED`。其中`F0`可先做，之后遇到资源/许可不足立即`WAIT_AUTHORIZED_NATIVE`；旧V1.5/B4科学STOP保持有效。禁止真实硬件/训练/未授权G7修改/遮盖PD误差/用object GT修学生动作/偷偷让教师接管。每阶段`ACTUAL.md/json`、完整启动与筛除分母、独立控制器版本/安全门/接收器身份、所有命令时戳/ACK、physical_ticks、STOP后ticks与GPU使用、失败类型。

**核心可证伪假设**：不改WBC低层权重/参数，仅添加正确的参考持续性和有界任务空间反馈，能否使已知EEF目标的真实绝对误差收敛到预注册任务要求？只有答案是“可以”，才能把当前Go2+ARX5冻结执行器作为精细SLA抓放后训练的稳定比较平台。
