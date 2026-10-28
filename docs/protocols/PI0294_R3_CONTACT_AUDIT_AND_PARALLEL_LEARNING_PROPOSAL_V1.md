# PI0294 R3：R2审核、接触归因与并行离线学习提案 V1

日期：2026-10-14。证据基线：`24ba08e2600a521f90e9c82a018258b88045397a`。

**状态：PROPOSED_AWAITING_EXPLICIT_ADOPTION；authorizes_execution=false。** 用户本次要求仔细审核并给出下一计划，不等于已经采用新物理保护接口、解除旧P1或授权新的PBS。本文上传和#11评论均不是执行收据。R2租约已关闭，R1/P1 FAIL、R2候选FAIL、父M512及所有预算永久保留；不能把原窗口或余额重新启动。现有incoming local-pc owner接续，不新增竞争owner。

## 1. 科学审核结论

R2按被采用协议停止是正确执行；完整B–E尚未完成。下一轮改变的是**实验依赖结构**，不是安全阈值，也不是追溯判定。

| 项目 | 固定实际结果 | 可以得出的结论 |
|---|---|---|
| G1-A 原臂+原夹爪 | 1081tick/2.162s原JOINT_LIMIT，全部保存高频行差0 | 失败可确定性复现；不是SSH丢包或本次随机波动的充分解释 |
| G1-B 原臂+全开 | 1118tick/2.236s ILLEGAL_CONTACT_OR_PENETRATION | 不闭爪也未解决安全，存在另一类接触/穿透终止；具体碰撞对象仍缺证据 |
| G1-C/D 原控制器保持TCP | 1200tick/2.4s均无native flag | 在此有限配对中，移动臂参考有贡献；未独立分离惯性、接触、几何或控制误差 |
| R2_VEL_C01 | 9.0931054738 Ns/m额外速度反馈；前五项过，第六项1078tick/2.156s失败 | 该单项候选失败，不代表所有阻尼方案必然失败；不得重跑选好结果 |
| 越界 | 右指0.0450229822345m，原有效上限0.045m | 超有效门22.982µm；名义上限0.044m，不能当仅超名义23µm |
| 力与测试覆盖 | 失败用例无motor饱和；前五项均保持初始臂TCP | 5/6不是完整任务或多工况资格，更不是83%策略成功；不能凭此申请更大力上限 |
| bias-only | 未执行；近末端重构bias正0.00890–0.01831N | 证据不足以支持优先试此候选，但不能把NOT_RUN标为FAIL |
| 消耗 | 10启动、15677积分、0RGB、0GPU/PBS/训练 | 模型仍M512，约31.354s累计仿真时长；Goal墙钟约23m30s不是物理/CPU用量 |

worker CPU约11.13967s，额外保存数组核对约0.05358s；元数据、SSH与staging开销UNOBSERVED，不写成0，不由Goal墙钟反推出CPU。

### 本次审核边界

已读R2 FINAL_REPORT、FORCE_BALANCE_AND_TIMING、DIAGNOSTIC_MATRIX、STRESS_RESULTS、EVIDENCE_MANIFEST、RESOURCE_DELTA和43ced701的实际代码差异。未SSH、未重新执行仿真、未下载私有NPZ独立复算。原命令算术及G1逐行重现可复用；不再重做没有新增信息的四组对照。

### 关键推理

1. 失败涉及移动机械臂条件。B出现接触/穿透，不能把整个问题缩成“给夹爪加阻尼”。
2. 保存状态重构的末端右指constraint约+0.811N，但qfrc_constraint是多个约束的广义合力，不等于已识别的桌面或方块接触。原warmstart/内部反力未记录；来源仍需按contact/limit约束分解。
3. 质量块对角0.091402kg的标量临界阻尼公式不构成运动臂、多接触系统的稳定性保证。原ctrl显式计算的速度反馈与MJCF被动阻尼并非自动同一种离散积分处理。
4. M512只有512更新。它可能生成静态SafetyGate合法、但进入不安全接触状态的运动。当前证据既不足以宣布“执行器唯一bug”，也不足以宣布“数据/模型学不出来”。
5. **上一版把D1/P1R合格设为任何离线梯度更新的前提，造成了不必要的串行依赖。** 修复评估执行系统与继续基于冻结成功数据的离线BC可以并行；不可把这一修订隐瞒为已通过旧P1。

## 2. 本轮目标与明确的范围变化

本轮不再调R2_VEL_C01，不猜第三套增益，不训练WBC。目标是同时交付：

- CoreLab：定位首次有意义的接触/约束事件，给出碰撞对象、力源和执行跟踪证据；实现并资格验证一个**仅拒绝、不改动作**的仿真专用执行前预检S1。
- Hopper：从完好的M512完整状态继续同一v3.0离线训练，新增1536步到**M2048**；即使接触归因尚未闭合，离线训练仍可完成。
- 交汇：S1通过工程/Teacher资格且无全局实验环境不一致时，用同一D0+S1评价M512/M2048，报告任务成功、执行前拒绝与真实native safety三类结果。未准入则只交离线结果和M2048，不运行未保护模型。

本轮**不授权M4096/6144或独立CONFIRM解封**。先得到M2048及两任务实际学习差异，再决定下一训练规模，避免没有行为证据就无限续训。

关键变更须在新的#11采用回执逐字明确：

1. `OFFLINE_TRAIN_ALLOWED_WITH_P1_D1_UNQUALIFIED=true`，只针对冻结静态数据训练。
2. `LIVE_MODEL_EXECUTION_REQUIRES_S1_QUALIFIED=true`，S1不是D1，旧D0/P1 FAIL保持。
3. `S1_IS_SIMULATOR_PRIVILEGED_REJECT_ONLY_PREFLIGHT=true`，不是裸Policy、真实机器人保障、任务规划器或新WBC。

未经这三项及窗口预算明确采用，仅可阅读/形成代码草案；不得调用新神经/物理或重新启用R2租约。

## 3. 固定不变的资产与方法

- 交接入口：根AGENTS/START_HERE与`docs/handover/2026-10-14/HANDOFF.md`；使用现有CoreLab enter/doctor和私有Hopper启动器，不为本轮重写统一CLI。
- 原M512数值实现父源：`a171997a704072e127d16410ba584c8aa0ddd573`；原R1推理修复源：`021d813fd669561a24ffbf89431a4a9812e0f2ca`。
- R2源：`43ced701d73f1df73389711f03b5b5ed68c90867`仅供失败审计；其可选finger_driver hook不得默认绑定已淘汰R2_VEL_C01。D0明确未绑定该hook，关键数值文件按原seal核验。
- M512私有16成员原inventory：`6fdc10c2cfb13841bbabd444aa0ab8a4a6aef09adfbffae7d5fb4dee24747a8a`；优先用现场已有完好状态，不反复下载约25.54GB。
- 公共M512 READY：`c698f54d0fc422ebfdd1e2ce1fc96552b6930c52`，repo `leoyuer/pi0294-public-artifacts`。其三份脱敏metadata与私有版本不同；不要求公共独立恢复作为主线前置。
- v3.0 manifest：`b8ddcb9808e6dfac4f3ac5fa5a4828128fa9644a330c17d1c57bf833ce5889df`；statistics：`fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7`。
- 官方source：`51d4c89f72fda44cbf77285c6a8114b52676b8a1`；base revision：`2fc962b973bccdd5d8ce4f67cc63b264d6886495`。
- EEF10/H40/prefix8、2ms、20Hz、500Hz native safety、原D0控制器与几何/摩擦/质量/solver不变；模型输入仍RGB+语言+实测EEF/夹爪，不增加privileged任务信息。
- 不放宽[-.001,.045]m有效夹爪范围或其他安全阈值，不改名义[0,.044]m，不增加8N上限，不裁剪qpos/qvel，不加隐藏力/吸附，不关碰撞，不将opening强制二值化或平滑后冒充原policy。
- 旧实验身份、失败和已退役候选不覆盖；v3.1不热切换、不重新拟合stats、不采新训练数据。

## 4. P0：新的分支式状态机，不再串行等待D1

新owner核对两端实际进程、账本、私有绑定和当前项目权限，采用新run UUID/manifest/明确窗口；原R2只读且保持终态。

状态机分开：

- 训练：`INPUTS_READY → FULLSTATE_RESTORED → OFFLINE_M2048_COMPLETE`。
- 仿真：`CONTACT_ATTRIBUTION → S1_CLONE_QUALIFIED → S1_TEACHER_QUALIFIED → PROTECTED_EVAL`。
- 科学：`EVIDENCE_INCOMPLETE / LEARNING_SIGNAL / TASK_FAILURE / ACTUAL_NATIVE_SAFETY_FAIL`，不自动写working policy。

某条线遇到本线阻断，只停止该线；共同数据/模型身份错误、权限/总预算问题才停止相关全部范围。任何真实受保护live rollout仍出现native safety则立即停止所有该版本live模型臂，保留失败；不因它中断健康的纯离线训练，除非共同状态/数据完整性受影响。

旧P1仅是旧driver/model组合的实际FAIL。新任务显式允许离线优化，不向旧代码注入假的P1 PASS，也不删原资格检查。入口应按执行模式验证新授权，不再把“许可离线训练”和“许可live执行”混成一个boolean。

## 5. CoreLab线A：接触与力源归因，不再盲改阻尼

### A0 先读本地现有结果

按R2 EVIDENCE_MANIFEST，从私有注册root读取`evidence/G2-failure_prefix_hold-R2_VEL_C01/RESULT.json`中的`final_contacts`（源码实际有该字段），以及G1-B的已保存状态、场景XML和安全触发条件。检查具体是哪一对geom触发ILLEGAL_CONTACT_OR_PENETRATION：机器人-桌面/地面，物体/手指接触穿透超阈，或其他碰撞。汇总中没列contact名字，不代表已证明是桌面。

必要时在保存状态零积分计算接触几何，标注RECONSTRUCTED，不当作原始力；仅在缺实际力时进入A1。保留完整触发原因列表，不能只看最终异常字符串。

### A1 最多4个有新增字段的短诊断

优先一次原G1-A、一次原G1-B带仪器记录，最多各2.4sim秒/1200积分，原first-abort即停。可有一次只读logger一致性检查和一次有明确原因的工程修复；禁止重复原4组获取相同表格。已淘汰候选可只读其保存结果，不再次进行候选资格挑战。

记录实际的：
- 每次mj_step前真实q/v、ctrl、finger/arm target；返回后、后续mj_forward覆盖前可用的qacc、bias/passive/actuator/constraint、contact/efc数组；再记录原snapshot时点。缺不可观测中间量显式UNKNOWN。
- geom/body名称、接触点/法向/dist、接触完整6D wrench、相关Jacobian与左右finger广义力；efc按limit/contact/equality/friction类型归类，核查sum(J^T f)与qfrc_constraint一致性。
- 目标与实际TCP位姿、指尖及其他碰撞几何到桌面/物体的净间隙、臂target slew与qvel、actual离散加速度，以及力饱和情况。
- 对任一selected源、控制器或采样路径更改先做差异seal。logger不能增加会改变live warmstart的额外mj_forward，也不将implicitfast换为step1/step2的另一积分路径。

先证明新的只读logger不改变A的原1..1081行和首违规tick，再引用instrumented forces。积分器的continuous qacc不强行等同(v_next-v_prev)/dt。默认MuJoCo参数与运行次序全部保持。

### A2 决策必须基于接触证据

- 若合法模型目标把真实碰撞几何送入不允许接触/穿透：记录`POLICY_OR_REFERENCE_CONTACT_VIOLATION`，不是再给finger加力硬顶过去。
- 若发现明确的视觉mesh误参与碰撞/重复geom等资产模型错误：输出最小修复diff**草案**及数据影响，本轮不自动改几何；离线训练可保留但必须标明旧domain，物理线等待显式模型修订。
- 若无接触而是惯性耦合/跟踪力不足：量化对应力和跟踪误差，给下一版本driver证据；本轮不继续扫kd或把qfrc_bias当成全耦合补偿。
- 若仍不能隔离：如实标识；不把“可能接触”写成唯一根因。该不确定性不阻断纯离线M2048，不准据此宣称D0修好了。

## 6. CoreLab线B：仿真专用S1预检，仅拒绝、不改动作

### 为什么允许新增这一层

原SafetyGate只证明参考层部分条件，没有预先证明这200tick实际轨迹不会越界。S1在同一仿真模型的**独立状态副本**上检查即将执行的prefix；这是额外安全准入，不是放宽检查，也不是第三种夹爪控制器修复。D0保持原状。

S1使用精确仿真状态（包括物体/接触），因此是**simulation-privileged safety filter**。它不学习任务、不能读取Teacher阶段/成功目标来挑动作，结果必须标`checkpoint + D0 + S1`，不与旧无S1裸policy直接合并成功率，不能声称可直接移植真机/WBC。

### B0 实现上限与克隆契约

复用现有Engine.step/relay，不另写动力学。检查安装版本3.2.7实际Python可用copy接口；复制完整mjData或经验证的等价状态，连同warmstart、time、act/ctrl、controller arm/finger targets、IK缓存/ik_data、所有会影响推进的Python状态、已有safety状态和RNG。模型只读共享或完整复制；所有可变数组不得别名。渲染器不共享可变上下文，不需要在shadow生成新RGB。

只允许原样8点/200tick。先在副本运行原D0和全部原500Hz safety检查；任何违规、缺字段、非有限状态或克隆验证失败→REJECT。未修改live状态的hash是门。失败的副本停止于first-abort，不继续穿透观察。

不得把shadow最终q/v拷回live“节省执行”，不得隐藏shadow积分为0，不得把shadow帧当真实在线观测。live每200tick仍通过原执行和实际ACK取得新RGB。

### B1 最小资格（不使用独立确认集）

1. 从至少8个不同安全上下文验证shadow vs 无附加logger的live原D0逐tick一致，覆盖两任务/开闭/臂运动与已有合法接触。context在读模型新结果前按已准入TRAIN教师数据固定。
2. 最小要求：相同控制点/ctrl、相同native verdict/首flag，位姿/速度/内部控制targets最大绝对差<=1e-10且相同计算模式下优先字节一致。不得结果出来再放大容差。阈值只是状态克隆工程门，不是安全阈值。
3. 已知R1/G1-A第6块必须被shadow拒绝，live不执行该块；明确预测tick偏移81。G1-B已知失败块也须拒绝。复用保存的失败做offline unit证据，必要replay单独算诊断，不算在线policy成功。
4. 错误或过期观测、重复packet、状态在预检与安装间改变、nonce/动作hash不匹配必须拒绝。预检通过后绑定整份state/code/model/command身份，原子校验再执行。
5. 至少4条原TRAIN Teacher完整32s回归（F0/000与F1/000，各T1/T2）通过全部原任务谓词、零S1拒绝、零高频违规。若Teacher也被拒绝，先检查实现或场景，不能关闭预检。

时间盒：S1实现+资格最多2小时主动工程工作。无法达到等价契约则STOP_LIVE，继续离线训练和离线分析；不为快速视频部署未经验证的S1。不得以需要先完善S1阻止模型训练启动。

### B2 拒绝的语义

S1拒绝后原地暂停仿真、终止该episode，保存`PRE_EXECUTION_SAFETY_REJECT`和shadow首违规；不提交危险prefix、不用额外hold action偷偷推进、不自动重采模型噪声、不缩短prefix到恰好能过、不卡住等Teacher替代、不修改模型夹爪或EEF、不重新计为NOT_ATTEMPTED。

该episode计入计划与已尝试分母，任务成功为false，policy原提议安全未过；live_native_violation为false仅因为提前拒绝。大量拒绝不等于得到安全可用policy，更不能只在接受子集上报成功率。

模拟暂停并不是物理机器人急停保证。真机/移动/WBC仍范围外。

## 7. Hopper线：独立从M512完整续训至M2048

在新授权、合法同一H200 allocation、固定数据/model/数值资格成立后即可启动，不等待A2根因或S1完成。复用完整M512、预训练source/processor和原resume工程证明，不从pristine重训，不重复全量HF下载。

新增1536步，到全局M2048。全部固定：
- 原v3.0和stats，不使用v3.1、失败重放或新Teacher记录做训练。
- 原冻结VLM/可训练projector+动作DiT及实际参数集合，官方masked flow-matching。
- microbatch2/accum16/effective32、workers0、BF16、确定性CUDA math SDP、原优化器参数。
- scheduler horizon6144/warmup308；从global512、consumed16384续接Adam、scheduler、Torch/NumPy等RNG、sampler真实消费cursor。
- 不重设warmup、不更换task/phase采样，不增加loss、不改SE3/夹爪编码，不因DEV结果改训练配方。

先核对恢复到512和正确下一批；必要隔离恢复测试最多4更新，不将分支权重计正式候选。评估与训练切换前保存/恢复训练RNG和mode，不让模型生成消耗改变已封存训练链。

只需LOCAL_CHECKPOINT_READY即可同节点评价，REMOTE_ARCHIVE_READY单列；归档403不杀死健康本地训练。新M2048完整状态先本地校验。公共脱敏衍生最多1份30GiB，只有本新包明确采用公开checkpoint权限才发布；raw/视频/私有资产不因此公开。

若CoreLab证明数据/processor/模型身份不一致等共同错误，则停止相关训练；若只是D0执行接触失败，训练可继续但模型状态为UNQUALIFIED，不作部署承诺。

## 8. 两条线交汇：受保护开发比较，而非独立成功资格

M512不要再次跑原无预检失败。S1资格与合法GPU成立后，在固定TRAIN两组F0/000、F1/000各T1/T2运行4条；可在M2048训练前或保存训练状态后执行，以不空等GPU为原则。S1未完成时先训练M2048，再统一加载M512/M2048评价；不要让GPU常驻等CoreLab开发。

之后同样4条TRAIN评价M2048，使用配对seed/初态和同一D0/S1。目的：判断早期学习是否改变危险轨迹、阶段和任务行为。两份checkpoint都完整保留，M512旧FAIL不覆盖。

M2048固定DEV8：沿已封存的F0/000、F1/000、F2/000、F1/001，每组T1/T2。各任务4条；不替换拒绝/失败，不打开独立CORE/CHALLENGE，不用这8条进行driver/预检超参数搜索。

每条必须记录：task_success、early_rejection、live_native_failure、时间/完成prefix、首接触类型、首次任务失败阶段、generated→installed→achieved误差、夹爪事件与实际范围、shadow/live成本、checkpoint/driver/preflight/dataset身份。原500Hz flag覆盖20Hz最后score；无完整尾部则UNKNOWN/FAIL，不默认安全PASS。

模型输入没有新增taskID或特权状态。S1只判安全，不检测是否接近正确目标来决定接受；Teacher与模型均使用同一保护链。

**本轮候选selected=NONE、deployment=NONE**：M2048 DEV是开发诊断，不替代原M2048/M4096选型与独立确认。只有未来另经授权的单候选确认达到原6/8、两任务配对5/8、回归6/8及零真实违规，才能报告原MULTITASK_SIM_WORKING_V1；S1改变了测试系统，须在报告标题披露。

## 9. 根据本轮结果决定下一实验，避免无界修复

| 结果 | 下一步，不属于本轮自动执行 |
|---|---|
| M2048在每个任务出现真实成功，且拒绝减少/任务阶段改善 | 同一v3.0/D0/S1继续至M4096，跑同DEV，固定候选做独立确认 |
| M2048在TRAIN也不学或两任务都无阶段进展 | 固定少量成功样本的过拟合/条件化诊断，检查语言、事件时序、实际trainable参数；不是自动扩数据 |
| TRAIN成功、DEV下降 | v3.0/v3.1覆盖对照；先数据资格与固定统计的新训练分支，不热换数据 |
| 多数拒绝来自同一不合法接触 | 基于contact atlas定位policy/参考碰撞；考虑后续恢复数据或明确的参考安全策略，不再单调kd |
| Teacher在相同条件也有接触或动力学故障 | 实验环境/执行器最小修订方案及数据lineage，保持本轮结果不冒称同domain公平比较 |
| S1预测PASS、live仍违规 | S1资格撤销，停止该版本live；查克隆/时序/引擎一致性，不增容差；离线训练结果可独立保留 |

不要求把R2的D1标PASS来获得新训练许可。相反，明确保留D1=NONE并用新授权记录不同的学习/物理准入规则。

## 10. 建议一次采用的新预算（不是自动授予）

新T0由incoming owner在明确采用后记录；新总窗口8小时，最后1小时收尾。旧父/R1/R2时间和开销独立保留，不继承未用余额、不重复计数。排队/断连不重置。

| 项目 | 新上限 |
|---|---:|
| 新PBS调用 | 2，失败也计数 |
| H200 allocated总时间 | 6小时，1张团队GPU；账号独立allocation仍最多2 |
| 正式新增更新 | 1536，到全局2048 |
| 隔离恢复测试 | 4更新，仅必要时 |
| 模型H40查询意图 | 1600，含失败预留 |
| CoreLab CPU | 24 core-hours、最多8核并发；live worker串行 |
| Hopper CPU | 16 core-hours |
| live/诊断物理启动 | 32，含最多4短instrumentation replay及Teacher/模型 |
| live/诊断积分 | 512000 |
| live RGB pairs | 20512；非观测诊断截图另计render<=120，不能冒充episode观测 |
| shadow推进调用 | 2500，独立记snapshot/ID/step |
| shadow积分 | 600000；不得算0物理或并入live里程 |
| 全部新增积分 | <=1112000，同时服从以上分项 |
| 新CoreLab/Hopper/local-PC峰值 | 16GiB/120GiB/2GiB，Hopper个人余量>=20GiB |
| 新公开checkpoint | 最多1份且<=30GiB，仅明确采用后 |
| 新采训练数据、WBC/真机、历史删除 | 0 |

现场按实际同时存在文件union预留，不重计已有M512。保留完整唯一M512、M2048和失败证据。不得为消除预算警告随意删除历史完整状态。若恢复测试峰值或打包临时峰值超限，只做已可完成阶段并报告不足。

计划实际live面板20条：Teacher4 + M512TRAIN4 + M2048TRAIN4 + M2048DEV8；其余启动余量仅诊断/有依据工程检查，不替换失败case。shadow全部CPU也算24core-hours内。至少预留2GPU小时用于实际评价；不足先保留M2048和已完成评价，而不是继续M4096。

## 11. 交付和停止规则

- 首个CoreLab交付：`CONTACT_EVENT_ATLAS.json`、`CONSTRAINT_FORCE_BREAKDOWN.json`、`LOGGER_EQUIVALENCE.json`，回答具体接触对和首次时序；不只是又一份joint_limit表。
- S1：`S1_SPEC.json`、`SHADOW_CLONE_EQUIVALENCE.json`、`KNOWN_FAILURE_PREFLIGHT.json`、`TEACHER_S1_RESULTS.json`。
- Hopper：`FULLSTATE_RESTORE.json`、`M2048_TRAINING_RECEIPT.json`、`CHECKPOINT_MANIFEST.json`；无模型行为时明确UNQUALIFIED。
- 交汇：`M512_S1_TRAIN4.json`、`M2048_S1_TRAIN4.json`、`M2048_S1_DEV8.json`、真实私有视频、失败/拒绝完整分母。
- 汇总：`RESOURCE_DELTA.json`、`FINAL_REPORT.md`、准确恢复指令；不存在的步骤NOT_RUN。

本轮工程只修改授权mode分支、最小只读instrumentation、S1 wrapper和必要测试；不迁移环境，不扩场景，不自动运行166份历史owner脚本，不重复上传M512。共享NFS只有一个mutation owner；源固定SHA+CPU相关测试+现场qualification，不能把CPU CI当物理安全证明。

按GitHub-first：普通提交push、文件blob/commit回读，#11记录实际科学状态，#47记录资源/部署；所有私有绑定、原始NPZ、视频与凭据留授权存储。新增未来脚本需读取明确authority，不能删除原guard来绕P1。S1真实回归失败只停止物理线，离线模型线独立按预算收尾。

## 12. 新owner直接读取指令

读取本提案与`reports/PI0294/design/R3_CONTACT_AND_LEARNING_TASK_CARD.json`，刷新#11/#47及当前lease。现在仍为PROPOSED；先向用户呈现一次合并采用项：离线训练与D1门解耦、仅拒绝S1的仿真保护接口、上述新窗口/预算、最多一份脱敏checkpoint公开权限。获得明确采用并发布#11回执后，CoreLab接触/S1与Hopper M2048自动并行；不逐小阶段再问同项GO。不能把本文件发布当成已经获得采用。

## 固定证据与技术依据

- [R2终局](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/24ba08e2600a521f90e9c82a018258b88045397a/reports/PI0294/20261014T072142Z-incoming-pi0294-r2/FINAL_REPORT.md)
- [R2实际力学](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/24ba08e2600a521f90e9c82a018258b88045397a/reports/PI0294/20261014T072142Z-incoming-pi0294-r2/FORCE_BALANCE_AND_TIMING.json)
- [R2对照](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/24ba08e2600a521f90e9c82a018258b88045397a/reports/PI0294/20261014T072142Z-incoming-pi0294-r2/DIAGNOSTIC_MATRIX.json)
- [R2压力结果](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/24ba08e2600a521f90e9c82a018258b88045397a/reports/PI0294/20261014T072142Z-incoming-pi0294-r2/STRESS_RESULTS.json)
- [MuJoCo3.2.7动力学与约束](https://mujoco.readthedocs.io/en/3.2.7/computation/index.html)
- [MuJoCo3.2.7状态/warmstart与复制](https://mujoco.readthedocs.io/en/3.2.7/programming/simulation.html#state-and-control)
- [MuJoCo3.2.7 API：contactForce、Jacobian、copyData](https://mujoco.readthedocs.io/en/3.2.7/APIreference/APIfunctions.html)
