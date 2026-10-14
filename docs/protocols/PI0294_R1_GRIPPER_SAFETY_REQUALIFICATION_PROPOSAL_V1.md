# PI0294 R1 夹爪安全阻断审核与专项修订建议 V1

日期：2026-10-14。状态：**PROPOSED_SCIENTIFIC_AMENDMENT_AWAITING_EXPLICIT_ADOPTION**。

本文件响应“P1真实夹爪限位失败、两PBS终态、需#11明确修订”的报告。它是审核和待采用任务卡，不把状态报告推定为修改控制器或新增物理执行的授权；未获用户明确采用前，R1保持BLOCKED，不启动第三次PBS、不重放物理、不旁路P1。发布本文件不是新执行收据。

## 1. 结论与证据

审核固定证据HEAD `0a174b107672937af2271012efe7e45f7cda0975`，科学执行源 `021d813fd669561a24ffbf89431a4a9812e0f2ca`，原run `20261014T025730Z-localpc-pi0294-r1`。保留其全部预算与失败，不覆盖原任务卡。

- R1已经真实加载M512，生成48条离线H40，专家和模型SafetyGate均0拒绝；真实闭环又生成/发送6条H40。不能再报告“尚无任何模型推理”。
- 首个TRAIN F0/000/T1仅完成5个完整prefix ACK；第6个prefix在第81次积分停止。总1081积分、44对RGB，失败发生于2.162s；3个剩余开发case为NOT_RUN，T2模型行为未测。
- `right_slide=0.045004271727453214 m`，原有效上界0.045m，超出4.271727微米。名义行程上界0.044m，原判定已含1mm容差；不能把超出有效门的4.27微米描述为“只是浮点误差”或放宽容差抹去失败。
- 全部发送opening在[0,1]内；实际slewed targets在[0.042,0.044]m内；saved motor ctrl按原公式重构最大误差0。
- 失败时requested opening=0.0771693289，slewed target=0.0435200000m；右指仍以+0.00574600497m/s向外，motor ctrl=-0.589111887N向内。这支持**有界命令下的实际位置动态超调**；不证明唯一根因是模型急闭、缺阻尼、单位错误或接触。
- 最后20Hz观测在2.15s，尚未见违规。500Hz终态和native abort优先；采样score的safety_valid=true不能覆盖其后真实失败。
- R1两次PBS均终态，GPU累计890s，query reservations55（1次pre-forward失败+48离线+6在线）；新增训练0，全局M512不变。CoreLab charged CPU经sidecar后5240.044481548s；1start/1081steps/44RGB不变。实时账本如有后来修订以最新实际收据为准。

本轮审核未SSH、未下载私有highrate payload重算、未运行MuJoCo或模型；依据已公开小收据与固定源码。

### 已读源码的重要区别

`pi0294.scene.make_engine`继承`pi0291_arx5_data.engine.Engine.step`。原控制链已有finger目标限速0.04m/s，motor力限幅±8N，比例增益400N/m；MJCF的两个slide已有被动damping=3、armature=.002，积分器implicitfast、dt=.002。配置中的finger_kd字段不能仅因未出现在motor公式就认定系统没有阻尼。

机械臂力矩含qfrc_bias补偿，手指motor为限速目标的P误差力。手指重力/科氏偏置、臂加速度耦合、接触/限位反力、积分器与被动阻尼都值得检查。**偏置补偿差异是可检验假设，不是已经确诊的bug。**

MuJoCo的约束采用软约束力学；目标或ctrl限幅不构成实际关节位置的硬不变量。应保持原安全判定，通过可追溯的执行器修复/资格解决，而不是令qpos直接服从裁剪。

## 2. 本次任务应回答的问题

固定M512、数据、processor、统计和任务语义，回答：

1. 保存的合法EEF/夹爪命令为何在D0执行器触发右指越界？
2. 是否能用**至多一项最终采用的、任务无关的夹爪力控制修订**消除此类失败，同时保留抓放/推送能力？
3. 修订后M512在新观测闭环能否完成原四个开发case而无真实安全违规？

不以修复执行器的收益冒充神经训练收益；不接师兄WBC，不更换训练数据，不重训pristine，不根据确认集选择修复。

## 3. 不可修改的边界

原有效夹爪范围[-.001,.045]m、速度/力/接触/穿透/finite检查、500Hz检查频率与故障停止保持。原nominal range[0,.044]m、MJCF几何/质量/摩擦/被动阻尼/solver/integrator/dt默认保持。原SafetyGate、H40/prefix8、参考时钟、RGB和EEF10、夹爪0/1语义、M512权重及v3.0 stats均不变。

禁止直接钳制qpos/qvel、禁用limit、增大容差或力上限、改物体位姿/weld/隐藏外力、扩大合法工作空间、任务特定接管、Teacher替模型决定动作。不得用切成0/1、事后平滑或重定时掩盖模型原始输出；需要此类策略变化须另列研究干预。

有意更改的夹爪driver使用独立versioned D1，不就地修改/覆盖D0源码快照。Teacher与模型都使用相同D1，不允许只保护模型、对照仍走D0后混淆结果。

## 4. G0：复用保存证据，补足力学归因（先不占GPU）

复用已通过的saved-command算术审计，不再反复证明同一张FAIL表。沿初态、6个原始H40/prefix、44条command和1081高频状态，按准确时序绑定requested→slewed→motor→achieved。

对500Hz边界分别记录pre-step与post-step，避免用post-step状态解释前一个ctrl。区分真实保存字段与CPU重新构造字段。需要时用相同模型在保存状态上做零积分动力学计算，核对qfrc_bias、passive damping、actuator transmission/force、约束反力及惯性耦合；模型构造/forward另计CPU，不称纯文本检查。未保存warmstart/内部状态时，重算约束力不是原运行实测反力，不宣称逐项精确还原。

理论核对：M(q) qdd + bias = actuator + passive + external + constraint；qfrc_bias不包含M_fa*qdd_arm，不能把补上bias称为完全抵消臂加速度耦合。检查左右slide轴符号、actuator gear与有效力，不把ctrl符号直接当世界坐标方向。

输出`FORCE_BALANCE_AND_TIMING.json`，列出已证实、支持但未隔离、缺字段三类。若已有数据能否定某假设，不为凑矩阵重复运行。

## 5. G1：小规模CPU物理因果试验

**此阶段需要本专项被明确采用后才可执行。** CoreLab单worker，隔离诊断run，不再查询GR00T，不将重放当在线policy结果或训练数据。

预封存2×2对照，每项最多2.4sim秒/1200积分：

| 诊断 | 臂参考 | 夹爪参考 | 用途 |
|---|---|---|---|
| A | 原保存6个prefix | 原保存6个prefix | D0重现原限位失败 |
| B | 同一原臂参考 | 全程原始全开命令1 | 分离夹爪指令变化的贡献 |
| C | 由原臂controller保持初始TCP | 原夹爪参考 | 分离臂运动的贡献 |
| D | 由原controller保持初始TCP | 全开1 | 基准保持 |

同一初态、原几何、力/限位/碰撞守卫；不得weld臂或逐步写qpos。每项第一次native abort即结束并保留所有已经发生的物理步。不取消旧保护以观察“越界后会怎样”。基线预期FAIL仅证明重现，仍是安全FAIL，不进入修复PASS分母。

先恢复D0整段起点状态，不从2.15s凭不完整qpos/qvel拼接“相同”运行。若A不能重现，先报告恢复状态/观测暂停/调用次序差异；最多一次有明确工程原因的复现修正，不能重复挑到失败或成功。定位完成后再决定候选，不允许无限调参。

## 6. G2：仅在独立夹爪driver中做最小修复

最终只采用一个修订，最多2个预封存候选，不能组合多项改动追分。候选由G0/G1证据选择，而非固定断言“缺D”。

- 若姿态相关偏置力证据支持：优先单独试验finger generalized bias feedforward，转换到已核验motor transmission，再与原P项合并并受原±8N上限约束。原被动damping不变。验证符号/单位，记录饱和率及抓握稳态力；不得新增qfrc_applied外力。
- 若主要是欠阻尼响应：单独试验基于实际finger velocity的额外耗散反馈，增益根据该构型有效质量和已有阻尼计算，在运行前固定其值和上限。只增加motor层反馈，不同时增大kp/改MJCF阻尼/改步长。耗散项应满足单独功率不为正，不据此声称全系统已严格稳定。
- 若前两类证据均不足，或必须改变limit solver、碰撞/几何、步长、输入语义才能通过，停止该修改路线，报告精确理由；本建议不授予无界执行器重设计。

每候选最多6项4sim秒压力测试：满开保持、满闭保持、开→闭、闭→开、已知高端附近变化/50ms变化、原保存失败prefix加**显式标注的**终点保持。覆盖原失败臂运动或预封存可达姿态；不加入未知危险姿态。使用完整原数据与新安全裕量/跟踪误差/饱和率/左右差报告。候选在任一用例触发真实安全停止即淘汰，该run不重试；第二候选只能改变预登记的另一单项，最多两个，不叠加。测试设计本身无法区分根因时如实标诊断局限。

采用第一个通过全部压力测试且作用符合因果假设的候选，不按最终模型成功率调driver。之后封存`GRIPPER_DRIVER_D1.json`、allowlisted diff和测试矩阵。

## 7. G3：Teacher任务回归与统一高频安全裁决

在4个预登记已准入TRAIN场景组各跑T1/T2，共8条完整32s Teacher。确定组的规则：先包含原开发F0/000、F1/000，再取v3.0 F2中按ID最小的两组**双任务均已准入**组；在运行前固定实际card/hash，不看新结果换seed。Teacher算法/任务评分不变，只将执行器切为D1。

要求8/8持久成功，全部高频原安全检查通过；夹爪不是一直锁住、任务不是被额外力代做。若失败，不继续用同组调driver；报告该候选未取得任务回归资格。本阶段不要求重采224条或重新拟合stats，也不承诺D1与D0闭环分布等价。

端到端episode safety应当保守聚合：任何native_abort或高频安全flag或非有限状态即FAIL；覆盖不完整则UNKNOWN，不以最后一帧20Hz score覆盖终态。补充新的结算/审计wrapper及负例测试，不回写旧数据/旧评分文件。保存所有检查频率与首违规tick。完整任务成功必须同时满足原任务谓词、32s完整性和高频安全。

G0–G3通过只授予`GRIPPER_EXECUTOR_REQUALIFIED_D1`，不是P1通过，也不是多任务policy成功。

## 8. G4：冻结M512的新观测P1R，随后才谈续训

本专项被明确采用、G0–G3实际通过、原R1窗口/剩余PBS/GPU/物理额度都成立后，才申请剩余第三个PBS。若不足完成必要验证，交付D1资格，不猜测延长原窗口。复用已通过的神经数值/CPU接口，不重下25GB、不重跑16-step smoke，不在CoreLab或登录节点加载GR00T。

使用相同M512、v3.0 processor/stats与原固定随机性。新attempt namespace对原F0/000、F1/000各T1/T2执行四条全新32s闭环；原R1旧attempt的1FAIL/3NOT_RUN永久保留，不能续接失败前5个ACK。D1令观测分布发生变化后，必须每次重新获取真实观测，不能播放D0保存预测代替新模型查询。

与旧实际端点/协议相同且仅替换D1时复用Teacher双SSH资格；如果路由/协议本身有改动，先用最多2个Teacher episode重新资格，并计入预算。模型对任何真实安全违规立即停止该模型臂剩余case。SafetyGate在执行前拒绝计合法受保护的模型失败，但未完成四个完整case不能声称P1R已完成。

四个完整case零真实安全违规与正确model-route/readiness成立后，可记`M512_P1R_COMPLETED_SAFE_D1`，同时单报T1/T2任务成功；不强迫早期M512必须任务成功才允许原生优化继续。旧P1仍FAIL。原完整续训计划要求的其他门全部继续保留，不用安全关闭代替通过。

若用户采用本专项同时明确允许“通过G4后在原R1余额内接续”，则同一合法allocation内恢复原global512/consumed16384/optimizer/RNG/sampler，按原M2048→DEV8→M4096→DEV8→选型顺序继续。不切v3.1、不改scheduler6144/warmup308、不全模型微调、不扩6144、不额外加loss。M512/M2048/M4096在新的评价分支都使用同一个D1；结果标注新executor身份，与旧D0失败分开，不把驱动收益归给神经网络。独立确认保持未打开，直到原选型规则成立。

## 9. 提议预算与时间（采用前无执行授权）

这是R1范围内的**替换性安全修订**，不是新10h窗口或新增3PBS：
- R1 T0保留2026-10-14 10:57:30 SGT；new-work cutoff19:57:30 SGT；终截止20:57:30 SGT。全部阶段包含排队/断连；过期不能自动开始。
- G0–G3最多min(4h,原R1剩余new-work窗口)，CoreLab最多8 CPU core-hours、4核并发，最多2个候选；模型/PBS/GPU为0。所有消耗从R1现有40core-hours/96starts/1536000steps等剩余额度扣减，不叠加。
- G0–G4专项封顶32物理starts/512000积分/20512对RGB，最多4GiB新文件；其中4因果诊断+1有据修正、12压力测试、8Teacher回归、最多2路由Teacher、4模型case≤31starts。零积分模型构造也登记，运行次数按实际记录。
- G4至多用R1剩余1次PBS；最多1H200，GPU累计不超过原R1 8h，仍需保留评价与收尾时间。已消耗2PBS/890GPU秒/55query等不得清零。
- 原R1总查询5200、正式新增3584、隔离恢复4updates、CPU和磁盘上限仍有效。专项诊断与后续评价共同占用原预算；任何未用额度不是自动绕过P1的授权。
- 若原R1时间已不足或结束，必须明确新的时间和增量授权，再做新的子run。保留本R1最终实际数，不能修改旧硬编码时间以伪装原任务延续。

## 10. 交付与采用条件

在统一工作区经RUN_MANIFEST/host profile/lease进入，单一已有local-pc owner；不创建第二套目录/环境。不公开私有raw/WBC/mesh、无历史删除，无真机、无WBC运行。

必须交付：`FORCE_BALANCE_AND_TIMING.json`、`DIAGNOSTIC_MATRIX.json`、`DRIVER_DIFF_AND_SAFETY_INVARIANTS.json`、`STRESS_RESULTS.json`、`TEACHER_REGRESSION.json`、新`P1R_MODEL_RESULTS.json`（未执行则NOT_RUN）、新`RESOURCE_DELTA.json`与简明`FINAL_REPORT.md`。失败/被拒/未尝试分开，高频负例测试与实际执行证据分开。所有旧记录保留。

采用本专项需要用户明确批准：**允许在CoreLab做上述有界因果物理测试及独立夹爪driver的小范围修订，并在G0–G4通过后，仅在R1原剩余额度/原窗口内续接。** 单纯收到blocked状态报告、发布本文、CPU CI通过或仍有一张PBS额度，都不等于此批准。

## 固定来源

- R1实际终局：https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a174b107672937af2271012efe7e45f7cda0975/reports/PI0294/20261014T025730Z-localpc-pi0294-r1/FINAL_REPORT.md
- 已保存命令/状态算术：https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a174b107672937af2271012efe7e45f7cda0975/reports/PI0294/20261014T025730Z-localpc-pi0294-r1/SAVED_GRIPPER_TRACE_ACTUAL.json
- 当前物理继承：https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a174b107672937af2271012efe7e45f7cda0975/scripts/pi0294/scene.py
- 原step和检查：https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a174b107672937af2271012efe7e45f7cda0975/scripts/pi0291_arx5_data/engine.py
- 原MJCF：https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a174b107672937af2271012efe7e45f7cda0975/scripts/pi0291_arx5_data/scene.py
- 官方MuJoCo3.2.7动力学/软约束：https://mujoco.readthedocs.io/en/3.2.7/computation/index.html
- 官方joint/damping/actuator配置：https://mujoco.readthedocs.io/en/3.2.7/XMLreference.html
