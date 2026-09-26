# PI0247-R2 — 接受 L 收口，推进既有 U 与配对判定

日期：2026-09-26（Asia/Singapore）。

Decision: `GO_EXISTING_U_AND_PARALLEL_L_SCREEN_THEN_PAIRED_VERDICT`  
Goal identifier: `PI0247_R2_U_AND_PAIRED_VERDICT_GO`  
沿用科学授权：`PI0247_GENERATION_TO_SELECTED_SIM_GO`。  
沿用 U 执行授权：`PI0247_R1_MAINLINE_LU_CONTINUATION_GO`。  
本次审核 execution head：`1c323da7b587c1088b8938e3fcf71f06f82e04be`。  
Issue #11 canonical；PR #77 pointer only，保持 unmerged。

## 0. 决定：L 不重训，U 不重复提交，L 评估不必等 U

**接受已发布证据所支持的 L2000 训练完整性收口与两份完整 Trainer 归档。继续唯一已提交 U 作业 `633503.hopper-m-02` 或有证据的合法后继，同时在 Corelab 执行原预算内 L1000/L2000 的两噪声组 screen。U 完成后补齐对应 screen，按冻结的 final2000 联合门作 L/U 配对判定。**

这是原 PI0247 的执行续行，不是新配方。D1/D2、P 起点、R*=S4、两臂目标、系数、2000 步预算、原16窗口、B0/B1、模型/processor/精度以及全部下游条件保持。新 goal identifier 不要求改写已封存 U card 中的 execution_authorization，也不要求为发布文档而重新封卡、取消作业或重建运行目录。

L 已有效完成；此前给 U 设定的“L 达到训练完整性收口”条件已经满足。U 不以 L 的任务质量 PASS 为前提；L 即使先得到有效 FAIL，也不得取消已经批准的 U。反之，L 先 PASS 也不隐藏 U 对照，不提前启动扩量。

近期唯一科学交付：**P 起点 / L-final2000 / U-final2000 在相同实际输入与噪声下的完整生成质量比较**。不再用 prepared、HF 登录成功、训练 Exit0 或参数更新代替模型验收。

## 1. 审核来源、当前状态和证据边界

### 1.1 已读取的 canonical 与代码

- [进度 canonical 5845123482](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5845123482)。发布时间 2026-09-26 09:38:52 UTC，即17:38:52新加坡时间。
- [PR pointer 5845130086](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/77#issuecomment-5845130086)。
- [工程失败计算量与 L 重放授权 5844159421](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5844159421)。
- 原科学 release `50e66efe6ccfbe814d04caf12499493821be7957` 的 `PI0247_GENERATION_CONSISTENCY_TO_SELECTED_SIM_V1.md`、`PI0247_EXECUTION_BINDING_NOTES_V1.md`。
- R1 执行审核 `354bf97c95547e4110b29ee3d890f7a7d4cd6993` / canonical5842565868。
- 当前执行 head 下 `reports/evidence/PI0247_R1_PROGRESS_UPDATE_L_CLOSEOUT_U_SUBMITTED_20260926.json`，观察时间2026-09-26 09:37:12 UTC。
- `reports/evidence/PI0247_LU_FINALIZER_CLOSEOUT_633415_20260926/` 中的 L2000 HF readback、L2000 全projector比较、U capacity记录。
- 当前 `scripts/pi0247_u_rollout_overlay.py`、`scripts/pi0247_lu_train.py` 的保存/恢复/预检与收口相关代码、`hpc/pbs/pi0247_lu_train_h200.pbs`；finalizer修复commit `ebca5491c2c9a28e755d492c13794d95f315e988`。

GPT本轮只进行GitHub源码、证据与协议审核及发布，没有SSH/live qstat、模型/训练/仿真、项目测试、HF权重下载或独立远端逐文件重算。认可的是上述来源支持的状态，不冒称已亲自运行验证器。查询本次进度comment之后的Issue记录未返回后续执行更新；这不是实时scheduler证明。

### 1.2 最新已证实状态

| 对象 | 本次来源支持的状态 | 不可扩大为 |
|---|---|---|
| D1/D2 | 已完成，R*=S4，B0/B1均有效FAIL | 重新挑M4/S16、追加噪声或重跑D0的理由 |
| 633321 | 1000次真实更新后checkpoint写坏，工程attempt已封存 | 0计算量，或一个合格L checkpoint |
| L 633415 | 完成2000 updates与2000样本消费；COMPLETE update audit；只读收口PASS_TRAINING_INTEGRITY | 模型联合PASS |
| L归档 | 1000/2000各9个完整Trainer文件，固定HF revision readback已记录PASS | 独立泛化、完整部署processor都在9文件中 |
| U 633503 | 进度时点Q/small、Hold_Types=n，等待ngpus；updates0 | 此刻必然仍排队，或需要再次提交 |
| 模型筛查 | 该次进度报告新增screen queries0，L/U质量尚待测量 | 没有训练活动，或训练失败 |
| 下游 | selected_checkpoint=NONE；M64/FULL/CALIB/sim尚未进入 | 下游授权被撤销 |

L已完成的两份归档：

| step | bytes | file count | pinned revision | tree SHA256 |
|---|---:|---:|---|---|
| 1000 | 20882507448 | 9 | cbc5d2d9e99c19324af7668d5d82856de85a778a | 482258bbd4f7166437fe5432c78f45196bb9ee2e6284b3ea11325becf50ba9aa |
| 2000 | 20882646143 | 9 | c5a334cdb18196a4817c94b9d0ae005b2f01f527 | 3d475b7755c3cbb1a71a672d77667a47a7aa017e976d4f02bdfcb3e84ff7ea14 |

Repo：`leoyuer/pi0246-gate-pass-campaign`；实际prefix：`arms/PI0247/r1-replay-5844159421/L`。不要使用最初空的`arms/PI0247/L`替代该执行身份。

L run card：`54fb7423c1dceb2be918515a50b4d118ea2917a83e129ef0098840fd97c618e5`。
L training source：`161314e73d5a4b33e6c16c0a1e069cf64c4c3392`。
L只读manifest：`6ca1e106aa7afd4c8f16f2c8d010becf3c17c945be8ba70f86ca810e8b00e984`。
最终收口源码：`ebca5491c2c9a28e755d492c13794d95f315e988`，与训练时源码身份分别记录。

U parent card：`3555aadc9723f7b54f23e2c60882b407337d4b8f2f10c18c8128583708e927a1`。
U current card：`ad350ff4cdb9e8b77aa32d5a7cf13c0f797e68b271fcd6fe66704222ef8adcd4`。
发布记录将差分归类为FINALIZER_FIX_ONLY，规范化card摘要相同；实际运行仍须使用提交时的完整binding，不将规范化hash当作所有字段可忽略的许可。

## 2. 科学主线审核：没有新配方偏移，不再退回旧排查

当前原问题仍是：相同P-final2000初始化、相同数据和优化预算下，完整生成终点监督U，是否优于局部动作估计监督L。

D1没有达到规定收益的runtime，保持S4；D2两个噪声组都失败，进入L/U是原树预定路线。L本轮完成证明局部监督对照已经真实运行；不能把它叫旧PT重训，因为初始化及本轮设计不同。

本次未发现已报告执行中改变原loss、数据、sampler或gate的证据。此前路径、存储、finalizer故障消耗工程资源，但不应被当成hypothesis FAIL。另一方面，参数有更新不意味着自由生成达标；当前仍不能判定“U无效”“只是数据量太小”或“WBC不能追踪”。U尚无结果，当前也未执行WBC。

工程记录只在影响模型输入/输出、梯度、保存恢复、首次有效计数时阻断。不要以反复P1审计、整库哈希、旧参数探针或新增归档框架替代这次科学交付。

## 3. G0 — 接管既有U，保持运行身份

1. PC agent检查最新Issue、真实qstat、作业owner、执行节点、staged card/source与输出。633503的Q是过去观察，不据此双投。
2. 若同一U健康运行，保护它，不在目录中pull本计划、不修改staged代码、不因本次goal更新取消。若仍Q，保持唯一合法作业；本次不要求新的hold或新的U提交。
3. 若已terminal，读取实际updates、完整checkpoint、parity/gradient记录与错误类型后恢复缺项。只有明确已退出或不合规的同一owner才能替代；不得与原进程并行重跑。
4. L训练完整性与归档证据复用，U的L-admission已经成立。不要把L screen PASS作为U启动条件；不再重训L修任何日志/报告。
5. 既有U运行字段继续使用原科学和R1执行授权。本文件的新identifier是外层goal，不要求为了通过旧代码的常量检查而改写原card。

## 4. G1 — 在U排队/运行时立即完成L的固定screen

**本节不是新增查询，而是把原来尚未执行的L评估提前与U训练并行。**

### 4.1 输入和模型物化

Corelab先检查实际占用和原GPU推理环境；不同服务器可以一个U训练owner与一个L screen owner并行，每个profile仍只能有一个首次有效执行者。不得终止他人任务。

直接从已验证HF revisions向Corelab取得L1000/L2000推理视图；不经本地PC中转大权重，不把optimizer下载到Hopper或推理显存。逐文件绑定正确L checkpoint源inventory、step、family、card、source hash和实际路径。

这批9文件是完整Trainer保存范围，包含模型、optimizer、scheduler、RNG、TrainerState和training_args；不应因为早期家族有18文件而直接判不完整。部署所需processor/TRAIN statistics/embodiment配置仍单独从冻结的C_T及其已验收源绑定，不假定它们包含于9文件，也不从HF默认branch或其他family临时取配置。

现有readback复用旧helper，因此出现`status=PASS_PT_FULL_HF_READBACK`与`pi0247_status=PASS_PI0247_LU_FULL_HF_READBACK`并存。按schema、family、step、revision及逐文件内容核验，不能仅凭旧helper字符串把L识别为PT，也不能删掉身份检查。

### 4.2 固定范围和顺序

只执行：`L1000 × 原16 × B0/B1`，然后`L2000 × 原16 × B0/B1`，共64个原定新逻辑cells。

- 原P/D2基线32逻辑cells直接复用；不要再次生成P结果或重新生成B0/B1。
- B0是原D2对应每source实际S0 noise/condition packet；B1是原已保存N1并保留其跨source共享的说明。保存实际tensor身份，seed不是替代物。
- 同一冻结S4、H40/dt=.05、C_T、TRAIN stats、BF16评估方式、native sampler、decoder、SafetyGate和metric。
- 复用既有D2/paired-screen接口，仅增加L/U family与step映射及receipt读取的薄适配。模型名与目录substring不是唯一身份。
- 先在CPU验证profile分派、manifest映射、缺文件/错hash、重复/冲突记录。首个预定cell兼正式集成，不增加独立scored smoke。
- 模型真实REJECT/高误差保留并继续完整矩阵；非法算术不得重采样掩盖。只恢复明确工程失效且尚无有效结果的cell，first-valid不覆盖。

L1000只观察；仅L2000对应两组决定L科学状态。无论L好坏，U原配方仍不变，也不将较好的L更新成U initializer。允许先发布L完整结果，但campaign最终决策等U完整收口及其screen完成。

## 5. G2 — 完成U2000，验证的是真正四步生成终点监督

### 5.1 不变配置

`PI0247_U_ROLLOUT_M16_2K_V1`；同一原P-final2000 weights，不从L或旧PT起步；fresh AdamW/scheduler；原16、2000项schedule、每窗125次，seed22301；LR1e-5、WD1e-5、maxgrad1、batch1/accum1/workers0、cosine2000/warmup10；原trainable/frozen、precision、augmentation、stats。

`loss_U = L_native + 0.1*ell_pos(a_free_S4) + 0.1*ell_delta(a_free_S4)`。

物理目标与L完全采用原定义：position按.05m归一化；相邻残差按.05s和1m/s归一化；XYZ平方求和后按有效点/transition平均，不额外除3。原scale-aware decoder与clip不变。

### 5.2 已修正的实现直接复用，不重新开工程大审计

当前U源已包含：保持dtype的byte-view hash、从native epsilon到stock入口dtype/device的显式映射、decode统计张量迁移、原生S4与可微loop的逐步/全动作/gate parity、四步autograd、eval模式下的生成及mode恢复。

但这些代码存在及CPU toy PASS，不等于真实模型已完成parity或真实U已经学习。它们须在原非选分parity预算与首个正式更新里实际验证，已有同身份证据则复用。

必须记录：
- 同一condition/stock_noise下原生S4参考与U loop的逐step输入、velocity、raw最终输出、完整decode与原gate decision/issues；原数值界不变。
- 纯噪声起点、label-free观测、无expert prefix/RTC；a_star仅用于native监督和终点loss，不成为循环状态。
- 原noise/tau捕获与转换前后身份，无额外全局随机抽样；raw与processed-once特征不重复处理。
- 四个Euler state的有限有效梯度、真实参数更新、native/U模式恢复；不得用requires_grad标记本身替代反向证据。
- native/position/delta/total、tau、saturation、实际rollout/head/recompute与optimizer计数。期望计数与实测计数分列，恢复不能只用常数补出“观测”。

原非选分kernel parity总额仍为8完整rollouts，先核对已经消耗量。当前U参考/候选成对验证从余额扣除，不是新实验额度。首个真实U更新计入2000；没有额外项目模型梯度smoke。

如果parity、梯度实现或输入绑定错误，修工程并保留失败证据，错误实现不能作为H3科学负结果。合法完整计算图确实无法容纳，按原`ROLLOUT_GRADIENT_CAPACITY_BLOCKED`停止；不能detach早期Euler步、降低N/H40或修改optimizer精度冒充原U。

### 5.3 保存、恢复和收口

完整保存1000/2000，通过当前持久工作盘→固定revision HF→逐文件readback路径形成完整收据。原L已证明的成功链复用，不重新引入tmpfs唯一源或登录节点重计算。

有同轨迹完整1000点时，按原受测exact resume恢复optimizer、scheduler、RNG和实际样本cursor；不得fresh optimizer接着训练并称恢复。完成2000后若只是finalizer或HF问题，只修传输/只读收口，不再调用训练入口。

若遇到新的无有效恢复点且已经发生updates的故障，保存全部原始计算与失败点；本文件不额外给予“看到负结果就从头重跑”的许可。沿原失败类型规则处理，若需要超出现有明确恢复授权，报告准确缺口，不能把先前仅针对633321的重放授权无限套用到其他科学轨迹。

最终来源应同时分开记录训练源码、card、只读finalizer源码与收据。frame/decoder/目标边界冲突不是小修可悄悄改变；已发生的科学有效结果保持。

## 6. 存储与资源：防止U再次在保存点中断

最新汇总观察为可用25,715,277,824 bytes，当前门槛25,184,706,243 bytes，超门槛约530,571,581 bytes（约0.49 GiB）。独立capacity文件较早观察为25,716,326,400 bytes；两次时间点不同，不能混成同一测量。当前实现门槛是约20.89GB的完整Trainer参考量加4GiB储备；不是总共只剩0.5GiB可写，也不是已证明用户quota充足。

本轮不重新制定容量数字门。agent应复用已有容量方法，并在实际计算节点、写入前及关键归档阶段复查同fileset实际余量和新增占用。日志/临时文件/并行下载/重复final模型副本均应纳入生命周期；单份滚动归档只有在1000点已经可靠持久化且本地无需继续保留后，才有后续2000点所需余量。

- Corelab L screen不要往U所在Hopper fileset复制L的20.9GB训练包。恢复或推理所需源从HF直接到对应执行服务器；本地PC仅控制。
- 不反复执行大体积写盘探针，不扫描整个历史目录寻找可删文件；不扩大现有删除权限。旧P initializer、数据和证据保留。
- 若前置容量因新占用不足，恢复合法空间/路径后继续；不通过改loss/精度/保存科学step来适配。路径mechanics变化须有子binding，不能改在跑文件。
- checkpoint的“安全保留”与“已回收”分开。当前finalizer的退休检查是滚动存储策略，不是科学质量门；若内容已可靠归档但退休状态/sidecar记录有问题，应只读核实并保留证据，不能驱动重训。
- 原full Trainer收据通过后才可退役被授权的冗余副本；screen只读取内容核验的推理视图，不伪称它是exact-resume源。
- H200用于已批准U完整训练；PBS auto合规分配，不要求某hostname。Corelab无梯度screen优先，不能因能推理就重复已知4080S全训练OOM。普通pytest/路径修复不占H200。

## 7. G3 — 配对screen收口和唯一晋级判定

U两份checkpoint就绪后，只补：`U1000/U2000 × 原16 × B0/B1 =64`逻辑cells。可逐份固定checkpoint归档后评估，但中期成绩不改变训练终点或recipe。与L共128个新逻辑cells，加原D2共享起点32，完整表160。

独立CPU复核从已封存数组重算：完整唯一source集合、checkpoint/processor/noise identity、raw/decoded形状与finite、gate decisions/issues、XYZ/SO3 RMSE、速度。没有新增model query，也不以手填summary代替原始结果。

每臂只final2000，每个noise组单独满足：

| 条件 | 原门槛 |
|---|---:|
| valid原窗口 | 16/16、原H40全部点及原mask |
| SafetyGate | REJECT=0；WARN保留 |
| median逐窗XYZ RMSE | ≤0.020m |
| linear p95逐窗XYZ RMSE | ≤0.035m |
| median逐窗SO3 geodesic RMSE | ≤0.050rad |

不得pool两组、挑较好noise、用1000代2000、删尾段、换失败窗口或改gate。B0/B1是同16TRAIN支持，不是独立holdout；单训练seed的配对比较不升级成普遍因果证明。

最少结果表：P起点/L1000/L2000/U1000/U2000，每个B0/B1各一行；主对比为L2000−P、U2000−P、U2000−L的同source/noise差值。报告精度、全部gate原因、峰值线/角速度、尾部SSE、bias和saturation。已有trace可解释残差，不新增solver/noise诊断。

决策：
1. 两臂都有效完成，只有一个两组联合PASS：锁定该配方。
2. 两者都PASS：沿原排序B0 p95、B1 p95、B0 medianXYZ、较简单L，选唯一fit recipe。
3. 两者均有效FAIL：`PI0247_M16_HYPOTHESIS_SET_EXHAUSTED`，保留结果，停止扩量，不加lambda/seed/步数。
4. U实现或资源未有效完成：标明确typed blocker，不写成U科学FAIL，也不把L已有成绩当作双臂完成。是否改变研究设计须另经PI。

L/U改善可支持所测目标的作用；两者都FAIL也不能无证据断言数据量、WBC或视觉容量是唯一原因。下一假设只从已得证据提出，不能在本goal中自动执行。

## 8. G4 — 原通往selected checkpoint与sim的条件授权继续有效

本文件不另加一次例行PI批准门。第7节得到唯一合格fit recipe后，按原PI0247主协议与绑定说明自动继续：

- **M64/5000**：从合格final权重开始，新optimizer/scheduler，继承该loss/S4；LR1e-5，原64窗口和旧16保持，两noise组各过门；完整1000/2500/5000及原screen范围。
- **FULL9792/10000**：M64通过才执行；继承loss/S4/LR1e-5及原schedule；完整2500/5000/7500/10000，保留既定1000/2500失稳barrier与所有候选的TRAIN检查。
- **CALIB/S3/S4/S5**：FULL family完整后才读原1088CALIB×baseline及四候选；原admissibility/ranking函数不变；没有admissible就保留NONE，不选最不差。
- **selected**：只在原正式S4选模成功后记录唯一FULL checkpoint+runtime+processor+完整来源。runtime_id=S4与selection_stage=S4字段分开；M16合格不等于selected。
- **sim**：winner自身对应TRAIN/原16保持门及原sim binding齐全后，自动frozen-WBC A–D C0 one-chunk，4/4通过再progress-preserving receding；真实world-TCP、0.035m任务成功半径、原clock/progress/termination规则不变。不依据sim成绩换winner。

M64/FULL/CALIB及sim的全部预算、两noise细节、原sim cap与停止点以原release第7–12节为准，本文件不追加查询。只做适时CPU准备，不为远期工具整理阻塞U或L当前screen。

DEV-E/F、Final-F、DAgger、真机、WBC权重/原gate/scorer修改和PR merge继续锁定。即使最终TRAIN sim PASS，也不称已经独立验证泛化。

## 9. 计算记账与交付

当前PI0247 L的科学轨迹为633415的2000更新；633321额外1000更新永久留在工程attempt/raw compute。因此目前L实际已执行3000 updates，不是2000也不是零。U成功2000后，本轮L/U canonical合计4000、包含该已知工程损耗的实际量至少5000；未来若还有实际计算必须继续累加。

PI0246旧8000更新/208cells不重新使用。PI0247总科学上界仍为19000 updates、原离线query6696和原parity8；已经用过的D1/D2不充值。L/U新增主screen仍128，D2起点32复用，不因提前L screen增加预算。U每update原生forward、生成rollout、内部4 head及反向重算分别记录。

Issue #11追加阶段性canonical，PR #77只pointer。不要改写旧授权正文、删除旧FAIL或重新生成旧hash。用机器可读manifest生成汇总表，保留观察时间及未知状态。这里没有新定时monitor授权。

本轮近期完成条件：L原64 screen已完成 + U2000实际训练/归档有效完成 + U原64 screen已完成 + 独立配对判定。成功者实际开始原M64或报告准确外部前置条件；有效双FAIL到原终点。不能仅交付“仍排队”“又一次自检通过”。

## 10. PC agent 启动正文

执行 `PI0247_R2_U_AND_PAIRED_VERDICT_GO`，沿用原科学`PI0247_GENERATION_TO_SELECTED_SIM_GO`与U运行字段中的`PI0247_R1_MAINLINE_LU_CONTINUATION_GO`。完整读本文件、原release两文件、R1审核及5844159421/5845123482。

L633415已2000且训练完整性收口、两份HF归档已通过。原Exit1是finalizer，禁止重训L。633321的1000工程更新保留raw accounting。D1/D2不重跑，R*=S4不改。

先查U633503真实状态与提交身份。健康执行保持，不因新文档hold/换card/重复qsub。L完整性已满足U准入，U不等L质量PASS。U仍从原P-final2000、同16/2000表、新optimizer、LR1e-5，完整纯噪声S4终点监督，四步全梯度；不是从L起步。

现在Corelab并行物化L1000/L2000并做两noise原64cells。使用本次r1-replay-5844159421/L的真实HF prefix和固定revision；processor/stats另按冻结C_T绑定。D2的P起点32复用，B0/B1实际packet不重建。不要把L完整训练包拷到U的Hopper fileset或本机PC。

U首次实际执行闭合原S4 parity、gate/issues、dtype/device、四步梯度与mode/RNG证据，计入原parity与训练预算；代码已有则直接复用，不扩诊断矩阵。若真正的工程异常，最小修复/回归/恢复；错误实现不能记科学FAIL，不能截断梯度适配显存。

持续保护单份完整checkpoint滚动保存容量；当前超预检门约0.5GiB是易变化的余量，不是quota保证。固定revision读回后才退役允许的副本；保存/归档/报告错误不重训。完成U2000后补其64cells，独立CPU复算，final2000、两noise分别原四门。

两臂都完成后：原规则选唯一合格fit recipe，自动原M64→FULL→原CALIB selector→winner TRAIN准入→原frozen-WBC one-chunk/receding。两臂有效FAIL停PI0247_M16_HYPOTHESIS_SET_EXHAUSTED，不加新配方。正式FULL选择前selected_checkpoint=NONE。

工程自主修复无任意次数上限，科学语义/预算不变。Issue #11追加canonical，PR #77只pointer，DEV/F/DAgger/真机/merge保持锁定，不新建monitor。本计划只发布审核与执行范围，GPT没有代你查询scheduler或启动作业。
