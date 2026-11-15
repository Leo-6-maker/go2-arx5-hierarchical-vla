# SLA 后训练推进 V6.1：最终专家计划、双 mask 与完整部署回放

日期：2026-11-14，Asia/Singapore。状态：**计划与验收规格；不是运行代码、现场 job 配置或已完成实验。**

配套：[机读矩阵](SLA_ACTION_CONTRACT_REPLAY_V6_1_20261114.json) · [执行交接](../handover/2026-11-14/SLA_ACTION_CONTRACT_V6_1_START.md)。继承 [V5 长程计划](SLA_LONG_HORIZON_EVIDENCE_V5_20261113.md)，本补充只调整未启动工作的语义、依赖与验收，不修改历史协议。

## 0. 本轮决定与交付顺序

用户已明确四个边界，本计划将其作为后续实现合同：

1. `EEF_STREAM_V2` 的轨迹重构属于**专家规划器**；计划监督取重构完成后的最终 H40。学生 raw 输出不再次经过该规划器。两者之后复用同一参考变换、校验、安装与执行适配器。
2. `supervision_mask` 与 `consumed_mask` 分开。经过验证的未执行计划后缀可作为独立计划蒸馏的监督，但不能计入实际消费、成功示范的物理证据或状态转移标签。
3. 顺序是 **统一物理语义 → 拟合并冻结 TRAIN 统计 → 完整编码/FP32/解码/部署入口与原生回放 → 训练**。归一化前物理数组检查只是预检查。
4. 在途 V6 参考等价标签路线独立继续。不得更改其数据、统计、输入、损失、模型起点、consumer、学习率、评价人口或计划监督定义。

近期交付优先级：**标签经学生部署链原生回放 → 覆盖阶段与接缝的16窗口可执行拟合 → 在途V6原定终点/任务评价 → 同数据空间监督 → 条件性计划蒸馏 → LeRobot双reader一致性**。CPU准备、采集和已有健康训练可并行；上述顺序是新研究的科学依赖，不要求暂停在途作业等待新检查。

本轮文档继续正常追加到计划 PR #383；执行依据为 PR #384 及后继。只发布文档，不启动作业、不合并执行栈、不变更服务器权限。

## 1. 核验基线与不能混称的结果

核验源码：`4273dcdf143a0f9bac28afc5f831eb597a240412`；V6训练快照 **2026-11-14 12:25:57 SGT**，报告观察时间12:26:22，旧N20原生终态12:07:35。[E1]

| 项目 | 固定快照事实 | 本计划如何使用 |
|---|---|---|
| 旧 N20 | 20 episodes/1220 windows，3K/96K T；TRAIN8为0成功、7策略拒绝、1技术失败 | 保留负对照，不重标数据或重跑失败筛选成功 |
| V6标签 | 1220/1220精确参考及1220/1220 FP32解码标签通过原H40包络；最大参考重建误差2.3315e-15 | 复用此静态证据；不是完整原生回放的替代 |
| V6新候选 | 92更新/2944 T访问；原87初始状态数组与旧N20相同；3K固定预算 | 接手先读最新终态，不重复提交或热改 |
| V6统计 | 当前state统计不变，action TRAIN统计重拟合；新数据manifest `29fd45ff31a2031f70f5d32402e6bfefe2f8b644e8e81ad1d2003d4537a1a6f5` | 作为固定回放与拟合输入，不用16窗口另算一套统计 |
| 专家数据 | 78条完整，正式五物体/两任务D100/D300未齐；paced PLACE仍为失败 | 分层继续采集，保持失败与原任务判据 |
| 旧2EP空间臂 | 2857累计更新，不同于N20数据 | 不能用于同数据FM/SF效果判断 |

这是GitHub证据，不是本次SSH读取。运行状态可能已有后继；未知作业先查，不推定停机或结束。保留旧S1、R0/R1、GR00T等真实结果，raw与endpoint-basis consumer结果不直接比较为同一学习曲线。

## 2. 专家规划与共同执行的边界

### 2.1 明确四层对象

```text
专家任务意图 / Teacher proposal
      → EEF_STREAM_V2及已版本化专家几何/节奏规划
      → expert_final_H40（本query最终计划，冻结身份）
                                                 ┐
SLA预测 → 反归一化/夹爪解码 → student_raw_H40       ├→ 共同执行适配器 → 冻结WBC
标签回放 → 同一codec/统计/FP32路径 → oracle_H40    ┘
```

规划器可利用专家真值、历史与上一夹爪命令；这些不自动进入学生输入。共同执行适配器只接收规范化物理动作、当前实测TCP、当前已消费参考、实际base及原控制历史，不按 `source_kind=expert/student` 偷换轨迹。

当前实现 `eef_stream_v2.expert_program` 从proposal末端目标重建分段平移/旋转/夹爪，并返回一个 `FiftyMsProgram`；学生 `sla_dense_policy_v5.compile_proposal` 保留40点并调用同一类。[E2][E3] 本轮先在**当前函数返回位置**记录最终 `program.decoded_H40`、`reference_path` 和planner身份，不为记录字段重构在途执行代码。后续冷版本确需提取纯规划函数时，使用同一输入逐点和安装参考等价测试，不能改变数值路径后仍沿旧版本号。

共用原 `FiftyMsProgram`／原生certifier/安装/TTL续期/P10消费链，优先复用函数而非复制“近似同一实现”。frame映射一次，sigmoid一次，机械夹爪映射一次。标签注入同学生入口，只替换模型产生动作的环节，不加载任务Teacher帮学生纠错。学生路径禁止再运行 `EEF_STREAM_V2`、endpoint basis、裁剪、减速或隐式平滑。

`EEF_STREAM_V2` 的具体分段设计不是通用任务的永久定理；开门/插书需要曲线型专家时，新planner也必须输出同一明确接口并有自己身份，不将当前直线/分段规划器套给学生。

### 2.2 参考等价公式及边界

设当前实际位姿为 `(p_A,R_A)`，当前参考为 `(p_0,R_0)`，已安装参考为 `(p_i,R_i)`。现有安装关系为：

```text
p_install = p_0 + p_nominal - p_A
R_install = R_0 @ R_A.T @ R_nominal
```

V6标签使用逆表达：

```text
p_label = p_A + p_i - p_0
R_label = R_A @ R_0.T @ R_i
```

之后才按当前实际TCP进行SLA相对位置/SO(3)编码。保留原时间和夹爪nominal，不用未来achieved状态替换命令。[E4]

V6的第一命令恰等于当前reference依赖其已检查前提；新计划记录必须显式保存 `reference_before_query`，不能无条件以 `installed[0]` 代替所有来源的query前参考。安装轨迹逐点一致也不意味着整段未来预览与原分段重规划一致；需要另核对原生预览/历史与执行前缀。

## 3. 两条监督路线严格隔离

| 类型 | 监督目标 | 数据身份 | 本轮位置 |
|---|---|---|---|
| `EXECUTED_REFERENCE_V6` | 未来真实消费参考在一个当前锚点中的等价表达 | 在途V6数据/统计/模型不变 | 主线验收 |
| `EXPERT_FINAL_PLAN_PD1` | 当前query下最终专家H40，含可验证但未执行的后缀 | 新schema、新manifest、新统计、新训练run | 条件性独立研究 |

旧N20依旧是旧nominal拼接语义，不通过新增metadata“升级”为V6/PD1。禁止把缺失的旧最终计划，按当前版本planner离线重算后冒称历史计划。若完整最终计划、query输入、planner版本和hash齐全，允许以新视图引用原数据；否则新采。

### 3.1 最小query记录

记录下列真实字段，实施时复用已有名称并在schema映射中固定：

- 身份：`episode_id/query_id/split/planner_version/adapter_version/codec_version/normalizer_sha`。
- 观察：两相机帧ID及真实时间/native_step、实际state、语言；expert阶段只作组织和评价。
- 边界：`actual_before_query/reference_before_query/base_before_query/previous_nominal_grip`，以及可用的参考历史与控制状态身份。
- 三层动作：`expert_proposal_H40`、`expert_final_H40`、`installed_reference_H40`；另保留实际achieved状态。学生回报相同的raw/installed/achieved区分。
- 时间：40个真实命令开始tick、20Hz/0.05s、P10执行前缀、重规划或取消时刻、原生2ms积分。
- 双mask与证据：`supervision_mask[40]`、`consumed_mask[40]`、逐点origin query/command ID、planner验证证据、安装/消费回执、episode任务终态。

所有旧原件只读；新增mask使用新sidecar/manifest，不能原地更改V6训练schema。实际query首次提交时写计划快照，关闭后用追加事件生成消费mask，最终封存版本不能再修改。

### 3.2 mask的精确定义

`supervision_mask[k]=1`：该点是本数据定义允许的专家目标，来源/单位/时间/最终计划检查通过。`consumed_mask[k]=1`：该**原query、原command ID、原物理区间**有完整的原native solver及夹爪消费确认；发送、安装ACK或TTL续期均不够。

| 样本 | supervision_mask | consumed_mask | 可得结论 |
|---|---|---|---|
| V6完整参考窗口（新sidecar） | 40个true | 40个true，允许原query ID不同且保留映射 | 40点历史参考曾被消费，不等于一次H40全开环执行 |
| 验证通过的最终计划，仅执行P10后重规划 | 40个true | 前10个true，其余false | 40点可监督，只有10点实际消费 |
| 合法计划中途在第6个完整atom后失败 | 按既定计划准入，首轮允许计划40点 | 6个true，未确认点false | 后缀只是计划；该episode仍失败，首轮PD生产不以它算完整成功示范 |
| 在planner/完整H40检查被拒绝 | 全false，记录但不训练 | 全false | 不是有效计划样本 |
| 目标缺失／未知 | 相应false | 依实际独立记录 | 不用补零伪造目标 |

原query未执行的点，后来别的query恰好给出相同数值，也不能回填原query的consumed_mask。多次续期ACK不增加已消费点数；物理积分推进但缺少消费确认的点保持false并另记不确定状态。成功率按完整物理episode统计，mask点数不是新episode数或成功率分母。

首轮PD1仅采用完整专家episode中、完整H40验证通过的最终计划，阶段/安全/物体成功仍独立记录。来自失败episode的合法局部计划可另建恢复/局部计划池，不在看到模型结果后临时调整准入。

### 3.3 训练不能只“存mask不使用”

现有官方 `Pi0Align.compute_align_losses` 对整个动作张量做mean，没有自动读取我们新字段。[O2] 因此PD1的部分监督mask属于新实现，旧V6训练不热改。

新分支按样本计算：

```text
L_FM(i) = sum_k supervision_mask[i,k] * sum_c(error[i,k,c]^2)
          / (32 * sum_k supervision_mask[i,k])
L_FM = mean_i L_FM(i)
```

全true时必须与原FM32归约/梯度一致；保留原32维学习定义，有效7维/padding25维只作分项报告，不借此次改mask把loss改成FM7。`consumed_mask`只进入证据与报表，不乘训练权重。图像padding/camera mask是另外的字段，不共用。

全false样本在数据入口排除，不能NaN除零。无效目标在noising前清洗且阻止其通过attention泄露进有效点；用“改变全部未监督target，合法部分forward/gradient仍不变”的定点测试验证。不能只将最终loss乘零、却让无效标签参与flow输入。首轮PD1实际优先使用全40点supervision=true，因此不必先将各种部分监督策略展开为新研究。

## 4. 统计与完整验收顺序

### 4.1 对新数据版本执行一次的顺序

1. 按episode与布局划分TRAIN/DEV/TEST，固定数据语义、TCP、frame、时间、mask与有效源。允许结构/来源预检查，不能根据学生结果选择样本。
2. 从唯一合格TRAIN窗口/目标点拟合统计：拟合输入必须先经过同一物理→相对位置/SO(3)/夹爪logit预变换，尚不做归一化；不能对世界坐标拟合统计后却拿来归一化latent。重复采样访问不重复计为独立数据。拟合统计使用supervision_mask，不用consumed_mask把计划后缀丢掉。记录加权/分位数实现与异常维度处理。
3. 冻结统计及样本清单的hash。若需改变统计，生成新版本、回到本步骤后重验；不得给在途checkpoint换normalizer。
4. 原始物理标签 → 同一数据adapter与相对动作编码 → 夹爪logit → 冻结统计归一化 → pad32/FP32真实序列化 → 部署反归一化 → sigmoid一次 → frame/TCP转换 → 原 `FiftyMsProgram`/完整native certifier。
5. 全集离线回放检查各阶段和拼接点；之后做选定实际原生回放。只有成功重建参考并经相同部署路径才进入新训练验收。

V6已有1220窗口、固定统计与FP32证明直接复用；新补的是部署封装与原生历史/消费，不重新拟合一次。以当前codec常量作为数值容差；不得新开一个更松容差让检查过关。比较XYZ、SO(3)角距、nominal grip与机械宽度，不用全部值一律浮点bit相等；对同精度离散元数据与身份要求exact。

### 4.2 必须覆盖的易错点

H40开始点与上一个已消费参考；9/10、19/20、29/30接缝；夹爪开闭事件；SO(3)左右乘；row-packed rot6d；相机顺序/missing mask；真实时间与episode尾部。全40点预览必须保留，未监督mask不能隐藏不合法输出后缀。参考TTL0.4s、续期≤0.35s、P10/2ms安全沿既有协议，不因录制或等待变化。

## 5. 第一项实际交付：ORACLE通过学生部署链的原生回放

### 5.1 离线重用与S16选择

在修正N20的1220窗口固定TRAIN集合上，仅补尚未完成的序列化/部署入口等价检查。冻结 `S16` 诊断子集：5阶段各2个不同窗口共10个，另6个跨重规划/事件接缝窗口；不得重复，尽量覆盖≥4个episode及4个布局。选择规则在模型输出前按episode/native_step固定，缺项明确报告，不拿HOLD补齐DESCEND。

S16使用**N20全TRAIN的已冻结统计**，不是以16窗口再算小normalizer。保留旧标签检查的545/1220作为历史，不删除不合法旧样本使新数据看似更好。

### 5.2 原生RP8（新增诊断，不替代V6正式TRAIN8）

固定8个oracle条件：PREGRASP/DESCEND/CLOSE/LIFT/HOLD各1个，加3个重规划接缝条件。每个条件来自一个已选TRAIN记录，独立恢复/重放到其anchor，再通过学生动作文件/同一codec/同一raw compiler注入**已冻结统计经过FP32往返的标签**。所有40点检查，实际只消费原P10；不把oracle当学生。

恢复必须包含必要的MuJoCo、WBC/solver、上一参考、夹爪和随机状态，不能只写qpos/qvel然后声称状态相同。若完整hidden state不能安全恢复，则用原专家前缀真实重放，并在anchor比较实际/参考/历史与记录，原安全全开。超出事先绑定的现有状态匹配容差，标为state-reconstruction不成立，不偷偷重锚、瞬移或反复筛选成功。原native所有权和登记复用；同条件最多一次，修复后另版本诊断。

记录：40点期望/重建参考、证书、10个预期消费区间、独立确认为真点数、跟踪与支撑、首失败及所有初始化成本。预览完整H40不等于后30点动态执行；不能为证明后缀临时把P10改P40。

RP8仅是“标签→学生入口→控制”的诊断。前缀靠专家到达anchor、标签包含离线未来命令时，整段结果不能作为自主策略成功。若8条件都通过，再进入小拟合；不通过则修复具体转换/历史/预览问题，不加数据或训练预算掩盖。

## 6. 第二项实际交付：S16有限可执行拟合

新具名诊断 `MICROFIT_S16_REFERENCE500`，不是继续V6：同SLA_base与既有LoRA/接口白名单，固定S16与N20统计，FM-only，500更新/B32=16000窗口访问。每批16窗口各2个独立已绑定noise draw，保持原state-token dropout .25、highest精度、micro8×4、既有500步/25warmup配方；无新输入、空间loss、EMA、滤波或平滑惩罚。

不抢占V6/SF/BOOK健康GPU。可在空闲已授权GPU运行，或在原任务冷结束后单卡运行。保存0/100/500及完整Adam/RNG；0、100只做已登记离线诊断，固定500是终点，不按结果重选。

固定16窗口×2推理noise=32条普通RGB预测，报告归一化loss、active7/pad25、XYZ RMSE、SO(3)、夹爪、完整H40准入率、接缝加速度，以及拒绝发生在哪个阶段。工程目标：32/32满足原包络，XYZ RMSE≤2mm、SO(3)均值≤.01rad、grip MAE≤.02；这些是诊断目标，不修改安全阈值、任务分数或保证成绩。只有小拟合证明模型可学，不证明泛化。

只有小拟合可执行性成立后，做固定4个TRAIN布局的一次完整学生任务诊断（4请求），全过程由模型决定，不在中途用标签/Teacher接管。任何结果保留；不通过时不循环同一500训练。若oracle过而模型不过，区分拟合不足、预测噪声/时间曲率、token处理、action/projection范围；空间监督和开源数据都不是自动补救。

本段新增工作：500更新/16000访问，RP8＋学生TRAIN4合计12个原生诊断条件。已有V6的3K/TRAIN8/DEV48是另外的现役工作，不重复计入或重跑。

## 7. 在途V6与后继研究如何衔接

V6按原3K/96K、保存/新进程RGB16、原TRAIN8/独立DEV48完成。新计划不把RP8或S16追加成中间checkpoint选择条件；不往V6输入添加reference error/history，不转成计划蒸馏，不改consumer。

V6原生验收和RP8共同支持参考标签路线后，使用同一修正数据集、父权重、普通初值、访问序列、统计、consumer做匹配SF3K。若现有V6 FM端点身份完全一致，直接复用，不再训一个重复FM。旧2EP空间臂按原计划收尾，但不与20EP V6当成匹配对照。新SF只补该数据像素/cache与空间梯度接缝，不反复重做全部训练资格。

## 8. 条件性计划蒸馏PD1：先小批记录，不替换V6

### 8.1 PD20数据

专家final-plan身份与共享adapter测试完成后，可在独立已授权世界采第一批4条完整episode，检查记录，再扩到20条（8个TRAIN布局，配额3/3/3/3/2/2/2/2；最多40次有界尝试，所有失败保留）。首轮只用已能完成的PICK_HOLD和当前物体，不冒称正式D100/多任务。

同一次新采集同时导出两个有名视图：`PD20_EXECUTED_REF` 与 `PD20_FINAL_PLAN`，共享RGB/state/query anchors/episode split。前者使用未来真实消费参考的当前锚点表达，后者使用当时已验证的最终专家H40；除标签及对应统计绑定之外不更改数据/训练设置。共同锚点必须有完整可用标签；未执行后缀不得借下一query的命令补齐计划。

这两个新视图必须另算各自正确的TRAIN统计，分别完成FP32＋同部署链验收。统计是表示变换的一部分，比较结论应称“监督表示＋相应预处理效果”，不能称纯loss变量。若后续要共享统计，另登记合法的共同TRAIN scaling实验。

### 8.2 有限监督表示对照

PD20若ready，先2臂×500/B32=32000访问，以同父模型/LoRA、初始数组/visit IDs/noise/camera/student inputs、consumer和计算预算比较。两臂都FM-only，不同时加空间loss/历史/额外输入。首轮评估32条相同anchor/noise预测和各TRAIN4，共8个学生原生条件；计划样本未消费后缀只影响supervision，不进入成功计数。

只有可执行性和任务证据支持，才另登记更长或大数据对照。不能拿PD20新观察与旧N20不同run的结果直接声称计划蒸馏增益；需要上面的同episode两视图。

部分监督mask由固定训练实现处理；旧V6保持不变。mask机制的全true与旧梯度一致性及“未监督target不影响有效梯度”的接缝测试最多新增8次资格更新/B32=256访问，单列非候选成本，不重开泛化实验。

## 9. LeRobot兼容随后做，以相同模型输入为准

优先参考已锁定SLA上游的README、config、data transforms、mask和loss，使用其`pyproject.toml`指定的LeRobot revision；不要直接升级最新数据格式。[O1][O2]

保留现有高效分片reader，增加只读兼容视图/导出。两条路径在同一S16上比较2个batch配置（1与8）×2次读取，每条reader共64次样本读取，两条合计128次：实际RGB与camera/padding mask、state/语言tokens、H40×32归一化标签、supervision/consumed/identity必须一致。采用相同已解码RGB或lossless编码；有损视频不宣称bit-exact。离散张量/原像素exact，FP32动作使用绑定容差，最终部署约束仍必须通过。

10Hz图像、20Hz动作、2Hz采样锚点与2ms物理时钟分开存。默认LeRobot行索引不能把0.5s锚点步长误作0.05s动作步长；单FPS容器表达不全时保留sidecar时间映射与明确custom transform，不复制帧冒称新观测、不合成未来状态、不跨episode自动延长动作。闭环以后物理回放仍是最终检查。

官方`compute_norm_stats`不一定认识新supervision_mask；需明确适配，保持与高效reader相同的唯一TRAIN样本、权重和归一化结果。mask必须被collate到GPU并真实作用于相应新loss，不能仅写在metadata；consumed_mask不变成学生输入。

此工作不能阻塞RP8、S16或V6交付。公共源包语义审计沿V5继续，默认先保留不同来源标签，不将官方格式兼容等同于目标执行器兼容。

## 10. 代码实施清单与回归范围

以下为**拟新增文件**，不是已经存在或可直接运行的CLI。接手若已有等价实现，复用并记录路径，不重复搭框架。

| 工作 | 首选复用位置 | 后继增量建议 | 变更边界 |
|---|---|---|---|
| 专家最终计划记录 | `world_model/eef_stream_v2.py`、原专家调用点 | `final_plan_record_v61.py` | 在函数返回后捕获真实final plan，默认旧run无新分支 |
| 双mask/schema | `sla_adapter/reference_dataset_v6.py`、原sealed manifest | `action_contract_v61.py`与新sidecar | 不改旧schema准入/标识 |
| 部署标签回放 | `sla_dense_policy_v5.py`、`FiftyMsProgram`、原native worker | `replay_label_contract_v61.py`＋有限脚本 | 模型替换为oracle packet，后续路径不换 |
| S16拟合 | `runtime_reference_v6.py`、原500配方 | 显式新recipe/spec | 新run，原parent和优化器边界明确 |
| PD1 mask训练 | 上游`pi0_align`/项目loss包装 | `masked_plan_loss_v61.py` | 全true等价；不热换V6方法 |
| 兼容读取 | 当前lossless shards与SLA data loader | `export_lerobot_view_v61.py` | 不重写Corelab采集热路径 |

新identity/schema模块只依赖标准库，不在native元数据导入SciPy/JAX/Torch/Flax，避免重现训练依赖泄入原生环境的已知故障。对改变的接口做定点回归；维持原生成随机数、dtype、padding、读取次序的可复核性，不为重新测试重复跑相机/GR00T/WBC旧资格。

## 11. 作业分工、效率与失败分流

复用现有Corelab四独立世界授权与实测配置，不擅自扩8/16；每世界单owner、共享checkout单writer。RP8涉及控制历史核对，先1个诊断世界跑完首个，再按独立状态支持并行；生产采集4世界继续独立。CPU-only标签验证和reader核对不占GPU。

H200/A40只在实际allocation可用时提交；未知旧job不取消、不复制。候选模型/Adam/梯度驻GPU；一次必要观察H2D，小动作及计数D2H。final-plan约1600B/query、两个bool mask合计80B/query（未计metadata），异步批量落盘，不为每个knot fsync或调用GPU。label hash/统计验收在封存/版本边界做，训练batch不重复hash/FK/渲染。

| 观测 | 处理 |
|---|---|
| 物理label过，但FP32/解码失败 | 修统计/顺序/dtype/旋转/夹爪，固定新身份重验；不加训练 |
| 完整数值链过，native replay失败 | 检查reference历史、preview、支持/接触、控制时钟；不降低WBC约束 |
| oracle过，小拟合raw失败 | 分析拟合/噪声/曲率和模型输入；不盲扩D1000、不偷偷用endpoint重建 |
| 小拟合过，新布局失败 | 数据覆盖、视觉/指令条件化与恢复；回到V5学习曲线 |
| PD监督增加但实际消费不增 | 分开报告计划可学与任务可行，不用mask掩盖拒绝 |
| 两reader不一致 | 修导出/transform，先继续已经验证的高效reader |

## 12. 顺序、预算和验收输出

| 包 | 新工作量 | 输出 |
|---|---:|---|
| W0 现场核对/在途V6 | 0新增候选更新，复用既定3K | 当前终态/未知边界，不重复提交 |
| W1 最终计划与mask合同 | 类型/定点CPU测试 | planner/adapter边界、schema、mask真值表 |
| W2 全链离线＋RP8 | 复用1220记录；8个native诊断 | 物理→FP32→部署→实际消费的对照 |
| W3 S16拟合 | 500×32=16000访问；4个学生任务诊断 | 固定500包、32条预测检查、任务结果 |
| W4 可选同数据SF | 新增3000×32=96000访问；复用FM端点 | 匹配数据的空间监督结果 |
| W5 可选PD20 | 最多40采集尝试；2×500×32=32000访问；8个学生诊断 | 同episode两视图，mask计数与闭环对照 |
| W6 后置reader核对 | 64样本读取，0模型/物理 | 官方兼容与高效路径张量一致 |

必需新增候选预算只有W3的500更新/16000访问。条件分支W4+W5全跑时，新增候选共4500更新/144000访问；mask资格≤8更新/256另计，全部新优化最多4508/144256。RP8+W3=12个新增原生诊断；PD加8后20个，不包含独立专家采集最多40、继承V6评价与未来正式DEV。已有run依身份去重，不能把继承预算再算一遍或自动排满可选矩阵。

每项阈值是工程验收目标，不是已达成绩。原生oracle不计学生成功，S16不计泛化，静态certifier不证明未来接触，官方reader兼容不证明动作语义正确。

下一次执行回报优先包含：**V6真实终态；RP8逐条件原生结果或首个明确阻断；S16固定500的实际可执行性；双mask计数；当前可复用/仍缺的数据与reader差异。** 不接受以新增文件数、排队状态、CPU测试数量替代策略进展。

## 固定来源

[E1]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4273dcdf143a0f9bac28afc5f831eb597a240412/reports/WM02/sla-long-horizon-v5-20261114/EXECUTION_20261114.md
[E2]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4273dcdf143a0f9bac28afc5f831eb597a240412/src/go2arx5_vla/world_model/eef_stream_v2.py#L30
[E3]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4273dcdf143a0f9bac28afc5f831eb597a240412/src/go2arx5_vla/world_model/sla_dense_policy_v5.py#L33
[E4]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4273dcdf143a0f9bac28afc5f831eb597a240412/src/go2arx5_vla/sla_adapter/reference_labels_v6.py
[O1]: https://github.com/OpenHelix-Team/Spatial-Forcing/blob/405ca8879c032a05f0a778adc7d86f4e221c5eba/SLA/README.md
[O2]: https://github.com/OpenHelix-Team/Spatial-Forcing/blob/405ca8879c032a05f0a778adc7d86f4e221c5eba/SLA/src/openpi/models/pi0_align.py
