# SLA → Go2＋ARX5＋冻结 WBC：本体后训练协议 V1

日期：2026-11-12（Asia/Singapore）  
状态：**DESIGN_READY / RUNTIME_NOT_ADMITTED**。本文件是用户要求的 Codex 执行计划，不是训练、仿真、模型下载或新站点作业已经完成的回执。

## 0. 目标、范围与立即动作

直接复用 `haofuly/SLA_base`，在已有原生机身＋腕部 TRAIN100 上完成新构型后训练；输出能正常保存恢复、普通 RGB 推理并通过原控制链执行的策略。**不重做多本体持续预训练，不等待 Table30 采集来源证明，不先引入 WM。**

主顺序：最新现场读回 → 模型/代码固定 → 数据与 EEF codec → 单卡真实训练资格 → S1 动作后训练500步 → 完整 TRAIN4及独立开发评价 → S2 同起点空间对齐对照 → 根据真实结果扩展。S2的教师下载、缓存准备可并行，不能阻塞S1。

本轮比较的是可部署的 `SLA + 明确动作适配 + ENDPOINT_BASIS + 冻结WBC`。它不是原始SLA稠密轨迹直接控制，也不是同一高层权重的零样本跨本体证明。原始轨迹与重建轨迹分别保存、分别评价。

## 1. 固定审阅证据与现状

- 项目 main：`6843163d641eb24a29c3b1557f868765346ba66c`，主要是已整理的入口；当前完整执行栈并未自动合入main。
- 当前执行 PR #376 审阅 HEAD：`893db1f49945dda8b318e846a58fd19ed7543940`。这是实现参考，不要求把在途任务回退到该SHA。启动时读最新合法HEAD、ACTUAL、owner、PID/队列。
- 新 body/wrist 数据：100回合、1500 captures、1450监督窗口，原50个干预capture保留但不直接用于BC；已有相机覆盖、原GR00T Processor和真实图像绑定验证。[R1]
- 最新T0独立回执：SFT2000＋固定endpoint basis在TRAIN01/07/13/25 **4/4完整成功**，技术失败及硬安全均0；不是新的SLA成绩，也不是独立泛化测试。PR正文仍可能停留在1/4读回，不用旧摘要覆盖新终态。[R2]
- 最新T1：官方GR00T真实加载、CUDA缓存、完整batch32比较、连续4对独立2＋2、普通RGB重建已通过；10次非候选优化更新、320访问、24次普通RGB生成。不得再为SLA计划重做这些GR00T资格。[R3]
- GR00T A_TARGET500是否已经启动/结束要现场查证；本计划不预填0、运行中或成功。健康作业自然收尾，已完成结果复用。
- 旧raw H40的0/8、BF16跨硬件REJECT、Teacher24最终19/24、旧DEV/TEST及历史checkpoint全部保留。SLA采用新运行身份，不改写旧负结果。

## 2. 本轮科学问题与候选

| ID | 初始化与目标训练 | 用途 | 是否第一优先 |
|---|---|---|---|
| S0 | SLA_base＋目标codec/统计，无更新 | 固定开环加载基线；默认不执行未资格动作 | 是 |
| S1 | SLA_base＋原生FM＋参数高效目标适配，500更新 | 核心交付：预训练SLA能否学会目标构型 | 是 |
| S2 | 与S1相同父权重、LoRA初始化、数据/噪声/优化器；FM＋空间对齐，500更新 | 单独测后训练空间监督增益 | S1链路可用、教师与梯度资格后 |
| G | 已有官方GR00T A_TARGET500；若尚未取得结果则保持待比较 | 工程系统选型，不声称纯架构/空间监督因果 | 不阻塞SLA |
| P | pi0.5_base＋与S1完全相同目标recipe | 进一步分离SLA预训练初始化价值 | 后继，不在首轮自动启动 |

S1不是论文完整后训练复现；S2接近其空间对齐机制，但采用项目的单卡PEFT、数据、时间接口和预算。S1/S2均不能据小任务成功宣称开门、书架或跨本体能力。

## 3. 上游、模型与许可固定（SLA-P0）

上游使用 `OpenHelix-Team/Spatial-Forcing` 的 `SLA/` 子目录，审阅代码固定：
`405ca8879c032a05f0a778adc7d86f4e221c5eba`。[R4]

权重首选 `haofuly/SLA_base`，不是LIBERO/RoboDojo专用checkpoint。该Hub发布已核验，但**本次未取得可独立确认的Hub revision及全部权重SHA**；规格中的null是待解析项，不得视为匹配任意版本。

Codex须执行一次有限元数据获取，将SLA_base、tokenizer、任何依赖权重及S2教师固定到不可变revision，记录每个实际加载文件的大小/SHA256、Orbax元数据、参数树与上游环境版本。下载后逐字节核验；已有相同资产可复用，不重复大文件下载。新镜像/替代权重不得静默替换。

记录代码LICENSE、LICENSE_GEMMA、基模与教师使用条款及权重发布说明。当前SLA_base缺乏完整模型卡，不能从训练组件的Apache标签推断全部权重相同许可。已允许的内部研究按实际条款进行；许可无法确认时明确列为该资产使用阻断，不自动重新分发权重。只发布配置、代码和证据摘要。

先复现上游模型构造、参数加载及一组固定输入上的自由生成。除明确新增的零初始化LoRA分支外，不得随机重置共享主干、embedding、action projection。加载必须列出missing/unexpected keys，正常去除训练专用align_projector也要记录。LoRA加入前后应在同输入/同噪声下保持初始函数在预登记容差内一致。

## 4. 代码组织与现场治理

文档分支从main建立。实现分支从**当前通过审阅的#376执行栈**建立独立worktree；不要直接用main假装拥有完整native runtime，不整体盲合历史730个提交。上游SLA放在独立用户环境/固定checkout或submodule，保留许可证，不覆盖现有GR00T环境。

建议新增小而明确的模块（名称为待实现接口，不是现有可执行命令）：
- `src/go2arx5_vla/sla_adapter/codec_v1.py`：物理EEF↔SLA内部动作。
- `dataset_v1.py`：既有sealed窗口→SLA观测/动作；不改旧数据。
- `config_v1.py`：精确PEFT/日程/损失配置与数据访问。
- `checkpoint_v1.py`：权重、统计、参数清单、恢复状态及推理导出。
- `provider_v1.py`：普通RGB→raw H40→既有basis的唯一神经提供者。
- `scripts/sla/qualify_v1.py`、`train_v1.py`、`evaluate_v1.py`：薄封装，复用上游训练与项目物理执行，不再造owner/调度器。
- 对应codec、mask、梯度、恢复与native seam测试。

一个共享物理世界只有一个owner。同卡重模型串行，独立站点可在合法既有授权内并行。UNKNOWN先查真实效果，CLOSED不复活；在途capsule不pull、不改seed、模型和协议。禁止sudo、系统服务/挂载/网络/凭据变更、他人GPU任务干预或清理私人资产。

## 5. 数据：首先复用，不先采集（SLA-P1）

输入父数据参考：
- dataset SHA256 `11ac2104e72ddb091045bb2ae0f8cba56bd4002022fc44825e983173565efbe1`；
- 原GR00T统计SHA256 `0faba237b8922ed9bbd53bd6732a9f449f46dcb6fc19017eae8f3cffa3afd9f2`。
它们绑定现有目标数据来源，不表示SLA应直接使用GR00T统计。[R3]

新的派生数据必须保留100 episode、1450窗口及各窗口原始状态、动作、干预mask、native时钟、RGB文件身份和命令消费证明。1500 captures完整保留，50干预不重新标成专家动作。文件缺失不静默删样本；先报告并修复访问。

每条输入仅包含新body RGB、原wrist RGB、当前实测EEF/夹爪、原语言。物体真值、Teacher阶段、未来图像、接触标签、评分不进入Student。几何真值只作数据校验和评价。

图像映射：body→`base_0_rgb`，wrist→`left_wrist_0_rgb`；`right_wrist_0_rgb`用固定空图并设置False mask，不能把同一wrist复制成有效第二只手。该名称只是单臂槽位约定，不代表机器人实际拥有左/右双臂。固定相机顺序、RGB通道、dtype、resize/padding；不修改相机安装。

首轮使用实际SLA的224×224 resize/pad，无随机几何裁剪/旋转和颜色增强，以保持S1/S2的视图一致并便于缓存。这是项目设计，区别于作者默认ColorJitter。重新检查SLA处理后的目标可见性（各阶段至少90%窗口有一路≥64可见像素）；原256和GR00T处理通过不能代替新检查。离线mask只用于审核；不得输入模型。若不通过，先检查转换而非训练；需改相机则另版本，不覆盖现役。

## 6. EEF、参考系、夹爪和时间的唯一合同（SLA-P1核心）

### 6.1 物理标签与固定任务坐标

物理监督继续使用原Teacher实际提交/消费的**名义H40**。不得改成未来实测TCP来降低误差，不把仿真成功终点当整段标签。

由现有官方项目codec解析每个EEF10的XYZ与row-rot6d得到SO(3)，夹爪沿用原实测state和原nominal action。必须先走已审阅的target frame/TCP链，不能重新硬加公开ARX的145mm偏移。

定义任务坐标M：在episode原post-hold启动时，以实际ARX安装基座建立已标定刚体变换T_WM；此变换在该episode固定。它来自机器人标定/启动状态，不使用物体真值。state和整个H40都用同一个T_MW转换。中途base运动不能逐点改锚；未来移动操作另立协议。

若旧数据缺失建立该帧所需的真实标定，先补齐已有日志的绑定，不能默认为固定底座零值。变换后的动作还原回当前项目原frame时，须在原参考/安全函数上保持同一物理轨迹。

### 6.2 内部状态与动作

单臂有意义的7维：`[px,py,pz,rotvec_x,rotvec_y,rotvec_z,gripper]`。state是当前绝对任务坐标姿态；动作在chunk起点s0上相对化：
- Δp_k = p_k − p_0（不是R_0转置后的body-local平移）。
- ΔR_k = R_k R_0^T；Δr_k = Log(ΔR_k)。
- 解码p_k = p_0＋Δp_k，R_k = Exp(Δr_k)R_0。
- 夹爪动作保持绝对nominal命令，不做相对差，不把受物体阻挡的actual aperture当command。
这些群运算与上游`DeltaActions/AbsoluteActions`的乘法侧一致。[R5]

优先把上述7维放入32维容器的前7维，其余25维规范补零；对照已发布单臂policy核验该槽位约定并记录。上游权重中的32维不是32个本体自由度。首轮保留上游32维原FM归约，不擅自改成7维masked loss；同时分开记录有效7维和padding25维loss/输出，避免padding下降伪装操作学习。只有前7维被物理解码，padding不能驱动机器人。

统计由同一份TRAIN100在上述转换后计算；S1/S2共用同一TRAIN-only quantile统计。不得用DEV/TEST、也不得沿用GR00T rot6d统计。记录q01/q99、常量维度处理、越界率、任何有损步骤。action transform不得隐式clip来伪造准入；state token离散化的上游规则单独说明。

### 6.3 时钟与采样器

配置`action_horizon=40`，标签时刻0,50,…,1950ms，执行40个50ms区间共2s；每2s重新观测。模型horizon50可配置为40，但不能把默认50点在输出端随意截断冒称相同时间合同；验证加载参数shape及同horizon自由生成。

SLA使用其原`sample_actions(num_steps=10)`生成数学，t=1为噪声、t=0为动作；不硬套GR00T四步采样。[R6] 两方法噪声形状/分布调用不同，不宣称相同整数seed就等于相同噪声；S1/S2内部需逐项匹配。

### 6.4 往返及真实消费阳性对照

对全部1450窗口和独立构造的旋转/夹爪边界样例，执行physical→task frame→SLA delta→归一化→逆变换→physical往返，检查SO(3)、左右乘、单位和锚点。FP64/FP32纯codec建议XYZ≤1e-6m、旋转测地误差≤1e-5rad、gripper≤1e-6；归一化常量维和有损处理单列，容差在看模型成绩前固定。

专家原H40与round-trip H40必须通过相同参考检查并给出一致解释。采用已有一个成功TRAIN参考，只在新codec实际改变消费入口时进行一次最小原生回放；不为已证明完全不变的物理层重复完整资格。

## 7. 首轮可训练范围与计算策略（SLA-P2）

### 7.1 明确PEFT范围，避免意外全模型训练

S1/S2基础配置：`pi05=True`、`gemma_2b_lora`＋`gemma_300m_lora`。固定上游LoRA：VLM attention/FFN rank16 alpha16，action expert rank32 alpha32。[R7]

允许更新：上述LoRA参数，以及`action_in_proj`、`action_out_proj`、`time_mlp_in`、`time_mlp_out`。S2另外更新`align_projector`。冻结SigLIP视觉塔、全部非LoRA语言/动作主干、embedding、其余norm/层。明确这是本计划的SLA低秩适配许可，不是解冻旧GR00T或整个VLM。

必须用**正向白名单**建立实际参数清单、numel、dtypes和冻结fingerprint；不能仅调用上游`get_freeze_filter()`后假定视觉塔已冻结。该函数的LoRA筛选主要针对llm，其他模块是否训练须由项目白名单决定。

所有LoRA分支在加载SLA密集权重之后按固定seed初始化，初始化函数等价需检查；S1/S2 LoRA和公共接口起点完全相同。禁止意外随机初始化预训练action投影后仍声称完整继承。

### 7.2 Core优先，不以未经测试的显存数字担保

默认Core 4080S、单GPU、microbatch1、累积32。密集冻结参数BF16；可训练参数、梯度及Adam状态按实际框架规范保留适当精度，记录驻留位置；训练激活不做CPU offload。

先有限实测加载、自由生成、forward/backward和一次完整effective batch32更新，区分JIT编译/加载与稳定步耗时。允许数学不变的rematerialization/activation checkpointing、按需释放无用副本；无全量EMA拷贝（首轮`ema_decay=None`）。不在同GPU同时驻留GR00T、SLA和OmniVGGT。

如果16GB无法容纳：记录真实OOM位置和常驻/激活分解，修复明确生命周期问题后最多一次同语义复验；仍不够则提交容量结论。不得静默减H40、分辨率、有效batch、loss维度或训练白名单。可继续CPU转换、测试和已资格推理；使用Vanda/Hopper仅在**当前现场权限和用户有效单卡授权**确认后单独登记，不把本文件自动解释成解除上一轮禁止新大卡作业的限制，不偷偷使用HF付费云作业或多GPU。

### 7.3 只缓存真正冻结的计算

可选缓存SigLIP输出，前提是输入图像处理不变且普通RGB与cache的输出、loss、梯度、更新一致。VLM有LoRA且接收语言/state，**不得缓存VLM输出或旧GR00T backbone_features**。不把整个冻结前缀no_grad错误扩展到需要传递梯度的图。

## 8. SLA真实资格与工程验收

复用现有owner/数据/物理证据，只新增SLA相关检查：
1. 固定八个TRAIN窗口（覆盖各阶段），执行两组噪声的普通RGB自由生成，记录32维与物理H40、无缓存可加载状态；不要求零更新模型已经会任务。
2. 实际白名单梯度非零、冻结权重不变；新统计/状态dropout、图像mask、相对anchor无泄漏。
3. 连续4更新与独立2＋2恢复，全态及下一步一致：模型、Adam、scheduler、JAX RNG、数据顺序、dropout/噪声、global_step。共8次**非候选**更新；有cache时另加两路线各1次完整batch32对照，共10次。
4. 同卡同版本重复与恢复优先要求bit-exact；若JAX算子造成非逐位但稳定可解释差异，先记录并在TRAIN资格阶段预登记数值容差，不能看DEV/任务成绩后放宽。跨设备不要求未经证明的bit-exact。
5. 从保存包在新进程做普通RGB推理，无训练cache、无教师仍可运行；LoRA是否保留或合并必须绑定，不能drop adapter。保存部署所需统计/tokenizer/codec/相机/采样器与父权重hash。

资格权重全部不晋级候选；正式S1/S2都从干净SLA父＋相同初始LoRA开始。

## 9. 正式S1训练与立即闭环（SLA-P3/P4）

首端点500更新，micro1×accum32=16000样本访问；seed20261112。AdamW，peak lr5e-5，weight_decay1e-5，global_grad_clip1.0；25更新线性warmup，随后余弦至第500步lr1e-5。保存0/250/500及LR向量hash，500为固定主端点。该单卡配方是本计划设计，不冒称作者的30K/大batch复现。

`state_dropout_prob=0.25`仅训练时丢弃**离散state token段**，保留真实state给相对动作编码/逆变换和执行。部署始终使用正常state。dropout不得把WBC反馈或动作anchor置零。首轮禁其他数据增强，S1/S2保持一致。

使用当前1450窗口及既有episode内32 anchors，共3200项exposure×5轮排列，固定16000访问；不称1450窗口均匀采样。S1/S2共享该map、noise/time和dropout种子流。原FM的生成时间Beta/噪声、有效batch平均与梯度累积语义保持上游定义；同时报告7/25维分解。日志每50更新，0/250/500固定开环探针；评估RNG不改变训练RNG。

500完成即保存恢复、普通RGB重建并进入新身份TRAIN01/07/13/25各1请求。使用已固定ENDPOINT_BASIS_V3、原H40完整消费和原原生检查；最多15查询/条件。raw先完整保存并计算包络，basis从实际EEF、上一个真实消费nominal gripper与预测末点重建；不得隐瞒中间点被替换的事实。

**不要求S1先赢GR00T才允许报告，也不自动续1000/2000。** TRAIN4是开发诊断，不是4条新训练示范、不是正式DEV/TEST。普通策略拒绝和失败保留完整分母。

## 10. S2：目标域空间对齐，不能只训练一个可丢弃投影器

完成SLA基础链路、教师身份与资源条件后，运行第二臂；不要求先有任务成功，不能根据S1的某个好运样本选择loss权重。

固定从同一SLA_base和同一初始公共LoRA开始，不从S1_500继续；否则变成多500步而不能隔离空间监督。

- 目标教师：固定revision的OmniVGGT，冻结、eval/inference_mode。首轮`use_camera_params=False`，只使用同一对RGB，不引入物体真值、未来帧或额外深度。
- 对齐配置：沿审阅公开配置固定`vla_layers_align=12`、`vla_adapt_layers_align=False`、教师末层、bilinear pooling、VLM norm、padding mask、alpha=0.5。这里是**固定层对齐**，不声称已经复现论文自适应多层门控。
- 教师目标可离线逐批提取到磁盘，再卸载教师；与实时教师在固定样本对齐。缓存身份包含教师revision/层/输入/掩码/图像顺序/预处理，不能把相机增强后特征当同一张图。
- 证明空间loss对部署时保留的VLM LoRA有非零梯度，教师无梯度；单独align_projector变好不算策略学习证据。
- S1与S2使用相同preprocess、state dropout、公共参数范围、LR、500更新和16000访问。为消除`train.py`和`train_align.py`默认增强差异，在项目wrapper绑定同一处理。alpha=0的align代码须复现S1的公共参数梯度/更新；不以训练接口差异冒充空间loss收益。
- S2输出checkpoint允许带训练projector，部署端明确移除它和OmniVGGT，普通Pi0推理加载公共参数及LoRA；不能意外调用teacher获得在线特权信息。

教师/许可/显存未就绪时，S1的原生结果仍必须交付；S2明确记为NOT_RUN/BLOCKED，不把缺teacher概括成SLA失败。

## 11. 统一原生开发评价与模型比较（SLA-P5）

首先S1/S2各固定TRAIN4，完整记录运行结果。随后建立**独立SLA_DEV16**：8个未参与优化的新布局×2重启/方法。根据原几何可达支持域选取，事先与旧TRAIN/DEV/TEST人口按实际scene身份去重，不读取旧TEST。具体布局、seed、顺序和初始条件hash在任何候选预测前登记。

默认完成全部16/方法，不采用“先看赢2/8再决定补剩下”的选样流程。实现完整且无硬安全问题即可评价；模型first-H40拒绝是有效策略失败。若发生原生硬安全STOP，按原协议停受影响执行并保留NOT_RUN，不强凑完整人口。技术故障必须修复/版本化，不能作为算法输赢。

模型至少包括S1_500；S2完成后同人口评价；G优先为同目标数据官方GR00T A500，尚未交付则保持待比较，不阻塞SLA。可列SFT2000历史新视角4/4但明确父、训练量不同，不能当公平匹配控制。相同scene seed和初态容差；不同架构不假定同数字seed等于同噪声。方法顺序交替，后一个restart反转顺序。

主指标：原真实双指抓持、抬升≥20mm、连续持稳≥1s、原V3 standing和原硬安全。完整保留旧2mm/0.5°诊断，不把它false改为任务false或删掉。辅指标：首次失败阶段、raw/basis/IK/关节/接触拒绝、实际消费H40数、物体滑移、夹爪事件、base/足状态、模型延迟、物理暂停及总成本。

报告每layout及总体点估计和布局聚类区间；两次restart不是两种独立几何。16条件是开发证据，不用2个成功差就宣称统计显著性。

## 12. 结果分流，避免再次陷入无限微调

| 结果 | 下一动作 |
|---|---|
| 模型/许可/接口/显存不通过 | 保存精确阻断，继续已可独立完成的CPU/数据/基线工作；不记作策略负结果 |
| TRAIN专家往返或原参考本身不通过 | 修codec/标签/时钟，禁止用更多更新掩盖 |
| S1 500在TRAIN开环仍无方向性改善 | 一个固定8–16窗口的有限记忆/梯度诊断作为独立实验；不自动扩数据/大卡 |
| 方向正确但全部原生参考拒绝 | 对比raw、basis、参考变换与关节证书，保留解码器干预身份 |
| 可执行但接触时序/恢复失败 | 记录真实状态缺口；仅在可靠Oracle成立后定向补数据，暂不批量96/32 |
| S1/S2已完整成功且DEV呈稳定收益 | 冻结候选、独立TEST方案及补种子，然后最小放置技能 |
| S2不优于S1 | 保留普通SLA适配主线；不因论文强而强保空间教师 |
| SLA不优于当前G | 接受本recipe下结果，先做匹配pi0.5/容量/表示诊断再决定替换，不无限并行分支 |

500之后的1000/2000、第三seed、公开回放、全动作专家解冻、多任务、WM或更短prefix不是自动续训。本轮结束必须给出明确科学判读与一项有依据的后继配置。不能仅因为单任务一次失败判定SLA或全部多本体预训练无效。

## 13. 后继长程路线（条件性，不阻塞首轮）

1. **预训练归因**：pi0.5_base与SLA_base同adapter、同PEFT、同数据/预算，对比初始化收益；若还要区分数据规模与空间监督，需对应消融权重，不能仅两模型完成该归因。
2. **数据效率**：预先固定按episode/布局划分的25/50/100示范子集与相同评价，比较学习曲线；不把窗口数当示范数。
3. **独立确认**：在开发选型结束后、同一权重固定，20新布局×2重启/方法，三个预登记训练seed；全部候选在共同TEST首次打开前冻结。TEST不进入调参。
4. **多技能**：先合格PLACE Teacher与释放/撤离，再共享抓取/放置；之后才是可达柜门与单书插槽。同图不同语言应对应不同正确动作；逐任务评价与旧任务保持。
5. **移动与跨本体**：当前只站立整机；更换基座姿态、主动移动和跨执行器是独立变量。目标数据微调成功不宣称高层完全本体无关。

## 14. 成本、保存与唯一交付

主预算S1=500×32=16000访问；S2同样；双臂总1000更新/32000访问，不含资格。资格基础8个更新/256访问，可选cache对照另2更新/64访问；两臂共享的已证明资格复用，S2只补改变的空间梯度/teacher接口。每次实际模型调用、教师调用、JIT编译、加载、训练、评估、物理积分分开计数。

每个候选至少保存原父hash、可训练参数及dense checkpoint/adapter、optimizer、scheduler、step、全部RNG、data map/cursor、SLA统计/tokenizer、codec/task frame、camera版本、precision/sampler、frozen指纹。推理导出必须可以在无训练缓存/无教师的新进程加载。失败及raw数组留存私有，公开摘要不包含host凭据、资产路径秘密、原始私人RGB和未授权权重。

每个冷端点输出：`QUESTION / CHANGED / ACTUAL / NEGATIVE / DECISION / NEXT`。同时给requested/attempted/terminal/unread/not-run、技术/科学/安全分类、完整分母、UTC+SGT和实际资源。

首批最终交付：
- `SLA_ASSET_AND_LOAD_ACTUAL.json`；
- `SLA_TARGET_CODEC_DATA_ACTUAL.json`；
- `SLA_CUDA_TRAIN_RESTORE_RGB_ACTUAL.json`；
- `S1_TARGET500_ACTUAL.json`及完整checkpoint；
- `S1_TRAIN4_ACTUAL.json`；
- 条件满足时`S2_TARGET500_ACTUAL.json`与`SLA_DEV16_COMPLETE_ACTUAL.json`。

这些是待实现产物，不预填PASS。不存在运行资格时，不运行声明性spec。工程计数和计划PR不是任务成功。

## 15. 一手来源和可复用入口

[R1] 当前执行PR及数据： https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/376  
[R2] T0固定4条件终态： https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/893db1f49945dda8b318e846a58fd19ed7543940/reports/WM02/native-arx5-transfer-20261112/T0_SFT2000_NATIVE_TRAIN4_FINAL_ACTUAL.json  
[R3] 官方GR00T真实资格： https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/893db1f49945dda8b318e846a58fd19ed7543940/reports/WM02/native-arx5-transfer-20261112/T1_OFFICIAL_PARENT_CUDA_CACHE_RESTORE_RGB_ACTUAL.json  
[R4] SLA发布入口： https://github.com/OpenHelix-Team/Spatial-Forcing/blob/405ca8879c032a05f0a778adc7d86f4e221c5eba/SLA/README.md  
[R5] 源动作转换： https://github.com/OpenHelix-Team/Spatial-Forcing/blob/405ca8879c032a05f0a778adc7d86f4e221c5eba/SLA/src/openpi/transforms.py  
[R6] 原FM/10步生成： https://github.com/OpenHelix-Team/Spatial-Forcing/blob/405ca8879c032a05f0a778adc7d86f4e221c5eba/SLA/src/openpi/models/pi0.py  
[R7] LoRA精确配置： https://github.com/OpenHelix-Team/Spatial-Forcing/blob/405ca8879c032a05f0a778adc7d86f4e221c5eba/SLA/src/openpi/models/gemma.py  
[R8] 空间后训练示例： https://github.com/OpenHelix-Team/Spatial-Forcing/blob/405ca8879c032a05f0a778adc7d86f4e221c5eba/SLA/src/openpi/external_config/train_robotwin.py  
[R9] 权重：https://huggingface.co/haofuly/SLA_base ，教师：https://huggingface.co/haofuly/SLA_training_component  
[R10] 论文：https://arxiv.org/abs/2611.08373 。本协议的方法设计以已读代码和摘要为依据；未把论文未核实的性能/算力数字当本体保证。
