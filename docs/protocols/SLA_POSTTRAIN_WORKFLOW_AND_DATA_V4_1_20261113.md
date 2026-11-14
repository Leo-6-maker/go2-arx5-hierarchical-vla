# SLA 后训练操作说明 V4.1：模型、数据集、训练顺序与交付

日期：2026-11-13（Asia/Singapore）。本文细化用户要求的“后训练具体流程、细节以及数据集使用”，配套[主计划V4](SLA_SOURCE_TARGET_MOBILE_V4_20261113.md)和[原机读矩阵](SLA_SOURCE_TARGET_MOBILE_V4_20261113.json)。

**主线固定为：SLA_base → 目标域监督适配 → 精选源/目标混合对照 → 专家纠正 → 独立闭环评价 → 分阶段移动适配。当前不使用强化学习，不重新做大规模持续预训练。**

本文新增明确的源数据五任务清单和操作细节，不改已启动run、旧标签、旧评分或WBC。表中的新数据规模是采集/筛选目标，不是已取得数据量；模型训练端点与数据版本必须用实际manifest绑定。当前运行入口仍沿现役执行分支，不能把本文当作已实现的CLI。

## 1. 从什么模型开始，哪些已经完成

- 模型：`haofuly/SLA_base`，项目已绑定revision `5bbdf3012ab643554aa2eb83ddeae9ace3dce476`。
- SLA上游代码：`OpenHelix-Team/Spatial-Forcing@405ca8879c032a05f0a778adc7d86f4e221c5eba`。
- 核验执行分支：[PR #382](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/382)，本次读到HEAD `3195f0eb664ed98db7f76ad0643ed1d742472a7b`，最新实际摘要20:59 SGT。
- R1已完成H200上500次更新、16,000次优化访问，已完成H200/Core普通RGB重建；R1原生任务评价仍为0。它使用旧稀疏TRAIN100，不是新D100。
- 新生产D100/D300/PICK_PLACE准入仍为0；工程collector成功、数据分片封存不等于生产数据已准入。
- 上述为GitHub证据快照，不是本次登录服务器实时检查。执行前读最新终态，健康作业不重投、不热改。

SLA_base已继承π0.5与空间持续预训练。我们的后训练负责相机/EEF/夹爪/任务适配，不重新下载全部持续预训练语料。OmniVGGT是训练时的冻结空间教师，不是动作标签的生成者。

## 2. 数据集到底使用哪几份

| 项目内数据ID | 来源与语义 | 本轮用途 | 目标数量 |
|---|---|---|---:|
| `T_legacy_TRAIN100` | 旧Go2+ARX5原生专家命令；100 episode、1,450窗口、50干预capture排除BC | 保留S1/S2/R1及原对照；不改名为新数据 | 已有100 episode |
| `T_D100_new` | 新合同下原生仿真完整成功专家示范 | 第一版新数据FM/空间两臂 | 100 |
| `T_D300_new` | 包含D100，补足同支持域的独立episode | 主参数配方与混合对照 | 累计300 |
| `T_D1000_new` | 包含D300，按证据扩量 | 数据规模后继 | 累计1,000，条件性 |
| `S_RoboTwin5_cmd_v1` | 下面固定五个RoboTwin任务，经命令语义转换的源池 | T-only vs 75%T+25%S主对照 | 1,000 TRAIN episode |
| `C_recovery_v1` | 学生在TRAIN场景偏离后，专家接管的正确执行片段 | 有监督恢复训练 | 首波至多200片段/400 rollout |
| `TARGET_DEV48` / `TARGET_TEST80` | 独立布局/对象/重启的在线评价条件 | DEV选型；TEST最终评价 | 每候选48 DEV；每最终seed80 TEST |

后五个数据/条件ID是项目定义，不是公开网站上的现成repo名。100/300/1000是完整独立episode数，不是帧数、窗口数或更新数。DEV/TEST额外生成，不从D300训练配额中扣除。恢复片段不得计作完整任务示范。

### 2.1 本轮目标任务与采集配额

主任务是`PICK_HOLD`与`PICK_PLACE`。分层为10个经过专家可执行性验证的位置/姿态层×5个TRAIN物体实例×2任务=100层。每层1/3/10条得到D100/D300/D1000；两任务均衡时每任务50/150/500条。

D100最多200次尝试；D300新增200条、最多新增400尝试；D1000新增700条、最多新增1,400尝试。每条真正执行成功并通过数据完整性检查才计入，不足时发布N_eff和欠覆盖层，不复制轨迹凑数。

PICK_PLACE必须包括转移、释放、撤离及物体终态；PICK_HOLD成功不能替代放置。如果只有一个任务/物体就绪，立即生成并训练具名小数据版本，不冒称已满足两任务五物体的D100，待专家就绪再建立正式两任务曲线。

同一场景/匹配初态下不同指令对应不同正确动作，D300目标包含50组成对控制经历（100条，已计入总300条），前提是专家可执行。仅改写prompt、重渲染或复制同一轨迹不增加独立episode数。

### 2.2 本轮开源数据明确选RoboTwin五任务，不是把所有数据集拼起来

首轮公开数据入口为`TianxingChen/RoboTwin2.0`。官方脚本默认读取`dataset/<task_name>/demo_clean.zip`，默认发布本体为`aloha_agilex`。本次核验RoboTwin代码main指向`ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`；实际数据revision尚须在下载前解析并锁定，不能用滚动main冒充固定数据版本。[官方下载脚本](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/scripts/download_xpolicylab_data.sh)

| 固定任务ID | 源技能 | S1000 TRAIN目标 |
|---|---|---:|
| `move_pillbottle_pad` | 将药瓶抓起放到垫上 | 200 |
| `move_stapler_pad` | 将订书机移到指定垫上 | 200 |
| `place_mouse_pad` | 抓鼠标放到指定垫上 | 200 |
| `place_empty_cup` | 将杯子放到杯垫 | 200 |
| `place_object_stand` | 将指定物体放到台座 | 200 |

这是**确定的首轮审核清单和数量目标，不是已下载/已合格的1,000条数据**。源TRAIN外另留固定源DEV用于拟合/保持诊断，按episode先划分再切窗，不能将源DEV也算进S1000。

官方任务描述支持单臂完成上述过程，但每条原数据仍须确认实际活动臂、其他臂是否接触/协助、TCP与command字段。不能无条件砍掉另一臂动作。明确排除`place_can_basket`：该完整任务还要求另一臂抬起篮子；也不纳入handover、lift_pot等双臂依赖任务。[官方任务文档](https://robotwin-platform.github.io/doc/tasks/index.html)

第一步只抽最多20条完整episode（五任务各4条）和64个固定窗口，核查时钟、命令、夹爪与转换；source reader的100–300条试点用于实现验证，数量包含在后续固定训练池，不额外计算成新增数据。数量不足时不重复采样冒充独立episode。

公开包里若只能得到future-achieved EEF或测量夹爪宽度，它属于`S_aux`，不能进入本轮`S_cmd`动作混合。首轮按以下规则决定：

1. 能核实desired EEF/夹爪命令及时间，则形成`origin_kind=public_recorded`的独立S1000。
2. 不能恢复命令或不足1,000条时，可在同一开源环境、同五任务、固定一个本体配置中重新执行专家，额外记录真正下发的命令，形成不同ID的`origin_kind=generated_in_source_sim`数据池。ARX-X5可作为该生成分支候选，必须先验证工具/相机/单活动臂合同。
3. 主混合臂锁定一个来源版本，不在同一S1000中默默替换公共下载与再生成来源；必要变更生成新manifest并重新绑定对照。
4. source仍不合格时继续T-only。现有ARX5 Table30审核可以复用，但未闭合命令来源前留在S_aux；DROID/RoboCasa/LIBERO不进入本轮首个混合池。

RoboTwin默认`endpose`字段是末端位姿/开度记录，名称本身不证明是desired command。默认保存频率也不等于我们的20Hz动作频率，必须按真实时间和控制语义转换。[配置说明](https://robotwin-platform.github.io/doc/usage/configurations.html)

本次未下载/审核这些任务的原始HDF5，不声称现有包的字段、数量、许可证和时钟已经通过。需绑定的字段仅是具体data revision、episode清单、source control/TCP/clock和站点路径，不是要求重新设计整个研究方案。源审核不能阻塞R1或目标D100。

## 3. 一条episode必须记录什么

| 字段 | 用途 | 输入学生？ |
|---|---|---|
| body/context RGB、实际活动臂wrist RGB、精确时间/native step | 图像输入与同步 | 是 |
| 当前实测EEF位姿、实测夹爪状态 | state及相对动作锚点 | 是 |
| language、task_id、目标指令 | 任务条件；task_id只作组织时不自动输入 | language是 |
| 专家下发的EEF/夹爪nominal、命令生效/消费区间 | 行为监督 | 作为训练标签 |
| 实际EEF/夹爪/机身反馈 | 跟踪诊断及下一时刻state | 仅声明的当前state |
| TCP/工具定义、坐标系、相机标定、单位、控制器版本 | 转换与可复现性 | 不新增特权输入 |
| source_domain、episode/layout/object/seed lineage、success、phase、intervention | 划分、平衡采样和审计 | 否 |

物体真值可以用于专家和评分，不自动给学生。源外部相机只是context view，不能声称是随Go2移动的机身相机；缺视角显式mask，不复制其他图像伪造。夹爪state是实测值，action是命令，两者分别保存。

Corelab默认原生物理2ms；RGB/state按10Hz实际记录，训练锚点每0.5s，名义参考20Hz。最终短前缀/完整chunk执行节拍以已验证consumer版本为准；10Hz记录或2Hz切窗不证明策略已经2Hz重规划。

每条成功episode额外真实执行至少2秒终末保持，形成后续窗口的真实命令。旧工程数据尾部不够就排除相应窗口，不补造未来静止标签；完整生产episode与部分有效窗口的口径分开。

## 4. 从原始episode生成训练张量

### 4.1 先划分，再切窗

先按episode、session、布局、物体实例及随机seed谱系确定TRAIN/DEV/TEST，去重后再切窗口。所有归一化统计只用TRAIN。对于每个观察锚点t，找到未来40条有明确来源的名义命令，采样时刻为t+0,0.05,...,1.95秒；实际消费覆盖2秒。

目标T的每点都要有原生消费证据；跨query时先还原到共同任务/世界坐标，再相对当前实测anchor编码。接管前错误学生动作、未执行预测后缀、跨episode补点均不进入正常BC。S_cmd只要求其源执行器命令的可核查语义，不伪造Go2消费。

### 4.2 固定工具、坐标和旋转约定

把当前实测EEF与未来命令都转换到同一个校准任务坐标系、同一个语义TCP。源关节位置目标只有在正确FK/TCP与控制含义明确时才能转换；关节速度/力矩不能直接当q做FK。

当前位姿为(p0,R0)，未来命令为(pk,Rk)：

- 位置：`delta_p[k] = p_cmd[k] - p_actual[0]`，在固定任务轴表达；
- 旋转：`delta_r[k] = Log(R_cmd[k] @ R_actual[0].T)`，左乘SO(3)约定；
- 夹爪：`g_latent[k] = logit(clip(g_nominal[k], 1e-4, 1-1e-4))`。

这是从同一个起点看整条H40，不是相邻点增量，不逐点累加。夹爪端点1e-4偏差明确记录，旧原始标签保留。state夹爪仍是实际开度，不变成logit命令。

把7个有效维度补入32维容器，目标shape为`[40,32]`；RGB按已验证的resize/pad转成224×224；状态编码为当前任务空间state并填充到模型所需宽度，语言使用固定tokenizer。保留当前FM32损失归约，不能把“25个padding维”悄悄改成新mask-loss实验，单独报告active7与padding25。

### 4.3 固定normalizer，防止混合臂偷偷改变动作定义

- D100两臂共享D100 TRAIN统计；D300四配方共享D300 TRAIN统计。
- T-only/M75/SEQ主混合对照统一使用同一D300 TRAIN normalizer；source先物理转换再用这套统计，不按域偷偷切换decoder。
- 源范围过大、某维退化时先报告；需要共同新scaling则基线和混合臂同时绑定，不能只改混合臂或隐藏clip。
- S_ONLY单独用source TRAIN统计，不用目标动作/状态统计或目标DEV调参。

### 4.4 封存为可训练分片

在线观察走内存snapshot，不写PNG再读回控制。异步writer封存无损图像/数值分片和manifest；训练只读committed版本。现有兼容归档可离线转换一次，但不把PNG解码、整库hash、FK和坐标检查放进每个training step。

建议manifest至少包含dataset/source revisions、episode与任务清单、source_semantics、实际帧/窗口数、split、codec、normalizer、camera、consumer、Teacher版本与shard校验和。一个run固定一个manifest，不边训练边append改变数据人口。

## 5. 模型训练哪些参数、使用什么损失

基础配方C00/C01沿既有V2：冻结SigLIP和VLM/action expert的密集主干；训练VLM LoRA(rank16,alpha16)、action LoRA(rank32,alpha32)、action_in_proj、action_out_proj、time_mlp_in/out。C01另外训练空间projector。加载SLA_base密集权重后再初始化LoRA，不能先随机化已有动作投影。

C10/C11是已登记的后继配方：VLM LoRA不变，完整action expert更新，密集action参数使用较低峰值LR1e-5。它们不是每次换数据必跑的四倍网格。

WBC、腿策略、增益、cost、solver和原物理/安全条件始终不参与这轮梯度更新。

### 5.1 动作损失：监督式Flow Matching

令a为归一化专家动作，epsilon为同shape标准高斯噪声，tau按固定上游Beta采样：

`x_tau = (1-tau)*a + tau*epsilon`

`u_target = epsilon - a`

`L_FM = mean((v_theta(obs, x_tau, tau) - u_target)^2)`。

样本有效性在reader决定；完整H40的来源不满足合同就不进入该样本。有效batch为32，4个microbatch8累积后更新一次，不是每个microbatch各更新一次。noise/time/dropout由逻辑visit身份派生，换分批方式不改变随机输入。

### 5.2 空间损失：只增加训练时的几何监督

`L = L_FM + 0.5*L_spatial`。

OmniVGGT冻结，用当前同一对RGB产生target；只用当前帧，不引入未来帧、物体真值或未声明深度。固定student第12层/teacher末层、现有projector、padding/view mask；本轮不是论文自适应多层融合的完整复现。

必须确认空间loss对部署保留的VLM LoRA有梯度；只训练将来会删除的projector不够。部署时移除OmniVGGT和训练projector。

可离线生成冻结teacher特征，每个实际训练锚点一次，并绑定图像预处理/teacher/layer/mask；不无条件对所有10Hz原始帧计算。不同crop或增强不能沿用错误cache。可训练VLM输出不能被离线缓存。

### 5.3 新3K配方的数值配置（继承V2）

| 参数 | 新D100/D300与主混合对照 |
|---|---|
| 有效batch | 32 |
| H200 microbatch/累积 | 8 / 4，复用实际合格版本 |
| 训练matmul precision | highest，模型/optimizer dtype沿已验清单；不等于全部改FP32 |
| AdamW | beta1=0.9，beta2=0.95，eps=1e-8，weight_decay=1e-5 |
| 全局gradient clip | 1.0 |
| LoRA/接口/projector峰值LR | 5e-5 |
| 完整action expert峰值LR | 1e-5，仅C10/C11 |
| schedule | 3,000更新；150更新线性warmup；余弦降到峰值20% |
| state-token dropout | 0.25，仅训练token段；动作anchor和WBC实测状态不置零 |
| 图像增强/EMA | 本轮无额外图像增强、无EMA |
| 保存点 | 0、500、1500、3000；固定3K为主端点 |
| 普通推理 | 原SLA 10步采样，H40，不需要teacher/cache |

旧R1的500步/25步warmup配方保持原样，不追溯套用150步warmup。新3K学习率、optimizer和visit map都要随checkpoint保存。

## 6. 真正的执行顺序：数据逐步扩大，不代表权重一路串训

### 阶段0：现役接口适配收尾

继续原S1/S2/G及R1既定评价。R1用旧TRAIN100，只回答新版有界夹爪与训练/部署接口；它不冒充多任务扩量或开源数据收益。已完成更新不重复跑。

### 阶段1：D100新数据监督适配

D100一旦完整准入，两个新run均从同一个SLA_base和相同公共adapter初值开始：

- `D100_FM_3K`：100% T_D100_new，L_FM，3,000×32=96,000次目标访问。
- `D100_SF_3K`：同一D100、相同96,000访问，L_FM+0.5L_spatial。

两者只改变空间监督及其必要projector。不能从FM训练后的权重再启动SF对照。500步checkpoint可做已登记诊断，不用其成绩私自替换固定主端点。旧计划中若已有独立pilot，单独计量，不重复伪装为一次训练。

### 阶段2：D300配方对照

Corelab在阶段1训练时继续生产D300；D300 C00/C01/C10/C11按既有预算各3K，从干净相同父模型开始，选择一个目标recipe。不是D100先训3K再D300继续3K后仍称“D300 3K”。

为了减少首轮资源竞争，先交付C00/C01；完整action expert两臂沿既定计划实施/报告，不隐瞒它们的LR差异。选好recipe后源混合不再同时改变参数范围或空间loss。

### 阶段3：D300与S1000混合对照

S1000尚未合格时阶段1/2继续。合格后先做两臂：

| run | 每次更新的32窗口 | 3K累计T/S访问 |
|---|---|---|
| `T100_3K` | 32T | 96,000 / 0 |
| `M75_3K` | 24T + 8S | 72,000 / 24,000 |

注意：`T100`里的100表示100%目标采样，**实际训练数据是D300，不是D100**。本文保留已登记ID但明确这个容易误读的命名。

两臂都从SLA_base初始化；T100与某个已完成D300端点在所有身份、visit及预算一致时直接复用。M75不从该目标端点继续，否则变成额外预适配。

域内抽样顺序固定：task均衡→该task内episode均衡→该episode存在的阶段均衡→阶段内合格anchor均匀。phase只用于离线采样，不作为学生输入；不把放置任务的release阶段硬套到PICK_HOLD。实际counts报告到domain/task/episode/phase。batch采用24/8配额，样本平均已经体现75/25，不再把域loss乘一次75/25。

M50、SEQ、S_ONLY以及两个4K端点是V4明确的条件后继，不是第一轮同时开完。M75_4K达到96k目标访问，比较时同时披露额外32k源访问/计算，不能冒称与T100_3K等计算。

### 阶段4：根据学生实际偏离采集专家纠正

首波最多400次TRAIN rollout，至多200个可信专家恢复片段。必须从学生实际到达状态重新求专家动作，不按相同时间复制旧轨迹。接管前错误学生动作和未执行后缀不作BC。

从同一选定checkpoint做两个匹配续训：`R_NOMINAL`为100%目标nominal，`R_CORRECT`为80%目标nominal+20%目标纠正，各1K×B32，optimizer同样重新初始化，normalizer固定。这里的80/20是nominal/correction，**不是source/target**；该对照默认不另加source replay。需要保留源时，另登记相同源比例的三桶对照，不暗加第三个20%。

### 阶段5：评价、部署与后继

每个正式候选做TRAIN8与固定DEV48。最终比较的一对方法增加两个训练seed，两个方法各三个seed，各TEST80。两任务各40条件，逐任务报告常规/挑战、策略失败、技术失败、硬安全、干预及未执行条件。

评价的核心是物体是否真实抓持/抬升/放置，原始输出是否合法、多少H40真正被WBC消费，及失败阶段。训练loss或专家任务成功不能替代学生闭环成功。DEV用于选型，TEST不进入训练、normalizer或数据源比例选择。

输出包必须含父权重revision、LoRA/接口权重、tokenizer、normalizer、codec、相机处理、采样器与consumer版本；完整训练checkpoint另含Adam、LR位置、随机数、sampler游标。新进程直接输入RGB/state/language应可复现，不依赖teacher或训练cache。

## 7. Corelab/H200如何分工

Corelab：任务专家+独立原生世界、持久相机、内存snapshot、有界异步writer；专家采集不加载SLA。先测2/4 actor的合格完整示范/小时，不以物理步数/窗口数冒充独立数据产能。

H200：固定数据快照stage后进行模型训练、teacher特征计算；模型/梯度/optimizer保持GPU驻留。允许必要的一次输入上传、紧凑动作返回、小指标和checkpoint持久化，不做逐层/逐flow-step图像或特征GPU→NumPy→GPU。

数据预处理离线完成一次；按shard增量交付。训练当前数据版本时Corelab生产下一版，不能读到未封存分片或中途改变normalizer。共用GPU时按真实显存配额管理teacher和student，不默认同时驻留就能容纳。

源RoboTwin渲染单独验证实际支持的设备。官方提醒采集/评价不宜直接假定A/H/V系列服务器GPU可用，H200是训练主机，不自动是RoboTwin采集后端。[采集文档](https://robotwin-platform.github.io/doc/usage/collect-data.html)

## 8. 现有代码还需要改哪些地方

当前`config_v1.py`仍固定500条LR、100 episode、1,450窗口和16,000访问；`runtime_v3.py`仍引用这条LR函数。因此新3K、多数据集和混合训练**不能只改max_steps**。

在保留旧run精确行为的前提下，扩展：

1. 新manifest reader：旧TRAIN100、新T分片和typed S_cmd分别读取，统一到同一模型batch；原始来源仍可追溯。
2. 配置化LR/sampler：steps、warmup、统计、数据版本、domain配额及随机流明确绑定；保存恢复后下一个visit一致。
3. Mixed sampler：逻辑effective32先定24/8身份与随机输入，再分microbatch，不能让microbatch8改变域比例或噪声。
4. 空间batch：正确view mask、teacher identity和GPU特征；alpha=0的公共参数梯度须与FM路径对应。
5. 数据准入：从已完成成功尾段、编码与分片读回，到真实TRAIN reader可读取；不要把所有工程数据重新命名为D100。

这是有限实现清单，不是新增逐函数审批；复用已通过的H200梯度/恢复资格，不重复同一全套检查。源准备仍不阻塞目标首策略。

## 9. 后续移动与RL的边界

新站位数据B100（5站位×2任务×10组合）先用于冻结策略评价和有限监督适配；导航后任务NAV100（10路线×2任务×5布局/重启）只有实际行走后端和稳定接管验证后采集。此前固定臂源数据没有base标签时mask该损失，不补假零。

任务空间EEF目标在固定task/world frame表达；需要机身frame时由执行层按当前实测底座位姿转换，不读取未来底座真值。导航到位后操作与连续移动中操作分开。高层不直接输出腿关节，WBC保持本轮冻结。

本轮是示范监督+空间对齐+交互式专家纠正，**不需要reward/critic/在线RL**。原始失败/接触/终止日志保留，只有出现可执行策略上的明确接触优化问题、可靠奖励和重置，才另立RL。

## 10. 这份说明的完成标准

不是“多写了一份计划”。执行端下一批应交付：R1原生评价增量；至少一个可读的新目标TRAIN数据版本；五任务源池的真实revision/episode清单/语义结论；一个按此配方训练并可普通RGB部署的checkpoint及逐条件闭环结果。

本说明不增加V4核心矩阵预算，五任务S1000是原S1000的具体化。新增源数据原始审核、下载、重新生成成本如实单列；当前并未执行这些工作。旧V3训练矩阵、额外D100空间臂、源混合矩阵、移动分支和历史实验分别计量，不以一个总数重复计算。

### 固定实现与配方来源

- [现役执行PR #382](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/382)
- [旧LR与采样实现](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3195f0eb664ed98db7f76ad0643ed1d742472a7b/src/go2arx5_vla/sla_adapter/config_v1.py)
- [GPU驻留更新实现](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3195f0eb664ed98db7f76ad0643ed1d742472a7b/src/go2arx5_vla/sla_adapter/runtime_v3.py)
- [V2完整数值配方](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b07936899b10f196653f617f95c5cd8b124e5e99/docs/protocols/SLA_GO2_ARX5_SCALEUP_V2_20261112.json)
- [RoboTwin move_pillbottle_pad](https://robotwin-platform.github.io/doc/tasks/move_pillbottle_pad.html)
- [RoboTwin move_stapler_pad](https://robotwin-platform.github.io/doc/tasks/move_stapler_pad.html)
- [RoboTwin place_mouse_pad](https://robotwin-platform.github.io/doc/tasks/place_mouse_pad.html)
- [RoboTwin place_empty_cup](https://robotwin-platform.github.io/doc/tasks/place_empty_cup.html)
- [RoboTwin place_object_stand](https://robotwin-platform.github.io/doc/tasks/place_object_stand.html)
- [排除的双臂place_can_basket](https://robotwin-platform.github.io/doc/tasks/place_can_basket.html)
