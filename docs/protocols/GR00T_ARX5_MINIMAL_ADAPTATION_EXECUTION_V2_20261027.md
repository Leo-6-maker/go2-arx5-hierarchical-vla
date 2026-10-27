# 最小 ARX5 适配：固定平台、固定512终点与执行合同 V2

2026-10-27，Asia/Singapore。用户在本线程活动目标中直接要求沿 V1 实施，并提供本轮收紧后的研究与部署要求。本文件登记该方向及实际 CPU 实施进展，不是新 GPU/PBS/物理资源采用回执。V1 原文与全部历史科学结果保留。

## 1. 唯一执行路线

| 平台 | 当前固定职责 |
|---|---|
| Vanda | 原始父模型与参数契约的有限离线模型诊断、E1/E2微调、完整训练状态保存和独立审计 |
| CoreLab 4080S | 原 MuJoCo 源域、模型推理、双RGB、原driver/relay/反馈/ACK/STOP/S1/评分和完整闭环Eval |
| AMD/管理侧 | 数据清单、CPU合同测试、已授权有限静态准备与结果整理 |

停止继续开发 Vanda 同机闭环 Eval、5080 迁移或其他并行评估路径。旧源码、失败回执、已收费输出及 UNKNOWN 不删除、不改判；停止开发不等于重放旧收尾或回滚账本。连接故障由管理员处理，不继续用研究包装层绕过 SSH/NFS/挂载问题。

最近一次人为请求下的真实连接检查：AMD可登录；4080S与5080-1完成SSH握手后等待liuyu-nfs公钥回复超时。5080-1改用内网IP且绕过客户端配置仍同样超时。这些是认证观测，不是服务器/GPU/任务终止证明。师兄截图使用不同账号，不证明liuyu-nfs已恢复。本阶段不再重复同类探针。

## 2. 科学定义和两项结论边界

主问题：在**同一原始预训练权重、目标类别初始化、12条示范、逐cursor曝光、推理和执行条件**下，只更新目标本体编码/解码参数，与进一步更新共享动作策略相比，闭环能力有何差异？

- E0：原始父模型的有限离线诊断参照，不宣称已具备ARX5零样本能力。
- E1：目标类别三个组件内七个CategorySpecificLinear的14个W/b切片参数。
- E2：同一14切片，加共享DiT、vlln与视觉语言self-attention；VLM及共享position embedding保持冻结。
- CORR/A/B/C50/D25/GEN1只作历史参照；不能替代同父、同数据、同曝光的E1/E2主对照。

E1也从T1/T2示范学习任务，不能称纯机械接口转换。T1/T2成功仅能支持这两项适配任务；跨任务能力需本项目适配集没有出现的任务家族另测。E2比E1差也不直接等于灾难性遗忘。

## 3. 已实现的参数合同及其严格边界

新冷态库 `scripts/pi0294/generalization/minimal_adapter.py` 不含模型加载器、调度器、launcher、认证或账本写入；未部署到任何运行capsule。实际模型先经原入口/资源/身份资格加载，再由现有Trainer冷接入。本库本身不赋予执行资格。

1. `TargetCategoryLinear` 保留完整冻结类别表，只把目标类别W/b变为独立optimizer参数。七个leaf精确列名；不靠整表梯度mask阻止AdamW衰减。内部132 padding不裁剪，目标类别也不伪装成其他机器人。
2. `ParameterContract` 明确E1/E2白名单，冻结其他参数；拒绝重复安装、非法类别、改变白名单、非零state dropout和错误恢复身份。共享position embedding两臂均冻结。
3. 两臂采用同一 `MATCHED_ACTION_HEAD_AND_BACKBONE_EVAL_WITH_AUTOGRAD_V1` 模式：Trainer重新调用train后，forward pre-hook对action head及backbone重新assert eval。该规则关闭两臂的模块dropout与可变运行统计，不关闭autograd；E2的共享权重仍可更新，E1输入适配器梯度仍穿过冻结共享网络。实际大模型必须验证原forward/重计算路径，不以小模型测试代替。
4. `matched_flow_rng` 按pair seed、optimizer step、accumulation microbatch分别绑定CPU Beta与选定设备的flow随机流，并恢复外部RNG；不把同一个总seed当作两臂匹配证明，也不让16个microbatch复用同一flow draw。梯度重计算时作用域必须覆盖backward。CUDA分支尚未实际验证。
5. 完整训练恢复仍走原Trainer：model/optimizer/scheduler/RNG/sampler及processor/R0完整保存。`restore_parameter_state`只负责额外核对arm/类别/参数合同及严格参数恢复，不冒称完整Trainer已接通。
6. `export_official_state`把类别表合并回原W/b键，供原模型结构严格重建；完整推理交付还须封存config、processor、类别映射、R0、源码、checkpoint摘要与评价计划，并证明真实模型重建前后输出等价。不能只发adapter文件给旧loader。

CPU实际证据：30项通过。其中19项参数/AdamW负对照/梯度/模式/恢复测试，9项数据/逐cursor合同（含实际冻结清单复现），2项直接使用官方51d4固定gitblob的原类别模块，测试用小hidden宽度，保留H40/132维。后者不是完整GR00T、预训练模型、CUDA、科学query或闭环资格。测试原实现了AdamW梯度mask导致其他类别被衰减的负对照；完整CPU优化器/scheduler/RNG/cursor重载与连续两步结果逐tensor一致。

冻结类别表作为buffer完整保留，另有目标切片参数；转换期间也可能暂时保留旧表副本。CPU测试不能证明真实峰值显存或固定加速比例，必须实测且纳入资源上界。

## 4. 实际12示范与曝光已冻结：元数据，不冒称新bank资格

Vanda只读取得原manifest SHA `b8ddcb9808e6dfac4f3ac5fa5a4828128fa9644a330c17d1c57bf833ce5889df`：158个父条目，其中126 TRAIN、32 DEV，原R0 SHA `fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7`。本次只读取JSON元数据，没有读取模型tensor、RGB或重采/重导出。

原TRAIN库有F0/F1/F2三类几何、30组完整T1/T2配对。初版CPU选择器误把GEN1的四布局概念带入原库，实际输入明确拒绝；未生成选择结果或训练。现按原库真实三类修正，规则在选样前固定：seed20261019，组内SHA256排序，F0/F1/F2轮转两次。不是依据模型结果筛选。

| 几何谱系 | 完整示范 |
|---|---|
| PI0294/TRAIN/F0/005 | T1、T2 |
| PI0294/TRAIN/F1/009 | T1、T2 |
| PI0294/TRAIN/F2/010 | T1、T2 |
| PI0294/TRAIN/F0/000 | T1、T2 |
| PI0294/TRAIN/F1/002 | T1、T2 |
| PI0294/TRAIN/F2/005 | T1、T2 |

清单仅引用原有合格原库，未加入PV1/C1或GEN1。每父601个H40窗口，共7212个**重叠窗口**；独立几何仍是六组。原始manifest、父numeric/raw/samples/physics/source身份均在机读清单内绑定。运行前仍须原native payload与processor/完整bank核验，不能把新清单当native准入。

`minimal_study.py`生成不含arm变量的固定窗口置换：512步访问16384次，条件1024总访问32768次，两臂逐cursor消费同一parent/anchor/task/group。元数据panel SHA `d603324fee6cc69f5b488b47cfaf13dd4bc1feb8d8dc6411a1a739822cfff74d`；首16384序列 SHA `98e44f2602a20fa3b2853ae5e5c981ae3b5383dfb4b329f6134cc2b31d5dd068`；全32768序列 SHA `cb73066ac016eca8aec659b66c87059c7edd203f3af8992379267f4a6c454658`。实际训练消费0，不能据序列已生成报更新。

R0保留，不重开P0；这属于固定统计校准下的12示范研究，不是只接触12条机器人信息的严格数据效率结论。

## 5. 原始父、零步等价和有限诊断的下一项合同

已找到历史真实原始父文件核验回执：`nvidia/GR00T-N1.7-3B` revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`，七文件摘要与当前R0明确区分。它是历史Hopper原始文件身份，**不是fresh Vanda文件核验或目标类别内容证明**。

在新合法allocation前后按角色推进：

- 元数据/CPU：固定实际源文件、config/processor、类别映射与目标切片来源；原始checkpoint、复制初始化、新初始化三者不能混称。默认两臂从同一原始checkpoint的类别10切片开始；其预训练历史UNKNOWN，不断言该行未训练。
- 有限真实模型资格：同TRAIN输入、原处理器、相同H40/noise，包装前后与E1/E2零步输出一致；实际检查horizon、position、padding/action mask，禁止静默missing keys或裁掉其余动作维度。
- 用实际loss验证每个目标切片有梯度、有更新，E1全部冻结参数/buffer不变，E2仅白名单变化；完整保存重载、optimizer/scheduler/RNG/sampler和推理包重建前向一致。
- 预先固定小型诊断面板，不以先完成432次生成为闭环前置。语言仅用两任务都合法的接近/接触前状态；视觉变化需真实一致观察，不能图像与物体状态错配。动作方向/姿态/夹爪语义和原拒绝一起报告，L2不是唯一指标。
- common home是独立初态版本，不自动证明视觉泛化；准备不阻塞原LEGACY_MATCHED首轮Eval。不在本阶段构造新的物理样本。

## 6. 固定终点、执行顺序与条件续训

4080S恢复前默认不启动E1/E2完整训练，也不堆积其他候选。已有C50、D25、GEN1均1024完整审计/once结算关闭，行为各0/24；不重训、不重stream已审计tensor或重结算。

恢复后先核真实硬件/显存、进程、磁盘/挂载、旧UNKNOWN control close与canonical LOCAL。LOCAL若不同先查原因，不覆盖历史。用原固定环境和实际反馈/ACK/STOP/零积分双RGB及原600秒80cycles资格恢复原链，不升级或临时量化/缩图/减推理步数。

1. 4080S先C50原N294101/24，再D25同24；Vanda并行完成新适配的有限真实模型资格。旧模型成功率不是新研究的额外GO门，测量链有效才是条件。
2. Core接续GEN1原24；Vanda在完整有效scope与资源采用后完成E1和E2各512步。固定1024 scheduler日程、batch32、原seed/LR/warmup；512是该1024日程的中间端点。可在一份预先批准且容量/walltime足够的PBS中串行两臂，完整保存/卸载E1后从同一原始父fresh加载E2，不继承权重或optimizer。
3. 两臂512均完成并完整审计/once关闭后，同一4080S评估器各原24条件，保留全部有效失败；不依据先完成臂成绩改另一臂或选择各自最好checkpoint。
4. 首轮结果有信息价值且另有合法范围时，两臂各从自己的完整512共同续至1024，恢复原optimizer/scheduler/RNG/sampler，不只延长落后臂，不重启学习率。不自动新增LoRA、多seed/LR网格或扩采。

固定512两臂仅回答该参数范围和受控优化设定下的差异，不宣称各方法已最优。分别报告T1/T2完整成功、参考/S1拒绝、完整任务失败、原生安全、实际trainable numel/optimizer存储/峰值显存/GPU与端到端时间；主差异E1−E2按几何配对，相关窗口/噪声不是独立样本，未显著不等于非劣。

## 7. 资源、时钟和仍未完成项

最新闭合GLOBAL used14786更新、PBS34，已采用cap16384更新、PBS37。matrix3和最终20GPUh/40GiB保护保留；原C/D/GEN待Eval、全部旧失败/current/peak/费用不挪用。首两臂512累计15810在更新维度内，但不代替其他资源准入；完整两臂1024累计16834超更新cap450。PBS34+matrix3已占37，无自由新PBS。切回Core评估需按实际全剩余工作形成一次分阶段全维复算，不沿用V1约32.53GPUh为已批准额度。

本次无native GPU/PBS/模型调用/构造/积分/训练/ledger写，CPU准备不静默扩cap。用户同范围普通工程自主权限保留，但不把它解释为无限资源。原T_CD1792932749.8133287、stop-new-training1793451149.8133287、deadline1793537549.8133287不重置；管理员恢复也不自动延长。若窗口或资格不够，保留NOT_RUN，不能标完整goal完成。

当前最小未完成项依次是：原始父在Vanda的实际身份及类别/config合同；原Trainer对新512端点/切片状态的冷接线；有限真实模型资格与完整资源scope；4080S管理侧恢复及原测量链；72条旧端点评估及48条新端点评估。前者可继续合法冷准备，后者不能靠CPU通过冒称完成。

柜门只在基础配对结果后作为单一新任务迁移环境；先机械可行与评分资格，再测固定模型。书架保留设计。不采柜门训练、不自动LoRA、不改WBC；EEF10不补零冒充BASE或core64。wholegoal保持未完成。

## 8. 实施增量：原始父实际字节与固定512 Trainer冷接线

本节是后续实际增量，前文30CPU/父身份待核描述保留为上一批历史，不把CPU前进写成native部署。最新实际报告为 `ARX5_MINIMAL_RAW_PARENT_AND_FIXED512_TRAINER_CPU_JOIN_ACTUAL_20261027.json`，原始父证据为 `ARX5_RAW_PRETRAINED_VANDA_CONFIG_AND_CATEGORY10_ACTUAL_20261027.json`。

- 原始父已在Vanda真实缓存定位。五份非权重文件摘要全匹配历史原始2fc revision；H40/132维/32类别/4步真实确认，category10的14个W/b已有实际字节SHA。两完整shard尚未全流核验，仍不能放行模型。原statistics实际4293096bytes，早先错误2MiB上界导致只读拒绝，改为有界分块hash后仍匹配原始c97b；没有实际统计污染或缓存修复。
- 原始配置的state/DiT/VL dropout都是0.2；新研究显式继承原ARX5 processor的state0/无增强/完整256RGB，两个动作head都采用相同eval-with-autograd规则。原始position embedding为1024×1536、两臂均冻结，不裁132padding。原映射没有命名category10，但物理槽存在；只证明槽来源，不证明其预训练历史。
- 头部推导目标切片10180740参数，E2合计1303340164；均非真实加载后的可训练参数审计，更不是显存或加速测量。
- `minimal_dataset.py`直接继承原NativeDataset，只选已冻12父；原hash、原numeric/RGB验证、原processor和标签保持。实际158条manifest/R0在隔离CPU临时输入中构造该reader通过，length7212、逐cursor一致；未初始化processor或读取RGB/numeric。
- `minimal_training.py`从原成熟recipe继承优化器/数值和原Trainer/loss/loader/scheduler，显式绑定raw父清单0e493187、E1/E2身份和固定panel。Trainer的max_steps保持1024，追加只读boundary observer在原owner commit callback之后核更新数/pending/unknown并于512保存停止，另有training_step拒绝第513步。无owner的真实效果回读不能启动这条新链。
- 全保存额外绑定参数白名单、冻结基线、所有参数与buffer的实际fingerprint；框架重载后必须逐项一致，防止宽松加载静默漏掉切片。optimizer/scheduler/RNG/sampler及processor/R0仍由原完整保存流程负责。
- 48CPU通过（此前30项+18项新增）；新的框架分发测试明确使用CPU fixture，本机没有Transformers/Accelerate，不冒称原框架或真实模型已通过。实际注册owner、有限模型资格、资源采用、两臂训练与所有Eval仍未执行。

## 9. 实施增量：真实接触前语言输入、原始加载器与首阶段全资源复算

最新报告 `ARX5_MINIMAL_RAW_LOADER_LANGUAGE_PANEL_AND_FULL_RESOURCE_CPU_ACTUAL_20261027.json` 记录76项CPU通过（新增28项，含6项全资源检查），无skip；真实完整GR00T、Transformers Trainer、CUDA和科学执行仍未运行。`minimal_model.py`只是原loader/processor/native-job路径的冷态消费者：新scope与same-job原owner release、完整七文件摘要、当前stat、原R0、严格loading info、H40/132/四步均须实际通过。尚无原生注册owner为此scope发release，库代码不构成GO。

两份完整raw shard的首次只读全流核验超过原40秒观测界，后续精确原进程句柄不存在，但没有最终摘要回执。保留未知结果与CPU费用不确定性，不将缺回包称作hash成功，不自动重复整份权重读取；后续应由合法作业原owner在原生命周期内完成并保存七文件核验。本次不延长RPC、不新增后台auditor。

已从六组冻结TRAIN几何读取真实anchor16双RGB/EEF10及原配对指令，六组第一次接触均在anchor29。按同观测、两合法任务指令、两固定noise seed，冻结每模型24次语言生成。72次是三模型语言部分，另有零步/兼容和末端TRAIN探针；首阶段训练侧全部生成上界147。输入不是抓住物体悬空后改推物，counterfactual语言从不成为BC标签；不重新渲染、采集或伪造视觉干预。语言反应差异不是任务成功。

完整首阶段草案为 `ARX5_MINIMAL_FIRST512_FULL_RESOURCE_PACKAGE_SCOPE_DRAFT_20261027.json`。它在captured实际GLOBAL993ea3f7与历史LOCAL7cea上复用原完整forecast，保留C50/D25/GEN1待Eval、原失败/current/peak、matrix3和最终20GPUh/40GiB。历史LOCAL仍PENDING，不是假fresh或原生准入。

| 维度 | 已采用上限 | 草案全包上限 |
|---|---:|---:|
| GPU小时 | 121 | 147 |
| PBS次数 | 37 | 39 |
| Vanda站点CPU小时 | 213 | 265 |
| Core CPU小时 | 320 | 380 |
| 总传输GiB | 332 | 455 |
| GPU站点输出GiB | 1176 | 1399 |
| Core输出GiB | 477 | 550 |

这些是**尚未采用**的精确变更；其他上限/原时钟不变。新增完整工作上界33.57GPU小时，但已有余量抵扣后总cap只需增加26小时；不把上界称必须花费。两臂保留原1024训练成本包络而不事先承诺512或少参数减半耗时。拟用一份16.5小时walltime PBS串行独立raw初始化的E1/E2资格与512，另保留一次严格已闭合零模型/查询/更新/构造/积分的冷工程替换；不重放UNKNOWN或有效行为失败。

实际资格每臂允许一次真实原Trainer更新和完整保存恢复验证，再从raw父fresh开始正式512。因此本阶段计1024正式更新+2资格更新=1026，累计15812仍在16384内；这两次费用不会擦除。该一更新资格路径仍待原owner/Trainer冷接线与真实验证，不能按已完成计。若后来共同续1024，计入资格后至少16836更新，超旧上限452，仍须条件范围，未预留或自动批准。

每臂保留70GB完整状态上界（资格与512状态均保留），每份推理包保守16GiB。沿原Eval向量只调整其中已有模型复制槽位，不另加第三份模型、不借旧paidIO或历史删除退款；实际保存/显存/速度仍须量测，超界即不得继续。两Eval保留原26start/600秒80cycles资格与所有原失败/S1/评分。Vanda预算中原Eval侧保守CPU上界不擅自扣减，Core神经CPU显式保留。

首阶段资源采用、真实原始模型资格、Core管理侧恢复与原测量资格仍是未完成项。4080S未恢复前默认不启动两臂完整训练；即使以后批准资源，也不能用批准替代真实测量链资格。没有新request、预付、qsub、GPU、模型、物理或账本写入，wholegoal未完成。

## 10. 七项上限实际采用与零学习率资格修正

人类直接答复“批准仅这七项上限调整”。固定批准文件为 `ARX5_MINIMAL_FIRST512_SEVEN_CAP_DIRECT_HUMAN_GRANT_20261027.json`（16f608c7，SHA256 6a9830b8），原草案和批准字节保持。源a1d4df58已在原GLOBAL通过原onceIO一次CAS采用，独立reconcile为CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT；GLOBAL从993ea3f7变为d3e48b11。仅新增七cap authority marker及revision，旧费用、used/reserved、行、package及历史逐项相同；实际仍14786更新/4511查询/PBS34。Core canonical LOCAL未写、仍历史PENDING，未创建任何新科学request/GPU/PBS。完整新工作包尚未预留，native admission仍false；不能把cap采用称整条执行链已部署。

首次包含完整队列的只读snapshot超过原40秒界，没有发出写操作，不推断远端终止。之后小型实际GLOBAL/原CLOSED lease与exact actor读核通过；它明确没有fresh fullqueue资格，不用于GPU放行。原生采用前readonly preflight CPU10.196s、CAS与独立回读通过；已知metadata CPU合计28.956s，首次超时CPU保持未知而非零，归入后续同包一次控制费用分类，不退款或假称已结算。原所有mutation UNKNOWN继续保留。

资格检查进一步发现：原 `LambdaLR` 的step0因64步warmup使用LR0。一次optimizer调用只改变优化器/调度状态，不能证明权重真的更新。现有冷态 `minimal_training` 增加明确的qualification-only会话，仍调用原Trainer/loss/optimizer/scheduler；每臂最多两次调用/64次样本访问，第二次使用原1e-5/64学习率，不改warmup或正式科学日程。第2次原backward后记录真实target梯度，原commit后完整保存时检查各target组件实际改变及全部冻结tensor不变；缺梯度、非有限或无更新不能PASS。

资格保存标记不属于开发端点，拒绝用它续正式512；同一已消费模型/contract也不能再次当fresh formal初始化。正式E1/E2仍各从raw父fresh optimizer开始512步、1024日程、原逐cursor。真实框架的optimizer/scheduler/RNG/sampler恢复和输出重建仍必须在合法allocation验证，CPU hooks并不代替这些资格。

计费完整保留每臂2资格调用：首阶段1024正式+4资格=1028，已有14786加本阶段为15814，仍≤16384；两臂条件续至1024时为16838，至少超旧cap454，未自动授权或预留。此前草案的1026/15812是一次调用资格的历史计划，不改原批准文件或重做cap采用。修正后的全未来复算 `ARX5_MINIMAL_FIRST512_POST_CAP_ADOPTION_AND_WARMUP_FUTURE_ACTUAL_20261027.json` 使用实际d3e48b11，七项新cap及其他旧cap均无缺口，GPU/CPU/PBS/IO包络不增加。

本批95项CPU通过，零skip，包括真实小CPU AdamW+原LambdaLR的首调用无参数变化负对照及第二调用改变参数、冻结/梯度和资格状态拒绝；仍没有真实GR00T/CUDA/新科学query/训练/物理。4080S管理侧恢复、原测量链、新scope原owner/入口/完整预留与真实模型资格仍未完成，wholegoal未完成。

## 11. 完整首阶段GLOBAL一次预留，Core镜像和执行仍待准入

源52b1ccc9以原onceIO实际预留完整包，独立reconcile为CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT，GLOBAL d3e48b11→d9a31457。新增包及对应holds，原used14786更新/4511查询/PBS34、旧费用/行/package/授权不变。原cap marker的NOT_YET_RESERVED是采用上限时的不可变历史；当前状态由execution_package表达，不回改历史marker。

新包上界120840 GPU秒/1028更新/4341查询/2PBS及完整CPU/输出/传输向量，当前GLOBAL总reserved149700 GPU秒/1028更新/6438查询、站点CPU529613秒/输出281827996472bytes。Core计划保存在同一包remaining_resources，canonical LOCAL仍未读写，这不是GPU分配或发生过的更新。

旧GLOBAL GPU hold28860秒，历史Core投影hold57720秒；新120840秒名义hold实际使max投影增加91980秒。只抵扣91980，不能减全名义量而少保留28860旧Core工作。真实captured输入证明全部维度预留前后预计开销相等，旧待Eval/matrix3/final保护不减，无新cap缺口。

五会话计划为E0诊断41query、两臂各2调用资格（各17query/64访问）、两臂各512正式（各36query/16384访问），正式均fresh raw。可变effect/session与新typed request、原owner/profile/manifest/lease/registry、fresh GPU/queue/quota及真实模型资格尚未接通，不冒称全链已后台运行。

108项适配CPU与54项旧准入回归、20subtests通过，不是CI、真实GR00T/Transformers/CUDA或闭环资格。首次本地报告生成漏src路径，import即失败，未写报告或触碰native；修正路径后完成，失败保留。无新GPU/PBS/模型/query/优化/物理。

Core恢复后先精确核UNKNOWN和canonical LOCAL、同步原账本并恢复原测量链。新scope后续按用户目标绑定一Vanda训练与一Core评估的并行角色，累计真实GPU秒，旧scope和其他任务不可借用；当前未部署新并行准入。正式新训练仍须测量链有效，旧模型成功率不是GO门。wholegoal未完成。
