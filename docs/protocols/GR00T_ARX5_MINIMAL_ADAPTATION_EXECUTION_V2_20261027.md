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
