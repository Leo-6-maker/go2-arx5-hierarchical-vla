# SCALE20K 官方配置复核与 Go2+ARX5 后续适配计划

2026-11-19，Asia/Singapore。研究评审与后继路线；本次新增训练、下载、生成、原生请求、积分和真机请求均为 0。

## 1. 当前决策与现役边界

**SCALE-MIX20K已完成独立恢复并提交前7次真实更新，应继续同一连续优化。多源长步数训练不再等待 Teacher、D5、J 或某个闭环成功率通过。** 接下来的关键实绩是非零完整检查点与200更新前缀，然后沿同一模型、Adam、RNG、数据银行和学习率历史推进。

PR418 仍绑定原计划 `926226914a0c08ee1ebb143e0edea1434728107b`。本评审不改变其 20k/32T+48R+48L、W 宽度项、模型范围、数据版本和 200 更新首前缀。首 200 步均在既定 500 步 warmup 内，不能把这个前缀的低成功率设为科学停止门。数值、数据或恢复错误应具体修复并保留失败，不用反复新开同名作业。

本文初审依据PR418 f7ec607e和PR417 3ffd8af2，并在发布前补读了17:35 SGT的新快照：PR418 `55d0111c0e433ed6ae2cc42ac9af2a3d97d1dfd5`、PR417 `05dfe6d42500e62c3e139222709ae24213235924`。这些是公开发布快照，不是现场实时轮询。旧 D4 按用户调度暂停在28/48；旧20条未启动，不因本评审自动重开。当前服务公开的绝对截止为2026-11-19 20:54:01 SGT，不能等同已落实整个20k的长期资源。

- [现役规模计划](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/926226914a0c08ee1ebb143e0edea1434728107b/docs/protocols/SLA_SCALE20K_DIRECTION_20261119.md)
- [本评审接续prompt](../handover/2026-11-19/START_SLA_SCALE20K_NEXT_ADAPTATION.md)
- [GPU资格实绩](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f7ec607ee9b0395ae74623977352de62cbe7e6a0/reports/WM02/sla-scale20k-20261119/CORE_GRADIENT_QUALIFICATION_ACTUAL.md)

## 2. 当前已完成什么

### 数据与参数

| 域 | TRAIN示范 | 窗口 | 当前覆盖 |
|---|---:|---:|---|
| Go2+ARX5 | 23 | 1768 | 20 HOLD、3 PLACE |
| RoboDojo ARX X5 | 48 | 50979 | 6任务 |
| LIBERO | 1261 | 209146 | 30任务 |
| 公共源合计 | 1309 | 260125 | 36任务 |

当前已不是原来的几十条源数据技术试验。但各域差异很大：公共真实源仅48条；目标3条PLACE都来自F1、2场景、1采集谱系，只含blue-left、red-right。目标新数据须优先解除颜色与方向的混淆，并补F2 PLACE。

三域资格阶段真实计算32/48/48个梯度窗口和4次完整十步H40，该阶段Adam更新为0。**17:35新实绩已确认独立step-zero恢复、7次优化更新、224目标+336RoboDojo+336LIBERO访问，共896窗口和28次训练内完整H40；4次资格生成另计。** 快照仍未封存非零前缀端点，持久更新账本不等同已经可以恢复step7的完整模型状态。

完整动作专家427,932,672参数、VLM LoRA27,869,184参数、每域I/O2,165,792参数，共462,299,232活动参数。140个模型/Adam/RNG数组已保存并在独立进程恢复。早期报告的11.94M LoRA和10.46GiB下界被后来的修正取代；最新常驻模型+Adam理论下界为10.61GiB。首7更新已测得JAX实际分配峰值19.40GiB，此值仍不等同所有进程开销或未来任意microbatch的峰值。

新增来源：[首批优化与吞吐](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/55d0111c0e433ed6ae2cc42ac9af2a3d97d1dfd5/reports/WM02/sla-scale20k-20261119/CORE_FIRST_TRAINING_UPDATES_ACTUAL.md)、[逐更新账本](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/55d0111c0e433ed6ae2cc42ac9af2a3d97d1dfd5/reports/WM02/sla-scale20k-20261119/CORE_FIRST_TRAINING_UPDATES_ACTUAL.json)。

来源：[数据](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f7ec607ee9b0395ae74623977352de62cbe7e6a0/reports/WM02/sla-scale20k-20261119/DATA_AND_RUNTIME_ACTUAL.md)、[目标与采样](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f7ec607ee9b0395ae74623977352de62cbe7e6a0/reports/WM02/sla-scale20k-20261119/TARGET_AND_CORE_HANDOFF_ACTUAL.md)、[最终资格机读证据](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f7ec607ee9b0395ae74623977352de62cbe7e6a0/reports/WM02/sla-scale20k-20261119/CORE_GRADIENT_QUALIFICATION_ACTUAL.json)。

### D5改变了问题判断

| 固定TRAIN预测面板 | B500 | W | T |
|---|---:|---:|---:|
| 宽度MAE / mm | 4.47 | 1.51 | 0.96 |
| 窗口内闭合跨越 | 1/20 | 19/20 | 19/20 |
| 跨越误差不超过100ms | 0/20 | 1/20 | 10/20 |
| 完整H40通过 | 95/96 | 85/96 | 83/96 |
| H40与含真实上一命令速度门同时通过 | 92/96 | 78/96 | 78/96 |

96来自48个固定TRAIN窗口各2份噪声，20是其中有闭合标注的预测实例，不能当独立episode或闭环成功。相同500步、相同目标输入下，物理监督显著改善了闭合，所以规模不足不是唯一机制。

由公布总数可精确分解为：B的H40拒绝1、H40过而额外速度门拒绝3；W为11+7；T为13+5。尚缺D5逐条首因，不能把这些拒绝全称为夹爪速度或边界加速度问题。应重用已保存96输出，通过原分析器列出线速度、角速度、夹爪速度、线加速度、角加速度的首因/位置/峰值，并计算“跨越且时序合格且两门通过”的联合数；这项分析不需要新模型生成。

旧D4还存在接近问题：两个F1布局8条HOLD的240,008帧中，目标手部接触为0，方块中心到双垫中点最近仍69.6–104.8mm。D5能闭合不保证闭合时已经到达物体。该距离是几何诊断，不替代原抓持评分，也不能据此直接断言坐标系bug。

D5最新Core复验为W/T合计32/32归一化输出逐位一致，32/32原门判定一致；物理解码严格逐位仍0/32，最大数组差3.33e-16，该历史严格结果保留。修复原生监督器SciPy依赖耦合后，GRASP24已在GPU1与SCALE并行启动；公开科学原生分母仍0/24，不把E0接口混入。[新Core/派发实绩](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/05dfe6d42500e62c3e139222709ae24213235924/reports/WM02/sla-d4-next-plan-20261119/D5_CORE_RGB32_GRASP24_DISPATCH_ACTUAL.md)

来源：[D5](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/17845a1b1e5a91f4e962ffae894470bc7e7f2622/reports/WM02/sla-d4-next-plan-20261119/D5_VANDA_AND_J0_ACTUAL.md)、[D4接近](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3ffd8af21ed156f5878719d51d0becf418084f4d/reports/WM02/sla-d4-next-plan-20261119/D4_F1_HOLD_APPROACH_ACTUAL.md)、[D4暂停终态](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3ffd8af21ed156f5878719d51d0becf418084f4d/reports/WM02/sla-d4-next-plan-20261119/D4_PAUSED_MATCHED_PREFIX_ACTUAL.md)。

## 3. 官方后训练配置：不能只照抄步数

本次官方master固定在`ea193bf3f345ba222cbac41170ffbc9e05474827`。相对此前18896e75，RoboTwin将`vla_adapt_layers_align`从False改为True，另有README文字变化，规模与优化参数未改变。此官方更新不意味着现役任务要拉取新代码。

| 配置 | LIBERO-Plus | RoboTwin | Piper真机示例 |
|---|---|---|---|
| 更新×全局batch | 30k×512 | 30k×256 | 20k×64 |
| H | 10 | 50 | 50 |
| 梯度累积默认 | 1 | 1 | 1 |
| LR | warmup10k到5e-5，之后peak=end=5e-5 | warmup1k到2.5e-5，30k降到2.5e-6 | 继承同一30k schedule，但只训20k |
| 训练范围 | 默认不冻结VLA参数 | VLM LoRA、完整action expert，视觉可训练 | 默认不冻结VLA参数 |
| State dropout | .25 | .25 | .25 |
| EMA | .999 | .999 | .999 |
| 统计 | q01/q99 | q01/q99 | q01/q99 |
| 空间对齐 | .5，第12层 | .5，最新自适应聚合 | .5，第12层 |
| 主要动作 | LIBERO原生 | EEF/SO(3)相对动作 | joint delta |
| 策略FSDP设备配置 | 8 | 8 | 1 |

三者AdamW默认都是beta1=.9、beta2=.95、eps=1e-8、weight_decay=1e-10、global gradient clip=1.0。不能套用常见框架的beta2=.999或weight_decay=.01/.1来描述官方。设备配置不是最低显存保证，不能从表直接推导必须申请8张80GB。

官方EMA每个优化步更新，checkpoint将EMA导出到`params`，默认推理加载该导出；raw参数与优化器保留在train_state。因此EMA改变了评价端点，并非仅日志平滑。

官方state dropout在tokenize前随机省略state-token段。本项目的控制器真实反馈和上一命令仍保留。当前SCALE已经实现25% state-token dropout，无需把它重复安排为一个尚未实现的前置。

官方混源loader的抽样概率与`len(dataset)*weight`成正比，`weight=1`不等于域等比例。MixedLeRobotDataConfig使用共享统计，是在来源先转成统一表示的设计下成立。我们保留不同native动作和域私有I/O，使用独立统计，是明确的新实验设计。

固定官方来源：[LIBERO](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/ea193bf3f345ba222cbac41170ffbc9e05474827/SLA/src/openpi/external_config/train_libero_plus.py)、[RoboTwin](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/ea193bf3f345ba222cbac41170ffbc9e05474827/SLA/src/openpi/external_config/train_robotwin.py)、[Piper](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/ea193bf3f345ba222cbac41170ffbc9e05474827/SLA/src/openpi/external_config/train_real_world_handover_vegetable.py)、[AdamW/LR](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/ea193bf3f345ba222cbac41170ffbc9e05474827/SLA/src/openpi/training/optimizer.py)、[冻结](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/ea193bf3f345ba222cbac41170ffbc9e05474827/SLA/src/openpi/models/pi0_config.py)、[训练/EMA](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/ea193bf3f345ba222cbac41170ffbc9e05474827/SLA/scripts/train_align.py)、[EMA导出](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/ea193bf3f345ba222cbac41170ffbc9e05474827/SLA/src/openpi/training/checkpoints.py)、[推理加载](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/ea193bf3f345ba222cbac41170ffbc9e05474827/SLA/src/openpi/policies/policy_config.py)、[混源配置](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/ea193bf3f345ba222cbac41170ffbc9e05474827/SLA/src/openpi/training/config.py)、[采样](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/ea193bf3f345ba222cbac41170ffbc9e05474827/SLA/src/openpi/training/data_loader.py)。

### RoboDojo版本与memory

论文RoboDojo-Real是40k×512，公开SLA_Memory配置是60k×512，README描述的发布checkpoint又写batch256且history在current views之后，而当前代码新训练默认history在前。这些是不同复现身份。

公开memory采用3当前视图和9历史头相机帧；buffer按相机/控制时钟维护并每episode重置，不能按模型查询次数计时。该公开集成未采用主仓同一`train_align.py`入口，不能把它与所有论文真机实验合并描述。当前闭合和接近问题不需要先添加memory；遮挡、跨阶段判断与长程指令出现明确需求后再加入。

来源：[论文§4.2](https://arxiv.org/html/2611.08373v1)、[RoboDojo README](https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/README.md)、[公开real配置](https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/openpi/src/openpi/external_config/train_robodojo_real.py)、[memory实现](https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/openpi/src/openpi/policies/robodojo_memory.py)。

## 4. 保持现役配置，并把差异说明白

| 项目 | 当前SCALE实现 | 建议 |
|---|---|---|
| 主线 | 20k、有效batch128=32/48/48 | 持续训练，不再串行堆500步资格 |
| LR | warmup500，2e-5→2e-6 | 保持已固定表；5k对照使用同一前缀 |
| 参数 | 冻结vision，VLM LoRA、完整action expert、独立I/O | 这次容量已经扩大，不再说只训练旧shared20 |
| 域损失 | T+1.5R+1.5L，分别先求均值 | 记录实际梯度，不把采样比当梯度比 |
| 目标aux | 原XYZ/速度项+W物理宽度，4次完整生成/更新 | 当前保留W，D5-T与约束修正另立后继版本 |
| dropout | 25% state token | 已存在，控制器反馈不drop |
| 精度 | FP32活动参数/累计梯度，BF16冻结存储及部分前向，highest matmul | 明确为混合精度；新部署自己验证，保留旧严格失败 |
| EMA | 当前新实现未实现 | 当前端点按raw评价；后继版本再考虑，不能无记录替换 |
| 图像增强 | 现有FM路径train=False | 后继目标版本可加受控光照/外观增强；不热改 |
| 空间对齐 | 当前未开启新OmniVGGT对齐 | 几何泛化不足时做独立对照，不阻塞主线 |

现场Adam五项实参来自private spec，公开代码和报告未完整披露，本次未核验。CPU测试中的weight_decay=.01不能冒充现场值。首更新报告应公开非敏感beta1/beta2/eps/weight_decay/clip_norm、dtype、LR和端点类型及其绑定哈希。

首资格batch共享梯度norm为T=.28191、R=.10929、L=.52165。乘权重后L约为T的2.78倍，但没有方向信息，单批不能证明负迁移。按检查点记录分域FM/aux、shared/目标I/O梯度、global clip触发率；必要时在少量固定batch报告源-目标梯度cosine。不要用不同量纲loss标量大小直接判断谁主导。

来源：[训练核心](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f7ec607ee9b0395ae74623977352de62cbe7e6a0/src/go2arx5_vla/sla_adapter/scale20k_training.py)、[参数/LR/Adam路由](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f7ec607ee9b0395ae74623977352de62cbe7e6a0/src/go2arx5_vla/sla_adapter/scale20k_contract.py)、[模型加载](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f7ec607ee9b0395ae74623977352de62cbe7e6a0/src/go2arx5_vla/sla_adapter/scale20k_model.py)。

## 5. 现在并行推进的工作

1. **接续唯一MIX20K进程。** 先只读确认现场身份，沿已经完成的独立恢复和7次更新继续，封存第一个非零完整端点及200更新前缀。公开实际global step、逐域访问、LR、参数/Adam变化与恢复结果。LoRA的B零初始化可使A首步梯度为0，不要求每个叶子第一步都变化。step-zero恢复不能替代后续非零Adam状态的接续证明。已有不确定/运行中进程不重复提交。
2. **补齐评价和部署出口。** 训练继续时并行建立新的dense-action-expert checkpoint读取、逐域离线端点评价和Go2 target Provider/native scope接入。不能假设旧仅LoRA/I/O的D4包格式自动支持新参数树。目标部署只加载目标I/O，保持原H40/前命令门/评分。第一个可用端点即复核，不能到5k才发现无法评价。
3. **重读D5已保存预测。** 五类H40首因、前命令门、crossing×时序×两门联合数、预测-映射参考-实际TCP三个层面的接近误差。只靠原记录能回答的问题先复用；W/T原Core复验和GRASP24独立继续，不成为SCALE启动条件。
4. **补数据身份的具体证据。** RoboDojo新增12与旧36TRAIN、12DEV已检查唯一，但公开准备代码未显式核对原12TEST。补旧60与新增12的path/hash交集，仅读元数据；现有证据不等于已经泄漏。若确有重叠，保留已读/已训练记录，隔离该数据并版本化修复，不能重置计数。
5. **扩大真实稳态吞吐样本。** 已有首更新125.2s、后4条中位数46.33s。后续20–50个暖态更新报告p50/p90秒/更新、data wait、FM反传、H40 aux、GPU利用率/峰值、保存与恢复。按当前4条中位数简单外推，20k约257.4 GPU小时、连续约10.7天；这是小样本条件估计，不含排队/冷恢复/评价，也不是实际工期承诺。资格422s包含冷加载/编译/保存，旧D4的10.89s/步不能用于新方案。
6. **继续目标数据建设。** 优先缺失语义组合和接近/接触覆盖，逐条准入，不等待整个Teacher矩阵通过。

当前microbatch=1意味着每个优化步128次FM反传和4次完整十步生成反传。工程优化优先有界预取、一次性文件校验缓存、视频帧缓存，再评估microbatch2/4。保持有效batch、数据银行、随机键、loss均值与全局步数；改变浮点归约顺序或源码需要独立记录数值差异与兼容接续，不能热改现场。按实测秒/步乘剩余步数估算工期，分有限allocation保存完整状态，不重置Adam。

首7更新中最后4步的wall合计185.884s，已记账三域data_wait合计22.825s，占12.28%。即使完全消除这部分、其余时间不变，理想吞吐也只增加约1.14倍；预取不能被承诺为数倍加速。需要分别测量128次FM反传、4次十步aux和优化器/调度成本，再选择更大的microbatch或实测更快的可用GPU。当前worker明确要求单GPU；多worker解码可以独立优化，分布式参数训练则需要真正的梯度同步/分片实现，不能启动多个独立优化器冒充同一全局batch。

实现证据：[新完整checkpoint](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f7ec607ee9b0395ae74623977352de62cbe7e6a0/src/go2arx5_vla/sla_adapter/scale20k_checkpoint.py)、[旧导出接口](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f7ec607ee9b0395ae74623977352de62cbe7e6a0/src/go2arx5_vla/sla_adapter/provider_multitask_v6.py)、[旧28数组包合同](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f7ec607ee9b0395ae74623977352de62cbe7e6a0/src/go2arx5_vla/sla_adapter/multisource_d4.py)、[源扩展准备](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f7ec607ee9b0395ae74623977352de62cbe7e6a0/scripts/sla/prepare_robodojo_scale20k.py)。

## 6. 目标数据扩充：先补缺失组合，再增加数量

目标采样目前16HOLD+16PLACE。20k对3条PLACE共320,000窗访问，平均约816次/PLACE窗；HOLD约233次/窗。RoboDojo与LIBERO均960,000访问，但分别只有48和1261条示范，按全域平均每个真实源episode承受约26.3倍于LIBERO的窗口访问。task/episode均衡能避免长轨迹支配，不能创造新的经历。

建议新的目标TRAIN建设目标为**120条新增独立示范**，是数据建设目标，不是宣称已采集或每次原生请求都会成功：

| 内容 | 覆盖设计 | 数量目标 |
|---|---|---:|
| 完整PLACE | F1/F2×红/蓝×左/右=8格，每格10条，多次独立重置 | 80 |
| HOLD | F1/F2×红/蓝=4格，每格5条，改变物体位姿与接近起点 | 20 |
| 纠偏/恢复 | 接近偏差、未夹紧、轻微滑移、落点偏差，各5条 | 20 |

先做红左、蓝右和F2的缺口。每条覆盖真实观察、名义预测、实际提交/消费指令、上一实际命令、时间戳、接触与终态。纠偏片段有单独类型和有效掩码，不把失败动作或未消费后缀当专家命令。示教、可执行规划、已有合格Teacher和人工纠偏都可提供数据，不设单一Teacher垄断准入。

旧DEV/TEST不回流。当前3PLACE使用C20统计，quantile归一化本身无clip、物理往返审计已通过，不能指控动作被截断。仍应报告各维超q01/q99比例、最大归一化幅值和state-token极端桶占比；tokenizer离散化超界状态可能丢失区分信息，需要实际分布证据后再决定新统计版本。

来源：[连续归一化](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f7ec607ee9b0395ae74623977352de62cbe7e6a0/src/go2arx5_vla/sla_adapter/codec_v1.py)、[PLACE准备及往返审计调用](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f7ec607ee9b0395ae74623977352de62cbe7e6a0/scripts/sla/prepare_target_scale20k.py)、[固定上游状态token离散化](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/405ca8879c032a05f0a778adc7d86f4e221c5eba/SLA/src/openpi/models/tokenizer.py)。

源扩充优先把ARX X5真实TRAIN从48条增加到实际可得的200–500条，并保持原留出身份；第一轮不再新增第四个公共数据源。目标新数据只在新命名curriculum版本进入，前5k两臂目标manifest/银行继续一致。若另开新数据分支，保留原SCALE主线与对照，不把新数据收益称为纯步数增长。

## 7. 后继本体适配路线

### A. 当前连续混源主线

保持MIX20K和TGT5K，保存500/2k/5k/10k/20k与目标对照500/2k/5k。原计划离线与原生预算不变。TRAIN拟合、源保留集和目标闭环分别解释。LIBERO的432条DEV来自Spatial整10任务留出，是未见任务/套件泛化，不是官方同任务闭环成绩，也不是同任务新场景评价。

5k对照比较同目标暴露、同参数范围和同LR前缀下的源数据+额外计算；MIX20K对TGT5K不支持纯源因果结论。主线低成功本身不暂停；若梯度/输出长期不动，具体定位，不无限重复相同设置。

### B. 目标集中适配候选

新增目标数据实际可用后，从明确SCALE检查点建立新curriculum候选，建议起始有效batch128=64目标+32RoboDojo+32LIBERO，首段5k更新；这不是当前20k的热改，也不自动扩展原预算。新阶段保留共享能力与部分源replay，目标LR可从比主线峰值低3–5倍的候选开始，并在首更新前固定完整schedule、Adam接续与评价端点。

目标占比提高必须伴随独立目标数据增长，不能把原3条PLACE再加倍重复。若同时改变数据、权重和损失，要把结果称为新本体适配方案，不宣称分离了每项因果贡献。D5的原GRASP24与已保存预测归因用于选择是否加入delta或针对实际拒绝通道的轨迹损失；不因某个TRAIN数好看就直接替换W。

### C. 空间、时序与移动能力

接近误差要沿“模型名义动作→映射参考→实测TCP”分别测量：参考本身偏离物体时重点查视觉/语言/数据；参考接近但实测远时查跟踪与可达性；固定变换不一致时查标定与动作合同。没有证据时不改WBC补偿。

若目标几何泛化仍不足，下一独立对照优先考虑官方空间对齐和受控图像增强；OmniVGGT是训练期空间表征教师，与采动作示范的Teacher是不同角色。对齐的视图、padding、图像增强与空间target必须对应；SLA_base已有预训练空间能力，当前不追加alignment并不等于从零训练，但也不能称为复现官方完整后训练。

EMA、空间对齐和视觉解冻不要在同一时点无区分全部加入。EMA若在后继版本启用，raw/EMA明确命名、保存、评价；不能用几个稀疏历史checkpoint拼出所谓逐步EMA。Memory留给遮挡历史、阶段识别、过去事件和多步任务，避免把当前接近/闭合问题转成新的memory工程前置。

在已有F1/F2和历史移动抓持实绩基础上，分层扩充“移动中保持、移动后抓取、移动中抓取”的目标数据和评价，逐步提高完整抓放与移动组合任务的难度；这不阻塞已经登记的移动评价。分离底座速度/姿态扰动与安装偏差，继续冻结WBC与原硬门。长程任务逐步加入多物体、多目标区和恢复，分别报告正确对象/正确区域/子任务顺序/整段成功；多个数据集本身不证明长程规划已经成立。

### D. 真实Go2+ARX5适配准备

当前阶段可完成数据schema和校准设计：相机内外参、TCP、夹爪实际宽度、命令与反馈分栏、实际控制周期、曝光/采集/推理/提交/反馈时间戳、上一实际命令、底座姿态与速度。真实本体另有TRAIN/DEV和统计版本。公开RoboDojo近本体数据提供迁移经验，不能替代这些目标控制事实。

先从固定底座低速抓放数据与评价建立真实接口，再增加底座移动和恢复。具体真机执行另需现场任务与资源落实；本次没有启动硬件。仿真、暂停时钟、离线H40与连续实时/真机结果分别报告。

## 8. 执行者应交付的下一批结果

1. 已发生7更新之后的实际step/cursor、第一个非零完整检查点与200更新前缀。
2. 真实稳态吞吐和剩余20k工期估计，包含aux与数据等待，不用旧配方外推。
3. 新密集action expert检查点的离线与目标Provider出口，接通既定500/2k检查点和5k闭环。
4. D5五类拒绝与crossing×时序×两门的联合表，原GRASP24真实分母。
5. 目标新增数据的语义组合、场景/谱系覆盖及准入数量；RoboDojo旧TEST元数据交集补证。

本评审不改写现役20k/5k的更新、访问、生成、原生预算，不自动启动5k目标后继、alignment、memory、40k或真机。研究建议与已经发生的效果分列。正常工程修复沿现有owner和版本机制推进，不新增多层审批或重复账本。
