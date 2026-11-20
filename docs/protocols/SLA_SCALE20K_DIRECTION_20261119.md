# SCALE20K：把规模化多源后训练作为下一阶段主线

2026-11-19，Asia/Singapore。本文依据用户关于“小规模训练可能不足以适配新本体，应综合多个数据集大规模后训练”的新决策更新工作优先级。状态为研究设计与准备，新增训练、下载、生成和原生执行均0。

## 1. 决策

**现在开始规模化数据和持续训练准备；数据/梯度/恢复接线与实际资源成立后，即启动一条预注册20,000更新的多源后训练主线。500和5,000更新是同一训练过程的检查点，不要求小模型先达到高抓取成功率。**

PR416原固定cf9d1acb及PR417正在执行的D4/D5/J身份保持。现役500步作业原样完成；它们提供对照、实现和损失信息，不再成为新主线逐级全部通过的前置。不热改健康作业的预算或学习率，不改写其失败。

- [机器研究设计](SLA_SCALE20K_DIRECTION_20261119.json)
- [Codex规模化启动prompt](../handover/2026-11-19/START_SLA_SCALE20K.md)
- [原D4/D5/J固定计划](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/cf9d1acbf72e96b1bd158322ff8cbaf8fe28fabb/docs/protocols/SLA_D4_CLOSED_LOOP_AND_GRASP_20261119.md)

## 2. 为什么现在应该扩大

D4单臂只有20条目标HOLD、1376窗、500更新与16000目标访问；目标阶段源访问0。它检验小样本下源初始化的效果，不能否定充分多源后训练后的本体适配能力。

500×32/1376≈11.6是平均窗口访问量，不是严格epoch。若只增加C20的训练步数，5k约116次、20k约465次平均访问，独立经历仍只有20条。因此同时扩大独立episode、任务/场景覆盖、连续优化量和适配参数范围。

### 官方数量级

| 配置 | 更新×有效batch | 采样窗口访问 | 相对当前D4单臂 |
|---|---:|---:|---:|
| 本项目D4 | 500×32 | 16000 | 1 |
| 官方Piper真机示例 | 20000×64 | 1280000 | 80 |
| 官方RoboTwin | 30000×256 | 7680000 | 480 |
| 官方LIBERO | 30000×512 | 15360000 | 960 |
| 论文RoboDojo-Real | 40000×512 | 20480000 | 1280 |

这仅是窗口访问数量级，不是独立数据、动作数、FLOP或性能倍数。官方LIBERO用H10，本项目H40，其他例子有H50。论文RoboDojo40k与另一个公开集成60k不同，不能混成同一执行身份。

官方LIBERO/Piper的默认freeze_filter不冻结VLA参数；RoboTwin使用VLM LoRA但完整action expert仍可训练。当前D4仅28活动叶约52.15M参数，容量和可改变的模块不同。本主线选择视觉编码器先冻结、VLM LoRA、完整action expert和各域I/O可训练；这是项目的新recipe，不是声称复现官方全部设置。

来源：[论文§4.2](https://arxiv.org/html/2611.08373v1)、[Piper配置](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/18896e757647ed9c2a68816b35a7100afda6e287/SLA/src/openpi/external_config/train_real_world_handover_vegetable.py)、[RoboTwin配置](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/18896e757647ed9c2a68816b35a7100afda6e287/SLA/src/openpi/external_config/train_robotwin.py)、[LIBERO配置](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/18896e757647ed9c2a68816b35a7100afda6e287/SLA/src/openpi/external_config/train_libero_plus.py)、[freeze filter](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/18896e757647ed9c2a68816b35a7100afda6e287/SLA/src/openpi/models/pi0_config.py)、[实际对齐训练入口](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/18896e757647ed9c2a68816b35a7100afda6e287/SLA/scripts/train_align.py)。

SLA_base已经包含大规模预训练；本计划从它适配，不从头复制整个预训练工程。论文预训练过滤了移动底座轨迹，所以公开桌面数据不能替代Go2移动底座、接触与WBC跟踪分布。目标本体数据须并行增加，但不要求先攒齐100条才开始多源学习。

## 3. 最新实绩对这个判断的影响

PR417固定dd7d90de已报告首布局10/12封存、441541tick、1539生成，正确抓持/抬升/成功均0/10；1539份H40没有跨22mm阈值，最小实测开度38.38mm。当前前缀的主要现象已经是策略不产生有效闭合，不只是先前离线预测疑点。

R0还确认每臂16000访问中有2124闭合跨越窗口访问；2000aux中有257次、覆盖40个不同跨越窗。不能继续说“完全没采到CLOSE”。这不能单独证明数据规模是唯一根因，损失与适配参数也需要检验。

同一发布快照中D5-W298更新、J0 88更新，J1源访问0。新规模计划不改变这些健康作业。

[最新闭环/作业实绩](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dd7d90deb1c0ab257c4b9b7bbe43d587b842c791/reports/WM02/sla-d4-next-plan-20261119/CORE_RGB32_AND_D4_PREFIX_ACTUAL.md) · [R0实绩](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dd7d90deb1c0ab257c4b9b7bbe43d587b842c791/reports/WM02/sla-d4-next-plan-20261119/ACTUAL.md)

## 4. 规模化启动只依赖训练正确性

保留四项实际前置：

1. 至少两个真实可用的公共源域与一个目标数据版本；各域动作/观察、时钟、normalizer、有效窗口和TRAIN分割明确。
2. 真实batch可前向、反向，有限有效梯度到达预注册参数；不串域更新私有I/O。
3. 模型、优化器、RNG、全局step与每域cursor完整恢复；恢复后下一更新使用同一数据和随机流。
4. 实际owner、GPU/CPU/存储及有限allocation成立。

不要求Teacher整格/QUAL48通过、D4全部结束、D5达到12/16、TRAIN96全过或N2合格。来源可按自己的native动作语义学习，不要求先证明与Go2 dispatched command等价。

技术前缀建议200个真实更新，纳入同一20k进程及学习率/optimizer，不在每个500步重新fresh Adam。低抓取成功率本身不使这条训练停止；数值故障、标签错误、数据泄漏、梯度接错、恢复损坏或实际资源结束才是对应暂停依据。

## 5. 数据池：三类作用明确

| 数据域 | 首轮来源 | 作用与边界 |
|---|---|---|
| Go2+ARX5目标域 | C20及逐条准入的历史完整PLACE，新TRAIN版本 | 学目标动作/接触；可从现有数据开始 |
| 近本体真实域 | 扩展RoboDojo-Real TRAIN | 增加真实外观、动作与多任务经验，不仅重复36条 |
| 多任务仿真域 | LIBERO TRAIN | 增加操作、指令、场景和对象变化 |

源数据建设目标为1000–5000条独立示范、至少20任务；这些是工程覆盖目标，不是效果保证或要求全部到齐才允许技术前缀。两源已有实际准入子集即可开始正式前缀，但每个报告必须给真实规模，36源+20目标不能被称为已经完成大规模多数据集训练。

目标数据并行扩到100–300条、覆盖HOLD/PLACE和偏差恢复，不把目标数量变为公共源学习前置。历史3条PLACE只在逐条原Score/标签/消费与split去重后准入，缺失者单列；不从D4/D5的DEV或任何旧TEST挑成功样本加入。

已审ARX X5子集沿用双臂H50/30Hz future logged reference与原有效窗口枚举，排除复制末标签；扩展优先同一本体。若新增RoboDojo其他本体或采集版本，须独立核验观察/动作索引、时钟、坐标与末标签，再冻结native合同，不自动继承首批ARX X5语义。官方RoboDojo存在25/30Hz等不同来源。LIBERO按官方native观察/动作/时钟建立独立I/O和TRAIN-only统计，不强转成Go2指令。目标保持G7/H40/50ms与实际上一命令。

按episode、task、scene和collection lineage建立inventory。保留原源域和目标域TRAIN/DEV/TEST分割与身份；DEV仅作规定评价，旧TEST不使用。task/domain分层抽样，避免长轨迹或重复脚本支配。不能把视频帧、重叠窗口数当独立示范数。

第一份混合manifest在训练前冻结；新数据在明确checkpoint进入另一个命名curriculum版本。换数据后的曲线不能继续解释为只增加计算量的对照。不同时接入第三个新公共数据源。

## 6. 两条训练，主次明确

从已校验官方SLA_base创建新的共同初始化。共享参数与目标I/O两臂逐位一致；各源私有I/O按基础权重匹配形状复制或统一seed初始化，并在首更新前固定。默认不继承B500的模型端点、S1000 LoRA或旧优化器。语义转换/统计可以复用已审代码。

| 臂 | 全局更新 | 每更新T/RoboDojo/LIBERO | 目标访问 | 源访问 |
|---|---:|---:|---:|---:|
| SCALE-MIX20K主线 | 20000 | 32/48/48 | 640000 | 1920000 |
| SCALE-TGT5K对照 | 5000 | 32/0/0 | 160000 | 0 |

主线有效batch128，microbatch由真实显存测量确定，通过梯度累积维持有效batch；对照32目标。两臂相同结构、目标loss、目标随机/采样流，使用同一20k学习率表的前缀。前5000更新的目标manifest和数据版本必须完全相同；不能只给混源臂更新目标数据。共同前缀默认固定数据，任何预登记curriculum变更须两臂同步且逐版本报告。主线总256万访问，是D4单臂访问数160倍，不能解释成独立数据或计算也恰为160倍。

固定目标loss为原FM32+0.1XYZ+0.1夹爪速度+1.0物理宽度L_w，沿用D5-W的完整生成定义；本版不加delta项。不要求D5-W先胜出，此项是已声明的研究选择，控制臂同样使用。两臂都开放完整action expert，故MIX5K与TGT5K的差别仍是额外源数据/计算；与旧D4比较则是新整套recipe，不能将收益全部归因于数据量。

各域先求均值，混合梯度为：

$$L=L_T+1.5\,\operatorname{mean}(L_R)+1.5\,\operatorname{mean}(L_L).$$

目标32、源48/48对应这个样本权重约定；不是官方sim/real比例。控制臂L=L_T，目标辅助与随机流相同。首更新前列出互斥的共享、目标、各源参数清单，禁止重复归属。共享VLM LoRA和action expert收各域梯度，各域I/O只收本域梯度；未参与该更新的域I/O及其momentum/optimizer state必须不更新，也不应用AdamW decay（尤其TGT5K的源头），不能只设零梯度；不使用旧J“源只能到shared20”的参数合同冒充更大容量。WBC/腿控/硬门/decoder保持，目标不新增BASE head。

目标aux仍4条完整10步生成/更新，mean40后mean4；完整生成成本很高，必须实报。第一版不开新的memory或OmniVGGT空间对齐，以控制实现范围；它不是官方完整recipe复现。空间对齐以后做独立版本，不成为规模主线准备前置。

## 7. 连续训练和学习曲线

MIX保存500、2k、5k、10k、20k。TGT保存500、2k、5k，使用完全相同的长程LR前缀，不为5k另设一条已经衰减完的短schedule。

建议20k全局cosine、warmup500、peak2e-5、末端2e-6作为新候选；在首个正式更新前，根据技术梯度/显存检查冻结完整表。不按闭环DEV调LR。已有500步小recipe的LR尾段不能循环拼接成20k。

新recipe/checkpoint/package需支持完整action-expert参数、Adam/RNG、global step、T/R/L cursor恢复。不得仅把固定500的校验改成20k，不得每6小时重启优化器。每个有限allocation结束保存，下一实际授权allocation独立接续相同状态；不擅自延长PBS额度。

5k是进展检查点，不是高成功资格门。看TRAIN物理拟合、独立DEV、每域损失和输出变化：

- TRAIN和DEV持续改善：继续原20k计划。
- TRAIN改善但DEV退化：指出数据覆盖/过拟合，后继版本扩数据；不偷偷替换当前评测集。
- TRAIN闭合拟合长期不动且梯度/输出显示停滞：检查loss/适配范围，不把无限加步当唯一方案。
- 离线改善而闭环仍失败：检查执行、接触和移动分布。
- 源改善但目标退化：记录负迁移，后继版本才改域权重。

20k之后是否40k，依据曲线和新资源另冻结续训身份与LR尾段。当前不自动延到40k，也不把不同LR/数据版本的20k/40k说成完全相同实验的无条件延长。

## 8. 评价伴随训练，不阻塞训练

离线在8个端点（MIX5个、TGT3个）各做目标TRAIN96、RoboDojo DEV96、LIBERO DEV96，以及每域独立RGB16恢复。固定同一bank和噪声；各域指标不直接混成一个单位不明的平均分。

规模模型使用新的4布局六指令DEV24。在TGT5K、MIX5K、MIX20K各评价一次，共72请求。5k可以比较同更新/同目标暴露的混源效果；20k是规模主线终态，不能拿它比TGT5K后声称纯源数据因果效应。反复使用的DEV一直是DEV。

MIX20K冻结后在另外4布局六指令确认24，不打开旧TEST。新确认只报告该端点能力，不宣称一次小样本证明普遍泛化。Core评价和离线训练在各自实际资源下并行；评价尚未结束不使健康的20k优化停住。

所有闭环仍逐查询通过原H40/前命令门，原2ms Score和接触STOP不变。原生源标识、dtype和包清单在部署前固定；已发布的严格浮点失败保持，后继数字等价合同须事先版本化，不能追认旧失败为通过。

## 9. 全条件规模预算与资源量级

| 项目 | 新规模计划上限 |
|---|---:|
| 优化更新 | 25000 |
| 目标访问 | 800000 |
| 源访问 | 1920000 |
| 训练完整生成 | 100000 |
| 离线完整生成 | 2760 |
| 科学原生请求 | 96 |
| 条件接口储备 | 2 |
| 原生tick（含接口） | 4399394 |
| 原生模型生成 | 15360 |
| 全部完整生成 | 118120 |
| 真机请求 | 0 |

离线2760=8×(96×3+16×3)+24训练资格+48 Core端点恢复。原生96=DEV72+新确认24，含32 HOLD/64 PLACE；接口仅在新包/路由需要原F1/F2非接触验收时各5953tick。首200技术更新已包含在20000之内，不另加预算。

这是一份新研究预算，不占用/改写现役D4/D5/J/M2账目，也不是现场GPU配额。未下载的大数据需要按实际manifest评估字节和存储，不无界下载。

D4旧配方平均约10.89s/update，旧设置线性外推20k约60.5小时/臂；新的batch128、完整action expert、三域输入和aux反传成本未测，不能沿用此值作工期承诺。先测真实前缀的wall/step、GPU峰值、CPU、数据等待、保存/恢复，据此绑定GPU小时和分段allocation。不能假设有未获授权的多GPU，也不能因为一次allocation只有6小时就把20k拆成40个fresh-Adam小模型。

## 10. 本次交付边界

本次只发布研究优先级、明确的规模路线和执行prompt；没有启动新训练、下载、模型生成或原生评价。现役PR417按原固定计划继续。下一执行者先完成数据inventory、规模engine与真实资源绑定，满足第4节即可开始20k，不再等待所有小实验达到高成功率。

## 11. 本增补核验

本增补新增3份规模文件并更新统一入口；原cf9d1acb七文件快照作为现役历史计划保留。两路独立复核检查了官方配置、共同前缀、RoboDojo各本体语义、域参数更新与现役作业边界。预算乘积、所有条件合计和12个相对链接静态核对通过。实际训练、下载、生成、原生与硬件请求均0；固定提交回读和PR状态在发布说明中记录。
