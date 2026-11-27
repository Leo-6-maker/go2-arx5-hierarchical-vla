# SLA V6.5 · Recovery-aware Target Adaptation — R4→A93-R3，先训练完整1K再测试

2026-11-27 SGT。用户批准发布新训练计划，并明确“训1k再测试”。继承 [V6.4 #527](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/527)、[V6.3 #526](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/526) 与原数据工厂/评估/预算/安全合同。**本轮新谱系直接从正式 step 0 到累计1000，再开始完整科学评价；不设置100/250/500的常规科学测试或晋级停点。** 定期完整checkpoint、数据/数值/资源完整性检查、异常停止及原安全门仍有效。旧A250失败和SOURCE500暂停保持，不复活旧谱系。此PR发布计划供现场合法owner执行，并非已经提交PBS。

配套 [RUNBOOK](SLA_RECOVERY_ADAPT_V65_RUNBOOK_20261127.md) · [REGISTRY](SLA_RECOVERY_ADAPT_V65_REGISTRY_20261127.json)。

## 1. 固定依据与科学目标

- [Hopper实绩 3bdce937](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3bdce937c0334600f88043c32a10a2191e85f281/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V64_ADAPT_DISPATCH_ACTUAL_20261127.md)：A250已250/250，PBS736530 F/0，29:39，p50 5.23s、p95 6.40s（去前5步；独立复核p95 6.47s），完整53叶/五Adam计数250。原生HOLD 0/2、PLACE抓持3/4、抬升2/4、入区1/4、完整0/4，四张真实安全STOP后0积分；R4对应2/2、4/4、3/4、2/4、0/4。42旧源DEV MAE未退化但原生门失败。A500/A1K不得自动接力。
- [Core实绩 f88de473](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f88de4730e1e425b181d6dbf82f9cf9037bfb841/reports/SLA_CORE_EMBODIMENT_FACTORY_V5/V64_CORE_SELECTED_REFERENCE_ACTUAL_20261127.md)：93条完整示范/7916窗，69 legacy+8 Mobile Grasp+8 Diverse PLACE+8 Reach；还缺4 Container/3 Push，两个Container QUAL已实际启动，不能当成功。Recovery 3/6实际尝试、3条真实教师接管后缀、115窗、1150实际消费区间，集中红右CARRY停滞；不代表独立OPEN接管已成功。N2短卡4次完整2过原状态门，另4次启动失败，N4未开始；生产仍N1。最新PBS空闲只是固定回执，开工必查实际qstat。
- [原参数分区](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/124af5eca57d0713714c2c74ed7f7456ec0b10d6/src/go2arx5_vla/sla_adapter/scale20k_contract.py) · [真实加载资格](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/124af5eca57d0713714c2c74ed7f7456ec0b10d6/reports/WM02/sla-scale20k-20261119/CORE_GRADIENT_QUALIFICATION_ACTUAL.md)：历史真实加载VLM LoRA 27,869,184、完整Action Expert 427,932,672、每域IO 2,165,792参数；现役A250继承共享与四域私有分区。不能把叶节点数当参数数；首新训练审计实际权重树、rank/alpha、dtype、参数总数及冻结指纹，不宣称已扫描现场新树。

本轮是任务突破的综合候选，不是单一变量因果消融：增加新本体数据和恢复监督、目标域占比、冻结VLM LoRA、改变warmup/cosine及去掉新Ville IO同时变化。动机是减少共享表示扰动、提高实际目标域监督、保留少量旧源知识。多域梯度冲突、VLM漂移和训练不足均为待检验假设，不能据小样本断言原因；低层实际选参考/跟踪尚待归因。不能用新数据或训练loss冒充已解决闭环。

## 2. 唯一新谱系与训练范围

实验ID V65-A93-R3-1K；父R4-200完整封印 f1f5a994005a9fc831934a156ca823d7e4b67794ce2533d72e087c2aeb29b9cb、正式父回执 0896270957c478197e586a71f2d2d3a55464a5bb7436cf3d390fa14804926433。按现役 corrected_parent 完整核验父模型及已训练私有IO，不能使用A250、SOURCE250、R5或仅推理数组作为父。显式新Adam、新RNG（课程seed建议655202611，实际封存），正式step0；去除非活动Ville域，不新建随机Ville IO。只使用已训练的Go2、RoboDojo、LIBERO私有IO。

冻结Vision Encoder、PaliGemma稠密基座、R4既有VLM LoRA的全部已训练值，保持其forward贡献，不删除/归零/随机初始化/合并造成数值漂移；完整dense Action Expert与三个活动domain IO参与训练。预计约434.43M可训练（历史427.93M+3×2.165792M，现场以实际树为准）。正向allowlist、完整库存、冻/训叶不重叠且并集完整；VLM LoRA必须不进入optimizer/weight decay/EMA，或有严格证明的冻结等价；从R4原参数值构建新分区，逐叶及完整固定输入预测在step0一致；若不一致先修复，不通过放宽语义。保留原FP32训练参数/梯度/Adam与BF16冻结基座，R4 LoRA自身现有dtype不暗转BF16；最高原matmul精度与原state dropout 0.25。原Action Expert的原生配置gemma_300m（非只LoRA）；不得全量解冻VLM、调rank/alpha、切FP8或改优化器。所有新旧WBC/OCS2/PD/腿策略、L1、2ms积分、50ms消费、H40/P10、40mm/s夹爪门、原安全及无GT学生不变。源域保持各自native H/clock/codec/mask，RoboDojo记录中的25/30Hz文字差异依据实际封存版本/reader核对，不重采样或偷偷修订旧合同。

## 3. 数据和固定银行（A93 + R3，不热换）

A93是完整教师示范93条/7916窗，不把3条Recovery说成96条完整示范。继承A74已验证52旧+22增量及A78/A83/A87/A93分批封印。A93最终增量manifest 1069a89550ea667686a5670249f7b41aa950eb3e687b835fc5e600ca61c08773，Hopper读回 0a52eb35bd1152cf07f688c0b839eddc875b1d13eb7f1ab6f788553d81797e53。现场逐批联合索引重新核7916窗、原52和余41、归一化、独立scene/episode/split及实际reader。R3限定原Recovery TRAIN_02/04/06真实后缀45/35/35窗；原标签回执 adc0c371c431017700462ab352532b5cd1df51c776d528f5afc4704f0ec118dc / 89ee4dcefa4d099aac2acadf5830ac90d7d129a3df1514f1486722d0e6e87ca2 / 9a788dc687a0f44e3b19e447c1d9ee786176e84d885ef382ff4845cf56130698。Core已准入但未证明Hopper训练读取。首发前建立独立R3便携清单、真实reader/双RGB起止、上一实际消费夹爪、50ms已ACK/消费、R4逆重锚和有效mask，核115窗和真实支持，学生前缀/未消费后缀/STOP后/DEV/CONF/TEST不作监督；接管身份只供审计不泄漏给学生。按场景group排除受保护人口，避免正常教师与恢复前缀混计。若R3任一关键身份/训练支持未过，先修复及交接；不静默改配比为A93-only，不等待A100/R10，不将新Container QUAL混入TRAIN。运行期新数据只留后续冷边界。

| FM域/子集 | 每步 | 1000步计划访问 |
|---|---:|---:|
| Go2 原52完整示范 | 48 | 48000 |
| Go2 A93新增41完整示范 | 46 | 46000 |
| Go2 Recovery3后缀 | 2 | 2000 |
| RoboDojo旧200 | 8 | 8000 |
| RoboDojo扩充107 | 8 | 8000 |
| LIBERO原合格源 | 16 | 16000 |
| **FM合计** | **128** | **128000** |
| Go2真实完整H40 aux额外 | 4 | 4000 |

Go2共96/128即75%，旧/新完整94/step，R3 2/step（~17.4次/窗名义平均，只有3个独立episode，明确重复风险）。RoboDojo只用已有真实训练资格的旧200+扩107、LIBERO现役旧源；Ville/SC-VLA为0，SOURCE608准备独立。每步旧48优先能力保持，建议至少24 HOLD（若固定真实库存不足，预先登记实际分配，不能伪造样本）；其余按任务→episode→事件→anchor分层。新46覆盖Mobile HOLD/Reach/PLACE等实际任务，优先接近/抓持、抬升、搬运、下放、真实OPEN/WITHDRAW，建议每步旧/新合计至少8真OPEN/WITHDRAW和8 CARRY/LOWER，按真实事件池事前核覆盖，不重复同一窗凑配额。Reach保持自己语言/评分，不能把无接触Reach改成HOLD。R3每步两不同episode按轮换均衡，真实可支持的CARRY/LOWER/OPEN/WITHDRAW按episode/phase抽样，记录唯一episode、唯一窗、重复和实际有效点；不夸大红右覆盖。四个aux为互异真实完整H40、旧2新2（不强制从稀缺R3抽取；对应真实FM样本），原十步全反传。辅助样本不是额外128中的FM样本。全部1000行、aux、LR、域reader顺序、manifest及seed固定封存，首行真实readback并逐行检查完整bank，禁止中途重采样/改配比。所有访问计实际优化器步骤，工程访问另计。

## 4. 优化配置与最小资格

- Effective FM B128，Go2 96按32+32+32微批，RoboDojo16、LIBERO16，各真实样本加权，aux4。共享FM按0.75/0.125/0.125加权，private同域权重，aux系数采用0.375（V6.3 aux/target FM=0.25/0.5=0.5，本轮保持0.5×0.75=0.375；这是新配比的明确设计选择，不重叠缩放）。保留原FM、mask、物理G7/H40辅助损失，不改变原时钟和动作语义。一轮全局clip+AdamW，beta1=.9 beta2=.95 eps=1e-8 weight_decay=1e-5 clip_norm=1.0，原有限值检查。
- Peak LR 8.481070829156122e-7；warmup 50步线性；随后cosine至累计1000的0.1peak。以s=1..1000，s≤50为peak*s/50，否则peak*(0.1+0.45*(1+cos(pi*(s-50)/950)))。1K之后不在本日程上自动继续；条件3K需基于结果和新日程另行登记。
- 不把现役v63 worker的固定250/四域绑定强行改数字绕门。现场在独立版本化capsule重用原读取器、LongLearner及恢复机制，明确三域参数/梯度join/optimizer状态键、冻结LoRA、固定银行和总步数。CPU实际输入形状Go2(96,40,32)、RoboDojo(16,50,32)、LIBERO(16,10,32)，aux(4,40)；真实mask与Recovery支持。先做最小GPU资格：完整配方实梯度、所有活动域正有限、冻叶指纹逐位不变、真实R3梯度和实际访问、完整状态冷恢复与固定输入预测一致。建议最多6次工程更新（实际旧工程15/24，最多剩9，必须先扣最新账本；不足则缩到有证明的最小资格或登记阻塞，不能扩大）。工程后正式从R4和新Adam/RNG step0重新起跑，不使用工程权重。历史A250资格不自动覆盖新配比/冻结树。可选零更新梯度分域诊断与A250旧端点只读分析并行，不以完成一套消融为首1K前置。

## 5. Hopper 调度：1K前不安排常规原生测试

单H200，原12 CPU/225GB，建议首张PBS申请3h（实际site测量/授权决定，不超过原6h；旧A250约5.23s/step，1K纯更新约87min，另计新配方编译、资格、读入、全checkpoint和安全退出；非保证）。实际项目授权/旧320GPUh及2026-12-05 12:41:52 SGT截止不扩大。已健康或未知PBS先读回，唯一owner、锁、job id、manifest/代码指纹；不重复提交。Corelab不等待训练，不抢Hopper做CPU-only。

正式训练0→1000保持同一课程、全状态及累计步号。每100步或30min先到完整保存，额外保留250/500与1000；保存parameters、全部实际Adam、rng、bank cursor、lr/step、stats/processor、完整成员SHA、frozen fingerprint、每域及子集/阶段实际访问、资源和工程账本。100/250/500不启动常规固定反馈/原生评估、不设保留门等待、不中断健康训练。运行时仍查真实输入身份、有限loss/grad/parameters、有效mask、冻结叶、完整恢复及实际可写容量/配额/安全封存、实际截止、STOP；不能把“1K再测”理解为绕过技术或安全停机。若实际PBS wall不足或临时资源中断，封存最近完整科学端点并合法精确恢复剩余步（包含Adam/RNG/LR/游标，不重warmup，不重复已封存更新，不以工程权重续）；未知状态先对账。无完整1K不得声称达到。除实际异常或安全原因，不因中途普通开发指标变化停训。按原每次至少15min保存预留与实际最坏合理save耗时决定是否继续，必要时追加合法后继PBS完成同一1K，不超原预算。

## 6. 1K完成后唯一正式测试与决定

Hopper完整端点封存、释放GPU、Core独立冷恢复并核实际1000个正式更新/128000 FM、96000 Go2（48000/46000/2000）、16000 RoboDojo（8000/8000）、16000 LIBERO、4000 aux及实际分布。精确三域+shared Adam计数1000，冻叶R4既有VLM LoRA保持逐位身份；不能从叶数猜分区。只在这里进行正式完整评价：
1. 原24固定反馈（相同输入/noise）、原42源DEV（RoboDojo24/Ville18）。本轮无Ville训练及私有IO，不能将其MAE伪装为本轮已训练源域；用原R4对照固定初始化Ville IO、相同状态/统计/噪声作共享模型迁移诊断，实际需独立评价临时IO而非加入训练树，无法同口径则明确NOT_COMPARABLE，不填假分。LIBERO按实际原登记独立源DEV补可用指标，绝不使用TRAIN冒充验证。原源域>10%恶化告警保留，只在同口径有效样本判断。
2. 原N02 HOLD 2和PLACE 4（P01/P02×701/702）六卡，R4/A250/SOURCE250历史同身份实绩复用。保持旧物理安全/评分/时钟/预算；N1原资格入口，实际selected-reference观察只有同拓扑/运行时已资格才附加独立trace（未测明示，不把accepted当selected）；不等待N2/N4。逐格报告raw/accepted/consumed、接缝/块内、实际抓持/≥20mm抬升/正确入区/支撑/释放/撤离/完整成功、STOP后0积分。
3. 原独立HOLD四张（N03/VAL-F1-G00×201/202）与R4/Main的既有配对结果对照。在实际旧原生槽和资格允许下，增加已预登记的独立未见几何/Reach/完整PLACE CONF，优先保留真正独立数据，资格卡不作学生训练。最低1K正式人口为10张原生（六+四），无合法槽时先核差额，不能凭预算算术冒充授权。A250历史卡全部保留，不以换seed/删除失败改善结果。
4. 晋级门保持：N02 HOLD 2/2、PLACE抓持4/4、抬升≥3/4、入区≥2/4、无新增真实安全STOP及重复抓前拒绝；完整PLACE≥1/4优先，或在其他阶段不退化前提下至少一项明确真实阶段进步；独立HOLD有效四格，至少不低于R4 2/4且遵守原父门/无重复新增失败机制；源域留存无未经解释实质退化。独立CONF阳性才能支持泛化；固定开发六卡样本有限。原门不变。未通过则封存负结果，分析高层输出→接收→实际selected→achieved和R3监督、梯度冲突/参数范围，禁止自动继续3K或SOURCE608。通过则报告1K结果，条件3K、A100/R10/200/R40、新SOURCE608只在真实评价/资格/原预算与后续明确计划满足时推进；本PR不批准自动训练到5K/10K。用户“训1K再测试”是本轮明确首目标。

## 7. 并行Core与预算

Core保持P0多环境资格和N1完整生产、75→93已完成，当前93→100只缺4 Container+3 Push；真实Recovery3→10→40，首3条只做训练后缀，后续新场景及OPEN/其他阶段真实独立接管，不重复旧失败/强制GT学生，不等Hopper。N2仍待实际首条WBC目标生效与回调时序、完整接触、角色故障隔离、真实双卡渲染和产量；原短卡已8次，2过原状态门，不无限补旧槽。N4待N2合格。608库存保持499+58+51，已完成独立H30 GPU工程3步/96窗及五域CPU首行读取，联合梯度/完整恢复/合格目标父尚待；新SOURCE608不抢首1K，旧SOURCE500保持HOLD。原V6.2实际容量/配额与安全封存门、Vanda暂停。

已知正式500（SOURCE250+A250），本轮完整1K后累计1500，默认8000剩6500（更晚消耗须扣），条件15000并非新增授权。工程已知15/24余≤9，新配方最多建议6（若用满累计21余3，留后续资格但不保证足够）；不重置可选控制1000/历史800 R诊断/原320GPUh/截止或Core原生额度。主线学生V6.4的50剩余是早期条件上界，A250六卡已用，故最新已知最多44，1K拟用10后最多34；每张新评价实际映射原未用槽，其他更晚用量继续扣减。若原槽不适用需显式登记授权差额。所有实际失败、工程更新、正式更新、实际GPU小时、原生和数据请求独立记账。普通工程在既有合法权限内自主修复，真实权限/预算/安全障碍只停受影响角色。每100步提交训练实绩摘要（不等评价）、每波/最多12h更新，1K后一次完整科学报告及明确下一步决定。

本次为GitHub计划发布；没有SSH、PBS提交/取消、实际训练、仿真、数据传输或删除；本轮正式与工程新执行均0。现场须验证实时资源、资格、R3实际接收及账本。
