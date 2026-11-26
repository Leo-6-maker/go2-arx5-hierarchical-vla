# SLA V6.2 · Science-first long-horizon：优先科学主线、并行知识学习与多技能迁移

2026-11-27 SGT。用户最新两条指令：“优先推进科学实验，更新长程实验计划到github”；“不用管存储限制，优先推进科学实验主线”。本增补**取消原项目自设固定存储余量门作为未来科学派发的前置条件**，不等待清理、迁移、NFS额度恢复或盘点报告。实际文件系统容量、实际用户配额、系统/站点强制规则、必要安全收尾和数据完整性仍须满足；不通过假造容量、放宽物理安全或删除唯一数据来实现科学进度。

继承 [V6.1 #523][v61]、[Core V5 #521][v5]、[SOURCE V6 #522][v6]。配套 [Runbook](SLA_SCIENCE_FIRST_LONG_HORIZON_V62_RUNBOOK_20261127.md) · [Registry](SLA_SCIENCE_FIRST_LONG_HORIZON_V62_REGISTRY_20261127.json)。本文件只更新未来派发优先级和科学合同；不修改历史回执或健康运行，不代表已部署、已执行。冻结WBC、原L1、OCS2/PD/腿策略、2ms、原安全/评分和无GT学生全部保留。

## 1. 最新实际进展（固定回执，不是本次实时SSH）

以 [训练执行分支4de2985][actual]、[数据工厂6a10550][factory]、[01:11 SGT维护实绩][storage] 为准。

| 角色 | 已证实 | 未结 |
| --- | --- | --- |
| R4-200 | N02/201、202 HOLD 2/2，46656tick/141前向，25.80/24.41mm，1.198/1.268s；PLACE4/4抓持、3/4抬升≥20mm、2/4入区、0/4完整，182812tick/640前向，0安全STOP/TECH | 独立HOLD六个未启动请求 |
| R4释放 | P01/701与P02/701入区后65+67=132返回，raw/消费最大开度3.36/3.10mm，实测19.29/19.27mm，原阈值39.6mm | 策略无充分释放意图；另两卡仍有抬升/搬运问题 |
| R5-200 | 共同MAIN3.8K父、正确标签、同源/辅助/LR；200步、400 OPEN+399 WITHDRAW替换，PBS735810 F/0、20m11s；p50 5.087s；原24反馈H40 24/24、raw seam2/24；另外24个入区反馈中0/24预测达到39.6mm | 自身闭环未测，不能宣称R5原生失败 |
| Go2数据 | 69=46 HOLD+23 PLACE；6100查询、5893完整H40；52+17条几何往返共239860实际有效点 | 额外17条真实reader/运行时mask资格与优化访问分别确认；首SOURCE保持52 |
| 公共ARX5 | 337=RoboDojo307(200旧+107新)+Ville30；最新单臂合并版已送Hopper，Core 3691 TRAIN/742 DEV窗；新增内胆/锅盖4条50Hz只是候选 | Hopper训练访问0；Mix500/2K未完成 |
| SOURCE | 5000步固定bank、Core四域实际reader已过两批；Hopper路径/父/银行已回读；LR、aux真实数值已核实 | 实际GPU资格、SOURCE PBS、优化更新0 |
| V5 | 新N1 Teacher 1/1移动HOLD，底座100.45mm，抬升24.81mm、保持1.026s；两移动容器冷态释放候选 | 新技能31条及Recovery；N2/N4控制/接触/STOP资格 |
| 存储 | SSD已退役145参数文件、9159933952 bytes≈8.53GiB；历史SSD92.17、HDD533.13、NFS共享1161.82GiB可用；NFS个人quota未知 | A0实际转存0、Habitat本地退役0；不再以这些维护任务阻塞科学 |

R4完整封印 `f1f5a994005a9fc831934a156ca823d7e4b67794ce2533d72e087c2aeb29b9cb`；R5封印 `e44934ce604dff1aef3c2d3e94b4c2235efd5a15e9883c9ceba6265bb8cc61d2`；SOURCE bank `1ba0092a0c982c51e785849b86332bfc9c3db636bad6a82cab97548bed027891`；独立HOLD名册 `c2af53f5b70340d1543082adbdbb769ea8e0e66231ac63d56059485136669605`。原R1/R2/R4共600+R5 200=800历史诊断更新，SOURCE仍0。M1000原六次碰撞保留。R4的200步与M1000的1000步不能作纯标签因果对照。完整PLACE和跨布局泛化仍待验证。

## 2. 科学优先调度：四条工作链并行

P0-A：Core补齐独立HOLD六卡；P0-B：Hopper真实四域资格及首250步SOURCE；P1-C：PLACE释放因果诊断与新监督；P1-D：Core新技能Teacher、公共数据及多环境生产。A/B在首250之前并行，不等待全部存储治理，不等待完整PLACE、R5、Mix500或Embodiment100。正式500续训必须等待A和SOURCE250评价。C/D不作为SOURCE首段额外前置。

### 2.1 取消人为固定存储门，保留真实写入能力

本节依据用户最新指令，覆盖旧V5/V6.1未来派发中SSD/HDD的固定15%+8GiB预留、50GiB等项目内部绝对预留及150GiB hot工作集上限的**硬阻塞作用**。它们可作为容量观察/提示，不再是本项目新科学请求的拒绝依据。历史失败及原门当时的判定保持；如果任何数值是实际管理员/站点强制限制而非项目自设，则仍必须遵守，不能将用户指令当作系统权限提升。

现场Codex在项目私有用户空间复用 [v5_storage_layout.py][layout]，做最小必要版本化修改：保留真实mount/device/writable、所有者和in-flight账本；把旧固定floor/HOT_LIMIT返回为 advisory 与历史数值，实际admit按真实可写容量、已知用户quota/站点硬限制、在途最大写入、checkpoint/结果临时重叠、可测的终态封存/安全停止所需余量判断。对实际使用卷逐一检查，不因为无关卷或NFS额度未知阻断不使用该卷的任务。不将未知quota当无限；实际写NFS的任务必须有真实适用的授权及可证明的有限写入/恢复能力；若不能证明则选择已有合法本地路径和有界封存，或只暂停受影响任务。不能填假free、伪造gate pass、直接删除系统或他人文件。检查容量只为安全写完实际工作，不做新的全盘审核或大系统重构。若容量紧张，用较小的有界请求、逐卡/逐批、安全收尾；不能缩短已注册物理判据、删RGB、隐瞒失败、改变任务来凑成功。保持原安全STOP、无STOP后积分。运行期间检查真实写入余量，达到实际硬边界前封存关闭。

优先原已验证SSD控制/必要缓冲与HDD大型产物路径，NFS用于已有合法归档/完整权重。路径和锁保持真实，不热搬活跃根，不改挂载、系统权限、凭据。没有足够实际写入能力的角色才BLOCKED；不用等待旧274.16/566.90GiB、NFS整批迁移或清理目标230–260GiB。已有55项授权内定点退役仅在有直接科研必要时处理，用户负责未授权资产的永久删除。保护69条唯一教师原件、337源、LIBERO1261、MAIN3.8K/R4/R5及必要M1000对照。145文件已退役不重复；A0/Habitat不作为科学前置。不启动Hopper CPU-only。

## 3. A：原独立HOLD六卡直接优先

沿V6.1已封存的N03/VAL-F1-G00两个Teacher合格且TRAIN无重叠的HOLD-blue布局、noise201/202，不重新挑布局。教师实际25.11mm/1.264s、25.06mm/1.206s；父MAIN3.8K N03两格已有效失败（x5_link6接蓝块），引用不重复。待运行逻辑槽：

| ID | 模型 | 布局 | noise |
| --- | --- | --- | --- |
| IH-R4-N03-201 | R4-200 | N03 | 201 |
| IH-R4-N03-202 | R4-200 | N03 | 202 |
| IH-P-VAL-201 | MAIN3.8K | VAL-F1-G00 | 201 |
| IH-R4-VAL-201 | R4-200 | VAL-F1-G00 | 201 |
| IH-P-VAL-202 | MAIN3.8K | VAL-F1-G00 | 202 |
| IH-R4-VAL-202 | R4-200 | VAL-F1-G00 | 202 |

每卡原35703tick/120前向；剩余六格最多214218tick/720前向，先单合法world，允许在已有独立资格下并发。实际请求ID映射原manifest，不生成新seed或重复已结算结果。原HOLD80mm定向底座、20mm抬升、1s保持、fresh feedback、原L1与评分全部不变。父门：既有N02 2/2与PLACE4/4抓持已经满足，独立R4四格有效结算、≥1/4完整且不低于同条件父；记录所有父成功→R4失败；相对父新增≥2个同机制抓前STOP阻断。完整PLACE成功非前置。若R4不合格，按原因登记新有限修复，不换测试人口或用M1000冒充父。

## 4. B：优先H200真实学习，不让单个Core环节拖住全部

### 4.1 首段两条互斥入口

标准：A通过 → R4全状态父放行 → 新域真实资格 → SOURCE250。

新V6.2条件入口：A尚未完成（现场评价与Hopper可同时开展）且无已知R4独立退化，R4已证明原N02 2/2及PLACE4/4抓持、0安全STOP，Core数据/reader/bank已真实封存，Hopper实际项目授权、GPU、数据、可写容量和完整恢复余量满足，则可由实际合法owner登记 **SOURCE250_PROVISIONAL**，运行原V6≤24工程资格，再最多250个科学形状的Adam更新。当前指令授权更新科学计划；现场执行需真实资源和合法owner，不能从PR推断取得站点权限。这个预放行段是探索性真实学习，不伪造父验收，不批准500后继。若A在训练中暴露退化，封存健康有界段后停止后继；完整性/授权异常按原安全收尾。先回读是否已存在相同作业，两个入口只能启动同一个首250，未知提交先对账。

探索和正式入口使用完全相同的父、数据、bank、RNG和学习率合同；A最终通过且250既定端点门也通过时，把已封存的同一前缀晋级为正式累计250，不重跑。失败则保留探索负结果、计入原V6累计成本，修复新版本仍受同一上限。禁止拿未验收的探索端点直接自动续500。

### 4.2 固定已准备好的科学配方

R4-200完整45/94叶、四Adam计数200及RNG作为父身份；新源单臂IO确定性初始化，新Adam/RNG，不用推理29数组拼完整状态。原R4-52正确标签，先冻结，不等待69全量新合同。337源（200旧、107新RoboDojo、30 Ville）实际收方训练reader及许可/独立split核验，原V6最低≥100新TRAIN/≥3任务组/真实新源维持，旧TEST probe污染原件隔离。新域工程≤24真实更新，至少每新域2真实梯度，完整配方6步（2warm4timed）、冷恢复下一更新，分开记账。工程后重新从共同科学初态启动，不把工程权重当父。

| 阶段 | Go2 | 单臂ARX5 | 双臂ARX5 | LIBERO | H40 aux |
| --- | ---: | ---: | ---: | ---: | ---: |
| SOURCE_EARLY | 16 | 16 | 80(40旧+40新) | 16 | 4 |
| SOURCE_EXPANDED | 16 | 64 | 32 | 16 | 4 |
| SOURCE_DUAL_ONLY | 16 | 0 | 96 | 16 | 4 |
| ADAPT_EARLY | 64 | 16 | 32 | 16 | 4 |
| ADAPT_EXPANDED | 64 | 32 | 16 | 16 | 4 |
| ADAPT_DUAL_ONLY | 64 | 0 | 48 | 16 | 4 |

128 FM，独立native H/时钟/codec/stat/mask，按B_group/128加权，分microbatch按真实样本数归约。原FM2:1:1、aux1，经实测aux/target=1/2，归一化SOURCE aux **1/16**、ADAPT **1/4**（若全体loss统一尺度改变只保比例，不重复缩放）。原4个真实完整H40、10步全反传、一次全局clip/Adam，FP32训练与Adam/BF16冻结基座、0.25状态dropout、共享Action Expert/VLM LoRA和独立domain IO。LR峰值 **1.6962141658312244e-6**，warm100，cosine到0.1peak at累计5000。首250银行4000 Go2、4000单臂、20000双臂（10000新/10000旧）、4000 LIBERO、1000 aux，均只是计划，实际账本独立回读。100步或30min先到及端点完整seal；PBS≤6h/≥15min保存预留，实际站点quota/保存重叠合格；恢复精确Adam/RNG/LR/银行，不每段warmup。

Core两批双臂冷读195.77/124.42s仅Core归档读取，不是H200吞吐。优先真实训练、实际测量Hopper data_wait/H2D/FM/aux/编译/保存及samples/s；原4workers/2prefetch、按需RAM缓存/独立reader，避免反复完整ZIP解压。必要Core预处理只针对真正瓶颈。有限Remat/独立冻结视觉子图优化在实际训练中进行，不重复失败R0/Nsight，不把B256/FP8或缓存VLM LoRA依赖当默认。优化未达目标不阻塞健康科学配方。Core负责所有数据、诊断和评价，Hopper仅H200后训练及直接训练准备（12 CPU/225GB），等待Core结果释放GPU，Vanda保持暂停。

### 4.3 250→500→1K→5K/10K

Core独立回读SOURCE250完整端点，固定原24反馈、六卡(N02两+PLACE四)及每新source≥16真实独立DEV窗。A已通过且端点完整恢复、真实新源访问、有限值、N02 2/2、PLACE正确抓持≥R4的4/4、无新增重复机制抓前STOP、源改善或可解释未收敛才续500。原250/500开发、1K正式、5K累计长程门保留。不能因低loss/H40代替闭环。源进步但Go2退化暂停SOURCE，优先实际归因/早期ADAPT，不盲长训。局部未达原门不事后放宽，另登记有证据修复。数据版本/配比只在冷边界（优先1K/5K）明确变更，同树可兼容的Adam/LR继续，不兼容新树显式新状态；单臂≥100合格且≥3独立技能才EXPANDED。条件5K→10K要求新增真实任务覆盖、源/目标进步与实际资源，明确新LR合同，非默认续跑20K。

ADAPT：实际69/100/200+Recovery40中通过正确标签的新数据，1K→3K→条件5K，源回放、新Adam/RNG、peak默认0.5 SOURCE、warm100、3K cosine，延长新日程。当前没有新的训练信号则不反复旧52跑5K。主线默认8000、条件15000累计实际更新（含探索失败前缀），原V6可选控制1000及工程24另列，已做800诊断单列；不重置旧320GPUh/2026-12-05 12:41:52 SGT。实际授权/期限不足封存报告，用户没有授权自动扩大实际站点额度。

## 5. C：释放问题独立科学专项，避免重复试错

R4两卡入区后132次返回均近闭爪；R5增加同回合释放/撤离访问却在24原R4入区反馈上0/24达到阈值。R5本身闭环仍未知，不能仅凭这些反馈决定最终性能。不要再次机械增加同52条的OPEN权重。研究：任务阶段识别、教师/学生观测支持偏移、真实放置监督和执行时序。禁止GT阶段切换、强制张爪/保持、人工搬动物体、删后缀、弱化支撑/撤离/接触。

- C0只读：复用既有24碰前、24入区固定反馈和132真实返回；额外预封存教师真实释放附近24窗、同任务仍需持物24窗，尽量跨合法独立几何，实际不足报告。R4/R5各比对同输入/noise/previous command，最多96新前向；已有同身份结果复用。比较raw H40/P10、G7/L1相位、有效mask、首张爪、教师状态与学生入区状态、原模型可见RGB/状态及仅供离线几何诊断的物体区域。若教师状态也不开，先查真实标签/phase/夹爪损失；若教师能开而学生不开，优先收集合法TRAIN偏离恢复；若raw开而未消费到，定位L1真实时序。不能把离线相关当已证因果。
- C1 R5原生：先P01/701、P02/701两张（原45703tick/160前向每卡），自身正常入口；若出现可解释新行为且无安全退化，补其余2 PLACE与2 N02 HOLD，最多6（254218tick/880前向）。旧同一模型条件不重复。若两卡无新行为即结算，不阻塞A/B，不当SOURCE新父。
- C2数据：优先V5独立TRAIN多构型完整搬运、下放、稳定支撑、释放、撤离；首10条Recovery原目标内选合适槽，最多原20请求，真实学生安全偏离→合法独立reset/前缀→教师纠正，只用真实消费支持，不用DEV归档当训练、不续STOP。只有C0/C1和新数据确认实质缺口时才新登记一个≤250更新修复对照，使用原可选1000容量（剩余相应扣减，不能再完整1000），同父与匹配对照、正确几何，不盲开新训练。完整PLACE阳性再原CONF独立评价。

## 6. D：数据工厂、跨本体与长程多任务

原69目标已补齐；31条Wave100新增真实技能为Reach/Orient8、MobileGrasp8、DiversePlace8、Container4、Push3。Embodiment100→200+Recovery40，原V5请求额度与几何组切分不重置，不凑数。优先跨构型安全接近、抬升、完整释放与撤离；先已有教师合格模板，必要时实际资格。全部新Go2使用R4正确逐查询逆映射、原mask/夹爪/真消费、数学往返与真实reader；17新增已有几何复演，未完成的运行时资格只补缺口。每4条/6h有变化/波末不可变封存并实际交接。新source/目标TRAIN/DEV/CONF/TEST隔离。

立即继续N1生产，不等待清盘/N2；N2真实控制、接触、selected-reference、STOP隔离和合格示范/小时达到原门才双卡1+1；N4条件2+2。原36/28资格容量保留；GPU冷渲染128份状态非128物理world，20CPU/62GiB历史容量现场核对，避免CPU过量。移动容器两个已有候选：高位搬运净空0.97/0.85mm，下探link6碰碗，高位释放约43mm落体仅静态证据；从合法初态真实Teacher验证碰撞、反弹、稳定承托、撤离。旧站姿0/2保留，不能移除碰撞体或强制释放。优先项目已有场景，复用已证许可资产，不重做失败缩放几何。

337→Mix500→Mix2K，优先单臂独立任务多样性，不只Footbag。Ville内胆/锅盖50Hz缺独立实测TCP/夹爪标定未准入，RoboTwin帧索引需确认时域，Table30/v2许可/语义与OpenNeo门控沿原合同；不把试读算TRAIN、不全量下载。真实READY可立即加入下一合法版本，不等待全配额。单臂100/3技能再提高占比。

长程科研终点：保留原V5 Drawer/Insert/组合目标未见任务的TRAIN排除与独立Teacher资格；先组成技能、再单一自然语言指令的顺序组合，正常入口、实际fresh反馈与消费、真实各阶段终态/顺序/安全/失败及恢复，不用脚本串联假装长程规划。R4、预登记SOURCE最终端点及ADAPT最终端点使用原独立人口，已完成同身份结果复用。源曾见任务叫目标域zero-shot而非全训练未见。原少样本5/10/20研究在零样本结果封存后另登记。源域误差不冒充Go2成功；匹配ADAPT-only控制按剩余容量、相同父/目标曝光、额外source计算差明确；严格source-first vs joint因果问题另固定共同SLA-base及匹配曝光。

## 7. 唯一调度、容量和实际交付

V6.1九端点六卡54逻辑，R4六卡均完成→48待新/映射。独立HOLD16逻辑中父N03两已完成→初始6待运行+最终SOURCE/ADAPT各4=14，主线剩余62新/映射学生。R5专项最多6（先2、条件追加4）为明示新诊断容量，若可映射旧未用槽则优先映射不重复；合计68新/映射学生；C0≤96新前向。原两个独立Teacher资格已复用、新资格0；V5其他生产/CONF容量不重置。技术/科学失败及实际成本全记，已终态不重复。单一合法owner：READBACK→READY→SUBMIT_ONCE→RECONCILE→SEALED→CORE_EVAL→DECISION。普通工程问题自主修复，不逐项用户GO；只对真正权限/资源/安全障碍停对应角色，不停整条主线。每个实际端点/最多12h报告新源优化曝光、实际完整任务分母、首次失败、有效新示范、剩余阻塞和下一合法任务。不能用盘点、文档、cold fixture、loss下降冒充科研进度。未授权删除不做，保持系统规则和用户数据。

本次发布：新增现场训练/仿真/下载/转存/删除均0；尚未部署新的存储admit策略或取得真实作业ID。现场以最新实际状态回读、授权与本合同执行。

[v61]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/221cc714f7648bb6f596745f9af1aaf1cae530d1/docs/protocols/SLA_HOPPER_V61_CORRECTED_RETENTION_PLAN_20261126.md
[v61reg]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/221cc714f7648bb6f596745f9af1aaf1cae530d1/docs/protocols/SLA_HOPPER_V61_CORRECTED_RETENTION_REGISTRY_20261126.json
[actual]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4de2985f05a9acf87566aa5b1dd363816bbdc34b/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/RETENTION_AND_SITE_ACTUAL_20261126.md
[factory]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6a10550dcc5beaf1e9a8c4180796198149d52023/reports/SLA_CORE_EMBODIMENT_FACTORY_V5/FIRST_SITE_ACTUAL_20261126.md
[storage]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6a10550dcc5beaf1e9a8c4180796198149d52023/reports/SLA_CORE_EMBODIMENT_FACTORY_V5/STORAGE_MAINTENANCE_ACTUAL_20261127.md
[layout]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6a10550dcc5beaf1e9a8c4180796198149d52023/scripts/sla/v5_storage_layout.py
[v5]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/2cbce375a4f5a01bedc4f065d5ad6581fe7cbb05/docs/protocols/SLA_CORE_EMBODIMENT_FACTORY_V5_PLAN_20261126.md
[v6]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d7c74953109374b69906036a0e2c6f39a32b3aa7/docs/protocols/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6_PLAN_20261126.md
[source]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4de2985f05a9acf87566aa5b1dd363816bbdc34b/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/FIRST_SITE_ACTUAL_20261126.md
