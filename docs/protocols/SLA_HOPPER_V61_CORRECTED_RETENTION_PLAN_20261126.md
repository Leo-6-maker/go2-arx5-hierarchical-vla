# SLA V6.1：Corrected Representation & Retention-aware Multi-source Training

2026-11-26（SGT）。用户最新决定：依据 R4-200 的闭环阳性调整 V5/V6，恢复有界主线训练，扩大公共 ARX5 与 Go2 本体数据，Corelab 专职数据与评价，Hopper 专职后训练。**本文件只发布更新计划，不代表现场启动、父模型新准入、后继 SOURCE 更新或新数据产量。**

入口：[Runbook](SLA_HOPPER_V61_CORRECTED_RETENTION_RUNBOOK_20261126.md) · [Registry](SLA_HOPPER_V61_CORRECTED_RETENTION_REGISTRY_20261126.json)。本增补优先于 [V4 #519][v4]、[V5 #521][v5] 和 [V6 #522][v6] 中冲突的未来派发、目标标签、父模型和资源分工规定；不改历史方案或已完成科学结果。不另造大调度平台。普通工程问题自主处理，权限、存储、资源和原安全约束真实有效。

## 1. 最新实绩与明确边界

以 [R4最新终态][actual] 固定提交 ec927393bae025d536371ec88f237ba8de29ee3f 的 22:28 SGT 实际回读为准（并非本次实时 SSH）；原 [V5现场回执][core] 的 22:02 存储状态仍是最近所读生产证据。

| 事项 | 已证实 |
| --- | --- |
| M1000 | 原1000步已封存，6卡抓前x5_link2触桌；V4 2K未提交、原路线结算关闭 |
| R4数据 | 52条、4450查询、4294完整H40、174880有效几何动作点；逐查询逆映射最大位置残差6.46e-7 mm、转角5.54e-8°；夹爪、padding、mask保持 |
| R1/R2/R4 | 同一MAIN3.8K父各200步，共600真实更新；各两端点独立Core恢复。R3未提交、原200容量已用于R4 |
| R4首步 | 同父、样本、辅助槽和LR，仅几何六通道改变；Go2 FM 0.901728→0.010383，Go2 gradient L2 4.63506→0.10750，合并裁剪前9.36965→1.05697；其他三项loss/grad相同 |
| R4-200 | 24/24固定H40几何通过、2/24 raw夹爪接缝超限；5.2595s热态p50、5.4392s p95；完整状态45参数叶/94 Adam叶、四计数200、RNG |
| N02/201 | R4成功，25.80mm抬升、1.198s连续保持、23203tick/70前向 |
| N02/202 | R4成功，24.41mm抬升、1.268s连续保持、23453tick/71前向 |
| N02合计 | 2/2有效完整成功、0技术中断、0安全STOP、46656tick/141前向/0更新；独立重评分，GPU与world租约关闭 |
| 同tick link2净空 | 在旧M1000碰撞前最后共同tick，201父/M1000/R4=130.22/3.57/137.19mm；202=137.28/0.37/143.89mm；不是全episode最小净空 |
| 原L1 | R4真实每卡2个合法接续且完整消费P10，接续后16.47–37.12mm/s；一次raw接缝165.13→37.12mm/s。原40门未改，固定反馈并未覆盖所有真实触发窗 |
| 公共ARX5 | Core TRAIN337=原RoboDojo200+新增107+Ville30；最新Ville10条仍在Core，前一批已送达Hopper，历史双方实际读通317；新SOURCE优化访问0 |
| 目标数据 | 历史67条45 HOLD+22 PLACE、5892查询、5691完整H40，两站旧消费合同READY；新增15条尚无全部R4新几何准入证据；原TRAIN069五条与DEV重合排除 |
| Core生产 | V5新原生示范0；双EGL设备及冷渲染池已测，N2/N4实际控制和接触资格未闭合 |
| 资源 | R4三张Hopper训练job均正常退出；R4两张评价为有界容量独占部署；22:02系统卷95/274.2GiB、HDD534/566.9GiB、私有NFS约1172GiB余量；不能推断正式生产门已解除 |

R4训练封印 `f1f5a994005a9fc831934a156ca823d7e4b67794ce2533d72e087c2aeb29b9cb`；R4数据版本 `1b57d74fd133a3acbfb57bdce10a7ac4b8d8b18fd7fad8ca72376530edcb2a21`；R4独立恢复回执 `820b1ef46077355da86c85c448bd83e2b552ab199f138439d057e82f53f79741`；HOLD独立评分回执 `db77327c1ff1bd3d1dd759d51afb4b7ada4c5d230b0da937aa893d0d8fbb243d`。

结论：重复重锚是可复演的确定性错误，R4的单因素首步和同布局闭环保留支持修复方向；不能声称已经完全解释M1000六卡（200 vs1000步不同、闭环输入分叉、内部selected-reference仍缺），也不能宣称R4超过父模型或泛化完成。HOLD两份noise仅一个几何。当前方向从纠错诊断转到正确合同下扩展能力；R1/R2不继续同错误配方，R3暂缓，不再为占GPU而机械跑新消融。

## 2. 永久目标动作合同：消除重复重锚

沿实际部署的 FiftyMsProgram/G7/L1 固定版本；保留训练意图与接收器实际消费的两个不同语义。以当前query时刻实测TCP `p_a,R_a`、实际同钟接收器参考 `p_r,R_r`（不是任意IK解、未来状态或未确认的控制器内部参考）、真实已消费的目标 `p_c,R_c`：

```text
ExecuteMap(p_n,R_n) =
    (p_n + (p_r-p_a), R_r @ R_a.T @ R_n)
CorrectedTarget(p_c,R_c) =
    (p_c - (p_r-p_a), R_a @ R_r.T @ R_c)
```

使用真实现役源码的坐标、单位、乘法侧、episode固定mount变换、当前query锚点、时钟与所有必要变换顺序；公式是固定实绩映射，不是对任意SE(3)接口的通用替代。逐查询、逐有效点做逆像，再由原G7编码与父统计处理。对真实后续消费拼接出的H40后缀也以**当前查询**一致frame/参考进行复演，不把未来query的锚点混入当前输出；记录真实后缀来自未来反馈重规划的限制。保留实际消费原始几何单独字段，不修改或抹掉历史标签。不得直接将修订后名义表示称为真实已执行命令。学生运行时仍只执行一次原L1/FiftyMsProgram映射，不在provider叠第二次，也不将R4逆变换后处理M1000输出（旧固定反馈直接逆补偿出现+16.06→-34.38mm垂向漂移，未经闭环验证）。

### 2.1 数据版本准入（Corelab）

1. 封存R4-52原件、已证实数据SHA与全量复演。独立读取剩余15条真实消费轨迹的同clock实测TCP、接收器参考、真实消费/P10、RGB、previous actual gripper、mask和任务/几何。67条是消费准入库存，不是67条已经通过修订；五条原DEV重合继续排除。
2. 对全部有效几何点做实际执行map往返。沿原实现的float/serialization精度，记录max/p95分阶段残差。首个扩版准入预设保守工程容差：位置≤1e-4 mm、SO(3)角≤1e-5°（比既有R4实测余量宽，仅数学一致性，不是物理安全门）；任一点不满足就查实际frame/时钟/算法，不靠平均掩盖。保存原参考/标签哈希、实际逆映射、完整H40/P10支持、全有效行数。旧精确复演原件不重跑。
3. 对照原有效mask、夹爪和padding必须保持，实际Reader noisy input/loss对无效time×dim标签不敏感；对原父统计q01/q99逐维、逐阶段报告真实越界与极值。R4下降/闭爪P10第二旋转分量100%→0%越界；后缀仍2.59%/2.18%，不裁剪、不强制删去合法极端、不默改统计。若新任务与旧支持范围不可兼容，先明示新domain/统计版本及受控对照，不以放宽门限直接替换。
4. 在真实decoder/FiftyMsProgram/L1下复演完整支持H40、实际前一夹爪命令、接续事件与P10；几何还原并不等于全身无碰撞。不能从未来achieved、未消费后缀、STOP后、reset gap或末点复制生成目标监督。不能强迫抓闭、释放或保持。GT只供Teacher和审计，学生观测保持原真实body/wrist、EEF/gripper、指令。
5. 每4条新准入/6小时有变化/波末封存不可变版本，与原V5 catalog一对一关联，旧数据不重复计数。首次优先R4-67（若所有15条通过），实际不足时按真实成员命名，如R4-52；新SOURCE首段不必等待15条齐备，已正确52可保留为最低可用目标回放。必须在开始前锁定实际READY数据版本，不能在段内热换。
6. 以后所有Teacher/Recovery新数据采用相同准入，并记录原始接收器reference身份、真实时钟、底座/臂、PD与可获得的selected reference用于归因。不用不存在的controller-selected identity填空。源域RoboDojo、Ville、RoboTwin、LIBERO保持各自真实command/achieved/OSC与时钟、mask、stats，不套Go2逆变换。

## 3. 先验证R4父模型，完整PLACE不作学习前置目标

N02/201与202原R4 2/2已完成，直接引用，不再重复。R4为新的首选**候选父**；原MAIN3.8K完整父永久保留；原M1000/MAIN250失败不能成为新父。完成以下有限开发评价后才释放首个SOURCE：

- R4在原V4四张PLACE (P01/701,P02/702,P02/701,P01/702，精确ID按原registry绑定) 使用相同L1、WBC、安全、原任务评分、实际噪声。父已有同条件3/4抓持、0/4完整，直接引用，不为比较重复健康原卡。统计抓持、抬升、承托、入区、开爪时机、真实脱离/撤离、全身净空/首次失败、ALIGN/SOURCE消费、STOP。HOLD原定向底座位移、实际物体抬升/保持及fresh feedback合同不变；不能给PLACE强加未登记的HOLD判据。
- 新独立HOLD开发：在此前未作为TRAIN/上述N02选择的、真实Teacher合格的2个独立几何，每个2份预封存noise，父/R4各4卡（共8）。优先复用已有同几何同合同的教师资格及父模型终态。缺教师时最多2个新资格槽（每布局一条，计入原V5对应可用容量优先，不额外无限试）。几何与noise在任何学生结果前封存，任务不改，不能事后只选父成功的布局。
- 首个SOURCE新放行规则（本增补新增、执行前锁定）：N02既有2/2、原4张PLACE有效结算且R4至少3/4正确抓持；原生失败不出现相对父新增系统性抓前同机制碰撞（2个及以上同机制的抓前新安全STOP判定阻断）；独立4张有效结算，R4至少1/4完整HOLD且总成功数不低于同条件父，任何父成功→R4失败单格必须记录和归因；若净退化/安全问题或配对结果未齐，不直接开启大长训。若局部结果混合、无严重安全退化但未达到预设门，保留原始结果并另登记最小有因后继，不事后修改成功率/门限。该门是主线工程筛选，不是统计显著性结论。
- **不要求R4完整PLACE成功才能学习SOURCE。** 父原本0/4，不能将待学习目标设置成前置完成条件。PLACE出现完整阳性后，按原V3/V5独立确认合同推进。
- R4在原始raw seam仍可超限；沿用实际原L1接续，全部40原点保留、按真实时钟消费，不截断/强制夹爪、不改40mm/s。逐阶段记录原raw与安装轨迹、桥接后真实source点数/相位、下放和释放是否被延迟。真实HOLD阳性不等于已解决PLACE释放。暂不增加BASE head、不解冻WBC、不重设计夹爪头。
- 若需科学拆分标签因素：先复用原M200完整端点（仅实际存在且恢复身份通过才可用），固定已登记开发反馈和同训练步数；尚无同200步闭环结果则单独登记有界匹配对照，不能把M1000和R4的差直接当纯标签因果。该科学附加对照不阻断健康主线，旧R3的200步已转R4，不自动重新发放。

## 4. 公共数据扩充：库存337并非全部Hopper READY

Corelab完成公开数据下载/解码/源许可/时钟/FK/真实标签/任务分组/独立统计/reader及采样银行，Hopper只接不可变可直接训练包。延续[V6数据合同][v6]，不下载整个TB级仓库。

实际库存307 RoboDojo +30 Ville，其中最新10条仅在Core（封印 `5fef6594f2b00ce719d9a0fe25065d65bdb6563965568641dd27540b95fe7be7`，1249完整H40、2独立SOURCE_DEV/246窗）。上一批到Hopper并非已优化访问；必须实载并回读实际READY索引。目标仍Mix500→Mix2K（原400/500/500/400/200配额保留），缺口163是总数量差，不是原分项配额已满足。原V6最低开训门为旧200+≥100真正新增TRAIN、≥3真实技能/任务组及至少一个新来源/有效episode群；数量目前Core已达到137新增，但任务覆盖、许可、实际接收与reader、资源仍需真实绑定。可以使用实载的317/327/337等真实版本，绝不提前叫Mix500。

Ville30主要来自同一个Footbag任务；不能因单臂近本体就让狭窄任务长期占半数梯度。优先丰富单臂的独立任务类型：合法Table30/v2、Ville其它有明确EEF语义的任务、获准OpenNeoData single；RoboTwin现有4条只完成真实读取，时域待确认，不计TRAIN。旧RoboDojo新107已真实独立读回115223 H50窗；不将新source访问0写为学习已发生。保持许可、版本、原始file/member哈希、TRAIN/SOURCE_DEV/TEST组隔离；不将公共同源镜像重复计数。公共预处理全在Core，存储压力时使用原允许的私有归档通道并按真实空间分波，拒绝未经授权公开私有数据。

## 5. 新阶段采样与优化合同

保留128个FM样本、4个完整真实支持Go2 H40辅助槽及原10步全反传；原视觉/语言基座、WBC/OCS2/PD/腿、2ms冻结；共享Action Expert、VLM LoRA、独立domain IO按V6训练范围，原BF16冻结/FP32 trainable-grad-Adam、0.25 state-token dropout，正常有限数值差只记录而非一票否决，但真实mask、归约、冻结叶、完整恢复和闭环必须通过。

| 配方 | Go2 | Single ARX5 | Dual ARX5 | LIBERO | Aux |
| --- | ---: | ---: | ---: | ---: | ---: |
| SOURCE_EARLY | 16 | 16 | 80 | 16 | 4 |
| SOURCE_EXPANDED | 16 | 64 | 32 | 16 | 4 |
| SOURCE_DUAL_ONLY（无合格single才用） | 16 | 0 | 96 | 16 | 4 |
| ADAPT_EARLY（单臂仍窄时） | 64 | 16 | 32 | 16 | 4 |
| ADAPT_EXPANDED | 64 | 32 | 16 | 16 | 4 |
| ADAPT_DUAL_ONLY | 64 | 0 | 48 | 16 | 4 |

SOURCE首选EARLY；单臂至少100条实际READY TRAIN且至少3类可区分任务/技能组、非同一Footbag任务换物体/seed伪造多技能，并经Core确认独立几何/session与源保留后，才允许EXPANDED。若仅某源受阻而无真实单臂数据可用，明确DUAL_ONLY，不宣称单臂学习。ADAPT在同样窄单臂时采用EARLY作为本增补保守默认，宽度合格再用EXPANDED。每个group内按预封存task→episode→query/phase分层，明确子source配比、真实噪声/tau/访问；组内不同native H、camera mask或统计独立计算。可将80双臂按例如40+40或32+32+16分真实microbatch（依据显存/吞吐选择），按实际样本数加权，不多次Adam，不因为大batch改变总有效曝光。group loss权重B_group/128，EARLY SOURCE=0.125/0.125/0.625/0.125；其余按实际B。辅助独立，用实际旧target-FM与aux有效系数之比换算，旧归一化目标0.5时SOURCE aux×0.25，ADAPT恢复，未核实际λ不猜。16目标仍预留四个互异完整支持的aux槽，不补零后缀。缺源不能在同一段临时挪用槽位。

### 5.1 新父与SOURCE接力

父选R4-200完整训练状态（非29数组推理包），通过§3。若不通过，原MAIN3.8K保持历史阳性，但不能直接用旧错标签启动V6.1；按真实失败原因登记修订的有限后继。新增域与课程首次建立新optimizer/RNG/银行，保留R4全部旧参数初值及新domain初始化映射；新树不拼旧Adam。固定历史谱系、R4的200步单列。R1峰值8.4810708e-7为V4四分之一，据此V4峰值3.39242832e-6，SOURCE建议峰值1.69621416e-6（原V6 0.5倍，实际执行前绑定原账本，差异报告不任意填）；warm100，cosine到0.1peak at累计5000。当前训练范围保持，不同时切LoRA冻结、B256、不同精度或新动作表示。

固定新SOURCE端点 **250→500→1000→5000**，条件10000。先真实新域资格（沿V6≤24工程更新，与正式谱系分开），首250使用EARLY且真实新源实际访问>0。250/500是开发安全检查，1000为首正式端点。每段完整封存，Core执行固定反馈与原生配对，Hopper释放GPU，不保留CPU-only/PBS等待。保留100步/30min恢复点以及科学端点；同版本接续严格完整恢复Adam/RNG/LR/visit，绝非每段重新warmup。新READY数据/比例仅在已经封存的段边界（优先1000/5000）启用，绑定新课程/数据/模型域参数；同树且统计和优化定义可兼容时保留原Adam与LR时钟并登记新银行版本，否则新谱系/优化器/LR显式预封存。无新增数据不在运行段内热换，不把改课称严格固定配对；全部实际SOURCE更新计入同一V6上限，不重置预算。

250/500每端点固定原24反馈窗口及下述六卡开发面板，1000、5000及条件10000亦用同合同。最低延续门：完整恢复、正确新源实际消费、有限参数/梯度、N02 2/2、PLACE抓持数不低于R4基线，且无新增重复机制抓前STOP；raw H40/接缝异常按真实L1与实际行为归因，不以离线通过替代闭环。独立源DEV至少原V6每新source16真实窗，episode独立、固定噪声与stats，报告目标及源指标。SOURCE源指标有改善或有明确可解释未收敛且目标保持才继续；若源提升但目标退化，暂停长训，评估早期ADAPT或限定修复，不能盲续5K。轻微局部退化必须实际分析后另登记放行，不事后重新定义门。1000→5000优先已真实扩充任务、稳定保留，资源/期限足够；5000→10000须新增数据覆盖且源/目标实际结果支持，另绑定延长LR，不复活衰减日程。不能为5K目标长时间等待Mix2K，但数据仍窄且指标饱和时结束当前段。完整父与R4回退均保留，不挑最高分中间点替换预注册正式结果。

### 5.2 ADAPT

Core V5持续100→200＋40恢复。若具备新多样化R4-compatible Go2数据，依据SOURCE完整端点与能力保留选择新ADAPT父；不要求必须200齐备，但新数据不足如实命名，不在52条反复进行5K。新ADAPT 1000→3000→条件5000；峰值默认0.5×SOURCE声明峰值，warm100、cosine至0.1peak at3000，延长须明确新LR合同；源回放及技能/episode/阶段分层。新数据版本严格冷边界接入，恢复片段不计完整示范，不给固定DEV/TEST反馈生成训练标签。保持之前真正成功的HOLD、避免旧技能覆盖下降，增加多物体、移动接近和完整搬运释放。若仅目标标签修正有效、公共知识扩充还未证明，不将R4的成功归因新公共数据（R4新公共源访问0）。

## 6. 有界评价与实验账本

原V4 M1000失败、V3 PT失败、R4两张成功、R1/R2及R0两次技术失败、旧站姿容器0/2全部独立结算，绝不覆盖。V6原5端点六卡容量优先映射，新增250/500与条件扩展独立登记。本轮六卡panel候选预定9个：R4-200、SOURCE250/500/1000/5000/条件10000、ADAPT1000/3000/条件5000，共54逻辑卡，含R4两张既有终态，新/可映射上限52；每个候选4 PLACE+2 N02 HOLD。未启动/不满足条件不计完成。独立HOLD：父/R4各4=8；最终预定完成SOURCE端点与最终预定完成ADAPT端点各4=8（各阶段仅一次，在阶段结算前已按课程目标绑定，不按成绩任意选中间），合计新/可映射学生≤68，独立新Teacher资格≤2优先占原V5未用槽。所有角色原有结果可复用，去重实际请求身份，不能把容量叠加旧重复人口。上述开发门不等于独立CONF或真实统计泛化；保留V5 Drawer/Insert/组合与原未见任务确认、V3 PLACE独立CONF容量，实际合格且身份一致才引用；不把这些目标技能提前放进TRAIN。源已见而目标未见才称目标域zero-shot。

原V6可选匹配控制1000保持（共同父、相同target数据/曝光/LR，对照有无source扩充，额外计算差单列）；不自动启动已延期R3。原600步研究消融已完成，R4两卡已完成。默认新主线SOURCE5K+ADAPT3K=8000，条件上限10K+5K=15000；可选控制1000与新域工程24另计且继承V6容量，不重新批准旧预算。所有实际失败、技术启动、未封存更新、CPU和GPU资源分别记账。原N2/N4/V5数据采集等预算不变。每卡按原任务time/tick/query/safety（N02继承35703 tick/120前向），新独立HOLD也预先固定与原同合同；实际无合法任务不启动。

## 7. 严格站点分工与存储

**Corelab**：全部公共数据准备、标签修正/质量、Teacher/Recovery、Scene Factory、独立检查点/离线推理与原生闭环、所有CPU诊断和采样银行生成。实际GPU0/1历史显存32760/16376MiB，R4 GPU0实测约24GiB，20 CPU核、62GiB RAM，已资格N1每world4核。双卡冷态EGL与128 MjData渲染不等于128物理world；N1生产继续按真实资源；优先补N2控制/selected reference/接触/peer STOP，合格才双卡1+1再2+2，学生混合负载单独资格。不要将CPU解码与world超配。已有四个冷编译Reach/Mobile场景、RoboCasa FruitBowl资产及侧向容器候选优先真实Teacher资格；站姿旧0/2不改。V5 67+33=100（其中legacy补缺2及31新技能，真实成员准入后计）；最终200+40纠正保持，不能用旧未修标签计新数据。

**Hopper**：仅真实H200训练/直接服务训练的有限GPU性能工程；12CPU/225GB RAM为训练必要读取、缓存、预取、JIT、保存、恢复和输入交接。禁止为独立CPU-only准备、数据下载/审计、原生/离线评价或等待任务占用Hopper PBS。训练收方必要的reader核验、完整恢复是实际训练的一部分；Core负责独立诊断。健康合法GPU训练继续；数据/评价等待期间释放GPU。只在Core有真正READY及门满足后派发。保持Vanda暂停。沿已授权通道交接，不公开私有资产或凭据。

当前系统卷95 vs274.2GiB、HDD534 vs566.9GiB仅是22:02读数；22:28 R4有界两卡已完结不等于全生产门合格。首先现场按实际每卷free/quota、在途和保留门重查。用户负责历史文件清理/迁移授权；不自行删除、移动或降低既有全局生产门。个别小型评价仅在真实明确授权、有独立有限容量、全程预留、终态与安全保证时执行，不把R4先例变成普遍豁免；不具备实际资源就保持READY，继续获准CPU工作或合法新源准备。私有NFS约1172GiB历史可用不是当前可用承诺。无安全空间不启动Teacher或批量评价；不把Hopper变成CPU替代站。

训练性能：维持现有约5.26s/步参考，不重复已失败R0；有界真实训练优化Remat、经证明独立冻结视觉子图、GPU传输和整域batch，CPU可准备的cache在Core。VLM LoRA依赖跨步缓存禁止；B256和FP8等新数学配方另登记，非本次默认。以同真实配方吞吐/恢复测量。旧R4实际约21.5min/200含开销；以5.2595s热态估算250约21.9min、1000约87.7min、5000约7.3h，不是新增四域实测。每PBS≤6h，保存预留≥15min并核实际p95、恢复/编译/写盘；阶段可按短实际时限申请，不占卡等候。原320GPUh和2026-12-05 12:41:52 SGT截止不重置，实际授权及旧已耗资源重新对账。原正式1K已结算，不续M1000。唯一合法owner、intent、实际jobID、UNKNOWN先对账、不重复提交。

## 8. 最先交付

1. Core实际资源、R4完整父与标签合同比对；旧15条逐项准入，同时补PLACE4和独立HOLD配对，复用N02两张，不重复已完成。
2. 真实公共READY（不必满500，实际337/更小子集按真实身份），最低任务覆盖/许可/新域reader，释放首个SOURCE。R4完整训练父、新域初始化、数据/银行、LR、batch/aux系数、PBS jobID全部绑定。
3. 首250实际训练更新和新源访问、完整checkpoint、Core回读与原生结果，再500/1000；新数据生产不断线但服从资源，先N1再N2/N4。
4. 有真实保留和数据/源知识改善才5K、条件10K与ADAPT；同步实际数据多样性、闭环和新技能迁移，禁止仅靠loss/H40或步数宣称成功。

本次只发布计划：0现场GPU/CPU任务、0下载/训练访问、0新原生请求、0删除/搬迁。实际现场执行由有权限的Codex按Runbook自主进行。

[actual]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/ec927393bae025d536371ec88f237ba8de29ee3f/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/RETENTION_AND_SITE_ACTUAL_20261126.md
[core]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/fe21674f20ffb37538bae33646122adf25e5cdb2/reports/SLA_CORE_EMBODIMENT_FACTORY_V5/FIRST_SITE_ACTUAL_20261126.md
[v4]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f72f3d57b470de3e62821cd5e1a33d938a669cb7/docs/protocols/SLA_HOPPER_AUTONOMOUS_V4_PLAN_20261126.md
[v4reg]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f72f3d57b470de3e62821cd5e1a33d938a669cb7/docs/protocols/SLA_HOPPER_AUTONOMOUS_V4_REGISTRY_20261126.json
[v5]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/2cbce375a4f5a01bedc4f065d5ad6581fe7cbb05/docs/protocols/SLA_CORE_EMBODIMENT_FACTORY_V5_PLAN_20261126.md
[v6]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d7c74953109374b69906036a0e2c6f39a32b3aa7/docs/protocols/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6_PLAN_20261126.md
[v6reg]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d7c74953109374b69906036a0e2c6f39a32b3aa7/docs/protocols/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6_REGISTRY_20261126.json
[m1000]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d806ab813e34ee1e3f3a6645409fa4c2402ece7d/reports/SLA_EXPERIMENT_PROGRESS/V4_HOPPER_HIGH_BATCH_20261126.md
[ville]: https://huggingface.co/datasets/villekuosmanen/drop_footbag_into_dice_tower/tree/cb6f2fb33737afba63852784d71db35c17076adb
