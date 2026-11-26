# SLA Hopper V4 执行手册：迁移、吞吐、独立评估

配套 [主计划](SLA_HOPPER_AUTONOMOUS_V4_PLAN_20261126.md) · [登记](SLA_HOPPER_AUTONOMOUS_V4_REGISTRY_20261126.json)。这是Codex现场接续指令，不是已启动capsule。沿用V7、V3原评分、安全、owner和实际权限；从实际可用旧入口绑定命令，不编造现场目录/argv。新计划分支只文档，不部署、不自动合并。

## 1. 开工顺序（可直接交给现场Codex）

从实际现役执行树读取 b1c5ae70e1d36ccddf2805d1ffa3391916113d3e 的 V3 两报告及最新现场增量。先核项目自有作业、Core服务、Hopper/Vanda授权、数据/模型和存储，冻结一个真实版本；确认旧attempt终态（V3 MAIN-02账本缺终态优先补）。已有健康任务继续，未知owner禁止复制。优先把MAIN_PLACE250未启动后继交给H200，原PT完成原250与八卡。Hopper资源就绪就开首个有限性能/训练capsule，不以其它路线未结束为借口停工。普通工程修复不需要每一步GO；真正站点授权、原安全门、实机范围不改变。

每块按 缺陷→证据→最小改动→预期→验证→真实后继 执行，记录实际作业ID与终态。不重建scheduler/approval ledger，不要求读回整个历史每次都重新哈希；可信不可变资产跨版本/站点验证，发生变化才重核。

## 2. 迁移及恢复合同

- 确认Hopper真实账户、project有效期/余额、合法队列、GPU UUID/显存/CPU/RAM、站点持久路径、quota和当前活跃作业。12:40旧读数不是实时状态。真实认证失效就用已授权正常认证入口，缺授权标BLOCKED，不假造或修改凭据。Vanda只用许可的数据路线；禁止旧CPU-only等待PBS。
- 原完整phase03800的140数组及Vanda完成清单一致是已读报告，现场恢复前核实际成员/大小/哈希/形状/optimizer/RNG/游标及对应执行SHA。3.9K部分写入目录禁止恢复；历史A/B是独立父，不串接。
- 区分三种包：(a) 完整三域训练状态，(b) V3 step0/new optimizer/固定19叶与样本计划，(c) Go2推理包（旧29数组）。(c)只用于Core评价，不用于三域恢复。PT官方父/LoRA包独立。
- 传输只读封存资产，按成员内容哈希去重/续传，接收端临时目录验证真实reader和同文件系统原子READY。node-local scratch只作为缓存，不当持久checkpoint。实际持久checkpoint先写同文件系统staging，fsync/完整清单/原子发布/独立读回，再备份Vanda；不写到在用文件。
- 空间准备以实际数组/缓存/临时文件/日志/配额最小值为准，至少预留本次完整保存staging＋下一完整保存＋在途传输/数据及安全退出所需空间，并将旧文件占用单列。历史5.55GB级140数组仅估算，不当当前精确值。配额不足给用户明确路径类别/需要量，不删除/覆盖旧checkpoint或私有数据。备份失败保留本地可恢复封存，按现场实际容量决定后继，不将partial标完成。
- PBS使用已有实际提交入口；单训练身份单writer、独立提交键，提交UNKNOWN先读队列和历史再决定。分段不跨原endpoint/owner/授权；恢复完整optimizer、RNG、sampler、last COMPLETE step及原定LR。旧已付未封存后缀按实际重放另计，不能把未更新的失败算已训练。
- H200→Core只导出绑定推理包/必要统计/processor/codec/运行版本哈希，已有Core canonical验证按实际同卡重载与原生入口复用；历史跨卡严格相等失败保留，不将跨设备bitwise相等作为新规范的无关障碍。真实模型、原数值门、原安全仍须核。新训练分支与推理时的L1身份分离，不热改现役capsule。

## 3. 多worker单learner：实现合同

优先修改实际V3 reader/训练入口或已有 `scripts/sla/hopper_engine.py` 对应版本。[已读实现](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b6869988b90851baf0e4ce207df3985fef1d2dc7/scripts/sla/hopper_engine.py) 的UpdateInputs用ThreadPoolExecutor(4)、每域Lock、两个logical batch；`_batched_fm` 每microbatch调用device_batch，`_batched_aux`每组同步。先查是否被实际入口调用，不能因代码存在而称已启用。

### 3.1 建议的数据通路（非已实现源码）

```text
central immutable visit plan(step,domain,position,episode,query,seed)
  -> bounded work queue
  -> CPU workers: read immutable shard / decode exact PTS / tokenize / transform
  -> ordered assembler (verify every assigned slot exactly once)
  -> two-buffer host -> device transfer, static shapes
  -> one JIT learner: per-domain batched FM / qualified aux
  -> weighted combined gradient, one clip, one optimizer update
  -> durable COMPLETE and cursor
```

worker只负责CPU prepare，不在各worker初始化CUDA learner或各自采随机数；有界queue满则背压，错误定位具体slot，不能丢/换样本。可用每worker只读reader/缓存，CPU解码线程限额与实际资源绑定。收集真实I/O wait、CPU PSS/RSS/PSI、页错误、GPU峰值，避免8×BLAS线程超配。确定性乱序完成后按visit position重排；恢复时discard未消费预取，按正式cursor重建，不把预取算优化访问。

逐个logical batch核对输出字节/有效数值、mask、真实noise/tau/dropout、任务phase权重、消费支持。加速不改变epoch/episode划分或实际图像增强；若要训练随机增广，先将随机变换身份按样本访问固定，缓存失效策略随之变化。

### 3.2 流匹配批归约

对域 d 的有效global batch B_d，microbatch实际行数m：
`g_d = sum_over_microbatches((m/B_d)*grad(mean_loss_microbatch))`，其中mean loss保持原每样本有效时间×动作维数归约；无支持样本不进正式batch。各域以实际旧2/1/1权重的既有归约组合，四个aux按m/4及原系数，最后一次global clip与Adam。不要对非均匀mask把每batch元素总和除错；若原损失不同，先复现原定义再版本化。每个有效target row可追溯真实consumed interval；不让mask掉的目标影响noisy input。辅助仍十步生成/全反传；不能为吞吐截断链。

先用同device同父同固定实际批的真实样本比较逐域FM/aux loss、全参数分项梯度、总梯度、finite/冻结叶；按预注册相对容差检查近零分母与绝对差（不隐瞒极小norm）。首步不合格不提交Adam；报告真实分项差和首偏离。核对随机数是否按原全batch draw、source padding/attention、bf16与fp32 accumulate、clip时机。若只是数值路径变化仍超门，旧等价失败保留；可另立科学profile，不事后修改该门。

### 3.3 性能测试顺序

先同配方参考全step计时，CPU worker1/4/8各16个真实batch（无Adam）。选择实际更低等待且不增加资源风险的worker数；默认4/2queue，8仅有收益。再REF→FM4→FM8→FM16，长期在最快合格FM上试AUX2/AUX4，cache最后单变量；无新证据的失败不重试。短训5×8≤40工程更新，长期7×8≤56；候选各自从同父新恢复，0–2步warm包含首步，后6步计时，全部工程checkpoint不作为正式训练起点。

使用真实同步边界测 wall/step p50、p95，报告GPU/CPU利用、实际H2D字节、编译、缓存构建/读写、FM/aux/同步、保存与恢复成本；缓存构建时间计入总时长，不能只比较缓存热态。最少一次冷重载和下一固定真实步/推理一致性检查，在同版本已存在实绩时复用。吞吐胜出以同配方参考1.25×以上、资源余量可持续、旧科学门通过为准，1.5×是目标不是保证。失败回落已合格参考；不因为工程调优延误正式250。

cache验证：固定8个真实窗口，逐个验证当前冻结prefix依赖、pad/history/attention mask、dtype、统计、输入和可训练参数。扰动action noise/tau及无效label，验证缓存部分不受影响、有效loss/梯度与未缓存真实网络满足原门；冻结部分才stop_gradient。缓存键至少含模型叶/processor/input/augmentation/mask/precision和shape。若状态token dropout改变prefix且无法正确按变体缓存则关闭该部分。长期VLM LoRA变更立即使依赖它的KV/hidden缓存失效。缓存命中低/构建成本不划算时保持非cache。PT若有后续迁移另验证相同规则，不能直接复用MAIN缓存。

## 4. 训练和评价执行顺序

### 4.1 V3 原人口（继承，不新增）

用原V3 registry、原任务、原顺序、实际noise与有限endpoint。MAIN250的八千次访问、19叶、目标侧mask及LR从实际V3合同封存；PT250 10 LoRA叶、有效32、250步、原短LR，若已完成不再训练。PLACE四卡＋N02两卡＋条件确认、PT八卡按原规格结算。V3六张教师已完成不能重填新成果；47格容量不重置。

### 4.2 已有权重比较 V4_AB_L1

三个冻结模型MAIN3.8K、A4.3K、B4.3K；各4 PLACE：P01/601、P02/602、P02/601、P01/602。先预封存条件、实际noise、原P01/P02物理资格和scorer；按每条件三模型轮转顺序（见登记）避免时间偏差。12格整块结算，首成功不取消其它已启用格。历史3.8K 0/4与A/B旧结果不作为新配对基线。若选择A/B作为长期父，只选已完整面板≥1完整PLACE、同L1的N02两原噪声回归2/2的候选；最多为一个事先按A→B顺序选出的已具PLACE阳性候选启用2格。若不通过，回落其它已就绪父，不另为挑选重复两卡。

### 4.3 新长期课程 V4_LONG

把长期父决策、所有域完整参数/optimizer初始化、源/目标bank、TRAIN成员与mask、目标采样、LR序列/seed、真实性能profile在首正式更新前固定。实际新数据若就绪，按计划60/40旧/新，旧组HOLD40/PLACE60；若无新技能合格，预封存旧组100%变体，另记录欠缺。首1000和累计2000使用同固定课程，1000后不能无声换bank；后续新版本才能接新READY。保留源域各原统计与实际源任务语义，Go2命令同当前query-frame，完整H40 aux支持和4槽的事前银行构建依据附回执。

首1000端点同一固定Core DEV6：原P01/P02×两新noise 701/702的4 PLACE，N02原两噪声HOLD2；同父6，长期M1000、匹配控制T1000、条件M2000各6，总24。控制1000必须与主1000匹配实际target visit/随机/损失、父、LR、可训练共享及target叶；源域无训练时源专属参数/Adam不更新。训练profile快慢或总成本差记录；非数值等价的后端不混作源数据唯一干预。

若第一新技能的TRAIN示范已进入主/控制的**同一**目标采样银行且DEV资格通过，额外用主1000与控制1000在该技能的D01/D02×两noise701/702各4，共8，评估匹配的多源增量收益；否则该8格不启用。预先选择技能顺序 PUSH→CONTAINER→DRAWER 中首个合格且已纳入共同TRAIN银行者，不能根据新成绩选。两臂同任务/物理/执行profile；这8格不是零样本。无控制完整端点不宣称因果收益。

1000学习结果和上述原任务开发评价：若至少一个完整PLACE阳性且N02两卡均成功，或与同期父在可比较任务上的阶段改善、TRAIN优化有效且未见明确HOLD/源域严重退化，则可按同固定2000日程继续；若无改善或机制不清，先分析raw/消费/实际、新数据覆盖与恢复状态；只有明确可解释未收敛/数据证据才延续，不以随机挑seed试好。源域watch至少复用既有固定三域96/96/96窗及真实原profile；报告每域/技能MAE和物理门、非闭环限制。不同源统计不横向比数值大小。

### 4.4 新技能 V4_NEW

对PUSH、CONTAINER、DRAWER分别封存两个DEV真实几何、一个目标与干扰物、合法接触对象、原完整成功谓词、独立TRAIN/DEV/CONF几何和真实安全配置。原资产不支持则标BLOCKED；新的允许接触定义仅限本任务原生合法接触，不降低通用硬门。教师资格6（每技能2），对应每个DEV布局两noise 801/802；学生候选MAIN3.8K、V3_PLACE250、OFFICIAL_PT_FROZEN最多36格；可用并确认未见目标技能才启动。PT不兼容则保留未启动，不为凑数更换任务物理。无源任务覆盖也可以测，但标SOURCE_UNSEEN/UNKNOWN；只对经过验证的source见过且target未训练才能称跨本体新技能迁移。正常未抓持入口、单初始指令、fresh closed-loop；同一冻结候选所有已启用格结算。

TRAIN首批每技能10个Teacher条件（5布局×2 reset；不能与DEV重叠），最多30启动，先达5条完整合格、≥2布局可进入下一训练；剩余未启动保留。几何先封存，允许复用兼容历史数据和明确的TRAIN-only恢复数据，完整与纠正前缀分开计。后续10/20条累计目标、新技能适配及长程组合另注册有限批次，不能把本登记容量解释为已准入示范。尽快建立一条共享多技能策略，不等待所有技能全部通过；实际源任务目录/访问证据与目标标签分类型，避免“已经多源，所以一定学会开抽屉”的假结论。

## 5. 多环境及慢动作诊断

复用旧N2/N4协议实际已完成的资格，先读back当前版本。未通过的F2 7.658mm瞬态及peer STOP缺口，检查selected reference、实际同钟状态、IPC/控制器cache与线程资源，不用时间平移/平均掩盖超限；不改变旧5.999mm。若缺真正新证据，保持N1；没有独立world真实owner/STOP证据不得并行生产。

单GPU batched policy server在已资格N2后单独绑定student profile：每请求携带world_id、query_id、checkpoint/processor/codec/actual_obs_tick、world-specific noise；单模型驻留，分shape合批，bounded wait，不能跨任务history混合或复用不同world previous_reference；late/STOP结果仅审计不install。固定真实输入对照单/批decode与原门，必要短原生隔离测试；正式评估沿用原任务，若批计算有数值/时序变化视新profile。N4依据实测完整合格轨迹/小时和资源，不以GPU利用率单独宣布成功。

速度从已有日志计算：T_reset/grasp/first_in_region/release/success、实际抓持持续窗、path length/Teacher同布局、source executed duration（去重atom）、ALIGN、实际query count、每query wall latency、native dt=2ms、realtime factor=sim/wall。成功终止截尾，失败时间不当成功耗时。所有时间按同native tick与实际wall定义；SOURCE和ALIGN分清，不把暂停仿真的GPU等待加到原生录像时间。只在证据支持下另注册查询时钟/轨迹效率试验，旧profile不变。

## 6. 有限容量与现场收尾

V4新增条件槽位118=已有权重比较12+候选HOLD2+长期端点评价24+新技能两1K对照8+新技能DEV资格6+新技能冻结评价36+新技能TRAIN教师30。模型/诊断82，零学生模型36；其中学生可能未启动/不兼容，容量不是分母。V3原47和旧N2/N4资格保持原身份不计此数；未来新技能适配/组合、PT后继确认等须新有限批次登记。V4新增长期正式最多3000更新，短/长期性能工程最多40/56；PT与MAIN原250按实际剩余量继承。原20K旧累计逻辑/已付成本单列。

新原生任务无可继承端点时先绑定50,000个2ms tick、最多160学生query、1800s wall，原安全/任务截止优先；若新技能需要不同合理有限端点，在首输出前按具体物理任务封存。Teacher零学生前向。训练job实际合法walltime（旧上限6h）按保存预留与p90决定，保留原总授权及期限；无需每小修人工GO。需要新授权或用户清理的真实 blocker 如实说明，继续不受影响工作。用已有有限运行入口和已有Journal/onceIO；不能无限后台、盲重复未知提交或自动续训练到无界。

每个工作块至少交付：当前真实code/父/数据/recipe与worker/CPU/GPU、实际样本访问和完成更新、性能指标与失败原件、完整原生分母和首失败、下一决定/作业终态。正式评估失败不重跑挑成功；技术修复另attempt，已付成本不抹。推送可公开的简报与索引至科学Issue #11 / 运维 #47 / 对应PR，保留私有原始数据、模型和机器身份在授权位置。计划不是已经部署的声明。

## 7. Codex 开工简短指令

执行 V4，但先核最新现场状态及 V3 01/02/03/PT实际端点。不要重复健康/未知作业。按 §2 恢复 Hopper 合法授权和可保存空间，优先迁未启动的 MAIN_PLACE250 到单 H200，先做最小 worker/batch资格，无法加速就用合法参考快速完成250。现役PT完成原250。Core保持原任务及WBC进行评估、归因和新技能教师资格，独立补N2/N4。就绪则从明确封存父启动V4长期1K多源，按实际结果推进2K与匹配控制；新技能不用等PLACE解决，但不训练评估集。所有普通工程问题自主修复，交付真实更新/原生成功/失败、资源/性能及下一有限块，而非新增大量空转文档。
