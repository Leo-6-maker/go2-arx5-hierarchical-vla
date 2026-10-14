# PI0294 长程目标审核与 M512 评价优先续接计划 V1

日期：2026-10-14。

**状态：REVIEW_COMPLETE / PROPOSED_AWAITING_NEW_EXECUTION_GO。** 用户本次要求审核完成情况并给出下一执行计划；这不是新的 PBS、GPU、物理运行或执行窗口授权。本文发布不启动任务，不复活原已关闭的夜间窗口。后续执行前须由用户明确采用本计划的新增预算和窗口。

## 1. 审核范围与总裁决

原执行计划实际编号为 **PI0294**：任务卡记载 PI0293 已有命名冲突，按既有规则改号。原科学 release `09e68e41752383fcb1087a23c52fe487c9a14a89`，run `20261013T142457Z-localpc-pi0294`。本次审核固定证据版本为 `50b75827e0935680e8af1a3cf04d3868997ed0cf`。

核对了原任务卡、FINAL_REPORT、DATASET_CARD、TRAINING_RECEIPT、三次早期失败、严格恢复资格、双 SSH 教师资格、评价面板、资源账本、M512 HF 失败及后续公共交付和 RESUME。属于源码/收据审查；没有重新执行仿真、模型、训练或独立下载全部 25.54 GB 重新计算远端 payload hash。

**裁决：数据生产目标基本实现，真实训练仅完成 M512，学习策略行为证据尚为空。接受 `PARTIAL_ACTUAL_GOAL_NOT_ACHIEVED`；不能判定模型已经可用，也不能把未测试写成模型任务质量失败。**

### 1.1 原目标逐项对照

| 目标 | 实际证据 | 审核 |
|---|---|---|
| 一份共享模型承担抓放和推送 | v3.0 两任务正式训练；任务不是模型选择器 | 结构/训练已实现，语言行为未验证 |
| 同初态、不同指令的多场景数据 | F0 基础、F1 空间、F2 视觉、F3 组合；224 新首次尝试全部终态 | 已完成计划尝试量 |
| v3.0 小包先训 | 80 新尝试，78 准入，复用 80 旧 episode；共 158 episode | 已合格、HF 私有交付、实际用于 M512 |
| CoreLab 在 Hopper 工作期间继续扩采 | v3.1 的 224 新尝试含 v3.0；215 准入、9 保留拒绝 | 实际发生，不是额外再采 224 条 |
| 真实 2/4 worker 并行 | 17.194/31.600/49.171 准入 episode/h | 1.838x/2.860x 实测；非普遍最优保证 |
| 原生模型 smoke/恢复 | 16 工程更新后，原容差下完整恢复 PASS | 已通过，保留前次 FAIL |
| M512 / M2048 / M4096 | M512=512 更新、16384 消费样本；后两者 NOT_RUN | 常规 4096 目标的 12.5%，不代表整体完成度 |
| M512 四条开发闭环 | generation queries=0 | NOT_RUN |
| DEV 选型、核心确认、挑战、旧任务回归、策略视频 | 全部无实际模型记录 | NOT_RUN；success/safety=UNKNOWN |
| 私有或公共 checkpoint 交付 | 公共脱敏 24 文件固定版本可见性/大小/SHA 核验 | 交付成立；独立公共副本恢复 NOT_RUN |
| WBC/移动 Go2/真机 | 未运行 | 保持范围外，不把固定底座结果扩大 |

不要给“总完成率 80%”之类加权不明的百分比。工程、数据、训练、任务行为及交付分别判定。

### 1.2 数据数量及质量

| 数据版本 | 新 T1 TRAIN/DEV | 新 T2 TRAIN/DEV | 复用旧 TRAIN/DEV | 全部 TRAIN/DEV episode | H40 TRAIN/DEV |
|---|---|---|---|---|---|
| v3.0 | 32/8 | 30/8 | 64/16 | 126/32 | 75726/19232 |
| v3.1 | 93/16 | 91/15 | 64/16 | 248/47 | 149048/28247 |

v3.1 的 215/224=95.9821% 是**新尝试准入率**，不等于模型成功率。112 个计划新场景组中，106 组两任务都准入，109 组至少一个任务准入。旧单任务组不计作完整双任务配对。

9 个拒绝：三个 TRAIN 配对组的 6 条静态 FOV 失败（零物理积分）；另 3 条 T2 安全完成物理任务但未过独立重构阈值 1e-5。保留拒绝、不换 seed、不改阈值。对静态失败与任务失败分栏，不把它们都称为推送失败。

v3.1 的全部 149048 TRAIN/28247 DEV 数值窗口、54 个分层真实 tensor 检查及 native 0/2/4-loader 消费/恢复已通过；初次 CPU timeout 留档。DEV 的 5 个 inverse-clipped 分量与 175 个 state preclip 分量披露。M512 仍只使用 v3.0 及其原 stats，未热切换。

v3.1 尚未 HF 交付。公共 `pi0294-public-artifacts` 是 checkpoint 的 dataset-type 存储仓库，不是新双任务 raw dataset 的公开发布。

### 1.3 M512 学习与资源事实

- 实际 task 消费 T1=8192、T2=8192；T1 中旧数据 891（10.8765%），新增 7301。
- 新场景族实际曝光 F0/F1/F2=5443/5439/5502；两任务阶段采样接近预设 20/25/25/20/10；真实 stage anchors、消费 trace 与 prefetch 隔离已记录。
- 冻结 backbone 不变，537 个可训练 tensor 发生更新。共享动作侧包括已封存配置中的 `tune_vlln`；不要只凭“projector+DiT”口头概括重新构造参数集。
- 实际 recipe：microbatch2 × accumulation16、workers0、BF16、原生 flow loss、确定性 `DETERMINISTIC_CUDA_MATH_SDP_V1`。
- **原实际 scheduler horizon=6144，warmup=308。** 常规目标到4096不等于应重建4096步scheduler。M512仅有204步在warmup之后。保留原scheduler/optimizer/RNG。
- 记录的 train loss=0.315059 不能推出抓放/推送成功率或 loss 已下降；没有模型任务结果。
- GPU allocated=3669s=1.01917h，仅占原12GPUh的8.493%；PBS calls=4/4、engineering updates=16/16。失效的是提交次数和旧窗口，不是12GPUh计算额度已用尽。
- CoreLab charged CPU=37.8424 core-hours；正式新尝试与审计约29.164 core-hours，包含拒绝成本，约0.135647 core-hours/准入新episode。
- 真实 M512 elapsed=2199.881s；loader wait=257.308s（11.70%）；同步优化器速率0.268837 updates/s，GPU allocated peak约35.4GB。microbatch4/8没有实际benchmark，不宣称现配置最快。

### 1.4 为什么停在 M512

| PBS | 实际结果 |
|---|---|
| 664410 | 离线 tokenizer 路径触发 remote-tag 查询；0更新 |
| 664414 | 已写checkpoint与剩余峰值被双重计数；6工程更新 |
| 664432 | 完整weight/optimizer恢复容差失败；2工程更新 |
| 664992 | 确定性修复后再8工程更新通过；正式512更新完成；后续私有HF preupload403令父任务结束 |

第三次失败不是已经证明样本顺序错：sample/cursor/RNG/noise/scheduler exact 项通过，weight/optimizer 数值未过门。后续确定性CUDA数学SDP模式在原容差下通过；不得为提速悄悄恢复先前未过门的核。

最后一次 main child exit0、delivery child exit1。当前编排把“远端HF交付成功”设成模型继续评估的硬前置，外部存储失败导致可用的本地 M512 未被评价。下一阶段应改变的是**外层运行依赖**，不是放宽模型/数据/安全检查。

DATASET_CARD 末尾仍有早期“M512交付受私有容量阻断”句子；应按最新 FINAL_REPORT/PUBLIC_CHECKPOINT_DELIVERY_ACTUAL 解释：公共副本已交付，原私有HTTP403没有修复。历史记录保留，追加状态订正。

## 2. 接受的资产与不能扩大解释的地方

### 2.1 固定身份

- 原训练数值代码：`a171997a704072e127d16410ba584c8aa0ddd573`。
- 原物理/数据 worker：`f1fdf3acd481687cb275334ba512ce038eba51e9`。
- 官方GR00T：`51d4c89f72fda44cbf77285c6a8114b52676b8a1`。
- 基模revision：`2fc962b973bccdd5d8ce4f67cc63b264d6886495`。
- v3.0 manifest SHA256：`b8ddcb9808e6dfac4f3ac5fa5a4828128fa9644a330c17d1c57bf833ce5889df`。
- v3.0 statistics SHA256：`fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7`。
- v3.0 私有HF READY revision：`6aa94f4f40d791bedd6b4e75489e4ede6504d10d`。
- 本地M512完整inventory：`6fdc10c2cfb13841bbabd444aa0ab8a4a6aef09adfbffae7d5fb4dee24747a8a`。
- 公共repo：`leoyuer/pi0294-public-artifacts`，repo_type=`dataset`。
- 公共prefix：`PI0294/20261013T142457Z-localpc-pi0294/public-M512-bc28535755c4`。
- 公共READY revision：`c698f54d0fc422ebfdd1e2ce1fc96552b6930c52`。
- 公共manifest SHA256：`bc28535755c4034c8c5f0cc10cb127cf46893f205f49aa663d6b1e417e5421f2`。

### 2.2 公共交付的准确含义

24 文件含16个完整Trainer成员，payload（不含READY）25541086052B。生产原件上传前后完整SHA通过；匿名消费者对大文件采用LFS SHA/size，小文件实际下载/计算hash。这是实际远端发布/内容身份检查，不是消费者把全部25.54GB下载后再次hash，也不是神经模型恢复测试。

公共衍生物的3个metadata文件已脱敏，数值model/optimizer/scheduler/RNG字节未变。公共metadata必须用公共inventory校验，不能拿私有metadata hash误判；允许的本地路径重绑只改role/path，不改训练字段或张量。Cosmos必须绑定已验证本地缓存，不能再次离线触发远端tag查询。公开归档采用dataset存储类型，不能直接当普通model repo根路径加载。

**主科学路径优先恢复仍存在的私有原件**，校验本地16成员、原config/processor/stats、step512、consumed16384和optimizer/scheduler/RNG。公共副本恢复独立列项；本地原件完好时，不要求为了启动推理再全量下载一次公共权重。

## 3. 下一执行目标：先评价 M512，再续训原数据

建议采用新续接run（parent指向原run与M512），保留全部旧计数和终局。**本文件不自动授予这项执行。** 新用户GO须明确新增PBS/GPU/物理预算与新窗口；旧09:00/10:00截止不得修改。

本轮唯一科学链：

`M512原完整状态 → 真实动作推理与四条已封存TRAIN开发闭环 → 同v3.0续训到M2048 → 原DEV8 → 同链到M4096 → 原DEV8 → 规则选定一个候选 → 核心确认/旧任务回归 → 有余量才跑挑战集`。

不从pristine重训，不换v3.1/statistics，不全量解冻，不换loss/normalizer/动作频率，不自动扩到M6144。若后续需要v3.1，单独做数据干预实验。新场景/新数据生产为0。

### P0：提交前CPU准备与外层修复

1. 单一local-pc owner从既有统一入口检查实际主机、项目CFP04-CF-007、自己的作业、个人quota、lease与固定16成员；不打断其它作业。
2. 复用已通过的全数据/模型工程资格；只做变化依赖的增量验证，不重跑全部177295窗口或原4-versus-2+2大smoke。
3. 原 `scripts.pi0294.neural` 等含已过期硬编码窗口，不能直接重跑。建立明确versioned续接入口：只参数化新authority/clock/增量ledger及local-checkpoint消费方式。保护旧sealed文件，列CODE_DELTA_ALLOWLIST并证明训练/decoder/teacher/scorer/数值路径不变。
4. 将检查点状态拆成 `LOCAL_CHECKPOINT_READY` 与 `REMOTE_ARCHIVE_READY`。本地完整状态原子保存、逐文件验证、manifest和可读性通过后即可同节点评估/续训；HF uploader在CPU侧异步或作业外运行。HTTP403只标归档BLOCKED，不自动杀死持有已验证本地状态的模型阶段。
5. 不删除任何未独立恢复或需要回退的唯一状态；预留M512、M2048、M4096及一次原子写入临时空间。跨节点没有实际权重的消费者仍必须通过传输/成员验证，不能凭LOCAL_READY远程假装就绪。
6. 用真实冻结观测做CPU消息/输入构造测试，确认model registry允许消费本地manifest；对错误role/旧tokenizer路径/重复episode/过期包/上传失败传播/配额重复计数增加小型回归测试。
7. 保留已有SSH通路。模型驻留worker仅在合法PBS allocation内读写既有文件通道，随job终止；不新开网络端口、隧道或外部常驻服务。不要每个0.4秒查询都重新加载整个3B模型。

### P1：M512 先产生行为证据（零正式新训练）

- 恢复M512并核验全部状态，但先仅eval。真实TRAIN观测的原生推理必须不带action标签、teacher phase、task ID路由或目标真值；使用实际policy observation-only构造和固定4步采样。
- 先在四条已封存开发episode、每条最多6个合法阶段anchor、两组固定noise做离线诊断，最多48条H40。相同专家committed轨迹经同decode/SafetyGate做阳性对照。
- 然后执行原M512面板4条：`PI0294/TRAIN/F0/000` 和 `PI0294/TRAIN/F1/000`，各T1/T2。面板hash `ff88c0c2a6475edd0684be7c53da77d13463d1de1133a5867419fce86b801b51`。不改为更简单seed；原未执行面板可以在新run有新attempt身份地首次评价。
- 每次query绑定最新双RGB/实测EEF，H40只装prefix8、推进200积分、收到真实ACK后再取新观测。不得重采noise直到接受、教师接管冒充成功、播放预生成动作或靠oracle提前停止。
- 交付4条原始视频、每任务完整评分/拒绝原因、generated→installed→achieved误差及语言配对行为。即使0/4也是新且有价值的实际结果，不是预先要求必须成功才允许训练。
- Gate在执行前拒绝属于受保护的模型失败；真实物理安全违规则停止该模型执行臂。若专家也被接口拒绝、加载身份/decoder错误或消息重复推进，先修工程并写新来源，不盲续训。
- 同一训练初态的T1/T2只改变语言；两个任务始终一套weights/processor/stats。明示任务字段只供评分和记账。

若本地原件不存在或明确需要公共副本恢复，按repo_type=dataset、固定revision与prefix获取缺失文件并实算字节SHA。可额外保留至多4个隔离工程更新，用于原始/公共副本相同M512状态各2步恢复对照；所得M514不作候选，正式训练仍从原M512开始。缓存复用与独立远端下载分别报告。未经测试不写公共restore PASS。

### P2：沿原链续训到 M2048，然后先评价

前提是P1的模型/接口链成立且没有未处理的真实安全错误。早期任务成功不足本身不等于工程失败。

恢复global_step=512、consumed_samples=16384、原Adam/scheduler/RNG/sampler。正式下一步513，不重新warmup。沿6144 horizon/308 warmup的**原保存scheduler**继续，阶段停止于2048。新增1536更新，不把它写成新训练2048。

参数集合、确定性数学SDP、TF32配置、BF16、原task/phase采样、microbatch2×16和workers0先保持不变。此次不重新做4/8 microbatch试验；效率优化不成为新前置项目。评估时保存并在续训前恢复训练RNG/模式，评估随机数不得改变随后训练样本和noise。

M2048完成后先局部manifest验证，再运行原8条DEV：F0/000、F1/000、F2/000、F1/001，每组T1/T2。HF可后台CPU发布或之后交付，不在这个节点再阻塞DEV。

### P3：M2048 → M4096，同一数据与状态；选定候选

无工程/真实安全阻断且余下时间足够保存完整状态与完成最低评估时，继续新增2048更新到4096，保留M2048。正式新增总上限3584，全球步数终点4096。M4096运行同一8条DEV，使用同样initial card、noise派生规则和原任务标准。

候选只在已完整评价的M2048/M4096中选择：先最大化较差任务DEV成功率，再最大化宏平均，再最少拒绝，平手选早期。M512仅开发诊断，不因观察到偶然更好表现临时加入选型。若时间只到M2048但其完整DEV存在，可按该规则形成可供探索性确认的单一候选，并明确4096训练目标未完成；不能称最佳迭代或达到全部原目标。

若两个任务在M2048均0成功，先检查接口阳性对照及保存的生成行为：工程错误停止；管线正确但明显欠拟合可在已批准预算内继续到4096。4096仍无任务证据时不追加6144，不改loss或数据来掩盖负结果，交精确诊断。

### P4：单一候选的独立闭环确认

使用原封存的核心8场景组、挑战8组、旧抓放回归8条。确认未曝光状态/原card hash须先核对；若已被他处使用，登记暴露，不把它继续称盲确认。

在候选选定并冻结后才解封核心/挑战；不拿这些结果反馈训练或更换候选。Teacher在同初态独立运行，每任务核心至少7/8、安全通过才解释对应模型确认。Teacher失败case不从模型分母删除。Teacher复验不等于模型成功。

先核心16条模型（8组×2任务），再旧抓放回归8条；最后仅在预算充足时挑战16条模型。Teacher可用已资格的独立进程并行，模型默认保留已验证串行ACK语义；不要同时改变模型batch推理/并行环境规则。

保持原工作策略最低标准：T1核心≥6/8、T2核心≥6/8、两指令都成功的组≥5/8、旧抓放回归≥6/8、真实安全违规0、Teacher每任务≥7/8。挑战单独报告。未尝试保留为NOT_RUN/coverage，不当实测0成功；同时给出计划分母和实际尝试分母，不能靠安全早停美化成功率。

只有实际满足以上门才报告 `MULTITASK_SIM_WORKING_V1`，含小样本/固定底座范围限制；仍不代表WBC、移动Go2、硬件部署或广泛统计显著性。

## 4. CoreLab/Hopper 并行推进，不再扩采竞争

| 时段 | Hopper关键路径 | CoreLab独立工作 |
|---|---|---|
| 提交前 | 项目/配额/离线依赖与M512状态检查 | 固定场景/入口/ACK readiness；只读v3.1归档准备 |
| P1 | M512动作推理及4条开发闭环 | 优先服务当前反馈循环和视频/评分 |
| 续训 | M2048/M4096更新 | 既有v3.1增量封包、manifest核对、任务视频整理；不重新codec全量验证或采新数据 |
| DEV/确认 | 模型驻留推理 | 暂停低优先I/O，保证环境/ACK，不抢占造成超时 |
| 收尾 | 本地完整状态与必要CPU归档 | per-task报告、数据索引、准确恢复说明 |

公共M512储存许可不自动允许公开v3.1 raw RGB/私有scene/mesh。v3.1仅在有现成合规私有容量/授权存储时增量交付；否则保留已校验本地原件和明确ARCHIVE_PENDING，不阻止v3.0模型测试。不得重复上传M512或把旧private repo改成public。

## 5. 建议新增授权包（等待用户采用）

原run的4 PBS、16工程更新、512正式更新、3669 allocated GPU秒及旧截止全部永久保留。新续接使用新run UUID和parent引用，状态PLANNED，直到明确GO。

- 新窗口：从新GO实际采用T0起10小时，最后60分钟只收尾；排队/断连占墙钟。与原夜间期限独立，不回填旧T0。
- 新PBS调用≤3，总新增H200 allocated≤8h、同时GR00T≤1GPU、账号独立GPU allocations≤2；不占/取消其它项目。建议主job≤6h，保留≤2h评价job；第三调用仅在已核销且未耗尽总GPU时作工程备用，不能再加8h。
- 新正式更新≤3584，到global4096；隔离恢复工程更新≤4，计账但不成为候选。M6144不在本次建议范围。
- 模型H40 generation queries≤5200，固定4步采样；拒绝也计查询，重复已完成query不重新抽样。
- CoreLab物理starts≤96、integrations≤1536000、RGB pairs≤61536；无新训练数据生产。最多覆盖2条必要relay复验、M512开发4、两终点DEV16、核心Teacher/模型32、回归8、挑战Teacher/模型32及2条有界工程预留；无需复验则不消耗。
- CoreLab新增CPU≤40 core-hours、并发≤8核；Hopper CPU≤16 core-hours并遵守实际PBS CPU分配。评价优先于打包。
- 新增峰值建议：Hopper≤120GiB、CoreLab≤32GiB、local-pc≤1GiB。容量按真实already-owned/new/transient求并集，不重复加已落盘checkpoint；启动前和保存前保留≥20GiB个人可用余量，未知quota不以共享df替代。
- 新公开checkpoint交付建议限已同意类别的至多2份脱敏衍生物、合计≤60GiB；该具体延伸应由新GO明确，未覆盖时仅保留本地状态。不给raw dataset增加公共授权，不删旧public/private历史。
- 不新增WBC/硬件/模型上CoreLab或login node、健康作业终止、历史模型/数据删除。

按已测M512 elapsed估算，3584剩余更新约4.28小时训练墙钟；同步step指标单独估算约3.70小时，差异主要为loader/其它开销。仅用于预算，不是完成时效承诺。按续接实际速度动态计算能否进入下阶段；必须给核心/回归预留至少2GPUh与收尾时间。不能为了凑4096把所有时间耗尽又交零评价。

## 6. 交付与停止原则

必须优先交出：M512真实生成数组、4条模型开发视频/评分；再交M2048/M4096完整状态、同一8条DEV结果、候选身份、核心/回归及有余量的挑战结果。任何阶段实际未运行都明确列出。

必要收据：ADOPTION_DELTA、LOCAL_CHECKPOINT_READY、CHECKPOINT_RESTORE_ACTUAL、M512_POLICY_EVAL、CONTINUATION_TRAINING_RECEIPT、PAIRED_DEV_RESULTS、CANDIDATE_SELECTION、CORE_CONFIRMATION、LEGACY_REGRESSION、CHALLENGE_RESULTS、ARCHIVE_STATUS、RESOURCE_LEDGER、FINAL_REPORT、RESUME。引用原物理/原模型/入口/数据/stats/source seal，运行变化与科学变化分列。

外部归档失败不得越过本地数据/模型完整性；本地完整性失败同样不能用已存在的远端READY忽略。任何真实安全违规立即停止相应模型物理臂；SafetyGate阈值不变，保留失败与未尝试，不教师接管后计成功。

计划目标不是生成更多测试、协议或checkpoint文件，而是第一次获得两技能策略的实际行为。若最终只到M512质量失败或管线错误，报告已获得的具体证据和唯一优先修复方向；不得宣称完成全部科学目标。

## 7. 固定证据索引

以下均位于固定GitHub版本 `50b75827e0935680e8af1a3cf04d3868997ed0cf`：

- 原任务卡：`reports/PI0294/design/GO_TASK_CARD.json`。
- 原协议：`docs/protocols/PI0294_MULTITASK_POLICY_AND_PARALLEL_DATA_OVERNIGHT_V1.md`。
- 终局及数据：`reports/PI0294/20261013T142457Z-localpc-pi0294/{FINAL_REPORT.md,DATASET_CARD.md,DATASET_MANIFEST.json}`。
- 训练及原失败：同run目录 `{TRAINING_RECEIPT.json,NATIVE_SMOKE_NUMERICAL_FAILURE.json,STRICT_REQUALIFICATION_ACTUAL_PASS.json,M512_HF_STORAGE_FAILURE_ACTUAL.json}`。
- 教师/面板：同run目录 `{TEACHER_DUAL_SSH_ACTUAL.json,FIXED_MODEL_EVAL_PANEL_RECEIPT.json}`。
- 交付/恢复/资源：同run目录 `{PUBLIC_CHECKPOINT_DELIVERY_ACTUAL.json,POLICY_CARD.md,RESUME.md,RESOURCE_LEDGER.json}`。
- 实际交付发布：Issue #11 comment6159094781，#47 comment6159094992。

本审核文件新增而不覆盖旧收据；用户批准新的执行范围前，所有拟议新增PBS、训练、物理和数据发布仍为0。
