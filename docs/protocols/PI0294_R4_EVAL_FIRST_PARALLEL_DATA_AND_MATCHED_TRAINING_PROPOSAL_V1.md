# PI0294 R4：先完成策略评价，再并行数据生产与等预算学习对照

日期：2026-10-15。审核源码/报告基线：`5ae777a78495b043deb96b4d805e97d9cc46c4fc`。最后读取的科学进度是 #11 comment 6173476155（2026-10-15 00:46 SGT）。

**PROPOSED_AWAITING_EXPLICIT_ADOPTION；authorizes_execution=false。** 本文是用户请求的审核与下一长程计划，不自动终止/替换仍可能排队的 R3，不继承其额外 allocation 许可，不修改旧预算和科学结论。现有 incoming local-pc owner 继续，禁止创建竞争 owner。实际接手必须先刷新 #11/#47 和原生作业状态。

## 1. 已核实进度与并行工作的真实含义

| 问题 | 实际证据 | 审核结论 |
|---|---|---|
| 模型是否仍停在M512 | R3新增1536更新、49152 TRAIN曝光、24576 microbatches，cursor16384→65536，全局M2048 | 已真实续训，不是仅有计划或权重复制 |
| 全状态是否恢复 | 1030 tensor、共享embedding/lm_head别名、Adam/scheduler/RNG/sampler在更新前核验通过 | 不需从pristine重训；已解除此前M512独立恢复未知 |
| 参数与loss | VLM未变，537个可训练tensor变化；本轮前/后64步平均loss0.068083/0.032608 | 有优化进展，不是双任务成功率 |
| 两台机器是否同时工作 | EXECUTION_PROGRESS_2204显示Hopper已到544且CoreLab前两条S1 Teacher完成、后两条未完成 | 有真实并行重叠，但不是全程饱和利用的证明 |
| CoreLab本轮是否继续生产TRAIN | R3明文禁止新TRAIN；实际生产接触/预检/Teacher验证证据 | R3没有持续扩采，符合最近获批范围；不应称为持续数据工厂 |
| 4-worker是否实际实现并用过 | 父PI0294四worker真实物理/RGB/动作/审计一致，完成224次新数据尝试 | 是；不是fixture，也不是本轮R3新数据产量 |
| 吞吐 | 初始收据17.248/31.735/49.235 episode/h，后续统一口径17.194/31.600/49.171 | 都支持约2.86倍；短批次测量，不承诺长期每小时49条 |
| 训练DataLoader是否4worker | 原/R3固定micro2×accum16、workers0；CPU管线0/2/4资格已有 | 采集worker、加载worker、GPU数是三种不同资源；训练尚未采用多加载进程 |
| 安全修复是否已完成 | D0未改、R2_VEL_C01仍淘汰；S1四Teacher32s成功、零拒绝/主仿真违规，八上下文clone误差约1e-13 | S1是合格的仿真预检，不是D1修复/裸策略或WBC资格 |
| 模型评价是否已经完成 | 最新小收据：666401评价PBS排队，模型查询/模型live积分仍0 | R3的M512 TRAIN4、M2048 TRAIN4/DEV8尚无结果；不能写0%或PASS |
| 资产交付 | M2048公开脱敏完整状态25,541,238,816字节，READY 4c3097652bb81e43ea40a2b0b423086ec6ea7091，实际全内容回读 | 交付与数值训练成立，行为能力仍未验证 |

已有 v3.0：158 episodes，TRAIN75726/DEV19232 H40；已有 v3.1：295 episodes，TRAIN149048/DEV28247。v3.1包含v3.0，未训练且尚无本轮新交付证明。v3.1相对v3.0新增已准入TRAIN为122条（T1/T2各61），不是295条独立新增数据。原DEV和独立确认不进入训练。

审核是GitHub源码/收据级：未SSH现场复核、未重跑私有NPZ或模型。所有“排队/进行中”均是已发布快照，而非本会话原生状态测量。

## 2. 现在有了更具体的故障证据

R3 CONTACT_EVENT_ATLAS与CONSTRAINT_FORCE_BREAKDOWN已定位：
- A约1.058s出现right_pad-cube接触；末端右指接触广义力约+4.916N、关节限位约-4.105N，合力约+0.811N，电机约-0.589N。相同tick目标/实际TCP位置误差约22.76mm。
- B终态新增table-left_pad接触，正常力约4.702N；不是仅“保持全开便可避免风险”。
- 这些是原运行次序下仪器化数据；仍不证明唯一根因是模型、控制器或几何。

因此不再扫阻尼、加力或猜bias；保留原D0、S1和全部高频安全。首先看M2048行为是否改善。如果命令/执行/数据身份发现实质错误，暂停相关学习比较，给最小缺陷证据，不用更多数据掩盖错误。

## 3. 首要目标与非目标

首要目标：获得一个**经过真实新观测闭环评价的共享T1抓放/T2推送模型**，并用匹配训练预算回答“继续在v3.0训练”和“使用扩展场景/定向数据”哪个更有价值。

最小合格交付：完成R3剩余模型面板或保留已实际完成的面板、发布真实M2048分任务结果；v3.1成为Hopper可读取的训练视图；报告多worker实际采用与重叠时段。

完整目标：C0(v3.0)与条件C1(扩展TRAIN)从**同一M2048完整状态**各新增2048步至各自global4096，固定DEV选出一份候选，再做核心双任务确认和旧抓放回归。最终结果标明 `checkpoint + D0 + S1`。

不从头训练、不热切R3数据、不重做224条、不修改WBC/硬件/几何/摩擦/力限位、不做全模型解冻或loss搜索。不要把“机器一直忙”和“产生有用科学证据”等同。

## 4. R0：优先收齐R3，不产生重复评价或作业

1. 读取R3新进度、666401原生Q/R/F、owner/lease/预留/窗口。最新R3截止是2026-10-15 03:16:40新工作停止、04:16:40收尾；不要据本文自行延长。
2. 若该任务仍健康且在原许可内，沿原R3完成，不重提qsub、不kill排队作业抢资源、不热更新其源。新文档只发布，不变更其runtime。
3. 若R3已有16条模型面板，直接引用原始结果，不重跑求更好成绩。若已终态/过期，明确哪些NOT_RUN；在本新包被采用后，仅剩余条件以新attempt来源接续，不继承旧时钟/额度。
4. 同一共享输入/解码错误会阻断相关训练；只是S1拒绝或任务失败不禁止纯离线学习。主仿真出现新native safety则撤销该S1版本live许可，停止live臂，但没有共享输入问题时保留离线产物。
5. 在展开新增主训练前，必须取得现有M2048至少原TRAIN4/DEV8的结果或明确工程阻断。不能继续堆到M4096、再一次完全没有行为视频。M512缺失对照也按原名单补齐。

必须输出每任务成功、S1拒绝、native违规、已执行prefix、首次失败阶段、生成/安装/实际跟踪误差；全部预注册尝试保留分母。拒绝后不重抽噪声、不缩prefix、不改夹爪语义、不让Teacher接管。

## 5. 两路分工：CoreLab真正准备/生产下一版数据，Hopper持续学习与评估

### 5.1 唯一owner与不同队列

- local-pc：唯一任务/资源协调者和Git发布者，分别SSH两端；不要求服务器直接互连。
- CoreLab：复用 `scripts/pi0294/dispatcher.py` 与 `cpu_pipeline.py`，分别建 collection、audit/packing、live-eval 三个隔离队列与输出命名空间。一个总账分配CPU/磁盘，不能各自声称有16核可用。
- Hopper：一个科学GPU slot，训练和评估串行切换、预留评价时间；不把作业投到别人的现有allocation。

优先级：已有/新模型live-eval > 数据审计与可交付打包 > 新Teacher采集。live评估需要Core资源时在episode边界降低采集为1或0worker；不是修改正在推进的物理场景。

### 5.2 数据线D0：先消费已经付出成本的v3.1

先验证既有v3.1 manifest/文件来源和CPU准入，不重复全量重采/重新拟合它的统计。构建去重的TRAIN集合：v3.0 TRAIN 与 v3.1-only TRAIN；exact source group/episode/file hashes交叉核验。保留所有旧失败和DEV，不把配对场景另一任务挪到另一split。

先把v3.1已有增量传到Hopper本地scratch，使用已允许的私有固定版本存储或已有local-pc双端流式传输；不能由公开checkpoint许可推定可公开raw。private HF容量不足时只阻断该传输路径，不删除历史或偷偷改仓库可见性。大文件分片、摘要、原子READY及消费者抽样真实读取，最后核验全成员内容身份。

**v3.1的原统计不热替换M2048的v3.0统计。** 本轮用于数据贡献对照的训练视图继续使用v3.0固定processor/stats，从绝对原始命令进行一次官方转换，报告扩展数据预裁剪/逆变换clip和分任务/阶段误差。无法满足既有codec语义和准入阈值时，不通过放宽或按DEV重拟合来强推C1；交付视图BLOCKED，继续C0。原v3.1数据集和统计原件不改。

### 5.3 数据线D1：有限、定向、多worker新增，不无限扩采

目标最多24个全新TRAIN场景组，每组T1/T2同初态，仅指令不同，共48条完整episode；另有4条独立pilot（2组双任务），均不作为正式数据。

- 12组contact-approach覆盖：沿已有合格布局范围，小幅改变初始TCP横向位置各±5mm、增加0..10mm起始高度。只通过物理reset设置初态，不在运行中改qpos/物体。任务源/目标和原几何不变；覆盖接近、闭爪/接触、搬运/推送、释放、保持。
- 12组空间/视觉复合覆盖：复用已合格F1空间范围与F2材质/灯光，不新增形状、摩擦、桌高或未知相机极端位姿。不要引入只能从场景外观猜T1/T2的相关性。
- Teacher重用原算法；若需要支持新的初始TCP，先验证其阶段初始化/可达性，而不是贴上旧未来标签。所有命令、实际反馈和图像必须来自该次完整实际物理。
- 不从DEV/CONFIRM的失败位置复制训练初态。具体24组参数、seed、task pair在首次rollout前封存，来源只用TRAIN诊断/现有训练支持域，和所有旧DEV/CONFIRM做近重复检查。
- 单条32s、16000积分、641真实双观测、640已提交命令、601完整H40。独立高频原安全、接触、FK、时序和原生codec审计不减少。新失败原样保留，不替换seed。
- pilot要求4/4完整成功且无native违规，才开正式队列。pilot失败停止新数据线而不是边改Teacher边污染正式bank；已合格v3.1仍可交付。
- 新增包要求两任务各至少22/24准入，才进入本轮训练候选池；全部48首次尝试须有准确终态。不足则封存partial，不把它装成合格48条，也不阻断已有v3.1。

本轮不采DAgger轨迹或从S1危险终态继续接管数据；这需要额外因果干预及状态恢复资格。模型任务ID/Teacher阶段/物体真值只在审计/采样侧，不能进学生输入。

### 5.4 防止新数据再次拖住训练

在新R4 T0+3小时固定 `AUG_FREEZE`：当时已经完成上述准入的新增包可与v3.1-only TRAIN组成novel池；未完成就仅用已合格v3.1-only TRAIN。冻结决定只依据预先质量门和交付状态，不依据C0模型成绩。若v3.1尚未到Hopper，C1标DATA_NOT_READY，不让C0训练/评价等待。

C0可以在R0完成后立即训练，同时CoreLab继续采集/制作。冻结之后到达的数据只归档为下一版本，绝不改正在运行的C1池/统计。不得把C1晚启动时的最新目录自动视作冻结数据集。

## 6. 多worker采用规范：复用真实脚本，明确三种worker

既有dispatcher实际使用spawn、新解释器和渲染上下文，支持workers1/2/4，每采集worker3核，BLAS/OMP1线程、Mesa3线程；一个coordinator预留CPU并发审计。四worker最多12核，另给协调、审计、打包/评价合计4核；总计不超过16个实际许可核。

- 先运行新pilot时用2worker；原source/引擎/渲染身份未变无需重做全部历史并行证明，但必须验证本次真实并发PID、CPU affinity、独立run输出和相关上下文。
- 正式48条默认4worker；没有16核余量则用已允许2或1worker，不超售。旧树源变动要做最小回归再扩大，不把fixture运行计入生产。
- 采集和审计允许重叠，但一个写总账；worker所有first-attempt、PID/starttime、资源票据和输出独立。沿用原停止保护，不通过关闭审计获得加速。
- 主仿真与S1 shadow都记积分/CPU；不能把shadow轨迹作为真实采集观测/训练标签。

另一个概念是训练DataLoader workers。旧/R3训练为0，CPU0/2/4合格不是实际训练提速证据。R4在C0第一更新前最多60分钟做固定快照的0/2/4 loader对比：至少256个固定样本、多任务/阶段/来源，比较消费顺序、processor输出、恢复cursor与内存。优化候选只能为2或4，最多一个被采用；persistent/prefetch/cache均有界。

可以使用已核验本地不可变数据的有界RGB/NPZ缓存，完整性检查前置到入库并保留成员身份；不能静默接受被修改文件。不能换dtype、相对编码、随机增强或跳过mask。实际优化器测试总≤16隔离更新，两种路径从相同状态比较loss/梯度/更新/恢复；失败或无端到端收益就用workers0，不为加载优化阻断科研。

若新loader通过且端到端每更新墙钟改善至少5%，C0/C1都采用同一固定profile；否则两者都沿原profile。micro2×accum16保持，不再同时搜索microbatch；不切高效attention、torch.compile、VLM特征缓存、GPU向量化仿真或多GPU。

## 7. Hopper等预算训练：区分多训练的收益与多数据的收益

R0的真实M2048模型面板和输入/decoder契约成立后才进入。任务0成功不等于工程错误；记录实际失败阶段后允许这次有界学习比较，不能用无限步数回避错误归因。

| 对照 | 初始状态 | TRAIN采样 | 新更新 | 名称 |
|---|---|---|---|---|
| C0 | 同一完整M2048 | 原v3.0和原sampler连续cursor | 2048 | C0_V30_M4096 |
| C1（条件） | 再从同一完整M2048独立复制，不从C0接着训 | 任务/阶段权重不变；每任务50%v3.0、50%novel冻结池，组/episode优先避免窗口数量支配 | 2048 | C1_AUG_M4096 |

两臂固定官方N1.7、冻结VLM/训练projector+DiT、原Adam和6144horizon/308warmup、BF16/mathSDP、有效batch32、原processor/v3.0stats、同一合格loader。每臂65536新增样本曝光；保存其实际任务/阶段/旧新来源计数。

C0必须完整恢复M2048的65536消费cursor。C1继承M2048模型/Adam/scheduler/模型RNG，但由于改变数据池，**新建独立数据sampler状态并明确标DATA_INTERVENTION_FORK**；记录parent_cursor65536与新的branch_cursor，不能假装是原采样轨迹完全连续。C1采样种子固定为29404，不按DEV成绩改。

两份global4096属于不同分支，不能覆盖同一checkpoint目录或相加说训练到global6144；总新增正式更新最多4096。原M512/M2048保留。

C1只有novel池和原stats兼容、最低数据门、资源余量成立才执行；否则交C0与DATA_ARM_NOT_RUN。无第三配方、无M6144自动延期。公共衍生最多两份60GiB须本包采用时明确包含授权，raw仍私有。

## 8. 评价与候选冻结

1. 优先完成/复用原R3的M512 TRAIN4、M2048 TRAIN4/DEV8，不能因换任务号重复抽seed。
2. C0、C1在同一原DEV8面板各跑一次，包含F0/000、F1/000、F2/000、F1/001各T1/T2，同初态/配对噪声/同一D0+S1。M2048原DEV结果作为预先存在的第三候选基准。
3. 选择规则：先最大化min(T1成功率,T2成功率)，再macro成功率，再较低S1拒绝率，完全相同选较早训练步/无新增数据的较简单候选。只有真实完成DEV8的模型有资格参与。至少每任务2/4、无主仿真native违规才进入核心确认；不达则交开发失败，独立确认保持封存。
4. 选定唯一候选之后，按旧封存核心8组×双任务作16条Teacher资格和16条模型评价；Teacher每任务至少7/8且零真实安全违规，否则停止模型确认、报告该面板资格不足，不删Teacher失败条件。
5. 模型核心T1/T2各≥6/8、同组两任务皆成功≥5/8、旧抓放回归≥6/8、零live_native_failure，才标 `MULTITASK_SIM_WITH_S1_WORKING_V1`。每条S1拒绝仍是任务失败，不把零实际事故当policy安全学成。
6. S1是完整仿真状态特权预检，只拒绝不改动作；结果不叫裸策略/真机/WBC成功。选定模型也始终deployment=NONE。此轮不打开挑战集、不移植WBC。

固定任务成功需要原接触/抓举或无抓举推送/释放/撤离/保持与完整高频记录。视频必须是当时真实观测，不用成功Teacher视频替模型，不将短失败padding成32s。

## 9. 资源和PBS管理：不重蹈训练完成却永远没评价

建议本新包一次明确采用以下总额，未采用前不执行：新T0起12h墙钟，T0+11h停止新工作，最后1h收尾；所有重连不重置。最多3个新PBS调用（失败算次数），单项目同时1H200，总10 allocated GPU-hours，其中**至少3h预留模型评价**。若训练预计会侵占评价余量，不开C1或保存部分状态，不把所有配额都换成checkpoint。

账号总独立allocation上限3是本新包须明确采用的边界，不自动继承R3临时例外；其他任务全部保留。已有R3健康queued/running作业在其原范围优先完成。没有新有效权限时只读，不通过不同脚本绕过额度。

一次新allocation尽量覆盖待补评价、C0、DEV、条件C1、DEV；若Core数据未就绪，不让GPU空等，也不取消合法其他任务腾槽。队列原因、请求/实际资源、admission失败分别记录。已有合格本地checkpoint即可同节点评估，不以远端归档作为硬前置。用精确自身PID/作业终态收尾。

新增总限额：
- 正式更新≤4096，隔离工程更新≤16，H40生成意图≤6000（失败也计）；CoreCPU≤96coreh/并发≤16核，HopperCPU≤32coreh/按allocation许可。
- 新训练采集≤48、pilot≤4；全部main/Teacher/eval/diagnostic启动≤160，main积分≤2560000，真实双RGB≤102560对。
- S1 shadow推进≤10000调用、shadow积分≤2000000；合计所有积分≤4560000；停止与失败也完整计账。
- 新增峰值Core≤96GiB/Hopper≤180GiB/local≤4GiB；Hopper个人实际配额余量≥20GiB。按真实数据、模型、优化器、临时归档和缓存峰值估算，不能用shared df代替个人quota。
- 公共脱敏checkpoint最多2份/60GiB；原私有路径/数据/视频/mesh/WBC不公开。历史删除、WBC、硬件=0。

若R3运行跨入本窗口，先由同一owner明确阶段边界；实际开销只记一次，不能用新T0刷新R3配额。R4计划发布不暂停R3，也不替任何旧失败签PASS。

## 10. 一定要能审计的并行和效率指标

输出 `PIPELINE_TIMELINE.jsonl` 和 `PARALLEL_EXECUTION_AUDIT.json`：每worker PID/CPU affinity、episode、开始/结束UTC和单调时间、审计/导出/传输窗口、GPU排队/模型加载/实际训练/推理/空等时间。

分别报告：
- data_overlap = 数据采集或审计运行与Hopper实际优化区间的交集长度；分母为Hopper实际优化区间长度，不以总allocation作分母。
- eval_overlap和packing_overlap单列，不把shadow或Teacher资格叫新增TRAIN。
- 每小时准入独立episode、每准入episode CPU成本、各worker峰值内存、失败率、渲染/PNG/哈希/审计/I/O分项；启动开销与稳态吞吐分开。
- DataLoader num_workers和实际端到端更新吞吐、wait fraction，清楚区别于采集worker数。
- 原历史约2.86x只是参考，不要求通过夸大时间口径重现。多开核若无收益自动回退。

用这些证据回答“是否按计划并行”，不要只提交架构图或同时存在两个进程的截图。机器空闲有资源/安全/任务依赖原因时明确注明；不得为了提高占用率无目的采数据。

## 11. 阻断与交付

权限/共享输入损坏停止相关全部范围；新数据pilot失败只关闭该新数据线，既有v3.1/C0仍可继续；S1资格失败只关闭live模型臂，离线学习独立；真实live native安全失败停止该版本live臂；工程重试与已经成立的科学失败分开。

不新增Driver候选，不改变SafetyGate/高频阈值、接触/限位/动力学，不因推理不成功而回滚失败或临时重新拟合DEV统计。计划预算内逐门自动推进，不每个小阶段再要同项GO。

必交：`AUTHORITY_ADOPTION.json`、`R3_RECONCILIATION.json`、`R3_MODEL_PANEL_RESULTS.json`、`PARALLEL_EXECUTION_AUDIT.json`、`PIPELINE_TIMELINE.jsonl`、`V31_CONSUMER_READY.json`、`CONTACT48_PLAN_AND_ADMISSION.json`、`AUG_TRAIN_VIEW.json`、`DATALOADER_ACCELERATION.json`、两臂完整恢复/消费/训练收据、`MATCHED_DEV_RESULTS.json`、`CANDIDATE_SELECTION.json`、条件核心/回归结果、真实视频私有索引、资源总账和`FINAL_REPORT.md`。未运行项NOT_RUN，不制造空PASS。

GitHub只提交源码/协议/小收据，普通提交保留并发更新；固定commit与评论回读。大资产先保留完整本地状态、授权后去敏归档；归档失败不杀健康数值阶段。继续沿统一入口与现有运行库，不重写所有CLI或重装环境。

## 12. 审核来源（固定基线下的相对路径）

- `reports/PI0294/20261014T121640Z-incoming-pi0294-r3/M2048_TRAINING_RECEIPT.json`
- 同目录 `FULLSTATE_RESTORE.json`、`EXECUTION_PROGRESS_2204.json`、`CONTACT_EVENT_ATLAS.json`、`CONSTRAINT_FORCE_BREAKDOWN.json`、`TEACHER_S1_RESULTS.json`、`SHADOW_CLONE_EQUIVALENCE.json`
- `reports/PI0294/20261013T142457Z-localpc-pi0294/FINAL_REPORT.md`、`PARALLEL_THROUGHPUT.json`
- `scripts/pi0294/dispatcher.py`、`cpu_pipeline.py`，根AGENTS
- #11 comments6171867754、6172831740、6173476155；#47 comments6172832335、6173476841
- PyTorch官方DataLoader性能说明：https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide （通用机制；本项目固定版本须实际验证，不据此升级依赖）。

**本提案的最终优先级：已有模型真实评价 > 已花成本数据进入训练 > 可解释的等预算学习对照 > 小型定向新增数据 > 纯吞吐优化。并行用于缩短关键路径，不是让科研再次变成无休止的数据和工程建设。**
