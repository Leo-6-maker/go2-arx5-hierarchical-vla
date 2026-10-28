# 当前科学状态 · 2026-10-29实际核验

**证据基线：`b6b5a28a7e75be889c898f7f116cabce174f7785`。这是仓库回执快照，不是实时服务器查询。** 后续实际回执优先；main合并不改变运行状态。总体研究未完成。

## 已完成的行为证据

| 模型 | 已完成训练 | T1抓放 | T2推物 | 首面板总体 |
|---|---|---:|---:|---:|
| A_CONTROL4096 | 4096新增更新 | 7/12 | 4/12 | 11/24，45.8% |
| B_COVERAGE4096 | 4096新增更新 | 5/12 | 1/12 | 6/24，25.0% |
| C50 | 1024新增更新 | 4/12 | 2/12 | 6/24，25.0% |

[A/B完整配对报告](docs/handover/2026-10-25/V4A_A24_B24_COMPLETE_FIRST_PANEL_RESULTS.md)：同N294101首银行、12个几何组，B−A为−20.8个百分点，几何block经验95%区间[−41.7,0.0]个百分点。没有显示本次覆盖替换收益；不是多场景普遍有害或正式选模的证明。

[C50完整终态](reports/PI0294/design/parallel-science-20261022/V5_C50_COMPLETE_ORIGINAL24_SIX_SUCCESSES_ONCE_SETTLED_THREE_CLOSED_AND_FIRST512_NATIVE_COLD_ACTUAL_20261028.json)：24有效、6成功、18有效失败（15次S1、1次参考门、2次完整任务失败），全部独立raw与因果审计成立、结算及三lease关闭。C50与A/B更新时程不同，仅作历史参照；D25−C50才是本轮混合比例主比较。

## 已有端点与待测研究

| 工作 | 实际证据 | 未完成／禁止推断 |
|---|---|---|
| D25 | 1024完整端点已审计、已固定交付Core | 0/24有效Eval；不能说成功率0% |
| GEN1 | 1024完整状态已审计并关闭；Teacher12次尝试、11成功、10条R0合格 | 0/24模型Eval；Teacher结果不是学生泛化 |
| E0 | 原始GR00T真实加载1次／39查询；39份输出独立SHA／shape／精度回读；所在作业已真实F并结算关闭 | 离线诊断，不是任务成功 |
| E1/E2 | E1资格原生2次优化、14目标参数实际变化、保存／完整恢复receipt；E2已有15零步输出 | E0/E1保存状态已完整独立流审计通过；E2优化0，两臂正式0/512 |
| Go2＋ARX5目标包 | 10段标定＋6几何双任务12首次尝试的独立提案 | 尚未采集新目标示范，不混入当前E1/E2 |
| DEC1冻结WBC迁移 | 原固定CORR2048/R0及设计保留 | core64=0/64；不是迁移失败率100% |

## 最新工程状态

2026-10-29后继冷接线：[剩余独立请求／物理custody／终态消费者](reports/PI0294/design/parallel-science-20261022/ARX5_REMAINING_CENTRAL_REQUEST_CUSTODY_TERMINAL_CANDIDATE_CPU_ACTUAL_20261029.json)已实现，保留实际非零前缀及旧费用；新三阶段仍仅1026更新／76查询／7加载。Vanda既有ACTIVE父控制目录已保存25文件候选差分，原Journal确认一次metadata写入，20项直接CPU断言通过，非pytest／完整原生登记资格。原本地pytest触及CPU限额的中断如实保留；未改运行capsule／GLOBAL，没有新request、qsub或模型效果。下一是复用缓存固定源码、实际custody／独占角色及全资源原生冷准入，不是追加前缀审计或重新传张量。两臂正式仍0／512，D25／GEN1仍各0／24。

2026-10-29后继冷接线：[剩余三阶段与队列CPU差分](reports/PI0294/design/parallel-science-20261022/ARX5_REMAINING_THREE_SESSION_EFFECTS_NINE_TICKETS_AND_QUEUE_CPU_DIFFERENTIAL_ACTUAL_20261029.json)新增独立的1026更新／76查询／7加载效果类型、九个原owner阶段门和复用原Trainer的worker。已保存15次E2零步输出仅作SHA及实际输入绑定的只读输入，不增加新查询链接；旧2更新／71查询／5加载全部保留。旧五阶段语义不变，99项定向CPU检查通过；此前异常类型fixture失败及31通过／23跳过的有界中断保留，非完整CI／原生模型资格。**中央登记、物理custody、运行选择及终态审计尚未接齐，新版本仍拒绝原生准入，没有新request或qsub。** Vanda实际读回93项原独立流审计witness的元数据及原raw审计，未重读张量；完整GLOBAL＋队列读取在原35秒内成功，实耗25.86CPU秒，确认旧15秒控制CPU不足，现仅在原已付1800秒控制范围使用30秒上界。Core管理文件安装UNKNOWN仍未核清，两次新增精确只读均35秒超时；没有重发安装或启动D25，不把超时当终止。D25／GEN1有效条件仍各0，正式两臂仍各0／512。

2026-10-29后继实际：[原64MiB控制计划及配对装配](reports/PI0294/design/parallel-science-20261022/D25_SAME64_NATIVE_DUAL_ADOPTED_AND_PAIRED_ASSEMBLY_FIXED_ACTUAL_20261029.json)已真实完成。GLOBAL `a4c5cf47`／LOCAL `cad80ab3`及原生intent、receipt独立回读匹配；仅将原未用只读差分64KiB明确转给冷配置，总额、CPU、历史UNKNOWN和终态／收尾预留不变。GLOBAL成功回包加SSH提示超过320字节，原UNKNOWN和费用上界保留，没有重发。Core已实际固定完整ASSEMBLY及两peer，运行源码仍`2a23eeb6`、原固定D25请求／模型不变。四项定向临时原Journal／onceIO与激活存储测试通过，服务器／GPU进程仍显式fixtures；不称原生模型资格。**GPU尚未启动，D25仍0/24。** 下一是完整有限启动接缝，再原180秒绑定／双采用和600秒80周期，不能把准备或CPU回归当作启动。

2026-10-29后继实际：[D25新源码与双账本登记](reports/PI0294/design/parallel-science-20261022/D25_REAL_DUAL_REGISTERED_THREE_COLD_ROLES_SOURCE_BOUND_NOT_STARTED_ACTUAL_20261029.json)已完成。源码`2a23eeb6`在两端1356文件全文SHA回读；原固定请求和nonce保留，新run04的owner、physical、neural三角色已真实登记。GLOBAL `0bf232b7`与Core LOCAL `dcc63c84`的原生CAS及独立回读匹配；GLOBAL回包35秒超时但实际CAS成功，原UNKNOWN和费用上界保留、未重发。独立读取33／38封存输入、源码和三ACTIVE lease，并实际确认owner／neural unit均not-found、MainPID0。**这仍是冷态登记：D25有效Eval为0，GPU、模型与物理未启动。** 剩余有限控制传输需先完成可行计划，再原生bind／完整双端资源／600秒80周期／模型放行；未动收尾保护、没有新增预算或重传D25。Vanda的剩余1026更新／76查询／7加载契约见PR103，尚未接成可提交入口，正式两臂仍0／512。

2026-10-28 最新管理观测：[Core Home／NFS恢复核验](reports/PI0294/design/parallel-science-20261022/CORE_NFS_HOME_MOUNT_CURRENT_RECOVERY_AND_HISTORICAL_CAUSE_BOUNDARY_ACTUAL_20261028.json)已通过AMD现有key成功登录4080S；Home当前nfs4／autofs正常，两端authorized_keys全文SHA及权限一致。本地盘可用146,997,317,632 bytes、共享Home可用1,501,241,016,320 bytes；未占满。公钥读取确实依赖NFS，因此历史挂载／IO阻塞是合理可能，但普通账号无系统日志访问，历史根因仍UNKNOWN，未重挂载或修改服务。

[2026-10-29最新实际资格与账本回执](reports/PI0294/design/parallel-science-20261022/ARX5_REAL_E0_E1_FULL_SAVED_PREFIX_AUDIT_AND_CANONICAL_CORE_HISTORY_CATCHUP_ACTUAL_20261029.json)：Vanda1504095原F1及一次费用结算、两原lease关闭不变。新登记有限CPU审计进程自然OS0并关闭own lease，独立流读取25,402,634,528 bytes保存状态与输出，另对同登录主机的原始父模型7文件完整流读取6,914,922,850 bytes。E0与E1/E2共有30对零步输出逐字节相同；E1完整step2/cursor64状态、AdamW step2、scheduler last_epoch2、四类RNG、14目标参数真实梯度／更新和冻结不变均核实；原Trainer重载前后固定输出SHA相同。实际新审计CPU97.185896秒／wall220.805秒，在原已支付480CPU／600wall内；无CUDA、模型、查询、更新或物理调用。此前180秒超时与两只读回包尺寸错误均保留，不改写为PASS。

剩余有界接续的原GLOBAL控制已一次采用：PBS41→42、3600siteCPU／600localCPU／64MiBsiteout／256KiBtransfer，GLOBAL f2c2b6da。旧实际累计仍180440GPU秒／14788更新／5908查询／PBS38；无新GPU请求或qsub。完整14维预测无shortfall，matrix3与最终20GPUh／40GiB不挪用。此控制采用不是剩余训练原生放行：E2优化仍0，正式E1/E2各0/512，不能重跑已完成E0/E1或从资格权重开始正式训练。

Core canonical LOCAL已从91ce7cc7一次补齐三次first512真实终态与已付控制分类，实际3d66e6d1、38条PBS行；原mirror字节与所有失败费用保留。Core/Vanda独立回读分别确认LOCAL3d66和GLOBALf2c2，14维预测总额没有变化。D25旧冷source66与manifest仍绑定旧请求，必须更新为诚实的新不可变源码／身份后再实际资格、加载和原24条件，不能直接调用旧writer。D25模型已交付，不重传／重训，行为仍0/24；两站运行继续独立。

[PC Rerun只读回放](scripts/pi0294/replay_viewer/README.md)已切换为官方Rerun＋薄Python适配器：[三条实际稀疏记录](reports/PI0294/design/parallel-science-20261022/PC_RERUN_THIN_ADAPTER_THREE_ACTUAL_SPARSE_REPLAYS_ACTUAL_20261028.json)含T1／T2 Teacher各41/641与C50 S1失败2/224保存状态，168张原PNG字节SHA／原时间轴／数值／审计均由官方读取器核对。11项定向测试通过；官方本地Web Viewer实际显示双RGB、记录点、曲线、审计并在16秒同步切换。原失败有224状态／225图像对，未补造状态或S1秒时间。完整关节／合法mesh仍未验证，C50成功未导入；旧自定义Viewer保留为历史且服务已停止。没有新物理、模型或评分，不是新能力证据。

此前[job1503589关闭原件](reports/PI0294/design/parallel-science-20261022/VANDA_FIRST512_SECOND_CONTINUATION_REAL_ENTRY_FAILURE_AUDITED_SETTLED_CLOSED_ACTUAL_20261028.json)保留：模型前旧profile常量拒绝，133GPU秒／240siteCPU秒一次结算、own lease CLOSED。所有历史失败、费用、缺失原件和UNKNOWN不改。

[first512首次失败](reports/PI0294/design/parallel-science-20261022/V5_D25_DELIVERED_FIRST512_F_EXIT1_AUDITED_SETTLED_AND_COLD_FIX_ACTUAL_20261028.json)为模型前身份／队列准入失败，0模型、0查询、0更新，旧作业和lease已关闭，不能重启或抹除费用。

[两站冷源码部署](reports/PI0294/design/parallel-science-20261022/UNIFIED_OWNER_BOTH_SITES_COLD_SOURCE_INSTALLED_ACTUAL_20261028.json)已经实际完成，但不等于新的评估和训练已经启动。

[最新D25有限范围采用](reports/PI0294/design/parallel-science-20261022/UNIFIED_D25_FINITE_DUAL_CONTROL_ADOPTED_AND_INDEPENDENT_SITES_ACTUAL_20261028.json)已在原双账本一次采用，完整D25＋first512有限后继复算无缺口；该回执仍明确没有新的模型、查询、优化或物理结果。运行时两站网络依赖必须解除，Core只评估、Vanda只微调。

## 当前决策

优先产出D25完整原24与E0/E1/E2真实模型资格，再完成GEN1、E1/E2行为评价。暂不自动扩采、扫描更多配方、训练柜门／书架或修改WBC。原R0诊断已结束，只有新的可复现缺陷才另立修复。

原后继范围的停止新训练时间：2026-10-31 20:52:29.813 SGT；硬截止：2026-11-01 20:52:29.813 SGT。此快照不延长时钟，不新增预算或科学许可。

完整增量见[原进度日志](docs/handover/2026-10-25/AUTONOMOUS_PROGRESS.md)。原状态长文[按原blob归档](docs/archive/2026-10-28/pre-consolidation/CURRENT_SCIENCE_STATUS.md)，不修改旧PASS/FAIL/UNKNOWN和原始指标。
