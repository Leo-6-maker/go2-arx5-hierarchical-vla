# 当前科学状态 · 2026-10-28整理快照

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
| E1/E2 | E1资格原生2次优化、14目标参数实际变化、保存／完整恢复receipt；E2已有15零步输出 | 大型checkpoint成员独立流审计未闭合；E2优化0，两臂正式0/512 |
| Go2＋ARX5目标包 | 10段标定＋6几何双任务12首次尝试的独立提案 | 尚未采集新目标示范，不混入当前E1/E2 |
| DEC1冻结WBC迁移 | 原固定CORR2048/R0及设计保留 | core64=0/64；不是迁移失败率100% |

## 最新工程状态

2026-10-28 最新管理观测：[Core Home／NFS恢复核验](reports/PI0294/design/parallel-science-20261022/CORE_NFS_HOME_MOUNT_CURRENT_RECOVERY_AND_HISTORICAL_CAUSE_BOUNDARY_ACTUAL_20261028.json)已通过AMD现有key成功登录4080S；Home当前nfs4／autofs正常，两端authorized_keys全文SHA及权限一致。本地盘可用146,997,317,632 bytes、共享Home可用1,501,241,016,320 bytes；未占满。公钥读取确实依赖NFS，因此历史挂载／IO阻塞是合理可能，但普通账号无系统日志访问，历史根因仍UNKNOWN，未重挂载或修改服务。

Vanda同一1504095已真实F／Exit1／run_count1：[最新部分模型证据与关闭](reports/PI0294/design/parallel-science-20261022/ARX5_FIRST512_REAL_E0_E1_PARTIAL_AUDITED_SETTLED_TWO_CLOSED_AND_COLD_READ_FIX_ACTUAL_20261028.json)。E0完成39查询；E1资格原生完成2次优化调用、17查询、3次加载，保存step2／cursor64并完成原Trainer独立重建与恢复；独立小文件回读确认14个目标参数有有限梯度且实际变化、冻结不变receipt，不能代替大型checkpoint成员全文审计。E2完成15份零步输出，随后自己的qstat只读查询返回255，在FIRST_UPDATE放行前退出。正式E1/E2均0/512，不是闭环任务失败，也不重跑E0／E1来替代审计。

原owner完成真实F／worker与wrapper final／稳定完整inventory的partial审计，一次结算1917GPU秒／1222siteCPU秒、2更新／71查询／5加载；GLOBAL eaa268ed，累计180440GPU秒／14788更新／5908查询／PBS38。SETTLE原Journal40秒UNKNOWN，但原生intent／receipt／完整after字节独立证明CAS已成功；不重发。随后两own lease分别once关闭并独立回读，原owner／wrapper自然OS1且不再活动。所有旧成本、gross、peak和UNKNOWN保留。

额外有界只读CPU审计在180秒walltime到达时停止，OS−9／实际53.042269CPU秒／原120CPU上界保留，未产生RESULT；全部该actor已退出，不能称E1完整独立资格PASS。下一须按保存原件分阶段完成流审计并接续剩余科学suffix。冷修复仅让精确本作业qstat读故障在同一原180秒阶段内最多三次失败；READY和最终核验共用截止、保留25秒单读限制，身份／STOP／解析错误仍立即拒绝。10新CPU／49定向全部通过，初始假时钟fixture错误已修并保留，不称CI或原生部署。新source尚未采用，没有新request／qsub／模型／训练。PBS当前38＋一次剩余接续1＋matrix3为42／41，须完整有界接续范围与资源采用，不能静默挪用保护。

Core当前LOCAL仍为最后确认的91ce7cc7字节，尚未同步后续完整终态历史。D25旧冷manifest绑定1502754，不能把已F1关闭的1504095当成运行main；原once mirror拒绝历史落后必须保留。须在独立Core有限范围内冷态更新身份／完整LOCAL历史后才能原生启动，不能调用旧writer或依赖Vanda运行心跳。D25已交付模型不重传／重训，行为仍0/24。

[PC Rerun只读回放](scripts/pi0294/replay_viewer/README.md)已切换为官方Rerun＋薄Python适配器：[三条实际稀疏记录](reports/PI0294/design/parallel-science-20261022/PC_RERUN_THIN_ADAPTER_THREE_ACTUAL_SPARSE_REPLAYS_ACTUAL_20261028.json)含T1／T2 Teacher各41/641与C50 S1失败2/224保存状态，168张原PNG字节SHA／原时间轴／数值／审计均由官方读取器核对。11项定向测试通过；官方本地Web Viewer实际显示双RGB、记录点、曲线、审计并在16秒同步切换。原失败有224状态／225图像对，未补造状态或S1秒时间。完整关节／合法mesh仍未验证，C50成功未导入；旧自定义Viewer保留为历史且服务已停止。没有新物理、模型或评分，不是新能力证据。

此前[job1503589关闭原件](reports/PI0294/design/parallel-science-20261022/VANDA_FIRST512_SECOND_CONTINUATION_REAL_ENTRY_FAILURE_AUDITED_SETTLED_CLOSED_ACTUAL_20261028.json)保留：模型前旧profile常量拒绝，133GPU秒／240siteCPU秒一次结算、own lease CLOSED。所有历史失败、费用、缺失原件和UNKNOWN不改。

[first512首次失败](reports/PI0294/design/parallel-science-20261022/V5_D25_DELIVERED_FIRST512_F_EXIT1_AUDITED_SETTLED_AND_COLD_FIX_ACTUAL_20261028.json)为模型前身份／队列准入失败，0模型、0查询、0更新，旧作业和lease已关闭，不能重启或抹除费用。

[两站冷源码部署](reports/PI0294/design/parallel-science-20261022/UNIFIED_OWNER_BOTH_SITES_COLD_SOURCE_INSTALLED_ACTUAL_20261028.json)已经实际完成，但不等于新的评估和训练已经启动。

[最新D25有限范围采用](reports/PI0294/design/parallel-science-20261022/UNIFIED_D25_FINITE_DUAL_CONTROL_ADOPTED_AND_INDEPENDENT_SITES_ACTUAL_20261028.json)已在原双账本一次采用，完整D25＋first512有限后继复算无缺口；该回执仍明确没有新的模型、查询、优化或物理结果。运行时两站网络依赖必须解除，Core只评估、Vanda只微调。

## 当前决策

优先产出D25完整原24与E0/E1/E2真实模型资格，再完成GEN1、E1/E2行为评价。暂不自动扩采、扫描更多配方、训练柜门／书架或修改WBC。原R0诊断已结束，只有新的可复现缺陷才另立修复。

原后继范围的停止新训练时间：2026-10-31 20:52:29.813 SGT；硬截止：2026-11-01 20:52:29.813 SGT。此快照不延长时钟，不新增预算或科学许可。

完整增量见[原进度日志](docs/handover/2026-10-25/AUTONOMOUS_PROGRESS.md)。原状态长文[按原blob归档](docs/archive/2026-10-28/pre-consolidation/CURRENT_SCIENCE_STATUS.md)，不修改旧PASS/FAIL/UNKNOWN和原始指标。
