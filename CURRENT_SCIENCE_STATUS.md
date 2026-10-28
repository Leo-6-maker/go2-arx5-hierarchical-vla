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
| E0 | 固定原始父模型与有限探针输入已准备 | 无成功原生模型加载或神经诊断结果 |
| E1/E2 | 同12示范、参数白名单、匹配曝光与恢复代码已准备 | 两臂0/512；CPU检查不是GR00T训练资格 |
| Go2＋ARX5目标包 | 10段标定＋6几何双任务12首次尝试的独立提案 | 尚未采集新目标示范，不混入当前E1/E2 |
| DEC1冻结WBC迁移 | 原固定CORR2048/R0及设计保留 | core64=0/64；不是迁移失败率100% |

## 最新工程状态

2026-10-28 最新管理观测：[Core Home／NFS恢复核验](reports/PI0294/design/parallel-science-20261022/CORE_NFS_HOME_MOUNT_CURRENT_RECOVERY_AND_HISTORICAL_CAUSE_BOUNDARY_ACTUAL_20261028.json)已通过AMD现有key成功登录4080S；Home当前nfs4／autofs正常，两端authorized_keys全文SHA及权限一致。本地盘可用146,997,317,632 bytes、共享Home可用1,501,241,016,320 bytes；未占满。公钥读取确实依赖NFS，因此历史挂载／IO阻塞是合理可能，但普通账号无系统日志访问，历史根因仍UNKNOWN，未重挂载或修改服务。

Vanda新source7c7e77d4、retry03已完成原生冷输入核验和原GLOBAL请求登记；[REGISTER真实CAS回读](reports/PI0294/design/parallel-science-20261022/VANDA_FIRST512_ENTRY_REGISTER_CAS_CONFIRMED_NO_QSUB_OWNER_STOP_ACTUAL_20261028.json)证明登记已生效，但原Journal超过40秒后仍永久保留UNKNOWN，owner／wrapper实际OS1。尚未执行QSUB、没有job绑定，不能称新训练在跑；未重发登记或重启owner。原PBS计数38／上限41，matrix3及最终保护不动；下一接续必须新冷态绑定实际确认和fresh queue／quota，不复用过期队列。E0仍无神经结果，E1/E2各0/512。

[PC Rerun只读回放](scripts/pi0294/replay_viewer/README.md)已切换为官方Rerun＋薄Python适配器：[三条实际稀疏记录](reports/PI0294/design/parallel-science-20261022/PC_RERUN_THIN_ADAPTER_THREE_ACTUAL_SPARSE_REPLAYS_ACTUAL_20261028.json)含T1／T2 Teacher各41/641与C50 S1失败2/224保存状态，168张原PNG字节SHA／原时间轴／数值／审计均由官方读取器核对。11项定向测试通过；官方本地Web Viewer实际显示双RGB、记录点、曲线、审计并在16秒同步切换。原失败有224状态／225图像对，未补造状态或S1秒时间。完整关节／合法mesh仍未验证，C50成功未导入；旧自定义Viewer保留为历史且服务已停止。没有新物理、模型或评分，不是新能力证据。

此前[job1503589关闭原件](reports/PI0294/design/parallel-science-20261022/VANDA_FIRST512_SECOND_CONTINUATION_REAL_ENTRY_FAILURE_AUDITED_SETTLED_CLOSED_ACTUAL_20261028.json)保留：模型前旧profile常量拒绝，133GPU秒／240siteCPU秒一次结算、own lease CLOSED。所有历史失败、费用、缺失原件和UNKNOWN不改。

[first512首次失败](reports/PI0294/design/parallel-science-20261022/V5_D25_DELIVERED_FIRST512_F_EXIT1_AUDITED_SETTLED_AND_COLD_FIX_ACTUAL_20261028.json)为模型前身份／队列准入失败，0模型、0查询、0更新，旧作业和lease已关闭，不能重启或抹除费用。

[两站冷源码部署](reports/PI0294/design/parallel-science-20261022/UNIFIED_OWNER_BOTH_SITES_COLD_SOURCE_INSTALLED_ACTUAL_20261028.json)已经实际完成，但不等于新的评估和训练已经启动。

[最新D25有限范围采用](reports/PI0294/design/parallel-science-20261022/UNIFIED_D25_FINITE_DUAL_CONTROL_ADOPTED_AND_INDEPENDENT_SITES_ACTUAL_20261028.json)已在原双账本一次采用，完整D25＋first512有限后继复算无缺口；该回执仍明确没有新的模型、查询、优化或物理结果。运行时两站网络依赖必须解除，Core只评估、Vanda只微调。

## 当前决策

优先产出D25完整原24与E0/E1/E2真实模型资格，再完成GEN1、E1/E2行为评价。暂不自动扩采、扫描更多配方、训练柜门／书架或修改WBC。原R0诊断已结束，只有新的可复现缺陷才另立修复。

原后继范围的停止新训练时间：2026-10-31 20:52:29.813 SGT；硬截止：2026-11-01 20:52:29.813 SGT。此快照不延长时钟，不新增预算或科学许可。

完整增量见[原进度日志](docs/handover/2026-10-25/AUTONOMOUS_PROGRESS.md)。原状态长文[按原blob归档](docs/archive/2026-10-28/pre-consolidation/CURRENT_SCIENCE_STATUS.md)，不修改旧PASS/FAIL/UNKNOWN和原始指标。
