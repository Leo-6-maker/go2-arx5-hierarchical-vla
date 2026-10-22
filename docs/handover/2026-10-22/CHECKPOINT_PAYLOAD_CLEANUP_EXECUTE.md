# 本人 GR00T-WBC 失败 Policy 权重清理与有效 checkpoint 保护

日期：2026-10-22，Asia/Singapore。ID：CKPT_PAYLOAD_CLEANUP_20261022。
状态：USER_AUTHORIZED_SCOPED_DELETION_AFTER_EXACT_NATIVE_CHECKS；本文件不是服务器 unlink 收据。
本修订优先于本文件旧版的 PI05/vla_wbc 删除候选、旧 Issue #47 comment 6334550091 中相应 PI05 建议。

## 0. 用户已明确划定的所有权

**唯一允许本轮清理的项目根，是用户本人登记的 Projects/GR00T-WBC 及经过该项目真实私有 registry 证明为自有的实验输出。** 由当前现有唯一 maintenance/science owner 依 host-aware registry 确认 mount、UID、真实路径与权限后操作。

**vla_wbc（包括 PI05 的 s6000_alltrain 1000/2000/3000/4000/5000/5999）属于他人：全部 EXCLUDED_NO_INSPECTION_OR_DELETION。** 即使某文件与本账号、路径名或先前盘点重合，本次也禁止触碰、申请将其纳入本项目、轮转或删除；先前 PI05 约129.4GB 的预计释放量一律废止，绝不是本项目可回收空间。其他不属于本人登记 GR00T-WBC 的数据、scratch、用户或项目也同样排除。

用户明确认可：**已终态、明确废弃、不会用于恢复/选模/基线/评估/部署/有效科学复现、没有当前/将来已登记消费者的失败 Policy，直接删除其大模型和 optimizer 载荷；只保留 config、日志、任务指标、源码/数据/statistics/模型身份、失败、费用、manifest与可核验的退役记录。无须为废弃权重另建一份大型备份，也不再逐个请求相同删除授权。** “策略任务失败”或“作业 Exit1”本身不构成删除判据：任何有效且已引用的 checkpoint 必须保留。

## 1. 2026-10-22 已执行收据：不可混作未来腾挪

依据 #47/6336784045，本人登记项目实际已完成一次小型安全清理：6个发布已完成的 Xet cache MDB，逻辑19,066,528B，allocated **19,079,168B**（约18.19MiB），所有unlink成功；发布 payload、log、HF READY/消费者读回均保留。

权威本人 scratch quota **881,689,518,080 / 1,099,511,627,776B**，同次删前后官方报告未显示下降；因此只说19,079,168B已unlink的原分配块，不说quota下降、也不将PI05的129GB算作可用。本项目 experiments 快速目录检查约24KiB；全 runs 树扫描超时，故无GB级无引用失败载荷已被实际证明，也不能编造可删量。旧清理记录和历史账不得重写或退款。

## 2. 按真实科学作用判断删除优先级

| 对象 | 决策 | 保留/删除理由 |
|---|---|---|
| 本人 GR00T-WBC 中已正式放弃、未选中、无baseline/teacher/论文复现/任何当前及封存future引用的失败 Policy 全重/adapter/optimizer/训练分片 | **DELETE_ON_NATIVE_PROOF** | 保留小型配置、模型来源、loss/任务指标、失败和账本；废弃载荷可直接unlink，无大备份要求。 |
| 本人已 CLOSED/SETTLED 的真正部分写坏的 SAVE_FAILED、临时权重/optimizer、孤立训练分片 | **DELETE_ON_NATIVE_PROOF** | 完整结果/来源已记录，不能恢复、无消费者；依原private manifest精确列文件删除。 |
| 已确证旧工程副本在后继独立恢复点验收后被完全取代，且所有活动/已封存消费者关闭 | **CONDITIONAL_DELETE** | 仅移除真正冗余的副本；唯一实际可恢复源不删。 |
| E1/MP1及历史大checkpoint | **逐项复核，不按“大/旧/失败”直接删除** | #47/6336784045记录有有效完整状态或现存引用；仅在精确引用退役、源/基线/选模价值归零的独立证明之后，才进入以上删除集合。不要求永远留全部历史 Policy，但不能牺牲已证明的关键实验依赖。 |
| 已失败的旧A job1474116、无full1024 checkpoint | **仅对应站点中发现的无用残片可以删** | 它实际发生在Vanda A40，不在Hopper项目文件空间；不杜撰25GB可删状态；仍保留1024真实updates、失败与资源费用原证据。 |
| 1474352 Exit1后先成功保存的 checkpoint-32 | **KEEP** | A_CONTROL_SAVE_RECOVERY1 step32全状态、15成员、约25.54GB，经独立checked_resume；历史失败是后续可选共享，不是这份状态无效。 |
| 当前A1 job1474599 的有效1024 full-state及12 TRAIN probes、后续2048、CORR@2048共同父与模型依赖 | **KEEP** | #47/6337023734显示17:03:48仍R，已真实完成1024保存和12 probes，并同作业恢复至至少step1030；不可使用旧Q快照删除正在使用的输入和checkpoint。 |
| B_COVERAGE现役或预登记输入、科学1024/4096、DEV/LOCKBOX、已选中或尚待评价的模型 | **KEEP** | 仍承担可复现训练/科学对照用途。 |
| Xet缓存已完成的本次6项 | **DONE，不重跑** | 已有准确inode与unlink收据；quota当时0变化不能因此重复删除。 |
| vla_wbc/PI05、其他人/其他项目、PI0292/data/环境/WBC/vendor/源代码/原始数据 | **EXCLUDED** | 非本轮授权，全部保留且不探测他人项目。 |

Vanda与Hopper是不同资源域。本轮仅授权本人登记项目的有证据维护动作；不能因为找到别的host上失败job，就在Hopper开展跨项目或跨文件系统删除。其他站点后续需依原scope、现场权限与自身维护协议执行。

## 3. 避免“无限盘点”，但不能删除未知引用

从登记的 run-index / model-index / lease / failure status 出发，**只审查明确候选集合**，不重复对整个 runs 树作数小时 du 或全量hash。一个候选符合以下条件即可执行，不需要再请用户确认：

1. 其本人项目身份、注册输出根、宿主/mount/UID/实际路径可证明；run已终态关账或对应载荷已与运行独立关闭；该文件非活跃输出。
2. 真实 manifest、模型/恢复/评估/发布引用、队列任务与原维护owner均确认无当前或未来已封存消费者。lsof空不等于跨节点无人使用。
3. 旧checkpoint确证为不可恢复、科学结论已记录且不再引用；**或**此前有效但已被独立验收的后继完全取代且无其他用途。正常 Exit1/0、loss好坏、目录大小/mtime不单独作为理由。
4. 先留下固定小型证据：训练/config/seed、父权重/数据/stats的SHA、run/optimizer身份、loss/失败metrics/任务成功率、原生FINAL、费用、可选历史CHECKPOINT_COMPLETE、成员hash/路径索引，以及新的明确 PAYLOAD_RETIRED 标记，避免以后loader错误把旧READY当成仍可恢复。
5. 按最小 allowlist、真实device/inode/link count/size/mtime/allocated blocks，每个具体载荷路径 unlink 一次，不用rm -rf运行根、glob、follow symlink、chmod共享文件或删除硬链接共享的受保护数据。未知或无法核清的一项标 HOLD，其余通过的可继续。

完成后对同一配额域记录before/after真实alloc/quota和已删inode/字节，区分“已unlink allocated”和“配额报告已下降”。共享nlink、打开文件、quota刷新滞后不得靠杀服务/删另一个有效hardlink硬凑释放。

## 4. 工程回执与权限

使用已有唯一维护owner的本机受检入口、登记的锁和已有维护CPU/IO余额。不得干扰1474599的watchdog/lease/release，或新建第二owner、提交GPU/PBS、重新加载模型、启动物理、新数据采集。真实权限/身份/引用缺失时只对该目标报告HOLD。

完成时出 HOPPER_GR00T_FAILED_POLICY_CLEANUP_ACTUAL（每个候选 DELETE/KEEP/HOLD、精确私有路径/inode、保留的小记录、真实unlink结果、allocated/本项目quota前后、仍保护的科学state），只将脱敏摘要登记#47。如确实没有无引用GB级checkpoint，报告NO_ELIGIBLE_LARGE_PAYLOAD_CURRENTLY，而不是重复删除健康资产。

**本GitHub修订本身没有SSH或远端unlink效果。原始失败的科学/预算历史仍不可篡改。**