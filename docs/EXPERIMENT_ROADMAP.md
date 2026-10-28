# 最新实验路线 · V6（2026-10-29）

**预训练能力复用 → 最小ARX5适配 → 闭环与独立泛化 → 冻结WBC复用 / 少样本整机适配。** Rerun已作为只读工具，不再把开发Viewer、扩场景数或训练量当默认路线。

[完整长程Goal V6](protocols/GR00T_WBC_LONG_HORIZON_GOAL_V6_20261029.md) · [Codex交接](handover/2026-10-29/POST_RERUN_SCIENCE_FIRST_EXECUTION.md) · [计划卡](../reports/PI0294/design/parallel-science-20261022/POST_RERUN_LONG_HORIZON_TASK_CARD_V6_20261029.json)。发布不是新GPU/PBS/物理授权。

## 当前两条科学线

| 平台 | 下一动作 | 真实交付 |
|---|---|---|
| CoreLab 4080S | 复用已实际补齐LOCAL历史，冷态绑定D25原24，再GEN1 | D−C配对表与GEN1行为结果 |
| Vanda | 复用已审计E0/E1及已采用剩余范围，补E2资格，两臂fresh各512 | 固定端点、消费/冻结/完整保存恢复 |
| PC Rerun支持 | 已有失败关键时间线、学生成功、保存E0输出 | 可核验解释，不阻塞上述两线 |

最新PR99已证明保存前缀完整审计、PBS42控制采用和Core历史补齐。**不要重复这些步骤；新request/qsub/训练及D25部署仍未发生。** 依据[实际报告](../reports/PI0294/design/parallel-science-20261022/ARX5_REAL_E0_E1_FULL_SAVED_PREFIX_AUDIT_AND_CANONICAL_CORE_HISTORY_CATCHUP_ACTUAL_20261029.json)，执行前刷新后继原件。

两站运行不互相SSH/查队列/等心跳。模型沿已授权HF固定revision，管理owner独立连接并终态核账。原Journal/onceIO和安全保持，不新建调度平台。

## 科学里程碑

| 阶段 | 问题 | 出口 |
|---|---|---|
| M0 剩余接续 | 如何复用已完成证据而不重算？ | 合法剩余运行真实启动 |
| M1 D25/GEN1 | 混合比例效果与现有布局端点能力？ | 完整原面板和配对/描述性结论 |
| M2 E1/E2@512 | 冻结共享策略、同12条示范能否适配？ | 两端点与各24条件 |
| M3 分流 | 确认、共同续1024还是一项机制实验？ | 结果决定，不自动扩计算 |
| M4 泛化/数据效率 | 新几何、视觉/语言与共同home能否迁移？ | 预先隔离的未暴露测试 |
| M5A DEC1 | 固定CORR/R0能否复用冻结WBC？ | 原64格实际结果或准确NOT_RUN |
| M5B 整机适配 | 10段标定＋12次整机尝试是否有增益？ | 等训练量控制与独立整机评价 |
| M6 新技能/研究包 | 单一柜门迁移及必要补数据？ | 先测后采、可复算正负结论 |

E1/E2保持[执行V2](protocols/GR00T_ARX5_MINIMAL_ADAPTATION_EXECUTION_V2_20261027.md)：同原始父、6几何12示范、逐cursor/噪声、batch32，各512处于1024 LR日程，原R0/EEF10/H40/Prefix8/四步与评分。资格权重不作正式初始化。E1仅目标类别14 W/b，E2再更新指定共享动作组件；VLM/其他类别等冻结。

E0已有39查询不是闭环；少参数不保证更快或泛化，退化不自动证明遗忘。旧DEV不变新TEST，未测不算0%，不显著不是非劣。原DEC1无目标任务训练与少样本目标适配分开。柜门先测试，书架/BASE/RL/世界模型不同时展开。

原stop-new为2026-10-31 20:52:29.813 SGT，hard deadline为2026-11-01 20:52:29.813 SGT，不延长。PBS已采用42不等于新任务已派发，剩余范围和矩阵/最终保护保持。当前状态与实际证据见[科学状态](../CURRENT_SCIENCE_STATUS.md)及#11。
