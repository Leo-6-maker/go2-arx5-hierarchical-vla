# 当前入口 · 2026-10-29 · 长程Goal V6

`main`是唯一集成入口。先读[科学状态](CURRENT_SCIENCE_STATUS.md)、[V6详细计划](docs/protocols/GR00T_WBC_LONG_HORIZON_GOAL_V6_20261029.md)、[路线索引](docs/EXPERIMENT_ROADMAP.md)和[AGENTS](AGENTS.md)。历史协议不自动提供当前权限。

## 下一步：两条科学线，一条只读支持线

| 位置 | 当前任务 | 完成标志 |
|---|---|---|
| CoreLab 4080S | 复用已补齐的LOCAL终态，冷态绑定D25原N294101/24，再GEN1 | 完整行为表、D−C配对及关闭 |
| Vanda | 复用已审计E0/E1和已采用剩余范围，补E2资格、fresh训练两臂512 | 固定端点、真实更新/消费/保存恢复 |
| PC Rerun | 已有Teacher/Student失败解释与保存输出分析 | 可核验只读证据，不阻塞科学运行 |

两站运行不互相SSH/查队列/等心跳；管理端独立连接/终态核账，模型走已授权HF固定revision，不重传已交付D25。

## 当前关键事实

- A4096=11/24、B4096=6/24、C50=6/24；源域已暴露DEV，不是独立泛化。D25/GEN1各1024完成，仍无有效Eval。
- Vanda1504095已F/Exit1、once结算、own lease关闭：E0 39查询，E1两步资格/17查询/14目标参数变化，E2 15零步输出/优化0。正式E1/E2仍0/512。
- **PR99新增实际证据：E0/E1完整保存前缀独立审计PASS，E1重载同输入/noise输出精确一致；PBS41→42及一次控制范围已采用；Core LOCAL38条终态已补齐。** [原报告](reports/PI0294/design/parallel-science-20261022/ARX5_REAL_E0_E1_FULL_SAVED_PREFIX_AUDIT_AND_CANONICAL_CORE_HISTORY_CATCHUP_ACTUAL_20261029.json)。不重做同一审计/cap/mirror/settle。
- **10月29日08:10 SGT实际：Vanda唯一作业1505362已R，E2资格已完成原GR00T加载及2次优化调用。** [真实回执](reports/PI0294/design/parallel-science-20261022/ARX5_UNIQUE_REMAINING_GPU_R_AND_REAL_E2_TWO_OPTIMIZER_CALLS_ACTUAL_20261029.json)。保护健康owner；08:19 SGT后继E2资格已有原生COMPLETE，独立审计待回读；E1正式阶段已ACTIVE，当时仍0/512，随后E2同512。D25新Core原生部署仍待接续。
- Rerun验证三条稀疏记录（Teacher各41/641、失败2/224），不是完整整机回放；full qpos/mesh不是训练Eval前置。
- Go2+ARX5标定/12次任务pilot独立于当前12条源域示范；原DEC1仍固定CORR/R0，不混目标训练。

## 接续与证据

[Codex交接](docs/handover/2026-10-29/POST_RERUN_SCIENCE_FIRST_EXECUTION.md) · [计划卡](reports/PI0294/design/parallel-science-20261022/POST_RERUN_LONG_HORIZON_TASK_CARD_V6_20261029.json) · [CODE_MAP](docs/CODE_MAP.md) · [仓库工作流](docs/REPOSITORY_WORKFLOW.md)。计划卡不是可派发RUN_MANIFEST。

刷新[Issue #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11)、[Issue #47](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47)和原owner的本机原件。GitHub不是实时GPU监控，后继回执优先；已完成只接剩余项。普通同范围工程最小冷修，真实外部缺口一次准确报告。

stop-new仍2026-10-31 20:52:29.813 SGT，hard deadline仍2026-11-01 20:52:29.813 SGT。本次只发布计划，不延期、不采用新额度、不启动作业/子代理/暂停自动化；原WBC/R0/评分/失败保持，运行source不热切换。

根入口保持简短，完整证据在reports。历史可由Git/原字节归档恢复；分支退役按独立维护回执，不随计划发布删除refs。
