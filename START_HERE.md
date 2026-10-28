# 当前入口 · 2026-10-29 · 长程Goal V6

`main`是唯一集成入口。先读[当前科学状态](CURRENT_SCIENCE_STATUS.md)、[V6长程计划](docs/protocols/GR00T_WBC_LONG_HORIZON_GOAL_V6_20261029.md)、[路线索引](docs/EXPERIMENT_ROADMAP.md)与[AGENTS](AGENTS.md)。历史协议不自动提供当前计算权限。

## 下一步：两条科学线，一条只读支持线

| 位置 | 任务 | 真实交付 |
|---|---|---|
| CoreLab 4080S | 真实核齐LOCAL终态/运行身份，完成D25原N294101/24，再GEN1 | 完整行为表、D−C配对分析及关闭 |
| Vanda | 审计已保存E0/E1，不重跑前缀；补E2资格并fresh训练两臂各512 | 完整原生证据、固定端点与消费/恢复审计 |
| PC Rerun | 已有Teacher/Student关键时间线与离线预测只读分析 | 可核验的失败解释，不阻塞科学执行 |

两站运行不互相SSH、查队列或等心跳。复用原有限范围和Journal/onceIO；管理端独立连接/终态核账，模型走已授权HF固定revision，不重传已交付D25。

## 当前关键事实与准备边界

- A4096=11/24、B4096=6/24、C50=6/24；均为源域已暴露开发面板。D25/GEN1各1024已完成，尚无有效Eval。
- Vanda1504095已F/Exit1、一次结算和两own lease关闭：E0 39查询，E1资格2优化/17查询/14目标参数变化，E2 15零步查询/优化0。正式E1/E2均0/512，保存大成员独立审计尚未闭合。qstat255是工程读故障，不是任务失败。
- main截至证据基线45aeba9c已含PR97 saved-prefix审计与一次剩余范围、PR98 Core终态catch-up代码。**代码/CPU准备不等于审计完成、PBS42或LOCAL已实际采用。** 先刷新实际回执，已完成的只接剩余项。
- Rerun官方Viewer已验证三条真实稀疏记录（Teacher各41/641、C50失败2/224）；不是完整3D整机回放，不再扩建第二套Viewer。
- Go2+ARX5十段标定/十二次目标尝试独立于当前12条源域示范；原DEC1仍固定CORR/R0，不混目标训练。

## 执行入口与证据

1. [Codex交接](docs/handover/2026-10-29/POST_RERUN_SCIENCE_FIRST_EXECUTION.md)和[V6计划卡](reports/PI0294/design/parallel-science-20261022/POST_RERUN_LONG_HORIZON_TASK_CARD_V6_20261029.json)。计划卡不是可直接派发的RUN_MANIFEST。
2. [CODE_MAP](docs/CODE_MAP.md)、[仓库工作流](docs/REPOSITORY_WORKFLOW.md)，复用现有模块，不另造调度器。
3. [Issue #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11)科学与[Issue #47](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47)维护的最新实际回执。
4. 原owner在本机核当前进程、lease、完整输入、资源和截止；GitHub页面不是实时监控。普通同范围工程自主冷修，真实外部缺口一次准确报告。

旧stop-new为2026-10-31 20:52:29.813 SGT，hard deadline为2026-11-01 20:52:29.813 SGT。本次文档不延期、不采用预算、不启动作业/子代理/暂停自动化。健康或闭合job不重启，运行source不热切换，原R0/WBC/评分/失败保持。

当前入口只写必要摘要，详细证据写原reports并链接。此前入口和状态均可由Git历史/原字节归档恢复；分支退役仍按独立维护清单和实际回执，不随计划发布删除refs。
