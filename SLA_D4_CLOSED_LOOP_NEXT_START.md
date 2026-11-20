# SLA 当前入口：多数据集后训练与本体闭环 Goal V2

2026-11-20，Asia/Singapore。用户要求自主长程推进、后台排队接续、Eval独立并行，目标取得多源后训练及真实学生抓持证据。

## 固定交付

- [完整Goal与执行路线](docs/protocols/SLA_MULTISOURCE_CLOSED_LOOP_GOAL_V2_20261120.md)
- [机读任务、触发与预算](docs/protocols/SLA_MULTISOURCE_CLOSED_LOOP_GOAL_V2_20261120.json)
- [Codex启动prompt](docs/handover/2026-11-20/START_SLA_MULTISOURCE_CLOSED_LOOP_GOAL_V2.md)
- [最新进度审核与问题分析](reports/WM02/sla-multisource-goal-v2-20261120/AUDIT_AND_PLAN.md)

现役MIX20k/TGT5k/ADAPT5k继续；MIX869恢复和D2读取已有完成证据，不重复做。补MIX1001后继、ADAPT到5k实际队列，2k一次D2切换不等Eval分数。Core冷态先为就绪A0部署及原DEV前6提供窗口，随后固定评价与训练/采集轮转。原SCALE96及ADAPT96保留，离线0/24不是禁止合法学生native的新门。

固定阶段面板ADAPT0→500 H40从84/88降至57/88，及时闭合10/24降至2/24，联合仍0/24。先读已有预测做零生成归因；2k联合仍0且标签/时钟/专家目标可行时，条件触发CLOCK-A/B各500的新时序监督实验。PR419和后续批处理/两卡工程独立，现役5k不借D2切换热换worker。

## 证据与范围

公开PR418 `0ef0abfc`观察12:49:41；用户14:08新报MIX1000封存、TGT4976、ADAPT953及部署资源拒绝，执行时读真实现场。计划/报告head不是全部runtime，源码映射见Goal。新Goal单列≤90工程更新、条件≤1000科学更新及≤72新科学原生，旧预算不重置；未触发不耗尽。

[原长程配方](docs/protocols/SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1_20261119.md)与[后台协议](docs/protocols/SLA_SCHEDULER_FIRST_ASYNC_EVAL_V1_20261119.md)在未被V2明确更新处仍有效；[11:40旧调整](docs/protocols/SLA_LONGRUN_ADJUSTMENT_20261120.md)保留为历史。WBC、安全、评分、旧失败/TEST和真实站点权限保持。

本次为计划发布，新增服务器提交、训练、模型生成、native/tick及硬件动作均0；未改自动化。执行Goal须交付真实JobID/unit、可恢复端点、固定分母和明确终态，不能把READY或CPU检查称为训练/闭环完成。
