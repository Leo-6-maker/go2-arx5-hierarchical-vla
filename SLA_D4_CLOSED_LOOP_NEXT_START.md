# SLA 当前入口：长训练接续、D2课程和学生闭环

2026-11-20，Asia/Singapore。最新用户要求：审核11:40实验进展并调整后续计划；沿用服务器后台排队接续、无需实时Codex监控、Eval可独立并行的决定。

## 当前任务

- [最新审核与详细调整](docs/protocols/SLA_LONGRUN_ADJUSTMENT_20261120.md)
- [机读任务、证据层级与预算](docs/protocols/SLA_LONGRUN_ADJUSTMENT_20261120.json)
- [Codex接续prompt](docs/handover/2026-11-20/START_SLA_LONGRUN_RECONCILIATION_D2_NATIVE.md)
- [审核与发布验证](reports/WM02/sla-longrun-review-20261120/REVIEW_AND_VALIDATION.md)

保持TGT5K与ADAPT健康训练；核对MIX869完整前缀及oomd终止后，从真实可恢复步继续。现在准备D2逐episode读取与ADAPT2001–5000后台发放；2k一次课程决策不等待Eval分数。补齐跨GPU部署数值证据后优先ADAPT0既定DEV24，不继续只报离线门通过率。

用户11:40报告HOLD专家2/2、46156tick，D2新增139窗；TGT2000/3500封存、运行快照4324；ADAPT750封存；MIX869外层终止。此较新私有回执未在本次ChatGPT会话直接读取。独立审核公开PR418 `3670aaca`最新现场报告约02:36，另读取实际训练/调度固定代码。公开证据与用户新报告在完整计划内分层标注，不把历史快照当live。

## 保持的合同

- [完整长程配方、课程和预算](docs/protocols/SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1_20261119.md)
- [后台接续与并行Eval协议](docs/protocols/SLA_SCHEDULER_FIRST_ASYNC_EVAL_V1_20261119.md)
- [现役SCALE合同92622691](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/926226914a0c08ee1ebb143e0edea1434728107b/docs/protocols/SLA_SCALE20K_DIRECTION_20261119.md)
- [旧D4/D5/J合同cf9d1acb](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/cf9d1acbf72e96b1bd158322ff8cbaf8fe28fabb/docs/protocols/SLA_D4_CLOSED_LOOP_AND_GRASP_20261119.md)

原SCALE/ADAPT总预算、现役模型与Adam/RNG/LR保持；Teacher全格、J全48、N2资格和高成功率不是健康训练接续前置。D2只在预登记2k冷态改变成员/bank，部署兼容V2只用于新部署判定，旧严格失败保留。报告head不等于完整runtime，按manifest固定各worker源码。WBC、H40/P10/50ms/2ms、安全、评分、历史失败及旧TEST保持。

本次更新是审核与计划发布，服务器作业提交、训练更新、生成、原生积分和硬件动作均0。实际后继以真实systemd/PBS及checkpoint/native回执为准。
