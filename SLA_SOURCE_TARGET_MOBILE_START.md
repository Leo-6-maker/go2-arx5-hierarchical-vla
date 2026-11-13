# SLA V4 入口：开源／自采混合后训练与移动底座适配

2026-11-13 · **计划已发布，不是训练或任务结果**

用户要求承接开源数据与Go2＋ARX5自采数据的混合训练、数据规模及移动底座讨论，制定详细推进计划并发布GitHub。本轮新增四个独立文档/规格文件，不改现役代码或其他开放计划的入口。

## 阅读入口

- [完整实验计划](docs/protocols/SLA_SOURCE_TARGET_MOBILE_V4_20261113.md)：数据语义与准入、数量、混合对照、预算、移动frame/mode与验收。
- [机读研究矩阵](docs/protocols/SLA_SOURCE_TARGET_MOBILE_V4_20261113.json)：PLANNED、execution_ready=false；不能直接当现场job配置。
- [Codex启动prompt与交接](docs/handover/2026-11-13/SLA_SOURCE_TARGET_MOBILE_V4_START.md)：先兑现目标policy，源准备并行，移动按能力分阶段。

## 本轮关键决定

先完成现役R1和D100/D300，不等源数据。源分为真实目标命令T、可恢复源命令S_cmd、不同语义S_aux；主混合先固定D300＋S1000，比较目标100%与目标75%/源25%。同总更新与同目标曝光分别比较，必要时添加顺序训练、源50%和纯源探索；不默认所有开源数据可直接拼接。

移动顺序是：部署控制器能力/坐标审计→新站位静止操作→导航后操作→负载运输与连续移动。固定臂来源没有base标签时不补零监督；高层不直接输出腿关节。保持既有冻结WBC和有效安全范围，不把旧站立小幅协调成绩当作导航能力。

## 固定执行依据

[PR #382](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/382) 的核验HEAD为 `b3e58cdbbefbc68bf733792ff9b7009a23d10eb3`；[回执](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b3e58cdbbefbc68bf733792ff9b7009a23d10eb3/reports/WM02/sla-corelab-v3-20261113/CURRENT_STAGE_ACTUAL.json)生成于12:56 SGT，主要端点12:54。R1当时已入H200队列、0已提交更新；D100_new/D300_new正式示范0。开工核对最新实际状态，不重投未知或已完成作业。

继承[并行采集与GPU驻留V3](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f672fb1aa574a2dbac3c91e047346eaa441bc9f5/docs/protocols/SLA_CORELAB_PARALLEL_DATA_H200_V3_20261113.md)和[现役监督优先交接](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b3e58cdbbefbc68bf733792ff9b7009a23d10eb3/docs/handover/2026-11-13/SLA_SUPERVISED_FIRST_EXECUTION.md)。Corelab独立多环境沿用用户既有授权，H200采用真实allocation；本次发布未启动训练、采集、自动化或实机。

main是发布整合入口，不代表全部现役runtime已合并。遵循[仓库工作流](docs/REPOSITORY_WORKFLOW.md)，实现从实际核验的执行SHA在独立worktree推进，正常PR整合，不热改健康capsule。
