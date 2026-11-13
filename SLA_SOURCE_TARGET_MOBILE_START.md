# SLA V4 入口：开源／自采混合后训练与移动底座适配

2026-11-13 · **计划与操作说明已发布；运行能力以独立实际回执为准**

用户要求进一步明确后训练流程、细节和数据集使用。本PR最初四个文件保留，新增一份V4.1操作说明并更新本入口；不改现役代码、WBC、其他开放计划入口或正在运行的作业。

## 先读操作说明

1. [V4.1 后训练具体流程与数据集使用](docs/protocols/SLA_POSTTRAIN_WORKFLOW_AND_DATA_V4_1_20261113.md)：固定模型、数据池、五个源task ID、样本构造、模块与超参数、每阶段初始化、batch配比、评价和实现缺口。
2. [完整V4实验计划](docs/protocols/SLA_SOURCE_TARGET_MOBILE_V4_20261113.md)：研究对照、归因、完整预算和移动frame/mode。
3. [机读研究矩阵](docs/protocols/SLA_SOURCE_TARGET_MOBILE_V4_20261113.json)：PLANNED、execution_ready=false；不能直接当现场job配置。
4. [Codex启动prompt与交接](docs/handover/2026-11-13/SLA_SOURCE_TARGET_MOBILE_V4_START.md)：与V4.1一起读取；现场状态采用最新实际端点。

## 明确主流程

SLA_base → 新目标D100的动作/空间两臂 → D300固定配方 → 合格RoboTwin源池与D300的T-only/M75对照 → 同父模型专家纠正 → 独立TEST → 新站位/导航后操作。

数据嵌套扩大不等于权重一路串训。D100、D300和主要混合消融从同一干净SLA_base起步；相同已完成端点可复用。只有明确的续训/纠正/顺序实验继续父checkpoint。

- 旧TRAIN100只用于既有S1/S2/R1，不能改名为D100_new。
- 新目标池为10姿态层×5物体×PICK_HOLD/PICK_PLACE，D100/D300/D1000累计100/300/1000完整episode。
- 首轮源池明确筛选`TianxingChen/RoboTwin2.0`中的`move_pillbottle_pad`、`move_stapler_pad`、`place_mouse_pad`、`place_empty_cup`、`place_object_stand`；每任务200条TRAIN为目标。其原始数据尚未审核准入，不声称1,000条已可用。
- 仅命令语义兼容的S_cmd进入动作混合；future-achieved/未知标签留在S_aux。另一臂需要协助的任务不直接裁剪动作。公开下载与开源环境再生成分别绑定origin/version。
- 首要对照实际使用D300：每batch32分别32T与24T+8S，各3K；`T100`是100%目标比例，不是100条数据。
- DROID/RoboCasa/LIBERO、未准入ARX5与移动源数据不进入首个混合池；不重做SLA大规模预训练，不使用在线RL。
- R1和目标D100/D300不等待source准备完成。源转换、码本、统计和数据读取缺口在V4.1列为明确实现项。

移动先验证低层能力/坐标→新站位静止操作→导航稳定接管→负载运输/连续移动。固定臂来源缺base标签就mask，不补假零；高层不直接输出腿关节，不在推理时使用未来底座真值。

## 最新核验快照与历史

本轮读到[PR #382](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/382) HEAD `3195f0eb664ed98db7f76ad0643ed1d742472a7b`，实际摘要为2026-11-13 20:59 SGT：R1已完成500更新/16,000访问及H200/Core RGB恢复；R1原生任务评价0；D100_new/D300_new/PICK_PLACE生产准入0。这是GitHub快照，不是本次登录现场。

V4最初使用的[12:56汇总/12:54端点](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b3e58cdbbefbc68bf733792ff9b7009a23d10eb3/reports/WM02/sla-corelab-v3-20261113/CURRENT_STAGE_ACTUAL.json)保留在原计划中，不能当作当前状态。开工检查后继端点，不重投未知或已完成作业。

继承[并行采集与GPU驻留V3](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f672fb1aa574a2dbac3c91e047346eaa441bc9f5/docs/protocols/SLA_CORELAB_PARALLEL_DATA_H200_V3_20261113.md)和现役监督优先方向。Corelab独立多环境沿用既有授权，H200采用真实allocation。此次文档更新未启动训练、采集、自动化或实机。

main是发布整合入口，不代表全部现役runtime已合并。遵循[仓库工作流](docs/REPOSITORY_WORKFLOW.md)，在独立worktree实施并正常PR整合，不热改健康capsule。
