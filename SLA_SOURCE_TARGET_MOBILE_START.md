# SLA 当前入口：V6.1 动作合同与部署回放

2026-11-14 · **只发布实施计划；未修改运行源码、现役数据/统计或作业。**

优先阅读 [V6.1详细推进计划](docs/protocols/SLA_ACTION_CONTRACT_REPLAY_V6_1_20261114.md)、[机读矩阵](docs/protocols/SLA_ACTION_CONTRACT_REPLAY_V6_1_20261114.json) 与 [执行启动prompt](docs/handover/2026-11-14/SLA_ACTION_CONTRACT_V6_1_START.md)。V5长程路线继续保留，本补充不重开整体路线。

本轮锁定：EEF_STREAM_V2属于专家规划，最终重构H40才是计划目标；专家/学生之后共用执行adapter。supervision_mask与consumed_mask分开，未执行后缀不作为消费或成功证据。先冻结TRAIN统计，再完整FP32/部署回放；在途V6不改动作定义、不加参考误差/历史、不转计划蒸馏。

近期顺序是**标签经学生入口的RP8原生回放 → S16阶段/接缝窗口的固定500步拟合 → V6原定终点与闭环 → 条件性匹配空间监督/独立计划蒸馏 → LeRobot双reader张量一致**。已启动V6健康训练独立继续，不等待新增检查、不抢占或重复提交。

最新核验执行[PR #384](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/384)固定SHA `4273dcdf143a0f9bac28afc5f831eb597a240412`，[报告](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4273dcdf143a0f9bac28afc5f831eb597a240412/reports/WM02/sla-long-horizon-v5-20261114/EXECUTION_20261114.md)对应V6训练快照2026-11-14 12:25:57 SGT：1220/1220参考与FP32标签包络通过，候选92更新；原N20 TRAIN8已0/8终态。不是当前服务器实时状态，接手先查后继终态。

仅旧AGENTS中与用户后续明确授权冲突的范围以新授权为准：Corelab独立多世界继续，当前已验证4-world；WBC/腿策略/2ms动力学与原安全、所有权、工作流保持不变。计划文档不自动启动仿真、H200、真机或RL。

---

## 下方为V5历史入口原文（2026-11-13快照，不作为最新运行状态）

# SLA 当前入口：长程推进 V5

2026-11-13 · **计划已发布，不是训练或原生能力结果**

优先阅读：
1. [长程推进 V5](docs/protocols/SLA_LONG_HORIZON_EVIDENCE_V5_20261113.md)：可执行策略→空间与条件化→恢复与源混合→共享技能→移动/跨执行器；世界模型/传感/RL有明确进入条件。
2. [V5 机读矩阵](docs/protocols/SLA_LONG_HORIZON_EVIDENCE_V5_20261113.json)：状态PLANNED、execution_ready=false，不能直接作为现场job配置。
3. [V5 启动 prompt](docs/handover/2026-11-13/SLA_LONG_HORIZON_V5_START.md)：先接续实际作业，复用资格，下一次交付真实数据、checkpoint与原生结果。
4. [V4.1 具体后训练流程](docs/protocols/SLA_POSTTRAIN_WORKFLOW_AND_DATA_V4_1_20261113.md)、[V4 源目标/移动计划](docs/protocols/SLA_SOURCE_TARGET_MOBILE_V4_20261113.md)与[原矩阵](docs/protocols/SLA_SOURCE_TARGET_MOBILE_V4_20261113.json)：保留历史设计；未启动工作的顺序与实际源数量以V5修正为准。

## 新的事实基线与直接下一步

执行已到[PR #384](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/384)，固定阅读SHA `cb09e71441b2f02307317b60d059ea06c38cb55b`。[回执](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/cb09e71441b2f02307317b60d059ea06c38cb55b/reports/WM02/sla-source-target-v4-20261113/CURRENT_STAGE_ACTUAL.json)时间2026-11-13 23:35:31 SGT；这是证据快照，不是现在服务器状态。

- R1_500训练/RGB恢复已完成，原生R1请求仍0。
- 具名PICK_HOLD小集2条/122窗口已经用于H200 job716146；快照513/3000更新、已有完整step500。先核实后继终态，勿重投。
- 固定专家16/16完成，正式D100/D300/PICK_PLACE仍0；122窗口空间cache已读回，空间梯度未完成。
- 五个RoboTwin默认包实际每任务50条、共250，20条已审；S_cmd仍0，15Hz字段不证明真实时钟。补缺口或仪器化源仿真，不能再空等“已经下载的1000条”。

先兑现当前候选原生执行和空间链，继续目标示范生产；小集可学习但不冒称D100泛化。开源准备不阻塞目标policy，新技能不等待抓放100%。源池固定实际数量比较，不为凑S1000放宽标签或复制episode。

## 执行与权限

Corelab独立并行世界沿用户已明确V3授权，每世界单owner；H200用真实有效allocation，避免CPU/GPU大张量反复搬运。旧AGENTS里的全Core串行/禁止H200首轮范围不覆盖后续明确授权；其他安全、冻结WBC/2ms、工作树和现场权限仍有效。健康capsule不热改，未知作业不盲重投。

当前不重做CPT、不自动启动在线RL、真机或WBC重训。移动先能力/坐标与新站位，再真实导航；固定臂缺base标签不补零。暂停仿真不是实时部署结果。

[PR #383](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/383)为计划分支，#384为执行分支；main不自动具备全部最新runtime。[仓库工作流](docs/REPOSITORY_WORKFLOW.md)仍适用。V5仅正常追加文档，不合并、不改在途实验；V4/V4.1旧内容和所有负结果保留。

继承：[设备驻留与Corelab并行V3](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f672fb1aa574a2dbac3c91e047346eaa441bc9f5/docs/protocols/SLA_CORELAB_PARALLEL_DATA_H200_V3_20261113.md) · [旧V4执行交接](docs/handover/2026-11-13/SLA_SOURCE_TARGET_MOBILE_V4_START.md)。
