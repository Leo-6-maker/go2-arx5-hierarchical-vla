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
