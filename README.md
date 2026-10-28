# GR00T + Go2–ARX5 · Frozen-WBC Adaptation

**主问题：预训练GR00T经多少ARX5本体适配，能获得可靠闭环操作，并复用冻结的Go2＋ARX5全身控制器？**

2026-10-28仓库整理后，`main`是源码、当前研究路线与交接的唯一集成入口。合并历史不代表模型达标、服务器已部署或新实验获准。实际运行始终固定完整commit、模型、数据、Processor、统计量和执行权限。

## 从这里开始

| 需要 | 入口 |
|---|---|
| 开始工作与下一动作 | [START_HERE.md](START_HERE.md) / [AGENTS.md](AGENTS.md) |
| 最新已核验结果 | [CURRENT_SCIENCE_STATUS.md](CURRENT_SCIENCE_STATUS.md) |
| 最新实验路线与推进关口 | [EXPERIMENT_ROADMAP](docs/EXPERIMENT_ROADMAP.md) |
| 代码位置与复用边界 | [CODE_MAP](docs/CODE_MAP.md) |
| 分支、合并与服务器同步 | [REPOSITORY_WORKFLOW](docs/REPOSITORY_WORKFLOW.md) |
| 科学／维护记录 | [Issue #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11) / [Issue #47](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47) |
| 本次整理及分支退役状态 | [CONSOLIDATION_REPORT](docs/maintenance/2026-10-28/CONSOLIDATION_REPORT.md) |

## 当前路线

**收齐C50/D25与GEN1行为结果 → E0预训练诊断、E1/E2匹配最小适配 → 独立布局／任务迁移验证 → 必要时少量Go2＋ARX5目标示范 → 冻结WBC执行域比较。**

先测量再补数据。E1只更新目标本体编码／解码参数；E2额外更新共享动作策略。两臂从同一原始权重和12条源域示范出发，先各512步。不默认扩采、延长训练或增加柜门／书架任务。

截至整理基线：A4096首面板11/24、B4096为6/24、C50为6/24；D25与GEN1已训练但尚无有效Eval，E0/E1/E2尚无成功的原生模型结果。详见带证据的状态页，未运行不等于成功率0%。

## 唯一平台分工

- **Vanda**：合法PBS内微调、有限离线神经诊断和完整状态保存。
- **CoreLab 4080S**：模型推理、MuJoCo双相机与完整闭环Eval。
- **AMD／管理端**：CPU、数据准备和有限部署／证据汇总。

运行时两站不互相SSH、不查对方队列、不等待对方心跳。仅通过已授权的HF固定revision交付模型；私有数据、凭据和WBC资产不得因此公开。有限本地执行记录与独立终态结算沿既有机制，不另外建立调度平台。

## 工作区与历史

人类入口仍为 `~/Projects/GR00T-WBC`。复用[统一入口规范](docs/operations/2026-10-13_UNIFIED_ENTRY_AND_GITHUB_SYNC_V1.md)与[WORKSPACE_ENTRY_ENFORCEMENT_V1.md](docs/operations/WORKSPACE_ENTRY_ENFORCEMENT_V1.md)；其中旧分支路由由[本次main提升记录](infrastructure/workspace/INTEGRATION_REF_CHANGE_20261028.json)取代，旧运行与固定source不变。

原 `pi0267/executor-decoupled-recovery` 已按已审阅SHA合入main，暂留为旧owner迁移兼容引用，不再是新工作的默认发布分支。不要在运行中的source capsule执行pull、checkout或清理。旧协议、负结果及报告完整保留；原长日志已[按原字节归档](docs/archive/2026-10-28/pre-consolidation/)。

本次未启动训练、物理或远端部署；GitHub合并与源码回读不是CPU全套测试、CI或模型资格PASS。
