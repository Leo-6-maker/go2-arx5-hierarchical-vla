# GR00T + Go2–ARX5 · Task-space policy learning

研究目标：学习真正完成物体操作的任务空间策略，以统一 EEF／夹爪接口连接冻结执行器，并分别验证源域任务能力、视觉语言策略迁移与 Go2＋ARX5 整机效果。

**下一研究线 WM01：小型结构化世界模型 → 任务策略学习 → 真实源域闭环 → 有条件的视觉／VLA 转移。** 先验证 WM 的决策价值，不预设 JEPA、GR00T 或 WM 一定优于基线。已有 V7 作业按原配置继续；本次整理不启动、停止或改写任何服务器运行。

## 入口

| 需要 | 唯一入口 |
|---|---|
| Codex 开始新工作 | [START_HERE](START_HERE.md) 与 [AGENTS](AGENTS.md) |
| WM01 执行交接 | [WM01_CODEX_EXECUTE](docs/handover/2026-11-03/WM01_CODEX_EXECUTE.md) |
| 详细科学计划 | [WM01 protocol](docs/protocols/WM01_TASKSPACE_WORLD_MODEL_POLICY_V1.md) |
| 参数与工作包 | [实验目录](experiments/wm01/README.md) / [plan.json](experiments/wm01/plan.json) |
| 实际状态，不含计划冒充的结果 | [CURRENT_SCIENCE_STATUS](CURRENT_SCIENCE_STATUS.md) |
| 代码复用与历史 | [CODE_MAP](docs/CODE_MAP.md) / [实验路线](docs/EXPERIMENT_ROADMAP.md) |
| 发布与运行边界 | [REPOSITORY_WORKFLOW](docs/REPOSITORY_WORKFLOW.md) |
| 本次整理范围与验证 | [整理报告](docs/maintenance/2026-11-03/WM01_ORGANIZATION_REPORT.md) |

## 不变的执行原则

`main` 是集成入口；新工作使用 feature branch / PR，服务器只消费固定 SHA。现役 source capsule、WBC 权重／参数、安全判据、原 DEV 面板与历史负结果不变。源固定底座 ARX5 成功不等于移动 Go2＋ARX5 成功；目标 S1/T2 隔离与真机限制继续有效。

使用现有 `~/Projects/GR00T-WBC`、profile、唯一执行 owner 与已授权 artifact 路由。Vanda/Core 的运行不互相依赖实时 SSH 或心跳。AMD 是共享存储／登录基础设施，不承接重型实验。4080S、A40、H200 的存在不是空闲显存或站点使用许可；以实际分配为准。

不删除模型、数据、分支或环境；不新增调度器、预算账本或逐函数审批框架。工程自主修复沿用 [Goal V7](docs/protocols/GR00T_WBC_SCIENCE_GOAL_V7_20261031.md)。

旧 README 的过时“D25/E1/E2 先于新技能”路线由 V7 及本次新研究方向替代；[原文完整保留在 Git](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/README.md)。本页是导航与研究方向，不是 WM 已实现、CI 已通过或远端已部署的证据。
