# Go2＋ARX5 Hierarchical Task-Space VLA

> **2026-10-13：统一工作入口与同步规范。** CoreLab 日常入口为 `~/Projects/GR00T-WBC`。main 仍保留较早 accepted 代码；本次只更新导航，没有把实验运行实现合入 main，也没有部署/启动远端任务。

## 唯一现行入口

以下链接固定到现役发布 `16d38237c660c7ae8b38c27fcfada274ca5670fd`：

| 需要 | 入口 |
|---|---|
| 新 Codex 开始工作 | [START_HERE](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/16d38237c660c7ae8b38c27fcfada274ca5670fd/START_HERE.md) / [AGENTS](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/16d38237c660c7ae8b38c27fcfada274ca5670fd/AGENTS.md) |
| 统一目录、环境、数据与 GitHub 同步 | [Canonical workspace standard](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/16d38237c660c7ae8b38c27fcfada274ca5670fd/docs/operations/2026-10-13_UNIFIED_ENTRY_AND_GITHUB_SYNC_V1.md) |
| 后续任务强制接入与固定版本部署 | [Enforcement supplement](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/16d38237c660c7ae8b38c27fcfada274ca5670fd/docs/operations/WORKSPACE_ENTRY_ENFORCEMENT_V1.md) / [ENTRY_POLICY](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/16d38237c660c7ae8b38c27fcfada274ca5670fd/infrastructure/workspace/ENTRY_POLICY.json) |
| 现有 local-pc Codex 接续 | [Execute prompt](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/16d38237c660c7ae8b38c27fcfada274ca5670fd/docs/handover/2026-10-13_WORKSPACE_ENTRY_EXECUTE_PROMPT.md) |
| 最新科学授权和实际结果 | [Issue #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11) |
| 实际入口部署、迁移和清理 | [Issue #47](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47) |
| 现役实现分支 | [pi0267/executor-decoupled-recovery](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/pi0267/executor-decoupled-recovery) |

## 版本和运行规则

过渡期唯一集成分支是 `pi0267/executor-decoupled-recovery`；不是默认 main。新运行绑定明确的 protocol/code/source/entry/data/model/processor/statistics/environment 版本和 owner/run UUID，不用 mutable HEAD 取代运行快照。通用实现以后经单独 PR、测试及路由切换记录提升到 main，不强行覆盖历史。

local-pc 负责协调和发布；CoreLab 负责仿真、数据和 CPU；Hopper 神经执行须合法 PBS。源码/模板/小收据进 GitHub；大数据、完整 Trainer、视频通过已授权私有 HF 固定 revision 和成员 hash 交付。私有 WBC、mesh、凭据和环境不打包进源码 Git，整个工作区不是一个 Git 仓库。

已有三个 CoreLab enter.sh profiles 曾通过实际验收；新通用 workspace CLI、强制接入和物理目录继续收敛必须由新的部署收据证明。不要因为看到这份规范就认定全部命令已实现、所有旧目录已删除，或现役实验可以重启。旧 PI0291 PARTIAL、当前 PI0292 运行身份和全部历史结果保留。

## 历史

之前 README、docs/PROJECT_STATUS、RESEARCH_INDEX、REPOSITORY_WORKFLOW、CONSOLIDATION/TIMELINE 文档保留各自历史日期；不再以其旧“最新状态”或截止构成新的执行权限。最新科学事实刷新 #11 与实际 run；工作区实施刷新 #47。当前工作应从上面的固定现行入口进入。
