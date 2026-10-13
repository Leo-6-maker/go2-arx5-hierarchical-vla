# Go2＋ARX5 Hierarchical Task-Space VLA

> **日常从 [START_HERE.md](START_HERE.md) 与根 [AGENTS.md](AGENTS.md) 开始。** CoreLab统一入口：`~/Projects/GR00T-WBC`。唯一现行规范是 [统一入口与GitHub同步](docs/operations/2026-10-13_UNIFIED_ENTRY_AND_GITHUB_SYNC_V1.md)；[强制接入补充](docs/operations/WORKSPACE_ENTRY_ENFORCEMENT_V1.md) 明确后续任务与固定版本部署要求，不另建平行入口。

## 代码、执行和证据入口

| 需要 | 入口 |
|---|---|
| 目录、代码、环境、数据及同步规范 | [Canonical workspace standard](docs/operations/2026-10-13_UNIFIED_ENTRY_AND_GITHUB_SYNC_V1.md) |
| 新任务接入与部署的补充要求 | [Entry enforcement](docs/operations/WORKSPACE_ENTRY_ENFORCEMENT_V1.md) / [ENTRY_POLICY](infrastructure/workspace/ENTRY_POLICY.json) |
| 现有Codex接续指令 | [Workspace execute prompt](docs/handover/2026-10-13_WORKSPACE_ENTRY_EXECUTE_PROMPT.md) |
| 科学授权及实际结果 | [Issue #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11) |
| 工作区/CI/存储实际结果 | [Issue #47](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47) |
| 当前ARX5数据/原生训练/闭环计划 | [PI0292 protocol](docs/protocols/PI0292_ARX5_NATIVE_B0_TRAINING_AND_CLOSED_LOOP_V1.md) |
| ARX5首版数据原窗口终局 | [PI0291 FINAL_REPORT](reports/PI0291/20261013T023717-localpc-arx5/FINAL_REPORT.md) |
| WBC独立资格的历史终局 | [PI0289](reports/PI0289/execution/REPORT.md) |
| 研究前提 | [任务空间研究章程](docs/RESEARCH_CHARTER_TASK_SPACE_MANIPULATION_V1.md) |

## 科学边界

视觉＋语言＋实际TCP/夹爪 → GR00T任务空间EEF/夹爪参考 → 显式frame/time/safety接口 → 已资格执行器 → 实际反馈。当前固定底座ARX5使用独立IK/有限力矩执行器，不代表原冻结WBC或移动Go2通过。上层数据/轨迹拟合不证明跨本体泛化，师兄WBC仅按对应协议使用。

## 分支和机器职责

过渡期唯一集成分支：`pi0267/executor-decoupled-recovery`。main为较早accepted代码与路由，不直接作为最新runtime。运行使用full SHA和独立source capsule，不能因远端更新而热pull活动目录。未来提升main经专门PR/测试/路由切换记录，不覆盖历史。

local-pc协调发布；CoreLab做仿真/数据/CPU；Hopper H200做合法PBS授权的GR00T神经工作。数据和完整模型使用已授权私有HF固定revision/prefix/成员hash。源码Git、私有配置和大资产分开，不git-init整棵工作区。

新workspace CLI/强制接入必须有实际部署收据；既有enter.sh的三个CoreLab profiles已通过前次验收。注册profile不等于运行模型，文档发布不等于目录已经迁完。旧run保持其授权、时钟和来源，不因规范改变而重跑。私有WBC/mesh/凭据、唯一原数据、完整Trainer与失败证据不因清理而丢弃。

## 历史保留

2026-10-12 README保存在commit `c187d9573bdff302783fb5f41d06fcaa1753d2a8`；START_HERE保留日期化旧交接。科学实时进度查#11及对应原始收据，不以旧导航的阶段/截止决定新执行权限。
