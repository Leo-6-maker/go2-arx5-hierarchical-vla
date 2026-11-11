# 当前入口 · 2026-11-11

## 先读什么

1. [原生机身＋腕部／公开 ARX5 迁移 V1](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/research/native-camera-arx5-transfer-v1-20261111/docs/protocols/NATIVE_CAMERA_PUBLIC_ARX5_TRANSFER_V1_20261111.md)：本轮具体设计与实现交接。
2. [CURRENT_SCIENCE_STATUS.md](CURRENT_SCIENCE_STATUS.md)：已核验事实、用户提供的最新结果、待补回执。
3. [AGENTS.md](AGENTS.md)：最新首轮范围与持续有效的工程／数据约束。
4. [docs/CODE_MAP.md](docs/CODE_MAP.md)：已有实现在哪一个固定版本。
5. [整合状态](docs/maintenance/2026-11-11/REPOSITORY_INTEGRATION_STATUS.md)：main 与执行分支的关系。

## 本轮下一步

先建立新相机版本并检查 TRAIN100 全阶段覆盖；用封存实际状态重渲染机身图，复用原腕图与原标签。公开 Table30 ARX5 的 episode 清单、时钟转换和分层划分可并行准备。相机通过后做四个 TRAIN 布局的 SFT2000 整机诊断，完成新官方起点／新缓存／GPU 常驻训练资格，再执行预注册 A/B。

不得把计划 JSON 当作已经满足准入的运行胶囊。现场路径、文件、GPU 所有权、完整端点、源数据时间约定和新 DEV 身份需要执行端提供实际回执。

## 固定代码与历史

当前参考执行 SHA：`f0e19c8812896f98053e74b165d6c52e85327b69`，来自 [PR #371](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/371)。它的 base 是另一执行分支，并非 main；main 整理不等于合并这条执行栈。新工作采用独立 checkout／worktree 与明确代码来源，不改变运行中的胶囊。

本轮不承接旧文档中的自动 Vanda/Hopper、WM 或扩大 Teacher 作业优先级；也不干预此前已授权且仍健康运行的作业。开工前只读核对实际状态。

- [整理前 START_HERE 全文](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/START_HERE.md)
- [整理前 CURRENT_SCIENCE_STATUS 全文](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/CURRENT_SCIENCE_STATUS.md)
- [固定执行快照的状态记录](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/CURRENT_SCIENCE_STATUS.md)
- [V2 文档 PR #372](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/372)；[V3/V3.1 文档 PR #373](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/373)

上述历史用于追溯，不应被当作当前机器状态或本轮已取得的实验结果。
