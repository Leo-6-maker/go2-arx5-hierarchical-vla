# 当前入口 · 2026-10-28

`main`是唯一集成入口。先读[当前科学状态](CURRENT_SCIENCE_STATUS.md)、[最新实验路线](docs/EXPERIMENT_ROADMAP.md)与[AGENTS](AGENTS.md)，不要从旧日期协议中推断当前权限或运行状态。

## 下一步只推进两条科学工作流

| 平台 | 当前任务 | 真正交付 |
|---|---|---|
| CoreLab 4080S | 完成D25原N294101的24条件，再评价GEN1 | 完整行为表、独立审计、配对分析与关闭 |
| Vanda | 修通现有有限后继入口，执行E0与E1/E2真实资格，再各512步 | 原始模型结果、真实梯度／冻结检查、两个完整端点 |

两站运行彼此独立。管理端在启动前封存有限范围和必要权限，运行中各站本地记录效果，结束后独立核账；不依赖Core→Vanda持续SSH。模型只走已批准HF固定版本，禁止重复传输已经核验的文件。工程错误不是策略任务失败。

## 当前必须知道的事实

- A4096=11/24、B4096=6/24；C50完整24条件中6成功，三者是源域开发面板，不是完整发布或泛化资格。
- D25和GEN1各1024训练已完成；截至整理基线，各自仍0个有效Eval条件。
- Vanda同一后继请求已首次实际提交1504095并进入R；E0原始模型1次加载／39查询及39份保存输出已独立回读，E1资格已有2次优化调用。完整资格、E1/E2各512及后续Eval尚未完成；原失败／UNKNOWN保留，运行source不随main更新。
- 新的Go2＋ARX5标定／目标示范计划独立于当前12条固定基座示范，不混入E1/E2。原DEC1固定CORR/R0目标比较仍0/64。

## 读取顺序与证据

1. [EXPERIMENT_ROADMAP](docs/EXPERIMENT_ROADMAP.md)：研究问题、固定对照、阶段出口。
2. [CURRENT_SCIENCE_STATUS](CURRENT_SCIENCE_STATUS.md)：有时间与来源的结果快照。
3. [CODE_MAP](docs/CODE_MAP.md)：已有模块与历史实现的区别。
4. [Issue #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11)、[Issue #47](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47)及[原生进度日志](docs/handover/2026-10-25/AUTONOMOUS_PROGRESS.md)：刷新最新实际回执。
5. 实际执行前，由原执行owner核本机进程、lease、资源、完整输入和有效截止；GitHub页面不是实时GPU监控。

## 整理边界

本次只改变仓库集成与导航。旧source、数据、模型、WBC、R0、安全门、评分、统计口径和实验时钟不变。旧分支内容已合并；远端ref删除仍见[精确清单与状态](docs/maintenance/2026-10-28/CONSOLIDATION_REPORT.md)，不得把清单当删除回执。

后续更新这份短入口，不再不断向文件顶部追加数百条完整日志。详细结果留原报告，当前状态写状态页。整理前完整START_HERE按相同Git blob保存在[历史副本](docs/archive/2026-10-28/pre-consolidation/START_HERE.md)；其相对链接按原位置解释，不是新的执行指令。
