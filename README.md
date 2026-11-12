# Go2 + ARX5 + GR00T / SLA + frozen WBC

研究目标：让视觉语言策略通过明确的 EEF 轨迹接口控制完整 Go2＋ARX5，在冻结 WBC、腿策略和原生安全条件下完成操作任务。

## 当前推进：Corelab 并行数据生产与 H200 后训练

2026-11-13 用户已允许 **Corelab 多环境并行采集**，要求关注效率并消除不必要的 CPU/GPU 往返。当前工作从真实数据与可部署策略推进：

- 原生独立 actor：持久 body/wrist renderer、内存观察、事件就绪、有界异步 writer；每世界独立控制。
- GPU 常驻 learner／teacher／策略服务：保留必要的一次输入／动作边界，不逐层或逐步搬运大张量。
- D100→D300→条件性 D1000，结合专家纠正与优化预算曲线；就绪门／书架技能从完整种子立即启动。
- [完整 V3 计划](docs/protocols/SLA_CORELAB_PARALLEL_DATA_H200_V3_20261113.md) · [机读矩阵](docs/protocols/SLA_CORELAB_PARALLEL_DATA_H200_V3_20261113.json) · [Codex 执行交接](docs/handover/2026-11-13/SLA_CORELAB_PARALLEL_V3_CODEX_START.md)。

## 已有结果与执行版本

最新核验快照是 [PR #380 的 aaf80b4](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/aaf80b4eda7ecb2816c0c251d3535a4e54ea62c9)：S1 已完成500更新／16,000访问；V2已有logit数据与batch CPU检查，真实V2训练／物理仍未取得结果。详见 [科学状态](CURRENT_SCIENCE_STATUS.md)。发布计划不表示已经达到吞吐或任务目标。

main是整合与发布入口，执行栈尚未整体合并。新实现基于实际核验的执行SHA，在独立worktree推进；不热改活动capsule。继承 [V2计划 #379](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b07936899b10f196653f617f95c5cd8b124e5e99/docs/protocols/SLA_GO2_ARX5_SCALEUP_V2_20261112.md)，保留现役S1→S2→G比较及原生相机／公开ARX5历史工作。

## 阅读与贡献

先读 [AGENTS.md](AGENTS.md)、[START_HERE.md](START_HERE.md) 和 [仓库工作流](docs/REPOSITORY_WORKFLOW.md)。旧CORR2048、SFT2000、C500、BF16 REJECT、DEV/TEST和Teacher24负结果均保留；源数据实测EEF与目标专家消费命令的标签定义仍分开。

- [代码地图](docs/CODE_MAP.md)与[11/11整合记录](docs/maintenance/2026-11-11/REPOSITORY_INTEGRATION_STATUS.md)用于历史定位。
- [11/11入口快照](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6843163d641eb24a29c3b1557f868765346ba66c/README.md)。
- [Goal V7历史总纲](docs/protocols/GR00T_WBC_SCIENCE_GOAL_V7_20261031.md)。
