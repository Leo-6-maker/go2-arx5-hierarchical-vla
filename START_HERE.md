# 当前入口 · 2026-11-13

## 先读什么

1. [Corelab并行数据生产＋H200后训练 V3](docs/protocols/SLA_CORELAB_PARALLEL_DATA_H200_V3_20261113.md)：本轮主计划、传输合同、实现顺序、数据与训练预算。
2. [机读矩阵](docs/protocols/SLA_CORELAB_PARALLEL_DATA_H200_V3_20261113.json)：设计，不是已就绪的运行配置。
3. [Codex执行交接](docs/handover/2026-11-13/SLA_CORELAB_PARALLEL_V3_CODEX_START.md)：第一批实际工作与回报。
4. [CURRENT_SCIENCE_STATUS.md](CURRENT_SCIENCE_STATUS.md)与[AGENTS.md](AGENTS.md)：事实快照、用户授权和保持的科学约束。

## 本轮优先顺序

用户已授权Corelab多环境采集。先核查现役unit／终态与实际资源，再并行推进：

- Corelab：body/wrist持久相机→内存snapshot与异步writer→事件就绪→独立2／4 actors→D100／D300。
- H200或实际可用训练设备：完成V2真实microbatch梯度与恢复→常驻learner／teacher→普通RGB包。
- 动作接口：logit夹爪的原生意义、保留中间路径的consumer与独立reference续期。
- 数据一版就训练一版，依据学生失败采恢复数据；更长训练只给选中配方，三seed匹配同一日程。

不再为每档actor数询问相同许可；每world独立owner。expert采集不需要SLA推理。避免PNG落盘再读回、重复创建renderer、逐microbatch同步和GPU特征经NumPy中转。

## 固定来源与现役衔接

- [最新核验执行 #380，aaf80b4](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/aaf80b4eda7ecb2816c0c251d3535a4e54ea62c9)。
- [最新实际快照，11/13 00:02:56 SGT](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/aaf80b4eda7ecb2816c0c251d3535a4e54ea62c9/reports/WM02/sla-scaleup-v2-20261112/CURRENT_STAGE_ACTUAL.json)。
- [继承V2 #379，b079368](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b07936899b10f196653f617f95c5cd8b124e5e99/docs/protocols/SLA_GO2_ARX5_SCALEUP_V2_20261112.md)。
- [原S1/S2/G执行 #378](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/378)：先读当前实际unit／终态，旧WAIT和有限期限不是实时状态。

main尚未包含整条执行栈。实现用独立worktree和明确source SHA，不把文档中的拟议组件当已有命令。无健康作业热改、无未知状态重投。

## 历史保留

[11/11入口](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6843163d641eb24a29c3b1557f868765346ba66c/START_HERE.md) · [11/11科学状态](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6843163d641eb24a29c3b1557f868765346ba66c/CURRENT_SCIENCE_STATUS.md) · [代码地图](docs/CODE_MAP.md) · [仓库工作流](docs/REPOSITORY_WORKFLOW.md)。

既有原生相机／公开ARX5、GR00T、Teacher24及旧DEV/TEST保留各自来源和结论。V3发布只表示计划与授权已记录。
