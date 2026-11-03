# 科学状态：结果与计划分开

整理日期：2026-11-03。证据基线：`0a27753bbda8ed75ab6b74aeb64755e0427362c7`；最后读取科学评论：Issue #11 comment `6601059002`。**本文件不是实时服务器监控。**

## 最后核验的 V7 结果（18:09–18:17 SGT）

| 对象 | 已有证据 | 不能由此推出 |
|---|---|---|
| D2 | 4096更新／131072 TRAIN访问，全状态与恢复证据保留、lease CLOSED；管理器随后因显存余量未达原门槛标失败 | 不抹去训练端点，也不声称新推理资格 |
| PB_CONTROL / PB_BOUNDARY | 前者480/4096，后者未启动 | 非完成训练或新DEV结果 |
| 共享 M32 | Vanda唯一后继1514400实际R、allocation通过、固定输入解包核验；77,847,684训练参数／原8192配方 | 未观察到模型更新、完整端点或新共享DEV |
| BOOK监督审计 | 21完整Teacher、各640提交／完成区间；最窄裕度在joint6 | 不能直接解释D0/D1的joint4越界 |
| TRAIN_VNEXT固定12 | 7几何静态通过、1拒绝、4恢复适配待完成，物理尝试与新增示范0 | 静态路径不是新成功数据 |
| Go2＋ARX5 | 几何、TCP、相机与四工作位仅离线核对 | 无行走／WBC动态／整机任务资格；S1/T2仍隔离 |

统一来源：[实际部署与科学结果](docs/operations/CORE_VANDA_DEPLOYMENT_CPU_RESULTS_20261103.md)；[科学评论](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-6601059002)；[运维评论](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47#issuecomment-6601052516)。这些作业当前状态必须由执行 owner 刷新，不重放旧身份。

## WM01 新研究线

| 项目 | 本次发布状态 |
|---|---|
| 科学计划、模块复用图、Codex工作包 | 已在本提交提供 |
| 静态计划检查器与其测试 | 已提供；本地检查范围见整理报告 |
| 状态转移采集／恢复适配器、WM、actor/critic、PPO与BPTT训练 | **NOT_IMPLEMENTED**，属于Codex后续实现 |
| WM专属有效数据、模型加载、优化更新、原生物理评价 | **本次均为0** |
| 远端部署、GPU/PBS/物理/真机启动 | **本次未执行** |
| WM优于PPO／可迁移至VLA或WBC | **待检验假设，不是结论** |

后续回执至少包含：真实数据量与失败分母、模型／策略更新、原生任务结果、总计算成本、首个机制性失败；不以fixture测试数、计划发布或loss下降代替任务成功。

## 历史可追溯性

[整理前完整状态](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/CURRENT_SCIENCE_STATUS.md) 保留所有原长日志；原 `reports/`、协议、模型和数据未改写。本页只保留当前摘要，历史失败／费用／STOP不由新计划撤销。
