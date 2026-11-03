# 当前入口 · V7 现役运行 + WM01 新研究线

更新时间：2026-11-03（Asia/Singapore）。本次读取基线：`0a27753bbda8ed75ab6b74aeb64755e0427362c7`。**仓库发布不等于远端运行。**

## 新交接：WM01

用户要求整理代码库，并将小型 WM 辅助任务策略学习的详细计划交给 Codex 执行。

1. 先读根 [AGENTS](AGENTS.md)；其 V7 工程自主修复、源域优先、冻结 WBC 与站点安全继续有效。
2. 读 [WM01 Codex 执行交接](docs/handover/2026-11-03/WM01_CODEX_EXECUTE.md)。按其中的依赖顺序实现和验证，不把本文当作已有训练程序。
3. 数值实验以 [WM01 protocol](docs/protocols/WM01_TASKSPACE_WORLD_MODEL_POLICY_V1.md) 与 [plan.json](experiments/wm01/plan.json) 为设计输入。现阶段 `runtime_status=NOT_IMPLEMENTED`，首个交付是可恢复的原生状态—命令转移和一个实际 WM smoke，不是新的大规模 VLA 微调。
4. 当前真实结果见 [状态页](CURRENT_SCIENCE_STATUS.md)。接管前只刷新相关 Issue #11/#47 及本地 owner/job 状态，不重读全部历史。

## 现役工作，不被新计划抢停

最后核对的 11月3日18:09–18:17 SGT 回执：D2 完成4096更新但后续显存余量分类失败；PB_CONTROL 到480/4096；Vanda 共享 M32 后继1514400在实际分配后核验输入，尚未报告模型更新。以上是当时快照，不是此刻服务器探针。[完整实际报告](docs/operations/CORE_VANDA_DEPLOYMENT_CPU_RESULTS_20261103.md)。

新 WM 采用独立实验 ID／数据／配置／输出；不改 M13、PB、shared59、8192、DEV68 或任何活动 source。现役资源被占用时，先做 WM 的无副作用代码与离线数据审计，在干净边界安排新作业。

## 只保留短导航

代码：[CODE_MAP](docs/CODE_MAP.md)；路线：[EXPERIMENT_ROADMAP](docs/EXPERIMENT_ROADMAP.md)；分支：[REPOSITORY_WORKFLOW](docs/REPOSITORY_WORKFLOW.md)。

原长入口不丢失：[完整前版本](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/START_HERE.md)。原长状态：[完整前版本](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/CURRENT_SCIENCE_STATUS.md)。历史协议／报告路径均未移动，未来详细回执写入 `reports/`，本页只更新摘要与链接。
