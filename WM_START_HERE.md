# WM02：直接学习任务空间 Actor

日期：2026-11-04，Asia/Singapore。用户要求基于最近讨论制定详细执行计划并交给 Codex。发布状态：**PLAN_PUBLISHED / RUNTIME_NOT_IMPLEMENTED**，不代表服务器已启动。

本后继实施分支的实际回执：**P0目标身份与阻断已核对；P1完成S域薄adapter／原生恢复与4进程吞吐，目标动态仍TARGET_BLOCKED。** 见[第一轮实际回执](reports/WM02/wm02-20261104-p0-p1/FINAL_REPORT.md)与[真实CLI](experiments/wm02/RUNTIME.md)。原计划发布状态和plan.json作为历史设计保留，不表示main已经部署。

主线：**Go2＋ARX5 原生冻结 WBC 物理交互 → 小型动作条件 WM → 短期想象优化 Actor → 原生完整任务评价 → 双 RGB 视觉学生**。GR00T 是可选对照，不是必经组件；固定底座 ARX5 只是方法诊断域。

1. 阅读根 `AGENTS.md`，再读 [Codex 交接](docs/handover/2026-11-04/WM02_CODEX_EXECUTE.md)。
2. 研究定义见 [WM02 协议](docs/protocols/WM02_GO2_ARX5_WORLD_MODEL_ACTOR_V1.md)；数字与工作包见 [plan.json](experiments/wm02/plan.json)。
3. 复用范围、来源和未解决接口见 [实现映射](docs/protocols/WM02_UPSTREAM_ADAPTER_MAP_V1.md)。
4. **最新本地资源顺序**：先按 [RTX 4080S 16GB本地优先交接](docs/handover/2026-11-04/WM02_4080S_LOCAL_FIRST.md) 完成原生后端核验、动作/状态/接触、WM及Actor短循环；具备真实测得的profile、可恢复checkpoint与目标合法资格后，才提交A40/H200 PBS。不要在本地调通前提前排正式百万步作业。

WM02 是独立新实验，不需要先合并 PR #342。WM01 在 `de64c252246e4746315f2a324535b5d48912783c` 保留，不改其结果或参数。WM02 对新的 WM 研究优先级取代“必须回到 GR00T”和“默认 PPO＋WM 混合主线”；不修改已封存 V7 运行、旧比较、WBC、安全、目标 S1/T2 隔离或站点权限。目标动态工作须绑定当前有效准入，不能把本计划当解除隔离的证据。

**第一交付不是训练曲线：是实际后端/WBC身份、动作时间映射、控制状态递推与恢复、目标接触链和真实采样吞吐。** 这些明确后才能解释模型学习结果。普通工程缺陷按 V7 自主修复，不另建调度、审批或预算系统。

本发布只有文档和机器可读设计，没有伪装成可运行程序的训练命令。新的训练/采集入口需 Codex 实现、测试并登记；本次未执行 CPU 测试、CI、GPU、仿真、SSH 或 PBS。
