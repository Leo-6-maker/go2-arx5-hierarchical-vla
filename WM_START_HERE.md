# WM02：直接学习任务空间 Actor

日期：2026-11-04，Asia/Singapore。原计划已发布；本实施分支已有实际入口和执行结果，不表示main或旧capsule已部署。

**最新动作响应／初始化候选已执行：** 当前40布局FIT定位到参考→TCP效应压平；固定200的M1/M2比较完成400主更新＋2恢复探针，另保留失败20更新。M2在诊断DEV的H8预测物体朝向效应约8.09mm，实际约7.94mm；尚未优于规则，时间响应不完整。目标R1＋停止锁存＋有界记录器已实际编译并核验两条加载链，但缺少init-only范围，原生初始化仍未验收、目标积分0。见[本轮真实命令、身份、成本与限制](reports/WM02/response-init-candidate/EXECUTION_REPORT.md)。

本后继实施分支的实际回执：**P0目标身份与阻断已核对；P1完成S域薄adapter／原生恢复与4进程吞吐，目标动态仍TARGET_BLOCKED。** 见[第一轮实际回执](reports/WM02/wm02-20261104-p0-p1/FINAL_REPORT.md)与[真实CLI](experiments/wm02/RUNTIME.md)。原计划发布状态和plan.json作为历史设计保留，不表示main已经部署。

**环境优先后继已实际完成：** Core预检23份冻结文件、独立CUDA环境、WM200主更新＋1恢复探针、Actor/Critic各1次H16更新；仅合成工程烟测。目标原生与真实学习闭环仍未完成。见[最新执行回执](reports/WM02/environment/EXECUTION_REPORT.md)、[控制状态缺口](reports/WM02/environment/TARGET_CONTROL_STATE_MAP.md)及[站立T1评分版本](reports/WM02/environment/TARGET_SCORER_BINDING.md)。

**真实S域后继：** 原872条→496个不同观测对／2物理组，真实Replay已实现；GPU1 WM主200＋恢复探针1完成。位置略优于持恒，但动作消融近似打平、姿态／抓持更差，结论INCONCLUSIVE；Actor0、目标动态0。见[真实数据执行与负结果](reports/WM02/real-replay/EXECUTION_REPORT.md)。

**最新配对动作辨识已执行：** 新24布局／96锚点／480分叉、主7,680条真实转移；匹配动作与无未来动作WM各200＋恢复探针1，同80轨迹评价。H16物体末端5.908／6.897mm、动作进展排序51/59对，但释放和可改善任务条件不足，Actor0。另在四DEV的1.6s完整状态完成新448条原生接触探针，名义闭爪4/4抓持、延迟闭爪1/4；未用于训练。本轮总新增S转移15,616，目标0，全部lease CLOSED。见[执行、完整SHA、成本与限制](reports/WM02/contact-identification/EXECUTION_REPORT.md)及[目标原始失败证据／新窗口路径](reports/WM02/contact-identification/TARGET_ISOLATION_EVIDENCE.json)。

主线：**Go2＋ARX5 原生冻结 WBC 物理交互 → 小型动作条件 WM → 短期想象优化 Actor → 原生完整任务评价 → 双 RGB 视觉学生**。GR00T 是可选对照，不是必经组件；固定底座 ARX5 只是方法诊断域。

1. 阅读根 `AGENTS.md`，再读 [Codex 交接](docs/handover/2026-11-04/WM02_CODEX_EXECUTE.md)。
2. 研究定义见 [WM02 协议](docs/protocols/WM02_GO2_ARX5_WORLD_MODEL_ACTOR_V1.md)；数字与工作包见 [plan.json](experiments/wm02/plan.json)。
3. 复用范围、来源和未解决接口见 [实现映射](docs/protocols/WM02_UPSTREAM_ADAPTER_MAP_V1.md)。
4. **最新本地资源顺序**：先按 [RTX 4080S 16GB本地优先交接](docs/handover/2026-11-04/WM02_4080S_LOCAL_FIRST.md) 完成原生后端核验、动作/状态/接触、WM及Actor短循环；具备真实测得的profile、可恢复checkpoint与目标合法资格后，才提交A40/H200 PBS。不要在本地调通前提前排正式百万步作业。

WM02 是独立新实验，不需要先合并 PR #342。WM01 在 `de64c252246e4746315f2a324535b5d48912783c` 保留，不改其结果或参数。WM02 对新的 WM 研究优先级取代“必须回到 GR00T”和“默认 PPO＋WM 混合主线”；不修改已封存 V7 运行、旧比较、WBC、安全、目标 S1/T2 隔离或站点权限。目标动态工作须绑定当前有效准入，不能把本计划当解除隔离的证据。

**第一交付不是训练曲线：是实际后端/WBC身份、动作时间映射、控制状态递推与恢复、目标接触链和真实采样吞吐。** 这些明确后才能解释模型学习结果。普通工程缺陷按 V7 自主修复，不另建调度、审批或预算系统。

原设计发布并非执行结果；本分支实现与实际执行量以固定SHA回执为准。真实S replay、配对采集与有限模型训练已有实际入口；正式任务reward／Actor在线建议兑现及目标闭环仍待实现与资格验证。
