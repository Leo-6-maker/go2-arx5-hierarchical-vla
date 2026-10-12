# Go2＋ARX5 Hierarchical Task-Space VLA

> **2026-10-12：新local-pc Codex先接管，再申请下一科学实验。** main保留较早accepted代码，本次只更新导航，不合并现役实验实现、不重写历史。

## 当前入口

| 需要 | 固定入口 |
|---|---|
| 新窗口完整接管prompt | [Local-pc adoption / 双服务器与HF规范](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3d3cecaa5a5d8b9c6a17158912b6f5816702b678/docs/handover/2026-10-12_LOCAL_PC_ADOPTION.md) |
| 现役分支入口 | [START_HERE](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3d3cecaa5a5d8b9c6a17158912b6f5816702b678/START_HERE.md) |
| 科学协议/实际进度/审核 | [Issue #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11) |
| 工程/CI/分支维护 | [Issue #47](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47) |
| 当前实验实现 | [pi0267/executor-decoupled-recovery](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/pi0267/executor-decoupled-recovery) |
| 分支审计，尚未删除 | [allowlist](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3d3cecaa5a5d8b9c6a17158912b6f5816702b678/docs/handover/2026-10-12_BRANCH_RETIREMENT_PLAN.json) / [精确lease清理流程](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3d3cecaa5a5d8b9c6a17158912b6f5816702b678/docs/handover/2026-10-12_BRANCH_CLEANUP_RUNBOOK.md) |
| 最新导航快照 | [Project status](docs/PROJECT_STATUS.md) |

## 研究目标与资源分工

视觉＋语言＋实际TCP/夹爪 → GR00T的任务空间EEF/夹爪参考 → 明确的坐标/时间/安全接口 → 师兄的冻结WBC → 实际反馈。

上层团队负责独立物理任务、可信专家数据和GR00T学习；不继续替师兄调WBC。EEF表示降低本体耦合，不自动证明任意参考可执行或跨本体泛化。

用户最新安排：Corelab做仿真/教师数据/预处理/物理推进/CPU评分/视频；Hopper H200做GR00T训练、神经推理和模型eval；local-pc为唯一owner。两服务器不能互连，数据和模型通过HF固定revision交接。真正闭环需要当前动作后的新观测，未来可由local-pc双SSH中继；HF不是每0.4秒的实时消息队列，离线预测不是闭环成功。

## 最近科学证据，非实时GPU状态

PI0285源Panda确认20/32未达26/32。PI0286 C0 TRAIN2/8、C1 TRAIN8/8与DEV5/16，新确认未运行。PI0289安全启动PASS，但首hold姿态RMS0.0652rad>0.05与时钟残差47.8ms>25ms未过门；其他五运动参考和目标教师数据未完成。最近已读取ledger终局为2026-10-10 11:08:34 SGT，开始工作时必须刷新#11。

TaskAnchor/UNITAS为研究设计稿，未作为新方法在本项目实训。接管本身不授权GPU、forward、teacher、仿真、WBC或硬件；旧deadline和预算不会随窗口重置。selected/deployment=NONE。

## main与历史保护

main仍为accepted较早代码＋导航；运行一律以各协议/收据固定SHA为准。旧PR77不是必写入口，旧hpc脚本不得自动重跑。私有WBC/vendor模型/源码/mesh/权重/密钥不公开，大资产留已授权HF/服务器，以完整manifest绑定。

之前main README完整保存在[2026-10-01导航版本](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1f05bfc919b25d3146839529d2763ac847010d7d/README.md)。旧[Research index](docs/RESEARCH_INDEX.md)、[Repository workflow](docs/REPOSITORY_WORKFLOW.md)、[Consolidation result](docs/maintenance/2026-10-01/RESULT.md)、[Timeline](docs/RESEARCH_TIMELINE.md)保留其历史日期；它们不构成当前科学GO。

本次已审计9分支、0开放PR；四个祖先head列入条件清理，三个含独立提交的旧branch保留。远端branch删除尚未执行，由local-pc备份兼容tag、检查实时SHA/保护/PR/lease后按allowlist执行并回读；本次不删除历史文件、模型或数据。
