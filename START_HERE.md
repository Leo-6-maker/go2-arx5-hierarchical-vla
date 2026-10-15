# GR00T＋WBC — 2026-10-15 当前科学入口

**最新实际任务结果：固定四个整机pilot成功0/4。** V1近场抓放原生安全FAIL保留；V2其余三条完成32秒、真实双RGB／参考／时序及独立CPU一致性通过，但物体任务均FAIL。成功数据0，v4.0／FB13未准入；未运行新训练。已封存全部原件／私有视频及逐文件回读，冻结WBC，不重复有效失败。见[完整结果与收口收据](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/FIXED_PILOT_RESULTS.md)。执行完成与任务成功分开，整体数据—训练—闭环仍PARTIAL，selected/deployment=NONE。

**最新用户决定：不调整WBC，只借用完整环境采集任务数据和训练GR00T。** [主线纠偏与Codex接续规则](docs/protocols/GR00T_TASK_DATA_FIRST_FROZEN_WBC_SCOPE_V1.md)（内含机器规则）。R4B D0已结束，D1及继续控制器归因/补偿/调参不再属于本团队下一步。WBC保持不变。

**新采集采用“真实物体任务成功＋原生安全＋数据完整性”准入；旧保持RMS质量FAIL保留，但不再作为全局数据采集/训练的前置阻断。** 不调高0.05rad、不删除旧FAIL、不忽略实际限位/非法碰撞。应补齐任务夹爪/相机/Teacher，直接检查原样环境能否完成抓放与推送，再制作EEF13数据、并行采集和微调。脚本存在不等于任务或数据已经通过。

本方向已由用户明确，不新增任何资源总量或窗口；owner需绑定有效任务/资源scope，不能自动重开关闭的lease。以下是已发布的事实与历史阶段依赖；其中的“依赖链已停止”描述原阶段终局，不代表新任务还要等待WBC修复或保持精度PASS。

---

**R4B D0实际只读归因已完成：同控制周期FK与力矩证据支持手臂伺服实现偏差主导姿态误差，并有身体预测/实现耦合；尚未证明可自主修的实现bug。** [WBC负责人缺陷包](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/R4B_D0/WBC_OWNER_DEFECT_NOTE.md) · [只读终账](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/R4B_D0/RESOURCE_AND_FINAL_REPORT.md)。新增物理/模型/PBS均0；R4B D1仍未采用，不新开窗口。

**最新实际结果：R4A 原生完整机器人完成10秒世界系EEF保持，姿态RMS0.06965rad超过原0.05rad门；时钟、参考安装及原安全检查通过。整机教师/数据/FB13依赖链已停止，尚无整机学习模型。** [实际报告与CPU收据](reports/PI0294/20261014T170930Z-incoming-pi0294-r4/FINAL_NATIVE_QUALIFICATION_REPORT.md)。原始失败保留；这是有效负结果，不是R4A整体完成，selected/deployment仍NONE。

**最新用户方向：完整Go2＋ARX5环境生产数据和测试，GR00T增加有界BASE速度意图，经原生BASE软位姿接口接入冻结WBC；工程问题在明确范围和既有资源内自主修复推进。**

- [R4A完整长程计划](docs/protocols/PI0294_R4A_FULL_GO2_ARX5_AND_BASE_INTENT_V1.md)
- [EEF13／BASE软参考接口契约](docs/interfaces/EEF13_BASE_SOFT_REFERENCE_V1.md)
- [R4A机器任务卡](reports/PI0294/design/R4A_FULLBODY_TASK_CARD.json)
- [现有Codex直接接手Prompt](docs/handover/2026-10-15/PI0294_R4A_EXECUTE_PROMPT.md)

读取根AGENTS、最新#11/#47与原生运行状态后采用。最新已核验模型是M2048；R4旧v3.1增量在v3.0统计下codec失败保留。固定底座实验是基线，不再是完整整机搭建的永久前置。已运行作业不得热改或重复提交；新scope只使用已授权R4剩余资源和窗口，不新增预算、不重置T0。完整WBC在CoreLab仅作冻结推理/仿真执行器，高层GR00T仍在合法Hopper allocation；不授权WBC重训、实机或历史删除。

完整整机、EEF13与BASE桥在有真实执行收据前是待实施，不因文档发布变成PASS。下列2026-10-14及更早的“当前状态/未授权/未推理”是历史接管记录，不应覆盖后续已发生的R2/R3/R4事实或本次范围增补。

---

# GR00T＋WBC — 唯一仓库与 CoreLab 工作入口（2026-10-13）

**新 Codex 与所有后续工作请先从这里开始。** GitHub 是本团队 VLA 代码、协议、测试、小型执行收据的版本权威；CoreLab 共享 `~/Projects/GR00T-WBC` 是人类统一工作入口，但不能把私有原始数据、WBC、权重都视为已保存在 GitHub。

- **[2026-10-14 新 agent 完整接管与科学推进计划](docs/handover/2026-10-14/NEXT_AGENT_EXECUTION_PLAN.md)** · [机器任务卡](docs/handover/2026-10-14/NEXT_AGENT_TASK_CARD.json) · [直接接手 Prompt](docs/handover/2026-10-14/NEXT_AGENT_BOOTSTRAP.md)：当前 PI0294 仍因真实夹爪原生限位 BLOCKED，修订待 GO；这是计划不是自动执行授权。今后交接由 GitHub 固定提交直接供 Codex 读取，不以本地 ZIP 为主。
- **[2026-10-14 完整工作总结与下一 owner 交接](docs/handover/2026-10-14/HANDOFF.md)**：Oct12-14 已执行事实、代码入口、固定 HF 资产、终局与剩余预算。当前 M512 实际闭环触发原生夹爪关节限位，科学线 BLOCKED；交接不授权重启。夹爪重资格方案仍未获明确采纳。
- **[现行统一入口／GitHub 同步规范](docs/operations/2026-10-13_UNIFIED_ENTRY_AND_GITHUB_SYNC_V1.md)**：host-aware profiles、固定源 SHA、运行 manifest、分支同步、旧目录退役验收。
- **CoreLab 现有可执行入口**：`~/Projects/GR00T-WBC/infrastructure/enter.sh amd-management`（AMD），或在 4080S 使用 `4080-simulation`／`4080-cpu-data`。不提供自动跨主机 SSH；不经许可启动物理／模型。
- **[最新真实维护终局](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47#issuecomment-6137981521)**：已实际退役33源路径；Projects仍7项，有受保护科学依赖。后续统一入口操作只凭新执行收据计为完成。
- **[科学主线 #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11)**：PI0292 以其独立 run capsule、source seal 和当前资源账为准，不要用当前分支 HEAD 自动替代已在执行的源码。
- **Git**：实际活跃科学分支 `pi0267/executor-decoupled-recovery`；`main` 是较早的稳定导航，不是当前仿真或训练启动源。只有作业边界+doctor通过，才允许新 `code/vla` checkout fast-forward，同步本地不改历史 run。

以下是 **2026-10-12 的历史接管文档**，保留供追溯；其中的“尚未授权实验／旧最新状态”不再作为当前实时状态或新 GO。

---

# 先读这里｜2026-10-12 新local-pc接管

**当前是接管与仓库维护，不是新的训练/仿真GO。**

## 唯一入口

- [完整local-pc接管prompt与双服务器/HF规范](docs/handover/2026-10-12_LOCAL_PC_ADOPTION.md)
- [分支清理allowlist及原SHA](docs/handover/2026-10-12_BRANCH_RETIREMENT_PLAN.json) — 尚未实际删除；local-pc先建立兼容tag并回读，再按精确lease删除四个祖先分支。
- [科学ledger #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11)
- [可复用代码/CI/仓库维护 #47](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47)

## 最新用户方向

我们负责独立任务仿真、可信专家数据与上层GR00T的EEF/夹爪行为学习。师兄WBC仅作冻结执行器，不再由上层团队调参修复。TaskAnchor/UNITAS只是设计参考，尚未实现。本轮只读接管、条件化分支维护与下一数据GO草案；不恢复旧实验、不启动新模型或物理。

| 位置 | 职责 |
|---|---|
| local-pc | 唯一owner；分别SSH两端、GitHub审核交互；未来可作闭环obs/action中继 |
| Corelab | 仿真/教师数据/预处理/物理推进/CPU评分/视频，非高层GR00T模型eval |
| Hopper H200 | GR00T训练微调、模型推理与模型eval，须合法PBS allocation |
| GitHub | 代码、协议、有限预算、小收据、审核 |
| HF | 数据/权重/大证据，固定revision、prefix和hash |

两服务器互不连通；不要尝试直连scp/ZMQ。HF用于大资产，不是每0.4秒的实时控制队列。闭环需要当前动作造成的新观测；local-pc双SSH中继尚需另行验证，离线回放不能冒称闭环。

## 最近完整科学状态

PI0285源Panda确认20/32未过26/32。PI0286 C0 TRAIN2/8；C1 TRAIN8/8、DEV5/16；未运行新确认。PI0289安全启动PASS，但首hold姿态RMS0.0652rad>0.05、时钟残差47.8ms>25ms；其余五参考NOT_RUN，目标教师数据0。最近ledger终局是2026-10-10 11:08:34 SGT；接手须刷新最新证据。selected/deployment=NONE。

现役代码分支pi0267/executor-decoupled-recovery。main仍是旧accepted代码＋导航，不能直接当训练runtime。旧README、hpc gate、10月8–10日deadline与PR77不构成当前执行授权。

第一轮交付：ADOPTION_REPORT、RESOURCE_MAP、ASSET_AND_HF_MAP、NETWORK_AND_EVAL_PLAN、实际BRANCH_MAINTENANCE_RECEIPT、NEXT_DATA_GOAL_DRAFT（未授权）；发#11回读后停住等新科学GO。
