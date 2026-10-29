# 当前入口 · 2026-10-29 · 长程Goal V6

`main`是唯一集成入口。先读[科学状态](CURRENT_SCIENCE_STATUS.md)、[V6详细计划](docs/protocols/GR00T_WBC_LONG_HORIZON_GOAL_V6_20261029.md)、[路线索引](docs/EXPERIMENT_ROADMAP.md)和[AGENTS](AGENTS.md)。历史协议不自动提供当前权限。

## 下一步：两条科学线，一条只读支持线

| 位置 | 当前任务 | 完成标志 |
|---|---|---|
| CoreLab 4080S | 复用已补齐的LOCAL终态，冷态绑定D25原N294101/24，再GEN1 | 完整行为表、D−C配对及关闭 |
| Vanda | 复用E1@512做推理验证，再从原始父模型正式训练E2固定512 | 固定端点、真实更新/消费/保存恢复 |
| PC Rerun | 已有Teacher/Student失败解释与保存输出分析 | 可核验只读证据，不阻塞科学运行 |

两站运行不互相SSH/查队列/等心跳；管理端独立连接/终态核账，模型走已授权HF固定revision，不重传已交付D25。

## 当前关键事实

- **Vanda saved512接续1506193已真实R／GN-A40-033，首次qsub、绑定及独立完整队列回读已完成。** [实际回执](reports/PI0294/design/parallel-science-20261022/ARX5_SAVED512_EXISTING_REGISTER40_FIRST_ACTUAL_QSUB_AND_BACKGROUND_HANDOFF_20261029.json)：2026-10-29 12:15 SGT，unknown／pending为空，后台原生owner和wrapper同kernel健康。只复用原PBS40登记，无新预付或计数；source769c、metadata446ac8。顺序为E1已审计512推理验证（0新更新）→E2从原始父模型fresh512；当前新模型加载／查询／更新仍0，不把R称为模型资格通过。原登记回包UNKNOWN、安装键名及解析失败和费用保留；新metadata角色已完整核验，不热改或重启。Core连接由管理员处理，Vanda运行不等Core。

- **已有E0语言诊断新增只读分析：** 六个TRAIN接近阶段观测、12个实际同噪声指令对，前8步位置差中位数0.256毫米；固定指令换噪声为143.089毫米。[方法与局限](docs/analysis/2026-10-29/E0_SAVED_LANGUAGE_RESPONSE.md)。不等于忽略语言、监督拟合或闭环成功；新增模型调用0，原配方不改。

- **E1@512原12项参考探针：** 前8步位置误差宏平均0.108米，原动作门12/12拒绝。[原指标与人口边界](docs/analysis/2026-10-29/E1_SAVED512_REFERENCE_PROBES.md)：原COV参考读器不同于12父优化人口；六窗口匹配其中两个原父。未重新查询／评分，不能当作完整训练拟合率或闭环0%。

- A4096=11/24、B4096=6/24、C50=6/24；源域已暴露DEV，不是独立泛化。D25/GEN1各1024完成，仍无有效Eval。
- Vanda1504095已F/Exit1、once结算、own lease关闭：E0 39查询，E1两步资格/17查询/14目标参数变化，E2 15零步输出/优化0。该旧作业未执行正式训练；后继当前终态见下条。
- **PR99新增实际证据：E0/E1完整保存前缀独立审计PASS，E1重载同输入/noise输出精确一致；PBS41→42及一次控制范围已采用；Core LOCAL38条终态已补齐。** [原报告](reports/PI0294/design/parallel-science-20261022/ARX5_REAL_E0_E1_FULL_SAVED_PREFIX_AUDIT_AND_CANONICAL_CORE_HISTORY_CATCHUP_ACTUAL_20261029.json)。不重做同一审计/cap/mirror/settle。
- **最新原生终态：1505362已F/Exit1，E1完整512步状态及原12项TRAIN探针已保存；E2两步资格COMPLETE，正式E2尚未启动。** [失败与冷修复回执](reports/PI0294/design/parallel-science-20261022/ARX5_E1_REAL512_NATIVE_EXPORT_NOISE_MODE_FAILURE_AND_COLD_REPAIR_ACTUAL_20261029.json)。E1导出对比记录了同观测/同seed却不同实际noise；整图评估模式接缝已有37项CPU回归，未native采用。原一次结算/own关闭已完成并全文独立回读（4162GPU秒/3585CPU秒）；接续保存状态独立审计，保护E1端点、不重训；随后只接未完成的固定工作，不复用死owner或旧qsub。
- **保存端点独立审计已真实PASS／OS0：E1完整512、E2两步资格及E1普通导出全部成员二次流读，实际CPU314.544秒；普通导出tensor逐项相等。** [实际回执及冷接线](reports/PI0294/design/parallel-science-20261022/ARX5_SAVED_E1_512_FULL_STATE_AND_ORDINARY_TENSOR_AUDIT_REAL_PASS_RECOVERY_COLD_20261029.json)。接续控制和exact PBS43已一次采用、完整GLOBAL及原intent／receipt回读；used仍39。新typed候选只接E1推理0更新／25查询／3加载和E2 raw512／37查询／2加载，六个原owner门；当前尚未native登记／qsub，不能称GPU已启动或E1输出资格完成。原130秒超限审计和全部旧费用／UNKNOWN保留。
- **最新零GPU冷检查失败已定位并关闭：原13份TRAIN探针保存为0600，其SHA／inode／时间戳全部未变。** [精确回读与cold02修复](reports/PI0294/design/parallel-science-20261022/ARX5_SAVED512_LEGACY13_PROBE0600_ACTUAL_UNCHANGED_AND_COLD02_REPAIR_20261029.json)。只按原精确13文件清单逐次重新hash小文件，模型张量仍0400；旧文件不chmod。cold01原件／费用全保留并计入同一70GB组，新immutable角色接续，未登记或qsub。
- **cold02真实数据／语言读取已通过，随后旧空间guard因调用传0而按原收尾reserve规则拒绝；仍零GPU，已OS1关闭。** [cold03窄接续](reports/PI0294/design/parallel-science-20261022/ARX5_SAVED512_ORIGINAL_CLOSE_RESERVE_COLD02_KNOWN_FAILURE_AND_COLD03_REPAIR_20261029.json)只传回原reserve，两份冷目录原件均计入同一70GB组；8项CPU检查含实际全历史14维通过。新GPU仍未登记／提交。
- **cold03实际源码／输入／7212窗口和完整14维已PASS／OS0，并独立全文回读。** [实际资格与未派发顶层身份补齐](reports/PI0294/design/parallel-science-20261022/ARX5_SAVED512_COLD03_REAL_INPUTS_FULL14_PASS_AND_UNSTARTED_TOP_IDENTITY_COMPLETION_20261029.json)。顶层仍旧remaining类型，原adapter精确拒绝；owner／PBS未启动。仅冷态补一项类型及其binding／lease／audit哈希，旧字节先归档、原onceIO，不改source769c或数据，不重做已过读器；随后实际新鲜提交。
- Rerun验证三条稀疏记录（Teacher各41/641、失败2/224），不是完整整机回放；full qpos/mesh不是训练Eval前置。
- Go2+ARX5标定/12次任务pilot独立于当前12条源域示范；原DEC1仍固定CORR/R0，不混目标训练。

## 接续与证据

[Codex交接](docs/handover/2026-10-29/POST_RERUN_SCIENCE_FIRST_EXECUTION.md) · [计划卡](reports/PI0294/design/parallel-science-20261022/POST_RERUN_LONG_HORIZON_TASK_CARD_V6_20261029.json) · [CODE_MAP](docs/CODE_MAP.md) · [仓库工作流](docs/REPOSITORY_WORKFLOW.md)。计划卡不是可派发RUN_MANIFEST。

刷新[Issue #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11)、[Issue #47](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47)和原owner的本机原件。GitHub不是实时GPU监控，后继回执优先；已完成只接剩余项。普通同范围工程最小冷修，真实外部缺口一次准确报告。

stop-new仍2026-10-31 20:52:29.813 SGT，hard deadline仍2026-11-01 20:52:29.813 SGT。本次只发布计划，不延期、不采用新额度、不启动作业/子代理/暂停自动化；原WBC/R0/评分/失败保持，运行source不热切换。

根入口保持简短，完整证据在reports。历史可由Git/原字节归档恢复；分支退役按独立维护回执，不随计划发布删除refs。
