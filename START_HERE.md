# 当前入口 · 2026-10-29 · 长程Goal V6

**最新技术方向：** [开源任务复用](docs/protocols/OPEN_SOURCE_TASK_REUSE_20261030.md)优先把RoboCasa柜门和LIBERO单书左右格接入现有MuJoCo源执行器，再用已有示范和MimicGen式生成构建共享四任务数据。已固定来源并静态核对三个真实书本任务，10项CPU检查通过；尚无新任务运行或示范。300条pilot→累计1,300条及200–400段恢复均为候选投入；源域学习独立于目标WBC成功，原[四任务策略](docs/protocols/SHARED_MULTITASK_SOURCE_POLICY_20261029.md)、E1/E2对照和冻结WBC保留。

**S1两次恢复已实际关闭，仍无整机成功示范：** [首次结果与核账](reports/PI0294/design/parallel-science-20261022/GO2_ARX5_S1_RECOVERY01_TWO_ACTUAL_FAILURES_CLOSED_20261030.json)。T1安装1段H40并实际执行0.4秒，保存8对RGB；下一段被历史旋转解码错误误拒。T2在初始化第24步真实触发小腿关节安全停止，未进入任务，该版本已隔离。累计5启动、实际7727/计费7736步、RGB8对；两run/lease均关闭、旧费用原件保留，不重试。T1实际保存状态的正确行解码通过原1051点证书，48项CPU检查通过；冷修未部署，不代表任务成功。T2启动失稳原因未确立；四任务新物理/训练及十天延期未采用，完整[S0–S5序列](docs/protocols/GO2_ARX5_EVIDENCE_CORRECTED_SEQUENCE_20261029.md)和原DEC1保留。

**2026-10-29最新实际：E1@512已真实HF交付固定版本 `f410cd2b`，10个推理文件／12576325049字节；源端前后全流SHA、远端全部成员及PC独立公开回读通过，源进程自然OS0、own source lease一次CLOSED。** [实际交付回执](reports/PI0294/design/parallel-science-20261022/ARX5_E1_QUALIFIED512_PUBLIC_HF_FIXED_REVISION_SOURCE_CLOSED_ACTUAL_20261029.json)。未重训／新增模型调用；Core下载、加载和E1闭环仍未完成。E2完整512、D25／GEN1行为仍未完成。



**2026-10-29最新实际：E2作业1507480已F／Exit1，模型／查询／更新均0；原owner身份分支遗漏，原300秒心跳保护随后终止。445GPU秒／310siteCPU秒已一次结算、own lease关闭并独立核验，GLOBAL `b3537b24`。** [完整原件与冷修复](reports/PI0294/design/parallel-science-20261022/ARX5_E2_1507480_ZERO_MODEL_FAILURE_CLOSED_AND_UNLIMITED_PBS_DIRECT_AUTHORITY_ACTUAL_20261029.json)。原E1@512保留且不重训，E2完整512仍未达成。人类最新已直接取消GPU PBS次数硬上限；精确授权已在原GLOBAL一次采用并独立全文回读，当前GLOBAL `1805b53e`；实际累计PBS41仍逐笔计费记账，次数不再作为阻断。不改其他caps／clock／科学／并发。旧44次授权和全部费用原件保留，不再为单纯多一次PBS重复问GO。

`main`是唯一集成入口。先读[科学状态](CURRENT_SCIENCE_STATUS.md)、[V6详细计划](docs/protocols/GR00T_WBC_LONG_HORIZON_GOAL_V6_20261029.md)、[路线索引](docs/EXPERIMENT_ROADMAP.md)和[AGENTS](AGENTS.md)。历史协议不自动提供当前权限。

## 下一步：两条科学线，一条只读支持线

| 位置 | 当前任务 | 完成标志 |
|---|---|---|
| CoreLab 4080S | 复用已补齐的LOCAL终态，D25原24第一；随后按已就绪E1→E2→GEN1非抢占评价 | 完整行为表、D−C配对及关闭 |
| Vanda | E1@512已通过推理资格，独立交付；E2原始父fresh512＋有界完整恢复点 | 固定端点、真实更新/消费/保存恢复 |
| PC Rerun | 已有Teacher/Student失败解释与保存输出分析 | 可核验只读证据，不阻塞科学运行 |

两站运行不互相SSH/查队列/等心跳；管理端独立连接/终态核账，模型走已授权HF固定revision，不重传已交付D25。

## 当前关键事实

- **E2接续PBS44已在原GLOBAL一次采用并独立全文回读。** [最新实际接续](reports/PI0294/design/parallel-science-20261022/ARX5_E2_ONLY_PBS44_ONCE_ADOPTED_COLD_PROCESS_AND_ACCESS_OBSERVATION_ACTUAL_20261029.json)：原始父fresh512入口已固定，原Trainer接入256／512完整保存、最多两份、原70GB内；有限冷态预检已后台启动，尚无原生PASS或新qsub。最后两次核验同一进程健康，随后完整只读35秒超时、最小进程读取SSH认证拒绝；不推断终止、不重启或重放。E1独立交付的own source-role修复已过CPU检查，尚未上传／到达Core。科学结果未新增。

- **V6已作职责／执行顺序修订。** E1按自身完整证据独立交付，不等E2；D25先完整24，随后在面板边界选已就绪E1、E2、GEN1，GEN1填空档，开始后不按成绩切换。E2原Trainer完整256／512两份恢复保存已接入新cold代码；原70GB内的实际预检和新GPU登记／提交仍未完成。四条历史整机Teacher仅只读分析，不扩Viewer／WBC／数据路线。见[现有V6](docs/protocols/GR00T_WBC_LONG_HORIZON_GOAL_V6_20261029.md)。

- **AMD职责已按人类要求收敛为共享存储和登录基础设施。** [保护边界](docs/operations/2026-10-29_AMD_INFRASTRUCTURE_PROTECTION.md)：不再在AMD启动重负载实验，不改挂载／NFS／SSH／系统服务，不复制或轮换凭据；Core下一次冷态运行须核清全部节点本地运行依赖。历史产物和已封存任务不迁移、不删除；这不是系统隔离已实施的声明。
- **CoreLab已复挂并真实登录／只读核验。** [恢复与断联因果审计](reports/PI0294/design/parallel-science-20261022/CORELAB_NFS_RESTORED_READONLY_CAUSAL_AUDIT_ACTUAL_20261029.json)：AMD NFS于13:51恢复；4080 Home NFS4.2、原key／权限及主机boot不变，SSH服务未重启。上次数据盘2卸载与NFS停止同秒，强依赖支持共享服务故障解释；触发者仍缺管理员日志。此次删除0、服务／挂载／key修改0，不以恢复连接为新作业或旧UNKNOWN重发许可。
- **E1@512可单独交付的实际输入已接通；E2 PBS44已获人类直接批准。** [E1原始10文件／12.58GB及完整14维控制复算](reports/PI0294/design/parallel-science-20261022/ARX5_QUALIFIED_E1_SAVED512_SEPARATE_SOURCE_TRANSFER_AND_CONTROL_FULL14_CPU_ACTUAL_20261029.json)复用完整tensor审计和真实同noise输出，不把失败的E2或旧owner OS1改成成功；HF上传尚未执行。[精确43→44授权](reports/PI0294/design/parallel-science-20261022/ARX5_E2_ONLY_RAW512_PBS44_DIRECT_CAP_AUTHORITY_20261029.json)仅增加1次PBS；其他caps、clock、原matrix／final不变；PBS44已native一次采用，新GPU尚未登记或提交。E1不重训，E2仍需原始父fresh512。


- **Vanda1506193已F／Exit1并完成一次结算、两lease关闭。** [E1真实推理资格与E2中断终态](reports/PI0294/design/parallel-science-20261022/ARX5_E1_SAVED512_REAL_OFFICIAL_OUTPUT_EQUIVALENCE_AND_E2_FORMAL_TRAINING_ACTUAL_20261029.json)：E1复用saved512，25查询／3加载／0新更新；同观测、同实际noise重载输出逐项精确一致，最大差0。E2正式122步／cursor3904后，后台读取共享文件遇到ESTALE退出，原心跳保护随后终止计算；无完整E2 checkpoint，不能伪称可从122恢复。2526GPU秒／1825CPU秒已真实settle，全部pending／unknown为空。后台本已独立运行；新冷修只让显式后继的只读ESTALE进入原有重读路径，未部署、不热改或重投旧作业。E1不重训；used40＋matrix3已占原PBS43，下一GPU须完整合法接续。

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

stop-new仍2026-10-31 20:52:29.813 SGT，hard deadline仍2026-11-01 20:52:29.813 SGT。后继执行以最新实际回执为准；文档发布不自动部署。原WBC/R0/评分/有效失败保持，不重启子代理或暂停的自动化，运行source不热切换。

根入口保持简短，完整证据在reports。历史可由Git/原字节归档恢复；分支退役按独立维护回执，不随计划发布删除refs。
