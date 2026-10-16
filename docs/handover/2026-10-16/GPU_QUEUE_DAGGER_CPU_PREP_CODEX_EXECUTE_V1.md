# Codex CPU PREP HANDOFF | E0排队期间并行准备DAgger

**现役唯一 local-PC Codex 接手。用户明确请求：GPU排队期先完成后续CPU准备，将完整数据—微调—DAgger流程写入GitHub。** 这是任务交接和仓库/本机非物理CPU准备授权方向，不是另一个E0、GPU/PBS、模型推理、科学物理或TF-R3重开许可。**本handoff已发布不代表远端Codex已读取、采纳或运行。**

固定协议（与本handoff同一提交ref读取）：
1. docs/protocols/PI0294_GPU_QUEUE_CPU_PREP_TO_DAGGER_FULL_PIPELINE_V1.md
2. reports/PI0294/design/GPU_QUEUE_CPU_PREP_TO_DAGGER_TASK_CARD_V1.json

真实操作前读取 START_HERE.md、AGENTS.md、统一入口和冻结WBC章程，并刷新#11/#47、owner/lease、当前E0和TF-R3的原生真实状态。使用统一入口及现有 owner；不要创建新协调者，禁止旧runner热改/kill在途任务。

## 一、首步必须对齐现役执行

现有E0已通过人类采用，来源数据v3.0、模型M2048、EEF10、source D0+S1固定。E0 run 20261016T083821Z-localpc-e0；已发布job 670049.hopper-m-02上一个状态快照Q，模型/物理结果未见。**先做一次只读真实qstat和owner/lease检查；只记录变化，不追加qsub、取消或更改该任务。** 原E0采用T0/截止、S1和TRAIN4→DEV8保持，0/4模型任务成功不作为DEV8新门。若E0已RUNNING，先保障它的数据/时间/CPU，prep资源退让。

上一版整机TF-R3曾有实际安全FAIL并封存；最新进展/STOP/owner锁按现场只读证据说话，不以旧GitHub文字复工。冻结WBC、夹爪原物理和任务评分不变。

## 二、立即执行的顺序（GPU无需启动）

Q0 绑定和去重：检查现有E0的五项原准备与已发布脱敏manifest；只在新增需要处补，不重复恢复权重、重新导出v3.0/拟合stats、重做旧CI。建立reports/PI0294/design/gpu-queue-dagger-prep-v1/PREP_INVENTORY.md，写清真实不变量和GPU队列快照日期。

Q1 动作/数据contract：源EEF10和未来target EEF13分开，使用旧processor和stats的小型fixture检查XYZ、rot6d、TCP、gripper、relative exactly once、20Hz/H40/prefix8、BASE absent vs soft zero。新增正常和故意损坏的CPU单元测试；若旧CPU测试已覆盖同契约，引用原实测结果，不重复复制测试。交ACTION_CONTRACT_MATRIX.md和新的实测test收据。

Q2 Source Recovery Oracle：先做一个source D0只读/saved-state/query的统一expert接口。目标是 oracle能从指定实际状态返回合法完整H40及阶段/失败原因；可以INVALID。使用已有合格Teacher函数与S1，只做保存状态/合法离线fixture，不启动新物理，不把synthetic扰动当student-on-policy。交ORACLE_QUERY_CONTRACT.md、ORACLE_FIXTURE_RECEIPT.json；如果离线真实expert query不可用，则交mock和具体缺口，不能宣称recovery PASS。

Q3 DAgger collector：实现纯CPU的动作chunk交易/状态机和数据血缘，包括从同一实际observation生成student和oracle动作、原安全验证、student/expert/none实际执行者、已安装前缀8点、未来完整40点label及valid mask、失败/成功bank分离、privileged truth不进入学生。先fake policy / fake environment / saved-state；不运行真实GR00T、不调新MuJoCo mj_step。测试提前结束、全拒绝、不可恢复、重复clock/chunk、无contact但source时间推进、有效数据漏入DEV等负例。交DAGGER_RECORD_SCHEMA.json和DAGGER_COLLECTOR_CPU_FIXTURE.md。

Q4 FlowDAgger适配审计：只读微软固定SHA 3426f246510fd67a9e39c3c08bd5b02d586ad688：
- flowdagger_gr00t/groot_inversion.py
- flowdagger_gr00t/groot_expert_inversion.py
- flowdagger_gr00t/train_groot_flowdagger.py
- shared/intervention_handler.py
确认它的LIBERO7D/action_horizon16/padded model tensor与我们v3.0 EEF10/H40的差异；官方GR00T trainer默认expert chunk概率1.0并不证明on-policy student访问。不得直接把LIBERO encoder和gripper方向替换现有processor。用mock小流头验证API形状/近似逆/正向结构，但绝不把mock通过写成M2048真实inversion通过。交FLOWDAGGER_PORTABILITY_AUDIT.md、MOCK_INVERSION_ROUNDTRIP.json或NOT_RUN、RUN_WITH_REAL_M2048_LATER.md。

Q5 数据/后续研究：固定source BC、student-visited oracle correction、expert takeover实际执行、目标成功、failure五类来源；拟定未来train/dev family隔离、任务阶段分层样本、绝对/相对动作、模型输入不含privileged truth。把普通同量专家示范BC、Action DAgger、FlowDAgger及source-only/0-5-20 target demo做成匹配的未来对照，不能借现有E0授权运行。交DAGGER_DATASET_AND_SPLIT_SPEC.md、E0_TO_DAGGER_DECISION.md。

Q6 聚合：尽可能在旧本地合格环境做定向CPU CI、source code/测试/脚本脱敏提交GitHub固定提交和回读；最终PREP_FINAL_REPORT.md必须列各阶段actual done / fixture pass / NOT_RUN、耗用的本地CPU/新增数据、下一个唯一科学阻断和E0实际已发布状态。同步#11/#47一次，不重传大型旧原件；其他项目不重启、不删除。

**只在当前有效本机资源规则下运行CPU mock/saved-state测试**，建议不超过4并发核、8 core-hours、2 GiB新增文件，真实更小限制取小；本文件不增加站点资源配额。GPU、PBS、新模型推理、优化器更新、真MuJoCo积分/物理采集、额外Teacher rollout、大下载、WBC/安全/评分变更、长期后台监控/cron全部为0。

## 三、未来门，不要越级

- E0仍排队：Q0–Q6 CPU准备可以形成READY，但学生真实状态、真实Flow反演和DAgger结果都是NOT_RUN。
- E0真的完成：阅读真实TRAIN4/DEV8结果，按未见任务/安全/接口错误选择下一项，不反向更改E0或源分集。
- Source Recovery Oracle在学生实际访问状态可输出安全有效H40：新的源DAgger研究才可单独申请（包括神经计算/物理）并开始。
- 完整Go2＋ARX5 Teacher真正做成安全T1/T2且有恢复能力：才能研究target DAgger；source教师成功不能自动当target教师。
- FlowDAgger需要用真实M2048同观察、同processor和数值设置完成expert-action→model-space inverse→forward→real action roundtrip，并与Action DAgger同预算比较；本轮只做结构准备。
- BASE3以后需要真实非零target动作标签，不能用源EEF10零padding造移动能力。

## 四、不要再重复准备/等待

已经有人类采用的E0和当前job保持单一生命周期，不因本CPU目标重新提交PBS或重设clock。Codex可以在仓库一份独立可读prep目录内递进实现和测试，无须每个常规CPU小修再问GO；如果遇资源/原始代码语义不符，仅合并报告一次真实科学/算力缺口。不能用本文件默许新科学条件。

结束标签：PREP_CPU_READY_NOT_DAGGER_EXECUTED / PREP_CPU_PARTIAL_WITH_EXACT_BLOCKER。selected/deployment仍NONE，直到独立真实评价决策改变。所有实测状态必须来自收据，而非文档措辞。

**Codex直接行动**：读取同一固定release的协议、机器卡，核对真实owner/队列，按Q0→Q6交付CPU预备包及测试，在GitHub #11/#47发布脱敏结果。E0一旦运行，避让已授权资源并保持独立。不要只生成另一份计划，也不要把未来DAgger研究写成已经发生。
