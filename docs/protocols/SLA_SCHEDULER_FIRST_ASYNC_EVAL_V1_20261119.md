# SLA 长程后台排队接续与并行 Eval · Scheduler-First V1

日期：2026-11-19（Asia/Singapore）。**后继执行设计，不是已提交服务器任务的回执。** 承接用户最新决定：“让 Codex 将长程接续训练任务挂载到 Corelab 和 VANDA 的后台队列，不必实时监控训练进度；Eval 可以并行”。

[主研究计划](SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1_20261119.md) · [机读任务/原预算](SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1_20261119.json) · [Codex执行交接](../handover/2026-11-19/START_SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1.md)

## 0. 目标与改变

**以站点原生后台机制替代 Codex 实时训练监督。** Codex 只需要部署、回读、交付一个真实注册的有界依赖图，站点调度器/用户级服务执行分段训练、保存/验收以及释放后继。Codex SSH/对话退出，任务仍按已批准的有限时段执行。不另建无限存活的轮询agent、每6小时LLM进度查询或依赖GitHub发布的训练继续开关。

Train与Eval并行：一个正式checkpoint seal触发两条互不等待的后继边——训练从s+1继续、评价读取s的不可变复制品。Eval失败只封存该Eval的失败，不阻塞健康训练；只有证实模型/数据/统计/梯度语义被污染，才暂停受影响的训练谱系。Native Eval仍受独占world约束；“Eval可以并行”不是同一张GPU或同一个world强行并发。

**不把提交计划当作真实挂载。** 只有实际 Core 用户unit状态、VANDA PBS JobID/状态、后台日志/关闭证明、checkpoint链全部读回后才称QUEUED/ARMED。本站点未来额度若未发放，不可预借无限时段；可以生成待用的脚本/任务图并标记AWAITING_ALLOCATION。

## 1. 固定事实、当前服务与不能改写之处

- Corelab已真实使用用户级systemd托管训练及after500独立守候服务，当前工作流可从非整齐的完整冷态端点恢复。旧after500服务的名字为 `wm02-sla-scale20k-mix-after500-handoff-20261119-01.service`；其原first500观察窗到2026-11-19 23:49:33 SGT。后继只能重新检查现在实际终态和新的有效时间，不自动延长旧窗。
- VANDA已使用PBS `qsub`，TGT原500步端点已封存，后继`1593003.stdct-mgmt-02`在已发布回读中由H释放成Q，原单卡作业时限06:00:00。是否支持相应`afterok`依赖语法、未来额外A40或其他项目配额，**须在站点实际读回或受控资格验证**，不能由通用PBS知识代替。
- Core原生world的前六条TRAIN采集已有独立用户级守候器；它的原safe-start窗到2026-11-19 23:39:04 SGT，原段截止11月20日00:12:04。窗口此时可能已经失效：必须读取owner/lease/RESULT/当前PID，不复制旧启动或认为“之前有守候”就表示现在仍活动。
- 当前固定SCALE合同 `926226914a0c08ee1ebb143e0edea1434728107b`，D4/D5/J合同 `cf9d1acbf72e96b1bd158322ff8cbaf8fe28fabb`。后继scheduler合同只决定任务排队、资源和产物交接；不改变既有数据、权重、Adam、训练总步数、H40/P10、WBC、评分或硬安全门。

证据：[Core守候实绩](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/reports/WM02/sla-scale20k-20261119/CORE_MIX_AFTER500_DETACHED_WATCH_ACTUAL.md) · [Core冷态接续](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/reports/WM02/sla-scale20k-20261119/SCALE_CORE_COLD_CONTINUITY_ACTUAL.md) · [VANDA PBS提交回执](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/93b3165f785a4c14a27c432f18b454c52571178a/reports/WM02/sla-scale20k-20261119/VANDA_TGT500_AND_NEXT_PBS_ACTUAL.md) · [首6原守候](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/39d8f6e341d2c974aa2baed31588b115e0c2c463/reports/WM02/sla-scale20k-20261119/TARGET_ADAPTATION_FIRST6_DETACHED_WAIT_ACTUAL.md)。

## 2. Task DAG（每条边都是一次完成事件，不要求轮询）

```text
Core MIX20K:
  predecessor active / previously queued
      -> [completed segment + verified full checkpoint s]
          +--> train(s+1 .. next bounded endpoint) -> checkpoint -> ...
          +--> eval-offline(s) [independent GPU job, never waits for train(s+1) terminal]
          +--> optional native-eval(s) [world exclusive queue]

VANDA TGT5K:
  existing PBS job / completed full checkpoint s
      -> [native dependency or verified completion event]
          +--> queued TGT next finite PBS segment
          +--> queued TGT eval(s) [separate A40 allocation if available]

ADAPT5K (new lineage):
  D1 qualified + MIX formal sealed parent + actual A40 allocation
      -> ADAPT[0..500] -> seal500 -> ADAPT[501..2000] -> seal2000
          +--> eval(0/500/2000) independently
          +--> once-only D2 cold curriculum decision -> ADAPT[2001..5000]
      -> seal5000 -> eval+confirm
```

这里的“train(s+1 ..)”是绝对global step或明确phase step，实际segment可在预算或站点walltime所允许的任意完整s结束。示意图中的500/2k为科学端点而非自动获得的6小时许可；依赖图不自动放宽任何时间。新ADAPT必须先有至少1条真实准入的新独立TRAIN episode、step-zero真实梯度/恢复资格和实际A40 allocation，原MIX/TGT不等待它。

**分支规则：**TrainSuccessSeal(s)同时允许TrainNext(s+1)与Eval(s)注册；它们不相互等待。只有Eval确需独占已有GPU、无额外卡时，Eval保持合法排队；不可为满足“并行”而抢占正在使用GPU0的训练。一个端点只能被正式评价一次，跨站转移也沿同一EvalID。

## 3. Corelab：用户级后台运行，不依赖Codex会话

Codex先只读发现真实host、`systemctl --user`支持、linger、原train service、after500 watcher、checkpoint和owner；识别旧服务RUNNING/STOPPED/UNKNOWN与其已有接续者，**不创建第二个同谱系后继**。检查已经存在的项目用户级服务/入口，优先对原有after500接缝做最小后继扩展，不替换活跃capsule源码、不新建全功能编排平台。

推荐两种受真实版本验证的方式，选择一种并封存：

1. **原生success依赖链（首选）**：用户级service每次运行一个有界训练段，完成保存、SHA和ledger校验后退出成功；已验证支持的`OnSuccess=`、受管`systemd`触发或项目现有等效事件入口再调度下一个一次性段。失败状态不能触发下一训练；启动下一单元前必须检查已存在owner/lease/队列身份。`systemd --user`托管确保SSH断开后持续。
2. **单一受管本地主控制器（兼容回退）**：如果当前用户systemd版本或现有站点限制不支持success依赖，则在一个用户级服务里顺序等待其**自己**的子进程终态，并在checkpoint seal后派发下一段；只观察本机子进程事件或本机受管单元终态，绝不忙等全训练step。控制器本身无GPU训练职责，失败只关闭对应后继。避免与原after500 watcher同时成为派发者。

每段独立的immutable spec标明：前驱runID/step和checkpoint SHA、明确source/runtime SHA、下一段绝对终点、任务访问银行/RNG/Adam、训练GPU、合法CPU集合、真实deadline、保存提前余量、唯一successor job identity、停止入口及两个评价消费边。按实测保存p95与GPU p90步耗决定段长，不预设Core可持续无限运行，也不把已经过期的时段改写成新时段。

只有前驱`CLOSED`、PID/cgroup清空、checkpoint 140个预期状态数组和完整update ledger/游标/同批次hash检验通过，才从s+1派发；非整齐s可继续。后继进程一律从不可变完整checkpoint恢复，不从活动权重目录或仅有trainer stdout的路径恢复。

## 4. VANDA：PBS队列作业及后继依赖

先对真实队列、账户、卡型、作业walltime、目录、token可用性和PBS依赖特性进行**一次受控核查**。已运行的TGT `1593003.stdct-mgmt-02`依原身份完成，若当前已终态，则读取新真实JobID/terminal；不按旧正文重新qsub。

PBS若支持并获站点许可，在**实际已取得的有限资源范围内**预提交一条有界链，下一卡依赖前一卡成功结束（例如经站点实际确认的`depend=afterok:<JobID>`）。预登记阶段的训练脚本使用原checkpoint和spec，进入GPU前重新核验前驱seal与完整状态。单段失败/不完整端点自动阻断其全部Train后继；保留PBS H/Q/R/F作业ID和原费用，不能简单重投。

若该PBS版本不支持依赖或依赖脚本不允许项目执行，则采用站点允许的**终态事件后提交一次**（例如已有合法batch epilogue/项目用户态onceIO），不做远端每分钟`qstat`的常驻Codex监控。仅当该事件能凭已封存seal、持久提交回执、有效新授权且安全去重时，才允许提交下卡；若无法满足，交付一个已准备、待有限授权的stage任务包并明确BLOCKED，不能谎称无人值守多日已排好。

PBS每个训练卡都是有限结束的计划段，不让一个6小时作业去追5k/20k而超时，亦不自动为所有未来时间段取得配额。段长由真实p90 update、保存/关闭p95、实际walltime与安全余量计算。PBS排队/资源限额不改变训练global step或学习率进度。TGT优先完成原5k；ADAPT与离线Eval仅在额外真实获配的A40、或TGT合法结束后，使用其自己的独立JobID/依赖图。

将TRAIN任务图、PBS jobID、segment/phase、afterok边、输出/标准错误、登录token和缺失资格状态写入**项目用户可读的manifest（不含任何凭据内容）**。凭证过期只阻塞需要新认证的派发，不杀掉合法已经入队的作业，也不尝试绕过认证。

## 5. 并行Eval的准确含义及隔离

**Eval注册时机：**正式模型和数据/统计/Processor/decoder输入契约完整封存并原子发布后；不是读活动checkpoint和猜训练日志step。支持同一次训练segment结尾同时释放TrainNext与Eval。旧TGT500已做288+48，直接复用，不因跨站再做一次。

- Offline Eval：各已封存端点沿原协议执行每域96预测+每域16独立恢复，或ADAPT自己的固定面板。与TrainNext用不同运行身份、读-only模型副本、日志与目标目录；可运行于实际空闲GPU/A40。源数据DEV/TEST使用既有固定split和资格，不开启TEST作为采样或调参信号。
- 跨站Eval：Core封存后独立执行SHA校验的传输/导出作业，VANDA取得**字节级封存的**模型、Processor、统计、模型配置与本端允许的数据后再释放eval。传输失败仅阻塞该Eval，不影响Core继续训练；传输和Eval不可将失败端点重命名为成功端点。
- Native Eval：可与异站GPU训练并行，但Core现有正式world只允许一个native owner。独立挂载world队列，与目标采集/J历史面板按原固定顺序轮转。N2未资格前不得同时启动两个正式world；STOP后该world积分0。若CoreGPU1完整模型不能装载，则合法排队或在自然冷态窗口使用CoreGPU0，不做未经验证的实时跨站推理。
- 评价结果不回写活动训练Adam/RNG、数据成员、原固定LR、选择器和模型。ADAPT的唯一D2课程切换只按预冻结数据准入，在phase2k冷态边界发生，不等待Eval结果。
- 一个`eval_key=(lineage,checkpoint_sha,panel_sha,recipe_sha,noise_sha)`只允许一个正式Claim。RUNNING/UNKNOWN不能在另一站再派同一key。失败与所有实际生成分母原样保留。

**并行不是增加科学预算。** 本文件不增加既有MIX/TGT/ADAPT的任何更新、H40生成、96 native科学面板、接口或tick；如额外GPU不可用，Eval可以排队稍后执行，Train本身不等待。

## 6. 任务状态机、回执、异常与停止控制

每条工作线至少有以下不可变状态转换，复用现有Journal/onceIO、lease和执行目录：

`PLANNED → VERIFIED_READY → REGISTERED(Q/H或active service) → RUNNING → SEALED_SUCCESS → TERMINAL`

失败/外部阻断分支为`FAILED`、`BLOCKED_RESOURCE`、`BLOCKED_AUTH`、`UNKNOWN_NEEDS_RECONCILIATION`、`STOPPED_BY_USER`。训练SUCCESS必须有完整checkpoint、update ledger、状态数组与实样哈希；PBS退出码0或systemd `active`本身都不是训练成功的充分证据。

- 使用`job_key=(lineage,source_sha,parent_checkpoint_sha,segment_start,segment_end,site)`防重；真实JobID/service unit和提交结果持久化。任何写操作响应未知时只读查实际效果，不重投。
- **失败默认fail closed，不盲重试：**前驱缺少完整seal、非有限梯度、数据跨split、错参路由、输入语义不符、quota/截止、权限失效 → 受影响的后继保持未释放；同机其他独立任务线不受其失败影响。
- **允许普通工程修复：**修复入口/依赖/序列化/冷态恢复后，在新的有限段明确运行，不修改历史失败或旧有效结果。正常网络瞬断以只读恢复核验，不重跑有效科学评价。
- 在项目自身用户空间支持`STOP.json`/等价stop token，**禁止未来新段注册**。停止正在运行的训练须走原进程优雅封存流程；不要把取消排队的后继误实现为直接杀当前健康进程。PBS取消仅项目自有job并核验，不碰其他人的作业。
- 不删除历史checkpoint/模型/数据；用户负责清理。旧有效allocation、真实站点quota、所有权、安全与WBC不因排队自动绕过。

## 7. 无实时监控的最小观察与通知

用户已明确不需要实时监控训练step。撤销上一版“至少每6小时主动状态摘要”的未来工作要求，不创建ChatGPT每6小时任务或SSH守候Codex。服务器原生心跳/日志、PBS accounting和systemd状态仍保留，用于故障/恢复及最终审计；这不等于人持续看训练。

**仅在事件上发布简洁ACTUAL：**新链实际REGISTERED（JobID/服务身份）、正式checkpoint封存、独立Eval终态、技术失败/权限阻断、整条谱系完成。GitHub发布由已有合法publisher在**非训练控制关键路径**执行；GitHub不可用不应使健康GPU训练停止。若没有可用自动publisher，站点先留私有持久回执，下一次Codex接手时补发，不虚构已经推送。

## 8. Codex一次性交付清单与验收

本协议授权Codex在原项目既有用户权限和新实际有限allocation内，**优先真正部署后台接续链而不是先重写训练器**；使用最小项目用户态代码复用已有runner/冷态恢复，不变更系统服务/管理员配置。以下均需实际回执：

1. **只读live audit**：Core各unit、after500和first6守候真实终态、训练目录/完整checkpoint、owner；VANDA各PBS job与真实依赖功能；GPU可用量、quota、当前完整端点、数据身份和截止。特别核对原11月19日23:39/23:49过期窗，旧状态不可当新授权。
2. **Core链至少一条真实登记**：读回后台unit及独立断开SSH后的原生进程/日志；有next-segment唯一路径、checkpoint gate和STOP入口。如果现场现有自动after500已经正确运行，就只确认它，不创建第二套。
3. **VANDA链至少一条真实排队**：回读PBS JobID、Q/H/R、队列/依赖边、walltime、前驱完整seal。能安全预排多个合法后继就按固定终点排；若PBS/认证/配额不能，准确报告已登记到哪段和其余BLOCKED，不用模拟JobID。
4. **至少一项可并行Eval注册或真实完成**：指向已封存端点，独立eval_key、GPU申请、输入SHA；已经完成的TGT500只挂既有结果，不重复。若仅一GPU可用，则展示Eval处于单独QUEUE而Train继续，不伪称物理同时运行。
5. **冷态恢复与失败门一次可证实**：已核验非里程碑s恢复链，或既有同等回执＋本次新调度入口所受影响的最小验证；UNKNOWN写入守护、STOP阻断下一段、checkpoint不完整不释放后继。
6. 发布一个`SCHEDULER_FIRST_DEPLOYMENT_ACTUAL.json/md`，字段：站点/当时观察时间、有限许可、原始与新作业、真实parent SHA/step、端点身份、PBS dependency原样字符串及是否验证、systemd unit、Eval独立claim、GPU/world实际情况、已提交/仅准备/阻断项目、实际新增训练/生成/native分母。只读和预检均须标0，不能把qsub视为训练完成。

**只有这里的部署回执才能说明“已挂载后台任务”。** 本文提交GitHub不自动SSH登录、qsub、systemd-run，也没有更改服务器。符合本站点资源条件的部分优先落实，其他阻塞项并列说明，不让一张缺卡停掉另一张健康训练。

## 9. 与主计划的关系

本文件**覆盖主计划第12节“每6小时发布运行摘要”及旧Codex prompt同款语句**，改为纯事件驱动回执；不覆盖原训练预算、数据采集上界和每条任务的科学终点。主计划中的近实时状态示例均只作为2026-11-19已发布历史快照。PR417/418运行合同仍按其固定SHA，不能仅因PR416更新就修改运行中胶囊。