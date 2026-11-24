# Codex 启动 prompt：SLA 多数据集后训练与本体闭环 Goal V2

将下文作为有项目现场权限的 Codex 长程任务。启动时以发布者提供的本文件固定提交作为 `PLAN_SHA`，先读取完整协议/JSON，不以浮动main或报告head代替runtime。本文本本身没有提交服务器作业。

---

你在 GR00T+WBC 项目中执行 **SLA 多数据集后训练与本体闭环 Goal V2**。仓库为 `Leo-6-maker/go2-arx5-hierarchical-vla`。用户要求自主推进、充分利用实际获配Corelab/VANDA资源、后台接续、不实时监控、Eval独立并行。目标是完成可恢复的多源后训练、实际Go2+ARX5学生闭环和来源清楚的能力结论，不能停在更多计划、CPU测试或排队截图。

## 先读与权限

在传入的固定 `PLAN_SHA` 下依次读取：

1. `docs/protocols/SLA_MULTISOURCE_CLOSED_LOOP_GOAL_V2_20261120.md`
2. `docs/protocols/SLA_MULTISOURCE_CLOSED_LOOP_GOAL_V2_20261120.json`
3. `reports/WM02/sla-multisource-goal-v2-20261120/AUDIT_AND_PLAN.md`
4. `AGENTS.md`、`docs/REPOSITORY_WORKFLOW.md` 和当前实际执行manifest。

最新用户授权与本Goal覆盖旧文档中的旧下一步优先级；科学历史不改写。你可自主修复项目用户空间的入口、数据读取、保存恢复、依赖、资源日志、性能与本Goal登记的时序实验，不为每项常规修复再请求GO。不绕过站点账户/凭证/资源锁，不改基础设施，不抢占他人GPU，不删除用户历史文件。实际交互登录确实过期且你无法合法续期时，明确交接该登录事项，并继续其他可执行工作。

## 第一项：核对当前现场，不重跑已经完成的工作

用户2026-11-20 14:08 SGT报告：MIX1000已封存，TGT4976/5000，ADAPT953/5000；ADAPT已登记到2k，2k后尚未qsub；D2已ready；部署V2资源检查退出、生成0；新ADAPT学生请求0。

公开PR418 `0ef0abfcc39096666ad8bed6a5842f3ba25cd2b7`观察时间更早，为12:49:41，只记录MIX875/TGT4629/ADAPT844。读最新真实scheduler、RESULT/COMPLETE、模型/Adam/RNG、账本、owner/lease、STOP和allocation，对账后补公开摘要。MIX869恢复、D2读取、TGT2k Eval、ADAPT500 Eval已经有完成证据，不再重做。任何RUNNING或UNKNOWN任务先核对真实效果，绝不重投。

## 固定执行源码

- 现役ADAPT train/eval：`1432c8dc88c57adf39e8fa363f5438c5e63aa8fd`。
- 2k→5k handoff与D2 selector：`f9c8457bbe47c35c54d173abfc619500320b1ad2`，实现已经存在。
- Core部署V2：`71881dcf10d2887b9dbfb31e012b0d1c1ad55c33`。
- MIX handoff：`2bb3ba25c39868bdc351fb289c757964d38ddfa9`；原worker `31076fc7a64302b0705ae4d7e526c4341209a100`。
- PR419吞吐工程：`e98f58d51e599384a5eee7652a7e600a7dc77ce0`。

计划分支和报告head不包含完整runtime，分别用真实固定来源建隔离worktree，保留并发提交和本地未提交工作，不对active capsule拉取/checkout/hotpatch。

## 立即并行推进四条线

### A. 主训练与后台队列

保持现有TGT至5k和ADAPT健康任务。复用现有PBS/user-systemd与Journal/onceIO；补MIX1001起的有限后继、ADAPT2001–5000的12个250步段和独立5k Eval。MIX仍至20k，TGT只至5k，ADAPT只至5k。不在每段重置Adam/RNG/LR，不因为低闭环分数终止健康主线。

实际PBS使用afterany加完整seal门。允许0或75完整保存前缀的训练接续；正式Eval要求精确端点与成功seal。若名义2k卡只完成到1970，先以唯一后继补齐2000，不能让2250后继直接跨过课程边界。提交只有真实JobID/依赖/状态独立回读才记REGISTERED；脚本准备完成不是QUEUED。未知qsub效果先查，不重试写操作。认证失败不主动杀现役或取消合法队列，但后继自身若qstat认证失败，有限只读重试后仍应记BLOCKED_AUTH，不能跳过身份核验。

精确2k冷态、独立2k Eval已登记时只决定一次D2：ready则切，不ready记录原因并用D1 fallback；不等Eval分数。D2为26条/2031窗，目标配额12新+52旧，源32+32。D1/C20/R48/LIBERO统计、codec、LR、Adam和原worker不变。若已过边界读取既定决定，不回滚或晚切。现役binding不允许顺便换PR419 source。

### B. 部署资源与真实学生闭环

先在新的项目wrapper补资源拒绝前的逐项快照，保留原阈值。记录GPU总/空闲/预留fraction/余量、host MemAvailable、cgroup与private pressure门、owner和逐项pass/fail。检查预留+余量是否在该GPU上根本不可满足；不要反复等待数学上不可能的条件，也不要降低门。

Core GPU1不满足时，绑定MIX自然冷态后的实际GPU0窗口；必要时做有资格的同配方MIX冷迁A40，不能抢占健康段或假设已获额外卡。部署V2仍为原16+独立恢复16、最多32生成，核对既有技术储备后使用，旧严格跨卡FAIL保留。

部署身份、资源及原安全链成立就派发原ADAPT0 DEV24前6，随后固定人口余下条件。离线联合0/24不是阻止合法native的附加门。原ADAPT A0/2k/5k各24+5k确认24共96，原SCALE TGT5k/MIX5k/MIX20k各24+MIX20k确认24另96，分账不混用。离线Eval只依各自seal和资源，不等Core部署；native按各端点自己的包资格，A0故障不全局锁住其他已资格端点。每个24为8 HOLD/16 PLACE，固定顺序和评分，不根据成绩挑端点，不重跑有效科学失败。N2未合格用单world，STOP后该world积分0。记录原始H40、门、提交ACK、P10实际消费、正确接触/抓持/抬升/移动保持/完整放置的同query漏斗。

### C. 已有预测的零生成诊断与数据

读取ADAPT0/500已有逐行JSON/NPZ，解释阶段H40 84/88→57/88、及时闭合10/24→2/24、联合0/24。计算有符号50ms首cross差，分过早/及时/过晚/缺失；交叉H40/接缝/后缀拒绝；检查非到期错误闭合、HOLD保持和释放；专家标签在同前命令/时钟/门下做可行性检查。不要从汇总断言所有24都被门拒绝，或22个都提前/延后。按独立episode计数。

复用RoboDojo/LIBERO源域预测补逐任务/通道、动态/低变化、persistence/zero-command对照。源预测误差不是源任务闭环成功。MIX5k/TGT5k才是共同目标暴露与LR前缀的额外源数据+计算比较；ADAPT与TGT不作纯源数据因果对照。

新TRAIN采集首批最多12次（4 HOLD、6 PLACE、2具备生产者的恢复），计入旧240尝试并扣已发生；学生就绪块先行。重复同格同生产者两次同首因后必须改变有证据的路径/几何或生产者。新D3不热加进当前5k。保持原命令/消费/标签层次、mask和split，失败片段不冒充完整成功，RoboDojo保持SOURCE_NATIVE。

### D. GPU吞吐工程

在真实空闲A40运行PR419的26更新（三路10/8/8），第8步对第8步比较140叶，serial的额外2步只用于profile。计3328访问、104训练内H40。若补每路16+16独立输出，先实现第8步完整工程checkpoint保存，最多96输出；当前代码只有hash，不把它冒充独立恢复。

再按协议最多4配置各16更新：1GPU mb1/2/4、同节点2GPU mb2；同父、同样本/RNG、全局64/32/32及4辅助。两卡实际实现后才申请运行，每卡32/16/16和2辅助，局部累积后全局归约、一次clip/Adam。两卡强缩放对照是同轮1GPU-mb2且pipeline flags相同；性能输出面板在首次GPU前按协议冻结，不能按结果挑窗口。记录正常全步吞吐、p50/p90、显存与GPU小时。新工程共90更新，不偷用或重置旧120池；工程权重永不作为正式父端点。未达兼容门保留FAIL，不能因旧mb4 27/32就声称能力更差，也不能追认其可热替换。

## 条件性科学分支：CLOCK500

科学触发是精确ADAPT2k固定24到期窗联合仍0/24，且逐行标签/时钟/前命令/codec/专家可行性审计通过。联合非0才记NOT_TRIGGERED；诊断未完或有语义问题记相应BLOCKED，科学触发后资源/工件/凭证缺失也记BLOCKED，不能改称不触发。就绪后执行固定ADAPT2k父的CLOCK-A/B各最多500，D2/bank/Adam/RNG/LR2001–2500/执行后端相同。

A原loss；B只改时间权重：严格复用原逐窗分类优先级P10_DUE_CLOSE→APPROACH→RELEASE_OR_RECOVERY→NOT_DUE_CLOSE→OTHER。只给due首cross±2点（截到0..39）权重4、NOT_DUE_CLOSE前10点权重4，其余1。保留原 `e=(predicted_width_m-label_width_m)/0.040`，B为每样本 `sum(weight*e**2)/sum(weight)`、再四辅助取原均值；width系数仍1。40点支持与A完全相同，不用consumed_P10 mask删后30点；phase只作监督元数据不进模型输入。全1权重必须退化原loss及梯度。保留全部动态max损失和十步完整采样，不自动搜参数或热改当前5k。

若主ADAPT2500与A逐项完全一致可复用A真实状态/账本，不能因step相同借用；省下预算不得开新候选。两臂各424离线/独立恢复，各16包复验。分支首次训练或预测前冻结新A/B配对DEV各24及固定B新确认24，共72；场景、reset、噪声、机器人/WBC/backend/评分匹配，A/B先后顺序预先固定。不覆盖原96或旧TEST，不按DEV挑确认模型。原ADAPT96仍继续，不等待该分支。详细新增上界以JSON为准：≤1090更新、139520访问、17306生成、74原生请求含条件接口、3302522tick；其中科学修复1000更新和72原生均为有条件上界，未触发或已复用就不消耗。

## 验收与停止

分别报告PIPELINE_COMPLETE及逐谱系HOLD_REPRODUCED_ADAPT/CLOCK、PLACE_REPRODUCED_ADAPT/CLOCK。各自固定确认人口中HOLD至少4/8成功、F1/F2各至少2且覆盖至少2布局；原80mm移动、20mm抬升、1s保持和正确目标/安全评分保持。PLACE各自至少2/16且左右各1。两组确认不池化、不事后只挑过门组；CLOCK未触发相应指标为NOT_APPLICABLE。不达标就报告有效科学失败，不放宽评分、不重试选成功、不无限延长。

T0封存campaign_deadline=T0+14天，每个wrapper激活前核验，实际截止取site/campaign较早者并预留保存余量。到期由既有站点机制触发一次STOP/finalizer，核实身份后取消未启动自有后继，在跑段优雅封存。最终收尾汇总全部G0–G8及CLOCK决定/实际终态；STOP/截止/无法恢复阻断也能触发，不只等success。缺必要科学范围时PIPELINE_COMPLETE=false，区分科学失败、技术阻断、外部资源/认证阻断。不要把0 native、只有排队或CPU验证叫完成本体适配。

## 本次Codex应实际交付

在能访问现场的执行会话中，先完成可完成的工程与真实后台注册，再回读JobID/unit、封存身份、下一依赖和停止入口。挂好有限链后不需要保持对话实时盯训练；由站点任务执行。使用原发布器按登记、checkpoint、Eval/native终态和阻断事件更新普通执行PR；GitHub失败不阻塞GPU健康运行。首份回报必须区分已完成、实际已排队、仅准备、缺凭证/资源四类，并给出真实新增计算分母。不要只提交另一个计划后就宣布Goal已执行。

---

配套：[完整协议](../../protocols/SLA_MULTISOURCE_CLOSED_LOOP_GOAL_V2_20261120.md) · [机读规范](../../protocols/SLA_MULTISOURCE_CLOSED_LOOP_GOAL_V2_20261120.json)。
