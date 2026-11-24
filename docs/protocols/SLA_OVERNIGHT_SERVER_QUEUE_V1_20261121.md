# SLA 夜间服务器后台接续 · 2026-11-21 · 不依赖用户电脑

**执行交接，不是已登入 Corelab/VANDA 后的新作业回执。** 用户明确要求在关闭本地电脑前，将继续推进的工作交给两站已获权限的后台系统。引用 [Goal V2](SLA_MULTISOURCE_CLOSED_LOOP_GOAL_V2_20261120.md) 和 [Scheduler-First](SLA_SCHEDULER_FIRST_ASYNC_EVAL_V1_20261119.md) 的预算、停止门和发布规则；本次只更新文档，不提交远端作业。

## 0. 事实、时间与证据层级

最新**用户现场报告**时间为 **2026-11-21 01:59 SGT**；本仓库截至本版读到的 PR418 最新发布快照约为 **2026-11-21 00:39 SGT**，两者不可混作同一现场回执。

| 谱系 | 用户 01:59 报告 | 已封存与新任务边界 |
|---|---|---|
| Core MIX20K | 已记账 **1,908**，Go2/RoboDojo/LIBERO **61,056/91,584/91,584** 次访问，热态约35.5秒/步 | 最新公开HF **1,768**；现役训练继续，严格到2,000才可释放MIX2K固定Eval，早期有界前缀合法完整封存也需保留 |
| VANDA ADAPT5K | 已记账 **1,644**，三域 **105,216/52,608/52,608** 次访问，热态约46.8秒/步 | 完整1,500已公开HF；`1599363` 至1,750正在运行，`1599364`到2K及独立`1599365`已注册；`1599367–1599379`至5K及Eval已注册 |
| VANDA TGT5K | 5,000训练和三域独立Eval完整 | 原SCALE固定DEV首六需要Core严格RGB16，资源不足时保持BLOCKED |
| Core ADAPT0 原生 | 原F1 **6/6**及F2 **5/6**为封存科学失败，合计**11个终态0/11成功**；F2蓝右第六条已观测**28,201 tick**仍进行中 | 最后一个RUNNING不允许重投或伪造终态；F2前5共104,749 tick/307次生成，F1前6共132,555 tick/396次生成，均保留原始分母 |
| D2 | Go2 **26条、2,031窗**，含两条独立专家HOLD成功、139窗已准入 | 仅在精确ADAPT2K冷态边界做一次D2/D1原规定决定，不能按1,644提前切 |
| G8 HOLD4 | 现有Core observer已登记，实际四条TRAIN原生仍0 | 必须先核验现役学生/world/资源所有权，再在原尝试预算内派发；不覆盖DEV/TEST |
| CLOCK500 | 依用户报告原条件队列重建，GPU科学更新0 | 只依精确ADAPT2K固定24窗联合指标及完整语义审计触发，不额外派新候选 |

公开身份入口：[PR418实绩](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/418) · [MIX1768与ADAPT端点交接](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/a37151b0607ba93698590f8227befa94e31a6d09/reports/WM02/sla-multisource-goal-v2-20261120/CORE_MIX1768_AND_ADAPT_ENDPOINT_HANDOFF_ACTUAL.md) · [ADAPT1500与PBS恢复](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/107b3414cc873713ae56b089cbe10ac5cbb16d9a/reports/WM02/sla-multisource-goal-v2-20261120/ADAPT1500_PUBLIC_HF_AND_PBS_RECOVERY_ACTUAL.md)。与此用户报告不同的更晚现场结果优先，但必须有独立真实签名／账本回执。

## 1. 不依赖本地电脑的基本准则

已经注册在站点的 user-level systemd / PBS 作业不依赖用户打开浏览器、保持SSH或笔记本通电。**Codex不需要每分钟SSH轮询；远端原生服务/任务依赖和持久日志接管。**

本文件发布在GitHub本身不会创建站点作业，也不会自动触发Codex会话。现场有权限的Codex须通过原合法访问方式对账，**只有PBS JobID、用户级systemd实际unit/PID/cgroup/lease及独立连接回读，才记BACKEND_REGISTERED**。禁止把仅写了脚本、GitHub PR或HF包称为已经运行。无需新的ChatGPT定时监控。

## 2. 就地保护：不允许重派的现役

- **F2第六条**：先只读核对真实unit/PID/world锁/原case进度，若已结束必须封存原完整H40、ACK、P10确认消费、2ms评分及首接触、末原子部分消费。若仍运行，让原服务继续至合法终态或原deadline/STOP；原F2前5和F1前6绝不重测。原生科学成功分母只有CLOSED任务；未封存第六条保持RUNNING，不宣称0/12或1/12。不得改变WBC、障碍碰撞、H40/P10、评分、保留旧失败。
- **Core MIX**：先核对实际完整状态和owner；现有训练继续至站点finite deadline或2K精确完整封存，2K之前的自然有界完整前缀保留。**不能杀健康GPU训练为新评估抢GPU**，不能双owner；所有后继从完整封存的绝对s+1续，Adam/RNG/LR/cursor连续。
- **VANDA ADAPT**：真实`1599363`继续；`1599364`和`1599365`及其后继，须只读核验实际Q/H/R/F和未到期资格。旧PBS被移出队列的原身份永久保留，不重复提交其曾经产生的效果。原模型worker `1432c8dc88c57adf39e8fa363f5438c5e63aa8fd` 和所绑定数据/统计/codec不热改。TGT5K已封存，禁止再训练相同TGT。

## 3. Corelab事件任务图：单native world，GPU0/1有界交接

```text
native F2 sixth RUNNING → 原单位CLOSED＋完整末尾/评分/seal
  ├─ 原固定ADAPT0 DEV12 已完成回执（6 F1 + 6 F2）
  └─ 保留剩余ADAPT0 DEV13..24的冻结人口，待真实资源/原排序轮转

Core MIX active (ledger≈1908 user snapshot) → exact MIX2000 VERIFIED_SEAL
  ├─ 独立固定离线Eval: 原三域96×3＋恢复48，单次正式claim
  ├─ 本站原有限后继与冷态公开HF发布，按原批准额度
  └─ Core合法冷态资源→TGT5K RGB16→TGT DEV首6；ADAPT后续/ADAPT2K部署排队

学生已终态且world/资源释放→ G8 HOLD4 TRAIN（原4尝试上限）→ G7准入/只读审计
```

核对至少以下已报道server owner，**unit可能已换代，须回读新真实身份**：Core MIX剩余窗口`wm02-sla-scale20k-mix-core-remaining-site-window-20261120-01.service`、MIX2K Eval观察者、ADAPT0尾部观察者、TGT5K RGB16观察者、HOLD4观察者、原Goal终态finalizer。现有代码已允许暂停SSH后继续，不再创第二套watcher。单world N2未资格，不同时启动两套物理native。

**站点原窗口到期**：历史Core窗口截止为 **2026-11-21 04:21:44 SGT**，其具体unit/lease可能更早截止。评估在该窗口内能否安全完成：若2K任务在保存余量前仍不够，先封存完整有界前缀，不强求2K，也不能超时抢Eval。若2K seal及时到达且额外GPU/时限合格，则合法释放原固定Eval；否则标`BLOCKED_SITE_WINDOW`，下一次新批准allocation后恢复。HF只发布完全封存的检查点，哈希逐件回读；不读正在训练的权重目录。

**GPU资源**：GPU1历史受其他进程占用、低于原约15.9GB十进制空闲门，等待不意味着可用。只读核验GPU/主存/PSI/owner/磁盘；若GPU0在MIX自然冷态且确有资源，才资格地迁至GPU0。禁止抢占第三方进程、降低显存或安全门、跨站远程控制原生world。

## 4. VANDA现有真实依赖图

```text
ADAPT1500 sealed → 1599363 (1501..1750) → 1599364 (1751..2000 exact)
                                              ├─ 1599365 independent 2K Eval
                                              ├─ once D2 cold-selection → 1599367..1599378 stage2 → 5K exact
                                              │                                         └─1599379 5K independent Eval
                                              └─ 1599382 CPU CLOCK trigger (after original fixed Eval)
                                                   └─ 1599385 conditional activation
                                                       └─ 1599386..1599391 CLOCK A/B + own Eval only if gates pass
```

以现场PBS原始依赖和每一执行卡为准，不因本图重新qsub。正式训练**不等待Eval成绩**；后继读取封存的父checkpoint，校验完整140状态、Adam/RNG/游标及已实际执行的更新账本。若原`afterany`触发，必须再检查完整seal，不能仅根据PBS退出码成功。合法exit75完整前缀可以另接绝对s+1；**不得绕过精确2K**。若2K未封存或数据D2资格失效，按现有D1 fallback独立决定，不能事后回切D2。

ADAPT确实到2K才做一次D2决定：新增Go2窗口12/旧52、RoboDojo32、LIBERO32，保留原ADAPT状态统计、C20动作codec及统计、源统计、原runtime、Adam/RNG/LR。同期独立Eval固定 Go2 TRAIN96+R DEV96+L DEV96+Go2阶段88+独立恢复48（合计424生成），要求准入、固定面板、输出与48/48恢复身份；不得把离线合法视为native成功。

CLOCK只在ADAPT2K正式Eval所读原固定24到期窗“及时且双门通过”为**0/24**且原标签/时钟/codec/专家H40和接缝可行性审计通过后科学触发；若非0则NOT_TRIGGERED，若缺信任证据或资源则BLOCKED并保留分母。原A/B各至500、原72native人口、原预算不变；不热改主ADAPT。

历史VANDA短链截止 **2026-11-22 08:33:10 SGT**，阶段二＋CLOCK最迟 **2026-11-24 09:35:31 SGT**；站点分配、每卡6小时与STOP优先。新PBS队列已登记；不假设未来凭证持续有效或有额外A40，队列撤销按原日志只读核验并fail closed。已经登记的导出HF/Go2包只在端点真实seal后执行，Core接收端严格核对，登录不通就阻塞这一传输，不阻断健康训练。

## 5. Core学生和专家：按优先级轮转，不让仿真吞没训练

1. **先封存F2第六条实际终态**，把F1六条、F2六条按原DEV24的首12累积；若第六条因外部中断未有效启动，归类技术与科学真实分母，不能自动重算或改写既有CASE。
2. **原ADAPT0后续冻结DEV13..24**：必须独立确认旧7..12已终态以及该段未启动身份；资源合格才派原顺序，每次最多6。已有前11/12科学失败不可删除。
3. **TGT5K**：原Core严格RGB16尚未完成，取得真实包/GPU独占并通过固定16与原门，才做原SCALE DEV首6；不抢正在训练GPU0，不需要把ADAPT0全部跑完才评价每个独立模型。
4. **ADAPT2K**：原VANDA独立Eval成功→CPU导出并校验Go2部署包→Core真正接收与独立固定RGB16→原ADAPT DEV24首块；严守原WBC/H40/P10/STOP/评分与split；没有端点不能凭用户1,644进度提前启动。
5. **G8 HOLD4**：在学生已封存块、world空闲且原受控时段内继续已冻结4次TRAIN尝试（最多142,812 native ticks和480 expert H40），成功独立示范才算入库，失败分类与未消费后缀保留；不得混入当前D2/ADAPT活跃银行。下一D3任务仍优先补F1蓝、F2红，以及PLACE红左、蓝右。

本轮不增加旧Goal新科学请求、更新或生成的预算，不创建额外样本选择或重测固定DEV/TEST。

## 6. 故障处理、事件回执与验收

- 只读调和`job_key=(lineage,source_sha,parent_checkpoint_sha,absolute_start,absolute_end,site)`，native依`population_sha+case_id+reset+policy_sha`单claim。遇UNKNOWN先查ledger/owner/启动时间/结果SHA，不重投。
- 原生STOP终态保留第一非法接触、原2ms积分、末部分动作和0 post-STOP tick。出现`ILLEGAL_CONTACT`即科学失败，不修改机器人碰撞几何、WBC、安全门或评分。实际生成/请求/积分与模拟器日志逐项对齐。
- 到期、STOP、资源不足、授权断连须产生真实`RESULT`与`BLOCKED_*`／`PARTIAL_SEALED`，不冒充成功。现役未结束时仅凭GitHub汇总不能替代站点最终状态。
- **事件发布**而非每6小时Codex监控：F2最后终态；MIX精确2K或末段有效前缀seal；ADAPT1750/2000与D2；独立Eval；CLOCK判定；下一批native成功或科学失败；site截止。GitHub/HF发布不在训练控制关键路径。
- 只有实际站点运行后的`OVERNIGHT_SERVER_MOUNT_ACTUAL.json/md`可以声称新后台挂载。必须包含：观察SGT时间、每站实际JobID/PID/状态/期限/owner/STOP、父检查点和访问银行SHA、排队依赖是否现场确认、成功/失败完整分母、已注册与仅计划区别、下一合法端点、当次新增优化更新/生成/native/ticks。若全部已有链合格，无须提交新任务，记录`EXISTING_BACKEND_SUFFICIENT`。

## 7. 限制

本交接不是 Corelab/VANDA 的远端shell或PBS会话。当前使用 GitHub connector 发布规范不自动执行`qsub`、`systemctl`、SSH或启动Codex。站点资源账户和现场授权的执行者在启动会话里完成真实新注册，已运行的后台作业与用户本地电脑无关。旧 Goal V2 T0+14天终点 **2026-12-04 15:29:58 SGT** 不变，每个作业以原站点较早截止为准。
