# Core 原生余条件和 MIX 站点剩余窗口接续

最后并行核对 **2026-11-20 20:42:50 SGT**，固定计划 `1b2a25b7c1720051a2627d30831fb18130462374`。[机读实绩](CORE_ADAPT0_TAIL_AND_MIX_SUFFIX_ACTUAL.json)

| Core 现场角色 | 当前状态与依赖 | 期限和停止入口 |
| --- | --- | --- |
| `sla-observer-adapt0-goal-v2-dev-tail-resource-20261120-02.service`，PID **901108** | active，重新校验了原前六条实际结果。仅旧第 7–12 条资源拒绝、未创建真实原生 owner，且原 HOLD4 无派发，才允许按原顺序接续第 7–24 条。三个六条件单元均未启动。 | 最晚派发 11 月 21 日 **03:45:44 SGT**，站点截止 **04:21:44**。作业根目录 `STOP.json`、`watch.log`、分项 `RESOURCE.private.json`。 |
| `sla-observer-mix-remaining-site-window-20261120-01.service`，PID **904454** | active，等待现役 PID **892030** 冷态关闭并通过逐更新和完整状态核查。仅在 2,000 步以下的完整前缀续接，原 worker、父状态、LR 和三域 cursor 继续。GPU 后继 PID **0**。 | 原现役段到 **00:30:27 SGT**，后继沿用原站点截止 **04:21:44**。作业根目录 `STOP.json`、`watch.log`。 |
| `sla-observer-mix2000-eval-20261120-05.service`，PID **904458** | active，等精确 2,000 步和前一评价观察者的实际回执。只有旧观察者因不足 2k 而未释放原评价，才可能从后继释放同一份固定面板。GPU Eval PID **0**。 | 站点截止 **04:21:44 SGT**。作业根目录 `STOP.json`、`watch.log`，真实 GPU worker 日志为 `supervisor.log`。 |

三份卡及 PID/cgroup/InvocationID 均通过独立连接核对。Core 持久 finalizer 仍使用原 T0+14 天截止，登记角色已核实 **34** 个，spec `547ec2a554b85ac697ffd3e09c13be4042c9d4c7c4888527b2b9b5d11684d9cd`。现役 Core 训练与旧结果沿用原身份。

ADAPT0 守候复测 GPU1 仍有另一现役计算进程，空闲 **14,706,278,400** 字节，原门要求严格大于 **15,884,668,109** 字节，空闲和独占均失败。内存压力 avg10 已回到 **0.00**，主存、磁盘、world 锁及 GPU 身份通过。原生仍是 **0 新请求、0 tick**，原先前六个科学条件 **0/6**。续接复用同一 GPU1 部署证据、原固定人口、WBC、H40、P10、STOP 与评分。新规只处理已验证的零执行拒绝，已有任何原生派发或候选目录会拒绝新启动。[ADAPT 原生守候源码](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/commit/a7d4807f62b3ff28fbc950a2e5b27453634412f1)

VANDA 正式 ADAPT **1,443** 步，原 `1594212` 仍 R，累计访问 **92,352／46,176／46,176**。此前被移除的依赖作业均无启动证据，VANDA 14 天收尾任务 `1597930` 也在 **20:31:04 SGT** 进入未运行终态。撤下来源未确认，本轮在 VANDA 无新提交。精确 2k、D2 决策、5k Eval 与 CLOCK 科学触发均未执行。Core MIX **1,394** 步，累计访问 **44,608／66,912／66,912**。

本轮相对上一次实绩 MIX 1,381 和 ADAPT 1,433，计 **23 次优化更新、2,944 次样本访问、92 条训练内 H40**。新增独立离线生成、原生请求、仿真 tick 和数据入库均 **0**。本地验证 29 项通过，4 项 JAX 用例因解释器缺少 JAX 跳过。服务器回读验证的是守候服务，尚未形成新增原生或 MIX2k 端点评价结果。
