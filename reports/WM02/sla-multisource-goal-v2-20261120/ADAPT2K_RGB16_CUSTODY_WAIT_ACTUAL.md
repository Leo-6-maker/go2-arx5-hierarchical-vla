# ADAPT2k Corelab RGB16 接收守候实绩

2026-11-20 19:37:03 SGT，通过独立连接确认 `sla-observer-adapt2k-core-rgb16-after-custody-20261120-01.service` 为 `active`，PID **836397** 在同名单元 cgroup 内，InvocationID `406a9b2dabd0495297b6070c0a02ce3f`，用户 linger `yes`。运行源 `998219db39146fb752043150b677ffbca87692da`，执行卡 `331abbf398fb5f7d9ab7dce7c0e46b23c9e71fb1d1a03e3c62034b6bea75bb71`。 [机读回执](ADAPT2K_RGB16_CUSTODY_WAIT_ACTUAL.json)

服务等候精确 2,000 的原 VANDA 独立 Eval 与导出。包经现有接收器逐文件核对并原子入库后，按原 D1 双 RGB、上一实际命令、固定面板、噪声、统计、父模型和 GPU 身份构造严格 RGB16 卡。服务等当前 ADAPT0 轮转冷态，并读取精确 GPU/主存/cgroup/pressure/owner/盘空间门后，最多释放单次 16 条原条件。只有严格输出和守门签名都通过才产生部署资格。观察时接收回执不存在，GPU worker `wm02-sla-adapt2k-core-rgb16-goal-v2-20261120-01.service` PID **0**。该阶段计数为 **0 更新、0 访问、0 生成、0 原生请求、0 tick**。

现有 Core 窗口仍在 **11 月 21 日 04:21:44 SGT** 截止，RGB16 最晚派发 **04:05:44 SGT**。未到达、容量失败或窗口不足会记终态。观察者日志是作业目录中的 `watch.log`，停止入口是 `STOP.json`，派发后 GPU worker 有自己的 `replay.log` 和 `STOP.json`。两个身份已纳入原 finalizer，累计 22 个身份且定时器与 STOP 监听均有效。6 项 CPU 定向测试通过，涵盖正确端点、错误 phase 和错误设备路径。

| VANDA 端点 | 独立 Eval | 现有 CPU 导出 | 现场数据 |
| --- | --- | --- | --- |
| ADAPT2k | `1594215` H | `1598414` H，`afterany:1594215` | 精确工件尚未封存，Core 接收守候已登记 |
| ADAPT5k | `1597845` H | `1598416` H，`afterany:1597845` | 精确工件尚未封存，Core 派发仍待适用站点窗口 |

Core 可与固定 VANDA 主机密钥匹配，但非交互用户认证仍被拒绝。接收守候可独立运行，跨站发送端尚未连通。已存在的双 SSH 传输路径可在导出完成后移交同一封存包。当前两站均为 **0 已导出／已接收端点**。CLOCK500 触发 `1598095` 与激活 `1598251` 仍为 H。

同截面 Core MIX **1,307 步**，三域游标 **41,824／62,736／62,736**。VANDA ADAPT `1594212` R，**1,362 步**，游标 **87,168／43,584／43,584**。按本次两次读回相减，原训练合计增加 **21 更新、2,688 访问、84 条训练内 H40**。现役训练沿原配置继续，正式 2k 均未封存。
