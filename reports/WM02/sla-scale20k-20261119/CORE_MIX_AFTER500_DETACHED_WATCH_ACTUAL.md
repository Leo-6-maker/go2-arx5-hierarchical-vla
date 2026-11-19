# Core MIX500 冷态接续已由服务器守候

Core 用户服务 `wm02-sla-scale20k-mix-after500-handoff-20261119-01.service` 已独立读回 **active**。服务 PID **3874724**，对应同名 cgroup，账号 linger 已确认。安装连接退出后，另一个连接核对了继续更新的心跳，仍处于 `WAIT_ORIGINAL_OWNER_COLD`。观察截止为原 first500 的有限期限 **2026-11-19 23:49:33 SGT**。

原训练服务 PID **3763767** 继续运行。读回时训练心跳在第 364 步，游标 Go2 11,648、RoboDojo 17,472、LIBERO 17,472。后继训练服务仍未启动，守候器的实际模型生成、优化更新及原生请求均为 0。

守候器只在原服务退出、原完整第 500 步检查点与训练账本核验通过、现场显存和主存允许后，使用已封存的接续入口派发一次。到原观察窗结束、收到 STOP 或出现未决所有权即退出。守候器的停止入口为自己目录下 `STOP.json`，日志为同目录 `watch.log`。停止守候器不停止原训练。

源码 `5febebc741617dfbef4b1cd7c17c42ba6cf1b1a9`，规格 SHA-256 `15197966d259c57ba66d42204d1e48b29804a0182afc2a2f34c416d7df2659b1`。本地 CPU 验证 17 项通过。后继真实恢复与更新以现场终态和更新账本为准。

[现场所有权与心跳证据](CORE_MIX_AFTER500_DETACHED_WATCH_ACTUAL.json)
