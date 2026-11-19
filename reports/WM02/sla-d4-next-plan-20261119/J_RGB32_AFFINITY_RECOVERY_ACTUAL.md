# J 的 RGB32 已进入 Corelab GPU1

原接续服务在 CPU 亲和性预检失败。它只有 CPU 17，而封存的 RGB32 工作卡需要 CPU 8/9。原失败时 RGB32 没有注册运行目录、用户服务或生成，原生请求为 0。失败日志和原规格保留。

按原预检卡在新的受管服务重接，现场核对 `wm02-sla-d416-j-core-rgb32-20261119-01.service` 为活动状态，PID **3817881**，确认属于其 cgroup。独立接续进程 PID **3817803** 等待 RGB32 封存。原 J_SCREEN48 仍为 **0/48**。现有截止时间为 2026-11-20 00:12:04 SGT。

[机器可读现场回执](J_RGB32_AFFINITY_RECOVERY_ACTUAL.json)
