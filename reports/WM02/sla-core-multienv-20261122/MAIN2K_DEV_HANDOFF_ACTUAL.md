# MAIN2K 原 DEV6 接班

Core 现场截面为 2026-11-22 06:03:14 SGT。固定计划 `4fe2c8cf1b69c168b2e3079e061bb7c702089cd3`。站点、源和运行身份见[机读回执](MAIN2K_DEV_HANDOFF_ACTUAL.json)。

- N1 第 4 张短卡实际 **1 请求、10,703 tick、200 个对齐消费点、20 对 RGB**，在原状态容差内。N1 累计 **4 请求、42,812 tick**。冻结控制器的实际选中参考身份仍待获取。N2、N4 原生维持 `NOT_RUN`。
- N1 原生子进程封存后，`wm02-sla-core-multienv-canonical-dev-20261122-08.service` 已接续并运行 MAIN2K 原 DEV6。协调 PID **2794896**，块 PID **2839636**，GPU worker PID **2839638**，后者已在原 GPU 实际分配显存。当前处于推理 worker 就绪等待，实际第一积分标记 **0/6**，终态 **0/6**。E0 接口沿用已封存的历史分母。
- 已封存压缩样本的测试副本被故意更改一个字节。验证器拒绝同样大小但 SHA 不匹配的 `physics.jsonl`，没有发布错误的还原目录。原归档和原始记录保持。[Phase0 与归档原实绩](PHASE0_DEV6_AND_COLD_ARCHIVE_ACTUAL.md)

06:04 SGT 独立回读确认红色 HOLD 已开始原生积分。MAIN2K 原 DEV 当前 **1/6 启动、0/6 终态**。
