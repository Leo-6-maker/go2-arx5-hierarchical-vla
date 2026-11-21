# Core canonical 冷生成终态与 IPC 依赖修复

现场截面 2026-11-22 04:55 SGT。固定计划 `4fe2c8cf1b69c168b2e3079e061bb7c702089cd3`。冷配对、现场身份和 SHA 见[机读回执](CANONICAL_COLD_AND_IPC_IMPORT_ACTUAL.json)。

- Core 4080 SUPER 上 phase0 冷 A/B 各 **16/16**，MAIN2K 冷 A/B 各 **16/16**。共 **64 次实际 intent、64 次实际生成、0 原生请求、0 优化更新**。每个端点两进程的完整成果 SHA 分别一致。该证据关闭冷生成阶段，IPC 仍需按端点各执行 16 次。
- 上一 IPC unit 的两条 worker 均在编译缓存模块导入处退出，真实分母为 **0 请求 / 0 生成**。两条原 worker lease 均已关闭。上一 DEV unit 也已以 **0 原生请求**封存为 `NOT_RUN`。全部失败保留。
- 修复版改用项目已有的原生 IPC 启动入口，为原数值导入顺序追加冷态封存依赖。两端点的原冷结果和请求条件复用。实际预检走同一 Python 入口，验证缓存模块、D4/D416/Scale 运行依赖、IPC Worker 和两份冷结果，预检生成与原生请求为 0。
- 新 IPC unit `wm02-sla-core-multienv-canonical-ipc-20261122-05.service` PID **2769742**，期限 **11 月 28 日 16:52:28 SGT**。新 DEV unit `wm02-sla-core-multienv-canonical-dev-20261122-06.service` PID **2769746**，期限 **11 月 29 日 02:52:28 SGT**。独立连接确认两 unit 活跃、DEV 运行目录已创建。当前 IPC 真正派发为 **0**，DEV 两块均为 **0/6**。对应执行目录可用 `STOP` 停止。
- Wave A 已封存 **8 条、203,593 tick、3 条 G7 准入**，仍在原单环境服务运行。正式目录最新 **TRAIN-059**，新增准入的实际优化访问 **0**。
