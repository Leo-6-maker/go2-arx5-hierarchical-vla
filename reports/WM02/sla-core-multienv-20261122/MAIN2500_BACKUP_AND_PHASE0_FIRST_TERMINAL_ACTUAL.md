# MAIN 2500 跨站备份与 phase0 首条闭环终态

现场截面为 2026-11-22 05:36 SGT。计划提交 `4fe2c8cf1b69c168b2e3079e061bb7c702089cd3`。身份、计数和端点 SHA 见[机读回执](MAIN2500_BACKUP_AND_PHASE0_FIRST_TERMINAL_ACTUAL.json)。

## Hopper 与 VANDA

- Hopper 的 MAIN **2,500 步**检查点已通过原站点直连传到 VANDA。VANDA 在写入封存前校验全部 **140 个数组和 COMPLETE.json** 的 SHA，合计 **5,547,651,598 bytes**。独立连接再次查到 **141 个正式文件**以及相同完整封存 SHA `144afb6a80d0575ecbe240f6b7dea5142a9a1bf3772cf92ad1130ee21dcdf49a`。此次传输新增 PBS 作业、训练、生成和原生仿真均为 **0**。原停用的 CPU 辅助链保留终态。
- 现役训练 `728156.hopper-m-02` 至 **2,549 次实际更新**。后继 `728212.hopper-m-02` 仍由 PBS 保持，等待 `afterok:728156.hopper-m-02`。它的实际更新为 **0**。已挂载的训练仍使用 TRAIN-052。TRAIN-063 已在 Hopper 的边界候选目录就绪，实际优化访问为 **0**，最早可在 **5K** 边界选择。[数据与冷读回执](TRAIN063_READY_AND_FIRST_DEV_INTEGRATION_ACTUAL.md)

## Core 学生 DEV 首块

- phase0 **2/6 已启动、1/6 有终态、0/1 成功**。红色 HOLD 实际 **1 次原生请求、21,203 tick、63 次策略生成、620 个完整确认消费点**。底座移动达标，定向位移 **141.3 mm**，路径 **304.4 mm**。全程未抓持，抬升 **0 mm**，原计分确认无非法接触。
- 第 **63 次**请求的完整 H40 通过原编译。上一条实际夹爪命令与新块第一点形成 **61.29 mm/s** 的接缝速度，超过冻结阈值 **40 mm/s**。等价于 **50 ms 内约 3.06 mm** 的开度变化，原阈值允许 **2.00 mm**。该请求没有安装，消费 **0 点**，首失败为 `G7_ACTUAL_PREVIOUS_COMMAND_SPEED_REJECTED`。夹爪接缝速度是该条可验证的终止原因。尚未完成的同臂及 MAIN2K 条件将给出它的发生频率。
- 蓝色 HOLD 已接续。MAIN2K 原 DEV 仍为 **0/6**，N2/N4 原生资格均为 `NOT_RUN`。保留冻结 WBC、原安全门和原条件顺序。
