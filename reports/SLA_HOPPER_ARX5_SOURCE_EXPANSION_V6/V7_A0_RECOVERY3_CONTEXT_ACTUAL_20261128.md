# V7-A0 · Recovery3 上一实耗命令回读

本次执行对应 [V7 固定计划](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/529) 的 A0 数据审计。Corelab 从三条已关闭的 TRAIN Recovery 原始消费账本回读 **115/115** 个查询，逐个核对真实夹爪命令、训练副本、物理范围和查询前一完整 50 ms 区间。固定 Recovery3 数据版本为 `362c8756c7b7ea62a539a441dcfa5a6bdf175f4eb92f79088ed7a7ec7a884991`。

| TRAIN 回合 | 真实已有窗口 | 教师首窗 tick | 继承的学生实耗开度 |
| --- | ---: | ---: | ---: |
| V64_RECOVERY_TRAIN_02 | 45 | 28,453 | 2.0137 mm |
| V64_RECOVERY_TRAIN_04 | 35 | 29,703 | 2.0183 mm |
| V64_RECOVERY_TRAIN_06 | 35 | 28,953 | 2.0857 mm |

每个教师首窗还核对了实际学生查询、上一完整 P10 的接收边界和控制参考身份。三处学生到教师的切换均通过。导出的上下文字段只有查询、时钟、上一实耗开度和有效位。归档手动复算与版本化 [回读脚本](../../scripts/sla/v7_recovery_context_audit.py) 使用同一批真实输入，报告字节一致，SHA-256 `8fabd1e61a6d724ac71f107a0372d002b622b95fa6998b6eb84758e487a48008`。

本次新增模型前向、训练更新、物理积分和数据准入均为 **0**。A93 的全量证据覆盖、Go2 条件通路和两臂同初始预测仍待实测。
