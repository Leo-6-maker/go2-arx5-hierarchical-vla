# Core/Hopper 首次服务器实绩

2026-11-21 12:45 SGT。新计划的第一个受管作业于 12:41:52 SGT 登记，14 天截止为 12 月 5 日 12:41:52 SGT。两项准备作业均已退出。H200 正式训练、专家原生采集和学生评价均待挂载。[机读回执](FIRST_MOUNT_ACTUAL.json)

| 角色 | 独立连接回读 | 实际成本 |
|---|---|---|
| VANDA → Hopper 父状态直传 | PBS `1601402.stdct-mgmt-02`，`batch_cpu`，会话 4015629，F/0。两站回执逐字节一致。 | 54 秒，140 个模型、Adam 与随机状态数组，加完整封存清单共 5,547,652,642 字节。 |
| Core 数值场景编译 | `sla-core-hopper-v1-geometries-20261121-01.service`，PID 1733158，用户 2009，`Linger=yes`。已从另一连接读回终态。 | 54 个数值场景，4,021,412,454 字节，与 237 个历史场景哈希核对。原生请求、积分和模型生成均为 0。 |

固定 ADAPT2K 清单 SHA256 为 `2bff547b6f2c3aca0e7e2fd8a790ce6b30ef1c5f33b004c6897e3083647358d5`。固定 HF 版本、VANDA 源与 Hopper 副本一致。Hopper 尚待冻结底座、三域输入核验、完整恢复、有效项目额度和计算分配。此次 H200 占用及更新均为 0。[公开父状态](https://huggingface.co/datasets/leoyuer/sla-goal-v2-checkpoints/tree/5c460f195f6509f2f4da55dea32c356b45e00d37/ADAPT5K/step-02000-2bff547b6f2c)

Core 已绑定 TRAIN240、VAL24、DEV24、CONF24、TECH2 的数值场景。场景指纹与冻结控制模型相符。新 scope 的原生执行入口和单 world 采集服务尚待完成。原始 HOLD4 的入口失败封存于旧身份。全链无人值守尚待验收。
