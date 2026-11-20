# MIX 1,768 封存与 ADAPT 端点交接实绩

2026-11-21 00:39 SGT，Corelab MIX 已记账 **1,778** 更新，三域访问 **56,896／85,344／85,344**。VANDA ADAPT `1599363` 已记账 **1,542** 阶段更新，三域 **98,688／49,344／49,344**；最近完整封存仍为 1,500。相对上一公开截面的两个账本共增加 **84 更新、10,752 访问、336 条训练内 H40**。本轮另行提交的作业均为 CPU 发布或导出，新增独立生成、原生请求及积分均为 0。

## Corelab

- 原 MIX 执行段以退出码 75 封存 **1,768**，已核对完整 140 个状态数组、388 次新增更新及其访问账本。原接续管理器已从该封存恢复，健康训练服务 `wm02-sla-scale20k-mix-core-remaining-site-window-20261120-01.service` PID **1106988**，截止 **04:21:44 SGT**。原精确 2K Eval 守候 PID **904458** 正在运行，尚无 2K 结果。
- MIX 1,768 已公开，匿名核验 **141/141** 文件 SHA256，共 **5,547,651,888 字节**，VANDA 独立匿名回读了同一封存清单。 [Hugging Face 固定版本](https://huggingface.co/datasets/leoyuer/sla-goal-v2-checkpoints/tree/5d483cdcd1ecfdd5b728cfd606e5ae67c6aa5596/SCALE-MIX20K/step-01768-de6648b27c7f)
- CPU 守候 `sla-observer-core-mix-after1768-cold-public-hf-20261121-01.service` 已从独立连接核对存活，PID **1119749**，InvocationID `978570ed4c924443971d3e6c09806634`，`Linger=yes`。仅在相同训练身份冷态、核对封存与原账本后才上传，保留原 **04:21:44 SGT** 期限。首次预检缺少导入依赖，未启动服务，补齐后通过。新守候实际发布 0 字节，停止入口 `STOP.json`，日志 `upload.log`。

## VANDA 2K／5K 分流

下表作业使用当前 PBS 身份，JobID 的服务器后缀均为 `.stdct-mgmt-02`。

| 端点 | 训练／后继 | 独立 Eval | Hugging Face 公共检查点 | Core 部署包导出 |
| --- | --- | --- | --- | --- |
| ADAPT 2K | `1599364` H，等待 1,750；`1599367` 等精确 2K | `1599365` H | `1599425` H，等训练封存 | `1599418` H，等正式 Eval；收尾 `1599419` W |
| ADAPT 5K | `1599378` H，等待前一训练段 | `1599379` H | `1599426` H，等训练封存 | `1599420` H，等正式 Eval；收尾 `1599421` W |

2K 仍等实际封存与一次 D2 选择，CPU CLOCK 触发作业 `1599382` 保持原依赖。2K 的新旧数据配比及完整状态定义沿用固定计划。已发布 ADAPT 1,500 与 TGT 5K 可匿名读取；2K／5K 的未来发布和导出均尚未运行。原短链、后继期限分别为 **11 月 22 日 08:33:10** 和 **11 月 24 日 09:35:31 SGT**。部署包经正式 Eval 和独立恢复证据绑定，需另做 Core 保管与 RGB16。 [固定计划](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1b2a25b7c1720051a2627d30831fb18130462374/docs/protocols/SLA_MULTISOURCE_CLOSED_LOOP_GOAL_V2_20261120.md)

## 学生与本体数据

Core GPU0 现役 MIX，GPU1 有其他用户计算进程，空闲 **14,025 MiB**，低于原 **15,148.8 MiB** 余量要求。ADAPT0 原六条 **0/6** 保留，余下冻结条件和 HOLD4 仍在守候；TGT5K 首六条因严格 RGB16 资格和资源未派发。这是资源及部署阻断。ADAPT2K 的 Core 守候最晚派发 **04:05:44 SGT**，早于按 VANDA 近期 20 步均时 **50.12 秒／步** 估算的 2K 封存时间，后续需有效 Core 窗口。

D2 冻结清单仍是 26 条、2,031 窗，两个既有 HOLD 正例均保留。D3 的四条 HOLD 已通过场景身份预检，实际新增 **0 尝试、0 tick、0 入库**。六个 PLACE 和两个恢复槽继续依现有资格判断。历史 F2 红目标优先补下探途中的腕侧与蓝色干扰物净空；蓝右放需核对搬运时持物与非目标块净空；红右放另须在 OPEN 阶段核对桌面支撑连续性。上述首因来自已封存轨迹。 [四条 HOLD 预检](G8_TRAIN_HOLD4_PREFLIGHT_ACTUAL.md) · [历史放置复核](../sla-d2-teacher-parallel-20261118/D2_PLACEMENT_PHASE_REPLAY_ACTUAL.md)

## 双卡记录复核

原双 A40 `1598205` 在第二次辅助反向由 replica 0 报告 5,124,037,576 字节工作区分配失败，完整工程更新仍为 1。首步 JAX 分配器统计为 31.89／16.72 GiB；PBS 整作业设备峰值字段分别为 33,222／33,212 MB。两组统计的时段与口径不同，现有记录不足以把第二步失败归因于 Adam 或持久的单卡占用不均。本轮新增吞吐训练、生成和 GPU 预算消耗均为 0。 [原终态](TGT5K_NATIVE_QUEUE_AND_TWO_A40_TERMINAL_ACTUAL.md)
