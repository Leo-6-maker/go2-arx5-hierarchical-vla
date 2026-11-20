# SLA 夜间后台接续现场回执

**2026-11-21 02:37 SGT · DETACH_READY · EXISTING_BACKEND_SUFFICIENT**

Corelab 和 VANDA 的原有作业已由服务器接管。首次和独立连接均已退出。两次连接间，Core MIX 从 **1,957** 增至 **1,963**，VANDA ADAPT 从 **1,689** 增至 **1,694**。Core 确认 `Linger=yes`、用户服务 PID 实际属于对应 cgroup、GPU0 文件锁仍由训练 worker 持有。VANDA 从 PBS 回读运行中单卡 allocation、会话和下游依赖。此次新增提交、训练启动、评价启动、生成请求、仿真请求及积分均为 **0**。 [固定夜间计划](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/60cf6068db794582c27ca51d8a8ab8f289c70e71/docs/protocols/SLA_OVERNIGHT_SERVER_QUEUE_V1_20261121.md)

## Corelab

| 工作 | 实际服务器身份 | 02:37 SGT 状态 |
| --- | --- | --- |
| MIX | `wm02-sla-scale20k-mix-core-remaining-site-window-20261120-01.service`，主 PID `1106988`，GPU worker `1107006` | 账本 **1,963** 步；Go2／RoboDojo／LIBERO **62,816／94,224／94,224** 次访问。最近确认完整端点 **1,768**。精确 2K 仍待封存。1,963 单步耗时 **167.64 s**，之前三步各约 **36–40 s**，按原期限继续。 |
| MIX2K 离线评价 | `sla-observer-mix2000-eval-20261120-05.service`，PID `904458` | 已登记。精确 2K 完整封存后，按原 GPU 资源与剩余时段门释放固定 **288＋48** 生成。当前 0。 |
| MIX 冷态发布 | `sla-observer-core-mix-after1768-cold-public-hf-20261121-01.service`，PID `1119749` | 守候本段完整封存，按之前的公开发布合同核验再上传。 |
| ADAPT0 最后 6 条 | 原守候 `sla-observer-adapt0-goal-v2-dev-tail-resource-20261120-02.service`，PID `901108`；子服务 `wm02-sla-adapt0-goal-v2-dev-tail-to24-20261120-01.service`，PID `1233594` | 既有守候已登记最后一块。子服务为 `WAIT_SHARED_GPU_RESOURCE`，最后 6 条新增原生请求为 **0**，单 world 队列继续。 |
| HOLD4 | `sla-observer-goal-v2-train-hold4-20261120-04.service`，PID `920117` | 等上述 ADAPT0 原固定 DEV 块交接，原生尝试、积分、入库均为 **0**。 |
| ADAPT2K Core RGB16 | `sla-observer-adapt2k-core-rgb16-after-custody-20261120-01.service`，PID `836397` | 等 VANDA 精确端点、正式 Eval 与包保管。按当前训练进度，本次最晚 **04:05:44** 派发窗不足。后续需要有效窗口和验证过的 Core 接收。 |

MIX 站点租约至 **04:21:44 SGT**，训练内截止 **04:20:44**。ADAPT0 当前块内截止 **04:18:44**。Core campaign 收尾 `sla-goal-v2-core-finalizer-20261120-01.path/.timer` 均已激活，计时器指向原 **12 月 4 日 15:29:58** 停止边界，各现役作业取较早站点截止。停止入口为项目原 `STOP.json`，日志通过相应用户服务 journal 和运行目录的 `worker.log`、`watch.log`、`UPDATES.private.jsonl`、`RESULT.private.json` 定位。TGT5K 已完成训练与独立离线评价。原生首批 **0**，原守候终态为 `TGT5K_STRICT_RGB16_UNQUALIFIED`，等下一次经核实的冷态资源与严格 RGB16。 [MIX 1,768 已封存检查点](https://huggingface.co/datasets/leoyuer/sla-goal-v2-checkpoints/tree/5d483cdcd1ecfdd5b728cfd606e5ae67c6aa5596/SCALE-MIX20K/step-01768-de6648b27c7f)

### F2 第六条及固定 DEV 分母

蓝色物体右放 `ADAPT_DEV-F2-1-PLACE-blue-right-ADAPT0` 已由原进程封存为科学失败。原生 **1** 次请求、**29,025** tick（含初始化 **5,703**），生成和完整 H40 编译 **94/94**，确认完整 P10 **93/94**。最后一次完成两个 50 ms 原子，第三个原子消费 **22/25 tick** 时触发 `ILLEGAL_CONTACT`。首次非法接触为 `x5_link6`（geom 73）与蓝色目标块（geom 79），STOP 后积分 **0**。目标抓持、抬升 20 mm、入选定区、稳定释放及撤离评分均未通过，最大抬升 **2.07 mm**。

| 封存学生条件 | 真实原生请求 | 生成 | 积分 tick | 抓持／抬升20 mm／任务成功 |
| --- | ---: | ---: | ---: | --- |
| 第一布局 F1 | 6 | 396 | 132,555 | 0/6 · 0/6 · 0/6 |
| 第一布局 F2 | 6 | 401 | 133,774 | 0/6 · 0/6 · 0/6 |
| 第二布局 F1，第 13–18 条 | 6 | 373 | 126,255 | 0/6 · 0/6 · 0/6 |
| 已封存学生累计 | **18** | **1,170** | **392,584** | **0/18 · 0/18 · 0/18** |

18 条均为原生碰撞保护终止，STOP 后积分合计 0。另有两条此前的接口请求共 11,906 tick。最后六条的协议仍为原冻结条件，等待其真实请求和终态分母。

## VANDA

训练 `1599363.stdct-mgmt-02` 为 **R**，PBS 会话 `2770971`，现有一张 A40，36 CPU、240 GB RAM、6 小时 walltime。ADAPT 账本 **1,694** 阶段更新，三域访问 **108,416／54,208／54,208**，最近确认完整检查点 **1,500**。本段训练内截止 **11 月 21 日 05:55:14 SGT**，至 2K 原短链截止 **11 月 22 日 08:33:10**，阶段二和 CLOCK 最晚 **11 月 24 日 09:35:31**。

- `1599364` 为依赖挂起 H，等待 `1599363` 结束并封存 1,750。精确 2K 封存后，独立 Eval `1599365`（固定 **424** 生成）和阶段二首段 `1599367` 各自释放。D2 合格银行仍是 **26 条／2,031 窗**，未发生课程切换。原 2K 边界只决策一次，训练后继不等待 Eval 成绩。
- `1599367–1599378` 为阶段二至 5K 的原 12 段，`1599379` 为独立 5K Eval，均回读为 H。公开 checkpoint 作业 `1599425/1599426` 与部署导出 `1599418/1599420` 已登记，当前 H。Core 无人值守接收正式部署包仍待闭环核验。
- CLOCK 触发 `1599382`、激活 `1599385` 和两臂 `1599386–1599391` 均为 H，实际新增训练为 **0**。仅在精确 ADAPT2K 固定 24 窗联合指标 **0/24** 且原语义审计通过时释放原配对。
- 日志入口为 PBS JobID 对应的 `pbs-output.private.log`、`pbs-error.private.log`、`train.log` 和 `train/UPDATES.private.jsonl`，停止入口为执行卡绑定的项目原 `STOP.json`。收尾作业 `1599366` 与 `1599392` 仍为 W。 [ADAPT 1,500 已封存检查点](https://huggingface.co/datasets/leoyuer/sla-goal-v2-checkpoints/tree/f39030655f2643e866253b041f0d21f61f703e97)

完整 source/spec SHA、访问银行、租约、服务会话及 PBS 依赖见 [机读回执](OVERNIGHT_SERVER_MOUNT_ACTUAL.json)。
