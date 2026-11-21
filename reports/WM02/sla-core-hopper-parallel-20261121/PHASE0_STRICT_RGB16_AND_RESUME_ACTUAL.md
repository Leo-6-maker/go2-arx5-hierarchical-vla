# Hopper 项目额度接续与 phase 0 跨卡复验

2026-11-21 17:50 SGT。Hopper 项目 `CFP04-CF-007` 的成员、有效期和可用余额已用 `amgr` 核实。VANDA 的 PBS 调度器已自行完成“第 100 步封存 → 独立 H200 参考推理 → 从封存点继续训练”。Core 的复验在第一个固定窗口就出现动作字节差异，phase 0 学生派发已封存为 `ENDPOINT_NATIVE_HELD`。独立连接读回了训练更新、模型接收回执和学生服务心跳。[逐条摘要和原始回执 SHA](PHASE0_STRICT_RGB16_AND_RESUME_ACTUAL.json)

| 环节 | 现场结果 |
|---|---|
| 正式训练 | `727341.hopper-m-02` 封存 100 步。`727394.hopper-m-02` 在 `hopper-17` 接续 38 步，累计 138/20,000，最近完整封存仍为 100。Go2 / RoboDojo / LIBERO 分别访问 8,832 / 4,416 / 4,416 次，完整 H40 反传 552 次。当前分段目标为 867。 |
| 独立 H200 参考 | `727376.hopper-m-02` 在 `hopper-28` 完成 16/16 并正常退出，计费 108 H200 秒。工程、首段训练和这次参考作业累计已结算 3,403 H200 秒，在跑分段另计。 |
| 模型跨站 | VANDA 将 29 个参数文件与清单共 1,831,892,085 字节发布到 HF。Core 从固定 revision 接收并核对全部 29 个文件。Hopper 训练用的 140 数组封存单独保管。[公开的 Go2 推理包](https://huggingface.co/datasets/leoyuer/sla-goal-v2-checkpoints/tree/d5542460abd57533b27a7ede06cda5cb652c1162/HOPPER-ADAPT20K-V1/Go2/MAIN-phase00000-fdf58d6078f3) |
| Core 严格 RGB16 | RTX 4080 SUPER 实际生成 16/16，启动到写出结果约 366 秒。16/16 的窗口、原始输入、噪声编号和原 H40／接缝判定一致。归一化动作与解码动作两项各为 0/16 逐位一致。首个分歧在 ordinal 0、窗口 21、noise ID 493863079。 |
| 学生原生仿真 | 独立人口核验写出 `ENDPOINT_NATIVE_HELD`。新执行服务 `wm02-sla-core-hopper-v1-native-students-20261121-01.service`、PID 2007686 正常心跳，等候合格端点。学生 DEV/CONF 派发 0，原生积分 0，抓持成功率无分母。 |

## 动作差异的证据范围

首个已记录差异出现在模型归一化输出，早于物理解码。Core 与 Hopper 的 12 个请求和模型路径源码文件 SHA 相同，JAX 0.5.3、jaxlib、CUDA 插件等十个安装包的版本标记一致。Core 启动环境显式设置了 XLA 确定性和禁用 autotune 的参数。Hopper 的 PBS 脚本显式设置 OMP=2，Core 为 1，Hopper 没有归档参考进程生效的 XLA 参数。不同 GPU、运行参数和噪声张量生成仍需区分。已有回执保存了输出摘要，无法从中计算误差幅度或确定产生差异的算子。

另外只读比对了 VANDA A40 已封存的 phase 0 冷恢复输出，16/16 的窗口与噪声编号对齐。A40 与 H200、A40 与 Core 的归一化和解码输出分别都是 0/16 逐位一致。A40 独立评价器在本机冷恢复时为 48/48 逐位一致。诊断新增生成 0，原 phase 0 资格结果保持。[同一端点的正式评价](MAIN_PHASE0_AND_TRAIN_CARRY_ACTUAL.md)

## Core 专家采集与数据

| 截止 17:50 SGT | 已终态 | 抓持／抬升 20 mm | 完整成功且准入 | 首次终止原因 | 终态积分 |
|---|---:|---:|---:|---|---:|
| TRAIN | 24 | 15 / 15 | 7 | 非法接触 15，搬运超时 2 | 697,016 |
| VAL | 19 | 11 / 11 | 5 | 非法接触 13，搬运超时 1 | 537,272 |

另有第 20 个 VAL 请求正在积分，所以总计 44 个专家请求已启动、43 个已封存。28 个硬 STOP 后积分均为 0。累计 TRAIN 快照已封存 4 条，7 条 TRAIN 获准入。VANDA 已有的三域训练资产已直传 Hopper。新 Core 示范的跨站私有交付仍待可用路径，主臂按已登记课程继续。

正式训练源码为 `b9581196d0acd02fb748222988e5a5543ce1f232`，独立参考源码为 `b52f3fc7a319dfeba1f9388a219e9574b94c0a23`，学生服务源码为 `b51fa2c102ee9c146b17bae29905db917f475eb2`。VANDA 调度与发布作业分别为 `1601905.stdct-mgmt-02`、`1601890.stdct-mgmt-02`。学生服务日志可由 `journalctl --user -u wm02-sla-core-hopper-v1-native-students-20261121-01.service` 查看，服务目录内的 `STOP` 为停止入口。长期目标截止 12 月 5 日 12:41:52 SGT。下一个预登记部署端点是 phase 2,000。[前一训练封存回执](MAIN100_AND_REFERENCE_HANDOFF_ACTUAL.md)
