# MIX 公开 HF 交接与 A40 恢复资格实绩

已创建 [公开检查点仓库](https://huggingface.co/datasets/leoyuer/sla-goal-v2-checkpoints)，匿名读回仓库可见。首次发布选 Corelab 已冷态封存的 MIX1000，检查点 SHA `54e98287d5df6897298fded5057c0ff8f45b806320d885d63401f4eced179399`。发布清单共 **141 文件、5,547,651,888 字节**，包含完整 manifest、45 份训练参数、94 份 Adam 状态和 1 份 RNG。清单限定为训练状态及原输入哈希。恢复仍需各站原冻结 SLA 底座和匹配数据。 [机读回执](MIX_PUBLIC_HF_AND_A40_RESTORE_ACTUAL.json)

首次 CPU 发布服务达到 4 GiB cgroup 上限后被终止。独立匿名读回确认仓库仍处于初始 revision，只有 `.gitattributes`，之后更换为单线程发布。后继 `sla-observer-public-hf-mix1000-20261120-02.service` PID **854078**，执行卡 `4d5333096d233c4e0bbd262e6f9bec132a650d1f3942ee37110a5045cf47a7ef`，运行源 `96cba8e2b63f8c8cdf5a320dd14b5342382c7c4e`。后继正常退出，固定 revision `58a7d0c25ca959904a6d0947b1d80a87e4ee6eb0` 上 **141/141** 文件大小和哈希通过。VANDA 独立匿名下载清单，其完整状态 SHA 与 Core 封存一致。 [MIX1000 固定 revision](https://huggingface.co/datasets/leoyuer/sla-goal-v2-checkpoints/tree/58a7d0c25ca959904a6d0947b1d80a87e4ee6eb0/SCALE-MIX20K/step-01000-54e98287d5df)

VANDA 上 `1598673` 实际获得一张 A40 后，在读取旧 Core 绝对时间审计路径时失败，训练参数恢复、梯度输入、优化、生成和原生均为 0。新身份使用原 trainer `31076fc7a64302b0705ae4d7e526c4341209a100`，仅复用既有、已审计的跨站路径解析。VANDA CPU 输入预检重建了 1,768／50,979／209,146 窗和三域原访问银行。PBS **`1598687`** 在 19:55 SGT 实际 R，执行卡 `11636f41697e32020b03944dcbacd5d1ab1be08e09650a85d8137fc1335bcad8`。CPU 到期收尾 **`1598688`** 为 W，原窗口截止 **21:50:16 SGT**，GPU 段最多 30 分钟。

A40 资格读取已在 VANDA 的 MIX500，只检查 140 个状态数组以及下一原配方批次的梯度。预算为 Go2／RoboDojo／LIBERO **32／48／48** 个梯度输入和 4 条训练内 H40。资格作业不执行 Adam，不产生科学父检查点。正式 MIX 延续仍需最新冷态检查点、通过的恢复证据和单一训练所有权。两个工作均未改动现役 MIX 或 ADAPT。

19:58 SGT，`1598687` 已关闭，PBS `F/1`。**真实 A40 恢复并回验 140/140 份参数、Adam 和 RNG 数组**，实际完成 Go2 32 个梯度输入和 4 条训练内 H40，未应用优化更新。随后 RoboDojo 首个读取缺少 `numeric/052.npz`，RoboDojo 和 LIBERO 梯度输入均为 0，完整跨站训练资格未取得。独立读回确认站点另有 TRAIN 归档，两域清单 SHA 均与 MIX500 原训练一致，RoboDojo 缺失文件和原始文件路径映射均存在。后续需验证正式 TRAIN 读取路径，保留本次分母与失败。

## 三域 A40 资格复验

VANDA CPU `1598712` 在 1 分 16 秒内实际读通原下一个批次的 **32／48／48** 个 TRAIN 输入。A40 `1598713` 按 `afterok:1598712` 启动，3 分 11 秒后以 **F/0** 退出。MIX500 训练状态 **140/140** 从 A40 回验一致，原三域梯度 **32／48／48** 全部完成，训练内 H40 **4**，优化更新、优化访问、原生请求及 tick 均 **0**。峰值设备分配 20,573,715,200 字节。原梯度 worker 与学习配置沿用，只有站点数据路径和经哈希验证的 TRAIN 读取映射变化。旧两条失败保留。结果 SHA `521eb18f7d1a196804fb27d21dacfc13ee8a50965da2422bf63b9ca33f3a28d5`。

## 两站公开发布接续

VANDA 使用该站原有 token，CPU `1598725` 自 **20:07:13 SGT** 起运行，发布已封存的 TGT5K，阶段生产作业 `1593720` 已核对 F/0。执行卡 `fc5ed1d09a2d2e4c9f02bda7054f4fd66f6ea4f44f5b08f8a8f2cc9669190d96`。TGT 完整状态 SHA `9eb71c0da7c0f34373d53c7b6c6b6dd3491e5a36dfd94d6ae0eb4d799fbdff08`，发布清单 141 文件、5,547,651,881 字节。到期作业 `1598726` 已登记为 W，截止 **21:50:16 SGT**。本作业的调度、上传和核验均在站点执行，使用 `STOP.json`、`stdout.private.log` 和 `stderr.private.log`。

Core 下一段冷态发布服务 `sla-observer-core-mix-next-cold-public-hf-20261120-01.service` 已由独立连接确认运行，PID **870042** 属于单元 cgroup。执行卡 `a3b35e2c77e140d7e5003e6fd76430fff09b2a5cc91f9ccec603ed8b32d5e65e`，站点截止 **21:47:33 SGT**。服务等待当前 Core MIX 的精确 owner 退出，核对已关闭的租约、退出码 75 的完整前缀或精确 2k 封存、逐更新访问和文件哈希。独立核对时该服务已发布 **0 字节**，新训练也为 **0**。全局 finalizer 已包含这一角色，停止入口 `STOP.json`，日志 `upload.log`。
