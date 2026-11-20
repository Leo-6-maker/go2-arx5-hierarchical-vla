# MIX 公开 HF 交接与 A40 恢复资格实绩

已创建 [公开检查点仓库](https://huggingface.co/datasets/leoyuer/sla-goal-v2-checkpoints)，匿名读回仓库可见。首次发布选 Corelab 已冷态封存的 MIX1000，检查点 SHA `54e98287d5df6897298fded5057c0ff8f45b806320d885d63401f4eced179399`。发布清单共 **141 文件、5,547,651,888 字节**，包含完整 manifest、45 份训练参数、94 份 Adam 状态和 1 份 RNG。清单限定为训练状态及原输入哈希。恢复仍需各站原冻结 SLA 底座和匹配数据。 [机读回执](MIX_PUBLIC_HF_AND_A40_RESTORE_ACTUAL.json)

首次 CPU 发布服务达到 4 GiB cgroup 上限后被终止。独立匿名读回确认仓库仍处于初始 revision，只有 `.gitattributes`，因此有证据支持更换为单线程发布。后继 `sla-observer-public-hf-mix1000-20261120-02.service` PID **854078**，执行卡 `4d5333096d233c4e0bbd262e6f9bec132a650d1f3942ee37110a5045cf47a7ef`，运行源 `96cba8e2b63f8c8cdf5a320dd14b5342382c7c4e`。截止沿用首轮 **21:05:21 SGT**，日志 `publish.log`，停止入口 `STOP.json`。最后读回时，服务在运行，公开上传仍待固定 revision 及匿名逐件校验。

VANDA 上 `1598673` 实际获得一张 A40 后，在读取旧 Core 绝对时间审计路径时失败，训练参数恢复、梯度输入、优化、生成和原生均为 0。新身份使用原 trainer `31076fc7a64302b0705ae4d7e526c4341209a100`，仅复用既有、已审计的跨站路径解析。VANDA CPU 输入预检重建了 1,768／50,979／209,146 窗和三域原访问银行。PBS **`1598687`** 在 19:55 SGT 实际 R，执行卡 `11636f41697e32020b03944dcbacd5d1ab1be08e09650a85d8137fc1335bcad8`。CPU 到期收尾 **`1598688`** 为 W，原窗口截止 **21:50:16 SGT**，GPU 段最多 30 分钟。

A40 资格读取已在 VANDA 的 MIX500，只检查 140 个状态数组以及下一原配方批次的梯度。预算为 Go2／RoboDojo／LIBERO **32／48／48** 个梯度输入和 4 条训练内 H40。资格作业不执行 Adam，不产生科学父检查点。正式 MIX 延续仍需最新冷态检查点、通过的恢复证据和单一训练所有权。两个工作均未改动现役 MIX 或 ADAPT。
