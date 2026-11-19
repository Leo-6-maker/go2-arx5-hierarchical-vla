# SCALE20K 数据与训练接入实绩

固定计划 `926226914a0c08ee1ebb143e0edea1434728107b`。已有 **1,309 条公共源 TRAIN 示范、36 个任务**。Corelab 已逐文件校验本轮新到的数据共 **463 份、4,090,433,116 字节**。SCALE 新 GPU 更新、优化访问、完整生成和原生请求均为 0。

| 数据域 | TRAIN 示范 | TRAIN 窗口 | 独立源 DEV 示范 | 动作与时钟 |
|---|---:|---:|---:|---|
| Go2+ARX5 | 20 | 1,376 | — | G7，H40，50 ms，真实上一命令 |
| RoboDojo-Real ARX X5 | 48 | 50,979 | 12 | 双臂 H50，名义 30 Hz，未来记录参考 |
| LIBERO | 1,261 | 209,146 | 432 | Franka 原生 H10，10 Hz，OSC 动作 |

RoboDojo 新增 12 条，实际复核 12,873 个 H50 窗口和 72 张当前 RGB。TRAIN 统计只使用 48 条 TRAIN，沿用旧 12 条 SOURCE_DEV 和旧 12 条 SOURCE_TEST 隔离。LIBERO 按原任务映射整组留出 Spatial，TRAIN 有 30 个任务，DEV 有 10 个，逐 episode 核对了状态、动作、时钟、文件和 6,772 帧抽样双 RGB。两个分割之间初始双 RGB 相同的 episode 为 0。上游未提供逐 episode 原始重置 UUID。[LIBERO 固定发布](https://huggingface.co/datasets/lerobot/libero/tree/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4) · [RoboDojo 固定发布](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/1127bda94f172d757386945af3eb3a6754d1a4e6)

Corelab 的实际 CPU 解释器读取并 tokenize 三域各 4 个真实窗口，得到 H40、H50、H10。首版 OpenCV 环境在 Corelab 无法解码 AV1。接续版使用 PyAV 16.0.1，验证精确视频 PTS 后完成三个数据入口读回。现有 20 条目标 HOLD 用于本次输入核验。三条有历史完整评分和标签记录的 PLACE 仍待与新评价人口核对场景及采集谱系。

同一个上游模型的三个域视图构建成功。参数清单含 **427,932,672** 个动作专家参数、**11,943,936** 个 VLM LoRA 参数和每域 **2,165,792** 个独立 I/O 参数。按冻结参数 BF16、活动参数和 Adam 状态 FP32 计算，常驻模型与 Adam 至少约 **10.46 GiB**。实际训练峰值仍需计入梯度、激活和算子工作区。小型 JAX CPU 回归已验证源域闲置时的模型与 Adam 不变，以及保存恢复后的下一步逐位一致。真实满模型的梯度、恢复和 GPU 前缀均待资格验证。

[机读实绩、资源快照和数据哈希](DATA_AND_RUNTIME_ACTUAL.json)
