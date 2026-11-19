# ADAPT0 已进入单卡资格，三组 A40 性能对照封存

2026-11-20 01:36:52 SGT。以下状态来自提交后的独立站点读回。 [机器可读实绩](ADAPT0_R200_AND_MATCHED_A40_ACTUAL.json)

## R200 与 ADAPT

VANDA 完成 **200 条、210,398 个 RoboDojo TRAIN 窗口**的保管。复用 48 条原始录制，追加 152 条，传输的 353 个文件共 27,480,874,350 字节。Core 和 VANDA 通过生产读取器实读 Go2 4 条、RoboDojo 12 条、LIBERO 6 条。全部 22 条输入指纹一致，访问银行哈希一致。RoboDojo 继续用 R48 统计进行归一化。

| 角色 | PBS JobID | 核验状态 |
| --- | --- | --- |
| ADAPT0 零更新资格 | `1593869.stdct-mgmt-02` | `R`，实际单卡 A40 分配在 `GN-A40-021[1]`，已进入 128 窗口梯度测量 |
| phase25 独立后继 | `1593877.stdct-mgmt-02` | `H`，PBS 的 `afterok` 依赖指向 ADAPT0 |

ADAPT0 源码为 `15aea3b2f7fa078ed18685d2545c60211a4ccfe6`，执行 spec SHA256 为 `97680b02a54151c07166a52f6972fa2bf3e32888b530f5a25996f7a7bf398cc8`。两项作业各申请最多 3 小时。phase25 开始前检查真实封存结果、独立恢复和原谱系身份。当前测量正在运行，正式优化更新为 **0**，128 窗口梯度完整回执和 ADAPT0 独立恢复尚未封存。PBS 自行处理依赖。 [资格入口源码](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/commit/15aea3b2f7fa078ed18685d2545c60211a4ccfe6)

## A40 性能与精度

三个工程作业均为 PBS `F/0`。三组共 **90 次优化更新、11,520 次窗口访问和 360 次训练内完整 H40**。每一步的绝对步号、学习率、访问游标和样本哈希一致。每组前 10 步暖机，统计后 20 步。

| microbatch | PBS JobID | 墙钟 p50 / p90 | 相对 microbatch 1 吞吐 | 状态数组 SHA 相同 |
| ---: | --- | ---: | ---: | ---: |
| 1 | `1593682.stdct-mgmt-02` | 50.08 / 50.28 秒 | 1.000× | 140/140 |
| 2 | `1593796.stdct-mgmt-02` | 44.26 / 47.67 秒 | 1.132× | 5/140 |
| 4 | `1593797.stdct-mgmt-02` | 35.13 / 35.18 秒 | 1.426× | 5/140 |

microbatch 4 达到暖态吞吐门。完整状态未逐位复现，原 H40 和动作输出差异仍待固定面板复验。正式训练尚未采用新实现。 [真实账本核对源码](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/commit/1c611d5dfa77f1f1d754fd69ad213752784b8a48)

## 第二批 Go2 TRAIN

截至 01:35，第二批 **2/6 请求已开始，1/6 到终态**。F1 红左正确抓持并抬升最高 59.6 mm，搬运阶段 `x5_link6` 与非目标蓝块接触触停。原生分母 32,458 tick，停后积分 0，未完成入区或释放。F1 蓝右仍在运行，本批新准入示范为 0。 [Core 后台与准入回执](TARGET_SECOND6_NATIVE_AND_ADAPT_PBS_ACTUAL.md)
