# Hopper V6 执行与公共 ARX5 首批核验

本轮执行以 [PR #522 固定提交](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/d7c74953109374b69906036a0e2c6f39a32b3aa7) 为准。五个源族均单独核查，未下载任何完整公开仓库。

## Hopper 和原生对照

`735204.hopper-m-02` 正常退出，已独立恢复 STEP1000 的全部参数、Adam 和 RNG。1,000 次实际更新，热态 p50 5.0866 秒，p95 5.1052 秒。访问为 64,000 Go2、32,000 RoboDojo、32,000 LIBERO 和 4,000 个有真实消费支持的辅助槽。

MAIN3.8K 在相同原生面板为 HOLD 2/2、PLACE 抓持 3/4、完整 PLACE 0/4。M1000 的两张 HOLD 和四张 PLACE 均在抓持前因同一臂段触桌触发原安全停止，完整成功 0/6。原 V4 2K 未通过接力条件。V6 首批候选父为已完整封存的 MAIN3.8K。

R0 两次 PBS 实际作业为 `735374.hopper-m-02` 和 `735409.hopper-m-02`，分别完成 7 和 3 次工程更新。均未得到完整设备 trace，按 R0 有界重试规则关闭。已有每域墙钟测量仍可复现，设备 kernel 利用率和 Remat 成本缺测。[完整端点、原生分母和 R0 回执](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d806ab813e34ee1e3f3a6645409fa4c2402ece7d/reports/SLA_EXPERIMENT_PROGRESS/V4_HOPPER_HIGH_BATCH_20261126.md)

## 公共数据实读

| 来源 | 固定实读 | 许可与结果 |
| --- | ---: | --- |
| RoboDojo Real ARX X5 | 4 条实机、4,100 行、3,900 个原 H50 窗口 | Apache-2.0。沿用既有 SOURCE_ARM_LOCAL_V1 实际读取器，双臂位姿和夹爪逐行符合后一帧记录，原 H50 按完整支持截断。既有 RGB 读取器三相机通过。30 Hz 为名义时钟，所读原始文件没有独立采样时间戳。 |
| Ville 原作者三份单臂数据 | 6 条实机、2,553 行、两路 RGB | 三份原库各标 Apache-2.0。20／20／50 Hz 原生时钟，真实视频 PTS 与数值行对应，最大差小于 1 μs。两条足袋轨迹提供实测 EEF 字段。 |
| RoboTwin ARX-X5 转换 | 4 条仿真、797 行、三路 RGB | 转换数据标 Apache-2.0，原发布标 MIT。固定的是 `randomized_500`。逐行复算实际 EEF 引用下一帧，最大数值误差 0。原格式四元数为 wxyz，归一化夹爪值 1 为张开。 |
| Table30 / v2 | v1 复用 8 条既有审计。v2 范围读取 55,214 字节 | v1 许可未确认，旧准入结论保留。v2 实读一份任务索引和七份回合元数据，抽取到的首条是 `rc_w1_7`，未建立可用单臂 ARX5 配对状态／视频。两个版本均未准入。 |
| OpenNeoData arx5_single | 0 条 | 公布 CC BY-NC-SA 4.0，实际数据文件需门控访问。现场按固定 revision 读头返回 401，未接受条款或读取数据。 |

Ville 的 [作者插件固定版本](https://github.com/villekuosmanen/lerobot-arx5/tree/d87b5f1d8bba052abf4182cbd27f20a7cc66f76f) 明确区分关节位置观测、位置命令和夹爪宽度，并有 leader 夹爪比例分支。现阶段未证明这份插件提交就是旧数据的写入版本。以 [SDK 的固定 X5 模型](https://github.com/real-stanford/arx5-sdk/blob/36eabcb618c2668dfdff228b18b30507fde71784/models/X5.urdf) 做实际 FK 检查，使用第 0 条拟合固定末端偏移，第 1 条独立复验。留出轨迹的 XYZ 中位误差从 14.47 mm 降至 1.51 mm，最大 2.90 mm。姿态最大误差 0.049°。原夹爪字段保留，物理标定和轨迹成功身份尚需独立核实。三份数据的时钟保持各自原样。

## 首批扩充与 Corelab

RoboDojo 固定 108 条新增候选，六项任务各 18 条，绑定原始路径、字节数及 LFS SHA-256。与既有 200 TRAIN、12 DEV 和原 12 TEST 元数据逐一排除重叠；原测试标签未读取。固定名册 SHA-256 为 `476dbe0e03ed309170ba76a333f19ff1b1fc18a7e2ac3bafff4bbff1696ec496`，新原始文件总上限 19,688,586,999 字节。Corelab 已启动有界 CPU 下载与本地验证，每批 18 条、3 个 worker。待完整读取与数据封存，通过的源窗口才进入正式版本。首报时新增正式 TRAIN 准入、SOURCE 实际优化器访问及 PBS SOURCE 作业均为 0。

Corelab V5 与上述源队列独立。已确认 67 条历史合格示范，其中 64 条完成 Hopper 正式读取器回读。两张 GPU 已验证独立 NVIDIA EGL 上下文，原生多环境资格仍需实际物理和 STOP 隔离。Core 系统盘、新栈可选本地 HDD 当前分别约 108 GiB 和 534 GiB 可用，低于原控制卷空间门约 274 GiB 和 567 GiB，当前没有新生产物理积分。学生六卡已按原租约关闭。[V5 现场生产与数据回执](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/bf59ef27c8d0c140840467d242679b56496041dc/reports/SLA_CORE_EMBODIMENT_FACTORY_V5/FIRST_SITE_ACTUAL_20261126.md)

公开原始数据来源见 [RoboDojo](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo) · [Ville 咖啡胶囊](https://huggingface.co/datasets/villekuosmanen/pick_coffee_prop_20Hz) · [Ville 足袋](https://huggingface.co/datasets/villekuosmanen/drop_footbag_into_dice_tower) · [Ville 电饭煲](https://huggingface.co/datasets/villekuosmanen/rice-cooker_step_2) · [RoboTwin 转换](https://huggingface.co/datasets/Traly/Robotwin2.0-lerobot) · [RoboTwin 原发布](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) · [Table30](https://huggingface.co/datasets/RoboChallenge/Table30) · [Table30v2](https://huggingface.co/datasets/RoboChallenge/Table30v2) · [OpenNeoData](https://huggingface.co/datasets/NeoteAIEmbodied/OpenNeoData)。
