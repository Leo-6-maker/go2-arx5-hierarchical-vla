# Hopper V6 执行与公共 ARX5 首批核验

本轮执行以 [PR #522 固定提交](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/d7c74953109374b69906036a0e2c6f39a32b3aa7) 为准。五个源族均单独核查，未下载任何完整公开仓库。

## Hopper 和原生对照

`735204.hopper-m-02` 正常退出，已独立恢复 STEP1000 的全部参数、Adam 和 RNG。1,000 次实际更新，热态 p50 5.0866 秒，p95 5.1052 秒。访问为 64,000 Go2、32,000 RoboDojo、32,000 LIBERO 和 4,000 个有真实消费支持的辅助槽。

MAIN3.8K 在相同原生面板为 HOLD 2/2、PLACE 抓持 3/4、完整 PLACE 0/4。M1000 的两张 HOLD 和四张 PLACE 均在抓持前因同一臂段触桌触发原安全停止，完整成功 0/6。原 V4 2K 未通过接力条件。V6 首批候选父为已完整封存的 MAIN3.8K。

R0 两次 PBS 实际作业为 `735374.hopper-m-02` 和 `735409.hopper-m-02`，分别完成 7 和 3 次工程更新。均未得到完整设备 trace，按 R0 有界重试规则关闭。已有每域墙钟测量仍可复现，设备 kernel 利用率和 Remat 成本缺测。[完整端点、原生分母和 R0 回执](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d806ab813e34ee1e3f3a6645409fa4c2402ece7d/reports/SLA_EXPERIMENT_PROGRESS/V4_HOPPER_HIGH_BATCH_20261126.md)

## 公共数据实读

| 来源 | 固定实读 | 许可与结果 |
| --- | ---: | --- |
| RoboDojo Real ARX X5 | 4 条早期实机探针，3 条可继续引用 | Apache-2.0。沿用既有 SOURCE_ARM_LOCAL_V1 实际读取器，双臂位姿和夹爪逐行符合后一帧记录，原 H50 按完整支持截断。既有 RGB 读取器三相机通过。30 Hz 为名义时钟，所读原始文件没有独立采样时间戳。 |
| Ville 原作者三份单臂数据 | 6 条实机、2,553 行、两路 RGB | 三份原库各标 Apache-2.0。20／20／50 Hz 原生时钟，真实视频 PTS 与数值行对应，最大差小于 1 μs。两条足袋轨迹提供实测 EEF 字段。 |
| RoboTwin ARX-X5 转换 | 4 条仿真、797 行、三路 RGB | 转换数据标 Apache-2.0，原发布标 MIT。固定的是 `randomized_500`。逐行复算实际 EEF 引用下一帧，最大数值误差 0。原格式四元数为 wxyz，归一化夹爪值 1 为张开。 |
| Table30 / v2 | v1 复用 8 条既有审计。v2 范围读取 55,214 字节 | v1 许可未确认，旧准入结论保留。v2 实读一份任务索引和七份回合元数据，抽取到的首条是 `rc_w1_7`，未建立可用单臂 ARX5 配对状态／视频。两个版本均未准入。 |
| OpenNeoData arx5_single | 0 条 | 公布 CC BY-NC-SA 4.0，实际数据文件需门控访问。现场按固定 revision 读头返回 401，未接受条款或读取数据。 |

Ville 的 [作者插件固定版本](https://github.com/villekuosmanen/lerobot-arx5/tree/d87b5f1d8bba052abf4182cbd27f20a7cc66f76f) 明确区分关节位置观测、位置命令和夹爪宽度，并有 leader 夹爪比例分支。现阶段未证明这份插件提交就是旧数据的写入版本。以 [SDK 的固定 X5 模型](https://github.com/real-stanford/arx5-sdk/blob/36eabcb618c2668dfdff228b18b30507fde71784/models/X5.urdf) 做实际 FK 检查，使用第 0 条拟合固定末端偏移，第 1 条独立复验。留出轨迹的 XYZ 中位误差从 14.47 mm 降至 1.51 mm，最大 2.90 mm。姿态最大误差 0.049°。原夹爪字段保留，物理标定和轨迹成功身份尚需独立核实。三份数据的时钟保持各自原样。

## 首批扩充与 Corelab

RoboDojo 固定 108 条新增候选，六项任务各 18 条，绑定原始路径、字节数及 LFS SHA-256。与既有 200 TRAIN、12 DEV 和原 12 TEST 元数据逐一排除重叠。固定名册 SHA-256 为 `476dbe0e03ed309170ba76a333f19ff1b1fc18a7e2ac3bafff4bbff1696ec496`，新原始文件总上限 19,688,586,999 字节。Corelab 已启动有界 CPU 下载与本地验证，每批 18 条、3 个 worker。待完整读取与数据封存，通过的源窗口才进入正式版本。源端 36 条已独立封存，见下方增量。正式 SOURCE 的优化器访问和 PBS 作业仍为 0。

Corelab V5 与上述源队列独立。已确认 67 条历史合格示范，其中 64 条完成 Hopper 正式读取器回读。两张 GPU 已验证独立 NVIDIA EGL 上下文，原生多环境资格仍需实际物理和 STOP 隔离。Core 系统盘、新栈可选本地 HDD 当前分别约 108 GiB 和 534 GiB 可用，低于原控制卷空间门约 274 GiB 和 567 GiB，当前没有新生产物理积分。学生六卡已按原租约关闭。[V5 现场生产与数据回执](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/bf59ef27c8d0c140840467d242679b56496041dc/reports/SLA_CORE_EMBODIMENT_FACTORY_V5/FIRST_SITE_ACTUAL_20261126.md)

公开原始数据来源见 [RoboDojo](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo) · [Ville 咖啡胶囊](https://huggingface.co/datasets/villekuosmanen/pick_coffee_prop_20Hz) · [Ville 足袋](https://huggingface.co/datasets/villekuosmanen/drop_footbag_into_dice_tower) · [Ville 电饭煲](https://huggingface.co/datasets/villekuosmanen/rice-cooker_step_2) · [RoboTwin 转换](https://huggingface.co/datasets/Traly/Robotwin2.0-lerobot) · [RoboTwin 原发布](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) · [Table30](https://huggingface.co/datasets/RoboChallenge/Table30) · [Table30v2](https://huggingface.co/datasets/RoboChallenge/Table30v2) · [OpenNeoData](https://huggingface.co/datasets/NeoteAIEmbodied/OpenNeoData)。

## 增量封存与原留出对账

原四条早期探针只先与 R200 TRAIN/DEV 核对，加入原留出名册后发现 `insert_tubes/episode_0000001` 属于原 TEST。该条已经被离线解码并计算 H50，原 TEST 的这张不能用于本轮盲评。原档与技术记录保留，数据进 V6 TRAIN／DEV／盲评的访问为 0。首报的原 TEST 读取口径在此纠正。先排除全部 12 条旧 TEST，再生成的固定 108 条名册与其交集为 0。该批仅用公开元数据和稳定哈希选取。分册对账 SHA-256 为 `ada5838f5bd4e7d09bcea6e206159f27f707c825fa7ff2f8786c2eb9e7330c45`。探针代码已改为读取完整排除清单后再开文件。

第一份新源端版本 `SOURCE-RD-V6-036` 已经封存并用正式 SourceDataset 在 Corelab 独立回读，新增 **36 条 TRAIN**、38,729 个完整 H50 窗口、72 组实际首尾观测，六个任务各 6 条，0 重复排除。沿用既有 R48 统计。manifest SHA-256 为 `abe4e3cc463907b2ebaf79652ce1f68f3f274a7209bf317aa9a3ae84861cd3b0`，READY 回执为 `d15cffb7ac88de84f27de3e4ecfa8ce92e08ca9357b1aceadd183fc6631d6d8a`。累计前五批 **90/90** 通过旧 codec，96,311 个窗口，后 54 条尚待源端合并封存时复读。两站实际优化器访问均为 0。

Ville 的两条足袋轨迹已生成 242 个原生 20 Hz、40 行、7 维的候选窗口。使用模型产生动作前的实测 EEF 锚点，保留原数值时间戳、夹爪带符号原值和真实支持 mask。历史 [作者写入器](https://github.com/villekuosmanen/lerobot/blob/bdbbb8511a49131b11c263a3c28f5f5c9dc0ac83/lerobot/common/robot_devices/robots/arx5.py) 的七维位置动作和 EEF 观测字段与足袋数据结构一致。双相机固定时间画面显示两条轨迹均有实物抓持、送入塔口及滑到底部。旧记录提交的精确运行身份及物理夹爪标定仍待复核。候选窗口暂不计 TRAIN。

Hopper 的 18:33 配额报表实读 `/scratch` 已用 930.4 GB、硬上限 1 TB。独立的用户工作盘仍有约 170 GiB 可用，可隔离新公共源传输，正式训练需结合完整检查点逐段核对实际配额。
