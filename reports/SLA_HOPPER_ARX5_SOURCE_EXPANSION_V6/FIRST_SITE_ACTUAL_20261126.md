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

## 首批新增 107 条 Core 封存

六批实际回读均已结算，108 个固定候选中 **107 条新增 TRAIN**。源端正式读取器逐回合独立读回 214 组首尾观测，完整 H50 为 **115,223 窗**。六项任务的新增数依次为 18、18、17、18、18、18。唯一排除是 `make_bread/episode_0000028` 的实际字节未通过固定 SHA-256，保留原失败文件与分母。与旧 200 合计为 **307 条**，达到首个双臂备用课程的数据端最小数量。单臂数据另行核验。

此版本 manifest 为 `fb33c0da28af0359b3b157d9288cd2b6fb3ac6c59d12d20f2e859a33f5a01317`，Core READY 为 `6204df0958d18ca063c3e8c5417a286ecee1d3fd5c5d1fe9e19ada2707d6669a`。已向 Hopper 独立工作盘启动同一版本传输，正式 RAW 绑定与模型读取器在收方重新逐文件核验。此回执写入时 Hopper 接收、优化器、PBS SOURCE 仍未完成。

双臂备用课程保持 Go2 / 单臂 / 双臂 / LIBERO = 16 / 0 / 96 / 16，再加 4 个实际完整 Go2 H40 样本。初始双臂批按旧 200 和新增 107 交替，实际批内按任务再按回合取样，RoboDojo 的原 R48 统计保留。实际 GPU 批资格需有 H200 有限梯度、全量恢复及显存余量。固定权重和采样分配做过 CPU 检查，未把它计作优化器访问。单独排期的 24 条单臂候选分配于咖啡与足袋两项，固定分册后读双相机和时钟。

## 单臂 Ville READY 与首发主课程

24 条固定增量样本已读取完成，咖啡与足袋各 12 条，下载 **217,647,328 字节**。12 条足袋片段的固定双相机时刻都显示抓持、抬起、送入塔口及释放。既有固定工具偏移应用于 12 条新轨迹，观测 EEF 最大位置差 3.54 mm、姿态最大差 0.35°。

采用实测 EEF 和实测夹爪字段构造记录后续状态参考。保留 20 Hz 原时钟、当前观测锚点和接下来 40 个真实位置，尾部支持不足的锚点排除。独立统计只用 TRAIN，缺失左腕用观测掩码表达。固定分配为 **10 TRAIN / 2 SOURCE_DEV**，分别 1,225 / 220 个完整 H40 窗口。Corelab 与 Hopper 已各用真实读取器读回 24 组双 RGB 边界观测及 `[40,32]` 学习张量。该版本只覆盖一个单臂任务，DEV 是预先固定的 episode 分离。版本 manifest 为 `23c2ff26ae17a67d268104c3e3a4d7cb1ff315502d7e5c5b65fd399b7fbdc381`。

实际首发优先主课程 **16 / 64 / 32 / 16 + 4**。使用 Ville 的独立归一化与掩码，新建确定性单臂 IO。Hopper CPU 以 MAIN3.8K 原 8 个私有参数叶核对了分支形状和确定性初始化，并检查 5 个活动 Adam 参数组的单次全局裁剪。该 CPU 检查未用模型真实梯度或更新科学权重。双臂完整版本仍在传输，首个新源 H200 真实更新及作业号以后续回执为准。

Hopper 已用正式宿主处理入口实际组成 64 条 Ville 样本，覆盖全部 10 条 TRAIN 轨迹，动作张量为 `[64,40,32]`。机身和对应腕部真实 RGB 各有效 64，另一腕掩码有效数为 0。此项为实际数据读取和模型喂入形状核验，优化器为 0 次。PBS 已按明示的 Hopper 服务器完成现役作业及项目可用授权查询，待双臂传输、读回及全配方宿主批一起通过后才释放资格作业。
