# SLA V6.5 · Recovery-aware Adaptation 1K 执行回执

绑定 [PR #528 固定协议](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/d7f4c1357fd9a48a6f3612c52c63172195a1da60/docs/protocols) 与完整 R4-200 父回执。正式训练从 step 0 起，1,000 步后执行科学评价。原 A250 和 SOURCE500 保留暂停状态。

## Recovery3 与固定训练输入

Corelab 回核 `V64_RECOVERY_TRAIN_02/04/06` 三份原封存、关闭的原生终态、接管后缀和逐段消费记录。窗口合计 **115**，其中 CARRY 26、LOWER 50、OPEN 12、WITHDRAW 27。实际接收并消费的 50 ms 区间为 **1,150**。重叠 H40 窗共有 **4,420** 个有效时间行，末端逐位保留实际支持。全部 115 窗对照前一实际夹爪命令，RGB 和状态用实录原生 tick 对齐。

三条后缀与 `EF5_DIVERSE_PLACE_008/010/012` 共享 TRAIN 场景几何，使用不同 reset 和独立闭环记录。对照 A93 全部 93 条示范和保护人口，未发现保护场景交叉。完整示范计数为 **93 条／7,916 窗**，Recovery 后缀独立登记。学生前缀和收尾未消费行未参与监督。

R3 便携清单 `362c8756c7b7ea62a539a441dcfa5a6bdf175f4eb92f79088ed7a7ec7a884991`。Hopper 使用原 Go2 图像、统计与训练读取合同重新读取全部 115 窗，六组起止边界另有指纹。回执 `03d3fe101028107560458c95845f362cf760c9af9b461977bf9ca27336b8accd`，115 个训练样本总指纹 `d838e2c4381ef71351a9056e67148ea1c374e07868dcfc3cfeb196f3a21c4114`。

固定 seed `655202611` 的 1,000 步银行通过所有行检查。每步 48 旧 Go2、46 新 Go2、2 Recovery、8 旧 RoboDojo、8 扩充 RoboDojo、16 LIBERO，及四个互异完整 H40 辅助样本。Recovery 的计划访问为 667／667／666 次。首行实际组批得到 Go2 `96×40×32`、RoboDojo `16×50×32`、LIBERO `16×10×32`、辅助 `4×40`。此阶段优化访问为 **0**。银行回执 `b8a29eea650acc2eb9f1ab8802b95c0d0a4ab88459c1e1fd7d54f2740ee9cb66`，二进制银行 `f4f3a37ff74639a40f2bc48a58d979e9e79d5f2f322256001c1d646e672ab6d5`，首批实际读取 `a3fe45e37cb92c77227e19f1f51e564d96540b92f097dd28360464a827536fbf`。

## Hopper 实际签发

PBS `737361.hopper-m-02` 已通过原 held-job 资源核对并释放。2026-11-27 **20:00:16 SGT** 确认状态 `R`，实际分配 1 张 H200、12 CPU、225 GB、3 小时。签发前确认旧正式谱系累计 500 update、工程消耗 15/24。V6.5 在同一作业内先执行 2 次独立工程更新，完成原值与冻结树、单独 Recovery 梯度、真实 batch、完整冷恢复验证，再从 R4-200、新 Adam、新 RNG 的正式 step 0 起跑。初次派发命令缺少现场 PBS server 环境绑定，在创建作业前退出。补齐固定 server 后仅提交了一次调度作业。

完整端点写入项目 scratch，预留约 **194.1 GB**，实际 64 MiB 持久化写入探针通过。签发 `4c265be366e162032116570ad088c0c7eca5a23535b160318e8979723a165c2f`，数据与实现绑定 `fa3bb01098fe04cef47be8f180ccee2584a535d98bc4b4122529612fbf1532d5`，写入容量 `f8eb653381f1a595e1b05883d25a2175aae2b8529e4db8b885002c25fe4de9c5`。PBS `R` 为现场运行状态。GPU 资格、实际优化步数和正式端点将在相应回执出现后结算。

## GPU 资格通过，正式训练已开始

Corelab 用现场读取器重新读取全部 Recovery3。其双 RGB、归一化状态、修订动作和逐点掩码的回执与 Hopper **逐字节一致**。实际 GPU 参数审计确认 Action Expert **427,932,672** 参数，三个独立 IO 各 **2,165,792** 参数。冻结的既有 LoRA **27,869,184** 参数保留原值。从完整 R4 固定输入生成的 step 0 原始 H40 与新参数分区逐元素相同。Recovery 隔离诊断真实读取两窗，各 40 个有效时间点，共享与 Go2 梯度 L2 分别为 **0.54045／0.33020**。该诊断没有更新参数。

两次实际工程更新均完成，完整冷恢复后固定 H40 相同，四个 Adam 计数均为 2。JAX 峰值约 **23.67 GiB**，运行额度约 **104.85 GiB**。资格回执 `5bd97f14c1062ef1c2042222264bf0317d787cf88aca5be6a316a1630e499a17`，step 0 与 Recovery 梯度回执 `f1050cad39b72ad87987a08d609988383ab935bbaa9e1b8c04e6f910060a424e`。工程累计 **17/24**。

同一 PBS 随即从 R4-200、全新正式 Adam/RNG step 0 起跑。**20:12:01 SGT** 实际正式账本 **49/1,000 步**，去前 5 步的每步中位数约 **4.98 s**。正式绑定 `e7348a2dbc242cf8687999139313c2e6f4dec0423794e3deca9810ee8c8ce55e`。这是还在推进的训练样本，尚无正式末端或闭环评价。

**20:16:57 SGT** 回读正式 step 100 完整封存，20 个成员，四个 Adam 计数和采样游标均为 100，冻结 LoRA 指纹与 R4 相同。前 100 步实际消费 **12,800 FM、400 个 H40 辅助位**，含 4,800 旧 Go2、4,600 新 Go2、200 Recovery。去前五步的更新中位数 **4.96 s**。检查时正式作业已推进至第 107 步。checkpoint `6c7e10652f3998314890d13f60bdf6fbc14f13ae5955d71ebee2299dea4a983e`，百步真实账本 `2f4b676a88e3c8ddba883c76bd7df774e89fa4a8ea87bba005dd3632051ef395`。这是训练完整性记录，科学评价仍安排在完成 1K 后。

## Corelab 并行结果

两张移动容器 QUAL 均在开爪前触发原安全门。完整 **0/2**，共 60,204 tick，STOP 后均为 0 tick。红卡已抓持并搬运，蓝卡止于抓持前。固定数据库存为 Embodiment **93/100**、Recovery **3/10**。容器接触与净空复算见 [Corelab 两卡终态](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/684c0ad6/reports/SLA_CORE_EMBODIMENT_FACTORY_V5/V64_CORE_SELECTED_REFERENCE_ACTUAL_20261127.md#两张-mobile-container-qual-在开爪前触发原安全门)。

## Go2 本体数据公开版

按照用户授权将 A93 和 Recovery3 上传到 Hugging Face。固定公开版本 `098afd49a39363d56be284a45b4500ed8882aacc`。内容共 **105 个文件、3,169,150,096 B**，含 54 条 HOLD、31 条 PLACE、8 条 Reach/Orient 和 3 条独立 Recovery 教师后缀。归档保留机身与腕部原始 RGB、R4 修订动作、逐点真实支持掩码、状态和实耗夹爪宽度。页面另有实际画面预览与逐回合索引。发布对象限于验收后的 Go2 仿真训练数据。

匿名核验固定版本中全部文件的成员清单、大小及对象指纹，独立下载两份索引和 PLACE、Reach、Recovery 三个代表性归档并逐文件通过 SHA-256。索引 `9d9396b503064b9fc94646db1e38c0c3a3ca1a1a24b254310644bd21fca43f56`。[Hugging Face 公开固定版](https://huggingface.co/datasets/leoyuer/go2-arx5-embodiment-a93-r3/tree/098afd49a39363d56be284a45b4500ed8882aacc)
