# Corelab Embodiment Factory V5 执行手册

配套 [计划](SLA_CORE_EMBODIMENT_FACTORY_V5_PLAN_20261126.md) · [机读登记](SLA_CORE_EMBODIMENT_FACTORY_V5_REGISTRY_20261126.json)。现场执行入口，不是已经运行的服务。复用当前执行版本及原生控制链，普通缺陷自主修复，不复制整个工作区/重建控制器/逐卡等待人工GO。

## 0. 先读现场与立即生产

- 核对 #519/#520 最新实际报告、Core当前active/queued/closed角色、唯一物理owner、原V4父模型面板、Hopper job及1K/2K真实完整状态。已读最新 [17:09 564步][hop]、[容器和N2][container] 仅作基线。健康任务继续。未知owner先对账，禁止重复提交或重启。
- 现在即可准备已合法N1的原未启动TRAIN和新Scene冷态工作；若当前学生拥有world先让其正常结束，不为采集取消。无需等待Hopper端点，也不因N2工程卡住N1。CPU离线编译、已有69条准入、数据编码与接收端准备可并行。
- 用现有现场授权入口核真实用户、GPU/CPU/主存/磁盘、租约与截止。举例只读探针（在现场合法shell执行，不在本次文档发布执行）：`nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu --format=csv`、`nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv`、`lscpu`、`/proc/self/status` 的Cpus_allowed_list、`/proc/meminfo`、cgroup CPU/memory限额及实际输出盘 `df`/站点quota。显卡index与EGL枚举不能直接等同；读实际renderer/driver/设备、子树及GPU渲染进程，验证真实设备绑定。只使用原授权路径；不重启Vanda、改SSH/网络/系统包、不清理旧文件。

## 1. 复用现有源码与场景资产

已有 [geometry][geometry] 仅对两个cube共同平移；[producer][producer] 用旧F1/F2模板、单world lease、同步child与单catalog。沿实际运行SHA选择现有编译器/teacher/admission/reader，版本化最小修改：
- Scene adapter：可组合 object/fixture/arena资产，统一world/mount/单位，固定license/源revision/哈希；实际编译后检查body/geom/joint，生成真实场景哈希；渲染设备和相机身份确认。先少量LIBERO资产，必要RoboCasa局部fixture，许可不明确不复制。已有场景第一，RoboDojo仅任务语义。
- Task adapter：每类任务独立语言、目标实体/region/合法接触、完成谓词与原生评分。Reach不冒充PLACE；Push不借助强制夹持；Container需要稳定承托、实际脱离及撤离；Draw/Insert保留未见评估。
- Teacher：复用历史实际成功Go2阶段路径；源demo只借鉴对象局部坐标、阶段意图。先静态全路径的碰撞/可达性、后实际WBC全路径验证。不要仅导入XML就声明教师可行。移动容器新任务与原站姿任务分开，旧5mm/5°失败保留；新任务依照冻结M_FREE控制和事先声明的原生移动、安全及成功规则。
- 初态/姿态/场景分组：同scene多指令有意复用，所有相关轨迹同split；实际几何而非seed去重；旧DEV/TEST只保护身份。每新family先绑定2个资格/2个VAL独立场景，已有同合同可引用；训练几何单独封存。
- 旧V4 PUSH/CONTAINER条件若任务/栈一致则一对一映射原ID；新M_FREE不冒用原站姿条件。历史失败不重跑同ID。新任务原生评估先封存，不因结果挑选几何或指令。

## 2. 采集执行与自动进度

从原52兼容集＋69候选重读实际成员和标签，按TRAIN-only真实command支持、split、几何与事件去重；如17新增不能兼容，给实际理由，优先补对应缺失覆盖。将机读registry范围展开为预封存实际条件manifest（每波/每小批先绑定，不生成后挑好结果）。Wave100目标累计69/8/8/8/4/3，Wave200累计69/20/25/40/26/20，legacy无需补则请求容量0；按配额缺口调度不同ready bucket，而非等所有桶齐。第一批每族已有合法教师路径即可用N1采，冷筛与编译异步做；不等N2/N4。DRAWER、INSERT及新组合不进通用Adapt训练，除非另注册少样本研究。旧V4已完成条件原账本保留。

生产worker只领取一个已绑定未开始的condition；协调者记录intent/lease，未知写入独立对账。所有尝试（失败/技术/静态拒绝/未启动）分别记录。完整任务实际成功、无原安全STOP且标签准入才入TRAIN；每个episode只归一个主桶。Recovery单独：首10/累计40，真实TRAIN student安全可恢复状态、独立重建起始与前缀，Teacher控制后实际命令标签；部分轨迹只算恢复片段。无GT在线student，教师数据不能回流DEV/TEST。

每4条新增准入、6小时有变化、或波终态生成不可变增量快照，按成员hash去重、真实reader和标签支持校验，跨站收方atomic READY；现役Hopper不换bank；实际下一版本opt访问另记。只走合法已有直传或私有资产渠道，Vanda暂停，不假定Core能直SSH。若上传路径缺权，先保留Core封存、报告精确阻断。

## 3. 双GPU/multiworld 启动与资源

实际执行树必须有多world pool显式授权/注册与独立子lease；不能改旧锁名或删除OwnerLock。先在合法N1跑一条代表性完整任务，记录完整子树PSS/RSS、cgroup CPU时间、GPU渲染峰值、实际写盘字节与冷/热耗时；旧单条件mem≥13GiB或其他历史门不等于总池有足够空间。统一资源预算先覆盖所有将启动的world、系统/IO/学生余量、在途trajectory和staging。按可分配物理CPU核划不重叠集合，固定WBC/BLAS线程数；不假定2核/world足够，也不启动超配。

```text
one existing coordinator / catalog owner
  -> world leases with unique CPU set, ROS & non-ROS IPC, clock, rng, storage
  -> GPU-role A: 1 teacher world initially
  -> GPU-role B: 1 teacher world initially
  -> each own MuJoCo data + WBC/MPC process tree + EGL context
  -> per-world atomic closed result -> single catalog writer
```

实际GPU UUID/renderer需匹配，不能只改CUDA_VISIBLE_DEVICES。每world独立ROS domain之外检查全部非ROS端点、shared memory、控制文件、tmpfs、可写缓存和日志，确认没有共享全局名称。只读资源可共享。单worldSTOP后该world新增publish/install/grant/integrate为0，其他独立world真实继续；池级故障隔离、不可安全控制的共享错误关闭受影响池。使用现有Journal/onceIO和进程身份，不新增大型scheduler。

1. 先读现有D3归档与旧N2/N4 protocol [固定资格][n2plan]，复用已证实的STOP部分，匹配当前profile。新双卡1+1需要各卡真实独立短匹配/完整接触、交叉world/CPU/GPU角色、STOP/故障隔离。重点记录controller实际选择的reference、solver状态/时钟和mount/TCP同tick分解；旧F1差值不能通过平移时间或改变阈值解决。尽量复用旧合格输入与对应条件，不重做无关已过卡。
2. 短匹配及完整接触按原predeclared容差，N2只有当前profile适用的完整证据才能生产。旧36 N2/28 N4容量不重置；已有工程尝试与费用保留，确需额外新拓扑资格另登记固定有限批。N2失败暂留N1，离线工作及Hopper继续。
3. N2合格后用实际任务的准入率、合格示范/h与匹配同期N1验证价值，目标1.5x；若GPU空闲但CPU或主存不足不扩。N4尝试2+2，原完整隔离/接触及吞吐目标1.25x N2。逐world申请子lease，先分配后派发；每卡渲染不超过实际资源。
4. 评价到达：保留原公平调度，学生等待≥30min或12个Teacher终态则让行；健康在途Teacher正常结束。不具备混合资格时排空所有冲突world，再跑原学生独占窗口。具备P4证据后允许GPU A单学生推理＋B单Teacher，P5两Teacher＋一Student需同卡双world和混合资格。两组P4正常/STOP交换（最多4工程启动），两组P5交换停止学生/教师（最多6）。STOP/peer继续、实际消费、正常完成及资源同时检验；工程学生使用固定诊断模型/独立输入，不作为V4正式条件。任何两学生batch推理后继另注册，当前不要求。
5. 若遇GPU不足，优先保留有实际任务价值的N1/评估；不挪其他用户GPU，不为Teacher申请无实际工作的H200。

## 4. 性能优化的小步实施

当前真实profile先分解warmup、推进、参考accept/selected/ACK、渲染、写盘及admission；原138.89s历史短卡的53s warmup、49.5s回调与44.7s advance有嵌套，不加总。三项各最多4个新增匹配工程启动（两个代表条件各参考与候选一对，可复用精确旧基准减少新启动）：

A. Persistent world：先同scene冷启动和warm reset，必须真实清除WBC/MPC/solver、时钟代次、previous ref、history、STOP/RNG与物体状态；新scene asset可只读缓存，不假定跨mjModel热重置；验证新episode正常初态、接触和实际消费。如果不一致保留cold，不继续“快但脏”的数据。

B. Event-driven ACK：只有controller实际selected-reference身份已证实，才替代有证据的冗余fixed wait。保持原2ms、原H40/clock及控制确认、STOP语义，不缩短必要等待或通过改变仿真速度优化墙钟。先固定输入对照，再真实完整接触。

C. Async RGB/IO：每捕获真实world/tick独立复制，2个CPU编码/审核worker起，最多4，队列每world≤2；满背压，不能丢帧、复用变动缓冲、改变图像/时间或隐藏失败。关键控制/STOP原同步账本保留。单catalog writer原子提交，只对不可信/变化成员重核。

不同时改变N2拓扑与reset/ACK/图像，旧失败全部保留。每项报告端到端完整封存耗时、内存/渲染峰值、任务质量；优化无收益就回退，不阻塞数据生产。新版本真实需要更广回归则登记有限后继，不无限反复试相同症状。

## 5. 真实数据合同

记录实际RGB/观测tick/语言/current EEF/grip、frame/标定、actual previous command、每原子accepted/selected/consumed、原生实测基座/臂状态、物体/接触/阶段/终态。当前学生输入保持原body/wrist、EEF/gripper及语言，GT及后续状态只作教师/评分审计。真实消费command按当前query固定frame和50ms重建H40，末端无支持时间显式mask并应用于FM noisy input/loss；完整aux只用实际连续支持40行。不能以未来achieved、未消费nominal、被拒绝指令、尾值复制冒充训练。记录不同skill的完整H40覆盖，避免为了4个aux槽删掉任务尾端关键标签；后继按真实支持与分层采样绑定。源域统计、OSC/achieved语义独立。

所有新scene、teacher版本、split、资产许可、真实任务评分和失败原件、实际接触/STOP保留。完整示范按自身技能目标判定，短纠正只记RECOVERY_PARTIAL或RECOVERY_COMPLETE（若真实完成该纠正目标），不混作200。VAL/CONF及工程不入TRAIN。实际数据在CoreREADY后独立校验目标端读入，Hopper新冷边界按原训练参数/授权决定接入；旧1K/2K不热改。

## 6. 开工与收尾

第一优先：Core实时资源、已有69准入、现有N1实际继续；并行构建S1–S4最小场景及每任务完整教师路径。Hopper旧作业继续/R0按旧计划，非依赖。接着N2双卡1+1、成功则正式生产；P3、P4/P5以实际收益启用；Warm reset/ACK/I/O独立验证。每波提供真实尝试/准入/几何/有效H40/恢复/耗时及跨站READY、实际GPU/CPU利用与资源不足理由，新的数据训练访问证据。报告原Hopper最新更新/端点，但不要重复旧健康提交。72h有限生产挂载期限服从真实授权，按原安全收尾；普通工程自主修复，不为新scene逐卡等待GO。历史存储由用户清理，无系统修改和实机动作。

```text
Codex 开工指令：
继续 Hopper 既有训练并行 Corelab 采集。先现场对账与硬件实际读回；按已资格N1立即推进可行任务和69准入，CPU并行轻量Scene Factory。优先LIBERO资产及对象中心轨迹，冻结原MuJoCo/WBC，新的Container采用独立M_FREE任务而非放宽旧站姿。尽快补N2真实参考/状态/隔离资格，在双卡1+1后条件扩2+2，学生评价优先且混合负载单独资格。优化warm reset/事件确认/异步IO，失败回退不停止就绪N1。目标Adapt100、Adapt200+Recovery40，按唯一catalog分批封存并交Hopper下一版本实际消费。保留旧V3/V4人口和失败；不重启Vanda、不删除数据、不重复未知/健康作业。按真实资源与有界工作块自主推进，持续提交实际回执，不只写计划。
```

[hop]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d175383b07cabb701492222bc333809d57ca5830/reports/SLA_EXPERIMENT_PROGRESS/V4_HOPPER_HIGH_BATCH_20261126.md
[container]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/d175383b07cabb701492222bc333809d57ca5830/reports/SLA_EXPERIMENT_PROGRESS/V4_STEP100_CONTAINER_20261126.md
[geometry]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/scripts/sla/core_hopper_geometry.py
[producer]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1fee437d12d9a79deff8e2be0175da3f148dd48e/scripts/sla/core_hopper_collection.py
[n2plan]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4fe2c8cf1b69c168b2e3079e061bb7c702089cd3/docs/protocols/SLA_CORE_MULTIENV_COLLECTION_V1_20261122.md
