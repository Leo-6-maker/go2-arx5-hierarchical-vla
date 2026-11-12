# Codex执行交接：Corelab并行采集＋H200后训练 V3

日期：2026-11-13，Asia/Singapore。
主计划：[SLA_CORELAB_PARALLEL_DATA_H200_V3_20261113.md](../../protocols/SLA_CORELAB_PARALLEL_DATA_H200_V3_20261113.md)；[机读矩阵](../../protocols/SLA_CORELAB_PARALLEL_DATA_H200_V3_20261113.json)。

## 可直接交给执行端的启动指令

承接Go2＋ARX5＋冻结WBC项目。用户已明确允许Corelab上多环境并行采集，并要求关注采集效率、避免CPU/GPU反复搬运。按V3计划实际推进，交付数据、checkpoint和闭环成绩；不要停在阅读文档或CPU检查。

### 1. 先核对真实现场，然后独立推进两站

- 读取AGENTS、V3 MD/JSON及当前科学状态。最新核验执行基线为PR #380的aaf80b4eda7ecb2816c0c251d3535a4e54ea62c9，V2计划为b07936899b10f196653f617f95c5cd8b124e5e99；开工时检查后继提交并绑定实际source。
- main尚未含完整执行栈。在独立worktree实现，保留活动capsule。不要把本计划中拟议session/pool当成已有CLI。
- 核对旧W0的unit、组、checkpoint和终态。00:02 SGT的WAIT只是快照；旧期限为11/13 09:39首次准入、18:39总期限。超期保留expired，已完成不重做，UNKNOWN不重投。
- Corelab多世界采集已有许可，不再为2/4/8 actor逐次询问。每个世界独立owner，不能共享一个live world或删除安全锁。H200按既有有效授权与实际allocation，现场不可用时Corelab继续。
- 核对本项目实际可用CPU/内存/GPU/节点存储；不碰他人进程、不在AMD做重计算、不更改系统服务。旧S1→S2→G人口、顺序、失败及身份保持。

### 2. Corelab首先解决已知采集接缝

从fusion_paused_observation、native_body_camera_capture_v1、native_camera_transfer_v1、native_collection_performance_v1、stable_task_prefix_v1开始：

1. BodyBaseCameraCapture当前没有进入SourceCameraSession缓存分支；接上body/wrist持久model/renderer和复用buffer，正确reset scene。
2. 观察直接从不可变内存snapshot进入policy／writer；移除PNG写盘再读回的控制依赖。必要hash/标定保留，在版本边界和异步writer处理。
3. 接通已有settle_atom事件就绪，保留pause ACK、native step、solver/gripper消费；不能直接删除固定wait。
4. writer按字节/帧双限额；slot必须等policy与writer都完成才能回收，或给writer受限独立副本。所有引用shard sealed/flush/checksum后才atomic commit。
5. 32个已保存状态对照后，用4个完整TRAIN episode核对语义。随后隔离2/4个原生actor，实际生产和计时。首版可以跨episode重启进程，先拿到episode内持久化和并行收益。
6. 每actor隔离MuJoCo、solver/controller/腿历史、ROS域/端口、时钟、RNG、renderer、可变cache、tmp与输出。核算所有native/BLAS/renderer线程，避免超额并行。
7. 专家采集不加载SLA。学生闭环再用每实际GPU一个常驻服务、batch1/2/4与独立请求身份。

当前原生CPU控制保持CPU驻留。CPU图像一次必要H2D后，GPU完成模型与flow；只返回紧凑H40及必要raw审计。经典EGL NumPy readback不算零拷贝，不能把GPU→host→GPU图像回环标成优化完成。优先持久CPU renderer建立产线，GPU device renderer另证实。

### 3. GPU训练与教师并行完成

- 在已有V2 batch/runtime上修复，不重建trainer；把旧500-step/16000-visit硬编码改成明确配置和可恢复游标。
- micro1/8/16比较保持有效batch32和同fixed32随机bank。先做真实SLA梯度、连续4对2+2恢复及新进程普通RGB重建，CPU synthetic检查不代替。
- 模型、梯度、优化器常驻GPU；persistent loader与当前/下一批有界预取，device内累积指标/梯度，不逐microbatch float(loss)或全局阻塞。每update保留有限性终态，失败不更新权重/data cursor。
- “一次上传”按sampling visit和cache驻留期计量，不为去重图像改变重复访问、noise、time、dropout或采样权重。
- teacher同卡走经过device/layout/stream/lifetime检查的DLPack，不绕NumPy；或冻结特征按有效anchor/shard分块预取、GPU padding。2 MiB/anchor的特征不能默认全驻D1000单卡。
- finite profile用transfer guard与profiler，计时等待实际设备完成；训练热路径不回传权重/梯度，日志与checkpoint成本单列。

### 4. 动作／数据合同在扩量前落实

GRIP_LOGIT_V2保留1e-4端点偏差定义并做原生专家夹持检查；R1_500要取得真实消费结果。EEF_STREAM保留H40/50ms/2ms和中间路径，先比较2s/0.5s prefix；reference TTL0.4s及≤0.35s续期独立运行，不能用policy cadence替换。

同步RGB/state10Hz、训练anchor2Hz、命令20Hz；绑定native step，不能重复帧冒充密集记录。监督是实际消费的专家nominal，跨query先还原共同坐标再重锚。未来TCP/错误学生命令/未消费suffix不自动成为标签；终末2s保持必须真实执行。

### 5. 从首批数据立即训练，按真实结果扩量

- D100_new：100条完整合格示范，最多200尝试；完成即500×B32短诊断＋10个TRAIN条件，另计pilot。
- D300_new：新增200条、最多400尝试；10姿态层×5物体×2任务保持同支持域，PLACE未ready则明确单任务版本。
- D300四配方各3k×B32，随后同recipe D100及条件性D1000；新增700条最多1400尝试。
- 抽样先平衡任务/episode/阶段，TRAIN-only统计；不拆相邻帧进TEST，不把旧稀疏TRAIN100改名为新数据。
- 选中配方若仍欠拟合，按预定3k→6k→12k→20k预算曲线；3k之后尾段固定到20k、末LR为第3k已用LR的20%。确认seed只截取同一固定曲线到所选U，不重缩cosine。
- 记录U_probe、U_selected和actual_completed_updates：29k+(U_probe−3k)+2*(U_selected−3k)，max80k/B32。未选中的探索照实计入。
- 纠正最多400 TRAIN rollout、目标至多200专家恢复片段；nominal100%与80/20纠正两臂各1k×B32匹配。最后两个seed匹配选中U和纠正配方，独立TEST。
- 就绪新技能10–30条完整种子立即500步首策略；不等待D1000或SLA胜GR00T。

### 6. 用真实产量选并行度，下一项工作不能只写“测试通过”

性能组固定8条件×2重启，原串行/优化1/2/4/条件性8/16各16条，max96；2 actor≥1.5×、4≥2.5×是相对优化单actor的工程目标。相邻倍增收益<15%就采用更优档位并定位瓶颈。额外至多32条与已授权learner重叠检查共享I/O，实际更新不暗增。

首个回报给出：最新W0终态、采用的source/资源、至少两条完整新episode或明确首次失败、实际模型更新/普通RGB包或现场GPU阻断；同时给attempted/valid/success/usable、示范每小时、耗时分解、传输bytes/calls与峰值内存。后续交付D100/D300和候选闭环成绩，别用文档/CPU检查数替代。

按V3有限任务端点和实际资源持续推进已授权工作。新性能或科学问题提出一项有证据改动，保留旧失败，不无限重启、不新增逐函数审批流程。公开GitHub只发布代码、配置和非敏感回执；原始图像、权重和vendor资产走既有授权存储。
