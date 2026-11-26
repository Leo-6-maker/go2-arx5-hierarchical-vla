# SLA V6.1 · 执行手册：R4正确标签、父模型验收、SOURCE接力

配套 [Plan](SLA_HOPPER_V61_CORRECTED_RETENTION_PLAN_20261126.md) · [Registry](SLA_HOPPER_V61_CORRECTED_RETENTION_REGISTRY_20261126.json)。本手册更新未来执行规则，不是已部署的服务或程序。继承#519/#521/#522原实现、预算、数据原件和全部历史结果。先读 [22:28 R4 HOLD终态][actual] 与 [V5真实资源][core]；不拿早期“待评价”的报告行覆盖已结算结果。

## A. Corelab 开工：并行三个有限工作队列

1. 现场对账owner、健康任务、GPU UUID/EGL、CPU/RAM/IPC、系统卷/HDD/NFS真实quota和在途；R4 N02两张已闭合，复用证据不重做。确认完整R4-200封印 `f1f5a994005a9fc831934a156ca823d7e4b67794ce2533d72e087c2aeb29b9cb` 与原R4-52数据 `1b57d74fd133a3acbfb57bdce10a7ac4b8d8b18fd7fad8ca72376530edcb2a21`，45/94叶、计数200、RNG、原基础、codec及实际运行版本。原M1000 2K关闭；R1/R2/R4共600已付，R3未运行且容量已用，不重复旧消融。
2. 在合法Core CPU资源上进行旧15条修订标签准入、公共源最新10条交接、其他数据source补充、Scene Factory冷编译/审计；不能为这些任务申请Hopper CPU/PBS。若公共视频加工挤占物理必要资源就限流，CPU/主存全池有界。67条是真实消费旧合同；按实际修订通过数量形成不可变R4-52/R4-67等，失败保留，不等待67齐全才准备训练。
3. Core原生资源允许时优先R4四张原PLACE、再两独立HOLD几何×两noise的父/R4配对；原N02/201/202已通过2/2，不重复。独立场景Teacher先行，复用已有真实资格；不足最多2条教师资格且优先占原V5槽。新布局/噪声、评分、时钟、安全、模型和request身份提前封存。双卡N2/N4未生产资格不得直接混跑；合法N1继续，补D3参考选择与状态差、双卡控制消费/接触/STOP。冷渲染128状态不是生产128world。原容器站姿0/2保留；侧向候选/新M_FREE另合同，不放宽旧门。

## B. 目标标签真实准入

每个query读取实际同clock TCP、接收器参考、实际已消费命令以及原时间/维mask与真实previous gripper，按已部署FiftyMsProgram/G7版本重建：

```text
corrected_position = consumed_position - (receiver_position - actual_position)
corrected_rotation = actual_rotation @ receiver_rotation.T @ consumed_rotation
```

在真实固定frame、单位、原时钟下执行原map验证全部有效点，位置最大≤1e-4mm、SO3角≤1e-5°；这只是数学复演容差，原物理门不变。原52已复演原件保留，新增15逐个核验；不套固定平均SE3/全局时移，不对M1000输出做逆补偿。未来消费后缀以当前query一致锚点，真实mask/padding/gripper不改；无效target扰动测试noise/loss及有效grad，原q01/q99、实际越界逐phase记录；不clip、不补末点、不用未消费/achieved冒充目标。用实际G7/decoder/L1检查完整H40及P10，并保存完整原件/逆映射及新标签哈希。保留旧失败、旧规范，V5五条DEV重合不准入，源域语义独立不套Go2变换。每4条/6小时有变化/波收尾封存，单writer，Core实际reader通过、真实合法交接。

## C. R4父释放及训练后继

- Core回归：原四PLACE实际有效结算，≥3/4正确抓持；N02 2/2已证；独立4张HOLD有效、R4≥1/4完整且不低于匹配父，记录任何配对退化；不出现相对父新增2个及以上相同机制抓前STOP。无关键缺失。完整PLACE不是SOURCE必备门。未达门先查确切原因，另登记有限后继，不重新定义门或用M1000。原L1保留所有40原点及真实时钟、安全、原40mm/s；记录实际桥接和消费、释放推迟、承托/撤离。R4是真实候选，不因一次2/2自动全域资格。必要时只读核原M200身份做同长度诊断；新的同长度闭环另有限登记，不阻塞有证据主线。
- 源READY：Core337（RoboDojo307+Ville30），旧200+实际≥100新增TRAIN且≥3任务组、有效新来源、真实许可、source独立split/codec/stats/mask。最新Ville10留Core需真实交接；按实际可读数量命名，不冒称Mix500。RoboTwin4只读试样时域未准入；Table30/Ville咖啡/OpenNeo按原缺口和授权，独立不阻塞其他源。先有限小包，不全库下载。
- 训练用实际完整R4-200而非29数组export，新optimizer/RNG、新域初始化与固定访问银行，校验原域参数未被无意改写。Core准备全部训练材料与真实父release；Hopper在实际GPU训练启动中完成必要reader/加载/restore，最多原V6 24新域工程更新真实资格（每域2实际梯度覆盖、整配方6步、保存恢复），不另发CPU-only job。
- SOURCE_EARLY 16/16/80/16+4aux；组权重B/128，80双臂可按实际资源分microbatch，按B比例合并、一次clip+Adam，原10步H40辅助与相对目标权重保持。单臂≥100合格且≥3技能/任务组才考虑16/64/32/16；单臂不可用才明确16/0/96/16。先固定实际数据/银行/域、峰值约1.69621416e-6（原峰值0.5，真实账本确认）、warm100、5000 cosine到0.1。保存250/500/1000/5000/条件10000端点，100步或30min恢复点。SOURCE计数从0，历史R4的200不算。每个段内数据、LR/采样固定；下一段同树合同恢复原Adam/RNG/LR/游标，不每250步重热身；新数据/比例在冷边界先封存明确版本，换树/stat/优化合同则另明确新训练状态，累计SOURCE预算不变。
- 每段完成并释放H200，Core固定24窗开发诊断及6卡面板；完整恢复、真实新增源访问、源独立DEV、N02 2/2、PLACE抓持数≥R4基线、无相对R4新增重复机制抓前STOP。只看loss/H40不够。源进步或有明确未收敛证据且目标保持，继续250→500→1K→5K；条件10K需更丰富数据及实际收益、另锁LR。新安全退化暂停，真实归因、早期ADAPT或独立修复，不自动重跑失败配方。完整PLACE不阻塞已满足保留门的SOURCE。候选9端点的六卡登记共54（R4 HOLD2已有，最多52新增/映射），独立父/R4各4与最终SOURCE/ADAPT各4共16，最大68新/映射学生；使用原V6/V5相关可用槽一对一映射，保留未启用/失败分母，旧CONF不重复。
- ADAPT使用实际新多样化R4合同目标数据及独立源回放，64/32/16/16；若单臂仍窄64/16/32/16，若无single64/0/48/16；固定新optimizer及LR（默认SOURCE峰值0.5×、warm100、3K cosine）。1000→条件3000→条件5000，原V6总容量8000默认/15000条件，控制1000、工程24单列不重置。100/200数据不齐如实命名；不反复旧52长训。后续恢复片段只在TRAIN安全可恢复状态合法采集，不用旧DEV、STOP后不积分。

## D. 站点/性能/资源边界

Core=公共/本体数据、独立离线和原生评价、CPU诊断；Hopper=真实GPU优化/有限训练性能，12CPU/225GB仅服务该训练。数据或评价等待释放GPU；Vanda暂停，无新的CPU守候PBS。复用合法唯一owner、Journal/intent、实际job ID和状态回读；UNKNOWN先对账，不重发，不取消健康无关job。每capsule≤6h且≥15min保存余量，实际p95和写盘核预算，旧320GPUh/截止不重置。CPU-only作业不派Hopper。Hopper性能先现有整域批、轻量真实训练Remat/独立冻结视觉cache资格；不重复失败R0，不默认B256/FP8/变训练语义。Hopper不承担Core批量诊断/评价。

Core两卡最新R4 GPU0实际约24GiB；20CPU/62GiB是历史实测，现场重查。系统卷/HDD原生产门仍有效；95/534GiB仅历史点，R4两卡有界例外不授予整池采集资格。用户负责清理/迁移，未经授权不删/搬原件、不降低全局门限；合法私有NFS余量按实测。受阻角色标BLOCKED，其他已获准角色继续，不假称系统全部停机或已恢复。V5目标67→100（补2旧缺+31新技能）→200＋40恢复，真实每条R4合同；Scene Factory、N1/N2/N4及公共Mix500→2K同时推进，符合资源/资格才扩。

## E. 给现场Codex的启动指令

```text
执行V6.1。先回读最新实际owner、Hopper终态及Core资源；复用R4-200已完成N02 2/2，不重跑M1000/R1/R2/R4训练或旧HOLD。Core按真实R4逆映射合同准入其余15条并对未来所有Go2数据做完整round-trip；并行公共数据准备与V5生产。完成R4原四PLACE和独立两布局×两noise父/R4 HOLD回归，按预设保留门放行，完整PLACE不是首轮SOURCE前置。R4全状态为首选父，新课程新Adam/RNG。实际已READY旧200+≥100新源及≥3任务组即可开训，不等待Mix500；337为Core库存，最新10条需真实交接。首SOURCE 16/16/80/16+4aux，固定128 FM、原辅助和WBC/L1，250→500→1K，阶段释放H200、Core做闭环；单臂≥100且≥3技能后才64单臂。具备真实保留与源改善再5K/条件10K和ADAPT。所有CPU数据/诊断及评价在Core，Hopper只训练，Vanda保持暂停。真实READY/资源/授权才提交，回报唯一实际PBS jobID、完整checkpoint、数据版本、实际访问及原生终态。不清理文件、不降低安全/存储门、不重置额度或旧失败。
```

[actual]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/ec927393bae025d536371ec88f237ba8de29ee3f/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/RETENTION_AND_SITE_ACTUAL_20261126.md
[core]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/fe21674f20ffb37538bae33646122adf25e5cdb2/reports/SLA_CORE_EMBODIMENT_FACTORY_V5/FIRST_SITE_ACTUAL_20261126.md
