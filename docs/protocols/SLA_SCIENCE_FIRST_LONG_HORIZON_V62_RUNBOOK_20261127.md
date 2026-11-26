# V6.2 Science-first · 现场执行手册

[Plan](SLA_SCIENCE_FIRST_LONG_HORIZON_V62_PLAN_20261127.md) · [Registry](SLA_SCIENCE_FIRST_LONG_HORIZON_V62_REGISTRY_20261127.json)。依据用户最新“不用管存储限制，优先推进科学实验主线”覆盖#523未来项目存储硬门。先核 [实际训练与评价][actual]、[V5实绩][factory]、[维护回执][storage]，已完成结果不重复。

## A. 最小现场启动与容量策略

1. 只回读实际Core/Hopper owner、GPU/PBS、实际卷free/已知quota、站点限制及必要写入预算。不重新盘点658项，不等待清理/迁移/NFS整批quota确认。项目自设15%+8GiB/50GiB/150GiB hot cap改为提示，非科学拒绝门；复用[原layout][layout]在新冷态版本保留真实设备、每world/在途/下一快照/终态预算，实际admit取真实写入能力与真实站点规则。不得假造free或通过旧门；旧gate记录原样保存。只有实际不可写或无法安全封存的任务局部暂停。无NFS写入的任务不受未知NFS额度阻塞，实际需写时确认有限合法容量或用已验证本地路径。原WBC、安全、锁、原分母与评分不改。无授权永久删除不做，现有145文件8.53GiB不重复计入；保护关键数据/模型。
2. 确认新路径不会重定向活跃进程、无错误mount/符号链接，控制SSD、产物HDD。用真实预算检查必要余量，不规定等待恢复274.16/566.90GiB。小型本地写入试验与受影响入口的最小资格通过后直接运行；不需要每个工程步骤重新问用户。不能改系统/挂载/权限或管理员硬限制。
3. Core独立CPU/数据/评价与Hopper训练严格分工，Vanda暂停。单合法owner及已有Journal复用，健康任务不打断，未知请求/提交先对账。

## B. 同时推进两个P0工作包

- Core原独立HOLD：名册 `c2af53f5b70340d1543082adbdbb769ea8e0e66231ac63d56059485136669605`。已完成父N03两格0/2、R4 N02两格2/2、R4 PLACE4/4抓持（完整0/4、0STOP）全部引用。仅运行R4 N03/201,202及父/R4 VAL-F1-G00各201,202六格。35703tick/120前向每卡。按原V6.1父门判定R4独立≥1/4且不低于父、无新增重复抓前STOP≥2；完整PLACE不作为SOURCE前置。
- Hopper实际核父 `f1f5a994005a9fc831934a156ca823d7e4b67794ce2533d72e087c2aeb29b9cb`、337源/真实reader与bank `1ba0092a0c982c51e785849b86332bfc9c3db636bad6a82cab97548bed027891`，项目授权、完整恢复/保存余量。原新域≤24工程资格。A若已通过，标准SOURCE250；A尚未结算且无已知R4独立退化时，在现场实际合法范围内直接启动唯一SOURCE250_PROVISIONAL。首段最多250科学形状更新，标准和探索同一固定课程，只允许一个job链，未知状态先对账。探索不是父验收，不能自动500。A与250端点均通过再晋级已做前缀、接力，不重跑。
- 配方16/16/80(40旧40新)/16+4，原52正确标签、128 FM、独立native时域/mask，group B/128、aux1/16、10步全反传、一次clip+Adam、新域新optimizer/RNG。峰值1.6962141658312244e-6、warm100、5K cosine末0.1；首250固定4000/4000/20000/4000+1000。每100步或30min、端点完整seal；≤6h且保存≥15min，真实H200读/缓存/预取/保存属于训练，Hopper不得承担独立CPU-only。Core冷读耗时不是Hopper吞吐，真实训练内有界优化，不反复profiler。等待Core释放H200。
- 250后Core固定24反馈、六卡(N02两/PLACE四)及source独立每域≥16窗；父门通过、N02 2/2、PLACE≥R4 4/4抓持、无新增重复抓前STOP、实际源访问/恢复/有限值/源进步或明确未收敛才500→1K→5K。5K→条件10K须新数据及实际收益、新LR合同。ADAPT 1K→3K/条件5K，真实新Go2数据+源回放，64/16/32/16窄单臂，64/32/16/16扩展，aux1/4。单臂100且3技能才扩SOURCE64单臂。新数据/配比只在已封存冷边界，真实累计更新不重置。主线默认8K/条件15K，工程24、可选1000继承，旧320GPUh/12月5日12:41:52 SGT不重置。

## C. 并行P1科学工作

- R5不因固定反馈阴性直接宣布原生失败；复用既有48固定窗及132真实返回，额外48教师释放/持物对照窗×两模型≤96前向，查真实语义/phase/support偏移。R5自身先P01/701、P02/701两卡，条件追加其余2 PLACE+2 N02，最多6，不阻断SOURCE。只有实际监督/恢复证据才用原可选1000内≤250修复，扣减控制余量，不再仅提高旧OPEN重复训练。
- Core继续69→100（新31：Reach8/Mobile8/Place8/Container4/Push3）→200+Recovery40。新增全部正确逆重锚复演，17旧增量只补实际运行时资格，真实READY/访问分开。先N1，不等N2；双卡1+1真实控制/接触/STOP和吞吐过门再N4。两个已有容器候选优先真实Teacher资格；冷静态不是完整任务。公共337→500→2K小批增独立技能，Footbag不主导，不下载全库、未准入不进训练。
- Drawer/Insert/组合沿V5目标未见人口、先真实组成技能，再单指令连续阶段、fresh反馈、真实终态/顺序/安全，不给学生GT或强制释放；原CONF独立、源曾见标目标域zero-shot。少样本及匹配源迁移控制保留原范围。

## D. 交付与启动指令

```text
执行 V6.2 Science-first。取消旧项目自设固定存储余量作为科研前置；只保留真实写入/配额/站点及安全收尾能力。不要等待清理或NFS迁移。Core立即补原六张独立HOLD；Hopper并行从完整R4父、已交接337源、现有5K bank做有界真实资格与唯一首250 SOURCE。未完成父回归时标PROVISIONAL，最多250，实际计成本；父回归及250原生门均通过才500/1K，不重跑已封存前缀。Core并行V5新技能生产、公共数据和R5释放专项，不等待完整PLACE。所有真实安全、冻结WBC/L1、原评分、数据隔离和现有授权保持；Hopper只后训练，等待释放GPU，Vanda暂停。普通工程自主修，真实能力不足仅局部BLOCKED。报告实际PBS ID、完整端点、优化访问、闭环结果和新合格示范，不继续以存储审查为主任务。
```
本次计划发布0现场更新/积分/清理；现场需回读真实状态，不把文档当部署。

[actual]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4de2985f05a9acf87566aa5b1dd363816bbdc34b/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/RETENTION_AND_SITE_ACTUAL_20261126.md
[factory]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6a10550dcc5beaf1e9a8c4180796198149d52023/reports/SLA_CORE_EMBODIMENT_FACTORY_V5/FIRST_SITE_ACTUAL_20261126.md
[storage]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6a10550dcc5beaf1e9a8c4180796198149d52023/reports/SLA_CORE_EMBODIMENT_FACTORY_V5/STORAGE_MAINTENANCE_ACTUAL_20261127.md
[layout]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6a10550dcc5beaf1e9a8c4180796198149d52023/scripts/sla/v5_storage_layout.py
