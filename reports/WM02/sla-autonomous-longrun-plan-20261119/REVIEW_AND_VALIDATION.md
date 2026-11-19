# 自主长程计划 V1：证据与发布前核验

2026-11-19，Asia/Singapore。本文记录文档/研究设计核验，不是服务器实验ACTUAL。

## 交付与授权范围

用户要求“制定详细的长程自主推进计划推送到github，充分利用corelab/vanda A40服务器资源”。本次更新既有计划PR416，以main为目标；新增完整协议、机读预算、Codex启动prompt及本报告，刷新SLA统一入口。原入口全文在cc96f170历史提交保留。现役PR417/418合同固定SHA保持。

本次服务器作业提交、训练更新、优化访问、模型生成、native请求/tick和真机请求均0。计划中的数量是后继研究上界，不是已经产生的效果。本次没有修改主机、运行胶囊、模型或自动化任务。

## 来源与观察时刻

- 初审PR418：93b3165f785a4c14a27c432f18b454c52571178a；PR417：afaf35970e56a0827afb092654ce6731f4723fe2。
- 发布前补读PR418：39d8f6e341d2c974aa2baed31588b115e0c2c463；PR417：4c9ccb9cc1db2dd5187152259762739bdbd98ee9。
- 两站更新观察时刻为22:36:12 SGT：MIX452，TGT898；原目标访问/LR前452逐步匹配。GitHub更新时间不能外推新的训练步数。
- J最新10/48，417689tick、1446 H40，抓持/抬升/任务0/10。D5原24完整封存，原CONFIRM32不放行。
- R200已在Core核验；VANDA的SOURCE_DEV就绪不代表R200/LIBERO全TRAIN就绪。
- 首6已运行独立WAIT守候，仍0请求/0新示范，最迟启动23:39:04。执行者需接管实际已有服务，过期后在新有限时段只续未执行卡。

[最新训练与microbatch CPU核验](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/39d8f6e341d2c974aa2baed31588b115e0c2c463/reports/WM02/sla-scale20k-20261119/SCALE_OPTIMIZER_AND_MICROBATCH_READINESS_ACTUAL.md) · [J10](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4c9ccb9cc1db2dd5187152259762739bdbd98ee9/reports/WM02/sla-d4-next-plan-20261119/J_SCREEN48_BLUE_LEFT_PAIR_ACTUAL.md) · [首6守候](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/39d8f6e341d2c974aa2baed31588b115e0c2c463/reports/WM02/sla-scale20k-20261119/TARGET_ADAPTATION_FIRST6_DETACHED_WAIT_ACTUAL.md)。其他固定来源见完整计划末尾索引。

## 三路独立评阅与修订

分别审查资源/接续、目标数据/训练设计、合同/预算。发现并修复：

1. aux内部原max超限归约保留，不能统一替换成mean40。
2. L3零更新梯度/stepzero资格；phase1与phase25非零恢复属于同一5k，消除额外更新和循环依赖。
3. E_new仅包含实际可训练准入episode，排除隔离partial、预检条件和失败诊断。
4. 阶段面板在ADAPT0首次预测前封存去重窗口、标签种类、桶和噪声，固定时序公式。
5. 离线联合时序/原门与native消费/接触/任务证据分开。
6. benchmark采用时从正式谱系最新冷态端点接续，不回滚benchmark父端点。
7. VANDA全TRAIN准备显式列为实际数据步骤；CoreGPU1显存fallback走原handoff，与既有watcher协调。
8. 首6已存在WAIT守候被纳入调度，不重复创建；旧有限窗过期不伪造新期限。

修订后合同/预算复读通过，未发现发布阻断。当前runtime的完整GPU/microbatch资格和新reader部署仍属于未来执行任务，本文不称它们已经通过。

## 机器算术与规则核对

完成35项文档/机读一致性检查，35通过。核对分类求和、各域访问、模型生成/专家查询分开、native tick公式、24条件组成、四端点离线、96阶段配额、64aux槽覆盖、LR端点和单调性、D2单次边界、旧预算、真机0、发布效果0及关键修订字段。

| 本版新增上界 | 数值 |
|---|---:|
| 科研ADAPT更新 | 5000 |
| 工程更新 | 120 |
| 优化访问T/R/L | 323840 / 165760 / 165760 |
| 总优化访问 | 655360 |
| 资格梯度窗口，0优化更新 | 128 |
| 训练模型生成 | 20480 |
| 离线模型生成 | 2548 |
| Native模型生成 | 15362 |
| 全部模型生成 | 38390 |
| 专家计划查询，另列 | 43200 |
| 科学/接口/采集/N2请求 | 96 / 2 / 240 / 24 |
| 全部native请求 | 362 |
| 全部native tick | 16824986 |

精确公式：240×50703 + 4×1096872 + 11906 + 24×10703 = 16824986。每个24条件面板为8×35703 + 16×50703 = 1096872 tick；对应8×120+16×180 =3840模型H40。每接口5953tick最多一个模型H40，两个接口最多2生成。

原SCALE25k/2720000访问/118120生成/98请求/4399394tick单独沿用，已发生进度不再加一份；旧D4/D5/J也不并入本版新增总数。

本次只做文档、预算和规则核验，没有运行新模型、MuJoCo或服务器CI。跨站逐位恢复、真实数据吞吐和新科学结果必须由后继ACTUAL给出。

## 发布文件指纹

下表记录本报告写入前四份交付内容的SHA-256；本报告不作自引用哈希。提交后还须通过GitHub逐文件回读与本地文本一致性、PR实际head及正文核对；结果在PR说明和最终交付中记录。

| 文件 | SHA-256 |
|---|---|
| docs/protocols/SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1_20261119.md | 2147bad368c933b27eb62f464383cb4fccd651ca2f7d81340dc79050062fe8e5 |
| docs/protocols/SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1_20261119.json | 85ea24399b0e5a4fc34dce714361e2877f4ba2a215709ef8c9a4b9e97a5f7596 |
| docs/handover/2026-11-19/START_SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1.md | 84b35560306d65b58daae4d40de53713436d3ab4a4c713a22b35874f01673f2b |
| SLA_D4_CLOSED_LOOP_NEXT_START.md | d086e80ee2207ce174f640f1d5f246c1efae7d333d02ba92d11b42a5dc66b7e3 |

## 入口

- [完整计划](../../../docs/protocols/SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1_20261119.md)
- [机读任务与预算](../../../docs/protocols/SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1_20261119.json)
- [Codex启动prompt](../../../docs/handover/2026-11-19/START_SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1.md)
