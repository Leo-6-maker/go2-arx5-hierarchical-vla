# 多源后训练 Goal V2：进度审核与计划依据

日期：2026-11-20，Asia/Singapore。本文是GitHub源码/报告审核及后继计划，不是服务器live核查。公开报告、用户较新现场摘要与后继建议分别标示。

[完整Goal](../../../docs/protocols/SLA_MULTISOURCE_CLOSED_LOOP_GOAL_V2_20261120.md) · [JSON](../../../docs/protocols/SLA_MULTISOURCE_CLOSED_LOOP_GOAL_V2_20261120.json) · [启动prompt](../../../docs/handover/2026-11-20/START_SLA_MULTISOURCE_CLOSED_LOOP_GOAL_V2.md)

## 1. 审核结论

项目已经从小规模单抓取训练推进到真实三域更新、可恢复状态和后台独立Eval。当前最主要缺口是**关键阶段预测时序与完整执行门不协调，以及新ADAPT尚未取得学生原生分母**。继续合理增加更新与数据是必要实验，但不能仅用累计step、H40普通面板或闭合出现率代替本体能力。

旧D4 0/28、D5 0/24、J 0/12是不同模型/人口的历史负结果，不能相加为新ADAPT成功率。专家HOLD2/2证明本体/场景存在可执行成功轨迹，不证明学生模型已掌握。部署V2资源检查0生成，证明资源前置未满足，不是一次模型数值或抓持失败。

## 2. 证据新鲜度

- [PR418固定报告MD][ACTUAL_MD]和[JSON][ACTUAL_JSON]的观察时间为12:49:41。公开可独立核验MIX869恢复、D2_READY、TGT2k/3500封存与ADAPT500评价；当时running step为875/4629/844。
- 用户14:08 SGT更新报告MIX1000封存、TGT4976、ADAPT953、部署资源拒绝、学生0请求。该更新更晚，本文采用并标为用户现场报告；未在此ChatGPT会话读取对应私有最终回执。
- PR419固定head `e98f58d5`报告128/128输入一致、25项CPU检查；GPU基准0。源码确认三路有限工程实现已存在，两卡实现尚不存在。[PR419][PR419]

因此下一执行会话应补真正现场对账，不把旧公开数字称为最新live，也不重跑已经完成的恢复/读取/Eval。

## 3. 闭合“更多”与闭合“正确”已经分离

以下均来自同一固定ADAPT面板，非跨实验拼接。[机读证据][ACTUAL_JSON]

| 指标 | ADAPT0 | ADAPT500 | 解释 |
|---|---:|---:|---|
| 普通96窗 H40 | 96/96 | 94/96 | 普通面板整体变化较小 |
| 普通96窗 H40+接缝 | 92/96 | 89/96 | 不能代表关键阶段 |
| 阶段88窗 H40 | 84/88 | 57/88 | 从95.45%降至64.77%，净降30.68个百分点 |
| 阶段88窗 H40+接缝 | 45/88 | 43/88 | 两个边际变化不同，需同窗交叉而不能直接相减归因 |
| 到期24窗 P10闭合 | 10/24 | 24/24 | 到期窗口中都出现跨越 |
| P10闭合且时差≤100ms | 10/24 | 2/24 | 及时性从41.67%降至8.33% |
| 及时且原双门通过 | 0/24 | 0/24 | 可执行的及时闭合尚未出现 |

现在只能确定：ADAPT500的两个及时样本至少一道门失败；不能确定其余22个全是提前或全是延后，也不能确定全部24都被门拒绝。窗口有重叠，不作独立二项试验或统计显著性结论。TGT2k普通双门93/96但唯一到期样本仍漏闭，也说明普通安全面板对抓持事件覆盖不足。

### 已确认的代码事实

`evaluate_adapt5k.diagnostic_metrics`保存专家/模型首跨越index、绝对时间差、是否P10到期及联合指标，评价器保存normalized/decoded数组。[评价源码][EVAL_CODE] 因此可以不新增模型生成，复算有符号时差、负例误闭、首拒绝和逐行交集。

当前辅助loss使用原XYZ动态、夹爪速度及全40点width MSE；`width_losses`的第二返回值没有启用。[训练源码][TRAIN_CODE] 这支持“均匀宽度拟合没有直接对齐首次闭合时刻”的机制假设，但还不能断言因果。第一步是确认真实前命令/时钟下的专家目标可行；原40mm/s×50ms约束意味着单点宽度变化上限2mm，严格跨过22mm需要足够提前量。不能用立即关闭所有P10窗口替代事件学习。

### 原生应补的证据

把离线预测正确性、原始H40资格、实际P10消费、正确目标接触、抓持、抬升、底座移动、保持和释放撤离连成同query/episode漏斗。0/24 TRAIN联合不是禁止运行原固定native的新门；真实输出仍逐次经过原保护。新学生闭环不应长期排在更多Teacher数据或性能工程之后。

## 4. 本体数据有增长，但独立经历仍有限

D1只有24条/1892窗，D2只有26条/2031窗；源为RoboDojo200与LIBERO1261。新增139窗只来自两个HOLD episode，不能把139当成139种独立接触经历。D2的12新+52旧来自累计3条新episode（1 PLACE+2 HOLD），不是只计算两个HOLD。固定统计与原codec保持，防止课程切换同时引入表示变化。[D2证明][ACTUAL_JSON]

源数据通过独立头、原生时钟与动作语义参与共享表示训练，有合理用途；它不能自动提供Go2移动底座、上一实际命令和目标接触的标定监督。因此继续多源更新与扩充本体独立场景应并行，不要求Teacher全格通过或先收满120条。新TRAIN数据下一批偏向成功率较可控的HOLD覆盖、针对性修复的PLACE和真实可用恢复，避免同首因反复烧仿真预算。

对来源贡献的结论也要保持匹配：原MIX5k/TGT5k共享目标流与LR前缀，支持“额外源数据和额外计算”的比较；ADAPT相对TGT又改变父、目标访问量、统计和D1/D2课程，不能将其差值全部归因源数据。MIX20k相对TGT5k同时改变步数。

## 5. 运行阻碍的定位与修复

### 5.1 无需再实现已有handoff

`f9c8457b`已经支持到5000的TRAIN_LIMITS，调用原 `1432c8dc` worker并在精确2k执行一次selector。[handoff][HANDOFF] · [selector][SELECTOR] 目前后半段未提交的已知原因是AMGR登录令牌过期。它是认证/调度阻断，不是模型或D2问题。保持现有PBS作业，合法认证后登记尚未提交部分；不能假报JobID。

现役使用afterany+seal门，退出75的完整前缀可以续训。正式Eval要求精确端点。2k未完整达到时必须补齐，不能被2250任务跨越。这是针对已有实现的接续修复，不需要新调度框架。

### 5.2 资源组合拒因应拆开记录

公开资源检查要求GPU free大于ceil(total×fraction)+2GiB、host available大于30GiB，失败前未将各实际值写入回执。[资源源码][DEPLOY] private守候额外压力门未在公开材料中给出，不能推断究竟哪项拒绝。新wrapper只补可观测性并检查预留算术，保留门。GPU1不满足时使用真实合格的GPU0冷态窗口；不是通过减小安全余量来获得通过。

### 5.3 性能工程已到GPU实测阶段

PR419三路各8个匹配更新后比较140叶，serial再做2步profile；不是10对8。真实更新总26、访问3328、训练内H40104。现实现只保存指纹，没有完整工程checkpoint及独立恢复预测；这些输出若要补，必须新增实现并另计生成。[benchmark][BENCH]

正式worker没有启用benchmark flag，且现役D2 binding不允许更换source_revision。因此本Goal保留ADAPT5k旧worker，性能资格优先服务新科学分支。后续四个16步工程配置包括同节点2卡，但两卡代码、实际allocation与全局梯度语义都必须实证。

## 6. 新Goal的研究决策

主线完成MIX20k/TGT5k/ADAPT5k、原SCALE96和ADAPT96实际学生评价。以下独立并行，不增加前置等待：零生成逐行诊断；后台接续；D2单次切换；新TRAIN采集；有限性能工程。

只有ADAPT2k固定联合仍0/24且标签/时钟/目标可行时，触发一个固定父、同D2、同500步曝光与LR的CLOCK-A/B。科学触发与资源就绪分开；认证/资源不足记BLOCKED，不能叫NOT_TRIGGERED。B只增加width监督的事件时间权重，保留原40mm误差归一化与完整40点监督支持，保持全部原动态约束；不把P10消费mask用于删掉后30点标签。原主ADAPT2500完全同条件时可复用控制A；否则独立训练。新A/B各24配对DEV及固定B确认24，场景、reset、噪声、执行后端与先后顺序预先固定，使时序监督能得到真实闭环比较；这72条不替代旧96，不改变旧TEST。

新增上界是1090更新（90工程+至多1000科学）、139520优化访问、17306各类生成、74原生请求含2条件接口、3302522tick。原科学/采集/工程额度不重置。技术生成储备按实际用途记账，不伪称科学更新或离线。详细乘积与复用条件以JSON为准。

目标区分流程跑通与能力：各谱系自己的确认HOLD至少4/8、F1/F2各2及至少2布局是本Goal预登记pilot里程碑；PLACE各自至少2/16且左右各1单列。ADAPT和CLOCK确认人口分别报告，不池化或事后择优。未达到就如实结束有限版本，不以改门、重试或自动延长追求成功。

## 7. 发布与验证边界

此变更为文档/机读任务/入口更新，未改运行源码或服务器。验证包括：固定证据及源码读取；三路独立科学/调度/性能审计；JSON与预算算术、任务依赖、关键不变量与相对链接检查；发布后固定提交逐文件回读。发布前实际完成37项文档/算术/任务依赖/边界检查，全部通过；17个新增相对链接解析通过。三路独立复审修正并复核了width的40mm归一化、原分类与完整40点支持、确认人口分开统计、offline/native依赖、科学触发与资源就绪分离、两卡匹配基线及campaign finalizer。提交后的10文件逐字/哈希回读结果在PR发布说明记录；没有运行GPU或服务器测试。

本次服务器作业提交、训练更新、模型生成、native请求/tick、真机动作及自动化变更均0。PR保持正常Draft发布，不自动合并或部署。真正后台注册与实验成果由后继执行PR发布。

[ACTUAL_MD]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0ef0abfcc39096666ad8bed6a5842f3ba25cd2b7/reports/WM02/sla-scale20k-20261119/LONGRUN_MIX_RECOVERY_D2_AND_DEPLOY_ACTUAL.md
[ACTUAL_JSON]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0ef0abfcc39096666ad8bed6a5842f3ba25cd2b7/reports/WM02/sla-scale20k-20261119/LONGRUN_MIX_RECOVERY_D2_AND_DEPLOY_ACTUAL.json
[PR419]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/419
[EVAL_CODE]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1432c8dc88c57adf39e8fa363f5438c5e63aa8fd/scripts/sla/evaluate_adapt5k.py
[TRAIN_CODE]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1432c8dc88c57adf39e8fa363f5438c5e63aa8fd/src/go2arx5_vla/sla_adapter/scale20k_training.py
[HANDOFF]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f9c8457bbe47c35c54d173abfc619500320b1ad2/scripts/sla/adapt5k_vanda_handoff.py
[SELECTOR]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f9c8457bbe47c35c54d173abfc619500320b1ad2/scripts/sla/adapt5k_curriculum.py
[DEPLOY]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/71881dcf10d2887b9dbfb31e012b0d1c1ad55c33/scripts/sla/replay_adapt0_core.py
[BENCH]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/e98f58d51e599384a5eee7652a7e600a7dc77ce0/scripts/sla/benchmark_adapt5k.py
