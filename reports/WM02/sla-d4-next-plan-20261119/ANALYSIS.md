# D4后训练进展审计与下一步判断

2026-11-19（新加坡时间）。以用户11:47更新为入口，独立回读PR415固定6d0ff9f8、相关执行代码与PR414终态。只读代码/报告审计，不是SSH现场回放或新原生实测。

## 1. 当前真正完成了什么

D4已完成可恢复的匹配源迁移实验：A从SLA_base学C20，B先装入S1000的20个共享LoRA，再用同一C20学500更新。两臂各16000目标访问、源访问0，87全态数组恢复与RGB16逐位一致。这支持“训练端点身份和恢复链可靠”，还不支持“已经学会真实移动抓持”。原生评价仍0/48、0tick。[终态](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6d0ff9f8d8d33c47e5dde308203c57bb5b2adc64/reports/WM02/sla-d4-multisource-c20-20261118/VANDA_ACTUAL.md) · [机器记录](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6d0ff9f8d8d33c47e5dde308203c57bb5b2adc64/reports/WM02/sla-d4-multisource-c20-20261118/VANDA_ACTUAL.json)

| 指标 | A | B | 能说明什么 |
|---|---:|---:|---|
| 完整H40通过 | 93/96 | 95/96 | B多2条离线安全包络通过 |
| H40与前命令夹爪速度同时通过 | 89/96 | 92/96 | B少3条组合拒绝 |
| 20条标注跨越记录漏检 | 18/20 | 19/20 | 两臂闭合轨迹拟合均有明显问题 |
| 末端机械宽度MAE | 7.11mm | 6.59mm | 平均误差略小不等于事件按时发生 |
| 原生完整任务成功 | 未测 | 未测 | 不能报告0%或比较成功率 |

96条是TRAIN窗口/噪声预测，20条是其中带跨越标签的预测记录，不能当96独立场景、20次独立抓持。配对H40为双方通过93、仅B通过2、双方拒绝1，效应证据很小。

B的唯一H40超速首索引k=5、峰值61.46mm/s；A三个在k=27。B首问题落在P10范围内，A的尾段拒绝也必须保留，因为原门检查完整H40。不能以拒绝条数更少宣称B整体更安全。

## 2. 四类阻碍需要分别解决

### A. 执行接线：D4原生路由尚未闭合

旧scope/campaign/pool仍要求M-BASE/M-SRC、1000/32000、6400 C20+25600 Teacher及其旧晋级条件。D4是独立schema、500/16000、D4-A/B。D4已具备独立Provider，尚需原生scope/worker/campaign的显式接线。

因此“待固定规格”具有具体工程内容。下一项应该是D4专用执行入口与最小接口证明，再运行原登记人口，不能等一个旧M2验证器自行接受新模型。[D4端点](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6d0ff9f8d8d33c47e5dde308203c57bb5b2adc64/src/go2arx5_vla/sla_adapter/multisource_d4.py) · [旧scope](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6d0ff9f8d8d33c47e5dde308203c57bb5b2adc64/src/go2arx5_vla/world_model/multitask_scope_v6.py) · [旧campaign](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6d0ff9f8d8d33c47e5dde308203c57bb5b2adc64/scripts/wm02/multitask_student_campaign_v6.py)

### B. 学习目标：平滑不是闭合，更不是抓持

代码确认D4的TransferEngine继承MultitaskEngine，只复用CloseEngine的采样与LR；实际目标为FM32+0.1XYZ+0.1夹爪速度。没有使用CloseEngine的物理宽度/时间增量项。速度惩罚本身允许恒定张口，但FM仍在监督闭合，不能断言整个训练没有闭合监督。[训练实现](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6d0ff9f8d8d33c47e5dde308203c57bb5b2adc64/src/go2arx5_vla/sla_adapter/multitask_training_v6.py) · [夹爪辅助](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6d0ff9f8d8d33c47e5dde308203c57bb5b2adc64/src/go2arx5_vla/sla_adapter/dense_g7_v6.py)

C20标签在38与40mm/s均0违规，最大37.4375mm/s。因此目前没有证据说明“按标签正确闭合必然违背原速度门”。优先试物理量监督有明确依据，但它是否真的改善抓持必须经闭环比较。[输入审计](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6d0ff9f8d8d33c47e5dde308203c57bb5b2adc64/reports/WM02/sla-d4-multisource-c20-20261118/ACTUAL.json)

采样已经按任务、episode、phase、anchor分层。不能未经访问表统计就声称长HOLD淹没CLOSE。32维损失中只有7个有效动作维、logit饱和和各损失梯度冲突可以查，但本轮不把它们同时改掉。

### C. 观察到动作到接触：目前缺闭环因果链

两秒H40每次只消费0.5秒P10。即便模型把闭合安排在未来，也可能在下一次重规划中再次推后。这是应测假设；离线窗口遗漏不能直接等同物理夹爪不闭合。

下一轮需要区分：模型不给闭合；给了但未安装/消费；消费了但实测宽度不响应；闭合但没夹中；抓住后抬升或移动时丢失。对应修复分别落在损失、执行接缝、机械映射、几何数据和动态时序。

旧episode在consume_prefix正常返回后才推进score。STOP发生在中途时，报告应以同一个原Score回读完整2ms日志到终点，保留不足25tick的部分原子；不能只看最后一个正常查询的缓存进度。[原episode](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6d0ff9f8d8d33c47e5dde308203c57bb5b2adc64/src/go2arx5_vla/world_model/multitask_student_episode_v6.py) · [原评分](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6d0ff9f8d8d33c47e5dde308203c57bb5b2adc64/src/go2arx5_vla/world_model/multitask_score_v6.py)

控制器最终selected参考仍UNKNOWN；ACK不能替代这个证据。补充已有consumer记录会改善归因，但不因某个诊断字段缺失再阻塞所有独立数据学习。

### D. 数据与任务覆盖：20条HOLD不能承担完整长程结论

C20有1376窗口，独立示范仍只有20条，而且只含HOLD。D4的PLACE是有意的零目标PLACE示范迁移。其失败若出现在下放、支撑、释放，不应被解释为已充分教过这些行为却学不会。

已有3条完整PLACE、392当前RGB/H40、3920已消费原子通过审核，可作为下一独立TRAIN版本的种子。它们都在F1、两个场景、同采集lineage，尚不支持F2放置覆盖。[Core审计](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6d0ff9f8d8d33c47e5dde308203c57bb5b2adc64/reports/WM02/sla-d4-multisource-c20-20261118/ACTUAL.json)

所以不必等待一位Teacher全部48资格成功；可以先使用逐条合格的目标示范、公开源数据和局部纠偏数据。安全门与标签语义继续逐条严格，完整Teacher人口门不再垄断后训练进度。

## 3. 两个容易误读的最新进度

### 闭合误差的符号

当前width_metrics.crossing_error_s使用绝对值，公开终态仅保存绝对误差列表。报告所写“晚1.35秒/0.15秒”尚不能由这两个汇总字段独立确认。保留原文，下一轮读取原predicted_first_crossing与expert_crossing_in_window，补有符号派生记录。22mm只是诊断阈值，不是抓持判据。[指标实现](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6d0ff9f8d8d33c47e5dde308203c57bb5b2adc64/scripts/sla/close_goal_g1_v1.py)

### N2已有新进展，但仍不能用于本轮

PR414最新D3工程把双向STOP补齐：两方向停止者0、同伴3500tick，4/4通过。N2吞吐仍约1.99倍，但原数值容差仅5/8；最新阻碍已经不是旧D2的STOP缺证据。N1自身重复EEF/mount峰差4.108/3.851mm，故应先查已存调度、状态和参考选择差异。[D3并行终态](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8d6df823e58bd73d441e2557329c3131f4427756/reports/WM02/sla-d3-contact-20261118/D3_PARALLEL_ACTUAL.md)

D3机制12/12已封存，处理里程碑0/6，唯一完整成功在P5对照；R2/Q/M2仍未启动。不能把D3的严格完整Teacher路线继续作为D4/J的全局依赖。[D3机制实绩](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/8d6df823e58bd73d441e2557329c3131f4427756/reports/WM02/sla-d3-contact-20261118/D3_CAUSAL_DIAGNOSTICS_ACTUAL.md)

## 4. 官方SLA资料能支持和不能支持的判断

官方论文的后训练按下游基准分别进行；真实跨本体D实验使用100条Piper和10条Franka示范，然后在目标Franka上评价。示范数不是训练采样占比。这支持“源经验加少量目标本体示范”，不支持把源双臂动作不经适配直接送给Go2+ARX5。[论文§4.1](https://arxiv.org/html/2611.08373v1)

官方数据加载器按filtered dataset长度×weight抽样，weight=1不等于均匀来源。因此没有可直接抄成统一sim:real百分比的后训练配置。本计划J的32+8是新实验选择，不是官方推荐比。[固定官方data_loader](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/18896e757647ed9c2a68816b35a7100afda6e287/SLA/src/openpi/training/data_loader.py)

主官方SLA配置中的空间对齐与RoboDojo Memory应按具体发布路径区分；不能把本项目G7/LoRA控制适配称作完整官方recipe复现，也不能把尚未开启这些模块当成当前闭合问题的已证根因。当前优先测试最直接的动作轨迹与执行假设，空间对齐/历史RGB以后独立比较。

## 5. 计划的核心取舍

- 先补执行入口并评价冻结A/B；不会只看离线+2就抛弃A。
- W/T从历史B step-zero重训500，复用B500为原loss对照，拆分宽度和增量监督；不与额外500步的模型做伪等预算比较。
- 新GRASP24筛查后用8个更新布局CONFIRM32；开发场景不重新叫作盲测。
- J0/J1从相同B500端点开始，各500、各16000目标访问；J1额外4000源访问。它回答额外源梯度效应，明确算力不匹配。
- 不将PLACE3、memory、padding/采样改动、WBC调整、输出限速同时塞进一个实验。
- 单world科学执行和独立站点训练可并行；N2资格、Teacher资格都不是所有训练的前置。
- 新数据采集根据真实首失败分布另定有限清单；本次预算不包含新Teacher人口、真机、TEST或大数据下载。

[详细规格与预算](../../../docs/protocols/SLA_D4_CLOSED_LOOP_AND_GRASP_20261119.md)给出所有条件、停止规则、loss公式、生成账目和执行接缝。
