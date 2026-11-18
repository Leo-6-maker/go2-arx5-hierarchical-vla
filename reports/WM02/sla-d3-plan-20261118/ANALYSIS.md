# SLA D2 终态分析与 D3 决策依据

日期：2026-11-18，Asia/Singapore。审计对象：PR #412 的固定终态 `4d597259f2968d01c1694f9a9ee93dbc717695d1`。本次仅审核及发布计划，新增仿真、模型生成、训练与访问均为 0。

## 1. 结论

当前关键路径是：在冻结 WBC 和既有动作接口下生成可完整执行、覆盖多指令的目标本体示范。新目标学生 M2 尚未训练，因此这些结果不能证明 SLA 的目标后训练退化、源迁移无效或模型无法理解 PLACE。

D2 按原门结束是正确的；但主线能力没有通过。其有效进展是把原先混在一起的抓放失败分成可检验的机制，并量出了双 world 的实际吞吐潜力。下一轮应先解决参考、几何和接触转换的一致性，再扩充目标数据。继续增加训练步数、混入未厘清标签的真机数据，不能直接消除这些 Teacher 执行缺口。

## 2. 已证实进度与证据边界

| 阶段 | 已执行结果 | 当前含义 |
|---|---|---|
| D2 B1 | 12/12，227,908 tick；HOLD 3/4，PLACE 0/8；1 STOP，0技术失败 | 完整路径预检并未形成完整 PLACE 示范 |
| D2 B2 | 12/12，274,759 tick；HOLD 1/4，PLACE 0/8；2 STOP，0技术失败 | 最终候选 G7 覆盖 1/12，准备门失败 |
| 合并资源账 | 24/24，502,667 tick；4个开发正例、3次 STOP | 两版本/两布局结果不能拼成一个 Teacher 的覆盖 |
| 双 world 工程 | 20请求，207,060 tick；N1记录器关闭技术失败1次 | 与科学24次的0技术失败分开报告 |
| N2 | 匹配吞吐1.993倍；状态容差7/8；同伴STOP后继续积分仅1/2方向有证据 | 未获正式科学后端资格 |
| QUAL48 | 0/48 | 前置未通过，正确停止 |
| M2 | 迁移预算、更新、访问、生成、学生原生均0 | 目标多任务适配的对照问题尚未被实验回答 |

B1/B2 的 Teacher 和布局同时变化。3/12→1/12 只能描述两个开发批次，不能识别补丁的因果收益或退化。0/16 PLACE 是开发结果，不是同一冻结策略在独立同分布样本上的可靠性估计。

最终两批服务均关闭、owner释放的判断来自固定 GitHub 封存回执；本次未重新登录 Core 作当前机器状态检查。[开发终态][development] [资格终态][qualification] [并行实绩][performance]

## 3. 六项主要机制

### 3.1 F1 腕部—持物的净空余量有坐标与相关性问题

B2 三次 F1 CARRY 首拒均为内部腕侧—持物配对：

| 条件 | 实际间隙 mm | 原余量 mm | 保留原门后的外障最小间隙 mm | 相对坐标诊断余量 mm |
|---|---:|---:|---:|---:|
| 红左放 | 17.692 | 19.179 | 20.239 | 12.891 |
| 蓝左放 | 17.644 | 18.487 | 21.989 | 13.184 |
| 蓝右放 | 18.087 | 20.480 | 36.822 | 16.002 |

红左原余量19.179mm中，几何采样项3.510mm，近期世界坐标跟踪残差15.669mm。离线分开配对后，外障保留原门仍通过，内部配对采用相对安装座残差也通过。这是共同底座运动误差被计入内部相对距离风险的较强证据，尚非动态安全证明。

两个物体共同施加同一个刚体变换不改变它们之间的距离。腕部与抓持物的相对变换可由 `T_EO = inverse(T_WE) * T_WO` 表示；共同世界变换在这个表达式里抵消。但真实抓持并非焊接，手指运动、滑移、局部姿态误差仍必须保留。

应修正碰撞配对的误差模型，而非把全部margin乘一个变小系数。外部障碍继续用原世界坐标余量；内部配对保留原采样项，并用明确的相对姿态/滑移误差项。公式、窗口、超出已验证域时的终止规则必须先登记，不能把12.891等事后数值直接当新门槛。旧失败不翻案。[归档复核][replay] [几何实现][geometry] [Teacher实现][teacher]

### 3.2 F2 红物体的接近走廊尚未打通

两批 F2 红放4次请求、22,812初始化tick、0任务H40；共212条时间预算内候选均在腕部—非目标蓝块接近段失败。B2最接近的候选仍差约3.59/3.61mm。两种布局下的红 HOLD 都在下探时实际发生同一碰撞对 STOP。

这比单纯静态拒绝更能支持真实接近空间冲突。但有限候选和局部IK全部失败仍不等于全局物理无解。落点偏移和目标区转腕，不能替代抓取前的方向、腕姿、退让路径和完整机械臂扫掠体积规划。

下一步需在原动作接口内搜索新的接近拓扑：先抬腕/侧向接近、从障碍另一侧对齐、满足腕部净空后下降；将几何不可行、局部IK失败、时间不足和实际跟踪超界分开。预检不能把“底座XY恰好跟随TCP位移”当成动力学保证。[归档复核][replay] [Teacher实现][teacher]

### 3.3 F2 蓝物体搬运是外部障碍问题

B2 蓝左、蓝右的首拒是持物蓝块—桌面红块：间隙4.190/7.970mm，原余量8.226/13.948mm。保持外障余量后依然失败；F1的内部相对误差方案不能用于这两条。

需要持物先离开另一方块的影响区，再转移或转腕的轨迹拓扑，按实际物体和机械臂的完整扫掠体积判断。更高抬升已实际达到约60mm，仍有失败，说明“再抬高一点”不是充分方案。路径长度、转角速度、原90秒限额和冻结H40的速度/加速度约束必须同时满足。[开发终态][development] [归档复核][replay]

### 3.4 OPEN 的困难已从影子投影推进到真实负载转移

B1蓝左的影子模型把实测双指约19mm骤然投到0mm，形成虚拟碰撞；B2已经修正这一投影。

B2红右在32,203tick进入OPEN，前后251个原生状态全部仍有抓持，只有91个满足原≥0.05N桌面支撑。失败帧支撑力为0，但物体底面几何gap仍约−0.030mm；不能描述成已经掉落、浮空或释放。命令2→5.157mm只是两秒打开曲线的前0.5秒，也不足以证明驱动失效。

当前代码见到一帧table_contact即转OPEN、锁定TCP，下一查询又要求当帧有支撑。它缺少稳定支撑确认与有界重新落座的过渡。下一方案应明确 LOWER→SUPPORTED→OPEN→RELEASED→WITHDRAW；支撑丢失时停止使用“物体固定在桌面”的影子假设，重新检查真实抓持状态，并仅允许通过原H40的保持/落座动作。不能去掉支撑前置或把近桌面当已释放。[归档复核][replay] [分段Teacher][staged]

### 3.5 HOLD 和抓取前半段缺少新增检查覆盖

`PlacementTeacher`只有PLACE创建`placement_geometry`，HOLD回退旧逻辑。PLACE在线P10检查也从LIFT开始，PREGRASP/DESCEND/CLOSE未纳入。于是D2的PLACE修复不能自动消除HOLD的腕/蓝块、pad/桌面接触。

B2蓝HOLD最后查询尚有约7.744mm pad净空，之后进入实际桌面接触。仅检查一次初始全路径不足以覆盖底座Z/roll/pitch变化、滤波余速及本次P10扫掠。应统一HOLD和PLACE的接近、下探、闭合逻辑，检查过滤后实际将消费的前缀，并保留全H40原门。已有离线回放显示直接把世界坐标最大残差复制到全部阶段会误拒蓝HOLD，因此需同时控制漏检和误拒。[Teacher实现][teacher] [连续Teacher][continuous] [归档复核][replay]

### 3.6 移动与保持存在同时性和可控性缺口

B2 F2蓝HOLD抬升26.65mm、连续保持28.478秒，位移峰98.32mm；但≥80mm只在13,468–17,683tick出现，保持达1秒从21,964tick才开始。保持合格期最大位移75.03mm，30,001个任务状态同时达标为0。原评分正确。

当前高层HOLD追一个固定抬升点，未把实时底座位移用于后续合法EEF路径选择。允许的研究是验证“现有EEF/夹爪路径能否使冻结WBC产生符合方向、持续覆盖持稳窗口的底座响应”。`M_FREE`明确禁止`base_poses`，循环使用`install(adapter,action,None)`。不能新增底座通道、改F2条件或改WBC来声称这个问题解决；若在原通道内无法做到，须报告动作接口/任务可控性缺口。[归档复核][replay] [M_FREE接口][mfree] [原评分][score]

## 4. 为什么慢，为什么现在还不能把N2作为科学后端

由原生tick乘2ms再除受管批次wall，B1 RTF≈0.13422、B2≈0.13266，合并≈0.13336；1秒模拟时间需要约7.50秒墙钟。这是包括初始化、观测、处理窗口、H40、积分和封存的端到端数值。

代表性N1短卡wall138.89秒：150ms回调窗口49.53秒，advance-to-pause44.72秒，H40安装26.07秒，RGB9.81秒。warmup53.02秒与子段嵌套，不能再相加。进程树CPU下界133.45核秒，表明有真实计算，49.53秒不全是空等。

N2相同8卡从1128.48秒降到566.24秒，吞吐1.993倍；每卡仍约139–142秒。它能让两个独立回合并行，不能把单回合变成实时。

N2有两个缺口：F2稳定卡早期350ms的EEF差7.658mm大于5.999mm，扣初始偏差后仍7.728mm；第二组STOP时同伴已完成，缺少反向继续积分证据。状态漂移的确切原因仍未识别；不能直接称为正常抖动，更不能扩大容差去适配观察结果。

MuJoCo3.2.7文档支持用独立`mjData`并行独立采样，但本项目还包含ROS、MPC、owner、渲染和缓存；更适合保留每world独立进程树。仅复制qpos/qvel也不保证动态续跑等价，warm-start、执行器状态和用户控制状态均可能影响后续。因此不把状态回放冒充原执行恢复，不为并行改模拟器/积分器。[并行实绩][performance] [MuJoCo3.2.7][mujoco]

## 5. 下一阶段决策

优先级：P0统一观测/参考/几何与接触语义；P1六项机制的新有限原生诊断；P2冻结唯一组合Teacher完成新12格；P3新QUAL48；P4复用未使用预算的M2-BASE/SRC。N2工程独立推进，不阻塞Teacher修复；科学后端在完整12格发车前固定，此后不按分数或资源静默切换。

同一冻结Teacher可包含按原指令和可观察几何确定的多个专门技能，但选择规则需在评价前固定。不能评价后把各版本成功格拼接。12/12是既定覆盖准备门，不是总体可靠性证明；当前PLACE0/16首先是结构性问题，不以统计效率为理由事后改门。

目标多任务训练应先得到每个任务完整的成功链。失败前缀可用于归因，不自动当成功BC标签；未执行的H40尾部不能伪装成真实消费。RoboDojo的源域未来状态学习和目标命令监督维持区别，RealMix继续BLOCKED。20/80仅指旧目标C20与新目标数据访问比例，并非真机/仿真比例。长程语义理解、组合指令和记忆要在目标闭环成立后独立评测，不能由手写Teacher的状态机成功直接推断SLA具备长程规划。

详细工作包、预算、晋级与停止条件见[执行协议](../../../docs/protocols/SLA_D3_CONTACT_AND_FEASIBILITY_20261118.md)，可复算数字见[AUDIT_FACTS.json](AUDIT_FACTS.json)。

## 固定来源

[development]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4d597259f2968d01c1694f9a9ee93dbc717695d1/reports/WM02/sla-d2-teacher-parallel-20261118/TEACHER_DEVELOPMENT_ACTUAL.md
[qualification]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4d597259f2968d01c1694f9a9ee93dbc717695d1/reports/WM02/sla-d2-teacher-parallel-20261118/TEACHER_TRAIN_QUAL_ACTUAL.md
[performance]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4d597259f2968d01c1694f9a9ee93dbc717695d1/reports/WM02/sla-d2-teacher-parallel-20261118/PERFORMANCE_PARALLEL_ACTUAL.md
[replay]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4d597259f2968d01c1694f9a9ee93dbc717695d1/reports/WM02/sla-d2-teacher-parallel-20261118/D2_PLACEMENT_PHASE_REPLAY_ACTUAL.md
[teacher]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4d597259f2968d01c1694f9a9ee93dbc717695d1/src/go2arx5_vla/world_model/multitask_teacher_d2.py
[geometry]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4d597259f2968d01c1694f9a9ee93dbc717695d1/src/go2arx5_vla/world_model/multitask_placement_geometry_d2.py
[staged]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4d597259f2968d01c1694f9a9ee93dbc717695d1/src/go2arx5_vla/world_model/multitask_staged_placement_d2.py
[continuous]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4d597259f2968d01c1694f9a9ee93dbc717695d1/src/go2arx5_vla/world_model/mobile_continuous_teacher_v2.py
[mfree]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4d597259f2968d01c1694f9a9ee93dbc717695d1/src/go2arx5_vla/world_model/mobile_wbc_reference_v1.py
[score]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4d597259f2968d01c1694f9a9ee93dbc717695d1/src/go2arx5_vla/world_model/multitask_score_v6.py
[mujoco]: https://mujoco.readthedocs.io/en/3.2.7/programming/simulation.html
