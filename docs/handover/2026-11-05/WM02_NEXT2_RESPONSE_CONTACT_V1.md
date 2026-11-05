# WM02 NEXT2：显式动作响应与连续接触执行

日期：2026-11-05，Asia/Singapore。执行基线：PR #357公开提交 `bef9a1a16f5f2d8490764fba6495f6179a165f1e`；实际三模型训练源码 `7c186104e9c0469e4ed0b63c169772e1f2a659c7`。本次仅计划；不启动worker、不修改冻结控制器、不复活CLOSED lease。首先读根AGENTS、[NEXT1原始报告](../../../reports/WM02/next1-20261105/NEXT1_FINAL.md)和本计划JSON。

## 0. 审核结论与证据边界

依据固定HEAD的 `target_training.py`、`target_training_v2.py`、`target_arx.py`、`next1_contact.py`、`next1_episode.py` 及原机读报告：

- 原生40次安全PASS、174920tick，28个有效窗口/672转移（20 TRAIN窗口来自4父，8 DEV窗口来自P07/P08），3×200主更新+3恢复探针，所有恢复误差0。接触正例和Actor更新0；不是单模型600步或40次完整任务成功。
- S0/S1已经证明在本地包络内稳定前缀和真实1mm动作作用可辨。不能继续把“物理没有动作信号”作为主要解释；不重复全部初始化资格，除非相关条件或执行器改变。
- 神经M0历史FAIL不改：SCALED相对NO_ACTION的H8/H16动作差误差只改善6.9%/11.7%，两DEV父H16方向均1/3。ARX分别0.0635/0.1593mm，SCALED为0.3473/0.4811mm，ARX两DEV父均3/3方向正确。
- 同前缀未来目标干预的H16世界TCP响应范数：RAW约0.000071mm、SCALED0.0120mm、ARX0.5887mm，真实匹配重启约0.6179mm。SCALED仅约1.94%真实幅度，ARX约95.27%。模型内干预不是物理严格counterfactual；独立重启噪声与真实作用必须分开。
- ARX本身是数据学习的局部动力学模型。神经模型失败不必阻止合格ARX的新父预测验证。ARX尚无在线控制收益、接触或大幅运动证据，不能用1mm数据外推100mm接近。
- 接触有两个独立问题：到达后不能持续保持；原8s端点又不足以衔接下降。最后预接触确认时仅余0.8s，名义80→20mm下降在原0.06m/s下至少1s，尚不含加减速、闭爪和保持。延长scope只解除时间不足，不保证漂移被修好。

## 1. 权限、范围与自主接续

NEXT1已结算，28worker退出、58lease CLOSED。新物理和GPU任务须经现有owner/registry/Journal/资源锁登记新的有限scope和不可变run；旧命令不能重放。可先在已授权资源做零积分分析、代码和静态检查，不逐小修请求GO，但不能伪造外部许可。

冻结WBC/腿权重、控制数学、增益、cost、solver、dt、机械及接触安全；不真机、不主动行走、不正式Actor/PPO/GR00T、不改系统网络/他人任务、不建第二调度器。最多2个既有自由空间slot，接触先1slot，同一时间总实例≤2，一个合法learner。

建议申请新有限窗口最多6小时、96次原生尝试、600000tick，资格/失败/拒绝/预热全部计入。**接触任务段申请上限20s，原生启动另计**；这是新版本专用范围，不把旧8s日志改写。预算必须按完整路径、原速度/加速度证书、settling和安全退出预先计算；20s内不可完成则缩短工作链，不越权延时。原站点更严限制优先，结束前至少20分钟收尾。场地/时间未批准时只做离线分支。

## 2. M0：先公平比较和因果审核，不追加物理

固定共享Replay `0b889a835a155b5e2697e3a139b49a4e35ae83e096e05a97d4e3b2c50d74a0de`。只有20个TRAIN窗口/4个TRAIN父，672不是672个独立序列。P07/P08已打开，仍保留DEV但不冒称新盲测；对4个TRAIN父做leave-one-parent-out，每折重拟合全部normalizer/feature scaling，候选和超参数先冻结再看DEV。

必须完成以下对照审计：

1. ARX使用当前状态、前一状态增量、全特征TRAIN标准化、一步delta回归和ridge闭式拟合；GRU采用16步自由展开的106维归一化均方loss。不能把差异全归于网络结构。给小型神经模型同一因果features及一步目标作受控诊断，检查TRAIN是否都学不出作用，再区分优化不足和多步漂移。
2. 原 `NO_ACTION_V2` 在 `encode()`中将整个action清零，该函数同时用于历史burn-in和未来预测。它是“所有参考缺失”对照，不是严格的“仅去掉未来候选动作”。保留旧结果；新增future-choice-only mask时保留过去参考、未干预的BASE/夹爪参考和clock，只遮蔽未来被选择的EEF扰动。不能把两个不同消融混称。
3. FREE与TEACHER_FORCED分别报告。SCALED H16 .0338mm teacher-forced与.4187mm FREE的差距提示局部预测和递推都有分析价值，不能把特权单步结果当部署预测。
4. 对同前缀分别更改未来动作，报告原生匹配重启差、模型差、H1/H8/H16符号和幅度。每父HOLD仅2次，其标准差和3sigma只是探索性噪声证据，不作精确统计显著性宣称。
5. 检查106维冗余状态的一致性：由预测qpos离线FK重建的TCP与独立预测TCP、SO3和ctrl。若新模型只输出局部TCP/BASE，明确这不是完整安全world model。

产物：`MODEL_FAIRNESS_AND_CAUSALITY`、逐父误差/梯度/有限差分、两个真实候选spec。该阶段0新增物理，不能再采同一面板来代替诊断。

## 3. M1：显式响应模型优先，神经残差为条件选项

A. 冻结原TRAIN-only ARX（ridge=1），完整复现，作为可进入前瞻验证的响应模型，而不只作陪衬。

B. 显式分开HOLD自然演化和动作增量：

`y_hat(h | history,a) = b_h(history) + R_h(history, reference(a)-reference(HOLD))`。

首版R用正则化线性/有限脉冲响应，强制R(0)=0；用原参考的时间分布预测H1/H8/H16。当前只有局部X/Y变化被采样，未测的Z/旋转/夹爪作用不能填0当作已经辨识。过去实际参考和控制记忆可用，未来achieved state/真实未来renewal时间绝不可作输入。

C. 只有TRAIN交叉验证出现可重复残差，才增加小残差网络；采用 `r(history,a)-r(history,HOLD)` 保持零动作锚点，同时保留绝对状态监督，防止旧M3共同偏移复现。优先紧凑输出，不重新盲拟合全部106维。

闭式线性拟合先用CPU。特征对齐神经诊断和小残差最多两个配方，各200主更新+1独立恢复探针；仅当预登记的TRAIN收敛诊断支持欠拟合，才继续到总800端点，保留200结果、另记最终恢复探针。不能看DEV决定临时延长或选择有利seed；不默认5000步。

**模型晋级与神经M0分离**：历史M0仍FAIL。新预测器建议目标为H8/H16动作差误差优于NO_ACTION至少20%、各父3个方向中3/3（2/3不足75%），绝对误差不显著劣于ARX，容差由TRAIN重复误差预登记。B/C不及ARX就保留ARX，不让神经模型必须胜出成为整个项目的前置。在线选择仍另验收。

## 4. V：冻结ARX/候选后，使用未触碰的新父场景

保留P09–P14原TRAIN、P15–P16原DEV标签。本轮所有八父先做零更新前瞻，P09–P14不拟合、不改称正式DEV；今后若训练使用它们另版记已曝光。模型/参考/阈值先冻结。

建议每父4原动作HOLD/+X/−X/+Y1mm＋1次HOLD重复，共40次有界尝试；必要的8次独立scene资格另计，最多48次。先保存每次决策前四候选预测，再执行预登记动作。所有拒绝、未稳定和失败保留分母。匹配重新初始化不冒称严格同状态fork或完整oracle。

首要检验是前瞻预测。满足幅度/方向与绝对误差后，才提出另一项有限局部EEF选择对原强规则的试验；ARX不能因低RMSE直接获得安全规划、接触规划或正式Actor资格。回退规则时分开报告纯模型、回退比例和混合表现。新父失败不在同批训练后再称盲测。

## 5. C0：查清预接触失去连续性的直接机制

对本轮8次接触专用初始化/接近记录做零积分审计：world和BASE-relative TCP、期望参考及其后缀、solver/消费关联、deadline/renewal、TCP速度、预接触slab逐项余量、pad/table/object接触、BASE运动。定位首次进入和离开slab时**哪个信号先变化**。

区分轨迹到时未到达、目标/续约/取消造成参考跳变、恒定且正确消费的参考下实际离开、坐标/抓点错误、BASE-arm耦合。现有54.76mm残差与10个连续预接触样本不足以证明是某一个bug。selected-policy仍DERIVED就明确标注；不把静态IK轨迹或名义到达时间当作actual attainment。

## 6. C1/C2：同一原生进程里的连续状态机

新薄入口采用：

`INIT → STEADY → APPROACH → PRECONTACT_HOLD → DESCEND → CLOSE → SUPPORTED_HOLD → RELEASE → WITHDRAW`。

各阶段用当前实际pose/contact/velocity触发转换；旧run的末态或PASS文件只作资格证据，不作当前状态。全过程不reset仿真/MPC、不复制qpos/qvel假恢复、不每阶段重启。当前 `next1_contact.py` / `next1_episode.py` 精确固定grade/160atom，要版本化实现新有限入口，不能删除检查或只修改max_ticks。

先只跑到PRECONTACT_HOLD：建议原几何条件连续满足1秒（20个50ms样本）且低速，低速阈值用TRAIN预登记；原接触/碰撞阈值保持。随后至少还能继续保持，不能将旧10样本瞬时通过改称新资格。参考续约350ms、TTL为source+400ms不变，拼接遵守原速度/加速度证书。

若固定参考正确而实际继续离开，先查实际控制选择、BASE协调、跟踪动态；不靠经验分轴目标补偿、调增益、强行下降或原地无限重试。

通过后按新批准任务段逐级下降到20mm合法候选、实际接触测量、有界闭爪、受支撑保持、释放、撤离。每次转移检查剩余时间是否足以完成该段及安全退出；不足则保持/退出并报 `INSUFFICIENT_REMAINING_TIME`。不能通过超速或放宽限位弥补8s端点的物理不可能。首个成功目标是稳定预接触和真实接触转移，不保证完整抓放。

新接触片段必须记录完整路径历史、实际安装/消费、夹爪内部记忆、左右pad与support事件；数据schema从固定goal切到piecewise轨迹须版本化，不能仅用最终目标伪标整条命令。事件正例实际出现后才训练事件头。

## 7. 自主执行与验收

顺序：E0_REVIEW→M0_FAIRNESS→M1_RESPONSE→V_PREDICTION；独立接触线E0→C0_TRACE→C1_HOLD→C2_CONTACT。一条研究线科学FAIL不阻断另一条已合规工作，原生安全/owner失效则立即停止该动态分支。无需为了收集更多保持数据回到大规模泛化面板，也不被神经模型必须获胜所阻挡。

最后交付独立标签：`LOCAL_RESPONSE_PREDICTOR_READY`、`PROSPECTIVE_RESPONSE_VERIFIED`、`PRECONTACT_CONTINUITY_VERIFIED`、`TARGET_CONTACT_TRANSITIONS_AVAILABLE`。`POLICY_READY`不由这些标签自动推出。逐父指标、输入身份、数据/model/replay/binary SHA、真正运行CLI、退出码、主更新/探针和全部成本/失败/worker终态均写入既有Journal和报告，不建另一调度账本。

本次文档发布新增原生积分0、模型更新0、Actor更新0、现场worker0；具体模块与新CLI必须由现场Codex实现和测试，计划名称不冒充已存在命令。
