# WM02 配对接触动作辨识 · 2026-11-04 SGT

**已实际采集、训练和评价。动作条件在本轮位置差异和部分排序上提供了增量，但接触转变与实际选动作的价值仍未建立；Actor不启动，主训练不扩到5000。** 本轮不是重复PR348的旧872条烟测。四个DEV状态上的独立修正锚点诊断也已执行，已获得真实“保持张爪／闭爪／延迟闭爪”的抓持分化，但尚未用于模型训练。

| 身份 | 完整SHA |
|---|---|
| 未合并PR348固定基线 | `50301d6ee5d38de72e2461af63dcef8e9ba9c850` |
| V1协议／采集器预先冻结 | `2387ddb188213530a5f952d2ba8db101cab495a6` |
| V1实际24场景采集 | `39270b752da14c5ae15d2bd67f730e067f3890a1` |
| 实际GPU两臂训练／重载／评价 | `84204cf8d23e151cd1af7da7cea7f2521f662f0f` |
| 固定预测的离线差异／覆盖分析 | `406b29b91d6772dcfac0719cb64d13731a73a917` |
| V2四个DEV预闭爪状态的实际原生探针 | `027ab08244dbd4bbe1085d1dfeb94c4fa91b8b28` |

独立checkout和分支 `research/wm02-contact-identification-20261104`；未切活动source，未改旧capsule／数据／checkpoint／WBC／安全阈值，未重开旧队列。私有原始轨迹、模型、vendor资产及主机绑定路径不进Git。PR348和上游WM02 PR均未假定已合并。

## V1：真实配对数据与完整恢复

[冻结协议](../../../experiments/wm02/contact_identification_v1.json)在首轮采集前提交。原WM01 TRAIN索引2..25，24个真实不同source/goal/yaw布局；16 FIT、4 DIAGNOSTIC_DEV、4 ONCE_HOLDOUT按完整父场景划分。它们都是原TRAIN来源的研究内分区，不是原DEV/TEST。布局变化范围有限：source xy跨度约15.14/24.33mm，goal约23.21/24.02mm，yaw跨度0.12881rad；相同机器人、物体类别和摩擦设置，不能泛化成广泛物体能力。

每场景四种锚点，各五种固定候选，16×50ms。400ms边界安装第一块，第二块按预定规则重新锚定实际TCP、零增量并保持第一块末夹爪命令。训练预测的第二块只用**预测TCP**，不读未来实际TCP。每个锚点保存相同8原子步真实历史、完整控制／积分／IK／RNG／Teacher／monitor状态；Teacher阶段仅供构造和分层，不进入WM。

候选是保持、名义、小幅左／右和夹爪时序。物理V2动作、world平移／旋转、原生row rot6d、H40末点持恒；未执行尾部不是监督。夹爪工作区间[.002/.044,.042/.044]，不改变机械限位／驱动。稳定抓持时不构造无支撑松开来凑失败。

[原生执行回执](COLLECTION_ACTUAL.json)与[数据审计](DATA_AUDIT.json)：

| V1实际量 | 数量 |
|---|---:|
| 布局／锚点／五候选完整分叉 | 24 / 96 / 480 |
| 主分叉真实转移 | 7,680 |
| 同动作重复／只改未消费尾部 | 1,536 / 1,536 |
| Teacher构造前缀 | 4,416 |
| V1新增真实50ms转移合计 | 15,168 |
| live／原S1 shadow积分 | 379,200 / 379,200 |
| 完整恢复重复／尾部物理等价 | 96/96 / 96/96；最大误差0 |
| 拒绝／提前停止／UNRUN／场景失败 | 0 / 0 / 0 / 0 |

先执行场景0的有限pilot，再执行互不重叠的1..23；没有把pilot当重复采样。四进程、每进程固定一CPU，原dt、S1和native watchdog不变。所有实际raw／安装／25 tick消费记录均通过原aligned检查。完整状态比较包括控制器缓存和最终完整snapshot，不只qpos。固定底座S的资格不推广为Go2整机恢复。

数值Replay：480×(8真实历史＋16未来)；50维真实arm/TCP/gripper/object schema，context5，四个原生事件；无BASE／腿补零。仅FIT不同真实状态拟合归一化，得到5,375个不同FP32状态、640个8-burn/8-predict训练窗。每臂51,200次窗口访问＝平均80次访问，不是51,200次独立物理经验。

**V1数据SHA256：`24bddf61a4db8ddd0f9605eef1b8a1b906f285b437454503dc88e999e78b7862`。** 重复与尾部物理对照不进入拟合。所有480个预定候选和96个锚点均保留分母。

普通工程问题：第二批的目录使用批内编号0..22，CARD中的全局场景身份1..23正确，首次数值导出因此FileNotFoundError、exit1。修复loader按实际CARD.ident_index找目录，未来collector也改成全局编号；原始文件不改名，成功导出使用新output，**没有重跑物理**。首次失败输出保留。

## 两个匹配模型与同起点评价

[模型机器回执](WM_ACTUAL.json)：两臂均18,905,387参数、5×2048＋GRU256、FP32、不启TF32，同初始化seed、FIT normalizer、AdamW、采样游标、有效batch256／micro32×8、200主更新。无未来动作模型保留真实历史和burn-in，只将未来reference置零。两个固定端点都完成后，才生成一次性留出的预测。

每臂checkpoint1先完整重载再继续；199→200各额外一次续训探针，model／optimizer误差0；200端点再完整重载，游标51200、误差0。**实际400主更新＋2探针更新；Actor／Critic0。** 两个模型同端点、没有从holdout选checkpoint。

神经编译器的SE3数值校验从1e-7对齐既有native `pose_matrix` 的1e-6容差，避免FP32递推姿态被更严的重复数值门拒绝；原物理动作double边界拒绝、原生机械／接触／速率阈值均未更改。相关编译器和饱和边界回归保留。

| checkpoint200 | SHA256 |
|---|---|
| 动作条件WM | `de5e4afb19bf74c297954d474c2993cb408ee8f2cc28b051bfadb1cbfb2c9ede` |
| 无未来动作WM | `120ffee5d279f2fbb1f8985909dc66a9e950523b329a1bc73110db3270df8e05` |
| 评价前保存的holdout预测 | `61dfb62af6c99f81044ed30a0d367285c8ce3bd12b669ac4991efae08c8c8166` |

H1/H8/H16始终用同80条候选轨迹，即**4个物理父场景×4锚点×5候选**。分叉和动作对彼此相关，不作80个独立episode或160个独立试验的显著性结论。未来动作错配只在同场景同锚点内循环候选，真实历史／初始hidden／当前状态不变，hidden逐位相同已断言。实际错配不是只改尾部：raw平移RMS0.721mm、旋转RMS0.001092rad、夹爪fraction RMS0.302，前8已安装位置向量RMS差5.682mm。

下表为物体位置误差，单位mm，**窗口RMSE / 最后一步RMSE**：

| 同80起点 | 动作条件 | 无未来动作 | 仅未来错配 | 状态持恒 | 恒速度 |
|---|---:|---:|---:|---:|---:|
| H1 0.05s | 0.692 / 0.692 | 0.796 / 0.796 | 0.692 / 0.692 | 0.771 / 0.771 | 0.606 / 0.606 |
| H8 0.4s | 3.203 / 5.788 | 3.007 / 5.063 | 4.048 / 7.103 | 5.418 / 9.733 | 3.684 / 5.405 |
| H16 0.8s | 4.622 / 5.908 | 4.904 / 6.897 | 5.716 / 7.072 | 7.745 / 9.368 | 7.159 / 12.514 |

H16改善，但H8仍差于无动作及恒速度末端；姿态H16末端0.006524rad，仍差于持恒0.003139rad。完整16点同起点曲线保存在WM_ACTUAL.json，不混用窗口与endpoint。

[固定预测离线分析](CONTACT_ANALYSIS.json)：160个预定动作对，100对实际物体末端差>1mm、60对平坦。动作条件的100个有效方向中正确56，未来错配44，无动作100个预测并列。配对物体差向量RMSE分别6.304／8.804／7.718mm，最后一个也是零差预测基线。稳定抓持方向31/36，已建立抓持附近8/28，释放附近17/36；不能只看总体位置改善。

诊断进展排序中59对真实进展有差异、101对平坦。动作条件51/59符号正确，未来错配26/59，无动作59对全部并列。错配排序从已保存预测的**逐位一致候选置换**及冻结进展差推导，没有重跑网络、改变历史或重新拟合。

接触问题仍明显：16个真实grasp结果不同的动作对，模型差异方向16/16正确，但最大概率差仅0.0376，真实差为1；grasp差RMSE0.3115，接近无动作0.3162。支撑释放H16 grasp Brier0.3036，无动作0.3108；不能把方向对解释为能可靠识别释放。全体H16 Brier0.07603／0.07773，当前真实事件持恒0.078125是**特权当前标签**基线；两个学习臂都没有直接输入该标签。四层接触结果单独保存，不让稳定状态稀释释放问题。

S驱动不消费H40后32行：96次物理尾部对照完全相同，而模型末端位置改变RMS **1.848微米**。说明模型并非严格尾部不变；该效应小于本轮mm级误差，不推广为目标WBC全preview不敏感。

## 选动作结论与V2纠正诊断

冻结训练代码给出“超过名义和固定hash随机动作、16锚点中至少60%”的诊断门槛，本轮为0/16。这个60%规则在采集后、训练前固定，**不把它宣称为采集前预注册的正式任务验收**。离线oracle审计发现，真实最优候选对名义动作的最大改善仅8.35e-6，所有16锚点都低于同一1e-5门槛，连oracle也0/16。因此0/16不能单独证明WM选动作失败；当前设计没有提供足够可改善的进展选择。模型建议的新实执行次数0；记录分叉上的选择成绩不冒充新在线建议兑现。

覆盖分析还发现V1主分叉“未抓住→抓住”转移0，96次grasp消失全部在支撑后释放类，非意外失抓资格。接触前锚点1.2s距离真实抓持尚远，grasp_onset锚点2.4s时已建立抓持。物体效应有信号，但该配置不足以回答接触建立和失抓机制。

因此单独冻结[V2 DEV探针](../../../experiments/wm02/dev_onset_probe_v2.json)，只使用原DIAGNOSTIC_DEV父16..19的前缀和恢复文件，在原prefix-04 **1.6s完整恢复**、相同8步真实历史执行既定五候选；没有新增采样或重选V1四个holdout，没有改V1协议／数据／checkpoint或追加训练。

实际新原生结果：4×5条完整分叉，主320＋重复64＋尾部64＝**448条新增真实转移**；live11200／shadow11200积分，重复与尾部各4/4通过、完整状态误差0，拒绝／terminal／UNRUN0。保持张爪4/4未建立grasp；名义、左移、右移各4/4建立；延迟闭爪仅1/4建立。20条主分叉中得到13次真实grasp建立转移；重复／尾部另有8次建立（独立父场景仍仅4，不能称13个独立布局）。这证明纠正锚点能产生动作可辨识的接触结果，**不证明旧WM已经学会它**。

V2的36份私有流／恢复／历史文件manifest SHA256：`005d3615e95b2d74c728201ad402f13d426ef9e2fc12bd6e1e46e72701d71ca1`。它不是V1训练NPZ身份，不混入其FIT或holdout。V2只做数据／恢复资格，WM和Actor更新均0。以后新版本须冻结新的FIT和未看过的留出组；不把这四个用于设计的DEV当新测试或悄悄训练它们再报泛化。

Formal reward仍未取得成功、空手接近、早放、抓住不放、安全失败五类真实资格。本轮进展是公开定义的诊断量，不是原生成功评分；缺reward未阻止动力学训练，也没有为凑类别合成失败。

## 目标隔离的具体证据和工程事项

[本轮只读核实的原始失败摘要与完整SHA](TARGET_ISOLATION_EVIDENCE.json)，原source `ee4f86d5fc9fd8a54008329394a0e0f1738526c6`，owner `EXISTING_LOCAL_PC`。原lease CLOSED，旧两任务许可于2026-10-30 19:31:39.836 SGT结束，累计5 starts已消耗。不是GPU容量或WM训练失败。

1. **T1：**实际200个任务积分后，`ORIGINAL_CARTESIAN_RATE_REJECT_AT_ACTUAL_REFERENCE_JOIN`；旧column解码拼接角速6.84830rad/s，row解码离线0.228107rad/s、线速0.056133m/s。row约定已有代码修复，离线gate通过不代表剩余IK、控制安全和任务通过。另在warmup steps38–40有3次右滑轨越界，最大58.73微米，原机械上界.044m。
2. **T2：**任务积分0，初始化steps24–27已出现7项腿部关节机械越界，原进程随后`OWNED_NATIVE_PROCESS_EXIT`。例FL_calf最差−2.91134rad，原下界−2.7227，越界0.18864rad；RR_calf也越界。需诊断冻结腿策略的reset初态、模式切换、启动与参考顺序；**独有根因尚未确定，更未修复**，不能用夹爪／rot6d修复来替代腿部证据。
3. 原raw audit、terminal audit及NATIVE_SAFETY_FAIL的SHA已重新读回。新窗口前须在新独立源码准备实际row6d wire／真实参考join检查、夹爪内区间reset与全tick限位审核、腿部初始化高率全关节审核、actual ACK／消费时钟、全身接触和站立T1传感评分绑定；不改腿权重、WBC、求解器或安全阈值。
4. 新窗口由**用户明确的新有限动态范围**，经原`EXISTING_LOCAL_PC` owner、既有profile／registry／lease和原生ACK／stop／安全准入确认。旧CLOSED scope不重开，不把GPU权限或本次S诊断当动态许可。合格后可以先合法顺序采集；同状态目标分叉仍须完整restore或重构资格。当前没有合格目标动作CLI，不伪造目标qualify命令。

本轮目标节点／目标积分／完整32秒任务／正式任务成功／Actor／Critic／PBS均0。首个T域阻断是有效动态准入缺失，且已有未解决的初始化安全证据。原隔离保持；本轮未运行目标physics或重新提交exit78。

## GPU、加速与实际成本

[GPU回执](GPU1_ACTUAL.json)：GPU1 UUID `GPU-845348f1-1511-dc19-0877-cc97658e3edd`，PCI83:00.0，NVML16376MiB，Torch CUDA容量16709255168字节。只暴露该UUID，allocator上限4GiB，CPU4–7，原owner锁。两个臂顺序执行以保证同配方和显存余量；CPU源物理使用四进程加速，不修改物理频率。

实际进程NVML采样190次，峰值1278MiB、设备最少空闲14615MiB。动作／无动作主训练阶段Torch峰值allocated641430016／717937664，reserved708837376／897581056字节；这些分臂记录不替代含恢复／评价的全进程NVML峰值。独立模型训练在16GB上足够；sim＋ROS2/WBC＋渲染＋训练同时驻留仍未测。

| 实际范围 | CPU秒 | 墙钟秒 | 退出与lease |
|---|---:|---:|---|
| V1 pilot1场景 | 19.476443 | 19.930914 | exit0 / CLOSED |
| V1其余23场景，四进程 | 878.896572 | 235.247923 | exit0 / CLOSED |
| V2四DEV锚点，四进程 | 27.857436 | 7.226752 | exit0 / CLOSED |
| GPU两臂＋恢复＋评价unit | 113.457058 | 108.396822 | exit0 / MainPID0 / CLOSED |

源物理CPU为子进程实测求和926.230450秒，GPU为unit含回归113.457058秒；单位GPU占用墙钟约0.03011卡时，不是全会话计费。预审／传输／staging／PC测试／发布未完整计量，货币成本未计。没有新PBS。最近只读快照GPU1空闲15899MiB；GPU0计算进程已只读核实属于另一Unix账号，未发送信号或改它们，三个旧队列PID均不存在。旧队列不由本轮重建；三个PC lease和一个GPU lease均已关闭。

**本轮合计：15,616条新增真实S转移＝8,000主分叉＋1,600重复＋1,600尾部＋4,416构造；live390,400／shadow390,400积分，500个完整16步主分叉；真实WM optimizer.step402，Actor0、目标0、完整任务0。** V2复用四个既有DEV父场景，不把布局数加成28。

## 实际入口、退出码与下一步

原有Journal中的一次性worker实际调用如下，变量仅用于脱敏路径；关闭的身份不重放：

```python
# PC pilot和remaining，源39270b...，均exit0
collect_contact.main(['--spec', str(root/'EXECUTION_SPEC.json'),
                     '--output', str(root/'collected-01'), '--run-id', 'collected-01'])
# Core GPU源84204c...，两臂400主＋2探针，exit0
train_contact.main(['--spec', str(root/'EXECUTION_SPEC.json'),
                   '--output', str(root/'matched-200-01'), '--run-id', 'matched-200-01'])
# PC独立DEV探针源027ab0...，exit0，不训练
probe_contact_onset.main(['--spec', str(root/'EXECUTION_SPEC.json'),
                         '--output', str(root/'collected-01'), '--run-id', 'collected-01'])
```

实际数值导出与分析命令：

```bash
PYTHONPATH=src "$WM02_CPU_PYTHON" -m scripts.wm02.build_paired_replay \
  --roots "$WM02_PILOT/collected-01" "$WM02_REMAINING/collected-01" \
  --output "$WM02_PRIVATE/paired-replay-02"
# exit0；之前paired-replay-01目录别名错误exit1，原输出保留
PYTHONPATH=src "$WM02_CPU_PYTHON" -m scripts.wm02.analyze_contact \
  --replay "$WM02_PRIVATE/paired-replay-02/PAIRED.npz" \
  --predictions "$WM02_PRIVATE/HOLDOUT_PREDICTIONS.npz" \
  --result "$WM02_PRIVATE/WM_RESULT.json" \
  --output "$WM02_PRIVATE/CONTACT_ANALYSIS_02.json"
# exit0；只分析冻结预测，新增更新0／积分0
```

world_model36用例normal和`-O`通过；Torch15用例normal和`-O`通过；GPU原环境新增学习合同3用例normal和`-O`通过。完整恢复资格来自实际96＋4次原生重复，不来自这些fixtures。全仓库CI NOT_RUN。

下一科学步骤是采用已通过原生验证的预闭爪锚点，单独冻结新FIT和新的未观察留出，再做匹配训练／接触差异评价；同时补可被候选改善的进展或任务条件。本轮不把同一批无接触建立的窗口再训练4800步，不把DEV探针或诊断分数变成Actor reward。

现有可立即执行的复核命令（无需GPU或目标许可，不重开lease）是上面的`analyze_contact`，改用新的`--output "$WM02_PRIVATE/CONTACT_ANALYSIS_REVIEW_03.json"`；它复核冻结数据／候选顺序、oracle机会、真实接触覆盖和配对差异。新的FIT采集范围和在线模型建议兑现尚未实现／获资格，不提供假命令。私有ACTUAL_COMMANDS记录了实际绝对路径、worker／stage SHA及执行退出码。
