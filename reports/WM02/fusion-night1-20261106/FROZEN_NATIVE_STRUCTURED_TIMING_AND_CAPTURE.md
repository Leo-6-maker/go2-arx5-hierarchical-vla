# FUSION 已封存原生面板、结构响应、接触时序与目标观察实测

本报告更新到2026-11-06已封存回执，公开数值见 [ACTUAL JSON](FROZEN_NATIVE_STRUCTURED_TIMING_AND_CAPTURE_ACTUAL.json)。最终192次DEV评价没有发现WM相对强静态校准的收益；新结构模型两臂均未达到预冻结TRAIN增益门槛，精确回退静态。接触时序基线提供了有用的动作条件预测对照，但尚未匹配神经模型的完整输入。目标观察已实际保存三对图像，仍未进行目标VLA查询或在线闭环。历史CORR查询的8点真实源后果链已核实，只能绑定该条旧查询。

这是指定封存批次的结果，不是全项目实时状态。后续资产搬运、fresh CORR source/target shadow和新接触输入变体不在本次封存范围。此前的失败更新、负例及原角色继续保留，见 [上一轮学习与原生实测](BALANCED_RESPONSE_LEARNING_AND_NATIVE_REFIT.md) 和 [动作语义审计](SEMANTICS_AND_TARGET_OBSERVATION_AUDIT.md)；不覆盖其历史快照。

## 冻结192次DEV原生面板：强静态校准优于旧WM

原面板源码为 **`0f235865a30f94cd877b7648598823d2039fa6ae`**，B3使用当时冻结的ARX残差MLP。此次没有用新结构模型替换旧B3。原模型、评分、WBC、容差和安全条件完整执行到终态，协议见 [冻结DEV面板](../../../docs/protocols/WM02_FUSION_FROZEN_DEV_PANEL_V1.md)。最终审计快照为2026-11-06T13:59:44.553648+08:00，lease CLOSED、worker退出0、MainPID=0。

六个DEV物理父场景×四个目标×两个独立重启×四方法，共 **192次评价、每方法48次、96个双slot pair**，另有六次场景资格。198/198次均为native PASS且证据有效，每次4103 tick：评价787,776 tick，资格24,618 tick，合计 **812,394 tick**。native PASS表示执行与安全通过；局部XY成功只有14/192，**178个失败全部保留**。六次资格均未达局部XY成功，不计入方法成功数。所有评价的完整任务成功标记均为false。这里是独立重启，不是隐藏状态完全相同的反事实fork。

局部XY容差保持 **0.18584456642062617mm**。评价目标为XY误差加非HOLD动作0.020mm成本。

| 冻结方法 | 局部XY成功 / 失败 | 平均XY误差 mm | 平均评价目标 mm | 动作0/1/2/3次数 |
|---|---:|---:|---:|---|
| B0 直接执行 | 3 / 45 | 0.621529580 | 0.636529580 | 12 / 12 / 12 / 12 |
| B1 TRAIN强静态校准 | 6 / 42 | 0.562524580 | 0.582524580 | 0 / 12 / 24 / 12 |
| B2 匹配反馈 | 3 / 45 | 0.621728745 | 0.636728745 | 12 / 12 / 12 / 12 |
| B3 历史条件WM修正 | 2 / 46 | 0.629529177 | 0.649112510 | 1 / 4 / 23 / 20 |

下表增益定义为“B1静态目标−候选目标”，正值才是改进。原预冻结有意义门槛为 **0.08124448952310876mm**；六父全部完整，无排除。10,000次层级bootstrap、seed366502的区间由最终回执重新计算核对。

| 相对B1的候选 | 等父平均增益 mm | 层级95%区间 mm | 正向父数 | 预声明收益成立 |
|---|---:|---|---:|---|
| B0 | −0.054005000 | [−0.090719497, −0.017225414] | 0/6 | 否 |
| B2 | −0.054204165 | [−0.091585247, −0.014992755] | 0/6 | 否 |
| B3 | **−0.066587930** | **[−0.138153263, −0.001176191]** | **1/6** | **否** |

B3在该局部DEV校准面板上的平均目标更差，未取得预声明控制增益。这既不构成正式TEST或完整任务结论，也不能把安全执行通过改记为任务成功。本面板VLA查询、WM/Actor/GR00T更新均为0。

## 结构响应：TRAIN整父选择后两臂精确回退静态

独立后继源码 **`7c12085a6c858b914423f14b425539fa6e224f1e`** 在原面板封存后运行。`fusion-night1-structured-study-01` 已CLOSED/退出0/MainPID=0，实际入口为 `scripts.wm02.train_fusion_structured`。它复用冻结48窗口：32 TRAIN/16 DEV、4个TRAIN物理父/2个DEV物理父，每个动作有两个独立重启；不产生新原生转移。平衡replay SHA为 `e5139d792d8d2f28e49ea28d7f57fbabbd5e79378e5358b747b2896c851ed795`，TRAIN规则SHA为 `f51d59764b8316a903e5e953205bb90805e82bd0f1d2ead157207213364aa0d9`。

新模型直接预测以当前TCP为锚的H16 XYZ位移，保持平移等变。动作条件臂以TRAIN HOLD曲线加显式动作响应为基底；盲臂从特征和基底同时移除候选动作，采用等动作、等父、等重启权重的TRAIN边际位移。未知未来动作不等于断言未来HOLD，因此这里的公平边际盲臂与旧ARX/HOLD盲臂实验分开记录，旧实验未改写。

固定ridge网格为0.1/1/10/100/1000/INF；仅用TRAIN整物理父LOPO选择，每折24条拟合、8条留出，留出父同时从归一化和静态曲线中剔除。正增益必须达到预冻结门槛，才采用有限ridge。共八个LOPO父折、40次闭式ridge求解、两个最终模型、**0次优化器更新**。两臂均先拟合、保存、精确重载数组并核对TRAIN四候选预测，再打开DEV评价；未用DEV拟合、归一化或选择。

| TRAIN LOPO pooled H16 XYZ RMSE（mm） | ridge0.1 | 1 | 10 | 100 | 1000 | INF静态 |
|---|---:|---:|---:|---:|---:|---:|
| 动作条件 AC | 0.135528449 | 0.114040003 | 0.083827914 | **0.079062058** | 0.088336187 | 0.094353811 |
| NO_FUTURE_ACTION | 0.410847550 | 0.343760174 | 0.262878012 | 0.236887503 | **0.230235367** | 0.230716397 |

AC最佳有限ridge100仅增益 **0.015291753167615427mm**；盲臂最佳ridge1000仅增益 **0.00048103048906122936mm**。两者都低于0.08124448952310876mm，因此实际选中 **INF**。实际回执中的门槛浮点表示为0.08124448952310875mm，完整值保留在JSON，未更改选择规则。

| 实际冻结输出及对应静态基线 | TRAIN pooled XYZ RMSE mm | DEV pooled XYZ RMSE mm | DEV H16 XY RMSE mm | 相对自身静态基线的TRAIN/DEV增益 |
|---|---:|---:|---:|---|
| AC / TRAIN平均动作响应 | 0.081103415 | 0.077067295 | 0.061252819 | 0 / 0，精确相同 |
| 盲臂 / TRAIN边际响应 | 0.229516378 | 0.227866961 | 0.358126581 | 0 / 0，精确相同 |

两模型的数组重载完全一致、TRAIN候选预测最大恢复误差0m；闭式模型没有优化器状态恢复。AC优于盲臂说明已知动作响应有信息，不能据此宣称学得残差有独占价值。此次强静态对照正好等于AC最终输出，没有为相同行为再运行一轮原生面板。模型冻结SHA、各父/各时域误差、全部LOPO分数与HOLD/匀速基线见ACTUAL JSON。没有神经训练、GPU调用、专家动作标签或真实控制收益声明。设计合同见 [结构响应与观察协议](../../../docs/protocols/WM02_FUSION_STRUCTURED_RESPONSE_AND_OBSERVATION_V1.md)。

## 接触时序基线：因果信息限制明确，完整输入匹配仍有缺口

`fusion-night1-contact-timing-baseline-01` 在源码 **`1ca3b833139e92019c29029ef6be5a4f368de717`** 上运行 `scripts.wm02.fit_contact_timing_baseline`，CLOSED/退出0/MainPID=0。它以与既有MIXED46神经比较相同的TRAIN manifest，使用28个TRAIN episode、8个物理父、4452个anchor；close72不参与拟合。两臂各六时域头、固定ridge1，共 **12次闭式头求解、2个模型、2次精确数组与预测重载、0次优化器更新**。时域1/4/8/16/32/64对应0.05/0.2/0.4/0.8/1.6/3.2s，逐时域有效TRAIN标签数为4452/4368/4256/4032/3584/2688。freeze SHA为 `345dddd31e2cd6e8cbf200b0dda011d072a1e30c9dbd45d75ecb645e4db27e5b`。

该FIR基线只读九行×十项夹爪/力/内部target/开度/原生事件历史，共 **90项观察特征**；既有神经模型使用927项完整历史。因此这里只是训练暴露匹配，**不是输入或架构完全匹配**。两者盲臂的当前开度来源也不同：旧FIR90重复内部 `memory.opening`，令未来增量为零；原神经replay的 `blind_plan` 重复anchor实际 `executed_reference_EEF10[-1]`，即当前已安装/执行的夹爪参考，不能把两个来源假定为同值。不使用episode clock、phase、parent、goal或close-start。AC只读各预测时域内的预声明命令增量，不能读更晚的程序后缀或未来观测。这个信息合同是 `PREDECLARED_PROGRAM_CONDITIONAL`，未来程序尚未安装/消耗；严格installed-reference-only版本未运行。事件标签仍是特权原生仿真测量，不是仅靠相机。协议见 [因果接触时序基线](../../../docs/protocols/WM02_CONTACT_TIMING_BASELINE_V1.md)。

所有评价都是既已暴露DEV和TRAIN72的回顾开发诊断，不是新独立TEST。C09_COMMON36为36 episode/12父/5724 anchor/30060有效预测；C10_TIMING12为12 episode/2父/1908 anchor/10020有效预测。两个cohort重叠两条episode，不能加成48条独立episode。C10的DEV为六条episode、一个物理父、954 anchor，不能把重叠anchor当独立试验。

以下为C10已暴露DEV的自然分布 `NATIVE_GRASP_END` Brier，越低越好。保留全部三个匹配MIXED46 AC seed，不选最佳seed：

| 固定预测器 | H1 / .05s | H16 / .8s | H32 / 1.6s | H64 / 3.2s |
|---|---:|---:|---:|---:|
| 当前事件保持 | 0.009433962 | 0.166666672 | 0.375000000 | 0.701388896 |
| TRAIN事件频率 | 0.257263452 | 0.263738304 | 0.266410977 | 0.247347668 |
| FIR盲臂 | 0.005924537 | 0.079745878 | 0.196641324 | 0.252037413 |
| FIR动作条件 | 0.004880590 | 0.022219360 | 0.023711914 | 0.038513911 |
| MIXED46神经AC seed363901 | 0.001162805 | 0.004296915 | 0.002641043 | 0.009621865 |
| MIXED46神经AC seed363902 | 0.002370038 | 0.002994911 | 0.001815588 | 0.026618095 |
| MIXED46神经AC seed363903 | 0.003393489 | 0.005163981 | 0.004308134 | 0.004992809 |

C10 DEV上，三个匹配神经AC seed在全部六个时域的自然grasp Brier都优于本FIR AC。这不构成所有cohort/事件上的神经优势：例如C09 DEV H64的FIR AC Brier为0.000436055801，优于三个匹配神经AC的0.000532788818/0.005075231367/0.002083295217。原12个固定神经比较器全部保留，其中六个PREFIX36模型的训练暴露不匹配，另六个MIXED46匹配；本次没有重训神经模型。

时序指标只在声明的六时域网格上寻找首次状态不同，**不是原生首次到达时间**。漏检单列，MAE仅对检出的真实转变计算，不把漏检误算为零误差：

| C10 DEV grasp网格时序 | 真实转变 | 检出 / 漏检 | 假转变 | 召回率 | 精度 | 检出后MAE s | p90绝对误差 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| FIR AC，全部954 anchor | 509 | 506 / 3 | 28 | 0.994106 | 0.947566 | 0.172233202 | 0.8 |
| FIR盲臂，全部954 anchor | 509 | 438 / 71 | 103 | 0.860511 | 0.809612 | 0.676484018 | 1.6 |
| FIR AC，close72的318 anchor | 173 | 170 / 3 | 22 | 0.982659 | 0.885417 | 0.151470588 | 0.8 |
| FIR盲臂，close72的318 anchor | 173 | 136 / 37 | 41 | 0.786127 | 0.768362 | 0.552941176 | 1.6 |

线性输出明确clip到概率[0,1]，不是动作或参考轨迹裁剪。原始概率越界也保留：C09 AC/盲臂46768/29037个（各120240个事件概率）；C10 AC/盲臂13805/9360个（各40080个）。没有新物理转移、原生控制评价、GR00T/Actor更新或自由滚动资格。下一步若补完整927项历史，应是新recipe和新冻结批次，不能把当前90项基线改称完全匹配。

## 目标观察：三对真实保存图像，尚无目标查询或在线控制

源码7c12085a上的 `fusion-night1-target-observation-replay-01` 已CLOSED/退出0/MainPID=0，状态为 `SAVED_TARGET_OBSERVATIONS_CAPTURED_SHADOW_ONLY`。实际入口 `scripts.wm02.capture_fusion_observation` 在T06封存原生记录的3503/3703/4103步（约7.006/7.406/8.206s）完成 **3次保存尝试、3次成功保存、6张双相机PNG、3个绑定观察**。后端实测为Mesa/llvmpipe（LLVM20.1.2，256bits）的CPU EGL路径；没有使用CUDA。

本次6次Python `mj_forward`、**0次积分、0次模型加载、0次VLA查询、0次更新**。4103 tick是原已有记录的长度，不能再次计作新物理。原记录source为0f235865，场景/结果/journal SHA和三次packet、标定、图像及绑定SHA见JSON；不公开像素或状态payload。保存状态重放不等于在线闭环或精确控制器隐藏状态恢复。

固定frame map SHA为 `fc347a59fe6b82f758d198f2db93955ba2697a55e8ec24e490faeda53c2761b1`，来自decision3703的源XML base_link z=.4与目标实际BASE FK，并核对完整静态手臂安装链；没有按TCP/object拟合变换。三次采集均标记EXPLICIT_RIGID_MAP。这不证明相机、任务、视觉分布或行为域等价。

语言实际借用了源T1 prompt，显式记录 `ORIGINAL_SOURCE_T1_PROMPT_ON_TARGET_CALIBRATION_SCENE_UNQUALIFIED_TASK_MATCH`，`task_prompt_matches_target_task=false`；原CARD SHA为 `9494434d910b1aea9d642bd3cf2a8be31849d7a5d5d99388d57accbe7898da2f`。T06自身是CALIBRATION_INTENT HOLD/open，不能把源指令当成T06已匹配本体任务。该采集只证明观察保存与绑定路径实际可用，不证明图像任务成功、目标具身资格或任务迁移。

## 历史CORR源后果：精确旧链通过，fresh draw仍须独立验证

只读审计 `CORR_SOURCE_OUTCOMES_AUDIT01` 对原源源码 **`887e65629291f16f73863e2b111123007542d268`**、S-CORR-2048 checkpoint manifest **`94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38`** 的归档query0，完成 **227/227项链检查**，状态 `PASS_ARCHIVED_CORR_QUERY0_PREFIX8_SOURCE_CHAIN`。实际保存的relay profile是B1，不从历史模型名的B0后缀推断。

审计绑定原观察→H40 wire解码→原Gate ACCEPT→未修改的前8点安装→ACK→200个历史2ms tick的8个真实后果。命令issue时钟为0至.35s，后果为.05至.40s；末32点未执行，不得作为后果标签。原Gate最大线速度0.152846526699035m/s、最大角速度0.26771715174309313rad/s、平移路径0.1294936053769747m；原S1为PASS且不修改动作。审计本身引擎启动、模型加载、网络、查询、积分、render、更新及源写入均为0。

八点名义与实际TCP距离均值 **3.072575856mm**、最大 **5.436143830mm**；SO3误差均值0.009405379853rad、最大0.015758105612rad。实际机械夹爪分数在点1/3/5略超[0,1]，保留原值不clip。READY.queries=0是启动快照、RESULT.model_calls=0是Core本地计数，不能据此断言历史Hopper查询不存在；本次也没有读取该Hopper实际draw/noise回执或重新核验全部checkpoint tensor。

这八个后果只能用于**相同checkpoint/query/observation/前8点**的历史链，原SOURCE_CONFIRM角色仍禁止拟合、模型选择或新TEST。保存姿态不重建IK cache与控制历史；链审计通过不构成通用源响应模型、fresh CPU/GPU抽样可复现、目标frame/动作合同或具身资格。AST验证原resident公式导出的query seed为2905439172280596819，但未读取实际noise，CARD的29201 noise-bank不是该query seed，不能声称bit-exact重放。

当前FUSION冻结E1的负例保持原样：manifest `383ad9b8fc3b81efffde132d4f06de92fb96fbdf9ec0b6c92c097e0f58eb2a8d` 的实际H40已被原源Gate REJECT（线速度4.9684424578m/s、角速度15.2314707762rad/s、路径5.6288115677m）。旧CORR后果不能冒充当前E1 SOURCE_OUTCOME，也不能自动绑定到下一次fresh CORR draw。后续条件须按 [冻结CORR协议](../../../docs/protocols/WM02_FUSION_CORR2048_FROZEN_CONDITION_V1.md) 单独封存。

## 实际成本与证据范围

下列CPU统计存在父子/系统归属重叠，**同一行不能求和**；未报告的MemoryPeak记为缺失，不记0。原生面板wall含顺序原生批次，后面三个CPU worker的wall是各自函数/包装层，不是整夜工期。

| 已封存工作 | 函数或manager wall s | 函数或manager CPU s | 包装/子进程 wall / CPU s | systemd CPU s | 内存 |
|---|---:|---:|---|---:|---|
| 192评价+6资格 | 4771.523657746 | 19.820765014 | 子进程CPU14446.233392 | 14494.006491 | 峰值3948982272B |
| 结构响应闭式拟合 | 1.260413483 | 0.100588512 | 1.918495359 / 0.363947 | 0.462200 | 函数峰值RSS46568KiB；unit缺失 |
| 接触时序闭式基线 | 44.004295455 | 29.897155892 | 46.523386669 / 31.004092 | 31.105927 | 函数峰值RSS1124564KiB；unit缺失 |
| 三对目标保存图像 | 5.451753033 | 2.965283702 | 6.127153318 / 3.280373 | 3.363065 | unit缺失 |
| 历史CORR链只读审计 | — | — | 0.302192206 / 0.275594 | 未提供 | 未提供 |

本报告仅从本地已存回执提取与复算数值；没有远端命令、新job、查询或积分，不把fixture测试当原生资格。原native完整journal没有为本次公开报告重新读取；最终审计回执与原封存文件hash共同绑定证据。各小worker的原RESULT文件独立SHA若未由快照提供，则不虚构，以含其完整结果的快照SHA引用。

| 私有回执标识（仅公开名称与hash） | SHA256 | 支持内容 |
|---|---|---|
| FOUR_ARM_NATIVE_FINAL_AUDIT01.private.json | `602a25f981b09ae6b9583a984b7af1d1137a0b3b4a256b990a76adfc4ae84ae1` | 192评价、6资格、最终配对分析及成本 |
| FUSION_SUCCESSORS_READ02.private.json | `8671e94aff3f0cbc3f4e3f396ee5d7920685bcfc4b4557dbd2afe6b43f154cbd` | 结构响应、目标采集终态 |
| CONTACT_TIMING_ACTUAL01.private.json | `e703598edfba92f33632611c6d7eaa4bf060c793db5ed37fbc2f3dcfa2a7b1e5` | 因果接触时序全部结果及比较器 |
| CORR_SOURCE_OUTCOMES_AUDIT01.private.json | `c261bd6ca993f14b541577a57c64e7619ff2860ca42c3b95374eaef9b6a8de14` | 历史query0前缀真实后果链 |
| RIGID_BASE_FRAME_MAP_AUDIT02.private.json | `2696c4b1dbc53251555729271bc1f2c5aea0d7a57c470e39d9bfc8f33b24e18b` | BASE与完整静态安装链frame关系 |
| OFFLINE_BRIDGE_SOURCE_MANIFEST.private.json | `300841422798242ebf197ebfdcf5274b04d261b081bf72da5526a23ee0a6f637` | source、语言provenance与原CARD hash |

ACTUAL JSON保留完整结构模型TRAIN/DEV指标、LOPO各折排除关系、各源SHA、精确恢复误差、时序基线四事件/六时域主要分层读数以及全部12个固定神经比较器的grasp读数；其它父级或close56/64细分由原回执hash绑定。公开文件不含私有主机路径、像素、原始状态、动作payload或凭据。
