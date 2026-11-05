# PILOT2 实际进展：先修采样，再增加物理父

2026-11-06，Asia/Singapore。执行计划固定于 `6d82fa23c892f167001abe743dd5eadfca6fd777`（PR #363，未合并）。本报告是中期封存，**PILOT2 尚未完成**。

已得到的主要结论是：原顺序采样确实妨碍了这组小 Actor 的教师拟合；无需增大网络，打乱采样即可在三个 seed 上显著改善。首批四个新物理 TRAIN 父也已完成真实采集、并驱动新一轮三种子训练。尚未进行新策略原生收益评价，不能把教师一致率当作控制成功。

## A0/A1：匹配数据、采样顺序与 seed

保持原 ARX、任务、两层 128 网络、batch128、AdamW lr0.001、纯软教师 CE 和 TRAIN-only normalizer。12 模型设计为 24/53 TRAIN 历史 × 原 grouped cyclic/global shuffle × seed36001/36002/36003。每历史的256目标是合成教师标签，不是256个物理父。DEV仍为来自两个物理父的8条历史，没有读取旧 TEST。

| 8k 主更新端点 | TRAIN 教师动作一致率，三个 seed | DEV 教师动作一致率，三个 seed |
| --- | --- | --- |
| 24 / grouped cyclic | 83.69%, 64.52%, 87.08% | 91.02%, 62.06%, 87.35% |
| 24 / global shuffle | 99.07%, 98.75%, 99.35% | 95.65%, 96.24%, 97.36% |
| 53 / grouped cyclic | 92.55%, 89.45%, 73.38% | 94.63%, 95.75%, 80.22% |
| 53 / global shuffle | 98.71%, 97.63%, 98.81% | 97.66%, 97.17%, 96.39% |

![三个 seed 的实际学习曲线](A1_MATCHED_CURVES.png)

曲线为全部三个 seed 的中位数和 min–max。右列是 **ARX 模型评分的 regret**，不是本体物理误差。自然 CE 含教师熵，不要求接近零。2k/8k 是等更新主对照；20/40完整标签 epoch 为单列辅助对照：24窗口在960/1920更新，53窗口在2120/4240更新。二者不能混称等 compute。

完整 sampler 保存 permutation、PCG64 generator、epoch、offset、draws，恢复探针同时核对模型、优化器、normalizer、Python/NumPy/Torch/CUDA RNG 和 sampler。12个模型均恢复误差0；原24/53 cyclic seed36001 的2k端点与既有 checkpoint 的模型和优化器完全一致。两个数据版本各自六个模型的 normalizer 数组哈希分别完全相同；不同数据版本仍使用各自 TRAIN 拟合，未声称只隔离了覆盖的单一因果因素。

按事前冻结的 TRAIN 高 margin 错误、2k→8k KL下降和 DEV 非劣化规则，arm0/1/5/8获得有限20k后继。每个从8k完整状态新增12k更新，其余八个未为凑步数继续训练。四个后继恢复误差均0；端点 TRAIN/DEV一致率依次为95.25/95.46%、92.60/96.83%、94.83/97.27%、97.17/97.90%。这些结果支持继续使用 shuffle 基线，不支持无限增加 cyclic 更新。

实际累计：12个不同模型，96,000主更新；4个延续新增48,000；合计144,000 Actor 主更新＋16个单次恢复探针。WM更新0，原生积分0。各 checkpoint、数据和逐父/目标/类别指标见 [匹配曲线数据](A1_MATCHED_CURVES.json) 与 [条件延续及normalizer审计](A1_CONDITIONAL_CONTINUATIONS.json)。

## B0：新物理父和真实动作覆盖

采集前固定18个新父及分区（12 TRAIN、6 DEV），仅改变允许的初始 root/object XY；四个原动作和原时钟不变。面板 SHA256：`8e502a719431b43db807ddec0994fea32080a0b91f41e0123b15011a8731b944`。已完成首批 P2T01–P2T04：

- 4次单场景资格＋16次双槽动作批次（32个动作 episode），36/36原生PASS，147,708个2ms tick，worker退出0且lease CLOSED。
- 每父 HOLD/+X/−X/+Y各两个独立重启，动作顺序和repeat/slot在采集前随机化；没有按教师喜好挑动作。
- 新增32个可用 TRAIN 窗口、768条晚决策50ms转移；原始动作采集记录2048个atom。完整prefix及raw记录不冒充768个独立初态。
- 合并后85 TRAIN/8 DEV，既有DEV保留，TEST未读。Replay SHA256：`89aa357c5bf90746ac4869fe720bdcfc21375061e3c0affa17193319df11ffa0`。
- 三个fresh seed使用同一shuffle配方各8k更新，24,000新增主更新＋3恢复探针，全部退出0/CLOSED、完整恢复误差0。TRAIN一致率98.13/98.57/98.52%，DEV97.02/97.90/98.14%。尚未选择唯一最佳 seed，也未执行新原生策略评价。三种子完整曲线及20/40标签epoch辅助端点见 [85窗口曲线](B0_85_THREE_SEED_CURVES.json)。

![三数据规模shuffle实际曲线](B0_85_SHUFFLE_CURVES.png)

| 动作 | 8个实际episode平均TCP响应 XY，mm | 同动作独立重启平均/最大差，mm |
| --- | --- | --- |
| HOLD | (0.4566, −0.2401) | 0.0544 / 0.1356 |
| +X 1mm | (1.0954, −0.2508) | 0.0625 / 0.1700 |
| −X 1mm | (−0.1408, −0.2862) | 0.0915 / 0.2089 |
| +Y 1mm | (0.5002, 0.2026) | 0.0410 / 0.0761 |

这是独立重启，**没有完整 MPC/WBC 中点恢复，不能称 exact fork 或精确 oracle regret**。HOLD自然运动及最大0.209mm重启差异说明，后续B1必须保留重复、按父分析并预登记实用改善阈值。采集阶段目标只是固定合同字段，任务成功率不作为策略收益证据。逐episode、pair、checkpoint与模型身份见 [首批实际数据](B0_FIRST_BATCH_ACTUAL.json)。

## C0/C1：真实接触与新训练入口

SC07B同一原生进程中，三个不同的有限新任务均已原生PASS（各6703 tick/168 atom，退出0/CLOSED）：KEEP64持续3.6s受支撑夹持；RELEASE56先保持1.2s、末端连续释放2.236s；OPEN72始终打开，夹持目标失败，保留为有效负例。累计20,109 tick/504 raw atom，均为脚本行为，不能算Policy学习或完整lift/carry/place。

随后固定了8 TRAIN＋4 DEV新支持布局（object/support XY各在原布局±0.5mm内同时平移）。每个布局检查229个完整初始化轨迹离线状态，12/12通过静态筛查、积分0；该结果不是动态PASS。新布局先单环境原生检查，首对新TRAIN/DEV的8次执行已全部原生PASS；其余10布局尚未动态验收。

已实现 `build_pilot2_contact` 与 `train_pilot2_contact_wm`：真实历史条件直接预测1/4/8/16/32/64个50ms步的TCP/物体位移及原生接触/双侧grasp/支撑/物理打开事件。未来输入仅由预登记程序生成，过去信息在action-conditioned和no-future-program消融中相同；实际未执行或跨安全回退的预测后缀被mask。按自然、事件转换、稳定区间分别报告Brier/precision/recall和位移误差，不将部分状态头当自由rollout模拟器。完整模型/优化器/RNG/sampler恢复和数据扩大时明确冻结旧TRAIN normalizer的接口已有软件测试；随后已完成两个真实bootstrap训练对照，见下文；非恒定Policy的真实训练和原生闭环仍未完成。

## 源码、命令与验证

- 采样/诊断/矩阵：`8219ef1ba32f5a3408df282462640e5a19ac475a`。
- 可变进度写入修复：`b75cac1b393bc53d3ff9cb53034500de80a123ff`；模型trainer字节未改。
- 支撑接触程序及20k后继：`c01ad06c82a07e8515eb049363c3e71cfdfeed5e`。
- B0真实随机动作、分区安全Replay与85窗口训练：`1506dc223ba8b5851c18c442d4eb5b648bba2b14`。
- 接触多视野builder/trainer：`49ba05dde242ec5dfcfc43afe9b271682fe603c4`。

实际模块包括 `scripts.wm02.train_pilot2_matrix`、`scripts.wm02.train_pilot1_actor`、`scripts.wm02.pilot1_native_worker`、`scripts.wm02.pilot1_parallel_worker`、`scripts.wm02.build_pilot2_coverage`；使用登记的不可变spec与实际Python。完整实际argv/退出码在各私有worker/build回执中保留，不把宿主机绑定信息或私有数据发布进Git。

本地验证命令：`PYTHONPATH=src <model-native-env>/bin/python -m unittest discover -s tests/world_model -q`，218 tests通过，退出0。新增多视野4项测试在`-O`下也通过，含真实训练函数的软件夹具恢复；这些夹具不计科研模型更新。GitHub CI尚未得到执行回执。

## 失败、成本和接续

第一矩阵worker在完成一个有效8k模型后因对可变进度调用exclusive writer而退出1；旧失败和完整模型保留。修复后fresh worker核对结果/spec/checkpoint/旧lease CLOSED及trainer字节，直接复用完整arm，未重训。初次B0传输曾遇SSH握手reset；独立只读核对无源目录/无登记/无dispatch后，才经原入口用已核验不可变父源码＋Git delta传输；没有修改网络或重放不明写入。

A1两矩阵worker合计144.650s wall/134.686s子CPU；四个20k延续worker合计78.004s wall/70.747s子CPU。16个训练阶段CUDA事件累计187.920s，峰值Torch reserved556MiB；该事件计时不等于全进程GPU成本。B0采集coordinator实际892.866s wall（含14.199s Replay build），20个独立qualification/batch worker合计55.374s自身CPU＋2416.569s子CPU，批次子树只计一次；不能用coordinator自己的2.363s CPU冒充全成本。85窗口三种子worker合计43.668s wall/41.350s子CPU。所有大数据、模型和日志均写入已验收的非根本地数据盘。

累计已封存PILOT2学习为168,000 Actor主更新＋19探针、真实新WM更新0；这是多模型累计。尚缺余下B0父、B1带重复DEV及冻结新TEST、C0布局动态验收和C1真实学习闭环。下一步实际入口为新接触试批的原worker及builder，封存后按原owner登记有限神经任务；其它READY批次继续接力。冻结WBC、安全、2ms dt和唯一owner均保持；未构造GPU物理多世界、未使用H200或分布式。


## 冷边界更新：接触新布局采集通过，封存依赖缺陷已修复

首对新 TRAIN/DEV 支持布局已完成2资格＋6脚本采集，8/8原生PASS，48,424 tick；KEEP64和RELEASE56在两布局均达到目标，OPEN64在两布局均为有效负例。接触采集是1008个raw atom，来自两个物理父，不能称1008个独立初态。

原coordinator在后续builder阶段退出1，原因是NumPy数据模块通过模型类导入了Torch，而原生ROS/MuJoCo环境没有Torch。原失败结果和CLOSED lease保持。修复源码 `ca070be5daef79b4837ebe3d1ad8211eac941752` 将过去状态特征和直接Replay与Torch网络分离；219项测试通过，新增子进程测试禁止Torch导入。未往原生环境安装Torch，未改WBC/系统依赖，也未重跑任何物理任务。

以新有限CPU身份重建已有6条episode，退出0、耗时0.900s、积分0；Bundle SHA256 `f3306c3cea0c2175107d944b479977638bf1f11290f7e9452c235926ef6f9228`，954个历史anchor、5010个有真实执行依据的多视野标签，1 TRAIN父＋1 DEV父。该bootstrap数据规模只能检查学习可行性，不代表完成8 TRAIN/4 DEV覆盖或可靠控制收益。下一步是已实现的双消融接触WM有限训练，同时恢复B0下一个四TRAIN父双槽采集；正式C1仍需新增TRAIN反馈、非恒定Policy和冻结原生评价。


## C1首个真实数据训练：已完成可行性检查，保留负结果

两个接触模型在上述6个实际episode上各做2000次主更新（总4000＋2次恢复探针），源码 `ca070be5daef79b4837ebe3d1ad8211eac941752`，worker退出0、lease CLOSED；完整恢复/端点重载误差均0。TRAIN和DEV各477历史anchor，均只来自1个物理父。GPU事件8.124s，worker wall10.939s/子CPU9.482s，Torch峰值保留90MiB。checkpoint SHA256：

- action-conditioned：`7b81e7bb68eca4039c5ba5924df3b8f8c50786dda9821fa1d26611ab24e3ae82`。
- no-future-program：`8852e635f13987490a8b9eb49fb3b08853447b4e2f90ccedad5620b7cce2e6ce`。

DEV 32步（1.6s）自然分布：grasp Brier分别0.06197 / 0.43165 / 0.25（动作条件/动作盲/持恒）；打开且支撑Brier为0.02760 / 0.48305 / 0.25。转换区间grasp为0.17063 / 0.34992 / 0.75591。存在有限的长视野命令预测正证据，但不能只挑这些结果：一步grasp Brier动作条件0.03919，输给持恒0.00629；恒定真实支撑的持恒Brier为0，神经头较差。动作盲模型DEV泛化很差，也可能受单父覆盖/优化/归一化影响，不能将其全部归因于动作因果性。

完整自然/稳定/转换分层、位移和四事件混淆计数见 [C1实际bootstrap](C1_ACTUAL_BOOTSTRAP.json)。这次两臂共享 action-conditioned TRAIN normalizer；未来已实现的受控分支可各用自身输入的TRAIN-only normalizer，保持历史部分一致，并导出经过NumPy/Torch核对的部分预测头。该后继尚未执行，旧结果不替换。

PILOT2当前已封存47次原生PASS、216241 tick、168000 Actor主更新＋19探针、4000 WM主更新＋2探针；B0第二批正在执行，其进行中尝试另列，不预计成功。C1的非恒定Policy、真实反馈更新和冻结评价尚未完成，本goal仍ACTIVE。


## 最新冷边界：117 TRAIN历史，归一化对照和待验收Policy

B0第二批P2T05–P2T08再次完成4资格＋32动作采集，36/36原生PASS，147708 tick，新增768条可用转移；869.137s coordinator wall，Replay build14.251s。两批累计新增8个物理TRAIN父、64个真实动作episode、1536条可用转移；总117 TRAIN/原8 DEV。新Replay SHA256 `9ce72009c7bf3505e196627c6ee170941d823cbe43fc3c30455cbb826b6f0791`。三种子各8k更新/完整恢复0，DEV教师一致率98.68/98.44/98.54%；这仍不是新物理DEV任务收益。三checkpoint、完整身份和成本见 [第二批](B0_SECOND_BATCH_ACTUAL.json)。

第二接触训练仅改变normalizer方案（同seed363101、同6个episode、每臂2k），新源码 `a02762212f71476d63b6c5afe52aaf60e76efa4c`。action-conditioned模型/优化器与首次对照最大差0，属于重复控制，不是新独立seed。动作盲臂改为自身TRAIN输入拟合normalizer后，DEV32步grasp Brier为0.50768，未改善原0.43165；不能把泛化问题简单归咎于这个归一化选择。两臂完整恢复0；NumPy/Torch预测最大差3.27e-6以内，便于后续原生环境不加载Torch。实际额外4000 WM更新＋2探针和成本均计入，见 [归一化控制](C1_NORMALIZER_CONTROL.json)。

已实现共同close56前缀的目标条件Policy训练CLI、三类教师接口（原目标FSM、真实父均值return、部分WM事件代理），以及5303原生tick暂停边界的NumPy决策。每个新场景必须先有KEEP56、RELEASE56两项真实成功资格，已持续500个原生tick支持夹持，且模型真实训练/恢复/哈希通过。失败prefix沿原开夹爪路径记录为ineligible；不伪造策略选择成功。新策略反馈数据只在实际选择之后才把未来program供给WM，禁止把自适应未来命令泄漏到早期历史。227项测试及新增13项-O测试通过；**真实Policy模型更新和原生Policy评价当前仍为0**。

当前单环境有限timing后继正在补齐原面板中的KEEP56/KEEP72/RELEASE64/RELEASE72，每布局先单环境；已完成资格和前三程序不重跑。完成后才推进真实Policy和新TRAIN反馈。B0仍缺最后4 TRAIN父、6新DEV父和B1重复/冻结TEST；C0仍缺更多动态父覆盖。

当前已封存累计83次原生PASS/363949 tick、192000 Actor主更新＋22探针、8000 WM主更新＋4探针。进行中的timing任务另列，未预记PASS。原生执行器C08 SHA256 `29b261541b9dc2b2ed5c6df065026b33a11c3019181b9a8cfa980a7878d91df2`；原ARX、任务和冻结WBC保持。PILOT2 goal继续ACTIVE。


## 实际双槽接触与首次训练策略反哺

源码 `1063aa1954d1ea09156925db409d352f3d48e826` 新增严格的双槽支持Policy合同：每子任务先有匹配scene/goal/method/model的真实单槽资格，原owner的最大2实例不变。首次commission只允许已通过单槽的两项TRAIN规则目标；生产必须再有6703 tick的实际双槽完整时长证明。不以任务成功筛除安全但失败的策略结果。GPU物理后端/4环境/分布式均未启动。

首次启动因父进程CPU绑定8–11与登记0–7不匹配而退出1，0物理积分；两个实际未启动子lease经独立核对后关闭，未复活CLOSED父任务。修正运行参数后以新身份双槽执行：两子各6703 tick、168 raw atom，分别KEEP/RELEASE且目标均成功；worker退出0、父子lease CLOSED，wall84.807s，自身CPU0.600s＋子CPU297.735s，一个物理owner。后续源码 `c4d7a486d9828f43739f8bcd5ecdb39de49462d3` 已增加“尚无启动marker且无native输出”时的严格未启动lease收尾；231项测试通过。两次私有bootstrap路径守卫错误也保留为前准入失败，未产生登记或物理效果。

真实Policy已有两个各2k的训练条件：规则教师与WM事件代理教师。它们TRAIN4标签均100%，DEV4标签均75%；WM教师尚未改善策略。规则教师Policy checkpoint `4caea9fe3643555e9ce1b172a4b65fa9d88dd22c2700957438ad03443a877d09`，NumPy `34eb468b587a09beefb9f66fa2008442766b41de614cf11f86642fb8bc1f5c84`；WM教师Policy checkpoint `6ffd494b2bb1c2d0edf23400dd199dda872e5d5ea3ad35a3937af9f1b6e62abf`。后者尚未原生评价。

规则教师Policy已在TRAIN同一物理父上完成KEEP/RELEASE各一次单槽执行，又完成一次双槽独立重启：四个真实rollout均原生PASS且微任务成功。每次于5303 tick暂停边界用过去状态＋已知目标推理并封存选择后才安装动作；KEEP总支持保持4.0s（含决策前1.2s、决策后2.8s），RELEASE要求先实际支持夹持再打开。双槽训练策略采样wall85.121s，自身＋子CPU298.235s。它们不是四个新父，也不是完整控制状态fork或独立TEST。逐模型与原生事实见 [C1模型/首次原生](C1_MODELS_AND_FIRST_NATIVE.json)。

四个真实Policy TRAIN rollout已加入旧14 episode，封存18-episode Bundle：`abed558af5d0e9a22e5493540b9c30bfb5dc807dbd4d34f0feba129396c646ab`。3024 raw atom、2450历史anchor、12586个有效多视野标签；自适应未来程序仅在实际选择之后作为WM输入。新增4000 WM主更新＋2恢复探针，以及Policy新增2000主更新＋1探针均完成，所有worker退出0/CLOSED、恢复误差0。

此次是真实“训练→原生策略采样→新TRAIN→更新”，但尚无泛化正结论：更新Policy TRAIN12个标签仍100%，原DEV4个标签由75%降至50%。新checkpoint `4258c4a6001164f13bc983910ed8341114df2586e3b88fd9634edd3202bf31f6`，须冻结后做原生DEV验证，不能用原训练父的4/4成功替代。接触数据仍为1 TRAIN父＋1 DEV父，下一优先级为独立父覆盖及输入泛化审查，继续叠加同父更新没有证据支持。具体更新、负结果与成本见 [真实策略反哺](C1_REAL_POLICY_FEEDBACK_UPDATE.json)。

当前已封存99次原生PASS、471197 tick；自由空间Actor192000主更新/22探针，接触Policy6000主更新/3探针，接触WM16000主更新/8探针。合计Actor/Policy198000主更新/25探针，均为跨模型累计。B0第三批P2T09–P2T12正在实际执行，未预记其32条计划窗口。新DEV/TEST控制收益、完整8 TRAIN＋4 DEV接触覆盖仍未完成，goal保持ACTIVE。
